#!/usr/bin/env python3
"""Live native full-history evaluation for Perception Gain V1 checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from scripts.perception_gain_evaluation import (
    ARTIFACT_ROOT,
    CAL_UPDATES,
    DATA_CONTRACT,
    DEFAULT_ASSETS,
    DEFAULT_EXTERNAL_ROOT,
    DEFAULT_ROLES,
    HORIZONS,
    PROJECT_ROOT,
    VARIANTS,
    PerceptionEvaluationError,
    _atomic_json,
    _external_reference,
    _load_evaluation_weights,
    _load_json,
    _sha256,
    build_additional_population_slices,
    build_role_base_dataset,
    select_live_population_units,
    summarize_additional_population_slices,
)


def build_native_prefix_key(
    *,
    reference_id: str,
    sequence_id: str,
    context_index: int,
    scan_ids: Sequence[str],
    scan_indices: Sequence[int],
    horizon: int,
    order_id: str = "canonical",
) -> dict[str, object]:
    from scripts.system_comparison_inference import validate_full_history_cache_key

    if (
        not isinstance(reference_id, str)
        or not reference_id
        or not isinstance(sequence_id, str)
        or not sequence_id
        or isinstance(context_index, bool)
        or not isinstance(context_index, int)
        or context_index < 0
        or order_id not in {"canonical", "reverse", "sha256_seed45"}
        or horizon not in HORIZONS
        or isinstance(scan_ids, (str, bytes))
        or isinstance(scan_indices, (str, bytes))
        or not 2 <= len(scan_ids) <= 5
        or len(scan_indices) != len(scan_ids)
        or horizon > len(scan_ids)
        or sequence_id != "-".join(scan_ids)
    ):
        raise PerceptionEvaluationError("native prefix identity is invalid")
    try:
        return validate_full_history_cache_key(
            {
                "master_sequence_id": sequence_id,
                "reference_scene_id": reference_id,
                "order_id": order_id,
                "context_index": context_index,
                "context_scan_indices": list(scan_indices),
                "horizon": horizon,
                "history_scan_ids": list(scan_ids[:horizon]),
                "scan_indices": list(scan_indices[:horizon]),
                "task_quality": True,
            }
        )
    except (TypeError, ValueError) as error:
        raise PerceptionEvaluationError(
            f"native prefix contract is invalid: {error}"
        ) from error


def _metric_accumulator(dataset_spec: Path):
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.system_comparison_metrics import CausalTaskAccumulator

    return CausalTaskAccumulator(
        metric_factory=lambda mode: OfficialMetricAccumulator(
            mode=mode,
            dataset_spec=str(dataset_spec),
            min_region_size=100,
        )
    )


def _metric_row(
    *,
    variant: str,
    optimizer_update: int,
    role: str,
    horizon: int,
    reference: str,
    logical_unit_count: int,
    reference_count: int,
    values: Mapping[str, float],
) -> dict[str, object]:
    return {
        "method": f"FH-{variant}-native",
        "variant": variant,
        "optimizer_update": optimizer_update,
        "data_role": role,
        "reference": reference,
        "reference_count": reference_count,
        "logical_unit_count": logical_unit_count,
        "T": horizon,
        "t_mAP": values["causal_prefix_t_mAP"],
        "t_mAP50": values["causal_prefix_t_mAP50"],
        "t_mAP25": values["causal_prefix_t_mAP25"],
        "t_REC": values["causal_prefix_t_REC"],
        "current_stage_AP": values["current_stage_AP"],
        "producer_id": "native-full-history-live",
        "output_policy": "native",
        "status": "MEASURED",
    }


def run_native_checkpoint_evaluation(
    *,
    variant: str,
    optimizer_update: int,
    role: str,
    checkpoint: Path | None,
    scorer_checkpoint: Path | None,
    horizons: Sequence[int] = HORIZONS,
    unit_limit: int | None = None,
    assets_path: Path = DEFAULT_ASSETS,
    roles_path: Path = DEFAULT_ROLES,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    output_path: Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    import hydra
    import torch
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.perception_gain_foundation import _resolve_cache_assets
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        PROTOCOL_B_POPULATION_ID,
        _episode_specs,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import (
        FullHistoryPredictionProducer,
        deterministic_inference_runtime,
    )
    from scripts.system_comparison_metrics import validate_causal_prefix_pair
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    requested_horizons = tuple(int(value) for value in horizons)
    if (
        variant not in VARIANTS
        or role not in {"CAL", "SEL", "PB", "ADDITIONAL"}
        or optimizer_update not in CAL_UPDATES
        or not requested_horizons
        or len(set(requested_horizons)) != len(requested_horizons)
        or any(value not in HORIZONS for value in requested_horizons)
        or (role == "ADDITIONAL" and requested_horizons != (2, 3, 4))
        or (
            unit_limit is not None
            and (
                isinstance(unit_limit, bool)
                or not isinstance(unit_limit, int)
                or unit_limit <= 0
            )
        )
    ):
        raise PerceptionEvaluationError("native evaluation request is invalid")

    assets = _resolve_cache_assets(assets_path)
    roles_payload = _load_json(roles_path).get("roles")
    references = roles_payload.get(role) if isinstance(roles_payload, Mapping) else None
    if (
        isinstance(references, (str, bytes))
        or not isinstance(references, Sequence)
        or not references
    ):
        raise PerceptionEvaluationError(f"{role} role is unavailable")
    device = _validate_cuda_device(device_name)
    run_dir = (
        external_root
        / "evaluation"
        / role.lower()
        / f"FH-{variant}-native"
        / f"update={optimizer_update:04d}"
    )
    config = compose_variant_config(
        variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
    )
    config.model.return_query_features = True
    if variant == "Q-SEM" and scorer_checkpoint is not None:
        config.perception_training.scorer_checkpoint = str(scorer_checkpoint)
    data_contract = _load_json(DATA_CONTRACT)
    if role == "ADDITIONAL":
        additional_slices = build_additional_population_slices(
            config=config,
            data_root=Path(assets["data_root"]),
            metadata_path=Path(assets["rio_metadata"]),
            data_contract=data_contract,
            role_references=references,
        )
        population_identity = summarize_additional_population_slices(additional_slices)
        population_manifest_sha256 = str(
            population_identity["population_manifest_sha256"]
        )
        population_slices = tuple(
            (
                item.base_dataset,
                item.episode_specs,
                item.units[:unit_limit] if unit_limit is not None else item.units,
                (item.horizon,),
            )
            for item in additional_slices
        )
    else:
        base_dataset = build_role_base_dataset(
            config=config,
            data_root=Path(assets["data_root"]),
            role=role,
            horizon=5,
        )
        population_id = (
            PROTOCOL_B_POPULATION_ID if role == "PB" else DEVELOPMENT_POPULATION_ID
        )
        base_dataset, masters, population_manifest_sha256 = build_baseline_population(
            base_dataset,
            data_contract=data_contract,
            metadata_path=Path(assets["rio_metadata"]),
            population_id=population_id,
            protocol_b_manifest_path=(
                Path(__file__).resolve().parents[1]
                / "artifacts/P6A/protocol_b_manifest.json"
            ),
        )
        episode_specs = _episode_specs(masters)
        units = select_live_population_units(
            role=role,
            role_references=references,
            episode_specs=episode_specs,
            expected_logical_units={"CAL": 23, "SEL": 24, "PB": 129}[role],
        )
        if unit_limit is not None:
            units = units[:unit_limit]
        population_slices = ((base_dataset, episode_specs, units, requested_horizons),)

    system = PerceptionGainTrainer(config)
    load_audit, checkpoint_sha, checkpoint_reference, new_parameters, weight_sources = (
        _load_evaluation_weights(
            system=system,
            variant=variant,
            optimizer_update=optimizer_update,
            r1_checkpoint=Path(assets["r1_checkpoint"]),
            checkpoint=checkpoint,
            scorer_checkpoint=scorer_checkpoint,
        )
    )
    if checkpoint is not None:
        checkpoint_reference = _external_reference(
            checkpoint, external_root=external_root
        )
    system.to(device).eval().requires_grad_(False)
    collator = hydra.utils.instantiate(config.data.validation_collation)
    config_sha = canonical_json_sha256(OmegaConf.to_container(config, resolve=True))
    protocol_path = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
    p6a_path = PROJECT_ROOT / "conf/p6a/default.yaml"
    p6a = yaml.safe_load(p6a_path.read_text(encoding="utf-8"))
    settings = p6a["baselines"]["b4"]
    provenance = {
        "source_commit": "6ef77620aa20926311eff3124a794a6ca2e32727",
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": config_sha,
        "protocol_sha256": _sha256(protocol_path),
    }
    producer_slices = tuple(
        (
            FullHistoryPredictionProducer(
                dataset=slice_dataset,
                collate=collator,
                system=system,
                device=device,
                provenance=provenance,
                class_mapper=build_rio_class_mapper(slice_dataset),
                move_data=_move_data_to_device,
                move_targets=_move_targets_to_device,
                background_class=int(settings["background_class"]),
                confidence_threshold=float(settings["confidence_threshold"]),
                mask_threshold=float(settings["mask_threshold"]),
                minimum_mask_support=int(settings["minimum_mask_support"]),
                seed=int(p6a["protocol_b"]["seed"]),
            ),
            slice_specs,
            slice_units,
            slice_horizons,
        )
        for slice_dataset, slice_specs, slice_units, slice_horizons in population_slices
    )
    dataset_spec = Path(assets["metric_dataset_spec"])
    pooled = {
        horizon: _metric_accumulator(dataset_spec) for horizon in requested_horizons
    }
    by_reference: dict[tuple[int, str], object] = {}
    completed = {horizon: [] for horizon in requested_horizons}
    completed_references = {horizon: set() for horizon in requested_horizons}
    incomplete = []
    started = time.perf_counter()
    total_units = sum(len(item[2]) for item in producer_slices)
    unit_index = 0
    units_by_horizon = {horizon: [] for horizon in requested_horizons}
    with deterministic_inference_runtime(45, device):
        for producer, episode_specs, units, slice_horizons in producer_slices:
            for unit in units:
                unit_index += 1
                spec = episode_specs[unit.spec_index]
                for horizon in slice_horizons:
                    units_by_horizon[horizon].append(unit)
                    key = build_native_prefix_key(
                        reference_id=unit.reference_id,
                        sequence_id=unit.sequence_id,
                        context_index=spec.context_index,
                        scan_ids=spec.scan_ids,
                        scan_indices=spec.scan_indices,
                        horizon=horizon,
                        order_id=(
                            ("canonical", "reverse", "sha256_seed45")[
                                spec.context_index % 3
                            ]
                            if role == "PB"
                            else "canonical"
                        ),
                    )
                    try:
                        produced = producer.produce_bundle(key)
                        pair = validate_causal_prefix_pair(
                            prediction=produced.processed.task_prediction,
                            target=produced.processed.target,
                            horizon=horizon,
                            observed_scan_ids=spec.scan_ids[:horizon],
                        )
                        pooled[horizon].update(pair)
                        reference_key = (horizon, unit.reference_id)
                        if reference_key not in by_reference:
                            by_reference[reference_key] = _metric_accumulator(
                                dataset_spec
                            )
                        by_reference[reference_key].update(pair)
                    except (
                        OSError,
                        RuntimeError,
                        ValueError,
                        KeyError,
                        IndexError,
                    ) as error:
                        incomplete.append(
                            {
                                "logical_unit_id": unit.logical_unit_id,
                                "reference_id": unit.reference_id,
                                "sequence_id": unit.sequence_id,
                                "T": horizon,
                                "error_type": type(error).__name__,
                                "reason": str(error),
                            }
                        )
                        torch.cuda.empty_cache()
                        continue
                    completed[horizon].append(unit.logical_unit_id)
                    completed_references[horizon].add(unit.reference_id)
                    print(
                        json.dumps(
                            {
                                "completed_unit": unit_index,
                                "T": horizon,
                                "role": role,
                                "total_units": total_units,
                                "update": optimizer_update,
                                "variant": variant,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    del produced, pair
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    rows = []
    for horizon in requested_horizons:
        if completed[horizon]:
            rows.append(
                _metric_row(
                    variant=variant,
                    optimizer_update=optimizer_update,
                    role=role,
                    horizon=horizon,
                    reference="all",
                    logical_unit_count=len(completed[horizon]),
                    reference_count=len(completed_references[horizon]),
                    values=pooled[horizon].compute(),
                )
            )
        for reference in sorted(completed_references[horizon]):
            accumulator = by_reference[(horizon, reference)]
            rows.append(
                _metric_row(
                    variant=variant,
                    optimizer_update=optimizer_update,
                    role=role,
                    horizon=horizon,
                    reference=reference,
                    logical_unit_count=len(
                        [
                            unit
                            for unit in units_by_horizon[horizon]
                            if unit.reference_id == reference
                            and unit.logical_unit_id in completed[horizon]
                        ]
                    ),
                    reference_count=1,
                    values=accumulator.compute(),
                )
            )
    expected_prefixes = sum(len(values) for values in units_by_horizon.values())
    expected_units = sum(len(item[2]) for item in population_slices)
    completed_prefixes = sum(len(values) for values in completed.values())
    status = (
        "PASS"
        if completed_prefixes == expected_prefixes and not incomplete
        else "PARTIAL"
    )
    summary = {
        "schema_version": "perception-gain-native-evaluation-v1",
        "status": status,
        "variant": variant,
        "optimizer_update": optimizer_update,
        "data_role": role,
        "method_id": f"FH-{variant}-native",
        "output_policy": "native",
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha,
        "weight_sources": weight_sources,
        "resolved_config_sha256": config_sha,
        "p6a_config_sha256": _sha256(p6a_path),
        "protocol_sha256": _sha256(protocol_path),
        "population_manifest_sha256": population_manifest_sha256,
        "roles_sha256": _sha256(roles_path),
        "horizons": list(requested_horizons),
        "expected_logical_unit_count": expected_units,
        "expected_prefix_count": expected_prefixes,
        "completed_prefix_count": completed_prefixes,
        "population_by_horizon": {
            str(horizon): {
                "reference_count": len(
                    {unit.reference_id for unit in units_by_horizon[horizon]}
                ),
                "logical_unit_count": len(units_by_horizon[horizon]),
            }
            for horizon in requested_horizons
        },
        "completed_units_by_horizon": {
            str(horizon): values for horizon, values in completed.items()
        },
        "incomplete_units": incomplete,
        "metric_rows": rows,
        "distinct_physical_forward_count": completed_prefixes,
        "new_parameter_count": new_parameters,
        "load_audit": load_audit,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "source_sha256": _sha256(Path(__file__)),
    }
    if output_path is None:
        output_path = (
            ARTIFACT_ROOT
            / "foundation/native_fh"
            / role.lower()
            / variant
            / f"update={optimizer_update:04d}.json"
        )
    _atomic_json(output_path, summary)
    del system
    torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument(
        "--role", choices=("CAL", "SEL", "PB", "ADDITIONAL"), default="CAL"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    parser.add_argument("--unit-limit", type=int)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_native_checkpoint_evaluation(
        variant=arguments.variant,
        optimizer_update=arguments.update,
        role=arguments.role,
        checkpoint=arguments.checkpoint,
        scorer_checkpoint=arguments.scorer_checkpoint,
        horizons=arguments.horizons,
        unit_limit=arguments.unit_limit,
        assets_path=arguments.assets,
        roles_path=arguments.roles,
        external_root=arguments.external_root,
        output_path=arguments.output,
        device_name=arguments.device,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_native_prefix_key",
    "main",
    "run_native_checkpoint_evaluation",
]
