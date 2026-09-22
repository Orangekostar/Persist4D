#!/usr/bin/env python3
"""Evaluate the causal mask refiner against its frozen D0 parent."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_ASSETS = Path("/home/ww/persist4d_runs/perception_gain_v1/assets.local.json")
DEFAULT_ROLES = ARTIFACT_ROOT / "DATA_ROLES.json"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
REFINER_UPDATES = frozenset((0, 500, 1000, 1500))


class RefinerEvaluationError(RuntimeError):
    """Raised when refiner evaluation differs from its frozen contract."""


def scored_horizons_for_episode(role: str, episode_horizon: int) -> tuple[int, ...]:
    if role == "ADDITIONAL" and episode_horizon in {2, 3, 4}:
        return (episode_horizon,)
    if role in {"CAL", "SEL", "PB"} and episode_horizon == 5:
        return (2, 3, 4, 5)
    raise RefinerEvaluationError("refiner episode horizon differs")


def validate_mask_only_refiner_output(
    parent: Mapping[str, object], refined: Mapping[str, object]
) -> dict[str, int]:
    import torch

    fields = {"pred_masks", "pred_scores", "pred_classes"}
    if set(parent) != fields or set(refined) != fields:
        raise RefinerEvaluationError("parent/refiner prediction fields differ")
    parent_masks = parent["pred_masks"]
    refined_masks = refined["pred_masks"]
    parent_scores = parent["pred_scores"]
    refined_scores = refined["pred_scores"]
    parent_classes = parent["pred_classes"]
    refined_classes = refined["pred_classes"]
    if (
        not isinstance(parent_masks, torch.Tensor)
        or not isinstance(refined_masks, torch.Tensor)
        or parent_masks.dtype != torch.bool
        or refined_masks.dtype != torch.bool
        or parent_masks.ndim != 2
        or refined_masks.shape != parent_masks.shape
        or not isinstance(parent_scores, torch.Tensor)
        or not isinstance(refined_scores, torch.Tensor)
        or not isinstance(parent_classes, torch.Tensor)
        or not isinstance(refined_classes, torch.Tensor)
        or parent_scores.ndim != 1
        or parent_classes.ndim != 1
        or parent_scores.shape != parent_classes.shape
        or parent_masks.shape[1] != parent_scores.numel()
    ):
        raise RefinerEvaluationError("parent/refiner prediction tensors do not align")
    if not torch.equal(parent_scores, refined_scores) or not torch.equal(
        parent_classes, refined_classes
    ):
        raise RefinerEvaluationError("refiner changed scores or classes")
    changed = parent_masks != refined_masks
    return {
        "candidate_count": int(parent_scores.numel()),
        "changed_candidate_count": int(changed.any(dim=0).sum().item()),
        "changed_point_count": int(changed.sum().item()),
    }


def append_live_refiner_target(
    targets: Sequence[Mapping[str, object]],
    raw_payload: Mapping[str, object],
    *,
    stage_index: int,
) -> tuple[dict[str, object], ...]:
    import torch

    if stage_index != len(targets) or not 0 <= stage_index < 5:
        raise RefinerEvaluationError("refiner targets differ from causal order")
    target = raw_payload.get("target")
    if not isinstance(target, Mapping):
        raise RefinerEvaluationError("live refiner target is unavailable")
    cloned = {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in target.items()
    }
    return (*targets, cloned)


def _load_refiner(
    checkpoint: Path,
    *,
    optimizer_update: int,
    device: object,
    expected_binding: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object]]:
    import torch

    from models.perception_gain import CausalMaskRefiner

    if optimizer_update not in REFINER_UPDATES or not checkpoint.is_file():
        raise RefinerEvaluationError("refiner checkpoint request is invalid")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"refiner_state_dict", "schema_version", "source_shards", "updates"}
    binding = None
    if (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == "perception-refiner-frozen-v2"
    ):
        from scripts.train_perception_refiner import (
            V2_BINDING_FIELDS,
            validate_refiner_v2_binding,
        )

        required |= V2_BINDING_FIELDS
        binding = {key: payload[key] for key in V2_BINDING_FIELDS if key in payload}
        try:
            validate_refiner_v2_binding(binding)
        except (RuntimeError, ValueError) as error:
            raise RefinerEvaluationError(
                "refiner checkpoint binding differs"
            ) from error
    if expected_binding is not None and binding != dict(expected_binding):
        raise RefinerEvaluationError(
            "refiner checkpoint binding differs from parent/mode/source"
        )
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload["schema_version"]
        not in {"perception-refiner-frozen-v1", "perception-refiner-frozen-v2"}
        or payload["updates"] != optimizer_update
        or not isinstance(payload["refiner_state_dict"], Mapping)
    ):
        raise RefinerEvaluationError("refiner checkpoint contract differs")
    refiner = CausalMaskRefiner(
        input_mode=str(binding["input_mode"]) if binding is not None else "OLD_NEW"
    )
    try:
        incompatible = refiner.load_state_dict(
            payload["refiner_state_dict"], strict=True
        )
    except RuntimeError as error:
        raise RefinerEvaluationError(
            f"refiner checkpoint tensors differ: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RefinerEvaluationError("refiner checkpoint load was not strict")
    refiner.to(device).eval().requires_grad_(False)
    return refiner, {
        "bytes": checkpoint.stat().st_size,
        "checkpoint": str(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in refiner.parameters()),
        "sha256": _sha256(checkpoint),
        "updates": optimizer_update,
        **({"binding": binding} if binding is not None else {}),
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_row(
    *,
    method: str,
    horizon: int,
    reference: str,
    reference_count: int,
    logical_unit_count: int,
    accumulator: object,
) -> dict[str, object]:
    values = accumulator.compute()
    return {
        "method": method,
        "reference": reference,
        "reference_count": reference_count,
        "logical_unit_count": logical_unit_count,
        "T": horizon,
        "t_mAP": values["causal_prefix_t_mAP"],
        "t_mAP50": values["causal_prefix_t_mAP50"],
        "t_mAP25": values["causal_prefix_t_mAP25"],
        "t_REC": values["causal_prefix_t_REC"],
        "current_stage_AP": values["current_stage_AP"],
    }


def run_refiner_evaluation(
    *,
    parent_variant: str,
    parent_update: int,
    parent_checkpoint: Path | None,
    scorer_checkpoint: Path | None,
    refiner_update: int,
    refiner_checkpoint: Path,
    role: str,
    assets_path: Path = DEFAULT_ASSETS,
    roles_path: Path = DEFAULT_ROLES,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    output_path: Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    import hydra
    import torch
    import yaml
    from omegaconf import OmegaConf

    from datasets.task_memory_episode import (
        TaskMemoryEpisodeCollator,
        TaskMemoryEpisodeDataset,
    )
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _latest_full_resolution_masks,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        cache_payload_from_inference,
    )
    from scripts.perception_gain_evaluation import (
        CAL_UPDATES,
        DATA_CONTRACT,
        VARIANTS,
        _atomic_json,
        _external_reference,
        _load_evaluation_weights,
        _load_json,
        build_additional_population_slices,
        build_live_cache_key,
        build_live_provenance,
        build_role_base_dataset,
        select_live_population_units,
        summarize_additional_population_slices,
    )
    from scripts.perception_gain_foundation import (
        _metric_class_mapping,
        _resolve_cache_assets,
    )
    from scripts.prepare_perception_refiner import advance_d0_identity
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_task_memory_controls import (
        _validate_collated_stage_identity,
        _window_observation,
    )
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        PROTOCOL_B_POPULATION_ID,
        _episode_specs,
        _target_for_prefix,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.system_comparison_metrics import (
        CausalTaskAccumulator,
        validate_causal_prefix_pair,
    )
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.task_memory_output import LagOnePublisher
    from scripts.train_perception_gain import compose_variant_config
    from scripts.train_perception_refiner import CausalRefinerRevisionTransform
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    if (
        parent_variant not in VARIANTS
        or parent_update not in CAL_UPDATES
        or refiner_update not in REFINER_UPDATES
        or role not in {"CAL", "SEL", "PB", "ADDITIONAL"}
    ):
        raise RefinerEvaluationError("refiner evaluation request is invalid")
    assets = _resolve_cache_assets(assets_path)
    roles = _load_json(roles_path).get("roles")
    references = roles.get(role) if isinstance(roles, Mapping) else None
    if (
        isinstance(references, (str, bytes))
        or not isinstance(references, Sequence)
        or not references
    ):
        raise RefinerEvaluationError(f"{role} references are unavailable")
    device = _validate_cuda_device(device_name)
    run_dir = (
        external_root
        / "evaluation"
        / role.lower()
        / "R-REFINE"
        / f"parent={parent_variant}-{parent_update:04d}"
        / f"update={refiner_update:04d}"
    )
    config = compose_variant_config(
        parent_variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
    )
    config.model.return_query_features = True
    if parent_variant == "Q-SEM" and scorer_checkpoint is not None:
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
                item.units,
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
            protocol_b_manifest_path=PROJECT_ROOT
            / "artifacts/P6A/protocol_b_manifest.json",
        )
        episode_specs = _episode_specs(masters)
        units = select_live_population_units(
            role=role,
            role_references=references,
            episode_specs=episode_specs,
            expected_logical_units={"CAL": 23, "SEL": 24, "PB": 129}[role],
        )
        population_slices = ((base_dataset, episode_specs, units, (2, 3, 4, 5)),)

    system = PerceptionGainTrainer(config)
    parent_load, parent_sha, parent_reference, _, parent_sources = (
        _load_evaluation_weights(
            system=system,
            variant=parent_variant,
            optimizer_update=parent_update,
            r1_checkpoint=Path(assets["r1_checkpoint"]),
            checkpoint=parent_checkpoint,
            scorer_checkpoint=scorer_checkpoint,
        )
    )
    if parent_checkpoint is not None:
        parent_reference = _external_reference(
            parent_checkpoint, external_root=external_root
        )
    system.to(device).eval().requires_grad_(False)
    refiner, refiner_identity = _load_refiner(
        refiner_checkpoint, optimizer_update=refiner_update, device=device
    )
    refiner_identity["checkpoint"] = _external_reference(
        refiner_checkpoint, external_root=external_root
    )
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(config.data.validation_collation)
    )
    metric_class_mapping = _metric_class_mapping(Path(assets["metric_dataset_spec"]))
    mapped_population_slices = []
    for base_dataset, episode_specs, units, scored_horizons in population_slices:
        class_mapper = build_rio_class_mapper(base_dataset)
        if tuple(class_mapper(index) for index in range(18)) != metric_class_mapping:
            raise RefinerEvaluationError("evaluation class mappings differ")
        mapped_population_slices.append(
            (base_dataset, episode_specs, units, scored_horizons, class_mapper)
        )
    p6a_path = PROJECT_ROOT / "conf/p6a/default.yaml"
    p6a = yaml.safe_load(p6a_path.read_text(encoding="utf-8"))
    settings = p6a["baselines"]["b4"]
    observation_settings = {
        "background_class": int(settings["background_class"]),
        "confidence_threshold": float(settings["confidence_threshold"]),
        "mask_threshold": float(settings["mask_threshold"]),
        "minimum_mask_support": int(settings["minimum_mask_support"]),
    }
    config_sha = canonical_json_sha256(OmegaConf.to_container(config, resolve=True))
    provenance = build_live_provenance(
        source_commit="6ef77620aa20926311eff3124a794a6ca2e32727",
        checkpoint_sha256=parent_sha,
        config_sha256=config_sha,
        episodes=tuple(
            {
                "reference_id": episode_specs[unit.spec_index].reference_id,
                "sequence_id": episode_specs[unit.spec_index].source_sequence_id,
                "scan_ids": list(episode_specs[unit.spec_index].scan_ids),
            }
            for _, episode_specs, units, _, _ in mapped_population_slices
            for unit in units
        ),
    )

    def metric_factory(mode: str) -> object:
        from scripts.p6a_metrics import OfficialMetricAccumulator

        return OfficialMetricAccumulator(
            mode=mode,
            dataset_spec=str(Path(assets["metric_dataset_spec"])),
            min_region_size=100,
        )

    accumulators: dict[tuple[str, int, str], object] = {}

    def accumulator(method: str, horizon: int, reference: str) -> object:
        key = (method, horizon, reference)
        if key not in accumulators:
            accumulators[key] = CausalTaskAccumulator(metric_factory=metric_factory)
        return accumulators[key]

    completed_units = []
    completed_references = set()
    completed_units_by_horizon: defaultdict[int, list[str]] = defaultdict(list)
    completed_references_by_horizon: defaultdict[int, set[str]] = defaultdict(set)
    incomplete_units = []
    reference_unit_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    change_totals: defaultdict[str, int] = defaultdict(int)
    transform_totals: defaultdict[str, int] = defaultdict(int)
    started = time.perf_counter()
    total_units = sum(len(item[2]) for item in mapped_population_slices)
    ordinal = 0
    with deterministic_inference_runtime(45, device):
        for (
            base_dataset,
            episode_specs,
            units,
            scored_horizons,
            class_mapper,
        ) in mapped_population_slices:
            for unit in units:
                ordinal += 1
                spec = episode_specs[unit.spec_index]
                unit_pairs: list[tuple[str, int, object]] = []
                unit_change_totals: defaultdict[str, int] = defaultdict(int)
                unit_transform_totals: defaultdict[str, int] = defaultdict(int)
                try:
                    episode = TaskMemoryEpisodeDataset(
                        base_dataset, (spec,), apply_augmentation=False
                    )[0]
                    batch = collator([episode])
                    targets: tuple[dict[str, object], ...] = ()
                    scan_ids = tuple(str(value) for value in spec.scan_ids)
                    if (
                        len(batch.stage_batches) != len(scan_ids)
                        or scored_horizons_for_episode(role, len(scan_ids))
                        != scored_horizons
                    ):
                        raise RefinerEvaluationError("refiner episode horizon differs")
                    state = None
                    parent_publisher = LagOnePublisher(
                        score_reducer="mean", iou_threshold=0.5
                    )
                    transform = CausalRefinerRevisionTransform(
                        system=system, refiner=refiner, device=device
                    )
                    refined_publisher = LagOnePublisher(
                        score_reducer="mean",
                        iou_threshold=0.5,
                        revision_mask_transform=transform,
                    )
                    for stage_batch in batch.stage_batches:
                        data, low_targets, names = stage_batch.model_batch
                        meta = stage_batch.stage_meta[0]
                        _validate_collated_stage_identity(
                            names=names, scan_ids_in_window=meta.scan_ids_in_window
                        )
                        full_targets = getattr(data, "target_full", None)
                        if (
                            not isinstance(full_targets, Sequence)
                            or len(full_targets) != 1
                        ):
                            raise RefinerEvaluationError(
                                "refiner stage lacks full target"
                            )
                        full_target = {
                            name: (
                                value.detach().cpu().clone()
                                if isinstance(value, torch.Tensor)
                                else value
                            )
                            for name, value in full_targets[0].items()
                        }
                        data = _move_data_to_device(data, device)
                        low_targets = _move_targets_to_device(low_targets, device)
                        low_target = low_targets[0]
                        segment_stages = _segment_stages(low_target)
                        latest_stage = int(segment_stages.max().item())
                        raw_coordinates = system._process_raw_coordinates(data)
                        with torch.inference_mode():
                            output = system(
                                data,
                                point2segment=[low_target["point2segment"]],
                                raw_coordinates=raw_coordinates,
                                is_eval=True,
                            )
                        local = build_local_observation(
                            output,
                            [segment_stages],
                            latest_stage=latest_stage,
                            **observation_settings,
                        )
                        observation = _window_observation(
                            local_observation=local,
                            output=output,
                            segment_stages=segment_stages,
                            latest_stage=latest_stage,
                            confidence_threshold=observation_settings[
                                "confidence_threshold"
                            ],
                            mask_threshold=observation_settings["mask_threshold"],
                            minimum_mask_support=observation_settings[
                                "minimum_mask_support"
                            ],
                        )
                        current_masks = _latest_full_resolution_masks(
                            system,
                            output,
                            low_target,
                            data,
                            latest_local_stage=latest_stage,
                        )
                        raw = cache_payload_from_inference(
                            key=build_live_cache_key(
                                master_sequence_id=spec.source_sequence_id,
                                reference_scene_id=spec.reference_id,
                                stage_index=meta.absolute_stage_index,
                                scan_ids=spec.scan_ids,
                                local_window_scan_ids=meta.scan_ids_in_window,
                            ),
                            provenance=provenance,
                            observation=local,
                            full_masks=current_masks,
                            full_target=full_target,
                            latest_local_stage=latest_stage,
                        )
                        targets = append_live_refiner_target(
                            targets,
                            raw,
                            stage_index=meta.absolute_stage_index,
                        )
                        state, identity_map = advance_d0_identity(
                            observation=observation, state=state, stage_meta=meta
                        )
                        prediction = extract_official_task_prediction(
                            system=system,
                            output=output,
                            target_low_resolution=low_target,
                            target_full_resolution=full_target,
                            data=data,
                            class_mapper=class_mapper,
                            latest_stage_index=latest_stage,
                            return_soft_evidence=True,
                        )
                        parent_prefix = parent_publisher.update(
                            prediction, identity_map, meta
                        )
                        refined_prefix = refined_publisher.update(
                            prediction, identity_map, meta
                        )
                        if parent_prefix.keys != refined_prefix.keys:
                            raise RefinerEvaluationError(
                                "refiner changed D0 identities"
                            )
                        for name, value in validate_mask_only_refiner_output(
                            parent_prefix.prediction, refined_prefix.prediction
                        ).items():
                            unit_change_totals[name] += value
                        for name, value in transform.last_audit.items():
                            unit_transform_totals[name] += value
                        horizon = meta.absolute_stage_index + 1
                        if horizon in scored_horizons:
                            target = _target_for_prefix(
                                targets, horizon=horizon, class_mapper=class_mapper
                            )
                            unit_pairs.append(
                                (
                                    "PARENT",
                                    horizon,
                                    validate_causal_prefix_pair(
                                        prediction=parent_prefix.prediction,
                                        target=target,
                                        horizon=horizon,
                                        observed_scan_ids=scan_ids[:horizon],
                                    ),
                                )
                            )
                            unit_pairs.append(
                                (
                                    "R-REFINE",
                                    horizon,
                                    validate_causal_prefix_pair(
                                        prediction=refined_prefix.prediction,
                                        target=target,
                                        horizon=horizon,
                                        observed_scan_ids=scan_ids[:horizon],
                                    ),
                                )
                            )
                        del (
                            data,
                            low_targets,
                            output,
                            local,
                            observation,
                            current_masks,
                            prediction,
                        )
                    if len(unit_pairs) != 2 * len(scored_horizons):
                        raise RefinerEvaluationError("refiner unit lacks scored pairs")
                except (
                    OSError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                    IndexError,
                ) as error:
                    incomplete_units.append(
                        {
                            "logical_unit_id": unit.logical_unit_id,
                            "reference_id": unit.reference_id,
                            "sequence_id": unit.sequence_id,
                            "error_type": type(error).__name__,
                            "reason": str(error),
                        }
                    )
                    torch.cuda.empty_cache()
                    continue
                for method, horizon, pair in unit_pairs:
                    accumulator(method, horizon, "all").update(pair)
                    accumulator(method, horizon, unit.reference_id).update(pair)
                for name, value in unit_change_totals.items():
                    change_totals[name] += value
                for name, value in unit_transform_totals.items():
                    transform_totals[name] += value
                completed_units.append(unit.logical_unit_id)
                completed_references.add(unit.reference_id)
                for horizon in scored_horizons:
                    completed_units_by_horizon[horizon].append(unit.logical_unit_id)
                    completed_references_by_horizon[horizon].add(unit.reference_id)
                    reference_unit_counts[(horizon, unit.reference_id)] += 1
                print(
                    json.dumps(
                        {
                            "completed": ordinal,
                            "refiner_update": refiner_update,
                            "role": role,
                            "total": total_units,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del episode
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not completed_units:
        first_failure = incomplete_units[0] if incomplete_units else None
        raise RefinerEvaluationError(
            f"refiner evaluation completed no units; first_failure={first_failure}"
        )

    metric_rows = []
    evaluated_horizons = tuple(sorted(completed_units_by_horizon))
    for method in ("PARENT", "R-REFINE"):
        for horizon in evaluated_horizons:
            horizon_references = completed_references_by_horizon[horizon]
            for reference in ("all", *sorted(horizon_references)):
                metric_rows.append(
                    _metric_row(
                        method=method,
                        horizon=horizon,
                        reference=reference,
                        reference_count=(
                            len(horizon_references) if reference == "all" else 1
                        ),
                        logical_unit_count=(
                            len(completed_units_by_horizon[horizon])
                            if reference == "all"
                            else reference_unit_counts[(horizon, reference)]
                        ),
                        accumulator=accumulators[(method, horizon, reference)],
                    )
                )
    refined_metrics = {
        int(row["T"]): float(row["t_mAP"])
        for row in metric_rows
        if row["method"] == "R-REFINE" and row["reference"] == "all"
    }
    parent_metrics = {
        int(row["T"]): float(row["t_mAP"])
        for row in metric_rows
        if row["method"] == "PARENT" and row["reference"] == "all"
    }
    candidate = {
        "method_id": "R-REFINE",
        "optimizer_update": refiner_update,
        "new_parameter_count": refiner_identity["parameter_count"],
        "metrics": refined_metrics,
        "coverage_status": (
            "COMPLETE" if len(completed_units) == total_units else "INCOMPLETE"
        ),
        "checkpoint": refiner_identity["checkpoint"],
        "checkpoint_sha256": refiner_identity["sha256"],
        "expected_logical_units": total_units,
        "completed_logical_units": len(completed_units),
    }
    status = (
        "PASS"
        if candidate["coverage_status"] == "COMPLETE" and not incomplete_units
        else "PARTIAL"
    )
    summary = {
        "schema_version": "perception-refiner-evaluation-v1",
        "status": status,
        "data_role": role,
        "output_policy": "D0/lag1/mean+mask-only-refiner",
        "official_metric": "pooled_t_mAP",
        "parent": {
            "variant": parent_variant,
            "optimizer_update": parent_update,
            "checkpoint": parent_reference,
            "checkpoint_sha256": parent_sha,
            "load_audit": parent_load,
            "weight_sources": parent_sources,
        },
        "refiner": refiner_identity,
        "resolved_config_sha256": config_sha,
        "population_manifest_sha256": population_manifest_sha256,
        "evaluation_input_mode": "LIVE_LOCAL_DATASET",
        "roles_sha256": _sha256(roles_path),
        "declared_reference_count": len(references),
        "expected_reference_count": len(
            {
                unit.reference_id
                for _, _, units, _, _ in mapped_population_slices
                for unit in units
            }
        ),
        "completed_reference_count": len(completed_references),
        "expected_logical_unit_count": total_units,
        "completed_logical_unit_count": len(completed_units),
        "population_by_horizon": {
            str(horizon): {
                "reference_count": len(completed_references_by_horizon[horizon]),
                "logical_unit_count": len(completed_units_by_horizon[horizon]),
            }
            for horizon in evaluated_horizons
        },
        "completed_units": completed_units,
        "incomplete_units": incomplete_units,
        "candidate": candidate,
        "parent_metrics": parent_metrics,
        "metric_rows": metric_rows,
        "mask_change_audit": dict(change_totals),
        "pairing_audit": dict(transform_totals),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "source_sha256": _sha256(Path(__file__)),
    }
    if output_path is None:
        output_path = (
            ARTIFACT_ROOT
            / "training/refiner/evaluation"
            / role.lower()
            / f"update={refiner_update:04d}.json"
        )
    _atomic_json(output_path, summary)
    del system, refiner
    torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-variant", required=True)
    parser.add_argument("--parent-update", type=int, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path)
    parser.add_argument("--refiner-update", type=int, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--role", choices=("CAL", "SEL", "PB", "ADDITIONAL"), required=True
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_refiner_evaluation(
        parent_variant=arguments.parent_variant,
        parent_update=arguments.parent_update,
        parent_checkpoint=arguments.parent_checkpoint,
        scorer_checkpoint=arguments.scorer_checkpoint,
        refiner_update=arguments.refiner_update,
        refiner_checkpoint=arguments.refiner_checkpoint,
        role=arguments.role,
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
    "RefinerEvaluationError",
    "append_live_refiner_target",
    "main",
    "run_refiner_evaluation",
    "validate_mask_only_refiner_output",
]
