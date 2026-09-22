#!/usr/bin/env python3
"""Generate causal frozen-parent soft records for Perception Gain V1 refiner."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.perception_gain_data import (
    select_refiner_references,
    sidecar_cache_bytes,
    write_soft_sidecar_shard,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/home/ww/persist4d_runs/perception_gain_v1/assets.local.json")
DEFAULT_ROLES = PROJECT_ROOT / "artifacts/perception_gain_v1/DATA_ROLES.json"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
PUBLIC_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1/training/refiner"
MAXIMUM_CACHE_BYTES = 32 * 1024**3


class RefinerDataError(RuntimeError):
    """Raised when frozen-parent soft data cannot satisfy the refiner contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RefinerDataError(f"JSON root must be an object: {path}")
    return value


def advance_d0_identity(
    *, observation: object, state: object | None, stage_meta: object
) -> tuple[object, dict[int, tuple[int, int]]]:
    """Advance the frozen D0/LAST route and return its committed identities."""

    from datasets.task_memory_episode import StageMeta
    from models.task_memory_routing import (
        PredictionObservation,
        commit_entities,
        route_entities,
    )
    from models.task_memory_state import TaskMemoryConfig, TaskMemoryState

    if not isinstance(observation, PredictionObservation) or not isinstance(
        stage_meta, StageMeta
    ):
        raise RefinerDataError("D0 identity inputs have the wrong type")
    config = TaskMemoryConfig(
        class_weight=0.25,
        association_threshold=0.5,
        update_mode="last",
        update_rate=0.2,
        max_update_rate=0.2,
    )
    if state is None:
        state = TaskMemoryState.empty(
            batch_size=observation.batch_size,
            capacity=100,
            feature_dim=observation.features.shape[2],
            class_count=observation.class_prob.shape[2],
            device=observation.features.device,
            dtype=observation.features.dtype,
            config=config,
        )
    elif not isinstance(state, TaskMemoryState) or state.config != config:
        raise RefinerDataError("D0 identity state differs from the frozen contract")
    route = route_entities(observation, state, [stage_meta])
    commit = commit_entities(observation, route, state, [stage_meta])
    return commit.state.detach(), commit.identity_map()


def build_refiner_episode_inventory(
    *,
    sequence_names: Sequence[str],
    sequence_indices: Sequence[Sequence[int]],
    reference_ids: Sequence[str],
    source_context_indices: Sequence[int] | None = None,
    reference_limit: int = 32,
    minimum_references: int = 8,
) -> dict[str, object]:
    if not (len(sequence_names) == len(sequence_indices) == len(reference_ids)):
        raise RefinerDataError("refiner context identity lengths differ")
    if source_context_indices is None:
        source_context_indices = tuple(range(len(sequence_names)))
    if len(source_context_indices) != len(sequence_names):
        raise RefinerDataError("refiner source context lengths differ")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    scan_ids_by_reference: dict[str, list[str]] = defaultdict(list)
    for name, raw_indices, reference, source_index in zip(
        sequence_names,
        sequence_indices,
        reference_ids,
        source_context_indices,
        strict=True,
    ):
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(reference, str)
            or not reference
            or isinstance(raw_indices, (str, bytes))
            or not isinstance(raw_indices, Sequence)
            or isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index < 0
        ):
            raise RefinerDataError("refiner context identity is invalid")
        scan_ids = tuple(name.split("-"))
        try:
            indices = tuple(int(value) for value in raw_indices)
        except (TypeError, ValueError) as error:
            raise RefinerDataError("refiner scan indices are invalid") from error
        if (
            len(scan_ids) != len(indices)
            or len(set(scan_ids)) != len(scan_ids)
            or len(set(indices)) != len(indices)
            or len(scan_ids) < 3
            or any(index < 0 for index in indices)
        ):
            continue
        grouped[reference].append(
            {
                "reference_id": reference,
                "sequence_id": name,
                "scan_ids": scan_ids,
                "scan_indices": indices,
                "source_context_index": source_index,
            }
        )
        scan_ids_by_reference[reference].extend(scan_ids)
    selected = select_refiner_references(
        scan_ids_by_reference,
        limit=reference_limit,
        minimum_references=minimum_references,
    )
    episodes = []
    for reference in selected["references"]:
        candidates = sorted(grouped[reference], key=lambda row: str(row["sequence_id"]))
        if not candidates:
            continue
        master = candidates[0]
        count = min(5, len(master["scan_ids"]))
        canonical_ids = tuple(master["scan_ids"][:count])
        canonical_indices = tuple(master["scan_indices"][:count])
        for order_id, scan_ids, indices in (
            ("canonical", canonical_ids, canonical_indices),
            (
                "reverse",
                tuple(reversed(canonical_ids)),
                tuple(reversed(canonical_indices)),
            ),
        ):
            episodes.append(
                {
                    "reference_id": reference,
                    "order_id": order_id,
                    "source_context_index": master["source_context_index"],
                    "source_master_sequence_id": master["sequence_id"],
                    "sequence_id": "-".join(scan_ids),
                    "scan_ids": scan_ids,
                    "scan_indices": indices,
                }
            )
    return {
        "status": selected["status"],
        "reference_count": selected["reference_count"],
        "references": selected["references"],
        "episodes": tuple(episodes),
    }


def run(
    *,
    variant: str,
    optimizer_update: int,
    checkpoint: Path | None,
    scorer_checkpoint: Path | None,
    assets_path: Path,
    roles_path: Path,
    external_root: Path,
    device_name: str,
    shard_records: int = 128,
    output_root: Path | None = None,
    manifest_output: Path | None = None,
    recipe_config: Mapping[str, object] | None = None,
    required_inventory: Mapping[str, object] | None = None,
) -> dict[str, object]:
    import hydra
    import torch
    import yaml
    from omegaconf import OmegaConf

    from datasets.task_memory_episode import (
        NativeEpisodeMaster,
        TaskMemoryEpisodeCollator,
        TaskMemoryEpisodeDataset,
        TaskMemoryEpisodeSpec,
    )
    from models.perception_gain import CausalMaskRefiner
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.evaluate_task_memory import _ProtocolOrderDataset
    from scripts.perception_gain_evaluation import (
        VARIANTS,
        _atomic_json,
        _external_reference,
        _load_evaluation_weights,
        _sha256,
    )
    from scripts.perception_gain_foundation import (
        _resolve_cache_assets,
        resolve_live_assets,
    )
    from scripts.preflight_task_memory_episode import (
        _rio_base_dataset,
        load_reference_by_scene,
    )
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_task_memory_controls import _window_observation
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.task_memory_output import LagOnePublisher
    from scripts.train_perception_gain import (
        compose_variant_config,
        filter_rio_train_indices,
    )
    from scripts.train_perception_refiner import (
        CausalRefinerRevisionTransform,
        build_refiner_training_records,
    )
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    if variant not in VARIANTS or optimizer_update < 0 or shard_records <= 0:
        raise RefinerDataError("refiner producer request is invalid")
    assets = (
        resolve_live_assets(assets_path)
        if recipe_config is not None
        else _resolve_cache_assets(assets_path)
    )
    code = None
    if recipe_config is not None:
        from scripts.perception_gain_v2_config import live_execution_provenance

        code = live_execution_provenance(recipe_config)
    roles = _read_json(roles_path).get("roles")
    train_references = roles.get("TRAIN") if isinstance(roles, Mapping) else None
    if not isinstance(train_references, list) or not train_references:
        raise RefinerDataError("TRAIN references are unavailable")
    device = _validate_cuda_device(device_name)
    config = compose_variant_config(
        variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=(
            output_root.parent / "feature_producer"
            if output_root is not None
            else external_root / "training/refiner/feature_producer"
        ),
        recipe_config=recipe_config,
    )
    config.model.return_query_features = True
    if variant == "Q-SEM" and scorer_checkpoint is not None:
        config.perception_training.scorer_checkpoint = str(scorer_checkpoint)
    base = _rio_base_dataset(config, data_root=Path(assets["data_root"]), horizon=5)
    indices, references = filter_rio_train_indices(
        base,
        reference_by_scene=load_reference_by_scene(Path(assets["rio_metadata"])),
        train_references=set(train_references),
    )
    empty_contexts = {
        int(value["sequence_index"]) for value in base.known_empty_scan_contexts
    }
    kept = [
        (index, reference)
        for index, reference in zip(indices, references, strict=True)
        if index not in empty_contexts
    ]
    inventory = build_refiner_episode_inventory(
        sequence_names=tuple(str(base.sequence_names[index]) for index, _ in kept),
        sequence_indices=tuple(
            tuple(base.sequence_indices[index]) for index, _ in kept
        ),
        reference_ids=tuple(reference for _, reference in kept),
        source_context_indices=tuple(index for index, _ in kept),
    )
    bases_by_horizon = {5: base}
    if recipe_config is not None:
        # Include references with three/four real scans before applying the fixed
        # hash selection. Each reference uses its longest available real master.
        for horizon in (3, 4):
            bases_by_horizon[horizon] = _rio_base_dataset(
                config,
                data_root=Path(assets["data_root"]),
                horizon=horizon,
            )
        candidates = []
        reference_horizons = {}
        reference_by_scene = load_reference_by_scene(Path(assets["rio_metadata"]))
        for horizon, source_base in sorted(bases_by_horizon.items()):
            source_indices, source_references = filter_rio_train_indices(
                source_base,
                reference_by_scene=reference_by_scene,
                train_references=set(train_references),
            )
            for index, reference in zip(source_indices, source_references, strict=True):
                name = str(source_base.sequence_names[index])
                scan_indices = tuple(
                    int(value) for value in source_base.sequence_indices[index]
                )
                if (
                    len(set(name.split("-"))) != horizon
                    or len(set(scan_indices)) != horizon
                ):
                    continue
                reference_horizons[reference] = horizon
                candidates.append((horizon, reference, index, name, scan_indices))
        candidates = [row for row in candidates if row[0] == reference_horizons[row[1]]]
        inventory = build_refiner_episode_inventory(
            sequence_names=tuple(row[3] for row in candidates),
            sequence_indices=tuple(row[4] for row in candidates),
            reference_ids=tuple(row[1] for row in candidates),
            source_context_indices=tuple(row[2] for row in candidates),
        )
        if required_inventory is not None and canonical_json_sha256(
            inventory
        ) != canonical_json_sha256(required_inventory):
            raise RefinerDataError("V2 refiner TRAIN inventory differs from R1")
    if inventory["status"] != "PASS":
        raise RefinerDataError("refiner inventory lacks eight TRAIN references")
    if recipe_config is not None and manifest_output is not None:
        _atomic_json(
            manifest_output.with_name("TRAIN_INVENTORY.json"),
            {
                "inventory": inventory,
                "inventory_sha256": canonical_json_sha256(inventory),
                "recipe": dict(recipe_config),
                "roles_sha256": _sha256(roles_path),
                "execution_provenance": code,
            },
        )
    episode_rows = inventory["episodes"]
    wrapped = _ProtocolOrderDataset(
        base,
        sequence_names=[row["sequence_id"] for row in episode_rows],
        sequence_indices=[row["scan_indices"] for row in episode_rows],
        source_context_indices=[row["source_context_index"] for row in episode_rows],
    )
    wrapped_by_horizon = {
        horizon: _ProtocolOrderDataset(
            source_base,
            sequence_names=[row["sequence_id"] for row in episode_rows],
            sequence_indices=[row["scan_indices"] for row in episode_rows],
            source_context_indices=[
                row["source_context_index"] for row in episode_rows
            ],
        )
        for horizon, source_base in bases_by_horizon.items()
    }
    masters = tuple(
        NativeEpisodeMaster(
            reference_id=str(row["reference_id"]),
            sequence_id=str(row["sequence_id"]),
            scan_ids=tuple(row["scan_ids"]),
            scan_indices=tuple(row["scan_indices"]),
            role="TRAIN",
            context_index=index,
        )
        for index, row in enumerate(episode_rows)
    )
    specs = tuple(
        TaskMemoryEpisodeSpec.from_master(
            master,
            horizon=len(master.scan_ids),
            augmentation_seed=45,
            draw_index=index,
            bucket=f"T{len(master.scan_ids)}",
        )
        for index, master in enumerate(masters)
    )

    system = PerceptionGainTrainer(config)
    load_audit, checkpoint_sha, checkpoint_reference, _, weight_sources = (
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
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(config.data.validation_collation)
    )
    class_mapper = build_rio_class_mapper(base)
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
    output_root = (
        output_root
        if output_root is not None
        else external_root / "training/refiner/data"
    )
    try:
        output_reference_root = output_root.resolve().relative_to(
            external_root.resolve()
        )
    except ValueError as error:
        raise RefinerDataError(
            "refiner output root must be within external root"
        ) from error
    shards = []
    written_count = 0
    completed_references = set()
    zero_refiner = CausalMaskRefiner() if recipe_config is not None else None

    def write_records(values):
        cache_limit = MAXIMUM_CACHE_BYTES
        if recipe_config is not None:
            cache_root = external_root / "cache"
            other_bytes = sum(
                path.stat().st_size
                for path in cache_root.rglob("*.pt")
                if not path.is_relative_to(output_root)
            )
            cache_limit -= other_bytes
        result = write_soft_sidecar_shard(
            output_root,
            shard_id=len(shards),
            records=values,
            maximum_cache_bytes=cache_limit,
        )
        shards.append(
            {
                **result,
                "external_reference": f"external:{output_reference_root}/{result['path']}",
            }
        )

    records: list[dict[str, object]] = []
    incomplete = []
    totals: defaultdict[str, int] = defaultdict(int)
    diagnostics: defaultdict[str, int] = defaultdict(int)
    started = time.perf_counter()
    with deterministic_inference_runtime(45, device):
        for ordinal, spec in enumerate(specs, start=1):
            episode_records = []
            try:
                episode = TaskMemoryEpisodeDataset(
                    (
                        wrapped_by_horizon[spec.horizon]
                        if recipe_config is not None
                        else wrapped
                    ),
                    (spec,),
                    apply_augmentation=False,
                )[0]
                batch = collator([episode])
                state = None
                transform = CausalRefinerRevisionTransform(
                    system=system,
                    refiner=(
                        zero_refiner
                        if zero_refiner is not None
                        else CausalMaskRefiner()
                    ),
                    device=device,
                )
                publisher = LagOnePublisher(
                    score_reducer="mean",
                    iou_threshold=0.5,
                    revision_mask_transform=transform,
                )
                parent_publisher = (
                    LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
                    if recipe_config is not None
                    else None
                )
                for stage_batch in batch.stage_batches:
                    data, targets, names = stage_batch.model_batch
                    meta = stage_batch.stage_meta[0]
                    if list(names) != ["-".join(meta.scan_ids_in_window)]:
                        raise RefinerDataError("refiner collator identity differs")
                    full_targets = getattr(data, "target_full", None)
                    if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
                        raise RefinerDataError("refiner stage lacks a full target")
                    full_target = {
                        key: (
                            value.detach().cpu().clone()
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        for key, value in full_targets[0].items()
                    }
                    data = _move_data_to_device(data, device)
                    targets = _move_targets_to_device(targets, device)
                    target = targets[0]
                    segment_stages = _segment_stages(target)
                    latest_stage = int(segment_stages.max().item())
                    raw_coordinates = system._process_raw_coordinates(data)
                    with torch.inference_mode():
                        output = system(
                            data,
                            point2segment=[target["point2segment"]],
                            raw_coordinates=raw_coordinates,
                            is_eval=True,
                        )
                    local_observation = build_local_observation(
                        output,
                        [segment_stages],
                        latest_stage=latest_stage,
                        **observation_settings,
                    )
                    observation = _window_observation(
                        local_observation=local_observation,
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
                    state, identity_map = advance_d0_identity(
                        observation=observation,
                        state=state,
                        stage_meta=meta,
                    )
                    prediction = extract_official_task_prediction(
                        system=system,
                        output=output,
                        target_low_resolution=target,
                        target_full_resolution=full_target,
                        data=data,
                        class_mapper=class_mapper,
                        latest_stage_index=latest_stage,
                        return_soft_evidence=True,
                    )
                    refined_prefix = publisher.update(prediction, identity_map, meta)
                    if parent_publisher is not None:
                        from scripts.perception_refiner_evaluation import (
                            validate_mask_only_refiner_output,
                        )

                        parent_prefix = parent_publisher.update(
                            prediction, identity_map, meta
                        )
                        change = validate_mask_only_refiner_output(
                            parent_prefix.prediction, refined_prefix.prediction
                        )
                        if (
                            parent_prefix.keys != refined_prefix.keys
                            or change["changed_point_count"]
                        ):
                            raise RefinerDataError(
                                "zero refiner changed real parent output"
                            )
                        diagnostics["zero_equivalence_prefixes"] += 1
                    if meta.absolute_stage_index:
                        stage_records, audit = build_refiner_training_records(
                            pair_records=transform.last_candidate_records,
                            new_prediction=prediction,
                            full_target=full_target,
                            stage_meta=meta,
                        )
                        episode_records.extend(stage_records)
                        for name, value in audit.items():
                            totals[name] += value
                        if recipe_config is not None:
                            for name, value in transform.last_audit.items():
                                diagnostics[name] += value
                            for record in stage_records:
                                new = record["new_logits"]
                                old = record["old_logits"]
                                target = record["target"]
                                wrong = (new > 0) != (target > 0.5)
                                modifiable = new.abs() < 2
                                old_closer = (old.sigmoid() - target).abs() < (
                                    new.sigmoid() - target
                                ).abs()
                                diagnostics["segments"] += new.numel()
                                diagnostics["wrong_segments"] += int(wrong.sum())
                                diagnostics["wrong_modifiable_segments"] += int(
                                    (wrong & modifiable).sum()
                                )
                                diagnostics["wrong_boundary_abs2_segments"] += int(
                                    (wrong & (new.abs() == 2)).sum()
                                )
                                for label, mask in (
                                    ("modifiable", modifiable),
                                    ("nonmodifiable", ~modifiable),
                                ):
                                    diagnostics[f"{label}_segments"] += int(mask.sum())
                                    diagnostics[f"old_closer_{label}_segments"] += int(
                                        (old_closer & mask).sum()
                                    )
                    del (
                        data,
                        targets,
                        output,
                        local_observation,
                        observation,
                        prediction,
                    )
                records.extend(episode_records)
                written_count += len(episode_records)
                completed_references.update(
                    str(record["reference_id"]) for record in episode_records
                )
                if recipe_config is not None:
                    while len(records) >= shard_records:
                        write_records(records[:shard_records])
                        del records[:shard_records]
                print(
                    json.dumps(
                        {
                            "completed": ordinal,
                            "reference_id": spec.reference_id,
                            "total": len(specs),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                incomplete.append(
                    {
                        "episode_id": spec.episode_id,
                        "reference_id": spec.reference_id,
                        "sequence_id": spec.source_sequence_id,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
                torch.cuda.empty_cache()
                if recipe_config is not None:
                    break
    torch.cuda.synchronize(device)
    for offset in range(0, len(records), shard_records):
        write_records(records[offset : offset + shard_records])
    elapsed = time.perf_counter() - started
    status = (
        "PASS"
        if not incomplete
        and len(completed_references) == inventory["reference_count"]
        and written_count
        else "PARTIAL"
    )
    summary = {
        "schema_version": "perception-refiner-data-v1",
        "status": status,
        "variant": variant,
        "optimizer_update": optimizer_update,
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha,
        "weight_sources": weight_sources,
        "resolved_config_sha256": config_sha,
        "roles_sha256": _sha256(roles_path),
        "p6a_config_sha256": _sha256(p6a_path),
        "selected_reference_count": inventory["reference_count"],
        "completed_reference_count": len(completed_references),
        "episode_count": len(specs),
        "candidate_count": written_count,
        "pairing_and_target_audit": dict(totals),
        "incomplete_units": incomplete,
        "load_audit": load_audit,
        "shards": shards,
        "cache_bytes": sidecar_cache_bytes(output_root),
        "cache_limit_bytes": MAXIMUM_CACHE_BYTES,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        **(
            {
                "recipe": dict(recipe_config),
                "execution_provenance": code,
                "inventory": inventory,
                "inventory_sha256": canonical_json_sha256(inventory),
                "residual_diagnostics": dict(diagnostics),
                "cache_binding": {
                    "executed_code_commit": code["executed_code_commit"],
                    "relevant_source_digest": code["relevant_source_digest"],
                    "parent_weight_hash": checkpoint_sha,
                    "inference_recipe_hash": recipe_config["inference_recipe_hash"],
                    "resolved_config_sha256": config_sha,
                    "input_manifest_hash": _sha256(
                        roles_path.parent / "data/STAGING_MANIFEST.json"
                    ),
                    "roles_sha256": _sha256(roles_path),
                    "inventory_sha256": canonical_json_sha256(inventory),
                    "point_order_transform": "canonical_vertices/identity_geometry",
                    "role": "TRAIN",
                    "eval_seed": 45,
                    "publisher": "D0/lag1/mean",
                },
                "diagnostic_scope": "Fixed TRAIN cache; segment targets follow new low partition. Counts are diagnostics, not an AP bound.",
            }
            if recipe_config is not None
            else {}
        ),
    }
    _atomic_json(
        (
            manifest_output
            if manifest_output is not None
            else PUBLIC_ROOT / "DATA_MANIFEST.json"
        ),
        summary,
    )
    del system
    torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-records", type=int, default=128)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--recipe-config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run(
        variant=arguments.variant,
        optimizer_update=arguments.update,
        checkpoint=arguments.checkpoint,
        scorer_checkpoint=arguments.scorer_checkpoint,
        assets_path=arguments.assets,
        roles_path=arguments.roles,
        external_root=arguments.external_root,
        device_name=arguments.device,
        shard_records=arguments.shard_records,
        output_root=arguments.output_root,
        manifest_output=arguments.manifest_output,
        recipe_config=(
            _read_json(arguments.recipe_config)
            if arguments.recipe_config is not None
            else None
        ),
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RefinerDataError",
    "advance_d0_identity",
    "build_refiner_episode_inventory",
    "main",
    "run",
]
