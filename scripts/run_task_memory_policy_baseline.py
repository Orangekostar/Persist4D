#!/usr/bin/env python3
"""Evaluate frozen R1+B4 under commit0 and lag1 publication policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from datasets.task_memory_episode import (
    NativeEpisodeMaster,
    StageMeta,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
)
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.task_memory_contracts import canonical_json_sha256
from scripts.task_memory_output import CommitZeroPublisher, LagOnePublisher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (2, 3, 4, 5)
POLICIES = ("commit0", "lag1")
EVALUATION_SEED = 45
DIAGNOSTIC_LIMIT = 12
CACHE_LIMIT_BYTES = 40 * 1024**3
CHECKPOINT_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"

POLICY_FIELDS = (
    "population_id",
    "reference_count",
    "master_count",
    "order_count",
    "source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "training_seed",
    "evaluation_seed",
    "policy",
    "reducer",
    "window",
    "K",
    "r",
    "state_bytes",
    "T",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "direct_current_AP",
    "forward_count",
    "archive_payload_bytes",
    "lag1_buffer_bytes",
    "materialized_output_bytes",
    "route_conflicts",
    "fallback_matches",
    "provisional_prefix_count",
    "attribution",
)

GAP_FIELDS = (
    "population_id",
    "diagnostic_panel_sha256",
    "source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "policy",
    "reducer",
    "T",
    "stratum",
    "master_count",
    "reference_count",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "direct_current_AP",
    "status",
)


class TaskMemoryPolicyBaselineError(RuntimeError):
    """Raised when policy baseline inputs or artifacts violate the contract."""


def select_diagnostic_masters(
    masters: Sequence[NativeEpisodeMaster], *, limit: int = DIAGNOSTIC_LIMIT
) -> tuple[NativeEpisodeMaster, ...]:
    """Choose a deterministic reference-round-robin diagnostic panel."""

    if (
        isinstance(masters, (str, bytes))
        or not isinstance(masters, Sequence)
        or not masters
        or any(not isinstance(master, NativeEpisodeMaster) for master in masters)
    ):
        raise TaskMemoryPolicyBaselineError("diagnostic masters are invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 12:
        raise TaskMemoryPolicyBaselineError("diagnostic limit must be within 1-12")
    if any(master.role != "development" for master in masters):
        raise TaskMemoryPolicyBaselineError(
            "diagnostic panel must use development only"
        )
    grouped: dict[str, list[NativeEpisodeMaster]] = defaultdict(list)
    for master in masters:
        grouped[master.reference_id].append(master)
    for values in grouped.values():
        values.sort(key=lambda value: value.sequence_id)
    selected = []
    round_index = 0
    while len(selected) < min(limit, len(masters)):
        added = False
        for reference_id in sorted(grouped):
            values = grouped[reference_id]
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == min(limit, len(masters)):
                    break
        if not added:
            break
        round_index += 1
    return tuple(selected)


def _target_ids(target: Mapping[str, object]) -> set[int]:
    values = target.get("gt_ids")
    if not isinstance(values, Tensor) or values.ndim != 1:
        raise TaskMemoryPolicyBaselineError("diagnostic target lacks rank-1 gt_ids")
    if values.dtype == torch.bool or values.is_floating_point():
        raise TaskMemoryPolicyBaselineError("diagnostic gt_ids must be integer")
    return {int(value) for value in values.detach().cpu().tolist()}


def classify_gap_event(
    stage_targets: Sequence[Mapping[str, object]], *, horizon: int
) -> str:
    """Classify whether a prefix contains a GT reappearance after absence."""

    if (
        isinstance(stage_targets, (str, bytes))
        or not isinstance(stage_targets, Sequence)
        or horizon not in range(1, 6)
        or len(stage_targets) < horizon
    ):
        raise TaskMemoryPolicyBaselineError("gap diagnostic prefix is invalid")
    visible = [_target_ids(stage_targets[index]) for index in range(horizon)]
    seen = set(visible[0])
    for stage in range(1, horizon):
        if any(
            identity in seen and identity not in visible[stage - 1]
            for identity in visible[stage]
        ):
            return "gap_event"
        seen.update(visible[stage])
    return "no_gap_event"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise TaskMemoryPolicyBaselineError("Git HEAD is invalid")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        status = subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False)
        if status.returncode != 0:
            raise TaskMemoryPolicyBaselineError(
                "tracked source tree must be clean before real baseline inference"
            )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryPolicyBaselineError(f"cannot decode JSON: {path}") from error
    if not isinstance(value, dict):
        raise TaskMemoryPolicyBaselineError(f"JSON root must be an object: {path}")
    return value


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TaskMemoryPolicyBaselineError(f"output cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    if not rows or any(set(row) != set(fields) for row in rows):
        raise TaskMemoryPolicyBaselineError(f"CSV rows differ for {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _panel_payload(
    masters: Sequence[NativeEpisodeMaster], *, source_commit: str, data_sha256: str
) -> dict[str, object]:
    selected = select_diagnostic_masters(masters)
    payload: dict[str, object] = {
        "schema_version": "task-memory-diagnostic-panel-v1",
        "selection_rule": (
            "reference_id_lexicographic_round_robin_then_sequence_id_lexicographic"
        ),
        "limit": DIAGNOSTIC_LIMIT,
        "source_commit": source_commit,
        "data_contract_sha256": data_sha256,
        "master_count": len(selected),
        "reference_count": len({master.reference_id for master in selected}),
        "masters": [
            {
                "reference_id": master.reference_id,
                "sequence_id": master.sequence_id,
                "scan_ids": list(master.scan_ids),
            }
            for master in selected
        ],
        "contains_model_outcomes": False,
        "status": "PREREGISTERED",
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _episode_specs(
    masters: Sequence[NativeEpisodeMaster],
) -> tuple[TaskMemoryEpisodeSpec, ...]:
    return tuple(
        TaskMemoryEpisodeSpec.from_master(
            master,
            horizon=5,
            augmentation_seed=EVALUATION_SEED,
            draw_index=index,
            bucket="T5",
        )
        for index, master in enumerate(
            sorted(masters, key=lambda value: value.sequence_id)
        )
    )


def _meta_payload(meta: StageMeta) -> dict[str, object]:
    return {
        "reference_id": meta.reference_id,
        "episode_id": meta.episode_id,
        "scan_ids_in_window": list(meta.scan_ids_in_window),
        "absolute_stage_index": meta.absolute_stage_index,
        "local_stage_ids": meta.local_stage_ids.clone(),
        "original_vertex_ids": [value.clone() for value in meta.original_vertex_ids],
        "scan_vertex_offsets": meta.scan_vertex_offsets.clone(),
        "point2segment": meta.point2segment.clone(),
        "segment_stage_ids": meta.segment_stage_ids.clone(),
        "augmentation_transform_id": meta.augmentation_transform_id,
        "coordinate_frame_id": meta.coordinate_frame_id,
        "voxel_inverse": meta.voxel_inverse.clone(),
        "full_resolution_point2segment": meta.full_resolution_point2segment.clone(),
    }


def _meta_from_payload(value: Mapping[str, object]) -> StageMeta:
    return StageMeta(
        reference_id=str(value["reference_id"]),
        episode_id=str(value["episode_id"]),
        scan_ids_in_window=tuple(value["scan_ids_in_window"]),
        absolute_stage_index=int(value["absolute_stage_index"]),
        local_stage_ids=value["local_stage_ids"],
        original_vertex_ids=tuple(value["original_vertex_ids"]),
        scan_vertex_offsets=value["scan_vertex_offsets"],
        point2segment=value["point2segment"],
        segment_stage_ids=value["segment_stage_ids"],
        augmentation_transform_id=str(value["augmentation_transform_id"]),
        coordinate_frame_id=str(value["coordinate_frame_id"]),
        voxel_inverse=value["voxel_inverse"],
        full_resolution_point2segment=value["full_resolution_point2segment"],
    )


def _prediction_payload(prediction: OfficialTaskPrediction) -> dict[str, Tensor | int]:
    prediction.validate()
    return {
        "pred_masks": prediction.pred_masks.clone(),
        "pred_scores": prediction.pred_scores.clone(),
        "pred_classes": prediction.pred_classes.clone(),
        "source_query_ids": prediction.source_query_ids.clone(),
        "source_class_ids": prediction.source_class_ids.clone(),
        "temporal_stages": prediction.temporal_stages.clone(),
        "latest_stage_index": prediction.latest_stage_index,
        "latest_stage_masks": prediction.latest_stage_masks.clone(),
    }


def _prediction_from_payload(value: Mapping[str, object]) -> OfficialTaskPrediction:
    prediction = OfficialTaskPrediction(
        pred_masks=value["pred_masks"],
        pred_scores=value["pred_scores"],
        pred_classes=value["pred_classes"],
        source_query_ids=value["source_query_ids"],
        source_class_ids=value["source_class_ids"],
        temporal_stages=value["temporal_stages"],
        latest_stage_index=int(value["latest_stage_index"]),
        latest_stage_masks=value["latest_stage_masks"],
    )
    prediction.validate()
    return prediction


def _cache_identity(
    provenance: Mapping[str, object], spec: TaskMemoryEpisodeSpec
) -> str:
    return canonical_json_sha256(
        {
            "provenance": dict(provenance),
            "reference_id": spec.reference_id,
            "sequence_id": spec.source_sequence_id,
            "scan_ids": list(spec.scan_ids),
            "augmentation_transform_id": "identity-v1",
        }
    )


def _state_bytes(state: object) -> int:
    tensors = getattr(state, "tensors", None)
    if not callable(tensors):
        raise TaskMemoryPolicyBaselineError("B4 state lacks tensor accounting")
    return sum(value.numel() * value.element_size() for value in tensors())


def _stage_target(raw_payload: Mapping[str, object]) -> dict[str, object]:
    target = raw_payload.get("target")
    if not isinstance(target, Mapping):
        raise TaskMemoryPolicyBaselineError("raw inference target is unavailable")
    return {
        key: value.clone() if isinstance(value, Tensor) else value
        for key, value in target.items()
    }


def _validate_collated_stage_identity(
    *, names: object, scan_ids_in_window: Sequence[str]
) -> None:
    expected = "-".join(scan_ids_in_window)
    if (
        isinstance(names, (str, bytes))
        or not isinstance(names, Sequence)
        or list(names) != [expected]
    ):
        raise TaskMemoryPolicyBaselineError(
            "collator changed the requested local-window identity"
        )


def _produce_episode(
    *,
    episode: object,
    collator: TaskMemoryEpisodeCollator,
    system: object,
    memory: object,
    observation_settings: Mapping[str, object],
    class_mapper: Callable[[int], int],
    device: torch.device,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _latest_full_resolution_masks,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import cache_payload_from_inference
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    batch = collator([episode])
    if len(batch.specs) != 1 or len(batch.stage_batches) != 5:
        raise TaskMemoryPolicyBaselineError("baseline episode batch must be one H5")
    state = None
    stages_out = []
    peak_state_bytes = 0
    for stage_batch in batch.stage_batches:
        data, targets, names = stage_batch.model_batch
        meta = stage_batch.stage_meta[0]
        spec = batch.specs[0]
        _validate_collated_stage_identity(
            names=names, scan_ids_in_window=meta.scan_ids_in_window
        )
        if len(targets) != 1:
            raise TaskMemoryPolicyBaselineError("collator changed episode batch size")
        full_targets = getattr(data, "target_full", None)
        if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
            raise TaskMemoryPolicyBaselineError("collated stage lacks full target")
        full_target = full_targets[0]
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target = targets[0]
        segment_stages = _segment_stages(target)
        latest_local_stage = int(segment_stages.max().item())
        raw_coordinates = system._process_raw_coordinates(data)
        with torch.inference_mode():
            output = system(
                data,
                point2segment=[target["point2segment"]],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        observation = build_local_observation(
            output,
            [segment_stages],
            latest_stage=latest_local_stage,
            **dict(observation_settings),
        )
        if state is None:
            state = memory.empty_state(observation)
        step = memory.step(observation, state, stage_index=meta.absolute_stage_index)
        state = step.state.detach()
        peak_state_bytes = max(peak_state_bytes, _state_bytes(state))
        identity_map = {
            query_id: (int(slot_id), 0)
            for query_id, slot_id in enumerate(step.slot_ids[0].detach().cpu().tolist())
            if slot_id >= 0
        }
        current_masks = _latest_full_resolution_masks(
            system,
            output,
            target,
            data,
            latest_local_stage=latest_local_stage,
        )
        raw = cache_payload_from_inference(
            key={
                "master_sequence_id": spec.source_sequence_id,
                "reference_scene_id": spec.reference_id,
                "order_id": "canonical",
                "stage_index": meta.absolute_stage_index,
                "history_scan_ids": list(
                    spec.scan_ids[: meta.absolute_stage_index + 1]
                ),
                "local_window_scan_ids": list(meta.scan_ids_in_window),
            },
            provenance=provenance,
            observation=observation,
            full_masks=current_masks,
            full_target=full_target,
            latest_local_stage=latest_local_stage,
        )
        official = extract_official_task_prediction(
            system=system,
            output=output,
            target_low_resolution=target,
            target_full_resolution=full_target,
            data=data,
            class_mapper=class_mapper,
            latest_stage_index=latest_local_stage,
        )
        stages_out.append(
            {
                "prediction": _prediction_payload(official),
                "identity_map": identity_map,
                "stage_meta": _meta_payload(meta),
                "target": _stage_target(raw),
            }
        )
        del output, observation, current_masks, data, targets
    return {
        "schema_version": "task-memory-policy-cache-v1",
        "provenance": dict(provenance),
        "episode": {
            "reference_id": batch.specs[0].reference_id,
            "sequence_id": batch.specs[0].source_sequence_id,
            "scan_ids": list(batch.specs[0].scan_ids),
            "episode_id": batch.specs[0].episode_id,
        },
        "peak_state_bytes": peak_state_bytes,
        "stages": stages_out,
    }


def _validate_cache(
    value: object,
    *,
    provenance: Mapping[str, object],
    spec: TaskMemoryEpisodeSpec,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TaskMemoryPolicyBaselineError("policy cache root must be a mapping")
    result = dict(value)
    episode = result.get("episode")
    stages = result.get("stages")
    if (
        result.get("schema_version") != "task-memory-policy-cache-v1"
        or result.get("provenance") != dict(provenance)
        or not isinstance(episode, Mapping)
        or episode.get("reference_id") != spec.reference_id
        or episode.get("sequence_id") != spec.source_sequence_id
        or episode.get("scan_ids") != list(spec.scan_ids)
        or not isinstance(stages, list)
        or len(stages) != 5
    ):
        raise TaskMemoryPolicyBaselineError("policy cache identity differs")
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise TaskMemoryPolicyBaselineError("policy cache stage is invalid")
        prediction = _prediction_from_payload(stage["prediction"])
        meta = _meta_from_payload(stage["stage_meta"])
        target = stage.get("target")
        identity_map = stage.get("identity_map")
        if (
            meta.absolute_stage_index != index
            or prediction.pred_masks.shape[0] != meta.local_stage_ids.numel()
            or not isinstance(target, Mapping)
            or not isinstance(identity_map, Mapping)
        ):
            raise TaskMemoryPolicyBaselineError("policy cache stage lineage differs")
        _target_ids(target)
    peak = result.get("peak_state_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise TaskMemoryPolicyBaselineError("policy cache state bytes are invalid")
    return result


def _save_cache(path: Path, value: Mapping[str, object]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if (
        path.exists()
        or digest_path.exists()
        or path.is_symlink()
        or digest_path.is_symlink()
    ):
        raise TaskMemoryPolicyBaselineError("partial or duplicate cache output exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(value), temporary)
        digest = _file_sha256(temporary)
        temporary.replace(path)
        _atomic_bytes(digest_path, f"{digest}  {path.name}\n".encode("ascii"))
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size, digest


def _load_cache(
    path: Path,
    *,
    provenance: Mapping[str, object],
    spec: TaskMemoryEpisodeSpec,
) -> tuple[dict[str, object], str]:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if any(
        candidate.is_symlink() or not candidate.is_file()
        for candidate in (path, digest_path)
    ):
        raise TaskMemoryPolicyBaselineError(
            "policy cache file or digest is unavailable"
        )
    fields = digest_path.read_text(encoding="ascii").strip().split()
    if len(fields) != 2 or fields[1] != path.name or fields[0] != _file_sha256(path):
        raise TaskMemoryPolicyBaselineError("policy cache SHA256 differs")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TaskMemoryPolicyBaselineError(
            "policy cache cannot be loaded safely"
        ) from error
    return _validate_cache(value, provenance=provenance, spec=spec), fields[0]


def _target_for_prefix(
    targets: Sequence[Mapping[str, object]],
    *,
    horizon: int,
    class_mapper: Callable[[int], int],
) -> dict[str, Tensor]:
    from scripts.evaluate_persist4d_p6a import build_temporal_target

    stage_payloads = [
        {"key": {"stage_index": stage_index}, "target": target}
        for stage_index, target in enumerate(targets[:horizon])
    ]
    target = build_temporal_target(stage_payloads)
    target["labels"] = torch.tensor(
        [class_mapper(int(value)) for value in target["labels"].tolist()],
        dtype=torch.long,
    )
    return target


def _sequence_pairs(
    cache: Mapping[str, object], *, class_mapper: Callable[[int], int]
) -> tuple[dict[tuple[str, int], object], dict[tuple[str, int], dict[str, int]]]:
    from scripts.system_comparison_metrics import validate_causal_prefix_pair

    publishers = {
        "commit0": CommitZeroPublisher(score_reducer="mean"),
        "lag1": LagOnePublisher(score_reducer="mean", iou_threshold=0.5),
    }
    pairs = {}
    accounting = {}
    stages = cache["stages"]
    targets = [stage["target"] for stage in stages]
    episode = cache["episode"]
    for stage_index, stage in enumerate(stages):
        prediction = _prediction_from_payload(stage["prediction"])
        meta = _meta_from_payload(stage["stage_meta"])
        for policy, publisher in publishers.items():
            prefix = publisher.update(prediction, stage["identity_map"], meta)
            horizon = stage_index + 1
            if horizon not in HORIZONS:
                continue
            target = _target_for_prefix(
                targets, horizon=horizon, class_mapper=class_mapper
            )
            pairs[(policy, horizon)] = validate_causal_prefix_pair(
                prediction=prefix.prediction,
                target=target,
                horizon=horizon,
                observed_scan_ids=episode["scan_ids"][:horizon],
            )
            accounting[(policy, horizon)] = {
                "archive_payload_bytes": prefix.accounting.archive_payload_bytes,
                "lag1_buffer_bytes": prefix.accounting.lag1_buffer_bytes,
                "materialized_output_bytes": prefix.accounting.materialized_output_bytes,
                "route_conflicts": sum(
                    record.route_conflicts for record in prefix.revision_log
                ),
                "fallback_matches": sum(
                    record.fallback_matches for record in prefix.revision_log
                ),
                "provisional_prefix_count": int(prefix.provisional_scan_id is not None),
            }
    return pairs, accounting


def _metric_values(accumulator: object) -> dict[str, float]:
    values = accumulator.compute()
    return {
        "t_mAP": values["t_mAP"],
        "t_mAP50": values["t_mAP50"],
        "t_mAP25": values["t_mAP25"],
        "t_REC": values["t_REC"],
        "prefix_overall_mAP": values["prefix_overall_mAP"],
        "direct_current_AP": values["local_current_AP"],
    }


def _analyze(
    *,
    specs: Sequence[TaskMemoryEpisodeSpec],
    cache_root: Path,
    provenance: Mapping[str, object],
    panel: Mapping[str, object],
    class_mapper: Callable[[int], int],
    output_root: Path,
) -> dict[str, object]:
    from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec

    metric_spec = resolve_metric_dataset_spec(PROJECT_ROOT)
    main_accumulators = {
        (policy, horizon): AllTBaselineAccumulator(dataset_spec=metric_spec)
        for policy in POLICIES
        for horizon in HORIZONS
    }
    diagnostic_accumulators: dict[tuple[str, int, str], object] = {}
    diagnostic_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    diagnostic_references: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    accounting_totals: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    max_state_bytes = 0
    diagnostic_ids = {item["sequence_id"] for item in panel["masters"]}
    cache_records = []
    for index, spec in enumerate(specs):
        cache_id = _cache_identity(provenance, spec)
        path = cache_root / f"{cache_id}.pt"
        cache, cache_sha256 = _load_cache(path, provenance=provenance, spec=spec)
        digest_path = path.with_suffix(path.suffix + ".sha256")
        cache_records.append(
            {
                "filename": path.name,
                "bytes": path.stat().st_size + digest_path.stat().st_size,
                "sha256": cache_sha256,
                "sequence_id": spec.source_sequence_id,
                "reference_id": spec.reference_id,
            }
        )
        max_state_bytes = max(max_state_bytes, int(cache["peak_state_bytes"]))
        pairs, accounting = _sequence_pairs(cache, class_mapper=class_mapper)
        targets = [stage["target"] for stage in cache["stages"]]
        for key, pair in pairs.items():
            main_accumulators[key].update(pair)
            for name, value in accounting[key].items():
                if name in {
                    "archive_payload_bytes",
                    "lag1_buffer_bytes",
                    "materialized_output_bytes",
                }:
                    accounting_totals[key][name] = max(
                        accounting_totals[key][name], value
                    )
                else:
                    accounting_totals[key][name] += value
            if spec.source_sequence_id in diagnostic_ids:
                stratum = classify_gap_event(targets, horizon=key[1])
                diagnostic_key = (key[0], key[1], stratum)
                if diagnostic_key not in diagnostic_accumulators:
                    diagnostic_accumulators[diagnostic_key] = AllTBaselineAccumulator(
                        dataset_spec=metric_spec
                    )
                diagnostic_accumulators[diagnostic_key].update(pair)
                diagnostic_counts[diagnostic_key] += 1
                diagnostic_references[diagnostic_key].add(spec.reference_id)
        print(
            json.dumps(
                {
                    "analysis": index + 1,
                    "total": len(specs),
                    "sequence_id": spec.source_sequence_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    main_rows = []
    references = len({spec.reference_id for spec in specs})
    for policy in POLICIES:
        for horizon in HORIZONS:
            key = (policy, horizon)
            main_rows.append(
                {
                    "population_id": "development_train_holdout_47_masters_canonical",
                    "reference_count": references,
                    "master_count": len(specs),
                    "order_count": len(specs),
                    "source_commit": provenance["source_commit"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "config_sha256": provenance["config_sha256"],
                    "training_seed": EVALUATION_SEED,
                    "evaluation_seed": EVALUATION_SEED,
                    "policy": policy,
                    "reducer": "mean",
                    "window": "W2",
                    "K": 100,
                    "r": 0,
                    "state_bytes": max_state_bytes,
                    "T": horizon,
                    **_metric_values(main_accumulators[key]),
                    "forward_count": len(specs) * 5,
                    **accounting_totals[key],
                    "attribution": "complete_output_system_policy",
                }
            )

    gap_rows = []
    for policy in POLICIES:
        for horizon in HORIZONS:
            for stratum in ("gap_event", "no_gap_event"):
                key = (policy, horizon, stratum)
                count = diagnostic_counts[key]
                metrics: Mapping[str, object]
                if count:
                    metrics = _metric_values(diagnostic_accumulators[key])
                    status = "MEASURED"
                else:
                    metrics = {
                        name: ""
                        for name in (
                            "t_mAP",
                            "t_mAP50",
                            "t_mAP25",
                            "t_REC",
                            "prefix_overall_mAP",
                            "direct_current_AP",
                        )
                    }
                    status = "NOT_AVAILABLE_NO_SEQUENCES"
                gap_rows.append(
                    {
                        "population_id": "development_preregistered_diagnostic_12",
                        "diagnostic_panel_sha256": panel["content_sha256"],
                        "source_commit": provenance["source_commit"],
                        "checkpoint_sha256": provenance["checkpoint_sha256"],
                        "config_sha256": provenance["config_sha256"],
                        "policy": policy,
                        "reducer": "mean",
                        "T": horizon,
                        "stratum": stratum,
                        "master_count": count,
                        "reference_count": len(diagnostic_references[key]),
                        **metrics,
                        "status": status,
                    }
                )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "policy_comparison.csv", main_rows, POLICY_FIELDS)
    _write_csv(output_root / "gap_event_strata.csv", gap_rows, GAP_FIELDS)
    cache_bytes = sum(record["bytes"] for record in cache_records)
    manifest: dict[str, object] = {
        "schema_version": "task-memory-policy-cache-manifest-v1",
        "source_commit": provenance["source_commit"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "config_sha256": provenance["config_sha256"],
        "entry_count": len(cache_records),
        "cache_bytes": cache_bytes,
        "cache_limit_bytes": CACHE_LIMIT_BYTES,
        "records": cache_records,
        "status": "PASS" if cache_bytes <= CACHE_LIMIT_BYTES else "FAIL",
    }
    manifest["content_sha256"] = canonical_json_sha256(manifest)
    _atomic_json(output_root / "cache_manifest.json", manifest)
    if cache_bytes > CACHE_LIMIT_BYTES:
        raise TaskMemoryPolicyBaselineError(
            "policy cache exceeds the frozen 40 GiB cap"
        )
    return manifest


def run_baseline(
    *,
    mode: str,
    device_name: str,
    shard_index: int,
    shard_count: int,
    data_root: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    run_root: Path,
    output_root: Path,
) -> dict[str, object]:
    import hydra

    from models.persistent_memory import PersistentMemory
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        build_rio_class_mapper,
    )
    from scripts.preflight_task_memory_episode import (
        _rio_base_dataset,
        _role_by_reference,
        load_reference_by_scene,
    )
    from scripts.r1_downstream_context import (
        _load_r1_system,
        compose_r1_runtime_config,
        load_r1_contract,
        validate_external_file_identity,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime

    if mode not in {"produce", "analyze", "all"}:
        raise TaskMemoryPolicyBaselineError("mode must be produce, analyze, or all")
    if (
        isinstance(shard_index, bool)
        or isinstance(shard_count, bool)
        or not isinstance(shard_index, int)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
    ):
        raise TaskMemoryPolicyBaselineError("invalid shard selection")
    source_commit = _git_head()
    data_root = data_root.expanduser().resolve(strict=True)
    metadata_path = metadata_path.expanduser().resolve(strict=True)
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    pretrained_path = pretrained_path.expanduser().resolve(strict=True)
    run_root = run_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    contract_path = PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml"
    contract = load_r1_contract(contract_path)
    validate_external_file_identity(
        checkpoint_path,
        expected_sha256=CHECKPOINT_SHA256,
        expected_bytes=754813672,
        require_digest_filename=True,
        label="R1 checkpoint",
    )
    validate_external_file_identity(
        pretrained_path,
        expected_sha256="845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        expected_bytes=433987358,
        require_digest_filename=False,
        label="Concerto pretrain",
    )
    runtime_config, memory_config = compose_r1_runtime_config(pretrained_path)
    base = _rio_base_dataset(runtime_config, data_root=data_root, horizon=5)
    data_contract = _load_json(
        PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
    )
    unsigned_data = dict(data_contract)
    data_sha256 = str(unsigned_data.pop("content_sha256"))
    if data_sha256 != canonical_json_sha256(unsigned_data):
        raise TaskMemoryPolicyBaselineError("data contract hash differs")
    masters = tuple(
        master
        for master in build_native_episode_masters(
            base,
            reference_by_scene=load_reference_by_scene(metadata_path),
            role_by_reference=_role_by_reference(data_contract),
        )
        if master.role == "development" and len(master.scan_ids) == 5
    )
    if len(masters) != 47 or len({master.reference_id for master in masters}) != 8:
        raise TaskMemoryPolicyBaselineError("development H5 population differs")
    specs = _episode_specs(masters)
    panel = _panel_payload(
        masters, source_commit=source_commit, data_sha256=data_sha256
    )
    panel_path = output_root / "diagnostic_panel.json"
    if panel_path.exists():
        if _load_json(panel_path) != panel:
            raise TaskMemoryPolicyBaselineError("diagnostic panel already differs")
    else:
        _atomic_json(panel_path, panel)
    config_document = {
        "schema_version": "task-memory-policy-baseline-config-v1",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "data_contract_sha256": data_sha256,
        "diagnostic_panel_sha256": panel["content_sha256"],
        "evaluation_seed": EVALUATION_SEED,
        "population": "development_train_holdout_47_masters_canonical",
        "policies": list(POLICIES),
        "reducer": "mean",
        "window": "W2",
        "B4": {
            "K": int(memory_config.capacity),
            "association_threshold": float(memory_config.association_threshold),
            "class_weight": float(memory_config.class_weight),
            "update_rate": float(memory_config.update_rate),
            "confidence_threshold": float(memory_config.confidence_threshold),
            "mask_threshold": float(memory_config.mask_threshold),
            "minimum_mask_support": int(memory_config.minimum_mask_support),
        },
        "lag1_iou_threshold": 0.5,
        "augmentation": "identity-v1",
    }
    provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": canonical_json_sha256(config_document),
        "dataset_sha256": canonical_json_sha256(
            [
                {
                    "reference_id": spec.reference_id,
                    "sequence_id": spec.source_sequence_id,
                    "scan_ids": list(spec.scan_ids),
                }
                for spec in specs
            ]
        ),
    }
    cache_root = run_root / "baseline" / "cache" / provenance["config_sha256"][:16]
    class_mapper = build_rio_class_mapper(base)

    produced = 0
    reused = 0
    if mode in {"produce", "all"}:
        if not torch.cuda.is_available():
            raise TaskMemoryPolicyBaselineError("CUDA is required for R1 inference")
        device = torch.device(device_name)
        if (
            device.type != "cuda"
            or device.index is None
            or device.index >= torch.cuda.device_count()
        ):
            raise TaskMemoryPolicyBaselineError("requested CUDA device is unavailable")
        system = _load_r1_system(runtime_config, checkpoint_path, device, contract)
        memory = PersistentMemory(
            capacity=int(memory_config.capacity),
            class_weight=float(memory_config.class_weight),
            association_threshold=float(memory_config.association_threshold),
            update_rate=float(memory_config.update_rate),
            max_update_rate=float(memory_config.max_update_rate),
        ).to(device)
        observation_settings = {
            "background_class": int(memory_config.background_class),
            "confidence_threshold": float(memory_config.confidence_threshold),
            "mask_threshold": float(memory_config.mask_threshold),
            "minimum_mask_support": int(memory_config.minimum_mask_support),
        }
        stage_collator = hydra.utils.instantiate(
            runtime_config.data.validation_collation
        )
        collator = TaskMemoryEpisodeCollator(stage_collator)
        selected = [
            (index, spec)
            for index, spec in enumerate(specs)
            if index % shard_count == shard_index
        ]
        with deterministic_inference_runtime(EVALUATION_SEED, device):
            for item_index, (global_index, spec) in enumerate(selected):
                cache_id = _cache_identity(provenance, spec)
                path = cache_root / f"{cache_id}.pt"
                if (
                    path.is_file()
                    and path.with_suffix(path.suffix + ".sha256").is_file()
                ):
                    _load_cache(path, provenance=provenance, spec=spec)
                    reused += 1
                else:
                    episode_dataset = TaskMemoryEpisodeDataset(
                        base, (spec,), apply_augmentation=False
                    )
                    with _frozen_inference_seed(EVALUATION_SEED, device):
                        cache = _produce_episode(
                            episode=episode_dataset[0],
                            collator=collator,
                            system=system,
                            memory=memory,
                            observation_settings=observation_settings,
                            class_mapper=class_mapper,
                            device=device,
                            provenance=provenance,
                        )
                    _validate_cache(cache, provenance=provenance, spec=spec)
                    _save_cache(path, cache)
                    produced += 1
                print(
                    json.dumps(
                        {
                            "global_index": global_index,
                            "produced": produced,
                            "reused": reused,
                            "sequence_id": spec.source_sequence_id,
                            "shard": f"{shard_index}/{shard_count}",
                            "shard_progress": f"{item_index + 1}/{len(selected)}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        del system
        torch.cuda.empty_cache()

    manifest: dict[str, object] = {
        "mode": mode,
        "produced": produced,
        "reused": reused,
        "source_commit": source_commit,
        "config_sha256": provenance["config_sha256"],
        "status": "PRODUCED",
    }
    if mode in {"analyze", "all"}:
        if mode == "all" and shard_count != 1:
            raise TaskMemoryPolicyBaselineError(
                "all mode requires one shard; use produce shards then analyze"
            )
        manifest = _analyze(
            specs=specs,
            cache_root=cache_root,
            provenance=provenance,
            panel=panel,
            class_mapper=class_mapper,
            output_root=output_root,
        )
    return manifest


def _parser() -> argparse.ArgumentParser:
    resolver = _load_json(
        PROJECT_ROOT / "artifacts/task_memory_retention_v2/external_assets.local.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("produce", "analyze", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--data-root", type=Path, default=resolver["external:data_root"]
    )
    parser.add_argument(
        "--rio-metadata", type=Path, default=resolver["external:rio_metadata"]
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=resolver["external:r1_checkpoint"]
    )
    parser.add_argument(
        "--pretrained", type=Path, default=resolver["external:concerto_pretrained"]
    )
    parser.add_argument("--run-root", type=Path, default=resolver["external:run_root"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task_memory_retention_v2/baseline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    manifest = run_baseline(
        mode=args.mode,
        device_name=args.device,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        data_root=args.data_root,
        metadata_path=args.rio_metadata,
        checkpoint_path=args.checkpoint,
        pretrained_path=args.pretrained,
        run_root=args.run_root,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "elapsed_seconds": time.time() - started,
                "result": manifest,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TaskMemoryPolicyBaselineError",
    "classify_gap_event",
    "run_baseline",
    "select_diagnostic_masters",
]
