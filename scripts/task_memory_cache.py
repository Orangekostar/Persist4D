"""Compact content-addressed cache for TaskMemory evaluation episodes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from datasets.task_memory_episode import StageMeta
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.system_comparison_inference import pack_bool_matrix, unpack_bool_matrix
from scripts.task_memory_contracts import canonical_json_sha256

CACHE_SCHEMA_VERSION = "task-memory-evaluation-cache-v1"
CACHE_LIMIT_BYTES = 40 * 1024**3
_KEY_FIELDS = {
    "checkpoint_sha256",
    "data_contract_sha256",
    "episode_id",
    "evaluation_seed",
    "history_scan_ids",
    "initial_state_sha256",
    "output_policy",
    "population_id",
    "postprocess_sha256",
    "reference_id",
    "resolved_config_sha256",
    "state_contract_sha256",
    "window_mode",
}
_EVENT_FIELDS = {
    "births",
    "matched_births",
    "matched_reactivations",
    "reactivations",
    "rejected_births",
}


class TaskMemoryCacheError(ValueError):
    """Raised when an evaluation cache is not exact or portable."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskMemoryCacheError(f"{name} must be a mapping")
    return value


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TaskMemoryCacheError(f"{name} must be a lowercase SHA256")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskMemoryCacheError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskMemoryCacheError(f"{name} must be a non-negative integer")
    return value


def _cpu_tensor(
    value: object,
    *,
    name: str,
    ndim: int,
    kind: str,
) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise TaskMemoryCacheError(f"{name} must be a rank-{ndim} tensor")
    tensor = value.detach().cpu().contiguous().clone()
    if kind == "bool":
        if tensor.dtype != torch.bool:
            raise TaskMemoryCacheError(f"{name} must use bool dtype")
    elif kind == "integer":
        if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
            raise TaskMemoryCacheError(f"{name} must use integer dtype")
        tensor = tensor.long()
    elif kind == "float":
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise TaskMemoryCacheError(f"{name} must contain finite floats")
        tensor = tensor.float()
    else:
        raise RuntimeError(f"unsupported tensor kind: {kind}")
    return tensor


def _scan_ids(value: object, *, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskMemoryCacheError(f"{name} must be a sequence")
    result = [_nonempty_string(item, name=f"{name} item") for item in value]
    if not result or len(result) > 5 or len(set(result)) != len(result):
        raise TaskMemoryCacheError(f"{name} must contain 1-5 unique scans")
    return result


def build_evaluation_cache_key(
    *,
    population_id: str,
    reference_id: str,
    episode_id: str,
    history_scan_ids: Sequence[str],
    window_mode: str,
    checkpoint_sha256: str,
    resolved_config_sha256: str,
    data_contract_sha256: str,
    state_contract_sha256: str,
    initial_state_sha256: str,
    output_policy: str,
    postprocess_sha256: str,
    evaluation_seed: int,
) -> dict[str, object]:
    key = {
        "checkpoint_sha256": checkpoint_sha256,
        "data_contract_sha256": data_contract_sha256,
        "episode_id": episode_id,
        "evaluation_seed": evaluation_seed,
        "history_scan_ids": list(history_scan_ids),
        "initial_state_sha256": initial_state_sha256,
        "output_policy": output_policy,
        "population_id": population_id,
        "postprocess_sha256": postprocess_sha256,
        "reference_id": reference_id,
        "resolved_config_sha256": resolved_config_sha256,
        "state_contract_sha256": state_contract_sha256,
        "window_mode": window_mode,
    }
    return validate_evaluation_cache_key(key)


def validate_evaluation_cache_key(value: object) -> dict[str, object]:
    key = _mapping(value, name="evaluation cache key")
    if set(key) != _KEY_FIELDS:
        raise TaskMemoryCacheError("evaluation cache key fields differ")
    result = {
        "checkpoint_sha256": _digest(
            key["checkpoint_sha256"], name="checkpoint_sha256"
        ),
        "data_contract_sha256": _digest(
            key["data_contract_sha256"], name="data_contract_sha256"
        ),
        "episode_id": _nonempty_string(key["episode_id"], name="episode_id"),
        "evaluation_seed": _nonnegative_integer(
            key["evaluation_seed"], name="evaluation_seed"
        ),
        "history_scan_ids": _scan_ids(
            key["history_scan_ids"], name="history_scan_ids"
        ),
        "initial_state_sha256": _digest(
            key["initial_state_sha256"], name="initial_state_sha256"
        ),
        "output_policy": _nonempty_string(
            key["output_policy"], name="output_policy"
        ),
        "population_id": _nonempty_string(
            key["population_id"], name="population_id"
        ),
        "postprocess_sha256": _digest(
            key["postprocess_sha256"], name="postprocess_sha256"
        ),
        "reference_id": _nonempty_string(
            key["reference_id"], name="reference_id"
        ),
        "resolved_config_sha256": _digest(
            key["resolved_config_sha256"], name="resolved_config_sha256"
        ),
        "state_contract_sha256": _digest(
            key["state_contract_sha256"], name="state_contract_sha256"
        ),
        "window_mode": _nonempty_string(key["window_mode"], name="window_mode"),
    }
    if result["window_mode"] not in {"local_pair", "full_history"}:
        raise TaskMemoryCacheError("window_mode must be local_pair or full_history")
    if result["output_policy"] not in {"lag1-v1", "commit0-v1"}:
        raise TaskMemoryCacheError("output_policy version is unsupported")
    return result


def cache_key_sha256(key: Mapping[str, object]) -> str:
    return canonical_json_sha256(validate_evaluation_cache_key(key))


def _stage_meta_payload(meta: StageMeta) -> dict[str, object]:
    if not isinstance(meta, StageMeta):
        raise TaskMemoryCacheError("stage_meta must be StageMeta")
    return {
        "absolute_stage_index": meta.absolute_stage_index,
        "augmentation_transform_id": meta.augmentation_transform_id,
        "coordinate_frame_id": meta.coordinate_frame_id,
        "episode_id": meta.episode_id,
        "full_resolution_point2segment": _cpu_tensor(
            meta.full_resolution_point2segment,
            name="full_resolution_point2segment",
            ndim=1,
            kind="integer",
        ),
        "local_stage_ids": _cpu_tensor(
            meta.local_stage_ids, name="local_stage_ids", ndim=1, kind="integer"
        ),
        "original_vertex_ids": [
            _cpu_tensor(value, name="original_vertex_ids", ndim=1, kind="integer")
            for value in meta.original_vertex_ids
        ],
        "point2segment": _cpu_tensor(
            meta.point2segment, name="point2segment", ndim=1, kind="integer"
        ),
        "reference_id": meta.reference_id,
        "scan_ids_in_window": list(meta.scan_ids_in_window),
        "scan_vertex_offsets": _cpu_tensor(
            meta.scan_vertex_offsets,
            name="scan_vertex_offsets",
            ndim=1,
            kind="integer",
        ),
        "segment_stage_ids": _cpu_tensor(
            meta.segment_stage_ids,
            name="segment_stage_ids",
            ndim=1,
            kind="integer",
        ),
        "voxel_inverse": _cpu_tensor(
            meta.voxel_inverse, name="voxel_inverse", ndim=1, kind="integer"
        ),
    }


def stage_meta_from_cache_record(stage: Mapping[str, object]) -> StageMeta:
    value = _mapping(stage.get("stage_meta"), name="cached stage_meta")
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


def _target_payload(value: Mapping[str, object], *, current_points: int) -> dict[str, object]:
    required = {
        "change_label_semantics",
        "change_labels_valid",
        "changes",
        "gt_class_semantics",
        "gt_classes",
        "gt_ids",
        "gt_masks",
    }
    target = _mapping(value, name="evaluation target")
    if set(target) != required:
        raise TaskMemoryCacheError("evaluation target fields differ")
    ids = _cpu_tensor(target["gt_ids"], name="gt_ids", ndim=1, kind="integer")
    classes = _cpu_tensor(
        target["gt_classes"], name="gt_classes", ndim=1, kind="integer"
    )
    masks = _cpu_tensor(target["gt_masks"], name="gt_masks", ndim=2, kind="bool")
    changes = _cpu_tensor(
        target["changes"], name="changes", ndim=1, kind="integer"
    )
    if (
        ids.shape != classes.shape
        or ids.shape != changes.shape
        or masks.shape != (ids.numel(), current_points)
        or ids.unique().numel() != ids.numel()
        or torch.any(ids < 0).item()
        or torch.any(classes < 0).item()
    ):
        raise TaskMemoryCacheError("evaluation target tensors do not align")
    if target["change_labels_valid"] is not False or torch.any(changes != 0).item():
        raise TaskMemoryCacheError("unavailable change labels must remain all-static")
    return {
        "change_label_semantics": _nonempty_string(
            target["change_label_semantics"], name="change_label_semantics"
        ),
        "change_labels_valid": False,
        "changes": changes,
        "gt_class_semantics": _nonempty_string(
            target["gt_class_semantics"], name="gt_class_semantics"
        ),
        "gt_classes": classes,
        "gt_ids": ids,
        "gt_masks": pack_bool_matrix(masks),
    }


def target_from_cache_record(stage: Mapping[str, object]) -> dict[str, object]:
    value = _mapping(stage.get("target"), name="cached target")
    return {
        **{key: item for key, item in value.items() if key != "gt_masks"},
        "gt_masks": unpack_bool_matrix(value["gt_masks"]),
    }


def _event_payload(value: Mapping[str, object]) -> dict[str, int]:
    event = _mapping(value, name="event diagnostics")
    if set(event) != _EVENT_FIELDS:
        raise TaskMemoryCacheError("event diagnostic fields differ")
    result = {
        name: _nonnegative_integer(event[name], name=name)
        for name in sorted(_EVENT_FIELDS)
    }
    if (
        result["matched_births"] > result["births"]
        or result["matched_reactivations"] > result["reactivations"]
    ):
        raise TaskMemoryCacheError("matched event count exceeds predictions")
    return result


def build_stage_cache_record(
    *,
    prediction: OfficialTaskPrediction,
    identity_map: Mapping[int, tuple[int, int]],
    stage_meta: StageMeta,
    target: Mapping[str, object],
    state_before_sha256: str,
    state_after_sha256: str,
    event_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(prediction, OfficialTaskPrediction):
        raise TaskMemoryCacheError("prediction must be OfficialTaskPrediction")
    prediction.validate()
    meta = _stage_meta_payload(stage_meta)
    if prediction.pred_masks.shape[0] != stage_meta.local_stage_ids.numel():
        raise TaskMemoryCacheError("prediction and StageMeta point counts differ")
    identities = []
    for query_id, identity in sorted(identity_map.items()):
        if not isinstance(identity, tuple) or len(identity) != 2:
            raise TaskMemoryCacheError("identity map values must be pairs")
        logical_id, generation = identity
        identities.append(
            {
                "generation": _nonnegative_integer(
                    generation, name="identity generation"
                ),
                "logical_id": _nonnegative_integer(
                    logical_id, name="identity logical_id"
                ),
                "query_id": _nonnegative_integer(query_id, name="identity query_id"),
            }
        )
    current_points = int(stage_meta.original_vertex_ids[-1].numel())
    return {
        "event_diagnostics": _event_payload(event_diagnostics),
        "identity_map": identities,
        "prediction": {
            "latest_stage_index": prediction.latest_stage_index,
            "pred_classes": _cpu_tensor(
                prediction.pred_classes,
                name="pred_classes",
                ndim=1,
                kind="integer",
            ),
            "pred_masks": pack_bool_matrix(prediction.pred_masks.detach().cpu()),
            "pred_scores": _cpu_tensor(
                prediction.pred_scores,
                name="pred_scores",
                ndim=1,
                kind="float",
            ),
            "source_class_ids": _cpu_tensor(
                prediction.source_class_ids,
                name="source_class_ids",
                ndim=1,
                kind="integer",
            ),
            "source_query_ids": _cpu_tensor(
                prediction.source_query_ids,
                name="source_query_ids",
                ndim=1,
                kind="integer",
            ),
            "temporal_stages": _cpu_tensor(
                prediction.temporal_stages,
                name="temporal_stages",
                ndim=1,
                kind="integer",
            ),
        },
        "stage_meta": meta,
        "state_after_sha256": _digest(
            state_after_sha256, name="state_after_sha256"
        ),
        "state_before_sha256": _digest(
            state_before_sha256, name="state_before_sha256"
        ),
        "target": _target_payload(target, current_points=current_points),
    }


def prediction_from_cache_record(stage: Mapping[str, object]) -> OfficialTaskPrediction:
    value = _mapping(stage.get("prediction"), name="cached prediction")
    masks = unpack_bool_matrix(value["pred_masks"])
    temporal_stages = value["temporal_stages"]
    latest_stage = int(value["latest_stage_index"])
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=value["pred_scores"],
        pred_classes=value["pred_classes"],
        source_query_ids=value["source_query_ids"],
        source_class_ids=value["source_class_ids"],
        temporal_stages=temporal_stages,
        latest_stage_index=latest_stage,
        latest_stage_masks=masks[temporal_stages == latest_stage].contiguous(),
    )
    prediction.validate()
    return prediction


def identity_map_from_cache_record(
    stage: Mapping[str, object],
) -> dict[int, tuple[int, int]]:
    values = stage.get("identity_map")
    if not isinstance(values, list):
        raise TaskMemoryCacheError("cached identity map must be a list")
    result = {
        int(value["query_id"]): (
            int(value["logical_id"]),
            int(value["generation"]),
        )
        for value in values
    }
    if len(result) != len(values):
        raise TaskMemoryCacheError("cached identity map contains duplicate queries")
    return result


def _update_digest(hasher: Any, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("ascii") + b"\0")
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        hasher.update(b"\0" + tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value):
            _update_digest(hasher, key)
            _update_digest(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence\0")
        for item in value:
            _update_digest(hasher, item)
    elif value is None or isinstance(value, (bool, int, str)):
        hasher.update(type(value).__name__.encode("ascii") + b"\0")
        hasher.update(json.dumps(value, ensure_ascii=True).encode() + b"\0")
    elif isinstance(value, float) and math.isfinite(value):
        hasher.update(b"float\0" + value.hex().encode("ascii") + b"\0")
    else:
        raise TaskMemoryCacheError(
            f"unsupported cache digest type: {type(value).__name__}"
        )


def _payload_sha256(payload: Mapping[str, object]) -> str:
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    hasher = hashlib.sha256()
    _update_digest(hasher, unsigned)
    return hasher.hexdigest()


def build_episode_cache_payload(
    *, key: Mapping[str, object], stages: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    normalized_key = validate_evaluation_cache_key(key)
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise TaskMemoryCacheError("cache stages must be a sequence")
    records = [dict(stage) for stage in stages]
    history = normalized_key["history_scan_ids"]
    if len(records) != len(history):
        raise TaskMemoryCacheError("cache stages must cover the complete history")
    previous_state = normalized_key["initial_state_sha256"]
    for index, stage in enumerate(records):
        meta = stage_meta_from_cache_record(stage)
        prediction_from_cache_record(stage)
        identity_map_from_cache_record(stage)
        target_from_cache_record(stage)
        _event_payload(stage.get("event_diagnostics"))
        if meta.absolute_stage_index != index:
            raise TaskMemoryCacheError("cache stages must be in causal order")
        expected_scans = (
            history[: index + 1]
            if normalized_key["window_mode"] == "full_history"
            else history[max(0, index - 1) : index + 1]
        )
        if list(meta.scan_ids_in_window) != expected_scans:
            raise TaskMemoryCacheError("cached StageMeta window differs from history")
        if stage.get("state_before_sha256") != previous_state:
            raise TaskMemoryCacheError("cached state trajectory is discontinuous")
        previous_state = _digest(
            stage.get("state_after_sha256"), name="state_after_sha256"
        )
    payload: dict[str, object] = {
        "key": normalized_key,
        "schema_version": CACHE_SCHEMA_VERSION,
        "stages": records,
    }
    payload["content_sha256"] = _payload_sha256(payload)
    return payload


def validate_episode_cache_payload(value: object) -> dict[str, object]:
    payload = _mapping(value, name="evaluation cache payload")
    if set(payload) != {"content_sha256", "key", "schema_version", "stages"}:
        raise TaskMemoryCacheError("evaluation cache payload fields differ")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise TaskMemoryCacheError("evaluation cache schema differs")
    rebuilt = build_episode_cache_payload(key=payload["key"], stages=payload["stages"])
    if rebuilt["content_sha256"] != payload["content_sha256"]:
        raise TaskMemoryCacheError("evaluation cache content SHA256 differs")
    return dict(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_task_memory_cache(
    cache_root: Path,
    payload: Mapping[str, object],
    *,
    max_total_bytes: int = CACHE_LIMIT_BYTES,
) -> dict[str, object]:
    if (
        isinstance(max_total_bytes, bool)
        or not isinstance(max_total_bytes, int)
        or not 0 < max_total_bytes <= CACHE_LIMIT_BYTES
    ):
        raise TaskMemoryCacheError("cache cap must be within the frozen 40 GiB limit")
    validated = validate_episode_cache_payload(payload)
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = f"{cache_key_sha256(validated['key'])}.pt"
    destination = cache_root / filename
    if destination.exists() or destination.is_symlink():
        raise TaskMemoryCacheError("cache destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{filename}.", suffix=".tmp", dir=cache_root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(validated, temporary)
        existing_bytes = sum(
            path.stat().st_size
            for path in cache_root.glob("*.pt")
            if path.is_file() and not path.is_symlink()
        )
        if existing_bytes + temporary.stat().st_size > max_total_bytes:
            raise TaskMemoryCacheError("cache cap would be exceeded")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "content_sha256": validated["content_sha256"],
        "file_bytes": destination.stat().st_size,
        "file_sha256": _file_sha256(destination),
        "filename": filename,
        "key_sha256": cache_key_sha256(validated["key"]),
    }


def load_task_memory_cache(
    path: Path, *, expected_key: Mapping[str, object] | None = None
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise TaskMemoryCacheError("cache file is unavailable or is a symlink")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise TaskMemoryCacheError("cache file cannot be loaded safely") from error
    payload = validate_episode_cache_payload(value)
    key_sha256 = cache_key_sha256(payload["key"])
    if path.name != f"{key_sha256}.pt":
        raise TaskMemoryCacheError("cache filename differs from its exact key")
    if expected_key is not None and payload["key"] != validate_evaluation_cache_key(
        expected_key
    ):
        raise TaskMemoryCacheError("cache key differs from the requested episode")
    return payload


__all__ = [
    "CACHE_LIMIT_BYTES",
    "CACHE_SCHEMA_VERSION",
    "TaskMemoryCacheError",
    "build_episode_cache_payload",
    "build_evaluation_cache_key",
    "build_stage_cache_record",
    "cache_key_sha256",
    "identity_map_from_cache_record",
    "load_task_memory_cache",
    "prediction_from_cache_record",
    "stage_meta_from_cache_record",
    "target_from_cache_record",
    "validate_episode_cache_payload",
    "validate_evaluation_cache_key",
    "write_task_memory_cache",
]
