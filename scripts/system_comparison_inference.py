"""Frozen full-history inference, postprocessing, and local cache contracts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_HISTORY_SCHEMA_VERSION = 1
FULL_HISTORY_MANIFEST_SCHEMA_VERSION = 1
_ORDERS = ("canonical", "reverse", "sha256_seed45")
_KEY_FIELDS = {
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "context_index",
    "context_scan_indices",
    "horizon",
    "history_scan_ids",
    "scan_indices",
    "task_quality",
}
_PROVENANCE_FIELDS = {
    "source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "protocol_sha256",
}
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class FullHistoryCacheError(ValueError):
    """Raised when a full-history request or cache violates its contract."""


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullHistoryCacheError(f"{name} must be a mapping")
    return dict(value)


def _sequence(value: object, *, name: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FullHistoryCacheError(f"{name} must be a sequence")
    return list(value)


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FullHistoryCacheError(f"{name} must be a nonempty string")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FullHistoryCacheError(f"{name} must be a non-negative integer")
    return value


def _finite_tensor(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise FullHistoryCacheError(f"{name} must be a tensor")
    tensor = value.detach().cpu().contiguous().clone()
    if ndim is not None and tensor.ndim != ndim:
        raise FullHistoryCacheError(f"{name} must have rank {ndim}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise FullHistoryCacheError(f"{name} must contain finite values")
    return tensor


def _integer_tensor(value: object, *, name: str, ndim: int = 1) -> Tensor:
    tensor = _finite_tensor(value, name=name, ndim=ndim)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise FullHistoryCacheError(f"{name} must use an integer dtype")
    return tensor.long()


def pack_bool_matrix(value: Tensor) -> dict[str, object]:
    """Pack a rank-2 CPU boolean matrix into a portable bit vector."""

    matrix = _finite_tensor(value, name="boolean matrix", ndim=2)
    if matrix.dtype != torch.bool:
        raise FullHistoryCacheError("boolean matrix must use bool dtype")
    flat = matrix.numpy().reshape(-1)
    packed = np.packbits(flat, bitorder="little")
    return {
        "encoding": "numpy-packbits-little-v1",
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "data": torch.from_numpy(packed.copy()),
    }


def unpack_bool_matrix(value: Mapping[str, object]) -> Tensor:
    packed = _mapping(value, name="packed boolean matrix")
    if set(packed) != {"encoding", "shape", "data"}:
        raise FullHistoryCacheError("packed boolean matrix fields differ")
    if packed["encoding"] != "numpy-packbits-little-v1":
        raise FullHistoryCacheError("packed boolean matrix encoding differs")
    shape = _sequence(packed["shape"], name="packed boolean matrix shape")
    if len(shape) != 2:
        raise FullHistoryCacheError("packed boolean matrix shape must have rank 2")
    rows = _nonnegative_integer(shape[0], name="packed rows")
    columns = _nonnegative_integer(shape[1], name="packed columns")
    data = _finite_tensor(packed["data"], name="packed boolean data", ndim=1)
    if data.dtype != torch.uint8:
        raise FullHistoryCacheError("packed boolean data must use uint8")
    count = rows * columns
    expected_bytes = (count + 7) // 8
    if data.numel() != expected_bytes:
        raise FullHistoryCacheError("packed boolean byte count differs from shape")
    unpacked = np.unpackbits(data.numpy(), bitorder="little", count=count)
    return torch.from_numpy(unpacked.astype(np.bool_, copy=False)).reshape(rows, columns)


def validate_full_history_cache_key(value: Mapping[str, object]) -> dict[str, object]:
    key = _mapping(value, name="full-history cache key")
    if set(key) != _KEY_FIELDS:
        raise FullHistoryCacheError("full-history cache key fields differ")
    master_id = _nonempty_string(
        key["master_sequence_id"], name="master_sequence_id"
    )
    reference_id = _nonempty_string(
        key["reference_scene_id"], name="reference_scene_id"
    )
    order_id = _nonempty_string(key["order_id"], name="order_id")
    if order_id not in _ORDERS:
        raise FullHistoryCacheError("order_id is not preregistered")
    context_index = _nonnegative_integer(key["context_index"], name="context_index")
    context_scan_indices = [
        _nonnegative_integer(item, name="context_scan_indices item")
        for item in _sequence(
            key["context_scan_indices"], name="context_scan_indices"
        )
    ]
    if len(context_scan_indices) != 5 or len(set(context_scan_indices)) != 5:
        raise FullHistoryCacheError("context_scan_indices must contain five unique scans")
    horizon = _nonnegative_integer(key["horizon"], name="horizon")
    if not 1 <= horizon <= 5:
        raise FullHistoryCacheError("horizon must be within T1-T5")
    scan_ids = [
        _nonempty_string(item, name="history_scan_ids item")
        for item in _sequence(key["history_scan_ids"], name="history_scan_ids")
    ]
    scan_indices = [
        _nonnegative_integer(item, name="scan_indices item")
        for item in _sequence(key["scan_indices"], name="scan_indices")
    ]
    if (
        len(scan_ids) != horizon
        or len(scan_indices) != horizon
        or len(set(scan_ids)) != horizon
        or len(set(scan_indices)) != horizon
    ):
        raise FullHistoryCacheError("history and scan indices must match the horizon")
    if not set(scan_indices) <= set(context_scan_indices):
        raise FullHistoryCacheError("history scan indices escape the master context")
    task_quality = key["task_quality"]
    if not isinstance(task_quality, bool) or task_quality is not (horizon >= 2):
        raise FullHistoryCacheError("task_quality flag differs from horizon semantics")
    return {
        "master_sequence_id": master_id,
        "reference_scene_id": reference_id,
        "order_id": order_id,
        "context_index": context_index,
        "context_scan_indices": context_scan_indices,
        "horizon": horizon,
        "history_scan_ids": scan_ids,
        "scan_indices": scan_indices,
        "task_quality": task_quality,
    }


def full_history_cache_keys(
    system_manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    manifest = _mapping(system_manifest, name="system comparison manifest")
    protocol = _mapping(manifest.get("protocol"), name="system protocol")
    if protocol.get("order_variants") != list(_ORDERS):
        raise FullHistoryCacheError("system manifest orders differ")
    if protocol.get("horizons") != [2, 3, 4, 5]:
        raise FullHistoryCacheError("system manifest task horizons differ")
    masters = _sequence(manifest.get("masters"), name="system masters")
    if len(masters) != 43:
        raise FullHistoryCacheError("system manifest must contain 43 masters")
    keys: list[dict[str, object]] = []
    for raw_master in masters:
        master = _mapping(raw_master, name="system master")
        orders = _mapping(master.get("orders"), name="system master orders")
        for order_id in _ORDERS:
            order = _mapping(orders.get(order_id), name="system order")
            visit_order = _sequence(order.get("visit_order"), name="visit_order")
            scan_indices = _sequence(order.get("scan_indices"), name="scan_indices")
            for horizon in range(1, 6):
                keys.append(
                    validate_full_history_cache_key(
                        {
                            "master_sequence_id": master["master_sequence_id"],
                            "reference_scene_id": master["reference_scene_id"],
                            "order_id": order_id,
                            "context_index": master["validation_index"],
                            "context_scan_indices": master["scan_indices"],
                            "horizon": horizon,
                            "history_scan_ids": visit_order[:horizon],
                            "scan_indices": scan_indices[:horizon],
                            "task_quality": horizon >= 2,
                        }
                    )
                )
    identities = {_key_identity(key) for key in keys}
    if len(keys) != 645 or len(identities) != len(keys):
        raise FullHistoryCacheError("full-history key coverage is not exact and unique")
    return keys


def _key_identity(key: Mapping[str, object]) -> str:
    validated = validate_full_history_cache_key(key)
    return json.dumps(validated, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _update_digest(hasher: Any, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("ascii") + b"\0")
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        hasher.update(b"\0")
        hasher.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value):
            if not isinstance(key, str):
                raise FullHistoryCacheError("digest mapping keys must be strings")
            hasher.update(key.encode("utf-8") + b"\0")
            _update_digest(hasher, value[key])
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        hasher.update(b"sequence\0")
        for item in value:
            _update_digest(hasher, item)
        return
    if value is None:
        hasher.update(b"none\0")
        return
    if isinstance(value, bool):
        hasher.update(b"bool\0" + (b"1" if value else b"0"))
        return
    if isinstance(value, int):
        hasher.update(b"int\0" + str(value).encode("ascii") + b"\0")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FullHistoryCacheError("digest values must be finite")
        hasher.update(b"float\0" + value.hex().encode("ascii") + b"\0")
        return
    if isinstance(value, str):
        hasher.update(b"str\0" + value.encode("utf-8") + b"\0")
        return
    raise FullHistoryCacheError(f"unsupported digest value: {type(value).__name__}")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    hasher = hashlib.sha256()
    _update_digest(hasher, payload)
    return hasher.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def _observation_fingerprints(observation: Mapping[str, Tensor]) -> dict[str, str]:
    names = (
        "features",
        "class_prob",
        "confidence",
        "valid",
        "masks",
        "mask_support",
        "local_query_ids",
    )
    fingerprints = {
        name: _tensor_sha256(_finite_tensor(observation[name], name=name))
        for name in names
    }
    hasher = hashlib.sha256()
    for name in names:
        hasher.update(name.encode("ascii") + b"\0")
        hasher.update(fingerprints[name].encode("ascii"))
    fingerprints["combined"] = hasher.hexdigest()
    return fingerprints


@dataclass(frozen=True)
class ProcessedFullHistory:
    task_prediction: dict[str, Tensor]
    identity_prediction: dict[str, Tensor]
    target: dict[str, Tensor]
    raw_observation: dict[str, Tensor]
    observation_fingerprints: dict[str, str]


def _map_classes(classes: Tensor, mapper: Callable[[int], int]) -> Tensor:
    values = []
    for value in classes.detach().cpu().long().tolist():
        mapped = mapper(int(value))
        if isinstance(mapped, bool) or not isinstance(mapped, int):
            raise FullHistoryCacheError("class mapper must return integer labels")
        values.append(mapped)
    return torch.tensor(values, dtype=torch.long)


def postprocess_full_history_output(
    *,
    system: object,
    output: Mapping[str, object],
    target_low_resolution: Mapping[str, object],
    target_full_resolution: Mapping[str, object],
    data: object,
    horizon: int,
    class_mapper: Callable[[int], int],
    background_class: int,
    confidence_threshold: float,
    mask_threshold: float,
    minimum_mask_support: int,
) -> ProcessedFullHistory:
    """Separate official task output from raw-query deployment identities."""

    if not 1 <= horizon <= 5:
        raise FullHistoryCacheError("horizon must be within T1-T5")
    outputs = _mapping(output, name="ReScene output")
    logits = _finite_tensor(outputs.get("pred_logits"), name="pred_logits", ndim=3)
    query_features = _finite_tensor(
        outputs.get("query_features"), name="query_features", ndim=3
    )
    raw_masks = outputs.get("pred_masks")
    if (
        logits.shape[0] != 1
        or query_features.shape[0] != 1
        or logits.shape[:2] != query_features.shape[:2]
        or isinstance(raw_masks, (str, bytes))
        or not isinstance(raw_masks, Sequence)
        or len(raw_masks) != 1
    ):
        raise FullHistoryCacheError("ReScene output must contain one aligned batch")
    segment_logits = _finite_tensor(raw_masks[0], name="pred_masks[0]", ndim=2)
    query_count = int(logits.shape[1])
    if segment_logits.shape[1] != query_count:
        raise FullHistoryCacheError("query and mask dimensions differ")
    if not 0 <= background_class < logits.shape[2]:
        raise FullHistoryCacheError("background class is outside logits")
    if (
        not 0.0 <= confidence_threshold <= 1.0
        or not 0.0 <= mask_threshold <= 1.0
        or minimum_mask_support <= 0
    ):
        raise FullHistoryCacheError("identity observation thresholds are invalid")

    get_predictions = getattr(system, "_get_predictions", None)
    get_batch_masks = getattr(system, "_get_batch_masks", None)
    get_mask_and_scores = getattr(system, "_get_mask_and_scores", None)
    get_full_res_mask = getattr(system, "_get_full_res_mask", None)
    filter_predictions = getattr(system, "_filter_and_sort_predictions", None)
    if not all(
        callable(value)
        for value in (
            get_predictions,
            get_batch_masks,
            get_mask_and_scores,
            get_full_res_mask,
            filter_predictions,
        )
    ):
        raise FullHistoryCacheError("system lacks official ReScene postprocessing")

    prediction = get_predictions(outputs)
    decoder_id = int(getattr(system, "decoder_id", -1))
    selected = prediction[decoder_id]
    low_masks = get_batch_masks(prediction, 0, [target_low_resolution])
    official_scores, official_low_masks, official_classes, official_heatmap = (
        get_mask_and_scores(
            selected["pred_logits"][0].detach().cpu(),
            low_masks,
            selected["pred_logits"][0].shape[0],
            logits.shape[2] - 1,
        )
    )
    inverse_maps = getattr(data, "inverse_maps", None)
    if not isinstance(inverse_maps, Sequence) or len(inverse_maps) != 1:
        raise FullHistoryCacheError("collated data must contain one inverse map")
    full_point2segment = _integer_tensor(
        target_full_resolution.get("point2segment"),
        name="target_full.point2segment",
    )
    official_masks = get_full_res_mask(
        official_low_masks,
        inverse_maps[0],
        full_point2segment,
    )
    official_heatmap = get_full_res_mask(
        official_heatmap,
        inverse_maps[0],
        full_point2segment,
        is_heatmap=True,
    )
    sorted_classes, sorted_masks, sorted_scores, _ = filter_predictions(
        np.asarray(official_masks),
        official_scores,
        official_classes,
        np.asarray(official_heatmap),
    )
    task_masks = torch.as_tensor(sorted_masks).bool().cpu().contiguous()
    task_scores = torch.as_tensor(sorted_scores).float().cpu().contiguous()
    task_classes = _map_classes(torch.as_tensor(sorted_classes), class_mapper)
    if (
        task_masks.ndim != 2
        or task_scores.ndim != 1
        or task_classes.ndim != 1
        or task_masks.shape[1] != task_scores.shape[0]
        or task_scores.shape != task_classes.shape
    ):
        raise FullHistoryCacheError("official task prediction tensors do not align")

    low_point2segment = _integer_tensor(
        target_low_resolution.get("point2segment"),
        name="target_low.point2segment",
    )
    train_on_segments = bool(
        getattr(getattr(system, "model", None), "train_on_segments", False)
    )
    thresholded_segment_masks = (
        segment_logits.sigmoid() >= mask_threshold
    ).float()
    query_low_masks = (
        thresholded_segment_masks[low_point2segment]
        if train_on_segments
        else thresholded_segment_masks
    )
    all_query_masks = torch.as_tensor(
        get_full_res_mask(
            query_low_masks,
            inverse_maps[0],
            full_point2segment,
        )
    ).bool().cpu().contiguous()
    temporal_stages = _integer_tensor(
        target_full_resolution.get("temporal_stages"),
        name="target_full.temporal_stages",
    )
    if all_query_masks.shape != (temporal_stages.numel(), query_count):
        raise FullHistoryCacheError("raw query masks do not cover full target points")
    if set(temporal_stages.tolist()) != set(range(horizon)):
        raise FullHistoryCacheError("full target temporal stages differ from prefix")
    latest_selector = temporal_stages == horizon - 1
    current_query_masks = all_query_masks[latest_selector]

    class_prob = logits[0].softmax(dim=-1)
    foreground = class_prob.clone()
    foreground[:, background_class] = -torch.inf
    confidence, model_classes = foreground.max(dim=-1)
    mask_support = current_query_masks.sum(dim=0, dtype=torch.long)
    valid = (confidence >= confidence_threshold) & (
        mask_support >= minimum_mask_support
    )
    issued_ids = torch.arange(query_count, dtype=torch.long)[valid]
    identity_prediction = {
        "pred_masks": current_query_masks[:, valid].contiguous(),
        "pred_scores": confidence[valid].float().cpu().contiguous(),
        "pred_classes": _map_classes(model_classes[valid], class_mapper),
        "issued_ids": issued_ids,
        "all_query_masks": current_query_masks,
    }

    target_masks = _finite_tensor(
        target_full_resolution.get("masks"), name="target_full.masks", ndim=2
    ).bool()
    target_labels = _map_classes(
        _integer_tensor(
            target_full_resolution.get("labels"), name="target_full.labels"
        ),
        class_mapper,
    )
    target_ids = _integer_tensor(
        target_full_resolution.get("ids"), name="target_full.ids"
    )
    if (
        target_masks.shape[0] != target_labels.shape[0]
        or target_labels.shape != target_ids.shape
        or target_masks.shape[1] != temporal_stages.numel()
        or task_masks.shape[0] != temporal_stages.numel()
    ):
        raise FullHistoryCacheError("full prediction and target tensors do not align")
    target = {
        "masks": target_masks,
        "labels": target_labels,
        "ids": target_ids,
        "changes": torch.zeros_like(target_ids),
        "temporal_stages": temporal_stages,
    }
    raw_observation = {
        "features": query_features[0],
        "class_prob": class_prob,
        "confidence": confidence,
        "valid": valid.cpu(),
        "masks": current_query_masks.transpose(0, 1).contiguous(),
        "mask_support": mask_support,
        "local_query_ids": torch.arange(query_count, dtype=torch.long),
    }
    return ProcessedFullHistory(
        task_prediction={
            "pred_masks": task_masks,
            "pred_scores": task_scores,
            "pred_classes": task_classes,
        },
        identity_prediction=identity_prediction,
        target=target,
        raw_observation=raw_observation,
        observation_fingerprints=_observation_fingerprints(raw_observation),
    )


def _validate_provenance(value: Mapping[str, object]) -> dict[str, str]:
    provenance = _mapping(value, name="full-history provenance")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise FullHistoryCacheError("full-history provenance fields differ")
    result: dict[str, str] = {}
    for name in sorted(_PROVENANCE_FIELDS):
        raw = provenance[name]
        pattern = _HEX40 if name == "source_commit" else _HEX64
        if not isinstance(raw, str) or pattern.fullmatch(raw) is None:
            raise FullHistoryCacheError(f"provenance {name} is invalid")
        result[name] = raw
    return result


def _validate_input_stats(value: Mapping[str, object], *, horizon: int) -> dict[str, object]:
    stats = _mapping(value, name="input_stats")
    expected = {
        "scan_count",
        "full_point_count",
        "low_resolution_point_count",
        "segment_count",
        "model_input_bytes",
        "scan_point_counts",
    }
    if set(stats) != expected:
        raise FullHistoryCacheError("input_stats fields differ")
    result = {
        name: _nonnegative_integer(stats[name], name=f"input_stats.{name}")
        for name in expected - {"scan_point_counts"}
    }
    counts = [
        _nonnegative_integer(item, name="scan_point_counts item")
        for item in _sequence(stats["scan_point_counts"], name="scan_point_counts")
    ]
    if (
        result["scan_count"] != horizon
        or len(counts) != horizon
        or any(count <= 0 for count in counts)
        or sum(counts) != result["full_point_count"]
        or result["low_resolution_point_count"] <= 0
        or result["segment_count"] <= 0
    ):
        raise FullHistoryCacheError("input_stats do not match the causal prefix")
    return {**result, "scan_point_counts": counts}


def build_full_history_payload(
    *,
    key: Mapping[str, object],
    provenance: Mapping[str, object],
    processed: ProcessedFullHistory,
    input_stats: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(processed, ProcessedFullHistory):
        raise FullHistoryCacheError("processed output has an invalid type")
    normalized_key = validate_full_history_cache_key(key)
    normalized_provenance = _validate_provenance(provenance)
    stats = _validate_input_stats(input_stats, horizon=int(normalized_key["horizon"]))
    identity = processed.identity_prediction
    payload: dict[str, object] = {
        "schema_version": FULL_HISTORY_SCHEMA_VERSION,
        "key": normalized_key,
        "provenance": normalized_provenance,
        "task_prediction": {
            "pred_masks": pack_bool_matrix(processed.task_prediction["pred_masks"]),
            "pred_scores": _finite_tensor(
                processed.task_prediction["pred_scores"],
                name="task pred_scores",
                ndim=1,
            ),
            "pred_classes": _integer_tensor(
                processed.task_prediction["pred_classes"], name="task pred_classes"
            ),
        },
        "identity_prediction": {
            "pred_masks": pack_bool_matrix(identity["pred_masks"]),
            "pred_scores": _finite_tensor(
                identity["pred_scores"], name="identity pred_scores", ndim=1
            ),
            "pred_classes": _integer_tensor(
                identity["pred_classes"], name="identity pred_classes"
            ),
            "issued_ids": _integer_tensor(
                identity["issued_ids"], name="identity issued_ids"
            ),
        },
        "target": {
            "masks": pack_bool_matrix(processed.target["masks"]),
            "labels": _integer_tensor(processed.target["labels"], name="target labels"),
            "ids": _integer_tensor(processed.target["ids"], name="target ids"),
            "changes": _integer_tensor(
                processed.target["changes"], name="target changes"
            ),
            "temporal_stages": _integer_tensor(
                processed.target["temporal_stages"], name="target temporal_stages"
            ),
        },
        "input_stats": stats,
        "observation_fingerprints": dict(processed.observation_fingerprints),
    }
    payload["content_sha256"] = _content_sha256(payload)
    return validate_full_history_payload(payload)


def validate_full_history_payload(
    value: Mapping[str, object],
    *,
    expected_key: Mapping[str, object] | None = None,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = _mapping(value, name="full-history payload")
    expected_fields = {
        "schema_version",
        "key",
        "provenance",
        "task_prediction",
        "identity_prediction",
        "target",
        "input_stats",
        "observation_fingerprints",
        "content_sha256",
    }
    if set(payload) != expected_fields:
        raise FullHistoryCacheError("full-history payload fields differ")
    if payload["schema_version"] != FULL_HISTORY_SCHEMA_VERSION:
        raise FullHistoryCacheError("full-history payload schema differs")
    key = validate_full_history_cache_key(payload["key"])
    provenance = _validate_provenance(payload["provenance"])
    if expected_key is not None and key != validate_full_history_cache_key(expected_key):
        raise FullHistoryCacheError("full-history payload key differs")
    if (
        expected_provenance is not None
        and provenance != _validate_provenance(expected_provenance)
    ):
        raise FullHistoryCacheError("full-history payload provenance differs")
    stats = _validate_input_stats(payload["input_stats"], horizon=int(key["horizon"]))

    task = _mapping(payload["task_prediction"], name="task_prediction")
    identity = _mapping(payload["identity_prediction"], name="identity_prediction")
    target = _mapping(payload["target"], name="target")
    if set(task) != {"pred_masks", "pred_scores", "pred_classes"}:
        raise FullHistoryCacheError("task_prediction fields differ")
    if set(identity) != {"pred_masks", "pred_scores", "pred_classes", "issued_ids"}:
        raise FullHistoryCacheError("identity_prediction fields differ")
    if set(target) != {"masks", "labels", "ids", "changes", "temporal_stages"}:
        raise FullHistoryCacheError("target fields differ")

    task_masks = unpack_bool_matrix(task["pred_masks"])
    task_scores = _finite_tensor(task["pred_scores"], name="task scores", ndim=1)
    task_classes = _integer_tensor(task["pred_classes"], name="task classes")
    identity_masks = unpack_bool_matrix(identity["pred_masks"])
    identity_scores = _finite_tensor(
        identity["pred_scores"], name="identity scores", ndim=1
    )
    identity_classes = _integer_tensor(
        identity["pred_classes"], name="identity classes"
    )
    issued_ids = _integer_tensor(identity["issued_ids"], name="issued IDs")
    target_masks = unpack_bool_matrix(target["masks"])
    labels = _integer_tensor(target["labels"], name="target labels")
    ids = _integer_tensor(target["ids"], name="target IDs")
    changes = _integer_tensor(target["changes"], name="target changes")
    stages = _integer_tensor(target["temporal_stages"], name="target temporal stages")
    horizon = int(key["horizon"])
    if set(stages.tolist()) != set(range(horizon)) or int(stages.max().item()) >= horizon:
        raise FullHistoryCacheError("target temporal stages contain future information")
    if (
        task_masks.shape[0] != stats["full_point_count"]
        or task_masks.shape[1] != task_scores.numel()
        or task_scores.shape != task_classes.shape
    ):
        raise FullHistoryCacheError("task prediction tensors do not align")
    current_points = int((stages == int(key["horizon"]) - 1).sum().item())
    if (
        identity_masks.shape[0] != current_points
        or identity_masks.shape[1] != identity_scores.numel()
        or identity_scores.shape != identity_classes.shape
        or identity_classes.shape != issued_ids.shape
        or len(set(issued_ids.tolist())) != issued_ids.numel()
    ):
        raise FullHistoryCacheError("identity prediction tensors do not align")
    if (
        target_masks.shape != (labels.numel(), stages.numel())
        or labels.shape != ids.shape
        or ids.shape != changes.shape
        or stages.numel() != stats["full_point_count"]
        or torch.any(changes != 0).item()
    ):
        raise FullHistoryCacheError("target tensors do not align")
    fingerprints = _mapping(
        payload["observation_fingerprints"], name="observation_fingerprints"
    )
    expected_fingerprints = {
        "features",
        "class_prob",
        "confidence",
        "valid",
        "masks",
        "mask_support",
        "local_query_ids",
        "combined",
    }
    if set(fingerprints) != expected_fingerprints or any(
        not isinstance(digest, str) or _HEX64.fullmatch(digest) is None
        for digest in fingerprints.values()
    ):
        raise FullHistoryCacheError("observation fingerprints are invalid")
    if payload["content_sha256"] != _content_sha256(payload):
        raise FullHistoryCacheError("full-history payload content digest differs")
    return payload


def full_history_prediction_fingerprint(
    value: Mapping[str, object],
) -> str:
    """Hash only masks, classes, IDs, and scores used by the smoke gate."""

    payload = validate_full_history_payload(value)
    return _content_sha256(
        {
            "task_prediction": payload["task_prediction"],
            "identity_prediction": payload["identity_prediction"],
        }
    )


def assert_t2_observation_regression(
    full_history_payload: Mapping[str, object],
    local_payload: Mapping[str, object],
) -> None:
    full = validate_full_history_payload(full_history_payload)
    full_key = full["key"]
    if full_key["horizon"] != 2:
        raise FullHistoryCacheError("T2 regression requires a full-history T2 payload")
    local = _mapping(local_payload, name="local T2 payload")
    local_key = _mapping(local.get("key"), name="local T2 key")
    if (
        local_key.get("stage_index") != 1
        or local_key.get("history_scan_ids") != full_key["history_scan_ids"]
        or local_key.get("local_window_scan_ids") != full_key["history_scan_ids"]
    ):
        raise FullHistoryCacheError("T2 local and full-history requests differ")
    observation = _mapping(local.get("observation"), name="local observation")
    try:
        local_fingerprints = _observation_fingerprints(observation)
    except KeyError as error:
        raise FullHistoryCacheError("local T2 observation is incomplete") from error
    if local_fingerprints != full["observation_fingerprints"]:
        raise FullHistoryCacheError("T2 observation fingerprint differs")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_filename(key: Mapping[str, object]) -> str:
    digest = hashlib.sha256(_key_identity(key).encode("utf-8")).hexdigest()
    return f"{digest}.pt"


def write_full_history_cache_entry(
    cache_directory: str | Path,
    payload: Mapping[str, object],
) -> dict[str, object]:
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = validate_full_history_cache_key(payload.get("key"))
    supplied_digest = _content_sha256(payload)
    output = directory / _entry_filename(key)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("cache key path is not a regular file")
        existing = torch.load(output, map_location="cpu", weights_only=False)
        existing = validate_full_history_payload(existing, expected_key=key)
        if existing["content_sha256"] != supplied_digest:
            raise FileExistsError("cache key already contains different content")
        return {
            "key": key,
            "filename": output.name,
            "sha256": _file_sha256(output),
            "byte_size": output.stat().st_size,
            "content_sha256": existing["content_sha256"],
        }
    validated = validate_full_history_payload(payload, expected_key=key)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(validated, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "key": key,
        "filename": output.name,
        "sha256": _file_sha256(output),
        "byte_size": output.stat().st_size,
        "content_sha256": validated["content_sha256"],
    }


def load_full_history_cache_entry(
    cache_directory: str | Path,
    entry: Mapping[str, object],
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    record = _mapping(entry, name="cache entry")
    expected_fields = {"key", "filename", "sha256", "byte_size", "content_sha256"}
    if set(record) != expected_fields:
        raise FullHistoryCacheError("cache entry fields differ")
    key = validate_full_history_cache_key(record["key"])
    filename = record["filename"]
    if filename != _entry_filename(key):
        raise FullHistoryCacheError("cache entry filename differs from logical key")
    path = Path(cache_directory) / filename
    if path.is_symlink() or not path.is_file():
        raise FullHistoryCacheError("cache entry file is unavailable")
    if path.stat().st_size != record["byte_size"] or _file_sha256(path) != record["sha256"]:
        raise FullHistoryCacheError("cache entry file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validated = validate_full_history_payload(
        payload,
        expected_key=key,
        expected_provenance=expected_provenance,
    )
    if validated["content_sha256"] != record["content_sha256"]:
        raise FullHistoryCacheError("cache entry content digest differs")
    return validated


def _validate_cache_entry_record(value: Mapping[str, object]) -> dict[str, object]:
    record = _mapping(value, name="cache entry")
    expected = {"key", "filename", "sha256", "byte_size", "content_sha256"}
    if set(record) != expected:
        raise FullHistoryCacheError("cache entry fields differ")
    key = validate_full_history_cache_key(record["key"])
    filename = _nonempty_string(record["filename"], name="cache entry filename")
    if filename != _entry_filename(key):
        raise FullHistoryCacheError("cache entry filename differs from logical key")
    sha256 = _nonempty_string(record["sha256"], name="cache entry sha256")
    content_sha256 = _nonempty_string(
        record["content_sha256"], name="cache entry content_sha256"
    )
    if _HEX64.fullmatch(sha256) is None or _HEX64.fullmatch(content_sha256) is None:
        raise FullHistoryCacheError("cache entry digest is invalid")
    byte_size = _nonnegative_integer(record["byte_size"], name="cache entry byte_size")
    return {
        "key": key,
        "filename": filename,
        "sha256": sha256,
        "byte_size": byte_size,
        "content_sha256": content_sha256,
    }


def _normalized_cache_entries(
    entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise FullHistoryCacheError("cache entries must be a sequence")
    normalized = [_validate_cache_entry_record(entry) for entry in entries]
    identities = [_key_identity(entry["key"]) for entry in normalized]
    if len(set(identities)) != len(identities):
        raise FullHistoryCacheError("cache entries contain duplicate logical keys")
    normalized.sort(key=lambda entry: _key_identity(entry["key"]))
    return normalized


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def build_full_history_cache_manifest(
    entries: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    """Build a fail-closed manifest for all 645 exact causal prefixes."""

    normalized = _normalized_cache_entries(entries)
    expected = [validate_full_history_cache_key(key) for key in expected_keys]
    expected_identities = [_key_identity(key) for key in expected]
    if len(set(expected_identities)) != len(expected_identities):
        raise FullHistoryCacheError("expected cache coverage contains duplicates")
    actual_identities = [_key_identity(entry["key"]) for entry in normalized]
    if set(actual_identities) != set(expected_identities):
        raise FullHistoryCacheError("cache manifest does not have exact coverage")
    provenance = _validate_provenance(expected_provenance)
    entries_sha256 = hashlib.sha256(
        _canonical_json_bytes({"entries": normalized})
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": FULL_HISTORY_MANIFEST_SCHEMA_VERSION,
        "status": "pass",
        "provenance": provenance,
        "entry_count": len(normalized),
        "entries_sha256": entries_sha256,
        "entries": normalized,
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    if cache_directory is not None:
        directory = Path(cache_directory)
        if directory.is_symlink() or not directory.is_dir():
            raise FullHistoryCacheError("cache directory must be a regular directory")
        expected_files = {entry["filename"] for entry in normalized}
        actual_files = {path.name for path in directory.iterdir()}
        if actual_files != expected_files:
            raise FullHistoryCacheError("cache directory coverage differs from manifest")
        for entry in normalized:
            load_full_history_cache_entry(
                directory,
                entry,
                expected_provenance=provenance,
            )
    return manifest


def discover_full_history_cache_entries(
    cache_directory: str | Path,
    *,
    expected_provenance: Mapping[str, object],
) -> list[dict[str, object]]:
    directory = Path(cache_directory)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise FullHistoryCacheError("cache directory must be a regular directory")
    directory.mkdir(parents=True, exist_ok=True)
    provenance = _validate_provenance(expected_provenance)
    entries: list[dict[str, object]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
            raise FullHistoryCacheError(
                f"cache directory contains unexpected path: {path.name}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validated = validate_full_history_payload(
            payload,
            expected_provenance=provenance,
        )
        record = {
            "key": validated["key"],
            "filename": path.name,
            "sha256": _file_sha256(path),
            "byte_size": path.stat().st_size,
            "content_sha256": validated["content_sha256"],
        }
        entries.append(_validate_cache_entry_record(record))
    return _normalized_cache_entries(entries)


def write_full_history_cache_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    cache_directory: str | Path,
) -> None:
    expected = build_full_history_cache_manifest(
        _mapping(manifest, name="full-history cache manifest").get("entries", []),
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
        cache_directory=cache_directory,
    )
    if dict(manifest) != expected:
        raise FullHistoryCacheError("full-history cache manifest fields differ")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(expected)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("cache manifest is not a regular file")
        if output.read_bytes() == payload:
            return
        raise FileExistsError("cache manifest already contains different content")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def deterministic_inference_runtime(seed: int, device: torch.device):
    """Restore all deterministic runtime state after a frozen inference group."""

    if device.type != "cuda" or device.index is None:
        raise FullHistoryCacheError("deterministic inference requires an indexed CUDA device")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        random.seed(seed)
        np.random.seed(seed)
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _model_input_bytes(data: object) -> int:
    ignored = {
        "target_full",
        "inverse_maps",
        "original_colors",
        "original_normals",
        "original_coordinates",
    }
    if isinstance(data, Mapping):
        values = ((key, value) for key, value in data.items())
    else:
        keys = getattr(data, "keys", lambda: ())()
        values = ((key, data[key]) for key in keys)
    total = 0
    for key, value in values:
        if key in ignored:
            continue
        if isinstance(value, Tensor):
            total += value.numel() * value.element_size()
    return total


@dataclass
class FullHistoryPredictionProducer:
    dataset: object
    collate: Callable[[list[object]], tuple[object, object, object]]
    system: object
    device: torch.device
    provenance: Mapping[str, object]
    class_mapper: Callable[[int], int]
    move_data: Callable[[object, torch.device], object]
    move_targets: Callable[[object, torch.device], object]
    background_class: int = 18
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    minimum_mask_support: int = 1
    seed: int = 45

    def __call__(self, logical_key: Mapping[str, object]) -> dict[str, object]:
        key = validate_full_history_cache_key(logical_key)
        names = _field(self.dataset, "sequence_names")
        indices = _field(self.dataset, "sequence_indices")
        context_index = int(key["context_index"])
        if (
            isinstance(names, (str, bytes))
            or not isinstance(names, Sequence)
            or context_index >= len(names)
            or names[context_index] != key["master_sequence_id"]
            or isinstance(indices, (str, bytes))
            or not isinstance(indices, Sequence)
            or tuple(int(item) for item in indices[context_index])
            != tuple(key["context_scan_indices"])
        ):
            raise FullHistoryCacheError("dataset context differs from cache key")
        from scripts.evaluate_persist4d_p6a import _frozen_inference_seed

        with _frozen_inference_seed(self.seed, self.device):
            sample = self.dataset.load_scan_indices(
                context_index,
                tuple(key["scan_indices"]),
                change_file=None,
            )
            data, targets, collated_names = self.collate([sample])
            if (
                not isinstance(targets, Sequence)
                or len(targets) != 1
                or list(collated_names) != [key["master_sequence_id"]]
            ):
                raise FullHistoryCacheError("collator changed full-history identity")
            target_full_values = _field(data, "target_full")
            if (
                isinstance(target_full_values, (str, bytes))
                or not isinstance(target_full_values, Sequence)
                or len(target_full_values) != 1
                or not isinstance(target_full_values[0], Mapping)
            ):
                raise FullHistoryCacheError("collated data lacks one full target")
            target_full = target_full_values[0]
            full_stages = _integer_tensor(
                target_full.get("temporal_stages"), name="full temporal stages"
            )
            scan_counts = [
                int((full_stages == stage).sum().item())
                for stage in range(int(key["horizon"]))
            ]
            model_bytes = _model_input_bytes(data)
            data = self.move_data(data, self.device)
            targets = self.move_targets(targets, self.device)
            target_low = targets[0]
            if not isinstance(target_low, Mapping):
                raise FullHistoryCacheError("collated low-resolution target is invalid")
            raw_coordinates = self.system._process_raw_coordinates(data)
            with torch.inference_mode():
                output = self.system(
                    data,
                    point2segment=[target_low["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
            processed = postprocess_full_history_output(
                system=self.system,
                output=output,
                target_low_resolution=target_low,
                target_full_resolution=target_full,
                data=data,
                horizon=int(key["horizon"]),
                class_mapper=self.class_mapper,
                background_class=self.background_class,
                confidence_threshold=self.confidence_threshold,
                mask_threshold=self.mask_threshold,
                minimum_mask_support=self.minimum_mask_support,
            )
            point2segment = _integer_tensor(
                target_low.get("point2segment"), name="low point2segment"
            )
            input_stats = {
                "scan_count": int(key["horizon"]),
                "full_point_count": int(full_stages.numel()),
                "low_resolution_point_count": int(
                    _integer_tensor(
                        target_low.get("temporal_stages"),
                        name="low temporal stages",
                    ).numel()
                ),
                "segment_count": int(point2segment.max().item()) + 1,
                "model_input_bytes": model_bytes,
                "scan_point_counts": scan_counts,
            }
        return build_full_history_payload(
            key=key,
            provenance=self.provenance,
            processed=processed,
            input_stats=input_stats,
        )


__all__ = [
    "FullHistoryCacheError",
    "FullHistoryPredictionProducer",
    "ProcessedFullHistory",
    "assert_t2_observation_regression",
    "build_full_history_cache_manifest",
    "build_full_history_payload",
    "deterministic_inference_runtime",
    "discover_full_history_cache_entries",
    "full_history_cache_keys",
    "full_history_prediction_fingerprint",
    "load_full_history_cache_entry",
    "pack_bool_matrix",
    "postprocess_full_history_output",
    "unpack_bool_matrix",
    "validate_full_history_cache_key",
    "validate_full_history_payload",
    "write_full_history_cache_entry",
    "write_full_history_cache_manifest",
]
