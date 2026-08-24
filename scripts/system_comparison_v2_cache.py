"""Versioned official-task sidecars for System Comparison V2."""

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

from scripts.p6a_cache import validate_cache_payload
from scripts.rescene_task_postprocess import OfficialTaskPrediction

SCHEMA_VERSION = 1
ROOT_KEYS = {"schema_version", "key", "provenance", "task_prediction"}
KEY_KEYS = {
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "stage_index",
    "history_scan_ids",
    "local_window_scan_ids",
}
PROVENANCE_KEYS = {
    "checkpoint_sha256",
    "config_hash",
    "protocol_manifest_hash",
    "source_raw_observation_fingerprint",
}
TASK_KEYS = {
    "pred_masks",
    "pred_scores",
    "pred_classes",
    "source_query_ids",
    "source_class_ids",
    "latest_stage_index",
}


def _exact_mapping(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} must be a mapping"
        )
    if set(value) != expected:
        raise ValueError(f"{name} fields differ")
    return value


def _sha(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _tensor(value: object, *, name: str, kind: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.device.type != "cpu" or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} CPU tensor")
    if kind == "float":
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating values")
    elif kind == "integer":
        try:
            torch.iinfo(value.dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"{name} must use integer dtype") from error
    elif kind == "bool":
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must use bool dtype")
    else:  # pragma: no cover - internal contract.
        raise RuntimeError(f"unsupported tensor kind {kind}")
    return value


def _validate_key(value: object) -> Mapping[str, Any]:
    key = _exact_mapping(value, KEY_KEYS, name="sidecar key")
    for name in ("master_sequence_id", "reference_scene_id"):
        if not isinstance(key[name], str) or not key[name]:
            raise ValueError(f"sidecar key {name} must be nonempty")
    if key["order_id"] not in {"canonical", "reverse", "sha256_seed45"}:
        raise ValueError("sidecar key order is not preregistered")
    stage = key["stage_index"]
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 4:
        raise ValueError("sidecar stage_index must be within [0, 4]")
    history = key["history_scan_ids"]
    window = key["local_window_scan_ids"]
    if (
        isinstance(history, (str, bytes))
        or not isinstance(history, Sequence)
        or isinstance(window, (str, bytes))
        or not isinstance(window, Sequence)
        or len(history) != stage + 1
        or list(window) != list(history[-1:] if stage == 0 else history[-2:])
        or any(not isinstance(item, str) or not item for item in (*history, *window))
    ):
        raise ValueError("sidecar key does not describe a causal local window")
    return key


def validate_task_sidecar(payload: object) -> None:
    root = _exact_mapping(payload, ROOT_KEYS, name="task sidecar")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported task sidecar schema_version")
    key = _validate_key(root["key"])
    provenance = _exact_mapping(
        root["provenance"], PROVENANCE_KEYS, name="task sidecar provenance"
    )
    for name in sorted(PROVENANCE_KEYS):
        _sha(provenance[name], name=f"task sidecar provenance {name}")
    task = _exact_mapping(
        root["task_prediction"], TASK_KEYS, name="task_prediction"
    )
    masks = _tensor(task["pred_masks"], name="pred_masks", kind="bool", ndim=2)
    scores = _tensor(
        task["pred_scores"], name="pred_scores", kind="float", ndim=1
    )
    classes = _tensor(
        task["pred_classes"], name="pred_classes", kind="integer", ndim=1
    )
    query_ids = _tensor(
        task["source_query_ids"],
        name="source_query_ids",
        kind="integer",
        ndim=1,
    )
    source_classes = _tensor(
        task["source_class_ids"],
        name="source_class_ids",
        kind="integer",
        ndim=1,
    )
    candidate_count = scores.numel()
    if masks.shape[0] <= 0 or any(
        value.numel() != candidate_count
        for value in (classes, query_ids, source_classes)
    ) or masks.shape[1] != candidate_count:
        raise ValueError("task candidate dimensions differ")
    if torch.any(query_ids < 0).item() or torch.any(source_classes < 0).item():
        raise ValueError("task candidate lineage must be nonnegative")
    latest_stage = task["latest_stage_index"]
    if latest_stage != key["stage_index"]:
        raise ValueError("task latest stage differs from sidecar key")


def _update_digest(hasher: Any, value: object) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().contiguous().cpu()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("ascii") + b"\0")
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        hasher.update(b"\0" + tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping\0")
        for key in sorted(value):
            _update_digest(hasher, key)
            _update_digest(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(b"sequence\0")
        for item in value:
            _update_digest(hasher, item)
        return
    if value is None or isinstance(value, (bool, int, str)):
        hasher.update(type(value).__name__.encode("ascii") + b"\0")
        hasher.update(json.dumps(value, ensure_ascii=True).encode("utf-8") + b"\0")
        return
    if isinstance(value, float) and math.isfinite(value):
        hasher.update(b"float\0" + value.hex().encode("ascii") + b"\0")
        return
    raise ValueError(f"unsupported sidecar digest type: {type(value).__name__}")


def observation_fingerprint(raw_cache_payload: Mapping[str, object]) -> str:
    validate_cache_payload(raw_cache_payload)
    hasher = hashlib.sha256()
    _update_digest(hasher, raw_cache_payload["observation"])
    return hasher.hexdigest()


def task_sidecar_digest(payload: Mapping[str, object]) -> str:
    validate_task_sidecar(payload)
    hasher = hashlib.sha256()
    _update_digest(hasher, payload)
    return hasher.hexdigest()


def _clone_cpu(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def build_task_sidecar(
    *,
    raw_cache_payload: Mapping[str, object],
    official_prediction: OfficialTaskPrediction,
    protocol_manifest_sha256: str,
) -> dict[str, object]:
    validate_cache_payload(raw_cache_payload)
    official_prediction.validate()
    _sha(protocol_manifest_sha256, name="protocol_manifest_sha256")
    raw_observation = raw_cache_payload["observation"]
    query_count = int(raw_observation["local_query_ids"].numel())
    class_count = int(raw_observation["class_prob"].shape[1])
    if official_prediction.source_query_ids.numel() and (
        int(official_prediction.source_query_ids.max().item()) >= query_count
        or int(official_prediction.source_class_ids.max().item()) >= class_count - 1
    ):
        raise ValueError("official task lineage is outside the raw observation")
    key = _clone_cpu(raw_cache_payload["key"])
    raw_provenance = raw_cache_payload["provenance"]
    task = official_prediction.prediction(latest_only=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "provenance": {
            "checkpoint_sha256": raw_provenance["checkpoint_sha256"],
            "config_hash": raw_provenance["config_sha256"],
            "protocol_manifest_hash": protocol_manifest_sha256,
            "source_raw_observation_fingerprint": observation_fingerprint(
                raw_cache_payload
            ),
        },
        "task_prediction": {
            "pred_masks": task["pred_masks"],
            "pred_scores": task["pred_scores"],
            "pred_classes": task["pred_classes"],
            "source_query_ids": task["source_query_ids"],
            "source_class_ids": task["source_class_ids"],
            "latest_stage_index": key["stage_index"],
        },
    }
    validate_task_sidecar(payload)
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bundle(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("task sidecar must be a regular non-symlink file")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("task sidecar cannot be loaded safely") from error
    bundle = _exact_mapping(
        bundle, {"content_sha256", "payload"}, name="task sidecar bundle"
    )
    expected = _sha(bundle["content_sha256"], name="bundle content_sha256")
    payload = bundle["payload"]
    validate_task_sidecar(payload)
    digest = task_sidecar_digest(payload)
    if digest != expected or path.name != f"{digest}.pt":
        raise ValueError("task sidecar content digest differs")
    return payload


def load_task_sidecar(path: Path) -> Mapping[str, object]:
    return _clone_cpu(_load_bundle(path))


def _entry(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "filename": path.name,
        "content_sha256": task_sidecar_digest(payload),
        "file_sha256": _file_sha256(path),
        "file_bytes": path.stat().st_size,
        "key": _clone_cpu(payload["key"]),
    }


def write_task_sidecar(
    cache_directory: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    validate_task_sidecar(payload)
    digest = task_sidecar_digest(payload)
    cache_directory.mkdir(parents=True, exist_ok=True)
    target = cache_directory / f"{digest}.pt"
    if target.exists() or target.is_symlink():
        loaded = load_task_sidecar(target)
        if task_sidecar_digest(loaded) != digest:
            raise ValueError("existing task sidecar has different content")
        return _entry(target, loaded)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".tmp", dir=cache_directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save({"content_sha256": digest, "payload": payload}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        loaded = load_task_sidecar(target)
        if task_sidecar_digest(loaded) != digest:
            raise ValueError("published task sidecar has different content")
        return _entry(target, loaded)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "SCHEMA_VERSION",
    "build_task_sidecar",
    "load_task_sidecar",
    "observation_fingerprint",
    "task_sidecar_digest",
    "validate_task_sidecar",
    "write_task_sidecar",
]
