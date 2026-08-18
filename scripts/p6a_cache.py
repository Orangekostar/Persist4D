"""Content-addressed frozen local-prediction cache for P6-A."""

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

SCHEMA_VERSION = 1
ROOT_KEYS = {"schema_version", "key", "provenance", "observation", "target"}
KEY_KEYS = {
    "master_sequence_id",
    "reference_scene_id",
    "order_id",
    "stage_index",
    "history_scan_ids",
    "local_window_scan_ids",
}
PROVENANCE_KEYS = {
    "source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "dataset_sha256",
}
OBSERVATION_KEYS = {
    "features",
    "class_prob",
    "confidence",
    "valid",
    "masks",
    "mask_support",
    "local_query_ids",
}
TARGET_KEYS = {"gt_ids", "gt_classes", "gt_masks", "changes"}
ENTRY_KEYS = {"filename", "content_sha256", "file_sha256", "file_bytes", "key"}


def _exact_mapping(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} must be a mapping"
        )
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [_nonempty_string(item, name=f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _sha(value: object, *, length: int, name: str) -> str:
    text = _nonempty_string(value, name=name)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal digest")
    return text


def _tensor(
    value: object,
    *,
    name: str,
    ndim: int,
    kind: str,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} must be a tensor"
        )
    if value.device.type != "cpu" or value.ndim != ndim:
        raise ValueError(f"{name} must be a CPU tensor with rank {ndim}")
    if kind == "float":
        if not value.is_floating_point() or not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain finite floating values")
    elif kind == "integer":
        try:
            torch.iinfo(value.dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"{name} must use an integer dtype") from error
    elif kind == "bool":
        if value.dtype != torch.bool:
            raise ValueError(f"{name} must use bool dtype")
    else:
        raise RuntimeError(f"unsupported tensor kind: {kind}")
    return value


def validate_cache_payload(payload: object) -> None:
    root = _exact_mapping(payload, ROOT_KEYS, name="cache payload")
    if (
        isinstance(root["schema_version"], bool)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported cache schema_version")

    key = _exact_mapping(root["key"], KEY_KEYS, name="cache key")
    _nonempty_string(key["master_sequence_id"], name="master_sequence_id")
    _nonempty_string(key["reference_scene_id"], name="reference_scene_id")
    _nonempty_string(key["order_id"], name="order_id")
    stage_index = key["stage_index"]
    if (
        isinstance(stage_index, bool)
        or not isinstance(stage_index, int)
        or stage_index < 0
    ):
        raise ValueError("stage_index must be a non-negative integer")
    history = _string_list(key["history_scan_ids"], name="history_scan_ids")
    local_window = _string_list(
        key["local_window_scan_ids"], name="local_window_scan_ids"
    )
    if len(history) != stage_index + 1:
        raise ValueError("history_scan_ids must end at stage_index")
    expected_window = history[-1:] if stage_index == 0 else history[-2:]
    if local_window != expected_window:
        raise ValueError("local_window_scan_ids must be the causal local window")

    provenance = _exact_mapping(root["provenance"], PROVENANCE_KEYS, name="provenance")
    _sha(provenance["source_commit"], length=40, name="source_commit")
    for name in ("checkpoint_sha256", "config_sha256", "dataset_sha256"):
        _sha(provenance[name], length=64, name=name)

    observation = _exact_mapping(
        root["observation"], OBSERVATION_KEYS, name="observation"
    )
    features = _tensor(observation["features"], name="features", ndim=2, kind="float")
    class_prob = _tensor(
        observation["class_prob"], name="class_prob", ndim=2, kind="float"
    )
    confidence = _tensor(
        observation["confidence"], name="confidence", ndim=1, kind="float"
    )
    valid = _tensor(observation["valid"], name="valid", ndim=1, kind="bool")
    masks = _tensor(observation["masks"], name="masks", ndim=2, kind="bool")
    support = _tensor(
        observation["mask_support"], name="mask_support", ndim=1, kind="integer"
    )
    query_ids = _tensor(
        observation["local_query_ids"],
        name="local_query_ids",
        ndim=1,
        kind="integer",
    )
    query_count = features.shape[0]
    if query_count <= 0 or features.shape[1] <= 0 or class_prob.shape[1] <= 1:
        raise ValueError("observation dimensions must be positive")
    if any(
        value.shape[0] != query_count
        for value in (class_prob, confidence, valid, masks, support, query_ids)
    ):
        raise ValueError("observation tensors must share query dimension")
    if masks.shape[1] <= 0:
        raise ValueError("observation masks must cover stage points")
    if torch.any(class_prob < 0).item() or not torch.allclose(
        class_prob.sum(dim=1),
        torch.ones(query_count, dtype=class_prob.dtype),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("class_prob rows must be normalized probabilities")
    if torch.any((confidence < 0) | (confidence > 1)).item():
        raise ValueError("confidence must be within [0, 1]")
    if not torch.equal(support, masks.sum(dim=1, dtype=support.dtype)):
        raise ValueError("mask_support must equal the mask point count")
    if not torch.equal(query_ids, torch.arange(query_count, dtype=query_ids.dtype)):
        raise ValueError("local_query_ids must preserve the local query index")

    target = _exact_mapping(root["target"], TARGET_KEYS, name="target")
    gt_ids = _tensor(target["gt_ids"], name="gt_ids", ndim=1, kind="integer")
    gt_classes = _tensor(
        target["gt_classes"], name="gt_classes", ndim=1, kind="integer"
    )
    gt_masks = _tensor(target["gt_masks"], name="gt_masks", ndim=2, kind="bool")
    changes = _tensor(target["changes"], name="changes", ndim=1, kind="integer")
    gt_count = gt_ids.shape[0]
    if any(value.shape[0] != gt_count for value in (gt_classes, gt_masks, changes)):
        raise ValueError("target tensors must share GT dimension")
    if gt_masks.shape[1] != masks.shape[1]:
        raise ValueError("prediction and target masks must cover the same points")
    if gt_ids.unique().numel() != gt_count:
        raise ValueError("gt_ids must be unique within a stage")


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
    raise ValueError(f"unsupported cache digest type: {type(value).__name__}")


def cache_payload_digest(payload: Mapping[str, object]) -> str:
    validate_cache_payload(payload)
    hasher = hashlib.sha256()
    _update_digest(hasher, payload)
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _plain_key(value: Mapping[str, object]) -> dict[str, object]:
    key = _exact_mapping(value, KEY_KEYS, name="cache key")
    return {
        "master_sequence_id": key["master_sequence_id"],
        "reference_scene_id": key["reference_scene_id"],
        "order_id": key["order_id"],
        "stage_index": key["stage_index"],
        "history_scan_ids": list(key["history_scan_ids"]),
        "local_window_scan_ids": list(key["local_window_scan_ids"]),
    }


def write_cache_entry(
    cache_directory: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    validate_cache_payload(payload)
    digest = cache_payload_digest(payload)
    cache_directory.mkdir(parents=True, exist_ok=True)
    target = cache_directory / f"{digest}.pt"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite cache entry: {target.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{digest}.", suffix=".tmp", dir=cache_directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save({"content_sha256": digest, "payload": payload}, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "filename": target.name,
        "content_sha256": digest,
        "file_sha256": _file_sha256(target),
        "file_bytes": target.stat().st_size,
        "key": _plain_key(payload["key"]),
    }


def load_cache_entry(
    path: Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("cache entry must be a regular non-symlink file")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("cache entry cannot be loaded safely") from error
    bundle = _exact_mapping(bundle, {"content_sha256", "payload"}, name="cache bundle")
    payload = bundle["payload"]
    validate_cache_payload(payload)
    digest = cache_payload_digest(payload)
    if bundle["content_sha256"] != digest or path.name != f"{digest}.pt":
        raise ValueError("cache entry content digest does not match its identity")
    if expected_provenance is not None and dict(payload["provenance"]) != dict(
        expected_provenance
    ):
        raise ValueError("cache entry provenance differs from the frozen run")
    return payload


def _key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(_plain_key(value), sort_keys=True, separators=(",", ":"))


def build_cache_manifest(
    entries: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    normalized = []
    identities = []
    for raw_entry in entries:
        entry = _exact_mapping(raw_entry, ENTRY_KEYS, name="cache manifest entry")
        _sha(entry["content_sha256"], length=64, name="content_sha256")
        _sha(entry["file_sha256"], length=64, name="file_sha256")
        if (
            isinstance(entry["file_bytes"], bool)
            or not isinstance(entry["file_bytes"], int)
            or entry["file_bytes"] <= 0
        ):
            raise ValueError("file_bytes must be a positive integer")
        filename = _nonempty_string(entry["filename"], name="filename")
        if (
            filename != f"{entry['content_sha256']}.pt"
            or Path(filename).name != filename
        ):
            raise ValueError("cache filename must be its content digest")
        plain = dict(entry)
        plain["key"] = _plain_key(entry["key"])
        normalized.append(plain)
        identities.append(_key_identity(plain["key"]))
    if len(set(identities)) != len(identities):
        raise ValueError("cache manifest contains duplicate logical keys")
    expected_identities = [_key_identity(key) for key in expected_keys]
    if len(set(expected_identities)) != len(expected_identities):
        raise ValueError("expected cache keys contain duplicates")
    if set(identities) != set(expected_identities):
        raise ValueError("cache manifest does not have exact expected coverage")
    normalized.sort(
        key=lambda entry: (
            entry["key"]["master_sequence_id"],
            entry["key"]["order_id"],
            entry["key"]["stage_index"],
        )
    )
    hasher = hashlib.sha256()
    _update_digest(hasher, normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "entry_count": len(normalized),
        "entries_sha256": hasher.hexdigest(),
        "entries": normalized,
    }
