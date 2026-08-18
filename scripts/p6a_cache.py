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
ORDER_IDS = ("canonical", "reverse", "sha256_seed45")
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
MANIFEST_ROOT_KEYS = {
    "schema_version",
    "status",
    "provenance",
    "entry_count",
    "entries_sha256",
    "entries",
}


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


def _validate_key(value: object, *, name: str = "cache key") -> Mapping[str, Any]:
    key = _exact_mapping(value, KEY_KEYS, name=name)
    _nonempty_string(key["master_sequence_id"], name="master_sequence_id")
    _nonempty_string(key["reference_scene_id"], name="reference_scene_id")
    if not isinstance(key["order_id"], str) or key["order_id"] not in ORDER_IDS:
        raise ValueError(f"order_id must be one of {ORDER_IDS}")
    stage_index = key["stage_index"]
    if (
        isinstance(stage_index, bool)
        or not isinstance(stage_index, int)
        or not 0 <= stage_index <= 4
    ):
        raise ValueError("stage_index must be an integer in [0, 4]")
    history = _string_list(key["history_scan_ids"], name="history_scan_ids")
    local_window = _string_list(
        key["local_window_scan_ids"], name="local_window_scan_ids"
    )
    if len(history) != stage_index + 1:
        raise ValueError("history_scan_ids must end at stage_index")
    expected_window = history[-1:] if stage_index == 0 else history[-2:]
    if local_window != expected_window:
        raise ValueError("local_window_scan_ids must be the causal local window")
    return key


def _validate_provenance(
    value: object, *, name: str = "provenance"
) -> Mapping[str, Any]:
    provenance = _exact_mapping(value, PROVENANCE_KEYS, name=name)
    _sha(provenance["source_commit"], length=40, name=f"{name}.source_commit")
    for key in ("checkpoint_sha256", "config_sha256", "dataset_sha256"):
        _sha(provenance[key], length=64, name=f"{name}.{key}")
    return provenance


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
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported cache schema_version")

    _validate_key(root["key"])

    _validate_provenance(root["provenance"])

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
    if torch.any(gt_ids < 0).item():
        raise ValueError("gt_ids must be non-negative")
    if torch.any(gt_classes < 0).item():
        raise ValueError("gt_classes must be non-negative")
    if torch.any(changes < 0).item():
        raise ValueError("changes must be non-negative")


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
    key = _validate_key(value)
    return {
        "master_sequence_id": key["master_sequence_id"],
        "reference_scene_id": key["reference_scene_id"],
        "order_id": key["order_id"],
        "stage_index": key["stage_index"],
        "history_scan_ids": list(key["history_scan_ids"]),
        "local_window_scan_ids": list(key["local_window_scan_ids"]),
    }


def _validate_entry_metadata(value: object) -> Mapping[str, Any]:
    entry = _exact_mapping(value, ENTRY_KEYS, name="cache manifest entry")
    _sha(entry["content_sha256"], length=64, name="content_sha256")
    _sha(entry["file_sha256"], length=64, name="file_sha256")
    if (
        isinstance(entry["file_bytes"], bool)
        or not isinstance(entry["file_bytes"], int)
        or entry["file_bytes"] <= 0
    ):
        raise ValueError("file_bytes must be a positive integer")
    filename = _nonempty_string(entry["filename"], name="filename")
    if filename != f"{entry['content_sha256']}.pt" or Path(filename).name != filename:
        raise ValueError("cache filename must be its content digest")
    _validate_key(entry["key"])
    return entry


def _entry_sort_key(entry: Mapping[str, object]) -> tuple[object, ...]:
    key = entry["key"]
    return (
        key["reference_scene_id"],
        key["master_sequence_id"],
        key["order_id"],
        key["stage_index"],
        tuple(key["history_scan_ids"]),
        tuple(key["local_window_scan_ids"]),
        entry["filename"],
        entry["content_sha256"],
        entry["file_sha256"],
        entry["file_bytes"],
    )


def _entry_from_path(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "filename": path.name,
        "content_sha256": cache_payload_digest(payload),
        "file_sha256": _file_sha256(path),
        "file_bytes": path.stat().st_size,
        "key": _plain_key(payload["key"]),
    }


def _clone_cpu_snapshot(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu_snapshot(item) for item in value)
    return value


def immutable_snapshot(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Return a validated, detached CPU snapshot for tracker fan-out."""

    validate_cache_payload(payload)
    snapshot = _clone_cpu_snapshot(payload)
    if not isinstance(snapshot, Mapping):  # pragma: no cover - validated above.
        raise TypeError("cache snapshot must be a mapping")
    return snapshot


def _require_regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")


def _load_cache_payload(path: Path) -> Mapping[str, object]:
    _require_regular_file(path, name="cache entry")
    try:
        bundle = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("cache entry cannot be loaded safely") from error
    bundle = _exact_mapping(bundle, {"content_sha256", "payload"}, name="cache bundle")
    _sha(bundle["content_sha256"], length=64, name="bundle.content_sha256")
    payload = bundle["payload"]
    validate_cache_payload(payload)
    digest = cache_payload_digest(payload)
    if bundle["content_sha256"] != digest or path.name != f"{digest}.pt":
        raise ValueError("cache entry content digest does not match its identity")
    return payload


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_cache_entry(
    target: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    loaded = load_cache_entry(target, expected_provenance=payload["provenance"])
    entry = _entry_from_path(target, loaded)
    if entry["content_sha256"] != cache_payload_digest(payload):
        raise ValueError("existing cache entry has different content")
    if entry["key"] != _plain_key(payload["key"]):
        raise ValueError("existing cache entry has different logical key")
    return entry


def write_cache_entry(
    cache_directory: Path, payload: Mapping[str, object]
) -> dict[str, object]:
    validate_cache_payload(payload)
    digest = cache_payload_digest(payload)
    if not isinstance(cache_directory, Path):
        raise TypeError("cache_directory must be a Path")
    cache_directory.mkdir(parents=True, exist_ok=True)
    target = cache_directory / f"{digest}.pt"
    if target.exists() or target.is_symlink():
        return _existing_cache_entry(target, payload)

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
            # Hard-link publication is atomic and never replaces an existing target.
            os.link(temporary, target)
            _fsync_directory(cache_directory)
        except FileExistsError:
            return _existing_cache_entry(target, payload)
        return _existing_cache_entry(target, payload)
    finally:
        temporary.unlink(missing_ok=True)


def validate_cache_entry(
    path: Path,
    expected_entry: Mapping[str, object],
    expected_provenance: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Validate file identity, serialized payload, logical key, and provenance."""

    if not isinstance(path, Path):
        raise TypeError("cache entry path must be a Path")
    entry = _validate_entry_metadata(expected_entry)
    _require_regular_file(path, name="cache entry")
    if path.name != entry["filename"]:
        raise ValueError("cache entry filename differs from manifest")
    actual_bytes = path.stat().st_size
    actual_sha = _file_sha256(path)
    if actual_bytes != entry["file_bytes"] or actual_sha != entry["file_sha256"]:
        raise ValueError("cache entry file evidence differs from manifest")
    payload = _load_cache_payload(path)
    if cache_payload_digest(payload) != entry["content_sha256"]:
        raise ValueError("cache entry content differs from manifest")
    if _plain_key(payload["key"]) != entry["key"]:
        raise ValueError("cache entry key differs from manifest")
    if expected_provenance is not None:
        provenance = _validate_provenance(
            expected_provenance, name="expected_provenance"
        )
        if dict(payload["provenance"]) != dict(provenance):
            raise ValueError("cache entry provenance differs from the frozen run")
    return immutable_snapshot(payload)


def load_cache_entry(
    path: Path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    payload = _load_cache_payload(path)
    if expected_provenance is not None:
        provenance = _validate_provenance(
            expected_provenance, name="expected_provenance"
        )
        if dict(payload["provenance"]) != dict(provenance):
            raise ValueError("cache entry provenance differs from the frozen run")
    return immutable_snapshot(payload)


def _key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(_plain_key(value), sort_keys=True, separators=(",", ":"))


def _normalize_manifest_entries(
    entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise TypeError("cache manifest entries must be a sequence")
    normalized = []
    for raw_entry in entries:
        entry = _validate_entry_metadata(raw_entry)
        plain = dict(entry)
        plain["key"] = _plain_key(entry["key"])
        normalized.append(plain)
    identities = [_key_identity(entry["key"]) for entry in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError("cache manifest contains duplicate logical keys")
    normalized.sort(key=_entry_sort_key)
    return normalized


def _validate_expected_coverage(
    entries: Sequence[Mapping[str, object]],
    expected_keys: Sequence[Mapping[str, object]],
) -> None:
    identities = [_key_identity(entry["key"]) for entry in entries]
    expected_identities = [_key_identity(key) for key in expected_keys]
    if len(set(expected_identities)) != len(expected_identities):
        raise ValueError("expected cache keys contain duplicates")
    if set(identities) != set(expected_identities):
        raise ValueError("cache manifest does not have exact expected coverage")


def _validate_cache_directory(
    cache_directory: Path, expected_filenames: set[str]
) -> None:
    if cache_directory.is_symlink() or not cache_directory.is_dir():
        raise ValueError("cache_directory must be a regular directory")
    unexpected = [
        path
        for path in cache_directory.iterdir()
        if path.is_file() or path.is_symlink()
        if path.name not in expected_filenames
    ]
    if unexpected:
        raise ValueError(f"cache_directory contains unexpected files: {unexpected[0]}")


def _manifest_root(
    normalized: list[dict[str, object]],
    expected_provenance: Mapping[str, object],
) -> dict[str, object]:
    provenance = dict(
        _validate_provenance(expected_provenance, name="expected_provenance")
    )
    hasher = hashlib.sha256()
    _update_digest(hasher, normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "provenance": provenance,
        "entry_count": len(normalized),
        "entries_sha256": hasher.hexdigest(),
        "entries": normalized,
    }


def validate_cache_manifest(
    manifest: object,
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object] | None = None,
    cache_directory: Path | None = None,
) -> None:
    root = _exact_mapping(manifest, MANIFEST_ROOT_KEYS, name="cache manifest")
    if (
        isinstance(root["schema_version"], bool)
        or not isinstance(root["schema_version"], int)
        or root["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("unsupported cache manifest schema_version")
    if root["status"] != "pass":
        raise ValueError("cache manifest status must be pass")
    provenance = _validate_provenance(root["provenance"])
    if expected_provenance is not None:
        expected = _validate_provenance(expected_provenance, name="expected_provenance")
        if dict(provenance) != dict(expected):
            raise ValueError("cache manifest provenance differs from the frozen run")
    entries = root["entries"]
    if not isinstance(entries, list):
        raise TypeError("cache manifest entries must be a list")
    normalized = _normalize_manifest_entries(entries)
    if normalized != entries:
        raise ValueError("cache manifest entries are not in canonical order")
    _validate_expected_coverage(normalized, expected_keys)
    if (
        isinstance(root["entry_count"], bool)
        or not isinstance(root["entry_count"], int)
        or root["entry_count"] != len(normalized)
    ):
        raise ValueError("cache manifest entry_count is inconsistent")
    _sha(root["entries_sha256"], length=64, name="entries_sha256")
    hasher = hashlib.sha256()
    _update_digest(hasher, normalized)
    if root["entries_sha256"] != hasher.hexdigest():
        raise ValueError("cache manifest entries_sha256 is inconsistent")
    if cache_directory is not None:
        if not isinstance(cache_directory, Path):
            raise TypeError("cache_directory must be a Path")
        expected_filenames = {entry["filename"] for entry in normalized}
        _validate_cache_directory(cache_directory, expected_filenames)
        for entry in normalized:
            validate_cache_entry(cache_directory / entry["filename"], entry, provenance)


def build_cache_manifest(
    entries: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    cache_directory: Path | None = None,
) -> dict[str, object]:
    normalized = _normalize_manifest_entries(entries)
    _validate_expected_coverage(normalized, expected_keys)
    manifest = _manifest_root(normalized, expected_provenance)
    validate_cache_manifest(
        manifest,
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
        cache_directory=cache_directory,
    )
    return manifest


def _manifest_json_text(manifest: Mapping[str, object]) -> str:
    return (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_cache_manifest(
    path: Path,
    manifest: Mapping[str, object],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    cache_directory: Path | None = None,
) -> Mapping[str, object]:
    validate_cache_manifest(
        manifest,
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
        cache_directory=cache_directory,
    )
    if not isinstance(path, Path):
        raise TypeError("manifest path must be a Path")
    text = _manifest_json_text(manifest)
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _require_regular_file(path, name="cache manifest")
        if path.read_bytes() == payload:
            return manifest
        raise FileExistsError(f"refusing to overwrite cache manifest: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            _require_regular_file(path, name="cache manifest")
            if path.read_bytes() == payload:
                return manifest
            raise FileExistsError(f"refusing to overwrite cache manifest: {path}")
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def load_cache_manifest(
    path: Path,
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
    cache_directory: Path | None = None,
) -> dict[str, object]:
    if not isinstance(path, Path):
        raise TypeError("manifest path must be a Path")
    _require_regular_file(path, name="cache manifest")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cache manifest cannot be decoded") from error
    validate_cache_manifest(
        manifest,
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
        cache_directory=cache_directory,
    )
    if not isinstance(manifest, dict):  # pragma: no cover - exact mapping above.
        raise TypeError("cache manifest must decode to a mapping")
    return manifest
