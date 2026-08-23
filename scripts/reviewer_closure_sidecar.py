"""Raw Full-History observation sidecars for cross-prefix association."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.system_comparison_inference import (
    FullHistoryCacheError,
    build_full_history_cache_manifest,
    full_history_cache_keys,
    full_history_content_sha256,
    full_history_observation_fingerprints,
    pack_bool_matrix,
    unpack_bool_matrix,
    validate_full_history_cache_key,
    validate_full_history_payload,
)

SIDECAR_SCHEMA_VERSION = "full-history-observations-v2"
_ORDERS = {"canonical", "reverse", "sha256_seed45"}
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_KEY_FIELDS = {
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "horizon",
    "history_scan_ids",
    "scan_indices",
}
_OBSERVATION_FIELDS = {
    "features",
    "class_prob",
    "confidence",
    "valid",
    "current_stage_masks",
    "mask_support",
    "local_query_ids",
}
_RAW_OBSERVATION_FIELDS = (_OBSERVATION_FIELDS - {"current_stage_masks"}) | {"masks"}
_PROVENANCE_FIELDS = {
    "source_prediction_content_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "source_commit",
}
_SOURCE_PROVENANCE_FIELDS = {
    "checkpoint_sha256",
    "config_sha256",
    "protocol_sha256",
    "source_commit",
}
_FINGERPRINT_FIELDS = {
    "features",
    "class_prob",
    "confidence",
    "valid",
    "masks",
    "mask_support",
    "local_query_ids",
    "combined",
}


class FullHistoryObservationSidecarError(ValueError):
    """Raised when a Full-History observation sidecar violates its contract."""


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullHistoryObservationSidecarError(f"{name} must be a mapping")
    return dict(value)


def _tensor(value: object, *, name: str, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise FullHistoryObservationSidecarError(f"{name} must be a tensor")
    result = value.detach().cpu().contiguous().clone()
    if ndim is not None and result.ndim != ndim:
        raise FullHistoryObservationSidecarError(f"{name} must have rank {ndim}")
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise FullHistoryObservationSidecarError(f"{name} must contain finite values")
    return result


def _integer_tensor(value: object, *, name: str) -> Tensor:
    result = _tensor(value, name=name, ndim=1)
    if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
        raise FullHistoryObservationSidecarError(f"{name} must use an integer dtype")
    return result.long()


def _digest(value: object, *, name: str, commit: bool = False) -> str:
    pattern = _HEX40 if commit else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FullHistoryObservationSidecarError(f"{name} is invalid")
    return value


def _normalize_key(value: object) -> dict[str, object]:
    key = _mapping(value, name="sidecar key")
    if set(key) != _KEY_FIELDS:
        raise FullHistoryObservationSidecarError("sidecar key fields differ")
    reference = key["reference_scene_id"]
    master = key["master_sequence_id"]
    order = key["order_id"]
    horizon = key["horizon"]
    scan_ids = key["history_scan_ids"]
    scan_indices = key["scan_indices"]
    if (
        not isinstance(reference, str)
        or not reference
        or not isinstance(master, str)
        or not master
        or order not in _ORDERS
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon not in {2, 3, 4, 5}
        or not isinstance(scan_ids, list)
        or not isinstance(scan_indices, list)
        or len(scan_ids) != horizon
        or len(scan_indices) != horizon
        or any(not isinstance(item, str) or not item for item in scan_ids)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in scan_indices
        )
        or len(set(scan_ids)) != horizon
        or len(set(scan_indices)) != horizon
    ):
        raise FullHistoryObservationSidecarError(
            "sidecar key is not an exact causal O2-O5 prefix"
        )
    return {
        "reference_scene_id": reference,
        "master_sequence_id": master,
        "order_id": order,
        "horizon": horizon,
        "history_scan_ids": list(scan_ids),
        "scan_indices": list(scan_indices),
    }


def _source_key(value: object) -> dict[str, object]:
    source = _mapping(value, name="source prediction key")
    if set(source) == _KEY_FIELDS:
        return _normalize_key(source)
    try:
        full = validate_full_history_cache_key(source)
    except FullHistoryCacheError as error:
        raise FullHistoryObservationSidecarError(
            "source prediction key differs from causal prefix"
        ) from error
    return _normalize_key(
        {
            "reference_scene_id": full["reference_scene_id"],
            "master_sequence_id": full["master_sequence_id"],
            "order_id": full["order_id"],
            "horizon": full["horizon"],
            "history_scan_ids": full["history_scan_ids"],
            "scan_indices": full["scan_indices"],
        }
    )


def _normalize_fingerprints(value: object) -> dict[str, str]:
    fingerprints = _mapping(value, name="source observation fingerprints")
    if set(fingerprints) != _FINGERPRINT_FIELDS:
        raise FullHistoryObservationSidecarError(
            "source observation fingerprint fields differ"
        )
    return {
        name: _digest(digest, name=f"fingerprint.{name}")
        for name, digest in fingerprints.items()
    }


def observation_fingerprints(
    observation: Mapping[str, Tensor],
) -> dict[str, str]:
    try:
        return full_history_observation_fingerprints(observation)
    except (FullHistoryCacheError, KeyError) as error:
        raise FullHistoryObservationSidecarError(
            "raw observation fingerprint cannot be computed"
        ) from error


def _normalize_raw_observation(
    value: Mapping[str, object],
) -> dict[str, Tensor]:
    raw = _mapping(value, name="raw observation")
    if set(raw) != _RAW_OBSERVATION_FIELDS:
        raise FullHistoryObservationSidecarError("raw observation fields differ")
    features = _tensor(raw["features"], name="features", ndim=2)
    class_prob = _tensor(raw["class_prob"], name="class_prob", ndim=2)
    confidence = _tensor(raw["confidence"], name="confidence", ndim=1)
    valid = _tensor(raw["valid"], name="valid", ndim=1)
    masks = _tensor(raw["masks"], name="masks", ndim=2)
    support = _integer_tensor(raw["mask_support"], name="mask_support")
    local_ids = _integer_tensor(raw["local_query_ids"], name="local_query_ids")
    query_count = features.shape[0]
    if (
        query_count <= 0
        or features.shape[1] <= 0
        or class_prob.shape[1] <= 1
        or class_prob.shape[0] != query_count
        or confidence.shape[0] != query_count
        or valid.shape[0] != query_count
        or masks.shape[0] != query_count
        or masks.shape[1] <= 0
        or support.shape[0] != query_count
        or local_ids.shape[0] != query_count
    ):
        raise FullHistoryObservationSidecarError(
            "feature/class/mask/query axes do not align"
        )
    if valid.dtype != torch.bool or masks.dtype != torch.bool:
        raise FullHistoryObservationSidecarError("valid and masks must use bool dtype")
    if not features.is_floating_point() or not class_prob.is_floating_point():
        raise FullHistoryObservationSidecarError(
            "features and class_prob must use floating dtype"
        )
    if not confidence.is_floating_point():
        raise FullHistoryObservationSidecarError("confidence must use floating dtype")
    if (
        torch.any(class_prob < 0).item()
        or torch.any(class_prob > 1).item()
        or not torch.allclose(
            class_prob.sum(dim=1),
            torch.ones(query_count, dtype=class_prob.dtype),
            rtol=1e-5,
            atol=1e-6,
        )
        or torch.any(confidence < 0).item()
        or torch.any(confidence > 1).item()
    ):
        raise FullHistoryObservationSidecarError(
            "class probabilities or confidence are invalid"
        )
    if not torch.equal(support, masks.sum(dim=1, dtype=torch.long)):
        raise FullHistoryObservationSidecarError("mask support differs from masks")
    if torch.any(local_ids < 0).item() or len(set(local_ids.tolist())) != query_count:
        raise FullHistoryObservationSidecarError(
            "local query IDs must be non-negative and unique"
        )
    return {
        "features": features,
        "class_prob": class_prob,
        "confidence": confidence,
        "valid": valid,
        "masks": masks,
        "mask_support": support,
        "local_query_ids": local_ids,
    }


def _source_prediction_binding(value: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(value, name="source prediction")
    if "task_prediction" in source:
        try:
            source = validate_full_history_payload(source)
        except FullHistoryCacheError as error:
            raise FullHistoryObservationSidecarError(
                "source prediction payload is invalid"
            ) from error
    required = {"key", "content_sha256", "provenance", "observation_fingerprints"}
    if not required <= set(source):
        raise FullHistoryObservationSidecarError("source prediction fields differ")
    provenance = _mapping(source["provenance"], name="source prediction provenance")
    if set(provenance) != _SOURCE_PROVENANCE_FIELDS:
        raise FullHistoryObservationSidecarError(
            "source prediction provenance fields differ"
        )
    normalized_provenance = {
        "checkpoint_sha256": _digest(
            provenance["checkpoint_sha256"], name="checkpoint_sha256"
        ),
        "config_sha256": _digest(provenance["config_sha256"], name="config_sha256"),
        "protocol_sha256": _digest(
            provenance["protocol_sha256"], name="protocol_sha256"
        ),
        "source_commit": _digest(
            provenance["source_commit"], name="source_commit", commit=True
        ),
    }
    return {
        "key": _source_key(source["key"]),
        "content_sha256": _digest(
            source["content_sha256"], name="source prediction content_sha256"
        ),
        "provenance": normalized_provenance,
        "observation_fingerprints": _normalize_fingerprints(
            source["observation_fingerprints"]
        ),
    }


def sidecar_content_sha256(value: Mapping[str, object]) -> str:
    try:
        return full_history_content_sha256(value)
    except FullHistoryCacheError as error:
        raise FullHistoryObservationSidecarError(
            "sidecar content digest cannot be computed"
        ) from error


def build_full_history_observation_sidecar(
    *,
    key: Mapping[str, object],
    raw_observation: Mapping[str, object],
    source_prediction: Mapping[str, object],
    sidecar_source_commit: str,
) -> dict[str, object]:
    normalized_key = _normalize_key(key)
    raw = _normalize_raw_observation(raw_observation)
    source = _source_prediction_binding(source_prediction)
    if source["key"] != normalized_key:
        raise FullHistoryObservationSidecarError(
            "source prediction key differs from sidecar prefix"
        )
    fingerprints = observation_fingerprints(raw)
    if fingerprints != source["observation_fingerprints"]:
        raise FullHistoryObservationSidecarError(
            "raw observation fingerprint differs from source prediction"
        )
    provenance = source["provenance"]
    source_commit = _digest(
        sidecar_source_commit, name="sidecar source_commit", commit=True
    )
    payload: dict[str, object] = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "key": normalized_key,
        "provenance": {
            "source_prediction_content_sha256": source["content_sha256"],
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "config_sha256": provenance["config_sha256"],
            "source_commit": source_commit,
        },
        "observation": {
            "features": raw["features"],
            "class_prob": raw["class_prob"],
            "confidence": raw["confidence"],
            "valid": raw["valid"],
            "current_stage_masks": pack_bool_matrix(raw["masks"]),
            "mask_support": raw["mask_support"],
            "local_query_ids": raw["local_query_ids"],
        },
        "source_observation_fingerprints": fingerprints,
    }
    payload["content_sha256"] = sidecar_content_sha256(payload)
    return payload


def validate_full_history_observation_sidecar(
    value: Mapping[str, object],
    *,
    expected_key: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = _mapping(value, name="observation sidecar")
    expected_fields = {
        "schema_version",
        "key",
        "provenance",
        "observation",
        "source_observation_fingerprints",
        "content_sha256",
    }
    if set(payload) != expected_fields:
        raise FullHistoryObservationSidecarError("observation sidecar fields differ")
    if payload["schema_version"] != SIDECAR_SCHEMA_VERSION:
        raise FullHistoryObservationSidecarError("observation sidecar schema differs")
    key = _normalize_key(payload["key"])
    if expected_key is not None and key != _normalize_key(expected_key):
        raise FullHistoryObservationSidecarError("observation sidecar key differs")
    provenance = _mapping(payload["provenance"], name="sidecar provenance")
    if set(provenance) != _PROVENANCE_FIELDS:
        raise FullHistoryObservationSidecarError("sidecar provenance fields differ")
    normalized_provenance = {
        "source_prediction_content_sha256": _digest(
            provenance["source_prediction_content_sha256"],
            name="source_prediction_content_sha256",
        ),
        "checkpoint_sha256": _digest(
            provenance["checkpoint_sha256"], name="checkpoint_sha256"
        ),
        "config_sha256": _digest(provenance["config_sha256"], name="config_sha256"),
        "source_commit": _digest(
            provenance["source_commit"], name="source_commit", commit=True
        ),
    }
    observation = _mapping(payload["observation"], name="sidecar observation")
    if set(observation) != _OBSERVATION_FIELDS:
        raise FullHistoryObservationSidecarError("sidecar observation fields differ")
    try:
        masks = unpack_bool_matrix(observation["current_stage_masks"])
    except FullHistoryCacheError as error:
        raise FullHistoryObservationSidecarError(
            "current-stage masks are invalid"
        ) from error
    raw = _normalize_raw_observation(
        {
            "features": observation["features"],
            "class_prob": observation["class_prob"],
            "confidence": observation["confidence"],
            "valid": observation["valid"],
            "masks": masks,
            "mask_support": observation["mask_support"],
            "local_query_ids": observation["local_query_ids"],
        }
    )
    fingerprints = _normalize_fingerprints(payload["source_observation_fingerprints"])
    if observation_fingerprints(raw) != fingerprints:
        raise FullHistoryObservationSidecarError(
            "sidecar observation fingerprint differs"
        )
    supplied_digest = _digest(payload["content_sha256"], name="content_sha256")
    if supplied_digest != sidecar_content_sha256(payload):
        raise FullHistoryObservationSidecarError("sidecar content digest differs")
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "key": key,
        "provenance": normalized_provenance,
        "observation": {
            "features": raw["features"],
            "class_prob": raw["class_prob"],
            "confidence": raw["confidence"],
            "valid": raw["valid"],
            "current_stage_masks": pack_bool_matrix(raw["masks"]),
            "mask_support": raw["mask_support"],
            "local_query_ids": raw["local_query_ids"],
        },
        "source_observation_fingerprints": fingerprints,
        "content_sha256": supplied_digest,
    }


def _key_identity(key: Mapping[str, object]) -> str:
    return json.dumps(
        _normalize_key(key), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _entry_filename(key: Mapping[str, object]) -> str:
    return hashlib.sha256(_key_identity(key).encode("utf-8")).hexdigest() + ".pt"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_full_history_observation_sidecar_entry(
    directory: str | Path,
    value: Mapping[str, object],
) -> dict[str, object]:
    payload = validate_full_history_observation_sidecar(value)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    output = root / _entry_filename(payload["key"])
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("sidecar key path is not a regular file")
        existing = validate_full_history_observation_sidecar(
            torch.load(output, map_location="cpu", weights_only=False),
            expected_key=payload["key"],
        )
        if existing["content_sha256"] != payload["content_sha256"]:
            raise FileExistsError("sidecar key already contains different content")
        return {
            "key": payload["key"],
            "filename": output.name,
            "sha256": _file_sha256(output),
            "byte_size": output.stat().st_size,
            "content_sha256": existing["content_sha256"],
            "source_prediction_content_sha256": existing["provenance"][
                "source_prediction_content_sha256"
            ],
            "sidecar_source_commit": existing["provenance"]["source_commit"],
        }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "key": payload["key"],
        "filename": output.name,
        "sha256": _file_sha256(output),
        "byte_size": output.stat().st_size,
        "content_sha256": payload["content_sha256"],
        "source_prediction_content_sha256": payload["provenance"][
            "source_prediction_content_sha256"
        ],
        "sidecar_source_commit": payload["provenance"]["source_commit"],
    }


def load_full_history_observation_sidecar_entry(
    directory: str | Path,
    entry: Mapping[str, object],
) -> dict[str, object]:
    record = _mapping(entry, name="sidecar entry")
    expected_fields = {
        "key",
        "filename",
        "sha256",
        "byte_size",
        "content_sha256",
        "source_prediction_content_sha256",
        "sidecar_source_commit",
    }
    if set(record) != expected_fields:
        raise FullHistoryObservationSidecarError("sidecar entry fields differ")
    key = _normalize_key(record["key"])
    if record["filename"] != _entry_filename(key):
        raise FullHistoryObservationSidecarError("sidecar entry filename differs")
    path = Path(directory) / record["filename"]
    if path.is_symlink() or not path.is_file():
        raise FullHistoryObservationSidecarError("sidecar entry file is unavailable")
    if (
        isinstance(record["byte_size"], bool)
        or not isinstance(record["byte_size"], int)
        or record["byte_size"] < 0
        or path.stat().st_size != record["byte_size"]
        or _file_sha256(path) != _digest(record["sha256"], name="entry sha256")
    ):
        raise FullHistoryObservationSidecarError("sidecar entry file hash differs")
    payload = validate_full_history_observation_sidecar(
        torch.load(path, map_location="cpu", weights_only=False),
        expected_key=key,
    )
    if (
        payload["content_sha256"]
        != _digest(record["content_sha256"], name="entry content_sha256")
        or payload["provenance"]["source_prediction_content_sha256"]
        != _digest(
            record["source_prediction_content_sha256"],
            name="entry source prediction content_sha256",
        )
        or payload["provenance"]["source_commit"]
        != _digest(
            record["sidecar_source_commit"],
            name="entry sidecar source commit",
            commit=True,
        )
    ):
        raise FullHistoryObservationSidecarError("sidecar entry content differs")
    return payload


def validate_source_prediction_manifest(
    value: Mapping[str, object],
    *,
    system_manifest: Mapping[str, object],
) -> dict[str, object]:
    source = _mapping(value, name="source prediction manifest")
    entries = source.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise FullHistoryObservationSidecarError(
            "source prediction manifest entries must be a sequence"
        )
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise FullHistoryObservationSidecarError(
            "source prediction manifest provenance must be a mapping"
        )
    try:
        rebuilt = build_full_history_cache_manifest(
            entries,
            expected_keys=full_history_cache_keys(system_manifest),
            expected_provenance=provenance,
        )
    except FullHistoryCacheError as error:
        raise FullHistoryObservationSidecarError(
            "source prediction manifest binding differs"
        ) from error
    if source != rebuilt:
        raise FullHistoryObservationSidecarError(
            "source prediction manifest content differs from rebuilt binding"
        )
    return rebuilt


def _source_entry_index(
    source_prediction_manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    source = _mapping(source_prediction_manifest, name="source prediction manifest")
    entries = source.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise FullHistoryObservationSidecarError(
            "source prediction manifest entries must be a sequence"
        )
    index: dict[str, dict[str, object]] = {}
    for raw_record in entries:
        record = _mapping(raw_record, name="source prediction entry")
        try:
            full_key = validate_full_history_cache_key(record.get("key"))
        except FullHistoryCacheError as error:
            raise FullHistoryObservationSidecarError(
                "source prediction entry key is invalid"
            ) from error
        if full_key["horizon"] < 2:
            continue
        key = _source_key(full_key)
        identity = _key_identity(key)
        if identity in index:
            raise FullHistoryObservationSidecarError(
                "source prediction manifest has duplicate O2-O5 keys"
            )
        content = _digest(
            record.get("content_sha256"), name="source entry content_sha256"
        )
        index[identity] = {**record, "content_sha256": content}
    return index


def source_prediction_entry_for_key(
    source_prediction_manifest: Mapping[str, object],
    key: Mapping[str, object],
) -> dict[str, object]:
    normalized = _normalize_key(key)
    record = _source_entry_index(source_prediction_manifest).get(
        _key_identity(normalized)
    )
    if record is None:
        raise FullHistoryObservationSidecarError(
            "source prediction manifest lacks sidecar key"
        )
    return record


def produce_bound_full_history_observation_sidecar(
    *,
    producer: object,
    source_prediction: Mapping[str, object],
    sidecar_key: Mapping[str, object],
    sidecar_source_commit: str,
) -> dict[str, object]:
    source = _source_prediction_binding(source_prediction)
    normalized_key = _normalize_key(sidecar_key)
    if source["key"] != normalized_key:
        raise FullHistoryObservationSidecarError(
            "source prediction key differs from sidecar prefix"
        )
    produce_bundle = getattr(producer, "produce_bundle", None)
    if not callable(produce_bundle):
        raise FullHistoryObservationSidecarError(
            "Full-History producer does not expose one-forward bundle production"
        )
    bundle = produce_bundle(
        _mapping(source_prediction, name="source prediction")["key"]
    )
    produced_payload = getattr(bundle, "payload", None)
    processed = getattr(bundle, "processed", None)
    raw_observation = getattr(processed, "raw_observation", None)
    if not isinstance(produced_payload, Mapping) or not isinstance(
        raw_observation, Mapping
    ):
        raise FullHistoryObservationSidecarError(
            "Full-History bundle lacks payload or raw observation"
        )
    produced_digest = produced_payload.get("content_sha256")
    if produced_digest != source["content_sha256"]:
        raise FullHistoryObservationSidecarError(
            "rerun failed source prediction parity"
        )
    if "task_prediction" in source_prediction:
        try:
            validate_full_history_payload(
                produced_payload,
                expected_key=source_prediction["key"],
                expected_provenance=source_prediction["provenance"],
            )
        except FullHistoryCacheError as error:
            raise FullHistoryObservationSidecarError(
                "rerun source prediction payload is invalid"
            ) from error
    return build_full_history_observation_sidecar(
        key=normalized_key,
        raw_observation=raw_observation,
        source_prediction=source_prediction,
        sidecar_source_commit=sidecar_source_commit,
    )


def discover_full_history_observation_sidecar_entries(
    directory: str | Path,
) -> list[dict[str, object]]:
    root = Path(directory)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise FullHistoryObservationSidecarError(
            "sidecar cache directory must be a regular directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".pt":
            raise FullHistoryObservationSidecarError(
                f"sidecar cache directory contains unexpected path: {path.name}"
            )
        payload = validate_full_history_observation_sidecar(
            torch.load(path, map_location="cpu", weights_only=False)
        )
        record = {
            "key": payload["key"],
            "filename": path.name,
            "sha256": _file_sha256(path),
            "byte_size": path.stat().st_size,
            "content_sha256": payload["content_sha256"],
            "source_prediction_content_sha256": payload["provenance"][
                "source_prediction_content_sha256"
            ],
            "sidecar_source_commit": payload["provenance"]["source_commit"],
        }
        records.append(_normalize_sidecar_entry_record(record))
    records.sort(key=lambda record: _key_identity(record["key"]))
    return records


def _normalize_sidecar_entry_record(
    value: Mapping[str, object],
) -> dict[str, object]:
    record = _mapping(value, name="sidecar entry record")
    expected = {
        "key",
        "filename",
        "sha256",
        "byte_size",
        "content_sha256",
        "source_prediction_content_sha256",
        "sidecar_source_commit",
    }
    if set(record) != expected:
        raise FullHistoryObservationSidecarError("sidecar entry record fields differ")
    key = _normalize_key(record["key"])
    filename = record["filename"]
    if filename != _entry_filename(key):
        raise FullHistoryObservationSidecarError("sidecar entry filename differs")
    size = record["byte_size"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise FullHistoryObservationSidecarError("sidecar entry byte_size is invalid")
    return {
        "key": key,
        "filename": filename,
        "sha256": _digest(record["sha256"], name="sidecar entry sha256"),
        "byte_size": size,
        "content_sha256": _digest(
            record["content_sha256"], name="sidecar entry content_sha256"
        ),
        "source_prediction_content_sha256": _digest(
            record["source_prediction_content_sha256"],
            name="sidecar source prediction content_sha256",
        ),
        "sidecar_source_commit": _digest(
            record["sidecar_source_commit"],
            name="sidecar source commit",
            commit=True,
        ),
    }


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


def build_full_history_observation_sidecar_manifest(
    entries: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    source_prediction_manifest: Mapping[str, object],
    system_manifest: Mapping[str, object],
    reviewer_manifest: Mapping[str, object],
    sidecar_code_commit: str,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise FullHistoryObservationSidecarError("sidecar entries must be a sequence")
    if (
        isinstance(expected_keys, (str, bytes))
        or not isinstance(expected_keys, Sequence)
        or not expected_keys
    ):
        raise FullHistoryObservationSidecarError(
            "expected sidecar keys must be a nonempty sequence"
        )
    source = validate_source_prediction_manifest(
        source_prediction_manifest, system_manifest=system_manifest
    )
    reviewer = _mapping(reviewer_manifest, name="reviewer manifest")
    reviewer_digest = _digest(
        reviewer.get("content_sha256"), name="reviewer manifest content_sha256"
    )
    commit = _digest(sidecar_code_commit, name="sidecar_code_commit", commit=True)
    expected = [_normalize_key(key) for key in expected_keys]
    expected_identities = [_key_identity(key) for key in expected]
    if len(set(expected_identities)) != len(expected_identities):
        raise FullHistoryObservationSidecarError(
            "expected sidecar coverage contains duplicates"
        )
    normalized = [_normalize_sidecar_entry_record(record) for record in entries]
    actual_identities = [_key_identity(record["key"]) for record in normalized]
    if len(set(actual_identities)) != len(actual_identities) or set(
        actual_identities
    ) != set(expected_identities):
        raise FullHistoryObservationSidecarError(
            "sidecar manifest does not have exact coverage"
        )
    normalized.sort(key=lambda record: _key_identity(record["key"]))
    for record in normalized:
        if record["sidecar_source_commit"] != commit:
            raise FullHistoryObservationSidecarError(
                "sidecar code commit differs from entry provenance"
            )
        source_record = source_prediction_entry_for_key(source, record["key"])
        if (
            record["source_prediction_content_sha256"]
            != source_record["content_sha256"]
        ):
            raise FullHistoryObservationSidecarError(
                "sidecar source prediction content binding differs"
            )
    if cache_directory is not None:
        directory = Path(cache_directory)
        if directory.is_symlink() or not directory.is_dir():
            raise FullHistoryObservationSidecarError(
                "sidecar cache directory must be a regular directory"
            )
        expected_files = {record["filename"] for record in normalized}
        actual_files = {path.name for path in directory.iterdir()}
        if actual_files != expected_files:
            raise FullHistoryObservationSidecarError(
                "sidecar cache directory coverage differs"
            )
        for record in normalized:
            load_full_history_observation_sidecar_entry(directory, record)
    entries_sha256 = hashlib.sha256(
        _canonical_json_bytes({"entries": normalized})
    ).hexdigest()
    manifest: dict[str, object] = {
        "schema_version": "full-history-observations-v2-manifest",
        "status": "pass",
        "reviewer_manifest_content_sha256": reviewer_digest,
        "source_prediction_manifest": {
            "reference": "repo:artifacts/system_comparison/full_history_predictions/manifest.json",
            "content_sha256": source["content_sha256"],
            "entries_sha256": source["entries_sha256"],
            "entry_count": source["entry_count"],
        },
        "sidecar_code_commit": commit,
        "entry_count": len(normalized),
        "entries_sha256": entries_sha256,
        "entries": normalized,
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    return manifest


__all__ = [
    "FullHistoryObservationSidecarError",
    "SIDECAR_SCHEMA_VERSION",
    "build_full_history_observation_sidecar",
    "build_full_history_observation_sidecar_manifest",
    "discover_full_history_observation_sidecar_entries",
    "load_full_history_observation_sidecar_entry",
    "observation_fingerprints",
    "produce_bound_full_history_observation_sidecar",
    "sidecar_content_sha256",
    "source_prediction_entry_for_key",
    "validate_source_prediction_manifest",
    "validate_full_history_observation_sidecar",
    "write_full_history_observation_sidecar_entry",
]
