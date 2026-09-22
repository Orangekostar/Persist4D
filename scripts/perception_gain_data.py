"""Data-role and deterministic sampling contracts for Perception Gain V1."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class PerceptionDataError(ValueError):
    """Raised when a frozen data role or sampling contract is violated."""


_EVALUATION_ROLES = ("CAL", "SEL", "PB", "LOCAL-T2", "ADDITIONAL")


def _string_set(value: object, *, name: str) -> set[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PerceptionDataError(f"{name} must be a sequence")
    result = set()
    for item in value:
        if not isinstance(item, str) or not item:
            raise PerceptionDataError(f"{name} must contain non-empty strings")
        result.add(item)
    return result


def build_data_roles(
    task_contract: Mapping[str, Any],
    crosswindow_roles: Mapping[str, Any],
    *,
    local_t2_reference_ids: Sequence[str],
) -> dict[str, list[str]]:
    """Preserve frozen evaluation roles and subtract their union from TRAIN."""

    task_roles = task_contract.get("roles")
    cross_roles = crosswindow_roles.get("roles")
    if not isinstance(task_roles, Mapping) or not isinstance(cross_roles, Mapping):
        raise PerceptionDataError("source data roles are unavailable")
    train = _string_set(
        task_roles.get("adaptation_reference_ids"), name="adaptation references"
    )
    roles = {
        "CAL": _string_set(cross_roles.get("DEV-CAL"), name="DEV-CAL"),
        "SEL": _string_set(cross_roles.get("DEV-SEL"), name="DEV-SEL"),
        "PB": _string_set(
            task_roles.get("protocol_b_reference_ids"), name="protocol B"
        ),
        "LOCAL-T2": _string_set(local_t2_reference_ids, name="LOCAL-T2"),
        "ADDITIONAL": _string_set(
            task_roles.get("additional_native_reference_ids"),
            name="additional native references",
        ),
    }
    evaluation = set().union(*roles.values())
    removed = train & evaluation
    result = {
        "TRAIN": sorted(train - evaluation),
        **{name: sorted(values) for name, values in roles.items()},
        "removed_train_overlap": sorted(removed),
    }
    validate_role_disjointness(result)
    return result


def validate_role_disjointness(roles: Mapping[str, object]) -> None:
    train = _string_set(roles.get("TRAIN"), name="TRAIN")
    evaluation = set()
    for name in _EVALUATION_ROLES:
        evaluation.update(_string_set(roles.get(name), name=name))
    overlap = sorted(train & evaluation)
    if overlap:
        raise PerceptionDataError(f"TRAIN overlaps evaluation references: {overlap}")


def _hash_order(prefix: str, value: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{prefix}{value}".encode("utf-8")).hexdigest()
    return digest, value


def select_scorer_pairs(
    pairs: Sequence[Mapping[str, object]],
    *,
    limit: int = 64,
    minimum_references: int = 8,
) -> dict[str, object]:
    """Choose scorer pairs in deterministic reference-first rounds."""

    if limit <= 0 or minimum_references <= 0:
        raise PerceptionDataError("scorer limits must be positive")
    by_reference: dict[str, list[dict[str, object]]] = {}
    identities = set()
    for row in pairs:
        reference = row.get("reference_id")
        pair_id = row.get("pair_id")
        if not isinstance(reference, str) or not reference:
            raise PerceptionDataError("scorer pair lacks reference_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise PerceptionDataError("scorer pair lacks pair_id")
        identity = (reference, pair_id)
        if identity in identities:
            raise PerceptionDataError("scorer pair identities must be unique")
        identities.add(identity)
        by_reference.setdefault(reference, []).append(dict(row))
    references = sorted(
        by_reference,
        key=lambda value: _hash_order("pgv1-scorer:", value),
    )
    for reference in references:
        by_reference[reference].sort(
            key=lambda row: _hash_order(
                f"pgv1-scorer:{reference}:", str(row["pair_id"])
            )
        )
    selected = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for reference in references:
            rows = by_reference[reference]
            if round_index < len(rows):
                selected.append(rows[round_index])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        round_index += 1
    reference_count = len(references)
    return {
        "pairs": tuple(selected),
        "reference_count": reference_count,
        "status": (
            "PASS"
            if reference_count >= minimum_references
            else "BLOCKED_TRAIN_COVERAGE"
        ),
    }


def segment_foreground_targets(
    *,
    point2segment: Tensor,
    semantic_labels: Tensor,
    thing_class_ids: Sequence[int],
    stuff_class_ids: Sequence[int],
    ignore_class_ids: Sequence[int],
) -> dict[str, Tensor]:
    """Build soft thing fractions without treating unknown labels as negatives."""

    if (
        not isinstance(point2segment, Tensor)
        or not isinstance(semantic_labels, Tensor)
        or point2segment.ndim != 1
        or semantic_labels.ndim != 1
        or point2segment.numel() != semantic_labels.numel()
    ):
        raise PerceptionDataError("segment labels require aligned vectors")
    if point2segment.is_floating_point() or semantic_labels.is_floating_point():
        raise PerceptionDataError("segment and semantic labels must be integers")
    point2segment = point2segment.long()
    semantic_labels = semantic_labels.long().to(point2segment.device)
    if point2segment.numel() == 0 or point2segment.min().item() < 0:
        raise PerceptionDataError("segment IDs must be non-empty and non-negative")
    segment_count = int(point2segment.max().item()) + 1
    if torch.any(torch.bincount(point2segment, minlength=segment_count) == 0).item():
        raise PerceptionDataError("segment IDs must form a contiguous namespace")
    thing = {int(value) for value in thing_class_ids}
    stuff = {int(value) for value in stuff_class_ids}
    ignored = {int(value) for value in ignore_class_ids}
    if thing & stuff or thing & ignored or stuff & ignored:
        raise PerceptionDataError("thing/stuff/ignore class sets must be disjoint")
    thing_tensor = torch.tensor(sorted(thing), device=semantic_labels.device)
    stuff_tensor = torch.tensor(sorted(stuff), device=semantic_labels.device)
    is_thing = torch.isin(semantic_labels, thing_tensor)
    is_known = is_thing | torch.isin(semantic_labels, stuff_tensor)
    valid_counts = torch.zeros(
        segment_count, dtype=torch.long, device=point2segment.device
    )
    thing_counts = torch.zeros_like(valid_counts)
    valid_counts.scatter_add_(0, point2segment, is_known.long())
    thing_counts.scatter_add_(0, point2segment, is_thing.long())
    valid = valid_counts > 0
    targets = torch.zeros(
        segment_count, dtype=torch.float32, device=point2segment.device
    )
    targets[valid] = thing_counts[valid].float() / valid_counts[valid].float()
    return {
        "targets": targets,
        "valid": valid,
        "valid_point_counts": valid_counts,
    }


def build_scorer_record(
    *,
    reference_id: str,
    pair_id: str,
    segment_features: Tensor,
    point2segment: Tensor,
    semantic_labels: Tensor,
    raw_coordinates: Tensor,
    temporal_stages: Tensor,
    thing_class_ids: Sequence[int],
    stuff_class_ids: Sequence[int],
    ignore_class_ids: Sequence[int],
) -> dict[str, object]:
    """Build one GT-isolated scorer-training record from a frozen R1 forward."""

    if not reference_id or not pair_id:
        raise PerceptionDataError("scorer record identity is empty")
    if (
        not isinstance(segment_features, Tensor)
        or segment_features.ndim != 2
        or segment_features.shape[1] != 128
        or not segment_features.is_floating_point()
        or not torch.isfinite(segment_features).all().item()
        or not isinstance(raw_coordinates, Tensor)
        or raw_coordinates.ndim != 2
        or raw_coordinates.shape[1] < 4
        or raw_coordinates.shape[0] != point2segment.numel()
        or not raw_coordinates.is_floating_point()
        or not torch.isfinite(raw_coordinates).all().item()
        or not isinstance(temporal_stages, Tensor)
        or temporal_stages.ndim != 1
        or temporal_stages.numel() != point2segment.numel()
    ):
        raise PerceptionDataError("scorer record tensor contract differs")
    labels = segment_foreground_targets(
        point2segment=point2segment,
        semantic_labels=semantic_labels,
        thing_class_ids=thing_class_ids,
        stuff_class_ids=stuff_class_ids,
        ignore_class_ids=ignore_class_ids,
    )
    from models.perception_gain import derive_segment_stage_ids

    point2segment = point2segment.long().to(raw_coordinates.device)
    stages = derive_segment_stage_ids(
        point2segment, temporal_stages.long().to(raw_coordinates.device)
    )
    segment_count = int(point2segment.max().item()) + 1
    if segment_features.shape[0] != segment_count:
        raise PerceptionDataError("scorer features and segment namespace differ")
    coordinate_sums = torch.zeros(
        (segment_count, raw_coordinates.shape[1]),
        dtype=raw_coordinates.dtype,
        device=raw_coordinates.device,
    )
    coordinate_sums.index_add_(0, point2segment, raw_coordinates)
    counts = torch.bincount(point2segment, minlength=segment_count).to(
        raw_coordinates.dtype
    )
    coordinates = coordinate_sums / counts[:, None]
    return {
        "reference_id": reference_id,
        "pair_id": pair_id,
        "segment_features": segment_features.detach().float().cpu().contiguous(),
        "segment_coordinates": coordinates.detach().float().cpu().contiguous(),
        "segment_stage_ids": stages.detach().long().cpu().contiguous(),
        "targets": labels["targets"].detach().float().cpu().contiguous(),
        "valid": labels["valid"].detach().bool().cpu().contiguous(),
    }


def sample_reference_segments(
    segments_by_reference: Mapping[str, Sequence[object]],
    *,
    count: int,
    seed: int,
    cursor: int = 0,
) -> tuple[tuple[str, object], ...]:
    """Produce an exact-resume reference-uniform deterministic sample stream."""

    if count < 0 or cursor < 0:
        raise PerceptionDataError("sample count and cursor must be non-negative")
    references = sorted(
        segments_by_reference,
        key=lambda value: _hash_order(f"pgv1-segment-ref:{seed}:", value),
    )
    if not references:
        raise PerceptionDataError("segment sampler has no references")
    for reference in references:
        values = segments_by_reference[reference]
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or not values
        ):
            raise PerceptionDataError("each sampled reference must contain segments")
    result = []
    for draw in range(cursor, cursor + count):
        reference = references[draw % len(references)]
        values = segments_by_reference[reference]
        digest = hashlib.sha256(
            f"pgv1-segment:{seed}:{reference}:{draw}".encode("utf-8")
        ).digest()
        index = int.from_bytes(digest[:8], "big") % len(values)
        result.append((reference, values[index]))
    return tuple(result)


def select_refiner_references(
    scan_ids_by_reference: Mapping[str, Sequence[str]],
    *,
    limit: int = 32,
    minimum_references: int = 8,
) -> dict[str, object]:
    if limit <= 0 or minimum_references <= 0:
        raise PerceptionDataError("refiner limits must be positive")
    eligible = []
    for reference, scans in scan_ids_by_reference.items():
        if not isinstance(reference, str) or not reference:
            raise PerceptionDataError("refiner reference IDs must be non-empty")
        if isinstance(scans, (str, bytes)) or not isinstance(scans, Sequence):
            raise PerceptionDataError("refiner scans must be a sequence")
        if len(set(scans)) >= 3:
            eligible.append(reference)
    eligible.sort(key=lambda value: _hash_order("pgv1-refiner:", value))
    references = tuple(eligible[:limit])
    return {
        "references": references,
        "reference_count": len(references),
        "status": (
            "PASS"
            if len(references) >= minimum_references
            else "BLOCKED_TRAIN_COVERAGE"
        ),
    }


_FORBIDDEN_INFERENCE_FIELDS = frozenset({"gt_ids", "gt_masks", "gt_classes"})


def _validate_no_gt(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_INFERENCE_FIELDS:
                raise PerceptionDataError(f"inference sidecar contains GT field {key}")
            _validate_no_gt(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _validate_no_gt(item)


def sidecar_cache_bytes(directory: str | Path) -> int:
    root = Path(directory)
    return sum(path.stat().st_size for path in root.glob("soft-*.pt") if path.is_file())


def write_soft_sidecar_shard(
    directory: str | Path,
    *,
    shard_id: int,
    records: Sequence[Mapping[str, object]],
    maximum_cache_bytes: int,
) -> dict[str, object]:
    if shard_id < 0 or maximum_cache_bytes <= 0:
        raise PerceptionDataError("shard ID and cache limit are invalid")
    if not records:
        raise PerceptionDataError("soft sidecar shard cannot be empty")
    _validate_no_gt(records)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"soft-{shard_id:06d}.pt"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".soft-", suffix=".tmp", dir=root
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(tuple(dict(record) for record in records), temporary)
        size = temporary.stat().st_size
        existing = sidecar_cache_bytes(root)
        if destination.is_file():
            existing -= destination.stat().st_size
        if existing + size > maximum_cache_bytes:
            raise PerceptionDataError("soft sidecar cache limit would be exceeded")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": destination.name,
        "bytes": size,
        "sha256": digest,
        "record_count": len(records),
    }


__all__ = [
    "PerceptionDataError",
    "build_data_roles",
    "build_scorer_record",
    "sample_reference_segments",
    "segment_foreground_targets",
    "select_refiner_references",
    "select_scorer_pairs",
    "sidecar_cache_bytes",
    "validate_role_disjointness",
    "write_soft_sidecar_shard",
]
