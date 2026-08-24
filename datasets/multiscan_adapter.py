"""Strict MultiScan metadata and no-GT-leakage adapter contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np

_SCAN_ID = re.compile(r"(scene_\d{5})_(\d{2})\Z")
_OFFICIAL_SPLITS = frozenset({"train", "val", "test", "unassigned"})
_SELECTED_RULE = "all physical scenes with >= 3 scans"
_STRUCTURAL_CLASSES = frozenset({"wall", "floor", "ceiling"})
_OBJECT_LABEL = re.compile(r"(.+)\.(\d+)\Z")
_FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "class_ids",
        "gt_correspondence",
        "gt_obb",
        "instance_gt",
        "instance_ids",
        "mobilityType",
        "mobility_label",
        "objectId",
        "object_id",
        "partId",
        "part_id",
        "semantic_gt",
        "stable_object_ids",
    }
)


class MultiScanAdapterError(ValueError):
    """Raised when official MultiScan data violates the frozen contract."""


@dataclass(frozen=True)
class MultiScanObjectAnnotation:
    object_id: int
    label: str
    class_name: str
    mobility_type: str | None
    eligible: bool


@dataclass(frozen=True)
class MultiScanAnnotation:
    scan_id: str
    objects: tuple[MultiScanObjectAnnotation, ...]


@dataclass(frozen=True)
class MultiScanInstancePayload:
    scan_id: str
    payload_keys: tuple[str, ...]
    inst2obj_id: Mapping[int, int]
    inst2obj: Mapping[int, str]


@dataclass(frozen=True)
class MultiScanIdentityRecord:
    scene_id: str
    scan_id: str
    local_instance_id: int
    object_id: int
    object_label: str
    class_name: str
    eligible: bool

    @property
    def identity_key(self) -> tuple[str, int]:
        return self.scene_id, self.object_id


def _readonly_array(value: object, *, dtype: np.dtype, name: str) -> np.ndarray:
    try:
        result = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    except (TypeError, ValueError) as error:
        raise MultiScanAdapterError(f"{name} cannot be converted to {dtype}") from error
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise MultiScanAdapterError(f"{name} contains non-finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MultiScanInferenceInput:
    xyz: np.ndarray
    normals: np.ndarray
    rgb: np.ndarray
    geometric_segment_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "xyz": _readonly_array(self.xyz, dtype=np.float32, name="xyz"),
            "normals": _readonly_array(self.normals, dtype=np.float32, name="normals"),
            "rgb": _readonly_array(self.rgb, dtype=np.uint8, name="rgb"),
            "geometric_segment_ids": _readonly_array(
                self.geometric_segment_ids,
                dtype=np.int64,
                name="geometric_segment_ids",
            ),
        }
        point_count = arrays["xyz"].shape[0]
        expected_shapes = {
            "xyz": (point_count, 3),
            "normals": (point_count, 3),
            "rgb": (point_count, 3),
            "geometric_segment_ids": (point_count,),
        }
        if point_count <= 0:
            raise MultiScanAdapterError("MultiScan inference input must not be empty")
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise MultiScanAdapterError(
                    f"{name} must have shape {expected_shape}, got {arrays[name].shape}"
                )
            object.__setattr__(self, name, arrays[name])

    def as_mapping(self) -> Mapping[str, np.ndarray]:
        payload = {
            "xyz": self.xyz,
            "normals": self.normals,
            "rgb": self.rgb,
            "geometric_segment_ids": self.geometric_segment_ids,
        }
        assert_no_gt_leakage(payload)
        return MappingProxyType(payload)


@dataclass(frozen=True)
class MultiScanEvaluatorTarget:
    scene_id: str
    scan_id: str
    class_ids: np.ndarray
    instance_ids: np.ndarray
    stable_object_ids: np.ndarray

    def __post_init__(self) -> None:
        observed_scene, _ = parse_multiscan_scan_id(self.scan_id)
        if observed_scene != self.scene_id:
            raise MultiScanAdapterError("evaluator target cannot mix physical scenes")
        arrays = {
            "class_ids": _readonly_array(
                self.class_ids, dtype=np.int32, name="class_ids"
            ),
            "instance_ids": _readonly_array(
                self.instance_ids, dtype=np.int32, name="instance_ids"
            ),
            "stable_object_ids": _readonly_array(
                self.stable_object_ids, dtype=np.int32, name="stable_object_ids"
            ),
        }
        lengths = {array.shape for array in arrays.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == (0,):
            raise MultiScanAdapterError(
                "evaluator target arrays must have equal length"
            )
        if any(array.ndim != 1 for array in arrays.values()):
            raise MultiScanAdapterError(
                "evaluator target arrays must be one-dimensional"
            )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


def assert_no_gt_leakage(value: object) -> None:
    """Reject evaluator-only MultiScan ground truth in model inputs."""
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_INFERENCE_KEYS.intersection(value)
        if forbidden:
            raise MultiScanAdapterError(
                "ground-truth leakage in inference payload: "
                + ", ".join(sorted(forbidden))
            )
        for child in value.values():
            assert_no_gt_leakage(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_gt_leakage(child)


def parse_multiscan_scan_id(value: object) -> tuple[str, int]:
    """Parse an exact official ``scene_xxxxx_xx`` identifier."""
    if not isinstance(value, str):
        raise MultiScanAdapterError("MultiScan scan ID must be a string")
    match = _SCAN_ID.fullmatch(value)
    if match is None:
        raise MultiScanAdapterError(f"invalid MultiScan scan ID: {value!r}")
    return match.group(1), int(match.group(2))


def _regular_file(path: str | Path, *, name: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MultiScanAdapterError(f"{name} must be a regular file: {source}")
    return source


def _object_label(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise MultiScanAdapterError("MultiScan object label must be non-empty")
    match = _OBJECT_LABEL.fullmatch(value)
    if match is None or not match.group(1):
        raise MultiScanAdapterError(f"invalid MultiScan object label: {value!r}")
    return value, match.group(1).lower()


def _eligible_object(label: str, class_name: str) -> bool:
    return not label.lower().startswith("remove") and class_name not in (
        _STRUCTURAL_CLASSES
    )


def read_multiscan_annotation(
    path: str | Path,
    *,
    expected_scan_id: str | None = None,
) -> MultiScanAnnotation:
    """Read explicit object IDs from one released annotations JSON file."""
    source = _regular_file(path, name="MultiScan annotation")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiScanAdapterError(
            f"cannot parse MultiScan annotation: {source}"
        ) from error
    if not isinstance(document, dict):
        raise MultiScanAdapterError("MultiScan annotation must be a JSON object")
    scan_id = document.get("scanId")
    parse_multiscan_scan_id(scan_id)
    if expected_scan_id is not None and scan_id != expected_scan_id:
        raise MultiScanAdapterError("MultiScan annotation scanId differs from expected")
    if source.name != f"{scan_id}.annotations.json":
        raise MultiScanAdapterError("MultiScan annotation filename differs from scanId")
    raw_objects = document.get("objects")
    if not isinstance(raw_objects, list):
        raise MultiScanAdapterError("MultiScan annotation objects must be a list")
    objects = []
    seen_ids: set[int] = set()
    for raw_object in raw_objects:
        if not isinstance(raw_object, dict):
            raise MultiScanAdapterError("MultiScan annotation object must be a mapping")
        object_id = raw_object.get("objectId")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, Integral)
            or object_id <= 0
        ):
            raise MultiScanAdapterError(
                "MultiScan annotation requires an explicit positive objectId"
            )
        normalized_id = int(object_id)
        if normalized_id in seen_ids:
            raise MultiScanAdapterError(
                f"duplicate annotation objectId in {scan_id}: {normalized_id}"
            )
        seen_ids.add(normalized_id)
        label, class_name = _object_label(raw_object.get("label"))
        mobility_type = raw_object.get("mobilityType", raw_object.get("type"))
        if mobility_type is not None and not isinstance(mobility_type, str):
            raise MultiScanAdapterError("MultiScan mobility type must be a string")
        objects.append(
            MultiScanObjectAnnotation(
                object_id=normalized_id,
                label=label,
                class_name=class_name,
                mobility_type=mobility_type,
                eligible=_eligible_object(label, class_name),
            )
        )
    if not objects:
        raise MultiScanAdapterError(f"MultiScan annotation has no objects: {scan_id}")
    return MultiScanAnnotation(scan_id=scan_id, objects=tuple(objects))


def _integer_mapping(value: object, *, name: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise MultiScanAdapterError(f"released payload {name} must be a mapping")
    result: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        if any(
            isinstance(item, bool) or not isinstance(item, Integral)
            for item in (raw_key, raw_value)
        ):
            raise MultiScanAdapterError(
                f"released payload {name} must contain integers"
            )
        key, mapped = int(raw_key), int(raw_value)
        if key < 0 or mapped <= 0 or key in result:
            raise MultiScanAdapterError(f"released payload {name} contains invalid IDs")
        result[key] = mapped
    if not result:
        raise MultiScanAdapterError(f"released payload {name} must not be empty")
    return result


def _label_mapping(value: object, *, name: str) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise MultiScanAdapterError(f"released payload {name} must be a mapping")
    result: dict[int, str] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, Integral):
            raise MultiScanAdapterError(
                f"released payload {name} keys must be integers"
            )
        key = int(raw_key)
        label, _ = _object_label(raw_value)
        if key < 0 or key in result:
            raise MultiScanAdapterError(f"released payload {name} contains invalid IDs")
        result[key] = label
    if not result:
        raise MultiScanAdapterError(f"released payload {name} must not be empty")
    return result


def inspect_multiscan_instance_payload(path: str | Path) -> MultiScanInstancePayload:
    """Inspect a trusted official benchmark PTH and require identity metadata."""
    source = _regular_file(path, name="MultiScan instance payload")
    try:
        import torch

        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        raise MultiScanAdapterError(
            f"cannot load trusted MultiScan instance payload: {source}"
        ) from error
    if not isinstance(payload, Mapping):
        raise MultiScanAdapterError("released MultiScan payload must be a mapping")
    if "inst2obj_id" not in payload:
        raise MultiScanAdapterError("released payload lacks inst2obj_id")
    if "inst2obj" not in payload:
        raise MultiScanAdapterError("released payload lacks inst2obj")
    scan_id = source.stem
    parse_multiscan_scan_id(scan_id)
    inst2obj_id = _integer_mapping(payload["inst2obj_id"], name="inst2obj_id")
    inst2obj = _label_mapping(payload["inst2obj"], name="inst2obj")
    if set(inst2obj_id) != set(inst2obj):
        raise MultiScanAdapterError("released inst2obj_id and inst2obj keys differ")
    return MultiScanInstancePayload(
        scan_id=scan_id,
        payload_keys=tuple(sorted(str(key) for key in payload)),
        inst2obj_id=MappingProxyType(inst2obj_id),
        inst2obj=MappingProxyType(inst2obj),
    )


def build_multiscan_identity_records(
    *,
    scene_id: str,
    annotations: Sequence[MultiScanAnnotation],
    payloads: Sequence[MultiScanInstancePayload],
) -> tuple[MultiScanIdentityRecord, ...]:
    """Cross-check local instance maps against explicit released annotations."""
    if re.fullmatch(r"scene_\d{5}", scene_id) is None:
        raise MultiScanAdapterError(f"invalid MultiScan scene ID: {scene_id!r}")
    if not annotations or len(annotations) != len(payloads):
        raise MultiScanAdapterError(
            "identity audit requires paired annotations/payloads"
        )
    annotations_by_scan = {annotation.scan_id: annotation for annotation in annotations}
    payloads_by_scan = {payload.scan_id: payload for payload in payloads}
    if len(annotations_by_scan) != len(annotations) or len(payloads_by_scan) != len(
        payloads
    ):
        raise MultiScanAdapterError("identity audit scan IDs must be unique")
    if set(annotations_by_scan) != set(payloads_by_scan):
        raise MultiScanAdapterError("annotation and payload scan IDs differ")
    labels_by_identity: dict[int, str] = {}
    records = []
    for scan_id in sorted(annotations_by_scan, key=parse_multiscan_scan_id):
        observed_scene, _ = parse_multiscan_scan_id(scan_id)
        if observed_scene != scene_id:
            raise MultiScanAdapterError("identity audit cannot mix physical scenes")
        annotation = annotations_by_scan[scan_id]
        payload = payloads_by_scan[scan_id]
        objects = {obj.object_id: obj for obj in annotation.objects}
        for local_instance_id, object_id in sorted(payload.inst2obj_id.items()):
            obj = objects.get(object_id)
            if obj is None:
                raise MultiScanAdapterError(
                    f"payload object ID lacks annotation objectId: {scan_id}/{object_id}"
                )
            label = payload.inst2obj[local_instance_id]
            if label != obj.label:
                raise MultiScanAdapterError(
                    f"payload and annotation labels differ: {scan_id}/{object_id}"
                )
            prior_label = labels_by_identity.setdefault(object_id, label)
            if prior_label != label:
                raise MultiScanAdapterError(
                    f"cross-scan objectId label conflict: {scene_id}/{object_id}"
                )
            records.append(
                MultiScanIdentityRecord(
                    scene_id=scene_id,
                    scan_id=scan_id,
                    local_instance_id=local_instance_id,
                    object_id=object_id,
                    object_label=label,
                    class_name=obj.class_name,
                    eligible=obj.eligible,
                )
            )
    return tuple(records)


def detect_natural_gaps(
    *,
    scene_id: str,
    scan_ids: Sequence[str],
    objects_by_scan: Sequence[Sequence[MultiScanObjectAnnotation]],
) -> list[dict[str, object]]:
    """Detect maximal visible-absent-visible episodes for eligible objects."""
    normalized_scan_ids = tuple(scan_ids)
    if not normalized_scan_ids or len(normalized_scan_ids) != len(objects_by_scan):
        raise MultiScanAdapterError("gap audit requires one object list per scan")
    indices = []
    for scan_id in normalized_scan_ids:
        observed_scene, index = parse_multiscan_scan_id(scan_id)
        if observed_scene != scene_id:
            raise MultiScanAdapterError("gap audit cannot mix physical scenes")
        indices.append(index)
    if len(set(normalized_scan_ids)) != len(normalized_scan_ids) or indices != sorted(
        indices
    ):
        raise MultiScanAdapterError(
            "gap audit scan order must be unique and increasing"
        )

    visible_by_identity: dict[int, list[int]] = defaultdict(list)
    class_by_identity: dict[int, str] = {}
    for scan_index, raw_objects in enumerate(objects_by_scan):
        seen: set[int] = set()
        for obj in raw_objects:
            if not isinstance(obj, MultiScanObjectAnnotation):
                raise MultiScanAdapterError("gap audit objects must be annotations")
            if obj.object_id in seen:
                raise MultiScanAdapterError("duplicate objectId within one scan")
            seen.add(obj.object_id)
            eligible = obj.eligible and _eligible_object(obj.label, obj.class_name)
            if not eligible:
                continue
            prior_class = class_by_identity.setdefault(obj.object_id, obj.class_name)
            if prior_class != obj.class_name:
                raise MultiScanAdapterError("cross-scan objectId class conflict")
            visible_by_identity[obj.object_id].append(scan_index)

    gaps = []
    for object_id in sorted(visible_by_identity):
        for left, right in pairwise(visible_by_identity[object_id]):
            gap_length = right - left - 1
            if gap_length <= 0:
                continue
            gaps.append(
                {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "class": class_by_identity[object_id],
                    "last_visible_before_gap": normalized_scan_ids[left],
                    "first_visible_after_gap": normalized_scan_ids[right],
                    "gap_length": gap_length,
                }
            )
    return gaps


def _selected_scene_list_sha256(scenes: list[dict[str, object]]) -> str:
    selected = [scene for scene in scenes if int(scene["number_of_scans"]) >= 3]
    payload = (
        json.dumps(
            selected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_multiscan_inventory(path: str | Path) -> dict[str, object]:
    """Build the official physical-scene inventory and frozen T>=3 subset."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MultiScanAdapterError("MultiScan split CSV must be a regular file")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != {"scanId", "split"}:
            raise MultiScanAdapterError(
                "MultiScan split CSV must contain exact official columns"
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise MultiScanAdapterError("MultiScan split CSV must not be empty")

    seen: set[str] = set()
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for row in raw_rows:
        scan_id = row.get("scanId")
        scene_id, scan_index = parse_multiscan_scan_id(scan_id)
        if scan_id in seen:
            raise MultiScanAdapterError(f"duplicate MultiScan scan ID: {scan_id}")
        seen.add(scan_id)
        raw_split = row.get("split")
        if not isinstance(raw_split, str):
            raise MultiScanAdapterError("official split must be a string")
        split = raw_split.strip() or "unassigned"
        if split not in _OFFICIAL_SPLITS:
            raise MultiScanAdapterError(f"unknown official split: {split!r}")
        grouped[scene_id].append((scan_index, scan_id, split))
        split_counts[split] += 1

    scenes: list[dict[str, object]] = []
    for scene_id in sorted(grouped):
        rows = sorted(grouped[scene_id])
        indices = [row[0] for row in rows]
        if indices != list(range(len(rows))):
            raise MultiScanAdapterError(
                f"scan indices must be contiguous for {scene_id}"
            )
        splits = {row[2] for row in rows}
        if len(splits) != 1:
            raise MultiScanAdapterError(
                f"physical scene has mixed official split values: {scene_id}"
            )
        scenes.append(
            {
                "scene_id": scene_id,
                "scan_ids": [row[1] for row in rows],
                "official_split": next(iter(splits)),
                "number_of_scans": len(rows),
            }
        )

    length_distribution = Counter(int(scene["number_of_scans"]) for scene in scenes)
    selected_scenes = [scene for scene in scenes if int(scene["number_of_scans"]) >= 3]
    return {
        "schema_version": 1,
        "status": "pass",
        "scan_count": len(seen),
        "scene_count": len(scenes),
        "split_scan_counts": {
            split: split_counts[split]
            for split in ("train", "val", "test", "unassigned")
        },
        "temporal_length_distribution": {
            str(length): count for length, count in sorted(length_distribution.items())
        },
        "threshold_scene_counts": {
            str(threshold): sum(
                int(scene["number_of_scans"]) >= threshold for scene in scenes
            )
            for threshold in (3, 4, 5)
        },
        "selected_rule": _SELECTED_RULE,
        "selected_scene_count": len(selected_scenes),
        "selected_scan_count": sum(
            int(scene["number_of_scans"]) for scene in selected_scenes
        ),
        "selected_scene_ids": [str(scene["scene_id"]) for scene in selected_scenes],
        "selected_scan_ids": [
            str(scan_id) for scene in selected_scenes for scan_id in scene["scan_ids"]
        ],
        "selected_scene_list_sha256": _selected_scene_list_sha256(scenes),
        "scenes": scenes,
    }


__all__ = [
    "MultiScanAdapterError",
    "MultiScanAnnotation",
    "MultiScanEvaluatorTarget",
    "MultiScanIdentityRecord",
    "MultiScanInferenceInput",
    "MultiScanInstancePayload",
    "MultiScanObjectAnnotation",
    "assert_no_gt_leakage",
    "build_multiscan_identity_records",
    "build_multiscan_inventory",
    "detect_natural_gaps",
    "inspect_multiscan_instance_payload",
    "parse_multiscan_scan_id",
    "read_multiscan_annotation",
]
