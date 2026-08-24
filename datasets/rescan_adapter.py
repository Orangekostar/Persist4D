"""Strict ReScan parsing and no-GT-leakage adapter contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
from plyfile import PlyData

_PLY_FIELDS = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "red",
    "green",
    "blue",
    "radius",
    "class_idx",
    "instance_idx",
)
_FORBIDDEN_INFERENCE_KEYS = frozenset(
    {
        "ambiguities",
        "class_idx",
        "class_ids",
        "instance_idx",
        "instance_ids",
        "labels",
        "object_transform",
        "stable_identity",
        "target_full",
    }
)


class RescanAdapterError(ValueError):
    """Raised when official ReScan data violates the frozen adapter contract."""


def _readonly_array(value: object, *, dtype: np.dtype, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise RescanAdapterError(f"{name} cannot be converted to {dtype}") from error
    result = np.ascontiguousarray(array).copy()
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise RescanAdapterError(f"{name} contains non-finite values")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RescanPointCloud:
    xyz: np.ndarray
    normals: np.ndarray
    rgb: np.ndarray
    radius: np.ndarray
    class_ids: np.ndarray
    instance_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "xyz": _readonly_array(self.xyz, dtype=np.float32, name="xyz"),
            "normals": _readonly_array(self.normals, dtype=np.float32, name="normals"),
            "rgb": _readonly_array(self.rgb, dtype=np.uint8, name="rgb"),
            "radius": _readonly_array(self.radius, dtype=np.float32, name="radius"),
            "class_ids": _readonly_array(
                self.class_ids, dtype=np.int32, name="class_ids"
            ),
            "instance_ids": _readonly_array(
                self.instance_ids, dtype=np.int32, name="instance_ids"
            ),
        }
        point_count = arrays["xyz"].shape[0]
        expected_shapes = {
            "xyz": (point_count, 3),
            "normals": (point_count, 3),
            "rgb": (point_count, 3),
            "radius": (point_count,),
            "class_ids": (point_count,),
            "instance_ids": (point_count,),
        }
        if point_count <= 0:
            raise RescanAdapterError("ReScan point cloud must not be empty")
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise RescanAdapterError(
                    f"{name} must have shape {expected}, got {arrays[name].shape}"
                )
            object.__setattr__(self, name, arrays[name])


@dataclass(frozen=True)
class RescanCapture:
    scene_id: str
    capture_id: str
    temporal_index: int
    ply_path: Path
    ambiguity_path: Path | None


@dataclass(frozen=True)
class RescanScene:
    scene_id: str
    captures: tuple[RescanCapture, ...]


@dataclass(frozen=True)
class RescanInventory:
    root: Path
    scenes: tuple[RescanScene, ...]


@dataclass(frozen=True)
class IdentityAlternatives:
    alternatives: Mapping[int, tuple[int, ...]]

    def __post_init__(self) -> None:
        normalized: dict[int, tuple[int, ...]] = {}
        for base, values in self.alternatives.items():
            if isinstance(base, bool) or not isinstance(base, int) or base < 0:
                raise RescanAdapterError("ambiguity base identity must be non-negative")
            accepted = tuple(values)
            if (
                not accepted
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in accepted
                )
                or len(set(accepted)) != len(accepted)
            ):
                raise RescanAdapterError("ambiguity alternatives are invalid")
            if base not in accepted:
                raise RescanAdapterError(
                    "ambiguity alternatives must contain their base identity"
                )
            normalized[base] = accepted
        object.__setattr__(self, "alternatives", MappingProxyType(normalized))

    def accepts(self, ground_truth_id: int, predicted_id: int) -> bool:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (ground_truth_id, predicted_id)
        ):
            raise RescanAdapterError("identity ids must be non-negative integers")
        accepted = self.alternatives.get(ground_truth_id, (ground_truth_id,))
        return predicted_id in accepted


@dataclass(frozen=True)
class RescanInferenceInput:
    xyz: np.ndarray
    normals: np.ndarray
    rgb: np.ndarray
    geometric_segment_ids: np.ndarray

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
class RescanEvaluatorTarget:
    scene_id: str
    capture_id: str
    class_ids: np.ndarray
    instance_ids: np.ndarray
    identity_keys: tuple[tuple[str, int], ...]
    ambiguities: IdentityAlternatives


@dataclass(frozen=True)
class RescanInputTargetSplit:
    inference: RescanInferenceInput
    target: RescanEvaluatorTarget


def geometric_voxel_segments(
    xyz: object,
    *,
    voxel_size_m: float = 0.1,
) -> np.ndarray:
    """Build deterministic geometry-only segments without semantic or instance GT."""

    if (
        isinstance(voxel_size_m, bool)
        or not isinstance(voxel_size_m, (int, float))
        or not np.isfinite(voxel_size_m)
        or voxel_size_m <= 0
    ):
        raise RescanAdapterError("voxel_size_m must be a positive finite number")
    coordinates = np.asarray(xyz, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or coordinates.shape[0] == 0
        or not np.isfinite(coordinates).all()
    ):
        raise RescanAdapterError("xyz must have finite shape [N, 3]")
    grid = np.floor(
        (coordinates - coordinates.min(axis=0, keepdims=True)) / float(voxel_size_m)
    ).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)
    return _readonly_array(inverse, dtype=np.int64, name="geometric_segment_ids")


def read_rescan_ply(path: str | Path) -> RescanPointCloud:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RescanAdapterError(f"ReScan PLY is not a regular file: {source}")
    try:
        document = PlyData.read(source)
    except Exception as error:
        raise RescanAdapterError(f"cannot parse ReScan PLY: {source}") from error
    if document.text or document.byte_order != "<":
        raise RescanAdapterError("ReScan PLY must be binary little endian")
    try:
        vertex = document["vertex"].data
    except (KeyError, TypeError) as error:
        raise RescanAdapterError("ReScan PLY lacks a vertex element") from error
    names = set(vertex.dtype.names or ())
    missing = [name for name in _PLY_FIELDS if name not in names]
    if missing:
        raise RescanAdapterError(
            "ReScan PLY lacks required fields: " + ", ".join(missing)
        )
    return RescanPointCloud(
        xyz=np.column_stack([vertex[name] for name in ("x", "y", "z")]),
        normals=np.column_stack([vertex[name] for name in ("nx", "ny", "nz")]),
        rgb=np.column_stack([vertex[name] for name in ("red", "green", "blue")]),
        radius=vertex["radius"],
        class_ids=vertex["class_idx"],
        instance_ids=vertex["instance_idx"],
    )


def discover_rescan_dataset(root: str | Path) -> RescanInventory:
    dataset_root = Path(root)
    scene_list = dataset_root / "scene_list.txt"
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise RescanAdapterError("ReScan root must be a directory")
    if scene_list.is_symlink() or not scene_list.is_file():
        raise RescanAdapterError("ReScan root lacks scene_list.txt")
    scene_ids = tuple(
        line.strip()
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise RescanAdapterError("scene_list.txt must contain unique scene ids")
    scenes = []
    for scene_id in scene_ids:
        if (
            Path(scene_id).name != scene_id
            or re.fullmatch(r"[A-Za-z0-9_]+", scene_id) is None
        ):
            raise RescanAdapterError(f"invalid scene id: {scene_id!r}")
        segmentation = dataset_root / scene_id / "gt_segmentation"
        if segmentation.is_symlink() or not segmentation.is_dir():
            raise RescanAdapterError(f"scene lacks gt_segmentation: {scene_id}")
        expression = re.compile(rf"{re.escape(scene_id)}_(\d+)\.ply\Z")
        indexed: dict[int, Path] = {}
        for path in segmentation.glob("*.ply"):
            match = expression.fullmatch(path.name)
            if match is None or path.is_symlink() or not path.is_file():
                raise RescanAdapterError(f"invalid capture PLY: {path.name}")
            temporal_index = int(match.group(1))
            if temporal_index in indexed:
                raise RescanAdapterError("duplicate capture temporal index")
            indexed[temporal_index] = path
        if not indexed:
            raise RescanAdapterError(f"scene has no capture PLY files: {scene_id}")
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise RescanAdapterError(
                f"scene capture indices are not contiguous: {scene_id}"
            )
        captures = []
        for temporal_index in expected:
            path = indexed[temporal_index]
            ambiguity_path = path.with_suffix(".txt")
            captures.append(
                RescanCapture(
                    scene_id=scene_id,
                    capture_id=path.stem,
                    temporal_index=temporal_index,
                    ply_path=path,
                    ambiguity_path=(
                        ambiguity_path if ambiguity_path.is_file() else None
                    ),
                )
            )
        scenes.append(RescanScene(scene_id=scene_id, captures=tuple(captures)))
    return RescanInventory(root=dataset_root, scenes=tuple(scenes))


def parse_rescan_ambiguities(path: str | Path) -> IdentityAlternatives:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RescanAdapterError(f"ambiguity file is not a regular file: {source}")
    alternatives: dict[int, tuple[int, ...]] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.replace("|", " ").split()
        try:
            values = tuple(int(field) for field in fields)
        except ValueError as error:
            raise RescanAdapterError(
                f"ambiguity row {line_number} contains a non-integer"
            ) from error
        if len(values) < 2:
            raise RescanAdapterError(f"ambiguity row {line_number} lacks alternatives")
        base, accepted = values[0], values[1:]
        if base in alternatives:
            raise RescanAdapterError(f"duplicate ambiguity base identity: {base}")
        alternatives[base] = accepted
    return IdentityAlternatives(alternatives)


def split_inference_and_evaluation(
    cloud: RescanPointCloud,
    *,
    scene_id: str,
    capture_id: str,
    ambiguities: IdentityAlternatives | None = None,
) -> RescanInputTargetSplit:
    if not isinstance(cloud, RescanPointCloud):
        raise RescanAdapterError("cloud must be a RescanPointCloud")
    if not scene_id or not capture_id:
        raise RescanAdapterError("scene_id and capture_id must be non-empty")
    segment_ids = geometric_voxel_segments(cloud.xyz)
    inference = RescanInferenceInput(
        xyz=cloud.xyz,
        normals=cloud.normals,
        rgb=cloud.rgb,
        geometric_segment_ids=segment_ids,
    )
    target = RescanEvaluatorTarget(
        scene_id=scene_id,
        capture_id=capture_id,
        class_ids=cloud.class_ids,
        instance_ids=cloud.instance_ids,
        identity_keys=tuple(
            (scene_id, int(instance_id)) for instance_id in cloud.instance_ids
        ),
        ambiguities=ambiguities or IdentityAlternatives({}),
    )
    assert_no_gt_leakage(inference.as_mapping())
    return RescanInputTargetSplit(inference=inference, target=target)


class RescanTemporalDataset:
    """Minimal ReScene-compatible view with evaluator GT kept out of samples."""

    def __init__(
        self,
        root: str | Path,
        *,
        geometry_segment_size_m: float = 0.1,
    ) -> None:
        self.inventory = discover_rescan_dataset(root)
        if (
            isinstance(geometry_segment_size_m, bool)
            or not isinstance(geometry_segment_size_m, (int, float))
            or not np.isfinite(geometry_segment_size_m)
            or geometry_segment_size_m <= 0
        ):
            raise RescanAdapterError(
                "geometry_segment_size_m must be a positive finite number"
            )
        self.geometry_segment_size_m = float(geometry_segment_size_m)
        self._captures = tuple(
            capture for scene in self.inventory.scenes for capture in scene.captures
        )
        sequence_indices = []
        offset = 0
        for scene in self.inventory.scenes:
            stop = offset + len(scene.captures)
            sequence_indices.append(tuple(range(offset, stop)))
            offset = stop
        self.sequence_indices = tuple(sequence_indices)
        self.sequence_names = tuple(scene.scene_id for scene in self.inventory.scenes)
        self.label_offset = 0
        self.dataset_name = "rescan"
        self.data_dir = (self.inventory.root,)

    def __len__(self) -> int:
        return len(self.sequence_indices)

    @property
    def captures(self) -> tuple[RescanCapture, ...]:
        return self._captures

    def __getitem__(self, index: int) -> tuple[object, ...]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise RescanAdapterError("dataset index must be an integer")
        context_index = int(index)
        if not 0 <= context_index < len(self):
            raise IndexError("dataset index is outside the scene range")
        return self.load_scan_indices(
            context_index,
            self.sequence_indices[context_index],
            change_file=None,
        )

    def _normalize_request(
        self,
        context_index: int,
        scan_indices: object,
    ) -> tuple[int, ...]:
        if isinstance(context_index, bool) or not isinstance(context_index, Integral):
            raise RescanAdapterError("context index must be an integer")
        context = int(context_index)
        if not 0 <= context < len(self.sequence_indices):
            raise IndexError("context index is outside the scene range")
        if isinstance(scan_indices, (str, bytes)):
            raise RescanAdapterError("scan indices must be a non-empty sequence")
        try:
            values = tuple(scan_indices)
        except TypeError as error:
            raise RescanAdapterError(
                "scan indices must be a non-empty sequence"
            ) from error
        if not values:
            raise RescanAdapterError("scan indices must be a non-empty sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise RescanAdapterError("scan indices must contain integers")
        normalized = tuple(int(value) for value in values)
        if len(set(normalized)) != len(normalized):
            raise RescanAdapterError("scan indices must not contain duplicates")
        allowed = set(self.sequence_indices[context])
        if any(value not in allowed for value in normalized):
            raise RescanAdapterError("scan indices must remain within one scene")
        if tuple(sorted(normalized)) != normalized:
            raise RescanAdapterError("scan indices must preserve official order")
        return normalized

    def load_scan_indices(
        self,
        context_idx: int,
        scan_indices: object,
        *,
        change_file: object = None,
    ) -> tuple[object, ...]:
        if change_file is not None:
            raise RescanAdapterError("ReScan has no model-input change labels")
        normalized = self._normalize_request(context_idx, scan_indices)
        coordinates = []
        colors = []
        normals = []
        segments = []
        segment_offset = 0
        for local_stage, scan_index in enumerate(normalized):
            cloud = read_rescan_ply(self._captures[scan_index].ply_path)
            stage_coordinates = np.column_stack(
                (
                    cloud.xyz,
                    np.full(cloud.xyz.shape[0], local_stage, dtype=np.float32),
                )
            ).astype(np.float32, copy=False)
            stage_segments = geometric_voxel_segments(
                cloud.xyz,
                voxel_size_m=self.geometry_segment_size_m,
            ).astype(np.int64, copy=True)
            stage_segments += segment_offset
            segment_offset = int(stage_segments.max()) + 1
            coordinates.append(stage_coordinates)
            colors.append(cloud.rgb.astype(np.float32) / 255.0)
            normals.append(cloud.normals.astype(np.float32, copy=True))
            segments.append(stage_segments)
        all_coordinates = np.concatenate(coordinates, axis=0)
        all_colors = np.concatenate(colors, axis=0)
        all_normals = np.concatenate(normals, axis=0)
        all_segments = np.concatenate(segments, axis=0)
        features = np.column_stack(
            (all_coordinates[:, :3], all_colors, all_normals)
        ).astype(np.float32, copy=False)
        inference_contract = {
            "xyz": all_coordinates[:, :3],
            "rgb": all_colors,
            "normals": all_normals,
            "geometric_segment_ids": all_segments,
        }
        assert_no_gt_leakage(inference_contract)
        placeholder_labels = np.column_stack(
            (
                np.full(all_segments.shape[0], 2, dtype=np.int64),
                np.zeros(all_segments.shape[0], dtype=np.int64),
                np.zeros(all_segments.shape[0], dtype=np.int64),
                all_segments,
            )
        )
        return (
            all_coordinates,
            features,
            placeholder_labels,
            self.sequence_names[int(context_idx)],
            all_colors,
            all_normals,
            all_coordinates.copy(),
            int(context_idx),
            None,
        )

    def evaluator_targets(
        self,
        scan_indices: object,
    ) -> tuple[RescanEvaluatorTarget, ...]:
        if isinstance(scan_indices, (str, bytes)):
            raise RescanAdapterError("scan indices must be a sequence")
        try:
            normalized = tuple(int(value) for value in scan_indices)
        except (TypeError, ValueError) as error:
            raise RescanAdapterError("scan indices must contain integers") from error
        if not normalized or any(
            value < 0 or value >= len(self._captures) for value in normalized
        ):
            raise RescanAdapterError("scan indices are outside the capture range")
        targets = []
        for scan_index in normalized:
            capture = self._captures[scan_index]
            cloud = read_rescan_ply(capture.ply_path)
            ambiguity = (
                parse_rescan_ambiguities(capture.ambiguity_path)
                if capture.ambiguity_path is not None
                else IdentityAlternatives({})
            )
            targets.append(
                RescanEvaluatorTarget(
                    scene_id=capture.scene_id,
                    capture_id=capture.capture_id,
                    class_ids=cloud.class_ids,
                    instance_ids=cloud.instance_ids,
                    identity_keys=tuple(
                        (capture.scene_id, int(value)) for value in cloud.instance_ids
                    ),
                    ambiguities=ambiguity,
                )
            )
        if len({target.scene_id for target in targets}) != 1:
            raise RescanAdapterError("evaluator targets must remain within one scene")
        return tuple(targets)


def assert_no_gt_leakage(value: object) -> None:
    """Reject evaluator-only object GT anywhere in a model-input structure."""

    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_INFERENCE_KEYS.intersection(value)
        if forbidden:
            raise RescanAdapterError(
                "ground-truth leakage in inference payload: "
                + ", ".join(sorted(forbidden))
            )
        for child in value.values():
            assert_no_gt_leakage(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_no_gt_leakage(child)


__all__ = [
    "IdentityAlternatives",
    "RescanAdapterError",
    "RescanCapture",
    "RescanEvaluatorTarget",
    "RescanInferenceInput",
    "RescanInputTargetSplit",
    "RescanInventory",
    "RescanPointCloud",
    "RescanScene",
    "RescanTemporalDataset",
    "assert_no_gt_leakage",
    "discover_rescan_dataset",
    "geometric_voxel_segments",
    "parse_rescan_ambiguities",
    "read_rescan_ply",
    "split_inference_and_evaluation",
]
