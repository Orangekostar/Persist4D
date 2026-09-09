"""Native-length causal episodes and label-free TaskMemory stage metadata."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

HORIZONS = (1, 2, 3, 4, 5)
_SCAN_ID = re.compile(r"^scene(?P<scene>[0-9]{4})_(?P<sub_scene>[0-9]{2})$")


class TaskMemoryEpisodeError(ValueError):
    """Raised when an episode violates a native-data or causal boundary."""


def _nonempty(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TaskMemoryEpisodeError(f"{label} must be a non-empty string")
    return value


def _positive_index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TaskMemoryEpisodeError(f"{label} must be a non-negative integer")
    return value


def _scan_scene(scan_id: object) -> int:
    if not isinstance(scan_id, str):
        raise TaskMemoryEpisodeError("scan ID must be a string")
    match = _SCAN_ID.fullmatch(scan_id)
    if match is None:
        raise TaskMemoryEpisodeError(f"scan ID {scan_id!r} has invalid syntax")
    return int(match.group("scene"))


def _validate_scan_sequence(
    scan_ids: object, scan_indices: object
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if (
        isinstance(scan_ids, (str, bytes))
        or not isinstance(scan_ids, tuple)
        or isinstance(scan_indices, (str, bytes))
        or not isinstance(scan_indices, tuple)
    ):
        raise TaskMemoryEpisodeError("scan identities must be tuples")
    if not 1 <= len(scan_ids) <= 5 or len(scan_ids) != len(scan_indices):
        raise TaskMemoryEpisodeError("episode must contain one to five real scans")
    if len(set(scan_ids)) != len(scan_ids) or len(set(scan_indices)) != len(scan_indices):
        raise TaskMemoryEpisodeError("episode scans must be unique")
    scenes = {_scan_scene(scan_id) for scan_id in scan_ids}
    if len(scenes) != 1:
        raise TaskMemoryEpisodeError("episode scans must belong to the same scene")
    for index in scan_indices:
        _positive_index(index, label="scan index")
    return scan_ids, scan_indices


@dataclass(frozen=True)
class NativeEpisodeMaster:
    reference_id: str
    sequence_id: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    role: str
    context_index: int

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, label="reference_id")
        _nonempty(self.sequence_id, label="sequence_id")
        _nonempty(self.role, label="role")
        scan_ids, _ = _validate_scan_sequence(self.scan_ids, self.scan_indices)
        if self.sequence_id != "-".join(scan_ids):
            raise TaskMemoryEpisodeError("sequence_id differs from scan_ids")
        _positive_index(self.context_index, label="context_index")


@dataclass(frozen=True)
class TaskMemoryEpisodeSpec:
    reference_id: str
    source_sequence_id: str
    episode_id: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    role: str
    context_index: int
    horizon: int
    augmentation_seed: int
    draw_index: int
    bucket: str

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, label="reference_id")
        _nonempty(self.source_sequence_id, label="source_sequence_id")
        _nonempty(self.episode_id, label="episode_id")
        _nonempty(self.role, label="role")
        _nonempty(self.bucket, label="bucket")
        _validate_scan_sequence(self.scan_ids, self.scan_indices)
        if self.horizon not in HORIZONS or self.horizon != len(self.scan_ids):
            raise TaskMemoryEpisodeError("episode horizon must match H1-H5 scans")
        expected_bucket = "single_scan" if self.horizon == 1 else f"T{self.horizon}"
        if self.bucket != expected_bucket:
            raise TaskMemoryEpisodeError("episode bucket differs from horizon")
        _positive_index(self.context_index, label="context_index")
        _positive_index(self.augmentation_seed, label="augmentation_seed")
        _positive_index(self.draw_index, label="draw_index")

    @classmethod
    def from_master(
        cls,
        master: NativeEpisodeMaster,
        *,
        horizon: int,
        augmentation_seed: int,
        draw_index: int,
        bucket: str,
    ) -> TaskMemoryEpisodeSpec:
        if horizon not in HORIZONS or horizon > len(master.scan_ids):
            raise TaskMemoryEpisodeError("master cannot supply the requested horizon")
        scan_ids = master.scan_ids[:horizon]
        episode_digest = hashlib.sha256(
            (
                f"task-memory-v2:{master.reference_id}:{augmentation_seed}:"
                f"{draw_index}"
            ).encode("ascii")
        ).hexdigest()[:16]
        return cls(
            reference_id=master.reference_id,
            source_sequence_id=master.sequence_id,
            episode_id=f"tmv2-{episode_digest}",
            scan_ids=scan_ids,
            scan_indices=master.scan_indices[:horizon],
            role=master.role,
            context_index=master.context_index,
            horizon=horizon,
            augmentation_seed=augmentation_seed,
            draw_index=draw_index,
            bucket=bucket,
        )

    @property
    def stage_scan_ids(self) -> tuple[tuple[str, ...], ...]:
        return (
            (self.scan_ids[0],),
            *((self.scan_ids[index - 1], self.scan_ids[index]) for index in range(1, self.horizon)),
        )

    @property
    def stage_scan_indices(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.scan_indices[0],),
            *(
                (self.scan_indices[index - 1], self.scan_indices[index])
                for index in range(1, self.horizon)
            ),
        )


@dataclass(frozen=True)
class EpisodeTransform:
    matrix: np.ndarray
    color_gain: np.ndarray
    color_offset: np.ndarray
    transform_id: str


@dataclass(frozen=True)
class TaskMemoryStageSample:
    model_sample: tuple[Any, ...]
    scan_ids_in_window: tuple[str, ...]
    absolute_stage_index: int
    local_stage_ids: Tensor
    original_vertex_ids: tuple[Tensor, ...]
    scan_vertex_offsets: Tensor
    augmentation_transform_id: str
    coordinate_frame_id: str


@dataclass(frozen=True)
class TaskMemoryEpisode:
    spec: TaskMemoryEpisodeSpec
    stage_samples: tuple[TaskMemoryStageSample, ...]
    ambiguity_metadata: object


@dataclass(frozen=True)
class StageMeta:
    reference_id: str
    episode_id: str
    scan_ids_in_window: tuple[str, ...]
    absolute_stage_index: int
    local_stage_ids: Tensor
    original_vertex_ids: tuple[Tensor, ...]
    scan_vertex_offsets: Tensor
    point2segment: Tensor
    segment_stage_ids: Tensor
    augmentation_transform_id: str
    coordinate_frame_id: str
    voxel_inverse: Tensor
    full_resolution_point2segment: Tensor

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, label="reference_id")
        _nonempty(self.episode_id, label="episode_id")
        _nonempty(self.augmentation_transform_id, label="augmentation_transform_id")
        _nonempty(self.coordinate_frame_id, label="coordinate_frame_id")
        _positive_index(self.absolute_stage_index, label="absolute_stage_index")
        if not self.scan_ids_in_window or len(self.scan_ids_in_window) > 2:
            raise TaskMemoryEpisodeError("stage window must contain one or two scans")
        if len(self.original_vertex_ids) != len(self.scan_ids_in_window):
            raise TaskMemoryEpisodeError("vertex identity groups differ from stage scans")
        tensor_fields = (
            self.local_stage_ids,
            self.scan_vertex_offsets,
            self.point2segment,
            self.segment_stage_ids,
            self.voxel_inverse,
            self.full_resolution_point2segment,
        )
        if any(
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.dtype != torch.long
            for value in tensor_fields
        ):
            raise TaskMemoryEpisodeError("StageMeta index fields must be 1D int64 tensors")
        if (
            self.scan_vertex_offsets.numel() != len(self.scan_ids_in_window) + 1
            or self.scan_vertex_offsets[0].item() != 0
            or self.scan_vertex_offsets[-1].item() != self.local_stage_ids.numel()
            or torch.any(self.scan_vertex_offsets[1:] < self.scan_vertex_offsets[:-1])
        ):
            raise TaskMemoryEpisodeError("scan vertex offsets are inconsistent")
        expected_lengths = (
            self.scan_vertex_offsets[1:] - self.scan_vertex_offsets[:-1]
        ).tolist()
        if any(
            not isinstance(ids, Tensor)
            or ids.ndim != 1
            or ids.dtype != torch.long
            or ids.numel() != expected
            for ids, expected in zip(self.original_vertex_ids, expected_lengths)
        ):
            raise TaskMemoryEpisodeError("original vertex identities are inconsistent")
        full_size = self.local_stage_ids.numel()
        if (
            self.voxel_inverse.numel() != full_size
            or self.full_resolution_point2segment.numel() != full_size
        ):
            raise TaskMemoryEpisodeError("full-resolution mappings differ from vertices")
        if full_size:
            if self.point2segment.numel() == 0:
                raise TaskMemoryEpisodeError("nonempty stage has no voxel segments")
            if self.voxel_inverse.min().item() < 0 or self.voxel_inverse.max().item() >= self.point2segment.numel():
                raise TaskMemoryEpisodeError("voxel inverse is outside voxel rows")
            expected_full = self.point2segment[self.voxel_inverse]
            if not torch.equal(self.full_resolution_point2segment, expected_full):
                raise TaskMemoryEpisodeError("full-resolution segment map differs")
        if self.point2segment.numel():
            if self.point2segment.min().item() < 0:
                raise TaskMemoryEpisodeError("segment IDs must be non-negative")
            expected_segments = self.point2segment.max().item() + 1
            if self.segment_stage_ids.numel() != expected_segments:
                raise TaskMemoryEpisodeError("segment stage map does not cover all segments")


@dataclass(frozen=True)
class TaskMemoryStageBatch:
    model_batch: Any
    stage_meta: tuple[StageMeta, ...]
    training_identity_keys: tuple[tuple[tuple[str, int], ...], ...]


@dataclass(frozen=True)
class TaskMemoryEpisodeBatch:
    specs: tuple[TaskMemoryEpisodeSpec, ...]
    stage_batches: tuple[TaskMemoryStageBatch, ...]
    ambiguity_metadata: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.specs or len({spec.horizon for spec in self.specs}) != 1:
            raise TaskMemoryEpisodeError("episode batch must use one horizon")
        if len(self.stage_batches) != self.specs[0].horizon:
            raise TaskMemoryEpisodeError("episode batch stage count differs")
        if len(self.ambiguity_metadata) != len(self.specs):
            raise TaskMemoryEpisodeError("episode ambiguity metadata differs")


def _stable_seed(namespace: str, seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_native_episode_masters(
    base_dataset: object,
    *,
    reference_by_scene: Mapping[int, str],
    role_by_reference: Mapping[str, str],
) -> tuple[NativeEpisodeMaster, ...]:
    names = getattr(base_dataset, "sequence_names", None)
    indices = getattr(base_dataset, "sequence_indices", None)
    if not isinstance(names, (list, tuple)) or indices is None:
        raise TaskMemoryEpisodeError("base dataset lacks sequence identity tables")
    if len(names) != len(set(names)):
        raise TaskMemoryEpisodeError("base dataset sequence names are not unique")
    if not isinstance(reference_by_scene, Mapping) or not isinstance(
        role_by_reference, Mapping
    ):
        raise TaskMemoryEpisodeError("reference and role maps must be mappings")
    result = []
    for context_index, sequence_id in enumerate(names):
        if not isinstance(sequence_id, str):
            raise TaskMemoryEpisodeError("base sequence ID must be a string")
        scan_ids = tuple(sequence_id.split("-"))
        scenes = {_scan_scene(scan_id) for scan_id in scan_ids}
        if len(scenes) != 1:
            raise TaskMemoryEpisodeError("episode scans must belong to the same scene")
        scene = next(iter(scenes))
        reference = reference_by_scene.get(scene)
        if not isinstance(reference, str) or not reference:
            raise TaskMemoryEpisodeError(
                f"scene {scene} lacks a frozen reference assignment"
            )
        role = role_by_reference.get(reference)
        if not isinstance(role, str) or not role:
            raise TaskMemoryEpisodeError(
                f"reference {reference!r} lacks a frozen role"
            )
        try:
            scan_indices = tuple(int(value) for value in indices[context_index])
        except (IndexError, TypeError, ValueError) as error:
            raise TaskMemoryEpisodeError("base scan indices are invalid") from error
        result.append(
            NativeEpisodeMaster(
                reference_id=reference,
                sequence_id=sequence_id,
                scan_ids=scan_ids,
                scan_indices=scan_indices,
                role=role,
                context_index=context_index,
            )
        )
    return tuple(result)


def build_task_memory_draw_plan(
    masters: Sequence[NativeEpisodeMaster],
    *,
    role: str,
    episode_count: int,
    seed: int,
    replica_group_size: int = 1,
) -> tuple[TaskMemoryEpisodeSpec, ...]:
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence) or any(
        not isinstance(master, NativeEpisodeMaster) for master in masters
    ):
        raise TaskMemoryEpisodeError("masters must contain NativeEpisodeMaster records")
    role = _nonempty(role, label="role")
    _positive_index(seed, label="seed")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
        or isinstance(replica_group_size, bool)
        or not isinstance(replica_group_size, int)
        or replica_group_size <= 0
        or episode_count % (len(HORIZONS) * replica_group_size)
    ):
        raise TaskMemoryEpisodeError(
            "episode count must contain equal complete rank-synchronous H1-H5 groups"
        )
    selected = tuple(master for master in masters if master.role == role)
    if not selected:
        raise TaskMemoryEpisodeError("requested role has no native masters")
    eligible: dict[int, dict[str, list[NativeEpisodeMaster]]] = {}
    for horizon in HORIZONS:
        by_reference: dict[str, list[NativeEpisodeMaster]] = defaultdict(list)
        for master in selected:
            if len(master.scan_ids) >= horizon:
                by_reference[master.reference_id].append(master)
        if not by_reference:
            raise TaskMemoryEpisodeError(f"T{horizon} bucket has no real native episode")
        for reference, values in by_reference.items():
            values.sort(
                key=lambda master: hashlib.sha256(
                    f"task-memory-master:{seed}:{horizon}:{master.sequence_id}".encode(
                        "ascii"
                    )
                ).hexdigest()
            )
            if not reference:
                raise TaskMemoryEpisodeError("native reference ID is empty")
        eligible[horizon] = dict(by_reference)

    result = []
    group_count = episode_count // replica_group_size
    for group_index in range(group_count):
        horizon = HORIZONS[group_index % len(HORIZONS)]
        bucket = "single_scan" if horizon == 1 else f"T{horizon}"
        by_reference = eligible[horizon]
        references = sorted(
            by_reference,
            key=lambda reference: hashlib.sha256(
                f"task-memory-reference:{seed}:{horizon}:{reference}".encode("ascii")
            ).hexdigest(),
        )
        bucket_round = group_index // len(HORIZONS)
        for replica_index in range(replica_group_size):
            draw_index = group_index * replica_group_size + replica_index
            position = bucket_round * replica_group_size + replica_index
            reference = references[position % len(references)]
            choices = by_reference[reference]
            master = choices[(position // len(references)) % len(choices)]
            result.append(
                TaskMemoryEpisodeSpec.from_master(
                    master,
                    horizon=horizon,
                    augmentation_seed=_stable_seed("task-memory-augmentation", seed, draw_index),
                    draw_index=draw_index,
                    bucket=bucket,
                )
            )
    return tuple(result)


def _episode_transform(seed: int) -> EpisodeTransform:
    rng = random.Random(seed)
    angle = rng.uniform(-np.pi, np.pi)
    scale = rng.uniform(0.9, 1.1)
    reflection_x = -1.0 if rng.random() < 0.5 else 1.0
    reflection_y = -1.0 if rng.random() < 0.5 else 1.0
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    reflection = np.diag([reflection_x, reflection_y, 1.0]).astype(np.float32)
    matrix = scale * (rotation @ reflection)
    color_gain = np.asarray(
        [rng.uniform(0.9, 1.1) for _ in range(3)], dtype=np.float32
    )
    color_offset = np.asarray(
        [rng.uniform(-5.0, 5.0) for _ in range(3)], dtype=np.float32
    )
    payload = {
        "color_gain": [round(float(value), 9) for value in color_gain],
        "color_offset": [round(float(value), 9) for value in color_offset],
        "matrix": [round(float(value), 9) for value in matrix.reshape(-1)],
        "seed": seed,
    }
    transform_id = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")
    ).hexdigest()
    return EpisodeTransform(
        matrix=matrix,
        color_gain=color_gain,
        color_offset=color_offset,
        transform_id=transform_id,
    )


@contextmanager
def _frozen_random_seed(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


@contextmanager
def _legacy_raw_mode(base_dataset: object):
    mode = getattr(base_dataset, "mode", None)
    if not isinstance(mode, str):
        raise TaskMemoryEpisodeError("base dataset lacks a string mode")
    base_dataset.mode = "task_memory_raw"
    try:
        yield
    finally:
        base_dataset.mode = mode


def _apply_transform(sample: object, transform: EpisodeTransform) -> tuple[Any, ...]:
    if not isinstance(sample, tuple) or len(sample) != 9:
        raise TaskMemoryEpisodeError("legacy loader must return its nine-field sample")
    coordinates = np.asarray(sample[0]).copy()
    features = np.asarray(sample[1]).copy()
    labels = np.asarray(sample[2]).copy()
    raw_color = np.asarray(sample[4]).copy()
    normals = np.asarray(sample[5]).copy()
    raw_coordinates = np.asarray(sample[6]).copy()
    if (
        coordinates.ndim != 2
        or raw_coordinates.ndim != 2
        or coordinates.shape[0] != raw_coordinates.shape[0]
        or coordinates.shape[1] not in {3, 4}
        or raw_coordinates.shape[1] not in {3, 4}
        or labels.ndim != 2
        or labels.shape[0] != coordinates.shape[0]
        or labels.shape[1] < 3
    ):
        raise TaskMemoryEpisodeError("legacy scan sample shapes are incompatible")
    coordinates[:, :3] = coordinates[:, :3] @ transform.matrix.T
    raw_coordinates[:, :3] = raw_coordinates[:, :3] @ transform.matrix.T
    raw_color[:, :3] = np.clip(
        raw_color[:, :3] * transform.color_gain + transform.color_offset,
        0.0,
        255.0,
    )
    if features.ndim == 2 and features.shape[0] == coordinates.shape[0] and features.shape[1] >= 3:
        features[:, :3] = features[:, :3] * transform.color_gain
    if normals.ndim == 2 and normals.shape == (coordinates.shape[0], 3):
        normal_matrix = transform.matrix / np.linalg.norm(transform.matrix[:, 0])
        normals = normals @ normal_matrix.T
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
    return (
        coordinates,
        features,
        labels,
        sample[3],
        raw_color,
        normals,
        raw_coordinates,
        sample[7],
        sample[8],
    )


def _renumber_segments(labels: np.ndarray, start: int) -> tuple[np.ndarray, int]:
    result = labels.copy()
    segments = result[:, -1]
    unique = np.unique(segments)
    if unique.size and unique[0] < 0:
        raise TaskMemoryEpisodeError("legacy segment IDs must be non-negative")
    for position, segment in enumerate(unique):
        result[segments == segment, -1] = start + position
    return result, start + len(unique)


def _compose_stage_sample(
    *,
    spec: TaskMemoryEpisodeSpec,
    absolute_stage_index: int,
    scan_positions: tuple[int, ...],
    transformed_scans: Mapping[int, tuple[Any, ...]],
    transform_id: str,
) -> TaskMemoryStageSample:
    arrays: dict[int, list[np.ndarray]] = {index: [] for index in (0, 1, 2, 4, 5, 6)}
    original_vertex_ids = []
    offsets = [0]
    local_stages = []
    segment_start = 0
    for local_stage, scan_position in enumerate(scan_positions):
        sample = transformed_scans[scan_position]
        point_count = np.asarray(sample[0]).shape[0]
        labels, segment_start = _renumber_segments(np.asarray(sample[2]), segment_start)
        coordinates = np.asarray(sample[0]).copy()
        raw_coordinates = np.asarray(sample[6]).copy()
        if coordinates.shape[1] == 3:
            coordinates = np.column_stack(
                (coordinates, np.full(point_count, local_stage, dtype=np.float32))
            )
        else:
            coordinates[:, 3] = local_stage
        if raw_coordinates.shape[1] == 3:
            raw_coordinates = np.column_stack(
                (raw_coordinates, np.full(point_count, local_stage, dtype=np.float32))
            )
        else:
            raw_coordinates[:, 3] = local_stage
        arrays[0].append(coordinates)
        arrays[1].append(np.asarray(sample[1]).copy())
        arrays[2].append(labels)
        arrays[4].append(np.asarray(sample[4]).copy())
        arrays[5].append(np.asarray(sample[5]).copy())
        arrays[6].append(raw_coordinates)
        original_vertex_ids.append(torch.arange(point_count, dtype=torch.long))
        offsets.append(offsets[-1] + point_count)
        local_stages.append(torch.full((point_count,), local_stage, dtype=torch.long))
    scan_ids = tuple(spec.scan_ids[position] for position in scan_positions)
    model_sample = (
        np.concatenate(arrays[0], axis=0),
        np.concatenate(arrays[1], axis=0),
        np.concatenate(arrays[2], axis=0),
        "-".join(scan_ids),
        np.concatenate(arrays[4], axis=0),
        np.concatenate(arrays[5], axis=0),
        np.concatenate(arrays[6], axis=0),
        spec.context_index,
        transformed_scans[scan_positions[-1]][8],
    )
    return TaskMemoryStageSample(
        model_sample=model_sample,
        scan_ids_in_window=scan_ids,
        absolute_stage_index=absolute_stage_index,
        local_stage_ids=torch.cat(local_stages),
        original_vertex_ids=tuple(original_vertex_ids),
        scan_vertex_offsets=torch.tensor(offsets, dtype=torch.long),
        augmentation_transform_id=transform_id,
        coordinate_frame_id=f"reference:{spec.reference_id}:transform:{transform_id}",
    )


class TaskMemoryEpisodeDataset(Dataset):
    """Load each scan once, then construct causal T1/W2 stage windows."""

    def __init__(
        self,
        base_dataset: object,
        draw_plan: Sequence[TaskMemoryEpisodeSpec],
    ) -> None:
        if isinstance(draw_plan, (str, bytes)) or not isinstance(draw_plan, Sequence) or not draw_plan or any(
            not isinstance(spec, TaskMemoryEpisodeSpec) for spec in draw_plan
        ):
            raise TaskMemoryEpisodeError(
                "draw_plan must contain TaskMemoryEpisodeSpec records"
            )
        names = getattr(base_dataset, "sequence_names", None)
        indices = getattr(base_dataset, "sequence_indices", None)
        loader = getattr(base_dataset, "load_scan_indices", None)
        if not isinstance(names, (list, tuple)) or indices is None or not callable(loader):
            raise TaskMemoryEpisodeError("base dataset lacks sequence loading interfaces")
        if getattr(base_dataset, "max_points_per_sample", None) is not None:
            raise TaskMemoryEpisodeError(
                "max_points_per_sample would destroy original vertex identity"
            )
        for spec in draw_plan:
            try:
                base_name = names[spec.context_index]
                base_indices = tuple(int(value) for value in indices[spec.context_index])
            except (IndexError, TypeError, ValueError) as error:
                raise TaskMemoryEpisodeError("base sequence context is unavailable") from error
            if base_name != spec.source_sequence_id or base_indices[: spec.horizon] != spec.scan_indices:
                raise TaskMemoryEpisodeError("draw plan differs from base sequence context")
        self.base_dataset = base_dataset
        self.draw_plan = tuple(draw_plan)

    def __len__(self) -> int:
        return len(self.draw_plan)

    def __getitem__(self, index: int) -> TaskMemoryEpisode:
        spec = self.draw_plan[index]
        transform = _episode_transform(spec.augmentation_seed)
        transformed_scans = {}
        for scan_position, (scan_id, scan_index) in enumerate(
            zip(spec.scan_ids, spec.scan_indices)
        ):
            load_seed = _stable_seed(
                f"task-memory-scan:{scan_id}", spec.augmentation_seed, 0
            )
            substitutions_before = getattr(
                self.base_dataset, "known_empty_scan_substitution_count", None
            )
            with _legacy_raw_mode(self.base_dataset), _frozen_random_seed(load_seed):
                sample = self.base_dataset.load_scan_indices(
                    spec.context_index, (scan_index,), change_file=None
                )
            substitutions_after = getattr(
                self.base_dataset, "known_empty_scan_substitution_count", None
            )
            if (
                substitutions_before is not None
                and substitutions_after != substitutions_before
            ):
                raise TaskMemoryEpisodeError("scan substitution is forbidden")
            transformed_scans[scan_position] = _apply_transform(sample, transform)
        stage_samples = []
        for stage_index in range(spec.horizon):
            positions = (0,) if stage_index == 0 else (stage_index - 1, stage_index)
            stage_samples.append(
                _compose_stage_sample(
                    spec=spec,
                    absolute_stage_index=stage_index,
                    scan_positions=positions,
                    transformed_scans=transformed_scans,
                    transform_id=transform.transform_id,
                )
            )
        return TaskMemoryEpisode(
            spec=spec,
            stage_samples=tuple(stage_samples),
            ambiguity_metadata=transformed_scans[0][8],
        )


def _point_field(point: object, name: str) -> object:
    if isinstance(point, Mapping):
        if name not in point:
            raise TaskMemoryEpisodeError(f"collated point lacks {name}")
        return point[name]
    try:
        return getattr(point, name)
    except AttributeError as error:
        raise TaskMemoryEpisodeError(f"collated point lacks {name}") from error


def _clone_model_sample(sample: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        value.copy()
        if isinstance(value, np.ndarray)
        else value.clone()
        if isinstance(value, Tensor)
        else deepcopy(value)
        for value in sample
    )


def _batch_sequence(value: object, *, name: str, size: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TaskMemoryEpisodeError(f"collated {name} must be a sequence")
    if len(value) != size:
        raise TaskMemoryEpisodeError(f"collated {name} batch size differs")
    return value


def _segment_stage_ids(point2segment: Tensor, temporal_stages: Tensor) -> Tensor:
    if point2segment.ndim != 1 or temporal_stages.ndim != 1 or point2segment.numel() != temporal_stages.numel():
        raise TaskMemoryEpisodeError("voxel segment and temporal rows differ")
    if point2segment.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    point2segment = point2segment.long().cpu()
    temporal_stages = temporal_stages.long().cpu()
    if point2segment.min().item() < 0:
        raise TaskMemoryEpisodeError("segment IDs must be non-negative")
    result = torch.empty(point2segment.max().item() + 1, dtype=torch.long)
    for segment in range(result.numel()):
        stages = temporal_stages[point2segment == segment].unique()
        if stages.numel() != 1:
            raise TaskMemoryEpisodeError("one segment spans multiple stages")
        result[segment] = stages[0]
    return result


def _canonical_identity_keys(
    target: Mapping[str, object], reference_id: str
) -> tuple[tuple[str, int], ...]:
    raw_ids = target.get("ids")
    if raw_ids is None:
        return ()
    if isinstance(raw_ids, Tensor):
        values = raw_ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(raw_ids, np.ndarray):
        values = raw_ids.reshape(-1).tolist()
    elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
        values = list(raw_ids)
    else:
        raise TaskMemoryEpisodeError("training target ids are invalid")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TaskMemoryEpisodeError("canonical instance ID must be an integer")
        normalized.append((reference_id, int(value)))
    if len(normalized) != len(set(normalized)):
        raise TaskMemoryEpisodeError("training target has duplicate canonical identities")
    return tuple(normalized)


def _build_stage_meta(
    *,
    point: object,
    targets: Sequence[Mapping[str, object]],
    stage_samples: Sequence[TaskMemoryStageSample],
    specs: Sequence[TaskMemoryEpisodeSpec],
) -> tuple[tuple[StageMeta, ...], tuple[tuple[tuple[str, int], ...], ...]]:
    size = len(stage_samples)
    inverse_maps = _batch_sequence(
        _point_field(point, "inverse_maps"), name="inverse_maps", size=size
    )
    labels = _batch_sequence(_point_field(point, "labels"), name="labels", size=size)
    temporal_stages = _batch_sequence(
        _point_field(point, "temporal_stages"), name="temporal_stages", size=size
    )
    if len(targets) != size:
        raise TaskMemoryEpisodeError("collated training target batch size differs")
    metas = []
    identity_keys = []
    for batch_index, (stage_sample, spec, target) in enumerate(
        zip(stage_samples, specs, targets)
    ):
        if not isinstance(target, Mapping):
            raise TaskMemoryEpisodeError("training target must be a mapping")
        raw_point2segment = target.get("point2segment")
        if raw_point2segment is None:
            label_tensor = labels[batch_index]
            if not isinstance(label_tensor, Tensor) or label_tensor.ndim != 2:
                raise TaskMemoryEpisodeError("collated labels cannot supply segments")
            raw_point2segment = label_tensor[:, -1]
        if not isinstance(raw_point2segment, Tensor):
            raise TaskMemoryEpisodeError("point2segment must be a tensor")
        inverse = inverse_maps[batch_index]
        temporal = temporal_stages[batch_index]
        if not isinstance(inverse, Tensor) or not isinstance(temporal, Tensor):
            raise TaskMemoryEpisodeError("inverse and temporal maps must be tensors")
        point2segment = raw_point2segment.detach().cpu().long().reshape(-1)
        inverse = inverse.detach().cpu().long().reshape(-1)
        temporal = temporal.detach().cpu().long().reshape(-1)
        full_point2segment = point2segment[inverse]
        meta = StageMeta(
            reference_id=spec.reference_id,
            episode_id=spec.episode_id,
            scan_ids_in_window=stage_sample.scan_ids_in_window,
            absolute_stage_index=stage_sample.absolute_stage_index,
            local_stage_ids=stage_sample.local_stage_ids.clone(),
            original_vertex_ids=tuple(
                ids.clone() for ids in stage_sample.original_vertex_ids
            ),
            scan_vertex_offsets=stage_sample.scan_vertex_offsets.clone(),
            point2segment=point2segment,
            segment_stage_ids=_segment_stage_ids(point2segment, temporal),
            augmentation_transform_id=stage_sample.augmentation_transform_id,
            coordinate_frame_id=stage_sample.coordinate_frame_id,
            voxel_inverse=inverse,
            full_resolution_point2segment=full_point2segment,
        )
        metas.append(meta)
        identity_keys.append(_canonical_identity_keys(target, spec.reference_id))
    return tuple(metas), tuple(identity_keys)


class TaskMemoryEpisodeCollator:
    """Collate episode stages and attach metadata outside model supervision."""

    def __init__(self, stage_collator: Callable[[list[Any]], Any]) -> None:
        if not callable(stage_collator):
            raise TaskMemoryEpisodeError("stage_collator must be callable")
        self.stage_collator = stage_collator

    def __call__(self, episodes: list[TaskMemoryEpisode]) -> TaskMemoryEpisodeBatch:
        if not episodes or any(
            not isinstance(episode, TaskMemoryEpisode) for episode in episodes
        ):
            raise TaskMemoryEpisodeError(
                "episode collator requires TaskMemoryEpisode values"
            )
        horizons = {episode.spec.horizon for episode in episodes}
        if len(horizons) != 1:
            raise TaskMemoryEpisodeError("episode batch must use one horizon")
        horizon = next(iter(horizons))
        specs = tuple(episode.spec for episode in episodes)
        stage_batches = []
        for stage_index in range(horizon):
            stage_samples = tuple(
                episode.stage_samples[stage_index] for episode in episodes
            )
            collate_identity = ":".join(
                sample.augmentation_transform_id for sample in stage_samples
            )
            collate_seed = int.from_bytes(
                hashlib.sha256(
                    f"task-memory-collate:{stage_index}:{collate_identity}".encode(
                        "ascii"
                    )
                ).digest()[:8],
                "big",
            ) % (2**32)
            with _frozen_random_seed(collate_seed):
                model_batch = self.stage_collator(
                    [_clone_model_sample(sample.model_sample) for sample in stage_samples]
                )
            if (
                not isinstance(model_batch, tuple)
                or len(model_batch) != 3
                or isinstance(model_batch[1], (str, bytes))
                or not isinstance(model_batch[1], Sequence)
            ):
                raise TaskMemoryEpisodeError(
                    "stage collator must return (point, targets, names)"
                )
            metas, identity_keys = _build_stage_meta(
                point=model_batch[0],
                targets=model_batch[1],
                stage_samples=stage_samples,
                specs=specs,
            )
            stage_batches.append(
                TaskMemoryStageBatch(
                    model_batch=model_batch,
                    stage_meta=metas,
                    training_identity_keys=identity_keys,
                )
            )
        return TaskMemoryEpisodeBatch(
            specs=specs,
            stage_batches=tuple(stage_batches),
            ambiguity_metadata=tuple(
                episode.ambiguity_metadata for episode in episodes
            ),
        )


__all__ = [
    "NativeEpisodeMaster",
    "StageMeta",
    "TaskMemoryEpisode",
    "TaskMemoryEpisodeBatch",
    "TaskMemoryEpisodeCollator",
    "TaskMemoryEpisodeDataset",
    "TaskMemoryEpisodeError",
    "TaskMemoryEpisodeSpec",
    "TaskMemoryStageBatch",
    "TaskMemoryStageSample",
    "build_native_episode_masters",
    "build_task_memory_draw_plan",
]
