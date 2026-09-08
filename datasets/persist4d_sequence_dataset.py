"""Deterministic real-prefix episode wrapper for Persist4D All-T training."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS = (2, 3, 4, 5)
EPISODE_HORIZONS = (1, *HORIZONS)


class SequenceDatasetError(ValueError):
    """Raised when an episode crosses a frozen data or causal boundary."""


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SequenceDatasetError(f"{name} must be a non-empty string")
    return value


def _scan_indices(value: object, *, expected_length: int) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, tuple):
        raise SequenceDatasetError("scan_indices must be a tuple")
    if (
        len(value) != expected_length
        or len(set(value)) != len(value)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise SequenceDatasetError("scan_indices do not define a real unique prefix")
    return value


@dataclass(frozen=True)
class EpisodeMaster:
    reference_id: str
    sequence_id: str
    scan_indices: tuple[int, ...]
    role: str
    context_index: int

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, "reference_id")
        _nonempty(self.sequence_id, "sequence_id")
        _nonempty(self.role, "role")
        if len(self.scan_indices) not in {1, 5}:
            raise SequenceDatasetError("episode master must contain one or five scans")
        _scan_indices(self.scan_indices, expected_length=len(self.scan_indices))
        if (
            isinstance(self.context_index, bool)
            or not isinstance(self.context_index, int)
            or self.context_index < 0
        ):
            raise SequenceDatasetError("context_index must be non-negative")


@dataclass(frozen=True)
class EpisodeSpec:
    reference_id: str
    sequence_id: str
    scan_indices: tuple[int, ...]
    role: str
    context_index: int
    horizon: int
    augmentation_seed: int
    draw_index: int

    def __post_init__(self) -> None:
        _nonempty(self.reference_id, "reference_id")
        _nonempty(self.sequence_id, "sequence_id")
        _nonempty(self.role, "role")
        if self.horizon not in EPISODE_HORIZONS:
            raise SequenceDatasetError("episode horizon must be H1-H5")
        _scan_indices(self.scan_indices, expected_length=self.horizon)
        for name in ("context_index", "augmentation_seed", "draw_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SequenceDatasetError(f"{name} must be non-negative")

    @property
    def stage_scan_indices(self) -> tuple[tuple[int, ...], ...]:
        return (
            (self.scan_indices[0],),
            *(
                (self.scan_indices[stage - 1], self.scan_indices[stage])
                for stage in range(1, self.horizon)
            ),
        )


@dataclass(frozen=True)
class Persist4DEpisode:
    spec: EpisodeSpec
    samples: tuple[Any, ...]
    ambiguity_metadata: object


@dataclass(frozen=True)
class Persist4DEpisodeBatch:
    specs: tuple[EpisodeSpec, ...]
    stage_batches: tuple[Any, ...]
    ambiguity_metadata: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.specs or len({spec.horizon for spec in self.specs}) != 1:
            raise SequenceDatasetError("episode batch must use the same horizon")
        if len(self.stage_batches) != self.specs[0].horizon:
            raise SequenceDatasetError("episode batch stage count differs")
        if len(self.ambiguity_metadata) != len(self.specs):
            raise SequenceDatasetError("episode batch ambiguity metadata differs")


class Persist4DEpisodeCollator:
    """Collate spatial samples stage-by-stage while retaining episode time."""

    def __init__(self, stage_collator: Callable[[list[Any]], Any]) -> None:
        if not callable(stage_collator):
            raise SequenceDatasetError("stage_collator must be callable")
        self.stage_collator = stage_collator

    def __call__(self, episodes: list[Persist4DEpisode]) -> Persist4DEpisodeBatch:
        if not episodes or any(
            not isinstance(episode, Persist4DEpisode) for episode in episodes
        ):
            raise SequenceDatasetError("episode collator requires Persist4DEpisode values")
        horizons = {episode.spec.horizon for episode in episodes}
        if len(horizons) != 1:
            raise SequenceDatasetError("episode batch must use the same horizon")
        horizon = horizons.pop()
        return Persist4DEpisodeBatch(
            specs=tuple(episode.spec for episode in episodes),
            stage_batches=tuple(
                self.stage_collator(
                    [episode.samples[stage_index] for episode in episodes]
                )
                for stage_index in range(horizon)
            ),
            ambiguity_metadata=tuple(
                episode.ambiguity_metadata for episode in episodes
            ),
        )


def _stable_draw_seed(seed: int, draw_index: int) -> int:
    digest = hashlib.sha256(f"allt-augmentation:{seed}:{draw_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_episode_draw_plan(
    masters: list[EpisodeMaster] | tuple[EpisodeMaster, ...],
    *,
    role: str,
    episode_count: int,
    seed: int,
    replica_group_size: int = 1,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[EpisodeSpec, ...]:
    if not isinstance(masters, (list, tuple)) or any(
        not isinstance(master, EpisodeMaster) for master in masters
    ):
        raise SequenceDatasetError("masters must contain EpisodeMaster records")
    role = _nonempty(role, "role")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(replica_group_size, bool)
        or not isinstance(replica_group_size, int)
        or replica_group_size <= 0
        or episode_count % replica_group_size != 0
        or not isinstance(horizons, tuple)
        or not horizons
        or any(horizon not in HORIZONS for horizon in horizons)
        or len(set(horizons)) != len(horizons)
    ):
        raise SequenceDatasetError("episode count, seed, or replica group is invalid")
    selected = [master for master in masters if master.role == role]
    if not selected:
        raise SequenceDatasetError("requested data role has no masters")
    if any(len(master.scan_indices) != 5 for master in selected):
        raise SequenceDatasetError("multi-horizon draw plan requires five-scan masters")
    by_reference: dict[str, list[EpisodeMaster]] = {}
    for master in selected:
        by_reference.setdefault(master.reference_id, []).append(master)
    rng = random.Random(seed)
    references = sorted(by_reference)
    rng.shuffle(references)
    for values in by_reference.values():
        values.sort(key=lambda master: master.sequence_id)
        rng.shuffle(values)

    result = []
    for draw_index in range(episode_count):
        horizon = horizons[
            (draw_index // replica_group_size) % len(horizons)
        ]
        reference = references[draw_index % len(references)]
        reference_masters = by_reference[reference]
        master_round = draw_index // len(references)
        master = reference_masters[master_round % len(reference_masters)]
        result.append(
            EpisodeSpec(
                reference_id=master.reference_id,
                sequence_id=master.sequence_id,
                scan_indices=master.scan_indices[:horizon],
                role=master.role,
                context_index=master.context_index,
                horizon=horizon,
                augmentation_seed=_stable_draw_seed(seed, draw_index),
                draw_index=draw_index,
            )
        )
    return tuple(result)


def build_single_scan_draw_plan(
    masters: list[EpisodeMaster] | tuple[EpisodeMaster, ...],
    *,
    role: str,
    episode_count: int,
    seed: int,
    replica_group_size: int = 1,
) -> tuple[EpisodeSpec, ...]:
    if not isinstance(masters, (list, tuple)) or any(
        not isinstance(master, EpisodeMaster) for master in masters
    ):
        raise SequenceDatasetError("masters must contain EpisodeMaster records")
    role = _nonempty(role, "role")
    if (
        isinstance(episode_count, bool)
        or not isinstance(episode_count, int)
        or episode_count <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(replica_group_size, bool)
        or not isinstance(replica_group_size, int)
        or replica_group_size <= 0
        or episode_count % replica_group_size != 0
    ):
        raise SequenceDatasetError("episode count, seed, or replica group is invalid")
    selected = [master for master in masters if master.role == role]
    if not selected or any(len(master.scan_indices) != 1 for master in selected):
        raise SequenceDatasetError("single-scan draw plan requires H1 masters")
    ordered = sorted(selected, key=lambda master: master.sequence_id)
    random.Random(seed).shuffle(ordered)
    return tuple(
        EpisodeSpec(
            reference_id=master.reference_id,
            sequence_id=master.sequence_id,
            scan_indices=master.scan_indices,
            role=master.role,
            context_index=master.context_index,
            horizon=1,
            augmentation_seed=_stable_draw_seed(seed, draw_index),
            draw_index=draw_index,
        )
        for draw_index in range(episode_count)
        for master in (ordered[draw_index % len(ordered)],)
    )


def build_episode_masters(
    base_dataset: object,
    protocol_masters: Sequence[object],
    role_by_sequence: Mapping[str, str],
) -> tuple[EpisodeMaster, ...]:
    names = getattr(base_dataset, "sequence_names", None)
    indices = getattr(base_dataset, "sequence_indices", None)
    if not isinstance(names, (list, tuple)) or indices is None:
        raise SequenceDatasetError("base dataset lacks sequence identity tables")
    if len(names) != len(set(names)):
        raise SequenceDatasetError("base dataset sequence names are not unique")
    context_by_name = {name: index for index, name in enumerate(names)}
    if not isinstance(role_by_sequence, Mapping):
        raise SequenceDatasetError("role mapping must be a mapping")
    protocol_masters = tuple(protocol_masters)
    sequence_ids = {getattr(master, "sequence_id", None) for master in protocol_masters}
    if set(role_by_sequence) != sequence_ids:
        raise SequenceDatasetError("role mapping differs from protocol masters")
    result = []
    for master in protocol_masters:
        sequence_id = getattr(master, "sequence_id", None)
        reference_id = getattr(master, "reference_scene_id", None)
        scan_indices = getattr(master, "scan_indices", None)
        if sequence_id not in context_by_name:
            raise SequenceDatasetError("protocol master is absent from base dataset")
        context_index = context_by_name[sequence_id]
        try:
            base_indices = tuple(int(value) for value in indices[context_index])
            normalized_indices = tuple(int(value) for value in scan_indices)
        except (TypeError, ValueError) as error:
            raise SequenceDatasetError("protocol scan indices are invalid") from error
        if base_indices != normalized_indices:
            raise SequenceDatasetError("protocol and base scan indices differ")
        result.append(
            EpisodeMaster(
                reference_id=reference_id,
                sequence_id=sequence_id,
                scan_indices=normalized_indices,
                role=role_by_sequence[sequence_id],
                context_index=context_index,
            )
        )
    return tuple(result)


def build_single_scan_masters(
    base_dataset: object,
    *,
    role: str,
) -> tuple[EpisodeMaster, ...]:
    names = getattr(base_dataset, "sequence_names", None)
    indices = getattr(base_dataset, "sequence_indices", None)
    if not isinstance(names, (list, tuple)) or indices is None or not names:
        raise SequenceDatasetError("single-scan dataset lacks sequence identity tables")
    role = _nonempty(role, "role")
    result = []
    for context_index, sequence_id in enumerate(names):
        try:
            scan_indices = tuple(int(value) for value in indices[context_index])
        except (TypeError, ValueError) as error:
            raise SequenceDatasetError("single-scan base indices are invalid") from error
        if not scan_indices:
            raise SequenceDatasetError("single-scan base context is empty")
        result.append(
            EpisodeMaster(
                reference_id=str(sequence_id),
                sequence_id=str(sequence_id),
                scan_indices=(scan_indices[0],),
                role=role,
                context_index=context_index,
            )
        )
    return tuple(result)


def build_mixed_source_schedule(
    *,
    group_count: int,
    replica_group_size: int,
    primary_weight: float,
    secondary_weight: float,
) -> tuple[str, ...]:
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count <= 0
        or isinstance(replica_group_size, bool)
        or not isinstance(replica_group_size, int)
        or replica_group_size <= 0
    ):
        raise SequenceDatasetError("source schedule sizes must be positive integers")
    try:
        primary = Fraction(str(primary_weight))
        secondary = Fraction(str(secondary_weight))
    except (ValueError, ZeroDivisionError) as error:
        raise SequenceDatasetError("source weights must be positive finite values") from error
    if primary <= 0 or secondary <= 0:
        raise SequenceDatasetError("source weights must be positive finite values")
    primary_share = primary / (primary + secondary)
    group_sources = []
    primary_count = 0
    for group_index in range(group_count):
        desired_count = round(float(primary_share * (group_index + 1)))
        source = "rio" if desired_count > primary_count else "scannet"
        primary_count += source == "rio"
        group_sources.append(source)
    return tuple(
        source
        for source in group_sources
        for _ in range(replica_group_size)
    )


class Persist4DMixedEpisodeDataset(Dataset):
    """Consume explicit component draw plans under a rank-synchronous schedule."""

    def __init__(self, datasets: dict[str, object], source_schedule: tuple[str, ...]):
        if set(datasets) != {"rio", "scannet"}:
            raise SequenceDatasetError("mixed dataset requires rio and scannet sources")
        if not source_schedule or any(source not in datasets for source in source_schedule):
            raise SequenceDatasetError("mixed source schedule is invalid")
        counts = {source: source_schedule.count(source) for source in datasets}
        if any(len(datasets[source]) != count for source, count in counts.items()):
            raise SequenceDatasetError("component draw plan length differs from schedule")
        offsets = {source: 0 for source in datasets}
        indices = []
        for source in source_schedule:
            indices.append((source, offsets[source]))
            offsets[source] += 1
        self.datasets = dict(datasets)
        self.source_indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int):
        source, source_index = self.source_indices[index]
        return self.datasets[source][source_index]


@contextmanager
def _frozen_augmentation_seed(seed: int):
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


class Persist4DEpisodeDataset(Dataset):
    """Load T1 alone and later adjacent windows with one fixed random draw."""

    def __init__(
        self,
        base_dataset: object,
        draw_plan: list[EpisodeSpec] | tuple[EpisodeSpec, ...],
        *,
        window_mode: str = "local_pair",
    ):
        if not isinstance(draw_plan, (list, tuple)) or not draw_plan or any(
            not isinstance(spec, EpisodeSpec) for spec in draw_plan
        ):
            raise SequenceDatasetError("draw_plan must contain EpisodeSpec records")
        if len({spec.role for spec in draw_plan}) != 1:
            raise SequenceDatasetError("one episode dataset cannot mix data roles")
        if window_mode not in {"local_pair", "full_history"}:
            raise SequenceDatasetError("window_mode must be local_pair or full_history")
        names = getattr(base_dataset, "sequence_names", None)
        indices = getattr(base_dataset, "sequence_indices", None)
        loader = getattr(base_dataset, "load_scan_indices", None)
        if not isinstance(names, (list, tuple)) or not callable(loader) or indices is None:
            raise SequenceDatasetError("base dataset lacks sequence loading interfaces")
        for spec in draw_plan:
            if (
                spec.context_index >= len(names)
                or names[spec.context_index] != spec.sequence_id
            ):
                raise SequenceDatasetError("episode master context differs")
            try:
                base_indices = tuple(int(value) for value in indices[spec.context_index])
            except (IndexError, TypeError, ValueError) as error:
                raise SequenceDatasetError("base sequence indices are unavailable") from error
            if base_indices[: spec.horizon] != spec.scan_indices:
                raise SequenceDatasetError("episode scan prefix differs from base context")
        self.base_dataset = base_dataset
        self.draw_plan = tuple(draw_plan)
        self.window_mode = window_mode

    def __len__(self) -> int:
        return len(self.draw_plan)

    def __getitem__(self, index: int) -> Persist4DEpisode:
        spec = self.draw_plan[index]
        samples = []
        scan_windows = (
            spec.stage_scan_indices
            if self.window_mode == "local_pair"
            else tuple(
                spec.scan_indices[:stage]
                for stage in range(1, spec.horizon + 1)
            )
        )
        for scan_indices in scan_windows:
            with _frozen_augmentation_seed(spec.augmentation_seed):
                samples.append(
                    self.base_dataset.load_scan_indices(
                        spec.context_index,
                        scan_indices,
                        change_file=None,
                    )
                )
        ambiguities = getattr(self.base_dataset, "ambiguities", None)
        ambiguity = (
            ambiguities[spec.context_index]
            if isinstance(ambiguities, (list, tuple))
            and spec.context_index < len(ambiguities)
            else None
        )
        return Persist4DEpisode(
            spec=spec,
            samples=tuple(samples),
            ambiguity_metadata=ambiguity,
        )


__all__ = [
    "EpisodeMaster",
    "EpisodeSpec",
    "Persist4DEpisode",
    "Persist4DEpisodeBatch",
    "Persist4DEpisodeCollator",
    "Persist4DEpisodeDataset",
    "Persist4DMixedEpisodeDataset",
    "SequenceDatasetError",
    "build_episode_draw_plan",
    "build_episode_masters",
    "build_mixed_source_schedule",
    "build_single_scan_draw_plan",
    "build_single_scan_masters",
]
