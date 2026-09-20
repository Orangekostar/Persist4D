"""Bounded resident and lag-one state for CrossWindow association."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class CrossWindowStateError(ValueError):
    """Raised when bounded CrossWindow state violates its runtime contract."""


GroupKey = tuple[str, str, str, int, int]


def _tensor_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def _valid_probability_rows(value: Tensor) -> bool:
    return bool(
        torch.isfinite(value).all().item()
        and not torch.any(value < 0).item()
        and not torch.any(value.sum(dim=-1) > 1.0 + 1e-5).item()
    )


@dataclass(frozen=True)
class CrossWindowState:
    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    occupied: Tensor
    active: Tensor
    age: Tensor
    last_seen: Tensor
    logical_ids: Tensor
    generations: Tensor
    next_logical_id: int
    stage_watermark: int
    episode_id: str | None

    @classmethod
    def empty(
        cls,
        *,
        capacity: int,
        feature_dim: int,
        class_count: int,
        device: torch.device | str = "cpu",
    ) -> CrossWindowState:
        for name, value in (
            ("capacity", capacity),
            ("feature_dim", feature_dim),
            ("class_count", class_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CrossWindowStateError(f"{name} must be a positive integer")
        state = cls(
            features=torch.zeros(
                capacity, feature_dim, dtype=torch.float32, device=device
            ),
            class_prob=torch.zeros(
                capacity, class_count, dtype=torch.float32, device=device
            ),
            confidence=torch.zeros(capacity, dtype=torch.float32, device=device),
            occupied=torch.zeros(capacity, dtype=torch.bool, device=device),
            active=torch.zeros(capacity, dtype=torch.bool, device=device),
            age=torch.zeros(capacity, dtype=torch.long, device=device),
            last_seen=torch.full((capacity,), -1, dtype=torch.long, device=device),
            logical_ids=torch.full((capacity,), -1, dtype=torch.long, device=device),
            generations=torch.full((capacity,), -1, dtype=torch.long, device=device),
            next_logical_id=0,
            stage_watermark=-1,
            episode_id=None,
        )
        state.validate()
        return state

    @property
    def capacity(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    @property
    def class_count(self) -> int:
        return self.class_prob.shape[1]

    @property
    def occupied_count(self) -> int:
        return int(self.occupied.sum().item())

    @property
    def state_bytes(self) -> int:
        tensors = (
            self.features,
            self.class_prob,
            self.confidence,
            self.occupied,
            self.active,
            self.age,
            self.last_seen,
            self.logical_ids,
            self.generations,
        )
        return sum(_tensor_bytes(value) for value in tensors) + 16

    def validate(self) -> None:
        if not isinstance(self.features, Tensor) or self.features.ndim != 2:
            raise CrossWindowStateError("features must have shape [K,D]")
        capacity, feature_dim = self.features.shape
        if capacity <= 0 or feature_dim <= 0 or self.features.dtype != torch.float32:
            raise CrossWindowStateError("features must be non-empty float32 [K,D]")
        if not isinstance(self.class_prob, Tensor) or self.class_prob.ndim != 2:
            raise CrossWindowStateError("class_prob must have shape [K,C]")
        if self.class_prob.shape[0] != capacity or self.class_prob.shape[1] <= 0:
            raise CrossWindowStateError("class_prob dimensions differ from state")
        if self.class_prob.dtype != torch.float32:
            raise CrossWindowStateError("class_prob must use float32")
        for name, value, dtype in (
            ("confidence", self.confidence, torch.float32),
            ("occupied", self.occupied, torch.bool),
            ("active", self.active, torch.bool),
            ("age", self.age, torch.long),
            ("last_seen", self.last_seen, torch.long),
            ("logical_ids", self.logical_ids, torch.long),
            ("generations", self.generations, torch.long),
        ):
            if not isinstance(value, Tensor) or value.shape != (capacity,):
                raise CrossWindowStateError(f"{name} must have shape [K]")
            if value.dtype != dtype:
                raise CrossWindowStateError(f"{name} has the wrong dtype")
            if value.device != self.features.device:
                raise CrossWindowStateError("all resident tensors must share a device")
        if self.class_prob.device != self.features.device:
            raise CrossWindowStateError("all resident tensors must share a device")
        if not torch.isfinite(self.features).all().item():
            raise CrossWindowStateError("resident features must be finite")
        if not _valid_probability_rows(self.class_prob):
            raise CrossWindowStateError("resident class probabilities are invalid")
        if (
            not torch.isfinite(self.confidence).all().item()
            or torch.any((self.confidence < 0) | (self.confidence > 1)).item()
        ):
            raise CrossWindowStateError("resident confidence must be within [0,1]")
        if torch.any(self.active & ~self.occupied).item():
            raise CrossWindowStateError("only occupied slots may be active")
        if torch.any(self.age < 0).item():
            raise CrossWindowStateError("resident age must be non-negative")
        identity_present = self.logical_ids >= 0
        generation_present = self.generations >= 0
        if not torch.equal(identity_present, self.occupied) or not torch.equal(
            generation_present, self.occupied
        ):
            raise CrossWindowStateError(
                "logical IDs and generations must match occupied slots"
            )
        if torch.any((self.last_seen >= 0) != self.occupied).item():
            raise CrossWindowStateError("last_seen must match occupied slots")
        identities = self.logical_ids[self.occupied]
        if identities.numel() != torch.unique(identities).numel():
            raise CrossWindowStateError("resident logical IDs must be unique")
        if (
            isinstance(self.next_logical_id, bool)
            or not isinstance(self.next_logical_id, int)
            or self.next_logical_id < 0
        ):
            raise CrossWindowStateError("next_logical_id must be non-negative")
        if identities.numel() and int(identities.max().item()) >= self.next_logical_id:
            raise CrossWindowStateError(
                "next_logical_id must exceed every resident logical ID"
            )
        if (
            isinstance(self.stage_watermark, bool)
            or not isinstance(self.stage_watermark, int)
            or self.stage_watermark < -1
        ):
            raise CrossWindowStateError("stage_watermark must be at least -1")
        if self.stage_watermark == -1:
            if self.episode_id is not None or self.occupied.any().item():
                raise CrossWindowStateError(
                    "empty stage state cannot belong to an episode"
                )
        elif not isinstance(self.episode_id, str) or not self.episode_id:
            raise CrossWindowStateError("committed state must name its episode")
        norms = torch.linalg.vector_norm(self.features[self.occupied], dim=-1)
        if (
            norms.numel()
            and torch.any(
                (norms > 1e-6)
                & ~torch.isclose(norms, torch.ones_like(norms), atol=1e-5)
            ).item()
        ):
            raise CrossWindowStateError("resident features must be normalized or zero")


@dataclass(frozen=True)
class BufferedGroup:
    group_key: GroupKey
    logical_id: int
    generation: int
    source_query_id: int
    feature: Tensor
    class_prob: Tensor
    confidence: float
    candidate_indices: tuple[int, ...]
    source_class_ids: tuple[int, ...]
    class_ids: tuple[int, ...]
    scores: tuple[float, ...]
    masks: tuple[Tensor, ...]

    @property
    def payload_bytes(self) -> int:
        return (
            _tensor_bytes(self.feature)
            + _tensor_bytes(self.class_prob)
            + sum(_tensor_bytes(mask) for mask in self.masks)
            + 48
            + 24 * len(self.masks)
        )

    def validate(self, *, point_count: int) -> None:
        if len(self.group_key) != 5:
            raise CrossWindowStateError("buffer group key is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.logical_id,
                self.generation,
                self.source_query_id,
            )
        ):
            raise CrossWindowStateError("buffer identities must be non-negative")
        if self.feature.ndim != 1 or self.feature.dtype != torch.float32:
            raise CrossWindowStateError("buffer feature must be rank-1 float32")
        if self.class_prob.ndim != 1 or self.class_prob.dtype != torch.float32:
            raise CrossWindowStateError("buffer class_prob must be rank-1 float32")
        if not torch.isfinite(self.feature).all().item() or not _valid_probability_rows(
            self.class_prob.unsqueeze(0)
        ):
            raise CrossWindowStateError(
                "buffer feature or class probability is invalid"
            )
        if not isinstance(self.confidence, float) or not 0.0 <= self.confidence <= 1.0:
            raise CrossWindowStateError("buffer confidence must be within [0,1]")
        lengths = {
            len(self.candidate_indices),
            len(self.source_class_ids),
            len(self.class_ids),
            len(self.scores),
            len(self.masks),
        }
        if len(lengths) != 1 or not self.masks:
            raise CrossWindowStateError(
                "buffer candidate fields must be non-empty and aligned"
            )
        if len(set(self.candidate_indices)) != len(self.candidate_indices):
            raise CrossWindowStateError("buffer candidate indices must be unique")
        if any(
            not isinstance(score, float) or not math.isfinite(score)
            for score in self.scores
        ):
            raise CrossWindowStateError("buffer scores must be finite floats")
        if len(set(self.class_ids)) != len(self.class_ids):
            raise CrossWindowStateError(
                "one buffered group may expose each output class at most once"
            )
        for mask in self.masks:
            if mask.shape != (point_count,) or mask.dtype != torch.bool:
                raise CrossWindowStateError("buffer masks must be bool [N]")
            if mask.device.type != "cpu":
                raise CrossWindowStateError("buffer masks must reside on CPU")


@dataclass(frozen=True)
class CrossWindowBuffer:
    reference_id: str
    episode_id: str
    scan_id: str
    absolute_stage: int
    vertex_ids: Tensor
    groups: tuple[BufferedGroup, ...]
    group_capacity: int = 100

    @property
    def buffer_bytes(self) -> int:
        return (
            _tensor_bytes(self.vertex_ids)
            + sum(group.payload_bytes for group in self.groups)
            + 32
        )

    def validate(self) -> None:
        for name, value in (
            ("reference_id", self.reference_id),
            ("episode_id", self.episode_id),
            ("scan_id", self.scan_id),
        ):
            if not isinstance(value, str) or not value:
                raise CrossWindowStateError(f"buffer {name} must be non-empty")
        if (
            isinstance(self.absolute_stage, bool)
            or not isinstance(self.absolute_stage, int)
            or self.absolute_stage < 0
        ):
            raise CrossWindowStateError("buffer stage must be non-negative")
        if (
            isinstance(self.group_capacity, bool)
            or not isinstance(self.group_capacity, int)
            or self.group_capacity <= 0
            or len(self.groups) > self.group_capacity
        ):
            raise CrossWindowStateError("buffer exceeds its fixed group capacity")
        if (
            self.vertex_ids.ndim != 1
            or self.vertex_ids.dtype != torch.long
            or self.vertex_ids.device.type != "cpu"
            or self.vertex_ids.unique().numel() != self.vertex_ids.numel()
        ):
            raise CrossWindowStateError("buffer vertex IDs must be unique CPU int64")
        identities = []
        feature_dim = None
        class_count = None
        for group in self.groups:
            group.validate(point_count=self.vertex_ids.numel())
            identities.append((group.logical_id, group.generation))
            feature_dim = feature_dim or group.feature.numel()
            class_count = class_count or group.class_prob.numel()
            if (
                group.feature.numel() != feature_dim
                or group.class_prob.numel() != class_count
            ):
                raise CrossWindowStateError("buffer group dimensions differ")
        if len(identities) != len(set(identities)):
            raise CrossWindowStateError("buffer may contain one group per entity")


@dataclass(frozen=True)
class CommittedObservation:
    public_id_for_group: tuple[int, ...]
    generation_for_group: tuple[int, ...]
    slot_for_group: tuple[int, ...]
    nonresident_groups: tuple[int, ...]
    buffer: CrossWindowBuffer
    assignment_sha256: str
    resident_bytes: int
    buffer_bytes: int


__all__ = [
    "BufferedGroup",
    "CommittedObservation",
    "CrossWindowBuffer",
    "CrossWindowState",
    "CrossWindowStateError",
    "GroupKey",
]
