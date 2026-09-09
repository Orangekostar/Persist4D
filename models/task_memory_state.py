from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from models.persistent_memory import PersistentMemoryState


class TaskMemoryStateError(ValueError):
    """Raised when a task-memory state violates its deployment contract."""


@dataclass(frozen=True)
class TaskMemoryConfig:
    class_weight: float = 0.25
    association_threshold: float = 0.5
    update_mode: str = "confidence_ema"
    update_rate: float = 0.2
    max_update_rate: float = 0.2

    def __post_init__(self) -> None:
        for name, value in (
            ("class_weight", self.class_weight),
            ("association_threshold", self.association_threshold),
            ("update_rate", self.update_rate),
            ("max_update_rate", self.max_update_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise TaskMemoryStateError(f"{name} must be a finite number")
        if not 0.0 <= float(self.class_weight) <= 1.0:
            raise TaskMemoryStateError("class_weight must be within [0, 1]")
        if self.update_mode not in {"confidence_ema", "fixed_ema", "last"}:
            raise TaskMemoryStateError(
                "update_mode must be confidence_ema, fixed_ema, or last"
            )
        if not 0.0 <= float(self.update_rate) <= 1.0:
            raise TaskMemoryStateError("update_rate must be within [0, 1]")
        if not 0.0 <= float(self.max_update_rate) <= 1.0:
            raise TaskMemoryStateError("max_update_rate must be within [0, 1]")
        if (
            self.update_mode == "confidence_ema"
            and self.update_rate > self.max_update_rate
        ):
            raise TaskMemoryStateError(
                "confidence EMA requires update_rate <= max_update_rate"
            )


@dataclass(frozen=True)
class TaskMemoryState:
    embedding: Tensor
    class_prob: Tensor
    confidence: Tensor
    occupied: Tensor
    active: Tensor
    age: Tensor
    last_seen: Tensor
    stage_watermark: Tensor
    logical_ids: Tensor
    generations: Tensor
    next_logical_id: Tensor
    config: TaskMemoryConfig

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        capacity: int,
        feature_dim: int,
        class_count: int,
        device: torch.device | str,
        dtype: torch.dtype,
        config: TaskMemoryConfig | None = None,
    ) -> TaskMemoryState:
        association = PersistentMemoryState.empty(
            batch_size=batch_size,
            capacity=capacity,
            feature_dim=feature_dim,
            class_count=class_count,
            device=device,
            dtype=dtype,
        )
        state = cls(
            *association.tensors(),
            logical_ids=torch.full(
                (batch_size, capacity),
                -1,
                device=association.embedding.device,
                dtype=torch.long,
            ),
            generations=torch.full(
                (batch_size, capacity),
                -1,
                device=association.embedding.device,
                dtype=torch.long,
            ),
            next_logical_id=torch.zeros(
                batch_size,
                device=association.embedding.device,
                dtype=torch.long,
            ),
            config=config or TaskMemoryConfig(),
        )
        state.validate()
        return state

    @property
    def batch_size(self) -> int:
        return self.embedding.shape[0]

    @property
    def capacity(self) -> int:
        return self.embedding.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.embedding.shape[2]

    @property
    def class_count(self) -> int:
        return self.class_prob.shape[2]

    @property
    def state_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def association_tensors(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.embedding,
            self.class_prob,
            self.confidence,
            self.occupied,
            self.active,
            self.age,
            self.last_seen,
            self.stage_watermark,
        )

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            *self.association_tensors(),
            self.logical_ids,
            self.generations,
            self.next_logical_id,
        )

    def as_persistent_state(self) -> PersistentMemoryState:
        state = PersistentMemoryState(*self.association_tensors())
        state.validate()
        return state

    def detach(self) -> TaskMemoryState:
        return TaskMemoryState(
            *(tensor.detach() for tensor in self.tensors()),
            config=self.config,
        )

    def validate(self) -> None:
        if not isinstance(self.config, TaskMemoryConfig):
            raise TaskMemoryStateError("config must be a TaskMemoryConfig")
        try:
            association = self.as_persistent_state()
        except ValueError as error:
            raise TaskMemoryStateError(str(error)) from error

        expected_shape = (association.batch_size, association.capacity)
        for name, tensor in (
            ("logical_ids", self.logical_ids),
            ("generations", self.generations),
        ):
            if not isinstance(tensor, Tensor) or tensor.shape != expected_shape:
                raise TaskMemoryStateError(f"{name} must have shape [B, K]")
            if tensor.dtype != torch.long:
                raise TaskMemoryStateError(f"{name} must use int64")
            if tensor.device != association.embedding.device:
                raise TaskMemoryStateError("all state tensors must share a device")
        if (
            not isinstance(self.next_logical_id, Tensor)
            or self.next_logical_id.shape != (association.batch_size,)
            or self.next_logical_id.dtype != torch.long
        ):
            raise TaskMemoryStateError("next_logical_id must have shape [B] and int64")
        if self.next_logical_id.device != association.embedding.device:
            raise TaskMemoryStateError("all state tensors must share a device")
        if torch.any(self.next_logical_id < 0).item():
            raise TaskMemoryStateError("next_logical_id must be non-negative")

        identity_present = self.logical_ids >= 0
        generation_present = self.generations >= 0
        if not torch.equal(identity_present, association.occupied):
            raise TaskMemoryStateError("logical IDs must exist exactly for occupied slots")
        if not torch.equal(generation_present, association.occupied):
            raise TaskMemoryStateError("generations must exist exactly for occupied slots")
        for batch_index in range(association.batch_size):
            identities = self.logical_ids[batch_index, association.occupied[batch_index]]
            if identities.numel() != torch.unique(identities).numel():
                raise TaskMemoryStateError("logical IDs must be unique within an episode")
            if identities.numel() and torch.any(
                identities >= self.next_logical_id[batch_index]
            ).item():
                raise TaskMemoryStateError(
                    "next_logical_id must exceed all issued logical IDs"
                )


__all__ = ["TaskMemoryConfig", "TaskMemoryState", "TaskMemoryStateError"]
