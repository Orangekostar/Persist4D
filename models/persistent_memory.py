from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LocalInstanceObservation:
    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    latest_mask: list[Tensor]
    valid: Tensor

    def validate(self) -> None:
        for name, tensor in (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
            ("valid", self.valid),
        ):
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{name} must be a tensor")  # noqa: TRY004

        if self.features.ndim != 3:
            raise ValueError("features must have shape [B, Q, D]")
        batch_size, query_count, feature_dim = self.features.shape
        if feature_dim <= 0:
            raise ValueError("features must have a positive feature dimension")

        if self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, Q, C]")
        if self.class_prob.shape[:2] != (batch_size, query_count):
            raise ValueError("class_prob must match the features batch and query axes")
        if self.class_prob.shape[2] <= 0:
            raise ValueError("class_prob must have a positive class dimension")

        expected_shape = (batch_size, query_count)
        if self.confidence.shape != expected_shape:
            raise ValueError("confidence must have shape [B, Q]")
        if self.valid.shape != expected_shape:
            raise ValueError("valid must have shape [B, Q]")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must have bool dtype")

        if not isinstance(self.latest_mask, list):
            raise ValueError("latest_mask must be a list of tensors")  # noqa: TRY004
        if len(self.latest_mask) != batch_size:
            raise ValueError("latest_mask must contain one tensor per batch item")
        for mask in self.latest_mask:
            if not isinstance(mask, Tensor):
                raise ValueError(  # noqa: TRY004
                    "latest_mask entries must be tensors"
                )
            if mask.ndim != 2 or mask.shape[0] != query_count:
                raise ValueError("latest_mask entries must have shape [Q, S_latest]")

        expected_device = self.features.device
        if any(
            tensor.device != expected_device
            for tensor in (self.class_prob, self.confidence, self.valid)
        ) or any(mask.device != expected_device for mask in self.latest_mask):
            raise ValueError("all observation tensors must use the same device")

        for name, tensor in (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
        ):
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have a floating dtype")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class PersistentMemoryState:
    embedding: Tensor
    class_prob: Tensor
    confidence: Tensor
    occupied: Tensor
    active: Tensor
    age: Tensor
    last_seen: Tensor

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
    ) -> PersistentMemoryState:
        dimensions = {
            "batch_size": batch_size,
            "capacity": capacity,
            "feature_dim": feature_dim,
            "class_count": class_count,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a floating torch dtype")  # noqa: TRY004
        try:
            dtype_probe = torch.empty((), dtype=dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError("dtype must be a floating torch dtype") from error
        if not dtype_probe.is_floating_point():
            raise ValueError("dtype must be a floating torch dtype")

        try:
            target_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a valid torch device") from error

        state = cls(
            embedding=torch.zeros(
                batch_size,
                capacity,
                feature_dim,
                device=target_device,
                dtype=dtype,
            ),
            class_prob=torch.zeros(
                batch_size,
                capacity,
                class_count,
                device=target_device,
                dtype=dtype,
            ),
            confidence=torch.zeros(
                batch_size, capacity, device=target_device, dtype=dtype
            ),
            occupied=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.bool
            ),
            active=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.bool
            ),
            age=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.long
            ),
            last_seen=torch.full(
                (batch_size, capacity),
                -1,
                device=target_device,
                dtype=torch.long,
            ),
        )
        state.validate()
        return state

    @property
    def batch_size(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[0]

    @property
    def capacity(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[1]

    @property
    def feature_dim(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[2]

    @property
    def class_count(self) -> int:
        if not isinstance(self.class_prob, Tensor) or self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, K, C]")
        return self.class_prob.shape[2]

    def validate(self) -> None:
        tensors = (
            ("embedding", self.embedding),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
            ("occupied", self.occupied),
            ("active", self.active),
            ("age", self.age),
            ("last_seen", self.last_seen),
        )
        for name, tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{name} must be a tensor")  # noqa: TRY004

        if self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        if self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, K, C]")
        for name, tensor in tensors[2:]:
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B, K]")

        batch_size, capacity, feature_dim = self.embedding.shape
        if batch_size <= 0 or capacity <= 0 or feature_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.class_prob.shape[:2] != (batch_size, capacity):
            raise ValueError("class_prob must match embedding batch and capacity")
        if self.class_prob.shape[2] <= 0:
            raise ValueError("class_prob must have a positive class dimension")
        expected_shape = (batch_size, capacity)
        for name, tensor in tensors[2:]:
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must match embedding batch and capacity")

        expected_device = self.embedding.device
        if any(tensor.device != expected_device for _, tensor in tensors[1:]):
            raise ValueError("all state tensors must be on the same device")

        float_dtype = self.embedding.dtype
        if not self.embedding.is_floating_point():
            raise ValueError("embedding must have a floating dtype")
        for name, tensor in (
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
        ):
            if not tensor.is_floating_point() or tensor.dtype != float_dtype:
                raise ValueError(
                    f"{name} must use the same floating dtype as embedding"
                )
        if self.occupied.dtype != torch.bool or self.active.dtype != torch.bool:
            raise ValueError("occupied and active must have bool dtype")
        if self.age.dtype != torch.long or self.last_seen.dtype != torch.long:
            raise ValueError("age and last_seen must have long dtype")

        for name, tensor in tensors[:3]:
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(self.active & ~self.occupied).item():
            raise ValueError("active slots must also be occupied")
        if torch.any(self.age < 0).item():
            raise ValueError("age must be non-negative")
        if torch.any(self.last_seen < -1).item():
            raise ValueError("last_seen must be at least -1")

    def detach(self) -> PersistentMemoryState:
        return PersistentMemoryState(*(tensor.detach() for tensor in self.tensors()))

    def tensors(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.embedding,
            self.class_prob,
            self.confidence,
            self.occupied,
            self.active,
            self.age,
            self.last_seen,
        )
