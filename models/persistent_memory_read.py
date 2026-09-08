"""Differentiable, prediction-state-only read adapter for Persist4D All-T."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MemoryReadError(ValueError):
    """Raised when the fixed All-T memory-read contract is violated."""


@dataclass(frozen=True)
class DetachedMemoryReadState:
    """A stop-gradient deployment-state snapshot without supervision fields."""

    embeddings: Tensor
    occupied_mask: Tensor
    active_mask: Tensor | None = None
    confidence: Tensor | None = None
    last_seen: Tensor | None = None

    def __post_init__(self) -> None:
        embeddings = self.embeddings
        occupied = self.occupied_mask
        if (
            not isinstance(embeddings, Tensor)
            or embeddings.ndim != 3
            or not embeddings.is_floating_point()
            or not torch.isfinite(embeddings).all().item()
            or not isinstance(occupied, Tensor)
            or occupied.ndim != 2
            or occupied.dtype != torch.bool
            or embeddings.shape[:2] != occupied.shape
        ):
            raise MemoryReadError("detached memory state tensors do not align")
        normalized: dict[str, Tensor | None] = {
            "embeddings": embeddings.detach().clone(),
            "occupied_mask": occupied.detach().clone(),
            "active_mask": self.active_mask,
            "confidence": self.confidence,
            "last_seen": self.last_seen,
        }
        for field in ("active_mask", "confidence", "last_seen"):
            value = normalized[field]
            if value is None:
                continue
            if not isinstance(value, Tensor) or value.shape != occupied.shape:
                raise MemoryReadError(f"memory {field} does not align")
            if field == "active_mask":
                if value.dtype != torch.bool or torch.any(value & ~occupied).item():
                    raise MemoryReadError("active memory slots must be occupied")
            elif field == "confidence":
                if (
                    not value.is_floating_point()
                    or not torch.isfinite(value).all().item()
                ):
                    raise MemoryReadError("memory confidence must be finite floating point")
            elif value.dtype == torch.bool or value.is_floating_point():
                raise MemoryReadError("memory last_seen must use integer dtype")
            normalized[field] = value.detach().clone()
        if normalized["active_mask"] is None:
            normalized["active_mask"] = normalized["occupied_mask"].clone()
        for field, value in normalized.items():
            object.__setattr__(self, field, value)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> DetachedMemoryReadState:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise MemoryReadError("memory batch size must be positive")
        return cls(
            embeddings=torch.zeros(batch_size, 100, 128, device=device, dtype=dtype),
            occupied_mask=torch.zeros(batch_size, 100, device=device, dtype=torch.bool),
        )


class PersistentMemoryRead(nn.Module):
    """Read all occupied history slots with an explicit abstention token."""

    hidden_dim = 128
    num_queries = 100
    capacity = 100

    def __init__(self) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.null_key = nn.Parameter(torch.empty(self.hidden_dim))
        self.null_value = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.null_key, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim))
        nn.init.normal_(self.null_value, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._last_diagnostics: Mapping[str, float | int | None] = MappingProxyType({})

    @property
    def last_diagnostics(self) -> Mapping[str, float | int | None]:
        return self._last_diagnostics

    def _validate(self, queries: Tensor, state: DetachedMemoryReadState) -> None:
        if (
            not isinstance(queries, Tensor)
            or queries.shape != (
                state.embeddings.shape[0],
                self.num_queries,
                self.hidden_dim,
            )
            or not queries.is_floating_point()
            or not torch.isfinite(queries).all().item()
            or state.embeddings.shape[1:] != (self.capacity, self.hidden_dim)
            or state.embeddings.device != queries.device
            or state.embeddings.dtype != queries.dtype
            or state.occupied_mask.device != queries.device
            or state.active_mask is None
            or state.active_mask.device != queries.device
        ):
            raise MemoryReadError("queries and detached memory do not match All-T shapes")

    def forward(self, queries: Tensor, state: DetachedMemoryReadState) -> Tensor:
        if not isinstance(state, DetachedMemoryReadState):
            raise MemoryReadError("memory read requires DetachedMemoryReadState")
        self._validate(queries, state)
        batch_size = queries.shape[0]
        normalized_queries = F.normalize(
            self.query_projection(self.query_norm(queries)), dim=-1, eps=1e-6
        )
        memory_keys = F.normalize(
            self.key_projection(state.embeddings), dim=-1, eps=1e-6
        )
        null_key = F.normalize(self.null_key, dim=0, eps=1e-6)
        keys = torch.cat(
            [memory_keys, null_key.view(1, 1, -1).expand(batch_size, 1, -1)],
            dim=1,
        )
        values = torch.cat(
            [
                self.value_projection(state.embeddings),
                self.null_value.view(1, 1, -1).expand(batch_size, 1, -1),
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                state.occupied_mask,
                torch.ones(batch_size, 1, dtype=torch.bool, device=queries.device),
            ],
            dim=1,
        )
        logits = torch.einsum("bqd,bkd->bqk", normalized_queries, keys)
        attention = torch.softmax(logits.masked_fill(~valid[:, None, :], -torch.inf), dim=-1)
        if not torch.isfinite(attention).all().item():
            raise MemoryReadError("memory attention is not finite")
        read = torch.einsum("bqk,bkd->bqd", attention, values)
        gate = torch.sigmoid(self.gate_projection(torch.cat([queries, read], dim=-1)))
        delta = gate * self.output_projection(read)
        has_memory = state.occupied_mask.any(dim=1)
        output = queries + delta * has_memory[:, None, None]

        active = state.active_mask
        dormant = state.occupied_mask & ~active
        active_mass = attention[:, :, : self.capacity].masked_fill(
            ~active[:, None, :], 0.0
        )
        dormant_mass = attention[:, :, : self.capacity].masked_fill(
            ~dormant[:, None, :], 0.0
        )
        diagnostics: dict[str, float | int | None] = {
            "active_attention_mass": float(active_mass.sum(dim=-1).mean().detach().item()),
            "dormant_attention_mass": float(
                dormant_mass.sum(dim=-1).mean().detach().item()
            ),
            "empty_sample_count": int((~has_memory).sum().detach().item()),
            "gate_mean": float(gate.mean().detach().item()),
            "null_attention_mass": float(attention[:, :, -1].mean().detach().item()),
            "occupied_slot_count": int(state.occupied_mask.sum().detach().item()),
            "read_output_norm": float(delta.norm(dim=-1).mean().detach().item()),
        }
        self._last_diagnostics = MappingProxyType(diagnostics)
        return output


__all__ = ["DetachedMemoryReadState", "MemoryReadError", "PersistentMemoryRead"]
