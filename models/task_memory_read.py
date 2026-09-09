"""Proposal-anchored read adapter for prediction-driven task memory."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.task_memory_routing import EntityRoute
from models.task_memory_state import TaskMemoryState


class TaskMemoryReadError(ValueError):
    """Raised when a task-memory read violates its fixed deployment contract."""


class TaskMemoryRead(nn.Module):
    """Read one routed entity slot or abstain through a zero-value null branch."""

    hidden_dim = 128
    num_queries = 100
    capacity = 100

    def __init__(self) -> None:
        super().__init__()
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.null_key = nn.Parameter(torch.empty(self.hidden_dim))
        self.attention_scale = nn.Parameter(
            torch.tensor(math.sqrt(self.hidden_dim), dtype=torch.float32)
        )
        nn.init.normal_(
            self.null_key, mean=0.0, std=1.0 / math.sqrt(self.hidden_dim)
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._last_diagnostics: Mapping[str, float | int] = MappingProxyType({})

    @property
    def last_diagnostics(self) -> Mapping[str, float | int]:
        return self._last_diagnostics

    def _validate(
        self, queries: Tensor, state: TaskMemoryState, route: EntityRoute
    ) -> None:
        if not isinstance(state, TaskMemoryState):
            raise TaskMemoryReadError("task read requires TaskMemoryState")
        if not isinstance(route, EntityRoute):
            raise TaskMemoryReadError("task read requires EntityRoute")
        state.validate()
        route.validate()
        expected = (state.batch_size, self.num_queries, self.hidden_dim)
        if (
            not isinstance(queries, Tensor)
            or queries.shape != expected
            or not queries.is_floating_point()
            or not torch.isfinite(queries).all().item()
            or state.capacity != self.capacity
            or state.feature_dim != self.hidden_dim
            or route.query_to_slot.shape != expected[:2]
            or queries.device != state.embedding.device
            or queries.dtype != state.embedding.dtype
            or route.query_to_slot.device != queries.device
        ):
            raise TaskMemoryReadError("queries, route, and state do not match fixed shapes")

        matched = route.query_to_slot >= 0
        safe_slots = route.query_to_slot.clamp_min(0)
        routed_occupied = state.occupied.gather(1, safe_slots)
        if torch.any(matched & ~routed_occupied).item():
            raise TaskMemoryReadError("matched route points to an unoccupied slot")
        routed_ids = state.logical_ids.gather(1, safe_slots)
        routed_generations = state.generations.gather(1, safe_slots)
        if torch.any(matched & (routed_ids != route.prior_logical_id)).item() or torch.any(
            matched & (routed_generations != route.prior_generation)
        ).item():
            raise TaskMemoryReadError("route identity differs from task state")

    def forward(
        self, queries: Tensor, state: TaskMemoryState, route: EntityRoute
    ) -> Tensor:
        self._validate(queries, state, route)
        batch_size, query_count, _ = queries.shape
        matched = route.query_to_slot >= 0
        safe_slots = route.query_to_slot.clamp_min(0)
        routed_embedding = state.embedding.gather(
            1, safe_slots.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )

        normalized_queries = F.normalize(
            self.query_projection(queries), dim=-1, eps=1e-6
        )
        routed_keys = F.normalize(
            self.key_projection(routed_embedding), dim=-1, eps=1e-6
        )
        null_key = F.normalize(self.null_key, dim=0, eps=1e-6)
        scale = self.attention_scale.clamp(min=1.0, max=64.0).to(
            dtype=queries.dtype
        )
        routed_logit = (normalized_queries * routed_keys).sum(dim=-1) * scale
        null_logit = torch.einsum("bqd,d->bq", normalized_queries, null_key) * scale
        attention = torch.softmax(
            torch.stack((routed_logit, null_logit), dim=-1), dim=-1
        )
        slot_wins = matched & (routed_logit > null_logit)
        routed_values = self.value_projection(routed_embedding)
        read = attention[..., :1] * routed_values
        read = read * slot_wins.unsqueeze(-1)
        gate = torch.sigmoid(
            self.gate_projection(torch.cat((queries, read), dim=-1))
        )
        delta = gate * self.output_projection(read)
        delta = delta * slot_wins.unsqueeze(-1)
        output = queries + delta

        diagnostics: dict[str, float | int] = {
            "attention_scale": float(scale.detach().item()),
            "matched_query_count": int(matched.sum().detach().item()),
            "slot_win_count": int(slot_wins.sum().detach().item()),
            "null_win_count": int((matched & ~slot_wins).sum().detach().item()),
            "unassigned_query_count": int((~matched).sum().detach().item()),
            "gate_mean": float(gate.detach().mean().item()),
            "read_output_norm": float(delta.detach().norm(dim=-1).mean().item()),
            "candidate_branches_per_matched_query": 2,
            "batch_size": batch_size,
            "query_count": query_count,
        }
        self._last_diagnostics = MappingProxyType(diagnostics)
        return output


__all__ = ["TaskMemoryRead", "TaskMemoryReadError"]
