"""Bounded prediction-only visual evidence memory for task-memory ablations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ObjectVisualMemoryError(ValueError):
    """Raised when the visual-memory deployment contract is violated."""


@dataclass(frozen=True)
class VisualCandidate:
    feature: Tensor
    quality: Tensor
    source_stage: int
    source_key: int

    def validate(self, feature_dim: int | None = None) -> None:
        if self.feature.ndim != 1 or (feature_dim is not None and self.feature.shape != (feature_dim,)) or not self.feature.is_floating_point():
            raise ObjectVisualMemoryError("visual candidate feature must be [D]")
        if self.quality.ndim != 0 or not self.quality.is_floating_point():
            raise ObjectVisualMemoryError("visual candidate quality must be scalar")
        if not torch.isfinite(self.feature).all().item() or not torch.isfinite(self.quality).item():
            raise ObjectVisualMemoryError("visual candidates must be finite")
        if not isinstance(self.source_stage, int) or self.source_stage < 0:
            raise ObjectVisualMemoryError("source_stage must be non-negative int")
        if not isinstance(self.source_key, int) or self.source_key < 0:
            raise ObjectVisualMemoryError("source_key must be non-negative int")


@dataclass(frozen=True)
class ObjectVisualState:
    features: Tensor
    valid: Tensor
    quality: Tensor
    source_stage: Tensor
    source_key: Tensor
    generations: Tensor

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        capacity: int = 100,
        representatives: int = 8,
        feature_dim: int = 128,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> ObjectVisualState:
        if capacity != 100 or representatives != 8 or feature_dim != 128:
            raise ObjectVisualMemoryError("visual state requires K=100, r=8, D=128")
        state = cls(
            features=torch.zeros((batch_size, capacity, representatives, feature_dim), device=device, dtype=dtype),
            valid=torch.zeros((batch_size, capacity, representatives), device=device, dtype=torch.bool),
            quality=torch.zeros((batch_size, capacity, representatives), device=device, dtype=dtype),
            source_stage=torch.full((batch_size, capacity, representatives), -1, device=device, dtype=torch.long),
            source_key=torch.full((batch_size, capacity, representatives), -1, device=device, dtype=torch.long),
            generations=torch.full((batch_size, capacity), -1, device=device, dtype=torch.long),
        )
        state.validate()
        return state

    @property
    def batch_size(self) -> int:
        return int(self.features.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.features.shape[1])

    @property
    def representatives(self) -> int:
        return int(self.features.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[3])

    @property
    def storage_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.tensors())

    def tensors(self) -> tuple[Tensor, ...]:
        return (
            self.features,
            self.valid,
            self.quality,
            self.source_stage,
            self.source_key,
            self.generations,
        )

    def detach(self) -> ObjectVisualState:
        return ObjectVisualState(*(tensor.detach() for tensor in self.tensors()))

    def validate(self) -> None:
        if self.features.ndim != 4 or self.features.shape[1:] != (100, 8, 128):
            raise ObjectVisualMemoryError("visual features must have shape [B,100,8,128]")
        expected = self.features.shape[:3]
        if self.valid.shape != expected or self.quality.shape != expected:
            raise ObjectVisualMemoryError("visual validity/quality shape differs")
        if self.source_stage.shape != expected or self.source_key.shape != expected:
            raise ObjectVisualMemoryError("visual source shape differs")
        if self.generations.shape != self.features.shape[:2]:
            raise ObjectVisualMemoryError("visual generations must have shape [B,100]")
        if self.valid.dtype is not torch.bool or self.source_stage.dtype is not torch.long or self.source_key.dtype is not torch.long or self.generations.dtype is not torch.long:
            raise ObjectVisualMemoryError("visual state index fields have invalid dtype")
        if not self.features.is_floating_point() or not self.quality.is_floating_point():
            raise ObjectVisualMemoryError("visual values must be floating point")
        if len({tensor.device for tensor in self.tensors()}) != 1:
            raise ObjectVisualMemoryError("visual tensors must share a device")
        if not torch.isfinite(self.features).all().item() or not torch.isfinite(self.quality).all().item():
            raise ObjectVisualMemoryError("visual state contains non-finite values")
        if torch.any(self.valid & (self.source_stage < 0)).item() or torch.any(self.valid & (self.source_key < 0)).item():
            raise ObjectVisualMemoryError("valid visual entries require source identities")
        if torch.any(~self.valid & (self.quality != 0)).item():
            raise ObjectVisualMemoryError("invalid visual entries must have zero quality")
        if torch.any(~self.valid & (self.source_stage != -1)).item() or torch.any(~self.valid & (self.source_key != -1)).item():
            raise ObjectVisualMemoryError("invalid visual entries must have sentinel sources")
        if torch.any(self.generations < -1).item():
            raise ObjectVisualMemoryError("visual generations must be -1 or non-negative")
        if self.storage_bytes > 2 * 1024 * 1024:
            raise ObjectVisualMemoryError("visual state exceeds 2 MiB")


def _candidate_order(candidates: Sequence[VisualCandidate]) -> list[int]:
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index].quality.detach().cpu().item()),
            -candidates[index].source_stage,
            candidates[index].source_key,
            index,
        ),
    )


def select_visual_representatives(
    candidates: Sequence[VisualCandidate],
    *,
    policy: str,
    limit: int = 8,
    recent_keys: set[int] | None = None,
) -> list[VisualCandidate]:
    if policy not in {"V-LAST", "V-CORE"}:
        raise ObjectVisualMemoryError("visual policy must be V-LAST or V-CORE")
    if not 0 < limit <= 8:
        raise ObjectVisualMemoryError("visual representative limit is invalid")
    unique: dict[int, VisualCandidate] = {}
    for candidate in candidates:
        candidate.validate()
        previous = unique.get(candidate.source_key)
        if previous is None or float(candidate.quality) > float(previous.quality):
            unique[candidate.source_key] = candidate
    pool = list(unique.values())
    if not pool:
        return []
    if policy == "V-LAST":
        selected = [pool[_candidate_order(pool)[0]]]
        remaining = [item for item in pool if item.source_key != selected[0].source_key]
        while remaining and len(selected) < limit:
            selected_features = torch.stack([item.feature.float() for item in selected])
            best_index = max(
                range(len(remaining)),
                key=lambda index: (
                    float(remaining[index].quality.detach().cpu().item())
                    * (0.5 + 0.5 * float(torch.cdist(remaining[index].feature.float().unsqueeze(0), selected_features).min().item())),
                    float(remaining[index].quality.detach().cpu().item()),
                    -remaining[index].source_stage,
                    -remaining[index].source_key,
                ),
            )
            selected.append(remaining.pop(best_index))
        return selected

    recent = recent_keys or set()
    recent_pool = [item for item in pool if item.source_key in recent]
    recent_pool = [recent_pool[index] for index in _candidate_order(recent_pool)[:2]]
    selected = list(recent_pool)
    selected_keys = {item.source_key for item in selected}
    remaining = [item for item in pool if item.source_key not in selected_keys]
    while remaining and len(selected) < limit:
        if not selected:
            index = _candidate_order(remaining)[0]
        else:
            selected_features = torch.stack([item.feature.float() for item in selected])
            index = max(
                range(len(remaining)),
                key=lambda candidate_index: (
                    float(remaining[candidate_index].quality.detach().cpu().item())
                    * (0.5 + 0.5 * float(torch.cdist(remaining[candidate_index].feature.float().unsqueeze(0), selected_features).min().item())),
                    float(remaining[candidate_index].quality.detach().cpu().item()),
                    -remaining[candidate_index].source_stage,
                    -remaining[candidate_index].source_key,
                ),
            )
        selected.append(remaining.pop(index))
    return selected


def update_visual_state(
    state: ObjectVisualState,
    *,
    slot_candidates: Sequence[Sequence[VisualCandidate]],
    slot_generations: Tensor,
    policy: str,
) -> ObjectVisualState:
    state.validate()
    if len(slot_candidates) != state.batch_size:
        raise ObjectVisualMemoryError("candidate batch size differs")
    if slot_generations.shape != (state.batch_size, state.capacity) or slot_generations.dtype is not torch.long:
        raise ObjectVisualMemoryError("slot generations must have shape [B,K] and int64")
    if slot_generations.device != state.features.device:
        raise ObjectVisualMemoryError("slot generations device differs")
    features = state.features.clone()
    valid = state.valid.clone()
    quality = state.quality.clone()
    source_stage = state.source_stage.clone()
    source_key = state.source_key.clone()
    generations = state.generations.clone()
    for batch_index, slots in enumerate(slot_candidates):
        if len(slots) != state.capacity:
            raise ObjectVisualMemoryError("candidate slots must have length K")
        for slot_index, current in enumerate(slots):
            generation = int(slot_generations[batch_index, slot_index].item())
            if generation < 0:
                valid[batch_index, slot_index] = False
                quality[batch_index, slot_index] = 0
                source_stage[batch_index, slot_index] = -1
                source_key[batch_index, slot_index] = -1
                features[batch_index, slot_index] = 0
                generations[batch_index, slot_index] = -1
                continue
            generation_matches = int(state.generations[batch_index, slot_index].item()) == generation
            old = [
                VisualCandidate(
                    feature=features[batch_index, slot_index, rep].detach().clone(),
                    quality=quality[batch_index, slot_index, rep].detach().clone(),
                    source_stage=int(source_stage[batch_index, slot_index, rep].item()),
                    source_key=int(source_key[batch_index, slot_index, rep].item()),
                )
                for rep in range(state.representatives)
                if generation_matches and bool(valid[batch_index, slot_index, rep].item())
            ]
            if current:
                for item in current:
                    item.validate(feature_dim=state.feature_dim)
                merged = (
                    list(current)
                    if policy == "V-LAST"
                    else old + list(current)
                )
                current_keys = {item.source_key for item in current}
                selected = select_visual_representatives(
                    merged,
                    policy=policy,
                    limit=state.representatives,
                    recent_keys=current_keys,
                )
            else:
                selected = old
            features[batch_index, slot_index] = 0
            valid[batch_index, slot_index] = False
            quality[batch_index, slot_index] = 0
            source_stage[batch_index, slot_index] = -1
            source_key[batch_index, slot_index] = -1
            generations[batch_index, slot_index] = generation
            for rep, item in enumerate(selected[: state.representatives]):
                features[batch_index, slot_index, rep] = item.feature.to(features)
                valid[batch_index, slot_index, rep] = True
                quality[batch_index, slot_index, rep] = item.quality.to(quality)
                source_stage[batch_index, slot_index, rep] = item.source_stage
                source_key[batch_index, slot_index, rep] = item.source_key
    updated = ObjectVisualState(features, valid, quality, source_stage, source_key, generations).detach()
    updated.validate()
    return updated


class ObjectVisualRead(nn.Module):
    hidden_dim = 128
    capacity = 100
    representatives = 8

    def __init__(self) -> None:
        super().__init__()
        self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_scale = nn.Parameter(torch.tensor(math.sqrt(self.hidden_dim), dtype=torch.float32))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._last_diagnostics = MappingProxyType({})

    @property
    def last_diagnostics(self):
        return self._last_diagnostics

    def forward(
        self,
        queries: Tensor,
        state: ObjectVisualState,
        query_to_slot: Tensor,
        query_generations: Tensor,
    ) -> Tensor:
        state.validate()
        if queries.shape != (state.batch_size, self.capacity, self.hidden_dim):
            raise ObjectVisualMemoryError("queries have incompatible shape")
        if query_to_slot.shape != query_generations.shape or query_to_slot.shape != queries.shape[:2]:
            raise ObjectVisualMemoryError("visual route shape differs")
        if query_to_slot.dtype is not torch.long or query_generations.dtype is not torch.long:
            raise ObjectVisualMemoryError("visual route must use int64")
        if queries.device != state.features.device:
            raise ObjectVisualMemoryError("visual query and state device differ")
        matched = query_to_slot >= 0
        safe_slot = query_to_slot.clamp_min(0)
        slot_valid = state.valid.any(dim=-1).gather(1, safe_slot)
        generation_match = state.generations.gather(1, safe_slot) == query_generations
        eligible = matched & slot_valid & generation_match
        gathered = state.features.gather(1, safe_slot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.representatives, self.hidden_dim))
        gathered_valid = state.valid.gather(1, safe_slot.unsqueeze(-1).expand(-1, -1, self.representatives)) & eligible.unsqueeze(-1)
        q = F.normalize(self.query_projection(queries), dim=-1, eps=1e-6)
        k = F.normalize(self.key_projection(gathered), dim=-1, eps=1e-6)
        v = self.value_projection(gathered)
        scale = self.attention_scale.clamp(1.0, 64.0).to(dtype=queries.dtype)
        logits = torch.einsum("bqd,bqrd->bqr", q, k) * scale
        logits = logits.masked_fill(~gathered_valid, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(gathered_valid, weights, torch.zeros_like(weights))
        read = torch.sum(weights.unsqueeze(-1) * v, dim=-2)
        gate = torch.sigmoid(self.gate_projection(torch.cat((queries, read), dim=-1)))
        delta = gate * self.output_projection(read)
        delta = delta * eligible.unsqueeze(-1)
        output = queries + delta
        self._last_diagnostics = MappingProxyType(
            {
                "matched_query_count": int(matched.sum().detach().item()),
                "valid_read_count": int(eligible.sum().detach().item()),
                "generation_mismatch_count": int((matched & ~generation_match).sum().detach().item()),
                "read_output_norm": float(delta.detach().float().norm(dim=-1).mean().item()),
                "attention_scale": float(scale.detach().item()),
            }
        )
        return output


__all__ = [
    "ObjectVisualMemoryError",
    "ObjectVisualRead",
    "ObjectVisualState",
    "VisualCandidate",
    "select_visual_representatives",
    "update_visual_state",
]
