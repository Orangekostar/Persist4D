"""Opt-in model primitives for the Perception Gain V1 campaign."""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SemanticQueryScorer(nn.Module):
    """Frozen segment foreground scorer used by Q-SEM query positioning."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class SoftOccurrence:
    scan_id: str
    window_index: int
    logical_id: Hashable
    generation: int
    source_class_id: int
    source_query_id: int
    candidate_index: int

    @property
    def pair_key(self) -> tuple[str, int, int, int]:
        return (
            self.scan_id,
            self.logical_id,
            self.generation,
            self.source_class_id,
        )


@dataclass(frozen=True)
class SoftPairDecision:
    new_index: int
    old_index: int | None
    reason: str


def pair_adjacent_soft_occurrences(
    *,
    old_occurrences: Sequence[SoftOccurrence],
    new_occurrences: Sequence[SoftOccurrence],
) -> tuple[SoftPairDecision, ...]:
    """Pair by frozen D0 identity/class, accepting exactly one adjacent old row."""

    decisions = []
    for new_index, new in enumerate(new_occurrences):
        matches = [
            old_index
            for old_index, old in enumerate(old_occurrences)
            if old.pair_key == new.pair_key and old.window_index == new.window_index - 1
        ]
        if len(matches) == 1:
            decisions.append(
                SoftPairDecision(
                    new_index=new_index,
                    old_index=matches[0],
                    reason="PAIRED_UNIQUE_ADJACENT",
                )
            )
        elif matches:
            decisions.append(
                SoftPairDecision(
                    new_index=new_index,
                    old_index=None,
                    reason="FALLBACK_MULTIPLE_ADJACENT_OLD",
                )
            )
        else:
            decisions.append(
                SoftPairDecision(
                    new_index=new_index,
                    old_index=None,
                    reason="FALLBACK_NO_ADJACENT_OLD",
                )
            )
    return tuple(decisions)


class CausalMaskRefiner(nn.Module):
    """Local residual refiner over frozen old/new soft evidence."""

    input_dim = 135

    def __init__(self, *, input_mode: str = "OLD_NEW") -> None:
        super().__init__()
        if input_mode not in {"NEW_ONLY", "OLD_NEW"}:
            raise ValueError("refiner input mode is invalid")
        self.input_mode = input_mode
        self.feature_norm = nn.LayerNorm(128, eps=1e-5, elementwise_affine=False)
        self.input = nn.Linear(self.input_dim, 64)
        self.activation = nn.GELU()
        self.output = nn.Linear(64, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _score_vector(
        value: Tensor | float,
        *,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype).detach()
        if tensor.ndim == 0:
            return tensor.expand(size)
        if tensor.ndim == 1 and tensor.numel() == size:
            return tensor
        raise ValueError("refiner scores must be scalar or align with segments")

    def forward(
        self,
        *,
        new_features: Tensor,
        old_logits: Tensor,
        new_logits: Tensor,
        old_score: Tensor | float,
        new_score: Tensor | float,
    ) -> tuple[Tensor, Tensor]:
        if self.input_mode == "NEW_ONLY":
            old_logits, old_score = new_logits, new_score
        if new_features.ndim != 2 or new_features.shape[1] != 128:
            raise ValueError("new_features must have shape [S, 128]")
        if (
            old_logits.ndim != 1
            or new_logits.ndim != 1
            or old_logits.shape != new_logits.shape
            or old_logits.numel() != new_features.shape[0]
        ):
            raise ValueError("old/new logits must align with new segments")
        features = self.feature_norm(new_features.detach())
        old = old_logits.detach().to(device=features.device, dtype=features.dtype)
        new = new_logits.detach().to(device=features.device, dtype=features.dtype)
        old_scores = self._score_vector(
            old_score,
            size=new.numel(),
            device=features.device,
            dtype=features.dtype,
        )
        new_scores = self._score_vector(
            new_score,
            size=new.numel(),
            device=features.device,
            dtype=features.dtype,
        )
        inputs = torch.cat(
            (
                features,
                old.clamp(-8, 8).unsqueeze(1),
                new.clamp(-8, 8).unsqueeze(1),
                (old - new).clamp(-8, 8).unsqueeze(1),
                old.sigmoid().unsqueeze(1),
                new.sigmoid().unsqueeze(1),
                old_scores.unsqueeze(1),
                new_scores.unsqueeze(1),
            ),
            dim=1,
        )
        delta = 2 * torch.tanh(
            self.output(self.activation(self.input(inputs))).squeeze(1)
        )
        return new + delta, delta


def _stage_query_quotas(stage_ids: Tensor, num_queries: int) -> dict[int, int]:
    stages = sorted({int(value) for value in stage_ids.detach().cpu().tolist()})
    if not stages:
        raise ValueError("semantic query selection requires at least one stage")
    base, remainder = divmod(num_queries, len(stages))
    return {stage: base + (index < remainder) for index, stage in enumerate(stages)}


def _normalized_xyz(coordinates: Tensor) -> Tensor:
    xyz = coordinates[:, :3].float()
    minimum = xyz.min(dim=0).values
    extent = xyz.max(dim=0).values - minimum
    return (xyz - minimum) / torch.where(extent > 0, extent, torch.ones_like(extent))


def _farthest_stable(
    candidates: list[int],
    *,
    selected: list[int],
    coordinates: Tensor,
    scores: Tensor,
    stable_keys: tuple[object, ...],
) -> int:
    if not selected:
        return min(
            candidates,
            key=lambda index: (-float(scores[index].item()), stable_keys[index]),
        )
    chosen = coordinates[selected]
    candidate_indices = torch.tensor(
        candidates, dtype=torch.long, device=coordinates.device
    )
    minimum_distances = (
        torch.cdist(coordinates[candidate_indices], chosen)
        .min(dim=1)
        .values.detach()
        .cpu()
        .tolist()
    )
    best_index = candidates[0]
    best_distance = -1.0
    for index, distance in zip(candidates, minimum_distances, strict=True):
        if distance > best_distance or (
            distance == best_distance and stable_keys[index] < stable_keys[best_index]
        ):
            best_index = index
            best_distance = distance
    return best_index


def select_semantic_query_indices(
    features: Tensor,
    coordinates: Tensor,
    stage_ids: Tensor,
    *,
    stable_keys: tuple[object, ...],
    scorer: nn.Module,
    num_queries: int = 100,
    semantic_fraction: float = 0.75,
) -> tuple[Tensor, dict[str, object]]:
    """Select deterministic stage-balanced Q-SEM query positions."""

    if (
        features.ndim != 2
        or coordinates.ndim != 2
        or coordinates.shape[1] < 3
        or stage_ids.ndim != 1
        or features.shape[0] != coordinates.shape[0]
        or features.shape[0] != stage_ids.numel()
        or len(stable_keys) != features.shape[0]
    ):
        raise ValueError("semantic query inputs must align on the segment axis")
    if num_queries <= 0:
        raise ValueError("num_queries must be positive")
    if not 0.0 <= semantic_fraction <= 1.0:
        raise ValueError("semantic_fraction must be within [0, 1]")
    quotas = _stage_query_quotas(stage_ids, num_queries)
    normalized = _normalized_xyz(coordinates)
    with torch.no_grad():
        scores = scorer(features.detach()).reshape(-1).detach()
    selected_all: list[int] = []
    repeat_count = 0
    for stage, quota in quotas.items():
        stage_candidates = torch.nonzero(stage_ids == stage, as_tuple=True)[0].tolist()
        semantic_quota = int(math.floor(semantic_fraction * quota))
        exploration_quota = quota - semantic_quota
        score_order = sorted(
            stage_candidates,
            key=lambda index: (-float(scores[index].item()), stable_keys[index]),
        )
        semantic_pool = score_order[: min(len(score_order), 4 * semantic_quota)]
        selected: list[int] = []
        remaining_semantic = list(semantic_pool)
        while remaining_semantic and len(selected) < semantic_quota:
            index = _farthest_stable(
                remaining_semantic,
                selected=selected,
                coordinates=normalized,
                scores=scores,
                stable_keys=stable_keys,
            )
            selected.append(index)
            remaining_semantic.remove(index)

        remaining = [index for index in stage_candidates if index not in selected]
        exploration_target = min(quota, len(selected) + exploration_quota)
        while remaining and len(selected) < exploration_target:
            index = _farthest_stable(
                remaining,
                selected=selected,
                coordinates=normalized,
                scores=scores,
                stable_keys=stable_keys,
            )
            selected.append(index)
            remaining.remove(index)
        while remaining and len(selected) < min(quota, len(stage_candidates)):
            index = _farthest_stable(
                remaining,
                selected=selected,
                coordinates=normalized,
                scores=scores,
                stable_keys=stable_keys,
            )
            selected.append(index)
            remaining.remove(index)
        unique_count = len(selected)
        if not selected:
            raise ValueError(f"stage {stage} has no semantic query candidates")
        while len(selected) < quota:
            selected.append(selected[(len(selected) - unique_count) % unique_count])
            repeat_count += 1
        selected_all.extend(selected)
    indices = torch.tensor(selected_all, dtype=torch.long, device=features.device)
    return indices, {
        "stage_quotas": quotas,
        "repeat_count": repeat_count,
        "candidate_count": int(features.shape[0]),
    }


def derive_segment_stage_ids(
    point2segment: Tensor,
    temporal_stages: Tensor,
) -> Tensor:
    """Map each contiguous segment ID to exactly one integer stage."""

    if (
        not isinstance(point2segment, Tensor)
        or not isinstance(temporal_stages, Tensor)
        or point2segment.ndim != 1
        or temporal_stages.ndim != 1
        or point2segment.numel() != temporal_stages.numel()
    ):
        raise ValueError("point2segment and temporal_stages must be aligned vectors")
    if point2segment.is_floating_point() or temporal_stages.is_floating_point():
        raise ValueError("segment and stage IDs must use integer tensors")
    point2segment = point2segment.long()
    temporal_stages = temporal_stages.long().to(point2segment.device)
    if point2segment.numel() == 0:
        return temporal_stages.new_empty(0)
    if point2segment.min().item() < 0:
        raise ValueError("segment IDs must be non-negative")
    segment_count = int(point2segment.max().item()) + 1
    counts = torch.bincount(point2segment, minlength=segment_count)
    if torch.any(counts == 0).item():
        raise ValueError("segment IDs must form a contiguous namespace")
    minimum = torch.full(
        (segment_count,),
        torch.iinfo(torch.long).max,
        dtype=torch.long,
        device=point2segment.device,
    )
    maximum = torch.full(
        (segment_count,),
        torch.iinfo(torch.long).min,
        dtype=torch.long,
        device=point2segment.device,
    )
    minimum.scatter_reduce_(
        0, point2segment, temporal_stages, reduce="amin", include_self=True
    )
    maximum.scatter_reduce_(
        0, point2segment, temporal_stages, reduce="amax", include_self=True
    )
    if not torch.equal(minimum, maximum):
        raise ValueError("a segment spans multiple stages")
    return minimum


def _sample_stage_support(
    stage_ids: Tensor,
    *,
    sample_fraction: float,
) -> tuple[Tensor, ...]:
    supports = []
    for stage in torch.unique(stage_ids, sorted=True):
        indices = torch.nonzero(stage_ids == stage, as_tuple=True)[0]
        if sample_fraction != -1:
            if not 0.0 < sample_fraction <= 1.0:
                raise ValueError("sample_fraction must be -1 or within (0, 1]")
            sample_count = max(1, int(sample_fraction * indices.numel()))
            order = torch.randperm(indices.numel(), device=indices.device)
            indices = indices[order[:sample_count]]
        supports.append(indices)
    return tuple(supports)


def stage_aware_mask_losses(
    logits: Tensor,
    targets: Tensor,
    stage_ids: Tensor,
    *,
    mode: str,
    sample_fraction: float = -1,
    alpha: float = 0.5,
    beta: float = 0.25,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Compute balanced or smooth-worst mask losses over real stage support."""

    if mode not in {"balanced", "worst"}:
        raise ValueError("mode must be balanced or worst")
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must have aligned [M, S] shapes")
    if stage_ids.ndim != 1 or stage_ids.numel() != logits.shape[1]:
        raise ValueError("stage_ids must align with the mask support axis")
    if stage_ids.is_floating_point():
        raise ValueError("stage IDs must use an integer tensor")
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and within [0, 1]")
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be finite and positive")
    zero = (
        torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits)).sum() * 0
    )
    supports = _sample_stage_support(
        stage_ids.to(logits.device), sample_fraction=sample_fraction
    )
    if logits.shape[0] == 0 or not supports:
        return (
            zero,
            zero,
            {
                "matched_instances": int(logits.shape[0]),
                "valid_stages": len(supports),
                "ghost_instance_stages": 0,
                "sampled_support": sum(index.numel() for index in supports),
            },
        )

    target_values = targets.to(device=logits.device, dtype=logits.dtype)
    bce_by_stage = []
    dice_by_stage = []
    ghost_instance_stages = 0
    for support in supports:
        stage_logits = logits[:, support]
        stage_targets = target_values[:, support]
        bce_by_stage.append(
            F.binary_cross_entropy_with_logits(
                stage_logits, stage_targets, reduction="none"
            ).mean(dim=1)
        )
        probabilities = stage_logits.sigmoid()
        numerator = 2 * (probabilities * stage_targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + stage_targets.sum(dim=1)
        dice_by_stage.append(1 - (numerator + 1) / (denominator + 1))
        ghost_instance_stages += int((stage_targets.sum(dim=1) == 0).sum().item())

    bce_values = torch.stack(bce_by_stage, dim=1)
    dice_values = torch.stack(dice_by_stage, dim=1)

    def aggregate(values: Tensor) -> Tensor:
        mean = values.mean(dim=1)
        if mode == "balanced":
            return mean
        smooth_worst = beta * (
            torch.logsumexp(values / beta, dim=1) - math.log(values.shape[1])
        )
        return (1 - alpha) * mean + alpha * smooth_worst

    return (
        aggregate(bce_values).mean(),
        aggregate(dice_values).mean(),
        {
            "matched_instances": int(logits.shape[0]),
            "valid_stages": len(supports),
            "ghost_instance_stages": ghost_instance_stages,
            "sampled_support": sum(index.numel() for index in supports),
        },
    )


__all__ = [
    "CausalMaskRefiner",
    "SemanticQueryScorer",
    "SoftOccurrence",
    "SoftPairDecision",
    "derive_segment_stage_ids",
    "pair_adjacent_soft_occurrences",
    "select_semantic_query_indices",
    "stage_aware_mask_losses",
]
