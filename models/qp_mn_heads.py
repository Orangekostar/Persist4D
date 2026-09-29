"""Single-module targeted interventions; historical heads and semantics stay intact."""
from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.short_module_heads import QualityHead
from models.short_module_heads import build_head as legacy_build_head


class Module(str, Enum):
    B0 = 'B0'
    Q2_F = 'Q2_F'
    Q_A0 = 'Q_A0'
    Q_P = 'Q_P'
    M_C = 'M_C'
    M_N = 'M_N'


class RawQualityHead(QualityHead):
    def __init__(self, input_dim: int, *, parent_skip: bool):
        super().__init__(input_dim)
        self.parent_skip = parent_skip
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: Tensor, parent_scores: Tensor) -> Tensor:
        residual = self.output(self.trunk(value)).squeeze(-1)
        if parent_scores.shape != residual.shape:
            raise ValueError('parent scores must align with retained candidates')
        return residual + parent_scores.detach() if self.parent_skip else residual


def build_head(module: str | Module, *, feature_dim: int, query_dim: int,
               classes: int, probability_dim: int, seed: int,
               thresholds: tuple[float, ...] = ()):
    module = Module(module)
    dimensions = {'feature_dim': feature_dim, 'query_dim': query_dim,
                      'classes': classes, 'probability_dim': probability_dim}
    if min(dimensions.values()) < 1:
        raise ValueError('parent dimensions must be positive')
    if module in (Module.Q_A0, Module.Q_P):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return RawQualityHead(query_dim + 4 * feature_dim + probability_dim + classes + 6,
                                  parent_skip=module is Module.Q_P)
    legacy = {Module.B0: 'B0', Module.Q2_F: 'Q2', Module.M_C: 'M1', Module.M_N: 'M1'}
    return legacy_build_head(legacy[module], **dimensions, seed=seed, thresholds=thresholds)


def quality_loss(scores: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """One weighted mean over the whole batch; empty supervision stays connected."""
    if scores.shape != targets.shape or scores.shape != weights.shape:
        raise ValueError('quality scores, targets and weights must align')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('quality weights must be finite and nonnegative')
    active = weights > 0
    if not active.any():
        return scores.sum() * 0
    return (weights[active] * (scores[active] - targets[active]).square()).sum() / weights.sum()


def shape_terms(*, logits: Tensor, delta: Tensor, targets: Tensor, weights: Tensor,
                segment_stages: Tensor, matched: Tensor, usable: Tensor,
                sample_indices: list[Tensor]) -> dict:
    """Original M1 per-slot BCE+Dice with explicit actual-supervision positions."""
    if logits.shape != targets.shape or logits.shape != delta.shape or logits.shape[1] == 0:
        raise ValueError('geometry matrices must align with nonempty candidate slots')
    samples = []
    for stage, indices in enumerate(sample_indices):
        if indices.numel() > 2048 or not torch.all(segment_stages[indices] == stage):
            raise ValueError('shared segment sample has wrong stage or size')
        samples.append(indices[weights[indices] > 0])
    has_points = any(indices.numel() > 0 for indices in samples)
    positive = usable.bool() & matched.bool() & has_points
    terms = []
    for candidate in range(logits.shape[1]):
        stages = []
        if positive[candidate]:
            for indices in samples:
                if not indices.numel():
                    continue
                w, y, z = weights[indices], targets[indices, candidate], logits[indices, candidate]
                bce = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() / w.sum()
                p = z.sigmoid()
                dice = 1 - (2 * (w * p * y).sum() + 1) / ((w * p).sum() + (w * y).sum() + 1)
                stages.append(bce + dice)
        terms.append(torch.stack(stages).mean() if stages else logits[:, candidate].sum() * 0)
    return {'shape': torch.stack(terms), 'positive': positive,
            'regularization': delta.square().mean(),
            'matched_without_sampled_points': int((usable & matched).sum()) if not has_points else 0,
            'unmatched': int((usable & ~matched).sum()), 'unknown': int((~usable).sum())}


def mask_loss(module: str | Module, records: list[dict]) -> tuple[Tensor, dict]:
    module = Module(module)
    if module not in (Module.M_C, Module.M_N) or not records:
        raise ValueError('mask loss requires M_C/M_N and nonempty records')
    shape = torch.cat([r['shape'] for r in records])
    positives = torch.cat([r['positive'] for r in records])
    numerator = shape.sum()
    count = int(positives.sum())
    denominator = shape.numel() if module is Module.M_C else count
    shape_loss = numerator / denominator if denominator else numerator * 0
    regularization = .01 * torch.stack([r['regularization'] for r in records]).mean()
    detail = {'B': shape.numel(), 'Npos': count, 'shape_numerator': numerator.detach().item(),
              'shape_mean_all': (numerator / shape.numel()).detach().item(),
              'shape_mean_pos': (numerator / count).detach().item() if count else 0.,
              'regularization': regularization.detach().item()}
    for key in ('matched_without_sampled_points', 'unmatched', 'unknown'):
        detail[key] = sum(r[key] for r in records)
    return shape_loss + regularization, detail
