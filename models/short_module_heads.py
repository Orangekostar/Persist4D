"""Independent prediction-only heads for the fixed ReScene round-one screen.

Parent tensors are detached by the descriptor adapter. No GT, parent model,
candidate filtering or score multiplication is part of these heads.
"""

import hashlib
import itertools
import math
from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class Module(str, Enum):
    B0 = "B0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"


@torch.no_grad()
def describe_candidates(features: Tensor, logits: Tensor, centroids: Tensor,
                        segment_stages: Tensor, stable_keys: tuple[str, ...], *,
                        horizon: int) -> tuple[Tensor, Tensor, Tensor]:
    """GT-free Nc×H×2D descriptors and per-stage support/probability scalars.

    Exploration ordering is SHA256('short-module-v1:'+stable_segment_key),
    shared across candidates; ties in top-logit fallback use the literal key.
    Foreground probability defaults to zero for empty original support.
    """
    segments, dimension = features.shape
    if (horizon not in (1, 2) or logits.shape[0] != segments
            or centroids.shape != (segments, 3)
            or segment_stages.shape != (segments,) or len(stable_keys) != segments
            or len(set(stable_keys)) != segments):
        raise ValueError("descriptor inputs must bind unique segments and observed stages")
    if set(segment_stages.tolist()) != set(range(horizon)):
        raise ValueError("each observed stage must contain segments")
    candidates = logits.shape[1]
    h = features.new_zeros(candidates, horizon, 2 * dimension)
    support = features.new_zeros(candidates, horizon)
    probability = features.new_zeros(candidates, horizon)
    hashes = [hashlib.sha256(("short-module-v1:" + k).encode()).hexdigest()
              for k in stable_keys]
    for stage in range(horizon):
        indices = torch.where(segment_stages == stage)[0]
        stable_order = sorted(indices.tolist(), key=lambda i: stable_keys[i])
        exploration_order = sorted(indices.tolist(), key=lambda i: (hashes[i], stable_keys[i]))
        for candidate in range(candidates):
            positive = indices[logits[indices, candidate] > 0]
            if positive.numel():
                h[candidate, stage, :dimension] = features[positive].mean(0)
                lower = centroids[positive].amin(0) - 0.10
                upper = centroids[positive].amax(0) + 0.10
                inside = ((centroids[indices] >= lower) & (centroids[indices] <= upper)).all(-1)
                region = indices[inside].tolist()
                probability[candidate, stage] = logits[positive, candidate].sigmoid().mean()
            else:
                ordered = torch.tensor(stable_order, device=logits.device)
                rank = torch.argsort(logits[ordered, candidate], descending=True, stable=True)
                region = ordered[rank[:32]].tolist()
            selected = set(region)
            exploration = [i for i in exploration_order if i not in selected][:32]
            union = sorted(selected.union(exploration))
            h[candidate, stage, dimension:] = features[union].mean(0)
            support[candidate, stage] = positive.numel() / indices.numel()
    return h, support, probability


def quality_inputs(query: Tensor, h: Tensor, class_probabilities: Tensor,
                   class_one_hot: Tensor, scores: Tensor, support: Tensor,
                   foreground_probability: Tensor) -> Tensor:
    """Fixed order; functional LayerNorm on q and mean h, never on flags.

    The absolute stage difference uses raw h values. Parent features and
    scalars are detached, preserving head-only training.
    """
    if h.ndim != 3 or h.shape[1] not in (1, 2):
        raise ValueError("quality descriptor requires H=1 or H=2")
    mean_h = h.mean(1)
    difference = (h[:, 0] - h[:, 1]).abs() if h.shape[1] == 2 else torch.zeros_like(mean_h)
    result = torch.cat((F.layer_norm(query, (query.shape[-1],)),
                        F.layer_norm(mean_h, (mean_h.shape[-1],)), difference,
                        class_probabilities, class_one_hot, scores[:, None],
                        support.amin(1, keepdim=True), support.amax(1, keepdim=True),
                        foreground_probability.amin(1, keepdim=True),
                        foreground_probability.amax(1, keepdim=True),
                        scores.new_full((scores.numel(), 1), h.shape[1] / 2)), dim=-1)
    return result.detach()


class QualityHead(nn.Module):
    def __init__(self, input_dim: int, *, thresholds: tuple[float, ...] = ()):
        super().__init__()
        if thresholds and (any(not 0 <= t <= 1 for t in thresholds)
                           or any(a >= b for a, b in itertools.pairwise(thresholds))):
            raise ValueError("thresholds must be strictly increasing within [0,1]")
        self.thresholds = tuple(thresholds)
        self.trunk = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(),
                                   nn.Linear(128, 64), nn.GELU())
        self.output = nn.Linear(64, len(thresholds) if thresholds else 1)
        if thresholds:
            with torch.no_grad():
                self.output.bias.fill_(-4)
                self.output.bias[0] = 0

    def probabilities(self, value: Tensor) -> Tensor:
        raw = self.output(self.trunk(value))
        if self.thresholds:
            increments = torch.cat((raw[:, :1], -F.softplus(raw[:, 1:])), dim=-1)
            return increments.cumsum(-1).sigmoid()
        return raw.sigmoid()

    def forward(self, value: Tensor) -> Tensor:
        return self.probabilities(value).mean(-1)


class MaskHead(nn.Module):
    def __init__(self, *, feature_dim: int, query_dim: int, classes: int,
                 per_stage: bool):
        super().__init__()
        self.feature_dim = feature_dim
        self.per_stage = per_stage
        self.trunk = nn.Sequential(nn.Linear(query_dim + 2 * feature_dim + classes, 128),
                                   nn.GELU())
        self.output = nn.Linear(128, feature_dim + 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: Tensor, query: Tensor, h: Tensor,
                class_one_hot: Tensor, segment_stages: Tensor) -> Tensor:
        """Return S×Nc residuals, including original negative-logit segments.

        h is Nc×H×2D; stages are zero-based observed scan indices. Shared heads
        use mean h; per-stage heads use h directly, with the same parameters.
        """
        if h.ndim != 3 or h.shape[1] not in (1, 2):
            raise ValueError("h must contain one or two real observed stages")
        if features.shape[1] != self.feature_dim or h.shape[2] != 2 * self.feature_dim:
            raise ValueError("feature dimension differs from the bound parent")
        if segment_stages.shape != (features.shape[0],):
            raise ValueError("segment stages must align with features")
        if segment_stages.numel() and (segment_stages.min() < 0
                                     or segment_stages.max() >= h.shape[1]):
            raise ValueError("segment stage is outside observed horizon")
        pooled = h if self.per_stage else h.mean(1, keepdim=True).expand_as(h)
        inputs = torch.cat((query[:, None, :].expand(-1, h.shape[1], -1), pooled,
                            class_one_hot[:, None, :].expand(-1, h.shape[1], -1)), -1)
        coefficients = self.output(self.trunk(inputs))
        # Select the stage for every segment; no detach, mask or bool conversion.
        a = coefficients[:, segment_stages, :-1].permute(1, 0, 2)
        b = coefficients[:, segment_stages, -1].transpose(0, 1)
        residual = (features[:, None, :] * a).sum(-1) / math.sqrt(self.feature_dim) + b
        return 2 * residual.tanh()


def build_head(module: str | Module, *, feature_dim: int, query_dim: int,
               classes: int, probability_dim: int, thresholds: tuple[float, ...],
               seed: int) -> QualityHead | MaskHead | None:
    module = Module(module)
    if min(feature_dim, query_dim, classes, probability_dim) < 1:
        raise ValueError("parent dimensions must be positive; no implicit padding")
    if module is Module.B0:
        return None
    # Seed locally: constructors do not perturb the shared sampling RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if module in (Module.Q1, Module.Q2, Module.Q3):
            if module is Module.Q3 and not thresholds:
                raise ValueError("Q3 requires bound official thresholds")
            # q + mean_h + |h1-h2| + probabilities + onehot + score +
            # support min/max + foreground probability min/max + H/2.
            input_dim = query_dim + 4 * feature_dim + probability_dim + classes + 6
            return QualityHead(input_dim, thresholds=thresholds if module is Module.Q3 else ())
        return MaskHead(feature_dim=feature_dim, query_dim=query_dim, classes=classes,
                        per_stage=module is Module.M2)


@torch.no_grad()
def apply_module(module: str | Module, head, parent, descriptor: dict, *, system) -> dict:
    """Prediction-only deployment boundary: one module, no GT arguments.

    Disable explicitly with B0. Geometry always traverses the actual native bool
    materializer, even at update zero, without another top-k/filter invocation.
    """
    from scripts.rescene_task_postprocess import materialize_segment_logits

    module = Module(module)
    result = parent.prediction()
    if module is Module.B0:
        if head is not None:
            raise ValueError("B0 must not load a head")
        return result
    evidence = parent.soft_evidence
    if evidence is None or head is None:
        raise ValueError("active module requires native soft evidence and one head")
    device = next(head.parameters()).device
    query = evidence.query_features.to(device)
    h = descriptor["h"].to(device)
    one_hot = F.one_hot(evidence.source_class_ids.to(device), descriptor["classes"]).float()
    if module in (Module.Q1, Module.Q2, Module.Q3):
        if not isinstance(head, QualityHead) or bool(head.thresholds) != (module is Module.Q3):
            raise ValueError("quality module/head mismatch")
        value = quality_inputs(query, h, evidence.class_probabilities.to(device), one_hot,
                               parent.pred_scores.to(device), descriptor["support"].to(device),
                               descriptor["foreground_probability"].to(device))
        result["pred_scores"] = head(value).cpu()
    else:
        if not isinstance(head, MaskHead) or head.per_stage != (module is Module.M2):
            raise ValueError("mask module/head mismatch")
        delta = head(evidence.segment_features.to(device), query, h, one_hot,
                     descriptor["segment_stages"].to(device))
        result["pred_masks"] = materialize_segment_logits(
            system=system, evidence=evidence,
            segment_logits=evidence.segment_logits + delta.cpu())
    return result
