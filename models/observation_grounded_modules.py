"""Differentiable, current-support observation evidence and one-shot feedback."""

import torch
from torch import nn
from torch.nn import functional as F

from datasets.native_long_dataset import stable_seed


EPS = 1e-6


def pool_observation(features, mask_logits, current_point_counts):
    """Pool in segment space; counts refer to current full-resolution points.

    The frozen features have no gradient. Mask probabilities and the quotient do.
    Past segments and padding have exactly zero counts.
    """
    if (features.ndim != 2 or mask_logits.ndim != 2
            or current_point_counts.shape != (features.shape[0],)
            or mask_logits.shape[0] != features.shape[0]):
        raise ValueError("OGR pooling requires aligned [S,D], [S,Q], [S] tensors")
    weighted = mask_logits.float().sigmoid() * current_point_counts.float()[:, None]
    mass = weighted.sum(0)
    valid = mass >= EPS
    values = weighted.T @ features.detach().float() / mass.clamp_min(EPS)[:, None]
    values = torch.where(valid[:, None], values, torch.zeros_like(values))
    return values, valid, mass


class IdentityEvidenceHead(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.Wc = nn.Linear(dim, dim, bias=False)
        self.Wh = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.Wc.weight)
            nn.init.eye_(self.Wh.weight)
        self.null_logit = nn.Parameter(torch.zeros(()))

    def forward(self, values, valid, prototypes, class_prob, history_class_prob):
        if (values.ndim != 2 or prototypes.ndim != 2
                or values.shape[1] != self.Wc.in_features
                or prototypes.shape[1] != values.shape[1]
                or valid.shape != (len(values),)
                or class_prob.ndim != 2 or history_class_prob.ndim != 2
                or class_prob.shape[0] != len(values)
                or history_class_prob.shape != (len(prototypes), class_prob.shape[1])):
            raise ValueError("OGR identity inputs do not align")
        current = F.normalize(self.Wc(values.float()), dim=-1, eps=EPS)
        history = F.normalize(self.Wh(prototypes.detach().float()), dim=-1, eps=EPS)
        logits = current @ history.T / .1
        current_fg = class_prob.detach().float()[:, :-1]
        history_fg = history_class_prob.detach().float()[:, :-1]
        current_mass = current_fg.sum(-1, keepdim=True)
        history_mass = history_fg.sum(-1, keepdim=True)
        overlap = (current_fg / current_mass.clamp_min(EPS)) @ (
            history_fg / history_mass.clamp_min(EPS)).T
        usable_prior = (current_mass >= EPS) & (history_mass.T >= EPS)
        logits = logits + torch.where(usable_prior, .25 * overlap.clamp_min(EPS).log(),
                                      torch.zeros_like(overlap))
        logits = torch.cat((logits, self.null_logit.expand(len(values), 1)), dim=-1)
        # Keep null as a separate column even for empty history. Invalid regions
        # cannot claim a historical identity, regardless of projection biases.
        real = logits[:, :-1].masked_fill(~valid[:, None], -1e6)
        logits = torch.cat((real, logits[:, -1:]), dim=-1)
        return {"logits": logits, "probabilities": logits.softmax(-1), "valid": valid}


class ObservationFeedback(nn.Module):
    def __init__(self, seed=246, dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stable_seed(seed, "ogr-common-feedback"))
            self.input_layer = nn.Linear(3 * dim, dim)
            self.output_layer = nn.Linear(dim, dim)
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, direct, history, source):
        combined = torch.cat((self.norm(direct), self.norm(history),
                              self.norm(source.detach())), dim=-1)
        return self.output_layer(F.gelu(self.input_layer(combined)))
