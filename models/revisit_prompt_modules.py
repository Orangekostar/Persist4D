"""Label-free, proposal-free current evidence retrieval for MPR-V1."""

from __future__ import annotations

import torch
import hashlib
from torch import nn
from torch.nn import functional as F

from datasets.native_long_dataset import stable_seed


def tensor_summary(value):
    tensor = value.detach().cpu().contiguous()
    numeric = tensor.double()
    if not torch.isfinite(numeric).all():
        raise FloatingPointError("a recorded input/pre-mask tensor is not finite")
    hasher = hashlib.sha256()
    hasher.update(str((tuple(tensor.shape),tensor.dtype)).encode())
    hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hasher.hexdigest(),
                sum=float(numeric.sum()),squared_sum=float(numeric.square().sum()),
                minimum=float(numeric.min()) if numeric.numel() else None,
                maximum=float(numeric.max()) if numeric.numel() else None)


def select_history_slots(state, batch_index, absolute_stage_index):
    occupied = state.occupied[batch_index].nonzero(as_tuple=True)[0].tolist()

    def priority(slot):
        return (-float(state.confidence[batch_index, slot]),
                -int(state.last_seen[batch_index, slot]),
                int(state.logical_ids[batch_index, slot]), slot)

    recent = [s for s in occupied if int(state.last_seen[batch_index, s]) == absolute_stage_index - 1]
    dormant = [s for s in occupied if int(state.last_seen[batch_index, s]) < absolute_stage_index - 1]
    if len(recent) + len(dormant) != len(occupied):
        raise ValueError("history contains an observation from the current or future stage")
    selected = sorted(recent, key=priority)[:16] + sorted(dormant, key=priority)[:16]
    selected += sorted((s for s in occupied if s not in selected), key=priority)[:32-len(selected)]
    selected.sort(key=lambda s: (int(state.logical_ids[batch_index, s]), s))
    return torch.tensor(selected, dtype=torch.long, device=state.embedding.device)


def segment_context(raw_coordinates, point2segment, features, padding_mask, stage_meta):
    """Derive the exact dense row geometry using input structure alone."""
    raw = raw_coordinates.to(features.device)
    p2s = point2segment.to(device=features.device, dtype=torch.long)
    if raw.ndim != 2 or raw.shape != (p2s.numel(), 4):
        raise ValueError("MPR requires aligned real XYZ and local temporal coordinates")
    count = int(p2s.max()) + 1 if p2s.numel() else 0
    if count != int((~padding_mask).sum()) or not torch.equal(
            (~padding_mask).nonzero(as_tuple=True)[0], torch.arange(count, device=features.device)):
        raise ValueError("dense feature rows and segment/padding order differ")
    if p2s.numel() and (p2s.min() < 0 or p2s.unique().numel() != count):
        raise ValueError("segment map must have contiguous nonnegative IDs")
    sums = raw.new_zeros((count, 3))
    sums.index_add_(0, p2s, raw[:, :3])
    support = torch.bincount(p2s, minlength=count).to(raw.dtype)
    centers = sums / support[:, None].clamp_min(1)
    stages = stage_meta.segment_stage_ids.to(features.device)
    if stages.shape != (count,) or not torch.equal(raw[:, 3], stages[p2s].to(raw.dtype)):
        raise ValueError("label-free StageMeta and raw segment times disagree")
    latest = int(stages.max()) if count else 0
    return {"features": features[:count], "centers": centers, "stage_ids": stages,
            "current_indices": (stages == latest).nonzero(as_tuple=True)[0],
            "latest_local_stage": latest}


def geometry6(points):
    """Population covariance/radius summary in real input coordinate units."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("geometry points must be [n,3]")
    if not points.shape[0]:
        return points.new_zeros(6), points.new_zeros(3), False
    center = points.mean(0)
    if points.shape[0] == 1:
        return points.new_zeros(6), center, True
    centered = points - center
    covariance = centered.T @ centered / points.shape[0]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).sqrt()
    distances = centered.norm(dim=1)
    radial = torch.stack((distances.mean(), distances.std(unbiased=False), distances.max()))
    return torch.cat((eigenvalues, radial)).log1p(), center, True


def nearest_indices(centers, anchor, maximum=32):
    # Input rows are in original segment-index order; stable argsort resolves ties.
    return ((centers - anchor).square().sum(-1)).argsort(stable=True)[:maximum]


class PromptAdapter(nn.Module):
    def __init__(self, *, retrieval=True, seed=146):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stable_seed(seed, "mpr-common-prompt"))
            self.input_layer = nn.Linear(270, 128)
            self.output_layer = nn.Linear(128, 128)
            nn.init.zeros_(self.output_layer.weight)
            nn.init.zeros_(self.output_layer.bias)
        if retrieval:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stable_seed(seed, "mpr-common-retrieval"))
                self.Wq = nn.Linear(128, 128)
                self.Wk = nn.Linear(128, 128)

    def forward(self, source, z, source_geometry, current_geometry, valid_flags):
        value = torch.cat((source, z, source_geometry, current_geometry, valid_flags), -1)
        return self.output_layer(F.gelu(self.input_layer(value)))

    def retrieve(self, source, features, centers):
        logits = F.normalize(self.Wq(source), dim=-1) @ F.normalize(self.Wk(features), dim=-1).T / .1
        if features.shape[0] == 0:
            return {"logits": logits, "empty": True}
        anchors = logits.argmax(-1)
        neighbors = torch.stack([nearest_indices(centers, centers[a]) for a in anchors])
        selected = logits.gather(1, neighbors)
        z = (selected.softmax(-1)[..., None] * features[neighbors]).sum(1)
        geometry = torch.stack([geometry6(centers[row])[0] for row in neighbors])
        return {"logits": logits, "anchor_indices": anchors, "neighbor_indices": neighbors,
                "z": z, "centers": centers[anchors], "geometry": geometry,
                "geometry_valid": torch.ones(source.shape[0], dtype=torch.bool, device=source.device),
                "empty": False}


class SupportHead(nn.Module):
    def __init__(self, *, seed=146):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stable_seed(seed, "mpr-support"))
            self.layers = nn.Sequential(nn.Linear(272, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.constant_(self.layers[-1].bias, 2.)

    def forward(self, value):
        return self.layers(value).squeeze(-1).sigmoid()
