"""Explicit evaluation-only execution policy for the existing FP32 mean."""
import torch
from torch_scatter import segment_csr


def stable_group_mean(x, index):
    if index.numel() == 0:
        return x.new_zeros((0, *x.shape[1:]))
    groups = int(index.max()) + 1
    order = torch.argsort(index, stable=True)
    counts = torch.bincount(index, minlength=groups)
    pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return segment_csr(x.index_select(0, order), pointers, reduce="mean")


def install_evaluation_policy():
    """Called only in dedicated evaluation/P5 children, never a training child."""
    from models.scatter import AdaptiveScatter
    AdaptiveScatter.mean_scatter = lambda self, x, index: stable_group_mean(x, index)
