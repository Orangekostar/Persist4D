import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max, scatter_min, scatter_sum, scatter_softmax
from models.layers import MLP

class AdaptiveScatter(nn.Module): 
    def __init__(self, scatter_type="mean",
        feat_dim=128, p=3, eps=1e-6
    ):
        super(AdaptiveScatter, self).__init__()
        self.scatter_type = scatter_type
        self.feat_dim = feat_dim

        if self.scatter_type == "adaptive":
            # initalize parameters 
            self.mean_mlp = MLP(in_features=feat_dim, out_features=feat_dim, hidden_features=[feat_dim, feat_dim])
            self.max_mlp = MLP(in_features=feat_dim, out_features=feat_dim, hidden_features=[feat_dim, feat_dim])
            self.merge_mlp = MLP(in_features=feat_dim*2, out_features=feat_dim)
        elif self.scatter_type == "gem":
            self.p = nn.Parameter(torch.ones(1)*p)
            self.eps = eps

    def forward(self, x, index):
        if self.scatter_type == "mean":
            return self.mean_scatter(x, index)
        elif self.scatter_type == "max":
            return self.max_scatter(x, index)
        elif self.scatter_type == "adaptive":
            return self.adaptive_scatter(x, index)
        elif self.scatter_type == "gem":
            return self.gem_scatter(x, index)
        else:
            assert False, "Scatter function not known"

    def mean_scatter(self, x, index):
        return scatter_mean(x, index, dim=0)

    def max_scatter(self, x, index):
        return scatter_max(x, index, dim=0)[0]

    def mode_scatter(self, values, index, num_categories=None, ignore_index=-1):
        """
        Compute per-group mode (majority vote) for integer `values`.

        Args:
            values (torch.Tensor): shape [N], integer-like values (e.g. class ids).
            index (torch.Tensor): shape [N], group ids (e.g. point2segment).
            num_categories (int|None): number of categories for bincount. If None, inferred from max(values)+1.
            ignore_index (int): values equal to this (or <0) are ignored.

        Returns:
            torch.Tensor: shape [S], per-group mode, with `ignore_index` where group has no valid values.
        """
        if values is None or index is None:
            return None

        if values.numel() == 0 or index.numel() == 0:
            return values.new_zeros((0,), dtype=torch.long)

        values = values.to(dtype=torch.long)
        index = index.to(dtype=torch.long)

        # Filter invalid values
        valid = values != int(ignore_index)
        valid &= values >= 0
        if not bool(valid.any()):
            num_groups = int(index.max().item()) + 1
            return torch.full((num_groups,), int(ignore_index), device=values.device, dtype=torch.long)

        v = values[valid]
        idx = index[valid]

        num_groups = int(index.max().item()) + 1
        if num_categories is None:
            num_categories = int(v.max().item()) + 1
        else:
            num_categories = int(num_categories)

        # Clamp to valid range to avoid bincount OOB
        v = torch.clamp(v, 0, num_categories - 1)

        key = idx * num_categories + v
        counts = torch.bincount(key, minlength=num_groups * num_categories).view(num_groups, num_categories)
        mode = counts.argmax(dim=1).to(torch.long)
        has_any = counts.sum(dim=1) > 0
        mode = torch.where(has_any, mode, torch.full_like(mode, int(ignore_index)))
        return mode

    def adaptive_scatter(self, x, index):
        x_mean = self.mean_scatter(x, index)
        x_max = self.max_scatter(x, index)

        x_mean = self.mean_mlp((x_mean[index]-x))
        x_mean = scatter_sum(scatter_softmax(x_mean, index, dim=0) * (x), index, dim=0)

        x_max = self.max_mlp((x_max[index]-x))
        x_max = scatter_sum(scatter_softmax(x_max, index, dim=0) * (x), index, dim=0)

        x = self.merge_mlp(torch.cat([x_mean, x_max], dim=-1))
        return x

    def gem_scatter(self, x, index):
        # GeM pooling for scatter aggregation
        # Apply power operation with learnable parameter p
        x_powered = x.clamp(min=self.eps).pow(self.p)
        
        # Scatter sum the powered values
        x_sum = scatter_sum(x_powered, index, dim=0)
        
        # Count number of elements per group for normalization
        counts = scatter_sum(torch.ones_like(x_powered), index, dim=0)
        
        # Apply GeM formula: (sum(x^p) / count)^(1/p)
        x_gem = (x_sum / counts.clamp(min=1)).pow(1.0 / self.p)
        
        return x_gem

def gem(x, p=3, eps=1e-6):
    # Original GeM for 2D pooling (kept for compatibility)
    # based on https://github.com/filipradenovic/cnnimageretrieval-pytorch implementation 
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
