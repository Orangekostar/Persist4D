"""Training-only stage sampling and query-conditioned segment feedback."""

import torch
from torch import nn


def stage_stratified_indices(stage_ids, count, *, generator):
    if stage_ids.ndim != 1 or stage_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("stage IDs must be an aligned integer vector")
    if not 0 <= count <= stage_ids.numel() or generator is None:
        raise ValueError("sampling requires a legal count and dedicated generator")
    stages, capacities = torch.unique(stage_ids, sorted=True, return_counts=True)
    capacities = capacities.tolist()
    quotas = [0] * len(capacities)
    remaining = count
    while remaining:
        for index, capacity in enumerate(capacities):
            if quotas[index] < capacity:
                quotas[index] += 1
                remaining -= 1
                if not remaining:
                    break
    selected = []
    for stage, quota in zip(stages, quotas, strict=True):
        candidates = torch.where(stage_ids == stage)[0]
        permutation = torch.randperm(candidates.numel(), device=stage_ids.device,
                                     generator=generator)
        selected.append(candidates[permutation[:quota]])
    return torch.cat(selected) if selected else stage_ids.new_empty(0, dtype=torch.long)


class QueryConditionedMaskFeedback(nn.Module):
    def __init__(self, width=128, heads=8, ffn_width=512, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.checkpoint_chunks = False
        self.feature_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(),
                                 nn.Linear(ffn_width, width))
        # MultiheadAttention includes its own output projection; this is W_o.
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    @property
    def output_projection(self):
        return self.attention.out_proj

    def _update_chunk(self, chunk, normalized_queries):
        attention, _ = self.attention(self.feature_norm(chunk), normalized_queries,
                                      normalized_queries, need_weights=False)
        updated = chunk + attention
        return updated + self.ffn(self.ffn_norm(updated))

    def forward(self, queries, mask_features, padding_mask):
        if mask_features.ndim != 3 or queries.ndim != 3:
            raise ValueError("feedback requires batched segment features and queries")
        if padding_mask.shape != mask_features.shape[:2]:
            raise ValueError("feedback padding does not align with features")
        features = mask_features.masked_fill(padding_mask[..., None], 0)
        normalized_queries = self.query_norm(queries)
        chunks = []
        for start in range(0, features.shape[1], self.chunk_size):
            chunk = features[:, start:start + self.chunk_size]
            if self.checkpoint_chunks and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                updated = checkpoint(self._update_chunk, chunk, normalized_queries,
                                     use_reentrant=False)
            else:
                updated = self._update_chunk(chunk, normalized_queries)
            chunks.append(updated)
        return torch.cat(chunks, dim=1).masked_fill(padding_mask[..., None], 0)
