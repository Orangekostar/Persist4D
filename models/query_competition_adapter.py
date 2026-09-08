"""Minimal QCL-inspired update driven only by current predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import torch
from torch import Tensor, nn


class QueryCompetitionError(RuntimeError):
    """Raised when the fixed query-competition contract is violated."""


class QueryCompetitionAdapter(nn.Module):
    hidden_dim = 128
    num_queries = 100

    def __init__(
        self,
        *,
        support_budget: int = 4096,
        overlap_threshold: float = 0.25,
    ) -> None:
        super().__init__()
        if (
            isinstance(support_budget, bool)
            or not isinstance(support_budget, int)
            or support_budget <= 0
        ):
            raise QueryCompetitionError("support budget must be a positive integer")
        if not 0.0 < overlap_threshold < 1.0:
            raise QueryCompetitionError("overlap threshold must be within (0, 1)")
        self.support_budget = support_budget
        self.overlap_threshold = overlap_threshold
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.message_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self._last_diagnostics: Mapping[str, float | int] = MappingProxyType({})

    @property
    def last_diagnostics(self) -> Mapping[str, float | int]:
        return self._last_diagnostics

    def _validate(
        self,
        queries: Tensor,
        class_logits: Tensor,
        mask_logits: Tensor,
        padding_mask: Tensor,
    ) -> None:
        if (
            queries.ndim != 3
            or queries.shape[1:] != (self.num_queries, self.hidden_dim)
            or class_logits.ndim != 3
            or class_logits.shape[:2] != queries.shape[:2]
            or class_logits.shape[-1] < 2
            or mask_logits.ndim != 3
            or mask_logits.shape[0] != queries.shape[0]
            or mask_logits.shape[2] != self.num_queries
            or padding_mask.shape != mask_logits.shape[:2]
            or padding_mask.dtype != torch.bool
            or any(
                tensor.device != queries.device
                for tensor in (class_logits, mask_logits, padding_mask)
            )
            or not all(
                tensor.is_floating_point()
                for tensor in (queries, class_logits, mask_logits)
            )
            or not all(
                torch.isfinite(tensor).all().item()
                for tensor in (queries, class_logits, mask_logits)
            )
        ):
            raise QueryCompetitionError("prediction tensors violate the L contract")

    def _support_weights(
        self,
        mask_logits: Tensor,
        padding_mask: Tensor,
        point2segment: Sequence[Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, support_count, query_count = mask_logits.shape
        weights = (~padding_mask).to(dtype=mask_logits.dtype)
        if point2segment is not None:
            if len(point2segment) != batch_size:
                raise QueryCompetitionError("point-to-segment batch size differs")
            for batch_index, mapping in enumerate(point2segment):
                valid_count = int((~padding_mask[batch_index]).sum().item())
                if (
                    not isinstance(mapping, Tensor)
                    or mapping.ndim != 1
                    or mapping.numel() == 0
                    or torch.any(mapping < 0).item()
                ):
                    raise QueryCompetitionError("point-to-segment mapping is invalid")
                counts = torch.bincount(
                    mapping.to(device=mask_logits.device, dtype=torch.long),
                    minlength=valid_count,
                )
                if counts.numel() != valid_count or torch.any(counts == 0).item():
                    raise QueryCompetitionError("segment support coverage differs")
                weights[batch_index].zero_()
                weights[batch_index, :valid_count] = counts.to(mask_logits.dtype)

        selected_indices = []
        selected_weights = []
        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(
                weights[batch_index] > 0, as_tuple=False
            ).flatten()
            if valid_indices.numel() == 0:
                raise QueryCompetitionError("prediction support is empty")
            if valid_indices.numel() > self.support_budget:
                positions = torch.div(
                    torch.arange(self.support_budget, device=mask_logits.device)
                    * valid_indices.numel(),
                    self.support_budget,
                    rounding_mode="floor",
                )
                valid_indices = valid_indices[positions]
            selected_indices.append(valid_indices)
            selected_weights.append(weights[batch_index, valid_indices])

        selected_count = max(index.numel() for index in selected_indices)
        indices = torch.zeros(
            batch_size,
            selected_count,
            dtype=torch.long,
            device=mask_logits.device,
        )
        sampled_weights = torch.zeros(
            batch_size,
            selected_count,
            dtype=mask_logits.dtype,
            device=mask_logits.device,
        )
        for batch_index, (index, weight) in enumerate(
            zip(selected_indices, selected_weights, strict=True)
        ):
            indices[batch_index, : index.numel()] = index
            sampled_weights[batch_index, : weight.numel()] = weight
        sampled_logits = mask_logits.gather(
            1, indices[:, :, None].expand(-1, -1, query_count)
        )
        if sampled_logits.shape[1] > support_count:
            raise QueryCompetitionError("sampled support exceeds prediction support")
        return sampled_logits, sampled_weights

    def forward(
        self,
        queries: Tensor,
        *,
        class_logits: Tensor,
        mask_logits: Tensor,
        padding_mask: Tensor,
        point2segment: Sequence[Tensor] | None,
    ) -> Tensor:
        self._validate(queries, class_logits, mask_logits, padding_mask)
        sampled_logits, support_weights = self._support_weights(
            mask_logits, padding_mask, point2segment
        )
        mask_probability = sampled_logits.sigmoid().transpose(1, 2)
        binary_masks = mask_probability.detach() >= 0.5
        weighted_binary = binary_masks.to(queries.dtype) * support_weights[:, None, :]
        intersection = torch.bmm(weighted_binary, binary_masks.to(queries.dtype).transpose(1, 2))
        mass = weighted_binary.sum(dim=-1)
        union = mass[:, :, None] + mass[:, None, :] - intersection
        overlap = torch.where(union > 0, intersection / union.clamp_min(1e-6), 0.0)

        class_probability = class_logits.softmax(dim=-1)[..., :-1]
        class_confidence, predicted_class = class_probability.max(dim=-1)
        foreground_support = weighted_binary.sum(dim=-1)
        mask_confidence = (
            (mask_probability * weighted_binary).sum(dim=-1)
            / foreground_support.clamp_min(1e-6)
        )
        quality = class_confidence * mask_confidence
        query_index = torch.arange(self.num_queries, device=queries.device)
        higher_quality = quality[:, None, :] > quality[:, :, None]
        tied_quality = torch.isclose(
            quality[:, None, :], quality[:, :, None], rtol=0.0, atol=1e-8
        )
        earlier_query = query_index[None, None, :] < query_index[None, :, None]
        ranks_higher = higher_quality | (tied_quality & earlier_query)
        same_class = predicted_class[:, :, None] == predicted_class[:, None, :]
        foreground = class_logits.argmax(dim=-1) != class_logits.shape[-1] - 1
        competing = (
            (overlap >= self.overlap_threshold)
            & same_class
            & foreground[:, :, None]
            & foreground[:, None, :]
            & ranks_higher
        )
        leader_weights = (
            overlap * quality[:, None, :] * competing.to(queries.dtype)
        )
        leader_mass = leader_weights.sum(dim=-1, keepdim=True)
        normalized_weights = leader_weights / leader_mass.clamp_min(1e-6)
        leader_context = torch.bmm(normalized_weights, queries)
        message = self.message_projection(
            self.query_norm(queries - leader_context)
        )
        gate = torch.sigmoid(
            self.gate_projection(torch.cat([queries, leader_context], dim=-1))
        )
        has_leader = leader_mass > 0
        delta = gate * self.output_projection(message) * has_leader
        output = queries + delta
        self._last_diagnostics = MappingProxyType(
            {
                "competing_pair_count": int(competing.sum().detach().item()),
                "competing_query_count": int(has_leader.sum().detach().item()),
                "gate_mean": float(gate[has_leader.expand_as(gate)].mean().detach().item())
                if has_leader.any().item()
                else 0.0,
                "sampled_support_count": int(sampled_logits.shape[1]),
                "update_norm": float(delta.norm(dim=-1).mean().detach().item()),
            }
        )
        return output


__all__ = ["QueryCompetitionAdapter", "QueryCompetitionError"]
