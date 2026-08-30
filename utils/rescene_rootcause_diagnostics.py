"""Deterministic tensor diagnostics for the ReScene decoder."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils.rescene_rootcause_evaluation import RootCauseEvaluationError


def _tensor(value: object, *, dimensions: int) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != dimensions
        or (value.is_floating_point() and not torch.isfinite(value).all().item())
    ):
        raise RootCauseEvaluationError("diagnostic tensor contract differs")
    return value.detach()


def _target_tensors(
    target: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masks = _tensor(target.get("masks"), dimensions=2).bool()
    point2segment = _tensor(target.get("point2segment"), dimensions=1).long()
    instance_ids = _tensor(target.get("ids"), dimensions=1).long()
    labels = _tensor(target.get("labels"), dimensions=1).long()
    if (
        masks.shape[0] == 0
        or masks.shape[1] != point2segment.numel()
        or instance_ids.numel() != masks.shape[0]
        or labels.numel() != masks.shape[0]
        or point2segment.numel() == 0
        or point2segment.min().item() < 0
    ):
        raise RootCauseEvaluationError("diagnostic tensor contract differs")
    return masks, point2segment, instance_ids, labels


def _size_bin(size_points: int) -> str:
    if size_points < 100:
        return "small_lt100"
    if size_points < 1000:
        return "medium_100_999"
    return "large_ge1000"


def query_initialization_records(
    *,
    file_name: str,
    sampled_indices: torch.Tensor,
    query_content_norms: torch.Tensor,
    target: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Measure evaluated-GT coverage at non-parametric FPS query positions."""

    masks, _, instance_ids, labels = _target_tensors(target)
    sampled_indices = _tensor(sampled_indices, dimensions=1).long().cpu()
    query_content_norms = _tensor(query_content_norms, dimensions=1).float().cpu()
    masks = masks.cpu()
    if (
        sampled_indices.numel() == 0
        or sampled_indices.numel() != query_content_norms.numel()
        or sampled_indices.min().item() < 0
        or sampled_indices.max().item() >= masks.shape[1]
    ):
        raise RootCauseEvaluationError("diagnostic tensor contract differs")
    memberships = masks[:, sampled_indices]
    foreground_queries = memberships.any(dim=0)
    query_counts = memberships.sum(dim=1)
    covered = query_counts > 0
    rows: list[dict[str, object]] = [
        {
            "record_type": "scene_summary",
            "file_name": file_name,
            "num_queries": sampled_indices.numel(),
            "foreground_query_fraction": foreground_queries.float().mean().item(),
            "background_query_fraction": (~foreground_queries).float().mean().item(),
            "gt_instance_count": masks.shape[0],
            "gt_instance_coverage": covered.float().mean().item(),
            "query_content_norm_mean": query_content_norms.mean().item(),
            "query_content_norm_max": query_content_norms.max().item(),
            "query_content_zero_fraction": (
                query_content_norms <= 1e-12
            ).float().mean().item(),
        }
    ]
    for index in range(masks.shape[0]):
        size_points = int(masks[index].sum().item())
        rows.append(
            {
                "record_type": "gt_instance",
                "file_name": file_name,
                "gt_instance_id": int(instance_ids[index].item()),
                "gt_label": int(labels[index].item()),
                "size_points": size_points,
                "size_bin": _size_bin(size_points),
                "query_count": int(query_counts[index].item()),
                "covered_by_fps_query": bool(covered[index].item()),
            }
        )
    return rows


def _prediction_layers(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    auxiliary = output.get("aux_outputs")
    if not isinstance(auxiliary, list) or any(
        not isinstance(layer, Mapping) for layer in auxiliary
    ):
        raise RootCauseEvaluationError("diagnostic prediction contract differs")
    return [*auxiliary, output]


def _batch_prediction(
    layer: Mapping[str, Any], *, batch_index: int, target: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    masks, point2segment, instance_ids, _ = _target_tensors(target)
    predicted_masks = layer.get("pred_masks")
    predicted_logits = layer.get("pred_logits")
    if (
        not isinstance(predicted_masks, list)
        or batch_index >= len(predicted_masks)
        or not isinstance(predicted_logits, torch.Tensor)
        or predicted_logits.ndim != 3
        or batch_index >= predicted_logits.shape[0]
    ):
        raise RootCauseEvaluationError("diagnostic prediction contract differs")
    segment_logits = _tensor(predicted_masks[batch_index], dimensions=2).float()
    class_logits = _tensor(predicted_logits[batch_index], dimensions=2).float()
    if (
        point2segment.max().item() >= segment_logits.shape[0]
        or segment_logits.shape[1] != class_logits.shape[0]
        or class_logits.shape[1] < 2
    ):
        raise RootCauseEvaluationError("diagnostic tensor contract differs")
    point_logits = segment_logits[point2segment]
    predicted = point_logits.sigmoid().transpose(0, 1) >= 0.5
    gt_masks = masks.bool()
    intersection = (predicted[:, None, :] & gt_masks[None, :, :]).sum(dim=2).float()
    union = (predicted[:, None, :] | gt_masks[None, :, :]).sum(dim=2).float()
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    foreground_scores = class_logits.softmax(dim=-1)[:, :-1].max(dim=-1).values
    return predicted, gt_masks, iou, foreground_scores, instance_ids


def _pairwise_mask_iou(masks: torch.Tensor) -> torch.Tensor:
    intersection = (masks[:, None, :] & masks[None, :, :]).sum(dim=2).float()
    union = (masks[:, None, :] | masks[None, :, :]).sum(dim=2).float()
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def query_conflict_records(
    *, file_name: str, output: Mapping[str, Any], target: Mapping[str, Any]
) -> list[dict[str, object]]:
    """Compute project-defined query redundancy diagnostics at every prediction head."""

    rows = []
    layers = _prediction_layers(output)
    for layer_index, layer in enumerate(layers):
        predicted, _, iou, scores, _ = _batch_prediction(
            layer, batch_index=0, target=target
        )
        query_count, gt_count = iou.shape
        active = predicted.any(dim=1)
        best_iou, best_gt = iou.max(dim=1)
        eligible = active & (best_iou >= 0.25)
        competed = torch.zeros(query_count, dtype=torch.bool, device=iou.device)
        pairwise_values = []
        for gt_index in range(gt_count):
            query_indices = torch.nonzero(
                eligible & (best_gt == gt_index), as_tuple=True
            )[0]
            if query_indices.numel() > 1:
                group_scores = scores[query_indices]
                maximum_index = query_indices[group_scores.argmax()]
                competed[query_indices] = True
                competed[maximum_index] = False
                pairwise = _pairwise_mask_iou(predicted[query_indices])
                upper = torch.triu_indices(
                    query_indices.numel(), query_indices.numel(), offset=1
                )
                pairwise_values.extend(pairwise[upper[0], upper[1]].tolist())
        queries_per_gt25 = (iou >= 0.25).sum(dim=0).float()
        queries_per_gt50 = (iou >= 0.5).sum(dim=0).float()
        eligible_count = int(eligible.sum().item())
        rows.append(
            {
                "file_name": file_name,
                "decoder_prediction_layer": layer_index,
                "feeds_next_attention": layer_index < len(layers) - 1,
                "query_count": query_count,
                "active_query_count": int(active.sum().item()),
                "gt_instance_count": gt_count,
                "gt_coverage_iou25": (queries_per_gt25 > 0).float().mean().item(),
                "gt_coverage_iou50": (queries_per_gt50 > 0).float().mean().item(),
                "mean_queries_per_gt_iou25": queries_per_gt25.mean().item(),
                "mean_queries_per_gt_iou50": queries_per_gt50.mean().item(),
                "competed_active_query_fraction": (
                    competed.sum().item() / eligible_count if eligible_count else 0.0
                ),
                "competing_query_pairwise_iou_mean": (
                    sum(pairwise_values) / len(pairwise_values)
                    if pairwise_values
                    else 0.0
                ),
                "distinct_gt_covered_iou25": int((queries_per_gt25 > 0).sum().item()),
                "query_utilization_iou25": eligible_count / query_count,
                "distinct_gt_per_utilized_query": (
                    int((queries_per_gt25 > 0).sum().item()) / eligible_count
                    if eligible_count
                    else 0.0
                ),
            }
        )
    return rows


def _one_to_one_matches(iou: torch.Tensor) -> list[tuple[int, int]]:
    if iou.numel() == 0:
        return []
    query_indices, gt_indices = linear_sum_assignment(-iou.detach().cpu().numpy())
    return list(zip(query_indices.tolist(), gt_indices.tolist()))


def attention_mask_records(
    *,
    file_name: str,
    output: Mapping[str, Any],
    target: Mapping[str, Any],
    reset_counts: Sequence[Mapping[str, int]],
) -> list[dict[str, object]]:
    """Measure matched-GT recall of masks that gate the following decoder layer."""

    layers = _prediction_layers(output)[:-1]
    if len(layers) != len(reset_counts):
        raise RootCauseEvaluationError("diagnostic reset contract differs")
    rows = []
    for layer_index, (layer, reset) in enumerate(zip(layers, reset_counts)):
        predicted, gt_masks, iou, _, instance_ids = _batch_prediction(
            layer, batch_index=0, target=target
        )
        reset_count = reset.get("reset_count")
        query_count = reset.get("query_count")
        if (
            not isinstance(reset_count, int)
            or not isinstance(query_count, int)
            or query_count != predicted.shape[0]
            or not 0 <= reset_count <= query_count
        ):
            raise RootCauseEvaluationError("diagnostic reset contract differs")
        for query_index, gt_index in _one_to_one_matches(iou):
            gt_mask = gt_masks[gt_index]
            allowed = predicted[query_index] & gt_mask
            allowed_fraction = allowed.sum().item() / gt_mask.sum().item()
            rows.append(
                {
                    "file_name": file_name,
                    "decoder_prediction_layer": layer_index,
                    "query_id": query_index,
                    "gt_instance_id": int(instance_ids[gt_index].item()),
                    "match_iou": iou[query_index, gt_index].item(),
                    "gt_point_count": int(gt_mask.sum().item()),
                    "allowed_gt_fraction": allowed_fraction,
                    "masked_gt_fraction": 1.0 - allowed_fraction,
                    "post_sample_all_masked_reset_count": reset_count,
                    "post_sample_query_count": query_count,
                    "post_sample_reset_fraction": reset_count / query_count,
                }
            )
    return rows


def superpoint_feature_records(
    *,
    file_name: str,
    segment_features: torch.Tensor,
    target: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Measure GT-conditioned superpoint compactness, margin, and purity."""

    masks, point2segment, instance_ids, labels = _target_tensors(target)
    features = _tensor(segment_features, dimensions=2).float()
    if point2segment.max().item() >= features.shape[0] or features.shape[0] == 0:
        raise RootCauseEvaluationError("diagnostic tensor contract differs")
    features = F.normalize(features, dim=1, eps=1e-12)
    masks = masks.to(features.device)
    point2segment = point2segment.to(features.device)
    segment_ids_by_gt = [
        point2segment[masks[index]].unique() for index in range(masks.shape[0])
    ]
    centroids = torch.stack(
        [features[segment_ids].mean(dim=0) for segment_ids in segment_ids_by_gt]
    )
    normalized_centroids = F.normalize(centroids, dim=1, eps=1e-12)
    centroid_similarity = normalized_centroids @ normalized_centroids.T
    centroid_similarity.fill_diagonal_(-torch.inf)
    segment_point_counts = torch.bincount(
        point2segment, minlength=features.shape[0]
    ).float()
    instance_segment_counts = torch.zeros(
        masks.shape[0], features.shape[0], device=features.device
    )
    for gt_index in range(masks.shape[0]):
        instance_segment_counts[gt_index].scatter_add_(
            0,
            point2segment,
            masks[gt_index].float(),
        )
    segment_purity = instance_segment_counts.max(dim=0).values / segment_point_counts
    gt_instances_per_segment = (instance_segment_counts > 0).sum(dim=0).float()
    rows = []
    for gt_index, segment_ids in enumerate(segment_ids_by_gt):
        deviations = features[segment_ids] - centroids[gt_index]
        nearest_margin: float | None = None
        if masks.shape[0] > 1:
            nearest_margin = 1.0 - centroid_similarity[gt_index].max().item()
            if not math.isfinite(nearest_margin):
                raise RootCauseEvaluationError("diagnostic feature margin is invalid")
        rows.append(
            {
                "file_name": file_name,
                "gt_instance_id": int(instance_ids[gt_index].item()),
                "gt_label": int(labels[gt_index].item()),
                "size_points": int(masks[gt_index].sum().item()),
                "size_bin": _size_bin(int(masks[gt_index].sum().item())),
                "segments_per_gt": segment_ids.numel(),
                "within_instance_feature_variance": (
                    deviations.square().sum(dim=1).mean().item()
                ),
                "nearest_instance_cosine_margin": nearest_margin,
                "mean_segment_purity": segment_purity[segment_ids].mean().item(),
                "mean_gt_instances_per_segment": (
                    gt_instances_per_segment[segment_ids].mean().item()
                ),
            }
        )
    return rows
