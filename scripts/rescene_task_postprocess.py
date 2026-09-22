"""Reusable ReScene official task post-processing with candidate lineage."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


class TaskPostprocessError(ValueError):
    """Raised when official task post-processing inputs or outputs are invalid."""


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TaskPostprocessError(f"{name} must be a mapping")
    return value


def _tensor(value: object, *, name: str, ndim: int | None = None) -> Tensor:
    if not isinstance(value, Tensor):
        raise TaskPostprocessError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise TaskPostprocessError(f"{name} must have rank {ndim}")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise TaskPostprocessError(f"{name} must be finite")
    return value


def _method(system: object, name: str) -> Callable[..., object]:
    method = getattr(system, name, None)
    if not callable(method):
        raise TaskPostprocessError(f"system lacks official ReScene method {name}")
    return method


def _map_classes(classes: Tensor, mapper: Callable[[int], int]) -> Tensor:
    values = []
    for value in classes.detach().cpu().long().tolist():
        mapped = mapper(int(value))
        if isinstance(mapped, bool) or not isinstance(mapped, int):
            raise TaskPostprocessError("class mapper must return integer labels")
        values.append(mapped)
    return torch.tensor(values, dtype=torch.long)


def _temporal_stages(value: object) -> Tensor:
    stages = _tensor(value, name="target temporal_stages", ndim=1).detach().cpu()
    if stages.is_floating_point():
        rounded = stages.round()
        if not torch.equal(stages, rounded):
            raise TaskPostprocessError("temporal stages must be integer valued")
        stages = rounded
    return stages.long().contiguous()


@dataclass(frozen=True)
class OfficialTaskSoftEvidence:
    representation: str
    segment_logits: Tensor
    low_resolution_logits: Tensor
    full_resolution_logits: Tensor
    full_resolution_probabilities: Tensor
    class_probabilities: Tensor
    query_features: Tensor
    segment_features: Tensor
    source_query_ids: Tensor
    source_class_ids: Tensor
    low_point2segment: Tensor
    voxel_inverse: Tensor
    full_point2segment: Tensor

    def validate(self) -> None:
        if self.representation != "FP32_SEGMENT_LOGITS":
            raise TaskPostprocessError("soft evidence representation differs")
        candidate_count = self.source_query_ids.numel()
        matrices = (
            ("segment_logits", self.segment_logits, 1),
            ("low_resolution_logits", self.low_resolution_logits, 1),
            ("full_resolution_logits", self.full_resolution_logits, 1),
            (
                "full_resolution_probabilities",
                self.full_resolution_probabilities,
                1,
            ),
            ("class_probabilities", self.class_probabilities, 0),
            ("query_features", self.query_features, 0),
        )
        for name, value, candidate_axis in matrices:
            if (
                value.ndim != 2
                or value.shape[candidate_axis] != candidate_count
                or not value.is_floating_point()
                or value.dtype != torch.float32
                or not torch.isfinite(value).all().item()
            ):
                raise TaskPostprocessError(f"{name} differs from the soft contract")
        if (
            self.segment_features.ndim != 2
            or self.segment_features.shape[0] != self.segment_logits.shape[0]
            or self.segment_features.dtype != torch.float32
            or not torch.isfinite(self.segment_features).all().item()
        ):
            raise TaskPostprocessError("segment_features differ from the soft contract")
        if not torch.equal(
            self.full_resolution_probabilities,
            self.full_resolution_logits.sigmoid(),
        ):
            raise TaskPostprocessError("soft evidence probability/logit values differ")
        for name, value, expected in (
            ("source_class_ids", self.source_class_ids, candidate_count),
            (
                "low_point2segment",
                self.low_point2segment,
                self.low_resolution_logits.shape[0],
            ),
            ("voxel_inverse", self.voxel_inverse, self.full_resolution_logits.shape[0]),
            (
                "full_point2segment",
                self.full_point2segment,
                self.full_resolution_logits.shape[0],
            ),
        ):
            if (
                value.ndim != 1
                or value.numel() != expected
                or value.is_floating_point()
            ):
                raise TaskPostprocessError(f"{name} differs from the soft contract")


@dataclass(frozen=True)
class OfficialTaskPrediction:
    pred_masks: Tensor
    pred_scores: Tensor
    pred_classes: Tensor
    source_query_ids: Tensor
    source_class_ids: Tensor
    temporal_stages: Tensor
    latest_stage_index: int
    latest_stage_masks: Tensor
    soft_evidence: OfficialTaskSoftEvidence | None = None

    def validate(self) -> None:
        candidate_count = self.pred_scores.numel()
        if self.pred_masks.dtype != torch.bool or self.pred_masks.ndim != 2:
            raise TaskPostprocessError("pred_masks must be a rank-2 bool tensor")
        if self.pred_masks.shape[1] != candidate_count:
            raise TaskPostprocessError("task candidate dimensions differ")
        for name, value in (
            ("pred_scores", self.pred_scores),
            ("pred_classes", self.pred_classes),
            ("source_query_ids", self.source_query_ids),
            ("source_class_ids", self.source_class_ids),
        ):
            if value.ndim != 1 or value.numel() != candidate_count:
                raise TaskPostprocessError(f"{name} must align with task candidates")
        if (
            not self.pred_scores.is_floating_point()
            or not torch.isfinite(self.pred_scores).all().item()
        ):
            raise TaskPostprocessError("pred_scores must be finite floating values")
        for name, value in (
            ("pred_classes", self.pred_classes),
            ("source_query_ids", self.source_query_ids),
            ("source_class_ids", self.source_class_ids),
            ("temporal_stages", self.temporal_stages),
        ):
            if value.dtype == torch.bool or value.is_floating_point():
                raise TaskPostprocessError(f"{name} must use integer dtype")
        if (
            self.temporal_stages.ndim != 1
            or self.temporal_stages.numel() != self.pred_masks.shape[0]
        ):
            raise TaskPostprocessError(
                "temporal stages must cover all prediction points"
            )
        selector = self.temporal_stages == self.latest_stage_index
        if not selector.any().item():
            raise TaskPostprocessError("latest stage is absent from task points")
        expected_latest = self.pred_masks[selector]
        if not torch.equal(self.latest_stage_masks, expected_latest):
            raise TaskPostprocessError("latest-stage masks differ from full task masks")
        if self.soft_evidence is not None:
            self.soft_evidence.validate()
            if not torch.equal(
                self.soft_evidence.source_query_ids, self.source_query_ids
            ) or not torch.equal(
                self.soft_evidence.source_class_ids, self.source_class_ids
            ):
                raise TaskPostprocessError("soft evidence lineage differs")

    def prediction(self, *, latest_only: bool = False) -> dict[str, Tensor]:
        self.validate()
        return {
            "pred_masks": (
                self.latest_stage_masks.clone()
                if latest_only
                else self.pred_masks.clone()
            ),
            "pred_scores": self.pred_scores.clone(),
            "pred_classes": self.pred_classes.clone(),
            "source_query_ids": self.source_query_ids.clone(),
            "source_class_ids": self.source_class_ids.clone(),
        }


def materialize_segment_logits(
    *,
    system: object,
    evidence: OfficialTaskSoftEvidence,
    segment_logits: Tensor,
) -> Tensor:
    """Apply the frozen official segment-logit to full-resolution bool path."""

    if not isinstance(evidence, OfficialTaskSoftEvidence):
        raise TaskPostprocessError("soft evidence has the wrong type")
    evidence.validate()
    logits = _tensor(segment_logits, name="refined segment logits", ndim=2)
    if logits.shape != evidence.segment_logits.shape or logits.dtype != torch.float32:
        raise TaskPostprocessError(
            "refined segment logits must preserve the candidate matrix"
        )
    low_masks = (logits.detach().cpu()[evidence.low_point2segment] > 0).float()
    full_masks = _method(system, "_get_full_res_mask")(
        low_masks,
        evidence.voxel_inverse,
        evidence.full_point2segment,
    )
    result = torch.as_tensor(full_masks).detach().cpu().bool().contiguous()
    if result.shape != evidence.full_resolution_logits.shape:
        raise TaskPostprocessError("refined full-resolution mask dimensions differ")
    return result


def extract_official_task_prediction(
    *,
    system: object,
    output: Mapping[str, object],
    target_low_resolution: Mapping[str, object],
    target_full_resolution: Mapping[str, object],
    data: object,
    class_mapper: Callable[[int], int],
    latest_stage_index: int,
    return_soft_evidence: bool = False,
) -> OfficialTaskPrediction:
    if isinstance(latest_stage_index, bool) or not isinstance(latest_stage_index, int):
        raise TaskPostprocessError("latest_stage_index must be an integer")
    outputs = _mapping(output, name="ReScene output")
    logits = _tensor(outputs.get("pred_logits"), name="pred_logits", ndim=3)
    if logits.shape[0] != 1:
        raise TaskPostprocessError("official task helper requires batch size one")

    get_predictions = _method(system, "_get_predictions")
    get_batch_masks = _method(system, "_get_batch_masks")
    get_mask_and_scores = _method(system, "_get_mask_and_scores")
    get_full_res_mask = _method(system, "_get_full_res_mask")
    filter_predictions = _method(system, "_filter_and_sort_predictions")

    prediction = get_predictions(outputs)
    decoder_id = int(getattr(system, "decoder_id", -1))
    selected = prediction[decoder_id]
    selected_logits = _tensor(
        selected["pred_logits"], name="selected pred_logits", ndim=3
    )
    low_masks = get_batch_masks(prediction, 0, [target_low_resolution])
    scored = get_mask_and_scores(
        selected_logits[0].detach().cpu(),
        low_masks,
        selected_logits[0].shape[0],
        logits.shape[2] - 1,
        return_lineage=True,
    )
    if not isinstance(scored, tuple) or len(scored) != 6:
        raise TaskPostprocessError("official scorer did not return candidate lineage")
    scores, low_masks, classes, heatmap, source_queries, source_classes = scored

    inverse_maps = getattr(data, "inverse_maps", None)
    if (
        isinstance(inverse_maps, (str, bytes))
        or not isinstance(inverse_maps, Sequence)
        or len(inverse_maps) != 1
    ):
        raise TaskPostprocessError("collated data must contain one inverse map")
    point2segment = _tensor(
        target_full_resolution.get("point2segment"),
        name="target_full.point2segment",
        ndim=1,
    )
    full_masks = get_full_res_mask(low_masks, inverse_maps[0], point2segment)
    get_full_res_mask(
        heatmap,
        inverse_maps[0],
        point2segment,
        is_heatmap=True,
    )

    lineage = np.stack(
        [
            _tensor(source_queries, name="source_query_ids", ndim=1)
            .detach()
            .cpu()
            .numpy(),
            _tensor(source_classes, name="source_class_ids", ndim=1)
            .detach()
            .cpu()
            .numpy(),
        ],
        axis=0,
    )
    filtered_classes, filtered_masks, filtered_scores, filtered_lineage = (
        filter_predictions(
            np.asarray(full_masks),
            scores,
            classes,
            lineage,
        )
    )
    task_masks = torch.as_tensor(filtered_masks).bool().cpu().contiguous()
    task_scores = torch.as_tensor(filtered_scores).float().cpu().contiguous()
    model_classes = torch.as_tensor(filtered_classes).long().cpu().contiguous()
    filtered_lineage = torch.as_tensor(filtered_lineage).long().cpu().contiguous()
    if filtered_lineage.shape != (2, task_scores.numel()):
        raise TaskPostprocessError("filtered candidate lineage dimensions differ")
    source_query_ids = filtered_lineage[0]
    source_class_ids = filtered_lineage[1]
    if not torch.equal(model_classes, source_class_ids):
        raise TaskPostprocessError("official class IDs and source lineage differ")

    stages = _temporal_stages(target_full_resolution.get("temporal_stages"))
    soft_evidence = None
    if return_soft_evidence:
        raw_masks = outputs.get("pred_masks")
        if (
            isinstance(raw_masks, (str, bytes))
            or not isinstance(raw_masks, Sequence)
            or len(raw_masks) != 1
        ):
            raise TaskPostprocessError("soft evidence requires one raw mask tensor")
        raw_segment_logits = _tensor(
            raw_masks[0], name="raw segment logits", ndim=2
        ).detach()
        if source_query_ids.numel() and (
            source_query_ids.min().item() < 0
            or source_query_ids.max().item() >= raw_segment_logits.shape[1]
        ):
            raise TaskPostprocessError("source query lineage exceeds raw mask logits")
        mask_query_ids = source_query_ids.to(raw_segment_logits.device)
        segment_logits = (
            raw_segment_logits[:, mask_query_ids].float().cpu().contiguous()
        )
        low_point2segment = (
            _tensor(
                target_low_resolution.get("point2segment"),
                name="target_low.point2segment",
                ndim=1,
            )
            .detach()
            .cpu()
            .long()
            .contiguous()
        )
        if low_point2segment.numel() and (
            low_point2segment.min().item() < 0
            or low_point2segment.max().item() >= segment_logits.shape[0]
        ):
            raise TaskPostprocessError("low-resolution segment mapping exceeds logits")
        low_resolution_logits = segment_logits[low_point2segment]
        voxel_inverse = (
            _tensor(inverse_maps[0], name="voxel_inverse", ndim=1)
            .detach()
            .cpu()
            .long()
            .contiguous()
        )
        if voxel_inverse.numel() and (
            voxel_inverse.min().item() < 0
            or voxel_inverse.max().item() >= low_resolution_logits.shape[0]
        ):
            raise TaskPostprocessError("voxel inverse exceeds low-resolution logits")
        full_resolution_logits = low_resolution_logits[voxel_inverse].contiguous()

        query_features = _tensor(
            outputs.get("query_features"), name="query_features", ndim=3
        )
        if query_features.shape[0] != 1:
            raise TaskPostprocessError("soft query features require batch size one")
        class_query_ids = source_query_ids.to(logits.device)
        feature_query_ids = source_query_ids.to(query_features.device)
        class_probabilities = logits.float().softmax(dim=-1)[0, class_query_ids]
        candidate_query_features = query_features[0, feature_query_ids]
        feature_layers = outputs.get("segment_features")
        if (
            isinstance(feature_layers, (str, bytes))
            or not isinstance(feature_layers, Sequence)
            or not feature_layers
            or isinstance(feature_layers[-1], (str, bytes))
            or not isinstance(feature_layers[-1], Sequence)
            or len(feature_layers[-1]) != 1
        ):
            raise TaskPostprocessError("soft evidence requires segment feature layers")
        segment_features = _tensor(
            feature_layers[-1][0], name="segment_features", ndim=2
        )
        reconstructed = get_full_res_mask(
            (low_resolution_logits > 0).float(),
            voxel_inverse,
            point2segment,
        )
        if not torch.equal(torch.as_tensor(reconstructed).bool(), task_masks):
            raise TaskPostprocessError(
                "zero-residual soft logits do not reproduce official bool masks"
            )
        soft_evidence = OfficialTaskSoftEvidence(
            representation="FP32_SEGMENT_LOGITS",
            segment_logits=segment_logits,
            low_resolution_logits=low_resolution_logits.float().cpu().contiguous(),
            full_resolution_logits=full_resolution_logits.float().cpu().contiguous(),
            full_resolution_probabilities=full_resolution_logits.float()
            .sigmoid()
            .cpu()
            .contiguous(),
            class_probabilities=class_probabilities.detach().float().cpu().contiguous(),
            query_features=candidate_query_features.detach().float().cpu().contiguous(),
            segment_features=segment_features.detach().float().cpu().contiguous(),
            source_query_ids=source_query_ids.clone(),
            source_class_ids=source_class_ids.clone(),
            low_point2segment=low_point2segment,
            voxel_inverse=voxel_inverse,
            full_point2segment=point2segment.detach().cpu().long().contiguous(),
        )
    result = OfficialTaskPrediction(
        pred_masks=task_masks,
        pred_scores=task_scores,
        pred_classes=_map_classes(model_classes, class_mapper),
        source_query_ids=source_query_ids,
        source_class_ids=source_class_ids,
        temporal_stages=stages,
        latest_stage_index=latest_stage_index,
        latest_stage_masks=task_masks[stages == latest_stage_index].contiguous(),
        soft_evidence=soft_evidence,
    )
    result.validate()
    return result


__all__ = [
    "OfficialTaskPrediction",
    "OfficialTaskSoftEvidence",
    "TaskPostprocessError",
    "extract_official_task_prediction",
    "materialize_segment_logits",
]
