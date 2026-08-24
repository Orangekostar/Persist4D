"""Frozen ReScan inference and ambiguity-aware external evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from datasets.rescan_adapter import (
    IdentityAlternatives,
    RescanEvaluatorTarget,
    RescanTemporalDataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
METHOD_NAMES = {
    "B1": "Pairwise Feature Association",
    "B2": "Pairwise Feature-Class Association",
    "B3": "EMA Temporal Association",
    "B4": "Persist4D",
}


class RescanEvaluationError(ValueError):
    """Raised when external evaluation violates the frozen protocol."""


@dataclass(frozen=True)
class RescanStageTarget:
    capture_id: str
    identity_ids: Tensor
    class_ids: Tensor
    masks: Tensor
    accepted_identity_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PreparedRescanBatch:
    data: Any
    target: dict[str, Tensor]
    full_point2segment: Tensor
    full_temporal_stages: Tensor


@dataclass(frozen=True)
class StageIdentityMatches:
    raw_identity_ids: tuple[int, ...]
    predicted_track_ids: tuple[Hashable | None, ...]
    alternatives: IdentityAlternatives

    def __post_init__(self) -> None:
        if len(self.raw_identity_ids) != len(self.predicted_track_ids):
            raise RescanEvaluationError("stage identity fields must align")
        if len(set(self.raw_identity_ids)) != len(self.raw_identity_ids):
            raise RescanEvaluationError("raw identities must be unique per stage")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.raw_identity_ids
        ):
            raise RescanEvaluationError("raw identities must be non-negative integers")
        for value in self.predicted_track_ids:
            if value is None:
                continue
            try:
                hash(value)
            except TypeError as error:
                raise RescanEvaluationError(
                    "predicted track ids must be hashable"
                ) from error


def _label_mappings(value: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    raw = value.get("mappings")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise RescanEvaluationError("label map lacks mappings")
    result: dict[int, Mapping[str, object]] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise RescanEvaluationError("label mapping entries must be mappings")
        source_id = entry.get("source_class_id")
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise RescanEvaluationError("source class ids must be integers")
        if source_id in result:
            raise RescanEvaluationError("source class ids must be unique")
        result[source_id] = entry
    return result


def build_rescan_stage_target(
    target: RescanEvaluatorTarget,
    label_map: Mapping[str, object],
    *,
    level: str,
    excluded_identity_ids: Sequence[int] = (),
) -> RescanStageTarget:
    if not isinstance(target, RescanEvaluatorTarget):
        raise RescanEvaluationError("target must be a RescanEvaluatorTarget")
    if level not in {"A", "B"}:
        raise RescanEvaluationError("level must be A or B")
    mappings = _label_mappings(label_map)
    valid_point = (
        (target.instance_ids >= 0)
        & (target.instance_ids < 256)
        & (target.class_ids > 0)
    )
    if level == "B":
        valid_point &= ~np.isin(target.class_ids, [1, 2, 22])
    excluded = tuple(int(value) for value in excluded_identity_ids)
    if excluded:
        valid_point &= ~np.isin(target.instance_ids, excluded)
    if level == "A":
        exact_ids = {
            source_id
            for source_id, entry in mappings.items()
            if entry.get("status") == "exact"
        }
        valid_point &= np.isin(target.class_ids, sorted(exact_ids))
    identities = sorted(
        int(value) for value in np.unique(target.instance_ids[valid_point])
    )
    masks = []
    classes = []
    accepted = []
    for identity in identities:
        mask = valid_point & (target.instance_ids == identity)
        source_classes = np.unique(target.class_ids[mask])
        if source_classes.shape != (1,):
            raise RescanEvaluationError("one official instance spans multiple classes")
        source_class = int(source_classes[0])
        if source_class not in mappings:
            raise RescanEvaluationError("label map does not cover an encountered class")
        if level == "A":
            mapped = mappings[source_class].get("target_class_id")
            if isinstance(mapped, bool) or not isinstance(mapped, int):
                raise RescanEvaluationError("exact classes require integer target ids")
            classes.append(mapped)
        else:
            classes.append(source_class)
        masks.append(torch.from_numpy(mask.copy()).bool())
        accepted.append(
            tuple(target.ambiguities.alternatives.get(identity, (identity,)))
        )
    point_count = int(target.instance_ids.shape[0])
    return RescanStageTarget(
        capture_id=target.capture_id,
        identity_ids=torch.tensor(identities, dtype=torch.long),
        class_ids=torch.tensor(classes, dtype=torch.long),
        masks=(
            torch.stack(masks)
            if masks
            else torch.zeros((0, point_count), dtype=torch.bool)
        ),
        accepted_identity_ids=tuple(accepted),
    )


def _tensor(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 1:
        raise RescanEvaluationError(f"{name} must be a rank-1 tensor")
    return value


def prepare_rescan_model_batch(
    data: Any,
    targets: Sequence[Mapping[str, object]],
) -> PreparedRescanBatch:
    if isinstance(targets, (str, bytes)) or len(targets) != 1:
        raise RescanEvaluationError("collator must return exactly one target")
    target = targets[0]
    if not isinstance(target, Mapping):
        raise RescanEvaluationError("collated target must be a mapping")
    point2segment = _tensor(target.get("point2segment"), name="point2segment")
    temporal_stages = _tensor(target.get("temporal_stages"), name="temporal_stages")
    if point2segment.shape != temporal_stages.shape or point2segment.numel() == 0:
        raise RescanEvaluationError("low-resolution geometry fields must align")
    try:
        full_targets = data["target_full"]
    except (KeyError, TypeError) as error:
        raise RescanEvaluationError(
            "collated data lacks safe full-resolution geometry"
        ) from error
    if (
        isinstance(full_targets, (str, bytes))
        or not isinstance(full_targets, Sequence)
        or len(full_targets) != 1
        or not isinstance(full_targets[0], Mapping)
    ):
        raise RescanEvaluationError("full-resolution geometry must contain one mapping")
    full = full_targets[0]
    if full.get("ambiguities") not in (None, (), []):
        raise RescanEvaluationError(
            "ambiguity metadata entered the collated model batch"
        )
    full_point2segment = _tensor(full.get("point2segment"), name="full point2segment")
    full_temporal_stages = _tensor(
        full.get("temporal_stages"), name="full temporal_stages"
    )
    if (
        full_point2segment.shape != full_temporal_stages.shape
        or full_point2segment.numel() == 0
    ):
        raise RescanEvaluationError("full-resolution geometry fields must align")
    for key in ("labels", "original_labels", "segment2label", "target_full"):
        data.pop(key, None)
    forbidden = {
        "ambiguities",
        "class_idx",
        "class_ids",
        "instance_idx",
        "instance_ids",
        "labels",
        "object_transform",
        "stable_identity",
        "target_full",
    }
    if forbidden.intersection(data):
        raise RescanEvaluationError("ground-truth field remains in model data")
    safe_target = {
        "point2segment": point2segment,
        "temporal_stages": temporal_stages,
    }
    return PreparedRescanBatch(
        data=data,
        target=safe_target,
        full_point2segment=full_point2segment,
        full_temporal_stages=full_temporal_stages,
    )


def canonicalize_ambiguous_identities(
    *,
    raw_identity_ids: Sequence[int],
    predicted_track_ids: Sequence[Hashable | None],
    alternatives: IdentityAlternatives,
    track_identity_history: Mapping[Hashable, int],
) -> tuple[int, ...]:
    raw = tuple(raw_identity_ids)
    predicted = tuple(predicted_track_ids)
    if len(raw) != len(predicted) or len(set(raw)) != len(raw):
        raise RescanEvaluationError(
            "ambiguity assignment inputs must align and be unique"
        )
    if not raw:
        return ()
    accepted = tuple(
        tuple(alternatives.alternatives.get(identity, (identity,))) for identity in raw
    )
    candidates = sorted({value for values in accepted for value in values})
    costs = np.full((len(raw), len(candidates)), 1.0e9, dtype=np.float64)
    for row, (raw_identity, track_id, allowed) in enumerate(
        zip(raw, predicted, accepted, strict=True)
    ):
        historical = (
            track_identity_history.get(track_id) if track_id is not None else None
        )
        for column, candidate in enumerate(candidates):
            if candidate not in allowed:
                continue
            score = 0.0
            if historical == candidate:
                score += 1000.0
            if raw_identity == candidate:
                score += 10.0
            score += 1.0e-6 / (1.0 + candidate)
            costs[row, column] = -score
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != len(raw) or any(
        costs[row, column] >= 1.0e8 for row, column in zip(rows, columns, strict=True)
    ):
        raise RescanEvaluationError(
            "official ambiguity alternatives have no one-to-one assignment"
        )
    result = [0] * len(raw)
    for row, column in zip(rows, columns, strict=True):
        result[int(row)] = candidates[int(column)]
    return tuple(result)


def evaluate_identity_sequence(
    stages: Sequence[StageIdentityMatches],
) -> dict[str, int | float]:
    if isinstance(stages, (str, bytes)) or not stages:
        raise RescanEvaluationError("identity sequence must contain stages")
    track_identity_history: dict[Hashable, int] = {}
    track_identity_sets: dict[Hashable, set[int]] = {}
    identity_track_sets: dict[int, set[Hashable]] = {}
    last_visible_stage: dict[int, int] = {}
    last_matched_track: dict[int, Hashable] = {}
    eligible: set[int] = set()
    observation_count = matched_count = transition_count = switch_count = 0
    gap_count = recovery_attempts = correct_recoveries = 0
    for stage_index, stage in enumerate(stages):
        canonical = canonicalize_ambiguous_identities(
            raw_identity_ids=stage.raw_identity_ids,
            predicted_track_ids=stage.predicted_track_ids,
            alternatives=stage.alternatives,
            track_identity_history=track_identity_history,
        )
        for identity, track_id in zip(
            canonical, stage.predicted_track_ids, strict=True
        ):
            eligible.add(identity)
            observation_count += 1
            previous_visible = last_visible_stage.get(identity)
            is_gap = previous_visible is not None and stage_index - previous_visible > 1
            if is_gap:
                gap_count += 1
            previous_track = last_matched_track.get(identity)
            if track_id is not None:
                matched_count += 1
                identity_track_sets.setdefault(identity, set()).add(track_id)
                track_identity_sets.setdefault(track_id, set()).add(identity)
                if previous_track is not None:
                    transition_count += 1
                    if track_id != previous_track:
                        switch_count += 1
                if is_gap and previous_track is not None:
                    recovery_attempts += 1
                    if track_id == previous_track:
                        correct_recoveries += 1
                track_identity_history.setdefault(track_id, identity)
                last_matched_track[identity] = track_id
            last_visible_stage[identity] = stage_index
    fragmentations = sum(
        max(0, len(values) - 1) for values in identity_track_sets.values()
    )
    merges = sum(max(0, len(values) - 1) for values in track_identity_sets.values())
    return {
        "eligible_identity_count": len(eligible),
        "eligible_observation_count": observation_count,
        "matched_observation_count": matched_count,
        "observation_coverage": matched_count / observation_count
        if observation_count
        else 0.0,
        "matched_identity_transitions": transition_count,
        "identity_switches": switch_count,
        "normalized_id_switch_rate": switch_count / transition_count
        if transition_count
        else 0.0,
        "fragmentation_count": fragmentations,
        "merge_count": merges,
        "gap_opportunities": gap_count,
        "recovery_attempts": recovery_attempts,
        "correct_recoveries": correct_recoveries,
        "gap_recovery_accuracy": (
            correct_recoveries / recovery_attempts if recovery_attempts else 0.0
        ),
        "gap_recovery_recall": correct_recoveries / gap_count if gap_count else 0.0,
    }


def _cache_observation(entry: Mapping[str, object], *, stage: int) -> dict[str, Tensor]:
    key = entry.get("key")
    observation = entry.get("observation")
    if not isinstance(key, Mapping) or key.get("stage_index") != stage:
        raise RescanEvaluationError(
            "cache entries must preserve contiguous stage order"
        )
    if not isinstance(observation, Mapping):
        raise RescanEvaluationError("cache entry lacks an observation")
    tensors: dict[str, Tensor] = {}
    for name in (
        "features",
        "class_prob",
        "confidence",
        "valid",
        "masks",
        "mask_support",
    ):
        value = observation.get(name)
        if not isinstance(value, Tensor):
            raise RescanEvaluationError(f"cached observation lacks tensor {name}")
        tensors[name] = value.detach().cpu().clone()
    query_count = tensors["features"].shape[0]
    if (
        tensors["features"].ndim != 2
        or tensors["class_prob"].ndim != 2
        or tensors["class_prob"].shape[0] != query_count
        or tensors["confidence"].shape != (query_count,)
        or tensors["valid"].shape != (query_count,)
        or tensors["masks"].ndim != 2
        or tensors["masks"].shape[0] != query_count
        or tensors["mask_support"].shape != (query_count,)
    ):
        raise RescanEvaluationError("cached observation tensor shapes differ")
    if tensors["valid"].dtype != torch.bool or tensors["masks"].dtype != torch.bool:
        raise RescanEvaluationError("cached valid and masks tensors must be boolean")
    tensors["latest_mask"] = tensors["masks"].float()
    return tensors


def _prediction_track_ids(prediction: Mapping[str, object]) -> tuple[Hashable, ...]:
    values = prediction.get("track_ids")
    if isinstance(values, Tensor):
        result = tuple(values.detach().cpu().tolist())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = tuple(values)
    else:
        raise RescanEvaluationError("prediction lacks track identities")
    return result


def _identity_matches(
    *,
    target: RescanStageTarget,
    prediction: Mapping[str, object],
    alternatives: IdentityAlternatives,
    class_aware: bool,
) -> StageIdentityMatches:
    from scripts.p6a_metrics import match_instances_hungarian

    masks = prediction.get("pred_masks")
    classes = prediction.get("pred_classes")
    if not isinstance(masks, Tensor) or masks.ndim != 2:
        raise RescanEvaluationError("prediction masks must have shape [P, K]")
    if not isinstance(classes, Tensor) or classes.ndim != 1:
        raise RescanEvaluationError("prediction classes must have shape [K]")
    track_ids = _prediction_track_ids(prediction)
    if masks.shape[1] != len(track_ids) or classes.shape[0] != len(track_ids):
        raise RescanEvaluationError("prediction fields disagree on instance count")
    pairs = match_instances_hungarian(
        target.masks,
        masks.transpose(0, 1),
        target.class_ids if class_aware else None,
        classes if class_aware else None,
        threshold=0.5,
    )
    matched: list[Hashable | None] = [None] * target.identity_ids.numel()
    for gt_index, prediction_index in pairs:
        matched[gt_index] = track_ids[prediction_index]
    return StageIdentityMatches(
        raw_identity_ids=tuple(target.identity_ids.tolist()),
        predicted_track_ids=tuple(matched),
        alternatives=alternatives,
    )


def _temporal_target_payload(
    stage: int, target: RescanStageTarget
) -> dict[str, object]:
    from scripts.p6a_cache import CHANGE_LABEL_SEMANTICS

    return {
        "stage": stage,
        "target": {
            "gt_ids": target.identity_ids,
            "gt_classes": target.class_ids,
            "gt_masks": target.masks,
            "changes": torch.zeros_like(target.identity_ids),
            "change_labels_valid": False,
            "change_label_semantics": CHANGE_LABEL_SEMANTICS,
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }


_IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def _efficient_iou_matrix(gt_masks: Tensor, prediction_masks: Tensor) -> Tensor:
    gt = gt_masks.detach().cpu().bool()
    prediction = prediction_masks.detach().cpu().bool()
    if gt.ndim != 2 or prediction.ndim != 2 or gt.shape[1] != prediction.shape[1]:
        raise RescanEvaluationError("IoU masks must have shape [N, P] with shared P")
    if gt.shape[0] == 0 or prediction.shape[0] == 0:
        return torch.zeros((gt.shape[0], prediction.shape[0]), dtype=torch.float32)
    gt_float = gt.float()
    prediction_float = prediction.float()
    intersections = gt_float @ prediction_float.transpose(0, 1)
    unions = (
        gt_float.sum(dim=1, keepdim=True)
        + prediction_float.sum(dim=1).unsqueeze(0)
        - intersections
    )
    return torch.where(unions > 0, intersections / unions, torch.zeros_like(unions))


def _scores_from_ious(
    *,
    ious: Tensor,
    gt_classes: Tensor,
    prediction_classes: Tensor,
    prediction_scores: Tensor,
    threshold: float,
) -> tuple[list[bool], list[float], int]:
    matched: set[int] = set()
    true_positives = []
    scores = []
    for prediction_index in sorted(
        range(prediction_scores.numel()),
        key=lambda index: (-float(prediction_scores[index]), index),
    ):
        compatible = [
            gt_index
            for gt_index in range(gt_classes.numel())
            if gt_index not in matched
            and bool(gt_classes[gt_index] == prediction_classes[prediction_index])
            and float(ious[gt_index, prediction_index]) >= threshold
        ]
        if compatible:
            selected = max(
                compatible,
                key=lambda index: (float(ious[index, prediction_index]), -index),
            )
            matched.add(selected)
            true_positives.append(True)
        else:
            true_positives.append(False)
        scores.append(float(prediction_scores[prediction_index]))
    return true_positives, scores, len(matched)


def _efficient_raw_local_metrics(
    predictions: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, Tensor]],
) -> dict[str, float]:
    from scripts.p6a_metrics import _average_precision

    if len(predictions) != len(targets):
        raise RescanEvaluationError("raw predictions and targets must align")
    per_frame_ap = []
    evaluated_thresholds = (0.25, *_IOU_THRESHOLDS)
    threshold_ap: dict[float, list[float]] = {
        threshold: [] for threshold in evaluated_thresholds
    }
    threshold_matches: dict[float, list[tuple[int, int]]] = {
        threshold: [] for threshold in evaluated_thresholds
    }
    for prediction, target in zip(predictions, targets, strict=True):
        masks = prediction["pred_masks"]
        classes = prediction["pred_classes"]
        scores = prediction["pred_scores"]
        if not all(isinstance(value, Tensor) for value in (masks, classes, scores)):
            raise RescanEvaluationError("raw prediction tensors are invalid")
        ious = _efficient_iou_matrix(target["masks"], masks.transpose(0, 1))
        frame_values = []
        for threshold in evaluated_thresholds:
            positives, ordered_scores, matched = _scores_from_ious(
                ious=ious,
                gt_classes=target["labels"],
                prediction_classes=classes,
                prediction_scores=scores,
                threshold=threshold,
            )
            gt_count = int(target["labels"].numel())
            ap = _average_precision(positives, ordered_scores, gt_count)
            if ap is not None:
                threshold_ap[threshold].append(ap)
                if threshold in _IOU_THRESHOLDS:
                    frame_values.append(ap)
            threshold_matches[threshold].append((matched, gt_count))
        if frame_values:
            per_frame_ap.append(float(np.mean(frame_values)))

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def recall(threshold: float) -> float:
        values = threshold_matches[threshold]
        denominator = sum(total for _, total in values)
        return sum(found for found, _ in values) / denominator if denominator else 0.0

    result = {
        "AP": mean(per_frame_ap),
        "AP50": mean(threshold_ap[0.5]),
        "AP25": mean(threshold_ap[0.25]),
        "REC": mean([recall(threshold) for threshold in _IOU_THRESHOLDS]),
        "REC50": recall(0.5),
        "REC25": recall(0.25),
    }
    result.update({f"raw_local_{key}": value for key, value in result.items()})
    return result


def _efficient_endpoint_metrics(
    accumulator: object,
    temporal_target: Mapping[str, Tensor],
    *,
    endpoint: int,
) -> dict[str, float]:
    from scripts.p6a_metrics import _average_precision, build_online_endpoint_prediction

    prediction = build_online_endpoint_prediction(accumulator, endpoint=endpoint)
    masks = prediction["pred_masks"]
    classes = prediction["pred_classes"]
    scores = prediction["pred_scores"]
    if not all(isinstance(value, Tensor) for value in (masks, classes, scores)):
        raise RescanEvaluationError("endpoint prediction tensors are invalid")
    stages = temporal_target["temporal_stages"]
    stage_ious = []
    for stage in range(endpoint + 1):
        selector = stages == stage
        stage_ious.append(
            _efficient_iou_matrix(
                temporal_target["masks"][:, selector],
                masks[selector].transpose(0, 1),
            )
        )
    temporal_ious = torch.stack(stage_ious).amin(dim=0)
    ap_values: dict[float, float] = {}
    recall_values: dict[float, float] = {}
    gt_count = int(temporal_target["labels"].numel())
    for threshold in (0.25, *_IOU_THRESHOLDS):
        positives, ordered_scores, matched = _scores_from_ious(
            ious=temporal_ious,
            gt_classes=temporal_target["labels"],
            prediction_classes=classes,
            prediction_scores=scores,
            threshold=threshold,
        )
        ap_values[threshold] = (
            _average_precision(positives, ordered_scores, gt_count) or 0.0
        )
        recall_values[threshold] = matched / gt_count if gt_count else 0.0
    metrics = {
        "t-mAP": float(np.mean([ap_values[value] for value in _IOU_THRESHOLDS])),
        "t-mAP50": ap_values[0.5],
        "t-mAP25": ap_values[0.25],
        "t-REC": float(np.mean([recall_values[value] for value in _IOU_THRESHOLDS])),
        "t-REC50": recall_values[0.5],
        "t-REC25": recall_values[0.25],
    }
    result = {f"online_{key}": value for key, value in metrics.items()}
    result.update(
        {f"offline_reconstructed_{key}": value for key, value in metrics.items()}
    )
    result.update(
        {key.replace("t-", "t_"): value for key, value in result.items() if "t-" in key}
    )
    return result


def _trackers(scene_id: str) -> dict[str, object]:
    from scripts.p6a_association import (
        B1FeatureTracker,
        B2FeatureClassTracker,
        B3EmaTracker,
        B4PersistentTracker,
    )

    shared = {
        "sequence_id": scene_id,
        "feature_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
    }
    return {
        "B1": B1FeatureTracker(**shared),
        "B2": B2FeatureClassTracker(**shared),
        "B3": B3EmaTracker(update_rate=0.2, **shared),
        "B4": B4PersistentTracker(
            sequence_id=scene_id,
            capacity=100,
            class_weight=0.25,
            association_threshold=0.5,
            update_rate=0.2,
            max_update_rate=0.2,
        ),
    }


def evaluate_rescan_sequence(
    *,
    scene_id: str,
    cache_entries: Sequence[Mapping[str, object]],
    evaluator_targets: Sequence[RescanEvaluatorTarget],
    label_map: Mapping[str, object],
    level_a_excluded_identity_ids: Sequence[int] = (),
) -> dict[str, object]:
    """Evaluate one official-order scene from a shared frozen observation cache."""

    from scripts.evaluate_persist4d_p6a import (
        build_temporal_target,
        stage_prediction_from_track_step,
    )
    from scripts.p6a_association import freeze_observation
    from scripts.p6a_metrics import (
        IdentityAccumulator,
    )

    if (
        not scene_id
        or len(cache_entries) != len(evaluator_targets)
        or not cache_entries
    ):
        raise RescanEvaluationError("scene cache and evaluator targets must align")
    trackers = _trackers(scene_id)
    accumulators = {method: IdentityAccumulator() for method in trackers}
    predictions: dict[str, list[Mapping[str, object]]] = {
        method: [] for method in trackers
    }
    identity_stages: dict[str, dict[str, list[StageIdentityMatches]]] = {
        method: {"A": [], "B": []} for method in trackers
    }
    raw_targets: list[dict[str, Tensor]] = []
    temporal_payloads: list[dict[str, object]] = []
    level_b_eligible: set[int] = set()
    for stage, (entry, evaluator_target) in enumerate(
        zip(cache_entries, evaluator_targets, strict=True)
    ):
        if evaluator_target.scene_id != scene_id:
            raise RescanEvaluationError("evaluator target changed scene identity")
        key = entry.get("key")
        if (
            not isinstance(key, Mapping)
            or key.get("target_capture_id") != evaluator_target.capture_id
        ):
            raise RescanEvaluationError("cache capture and evaluator target differ")
        observation = _cache_observation(entry, stage=stage)
        level_a = build_rescan_stage_target(
            evaluator_target,
            label_map,
            level="A",
            excluded_identity_ids=level_a_excluded_identity_ids,
        )
        level_b = build_rescan_stage_target(evaluator_target, label_map, level="B")
        level_b_eligible.update(int(value) for value in level_b.identity_ids.tolist())
        raw_targets.append({"masks": level_a.masks, "labels": level_a.class_ids})
        temporal_payloads.append(_temporal_target_payload(stage, level_a))
        frozen = freeze_observation(observation)
        stage_payload = {"stage": stage, "observation": observation}
        for method, tracker in trackers.items():
            step = tracker.step(frozen, stage_id=stage)
            prediction = stage_prediction_from_track_step(
                stage_payload,
                step,
                background_class=18,
            )
            predictions[method].append(prediction)
            accumulators[method].add_stage(prediction)
            identity_stages[method]["A"].append(
                _identity_matches(
                    target=level_a,
                    prediction=prediction,
                    alternatives=evaluator_target.ambiguities,
                    class_aware=True,
                )
            )
            identity_stages[method]["B"].append(
                _identity_matches(
                    target=level_b,
                    prediction=prediction,
                    alternatives=evaluator_target.ambiguities,
                    class_aware=False,
                )
            )
    temporal_target = build_temporal_target(temporal_payloads)
    methods: dict[str, object] = {}
    for method in trackers:
        level_a_metrics = _efficient_raw_local_metrics(predictions[method], raw_targets)
        level_a_metrics.update(
            _efficient_endpoint_metrics(
                accumulators[method],
                temporal_target,
                endpoint=len(cache_entries) - 1,
            )
        )
        level_a_metrics.update(
            {
                f"identity_{key}": value
                for key, value in evaluate_identity_sequence(
                    identity_stages[method]["A"]
                ).items()
            }
        )
        methods[method] = {
            "method_name": METHOD_NAMES[method],
            "level_a": level_a_metrics,
            "level_b": evaluate_identity_sequence(identity_stages[method]["B"]),
        }
    return {
        "scene_id": scene_id,
        "capture_count": len(cache_entries),
        "level_b_eligible_identity_count": len(level_b_eligible),
        "methods": methods,
    }


def _numeric_metrics(value: Mapping[str, object]) -> dict[str, float]:
    return {
        key: float(item)
        for key, item in value.items()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    }


def aggregate_rescan_results(
    scenes: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int = 10000,
    seed: int = 45,
) -> dict[str, object]:
    if not scenes or bootstrap_replicates <= 0:
        raise RescanEvaluationError(
            "scene results and bootstrap replicates are required"
        )
    scene_ids = [scene.get("scene_id") for scene in scenes]
    if any(not isinstance(value, str) or not value for value in scene_ids) or len(
        set(scene_ids)
    ) != len(scene_ids):
        raise RescanEvaluationError("scene results require unique scene ids")
    method_aggregate: dict[str, object] = {}
    for method, method_name in METHOD_NAMES.items():
        levels: dict[str, object] = {}
        for level_key in ("level_a", "level_b"):
            per_scene = []
            for scene in scenes:
                methods = scene.get("methods")
                if not isinstance(methods, Mapping) or not isinstance(
                    methods.get(method), Mapping
                ):
                    raise RescanEvaluationError("every scene must contain all methods")
                metrics = methods[method].get(level_key)
                if not isinstance(metrics, Mapping):
                    raise RescanEvaluationError(
                        "method result lacks an evaluation level"
                    )
                per_scene.append(_numeric_metrics(metrics))
            metric_names = sorted(set.intersection(*(set(row) for row in per_scene)))
            levels[level_key] = {
                name: float(np.mean([row[name] for row in per_scene]))
                for name in metric_names
            }
        method_aggregate[method] = {
            "method_name": method_name,
            **levels,
        }
    comparison_metrics = (
        ("level_a", "online_t_mAP", True),
        ("level_a", "online_t_REC", True),
        ("level_b", "observation_coverage", True),
        ("level_b", "gap_recovery_accuracy", True),
        ("level_b", "gap_recovery_recall", True),
        ("level_b", "normalized_id_switch_rate", False),
        ("level_b", "fragmentation_count", False),
        ("level_b", "merge_count", False),
    )
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, len(scenes), size=(bootstrap_replicates, len(scenes))
    )
    bootstrap_rows = []
    for level, metric, higher_is_better in comparison_metrics:
        if any(
            metric not in scene["methods"][method][level]
            for scene in scenes
            for method in ("B2", "B4")
        ):
            continue
        comparator = np.asarray(
            [float(scene["methods"]["B2"][level][metric]) for scene in scenes],
            dtype=np.float64,
        )
        persist4d = np.asarray(
            [float(scene["methods"]["B4"][level][metric]) for scene in scenes],
            dtype=np.float64,
        )
        effects = persist4d - comparator
        replicates = effects[sample_indices].mean(axis=1)
        comparator_mean = float(comparator.mean())
        mean_effect = float(effects.mean())
        bootstrap_rows.append(
            {
                "level": level,
                "metric": metric,
                "higher_is_better": higher_is_better,
                "comparator": "B2",
                "comparator_mean": comparator_mean,
                "persist4d_mean": float(persist4d.mean()),
                "mean_effect": mean_effect,
                "mean_absolute_effect": float(np.abs(effects).mean()),
                "relative_effect": (
                    mean_effect / abs(comparator_mean)
                    if comparator_mean != 0.0
                    else None
                ),
                "ci95_low": float(np.quantile(replicates, 0.025)),
                "ci95_high": float(np.quantile(replicates, 0.975)),
            }
        )
    b4_gaps = [
        int(scene["methods"]["B4"]["level_b"]["gap_opportunities"]) for scene in scenes
    ]
    return {
        "schema_version": 1,
        "population": {
            "scene_count": len(scenes),
            "gap_opportunity_count": sum(b4_gaps),
            "gap_scene_cluster_count": sum(value > 0 for value in b4_gaps),
        },
        "method_aggregate": method_aggregate,
        "bootstrap": bootstrap_rows,
        "scenes": list(scenes),
    }


def derive_external_gate(
    aggregate: Mapping[str, object],
    *,
    minimum_gap_opportunities: int = 10,
    minimum_gap_scene_clusters: int = 3,
    minimum_observation_coverage: float = 0.1,
) -> dict[str, object]:
    population = aggregate.get("population")
    methods = aggregate.get("method_aggregate")
    if not isinstance(population, Mapping) or not isinstance(methods, Mapping):
        raise RescanEvaluationError("aggregate lacks gate inputs")
    gaps = int(population.get("gap_opportunity_count", 0))
    gap_scenes = int(population.get("gap_scene_cluster_count", 0))
    try:
        coverage = float(methods["B4"]["level_b"]["observation_coverage"])
    except (KeyError, TypeError, ValueError) as error:
        raise RescanEvaluationError("aggregate lacks Persist4D coverage") from error
    failures = []
    if gaps < minimum_gap_opportunities:
        failures.append("insufficient_gap_opportunities")
    if gap_scenes < minimum_gap_scene_clusters:
        failures.append("insufficient_gap_scene_clusters")
    if coverage < minimum_observation_coverage:
        failures.append("insufficient_observation_coverage")
    classification = "EXTERNAL_INCONCLUSIVE"
    if not failures:
        b2 = methods["B2"]
        b4 = methods["B4"]
        gap_effect = float(b4["level_b"]["gap_recovery_recall"]) - float(
            b2["level_b"]["gap_recovery_recall"]
        )
        idsw_effect = float(b4["level_b"]["normalized_id_switch_rate"]) - float(
            b2["level_b"]["normalized_id_switch_rate"]
        )
        task_effect = float(b4["level_a"]["online_t_mAP"]) - float(
            b2["level_a"]["online_t_mAP"]
        )
        if gap_effect > 0 and idsw_effect <= 0 and task_effect >= -0.1:
            classification = "EXTERNAL_SUPPORT"
        elif gap_effect > 0 and idsw_effect <= 0:
            classification = "EXTERNAL_PARTIAL"
        else:
            classification = "EXTERNAL_CONTRADICTS"
    return {
        "schema_version": 1,
        "classification": classification,
        "observed_gap_opportunities": gaps,
        "observed_gap_scene_clusters": gap_scenes,
        "persist4d_observation_coverage": coverage,
        "thresholds": {
            "minimum_gap_opportunities": minimum_gap_opportunities,
            "minimum_gap_scene_clusters": minimum_gap_scene_clusters,
            "minimum_observation_coverage": minimum_observation_coverage,
        },
        "failed_thresholds": failures,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _json_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _cache_provenance(
    *,
    dataset_manifest: Mapping[str, object],
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint_digest = _sha256_file(checkpoint_path)
    if checkpoint_digest != CHECKPOINT_SHA256:
        raise RescanEvaluationError("frozen ReScene checkpoint SHA256 differs")
    dataset_digest = dataset_manifest.get("dataset_content_sha256")
    if not isinstance(dataset_digest, str) or len(dataset_digest) != 64:
        raise RescanEvaluationError("dataset manifest lacks a content digest")
    return {
        "source_commit": _git_commit(),
        "checkpoint_sha256": checkpoint_digest,
        "external_config_sha256": _sha256_file(config_path),
        "dataset_content_sha256": dataset_digest,
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "geometry_segment_size_m": 0.1,
        "model_voxel_size_m": 0.02,
        "seed": 45,
    }


def _tensor_cpu(value: Tensor) -> Tensor:
    return value.detach().cpu().contiguous().clone()


def _full_resolution_query_masks(
    *,
    system: Any,
    output: Mapping[str, object],
    data: Any,
    target: Mapping[str, Tensor],
    full_point2segment: Tensor,
    full_temporal_stages: Tensor,
    latest_stage: int,
) -> Tensor:
    raw_masks = output.get("pred_masks")
    if (
        isinstance(raw_masks, (str, bytes))
        or not isinstance(raw_masks, Sequence)
        or len(raw_masks) != 1
        or not isinstance(raw_masks[0], Tensor)
    ):
        raise RescanEvaluationError("model output lacks one segment-mask tensor")
    segment_logits = raw_masks[0]
    point2segment = target["point2segment"]
    if segment_logits.ndim != 2 or int(point2segment.max()) >= segment_logits.shape[0]:
        raise RescanEvaluationError("model masks do not cover low-resolution segments")
    low_resolution = (segment_logits > 0).float()[point2segment]
    full_masks = system._get_full_res_mask(
        low_resolution,
        data.inverse_maps[0],
        full_point2segment,
    ).bool()
    selector = full_temporal_stages == latest_stage
    if not torch.any(selector).item():
        raise RescanEvaluationError("full-resolution geometry lacks the latest stage")
    return _tensor_cpu(full_masks[selector].transpose(0, 1))


def _cache_entry_path(cache_root: Path, scene_id: str, stage_index: int) -> Path:
    return cache_root / "entries" / f"{scene_id}_{stage_index}.pt"


def _save_cache_entry(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RescanEvaluationError(f"cache entry already exists: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _same_provenance(left: object, right: Mapping[str, object]) -> bool:
    return isinstance(left, Mapping) and dict(left) == dict(right)


def run_rescan_inference(
    *,
    dataset_root: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    rio_metric_spec_path: Path,
    cache_root: Path,
    device_name: str,
    scene_limit: int | None = None,
) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import hydra

    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _compose_runtime_config,
        _load_system,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )

    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    provenance = _cache_provenance(
        dataset_manifest=dataset_manifest,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    dataset = RescanTemporalDataset(dataset_root, geometry_segment_size_m=0.1)
    selected_scene_count = len(dataset)
    if scene_limit is not None:
        if scene_limit <= 0:
            raise RescanEvaluationError("scene_limit must be positive")
        selected_scene_count = min(scene_limit, selected_scene_count)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RescanEvaluationError("formal external inference requires CUDA")
    config, memory = _compose_runtime_config()
    config.instance_metric.dataset = str(rio_metric_spec_path)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    system = _load_system(config, checkpoint_path, device)
    system.eval()

    previous_settings = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    entries = []
    try:
        random.seed(45)
        np.random.seed(45)
        torch.manual_seed(45)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for scene_index in range(selected_scene_count):
            scene_indices = dataset.sequence_indices[scene_index]
            for stage_index, target_scan_index in enumerate(scene_indices):
                entry_path = _cache_entry_path(
                    cache_root, dataset.sequence_names[scene_index], stage_index
                )
                if entry_path.is_file():
                    cached = torch.load(
                        entry_path, map_location="cpu", weights_only=True
                    )
                    if not _same_provenance(cached.get("provenance"), provenance):
                        raise RescanEvaluationError(
                            f"cache provenance differs: {entry_path.name}"
                        )
                else:
                    local_indices = (
                        (target_scan_index,)
                        if stage_index == 0
                        else (scene_indices[stage_index - 1], target_scan_index)
                    )
                    sample = dataset.load_scan_indices(
                        scene_index, local_indices, change_file=None
                    )
                    data, targets, names = collate([sample])
                    if list(names) != [dataset.sequence_names[scene_index]]:
                        raise RescanEvaluationError(
                            "collator changed the scene identity"
                        )
                    prepared = prepare_rescan_model_batch(data, targets)
                    safe_data_keys = sorted(str(key) for key in prepared.data)
                    data = _move_data_to_device(prepared.data, device)
                    target = _move_targets_to_device([prepared.target], device)[0]
                    stages = _segment_stages(target)
                    latest_stage = int(stages.max().item())
                    raw_coordinates = system._process_raw_coordinates(data)
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output = system(
                            data,
                            point2segment=[target["point2segment"]],
                            raw_coordinates=raw_coordinates,
                            is_eval=True,
                        )
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    observation = build_local_observation(
                        output,
                        [stages],
                        latest_stage=latest_stage,
                        background_class=int(memory.background_class),
                        confidence_threshold=float(memory.confidence_threshold),
                        mask_threshold=float(memory.mask_threshold),
                        minimum_mask_support=int(memory.minimum_mask_support),
                    )
                    masks = _full_resolution_query_masks(
                        system=system,
                        output=output,
                        data=data,
                        target=target,
                        full_point2segment=prepared.full_point2segment,
                        full_temporal_stages=prepared.full_temporal_stages,
                        latest_stage=latest_stage,
                    )
                    support = masks.sum(dim=1, dtype=torch.long)
                    cached = {
                        "schema_version": 1,
                        "provenance": provenance,
                        "key": {
                            "scene_id": dataset.sequence_names[scene_index],
                            "scene_index": scene_index,
                            "stage_index": stage_index,
                            "target_capture_id": dataset.captures[
                                target_scan_index
                            ].capture_id,
                            "local_capture_ids": [
                                dataset.captures[index].capture_id
                                for index in local_indices
                            ],
                            "target_scan_index": target_scan_index,
                        },
                        "observation": {
                            "features": _tensor_cpu(observation.features[0]),
                            "class_prob": _tensor_cpu(observation.class_prob[0]),
                            "confidence": _tensor_cpu(observation.confidence[0]),
                            "valid": _tensor_cpu(observation.valid[0]),
                            "masks": masks,
                            "mask_support": support,
                        },
                        "runtime": {
                            "forward_seconds": elapsed,
                            "peak_gpu_memory_mib": (
                                torch.cuda.max_memory_allocated(device) / 1024**2
                            ),
                            "raw_target_point_count": int(masks.shape[1]),
                            "low_resolution_point_count": int(
                                target["point2segment"].shape[0]
                            ),
                            "geometry_segment_count": int(
                                target["point2segment"].max().item()
                            )
                            + 1,
                            "valid_observation_count": int(
                                observation.valid.sum().item()
                            ),
                        },
                        "no_gt_leakage": {
                            "status": "pass",
                            "model_data_keys": safe_data_keys,
                            "model_target_keys": sorted(prepared.target),
                            "evaluator_target_loaded_during_inference": False,
                        },
                    }
                    _save_cache_entry(entry_path, cached)
                    del data, target, output, observation, masks
                    torch.cuda.empty_cache()
                entries.append(
                    {
                        "filename": entry_path.name,
                        "bytes": entry_path.stat().st_size,
                        "sha256": _sha256_file(entry_path),
                        "key": cached["key"],
                    }
                )
    finally:
        torch.use_deterministic_algorithms(
            previous_settings["deterministic"],
            warn_only=previous_settings["warn_only"],
        )
        torch.backends.cudnn.benchmark = previous_settings["benchmark"]
        torch.backends.cudnn.deterministic = previous_settings["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = previous_settings["cuda_tf32"]
        torch.backends.cudnn.allow_tf32 = previous_settings["cudnn_tf32"]
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "provenance": provenance,
        "scene_count": selected_scene_count,
        "entry_count": len(entries),
        "entries": entries,
    }
    _atomic_json(cache_root / "manifest.json", manifest)
    return manifest


def _load_json_mapping(path: Path, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RescanEvaluationError(f"cannot load {name}: {path}") from error
    if not isinstance(value, Mapping):
        raise RescanEvaluationError(f"{name} must be a JSON object")
    return dict(value)


def evaluate_rescan_cache(
    *,
    dataset_root: Path,
    dataset_manifest_path: Path,
    label_map_path: Path,
    cache_root: Path,
) -> dict[str, object]:
    dataset_manifest = _load_json_mapping(
        dataset_manifest_path, name="dataset manifest"
    )
    label_map = _load_json_mapping(label_map_path, name="label map")
    cache_manifest = _load_json_mapping(
        cache_root / "manifest.json", name="cache manifest"
    )
    if cache_manifest.get("status") != "pass":
        raise RescanEvaluationError("cache manifest has not passed")
    dataset = RescanTemporalDataset(dataset_root, geometry_segment_size_m=0.1)
    scene_count = cache_manifest.get("scene_count")
    if isinstance(scene_count, bool) or not isinstance(scene_count, int):
        raise RescanEvaluationError("cache manifest lacks scene_count")
    if scene_count != len(dataset):
        raise RescanEvaluationError("formal evaluation requires all official scenes")
    cache_records = cache_manifest.get("entries")
    if not isinstance(cache_records, Sequence) or isinstance(
        cache_records, (str, bytes)
    ):
        raise RescanEvaluationError("cache manifest lacks entry records")
    expected_cache_files = {
        record.get("filename"): record
        for record in cache_records
        if isinstance(record, Mapping) and isinstance(record.get("filename"), str)
    }
    manifest_scenes = dataset_manifest.get("scenes")
    if not isinstance(manifest_scenes, Sequence) or isinstance(
        manifest_scenes, (str, bytes)
    ):
        raise RescanEvaluationError("dataset manifest lacks scenes")
    scene_metadata = {
        scene.get("scene_id"): scene
        for scene in manifest_scenes
        if isinstance(scene, Mapping) and isinstance(scene.get("scene_id"), str)
    }
    if set(scene_metadata) != set(dataset.sequence_names):
        raise RescanEvaluationError("dataset manifest scene identities differ")
    scenes = []
    for scene_index, scene_id in enumerate(dataset.sequence_names):
        indices = dataset.sequence_indices[scene_index]
        entries = []
        for stage_index in range(len(indices)):
            path = _cache_entry_path(cache_root, scene_id, stage_index)
            if not path.is_file():
                raise RescanEvaluationError(f"missing cache entry: {path.name}")
            record = expected_cache_files.get(path.name)
            if not isinstance(record, Mapping) or record.get("sha256") != _sha256_file(
                path
            ):
                raise RescanEvaluationError(f"cache entry hash differs: {path.name}")
            entry = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(entry, Mapping):
                raise RescanEvaluationError("cache entry must contain a mapping")
            entries.append(entry)
        targets = dataset.evaluator_targets(indices)
        metadata = scene_metadata[scene_id]
        excluded = metadata.get("semantic_inconsistent_identity_ids", [])
        if not isinstance(excluded, Sequence) or isinstance(excluded, (str, bytes)):
            raise RescanEvaluationError("semantic inconsistency list is invalid")
        scenes.append(
            evaluate_rescan_sequence(
                scene_id=scene_id,
                cache_entries=entries,
                evaluator_targets=targets,
                label_map=label_map,
                level_a_excluded_identity_ids=[int(value) for value in excluded],
            )
        )
    aggregate = aggregate_rescan_results(scenes, bootstrap_replicates=10000, seed=45)
    gate = derive_external_gate(aggregate)
    return {
        "schema_version": 1,
        "status": "pass",
        "provenance": {
            "source_commit": _git_commit(),
            "dataset_content_sha256": dataset_manifest.get("dataset_content_sha256"),
            "label_map_sha256": _sha256_file(label_map_path),
            "cache_manifest_sha256": _sha256_file(cache_root / "manifest.json"),
            "evaluator_sha256": _sha256_file(Path(__file__)),
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "aggregate": aggregate,
        "external_gate": gate,
    }


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise RescanEvaluationError(f"CSV output has no rows: {path.name}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _flatten_method_row(
    *, scene_id: str | None, method: str, value: Mapping[str, object]
) -> dict[str, object]:
    row: dict[str, object] = {
        "scene_id": scene_id if scene_id is not None else "ALL_SCENES_MEAN",
        "method_code": method,
        "method": value.get("method_name", METHOD_NAMES[method]),
    }
    for level in ("level_a", "level_b"):
        metrics = value.get(level)
        if not isinstance(metrics, Mapping):
            raise RescanEvaluationError("method result lacks level metrics")
        for key, metric in metrics.items():
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                row[f"{level}_{key}"] = metric
    return row


def write_rescan_evaluation_artifacts(
    *, output_dir: Path, result: Mapping[str, object]
) -> None:
    aggregate = result.get("aggregate")
    gate = result.get("external_gate")
    if not isinstance(aggregate, Mapping) or not isinstance(gate, Mapping):
        raise RescanEvaluationError("evaluation result lacks aggregate or gate")
    scenes = aggregate.get("scenes")
    methods = aggregate.get("method_aggregate")
    bootstrap = aggregate.get("bootstrap")
    if (
        not isinstance(scenes, Sequence)
        or isinstance(scenes, (str, bytes))
        or not isinstance(methods, Mapping)
        or not isinstance(bootstrap, Sequence)
        or isinstance(bootstrap, (str, bytes))
    ):
        raise RescanEvaluationError("aggregate artifact fields are invalid")
    external = output_dir / "external"
    _atomic_json(external / "rescan_raw.json", result)
    per_scene_rows = [
        _flatten_method_row(
            scene_id=str(scene["scene_id"]),
            method=method,
            value=scene["methods"][method],
        )
        for scene in scenes
        for method in METHOD_NAMES
    ]
    aggregate_rows = [
        _flatten_method_row(scene_id=None, method=method, value=methods[method])
        for method in METHOD_NAMES
    ]
    _atomic_csv(external / "rescan_per_scene.csv", per_scene_rows)
    _atomic_csv(external / "rescan_results.csv", aggregate_rows)
    _atomic_csv(external / "rescan_scene_bootstrap.csv", list(bootstrap))
    _atomic_json(output_dir / "external_gate.json", gate)
    population = aggregate["population"]
    report = (
        "# Independent ReScan External Validation\n\n"
        f"- Classification: `{gate['classification']}`\n"
        f"- Independent scenes: {population['scene_count']}\n"
        f"- Natural gap opportunities: {population['gap_opportunity_count']}\n"
        f"- Gap-bearing scenes: {population['gap_scene_cluster_count']}\n"
        f"- Persist4D observation coverage: {gate['persist4d_observation_coverage']:.6f}\n"
        f"- Failed preregistered thresholds: {', '.join(gate['failed_thresholds']) or 'none'}\n\n"
        "All four methods consume the same frozen local-observation cache. "
        "Identity metrics use official ambiguity alternatives and scene-clustered statistics. "
        "The gate classification is determined before interpreting method effects.\n"
    )
    path = output_dir / "EXTERNAL_VALIDATION_REPORT.md"
    path.write_text(report, encoding="utf-8")


def _smoke_summary(
    manifest: Mapping[str, object], *, cache_root: Path
) -> dict[str, object]:
    entries = manifest.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise RescanEvaluationError("inference manifest lacks entries")
    runtimes = []
    no_gt_checks = []
    for record in entries:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("filename"), str
        ):
            raise RescanEvaluationError("smoke cache record is invalid")
        entry = torch.load(
            cache_root / "entries" / record["filename"],
            map_location="cpu",
            weights_only=True,
        )
        runtime = entry.get("runtime")
        leakage = entry.get("no_gt_leakage")
        if not isinstance(runtime, Mapping) or not isinstance(leakage, Mapping):
            raise RescanEvaluationError("smoke cache entry lacks audit records")
        runtimes.append(runtime)
        no_gt_checks.append(
            leakage.get("status") == "pass"
            and leakage.get("evaluator_target_loaded_during_inference") is False
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "scene_count": manifest.get("scene_count"),
        "entry_count": manifest.get("entry_count"),
        "provenance": manifest.get("provenance"),
        "checks": {
            "cache_entries_hashed": all(
                isinstance(entry, Mapping)
                and isinstance(entry.get("sha256"), str)
                and len(entry["sha256"]) == 64
                for entry in entries
            ),
            "no_gt_leakage_contract": all(no_gt_checks),
        },
        "runtime": {
            "total_forward_seconds": float(
                sum(float(value["forward_seconds"]) for value in runtimes)
            ),
            "mean_forward_seconds": float(
                np.mean([float(value["forward_seconds"]) for value in runtimes])
            ),
            "maximum_peak_gpu_memory_mib": float(
                max(float(value["peak_gpu_memory_mib"]) for value in runtimes)
            ),
            "total_valid_observations": int(
                sum(int(value["valid_observation_count"]) for value in runtimes)
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("infer", "evaluate", "all"), default="all")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/shared/ww/persist4d-final-evidence/rescan/dataset"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/final_evidence/external/rescan_dataset_manifest.json",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/final_evidence/rescan_to_rescene_label_map.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/final_evidence/rescan.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument(
        "--rio-metric-spec",
        type=Path,
        default=Path("/home/ww/paper5/data/processed/rio/rio.yaml"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/shared/ww/persist4d-final-evidence/rescan/persist4d-cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/final_evidence",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-limit", type=int)
    parser.add_argument("--smoke-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.mode in {"infer", "all"}:
        manifest = run_rescan_inference(
            dataset_root=arguments.dataset_root,
            dataset_manifest_path=arguments.dataset_manifest,
            config_path=arguments.config,
            checkpoint_path=arguments.checkpoint,
            rio_metric_spec_path=arguments.rio_metric_spec,
            cache_root=arguments.cache_root,
            device_name=arguments.device,
            scene_limit=arguments.scene_limit,
        )
        if arguments.smoke_output is not None:
            _atomic_json(
                arguments.smoke_output,
                _smoke_summary(manifest, cache_root=arguments.cache_root),
            )
    if arguments.mode in {"evaluate", "all"}:
        if arguments.scene_limit is not None:
            raise RescanEvaluationError("formal evaluation does not accept scene_limit")
        result = evaluate_rescan_cache(
            dataset_root=arguments.dataset_root,
            dataset_manifest_path=arguments.dataset_manifest,
            label_map_path=arguments.label_map,
            cache_root=arguments.cache_root,
        )
        write_rescan_evaluation_artifacts(
            output_dir=arguments.output_dir, result=result
        )
        print(json.dumps(result["external_gate"], allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "PreparedRescanBatch",
    "RescanEvaluationError",
    "RescanStageTarget",
    "StageIdentityMatches",
    "aggregate_rescan_results",
    "build_rescan_stage_target",
    "canonicalize_ambiguous_identities",
    "derive_external_gate",
    "evaluate_identity_sequence",
    "evaluate_rescan_cache",
    "evaluate_rescan_sequence",
    "prepare_rescan_model_batch",
    "run_rescan_inference",
    "write_rescan_evaluation_artifacts",
]


if __name__ == "__main__":
    raise SystemExit(main())
