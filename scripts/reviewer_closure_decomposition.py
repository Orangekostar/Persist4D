"""Reviewer-closure performance-decomposition primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.p6a_association import OracleStageTarget, run_oracle_posthoc
from scripts.p6a_metrics import IdentityAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_SPEC = PROJECT_ROOT / "data/processed/rio/rio.yaml"
COVERAGE_CATEGORIES = (
    "no_candidate_observation",
    "wrong_class",
    "insufficient_iou",
    "associable",
)
FAILURE_CATEGORIES = (
    "local_observation_miss",
    "class_failure",
    "high_iou_mask_failure",
    "identity_fragmentation",
    "identity_merge",
    "wrong_gap_recovery",
    "capacity_failure",
    "unknown_unresolved",
)


def _cpu_tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    tensor = value.detach().cpu().contiguous().clone()
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must contain finite values")
    return tensor


class OfficialTemporalCurveAccumulator:
    """Pooled class-macro temporal AP curve via the public stmetrics matcher/head."""

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        dataset_spec: str | Path = DEFAULT_DATASET_SPEC,
        min_region_size: int = 100,
    ) -> None:
        values = tuple(float(value) for value in thresholds)
        if (
            not values
            or len(set(values)) != len(values)
            or any(not math.isfinite(value) or not 0 < value <= 1 for value in values)
        ):
            raise ValueError("thresholds must be unique, finite, and within (0, 1]")
        if (
            isinstance(min_region_size, bool)
            or not isinstance(min_region_size, int)
            or min_region_size <= 0
        ):
            raise ValueError("min_region_size must be a positive integer")
        specification = Path(dataset_spec)
        if not specification.is_file():
            raise ValueError("dataset_spec must be an existing YAML file")

        from stmetrics import InstanceMatcher, TemporalEvaluator

        self.thresholds = values
        self._updates = 0
        self._matcher = InstanceMatcher(
            dataset=str(specification),
            min_region_size=min_region_size,
            timestep_key="temporal_stages",
        )
        self._matcher.config.overlaps = torch.tensor(values, dtype=torch.float32)
        self._head = TemporalEvaluator(recall=True, aux="changes")
        self._head.bind(
            config=self._matcher.config,
            spec=self._matcher.spec,
            log_prefix="val",
        )

    def update(
        self,
        prediction: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> None:
        self._head.update(self._matcher.match_batch([prediction], [target]))
        self._updates += 1

    def compute(self) -> dict[float, float]:
        if self._updates == 0:
            raise ValueError("curve accumulator has no observations")
        self._head.compute()
        aps = self._head._last_aps
        if not isinstance(aps, Tensor) or aps.shape[1] != len(self.thresholds):
            raise ValueError("stmetrics temporal AP state has unexpected shape")
        result = {}
        for index, threshold in enumerate(self.thresholds):
            class_values = aps[0, index, 0, 0, 0]
            value = float(torch.nanmean(class_values).detach().cpu().item())
            if not math.isfinite(value):
                raise ValueError("stmetrics temporal AP must be finite")
            result[threshold] = value
        return result


class OfficialTemporalThresholdAccumulator:
    """Pooled class-macro temporal AP at one IoU threshold via stmetrics."""

    def __init__(
        self,
        threshold: float,
        *,
        dataset_spec: str | Path = DEFAULT_DATASET_SPEC,
        min_region_size: int = 100,
    ) -> None:
        self.threshold = float(threshold)
        self._curve = OfficialTemporalCurveAccumulator(
            (self.threshold,),
            dataset_spec=dataset_spec,
            min_region_size=min_region_size,
        )

    def update(
        self,
        prediction: Mapping[str, Any],
        target: Mapping[str, Any],
    ) -> None:
        self._curve.update(prediction, target)

    def compute(self) -> float:
        return self._curve.compute()[self.threshold]


def _mask_iou_matrix(target_masks: Tensor, prediction_masks: Tensor) -> Tensor:
    target = target_masks.bool()
    prediction = prediction_masks.bool()
    intersection = target.to(torch.float64) @ prediction.to(torch.float64).T
    target_size = target.sum(dim=1, dtype=torch.float64)[:, None]
    prediction_size = prediction.sum(dim=1, dtype=torch.float64)[None, :]
    union = target_size + prediction_size - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def classify_observation_coverage(
    *,
    prediction_masks: Tensor,
    prediction_classes: Tensor,
    valid: Tensor,
    target_masks: Tensor,
    target_classes: Tensor,
    threshold: float,
) -> tuple[str, ...]:
    """Classify each GT stage into one mutually exclusive coverage outcome."""

    predictions = _cpu_tensor(prediction_masks, name="prediction_masks", ndim=2)
    classes = _cpu_tensor(prediction_classes, name="prediction_classes", ndim=1).long()
    valid_mask = _cpu_tensor(valid, name="valid", ndim=1)
    targets = _cpu_tensor(target_masks, name="target_masks", ndim=2)
    target_labels = _cpu_tensor(target_classes, name="target_classes", ndim=1).long()
    if predictions.dtype != torch.bool or targets.dtype != torch.bool:
        raise ValueError("prediction and target masks must use bool dtype")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid must use bool dtype")
    if predictions.shape[0] != classes.numel() or classes.shape != valid_mask.shape:
        raise ValueError("prediction fields must share the candidate dimension")
    if targets.shape[0] != target_labels.numel():
        raise ValueError("target fields must share the GT dimension")
    if predictions.shape[1] != targets.shape[1]:
        raise ValueError("prediction and target masks must share the point dimension")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and within [0, 1]")

    selected_masks = predictions[valid_mask]
    selected_classes = classes[valid_mask]
    if selected_masks.shape[0] == 0:
        return tuple("no_candidate_observation" for _ in range(targets.shape[0]))
    ious = _mask_iou_matrix(targets, selected_masks)
    outcomes = []
    for index, target_class in enumerate(target_labels.tolist()):
        positive = ious[index] > 0
        if not torch.any(positive).item():
            outcomes.append("no_candidate_observation")
            continue
        compatible = positive & (selected_classes == int(target_class))
        if not torch.any(compatible).item():
            outcomes.append("wrong_class")
            continue
        best = float(ious[index, compatible].max().item())
        outcomes.append("associable" if best >= float(threshold) else "insufficient_iou")
    return tuple(outcomes)


def classify_decomposition_failure(
    event: object,
    *,
    coverage_category: str | None,
) -> str:
    """Map one P6-A failure event into the paper decomposition taxonomy."""

    from scripts.p6a_analysis import classify_failure

    if coverage_category is not None and coverage_category not in COVERAGE_CATEGORIES:
        raise ValueError("coverage_category is outside the registered taxonomy")
    code = classify_failure(event)
    if code == "F1":
        return {
            "no_candidate_observation": "local_observation_miss",
            "wrong_class": "class_failure",
            "insufficient_iou": "high_iou_mask_failure",
        }.get(coverage_category, "unknown_unresolved")
    return {
        "F3": "identity_fragmentation",
        "F4": "identity_merge",
        "F5": "wrong_gap_recovery",
        "F6": "class_failure",
        "F7": "capacity_failure",
    }.get(code, "unknown_unresolved")


def classify_ceiling(
    *,
    persistent: Mapping[int, float],
    full_history: Mapping[int, float],
    oracle: Mapping[int, float],
    minimum_gain: float = 0.05,
    minimum_gap_closure: float = 0.50,
) -> str:
    """Apply the preregistered T4/T5 association-vs-perception gate."""

    if set(persistent) != {4, 5} or set(full_history) != {4, 5} or set(oracle) != {4, 5}:
        raise ValueError("ceiling metrics must cover exactly T4 and T5")
    if not 0 <= float(minimum_gain) <= 1 or not 0 <= float(minimum_gap_closure) <= 1:
        raise ValueError("ceiling thresholds must be within [0, 1]")
    for horizon in (4, 5):
        incumbent = float(persistent[horizon])
        reference = float(full_history[horizon])
        upper = float(oracle[horizon])
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (incumbent, reference, upper)):
            raise ValueError("ceiling metrics must be finite rates")
        gain = upper - incumbent
        if gain < float(minimum_gain):
            continue
        gap = reference - incumbent
        if gap <= 0 or gain / gap >= float(minimum_gap_closure):
            return "ASSOCIATION_CEILING"
    return "PERCEPTION_CEILING"


def build_oracle_accumulator(
    payloads: Sequence[Mapping[str, object]],
    *,
    sequence_id: str,
    background_class: int,
) -> IdentityAccumulator:
    """Build post-hoc GT-associated identities from frozen local observations."""

    from scripts.evaluate_persist4d_p6a import (
        cache_payload_to_frozen_observation,
        stage_prediction_from_track_step,
    )

    if not payloads:
        raise ValueError("payloads must not be empty")
    observations = tuple(cache_payload_to_frozen_observation(payload) for payload in payloads)
    targets = []
    for payload in payloads:
        target = payload.get("target")
        if not isinstance(target, Mapping):
            raise TypeError("payload target must be a mapping")
        gt_ids = _cpu_tensor(target.get("gt_ids"), name="gt_ids", ndim=1).long()
        gt_classes = _cpu_tensor(
            target.get("gt_classes"), name="gt_classes", ndim=1
        ).long()
        gt_masks = _cpu_tensor(target.get("gt_masks"), name="gt_masks", ndim=2)
        targets.append(
            OracleStageTarget(
                gt_ids=tuple(int(value) for value in gt_ids.tolist()),
                classes=tuple(int(value) for value in gt_classes.tolist()),
                masks=gt_masks.bool(),
            )
        )
    steps = run_oracle_posthoc(
        observations,
        tuple(targets),
        sequence_id=sequence_id,
        stage_ids=tuple(range(len(payloads))),
        background_class=background_class,
    )
    accumulator = IdentityAccumulator()
    for payload, step in zip(payloads, steps, strict=True):
        accumulator.add_stage(
            stage_prediction_from_track_step(
                payload,
                step,
                background_class=background_class,
            )
        )
    return accumulator
