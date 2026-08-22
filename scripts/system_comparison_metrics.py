"""Causal task quality and deployment identity metrics for system comparison."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

from scripts.p6a_metrics import OfficialMetricAccumulator, match_instances_hungarian
from scripts.system_comparison_inference import (
    unpack_bool_matrix,
    validate_full_history_payload,
)


class SystemMetricError(ValueError):
    """Raised when task or deployment identity inputs are not comparable."""


def _finite_tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise SystemMetricError(f"{name} must be a tensor")
    tensor = value.detach().cpu().contiguous().clone()
    if tensor.ndim != ndim:
        raise SystemMetricError(f"{name} must have rank {ndim}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise SystemMetricError(f"{name} must contain finite values")
    return tensor


def _integer_tensor(value: object, *, name: str, ndim: int = 1) -> Tensor:
    tensor = _finite_tensor(value, name=name, ndim=ndim)
    if tensor.dtype == torch.bool or tensor.is_floating_point() or tensor.is_complex():
        raise SystemMetricError(f"{name} must use an integer dtype")
    return tensor.long()


def _prediction(value: Mapping[str, object]) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or set(value) != {
        "pred_masks",
        "pred_scores",
        "pred_classes",
    }:
        raise SystemMetricError("prediction fields differ")
    masks = _finite_tensor(value["pred_masks"], name="pred_masks", ndim=2)
    scores = _finite_tensor(value["pred_scores"], name="pred_scores", ndim=1).float()
    classes = _integer_tensor(value["pred_classes"], name="pred_classes")
    if masks.dtype != torch.bool:
        raise SystemMetricError("pred_masks must use bool dtype")
    if masks.shape[1] != scores.numel() or scores.shape != classes.shape:
        raise SystemMetricError("prediction tensors do not align")
    return {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_classes": classes,
    }


def _target(value: Mapping[str, object]) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or set(value) != {
        "masks",
        "labels",
        "ids",
        "changes",
        "temporal_stages",
    }:
        raise SystemMetricError("target fields differ")
    masks = _finite_tensor(value["masks"], name="target masks", ndim=2)
    labels = _integer_tensor(value["labels"], name="target labels")
    ids = _integer_tensor(value["ids"], name="target IDs")
    changes = _integer_tensor(value["changes"], name="target changes")
    stages = _integer_tensor(value["temporal_stages"], name="temporal stages")
    if masks.dtype != torch.bool:
        raise SystemMetricError("target masks must use bool dtype")
    if (
        masks.shape != (labels.numel(), stages.numel())
        or labels.shape != ids.shape
        or ids.shape != changes.shape
        or len(set(ids.tolist())) != ids.numel()
        or torch.any(changes != 0).item()
    ):
        raise SystemMetricError("target tensors do not align")
    return {
        "masks": masks,
        "labels": labels,
        "ids": ids,
        "changes": changes,
        "temporal_stages": stages,
    }


@dataclass(frozen=True)
class CausalPrefixPair:
    prediction: dict[str, Tensor]
    target: dict[str, Tensor]
    horizon: int
    observed_scan_ids: tuple[str, ...]


def validate_causal_prefix_pair(
    *,
    prediction: Mapping[str, object],
    target: Mapping[str, object],
    horizon: int,
    observed_scan_ids: Sequence[str],
) -> CausalPrefixPair:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 5:
        raise SystemMetricError("horizon must be within T1-T5")
    if isinstance(observed_scan_ids, (str, bytes)) or not isinstance(
        observed_scan_ids, Sequence
    ):
        raise SystemMetricError("observed scan IDs must be a sequence")
    scan_ids = tuple(observed_scan_ids)
    if (
        len(scan_ids) != horizon
        or len(set(scan_ids)) != horizon
        or any(not isinstance(value, str) or not value for value in scan_ids)
    ):
        raise SystemMetricError("observed scan IDs must match the horizon")
    normalized_prediction = _prediction(prediction)
    normalized_target = _target(target)
    stages = normalized_target["temporal_stages"]
    if stages.numel() == 0 or set(stages.tolist()) != set(range(horizon)):
        raise SystemMetricError("target temporal stages contain future or missing data")
    if int(stages.max().item()) >= horizon:
        raise SystemMetricError("target contains future temporal stages")
    if normalized_prediction["pred_masks"].shape[0] != stages.numel():
        raise SystemMetricError("prediction points differ from the causal target")
    return CausalPrefixPair(
        prediction=normalized_prediction,
        target=normalized_target,
        horizon=horizon,
        observed_scan_ids=scan_ids,
    )


def causal_prefix_pair_from_payload(
    payload: Mapping[str, object],
) -> CausalPrefixPair:
    validated = validate_full_history_payload(payload)
    key = validated["key"]
    task = validated["task_prediction"]
    target = validated["target"]
    return validate_causal_prefix_pair(
        prediction={
            "pred_masks": unpack_bool_matrix(task["pred_masks"]),
            "pred_scores": task["pred_scores"],
            "pred_classes": task["pred_classes"],
        },
        target={
            "masks": unpack_bool_matrix(target["masks"]),
            "labels": target["labels"],
            "ids": target["ids"],
            "changes": target["changes"],
            "temporal_stages": target["temporal_stages"],
        },
        horizon=int(key["horizon"]),
        observed_scan_ids=tuple(key["history_scan_ids"]),
    )


def current_stage_pair(pair: CausalPrefixPair) -> CausalPrefixPair:
    if not isinstance(pair, CausalPrefixPair):
        raise SystemMetricError("current-stage input must be a causal prefix pair")
    selector = pair.target["temporal_stages"] == pair.horizon - 1
    if not torch.any(selector).item():
        raise SystemMetricError("causal prefix lacks current-stage points")
    target_masks = pair.target["masks"][:, selector]
    present = target_masks.any(dim=1)
    return validate_causal_prefix_pair(
        prediction={
            **pair.prediction,
            "pred_masks": pair.prediction["pred_masks"][selector],
        },
        target={
            "masks": target_masks[present],
            "labels": pair.target["labels"][present],
            "ids": pair.target["ids"][present],
            "changes": torch.zeros(int(present.sum().item()), dtype=torch.long),
            "temporal_stages": torch.zeros(int(selector.sum().item()), dtype=torch.long),
        },
        horizon=1,
        observed_scan_ids=(pair.observed_scan_ids[-1],),
    )


def _metric_factory(mode: str) -> OfficialMetricAccumulator:
    return OfficialMetricAccumulator(mode=mode)


class CausalTaskAccumulator:
    """Stream causal-prefix and current-stage official metrics at one horizon."""

    def __init__(
        self,
        *,
        metric_factory: Callable[[str], object] = _metric_factory,
    ) -> None:
        if not callable(metric_factory):
            raise SystemMetricError("task metric factory must be callable")
        self._prefix_metric = metric_factory("strict_online")
        self._current_metric = metric_factory("raw_local")
        self._horizon: int | None = None
        self._count = 0

    def update(self, pair: CausalPrefixPair) -> None:
        if not isinstance(pair, CausalPrefixPair):
            raise SystemMetricError("task metric value must be a causal prefix pair")
        if self._horizon is None:
            self._horizon = pair.horizon
        elif pair.horizon != self._horizon:
            raise SystemMetricError("task metric pairs must share one horizon")
        self._prefix_metric.update(pair.prediction, pair.target)
        current = current_stage_pair(pair)
        self._current_metric.update(current.prediction, current.target)
        self._count += 1

    def compute(self) -> dict[str, float]:
        if not self._count:
            raise SystemMetricError("task metrics require causal prefix pairs")
        prefix_values = self._prefix_metric.compute()
        current_values = self._current_metric.compute()
        mapping = {
            "causal_prefix_t_mAP": prefix_values["online_t-mAP"],
            "causal_prefix_t_mAP50": prefix_values["online_t-mAP50"],
            "causal_prefix_t_mAP25": prefix_values["online_t-mAP25"],
            "causal_prefix_t_REC": prefix_values["online_t-REC"],
            "causal_prefix_t_REC50": prefix_values["online_t-REC50"],
            "causal_prefix_t_REC25": prefix_values["online_t-REC25"],
            "current_stage_AP": current_values["raw_local_AP"],
            "current_stage_AP50": current_values["raw_local_AP50"],
            "current_stage_AP25": current_values["raw_local_AP25"],
            "current_stage_REC": current_values["raw_local_REC"],
        }
        result = {name: float(value) for name, value in mapping.items()}
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in result.values()
        ):
            raise SystemMetricError("task metrics must be finite rates")
        return result


def compute_causal_task_metrics(
    pairs: Sequence[CausalPrefixPair],
    *,
    metric_factory: Callable[[str], object] = _metric_factory,
) -> dict[str, float]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
        raise SystemMetricError("task metrics require causal prefix pairs")
    if any(not isinstance(pair, CausalPrefixPair) for pair in pairs):
        raise SystemMetricError("task metric values must be causal prefix pairs")
    accumulator = CausalTaskAccumulator(metric_factory=metric_factory)
    for pair in pairs:
        accumulator.update(pair)
    return accumulator.compute()


def _identity_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SystemMetricError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class IdentityAssignmentUpdate:
    horizon: int
    visible_gt_ids: tuple[int, ...]
    assignments: dict[int, int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or not 1 <= self.horizon <= 5
        ):
            raise SystemMetricError("identity update horizon must be within T1-T5")
        if not isinstance(self.visible_gt_ids, tuple):
            raise SystemMetricError("visible GT IDs must be a tuple")
        visible = tuple(
            _identity_id(value, name="visible GT ID") for value in self.visible_gt_ids
        )
        if len(set(visible)) != len(visible):
            raise SystemMetricError("visible GT IDs must be unique")
        if not isinstance(self.assignments, dict):
            raise SystemMetricError("identity assignments must be a dict")
        normalized = {
            _identity_id(gt_id, name="assignment GT ID"): _identity_id(
                issued_id, name="issued ID"
            )
            for gt_id, issued_id in self.assignments.items()
        }
        if not set(normalized) <= set(visible):
            raise SystemMetricError("assignments contain a non-visible GT ID")
        if len(set(normalized.values())) != len(normalized):
            raise SystemMetricError("issued IDs must be unique within one update")
        object.__setattr__(self, "visible_gt_ids", visible)
        object.__setattr__(self, "assignments", normalized)


def match_identity_update(
    *,
    horizon: int,
    gt_ids: Tensor,
    gt_classes: Tensor,
    gt_masks: Tensor,
    issued_ids: Tensor,
    pred_classes: Tensor,
    pred_masks: Tensor,
    minimum_iou: float = 0.5,
) -> IdentityAssignmentUpdate:
    if not 0.0 <= minimum_iou <= 1.0:
        raise SystemMetricError("minimum_iou must be within [0, 1]")
    normalized_gt_ids = _integer_tensor(gt_ids, name="GT IDs")
    normalized_gt_classes = _integer_tensor(gt_classes, name="GT classes")
    normalized_gt_masks = _finite_tensor(gt_masks, name="GT masks", ndim=2)
    normalized_issued_ids = _integer_tensor(issued_ids, name="issued IDs")
    normalized_pred_classes = _integer_tensor(pred_classes, name="predicted classes")
    normalized_pred_masks = _finite_tensor(pred_masks, name="predicted masks", ndim=2)
    if normalized_gt_masks.dtype != torch.bool or normalized_pred_masks.dtype != torch.bool:
        raise SystemMetricError("identity masks must use bool dtype")
    if (
        normalized_gt_ids.shape != normalized_gt_classes.shape
        or normalized_gt_masks.shape[0] != normalized_gt_ids.numel()
        or normalized_pred_masks.shape[1] != normalized_issued_ids.numel()
        or normalized_issued_ids.shape != normalized_pred_classes.shape
        or normalized_gt_masks.shape[1] != normalized_pred_masks.shape[0]
        or len(set(normalized_gt_ids.tolist())) != normalized_gt_ids.numel()
        or len(set(normalized_issued_ids.tolist())) != normalized_issued_ids.numel()
    ):
        raise SystemMetricError("identity matching tensors do not align")
    present = normalized_gt_masks.any(dim=1)
    visible_ids = normalized_gt_ids[present]
    visible_classes = normalized_gt_classes[present]
    visible_masks = normalized_gt_masks[present]
    try:
        pairs = match_instances_hungarian(
            visible_masks,
            normalized_pred_masks.transpose(0, 1).contiguous(),
            gt_classes=visible_classes,
            pred_classes=normalized_pred_classes,
            threshold=minimum_iou,
        )
    except (TypeError, ValueError) as error:
        raise SystemMetricError("identity Hungarian matching failed") from error
    assignments = {
        int(visible_ids[gt_index].item()): int(
            normalized_issued_ids[pred_index].item()
        )
        for gt_index, pred_index in pairs
    }
    return IdentityAssignmentUpdate(
        horizon=horizon,
        visible_gt_ids=tuple(int(value) for value in visible_ids.tolist()),
        assignments=assignments,
    )


def identity_update_from_payload(
    payload: Mapping[str, object],
    *,
    minimum_iou: float = 0.5,
) -> IdentityAssignmentUpdate:
    validated = validate_full_history_payload(payload)
    key = validated["key"]
    horizon = int(key["horizon"])
    target = validated["target"]
    stages = _integer_tensor(target["temporal_stages"], name="target stages")
    selector = stages == horizon - 1
    target_masks = unpack_bool_matrix(target["masks"])[:, selector]
    identity = validated["identity_prediction"]
    return match_identity_update(
        horizon=horizon,
        gt_ids=target["ids"],
        gt_classes=target["labels"],
        gt_masks=target_masks,
        issued_ids=identity["issued_ids"],
        pred_classes=identity["pred_classes"],
        pred_masks=unpack_bool_matrix(identity["pred_masks"]),
        minimum_iou=minimum_iou,
    )


def identity_updates_from_payloads(
    payloads: Sequence[Mapping[str, object]],
    *,
    minimum_iou: float = 0.5,
) -> tuple[IdentityAssignmentUpdate, ...]:
    if isinstance(payloads, (str, bytes)) or not isinstance(payloads, Sequence):
        raise SystemMetricError("identity payloads must be a sequence")
    normalized = [validate_full_history_payload(payload) for payload in payloads]
    normalized.sort(key=lambda item: int(item["key"]["horizon"]))
    if [item["key"]["horizon"] for item in normalized] != [1, 2, 3, 4, 5]:
        raise SystemMetricError("identity payloads must cover exact T1-T5 updates")
    first = normalized[0]["key"]
    for item in normalized:
        key = item["key"]
        if any(
            key[field] != first[field]
            for field in (
                "master_sequence_id",
                "reference_scene_id",
                "order_id",
            )
        ):
            raise SystemMetricError("identity payload sequence scope differs")
        horizon = int(key["horizon"])
        if key["history_scan_ids"] != normalized[-1]["key"]["history_scan_ids"][:horizon]:
            raise SystemMetricError("identity payloads are not nested exact prefixes")
    return tuple(
        identity_update_from_payload(item, minimum_iou=minimum_iou)
        for item in normalized
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute_deployment_identity_metrics(
    updates: Sequence[IdentityAssignmentUpdate],
) -> dict[str, int | float | None]:
    if isinstance(updates, (str, bytes)) or not isinstance(updates, Sequence) or not updates:
        raise SystemMetricError("deployment identity requires update records")
    if any(not isinstance(update, IdentityAssignmentUpdate) for update in updates):
        raise SystemMetricError("deployment identity records have invalid types")
    horizons = [update.horizon for update in updates]
    if horizons != list(range(horizons[0], horizons[0] + len(horizons))):
        raise SystemMetricError("deployment identity updates must be consecutive")

    transition_opportunities = 0
    switches = 0
    for previous, current in pairwise(updates):
        comparable = set(previous.assignments) & set(current.assignments)
        transition_opportunities += len(comparable)
        switches += sum(
            previous.assignments[gt_id] != current.assignments[gt_id]
            for gt_id in comparable
        )

    assignments_by_gt: dict[int, list[int]] = defaultdict(list)
    assignments_by_issued: dict[int, list[int]] = defaultdict(list)
    for update in updates:
        for gt_id, issued_id in update.assignments.items():
            assignments_by_gt[gt_id].append(issued_id)
            assignments_by_issued[issued_id].append(gt_id)
    fragmentation_count = sum(
        max(0, len(set(values)) - 1) for values in assignments_by_gt.values()
    )
    fragmentation_opportunities = sum(
        max(0, len(values) - 1) for values in assignments_by_gt.values()
    )
    merge_count = sum(
        max(0, len(set(values)) - 1) for values in assignments_by_issued.values()
    )
    merge_opportunities = sum(
        max(0, len(values) - 1) for values in assignments_by_issued.values()
    )

    updates_by_horizon = {update.horizon: update for update in updates}
    visible_horizons: dict[int, list[int]] = defaultdict(list)
    for update in updates:
        for gt_id in update.visible_gt_ids:
            visible_horizons[gt_id].append(update.horizon)
    gap_opportunities = 0
    recovery_attempts = 0
    correct_recoveries = 0
    for gt_id, horizons_for_gt in visible_horizons.items():
        for previous_horizon, current_horizon in pairwise(horizons_for_gt):
            if current_horizon - previous_horizon <= 1:
                continue
            gap_opportunities += 1
            previous_id = updates_by_horizon[previous_horizon].assignments.get(gt_id)
            current_id = updates_by_horizon[current_horizon].assignments.get(gt_id)
            if previous_id is None or current_id is None:
                continue
            recovery_attempts += 1
            correct_recoveries += int(previous_id == current_id)

    return {
        "deployment_id_switches": int(switches),
        "identity_transition_opportunities": int(transition_opportunities),
        "normalized_id_switch_rate": _rate(switches, transition_opportunities),
        "fragmentation_count": int(fragmentation_count),
        "fragmentation_opportunities": int(fragmentation_opportunities),
        "fragmentation_rate": _rate(
            fragmentation_count, fragmentation_opportunities
        ),
        "merge_count": int(merge_count),
        "merge_opportunities": int(merge_opportunities),
        "merge_rate": _rate(merge_count, merge_opportunities),
        "gap_opportunities": int(gap_opportunities),
        "recovery_attempts": int(recovery_attempts),
        "correct_recoveries": int(correct_recoveries),
        "gap_recovery_accuracy": _rate(correct_recoveries, recovery_attempts),
        "gap_recovery_recall": _rate(correct_recoveries, gap_opportunities),
    }


def deployment_identity_metrics_by_horizon(
    updates: Sequence[IdentityAssignmentUpdate],
    *,
    report_horizons: Sequence[int] = (2, 3, 4, 5),
) -> dict[int, dict[str, int | float | None]]:
    if [update.horizon for update in updates] != [1, 2, 3, 4, 5]:
        raise SystemMetricError("identity horizon analysis requires T1-T5 updates")
    if tuple(report_horizons) != (2, 3, 4, 5):
        raise SystemMetricError("reported identity horizons must be T2-T5")
    return {
        horizon: compute_deployment_identity_metrics(updates[:horizon])
        for horizon in report_horizons
    }


__all__ = [
    "CausalPrefixPair",
    "CausalTaskAccumulator",
    "IdentityAssignmentUpdate",
    "SystemMetricError",
    "causal_prefix_pair_from_payload",
    "compute_causal_task_metrics",
    "compute_deployment_identity_metrics",
    "current_stage_pair",
    "deployment_identity_metrics_by_horizon",
    "identity_update_from_payload",
    "identity_updates_from_payloads",
    "match_identity_update",
    "validate_causal_prefix_pair",
]
