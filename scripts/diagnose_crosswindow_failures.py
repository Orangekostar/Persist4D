"""GT-isolated diagnostics and deterministic CrossWindow selection helpers."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key

import torch
from torch import Tensor

from models.crosswindow_state import CrossWindowBuffer, CrossWindowState
from models.overlap_entity_association import (
    AssignmentPlan,
    EvidenceBundle,
    _assignment_sha256,
    _private_dummy_assignment,
    build_evidence,
    commit_observation,
)
from scripts.crosswindow_cache import CanonicalFrame, build_canonical_frame
from scripts.replay_crosswindow_association import (
    PublishedPrefix,
    canonicalize_stage_target,
    evaluator_prediction,
    publish,
)


class CrossWindowDiagnosticError(ValueError):
    """Raised when diagnostic evidence violates its frozen scope."""


@dataclass(frozen=True)
class DiagnosticCandidate:
    scan_id: str
    stage_index: int
    source_class_id: int
    class_id: int
    score: float
    mask: Tensor
    source: str
    source_key: str


@dataclass(frozen=True)
class E1SequenceResult:
    metric_pairs: dict[tuple[str, int], object]
    coverage_events: tuple[dict[str, object], ...]
    assignment_events: tuple[dict[str, object], ...]


def _finite_rate(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CrossWindowDiagnosticError(f"{name} must be finite")
    return float(value)


def _count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CrossWindowDiagnosticError(f"{name} must be a non-negative integer")
    return value


def evaluate_headroom_gate(
    *,
    long_gain: float,
    candidate_complete_failure_count: int,
    published_failure_count: int,
    event_reference_count: int,
    diagnostic_status: str,
    minimum_long_gain: float = 0.005,
    minimum_failure_fraction: float = 0.10,
    minimum_events: int = 20,
    minimum_references: int = 3,
) -> dict[str, object]:
    gain = _finite_rate(long_gain, name="long_gain")
    complete = _count(
        candidate_complete_failure_count,
        name="candidate_complete_failure_count",
    )
    failures = _count(published_failure_count, name="published_failure_count")
    references = _count(event_reference_count, name="event_reference_count")
    if complete > failures:
        raise CrossWindowDiagnosticError(
            "candidate-complete failures cannot exceed published failures"
        )
    for name, value in (
        ("minimum_long_gain", minimum_long_gain),
        ("minimum_failure_fraction", minimum_failure_fraction),
    ):
        threshold = _finite_rate(value, name=name)
        if threshold < 0:
            raise CrossWindowDiagnosticError(f"{name} must be non-negative")
    minimum_events = _count(minimum_events, name="minimum_events")
    minimum_references = _count(minimum_references, name="minimum_references")
    if not isinstance(diagnostic_status, str) or not diagnostic_status:
        raise CrossWindowDiagnosticError("diagnostic_status must be non-empty")
    fraction = complete / failures if failures else None
    gain_support = gain >= minimum_long_gain
    coverage_support = bool(
        fraction is not None
        and fraction >= minimum_failure_fraction
        and complete >= minimum_events
        and references >= minimum_references
    )
    if gain_support or coverage_support:
        decision = "ASSOCIATION_HEADROOM_SUPPORTED"
    elif diagnostic_status != "PASS":
        decision = "INCONCLUSIVE"
    elif failures < minimum_events or references < minimum_references:
        decision = "INSUFFICIENT_EVENTS"
    else:
        decision = "MASK_OR_OTHER_DOMINANT"
    return {
        "schema_version": "crosswindow-headroom-gate-v1",
        "decision": decision,
        "long_gain": gain,
        "minimum_long_gain": minimum_long_gain,
        "candidate_complete_failure_count": complete,
        "published_failure_count": failures,
        "candidate_complete_failure_fraction": fraction,
        "minimum_failure_fraction": minimum_failure_fraction,
        "event_reference_count": references,
        "minimum_events": minimum_events,
        "minimum_references": minimum_references,
        "diagnostic_status": diagnostic_status,
        "run_full_grid": decision == "ASSOCIATION_HEADROOM_SUPPORTED",
        "authorize_e6_by_e1": decision == "ASSOCIATION_HEADROOM_SUPPORTED",
    }


def rank_family_candidates(
    rows: Sequence[Mapping[str, object]],
    *,
    d0_by_horizon: Mapping[int, float],
    tolerance: float = 1e-6,
) -> list[dict[str, object]]:
    if not rows:
        raise CrossWindowDiagnosticError("candidate ranking rows are empty")
    tolerance = _finite_rate(tolerance, name="tolerance")
    if tolerance < 0:
        raise CrossWindowDiagnosticError("tolerance must be non-negative")
    horizons = (2, 3, 4, 5)
    if set(d0_by_horizon) != set(horizons):
        raise CrossWindowDiagnosticError("D0 ranking baseline must cover T2-T5")
    baseline = {
        horizon: _finite_rate(d0_by_horizon[horizon], name=f"D0 T{horizon}")
        for horizon in horizons
    }
    by_config: dict[str, dict[int, float]] = {}
    for row in rows:
        config_id = row.get("config_id")
        horizon = row.get("T")
        if not isinstance(config_id, str) or not config_id:
            raise CrossWindowDiagnosticError("ranking row lacks config_id")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise CrossWindowDiagnosticError("ranking row lacks integer T")
        value = _finite_rate(row.get("t_mAP"), name="candidate t_mAP")
        values = by_config.setdefault(config_id, {})
        if horizon in values:
            raise CrossWindowDiagnosticError("candidate ranking contains duplicate T")
        values[horizon] = value
    ranked = []
    for config_id, values in by_config.items():
        if set(values) != set(horizons):
            raise CrossWindowDiagnosticError(
                f"candidate {config_id} does not cover T2-T5"
            )
        deltas = {horizon: values[horizon] - baseline[horizon] for horizon in horizons}
        ranked.append(
            {
                "config_id": config_id,
                **{f"delta_T{horizon}": deltas[horizon] for horizon in horizons},
                "S_min": min(deltas.values()),
                "S_long": (deltas[4] + deltas[5]) / 2.0,
                "S_mean": sum(deltas.values()) / len(deltas),
            }
        )

    def compare(left: Mapping[str, object], right: Mapping[str, object]) -> int:
        for field in ("S_min", "S_long", "S_mean"):
            delta = float(left[field]) - float(right[field])
            if abs(delta) > tolerance:
                return -1 if delta > 0 else 1
        left_id, right_id = str(left["config_id"]), str(right["config_id"])
        return -1 if left_id < right_id else 1 if left_id > right_id else 0

    return sorted(ranked, key=cmp_to_key(compare))


def evaluate_dev_selection_gate(
    *,
    deltas: Mapping[int, float],
    positive_reference_count: int,
    reference_count: int,
    candidate_merge_rate: float | None,
    baseline_merge_rate: float | None,
    merge_event_count: int,
    candidate_wrong_reactivation_rate: float | None,
    baseline_wrong_reactivation_rate: float | None,
    reactivation_event_count: int,
    max_per_t_drop: float = 0.001,
    minimum_long_gain: float = 0.005,
    minimum_mean_gain: float = 0.0,
    minimum_positive_references: int = 3,
    maximum_error_rate_increase: float = 0.01,
    minimum_identity_events: int = 20,
) -> dict[str, object]:
    """Apply the frozen DEV-SEL quality and identity gates."""
    horizons = (2, 3, 4, 5)
    if set(deltas) != set(horizons):
        raise CrossWindowDiagnosticError("DEV-SEL deltas must cover T2-T5")
    normalized = {
        horizon: _finite_rate(deltas[horizon], name=f"delta T{horizon}")
        for horizon in horizons
    }
    positive_reference_count = _count(
        positive_reference_count, name="positive_reference_count"
    )
    reference_count = _count(reference_count, name="reference_count")
    merge_event_count = _count(merge_event_count, name="merge_event_count")
    reactivation_event_count = _count(
        reactivation_event_count, name="reactivation_event_count"
    )
    if positive_reference_count > reference_count:
        raise CrossWindowDiagnosticError(
            "positive references cannot exceed reference count"
        )
    thresholds = {
        "max_per_t_drop": max_per_t_drop,
        "minimum_long_gain": minimum_long_gain,
        "minimum_mean_gain": minimum_mean_gain,
        "maximum_error_rate_increase": maximum_error_rate_increase,
    }
    for name, value in thresholds.items():
        if _finite_rate(value, name=name) < 0:
            raise CrossWindowDiagnosticError(f"{name} must be non-negative")
    minimum_positive_references = _count(
        minimum_positive_references, name="minimum_positive_references"
    )
    minimum_identity_events = _count(
        minimum_identity_events, name="minimum_identity_events"
    )
    long_gain = (normalized[4] + normalized[5]) / 2.0
    mean_gain = sum(normalized.values()) / len(normalized)
    failed = []
    if min(normalized.values()) < -max_per_t_drop:
        failed.append("per_T_floor")
    if long_gain < minimum_long_gain:
        failed.append("long_gain")
    if mean_gain < minimum_mean_gain:
        failed.append("mean_gain")

    small_reference_count = reference_count < minimum_positive_references
    if (
        not small_reference_count
        and positive_reference_count < minimum_positive_references
    ):
        failed.append("positive_references")

    def check_identity_rate(
        *,
        name: str,
        candidate: float | None,
        baseline: float | None,
        event_count: int,
    ) -> bool:
        if event_count < minimum_identity_events:
            return False
        if candidate is None or baseline is None:
            raise CrossWindowDiagnosticError(
                f"{name} rates are required when event evidence is sufficient"
            )
        candidate_value = _finite_rate(candidate, name=f"candidate {name}")
        baseline_value = _finite_rate(baseline, name=f"baseline {name}")
        if not 0.0 <= candidate_value <= 1.0 or not 0.0 <= baseline_value <= 1.0:
            raise CrossWindowDiagnosticError(f"{name} rates must be within [0,1]")
        if candidate_value - baseline_value > maximum_error_rate_increase:
            failed.append(name)
        return True

    merge_conclusive = check_identity_rate(
        name="merge_rate",
        candidate=candidate_merge_rate,
        baseline=baseline_merge_rate,
        event_count=merge_event_count,
    )
    reactivation_conclusive = check_identity_rate(
        name="wrong_reactivation_rate",
        candidate=candidate_wrong_reactivation_rate,
        baseline=baseline_wrong_reactivation_rate,
        event_count=reactivation_event_count,
    )
    small_event_count = not (merge_conclusive and reactivation_conclusive)
    eligible = not failed
    if not eligible:
        status = "FAIL_SELECTION_GATE"
    elif small_event_count and small_reference_count:
        status = "PROVISIONAL_SMALL_EVENT_AND_REFERENCE_COUNT"
    elif small_event_count:
        status = "PROVISIONAL_SMALL_EVENT_COUNT"
    elif small_reference_count:
        status = "PROVISIONAL_SMALL_REFERENCE_COUNT"
    else:
        status = "PASS"
    return {
        "eligible": eligible,
        "status": status,
        "failed_gates": failed,
        "S_min": min(normalized.values()),
        "S_long": long_gain,
        "S_mean": mean_gain,
        "positive_reference_count": positive_reference_count,
        "reference_count": reference_count,
        "merge_event_count": merge_event_count,
        "reactivation_event_count": reactivation_event_count,
        "identity_event_evidence_conclusive": not small_event_count,
    }


def _mask_iou(left: Tensor, right: Tensor) -> float:
    if (
        left.dtype != torch.bool
        or right.dtype != torch.bool
        or left.shape != right.shape
    ):
        raise CrossWindowDiagnosticError(
            "diagnostic masks must be aligned bool vectors"
        )
    union = int((left | right).sum().item())
    return int((left & right).sum().item()) / union if union else 0.0


def _frame_candidates_for_scan(
    frame: CanonicalFrame, *, scan_id: str, source: str
) -> tuple[DiagnosticCandidate, ...]:
    result = []
    for candidate in frame.candidates:
        slices = [value for value in candidate.slices if value.scan_id == scan_id]
        if len(slices) != 1:
            raise CrossWindowDiagnosticError("candidate scan slice coverage differs")
        candidate_slice = slices[0]
        mask = candidate_slice.mask.detach().cpu().bool().clone()
        if not mask.any().item():
            continue
        result.append(
            DiagnosticCandidate(
                scan_id=scan_id,
                stage_index=frame.absolute_stage,
                source_class_id=candidate.key.source_class_id,
                class_id=candidate.predicted_class_id,
                score=float(candidate.score),
                mask=mask,
                source=source,
                source_key=repr(candidate.key),
            )
        )
    return tuple(result)


def build_candidate_pools(
    frames: Sequence[CanonicalFrame], *, horizon: int
) -> dict[str, tuple[tuple[DiagnosticCandidate, ...], ...]]:
    if len(frames) != 5 or horizon not in (2, 3, 4, 5):
        raise CrossWindowDiagnosticError(
            "candidate pools require five frames and T2-T5"
        )
    policy = []
    relaxed = []
    for scan_index in range(horizon):
        scan_id = frames[scan_index].source_window[-1]
        old = _frame_candidates_for_scan(
            frames[scan_index], scan_id=scan_id, source="old"
        )
        new = (
            _frame_candidates_for_scan(
                frames[scan_index + 1], scan_id=scan_id, source="new"
            )
            if scan_index + 1 < horizon
            else ()
        )
        policy.append(new if new else old)
        relaxed.append((*old, *new))
    return {"POLICY_POOL": tuple(policy), "RELAXED_OLD_NEW_POOL": tuple(relaxed)}


def _target_rows(target: Mapping[str, object]) -> dict[int, tuple[int, Tensor]]:
    ids = target.get("gt_ids")
    classes = target.get("gt_classes")
    masks = target.get("gt_masks")
    if not all(isinstance(value, Tensor) for value in (ids, classes, masks)):
        raise CrossWindowDiagnosticError("stage target tensors are unavailable")
    ids = ids.detach().cpu().long()
    classes = classes.detach().cpu().long()
    masks = masks.detach().cpu().bool()
    if ids.shape != classes.shape or masks.shape[0] != ids.numel():
        raise CrossWindowDiagnosticError("stage target tensors do not align")
    return {
        int(gt_id.item()): (int(classes[index].item()), masks[index].clone())
        for index, gt_id in enumerate(ids)
        if masks[index].any().item()
    }


def _published_match_by_gt(
    prefix: object,
    targets: Sequence[Mapping[str, object]],
    *,
    horizon: int,
    class_mapping: tuple[int, ...],
) -> dict[int, bool]:
    from scripts.evaluate_persist4d_p6a import build_temporal_target

    if not hasattr(prefix, "prediction"):
        raise CrossWindowDiagnosticError("D0 prefix is unavailable")
    target = build_temporal_target(
        [
            {"key": {"stage_index": index}, "target": targets[index]}
            for index in range(horizon)
        ]
    )
    target["labels"] = torch.tensor(
        [class_mapping[int(value)] for value in target["labels"].tolist()],
        dtype=torch.long,
    )
    prediction = prefix.prediction
    masks = prediction["pred_masks"].detach().cpu().bool()
    classes = prediction["pred_classes"].detach().cpu().long()
    result = {}
    for gt_index, gt_id in enumerate(target["ids"].tolist()):
        gt_mask = target["masks"][gt_index].detach().cpu().bool()
        gt_class = int(target["labels"][gt_index].item())
        result[int(gt_id)] = any(
            int(classes[index].item()) == gt_class
            and _mask_iou(gt_mask, masks[:, index]) > 0.5
            for index in range(classes.numel())
        )
    return result


def diagnose_candidate_coverage(
    *,
    reference_id: str,
    master_id: str,
    order_id: str,
    frames: Sequence[CanonicalFrame],
    original_targets: Sequence[Mapping[str, object]],
    canonical_targets: Sequence[Mapping[str, object]],
    d0_prefixes: Sequence[object],
    class_mapping: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    if not (
        len(frames)
        == len(original_targets)
        == len(canonical_targets)
        == len(d0_prefixes)
        == 5
    ):
        raise CrossWindowDiagnosticError("coverage inputs must contain five stages")
    rows = []
    for horizon in (2, 3, 4, 5):
        pools = build_candidate_pools(frames, horizon=horizon)
        published = _published_match_by_gt(
            d0_prefixes[horizon - 1],
            original_targets,
            horizon=horizon,
            class_mapping=class_mapping,
        )
        stage_targets = [_target_rows(value) for value in canonical_targets[:horizon]]
        gt_ids = sorted({gt_id for target in stage_targets for gt_id in target})
        for pool_name, pool in pools.items():
            for gt_id in gt_ids:
                visible = [
                    index
                    for index, target in enumerate(stage_targets)
                    if gt_id in target
                ]
                gt_class = stage_targets[visible[0]][gt_id][0]
                stage_best_iou = []
                stage_candidate_count = []
                stage_source = []
                class_failure = False
                for stage_index in visible:
                    candidates = [
                        candidate
                        for candidate in pool[stage_index]
                        if candidate.source_class_id == gt_class
                    ]
                    class_failure |= not candidates
                    gt_mask = stage_targets[stage_index][gt_id][1]
                    scored = [
                        (_mask_iou(gt_mask, candidate.mask), candidate)
                        for candidate in candidates
                    ]
                    scored.sort(
                        key=lambda value: (
                            -value[0],
                            -value[1].score,
                            value[1].source_key,
                        )
                    )
                    stage_best_iou.append(scored[0][0] if scored else 0.0)
                    stage_candidate_count.append(len(candidates))
                    stage_source.append(scored[0][1].source if scored else "none")
                for threshold in (0.25, 0.5, 0.75):
                    complete = all(value > threshold for value in stage_best_iou)
                    published_failure = not published.get(gt_id, False)
                    first_failure = next(
                        (
                            visible[index]
                            for index, value in enumerate(stage_best_iou)
                            if value <= threshold
                        ),
                        None,
                    )
                    rows.append(
                        {
                            "reference_id": reference_id,
                            "master_id": master_id,
                            "order_id": order_id,
                            "prefix_T": horizon,
                            "gt_id": gt_id,
                            "tau": threshold,
                            "pool": pool_name,
                            "candidate_complete": complete,
                            "published_failure": published_failure,
                            "diag_complete_candidate_failure": (
                                complete and published_failure
                            ),
                            "visible_stage_count": len(visible),
                            "first_failure_stage": first_failure,
                            "gap": max(0, visible[-1] - visible[0] + 1 - len(visible)),
                            "class_failure": class_failure,
                            "stage_best_iou": ";".join(
                                f"{value:.9g}" for value in stage_best_iou
                            ),
                            "stage_candidate_count": ";".join(
                                str(value) for value in stage_candidate_count
                            ),
                            "selected_source": ";".join(stage_source),
                            "ambiguity_status": "AMBIGUITY_METADATA_UNAVAILABLE",
                        }
                    )
    return tuple(rows)


def _group_target_tags(
    frame: CanonicalFrame, target: Mapping[str, object]
) -> tuple[frozenset[int], ...]:
    targets = _target_rows(target)
    scan_id = frame.source_window[-1]
    groups = sorted(frame.groups, key=lambda group: group.source_query_id)
    tags = []
    for group in groups:
        matches = set()
        for candidate_index in group.candidate_indices:
            candidate = frame.candidates[candidate_index]
            slices = [value for value in candidate.slices if value.scan_id == scan_id]
            if len(slices) != 1:
                raise CrossWindowDiagnosticError("group current slice differs")
            mask = slices[0].mask.detach().cpu().bool()
            if not mask.any().item():
                continue
            for gt_id, (gt_class, gt_mask) in targets.items():
                if (
                    candidate.key.source_class_id == gt_class
                    and _mask_iou(mask, gt_mask) > 0.5
                ):
                    matches.add(gt_id)
        tags.append(frozenset(matches))
    return tuple(tags)


def _gt_assignment_plan(
    bundle: EvidenceBundle,
    *,
    group_tags: tuple[frozenset[int], ...],
    anchor_tags: Mapping[tuple[int, int], frozenset[int]],
) -> AssignmentPlan:
    bundle.validate()
    if len(group_tags) != bundle.group_count:
        raise CrossWindowDiagnosticError("diagnostic group tags do not align")
    score = torch.zeros((bundle.group_count, bundle.anchor_count), dtype=torch.float32)
    for row, tags in enumerate(group_tags):
        if len(tags) != 1:
            continue
        for column, key in enumerate(
            zip(bundle.anchor_logical_ids, bundle.anchor_generations, strict=True)
        ):
            if anchor_tags.get(key, frozenset()) == tags:
                score[row, column] = 1.0
    matches = _private_dummy_assignment(
        score,
        tau=0.5,
        row_indices=tuple(range(bundle.group_count)),
        column_indices=tuple(range(bundle.anchor_count)),
    )
    entity_for_group = []
    generation_for_group = []
    anchor_for_group = []
    is_new_id = []
    unmatched_reason = []
    decision_score = []
    decision_margin = []
    source_edge = []
    next_logical_id = bundle.next_logical_id
    for row in range(bundle.group_count):
        column = matches.get(row, -1)
        if column >= 0:
            entity_for_group.append(bundle.anchor_logical_ids[column])
            generation_for_group.append(bundle.anchor_generations[column])
            anchor_for_group.append(column)
            is_new_id.append(False)
            unmatched_reason.append(None)
            decision_score.append(1.0)
            decision_margin.append(1.0)
            source_edge.append("GT_TAG_DIAGNOSTIC")
        else:
            entity_for_group.append(next_logical_id)
            next_logical_id += 1
            generation_for_group.append(0)
            anchor_for_group.append(-1)
            is_new_id.append(True)
            unmatched_reason.append(
                "ONE_TO_ONE_CONFLICT"
                if bool((score[row] > 0.5).any().item())
                else "GT_TAG_UNAVAILABLE"
            )
            decision_score.append(0.0)
            decision_margin.append(0.0)
            source_edge.append("PRIVATE_DUMMY")

    slot_for_group = [-1] * bundle.group_count
    residency_reason = [""] * bundle.group_count
    for row, column in matches.items():
        slot = bundle.anchor_slots[column]
        if slot >= 0:
            slot_for_group[row] = slot
            residency_reason[row] = "INHERITED_RESIDENT"
    free_slots = list(bundle.free_slots)

    def allocation_key(row: int) -> tuple[float, object]:
        return -bundle.group_confidence[row], bundle.group_keys[row]

    promotions = sorted(
        (
            row
            for row, column in matches.items()
            if bundle.anchor_slots[column] < 0
            and bundle.group_valid[row]
            and bundle.group_current_supported[row]
        ),
        key=allocation_key,
    )
    births = sorted(
        (
            row
            for row in range(bundle.group_count)
            if row not in matches
            and bundle.group_valid[row]
            and bundle.group_current_supported[row]
        ),
        key=allocation_key,
    )
    for row in promotions + births:
        if free_slots:
            slot_for_group[row] = free_slots.pop(0)
            residency_reason[row] = (
                "PROMOTED_BUFFER" if row in matches else "BIRTH_RESIDENT"
            )
        else:
            residency_reason[row] = (
                "BUFFER_ONLY_CAPACITY_FULL" if row in matches else "BIRTH_CAPACITY_FULL"
            )
    for row in range(bundle.group_count):
        if residency_reason[row]:
            continue
        residency_reason[row] = (
            "BUFFER_ONLY_NO_CURRENT_STATE"
            if row in matches
            else "NEW_ID_NO_CURRENT_STATE"
        )
    values = {
        "schema_version": "crosswindow-assignment-plan-v1",
        "method_id": "GT-ID-LAG1",
        "source_state_sha256": bundle.source_state_sha256,
        "frame_sha256": bundle.frame_sha256,
        "reference_id": bundle.reference_id,
        "episode_id": bundle.episode_id,
        "absolute_stage": bundle.absolute_stage,
        "group_keys": bundle.group_keys,
        "entity_for_group": tuple(entity_for_group),
        "generation_for_group": tuple(generation_for_group),
        "slot_for_group": tuple(slot_for_group),
        "anchor_for_group": tuple(anchor_for_group),
        "is_new_id": tuple(is_new_id),
        "unmatched_reason": tuple(unmatched_reason),
        "residency_reason": tuple(residency_reason),
        "decision_score": tuple(decision_score),
        "decision_margin": tuple(decision_margin),
        "source_edge": tuple(source_edge),
    }
    provisional = AssignmentPlan(content_sha256="0" * 64, **values)
    plan = AssignmentPlan(
        content_sha256=_assignment_sha256(provisional, include_digest=False), **values
    )
    plan.validate()
    return plan


def gt_lag1_prefixes(
    *,
    frames: Sequence[CanonicalFrame],
    canonical_targets: Sequence[Mapping[str, object]],
) -> tuple[tuple[PublishedPrefix, ...], tuple[dict[str, object], ...]]:
    if len(frames) != 5 or len(canonical_targets) != 5:
        raise CrossWindowDiagnosticError("GT-ID-LAG1 requires five stages")
    first = frames[0].groups[0]
    state = CrossWindowState.empty(
        capacity=100,
        feature_dim=first.feature.numel(),
        class_count=first.class_prob.numel(),
    )
    buffer: CrossWindowBuffer | None = None
    boundary = None
    anchor_tags: dict[tuple[int, int], frozenset[int]] = {}
    prefixes = []
    events = []
    for frame, target in zip(frames, canonical_targets, strict=True):
        bundle = build_evidence(frame, state, buffer)
        group_tags = _group_target_tags(frame, target)
        plan = _gt_assignment_plan(
            bundle, group_tags=group_tags, anchor_tags=anchor_tags
        )
        next_state, committed = commit_observation(frame, state, plan)
        prefix = publish(frame, plan, mask_selection="new", history_boundary=boundary)
        for index, tags in enumerate(group_tags):
            key = (plan.entity_for_group[index], plan.generation_for_group[index])
            anchor_tags[key] = frozenset(anchor_tags.get(key, frozenset()) | tags)
        live = {
            (
                int(next_state.logical_ids[slot].item()),
                int(next_state.generations[slot].item()),
            )
            for slot in next_state.occupied.nonzero(as_tuple=True)[0].tolist()
        } | {(group.logical_id, group.generation) for group in committed.buffer.groups}
        anchor_tags = {key: value for key, value in anchor_tags.items() if key in live}
        events.append(
            {
                "absolute_stage": frame.absolute_stage,
                "group_count": len(plan.group_keys),
                "gt_tag_match_count": sum(
                    source == "GT_TAG_DIAGNOSTIC" for source in plan.source_edge
                ),
                "birth_count": sum(plan.is_new_id),
                "ambiguous_group_count": sum(len(tags) > 1 for tags in group_tags),
                "untagged_group_count": sum(not tags for tags in group_tags),
                "diagnostic_only": True,
            }
        )
        state = next_state
        buffer = committed.buffer
        boundary = prefix.history_boundary
        prefixes.append(prefix)
    return tuple(prefixes), tuple(events)


def gt_relaxed_prediction(
    *,
    policy_pool: Sequence[Sequence[DiagnosticCandidate]],
    canonical_targets: Sequence[Mapping[str, object]],
) -> dict[str, Tensor]:
    from scripts.p6a_metrics import match_instances_hungarian

    if len(policy_pool) != len(canonical_targets) or not policy_pool:
        raise CrossWindowDiagnosticError("GT-ID-RELAXED inputs do not align")
    occurrences: dict[tuple[str, int], list[tuple[int, float, Tensor]]] = defaultdict(
        list
    )
    point_counts = []
    for stage_index, (candidates, target) in enumerate(
        zip(policy_pool, canonical_targets, strict=True)
    ):
        target_rows = _target_rows(target)
        target_masks = target.get("gt_masks")
        if not isinstance(target_masks, Tensor) or target_masks.ndim != 2:
            raise CrossWindowDiagnosticError("canonical target masks are unavailable")
        point_count = target_masks.shape[1]
        point_counts.append(point_count)
        gt_ids = tuple(target_rows)
        gt_classes = torch.tensor(
            [target_rows[gt_id][0] for gt_id in gt_ids], dtype=torch.long
        )
        gt_masks = (
            torch.stack([target_rows[gt_id][1] for gt_id in gt_ids])
            if gt_ids
            else torch.empty((0, point_count), dtype=torch.bool)
        )
        pred_classes = torch.tensor(
            [candidate.source_class_id for candidate in candidates], dtype=torch.long
        )
        pred_masks = (
            torch.stack([candidate.mask for candidate in candidates])
            if candidates
            else torch.empty((0, point_count), dtype=torch.bool)
        )
        pairs = match_instances_hungarian(
            gt_masks,
            pred_masks,
            gt_classes=gt_classes,
            pred_classes=pred_classes,
            threshold=math.nextafter(0.5, 1.0),
        )
        gt_for_candidate = {
            candidate_index: gt_ids[gt_index] for gt_index, candidate_index in pairs
        }
        for candidate_index, candidate in enumerate(candidates):
            identity = (
                f"gt:{gt_for_candidate[candidate_index]}"
                if candidate_index in gt_for_candidate
                else f"unmatched:{stage_index}:{candidate_index}:{candidate.source_key}"
            )
            occurrences[(identity, candidate.class_id)].append(
                (stage_index, candidate.score, candidate.mask)
            )
    keys = sorted(occurrences)
    offsets = [0]
    for count in point_counts:
        offsets.append(offsets[-1] + count)
    masks = torch.zeros((offsets[-1], len(keys)), dtype=torch.bool)
    scores = []
    classes = []
    for column, key in enumerate(keys):
        values = occurrences[key]
        for stage_index, _, mask in values:
            start, stop = offsets[stage_index : stage_index + 2]
            masks[start:stop, column] = mask
        scores.append(sum(value[1] for value in values) / len(values))
        classes.append(key[1])
    return {
        "pred_masks": masks,
        "pred_scores": torch.tensor(scores, dtype=torch.float32),
        "pred_classes": torch.tensor(classes, dtype=torch.long),
    }


class E1DiagnosticAccumulator:
    """Stream GT-isolated E1 evidence while keeping production modules GT-free."""

    horizons = (2, 3, 4, 5)

    def __init__(
        self,
        *,
        dataset_spec: str,
        class_mapping: tuple[int, ...],
        source_commit: str,
        checkpoint_sha256: str,
    ) -> None:
        from scripts.analyze_persist4d_allt import AllTBaselineAccumulator

        if len(class_mapping) != 18 or len(set(class_mapping)) != 18:
            raise CrossWindowDiagnosticError("RIO class mapping must contain 18 IDs")
        self.class_mapping = class_mapping
        self.source_commit = source_commit
        self.checkpoint_sha256 = checkpoint_sha256
        self.metrics = {
            (method, horizon): AllTBaselineAccumulator(dataset_spec=dataset_spec)
            for method in ("D0", "GT-ID-LAG1", "GT-ID-RELAXED")
            for horizon in self.horizons
        }
        self.coverage_events: list[dict[str, object]] = []
        self.assignment_events: list[dict[str, object]] = []
        self.reference_ids: set[str] = set()
        self.unit_count = 0

    def _class_mapper(self, value: int) -> int:
        if isinstance(value, bool) or not 0 <= value < len(self.class_mapping):
            raise CrossWindowDiagnosticError("model class is outside RIO mapping")
        return self.class_mapping[value]

    def update(
        self,
        *,
        logical_unit_id: str,
        base: Mapping[str, object],
        supplement: Mapping[str, object],
    ) -> None:
        from scripts.run_task_memory_controls import (
            prediction_observation_from_payload,
            run_control_trajectory,
        )
        from scripts.run_task_memory_policy_baseline import (
            _meta_from_payload,
            _prediction_from_payload,
            _target_for_prefix,
        )
        from scripts.system_comparison_metrics import validate_causal_prefix_pair
        from scripts.task_memory_output import LagOnePublisher

        base_stages = base.get("stages")
        supplement_stages = supplement.get("stages")
        episode = base.get("episode")
        if (
            not isinstance(base_stages, list)
            or not isinstance(supplement_stages, list)
            or len(base_stages) != 5
            or len(supplement_stages) != 5
            or not isinstance(episode, Mapping)
        ):
            raise CrossWindowDiagnosticError("E1 cache stage coverage differs")
        metas = [_meta_from_payload(stage["stage_meta"]) for stage in base_stages]
        observations = [
            prediction_observation_from_payload(stage["observation"])
            for stage in supplement_stages
        ]
        predictions = [
            _prediction_from_payload(stage["prediction"]) for stage in supplement_stages
        ]
        frames = [
            build_canonical_frame(
                producer_id="R1-B4-policy",
                order_id=logical_unit_id,
                observation=observation,
                prediction=prediction,
                stage_meta=meta,
            )
            for observation, prediction, meta in zip(
                observations, predictions, metas, strict=True
            )
        ]
        targets = [stage["target"] for stage in base_stages]
        canonical_targets = [
            canonicalize_stage_target(
                target,
                original_vertex_ids=meta.original_vertex_ids[-1],
            )
            for target, meta in zip(targets, metas, strict=True)
        ]
        scan_ids = tuple(str(value) for value in episode["scan_ids"])
        trajectory = run_control_trajectory(
            observations,
            metas,
            update_mode="last",
            capacity=100,
            class_weight=0.25,
            association_threshold=0.5,
            update_rate=0.2,
        )
        publisher = LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
        d0_prefixes = [
            publisher.update(prediction, identity_map, meta)
            for prediction, identity_map, meta in zip(
                predictions, trajectory.identity_maps, metas, strict=True
            )
        ]
        lag1_prefixes, assignment_events = gt_lag1_prefixes(
            frames=frames, canonical_targets=canonical_targets
        )
        master_id = f"{episode['reference_id']}:{'|'.join(sorted(scan_ids))}"
        self.coverage_events.extend(
            diagnose_candidate_coverage(
                reference_id=str(episode["reference_id"]),
                master_id=master_id,
                order_id=str(episode["sequence_id"]),
                frames=frames,
                original_targets=targets,
                canonical_targets=canonical_targets,
                d0_prefixes=d0_prefixes,
                class_mapping=self.class_mapping,
            )
        )
        self.assignment_events.extend(
            {
                "logical_unit_id": logical_unit_id,
                "reference_id": episode["reference_id"],
                "master_id": master_id,
                "order_id": episode["sequence_id"],
                **event,
            }
            for event in assignment_events
        )
        for horizon in self.horizons:
            old_target = _target_for_prefix(
                targets, horizon=horizon, class_mapper=self._class_mapper
            )
            canonical_target = _target_for_prefix(
                canonical_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            d0_pair = validate_causal_prefix_pair(
                prediction=d0_prefixes[horizon - 1].prediction,
                target=old_target,
                horizon=horizon,
                observed_scan_ids=scan_ids[:horizon],
            )
            lag1_pair = validate_causal_prefix_pair(
                prediction=evaluator_prediction(lag1_prefixes[horizon - 1]),
                target=canonical_target,
                horizon=horizon,
                observed_scan_ids=scan_ids[:horizon],
            )
            policy_pool = build_candidate_pools(frames, horizon=horizon)["POLICY_POOL"]
            relaxed_pair = validate_causal_prefix_pair(
                prediction=gt_relaxed_prediction(
                    policy_pool=policy_pool,
                    canonical_targets=canonical_targets[:horizon],
                ),
                target=canonical_target,
                horizon=horizon,
                observed_scan_ids=scan_ids[:horizon],
            )
            self.metrics[("D0", horizon)].update(d0_pair)
            self.metrics[("GT-ID-LAG1", horizon)].update(lag1_pair)
            self.metrics[("GT-ID-RELAXED", horizon)].update(relaxed_pair)
        self.reference_ids.add(str(episode["reference_id"]))
        self.unit_count += 1

    def finalize(self) -> dict[str, object]:
        measured = {
            (method, horizon): accumulator.compute()
            for (method, horizon), accumulator in self.metrics.items()
        }
        metric_rows = []
        for method in ("D0", "GT-ID-LAG1", "GT-ID-RELAXED"):
            for horizon in self.horizons:
                values = measured[(method, horizon)]
                metric_rows.append(
                    {
                        "population_id": "development",
                        "data_role": "DEV-CAL",
                        "reference_count": len(self.reference_ids),
                        "logical_unit_count": self.unit_count,
                        "method": method,
                        "T": horizon,
                        "t_mAP": values["t_mAP"],
                        "t_mAP50": values["t_mAP50"],
                        "t_mAP25": values["t_mAP25"],
                        "t_REC": values["t_REC"],
                        "prefix_overall_mAP": values["prefix_overall_mAP"],
                        "published_current_AP": values["local_current_AP"],
                        "source_commit": self.source_commit,
                        "R1_SHA": self.checkpoint_sha256,
                        "diagnostic_only": method.startswith("GT-"),
                        "status": "MEASURED",
                        "reason": "",
                    }
                )
        grouped: dict[tuple[str, float, int], list[Mapping[str, object]]] = defaultdict(
            list
        )
        for event in self.coverage_events:
            grouped[
                (str(event["pool"]), float(event["tau"]), int(event["prefix_T"]))
            ].append(event)
        coverage_rows = []
        for (pool, threshold, horizon), events in sorted(grouped.items()):
            complete = sum(bool(event["candidate_complete"]) for event in events)
            failures = sum(bool(event["published_failure"]) for event in events)
            complete_failures = sum(
                bool(event["diag_complete_candidate_failure"]) for event in events
            )
            coverage_rows.append(
                {
                    "population_id": "development",
                    "data_role": "DEV-CAL",
                    "pool": pool,
                    "tau": threshold,
                    "T": horizon,
                    "event_count": len(events),
                    "candidate_complete_count": complete,
                    "candidate_complete_fraction": complete / len(events),
                    "published_failure_count": failures,
                    "diag_complete_candidate_failure_count": complete_failures,
                    "diag_complete_candidate_failure_fraction": (
                        complete_failures / failures if failures else None
                    ),
                    "reference_count": len(
                        {str(event["reference_id"]) for event in events}
                    ),
                    "ambiguity_status": "AMBIGUITY_METADATA_UNAVAILABLE",
                }
            )
        primary = [
            event
            for event in self.coverage_events
            if event["pool"] == "POLICY_POOL" and event["tau"] == 0.5
        ]
        published_failures = [event for event in primary if event["published_failure"]]
        complete_failures = [
            event for event in primary if event["diag_complete_candidate_failure"]
        ]
        relevant = complete_failures if complete_failures else published_failures
        long_gain = (
            sum(
                measured[("GT-ID-LAG1", horizon)]["t_mAP"]
                - measured[("D0", horizon)]["t_mAP"]
                for horizon in (4, 5)
            )
            / 2.0
        )
        gate = evaluate_headroom_gate(
            long_gain=long_gain,
            candidate_complete_failure_count=len(complete_failures),
            published_failure_count=len(published_failures),
            event_reference_count=len(
                {str(event["reference_id"]) for event in relevant}
            ),
            diagnostic_status="PASS",
        )
        gate.update(
            {
                "data_role": "DEV-CAL",
                "ambiguity_metadata_status": "AMBIGUITY_METADATA_UNAVAILABLE",
                "logical_unit_count": self.unit_count,
                "reference_count": len(self.reference_ids),
            }
        )
        return {
            "metric_rows": metric_rows,
            "coverage_rows": coverage_rows,
            "coverage_events": self.coverage_events,
            "assignment_events": self.assignment_events,
            "gate": gate,
            "status": "PASS",
        }


__all__ = [
    "CrossWindowDiagnosticError",
    "DiagnosticCandidate",
    "E1DiagnosticAccumulator",
    "E1SequenceResult",
    "build_candidate_pools",
    "diagnose_candidate_coverage",
    "evaluate_headroom_gate",
    "gt_lag1_prefixes",
    "gt_relaxed_prediction",
    "rank_family_candidates",
]
