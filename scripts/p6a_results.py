"""Deterministic row builders for the P6-A diagnostic CSV tables."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from numbers import Integral, Real

from scripts.p6a_analysis import (
    AssociationEvent,
    CapacitySnapshot,
    aggregate_metrics_by_sequence,
    aggregate_reactivation_metrics,
    failure_breakdown,
    validate_association_events,
)
from scripts.p6a_artifacts import CSV_COLUMN_SCHEMAS

__all__ = (
    "association_event_rows",
    "capacity_audit_rows",
    "failure_breakdown_rows",
    "per_sequence_result_rows",
    "reactivation_audit_rows",
    "reactivation_by_gap_rows",
    "reactivation_distribution_rows",
)

ONLINE_METHODS = ("B0", "B0_sanity", "B1", "B2", "B3", "B4")
HORIZONS = (2, 3, 4, 5)
REACTIVATION_METHODS = ("B1", "B2", "B3", "B4")
REACTIVATION_HORIZONS = (3, 4, 5)
FAILURE_CATEGORIES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "unclassified")
OUTCOMES = ("correct", "wrong")

_AssociationInput = AssociationEvent | Mapping[str, object]
_CapacityInput = CapacitySnapshot | Mapping[str, object]
_Row = dict[str, object]


def _ordered_row(values: Mapping[str, object], schema_path: str) -> _Row:
    schema = CSV_COLUMN_SCHEMAS[schema_path]
    return {field: values[field] for field in schema}


def _validate_result_scope(events: Sequence[AssociationEvent]) -> None:
    for event in events:
        if event.method not in ONLINE_METHODS:
            raise ValueError(
                f"P6-A result rows support only B0, B0_sanity, and B1-B4; got {event.method!r}"
            )
        if event.prefix not in HORIZONS:
            raise ValueError("P6-A result rows require horizon prefixes 2 through 5")


def _validated_events(
    events: Iterable[_AssociationInput],
) -> tuple[AssociationEvent, ...]:
    validated = validate_association_events(events)
    _validate_result_scope(validated)
    return validated


def _empty_reactivation_metrics() -> dict[str, object]:
    return {
        "gap_opportunities": 0,
        "reactivation_attempts": 0,
        "predicted_reactivation_events": 0,
        "correct_reactivations": 0,
        "wrong_reactivations": 0,
        "no_attempts": 0,
        "reactivation_accuracy": None,
        "reactivation_precision": None,
        "reactivation_recall": None,
        "reactivation_coverage": None,
    }


def _reactivation_groups(
    events: Sequence[AssociationEvent],
) -> dict[tuple[str, int], list[AssociationEvent]]:
    groups: dict[tuple[str, int], list[AssociationEvent]] = defaultdict(list)
    for event in events:
        key = (event.method, event.prefix)
        if key in {
            (method, horizon)
            for method in REACTIVATION_METHODS
            for horizon in REACTIVATION_HORIZONS
        }:
            groups[key].append(event)
    return groups


def association_event_rows(
    events: Iterable[_AssociationInput],
) -> tuple[_Row, ...]:
    """Validate events and emit rows in the registered association schema order."""

    validated = _validated_events(events)
    return tuple(
        _ordered_row(event.as_dict(), "association_events.csv")
        for event in validated
    )


def per_sequence_result_rows(
    events: Iterable[_AssociationInput],
) -> tuple[_Row, ...]:
    """Emit one aggregate row per method/sequence/prefix scope."""

    aggregates = aggregate_metrics_by_sequence(events)
    for aggregate in aggregates:
        if aggregate["method"] not in ONLINE_METHODS:
            raise ValueError(
                f"P6-A result rows support only B0, B0_sanity, and B1-B4; got {aggregate['method']!r}"
            )
        prefix = aggregate["prefix"]
        if prefix not in HORIZONS:
            raise ValueError("P6-A result rows require horizon prefixes 2 through 5")
    return tuple(
        _ordered_row({**aggregate, "T": aggregate["prefix"]}, "per_sequence_results.csv")
        for aggregate in aggregates
    )


def failure_breakdown_rows(
    events: Iterable[_AssociationInput],
) -> tuple[_Row, ...]:
    """Emit all F1-F7/unclassified rows for every online method and horizon."""

    validated = _validated_events(events)
    groups: dict[tuple[str, int], list[AssociationEvent]] = defaultdict(list)
    for event in validated:
        groups[(event.method, event.prefix)].append(event)

    rows: list[_Row] = []
    for method in ONLINE_METHODS:
        for horizon in HORIZONS:
            group = groups.get((method, horizon), [])
            report = failure_breakdown(group) if group else None
            counts = (
                report["counts"]
                if report is not None
                else {category: 0 for category in FAILURE_CATEGORIES}
            )
            total = int(report["total_failures"]) if report is not None else 0
            for category in FAILURE_CATEGORIES:
                count = int(counts[category])
                rows.append(
                    _ordered_row(
                        {
                            "method": method,
                            "T": horizon,
                            "category": category,
                            "count": count,
                            "share": count / total if total else 0.0,
                        },
                        "error_breakdown.csv",
                    )
                )
    return tuple(rows)


def reactivation_audit_rows(
    events: Iterable[_AssociationInput],
) -> tuple[_Row, ...]:
    """Emit the complete B1-B4/T3-T5 reactivation audit grid."""

    validated = _validated_events(events)
    groups = _reactivation_groups(validated)
    rows: list[_Row] = []
    schema_path = "reactivation_audit.csv"
    for method in REACTIVATION_METHODS:
        for horizon in REACTIVATION_HORIZONS:
            group = groups.get((method, horizon), [])
            metrics = aggregate_reactivation_metrics(group) if group else _empty_reactivation_metrics()
            rows.append(
                _ordered_row(
                    {"method": method, "T": horizon, **metrics},
                    schema_path,
                )
            )
    return tuple(rows)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_edges(edges: Iterable[object]) -> tuple[float, ...]:
    try:
        values = tuple(_finite_real(edge, name="edges") for edge in edges)
    except TypeError as error:
        raise ValueError("edges must be a finite increasing sequence") from error
    if len(values) < 2:
        raise ValueError("edges must contain at least two values")
    if any(left >= right for left, right in pairwise(values)):
        raise ValueError("edges must be strictly increasing")
    return values


def _reactivation_outcome(event: AssociationEvent) -> str:
    if event.reactivation_correct is True:
        return "correct"
    if event.reactivation_correct is False:
        return "wrong"
    raise ValueError("reactivation events require explicit correctness")


def _reactivation_values(
    events: Sequence[AssociationEvent], field: str
) -> dict[tuple[str, int, str], list[float]]:
    values: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    groups = _reactivation_groups(events)
    for (method, horizon), group in groups.items():
        for event in group:
            if event.reactivation is not True:
                continue
            outcome = _reactivation_outcome(event)
            value = _finite_real(getattr(event, field), name=field)
            values[(method, horizon, outcome)].append(value)
    return values


def _bin_index(value: float, edges: Sequence[float]) -> int:
    for index, (low, high) in enumerate(pairwise(edges)):
        if low <= value < high or (index == len(edges) - 2 and value <= high):
            return index
    raise ValueError("reactivation value falls outside bin edges")


def reactivation_distribution_rows(
    events: Iterable[_AssociationInput],
    *,
    field: str,
    edges: Iterable[object],
) -> tuple[_Row, ...]:
    """Bin reactivation scores or margins over the complete diagnostic grid."""

    if field not in {"best_score", "score_margin"}:
        raise ValueError("field must be best_score or score_margin")
    validated = _validated_events(events)
    bin_edges = _validated_edges(edges)
    values = _reactivation_values(validated, field)
    observed = [value for group in values.values() for value in group]
    if observed and (min(observed) < bin_edges[0] or max(observed) > bin_edges[-1]):
        raise ValueError("bin edges must cover every reactivation value")

    schema_path = (
        "reactivation_score_distribution.csv"
        if field == "best_score"
        else "reactivation_margin_distribution.csv"
    )
    rows: list[_Row] = []
    for method in REACTIVATION_METHODS:
        for horizon in REACTIVATION_HORIZONS:
            for outcome in OUTCOMES:
                counts = [0] * (len(bin_edges) - 1)
                for value in values.get((method, horizon, outcome), []):
                    counts[_bin_index(value, bin_edges)] += 1
                total = sum(counts)
                for index, count in enumerate(counts):
                    rows.append(
                        _ordered_row(
                            {
                                "method": method,
                                "T": horizon,
                                "outcome": outcome,
                                "bin_low": bin_edges[index],
                                "bin_high": bin_edges[index + 1],
                                "count": count,
                                "fraction": count / total if total else 0.0,
                            },
                            schema_path,
                        )
                    )
    return tuple(rows)


def reactivation_by_gap_rows(
    events: Iterable[_AssociationInput],
) -> tuple[_Row, ...]:
    """Emit paired correct/wrong reactivation counts for each observed gap."""

    validated = _validated_events(events)
    groups = _reactivation_groups(validated)
    rows: list[_Row] = []
    for method in REACTIVATION_METHODS:
        for horizon in REACTIVATION_HORIZONS:
            outcome_gaps: dict[int, dict[str, int]] = defaultdict(
                lambda: {outcome: 0 for outcome in OUTCOMES}
            )
            for event in groups.get((method, horizon), []):
                if event.reactivation is not True:
                    continue
                if event.gap_length is None:
                    raise ValueError(
                        "reactivation events require a non-negative gap_length"
                    )
                if isinstance(event.gap_length, bool) or not isinstance(
                    event.gap_length, Integral
                ):
                    raise ValueError(  # noqa: TRY004
                        "gap_length must be a non-negative integer"
                    )
                outcome_gaps[int(event.gap_length)][_reactivation_outcome(event)] += 1

            if not outcome_gaps:
                output_gaps: Sequence[int | None] = (None,)
            else:
                output_gaps = tuple(sorted(outcome_gaps))
            for gap_length in output_gaps:
                counts = (
                    outcome_gaps[gap_length]
                    if gap_length is not None
                    else {outcome: 0 for outcome in OUTCOMES}
                )
                total = sum(counts.values())
                for outcome in OUTCOMES:
                    rows.append(
                        _ordered_row(
                            {
                                "method": method,
                                "T": horizon,
                                "gap_length": gap_length,
                                "outcome": outcome,
                                "count": counts[outcome],
                                "fraction": counts[outcome] / total if total else 0.0,
                            },
                            "reactivation_by_gap.csv",
                        )
                    )
    return tuple(rows)


def _coerce_capacity_snapshot(raw: _CapacityInput) -> CapacitySnapshot:
    snapshot = raw if isinstance(raw, CapacitySnapshot) else CapacitySnapshot.from_mapping(raw)
    snapshot.validate()
    return snapshot


def capacity_audit_rows(
    snapshots: Iterable[_CapacityInput],
) -> tuple[_Row, ...]:
    """Aggregate the complete causal B4 capacity audit grid."""

    validated = tuple(_coerce_capacity_snapshot(raw) for raw in snapshots)
    if not validated:
        raise ValueError("capacity audit requires at least one snapshot")
    for snapshot in validated:
        if snapshot.method != "B4":
            raise ValueError("capacity audit supports only B4")
        if snapshot.horizon not in HORIZONS:
            raise ValueError("capacity audit requires horizons 2 through 5")
        if snapshot.stage_id >= snapshot.horizon:
            raise ValueError("capacity audit stage_id must be less than horizon")

    capacities = {snapshot.capacity for snapshot in validated}
    if len(capacities) != 1:
        raise ValueError("capacity must be constant across snapshots")

    expected = {(horizon, stage) for horizon in HORIZONS for stage in range(horizon)}
    groups: dict[tuple[int, int], list[CapacitySnapshot]] = defaultdict(list)
    for snapshot in validated:
        groups[(snapshot.horizon, snapshot.stage_id)].append(snapshot)
    actual = set(groups)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"capacity audit must cover every horizon/stage group; missing={missing}, extra={extra}"
        )

    horizon_peaks: dict[int, tuple[int, int, int]] = {}
    for horizon in HORIZONS:
        horizon_snapshots = [snapshot for snapshot in validated if snapshot.horizon == horizon]
        horizon_peaks[horizon] = (
            max(snapshot.occupied_count for snapshot in horizon_snapshots),
            max(snapshot.active_count for snapshot in horizon_snapshots),
            max(
                snapshot.dormant_count
                if snapshot.dormant_count is not None
                else snapshot.occupied_count - snapshot.active_count
                for snapshot in horizon_snapshots
            ),
        )

    rows: list[_Row] = []
    for horizon in HORIZONS:
        peak_occupied, peak_active, peak_dormant = horizon_peaks[horizon]
        for stage in range(horizon):
            group = groups[(horizon, stage)]
            selected = max(
                group,
                key=lambda snapshot: (
                    max(snapshot.occupied_count, snapshot.active_count),
                    snapshot.occupied_count,
                    snapshot.active_count,
                ),
            )
            state_sizes = [
                snapshot.persistent_state_bytes
                for snapshot in group
                if snapshot.persistent_state_bytes is not None
            ]
            dormant = (
                selected.dormant_count
                if selected.dormant_count is not None
                else selected.occupied_count - selected.active_count
            )
            rows.append(
                _ordered_row(
                    {
                        "method": "B4",
                        "T": horizon,
                        "stage_id": stage,
                        "capacity": selected.capacity,
                        "birth_count": sum(snapshot.birth_count for snapshot in group),
                        "occupied_count": selected.occupied_count,
                        "active_count": selected.active_count,
                        "dormant_count": dormant,
                        "peak_occupied": peak_occupied,
                        "peak_active": peak_active,
                        "peak_dormant": peak_dormant,
                        "occupancy_ratio": selected.occupied_count / selected.capacity,
                        "rejected_births": sum(
                            snapshot.rejected_births for snapshot in group
                        ),
                        "persistent_state_bytes": max(state_sizes)
                        if state_sizes
                        else None,
                    },
                    "capacity_audit.csv",
                )
            )
    return tuple(rows)
