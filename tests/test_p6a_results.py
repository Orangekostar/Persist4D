from __future__ import annotations

import math

import pytest

from scripts.p6a_analysis import (
    AssociationEvent,
    CapacitySnapshot,
    persistent_state_bytes,
)
from scripts.p6a_artifacts import CSV_COLUMN_SCHEMAS
from scripts.p6a_results import (
    association_event_rows,
    capacity_audit_rows,
    failure_breakdown_rows,
    per_sequence_result_rows,
    reactivation_audit_rows,
    reactivation_by_gap_rows,
    reactivation_distribution_rows,
)


def _event(index: int = 0, **overrides: object) -> AssociationEvent:
    values: dict[str, object] = {
        "event_id": f"event-{index}",
        "scene_id": "scene-a",
        "sequence_id": "sequence-a",
        "reference_scene_id": "reference-a",
        "master_sequence_id": "master-a",
        "order_id": "canonical",
        "prefix": 3,
        "method": "B1",
        "stage_id": index,
        "event_kind": "prediction",
        "query_id": f"query-{index}",
        "candidate_slot_id": index,
        "predicted_identity_id": 100 + index,
        "gt_entity_id": 200 + index,
        "association_correct": True,
        "association_result": "active_correct",
        "gt_present": True,
        "prediction_present": True,
        "gap_opportunity": False,
        "reactivation_attempt": False,
        "reactivation": False,
        "reactivation_correct": None,
        "is_failure": False,
        "prediction_digest": "a" * 64,
        "cache_digest": "a" * 64,
    }
    values.update(overrides)
    return AssociationEvent(**values)


def _reactivation_event(
    index: int,
    *,
    method: str = "B1",
    prefix: int = 3,
    correct: bool = True,
    gap_length: int = 1,
    best_score: float = 0.5,
    score_margin: float = 0.2,
) -> AssociationEvent:
    return _event(
        index,
        method=method,
        prefix=prefix,
        stage_id=index,
        gt_entity_id=200,
        predicted_identity_id=100,
        gap_opportunity=True,
        reactivation_attempt=True,
        reactivation=True,
        reactivation_correct=correct,
        wrong_reactivation=not correct,
        association_correct=correct,
        association_result="reactivation_correct" if correct else "reactivation_wrong",
        is_failure=not correct,
        gap_length=gap_length,
        best_score=best_score,
        score_margin=score_margin,
    )


def test_association_rows_use_exact_schema_order_after_validation() -> None:
    row = association_event_rows([_event(0)])[0]

    assert tuple(row) == CSV_COLUMN_SCHEMAS["association_events.csv"]
    assert row["event_id"] == "event-0"
    assert row["prefix"] == 3


def test_per_sequence_rows_add_prefix_horizon_without_coercing_none() -> None:
    row = per_sequence_result_rows([_event(0, prefix=2, stage_id=0)])[0]

    assert tuple(row) == CSV_COLUMN_SCHEMAS["per_sequence_results.csv"]
    assert row["T"] == row["prefix"] == 2
    assert row["id_switch_rate"] is None
    assert row["reactivation_accuracy"] is None


def test_failure_rows_cover_full_method_horizon_category_grid() -> None:
    rows = failure_breakdown_rows(
        [
            _event(
                0,
                method="B0",
                prefix=2,
                association_correct=False,
                association_miss=True,
                is_failure=True,
            )
        ]
    )

    assert len(rows) == 6 * 4 * 8
    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["error_breakdown.csv"]
    group = [row for row in rows if row["method"] == "B0" and row["T"] == 2]
    assert {row["category"] for row in group} == {
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "unclassified",
    }
    assert next(row for row in group if row["category"] == "F2")["share"] == 1.0
    empty = [row for row in rows if row["method"] == "B4" and row["T"] == 5]
    assert all(row["share"] == 0.0 for row in empty)


def test_reactivation_audit_keeps_none_ratios_for_zero_opportunity_group() -> None:
    rows = reactivation_audit_rows(
        [
            _reactivation_event(0, method="B1", prefix=3, correct=True),
            _event(1, method="B1", prefix=4, stage_id=0),
        ]
    )

    assert len(rows) == 12
    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["reactivation_audit.csv"]
    active = next(row for row in rows if (row["method"], row["T"]) == ("B1", 3))
    assert active["correct_reactivations"] == 1
    assert active["reactivation_accuracy"] == 1.0
    zero = next(row for row in rows if (row["method"], row["T"]) == ("B1", 4))
    assert zero["gap_opportunities"] == 0
    assert zero["reactivation_accuracy"] is None
    assert zero["reactivation_precision"] is None


def test_reactivation_distribution_bins_edges_and_emits_empty_outcomes() -> None:
    rows = reactivation_distribution_rows(
        [
            _reactivation_event(0, correct=True, best_score=0.5),
            _reactivation_event(1, correct=False, best_score=0.75),
        ],
        field="best_score",
        edges=(0.0, 0.5, 1.0),
    )

    assert len(rows) == 4 * 3 * 2 * 2
    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["reactivation_score_distribution.csv"]
    correct_middle = next(
        row
        for row in rows
        if row["method"] == "B1"
        and row["T"] == 3
        and row["outcome"] == "correct"
        and row["bin_low"] == 0.5
    )
    wrong_middle = next(
        row
        for row in rows
        if row["method"] == "B1"
        and row["T"] == 3
        and row["outcome"] == "wrong"
        and row["bin_low"] == 0.5
    )
    assert correct_middle["count"] == 1
    assert correct_middle["fraction"] == 1.0
    assert wrong_middle["count"] == 1
    assert wrong_middle["fraction"] == 1.0
    assert all(
        row["fraction"] == 0.0
        for row in rows
        if row["method"] == "B4" and row["T"] == 5
    )


def test_reactivation_distribution_rejects_invalid_field_edges_and_values() -> None:
    with pytest.raises(ValueError, match="field"):
        reactivation_distribution_rows(
            [_reactivation_event(0)], field="total_score", edges=(0.0, 1.0)
        )
    with pytest.raises(ValueError, match="increasing"):
        reactivation_distribution_rows(
            [_reactivation_event(0)], field="best_score", edges=(0.0, 0.0, 1.0)
        )
    with pytest.raises(ValueError, match="cover"):
        reactivation_distribution_rows(
            [_reactivation_event(0, best_score=2.0)],
            field="best_score",
            edges=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="finite"):
        reactivation_distribution_rows(
            [_reactivation_event(0, best_score=math.inf)],
            field="best_score",
            edges=(0.0, 1.0),
        )


def test_reactivation_by_gap_normalizes_each_gap_and_uses_none_for_empty_group() -> None:
    rows = reactivation_by_gap_rows(
        [
            _reactivation_event(0, method="B2", prefix=4, correct=True, gap_length=1),
            _reactivation_event(1, method="B2", prefix=4, correct=False, gap_length=2),
        ]
    )

    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["reactivation_by_gap.csv"]
    gap_one = [
        row
        for row in rows
        if row["method"] == "B2" and row["T"] == 4 and row["gap_length"] == 1
    ]
    assert {row["outcome"] for row in gap_one} == {"correct", "wrong"}
    assert next(row for row in gap_one if row["outcome"] == "correct")["fraction"] == 1.0
    gap_two_wrong = next(
        row
        for row in rows
        if row["method"] == "B2"
        and row["T"] == 4
        and row["gap_length"] == 2
        and row["outcome"] == "wrong"
    )
    assert gap_two_wrong["count"] == 1
    empty = [
        row
        for row in rows
        if row["method"] == "B4" and row["T"] == 5
    ]
    assert {(row["gap_length"], row["outcome"]) for row in empty} == {
        (None, "correct"),
        (None, "wrong"),
    }


def _capacity_snapshot(
    horizon: int,
    stage: int,
    *,
    capacity: int = 4,
    method: str = "B4",
    occupied: int | None = None,
) -> CapacitySnapshot:
    occupied = min(stage + 1, capacity) if occupied is None else occupied
    active = max(0, occupied - 1)
    return CapacitySnapshot(
        method=method,
        horizon=horizon,
        stage_id=stage,
        capacity=capacity,
        birth_count=stage + 1,
        occupied_count=occupied,
        active_count=active,
        dormant_count=occupied - active,
        rejected_births=stage,
        persistent_state_bytes=persistent_state_bytes(capacity, 2, 3),
        feature_dim=2,
        class_count=3,
    )


def test_capacity_rows_aggregate_complete_grid_and_repeat_horizon_peaks() -> None:
    snapshots = [
        _capacity_snapshot(horizon, stage)
        for horizon in range(2, 6)
        for stage in range(horizon)
    ]
    snapshots.extend(
        [
            _capacity_snapshot(3, 1, occupied=2),
            _capacity_snapshot(3, 1, occupied=3),
        ]
    )

    rows = capacity_audit_rows(snapshots)

    assert len(rows) == 14
    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["capacity_audit.csv"]
    selected = next(row for row in rows if (row["T"], row["stage_id"]) == (3, 1))
    assert selected["birth_count"] == 6
    assert selected["rejected_births"] == 3
    assert selected["occupied_count"] == 3
    assert selected["active_count"] == 2
    assert selected["dormant_count"] == 1
    assert selected["peak_occupied"] == 3
    assert selected["peak_active"] == 2
    assert selected["occupancy_ratio"] == 0.75


def test_capacity_rows_reject_method_missing_or_extra_stage_and_mixed_capacity() -> None:
    complete = [
        _capacity_snapshot(horizon, stage)
        for horizon in range(2, 6)
        for stage in range(horizon)
    ]
    with pytest.raises(ValueError, match="B4"):
        capacity_audit_rows(
            [_capacity_snapshot(2, 0, method="B3")] + complete[1:]
        )
    with pytest.raises(ValueError, match="cover|group|stage"):
        capacity_audit_rows(complete[:-1])
    with pytest.raises(ValueError, match="capacity"):
        capacity_audit_rows(complete + [_capacity_snapshot(2, 0, capacity=5)])
