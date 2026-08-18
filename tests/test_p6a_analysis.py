from __future__ import annotations

import math
from dataclasses import replace

import pytest

from scripts.p6a_analysis import (
    AssociationEvent,
    CapacitySnapshot,
    EfficiencyRecord,
    PairedMetricRecord,
    aggregate_event_metrics,
    aggregate_reactivation_metrics,
    audit_capacity,
    classify_failure,
    evaluate_gates,
    paired_cluster_bootstrap,
    persistent_state_bytes,
    validate_association_events,
    validate_efficiency_rows,
)


def _event(**overrides: object) -> AssociationEvent:
    values: dict[str, object] = {
        "scene_id": "scene-a",
        "sequence_id": "master-a",
        "reference_scene_id": "ref-a",
        "master_sequence_id": "master-a",
        "prefix": 3,
        "method": "Persist4D",
        "stage_id": 0,
        "event_kind": "prediction",
        "query_id": "q0",
        "candidate_slot_id": 10,
        "predicted_identity_id": 10,
        "gt_entity_id": 7,
        "gt_present": True,
        "prediction_present": True,
        "association_correct": True,
        "association_result": "active_correct",
        "transition_opportunity": False,
        "id_switch": False,
        "gap_opportunity": False,
        "reactivation_attempt": False,
        "reactivation_correct": None,
        "new_birth": False,
        "false_birth": False,
    }
    values.update(overrides)
    return AssociationEvent(**values)


def test_typed_event_rows_reject_nan_and_sentinel_ids() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_association_events([_event(total_score=math.nan)])

    with pytest.raises(ValueError, match="sentinel"):
        validate_association_events([_event(candidate_slot_id=-1)])


def test_event_table_reconstructs_identity_and_reactivation_aggregates() -> None:
    rows = [
        _event(stage_id=0, query_id="q0", association_result="birth", new_birth=True),
        _event(
            stage_id=1,
            query_id="q1",
            transition_opportunity=True,
            id_switch=False,
            association_result="active_correct",
        ),
        _event(
            stage_id=2,
            query_id=None,
            candidate_slot_id=None,
            predicted_identity_id=None,
            prediction_present=False,
            association_correct=None,
            association_result="no_attempt",
            gap_opportunity=False,
            event_kind="gt_miss",
        ),
        _event(
            stage_id=3,
            query_id="q3",
            candidate_slot_id=11,
            predicted_identity_id=11,
            association_correct=False,
            association_result="reactivation_wrong",
            transition_opportunity=True,
            id_switch=True,
            gap_opportunity=True,
            reactivation_attempt=True,
            reactivation_correct=False,
            reactivation=True,
        ),
        _event(
            stage_id=1,
            query_id="q2",
            candidate_slot_id=12,
            predicted_identity_id=12,
            gt_entity_id=8,
            association_correct=False,
            association_result="birth",
            new_birth=True,
            false_birth=True,
        ),
    ]
    identity, reactivation = aggregate_event_metrics(rows)

    assert identity["transition_opportunities"] == 2
    assert identity["id_switches"] == 1
    assert identity["id_switch_rate"] == 0.5
    assert identity["births"] == 2
    assert identity["false_births"] == 1
    assert reactivation == {
        "gap_opportunities": 1,
        "reactivation_attempts": 1,
        "correct_reactivations": 0,
        "wrong_reactivations": 1,
        "no_attempts": 0,
        "reactivation_accuracy": 0.0,
        "reactivation_precision": 0.0,
        "reactivation_recall": 0.0,
        "reactivation_coverage": 1.0,
    }


def test_reactivation_no_attempt_is_explicit_and_zero_denominators_are_none() -> None:
    row = _event(
        stage_id=4,
        query_id=None,
        candidate_slot_id=None,
        predicted_identity_id=None,
        prediction_present=False,
        association_correct=None,
        association_result="no_attempt",
        gap_opportunity=True,
        reactivation_attempt=False,
        reactivation_correct=None,
        event_kind="gt_miss",
    )
    result = aggregate_reactivation_metrics([row])

    assert result["gap_opportunities"] == 1
    assert result["reactivation_attempts"] == 0
    assert result["no_attempts"] == 1
    assert result["reactivation_accuracy"] is None
    assert result["reactivation_precision"] is None
    assert result["reactivation_recall"] == 0.0
    assert result["reactivation_coverage"] == 0.0


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ({"local_observation_available": False}, "F1"),
        ({"association_miss": True}, "F2"),
        ({"identity_fragmentation": True}, "F3"),
        ({"identity_merge": True}, "F4"),
        ({"reactivation": True, "reactivation_correct": False}, "F5"),
        ({"semantic_drift": True}, "F6"),
        ({"capacity_failure": True}, "F7"),
        ({}, "unclassified"),
    ],
)
def test_failure_categories_are_executable_and_mutually_exclusive(
    evidence: dict[str, object], expected: str
) -> None:
    assert classify_failure(_event(association_correct=False, **evidence)) == expected


def test_failure_priority_assigns_one_primary_when_flags_repeat() -> None:
    event = _event(
        association_correct=False,
        local_observation_available=False,
        association_miss=True,
        identity_fragmentation=True,
        identity_merge=True,
        reactivation=True,
        reactivation_correct=False,
        semantic_drift=True,
        capacity_failure=True,
    )
    assert classify_failure(event) == "F1"


def test_capacity_audit_enforces_state_invariants_and_formula() -> None:
    assert persistent_state_bytes(100, feature_dim=128, class_count=26) == 63_808
    report = audit_capacity(
        [
            CapacitySnapshot(
                method="Persist4D",
                horizon=3,
                stage_id=0,
                capacity=4,
                birth_count=2,
                occupied_count=2,
                active_count=2,
                dormant_count=0,
                rejected_births=0,
            ),
            CapacitySnapshot(
                method="Persist4D",
                horizon=3,
                stage_id=1,
                capacity=4,
                birth_count=1,
                occupied_count=3,
                active_count=2,
                dormant_count=1,
                rejected_births=0,
            ),
        ]
    )
    assert report["peak_occupied"] == 3
    assert report["peak_active"] == 2
    assert report["peak_dormant"] == 1
    assert report["occupancy_ratio"] == 0.75

    with pytest.raises(ValueError, match="active.*occupied"):
        audit_capacity(
            [
                CapacitySnapshot(
                    method="Persist4D",
                    horizon=2,
                    stage_id=0,
                    capacity=2,
                    occupied_count=0,
                    active_count=1,
                )
            ]
        )


def test_efficiency_rows_keep_bootstrap_and_new_visit_types_disjoint() -> None:
    rows = validate_efficiency_rows(
        [
            EfficiencyRecord(
                method="Persist4D",
                horizon=3,
                stage_id=0,
                row_type="bootstrap",
                bootstrap_latency_ms=2.0,
                gpu_peak_memory_bytes=100,
                persistent_state_bytes=20,
            ),
            EfficiencyRecord(
                method="Persist4D",
                horizon=3,
                stage_id=1,
                row_type="new_visit",
                new_visit_latency_ms=1.0,
                association_overhead_ms=0.2,
                memory_update_overhead_ms=0.1,
                gpu_peak_memory_bytes=110,
                persistent_state_bytes=20,
            ),
        ]
    )
    assert [row.row_type for row in rows] == ["bootstrap", "new_visit"]

    with pytest.raises(ValueError, match="bootstrap.*new_visit"):
        validate_efficiency_rows(
            [
                EfficiencyRecord(
                    method="Persist4D",
                    horizon=3,
                    stage_id=0,
                    row_type="bootstrap",
                    bootstrap_latency_ms=2.0,
                    new_visit_latency_ms=1.0,
                )
            ]
        )


def _paired_rows() -> list[PairedMetricRecord]:
    rows: list[PairedMetricRecord] = []
    for index, reference in enumerate(("ref-a", "ref-b", "ref-c")):
        common = {
            "reference_scene_id": reference,
            "master_sequence_id": f"master-{index}",
            "prefix": 4,
            "prediction_digest": "digest-4",
            "metric": "id_sw_rate",
        }
        rows.extend(
            [
                PairedMetricRecord(method="Persist4D", value=0.08 + index * 0.01, **common),
                PairedMetricRecord(method="EMA", value=0.10 + index * 0.01, **common),
            ]
        )
    return rows


def test_paired_cluster_bootstrap_is_seeded_and_reference_scene_clustered() -> None:
    rows = _paired_rows()
    first = paired_cluster_bootstrap(rows, method="Persist4D", baseline_method="EMA")
    second = paired_cluster_bootstrap(rows, method="Persist4D", baseline_method="EMA")

    assert first == second
    assert first["cluster_field"] == "reference_scene_id"
    assert first["n_bootstrap"] == 10_000
    assert first["seed"] == 45
    assert first["mean_delta"] == pytest.approx(-0.02)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cross-cache"])
def test_paired_bootstrap_rejects_incomplete_duplicate_or_cross_cache_pairs(
    mutation: str,
) -> None:
    rows = _paired_rows()
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[-1] = replace(rows[-1], prediction_digest="other-digest")

    with pytest.raises(ValueError, match="pair|cache|duplicate|missing"):
        paired_cluster_bootstrap(rows, method="Persist4D", baseline_method="EMA")


def test_window_bootstrap_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference_scene"):
        paired_cluster_bootstrap(
            _paired_rows(),
            method="Persist4D",
            baseline_method="EMA",
            cluster_by="sequence_id",
        )


def _gate_input() -> dict[str, object]:
    return {
        "paired_idsw": {
            4: {"relative_reduction": 0.20, "ci_high": 0.0},
            5: {"relative_reduction": 0.20, "ci_high": 0.0},
        },
        "reactivation": {
            "Persist4D": {
                3: {"accuracy": 0.70, "recall": 0.25},
                4: {"accuracy": 0.70, "recall": 0.25},
                5: {"accuracy": 0.70, "recall": 0.25},
            },
            "EMA": {
                3: {"accuracy": 0.60, "recall": 0.20},
                4: {"accuracy": 0.60, "recall": 0.20},
                5: {"accuracy": 0.60, "recall": 0.20},
            },
        },
        "raw_prediction_fingerprints": {
            "Persist4D": "same-digest",
            "EMA": "same-digest",
        },
        "raw_local_ap": {"Persist4D": [0.5, 0.6], "EMA": [0.5, 0.6]},
        "online_task": {
            "Persist4D": {
                2: {"t_mAP": 0.95, "t_REC": 0.95},
                4: {"t_mAP": 0.81, "t_REC": 0.81},
                5: {"t_mAP": 0.81, "t_REC": 0.81},
            },
            "EMA": {
                2: {"t_mAP": 1.0, "t_REC": 1.0},
                4: {"t_mAP": 0.80, "t_REC": 0.80},
                5: {"t_mAP": 0.80, "t_REC": 0.80},
            },
        },
        "failure_counts": {"F1": 9, "unclassified": 1},
    }


def test_preregistered_gates_accept_exact_boundaries() -> None:
    result = evaluate_gates(_gate_input())

    assert result["G6A-1"]["passed"] is True
    assert result["G6A-2"]["passed"] is True
    assert result["G6A-3"]["passed"] is True
    assert result["G6A-4"]["passed"] is True
    assert result["G6A-5"]["passed"] is True
    assert result["overall_passed"] is True


def test_gate_boundary_failures_are_not_rounded_up() -> None:
    values = _gate_input()
    values["paired_idsw"] = {
        4: {"relative_reduction": 0.199999999, "ci_high": 0.0},
        5: {"relative_reduction": 0.20, "ci_high": 0.0},
    }
    result = evaluate_gates(values)
    assert result["G6A-1"]["passed"] is False
