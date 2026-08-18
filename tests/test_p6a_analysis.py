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
    aggregate_metrics_by_sequence,
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
        "event_id": "event-placeholder",
        "scene_id": "scene-a",
        "sequence_id": "master-a",
        "reference_scene_id": "ref-a",
        "master_sequence_id": "master-a",
        "order_id": "canonical",
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
        "reactivation": False,
        "new_birth": False,
        "false_birth": False,
        "is_failure": False,
        "prediction_digest": "cache-a",
        "cache_digest": "cache-a",
    }
    values.update(overrides)
    if "event_id" not in overrides:
        identity = (
            values["query_id"]
            if values["event_kind"] == "prediction"
            else values["gt_entity_id"]
        )
        values["event_id"] = (
            ":".join(
                str(values[key])
                for key in ("method", "order_id", "prefix", "stage_id", "event_kind")
            )
            + f":{identity}"
        )
    return AssociationEvent(**values)


def test_typed_event_rows_reject_nan_and_sentinel_ids() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_association_events([_event(total_score=math.nan)])

    with pytest.raises(ValueError, match="sentinel"):
        validate_association_events([_event(candidate_slot_id=-1)])


def test_event_mapping_rejects_unknown_fields_and_cache_mismatch() -> None:
    row = _event().as_dict()
    row["typo_score"] = 1.0
    with pytest.raises(ValueError, match="unknown association event fields"):
        validate_association_events([row])

    with pytest.raises(ValueError, match="digest"):
        validate_association_events([_event(cache_digest="cache-b")])


def test_event_table_cannot_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_association_events([])


def test_event_identity_is_unique_but_decisions_are_method_and_order_scoped() -> None:
    first = _event(event_id="one")
    second_method = _event(event_id="two", method="EMA")
    second_order = _event(event_id="three", order_id="reverse")
    assert len(validate_association_events([first, second_method, second_order])) == 3

    with pytest.raises(ValueError, match="event_id"):
        validate_association_events([first, replace(first, method="EMA")])
    with pytest.raises(ValueError, match="duplicate association decision"):
        validate_association_events(
            [first, replace(first, event_id="different", candidate_slot_id=99)]
        )


def test_partial_explicit_transition_flags_are_rejected() -> None:
    with pytest.raises(ValueError, match="all events or no events"):
        aggregate_event_metrics(
            [_event(event_id="one"), _event(event_id="two", stage_id=1, id_switch=None)]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gap_opportunity", None),
        ("reactivation_attempt", None),
        ("reactivation", None),
    ],
)
def test_reactivation_audit_fields_are_required(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        aggregate_reactivation_metrics([_event(**{field: value})])


def test_correct_reactivation_requires_gt_and_gap_attempt() -> None:
    with pytest.raises(ValueError, match="GT entity|gt_entity_id"):
        aggregate_reactivation_metrics(
            [
                _event(
                    gt_entity_id=None,
                    gap_opportunity=True,
                    reactivation_attempt=True,
                    reactivation=True,
                    reactivation_correct=True,
                )
            ]
        )


@pytest.mark.parametrize(
    "flags",
    [
        {"transition_opportunity": True, "id_switch": False},
        {"transition_opportunity": True, "id_switch": True},
        {"gap_opportunity": True},
    ],
)
def test_identity_and_gap_opportunities_require_a_gt_entity(
    flags: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="GT entity"):
        validate_association_events(
            [_event(gt_entity_id=None, gt_present=False, **flags)]
        )
    with pytest.raises(ValueError, match="gap opportunity"):
        aggregate_reactivation_metrics(
            [
                _event(
                    gap_opportunity=False,
                    reactivation_attempt=True,
                    reactivation=True,
                    reactivation_correct=True,
                )
            ]
        )


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
            is_failure=True,
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
            is_failure=True,
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
            is_failure=True,
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
        "predicted_reactivation_events": 1,
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
        is_failure=True,
    )
    result = aggregate_reactivation_metrics([row])

    assert result["gap_opportunities"] == 1
    assert result["reactivation_attempts"] == 0
    assert result["predicted_reactivation_events"] == 0
    assert result["no_attempts"] == 1
    assert result["reactivation_accuracy"] is None
    assert result["reactivation_precision"] is None
    assert result["reactivation_recall"] == 0.0
    assert result["reactivation_coverage"] == 0.0


def test_transition_inference_counts_next_match_across_a_gap_without_cross_scope_mixing() -> (
    None
):
    rows = [
        _event(
            event_id="a0",
            method="Persist4D",
            stage_id=0,
            transition_opportunity=None,
            id_switch=None,
        ),
        _event(
            event_id="a2",
            method="Persist4D",
            stage_id=2,
            query_id="q2",
            predicted_identity_id=11,
            candidate_slot_id=11,
            transition_opportunity=None,
            id_switch=None,
        ),
        _event(
            event_id="b1",
            method="EMA",
            stage_id=1,
            query_id="q1",
            predicted_identity_id=99,
            candidate_slot_id=99,
            transition_opportunity=None,
            id_switch=None,
        ),
    ]
    identity, _ = aggregate_event_metrics(rows)
    assert identity["transition_opportunities"] == 1
    assert identity["id_switches"] == 1


def test_reactivation_precision_uses_all_dormant_reuses_not_only_gt_gap_attempts() -> (
    None
):
    rows = [
        _event(
            event_id="correct",
            stage_id=2,
            gap_opportunity=True,
            reactivation_attempt=True,
            reactivation=True,
            reactivation_correct=True,
            association_result="reactivation_correct",
        ),
        _event(
            event_id="false-reuse",
            gt_entity_id=8,
            stage_id=2,
            query_id="q8",
            gap_opportunity=False,
            reactivation_attempt=False,
            reactivation=True,
            reactivation_correct=False,
            wrong_reactivation=True,
            association_correct=False,
            association_result="reactivation_wrong",
            is_failure=True,
        ),
    ]
    result = aggregate_reactivation_metrics(rows)
    assert result["gap_opportunities"] == 1
    assert result["reactivation_attempts"] == 1
    assert result["predicted_reactivation_events"] == 2
    assert result["reactivation_accuracy"] == 1.0
    assert result["reactivation_precision"] == 0.5
    assert result["reactivation_recall"] == 1.0


def test_same_gt_cannot_receive_two_decisions_in_one_stage() -> None:
    with pytest.raises(ValueError, match="GT entity has duplicate stage decisions"):
        validate_association_events(
            [
                _event(event_id="one", query_id="q1"),
                _event(event_id="two", query_id="q2", candidate_slot_id=11),
            ]
        )


def test_rejected_birth_is_not_counted_as_a_created_false_birth() -> None:
    identity, _ = aggregate_event_metrics(
        [
            _event(
                association_correct=False,
                association_result="birth_rejected",
                birth_rejected=True,
                new_birth=False,
                is_failure=True,
            )
        ]
    )
    assert identity["births"] == 0
    assert identity["false_births"] == 0
    assert identity["rejected_births"] == 1


def test_any_missing_local_evidence_classifies_as_perception_failure() -> None:
    assert (
        classify_failure(
            _event(
                association_correct=False,
                local_observation_available=True,
                local_match_available=False,
                is_failure=True,
            )
        )
        == "F1"
    )


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
    assert (
        classify_failure(_event(association_correct=False, is_failure=True, **evidence))
        == expected
    )


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
        is_failure=True,
    )
    assert classify_failure(event) == "F1"


def test_specific_failure_mechanism_precedes_generic_association_miss() -> None:
    event = _event(
        association_correct=False,
        association_miss=True,
        reactivation=True,
        reactivation_correct=False,
        wrong_reactivation=True,
        is_failure=True,
    )
    assert classify_failure(event) == "F5"


def test_sequence_aggregates_preserve_method_order_prefix_and_cache_scope() -> None:
    rows = [
        _event(event_id="one", method="Persist4D", order_id="canonical", prefix=2),
        _event(event_id="two", method="EMA", order_id="canonical", prefix=2),
        _event(event_id="three", method="Persist4D", order_id="reverse", prefix=3),
    ]
    result = aggregate_metrics_by_sequence(rows)
    assert len(result) == 3
    assert {
        (row["method"], row["order_id"], row["prefix"], row["prediction_digest"])
        for row in result
    } == {
        ("Persist4D", "canonical", 2, "cache-a"),
        ("EMA", "canonical", 2, "cache-a"),
        ("Persist4D", "reverse", 3, "cache-a"),
    }


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
    for index, reference in enumerate(
        ("ref-a", "ref-b", "ref-c", "ref-d", "ref-e", "ref-f")
    ):
        common = {
            "reference_scene_id": reference,
            "master_sequence_id": f"master-{index}",
            "prefix": 4,
            "prediction_digest": "digest-4",
            "metric": "id_sw_rate",
            "order_id": "canonical",
        }
        rows.extend(
            [
                PairedMetricRecord(
                    method="Persist4D", value=0.08 + index * 0.01, **common
                ),
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


def test_paired_cluster_bootstrap_point_estimate_is_cluster_balanced() -> None:
    rows: list[PairedMetricRecord] = []
    for reference, pair_count, ours, baseline in (
        ("ref-many", 3, 0.0, 1.0),
        ("ref-one", 1, 1.0, 1.0),
        ("ref-two", 1, 1.0, 1.0),
        ("ref-three", 1, 1.0, 1.0),
        ("ref-four", 1, 1.0, 1.0),
        ("ref-five", 1, 1.0, 1.0),
    ):
        for index in range(pair_count):
            common = {
                "reference_scene_id": reference,
                "master_sequence_id": f"{reference}-{index}",
                "prefix": 4,
                "prediction_digest": f"digest-{reference}-{index}",
                "metric": "id_sw_rate",
                "order_id": "canonical",
            }
            rows.append(PairedMetricRecord(method="Persist4D", value=ours, **common))
            rows.append(PairedMetricRecord(method="EMA", value=baseline, **common))

    result = paired_cluster_bootstrap(rows, method="Persist4D", baseline_method="EMA")
    assert result["mean_delta"] == pytest.approx(-1.0 / 6.0)
    assert result["method_mean"] == pytest.approx(5.0 / 6.0)
    assert result["baseline_mean"] == pytest.approx(1.0)
    assert result["relative_reduction"] == pytest.approx(1.0 / 6.0)


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


def test_paired_bootstrap_requires_all_six_reference_clusters() -> None:
    with pytest.raises(ValueError, match="exactly six"):
        paired_cluster_bootstrap(
            _paired_rows()[:-2], method="Persist4D", baseline_method="EMA"
        )


def test_compact_paired_rows_cannot_bypass_record_validation() -> None:
    row = {
        "reference_scene_id": "ref-a",
        "master_sequence_id": "master-a",
        "prefix": 4,
        "metric": "id_sw_rate",
        "order_id": "",
        "prediction_digest": "digest-a",
        "persisted_value": 0.1,
        "baseline_value": 0.2,
    }
    with pytest.raises(ValueError, match="order_id"):
        paired_cluster_bootstrap([row], method="Persist4D", baseline_method="EMA")


def _gate_input() -> dict[str, object]:
    return {
        "paired_idsw": {
            4: {
                "relative_reduction": 0.20,
                "ci_high": 0.0,
                "n_clusters": 6,
                "n_pairs": 129,
                "clusters": [f"ref-{index}" for index in range(6)],
            },
            5: {
                "relative_reduction": 0.20,
                "ci_high": 0.0,
                "n_clusters": 6,
                "n_pairs": 129,
                "clusters": [f"ref-{index}" for index in range(6)],
            },
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
            "Persist4D": "a" * 64,
            "EMA": "a" * 64,
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
        "failure_counts": {
            "F1": 9,
            "F2": 0,
            "F3": 0,
            "F4": 0,
            "F5": 0,
            "F6": 0,
            "F7": 0,
            "unclassified": 1,
        },
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
        4: {
            "relative_reduction": 0.199999999,
            "ci_high": 0.0,
            "n_clusters": 6,
            "n_pairs": 129,
            "clusters": [f"ref-{index}" for index in range(6)],
        },
        5: {
            "relative_reduction": 0.20,
            "ci_high": 0.0,
            "n_clusters": 6,
            "n_pairs": 129,
            "clusters": [f"ref-{index}" for index in range(6)],
        },
    }
    result = evaluate_gates(values)
    assert result["G6A-1"]["passed"] is False


def test_reactivation_gate_requires_both_accuracy_and_recall_to_beat_baseline() -> None:
    values = _gate_input()
    values["reactivation"]["Persist4D"][4] = {"accuracy": 0.70, "recall": 0.25}
    values["reactivation"]["EMA"][4] = {"accuracy": 0.60, "recall": 0.30}
    result = evaluate_gates(values)
    assert result["G6A-2"]["passed"] is False
    assert result["G6A-2"]["checks"]["T4"]["accuracy_improved"] is True
    assert result["G6A-2"]["checks"]["T4"]["recall_improved"] is False


@pytest.mark.parametrize("missing", ["raw_prediction_fingerprints", "raw_local_ap"])
def test_local_invariance_gate_requires_both_fingerprint_and_metrics(
    missing: str,
) -> None:
    values = _gate_input()
    values.pop(missing)
    assert evaluate_gates(values)["G6A-3"]["passed"] is False


@pytest.mark.parametrize("fingerprint", [None, "", "not-a-sha256"])
def test_local_invariance_gate_rejects_invalid_fingerprints(
    fingerprint: object,
) -> None:
    values = _gate_input()
    values["raw_prediction_fingerprints"]["EMA"] = fingerprint
    assert evaluate_gates(values)["G6A-3"]["passed"] is False


@pytest.mark.parametrize("raw", [0.5, [0.5, 0.5]])
def test_local_invariance_gate_requires_method_wise_metrics(raw: object) -> None:
    values = _gate_input()
    values["raw_local_ap"] = raw
    assert evaluate_gates(values)["G6A-3"]["passed"] is False


@pytest.mark.parametrize(
    ("path", "value", "gate"),
    [
        (("paired_idsw", 4, "relative_reduction"), 1.01, "G6A-1"),
        (("paired_idsw", 4, "n_clusters"), 5, "G6A-1"),
        (("reactivation", "Persist4D", 3, "accuracy"), 1.01, "G6A-2"),
        (("raw_local_ap", "Persist4D", 0), -0.01, "G6A-3"),
        (("online_task", "Persist4D", 2, "t_mAP"), 1.01, "G6A-4"),
    ],
)
def test_metric_gates_fail_closed_on_out_of_domain_values(
    path: tuple[object, ...], value: object, gate: str
) -> None:
    values = _gate_input()
    target: object = values
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert evaluate_gates(values)[gate]["passed"] is False


def test_explainability_gate_rejects_impossible_share() -> None:
    values = _gate_input()
    values["explainability_share"] = 2.0
    assert evaluate_gates(values)["G6A-5"]["passed"] is False


def test_explainability_gate_rejects_share_count_disagreement() -> None:
    values = _gate_input()
    values["explainability_share"] = 0.9
    values["failure_counts"] = {
        **{code: 0 for code in ("F1", "F2", "F3", "F4", "F5", "F6", "F7")},
        "unclassified": 100,
    }
    assert evaluate_gates(values)["G6A-5"]["passed"] is False


def test_explainability_gate_requires_integer_failure_counts() -> None:
    values = _gate_input()
    values["failure_counts"] = {
        **{code: 0 for code in ("F2", "F3", "F4", "F5", "F6", "F7")},
        "F1": 0.9,
        "unclassified": 0.1,
    }
    assert evaluate_gates(values)["G6A-5"]["passed"] is False


@pytest.mark.parametrize("missing", ["n_pairs", "clusters"])
def test_identity_gate_requires_complete_paired_coverage(missing: str) -> None:
    values = _gate_input()
    values["paired_idsw"][4].pop(missing)
    assert evaluate_gates(values)["G6A-1"]["passed"] is False
