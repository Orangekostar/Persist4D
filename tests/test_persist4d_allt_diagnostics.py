import pytest
import torch

from scripts.diagnose_persist4d_allt import (
    AllTDiagnosticError,
    build_failure_summary_rows,
    classify_prefix_failure,
    failure_summary_statistics,
    prefix_failure_records,
    select_diagnostic_panel,
    select_l_design,
    summarize_decoder_rows,
)
from scripts.system_comparison_metrics import (
    IdentityAssignmentUpdate,
    validate_causal_prefix_pair,
)


def test_diagnostic_panel_is_reference_sorted_and_score_independent() -> None:
    masters = [
        {"reference_scene_id": "ref-b", "sequence_id": "b-2", "score": 0.01},
        {"reference_scene_id": "ref-a", "sequence_id": "a-2", "score": 0.99},
        {"reference_scene_id": "ref-b", "sequence_id": "b-1", "score": 0.98},
        {"reference_scene_id": "ref-a", "sequence_id": "a-1", "score": 0.02},
        {"reference_scene_id": "ref-a", "sequence_id": "a-3", "score": 0.00},
    ]

    selected = select_diagnostic_panel(
        masters, per_reference=2, maximum_sequences=4
    )

    assert selected == [
        {"reference_scene_id": "ref-a", "sequence_id": "a-1"},
        {"reference_scene_id": "ref-a", "sequence_id": "a-2"},
        {"reference_scene_id": "ref-b", "sequence_id": "b-1"},
        {"reference_scene_id": "ref-b", "sequence_id": "b-2"},
    ]


def test_diagnostic_panel_rejects_incomplete_reference_coverage() -> None:
    with pytest.raises(AllTDiagnosticError, match="per-reference"):
        select_diagnostic_panel(
            [{"reference_scene_id": "ref-a", "sequence_id": "a-1"}],
            per_reference=2,
            maximum_sequences=12,
        )


def test_l_gate_prioritizes_real_query_competition() -> None:
    summary = {
        "query_competition": {
            "competed_active_query_fraction": 0.26,
            "mean_queries_per_gt_iou25": 2.1,
            "valid_scene_count": 12,
        },
        "earliest_attention": {
            "allowed_gt_fraction": 0.1,
            "valid_match_count": 20,
        },
    }
    assert select_l_design(summary) == {
        "design": "qcl_inspired_current_prediction",
        "reason": "substantial_query_competition",
        "status": "JUSTIFIED",
    }


def test_l_gate_uses_attention_relaxation_only_without_competition() -> None:
    summary = {
        "query_competition": {
            "competed_active_query_fraction": 0.1,
            "mean_queries_per_gt_iou25": 1.5,
            "valid_scene_count": 12,
        },
        "earliest_attention": {
            "allowed_gt_fraction": 0.49,
            "valid_match_count": 20,
        },
    }
    assert select_l_design(summary)["design"] == "first_cross_attention_relaxation"


def test_l_gate_returns_not_justified_when_neither_failure_is_present() -> None:
    summary = {
        "query_competition": {
            "competed_active_query_fraction": 0.1,
            "mean_queries_per_gt_iou25": 1.5,
            "valid_scene_count": 12,
        },
        "earliest_attention": {
            "allowed_gt_fraction": 0.8,
            "valid_match_count": 20,
        },
    }
    assert select_l_design(summary) == {
        "design": None,
        "reason": "diagnostic_gate_not_met",
        "status": "NOT_JUSTIFIED",
    }


def test_failure_classification_distinguishes_absence_mask_and_class() -> None:
    assert classify_prefix_failure([], []) == {
        "best_any_iou": 0.0,
        "best_class_compatible_iou": 0.0,
        "class_change": False,
        "mask_insufficient": False,
        "no_candidate": True,
    }
    assert classify_prefix_failure([0.3], [0.8]) == {
        "best_any_iou": 0.8,
        "best_class_compatible_iou": 0.3,
        "class_change": True,
        "mask_insufficient": True,
        "no_candidate": False,
    }
    assert classify_prefix_failure([0.8], [0.9]) == {
        "best_any_iou": 0.9,
        "best_class_compatible_iou": 0.8,
        "class_change": False,
        "mask_insufficient": False,
        "no_candidate": False,
    }
    assert classify_prefix_failure([0.0], [0.0])["no_candidate"] is True


def test_prefix_failure_records_use_class_compatible_iou_and_valid_denominator() -> None:
    pair = validate_causal_prefix_pair(
        prediction={
            "pred_masks": torch.tensor(
                [
                    [True, False],
                    [True, True],
                    [True, True],
                    [True, True],
                ]
            ),
            "pred_scores": torch.tensor([0.9, 0.8]),
            "pred_classes": torch.tensor([4, 3]),
        },
        target={
            "masks": torch.tensor([[True, True, True, True]]),
            "labels": torch.tensor([3]),
            "ids": torch.tensor([101]),
            "changes": torch.tensor([0]),
            "temporal_stages": torch.tensor([0, 1, 1, 1]),
        },
        horizon=2,
        observed_scan_ids=("s0", "s1"),
    )

    assert prefix_failure_records(pair, min_region_size=4) == [
        {
            "best_any_iou": 1.0,
            "best_class_compatible_iou": 0.75,
            "class_change": False,
            "gt_class": 3,
            "gt_id": 101,
            "gt_point_count": 4,
            "mask_insufficient": False,
            "no_candidate": False,
            "official_valid": True,
        }
    ]


def test_decoder_summary_uses_attention_feeding_and_earliest_rows() -> None:
    query_rows = [
        {
            "file_name": "a",
            "feeds_next_attention": True,
            "competed_active_query_fraction": 0.2,
            "mean_queries_per_gt_iou25": 2.0,
            "query_utilization_iou25": 0.5,
        },
        {
            "file_name": "b",
            "feeds_next_attention": True,
            "competed_active_query_fraction": 0.4,
            "mean_queries_per_gt_iou25": 4.0,
            "query_utilization_iou25": 0.7,
        },
        {
            "file_name": "a",
            "feeds_next_attention": False,
            "competed_active_query_fraction": 0.9,
            "mean_queries_per_gt_iou25": 9.0,
            "query_utilization_iou25": 0.9,
        },
    ]
    attention_rows = [
        {"decoder_prediction_layer": 0, "allowed_gt_fraction": 0.1},
        {"decoder_prediction_layer": 0, "allowed_gt_fraction": 0.3},
        {"decoder_prediction_layer": 1, "allowed_gt_fraction": 0.9},
    ]

    summary = summarize_decoder_rows(query_rows, attention_rows)

    assert summary["query_competition"] == {
        "competed_active_query_fraction": pytest.approx(0.3),
        "mean_queries_per_gt_iou25": pytest.approx(3.0),
        "query_utilization_iou25": pytest.approx(0.6),
        "valid_row_count": 2,
        "valid_scene_count": 2,
    }
    assert summary["earliest_attention"] == {
        "allowed_gt_fraction": pytest.approx(0.2),
        "decoder_prediction_layer": 0,
        "severe_below_0_25_fraction": pytest.approx(0.5),
        "valid_match_count": 2,
    }
    assert summary["l_gate"]["design"] == "qcl_inspired_current_prediction"


def _diagnostic_pair(
    *, horizon: int, compatible_mask: list[bool], other_mask: list[bool]
):
    point_count = len(compatible_mask)
    return validate_causal_prefix_pair(
        prediction={
            "pred_masks": torch.tensor(
                list(zip(compatible_mask, other_mask, strict=True)), dtype=torch.bool
            ),
            "pred_scores": torch.tensor([0.9, 0.8]),
            "pred_classes": torch.tensor([3, 4]),
        },
        target={
            "masks": torch.tensor([[True] * point_count]),
            "labels": torch.tensor([3]),
            "ids": torch.tensor([101]),
            "changes": torch.tensor([0]),
            "temporal_stages": torch.tensor(
                [min(index, horizon - 1) for index in range(point_count)]
            ),
        },
        horizon=horizon,
        observed_scan_ids=tuple(f"s{index}" for index in range(horizon)),
    )


def test_failure_summary_binds_trajectory_fh_cold_start_and_identity_events() -> None:
    b4_pairs = [
        _diagnostic_pair(horizon=1, compatible_mask=[False], other_mask=[False]),
        _diagnostic_pair(
            horizon=2,
            compatible_mask=[False, False],
            other_mask=[True, True],
        ),
        _diagnostic_pair(
            horizon=3,
            compatible_mask=[True, True, True],
            other_mask=[False, False, False],
        ),
        _diagnostic_pair(
            horizon=4,
            compatible_mask=[True, False, False, False],
            other_mask=[False, False, False, False],
        ),
        _diagnostic_pair(
            horizon=5,
            compatible_mask=[True, True, True, True, True],
            other_mask=[False, False, False, False, False],
        ),
    ]
    full_history_pairs = [
        _diagnostic_pair(
            horizon=horizon,
            compatible_mask=[True] * horizon,
            other_mask=[False] * horizon,
        )
        for horizon in range(2, 6)
    ]
    updates = [
        IdentityAssignmentUpdate(1, (101,), {101: 7}),
        IdentityAssignmentUpdate(2, (101,), {101: 7}),
        IdentityAssignmentUpdate(3, (101,), {101: 8}),
        IdentityAssignmentUpdate(4, (101, 202), {101: 8, 202: 7}),
        IdentityAssignmentUpdate(5, (101,), {101: 8}),
    ]

    rows = build_failure_summary_rows(
        reference_scene_id="ref-a",
        master_sequence_id="master-a",
        order_id="canonical",
        b4_pairs=b4_pairs,
        full_history_pairs=full_history_pairs,
        identity_updates=updates,
        min_region_size=4,
    )

    assert [row["T"] for row in rows] == [2, 3, 4, 5]
    assert rows[0]["class_change"] is True
    assert rows[0]["full_history_success_b4_failure"] is True
    assert rows[1]["fragmentation_event"] is True
    assert rows[2]["merge_event"] is False
    assert rows[0]["trajectory_worst_b4_iou"] == pytest.approx(0.0)
    assert rows[0]["first_b4_below_0_25_T"] == 2
    assert rows[0]["first_b4_below_0_50_T"] == 2
    assert rows[0]["t1_cold_start_failure"] is True
    assert rows[0]["t1_cold_start_contributes"] is True
    assert rows[2]["official_valid"] is True

    statistics = failure_summary_statistics(rows)
    assert statistics["row_count"] == 4
    assert statistics["official_valid_row_count"] == 2
    assert statistics["t1_cold_start_contribution"]["failed_row_count"] == 2
    assert statistics["t1_cold_start_contribution"]["contributed_row_count"] == 2
    assert statistics["category_counts_all"]["class_change"] == 1
