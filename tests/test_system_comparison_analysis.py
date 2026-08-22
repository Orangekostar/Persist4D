from __future__ import annotations

import copy

import pytest

from scripts.system_comparison_analysis import (
    AnalysisError,
    aggregate_identity_metrics,
    build_statistical_tables,
    detect_order_direction_reversal,
    leave_one_scene_out,
    paired_cluster_bootstrap,
    paired_cluster_values,
    validate_result_coverage,
)

METHODS = ("FullHistory", "Persist4D")
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)


def _rows(master_count: int = 6) -> list[dict[str, object]]:
    rows = []
    for master_index in range(master_count):
        reference = f"reference-{master_index % 6}"
        for order_index, order in enumerate(ORDERS):
            for horizon in HORIZONS:
                full_value = 0.10 + master_index / 100 + horizon / 1000
                order_delta = (0.01, -0.02, 0.01)[order_index]
                for method in METHODS:
                    value = (
                        full_value
                        if method == "FullHistory"
                        else full_value + order_delta
                    )
                    rows.append(
                        {
                            "method": method,
                            "reference_scene_id": reference,
                            "master_sequence_id": f"master-{master_index}",
                            "order_id": order,
                            "horizon": horizon,
                            "causal_prefix_t_mAP": value,
                            "causal_prefix_t_REC": value + 0.1,
                            "normalized_id_switch_rate": value / 2,
                            "gap_recovery_recall": None if horizon == 2 else value / 3,
                        }
                    )
    return rows


def test_result_coverage_requires_every_method_order_horizon_master_cell() -> None:
    rows = _rows()
    summary = validate_result_coverage(rows, expected_master_count=6)
    assert summary == {
        "method_count": 2,
        "master_count": 6,
        "reference_scene_count": 6,
        "order_count": 3,
        "horizon_count": 4,
        "row_count": 2 * 6 * 3 * 4,
    }

    with pytest.raises(AnalysisError, match="coverage"):
        validate_result_coverage(rows[:-1], expected_master_count=6)
    duplicate = [*rows, copy.deepcopy(rows[0])]
    with pytest.raises(AnalysisError, match="duplicate"):
        validate_result_coverage(duplicate, expected_master_count=6)


def test_paired_cluster_values_pair_before_aggregating() -> None:
    clusters = paired_cluster_values(
        _rows(), metric="causal_prefix_t_mAP", horizon=4
    )

    assert len(clusters) == 6
    assert {row["reference_scene_id"] for row in clusters} == {
        f"reference-{index}" for index in range(6)
    }
    # Across the three orders the preregistered deltas average to zero.
    assert all(row["difference"] == pytest.approx(0.0) for row in clusters)


def test_bootstrap_uses_10000_seed45_cluster_resamples_and_relative_difference() -> None:
    rows = _rows()
    for row in rows:
        if row["method"] == "Persist4D":
            row["causal_prefix_t_mAP"] = float(row["causal_prefix_t_mAP"]) + 0.02
    first = paired_cluster_bootstrap(
        rows,
        metrics=("causal_prefix_t_mAP",),
        replicates=10_000,
        seed=45,
    )
    second = paired_cluster_bootstrap(
        rows,
        metrics=("causal_prefix_t_mAP",),
        replicates=10_000,
        seed=45,
    )

    assert first == second
    assert len(first) == 4
    for row in first:
        assert row["bootstrap_replicates"] == 10_000
        assert row["seed"] == 45
        assert row["cluster_count"] == 6
        assert row["difference"] == pytest.approx(0.02)
        assert row["ci_lower"] == pytest.approx(0.02)
        assert row["ci_upper"] == pytest.approx(0.02)
        assert row["relative_difference"] == pytest.approx(
            row["difference"] / row["full_history_mean"]
        )


def test_loso_has_exactly_six_deterministic_drops() -> None:
    result = leave_one_scene_out(
        _rows(), metrics=("causal_prefix_t_mAP",), horizons=(5,)
    )

    assert len(result) == 6
    assert [row["dropped_reference_scene_id"] for row in result] == sorted(
        row["dropped_reference_scene_id"] for row in result
    )
    assert all(row["remaining_cluster_count"] == 5 for row in result)


def test_order_direction_reversal_is_explicit() -> None:
    result = detect_order_direction_reversal(
        _rows(), metric="causal_prefix_t_mAP", horizon=5
    )

    assert result["direction_reversal"] is True
    assert result["differences_by_order"]["canonical"] > 0
    assert result["differences_by_order"]["reverse"] < 0
    assert result["differences_by_order"]["sha256_seed45"] > 0


def test_identity_aggregation_uses_explicit_denominators() -> None:
    result = aggregate_identity_metrics(
        [
            {
                "deployment_id_switches": 1,
                "identity_transition_opportunities": 1,
                "fragmentation_count": 0,
                "fragmentation_opportunities": 2,
                "merge_count": 1,
                "merge_opportunities": 2,
                "gap_opportunities": 4,
                "recovery_attempts": 2,
                "correct_recoveries": 1,
            },
            {
                "deployment_id_switches": 0,
                "identity_transition_opportunities": 9,
                "fragmentation_count": 1,
                "fragmentation_opportunities": 2,
                "merge_count": 0,
                "merge_opportunities": 2,
                "gap_opportunities": 1,
                "recovery_attempts": 1,
                "correct_recoveries": 1,
            },
        ]
    )

    assert result["normalized_id_switch_rate"] == pytest.approx(0.1)
    assert result["fragmentation_rate"] == pytest.approx(0.25)
    assert result["merge_rate"] == pytest.approx(0.25)
    assert result["gap_recovery_accuracy"] == pytest.approx(2 / 3)
    assert result["gap_recovery_recall"] == pytest.approx(0.4)


def test_statistical_tables_include_task_identity_gap_and_latency() -> None:
    rows = _rows()
    profile_rows = [
        {
            "method": method,
            "reference_scene_id": f"reference-{index}",
            "master_sequence_id": f"master-{index}",
            "order_id": "canonical",
            "horizon": horizon,
            "median_latency_ms": float(index + horizon)
            + (1.0 if method == "Persist4D" else 0.0),
        }
        for index in range(6)
        for horizon in HORIZONS
        for method in METHODS
    ]
    tables = build_statistical_tables(
        rows,
        profile_rows,
        expected_master_count=6,
    )

    assert len(tables["cluster_bootstrap"]) == 5 * 4
    assert len(tables["leave_one_scene_out"]) == 5 * 4 * 6
    assert len(tables["order_robustness"]) == 4 * 4
    assert {row["metric"] for row in tables["cluster_bootstrap"]} == {
        "causal_prefix_t_mAP",
        "causal_prefix_t_REC",
        "normalized_id_switch_rate",
        "gap_recovery_recall",
        "median_latency_ms",
    }


def test_gap_statistics_preserve_loso_when_one_cluster_has_no_finite_pairs() -> None:
    rows = _rows()
    for row in rows:
        if (
            row["reference_scene_id"] == "reference-5"
            and row["horizon"] == 3
        ):
            row["gap_recovery_recall"] = None
    profile_rows = [
        {
            "method": method,
            "reference_scene_id": f"reference-{index}",
            "master_sequence_id": f"master-{index}",
            "order_id": "canonical",
            "horizon": horizon,
            "median_latency_ms": float(index + horizon),
        }
        for index in range(6)
        for horizon in HORIZONS
        for method in METHODS
    ]

    tables = build_statistical_tables(
        rows,
        profile_rows,
        expected_master_count=6,
    )

    bootstrap = next(
        row
        for row in tables["cluster_bootstrap"]
        if row["metric"] == "gap_recovery_recall" and row["horizon"] == 3
    )
    assert bootstrap["cluster_count"] == 5
    loso = [
        row
        for row in tables["leave_one_scene_out"]
        if row["metric"] == "gap_recovery_recall" and row["horizon"] == 3
    ]
    assert len(loso) == 6
    assert {row["remaining_cluster_count"] for row in loso} == {4, 5}

    undefined_loso = [
        row
        for row in tables["leave_one_scene_out"]
        if row["metric"] == "gap_recovery_recall" and row["horizon"] == 2
    ]
    assert len(undefined_loso) == 6
    assert all(row["remaining_cluster_count"] == 0 for row in undefined_loso)
