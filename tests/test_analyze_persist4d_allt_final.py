from __future__ import annotations

import pytest

from scripts.analyze_persist4d_allt_final import (
    TASK_METRICS,
    FinalAnalysisError,
    aggregate_identity_rows,
    bootstrap_equal_cluster_effect,
    build_paired_reference_rows,
)


def _metric_cells(model: str) -> list[dict[str, object]]:
    rows = []
    for reference_index in range(6):
        for horizon in range(2, 6):
            base = float(reference_index + horizon) / 100
            rows.append(
                {
                    "model": model,
                    "reference_scene_id": f"reference-{reference_index}",
                    "T": horizon,
                    "sequence_count": 3,
                    **{
                        metric: base + metric_index / 100
                        for metric_index, metric in enumerate(TASK_METRICS)
                    },
                }
            )
    return rows


def test_paired_reference_rows_require_exact_matched_coverage() -> None:
    candidate = _metric_cells("C2")
    baseline = _metric_cells("FH-adapt")
    for row in candidate:
        for metric in TASK_METRICS:
            row[metric] = float(row[metric]) + 0.25

    paired = build_paired_reference_rows(candidate, baseline)

    assert len(paired) == 120
    assert {
        (row["reference_scene_id"], row["T"], row["metric"]) for row in paired
    } == {
        (f"reference-{reference}", horizon, metric)
        for reference in range(6)
        for horizon in range(2, 6)
        for metric in TASK_METRICS
    }
    assert all(row["delta"] == pytest.approx(0.25) for row in paired)
    assert all(row["candidate_model"] == "C2" for row in paired)
    assert all(row["baseline_model"] == "FH-adapt" for row in paired)

    with pytest.raises(FinalAnalysisError, match="coverage"):
        build_paired_reference_rows(candidate, baseline[:-1])
    with pytest.raises(FinalAnalysisError, match="duplicate"):
        build_paired_reference_rows([*candidate, candidate[0]], baseline)


def _identity_sequence_rows() -> list[dict[str, object]]:
    rows = []
    for model in ("C2", "FH-adapt"):
        for sequence in range(2):
            for horizon in range(2, 6):
                rows.append(
                    {
                        "model": model,
                        "reference_scene_id": f"reference-{sequence}",
                        "master_sequence_id": f"master-{sequence}",
                        "order_id": "canonical",
                        "T": horizon,
                        "deployment_id_switches": sequence,
                        "identity_transition_opportunities": 2,
                        "fragmentation_count": 1,
                        "fragmentation_opportunities": 4,
                        "merge_count": 0,
                        "merge_opportunities": 2,
                        "gap_opportunities": 0 if horizon == 2 else 2,
                        "recovery_attempts": 0 if horizon == 2 else 1,
                        "correct_recoveries": 0 if horizon == 2 else sequence,
                    }
                )
    return rows


def test_identity_aggregation_preserves_counts_denominators_and_na() -> None:
    aggregated = aggregate_identity_rows(
        _identity_sequence_rows(), expected_sequence_count=2
    )

    assert len(aggregated) == 8
    c2_t2 = next(row for row in aggregated if row["model"] == "C2" and row["T"] == 2)
    assert c2_t2["deployment_id_switches"] == 1
    assert c2_t2["identity_transition_opportunities"] == 4
    assert c2_t2["normalized_id_switch_rate"] == pytest.approx(0.25)
    assert c2_t2["fragmentation_rate"] == pytest.approx(0.25)
    assert c2_t2["gap_recovery_accuracy"] is None
    assert c2_t2["gap_recovery_recall"] is None


def test_equal_cluster_bootstrap_is_fixed_seed_and_descriptive() -> None:
    paired = []
    for metric in TASK_METRICS:
        for horizon in range(2, 6):
            for reference in range(6):
                paired.append(
                    {
                        "reference_scene_id": f"reference-{reference}",
                        "T": horizon,
                        "metric": metric,
                        "delta": float(reference),
                    }
                )

    first = bootstrap_equal_cluster_effect(paired, seed=45, resamples=1000)
    second = bootstrap_equal_cluster_effect(paired, seed=45, resamples=1000)

    assert first == second
    assert len(first) == 20
    assert all(row["estimand"] == "equal_cluster_effect" for row in first)
    assert all(row["cluster_count"] == 6 for row in first)
    assert all(row["equal_cluster_mean_delta"] == pytest.approx(2.5) for row in first)
    assert first[0]["bootstrap_ci_lower"] == pytest.approx(1.1666666666666667)
    assert first[0]["bootstrap_ci_upper"] == pytest.approx(3.8333333333333335)
