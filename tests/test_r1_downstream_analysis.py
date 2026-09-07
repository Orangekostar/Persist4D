from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _analysis():
    return importlib.import_module("scripts.analyze_r1_downstream_validation")


def _identity_counts(**overrides: int) -> dict[str, int]:
    row = {
        "deployment_id_switches": 0,
        "identity_transition_opportunities": 0,
        "fragmentation_count": 0,
        "fragmentation_opportunities": 0,
        "merge_count": 0,
        "merge_opportunities": 0,
        "gap_opportunities": 0,
        "recovery_attempts": 0,
        "correct_recoveries": 0,
        "association_event_count": 0,
        "new_birth_count": 0,
        "false_birth_count": 0,
        "birth_rejected_count": 0,
    }
    row.update(overrides)
    return row


def test_identity_aggregation_derives_attempt_coverage_from_pooled_counts() -> None:
    analysis = _analysis()

    result = analysis.aggregate_identity_rows(
        [
            _identity_counts(
                deployment_id_switches=1,
                identity_transition_opportunities=4,
                gap_opportunities=3,
                recovery_attempts=2,
                correct_recoveries=1,
            ),
            _identity_counts(
                deployment_id_switches=2,
                identity_transition_opportunities=6,
                gap_opportunities=2,
                recovery_attempts=2,
                correct_recoveries=2,
            ),
        ]
    )

    assert result["deployment_id_switches"] == 3
    assert result["identity_transition_opportunities"] == 10
    assert result["normalized_id_switch_rate"] == pytest.approx(0.3)
    assert result["gap_opportunities"] == 5
    assert result["recovery_attempts"] == 4
    assert result["correct_recoveries"] == 3
    assert result["recovery_attempt_coverage"] == pytest.approx(0.8)
    assert result["gap_recovery_accuracy"] == pytest.approx(0.75)
    assert result["gap_recovery_recall"] == pytest.approx(0.6)


def test_recovery_decision_requires_positive_pooled_and_four_clusters_at_t4_t5() -> None:
    analysis = _analysis()
    aggregate = []
    clusters = []
    for horizon in (4, 5):
        aggregate.extend(
            [
                {"method": "B2", "order_id": "all", "horizon": horizon,
                 "gap_recovery_recall": 0.2},
                {"method": "B4", "order_id": "all", "horizon": horizon,
                 "gap_recovery_recall": 0.3},
            ]
        )
        for index, delta in enumerate((0.1, 0.1, 0.05, 0.01, -0.02, -0.03)):
            reference = f"cluster-{index}"
            clusters.extend(
                [
                    {"method": "B2", "order_id": "all", "horizon": horizon,
                     "reference_scene_id": reference, "gap_recovery_recall": 0.2},
                    {"method": "B4", "order_id": "all", "horizon": horizon,
                     "reference_scene_id": reference,
                     "gap_recovery_recall": 0.2 + delta},
                ]
            )

    decision = analysis.classify_recovery(aggregate, clusters)

    assert decision["status"] == "RECOVERY_SUPPORTED"
    assert decision["T4"]["pooled_b4_minus_b2"] == pytest.approx(0.1)
    assert decision["T4"]["positive_cluster_count"] == 4
    assert decision["T4"]["valid_cluster_count"] == 6

    clusters[-6]["gap_recovery_recall"] = 0.25
    clusters[-5]["gap_recovery_recall"] = 0.2
    assert analysis.classify_recovery(aggregate, clusters)["status"] == (
        "RECOVERY_NOT_SUPPORTED"
    )


def test_paired_cluster_bootstrap_is_seed45_deterministic_and_b4_minus_b2() -> None:
    analysis = _analysis()
    rows = []
    for index, delta in enumerate((-0.03, -0.01, 0.01, 0.03, 0.05, 0.07)):
        reference = f"cluster-{index}"
        rows.extend(
            [
                {"method": "B2", "score_reducer": "mean", "order_id": "all",
                 "horizon": 4, "reference_scene_id": reference,
                 "causal_prefix_t_mAP": 0.2},
                {"method": "B4", "score_reducer": "mean", "order_id": "all",
                 "horizon": 4, "reference_scene_id": reference,
                 "causal_prefix_t_mAP": 0.2 + delta},
            ]
        )

    first = analysis.paired_cluster_bootstrap(
        rows,
        metric="causal_prefix_t_mAP",
        horizon=4,
        score_reducer="mean",
        replicates=10_000,
        seed=45,
    )
    second = analysis.paired_cluster_bootstrap(
        rows,
        metric="causal_prefix_t_mAP",
        horizon=4,
        score_reducer="mean",
        replicates=10_000,
        seed=45,
    )

    assert first == second
    assert first["cluster_count"] == 6
    assert first["b4_minus_b2"] == pytest.approx(0.02)
    assert first["positive_cluster_count"] == 4
    assert first["ci_lower"] < first["b4_minus_b2"] < first["ci_upper"]
    with pytest.raises(analysis.R1AnalysisError, match="10,000.*seed 45"):
        analysis.paired_cluster_bootstrap(
            rows,
            metric="causal_prefix_t_mAP",
            horizon=4,
            score_reducer="mean",
            replicates=999,
            seed=45,
        )


def test_gap_ledger_keeps_gap_opportunities_and_reactivation_attempts_only() -> None:
    analysis = _analysis()
    rows = analysis.gap_event_rows(
        [
            {"event_id": "active", "gap_opportunity": False,
             "reactivation_attempt": False},
            {"event_id": "miss", "gap_opportunity": True,
             "reactivation_attempt": False},
            {"event_id": "attempt", "gap_opportunity": True,
             "reactivation_attempt": True},
        ]
    )

    assert [row["event_id"] for row in rows] == ["miss", "attempt"]
    assert rows[0]["attempt_coverage_event"] is False
    assert rows[1]["attempt_coverage_event"] is True


def test_per_sequence_coverage_is_exact_for_frozen_protocol() -> None:
    analysis = _analysis()
    orders = ("canonical", "reverse", "sha256_seed45")
    horizons = (2, 4, 5)
    task_branches = (
        ("FullHistory", "official"),
        *((method, reducer) for method in ("B2", "B4")
          for reducer in ("mean", "latest", "max")),
    )
    task_rows = []
    identity_rows = []
    local_rows = []
    for master_index in range(43):
        reference = f"cluster-{master_index % 6}"
        master = f"master-{master_index}"
        for order in orders:
            for horizon in horizons:
                local_rows.append(
                    {"scope": "sequence", "reference_scene_id": reference,
                     "master_sequence_id": master, "order_id": order,
                     "horizon": horizon}
                )
                for method, reducer in task_branches:
                    task_rows.append(
                        {"method": method, "score_reducer": reducer,
                         "reference_scene_id": reference,
                         "master_sequence_id": master, "order_id": order,
                         "horizon": horizon}
                    )
                for method in ("B2", "B4"):
                    identity_rows.append(
                        {"method": method, "reference_scene_id": reference,
                         "master_sequence_id": master, "order_id": order,
                         "horizon": horizon}
                    )

    result = analysis.validate_per_sequence_coverage(
        task_rows=task_rows,
        identity_rows=identity_rows,
        local_rows=local_rows,
    )

    assert result == {
        "master_count": 43,
        "sequence_count": 129,
        "reference_cluster_count": 6,
        "task_row_count": 2709,
        "identity_row_count": 774,
        "local_row_count": 387,
    }
    task_rows.pop()
    with pytest.raises(analysis.R1AnalysisError, match="task coverage"):
        analysis.validate_per_sequence_coverage(
            task_rows=task_rows,
            identity_rows=identity_rows,
            local_rows=local_rows,
        )


def test_metric_dataset_spec_is_resolved_from_explicit_data_root(
    tmp_path: Path,
) -> None:
    analysis = _analysis()
    expected = tmp_path / "data/processed/rio/rio.yaml"
    expected.parent.mkdir(parents=True)
    expected.write_text("dataset: rio\n", encoding="utf-8")

    assert analysis.resolve_metric_dataset_spec(tmp_path) == expected.resolve()
    expected.unlink()
    with pytest.raises(analysis.R1AnalysisError, match="metric dataset spec"):
        analysis.resolve_metric_dataset_spec(tmp_path)
