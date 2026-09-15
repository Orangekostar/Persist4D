from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_task_memory_final import (
    REPORT_HORIZONS,
    TASK_METRICS,
    FinalAnalysisError,
    bootstrap_equal_reference_deltas,
    build_all_t_comparison,
    build_retention_rows,
    load_legacy_baseline_rows,
    load_resource_status,
    validate_identity_events,
    validate_identity_rows,
    validate_per_reference_rows,
    validate_primary_rows,
)
from scripts.task_memory_contracts import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _primary_rows(variant: str, checkpoint: str, base: float):
    return [
        {
            "population_id": "protocol_b_43_masters_3_orders",
            "variant": variant,
            "checkpoint_sha256": checkpoint,
            "training_seed": 45,
            "evaluation_seed": 45,
            "T": horizon,
            "episode_count": 129,
            "policy": "lag1",
            "reducer": "mean",
            "reference_count": 6,
            "master_count": 43,
            **{
                metric: base + horizon / 100 + index / 1000
                for index, metric in enumerate(TASK_METRICS)
            },
            "local_current_AP": base + horizon / 200,
        }
        for horizon in REPORT_HORIZONS
    ]


def test_legacy_b4_rows_remain_commit0_evidence() -> None:
    rows = load_legacy_baseline_rows(
        PROJECT_ROOT / "artifacts/allt_task_superiority_v1/baseline/all_t_metrics.csv"
    )

    b4 = [row for row in rows if row["variant"] == "B4-commit0"]
    assert len(b4) == 4
    assert {row["policy"] for row in b4} == {"commit0"}


def test_primary_rows_lock_checkpoint_policy_reducer_and_full_metric_schema() -> None:
    checkpoint = "a" * 64
    rows = validate_primary_rows(
        _primary_rows("M3-V-CORE", checkpoint, 0.2),
        expected_variant="M3-V-CORE",
        expected_checkpoint_sha256=checkpoint,
    )

    assert len(rows) == 4
    assert {row["T"] for row in rows} == set(REPORT_HORIZONS)
    assert all(row["direct_current_AP"] > 0 for row in rows)

    mixed = _primary_rows("M3-V-CORE", checkpoint, 0.2)
    mixed[-1]["reducer"] = "max"
    with pytest.raises(FinalAnalysisError, match="policy/reducer"):
        validate_primary_rows(
            mixed,
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256=checkpoint,
        )

    wrong_population = _primary_rows("M3-V-CORE", checkpoint, 0.2)
    for row in wrong_population:
        row["episode_count"] = 128
    with pytest.raises(FinalAnalysisError, match="Protocol-B population"):
        validate_primary_rows(
            wrong_population,
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256=checkpoint,
        )


def test_all_t_comparison_counts_all_twenty_cells_and_uses_strict_epsilon() -> None:
    candidate = _primary_rows("M3-V-CORE", "a" * 64, 0.3)
    baseline = _primary_rows("FH-CONT", "b" * 64, 0.2)
    candidate[2]["t_mAP"] = float(baseline[2]["t_mAP"]) + 1e-6

    result = build_all_t_comparison(
        candidate,
        baseline,
        comparison_name="M3-V-CORE_vs_FH-CONT",
    )

    assert len(result["rows"]) == 20
    assert result["positive_cells"] == 19
    assert result["tmap_all_t"] == "FAIL"
    assert result["task_metrics_all_t"] == "FAIL"
    assert result["failed_tmap_horizons"] == [4]
    assert result["minimum_tmap_delta"] == pytest.approx(1e-6)
    assert result["mean_tmap_delta"] == pytest.approx(0.07500025)


def test_retention_reports_absolute_relative_and_global_drop() -> None:
    rows = _primary_rows("M3-V-CORE", "a" * 64, 0.2)
    for row, value in zip(rows, (0.5, 0.45, 0.4, 0.42), strict=True):
        row["t_mAP"] = value

    retention = build_retention_rows(rows)

    assert [row["A_t_mAP"] for row in retention] == [0.5, 0.45, 0.4, 0.42]
    assert [row["R_t_mAP"] for row in retention] == pytest.approx([1.0, 0.9, 0.8, 0.84])
    assert all(row["Dmax_t_mAP"] == pytest.approx(0.1) for row in retention)


def test_resource_status_uses_valid_profile_and_missing_means_not_measured(
    tmp_path,
) -> None:
    summary = tmp_path / "run_summary.json"
    assert load_resource_status(summary) == "NOT_MEASURED"

    payload = {"status": "PASS", "resource_status": "TRADEOFF"}
    payload["content_sha256"] = canonical_json_sha256(payload)
    summary.write_text(json.dumps(payload), encoding="utf-8")

    assert load_resource_status(summary) == "TRADEOFF"

    payload["resource_status"] = "PASS"
    payload["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "content_sha256"}
    )
    summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalAnalysisError, match="resource status"):
        load_resource_status(summary)


def test_reference_bootstrap_is_six_cluster_fixed_seed_and_descriptive() -> None:
    paired = [
        {
            "reference_id": f"reference-{reference}",
            "T": horizon,
            "metric": metric,
            "delta": float(reference),
        }
        for metric in TASK_METRICS
        for horizon in (2, 5)
        for reference in range(6)
    ]

    first = bootstrap_equal_reference_deltas(paired, seed=45, resamples=1000)
    second = bootstrap_equal_reference_deltas(paired, seed=45, resamples=1000)

    assert first == second
    assert len(first) == 10
    assert all(row["reference_count"] == 6 for row in first)
    assert all(row["estimand"] == "equal_reference_descriptive" for row in first)
    assert all(row["mean_delta"] == pytest.approx(2.5) for row in first)
    with pytest.raises(FinalAnalysisError, match="1000"):
        bootstrap_equal_reference_deltas(paired, seed=45, resamples=1001)


def test_identity_rows_preserve_counts_denominators_and_na() -> None:
    rows = []
    for horizon in REPORT_HORIZONS:
        gaps = 0 if horizon == 2 else 10
        attempts = 0 if horizon == 2 else 5
        correct = 0 if horizon == 2 else 4
        rows.append(
            {
                "population_id": "protocol_b_43_masters_3_orders",
                "variant": "M3-V-CORE",
                "checkpoint_sha256": "a" * 64,
                "training_seed": 45,
                "evaluation_seed": 45,
                "visual_content_control": "native",
                "T": horizon,
                "episode_count": 129,
                "identity_linker": "lag1-route-then-class-iou-v1",
                "policy": "lag1",
                "correct_recoveries": correct,
                "deployment_id_switches": 2,
                "fragmentation_count": 3,
                "fragmentation_opportunities": 6,
                "gap_opportunities": gaps,
                "identity_transition_opportunities": 8,
                "merge_count": 1,
                "merge_opportunities": 4,
                "recovery_attempts": attempts,
                "fragmentation_rate": 0.5,
                "gap_recovery_accuracy": None if not attempts else 0.8,
                "gap_recovery_attempt_coverage": None if not gaps else 0.5,
                "gap_recovery_recall": None if not gaps else 0.4,
                "merge_rate": 0.25,
                "normalized_id_switch_rate": 0.25,
                "reference_count": 6,
                "master_count": 43,
            }
        )

    validated = validate_identity_rows(
        rows,
        expected_variant="M3-V-CORE",
        expected_checkpoint_sha256="a" * 64,
    )

    assert validated[0]["gap_recovery_accuracy"] is None
    assert validated[-1]["gap_recovery_attempt_coverage"] == pytest.approx(0.5)
    broken = [dict(row) for row in rows]
    broken[-1]["gap_recovery_recall"] = 0.7
    with pytest.raises(FinalAnalysisError, match="rate"):
        validate_identity_rows(
            broken,
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("episode_count", 128),
        ("reference_count", 5),
        ("master_count", 42),
        ("identity_linker", "raw-query-index"),
    ),
)
def test_identity_rows_lock_protocol_b_population_and_linker(
    field: str, value: object
) -> None:
    rows = []
    for horizon in REPORT_HORIZONS:
        rows.append(
            {
                "population_id": "protocol_b_43_masters_3_orders",
                "variant": "M3-V-CORE",
                "checkpoint_sha256": "a" * 64,
                "training_seed": 45,
                "evaluation_seed": 45,
                "visual_content_control": "native",
                "T": horizon,
                "episode_count": 129,
                "identity_linker": "lag1-route-then-class-iou-v1",
                "policy": "lag1",
                "correct_recoveries": 1,
                "deployment_id_switches": 2,
                "fragmentation_count": 3,
                "fragmentation_opportunities": 6,
                "gap_opportunities": 2,
                "identity_transition_opportunities": 8,
                "merge_count": 1,
                "merge_opportunities": 4,
                "recovery_attempts": 2,
                "fragmentation_rate": 0.5,
                "gap_recovery_accuracy": 0.5,
                "gap_recovery_attempt_coverage": 1.0,
                "gap_recovery_recall": 0.5,
                "merge_rate": 0.25,
                "normalized_id_switch_rate": 0.25,
                "reference_count": 6,
                "master_count": 43,
            }
        )
    rows[-1][field] = value

    with pytest.raises(FinalAnalysisError, match="identity (population|linker)"):
        validate_identity_rows(
            rows,
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256="a" * 64,
        )


def test_identity_events_include_error_counts_and_exact_rates() -> None:
    events = validate_identity_events(
        {
            "birth_count": 5,
            "false_birth_count": 2,
            "false_birth_rate": 0.4,
            "reactivation_count": 4,
            "false_reactivation_count": 1,
            "false_reactivation_rate": 0.25,
            "rejected_birth_count": 3,
        }
    )

    assert events["false_birth_count"] == 2
    assert events["false_reactivation_count"] == 1
    broken = dict(events)
    broken["false_reactivation_rate"] = 0.5
    with pytest.raises(FinalAnalysisError, match="identity event rate"):
        validate_identity_events(broken)


def test_per_reference_rows_require_six_references_and_global_43_by_129() -> None:
    rows = [
        {
            "population_id": "protocol_b_43_masters_3_orders",
            "variant": "M3-V-CORE",
            "checkpoint_sha256": "a" * 64,
            "reference_id": f"reference-{reference}",
            "T": horizon,
            **{metric: 0.2 for metric in TASK_METRICS},
            "direct_current_AP": 0.3,
            "master_count": count,
            "order_count": count * 3,
        }
        for reference, count in enumerate((8, 7, 7, 7, 7, 7))
        for horizon in REPORT_HORIZONS
    ]

    validated = validate_per_reference_rows(
        rows,
        expected_variant="M3-V-CORE",
        expected_checkpoint_sha256="a" * 64,
    )

    assert len(validated) == 24
    broken = [dict(row) for row in rows]
    broken[-1]["order_count"] = 18
    with pytest.raises(FinalAnalysisError, match="per-reference population"):
        validate_per_reference_rows(
            broken,
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256="a" * 64,
        )
