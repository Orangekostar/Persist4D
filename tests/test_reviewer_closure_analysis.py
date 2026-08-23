from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from scripts import reviewer_closure_analysis as analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "artifacts/reviewer_closure/full_history_tracker_raw.json"
SYSTEM_RESULTS_PATH = REPO_ROOT / "artifacts/system_comparison/per_sequence_results.csv"
ANALYSIS_CONFIG_PATH = REPO_ROOT / "configs/reviewer_closure/phase_i_analysis.yaml"


def _identity_counts(*, switches: int, transitions: int, correct: int, gaps: int):
    return {
        "deployment_id_switches": switches,
        "identity_transition_opportunities": transitions,
        "fragmentation_count": switches,
        "fragmentation_opportunities": transitions,
        "merge_count": 0,
        "merge_opportunities": transitions,
        "gap_opportunities": gaps,
        "recovery_attempts": gaps,
        "correct_recoveries": correct,
    }


def _rows(
    *,
    tracker_switches: int = 4,
    persist_switches: int = 1,
    tracker_correct: int = 1,
    persist_correct: int = 4,
):
    rows = []
    for reference_index in range(6):
        for order in analysis.ORDERS:
            for horizon in (4, 5):
                common = {
                    "reference_scene_id": f"reference-{reference_index}",
                    "master_sequence_id": f"master-{reference_index}",
                    "order_id": order,
                    "horizon": horizon,
                }
                rows.extend(
                    (
                        {
                            **common,
                            "method_id": "B1",
                            **_identity_counts(
                                switches=tracker_switches,
                                transitions=10,
                                correct=tracker_correct,
                                gaps=5,
                            ),
                        },
                        {
                            **common,
                            "method_id": "Persist4D",
                            **_identity_counts(
                                switches=persist_switches,
                                transitions=10,
                                correct=persist_correct,
                                gaps=5,
                            ),
                        },
                    )
                )
    return rows


def test_phase_i_analysis_config_freezes_selection_statistics_and_gate() -> None:
    config = analysis.load_phase_i_analysis_config(ANALYSIS_CONFIG_PATH)

    assert config["source_tracker_artifact_content_sha256"] == (
        "d6f2480e2787a943b5fd2a7f9c412a14c32bf0106bc23130f363fc72584efbf9"
    )
    assert config["strongest_simple_tracker"]["eligible_method_ids"] == [
        "B1",
        "B2",
        "B3",
    ]
    assert config["strongest_simple_tracker"]["diagnostic_excluded_method_id"] == "B4"
    assert config["statistics"]["bootstrap_replicates"] == 10_000
    assert config["gate_i"]["requires_complete_six_cluster_population"] is True


def test_select_strongest_simple_tracker_uses_count_aggregation_and_ties() -> None:
    base = {
        "reference_scene_id": "reference-0",
        "master_sequence_id": "master-0",
        "order_id": "canonical",
    }
    rows = []
    for method_id, cells in {
        "B1": ((1, 2, 1, 2), (10, 100, 1, 2)),
        "B2": ((1, 10, 0, 2), (10, 100, 2, 2)),
        "B3": ((1, 10, 0, 2), (10, 100, 2, 2)),
        "B4": ((0, 100, 2, 2), (0, 100, 2, 2)),
    }.items():
        for horizon, (switches, transitions, correct, gaps) in zip((4, 5), cells):
            rows.append(
                {
                    **base,
                    "method_id": method_id,
                    "horizon": horizon,
                    **_identity_counts(
                        switches=switches,
                        transitions=transitions,
                        correct=correct,
                        gaps=gaps,
                    ),
                }
            )

    selection = analysis.select_strongest_simple_tracker(rows)

    assert selection["selected_method_id"] == "B2"
    assert selection["eligible_method_ids"] == ["B1", "B2", "B3"]
    assert selection["diagnostic_excluded_method_id"] == "B4"
    assert [row["method_id"] for row in selection["ranking"]] == ["B2", "B3", "B1"]
    assert selection["ranking"][0]["normalized_id_switch_rate"] == pytest.approx(0.1)


def test_paired_cluster_bootstrap_is_six_cluster_and_deterministic() -> None:
    rows = _rows()

    first = analysis.paired_cluster_bootstrap(
        rows,
        tracker_method_id="B1",
        metrics=("normalized_id_switch_rate", "gap_recovery_recall"),
        horizons=(4, 5),
    )
    second = analysis.paired_cluster_bootstrap(
        rows,
        tracker_method_id="B1",
        metrics=("normalized_id_switch_rate", "gap_recovery_recall"),
        horizons=(4, 5),
    )

    assert first == second
    assert len(first) == 4
    switch = next(
        row
        for row in first
        if row["metric"] == "normalized_id_switch_rate" and row["horizon"] == 4
    )
    assert switch["cluster_count"] == 6
    assert switch["pair_count"] == 18
    assert switch["difference"] == pytest.approx(-0.3)
    assert switch["ci_lower"] == pytest.approx(-0.3)
    assert switch["ci_upper"] == pytest.approx(-0.3)


def test_order_loso_and_gate_require_robust_persist4d_advantage() -> None:
    rows = _rows()
    bootstrap = analysis.paired_cluster_bootstrap(
        rows,
        tracker_method_id="B1",
        metrics=("normalized_id_switch_rate", "gap_recovery_recall"),
        horizons=(4, 5),
    )
    order = analysis.order_robustness(
        rows,
        tracker_method_id="B1",
        metrics=("normalized_id_switch_rate", "gap_recovery_recall"),
        horizons=(4, 5),
    )
    loso = analysis.leave_one_scene_out(
        rows,
        tracker_method_id="B1",
        metrics=("normalized_id_switch_rate", "gap_recovery_recall"),
        horizons=(4, 5),
    )

    gate = analysis.derive_gate_i(
        tracker_method_id="B1",
        bootstrap_rows=bootstrap,
        order_rows=order,
        loso_rows=loso,
    )

    assert len(order) == 4
    assert all(row["expected_direction_consistent"] for row in order)
    assert all(row["complete_cluster_population"] for row in order)
    assert len(loso) == 24
    assert gate["classification"] == "TRACKER_REJECTED"
    assert len(gate["qualifying_advantages"]) == 4

    reversed_order = copy.deepcopy(order)
    reversed_order[0]["expected_direction_consistent"] = False
    reversed_loso = copy.deepcopy(loso)
    for row in reversed_loso:
        if (
            row["metric"] == reversed_order[0]["metric"]
            and row["horizon"] == reversed_order[0]["horizon"]
        ):
            row["expected_direction_consistent"] = False
    weakened_bootstrap = copy.deepcopy(bootstrap)
    for row in weakened_bootstrap:
        if (row["metric"], row["horizon"]) != (
            reversed_order[0]["metric"],
            reversed_order[0]["horizon"],
        ):
            row["ci_lower"] = -0.1
            row["ci_upper"] = 0.1
    explained = analysis.derive_gate_i(
        tracker_method_id="B1",
        bootstrap_rows=weakened_bootstrap,
        order_rows=reversed_order,
        loso_rows=reversed_loso,
    )
    assert explained["classification"] == "TRACKER_EXPLAINS_IDENTITY"
    assert explained["qualifying_advantages"] == []

    with pytest.raises(analysis.ReviewerClosureAnalysisError, match="coverage"):
        analysis.derive_gate_i(
            tracker_method_id="B1",
            bootstrap_rows=bootstrap[:-1],
            order_rows=order,
            loso_rows=loso,
        )


def test_real_phase_i_merge_has_exact_method_and_prefix_coverage() -> None:
    raw = analysis.load_tracker_raw_artifact(RAW_PATH)
    system_rows = analysis.read_system_per_sequence_results(SYSTEM_RESULTS_PATH)

    rows = analysis.merge_phase_i_results(raw, system_rows)

    assert len(rows) == 43 * 3 * 4 * 6
    assert {row["method_id"] for row in rows} == {
        "FullHistoryNative",
        "B1",
        "B2",
        "B3",
        "B4",
        "Persist4D",
    }
    assert {row["horizon"] for row in rows} == {2, 3, 4, 5}
    tracker = next(row for row in rows if row["method_id"] == "B1")
    full = next(
        row
        for row in rows
        if row["method_id"] == "FullHistoryNative"
        and all(
            row[field] == tracker[field]
            for field in (
                "reference_scene_id",
                "master_sequence_id",
                "order_id",
                "horizon",
            )
        )
    )
    assert tracker["causal_prefix_t_mAP"] == full["causal_prefix_t_mAP"]
    assert tracker["task_metric_source"] == "frozen_full_history_cache"
    assert full["task_metric_source"] == "frozen_full_history_cache"
    assert any(row["task_metric_source"] == "frozen_persist4d_cache" for row in rows)

    selection = analysis.select_strongest_simple_tracker(rows)
    bootstrap = analysis.paired_cluster_bootstrap(
        rows,
        tracker_method_id=selection["selected_method_id"],
    )
    t4_gap = next(
        row
        for row in bootstrap
        if row["metric"] == "gap_recovery_recall" and row["horizon"] == 4
    )
    assert t4_gap["reference_scene_count"] == 6
    assert t4_gap["cluster_count"] == 5
    assert t4_gap["missing_cluster_count"] == 1


def test_build_phase_i_artifacts_is_exact_and_idempotent(tmp_path: Path) -> None:
    result = analysis.build_phase_i_artifacts(
        raw_path=RAW_PATH,
        system_results_path=SYSTEM_RESULTS_PATH,
        output_root=tmp_path,
    )
    repeated = analysis.build_phase_i_artifacts(
        raw_path=RAW_PATH,
        system_results_path=SYSTEM_RESULTS_PATH,
        output_root=tmp_path,
    )

    assert repeated == result
    assert result["status"] == "pass"
    assert result["row_count"] == 43 * 3 * 4 * 6
    assert result["selected_tracker_method_id"] in {"B1", "B2", "B3"}
    assert result["gate_i_classification"] in {
        "TRACKER_REJECTED",
        "TRACKER_EXPLAINS_IDENTITY",
    }
    expected = {
        "full_history_tracker_results.csv",
        "full_history_tracker_aggregate.csv",
        "full_history_tracker_cluster_bootstrap.csv",
        "full_history_tracker_loso.csv",
        "full_history_tracker_order_robustness.csv",
        "full_history_tracker_selection.json",
        "gate_i.json",
        "FULL_HISTORY_TRACKER_AUDIT.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    with (tmp_path / "full_history_tracker_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == result["row_count"]
    gate = json.loads((tmp_path / "gate_i.json").read_text(encoding="utf-8"))
    assert gate["classification"] == result["gate_i_classification"]


def test_build_task_drift_rows_reports_signed_absolute_and_relative_delta() -> None:
    reference = {
        2: {field: 0.5 for field in analysis.TASK_FIELDS},
        3: {field: 0.4 for field in analysis.TASK_FIELDS},
        4: {field: 0.25 for field in analysis.TASK_FIELDS},
        5: {field: 0.0 for field in analysis.TASK_FIELDS},
    }
    replay = copy.deepcopy(reference)
    replay[4]["causal_prefix_t_mAP"] = 0.27
    replay[5]["causal_prefix_t_REC"] = 0.01

    rows = analysis.build_task_drift_rows(reference, replay)

    assert len(rows) == 4 * len(analysis.TASK_FIELDS)
    changed = next(
        row
        for row in rows
        if row["horizon"] == 4 and row["metric"] == "causal_prefix_t_mAP"
    )
    assert changed["difference"] == pytest.approx(0.02)
    assert changed["absolute_difference"] == pytest.approx(0.02)
    assert changed["relative_difference"] == pytest.approx(0.08)
    zero_reference = next(
        row
        for row in rows
        if row["horizon"] == 5 and row["metric"] == "causal_prefix_t_REC"
    )
    assert zero_reference["relative_difference"] is None
