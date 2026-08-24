from __future__ import annotations

import pytest

from scripts.system_comparison_v2_analysis import TASK_FIELDS
from scripts.system_comparison_v2_attribution import (
    METHODS,
    _none_step,
    _report,
    _summary_row,
)


class _FakeAccumulator:
    def compute(self) -> dict[str, float]:
        return {field: 0.5 for field in TASK_FIELDS}


def test_no_identity_step_keeps_every_query_unmatched() -> None:
    step = _none_step(stage=2, query_count=3, sequence_id="sequence")

    assert step.stage_id == 2
    assert step.track_ids == (None, None, None)
    assert step.valid == (True, True, True)


def test_summary_reports_candidate_empty_mask_and_score_distribution() -> None:
    stats = {
        "sequence_count": 1,
        "candidate_count": 2,
        "empty_count": 1,
        "current_count": 0,
        "scores": [0.25, 0.75],
    }

    row = _summary_row(
        method="L0",
        order="all",
        horizon=2,
        accumulator=_FakeAccumulator(),
        stats=stats,
    )

    assert row["trajectory_candidate_count_total"] == 2
    assert row["empty_trajectory_mask_count"] == 1
    assert row["empty_trajectory_mask_rate"] == pytest.approx(0.5)
    assert row["score_median"] == pytest.approx(0.5)
    assert row["score_mean"] == pytest.approx(0.5)


def test_report_names_all_paths_and_forbids_additive_interpretation() -> None:
    rows = []
    for method in METHODS:
        for horizon in (2, 3, 4, 5):
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "order_id": "all",
                    "trajectory_candidate_count_mean": 1.0,
                    "current_stage_candidate_count_mean": 1.0,
                    "current_stage_AP": 0.1,
                    "causal_prefix_t_mAP": 0.2,
                    "causal_prefix_t_REC": 0.3,
                }
            )

    report = _report(rows).decode("utf-8")

    assert all(f"| {method} |" in report for method in METHODS)
    assert "not assumed to add" in report


def test_runner_attribute_stage_dispatches_attribution(monkeypatch, tmp_path) -> None:
    from scripts import run_system_comparison_v2 as runner

    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "pass", "row_count": 80}

    monkeypatch.setattr(
        "scripts.system_comparison_v2_attribution.run_postprocessing_attribution",
        fake_run,
    )

    assert (
        runner.main(
            [
                "attribute",
                "--metadata",
                str(tmp_path / "metadata.json"),
                "--cache-root",
                str(tmp_path / "cache"),
                "--cache-manifest",
                str(tmp_path / "manifest.json"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
                "--attribution-root",
                str(tmp_path / "attribution"),
            ]
        )
        == 0
    )
    assert observed["output_root"] == tmp_path / "attribution"
