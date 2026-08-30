from __future__ import annotations

import csv
import json

import pytest

from scripts.finalize_rescene_rootcause_full_training import (
    build_full_training_manifest,
    read_full_validation_trajectory,
    select_full_checkpoint,
)
from utils.rescene_rootcause_evaluation import RootCauseEvaluationError
from utils.rescene_rootcause_preflight import canonical_sha256

VALIDATION_EPOCHS = tuple(range(15, 451, 15))


def _write_metrics(path, epochs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "epoch",
        "step",
        "val_loss",
        "val_mean_stage1-AP",
        "val_mean_stage2-AP",
        "val_mean_AP",
        "val_mean_t-AP",
        "val_mean_t-AP_50",
        "val_mean_t-AP_25",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for completed_epoch in epochs:
            t_map = 0.40 if completed_epoch == 315 else 0.20 + completed_epoch / 10_000
            writer.writerow(
                {
                    "epoch": completed_epoch - 1,
                    "step": completed_epoch * 66 - 1,
                    "val_loss": 5.0,
                    "val_mean_stage1-AP": 0.30,
                    "val_mean_stage2-AP": 0.32,
                    "val_mean_AP": 0.36,
                    "val_mean_t-AP": t_map,
                    "val_mean_t-AP_50": 0.35,
                    "val_mean_t-AP_25": 0.45,
                }
            )


def test_full_validation_trajectory_joins_resume_logger_versions(tmp_path) -> None:
    first = tmp_path / "local_metrics/version_0/metrics.csv"
    resumed = tmp_path / "local_metrics/version_1/metrics.csv"
    _write_metrics(first, VALIDATION_EPOCHS[:6])
    _write_metrics(resumed, VALIDATION_EPOCHS[6:])

    result = read_full_validation_trajectory([first, resumed])

    assert [row["completed_epoch"] for row in result["rows"]] == list(VALIDATION_EPOCHS)
    assert result["rows"][20]["t_mAP"] == pytest.approx(0.40)
    assert set(result["sources"]) == {
        "local_metrics/version_0/metrics.csv",
        "local_metrics/version_1/metrics.csv",
    }

    _write_metrics(resumed, VALIDATION_EPOCHS[5:])
    with pytest.raises(RootCauseEvaluationError, match="duplicate"):
        read_full_validation_trajectory([first, resumed])


def test_full_checkpoint_selection_matches_exact_validation_maximum(tmp_path) -> None:
    metrics = tmp_path / "metrics.csv"
    _write_metrics(metrics, VALIDATION_EPOCHS)
    rows = read_full_validation_trajectory([metrics])["rows"]
    records = [
        {
            "role": "best_validation",
            "completed_epoch": 315,
            "selected_step": 20_790,
            "selection_metric_exact": 0.40,
            "sha256": "a" * 64,
            "bytes": 100,
        },
        {
            "role": "exact_full_boundary",
            "completed_epoch": 450,
            "selected_step": 29_700,
            "sha256": "b" * 64,
            "bytes": 101,
        },
    ]

    selection = select_full_checkpoint(rows, records)

    assert selection["selected_epoch"] == 315
    assert selection["selection_metric_exact"] == pytest.approx(0.40)
    assert selection["full_budget_checkpoint_sha256"] == "b" * 64

    records[0]["selection_metric_exact"] = 0.39
    with pytest.raises(RootCauseEvaluationError, match="highest validation"):
        select_full_checkpoint(rows, records)


def test_full_training_manifest_binds_resume_decision_and_selection() -> None:
    decision = {
        "content_sha256": "1" * 64,
        "experiment": "rescene_strong_local_v1",
        "selected_variant": "R1",
        "full_training_authorized": True,
    }
    resume = {
        "content_sha256": "2" * 64,
        "variant": "R1",
        "candidate_id": "3" * 64,
        "runtime_selector_exact_match": True,
        "completed_epoch": 90,
        "selected_step": 5_940,
    }
    selection = {
        "monitor": "val_mean_t-AP",
        "mode": "max",
        "validation_event_count": 30,
        "selected_epoch": 315,
        "selected_step": 20_790,
        "selection_metric_exact": 0.40,
        "selected_checkpoint_sha256": "4" * 64,
        "selected_checkpoint_bytes": 100,
        "full_budget_checkpoint_sha256": "5" * 64,
        "full_budget_checkpoint_bytes": 101,
    }
    manifest = build_full_training_manifest(
        variant="R1",
        candidate_id="3" * 64,
        authorization_sha256="6" * 64,
        config_sha256="7" * 64,
        decision=decision,
        resume_plan=resume,
        selection=selection,
        validation_sources={"metrics.csv": {"bytes": 10, "sha256": "8" * 64}},
    )

    assert manifest["status"] == "pass"
    assert manifest["experiment"] == "rescene_strong_local_v1"
    assert manifest["budget"]["completed_epoch"] == 450
    assert manifest["budget"]["optimizer_steps"] == 29_700
    assert manifest["selection"]["selected_epoch"] == 315
    assert manifest["content_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )

    broken = json.loads(json.dumps(resume))
    broken["selected_step"] = 5_939
    with pytest.raises(RootCauseEvaluationError, match="resume"):
        build_full_training_manifest(
            variant="R1",
            candidate_id="3" * 64,
            authorization_sha256="6" * 64,
            config_sha256="7" * 64,
            decision=decision,
            resume_plan=broken,
            selection=selection,
            validation_sources={"metrics.csv": {"bytes": 10, "sha256": "8" * 64}},
        )
