from __future__ import annotations

import csv
import json

import pytest

from scripts.summarize_rescene_rootcause_curves import (
    summarize_learning_curves,
)
from utils.rescene_rootcause_evaluation import RootCauseEvaluationError
from utils.rescene_rootcause_preflight import canonical_sha256

VALIDATION_EPOCHS = (15, 30, 45, 60, 75, 90)


def _authorization() -> dict[str, object]:
    payload = {
        "status": "authorized",
        "source_commit": "1" * 40,
        "selected_variants": ["R0", "R1"],
        "variants": {
            "R0": {"config_sha256": "2" * 64},
            "R1": {"config_sha256": "3" * 64},
        },
        "initialization": {
            "common_state": {"sha256": "4" * 64},
            "pretrained": {"sha256": "5" * 64},
        },
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def _candidate(variant: str, authorization: dict[str, object]) -> dict[str, object]:
    payload = {
        "variant": variant,
        "source_commit": authorization["source_commit"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": authorization["variants"][variant]["config_sha256"],
        "common_initialization_sha256": "4" * 64,
        "pretrained_sha256": "5" * 64,
    }
    payload["candidate_id"] = canonical_sha256(payload)
    return payload


def _write_run(tmp_path, variant: str, *, offset: float = 0.0):
    run = tmp_path / variant
    metrics = run / "local_metrics" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    fields = (
        "epoch",
        "step",
        "val_mean_stage1-AP",
        "val_mean_stage2-AP",
        "val_mean_AP",
        "val_mean_t-AP",
        "val_mean_t-AP_50",
        "val_mean_t-AP_25",
        "val_loss",
    )
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for completed_epoch in VALIDATION_EPOCHS:
            writer.writerow(
                {
                    "epoch": completed_epoch - 1,
                    "step": completed_epoch * 66 - 1,
                    "val_mean_stage1-AP": 0.2 + completed_epoch / 1000 + offset,
                    "val_mean_stage2-AP": 0.3 + completed_epoch / 1000 + offset,
                    "val_mean_AP": 0.25 + completed_epoch / 1000 + offset,
                    "val_mean_t-AP": 0.1 + completed_epoch / 1000 + offset,
                    "val_mean_t-AP_50": 0.2 + completed_epoch / 1000 + offset,
                    "val_mean_t-AP_25": 0.3 + completed_epoch / 1000 + offset,
                    "val_loss": 10.0 - completed_epoch / 100,
                }
            )
    return run, metrics


def test_learning_curve_summary_requires_exact_standard_checkpoints(tmp_path) -> None:
    authorization = _authorization()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    runs = {}
    for variant, offset in (("R0", 0.0), ("R1", 0.02)):
        run, _ = _write_run(tmp_path, variant, offset=offset)
        (run / ".rootcause_candidate.json").write_text(
            json.dumps(_candidate(variant, authorization)), encoding="utf-8"
        )
        runs[variant] = run

    result = summarize_learning_curves(
        run_directories=runs,
        authorization_path=authorization_path,
    )

    assert len(result["rows"]) == 12
    assert result["rows"][-1]["variant"] == "R1"
    assert result["rows"][-1]["completed_epoch"] == 90
    assert result["rows"][-1]["optimizer_step"] == 5_940
    assert result["rows"][-1]["SpatialStageMean"] == pytest.approx(0.36)
    assert result["validation_leads"]["R1"] == {75: True, 90: True}
    assert len(result["sources"]["R0"]["metrics_csv_sha256"]) == 64


def test_learning_curve_summary_rejects_missing_or_stale_rows(tmp_path) -> None:
    authorization = _authorization()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    run, metrics = _write_run(tmp_path, "R0")
    (run / ".rootcause_candidate.json").write_text(
        json.dumps(_candidate("R0", authorization)), encoding="utf-8"
    )

    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows[:-1])
    with pytest.raises(RootCauseEvaluationError, match="validation checkpoints"):
        summarize_learning_curves(
            run_directories={"R0": run},
            authorization_path=authorization_path,
        )


def test_learning_curve_summary_rejects_candidate_or_step_mismatch(tmp_path) -> None:
    authorization = _authorization()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    run, metrics = _write_run(tmp_path, "R0")
    candidate = _candidate("R0", authorization)
    candidate["config_sha256"] = "9" * 64
    (run / ".rootcause_candidate.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    with pytest.raises(RootCauseEvaluationError, match="candidate"):
        summarize_learning_curves(
            run_directories={"R0": run},
            authorization_path=authorization_path,
        )

    (run / ".rootcause_candidate.json").write_text(
        json.dumps(_candidate("R0", authorization)), encoding="utf-8"
    )
    rows = list(csv.DictReader(metrics.open(encoding="utf-8")))
    rows[-1]["step"] = "5938"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RootCauseEvaluationError, match="step"):
        summarize_learning_curves(
            run_directories={"R0": run},
            authorization_path=authorization_path,
        )
