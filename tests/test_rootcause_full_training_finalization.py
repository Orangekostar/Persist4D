from __future__ import annotations

import csv
import json

import pytest

from scripts.finalize_rescene_rootcause_full_training import (
    build_full_training_manifest,
    main,
    read_full_validation_trajectory,
    select_full_checkpoint,
)
from utils.rescene_rootcause_evaluation import RootCauseEvaluationError
from utils.rescene_rootcause_preflight import canonical_sha256

VALIDATION_EPOCHS = tuple(range(15, 451, 15))


def _write_metrics(path, epochs, *, t_map_delta: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "epoch",
        "step",
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
            t_map = (
                0.40 if completed_epoch == 315 else 0.20 + completed_epoch / 10_000
            ) + t_map_delta
            writer.writerow(
                {
                    "epoch": completed_epoch - 1,
                    "step": completed_epoch * 66 - 1,
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


def test_full_validation_trajectory_uses_signed_recovery_cutover(tmp_path) -> None:
    original = tmp_path / "local_metrics/version_2/metrics.csv"
    recovered = tmp_path / "local_metrics/version_3/metrics.csv"
    _write_metrics(original, VALIDATION_EPOCHS[:27])
    _write_metrics(recovered, VALIDATION_EPOCHS[26:], t_map_delta=0.01)
    lineage = {
        "schema_version": 1,
        "status": "pass",
        "reason": "cuda_allocator_fragmentation_recovery",
        "resume_checkpoint": {
            "completed_epoch": 390,
            "optimizer_step": 25_740,
            "sha256": "9" * 64,
        },
        "authoritative_source": "local_metrics/version_3/metrics.csv",
        "superseded_sources": ["local_metrics/version_2/metrics.csv"],
    }
    lineage["content_sha256"] = canonical_sha256(lineage)

    with pytest.raises(RootCauseEvaluationError, match="duplicate"):
        read_full_validation_trajectory([original, recovered])

    result = read_full_validation_trajectory(
        [original, recovered], recovery_lineage=lineage
    )

    selected = result["rows"][26]
    assert selected["completed_epoch"] == 405
    assert selected["t_mAP"] == pytest.approx(0.2505)
    assert (
        selected["metrics_csv_sha256"]
        == result["sources"]["local_metrics/version_3/metrics.csv"]["sha256"]
    )
    assert result["recovery_lineage"]["policy"] == lineage
    assert result["recovery_lineage"]["superseded_validation_rows"] == [
        {
            "completed_epoch": 405,
            "optimizer_step": 26_730,
            "train_log_step": 26_729,
            "stage1_mAP": 0.30,
            "stage2_mAP": 0.32,
            "overall_mAP": 0.36,
            "t_mAP": 0.24050000000000002,
            "t_mAP50": 0.35,
            "t_mAP25": 0.45,
            "SpatialStageMean": 0.31,
            "metrics_csv_sha256": result["sources"][
                "local_metrics/version_2/metrics.csv"
            ]["sha256"],
            "source": "local_metrics/version_2/metrics.csv",
        }
    ]


def test_full_validation_trajectory_rejects_tampered_recovery_cutover(
    tmp_path,
) -> None:
    metrics = tmp_path / "local_metrics/version_3/metrics.csv"
    _write_metrics(metrics, VALIDATION_EPOCHS)
    lineage = {
        "schema_version": 1,
        "status": "pass",
        "reason": "cuda_allocator_fragmentation_recovery",
        "resume_checkpoint": {
            "completed_epoch": 390,
            "optimizer_step": 25_740,
            "sha256": "9" * 64,
        },
        "authoritative_source": "local_metrics/version_3/metrics.csv",
        "superseded_sources": [],
    }
    lineage["content_sha256"] = "0" * 64

    with pytest.raises(RootCauseEvaluationError, match="lineage.*hash"):
        read_full_validation_trajectory([metrics], recovery_lineage=lineage)


def test_full_training_cli_accepts_recovery_lineage_option(tmp_path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(RootCauseEvaluationError, match="variant authorization"):
        main(
            [
                "--variant",
                "R1",
                "--run-directory",
                str(tmp_path),
                "--authorization",
                str(missing),
                "--decision",
                str(missing),
                "--resume-plan",
                str(missing),
                "--recovery-lineage",
                str(missing),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


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
        "variant_authorization_sha256": "6" * 64,
    }
    resume = {
        "content_sha256": "2" * 64,
        "variant": "R1",
        "candidate_id": "3" * 64,
        "short_decision_sha256": "1" * 64,
        "variant_authorization_sha256": "6" * 64,
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
    assert manifest["decision_authorization_sha256"] == "6" * 64
    assert "runtime_migration" not in manifest
    assert manifest["content_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )

    policy = {
        "schema_version": 1,
        "status": "pass",
        "reason": "cuda_allocator_fragmentation_recovery",
        "resume_checkpoint": {
            "completed_epoch": 390,
            "optimizer_step": 25_740,
            "sha256": "9" * 64,
        },
        "authoritative_source": "local_metrics/version_3/metrics.csv",
        "superseded_sources": ["local_metrics/version_2/metrics.csv"],
    }
    policy["content_sha256"] = canonical_sha256(policy)
    validation_lineage = {
        "policy": policy,
        "superseded_validation_rows": [
            {
                "completed_epoch": 405,
                "optimizer_step": 26_730,
                "train_log_step": 26_729,
                "stage1_mAP": 0.30,
                "stage2_mAP": 0.32,
                "overall_mAP": 0.36,
                "t_mAP": 0.24,
                "t_mAP50": 0.35,
                "t_mAP25": 0.45,
                "SpatialStageMean": 0.31,
                "metrics_csv_sha256": "8" * 64,
                "source": "local_metrics/version_2/metrics.csv",
            }
        ],
    }
    replay_manifest = build_full_training_manifest(
        variant="R1",
        candidate_id="3" * 64,
        authorization_sha256="6" * 64,
        config_sha256="7" * 64,
        decision=decision,
        resume_plan=resume,
        selection=selection,
        validation_sources={
            "local_metrics/version_2/metrics.csv": {
                "bytes": 10,
                "sha256": "8" * 64,
            },
            "local_metrics/version_3/metrics.csv": {
                "bytes": 11,
                "sha256": "9" * 64,
            },
        },
        validation_lineage=validation_lineage,
    )

    assert replay_manifest["validation_lineage"] == validation_lineage
    assert replay_manifest["content_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in replay_manifest.items()
            if key != "content_sha256"
        }
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


def test_full_training_manifest_binds_runtime_only_authorization_migration() -> None:
    decision = {
        "content_sha256": "1" * 64,
        "selected_variant": "R1",
        "full_training_authorized": True,
        "variant_authorization_sha256": "5" * 64,
    }
    runtime_migration = {
        "changed_fields": ["runtime.gpu_count", "runtime.gpu_models"],
        "decision_authorization_sha256": "5" * 64,
        "runtime_authorization_sha256": "6" * 64,
        "provenance": {"bytes": 100, "sha256": "9" * 64},
    }
    resume = {
        "content_sha256": "2" * 64,
        "variant": "R1",
        "candidate_id": "3" * 64,
        "short_decision_sha256": "1" * 64,
        "variant_authorization_sha256": "6" * 64,
        "decision_authorization_sha256": "5" * 64,
        "runtime_migration": runtime_migration,
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
        "full_budget_checkpoint_sha256": "7" * 64,
        "full_budget_checkpoint_bytes": 101,
    }

    manifest = build_full_training_manifest(
        variant="R1",
        candidate_id="3" * 64,
        authorization_sha256="6" * 64,
        config_sha256="8" * 64,
        decision=decision,
        resume_plan=resume,
        selection=selection,
        validation_sources={"metrics.csv": {"bytes": 10, "sha256": "a" * 64}},
    )

    assert manifest["decision_authorization_sha256"] == "5" * 64
    assert manifest["runtime_migration"] == runtime_migration

    resume["runtime_migration"]["changed_fields"] = ["runtime.gpu_count"]
    with pytest.raises(RootCauseEvaluationError, match="authorization migration"):
        build_full_training_manifest(
            variant="R1",
            candidate_id="3" * 64,
            authorization_sha256="6" * 64,
            config_sha256="8" * 64,
            decision=decision,
            resume_plan=resume,
            selection=selection,
            validation_sources={
                "metrics.csv": {"bytes": 10, "sha256": "a" * 64}
            },
        )
