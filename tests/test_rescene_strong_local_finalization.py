from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_rescene_rootcause_training import PER_SEED_FIELDS
from scripts.finalize_rescene_strong_local import build_strong_outputs
from scripts.summarize_rescene_rootcause_curves import (
    CSV_FIELDS as LEARNING_CURVE_FIELDS,
)
from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    RootCauseEvaluationError,
    build_checkpoint_manifest,
)
from utils.rescene_rootcause_preflight import canonical_sha256
from utils.rescene_strong_local import decide_strong_result

VALIDATION_EPOCHS = (15, 30, 45, 60, 75, 90)


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _metrics(gain: float, seed: int) -> dict[str, float]:
    offset = {45: 0.0, 46: 0.001, 47: -0.001}[seed]
    return {
        "t_mAP": 0.20 + gain + offset,
        "t_mAP50": 0.30 + gain + offset,
        "t_mAP25": 0.40 + gain + offset,
        "overall_mAP": 0.35 + gain + offset,
        "stage1_mAP": 0.25 + gain + offset,
        "stage2_mAP": 0.27 + gain + offset,
    }


def _root_decision(path: Path) -> None:
    runs = {seed: _metrics(0.0, seed) for seed in EVALUATION_SEEDS}
    summary = decide_strong_result(
        variant="A1",
        base_runs=runs,
        variant_runs=runs,
        validation_leads={75: True, 90: True},
        contract_integrity=True,
    )["base_summary"]
    payload = {
        "schema_version": 1,
        "status": "pass",
        "selected_variant": None,
        "epoch60_summary": {"R1": summary},
        "epoch90_summary": {"R1": summary},
    }
    payload["content_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="ascii")


def _root_curves(path: Path) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEARNING_CURVE_FIELDS)
        writer.writeheader()
        for epoch in VALIDATION_EPOCHS:
            writer.writerow(
                {
                    "variant": "R1",
                    "completed_epoch": epoch,
                    "optimizer_step": epoch * 66,
                    "train_log_step": epoch * 66 - 1,
                    "stage1_mAP": 0.20 + epoch / 1000,
                    "stage2_mAP": 0.22 + epoch / 1000,
                    "overall_mAP": 0.30 + epoch / 1000,
                    "t_mAP": 0.18 + epoch / 1000,
                    "t_mAP50": 0.28 + epoch / 1000,
                    "t_mAP25": 0.38 + epoch / 1000,
                    "SpatialStageMean": 0.21 + epoch / 1000,
                    "candidate_id": "1" * 64,
                    "config_sha256": "2" * 64,
                    "variant_authorization_sha256": "3" * 64,
                    "metrics_csv_sha256": "4" * 64,
                }
            )


def _root_official(path: Path, epoch: int) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SEED_FIELDS)
        writer.writeheader()
        for seed in EVALUATION_SEEDS:
            metrics = _metrics(0.0, seed)
            writer.writerow(
                {
                    "variant": "R1",
                    "completed_epoch": epoch,
                    "seed": seed,
                    **metrics,
                    "SpatialStageMean": (metrics["stage1_mAP"] + metrics["stage2_mAP"])
                    / 2.0,
                    "validation_sequence_count": 154,
                    "checkpoint_sha256": "5" * 64,
                    "elapsed_seconds": 1.0,
                }
            )


def _authorization(
    path: Path,
    *,
    root_decision: Path,
    root_curves: Path,
    root_epoch60: Path,
    root_epoch90: Path,
) -> dict[str, object]:
    config = {"model": {"use_np_features": True, "scatter_type": "mean"}}
    payload = {
        "schema_version": 1,
        "status": "authorized",
        "source_commit": "6" * 40,
        "experiment": "rescene_strong_local_v1",
        "checkpoint_namespace": "rescene_strong_local",
        "selected_variants": ["A1"],
        "base_variant": "R1",
        "upstream_evidence": {
            "short_decision": _identity(root_decision),
            "root_learning_curves": _identity(root_curves),
            "root_official_like_epoch60": _identity(root_epoch60),
            "root_official_like_epoch90": _identity(root_epoch90),
        },
        "initialization": {
            "common_state": {"sha256": "7" * 64},
            "pretrained": {"sha256": "8" * 64},
        },
        "schedule": {
            "optimizer_steps_per_epoch": 66,
            "total_optimizer_steps": 29_700,
        },
        "variants": {
            "A1": {
                "config_sha256": canonical_sha256(config),
                "resolved_config": config,
                "expected_state_dict_entries": 802,
            }
        },
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="ascii")
    return payload


def _candidate(authorization: dict[str, object]) -> dict[str, object]:
    payload = {
        "variant": "A1",
        "source_commit": authorization["source_commit"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": authorization["variants"]["A1"]["config_sha256"],
        "common_initialization_sha256": "7" * 64,
        "pretrained_sha256": "8" * 64,
    }
    payload["candidate_id"] = canonical_sha256(payload)
    return payload


def _strong_curve(root: Path, authorization: dict[str, object]) -> Path:
    run = root / "run"
    metrics_path = run / "local_metrics/version_0/metrics.csv"
    metrics_path.parent.mkdir(parents=True)
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
    with metrics_path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in VALIDATION_EPOCHS:
            writer.writerow(
                {
                    "epoch": epoch - 1,
                    "step": epoch * 66 - 1,
                    "val_mean_stage1-AP": 0.215 + epoch / 1000,
                    "val_mean_stage2-AP": 0.235 + epoch / 1000,
                    "val_mean_AP": 0.315 + epoch / 1000,
                    "val_mean_t-AP": 0.195 + epoch / 1000,
                    "val_mean_t-AP_50": 0.295 + epoch / 1000,
                    "val_mean_t-AP_25": 0.395 + epoch / 1000,
                }
            )
    (run / ".rootcause_candidate.json").write_text(
        json.dumps(_candidate(authorization)), encoding="ascii"
    )
    return run


def _strong_evaluation(root: Path, authorization: dict[str, object]) -> Path:
    evaluation = root / "evaluation"
    candidate = _candidate(authorization)
    for epoch in (60, 90):
        output = evaluation / "A1" / f"epoch{epoch:03d}"
        output.mkdir(parents=True)
        facts = {
            "selected_epoch": epoch,
            "selected_step": epoch * 66,
            "training_config_sha256": authorization["variants"]["A1"]["config_sha256"],
        }
        manifest = build_checkpoint_manifest(
            variant="A1",
            completed_epoch=epoch,
            authorization=authorization,
            candidate=candidate,
            file_identity={"bytes": 100, "sha256": "9" * 64},
            checkpoint_facts=facts,
        )
        (output / "checkpoint_manifest.json").write_text(
            json.dumps(manifest), encoding="ascii"
        )
        for seed in EVALUATION_SEEDS:
            metrics = _metrics(0.015, seed)
            run = {
                "status": "pass",
                "scope": "official_like_t2",
                "variant": "A1",
                "completed_epoch": epoch,
                "seed": seed,
                "variant_authorization_sha256": authorization["authorization_sha256"],
                "checkpoint_manifest_sha256": manifest["content_sha256"],
                "checkpoint_sha256": "9" * 64,
                "validation_sequence_count": 154,
                "metrics": metrics,
                "SpatialStageMean": (metrics["stage1_mAP"] + metrics["stage2_mAP"])
                / 2.0,
                "elapsed_seconds": 1.0,
            }
            (output / f"seed{seed}.json").write_text(json.dumps(run), encoding="ascii")
    return evaluation


def _study(tmp_path: Path):
    decision = tmp_path / "decision.json"
    curves = tmp_path / "learning_curves.csv"
    epoch60 = tmp_path / "epoch60.csv"
    epoch90 = tmp_path / "epoch90.csv"
    _root_decision(decision)
    _root_curves(curves)
    _root_official(epoch60, 60)
    _root_official(epoch90, 90)
    authorization_path = tmp_path / "authorization.json"
    authorization = _authorization(
        authorization_path,
        root_decision=decision,
        root_curves=curves,
        root_epoch60=epoch60,
        root_epoch90=epoch90,
    )
    run = _strong_curve(tmp_path, authorization)
    evaluation = _strong_evaluation(tmp_path, authorization)
    return authorization_path, decision, curves, epoch60, epoch90, run, evaluation


def test_strong_finalization_authorizes_full_run_from_exact_spatial_gate(
    tmp_path,
) -> None:
    authorization, decision, curves, epoch60, epoch90, run, evaluation = _study(
        tmp_path
    )

    outputs = build_strong_outputs(
        authorization_path=authorization,
        root_decision_path=decision,
        root_learning_curves_path=curves,
        root_epoch60_path=epoch60,
        root_epoch90_path=epoch90,
        run_directory=run,
        evaluation_root=evaluation,
    )

    verdict = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])
    assert verdict["variant"] == "A1"
    assert verdict["base_variant"] == "R1"
    assert verdict["all_gates_pass"] is True
    assert verdict["full_training_status"] == "authorized"
    assert verdict["selection_used_persist4d"] is False
    assert outputs["learning_curves.csv"].count(b"\n") == 13
    assert outputs["official_like_per_seed.csv"].count(b"\n") == 13


def test_strong_finalization_joins_resumed_logger_versions(tmp_path) -> None:
    authorization, decision, curves, epoch60, epoch90, run, evaluation = _study(
        tmp_path
    )
    original = run / "local_metrics/version_0/metrics.csv"
    rows = list(csv.DictReader(original.open(encoding="ascii")))
    fields = tuple(rows[0])
    original.unlink()
    for version, selected in enumerate((rows[:2], rows[2:])):
        path = run / f"local_metrics/version_{version}/metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)

    outputs = build_strong_outputs(
        authorization_path=authorization,
        root_decision_path=decision,
        root_learning_curves_path=curves,
        root_epoch60_path=epoch60,
        root_epoch90_path=epoch90,
        run_directory=run,
        evaluation_root=evaluation,
    )

    provenance = json.loads(outputs["STRONG_LOCAL_PROVENANCE.json"])
    assert set(provenance["strong_curve_sources"]["metrics"]) == {
        "local_metrics/version_0/metrics.csv",
        "local_metrics/version_1/metrics.csv",
    }


def test_strong_finalization_rejects_rebound_root_evidence(tmp_path) -> None:
    authorization, decision, curves, epoch60, epoch90, run, evaluation = _study(
        tmp_path
    )
    with curves.open("a", encoding="ascii") as handle:
        handle.write("tampered\n")

    with pytest.raises(RootCauseEvaluationError, match="identity"):
        build_strong_outputs(
            authorization_path=authorization,
            root_decision_path=decision,
            root_learning_curves_path=curves,
            root_epoch60_path=epoch60,
            root_epoch90_path=epoch90,
            run_directory=run,
            evaluation_root=evaluation,
        )
