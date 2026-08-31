from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.finalize_rescene_rootcause_training import (
    _validate_infeasible_variants,
    build_short_curve_outputs,
)
from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    RootCauseEvaluationError,
    build_checkpoint_manifest,
)
from utils.rescene_rootcause_preflight import canonical_sha256

VARIANTS = ("R0", "R1")
VALIDATION_EPOCHS = (15, 30, 45, 60, 75, 90)


def _authorization(variants: tuple[str, ...] = VARIANTS) -> dict[str, object]:
    payload = {
        "status": "authorized",
        "source_commit": "1" * 40,
        "selected_variants": list(variants),
        "variants": {
            variant: {"config_sha256": str(index + 2) * 64}
            for index, variant in enumerate(variants)
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


def _write_curve(
    root: Path,
    variant: str,
    authorization: dict[str, object],
    *,
    offset: float,
) -> Path:
    run = root / "runs" / variant
    metrics = run / "local_metrics/version_0/metrics.csv"
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
    )
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in VALIDATION_EPOCHS:
            writer.writerow(
                {
                    "epoch": epoch - 1,
                    "step": epoch * 66 - 1,
                    "val_mean_stage1-AP": 0.20 + epoch / 1000 + offset,
                    "val_mean_stage2-AP": 0.22 + epoch / 1000 + offset,
                    "val_mean_AP": 0.30 + epoch / 1000 + offset,
                    "val_mean_t-AP": 0.18 + epoch / 1000 + offset,
                    "val_mean_t-AP_50": 0.28 + epoch / 1000 + offset,
                    "val_mean_t-AP_25": 0.38 + epoch / 1000 + offset,
                }
            )
    (run / ".rootcause_candidate.json").write_text(
        json.dumps(_candidate(variant, authorization)), encoding="utf-8"
    )
    return run


def _write_evaluation(
    root: Path,
    variant: str,
    epoch: int,
    authorization: dict[str, object],
    *,
    spatial_offset: float,
) -> None:
    candidate = _candidate(variant, authorization)
    checkpoint_sha = ("a" if variant == "R0" else "b") * 64
    facts = {
        "selected_epoch": epoch,
        "selected_step": epoch * 66,
        "training_config_sha256": authorization["variants"][variant]["config_sha256"],
    }
    manifest = build_checkpoint_manifest(
        variant=variant,
        completed_epoch=epoch,
        authorization=authorization,
        candidate=candidate,
        file_identity={"bytes": 100, "sha256": checkpoint_sha},
        checkpoint_facts=facts,
    )
    output = root / variant / f"epoch{epoch:03d}"
    output.mkdir(parents=True)
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for seed, seed_offset in zip(EVALUATION_SEEDS, (0.0, 0.001, -0.001)):
        stage1 = 0.30 + spatial_offset + seed_offset
        stage2 = 0.32 + spatial_offset + seed_offset
        metrics = {
            "t_mAP": 0.25 + spatial_offset + seed_offset,
            "t_mAP50": 0.35 + spatial_offset + seed_offset,
            "t_mAP25": 0.45 + spatial_offset + seed_offset,
            "overall_mAP": 0.36 + spatial_offset + seed_offset,
            "stage1_mAP": stage1,
            "stage2_mAP": stage2,
        }
        run = {
            "schema_version": 1,
            "status": "pass",
            "scope": "official_like_t2",
            "variant": variant,
            "completed_epoch": epoch,
            "seed": seed,
            "source_commit": "6" * 40,
            "contract_sha256": "7" * 64,
            "variant_authorization_sha256": authorization["authorization_sha256"],
            "checkpoint_manifest_sha256": manifest["content_sha256"],
            "checkpoint_sha256": checkpoint_sha,
            "evaluation_config_sha256": "8" * 64,
            "validation_sequence_count": 154,
            "metrics": metrics,
            "SpatialStageMean": (stage1 + stage2) / 2,
            "elapsed_seconds": 10.0,
        }
        (output / f"seed{seed}.json").write_text(json.dumps(run), encoding="utf-8")


def _study(tmp_path: Path, *, gain: float = 0.015):
    authorization = _authorization()
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    runs = {
        "R0": _write_curve(tmp_path, "R0", authorization, offset=0.0),
        "R1": _write_curve(tmp_path, "R1", authorization, offset=gain),
    }
    evaluation_root = tmp_path / "evaluation"
    for epoch in (60, 90):
        _write_evaluation(
            evaluation_root,
            "R0",
            epoch,
            authorization,
            spatial_offset=0.0,
        )
        _write_evaluation(
            evaluation_root,
            "R1",
            epoch,
            authorization,
            spatial_offset=gain,
        )
    return authorization_path, runs, evaluation_root


def _write_infeasible(
    root: Path,
    variant: str,
    authorization: dict[str, object],
) -> Path:
    candidate = _candidate(variant, authorization)
    attempts = []
    for index, allocator in enumerate(("default", "expandable_segments"), start=1):
        relative = Path("failed_attempts") / f"{variant}_attempt{index}"
        attempt_root = root / relative
        attempt_root.mkdir(parents=True)
        log = f"{variant} {allocator} CUDA out of memory\n".encode("ascii")
        (attempt_root / f"{variant}.launch.log").write_bytes(log)
        (attempt_root / ".rootcause_candidate.json").write_text(
            json.dumps(candidate), encoding="utf-8"
        )
        failure = {
            "schema_version": 1,
            "status": "failed",
            "variant": variant,
            "candidate_id": candidate["candidate_id"],
            "config_sha256": candidate["config_sha256"],
            "common_initialization_sha256": candidate[
                "common_initialization_sha256"
            ],
            "failed_batch_index": 254,
            "failure": "CUDA out of memory",
            "launch_log_sha256": hashlib.sha256(log).hexdigest(),
        }
        evidence = attempt_root / "FAILURE.json"
        evidence.write_text(json.dumps(failure), encoding="utf-8")
        attempts.append(
            {
                "allocator": allocator,
                "evidence": relative.joinpath("FAILURE.json").as_posix(),
                "launch_log_sha256": failure["launch_log_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "status": "infeasible",
        "variant": variant,
        "reason_code": "full_dataset_cuda_oom",
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "candidate_id": candidate["candidate_id"],
        "config_sha256": candidate["config_sha256"],
        "common_initialization_sha256": candidate["common_initialization_sha256"],
        "physical_global_batch": 8,
        "effective_global_batch": 32,
        "failed_batch_index": 254,
        "attempts": attempts,
        "replacement_variant": None,
    }
    path = root / variant / "INFEASIBLE.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_nonfinite_infeasible(
    root: Path,
    variant: str,
    authorization: dict[str, object],
) -> Path:
    candidate = _candidate(variant, authorization)
    failed_file_names = [
        "scene0433_01-scene0433_03",
        "scene0433_03-scene0433_00",
    ]
    attempts = []
    for index in (1, 2):
        relative = Path("failed_attempts") / f"{variant}_attempt{index}"
        attempt_root = root / relative
        attempt_root.mkdir(parents=True)
        log = (
            f"{variant} batch=412 non-finite raw objective term 'loss_ce'\n"
        ).encode("ascii")
        (attempt_root / f"{variant}.launch.log").write_bytes(log)
        (attempt_root / ".rootcause_candidate.json").write_text(
            json.dumps(candidate), encoding="utf-8"
        )
        failure = {
            "schema_version": 1,
            "status": "failed",
            "variant": variant,
            "candidate_id": candidate["candidate_id"],
            "config_sha256": candidate["config_sha256"],
            "common_initialization_sha256": candidate[
                "common_initialization_sha256"
            ],
            "completed_epoch": 2,
            "failed_batch_index": 412,
            "failed_rank": 1,
            "failed_file_names": failed_file_names,
            "failure_kind": "nonfinite_objective",
            "objective_term": "loss_ce",
            "launch_log_sha256": hashlib.sha256(log).hexdigest(),
        }
        evidence = attempt_root / "FAILURE.json"
        evidence.write_text(json.dumps(failure), encoding="utf-8")
        attempts.append(
            {
                "attempt_id": f"same_authorized_config_{index}",
                "evidence": relative.joinpath("FAILURE.json").as_posix(),
                "launch_log_sha256": failure["launch_log_sha256"],
            }
        )
    payload = {
        "schema_version": 1,
        "status": "infeasible",
        "variant": variant,
        "reason_code": "deterministic_nonfinite_objective",
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "candidate_id": candidate["candidate_id"],
        "config_sha256": candidate["config_sha256"],
        "common_initialization_sha256": candidate[
            "common_initialization_sha256"
        ],
        "failed_batch_index": 412,
        "failed_rank": 1,
        "failed_file_names": failed_file_names,
        "objective_term": "loss_ce",
        "attempts": attempts,
        "replacement_variant": None,
    }
    path = root / variant / "INFEASIBLE.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_short_finalization_builds_required_outputs_and_selects_one(tmp_path) -> None:
    authorization, runs, evaluation_root = _study(tmp_path)

    outputs = build_short_curve_outputs(
        run_directories=runs,
        authorization_path=authorization,
        evaluation_root=evaluation_root,
    )

    assert {
        "learning_curves.csv",
        "official_like_epoch60.csv",
        "official_like_epoch90.csv",
        "rootcause_per_seed.csv",
        "rootcause_summary.csv",
        "ROOTCAUSE_SHORT_DECISION.json",
        "ROOTCAUSE_SHORT_DECISION.md",
        "ROOTCAUSE_SHORT_PROVENANCE.json",
    } == set(outputs)
    decision = json.loads(outputs["ROOTCAUSE_SHORT_DECISION.json"])
    assert decision["selected_variant"] == "R1"
    assert decision["decisions"]["R1"]["all_gates_pass"] is True
    assert decision["full_training_authorized"] is True
    assert outputs["rootcause_per_seed.csv"].decode("ascii").count("\n") == 13
    assert outputs["rootcause_summary.csv"].decode("ascii").count("\n") == 5


def test_short_finalization_excludes_authorized_runtime_infeasible_variant(
    tmp_path,
) -> None:
    variants = ("R0", "R1", "R2")
    authorization = _authorization(variants)
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
    runs = {
        "R0": _write_curve(tmp_path, "R0", authorization, offset=0.0),
        "R1": _write_curve(tmp_path, "R1", authorization, offset=0.015),
    }
    evaluation_root = tmp_path / "evaluation"
    for epoch in (60, 90):
        _write_evaluation(
            evaluation_root, "R0", epoch, authorization, spatial_offset=0.0
        )
        _write_evaluation(
            evaluation_root, "R1", epoch, authorization, spatial_offset=0.015
        )
    infeasible = {"R2": _write_infeasible(tmp_path, "R2", authorization)}

    outputs = build_short_curve_outputs(
        run_directories=runs,
        infeasible_variants=infeasible,
        authorization_path=authorization_path,
        evaluation_root=evaluation_root,
    )

    decision = json.loads(outputs["ROOTCAUSE_SHORT_DECISION.json"])
    provenance = json.loads(outputs["ROOTCAUSE_SHORT_PROVENANCE.json"])
    assert decision["selected_variant"] == "R1"
    assert decision["completed_variants"] == ["R0", "R1"]
    assert decision["infeasible_variants"] == {
        "R2": {
            "candidate_id": _candidate("R2", authorization)["candidate_id"],
            "reason_code": "full_dataset_cuda_oom",
        }
    }
    assert set(provenance["infeasible_sources"]) == {
        "R2/INFEASIBLE.json",
        "R2/attempt1/FAILURE.json",
        "R2/attempt1/.rootcause_candidate.json",
        "R2/attempt1/R2.launch.log",
        "R2/attempt2/FAILURE.json",
        "R2/attempt2/.rootcause_candidate.json",
        "R2/attempt2/R2.launch.log",
    }
    assert outputs["rootcause_per_seed.csv"].decode("ascii").count("\n") == 13
    assert outputs["rootcause_summary.csv"].decode("ascii").count("\n") == 5


def test_runtime_infeasible_evidence_resolves_from_symlinked_variant_root(
    tmp_path,
) -> None:
    authorization = _authorization(("R0", "R1", "R2"))
    live_root = tmp_path / "live"
    live_record = _write_infeasible(live_root, "R2", authorization)
    shared_root = tmp_path / "shared"
    shared_root.mkdir()
    (shared_root / "R2").symlink_to(live_record.parent, target_is_directory=True)

    records, sources = _validate_infeasible_variants(
        infeasible_variants={"R2": shared_root / "R2/INFEASIBLE.json"},
        authorization=authorization,
    )

    assert records["R2"]["reason_code"] == "full_dataset_cuda_oom"
    assert len(sources) == 7


def test_short_finalization_accepts_two_distinct_runtime_infeasibility_modes(
    tmp_path: Path,
) -> None:
    authorization = _authorization(("R0", "R1", "R2", "R4"))
    records, sources = _validate_infeasible_variants(
        infeasible_variants={
            "R2": _write_infeasible(tmp_path, "R2", authorization),
            "R4": _write_nonfinite_infeasible(tmp_path, "R4", authorization),
        },
        authorization=authorization,
    )

    assert records == {
        "R2": {
            "candidate_id": _candidate("R2", authorization)["candidate_id"],
            "reason_code": "full_dataset_cuda_oom",
        },
        "R4": {
            "candidate_id": _candidate("R4", authorization)["candidate_id"],
            "reason_code": "deterministic_nonfinite_objective",
        },
    }
    assert len(sources) == 14


def test_nonfinite_infeasibility_rejects_attempt_semantic_mismatch(
    tmp_path: Path,
) -> None:
    authorization = _authorization(("R0", "R1", "R4"))
    record = _write_nonfinite_infeasible(tmp_path, "R4", authorization)
    evidence = tmp_path / "failed_attempts/R4_attempt2/FAILURE.json"
    failure = json.loads(evidence.read_text(encoding="utf-8"))
    failure["objective_term"] = "loss_mask"
    evidence.write_text(json.dumps(failure), encoding="utf-8")

    with pytest.raises(
        RootCauseEvaluationError,
        match="non-finite evidence binding differs",
    ):
        _validate_infeasible_variants(
            infeasible_variants={"R4": record},
            authorization=authorization,
        )


def test_short_finalization_rejects_manifest_or_run_rebinding(tmp_path) -> None:
    authorization, runs, evaluation_root = _study(tmp_path)
    manifest_path = evaluation_root / "R1/epoch090/checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint"]["sha256"] = "c" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RootCauseEvaluationError, match="hash|binding"):
        build_short_curve_outputs(
            run_directories=runs,
            authorization_path=authorization,
            evaluation_root=evaluation_root,
        )


def test_short_finalization_records_gate_skipped_without_candidate(tmp_path) -> None:
    authorization, runs, evaluation_root = _study(tmp_path, gain=0.005)

    outputs = build_short_curve_outputs(
        run_directories=runs,
        authorization_path=authorization,
        evaluation_root=evaluation_root,
    )

    decision = json.loads(outputs["ROOTCAUSE_SHORT_DECISION.json"])
    assert decision["selected_variant"] is None
    assert decision["full_training_authorized"] is False
    assert decision["full_training_status"] == "gate_skipped"
    assert b"gate_skipped" in outputs["ROOTCAUSE_SHORT_DECISION.md"]
