from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def test_runtime_callback_records_completed_epoch_evidence(tmp_path: Path) -> None:
    from utils.sonata_training_evidence import SonataTrainingEvidenceCallback

    callback = SonataTrainingEvidenceCallback(output_dir=str(tmp_path))
    trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        global_step=66,
        callback_metrics={
            "train_loss_epoch": torch.tensor(12.5),
            "nonfinite": torch.tensor(float("nan")),
        },
        optimizers=[SimpleNamespace(param_groups=[{"lr": 4.0e-5}])],
    )
    target = [
        {"temporal_stages": torch.tensor([0, 0, 1])},
        {"temporal_stages": torch.tensor([0])},
    ]
    batch = (SimpleNamespace(), target, ["a", "b"])

    callback.on_fit_start(trainer, SimpleNamespace())
    callback.on_train_epoch_start(trainer, SimpleNamespace())
    callback.on_train_batch_end(trainer, SimpleNamespace(), None, batch, 0)
    callback.on_train_epoch_end(trainer, SimpleNamespace())
    callback.on_fit_end(trainer, SimpleNamespace())

    events = _events(tmp_path / ".sonata_runtime_events.jsonl")
    assert [event["event"] for event in events] == [
        "fit_started",
        "epoch_completed",
        "fit_completed",
    ]
    epoch = events[1]
    assert epoch["completed_epoch"] == 0
    assert epoch["optimizer_steps"] == 66
    assert epoch["samples_seen_epoch"] == 2
    assert epoch["stage_observations_epoch"] == 3
    assert epoch["learning_rate"] == pytest.approx(4.0e-5)
    assert epoch["metrics"] == {"train_loss_epoch": 12.5}
    assert events[2]["completed_epoch"] == 0


def test_runtime_callback_records_exception_without_private_message(
    tmp_path: Path,
) -> None:
    from utils.sonata_training_evidence import SonataTrainingEvidenceCallback

    callback = SonataTrainingEvidenceCallback(output_dir=str(tmp_path))
    trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=4,
        global_step=264,
        callback_metrics={},
        optimizers=[],
    )

    callback.on_exception(
        trainer,
        SimpleNamespace(),
        RuntimeError("failed below /home/private-user/data"),
    )

    event = _events(tmp_path / ".sonata_runtime_events.jsonl")[0]
    assert event["event"] == "fit_interrupted"
    assert event["exception_type"] == "RuntimeError"
    assert "/home" not in json.dumps(event)


def test_unique_candidate_allows_only_identical_resume(tmp_path: Path) -> None:
    from scripts.run_sonata_second_training import authorize_unique_candidate

    contract = {
        "schema_version": 1,
        "status": "active",
        "candidate_id": "a" * 64,
        "bindings": {"source_tree_sha256": "b" * 64},
        "recipe": {"seed": 45, "epochs": 450, "devices": [1, 2]},
    }

    assert authorize_unique_candidate(tmp_path, contract) == "fresh"
    assert authorize_unique_candidate(tmp_path, contract) == "resume"

    changed = json.loads(json.dumps(contract))
    changed["recipe"]["seed"] = 46
    with pytest.raises(ValueError, match="candidate contract mismatch"):
        authorize_unique_candidate(tmp_path, changed)


def test_unique_candidate_refuses_unowned_checkpoints(tmp_path: Path) -> None:
    from scripts.run_sonata_second_training import authorize_unique_candidate

    (tmp_path / "last.ckpt").write_bytes(b"not-owned")
    contract = {
        "schema_version": 1,
        "status": "active",
        "candidate_id": "a" * 64,
        "bindings": {},
        "recipe": {"seed": 45, "epochs": 450, "devices": [1, 2]},
    }

    with pytest.raises(ValueError, match="unowned checkpoint"):
        authorize_unique_candidate(tmp_path, contract)


def test_candidate_contract_uses_frozen_real_budget(tmp_path: Path) -> None:
    from scripts.run_sonata_second_training import (
        RESOURCE_BLOCKER_PATH,
        _candidate_contract,
    )
    from utils.sonata_second_preflight import file_sha256

    (tmp_path / "preflight_authorization.json").write_text(
        json.dumps({"authorization_sha256": "a" * 64, "bindings": {}}),
        encoding="ascii",
    )
    (tmp_path / "data_manifest.json").write_text(
        json.dumps({"mixed_runtime": {"sampler_num_samples": 2112}}),
        encoding="ascii",
    )
    (tmp_path / "training_semantics.json").write_text(
        json.dumps(
            {
                "physical_batch_per_device": 2,
                "accumulate_grad_batches": 8,
                "effective_global_batch": 32,
            }
        ),
        encoding="ascii",
    )
    smoke = {
        "authorization_sha256": "b" * 64,
        "bindings": {
            "resource_blocker_sha256": file_sha256(RESOURCE_BLOCKER_PATH)
        },
    }

    contract = _candidate_contract(
        artifact_dir=tmp_path,
        smoke_authorization=smoke,
        devices=(1, 2),
    )

    assert contract["recipe"]["samples_per_epoch"] == 2112
    assert contract["recipe"]["optimizer_steps_per_epoch"] == 66
    assert contract["recipe"]["microbatch_per_gpu"] == 2
    assert contract["recipe"]["accumulate_grad_batches"] == 8
    assert contract["recipe"]["effective_global_batch"] == 32
    assert contract["reauthorization"] == {
        "basis": "user_authorized_recommended_configuration_after_resource_failure",
        "reason_gate": "SS4-RESOURCE-BLOCKED",
        "resource_blocker_sha256": file_sha256(RESOURCE_BLOCKER_PATH),
        "supersedes_candidate_id": (
            "d98291fd2dd14d089663d86e79acd9d0c5daad11012bc818e80bee6c7f5a17f8"
        ),
    }


def _completed_runtime(
    output_dir: Path, *, epochs: int = 450, include_fit_completed: bool = True
) -> None:
    from utils.sonata_training_evidence import append_runtime_event

    for epoch in range(epochs):
        metrics = {"train_loss_epoch": 10.0 - epoch / 100.0}
        if (epoch + 1) % 15 == 0:
            metrics["val_mean_t-AP"] = 0.321234 if epoch == 434 else 0.2
        append_runtime_event(
            output_dir,
            {
                "schema_version": 1,
                "event": "epoch_completed",
                "completed_epoch": epoch,
                "optimizer_steps": (epoch + 1) * 66,
                "samples_seen_epoch": 2112,
                "samples_seen_total": (epoch + 1) * 2112,
                "stage_observations_epoch": 3200,
                "stage_observations_total": (epoch + 1) * 3200,
                "learning_rate": 1.0e-5,
                "metrics": metrics,
                "peak_allocated_vram_mib": 1000.0,
                "peak_reserved_vram_mib": 2000.0,
                "process_wall_clock_seconds": float(epoch + 1),
            },
        )
    if include_fit_completed:
        append_runtime_event(
            output_dir,
            {
                "schema_version": 1,
                "event": "fit_completed",
                "completed_epoch": epochs - 1,
                "optimizer_steps": epochs * 66,
                "process_wall_clock_seconds": float(epochs),
            },
        )


def _candidate_fixture() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "active",
        "candidate_id": "a" * 64,
        "bindings": {
            "source_commit": "b" * 40,
            "source_tree_sha256": "c" * 64,
        },
        "recipe": {
            "seed": 45,
            "epochs": 450,
            "devices": [1, 2],
            "samples_per_epoch": 2112,
            "optimizer_steps_per_epoch": 66,
            "checkpoint_selection": "highest val_mean_t-AP",
        },
    }


def _source_snapshot_fixture() -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "pass",
        "training_source": {
            "schema_version": 1,
            "status": "pass",
            "source_commit": "b" * 40,
            "content_sha256": "c" * 64,
            "files": [],
        },
        "evidence_producer": {
            "schema_version": 1,
            "status": "pass",
            "source_commit": "d" * 40,
            "content_sha256": "e" * 64,
            "files": [],
        },
    }


def _write_top1_alias_checkpoints(output_dir: Path) -> None:
    best = output_dir / "epoch=434-val_mean_t-AP=0.321.ckpt"
    torch.save(
        {
            "epoch": 434,
            "global_step": 28710,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"state": {0: {"step": 28710}}}],
            "lr_schedulers": [{"last_epoch": 28710}],
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_mean_t-AP",
                    "best_model_score": torch.tensor(0.321234),
                }
            },
        },
        best,
    )
    shutil.copyfile(best, output_dir / "last.ckpt")


def test_completed_epochs_use_latest_duplicate_and_reconstruct_totals() -> None:
    from scripts.finalize_sonata_second_training import _completed_epochs

    events: list[dict[str, object]] = []
    for epoch in range(450):
        events.append(
            {
                "event": "epoch_completed",
                "completed_epoch": epoch,
                "optimizer_steps": (epoch + 1) * 66,
                "samples_seen_epoch": 2112,
                "samples_seen_total": (epoch % 20 + 1) * 2112,
                "stage_observations_epoch": 3200,
                "stage_observations_total": (epoch % 20 + 1) * 3200,
            }
        )
    events.append(
        {
            "event": "epoch_completed",
            "completed_epoch": 30,
            "optimizer_steps": 31 * 66,
            "samples_seen_epoch": 2112,
            "samples_seen_total": 2112,
            "stage_observations_epoch": 3333,
            "stage_observations_total": 3333,
        }
    )
    events.append(
        {
            "event": "fit_completed",
            "completed_epoch": 449,
            "optimizer_steps": 29_700,
        }
    )

    completed = _completed_epochs(
        events,
        epochs=450,
        steps_per_epoch=66,
        samples_per_epoch=2112,
    )

    assert completed[30]["stage_observations_epoch"] == 3333
    assert completed[30]["samples_seen_total_raw"] == 2112
    assert completed[30]["stage_observations_total_raw"] == 3333
    assert completed[30]["samples_seen_total_reconstructed"] == 31 * 2112
    assert completed[30]["stage_observations_total_reconstructed"] == 30 * 3200 + 3333
    assert completed[-1]["samples_seen_total_reconstructed"] == 950_400
    assert completed[-1]["stage_observations_total_reconstructed"] == 1_440_133


def test_learning_curve_marks_validation_cadence_and_blanks_stale_metrics() -> None:
    from scripts.finalize_sonata_second_training import _learning_curve

    events = []
    for epoch in (13, 14, 15):
        events.append(
            {
                "completed_epoch": epoch,
                "optimizer_steps": (epoch + 1) * 66,
                "samples_seen_epoch": 2112,
                "samples_seen_total_raw": (epoch + 1) * 2112,
                "samples_seen_total_reconstructed": (epoch + 1) * 2112,
                "stage_observations_epoch": 3200,
                "stage_observations_total_raw": (epoch + 1) * 3200,
                "stage_observations_total_reconstructed": (epoch + 1) * 3200,
                "metrics": {
                    "train_loss_epoch": 1.0,
                    "val_mean_t-AP": 0.25,
                },
            }
        )

    rows = list(csv.DictReader(io.StringIO(_learning_curve(events).decode("ascii"))))

    assert [row["validation_performed"] for row in rows] == [
        "false",
        "true",
        "false",
    ]
    assert [row["val_mean_t-AP"] for row in rows] == ["", "0.25", ""]
    assert all(row["train_loss_epoch"] == "1.0" for row in rows)
    assert rows[-1]["samples_seen_total_raw"] == str(16 * 2112)
    assert rows[-1]["samples_seen_total_reconstructed"] == str(16 * 2112)
    assert rows[-1]["stage_observations_total_raw"] == str(16 * 3200)
    assert rows[-1]["stage_observations_total_reconstructed"] == str(16 * 3200)


def test_checkpoint_record_separates_exact_score_from_rounded_filename(
    tmp_path: Path,
) -> None:
    from scripts.finalize_sonata_second_training import _checkpoint_record

    path = tmp_path / "epoch=434-val_mean_t-AP=0.321.ckpt"
    torch.save(
        {
            "epoch": 434,
            "global_step": 28710,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"state": {0: {"step": 28710}}}],
            "lr_schedulers": [{"last_epoch": 28710}],
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_mean_t-AP",
                    "best_model_score": torch.tensor(0.321234),
                }
            },
        },
        path,
    )

    record = _checkpoint_record(path)

    assert record["selection_metric_name"] == "val_mean_t-AP"
    assert record["selection_metric_exact"] == pytest.approx(0.321234)
    assert record["selection_metric_filename_rounded"] == pytest.approx(0.321)


def test_checkpoint_inventory_records_full_budget_top1_alias(
    tmp_path: Path,
) -> None:
    from scripts.finalize_sonata_second_training import _checkpoint_inventory

    best = tmp_path / "epoch=449-val_mean_t-AP=0.321.ckpt"
    torch.save(
        {
            "epoch": 449,
            "global_step": 29_700,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"state": {0: {"step": 29_700}}}],
            "lr_schedulers": [{"last_epoch": 29_700}],
            "callbacks": {
                "ModelCheckpoint": {
                    "monitor": "val_mean_t-AP",
                    "best_model_score": torch.tensor(0.321234),
                }
            },
        },
        best,
    )
    shutil.copyfile(best, tmp_path / "last.ckpt")

    _, semantics = _checkpoint_inventory(tmp_path)

    assert semantics == {
        "last_is_full_budget": True,
        "last_is_top1_byte_identical_alias": True,
    }


def test_training_finalizer_accepts_last_as_byte_identical_top1_alias(
    tmp_path: Path,
) -> None:
    from scripts.finalize_sonata_second_training import finalize_training

    output_dir = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    candidate = _candidate_fixture()
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir)
    _write_top1_alias_checkpoints(output_dir)

    manifest = finalize_training(
        training_output_dir=output_dir,
        artifact_dir=artifact_dir,
        expected_candidate=candidate,
        source_snapshot_manifest=_source_snapshot_fixture(),
    )

    assert manifest["status"] == "pass"
    assert manifest["budget"]["optimizer_steps"] == 29700
    assert manifest["budget"]["samples_seen"] == 950400
    assert manifest["checkpoint_semantics"] == {
        "last_is_full_budget": False,
        "last_is_top1_byte_identical_alias": True,
    }
    assert len(manifest["checkpoints"]) == 2
    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "TRAINING_MANIFEST.json",
        "TRAINING_REPORT.md",
        "checkpoint_inventory.csv",
        "learning_curve.csv",
        "runtime_events.jsonl",
        "source_snapshot_manifest.json",
    ]


def test_training_finalizer_reconciles_correction_v2_counts(
    tmp_path: Path,
) -> None:
    from scripts.finalize_sonata_second_training import finalize_training
    from utils.sonata_training_evidence import append_runtime_event

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate = _candidate_fixture()
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir)
    correction_events = [
        {"event": "launch_authorized", "launch_mode": "resume", "checkpoint_count": 0},
        {"event": "fit_interrupted"},
        {"event": "launch_authorized", "launch_mode": "resume", "checkpoint_count": 2},
        {"event": "storage_migration_started", "termination": "sigkill"},
        {"event": "launch_authorized", "launch_mode": "resume", "checkpoint_count": 2},
        {
            "event": "runtime_evidence_semantics_correction",
            "correction_version": 1,
            "basis_checkpoint_count": 0,
            "original_launch_mode": "resume",
            "corrected_launch_mode": "retry_before_checkpoint",
            "expected_checkpoint_resume_count": 2,
            "expected_infrastructure_interruption_count": 2,
        },
        {"event": "checkpoint_callback_path_repaired", "termination": "sigkill"},
        {"event": "launch_authorized", "launch_mode": "resume", "checkpoint_count": 2},
        {
            "event": "runtime_evidence_semantics_correction",
            "correction_version": 2,
            "supersedes_correction_version": 1,
            "expected_checkpoint_resume_count": 3,
            "expected_infrastructure_interruption_count": 3,
        },
    ]
    for event in correction_events:
        append_runtime_event(output_dir, {"schema_version": 1, **event})
    _write_top1_alias_checkpoints(output_dir)

    manifest = finalize_training(
        training_output_dir=output_dir,
        artifact_dir=tmp_path / "artifacts",
        expected_candidate=candidate,
        source_snapshot_manifest=_source_snapshot_fixture(),
    )

    assert manifest["runtime"]["interruption_count"] == 3
    assert manifest["runtime"]["resume_launch_count"] == 3
    assert manifest["runtime"]["correction_version"] == 2


def test_training_finalizer_rejects_checkpoint_below_best_validation(
    tmp_path: Path,
) -> None:
    from scripts.finalize_sonata_second_training import (
        SonataTrainingFinalizationError,
        finalize_training,
    )
    from utils.sonata_training_evidence import append_runtime_event

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate = _candidate_fixture()
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir)
    append_runtime_event(
        output_dir,
        {
            "schema_version": 1,
            "event": "epoch_completed",
            "completed_epoch": 449,
            "optimizer_steps": 29_700,
            "samples_seen_epoch": 2112,
            "samples_seen_total": 950_400,
            "stage_observations_epoch": 3200,
            "stage_observations_total": 1_440_000,
            "metrics": {"val_mean_t-AP": 0.5},
        },
    )
    _write_top1_alias_checkpoints(output_dir)

    with pytest.raises(
        SonataTrainingFinalizationError,
        match="highest validation",
    ):
        finalize_training(
            training_output_dir=output_dir,
            artifact_dir=tmp_path / "artifacts",
            expected_candidate=candidate,
            source_snapshot_manifest=_source_snapshot_fixture(),
        )


def test_training_finalizer_rejects_incomplete_budget(tmp_path: Path) -> None:
    from scripts.finalize_sonata_second_training import (
        SonataTrainingFinalizationError,
        finalize_training,
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate = _candidate_fixture()
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir, epochs=1)

    with pytest.raises(SonataTrainingFinalizationError, match="450 completed epochs"):
        finalize_training(
            training_output_dir=output_dir,
            artifact_dir=tmp_path / "artifacts",
            expected_candidate=candidate,
            source_snapshot_manifest=_source_snapshot_fixture(),
        )


def test_training_finalizer_rejects_missing_fit_completed(tmp_path: Path) -> None:
    from scripts.finalize_sonata_second_training import (
        SonataTrainingFinalizationError,
        finalize_training,
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate = _candidate_fixture()
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir, include_fit_completed=False)

    with pytest.raises(
        SonataTrainingFinalizationError,
        match="full-budget fit completion is absent",
    ):
        finalize_training(
            training_output_dir=output_dir,
            artifact_dir=tmp_path / "artifacts",
            expected_candidate=candidate,
            source_snapshot_manifest=_source_snapshot_fixture(),
        )
