from __future__ import annotations

import json
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


def _completed_runtime(output_dir: Path, *, epochs: int = 450) -> None:
    from utils.sonata_training_evidence import append_runtime_event

    for epoch in range(epochs):
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
                "metrics": {"train_loss_epoch": 10.0 - epoch / 100.0},
                "peak_allocated_vram_mib": 1000.0,
                "peak_reserved_vram_mib": 2000.0,
                "process_wall_clock_seconds": float(epoch + 1),
            },
        )
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


def test_training_finalizer_requires_and_records_full_budget(tmp_path: Path) -> None:
    from scripts.finalize_sonata_second_training import finalize_training

    output_dir = tmp_path / "output"
    artifact_dir = tmp_path / "artifacts"
    output_dir.mkdir()
    candidate = {
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
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir)
    torch.save(
        {
            "epoch": 449,
            "global_step": 29700,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"state": {0: {"step": 29700}}}],
            "lr_schedulers": [{"last_epoch": 29700}],
        },
        output_dir / "last.ckpt",
    )
    torch.save(
        {
            "epoch": 434,
            "global_step": 28710,
            "state_dict": {"weight": torch.ones(1)},
            "optimizer_states": [{"state": {0: {"step": 28710}}}],
            "lr_schedulers": [{"last_epoch": 28710}],
        },
        output_dir / "epoch=434-val_mean_t-AP=0.321.ckpt",
    )
    source_snapshot = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": "b" * 40,
        "content_sha256": "c" * 64,
        "files": [],
    }

    manifest = finalize_training(
        training_output_dir=output_dir,
        artifact_dir=artifact_dir,
        expected_candidate=candidate,
        source_snapshot_manifest=source_snapshot,
    )

    assert manifest["status"] == "pass"
    assert manifest["budget"]["optimizer_steps"] == 29700
    assert manifest["budget"]["samples_seen"] == 950400
    assert len(manifest["checkpoints"]) == 2
    assert sorted(path.name for path in artifact_dir.iterdir()) == [
        "TRAINING_MANIFEST.json",
        "TRAINING_REPORT.md",
        "checkpoint_inventory.csv",
        "learning_curve.csv",
        "runtime_events.jsonl",
        "source_snapshot_manifest.json",
    ]


def test_training_finalizer_rejects_incomplete_budget(tmp_path: Path) -> None:
    from scripts.finalize_sonata_second_training import (
        SonataTrainingFinalizationError,
        finalize_training,
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    candidate = {
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
    (output_dir / ".sonata_second_candidate.json").write_text(
        json.dumps(candidate), encoding="ascii"
    )
    _completed_runtime(output_dir, epochs=1)

    with pytest.raises(SonataTrainingFinalizationError, match="450 completed epochs"):
        finalize_training(
            training_output_dir=output_dir,
            artifact_dir=tmp_path / "artifacts",
            expected_candidate=candidate,
            source_snapshot_manifest={
                "source_commit": "b" * 40,
                "content_sha256": "c" * 64,
            },
        )
