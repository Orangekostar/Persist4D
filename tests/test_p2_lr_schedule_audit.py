import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_p2_lr_schedule as audit
from scripts.audit_p2_lr_schedule import (
    CSV_FIELDS,
    _complete_scannet_gate_passed,
    run_audit,
)
from utils.p2_preflight import P2_PREFLIGHT_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_p2_lr_schedule.py"
SHARED_AUTHORIZATION_FIELDS = (
    "config_contract",
    "source_tree_contract",
    "runtime_source_contract",
    "runtime_environment_contract",
    "official_split_identity",
    "input_manifest",
    "authorization",
)


def _complete_preflight(schema_version: int) -> dict:
    payload = {
        "schema_version": schema_version,
        "status": "pass",
        "formal_p2_training_authorized": True,
        "official_source_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "split_metadata_status": "pass",
        "expected_split_counts": {
            "train": 1201,
            "validation": 312,
            "test": 100,
        },
        "errors": [],
        "raw_assets": {"status": "pass"},
        "processed_assets": {"status": "pass"},
        "class_taxonomy": {"status": "pass"},
        "mix_instantiation": {"status": "pass"},
    }
    payload.update({field: {} for field in SHARED_AUTHORIZATION_FIELDS})
    return payload


def test_complete_scannet_gate_accepts_current_preflight_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(
        json.dumps(_complete_preflight(P2_PREFLIGHT_SCHEMA_VERSION)),
        encoding="utf-8",
    )

    monkeypatch.setattr(audit, "_shared_p2_authorization_gate", lambda _path: True)

    assert _complete_scannet_gate_passed(preflight_path) is True


def test_complete_scannet_gate_rejects_legacy_preflight_schema(
    tmp_path: Path,
) -> None:
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(
        json.dumps(_complete_preflight(P2_PREFLIGHT_SCHEMA_VERSION - 1)),
        encoding="utf-8",
    )

    assert _complete_scannet_gate_passed(preflight_path) is False


@pytest.mark.parametrize(
    "missing_field",
    [
        "config_contract",
        "source_tree_contract",
        "runtime_source_contract",
        "runtime_environment_contract",
        "official_split_identity",
        "input_manifest",
        "authorization",
    ],
)
def test_complete_gate_requires_every_shared_authorization_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    payload = _complete_preflight(P2_PREFLIGHT_SCHEMA_VERSION)
    payload.pop(missing_field)
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        audit,
        "_shared_p2_authorization_gate",
        lambda _path: True,
    )

    assert _complete_scannet_gate_passed(preflight_path) is False


def test_complete_gate_rejects_shared_authorization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(
        json.dumps(_complete_preflight(P2_PREFLIGHT_SCHEMA_VERSION)),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "_compose_p2_config", lambda: object())

    def reject_authorization(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("shared authorization rejected")

    monkeypatch.setattr(
        audit,
        "require_p2_preflight_authorization",
        reject_authorization,
    )

    assert _complete_scannet_gate_passed(preflight_path) is False


def test_complete_gate_allows_ready_only_after_shared_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(
        json.dumps(_complete_preflight(P2_PREFLIGHT_SCHEMA_VERSION)),
        encoding="utf-8",
    )
    config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(audit, "_compose_p2_config", lambda: config)

    def allow_authorization(received_config: object, *, artifact_path: Path) -> Path:
        calls.append((received_config, artifact_path))
        return artifact_path

    monkeypatch.setattr(
        audit,
        "require_p2_preflight_authorization",
        allow_authorization,
    )

    assert _complete_scannet_gate_passed(preflight_path) is True
    assert calls == [(config, preflight_path)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def completed_audit(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("p2-lr-audit")
    output_dir = root / "artifacts"
    result = run_audit(
        output_dir=output_dir,
        processed_scannet_dir=root / "missing-scannet",
    )
    return result, output_dir, root


def test_runtime_audit_uses_lightning_automatic_optimization_and_tail_window(
    completed_audit,
) -> None:
    result, output_dir, _ = completed_audit
    rows = result["rows"]

    assert result["runtime"]["engine"] == "pytorch_lightning.Trainer.fit"
    assert result["runtime"]["lightning_version"] == "2.6.5"
    assert result["runtime"]["automatic_optimization"] is True
    assert result["runtime"]["accelerator"] == "cpu"
    assert result["runtime"]["simulation_devices"] == 1
    assert result["runtime"]["simulation_microbatches"] == 10
    assert result["runtime"]["simulation_total_steps"] == 3
    assert len(rows) == 10
    assert [row["micro_step"] for row in rows] == list(range(1, 11))
    assert [row["micro_step"] for row in rows if row["did_optimizer_step"]] == [
        4,
        8,
        10,
    ]
    assert rows[-1]["accumulation_window_size"] == 2
    assert rows[-1]["optimizer_step_after"] == 3
    assert rows[-1]["global_step_after"] == 3
    assert all(row["target_window_samples"] == 32 for row in rows[:8])
    assert all(row["target_window_samples"] == 16 for row in rows[8:])
    assert all(row["normalization_denominator_microbatches"] == 4 for row in rows)
    assert all(row["relative_gradient_scale"] == 1.0 for row in rows[:8])
    assert all(row["relative_gradient_scale"] == 0.5 for row in rows[8:])
    assert (output_dir / "lr_schedule_audit.csv").is_file()
    assert (output_dir / "lr_schedule_audit.md").is_file()


def test_optimizer_global_step_scheduler_and_lr_advance_together(
    completed_audit,
) -> None:
    result, _, _ = completed_audit
    rows = result["rows"]
    changed_lr_steps = []

    for row in rows:
        optimizer_delta = row["optimizer_step_after"] - row["optimizer_step_before"]
        global_delta = row["global_step_after"] - row["global_step_before"]
        scheduler_delta = (
            row["scheduler_last_epoch_after"] - row["scheduler_last_epoch_before"]
        )
        lr_changed = row["lr_after"] != pytest.approx(row["lr_before"], abs=1e-15)

        assert optimizer_delta == int(row["did_optimizer_step"])
        assert global_delta == optimizer_delta
        assert scheduler_delta == optimizer_delta
        assert row["optimizer_step_before"] == row["global_step_before"]
        assert row["optimizer_step_after"] == row["global_step_after"]
        if lr_changed:
            changed_lr_steps.append(row["micro_step"])
            assert row["did_optimizer_step"] is True

    assert changed_lr_steps
    assert result["scheduler"]["name"] == "OneCycleLR"
    assert result["scheduler"]["interval"] == "step"
    assert result["scheduler"]["max_lr_contract"] == pytest.approx(5e-4)
    assert result["scheduler"]["resolved_max_lr"] == pytest.approx(5e-4)
    assert (
        max(value for row in rows for value in (row["lr_before"], row["lr_after"]))
        <= 5e-4 + 1e-15
    )
    assert result["scheduler"]["sampled_max_lr_must_equal_contract"] is False


def test_metadata_records_planned_batch_contract_while_formal_run_is_blocked(
    completed_audit,
) -> None:
    result, output_dir, _ = completed_audit

    assert result["target_topology"] == {
        "gpus": 2,
        "batch_per_gpu": 4,
        "physical_global_batch": 8,
        "gradient_accumulation": 4,
        "effective_batch": 32,
        "formula": "2 GPUs * 4 samples/GPU * 4 accumulation steps = 32",
    }
    assert result["formal_training"]["primary_dataset_samples"] == 1174
    assert result["formal_training"]["raw_sampler_num_samples"] == 2113
    assert result["formal_training"]["sampler_num_samples"] == 2112
    assert result["formal_training"] == {
        "status": "blocked_missing_scannet",
        "contract_kind": "planned_not_observed",
        "observed_formal_run": False,
        "dataset_mix": "3RScan T=2 (1.0) + ScanNet T=1 (0.8)",
        "primary_dataset_samples": 1174,
        "dataset_weights": [1.0, 0.8],
        "raw_sampler_num_samples": 2113,
        "epoch_sample_multiple": 32,
        "sampler_num_samples": 2112,
        "sampler_seed": 45,
        "sampler_seed_scope": (
            "fresh_start_and_completed_epoch_boundary_resume"
        ),
        "sampler_generator_state_checkpointed": True,
        "sampler_checkpoint_scope": "completed_epoch_boundary_only",
        "sampler_checkpoint_save_timing": (
            "p2_normalized_train_epoch_end_callbacks"
        ),
        "sampler_non_boundary_resume_verified": False,
        "sampler_mid_epoch_resume_supported": False,
        "sampler_dataloader_prefetch_state_checkpointed": False,
        "samples_per_rank": 1056,
        "epochs": 450,
        "optimizer_steps_per_epoch": 66,
        "total_steps": 29700,
        "scannet_ref": "repo:data/processed/scannet",
        "preflight_ref": "repo:artifacts/P2/scannet_preflight.json",
        "epoch_microbatch_divisibility": {
            "status": "planned_aligned",
            "scope": "per_rank",
            "epoch_microbatches": 264,
            "accumulation_steps": 4,
            "remainder": 0,
            "drop_last": False,
        },
        "reason": (
            "ScanNet prerequisites are missing, so no formal mixed-data run was "
            "observed; the planned sampler and optimizer-step contract remains "
            "computable from the locked P2 configuration."
        ),
    }

    report = (output_dir / "lr_schedule_audit.md").read_text(encoding="utf-8")
    assert "scheduler semantics preflight" in report
    assert "not formal mixed-data training" in report
    assert "2 GPUs * 4 samples/GPU * 4 accumulation steps = 32" in report
    assert "sampler checkpoint scope: completed_epoch_boundary_only" in report
    assert (
        "sampler checkpoint save timing: "
        "p2_normalized_train_epoch_end_callbacks"
    ) in report
    assert "sampler non-boundary resume verified: false" in report
    assert "sampler mid-epoch resume supported: false" in report
    assert "sampler DataLoader prefetch state checkpointed: false" in report
    assert "formal status: blocked_missing_scannet" in report
    assert "formal run observed: false" in report
    assert "formal epochs: 450" in report
    assert "planned sampler num_samples: 2112" in report
    assert "planned raw sampler num_samples: 2113" in report
    assert (
        "planned sampler seed scope: "
        "fresh_start_and_completed_epoch_boundary_resume"
    ) in report
    assert "sampler generator state checkpointed: true" in report
    assert "planned samples per rank: 1056" in report
    assert "planned optimizer steps per epoch: 66" in report
    assert "planned total_steps: 29700" in report
    assert "planned epoch microbatch divisibility: planned_aligned" in report
    assert "planned epoch microbatches per rank: 264" in report
    assert "planned accumulation remainder: 0" in report
    assert "tail target samples=16" in report
    assert "relative gradient scale=0.5" in report
    assert "lr_before is applied to the current optimizer update" in report
    assert "lr_after is scheduled for the next optimizer update" in report
    assert "short simulation need not reach max_lr exactly" in report


def test_incomplete_scannet_metadata_stays_blocked(tmp_path: Path) -> None:
    processed_dir = tmp_path / "partial-scannet"
    processed_dir.mkdir()
    (processed_dir / "train_database.yaml").write_text("[]\n", encoding="utf-8")
    (processed_dir / "validation_database.yaml").write_text("[]\n", encoding="utf-8")
    (processed_dir / "scannet.yaml").write_text("{}\n", encoding="utf-8")
    (processed_dir / "label_database.yaml").write_text("{}\n", encoding="utf-8")
    incomplete_preflight = tmp_path / "incomplete-preflight.json"
    incomplete_preflight.write_text(
        json.dumps(
            {
                "status": "pass",
                "formal_p2_training_authorized": True,
            }
        ),
        encoding="utf-8",
    )

    result = run_audit(
        output_dir=tmp_path / "artifacts",
        processed_scannet_dir=processed_dir,
        scannet_preflight_path=incomplete_preflight,
    )

    formal = result["formal_training"]
    assert formal["status"] == "blocked_missing_scannet"
    assert formal["contract_kind"] == "planned_not_observed"
    assert formal["observed_formal_run"] is False
    assert formal["total_steps"] == 29700


def test_csv_schema_and_artifacts_are_deterministic_and_private(
    completed_audit,
) -> None:
    _, output_dir, root = completed_audit
    first = {
        path.name: path.read_bytes()
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    second_output = root / "second-artifacts"
    run_audit(
        output_dir=second_output,
        processed_scannet_dir=root / "missing-scannet",
    )
    second = {
        path.name: path.read_bytes()
        for path in sorted(second_output.iterdir())
        if path.is_file()
    }

    assert second == first
    csv_path = output_dir / "lr_schedule_audit.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == CSV_FIELDS
        assert {
            "target_window_samples",
            "normalization_denominator_microbatches",
            "relative_gradient_scale",
        } <= set(reader.fieldnames or ())
    assert b"\r\n" not in csv_path.read_bytes()
    assert len(_read_csv(csv_path)) == 10

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    )
    assert "/" + "ho" + "me" + "/" not in artifact_text
    assert "/" + "Us" + "ers" + "/" not in artifact_text
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", artifact_text)
    assert not re.search(
        r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b",
        artifact_text,
    )
    assert "GPU" + "-" not in artifact_text


def test_cli_generates_both_artifacts_and_exits_zero(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--processed-scannet-dir",
            str(tmp_path / "missing-scannet"),
            "--scannet-preflight",
            str(tmp_path / "missing-preflight.json"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "lr_schedule_audit.csv",
        "lr_schedule_audit.md",
    ]
    assert "optimizer_steps=3" in result.stdout
    assert "formal_status=blocked_missing_scannet" in result.stdout
    assert "planned_total_steps=29700" in result.stdout
    assert "formal_run_observed=false" in result.stdout
    assert "formal_total_steps" not in result.stdout
