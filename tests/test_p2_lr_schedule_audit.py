import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit_p2_lr_schedule import CSV_FIELDS, run_audit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_p2_lr_schedule.py"


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


def test_metadata_locks_effective_batch_and_blocks_formal_total_steps(
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
    assert result["formal_training"] == {
        "status": "blocked_missing_scannet",
        "dataset_mix": "3RScan T=2 (1.0) + ScanNet T=1 (0.8)",
        "epochs": 450,
        "total_steps": None,
        "scannet_ref": "repo:data/processed/scannet",
        "preflight_ref": "repo:artifacts/P2/scannet_preflight.json",
        "epoch_microbatch_divisibility": {
            "status": "pending_missing_scannet",
            "epoch_microbatches": None,
            "accumulation_steps": 4,
            "remainder": None,
            "drop_last": False,
        },
        "reason": (
            "ScanNet prerequisites are missing; the formal mixed-data loader "
            "length and total_steps cannot be computed."
        ),
    }

    report = (output_dir / "lr_schedule_audit.md").read_text(encoding="utf-8")
    assert "scheduler semantics preflight" in report
    assert "not formal mixed-data training" in report
    assert "2 GPUs * 4 samples/GPU * 4 accumulation steps = 32" in report
    assert "formal status: blocked_missing_scannet" in report
    assert "formal epochs: 450" in report
    assert "formal total_steps: null" in report
    assert "formal epoch microbatch divisibility: pending_missing_scannet" in report
    assert "formal epoch microbatches: null" in report
    assert "formal accumulation remainder: null" in report
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

    assert result["formal_training"]["status"] == "blocked_missing_scannet"
    assert result["formal_training"]["total_steps"] is None


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
    assert {
        name: (REPO_ROOT / "artifacts" / "P2" / name).read_bytes() for name in first
    } == first
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
    assert "formal_total_steps=null" in result.stdout
