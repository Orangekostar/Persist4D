from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


def v2_config():
    return importlib.import_module("scripts.perception_gain_v2_config")


def test_h_l_recipes_drive_real_optimizer_and_keep_full_schedule(tmp_path: Path):
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    helper = v2_config()
    hashes = set()
    for label, rate in (("H", 5e-5), ("L", 1e-5)):
        recipe = helper.resolve_recipe("C0", learning_rate=label, seed=45, devices=2)
        config = compose_variant_config(
            "C0",
            pretrained=tmp_path / "base.pth",
            run_dir=tmp_path / label,
            recipe_config=recipe,
        )
        tiny = torch.nn.Linear(2, 1)
        tiny.config = config
        configured = PerceptionGainTrainer.configure_optimizers(tiny)
        assert configured["optimizer"].param_groups[0]["initial_lr"] == rate
        assert config.perception_training.optimizer_updates == 3000
        assert config.perception_training.warmup_updates == 150
        assert config.data.num_workers * config.perception_training.devices <= 8
        hashes.add(config.perception_recipe.training_recipe_hash)
    assert len(hashes) == 2


def test_resume_rejects_high_lr_optimizer_for_low_recipe_and_binds_import():
    helper = v2_config()
    high = helper.resolve_recipe("S-BAL", learning_rate="H", seed=45, devices=2)
    low = helper.resolve_recipe("S-BAL", learning_rate="L", seed=45, devices=2)
    checkpoint = {"perception_recipe": high}
    helper.validate_resume_recipe(checkpoint, high)
    with pytest.raises(ValueError, match="recipe"):
        helper.validate_resume_recipe(checkpoint, low)
    with pytest.raises(ValueError, match="legacy"):
        helper.validate_resume_recipe({}, high)


def test_budget_deducts_prior_and_unique_events_with_lower_cap():
    helper = v2_config()
    events = [
        {"event_id": "old", "gpu_hours": 2.5},
        {"event_id": "old", "gpu_hours": 2.5},
        {"event_id": "new", "gpu_hours": 1.0},
    ]
    assert helper.remaining_gpu_hours(32, 11.803828054742961, events) == pytest.approx(
        16.69617194525704
    )


def test_live_asset_resolution_does_not_require_or_rewrite_legacy_caches(tmp_path):
    from scripts.perception_gain_foundation import resolve_live_assets

    path = tmp_path / "assets.local.json"
    assets = {
        key: str(tmp_path / key)
        for key in (
            "r1_checkpoint",
            "concerto_pretrained",
            "data_root",
            "rio_metadata",
            "metric_dataset_spec",
        )
    }
    text = json.dumps(assets)
    path.write_text(text)
    assert resolve_live_assets(path) == assets
    assert path.read_text() == text


def test_data_loader_timeout_is_disabled_only_for_synchronous_loading():
    from scripts.train_perception_gain import loader_timeout

    assert loader_timeout(num_workers=2, timeout_seconds=180) == 180
    assert loader_timeout(num_workers=0, timeout_seconds=180) == 0


def test_repair_dependency_closure_excludes_perception_and_failed_high_is_local():
    runner = importlib.import_module("scripts.perception_gain_v2")
    assert runner.dependency_closure("REPAIR_R1") == ("BIND", "DATA", "REPAIR_R1")
    tasks = {name: {"status": "PENDING"} for name in runner.TASKS}
    tasks["BIND"]["status"] = tasks["DATA"]["status"] = "COMPLETE"
    tasks["HIGH_CONT"]["status"] = "BLOCKED"
    assert "REPAIR_R1" in runner.ready_tasks(tasks, target="all")
    assert "LOW_PAIR" in runner.ready_tasks(tasks, target="all")


def test_manifest_staging_preserves_data_and_localizes_symlink_target(tmp_path):
    runner = importlib.import_module("scripts.perception_gain_v2")
    source = tmp_path / "source"
    source.mkdir()
    (source / "actual.npy").write_bytes(b"real array payload")
    (source / "scan.npy").symlink_to(source / "actual.npy")
    destination = tmp_path / "local/scan.npy"
    record = runner.stage_input_file(source / "scan.npy", destination, copy_file=True)
    assert destination.read_bytes() == b"real array payload"
    assert not destination.is_symlink()
    assert record["bytes"] == 18
    assert len(record["sha256"]) == 64


def test_parallel_task_progress_cannot_mask_stalled_training(tmp_path):
    import datetime as dt
    import os

    runner = importlib.import_module("scripts.perception_gain_v2")
    for recipe_id, started, progress in (
        ("S-BAL-H-s45", 1000, 1100),
        ("C0-L-s45", 1001, 5000),
    ):
        directory = tmp_path / "training" / recipe_id
        runner.write_json(
            directory / "invocation-00.json",
            {
                "start_utc": dt.datetime.fromtimestamp(
                    started, dt.timezone.utc
                ).isoformat(),
            },
        )
        runner.write_json(directory / "PROGRESS.json", {})
        os.utime(directory / "PROGRESS.json", (progress, progress))
    assert runner.training_watchdog_deadline("HIGH_CONT", tmp_path, 999, 180) == 1280
    assert runner.training_watchdog_deadline("LOW_PAIR", tmp_path, 999, 180) == 5180
