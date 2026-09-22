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


@pytest.mark.parametrize(
    "new_delta,pair_delta,expected",
    [
        (0.003, 0.005, "PAIR"),
        (0.004, 0.0045, "NEW"),
        (0.0, -0.001, "KEEP_PARENT"),
        (0.0029, 0.0031, "KEEP_PARENT"),
    ],
)
def test_supplied_v2_repair_policy_examples(new_delta, pair_delta, expected):
    from scripts.perception_gain_v2_selection import select_repair

    def candidate(name, delta):
        return {
            "method_id": name,
            "metrics": {t: 0.2 + delta for t in (2, 3, 4, 5)},
            "coverage_status": "COMPLETE",
            "optimizer_update": 500,
            "new_parameter_count": 8769,
        }

    result = select_repair(
        parent=candidate("PARENT", 0),
        new=candidate("NEW", new_delta),
        pair=candidate("PAIR", pair_delta),
    )
    assert result["selected_mode"] == expected


def test_v2_missing_coverage_is_null_and_combined_tie_uses_total_updates():
    from scripts.perception_gain_v2_selection import compare, rank, strict_all_t

    baseline = {
        "method_id": "R1",
        "metrics": {t: 0.2 for t in (2, 3, 4, 5)},
        "coverage_status": "COMPLETE",
    }
    missing = {"method_id": "FH", "metrics": {}, "coverage_status": "INCOMPLETE"}
    assert compare(baseline, missing)["S_mean"] is None
    assert strict_all_t(baseline, missing) is None
    assert strict_all_t(baseline, baseline) is False
    common = {
        "metrics": {t: 0.21 for t in (2, 3, 4, 5)},
        "coverage_status": "COMPLETE",
        "new_parameter_count": 10,
    }
    a = {**common, "method_id": "a", "optimizer_update": 500, "tie_update": 2750}
    b = {**common, "method_id": "b", "optimizer_update": 1000, "tie_update": 1750}
    assert rank([a, b], baseline)[0]["method_id"] == "b"


def test_full_promotion_selects_one_mechanism_with_its_own_lr_control():
    from scripts.perception_gain_v2_selection import select_full_promotion

    baseline = {
        "method_id": "R1",
        "coverage_status": "COMPLETE",
        "metrics": {t: 0.2 for t in (2, 3, 4, 5)},
    }
    candidates = [
        {
            **baseline,
            "method_id": name,
            "optimizer_update": step,
            "metrics": {t: 0.2 + delta for t in (2, 3, 4, 5)},
        }
        for name, delta in (
            ("C0-H", 0),
            ("C0-L", 0.003),
            ("S-BAL-H", 0.005),
            ("S-BAL-L", 0.004),
        )
        for step in (250, 750)
    ]
    controls = {"S-BAL-H": "C0-H", "S-BAL-L": "C0-L"}
    result = select_full_promotion(candidates, baseline, control_by_method=controls)
    assert result["full_training_recipes"] == ["C0-H", "S-BAL-H"]
    without_control = [row for row in candidates if row["method_id"] != "C0-H"]
    result = select_full_promotion(
        without_control, baseline, control_by_method=controls
    )
    assert result["full_training_recipes"] == ["C0-L", "S-BAL-L"]
    assert "S-BAL-H" in result["missing_controls"]


def test_learning_rate_tie_and_incomparable_high_choose_low():
    from scripts.perception_gain_v2_selection import select_learning_rate

    baseline = {
        "method_id": "R1",
        "coverage_status": "COMPLETE",
        "metrics": {t: 0.2 for t in (2, 3, 4, 5)},
    }
    row = {**baseline, "optimizer_update": 750}
    assert (
        select_learning_rate({"H": [row], "L": [row]}, baseline, comparable_high=True)[
            "learning_rate_label"
        ]
        == "L"
    )
    high = {**row, "metrics": {t: 0.3 for t in (2, 3, 4, 5)}}
    assert (
        select_learning_rate(
            {"H": [high], "L": [row]}, baseline, comparable_high=False
        )["learning_rate_label"]
        == "L"
    )


def test_refiner_cache_rejects_changed_roles_and_unreviewed_code(tmp_path, monkeypatch):
    from scripts.perception_gain_v2 import file_hash, write_json
    from scripts.perception_gain_v2_config import content_hash
    from scripts.perception_gain_v2_evidence import validate_refiner_cache

    recipe = v2_config().resolve_recipe("C0", learning_rate="L")
    write_json(tmp_path / "data/STAGING_MANIFEST.json", {"files": []})
    write_json(tmp_path / "DATA_ROLES.json", {"TRAIN": ["train-reference"]})
    shard = tmp_path / "shard.pt"
    shard.write_bytes(b"test shard content")
    source = {
        "source_files": {"producer.py": "a" * 64},
        "executed_code_commit": "b" * 40,
    }
    monkeypatch.setattr(v2_config(), "live_execution_provenance", lambda _: source)
    binding = {
        "parent_weight_hash": "c" * 64,
        "inference_recipe_hash": recipe["inference_recipe_hash"],
        "input_manifest_hash": file_hash(tmp_path / "data/STAGING_MANIFEST.json"),
        "roles_sha256": file_hash(tmp_path / "DATA_ROLES.json"),
        "inventory_sha256": content_hash({"episodes": ["fixed"]}),
        "point_order_transform": "canonical_vertices/identity_geometry",
        "role": "TRAIN",
        "eval_seed": 45,
        "publisher": "D0/lag1/mean",
        "relevant_source_digest": content_hash(source["source_files"]),
        "executed_code_commit": source["executed_code_commit"],
    }
    manifest = {
        "status": "PASS",
        "recipe": recipe,
        "checkpoint_sha256": "c" * 64,
        "inventory": {"episodes": ["fixed"]},
        "cache_binding": binding,
        "execution_provenance": source,
        "shards": [{"path": shard.name, "sha256": file_hash(shard)}],
    }
    kwargs = dict(
        recipe=recipe,
        parent_weight="c" * 64,
        artifacts=tmp_path,
        cache_root=tmp_path,
        required_inventory=None,
    )
    assert validate_refiner_cache(manifest, **kwargs)["status"] == "VERIFIED"
    write_json(tmp_path / "DATA_ROLES.json", {"TRAIN": ["held-out-reference"]})
    with pytest.raises(ValueError, match="binding differs"):
        validate_refiner_cache(manifest, **kwargs)
    write_json(tmp_path / "DATA_ROLES.json", {"TRAIN": ["train-reference"]})
    monkeypatch.setattr(
        v2_config(),
        "live_execution_provenance",
        lambda _: {**source, "source_files": {"producer.py": "d" * 64}},
    )
    with pytest.raises(ValueError, match="without exact compatibility review"):
        validate_refiner_cache(manifest, **kwargs)


def test_repair_watchdog_handles_startup_and_its_own_training_checkpoint(tmp_path):
    import os
    from scripts.perception_gain_v2 import training_watchdog_deadline

    assert training_watchdog_deadline("REPAIR_R1", tmp_path, 1000, 180) == 1600
    checkpoint = tmp_path / "training/refiner/R1/NEW-s45/last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"saved")
    os.utime(checkpoint, (1500, 1500))
    assert training_watchdog_deadline("REPAIR_R1", tmp_path, 1000, 180) == 2100


def test_external_recovery_preserves_attempt_and_charges_cost_once(
    tmp_path, monkeypatch
):
    from scripts import perception_gain_v2 as runner

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "executed_identity", lambda *args: {})
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, {})
    artifacts, external = tmp_path / "artifacts", tmp_path / "external"
    runner.write_json(external / "assets.local.json", {})
    tasks = {name: {"status": "COMPLETE"} for name in runner.TASKS}
    tasks["REPAIR_R1"] = {"status": "BLOCKED", "gpu_hours": 1.0}
    runner.write_json(
        artifacts / "RUN_STATE.json",
        {
            "tasks": tasks,
            "identity": {"config_sha256": runner.file_hash(config_path)},
            "prior_gpu_hours": 10.0,
            "confirmation_reserve_gpu_hours": 40.0,
        },
    )
    for identifier, hours in (("failed", 1.0), ("recovered", 2.0)):
        runner.append_event(
            artifacts / "budget/LEDGER.jsonl",
            {"event_id": identifier, "scope": "V2", "gpu_hours": hours},
        )
    result = external / "tasks/recovered.json"
    runner.write_json(result, {"status": "COMPLETE"})
    runner.append_event(
        artifacts / "RECOVERY_EVENTS.jsonl",
        {
            "event_id": "recovered",
            "task": "REPAIR_R1",
            "exit_code": 0,
            "gpu_hours": 2.0,
            "result_path": "tasks/recovered.json",
            "result_sha256": runner.file_hash(result),
        },
    )
    config = {
        "artifact_root": "artifacts",
        "cumulative_gpu_hour_cap": 100,
        "runtime": {
            "perception_devices": 2,
            "maximum_gpus": 4,
            "maximum_training_jobs": 2,
        },
    }
    for _ in range(2):
        runner.run_tasks(
            config, config_path=config_path, external_root=external, target="REPAIR_R1"
        )
        state = runner.read_json(artifacts / "RUN_STATE.json")
        assert state["tasks"]["REPAIR_R1"]["status"] == "COMPLETE"
        assert state["tasks"]["REPAIR_R1"]["previous_attempt"]["status"] == "BLOCKED"
        assert state["v2_gpu_hours"] == state["tasks"]["REPAIR_R1"]["gpu_hours"] == 3.0
