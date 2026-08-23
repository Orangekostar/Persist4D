from __future__ import annotations

import json
from pathlib import Path

from scripts import reviewer_closure_training as training

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = REPO_ROOT / "configs/reviewer_closure/rescene_t3_adapted.yaml"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"


def test_t3_recipe_freezes_level2_identity_budget_and_provenance() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)

    assert recipe["comparison_level"] == 2
    assert recipe["paper_name"] == "ReScene4D T2-to-T3 Horizon-Adapted"
    assert recipe["source"]["checkpoint_epoch"] == 404
    assert recipe["source"]["checkpoint_global_step"] == 26730
    assert recipe["source"]["initialization"] == "weights_only_strict"
    assert recipe["data"]["rio_temporal_window"] == 3
    assert recipe["data"]["scannet_temporal_window"] == 1
    assert recipe["data"]["mixed_epoch_samples"] == 1536
    assert recipe["optimization"]["effective_batch_size"] == 32
    assert recipe["optimization"]["optimizer_updates_per_epoch"] == 48
    assert recipe["optimization"]["total_optimizer_updates"] == 2160
    assert recipe["evidence"]["recipe_changes_after_smoke"] == "forbidden"


def test_composed_t3_config_changes_only_frozen_adaptation_fields() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)
    t2, adapted = training.compose_t3_adaptation_config(recipe)
    differences = training.adaptation_config_differences(t2, adapted)

    assert "p2_preflight" in t2
    assert "p2_preflight" not in adapted
    assert adapted.model.config.temporal_window == 3
    assert adapted.data.train_dataset.datasets[0].temporal_window == 3
    assert adapted.data.train_dataset.datasets[1].temporal_window == 1
    assert adapted.data.validation_dataset.temporal_window == 3
    assert adapted.data.test_dataset.temporal_window == 3
    assert adapted.data.batch_size == 1
    assert adapted.trainer.accumulate_grad_batches == 16
    assert adapted.trainer.max_epochs == 45
    assert adapted.optimizer.lr == 5.0e-5
    assert adapted.scheduler.scheduler.total_steps == -1
    assert adapted.general.checkpoint == str(CHECKPOINT_PATH)
    assert adapted.general.reviewer_closure_weighted_objective is True
    assert adapted.general.reviewer_closure_fail_closed_runtime is True
    assert adapted.general.get("p2_weighted_objective") is None
    assert adapted.general.get("p2_fail_closed_runtime") is None
    assert set(differences) == set(training.ALLOWED_ADAPTATION_CONFIG_PATHS)


def test_source_checkpoint_metadata_matches_recipe() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)
    metadata = training.inspect_t3_source_checkpoint(
        CHECKPOINT_PATH,
        expected_sha256=recipe["source"]["checkpoint_sha256"],
    )

    assert metadata["epoch"] == 404
    assert metadata["global_step"] == 26730
    assert metadata["optimizer_state_count"] == 1
    assert metadata["scheduler_state_count"] == 1
    assert metadata["scheduler_last_epoch"] == 26730
    assert metadata["source_temporal_window"] == 2
    assert metadata["state_dict_entry_count"] == 798


def test_training_audit_classifies_all_fields_without_level1_claim(
    tmp_path: Path,
) -> None:
    result = training.build_t3_training_audit(
        recipe_path=RECIPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_root=tmp_path,
    )
    repeated = training.build_t3_training_audit(
        recipe_path=RECIPE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        output_root=tmp_path,
    )

    assert repeated == result
    assert result["status"] == "pass"
    assert result["selected_level"] == 2
    assert result["paper_name"] == "ReScene4D T2-to-T3 Horizon-Adapted"
    assert set(result["classification_counts"]) == {
        "known",
        "unknown",
        "reconstructed",
        "assumed",
    }
    assert all(value > 0 for value in result["classification_counts"].values())
    report = (tmp_path / "REScene_HORIZON_TRAINING_AUDIT.md").read_text()
    assert "Level 1 is not claimed" in report
    assert "epoch 404" in report
    manifest = json.loads(
        (tmp_path / "t3_training_recipe_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["content_sha256"] == result["content_sha256"]
