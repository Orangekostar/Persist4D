from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

import main_instance_segmentation as training_entrypoint
from scripts import reviewer_closure_training as training
from scripts import run_reviewer_closure_t3 as t3_runner
from trainer import trainer as trainer_module

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
    assert adapted.callbacks[2].filename == "rescene4d_t2_to_t3_horizon_adapted"
    assert adapted.callbacks[2].every_n_epochs == 45
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


def test_reviewer_flags_enable_weighted_fail_closed_objective_without_p2_identity() -> (
    None
):
    config = SimpleNamespace(
        general=SimpleNamespace(
            reviewer_closure_weighted_objective=True,
            reviewer_closure_fail_closed_runtime=True,
        )
    )
    owner = SimpleNamespace(
        config=config,
        criterion=SimpleNamespace(weight_dict={"loss_ce": 2.0}),
    )
    losses = {
        "loss_ce": torch.tensor(1.0),
        "loss_segment_contrastive_layer0": torch.tensor(100.0),
    }

    objective = trainer_module._configured_objective_loss(owner, losses)

    assert objective.item() == 2.0
    assert trainer_module._runtime_safety_enabled(config) is True
    assert trainer_module._weighted_objective_enabled(config) is True
    assert trainer_module._p2_general_flag(config, "p2_fail_closed_runtime") is False


def test_reviewer_runtime_filters_frozen_optimizer_parameters() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)
    _, adapted_config = training.compose_t3_adaptation_config(recipe)
    adapted_config.scheduler.scheduler.total_steps = 2
    trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    class OptimizerOwner:
        config = adapted_config

        @staticmethod
        def parameters():
            return iter((trainable, frozen))

    optimizers, _ = trainer_module.InstanceSegmentation.configure_optimizers(
        OptimizerOwner()
    )

    assert optimizers[0].param_groups[0]["params"] == [trainable]
    assert training_entrypoint._is_formal_p2_training(adapted_config) is False


def test_t3_batch_semantics_require_three_stages_and_valid_segment_mapping() -> None:
    data = SimpleNamespace(
        temporal_stages=[torch.tensor([0, 0, 1, 1, 2, 2])],
        features=torch.ones(6, 3),
    )
    targets = [
        {
            "labels": torch.tensor([1, 2]),
            "point2segment": torch.tensor([0, 0, 1, 1, 2, 2]),
            "segment_mask": torch.tensor([[True, False, False], [False, True, True]]),
        }
    ]

    result = t3_runner.validate_t3_batch_semantics(data, targets, ["sample"])

    assert result == {
        "batch_size": 1,
        "point_count": 6,
        "supervised_instances": 2,
        "temporal_stages": [0, 1, 2],
        "segment_count": 3,
    }


def test_real_t3_loader_materializes_exact_three_stage_batch() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)
    _, config = training.compose_t3_adaptation_config(recipe)

    mixed, _, _, names, sample = t3_runner._materialize_t3_smoke_batch(
        config, torch.device("cpu")
    )

    assert sample["temporal_stages"] == [0, 1, 2]
    assert sample["rio_raw_train_sequence_count"] == 858
    assert sample["rio_active_train_sequence_count"] == 855
    assert sample["rio_excluded_empty_supervision_windows"] == 3
    assert sample["mixed_epoch_samples"] == 1536
    assert [len(dataset) for dataset in mixed.datasets] == [855, 1199]
    assert names == [sample["sample_name"]]


def test_strict_adaptation_load_rejects_partial_state(tmp_path: Path) -> None:
    source = torch.nn.Linear(2, 2)
    checkpoint = tmp_path / "partial.ckpt"
    torch.save({"state_dict": {"weight": source.weight.detach().clone()}}, checkpoint)

    with pytest.raises(RuntimeError, match="strict weights-only"):
        t3_runner.strict_load_adaptation_weights(source, checkpoint)


def test_completed_training_runtime_requires_exact_frozen_budget() -> None:
    recipe = training.load_t3_adaptation_recipe(RECIPE_PATH)
    runtime = {
        "status": "completed",
        "world_size": 2,
        "completed_epochs": 45,
        "optimizer_updates": 2160,
        "global_sample_exposures": 69120,
        "global_scan_exposures": 138240,
        "rio_t3_sample_exposures": 34560,
        "scannet_t1_sample_exposures": 34560,
        "wall_clock_seconds": 3600.0,
        "gpu_hours": 2.0,
        "peak_allocated_vram_mib": 12000.0,
    }

    validated = t3_runner.validate_completed_training_runtime(runtime, recipe)

    assert validated["global_sample_exposures"] == 69120
