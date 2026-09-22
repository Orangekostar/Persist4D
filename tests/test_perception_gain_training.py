from __future__ import annotations

import importlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch


def _training_module():
    try:
        return importlib.import_module("trainer.perception_gain_trainer")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception training module is unavailable: {error}")


def _runner_module():
    try:
        return importlib.import_module("scripts.train_perception_gain")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception training runner is unavailable: {error}")


class _TinyPerceptionModel(torch.nn.Module):
    def __init__(self, *, scorer: bool = False) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.base = torch.nn.Linear(3, 2)
        if scorer:
            self.model.semantic_query_scorer = torch.nn.Linear(2, 1)


def test_strict_r1_load_allows_only_the_named_q_sem_scorer() -> None:
    training = _training_module()
    source = _TinyPerceptionModel()
    target = _TinyPerceptionModel(scorer=True)

    audit = training.strict_load_r1_perception(
        target,
        source.state_dict(),
        allow_semantic_scorer=True,
    )

    assert audit["loaded_key_count"] == len(source.state_dict())
    assert audit["r1_subtree_exact"] is True
    assert audit["missing_keys"] == [
        "model.semantic_query_scorer.bias",
        "model.semantic_query_scorer.weight",
    ]
    broken = dict(source.state_dict())
    broken.pop("model.base.bias")
    with pytest.raises(training.PerceptionTrainingError, match="unapproved missing"):
        training.strict_load_r1_perception(
            target,
            broken,
            allow_semantic_scorer=True,
        )
    with pytest.raises(training.PerceptionTrainingError, match="unexpected"):
        training.strict_load_r1_perception(
            target,
            {**source.state_dict(), "model.unknown": torch.ones(1)},
            allow_semantic_scorer=True,
        )


def test_scheduler_has_150_update_warmup_and_one_3000_update_cosine() -> None:
    training = _training_module()
    values = [training.perception_lr_multiplier(step) for step in range(3000)]

    assert values[0] == pytest.approx(1 / 150)
    assert values[149] == pytest.approx(1.0)
    assert values[150] == pytest.approx(1.0)
    assert values[749] > values[1499] > values[2249] > values[2999]
    assert values[-1] == pytest.approx(0.1)
    assert min(values) > 0.0


def test_common_weighted_sample_plan_is_variant_independent_and_resumable() -> None:
    runner = _runner_module()
    first = runner.build_weighted_sample_plan(
        dataset_sizes=(7, 11),
        nominal_weights=(1.0, 0.8),
        total_draws=320,
        seed=45,
    )
    repeated = runner.build_weighted_sample_plan(
        dataset_sizes=(7, 11),
        nominal_weights=(1.0, 0.8),
        total_draws=320,
        seed=45,
    )
    changed = runner.build_weighted_sample_plan(
        dataset_sizes=(7, 11),
        nominal_weights=(1.0, 0.8),
        total_draws=320,
        seed=46,
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.equal(first[160:], repeated[160:])
    source_count = int((first < 7).sum())
    assert 140 < source_count < 220

    def random_collator(samples):
        return (
            tuple(samples),
            random.random(),
            float(np.random.rand()),
            torch.rand(2),
        )

    collator = runner.DeterministicPerceptionCollator(random_collator, seed=45)
    batch = [
        runner.PerceptionDrawSample(draw_index=12, sample=("a",)),
        runner.PerceptionDrawSample(draw_index=13, sample=("b",)),
    ]
    left = collator(batch)
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    right = collator(batch)
    assert left[:3] == right[:3]
    assert torch.equal(left[3], right[3])


def test_runtime_contract_and_fixed_evaluation_boundaries() -> None:
    training = _training_module()
    contract = training.PerceptionRuntimeContract(
        devices=2,
        batch_size_per_gpu=2,
        gradient_accumulation=8,
        precision="32-true",
        gradient_clip_norm=1.0,
        total_updates=3000,
        evaluation_updates=(0, 250, 750, 1500, 2250, 3000),
    )

    assert contract.effective_batch_size == 32
    assert contract.total_global_draws == 96_000
    assert contract.checkpoint_updates == (250, 750, 1500, 2250, 3000)
    with pytest.raises(training.PerceptionTrainingError, match="precision"):
        training.PerceptionRuntimeContract(
            devices=2,
            batch_size_per_gpu=2,
            gradient_accumulation=8,
            precision="16-mixed",
            gradient_clip_norm=1.0,
            total_updates=3000,
            evaluation_updates=(0, 250, 750, 1500, 2250, 3000),
        )


def test_optimizer_contract_covers_every_trainable_parameter_once() -> None:
    training = _training_module()
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    module[0].requires_grad_(False)
    optimizer = torch.optim.AdamW(module[1].parameters(), lr=5e-5)

    audit = training.audit_optimizer_parameters(module, optimizer)

    assert audit["status"] == "PASS"
    assert audit["trainable_parameter_names"] == ["1.bias", "1.weight"]
    bad = torch.optim.AdamW([module[1].weight], lr=5e-5)
    with pytest.raises(training.PerceptionTrainingError, match="coverage"):
        training.audit_optimizer_parameters(module, bad)


def test_exact_resume_payload_preserves_cursor_rng_and_schedule_state() -> None:
    training = _training_module()
    progress = training.PerceptionProgress.initial().advance(
        optimizer_updates=17,
        local_batches=17 * 8,
        global_draws=17 * 32,
    )
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    payload = training.build_perception_resume_payload(progress)
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored = training.restore_perception_resume_payload(payload)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))

    assert restored == progress
    assert restored.next_global_draw_index == 544
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_all_arm_configs_preserve_shared_training_contract(tmp_path: Path) -> None:
    runner = _runner_module()
    expected_modes = {
        "C0": "legacy",
        "S-BAL": "balanced",
        "S-WORST": "worst",
        "Q-SEM": "legacy",
        "A-OPEN": "legacy",
    }
    configs = {
        variant: runner.compose_variant_config(
            variant,
            pretrained=tmp_path / "concerto.pth",
            run_dir=tmp_path / variant,
        )
        for variant in expected_modes
    }

    for variant, config in configs.items():
        assert config.loss.mask_loss_mode == expected_modes[variant]
        assert config.perception_training.optimizer_updates == 3000
        assert config.perception_training.warmup_updates == 150
        assert config.perception_training.evaluation_updates == [
            0,
            250,
            750,
            1500,
            2250,
            3000,
        ]
        assert config.data.batch_size == 2
        assert config.trainer.accumulate_grad_batches == 8
        assert config.trainer.precision == "32-true"
        assert config.trainer.gradient_clip_val == 1.0
    assert configs["Q-SEM"].model.semantic_query_positioning is True
    assert configs["A-OPEN"].model.open_first_cross_attention is True


def test_replication_cli_uses_explicit_seed_and_isolated_run_directory(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    arguments = runner._parser().parse_args(
        [
            "--variant",
            "C0",
            "--external-root",
            str(tmp_path),
            "--train-seed",
            "46",
            "--run-subdir",
            "replication/seed46/C0",
        ]
    )

    assert arguments.train_seed == 46
    assert (
        runner.resolve_training_run_dir(
            arguments.external_root,
            variant=arguments.variant,
            run_subdir=arguments.run_subdir,
        )
        == tmp_path / "training/replication/seed46/C0"
    )

    with pytest.raises(runner.PerceptionTrainingError, match="run subdirectory"):
        runner.resolve_training_run_dir(
            tmp_path,
            variant="C0",
            run_subdir=Path("../seed45"),
        )


def test_scorer_runner_uses_fixed_batch_and_emits_a_frozen_checkpoint(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    shard = tmp_path / "scorer.pt"
    torch.save(
        tuple(
            {
                "reference_id": f"r{index}",
                "segment_features": torch.stack((torch.zeros(128), torch.ones(128))),
                "targets": torch.tensor([0.0, 1.0]),
                "valid": torch.tensor([True, True]),
            }
            for index in range(8)
        ),
        shard,
    )

    summary = runner.train_semantic_scorer(
        cache_paths=(shard,),
        output_dir=tmp_path / "training",
        summary_output=tmp_path / "public/run_summary.json",
        stop_after_updates=2,
        device="cpu",
    )

    assert summary["batch_segments"] == 1024
    assert summary["completed_updates"] == 2
    assert summary["status"] == "SMOKE"
    assert all(row["gradient_norm"] > 0 for row in summary["training_audit"])
    assert all(row["parameter_change_norm"] > 0 for row in summary["training_audit"])
    checkpoint = torch.load(
        tmp_path / "training/update=0002.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["updates"] == 2
    assert set(checkpoint) == {
        "schema_version",
        "scorer_state_dict",
        "source_shards",
        "updates",
    }
    assert (tmp_path / "public/run_summary.json").is_file()
