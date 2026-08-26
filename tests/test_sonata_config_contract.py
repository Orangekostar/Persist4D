from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from utils.sonata_second_preflight import (
    SONATA_CONFIG_NAME,
    validate_sonata_training_config_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "conf"


def _compose(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SONATA_CHECKPOINT", "/verified/sonata.pth")
    monkeypatch.setenv("SONATA_OUTPUT_DIR", "/training/sonata-second")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        return compose(config_name=SONATA_CONFIG_NAME)


@contextmanager
def _load_trainer_module(monkeypatch: pytest.MonkeyPatch):
    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_mean = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch_scatter", torch_scatter)

    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = object
    lightning.utilities = types.SimpleNamespace(grad_norm=lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)

    previous_module = sys.modules.pop("trainer.trainer", None)
    trainer_package = sys.modules.get("trainer")
    loaded_module = importlib.import_module("trainer.trainer")
    try:
        yield loaded_module
    finally:
        sys.modules.pop("trainer.trainer", None)
        if previous_module is not None:
            sys.modules["trainer.trainer"] = previous_module
        if trainer_package is not None:
            trainer_package.trainer = previous_module


def test_primary_sonata_config_locks_paper_supported_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose(monkeypatch)

    assert cfg.sonata_second_preflight.target == "sonata_second_perception"
    assert cfg.general.seed == 45
    assert cfg.general.freeze == "backbone_encoder"
    assert cfg.general.frozen_encoder_eval is False
    assert cfg.general.sonata_weighted_objective is True
    assert cfg.general.sonata_fail_closed_runtime is True
    assert cfg.general.gpus == 2
    assert cfg.general.workspace is None
    assert cfg.backbone.name == "/verified/sonata.pth"
    assert cfg.backbone.repo_id == "facebook/sonata"
    assert cfg.backbone.model_lib == "sonata"
    assert cfg.backbone.custom_config.enc_mode is False
    assert list(cfg.backbone.decoder_serializations) == [
        "standard",
        "temporal_overlay",
    ]
    assert cfg.model.num_queries == 100
    assert cfg.model.non_parametric_queries is True
    assert cfg.model.config.temporal_window == 2
    assert cfg.model.temporal_masking is True
    assert cfg.loss.contrastive_loss is False
    assert cfg.loss.eos_coef == 0.2
    assert (cfg.matcher.cost_class, cfg.matcher.cost_mask, cfg.matcher.cost_dice) == (
        2.0,
        5.0,
        2.0,
    )
    assert cfg.data.voxel_size == 0.02
    assert [item.dataset_name for item in cfg.data.train_dataset.datasets] == [
        "rio",
        "scannet",
    ]
    assert [item.temporal_window for item in cfg.data.train_dataset.datasets] == [
        2,
        1,
    ]
    assert list(cfg.data.train_dataset.weights) == [1.0, 0.8]
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.optimizer.lr == 5e-4
    assert cfg.scheduler.scheduler._target_ == "torch.optim.lr_scheduler.OneCycleLR"
    assert cfg.scheduler.scheduler.max_lr == 5e-4
    assert cfg.scheduler.pytorch_lightning_params.interval == "step"
    assert cfg.trainer.max_epochs == 450
    assert cfg.trainer.precision == "32-true"
    assert cfg.data.batch_size == 4
    assert cfg.trainer.accumulate_grad_batches == 4
    assert cfg.general.gpus * cfg.data.batch_size * cfg.trainer.accumulate_grad_batches == 32
    callbacks = OmegaConf.to_container(cfg.callbacks, resolve=True)
    checkpoints = [
        callback
        for callback in callbacks
        if callback.get("_target_")
        == "pytorch_lightning.callbacks.ModelCheckpoint"
    ]
    assert len(checkpoints) == 1
    assert checkpoints[0]["monitor"] == "val_mean_t-AP"
    assert checkpoints[0]["mode"] == "max"
    assert checkpoints[0]["save_top_k"] == 1
    assert checkpoints[0]["save_last"] is True

    assert validate_sonata_training_config_contract(
        cfg,
        expected_weight_path=Path("/verified/sonata.pth"),
        expected_output_dir=Path("/training/sonata-second"),
    ) == []


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    [
        ("model.temporal_masking", False, "model.temporal_masking"),
        ("loss.contrastive_loss", True, "loss.contrastive_loss"),
        ("loss.eos_coef", 0.1, "loss.eos_coef"),
        ("general.freeze", None, "general.freeze"),
        ("trainer.max_epochs", 449, "trainer.max_epochs"),
        ("trainer.accumulate_grad_batches", 8, "effective_global_batch"),
    ],
)
def test_config_contract_rejects_primary_recipe_drift(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    value: object,
    expected_error: str,
) -> None:
    cfg = _compose(monkeypatch)
    OmegaConf.update(cfg, path, value)

    errors = validate_sonata_training_config_contract(
        cfg,
        expected_weight_path=Path("/verified/sonata.pth"),
        expected_output_dir=Path("/training/sonata-second"),
    )

    assert any(expected_error in error for error in errors)


def test_sonata_runtime_flags_enable_weighted_fail_closed_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose(monkeypatch)
    with _load_trainer_module(monkeypatch) as trainer_module:
        owner = types.SimpleNamespace(
            config=cfg,
            criterion=types.SimpleNamespace(
                weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 2.0}
            ),
        )
        losses = {
            "loss_ce": torch.tensor(1.0),
            "loss_mask": torch.tensor(2.0),
            "loss_dice": torch.tensor(3.0),
        }

        objective = trainer_module._configured_objective_loss(owner, losses)

        assert trainer_module._runtime_safety_enabled(cfg) is True
        assert trainer_module._weighted_objective_enabled(cfg) is True
        assert objective.item() == 18.0


def test_sonata_config_does_not_contain_p2_identity_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = OmegaConf.to_container(_compose(monkeypatch), resolve=True)
    serialized = str(payload)

    assert "p2_preflight" not in payload
    assert "p2_weighted_objective" not in payload["general"]
    assert "p2_fail_closed_runtime" not in payload["general"]
    assert "rescene4d_concerto_t2" not in serialized
