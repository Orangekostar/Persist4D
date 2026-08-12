import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "conf"


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        return compose(config_name=config_name, overrides=overrides or [])


@contextmanager
def _load_trainer_module(monkeypatch):
    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_mean = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch_scatter", torch_scatter)

    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = object
    lightning.utilities = types.SimpleNamespace(grad_norm=lambda *args, **kwargs: {})
    monkeypatch.setitem(sys.modules, "pytorch_lightning", lightning)

    previous_module = sys.modules.pop("trainer.trainer", None)
    trainer_package = sys.modules.get("trainer")
    missing = object()
    previous_attribute = (
        getattr(trainer_package, "trainer", missing)
        if trainer_package is not None
        else missing
    )
    loaded_module = importlib.import_module("trainer.trainer")
    try:
        yield loaded_module
    finally:
        sys.modules.pop("trainer.trainer", None)
        trainer_package = sys.modules.get("trainer")
        if previous_module is not None:
            sys.modules["trainer.trainer"] = previous_module
        if trainer_package is not None:
            if previous_attribute is missing:
                if getattr(trainer_package, "trainer", None) is loaded_module:
                    delattr(trainer_package, "trainer")
            else:
                trainer_package.trainer = previous_attribute


def test_base_contrastive_override_composes_after_set_criterion() -> None:
    default_cfg = _compose("config_base_instance_segmentation")
    contrastive_cfg = _compose(
        "config_base_instance_segmentation",
        overrides=["loss/contrastive=infoNCE"],
    )

    assert default_cfg.loss.contrastive_loss is False
    assert contrastive_cfg.loss.contrastive_loss is True
    assert contrastive_cfg.loss.contrastive_loss_type == "infoNCE"


def test_p2_config_locks_the_reproduction_contract(monkeypatch) -> None:
    monkeypatch.delenv("CONCERTO_CHECKPOINT", raising=False)
    monkeypatch.setenv("HOME", "/tmp/p2-test-home")
    cfg = _compose("config_p2_rescene4d_concerto_t2")

    assert cfg.general.seed == 45
    assert cfg.general.freeze == "backbone_encoder"
    assert cfg.general.gpus == 2
    assert cfg.general.p2_weighted_objective is True
    assert cfg.general.p2_fail_closed_runtime is True

    assert cfg.data.train_dataset._target_ == (
        "datasets.multi_dataset.MultiDataset.from_config"
    )
    assert [dataset.dataset_name for dataset in cfg.data.train_dataset.datasets] == [
        "rio",
        "scannet",
    ]
    assert [dataset.temporal_window for dataset in cfg.data.train_dataset.datasets] == [
        2,
        1,
    ]
    assert list(cfg.data.train_dataset.weights) == [1.0, 0.8]
    assert cfg.data.train_dataset.fail_closed is True
    assert cfg.data.train_dataset.epoch_sample_multiple == 32
    assert cfg.data.train_dataset.sampler_seed == 45
    assert (
        cfg.data.train_dataset.known_empty_scan_policy
        == "official_substitute"
    )
    assert list(cfg.data.train_dataset.filter_out_classes) == [0, 1, 255]
    assert cfg.data.train_dataset.exclude_unsupervised_sequences is True
    assert cfg.data.validation_dataset.fail_closed is True
    assert cfg.data.validation_dataset.exclude_unsupervised_sequences is True
    assert (
        cfg.data.validation_dataset.known_empty_scan_policy
        == "official_substitute"
    )
    assert list(cfg.data.validation_dataset.filter_out_classes) == [0, 1, 255]
    assert cfg.data.test_dataset.fail_closed is True
    assert (
        cfg.data.test_dataset.known_empty_scan_policy
        == "official_substitute"
    )
    assert list(cfg.data.test_dataset.filter_out_classes) == [0, 1, 255]
    assert cfg.data.test_dataset.exclude_unsupervised_sequences is True
    assert list(cfg.data.train_collation.filter_out_classes) == [0, 1, 255]
    assert list(cfg.data.validation_collation.filter_out_classes) == [0, 1, 255]
    assert list(cfg.data.test_collation.filter_out_classes) == [0, 1, 255]
    assert cfg.data.validation_dataset.temporal_window == 2
    assert cfg.data.test_dataset.temporal_window == 2

    assert cfg.backbone._target_ == "models.PointceptBackbone"
    assert cfg.backbone.model_lib == "concerto"
    assert cfg.backbone.name == (
        "/tmp/p2-test-home/.cache/persist4d/concerto/concerto_base.pth"
    )
    assert list(cfg.backbone.decoder_serializations) == [
        "standard",
        "temporal_overlay",
    ]

    assert cfg.model.num_queries == 100
    assert cfg.model.non_parametric_queries is True
    assert cfg.model.random_query_both is False
    assert cfg.model.random_normal is False
    assert cfg.model.random_queries is False
    assert cfg.model.config.temporal_window == 2
    assert cfg.model.temporal_masking is False

    assert cfg.data.voxel_size == 0.02
    assert cfg.loss.contrastive_loss is True
    assert cfg.loss.contrastive_loss_type == "infoNCE"
    assert cfg.loss.eos_coef == 0.2
    assert (cfg.matcher.cost_class, cfg.matcher.cost_mask, cfg.matcher.cost_dice) == (
        2.0,
        5.0,
        2.0,
    )

    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.optimizer.lr == 5e-4
    assert cfg.scheduler.scheduler._target_ == ("torch.optim.lr_scheduler.OneCycleLR")
    assert cfg.scheduler.scheduler.max_lr == 5e-4
    assert cfg.scheduler.scheduler.total_steps == -1
    assert cfg.scheduler.pytorch_lightning_params.interval == "step"
    assert cfg.trainer.max_epochs == 450
    assert cfg.trainer.precision == "32-true"

    assert cfg.data.batch_size == 2
    assert cfg.data.train_dataloader.batch_size == 2
    assert cfg.trainer.accumulate_grad_batches == 8
    assert (
        cfg.general.gpus * cfg.data.batch_size * cfg.trainer.accumulate_grad_batches
        == 32
    )
    assert cfg.general.experiment_name == "rescene4d_concerto_t2_repro"
    assert cfg.general.save_dir == "checkpoints/rescene4d_concerto_t2_repro"
    assert cfg.general.project_name == "rescene4d_concerto_t2_repro"
    assert cfg.general.workspace is None
    assert len(cfg.logging) == 1
    assert cfg.logging[0]._target_ == "pytorch_lightning.loggers.CSVLogger"
    assert "entity" not in cfg.logging[0]
    assert OmegaConf.to_container(cfg.callbacks, resolve=True) == [
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "val_mean_t-AP",
            "mode": "max",
            "save_top_k": 1,
            "save_last": True,
            "dirpath": "checkpoints/rescene4d_concerto_t2_repro",
            "filename": (
                "epoch={epoch:03d}-val_mean_t-AP={val_mean_t-AP:.3f}"
            ),
            "every_n_epochs": 1,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": None,
            "save_top_k": -1,
            "save_last": False,
            "dirpath": "checkpoints/rescene4d_concerto_t2_repro",
            "filename": "periodic-epoch={epoch:03d}",
            "every_n_epochs": 25,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": None,
            "save_top_k": -1,
            "save_last": False,
            "dirpath": "checkpoints",
            "filename": "rescene4d_concerto_t2_repro",
            "every_n_epochs": 450,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
            "enable_version_counter": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
        },
    ]
    assert cfg.p2_preflight.target == "rescene4d_concerto_t2"
    assert cfg.p2_preflight.artifact_path == "artifacts/P2/scannet_preflight.json"

    config_source = yaml.safe_load(
        (CONFIG_DIR / "config_p2_rescene4d_concerto_t2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config_source["backbone"]["name"] == (
        "${oc.env:CONCERTO_CHECKPOINT,${oc.env:HOME}/.cache/persist4d/concerto/concerto_base.pth}"
    )


def test_p2_ignore_instance_filter_does_not_change_the_base_profile() -> None:
    cfg = _compose("config_base_instance_segmentation")

    assert list(cfg.data.train_dataset.filter_out_classes) == [0, 1]
    assert list(cfg.data.validation_dataset.filter_out_classes) == [0, 1]
    assert list(cfg.data.test_dataset.filter_out_classes) == [0, 1]


def test_p2_config_accepts_checkpoint_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CONCERTO_CHECKPOINT", "/checkpoints/concerto.pth")

    cfg = _compose("config_p2_rescene4d_concerto_t2")

    assert cfg.backbone.name == "/checkpoints/concerto.pth"


def test_p2_preflight_artifact_path_cannot_be_redirected_by_environment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("P2_SCANNET_PREFLIGHT", "/tmp/forged.json")

    cfg = _compose("config_p2_rescene4d_concerto_t2")

    assert cfg.p2_preflight.artifact_path == (
        "artifacts/P2/scannet_preflight.json"
    )


def test_p2_config_disables_the_non_g2_auxiliary_metric() -> None:
    cfg = _compose("config_p2_rescene4d_concerto_t2")

    assert cfg.aux_metric is None


def test_p2_config_locks_optimizer_scheduler_reproduction_defaults() -> None:
    cfg = _compose("config_p2_rescene4d_concerto_t2")

    assert OmegaConf.to_container(cfg.optimizer, resolve=True) == {
        "_target_": "torch.optim.AdamW",
        "lr": 5e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    assert OmegaConf.to_container(cfg.scheduler.scheduler, resolve=True) == {
        "_target_": "torch.optim.lr_scheduler.OneCycleLR",
        "max_lr": 5e-4,
        "epochs": 450,
        "total_steps": -1,
        "pct_start": 0.3,
        "anneal_strategy": "cos",
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "div_factor": 25.0,
        "final_div_factor": 10000.0,
        "three_phase": False,
        "last_epoch": -1,
    }


def test_objective_applies_segmentation_weights_without_counting_diagnostics(
    monkeypatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        aggregate_objective_loss = getattr(
            trainer_module, "aggregate_objective_loss", None
        )
        assert callable(aggregate_objective_loss), "objective helper is missing"

        losses = {
            "loss_ce": torch.tensor(1.0),
            "loss_mask": torch.tensor(2.0),
            "loss_dice": torch.tensor(3.0),
            "loss_ce_0": torch.tensor(4.0),
            "loss_mask_0": torch.tensor(5.0),
            "loss_dice_0": torch.tensor(6.0),
            "loss_segment_contrastive": torch.tensor(7.0),
            "loss_aux_contrastive": torch.tensor(8.0),
            "loss_changes_ce": torch.tensor(9.0),
            "loss_segment_contrastive_layer0": torch.tensor(100.0),
            "loss_aux_contrastive_layer_0": torch.tensor(200.0),
        }
        weight_dict = {
            "loss_ce": 2.0,
            "loss_mask": 5.0,
            "loss_dice": 2.0,
            "loss_ce_0": 2.0,
            "loss_mask_0": 5.0,
            "loss_dice_0": 2.0,
        }

        objective = aggregate_objective_loss(losses, weight_dict)

        assert objective.item() == 87.0


def test_objective_preserves_true_objective_gradients_and_excludes_diagnostics(
    monkeypatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        aggregate_objective_loss = getattr(
            trainer_module, "aggregate_objective_loss", None
        )
        assert callable(aggregate_objective_loss), "objective helper is missing"

        segmentation = torch.tensor(2.0, requires_grad=True)
        segment_contrastive = torch.tensor(3.0, requires_grad=True)
        aux_contrastive = torch.tensor(5.0, requires_grad=True)
        other_objective = torch.tensor(7.0, requires_grad=True)
        segment_diagnostic = torch.tensor(100.0, requires_grad=True)
        aux_diagnostic = torch.tensor(200.0, requires_grad=True)
        losses = {
            "loss_ce": segmentation,
            "loss_segment_contrastive": segment_contrastive,
            "loss_aux_contrastive": aux_contrastive,
            "loss_changes_ce": other_objective,
            "loss_segment_contrastive_layer0": segment_diagnostic,
            "loss_aux_contrastive_layer_0": aux_diagnostic,
        }

        objective = aggregate_objective_loss(losses, {"loss_ce": 2.0})
        objective.backward()

        assert objective.item() == 19.0
        assert segmentation.grad.item() == 2.0
        assert segment_contrastive.grad.item() == 1.0
        assert aux_contrastive.grad.item() == 1.0
        assert other_objective.grad.item() == 1.0
        assert segment_diagnostic.grad is None
        assert aux_diagnostic.grad is None


def test_objective_rejects_losses_without_trainable_objective(monkeypatch) -> None:
    with (
        _load_trainer_module(monkeypatch) as trainer_module,
        pytest.raises(ValueError, match="no objective loss terms"),
    ):
        trainer_module.aggregate_objective_loss(
            {"loss_segment_contrastive_layer0": torch.tensor(1.0)},
            {},
        )


def test_training_and_validation_use_the_weighted_objective_and_log_raw_losses(
    monkeypatch,
) -> None:
    with _load_trainer_module(monkeypatch) as trainer_module:
        instance = object.__new__(trainer_module.InstanceSegmentation)
        losses = {
            "loss_ce": torch.tensor(1.0),
            "loss_segment_contrastive": torch.tensor(2.0),
            "loss_segment_contrastive_layer0": torch.tensor(100.0),
        }

        def criterion(*args, **kwargs):
            return losses

        criterion.weight_dict = {"loss_ce": 2.0}
        logged = []
        instance.config = types.SimpleNamespace(
            general=types.SimpleNamespace(
                max_batch_size=10,
                use_dbscan=False,
                p2_weighted_objective=True,
                p2_fail_closed_runtime=True,
            )
        )
        instance.mask_type = "segment_mask"
        instance.criterion = criterion
        instance.forward = lambda *args, **kwargs: {
            "pred_logits": torch.zeros((1, 1, 2)),
            "pred_masks": [torch.zeros((1, 1))],
        }
        instance._process_raw_coordinates = lambda data: None
        instance.log_dict = lambda values, **kwargs: logged.append(values)

        data = types.SimpleNamespace(
            features=torch.zeros(1, 1),
            coordinates=torch.ones(1, 1),
            batch_size=1,
            inverse_maps=[],
            target_full=[],
            original_colors=[],
            idx=[],
            original_normals=[],
            original_coordinates=[],
        )
        target = [
            {
                "labels": torch.tensor([0]),
                "point2segment": torch.tensor([0]),
            }
        ]
        batch = (data, target, ["scene"])

        train_objective = instance.training_step(batch, 0)
        instance._process_predictions = lambda **kwargs: []
        instance.instance_metric = types.SimpleNamespace(update=lambda *args: None)
        instance.aux_metric = None
        validation_objective = instance._eval_step(batch, "val")

        assert train_objective.item() == 4.0
        assert validation_objective.item() == 4.0
        assert logged[0]["train_loss"].item() == 4.0
        assert logged[1]["val_loss"].item() == 4.0
        assert logged[0]["train_loss_segment_contrastive_layer0"].item() == 100.0
        assert logged[1]["val_loss_segment_contrastive_layer0"].item() == 100.0
