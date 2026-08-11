import hashlib
import json
from pathlib import Path

import hydra
import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import main_instance_segmentation as training_entrypoint

CONFIG_DIR = Path(__file__).resolve().parents[1] / "conf"
P2_CONFIG_NAME = "config_p2_rescene4d_concerto_t2"
FORMAL_TEST_MODEL_STATE_SCHEMA = {
    "model.frozen_buffer": {"shape": [3], "dtype": "torch.float32"},
    "model.weight": {"shape": [1], "dtype": "torch.float32"},
}
FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA = [
    ["model.weight", [1], "torch.float32"],
]


def _model_state_schema_sha256(schema: dict) -> str:
    entries = [
        [name, metadata["shape"], metadata["dtype"]]
        for name, metadata in sorted(schema.items())
    ]
    payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _parameter_schema_sha256(entries: list[list[object]]) -> str:
    payload = json.dumps(entries, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@pytest.fixture(autouse=True)
def _use_test_model_state_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_MODEL_STATE_SCHEMA_SHA256",
        _model_state_schema_sha256(FORMAL_TEST_MODEL_STATE_SCHEMA),
    )
    monkeypatch.setattr(
        training_entrypoint,
        "_P2_FORMAL_TRAINABLE_PARAMETER_SCHEMA_SHA256",
        _parameter_schema_sha256(FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA),
    )


class TinyResumeModule(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        return F.mse_loss(self.layer(inputs), targets)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class TinyDictConfigResumeModule(TinyResumeModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters(
            {"runtime_cfg": OmegaConf.create({"seed": 45})}
        )

    def train_dataloader(self):
        inputs = torch.tensor([[1.0], [2.0]])
        targets = torch.tensor([[2.0], [4.0]])
        return DataLoader(TensorDataset(inputs, targets), batch_size=1)


def _full_checkpoint(*, epoch: int, global_step: int) -> dict:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for _ in range(global_step):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=0,
    )
    return {
        "pytorch-lightning_version": "2.6.5",
        "state_dict": {"model.weight": torch.ones(1)},
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        "loops": {
            "fit_loop": {},
            "validate_loop": {},
            "test_loop": {},
            "predict_loop": {},
        },
        "epoch": epoch,
        "global_step": global_step,
        "callbacks": {
            checkpoint_callback.state_key: checkpoint_callback.state_dict()
        },
    }


def _save_checkpoint(path: Path, *, epoch: int, global_step: int) -> None:
    payload = _full_checkpoint(epoch=epoch, global_step=global_step)
    torch.save(payload, path)


def _add_sampler_generator_state(payload: dict) -> dict:
    generator = torch.Generator()
    generator.manual_seed(45)
    payload["p2_train_sampler_generator"] = {
        "schema_version": 1,
        "resume_scope": "completed_epoch_boundary_only",
        "mid_epoch_resume_supported": False,
        "dataloader_prefetch_state_checkpointed": False,
        "generator_state": generator.get_state(),
    }
    return payload


def _formal_config(save_dir: Path):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        cfg = compose(config_name=P2_CONFIG_NAME)
    cfg.general.save_dir = str(save_dir)
    with open_dict(cfg.callbacks[2]):
        cfg.callbacks[2].dirpath = str(save_dir / "callback-2")
    return cfg


def _formal_checkpoint(cfg, *, epoch: int, global_step: int) -> dict:
    payload = _full_checkpoint(epoch=epoch, global_step=global_step)
    payload["hyper_parameters"] = cfg
    payload["state_dict"]["model.frozen_buffer"] = torch.zeros(3)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=[parameter])
    scheduler_cfg = cfg.scheduler.scheduler.copy()
    scheduler_cfg.total_steps = 29_700
    scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
    for _ in range(global_step):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    callbacks = [
        hydra.utils.instantiate(callback_cfg)
        for callback_cfg in cfg.callbacks
        if callback_cfg.get("_target_")
        == "pytorch_lightning.callbacks.ModelCheckpoint"
    ]
    for callback in callbacks:
        interval = callback._every_n_epochs
        if interval is None or epoch + 1 < interval:
            continue
        callback_path = str(
            Path(callback.dirpath) / f"fixture-epoch={epoch:03d}.ckpt"
        )
        Path(callback_path).parent.mkdir(parents=True, exist_ok=True)
        Path(callback_path).touch()
        callback.best_model_path = callback_path
        if callback.monitor is not None:
            score = torch.tensor(0.5)
            callback.best_model_score = score
            callback.current_score = score.clone()
            callback.best_k_models = {callback_path: score.clone()}
            callback.kth_best_model_path = callback_path
            callback.kth_value = score.clone()
            last_path = Path(callback.dirpath) / "last.ckpt"
            last_path.touch()
            callback.last_model_path = str(last_path)
    payload["optimizer_states"] = [optimizer.state_dict()]
    payload["lr_schedulers"] = [scheduler.state_dict()]
    payload["callbacks"] = {
        callback.state_key: callback.state_dict() for callback in callbacks
    }
    parameter_id = payload["optimizer_states"][0]["param_groups"][0]["params"][0]
    model_state = {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in payload["state_dict"].items()
    }
    payload["p2_optimizer_parameter_contract"] = {
        "schema_version": 1,
        "state_dict": model_state,
        "state_dict_schema_sha256": _model_state_schema_sha256(model_state),
        "param_groups": [[parameter_id]],
        "parameters": {
            parameter_id: {
                "name": "model.weight",
                "shape": [1],
                "dtype": "torch.float32",
            }
        },
        "trainable_parameters": FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA.copy(),
        "trainable_parameter_schema_sha256": _parameter_schema_sha256(
            FORMAL_TEST_TRAINABLE_PARAMETER_SCHEMA
        ),
    }
    epoch_progress = {
        "ready": epoch + 1,
        "completed": epoch + 1,
        "started": epoch + 1,
        "processed": epoch + 1,
    }
    optimizer_progress = {"ready": global_step, "completed": global_step}
    batch_count = (epoch + 1) * 264
    batch_progress = {
        "ready": batch_count,
        "completed": batch_count,
        "started": batch_count,
        "processed": batch_count,
    }
    idle_progress = {
        "ready": 0,
        "completed": 0,
        "started": 0,
        "processed": 0,
    }
    payload["loops"] = {
        "fit_loop": {
            "state_dict": {},
            "epoch_progress": {
                "total": epoch_progress,
                "current": epoch_progress.copy(),
            },
            "epoch_loop.batch_progress": {
                "total": batch_progress,
                "current": {
                    "ready": 0,
                    "completed": 0,
                    "started": 0,
                    "processed": 0,
                },
                "is_last_batch": False,
            },
            "epoch_loop.state_dict": {"_batches_that_stepped": global_step},
            "epoch_loop.scheduler_progress": {
                "total": optimizer_progress,
                "current": {"ready": 0, "completed": 0},
            },
            "epoch_loop.automatic_optimization.optim_progress": {
                "optimizer": {
                    "step": {
                        "total": optimizer_progress,
                        "current": {"ready": 0, "completed": 0},
                    },
                    "zero_grad": {
                        "total": {
                            "ready": global_step,
                            "completed": global_step,
                            "started": global_step,
                        },
                        "current": {
                            "ready": 0,
                            "completed": 0,
                            "started": 0,
                        },
                    },
                }
            },
            "epoch_loop.val_loop.state_dict": {},
            "epoch_loop.val_loop.batch_progress": {
                "total": idle_progress.copy(),
                "current": idle_progress.copy(),
                "is_last_batch": False,
            },
        },
        "validate_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": idle_progress.copy(),
                "current": idle_progress.copy(),
            },
        },
        "test_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": idle_progress.copy(),
                "current": idle_progress.copy(),
            },
        },
        "predict_loop": {
            "state_dict": {},
            "batch_progress": {
                "total": idle_progress.copy(),
                "current": idle_progress.copy(),
            },
        },
    }
    return payload


def test_find_best_tap_checkpoint_parses_standard_lightning_filename(
    tmp_path: Path,
) -> None:
    lower = tmp_path / "epoch=8-val_mean_t-AP=0.348.ckpt"
    higher = tmp_path / "epoch=4-val_mean_t-AP=0.525.ckpt"
    _save_checkpoint(lower, epoch=8, global_step=80)
    _save_checkpoint(higher, epoch=4, global_step=40)

    selected = training_entrypoint.find_best_tap_checkpoint(tmp_path)

    assert selected == str(higher)


def test_resume_selects_newer_epoch_over_stale_last_checkpoint(
    tmp_path: Path,
) -> None:
    last = tmp_path / "last.ckpt"
    newer = tmp_path / "epoch=9-val_mean_t-AP=0.348.ckpt"
    _save_checkpoint(last, epoch=2, global_step=20)
    _save_checkpoint(newer, epoch=9, global_step=90)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(newer)


def test_resume_includes_last_versions_and_uses_numeric_version_tie_break(
    tmp_path: Path,
) -> None:
    last = tmp_path / "last.ckpt"
    version_nine = tmp_path / "last-v9.ckpt"
    version_ten = tmp_path / "last-v10.ckpt"
    _save_checkpoint(last, epoch=7, global_step=70)
    _save_checkpoint(version_nine, epoch=8, global_step=80)
    _save_checkpoint(version_ten, epoch=8, global_step=80)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(version_ten)


def test_resume_skips_corruption_and_selects_latest_valid_epoch(
    tmp_path: Path,
) -> None:
    (tmp_path / "last.ckpt").write_bytes(b"corrupt")
    (tmp_path / "epoch=10-val_mean_t-AP=0.900.ckpt").write_bytes(b"corrupt")
    last_epoch = tmp_path / "last-epoch.ckpt"
    latest_valid = tmp_path / "epoch=8-val_mean_t-AP=0.200.ckpt"
    best_tap_without_epoch = tmp_path / "run-val_mean_t-AP=0.800.ckpt"
    _save_checkpoint(last_epoch, epoch=7, global_step=70)
    _save_checkpoint(latest_valid, epoch=8, global_step=80)
    _save_checkpoint(best_tap_without_epoch, epoch=6, global_step=60)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(latest_valid)


def test_formal_resume_skips_newer_checkpoint_without_sampler_generator_state(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=7-val_mean_t-AP=0.200.ckpt"
    newer_without_sampler = tmp_path / "epoch=8-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=7, global_step=528)
        ),
        older_complete,
    )
    torch.save(
        _formal_checkpoint(cfg, epoch=8, global_step=594),
        newer_without_sampler,
    )

    selected = training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    )

    assert selected == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_from_different_config(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    other_cfg = _formal_config(tmp_path)
    other_cfg.general.seed = cfg.general.seed + 1
    older_matching = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_mismatch = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_matching,
    )
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(other_cfg, epoch=1, global_step=132)
        ),
        newer_mismatch,
    )

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_matching)


def test_formal_resume_skips_newer_symlink_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_regular = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    newer_target = target_dir / "newer.ckpt"
    newer_symlink = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_regular,
    )
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=1, global_step=132)
        ),
        newer_target,
    )
    newer_symlink.symlink_to(newer_target)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_regular)


def test_formal_resume_skips_newer_checkpoint_from_wrong_lightning_version(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_matching = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_wrong_version = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_matching,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    invalid["pytorch-lightning_version"] = "2.6.4"
    torch.save(invalid, newer_wrong_version)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_matching)


def test_formal_resume_skips_out_of_range_epoch_without_aborting_fallback(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_out_of_range = tmp_path / "epoch=450-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=0, global_step=66)
    )
    invalid["epoch"] = 450
    invalid["global_step"] = 29_766
    invalid["lr_schedulers"][0]["last_epoch"] = 29_766
    invalid["lr_schedulers"][0]["_step_count"] = 29_767
    torch.save(invalid, newer_out_of_range)

    selected = training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    )

    assert selected == str(older_complete)


def test_formal_resume_selects_fixed_final_callback_checkpoint(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "checkpoints" / "rescene4d_concerto_t2_repro"
    save_dir.mkdir(parents=True)
    cfg = _formal_config(save_dir)
    with open_dict(cfg.callbacks[2]):
        cfg.callbacks[2].dirpath = str(save_dir.parent)
    final_checkpoint = save_dir.parent / "rescene4d_concerto_t2_repro.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=449, global_step=29_700)
        ),
        final_checkpoint,
    )

    assert training_entrypoint.find_resume_checkpoint(
        save_dir,
        formal_p2=True,
        cfg=cfg,
    ) == str(final_checkpoint)


def test_formal_resume_skips_newer_checkpoint_with_missing_model_state(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_incomplete = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    invalid["state_dict"].pop("model.frozen_buffer")
    torch.save(invalid, newer_incomplete)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_with_complex_callback_score(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_invalid = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    monitor_state = next(iter(invalid["callbacks"].values()))
    monitor_state["best_model_score"] = torch.tensor(0.5 + 0.0j)
    torch.save(invalid, newer_invalid)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_with_nonfinite_model_tensor(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_invalid = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    invalid["state_dict"]["model.weight"] = torch.tensor([float("nan")])
    torch.save(invalid, newer_invalid)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_with_sparse_model_tensor(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_invalid = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    invalid["state_dict"]["model.weight"] = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        size=(1,),
    )
    torch.save(invalid, newer_invalid)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_with_dangling_best_history(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_invalid = tmp_path / "epoch=1-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=1, global_step=132)
    )
    monitor_state = next(
        state
        for state in invalid["callbacks"].values()
        if state["monitor"] is not None
    )
    Path(monitor_state["best_model_path"]).unlink()
    torch.save(invalid, newer_invalid)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_formal_resume_skips_newer_checkpoint_with_dangling_last_history(
    tmp_path: Path,
) -> None:
    cfg = _formal_config(tmp_path)
    older_complete = tmp_path / "epoch=0-val_mean_t-AP=0.200.ckpt"
    newer_invalid = tmp_path / "epoch=14-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(
            _formal_checkpoint(cfg, epoch=0, global_step=66)
        ),
        older_complete,
    )
    invalid = _add_sampler_generator_state(
        _formal_checkpoint(cfg, epoch=14, global_step=990)
    )
    monitor_state = next(
        state
        for state in invalid["callbacks"].values()
        if state["monitor"] is not None
    )
    monitor_state["last_model_path"] = str(tmp_path / "missing-last.ckpt")
    torch.save(invalid, newer_invalid)

    assert training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
        cfg=cfg,
    ) == str(older_complete)


def test_non_p2_resume_accepts_checkpoint_without_sampler_generator_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch=8-val_mean_t-AP=0.900.ckpt"
    _save_checkpoint(checkpoint, epoch=8, global_step=80)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(checkpoint)


def test_non_p2_resume_keeps_legacy_shallow_state_compatibility(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    payload = _full_checkpoint(epoch=1, global_step=1)
    payload["optimizer_states"][0]["state"] = {}
    payload["lr_schedulers"] = [{"legacy_scheduler_state": True}]
    payload["callbacks"] = {"legacy_callback": {}}
    torch.save(payload, checkpoint)

    assert training_entrypoint.find_resume_checkpoint(tmp_path) == str(checkpoint)


def test_formal_resume_selection_requires_the_current_config(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires current config"):
        training_entrypoint.find_resume_checkpoint(tmp_path, formal_p2=True)


def test_resume_skips_weights_only_and_partial_training_checkpoints(
    tmp_path: Path,
) -> None:
    weights_only = _full_checkpoint(epoch=9, global_step=90)
    weights_only.pop("optimizer_states")
    weights_only.pop("lr_schedulers")
    torch.save(weights_only, tmp_path / "last.ckpt")

    partial = _full_checkpoint(epoch=8, global_step=80)
    partial["optimizer_states"] = [{}]
    torch.save(partial, tmp_path / "epoch=8-val_mean_t-AP=0.900.ckpt")

    valid = tmp_path / "run-val_mean_t-AP=0.700.ckpt"
    _save_checkpoint(valid, epoch=7, global_step=70)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(valid)


def test_resume_same_epoch_prefers_global_step_then_numeric_version(
    tmp_path: Path,
) -> None:
    lower_step = tmp_path / "epoch=8-val_mean_t-AP=0.900-v11.ckpt"
    version_nine = tmp_path / "epoch=8-val_mean_t-AP=0.300-v9.ckpt"
    version_ten = tmp_path / "epoch=8-val_mean_t-AP=0.200-v10.ckpt"
    _save_checkpoint(lower_step, epoch=8, global_step=79)
    _save_checkpoint(version_nine, epoch=8, global_step=80)
    _save_checkpoint(version_ten, epoch=8, global_step=80)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(version_ten)


def test_resume_tries_next_best_tap_when_higher_tap_is_corrupt(
    tmp_path: Path,
) -> None:
    (tmp_path / "run-val_mean_t-AP=0.900.ckpt").write_bytes(b"corrupt")
    valid = tmp_path / "run-val_mean_t-AP=0.700.ckpt"
    _save_checkpoint(valid, epoch=3, global_step=30)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(valid)


def test_find_best_tap_skips_corrupt_and_incomplete_higher_metrics(
    tmp_path: Path,
) -> None:
    (tmp_path / "run-val_mean_t-AP=0.950.ckpt").write_bytes(b"corrupt")
    incomplete = _full_checkpoint(epoch=5, global_step=50)
    incomplete.pop("optimizer_states")
    torch.save(incomplete, tmp_path / "run-val_mean_t-AP=0.900.ckpt")
    valid = tmp_path / "run-val_mean_t-AP=0.700.ckpt"
    _save_checkpoint(valid, epoch=4, global_step=40)

    selected = training_entrypoint.find_best_tap_checkpoint(tmp_path)

    assert selected == str(valid)


def test_actual_lightning_checkpoint_is_selected_and_restores_after_bad_newer_files(
    tmp_path: Path,
) -> None:
    inputs = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([[2.0], [4.0]])
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    source_callback = ModelCheckpoint(
        dirpath=tmp_path / "source",
        save_last=True,
        save_top_k=0,
    )
    source_trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[source_callback],
    )
    source_trainer.fit(TinyResumeModule(), train_loader)

    valid = tmp_path / "epoch=1-val_mean_t-AP=0.700.ckpt"
    source_trainer.save_checkpoint(valid)
    valid_payload = torch.load(valid, map_location="cpu", weights_only=False)
    incomplete_payload = dict(valid_payload)
    incomplete_payload["optimizer_states"] = [{}]
    torch.save(
        incomplete_payload,
        tmp_path / "epoch=99-val_mean_t-AP=0.900.ckpt",
    )
    (tmp_path / "last.ckpt").write_bytes(b"corrupt")

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(valid)
    restored_trainer = Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=tmp_path / "restored",
                save_last=True,
                save_top_k=0,
            )
        ],
    )
    restored_trainer.fit(
        TinyResumeModule(),
        train_loader,
        ckpt_path=selected,
    )
    assert restored_trainer.global_step > valid_payload["global_step"]


def test_training_entrypoint_restores_real_dictconfig_lightning_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "last.ckpt"
    source_callback = ModelCheckpoint(
        dirpath=tmp_path / "source",
        save_last=True,
        save_top_k=0,
    )
    source_trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[source_callback],
    )
    source_trainer.fit(TinyDictConfigResumeModule())
    source_trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert type(payload["hyper_parameters"]["runtime_cfg"]).__name__ == "DictConfig"

    cfg = OmegaConf.create(
        {
            "general": {
                "save_dir": str(tmp_path),
                "experiment_name": "dictconfig-resume-test",
                "project_name": "tests",
                "gpus": 1,
            },
            "logging": [],
            "callbacks": [],
            "trainer": {},
        }
    )
    restored_trainers = []

    def cpu_trainer(**_kwargs):
        trainer = Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                ModelCheckpoint(
                    dirpath=tmp_path / "source",
                    save_last=True,
                    save_top_k=0,
                )
            ],
        )
        restored_trainers.append(trainer)
        return trainer

    monkeypatch.setattr(training_entrypoint, "Trainer", cpu_trainer)
    monkeypatch.setattr(
        training_entrypoint,
        "get_parameters",
        lambda candidate_cfg: (candidate_cfg, TinyDictConfigResumeModule()),
    )

    training_entrypoint.train.__wrapped__(cfg)

    assert restored_trainers[0].global_step > payload["global_step"]


def test_resume_returns_none_when_checkpoint_directory_is_empty(
    tmp_path: Path,
) -> None:
    assert training_entrypoint.find_resume_checkpoint(tmp_path) is None


def test_resume_fails_when_checkpoint_files_exist_but_none_are_resumable(
    tmp_path: Path,
) -> None:
    (tmp_path / "last.ckpt").write_bytes(b"corrupt")
    incomplete = _full_checkpoint(epoch=3, global_step=30)
    incomplete["optimizer_states"] = [{}]
    torch.save(incomplete, tmp_path / "epoch=3-val_mean_t-AP=0.900.ckpt")

    with pytest.raises(RuntimeError, match="none are fully resumable"):
        training_entrypoint.find_resume_checkpoint(tmp_path)
