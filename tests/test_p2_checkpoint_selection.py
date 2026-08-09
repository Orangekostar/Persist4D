from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

import main_instance_segmentation as training_entrypoint


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
    return {
        "pytorch-lightning_version": "2.6.5",
        "state_dict": {"model.weight": torch.ones(1)},
        "optimizer_states": [
            {
                "state": {},
                "param_groups": [{"params": [0], "lr": 1e-3}],
            }
        ],
        "lr_schedulers": [{"last_epoch": global_step}],
        "loops": {
            "fit_loop": {},
            "validate_loop": {},
            "test_loop": {},
            "predict_loop": {},
        },
        "epoch": epoch,
        "global_step": global_step,
        "callbacks": {"ModelCheckpoint": {"best_model_path": ""}},
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
    older_complete = tmp_path / "epoch=7-val_mean_t-AP=0.200.ckpt"
    newer_without_sampler = tmp_path / "epoch=8-val_mean_t-AP=0.900.ckpt"
    torch.save(
        _add_sampler_generator_state(_full_checkpoint(epoch=7, global_step=70)),
        older_complete,
    )
    _save_checkpoint(newer_without_sampler, epoch=8, global_step=80)

    selected = training_entrypoint.find_resume_checkpoint(
        tmp_path,
        formal_p2=True,
    )

    assert selected == str(older_complete)


def test_non_p2_resume_accepts_checkpoint_without_sampler_generator_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "epoch=8-val_mean_t-AP=0.900.ckpt"
    _save_checkpoint(checkpoint, epoch=8, global_step=80)

    selected = training_entrypoint.find_resume_checkpoint(tmp_path)

    assert selected == str(checkpoint)


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
