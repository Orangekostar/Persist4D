"""Exercise the actual Lightning checkpoint hooks and sliced global sampler."""

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from datasets.native_long_dataset import GlobalDrawSampler
from scripts.native_long_execution import UpdateBoundary
from trainer.native_long_trainer import NativeLongTrainer
from trainer.task_memory_trainer import restore_task_memory_rng_state


class DrawDataset(Dataset):
    def __len__(self):
        return 29700 * 32

    def __getitem__(self, index):
        return torch.tensor([index % 17 / 17.]), index


class ResumeFixture(NativeLongTrainer):
    def __init__(self):
        pl.LightningModule.__init__(self)
        self.config = OmegaConf.create({"trainer": {"accumulate_grad_batches": 16}, "general": {}})
        self.next_global_draw = 0
        self.native_identity = {"test": "real-lightning-optimizer-boundary"}
        self._native_pending_rng = None
        self._native_batches_since_resume = 0
        self.layer = torch.nn.Linear(1, 1)
        self.draws = []

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if self._native_pending_rng:
            restore_task_memory_rng_state(self._native_pending_rng[0])
            self._native_pending_rng = None
        return DataLoader(DrawDataset(), batch_size=2, sampler=GlobalDrawSampler(
            next_draw=self.next_global_draw, total_updates=29700, rank=0,
            world_size=1, per_rank_batch=2))

    def training_step(self, batch, batch_idx):
        inputs, indices = batch
        self.draws.extend(indices.tolist())
        self._native_batches_since_resume += 1
        return self.layer(inputs).square().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=.0005)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=.0005,
                                                       total_steps=29700)
        return [optimizer], [{"scheduler": schedule, "interval": "step"}]

    def on_before_optimizer_step(self, optimizer):
        pass

    def on_train_epoch_start(self):
        pass


def fit(model, steps, path=None, callbacks=None):
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=steps, max_epochs=-1,
                         accumulate_grad_batches=16, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         num_sanity_val_steps=0, callbacks=callbacks)
    trainer.fit(model, ckpt_path=str(path) if path else None, weights_only=False)
    return trainer


def test_optimizer_boundary_resume_preserves_draw_lr_and_parameters(tmp_path: Path):
    torch.manual_seed(45)
    continuous = ResumeFixture()
    full = fit(continuous, 4)
    torch.manual_seed(45)
    first = ResumeFixture()
    split = fit(first, 2)
    checkpoint = tmp_path / "boundary.ckpt"
    split.save_checkpoint(checkpoint)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["native_long_resume"]["next_global_draw"] == 64
    resumed = ResumeFixture()
    final = fit(resumed, 4, checkpoint)
    assert first.draws + resumed.draws == continuous.draws == list(range(128))
    assert full.optimizers[0].param_groups[0]["lr"] == final.optimizers[0].param_groups[0]["lr"]
    assert full.lr_scheduler_configs[0].scheduler.state_dict() == final.lr_scheduler_configs[0].scheduler.state_dict()
    assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in continuous.state_dict().items())


@pytest.mark.parametrize("budget_stop", [False, True])
def test_resumed_callback_waits_for_next_optimizer_boundary(tmp_path, budget_stop):
    torch.manual_seed(45)
    continuous = ResumeFixture()
    full = fit(continuous, 4)
    torch.manual_seed(45)
    first = ResumeFixture()
    split = fit(first, 2)
    checkpoint = tmp_path / "boundary.ckpt"
    split.save_checkpoint(checkpoint)
    resumed = ResumeFixture()
    callback = UpdateBoundary(tmp_path, "E0", 45, 4)
    if budget_stop:
        callback.gpu_hour_limit = .001
    final = fit(resumed, 4, checkpoint, [callback])
    expected = 3 if budget_stop else 4
    assert final.global_step == expected
    assert first.draws + resumed.draws == list(range(expected * 32))
    saved = torch.load(tmp_path / "training/E0/seed45/last.ckpt", weights_only=False)
    assert saved["global_step"] == expected
    assert saved["native_long_resume"]["next_global_draw"] == expected * 32
    assert saved["lr_schedulers"][0]["last_epoch"] == expected
    if not budget_stop:
        assert full.lr_scheduler_configs[0].scheduler.state_dict() == final.lr_scheduler_configs[0].scheduler.state_dict()
        assert all(torch.equal(v, resumed.state_dict()[k]) for k, v in continuous.state_dict().items())


def test_periodic_checkpoint_resume_with_actual_lightning_callback(tmp_path):
    torch.manual_seed(45)
    first = ResumeFixture()
    split = fit(first, 990)
    checkpoint = tmp_path / "periodic.ckpt"
    split.save_checkpoint(checkpoint)
    resumed = ResumeFixture()
    final = fit(resumed, 992, checkpoint, [UpdateBoundary(tmp_path, "E0", 45, 992)])
    assert final.global_step == 992
    assert first.draws + resumed.draws == list(range(992 * 32))
    saved = torch.load(tmp_path / "training/E0/seed45/last.ckpt", weights_only=False)
    assert saved["global_step"] == saved["lr_schedulers"][0]["last_epoch"] == 992
    assert saved["native_long_resume"]["next_global_draw"] == 992 * 32
