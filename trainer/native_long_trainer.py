"""Native Lightning training with global-draw checkpoint boundaries."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.native_long_dataset import (
    GlobalDrawSampler,
    NativeDrawPlan,
    NativeLongCollator,
    NativeLongDataset,
    build_native_datasets,
    isolated_rng,
    stable_seed,
)
from trainer.task_memory_trainer import (
    _reset_lightning_batch_progress_for_sliced_resume,
    restore_task_memory_rng_state,
)
from trainer.trainer import InstanceSegmentation


class NativeLongTrainer(InstanceSegmentation):
    def __init__(self, config):
        self.next_global_draw = 0
        self.native_identity = None
        self._native_pending_rng = None
        self._native_batches_since_resume = 0
        super().__init__(config)

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            return
        population = json.loads(Path(self.config.native_long.population_file).read_text())
        self.native_datasets = build_native_datasets(
            self.config, self.config.native_long.data_root, augmentation=True)
        plan = NativeDrawPlan(population["references"],
                              scannet_count=len(self.native_datasets["scannet"]),
                              seed=int(self.config.general.seed))
        self.train_dataset = NativeLongDataset(self.native_datasets, plan,
                                               arm=self.config.native_long.arm)

    def train_dataloader(self):
        if self._native_pending_rng is not None:
            states = self._native_pending_rng
            if len(states) != self.trainer.world_size:
                raise ValueError("resume world size differs from bound optimizer trajectory")
            saved = states[self.global_rank]
            restore_task_memory_rng_state({**saved, "cuda": []})
            if saved["cuda"]:
                if len(saved["cuda"]) != 1 or self.device.type != "cuda":
                    raise ValueError("resume GPU RNG is not rank-local")
                torch.cuda.set_rng_state(saved["cuda"][0].cpu(), self.device)
            self._native_pending_rng = None
        sampler = GlobalDrawSampler(next_draw=self.next_global_draw, total_updates=29700,
                                    rank=self.global_rank, world_size=self.trainer.world_size,
                                    per_rank_batch=int(self.config.data.batch_size))
        return DataLoader(self.train_dataset, batch_size=int(self.config.data.batch_size),
                          sampler=sampler, num_workers=int(self.config.data.num_workers),
                          collate_fn=NativeLongCollator(self.config, training=True),
                          generator=torch.Generator().manual_seed(int(self.config.general.seed)),
                          pin_memory=True, drop_last=True)

    def training_step(self, batch, batch_idx):
        draw_ids = [int(name.split(":", 2)[1]) for name in batch[2]]
        self.model.native_long_draw_ids = draw_ids
        with isolated_rng(stable_seed(self.config.general.seed, "model", *draw_ids),
                          cuda_devices=(self.device.index,) if self.device.type == "cuda" else ()):
            result = super().training_step(batch, batch_idx)
        self._native_batches_since_resume += 1
        return result

    def on_save_checkpoint(self, checkpoint):
        update = int(checkpoint["global_step"])
        expected_batches = (update - self.next_global_draw // 32) * int(
            self.config.trainer.accumulate_grad_batches)
        if expected_batches != self._native_batches_since_resume:
            raise ValueError("checkpoint is not at a completed optimizer boundary")
        rng = {"torch": torch.random.get_rng_state().cpu().clone(),
               "numpy": np.random.get_state(), "python": random.getstate(),
               "cuda": [torch.cuda.get_rng_state(self.device).cpu().clone()]
               if self.device.type == "cuda" else []}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            states = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(states, rng)
        else:
            states = [rng]
        checkpoint["native_long_resume"] = {"schema": 1, "next_global_draw": update * 32,
                                             "identity": self.native_identity, "rng_by_rank": states}
        _reset_lightning_batch_progress_for_sliced_resume(
            checkpoint, completed_local_episodes=self._native_batches_since_resume)

    def on_load_checkpoint(self, checkpoint):
        payload = checkpoint.get("native_long_resume")
        if payload is None or payload["schema"] != 1 or payload["identity"] != self.native_identity:
            raise ValueError("resume identity differs from native model/data/config trajectory")
        if payload["next_global_draw"] != int(checkpoint["global_step"]) * 32:
            raise ValueError("checkpoint optimizer/draw positions disagree")
        self.next_global_draw = payload["next_global_draw"]
        self._native_pending_rng = payload["rng_by_rank"]
        self._native_batches_since_resume = 0

    def val_dataloader(self):
        return []
