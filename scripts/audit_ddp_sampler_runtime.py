#!/usr/bin/env python3
"""Inspect the resolved two-rank Lightning sampler and its actual streams."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch.utils.data import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_preflight import canonical_sha256
from utils.rescene_runtime_audit import (
    analyze_ddp_rank_streams,
    serialize_sampler_chain,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _compose_config() -> Any:
    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(config_name="config_rescene4d_concerto_rootcause")
    with open_dict(config):
        config.data.num_workers = 0
        config.data.pin_memory = False
        config.data.train_dataset.image_augmentations_path = None
        config.data.train_dataset.volume_augmentations_path = None
    return config


def _sample_reference(dataset: Any, index: int) -> tuple[str, str, int]:
    lower = 0
    for child in dataset.datasets:
        upper = lower + len(child)
        if index < upper:
            child_index = index - lower
            names = getattr(child, "sequence_names", None)
            sample_id = (
                str(names[child_index])
                if names is not None
                else Path(str(child.data[child_index]["filepath"])).stem
            )
            return str(child.dataset_name), sample_id, child_index
        lower = upper
    raise IndexError(f"sample index {index} exceeds the mixed dataset")


def _rank_positions(sampler: Any) -> list[int]:
    positions = DistributedSampler(
        list(range(len(sampler.dataset))),
        num_replicas=int(sampler.num_replicas),
        rank=int(sampler.rank),
        shuffle=bool(sampler.shuffle),
        seed=int(sampler.seed),
        drop_last=bool(sampler.drop_last),
    )
    positions.set_epoch(int(sampler.epoch))
    return [int(value) for value in positions]


class _SamplerProbe(pl.LightningModule):
    def __init__(self, config: Any, output_dir: Path) -> None:
        super().__init__()
        self.config = config
        self.output_dir = output_dir
        self.probe_weight = torch.nn.Parameter(torch.zeros(()))
        self.dataset = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.dataset = hydra.utils.instantiate(self.config.data.train_dataset)

    def train_dataloader(self):
        collate = hydra.utils.instantiate(self.config.data.train_collation)
        return hydra.utils.instantiate(
            self.config.data.train_dataloader,
            self.dataset,
            collate_fn=collate,
            sampler=self.dataset.sampler,
            shuffle=False,
        )

    def on_train_start(self) -> None:
        sampler = self.trainer.train_dataloader.sampler
        wrapper_dataset = getattr(sampler, "dataset", None)
        inner = getattr(wrapper_dataset, "_sampler", None)
        if inner is None:
            raise RuntimeError("Lightning did not wrap the custom sampler")
        observed = [int(value) for value in sampler]
        positions = _rank_positions(sampler)
        local = {
            "rank": int(self.global_rank),
            "chain": serialize_sampler_chain(sampler),
            "draws": observed,
            "positions": positions,
        }
        gathered: list[dict[str, object] | None] = [None] * int(self.trainer.world_size)
        torch.distributed.all_gather_object(gathered, local)
        if not self.trainer.is_global_zero:
            return
        records = [record for record in gathered if record is not None]
        if len(records) != 2:
            raise RuntimeError("sampler audit requires exactly two rank records")
        generator = torch.Generator().manual_seed(
            int(self.config.data.train_dataset.sampler_seed)
        )
        global_draws = torch.multinomial(
            inner.weights,
            int(inner.num_samples),
            bool(inner.replacement),
            generator=generator,
        ).tolist()
        rank_draws = {int(record["rank"]): record["draws"] for record in records}
        rank_positions = {
            int(record["rank"]): record["positions"] for record in records
        }
        analysis = analyze_ddp_rank_streams(
            global_draws=global_draws,
            rank_draws=rank_draws,
            rank_positions=rank_positions,
            world_size=2,
        )
        dataset_counts: dict[str, int] = {}
        rows = []
        for record in sorted(records, key=lambda item: int(item["rank"])):
            rank = int(record["rank"])
            for offset, (global_position, mixed_index) in enumerate(
                zip(record["positions"], record["draws"], strict=True)
            ):
                dataset_name, sample_id, child_index = _sample_reference(
                    self.dataset, int(mixed_index)
                )
                key = f"rank{rank}:{dataset_name}"
                dataset_counts[key] = dataset_counts.get(key, 0) + 1
                rows.append(
                    {
                        "rank": rank,
                        "rank_offset": offset,
                        "global_draw_position": int(global_position),
                        "mixed_dataset_index": int(mixed_index),
                        "dataset": dataset_name,
                        "child_index": child_index,
                        "sample_id": sample_id,
                    }
                )
        summary = {
            "schema_version": 1,
            "status": "pass",
            "source_commit": _git_head(),
            "topology": {"accelerator": "cuda", "world_size": 2},
            "sampler_chains": {
                str(record["rank"]): record["chain"] for record in records
            },
            "global_sampler": {
                "num_samples": len(global_draws),
                "generator_seed": int(self.config.data.train_dataset.sampler_seed),
                "contract_sha256": canonical_sha256(global_draws),
            },
            "dataset_draws": dataset_counts,
            "analysis": analysis,
        }
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _publish(
            self.output_dir / "ddp_sampler_rank_trace.csv",
            output.getvalue().encode("ascii"),
        )
        _publish(self.output_dir / "ddp_sampler_summary.json", _json_bytes(summary))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self.probe_weight.square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    pl.seed_everything(45, workers=True)
    config = _compose_config()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(_SamplerProbe(config, args.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
