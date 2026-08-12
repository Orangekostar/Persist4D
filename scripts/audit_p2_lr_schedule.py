#!/usr/bin/env python3
"""Audit P2 gradient accumulation and OneCycleLR step semantics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.p2_preflight import (
    P2_CONFIG_NAME,
    P2_FORMAL_EPOCH_SAMPLE_MULTIPLE,
    P2_PREFLIGHT_SCHEMA_VERSION,
    P2_RIO_SEQUENCE_FILTER_COUNTS,
    require_p2_preflight_authorization,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "P2"
DEFAULT_SCANNET_DIR = PROJECT_ROOT / "data" / "processed" / "scannet"
DEFAULT_SCANNET_PREFLIGHT = PROJECT_ROOT / "artifacts" / "P2" / "scannet_preflight.json"

SEED = 45
TARGET_GPUS = 2
TARGET_BATCH_PER_GPU = 2
TARGET_ACCUMULATION = 8
TARGET_PHYSICAL_GLOBAL_BATCH = TARGET_GPUS * TARGET_BATCH_PER_GPU
TARGET_EFFECTIVE_BATCH = TARGET_PHYSICAL_GLOBAL_BATCH * TARGET_ACCUMULATION
TARGET_EPOCHS = 450
TARGET_PRIMARY_DATASET_SAMPLES = P2_RIO_SEQUENCE_FILTER_COUNTS["train"][
    "retained_count"
]
TARGET_DATASET_WEIGHTS = (1.0, 0.8)
TARGET_EPOCH_SAMPLE_MULTIPLE = P2_FORMAL_EPOCH_SAMPLE_MULTIPLE
TARGET_SAMPLER_SEED = SEED
MAX_LR = 5e-4
SIMULATION_MICROBATCHES = 10
LIGHTNING_VERSION_CONTRACT = "2.6.5"
SCANNET_SPLIT_COUNTS = {"train": 1201, "validation": 312}
OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
SHARED_AUTHORIZATION_FIELDS = (
    "config_contract",
    "source_tree_contract",
    "runtime_source_contract",
    "runtime_environment_contract",
    "official_split_identity",
    "input_manifest",
    "authorization",
)

CSV_FIELDS = (
    "micro_step",
    "accumulation_window",
    "accumulation_window_size",
    "target_window_samples",
    "normalization_denominator_microbatches",
    "relative_gradient_scale",
    "optimizer_step_before",
    "optimizer_step_after",
    "global_step_before",
    "global_step_after",
    "scheduler_last_epoch_before",
    "scheduler_last_epoch_after",
    "lr_before",
    "lr_after",
    "did_optimizer_step",
)


class _CountingAdamW(torch.optim.AdamW):
    """AdamW with an observable count of completed real optimizer steps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.completed_steps = 0

    def step(self, closure=None):
        result = super().step(closure=closure)
        self.completed_steps += 1
        return result


class _SyntheticAutomaticOptimizationModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.automatic_optimization = True
        self.projection = nn.Linear(1, 1)
        self.simulation_total_steps: int | None = None
        self.resolved_max_lr: float | None = None

    def training_step(self, batch, batch_idx):
        features, targets = batch
        predictions = self.projection(features)
        return torch.nn.functional.mse_loss(predictions, targets)

    def configure_optimizers(self):
        optimizer = _CountingAdamW(
            self.parameters(),
            lr=MAX_LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False,
            last_epoch=-1,
        )
        self.simulation_total_steps = total_steps
        self.resolved_max_lr = float(optimizer.param_groups[0]["max_lr"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class _StepTraceCallback(pl.Callback):
    def __init__(self, *, microbatches: int, accumulation: int) -> None:
        self.microbatches = microbatches
        self.accumulation = accumulation
        self.rows: list[dict[str, Any]] = []
        self._before: dict[str, Any] | None = None

    @staticmethod
    def _state(trainer: pl.Trainer) -> dict[str, Any]:
        optimizer = trainer.optimizers[0]
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        return {
            "optimizer_step": int(optimizer.completed_steps),
            "global_step": int(trainer.global_step),
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._before = self._state(trainer)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._before is None:
            raise RuntimeError("missing start-of-microbatch audit state")
        after = self._state(trainer)
        before = self._before
        window_index = batch_idx // self.accumulation + 1
        window_start = (window_index - 1) * self.accumulation
        window_size = min(self.accumulation, self.microbatches - window_start)
        self.rows.append(
            {
                "micro_step": batch_idx + 1,
                "accumulation_window": window_index,
                "accumulation_window_size": window_size,
                "target_window_samples": TARGET_PHYSICAL_GLOBAL_BATCH * window_size,
                "normalization_denominator_microbatches": self.accumulation,
                "relative_gradient_scale": window_size / self.accumulation,
                "optimizer_step_before": before["optimizer_step"],
                "optimizer_step_after": after["optimizer_step"],
                "global_step_before": before["global_step"],
                "global_step_after": after["global_step"],
                "scheduler_last_epoch_before": before["scheduler_last_epoch"],
                "scheduler_last_epoch_after": after["scheduler_last_epoch"],
                "lr_before": before["lr"],
                "lr_after": after["lr"],
                "did_optimizer_step": (
                    after["optimizer_step"] > before["optimizer_step"]
                ),
            }
        )
        self._before = None


def _synthetic_loader() -> DataLoader:
    sample_count = SIMULATION_MICROBATCHES * TARGET_BATCH_PER_GPU
    features = torch.linspace(-1.0, 1.0, sample_count).reshape(-1, 1)
    targets = 3.0 * features - 0.25
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=TARGET_BATCH_PER_GPU,
        shuffle=False,
        num_workers=0,
    )


def _resolve_database_asset(
    value: Any,
    *,
    processed_scannet_dir: Path,
    split: str,
    asset_subdir: str | None = None,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path]
    candidates.extend(
        [
            processed_scannet_dir / path,
            processed_scannet_dir / split / path.name,
        ]
    )
    if asset_subdir is not None:
        candidates.append(processed_scannet_dir / asset_subdir / split / path.name)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _scannet_processed_assets_complete(processed_scannet_dir: Path) -> bool:
    required = (
        "train_database.yaml",
        "validation_database.yaml",
        "scannet.yaml",
        "label_database.yaml",
    )
    if not all((processed_scannet_dir / name).is_file() for name in required):
        return False
    try:
        metadata = yaml.safe_load(
            (processed_scannet_dir / "scannet.yaml").read_text(encoding="utf-8")
        )
        label_database = yaml.safe_load(
            (processed_scannet_dir / "label_database.yaml").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, Mapping) or not metadata:
            return False
        if not isinstance(label_database, Mapping) or not label_database:
            return False
        for split, expected_count in SCANNET_SPLIT_COUNTS.items():
            database = yaml.safe_load(
                (processed_scannet_dir / f"{split}_database.yaml").read_text(
                    encoding="utf-8"
                )
            )
            if not isinstance(database, list) or len(database) != expected_count:
                return False
            for record in database:
                if not isinstance(record, Mapping):
                    return False
                if (
                    not isinstance(record.get("file_len"), int)
                    or record["file_len"] < 1
                ):
                    return False
                point_file = _resolve_database_asset(
                    record.get("filepath"),
                    processed_scannet_dir=processed_scannet_dir,
                    split=split,
                )
                instance_file = _resolve_database_asset(
                    record.get("instance_gt_filepath"),
                    processed_scannet_dir=processed_scannet_dir,
                    split=split,
                    asset_subdir="instance_gt",
                )
                if point_file is None or point_file.suffix != ".npy":
                    return False
                if instance_file is None:
                    return False
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    return True


def _portable_preflight_ref(scannet_preflight_path: Path) -> str:
    try:
        relative = scannet_preflight_path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return "external:injected/scannet_preflight.json"
    return f"repo:{relative.as_posix()}"


def _compose_p2_config() -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "conf"), version_base="1.2"
    ):
        return compose(config_name=P2_CONFIG_NAME)


def _shared_p2_authorization_gate(path: Path) -> bool:
    try:
        cfg = _compose_p2_config()
        require_p2_preflight_authorization(cfg, artifact_path=path)
    except Exception:  # noqa: BLE001 - the ready gate must fail closed.
        return False
    return True


def _complete_scannet_gate_passed(scannet_preflight_path: Path) -> bool:
    try:
        payload = json.loads(scannet_preflight_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    expected_scalars = {
        "schema_version": P2_PREFLIGHT_SCHEMA_VERSION,
        "status": "pass",
        "formal_p2_training_authorized": True,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "split_metadata_status": "pass",
        "expected_split_counts": {"train": 1201, "validation": 312, "test": 100},
        "errors": [],
    }
    if any(payload.get(key) != value for key, value in expected_scalars.items()):
        return False
    if not all(
        isinstance(payload.get(field), Mapping)
        for field in SHARED_AUTHORIZATION_FIELDS
    ):
        return False
    local_gate_passed = all(
        isinstance(payload.get(section), Mapping)
        and payload[section].get("status") == "pass"
        for section in (
            "raw_assets",
            "processed_assets",
            "class_taxonomy",
            "mix_instantiation",
        )
    )
    return local_gate_passed and _shared_p2_authorization_gate(scannet_preflight_path)


def _formal_training_status(
    processed_scannet_dir: Path,
    scannet_preflight_path: Path,
) -> dict[str, Any]:
    preflight_ref = _portable_preflight_ref(scannet_preflight_path)
    gate_passed = _complete_scannet_gate_passed(scannet_preflight_path)
    assets_complete = _scannet_processed_assets_complete(processed_scannet_dir)
    primary_weight = TARGET_DATASET_WEIGHTS[0]
    secondary_weight_sum = sum(TARGET_DATASET_WEIGHTS[1:])
    raw_sampler_num_samples = int(
        TARGET_PRIMARY_DATASET_SAMPLES
        * (1 + secondary_weight_sum / primary_weight)
    )
    sampler_num_samples = (
        raw_sampler_num_samples
        - raw_sampler_num_samples % TARGET_EPOCH_SAMPLE_MULTIPLE
    )
    samples_per_rank = (sampler_num_samples + TARGET_GPUS - 1) // TARGET_GPUS
    epoch_microbatches = (
        samples_per_rank + TARGET_BATCH_PER_GPU - 1
    ) // TARGET_BATCH_PER_GPU
    accumulation_remainder = epoch_microbatches % TARGET_ACCUMULATION
    optimizer_steps_per_epoch = (
        epoch_microbatches + TARGET_ACCUMULATION - 1
    ) // TARGET_ACCUMULATION
    planned_contract = {
        "contract_kind": "planned_not_observed",
        "observed_formal_run": False,
        "dataset_mix": "3RScan T=2 (1.0) + ScanNet T=1 (0.8)",
        "primary_dataset_samples": TARGET_PRIMARY_DATASET_SAMPLES,
        "dataset_weights": list(TARGET_DATASET_WEIGHTS),
        "raw_sampler_num_samples": raw_sampler_num_samples,
        "epoch_sample_multiple": TARGET_EPOCH_SAMPLE_MULTIPLE,
        "sampler_num_samples": sampler_num_samples,
        "sampler_seed": TARGET_SAMPLER_SEED,
        "sampler_seed_scope": (
            "fresh_start_and_completed_epoch_boundary_resume"
        ),
        "sampler_generator_state_checkpointed": True,
        "sampler_checkpoint_scope": "completed_epoch_boundary_only",
        "sampler_checkpoint_save_timing": (
            "p2_normalized_train_epoch_end_callbacks"
        ),
        "sampler_non_boundary_resume_verified": False,
        "sampler_mid_epoch_resume_supported": False,
        "sampler_dataloader_prefetch_state_checkpointed": False,
        "samples_per_rank": samples_per_rank,
        "epochs": TARGET_EPOCHS,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "total_steps": optimizer_steps_per_epoch * TARGET_EPOCHS,
        "scannet_ref": "repo:data/processed/scannet",
        "preflight_ref": preflight_ref,
        "epoch_microbatch_divisibility": {
            "status": (
                "planned_aligned"
                if accumulation_remainder == 0
                else "planned_tail_window"
            ),
            "scope": "per_rank",
            "epoch_microbatches": epoch_microbatches,
            "accumulation_steps": TARGET_ACCUMULATION,
            "remainder": accumulation_remainder,
            "drop_last": False,
        },
    }
    if not gate_passed or not assets_complete:
        return {
            "status": "blocked_missing_scannet",
            **planned_contract,
            "reason": (
                "ScanNet prerequisites are missing, so no formal mixed-data run was "
                "observed; the planned sampler and optimizer-step contract remains "
                "computable from the locked P2 configuration."
            ),
        }
    return {
        "status": "deferred_to_formal_mixed_data_preflight",
        **planned_contract,
        "reason": (
            "This scheduler-only preflight records the locked planned contract but "
            "does not instantiate or observe a formal mixed-data training run."
        ),
    }


def _validate_trace(rows: Sequence[Mapping[str, Any]], expected_steps: int) -> None:
    if len(rows) != SIMULATION_MICROBATCHES:
        raise AssertionError("runtime did not consume all synthetic microbatches")
    optimizer_events = []
    lr_change_events = []
    for row in rows:
        optimizer_delta = int(row["optimizer_step_after"]) - int(
            row["optimizer_step_before"]
        )
        global_delta = int(row["global_step_after"]) - int(row["global_step_before"])
        scheduler_delta = int(row["scheduler_last_epoch_after"]) - int(
            row["scheduler_last_epoch_before"]
        )
        if optimizer_delta not in (0, 1):
            raise AssertionError(
                "each microbatch may complete at most one optimizer step"
            )
        if global_delta != optimizer_delta or scheduler_delta != optimizer_delta:
            raise AssertionError(
                "optimizer, Lightning global_step, and scheduler must advance together"
            )
        if int(row["optimizer_step_before"]) != int(row["global_step_before"]):
            raise AssertionError(
                "optimizer and global step diverged before a microbatch"
            )
        if int(row["optimizer_step_after"]) != int(row["global_step_after"]):
            raise AssertionError(
                "optimizer and global step diverged after a microbatch"
            )
        lr_changed = abs(float(row["lr_after"]) - float(row["lr_before"])) > 1e-15
        if lr_changed and not optimizer_delta:
            raise AssertionError("learning rate changed without an optimizer step")
        if optimizer_delta:
            optimizer_events.append(int(row["micro_step"]))
        if lr_changed:
            lr_change_events.append(int(row["micro_step"]))

    if optimizer_events != [8, 10]:
        raise AssertionError(f"unexpected accumulation boundaries: {optimizer_events}")
    if not lr_change_events:
        raise AssertionError(
            "OneCycleLR did not produce any observed learning-rate change"
        )
    if int(rows[-1]["global_step_after"]) != expected_steps:
        raise AssertionError(
            "final global_step does not equal estimated optimizer steps"
        )


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["lr_before"] = format(float(row["lr_before"]), ".17g")
            serialized["lr_after"] = format(float(row["lr_after"]), ".17g")
            serialized["relative_gradient_scale"] = format(
                float(row["relative_gradient_scale"]), ".17g"
            )
            serialized["did_optimizer_step"] = str(
                bool(row["did_optimizer_step"])
            ).lower()
            writer.writerow(serialized)


def _write_markdown(result: Mapping[str, Any], path: Path) -> None:
    runtime = result["runtime"]
    scheduler = result["scheduler"]
    topology = result["target_topology"]
    formal = result["formal_training"]
    divisibility = formal["epoch_microbatch_divisibility"]
    lines = [
        "# P2 LR schedule audit",
        "",
        "- scope: scheduler semantics preflight",
        f"- engine: {runtime['engine']}",
        f"- PyTorch Lightning: {runtime['lightning_version']}",
        f"- automatic optimization: {str(runtime['automatic_optimization']).lower()}",
        "- runtime: single-process CPU synthetic microbatches; not formal mixed-data training",
        f"- target topology: {topology['formula']}",
        "- accumulation windows: 8 + 2 microbatches",
        (
            "- tail-window demonstration only: tail target samples=4; "
            "normalization denominator microbatches=8; relative gradient scale=0.25"
        ),
        f"- simulated optimizer steps: {runtime['simulation_total_steps']}",
        f"- scheduler: {scheduler['name']}, interval={scheduler['interval']}",
        f"- max_lr contract: {scheduler['max_lr_contract']:.17g}",
        (
            "- the short simulation need not reach max_lr exactly; it verifies the "
            "configured ceiling and step semantics"
        ),
        (
            "- LR semantics: lr_before is applied to the current optimizer update; "
            "lr_after is scheduled for the next optimizer update"
        ),
        f"- formal status: {formal['status']}",
        f"- formal contract kind: {formal['contract_kind']}",
        f"- formal run observed: {str(formal['observed_formal_run']).lower()}",
        f"- formal epochs: {formal['epochs']}",
        f"- planned raw sampler num_samples: {formal['raw_sampler_num_samples']}",
        f"- planned epoch sample multiple: {formal['epoch_sample_multiple']}",
        f"- planned sampler num_samples: {formal['sampler_num_samples']}",
        f"- planned sampler seed: {formal['sampler_seed']}",
        f"- planned sampler seed scope: {formal['sampler_seed_scope']}",
        (
            "- sampler generator state checkpointed: "
            f"{str(formal['sampler_generator_state_checkpointed']).lower()}"
        ),
        f"- sampler checkpoint scope: {formal['sampler_checkpoint_scope']}",
        (
            "- sampler checkpoint save timing: "
            f"{formal['sampler_checkpoint_save_timing']}"
        ),
        (
            "- sampler non-boundary resume verified: "
            f"{str(formal['sampler_non_boundary_resume_verified']).lower()}"
        ),
        (
            "- sampler mid-epoch resume supported: "
            f"{str(formal['sampler_mid_epoch_resume_supported']).lower()}"
        ),
        (
            "- sampler DataLoader prefetch state checkpointed: "
            f"{str(formal['sampler_dataloader_prefetch_state_checkpointed']).lower()}"
        ),
        f"- planned samples per rank: {formal['samples_per_rank']}",
        f"- planned optimizer steps per epoch: {formal['optimizer_steps_per_epoch']}",
        f"- planned total_steps: {formal['total_steps']}",
        f"- planned epoch microbatch divisibility: {divisibility['status']}",
        f"- planned epoch microbatches per rank: {divisibility['epoch_microbatches']}",
        f"- planned accumulation remainder: {divisibility['remainder']}",
        (
            f"- formal readiness condition: epoch_microbatches % {TARGET_ACCUMULATION} == 0, or an "
            "explicit drop_last/tail-normalization policy; otherwise formal training "
            "is prohibited"
        ),
        f"- formal dataset ref: {formal['scannet_ref']}",
        f"- formal gate ref: {formal['preflight_ref']}",
        f"- formal status reason: {formal['reason']}",
        "",
        (
            "| micro | window | window size | target samples | norm denom | rel grad | "
            "optimizer before | optimizer after | global before | global after | "
            "scheduler before | scheduler after | LR before | LR after | optimizer step |"
        ),
        (
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | :---: |"
        ),
    ]
    for row in result["rows"]:
        lines.append(
            "| {micro_step} | {accumulation_window} | {accumulation_window_size} | "
            "{target_window_samples} | {normalization_denominator_microbatches} | "
            "{relative_gradient_scale:.17g} | "
            "{optimizer_step_before} | {optimizer_step_after} | "
            "{global_step_before} | {global_step_after} | "
            "{scheduler_last_epoch_before} | {scheduler_last_epoch_after} | "
            "{lr_before:.17g} | {lr_after:.17g} | {did_optimizer_step} |".format(**row)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    processed_scannet_dir: Path = DEFAULT_SCANNET_DIR,
    scannet_preflight_path: Path = DEFAULT_SCANNET_PREFLIGHT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    processed_scannet_dir = Path(processed_scannet_dir)
    scannet_preflight_path = Path(scannet_preflight_path)
    if pl.__version__ != LIGHTNING_VERSION_CONTRACT:
        raise RuntimeError(
            "This audit requires PyTorch Lightning "
            f"{LIGHTNING_VERSION_CONTRACT}, found {pl.__version__}."
        )
    pl.seed_everything(SEED, workers=True, verbose=False)

    callback = _StepTraceCallback(
        microbatches=SIMULATION_MICROBATCHES,
        accumulation=TARGET_ACCUMULATION,
    )
    model = _SyntheticAutomaticOptimizationModule()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="GPU available but not used.*")
        warnings.filterwarnings(
            "ignore", message="The 'train_dataloader' does not have many workers.*"
        )
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            accumulate_grad_batches=TARGET_ACCUMULATION,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[callback],
        )
        trainer.fit(model, train_dataloaders=_synthetic_loader())

    if model.simulation_total_steps is None or model.resolved_max_lr is None:
        raise RuntimeError("Lightning did not configure the OneCycle scheduler")
    _validate_trace(callback.rows, expected_steps=model.simulation_total_steps)
    observed_lrs = [
        float(value)
        for row in callback.rows
        for value in (row["lr_before"], row["lr_after"])
    ]
    if max(observed_lrs) > MAX_LR + 1e-15:
        raise AssertionError("observed learning rate exceeds the max_lr contract")

    result: dict[str, Any] = {
        "runtime": {
            "engine": "pytorch_lightning.Trainer.fit",
            "lightning_version": pl.__version__,
            "torch_version": torch.__version__,
            "automatic_optimization": model.automatic_optimization,
            "accelerator": "cpu",
            "simulation_devices": 1,
            "simulation_batch_per_process": TARGET_BATCH_PER_GPU,
            "simulation_microbatches": SIMULATION_MICROBATCHES,
            "simulation_total_steps": model.simulation_total_steps,
            "seed": SEED,
        },
        "target_topology": {
            "gpus": TARGET_GPUS,
            "batch_per_gpu": TARGET_BATCH_PER_GPU,
            "physical_global_batch": TARGET_PHYSICAL_GLOBAL_BATCH,
            "gradient_accumulation": TARGET_ACCUMULATION,
            "effective_batch": TARGET_EFFECTIVE_BATCH,
            "formula": "2 GPUs * 2 samples/GPU * 8 accumulation steps = 32",
        },
        "scheduler": {
            "name": "OneCycleLR",
            "interval": "step",
            "max_lr_contract": MAX_LR,
            "resolved_max_lr": model.resolved_max_lr,
            "sampled_max_lr": max(observed_lrs),
            "sampled_max_lr_must_equal_contract": False,
        },
        "formal_training": _formal_training_status(
            processed_scannet_dir, scannet_preflight_path
        ),
        "rows": callback.rows,
    }
    _write_csv(result["rows"], output_dir / "lr_schedule_audit.csv")
    _write_markdown(result, output_dir / "lr_schedule_audit.md")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--processed-scannet-dir", type=Path, default=DEFAULT_SCANNET_DIR
    )
    parser.add_argument(
        "--scannet-preflight", type=Path, default=DEFAULT_SCANNET_PREFLIGHT
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_audit(
        output_dir=args.output_dir,
        processed_scannet_dir=args.processed_scannet_dir,
        scannet_preflight_path=args.scannet_preflight,
    )
    formal_status = result["formal_training"]["status"]
    planned_total_steps = result["formal_training"]["total_steps"]
    formal_run_observed = result["formal_training"]["observed_formal_run"]
    print(
        "P2 LR scheduler audit complete: "
        f"optimizer_steps={result['runtime']['simulation_total_steps']}; "
        f"formal_status={formal_status}; "
        f"planned_total_steps={planned_total_steps}; "
        f"formal_run_observed={str(formal_run_observed).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
