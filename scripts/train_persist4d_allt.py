#!/usr/bin/env python3
"""Train one frozen-budget Persist4D All-T comparison variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from datasets.persist4d_sequence_dataset import (
    Persist4DEpisodeCollator,
    Persist4DEpisodeDataset,
    Persist4DMixedEpisodeDataset,
    build_episode_draw_plan,
    build_episode_masters,
    build_mixed_source_schedule,
    build_single_scan_draw_plan,
    build_single_scan_masters,
)
from datasets.semseg import SemanticSegmentationDataset
from models.persist4d_allt import strict_load_r1_with_named_adapters
from scripts.p6a_protocol import load_t5_masters
from trainer.persist4d_allt_trainer import Persist4DAllTTrainer

R1_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
R1_BYTES = 754_813_672
FORMAL_OPTIMIZER_UPDATES = 400
DEFAULT_CHECKPOINT = Path(
    "/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/"
    f"{R1_SHA256}.ckpt"
)
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT / "artifacts/allt_task_superiority_v1/split_manifest.json"
)
DEFAULT_EXTERNAL_ROOT = Path(
    "/mnt/shared/ww/persist4d-allt-task-superiority-v1/training"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1/training"

VARIANTS = {
    "C0": {
        "local_enhancement_enabled": False,
        "memory_read_enabled": False,
        "window_mode": "local_pair",
    },
    "C1": {
        "local_enhancement_enabled": True,
        "memory_read_enabled": False,
        "window_mode": "local_pair",
    },
    "C2": {
        "local_enhancement_enabled": False,
        "memory_read_enabled": True,
        "window_mode": "local_pair",
    },
    "C3": {
        "local_enhancement_enabled": True,
        "memory_read_enabled": True,
        "window_mode": "local_pair",
    },
    "FH-adapt": {
        "local_enhancement_enabled": False,
        "memory_read_enabled": False,
        "window_mode": "full_history",
    },
    "FH-L": {
        "local_enhancement_enabled": True,
        "memory_read_enabled": False,
        "window_mode": "full_history",
    },
}


class AllTTrainingError(RuntimeError):
    """Raised when a formal training input or invariant differs."""


def _adapter_missing_prefixes(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise AllTTrainingError(f"unsupported All-T variant: {variant}")
    specification = VARIANTS[variant]
    prefixes = []
    if specification["local_enhancement_enabled"]:
        prefixes.append("model.local_enhancement.")
    if specification["memory_read_enabled"]:
        prefixes.append("model.memory_read.")
    return tuple(prefixes)


def _checkpoint_interval(optimizer_updates: int, *, smoke: bool) -> int:
    if smoke:
        return 1
    if optimizer_updates != FORMAL_OPTIMIZER_UPDATES or optimizer_updates % 4:
        raise AllTTrainingError("formal updates do not match the frozen quarters")
    return optimizer_updates // 4


def _git_head() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise AllTTrainingError("Git HEAD is invalid")
    return value


def _summarize_gradient_steps(
    steps: list[dict[str, object]],
) -> dict[str, object]:
    nonzero_counts: dict[str, int] = {}
    frozen = set()
    nonfinite = set()
    for step in steps:
        for name in step.get("nonzero_adapter_gradients", []):
            nonzero_counts[str(name)] = nonzero_counts.get(str(name), 0) + 1
        frozen.update(str(name) for name in step.get("frozen_gradient_names", []))
        nonfinite.update(
            str(name) for name in step.get("nonfinite_gradient_names", [])
        )
    return {
        "frozen_gradient_names": sorted(frozen),
        "nonfinite_gradient_names": sorted(nonfinite),
        "nonzero_gradient_step_counts": dict(sorted(nonzero_counts.items())),
        "optimizer_step_calls": len(steps),
    }


def _capture_adapter_state(
    system: Persist4DAllTTrainer, prefixes: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in system.state_dict().items()
        if name.startswith(prefixes)
    }


def _parameter_update_summary(
    initial: Mapping[str, torch.Tensor],
    current: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    if set(initial) != set(current):
        raise AllTTrainingError("adapter state keys changed during training")
    changed = {}
    for name in sorted(initial):
        delta = (current[name].detach().cpu() - initial[name]).abs()
        if torch.count_nonzero(delta).item():
            changed[name] = {
                "max_abs_delta": float(delta.max().item()),
                "norm_delta": float(delta.norm().item()),
            }
    return {
        "changed_parameter_count": len(changed),
        "changed_parameters": changed,
        "parameter_count": len(initial),
    }


def _workload_counts(
    rio_horizon_counts: Mapping[int, int],
    scannet_count: int,
    *,
    window_mode: str,
) -> dict[str, int]:
    if window_mode not in {"local_pair", "full_history"}:
        raise AllTTrainingError("All-T workload window mode is invalid")
    supervised = scannet_count + sum(
        horizon * count for horizon, count in rio_horizon_counts.items()
    )
    if window_mode == "local_pair":
        encoder_scans = scannet_count + sum(
            (1 + 2 * (horizon - 1)) * count
            for horizon, count in rio_horizon_counts.items()
        )
    else:
        encoder_scans = scannet_count + sum(
            horizon * (horizon + 1) // 2 * count
            for horizon, count in rio_horizon_counts.items()
        )
    return {
        "encoder_scan_count": encoder_scans,
        "supervised_stage_count": supervised,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AllTTrainingError(f"cannot load JSON input: {path}") from error
    if not isinstance(value, dict):
        raise AllTTrainingError(f"JSON input must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_r1_binding(path: Path) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != R1_BYTES
        or path.name != f"{R1_SHA256}.ckpt"
    ):
        raise AllTTrainingError("R1 checkpoint path/bytes differ from the verified binding")


def _compose_config(
    *,
    variant: str,
    pretrained: Path,
    run_dir: Path,
    optimizer_updates: int,
    devices: int,
    gradient_accumulation: int,
) -> Any:
    if variant not in VARIANTS:
        raise AllTTrainingError(f"unsupported All-T variant: {variant}")
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "conf").resolve()), version_base="1.2"
    ):
        config = compose(config_name="config_persist4d_allt")
    specification = VARIANTS[variant]
    with open_dict(config):
        config.general.experiment_name = variant
        config.general.save_dir = str(run_dir)
        config.general.gpus = devices
        config.backbone.name = str(pretrained)
        config.model.memory_read_enabled = specification["memory_read_enabled"]
        config.model.local_enhancement = (
            {
                "_target_": (
                    "models.query_competition_adapter.QueryCompetitionAdapter"
                ),
                "overlap_threshold": 0.25,
                "support_budget": 4096,
            }
            if specification["local_enhancement_enabled"]
            else None
        )
        config.allt_training.variant = variant
        config.allt_training.window_mode = specification["window_mode"]
        config.allt_training.memory_read_enabled = specification[
            "memory_read_enabled"
        ]
        config.allt_training.optimizer_updates = optimizer_updates
        config.allt_training.gradient_accumulation = gradient_accumulation
        config.allt_training.effective_episode_batch = (
            devices * gradient_accumulation
        )
        config.trainer.max_steps = optimizer_updates
        config.trainer.strategy = (
            "ddp_find_unused_parameters_true" if devices > 1 else "auto"
        )
    return config


def _resolved_data_path(value: object, data_root: Path) -> object:
    if not isinstance(value, str) or not value.startswith("data/"):
        return value
    return str(data_root / Path(value).relative_to("data"))


def _base_dataset(config: Any, *, source: str, data_root: Path):
    training = OmegaConf.to_container(
        config.data.train_dataset, resolve=True
    )
    if not isinstance(training, dict) or not isinstance(training.get("datasets"), list):
        raise AllTTrainingError("training dataset config is not the frozen RIO/ScanNet mix")
    excluded = {
        "_target_",
        "datasets",
        "epoch_sample_multiple",
        "sampler_seed",
        "weights",
    }
    common = {key: value for key, value in training.items() if key not in excluded}
    candidates = [
        item
        for item in training["datasets"]
        if isinstance(item, dict) and item.get("dataset_name") == source
    ]
    if len(candidates) != 1:
        raise AllTTrainingError(f"dataset config does not bind one {source} source")
    parameters = {**common, **candidates[0]}
    parameters.pop("target", None)
    parameters.pop("_target_", None)
    parameters["temporal_window"] = 5 if source == "rio" else 1
    for key in (
        "data_dir",
        "label_db_filepath",
        "change_label_db_filepath",
        "color_mean_std",
    ):
        if key in parameters:
            parameters[key] = _resolved_data_path(parameters[key], data_root)
    return SemanticSegmentationDataset(**parameters)


def _role_by_sequence(split_manifest: Mapping[str, object]) -> dict[str, str]:
    assignments = split_manifest.get("master_assignments")
    if not isinstance(assignments, list):
        raise AllTTrainingError("split manifest lacks master assignments")
    result = {}
    for record in assignments:
        if not isinstance(record, Mapping):
            raise AllTTrainingError("split master assignment is invalid")
        sequence_id = record.get("sequence_id")
        role = record.get("role")
        if (
            not isinstance(sequence_id, str)
            or role not in {"adaptation", "development"}
            or sequence_id in result
        ):
            raise AllTTrainingError("split master assignment identity is invalid")
        result[sequence_id] = str(role)
    return result


def _build_train_loader(
    config: Any,
    *,
    data_root: Path,
    metadata: Path,
    split_manifest: Path,
    devices: int,
    optimizer_updates: int,
    gradient_accumulation: int,
    smoke_contract: bool = False,
) -> tuple[DataLoader, dict[str, object]]:
    rio_base = _base_dataset(config, source="rio", data_root=data_root)
    scannet_base = _base_dataset(config, source="scannet", data_root=data_root)
    protocol_masters = load_t5_masters(
        data_root / "processed/rio/sequence_database_sliding_5.yaml",
        data_root / "processed/rio/train_database.yaml",
        metadata_path=metadata,
        expected_split="train",
        expected_master_count=262,
        expected_cluster_count=44,
        require_supervised=True,
        substitution_policy="reject",
    )
    roles = _role_by_sequence(_load_json(split_manifest))
    rio_masters = build_episode_masters(rio_base, protocol_masters, roles)
    scannet_masters = build_single_scan_masters(scannet_base, role="adaptation")
    group_count = optimizer_updates * gradient_accumulation
    if smoke_contract:
        if optimizer_updates != 2 or gradient_accumulation != 1:
            raise AllTTrainingError(
                "real smoke requires two updates and no accumulation"
            )
        source_schedule = tuple(
            source
            for source in ("scannet", "rio")
            for _ in range(devices)
        )
    else:
        source_schedule = build_mixed_source_schedule(
            group_count=group_count,
            replica_group_size=devices,
            primary_weight=float(config.allt_training.rio_weight),
            secondary_weight=float(config.allt_training.scannet_weight),
        )
    rio_count = source_schedule.count("rio")
    scannet_count = source_schedule.count("scannet")
    seed = int(config.general.seed)
    rio_plan = build_episode_draw_plan(
        rio_masters,
        role="adaptation",
        episode_count=rio_count,
        seed=seed,
        replica_group_size=devices,
        horizons=(3,) if smoke_contract else (2, 3, 4, 5),
    )
    scannet_plan = build_single_scan_draw_plan(
        scannet_masters,
        role="adaptation",
        episode_count=scannet_count,
        seed=seed,
        replica_group_size=devices,
    )
    window_mode = str(config.allt_training.window_mode)
    mixed = Persist4DMixedEpisodeDataset(
        {
            "rio": Persist4DEpisodeDataset(
                rio_base, rio_plan, window_mode=window_mode
            ),
            "scannet": Persist4DEpisodeDataset(
                scannet_base, scannet_plan, window_mode=window_mode
            ),
        },
        source_schedule,
    )
    stage_collator = hydra.utils.instantiate(config.data.train_collation)
    loader = DataLoader(
        mixed,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.data.num_workers),
        pin_memory=bool(config.data.pin_memory),
        collate_fn=Persist4DEpisodeCollator(stage_collator),
        persistent_workers=int(config.data.num_workers) > 0,
    )
    rio_horizon_counts = {
        horizon: sum(spec.horizon == horizon for spec in rio_plan)
        for horizon in range(2, 6)
    }
    plan_summary = {
        "adaptation_reference_count": len(
            {master.reference_id for master in rio_masters if master.role == "adaptation"}
        ),
        "development_reference_count": len(
            {master.reference_id for master in rio_masters if master.role == "development"}
        ),
        "global_episode_count": len(mixed),
        "group_count": group_count,
        "rio_episode_count": rio_count,
        "rio_horizon_counts": {
            str(horizon): count for horizon, count in rio_horizon_counts.items()
        },
        "scannet_episode_count": scannet_count,
        "source_schedule_sha256": hashlib.sha256(
            "\n".join(source_schedule).encode("ascii")
        ).hexdigest(),
        **_workload_counts(
            rio_horizon_counts,
            scannet_count,
            window_mode=window_mode,
        ),
    }
    return loader, plan_summary


def _load_r1(system: Persist4DAllTTrainer, checkpoint: Path, variant: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("state_dict"), Mapping
    ):
        raise AllTTrainingError("R1 checkpoint lacks a Lightning state_dict")
    state_dict = payload["state_dict"]
    missing_prefixes = _adapter_missing_prefixes(variant)
    if missing_prefixes:
        return strict_load_r1_with_named_adapters(
            system,
            state_dict,
            allowed_missing_prefixes=missing_prefixes,
        )
    incompatible = system.load_state_dict(state_dict, strict=True)
    return {
        "loaded_key_count": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--devices", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--optimizer-updates", type=int, default=FORMAL_OPTIMIZER_UPDATES
    )
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.optimizer_updates <= 0 or args.gradient_accumulation <= 0:
        raise AllTTrainingError("optimizer updates and accumulation must be positive")
    if not args.smoke and (
        args.optimizer_updates != FORMAL_OPTIMIZER_UPDATES
        or args.gradient_accumulation != 4
        or args.devices != 2
    ):
        raise AllTTrainingError("formal run differs from the frozen budget")
    checkpoint = args.checkpoint.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    metadata = args.metadata.expanduser().resolve(strict=True)
    split_manifest = args.split_manifest.expanduser().resolve(strict=True)
    _require_r1_binding(checkpoint)
    namespace = "smoke" if args.smoke else "formal"
    run_dir = (args.external_root / namespace / args.variant).resolve()
    artifact_dir = (args.artifact_root / namespace / args.variant).resolve()
    relaunched_worker = "LOCAL_RANK" in os.environ
    if (
        not args.smoke
        and not relaunched_worker
        and run_dir.exists()
        and any(run_dir.iterdir())
    ):
        raise AllTTrainingError(f"formal run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(45, workers=True)
    config = _compose_config(
        variant=args.variant,
        pretrained=pretrained,
        run_dir=run_dir,
        optimizer_updates=args.optimizer_updates,
        devices=args.devices,
        gradient_accumulation=args.gradient_accumulation,
    )
    loader, plan_summary = _build_train_loader(
        config,
        data_root=data_root,
        metadata=metadata,
        split_manifest=split_manifest,
        devices=args.devices,
        optimizer_updates=args.optimizer_updates,
        gradient_accumulation=args.gradient_accumulation,
        smoke_contract=args.smoke,
    )
    system = Persist4DAllTTrainer(config)
    load_audit = _load_r1(system, checkpoint, args.variant)
    adapter_prefixes = _adapter_missing_prefixes(args.variant)
    initial_adapter_state = _capture_adapter_state(system, adapter_prefixes)
    # Adapter construction may consume RNG; training stochasticity starts identically.
    seed_everything(45, workers=True)
    if not relaunched_worker:
        OmegaConf.save(config, artifact_dir / "resolved_config.yaml", resolve=True)
        _atomic_json(
            artifact_dir / "run_plan.json",
            {
                "checkpoint_sha256": R1_SHA256,
                "code_commit_at_run": _git_head(),
                "devices": args.devices,
                "gradient_accumulation": args.gradient_accumulation,
                "load_audit": load_audit,
                "optimizer_updates": args.optimizer_updates,
                "plan": plan_summary,
                "seed": 45,
                "variant": args.variant,
                "window_mode": str(config.allt_training.window_mode),
            },
        )
        initial_checkpoint = run_dir / "update=0000.ckpt"
        torch.save(
            {
                "allt_metadata": {
                    "r1_checkpoint_sha256": R1_SHA256,
                    "training_seed": 45,
                    "variant": args.variant,
                },
                "epoch": 0,
                "global_step": 0,
                "state_dict": system.state_dict(),
            },
            initial_checkpoint,
        )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="update={step:04d}",
        every_n_train_steps=_checkpoint_interval(
            args.optimizer_updates, smoke=args.smoke
        ),
        save_last=True,
        save_on_train_epoch_end=True,
        save_top_k=-1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=("ddp_find_unused_parameters_true" if args.devices > 1 else "auto"),
        max_epochs=1,
        max_steps=args.optimizer_updates,
        accumulate_grad_batches=1,
        precision="32-true",
        deterministic=False,
        gradient_clip_val=None,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=run_dir, name="training_metrics"),
        default_root_dir=run_dir,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    started = time.time()
    trainer.fit(system, train_dataloaders=loader)
    elapsed = time.time() - started
    if not trainer.is_global_zero:
        return 0
    gradient_audit = _summarize_gradient_steps(system.gradient_audit_steps)
    parameter_update_audit = _parameter_update_summary(
        initial_adapter_state,
        _capture_adapter_state(system, adapter_prefixes),
    )
    optimizer_parameter_ids = {
        id(parameter)
        for optimizer in trainer.optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_optimizer_parameters = sorted(
        name
        for name, parameter in system.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    )
    gradient_audit["missing_optimizer_parameters"] = missing_optimizer_parameters
    if gradient_audit["frozen_gradient_names"] or gradient_audit[
        "nonfinite_gradient_names"
    ] or missing_optimizer_parameters:
        raise AllTTrainingError("training gradient audit failed")
    if adapter_prefixes:
        expected_outputs = [
            name
            for name in initial_adapter_state
            if name.endswith("output_projection.weight")
        ]
        nonzero_counts = gradient_audit["nonzero_gradient_step_counts"]
        if any(not nonzero_counts.get(name, 0) for name in expected_outputs):
            raise AllTTrainingError("adapter output projection received no gradient")
    checkpoint_paths = sorted(run_dir.glob("*.ckpt"))
    manifests = [
        {
            "bytes": path.stat().st_size,
            "name": path.name,
            "sha256": _sha256(path),
        }
        for path in checkpoint_paths
    ]
    _atomic_json(
        artifact_dir / "checkpoint_manifest.json",
        {
            "checkpoints": manifests,
            "external_reference": (
                f"external:allt_task_superiority_v1/training/{namespace}/{args.variant}"
            ),
            "variant": args.variant,
        },
    )
    _atomic_json(
        artifact_dir / "run_summary.json",
        {
            "completed_global_step": int(trainer.global_step),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * args.devices / 3600.0,
            "execution_status": (
                "COMPLETE"
                if int(trainer.global_step) == args.optimizer_updates
                else "PARTIAL"
            ),
            "optimizer_updates_requested": args.optimizer_updates,
            "total_parameter_count": sum(
                parameter.numel() for parameter in system.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in system.parameters()
                if parameter.requires_grad
            ),
            "gradient_audit": gradient_audit,
            "parameter_update_audit": parameter_update_audit,
            "plan": plan_summary,
            "smoke": args.smoke,
            "variant": args.variant,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
