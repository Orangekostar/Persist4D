#!/usr/bin/env python3
"""Run the frozen reviewer-closure T2-to-T3 adaptation protocol."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import reviewer_closure_training as protocol

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure"
SMOKE_ARTIFACT = ARTIFACT_ROOT / "t3_smoke_report.json"
TRAINING_MANIFEST = ARTIFACT_ROOT / "rescene_horizon_training_manifest.json"
RUNTIME_RECORD_NAME = "training_runtime.json"


class T3AdaptationError(RuntimeError):
    """Raised when the formal T3 adaptation contract is violated."""


def _as_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise T3AdaptationError(f"{name} must be a mapping")
    return value


def _finite_positive(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _atomic_runtime_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
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


def _git_source_commit(*, require_clean: bool) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if require_clean and status:
        raise T3AdaptationError(
            "formal T3 execution requires a clean source tree: "
            + ", ".join(line[3:] for line in status[:10])
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_smoke_source_binding(smoke_commit: str, current_commit: str) -> None:
    if smoke_commit == current_commit:
        return
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{smoke_commit}..{current_commit}"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if changed != ["artifacts/reviewer_closure/t3_smoke_report.json"]:
        raise T3AdaptationError("source changed after T3 smoke: " + ", ".join(changed))


def validate_t3_batch_semantics(
    data: object,
    targets: Sequence[Mapping[str, object]],
    names: Sequence[str],
) -> dict[str, object]:
    if len(names) != 1 or len(targets) != 1:
        raise T3AdaptationError("T3 smoke requires exactly one collated sample")
    stage_batches = getattr(data, "temporal_stages", None)
    features = getattr(data, "features", None)
    if not isinstance(stage_batches, (list, tuple)) or len(stage_batches) != 1:
        raise T3AdaptationError("collated batch has no per-sample temporal stages")
    stages = stage_batches[0]
    if not isinstance(stages, torch.Tensor) or stages.ndim != 1:
        raise T3AdaptationError("temporal stages must be a one-dimensional tensor")
    unique_stages = sorted(int(value) for value in torch.unique(stages).cpu().tolist())
    if unique_stages != [0, 1, 2]:
        raise T3AdaptationError(f"T3 batch has temporal stages {unique_stages}")
    if not isinstance(features, torch.Tensor) or features.shape[0] != stages.numel():
        raise T3AdaptationError("feature and temporal-stage rows differ")

    target = _as_mapping(targets[0], name="T3 target")
    labels = target.get("labels")
    point2segment = target.get("point2segment")
    segment_mask = target.get("segment_mask")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or not isinstance(point2segment, torch.Tensor)
        or point2segment.ndim != 1
        or not isinstance(segment_mask, torch.Tensor)
        or segment_mask.ndim != 2
    ):
        raise T3AdaptationError("T3 target mappings are incomplete")
    if point2segment.numel() != stages.numel():
        raise T3AdaptationError("point2segment and temporal-stage rows differ")
    if labels.numel() != segment_mask.shape[0] or labels.numel() == 0:
        raise T3AdaptationError("T3 instance labels and segment masks differ")
    segment_count = int(segment_mask.shape[1])
    if (
        segment_count == 0
        or int(point2segment.min().item()) < 0
        or int(point2segment.max().item()) >= segment_count
    ):
        raise T3AdaptationError("T3 point2segment values are outside segment masks")
    return {
        "batch_size": 1,
        "point_count": int(features.shape[0]),
        "supervised_instances": int(labels.numel()),
        "temporal_stages": unique_stages,
        "segment_count": segment_count,
    }


def strict_load_adaptation_weights(
    module: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    source = Path(checkpoint_path)
    if source.is_symlink() or not source.is_file():
        raise T3AdaptationError("adaptation source checkpoint is not a regular file")
    if expected_sha256 is not None:
        digest = protocol.sha256_file(source)
        if digest != expected_sha256:
            raise T3AdaptationError("adaptation source checkpoint SHA256 differs")
    else:
        digest = None
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = _as_mapping(checkpoint, name="adaptation checkpoint").get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise T3AdaptationError("adaptation checkpoint has no state_dict")
    try:
        incompatible = module.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"strict weights-only adaptation load failed: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise T3AdaptationError("strict weights-only adaptation load was incomplete")
    return {
        "state_dict_entry_count": len(state_dict),
        "checkpoint_sha256": digest,
        "optimizer_state_resumed": False,
        "scheduler_state_resumed": False,
    }


def validate_completed_training_runtime(
    runtime: Mapping[str, object], recipe: Mapping[str, object]
) -> dict[str, object]:
    optimization = _as_mapping(recipe.get("optimization"), name="optimization")
    data = _as_mapping(recipe.get("data"), name="data")
    expected_samples = int(data["mixed_epoch_samples"]) * int(optimization["epochs"])
    exact = {
        "status": "completed",
        "world_size": int(optimization["devices"]),
        "completed_epochs": int(optimization["epochs"]),
        "optimizer_updates": int(optimization["total_optimizer_updates"]),
        "global_sample_exposures": expected_samples,
    }
    for field, expected in exact.items():
        if runtime.get(field) != expected:
            raise T3AdaptationError(
                f"training runtime field {field} differs: "
                f"expected {expected!r}, got {runtime.get(field)!r}"
            )
    rio = runtime.get("rio_t3_sample_exposures")
    scannet = runtime.get("scannet_t1_sample_exposures")
    scans = runtime.get("global_scan_exposures")
    if (
        not isinstance(rio, int)
        or isinstance(rio, bool)
        or not isinstance(scannet, int)
        or isinstance(scannet, bool)
        or rio <= 0
        or scannet <= 0
        or rio + scannet != expected_samples
        or scans != 3 * rio + scannet
    ):
        raise T3AdaptationError("training scan-exposure accounting differs")
    for field in (
        "wall_clock_seconds",
        "gpu_hours",
        "peak_allocated_vram_mib",
    ):
        if not _finite_positive(runtime.get(field)):
            raise T3AdaptationError(f"training runtime field {field} is invalid")
    expected_gpu_hours = (
        float(runtime["wall_clock_seconds"]) * int(runtime["world_size"]) / 3600.0
    )
    if not math.isclose(
        float(runtime["gpu_hours"]), expected_gpu_hours, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise T3AdaptationError("training GPU-hours accounting differs")
    return copy.deepcopy(dict(runtime))


def _seed_everything(seed: int) -> None:
    import numpy as np
    import pytorch_lightning as pl

    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _materialize_t3_smoke_batch(config: object, device: torch.device):
    import hydra
    import numpy as np

    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )

    mixed = hydra.utils.instantiate(config.data.train_dataset)
    if len(mixed.datasets) != 2:
        raise T3AdaptationError("formal mixed dataset must contain RIO and ScanNet")
    rio, scannet = mixed.datasets
    excluded_rio_windows = len(rio.known_empty_scan_contexts)
    if (
        rio.dataset_name != "rio"
        or rio.temporal_window != 3
        or scannet.dataset_name != "scannet"
        or scannet.temporal_window != 1
        or len(rio) != 855
        or excluded_rio_windows != 3
        or len(rio) + excluded_rio_windows != 858
        or mixed.sampler.num_samples != 1536
    ):
        raise T3AdaptationError("formal mixed dataset identity differs")
    candidates = []
    for index, scan_indices in enumerate(rio.sequence_indices):
        point_count = sum(int(rio.data[int(item)]["file_len"]) for item in scan_indices)
        candidates.append((point_count, rio.sequence_names[index], index))
    collate = hydra.utils.instantiate(config.data.train_collation)
    last_error: Exception | None = None
    for raw_points, sample_name, index in sorted(candidates):
        try:
            _seed_everything(int(config.general.seed))
            sample = rio[index]
            raw_stages = sorted(
                int(value) for value in np.unique(sample[0][:, 3]).tolist()
            )
            if raw_stages != [0, 1, 2]:
                raise T3AdaptationError(
                    f"raw T3 sample has temporal stages {raw_stages}"
                )
            _seed_everything(int(config.general.seed))
            data, targets, names = collate([sample])
            semantics = validate_t3_batch_semantics(data, targets, names)
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            return (
                mixed,
                data,
                targets,
                list(names),
                {
                    **semantics,
                    "sample_name": sample_name,
                    "raw_point_count": int(raw_points),
                    "rio_raw_train_sequence_count": 858,
                    "rio_active_train_sequence_count": len(rio),
                    "rio_excluded_empty_supervision_windows": excluded_rio_windows,
                    "scannet_train_sample_count": len(scannet),
                    "mixed_epoch_samples": int(mixed.sampler.num_samples),
                },
            )
        except Exception as error:
            last_error = error
    raise T3AdaptationError("no valid supervised T3 smoke sample") from last_error


def _fresh_optimizer_scheduler(
    config: object, system: torch.nn.Module, total_steps: int
):
    import hydra
    from omegaconf import OmegaConf

    parameters = [
        parameter for parameter in system.parameters() if parameter.requires_grad
    ]
    optimizer = hydra.utils.instantiate(config.optimizer, params=parameters)
    if optimizer.state:
        raise T3AdaptationError("fresh adaptation optimizer unexpectedly has state")
    scheduler_config = OmegaConf.create(
        OmegaConf.to_container(config.scheduler.scheduler, resolve=True)
    )
    scheduler_config.total_steps = total_steps
    scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
    return optimizer, scheduler


def run_t3_smoke(device: torch.device) -> dict[str, object]:
    from omegaconf import open_dict

    from scripts.run_p2_native_smoke import (
        _forward_losses,
        _gradient_summary,
        _tensor_digest,
        classify_parameters,
    )
    from trainer.trainer import InstanceSegmentation

    source_commit = _git_source_commit(require_clean=True)
    recipe = protocol.load_t3_adaptation_recipe()
    _, config = protocol.compose_t3_adaptation_config(recipe)
    with open_dict(config):
        config.general.gpus = 1
        config.data.num_workers = 0
    _seed_everything(int(config.general.seed))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    system = InstanceSegmentation(config).to(device)
    source_load = strict_load_adaptation_weights(
        system,
        protocol.CHECKPOINT_PATH,
        expected_sha256=str(recipe["source"]["checkpoint_sha256"]),
    )
    mixed, data, targets, names, sample = _materialize_t3_smoke_batch(config, device)
    groups = classify_parameters(system.named_parameters())
    frozen_names = groups["frozen_encoder"]
    trainable_names = [
        name
        for group, names_in_group in groups.items()
        if group != "frozen_encoder"
        for name in names_in_group
    ]
    named = dict(system.named_parameters())
    frozen_before = _tensor_digest((name, named[name]) for name in frozen_names)
    trainable_before = _tensor_digest((name, named[name]) for name in trainable_names)
    optimizer, scheduler = _fresh_optimizer_scheduler(config, system, total_steps=2)
    system.train()
    optimizer.zero_grad(set_to_none=True)
    _, losses, breakdown = _forward_losses(system, data, targets)
    finite_losses = {
        key: float(value.detach().cpu())
        for key, value in losses.items()
        if "_contrastive_layer" not in key
    }
    if not all(math.isfinite(value) for value in finite_losses.values()):
        raise T3AdaptationError("T3 smoke produced a non-finite loss")
    breakdown["objective"].backward()
    gradients = {
        "frozen_encoder": _gradient_summary(system, frozen_names),
        "trainable_parameters": _gradient_summary(system, trainable_names),
    }
    if (
        gradients["frozen_encoder"]["nonzero_grad_tensors"] != 0
        or not gradients["trainable_parameters"]["finite"]
        or gradients["trainable_parameters"]["nonzero_grad_tensors"] == 0
    ):
        raise T3AdaptationError("T3 smoke gradient contract differs")
    optimizer.step()
    scheduler.step()
    if scheduler.last_epoch != 1:
        raise T3AdaptationError("T3 smoke fresh scheduler did not advance")
    frozen_after = _tensor_digest((name, named[name]) for name in frozen_names)
    trainable_after = _tensor_digest((name, named[name]) for name in trainable_names)
    if frozen_after != frozen_before or trainable_after == trainable_before:
        raise T3AdaptationError("T3 smoke parameter update contract differs")

    with tempfile.TemporaryDirectory(prefix="reviewer-t3-smoke-") as directory:
        checkpoint_path = Path(directory) / "weights.ckpt"
        torch.save({"state_dict": system.state_dict()}, checkpoint_path)
        saved_digest = _tensor_digest(system.named_parameters())
        with torch.no_grad():
            named[trainable_names[0]].add_(1.0)
        roundtrip = strict_load_adaptation_weights(system, checkpoint_path)
        if _tensor_digest(system.named_parameters()) != saved_digest:
            raise T3AdaptationError("T3 smoke checkpoint reload differs")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    artifact: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "scope": "pretraining-smoke-only",
        "source_commit": source_commit,
        "paper_name": recipe["paper_name"],
        "recipe_sha256": protocol.sha256_file(protocol.RECIPE_PATH),
        "source_checkpoint": source_load,
        "sample": sample,
        "losses": finite_losses,
        "weighted_objective": float(breakdown["objective"].detach().cpu()),
        "gradients": gradients,
        "fresh_optimizer_state_before_step": True,
        "fresh_scheduler_last_epoch_after_step": int(scheduler.last_epoch),
        "checkpoint_reload": {
            "strict": True,
            "state_dict_entry_count": roundtrip["state_dict_entry_count"],
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "formal_training_started": False,
        "recipe_changed": False,
        "mixed_dataset_sizes": [len(dataset) for dataset in mixed.datasets],
    }
    artifact["content_sha256"] = protocol._content_sha256(artifact)
    protocol._publish_exact(
        SMOKE_ARTIFACT,
        (json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    return artifact


class TrainingEvidenceCallback(Callback):
    """Lightning callback that records actual global training exposures."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.started = 0.0
        self.local_samples = 0
        self.local_scans = 0
        self.local_rio = 0
        self.local_scannet = 0
        self.local_optimizer_updates = 0
        self.initial_optimizer_state_count: int | None = None
        self.initial_lr: float | None = None

    @property
    def state_key(self) -> str:
        return "reviewer_closure_training_evidence_v1"

    def on_fit_start(self, trainer: object, pl_module: object) -> None:
        self.started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(pl_module.device)
        optimizer = trainer.optimizers[0]
        self.initial_optimizer_state_count = len(optimizer.state)
        self.initial_lr = float(optimizer.param_groups[0]["lr"])
        if self.initial_optimizer_state_count != 0:
            raise T3AdaptationError("adaptation optimizer did not start fresh")

    def on_train_batch_start(
        self,
        trainer: object,
        pl_module: object,
        batch: object,
        batch_idx: int,
    ) -> None:
        data, targets, names = batch
        stage_batches = getattr(data, "temporal_stages", None)
        if not isinstance(stage_batches, (list, tuple)) or len(stage_batches) != len(
            names
        ):
            raise T3AdaptationError("formal batch temporal-stage accounting differs")
        for stages in stage_batches:
            count = int(torch.unique(stages).numel())
            if count == 3:
                self.local_rio += 1
            elif count == 1:
                self.local_scannet += 1
            else:
                raise T3AdaptationError(f"formal batch contains {count} stages")
            self.local_samples += 1
            self.local_scans += count

    def on_before_optimizer_step(
        self, trainer: object, pl_module: object, optimizer: object
    ) -> None:
        self.local_optimizer_updates += 1

    def on_train_end(self, trainer: object, pl_module: object) -> None:
        device = pl_module.device
        sums = torch.tensor(
            [
                self.local_samples,
                self.local_scans,
                self.local_rio,
                self.local_scannet,
            ],
            dtype=torch.int64,
            device=device,
        )
        updates = torch.tensor(
            self.local_optimizer_updates, dtype=torch.int64, device=device
        )
        peak = torch.tensor(
            float(torch.cuda.max_memory_allocated(device) / 1024**2),
            dtype=torch.float64,
            device=device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(updates, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        wall = time.perf_counter() - self.started
        if trainer.is_global_zero:
            runtime = {
                "status": "completed",
                "world_size": int(trainer.world_size),
                "completed_epochs": int(trainer.current_epoch),
                "optimizer_updates": int(updates.item()),
                "global_sample_exposures": int(sums[0].item()),
                "global_scan_exposures": int(sums[1].item()),
                "rio_t3_sample_exposures": int(sums[2].item()),
                "scannet_t1_sample_exposures": int(sums[3].item()),
                "wall_clock_seconds": wall,
                "gpu_hours": wall * int(trainer.world_size) / 3600.0,
                "peak_allocated_vram_mib": float(peak.item()),
                "initial_optimizer_state_count": self.initial_optimizer_state_count,
                "initial_lr": self.initial_lr,
                "final_global_step": int(trainer.global_step),
            }
            _atomic_runtime_json(self.output_path, runtime)


def _checkpoint_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise T3AdaptationError(f"required training checkpoint is absent: {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    mapping = _as_mapping(checkpoint, name="training checkpoint")
    state_dict = mapping.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise T3AdaptationError("training checkpoint has no state_dict")
    try:
        reference = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise T3AdaptationError("training checkpoint is outside repository") from error
    return {
        "reference": f"repo:{reference}",
        "sha256": protocol.sha256_file(path),
        "byte_size": path.stat().st_size,
        "epoch": mapping.get("epoch"),
        "global_step": mapping.get("global_step"),
        "state_dict_entry_count": len(state_dict),
        "optimizer_state_count": len(mapping.get("optimizer_states", [])),
        "scheduler_state_count": len(mapping.get("lr_schedulers", [])),
    }


def run_formal_training() -> dict[str, object]:
    import hydra
    import pytorch_lightning as pl

    from trainer.trainer import InstanceSegmentation

    source_commit = _git_source_commit(require_clean=True)
    if not SMOKE_ARTIFACT.is_file():
        raise T3AdaptationError("passing T3 smoke artifact is required")
    smoke = json.loads(SMOKE_ARTIFACT.read_text(encoding="utf-8"))
    if smoke.get("status") != "pass" or not isinstance(smoke.get("source_commit"), str):
        raise T3AdaptationError("T3 smoke does not bind the current source commit")
    _require_smoke_source_binding(smoke["source_commit"], source_commit)
    if TRAINING_MANIFEST.exists() or TRAINING_MANIFEST.is_symlink():
        raise T3AdaptationError("formal T3 training manifest already exists")

    recipe = protocol.load_t3_adaptation_recipe()
    _, config = protocol.compose_t3_adaptation_config(recipe)
    save_dir = PROJECT_ROOT / str(config.general.save_dir)
    existing = list(save_dir.glob("*.ckpt")) if save_dir.is_dir() else []
    if existing:
        raise T3AdaptationError("formal T3 checkpoint directory is not empty")
    _seed_everything(int(config.general.seed))
    system = InstanceSegmentation(config)
    source_load = strict_load_adaptation_weights(
        system,
        protocol.CHECKPOINT_PATH,
        expected_sha256=str(recipe["source"]["checkpoint_sha256"]),
    )
    runtime_path = save_dir / RUNTIME_RECORD_NAME
    evidence = TrainingEvidenceCallback(runtime_path)
    callbacks = [hydra.utils.instantiate(spec) for spec in config.callbacks]
    callbacks.append(evidence)
    loggers = [hydra.utils.instantiate(spec) for spec in config.logging]
    trainer = pl.Trainer(
        logger=loggers,
        accelerator="gpu",
        devices=int(config.general.gpus),
        callbacks=callbacks,
        default_root_dir=str(save_dir),
        **config.trainer,
    )
    trainer.fit(system, ckpt_path=None)
    if not trainer.is_global_zero:
        return {"status": "worker-completed"}
    if not runtime_path.is_file():
        raise T3AdaptationError("formal training runtime record is absent")
    runtime = validate_completed_training_runtime(
        json.loads(runtime_path.read_text(encoding="utf-8")), recipe
    )
    best_path = Path(callbacks[0].best_model_path)
    last_path = Path(callbacks[0].last_model_path)
    canonical_path = (
        PROJECT_ROOT / "checkpoints/rescene4d_t2_to_t3_horizon_adapted.ckpt"
    )
    records = {
        "best_validation": _checkpoint_record(best_path),
        "last": _checkpoint_record(last_path),
        "canonical_final": _checkpoint_record(canonical_path),
    }
    final_record = records["canonical_final"]
    if final_record["epoch"] != 44 or final_record["global_step"] != 2160:
        raise T3AdaptationError("canonical final checkpoint budget differs")
    system.cpu()
    reload_result = strict_load_adaptation_weights(system, canonical_path)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "paper_name": recipe["paper_name"],
        "selected_level": recipe["comparison_level"],
        "source_commit": source_commit,
        "recipe_sha256": protocol.sha256_file(protocol.RECIPE_PATH),
        "training_audit_content_sha256": json.loads(
            (ARTIFACT_ROOT / "t3_training_recipe_audit.json").read_text(
                encoding="utf-8"
            )
        )["content_sha256"],
        "initialization": {
            **source_load,
            "mode": "weights_only_strict",
            "fresh_optimizer_and_scheduler": True,
        },
        "runtime": runtime,
        "checkpoints": records,
        "canonical_checkpoint_reload": {
            "strict": True,
            "state_dict_entry_count": reload_result["state_dict_entry_count"],
        },
        "recipe_changed_after_smoke": False,
    }
    manifest["content_sha256"] = protocol._content_sha256(manifest)
    protocol._publish_exact(
        TRAINING_MANIFEST,
        (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", default="cuda:0")
    subparsers.add_parser("train")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = run_t3_smoke(torch.device(args.device))
    else:
        result = run_formal_training()
    print(
        json.dumps(
            {
                "status": result["status"],
                "content_sha256": result.get("content_sha256"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
