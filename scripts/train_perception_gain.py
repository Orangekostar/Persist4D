"""Train one frozen-budget Perception Gain V1 adaptation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.semseg import SemanticSegmentationDataset
from scripts.perception_gain_data import sample_reference_segments
from scripts.preflight_task_memory_episode import (
    _resolved_data_path,
    _rio_base_dataset,
    load_reference_by_scene,
)
from trainer.perception_gain_trainer import (
    PerceptionGainTrainer,
    PerceptionProgress,
    PerceptionRuntimeContract,
    PerceptionTrainingError,
    SemanticScorerTrainer,
    load_frozen_semantic_scorer,
    strict_load_r1_perception,
)
from trainer.task_memory_trainer import (
    capture_task_memory_rng_state,
    restore_task_memory_rng_state,
)

R1_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
R1_BYTES = 754_813_672
VARIANTS = ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")
SCORER_UPDATES = 500
SCORER_BATCH_SEGMENTS = 1024
DEFAULT_ROLES = PROJECT_ROOT / "artifacts/perception_gain_v1/DATA_ROLES.json"
DEFAULT_ASSETS = Path("/home/ww/persist4d_runs/perception_gain_v1/assets.local.json")
_SCENE_ID = re.compile(r"^scene(?P<scene>[0-9]{4})_[0-9]{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PerceptionTrainingError(f"cannot decode JSON input: {path}") from error
    if not isinstance(value, Mapping):
        raise PerceptionTrainingError(f"JSON input must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def compose_variant_config(
    variant: str,
    *,
    pretrained: Path,
    run_dir: Path,
    recipe_config: Mapping[str, object] | None = None,
) -> DictConfig:
    if variant not in VARIANTS:
        raise PerceptionTrainingError(f"unsupported perception variant: {variant}")
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "conf").resolve()), version_base="1.2"
    ):
        config = compose(config_name=f"perception_gain_v1/{variant}")
    with open_dict(config):
        config.backbone.name = str(pretrained)
        config.general.save_dir = str(run_dir)
    if recipe_config is not None:
        from scripts.perception_gain_v2_config import apply_recipe

        apply_recipe(config, recipe_config)
    return config


def resolve_training_run_dir(
    external_root: Path, *, variant: str, run_subdir: Path | None
) -> Path:
    relative = Path(variant) if run_subdir is None else Path(run_subdir)
    training_root = Path(external_root) / "training"
    if (
        variant not in VARIANTS
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise PerceptionTrainingError("training run subdirectory is invalid")
    destination = training_root / relative
    if not destination.resolve().is_relative_to(training_root.resolve()):
        raise PerceptionTrainingError("training run subdirectory escapes its root")
    return destination


def build_weighted_sample_plan(
    *,
    dataset_sizes: Sequence[int],
    nominal_weights: Sequence[float],
    total_draws: int,
    seed: int,
) -> torch.Tensor:
    """Reproduce weighted-with-replacement sampling with an explicit generator."""

    if (
        isinstance(dataset_sizes, (str, bytes))
        or isinstance(nominal_weights, (str, bytes))
        or len(dataset_sizes) != len(nominal_weights)
        or not dataset_sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in dataset_sizes
        )
        or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math_is_finite_positive(float(weight))
            for weight in nominal_weights
        )
        or isinstance(total_draws, bool)
        or not isinstance(total_draws, int)
        or total_draws <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise PerceptionTrainingError("weighted sample-plan arguments are invalid")
    per_sample = torch.cat(
        [
            torch.full((size,), float(weight) / size, dtype=torch.double)
            for size, weight in zip(dataset_sizes, nominal_weights, strict=True)
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.multinomial(
        per_sample,
        total_draws,
        replacement=True,
        generator=generator,
    )


def math_is_finite_positive(value: float) -> bool:
    return np.isfinite(value) and value > 0.0


def sample_plan_sha256(plan: torch.Tensor) -> str:
    value = plan.detach().cpu().contiguous().long()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@contextmanager
def _sample_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256(
        "\0".join(str(value) for value in values).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _scene_number(sequence_name: str) -> int:
    scenes = set()
    for scan_id in sequence_name.split("-"):
        match = _SCENE_ID.fullmatch(scan_id)
        if match is None:
            raise PerceptionTrainingError(f"invalid scan identity: {scan_id}")
        scenes.add(int(match.group("scene")))
    if len(scenes) != 1:
        raise PerceptionTrainingError("one training pair spans multiple scenes")
    return next(iter(scenes))


def filter_rio_train_indices(
    dataset: SemanticSegmentationDataset,
    *,
    reference_by_scene: Mapping[int, str],
    train_references: set[str],
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    indices = []
    references = []
    for index, sequence_name in enumerate(dataset.sequence_names):
        scene = _scene_number(sequence_name)
        reference = reference_by_scene.get(scene)
        if not isinstance(reference, str):
            raise PerceptionTrainingError(
                f"RIO scene {scene} lacks a frozen reference assignment"
            )
        if reference in train_references:
            indices.append(index)
            references.append(reference)
    if not indices:
        raise PerceptionTrainingError("TRAIN filtering removed every RIO pair")
    return tuple(indices), tuple(references)


def _scannet_base_dataset(config: DictConfig, *, data_root: Path):
    training = OmegaConf.to_container(config.data.train_dataset, resolve=True)
    if not isinstance(training, dict) or not isinstance(training.get("datasets"), list):
        raise PerceptionTrainingError("training config lacks the frozen ScanNet source")
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
        if isinstance(item, dict) and item.get("dataset_name") == "scannet"
    ]
    if len(candidates) != 1:
        raise PerceptionTrainingError(
            "training config does not bind one ScanNet source"
        )
    parameters = {**common, **candidates[0]}
    parameters.pop("target", None)
    parameters.pop("_target_", None)
    for key in (
        "data_dir",
        "label_db_filepath",
        "change_label_db_filepath",
        "color_mean_std",
    ):
        if key in parameters:
            parameters[key] = _resolved_data_path(parameters[key], data_root)
    return SemanticSegmentationDataset(**parameters)


class PerceptionDrawDataset(Dataset):
    """Materialize one frozen weighted stream with per-draw augmentation RNG."""

    def __init__(
        self,
        *,
        rio: Dataset,
        scannet: Dataset,
        rio_indices: Sequence[int],
        rio_references: Sequence[str],
        plan: torch.Tensor,
        start_draw_index: int,
        seed: int,
    ) -> None:
        if (
            len(rio_indices) != len(rio_references)
            or not rio_indices
            or plan.ndim != 1
            or not 0 <= start_draw_index <= plan.numel()
        ):
            raise PerceptionTrainingError("perception draw dataset is invalid")
        expected_size = len(rio_indices) + len(scannet)
        if plan.numel() and (
            plan.min().item() < 0 or plan.max().item() >= expected_size
        ):
            raise PerceptionTrainingError("sample plan index is outside its datasets")
        self.rio = rio
        self.scannet = scannet
        self.rio_indices = tuple(int(value) for value in rio_indices)
        self.rio_references = tuple(rio_references)
        self.plan = plan.detach().cpu().long()
        self.start_draw_index = int(start_draw_index)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.plan.numel() - self.start_draw_index

    def __getitem__(self, local_index: int):
        draw_index = self.start_draw_index + int(local_index)
        source_index = int(self.plan[draw_index].item())
        if source_index < len(self.rio_indices):
            dataset = self.rio
            sample_index = self.rio_indices[source_index]
            reference = self.rio_references[source_index]
            pair_id = str(self.rio.sequence_names[sample_index])
        else:
            dataset = self.scannet
            sample_index = source_index - len(self.rio_indices)
            pair_id = str(self.scannet.sequence_names[sample_index])
            reference = f"scannet:{pair_id.split('-', 1)[0]}"
        augmentation_seed = _stable_seed(
            "pgv1-augmentation",
            self.seed,
            draw_index,
            reference,
            pair_id,
        )
        with _sample_rng(augmentation_seed):
            sample = dataset[sample_index]
        return PerceptionDrawSample(draw_index=draw_index, sample=sample)


@dataclass(frozen=True)
class PerceptionDrawSample:
    draw_index: int
    sample: tuple[Any, ...]


class DeterministicPerceptionCollator:
    def __init__(self, collator: Any, *, seed: int) -> None:
        if not callable(collator):
            raise PerceptionTrainingError("perception collator must be callable")
        self.collator = collator
        self.seed = int(seed)

    def __call__(self, records: list[PerceptionDrawSample]):
        if not records or any(
            not isinstance(record, PerceptionDrawSample) for record in records
        ):
            raise PerceptionTrainingError("perception batch lacks draw identities")
        draw_indices = tuple(record.draw_index for record in records)
        collate_seed = _stable_seed(
            "pgv1-collate",
            self.seed,
            *draw_indices,
        )
        with _sample_rng(collate_seed):
            return self.collator([record.sample for record in records])


class PerceptionCheckpointCallback(Callback):
    def __init__(
        self,
        run_dir: Path,
        evaluation_updates: Sequence[int],
        *,
        report_progress: bool = False,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.evaluation_updates = {
            int(value) for value in evaluation_updates if value
        }
        self.last_saved_step = -1
        self.report_progress = report_progress
        self.last_batch_time = None
        self.maximum_batch_interval_seconds = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        del pl_module
        # Restored global_step persists through the first accumulation batches.
        # Those batches must not overwrite an already completed checkpoint.
        self.last_saved_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch
        if self.report_progress and trainer.is_global_zero:
            now = time.time()
            if self.last_batch_time is not None:
                self.maximum_batch_interval_seconds = max(
                    self.maximum_batch_interval_seconds,
                    now - self.last_batch_time,
                )
            self.last_batch_time = now
            _atomic_json(
                self.run_dir / "PROGRESS.json",
                {
                    "completed_optimizer_updates": int(trainer.global_step),
                    "completed_batch_index": batch_idx,
                    "updated_unix": now,
                    "maximum_batch_interval_seconds": self.maximum_batch_interval_seconds,
                },
            )
        step = int(trainer.global_step)
        if step <= 0 or step == self.last_saved_step:
            return
        if step in self.evaluation_updates:
            trainer.save_checkpoint(self.run_dir / f"update={step:04d}.ckpt")
        if step % 50 == 0 or step in self.evaluation_updates:
            trainer.save_checkpoint(self.run_dir / "last.ckpt")
            self.last_saved_step = step

    def on_train_end(self, trainer, pl_module) -> None:
        del pl_module
        trainer.save_checkpoint(self.run_dir / "last.ckpt")


def _resume_progress(path: Path | None) -> PerceptionProgress:
    if path is None:
        return PerceptionProgress.initial()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload = (
        checkpoint.get("perception_resume") if isinstance(checkpoint, Mapping) else None
    )
    progress = payload.get("progress") if isinstance(payload, Mapping) else None
    if not isinstance(progress, Mapping):
        raise PerceptionTrainingError("resume checkpoint lacks exact sample progress")
    return PerceptionProgress.from_state_dict(progress)


def _load_r1(system: PerceptionGainTrainer, checkpoint: Path, variant: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise PerceptionTrainingError("R1 checkpoint lacks a state_dict")
    return strict_load_r1_perception(
        system,
        state,
        allow_semantic_scorer=variant == "Q-SEM",
    )


def _runtime_contract(config: DictConfig) -> PerceptionRuntimeContract:
    settings = config.perception_training
    return PerceptionRuntimeContract(
        devices=int(settings.devices),
        batch_size_per_gpu=int(settings.batch_size_per_gpu),
        gradient_accumulation=int(settings.gradient_accumulation),
        precision=str(settings.precision),
        gradient_clip_norm=float(settings.gradient_clip_norm),
        total_updates=int(settings.optimizer_updates),
        evaluation_updates=tuple(int(value) for value in settings.evaluation_updates),
    )


def _load_scorer_segments(
    cache_paths: Sequence[Path],
) -> dict[str, tuple[tuple[Tensor, Tensor], ...]]:
    by_reference: dict[str, list[tuple[Tensor, Tensor]]] = {}
    for path in cache_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise PerceptionTrainingError(f"invalid scorer shard: {path}")
        for record in payload:
            if not isinstance(record, Mapping):
                raise PerceptionTrainingError("scorer record must be a mapping")
            reference = record.get("reference_id")
            features = record.get("segment_features")
            targets = record.get("targets")
            valid = record.get("valid")
            if (
                not isinstance(reference, str)
                or not reference
                or not isinstance(features, Tensor)
                or features.ndim != 2
                or features.shape[1] != 128
                or not isinstance(targets, Tensor)
                or targets.shape != (features.shape[0],)
                or not isinstance(valid, Tensor)
                or valid.dtype != torch.bool
                or valid.shape != targets.shape
            ):
                raise PerceptionTrainingError("scorer record fields are invalid")
            for index in valid.nonzero(as_tuple=True)[0].tolist():
                by_reference.setdefault(reference, []).append(
                    (
                        features[index].detach().float().cpu(),
                        targets[index].detach().float().cpu(),
                    )
                )
    result = {
        reference: tuple(values) for reference, values in by_reference.items() if values
    }
    if len(result) < 8:
        raise PerceptionTrainingError(
            "scorer cache has fewer than eight train references"
        )
    return result


def train_semantic_scorer(
    *,
    cache_paths: Sequence[Path],
    output_dir: Path,
    summary_output: Path | None = None,
    stop_after_updates: int = SCORER_UPDATES,
    resume: Path | None = None,
    device: str = "cuda",
) -> dict[str, object]:
    if not 1 <= stop_after_updates <= SCORER_UPDATES:
        raise PerceptionTrainingError(
            "scorer endpoint is outside the 500-update budget"
        )
    segments = _load_scorer_segments(cache_paths)
    torch.manual_seed(45)
    np.random.seed(45)
    random.seed(45)
    system = SemanticScorerTrainer().to(device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    completed = 0
    losses = []
    training_audit = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        required = {
            "completed_updates",
            "losses",
            "optimizer_state_dict",
            "rng",
            "scorer_state_dict",
            "schema_version",
            "training_audit",
        }
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != required:
            raise PerceptionTrainingError("scorer resume checkpoint is invalid")
        if checkpoint["schema_version"] != "perception-scorer-resume-v1":
            raise PerceptionTrainingError("scorer resume schema differs")
        system.scorer.load_state_dict(checkpoint["scorer_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed = int(checkpoint["completed_updates"])
        losses = list(checkpoint["losses"])
        training_audit = list(checkpoint["training_audit"])
        rng = checkpoint["rng"]
        if not isinstance(rng, Mapping):
            raise PerceptionTrainingError("scorer RNG checkpoint is invalid")
        restore_task_memory_rng_state(rng)
    if completed > stop_after_updates:
        raise PerceptionTrainingError("scorer resume is beyond requested endpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for update in range(completed, stop_after_updates):
        sampled = sample_reference_segments(
            segments,
            count=SCORER_BATCH_SEGMENTS,
            seed=45,
            cursor=update * SCORER_BATCH_SEGMENTS,
        )
        features = torch.stack([value[0] for _, value in sampled]).to(device)
        targets = torch.stack([value[1] for _, value in sampled]).to(device)
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in system.named_parameters()
        }
        optimizer.zero_grad(set_to_none=True)
        loss = system(features, targets)
        loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().norm().item()) ** 2
                for parameter in system.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if len(training_audit) < 2:
            change_norm = math.sqrt(
                sum(
                    float((parameter.detach().cpu() - before[name]).float().norm()) ** 2
                    for name, parameter in system.named_parameters()
                )
            )
            training_audit.append(
                {
                    "gradient_norm": gradient_norm,
                    "optimizer_update": update + 1,
                    "parameter_change_norm": change_norm,
                }
            )
        if (update + 1) % 50 == 0 or update + 1 == stop_after_updates:
            _atomic_torch_save(
                output_dir / "last.ckpt",
                {
                    "completed_updates": update + 1,
                    "losses": losses,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng": capture_task_memory_rng_state(),
                    "schema_version": "perception-scorer-resume-v1",
                    "scorer_state_dict": system.scorer.state_dict(),
                    "training_audit": training_audit,
                },
            )
    checkpoint_path = output_dir / f"update={stop_after_updates:04d}.ckpt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "scorer_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in system.scorer.state_dict().items()
            },
            "schema_version": "perception-scorer-frozen-v1",
            "source_shards": [str(path) for path in cache_paths],
            "updates": stop_after_updates,
        },
    )
    summary = {
        "batch_segments": SCORER_BATCH_SEGMENTS,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "completed_updates": stop_after_updates,
        "elapsed_seconds": time.time() - started,
        "final_loss": losses[-1],
        "reference_count": len(segments),
        "status": "COMPLETE" if stop_after_updates == SCORER_UPDATES else "SMOKE",
        "training_audit": training_audit,
    }
    summary["gpu_hours"] = float(summary["elapsed_seconds"]) / 3600.0
    _atomic_json(output_dir / "run_summary.json", summary)
    if summary_output is not None:
        _atomic_json(summary_output, summary)
    return summary


def loader_timeout(*, num_workers: int, timeout_seconds: float = 0) -> float:
    if num_workers < 0 or timeout_seconds < 0:
        raise PerceptionTrainingError("loader worker count/timeout cannot be negative")
    return float(timeout_seconds) if num_workers else 0.0


def _build_loader(
    config: DictConfig,
    *,
    data_root: Path,
    metadata: Path,
    roles_path: Path,
    next_global_draw_index: int,
) -> tuple[DataLoader, dict[str, object]]:
    contract = _runtime_contract(config)
    roles_payload = _load_json(roles_path)
    roles = roles_payload.get("roles")
    train_references = roles.get("TRAIN") if isinstance(roles, Mapping) else None
    if (
        isinstance(train_references, (str, bytes))
        or not isinstance(train_references, Sequence)
        or not train_references
    ):
        raise PerceptionTrainingError("DATA_ROLES lacks TRAIN references")
    rio = _rio_base_dataset(config, data_root=data_root, horizon=2)
    scannet = _scannet_base_dataset(config, data_root=data_root)
    rio_indices, rio_references = filter_rio_train_indices(
        rio,
        reference_by_scene=load_reference_by_scene(metadata),
        train_references=set(train_references),
    )
    plan = build_weighted_sample_plan(
        dataset_sizes=(len(rio_indices), len(scannet)),
        nominal_weights=tuple(
            float(value) for value in config.perception_training.nominal_dataset_weights
        ),
        total_draws=contract.total_global_draws,
        seed=int(config.general.seed),
    )
    if next_global_draw_index % contract.effective_batch_size:
        raise PerceptionTrainingError("resume cursor is not an optimizer boundary")
    dataset = PerceptionDrawDataset(
        rio=rio,
        scannet=scannet,
        rio_indices=rio_indices,
        rio_references=rio_references,
        plan=plan,
        start_draw_index=next_global_draw_index,
        seed=int(config.general.seed),
    )
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sampler = (
        None
        if contract.devices == 1
        else DistributedSampler(
            dataset,
            num_replicas=contract.devices,
            rank=rank,
            shuffle=False,
            drop_last=True,
        )
    )
    collator = DeterministicPerceptionCollator(
        hydra.utils.instantiate(config.data.train_collation),
        seed=int(config.general.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=contract.batch_size_per_gpu,
        shuffle=False,
        sampler=sampler,
        num_workers=int(config.data.num_workers),
        pin_memory=bool(config.data.pin_memory),
        collate_fn=collator,
        persistent_workers=int(config.data.num_workers) > 0,
        timeout=loader_timeout(
            num_workers=int(config.data.num_workers),
            timeout_seconds=float(config.perception_training.get("timeout_seconds", 0)),
        ),
        drop_last=True,
    )
    source_counts = {
        "rio": int((plan < len(rio_indices)).sum().item()),
        "scannet": int((plan >= len(rio_indices)).sum().item()),
    }
    return loader, {
        "next_global_draw_index": next_global_draw_index,
        "plan_sha256": sample_plan_sha256(plan),
        "rio_pair_count": len(rio_indices),
        "scannet_scan_count": len(scannet),
        "source_counts": source_counts,
        "total_global_draws": contract.total_global_draws,
        "train_reference_count": len(set(rio_references)),
    }


def run(args: argparse.Namespace) -> int:
    assets = _load_json(args.assets)
    r1_checkpoint = Path(assets["r1_checkpoint"])
    pretrained = Path(assets["concerto_pretrained"])
    data_root = Path(assets["data_root"])
    metadata = Path(assets["rio_metadata"])
    if args.resume is None and (
        r1_checkpoint.stat().st_size != R1_BYTES or _sha256(r1_checkpoint) != R1_SHA256
    ):
        raise PerceptionTrainingError("R1 checkpoint identity differs")
    run_dir = resolve_training_run_dir(
        args.external_root,
        variant=args.variant,
        run_subdir=args.run_subdir,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = getattr(args, "recipe_config", None)
    recipe = _load_json(recipe_path) if recipe_path is not None else None
    code = None
    if recipe is not None:
        from scripts.perception_gain_v2_config import live_execution_provenance

        code = live_execution_provenance(recipe)
    config = compose_variant_config(
        args.variant,
        pretrained=pretrained,
        run_dir=run_dir,
        recipe_config=recipe,
    )
    with open_dict(config):
        if recipe is None:
            config.general.seed = int(args.train_seed)
        legacy_path = getattr(args, "legacy_resume_config", None)
        if legacy_path is not None:
            config.perception_legacy_resume_config = OmegaConf.to_container(
                OmegaConf.load(legacy_path),
                resolve=True,
            )
    if recipe is not None:
        seed_everything(int(config.general.seed), workers=True)
    contract = _runtime_contract(config)
    if not 1 <= args.stop_after_updates <= contract.total_updates:
        raise PerceptionTrainingError("stop-after-updates is outside the V1 budget")
    progress = _resume_progress(args.resume)
    if recipe is not None and args.resume is not None:
        from scripts.perception_gain_v2_config import validate_resume_recipe

        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_recipe(
            saved,
            recipe,
            legacy_config=OmegaConf.to_container(config, resolve=True).get(
                "perception_legacy_resume_config"
            ),
        )
        del saved
    if progress.completed_optimizer_updates > args.stop_after_updates:
        raise PerceptionTrainingError("resume checkpoint is beyond requested endpoint")
    loader, plan_summary = _build_loader(
        config,
        data_root=data_root,
        metadata=metadata,
        roles_path=args.roles,
        next_global_draw_index=progress.next_global_draw_index,
    )
    system = PerceptionGainTrainer(config)
    load_audit = None
    scorer_audit = None
    if args.resume is None:
        load_audit = _load_r1(system, r1_checkpoint, args.variant)
        if args.variant == "Q-SEM":
            scorer_checkpoint = config.perception_training.scorer_checkpoint
            if not isinstance(scorer_checkpoint, str) or not scorer_checkpoint:
                raise PerceptionTrainingError(
                    "Q-SEM requires the fixed scorer checkpoint"
                )
            scorer_audit = load_frozen_semantic_scorer(system, scorer_checkpoint)
    seed_everything(int(config.general.seed), workers=True)
    if "LOCAL_RANK" not in os.environ:
        OmegaConf.save(config, run_dir / "resolved_config.yaml", resolve=True)
        _atomic_json(
            run_dir / "run_plan.json",
            {
                "contract": {
                    "batch_size_per_gpu": contract.batch_size_per_gpu,
                    "devices": contract.devices,
                    "effective_batch_size": contract.effective_batch_size,
                    "evaluation_updates": list(contract.evaluation_updates),
                    "gradient_accumulation": contract.gradient_accumulation,
                    "gradient_clip_norm": contract.gradient_clip_norm,
                    "precision": contract.precision,
                    "total_updates": contract.total_updates,
                },
                "load_audit": load_audit,
                "plan": plan_summary,
                "r1_checkpoint_sha256": R1_SHA256,
                "resume": args.resume is not None,
                "scorer_audit": scorer_audit,
                "seed": int(config.general.seed),
                "stop_after_updates": args.stop_after_updates,
                "variant": args.variant,
                **({"recipe": dict(recipe)} if recipe is not None else {}),
                **({"execution_provenance": code} if code is not None else {}),
            },
        )
    callback = PerceptionCheckpointCallback(
        run_dir,
        contract.checkpoint_updates,
        report_progress=recipe is not None,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=contract.devices,
        strategy=str(config.trainer.strategy),
        accumulate_grad_batches=contract.gradient_accumulation,
        precision=contract.precision,
        gradient_clip_val=contract.gradient_clip_norm,
        max_epochs=-1,
        max_steps=args.stop_after_updates,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[callback, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=run_dir, name="training_metrics"),
        default_root_dir=run_dir,
        enable_checkpointing=False,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    started = time.time()
    trainer.fit(
        system,
        train_dataloaders=loader,
        ckpt_path=str(args.resume) if args.resume is not None else None,
        weights_only=False,
    )
    elapsed = time.time() - started
    if not trainer.is_global_zero:
        return 0
    checkpoints = sorted(run_dir.glob("*.ckpt"))
    _atomic_json(
        run_dir / "run_summary.json",
        {
            "checkpoints": [
                {
                    "bytes": path.stat().st_size,
                    "name": path.name,
                    "sha256": _sha256(path),
                }
                for path in checkpoints
            ],
            "completed_global_step": int(trainer.global_step),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * contract.devices / 3600.0,
            "optimizer_audit": system.optimizer_audit,
            "progress": system.progress.state_dict(),
            "status": (
                "COMPLETE"
                if int(trainer.global_step) == args.stop_after_updates
                else "PARTIAL"
            ),
            "training_audit": system.training_audit,
            "seed": int(config.general.seed),
            "variant": args.variant,
            **({"recipe": dict(recipe)} if recipe is not None else {}),
            **({"execution_provenance": code} if code is not None else {}),
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("perception", "scorer"), default="perception"
    )
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("/home/ww/persist4d_runs/perception_gain_v1"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-seed", type=int, choices=(45, 46), default=45)
    parser.add_argument("--run-subdir", type=Path)
    parser.add_argument("--recipe-config", type=Path)
    parser.add_argument("--legacy-resume-config", type=Path)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--scorer-cache", type=Path, nargs="*")
    parser.add_argument("--scorer-output", type=Path)
    parser.add_argument("--scorer-summary", type=Path)
    parser.add_argument("--scorer-device", default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "scorer":
        if not args.scorer_cache or args.scorer_output is None:
            raise PerceptionTrainingError(
                "scorer mode requires --scorer-cache and --scorer-output"
            )
        train_semantic_scorer(
            cache_paths=args.scorer_cache,
            output_dir=args.scorer_output,
            summary_output=args.scorer_summary,
            stop_after_updates=args.stop_after_updates or SCORER_UPDATES,
            resume=args.resume,
            device=args.scorer_device,
        )
        return 0
    if args.variant is None:
        raise PerceptionTrainingError("perception mode requires --variant")
    if args.stop_after_updates is None:
        args.stop_after_updates = 3000
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
