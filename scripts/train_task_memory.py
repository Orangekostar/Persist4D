#!/usr/bin/env python3
"""Train one frozen-budget TaskMemory Retention V2 M2 variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fcntl

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datasets.task_memory_episode import (
    NativeEpisodeMaster,
    TaskMemoryEpisode,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
)
from models.persist4d_task_memory import strict_load_r1_task_memory
from scripts.preflight_task_memory_episode import (
    _rio_base_dataset,
    _role_by_reference,
    load_reference_by_scene,
)
from trainer.task_memory_trainer import TaskMemoryProgress, TaskMemoryTrainer

R1_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
R1_BYTES = 754_813_672
FORMAL_OPTIMIZER_UPDATES = 3000
PILOT_OPTIMIZER_UPDATES = 300
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_DATA_CONTRACT = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
)
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2/training"
LOCAL_ASSET_RESOLVER = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/external_assets.local.json"
)
VARIANTS = ("W-BASE", "Q-INDEP", "Q-TALA", "FH-MATCH")
SMOKE_REFERENCE_ID = "09582244-e2c2-2de1-956c-357092d949d1"
SMOKE_SEQUENCE_ID = (
    "scene0007_00-scene0007_01-scene0007_03-scene0007_02-scene0007_04"
)


class TaskMemoryTrainingError(RuntimeError):
    """Raised when a formal training input or output contract differs."""


def compose_variant_config(
    variant: str,
    *,
    pretrained: Path,
    run_dir: Path,
) -> DictConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported TaskMemory variant: {variant}")
    with initialize_config_dir(
        config_dir=str((PROJECT_ROOT / "conf").resolve()), version_base="1.2"
    ):
        config = compose(config_name=f"task_memory_v2/{variant}")
    with open_dict(config):
        config.backbone.name = str(pretrained)
        config.general.save_dir = str(run_dir)
    return config


def _flatten_config(value: object, *, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        flattened = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(value[key], prefix=path))
        return flattened
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def resolved_variant_diff(
    left: DictConfig,
    right: DictConfig,
    *,
    ignore_paths: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    left_value = OmegaConf.to_container(left, resolve=True)
    right_value = OmegaConf.to_container(right, resolve=True)
    left_flat = _flatten_config(left_value)
    right_flat = _flatten_config(right_value)
    ignored = ignore_paths or set()
    missing = "<MISSING>"
    return {
        path: {
            "left": left_flat.get(path, missing),
            "right": right_flat.get(path, missing),
        }
        for path in sorted(set(left_flat) | set(right_flat))
        if path not in ignored
        and left_flat.get(path, missing) != right_flat.get(path, missing)
    }


def checkpoint_interval(*, smoke: bool) -> int:
    return 1 if smoke else FORMAL_OPTIMIZER_UPDATES // 4


def validate_run_budget(
    *,
    stop_after_updates: int,
    devices: int,
    gradient_accumulation: int,
    smoke: bool,
) -> None:
    if smoke:
        if (
            stop_after_updates != 2
            or devices not in {1, 2}
            or gradient_accumulation != 1
        ):
            raise ValueError("smoke run must use two updates and no accumulation")
        return
    if (
        stop_after_updates not in {PILOT_OPTIMIZER_UPDATES, FORMAL_OPTIMIZER_UPDATES}
        or devices != 2
        or gradient_accumulation != 4
    ):
        raise ValueError("run differs from the frozen budget")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_content_sha256(value: Mapping[str, object]) -> str:
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(
        content,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise TaskMemoryTrainingError("Git HEAD is invalid")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryTrainingError(f"cannot decode JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise TaskMemoryTrainingError(f"JSON input must be an object: {path}")
    return value


def _stable_seed(namespace: str, seed: int, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_rank_synchronous_draw_plan(
    masters_by_horizon: Mapping[int, Sequence[NativeEpisodeMaster]],
    *,
    episode_count: int,
    seed: int,
    replica_group_size: int,
) -> tuple[TaskMemoryEpisodeSpec, ...]:
    if episode_count <= 0 or episode_count % (5 * replica_group_size):
        raise TaskMemoryTrainingError(
            "draw count must contain complete rank-synchronous H1-H5 groups"
        )
    pools: dict[int, dict[str, list[NativeEpisodeMaster]]] = {}
    for horizon in range(1, 6):
        by_reference: dict[str, list[NativeEpisodeMaster]] = {}
        for master in masters_by_horizon.get(horizon, ()):
            if master.role == "adaptation" and len(master.scan_ids) >= horizon:
                by_reference.setdefault(master.reference_id, []).append(master)
        if not by_reference:
            raise TaskMemoryTrainingError(f"T{horizon} training bucket is empty")
        for values in by_reference.values():
            values.sort(
                key=lambda item: hashlib.sha256(
                    f"task-memory-master:{seed}:{horizon}:{item.sequence_id}".encode(
                        "ascii"
                    )
                ).hexdigest()
            )
        pools[horizon] = by_reference

    result = []
    for group_index in range(episode_count // replica_group_size):
        horizon = group_index % 5 + 1
        bucket_round = group_index // 5
        by_reference = pools[horizon]
        references = sorted(
            by_reference,
            key=lambda reference: hashlib.sha256(
                f"task-memory-reference:{seed}:{horizon}:{reference}".encode("ascii")
            ).hexdigest(),
        )
        for replica_index in range(replica_group_size):
            draw_index = group_index * replica_group_size + replica_index
            position = bucket_round * replica_group_size + replica_index
            reference = references[position % len(references)]
            choices = by_reference[reference]
            master = choices[(position // len(references)) % len(choices)]
            result.append(
                TaskMemoryEpisodeSpec.from_master(
                    master,
                    horizon=horizon,
                    augmentation_seed=_stable_seed(
                        "task-memory-augmentation", seed, draw_index
                    ),
                    draw_index=draw_index,
                    bucket="single_scan" if horizon == 1 else f"T{horizon}",
                )
            )
    return tuple(result)


def build_real_smoke_draw_plan(
    masters: Sequence[NativeEpisodeMaster],
    *,
    episode_count: int,
    seed: int,
) -> tuple[TaskMemoryEpisodeSpec, ...]:
    matches = tuple(
        master
        for master in masters
        if master.reference_id == SMOKE_REFERENCE_ID
        and master.sequence_id == SMOKE_SEQUENCE_ID
    )
    if len(matches) != 1 or episode_count <= 0:
        raise TaskMemoryTrainingError("fixed real smoke H5 master is unavailable")
    master = matches[0]
    return tuple(
        TaskMemoryEpisodeSpec.from_master(
            master,
            horizon=5,
            augmentation_seed=_stable_seed(
                "task-memory-real-smoke-augmentation", seed, draw_index
            ),
            draw_index=draw_index,
            bucket="T5",
        )
        for draw_index in range(episode_count)
    )


def classify_smoke_identity_events(
    stage_identities: Sequence[set[int] | frozenset[int]],
) -> dict[str, object]:
    if len(stage_identities) < 4:
        raise ValueError("real smoke requires at least four identity stages")
    normalized = tuple(frozenset(int(value) for value in stage) for stage in stage_identities)
    if any(value < 0 for stage in normalized for value in stage):
        raise ValueError("real smoke identities must be non-negative")
    later = frozenset().union(*normalized[1:])
    inherited = sorted(normalized[0] & later)
    newborn = sorted(later - normalized[0])
    reappearing = []
    for identity in sorted(frozenset().union(*normalized)):
        present = [identity in stage for stage in normalized]
        if any(
            present[first]
            and not present[missing]
            and present[again]
            for first in range(len(present) - 2)
            for missing in range(first + 1, len(present) - 1)
            for again in range(missing + 1, len(present))
        ):
            reappearing.append(identity)
    if not inherited:
        raise ValueError("real smoke lacks an inherited entity")
    if not reappearing:
        raise ValueError("real smoke lacks an absence and reappearance event")
    if not newborn:
        raise ValueError("real smoke lacks a newborn entity")
    return {
        "absent_entity_id": reappearing[0],
        "inherited_entity_id": inherited[0],
        "newborn_entity_id": newborn[0],
        "reappearing_entity_id": reappearing[0],
        "stage_identity_counts": [len(stage) for stage in normalized],
    }


def build_real_gradient_smoke_payload(
    *,
    code_commit: str,
    common_initialization_sha256: str,
    initial_task_read_sha256: str,
    final_task_read_sha256: str,
    initialization_audit: Mapping[str, object],
    load_audit: Mapping[str, object],
    plan_summary: Mapping[str, object],
    progress: Mapping[str, object],
    training_audit: Sequence[Mapping[str, object]],
    elapsed_seconds: float,
    gpu_name: str,
) -> dict[str, object]:
    if len(training_audit) != 2:
        raise TaskMemoryTrainingError("gradient smoke requires exactly two updates")
    first, second = training_audit
    first_gradient = first.get("task_read_gradient", {})
    second_gradient = second.get("task_read_gradient", {})
    first_change = first.get("task_read_parameter_change", {})
    second_change = second.get("task_read_parameter_change", {})
    if any(
        not isinstance(value, Mapping)
        for value in (first_gradient, second_gradient, first_change, second_change)
    ):
        raise TaskMemoryTrainingError("gradient smoke audit is incomplete")

    def positive(mapping: Mapping[str, object], name: str) -> bool:
        value = mapping.get(name)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0

    def zero(mapping: Mapping[str, object], name: str) -> bool:
        value = mapping.get(name)
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0

    def clean_gradient(mapping: Mapping[str, object]) -> bool:
        return mapping.get("missing_gradient_names") == [] and mapping.get(
            "nonfinite_gradient_names"
        ) == []

    def chunk_has_gradient(audit: Mapping[str, object]) -> bool:
        records = audit.get("chunk_state_gradients")
        return isinstance(records, Sequence) and any(
            isinstance(record, Mapping) and positive(record, "gradient_norm")
            for record in records
        )

    def boundaries_detached(audit: Mapping[str, object]) -> bool:
        values = audit.get("chunk_boundary_state_detached")
        return isinstance(values, list) and bool(values) and all(value is True for value in values)

    def route_commit_exact(audit: Mapping[str, object]) -> bool:
        return (
            audit.get("horizon") == 5
            and audit.get("route_calls") == 5
            and audit.get("commit_calls") == 5
            and audit.get("state_parity_max_abs") == 0.0
        )

    lineage = [
        record
        for audit in training_audit
        for record in audit.get("stage_lineage", [])
        if isinstance(record, Mapping)
    ]
    lineage_fields_are_prediction_only = all(
        not any(
            str(name).startswith(("gt", "target"))
            for name in record
        )
        for record in lineage
    )
    events = plan_summary.get("smoke_events")
    required_events = {
        "absent_entity_id",
        "inherited_entity_id",
        "newborn_entity_id",
        "reappearing_entity_id",
    }
    gates = {
        "adapter_weights_changed": initial_task_read_sha256
        != final_task_read_sha256
        and positive(first_change, "total_change_norm")
        and positive(second_change, "total_change_norm"),
        "chunk_boundaries_detached": all(
            boundaries_detached(audit) for audit in training_audit
        ),
        "common_initialization_exact": common_initialization_sha256
        == initial_task_read_sha256,
        "event_panel_complete": isinstance(events, Mapping)
        and required_events <= set(events),
        "first_update_zero_init_boundary": clean_gradient(first_gradient)
        and positive(first_gradient, "output_projection_gradient_norm")
        and zero(first_gradient, "upstream_gradient_norm"),
        "prediction_only_route_birth_lineage": len(lineage) == 10
        and lineage_fields_are_prediction_only
        and any(int(record.get("birth_count", 0)) > 0 for record in lineage)
        and any(int(record.get("matched_query_count", 0)) > 0 for record in lineage),
        "r1_subtree_exact": load_audit.get("r1_subtree_exact") is True,
        "route_commit_once_per_stage": all(
            route_commit_exact(audit) for audit in training_audit
        ),
        "second_update_reaches_adapter_upstream": clean_gradient(second_gradient)
        and positive(second_gradient, "upstream_gradient_norm"),
        "two_optimizer_updates": all(
            audit.get("optimizer_step") is True for audit in training_audit
        )
        and progress.get("completed_global_episodes")
        == progress.get("next_draw_index")
        and progress.get("next_draw_index") in {2, 4},
        "within_chunk_state_gradient": chunk_has_gradient(second),
        "zero_initialized_output_projection": initialization_audit.get(
            "output_projection_zero"
        )
        is True,
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    if failed:
        raise TaskMemoryTrainingError(
            f"real gradient smoke failed gates: {', '.join(failed)}"
        )
    payload: dict[str, object] = {
        "code_commit": code_commit,
        "common_initialization_sha256": common_initialization_sha256,
        "elapsed_seconds": elapsed_seconds,
        "final_task_read_sha256": final_task_read_sha256,
        "gates": gates,
        "gpu_name": gpu_name,
        "initial_task_read_sha256": initial_task_read_sha256,
        "initialization": dict(initialization_audit),
        "load": dict(load_audit),
        "plan": dict(plan_summary),
        "progress": dict(progress),
        "r1_disabled_parity_evidence": (
            "repo:artifacts/task_memory_retention_v2/implementation/"
            "r1_load_report.json"
        ),
        "schema_version": "task-memory-real-gradient-smoke-v1",
        "status": "PASS",
        "updates": [dict(audit) for audit in training_audit],
    }
    payload["content_sha256"] = _json_content_sha256(payload)
    return payload


def _episode_current_identity_sets(
    episode: TaskMemoryEpisode,
) -> tuple[frozenset[int], ...]:
    result = []
    for stage in episode.stage_samples:
        labels = stage.model_sample[2]
        if not isinstance(labels, (torch.Tensor,)) and not hasattr(labels, "shape"):
            raise TaskMemoryTrainingError("real smoke labels are unavailable")
        if len(labels.shape) != 2 or labels.shape[1] < 2:
            raise TaskMemoryTrainingError("real smoke labels are malformed")
        current = stage.local_stage_ids == int(stage.local_stage_ids.max().item())
        identity_column = torch.as_tensor(labels[:, 1], dtype=torch.long)
        identities = frozenset(
            int(value)
            for value in identity_column[current].unique().tolist()
            if int(value) >= 0
        )
        result.append(identities)
    return tuple(result)


class _MultiHorizonEpisodeDataset(Dataset):
    def __init__(
        self,
        bases: Mapping[int, object],
        draw_plan: Sequence[TaskMemoryEpisodeSpec],
        *,
        window_mode: str,
    ) -> None:
        self.draw_plan = tuple(draw_plan)
        self._datasets = {
            horizon: TaskMemoryEpisodeDataset(
                base,
                tuple(spec for spec in self.draw_plan if spec.horizon == horizon),
                window_mode=window_mode,
            )
            for horizon, base in bases.items()
        }
        self._local_index = {}
        counts = {horizon: 0 for horizon in bases}
        for index, spec in enumerate(self.draw_plan):
            self._local_index[index] = counts[spec.horizon]
            counts[spec.horizon] += 1

    def __len__(self) -> int:
        return len(self.draw_plan)

    def __getitem__(self, index: int):
        spec = self.draw_plan[index]
        return self._datasets[spec.horizon][self._local_index[index]]


def _resume_progress(path: Path | None) -> TaskMemoryProgress:
    if path is None:
        return TaskMemoryProgress.initial()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    payload = (
        checkpoint.get("task_memory_resume")
        if isinstance(checkpoint, Mapping)
        else None
    )
    progress = payload.get("progress") if isinstance(payload, Mapping) else None
    if not isinstance(progress, Mapping):
        raise TaskMemoryTrainingError("resume checkpoint lacks exact draw progress")
    return TaskMemoryProgress.from_state_dict(progress)


def _build_loader(
    config: DictConfig,
    *,
    data_root: Path,
    metadata: Path,
    data_contract: Path,
    devices: int,
    gradient_accumulation: int,
    next_draw_index: int,
    smoke: bool = False,
) -> tuple[DataLoader, dict[str, object]]:
    contract = _load_json(data_contract)
    reference_by_scene = load_reference_by_scene(metadata)
    role_by_reference = _role_by_reference(contract)
    horizons = (5,) if smoke else tuple(range(1, 6))
    bases = {
        horizon: _rio_base_dataset(config, data_root=data_root, horizon=horizon)
        for horizon in horizons
    }
    masters = {
        horizon: build_native_episode_masters(
            base,
            reference_by_scene=reference_by_scene,
            role_by_reference=role_by_reference,
        )
        for horizon, base in bases.items()
    }
    total_draws = (
        2 if smoke else FORMAL_OPTIMIZER_UPDATES
    ) * gradient_accumulation * devices
    plan = (
        build_real_smoke_draw_plan(
            masters[5], episode_count=total_draws, seed=int(config.general.seed)
        )
        if smoke
        else build_rank_synchronous_draw_plan(
            masters,
            episode_count=total_draws,
            seed=int(config.general.seed),
            replica_group_size=devices,
        )
    )
    remaining = tuple(spec for spec in plan if spec.draw_index >= next_draw_index)
    if not remaining or remaining[0].draw_index != next_draw_index:
        raise TaskMemoryTrainingError("resume draw cursor is outside the frozen plan")
    dataset = _MultiHorizonEpisodeDataset(
        bases,
        remaining,
        window_mode=str(config.task_memory_training.window_mode),
    )
    smoke_events = (
        classify_smoke_identity_events(
            _episode_current_identity_sets(dataset[0])
        )
        if smoke
        else None
    )
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    sampler = (
        None
        if devices == 1
        else DistributedSampler(
            dataset,
            num_replicas=devices,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    )
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(config.data.train_collation)
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=int(config.data.num_workers),
        pin_memory=bool(config.data.pin_memory),
        collate_fn=collator,
        persistent_workers=int(config.data.num_workers) > 0,
    )
    bucket_counts = {
        "single_scan" if horizon == 1 else f"T{horizon}": sum(
            spec.horizon == horizon for spec in plan
        )
        for horizon in range(1, 6)
    }
    return loader, {
        "bucket_counts": bucket_counts,
        "next_draw_index": next_draw_index,
        "remaining_global_draws": len(remaining),
        "smoke_events": smoke_events,
        "source_plan_sha256": hashlib.sha256(
            "\n".join(spec.episode_id for spec in plan).encode("ascii")
        ).hexdigest(),
        "total_global_draws": len(plan),
    }


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def remap_r1_training_state(
    state: Mapping[str, torch.Tensor],
    *,
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    mapped = {}
    aliases = {}
    for source_name, tensor in state.items():
        target_name = source_name
        if source_name not in target_keys and source_name.startswith("criterion."):
            candidate = source_name.replace(
                "criterion.", "criterion.base_criterion.", 1
            )
            if candidate in target_keys:
                target_name = candidate
                aliases[source_name] = candidate
        if target_name in mapped:
            raise TaskMemoryTrainingError("R1 criterion alias collides with a state key")
        mapped[target_name] = tensor
    return mapped, aliases


def _load_r1(system: TaskMemoryTrainer, checkpoint: Path) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise TaskMemoryTrainingError("R1 checkpoint lacks a state_dict")
    observed = system.state_dict()
    mapped_state, aliases = remap_r1_training_state(
        state, target_keys=set(observed)
    )
    if bool(system.config.task_memory_training.state_enabled):
        audit = strict_load_r1_task_memory(system, mapped_state)
    else:
        incompatible = system.load_state_dict(mapped_state, strict=True)
        audit = {
            "loaded_key_count": len(mapped_state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    observed = system.state_dict()
    mismatches = sorted(
        name
        for name, expected in mapped_state.items()
        if not isinstance(expected, torch.Tensor)
        or name not in observed
        or not torch.equal(observed[name].detach().cpu(), expected.detach().cpu())
    )
    return {
        **audit,
        "r1_key_aliases": aliases,
        "r1_subtree_exact": not mismatches,
        "r1_subtree_mismatches": mismatches,
    }


def _apply_common_task_read_initialization(
    system: TaskMemoryTrainer,
    path: Path,
) -> str | None:
    if not bool(system.config.task_memory_training.state_enabled):
        return None
    task_read = system.model.task_read
    if task_read is None:
        raise TaskMemoryTrainingError("enabled variant lacks task_read")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
        else:
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in task_read.state_dict().items()
            }
            temporary = path.with_suffix(f"{path.suffix}.tmp")
            torch.save(state, temporary)
            temporary.replace(path)
        fcntl.flock(lock, fcntl.LOCK_UN)
    task_read.load_state_dict(state, strict=True)
    return _tensor_state_sha256(state)


def _parser() -> argparse.ArgumentParser:
    resolver = _load_json(LOCAL_ASSET_RESOLVER) if LOCAL_ASSET_RESOLVER.exists() else {}

    def external_default(environment: str, logical: str, fallback=None):
        value = os.environ.get(environment, resolver.get(logical, fallback))
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=external_default(
            "PERSIST4D_R1_CHECKPOINT", "external:r1_checkpoint"
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=external_default(
            "PERSIST4D_CONCERTO_PRETRAINED", "external:concerto_pretrained"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=external_default(
            "PERSIST4D_DATA_ROOT", "external:data_root", DEFAULT_DATA_ROOT
        ),
    )
    parser.add_argument(
        "--rio-metadata",
        type=Path,
        default=external_default(
            "PERSIST4D_RIO_METADATA", "external:rio_metadata"
        ),
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=external_default(
            "PERSIST4D_RUN_ROOT", "external:run_root"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument(
        "--stop-after-updates", type=int, default=FORMAL_OPTIMIZER_UPDATES
    )
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    missing = [
        option
        for option, value in (
            ("--checkpoint", args.checkpoint),
            ("--pretrained", args.pretrained),
            ("--data-root", args.data_root),
            ("--rio-metadata", args.rio_metadata),
            ("--external-root", args.external_root),
        )
        if value is None
    ]
    if missing:
        parser.error(
            "missing external inputs (set environment variables or pass flags): "
            + ", ".join(missing)
        )
    validate_run_budget(
        stop_after_updates=args.stop_after_updates,
        devices=args.devices,
        gradient_accumulation=args.gradient_accumulation,
        smoke=args.smoke,
    )
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    pretrained = args.pretrained.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    metadata = args.rio_metadata.expanduser().resolve(strict=True)
    data_contract = args.data_contract.expanduser().resolve(strict=True)
    resume = args.resume.expanduser().resolve(strict=True) if args.resume else None
    if checkpoint.stat().st_size != R1_BYTES or _sha256(checkpoint) != R1_SHA256:
        raise TaskMemoryTrainingError("R1 checkpoint identity differs")
    namespace = "smoke" if args.smoke else "formal"
    training_root = (args.external_root / "training").resolve()
    run_dir = training_root / namespace / args.variant
    artifact_dir = (args.artifact_root / namespace / args.variant).resolve()
    if resume is None and run_dir.exists() and any(run_dir.iterdir()):
        raise TaskMemoryTrainingError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(45, workers=True)
    config = compose_variant_config(
        args.variant,
        pretrained=pretrained,
        run_dir=run_dir,
    )
    with open_dict(config):
        config.task_memory_training.devices = args.devices
        config.task_memory_training.gradient_accumulation = args.gradient_accumulation
        config.task_memory_training.effective_episode_batch = (
            args.devices * args.gradient_accumulation
        )
        config.task_memory_training.smoke_audit = args.smoke
        config.trainer.max_steps = args.stop_after_updates
        config.trainer.strategy = (
            "ddp_find_unused_parameters_false" if args.devices > 1 else "auto"
        )
    progress = _resume_progress(resume)
    loader, plan_summary = _build_loader(
        config,
        data_root=data_root,
        metadata=metadata,
        data_contract=data_contract,
        devices=args.devices,
        gradient_accumulation=args.gradient_accumulation,
        next_draw_index=progress.next_draw_index,
        smoke=args.smoke,
    )
    seed_everything(45, workers=True)
    system = TaskMemoryTrainer(config)
    load_audit = _load_r1(system, checkpoint)
    common_init_sha = _apply_common_task_read_initialization(
        system,
        training_root / "common/task_read_init.pt",
    )
    if args.smoke and not bool(config.task_memory_training.state_enabled):
        raise TaskMemoryTrainingError(
            "real gradient smoke requires Q-INDEP or Q-TALA"
        )
    initialization_audit = system.begin_smoke_audit() if args.smoke else None
    initial_task_read_sha = (
        _tensor_state_sha256(system.model.task_read.state_dict())
        if args.smoke
        else None
    )
    seed_everything(45, workers=True)
    if "LOCAL_RANK" not in os.environ:
        OmegaConf.save(
            config, run_dir / "resolved_config.local.yaml", resolve=True
        )
        _atomic_json(
            artifact_dir / "run_plan.json",
            {
                "code_commit_at_run": _git_head(),
                "common_task_read_initialization_sha256": common_init_sha,
                "devices": args.devices,
                "gradient_accumulation": args.gradient_accumulation,
                "load_audit": load_audit,
                "plan": plan_summary,
                "resume": resume is not None,
                "r1_checkpoint_sha256": R1_SHA256,
                "scheduler_total_updates": FORMAL_OPTIMIZER_UPDATES,
                "seed": 45,
                "stop_after_updates": args.stop_after_updates,
                "variant": args.variant,
            },
        )
    callback = ModelCheckpoint(
        dirpath=run_dir,
        filename="update={step:04d}",
        every_n_train_steps=checkpoint_interval(smoke=args.smoke),
        save_last=True,
        save_on_train_epoch_end=True,
        save_top_k=-1,
        save_weights_only=False,
        auto_insert_metric_name=False,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=config.trainer.strategy,
        max_epochs=1,
        max_steps=args.stop_after_updates,
        accumulate_grad_batches=1,
        precision="32-true",
        deterministic=False,
        gradient_clip_val=None,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        callbacks=[callback, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=run_dir, name="training_metrics"),
        default_root_dir=run_dir,
        log_every_n_steps=1,
        enable_progress_bar=True,
    )
    started = time.time()
    trainer.fit(
        system, train_dataloaders=loader, ckpt_path=str(resume) if resume else None
    )
    elapsed = time.time() - started
    if not trainer.is_global_zero:
        return 0
    final_task_read_sha = (
        _tensor_state_sha256(system.model.task_read.state_dict())
        if args.smoke
        else None
    )
    checkpoint_paths = sorted(run_dir.glob("*.ckpt"))
    _atomic_json(
        artifact_dir / "run_summary.json",
        {
            "checkpoints": [
                {
                    "bytes": path.stat().st_size,
                    "name": path.name,
                    "sha256": _sha256(path),
                }
                for path in checkpoint_paths
            ],
            "completed_global_step": int(trainer.global_step),
            "elapsed_seconds": elapsed,
            "gpu_hours": elapsed * args.devices / 3600.0,
            "progress": system.progress.state_dict(),
            "status": "COMPLETE"
            if int(trainer.global_step) == args.stop_after_updates
            else "PARTIAL",
            "training_audit": system.training_audit,
            "variant": args.variant,
        },
    )
    if args.smoke:
        if (
            common_init_sha is None
            or initial_task_read_sha is None
            or final_task_read_sha is None
            or initialization_audit is None
        ):
            raise TaskMemoryTrainingError("gradient smoke state audit is unavailable")
        smoke_payload = build_real_gradient_smoke_payload(
            code_commit=_git_head(),
            common_initialization_sha256=common_init_sha,
            initial_task_read_sha256=initial_task_read_sha,
            final_task_read_sha256=final_task_read_sha,
            initialization_audit=initialization_audit,
            load_audit=load_audit,
            plan_summary=plan_summary,
            progress=system.progress.state_dict(),
            training_audit=system.training_audit,
            elapsed_seconds=elapsed,
            gpu_name=torch.cuda.get_device_name(0),
        )
        _atomic_json(
            args.artifact_root.resolve().parent
            / "implementation/real_gradient_smoke.json",
            smoke_payload,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
