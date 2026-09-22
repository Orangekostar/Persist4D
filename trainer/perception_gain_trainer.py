"""Frozen-budget perception adaptation and exact-resume contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.perception_gain import SemanticQueryScorer
from trainer.task_memory_trainer import (
    _reset_lightning_batch_progress_for_sliced_resume,
    capture_task_memory_rng_state,
    restore_task_memory_rng_state,
)
from trainer.trainer import InstanceSegmentation

_RESUME_SCHEMA = "perception-gain-exact-resume-v1"
_SCORER_PREFIX = "model.semantic_query_scorer."


class PerceptionTrainingError(RuntimeError):
    """Raised when a perception training contract is violated."""


def perception_lr_multiplier(
    step: int,
    *,
    total_updates: int = 3000,
    warmup_updates: int = 150,
    minimum_fraction: float = 0.1,
) -> float:
    """Linear warmup followed by one cosine schedule over the full run."""

    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_updates, bool)
        or not isinstance(total_updates, int)
        or isinstance(warmup_updates, bool)
        or not isinstance(warmup_updates, int)
        or total_updates < 2
        or not 1 <= warmup_updates < total_updates
        or not 0 <= step < total_updates
        or not 0.0 < minimum_fraction <= 1.0
    ):
        raise PerceptionTrainingError("scheduler arguments are invalid")
    if step < warmup_updates:
        return (step + 1) / warmup_updates
    denominator = max(1, total_updates - 1 - warmup_updates)
    progress = min(1.0, (step - warmup_updates) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


@dataclass(frozen=True)
class PerceptionRuntimeContract:
    devices: int
    batch_size_per_gpu: int
    gradient_accumulation: int
    precision: str
    gradient_clip_norm: float
    total_updates: int
    evaluation_updates: tuple[int, ...]

    def __post_init__(self) -> None:
        integers = (
            self.devices,
            self.batch_size_per_gpu,
            self.gradient_accumulation,
            self.total_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integers
        ):
            raise PerceptionTrainingError("runtime sizes must be positive integers")
        if self.precision != "32-true":
            raise PerceptionTrainingError(
                "perception training precision must be 32-true"
            )
        if not math.isclose(float(self.gradient_clip_norm), 1.0):
            raise PerceptionTrainingError("gradient clip norm must be 1.0")
        expected_updates = (0, 250, 750, 1500, 2250, 3000)
        if self.total_updates != 3000 or self.evaluation_updates != expected_updates:
            raise PerceptionTrainingError("evaluation/update schedule differs from V1")
        if self.effective_batch_size != 32:
            raise PerceptionTrainingError("effective batch size must be 32")

    @property
    def effective_batch_size(self) -> int:
        return self.devices * self.batch_size_per_gpu * self.gradient_accumulation

    @property
    def total_global_draws(self) -> int:
        return self.total_updates * self.effective_batch_size

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        return self.evaluation_updates[1:]


@dataclass(frozen=True)
class PerceptionProgress:
    completed_optimizer_updates: int
    completed_local_batches: int
    next_global_draw_index: int

    @classmethod
    def initial(cls) -> PerceptionProgress:
        return cls(0, 0, 0)

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.completed_optimizer_updates,
                self.completed_local_batches,
                self.next_global_draw_index,
            )
        ):
            raise PerceptionTrainingError("training progress must be non-negative")

    def advance(
        self,
        *,
        optimizer_updates: int,
        local_batches: int,
        global_draws: int,
    ) -> PerceptionProgress:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (optimizer_updates, local_batches, global_draws)
        ):
            raise PerceptionTrainingError("progress increments must be positive")
        return PerceptionProgress(
            self.completed_optimizer_updates + optimizer_updates,
            self.completed_local_batches + local_batches,
            self.next_global_draw_index + global_draws,
        )

    def state_dict(self) -> dict[str, int | str]:
        return {
            "completed_local_batches": self.completed_local_batches,
            "completed_optimizer_updates": self.completed_optimizer_updates,
            "next_global_draw_index": self.next_global_draw_index,
            "schema_version": _RESUME_SCHEMA,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> PerceptionProgress:
        expected = {
            "completed_local_batches",
            "completed_optimizer_updates",
            "next_global_draw_index",
            "schema_version",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value["schema_version"] != _RESUME_SCHEMA
        ):
            raise PerceptionTrainingError("perception progress checkpoint is invalid")
        return cls(
            completed_optimizer_updates=value["completed_optimizer_updates"],
            completed_local_batches=value["completed_local_batches"],
            next_global_draw_index=value["next_global_draw_index"],
        )


def build_perception_resume_payload(
    progress: PerceptionProgress,
    *,
    rng_by_rank: list[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if not isinstance(progress, PerceptionProgress):
        raise PerceptionTrainingError("resume progress has an invalid type")
    states = (
        [capture_task_memory_rng_state()] if rng_by_rank is None else list(rng_by_rank)
    )
    if not states or any(not isinstance(state, Mapping) for state in states):
        raise PerceptionTrainingError("rank RNG checkpoint is invalid")
    return {
        "progress": progress.state_dict(),
        "rng_by_rank": states,
        "schema_version": _RESUME_SCHEMA,
        "world_size": len(states),
    }


def restore_perception_resume_payload(
    payload: Mapping[str, object],
    *,
    global_rank: int = 0,
    world_size: int = 1,
) -> PerceptionProgress:
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"progress", "rng_by_rank", "schema_version", "world_size"}
        or payload["schema_version"] != _RESUME_SCHEMA
        or not isinstance(payload["progress"], Mapping)
        or not isinstance(payload["rng_by_rank"], list)
        or payload["world_size"] != world_size
        or not 0 <= global_rank < world_size
        or len(payload["rng_by_rank"]) != world_size
        or any(not isinstance(state, Mapping) for state in payload["rng_by_rank"])
    ):
        raise PerceptionTrainingError("exact resume payload is invalid")
    progress = PerceptionProgress.from_state_dict(payload["progress"])
    restore_task_memory_rng_state(payload["rng_by_rank"][global_rank])
    return progress


def strict_load_r1_perception(
    module: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    allow_semantic_scorer: bool,
) -> dict[str, object]:
    """Load every R1 key exactly, permitting only a new Q-SEM scorer."""

    if not isinstance(module, nn.Module) or not isinstance(state_dict, Mapping):
        raise PerceptionTrainingError("R1 load inputs are invalid")
    observed = module.state_dict()
    incoming = set(state_dict)
    expected = set(observed)
    unexpected = sorted(incoming - expected)
    missing = sorted(expected - incoming)
    allowed_missing = (
        sorted(name for name in expected if name.startswith(_SCORER_PREFIX))
        if allow_semantic_scorer
        else []
    )
    if unexpected:
        raise PerceptionTrainingError(f"R1 state has unexpected keys: {unexpected}")
    if missing != allowed_missing:
        unapproved = sorted(set(missing) - set(allowed_missing))
        raise PerceptionTrainingError(
            f"R1 state has unapproved missing keys: {unapproved or missing}"
        )
    try:
        incompatible = module.load_state_dict(state_dict, strict=False)
    except RuntimeError as error:
        raise PerceptionTrainingError(f"R1 tensor contract differs: {error}") from error
    if sorted(incompatible.missing_keys) != missing or incompatible.unexpected_keys:
        raise PerceptionTrainingError("R1 state changed during strict loading")
    loaded = module.state_dict()
    mismatches = sorted(
        name
        for name, expected_tensor in state_dict.items()
        if not isinstance(expected_tensor, Tensor)
        or not torch.equal(loaded[name].detach().cpu(), expected_tensor.detach().cpu())
    )
    if mismatches:
        raise PerceptionTrainingError("R1 subtree did not load exactly")
    return {
        "loaded_key_count": len(state_dict),
        "missing_keys": missing,
        "r1_subtree_exact": True,
        "r1_subtree_mismatches": [],
        "unexpected_keys": unexpected,
    }


def load_frozen_semantic_scorer(module: nn.Module, path: str) -> dict[str, object]:
    scorer = getattr(getattr(module, "model", None), "semantic_query_scorer", None)
    if not isinstance(scorer, SemanticQueryScorer):
        raise PerceptionTrainingError("Q-SEM scorer is unavailable on the model")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("scorer_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise PerceptionTrainingError("scorer checkpoint lacks scorer_state_dict")
    incompatible = scorer.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PerceptionTrainingError("scorer checkpoint is incompatible")
    scorer.requires_grad_(False)
    scorer.eval()
    return {"loaded_key_count": len(state), "path": str(path)}


def audit_optimizer_parameters(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    trainable = {
        id(parameter): name
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        id(parameter): name
        for name, parameter in module.named_parameters()
        if not parameter.requires_grad
    }
    seen: dict[int, int] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            seen[id(parameter)] = seen.get(id(parameter), 0) + 1
    missing = sorted(trainable[identity] for identity in set(trainable) - set(seen))
    duplicates = sorted(
        trainable.get(identity, f"unknown:{identity}")
        for identity, count in seen.items()
        if count != 1
    )
    frozen_in_optimizer = sorted(
        frozen[identity] for identity in set(frozen) & set(seen)
    )
    unknown = sorted(identity for identity in set(seen) - set(trainable) - set(frozen))
    if missing or duplicates or frozen_in_optimizer or unknown:
        raise PerceptionTrainingError(
            "optimizer parameter coverage differs: "
            f"missing={missing}, duplicates={duplicates}, "
            f"frozen={frozen_in_optimizer}, unknown={unknown}"
        )
    return {
        "frozen_parameter_count": len(frozen),
        "status": "PASS",
        "trainable_parameter_count": len(trainable),
        "trainable_parameter_names": sorted(trainable.values()),
    }


def _gradient_snapshot(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> dict[str, object]:
    norms = {}
    missing = []
    nonfinite = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
        else:
            norms[name] = float(gradient.detach().float().norm().cpu().item())
    return {
        "missing_gradient_names": sorted(missing),
        "nonfinite_gradient_names": sorted(nonfinite),
        "nonzero_gradient_names": sorted(
            name for name, norm in norms.items() if norm > 0
        ),
        "parameter_gradient_norms": dict(sorted(norms.items())),
        "total_gradient_norm": math.sqrt(sum(norm * norm for norm in norms.values())),
    }


def _parameter_snapshot(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


class PerceptionGainTrainer(InstanceSegmentation):
    """Normal InstanceSegmentation supervision with a frozen V1 budget."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.progress = PerceptionProgress.initial()
        self.training_audit: list[dict[str, object]] = []
        self.optimizer_audit: dict[str, object] | None = None
        self._pending_perception_rng: Mapping[str, object] | None = None
        self._pre_step_parameters: dict[str, Tensor] | None = None
        self._pre_step_gradients: dict[str, object] | None = None

    def configure_optimizers(self):
        settings = self.config.perception_training
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise PerceptionTrainingError("perception optimizer has no parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(settings.existing_lr),
            betas=tuple(float(value) for value in settings.betas),
            eps=float(settings.eps),
            weight_decay=float(settings.weight_decay),
        )
        self.optimizer_audit = audit_optimizer_parameters(self, optimizer)
        total_updates = int(settings.optimizer_updates)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: perception_lr_multiplier(
                min(step, total_updates - 1),
                total_updates=total_updates,
                warmup_updates=int(settings.warmup_updates),
                minimum_fraction=float(settings.minimum_lr_fraction),
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        if len(self.training_audit) < 2:
            self._pre_step_gradients = _gradient_snapshot(self.named_parameters())
            self._pre_step_parameters = _parameter_snapshot(self)

    def optimizer_step(self, *args: object, **kwargs: object) -> None:
        super().optimizer_step(*args, **kwargs)
        settings = self.config.perception_training
        self.progress = self.progress.advance(
            optimizer_updates=1,
            local_batches=int(settings.gradient_accumulation),
            global_draws=int(settings.effective_batch_size),
        )
        if (
            self._pre_step_parameters is not None
            and self._pre_step_gradients is not None
        ):
            current = _parameter_snapshot(self)
            if set(current) != set(self._pre_step_parameters):
                raise PerceptionTrainingError("trainable parameter set changed")
            changes = {
                name: float(
                    (current[name] - self._pre_step_parameters[name])
                    .float()
                    .norm()
                    .item()
                )
                for name in current
            }
            self.training_audit.append(
                {
                    "gradient": self._pre_step_gradients,
                    "optimizer_update": self.progress.completed_optimizer_updates,
                    "parameter_change_norms": dict(sorted(changes.items())),
                    "total_parameter_change_norm": math.sqrt(
                        sum(value * value for value in changes.values())
                    ),
                }
            )
            self._pre_step_parameters = None
            self._pre_step_gradients = None

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if (
            not isinstance(checkpoint.get("optimizer_states"), list)
            or len(checkpoint["optimizer_states"]) != 1
            or not isinstance(checkpoint.get("lr_schedulers"), list)
            or len(checkpoint["lr_schedulers"]) != 1
        ):
            raise PerceptionTrainingError(
                "exact resume requires one optimizer and one scheduler"
            )
        local_rng = capture_task_memory_rng_state()
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        rng_by_rank: list[Mapping[str, object] | None] = [None] * world_size
        if world_size > 1:
            torch.distributed.all_gather_object(rng_by_rank, local_rng)
        else:
            rng_by_rank[0] = local_rng
        if any(state is None for state in rng_by_rank):
            raise PerceptionTrainingError("failed to gather rank RNG checkpoints")
        checkpoint["perception_resume"] = build_perception_resume_payload(
            self.progress,
            rng_by_rank=[state for state in rng_by_rank if state is not None],
        )
        checkpoint["perception_training_audit"] = list(self.training_audit)
        if self.config.get("perception_recipe") is not None:
            from omegaconf import OmegaConf

            checkpoint["perception_recipe"] = OmegaConf.to_container(
                self.config.perception_recipe,
                resolve=True,
            )

    def on_load_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        if self.config.get("perception_recipe") is not None:
            from omegaconf import OmegaConf

            from scripts.perception_gain_v2_config import validate_resume_recipe

            resolved = OmegaConf.to_container(self.config, resolve=True)
            validate_resume_recipe(
                checkpoint,
                resolved["perception_recipe"],
                legacy_config=resolved.get("perception_legacy_resume_config"),
            )
        if (
            not isinstance(checkpoint.get("optimizer_states"), list)
            or len(checkpoint["optimizer_states"]) != 1
            or not isinstance(checkpoint.get("lr_schedulers"), list)
            or len(checkpoint["lr_schedulers"]) != 1
        ):
            raise PerceptionTrainingError(
                "exact resume checkpoint lacks optimizer or scheduler state"
            )
        payload = checkpoint.get("perception_resume")
        if not isinstance(payload, Mapping):
            raise PerceptionTrainingError("exact resume payload is unavailable")
        progress_payload = payload.get("progress")
        rng_by_rank = payload.get("rng_by_rank")
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        global_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        if (
            payload.get("schema_version") != _RESUME_SCHEMA
            or not isinstance(progress_payload, Mapping)
            or payload.get("world_size") != world_size
            or not isinstance(rng_by_rank, list)
            or len(rng_by_rank) != world_size
            or not isinstance(rng_by_rank[global_rank], Mapping)
        ):
            raise PerceptionTrainingError("exact resume payload is invalid")
        self.progress = PerceptionProgress.from_state_dict(progress_payload)
        _reset_lightning_batch_progress_for_sliced_resume(
            checkpoint,
            completed_local_episodes=self.progress.completed_local_batches,
        )
        self._pending_perception_rng = rng_by_rank[global_rank]
        audit = checkpoint.get("perception_training_audit", [])
        if not isinstance(audit, list):
            raise PerceptionTrainingError("training audit checkpoint is invalid")
        self.training_audit = list(audit)

    def on_train_start(self) -> None:
        if self._pending_perception_rng is not None:
            restore_task_memory_rng_state(self._pending_perception_rng)
            self._pending_perception_rng = None


class SemanticScorerTrainer(torch.nn.Module):
    """Small scorer objective used by the fixed 500-update Q-SEM pretrain."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = SemanticQueryScorer()

    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != 128:
            raise PerceptionTrainingError("scorer features must have shape [N,128]")
        if targets.ndim != 1 or targets.numel() != features.shape[0]:
            raise PerceptionTrainingError("scorer targets must align with features")
        return F.binary_cross_entropy_with_logits(
            self.scorer(features),
            targets.to(device=features.device, dtype=features.dtype),
        )


__all__ = [
    "PerceptionGainTrainer",
    "PerceptionProgress",
    "PerceptionRuntimeContract",
    "PerceptionTrainingError",
    "SemanticScorerTrainer",
    "audit_optimizer_parameters",
    "build_perception_resume_payload",
    "load_frozen_semantic_scorer",
    "perception_lr_multiplier",
    "restore_perception_resume_payload",
    "strict_load_r1_perception",
]
