"""Stage-wise training with exact TaskMemory episode and resume contracts."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch import Tensor

from datasets.task_memory_episode import (
    StageMeta,
    TaskMemoryEpisodeBatch,
    TaskMemoryStageBatch,
)
from models.task_memory_criterion import TaskMemoryCriterion, TaskMemoryLossResult
from models.task_memory_routing import (
    CommitResult,
    EntityRoute,
    PredictionObservation,
    commit_entities,
)
from models.task_memory_state import TaskMemoryState
from models.task_memory_supervision import (
    IndexPair,
    QualifiedIdentity,
    TrainingIdentityLedger,
    prediction_stage_ids,
)
from trainer.persist4d_allt_trainer import connect_all_trainable_parameters
from trainer.trainer import InstanceSegmentation, _configured_objective_loss

_RESUME_SCHEMA = "task-memory-exact-resume-v1"
_TASK_ADAPTER_PREFIX = "model.task_read."


class TaskMemoryTrainerError(RuntimeError):
    """Raised when a training or exact-resume invariant differs."""


def _reset_lightning_batch_progress_for_sliced_resume(
    checkpoint: Mapping[str, object], *, completed_local_episodes: int
) -> None:
    loops = checkpoint.get("loops")
    if loops is None:
        return
    if not isinstance(loops, MutableMapping):
        raise TaskMemoryTrainerError("Lightning resume loop state is invalid")
    fit_loop = loops.get("fit_loop")
    if not isinstance(fit_loop, MutableMapping):
        raise TaskMemoryTrainerError("Lightning fit-loop resume state is invalid")
    batch_progress = fit_loop.get("epoch_loop.batch_progress")
    if not isinstance(batch_progress, MutableMapping):
        raise TaskMemoryTrainerError("Lightning batch resume state is invalid")
    for scope in ("current", "total"):
        tracker = batch_progress.get(scope)
        if not isinstance(tracker, MutableMapping) or set(tracker) != {
            "completed",
            "processed",
            "ready",
            "started",
        }:
            raise TaskMemoryTrainerError("Lightning batch progress schema differs")
        values = {name: tracker[name] for name in tracker}
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in values.values())
            or values["ready"] != completed_local_episodes
            or values["started"] != completed_local_episodes
            or values["processed"] != completed_local_episodes
            or values["completed"]
            not in {completed_local_episodes - 1, completed_local_episodes}
        ):
            raise TaskMemoryTrainerError(
                "Lightning batch progress differs from the task-memory cursor"
            )
        tracker.update({name: 0 for name in tracker})
    batch_progress["is_last_batch"] = False


def stage_mean_coefficients(horizon: int) -> tuple[float, ...]:
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or not 1 <= horizon <= 5
    ):
        raise TaskMemoryTrainerError("horizon must be an integer in [1,5]")
    return (1.0 / horizon,) * horizon


def tbptt_stage_chunks(
    horizon: int, *, chunk_size: int = 2
) -> tuple[tuple[int, ...], ...]:
    stage_mean_coefficients(horizon)
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise TaskMemoryTrainerError("TBPTT chunk size must be positive")
    return tuple(
        tuple(range(start, min(start + chunk_size, horizon)))
        for start in range(0, horizon, chunk_size)
    )


def training_stage_chunks(
    horizon: int,
    *,
    state_enabled: bool,
    window_mode: str,
    tbptt_steps: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(state_enabled, bool):
        raise TaskMemoryTrainerError("state_enabled must be boolean")
    if window_mode not in {"full_history", "local_pair"}:
        raise TaskMemoryTrainerError("window_mode is invalid")
    chunk_size = (
        1 if not state_enabled and window_mode == "full_history" else tbptt_steps
    )
    return tbptt_stage_chunks(horizon, chunk_size=chunk_size)


def task_memory_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
    minimum_fraction: float = 0.1,
) -> float:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or total_steps < 2
        or not 0 <= step < total_steps
        or not 0.0 < warmup_fraction < 1.0
        or not 0.0 < minimum_fraction <= 1.0
    ):
        raise TaskMemoryTrainerError("scheduler arguments are invalid")
    warmup_steps = min(total_steps - 1, max(1, round(total_steps * warmup_fraction)))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    denominator = max(1, total_steps - 1 - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


def task_read_gradient_snapshot(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> dict[str, object]:
    per_parameter = {}
    missing = []
    nonfinite = []
    output_squared = 0.0
    upstream_squared = 0.0
    found = 0
    for name, parameter in named_parameters:
        if not name.startswith(_TASK_ADAPTER_PREFIX) or not parameter.requires_grad:
            continue
        found += 1
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
            continue
        norm = float(gradient.detach().float().norm().cpu().item())
        per_parameter[name] = norm
        if name.startswith(f"{_TASK_ADAPTER_PREFIX}output_projection."):
            output_squared += norm * norm
        else:
            upstream_squared += norm * norm
    if found == 0:
        raise TaskMemoryTrainerError("task-read gradient audit found no parameters")
    return {
        "missing_gradient_names": sorted(missing),
        "nonfinite_gradient_names": sorted(nonfinite),
        "output_projection_gradient_norm": math.sqrt(output_squared),
        "parameter_gradient_norms": dict(sorted(per_parameter.items())),
        "upstream_gradient_norm": math.sqrt(upstream_squared),
    }


def _task_read_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
        if name.startswith(_TASK_ADAPTER_PREFIX) and parameter.requires_grad
    }


def _task_read_parameter_change(
    named_parameters: Iterable[tuple[str, Tensor]],
    previous: Mapping[str, Tensor],
) -> tuple[dict[str, object], dict[str, Tensor]]:
    current = _task_read_parameters(named_parameters)
    if set(current) != set(previous) or not current:
        raise TaskMemoryTrainerError("task-read parameter audit schema changed")
    changes = {
        name: float((current[name] - previous[name]).float().norm().item())
        for name in current
    }
    return (
        {
            "changed_parameter_names": sorted(
                name for name, norm in changes.items() if norm > 0.0
            ),
            "parameter_change_norms": dict(sorted(changes.items())),
            "total_change_norm": math.sqrt(
                sum(norm * norm for norm in changes.values())
            ),
        },
        current,
    )


@dataclass(frozen=True)
class TaskMemoryProgress:
    completed_local_episodes: int
    completed_global_episodes: int
    completed_global_stages: int
    next_draw_index: int

    @classmethod
    def initial(cls, *, next_draw_index: int = 0) -> TaskMemoryProgress:
        return cls(0, 0, 0, next_draw_index)

    def __post_init__(self) -> None:
        values = (
            self.completed_local_episodes,
            self.completed_global_episodes,
            self.completed_global_stages,
            self.next_draw_index,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise TaskMemoryTrainerError(
                "training progress must use non-negative integers"
            )

    def advance(
        self,
        *,
        local_draw_indices: Sequence[int],
        horizon: int,
        world_size: int,
        global_rank: int,
    ) -> TaskMemoryProgress:
        stage_mean_coefficients(horizon)
        if (
            isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
            or isinstance(global_rank, bool)
            or not isinstance(global_rank, int)
            or not 0 <= global_rank < world_size
            or tuple(local_draw_indices) != (self.next_draw_index + global_rank,)
        ):
            raise TaskMemoryTrainerError("draw cursor is not rank-synchronous")
        return TaskMemoryProgress(
            completed_local_episodes=self.completed_local_episodes + 1,
            completed_global_episodes=self.completed_global_episodes + world_size,
            completed_global_stages=self.completed_global_stages + horizon * world_size,
            next_draw_index=self.next_draw_index + world_size,
        )

    def state_dict(self) -> dict[str, int | str]:
        return {
            "completed_global_episodes": self.completed_global_episodes,
            "completed_global_stages": self.completed_global_stages,
            "completed_local_episodes": self.completed_local_episodes,
            "next_draw_index": self.next_draw_index,
            "schema_version": _RESUME_SCHEMA,
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> TaskMemoryProgress:
        fields = {
            "completed_global_episodes",
            "completed_global_stages",
            "completed_local_episodes",
            "next_draw_index",
            "schema_version",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["schema_version"] != _RESUME_SCHEMA
        ):
            raise TaskMemoryTrainerError("training progress checkpoint is invalid")
        return cls(
            completed_local_episodes=value["completed_local_episodes"],
            completed_global_episodes=value["completed_global_episodes"],
            completed_global_stages=value["completed_global_stages"],
            next_draw_index=value["next_draw_index"],
        )


def capture_task_memory_rng_state() -> dict[str, object]:
    return {
        "cuda": [
            state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.random.get_rng_state().detach().cpu().clone(),
    }


def restore_task_memory_rng_state(value: Mapping[str, object]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "cuda",
        "numpy",
        "python",
        "torch",
    }:
        raise TaskMemoryTrainerError("RNG checkpoint is invalid")
    torch_state = value["torch"]
    cuda_states = value["cuda"]
    if not isinstance(torch_state, Tensor) or not isinstance(cuda_states, list):
        raise TaskMemoryTrainerError("RNG tensor checkpoint is invalid")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.random.set_rng_state(torch_state.detach().cpu())
    if cuda_states:
        if not torch.cuda.is_available() or any(
            not isinstance(state, Tensor) for state in cuda_states
        ):
            raise TaskMemoryTrainerError("CUDA RNG checkpoint cannot be restored")
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])


def _unit_interval(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TaskMemoryTrainerError(f"{name} must be finite and within [0,1]")
    return float(value)


def build_final_prediction_observation(
    output: Mapping[str, object],
    stage_meta: Sequence[StageMeta],
    *,
    background_class: int,
    confidence_threshold: float,
    mask_threshold: float,
    minimum_mask_support: int,
) -> PredictionObservation:
    features = output.get("query_features")
    logits = output.get("pred_logits")
    masks = output.get("pred_masks")
    if (
        not isinstance(features, Tensor)
        or features.ndim != 3
        or not isinstance(logits, Tensor)
        or logits.ndim != 3
        or features.shape[:2] != logits.shape[:2]
        or isinstance(masks, (str, bytes))
        or not isinstance(masks, Sequence)
        or len(masks) != features.shape[0]
        or len(stage_meta) != features.shape[0]
        or any(not isinstance(meta, StageMeta) for meta in stage_meta)
    ):
        raise TaskMemoryTrainerError("final prediction fields are misaligned")
    if (
        isinstance(background_class, bool)
        or not isinstance(background_class, int)
        or not 0 <= background_class < logits.shape[2]
        or isinstance(minimum_mask_support, bool)
        or not isinstance(minimum_mask_support, int)
        or minimum_mask_support <= 0
    ):
        raise TaskMemoryTrainerError("final prediction policy is invalid")
    confidence_threshold = _unit_interval(
        confidence_threshold, name="confidence_threshold"
    )
    mask_threshold = _unit_interval(mask_threshold, name="mask_threshold")
    class_prob = logits.detach().softmax(dim=-1).to(dtype=features.dtype)
    foreground = class_prob.clone()
    foreground[..., background_class] = -torch.inf
    confidence = foreground.amax(dim=-1)
    current_support = []
    previous_support = []
    for batch_index, (mask, meta) in enumerate(zip(masks, stage_meta, strict=True)):
        if (
            not isinstance(mask, Tensor)
            or mask.ndim != 2
            or mask.shape[1] != features.shape[1]
        ):
            raise TaskMemoryTrainerError("final prediction mask is invalid")
        stage_ids = prediction_stage_ids(meta, mask.shape[0]).to(device=mask.device)
        latest = int(stage_ids.max().item())
        positive = mask.detach().sigmoid() >= mask_threshold
        current_support.append(
            positive[stage_ids == latest].sum(dim=0).ge(minimum_mask_support)
        )
        previous = stage_ids != latest
        previous_support.append(
            positive[previous].sum(dim=0).ge(minimum_mask_support)
            if previous.any().item()
            else torch.zeros(features.shape[1], dtype=torch.bool, device=mask.device)
        )
    current = torch.stack(current_support).to(device=features.device)
    previous = torch.stack(previous_support).to(device=features.device)
    observation = PredictionObservation(
        features=features,
        class_prob=class_prob,
        confidence=confidence,
        valid=(confidence >= confidence_threshold) & (current | previous),
        current_supported=current,
        previous_supported=previous,
    )
    observation.validate()
    return observation


def _state_parity_max_abs(left: TaskMemoryState, right: TaskMemoryState) -> float:
    maximum = 0.0
    for left_tensor, right_tensor in zip(left.tensors(), right.tensors(), strict=True):
        if left_tensor.dtype == torch.bool or not left_tensor.is_floating_point():
            if not torch.equal(left_tensor.detach(), right_tensor.detach()):
                return float("inf")
            continue
        if left_tensor.numel():
            maximum = max(
                maximum,
                float(
                    (left_tensor.detach() - right_tensor.detach()).abs().max().item()
                ),
            )
    return maximum


@dataclass(frozen=True)
class TaskMemoryStageTransition:
    commit: CommitResult
    graph_state: TaskMemoryState
    runtime_state: TaskMemoryState
    parity_max_abs: float


def commit_task_memory_stage(
    *,
    observation: PredictionObservation,
    route: EntityRoute,
    state: TaskMemoryState,
    stage_meta: Sequence[StageMeta],
) -> TaskMemoryStageTransition:
    commit = commit_entities(observation, route, state, stage_meta)
    graph_state = commit.state
    runtime_state = graph_state.detach()
    parity = _state_parity_max_abs(graph_state, runtime_state)
    if parity != 0.0:
        raise TaskMemoryTrainerError("runtime and graph state values differ")
    return TaskMemoryStageTransition(
        commit=commit,
        graph_state=graph_state,
        runtime_state=runtime_state,
        parity_max_abs=parity,
    )


def bind_training_births(
    *,
    commit: CommitResult,
    ledgers: Sequence[TrainingIdentityLedger],
    assigned_indices: Sequence[IndexPair],
    identity_keys: Sequence[Sequence[QualifiedIdentity]],
    target_labels: Sequence[Tensor],
    ambiguity_metadata: Sequence[object],
) -> dict[str, int]:
    batch_size = commit.births.shape[0]
    if any(
        len(values) != batch_size
        for values in (
            ledgers,
            assigned_indices,
            identity_keys,
            target_labels,
            ambiguity_metadata,
        )
    ):
        raise TaskMemoryTrainerError("birth binding batch metadata is misaligned")
    totals = {
        "ambiguous_births_excluded": 0,
        "bound_births": 0,
        "ignored_births_excluded": 0,
        "unmatched_births": 0,
    }
    for batch_index in range(batch_size):
        diagnostics = ledgers[batch_index].bind_births(
            commit,
            batch_index=batch_index,
            assigned_indices=assigned_indices[batch_index],
            identity_keys=identity_keys[batch_index],
            target_labels=target_labels[batch_index],
            ambiguity_metadata=ambiguity_metadata[batch_index],
        )
        for name, value in diagnostics.items():
            totals[name] += value
    return totals


class TaskMemoryTrainer(InstanceSegmentation):
    """Optimize one effective episode batch with two-stage state gradients."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        settings = config.task_memory_training
        self.criterion = TaskMemoryCriterion(
            self.criterion,
            matcher_mode=str(settings.matcher_mode),
            post_conditioning_aux_start=int(settings.post_conditioning_aux_start),
            empty_current_weight=float(settings.empty_current_weight),
        )
        self.automatic_optimization = False
        self.progress = TaskMemoryProgress.initial()
        self._pending_task_memory_rng: Mapping[str, object] | None = None
        self.training_audit: list[dict[str, object]] = []
        self._smoke_previous_task_read: dict[str, Tensor] | None = None
        self.smoke_initialization_audit: dict[str, object] | None = None

    def begin_smoke_audit(self) -> dict[str, object]:
        if not bool(self.config.task_memory_training.state_enabled):
            raise TaskMemoryTrainerError("gradient smoke requires task memory")
        parameters = _task_read_parameters(self.named_parameters())
        output = {
            name: tensor
            for name, tensor in parameters.items()
            if name.startswith(f"{_TASK_ADAPTER_PREFIX}output_projection.")
        }
        if not parameters or not output:
            raise TaskMemoryTrainerError("gradient smoke task-read parameters are absent")
        self._smoke_previous_task_read = parameters
        self.smoke_initialization_audit = {
            "output_projection_zero": all(
                torch.count_nonzero(tensor).item() == 0 for tensor in output.values()
            ),
            "task_read_parameter_count": len(parameters),
        }
        return dict(self.smoke_initialization_audit)

    def setup(self, stage: str | None = None) -> None:
        if stage not in {None, "fit"}:
            raise TaskMemoryTrainerError("TaskMemoryTrainer supports fit only")

    def transfer_batch_to_device(
        self,
        batch: object,
        device: torch.device,
        dataloader_idx: int,
    ) -> object:
        if not isinstance(batch, TaskMemoryEpisodeBatch):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        stages = []
        for stage in batch.stage_batches:
            data, targets, names = stage.model_batch
            stages.append(
                TaskMemoryStageBatch(
                    model_batch=(
                        move_data_to_device(data, device),
                        move_data_to_device(targets, device),
                        names,
                    ),
                    stage_meta=stage.stage_meta,
                    training_identity_keys=stage.training_identity_keys,
                )
            )
        return TaskMemoryEpisodeBatch(
            specs=batch.specs,
            stage_batches=tuple(stages),
            ambiguity_metadata=batch.ambiguity_metadata,
        )

    def _backward_context(self, synchronize: bool):
        strategy = getattr(getattr(self, "trainer", None), "strategy", None)
        blocker = getattr(strategy, "block_backward_sync", None)
        return nullcontext() if synchronize or blocker is None else blocker()

    def _new_task_state(self, batch_size: int) -> TaskMemoryState:
        parameter = next(self.model.parameters())
        return TaskMemoryState.empty(
            batch_size=batch_size,
            capacity=int(self.model.task_memory_capacity),
            feature_dim=int(self.model.mask_dim),
            class_count=int(self.model.num_classes),
            device=parameter.device,
            dtype=parameter.dtype,
            config=self.model.task_memory_config,
        )

    def _forward_stage(
        self,
        stage: TaskMemoryStageBatch,
        state: TaskMemoryState | None,
    ) -> tuple[dict[str, object], list[dict[str, Any]]]:
        data, targets, _ = stage.model_batch
        data.device = self.device
        point2segment = [target["point2segment"] for target in targets]
        raw_coordinates = self._process_raw_coordinates(data)
        if bool(self.config.task_memory_training.state_enabled):
            output = self.model(
                data,
                point2segment,
                raw_coordinates=raw_coordinates,
                task_state=state,
                stage_meta=stage.stage_meta,
            )
        else:
            output = self.model(
                data,
                point2segment,
                raw_coordinates=raw_coordinates,
            )
        return output, targets

    def training_step(self, batch: TaskMemoryEpisodeBatch, batch_idx: int) -> Tensor:
        if not isinstance(batch, TaskMemoryEpisodeBatch):
            raise TaskMemoryTrainerError("trainer requires TaskMemoryEpisodeBatch")
        horizon = batch.specs[0].horizon
        coefficients = stage_mean_coefficients(horizon)
        settings = self.config.task_memory_training
        accumulation = int(settings.gradient_accumulation)
        if accumulation <= 0:
            raise TaskMemoryTrainerError("gradient accumulation must be positive")
        accumulation_index = self.progress.completed_local_episodes % accumulation
        should_step = accumulation_index == accumulation - 1
        optimizer = self.optimizers()
        if accumulation_index == 0:
            optimizer.zero_grad()

        state_enabled = bool(settings.state_enabled)
        state = self._new_task_state(len(batch.specs)) if state_enabled else None
        runtime_state = state
        ledgers = tuple(TrainingIdentityLedger() for _ in batch.specs)
        detached_episode_loss = torch.zeros((), device=self.device)
        route_calls = 0
        commit_calls = 0
        parity_max_abs = 0.0
        birth_totals = {
            "ambiguous_births_excluded": 0,
            "bound_births": 0,
            "ignored_births_excluded": 0,
            "unmatched_births": 0,
        }
        stage_logs: dict[str, Tensor] = {}
        smoke_audit = bool(getattr(settings, "smoke_audit", False))
        if smoke_audit and self._smoke_previous_task_read is None:
            raise TaskMemoryTrainerError("gradient smoke audit was not initialized")
        chunk_state_gradients = []
        chunk_boundary_detached = []
        stage_lineage = []
        chunks = training_stage_chunks(
            horizon,
            state_enabled=state_enabled,
            window_mode=str(settings.window_mode),
            tbptt_steps=int(settings.tbptt_steps),
        )

        for chunk_index, chunk in enumerate(chunks):
            chunk_loss = torch.zeros((), device=self.device)
            chunk_shadow: Tensor | None = None
            chunk_shadow_stage: int | None = None
            for stage_index in chunk:
                stage = batch.stage_batches[stage_index]
                output, targets = self._forward_stage(stage, state)
                route = output.get("task_memory_route") if state_enabled else None
                if state_enabled:
                    if not isinstance(route, EntityRoute) or state is None:
                        raise TaskMemoryTrainerError("task route is unavailable")
                    route_calls += 1
                result: TaskMemoryLossResult = self.criterion.compute_with_assignments(
                    output,
                    targets,
                    mask_type=self.mask_type,
                    route=route,
                    ledgers=ledgers if state_enabled else None,
                    identity_keys=(
                        stage.training_identity_keys if state_enabled else None
                    ),
                    ambiguity_metadata=(
                        batch.ambiguity_metadata if state_enabled else None
                    ),
                    stage_meta=stage.stage_meta if state_enabled else None,
                )
                stage_loss = _configured_objective_loss(self, result.losses)
                if not torch.isfinite(stage_loss).item():
                    raise TaskMemoryTrainerError("stage loss is non-finite")
                weighted = stage_loss * coefficients[stage_index]
                chunk_loss = chunk_loss + weighted
                detached_episode_loss = detached_episode_loss + weighted.detach()
                stage_logs[f"train_stage_{stage_index + 1}_loss"] = stage_loss.detach()

                if state_enabled:
                    observation = build_final_prediction_observation(
                        output,
                        stage.stage_meta,
                        background_class=int(self.model.task_background_class),
                        confidence_threshold=float(
                            self.model.task_confidence_threshold
                        ),
                        mask_threshold=float(self.model.task_mask_threshold),
                        minimum_mask_support=int(self.model.task_minimum_mask_support),
                    )
                    transition = commit_task_memory_stage(
                        observation=observation,
                        route=route,
                        state=state,
                        stage_meta=stage.stage_meta,
                    )
                    commit_calls += 1
                    state = transition.graph_state
                    runtime_state = transition.runtime_state
                    parity_max_abs = max(parity_max_abs, transition.parity_max_abs)
                    matched = route.query_to_slot >= 0
                    if smoke_audit:
                        stage_lineage.append(
                            {
                                "birth_count": int(
                                    transition.commit.births.sum().detach().item()
                                ),
                                "birth_logical_ids": sorted(
                                    set(
                                        transition.commit.query_to_logical_id[
                                            transition.commit.births
                                        ]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    )
                                ),
                                "inherited_logical_ids": sorted(
                                    set(
                                        route.prior_logical_id[matched]
                                        .detach()
                                        .cpu()
                                        .tolist()
                                    )
                                ),
                                "matched_query_count": int(
                                    matched.sum().detach().item()
                                ),
                                "stage": stage_index + 1,
                            }
                        )
                    if (
                        smoke_audit
                        and stage_index != chunk[-1]
                        and transition.graph_state.embedding.requires_grad
                    ):
                        transition.graph_state.embedding.retain_grad()
                        chunk_shadow = transition.graph_state.embedding
                        chunk_shadow_stage = stage_index + 1
                    bound = bind_training_births(
                        commit=transition.commit,
                        ledgers=ledgers,
                        assigned_indices=result.final_assignment.indices,
                        identity_keys=stage.training_identity_keys,
                        target_labels=tuple(target["labels"] for target in targets),
                        ambiguity_metadata=batch.ambiguity_metadata,
                    )
                    for name, value in bound.items():
                        birth_totals[name] += value

            final_chunk = chunk_index == len(chunks) - 1
            synchronize = should_step and final_chunk
            with self._backward_context(synchronize):
                connected = connect_all_trainable_parameters(
                    chunk_loss / accumulation,
                    self.named_parameters(),
                )
                self.manual_backward(connected)
            if smoke_audit and chunk_shadow is not None:
                gradient = chunk_shadow.grad
                chunk_state_gradients.append(
                    {
                        "chunk": chunk_index + 1,
                        "gradient_norm": 0.0
                        if gradient is None
                        else float(gradient.detach().float().norm().cpu().item()),
                        "source_stage": chunk_shadow_stage,
                    }
                )
            if state_enabled:
                if runtime_state is None:
                    raise TaskMemoryTrainerError("runtime state is unavailable")
                state = runtime_state
                if smoke_audit:
                    chunk_boundary_detached.append(
                        all(
                            tensor.grad_fn is None and not tensor.requires_grad
                            for tensor in state.tensors()
                        )
                    )

        if state_enabled and (route_calls != horizon or commit_calls != horizon):
            raise TaskMemoryTrainerError("route and commit must run once per stage")
        gradient_audit = None
        parameter_change = None
        if should_step and smoke_audit:
            gradient_audit = task_read_gradient_snapshot(self.named_parameters())
            if gradient_audit["nonfinite_gradient_names"]:
                raise TaskMemoryTrainerError("task-read gradient is non-finite")
        if should_step:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=float(self.config.trainer.gradient_clip_val),
                gradient_clip_algorithm="norm",
            )
            optimizer.step()
            scheduler = self.lr_schedulers()
            if scheduler is not None:
                scheduler.step()
            if smoke_audit:
                parameter_change, self._smoke_previous_task_read = (
                    _task_read_parameter_change(
                        self.named_parameters(), self._smoke_previous_task_read
                    )
                )

        world_size = int(self.trainer.world_size)
        global_rank = int(self.trainer.global_rank)
        self.progress = self.progress.advance(
            local_draw_indices=tuple(spec.draw_index for spec in batch.specs),
            horizon=horizon,
            world_size=world_size,
            global_rank=global_rank,
        )
        audit = {
            "births": dict(birth_totals),
            "commit_calls": commit_calls,
            "draw_cursor": self.progress.next_draw_index,
            "horizon": horizon,
            "optimizer_step": should_step,
            "route_calls": route_calls,
            "state_parity_max_abs": parity_max_abs,
        }
        if smoke_audit:
            audit.update(
                {
                    "chunk_boundary_state_detached": chunk_boundary_detached,
                    "chunk_state_gradients": chunk_state_gradients,
                    "stage_lineage": stage_lineage,
                    "task_read_gradient": gradient_audit,
                    "task_read_parameter_change": parameter_change,
                }
            )
        self.training_audit.append(audit)
        self.log_dict(
            {
                **stage_logs,
                "train_loss": detached_episode_loss,
                "train_episode_horizon": float(horizon),
                "train_global_episodes": float(self.progress.completed_global_episodes),
                "train_global_stages": float(self.progress.completed_global_stages),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=len(batch.specs),
        )
        return detached_episode_loss

    def configure_optimizers(self):
        settings = self.config.task_memory_training
        base_parameters = []
        adapter_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            (
                adapter_parameters
                if name.startswith(_TASK_ADAPTER_PREFIX)
                else base_parameters
            ).append(parameter)
        if not base_parameters:
            raise TaskMemoryTrainerError(
                "optimizer has no inherited trainable parameters"
            )
        if bool(settings.state_enabled) and not adapter_parameters:
            raise TaskMemoryTrainerError(
                "task-memory optimizer has no adapter parameters"
            )
        groups = [{"params": base_parameters, "lr": float(settings.inherited_lr)}]
        if adapter_parameters:
            groups.append(
                {"params": adapter_parameters, "lr": float(settings.adapter_lr)}
            )
        optimizer = torch.optim.AdamW(
            groups,
            betas=tuple(float(value) for value in settings.betas),
            eps=float(settings.eps),
            weight_decay=float(settings.weight_decay),
            amsgrad=bool(settings.amsgrad),
        )
        total_steps = int(settings.scheduler_total_updates)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: task_memory_lr_multiplier(
                min(step, total_steps - 1),
                total_steps=total_steps,
                warmup_fraction=float(settings.warmup_fraction),
                minimum_fraction=float(settings.minimum_lr_fraction),
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict[str, object]) -> None:
        if (
            not isinstance(checkpoint.get("optimizer_states"), list)
            or len(checkpoint["optimizer_states"]) != 1
            or not isinstance(checkpoint.get("lr_schedulers"), list)
            or len(checkpoint["lr_schedulers"]) != 1
        ):
            raise TaskMemoryTrainerError(
                "exact resume requires one optimizer and one scheduler state"
            )
        checkpoint["task_memory_resume"] = {
            "progress": self.progress.state_dict(),
            "rng": capture_task_memory_rng_state(),
            "schema_version": _RESUME_SCHEMA,
        }

    def on_load_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        if (
            not isinstance(checkpoint.get("optimizer_states"), list)
            or len(checkpoint["optimizer_states"]) != 1
            or not isinstance(checkpoint.get("lr_schedulers"), list)
            or len(checkpoint["lr_schedulers"]) != 1
        ):
            raise TaskMemoryTrainerError(
                "exact resume checkpoint lacks optimizer or scheduler state"
            )
        payload = checkpoint.get("task_memory_resume")
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"progress", "rng", "schema_version"}
            or payload["schema_version"] != _RESUME_SCHEMA
            or not isinstance(payload["progress"], Mapping)
            or not isinstance(payload["rng"], Mapping)
        ):
            raise TaskMemoryTrainerError("exact resume payload is invalid")
        self.progress = TaskMemoryProgress.from_state_dict(payload["progress"])
        _reset_lightning_batch_progress_for_sliced_resume(
            checkpoint,
            completed_local_episodes=self.progress.completed_local_episodes,
        )
        self._pending_task_memory_rng = payload["rng"]

    def on_train_start(self) -> None:
        if self._pending_task_memory_rng is not None:
            restore_task_memory_rng_state(self._pending_task_memory_rng)
            self._pending_task_memory_rng = None


__all__ = [
    "TaskMemoryProgress",
    "TaskMemoryStageTransition",
    "TaskMemoryTrainer",
    "TaskMemoryTrainerError",
    "bind_training_births",
    "build_final_prediction_observation",
    "capture_task_memory_rng_state",
    "commit_task_memory_stage",
    "restore_task_memory_rng_state",
    "stage_mean_coefficients",
    "task_memory_lr_multiplier",
    "task_read_gradient_snapshot",
    "tbptt_stage_chunks",
    "training_stage_chunks",
]
