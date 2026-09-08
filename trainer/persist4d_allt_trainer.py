"""Sequence trainer for prediction-driven Persist4D All-T adaptation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch import Tensor, nn

from datasets.persist4d_sequence_dataset import Persist4DEpisodeBatch
from models.persistent_memory import (
    PersistentMemory,
    PersistentMemoryState,
    build_local_observation,
)
from models.persistent_memory_read import DetachedMemoryReadState
from trainer.trainer import InstanceSegmentation, _configured_objective_loss

_ADAPTER_PREFIXES = ("model.memory_read.", "model.local_enhancement.")


def connect_all_trainable_parameters(
    loss: Tensor,
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> Tensor:
    """Attach zero-valued gradients for conditionally unused DDP parameters."""
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise ValueError("connected loss must be a scalar tensor")
    connected = loss
    for _, parameter in named_parameters:
        if parameter.requires_grad and parameter.numel():
            connected = connected + parameter.reshape(-1)[0] * 0.0
    return connected


def adapter_gradient_snapshot(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> dict[str, object]:
    nonfinite = []
    frozen = []
    missing_adapter = []
    nonzero_adapter = []
    zero_adapter = []
    adapter_norms = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all().item():
            nonfinite.append(name)
        if not parameter.requires_grad and gradient is not None:
            frozen.append(name)
        if not name.startswith(_ADAPTER_PREFIXES):
            continue
        if gradient is None:
            missing_adapter.append(name)
            continue
        norm = float(gradient.detach().norm().item())
        adapter_norms[name] = norm
        (nonzero_adapter if norm > 0.0 else zero_adapter).append(name)
    return {
        "adapter_gradient_norms": dict(sorted(adapter_norms.items())),
        "frozen_gradient_names": sorted(frozen),
        "missing_adapter_gradients": sorted(missing_adapter),
        "nonfinite_gradient_names": sorted(nonfinite),
        "nonzero_adapter_gradients": sorted(nonzero_adapter),
        "zero_adapter_gradients": sorted(zero_adapter),
    }


def bounded_warmup_steps(total_steps: int, fraction: float) -> int:
    if total_steps < 2:
        raise ValueError("scheduler requires at least two optimizer steps")
    return min(total_steps - 1, max(1, round(total_steps * fraction)))


def prefix_balanced_stage_coefficients(horizon: int) -> tuple[float, ...]:
    """Return the frozen prefix-balanced coefficients for one episode."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 5:
        raise ValueError("horizon must be an integer in [1, 5]")
    if horizon == 1:
        return (1.0,)
    prefix_count = horizon - 1
    coefficients = tuple(
        sum(1.0 / prefix for prefix in range(max(2, stage), horizon + 1))
        / prefix_count
        for stage in range(1, horizon + 1)
    )
    if not math.isclose(sum(coefficients), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("prefix-balanced coefficients do not sum to one")
    return coefficients


def segment_stages_from_target(target: dict[str, Any]) -> Tensor:
    point2segment = target.get("point2segment")
    temporal_stages = target.get("temporal_stages")
    if (
        not isinstance(point2segment, Tensor)
        or not isinstance(temporal_stages, Tensor)
        or point2segment.ndim != 1
        or temporal_stages.ndim != 1
        or point2segment.shape != temporal_stages.shape
        or point2segment.numel() == 0
        or torch.any(point2segment < 0).item()
        or torch.any(temporal_stages < 0).item()
    ):
        raise ValueError("target point2segment and temporal_stages must align")
    segment_count = int(point2segment.max().item()) + 1
    maximum_stage = int(temporal_stages.max().item())
    minimum = torch.full(
        (segment_count,),
        maximum_stage + 1,
        dtype=temporal_stages.dtype,
        device=temporal_stages.device,
    )
    maximum = torch.full_like(minimum, -1)
    minimum.scatter_reduce_(
        0,
        point2segment.long(),
        temporal_stages,
        reduce="amin",
        include_self=True,
    )
    maximum.scatter_reduce_(
        0,
        point2segment.long(),
        temporal_stages,
        reduce="amax",
        include_self=True,
    )
    if torch.any(minimum != maximum).item():
        raise ValueError("one segment cannot span temporal stages")
    return minimum


def build_detached_memory_read_state(
    state: PersistentMemoryState,
) -> DetachedMemoryReadState:
    if not isinstance(state, PersistentMemoryState):
        raise TypeError("read state requires PersistentMemoryState")
    state.validate()
    return DetachedMemoryReadState(
        embeddings=state.embedding,
        occupied_mask=state.occupied,
        active_mask=state.active,
        confidence=state.confidence,
        last_seen=state.last_seen,
    )


def _detached_prediction_output(output: dict[str, object]) -> dict[str, object]:
    required = ("query_features", "pred_logits", "pred_masks")
    if any(key not in output for key in required):
        raise ValueError("prediction output lacks memory observation fields")
    masks = output["pred_masks"]
    if not isinstance(masks, list) or any(not isinstance(mask, Tensor) for mask in masks):
        raise ValueError("prediction masks must be a tensor list")
    features = output["query_features"]
    logits = output["pred_logits"]
    if not isinstance(features, Tensor) or not isinstance(logits, Tensor):
        raise TypeError("prediction features and logits must be tensors")
    return {
        "query_features": features.detach(),
        "pred_logits": logits.detach(),
        "pred_masks": [mask.detach() for mask in masks],
    }


def update_prediction_memory(
    *,
    output: dict[str, object],
    targets: list[dict[str, Any]],
    state: PersistentMemoryState | None,
    memory: PersistentMemory,
    stage_index: int,
    observation_settings: dict[str, object],
) -> tuple[PersistentMemoryState, DetachedMemoryReadState]:
    if not isinstance(targets, list) or not targets:
        raise ValueError("memory update requires a non-empty target list")
    segment_stages = [segment_stages_from_target(target) for target in targets]
    latest_stages = {int(stages.max().item()) for stages in segment_stages}
    if len(latest_stages) != 1:
        raise ValueError("one batch must share the latest local stage")
    observation = build_local_observation(
        _detached_prediction_output(output),
        segment_stages,
        latest_stage=latest_stages.pop(),
        **observation_settings,
    )
    if state is None:
        state = memory.empty_state(observation)
    next_state = memory.step(
        observation,
        state,
        stage_index=stage_index,
    ).state.detach()
    return next_state, build_detached_memory_read_state(next_state)


class Persist4DAllTTrainer(InstanceSegmentation):
    """Backpropagate one frozen-objective stage at a time, then update B4."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        persist4d = config.persist4d
        self.persistent_memory = PersistentMemory(
            capacity=persist4d.capacity,
            class_weight=persist4d.class_weight,
            association_threshold=persist4d.association_threshold,
            update_rate=persist4d.update_rate,
            max_update_rate=persist4d.max_update_rate,
        )
        self.observation_settings = {
            "background_class": int(persist4d.background_class),
            "confidence_threshold": float(persist4d.confidence_threshold),
            "mask_threshold": float(persist4d.mask_threshold),
            "minimum_mask_support": int(persist4d.minimum_mask_support),
        }
        self.automatic_optimization = False
        self.gradient_audit_steps: list[dict[str, object]] = []

    def on_before_optimizer_step(self, optimizer) -> None:
        super().on_before_optimizer_step(optimizer)
        self.gradient_audit_steps.append(
            adapter_gradient_snapshot(self.named_parameters())
        )

    def transfer_batch_to_device(
        self,
        batch: object,
        device: torch.device,
        dataloader_idx: int,
    ) -> object:
        if not isinstance(batch, Persist4DEpisodeBatch):
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        stage_batches = []
        for stage_batch in batch.stage_batches:
            if not isinstance(stage_batch, (tuple, list)) or len(stage_batch) != 3:
                raise ValueError("collated stage must be (data, targets, names)")
            data, targets, names = stage_batch
            stage_batches.append(
                (
                    move_data_to_device(data, device),
                    move_data_to_device(targets, device),
                    names,
                )
            )
        return Persist4DEpisodeBatch(
            specs=batch.specs,
            stage_batches=tuple(stage_batches),
            ambiguity_metadata=batch.ambiguity_metadata,
        )

    def forward(
        self,
        x: object,
        point2segment: object = None,
        raw_coordinates: object = None,
        is_eval: bool = False,
        *,
        memory_read_state: DetachedMemoryReadState | None = None,
    ) -> dict[str, object]:
        x.device = self.device
        return self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            memory_read_state=memory_read_state,
        )

    def _backward_context(self, synchronize: bool):
        strategy = getattr(getattr(self, "trainer", None), "strategy", None)
        blocker = getattr(strategy, "block_backward_sync", None)
        return nullcontext() if synchronize or blocker is None else blocker()

    def training_step(self, batch: Persist4DEpisodeBatch, batch_idx: int) -> Tensor:
        if not isinstance(batch, Persist4DEpisodeBatch):
            raise TypeError("All-T trainer requires Persist4DEpisodeBatch")
        horizons = {spec.horizon for spec in batch.specs}
        if len(horizons) != 1:
            raise ValueError("All-T batch horizons differ")
        horizon = horizons.pop()
        window_mode = str(self.config.allt_training.window_mode)
        if window_mode not in {"local_pair", "full_history"}:
            raise ValueError("All-T window mode is invalid")
        coefficients = prefix_balanced_stage_coefficients(horizon)
        accumulation = int(self.config.allt_training.gradient_accumulation)
        if accumulation <= 0:
            raise ValueError("gradient accumulation must be positive")
        accumulation_index = batch_idx % accumulation
        should_step = accumulation_index == accumulation - 1
        optimizer = self.optimizers()
        if accumulation_index == 0:
            optimizer.zero_grad()

        state: PersistentMemoryState | None = None
        read_state: DetachedMemoryReadState | None = None
        detached_episode_loss = torch.zeros((), device=self.device)
        stage_logs: dict[str, Tensor] = {}
        for stage_index, (coefficient, stage_batch) in enumerate(
            zip(coefficients, batch.stage_batches, strict=True)
        ):
            if not isinstance(stage_batch, (tuple, list)) or len(stage_batch) != 3:
                raise ValueError("collated stage must be (data, targets, names)")
            data, targets, _ = stage_batch
            if not isinstance(targets, list) or len(targets) != len(batch.specs):
                raise ValueError("stage targets differ from episode batch")
            synchronize = should_step and stage_index == horizon - 1
            with self._backward_context(synchronize):
                raw_coordinates = self._process_raw_coordinates(data)
                output = self.forward(
                    data,
                    point2segment=[target["point2segment"] for target in targets],
                    raw_coordinates=raw_coordinates,
                    memory_read_state=(
                        read_state
                        if getattr(self.model, "memory_read_enabled", False)
                        else None
                    ),
                )
                losses = self.criterion(output, targets, mask_type=self.mask_type)
                stage_loss = _configured_objective_loss(self, losses)
                if not torch.isfinite(stage_loss).item():
                    raise RuntimeError("All-T stage loss is non-finite")
                weighted_loss = connect_all_trainable_parameters(
                    stage_loss * coefficient,
                    self.named_parameters(),
                )
                self.manual_backward(weighted_loss / accumulation)
            detached_episode_loss = detached_episode_loss + weighted_loss.detach()
            stage_logs[f"train_stage_{stage_index + 1}_loss"] = stage_loss.detach()
            state, read_state = update_prediction_memory(
                output=output,
                targets=targets,
                state=state,
                memory=self.persistent_memory,
                stage_index=stage_index,
                observation_settings=self.observation_settings,
            )

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

        self.log_dict(
            {
                **stage_logs,
                "train_loss": detached_episode_loss,
                "train_episode_horizon": float(horizon),
                "train_supervised_stages": float(horizon),
                "train_encoder_scans": float(
                    1 + 2 * (horizon - 1)
                    if window_mode == "local_pair"
                    else horizon * (horizon + 1) // 2
                ),
            },
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=len(batch.specs),
        )
        return detached_episode_loss

    def configure_optimizers(self):
        settings = self.config.allt_training
        adapter_prefixes = ("model.memory_read.", "model.local_enhancement.")
        base_parameters = []
        adapter_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                adapter_parameters
                if name.startswith(adapter_prefixes)
                else base_parameters
            )
            target.append(parameter)
        if not base_parameters:
            raise RuntimeError("All-T optimizer has no inherited trainable parameters")
        parameter_groups = [
            {"params": base_parameters, "lr": float(settings.inherited_lr)}
        ]
        if adapter_parameters:
            parameter_groups.append(
                {"params": adapter_parameters, "lr": float(settings.adapter_lr)}
            )
        optimizer = torch.optim.AdamW(
            parameter_groups,
            betas=tuple(float(value) for value in settings.betas),
            eps=float(settings.eps),
            weight_decay=float(settings.weight_decay),
            amsgrad=bool(settings.amsgrad),
        )
        total_steps = int(settings.optimizer_updates)
        warmup_steps = bounded_warmup_steps(
            total_steps, float(settings.warmup_fraction)
        )
        minimum = float(settings.minimum_lr_fraction)
        if not 0 < minimum <= 1:
            raise ValueError("All-T scheduler settings are invalid")

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return max(1, step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum + (1.0 - minimum) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = [
    "Persist4DAllTTrainer",
    "adapter_gradient_snapshot",
    "bounded_warmup_steps",
    "build_detached_memory_read_state",
    "connect_all_trainable_parameters",
    "prefix_balanced_stage_coefficients",
    "segment_stages_from_target",
    "update_prediction_memory",
]
