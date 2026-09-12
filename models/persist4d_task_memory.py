"""ReScene with one proposal-anchored task-memory read."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from datasets.task_memory_episode import StageMeta
from models.object_visual_memory import ObjectVisualRead, ObjectVisualState
from models.persist4d_allt import (
    Persist4DModelError,
    strict_load_r1_with_named_adapters,
)
from models.rescene import ReScene
from models.task_memory_read import TaskMemoryRead
from models.task_memory_routing import (
    EntityRoute,
    PredictionObservation,
    route_entities,
)
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState


class TaskMemoryModelError(RuntimeError):
    """Raised when task-memory model construction or execution is unsafe."""


def strict_load_r1_task_memory(
    module: nn.Module, state_dict: Mapping[str, Tensor]
) -> dict[str, object]:
    """Load an R1 state while allowing only the named task-read parameters."""
    try:
        return strict_load_r1_with_named_adapters(
            module,
            state_dict,
            allowed_missing_prefixes=("model.task_read.", "model.visual_read."),
        )
    except Persist4DModelError as error:
        raise TaskMemoryModelError(str(error)) from error


def strict_load_task_memory_parent(
    module: nn.Module, state_dict: Mapping[str, Tensor]
) -> dict[str, object]:
    """Load an M2 task-memory parent, allowing only a new visual adapter."""
    try:
        return strict_load_r1_with_named_adapters(
            module,
            state_dict,
            allowed_missing_prefixes=("model.visual_read.",),
        )
    except Persist4DModelError as error:
        raise TaskMemoryModelError(str(error)) from error


def _finite_unit_interval(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise TaskMemoryModelError(f"{name} must be finite and within [0,1]")
    return float(value)


class Persist4DTaskMemory(ReScene):
    """Route R1 proposals to one historical entity before later decoder stages."""

    def __init__(
        self,
        *args: object,
        task_memory_enabled: bool = False,
        task_memory_capacity: int = 100,
        task_background_class: int = 18,
        task_confidence_threshold: float = 0.5,
        task_mask_threshold: float = 0.5,
        task_minimum_mask_support: int = 1,
        task_class_weight: float = 0.25,
        task_association_threshold: float = 0.5,
        task_update_rate: float = 0.2,
        task_max_update_rate: float = 0.2,
        task_visual_enabled: bool = False,
        task_visual_policy: str = "V-LAST",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if type(task_memory_enabled) is not bool:
            raise TaskMemoryModelError("task_memory_enabled must be a boolean")
        if type(task_visual_enabled) is not bool:
            raise TaskMemoryModelError("task_visual_enabled must be a boolean")
        if task_visual_enabled and not task_memory_enabled:
            raise TaskMemoryModelError("visual memory requires task memory")
        if task_visual_policy not in {"V-LAST", "V-CORE"}:
            raise TaskMemoryModelError("task_visual_policy must be V-LAST or V-CORE")
        if (
            isinstance(task_memory_capacity, bool)
            or not isinstance(task_memory_capacity, int)
            or task_memory_capacity != TaskMemoryRead.capacity
            or self.num_queries != TaskMemoryRead.num_queries
            or self.mask_dim != TaskMemoryRead.hidden_dim
        ):
            raise TaskMemoryModelError("task memory requires Q=K=100 and D=128")
        if (
            isinstance(task_background_class, bool)
            or not isinstance(task_background_class, int)
            or not 0 <= task_background_class < self.num_classes
        ):
            raise TaskMemoryModelError("task background class is invalid")
        if (
            isinstance(task_minimum_mask_support, bool)
            or not isinstance(task_minimum_mask_support, int)
            or task_minimum_mask_support <= 0
        ):
            raise TaskMemoryModelError("minimum mask support must be positive")

        self.task_memory_enabled = task_memory_enabled
        self.task_memory_capacity = task_memory_capacity
        self.task_background_class = task_background_class
        self.task_confidence_threshold = _finite_unit_interval(
            task_confidence_threshold, name="task_confidence_threshold"
        )
        self.task_mask_threshold = _finite_unit_interval(
            task_mask_threshold, name="task_mask_threshold"
        )
        self.task_minimum_mask_support = task_minimum_mask_support
        self.task_memory_config = TaskMemoryConfig(
            class_weight=task_class_weight,
            association_threshold=task_association_threshold,
            update_mode="confidence_ema",
            update_rate=task_update_rate,
            max_update_rate=task_max_update_rate,
        )
        self.task_read = TaskMemoryRead() if task_memory_enabled else None
        self.task_visual_enabled = task_visual_enabled
        self.task_visual_policy = task_visual_policy
        self.visual_read = ObjectVisualRead() if task_visual_enabled else None
        self._task_state: TaskMemoryState | None = None
        self._task_visual_state: ObjectVisualState | None = None
        self._task_stage_meta: tuple[StageMeta, ...] | None = None
        self._task_route: EntityRoute | None = None
        self._task_pre_observation: PredictionObservation | None = None
        self._task_read_call_count = 0
        self._task_visual_features: Tensor | None = None
        self._task_visual_padding_mask: Tensor | None = None

    def _build_pre_observation(
        self,
        queries: Tensor,
        class_logits: Tensor,
        mask_logits: Tensor,
        padding_mask: Tensor,
        stage_meta: Sequence[StageMeta],
    ) -> PredictionObservation:
        batch_size, query_count, _ = queries.shape
        if (
            class_logits.shape != (batch_size, query_count, self.num_classes)
            or mask_logits.ndim != 3
            or mask_logits.shape[:1] != (batch_size,)
            or mask_logits.shape[2] != query_count
            or padding_mask.shape != mask_logits.shape[:2]
            or padding_mask.dtype != torch.bool
            or len(stage_meta) != batch_size
        ):
            raise TaskMemoryModelError("decoder prediction context is misaligned")
        class_prob = class_logits.softmax(dim=-1)
        foreground = class_prob.clone()
        foreground[..., self.task_background_class] = -torch.inf
        confidence = foreground.amax(dim=-1)
        current_support = []
        previous_support = []
        for batch_index, meta in enumerate(stage_meta):
            if not isinstance(meta, StageMeta):
                raise TaskMemoryModelError("stage metadata entries must be StageMeta")
            unpadded = mask_logits[batch_index, ~padding_mask[batch_index]]
            segment_stages = meta.segment_stage_ids.to(device=mask_logits.device)
            if (
                segment_stages.ndim != 1
                or segment_stages.numel() != unpadded.shape[0]
                or segment_stages.numel() == 0
            ):
                raise TaskMemoryModelError("segment stage metadata is misaligned")
            latest_stage = int(segment_stages.max().item())
            positive = unpadded.sigmoid() >= self.task_mask_threshold
            current_support.append(
                positive[segment_stages == latest_stage]
                .sum(dim=0)
                .ge(self.task_minimum_mask_support)
            )
            previous_selector = segment_stages != latest_stage
            if previous_selector.any().item():
                previous_support.append(
                    positive[previous_selector]
                    .sum(dim=0)
                    .ge(self.task_minimum_mask_support)
                )
            else:
                previous_support.append(torch.zeros(query_count, dtype=torch.bool, device=queries.device))
        current = torch.stack(current_support)
        previous = torch.stack(previous_support)
        valid = (confidence >= self.task_confidence_threshold) & (current | previous)
        observation = PredictionObservation(
            features=self.decoder_norm(queries).detach().clone(),
            class_prob=class_prob.detach().clone(),
            confidence=confidence.detach().clone(),
            valid=valid.detach().clone(),
            current_supported=current.detach().clone(),
            previous_supported=previous.detach().clone(),
        )
        observation.validate()
        return observation

    def after_decoder_stage(
        self,
        queries: Tensor,
        *,
        execution_stage_idx: int,
        shared_parameter_idx: int,
        decoder_features: Tensor | None = None,
        decoder_padding_mask: Tensor | None = None,
        point2segment: Sequence[Tensor] | None = None,
    ) -> Tensor:
        queries = super().after_decoder_stage(
            queries,
            execution_stage_idx=execution_stage_idx,
            shared_parameter_idx=shared_parameter_idx,
            decoder_features=decoder_features,
            decoder_padding_mask=decoder_padding_mask,
            point2segment=point2segment,
        )
        if execution_stage_idx != len(self.hlevels) - 1 or not self.task_memory_enabled:
            return queries
        if self.task_read is None:
            raise TaskMemoryModelError("enabled task read module is missing")
        if decoder_features is None or decoder_padding_mask is None:
            raise TaskMemoryModelError("task routing prediction context is unavailable")
        if self._task_state is None or self._task_stage_meta is None:
            raise TaskMemoryModelError("task state and stage metadata are unavailable")

        class_logits, _, mask_logits = self.mask_module(queries, decoder_features)
        observation = self._build_pre_observation(
            queries,
            class_logits,
            mask_logits,
            decoder_padding_mask,
            self._task_stage_meta,
        )
        route = route_entities(observation, self._task_state, self._task_stage_meta)
        queries = self.task_read(queries, self._task_state, route)
        if getattr(self, "task_visual_enabled", False):
            if self.visual_read is None or self._task_visual_state is None:
                raise TaskMemoryModelError("enabled visual read requires visual state")
            queries = self.visual_read(
                queries,
                self._task_visual_state,
                route.query_to_slot,
                route.prior_generation,
            )
            self._task_visual_features = decoder_features.detach().clone()
            self._task_visual_padding_mask = decoder_padding_mask.detach().clone()
        self._task_pre_observation = observation
        self._task_route = route
        self._task_read_call_count += 1
        return queries

    def forward(
        self,
        x: object,
        point2segment: object = None,
        raw_coordinates: object = None,
        is_eval: bool = False,
        *,
        task_state: TaskMemoryState | None = None,
        task_visual_state: ObjectVisualState | None = None,
        stage_meta: Sequence[StageMeta] | None = None,
    ) -> dict[str, object]:
        if not self.task_memory_enabled:
            if task_state is not None or task_visual_state is not None:
                raise TaskMemoryModelError(
                    "task state was provided while task memory is disabled"
                )
            return super().forward(
                x,
                point2segment=point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
        if not isinstance(task_state, TaskMemoryState):
            raise TaskMemoryModelError("enabled task memory requires TaskMemoryState")
        if (
            isinstance(stage_meta, (str, bytes))
            or not isinstance(stage_meta, Sequence)
            or len(stage_meta) != task_state.batch_size
            or any(not isinstance(item, StageMeta) for item in stage_meta)
        ):
            raise TaskMemoryModelError("enabled task memory requires aligned StageMeta")
        task_state.validate()
        if getattr(self, "task_visual_enabled", False):
            if not isinstance(task_visual_state, ObjectVisualState):
                raise TaskMemoryModelError("enabled visual memory requires ObjectVisualState")
            task_visual_state.validate()
            if task_visual_state.batch_size != task_state.batch_size:
                raise TaskMemoryModelError("visual and task state batch sizes differ")
        elif task_visual_state is not None:
            raise TaskMemoryModelError("visual state was provided while visual memory is disabled")
        self._task_state = task_state
        self._task_visual_state = task_visual_state
        self._task_stage_meta = tuple(stage_meta)
        self._task_route = None
        self._task_pre_observation = None
        self._task_read_call_count = 0
        self._task_visual_features = None
        self._task_visual_padding_mask = None
        try:
            output = super().forward(
                x,
                point2segment=point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=is_eval,
            )
            if (
                self._task_read_call_count != 1
                or self.task_read is None
                or self._task_route is None
                or self._task_pre_observation is None
            ):
                raise TaskMemoryModelError("task route/read did not execute exactly once")
            route = self._task_route
            result = dict(output)
            result["task_memory_route"] = route
            result["task_memory_lineage"] = {
                "query_to_slot": route.query_to_slot.clone(),
                "prior_logical_id": route.prior_logical_id.clone(),
                "prior_generation": route.prior_generation.clone(),
                "current_supported": route.current_supported.clone(),
                "previous_supported": route.previous_supported.clone(),
            }
            result["task_memory_read_diagnostics"] = dict(
                self.task_read.last_diagnostics
            )
            if getattr(self, "task_visual_enabled", False):
                if self._task_visual_features is None or self._task_visual_padding_mask is None or self.visual_read is None:
                    raise TaskMemoryModelError("visual feature capture did not execute")
                result["task_memory_visual_features"] = self._task_visual_features
                result["task_memory_visual_padding_mask"] = self._task_visual_padding_mask
                result["task_memory_visual_read_diagnostics"] = dict(
                    self.visual_read.last_diagnostics
                )
            return result
        finally:
            self._task_state = None
            self._task_visual_state = None
            self._task_stage_meta = None
            self._task_route = None
            self._task_pre_observation = None
            self._task_visual_features = None
            self._task_visual_padding_mask = None


__all__ = [
    "Persist4DTaskMemory",
    "TaskMemoryModelError",
    "strict_load_r1_task_memory",
    "strict_load_task_memory_parent",
]
