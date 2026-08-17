from __future__ import annotations

import math
import operator
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from models.persistent_memory import (
    PersistentMemory,
    PersistentMemoryState,
    build_local_observation,
)

_OBSERVATION_SETTING_KEYS = frozenset(
    {
        "background_class",
        "confidence_threshold",
        "mask_threshold",
        "minimum_mask_support",
    }
)


def _validate_observation_settings(settings: dict[str, object]) -> None:
    background_class = settings["background_class"]
    if (
        not isinstance(background_class, int)
        or isinstance(background_class, bool)
        or background_class < 0
    ):
        raise ValueError("background_class must be a non-negative integer")

    for name in ("confidence_threshold", "mask_threshold"):
        value = settings[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 <= value <= 1.0
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and within [0, 1]")

    minimum_mask_support = settings["minimum_mask_support"]
    if (
        not isinstance(minimum_mask_support, int)
        or isinstance(minimum_mask_support, bool)
        or minimum_mask_support <= 0
    ):
        raise ValueError("minimum_mask_support must be a positive integer")


def _latest_local_stage(segment_stages: list[Tensor]) -> int:
    latest_local_stages: list[int] = []
    for batch_index, stages in enumerate(segment_stages):
        if (
            not isinstance(stages, Tensor)
            or stages.ndim != 1
            or stages.numel() == 0
        ):
            raise ValueError(
                f"segment_stages[{batch_index}] must be a non-empty 1D tensor"
            )
        try:
            torch.iinfo(stages.dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError(
                f"segment_stages[{batch_index}] must use an integer dtype"
            ) from error
        if torch.any(stages < 0).item():
            raise ValueError(
                f"segment_stages[{batch_index}] must be non-negative"
            )
        latest_local_stages.append(stages.max().item())

    if len(set(latest_local_stages)) != 1:
        raise ValueError("all samples must share the latest local stage")
    return latest_local_stages[0]


def _validate_stage_index(stage_index: object) -> int:
    if isinstance(stage_index, bool):
        raise ValueError(  # noqa: TRY004
            "stage_index must be a non-negative integer"
        )
    try:
        validated_stage_index = operator.index(stage_index)
    except TypeError as error:
        raise ValueError(
            "stage_index must be a non-negative integer"
        ) from error
    if not 0 <= validated_stage_index <= torch.iinfo(torch.long).max:
        raise ValueError("stage_index must be a non-negative integer")
    return validated_stage_index


class StreamingReScene(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        memory: PersistentMemory,
        observation_settings: Mapping[str, object],
    ) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise ValueError("base_model must be an nn.Module")  # noqa: TRY004
        if getattr(base_model, "return_query_features", None) is not True:
            raise ValueError("base ReScene must enable return_query_features")
        if not isinstance(memory, PersistentMemory):
            raise ValueError(  # noqa: TRY004
                "memory must be a PersistentMemory"
            )
        if not isinstance(observation_settings, Mapping):
            raise ValueError(  # noqa: TRY004
                "observation_settings must be a mapping"
            )

        copied_settings = dict(observation_settings)
        if copied_settings.keys() != _OBSERVATION_SETTING_KEYS:
            raise ValueError(
                "observation_settings must contain exactly background_class, "
                "confidence_threshold, mask_threshold, and minimum_mask_support"
            )
        _validate_observation_settings(copied_settings)

        self.base_model = base_model
        self.memory = memory
        self.observation_settings = copied_settings

    def forward_step(
        self,
        *,
        x: object,
        point2segment: object,
        raw_coordinates: object,
        segment_stages: list[Tensor],
        state: PersistentMemoryState | None,
        stage_index: object,
        is_eval: bool = True,
    ) -> tuple[dict[str, object], PersistentMemoryState]:
        if not isinstance(segment_stages, list) or not segment_stages:
            raise ValueError(
                "segment_stages must be a non-empty list"
            )
        latest_local_stage = _latest_local_stage(segment_stages)
        validated_stage_index = _validate_stage_index(stage_index)

        try:
            point_batch_size = len(point2segment)
        except TypeError as error:
            raise ValueError(
                "point2segment must contain one item per batch sample"
            ) from error
        batch_size = len(segment_stages)
        if point_batch_size != batch_size:
            raise ValueError(
                "point2segment and segment_stages batch sizes must match"
            )

        if state is not None:
            if not isinstance(state, PersistentMemoryState):
                raise ValueError(
                    "state must be a PersistentMemoryState"
                )
            state.validate()
            if state.batch_size != batch_size:
                raise ValueError(
                    "state and segment_stages batch sizes must match"
                )
            if state.capacity != self.memory.capacity:
                raise ValueError(
                    "state capacity must match persistent memory capacity"
                )
            if validated_stage_index <= state.stage_watermark.max().item():
                raise ValueError(
                    "stage_index must be later than the processed-stage watermark"
                )

        outputs = self.base_model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
        )
        observation = build_local_observation(
            outputs,
            segment_stages,
            latest_stage=latest_local_stage,
            **self.observation_settings,
        )
        if state is None:
            state = self.memory.empty_state(observation)
        step = self.memory.step(
            observation,
            state,
            stage_index=validated_stage_index,
        )

        result = dict(outputs)
        result["persistent_slot_ids"] = step.slot_ids
        result["persistent_association_scores"] = step.association_scores
        result["persistent_rejected_births"] = step.rejected_births
        return result, step.state
