from __future__ import annotations

from collections.abc import Mapping

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
        stage_index: int,
        is_eval: bool = True,
    ) -> tuple[dict[str, object], PersistentMemoryState]:
        if not isinstance(segment_stages, list) or not segment_stages:
            raise ValueError(
                "segment_stages must be a non-empty list"
            )

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
            latest_local_stages.append(int(stages.max().item()))
        if len(set(latest_local_stages)) != 1:
            raise ValueError("all samples must share the latest local stage")

        outputs = self.base_model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
        )
        observation = build_local_observation(
            outputs,
            segment_stages,
            latest_stage=latest_local_stages[0],
            **self.observation_settings,
        )
        if state is None:
            state = self.memory.empty_state(observation)
        step = self.memory.step(observation, state, stage_index=stage_index)

        result = dict(outputs)
        result["persistent_slot_ids"] = step.slot_ids
        result["persistent_association_scores"] = step.association_scores
        result["persistent_rejected_births"] = step.rejected_births
        return result, step.state
