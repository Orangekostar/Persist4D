"""P6-B tracker adapter for the frozen P6-A cached-observation protocol."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor

from models.persistent_memory import LocalInstanceObservation, PersistentMemoryState
from models.persistent_memory_p6b import (
    P6BMemoryConfig,
    P6BPersistentMemory,
    P6BStepResult,
)
from scripts.p6a_association import (
    AssociationDiagnostics,
    FrozenObservation,
    TrackStep,
    _as_p5_observation,
    _clone_persistent_state,
    _single_observation,
    freeze_observation,
)


@dataclass(frozen=True)
class P6BTransitionDetails:
    consolidated: tuple[bool, ...]
    reactivations: tuple[bool, ...]
    rejected_birth_confidence: tuple[bool, ...]
    rejected_birth_support: tuple[bool, ...]
    rejected_birth_entropy: tuple[bool, ...]
    rejected_birth_capacity: tuple[bool, ...]
    birth_mask_support: tuple[int, ...]
    birth_entropy: tuple[float, ...]

    def __post_init__(self) -> None:
        fields = (
            self.consolidated,
            self.reactivations,
            self.rejected_birth_confidence,
            self.rejected_birth_support,
            self.rejected_birth_entropy,
            self.rejected_birth_capacity,
            self.birth_mask_support,
            self.birth_entropy,
        )
        if not all(isinstance(field, tuple) for field in fields):
            raise ValueError("transition fields must be tuples")
        if len({len(field) for field in fields}) != 1:
            raise ValueError("transition fields must be query-aligned")
        for field in fields[:6]:
            if any(not isinstance(value, bool) for value in field):
                raise ValueError("transition flags must be booleans")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.birth_mask_support
        ):
            raise ValueError("birth_mask_support must contain nonnegative integers")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in self.birth_entropy
        ):
            raise ValueError("birth_entropy must contain finite values in [0, 1]")


def _tuple_bools(value: Tensor) -> tuple[bool, ...]:
    return tuple(bool(item) for item in value.detach().cpu().tolist())


def _transition_details(result: P6BStepResult) -> P6BTransitionDetails:
    return P6BTransitionDetails(
        consolidated=_tuple_bools(result.consolidated[0]),
        reactivations=_tuple_bools(result.reactivations[0]),
        rejected_birth_confidence=_tuple_bools(
            result.rejected_birth_confidence[0]
        ),
        rejected_birth_support=_tuple_bools(result.rejected_birth_support[0]),
        rejected_birth_entropy=_tuple_bools(result.rejected_birth_entropy[0]),
        rejected_birth_capacity=_tuple_bools(result.rejected_birth_capacity[0]),
        birth_mask_support=tuple(
            int(item) for item in result.birth_mask_support[0].detach().cpu().tolist()
        ),
        birth_entropy=tuple(
            float(item) for item in result.birth_entropy[0].detach().cpu().tolist()
        ),
    )


def _build_diagnostics(
    *,
    current: FrozenObservation,
    state_before: PersistentMemoryState,
    feature_score: Tensor,
    class_score: Tensor,
    total_score: Tensor,
    result: P6BStepResult,
) -> AssociationDiagnostics:
    query_count = current.query_count
    selected_identity: list[object | None] = [None] * query_count
    best_identity: list[object | None] = [None] * query_count
    chosen_feature: list[float | None] = [None] * query_count
    chosen_class: list[float | None] = [None] * query_count
    chosen_total: list[float | None] = [None] * query_count
    best_score: list[float | None] = [None] * query_count
    second_score: list[float | None] = [None] * query_count
    margin: list[float | None] = [None] * query_count
    slot_age: list[int | None] = [None] * query_count
    last_seen_stage: list[int | None] = [None] * query_count
    slot_active: list[bool | None] = [None] * query_count
    slot_occupied: list[bool | None] = [None] * query_count
    reactivation: list[bool | None] = [None] * query_count

    occupied_slots = state_before.occupied[0].nonzero(as_tuple=True)[0].tolist()
    occupied_before = state_before.occupied[0].detach().cpu()
    active_before = state_before.active[0].detach().cpu()
    age_before = state_before.age[0].detach().cpu()
    last_seen_before = state_before.last_seen[0].detach().cpu()
    slot_values = result.slot_ids[0].detach().cpu().tolist()
    selected_scores = result.association_scores[0].detach().cpu().tolist()
    selected_features = result.feature_scores[0].detach().cpu().tolist()
    selected_classes = result.class_scores[0].detach().cpu().tolist()
    selected_reactivations = result.reactivations[0].detach().cpu().tolist()

    for query_index in range(query_count):
        if not bool(current.valid[query_index]):
            continue
        ranked_slots = sorted(
            occupied_slots,
            key=lambda slot: (
                -float(total_score[slot, query_index].item()),
                int(slot),
            ),
        )
        best_slot: int | None = None
        if ranked_slots:
            best_slot = int(ranked_slots[0])
            best_value = float(total_score[best_slot, query_index].item())
            best_identity[query_index] = best_slot
            best_score[query_index] = best_value
            if len(ranked_slots) > 1:
                second_value = float(
                    total_score[ranked_slots[1], query_index].item()
                )
                second_score[query_index] = second_value
                margin[query_index] = best_value - second_value

        selected_slot: int | None = None
        slot = int(slot_values[query_index])
        selected_score = float(selected_scores[query_index])
        if (
            0 <= slot < state_before.capacity
            and bool(occupied_before[slot])
            and math.isfinite(selected_score)
        ):
            selected_slot = slot
            selected_identity[query_index] = slot
            chosen_feature[query_index] = float(selected_features[query_index])
            chosen_class[query_index] = float(selected_classes[query_index])
            chosen_total[query_index] = selected_score

        metadata_slot = selected_slot if selected_slot is not None else best_slot
        if metadata_slot is None:
            continue
        slot_age[query_index] = int(age_before[metadata_slot].item())
        last_seen_stage[query_index] = int(last_seen_before[metadata_slot].item())
        slot_active[query_index] = bool(active_before[metadata_slot].item())
        slot_occupied[query_index] = bool(occupied_before[metadata_slot].item())
        reactivation[query_index] = False
        if selected_slot is not None:
            reactivation[query_index] = bool(
                selected_reactivations[query_index]
            )

    return AssociationDiagnostics(
        selected_candidate_identity=tuple(selected_identity),
        best_candidate_identity=tuple(best_identity),
        chosen_feature_similarity=tuple(chosen_feature),
        chosen_class_similarity=tuple(chosen_class),
        chosen_total_score=tuple(chosen_total),
        best_score=tuple(best_score),
        second_best_score=tuple(second_score),
        score_margin=tuple(margin),
        slot_age=tuple(slot_age),
        last_seen_stage=tuple(last_seen_stage),
        slot_active=tuple(slot_active),
        slot_occupied=tuple(slot_occupied),
        reactivation=tuple(reactivation),
    )


class P6BTracker:
    method = "P6B"

    def __init__(
        self,
        *,
        sequence_id: str,
        config: P6BMemoryConfig | None = None,
    ) -> None:
        self.sequence_id = str(sequence_id)
        self.memory = P6BPersistentMemory(config)
        self._state: PersistentMemoryState | None = None
        self._last_stage: int | None = None
        self._previous_slot_to_query: dict[int, int] = {}
        self._last_transition: P6BTransitionDetails | None = None

    @property
    def state(self) -> PersistentMemoryState | None:
        return self._state

    @property
    def last_transition(self) -> P6BTransitionDetails | None:
        return self._last_transition

    def reset(self, *, sequence_id: str | None = None) -> None:
        if sequence_id is not None:
            self.sequence_id = str(sequence_id)
        self._state = None
        self._last_stage = None
        self._previous_slot_to_query = {}
        self._last_transition = None

    def step(
        self,
        observation: FrozenObservation
        | LocalInstanceObservation
        | Mapping[str, Any],
        *,
        stage_id: int,
    ) -> TrackStep:
        if (
            not isinstance(stage_id, int)
            or isinstance(stage_id, bool)
            or stage_id < 0
        ):
            raise ValueError("stage_id must be a nonnegative integer")
        if self._last_stage is not None and stage_id <= self._last_stage:
            raise ValueError("stage_id must increase for each tracker step")
        current = _single_observation(freeze_observation(observation))
        current.validate()
        local_observation = _as_p5_observation(current)
        state_before = self._state
        if state_before is None:
            state_before = self.memory.empty_state(local_observation)
        feature_score, class_score, total_score = self.memory._score_components(
            local_observation, state_before
        )
        result = self.memory.step(
            local_observation,
            state_before,
            stage_index=stage_id,
        )
        slot_values = result.slot_ids[0].detach().cpu().tolist()
        score_values = result.association_scores[0].detach().cpu().tolist()
        rejected_values = result.rejected_births[0].detach().cpu().tolist()
        occupied_before = state_before.occupied[0].detach().cpu()
        track_ids: list[object] = [None] * current.query_count
        matched_previous = [-1] * current.query_count
        scores: list[float | None] = [None] * current.query_count
        births = [False] * current.query_count
        current_slot_to_query: dict[int, int] = {}

        for query_index in range(current.query_count):
            if not bool(current.valid[query_index]):
                continue
            if bool(rejected_values[query_index]):
                continue
            slot = int(slot_values[query_index])
            if slot < 0:
                continue
            track_ids[query_index] = slot
            current_slot_to_query[slot] = query_index
            previous_query = self._previous_slot_to_query.get(slot)
            if previous_query is not None:
                matched_previous[query_index] = previous_query
            births[query_index] = not bool(occupied_before[slot])
            if not births[query_index]:
                score = float(score_values[query_index])
                if math.isfinite(score):
                    scores[query_index] = score

        self._state = result.state
        self._last_stage = stage_id
        self._previous_slot_to_query = current_slot_to_query
        self._last_transition = _transition_details(result)
        return TrackStep(
            method=self.method,
            sequence_id=self.sequence_id,
            stage_id=stage_id,
            track_ids=tuple(track_ids),
            matched_previous=tuple(matched_previous),
            scores=tuple(scores),
            births=tuple(births),
            valid=tuple(bool(value) for value in current.valid.tolist()),
            rejected_births=tuple(bool(value) for value in rejected_values),
            state_snapshot=_clone_persistent_state(result.state),
            diagnostics=_build_diagnostics(
                current=current,
                state_before=state_before,
                feature_score=feature_score[0],
                class_score=class_score[0],
                total_score=total_score[0],
                result=result,
            ),
        )
