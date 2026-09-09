from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
import torch

from datasets.task_memory_episode import StageMeta
from models.persistent_memory import LocalInstanceObservation, PersistentMemory
from models.task_memory_routing import (
    PredictionObservation,
    commit_entities,
    route_entities,
)
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState


def _meta(stage: int) -> StageMeta:
    scan_ids = ("scan-0",) if stage == 0 else (f"scan-{stage - 1}", f"scan-{stage}")
    point_count = len(scan_ids)
    indices = torch.arange(point_count, dtype=torch.long)
    offsets = torch.arange(point_count + 1, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=scan_ids,
        absolute_stage_index=stage,
        local_stage_ids=indices,
        original_vertex_ids=tuple(torch.tensor([0]) for _ in scan_ids),
        scan_vertex_offsets=offsets,
        point2segment=indices,
        segment_stage_ids=indices,
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=indices,
        full_resolution_point2segment=indices,
    )


def _observation(
    features: list[list[float]],
    *,
    confidence: list[float] | None = None,
    valid: list[bool] | None = None,
    current: list[bool] | None = None,
    previous: list[bool] | None = None,
    class_prob: list[list[float]] | None = None,
) -> PredictionObservation:
    query_count = len(features)
    confidence = confidence or [0.9] * query_count
    valid = valid or [True] * query_count
    current = current or [True] * query_count
    previous = previous or [False] * query_count
    class_prob = class_prob or [[0.5, 0.5] for _ in range(query_count)]
    return PredictionObservation(
        features=torch.tensor([features], dtype=torch.float32),
        class_prob=torch.tensor([class_prob], dtype=torch.float32),
        confidence=torch.tensor([confidence], dtype=torch.float32),
        valid=torch.tensor([valid], dtype=torch.bool),
        current_supported=torch.tensor([current], dtype=torch.bool),
        previous_supported=torch.tensor([previous], dtype=torch.bool),
    )


def _occupied_state(
    *,
    capacity: int = 2,
    embeddings: list[list[float]] | None = None,
    config: TaskMemoryConfig | None = None,
) -> TaskMemoryState:
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=capacity,
        feature_dim=2,
        class_count=2,
        device="cpu",
        dtype=torch.float32,
        config=config or TaskMemoryConfig(),
    )
    embeddings = embeddings or [[1.0, 0.0], [0.0, 1.0]][:capacity]
    occupied = torch.ones((1, capacity), dtype=torch.bool)
    result = replace(
        state,
        embedding=torch.tensor([embeddings], dtype=torch.float32),
        class_prob=torch.full((1, capacity, 2), 0.5),
        confidence=torch.full((1, capacity), 0.8),
        occupied=occupied,
        active=occupied.clone(),
        age=torch.arange(capacity, dtype=torch.long).unsqueeze(0),
        last_seen=torch.zeros((1, capacity), dtype=torch.long),
        stage_watermark=torch.tensor([0], dtype=torch.long),
        logical_ids=torch.arange(10, 10 + capacity, dtype=torch.long).unsqueeze(0),
        generations=torch.arange(2, 2 + capacity, dtype=torch.long).unsqueeze(0),
        next_logical_id=torch.tensor([10 + capacity], dtype=torch.long),
    )
    result.validate()
    return result


def test_route_is_read_only_and_filters_only_after_complete_assignment() -> None:
    state = _occupied_state(
        embeddings=[[1.0, 0.0], [0.8, 0.6]],
        config=TaskMemoryConfig(class_weight=0.0, association_threshold=0.85),
    )
    observation = _observation([[1.0, 0.0], [0.8, -0.6]])
    before = tuple(tensor.clone() for tensor in state.tensors())

    route = route_entities(observation, state, [_meta(1)])

    # Complete assignment selects the two 0.8 cross-edges, then thresholding
    # rejects both. Threshold-aware matching would incorrectly retain 1.0.
    assert route.query_to_slot.tolist() == [[-1, -1]]
    assert route.slot_to_query.tolist() == [[-1, -1]]
    assert torch.isneginf(route.route_score).all()
    assert all(torch.equal(old, new) for old, new in zip(before, state.tensors()))

    lower_threshold = replace(
        state,
        config=replace(state.config, association_threshold=0.75),
    )
    accepted = route_entities(observation, lower_threshold, [_meta(1)])
    assert accepted.query_to_slot.tolist() == [[1, 0]]
    assert accepted.slot_to_query.tolist() == [[1, 0]]
    assert accepted.route_score.tolist()[0] == pytest.approx([0.8, 0.8])
    assert accepted.prior_logical_id.tolist() == [[11, 10]]
    assert accepted.prior_generation.tolist() == [[3, 2]]


def test_previous_only_route_stays_dormant_and_preserves_identity() -> None:
    state = _occupied_state(capacity=1, embeddings=[[1.0, 0.0]])
    before_embedding = state.embedding.clone()
    observation = _observation(
        [[1.0, 0.0]], current=[False], previous=[True]
    )

    route = route_entities(observation, state, [_meta(1)])
    result = commit_entities(observation, route, state, [_meta(1)])

    assert route.current_supported.tolist() == [[False]]
    assert route.previous_supported.tolist() == [[True]]
    assert result.query_to_logical_id.tolist() == [[10]]
    assert result.query_to_generation.tolist() == [[2]]
    assert not result.state.active.any().item()
    assert result.state.last_seen.tolist() == [[0]]
    assert result.state.age.tolist() == [[1]]
    assert result.state.stage_watermark.tolist() == [1]
    assert torch.equal(result.state.embedding, before_embedding)


def test_default_commit_matches_b4_confidence_ema_without_reassociation() -> None:
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=2,
        feature_dim=2,
        class_count=2,
        device="cpu",
        dtype=torch.float32,
    )
    state = replace(
        state,
        embedding=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
        class_prob=torch.tensor([[[0.7, 0.3], [0.0, 0.0]]]),
        confidence=torch.tensor([[0.6, 0.0]]),
        occupied=torch.tensor([[True, False]]),
        active=torch.tensor([[True, False]]),
        age=torch.tensor([[4, 0]]),
        last_seen=torch.tensor([[0, -1]]),
        stage_watermark=torch.tensor([0]),
        logical_ids=torch.tensor([[7, -1]]),
        generations=torch.tensor([[4, -1]]),
        next_logical_id=torch.tensor([8]),
    )
    state.validate()
    observation = _observation(
        [[0.9, 0.1], [0.0, 1.0]],
        confidence=[0.8, 0.7],
        class_prob=[[0.6, 0.4], [0.2, 0.8]],
    )

    route = route_entities(observation, state, [_meta(1)])
    result = commit_entities(observation, route, state, [_meta(1)])

    old_memory = PersistentMemory(capacity=2)
    old_step = old_memory.step(
        LocalInstanceObservation(
            features=observation.features,
            class_prob=observation.class_prob,
            confidence=observation.confidence,
            latest_mask=[torch.ones((2, 1))],
            valid=observation.valid,
        ),
        state.as_persistent_state(),
        stage_index=1,
    )
    assert route.query_to_slot.tolist() == [[0, -1]]
    assert torch.equal(result.query_to_slot, old_step.slot_ids)
    assert torch.equal(route.route_score, old_step.association_scores)
    assert torch.equal(result.rejected_births, old_step.rejected_births)
    for actual, expected in zip(
        result.state.association_tensors(), old_step.state.tensors()
    ):
        assert torch.equal(actual, expected)
    assert result.query_to_logical_id.tolist() == [[7, 8]]
    assert result.query_to_generation.tolist() == [[4, 0]]


def test_commit_uses_frozen_route_even_when_final_features_prefer_another_slot() -> None:
    state = _occupied_state()
    pre_output = _observation([[1.0, 0.0]])
    route = route_entities(pre_output, state, [_meta(1)])
    assert route.query_to_slot.tolist() == [[0]]

    final_output = _observation([[0.0, 1.0]], confidence=[1.0])
    result = commit_entities(final_output, route, state, [_meta(1)])

    assert result.query_to_slot.tolist() == [[0]]
    assert result.state.last_seen.tolist() == [[1, 0]]
    assert result.state.active.tolist() == [[True, False]]
    assert result.state.embedding[0, 0, 1] > 0
    assert torch.equal(result.state.embedding[0, 1], state.embedding[0, 1])


def test_births_are_score_ordered_and_overflow_is_rejected_without_reuse() -> None:
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=2,
        feature_dim=2,
        class_count=2,
        device="cpu",
        dtype=torch.float32,
        config=TaskMemoryConfig(association_threshold=2.0),
    )
    observation = _observation(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        confidence=[0.5, 0.9, 0.9],
    )
    route = route_entities(observation, state, [_meta(0)])
    result = commit_entities(observation, route, state, [_meta(0)])

    assert result.births.tolist() == [[False, True, True]]
    assert result.rejected_births.tolist() == [[True, False, False]]
    assert result.query_to_slot.tolist() == [[-1, 0, 1]]
    assert result.query_to_logical_id.tolist() == [[-1, 0, 1]]
    assert result.state.logical_ids.tolist() == [[0, 1]]
    assert result.state.generations.tolist() == [[0, 0]]
    assert result.state.next_logical_id.tolist() == [2]

    next_observation = _observation([[1.0, -1.0]], confidence=[1.0])
    next_route = route_entities(next_observation, result.state, [_meta(1)])
    overflow = commit_entities(
        next_observation, next_route, result.state, [_meta(1)]
    )
    assert overflow.rejected_births.tolist() == [[True]]
    assert overflow.state.logical_ids.tolist() == [[0, 1]]
    assert overflow.state.generations.tolist() == [[0, 0]]
    assert overflow.state.next_logical_id.tolist() == [2]


def test_route_commitment_rejects_wrong_state_and_second_commit() -> None:
    state = _occupied_state(capacity=1, embeddings=[[1.0, 0.0]])
    observation = _observation([[1.0, 0.0]])
    route = route_entities(observation, state, [_meta(1)])
    result = commit_entities(observation, route, state, [_meta(1)])

    with pytest.raises(ValueError, match="commitment|state|watermark"):
        commit_entities(observation, route, result.state, [_meta(1)])
    with pytest.raises(ValueError, match="metadata|stage"):
        commit_entities(observation, route, state, [_meta(2)])


@pytest.mark.parametrize(
    ("mode", "expected"),
    (("last", [0.0, 1.0]), ("fixed_ema", [0.9701425, 0.2425356])),
)
def test_control_update_modes_are_fixed_and_prediction_only(
    mode: str, expected: list[float]
) -> None:
    state = _occupied_state(
        capacity=1,
        embeddings=[[1.0, 0.0]],
        config=TaskMemoryConfig(
            class_weight=0.0,
            association_threshold=-1.0,
            update_mode=mode,
            update_rate=0.2,
        ),
    )
    observation = _observation([[0.0, 1.0]], confidence=[0.3])
    route = route_entities(observation, state, [_meta(1)])
    result = commit_entities(observation, route, state, [_meta(1)])

    assert result.state.embedding[0, 0].tolist() == pytest.approx(expected)


def test_deployed_state_has_no_ground_truth_fields_and_stays_under_cap() -> None:
    forbidden = ("gt", "ground_truth", "target", "quality", "mask")
    field_names = tuple(field.name.lower() for field in dataclasses.fields(TaskMemoryState))
    assert not any(token in name for token in forbidden for name in field_names)
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=100,
        feature_dim=128,
        class_count=21,
        device="cpu",
        dtype=torch.float32,
    )
    assert state.state_bytes <= 2 * 1024**2
