from __future__ import annotations

import math

import pytest
import torch

from models.persistent_memory import LocalInstanceObservation, PersistentMemoryState
from models.persistent_memory_p6b import (
    P6BMemoryConfig,
    P6BPersistentMemory,
    threshold_aware_assignment,
)


def _observation(
    features: list[list[float]],
    class_prob: list[list[float]],
    *,
    confidence: list[float] | None = None,
    masks: list[list[float]] | None = None,
) -> LocalInstanceObservation:
    feature_tensor = torch.tensor([features], dtype=torch.float32)
    query_count = feature_tensor.shape[1]
    if confidence is None:
        confidence = [1.0] * query_count
    if masks is None:
        masks = [[1.0] * 4 for _ in range(query_count)]
    return LocalInstanceObservation(
        features=feature_tensor,
        class_prob=torch.tensor([class_prob], dtype=torch.float32),
        confidence=torch.tensor([confidence], dtype=torch.float32),
        latest_mask=[torch.tensor(masks, dtype=torch.float32)],
        valid=torch.ones((1, query_count), dtype=torch.bool),
    )


def _state(
    features: list[list[float]],
    class_prob: list[list[float]],
    *,
    active: list[bool],
    confidence: list[float] | None = None,
    capacity: int | None = None,
) -> PersistentMemoryState:
    occupied_count = len(features)
    capacity = occupied_count if capacity is None else capacity
    feature_dim = len(features[0])
    class_count = len(class_prob[0])
    state = PersistentMemoryState.empty(
        batch_size=1,
        capacity=capacity,
        feature_dim=feature_dim,
        class_count=class_count,
        device="cpu",
        dtype=torch.float32,
    )
    state.embedding[0, :occupied_count] = torch.tensor(features)
    state.class_prob[0, :occupied_count] = torch.tensor(class_prob)
    state.confidence[0, :occupied_count] = torch.tensor(
        confidence if confidence is not None else [1.0] * occupied_count
    )
    state.occupied[0, :occupied_count] = True
    state.active[0, :occupied_count] = torch.tensor(active)
    state.last_seen[0, :occupied_count] = 0
    state.stage_watermark[0] = 0
    state.validate()
    return state


def test_default_p6b_memory_config_is_valid_and_immutable() -> None:
    config = P6BMemoryConfig()

    assert config.active_threshold == 0.50
    assert config.reactivation_threshold == 0.85
    assert config.reactivation_threshold >= config.active_threshold
    assert config.assignment_mode == "threshold_aware"
    assert config.class_mode == "foreground_normalized"
    with pytest.raises(AttributeError):
        config.active_threshold = 0.75  # type: ignore[misc]


def test_reactivation_margin_can_be_disabled_explicitly() -> None:
    config = P6BMemoryConfig(reactivation_margin=None)

    assert config.reactivation_margin is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"capacity": 0}, "capacity"),
        ({"active_threshold": math.nan}, "active_threshold"),
        ({"reactivation_threshold": math.inf}, "reactivation_threshold"),
        (
            {"active_threshold": 0.75, "reactivation_threshold": 0.50},
            "reactivation_threshold",
        ),
        ({"reactivation_margin": -0.01}, "reactivation_margin"),
        ({"class_weight": 1.01}, "class_weight"),
        ({"class_mode": "raw"}, "class_mode"),
        ({"background_class": -1}, "background_class"),
        ({"update_rate": 0.0}, "update_rate"),
        ({"max_update_rate": 1.01}, "max_update_rate"),
        (
            {"update_rate": 0.50, "max_update_rate": 0.25},
            "max_update_rate",
        ),
        ({"consolidation_confidence": 1.01}, "consolidation_confidence"),
        ({"consolidation_margin": -0.01}, "consolidation_margin"),
        ({"birth_confidence": -0.01}, "birth_confidence"),
        ({"birth_minimum_mask_support": 0}, "birth_minimum_mask_support"),
        ({"birth_max_entropy": 1.01}, "birth_max_entropy"),
        ({"mask_threshold": math.nan}, "mask_threshold"),
        ({"assignment_mode": "post_threshold"}, "assignment_mode"),
    ],
)
def test_p6b_memory_config_rejects_invalid_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        P6BMemoryConfig(**overrides)


def test_threshold_aware_assignment_preserves_allowed_cardinality() -> None:
    score = torch.tensor(
        [[0.99, 0.74], [0.73, 0.49]],
        dtype=torch.float64,
    )

    pairs = threshold_aware_assignment(score, score >= 0.50)

    assert pairs == ((0, 1), (1, 0))


def test_threshold_aware_assignment_handles_rectangular_scores() -> None:
    score = torch.tensor(
        [[0.10, 0.80, 0.70], [0.90, 0.40, 0.20]],
        dtype=torch.float64,
    )
    allowed = torch.tensor(
        [[False, True, True], [True, False, False]],
        dtype=torch.bool,
    )

    assert threshold_aware_assignment(score, allowed) == ((0, 1), (1, 0))
    assert threshold_aware_assignment(score.T, allowed.T) == ((0, 1), (1, 0))


def test_threshold_aware_assignment_returns_empty_when_all_edges_forbidden() -> None:
    score = torch.ones((2, 3), dtype=torch.float32)

    assert threshold_aware_assignment(score, torch.zeros_like(score, dtype=torch.bool)) == ()
    assert threshold_aware_assignment(score[:0], torch.zeros_like(score[:0], dtype=torch.bool)) == ()


def test_threshold_aware_assignment_uses_stable_low_index_ties() -> None:
    score = torch.ones((3, 3), dtype=torch.float64)
    allowed = torch.ones_like(score, dtype=torch.bool)

    for _ in range(3):
        assert threshold_aware_assignment(score, allowed) == (
            (0, 0),
            (1, 1),
            (2, 2),
        )


def test_threshold_aware_assignment_never_selects_forbidden_high_scores() -> None:
    score = torch.tensor([[1000.0, 0.25], [0.50, 999.0]])
    allowed = torch.tensor([[False, True], [True, False]])

    assert threshold_aware_assignment(score, allowed) == ((0, 1), (1, 0))


@pytest.mark.parametrize(
    ("score", "allowed", "message"),
    [
        (torch.ones(2), torch.ones(2, dtype=torch.bool), "shape"),
        (torch.ones((2, 2)), torch.ones((1, 2), dtype=torch.bool), "shape"),
        (torch.ones((2, 2)), torch.ones((2, 2)), "bool"),
        (
            torch.tensor([[float("nan")]]),
            torch.ones((1, 1), dtype=torch.bool),
            "finite",
        ),
    ],
)
def test_threshold_aware_assignment_rejects_invalid_inputs(
    score: torch.Tensor, allowed: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        threshold_aware_assignment(score, allowed)


def test_dormant_slot_must_pass_reactivation_threshold() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=1,
            active_threshold=0.50,
            reactivation_threshold=0.85,
            class_weight=0.0,
            background_class=2,
            birth_confidence=1.0,
        )
    )
    state = _state([[1.0, 0.0]], [[1.0, 0.0, 0.0]], active=[False])
    observation = _observation(
        [[0.80, 0.60]],
        [[1.0, 0.0, 0.0]],
        confidence=[0.90],
    )

    result = memory.step(observation, state, stage_index=1)

    assert result.slot_ids.tolist() == [[-1]]
    assert result.reactivations.tolist() == [[False]]
    assert result.rejected_births.tolist() == [[True]]


def test_low_margin_dormant_edge_yields_to_allowed_active_edge() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=2,
            active_threshold=0.50,
            reactivation_threshold=0.85,
            reactivation_margin=0.10,
            class_weight=0.0,
            background_class=2,
        )
    )
    state = _state(
        [[0.95, math.sqrt(1.0 - 0.95**2)], [0.90, math.sqrt(1.0 - 0.90**2)]],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        active=[False, True],
    )
    observation = _observation([[1.0, 0.0]], [[1.0, 0.0, 0.0]])

    result = memory.step(observation, state, stage_index=1)

    assert result.slot_ids.tolist() == [[1]]
    assert result.reactivations.tolist() == [[False]]
    assert result.association_margins[0, 0].item() == pytest.approx(-0.05, abs=1e-6)


def test_accepted_dormant_match_is_reported_as_reactivation() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=1,
            active_threshold=0.50,
            reactivation_threshold=0.85,
            reactivation_margin=0.0,
            class_weight=0.0,
            background_class=2,
        )
    )
    state = _state([[1.0, 0.0]], [[1.0, 0.0, 0.0]], active=[False])

    result = memory.step(
        _observation([[1.0, 0.0]], [[1.0, 0.0, 0.0]]),
        state,
        stage_index=1,
    )

    assert result.slot_ids.tolist() == [[0]]
    assert result.reactivations.tolist() == [[True]]
    assert result.state.active.tolist() == [[True]]
    assert result.state.last_seen.tolist() == [[1]]


def test_low_confidence_match_updates_identity_without_consolidating() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=1,
            active_threshold=0.50,
            reactivation_threshold=0.85,
            reactivation_margin=0.0,
            class_weight=0.0,
            background_class=2,
            consolidation_confidence=0.90,
            consolidation_margin=None,
        )
    )
    state = _state(
        [[1.0, 0.0]],
        [[0.8, 0.1, 0.1]],
        active=[True],
        confidence=[0.95],
    )
    observation = _observation(
        [[0.8, 0.6]],
        [[0.2, 0.7, 0.1]],
        confidence=[0.80],
    )

    result = memory.step(observation, state, stage_index=1)

    assert result.slot_ids.tolist() == [[0]]
    assert result.consolidated.tolist() == [[False]]
    assert torch.equal(result.state.embedding, state.embedding)
    assert torch.equal(result.state.class_prob, state.class_prob)
    assert torch.equal(result.state.confidence, state.confidence)
    assert result.state.active.tolist() == [[True]]
    assert result.state.last_seen.tolist() == [[1]]


def test_foreground_normalized_class_compatibility_ignores_background_mass() -> None:
    state = _state(
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.001, 0.019, 0.98], [0.50, 0.00, 0.50]],
        active=[True, True],
    )
    observation = _observation([[0.0, 0.0]], [[0.01, 0.00, 0.99]])
    common = {
        "capacity": 2,
        "active_threshold": 0.0,
        "reactivation_threshold": 0.0,
        "reactivation_margin": 0.0,
        "class_weight": 1.0,
        "background_class": 2,
        "consolidation_confidence": None,
        "consolidation_margin": None,
    }

    full = P6BPersistentMemory(P6BMemoryConfig(**common, class_mode="full"))
    foreground = P6BPersistentMemory(
        P6BMemoryConfig(**common, class_mode="foreground_normalized")
    )

    assert full.step(observation, state, stage_index=1).slot_ids.tolist() == [[0]]
    assert foreground.step(observation, state, stage_index=1).slot_ids.tolist() == [[1]]


def test_low_support_birth_is_rejected_without_consuming_capacity() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=2,
            background_class=2,
            birth_confidence=0.75,
            birth_minimum_mask_support=2,
            birth_max_entropy=None,
        )
    )
    observation = _observation(
        [[1.0, 0.0]],
        [[0.9, 0.05, 0.05]],
        confidence=[0.99],
        masks=[[10.0, -10.0, -10.0]],
    )
    state = memory.empty_state(observation)

    result = memory.step(observation, state, stage_index=0)

    assert result.slot_ids.tolist() == [[-1]]
    assert result.rejected_births.tolist() == [[True]]
    assert result.birth_mask_support.tolist() == [[1]]
    assert not result.state.occupied.any().item()


def test_high_entropy_birth_is_rejected_without_consuming_capacity() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=1,
            background_class=2,
            birth_confidence=0.75,
            birth_minimum_mask_support=2,
            birth_max_entropy=0.50,
        )
    )
    observation = _observation(
        [[1.0, 0.0]],
        [[0.49, 0.49, 0.02]],
        confidence=[0.99],
        masks=[[10.0, 10.0, 10.0]],
    )
    state = memory.empty_state(observation)

    result = memory.step(observation, state, stage_index=0)

    assert result.slot_ids.tolist() == [[-1]]
    assert result.rejected_births.tolist() == [[True]]
    assert result.birth_entropy[0, 0].item() == pytest.approx(1.0)
    assert not result.state.occupied.any().item()


def test_high_quality_birth_is_allocated_to_first_free_slot() -> None:
    memory = P6BPersistentMemory(
        P6BMemoryConfig(
            capacity=2,
            background_class=2,
            birth_confidence=0.75,
            birth_minimum_mask_support=2,
            birth_max_entropy=0.50,
        )
    )
    observation = _observation(
        [[1.0, 0.0]],
        [[0.98, 0.01, 0.01]],
        confidence=[0.99],
        masks=[[10.0, 10.0, 10.0]],
    )

    result = memory.step(
        observation,
        memory.empty_state(observation),
        stage_index=0,
    )

    assert result.slot_ids.tolist() == [[0]]
    assert result.rejected_births.tolist() == [[False]]
    assert result.state.occupied.tolist() == [[True, False]]


def test_runtime_step_does_not_accept_ground_truth_fields() -> None:
    observation = _observation([[1.0, 0.0]], [[0.98, 0.01, 0.01]])
    memory = P6BPersistentMemory(
        P6BMemoryConfig(capacity=1, background_class=2)
    )

    with pytest.raises(TypeError, match="ground_truth"):
        memory.step(
            observation,
            memory.empty_state(observation),
            stage_index=0,
            ground_truth=torch.tensor([1]),  # type: ignore[call-arg]
        )
