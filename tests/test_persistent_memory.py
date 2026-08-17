from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir

from models.persistent_memory import (
    LocalInstanceObservation,
    MemoryStepResult,
    PersistentMemory,
    PersistentMemoryState,
    build_local_observation,
)


def _valid_state() -> PersistentMemoryState:
    return PersistentMemoryState.empty(
        batch_size=2,
        capacity=3,
        feature_dim=4,
        class_count=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _valid_observation() -> LocalInstanceObservation:
    return LocalInstanceObservation(
        features=torch.zeros(2, 3, 4),
        class_prob=torch.zeros(2, 3, 5),
        confidence=torch.zeros(2, 3),
        latest_mask=[torch.zeros(3, 7), torch.zeros(3, 0)],
        valid=torch.ones(2, 3, dtype=torch.bool),
    )


def _step_observation(
    features: torch.Tensor,
    class_prob: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
) -> LocalInstanceObservation:
    batch_size, query_count = features.shape[:2]
    if confidence is None:
        confidence = torch.ones(
            batch_size,
            query_count,
            device=features.device,
            dtype=features.dtype,
        )
    if valid is None:
        valid = torch.ones(
            batch_size,
            query_count,
            device=features.device,
            dtype=torch.bool,
        )
    return LocalInstanceObservation(
        features=features,
        class_prob=class_prob,
        confidence=confidence,
        latest_mask=[
            torch.zeros(
                query_count,
                0,
                device=features.device,
                dtype=features.dtype,
            )
            for _ in range(batch_size)
        ],
        valid=valid,
    )


def _valid_builder_inputs() -> tuple[dict[str, object], list[torch.Tensor]]:
    outputs = {
        "query_features": torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        "pred_logits": torch.tensor(
            [
                [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
                [[2.0, 0.0, 1.0], [1.0, 0.0, 2.0]],
            ]
        ),
        "pred_masks": [
            torch.tensor(
                [
                    [10.0, 10.0],
                    [-10.0, 10.0],
                    [10.0, -10.0],
                ]
            ),
            torch.tensor(
                [
                    [10.0, -10.0],
                    [10.0, 10.0],
                    [-10.0, 10.0],
                    [10.0, 10.0],
                ]
            ),
        ],
    }
    segment_stages = [torch.tensor([1, 2, 2]), torch.tensor([2, 1, 2, 2])]
    return outputs, segment_stages


def test_memory_step_result_is_frozen() -> None:
    result = MemoryStepResult(
        state=_valid_state(),
        slot_ids=torch.tensor([[0]]),
        association_scores=torch.tensor([[1.0]]),
        rejected_births=torch.tensor([[False]]),
    )

    with pytest.raises(FrozenInstanceError):
        result.slot_ids = torch.tensor([[1]])


def test_persistent_memory_is_module_with_expected_defaults() -> None:
    memory = PersistentMemory()

    assert isinstance(memory, torch.nn.Module)
    assert memory.capacity == 100
    assert memory.class_weight == 0.25
    assert memory.association_threshold == 0.5
    assert memory.update_rate == 0.2
    assert memory.max_update_rate == 0.2


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("capacity", 0),
        ("capacity", -1),
        ("capacity", True),
        ("capacity", 1.0),
        ("capacity", None),
        ("class_weight", -0.01),
        ("class_weight", 1.01),
        ("class_weight", float("nan")),
        ("class_weight", float("inf")),
        ("class_weight", True),
        ("class_weight", 10**1000),
        ("association_threshold", float("nan")),
        ("association_threshold", float("-inf")),
        ("association_threshold", False),
        ("association_threshold", 10**1000),
        ("association_threshold", "0.5"),
        ("update_rate", -0.01),
        ("update_rate", 0.21),
        ("update_rate", float("nan")),
        ("update_rate", True),
        ("update_rate", 10**1000),
        ("max_update_rate", -0.01),
        ("max_update_rate", 0.19),
        ("max_update_rate", 1.01),
        ("max_update_rate", float("nan")),
        ("max_update_rate", False),
        ("max_update_rate", 10**1000),
    ],
)
def test_persistent_memory_rejects_invalid_constructor_parameters(
    parameter: str, value: object
) -> None:
    with pytest.raises(ValueError):
        PersistentMemory(**{parameter: value})


@pytest.mark.parametrize(
    "parameters",
    [
        {
            "capacity": 1,
            "class_weight": 0.0,
            "association_threshold": -1e300,
            "update_rate": 0.0,
            "max_update_rate": 0.0,
        },
        {
            "capacity": 1,
            "class_weight": 1.0,
            "association_threshold": 1e300,
            "update_rate": 1.0,
            "max_update_rate": 1.0,
        },
    ],
)
def test_persistent_memory_accepts_constructor_boundaries(
    parameters: dict[str, object],
) -> None:
    memory = PersistentMemory(**parameters)

    for name, value in parameters.items():
        assert getattr(memory, name) == value


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_empty_state_follows_observation_shape_device_and_dtype(
    dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    observation = _step_observation(
        torch.zeros(2, 3, 4, device=device, dtype=dtype),
        torch.zeros(2, 3, 5, device=device, dtype=dtype),
    )

    state = PersistentMemory(capacity=7).empty_state(observation)

    assert state.embedding.shape == (2, 7, 4)
    assert state.class_prob.shape == (2, 7, 5)
    assert state.confidence.shape == (2, 7)
    assert state.occupied.shape == (2, 7)
    assert all(tensor.device.type == device for tensor in state.tensors())
    assert state.embedding.dtype == dtype
    assert state.class_prob.dtype == dtype
    assert state.confidence.dtype == dtype
    assert not torch.any(state.occupied)
    assert not torch.any(state.active)
    assert torch.all(state.last_seen == -1)


def test_empty_state_validates_observation() -> None:
    observation = replace(
        _valid_observation(),
        valid=torch.ones(2, 3),
    )

    with pytest.raises(ValueError):
        PersistentMemory().empty_state(observation)


def test_empty_state_rejects_wrong_observation_type() -> None:
    with pytest.raises(ValueError):
        PersistentMemory().empty_state(None)


def test_step_births_in_query_order_into_lowest_free_slots() -> None:
    memory = PersistentMemory(capacity=4, association_threshold=2.0)
    observation = _step_observation(
        torch.tensor([[[0.0, 2.0], [5.0, 5.0], [2.0, 2.0]]]),
        torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.7, 0.3]]]),
        confidence=torch.tensor([[0.6, 0.4, 0.9]]),
        valid=torch.tensor([[True, False, True]]),
    )
    state = replace(
        memory.empty_state(observation),
        embedding=torch.tensor(
            [[[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]]
        ),
        class_prob=torch.tensor(
            [[[0.9, 0.1], [0.0, 0.0], [0.1, 0.9], [0.0, 0.0]]]
        ),
        confidence=torch.tensor([[0.8, 0.0, 0.7, 0.0]]),
        occupied=torch.tensor([[True, False, True, False]]),
        active=torch.tensor([[True, False, True, False]]),
        age=torch.tensor([[4, 0, 1, 0]]),
        last_seen=torch.tensor([[2, -1, 2, -1]]),
    )

    result = memory.step(observation, state, stage_index=3)

    assert torch.equal(result.slot_ids, torch.tensor([[1, -1, 3]]))
    assert torch.isneginf(result.association_scores).all()
    assert not torch.any(result.rejected_births)
    assert torch.equal(result.state.occupied, torch.ones(1, 4, dtype=torch.bool))
    assert torch.equal(
        result.state.active, torch.tensor([[False, True, False, True]])
    )
    assert torch.equal(result.state.age, torch.tensor([[5, 0, 2, 0]]))
    assert torch.equal(result.state.last_seen, torch.tensor([[2, 3, 2, 3]]))
    torch.testing.assert_close(result.state.embedding[0, 0], state.embedding[0, 0])
    torch.testing.assert_close(result.state.embedding[0, 2], state.embedding[0, 2])
    torch.testing.assert_close(result.state.embedding[0, 1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(
        result.state.embedding[0, 3], F.normalize(torch.tensor([2.0, 2.0]), dim=0)
    )
    torch.testing.assert_close(
        result.state.class_prob[0, [1, 3]], observation.class_prob[0, [0, 2]]
    )
    torch.testing.assert_close(
        result.state.confidence[0, [1, 3]], observation.confidence[0, [0, 2]]
    )


def test_step_match_preserves_slot_and_applies_confidence_scaled_ema() -> None:
    memory = PersistentMemory(
        capacity=2,
        class_weight=0.0,
        association_threshold=0.75,
        update_rate=0.2,
        max_update_rate=0.2,
    )
    birth = _step_observation(
        torch.tensor([[[2.0, 0.0]]]),
        torch.tensor([[[0.8, 0.2]]]),
        confidence=torch.tensor([[0.9]]),
    )
    source = memory.step(birth, memory.empty_state(birth), stage_index=0).state
    snapshots = tuple(tensor.clone() for tensor in source.tensors())
    observation = _step_observation(
        torch.tensor([[[0.8, 0.6]]]),
        torch.tensor([[[0.2, 0.8]]]),
        confidence=torch.tensor([[0.5]]),
    )

    result = memory.step(observation, source, stage_index=1)

    assert torch.equal(result.slot_ids, torch.tensor([[0]]))
    torch.testing.assert_close(result.association_scores, torch.tensor([[0.8]]))
    assert not torch.any(result.rejected_births)
    expected_embedding = F.normalize(torch.tensor([0.98, 0.06]), dim=0)
    torch.testing.assert_close(result.state.embedding[0, 0], expected_embedding)
    torch.testing.assert_close(
        result.state.class_prob[0, 0], torch.tensor([0.74, 0.26])
    )
    torch.testing.assert_close(result.state.confidence[0, 0], torch.tensor(0.86))
    assert result.state.active[0, 0]
    assert result.state.age[0, 0].item() == 1
    assert result.state.last_seen[0, 0].item() == 1
    for source_tensor, snapshot, result_tensor in zip(
        source.tensors(), snapshots, result.state.tensors(), strict=True
    ):
        torch.testing.assert_close(source_tensor, snapshot)
        assert result_tensor.data_ptr() != source_tensor.data_ptr()


@pytest.mark.parametrize(
    ("observation_confidence", "expected_rate"),
    [(-2.0, 0.0), (2.0, 0.2)],
)
def test_step_clamps_confidence_scaled_update_rate(
    observation_confidence: float, expected_rate: float
) -> None:
    memory = PersistentMemory(
        capacity=1,
        class_weight=0.0,
        association_threshold=0.7,
        update_rate=0.2,
        max_update_rate=0.2,
    )
    observation = _step_observation(
        torch.tensor([[[0.8, 0.6]]]),
        torch.tensor([[[0.0, 1.0]]]),
        confidence=torch.tensor([[observation_confidence]]),
    )
    state = PersistentMemoryState(
        embedding=torch.tensor([[[1.0, 0.0]]]),
        class_prob=torch.tensor([[[1.0, 0.0]]]),
        confidence=torch.tensor([[0.4]]),
        occupied=torch.tensor([[True]]),
        active=torch.tensor([[True]]),
        age=torch.tensor([[0]]),
        last_seen=torch.tensor([[0]]),
    )

    result = memory.step(observation, state, stage_index=1)

    expected_embedding = F.normalize(
        (1.0 - expected_rate) * state.embedding[0, 0]
        + expected_rate * observation.features[0, 0],
        dim=0,
    )
    torch.testing.assert_close(result.state.embedding[0, 0], expected_embedding)
    torch.testing.assert_close(
        result.state.class_prob[0, 0],
        torch.tensor([1.0 - expected_rate, expected_rate]),
    )
    expected_confidence = (
        (1.0 - expected_rate) * 0.4
        + expected_rate * observation_confidence
    )
    torch.testing.assert_close(
        result.state.confidence[0, 0], torch.tensor(expected_confidence)
    )


def test_step_keeps_unmatched_slot_dormant_and_reactivates_same_slot() -> None:
    memory = PersistentMemory(
        capacity=2,
        class_weight=0.0,
        association_threshold=0.9,
    )
    valid_observation = _step_observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.75, 0.25]]]),
        confidence=torch.tensor([[0.8]]),
    )
    born = memory.step(
        valid_observation,
        memory.empty_state(valid_observation),
        stage_index=0,
    )
    invalid_observation = replace(
        valid_observation, valid=torch.tensor([[False]])
    )

    dormant = memory.step(invalid_observation, born.state, stage_index=1)

    assert torch.equal(dormant.slot_ids, torch.tensor([[-1]]))
    assert torch.isneginf(dormant.association_scores).all()
    assert not torch.any(dormant.rejected_births)
    assert dormant.state.occupied[0, 0]
    assert not dormant.state.active[0, 0]
    assert dormant.state.age[0, 0].item() == 1
    assert dormant.state.last_seen[0, 0].item() == 0
    torch.testing.assert_close(dormant.state.embedding, born.state.embedding)
    torch.testing.assert_close(dormant.state.class_prob, born.state.class_prob)
    torch.testing.assert_close(dormant.state.confidence, born.state.confidence)

    reactivated = memory.step(valid_observation, dormant.state, stage_index=2)

    assert torch.equal(reactivated.slot_ids, torch.tensor([[0]]))
    assert reactivated.state.active[0, 0]
    assert reactivated.state.age[0, 0].item() == 2
    assert reactivated.state.last_seen[0, 0].item() == 2


def test_step_marks_only_valid_unmatched_queries_as_rejected_when_full() -> None:
    memory = PersistentMemory(capacity=1, association_threshold=2.0)
    birth = _step_observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    full_state = memory.step(
        birth, memory.empty_state(birth), stage_index=0
    ).state
    observation = _step_observation(
        torch.tensor([[[0.0, 1.0], [1.0, 1.0]]]),
        torch.tensor([[[0.0, 1.0], [0.5, 0.5]]]),
        valid=torch.tensor([[True, False]]),
    )

    result = memory.step(observation, full_state, stage_index=1)

    assert torch.equal(result.slot_ids, torch.tensor([[-1, -1]]))
    assert torch.isneginf(result.association_scores).all()
    assert torch.equal(result.rejected_births, torch.tensor([[True, False]]))
    assert result.state.occupied[0, 0]
    assert not result.state.active[0, 0]
    assert result.state.age[0, 0].item() == 1
    assert result.state.last_seen[0, 0].item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_step_preserves_batched_shapes_device_and_dtype(
    dtype: torch.dtype, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    observation = _step_observation(
        torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            device=device,
            dtype=dtype,
        ),
        torch.zeros(2, 2, 4, device=device, dtype=dtype),
        valid=torch.tensor(
            [[True, False], [True, True]], device=device, dtype=torch.bool
        ),
    )
    memory = PersistentMemory(capacity=2)

    result = memory.step(
        observation, memory.empty_state(observation), stage_index=0
    )

    assert torch.equal(
        result.slot_ids.cpu(), torch.tensor([[0, -1], [0, 1]])
    )
    assert result.slot_ids.shape == (2, 2)
    assert result.slot_ids.dtype == torch.long
    assert result.association_scores.shape == (2, 2)
    assert result.association_scores.dtype == dtype
    assert result.rejected_births.shape == (2, 2)
    assert result.rejected_births.dtype == torch.bool
    assert result.state.embedding.shape == (2, 2, 3)
    assert result.state.class_prob.shape == (2, 2, 4)
    assert all(tensor.device.type == device for tensor in result.state.tensors())
    assert result.slot_ids.device.type == device
    assert result.association_scores.device.type == device
    assert result.rejected_births.device.type == device


@pytest.mark.parametrize("mismatch", ["batch", "feature", "class", "dtype", "capacity"])
def test_step_rejects_state_observation_contract_mismatch(mismatch: str) -> None:
    memory = PersistentMemory(capacity=2)
    base_observation = _step_observation(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )
    state = memory.empty_state(base_observation)
    observation = base_observation
    if mismatch == "batch":
        observation = _step_observation(
            torch.zeros(2, 1, 2), torch.zeros(2, 1, 2)
        )
    elif mismatch == "feature":
        observation = _step_observation(
            torch.zeros(1, 1, 3), torch.zeros(1, 1, 2)
        )
    elif mismatch == "class":
        observation = _step_observation(
            torch.zeros(1, 1, 2), torch.zeros(1, 1, 3)
        )
    elif mismatch == "dtype":
        observation = _step_observation(
            torch.zeros(1, 1, 2, dtype=torch.float64),
            torch.zeros(1, 1, 2, dtype=torch.float64),
        )
    else:
        state = PersistentMemoryState.empty(
            batch_size=1,
            capacity=3,
            feature_dim=2,
            class_count=2,
            device="cpu",
            dtype=torch.float32,
        )

    with pytest.raises(ValueError):
        memory.step(observation, state, stage_index=0)


def test_step_rejects_device_mismatch() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    memory = PersistentMemory(capacity=1)
    cpu_observation = _step_observation(
        torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
    )
    cuda_observation = _step_observation(
        torch.zeros(1, 1, 2, device="cuda"),
        torch.zeros(1, 1, 2, device="cuda"),
    )

    with pytest.raises(ValueError, match="device"):
        memory.step(
            cuda_observation,
            memory.empty_state(cpu_observation),
            stage_index=0,
        )


@pytest.mark.parametrize("invalid_input", ["observation", "state"])
def test_step_validates_observation_and_state(invalid_input: str) -> None:
    memory = PersistentMemory(capacity=1)
    observation = _step_observation(
        torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
    )
    state = memory.empty_state(observation)
    if invalid_input == "observation":
        observation = replace(observation, valid=torch.ones(1, 1))
    else:
        state = replace(state, occupied=torch.zeros(1, 1, dtype=torch.long))

    with pytest.raises(ValueError):
        memory.step(observation, state, stage_index=0)


@pytest.mark.parametrize("stage_index", [-1, True, 0.0, None, 10**1000])
def test_step_rejects_invalid_stage_index(stage_index: object) -> None:
    observation = _step_observation(
        torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
    )
    memory = PersistentMemory(capacity=1)

    with pytest.raises(ValueError):
        memory.step(
            observation,
            memory.empty_state(observation),
            stage_index=stage_index,
        )


@pytest.mark.parametrize("stage_index", [4, 3, 0])
def test_step_requires_stage_later_than_every_occupied_slot(
    stage_index: int,
) -> None:
    observation = _step_observation(
        torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[1.0, 0.0]]])
    )
    memory = PersistentMemory(capacity=1)
    state = memory.step(
        observation, memory.empty_state(observation), stage_index=4
    ).state

    with pytest.raises(ValueError, match="later"):
        memory.step(observation, state, stage_index=stage_index)


def test_step_keeps_capacity_and_shapes_constant_for_one_hundred_stages() -> None:
    memory = PersistentMemory(capacity=4, association_threshold=2.0)
    observation = _step_observation(
        torch.tensor([[[1.0, 1.0]]]), torch.tensor([[[1.0, 0.0]]])
    )
    state = memory.empty_state(observation)
    expected_shapes = tuple(tensor.shape for tensor in state.tensors())

    for stage_index in range(100):
        observation = replace(
            observation,
            features=torch.tensor([[[float(stage_index + 1), 1.0]]]),
        )
        result = memory.step(observation, state, stage_index=stage_index)
        state = result.state
        assert tuple(tensor.shape for tensor in state.tensors()) == expected_shapes

    assert torch.all(state.occupied)
    assert torch.equal(state.last_seen, torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(result.slot_ids, torch.tensor([[-1]]))
    assert torch.equal(result.rejected_births, torch.tensor([[True]]))


def test_step_updates_remain_connected_to_autograd() -> None:
    state_embedding = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    state_class_prob = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    state_confidence = torch.tensor([[0.9]], requires_grad=True)
    state = PersistentMemoryState(
        embedding=state_embedding,
        class_prob=state_class_prob,
        confidence=state_confidence,
        occupied=torch.tensor([[True]]),
        active=torch.tensor([[True]]),
        age=torch.tensor([[0]]),
        last_seen=torch.tensor([[0]]),
    )
    observation_features = torch.tensor(
        [[[0.8, 0.6]]], requires_grad=True
    )
    observation_class_prob = torch.tensor(
        [[[0.2, 0.8]]], requires_grad=True
    )
    observation_confidence = torch.tensor([[0.5]], requires_grad=True)
    observation = _step_observation(
        observation_features,
        observation_class_prob,
        confidence=observation_confidence,
    )

    result = PersistentMemory(
        capacity=1,
        class_weight=0.0,
        association_threshold=0.75,
    ).step(observation, state, stage_index=1)
    loss = (
        result.state.embedding.sum()
        + result.state.class_prob.sum()
        + result.state.confidence.sum()
    )
    loss.backward()

    assert result.state.embedding.requires_grad
    assert result.state.class_prob.requires_grad
    assert result.state.confidence.requires_grad
    for tensor in (
        state_embedding,
        state_class_prob,
        state_confidence,
        observation_features,
        observation_class_prob,
        observation_confidence,
    ):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize("path", ["birth", "update"])
def test_step_normalizes_float16_zero_vectors_without_nan(path: str) -> None:
    if path == "birth":
        memory = PersistentMemory(capacity=1)
        observation = _step_observation(
            torch.zeros(1, 1, 2, dtype=torch.float16),
            torch.zeros(1, 1, 2, dtype=torch.float16),
        )
        state = memory.empty_state(observation)
        stage_index = 0
    else:
        memory = PersistentMemory(
            capacity=1,
            class_weight=0.0,
            association_threshold=-1.0,
            update_rate=0.5,
            max_update_rate=0.5,
        )
        observation = _step_observation(
            torch.tensor([[[-1.0, 0.0]]], dtype=torch.float16),
            torch.zeros(1, 1, 2, dtype=torch.float16),
        )
        state = PersistentMemoryState(
            embedding=torch.tensor([[[1.0, 0.0]]], dtype=torch.float16),
            class_prob=torch.zeros(1, 1, 2, dtype=torch.float16),
            confidence=torch.ones(1, 1, dtype=torch.float16),
            occupied=torch.tensor([[True]]),
            active=torch.tensor([[True]]),
            age=torch.tensor([[0]]),
            last_seen=torch.tensor([[0]]),
        )
        stage_index = 1

    result = memory.step(observation, state, stage_index=stage_index)

    assert torch.equal(
        result.state.embedding, torch.zeros(1, 1, 2, dtype=torch.float16)
    )
    assert torch.isfinite(result.state.embedding).all()


def test_persist4d_config_composes_into_persist4d_package() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "conf"
    expected = {
        "capacity": 100,
        "class_weight": 0.25,
        "association_threshold": 0.5,
        "update_rate": 0.2,
        "max_update_rate": 0.2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
        "background_class": 18,
    }

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        config = compose(config_name="model/persist4d")

    assert set(config) == {"persist4d"}
    assert dict(config.persist4d) == expected


def test_empty_state_has_expected_shapes_dtypes_and_sentinels() -> None:
    state = PersistentMemoryState.empty(
        batch_size=2,
        capacity=3,
        feature_dim=4,
        class_count=5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert state.embedding.shape == (2, 3, 4)
    assert state.class_prob.shape == (2, 3, 5)
    assert state.confidence.shape == (2, 3)
    assert state.occupied.shape == (2, 3)
    assert state.active.shape == (2, 3)
    assert state.age.shape == (2, 3)
    assert state.last_seen.shape == (2, 3)
    assert state.embedding.dtype == torch.float64
    assert state.class_prob.dtype == torch.float64
    assert state.confidence.dtype == torch.float64
    assert state.occupied.dtype == torch.bool
    assert state.active.dtype == torch.bool
    assert state.age.dtype == torch.long
    assert state.last_seen.dtype == torch.long
    assert torch.count_nonzero(state.embedding) == 0
    assert torch.count_nonzero(state.class_prob) == 0
    assert torch.count_nonzero(state.confidence) == 0
    assert not torch.any(state.occupied)
    assert not torch.any(state.active)
    assert torch.count_nonzero(state.age) == 0
    assert torch.all(state.last_seen == -1)
    assert state.batch_size == 2
    assert state.capacity == 3
    assert state.feature_dim == 4
    assert state.class_count == 5
    assert state.validate() is None


@pytest.mark.parametrize(
    ("dimension", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("capacity", 0),
        ("capacity", -1),
        ("feature_dim", 0),
        ("feature_dim", -1),
        ("class_count", 0),
        ("class_count", -1),
    ],
)
def test_empty_state_rejects_non_positive_dimensions(
    dimension: str, value: int
) -> None:
    dimensions = {
        "batch_size": 2,
        "capacity": 3,
        "feature_dim": 4,
        "class_count": 5,
    }
    dimensions[dimension] = value

    with pytest.raises(ValueError):
        PersistentMemoryState.empty(
            **dimensions,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.long, None, "float32"])
def test_empty_state_rejects_invalid_dtype(dtype: object) -> None:
    with pytest.raises(ValueError):
        PersistentMemoryState.empty(
            batch_size=2,
            capacity=3,
            feature_dim=4,
            class_count=5,
            device=torch.device("cpu"),
            dtype=dtype,
        )


def test_state_tensors_follow_field_order() -> None:
    state = _valid_state()

    assert state.tensors() == (
        state.embedding,
        state.class_prob,
        state.confidence,
        state.occupied,
        state.active,
        state.age,
        state.last_seen,
    )


@pytest.mark.parametrize(
    "property_name", ["batch_size", "capacity", "feature_dim", "class_count"]
)
@pytest.mark.parametrize("value", [None, torch.zeros(2, 3)])
def test_state_properties_reject_invalid_source_tensors(
    property_name: str, value: object
) -> None:
    source_field = "class_prob" if property_name == "class_count" else "embedding"
    state = replace(_valid_state(), **{source_field: value})

    with pytest.raises(ValueError):
        getattr(state, property_name)


def test_detach_returns_new_state_without_mutating_source() -> None:
    embedding = torch.randn(2, 3, 4, requires_grad=True)
    class_logits = torch.randn(2, 3, 5, requires_grad=True)
    confidence_logits = torch.randn(2, 3, requires_grad=True)
    state = PersistentMemoryState(
        embedding=embedding * 2,
        class_prob=class_logits.softmax(dim=-1),
        confidence=confidence_logits.sigmoid(),
        occupied=torch.tensor(
            [[True, True, False], [True, False, False]], dtype=torch.bool
        ),
        active=torch.tensor(
            [[True, False, False], [True, False, False]], dtype=torch.bool
        ),
        age=torch.tensor([[0, 2, 0], [1, 0, 0]], dtype=torch.long),
        last_seen=torch.tensor([[4, 1, -1], [3, -1, -1]], dtype=torch.long),
    )
    original_values = tuple(tensor.clone() for tensor in state.tensors())

    detached = state.detach()

    assert detached is not state
    for source, snapshot, result in zip(
        state.tensors(), original_values, detached.tensors(), strict=True
    ):
        torch.testing.assert_close(source, snapshot)
        torch.testing.assert_close(result, source)
        assert result is not source
        assert result.grad_fn is None
        assert not result.requires_grad
    assert state.embedding.grad_fn is not None
    assert state.class_prob.grad_fn is not None
    assert state.confidence.grad_fn is not None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding", torch.zeros(2, 3)),
        ("class_prob", torch.zeros(2, 3)),
        ("confidence", torch.zeros(2, 3, 1)),
        ("occupied", torch.zeros(2, 3, 1, dtype=torch.bool)),
        ("active", torch.zeros(2, 3, 1, dtype=torch.bool)),
        ("age", torch.zeros(2, 3, 1, dtype=torch.long)),
        ("last_seen", torch.zeros(2, 3, 1, dtype=torch.long)),
    ],
)
def test_state_validation_rejects_wrong_ranks(
    field: str, value: torch.Tensor
) -> None:
    state = replace(_valid_state(), **{field: value})

    with pytest.raises(ValueError):
        state.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_prob", torch.zeros(1, 3, 5)),
        ("confidence", torch.zeros(2, 4)),
        ("occupied", torch.zeros(1, 3, dtype=torch.bool)),
        ("active", torch.zeros(2, 4, dtype=torch.bool)),
        ("age", torch.zeros(1, 3, dtype=torch.long)),
        ("last_seen", torch.zeros(2, 4, dtype=torch.long)),
    ],
)
def test_state_validation_rejects_inconsistent_batch_or_capacity(
    field: str, value: torch.Tensor
) -> None:
    state = replace(_valid_state(), **{field: value})

    with pytest.raises(ValueError):
        state.validate()


@pytest.mark.parametrize("field", ["embedding", "class_prob", "confidence"])
def test_state_validation_rejects_non_finite_values(field: str) -> None:
    state = _valid_state()
    value = getattr(state, field).clone()
    value.view(-1)[0] = float("nan")

    with pytest.raises(ValueError):
        replace(state, **{field: value}).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding", torch.zeros(2, 3, 4, dtype=torch.long)),
        ("class_prob", torch.zeros(2, 3, 5, dtype=torch.float64)),
        ("confidence", torch.zeros(2, 3, dtype=torch.float64)),
        ("occupied", torch.zeros(2, 3, dtype=torch.long)),
        ("active", torch.zeros(2, 3, dtype=torch.long)),
        ("age", torch.zeros(2, 3, dtype=torch.int32)),
        ("last_seen", torch.zeros(2, 3, dtype=torch.int32)),
    ],
)
def test_state_validation_rejects_invalid_dtypes(
    field: str, value: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        replace(_valid_state(), **{field: value}).validate()


def test_state_validation_rejects_mixed_devices() -> None:
    class_prob = torch.zeros(2, 3, 5, device="meta")

    with pytest.raises(ValueError):
        replace(_valid_state(), class_prob=class_prob).validate()


def test_state_validation_rejects_active_unoccupied_slot() -> None:
    state = replace(
        _valid_state(), active=torch.tensor([[True, False, False]] * 2)
    )

    with pytest.raises(ValueError):
        state.validate()


def test_state_validation_rejects_negative_age() -> None:
    state = replace(
        _valid_state(), age=torch.tensor([[0, -1, 0], [0, 0, 0]])
    )

    with pytest.raises(ValueError):
        state.validate()


def test_state_validation_rejects_last_seen_below_sentinel() -> None:
    state = replace(
        _valid_state(), last_seen=torch.tensor([[-1, -2, -1], [-1, -1, -1]])
    )

    with pytest.raises(ValueError):
        state.validate()


def test_valid_observation_accepts_variable_and_empty_latest_masks() -> None:
    observation = _valid_observation()

    assert observation.validate() is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_prob", torch.zeros(2, 3, 5, device="meta")),
        ("confidence", torch.zeros(2, 3, device="meta")),
        ("valid", torch.ones(2, 3, dtype=torch.bool, device="meta")),
        (
            "latest_mask",
            [torch.zeros(3, 7), torch.zeros(3, 0, device="meta")],
        ),
    ],
)
def test_observation_validation_rejects_mixed_devices(
    field: str, value: object
) -> None:
    observation = replace(_valid_observation(), **{field: value})

    with pytest.raises(ValueError, match="same device"):
        observation.validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("features", torch.zeros(2, 3)),
        ("features", torch.zeros(2, 3, 0)),
        ("class_prob", torch.zeros(2, 3)),
        ("class_prob", torch.zeros(2, 3, 0)),
        ("class_prob", torch.zeros(1, 3, 5)),
        ("class_prob", torch.zeros(2, 4, 5)),
        ("confidence", torch.zeros(2, 3, 1)),
        ("confidence", torch.zeros(2, 4)),
        ("valid", torch.zeros(2, 3, 1, dtype=torch.bool)),
        ("valid", torch.zeros(2, 4, dtype=torch.bool)),
        ("valid", torch.zeros(2, 3)),
    ],
)
def test_observation_validation_rejects_invalid_tensor_contracts(
    field: str, value: torch.Tensor
) -> None:
    observation = replace(_valid_observation(), **{field: value})

    with pytest.raises(ValueError):
        observation.validate()


@pytest.mark.parametrize(
    "latest_mask",
    [
        [torch.zeros(3, 7)],
        [torch.zeros(3), torch.zeros(3, 7)],
        [torch.zeros(4, 7), torch.zeros(3, 7)],
        (torch.zeros(3, 7), torch.zeros(3, 7)),
    ],
)
def test_observation_validation_rejects_invalid_latest_masks(
    latest_mask: object,
) -> None:
    observation = replace(_valid_observation(), latest_mask=latest_mask)

    with pytest.raises(ValueError):
        observation.validate()


@pytest.mark.parametrize("field", ["features", "class_prob", "confidence"])
def test_observation_validation_rejects_non_finite_values(field: str) -> None:
    observation = _valid_observation()
    value = getattr(observation, field).clone()
    value.view(-1)[0] = float("inf")

    with pytest.raises(ValueError):
        replace(observation, **{field: value}).validate()


@pytest.mark.parametrize("field", ["features", "class_prob", "confidence"])
def test_observation_validation_rejects_non_floating_values(field: str) -> None:
    observation = _valid_observation()
    value = getattr(observation, field).long()

    with pytest.raises(ValueError, match="floating dtype"):
        replace(observation, **{field: value}).validate()


def test_build_local_observation_selects_latest_masks_and_filters_queries() -> None:
    outputs, segment_stages = _valid_builder_inputs()

    observation = build_local_observation(
        outputs,
        segment_stages,
        latest_stage=2,
        background_class=1,
        confidence_threshold=0.5,
        mask_threshold=0.5,
        minimum_mask_support=1,
    )

    assert observation.features is outputs["query_features"]
    torch.testing.assert_close(
        observation.class_prob,
        outputs["pred_logits"].softmax(dim=-1),
    )
    expected_confidence = observation.class_prob[:, :, [0, 2]].amax(dim=-1)
    torch.testing.assert_close(observation.confidence, expected_confidence)
    assert [mask.shape for mask in observation.latest_mask] == [(2, 2), (2, 3)]
    torch.testing.assert_close(
        observation.latest_mask[0],
        torch.tensor([[-10.0, 10.0], [10.0, -10.0]]),
    )
    assert torch.equal(
        observation.valid,
        torch.tensor([[True, False], [True, True]]),
    )
    assert observation.validate() is None


@pytest.mark.parametrize("missing_key", ["query_features", "pred_logits", "pred_masks"])
def test_build_local_observation_rejects_missing_output_key(
    missing_key: str,
) -> None:
    outputs, segment_stages = _valid_builder_inputs()
    del outputs[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query_features", torch.zeros(2, 2)),
        ("pred_logits", torch.zeros(2, 2)),
        ("pred_logits", torch.zeros(1, 2, 3)),
        ("pred_logits", torch.zeros(2, 3, 3)),
        ("pred_masks", (torch.zeros(3, 2), torch.zeros(4, 2))),
        ("pred_masks", [torch.zeros(3, 2)]),
    ],
)
def test_build_local_observation_rejects_invalid_output_shapes(
    field: str, value: object
) -> None:
    outputs, segment_stages = _valid_builder_inputs()
    outputs[field] = value

    with pytest.raises(ValueError):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize(
    ("mask", "stages"),
    [
        (torch.zeros(3, 2, 1), torch.tensor([1, 2, 2])),
        (torch.zeros(3, 3), torch.tensor([1, 2, 2])),
        (torch.zeros(3, 2), torch.tensor([[1, 2, 2]])),
        (torch.zeros(3, 2), torch.tensor([1, 2])),
    ],
)
def test_build_local_observation_rejects_invalid_mask_stage_shapes(
    mask: torch.Tensor, stages: torch.Tensor
) -> None:
    outputs, segment_stages = _valid_builder_inputs()
    outputs["pred_masks"][0] = mask
    segment_stages[0] = stages

    with pytest.raises(ValueError):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize(
    ("masks", "stages"),
    [
        ([torch.zeros(3, 2)], [torch.tensor([1, 2, 2])] * 2),
        (
            [torch.zeros(3, 2), torch.zeros(4, 2)],
            (torch.tensor([1, 2, 2]), torch.tensor([2, 1, 2, 2])),
        ),
    ],
)
def test_build_local_observation_rejects_invalid_batch_collections(
    masks: object, stages: object
) -> None:
    outputs, _ = _valid_builder_inputs()
    outputs["pred_masks"] = masks

    with pytest.raises(ValueError):
        build_local_observation(
            outputs,
            stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


def test_build_local_observation_rejects_absent_latest_stage() -> None:
    outputs, segment_stages = _valid_builder_inputs()
    segment_stages[1] = torch.tensor([0, 0, 1, 1])

    with pytest.raises(ValueError, match="latest_stage"):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize("background_class", [-1, 3, 1.0, True])
def test_build_local_observation_rejects_invalid_background_class(
    background_class: object,
) -> None:
    outputs, segment_stages = _valid_builder_inputs()

    with pytest.raises(ValueError, match="background_class"):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=background_class,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


def test_build_local_observation_requires_a_foreground_class() -> None:
    outputs, segment_stages = _valid_builder_inputs()
    outputs["pred_logits"] = torch.zeros(2, 2, 1)

    with pytest.raises(ValueError, match="at least two classes"):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=0,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize(
    ("source", "value"),
    [
        ("query_features", float("inf")),
        ("pred_logits", float("nan")),
        ("pred_masks", float("inf")),
        ("segment_stages", float("nan")),
    ],
)
def test_build_local_observation_rejects_non_finite_inputs(
    source: str, value: float
) -> None:
    outputs, segment_stages = _valid_builder_inputs()
    if source in {"query_features", "pred_logits"}:
        outputs[source][0, 0, 0] = value
    elif source == "pred_masks":
        outputs[source][0][0, 0] = value
    else:
        segment_stages[0] = segment_stages[0].float()
        segment_stages[0][0] = value

    with pytest.raises(ValueError, match="finite"):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("latest_stage", 2.0),
        ("latest_stage", True),
        ("confidence_threshold", -0.01),
        ("confidence_threshold", 1.01),
        ("confidence_threshold", float("nan")),
        ("confidence_threshold", 10**1000),
        ("confidence_threshold", True),
        ("mask_threshold", -0.01),
        ("mask_threshold", 1.01),
        ("mask_threshold", float("inf")),
        ("mask_threshold", 10**1000),
        ("mask_threshold", False),
        ("minimum_mask_support", 0),
        ("minimum_mask_support", -1),
        ("minimum_mask_support", 1.0),
        ("minimum_mask_support", True),
    ],
)
def test_build_local_observation_rejects_invalid_parameters(
    parameter: str, value: object
) -> None:
    outputs, segment_stages = _valid_builder_inputs()
    parameters = {
        "latest_stage": 2,
        "background_class": 1,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }
    parameters[parameter] = value

    with pytest.raises(ValueError):
        build_local_observation(outputs, segment_stages, **parameters)


@pytest.mark.parametrize(
    ("confidence_threshold", "mask_threshold"), [(0.0, 0.0), (1.0, 1.0)]
)
def test_build_local_observation_accepts_threshold_boundaries(
    confidence_threshold: float, mask_threshold: float
) -> None:
    outputs, segment_stages = _valid_builder_inputs()

    observation = build_local_observation(
        outputs,
        segment_stages,
        latest_stage=2,
        background_class=1,
        confidence_threshold=confidence_threshold,
        mask_threshold=mask_threshold,
        minimum_mask_support=1,
    )

    assert observation.valid.shape == (2, 2)


def test_build_local_observation_rejects_mixed_devices() -> None:
    outputs, segment_stages = _valid_builder_inputs()
    outputs["pred_masks"][1] = outputs["pred_masks"][1].to("meta")

    with pytest.raises(ValueError, match="same device"):
        build_local_observation(
            outputs,
            segment_stages,
            latest_stage=2,
            background_class=1,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            minimum_mask_support=1,
        )
