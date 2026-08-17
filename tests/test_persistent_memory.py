from dataclasses import replace

import pytest
import torch

from models.persistent_memory import (
    LocalInstanceObservation,
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
