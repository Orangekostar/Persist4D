from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from models.persistent_memory import (
    PersistentMemory,
    PersistentMemoryState,
)
from models.streaming_rescene import StreamingReScene

_PERSISTENT_KEYS = {
    "persistent_slot_ids",
    "persistent_association_scores",
    "persistent_rejected_births",
}


def _settings() -> dict[str, object]:
    return {
        "background_class": 2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }


def _base_output(batch_size: int = 1) -> dict[str, object]:
    query_features = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]]
    ).repeat(batch_size, 1, 1)
    pred_logits = torch.tensor(
        [[[4.0, 0.0, -4.0], [0.0, 4.0, -4.0]]]
    ).repeat(batch_size, 1, 1)
    return {
        "pred_logits": pred_logits,
        "pred_changes": object(),
        "pred_masks": [
            torch.tensor([[10.0, 10.0], [10.0, 10.0]])
            for _ in range(batch_size)
        ],
        "aux_outputs": [object()],
        "sampled_coords": object(),
        "backbone_features": object(),
        "segment_features": [object()],
        "query_features": query_features,
    }


class _FakeReScene(nn.Module):
    def __init__(self, output: Mapping[str, object] | None = None) -> None:
        super().__init__()
        self.return_query_features = True
        self.output = dict(_base_output() if output is None else output)
        self.calls: list[tuple[object, object, object, bool]] = []

    def forward(
        self,
        x: object,
        point2segment: object = None,
        raw_coordinates: object = None,
        is_eval: bool = False,
    ) -> dict[str, object]:
        self.calls.append((x, point2segment, raw_coordinates, is_eval))
        return self.output


def _forward_step(
    wrapper: StreamingReScene,
    *,
    state: PersistentMemoryState | None = None,
    stage_index: int = 1,
    segment_stages: list[torch.Tensor] | object | None = None,
) -> tuple[dict[str, object], PersistentMemoryState]:
    if segment_stages is None:
        segment_stages = [torch.tensor([0, 1])]
    return wrapper.forward_step(
        x=object(),
        point2segment=[torch.tensor([0, 1])],
        raw_coordinates=None,
        segment_stages=segment_stages,
        state=state,
        stage_index=stage_index,
        is_eval=True,
    )


def test_init_registers_modules_and_requires_query_export() -> None:
    base_model = _FakeReScene()
    memory = PersistentMemory(capacity=3)

    wrapper = StreamingReScene(base_model, memory, _settings())

    assert wrapper.base_model is base_model
    assert wrapper.memory is memory
    assert dict(wrapper.named_children()) == {
        "base_model": base_model,
        "memory": memory,
    }

    base_model.return_query_features = False
    with pytest.raises(ValueError, match="return_query_features"):
        StreamingReScene(base_model, memory, _settings())


def test_init_rejects_invalid_module_types() -> None:
    class _NotAModule:
        return_query_features = True

    with pytest.raises(ValueError, match="nn.Module"):
        StreamingReScene(
            _NotAModule(),
            PersistentMemory(),
            _settings(),
        )

    with pytest.raises(ValueError, match="PersistentMemory"):
        StreamingReScene(_FakeReScene(), nn.Identity(), _settings())


@pytest.mark.parametrize("missing_key", sorted(_settings()))
def test_init_rejects_missing_observation_setting(missing_key: str) -> None:
    settings = _settings()
    del settings[missing_key]

    with pytest.raises(ValueError, match="exactly"):
        StreamingReScene(_FakeReScene(), PersistentMemory(), settings)


def test_init_rejects_unknown_observation_setting() -> None:
    settings = _settings()
    settings["unknown"] = 1

    with pytest.raises(ValueError, match="exactly"):
        StreamingReScene(_FakeReScene(), PersistentMemory(), settings)


def test_init_defensively_copies_observation_settings() -> None:
    settings = _settings()
    expected = settings.copy()
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        settings,
    )

    settings.clear()
    result, _ = _forward_step(wrapper)

    assert wrapper.observation_settings == expected
    assert wrapper.observation_settings is not settings
    assert _PERSISTENT_KEYS <= result.keys()


def test_forward_step_preserves_predictions_and_passes_base_arguments() -> None:
    base_model = _FakeReScene()
    original_keys = tuple(base_model.output)
    original_value_ids = {
        key: id(value) for key, value in base_model.output.items()
    }
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    x = object()
    point2segment = [torch.tensor([0, 1])]
    raw_coordinates = object()

    result, state = wrapper.forward_step(
        x=x,
        point2segment=point2segment,
        raw_coordinates=raw_coordinates,
        segment_stages=[torch.tensor([0, 1])],
        state=None,
        stage_index=7,
        is_eval=False,
    )

    assert base_model.calls == [(x, point2segment, raw_coordinates, False)]
    assert result is not base_model.output
    assert set(result) == set(original_keys) | _PERSISTENT_KEYS
    for key, value in base_model.output.items():
        assert result[key] is value
        assert id(value) == original_value_ids[key]
    assert tuple(base_model.output) == original_keys
    assert _PERSISTENT_KEYS.isdisjoint(base_model.output)
    assert torch.equal(result["persistent_slot_ids"], torch.tensor([[0, 1]]))
    assert torch.isneginf(result["persistent_association_scores"]).all()
    assert not torch.any(result["persistent_rejected_births"])
    assert torch.equal(
        state.occupied,
        torch.tensor([[True, True, False]]),
    )
    assert torch.equal(state.stage_watermark, torch.tensor([7]))
    assert not hasattr(wrapper, "state")
    assert not hasattr(wrapper, "outputs")
    assert not hasattr(wrapper, "observation")


def test_forward_step_exposes_rejected_births() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=1),
        _settings(),
    )

    result, _ = _forward_step(wrapper)

    assert torch.equal(
        result["persistent_rejected_births"],
        torch.tensor([[False, True]]),
    )


def test_forward_step_passes_state_explicitly_across_steps() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        _settings(),
    )
    first_result, first_state = _forward_step(wrapper, stage_index=4)

    second_result, second_state = _forward_step(
        wrapper,
        state=first_state,
        stage_index=5,
        segment_stages=[torch.tensor([1, 2])],
    )

    assert torch.equal(
        second_result["persistent_slot_ids"],
        first_result["persistent_slot_ids"],
    )
    assert second_state is not first_state
    assert torch.equal(first_state.stage_watermark, torch.tensor([4]))
    assert torch.equal(second_state.stage_watermark, torch.tensor([5]))
    assert torch.equal(first_state.age, torch.zeros_like(first_state.age))
    assert torch.equal(
        second_state.age,
        torch.tensor([[1, 1, 0]]),
    )


def test_forward_step_state_none_resets_memory() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        _settings(),
    )

    first_result, first_state = _forward_step(wrapper, stage_index=3)
    reset_result, reset_state = _forward_step(wrapper, stage_index=3)

    assert torch.equal(
        reset_result["persistent_slot_ids"],
        first_result["persistent_slot_ids"],
    )
    assert reset_state is not first_state
    assert torch.equal(reset_state.age, torch.zeros_like(reset_state.age))
    assert not hasattr(wrapper, "state")


def test_forward_step_rejects_state_batch_mismatch() -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )
    _, one_item_state = _forward_step(wrapper)
    base_model.output = _base_output(batch_size=2)

    with pytest.raises(ValueError, match="batch sizes"):
        _forward_step(
            wrapper,
            state=one_item_state,
            stage_index=2,
            segment_stages=[torch.tensor([1, 2]), torch.tensor([0, 2])],
        )


def test_forward_step_rejects_decreasing_global_stage() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(),
        PersistentMemory(capacity=3),
        _settings(),
    )
    _, state = _forward_step(wrapper, stage_index=4)

    with pytest.raises(ValueError, match="later than"):
        _forward_step(wrapper, state=state, stage_index=3)


def test_forward_step_rejects_missing_query_features() -> None:
    output = _base_output()
    del output["query_features"]
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="query_features"):
        _forward_step(wrapper)


@pytest.mark.parametrize("output_key", ["query_features", "pred_logits", "pred_masks"])
def test_forward_step_rejects_non_finite_base_outputs(output_key: str) -> None:
    output = _base_output()
    if output_key == "pred_masks":
        output[output_key][0][0, 0] = float("inf")
    else:
        output[output_key][0, 0, 0] = float("nan")
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="finite"):
        _forward_step(wrapper)


@pytest.mark.parametrize(
    "segment_stages",
    [
        (),
        [],
        [object()],
        [torch.tensor([])],
        [torch.tensor([[0, 1]])],
    ],
)
def test_forward_step_rejects_invalid_local_stage_collections(
    segment_stages: object,
) -> None:
    base_model = _FakeReScene()
    wrapper = StreamingReScene(
        base_model,
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="segment_stages"):
        _forward_step(wrapper, segment_stages=segment_stages)

    assert not base_model.calls


def test_forward_step_rejects_mixed_latest_local_stages() -> None:
    wrapper = StreamingReScene(
        _FakeReScene(_base_output(batch_size=2)),
        PersistentMemory(capacity=3),
        _settings(),
    )

    with pytest.raises(ValueError, match="latest local stage"):
        _forward_step(
            wrapper,
            segment_stages=[torch.tensor([0, 1]), torch.tensor([0, 2])],
        )


def test_forward_step_uses_maximum_as_latest_local_stage() -> None:
    output = _base_output()
    output["pred_masks"] = [
        torch.tensor([[10.0, 10.0], [-10.0, -10.0]])
    ]
    wrapper = StreamingReScene(
        _FakeReScene(output),
        PersistentMemory(capacity=3),
        _settings(),
    )

    result, _ = _forward_step(
        wrapper,
        segment_stages=[torch.tensor([2, 0])],
    )

    assert torch.equal(result["persistent_slot_ids"], torch.tensor([[0, 1]]))
