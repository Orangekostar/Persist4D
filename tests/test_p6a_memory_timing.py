import pytest
import torch

from models.persistent_memory import (
    LocalInstanceObservation,
    MemoryStepResult,
    PersistentMemory,
    PersistentMemoryState,
)


def _observation() -> LocalInstanceObservation:
    return LocalInstanceObservation(
        features=torch.tensor([[[1.0, 0.0]]]),
        class_prob=torch.tensor([[[1.0, 0.0]]]),
        confidence=torch.tensor([[1.0]]),
        latest_mask=[torch.zeros(1, 0)],
        valid=torch.tensor([[True]]),
    )


def _assert_result_equal(
    actual: MemoryStepResult, expected: MemoryStepResult
) -> None:
    for actual_tensor, expected_tensor in zip(
        actual.state.tensors(), expected.state.tensors(), strict=True
    ):
        assert torch.equal(actual_tensor, expected_tensor)
    assert torch.equal(actual.slot_ids, expected.slot_ids)
    assert torch.equal(actual.association_scores, expected.association_scores)
    assert torch.equal(actual.rejected_births, expected.rejected_births)


def test_step_timing_sink_is_opt_in_and_preserves_result() -> None:
    observation = _observation()
    default_memory = PersistentMemory(capacity=1)
    timed_memory = PersistentMemory(capacity=1)

    default_result = default_memory.step(
        observation,
        default_memory.empty_state(observation),
        stage_index=0,
    )
    timing_events: list[dict[str, float]] = []
    timed_result = timed_memory.step(
        observation,
        timed_memory.empty_state(observation),
        stage_index=0,
        timing_sink=timing_events.append,
    )

    _assert_result_equal(timed_result, default_result)
    assert len(timing_events) == 1
    assert set(timing_events[0]) == {
        "association_overhead_ms",
        "memory_update_overhead_ms",
    }


def test_step_timing_uses_injected_clock_and_excludes_association() -> None:
    observation = _observation()
    memory = PersistentMemory(capacity=1)
    timing_events: list[dict[str, float]] = []
    clock_values = iter(
        (
            0,
            2_000_000,
            5_000_000,
            11_000_000,
        )
    )

    memory.step(
        observation,
        memory.empty_state(observation),
        stage_index=0,
        timing_sink=timing_events.append,
        clock_ns=lambda: next(clock_values),
    )

    assert timing_events == [
        {
            "association_overhead_ms": 3.0,
            "memory_update_overhead_ms": 8.0,
        }
    ]


def test_step_does_not_publish_timing_when_update_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    memory = PersistentMemory(capacity=1)
    state = memory.empty_state(observation)
    timing_events: list[dict[str, float]] = []
    original_validate = PersistentMemoryState.validate

    def fail_next_state_validation(candidate: PersistentMemoryState) -> None:
        if candidate is not state:
            raise RuntimeError("injected next-state validation failure")
        original_validate(candidate)

    monkeypatch.setattr(
        PersistentMemoryState,
        "validate",
        fail_next_state_validation,
    )

    with pytest.raises(
        RuntimeError, match="injected next-state validation failure"
    ):
        memory.step(
            observation,
            state,
            stage_index=0,
            timing_sink=timing_events.append,
            clock_ns=iter((0, 1_000_000, 2_000_000, 3_000_000)).__next__,
        )

    assert timing_events == []
