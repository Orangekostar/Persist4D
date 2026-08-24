import pytest

from scripts.evaluate_rescan_persist4d import (
    RescanEvaluationError,
    select_rescan_inference_indices,
)


def test_local_pair_history_uses_only_previous_and_current_capture() -> None:
    indices = (10, 11, 12, 13)

    assert select_rescan_inference_indices(indices, 0, "local_pair") == (10,)
    assert select_rescan_inference_indices(indices, 2, "local_pair") == (11, 12)


def test_full_history_uses_every_capture_through_current_stage() -> None:
    indices = (10, 11, 12, 13)

    assert select_rescan_inference_indices(indices, 0, "full_history") == (10,)
    assert select_rescan_inference_indices(indices, 2, "full_history") == (10, 11, 12)


@pytest.mark.parametrize(
    ("indices", "stage_index", "history_strategy"),
    [
        ((), 0, "local_pair"),
        ((10,), -1, "local_pair"),
        ((10,), 1, "local_pair"),
        ((10,), True, "local_pair"),
        ((10,), 0, "unknown"),
    ],
)
def test_history_selection_fails_closed_on_invalid_contract(
    indices: tuple[int, ...], stage_index: int, history_strategy: str
) -> None:
    with pytest.raises(RescanEvaluationError):
        select_rescan_inference_indices(indices, stage_index, history_strategy)
