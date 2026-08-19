from __future__ import annotations

import math

import pytest
import torch

from models.persistent_memory_p6b import (
    P6BMemoryConfig,
    threshold_aware_assignment,
)


def test_default_p6b_memory_config_is_valid_and_immutable() -> None:
    config = P6BMemoryConfig()

    assert config.active_threshold == 0.50
    assert config.reactivation_threshold == 0.85
    assert config.reactivation_threshold >= config.active_threshold
    assert config.assignment_mode == "threshold_aware"
    assert config.class_mode == "foreground_normalized"
    with pytest.raises(AttributeError):
        config.active_threshold = 0.75  # type: ignore[misc]


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
