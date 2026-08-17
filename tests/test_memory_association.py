import random
import time
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import permutations

import pytest
import torch

from models import persistent_memory
from models.persistent_memory import (
    AssociationResult,
    LocalInstanceObservation,
    PersistentMemoryState,
    associate_observations,
)


def _observation(
    features: torch.Tensor,
    class_prob: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> LocalInstanceObservation:
    batch_size, query_count = features.shape[:2]
    if valid is None:
        valid = torch.ones(
            batch_size,
            query_count,
            dtype=torch.bool,
            device=features.device,
        )
    return LocalInstanceObservation(
        features=features,
        class_prob=class_prob,
        confidence=torch.ones(
            batch_size,
            query_count,
            dtype=features.dtype,
            device=features.device,
        ),
        latest_mask=[
            torch.zeros(
                query_count,
                0,
                dtype=features.dtype,
                device=features.device,
            )
            for _ in range(batch_size)
        ],
        valid=valid,
    )


def _state(
    embedding: torch.Tensor,
    class_prob: torch.Tensor,
    *,
    occupied: torch.Tensor | None = None,
) -> PersistentMemoryState:
    batch_size, capacity = embedding.shape[:2]
    if occupied is None:
        occupied = torch.ones(
            batch_size,
            capacity,
            dtype=torch.bool,
            device=embedding.device,
        )
    return PersistentMemoryState(
        embedding=embedding,
        class_prob=class_prob,
        confidence=torch.ones(
            batch_size,
            capacity,
            dtype=embedding.dtype,
            device=embedding.device,
        ),
        occupied=occupied,
        active=occupied.clone(),
        age=torch.zeros(
            batch_size,
            capacity,
            dtype=torch.long,
            device=embedding.device,
        ),
        last_seen=torch.where(
            occupied,
            torch.zeros_like(occupied, dtype=torch.long),
            torch.full_like(occupied, -1, dtype=torch.long),
        ),
    )


def _all_assignments(
    row_count: int,
    column_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    if row_count <= column_count:
        return [
            tuple(enumerate(columns))
            for columns in permutations(range(column_count), row_count)
        ]
    return [
        tuple((row, column) for column, row in enumerate(rows))
        for rows in permutations(range(row_count), column_count)
    ]


def _exact_assignment_score(
    score: torch.Tensor,
    assignment: tuple[tuple[int, int], ...],
) -> Fraction:
    return sum(
        (Fraction.from_float(score[row, column].item()) for row, column in assignment),
        start=Fraction(),
    )


def test_association_result_is_frozen() -> None:
    result = AssociationResult(
        slot_for_query=torch.tensor([[0]]),
        query_for_slot=torch.tensor([[0]]),
        score_for_query=torch.tensor([[1.0]]),
    )

    with pytest.raises(FrozenInstanceError):
        result.slot_for_query = torch.tensor([[1]])


def test_associate_observations_matches_exact_features_one_to_one() -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
    )
    observation = _observation(
        torch.tensor([[[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]]),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=0.25,
        association_threshold=1.0,
    )

    assert torch.equal(result.slot_for_query, torch.tensor([[1, 0, 2]]))
    assert torch.equal(result.query_for_slot, torch.tensor([[1, 0, 2]]))
    torch.testing.assert_close(
        result.score_for_query,
        torch.full((1, 3), 1.25),
    )
    assert result.slot_for_query.dtype == torch.long
    assert result.query_for_slot.dtype == torch.long
    assert result.slot_for_query.device == observation.features.device
    assert result.query_for_slot.device == observation.features.device
    assert result.score_for_query.device == observation.features.device
    assert result.slot_for_query.unique().numel() == 3
    assert result.query_for_slot.unique().numel() == 3


def test_associate_observations_uses_class_score_to_resolve_conflict() -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0], [0.8, 0.6]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[0.0, 1.0]]]),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=0.5,
        association_threshold=0.0,
    )

    assert result.slot_for_query.item() == 1
    assert torch.equal(result.query_for_slot, torch.tensor([[-1, 0]]))
    torch.testing.assert_close(result.score_for_query, torch.tensor([[1.3]]))


def test_associate_observations_rejects_assignment_below_threshold() -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=0.0,
        association_threshold=1.01,
    )

    assert torch.equal(result.slot_for_query, torch.tensor([[-1]]))
    assert torch.equal(result.query_for_slot, torch.tensor([[-1]]))
    assert torch.isneginf(result.score_for_query).all()


@pytest.mark.parametrize(
    ("occupied", "valid"),
    [
        (torch.tensor([[False, False]]), torch.tensor([[True, True, True]])),
        (torch.tensor([[True, True]]), torch.tensor([[False, False, False]])),
    ],
    ids=["empty-memory", "no-valid-query"],
)
def test_associate_observations_returns_sentinels_without_candidates(
    occupied: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        occupied=occupied,
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
        valid=valid,
    )

    result = associate_observations(
        observation,
        state,
        class_weight=0.0,
        association_threshold=-1.0,
    )

    assert torch.equal(result.slot_for_query, torch.full((1, 3), -1))
    assert torch.equal(result.query_for_slot, torch.full((1, 2), -1))
    assert torch.isneginf(result.score_for_query).all()


def test_associate_observations_handles_zero_feature_vectors_without_nan() -> None:
    state = _state(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )
    observation = _observation(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=1.0,
        association_threshold=0.0,
    )

    assert result.slot_for_query.item() == 0
    assert result.query_for_slot.item() == 0
    assert torch.equal(result.score_for_query, torch.zeros(1, 1))
    assert torch.isfinite(result.score_for_query).all()


def test_associate_observations_breaks_ties_stably_by_low_index_alignment() -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]),
    )
    expected_slot_for_query = torch.tensor([[0, 1, 2]])
    expected_query_for_slot = torch.tensor([[0, 1, 2, -1]])

    for _ in range(10):
        result = associate_observations(
            observation,
            state,
            class_weight=0.0,
            association_threshold=1.0,
        )

        assert torch.equal(result.slot_for_query, expected_slot_for_query)
        assert torch.equal(result.query_for_slot, expected_query_for_slot)
        torch.testing.assert_close(
            result.score_for_query,
            torch.ones(1, 3),
        )


def test_associate_observations_preserves_a_strict_float64_optimum() -> None:
    state = _state(
        torch.zeros(1, 3, 1, dtype=torch.float64),
        torch.tensor(
            [[[0.9999999999999999], [0.0], [1.0000000000000002]]],
            dtype=torch.float64,
        ),
    )
    observation = _observation(
        torch.zeros(1, 1, 1, dtype=torch.float64),
        torch.ones(1, 1, 1, dtype=torch.float64),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=1.0,
        association_threshold=0.0,
    )

    assert result.slot_for_query.item() == 2
    assert torch.equal(result.query_for_slot, torch.tensor([[-1, -1, 0]]))
    assert result.score_for_query.item() == 1.0000000000000002


def test_exact_assignment_preserves_a_competing_two_by_two_optimum() -> None:
    score = torch.tensor(
        [
            [0.9999999999999999, 1.0],
            [1.0, 1.0000000000000002],
        ],
        dtype=torch.float64,
    )

    rows, columns = persistent_memory._optimal_assignment_with_stable_ties(score)

    assert torch.equal(rows, torch.tensor([0, 1]))
    assert torch.equal(columns, torch.tensor([0, 1]))
    assert _exact_assignment_score(score, ((0, 0), (1, 1))) > (
        _exact_assignment_score(score, ((0, 1), (1, 0)))
    )


def test_exact_assignment_matches_exhaustive_random_small_matrices() -> None:
    random_generator = random.Random(2027)
    values = [
        -1.0,
        0.0,
        0.9999999999999999,
        1.0,
        1.0000000000000002,
        2.0,
    ]

    for _ in range(64):
        row_count = random_generator.randint(1, 4)
        column_count = random_generator.randint(1, 4)
        score = torch.tensor(
            [
                [random_generator.choice(values) for _ in range(column_count)]
                for _ in range(row_count)
            ],
            dtype=torch.float64,
        )
        candidates = _all_assignments(row_count, column_count)
        candidate_scores = [
            _exact_assignment_score(score, candidate) for candidate in candidates
        ]
        best_score = max(candidate_scores)
        best_alignment = min(
            sum(abs(row - column) for row, column in candidate)
            for candidate, candidate_score in zip(
                candidates, candidate_scores, strict=True
            )
            if candidate_score == best_score
        )

        rows, columns = persistent_memory._optimal_assignment_with_stable_ties(score)
        assignment = tuple(zip(rows.tolist(), columns.tolist(), strict=True))

        assert _exact_assignment_score(score, assignment) == best_score
        assert sum(abs(row - column) for row, column in assignment) == best_alignment
        for _ in range(2):
            repeated_rows, repeated_columns = (
                persistent_memory._optimal_assignment_with_stable_ties(score)
            )
            assert torch.equal(repeated_rows, rows)
            assert torch.equal(repeated_columns, columns)


@pytest.mark.parametrize(
    ("score", "expected_rows", "expected_columns"),
    [
        (
            torch.tensor([[0.0, 3.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        ),
        (
            torch.tensor([[0.0, 2.0], [3.0, 0.0], [0.0, 0.0]]),
            torch.tensor([0, 1]),
            torch.tensor([1, 0]),
        ),
    ],
    ids=["rows-fewer-than-columns", "rows-more-than-columns"],
)
def test_exact_assignment_maps_both_rectangular_directions(
    score: torch.Tensor,
    expected_rows: torch.Tensor,
    expected_columns: torch.Tensor,
) -> None:
    rows, columns = persistent_memory._optimal_assignment_with_stable_ties(score)

    assert torch.equal(rows, expected_rows)
    assert torch.equal(columns, expected_columns)


def test_exact_assignment_handles_k100_within_performance_budget() -> None:
    score = torch.rand(
        100,
        100,
        dtype=torch.float64,
        generator=torch.Generator().manual_seed(2027),
    )
    durations = []

    for _ in range(3):
        start = time.perf_counter()
        persistent_memory._optimal_assignment_with_stable_ties(score)
        durations.append(time.perf_counter() - start)

    assert min(durations) < 0.1


@pytest.mark.parametrize("invalid_input", ["observation", "state"])
def test_associate_observations_validates_each_input(invalid_input: str) -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    if invalid_input == "observation":
        observation = replace(
            observation,
            valid=torch.ones(1, 1),
        )
    else:
        state = replace(
            state,
            occupied=torch.ones(1, 1, dtype=torch.long),
        )

    with pytest.raises(ValueError):
        associate_observations(
            observation,
            state,
            class_weight=0.0,
            association_threshold=0.0,
        )


@pytest.mark.parametrize("mismatch", ["batch", "feature", "class"])
def test_associate_observations_rejects_dimension_mismatch(
    mismatch: str,
) -> None:
    observation_shape = {
        "batch": (2, 1, 2, 2),
        "feature": (1, 1, 3, 2),
        "class": (1, 1, 2, 3),
    }[mismatch]
    batch_size, query_count, feature_dim, class_count = observation_shape
    observation = _observation(
        torch.zeros(batch_size, query_count, feature_dim),
        torch.zeros(batch_size, query_count, class_count),
    )
    state = _state(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )

    with pytest.raises(ValueError):
        associate_observations(
            observation,
            state,
            class_weight=0.0,
            association_threshold=0.0,
        )


def test_associate_observations_rejects_device_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )
    state = _state(
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
    )
    state_on_meta = replace(
        state,
        **{
            name: tensor.to("meta")
            for name, tensor in zip(
                (
                    "embedding",
                    "class_prob",
                    "confidence",
                    "occupied",
                    "active",
                    "age",
                    "last_seen",
                ),
                state.tensors(),
                strict=True,
            )
        },
    )
    monkeypatch.setattr(PersistentMemoryState, "validate", lambda self: None)

    with pytest.raises(ValueError, match="device"):
        associate_observations(
            observation,
            state_on_meta,
            class_weight=0.0,
            association_threshold=0.0,
        )


@pytest.mark.parametrize(
    ("class_weight", "association_threshold"),
    [
        (-0.1, 0.0),
        (float("inf"), 0.0),
        (float("nan"), 0.0),
        (True, 0.0),
        (0.0, float("inf")),
        (0.0, float("-inf")),
        (0.0, float("nan")),
        (0.0, False),
    ],
)
def test_associate_observations_rejects_invalid_parameters(
    class_weight: float,
    association_threshold: float,
) -> None:
    state = _state(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0]]]),
    )

    with pytest.raises(ValueError):
        associate_observations(
            observation,
            state,
            class_weight=class_weight,
            association_threshold=association_threshold,
        )


@pytest.mark.parametrize(
    "class_weight",
    [1.0000000000000002, float(2**53), 1e39],
)
def test_associate_observations_rejects_class_weight_above_one(
    class_weight: float,
) -> None:
    state = _state(
        torch.tensor([[[1.0]]]),
        torch.tensor([[[1.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0]]]),
        torch.tensor([[[1.0]]]),
    )

    with pytest.raises(ValueError, match=r"class_weight.*\[0, 1\]"):
        associate_observations(
            observation,
            state,
            class_weight=class_weight,
            association_threshold=0.0,
        )


def test_associate_observations_accepts_class_weight_one() -> None:
    state = _state(
        torch.tensor([[[1.0]]]),
        torch.tensor([[[1.0]]]),
    )
    observation = _observation(
        torch.tensor([[[1.0]]]),
        torch.tensor([[[1.0]]]),
    )

    result = associate_observations(
        observation,
        state,
        class_weight=1.0,
        association_threshold=0.0,
    )

    assert result.slot_for_query.item() == 0
