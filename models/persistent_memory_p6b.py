from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

_CLASS_MODES = frozenset({"full", "foreground_normalized"})
_ASSIGNMENT_MODES = frozenset({"legacy_post_threshold", "threshold_aware"})


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(  # noqa: TRY004
            f"{name} must be a finite number"
        )
    try:
        finite = math.isfinite(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _unit_interval(value: object, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def _optional_unit_interval(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _unit_interval(value, name)


def _optional_nonnegative(value: object, name: str) -> float | None:
    if value is None:
        return None
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


@dataclass(frozen=True)
class P6BMemoryConfig:
    capacity: int = 100
    active_threshold: float = 0.50
    reactivation_threshold: float = 0.85
    reactivation_margin: float = 0.10
    class_weight: float = 0.25
    class_mode: str = "foreground_normalized"
    background_class: int = 18
    update_rate: float = 0.20
    max_update_rate: float = 0.20
    consolidation_confidence: float | None = 0.90
    consolidation_margin: float | None = 0.10
    birth_confidence: float = 0.75
    birth_minimum_mask_support: int = 128
    birth_max_entropy: float | None = 0.50
    mask_threshold: float = 0.50
    assignment_mode: str = "threshold_aware"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.capacity, int)
            or isinstance(self.capacity, bool)
            or self.capacity <= 0
        ):
            raise ValueError("capacity must be a positive integer")
        active_threshold = _finite_number(
            self.active_threshold, "active_threshold"
        )
        reactivation_threshold = _finite_number(
            self.reactivation_threshold, "reactivation_threshold"
        )
        if reactivation_threshold < active_threshold:
            raise ValueError(
                "reactivation_threshold must be at least active_threshold"
            )
        if _finite_number(self.reactivation_margin, "reactivation_margin") < 0.0:
            raise ValueError("reactivation_margin must be nonnegative")
        _unit_interval(self.class_weight, "class_weight")
        if self.class_mode not in _CLASS_MODES:
            raise ValueError(
                "class_mode must be 'full' or 'foreground_normalized'"
            )
        if (
            not isinstance(self.background_class, int)
            or isinstance(self.background_class, bool)
            or self.background_class < 0
        ):
            raise ValueError("background_class must be a nonnegative integer")
        update_rate = _unit_interval(self.update_rate, "update_rate")
        if update_rate == 0.0:
            raise ValueError("update_rate must be greater than zero")
        max_update_rate = _unit_interval(
            self.max_update_rate, "max_update_rate"
        )
        if max_update_rate < update_rate:
            raise ValueError("max_update_rate must be at least update_rate")
        _optional_unit_interval(
            self.consolidation_confidence, "consolidation_confidence"
        )
        _optional_nonnegative(
            self.consolidation_margin, "consolidation_margin"
        )
        _unit_interval(self.birth_confidence, "birth_confidence")
        if (
            not isinstance(self.birth_minimum_mask_support, int)
            or isinstance(self.birth_minimum_mask_support, bool)
            or self.birth_minimum_mask_support <= 0
        ):
            raise ValueError(
                "birth_minimum_mask_support must be a positive integer"
            )
        _optional_unit_interval(self.birth_max_entropy, "birth_max_entropy")
        _unit_interval(self.mask_threshold, "mask_threshold")
        if self.assignment_mode not in _ASSIGNMENT_MODES:
            raise ValueError(
                "assignment_mode must be 'legacy_post_threshold' or "
                "'threshold_aware'"
            )


def _integer_score_units(score: Tensor) -> list[list[int]]:
    values = score.detach().to(device="cpu", dtype=torch.float64).tolist()
    ratios = [[float(value).as_integer_ratio() for value in row] for row in values]
    denominator = max(
        (value_denominator for row in ratios for _, value_denominator in row),
        default=1,
    )
    return [
        [
            numerator * (denominator // value_denominator)
            for numerator, value_denominator in row
        ]
        for row in ratios
    ]


def _maximum_weight_assignment(weights: list[list[int]]) -> list[int]:
    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if column_count == 0 or any(len(row) != column_count for row in weights):
        raise ValueError("assignment weights must be a non-empty rectangle")
    if row_count > column_count:
        raise ValueError("assignment requires at least as many columns as rows")

    maximum_weight = max(max(row) for row in weights)
    cost = [[maximum_weight - value for value in row] for row in weights]
    row_potential = [0] * (row_count + 1)
    column_potential = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)

    for row in range(1, row_count + 1):
        matched_row[0] = row
        minimum_reduced_cost: list[int | None] = [None] * (column_count + 1)
        used_column = [False] * (column_count + 1)
        column = 0
        while True:
            used_column[column] = True
            current_row = matched_row[column]
            delta: int | None = None
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used_column[candidate]:
                    continue
                reduced = (
                    cost[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate]
                )
                previous = minimum_reduced_cost[candidate]
                if previous is None or reduced < previous:
                    minimum_reduced_cost[candidate] = reduced
                    previous_column[candidate] = column
                    previous = reduced
                if delta is None or previous < delta:
                    delta = previous
                    next_column = candidate
            if delta is None:
                raise RuntimeError("assignment has no augmenting path")
            for candidate in range(column_count + 1):
                if used_column[candidate]:
                    row_potential[matched_row[candidate]] += delta
                    column_potential[candidate] -= delta
                elif minimum_reduced_cost[candidate] is not None:
                    minimum_reduced_cost[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            predecessor = previous_column[column]
            matched_row[column] = matched_row[predecessor]
            column = predecessor
            if column == 0:
                break

    assigned = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assigned[matched_row[column] - 1] = column - 1
    return assigned


def threshold_aware_assignment(
    score: Tensor, allowed: Tensor
) -> tuple[tuple[int, int], ...]:
    """Match only allowed edges, prioritizing cardinality before total score."""

    if not isinstance(score, Tensor) or score.ndim != 2:
        raise ValueError("score must have shape [N, M]")
    if not isinstance(allowed, Tensor) or allowed.shape != score.shape:
        raise ValueError("allowed must have the same shape as score")
    if allowed.dtype != torch.bool:
        raise ValueError("allowed must have bool dtype")
    score_cpu = score.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(score_cpu).all().item():
        raise ValueError("score must contain only finite values")
    allowed_cpu = allowed.detach().to(device="cpu")
    row_count, column_count = score_cpu.shape
    if row_count == 0 or column_count == 0 or not allowed_cpu.any().item():
        return ()

    score_units = _integer_score_units(score_cpu)
    maximum_dimension = row_count + column_count
    maximum_matching = min(row_count, column_count)
    tie_span = maximum_dimension * maximum_dimension + 1
    score_factor = tie_span + 1
    maximum_abs_score = max(
        abs(score_units[row][column])
        for row in range(row_count)
        for column in range(column_count)
        if bool(allowed_cpu[row, column])
    )
    cardinality_bonus = (
        (2 * maximum_abs_score * maximum_matching + 1) * score_factor
        + 2 * tie_span * maximum_matching
        + 1
    )

    dimension = row_count + column_count
    weights = [[0 for _ in range(dimension)] for _ in range(dimension)]
    for row in range(row_count):
        for column in range(column_count):
            if bool(allowed_cpu[row, column]):
                tie = maximum_dimension - abs(row - column)
                weights[row][column] = (
                    cardinality_bonus
                    + score_units[row][column] * score_factor
                    + tie
                )

    assignment = _maximum_weight_assignment(weights)
    pairs = []
    for row in range(row_count):
        column = assignment[row]
        if column < column_count and bool(allowed_cpu[row, column]):
            pairs.append((row, column))
    return tuple(pairs)
