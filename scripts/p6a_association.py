"""CPU association baselines for the P6-A common-prefix evaluator.

The module deliberately keeps evaluator identities separate from P5's fixed
memory capacity.  Each tracker consumes one frozen local observation per
stage, retains only the state required by its declared baseline, and returns
immutable step records.  B4 delegates state transitions to the frozen P5
implementation; Oracle is a post-hoc diagnostic with no inference state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from models.persistent_memory import (
    LocalInstanceObservation,
    PersistentMemory,
    PersistentMemoryState,
    _normalize_feature_vectors,
)
from scripts.p6a_metrics import global_hungarian_match


@dataclass(frozen=True)
class FrozenObservation:
    """Detached, deep-copied observation used by every association method."""

    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    valid: Tensor
    latest_mask: tuple[Tensor, ...] = ()

    @property
    def query_count(self) -> int:
        return int(self.features.shape[-2])

    def validate(self) -> None:
        if self.features.ndim not in (2, 3):
            raise ValueError("features must have shape [Q, D] or [B, Q, D]")
        if self.class_prob.ndim != self.features.ndim:
            raise ValueError("class_prob must have the same rank as features")
        if self.features.shape[:-1] != self.class_prob.shape[:-1]:
            raise ValueError("features and class_prob must share batch/query axes")
        if self.features.shape[-1] <= 0 or self.class_prob.shape[-1] <= 0:
            raise ValueError("feature and class dimensions must be positive")
        expected = self.features.shape[:-1]
        if self.confidence.shape != expected or self.valid.shape != expected:
            raise ValueError("confidence and valid must match batch/query axes")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must have bool dtype")
        tensors = (self.features, self.class_prob, self.confidence, self.valid)
        device = self.features.device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("observation tensors must share a device")
        for name, tensor in (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
        ):
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have a floating dtype")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(self.class_prob < 0).item():
            raise ValueError("class_prob must be non-negative")
        if self.latest_mask:
            if self.features.ndim == 2:
                expected_query_count = self.features.shape[0]
            else:
                expected_query_count = self.features.shape[1]
            for mask in self.latest_mask:
                if not isinstance(mask, Tensor) or mask.ndim != 2:
                    raise ValueError("latest_mask entries must have shape [Q, S]")
                if mask.shape[0] != expected_query_count:
                    raise ValueError("latest_mask entries must match query count")
                if mask.device != device:
                    raise ValueError("latest_mask tensors must share a device")
                if not mask.is_floating_point() or not torch.isfinite(mask).all().item():
                    raise ValueError("latest_mask must contain finite floats")


# A descriptive alias makes the fan-out contract discoverable without
# changing the P5 LocalInstanceObservation type.
ImmutableObservation = FrozenObservation


def _clone_tensor(value: Tensor) -> Tensor:
    return value.detach().clone().requires_grad_(False)


def freeze_observation(
    observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
) -> FrozenObservation:
    """Return a detached deep copy without changing the caller's tensors.

    P5 observations are batched ``[B, Q, ...]`` values; the returned object
    keeps that shape so it can be fanned out to all methods exactly once.
    Trackers accept either this batched form with ``B == 1`` or an unbatched
    ``[Q, ...]`` mapping.
    """

    if isinstance(observation, FrozenObservation):
        source = observation
    elif isinstance(observation, LocalInstanceObservation):
        source = FrozenObservation(
            features=observation.features,
            class_prob=observation.class_prob,
            confidence=observation.confidence,
            valid=observation.valid,
            latest_mask=tuple(observation.latest_mask),
        )
    elif isinstance(observation, Mapping):
        required = ("features", "class_prob", "confidence", "valid")
        missing = [key for key in required if key not in observation]
        if missing:
            raise ValueError(f"observation is missing keys: {', '.join(missing)}")
        latest_mask = observation.get("latest_mask", ())
        if isinstance(latest_mask, Tensor):
            latest_mask = (latest_mask,)
        elif latest_mask is None:
            latest_mask = ()
        else:
            latest_mask = tuple(latest_mask)
        source = FrozenObservation(
            features=observation["features"],
            class_prob=observation["class_prob"],
            confidence=observation["confidence"],
            valid=observation["valid"],
            latest_mask=latest_mask,
        )
    else:
        raise ValueError(  # noqa: TRY004
            "observation must be a P5 observation or mapping"
        )

    for name in ("features", "class_prob", "confidence", "valid"):
        if not isinstance(getattr(source, name), Tensor):
            raise ValueError(f"{name} must be a tensor")  # noqa: TRY004
    frozen = FrozenObservation(
        features=_clone_tensor(source.features),
        class_prob=_clone_tensor(source.class_prob),
        confidence=_clone_tensor(source.confidence),
        valid=_clone_tensor(source.valid),
        latest_mask=tuple(_clone_tensor(mask) for mask in source.latest_mask),
    )
    frozen.validate()
    return frozen


def fan_out_observation(
    observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
    methods: Iterable[str],
) -> dict[str, FrozenObservation]:
    """Deep-copy one frozen observation for independent method consumers."""

    method_names = tuple(methods)
    if len(set(method_names)) != len(method_names):
        raise ValueError("fan-out method names must be unique")
    frozen = freeze_observation(observation)
    return {name: freeze_observation(frozen) for name in method_names}


def _single_observation(observation: FrozenObservation) -> FrozenObservation:
    frozen = freeze_observation(observation)
    if frozen.features.ndim == 2:
        return frozen
    if frozen.features.shape[0] != 1:
        raise ValueError("trackers accept one observation batch item at a time")
    return FrozenObservation(
        features=frozen.features[0],
        class_prob=frozen.class_prob[0],
        confidence=frozen.confidence[0],
        valid=frozen.valid[0],
        latest_mask=(frozen.latest_mask[0],) if frozen.latest_mask else (),
    )


class IdentityNamespace:
    """Independent, unbounded evaluator identity counters.

    Counters are keyed by ``(method, sequence)``.  Resetting one scope never
    affects another method or sequence, and no capacity or query count is
    involved in allocation.
    """

    def __init__(self) -> None:
        self._next: dict[tuple[str, str], int] = {}

    def reset(self, method: str, sequence_id: str) -> None:
        self._next[(str(method), str(sequence_id))] = 0

    def next_id(self, method: str, sequence_id: str) -> int:
        key = (str(method), str(sequence_id))
        value = self._next.get(key, 0)
        self._next[key] = value + 1
        return value

    def snapshot(self) -> dict[tuple[str, str], int]:
        return dict(self._next)


@dataclass(frozen=True)
class AssociationDiagnostics:
    """Immutable, query-aligned association evidence for one stage."""

    selected_candidate_identity: tuple[object | None, ...]
    best_candidate_identity: tuple[object | None, ...]
    chosen_feature_similarity: tuple[float | None, ...]
    chosen_class_similarity: tuple[float | None, ...]
    chosen_total_score: tuple[float | None, ...]
    best_score: tuple[float | None, ...]
    second_best_score: tuple[float | None, ...]
    score_margin: tuple[float | None, ...]
    slot_age: tuple[int | None, ...]
    last_seen_stage: tuple[int | None, ...]
    slot_active: tuple[bool | None, ...]
    slot_occupied: tuple[bool | None, ...]
    reactivation: tuple[bool | None, ...]

    def __post_init__(self) -> None:
        fields = self.per_query_fields()
        if not all(isinstance(value, tuple) for value in fields):
            raise ValueError("association diagnostics fields must be tuples")
        lengths = {len(value) for value in fields}
        if len(lengths) != 1:
            raise ValueError("association diagnostics fields must be query-aligned")
        for name in (
            "chosen_feature_similarity",
            "chosen_class_similarity",
            "chosen_total_score",
            "best_score",
            "second_best_score",
            "score_margin",
        ):
            for value in getattr(self, name):
                if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"{name} must contain finite floats or None")
        for name in ("slot_age", "last_seen_stage"):
            for value in getattr(self, name):
                if value is not None and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or (name == "slot_age" and value < 0)
                    or (name == "last_seen_stage" and value < -1)
                ):
                    raise ValueError(f"{name} must contain integers or None")
        for name in ("slot_active", "slot_occupied", "reactivation"):
            for value in getattr(self, name):
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"{name} must contain booleans or None")

    @classmethod
    def empty(cls, query_count: int) -> AssociationDiagnostics:
        if (
            not isinstance(query_count, int)
            or isinstance(query_count, bool)
            or query_count < 0
        ):
            raise ValueError("query_count must be a non-negative integer")
        none_values = (None,) * query_count
        return cls(
            selected_candidate_identity=none_values,
            best_candidate_identity=none_values,
            chosen_feature_similarity=none_values,
            chosen_class_similarity=none_values,
            chosen_total_score=none_values,
            best_score=none_values,
            second_best_score=none_values,
            score_margin=none_values,
            slot_age=none_values,
            last_seen_stage=none_values,
            slot_active=none_values,
            slot_occupied=none_values,
            reactivation=none_values,
        )

    @property
    def query_count(self) -> int:
        return len(self.selected_candidate_identity)

    def per_query_fields(self) -> tuple[tuple[object, ...], ...]:
        return (
            self.selected_candidate_identity,
            self.best_candidate_identity,
            self.chosen_feature_similarity,
            self.chosen_class_similarity,
            self.chosen_total_score,
            self.best_score,
            self.second_best_score,
            self.score_margin,
            self.slot_age,
            self.last_seen_stage,
            self.slot_active,
            self.slot_occupied,
            self.reactivation,
        )

    # Short aliases keep the typed record convenient for tabular consumers.
    @property
    def selected_identity(self) -> tuple[object | None, ...]:
        return self.selected_candidate_identity

    @property
    def best_identity(self) -> tuple[object | None, ...]:
        return self.best_candidate_identity

    @property
    def feature_similarity(self) -> tuple[float | None, ...]:
        return self.chosen_feature_similarity

    @property
    def class_similarity(self) -> tuple[float | None, ...]:
        return self.chosen_class_similarity

    @property
    def total_score(self) -> tuple[float | None, ...]:
        return self.chosen_total_score


@dataclass(frozen=True)
class TrackStep:
    """Immutable identity assignment for one stage."""

    method: str
    sequence_id: str
    stage_id: int
    track_ids: tuple[object, ...]
    matched_previous: tuple[int, ...]
    scores: tuple[float | None, ...]
    births: tuple[bool, ...]
    valid: tuple[bool, ...]
    rejected_births: tuple[bool, ...] = ()
    state_snapshot: PersistentMemoryState | None = None
    diagnostics: AssociationDiagnostics | None = None

    @property
    def evaluator_ids(self) -> tuple[object, ...]:
        return self.track_ids

    @property
    def query_count(self) -> int:
        return len(self.track_ids)


@dataclass(frozen=True)
class MatchingResult:
    """Deterministic one-to-one matching in previous/current query indices."""

    pairs: tuple[tuple[int, int], ...]
    scores: tuple[float, ...]


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def foreground_normalized_class_prob(
    class_prob: Tensor,
    *,
    background_class: int = 0,
    eps: float = 1e-12,
) -> Tensor:
    """Remove background and renormalize the remaining class posterior."""

    if not isinstance(class_prob, Tensor) or class_prob.ndim < 1:
        raise ValueError("class_prob must be a tensor with a class dimension")
    if (
        not isinstance(background_class, int)
        or isinstance(background_class, bool)
        or not 0 <= background_class < class_prob.shape[-1]
    ):
        raise ValueError("background_class must index the class dimension")
    eps_value = _finite_number(eps, "eps")
    if eps_value <= 0:
        raise ValueError("eps must be positive")
    if not class_prob.is_floating_point() or not torch.isfinite(class_prob).all().item():
        raise ValueError("class_prob must contain finite floating values")
    if torch.any(class_prob < 0).item():
        raise ValueError("class_prob must be non-negative")
    foreground = torch.cat(
        (
            class_prob[..., :background_class],
            class_prob[..., background_class + 1 :],
        ),
        dim=-1,
    )
    denominator = foreground.sum(dim=-1, keepdim=True)
    return torch.where(
        denominator > eps_value,
        foreground / denominator.clamp_min(eps_value),
        torch.zeros_like(foreground),
    )


def _cosine_scores(previous: Tensor, current: Tensor) -> Tensor:
    if previous.ndim != 2 or current.ndim != 2:
        raise ValueError("feature matrices must have shape [N, D] and [Q, D]")
    if previous.shape[1] != current.shape[1]:
        raise ValueError("previous and current feature dimensions must match")
    dtype = torch.promote_types(previous.dtype, current.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    previous_norm = F.normalize(previous.to(dtype=dtype), dim=-1, eps=1e-12)
    current_norm = F.normalize(current.to(dtype=dtype), dim=-1, eps=1e-12)
    return previous_norm @ current_norm.transpose(0, 1)


def association_score_matrix(
    previous_features: Tensor,
    current_features: Tensor,
    *,
    previous_class_prob: Tensor | None = None,
    current_class_prob: Tensor | None = None,
    class_weight: float = 0.0,
    background_class: int = 0,
) -> Tensor:
    """Compute B1/B2 scores without mutating either input."""

    weight = _finite_number(class_weight, "class_weight")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("class_weight must be within [0, 1]")
    score = _cosine_scores(previous_features, current_features)
    if weight == 0.0:
        return score
    if previous_class_prob is None or current_class_prob is None:
        raise ValueError("class probabilities are required for class-weighted scores")
    previous_fg = foreground_normalized_class_prob(
        previous_class_prob, background_class=background_class
    )
    current_fg = foreground_normalized_class_prob(
        current_class_prob, background_class=background_class
    )
    if previous_fg.shape[0] != previous_features.shape[0]:
        raise ValueError("previous class probabilities must match features")
    if current_fg.shape[0] != current_features.shape[0]:
        raise ValueError("current class probabilities must match features")
    if previous_fg.shape[1] != current_fg.shape[1]:
        raise ValueError("class dimensions must match")
    return score + weight * (previous_fg @ current_fg.transpose(0, 1))


def _integer_score_units(score: Tensor) -> list[list[int]]:
    values = score.detach().to(device="cpu", dtype=torch.float64).tolist()
    ratios = [[float(value).as_integer_ratio() for value in row] for row in values]
    denominator = max(
        (denominator for row in ratios for _, denominator in row),
        default=1,
    )
    return [
        [numerator * (denominator // value_denominator) for numerator, value_denominator in row]
        for row in ratios
    ]


def _maximum_weight_assignment(weights: list[list[int]]) -> list[int]:
    """Hungarian maximum assignment for a rectangular integer matrix."""

    if not weights:
        return []
    row_count = len(weights)
    column_count = len(weights[0])
    if column_count == 0 or any(len(row) != column_count for row in weights):
        raise ValueError("assignment weights must be a non-empty rectangular matrix")
    transposed = row_count > column_count
    if transposed:
        weights = [
            [weights[row][column] for row in range(row_count)]
            for column in range(column_count)
        ]
        row_count, column_count = column_count, row_count

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
    if not transposed:
        return assigned
    # The transposed assignment is indexed by original columns.  Convert it
    # back to original-row -> original-column form.
    original = [-1] * column_count
    for transposed_row, original_column in enumerate(assigned):
        if original_column >= 0:
            original[original_column] = transposed_row
    return original


def threshold_aware_hungarian(score: Tensor, threshold: float) -> tuple[tuple[int, int], ...]:
    """Maximize allowed one-to-one scores, forbidding low edges first.

    A cardinality term makes every finite edge at or above the threshold
    preferable to leaving both endpoints unmatched, including when callers
    intentionally use a negative threshold.  Exact integer conversion and a
    diagonal tertiary term make equal-score assignments repeatable.
    """

    cutoff = _finite_number(threshold, "threshold")
    if not isinstance(score, Tensor) or score.ndim != 2:
        raise ValueError("score must have shape [N, M]")
    row_count, column_count = score.shape
    if row_count == 0 or column_count == 0:
        return ()
    score_cpu = score.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(score_cpu).all().item():
        raise ValueError("score must contain only finite values")
    allowed = score_cpu >= cutoff
    if not allowed.any().item():
        return ()

    score_units = _integer_score_units(score_cpu)
    maximum_dimension = row_count + column_count
    matching_size = min(row_count, column_count)
    tie_span = maximum_dimension * maximum_dimension + 1
    score_factor = tie_span + 1
    maximum_abs_score = max(
        abs(score_units[row][column])
        for row in range(row_count)
        for column in range(column_count)
        if bool(allowed[row, column])
    )
    cardinality_bonus = (
        (2 * maximum_abs_score * matching_size + 1) * score_factor
        + 2 * tie_span * matching_size
        + 1
    )

    # Real rows/columns plus dummy endpoints allow unmatched observations while
    # retaining a standard square Hungarian optimization problem.
    dimension = row_count + column_count
    weights = [[0 for _ in range(dimension)] for _ in range(dimension)]
    for row in range(row_count):
        for column in range(column_count):
            if bool(allowed[row, column]):
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
        if 0 <= column < column_count and bool(allowed[row, column]):
            pairs.append((row, column))
    return tuple(pairs)


def match_feature_observations(
    previous_features: Tensor,
    current_features: Tensor,
    *,
    threshold: float = 0.5,
) -> MatchingResult:
    """Pure B1 feature-only matching helper."""

    score = _cosine_scores(previous_features, current_features)
    pairs = threshold_aware_hungarian(score, threshold)
    return MatchingResult(
        pairs=pairs,
        scores=tuple(float(score[row, column].item()) for row, column in pairs),
    )


@dataclass
class _Track:
    identity: int
    prototype: Tensor
    class_prob: Tensor
    query_index: int


def _build_adjacent_diagnostics(
    *,
    score_matrix: Tensor,
    feature_matrix: Tensor,
    class_matrix: Tensor | None,
    previous: tuple[_Track, ...],
    current_indices: Sequence[int],
    selected_by_query: Mapping[int, tuple[int, int]],
    query_count: int,
) -> AssociationDiagnostics:
    selected_identity: list[object | None] = [None] * query_count
    best_identity: list[object | None] = [None] * query_count
    chosen_feature: list[float | None] = [None] * query_count
    chosen_class: list[float | None] = [None] * query_count
    chosen_total: list[float | None] = [None] * query_count
    best_score: list[float | None] = [None] * query_count
    second_score: list[float | None] = [None] * query_count
    margin: list[float | None] = [None] * query_count

    for compact_index, query_index in enumerate(current_indices):
        if score_matrix.shape[0] == 0:
            continue
        ranked_rows = sorted(
            range(score_matrix.shape[0]),
            key=lambda row: (-float(score_matrix[row, compact_index].item()), row),
        )
        best_row = ranked_rows[0]
        best_value = float(score_matrix[best_row, compact_index].item())
        best_identity[query_index] = previous[best_row].identity
        best_score[query_index] = best_value
        if len(ranked_rows) > 1:
            second_value = float(score_matrix[ranked_rows[1], compact_index].item())
            second_score[query_index] = second_value
            margin[query_index] = best_value - second_value
        selected = selected_by_query.get(query_index)
        if selected is None:
            continue
        previous_index, selected_compact_index = selected
        selected_identity[query_index] = previous[previous_index].identity
        chosen_feature[query_index] = float(
            feature_matrix[previous_index, selected_compact_index].item()
        )
        if class_matrix is not None:
            chosen_class[query_index] = float(
                class_matrix[previous_index, selected_compact_index].item()
            )
        chosen_total[query_index] = float(
            score_matrix[previous_index, selected_compact_index].item()
        )

    return AssociationDiagnostics(
        selected_candidate_identity=tuple(selected_identity),
        best_candidate_identity=tuple(best_identity),
        chosen_feature_similarity=tuple(chosen_feature),
        chosen_class_similarity=tuple(chosen_class),
        chosen_total_score=tuple(chosen_total),
        best_score=tuple(best_score),
        second_best_score=tuple(second_score),
        score_margin=tuple(margin),
        slot_age=tuple(None for _ in range(query_count)),
        last_seen_stage=tuple(None for _ in range(query_count)),
        slot_active=tuple(None for _ in range(query_count)),
        slot_occupied=tuple(None for _ in range(query_count)),
        reactivation=tuple(None for _ in range(query_count)),
    )


class _AdjacentTracker:
    method = ""
    use_class = False

    def __init__(
        self,
        *,
        sequence_id: str,
        namespace: IdentityNamespace | None = None,
        feature_threshold: float = 0.5,
        class_weight: float = 0.25,
        background_class: int = 0,
    ) -> None:
        self.sequence_id = str(sequence_id)
        self.namespace = namespace or IdentityNamespace()
        self.feature_threshold = _finite_number(feature_threshold, "feature_threshold")
        self.class_weight = _finite_number(class_weight, "class_weight")
        if not 0.0 <= self.class_weight <= 1.0:
            raise ValueError("class_weight must be within [0, 1]")
        if (
            not isinstance(background_class, int)
            or isinstance(background_class, bool)
            or background_class < 0
        ):
            raise ValueError("background_class must be a non-negative integer")
        self.background_class = background_class
        self.reset(sequence_id=self.sequence_id)

    def reset(self, *, sequence_id: str | None = None) -> None:
        if sequence_id is not None:
            self.sequence_id = str(sequence_id)
        self.namespace.reset(self.method, self.sequence_id)
        self._last_stage: int | None = None
        self._previous_tracks: tuple[_Track, ...] = ()

    def _check_stage(self, stage_id: int) -> None:
        if not isinstance(stage_id, int) or isinstance(stage_id, bool) or stage_id < 0:
            raise ValueError("stage_id must be a non-negative integer")
        if self._last_stage is not None and stage_id <= self._last_stage:
            raise ValueError("stage_id must increase for each tracker step")

    def _score_components(
        self,
        previous: tuple[_Track, ...],
        current: FrozenObservation,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        if not previous:
            empty = torch.empty(
                (0, int(current.valid.sum().item())),
                dtype=current.features.dtype,
                device=current.features.device,
            )
            return empty, None, empty
        previous_features = torch.stack([track.prototype for track in previous])
        current_indices = current.valid.nonzero(as_tuple=True)[0]
        current_features = current.features[current_indices]
        feature_score = _cosine_scores(previous_features, current_features)
        if self.use_class:
            previous_class = torch.stack([track.class_prob for track in previous])
            current_class = current.class_prob[current_indices]
            previous_foreground = foreground_normalized_class_prob(
                previous_class, background_class=self.background_class
            )
            current_foreground = foreground_normalized_class_prob(
                current_class, background_class=self.background_class
            )
            class_score = previous_foreground @ current_foreground.transpose(0, 1)
            return feature_score, class_score, feature_score + self.class_weight * class_score
        return feature_score, None, feature_score

    def _score(self, previous: tuple[_Track, ...], current: FrozenObservation) -> Tensor:
        return self._score_components(previous, current)[2]

    def _new_track(
        self, current: FrozenObservation, query_index: int, identity: int
    ) -> _Track:
        return _Track(
            identity=identity,
            prototype=current.features[query_index].detach().clone(),
            class_prob=current.class_prob[query_index].detach().clone(),
            query_index=query_index,
        )

    def _update_prototype(self, track: _Track, current: FrozenObservation, query: int) -> Tensor:
        return current.features[query].detach().clone()

    def step(
        self,
        observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
        *,
        stage_id: int,
    ) -> TrackStep:
        self._check_stage(stage_id)
        current = _single_observation(freeze_observation(observation))
        current.validate()
        if self.background_class >= current.class_prob.shape[-1]:
            raise ValueError("background_class must index the observation classes")
        query_count = current.query_count
        track_ids: list[object] = [None] * query_count
        matched_previous = [-1] * query_count
        scores: list[float | None] = [None] * query_count
        births = [False] * query_count
        current_indices = current.valid.nonzero(as_tuple=True)[0].tolist()
        previous = self._previous_tracks
        contiguous = self._last_stage is not None and stage_id == self._last_stage + 1
        pairs: tuple[tuple[int, int], ...] = ()
        score_matrix = torch.empty((0, len(current_indices)), dtype=current.features.dtype)
        feature_matrix = score_matrix
        class_matrix: Tensor | None = None
        if contiguous and previous and current_indices:
            feature_matrix, class_matrix, score_matrix = self._score_components(
                previous, current
            )
            pairs = threshold_aware_hungarian(score_matrix, self.feature_threshold)

        next_tracks: list[_Track] = []
        matched_by_current: dict[int, _Track] = {}
        selected_by_query: dict[int, tuple[int, int]] = {}
        for previous_index, compact_current_index in pairs:
            query_index = current_indices[compact_current_index]
            track = previous[previous_index]
            matched_by_current[query_index] = track
            selected_by_query[query_index] = (previous_index, compact_current_index)
            track_ids[query_index] = track.identity
            matched_previous[query_index] = track.query_index
            scores[query_index] = float(score_matrix[previous_index, compact_current_index].item())

        for query_index in current_indices:
            track = matched_by_current.get(query_index)
            if track is None:
                identity = self.namespace.next_id(self.method, self.sequence_id)
                track = self._new_track(current, query_index, identity)
                track_ids[query_index] = identity
                births[query_index] = True
            else:
                track = _Track(
                    identity=track.identity,
                    prototype=self._update_prototype(track, current, query_index),
                    class_prob=current.class_prob[query_index].detach().clone(),
                    query_index=query_index,
                )
            next_tracks.append(track)

        self._previous_tracks = tuple(next_tracks)
        self._last_stage = stage_id
        return TrackStep(
            method=self.method,
            sequence_id=self.sequence_id,
            stage_id=stage_id,
            track_ids=tuple(track_ids),
            matched_previous=tuple(matched_previous),
            scores=tuple(scores),
            births=tuple(births),
            valid=tuple(bool(value) for value in current.valid.tolist()),
            rejected_births=tuple(False for _ in range(query_count)),
            diagnostics=_build_adjacent_diagnostics(
                score_matrix=score_matrix,
                feature_matrix=feature_matrix,
                class_matrix=class_matrix,
                previous=previous,
                current_indices=current_indices,
                selected_by_query=selected_by_query,
                query_count=query_count,
            ),
        )


class B1FeatureTracker(_AdjacentTracker):
    method = "B1"


class B2FeatureClassTracker(_AdjacentTracker):
    method = "B2"
    use_class = True


class B3EmaTracker(_AdjacentTracker):
    method = "B3"
    use_class = True

    def __init__(self, *, update_rate: float = 0.2, **kwargs: Any) -> None:
        self.update_rate = _finite_number(update_rate, "update_rate")
        if not 0.0 <= self.update_rate <= 1.0:
            raise ValueError("update_rate must be within [0, 1]")
        super().__init__(**kwargs)

    @property
    def prototypes(self) -> dict[int, Tensor]:
        return {
            track.identity: track.prototype.detach().clone()
            for track in self._previous_tracks
        }

    def _update_prototype(self, track: _Track, current: FrozenObservation, query: int) -> Tensor:
        return (
            (1.0 - self.update_rate) * track.prototype
            + self.update_rate * current.features[query]
        ).detach().clone()


class B0StageUniqueTracker:
    method = "B0"

    def __init__(self, *, sequence_id: str) -> None:
        self.sequence_id = str(sequence_id)
        self._last_stage: int | None = None

    def reset(self, *, sequence_id: str | None = None) -> None:
        if sequence_id is not None:
            self.sequence_id = str(sequence_id)
        self._last_stage = None

    def step(
        self,
        observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
        *,
        stage_id: int,
    ) -> TrackStep:
        if not isinstance(stage_id, int) or isinstance(stage_id, bool) or stage_id < 0:
            raise ValueError("stage_id must be a non-negative integer")
        if self._last_stage is not None and stage_id <= self._last_stage:
            raise ValueError("stage_id must increase for each tracker step")
        current = _single_observation(freeze_observation(observation))
        current.validate()
        track_ids = tuple(
            (stage_id, query_index) if bool(current.valid[query_index]) else None
            for query_index in range(current.query_count)
        )
        self._last_stage = stage_id
        return TrackStep(
            method=self.method,
            sequence_id=self.sequence_id,
            stage_id=stage_id,
            track_ids=track_ids,
            matched_previous=tuple(-1 for _ in track_ids),
            scores=tuple(None for _ in track_ids),
            births=tuple(identity is not None for identity in track_ids),
            valid=tuple(bool(value) for value in current.valid.tolist()),
            rejected_births=tuple(False for _ in track_ids),
            diagnostics=AssociationDiagnostics.empty(current.query_count),
        )


class B0SanityTracker(B0StageUniqueTracker):
    method = "B0-sanity"

    def step(
        self,
        observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
        *,
        stage_id: int,
    ) -> TrackStep:
        result = super().step(observation, stage_id=stage_id)
        track_ids = tuple(
            index if identity is not None else None
            for index, identity in enumerate(result.track_ids)
        )
        return TrackStep(
            method=self.method,
            sequence_id=result.sequence_id,
            stage_id=result.stage_id,
            track_ids=track_ids,
            matched_previous=result.matched_previous,
            scores=result.scores,
            births=result.births,
            valid=result.valid,
            rejected_births=result.rejected_births,
            diagnostics=result.diagnostics,
        )


def _clone_persistent_state(state: PersistentMemoryState) -> PersistentMemoryState:
    return PersistentMemoryState(
        *(tensor.detach().clone() for tensor in state.tensors())
    )


def _as_p5_observation(observation: FrozenObservation) -> LocalInstanceObservation:
    """Adapt one unbatched frozen observation to the unchanged P5 contract."""

    if observation.features.ndim != 2:
        raise ValueError("P5 adaptation requires one unbatched observation")
    feature_dtype = observation.features.dtype
    if observation.latest_mask:
        if len(observation.latest_mask) != 1:
            raise ValueError("one observation must provide at most one latest mask")
        latest_mask = observation.latest_mask[0]
        if latest_mask.shape[0] != observation.query_count:
            raise ValueError("latest_mask must match observation query count")
        if latest_mask.dtype != feature_dtype:
            latest_mask = latest_mask.to(dtype=feature_dtype)
    else:
        latest_mask = torch.empty(
            (observation.query_count, 0),
            device=observation.features.device,
            dtype=feature_dtype,
        )
    return LocalInstanceObservation(
        features=observation.features.unsqueeze(0),
        class_prob=observation.class_prob.to(dtype=feature_dtype).unsqueeze(0),
        confidence=observation.confidence.to(dtype=feature_dtype).unsqueeze(0),
        latest_mask=[latest_mask],
        valid=observation.valid.unsqueeze(0),
    )


def _p5_score_components(
    observation: LocalInstanceObservation,
    state: PersistentMemoryState,
    *,
    class_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Read-only reproduction of P5's pre-step score components."""

    score_dtype = observation.features.dtype
    for dtype in (
        observation.class_prob.dtype,
        state.embedding.dtype,
        state.class_prob.dtype,
    ):
        score_dtype = torch.promote_types(score_dtype, dtype)
    if score_dtype in (torch.float16, torch.bfloat16):
        score_dtype = torch.float32
    query_features = _normalize_feature_vectors(
        observation.features.to(dtype=score_dtype)
    )
    memory_features = _normalize_feature_vectors(
        state.embedding.to(dtype=score_dtype)
    )
    feature_score = torch.einsum(
        "bkd,bqd->bkq", memory_features, query_features
    )
    class_score = torch.einsum(
        "bkc,bqc->bkq",
        state.class_prob.to(dtype=score_dtype),
        observation.class_prob.to(dtype=score_dtype),
    )
    return (
        feature_score[0],
        class_score[0],
        feature_score[0] + class_weight * class_score[0],
    )


def _build_b4_diagnostics(
    *,
    current: FrozenObservation,
    state_before: PersistentMemoryState,
    feature_score: Tensor,
    class_score: Tensor,
    total_score: Tensor,
    slot_values: Sequence[object],
    score_values: Sequence[object],
    rejected_values: Sequence[object],
) -> AssociationDiagnostics:
    query_count = current.query_count
    selected_identity: list[object | None] = [None] * query_count
    best_identity: list[object | None] = [None] * query_count
    chosen_feature: list[float | None] = [None] * query_count
    chosen_class: list[float | None] = [None] * query_count
    chosen_total: list[float | None] = [None] * query_count
    best_score: list[float | None] = [None] * query_count
    second_score: list[float | None] = [None] * query_count
    margin: list[float | None] = [None] * query_count
    slot_age: list[int | None] = [None] * query_count
    last_seen_stage: list[int | None] = [None] * query_count
    slot_active: list[bool | None] = [None] * query_count
    slot_occupied: list[bool | None] = [None] * query_count
    reactivation: list[bool | None] = [None] * query_count

    occupied_slots = state_before.occupied[0].nonzero(as_tuple=True)[0].tolist()
    occupied_before = state_before.occupied[0].detach().cpu()
    active_before = state_before.active[0].detach().cpu()
    age_before = state_before.age[0].detach().cpu()
    last_seen_before = state_before.last_seen[0].detach().cpu()

    for query_index in range(query_count):
        if not bool(current.valid[query_index]):
            continue
        ranked_slots = sorted(
            occupied_slots,
            key=lambda slot: (
                -float(total_score[slot, query_index].item()),
                int(slot),
            ),
        )
        best_slot: int | None = None
        if ranked_slots:
            best_slot = int(ranked_slots[0])
            best_value = float(total_score[best_slot, query_index].item())
            best_identity[query_index] = best_slot
            best_score[query_index] = best_value
            if len(ranked_slots) > 1:
                second_value = float(
                    total_score[ranked_slots[1], query_index].item()
                )
                second_score[query_index] = second_value
                margin[query_index] = best_value - second_value

        selected_slot: int | None = None
        if not bool(rejected_values[query_index]):
            candidate_slot = int(slot_values[query_index])
            if (
                candidate_slot >= 0
                and candidate_slot < state_before.capacity
                and bool(occupied_before[candidate_slot])
                and math.isfinite(float(score_values[query_index]))
            ):
                selected_slot = candidate_slot
                selected_identity[query_index] = selected_slot
                chosen_feature[query_index] = float(
                    feature_score[selected_slot, query_index].item()
                )
                chosen_class[query_index] = float(
                    class_score[selected_slot, query_index].item()
                )
                chosen_total[query_index] = float(
                    score_values[query_index]
                )

        metadata_slot = selected_slot if selected_slot is not None else best_slot
        if metadata_slot is None:
            continue
        slot_age[query_index] = int(age_before[metadata_slot].item())
        last_seen_stage[query_index] = int(last_seen_before[metadata_slot].item())
        slot_active[query_index] = bool(active_before[metadata_slot].item())
        slot_occupied[query_index] = bool(occupied_before[metadata_slot].item())
        reactivation[query_index] = False
        if selected_slot is not None:
            reactivation[query_index] = not bool(active_before[selected_slot].item())

    return AssociationDiagnostics(
        selected_candidate_identity=tuple(selected_identity),
        best_candidate_identity=tuple(best_identity),
        chosen_feature_similarity=tuple(chosen_feature),
        chosen_class_similarity=tuple(chosen_class),
        chosen_total_score=tuple(chosen_total),
        best_score=tuple(best_score),
        second_best_score=tuple(second_score),
        score_margin=tuple(margin),
        slot_age=tuple(slot_age),
        last_seen_stage=tuple(last_seen_stage),
        slot_active=tuple(slot_active),
        slot_occupied=tuple(slot_occupied),
        reactivation=tuple(reactivation),
    )


class B4PersistentTracker:
    """Thin evaluator adapter around the frozen P5 PersistentMemory module."""

    method = "B4"

    def __init__(
        self,
        *,
        sequence_id: str,
        capacity: int = 100,
        class_weight: float = 0.25,
        association_threshold: float = 0.5,
        update_rate: float = 0.2,
        max_update_rate: float = 0.2,
    ) -> None:
        self.sequence_id = str(sequence_id)
        self.memory = PersistentMemory(
            capacity=capacity,
            class_weight=class_weight,
            association_threshold=association_threshold,
            update_rate=update_rate,
            max_update_rate=max_update_rate,
        )
        self._state: PersistentMemoryState | None = None
        self._last_stage: int | None = None
        self._previous_slot_to_query: dict[int, int] = {}

    @property
    def state(self) -> PersistentMemoryState | None:
        return self._state

    def reset(self, *, sequence_id: str | None = None) -> None:
        if sequence_id is not None:
            self.sequence_id = str(sequence_id)
        self._state = None
        self._last_stage = None
        self._previous_slot_to_query = {}

    def step(
        self,
        observation: FrozenObservation | LocalInstanceObservation | Mapping[str, Any],
        *,
        stage_id: int,
        timing_sink: Callable[[Mapping[str, float]], object] | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> TrackStep:
        if (
            not isinstance(stage_id, int)
            or isinstance(stage_id, bool)
            or stage_id < 0
        ):
            raise ValueError("stage_id must be a non-negative integer")
        if self._last_stage is not None and stage_id <= self._last_stage:
            raise ValueError("stage_id must increase for each tracker step")
        current = _single_observation(freeze_observation(observation))
        current.validate()
        p5_observation = _as_p5_observation(current)
        state_before = self._state
        if state_before is None:
            state_before = self.memory.empty_state(p5_observation)
        feature_score, class_score, total_score = _p5_score_components(
            p5_observation,
            state_before,
            class_weight=self.memory.class_weight,
        )
        # This is the sole B4 transition.  The adapter intentionally does not
        # reimplement P5 scoring, assignment, or EMA updates.
        timing_kwargs: dict[str, Any] = {"timing_sink": timing_sink}
        if clock_ns is not None:
            timing_kwargs["clock_ns"] = clock_ns
        result = self.memory.step(
            p5_observation,
            state_before,
            stage_id,
            **timing_kwargs,
        )
        next_state = result.state
        slot_values = result.slot_ids[0].detach().cpu().tolist()
        score_values = result.association_scores[0].detach().cpu().tolist()
        rejected_values = result.rejected_births[0].detach().cpu().tolist()
        previous_slot_to_query = self._previous_slot_to_query
        track_ids: list[object] = [None] * current.query_count
        matched_previous = [-1] * current.query_count
        scores: list[float | None] = [None] * current.query_count
        births = [False] * current.query_count
        current_slot_to_query: dict[int, int] = {}
        occupied_before = state_before.occupied[0].detach().cpu()

        for query_index in range(current.query_count):
            if not bool(current.valid[query_index]):
                continue
            if bool(rejected_values[query_index]):
                continue
            slot = int(slot_values[query_index])
            if slot < 0:
                continue
            track_ids[query_index] = slot
            current_slot_to_query[slot] = query_index
            previous_query = previous_slot_to_query.get(slot)
            if previous_query is not None:
                matched_previous[query_index] = previous_query
            if not bool(occupied_before[slot]):
                births[query_index] = True
            if not births[query_index]:
                score = float(score_values[query_index])
                if math.isfinite(score):
                    scores[query_index] = score

        self._state = next_state
        self._last_stage = stage_id
        self._previous_slot_to_query = current_slot_to_query
        return TrackStep(
            method=self.method,
            sequence_id=self.sequence_id,
            stage_id=stage_id,
            track_ids=tuple(track_ids),
            matched_previous=tuple(matched_previous),
            scores=tuple(scores),
            births=tuple(births),
            valid=tuple(bool(value) for value in current.valid.tolist()),
            rejected_births=tuple(bool(value) for value in rejected_values),
            state_snapshot=_clone_persistent_state(next_state),
            diagnostics=_build_b4_diagnostics(
                current=current,
                state_before=state_before,
                feature_score=feature_score,
                class_score=class_score,
                total_score=total_score,
                slot_values=slot_values,
                score_values=score_values,
                rejected_values=rejected_values,
            ),
        )


@dataclass(frozen=True)
class OracleStageTarget:
    """Typed GT input accepted only by the post-hoc Oracle diagnostic."""

    gt_ids: tuple[Hashable, ...]
    classes: tuple[Hashable, ...]
    masks: Tensor

    def __post_init__(self) -> None:
        gt_ids = _normalize_oracle_values(self.gt_ids, name="gt_ids")
        classes = _normalize_oracle_values(self.classes, name="classes")
        masks = torch.as_tensor(self.masks).detach().cpu().clone()
        if masks.ndim != 2:
            raise ValueError("Oracle target masks must have shape [G, P]")
        if len(gt_ids) != masks.shape[0] or len(classes) != masks.shape[0]:
            raise ValueError("Oracle target fields must share the GT count")
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("Oracle gt_ids must be unique within a stage")
        object.__setattr__(self, "gt_ids", gt_ids)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "masks", masks.bool())

    def validate(self) -> None:
        if self.masks.ndim != 2:
            raise ValueError("Oracle target masks must have shape [G, P]")
        if len(self.gt_ids) != self.masks.shape[0] or len(self.classes) != self.masks.shape[0]:
            raise ValueError("Oracle target fields must share the GT count")


def _normalize_oracle_values(value: Any, *, name: str) -> tuple[Hashable, ...]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"Oracle {name} must have shape [G]")
        values = value.detach().cpu().tolist()
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError(f"Oracle {name} must be a one-dimensional sequence") from error
    normalized = tuple(values)
    if any(not isinstance(item, Hashable) for item in normalized):
        raise ValueError(f"Oracle {name} values must be hashable")
    return normalized


def _oracle_masks(
    observation: FrozenObservation,
    query_indices: Sequence[int],
    point_count: int,
) -> Tensor:
    if not observation.latest_mask:
        return torch.zeros(
            (len(query_indices), point_count), dtype=torch.bool
        )
    mask_logits = observation.latest_mask[0]
    if mask_logits.ndim != 2 or mask_logits.shape[0] != observation.query_count:
        raise ValueError("Oracle latest_mask must have shape [Q, P]")
    if mask_logits.shape[1] == 0:
        return torch.zeros(
            (len(query_indices), point_count), dtype=torch.bool
        )
    if mask_logits.shape[1] != point_count:
        raise ValueError("Oracle masks must share the target point count")
    selected = mask_logits[list(query_indices)].detach().cpu()
    if selected.dtype == torch.bool:
        return selected
    return selected.sigmoid() >= 0.5


def _mask_iou(gt_mask: Tensor, prediction_mask: Tensor) -> float:
    intersection = (gt_mask & prediction_mask).sum().item()
    union = (gt_mask | prediction_mask).sum().item()
    return float(intersection / union) if union else 0.0


def run_oracle_posthoc(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    targets: Sequence[OracleStageTarget],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    background_class: int = 0,
    iou_threshold: float = 0.5,
) -> tuple[TrackStep, ...]:
    """Assign GT identities after inference without creating tracker state."""

    if len(observations) != len(targets):
        raise ValueError("observations and targets must have equal length")
    if stage_ids is None:
        stage_ids = tuple(range(len(observations)))
    if len(stage_ids) != len(observations):
        raise ValueError("stage_ids must match observations")
    if (
        not isinstance(background_class, int)
        or isinstance(background_class, bool)
        or background_class < 0
    ):
        raise ValueError("background_class must be a non-negative integer")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    frozen_observations = [
        _single_observation(freeze_observation(observation))
        for observation in observations
    ]
    results: list[TrackStep] = []
    for current, target, stage_id in zip(
        frozen_observations, targets, stage_ids, strict=True
    ):
        if not isinstance(stage_id, int) or isinstance(stage_id, bool) or stage_id < 0:
            raise ValueError("stage_id must be a non-negative integer")
        if not isinstance(target, OracleStageTarget):
            raise ValueError(  # noqa: TRY004
                "targets must contain OracleStageTarget values"
            )
        target.validate()
        if background_class >= current.class_prob.shape[-1]:
            raise ValueError("background_class must index the observation classes")
        valid_indices = current.valid.nonzero(as_tuple=True)[0].tolist()
        prediction_masks = _oracle_masks(
            current, valid_indices, int(target.masks.shape[1])
        )
        foreground_indices = [
            index
            for index in range(current.class_prob.shape[-1])
            if index != background_class
        ]
        if not foreground_indices:
            raise ValueError("observation must contain a foreground class")
        if valid_indices:
            foreground_prob = current.class_prob[valid_indices][
                ..., foreground_indices
            ]
            prediction_classes = torch.tensor(
                [
                    foreground_indices[int(index)]
                    for index in foreground_prob.argmax(dim=-1).detach().cpu().tolist()
                ],
                dtype=torch.long,
            )
        else:
            prediction_classes = torch.empty(0, dtype=torch.long)
        if valid_indices and target.masks.shape[0] and prediction_masks.shape[1]:
            pairs = global_hungarian_match(
                target.masks,
                prediction_masks,
                gt_classes=torch.as_tensor(target.classes),
                pred_classes=prediction_classes,
                threshold=float(iou_threshold),
            )
        else:
            pairs = []
        matched_predictions = {prediction: gt for gt, prediction in pairs}
        track_ids: list[object] = [None] * current.query_count
        scores: list[float | None] = [None] * current.query_count
        births = [False] * current.query_count
        for compact_index, query_index in enumerate(valid_indices):
            if compact_index in matched_predictions:
                gt_index = matched_predictions[compact_index]
                track_ids[query_index] = target.gt_ids[gt_index]
                scores[query_index] = _mask_iou(
                    target.masks[gt_index], prediction_masks[compact_index]
                )
            else:
                track_ids[query_index] = ("Oracle", int(stage_id), int(query_index))
                births[query_index] = True
        results.append(
            TrackStep(
                method="Oracle",
                sequence_id=str(sequence_id),
                stage_id=int(stage_id),
                track_ids=tuple(track_ids),
                matched_previous=tuple(-1 for _ in track_ids),
                scores=tuple(scores),
                births=tuple(births),
                valid=tuple(bool(value) for value in current.valid.tolist()),
                rejected_births=tuple(False for _ in track_ids),
                state_snapshot=None,
                diagnostics=AssociationDiagnostics.empty(current.query_count),
            )
        )
    return tuple(results)


def _run_tracker(
    tracker: Any,
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    stage_ids: Sequence[int] | None,
) -> tuple[TrackStep, ...]:
    if stage_ids is None:
        stage_ids = tuple(range(len(observations)))
    if len(stage_ids) != len(observations):
        raise ValueError("stage_ids must match the number of observations")
    return tuple(
        tracker.step(observation, stage_id=stage_id)
        for observation, stage_id in zip(observations, stage_ids, strict=True)
    )


def run_b0(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B0StageUniqueTracker(sequence_id=sequence_id), observations, stage_ids=stage_ids
    )


def run_b0_sanity(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B0SanityTracker(sequence_id=sequence_id), observations, stage_ids=stage_ids
    )


def run_b1(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    namespace: IdentityNamespace | None = None,
    feature_threshold: float = 0.5,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B1FeatureTracker(
            sequence_id=sequence_id,
            namespace=namespace,
            feature_threshold=feature_threshold,
            class_weight=0.0,
        ),
        observations,
        stage_ids=stage_ids,
    )


def run_b2(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    namespace: IdentityNamespace | None = None,
    feature_threshold: float = 0.5,
    class_weight: float = 0.25,
    background_class: int = 0,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B2FeatureClassTracker(
            sequence_id=sequence_id,
            namespace=namespace,
            feature_threshold=feature_threshold,
            class_weight=class_weight,
            background_class=background_class,
        ),
        observations,
        stage_ids=stage_ids,
    )


def run_b3(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    namespace: IdentityNamespace | None = None,
    feature_threshold: float = 0.5,
    class_weight: float = 0.25,
    background_class: int = 0,
    update_rate: float = 0.2,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B3EmaTracker(
            sequence_id=sequence_id,
            namespace=namespace,
            feature_threshold=feature_threshold,
            class_weight=class_weight,
            background_class=background_class,
            update_rate=update_rate,
        ),
        observations,
        stage_ids=stage_ids,
    )


def run_b4(
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    capacity: int = 100,
    class_weight: float = 0.25,
    association_threshold: float = 0.5,
    update_rate: float = 0.2,
    max_update_rate: float = 0.2,
) -> tuple[TrackStep, ...]:
    return _run_tracker(
        B4PersistentTracker(
            sequence_id=sequence_id,
            capacity=capacity,
            class_weight=class_weight,
            association_threshold=association_threshold,
            update_rate=update_rate,
            max_update_rate=max_update_rate,
        ),
        observations,
        stage_ids=stage_ids,
    )


def run_baseline(
    method: str,
    observations: Sequence[FrozenObservation | LocalInstanceObservation | Mapping[str, Any]],
    *,
    sequence_id: str,
    stage_ids: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[TrackStep, ...]:
    """Dispatch one baseline while keeping a fresh per-method/sequence state."""

    normalized = method.strip().lower()
    runners = {
        "b0": run_b0,
        "b0-sanity": run_b0_sanity,
        "b0_sanity": run_b0_sanity,
        "b1": run_b1,
        "b2": run_b2,
        "b3": run_b3,
        "b4": run_b4,
    }
    if normalized not in runners:
        raise ValueError("method must be one of B0, B0-sanity, B1, B2, or B3/B4")
    return runners[normalized](
        observations,
        sequence_id=sequence_id,
        stage_ids=stage_ids,
        **kwargs,
    )


__all__ = [
    "AssociationDiagnostics",
    "B0SanityTracker",
    "B0StageUniqueTracker",
    "B1FeatureTracker",
    "B2FeatureClassTracker",
    "B3EmaTracker",
    "B4PersistentTracker",
    "FrozenObservation",
    "IdentityNamespace",
    "ImmutableObservation",
    "LocalInstanceObservation",
    "MatchingResult",
    "OracleStageTarget",
    "TrackStep",
    "association_score_matrix",
    "fan_out_observation",
    "foreground_normalized_class_prob",
    "freeze_observation",
    "match_feature_observations",
    "run_b0",
    "run_b0_sanity",
    "run_b1",
    "run_b2",
    "run_b3",
    "run_b4",
    "run_baseline",
    "run_oracle_posthoc",
    "threshold_aware_hungarian",
]
