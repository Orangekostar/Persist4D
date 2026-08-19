from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from models.persistent_memory import LocalInstanceObservation, PersistentMemoryState

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


def _normalize_features(features: Tensor) -> Tensor:
    compute_dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    compute_features = features.to(dtype=compute_dtype)
    maximum = compute_features.abs().amax(dim=-1, keepdim=True)
    nonzero = maximum > 0
    scaled = compute_features / torch.where(
        nonzero, maximum, torch.ones_like(maximum)
    )
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    return scaled / torch.where(nonzero, norm, torch.ones_like(norm))


def _foreground_normalized(class_prob: Tensor, background_class: int) -> Tensor:
    class_count = class_prob.shape[-1]
    if not 0 <= background_class < class_count:
        raise ValueError("background_class must index the class dimension")
    foreground = torch.cat(
        (
            class_prob[..., :background_class],
            class_prob[..., background_class + 1 :],
        ),
        dim=-1,
    )
    if foreground.shape[-1] == 0:
        raise ValueError("class probabilities must include a foreground class")
    total = foreground.sum(dim=-1, keepdim=True)
    return torch.where(total > 0, foreground / total.clamp_min(torch.finfo(foreground.dtype).tiny), foreground)


def _foreground_entropy(class_prob: Tensor, background_class: int) -> Tensor:
    foreground = _foreground_normalized(class_prob, background_class)
    if foreground.shape[-1] == 1:
        return torch.zeros_like(foreground[..., 0])
    terms = torch.where(
        foreground > 0,
        foreground * foreground.clamp_min(torch.finfo(foreground.dtype).tiny).log(),
        torch.zeros_like(foreground),
    )
    return -terms.sum(dim=-1) / math.log(foreground.shape[-1])


def _edge_margins(score: Tensor) -> Tensor:
    if score.shape[0] == 1:
        return score.clone()
    top_values, top_rows = torch.topk(score, k=2, dim=0, largest=True, sorted=True)
    rows = torch.arange(score.shape[0], device=score.device).unsqueeze(1)
    competitor = torch.where(
        rows == top_rows[0].unsqueeze(0),
        top_values[1].unsqueeze(0),
        top_values[0].unsqueeze(0),
    )
    return score - competitor


@dataclass(frozen=True)
class P6BStepResult:
    state: PersistentMemoryState
    slot_ids: Tensor
    association_scores: Tensor
    feature_scores: Tensor
    class_scores: Tensor
    association_margins: Tensor
    reactivations: Tensor
    consolidated: Tensor
    rejected_births: Tensor
    rejected_birth_confidence: Tensor
    rejected_birth_support: Tensor
    rejected_birth_entropy: Tensor
    rejected_birth_capacity: Tensor
    birth_mask_support: Tensor
    birth_entropy: Tensor


class P6BPersistentMemory(nn.Module):
    def __init__(self, config: P6BMemoryConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = P6BMemoryConfig()
        if not isinstance(config, P6BMemoryConfig):
            raise ValueError("config must be a P6BMemoryConfig")  # noqa: TRY004
        self.config = config

    @property
    def capacity(self) -> int:
        return self.config.capacity

    def empty_state(
        self, observation: LocalInstanceObservation
    ) -> PersistentMemoryState:
        if not isinstance(observation, LocalInstanceObservation):
            raise ValueError(  # noqa: TRY004
                "observation must be a LocalInstanceObservation"
            )
        observation.validate()
        if self.config.background_class >= observation.class_prob.shape[-1]:
            raise ValueError("background_class must index the class dimension")
        return PersistentMemoryState.empty(
            batch_size=observation.features.shape[0],
            capacity=self.capacity,
            feature_dim=observation.features.shape[2],
            class_count=observation.class_prob.shape[2],
            device=observation.features.device,
            dtype=observation.features.dtype,
        )

    def _validate_step(
        self,
        observation: LocalInstanceObservation,
        state: PersistentMemoryState,
        stage_index: int,
    ) -> None:
        if not isinstance(observation, LocalInstanceObservation):
            raise ValueError(  # noqa: TRY004
                "observation must be a LocalInstanceObservation"
            )
        if not isinstance(state, PersistentMemoryState):
            raise ValueError(  # noqa: TRY004
                "state must be a PersistentMemoryState"
            )
        observation.validate()
        state.validate()
        if (
            not isinstance(stage_index, int)
            or isinstance(stage_index, bool)
            or stage_index < 0
        ):
            raise ValueError("stage_index must be a nonnegative integer")
        if torch.any(state.stage_watermark >= stage_index).item():
            raise ValueError("stage_index must follow the processed stage watermark")
        if state.capacity != self.capacity:
            raise ValueError("state capacity must match config capacity")
        if observation.features.shape[0] != state.batch_size:
            raise ValueError("observation and state batch sizes must match")
        if observation.features.shape[2] != state.feature_dim:
            raise ValueError("observation and state feature dimensions must match")
        if observation.class_prob.shape[2] != state.class_count:
            raise ValueError("observation and state class dimensions must match")
        if self.config.background_class >= state.class_count:
            raise ValueError("background_class must index the class dimension")
        if observation.features.device != state.embedding.device:
            raise ValueError("observation and state must use the same device")
        if any(
            tensor.dtype != state.embedding.dtype
            for tensor in (
                observation.features,
                observation.class_prob,
                observation.confidence,
            )
        ):
            raise ValueError("observation and state must use the same dtype")

    def _score_components(
        self,
        observation: LocalInstanceObservation,
        state: PersistentMemoryState,
    ) -> tuple[Tensor, Tensor, Tensor]:
        compute_dtype = (
            torch.float64 if state.embedding.dtype == torch.float64 else torch.float32
        )
        memory_features = _normalize_features(
            state.embedding.to(dtype=compute_dtype)
        )
        query_features = _normalize_features(
            observation.features.to(dtype=compute_dtype)
        )
        feature_score = torch.einsum(
            "bkd,bqd->bkq", memory_features, query_features
        )
        memory_class = state.class_prob.to(dtype=compute_dtype)
        query_class = observation.class_prob.to(dtype=compute_dtype)
        if self.config.class_mode == "foreground_normalized":
            memory_class = _foreground_normalized(
                memory_class, self.config.background_class
            )
            query_class = _foreground_normalized(
                query_class, self.config.background_class
            )
        class_score = torch.einsum("bkc,bqc->bkq", memory_class, query_class)
        total_score = feature_score + self.config.class_weight * class_score
        return feature_score, class_score, total_score

    def step(
        self,
        observation: LocalInstanceObservation,
        state: PersistentMemoryState,
        stage_index: int,
    ) -> P6BStepResult:
        self._validate_step(observation, state, stage_index)
        (
            embedding,
            class_prob,
            confidence,
            occupied,
            active,
            age,
            last_seen,
            stage_watermark,
        ) = (tensor.clone() for tensor in state.tensors())
        age.add_(occupied.to(dtype=torch.long))
        active.zero_()

        feature_matrix, class_matrix, total_matrix = self._score_components(
            observation, state
        )
        batch_size, query_count = observation.valid.shape
        device = observation.features.device
        output_float = total_matrix.dtype
        slot_ids = torch.full(
            (batch_size, query_count), -1, dtype=torch.long, device=device
        )
        association_scores = torch.full(
            (batch_size, query_count),
            -torch.inf,
            dtype=output_float,
            device=device,
        )
        feature_scores = association_scores.clone()
        class_scores = association_scores.clone()
        association_margins = association_scores.clone()
        reactivations = torch.zeros_like(observation.valid)
        consolidated = torch.zeros_like(observation.valid)

        for batch_index in range(batch_size):
            occupied_slots = state.occupied[batch_index].nonzero(as_tuple=True)[0]
            valid_queries = observation.valid[batch_index].nonzero(as_tuple=True)[0]
            if occupied_slots.numel() == 0 or valid_queries.numel() == 0:
                continue
            candidate_score = total_matrix[batch_index][occupied_slots][
                :, valid_queries
            ]
            candidate_margin = _edge_margins(candidate_score)
            candidate_active = state.active[batch_index, occupied_slots].unsqueeze(1)
            active_allowed = candidate_score >= self.config.active_threshold
            dormant_allowed = (
                (candidate_score >= self.config.reactivation_threshold)
                & (candidate_margin >= self.config.reactivation_margin)
            )
            allowed = torch.where(
                candidate_active, active_allowed, dormant_allowed
            )
            if self.config.assignment_mode == "threshold_aware":
                candidate_pairs = threshold_aware_assignment(
                    candidate_score, allowed
                )
            else:
                candidate_pairs = tuple(
                    pair
                    for pair in threshold_aware_assignment(
                        candidate_score, torch.ones_like(allowed)
                    )
                    if bool(allowed[pair[0], pair[1]])
                )
            for compact_slot, compact_query in candidate_pairs:
                slot = int(occupied_slots[compact_slot].item())
                query = int(valid_queries[compact_query].item())
                slot_ids[batch_index, query] = slot
                association_scores[batch_index, query] = candidate_score[
                    compact_slot, compact_query
                ]
                feature_scores[batch_index, query] = feature_matrix[
                    batch_index, slot, query
                ]
                class_scores[batch_index, query] = class_matrix[
                    batch_index, slot, query
                ]
                association_margins[batch_index, query] = candidate_margin[
                    compact_slot, compact_query
                ]
                reactivations[batch_index, query] = not bool(
                    state.active[batch_index, slot]
                )

        for batch_index in range(batch_size):
            matched_queries = (slot_ids[batch_index] >= 0).nonzero(
                as_tuple=True
            )[0]
            for query_tensor in matched_queries:
                query = int(query_tensor.item())
                slot = int(slot_ids[batch_index, query].item())
                confidence_allowed = (
                    self.config.consolidation_confidence is None
                    or float(observation.confidence[batch_index, query].item())
                    >= self.config.consolidation_confidence
                )
                margin_allowed = (
                    self.config.consolidation_margin is None
                    or float(association_margins[batch_index, query].item())
                    >= self.config.consolidation_margin
                )
                active[batch_index, slot] = True
                last_seen[batch_index, slot] = stage_index
                if not confidence_allowed or not margin_allowed:
                    continue
                compute_dtype = (
                    torch.float64
                    if state.embedding.dtype == torch.float64
                    else torch.float32
                )
                observed_confidence = observation.confidence[
                    batch_index, query
                ].to(dtype=compute_dtype)
                update_rate = torch.clamp(
                    self.config.update_rate * observed_confidence,
                    min=0.0,
                    max=self.config.max_update_rate,
                )
                old_embedding = state.embedding[batch_index, slot].to(
                    dtype=compute_dtype
                )
                observed_embedding = observation.features[
                    batch_index, query
                ].to(dtype=compute_dtype)
                embedding[batch_index, slot] = _normalize_features(
                    (1.0 - update_rate) * old_embedding
                    + update_rate * observed_embedding
                ).to(dtype=state.embedding.dtype)
                class_prob[batch_index, slot] = (
                    (1.0 - update_rate)
                    * state.class_prob[batch_index, slot].to(dtype=compute_dtype)
                    + update_rate
                    * observation.class_prob[batch_index, query].to(
                        dtype=compute_dtype
                    )
                ).to(dtype=state.class_prob.dtype)
                confidence[batch_index, slot] = (
                    (1.0 - update_rate)
                    * state.confidence[batch_index, slot].to(dtype=compute_dtype)
                    + update_rate * observed_confidence
                ).to(dtype=state.confidence.dtype)
                consolidated[batch_index, query] = True

        birth_mask_support = torch.stack(
            [
                (mask.sigmoid() >= self.config.mask_threshold).sum(dim=1)
                for mask in observation.latest_mask
            ]
        )
        birth_entropy = _foreground_entropy(
            observation.class_prob.to(dtype=output_float),
            self.config.background_class,
        )
        rejected_birth_confidence = torch.zeros_like(observation.valid)
        rejected_birth_support = torch.zeros_like(observation.valid)
        rejected_birth_entropy = torch.zeros_like(observation.valid)
        rejected_birth_capacity = torch.zeros_like(observation.valid)

        for batch_index in range(batch_size):
            unmatched_queries = (
                observation.valid[batch_index] & (slot_ids[batch_index] < 0)
            ).nonzero(as_tuple=True)[0]
            if unmatched_queries.numel() == 0:
                continue
            confidence_failed = observation.confidence[
                batch_index, unmatched_queries
            ] < self.config.birth_confidence
            support_failed = birth_mask_support[
                batch_index, unmatched_queries
            ] < self.config.birth_minimum_mask_support
            if self.config.birth_max_entropy is None:
                entropy_failed = torch.zeros_like(confidence_failed)
            else:
                entropy_failed = birth_entropy[
                    batch_index, unmatched_queries
                ] > self.config.birth_max_entropy
            rejected_birth_confidence[
                batch_index, unmatched_queries
            ] = confidence_failed
            rejected_birth_support[batch_index, unmatched_queries] = support_failed
            rejected_birth_entropy[batch_index, unmatched_queries] = entropy_failed
            quality_allowed = ~(confidence_failed | support_failed | entropy_failed)
            birth_queries = unmatched_queries[quality_allowed]
            free_slots = (~occupied[batch_index]).nonzero(as_tuple=True)[0]
            birth_count = min(birth_queries.numel(), free_slots.numel())
            if birth_count:
                selected_queries = birth_queries[:birth_count]
                selected_slots = free_slots[:birth_count]
                embedding[batch_index, selected_slots] = _normalize_features(
                    observation.features[batch_index, selected_queries]
                ).to(dtype=state.embedding.dtype)
                class_prob[batch_index, selected_slots] = observation.class_prob[
                    batch_index, selected_queries
                ]
                confidence[batch_index, selected_slots] = observation.confidence[
                    batch_index, selected_queries
                ]
                occupied[batch_index, selected_slots] = True
                active[batch_index, selected_slots] = True
                age[batch_index, selected_slots] = 0
                last_seen[batch_index, selected_slots] = stage_index
                slot_ids[batch_index, selected_queries] = selected_slots
            if birth_count < birth_queries.numel():
                rejected_birth_capacity[
                    batch_index, birth_queries[birth_count:]
                ] = True

        rejected_births = (
            rejected_birth_confidence
            | rejected_birth_support
            | rejected_birth_entropy
            | rejected_birth_capacity
        )
        stage_watermark.fill_(stage_index)
        next_state = PersistentMemoryState(
            embedding=embedding,
            class_prob=class_prob,
            confidence=confidence,
            occupied=occupied,
            active=active,
            age=age,
            last_seen=last_seen,
            stage_watermark=stage_watermark,
        )
        next_state.validate()
        return P6BStepResult(
            state=next_state,
            slot_ids=slot_ids,
            association_scores=association_scores,
            feature_scores=feature_scores,
            class_scores=class_scores,
            association_margins=association_margins,
            reactivations=reactivations,
            consolidated=consolidated,
            rejected_births=rejected_births,
            rejected_birth_confidence=rejected_birth_confidence,
            rejected_birth_support=rejected_birth_support,
            rejected_birth_entropy=rejected_birth_entropy,
            rejected_birth_capacity=rejected_birth_capacity,
            birth_mask_support=birth_mask_support,
            birth_entropy=birth_entropy,
        )
