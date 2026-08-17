from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


@dataclass(frozen=True)
class LocalInstanceObservation:
    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    latest_mask: list[Tensor]
    valid: Tensor

    def validate(self) -> None:
        for name, tensor in (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
            ("valid", self.valid),
        ):
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{name} must be a tensor")  # noqa: TRY004

        if self.features.ndim != 3:
            raise ValueError("features must have shape [B, Q, D]")
        batch_size, query_count, feature_dim = self.features.shape
        if feature_dim <= 0:
            raise ValueError("features must have a positive feature dimension")

        if self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, Q, C]")
        if self.class_prob.shape[:2] != (batch_size, query_count):
            raise ValueError("class_prob must match the features batch and query axes")
        if self.class_prob.shape[2] <= 0:
            raise ValueError("class_prob must have a positive class dimension")

        expected_shape = (batch_size, query_count)
        if self.confidence.shape != expected_shape:
            raise ValueError("confidence must have shape [B, Q]")
        if self.valid.shape != expected_shape:
            raise ValueError("valid must have shape [B, Q]")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must have bool dtype")

        if not isinstance(self.latest_mask, list):
            raise ValueError("latest_mask must be a list of tensors")  # noqa: TRY004
        if len(self.latest_mask) != batch_size:
            raise ValueError("latest_mask must contain one tensor per batch item")
        for mask in self.latest_mask:
            if not isinstance(mask, Tensor):
                raise ValueError(  # noqa: TRY004
                    "latest_mask entries must be tensors"
                )
            if mask.ndim != 2 or mask.shape[0] != query_count:
                raise ValueError("latest_mask entries must have shape [Q, S_latest]")

        expected_device = self.features.device
        if any(
            tensor.device != expected_device
            for tensor in (self.class_prob, self.confidence, self.valid)
        ) or any(mask.device != expected_device for mask in self.latest_mask):
            raise ValueError("all observation tensors must use the same device")

        for name, tensor in (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
        ):
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must have a floating dtype")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")


def build_local_observation(
    outputs: Mapping[str, object],
    segment_stages: list[Tensor],
    *,
    latest_stage: int,
    background_class: int,
    confidence_threshold: float,
    mask_threshold: float,
    minimum_mask_support: int,
) -> LocalInstanceObservation:
    if not isinstance(latest_stage, int) or isinstance(latest_stage, bool):
        raise ValueError("latest_stage must be an integer")  # noqa: TRY004
    for name, threshold in (
        ("confidence_threshold", confidence_threshold),
        ("mask_threshold", mask_threshold),
    ):
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0.0 <= threshold <= 1.0
            or not math.isfinite(threshold)
        ):
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if (
        not isinstance(minimum_mask_support, int)
        or isinstance(minimum_mask_support, bool)
        or minimum_mask_support <= 0
    ):
        raise ValueError("minimum_mask_support must be a positive integer")

    if not isinstance(outputs, Mapping):
        raise ValueError("outputs must be a mapping")  # noqa: TRY004
    required_keys = ("query_features", "pred_logits", "pred_masks")
    for key in required_keys:
        if key not in outputs:
            raise ValueError(f"outputs is missing required key: {key}")

    features = outputs["query_features"]
    logits = outputs["pred_logits"]
    mask_logits_by_batch = outputs["pred_masks"]
    if not isinstance(features, Tensor):
        raise ValueError("query_features must be a tensor")  # noqa: TRY004
    if not isinstance(logits, Tensor):
        raise ValueError("pred_logits must be a tensor")  # noqa: TRY004
    if features.ndim != 3:
        raise ValueError("query_features must have shape [B, Q, D]")
    if logits.ndim != 3:
        raise ValueError("pred_logits must have shape [B, Q, C]")
    if features.shape[:2] != logits.shape[:2]:
        raise ValueError("query_features and pred_logits must share B and Q")
    if logits.shape[2] < 2:
        raise ValueError("pred_logits must contain at least two classes")
    if (
        not isinstance(background_class, int)
        or isinstance(background_class, bool)
        or not 0 <= background_class < logits.shape[2]
    ):
        raise ValueError("background_class must index the class dimension")
    if not features.is_floating_point():
        raise ValueError("query_features must have a floating dtype")
    if not logits.is_floating_point():
        raise ValueError("pred_logits must have a floating dtype")

    batch_size, query_count = features.shape[:2]
    if not isinstance(mask_logits_by_batch, list):
        raise ValueError("pred_masks must be a list")  # noqa: TRY004
    if len(mask_logits_by_batch) != batch_size:
        raise ValueError("pred_masks must contain one tensor per batch item")
    if not isinstance(segment_stages, list):
        raise ValueError("segment_stages must be a list")  # noqa: TRY004
    if len(segment_stages) != batch_size:
        raise ValueError("segment_stages must contain one tensor per batch item")

    expected_device = features.device
    if logits.device != expected_device:
        raise ValueError("all observation inputs must use the same device")
    for batch_index, (mask_logits, stages) in enumerate(
        zip(mask_logits_by_batch, segment_stages, strict=True)
    ):
        if not isinstance(mask_logits, Tensor):
            raise ValueError(  # noqa: TRY004
                f"pred_masks[{batch_index}] must be a tensor"
            )
        if not isinstance(stages, Tensor):
            raise ValueError(  # noqa: TRY004
                f"segment_stages[{batch_index}] must be a tensor"
            )
        if mask_logits.ndim != 2 or mask_logits.shape[1] != query_count:
            raise ValueError(
                f"pred_masks[{batch_index}] must have shape [S, Q]"
            )
        if stages.ndim != 1 or stages.shape[0] != mask_logits.shape[0]:
            raise ValueError(
                f"segment_stages[{batch_index}] must have shape [S]"
            )
        if not mask_logits.is_floating_point():
            raise ValueError(
                f"pred_masks[{batch_index}] must have a floating dtype"
            )
        if (
            mask_logits.device != expected_device
            or stages.device != expected_device
        ):
            raise ValueError("all observation inputs must use the same device")

    finite_inputs = (("query_features", features), ("pred_logits", logits))
    for name, tensor in finite_inputs:
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")
    for batch_index, (mask_logits, stages) in enumerate(
        zip(mask_logits_by_batch, segment_stages, strict=True)
    ):
        if not torch.isfinite(mask_logits).all().item():
            raise ValueError(
                f"pred_masks[{batch_index}] must contain only finite values"
            )
        if not torch.isfinite(stages).all().item():
            raise ValueError(
                f"segment_stages[{batch_index}] must contain only finite values"
            )

    class_prob = logits.softmax(dim=-1)
    foreground_prob = torch.cat(
        (
            class_prob[..., :background_class],
            class_prob[..., background_class + 1 :],
        ),
        dim=-1,
    )
    confidence = foreground_prob.amax(dim=-1)

    latest_mask: list[Tensor] = []
    mask_support: list[Tensor] = []
    for batch_index, (mask_logits, stages) in enumerate(
        zip(mask_logits_by_batch, segment_stages, strict=True)
    ):
        latest_selector = stages == latest_stage
        if not torch.any(latest_selector).item():
            raise ValueError(
                f"segment_stages[{batch_index}] does not contain latest_stage"
            )
        selected_mask = mask_logits[latest_selector].transpose(0, 1)
        latest_mask.append(selected_mask)
        mask_support.append(
            (selected_mask.sigmoid() >= mask_threshold).sum(dim=1)
        )

    if mask_support:
        support = torch.stack(mask_support)
    else:
        support = torch.empty_like(confidence, dtype=torch.long)
    valid = (confidence >= confidence_threshold) & (
        support >= minimum_mask_support
    )
    observation = LocalInstanceObservation(
        features=features,
        class_prob=class_prob,
        confidence=confidence,
        latest_mask=latest_mask,
        valid=valid,
    )
    observation.validate()
    return observation


@dataclass(frozen=True)
class PersistentMemoryState:
    embedding: Tensor
    class_prob: Tensor
    confidence: Tensor
    occupied: Tensor
    active: Tensor
    age: Tensor
    last_seen: Tensor

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        capacity: int,
        feature_dim: int,
        class_count: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> PersistentMemoryState:
        dimensions = {
            "batch_size": batch_size,
            "capacity": capacity,
            "feature_dim": feature_dim,
            "class_count": class_count,
        }
        for name, value in dimensions.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be a floating torch dtype")  # noqa: TRY004
        try:
            dtype_probe = torch.empty((), dtype=dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError("dtype must be a floating torch dtype") from error
        if not dtype_probe.is_floating_point():
            raise ValueError("dtype must be a floating torch dtype")

        try:
            target_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a valid torch device") from error

        state = cls(
            embedding=torch.zeros(
                batch_size,
                capacity,
                feature_dim,
                device=target_device,
                dtype=dtype,
            ),
            class_prob=torch.zeros(
                batch_size,
                capacity,
                class_count,
                device=target_device,
                dtype=dtype,
            ),
            confidence=torch.zeros(
                batch_size, capacity, device=target_device, dtype=dtype
            ),
            occupied=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.bool
            ),
            active=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.bool
            ),
            age=torch.zeros(
                batch_size, capacity, device=target_device, dtype=torch.long
            ),
            last_seen=torch.full(
                (batch_size, capacity),
                -1,
                device=target_device,
                dtype=torch.long,
            ),
        )
        state.validate()
        return state

    @property
    def batch_size(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[0]

    @property
    def capacity(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[1]

    @property
    def feature_dim(self) -> int:
        if not isinstance(self.embedding, Tensor) or self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        return self.embedding.shape[2]

    @property
    def class_count(self) -> int:
        if not isinstance(self.class_prob, Tensor) or self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, K, C]")
        return self.class_prob.shape[2]

    def validate(self) -> None:
        tensors = (
            ("embedding", self.embedding),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
            ("occupied", self.occupied),
            ("active", self.active),
            ("age", self.age),
            ("last_seen", self.last_seen),
        )
        for name, tensor in tensors:
            if not isinstance(tensor, Tensor):
                raise ValueError(f"{name} must be a tensor")  # noqa: TRY004

        if self.embedding.ndim != 3:
            raise ValueError("embedding must have shape [B, K, D]")
        if self.class_prob.ndim != 3:
            raise ValueError("class_prob must have shape [B, K, C]")
        for name, tensor in tensors[2:]:
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B, K]")

        batch_size, capacity, feature_dim = self.embedding.shape
        if batch_size <= 0 or capacity <= 0 or feature_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if self.class_prob.shape[:2] != (batch_size, capacity):
            raise ValueError("class_prob must match embedding batch and capacity")
        if self.class_prob.shape[2] <= 0:
            raise ValueError("class_prob must have a positive class dimension")
        expected_shape = (batch_size, capacity)
        for name, tensor in tensors[2:]:
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} must match embedding batch and capacity")

        expected_device = self.embedding.device
        if any(tensor.device != expected_device for _, tensor in tensors[1:]):
            raise ValueError("all state tensors must be on the same device")

        float_dtype = self.embedding.dtype
        if not self.embedding.is_floating_point():
            raise ValueError("embedding must have a floating dtype")
        for name, tensor in (
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
        ):
            if not tensor.is_floating_point() or tensor.dtype != float_dtype:
                raise ValueError(
                    f"{name} must use the same floating dtype as embedding"
                )
        if self.occupied.dtype != torch.bool or self.active.dtype != torch.bool:
            raise ValueError("occupied and active must have bool dtype")
        if self.age.dtype != torch.long or self.last_seen.dtype != torch.long:
            raise ValueError("age and last_seen must have long dtype")

        for name, tensor in tensors[:3]:
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if torch.any(self.active & ~self.occupied).item():
            raise ValueError("active slots must also be occupied")
        if torch.any(self.age < 0).item():
            raise ValueError("age must be non-negative")
        if torch.any(self.last_seen < -1).item():
            raise ValueError("last_seen must be at least -1")

    def detach(self) -> PersistentMemoryState:
        return PersistentMemoryState(*(tensor.detach() for tensor in self.tensors()))

    def tensors(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.embedding,
            self.class_prob,
            self.confidence,
            self.occupied,
            self.active,
            self.age,
            self.last_seen,
        )


@dataclass(frozen=True)
class AssociationResult:
    slot_for_query: Tensor
    query_for_slot: Tensor
    score_for_query: Tensor


def _score_as_exact_integers(score: Tensor) -> list[list[int]]:
    ratios = [[value.as_integer_ratio() for value in row] for row in score.tolist()]
    # IEEE float denominators are powers of two, so the maximum is common.
    common_denominator = max(denominator for row in ratios for _, denominator in row)
    return [
        [
            numerator * (common_denominator // denominator)
            for numerator, denominator in row
        ]
        for row in ratios
    ]


def _optimal_assignment_with_stable_ties(
    score: Tensor,
) -> tuple[Tensor, Tensor]:
    row_count, column_count = score.shape
    size = max(row_count, column_count)
    padded_score = torch.zeros(size, size, dtype=torch.float64)
    padded_score[:row_count, :column_count] = score.detach().to(
        device="cpu", dtype=torch.float64
    )

    initial_rows, initial_columns = linear_sum_assignment(-padded_score.numpy())
    matched_column = [-1] * size
    for row, column in zip(initial_rows, initial_columns, strict=True):
        matched_column[row] = column

    score_units = _score_as_exact_integers(padded_score)
    # Matched-edge difference constraints recover an exact optimal dual.
    column_potential = [0] * size
    for _ in range(size - 1):
        updated = False
        for row, row_score in enumerate(score_units):
            matched = matched_column[row]
            matched_score = row_score[matched]
            matched_potential = column_potential[matched]
            for column, pair_score in enumerate(row_score):
                upper_bound = matched_potential + matched_score - pair_score
                if column_potential[column] > upper_bound:
                    column_potential[column] = upper_bound
                    updated = True
        if not updated:
            break

    tie_cost = torch.full((size, size), torch.inf, dtype=torch.float64)
    for row, row_score in enumerate(score_units):
        matched = matched_column[row]
        matched_score = row_score[matched]
        matched_potential = column_potential[matched]
        for column, pair_score in enumerate(row_score):
            if column_potential[column] != (
                matched_potential + matched_score - pair_score
            ):
                continue
            if row < row_count and column < column_count:
                tie_cost[row, column] = abs(row - column)
            else:
                tie_cost[row, column] = 0.0

    # Every complete matching on these exact dual-tight edges has the
    # original optimal total, so this secondary objective cannot change it.
    assigned_rows, assigned_columns = linear_sum_assignment(tie_cost.numpy())
    assigned_rows = torch.from_numpy(assigned_rows)
    assigned_columns = torch.from_numpy(assigned_columns)
    real_pair = (assigned_rows < row_count) & (assigned_columns < column_count)
    return assigned_rows[real_pair], assigned_columns[real_pair]


def associate_observations(
    observation: LocalInstanceObservation,
    state: PersistentMemoryState,
    *,
    class_weight: float,
    association_threshold: float,
) -> AssociationResult:
    observation.validate()
    state.validate()

    if observation.features.shape[0] != state.batch_size:
        raise ValueError("observation and state batch sizes must match")
    if observation.features.shape[2] != state.feature_dim:
        raise ValueError("observation and state feature dimensions must match")
    if observation.class_prob.shape[2] != state.class_count:
        raise ValueError("observation and state class dimensions must match")
    if observation.features.device != state.embedding.device:
        raise ValueError("observation and state must use the same device")

    if (
        not isinstance(class_weight, (int, float))
        or isinstance(class_weight, bool)
        or not math.isfinite(class_weight)
        or class_weight < 0.0
    ):
        raise ValueError("class_weight must be finite and non-negative")
    if (
        not isinstance(association_threshold, (int, float))
        or isinstance(association_threshold, bool)
        or not math.isfinite(association_threshold)
    ):
        raise ValueError("association_threshold must be finite")

    score_dtype = observation.features.dtype
    for dtype in (
        observation.class_prob.dtype,
        state.embedding.dtype,
        state.class_prob.dtype,
    ):
        score_dtype = torch.promote_types(score_dtype, dtype)
    if score_dtype in (torch.float16, torch.bfloat16):
        score_dtype = torch.float32

    query_features = observation.features.to(dtype=score_dtype)
    memory_features = state.embedding.to(dtype=score_dtype)
    normalization_floor = torch.finfo(score_dtype).tiny
    query_features = query_features / torch.linalg.vector_norm(
        query_features, dim=-1, keepdim=True
    ).clamp_min(normalization_floor)
    memory_features = memory_features / torch.linalg.vector_norm(
        memory_features, dim=-1, keepdim=True
    ).clamp_min(normalization_floor)

    cosine_score = torch.einsum("bkd,bqd->bkq", memory_features, query_features)
    class_score = torch.einsum(
        "bkc,bqc->bkq",
        state.class_prob.to(dtype=score_dtype),
        observation.class_prob.to(dtype=score_dtype),
    )
    score = cosine_score + class_weight * class_score

    batch_size, query_count = observation.valid.shape
    capacity = state.capacity
    device = observation.features.device
    slot_for_query = torch.full(
        (batch_size, query_count), -1, dtype=torch.long, device=device
    )
    query_for_slot = torch.full(
        (batch_size, capacity), -1, dtype=torch.long, device=device
    )
    score_for_query = score.new_full((batch_size, query_count), -torch.inf)

    for batch_index in range(batch_size):
        occupied_slots = state.occupied[batch_index].nonzero(as_tuple=True)[0]
        valid_queries = observation.valid[batch_index].nonzero(as_tuple=True)[0]
        if occupied_slots.numel() == 0 or valid_queries.numel() == 0:
            continue

        candidate_score = score[batch_index][occupied_slots][:, valid_queries]
        assigned_rows, assigned_columns = _optimal_assignment_with_stable_ties(
            candidate_score
        )

        assigned_rows = assigned_rows.to(device=device)
        assigned_columns = assigned_columns.to(device=device)
        assigned_scores = candidate_score[assigned_rows, assigned_columns]
        accepted = assigned_scores >= association_threshold
        accepted_slots = occupied_slots[assigned_rows[accepted]]
        accepted_queries = valid_queries[assigned_columns[accepted]]
        accepted_scores = assigned_scores[accepted]

        slot_for_query[batch_index, accepted_queries] = accepted_slots
        query_for_slot[batch_index, accepted_slots] = accepted_queries
        score_for_query[batch_index, accepted_queries] = accepted_scores

    return AssociationResult(
        slot_for_query=slot_for_query,
        query_for_slot=query_for_slot,
        score_for_query=score_for_query,
    )
