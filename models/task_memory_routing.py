from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from datasets.task_memory_episode import StageMeta
from models.persistent_memory import (
    LocalInstanceObservation,
    _normalize_feature_vectors,
    associate_observations,
)
from models.task_memory_state import TaskMemoryState


class TaskMemoryRoutingError(ValueError):
    """Raised when route or commit inputs violate the causal state contract."""


@dataclass(frozen=True)
class PredictionObservation:
    features: Tensor
    class_prob: Tensor
    confidence: Tensor
    valid: Tensor
    current_supported: Tensor
    previous_supported: Tensor

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def query_count(self) -> int:
        return self.features.shape[1]

    def validate(self) -> None:
        tensors = (
            ("features", self.features),
            ("class_prob", self.class_prob),
            ("confidence", self.confidence),
            ("valid", self.valid),
            ("current_supported", self.current_supported),
            ("previous_supported", self.previous_supported),
        )
        if any(not isinstance(value, Tensor) for _, value in tensors):
            raise TaskMemoryRoutingError("all observation fields must be tensors")
        if self.features.ndim != 3 or self.class_prob.ndim != 3:
            raise TaskMemoryRoutingError("features and class_prob must have shape [B,Q,*]")
        if self.features.shape[:2] != self.class_prob.shape[:2]:
            raise TaskMemoryRoutingError("observation batch/query dimensions differ")
        if self.features.shape[2] <= 0 or self.class_prob.shape[2] <= 0:
            raise TaskMemoryRoutingError("feature and class dimensions must be positive")
        expected = self.features.shape[:2]
        for name, value in tensors[2:]:
            if value.shape != expected:
                raise TaskMemoryRoutingError(f"{name} must have shape [B,Q]")
        if any(value.dtype != torch.bool for _, value in tensors[3:]):
            raise TaskMemoryRoutingError("support and validity tensors must be bool")
        if not self.features.is_floating_point():
            raise TaskMemoryRoutingError("features must use a floating dtype")
        if any(
            not value.is_floating_point() or value.dtype != self.features.dtype
            for _, value in tensors[1:3]
        ):
            raise TaskMemoryRoutingError(
                "features, class_prob, and confidence must share a floating dtype"
            )
        if any(value.device != self.features.device for _, value in tensors[1:]):
            raise TaskMemoryRoutingError("all observation tensors must share a device")
        for name, value in tensors[:3]:
            if not torch.isfinite(value).all().item():
                raise TaskMemoryRoutingError(f"{name} must contain finite values")
        if torch.any(self.class_prob < 0).item():
            raise TaskMemoryRoutingError("class_prob must be non-negative")
        if torch.any((self.confidence < 0) | (self.confidence > 1)).item():
            raise TaskMemoryRoutingError("confidence must be within [0,1]")
        if torch.any(
            self.valid & ~(self.current_supported | self.previous_supported)
        ).item():
            raise TaskMemoryRoutingError("valid queries must have window support")

    def as_local_instance(self) -> LocalInstanceObservation:
        self.validate()
        masks = [
            self.current_supported[index].to(dtype=self.features.dtype).unsqueeze(-1)
            for index in range(self.batch_size)
        ]
        observation = LocalInstanceObservation(
            features=self.features,
            class_prob=self.class_prob,
            confidence=self.confidence,
            latest_mask=masks,
            valid=self.valid,
        )
        observation.validate()
        return observation


@dataclass(frozen=True)
class EntityRoute:
    query_to_slot: Tensor
    slot_to_query: Tensor
    route_score: Tensor
    prior_logical_id: Tensor
    prior_generation: Tensor
    current_supported: Tensor
    previous_supported: Tensor
    valid: Tensor
    stage_indices: Tensor
    source_state_sha256: str
    metadata_sha256: str
    content_sha256: str

    def validate(self) -> None:
        if self.query_to_slot.ndim != 2 or self.slot_to_query.ndim != 2:
            raise TaskMemoryRoutingError("route maps must have shape [B,Q] and [B,K]")
        batch_size, query_count = self.query_to_slot.shape
        capacity = self.slot_to_query.shape[1]
        if self.slot_to_query.shape[0] != batch_size:
            raise TaskMemoryRoutingError("route batch dimensions differ")
        for name, value in (
            ("route_score", self.route_score),
            ("prior_logical_id", self.prior_logical_id),
            ("prior_generation", self.prior_generation),
            ("current_supported", self.current_supported),
            ("previous_supported", self.previous_supported),
            ("valid", self.valid),
        ):
            if value.shape != (batch_size, query_count):
                raise TaskMemoryRoutingError(f"{name} must have shape [B,Q]")
        for value in (
            self.query_to_slot,
            self.slot_to_query,
            self.prior_logical_id,
            self.prior_generation,
            self.stage_indices,
        ):
            if value.dtype != torch.long:
                raise TaskMemoryRoutingError("route indices must use int64")
        for value in (self.current_supported, self.previous_supported, self.valid):
            if value.dtype != torch.bool:
                raise TaskMemoryRoutingError("route support fields must use bool")
        if self.stage_indices.shape != (batch_size,):
            raise TaskMemoryRoutingError("stage_indices must have shape [B]")
        matched = self.query_to_slot >= 0
        if torch.any(self.query_to_slot >= capacity).item():
            raise TaskMemoryRoutingError("query route exceeds state capacity")
        if torch.any(self.slot_to_query >= query_count).item():
            raise TaskMemoryRoutingError("slot route exceeds query count")
        if torch.any(matched & ~self.valid).item():
            raise TaskMemoryRoutingError("invalid query cannot inherit a route")
        if not torch.isfinite(self.route_score[matched]).all().item():
            raise TaskMemoryRoutingError("matched route scores must be finite")
        if not torch.isneginf(self.route_score[~matched]).all().item():
            raise TaskMemoryRoutingError("unmatched route scores must be -inf")
        if torch.any(matched & (self.prior_logical_id < 0)).item() or torch.any(
            matched & (self.prior_generation < 0)
        ).item():
            raise TaskMemoryRoutingError("matched routes require prior identity")
        if torch.any(~matched & (self.prior_logical_id != -1)).item() or torch.any(
            ~matched & (self.prior_generation != -1)
        ).item():
            raise TaskMemoryRoutingError("unmatched routes must use identity sentinels")
        for batch_index in range(batch_size):
            for query_index in matched[batch_index].nonzero(as_tuple=True)[0].tolist():
                slot = int(self.query_to_slot[batch_index, query_index].item())
                if int(self.slot_to_query[batch_index, slot].item()) != query_index:
                    raise TaskMemoryRoutingError("route maps are not inverse")
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in (
                self.source_state_sha256,
                self.metadata_sha256,
                self.content_sha256,
            )
        ):
            raise TaskMemoryRoutingError("route commitments must be SHA256 strings")
        if self.content_sha256 != _route_sha256(self):
            raise TaskMemoryRoutingError("route content commitment differs")


@dataclass(frozen=True)
class CommitResult:
    state: TaskMemoryState
    query_to_slot: Tensor
    query_to_logical_id: Tensor
    query_to_generation: Tensor
    births: Tensor
    rejected_births: Tensor
    active: Tensor
    route_commitment: str

    def identity_map(self, batch_index: int = 0) -> dict[int, tuple[int, int]]:
        if not 0 <= batch_index < self.query_to_logical_id.shape[0]:
            raise TaskMemoryRoutingError("batch_index is outside the commit result")
        return {
            query: (
                int(self.query_to_logical_id[batch_index, query].item()),
                int(self.query_to_generation[batch_index, query].item()),
            )
            for query in range(self.query_to_logical_id.shape[1])
            if self.query_to_logical_id[batch_index, query] >= 0
        }


def _update_hash_with_tensor(digest: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _state_sha256(state: TaskMemoryState) -> str:
    state.validate()
    digest = hashlib.sha256()
    for tensor in state.tensors():
        _update_hash_with_tensor(digest, tensor)
    digest.update(
        json.dumps(
            {
                "association_threshold": state.config.association_threshold,
                "class_weight": state.config.class_weight,
                "max_update_rate": state.config.max_update_rate,
                "update_mode": state.config.update_mode,
                "update_rate": state.config.update_rate,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return digest.hexdigest()


def _metadata_sha256(stage_meta: Sequence[StageMeta]) -> str:
    payload = [
        {
            "absolute_stage_index": item.absolute_stage_index,
            "augmentation_transform_id": item.augmentation_transform_id,
            "coordinate_frame_id": item.coordinate_frame_id,
            "episode_id": item.episode_id,
            "reference_id": item.reference_id,
            "scan_ids_in_window": list(item.scan_ids_in_window),
        }
        for item in stage_meta
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_ROUTE_TENSOR_FIELDS = (
    "query_to_slot",
    "slot_to_query",
    "route_score",
    "prior_logical_id",
    "prior_generation",
    "current_supported",
    "previous_supported",
    "valid",
    "stage_indices",
)


def _route_sha256(route: EntityRoute) -> str:
    digest = hashlib.sha256()
    for name in _ROUTE_TENSOR_FIELDS:
        _update_hash_with_tensor(digest, getattr(route, name))
    digest.update(route.source_state_sha256.encode("ascii"))
    digest.update(route.metadata_sha256.encode("ascii"))
    return digest.hexdigest()


def _validate_stage_meta(
    stage_meta: Sequence[StageMeta], state: TaskMemoryState
) -> Tensor:
    if not isinstance(stage_meta, (list, tuple)) or len(stage_meta) != state.batch_size:
        raise TaskMemoryRoutingError("stage metadata must contain one item per batch")
    if any(not isinstance(item, StageMeta) for item in stage_meta):
        raise TaskMemoryRoutingError("stage metadata entries must be StageMeta")
    indices = torch.tensor(
        [item.absolute_stage_index for item in stage_meta],
        dtype=torch.long,
        device=state.embedding.device,
    )
    if torch.any(indices <= state.stage_watermark).item():
        raise TaskMemoryRoutingError(
            "stage metadata must advance each state watermark exactly once"
        )
    return indices


def _validate_observation_state(
    observation: PredictionObservation, state: TaskMemoryState
) -> None:
    observation.validate()
    state.validate()
    if observation.batch_size != state.batch_size:
        raise TaskMemoryRoutingError("observation and state batch sizes differ")
    if observation.features.shape[2] != state.feature_dim:
        raise TaskMemoryRoutingError("observation and state feature dimensions differ")
    if observation.class_prob.shape[2] != state.class_count:
        raise TaskMemoryRoutingError("observation and state class dimensions differ")
    if observation.features.device != state.embedding.device:
        raise TaskMemoryRoutingError("observation and state devices differ")
    if observation.features.dtype != state.embedding.dtype:
        raise TaskMemoryRoutingError("observation and state dtypes differ")


def route_entities(
    pre_output: PredictionObservation,
    old_state: TaskMemoryState,
    stage_meta: Sequence[StageMeta],
) -> EntityRoute:
    _validate_observation_state(pre_output, old_state)
    stage_indices = _validate_stage_meta(stage_meta, old_state)
    association = associate_observations(
        pre_output.as_local_instance(),
        old_state.as_persistent_state(),
        class_weight=old_state.config.class_weight,
        association_threshold=old_state.config.association_threshold,
    )
    query_to_slot = association.slot_for_query.detach().clone()
    slot_to_query = association.query_for_slot.detach().clone()
    route_score = association.score_for_query.detach().clone()
    prior_logical_id = torch.full_like(query_to_slot, -1)
    prior_generation = torch.full_like(query_to_slot, -1)
    for batch_index in range(old_state.batch_size):
        matched = (query_to_slot[batch_index] >= 0).nonzero(as_tuple=True)[0]
        if matched.numel():
            slots = query_to_slot[batch_index, matched]
            prior_logical_id[batch_index, matched] = old_state.logical_ids[
                batch_index, slots
            ]
            prior_generation[batch_index, matched] = old_state.generations[
                batch_index, slots
            ]
    values: dict[str, Any] = {
        "query_to_slot": query_to_slot,
        "slot_to_query": slot_to_query,
        "route_score": route_score,
        "prior_logical_id": prior_logical_id,
        "prior_generation": prior_generation,
        "current_supported": pre_output.current_supported.detach().clone(),
        "previous_supported": pre_output.previous_supported.detach().clone(),
        "valid": pre_output.valid.detach().clone(),
        "stage_indices": stage_indices.detach().clone(),
        "source_state_sha256": _state_sha256(old_state),
        "metadata_sha256": _metadata_sha256(stage_meta),
    }
    provisional = EntityRoute(content_sha256="0" * 64, **values)
    route = EntityRoute(content_sha256=_route_sha256(provisional), **values)
    route.validate()
    return route


def _update_rates(final_output: PredictionObservation, state: TaskMemoryState, batch: int, queries: Tensor) -> Tensor:
    compute_dtype = torch.float64 if state.embedding.dtype == torch.float64 else torch.float32
    if state.config.update_mode == "last":
        return torch.ones(queries.numel(), device=queries.device, dtype=compute_dtype)
    if state.config.update_mode == "fixed_ema":
        return torch.full(
            (queries.numel(),),
            state.config.update_rate,
            device=queries.device,
            dtype=compute_dtype,
        )
    return (
        state.config.update_rate
        * final_output.confidence[batch, queries].to(dtype=compute_dtype)
    ).clamp(min=0.0, max=state.config.max_update_rate)


def commit_entities(
    final_output: PredictionObservation,
    route: EntityRoute,
    old_state: TaskMemoryState,
    stage_meta: Sequence[StageMeta],
) -> CommitResult:
    _validate_observation_state(final_output, old_state)
    route.validate()
    stage_indices = _validate_stage_meta(stage_meta, old_state)
    if route.source_state_sha256 != _state_sha256(old_state):
        raise TaskMemoryRoutingError("route commitment belongs to a different state")
    if route.metadata_sha256 != _metadata_sha256(stage_meta):
        raise TaskMemoryRoutingError("route metadata commitment differs")
    if not torch.equal(route.stage_indices, stage_indices):
        raise TaskMemoryRoutingError("route stage indices differ from metadata")
    expected_query_shape = (old_state.batch_size, final_output.query_count)
    if route.query_to_slot.shape != expected_query_shape:
        raise TaskMemoryRoutingError("final output query count differs from route")

    (
        embedding,
        class_prob,
        confidence,
        occupied,
        active,
        age,
        last_seen,
        watermark,
        logical_ids,
        generations,
        next_logical_id,
    ) = (tensor.clone() for tensor in old_state.tensors())
    age.add_(occupied.to(dtype=torch.long))
    active.zero_()
    query_to_slot = route.query_to_slot.clone()
    query_to_logical_id = route.prior_logical_id.clone()
    query_to_generation = route.prior_generation.clone()
    births = torch.zeros_like(final_output.valid)
    rejected_births = torch.zeros_like(final_output.valid)

    for batch_index in range(old_state.batch_size):
        update_queries = (
            (route.query_to_slot[batch_index] >= 0)
            & final_output.valid[batch_index]
            & final_output.current_supported[batch_index]
        ).nonzero(as_tuple=True)[0]
        if update_queries.numel():
            update_slots = route.query_to_slot[batch_index, update_queries]
            compute_dtype = (
                torch.float64
                if old_state.embedding.dtype == torch.float64
                else torch.float32
            )
            rates = _update_rates(
                final_output, old_state, batch_index, update_queries
            )
            vector_rates = rates.unsqueeze(-1)
            mixed_embedding = (
                (1.0 - vector_rates)
                * old_state.embedding[batch_index, update_slots].to(
                    dtype=compute_dtype
                )
                + vector_rates
                * final_output.features[batch_index, update_queries].to(
                    dtype=compute_dtype
                )
            )
            embedding[batch_index, update_slots] = _normalize_feature_vectors(
                mixed_embedding
            ).to(dtype=old_state.embedding.dtype)
            class_prob[batch_index, update_slots] = (
                (1.0 - vector_rates)
                * old_state.class_prob[batch_index, update_slots].to(
                    dtype=compute_dtype
                )
                + vector_rates
                * final_output.class_prob[batch_index, update_queries].to(
                    dtype=compute_dtype
                )
            ).to(dtype=old_state.class_prob.dtype)
            confidence[batch_index, update_slots] = (
                (1.0 - rates)
                * old_state.confidence[batch_index, update_slots].to(
                    dtype=compute_dtype
                )
                + rates
                * final_output.confidence[batch_index, update_queries].to(
                    dtype=compute_dtype
                )
            ).to(dtype=old_state.confidence.dtype)
            active[batch_index, update_slots] = True
            last_seen[batch_index, update_slots] = stage_indices[batch_index]

        birth_candidates = (
            (route.query_to_slot[batch_index] < 0)
            & final_output.valid[batch_index]
            & final_output.current_supported[batch_index]
        ).nonzero(as_tuple=True)[0].tolist()
        birth_candidates.sort(
            key=lambda query: (
                -float(final_output.confidence[batch_index, query].item()),
                query,
            )
        )
        free_slots = (~occupied[batch_index]).nonzero(as_tuple=True)[0].tolist()
        accepted_count = min(len(birth_candidates), len(free_slots))
        for query, slot in zip(
            birth_candidates[:accepted_count], free_slots[:accepted_count]
        ):
            embedding[batch_index, slot] = _normalize_feature_vectors(
                final_output.features[batch_index, query].unsqueeze(0)
            )[0].to(dtype=old_state.embedding.dtype)
            class_prob[batch_index, slot] = final_output.class_prob[
                batch_index, query
            ]
            confidence[batch_index, slot] = final_output.confidence[
                batch_index, query
            ]
            occupied[batch_index, slot] = True
            active[batch_index, slot] = True
            age[batch_index, slot] = 0
            last_seen[batch_index, slot] = stage_indices[batch_index]
            logical_id = int(next_logical_id[batch_index].item())
            logical_ids[batch_index, slot] = logical_id
            generations[batch_index, slot] = 0
            next_logical_id[batch_index] += 1
            query_to_slot[batch_index, query] = slot
            query_to_logical_id[batch_index, query] = logical_id
            query_to_generation[batch_index, query] = 0
            births[batch_index, query] = True
        if accepted_count < len(birth_candidates):
            rejected_births[
                batch_index,
                torch.tensor(
                    birth_candidates[accepted_count:],
                    device=old_state.embedding.device,
                    dtype=torch.long,
                ),
            ] = True
        watermark[batch_index] = stage_indices[batch_index]

    next_state = TaskMemoryState(
        embedding=embedding,
        class_prob=class_prob,
        confidence=confidence,
        occupied=occupied,
        active=active,
        age=age,
        last_seen=last_seen,
        stage_watermark=watermark,
        logical_ids=logical_ids,
        generations=generations,
        next_logical_id=next_logical_id,
        config=old_state.config,
    )
    next_state.validate()
    return CommitResult(
        state=next_state,
        query_to_slot=query_to_slot,
        query_to_logical_id=query_to_logical_id,
        query_to_generation=query_to_generation,
        births=births,
        rejected_births=rejected_births,
        active=active.clone(),
        route_commitment=route.content_sha256,
    )


__all__ = [
    "CommitResult",
    "EntityRoute",
    "PredictionObservation",
    "TaskMemoryRoutingError",
    "commit_entities",
    "route_entities",
]
