"""Training-only identity ledger and tracklet-aware loss assignment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

from datasets.task_memory_episode import StageMeta
from models.task_memory_routing import CommitResult, EntityRoute, _metadata_sha256

QualifiedIdentity = tuple[str, int]
IndexPair = tuple[Tensor, Tensor]


class TaskMemorySupervisionError(ValueError):
    """Raised when training-only identity supervision is inconsistent."""


def _empty_indices() -> IndexPair:
    empty = torch.empty(0, dtype=torch.long)
    return empty, empty.clone()


def _index_pair(value: object, *, name: str) -> IndexPair:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(not isinstance(item, Tensor) for item in value)
    ):
        raise TaskMemorySupervisionError(f"{name} must be a tensor index pair")
    source, target = value
    if (
        source.ndim != 1
        or target.ndim != 1
        or source.dtype != torch.long
        or target.dtype != torch.long
        or source.numel() != target.numel()
    ):
        raise TaskMemorySupervisionError(f"{name} index tensors are invalid")
    return source.detach().cpu().clone(), target.detach().cpu().clone()


def _validate_index_bounds(
    pair: IndexPair,
    *,
    query_count: int,
    target_count: int,
    name: str,
) -> IndexPair:
    source, target = pair
    if (
        torch.any(source < 0).item()
        or torch.any(source >= query_count).item()
        or torch.any(target < 0).item()
        or torch.any(target >= target_count).item()
    ):
        raise TaskMemorySupervisionError(f"{name} index is out of range")
    if (
        source.unique().numel() != source.numel()
        or target.unique().numel() != target.numel()
    ):
        raise TaskMemorySupervisionError(f"{name} indices must be one-to-one")
    return pair


def _qualified_identity(value: object) -> QualifiedIdentity:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], str)
        or not value[0]
        or isinstance(value[1], bool)
        or not isinstance(value[1], int)
        or value[1] < 0
    ):
        raise TaskMemorySupervisionError("qualified identity is invalid")
    return value


def ambiguous_identity_ids(metadata: object) -> frozenset[int]:
    """Return IDs belonging to a non-singleton official ambiguity group."""
    if metadata is None:
        return frozenset()
    alternatives = getattr(metadata, "alternatives", None)
    if alternatives is not None:
        if not isinstance(alternatives, Mapping):
            raise TaskMemorySupervisionError("ambiguity alternatives are invalid")
        groups: object = list(alternatives.values())
    elif isinstance(metadata, Mapping):
        if set(metadata) == {"ambiguities"}:
            groups = metadata["ambiguities"]
        else:
            groups = list(metadata.values())
    else:
        groups = metadata
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise TaskMemorySupervisionError("ambiguity metadata must contain groups")
    result: set[int] = set()
    for group in groups:
        if isinstance(group, (str, bytes)) or not isinstance(group, Sequence):
            raise TaskMemorySupervisionError("ambiguity group must be a sequence")
        normalized = []
        for value in group:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TaskMemorySupervisionError("ambiguity identity is invalid")
            normalized.append(value)
        if len(set(normalized)) > 1:
            result.update(normalized)
    return frozenset(result)


class TrainingIdentityLedger:
    """Per-episode mapping from predicted logical identities to qualified GT IDs."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[int, int], QualifiedIdentity] = {}

    @property
    def bindings(self) -> Mapping[tuple[int, int], QualifiedIdentity]:
        return MappingProxyType(dict(self._bindings))

    def bind(
        self,
        logical_id: int,
        generation: int,
        identity: QualifiedIdentity,
    ) -> None:
        if (
            isinstance(logical_id, bool)
            or not isinstance(logical_id, int)
            or logical_id < 0
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise TaskMemorySupervisionError("logical identity is invalid")
        identity = _qualified_identity(identity)
        key = (logical_id, generation)
        existing = self._bindings.get(key)
        if existing is not None and existing != identity:
            raise TaskMemorySupervisionError("logical identity cannot be rebound")
        self._bindings[key] = identity

    def resolve(self, logical_id: int, generation: int) -> QualifiedIdentity | None:
        return self._bindings.get((logical_id, generation))

    def state_dict(self) -> dict[str, object]:
        return {
            "bindings": [
                {
                    "canonical_instance_id": identity[1],
                    "generation": key[1],
                    "logical_id": key[0],
                    "reference_id": identity[0],
                }
                for key, identity in sorted(self._bindings.items())
            ],
            "schema_version": "task-memory-training-ledger-v1",
        }

    @classmethod
    def from_state_dict(cls, value: Mapping[str, object]) -> TrainingIdentityLedger:
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "bindings",
                "schema_version",
            }
            or value["schema_version"] != "task-memory-training-ledger-v1"
        ):
            raise TaskMemorySupervisionError("training ledger state is invalid")
        records = value["bindings"]
        if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
            raise TaskMemorySupervisionError("training ledger bindings are invalid")
        ledger = cls()
        expected_fields = {
            "canonical_instance_id",
            "generation",
            "logical_id",
            "reference_id",
        }
        for record in records:
            if not isinstance(record, Mapping) or set(record) != expected_fields:
                raise TaskMemorySupervisionError("training ledger binding is invalid")
            ledger.bind(
                record["logical_id"],
                record["generation"],
                (record["reference_id"], record["canonical_instance_id"]),
            )
        if len(ledger.bindings) != len(records):
            raise TaskMemorySupervisionError("training ledger bindings are duplicated")
        return ledger

    def bind_births(
        self,
        commit: CommitResult,
        *,
        batch_index: int,
        assigned_indices: IndexPair,
        identity_keys: Sequence[QualifiedIdentity],
        target_labels: Tensor,
        ambiguity_metadata: object,
    ) -> dict[str, int]:
        if not isinstance(commit, CommitResult):
            raise TaskMemorySupervisionError(
                "ledger birth binding requires CommitResult"
            )
        if (
            isinstance(batch_index, bool)
            or not isinstance(batch_index, int)
            or not 0 <= batch_index < commit.births.shape[0]
        ):
            raise TaskMemorySupervisionError("ledger batch index is invalid")
        source, target = _index_pair(assigned_indices, name="birth assignment")
        if (
            not isinstance(target_labels, Tensor)
            or target_labels.ndim != 1
            or len(identity_keys) != target_labels.numel()
        ):
            raise TaskMemorySupervisionError("birth target identities are misaligned")
        identities = tuple(_qualified_identity(value) for value in identity_keys)
        ambiguous = ambiguous_identity_ids(ambiguity_metadata)
        match_by_query = {
            int(query): int(target_row)
            for query, target_row in zip(source.tolist(), target.tolist(), strict=True)
        }
        diagnostics = {
            "bound_births": 0,
            "ambiguous_births_excluded": 0,
            "ignored_births_excluded": 0,
            "unmatched_births": 0,
        }
        for query in commit.births[batch_index].nonzero(as_tuple=True)[0].tolist():
            target_row = match_by_query.get(query)
            if target_row is None:
                diagnostics["unmatched_births"] += 1
                continue
            if not 0 <= target_row < len(identities):
                raise TaskMemorySupervisionError(
                    "birth assignment target is out of range"
                )
            identity = identities[target_row]
            if int(target_labels[target_row].item()) == 253:
                diagnostics["ignored_births_excluded"] += 1
                continue
            if identity[1] in ambiguous:
                diagnostics["ambiguous_births_excluded"] += 1
                continue
            logical_id = int(commit.query_to_logical_id[batch_index, query].item())
            generation = int(commit.query_to_generation[batch_index, query].item())
            self.bind(logical_id, generation, identity)
            diagnostics["bound_births"] += 1
        return diagnostics


@dataclass(frozen=True)
class TaskMemoryAssignment:
    matcher_mode: str
    indices: tuple[IndexPair, ...]
    fixed_indices: tuple[IndexPair, ...]
    residual_indices: tuple[IndexPair, ...]
    reserved_queries: tuple[Tensor, ...]
    absent_inherited_queries: tuple[Tensor, ...]
    duplicate_queries: tuple[Tensor, ...]
    empty_current_queries: tuple[Tensor, ...]
    diagnostics: Mapping[str, int]


def _validate_outputs_targets(
    outputs: Mapping[str, object],
    targets: Sequence[Mapping[str, object]],
    *,
    mask_type: str,
) -> tuple[Tensor, Sequence[Tensor]]:
    logits = outputs.get("pred_logits")
    masks = outputs.get("pred_masks")
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or isinstance(masks, (str, bytes))
        or not isinstance(masks, Sequence)
        or len(masks) != logits.shape[0]
        or len(targets) != logits.shape[0]
    ):
        raise TaskMemorySupervisionError("prediction batch is invalid")
    for batch_index, (mask, target) in enumerate(zip(masks, targets, strict=True)):
        labels = target.get("labels") if isinstance(target, Mapping) else None
        target_masks = target.get(mask_type) if isinstance(target, Mapping) else None
        if (
            not isinstance(mask, Tensor)
            or mask.ndim != 2
            or mask.shape[1] != logits.shape[1]
            or not isinstance(labels, Tensor)
            or labels.ndim != 1
            or not isinstance(target_masks, Tensor)
            or target_masks.ndim != 2
            or target_masks.shape != (labels.numel(), mask.shape[0])
        ):
            raise TaskMemorySupervisionError(
                f"prediction/target masks differ for batch {batch_index}"
            )
    return logits, masks


def build_independent_assignment(
    *,
    outputs: Mapping[str, object],
    targets: Sequence[Mapping[str, object]],
    matcher: object,
    mask_type: str,
) -> TaskMemoryAssignment:
    logits, _ = _validate_outputs_targets(outputs, targets, mask_type=mask_type)
    if not callable(matcher):
        raise TaskMemorySupervisionError("matcher must be callable")
    raw = matcher(outputs, targets, mask_type)
    if (
        isinstance(raw, (str, bytes))
        or not isinstance(raw, Sequence)
        or len(raw) != logits.shape[0]
    ):
        raise TaskMemorySupervisionError("matcher returned an invalid batch")
    indices = tuple(
        _validate_index_bounds(
            _index_pair(value, name=f"independent assignment {index}"),
            query_count=logits.shape[1],
            target_count=targets[index]["labels"].numel(),
            name=f"independent assignment {index}",
        )
        for index, value in enumerate(raw)
    )
    empty_pairs = tuple(_empty_indices() for _ in indices)
    empty_queries = tuple(torch.empty(0, dtype=torch.long) for _ in indices)
    return TaskMemoryAssignment(
        matcher_mode="independent",
        indices=indices,
        fixed_indices=empty_pairs,
        residual_indices=indices,
        reserved_queries=empty_queries,
        absent_inherited_queries=empty_queries,
        duplicate_queries=empty_queries,
        empty_current_queries=empty_queries,
        diagnostics=MappingProxyType(
            {
                "absent_inherited_queries": 0,
                "ambiguous_inheritance_excluded": 0,
                "duplicate_inherited_queries": 0,
                "fixed_inherited_queries": 0,
                "ignored_inheritance_excluded": 0,
                "previous_only_positive_queries": 0,
                "residual_matches": sum(pair[0].numel() for pair in indices),
                "wrong_route_repairs": 0,
            }
        ),
    )


def _residual_assignment(
    *,
    outputs: Mapping[str, object],
    target: Mapping[str, object],
    matcher: object,
    mask_type: str,
    batch_index: int,
    eligible_queries: Sequence[int],
    eligible_targets: Sequence[int],
) -> IndexPair:
    if not eligible_queries or not eligible_targets:
        return _empty_indices()
    query_index = torch.tensor(eligible_queries, dtype=torch.long)
    target_index = torch.tensor(eligible_targets, dtype=torch.long)
    masks = outputs["pred_masks"]
    subset_outputs = {
        "pred_logits": outputs["pred_logits"][
            batch_index : batch_index + 1, query_index
        ],
        "pred_masks": [masks[batch_index][:, query_index]],
    }
    subset_target = {
        "labels": target["labels"][target_index],
        mask_type: target[mask_type][target_index],
    }
    raw = matcher(subset_outputs, [subset_target], mask_type)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 1:
        raise TaskMemorySupervisionError("residual matcher returned an invalid batch")
    local_query, local_target = _validate_index_bounds(
        _index_pair(raw[0], name="residual assignment"),
        query_count=len(eligible_queries),
        target_count=len(eligible_targets),
        name="residual assignment",
    )
    mapped_queries = torch.tensor(
        [eligible_queries[index] for index in local_query.tolist()], dtype=torch.long
    )
    mapped_targets = torch.tensor(
        [eligible_targets[index] for index in local_target.tolist()], dtype=torch.long
    )
    return mapped_queries, mapped_targets


def _sorted_tensor(values: Sequence[int]) -> Tensor:
    return torch.tensor(sorted(values), dtype=torch.long)


def prediction_stage_ids(meta: StageMeta, prediction_rows: int) -> Tensor:
    """Resolve local temporal stage IDs for segment- or point-level mask rows."""
    if (
        isinstance(prediction_rows, bool)
        or not isinstance(prediction_rows, int)
        or prediction_rows <= 0
    ):
        raise TaskMemorySupervisionError("prediction row count is invalid")
    if meta.segment_stage_ids.numel() == prediction_rows:
        return meta.segment_stage_ids.clone()
    if meta.point2segment.numel() == prediction_rows:
        if (
            meta.point2segment.numel() == 0
            or meta.point2segment.min().item() < 0
            or meta.point2segment.max().item() >= meta.segment_stage_ids.numel()
        ):
            raise TaskMemorySupervisionError("point-to-segment stage map is invalid")
        return meta.segment_stage_ids[meta.point2segment]
    raise TaskMemorySupervisionError("stage metadata differs from prediction rows")


def build_tala_assignment(
    *,
    outputs: Mapping[str, object],
    targets: Sequence[Mapping[str, object]],
    matcher: object,
    mask_type: str,
    route: EntityRoute,
    ledgers: Sequence[TrainingIdentityLedger],
    identity_keys: Sequence[Sequence[QualifiedIdentity]],
    ambiguity_metadata: Sequence[object],
    stage_meta: Sequence[StageMeta],
) -> TaskMemoryAssignment:
    """Fix qualified inherited pairs, then match only remaining queries and GT."""
    logits, masks = _validate_outputs_targets(outputs, targets, mask_type=mask_type)
    if not isinstance(route, EntityRoute):
        raise TaskMemorySupervisionError("TALA requires an EntityRoute")
    route.validate()
    batch_size, query_count = logits.shape[:2]
    if (
        not callable(matcher)
        or len(ledgers) != batch_size
        or len(identity_keys) != batch_size
        or len(ambiguity_metadata) != batch_size
        or len(stage_meta) != batch_size
        or route.query_to_slot.shape != (batch_size, query_count)
        or any(not isinstance(ledger, TrainingIdentityLedger) for ledger in ledgers)
        or any(not isinstance(meta, StageMeta) for meta in stage_meta)
    ):
        raise TaskMemorySupervisionError("TALA batch metadata is misaligned")
    if route.metadata_sha256 != _metadata_sha256(stage_meta):
        raise TaskMemorySupervisionError("route metadata commitment differs")

    totals = {
        "absent_inherited_queries": 0,
        "ambiguous_inheritance_excluded": 0,
        "duplicate_inherited_queries": 0,
        "fixed_inherited_queries": 0,
        "ignored_inheritance_excluded": 0,
        "previous_only_positive_queries": 0,
        "residual_matches": 0,
        "wrong_route_repairs": 0,
    }
    combined: list[IndexPair] = []
    fixed_all: list[IndexPair] = []
    residual_all: list[IndexPair] = []
    reserved_all = []
    absent_all = []
    duplicate_all = []
    empty_current_all = []

    for batch_index in range(batch_size):
        meta = stage_meta[batch_index]
        if int(route.stage_indices[batch_index].item()) != meta.absolute_stage_index:
            raise TaskMemorySupervisionError(
                "route stage differs from supervision metadata"
            )
        identities = tuple(
            _qualified_identity(value) for value in identity_keys[batch_index]
        )
        target_labels = targets[batch_index]["labels"]
        if len(identities) != target_labels.numel() or any(
            identity[0] != meta.reference_id for identity in identities
        ):
            raise TaskMemorySupervisionError("TALA target identities are misaligned")
        target_by_identity = {
            identity: index for index, identity in enumerate(identities)
        }
        if len(target_by_identity) != len(identities):
            raise TaskMemorySupervisionError("TALA target identities are duplicated")
        ambiguous = ambiguous_identity_ids(ambiguity_metadata[batch_index])
        candidates: dict[QualifiedIdentity, list[int]] = {}
        for query in (
            (route.query_to_slot[batch_index] >= 0).nonzero(as_tuple=True)[0].tolist()
        ):
            logical_id = int(route.prior_logical_id[batch_index, query].item())
            generation = int(route.prior_generation[batch_index, query].item())
            identity = ledgers[batch_index].resolve(logical_id, generation)
            if identity is None:
                continue
            if identity[0] != meta.reference_id:
                raise TaskMemorySupervisionError("ledger reference differs from stage")
            if identity[1] in ambiguous:
                totals["ambiguous_inheritance_excluded"] += 1
                continue
            target_row = target_by_identity.get(identity)
            if target_row is not None and int(target_labels[target_row].item()) == 253:
                totals["ignored_inheritance_excluded"] += 1
                continue
            candidates.setdefault(identity, []).append(query)

        fixed_queries = []
        fixed_targets = []
        reserved: set[int] = set()
        absent = []
        duplicates = []
        empty_current = []
        for identity in sorted(candidates):
            queries = sorted(
                candidates[identity],
                key=lambda query: (
                    -float(route.route_score[batch_index, query].item()),
                    query,
                ),
            )
            primary, *losers = queries
            reserved.update(queries)
            duplicates.extend(losers)
            empty_current.extend(losers)
            totals["duplicate_inherited_queries"] += len(losers)
            target_row = target_by_identity.get(identity)
            if target_row is None:
                absent.append(primary)
                empty_current.append(primary)
                totals["absent_inherited_queries"] += 1
                continue
            fixed_queries.append(primary)
            fixed_targets.append(target_row)
            totals["fixed_inherited_queries"] += 1
            target_mask = targets[batch_index][mask_type][target_row]
            mask_stage_ids = prediction_stage_ids(meta, masks[batch_index].shape[0])
            latest_local_stage = int(mask_stage_ids.max().item())
            current_selector = mask_stage_ids == latest_local_stage
            current_selector = current_selector.to(device=target_mask.device)
            if (
                target_mask.any().item()
                and not target_mask[current_selector].any().item()
            ):
                totals["previous_only_positive_queries"] += 1
                empty_current.append(primary)

        fixed = (
            torch.tensor(fixed_queries, dtype=torch.long),
            torch.tensor(fixed_targets, dtype=torch.long),
        )
        fixed_target_set = set(fixed_targets)
        eligible_queries = [
            query for query in range(query_count) if query not in reserved
        ]
        eligible_targets = [
            target_row
            for target_row in range(target_labels.numel())
            if target_row not in fixed_target_set
        ]
        residual = _residual_assignment(
            outputs=outputs,
            target=targets[batch_index],
            matcher=matcher,
            mask_type=mask_type,
            batch_index=batch_index,
            eligible_queries=eligible_queries,
            eligible_targets=eligible_targets,
        )
        totals["residual_matches"] += residual[0].numel()
        combined.append(
            (
                torch.cat((fixed[0], residual[0])),
                torch.cat((fixed[1], residual[1])),
            )
        )
        fixed_all.append(fixed)
        residual_all.append(residual)
        reserved_all.append(_sorted_tensor(tuple(reserved)))
        absent_all.append(_sorted_tensor(absent))
        duplicate_all.append(_sorted_tensor(duplicates))
        empty_current_all.append(_sorted_tensor(empty_current))

    return TaskMemoryAssignment(
        matcher_mode="tala",
        indices=tuple(combined),
        fixed_indices=tuple(fixed_all),
        residual_indices=tuple(residual_all),
        reserved_queries=tuple(reserved_all),
        absent_inherited_queries=tuple(absent_all),
        duplicate_queries=tuple(duplicate_all),
        empty_current_queries=tuple(empty_current_all),
        diagnostics=MappingProxyType(totals),
    )


__all__ = [
    "QualifiedIdentity",
    "TaskMemoryAssignment",
    "TaskMemorySupervisionError",
    "TrainingIdentityLedger",
    "ambiguous_identity_ids",
    "build_independent_assignment",
    "build_tala_assignment",
    "prediction_stage_ids",
]
