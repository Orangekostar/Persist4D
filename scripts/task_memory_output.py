"""Causal commit-zero and one-scan-lag publication policies."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from datasets.task_memory_episode import StageMeta
from models.persistent_memory import _optimal_assignment_with_stable_ties
from scripts.rescene_task_postprocess import OfficialTaskPrediction


class TaskMemoryOutputError(ValueError):
    """Raised when a publication update violates its causal contract."""


@dataclass(frozen=True)
class PublishedIdentity:
    logical_id: Hashable
    generation: int
    class_id: int

    def __post_init__(self) -> None:
        if self.logical_id is None:
            raise TaskMemoryOutputError("logical_id cannot be None")
        try:
            hash(self.logical_id)
        except TypeError as error:
            raise TaskMemoryOutputError("logical_id must be hashable") from error
        for name, value in (
            ("generation", self.generation),
            ("class_id", self.class_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TaskMemoryOutputError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class ArchivedCandidate:
    identity: PublishedIdentity
    source_query_id: int
    score: float
    packed_mask: bytes
    point_count: int

    def mask(self) -> Tensor:
        values = np.unpackbits(
            np.frombuffer(self.packed_mask, dtype=np.uint8),
            count=self.point_count,
            bitorder="little",
        ).copy()
        return torch.from_numpy(values).bool()


@dataclass(frozen=True)
class ArchivedScan:
    scan_id: str
    absolute_stage_index: int
    point_count: int
    candidates: tuple[ArchivedCandidate, ...]
    payload_bytes: int
    content_sha256: str


@dataclass(frozen=True)
class RevisionRecord:
    absolute_stage_index: int
    scan_id: str
    replaced_sha256: str
    committed_sha256: str
    old_candidate_count: int
    new_candidate_count: int
    route_conflicts: int
    fallback_matches: int


@dataclass(frozen=True)
class PublicationAccounting:
    archive_payload_bytes: int
    lag1_buffer_bytes: int
    materialized_output_bytes: int


@dataclass(frozen=True)
class PublishedPrefix:
    prediction: dict[str, Tensor]
    keys: tuple[PublishedIdentity, ...]
    scan_ids: tuple[str, ...]
    scan_vertex_offsets: Tensor
    provisional_scan_id: str | None
    archive: tuple[ArchivedScan, ...]
    revision_log: tuple[RevisionRecord, ...]
    accounting: PublicationAccounting
    score_reducer: str


@dataclass(frozen=True)
class _DenseCandidate:
    identity: PublishedIdentity
    source_query_id: int
    score: float
    mask: Tensor


@dataclass(frozen=True)
class _DenseScan:
    scan_id: str
    absolute_stage_index: int
    vertex_ids: Tensor
    candidates: tuple[_DenseCandidate, ...]


@dataclass(frozen=True)
class _WindowCandidate:
    candidate_index: int
    source_query_id: int
    class_id: int
    score: float
    previous_mask: Tensor | None
    current_mask: Tensor
    routed_identity: PublishedIdentity | None

    @property
    def has_previous_support(self) -> bool:
        return self.previous_mask is not None and self.previous_mask.any().item()

    @property
    def has_current_support(self) -> bool:
        return self.current_mask.any().item()


_SCORE_REDUCERS = frozenset({"mean", "latest", "max"})


def _validate_reducer(score_reducer: str) -> str:
    if score_reducer not in _SCORE_REDUCERS:
        raise TaskMemoryOutputError(
            f"score_reducer must be one of {sorted(_SCORE_REDUCERS)}"
        )
    return score_reducer


def _normalize_identity_map(
    identity_map: Mapping[int, tuple[Hashable, int] | None],
) -> dict[int, tuple[Hashable, int]]:
    if not isinstance(identity_map, Mapping):
        raise TaskMemoryOutputError("identity_map must be a mapping")
    normalized: dict[int, tuple[Hashable, int]] = {}
    for query_id, value in identity_map.items():
        if isinstance(query_id, bool) or not isinstance(query_id, int) or query_id < 0:
            raise TaskMemoryOutputError(
                "identity_map query IDs must be non-negative integers"
            )
        if value is None:
            continue
        if not isinstance(value, tuple) or len(value) != 2:
            raise TaskMemoryOutputError(
                "identity_map values must be (logical_id, generation) tuples"
            )
        logical_id, generation = value
        if logical_id is None:
            raise TaskMemoryOutputError("routed logical_id cannot be None")
        try:
            hash(logical_id)
        except TypeError as error:
            raise TaskMemoryOutputError("routed logical_id must be hashable") from error
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise TaskMemoryOutputError(
                "routed generation must be a non-negative integer"
            )
        normalized[query_id] = (logical_id, generation)
    return normalized


def _pack_mask(mask: Tensor) -> bytes:
    values = mask.detach().cpu().bool().contiguous().numpy().astype(np.uint8)
    return np.packbits(values, bitorder="little").tobytes()


def _logical_id_bytes(logical_id: Hashable) -> int:
    return len(f"{type(logical_id).__qualname__}:{logical_id!r}".encode())


def _archive_scan(scan: _DenseScan) -> ArchivedScan:
    archived_candidates = tuple(
        ArchivedCandidate(
            identity=candidate.identity,
            source_query_id=candidate.source_query_id,
            score=candidate.score,
            packed_mask=_pack_mask(candidate.mask),
            point_count=int(candidate.mask.numel()),
        )
        for candidate in scan.candidates
    )
    digest = hashlib.sha256()
    digest.update(scan.scan_id.encode("utf-8"))
    digest.update(scan.absolute_stage_index.to_bytes(8, "little", signed=False))
    digest.update(scan.vertex_ids.detach().cpu().long().contiguous().numpy().tobytes())
    payload_bytes = len(scan.scan_id.encode("utf-8")) + 16
    for candidate in archived_candidates:
        digest.update(repr(candidate.identity.logical_id).encode("utf-8"))
        digest.update(candidate.identity.generation.to_bytes(8, "little"))
        digest.update(candidate.identity.class_id.to_bytes(8, "little"))
        digest.update(candidate.source_query_id.to_bytes(8, "little"))
        digest.update(np.float64(candidate.score).tobytes())
        digest.update(candidate.packed_mask)
        payload_bytes += (
            len(candidate.packed_mask)
            + _logical_id_bytes(candidate.identity.logical_id)
            + 32
        )
    return ArchivedScan(
        scan_id=scan.scan_id,
        absolute_stage_index=scan.absolute_stage_index,
        point_count=int(scan.vertex_ids.numel()),
        candidates=archived_candidates,
        payload_bytes=payload_bytes,
        content_sha256=digest.hexdigest(),
    )


def _dense_from_archive(scan: ArchivedScan) -> _DenseScan:
    return _DenseScan(
        scan_id=scan.scan_id,
        absolute_stage_index=scan.absolute_stage_index,
        vertex_ids=torch.arange(scan.point_count, dtype=torch.long),
        candidates=tuple(
            _DenseCandidate(
                identity=candidate.identity,
                source_query_id=candidate.source_query_id,
                score=candidate.score,
                mask=candidate.mask(),
            )
            for candidate in scan.candidates
        ),
    )


def _dense_payload_bytes(scan: _DenseScan | None) -> int:
    if scan is None:
        return 0
    size = scan.vertex_ids.numel() * scan.vertex_ids.element_size()
    size += len(scan.scan_id.encode("utf-8")) + 16
    for candidate in scan.candidates:
        size += candidate.mask.numel() * candidate.mask.element_size()
        size += _logical_id_bytes(candidate.identity.logical_id) + 32
    return size


def _scan_digest(scan: _DenseScan) -> str:
    return _archive_scan(scan).content_sha256


def _materialize(
    *,
    archive: Sequence[ArchivedScan],
    buffer: _DenseScan | None,
    revision_log: Sequence[RevisionRecord],
    score_reducer: str,
    provisional_scan_id: str | None,
) -> PublishedPrefix:
    scans = [_dense_from_archive(scan) for scan in archive]
    if buffer is not None:
        scans.append(buffer)
    keys: list[PublishedIdentity] = []
    occurrences: dict[PublishedIdentity, list[tuple[int, _DenseCandidate]]] = (
        defaultdict(list)
    )
    point_counts = []
    for scan_index, scan in enumerate(scans):
        point_counts.append(int(scan.vertex_ids.numel()))
        for candidate in scan.candidates:
            if candidate.identity not in occurrences:
                keys.append(candidate.identity)
            occurrences[candidate.identity].append((scan_index, candidate))

    offsets = [0]
    for count in point_counts:
        offsets.append(offsets[-1] + count)
    masks = torch.zeros((offsets[-1], len(keys)), dtype=torch.bool)
    scores = torch.empty(len(keys), dtype=torch.float32)
    classes = torch.empty(len(keys), dtype=torch.long)
    for column, key in enumerate(keys):
        values = occurrences[key]
        score_values = []
        for scan_index, candidate in values:
            start, stop = offsets[scan_index : scan_index + 2]
            if candidate.mask.numel() != stop - start:
                raise TaskMemoryOutputError(
                    "stored scan mask has the wrong point count"
                )
            masks[start:stop, column] = candidate.mask
            score_values.append(candidate.score)
        if score_reducer == "mean":
            reduced_score = sum(score_values) / len(score_values)
        elif score_reducer == "latest":
            reduced_score = score_values[-1]
        else:
            reduced_score = max(score_values)
        scores[column] = reduced_score
        classes[column] = key.class_id

    materialized_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in (masks, scores, classes)
    )
    return PublishedPrefix(
        prediction={
            "pred_masks": masks,
            "pred_scores": scores,
            "pred_classes": classes,
        },
        keys=tuple(keys),
        scan_ids=tuple(scan.scan_id for scan in scans),
        scan_vertex_offsets=torch.tensor(offsets, dtype=torch.long),
        provisional_scan_id=provisional_scan_id,
        archive=tuple(archive),
        revision_log=tuple(revision_log),
        accounting=PublicationAccounting(
            archive_payload_bytes=sum(scan.payload_bytes for scan in archive),
            lag1_buffer_bytes=_dense_payload_bytes(buffer),
            materialized_output_bytes=materialized_bytes,
        ),
        score_reducer=score_reducer,
    )


def _validate_stage(
    *,
    prediction: OfficialTaskPrediction,
    stage_meta: StageMeta,
    expected_stage: int,
    expected_episode: tuple[str, str] | None,
) -> tuple[str, str]:
    if not isinstance(prediction, OfficialTaskPrediction):
        raise TaskMemoryOutputError("prediction must be OfficialTaskPrediction")
    if not isinstance(stage_meta, StageMeta):
        raise TaskMemoryOutputError("stage_meta must be StageMeta")
    prediction.validate()
    episode = (stage_meta.reference_id, stage_meta.episode_id)
    if expected_episode is not None and episode != expected_episode:
        raise TaskMemoryOutputError("publisher cannot mix episodes")
    if stage_meta.absolute_stage_index != expected_stage:
        raise TaskMemoryOutputError("stages must be published in order")
    expected_window = 1 if expected_stage == 0 else 2
    if len(stage_meta.scan_ids_in_window) != expected_window:
        raise TaskMemoryOutputError("stage does not use the required causal window")
    point_count = int(stage_meta.scan_vertex_offsets[-1].item())
    if prediction.pred_masks.shape[0] != point_count:
        raise TaskMemoryOutputError("prediction and stage metadata point counts differ")
    stages = prediction.temporal_stages.detach().cpu().long().contiguous()
    meta_stages = stage_meta.local_stage_ids.detach().cpu().long().contiguous()
    if not torch.equal(stages, meta_stages):
        raise TaskMemoryOutputError(
            "prediction temporal lineage differs from StageMeta"
        )
    if prediction.latest_stage_index != expected_window - 1:
        raise TaskMemoryOutputError("prediction latest stage differs from current scan")
    offsets = stage_meta.scan_vertex_offsets.detach().cpu().long()
    for local_stage, vertex_ids in enumerate(stage_meta.original_vertex_ids):
        start, stop = (int(value) for value in offsets[local_stage : local_stage + 2])
        ids = vertex_ids.detach().cpu().long().contiguous()
        if ids.numel() != stop - start:
            raise TaskMemoryOutputError("original vertex IDs differ from scan points")
        if ids.unique().numel() != ids.numel():
            raise TaskMemoryOutputError("original vertex IDs must be unique per scan")
        if not torch.all(meta_stages[start:stop] == local_stage).item():
            raise TaskMemoryOutputError("StageMeta scan offsets cross temporal stages")
    if prediction.source_query_ids.numel() and (
        prediction.source_query_ids.min().item() < 0
    ):
        raise TaskMemoryOutputError("source query IDs must be non-negative")
    return episode


def _ephemeral_identity(
    *, stage_meta: StageMeta, candidate_index: int, query_id: int, class_id: int
) -> PublishedIdentity:
    logical_id = (
        f"ephemeral:{stage_meta.episode_id}:{stage_meta.absolute_stage_index}:"
        f"{query_id}:{candidate_index}"
    )
    return PublishedIdentity(logical_id, 0, class_id)


def _window_candidates(
    *,
    prediction: OfficialTaskPrediction,
    identity_map: Mapping[int, tuple[Hashable, int]],
    stage_meta: StageMeta,
) -> list[_WindowCandidate]:
    masks = prediction.pred_masks.detach().cpu().bool().contiguous()
    scores = prediction.pred_scores.detach().cpu().float().contiguous()
    classes = prediction.pred_classes.detach().cpu().long().contiguous()
    query_ids = prediction.source_query_ids.detach().cpu().long().contiguous()
    offsets = stage_meta.scan_vertex_offsets.detach().cpu().long()
    has_previous = len(stage_meta.scan_ids_in_window) == 2
    previous_stop = int(offsets[1].item()) if has_previous else 0
    current_start = int(offsets[-2].item())
    result = []
    for index in range(scores.numel()):
        query_id = int(query_ids[index].item())
        class_id = int(classes[index].item())
        route = identity_map.get(query_id)
        routed_identity = (
            PublishedIdentity(route[0], route[1], class_id)
            if route is not None
            else None
        )
        candidate = _WindowCandidate(
            candidate_index=index,
            source_query_id=query_id,
            class_id=class_id,
            score=float(scores[index].item()),
            previous_mask=masks[:previous_stop, index].clone()
            if has_previous
            else None,
            current_mask=masks[current_start:, index].clone(),
            routed_identity=routed_identity,
        )
        if candidate.has_previous_support or candidate.has_current_support:
            result.append(candidate)
    return result


def _winner_key(candidate: _WindowCandidate) -> tuple[float, int, int]:
    return (-candidate.score, candidate.source_query_id, candidate.candidate_index)


def _iou_matrix(
    prior_candidates: Sequence[_DenseCandidate],
    window_candidates: Sequence[_WindowCandidate],
    *,
    source_vertex_ids: Tensor,
    target_vertex_ids: Tensor,
) -> Tensor:
    source_ids = source_vertex_ids.detach().cpu().long().tolist()
    target_ids = target_vertex_ids.detach().cpu().long().tolist()
    if len(source_ids) != len(target_ids) or set(source_ids) != set(target_ids):
        raise TaskMemoryOutputError(
            "repeated scan must expose the same original vertex collection"
        )
    source_position = {vertex_id: index for index, vertex_id in enumerate(source_ids)}
    reorder = torch.tensor(
        [source_position[vertex_id] for vertex_id in target_ids], dtype=torch.long
    )
    prior_masks = torch.stack([candidate.mask for candidate in prior_candidates], dim=0)
    prior_masks = prior_masks[:, reorder]
    new_masks = torch.stack(
        [candidate.previous_mask for candidate in window_candidates], dim=0
    )
    intersection = (prior_masks[:, None] & new_masks[None]).sum(dim=-1).double()
    union = (prior_masks[:, None] | new_masks[None]).sum(dim=-1).double()
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def _stable_matches(
    *,
    prior_candidates: Sequence[_DenseCandidate],
    window_candidates: Sequence[_WindowCandidate],
    source_vertex_ids: Tensor,
    target_vertex_ids: Tensor,
    threshold: float,
) -> list[tuple[int, int, float]]:
    if not prior_candidates or not window_candidates:
        return []
    score = _iou_matrix(
        prior_candidates,
        window_candidates,
        source_vertex_ids=source_vertex_ids,
        target_vertex_ids=target_vertex_ids,
    )
    rows, columns = _optimal_assignment_with_stable_ties(score)
    matches = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        value = float(score[row, column].item())
        if value >= threshold:
            matches.append((row, column, value))
    return matches


def _resolve_identities(
    *,
    candidates: Sequence[_WindowCandidate],
    stage_meta: StageMeta,
    previous_buffer: _DenseScan | None,
    iou_threshold: float,
) -> tuple[dict[int, PublishedIdentity], int, int]:
    identities: dict[int, PublishedIdentity] = {}
    forced_ephemeral: set[int] = set()
    route_groups: dict[PublishedIdentity, list[_WindowCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.routed_identity is not None:
            route_groups[candidate.routed_identity].append(candidate)
    routed_winners = []
    for identity, group in route_groups.items():
        ordered = sorted(group, key=_winner_key)
        winner = ordered[0]
        identities[winner.candidate_index] = identity
        routed_winners.append(winner)
        forced_ephemeral.update(candidate.candidate_index for candidate in ordered[1:])

    route_conflicts = 0
    fallback_matches = 0
    reserved_prior: set[int] = set()
    if previous_buffer is not None:
        source_ids = stage_meta.original_vertex_ids[0]
        target_ids = previous_buffer.vertex_ids
        classes = sorted(
            {candidate.class_id for candidate in candidates}
            | {candidate.identity.class_id for candidate in previous_buffer.candidates}
        )
        for class_id in classes:
            prior_indices = [
                index
                for index, candidate in enumerate(previous_buffer.candidates)
                if candidate.identity.class_id == class_id
            ]
            routed = sorted(
                (
                    candidate
                    for candidate in routed_winners
                    if candidate.class_id == class_id and candidate.has_previous_support
                ),
                key=lambda candidate: (
                    candidate.source_query_id,
                    candidate.candidate_index,
                ),
            )
            route_matches = _stable_matches(
                prior_candidates=[
                    previous_buffer.candidates[index] for index in prior_indices
                ],
                window_candidates=routed,
                source_vertex_ids=source_ids,
                target_vertex_ids=target_ids,
                threshold=iou_threshold,
            )
            for prior_local, candidate_local, _ in route_matches:
                prior_index = prior_indices[prior_local]
                reserved_prior.add(prior_index)
                old_identity = previous_buffer.candidates[prior_index].identity
                new_identity = routed[candidate_local].routed_identity
                if old_identity != new_identity:
                    route_conflicts += 1

        routed_keys = set(identities.values())
        for class_id in classes:
            prior_indices = [
                index
                for index, candidate in enumerate(previous_buffer.candidates)
                if index not in reserved_prior
                and candidate.identity.class_id == class_id
                and candidate.identity not in routed_keys
            ]
            unmatched = sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.class_id == class_id
                    and candidate.routed_identity is None
                    and candidate.candidate_index not in forced_ephemeral
                    and candidate.candidate_index not in identities
                    and candidate.has_previous_support
                ),
                key=lambda candidate: (
                    candidate.source_query_id,
                    candidate.candidate_index,
                ),
            )
            matches = _stable_matches(
                prior_candidates=[
                    previous_buffer.candidates[index] for index in prior_indices
                ],
                window_candidates=unmatched,
                source_vertex_ids=source_ids,
                target_vertex_ids=target_ids,
                threshold=iou_threshold,
            )
            for prior_local, candidate_local, _ in matches:
                candidate = unmatched[candidate_local]
                identities[candidate.candidate_index] = previous_buffer.candidates[
                    prior_indices[prior_local]
                ].identity
                fallback_matches += 1

    for candidate in candidates:
        if candidate.candidate_index not in identities:
            identities[candidate.candidate_index] = _ephemeral_identity(
                stage_meta=stage_meta,
                candidate_index=candidate.candidate_index,
                query_id=candidate.source_query_id,
                class_id=candidate.class_id,
            )
    return identities, route_conflicts, fallback_matches


def _align_masks(
    *, source_vertex_ids: Tensor, target_vertex_ids: Tensor, mask: Tensor
) -> Tensor:
    source = source_vertex_ids.detach().cpu().long().tolist()
    target = target_vertex_ids.detach().cpu().long().tolist()
    if len(source) != len(target) or set(source) != set(target):
        raise TaskMemoryOutputError(
            "repeated scan must expose the same original vertex collection"
        )
    source_position = {vertex_id: index for index, vertex_id in enumerate(source)}
    return mask[
        torch.tensor(
            [source_position[vertex_id] for vertex_id in target], dtype=torch.long
        )
    ].clone()


def _build_dense_scan(
    *,
    candidates: Sequence[_WindowCandidate],
    identities: Mapping[int, PublishedIdentity],
    stage_meta: StageMeta,
    scan_position: int,
    target_vertex_ids: Tensor | None = None,
) -> _DenseScan:
    source_vertex_ids = (
        stage_meta.original_vertex_ids[scan_position]
        .detach()
        .cpu()
        .long()
        .contiguous()
        .clone()
    )
    vertex_ids = (
        source_vertex_ids
        if target_vertex_ids is None
        else target_vertex_ids.detach().cpu().long().contiguous().clone()
    )
    dense_candidates = []
    for candidate in candidates:
        mask = (
            candidate.previous_mask
            if scan_position == 0 and len(stage_meta.scan_ids_in_window) == 2
            else candidate.current_mask
        )
        if mask is None or not mask.any().item():
            continue
        if target_vertex_ids is not None:
            mask = _align_masks(
                source_vertex_ids=source_vertex_ids,
                target_vertex_ids=vertex_ids,
                mask=mask,
            )
        dense_candidates.append(
            _DenseCandidate(
                identity=identities[candidate.candidate_index],
                source_query_id=candidate.source_query_id,
                score=candidate.score,
                mask=mask.detach().cpu().bool().contiguous().clone(),
            )
        )
    return _DenseScan(
        scan_id=stage_meta.scan_ids_in_window[scan_position],
        absolute_stage_index=(
            stage_meta.absolute_stage_index
            if scan_position == len(stage_meta.scan_ids_in_window) - 1
            else stage_meta.absolute_stage_index - 1
        ),
        vertex_ids=vertex_ids,
        candidates=tuple(dense_candidates),
    )


class LagOnePublisher:
    """Replace only the prior provisional scan using the current W=2 output."""

    def __init__(self, *, score_reducer: str = "mean", iou_threshold: float = 0.5):
        self.score_reducer = _validate_reducer(score_reducer)
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not math.isfinite(iou_threshold)
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise TaskMemoryOutputError(
                "iou_threshold must be finite and within [0, 1]"
            )
        self.iou_threshold = float(iou_threshold)
        self._archive: tuple[ArchivedScan, ...] = ()
        self._buffer: _DenseScan | None = None
        self._revision_log: tuple[RevisionRecord, ...] = ()
        self._episode: tuple[str, str] | None = None
        self._next_stage = 0

    @property
    def buffer_scan_id(self) -> str | None:
        return None if self._buffer is None else self._buffer.scan_id

    def update(
        self,
        prediction: OfficialTaskPrediction,
        identity_map: Mapping[int, tuple[Hashable, int] | None],
        stage_meta: StageMeta,
    ) -> PublishedPrefix:
        episode = _validate_stage(
            prediction=prediction,
            stage_meta=stage_meta,
            expected_stage=self._next_stage,
            expected_episode=self._episode,
        )
        routes = _normalize_identity_map(identity_map)
        candidates = _window_candidates(
            prediction=prediction,
            identity_map=routes,
            stage_meta=stage_meta,
        )
        identities, route_conflicts, fallback_matches = _resolve_identities(
            candidates=candidates,
            stage_meta=stage_meta,
            previous_buffer=self._buffer,
            iou_threshold=self.iou_threshold,
        )

        if self._buffer is not None:
            if stage_meta.scan_ids_in_window[0] != self._buffer.scan_id:
                raise TaskMemoryOutputError(
                    "W=2 previous scan differs from lag-one buffer"
                )
            revised = _build_dense_scan(
                candidates=candidates,
                identities=identities,
                stage_meta=stage_meta,
                scan_position=0,
                target_vertex_ids=self._buffer.vertex_ids,
            )
            frozen = _archive_scan(revised)
            revision = RevisionRecord(
                absolute_stage_index=stage_meta.absolute_stage_index,
                scan_id=revised.scan_id,
                replaced_sha256=_scan_digest(self._buffer),
                committed_sha256=frozen.content_sha256,
                old_candidate_count=len(self._buffer.candidates),
                new_candidate_count=len(revised.candidates),
                route_conflicts=route_conflicts,
                fallback_matches=fallback_matches,
            )
            self._archive = (*self._archive, frozen)
            self._revision_log = (*self._revision_log, revision)

        current_position = len(stage_meta.scan_ids_in_window) - 1
        self._buffer = _build_dense_scan(
            candidates=candidates,
            identities=identities,
            stage_meta=stage_meta,
            scan_position=current_position,
        )
        self._episode = episode
        self._next_stage += 1
        return _materialize(
            archive=self._archive,
            buffer=self._buffer,
            revision_log=self._revision_log,
            score_reducer=self.score_reducer,
            provisional_scan_id=self._buffer.scan_id,
        )


class CommitZeroPublisher:
    """Commit only each arrived current scan and never revise it."""

    def __init__(self, *, score_reducer: str = "mean") -> None:
        self.score_reducer = _validate_reducer(score_reducer)
        self._archive: tuple[ArchivedScan, ...] = ()
        self._episode: tuple[str, str] | None = None
        self._next_stage = 0
        self._latest_scan_id: str | None = None

    def update(
        self,
        prediction: OfficialTaskPrediction,
        identity_map: Mapping[int, tuple[Hashable, int] | None],
        stage_meta: StageMeta,
    ) -> PublishedPrefix:
        episode = _validate_stage(
            prediction=prediction,
            stage_meta=stage_meta,
            expected_stage=self._next_stage,
            expected_episode=self._episode,
        )
        if (
            self._latest_scan_id is not None
            and stage_meta.scan_ids_in_window[0] != self._latest_scan_id
        ):
            raise TaskMemoryOutputError(
                "W=2 previous scan differs from committed prefix"
            )
        routes = _normalize_identity_map(identity_map)
        candidates = _window_candidates(
            prediction=prediction,
            identity_map=routes,
            stage_meta=stage_meta,
        )
        identities, _, _ = _resolve_identities(
            candidates=candidates,
            stage_meta=stage_meta,
            previous_buffer=None,
            iou_threshold=0.5,
        )
        current_position = len(stage_meta.scan_ids_in_window) - 1
        current = _build_dense_scan(
            candidates=candidates,
            identities=identities,
            stage_meta=stage_meta,
            scan_position=current_position,
        )
        self._archive = (*self._archive, _archive_scan(current))
        self._episode = episode
        self._latest_scan_id = current.scan_id
        self._next_stage += 1
        return _materialize(
            archive=self._archive,
            buffer=None,
            revision_log=(),
            score_reducer=self.score_reducer,
            provisional_scan_id=None,
        )


__all__ = [
    "ArchivedCandidate",
    "ArchivedScan",
    "CommitZeroPublisher",
    "LagOnePublisher",
    "PublicationAccounting",
    "PublishedIdentity",
    "PublishedPrefix",
    "RevisionRecord",
    "TaskMemoryOutputError",
]
