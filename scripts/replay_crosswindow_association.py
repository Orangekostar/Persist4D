"""Canonical replay, lossless publication, and metric boundaries for CrossWindow."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from models.crosswindow_state import CrossWindowState, GroupKey
from models.overlap_entity_association import (
    A0_DEFAULT,
    A1_DEFAULT,
    A2_DEFAULT,
    AssignmentPlan,
    AssociationConfig,
    associate,
    build_evidence,
    canonical_frame_sha256,
    commit_observation,
)
from scripts.crosswindow_cache import (
    CandidateKey,
    CandidateRecord,
    CanonicalFrame,
    align_mask,
    build_canonical_frame,
)
from scripts.rescene_task_postprocess import OfficialTaskPrediction


class CrossWindowReplayError(ValueError):
    """Raised when replay or publication violates its frozen identity boundary."""


PublishedKey = tuple[int | str, int, int]


@dataclass(frozen=True)
class PublishedOccurrence:
    identity: PublishedKey
    source_key: CandidateKey
    source_query_id: int
    score: float
    mask: Tensor


@dataclass(frozen=True)
class PublishedScan:
    scan_id: str
    absolute_stage: int
    vertex_ids: Tensor
    candidates: tuple[PublishedOccurrence, ...]
    content_sha256: str
    payload_bytes: int


@dataclass(frozen=True)
class PublicationRevision:
    absolute_stage: int
    scan_id: str
    replaced_sha256: str
    committed_sha256: str
    old_candidate_count: int
    new_candidate_count: int
    mask_selection: str
    score_mode: str


@dataclass(frozen=True)
class PublicationAccounting:
    archive_bytes: int
    provisional_bytes: int
    materialized_bytes: int


@dataclass(frozen=True)
class HistoryBoundary:
    reference_id: str
    episode_id: str
    next_stage: int
    archive: tuple[PublishedScan, ...]
    provisional: PublishedScan | None
    revisions: tuple[PublicationRevision, ...]
    assignment_sha256: tuple[str, ...]


@dataclass(frozen=True)
class PublishedPrefix:
    prediction: dict[str, Tensor]
    keys: tuple[PublishedKey, ...]
    identity_map: tuple[tuple[GroupKey, tuple[int, int]], ...]
    scan_ids: tuple[str, ...]
    scan_vertex_offsets: Tensor
    archive: tuple[PublishedScan, ...]
    provisional: PublishedScan
    revisions: tuple[PublicationRevision, ...]
    accounting: PublicationAccounting
    history_boundary: HistoryBoundary
    collision_fallback_count: int
    score_reducer: str = "mean"


def _hash_tensor(digest: Any, value: Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def _archive_scan(
    *,
    scan_id: str,
    absolute_stage: int,
    vertex_ids: Tensor,
    candidates: tuple[PublishedOccurrence, ...],
) -> PublishedScan:
    canonical_ids = vertex_ids.detach().cpu().long().contiguous().clone()
    if (
        canonical_ids.ndim != 1
        or canonical_ids.unique().numel() != canonical_ids.numel()
        or not torch.equal(canonical_ids, canonical_ids.sort().values)
    ):
        raise CrossWindowReplayError("published scan vertex IDs must be canonical")
    digest = hashlib.sha256()
    digest.update(scan_id.encode())
    digest.update(absolute_stage.to_bytes(8, "little", signed=False))
    _hash_tensor(digest, canonical_ids)
    payload_bytes = canonical_ids.numel() * canonical_ids.element_size() + 16
    seen = set()
    for candidate in candidates:
        if candidate.identity in seen:
            raise CrossWindowReplayError(
                "assignment plan must resolve duplicate entity/scan/class keys"
            )
        seen.add(candidate.identity)
        if (
            candidate.mask.shape != canonical_ids.shape
            or candidate.mask.dtype != torch.bool
        ):
            raise CrossWindowReplayError("published candidate mask differs from scan")
        digest.update(repr(candidate.identity).encode())
        digest.update(repr(candidate.source_key).encode())
        digest.update(candidate.source_query_id.to_bytes(8, "little", signed=False))
        digest.update(
            torch.tensor(candidate.score, dtype=torch.float64).numpy().tobytes()
        )
        _hash_tensor(digest, candidate.mask)
        payload_bytes += candidate.mask.numel() * candidate.mask.element_size() + 40
    return PublishedScan(
        scan_id=scan_id,
        absolute_stage=absolute_stage,
        vertex_ids=canonical_ids,
        candidates=candidates,
        content_sha256=digest.hexdigest(),
        payload_bytes=payload_bytes,
    )


def _frame_group_keys(frame: CanonicalFrame) -> tuple[GroupKey, ...]:
    return tuple(
        sorted(
            (
                frame.producer_id,
                frame.episode_id,
                frame.order_id,
                frame.absolute_stage,
                group.source_query_id,
            )
            for group in frame.groups
        )
    )


def _scan_from_frame(
    frame: CanonicalFrame,
    plan: AssignmentPlan,
    *,
    scan_id: str,
    publication_identities: Mapping[CandidateKey, PublishedKey],
) -> PublishedScan:
    vertex_ids = frame.canonical_vertex_ids[scan_id]
    occurrences = []
    for candidate in frame.candidates:
        slices = [value for value in candidate.slices if value.scan_id == scan_id]
        if len(slices) != 1:
            raise CrossWindowReplayError("candidate scan slice coverage differs")
        candidate_slice = slices[0]
        mask = (
            align_mask(
                candidate_slice.mask,
                from_ids=candidate_slice.canonical_vertex_ids,
                to_ids=vertex_ids,
            )
            .detach()
            .cpu()
            .bool()
        )
        if not mask.any().item():
            continue
        occurrences.append(
            PublishedOccurrence(
                identity=publication_identities[candidate.key],
                source_key=candidate.key,
                source_query_id=candidate.key.source_query_id,
                score=float(candidate.score),
                mask=mask,
            )
        )
    return _archive_scan(
        scan_id=scan_id,
        absolute_stage=frame.absolute_stage,
        vertex_ids=vertex_ids,
        candidates=tuple(occurrences),
    )


def _publication_identities(
    frame: CanonicalFrame, plan: AssignmentPlan
) -> tuple[dict[CandidateKey, PublishedKey], int]:
    group_index = {key[-1]: index for index, key in enumerate(plan.group_keys)}
    if len(group_index) != len(plan.group_keys):
        raise CrossWindowReplayError("plan query groups must be unique")
    grouped: dict[PublishedKey, list[CandidateRecord]] = defaultdict(list)
    for candidate in frame.candidates:
        plan_index = group_index.get(candidate.key.source_query_id)
        if plan_index is None:
            raise CrossWindowReplayError(
                "candidate query is absent from assignment plan"
            )
        grouped[
            (
                plan.entity_for_group[plan_index],
                plan.generation_for_group[plan_index],
                candidate.predicted_class_id,
            )
        ].append(candidate)
    result: dict[CandidateKey, PublishedKey] = {}
    fallback_count = 0
    for identity, candidates in grouped.items():
        ordered = sorted(
            candidates,
            key=lambda value: (
                -value.score,
                value.key.source_query_id,
                value.key.candidate_index,
            ),
        )
        result[ordered[0].key] = identity
        for candidate in ordered[1:]:
            fallback_count += 1
            key = candidate.key
            result[key] = (
                (
                    f"collision:{key.producer_id}:{key.episode_id}:{key.order_id}:"
                    f"{key.absolute_stage}:{key.source_query_id}:{key.candidate_index}"
                ),
                0,
                candidate.predicted_class_id,
            )
    return result, fallback_count


def _materialize(
    archive: tuple[PublishedScan, ...],
    provisional: PublishedScan,
) -> tuple[
    dict[str, Tensor],
    tuple[PublishedKey, ...],
    tuple[str, ...],
    Tensor,
    int,
]:
    scans = (*archive, provisional)
    scan_ids = tuple(scan.scan_id for scan in scans)
    if len(scan_ids) != len(set(scan_ids)):
        raise CrossWindowReplayError("published prefix contains a duplicate scan")
    keys = tuple(
        sorted(
            {candidate.identity for scan in scans for candidate in scan.candidates},
            key=lambda value: (type(value[0]).__name__, repr(value[0]), value[1:]),
        )
    )
    key_index = {key: index for index, key in enumerate(keys)}
    offsets = [0]
    for scan in scans:
        offsets.append(offsets[-1] + scan.vertex_ids.numel())
    masks = torch.zeros((offsets[-1], len(keys)), dtype=torch.bool)
    scores: dict[PublishedKey, list[float]] = defaultdict(list)
    for scan_index, scan in enumerate(scans):
        start, stop = offsets[scan_index : scan_index + 2]
        for candidate in scan.candidates:
            column = key_index[candidate.identity]
            if masks[start:stop, column].any().item():
                raise CrossWindowReplayError(
                    "one entity/scan/class occurrence may be materialized once"
                )
            masks[start:stop, column] = candidate.mask
            scores[candidate.identity].append(candidate.score)
    prediction = {
        "pred_masks": masks,
        "pred_scores": torch.tensor(
            [sum(scores[key]) / len(scores[key]) for key in keys],
            dtype=torch.float32,
        ),
        "pred_classes": torch.tensor([key[2] for key in keys], dtype=torch.long),
    }
    materialized_bytes = sum(
        value.numel() * value.element_size() for value in prediction.values()
    )
    return (
        prediction,
        keys,
        scan_ids,
        torch.tensor(offsets, dtype=torch.long),
        materialized_bytes,
    )


def select_revision_scan(
    old: PublishedScan,
    new: PublishedScan,
    *,
    selector: str,
    score_mode: str,
) -> tuple[PublishedScan, tuple[dict[str, object], ...]]:
    """Select old/new masks under a frozen published identity trajectory."""
    if selector not in {"M-new", "M-old", "M-score"}:
        raise CrossWindowReplayError("revision selector must be M-new, M-old, or M-score")
    if score_mode not in {"MASK_ONLY_FIXED_SCORE", "SYSTEM_SELECTED_SCORE"}:
        raise CrossWindowReplayError("revision score mode differs")
    if old.scan_id != new.scan_id or not torch.equal(
        old.vertex_ids, new.vertex_ids
    ):
        raise CrossWindowReplayError("revision scans are not canonically aligned")
    old_by_identity = {candidate.identity: candidate for candidate in old.candidates}
    new_by_identity = {candidate.identity: candidate for candidate in new.candidates}
    if len(old_by_identity) != len(old.candidates) or len(new_by_identity) != len(
        new.candidates
    ):
        raise CrossWindowReplayError("revision scan identities must be unique")
    selected = []
    events = []
    for new_candidate in new.candidates:
        old_candidate = old_by_identity.get(new_candidate.identity)
        if old_candidate is None:
            selected.append(new_candidate)
            events.append(
                {
                    "identity": repr(new_candidate.identity),
                    "choice": "new_only",
                    "selector": selector,
                    "score_mode": score_mode,
                    "mask_equal": None,
                    "old_score": None,
                    "new_score": new_candidate.score,
                    "selected_score": new_candidate.score,
                }
            )
            continue
        mask_equal = torch.equal(old_candidate.mask, new_candidate.mask)
        choose_new = selector == "M-new" or (
            selector == "M-score"
            and (mask_equal or new_candidate.score > old_candidate.score + 1e-6)
        )
        chosen = new_candidate if choose_new else old_candidate
        selected_score = (
            new_candidate.score
            if score_mode == "MASK_ONLY_FIXED_SCORE"
            else chosen.score
        )
        selected.append(
            PublishedOccurrence(
                identity=new_candidate.identity,
                source_key=chosen.source_key,
                source_query_id=chosen.source_query_id,
                score=selected_score,
                mask=chosen.mask,
            )
        )
        events.append(
            {
                "identity": repr(new_candidate.identity),
                "choice": "new" if choose_new else "old",
                "selector": selector,
                "score_mode": score_mode,
                "mask_equal": mask_equal,
                "old_score": old_candidate.score,
                "new_score": new_candidate.score,
                "selected_score": selected_score,
            }
        )
    for identity, old_candidate in old_by_identity.items():
        if identity not in new_by_identity:
            events.append(
                {
                    "identity": repr(identity),
                    "choice": "old_only_dropped",
                    "selector": selector,
                    "score_mode": score_mode,
                    "mask_equal": None,
                    "old_score": old_candidate.score,
                    "new_score": None,
                    "selected_score": None,
                }
            )
    return (
        _archive_scan(
            scan_id=new.scan_id,
            absolute_stage=new.absolute_stage,
            vertex_ids=new.vertex_ids,
            candidates=tuple(selected),
        ),
        tuple(events),
    )


def publish(
    frame: CanonicalFrame,
    plan: AssignmentPlan,
    *,
    mask_selection: str,
    history_boundary: HistoryBoundary | None,
    score_mode: str = "SYSTEM_SELECTED_SCORE",
) -> PublishedPrefix:
    """Publish one causal stage using only the supplied immutable assignment plan."""
    plan.validate()
    selector = "M-new" if mask_selection == "new" else mask_selection
    if selector not in {"M-new", "M-old", "M-score"}:
        raise CrossWindowReplayError("publisher mask selector differs")
    if score_mode not in {"MASK_ONLY_FIXED_SCORE", "SYSTEM_SELECTED_SCORE"}:
        raise CrossWindowReplayError("publisher score mode differs")
    if plan.frame_sha256 != canonical_frame_sha256(frame):
        raise CrossWindowReplayError("assignment plan belongs to another frame")
    if plan.group_keys != _frame_group_keys(frame):
        raise CrossWindowReplayError("assignment group order differs from frame")
    if history_boundary is None:
        if frame.absolute_stage != 0:
            raise CrossWindowReplayError(
                "publication must begin at absolute stage zero"
            )
        archive: tuple[PublishedScan, ...] = ()
        revisions: tuple[PublicationRevision, ...] = ()
        assignment_hashes: tuple[str, ...] = ()
        prior = None
    else:
        if (
            history_boundary.reference_id != frame.reference_id
            or history_boundary.episode_id != frame.episode_id
            or history_boundary.next_stage != frame.absolute_stage
        ):
            raise CrossWindowReplayError("publication history belongs to another stage")
        archive = history_boundary.archive
        revisions = history_boundary.revisions
        assignment_hashes = history_boundary.assignment_sha256
        prior = history_boundary.provisional
    publication_identities, collision_fallback_count = _publication_identities(
        frame, plan
    )
    if prior is not None:
        if prior.scan_id not in frame.source_window:
            raise CrossWindowReplayError("lag-one frame omits the provisional scan")
        new_revision = _scan_from_frame(
            frame,
            plan,
            scan_id=prior.scan_id,
            publication_identities=publication_identities,
        )
        revised, _ = select_revision_scan(
            prior,
            new_revision,
            selector=selector,
            score_mode=score_mode,
        )
        archive = (*archive, revised)
        revisions = (
            *revisions,
            PublicationRevision(
                absolute_stage=frame.absolute_stage,
                scan_id=prior.scan_id,
                replaced_sha256=prior.content_sha256,
                committed_sha256=revised.content_sha256,
                old_candidate_count=len(prior.candidates),
                new_candidate_count=len(revised.candidates),
                mask_selection=selector,
                score_mode=score_mode,
            ),
        )
    current_scan_id = frame.source_window[-1]
    provisional = _scan_from_frame(
        frame,
        plan,
        scan_id=current_scan_id,
        publication_identities=publication_identities,
    )
    prediction, keys, scan_ids, offsets, materialized_bytes = _materialize(
        archive, provisional
    )
    boundary = HistoryBoundary(
        reference_id=frame.reference_id,
        episode_id=frame.episode_id,
        next_stage=frame.absolute_stage + 1,
        archive=archive,
        provisional=provisional,
        revisions=revisions,
        assignment_sha256=(*assignment_hashes, plan.content_sha256),
    )
    return PublishedPrefix(
        prediction=prediction,
        keys=keys,
        identity_map=tuple(
            zip(
                plan.group_keys,
                zip(
                    plan.entity_for_group,
                    plan.generation_for_group,
                    strict=True,
                ),
                strict=True,
            )
        ),
        scan_ids=scan_ids,
        scan_vertex_offsets=offsets,
        archive=archive,
        provisional=provisional,
        revisions=revisions,
        accounting=PublicationAccounting(
            archive_bytes=sum(scan.payload_bytes for scan in archive),
            provisional_bytes=provisional.payload_bytes,
            materialized_bytes=materialized_bytes,
        ),
        history_boundary=boundary,
        collision_fallback_count=collision_fallback_count,
    )


def evaluator_prediction(prefix: PublishedPrefix) -> dict[str, Tensor]:
    if not isinstance(prefix, PublishedPrefix):
        raise CrossWindowReplayError("evaluator input must be a published prefix")
    if set(prefix.prediction) != {"pred_masks", "pred_scores", "pred_classes"}:
        raise CrossWindowReplayError("published evaluator fields differ")
    return {
        name: value.detach().cpu().clone() for name, value in prefix.prediction.items()
    }


def _raw_prediction(prediction: OfficialTaskPrediction) -> dict[str, Tensor]:
    prediction.validate()
    return {
        "pred_masks": prediction.latest_stage_masks.detach().cpu().bool().clone(),
        "pred_scores": prediction.pred_scores.detach().cpu().float().clone(),
        "pred_classes": prediction.pred_classes.detach().cpu().long().clone(),
    }


def canonicalize_stage_target(
    target: Mapping[str, object],
    *,
    original_vertex_ids: Tensor,
) -> dict[str, object]:
    required = {
        "gt_ids",
        "gt_classes",
        "gt_masks",
        "changes",
        "change_labels_valid",
        "change_label_semantics",
        "gt_class_semantics",
    }
    if set(target) != required:
        raise CrossWindowReplayError("stage target fields differ")
    masks = target["gt_masks"]
    if not isinstance(masks, Tensor) or masks.ndim != 2 or masks.dtype != torch.bool:
        raise CrossWindowReplayError("stage target masks must be bool [G,N]")
    source_ids = original_vertex_ids.detach().cpu().long()
    if masks.shape[1] != source_ids.numel():
        raise CrossWindowReplayError("stage target points differ from vertex IDs")
    canonical_ids = source_ids.sort().values
    result = {
        name: value.detach().cpu().clone() if isinstance(value, Tensor) else value
        for name, value in target.items()
    }
    result["gt_masks"] = align_mask(
        masks.detach().cpu().T,
        from_ids=source_ids,
        to_ids=canonical_ids,
    ).T.bool()
    return result


def canonical_window_prediction(frame: CanonicalFrame) -> dict[str, Tensor]:
    candidates = tuple(
        candidate
        for candidate in frame.candidates
        if any(
            candidate_slice.mask.any().item() for candidate_slice in candidate.slices
        )
    )
    masks = []
    for scan_id in frame.source_window:
        scan_masks = []
        for candidate in candidates:
            slices = [value for value in candidate.slices if value.scan_id == scan_id]
            if len(slices) != 1:
                raise CrossWindowReplayError("candidate scan slice coverage differs")
            candidate_slice = slices[0]
            scan_masks.append(
                align_mask(
                    candidate_slice.mask,
                    from_ids=candidate_slice.canonical_vertex_ids,
                    to_ids=frame.canonical_vertex_ids[scan_id],
                ).bool()
            )
        masks.append(
            torch.stack(scan_masks, dim=1)
            if scan_masks
            else torch.empty(
                (frame.canonical_vertex_ids[scan_id].numel(), 0), dtype=torch.bool
            )
        )
    return {
        "pred_masks": torch.cat(masks, dim=0),
        "pred_scores": torch.tensor(
            [candidate.score for candidate in candidates], dtype=torch.float32
        ),
        "pred_classes": torch.tensor(
            [candidate.predicted_class_id for candidate in candidates],
            dtype=torch.long,
        ),
    }


def prediction_multiset_equal(
    left: Mapping[str, Tensor], right: Mapping[str, Tensor]
) -> bool:
    if set(left) != {"pred_masks", "pred_scores", "pred_classes"} or set(right) != {
        "pred_masks",
        "pred_scores",
        "pred_classes",
    }:
        raise CrossWindowReplayError("prediction parity fields differ")

    def signatures(value: Mapping[str, Tensor]) -> list[tuple[int, bytes, bytes]]:
        masks = value["pred_masks"].detach().cpu().bool()
        scores = value["pred_scores"].detach().cpu().float()
        classes = value["pred_classes"].detach().cpu().long()
        if (
            masks.ndim != 2
            or masks.shape[1] != scores.numel()
            or scores.shape != classes.shape
        ):
            raise CrossWindowReplayError("prediction parity tensors do not align")
        return sorted(
            (
                int(classes[index].item()),
                scores[index].contiguous().numpy().tobytes(),
                masks[:, index].contiguous().numpy().tobytes(),
            )
            for index in range(scores.numel())
        )

    return signatures(left) == signatures(right)


class E0ReplayAccumulator:
    """Stream DEV-CAL E0 evidence without retaining cache payloads."""

    horizons = (2, 3, 4, 5)

    def __init__(
        self,
        *,
        dataset_spec: str,
        class_mapping: tuple[int, ...],
        checkpoint_sha256: str,
        source_commit: str,
        index_trigger_count: int,
    ) -> None:
        from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
        from scripts.p6a_metrics import OfficialMetricAccumulator

        if len(class_mapping) != 18 or len(set(class_mapping)) != 18:
            raise CrossWindowReplayError("RIO class mapping must contain 18 IDs")
        if index_trigger_count != 0:
            raise CrossWindowReplayError(
                "D-INDEXFIX requires an explicit corrected replay when point order triggers"
            )
        self.dataset_spec = dataset_spec
        self.class_mapping = class_mapping
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_commit = source_commit
        self.index_trigger_count = index_trigger_count
        self.metrics = {
            (method, horizon): AllTBaselineAccumulator(dataset_spec=dataset_spec)
            for method in ("D-LAST", "D-EMA", "A0-U-default")
            for horizon in self.horizons
        }
        self.raw_metrics = {
            horizon: OfficialMetricAccumulator(
                mode="raw_local", dataset_spec=dataset_spec, min_region_size=100
            )
            for horizon in self.horizons
        }
        self.native_t2_metric = AllTBaselineAccumulator(dataset_spec=dataset_spec)
        self.source_rows: list[dict[str, object]] = []
        self.t2_checked = 0
        self.t2_failures: list[str] = []
        self.unit_count = 0
        self.reference_ids: set[str] = set()
        self.a0_collision_fallbacks = 0
        self.route_conflicts = {"D-LAST": 0, "D-EMA": 0}
        self.fallback_matches = {"D-LAST": 0, "D-EMA": 0}
        self.accounting = {
            method: {
                "resident_bytes": 0,
                "archive_bytes": 0,
                "buffer_bytes": 0,
                "materialized_bytes": 0,
            }
            for method in ("D-LAST", "D-EMA", "A0-U-default")
        }

    def _class_mapper(self, value: int) -> int:
        if isinstance(value, bool) or not 0 <= value < len(self.class_mapping):
            raise CrossWindowReplayError("model class is outside the fixed RIO mapping")
        return self.class_mapping[value]

    def update(
        self,
        *,
        logical_unit_id: str,
        base: Mapping[str, object],
        supplement: Mapping[str, object],
    ) -> None:
        from scripts.run_task_memory_controls import (
            prediction_observation_from_payload,
            run_control_trajectory,
        )
        from scripts.run_task_memory_policy_baseline import (
            _meta_from_payload,
            _prediction_from_payload,
            _target_for_prefix,
        )
        from scripts.system_comparison_metrics import validate_causal_prefix_pair
        from scripts.task_memory_output import LagOnePublisher

        base_stages = base.get("stages")
        supplement_stages = supplement.get("stages")
        episode = base.get("episode")
        if (
            not isinstance(base_stages, list)
            or not isinstance(supplement_stages, list)
            or len(base_stages) != 5
            or len(supplement_stages) != 5
            or not isinstance(episode, Mapping)
        ):
            raise CrossWindowReplayError("E0 cache stage coverage differs")
        metas = [_meta_from_payload(stage["stage_meta"]) for stage in base_stages]
        observations = [
            prediction_observation_from_payload(stage["observation"])
            for stage in supplement_stages
        ]
        predictions = [
            _prediction_from_payload(stage["prediction"]) for stage in supplement_stages
        ]
        frames = [
            build_canonical_frame(
                producer_id="R1-B4-policy",
                order_id=logical_unit_id,
                observation=observation,
                prediction=prediction,
                stage_meta=meta,
            )
            for observation, prediction, meta in zip(
                observations, predictions, metas, strict=True
            )
        ]
        targets = [stage["target"] for stage in base_stages]
        canonical_targets = [
            canonicalize_stage_target(
                target,
                original_vertex_ids=meta.original_vertex_ids[-1],
            )
            for target, meta in zip(targets, metas, strict=True)
        ]
        scan_ids = tuple(str(value) for value in episode["scan_ids"])
        if len(scan_ids) != 5:
            raise CrossWindowReplayError("E0 episode scan coverage differs")

        legacy_prefixes: dict[str, list[object]] = {}
        for method, update_mode in (
            ("D-LAST", "last"),
            ("D-EMA", "fixed_ema"),
        ):
            trajectory = run_control_trajectory(
                observations,
                metas,
                update_mode=update_mode,
                capacity=100,
                class_weight=0.25,
                association_threshold=0.5,
                update_rate=0.2,
            )
            publisher = LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
            prefixes = [
                publisher.update(prediction, identity_map, meta)
                for prediction, identity_map, meta in zip(
                    predictions, trajectory.identity_maps, metas, strict=True
                )
            ]
            legacy_prefixes[method] = prefixes
            final_prefix = prefixes[-1]
            self.route_conflicts[method] += sum(
                revision.route_conflicts for revision in final_prefix.revision_log
            )
            self.fallback_matches[method] += sum(
                revision.fallback_matches for revision in final_prefix.revision_log
            )
            self.accounting[method]["resident_bytes"] = max(
                self.accounting[method]["resident_bytes"],
                trajectory.diagnostics.peak_state_bytes,
            )
            self.accounting[method]["archive_bytes"] = max(
                self.accounting[method]["archive_bytes"],
                final_prefix.accounting.archive_payload_bytes,
            )
            self.accounting[method]["buffer_bytes"] = max(
                self.accounting[method]["buffer_bytes"],
                final_prefix.accounting.lag1_buffer_bytes,
            )
            self.accounting[method]["materialized_bytes"] = max(
                self.accounting[method]["materialized_bytes"],
                final_prefix.accounting.materialized_output_bytes,
            )

        first = observations[0]
        state = CrossWindowState.empty(
            capacity=100,
            feature_dim=first.features.shape[-1],
            class_count=first.class_prob.shape[-1],
        )
        buffer = None
        boundary = None
        a0_prefixes = []
        for frame in frames:
            plan = associate(build_evidence(frame, state, buffer), A0_DEFAULT)
            state, committed = commit_observation(frame, state, plan)
            prefix = publish(
                frame,
                plan,
                mask_selection="new",
                history_boundary=boundary,
            )
            buffer = committed.buffer
            boundary = prefix.history_boundary
            a0_prefixes.append(prefix)
            self.a0_collision_fallbacks += prefix.collision_fallback_count
            self.accounting["A0-U-default"]["resident_bytes"] = max(
                self.accounting["A0-U-default"]["resident_bytes"],
                committed.resident_bytes,
            )
            self.accounting["A0-U-default"]["archive_bytes"] = max(
                self.accounting["A0-U-default"]["archive_bytes"],
                prefix.accounting.archive_bytes,
            )
            self.accounting["A0-U-default"]["buffer_bytes"] = max(
                self.accounting["A0-U-default"]["buffer_bytes"],
                committed.buffer_bytes,
            )
            self.accounting["A0-U-default"]["materialized_bytes"] = max(
                self.accounting["A0-U-default"]["materialized_bytes"],
                prefix.accounting.materialized_bytes,
            )

        for stage_index, (base_stage, supplement_stage) in enumerate(
            zip(base_stages, supplement_stages, strict=True)
        ):
            overlap = supplement_stage["base_overlap"]
            self.source_rows.append(
                {
                    "logical_unit_id": logical_unit_id,
                    "reference_id": episode["reference_id"],
                    "sequence_id": episode["sequence_id"],
                    "absolute_stage": stage_index,
                    "generated_candidate_count": overlap["generated_candidate_count"],
                    "base_candidate_count": overlap["base_candidate_count"],
                    "common_candidate_count": overlap["common_candidate_count"],
                    "score_max_abs": overlap["score_max_abs"],
                    "aligned_mask_iou_mean": overlap["aligned_mask_iou_mean"],
                    "generated_vs_base_exact": overlap["exact"],
                    "d_and_a_physical_forward": True,
                }
            )
            horizon = stage_index + 1
            if horizon not in self.horizons:
                continue
            old_target = _target_for_prefix(
                targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            canonical_target = _target_for_prefix(
                canonical_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            for method in ("D-LAST", "D-EMA"):
                pair = validate_causal_prefix_pair(
                    prediction=legacy_prefixes[method][stage_index].prediction,
                    target=old_target,
                    horizon=horizon,
                    observed_scan_ids=scan_ids[:horizon],
                )
                self.metrics[(method, horizon)].update(pair)
            a0_pair = validate_causal_prefix_pair(
                prediction=evaluator_prediction(a0_prefixes[stage_index]),
                target=canonical_target,
                horizon=horizon,
                observed_scan_ids=scan_ids[:horizon],
            )
            self.metrics[("A0-U-default", horizon)].update(a0_pair)
            current_target = _target_for_prefix(
                [targets[stage_index]],
                horizon=1,
                class_mapper=self._class_mapper,
            )
            self.raw_metrics[horizon].update(
                _raw_prediction(predictions[stage_index]), current_target
            )
            if horizon == 2:
                self.t2_checked += 1
                native_prediction = canonical_window_prediction(frames[stage_index])
                if not prediction_multiset_equal(
                    native_prediction, evaluator_prediction(a0_prefixes[stage_index])
                ):
                    self.t2_failures.append(logical_unit_id)
                self.native_t2_metric.update(
                    validate_causal_prefix_pair(
                        prediction=native_prediction,
                        target=canonical_target,
                        horizon=horizon,
                        observed_scan_ids=scan_ids[:horizon],
                    )
                )

        self.unit_count += 1
        self.reference_ids.add(str(episode["reference_id"]))

    def finalize(self) -> dict[str, object]:
        raw = {}
        for horizon, metric in self.raw_metrics.items():
            values = metric.compute()
            raw[horizon] = float(values["raw_local_AP"])
        measured = {
            (method, horizon): accumulator.compute()
            for (method, horizon), accumulator in self.metrics.items()
        }
        aliases = (
            ("D-LEGACY-LAST", "D-LAST", "MEASURED"),
            ("D-INDEXFIX-LAST", "D-LAST", "EQUIVALENT_NO_POINT_ORDER_TRIGGER"),
            (
                "D-CANONICAL-LAST",
                "D-LAST",
                "EQUIVALENT_LEGACY_FALLBACK_ALREADY_LOSSLESS",
            ),
            ("D0", "D-LAST", "MEASURED_CORRECTED_BASELINE"),
            ("D-LEGACY-EMA", "D-EMA", "MEASURED"),
            ("D-INDEXFIX-EMA", "D-EMA", "EQUIVALENT_NO_POINT_ORDER_TRIGGER"),
            (
                "D-CANONICAL-EMA",
                "D-EMA",
                "EQUIVALENT_LEGACY_FALLBACK_ALREADY_LOSSLESS",
            ),
            ("A0-U-default", "A0-U-default", "MEASURED"),
        )
        rows = []
        for label, source, status in aliases:
            for horizon in self.horizons:
                values = measured[(source, horizon)]
                rows.append(
                    {
                        "population_id": "development",
                        "data_role": "DEV-CAL",
                        "reference_count": len(self.reference_ids),
                        "logical_unit_count": self.unit_count,
                        "method": label,
                        "config_id": label,
                        "source_method": source,
                        "source_commit": self.source_commit,
                        "R1_SHA": self.checkpoint_sha256,
                        "head_SHA": None,
                        "producer_id": "R1-B4-policy",
                        "policy": "lag1",
                        "mask_selector": "M-new",
                        "score_mode": "mean",
                        "K": 100,
                        "reference": "all",
                        "master": "all",
                        "order": "all",
                        "T": horizon,
                        "t_mAP": values["t_mAP"],
                        "t_mAP50": values["t_mAP50"],
                        "t_mAP25": values["t_mAP25"],
                        "t_REC": values["t_REC"],
                        "prefix_overall_mAP": values["prefix_overall_mAP"],
                        "raw_current_AP": raw[horizon],
                        "published_current_AP": values["local_current_AP"],
                        **self.accounting[source],
                        "status": status,
                        "reason": "",
                    }
                )
        checkpoint_rows = [
            {
                "checkpoint_family": "R1-B4-policy",
                "checkpoint_sha256": self.checkpoint_sha256,
                "candidate_producer": "R1-B4-policy",
                "output_policy": "native-W2",
                "coverage": "DEV-CAL:T2",
                "status": "MEASURED",
                "reason": "same cached physical forward as lag1 T2",
            },
            {
                "checkpoint_family": "R1-B4-policy",
                "checkpoint_sha256": self.checkpoint_sha256,
                "candidate_producer": "R1-B4-policy",
                "output_policy": "lag1",
                "coverage": "DEV-CAL:T2-T5",
                "status": "MEASURED",
                "reason": "shared development supplement",
            },
            {
                "checkpoint_family": "FH-CONT",
                "checkpoint_sha256": None,
                "candidate_producer": "FH-CONT",
                "output_policy": "native-full-prefix",
                "coverage": "none",
                "status": "BLOCKED_ASSET",
                "reason": "full-history native payload unavailable",
            },
            {
                "checkpoint_family": "FH-CONT",
                "checkpoint_sha256": None,
                "candidate_producer": "FH-CONT",
                "output_policy": "lag1",
                "coverage": "none",
                "status": "BLOCKED_ASSET",
                "reason": "W2 candidates cannot be reconstructed from native FH payload",
            },
        ]
        native_t2 = self.native_t2_metric.compute()
        a0_t2 = measured[("A0-U-default", 2)]
        parity_metric_fields = (
            "t_mAP",
            "t_mAP50",
            "t_mAP25",
            "t_REC",
            "prefix_overall_mAP",
            "local_current_AP",
        )
        metric_differences = {
            field: abs(float(native_t2[field]) - float(a0_t2[field]))
            for field in parity_metric_fields
        }
        parity_pass = not self.t2_failures and max(metric_differences.values()) <= 1e-12
        return {
            "metric_rows": rows,
            "source_rows": self.source_rows,
            "checkpoint_rows": checkpoint_rows,
            "t2_parity": {
                "schema_version": "crosswindow-t2-same-forward-parity-v1",
                "checked_logical_units": self.t2_checked,
                "failure_count": len(self.t2_failures),
                "failed_logical_units": self.t2_failures,
                "candidate_fields": ["mask", "class", "score"],
                "metric_abs_differences": metric_differences,
                "metric_tolerance": 1e-12,
                "status": "PASS" if parity_pass else "FAIL",
            },
            "route_conflicts": self.route_conflicts,
            "fallback_matches": self.fallback_matches,
            "index_trigger_count": self.index_trigger_count,
            "a0_collision_fallback_count": self.a0_collision_fallbacks,
            "status": "PASS" if parity_pass else "FAIL",
        }


_IDENTITY_COUNT_FIELDS = (
    "deployment_id_switches",
    "identity_transition_opportunities",
    "fragmentation_count",
    "fragmentation_opportunities",
    "merge_count",
    "merge_opportunities",
    "gap_opportunities",
    "recovery_attempts",
    "correct_recoveries",
)


def _empty_identity_counts() -> dict[str, int]:
    return {field: 0 for field in _IDENTITY_COUNT_FIELDS}


def _identity_rates(counts: Mapping[str, int]) -> dict[str, int | float | None]:
    normalized = {field: int(counts[field]) for field in _IDENTITY_COUNT_FIELDS}

    def rate(numerator: str, denominator: str) -> float | None:
        total = normalized[denominator]
        return normalized[numerator] / total if total else None

    wrong_reactivations = (
        normalized["recovery_attempts"] - normalized["correct_recoveries"]
    )
    return {
        **normalized,
        "normalized_id_switch_rate": rate(
            "deployment_id_switches", "identity_transition_opportunities"
        ),
        "fragmentation_rate": rate(
            "fragmentation_count", "fragmentation_opportunities"
        ),
        "merge_rate": rate("merge_count", "merge_opportunities"),
        "wrong_reactivations": wrong_reactivations,
        "wrong_reactivation_rate": (
            wrong_reactivations / normalized["recovery_attempts"]
            if normalized["recovery_attempts"]
            else None
        ),
        "gap_recovery_accuracy": rate(
            "correct_recoveries", "recovery_attempts"
        ),
        "gap_recovery_recall": rate("correct_recoveries", "gap_opportunities"),
    }


class E2ReplayAccumulator:
    """Evaluate fixed association configurations on one development role."""

    horizons = (2, 3, 4, 5)

    def __init__(
        self,
        *,
        dataset_spec: str,
        class_mapping: tuple[int, ...],
        checkpoint_sha256: str,
        source_commit: str,
        data_role: str,
        association_configs: Mapping[str, AssociationConfig] | None = None,
        capacity: int = 100,
    ) -> None:
        from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
        from scripts.p6a_metrics import OfficialMetricAccumulator

        if len(class_mapping) != 18 or len(set(class_mapping)) != 18:
            raise CrossWindowReplayError("RIO class mapping must contain 18 IDs")
        if data_role not in {"DEV-CAL", "DEV-SEL", "PROTOCOL-B"}:
            raise CrossWindowReplayError("association evaluation role differs")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise CrossWindowReplayError("association capacity must be positive")
        configs = dict(
            {
                "A0-U-default": A0_DEFAULT,
                "A1-default": A1_DEFAULT,
                "A2-default": A2_DEFAULT,
            }
            if association_configs is None
            else association_configs
        )
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(config, AssociationConfig)
            for name, config in configs.items()
        ):
            raise CrossWindowReplayError("E2 association configurations differ")
        self.dataset_spec = dataset_spec
        self.class_mapping = class_mapping
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_commit = source_commit
        self.data_role = data_role
        self.capacity = capacity
        self.association_configs = configs
        self.methods = ("D0", *configs)
        self.metrics = {
            (method, horizon): AllTBaselineAccumulator(dataset_spec=dataset_spec)
            for method in self.methods
            for horizon in self.horizons
        }
        self.reference_metrics: dict[tuple[str, str, int], object] = {}
        self.raw_metrics = {
            horizon: OfficialMetricAccumulator(
                mode="raw_local", dataset_spec=dataset_spec, min_region_size=100
            )
            for horizon in self.horizons
        }
        self.identity_counts = defaultdict(_empty_identity_counts)
        self.reference_identity_counts = defaultdict(_empty_identity_counts)
        self.assignment_events: list[dict[str, object]] = []
        self.unit_count = 0
        self.reference_ids: set[str] = set()
        self.collision_fallbacks = defaultdict(int)
        self.capacity_accounting = {
            method: {
                "peak_occupied_slots": 0,
                "rejected_births": 0,
                "resident_bytes": 0,
                "buffer_bytes": 0,
                "archive_bytes": 0,
                "materialized_bytes": 0,
            }
            for method in self.methods
        }

    def _class_mapper(self, value: int) -> int:
        if isinstance(value, bool) or not 0 <= value < len(self.class_mapping):
            raise CrossWindowReplayError("model class is outside the fixed RIO mapping")
        return self.class_mapping[value]

    def _reference_metric(self, reference_id: str, method: str, horizon: int):
        from scripts.analyze_persist4d_allt import AllTBaselineAccumulator

        key = (reference_id, method, horizon)
        if key not in self.reference_metrics:
            self.reference_metrics[key] = AllTBaselineAccumulator(
                dataset_spec=self.dataset_spec
            )
        return self.reference_metrics[key]

    @staticmethod
    def _issued_ids(
        keys: Sequence[object], registry: dict[str, int]
    ) -> Tensor:
        values = []
        for key in keys:
            stable_key = f"{type(key).__qualname__}:{key!r}"
            if stable_key not in registry:
                registry[stable_key] = len(registry)
            values.append(registry[stable_key])
        return torch.tensor(values, dtype=torch.long)

    def _record_identity(
        self,
        *,
        method: str,
        reference_id: str,
        prefixes: Sequence[object],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> None:
        from scripts.system_comparison_metrics import (
            compute_deployment_identity_metrics,
            match_identity_update,
        )

        registry: dict[str, int] = {}
        updates = []
        for horizon, (prefix, target) in enumerate(
            zip(prefixes, targets, strict=True), start=1
        ):
            prediction = prefix.prediction
            temporal_stages = target["temporal_stages"].detach().cpu().long()
            current_target = temporal_stages == horizon - 1
            offsets = prefix.scan_vertex_offsets.detach().cpu().long()
            current_start = int(offsets[-2].item())
            if int(current_target.sum().item()) != int(
                prediction["pred_masks"].shape[0] - current_start
            ):
                raise CrossWindowReplayError(
                    "identity current-scan target and prediction differ"
                )
            updates.append(
                match_identity_update(
                    horizon=horizon,
                    gt_ids=target["ids"],
                    gt_classes=target["labels"],
                    gt_masks=target["masks"][:, current_target],
                    issued_ids=self._issued_ids(prefix.keys, registry),
                    pred_classes=prediction["pred_classes"],
                    pred_masks=prediction["pred_masks"][current_start:],
                )
            )
        for horizon in self.horizons:
            values = compute_deployment_identity_metrics(updates[:horizon])
            for scope in (
                self.identity_counts[(method, horizon)],
                self.reference_identity_counts[(reference_id, method, horizon)],
            ):
                for field in _IDENTITY_COUNT_FIELDS:
                    scope[field] += int(values[field])

    def update(
        self,
        *,
        logical_unit_id: str,
        base: Mapping[str, object],
        supplement: Mapping[str, object],
    ) -> None:
        from scripts.run_task_memory_controls import (
            prediction_observation_from_payload,
            run_control_trajectory,
        )
        from scripts.run_task_memory_policy_baseline import (
            _meta_from_payload,
            _prediction_from_payload,
            _target_for_prefix,
        )
        from scripts.system_comparison_metrics import validate_causal_prefix_pair
        from scripts.task_memory_output import LagOnePublisher

        base_stages = base.get("stages")
        supplement_stages = supplement.get("stages")
        episode = base.get("episode")
        if (
            not isinstance(base_stages, list)
            or not isinstance(supplement_stages, list)
            or len(base_stages) != 5
            or len(supplement_stages) != 5
            or not isinstance(episode, Mapping)
        ):
            raise CrossWindowReplayError("E2 cache stage coverage differs")
        reference_id = str(episode["reference_id"])
        scan_ids = tuple(str(value) for value in episode["scan_ids"])
        if len(scan_ids) != 5:
            raise CrossWindowReplayError("E2 episode scan coverage differs")
        metas = [_meta_from_payload(stage["stage_meta"]) for stage in base_stages]
        observations = [
            prediction_observation_from_payload(stage["observation"])
            for stage in supplement_stages
        ]
        predictions = [
            _prediction_from_payload(stage["prediction"])
            for stage in supplement_stages
        ]
        frames = [
            build_canonical_frame(
                producer_id="R1-B4-policy",
                order_id=logical_unit_id,
                observation=observation,
                prediction=prediction,
                stage_meta=meta,
            )
            for observation, prediction, meta in zip(
                observations, predictions, metas, strict=True
            )
        ]
        original_targets = [stage["target"] for stage in base_stages]
        canonical_targets = [
            canonicalize_stage_target(
                target,
                original_vertex_ids=meta.original_vertex_ids[-1],
            )
            for target, meta in zip(original_targets, metas, strict=True)
        ]
        d0_trajectory = run_control_trajectory(
            observations,
            metas,
            update_mode="last",
            capacity=self.capacity,
            class_weight=0.25,
            association_threshold=0.5,
            update_rate=0.2,
        )
        d0_publisher = LagOnePublisher(score_reducer="mean", iou_threshold=0.5)
        d0_prefixes = [
            d0_publisher.update(prediction, identity_map, meta)
            for prediction, identity_map, meta in zip(
                predictions, d0_trajectory.identity_maps, metas, strict=True
            )
        ]
        d0_final = d0_prefixes[-1]
        self.capacity_accounting["D0"]["peak_occupied_slots"] = max(
            self.capacity_accounting["D0"]["peak_occupied_slots"],
            d0_trajectory.diagnostics.peak_occupied_slots,
        )
        self.capacity_accounting["D0"]["rejected_births"] += (
            d0_trajectory.diagnostics.rejected_births
        )
        self.capacity_accounting["D0"]["resident_bytes"] = max(
            self.capacity_accounting["D0"]["resident_bytes"],
            d0_trajectory.diagnostics.peak_state_bytes,
        )
        self.capacity_accounting["D0"]["buffer_bytes"] = max(
            self.capacity_accounting["D0"]["buffer_bytes"],
            d0_final.accounting.lag1_buffer_bytes,
        )
        self.capacity_accounting["D0"]["archive_bytes"] = max(
            self.capacity_accounting["D0"]["archive_bytes"],
            d0_final.accounting.archive_payload_bytes,
        )
        self.capacity_accounting["D0"]["materialized_bytes"] = max(
            self.capacity_accounting["D0"]["materialized_bytes"],
            d0_final.accounting.materialized_output_bytes,
        )

        association_prefixes: dict[str, list[PublishedPrefix]] = {}
        for method, config in self.association_configs.items():
            first = observations[0]
            state = CrossWindowState.empty(
                capacity=self.capacity,
                feature_dim=first.features.shape[-1],
                class_count=first.class_prob.shape[-1],
            )
            buffer = None
            boundary = None
            prefixes = []
            for frame in frames:
                evidence = build_evidence(frame, state, buffer)
                plan = associate(evidence, config)
                state, committed = commit_observation(frame, state, plan)
                prefix = publish(
                    frame,
                    plan,
                    mask_selection="new",
                    history_boundary=boundary,
                )
                buffer = committed.buffer
                boundary = prefix.history_boundary
                prefixes.append(prefix)
                self.collision_fallbacks[method] += prefix.collision_fallback_count
                accounting = self.capacity_accounting[method]
                accounting["peak_occupied_slots"] = max(
                    accounting["peak_occupied_slots"], state.occupied_count
                )
                accounting["rejected_births"] += sum(
                    reason in {"BIRTH_CAPACITY_FULL", "BUFFER_ONLY_CAPACITY_FULL"}
                    for reason in plan.residency_reason
                )
                accounting["resident_bytes"] = max(
                    accounting["resident_bytes"], committed.resident_bytes
                )
                accounting["buffer_bytes"] = max(
                    accounting["buffer_bytes"], committed.buffer_bytes
                )
                accounting["archive_bytes"] = max(
                    accounting["archive_bytes"], prefix.accounting.archive_bytes
                )
                accounting["materialized_bytes"] = max(
                    accounting["materialized_bytes"],
                    prefix.accounting.materialized_bytes,
                )
                matched = sum(anchor >= 0 for anchor in plan.anchor_for_group)
                self.assignment_events.append(
                    {
                        "event_type": "ASSIGNMENT_SUMMARY",
                        "data_role": self.data_role,
                        "logical_unit_id": logical_unit_id,
                        "reference_id": reference_id,
                        "method": method,
                        "config_id": config.config_id,
                        "absolute_stage": frame.absolute_stage,
                        "group_count": evidence.group_count,
                        "anchor_count": evidence.anchor_count,
                        "matched_group_count": matched,
                        "new_identity_count": sum(plan.is_new_id),
                        "nonresident_count": sum(
                            slot < 0 for slot in plan.slot_for_group
                        ),
                        "overlap_lock_count": sum(
                            source == "OVERLAP_LOCK" for source in plan.source_edge
                        ),
                        "feature_edge_count": sum(
                            source == "FEATURE_ASSIGNMENT"
                            for source in plan.source_edge
                        ),
                        "joint_edge_count": sum(
                            source == "JOINT_ASSIGNMENT"
                            for source in plan.source_edge
                        ),
                        "overlap_edge_count": int(evidence.has_overlap.sum().item()),
                        "missing_overlap_edge_count": int(
                            evidence.has_overlap.numel()
                            - evidence.has_overlap.sum().item()
                        ),
                        "zero_norm_query_count": evidence.zero_norm_query_count,
                        "zero_norm_anchor_count": evidence.zero_norm_anchor_count,
                        "mean_decision_score": (
                            sum(plan.decision_score) / len(plan.decision_score)
                            if plan.decision_score
                            else None
                        ),
                        "mean_decision_margin": (
                            sum(plan.decision_margin) / len(plan.decision_margin)
                            if plan.decision_margin
                            else None
                        ),
                        "assignment_sha256": plan.content_sha256,
                    }
                )
            association_prefixes[method] = prefixes

        original_prefix_targets = [
            _target_for_prefix(
                original_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            for horizon in range(1, 6)
        ]
        canonical_prefix_targets = [
            _target_for_prefix(
                canonical_targets,
                horizon=horizon,
                class_mapper=self._class_mapper,
            )
            for horizon in range(1, 6)
        ]
        self._record_identity(
            method="D0",
            reference_id=reference_id,
            prefixes=d0_prefixes,
            targets=original_prefix_targets,
        )
        for method, prefixes in association_prefixes.items():
            self._record_identity(
                method=method,
                reference_id=reference_id,
                prefixes=prefixes,
                targets=canonical_prefix_targets,
            )

        for horizon in self.horizons:
            stage_index = horizon - 1
            method_prefixes = {
                "D0": (d0_prefixes[stage_index], original_prefix_targets[stage_index]),
                **{
                    method: (prefixes[stage_index], canonical_prefix_targets[stage_index])
                    for method, prefixes in association_prefixes.items()
                },
            }
            for method, (prefix, target) in method_prefixes.items():
                prediction = (
                    prefix.prediction
                    if method == "D0"
                    else evaluator_prediction(prefix)
                )
                pair = validate_causal_prefix_pair(
                    prediction=prediction,
                    target=target,
                    horizon=horizon,
                    observed_scan_ids=scan_ids[:horizon],
                )
                self._reference_metric(reference_id, method, horizon).update(pair)
            current_target = _target_for_prefix(
                [original_targets[stage_index]],
                horizon=1,
                class_mapper=self._class_mapper,
            )
            self.raw_metrics[horizon].update(
                _raw_prediction(predictions[stage_index]), current_target
            )

        self.unit_count += 1
        self.reference_ids.add(reference_id)

    def _metric_row(
        self,
        *,
        method: str,
        horizon: int,
        reference: str,
        logical_unit_count: int,
        values: Mapping[str, object],
        raw_current_ap: float | None,
    ) -> dict[str, object]:
        config = self.association_configs.get(method)
        identity_source = (
            self.identity_counts[(method, horizon)]
            if reference == "all"
            else self.reference_identity_counts[(reference, method, horizon)]
        )
        identity = _identity_rates(identity_source)
        return {
            "population_id": "development",
            "data_role": self.data_role,
            "reference_count": len(self.reference_ids) if reference == "all" else 1,
            "logical_unit_count": logical_unit_count,
            "method": method,
            "config_id": config.config_id if config is not None else "D0",
            "source_commit": self.source_commit,
            "R1_SHA": self.checkpoint_sha256,
            "head_SHA": None,
            "producer_id": "R1-B4-policy",
            "policy": "lag1",
            "mask_selector": "M-new",
            "score_mode": "mean",
            "K": self.capacity,
            "reference": reference,
            "master": "all",
            "order": "all",
            "T": horizon,
            "t_mAP": values["t_mAP"],
            "t_mAP50": values["t_mAP50"],
            "t_mAP25": values["t_mAP25"],
            "t_REC": values["t_REC"],
            "prefix_overall_mAP": values["prefix_overall_mAP"],
            "raw_current_AP": raw_current_ap,
            "published_current_AP": values["local_current_AP"],
            **identity,
            "status": "MEASURED",
            "reason": "",
        }

    def finalize(self) -> dict[str, object]:
        for reference in sorted(self.reference_ids):
            for method in self.methods:
                for horizon in self.horizons:
                    self.metrics[(method, horizon)].merge(
                        self.reference_metrics[(reference, method, horizon)]
                    )
        raw = {
            horizon: float(metric.compute()["raw_local_AP"])
            for horizon, metric in self.raw_metrics.items()
        }
        rows = []
        for method in self.methods:
            for horizon in self.horizons:
                rows.append(
                    self._metric_row(
                        method=method,
                        horizon=horizon,
                        reference="all",
                        logical_unit_count=self.unit_count,
                        values=self.metrics[(method, horizon)].compute(),
                        raw_current_ap=raw[horizon],
                    )
                )
        for reference in sorted(self.reference_ids):
            reference_unit_count = 0
            for key in self.reference_metrics:
                if key[0] == reference and key[1] == "D0" and key[2] == 2:
                    reference_unit_count = self.reference_metrics[key].sequence_count
                    break
            for method in self.methods:
                for horizon in self.horizons:
                    rows.append(
                        self._metric_row(
                            method=method,
                            horizon=horizon,
                            reference=reference,
                            logical_unit_count=reference_unit_count,
                            values=self.reference_metrics[
                                (reference, method, horizon)
                            ].compute(),
                            raw_current_ap=None,
                        )
                    )
        return {
            "metric_rows": rows,
            "assignment_events": self.assignment_events,
            "collision_fallbacks": dict(self.collision_fallbacks),
            "capacity_accounting": self.capacity_accounting,
            "status": "PASS",
        }


def _metric_values(value: Mapping[str, object], *, prefix: str) -> dict[str, float]:
    result = {}
    for suffix in ("AP", "AP50", "AP25", "REC"):
        source = f"raw_local_{suffix}"
        number = value.get(source)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise CrossWindowReplayError("raw-local metric output differs")
        result[f"{prefix}_{suffix}"] = float(number)
    return result


def evaluate_raw_current(
    prediction: OfficialTaskPrediction,
    target: Mapping[str, object],
    *,
    published_prefix: object | None = None,
    metric_factory: Callable[[str], object] | None = None,
) -> dict[str, float]:
    """Evaluate raw current candidates; published identities are intentionally ignored."""
    del published_prefix
    if metric_factory is None:
        from scripts.p6a_metrics import OfficialMetricAccumulator

        metric_factory = lambda mode: OfficialMetricAccumulator(mode=mode)
    metric = metric_factory("raw_local")
    metric.update(_raw_prediction(prediction), target)
    return _metric_values(metric.compute(), prefix="raw_current")


def evaluate_published_current(
    prefix: PublishedPrefix,
    target: Mapping[str, object],
    *,
    metric_factory: Callable[[str], object] | None = None,
) -> dict[str, float]:
    if metric_factory is None:
        from scripts.p6a_metrics import OfficialMetricAccumulator

        metric_factory = lambda mode: OfficialMetricAccumulator(mode=mode)
    stop = int(prefix.scan_vertex_offsets[-1].item())
    start = int(prefix.scan_vertex_offsets[-2].item())
    prediction = evaluator_prediction(prefix)
    prediction["pred_masks"] = prediction["pred_masks"][start:stop]
    metric = metric_factory("raw_local")
    metric.update(prediction, target)
    return _metric_values(metric.compute(), prefix="published_current")


__all__ = [
    "CrossWindowReplayError",
    "E0ReplayAccumulator",
    "HistoryBoundary",
    "PublicationAccounting",
    "PublicationRevision",
    "PublishedOccurrence",
    "PublishedPrefix",
    "PublishedScan",
    "canonical_window_prediction",
    "canonicalize_stage_target",
    "evaluate_published_current",
    "evaluate_raw_current",
    "evaluator_prediction",
    "prediction_multiset_equal",
    "publish",
]
