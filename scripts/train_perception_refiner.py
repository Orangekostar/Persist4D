#!/usr/bin/env python3
"""Train the frozen-parent causal mask refiner for Perception Gain V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from models.perception_gain import CausalMaskRefiner, pair_adjacent_soft_occurrences
from trainer.task_memory_trainer import (
    capture_task_memory_rng_state,
    restore_task_memory_rng_state,
)

REFINER_UPDATES = 1500
REFINER_WARMUP_UPDATES = 75
REFINER_MINIMUM_LR_FRACTION = 0.1
REFINER_BATCH_CANDIDATES = 16
REFINER_MAXIMUM_SEGMENTS = 2048
_RESUME_SCHEMA = "perception-refiner-resume-v1"
_FROZEN_SCHEMA = "perception-refiner-frozen-v1"


class RefinerTrainingError(RuntimeError):
    """Raised when refiner data or optimization violates the frozen contract."""


def _integer_vector(value: object, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
    ):
        raise RefinerTrainingError(f"{name} must be an integer vector")
    return value.detach().long().contiguous()


def _floating_vector(value: object, *, name: str) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 1
        or not value.is_floating_point()
        or not torch.isfinite(value).all().item()
    ):
        raise RefinerTrainingError(f"{name} must be a finite floating vector")
    return value.detach().float().contiguous()


def align_old_probabilities_to_new_segments(
    *,
    old_vertex_ids: Tensor,
    old_point_probabilities: Tensor,
    new_vertex_ids: Tensor,
    new_low_point2segment: Tensor,
    new_voxel_inverse: Tensor,
    new_segment_ids: Tensor,
) -> Tensor:
    """Align old point probabilities by vertex ID and average on new segments."""

    old_ids = _integer_vector(old_vertex_ids, name="old vertex IDs").cpu()
    new_ids = _integer_vector(new_vertex_ids, name="new vertex IDs").cpu()
    probabilities = _floating_vector(
        old_point_probabilities, name="old point probabilities"
    )
    low_point2segment = _integer_vector(
        new_low_point2segment, name="new low point2segment"
    ).cpu()
    voxel_inverse = _integer_vector(new_voxel_inverse, name="new voxel inverse").cpu()
    segment_ids = _integer_vector(new_segment_ids, name="new segment IDs").cpu()
    if (
        old_ids.numel() != probabilities.numel()
        or old_ids.numel() != new_ids.numel()
        or old_ids.unique().numel() != old_ids.numel()
        or new_ids.unique().numel() != new_ids.numel()
        or not torch.equal(old_ids.sort().values, new_ids.sort().values)
    ):
        raise RefinerTrainingError("old/new vertex identities differ")
    if (
        probabilities.numel() == 0
        or torch.any((probabilities < 0) | (probabilities > 1)).item()
        or voxel_inverse.numel() != new_ids.numel()
        or voxel_inverse.numel() == 0
        or voxel_inverse.min().item() < 0
        or voxel_inverse.max().item() >= low_point2segment.numel()
        or segment_ids.numel() == 0
        or segment_ids.unique().numel() != segment_ids.numel()
    ):
        raise RefinerTrainingError("refiner point/segment mapping is invalid")

    old_position = {int(vertex): index for index, vertex in enumerate(old_ids.tolist())}
    reorder = torch.tensor(
        [old_position[int(vertex)] for vertex in new_ids.tolist()],
        dtype=torch.long,
        device=probabilities.device,
    )
    aligned_probabilities = probabilities[reorder]
    full_segment_ids = low_point2segment[voxel_inverse].to(probabilities.device)
    values = []
    for segment_id in segment_ids.tolist():
        selector = full_segment_ids == int(segment_id)
        if not selector.any().item():
            raise RefinerTrainingError("new segment lacks full-resolution vertices")
        values.append(aligned_probabilities[selector].mean())
    means = torch.stack(values).clamp(1.0e-4, 1.0 - 1.0e-4)
    return torch.logit(means)


class CausalRefinerRevisionTransform:
    """Refine only the lag-one revision mask after D0 identity resolution."""

    def __init__(
        self,
        *,
        system: object,
        refiner: CausalMaskRefiner,
        device: str | torch.device,
    ) -> None:
        if not isinstance(refiner, CausalMaskRefiner):
            raise RefinerTrainingError("revision refiner has the wrong type")
        self.system = system
        self.device = torch.device(device)
        self.refiner = refiner.to(self.device).eval()
        self._old_stage: tuple[object, object] | None = None
        self._current_stage: tuple[object, object] | None = None
        self.last_audit: dict[str, int] = {
            "paired_candidate_count": 0,
            "fallback_candidate_count": 0,
        }
        self.last_candidate_records: tuple[dict[str, object], ...] = ()

    def observe_stage(self, prediction: object, stage_meta: object) -> None:
        from datasets.task_memory_episode import StageMeta
        from scripts.rescene_task_postprocess import OfficialTaskPrediction

        if not isinstance(prediction, OfficialTaskPrediction) or not isinstance(
            stage_meta, StageMeta
        ):
            raise RefinerTrainingError("revision observation has the wrong type")
        prediction.validate()
        if prediction.soft_evidence is None:
            raise RefinerTrainingError("revision observation lacks soft evidence")
        if self._current_stage is None or stage_meta.absolute_stage_index == 0:
            self._old_stage = None
            self._current_stage = (prediction, stage_meta)
            return
        previous_prediction, previous_meta = self._current_stage
        if (
            previous_meta.reference_id != stage_meta.reference_id
            or previous_meta.episode_id != stage_meta.episode_id
            or stage_meta.absolute_stage_index != previous_meta.absolute_stage_index + 1
        ):
            raise RefinerTrainingError("revision observations are not adjacent")
        self._old_stage = (previous_prediction, previous_meta)
        self._current_stage = (prediction, stage_meta)

    @staticmethod
    def _occurrence(candidate: object, evidence: object, *, scan_id: str, window: int):
        from models.perception_gain import SoftOccurrence

        index = candidate.candidate_index
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < evidence.source_query_ids.numel()
            or int(evidence.source_query_ids[index].item()) != candidate.source_query_id
        ):
            raise RefinerTrainingError("revision candidate lineage differs")
        return SoftOccurrence(
            scan_id=scan_id,
            window_index=window,
            logical_id=candidate.identity.logical_id,
            generation=candidate.identity.generation,
            source_class_id=int(evidence.source_class_ids[index].item()),
            source_query_id=candidate.source_query_id,
            candidate_index=index,
        )

    def __call__(self, request: object) -> Mapping[int, Tensor]:
        from scripts.rescene_task_postprocess import materialize_segment_logits
        from scripts.task_memory_output import RevisionMaskRequest

        if not isinstance(request, RevisionMaskRequest):
            raise RefinerTrainingError("revision mask request has the wrong type")
        if self._old_stage is None or self._current_stage is None:
            raise RefinerTrainingError("revision transform lacks adjacent observations")
        old_prediction, old_meta = self._old_stage
        new_prediction, new_meta = self._current_stage
        if (
            request.absolute_stage_index != new_meta.absolute_stage_index
            or request.scan_id != old_meta.scan_ids_in_window[-1]
            or request.scan_id != new_meta.scan_ids_in_window[0]
            or request.augmentation_transform_id != new_meta.augmentation_transform_id
            or request.coordinate_frame_id != new_meta.coordinate_frame_id
            or old_meta.augmentation_transform_id != new_meta.augmentation_transform_id
            or old_meta.coordinate_frame_id != new_meta.coordinate_frame_id
            or not torch.equal(
                request.source_vertex_ids,
                new_meta.original_vertex_ids[0].detach().cpu(),
            )
            or set(request.target_vertex_ids.tolist())
            != set(old_meta.original_vertex_ids[-1].detach().cpu().tolist())
        ):
            raise RefinerTrainingError("revision request differs from observed stages")
        old_evidence = old_prediction.soft_evidence
        new_evidence = new_prediction.soft_evidence
        if old_evidence is None or new_evidence is None:
            raise RefinerTrainingError("revision soft evidence is unavailable")

        old_occurrences = tuple(
            self._occurrence(
                candidate,
                old_evidence,
                scan_id=request.scan_id,
                window=old_meta.absolute_stage_index,
            )
            for candidate in request.old_candidates
        )
        new_occurrences = tuple(
            self._occurrence(
                candidate,
                new_evidence,
                scan_id=request.scan_id,
                window=new_meta.absolute_stage_index,
            )
            for candidate in request.new_candidates
        )
        decisions = pair_adjacent_soft_occurrences(
            old_occurrences=old_occurrences,
            new_occurrences=new_occurrences,
        )
        segment_ids = torch.nonzero(
            new_meta.segment_stage_ids == 0, as_tuple=False
        ).flatten()
        if segment_ids.numel() == 0:
            raise RefinerTrainingError("revision scan has no segments")
        old_start = int(old_meta.scan_vertex_offsets[-2].item())
        old_stop = int(old_meta.scan_vertex_offsets[-1].item())
        new_stop = int(new_meta.scan_vertex_offsets[1].item())
        refined_segment_logits = new_evidence.segment_logits.clone()
        paired_indices = []
        candidate_records = []
        old_by_index = {
            value.candidate_index: index for index, value in enumerate(old_occurrences)
        }
        new_by_index = {
            value.candidate_index: index for index, value in enumerate(new_occurrences)
        }
        with torch.inference_mode():
            for decision in decisions:
                new_occurrence = new_occurrences[decision.new_index]
                if decision.old_index is None:
                    continue
                old_occurrence = old_occurrences[decision.old_index]
                old_candidate = request.old_candidates[
                    old_by_index[old_occurrence.candidate_index]
                ]
                new_candidate = request.new_candidates[
                    new_by_index[new_occurrence.candidate_index]
                ]
                old_logits = align_old_probabilities_to_new_segments(
                    old_vertex_ids=old_meta.original_vertex_ids[-1],
                    old_point_probabilities=old_evidence.full_resolution_probabilities[
                        old_start:old_stop, old_occurrence.candidate_index
                    ],
                    new_vertex_ids=new_meta.original_vertex_ids[0],
                    new_low_point2segment=new_evidence.low_point2segment,
                    new_voxel_inverse=new_evidence.voxel_inverse[:new_stop],
                    new_segment_ids=segment_ids,
                )
                new_logits = new_evidence.segment_logits[
                    segment_ids, new_occurrence.candidate_index
                ]
                candidate_records.append(
                    {
                        "reference_id": new_meta.reference_id,
                        "episode_id": new_meta.episode_id,
                        "candidate_id": (
                            f"{request.scan_id}:{new_meta.absolute_stage_index}:"
                            f"{new_occurrence.logical_id!r}:"
                            f"{new_occurrence.generation}:"
                            f"{new_occurrence.source_class_id}:"
                            f"{new_occurrence.candidate_index}"
                        ),
                        "candidate_index": new_occurrence.candidate_index,
                        "scan_id": request.scan_id,
                        "absolute_stage_index": new_meta.absolute_stage_index,
                        "logical_id": new_occurrence.logical_id,
                        "generation": new_occurrence.generation,
                        "source_class_id": new_occurrence.source_class_id,
                        "source_query_id": new_occurrence.source_query_id,
                        "segment_ids": segment_ids.clone(),
                        "new_features": new_evidence.segment_features[
                            segment_ids
                        ].clone(),
                        "old_logits": old_logits.clone(),
                        "new_logits": new_logits.clone(),
                        "old_score": old_candidate.score,
                        "new_score": new_candidate.score,
                    }
                )
                refined, _ = self.refiner(
                    new_features=new_evidence.segment_features[segment_ids].to(
                        self.device
                    ),
                    old_logits=old_logits.to(self.device),
                    new_logits=new_logits.to(self.device),
                    old_score=old_candidate.score,
                    new_score=new_candidate.score,
                )
                refined_segment_logits[segment_ids, new_occurrence.candidate_index] = (
                    refined.detach().cpu()
                )
                paired_indices.append(new_occurrence.candidate_index)
        full_masks = materialize_segment_logits(
            system=self.system,
            evidence=new_evidence,
            segment_logits=refined_segment_logits,
        )
        self.last_audit = {
            "paired_candidate_count": len(paired_indices),
            "fallback_candidate_count": len(decisions) - len(paired_indices),
        }
        self.last_candidate_records = tuple(candidate_records)
        return {index: full_masks[:new_stop, index].clone() for index in paired_indices}


def match_refiner_candidates_to_gt(
    *,
    candidate_masks: Tensor,
    candidate_classes: Tensor,
    gt_masks: Tensor,
    gt_classes: Tensor,
    iou_threshold: float = 0.1,
) -> dict[str, object]:
    """Match unrefined candidates to class-compatible GT with tie rejection."""

    from models.persistent_memory import _optimal_assignment_with_stable_ties

    if (
        not isinstance(candidate_masks, Tensor)
        or candidate_masks.ndim != 2
        or candidate_masks.dtype != torch.bool
        or not isinstance(gt_masks, Tensor)
        or gt_masks.ndim != 2
        or gt_masks.dtype != torch.bool
        or candidate_masks.shape[0] != gt_masks.shape[1]
        or not isinstance(candidate_classes, Tensor)
        or candidate_classes.ndim != 1
        or candidate_classes.numel() != candidate_masks.shape[1]
        or candidate_classes.dtype == torch.bool
        or candidate_classes.is_floating_point()
        or not isinstance(gt_classes, Tensor)
        or gt_classes.ndim != 1
        or gt_classes.numel() != gt_masks.shape[0]
        or gt_classes.dtype == torch.bool
        or gt_classes.is_floating_point()
        or isinstance(iou_threshold, bool)
        or not isinstance(iou_threshold, (int, float))
        or not math.isfinite(float(iou_threshold))
        or not 0.0 <= float(iou_threshold) <= 1.0
    ):
        raise RefinerTrainingError("refiner GT matching inputs are invalid")
    candidate_masks = candidate_masks.detach().cpu()
    gt_masks = gt_masks.detach().cpu()
    candidate_classes = candidate_classes.detach().cpu().long()
    gt_classes = gt_classes.detach().cpu().long()
    assignments: list[int | None] = [None] * candidate_masks.shape[1]
    ambiguous: set[int] = set()
    for class_id in sorted(set(candidate_classes.tolist()) & set(gt_classes.tolist())):
        candidate_indices = torch.nonzero(
            candidate_classes == class_id, as_tuple=False
        ).flatten()
        gt_indices = torch.nonzero(gt_classes == class_id, as_tuple=False).flatten()
        if not candidate_indices.numel() or not gt_indices.numel():
            continue
        candidates = candidate_masks[:, candidate_indices].T
        targets = gt_masks[gt_indices]
        intersection = (candidates[:, None] & targets[None]).sum(dim=-1).double()
        union = (candidates[:, None] | targets[None]).sum(dim=-1).double()
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        rows, columns = _optimal_assignment_with_stable_ties(iou)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            value = float(iou[row, column].item())
            if value < iou_threshold:
                continue
            equal = torch.isclose(
                iou,
                torch.tensor(value, dtype=iou.dtype),
                atol=1.0e-12,
                rtol=0.0,
            ) & (iou >= iou_threshold)
            row_ties = torch.nonzero(equal[row], as_tuple=False).flatten()
            column_ties = torch.nonzero(equal[:, column], as_tuple=False).flatten()
            if row_ties.numel() > 1 or column_ties.numel() > 1:
                ambiguous.update(
                    int(candidate_indices[index].item())
                    for index in column_ties.tolist()
                )
                ambiguous.add(int(candidate_indices[row].item()))
                continue
            assignments[int(candidate_indices[row].item())] = int(
                gt_indices[column].item()
            )
    for index in ambiguous:
        assignments[index] = None
    matched_count = sum(value is not None for value in assignments)
    return {
        "assignments": tuple(assignments),
        "ambiguous_candidate_indices": tuple(sorted(ambiguous)),
        "matched_candidate_count": matched_count,
        "unmatched_candidate_count": len(assignments) - matched_count - len(ambiguous),
    }


def build_refiner_training_records(
    *,
    pair_records: Sequence[Mapping[str, Any]],
    new_prediction: object,
    full_target: Mapping[str, object],
    stage_meta: object,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    """Attach offline GT segment targets to already paired soft occurrences."""

    from datasets.task_memory_episode import StageMeta
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    if (
        isinstance(pair_records, (str, bytes))
        or not isinstance(pair_records, Sequence)
        or not isinstance(new_prediction, OfficialTaskPrediction)
        or not isinstance(stage_meta, StageMeta)
        or not isinstance(full_target, Mapping)
    ):
        raise RefinerTrainingError("refiner training record inputs are invalid")
    new_prediction.validate()
    evidence = new_prediction.soft_evidence
    masks = full_target.get("masks")
    labels = full_target.get("labels")
    if (
        evidence is None
        or not isinstance(masks, Tensor)
        or masks.ndim != 2
        or masks.dtype != torch.bool
        or not isinstance(labels, Tensor)
        or labels.ndim != 1
        or labels.numel() != masks.shape[0]
        or labels.dtype == torch.bool
        or labels.is_floating_point()
        or masks.shape[1] != stage_meta.local_stage_ids.numel()
    ):
        raise RefinerTrainingError("refiner full target is invalid")
    selector = stage_meta.local_stage_ids == 0
    previous_point_count = int(stage_meta.scan_vertex_offsets[1].item())
    if (
        int(selector.sum().item()) != previous_point_count
        or not torch.all(selector[:previous_point_count]).item()
        or new_prediction.pred_masks.shape[0] != selector.numel()
    ):
        raise RefinerTrainingError("refiner previous-scan target lineage differs")
    previous_gt_masks = masks.detach().cpu()[:, selector]
    present = previous_gt_masks.any(dim=1)
    gt_indices = torch.nonzero(present, as_tuple=False).flatten()
    matching = match_refiner_candidates_to_gt(
        candidate_masks=new_prediction.pred_masks.detach().cpu()[selector],
        candidate_classes=evidence.source_class_ids,
        gt_masks=previous_gt_masks[gt_indices],
        gt_classes=labels.detach().cpu().long()[gt_indices],
    )
    assignments = matching["assignments"]
    ambiguous = set(matching["ambiguous_candidate_indices"])
    full_segment_ids = stage_meta.full_resolution_point2segment[selector]
    records = []
    matched_count = 0
    empty_count = 0
    ambiguous_count = 0
    for source in pair_records:
        if not isinstance(source, Mapping):
            raise RefinerTrainingError("paired refiner record must be a mapping")
        candidate_index = source.get("candidate_index")
        segment_ids = source.get("segment_ids")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or not 0 <= candidate_index < len(assignments)
            or not isinstance(segment_ids, Tensor)
            or segment_ids.ndim != 1
            or segment_ids.dtype == torch.bool
            or segment_ids.is_floating_point()
            or segment_ids.numel() == 0
        ):
            raise RefinerTrainingError("paired refiner record lineage is invalid")
        if candidate_index in ambiguous:
            ambiguous_count += 1
            continue
        assignment = assignments[candidate_index]
        if assignment is None:
            target = torch.zeros(segment_ids.numel(), dtype=torch.float32)
            empty_count += 1
        else:
            original_gt_index = int(gt_indices[assignment].item())
            gt_mask = previous_gt_masks[original_gt_index].float()
            values = []
            for segment_id in segment_ids.detach().cpu().long().tolist():
                segment_selector = full_segment_ids == segment_id
                if not segment_selector.any().item():
                    raise RefinerTrainingError(
                        "paired refiner segment lacks previous-scan vertices"
                    )
                values.append(gt_mask[segment_selector].mean())
            target = torch.stack(values).float().contiguous()
            matched_count += 1
        record = dict(source)
        record.pop("segment_ids", None)
        record.pop("candidate_index", None)
        record["target"] = target
        records.append(record)
    audit = {
        "paired_candidate_count": len(pair_records),
        "written_candidate_count": len(records),
        "matched_candidate_count": matched_count,
        "empty_target_candidate_count": empty_count,
        "ambiguous_candidate_count": ambiguous_count,
    }
    return tuple(records), audit


def refiner_candidate_loss(
    *,
    refined_logits: Tensor,
    targets: Tensor,
    delta: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    if (
        not isinstance(refined_logits, Tensor)
        or not isinstance(targets, Tensor)
        or not isinstance(delta, Tensor)
        or refined_logits.ndim != 1
        or targets.shape != refined_logits.shape
        or delta.shape != refined_logits.shape
        or refined_logits.numel() == 0
        or not refined_logits.is_floating_point()
        or not targets.is_floating_point()
        or not delta.is_floating_point()
        or not torch.isfinite(refined_logits).all().item()
        or not torch.isfinite(targets).all().item()
        or not torch.isfinite(delta).all().item()
        or torch.any((targets < 0) | (targets > 1)).item()
    ):
        raise RefinerTrainingError("refiner loss tensors are invalid")
    target_values = targets.to(device=refined_logits.device, dtype=refined_logits.dtype)
    delta_values = delta.to(device=refined_logits.device, dtype=refined_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(refined_logits, target_values)
    probabilities = refined_logits.sigmoid()
    dice = 1 - (2 * (probabilities * target_values).sum() + 1) / (
        probabilities.sum() + target_values.sum() + 1
    )
    regularization = 0.01 * delta_values.square().mean()
    loss = bce + dice + regularization
    return loss, {
        "bce": float(bce.detach().cpu().item()),
        "dice": float(dice.detach().cpu().item()),
        "delta_regularization": float(regularization.detach().cpu().item()),
    }


def refiner_lr_multiplier(
    step: int,
    *,
    total_updates: int = REFINER_UPDATES,
    warmup_updates: int = REFINER_WARMUP_UPDATES,
    minimum_fraction: float = REFINER_MINIMUM_LR_FRACTION,
) -> float:
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_updates, bool)
        or not isinstance(total_updates, int)
        or isinstance(warmup_updates, bool)
        or not isinstance(warmup_updates, int)
        or total_updates < 2
        or not 1 <= warmup_updates < total_updates
        or not 0 <= step < total_updates
        or not 0.0 < minimum_fraction <= 1.0
    ):
        raise RefinerTrainingError("refiner scheduler arguments are invalid")
    if step < warmup_updates:
        return (step + 1) / warmup_updates
    denominator = max(1, total_updates - 1 - warmup_updates)
    progress = min(1.0, (step - warmup_updates) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


def _hash_index(namespace: str, values: Sequence[object], size: int) -> int:
    digest = hashlib.sha256(
        (namespace + ":" + ":".join(str(value) for value in values)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % size


def sample_refiner_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int,
    cursor: int = 0,
) -> tuple[Mapping[str, Any], ...]:
    """Sample reference, then episode, then candidate with exact cursor resume."""

    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or not records
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        or isinstance(cursor, bool)
        or not isinstance(cursor, int)
        or cursor < 0
    ):
        raise RefinerTrainingError("refiner sampling arguments are invalid")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    identities = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise RefinerTrainingError("refiner record must be a mapping")
        reference = record.get("reference_id")
        episode = record.get("episode_id")
        candidate = record.get("candidate_id")
        if any(
            not isinstance(value, str) or not value
            for value in (reference, episode, candidate)
        ):
            raise RefinerTrainingError("refiner record identity is invalid")
        identity = (reference, episode, candidate)
        if identity in identities:
            raise RefinerTrainingError("refiner candidate identities must be unique")
        identities.add(identity)
        grouped.setdefault(reference, {}).setdefault(episode, []).append(record)

    references = sorted(
        grouped,
        key=lambda value: (
            hashlib.sha256(
                f"pgv1-refiner-sample-ref:{seed}:{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )
    result = []
    for draw in range(cursor, cursor + count):
        reference = references[draw % len(references)]
        episodes = sorted(grouped[reference])
        episode = episodes[
            _hash_index(
                "pgv1-refiner-sample-episode", (seed, reference, draw), len(episodes)
            )
        ]
        candidates = sorted(
            grouped[reference][episode], key=lambda row: str(row["candidate_id"])
        )
        result.append(
            candidates[
                _hash_index(
                    "pgv1-refiner-sample-candidate",
                    (seed, reference, episode, draw),
                    len(candidates),
                )
            ]
        )
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_refiner_records(
    cache_paths: Sequence[Path],
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    required = {
        "reference_id",
        "episode_id",
        "candidate_id",
        "new_features",
        "old_logits",
        "new_logits",
        "old_score",
        "new_score",
        "target",
    }
    for path in cache_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, (str, bytes)) or not isinstance(payload, Sequence):
            raise RefinerTrainingError(f"invalid refiner shard: {path}")
        for record in payload:
            if not isinstance(record, Mapping) or not required <= set(record):
                raise RefinerTrainingError("refiner record fields are incomplete")
            features = record["new_features"]
            old_logits = record["old_logits"]
            new_logits = record["new_logits"]
            target = record["target"]
            if (
                not isinstance(features, Tensor)
                or features.ndim != 2
                or features.shape[1] != 128
                or not features.is_floating_point()
                or not torch.isfinite(features).all().item()
                or not isinstance(old_logits, Tensor)
                or not isinstance(new_logits, Tensor)
                or not isinstance(target, Tensor)
                or old_logits.shape != (features.shape[0],)
                or new_logits.shape != old_logits.shape
                or target.shape != old_logits.shape
                or features.shape[0] == 0
                or not old_logits.is_floating_point()
                or not new_logits.is_floating_point()
                or not target.is_floating_point()
                or not torch.isfinite(old_logits).all().item()
                or not torch.isfinite(new_logits).all().item()
                or not torch.isfinite(target).all().item()
                or torch.any((target < 0) | (target > 1)).item()
            ):
                raise RefinerTrainingError("refiner record tensors are invalid")
            for score_name in ("old_score", "new_score"):
                score = record[score_name]
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise RefinerTrainingError("refiner record score is invalid")
            records.append(record)
    if not records:
        raise RefinerTrainingError("refiner cache contains no candidates")
    sample_refiner_candidates(records, count=0, seed=45)
    return tuple(records)


def _candidate_tensors(
    record: Mapping[str, Any],
    *,
    maximum_segments: int,
    seed: int,
    draw_index: int,
    device: str | torch.device,
) -> dict[str, Tensor | float]:
    segment_count = int(record["new_logits"].numel())
    take = min(segment_count, maximum_segments)
    generator = torch.Generator().manual_seed(
        int.from_bytes(
            hashlib.sha256(
                (
                    f"pgv1-refiner-segments:{seed}:{draw_index}:"
                    f"{record['candidate_id']}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
    )
    indices = torch.randperm(segment_count, generator=generator)[:take]
    return {
        "new_features": record["new_features"][indices].float().to(device),
        "old_logits": record["old_logits"][indices].float().to(device),
        "new_logits": record["new_logits"][indices].float().to(device),
        "old_score": float(record["old_score"]),
        "new_score": float(record["new_score"]),
        "target": record["target"][indices].float().to(device),
    }


def _frozen_payload(
    refiner: CausalMaskRefiner,
    *,
    cache_paths: Sequence[Path],
    updates: int,
) -> dict[str, object]:
    return {
        "refiner_state_dict": {
            name: tensor.detach().cpu() for name, tensor in refiner.state_dict().items()
        },
        "schema_version": _FROZEN_SCHEMA,
        "source_shards": [str(path) for path in cache_paths],
        "updates": updates,
    }


def train_mask_refiner(
    *,
    cache_paths: Sequence[Path],
    output_dir: Path,
    stop_after_updates: int = REFINER_UPDATES,
    resume: Path | None = None,
    device: str = "cuda",
    batch_candidates: int = REFINER_BATCH_CANDIDATES,
    maximum_segments: int = REFINER_MAXIMUM_SEGMENTS,
    seed: int = 45,
) -> dict[str, object]:
    if (
        isinstance(stop_after_updates, bool)
        or not isinstance(stop_after_updates, int)
        or not 1 <= stop_after_updates <= REFINER_UPDATES
        or isinstance(batch_candidates, bool)
        or not isinstance(batch_candidates, int)
        or batch_candidates <= 0
        or isinstance(maximum_segments, bool)
        or not isinstance(maximum_segments, int)
        or maximum_segments <= 0
        or seed not in {45, 46}
    ):
        raise RefinerTrainingError("refiner endpoint or batch contract is invalid")
    records = _load_refiner_records(cache_paths)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    refiner = CausalMaskRefiner().to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    completed = 0
    losses: list[float] = []
    training_audit: list[dict[str, float | int]] = []
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        required = {
            "completed_updates",
            "losses",
            "optimizer_state_dict",
            "refiner_state_dict",
            "rng",
            "schema_version",
            "seed",
            "training_audit",
        }
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != required:
            raise RefinerTrainingError("refiner resume checkpoint is invalid")
        if checkpoint["schema_version"] != _RESUME_SCHEMA:
            raise RefinerTrainingError("refiner resume schema differs")
        if checkpoint["seed"] != seed:
            raise RefinerTrainingError("refiner resume seed differs")
        refiner.load_state_dict(checkpoint["refiner_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed = int(checkpoint["completed_updates"])
        losses = list(checkpoint["losses"])
        training_audit = list(checkpoint["training_audit"])
        if not isinstance(checkpoint["rng"], Mapping):
            raise RefinerTrainingError("refiner resume RNG is invalid")
        restore_task_memory_rng_state(checkpoint["rng"])
    if completed > stop_after_updates:
        raise RefinerTrainingError("refiner resume exceeds the requested endpoint")

    output_dir.mkdir(parents=True, exist_ok=True)
    zero_checkpoint = output_dir / "update=0000.ckpt"
    if completed == 0 and not zero_checkpoint.exists():
        _atomic_torch_save(
            zero_checkpoint,
            _frozen_payload(refiner, cache_paths=cache_paths, updates=0),
        )
    started = time.perf_counter()
    for update in range(completed, stop_after_updates):
        learning_rate = 1.0e-3 * refiner_lr_multiplier(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        sampled = sample_refiner_candidates(
            records,
            count=batch_candidates,
            seed=seed,
            cursor=update * batch_candidates,
        )
        before = (
            {
                name: parameter.detach().cpu().clone()
                for name, parameter in refiner.named_parameters()
            }
            if len(training_audit) < 2
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        candidate_losses = []
        for slot, record in enumerate(sampled):
            tensors = _candidate_tensors(
                record,
                maximum_segments=maximum_segments,
                seed=seed,
                draw_index=update * batch_candidates + slot,
                device=device,
            )
            refined, delta = refiner(
                new_features=tensors["new_features"],
                old_logits=tensors["old_logits"],
                new_logits=tensors["new_logits"],
                old_score=tensors["old_score"],
                new_score=tensors["new_score"],
            )
            candidate_loss, _ = refiner_candidate_loss(
                refined_logits=refined,
                targets=tensors["target"],
                delta=delta,
            )
            candidate_losses.append(candidate_loss)
        loss = torch.stack(candidate_losses).mean()
        loss.backward()
        gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().norm().item()) ** 2
                for parameter in refiner.parameters()
                if parameter.grad is not None
            )
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if before is not None:
            change_norm = math.sqrt(
                sum(
                    float(
                        (parameter.detach().cpu() - before[name]).float().norm().item()
                    )
                    ** 2
                    for name, parameter in refiner.named_parameters()
                )
            )
            training_audit.append(
                {
                    "gradient_norm": gradient_norm,
                    "optimizer_update": update + 1,
                    "parameter_change_norm": change_norm,
                }
            )
        resume_payload = {
            "completed_updates": update + 1,
            "losses": losses,
            "optimizer_state_dict": optimizer.state_dict(),
            "refiner_state_dict": refiner.state_dict(),
            "rng": capture_task_memory_rng_state(),
            "schema_version": _RESUME_SCHEMA,
            "seed": seed,
            "training_audit": training_audit,
        }
        if (update + 1) % 50 == 0 or update + 1 == stop_after_updates:
            _atomic_torch_save(output_dir / "last.ckpt", resume_payload)
        if (update + 1) in {500, 1000, 1500}:
            _atomic_torch_save(
                output_dir / f"update={update + 1:04d}.ckpt",
                _frozen_payload(refiner, cache_paths=cache_paths, updates=update + 1),
            )
    checkpoint_path = output_dir / f"update={stop_after_updates:04d}.ckpt"
    _atomic_torch_save(
        checkpoint_path,
        _frozen_payload(refiner, cache_paths=cache_paths, updates=stop_after_updates),
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": "perception-refiner-training-summary-v1",
        "status": "COMPLETE" if stop_after_updates == REFINER_UPDATES else "SMOKE",
        "completed_updates": stop_after_updates,
        "batch_candidates": batch_candidates,
        "maximum_segments": maximum_segments,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0 if str(device).startswith("cuda") else 0.0,
        "final_loss": losses[-1],
        "reference_count": len({str(record["reference_id"]) for record in records}),
        "seed": seed,
        "candidate_count": len(records),
        "training_audit": training_audit,
    }
    _atomic_json(output_dir / "run_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-updates", type=int, default=REFINER_UPDATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=(45, 46), default=45)
    parser.add_argument(
        "--batch-candidates", type=int, default=REFINER_BATCH_CANDIDATES
    )
    parser.add_argument(
        "--maximum-segments", type=int, default=REFINER_MAXIMUM_SEGMENTS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    train_mask_refiner(
        cache_paths=arguments.cache,
        output_dir=arguments.output,
        stop_after_updates=arguments.stop_after_updates,
        resume=arguments.resume,
        device=arguments.device,
        batch_candidates=arguments.batch_candidates,
        maximum_segments=arguments.maximum_segments,
        seed=arguments.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CausalRefinerRevisionTransform",
    "RefinerTrainingError",
    "align_old_probabilities_to_new_segments",
    "build_refiner_training_records",
    "match_refiner_candidates_to_gt",
    "refiner_candidate_loss",
    "refiner_lr_multiplier",
    "sample_refiner_candidates",
    "train_mask_refiner",
]
