"""GT-free evidence, assignment, and commit for CrossWindow association."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from models.crosswindow_state import (
    BufferedGroup,
    CommittedObservation,
    CrossWindowBuffer,
    CrossWindowState,
    CrossWindowStateError,
    GroupKey,
)
from models.persistent_memory import _optimal_assignment_with_stable_ties
from scripts.crosswindow_cache import CanonicalFrame, QueryGroup, align_mask


class CrossWindowAssociationError(ValueError):
    """Raised when evidence, assignment, or commit breaks the fixed contract."""


@dataclass(frozen=True)
class AssociationConfig:
    family: str
    tau: float
    overlap_theta: float = 0.65
    mutual_margin: float = 0.10
    lambda_: float = 0.0

    def __post_init__(self) -> None:
        if self.family not in {"A0-U", "A1", "A2"}:
            raise CrossWindowAssociationError(
                "association family must be A0-U, A1, or A2"
            )
        for name, value in (
            ("tau", self.tau),
            ("overlap_theta", self.overlap_theta),
            ("mutual_margin", self.mutual_margin),
            ("lambda_", self.lambda_),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CrossWindowAssociationError(f"{name} must be finite")
        if not 0.0 <= float(self.tau) <= 1.0:
            raise CrossWindowAssociationError("tau must be within [0,1]")
        if not 0.0 <= float(self.overlap_theta) <= 1.0:
            raise CrossWindowAssociationError("overlap_theta must be within [0,1]")
        if not 0.0 <= float(self.mutual_margin) <= 1.0:
            raise CrossWindowAssociationError("mutual_margin must be within [0,1]")
        if not 0.0 <= float(self.lambda_) <= 1.0:
            raise CrossWindowAssociationError("lambda_ must be within [0,1]")
        if self.family != "A2" and float(self.lambda_) != 0.0:
            raise CrossWindowAssociationError("only A2 may use non-zero lambda")

    @property
    def config_id(self) -> str:
        if self.family == "A0-U":
            return f"A0-U-tau-{float(self.tau):.12g}"
        if self.family == "A1":
            return f"A1-theta-{float(self.overlap_theta):.12g}"
        return f"A2-lambda-{float(self.lambda_):.12g}-tau-{float(self.tau):.12g}"


A0_DEFAULT = AssociationConfig(family="A0-U", tau=2.0 / 3.0)
A1_DEFAULT = AssociationConfig(family="A1", tau=2.0 / 3.0, overlap_theta=0.65)
A2_DEFAULT = AssociationConfig(family="A2", tau=2.0 / 3.0, lambda_=0.25)


def preregistered_association_configs() -> tuple[AssociationConfig, ...]:
    """Return exactly the fixed 3 A0-U, 3 A1, and 6 A2 configurations."""
    tau_values = (0.60, 2.0 / 3.0, 0.73)
    return (
        *(AssociationConfig(family="A0-U", tau=tau) for tau in tau_values),
        *(
            AssociationConfig(family="A1", tau=2.0 / 3.0, overlap_theta=theta)
            for theta in (0.50, 0.65, 0.80)
        ),
        *(
            AssociationConfig(family="A2", tau=tau, lambda_=lambda_)
            for lambda_ in (0.25, 0.50)
            for tau in tau_values
        ),
    )


@dataclass(frozen=True)
class EvidenceBundle:
    source_state_sha256: str
    frame_sha256: str
    reference_id: str
    episode_id: str
    absolute_stage: int
    group_keys: tuple[GroupKey, ...]
    group_valid: tuple[bool, ...]
    group_current_supported: tuple[bool, ...]
    group_confidence: tuple[float, ...]
    anchor_logical_ids: tuple[int, ...]
    anchor_generations: tuple[int, ...]
    anchor_slots: tuple[int, ...]
    anchor_sources: tuple[str, ...]
    free_slots: tuple[int, ...]
    capacity: int
    next_logical_id: int
    cosine: Tensor
    class_compatibility: Tensor
    base: Tensor
    overlap: Tensor
    has_overlap: Tensor
    zero_norm_query_count: int
    zero_norm_anchor_count: int

    @property
    def group_count(self) -> int:
        return len(self.group_keys)

    @property
    def anchor_count(self) -> int:
        return len(self.anchor_logical_ids)

    def validate(self) -> None:
        n = self.group_count
        m = self.anchor_count
        if not (
            len(self.group_valid)
            == len(self.group_current_supported)
            == len(self.group_confidence)
            == n
        ):
            raise CrossWindowAssociationError("group evidence fields differ in length")
        if not (
            len(self.anchor_generations)
            == len(self.anchor_slots)
            == len(self.anchor_sources)
            == m
        ):
            raise CrossWindowAssociationError("anchor evidence fields differ in length")
        if tuple(sorted(self.group_keys)) != self.group_keys:
            raise CrossWindowAssociationError(
                "group evidence must use stable key order"
            )
        anchor_keys = tuple(zip(self.anchor_logical_ids, self.anchor_generations))
        if tuple(sorted(anchor_keys)) != anchor_keys or len(set(anchor_keys)) != m:
            raise CrossWindowAssociationError(
                "anchor evidence must be unique and sorted"
            )
        if len({slot for slot in self.anchor_slots if slot >= 0}) != sum(
            slot >= 0 for slot in self.anchor_slots
        ):
            raise CrossWindowAssociationError("resident anchor slots must be unique")
        for name, value, dtype in (
            ("cosine", self.cosine, torch.float32),
            ("class_compatibility", self.class_compatibility, torch.float32),
            ("base", self.base, torch.float32),
            ("overlap", self.overlap, torch.float32),
            ("has_overlap", self.has_overlap, torch.bool),
        ):
            if (
                value.shape != (n, m)
                or value.dtype != dtype
                or value.device.type != "cpu"
            ):
                raise CrossWindowAssociationError(
                    f"{name} must be a CPU {dtype} matrix [groups,anchors]"
                )
        if not all(
            torch.isfinite(value).all().item()
            for value in (
                self.cosine,
                self.class_compatibility,
                self.base,
                self.overlap,
            )
        ):
            raise CrossWindowAssociationError("evidence scores must be finite")
        if torch.any((self.base < 0) | (self.base > 1)).item():
            raise CrossWindowAssociationError("base evidence must be within [0,1]")
        if torch.any((self.overlap < 0) | (self.overlap > 1)).item():
            raise CrossWindowAssociationError("overlap evidence must be within [0,1]")
        if self.capacity <= 0 or any(
            slot < 0 or slot >= self.capacity for slot in self.free_slots
        ):
            raise CrossWindowAssociationError("free slots differ from state capacity")


@dataclass(frozen=True)
class AssignmentPlan:
    schema_version: str
    method_id: str
    source_state_sha256: str
    frame_sha256: str
    reference_id: str
    episode_id: str
    absolute_stage: int
    group_keys: tuple[GroupKey, ...]
    entity_for_group: tuple[int, ...]
    generation_for_group: tuple[int, ...]
    slot_for_group: tuple[int, ...]
    anchor_for_group: tuple[int, ...]
    is_new_id: tuple[bool, ...]
    unmatched_reason: tuple[str | None, ...]
    residency_reason: tuple[str, ...]
    decision_score: tuple[float, ...]
    decision_margin: tuple[float, ...]
    source_edge: tuple[str, ...]
    content_sha256: str

    def validate(self) -> None:
        if self.schema_version != "crosswindow-assignment-plan-v1":
            raise CrossWindowAssociationError("assignment plan schema differs")
        n = len(self.group_keys)
        fields = (
            self.entity_for_group,
            self.generation_for_group,
            self.slot_for_group,
            self.anchor_for_group,
            self.is_new_id,
            self.unmatched_reason,
            self.residency_reason,
            self.decision_score,
            self.decision_margin,
            self.source_edge,
        )
        if any(len(value) != n for value in fields):
            raise CrossWindowAssociationError("assignment plan fields differ in length")
        if len(set(self.entity_for_group)) != n:
            raise CrossWindowAssociationError("one entity may serve at most one group")
        if any(value < 0 for value in self.entity_for_group):
            raise CrossWindowAssociationError("public entity IDs must be non-negative")
        if any(value < 0 for value in self.generation_for_group):
            raise CrossWindowAssociationError("generations must be non-negative")
        resident_slots = [slot for slot in self.slot_for_group if slot >= 0]
        if len(resident_slots) != len(set(resident_slots)):
            raise CrossWindowAssociationError("one slot may serve at most one group")
        if any(not math.isfinite(value) for value in self.decision_score):
            raise CrossWindowAssociationError("decision scores must be finite")
        if any(value < 0 or not math.isfinite(value) for value in self.decision_margin):
            raise CrossWindowAssociationError(
                "decision margins must be finite and non-negative"
            )
        if _assignment_sha256(self, include_digest=False) != self.content_sha256:
            raise CrossWindowAssociationError("assignment plan digest differs")


def _hash_tensor(digest: Any, value: Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def _state_sha256(state: CrossWindowState) -> str:
    state.validate()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "next_logical_id": state.next_logical_id,
                "stage_watermark": state.stage_watermark,
                "episode_id": state.episode_id,
            },
            sort_keys=True,
        ).encode()
    )
    for tensor in (
        state.features,
        state.class_prob,
        state.confidence,
        state.occupied,
        state.active,
        state.age,
        state.last_seen,
        state.logical_ids,
        state.generations,
    ):
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _group_key(frame: CanonicalFrame, group: QueryGroup) -> GroupKey:
    return (
        frame.producer_id,
        frame.episode_id,
        frame.order_id,
        frame.absolute_stage,
        group.source_query_id,
    )


def canonical_frame_sha256(frame: CanonicalFrame) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "producer_id": frame.producer_id,
                "reference_id": frame.reference_id,
                "episode_id": frame.episode_id,
                "order_id": frame.order_id,
                "absolute_stage": frame.absolute_stage,
                "source_window": frame.source_window,
            },
            sort_keys=True,
        ).encode()
    )
    for scan_id in frame.source_window:
        digest.update(scan_id.encode())
        _hash_tensor(digest, frame.canonical_vertex_ids[scan_id])
    for group in frame.groups:
        digest.update(repr(_group_key(frame, group)).encode())
        _hash_tensor(digest, group.feature)
        _hash_tensor(digest, group.class_prob)
        digest.update(
            repr(
                (
                    group.confidence,
                    group.valid,
                    group.current_supported,
                    group.previous_supported,
                    group.candidate_indices,
                )
            ).encode()
        )
    for candidate in frame.candidates:
        digest.update(repr(candidate.key).encode())
        digest.update(repr((candidate.predicted_class_id, candidate.score)).encode())
        for candidate_slice in candidate.slices:
            digest.update(candidate_slice.scan_id.encode())
            _hash_tensor(digest, candidate_slice.original_vertex_ids)
            _hash_tensor(digest, candidate_slice.canonical_vertex_ids)
            _hash_tensor(digest, candidate_slice.mask)
    return digest.hexdigest()


def _validate_probability_matrix(value: Tensor, *, name: str) -> Tensor:
    matrix = value.detach().cpu().to(dtype=torch.float32).contiguous()
    if not torch.isfinite(matrix).all().item() or torch.any(matrix < 0).item():
        raise CrossWindowAssociationError(f"{name} must be finite and non-negative")
    if torch.any(matrix.sum(dim=-1) > 1.0 + 1e-5).item():
        raise CrossWindowAssociationError(f"{name} row sums exceed one")
    return matrix


def _group_masks_for_scan(
    frame: CanonicalFrame,
    group: QueryGroup,
    *,
    scan_id: str,
    to_vertex_ids: Tensor,
) -> tuple[Tensor, ...]:
    masks = []
    for candidate_index in group.candidate_indices:
        if candidate_index < 0 or candidate_index >= len(frame.candidates):
            raise CrossWindowAssociationError("group candidate index is outside frame")
        candidate = frame.candidates[candidate_index]
        if candidate.key.source_query_id != group.source_query_id:
            raise CrossWindowAssociationError(
                "candidate belongs to another query group"
            )
        matches = [value for value in candidate.slices if value.scan_id == scan_id]
        if len(matches) != 1:
            raise CrossWindowAssociationError("candidate scan slice coverage differs")
        candidate_slice = matches[0]
        mask = align_mask(
            candidate_slice.mask,
            from_ids=candidate_slice.canonical_vertex_ids,
            to_ids=to_vertex_ids,
        ).bool()
        if mask.any().item():
            masks.append(mask)
    return tuple(masks)


def _maximum_mask_iou(left: tuple[Tensor, ...], right: tuple[Tensor, ...]) -> float:
    best = 0.0
    for left_mask in left:
        for right_mask in right:
            union = int((left_mask | right_mask).sum().item())
            if union:
                value = int((left_mask & right_mask).sum().item()) / union
                best = max(best, value)
    return best


def build_evidence(
    frame: CanonicalFrame,
    state: CrossWindowState,
    previous_buffer: CrossWindowBuffer | None,
) -> EvidenceBundle:
    """Build float32 association evidence without accepting targets or GT identities."""
    state.validate()
    expected_stage = state.stage_watermark + 1
    if frame.absolute_stage != expected_stage:
        raise CrossWindowAssociationError("frame stage is not the next causal stage")
    if state.episode_id is not None and state.episode_id != frame.episode_id:
        raise CrossWindowAssociationError("resident state belongs to another episode")
    if previous_buffer is not None:
        previous_buffer.validate()
        if (
            previous_buffer.reference_id != frame.reference_id
            or previous_buffer.episode_id != frame.episode_id
            or previous_buffer.absolute_stage != frame.absolute_stage - 1
            or previous_buffer.scan_id not in frame.source_window
        ):
            raise CrossWindowAssociationError(
                "previous buffer lineage differs from frame"
            )
        resident_generation = {
            int(state.logical_ids[slot].item()): int(state.generations[slot].item())
            for slot in state.occupied.nonzero(as_tuple=True)[0].tolist()
        }
        for buffer_group in previous_buffer.groups:
            known_generation = resident_generation.get(buffer_group.logical_id)
            if (
                known_generation is not None
                and known_generation != buffer_group.generation
            ):
                raise CrossWindowAssociationError(
                    "buffer generation differs from resident identity"
                )
            if buffer_group.logical_id >= state.next_logical_id:
                raise CrossWindowAssociationError(
                    "buffer identity was not reserved by resident state"
                )

    group_pairs = sorted(
        ((_group_key(frame, group), group) for group in frame.groups),
        key=lambda value: value[0],
    )
    group_keys = tuple(value[0] for value in group_pairs)
    groups = tuple(value[1] for value in group_pairs)
    if len(group_keys) != len(set(group_keys)):
        raise CrossWindowAssociationError("frame group keys must be unique")
    if groups:
        query_features = torch.stack(
            [group.feature.detach().cpu().float() for group in groups]
        )
        query_class = _validate_probability_matrix(
            torch.stack([group.class_prob.detach().cpu() for group in groups]),
            name="query class_prob",
        )
    else:
        query_features = torch.empty((0, state.feature_dim), dtype=torch.float32)
        query_class = torch.empty((0, state.class_count), dtype=torch.float32)
    if query_features.shape != (len(groups), state.feature_dim):
        raise CrossWindowAssociationError("query feature dimensions differ from state")
    if query_class.shape != (len(groups), state.class_count):
        raise CrossWindowAssociationError("query class dimensions differ from state")
    if not torch.isfinite(query_features).all().item():
        raise CrossWindowAssociationError("query features must be finite")

    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for slot in state.occupied.nonzero(as_tuple=True)[0].tolist():
        key = (
            int(state.logical_ids[slot].item()),
            int(state.generations[slot].item()),
        )
        anchors[key] = {
            "slot": slot,
            "source": "resident",
            "feature": state.features[slot].detach().cpu().float(),
            "class_prob": state.class_prob[slot].detach().cpu().float(),
            "buffer_group": None,
        }
    if previous_buffer is not None:
        for buffer_group in previous_buffer.groups:
            key = (buffer_group.logical_id, buffer_group.generation)
            if key in anchors:
                anchors[key]["buffer_group"] = buffer_group
            else:
                anchors[key] = {
                    "slot": -1,
                    "source": "buffer-only",
                    "feature": buffer_group.feature.detach().cpu().float(),
                    "class_prob": buffer_group.class_prob.detach().cpu().float(),
                    "buffer_group": buffer_group,
                }
    anchor_keys = tuple(sorted(anchors))
    if anchor_keys:
        anchor_features = torch.stack(
            [anchors[key]["feature"] for key in anchor_keys]  # type: ignore[list-item]
        )
        anchor_class = _validate_probability_matrix(
            torch.stack(
                [anchors[key]["class_prob"] for key in anchor_keys]  # type: ignore[list-item]
            ),
            name="anchor class_prob",
        )
    else:
        anchor_features = torch.empty((0, state.feature_dim), dtype=torch.float32)
        anchor_class = torch.empty((0, state.class_count), dtype=torch.float32)

    query_norms = torch.linalg.vector_norm(query_features, dim=-1, keepdim=True)
    anchor_norms = torch.linalg.vector_norm(anchor_features, dim=-1, keepdim=True)
    normalized_queries = torch.where(
        query_norms > 0,
        query_features / query_norms.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(query_features),
    )
    normalized_anchors = torch.where(
        anchor_norms > 0,
        anchor_features / anchor_norms.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(anchor_features),
    )
    cosine = normalized_queries @ normalized_anchors.T
    class_compatibility = query_class @ anchor_class.T
    base = ((cosine + 0.25 * class_compatibility + 1.0) / 2.25).clamp(0, 1)

    overlap = torch.zeros_like(base, dtype=torch.float32)
    has_overlap = torch.zeros_like(base, dtype=torch.bool)
    if previous_buffer is not None:
        current_masks = {
            group.source_query_id: _group_masks_for_scan(
                frame,
                group,
                scan_id=previous_buffer.scan_id,
                to_vertex_ids=previous_buffer.vertex_ids,
            )
            for group in groups
        }
        for group_index, group in enumerate(groups):
            masks = current_masks[group.source_query_id]
            if not masks:
                continue
            for anchor_index, key in enumerate(anchor_keys):
                buffer_group = anchors[key]["buffer_group"]
                if not isinstance(buffer_group, BufferedGroup):
                    continue
                old_masks = tuple(
                    mask for mask in buffer_group.masks if mask.any().item()
                )
                if not old_masks:
                    continue
                has_overlap[group_index, anchor_index] = True
                overlap[group_index, anchor_index] = _maximum_mask_iou(masks, old_masks)

    bundle = EvidenceBundle(
        source_state_sha256=_state_sha256(state),
        frame_sha256=canonical_frame_sha256(frame),
        reference_id=frame.reference_id,
        episode_id=frame.episode_id,
        absolute_stage=frame.absolute_stage,
        group_keys=group_keys,
        group_valid=tuple(group.valid for group in groups),
        group_current_supported=tuple(group.current_supported for group in groups),
        group_confidence=tuple(float(group.confidence) for group in groups),
        anchor_logical_ids=tuple(key[0] for key in anchor_keys),
        anchor_generations=tuple(key[1] for key in anchor_keys),
        anchor_slots=tuple(int(anchors[key]["slot"]) for key in anchor_keys),
        anchor_sources=tuple(str(anchors[key]["source"]) for key in anchor_keys),
        free_slots=tuple((~state.occupied).nonzero(as_tuple=True)[0].tolist()),
        capacity=state.capacity,
        next_logical_id=state.next_logical_id,
        cosine=cosine.float().cpu(),
        class_compatibility=class_compatibility.float().cpu(),
        base=base.float().cpu(),
        overlap=overlap.float().cpu(),
        has_overlap=has_overlap.cpu(),
        zero_norm_query_count=int((query_norms.squeeze(-1) == 0).sum().item()),
        zero_norm_anchor_count=int((anchor_norms.squeeze(-1) == 0).sum().item()),
    )
    bundle.validate()
    return bundle


def score_a2(bundle: EvidenceBundle, *, lambda_: float) -> Tensor:
    bundle.validate()
    if (
        isinstance(lambda_, bool)
        or not isinstance(lambda_, (int, float))
        or not math.isfinite(float(lambda_))
        or not 0.0 <= float(lambda_) <= 1.0
    ):
        raise CrossWindowAssociationError("lambda_ must be within [0,1]")
    mixed = (1.0 - float(lambda_)) * bundle.base + float(lambda_) * bundle.overlap
    return torch.where(bundle.has_overlap, mixed, bundle.base).float()


def _row_margin(score: Tensor, row: int) -> float:
    if score.shape[1] == 0:
        return 0.0
    values = score[row].sort(descending=True).values
    if score.shape[1] == 1:
        return max(0.0, float(values[0].item()))
    return max(0.0, float((values[0] - values[1]).item()))


def _private_dummy_assignment(
    score: Tensor,
    *,
    tau: float,
    row_indices: tuple[int, ...],
    column_indices: tuple[int, ...],
) -> dict[int, int]:
    if not row_indices or not column_indices:
        return {}
    real = score[list(row_indices)][:, list(column_indices)].float()
    row_count = len(row_indices)
    private = torch.full((row_count, row_count), -1_000_000.0)
    private[torch.arange(row_count), torch.arange(row_count)] = 0.0
    surplus = torch.cat((real - float(tau), private), dim=1)
    assigned_rows, assigned_columns = _optimal_assignment_with_stable_ties(surplus)
    matches = {}
    for local_row, local_column in zip(
        assigned_rows.tolist(), assigned_columns.tolist(), strict=True
    ):
        if local_column >= len(column_indices):
            continue
        row = row_indices[local_row]
        column = column_indices[local_column]
        if float(score[row, column].item()) > float(tau):
            matches[row] = column
    return matches


def _a1_locks(
    bundle: EvidenceBundle,
    config: AssociationConfig,
) -> dict[int, int]:
    candidates = []
    for row in range(bundle.group_count):
        available_columns = bundle.has_overlap[row].nonzero(as_tuple=True)[0]
        if available_columns.numel() == 0:
            continue
        row_values = bundle.overlap[row, available_columns]
        ordered = row_values.sort(descending=True).values
        row_top = float(ordered[0].item())
        row_second = float(ordered[1].item()) if ordered.numel() > 1 else 0.0
        row_margin = row_top - row_second
        if row_margin < config.mutual_margin:
            continue
        row_best = available_columns[row_values == ordered[0]]
        if row_best.numel() != 1:
            continue
        column = int(row_best.item())
        available_rows = bundle.has_overlap[:, column].nonzero(as_tuple=True)[0]
        if available_rows.numel() == 0:
            continue
        column_values = bundle.overlap[available_rows, column]
        column_ordered = column_values.sort(descending=True).values
        column_second = (
            float(column_ordered[1].item()) if column_ordered.numel() > 1 else 0.0
        )
        column_margin = float(column_ordered[0].item()) - column_second
        column_best = available_rows[column_values == column_ordered[0]]
        if (
            column_margin < config.mutual_margin
            or column_best.numel() != 1
            or int(column_best.item()) != row
            or row_top < config.overlap_theta
            or float(bundle.base[row, column].item()) <= config.tau
        ):
            continue
        candidates.append((row_top, bundle.group_keys[row], column, row))
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    matches: dict[int, int] = {}
    used_columns = set()
    for _, _, column, row in candidates:
        if row not in matches and column not in used_columns:
            matches[row] = column
            used_columns.add(column)
    return matches


def _assignment_sha256(plan: AssignmentPlan, *, include_digest: bool) -> str:
    values = {
        name: getattr(plan, name)
        for name in plan.__dataclass_fields__
        if include_digest or name != "content_sha256"
    }
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def associate(
    bundle: EvidenceBundle, method_config: AssociationConfig
) -> AssignmentPlan:
    bundle.validate()
    if not isinstance(method_config, AssociationConfig):
        raise CrossWindowAssociationError("method_config must be AssociationConfig")
    n = bundle.group_count
    m = bundle.anchor_count
    if method_config.family == "A2":
        score = score_a2(bundle, lambda_=method_config.lambda_)
    else:
        score = bundle.base

    locked: dict[int, int] = {}
    source_by_group: dict[int, str] = {}
    if method_config.family == "A1":
        locked = _a1_locks(bundle, method_config)
        source_by_group.update({row: "OVERLAP_LOCK" for row in locked})
    remaining_rows = tuple(row for row in range(n) if row not in locked)
    used_columns = set(locked.values())
    remaining_columns = tuple(
        column for column in range(m) if column not in used_columns
    )
    residual_score = bundle.base if method_config.family == "A1" else score
    matches = dict(locked)
    matches.update(
        _private_dummy_assignment(
            residual_score,
            tau=method_config.tau,
            row_indices=remaining_rows,
            column_indices=remaining_columns,
        )
    )

    entity_for_group = []
    generation_for_group = []
    anchor_for_group = []
    is_new_id = []
    unmatched_reason: list[str | None] = []
    decision_score = []
    decision_margin = []
    source_edge = []
    next_logical_id = bundle.next_logical_id
    for row in range(n):
        column = matches.get(row, -1)
        if column >= 0:
            entity_for_group.append(bundle.anchor_logical_ids[column])
            generation_for_group.append(bundle.anchor_generations[column])
            anchor_for_group.append(column)
            is_new_id.append(False)
            unmatched_reason.append(None)
            if row in locked:
                selected_score = float(bundle.overlap[row, column].item())
                margin = _row_margin(bundle.overlap, row)
            else:
                selected_score = float(residual_score[row, column].item())
                margin = _row_margin(residual_score, row)
            decision_score.append(selected_score)
            decision_margin.append(margin)
            if row in source_by_group:
                source_edge.append(source_by_group[row])
            elif method_config.family == "A1":
                source_edge.append("BASE_RESIDUAL")
            elif method_config.family == "A2" and bundle.has_overlap[row, column]:
                source_edge.append("BASE_OVERLAP")
            elif method_config.family == "A2":
                source_edge.append("BASE_MISSING_OVERLAP")
            else:
                source_edge.append("BASE")
        else:
            entity_for_group.append(next_logical_id)
            next_logical_id += 1
            generation_for_group.append(0)
            anchor_for_group.append(-1)
            is_new_id.append(True)
            qualifies = bool((residual_score[row] > method_config.tau).any().item())
            unmatched_reason.append(
                "ONE_TO_ONE_CONFLICT" if qualifies else "NO_ACCEPTED_EDGE"
            )
            decision_score.append(0.0)
            decision_margin.append(_row_margin(residual_score, row))
            source_edge.append("PRIVATE_DUMMY")

    slot_for_group = [-1] * n
    residency_reason = [""] * n
    for row, column in matches.items():
        slot = bundle.anchor_slots[column]
        if slot >= 0:
            slot_for_group[row] = slot
            residency_reason[row] = "INHERITED_RESIDENT"
    free_slots = list(bundle.free_slots)

    def allocation_key(row: int) -> tuple[float, GroupKey]:
        return -bundle.group_confidence[row], bundle.group_keys[row]

    promotions = sorted(
        (
            row
            for row, column in matches.items()
            if bundle.anchor_slots[column] < 0
            and bundle.group_valid[row]
            and bundle.group_current_supported[row]
        ),
        key=allocation_key,
    )
    births = sorted(
        (
            row
            for row in range(n)
            if row not in matches
            and bundle.group_valid[row]
            and bundle.group_current_supported[row]
        ),
        key=allocation_key,
    )
    for row in promotions + births:
        if free_slots:
            slot_for_group[row] = free_slots.pop(0)
            residency_reason[row] = (
                "PROMOTED_BUFFER" if row in matches else "BIRTH_RESIDENT"
            )
        else:
            residency_reason[row] = (
                "BUFFER_ONLY_CAPACITY_FULL" if row in matches else "BIRTH_CAPACITY_FULL"
            )
    for row in range(n):
        if residency_reason[row]:
            continue
        if row in matches:
            residency_reason[row] = "BUFFER_ONLY_NO_CURRENT_STATE"
        else:
            residency_reason[row] = "NEW_ID_NO_CURRENT_STATE"

    values = {
        "schema_version": "crosswindow-assignment-plan-v1",
        "method_id": method_config.config_id,
        "source_state_sha256": bundle.source_state_sha256,
        "frame_sha256": bundle.frame_sha256,
        "reference_id": bundle.reference_id,
        "episode_id": bundle.episode_id,
        "absolute_stage": bundle.absolute_stage,
        "group_keys": bundle.group_keys,
        "entity_for_group": tuple(entity_for_group),
        "generation_for_group": tuple(generation_for_group),
        "slot_for_group": tuple(slot_for_group),
        "anchor_for_group": tuple(anchor_for_group),
        "is_new_id": tuple(is_new_id),
        "unmatched_reason": tuple(unmatched_reason),
        "residency_reason": tuple(residency_reason),
        "decision_score": tuple(decision_score),
        "decision_margin": tuple(decision_margin),
        "source_edge": tuple(source_edge),
    }
    provisional = AssignmentPlan(content_sha256="0" * 64, **values)
    plan = AssignmentPlan(
        content_sha256=_assignment_sha256(provisional, include_digest=False), **values
    )
    plan.validate()
    return plan


def _normalize_feature(value: Tensor) -> Tensor:
    feature = value.detach().to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(feature)
    if float(norm.item()) == 0.0:
        return torch.zeros_like(feature)
    return feature / norm


def _frame_groups_by_key(frame: CanonicalFrame) -> tuple[QueryGroup, ...]:
    pairs = sorted(
        ((_group_key(frame, group), group) for group in frame.groups),
        key=lambda value: value[0],
    )
    return tuple(value[1] for value in pairs)


def _build_buffer(
    frame: CanonicalFrame,
    groups: tuple[QueryGroup, ...],
    plan: AssignmentPlan,
) -> CrossWindowBuffer:
    scan_id = frame.source_window[-1]
    vertex_ids = frame.canonical_vertex_ids[scan_id].detach().cpu().long().clone()
    buffered_groups = []
    for group_index, group in enumerate(groups):
        candidates = [frame.candidates[index] for index in group.candidate_indices]
        if not candidates:
            continue
        masks = []
        for candidate in candidates:
            slices = [value for value in candidate.slices if value.scan_id == scan_id]
            if len(slices) != 1:
                raise CrossWindowAssociationError(
                    "current candidate slice coverage differs"
                )
            candidate_slice = slices[0]
            masks.append(
                align_mask(
                    candidate_slice.mask,
                    from_ids=candidate_slice.canonical_vertex_ids,
                    to_ids=vertex_ids,
                )
                .detach()
                .cpu()
                .bool()
            )
        buffered_groups.append(
            BufferedGroup(
                group_key=plan.group_keys[group_index],
                logical_id=plan.entity_for_group[group_index],
                generation=plan.generation_for_group[group_index],
                source_query_id=group.source_query_id,
                feature=group.feature.detach().cpu().float().clone(),
                class_prob=group.class_prob.detach().cpu().float().clone(),
                confidence=float(group.confidence),
                candidate_indices=tuple(group.candidate_indices),
                source_class_ids=tuple(
                    candidate.key.source_class_id for candidate in candidates
                ),
                class_ids=tuple(
                    candidate.predicted_class_id for candidate in candidates
                ),
                scores=tuple(float(candidate.score) for candidate in candidates),
                masks=tuple(masks),
            )
        )
    buffer = CrossWindowBuffer(
        reference_id=frame.reference_id,
        episode_id=frame.episode_id,
        scan_id=scan_id,
        absolute_stage=frame.absolute_stage,
        vertex_ids=vertex_ids,
        groups=tuple(buffered_groups),
    )
    buffer.validate()
    return buffer


def commit_observation(
    frame: CanonicalFrame,
    state: CrossWindowState,
    plan: AssignmentPlan,
) -> tuple[CrossWindowState, CommittedObservation]:
    """Consume one immutable plan; no association or identity fallback occurs here."""
    state.validate()
    plan.validate()
    if plan.source_state_sha256 != _state_sha256(state):
        raise CrossWindowAssociationError("assignment plan belongs to another state")
    if plan.frame_sha256 != canonical_frame_sha256(frame):
        raise CrossWindowAssociationError("assignment plan belongs to another frame")
    if (
        frame.absolute_stage != state.stage_watermark + 1
        or plan.absolute_stage != frame.absolute_stage
        or plan.reference_id != frame.reference_id
        or plan.episode_id != frame.episode_id
    ):
        raise CrossWindowAssociationError("assignment plan causal lineage differs")
    if state.episode_id is not None and state.episode_id != frame.episode_id:
        raise CrossWindowAssociationError("cannot commit across episodes")
    groups = _frame_groups_by_key(frame)
    if tuple(_group_key(frame, group) for group in groups) != plan.group_keys:
        raise CrossWindowAssociationError("assignment group order differs from frame")
    expected_new_ids = tuple(
        range(
            state.next_logical_id,
            state.next_logical_id + sum(plan.is_new_id),
        )
    )
    actual_new_ids = tuple(
        entity
        for entity, is_new in zip(plan.entity_for_group, plan.is_new_id, strict=True)
        if is_new
    )
    if actual_new_ids != expected_new_ids:
        raise CrossWindowAssociationError(
            "new public ID reservations are not contiguous"
        )

    features = state.features.clone()
    class_prob = state.class_prob.clone()
    confidence = state.confidence.clone()
    occupied = state.occupied.clone()
    active = torch.zeros_like(state.active)
    age = state.age.clone()
    age[occupied] += 1
    last_seen = state.last_seen.clone()
    logical_ids = state.logical_ids.clone()
    generations = state.generations.clone()
    for group_index, group in enumerate(groups):
        slot = plan.slot_for_group[group_index]
        if slot < 0:
            continue
        if slot >= state.capacity:
            raise CrossWindowAssociationError("assignment slot exceeds state capacity")
        entity = plan.entity_for_group[group_index]
        generation = plan.generation_for_group[group_index]
        if occupied[slot]:
            if (
                int(logical_ids[slot].item()) != entity
                or int(generations[slot].item()) != generation
            ):
                raise CrossWindowAssociationError("inherited slot identity differs")
        else:
            if not (group.valid and group.current_supported):
                raise CrossWindowAssociationError(
                    "non-current group cannot enter a resident slot"
                )
            occupied[slot] = True
            logical_ids[slot] = entity
            generations[slot] = generation
            age[slot] = 0
        if group.valid and group.current_supported:
            features[slot] = _normalize_feature(group.feature).to(features.device)
            class_prob[slot] = group.class_prob.detach().to(
                device=class_prob.device, dtype=torch.float32
            )
            confidence[slot] = float(group.confidence)
            active[slot] = True
            last_seen[slot] = frame.absolute_stage

    next_state = CrossWindowState(
        features=features,
        class_prob=class_prob,
        confidence=confidence,
        occupied=occupied,
        active=active,
        age=age,
        last_seen=last_seen,
        logical_ids=logical_ids,
        generations=generations,
        next_logical_id=state.next_logical_id + sum(plan.is_new_id),
        stage_watermark=frame.absolute_stage,
        episode_id=frame.episode_id,
    )
    try:
        next_state.validate()
    except CrossWindowStateError as error:
        raise CrossWindowAssociationError(str(error)) from error
    buffer = _build_buffer(frame, groups, plan)
    committed = CommittedObservation(
        public_id_for_group=plan.entity_for_group,
        generation_for_group=plan.generation_for_group,
        slot_for_group=plan.slot_for_group,
        nonresident_groups=tuple(
            index for index, slot in enumerate(plan.slot_for_group) if slot < 0
        ),
        buffer=buffer,
        assignment_sha256=plan.content_sha256,
        resident_bytes=next_state.state_bytes,
        buffer_bytes=buffer.buffer_bytes,
    )
    return next_state, committed


__all__ = [
    "A0_DEFAULT",
    "A1_DEFAULT",
    "A2_DEFAULT",
    "AssignmentPlan",
    "AssociationConfig",
    "CrossWindowAssociationError",
    "EvidenceBundle",
    "associate",
    "build_evidence",
    "canonical_frame_sha256",
    "commit_observation",
    "preregistered_association_configs",
    "score_a2",
]
