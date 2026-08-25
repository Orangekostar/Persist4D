"""Causal official-candidate trajectories for System Comparison V2."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from scripts.system_comparison_v2_cache import validate_task_sidecar


class V2InferenceError(ValueError):
    """Raised when official candidates cannot be bound to tracker identities."""


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise V2InferenceError(f"{name} must be a rank-{ndim} tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise V2InferenceError(f"{name} must contain finite values")
    return result


def _values(value: object, *, name: str) -> tuple[object, ...]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise V2InferenceError(f"{name} must be one-dimensional")
        return tuple(value.detach().cpu().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise V2InferenceError(f"{name} must be one-dimensional")


@dataclass(frozen=True)
class CandidateTrajectoryKey:
    kind: str
    class_id: int
    persistent_track_id: Hashable | None = None
    stage_index: int | None = None
    source_query_id: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"persistent", "ephemeral"}:
            raise V2InferenceError("trajectory key kind is invalid")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise V2InferenceError("trajectory class_id must be an integer")
        if self.kind == "persistent":
            if self.persistent_track_id is None:
                raise V2InferenceError("persistent key requires a track ID")
            try:
                hash(self.persistent_track_id)
            except TypeError as error:
                raise V2InferenceError(
                    "persistent track ID must be hashable"
                ) from error
            if self.stage_index is not None or self.source_query_id is not None:
                raise V2InferenceError("persistent key cannot contain ephemeral fields")
        elif (
            self.persistent_track_id is not None
            or isinstance(self.stage_index, bool)
            or not isinstance(self.stage_index, int)
            or self.stage_index < 0
            or isinstance(self.source_query_id, bool)
            or not isinstance(self.source_query_id, int)
            or self.source_query_id < 0
        ):
            raise V2InferenceError("ephemeral key fields are invalid")


@dataclass(frozen=True)
class V2TrajectorySnapshot:
    prediction: dict[str, Tensor]
    keys: tuple[CandidateTrajectoryKey, ...]
    stage_count: int
    score_reducer: str


@dataclass(frozen=True)
class _Occurrence:
    stage_index: int
    mask: Tensor
    score: float


class OfficialCandidateTrajectoryAccumulator:
    """Commit latest-stage official candidates under persistent/ephemeral keys."""

    _SCORE_REDUCERS = frozenset({"mean", "latest", "max"})

    def __init__(self, *, score_reducer: str = "mean") -> None:
        if score_reducer not in self._SCORE_REDUCERS:
            raise V2InferenceError(
                f"score reducer must be one of {sorted(self._SCORE_REDUCERS)}"
            )
        self.score_reducer = score_reducer
        self._stage_point_counts: list[int] = []
        self._keys: list[CandidateTrajectoryKey] = []
        self._occurrences: dict[CandidateTrajectoryKey, list[_Occurrence]] = {}

    @property
    def stage_count(self) -> int:
        return len(self._stage_point_counts)

    def add_stage(self, sidecar: Mapping[str, object], track_step: object) -> None:
        validate_task_sidecar(sidecar)
        key = sidecar["key"]
        task = sidecar["task_prediction"]
        if not isinstance(key, Mapping) or not isinstance(task, Mapping):
            raise V2InferenceError("sidecar fields must be mappings")
        stage = key["stage_index"]
        if stage != self.stage_count:
            raise V2InferenceError("sidecar stages must be committed in order")
        if _field(track_step, "stage_id") != stage:
            raise V2InferenceError("tracker step stage differs from sidecar")

        track_ids = _values(_field(track_step, "track_ids"), name="track_ids")
        masks = _tensor(task["pred_masks"], name="candidate masks", ndim=2)
        scores = _tensor(task["pred_scores"], name="candidate scores", ndim=1)
        classes = _tensor(task["pred_classes"], name="candidate classes", ndim=1)
        query_ids = _tensor(
            task["source_query_ids"], name="candidate source query IDs", ndim=1
        )
        candidate_count = int(scores.numel())
        if masks.shape[1] != candidate_count or any(
            value.numel() != candidate_count for value in (classes, query_ids)
        ):
            raise V2InferenceError("official candidate tensors do not align")
        if query_ids.numel() and (
            int(query_ids.min().item()) < 0
            or int(query_ids.max().item()) >= len(track_ids)
        ):
            raise V2InferenceError("candidate source query is outside tracker step")

        self._stage_point_counts.append(int(masks.shape[0]))
        stage_keys: set[CandidateTrajectoryKey] = set()
        for candidate in range(candidate_count):
            query_id = int(query_ids[candidate].item())
            class_id = int(classes[candidate].item())
            track_id = track_ids[query_id]
            trajectory_key = (
                CandidateTrajectoryKey(
                    kind="persistent",
                    persistent_track_id=track_id,
                    class_id=class_id,
                )
                if track_id is not None
                else CandidateTrajectoryKey(
                    kind="ephemeral",
                    stage_index=stage,
                    source_query_id=query_id,
                    class_id=class_id,
                )
            )
            if trajectory_key in stage_keys:
                raise V2InferenceError("one stage contains a duplicate trajectory key")
            stage_keys.add(trajectory_key)
            if trajectory_key not in self._occurrences:
                self._keys.append(trajectory_key)
                self._occurrences[trajectory_key] = []
            score = float(scores[candidate].item())
            if not math.isfinite(score):  # pragma: no cover - sidecar validation.
                raise V2InferenceError("candidate score must be finite")
            self._occurrences[trajectory_key].append(
                _Occurrence(
                    stage_index=stage,
                    mask=masks[:, candidate].bool().clone(),
                    score=score,
                )
            )

    def snapshot(self) -> V2TrajectorySnapshot:
        total_points = sum(self._stage_point_counts)
        candidate_count = len(self._keys)
        output_masks = torch.zeros((total_points, candidate_count), dtype=torch.bool)
        output_scores = torch.empty(candidate_count, dtype=torch.float32)
        output_classes = torch.empty(candidate_count, dtype=torch.long)
        offsets = [0]
        for count in self._stage_point_counts:
            offsets.append(offsets[-1] + count)

        for column, key in enumerate(self._keys):
            occurrences = self._occurrences[key]
            for occurrence in occurrences:
                start = offsets[occurrence.stage_index]
                stop = offsets[occurrence.stage_index + 1]
                if occurrence.mask.numel() != stop - start:
                    raise V2InferenceError("committed mask point count differs")
                output_masks[start:stop, column] = occurrence.mask
            if self.score_reducer == "mean":
                score = sum(occurrence.score for occurrence in occurrences) / len(
                    occurrences
                )
            elif self.score_reducer == "latest":
                score = occurrences[-1].score
            else:
                score = max(occurrence.score for occurrence in occurrences)
            output_scores[column] = score
            output_classes[column] = key.class_id
        return V2TrajectorySnapshot(
            prediction={
                "pred_masks": output_masks,
                "pred_scores": output_scores,
                "pred_classes": output_classes,
            },
            keys=tuple(self._keys),
            stage_count=self.stage_count,
            score_reducer=self.score_reducer,
        )


__all__ = [
    "CandidateTrajectoryKey",
    "OfficialCandidateTrajectoryAccumulator",
    "V2InferenceError",
    "V2TrajectorySnapshot",
]
