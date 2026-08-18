"""Causal P6-A metric primitives.

This module deliberately does not depend on the P5 evaluator or on tracker
capacity.  Local observations are frozen once, endpoint predictions are
constructed from prefix state, and the diagnostic GT assignment is kept
separate from both metric paths.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

_STAGE_KEYS = ("temporal_stages", "timesteps", "stage_ids")
_MASK_KEY = "pred_masks"
_TARGET_MASK_KEY = "masks"
_OFFICIAL_IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return copy.deepcopy(value)


def _canonical(value: Any) -> Any:
    """Return a JSON-safe representation preserving tensor bytes exactly."""

    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.numpy().tobytes()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: repr(item[0]))
        return {
            "__mapping__": [
                [_canonical(key), _canonical(item)] for key, item in items
            ]
        }
    if isinstance(value, (list, tuple)):
        return {
            "__sequence__": [_canonical(item) for item in value]
        }
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "inf" if value > 0 else "-inf"}
        return {"__float__": value.hex()}
    if hasattr(value, "item"):
        return _canonical(value.item())
    return {"__repr__": repr(value)}


def _digest(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FrozenRawObservation:
    """Detached CPU observation with a content-addressed fingerprint."""

    data: Mapping[str, Any]
    fingerprint: str

    def __post_init__(self) -> None:
        cloned = _clone_cpu(dict(self.data))
        object.__setattr__(self, "data", MappingProxyType(cloned))


def freeze_raw_observation(observation: Mapping[str, Any]) -> FrozenRawObservation:
    data = _clone_cpu(observation)
    return FrozenRawObservation(data=MappingProxyType(data), fingerprint=_digest(data))


def raw_observation_fingerprint(observation: Any) -> str:
    if isinstance(observation, FrozenRawObservation):
        return observation.fingerprint
    return _digest(observation)


def observation_fingerprint(observation: Any) -> str:
    """Compatibility alias used by cache/evaluation callers."""

    return raw_observation_fingerprint(observation)


def _fingerprint_prediction_collection(value: Any) -> str:
    if isinstance(value, FrozenRawObservation):
        return value.fingerprint
    if isinstance(value, Mapping) and _MASK_KEY in value:
        return raw_observation_fingerprint(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _digest([_clone_cpu(item) for item in value])
    return raw_observation_fingerprint(value)


def assert_shared_raw_predictions(predictions_by_method: Mapping[str, Any]) -> str:
    """Require every association method to consume the exact same raw output."""

    if not predictions_by_method:
        raise ValueError("at least one method is required")
    fingerprints = {
        name: _fingerprint_prediction_collection(value)
        for name, value in predictions_by_method.items()
    }
    first = next(iter(fingerprints.values()))
    if any(value != first for value in fingerprints.values()):
        raise ValueError(f"raw prediction fingerprint mismatch: {fingerprints}")
    return first


def _stage_selector(target: Mapping[str, Any], stage: int | None) -> Tensor | None:
    key = next((name for name in _STAGE_KEYS if name in target), None)
    if key is None:
        return None
    stages = target[key]
    if not isinstance(stages, Tensor):
        stages = torch.as_tensor(stages)
    if stages.ndim != 1:
        raise ValueError(f"{key} must have shape [N]")
    if stage is None:
        stage = int(stages.max().item()) if stages.numel() else 0
    return stages == int(stage)


def _prediction_copy(prediction: Mapping[str, Any]) -> dict[str, Any]:
    required = (_MASK_KEY, "pred_classes", "pred_scores")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"prediction is missing {missing}")
    masks = prediction[_MASK_KEY]
    if not isinstance(masks, Tensor) or masks.ndim != 2:
        raise ValueError("pred_masks must have shape [N, K]")
    classes = prediction["pred_classes"]
    scores = prediction["pred_scores"]
    if not isinstance(classes, Tensor) or not isinstance(scores, Tensor):
        raise ValueError(  # noqa: TRY004 - public input validation uses ValueError.
            "pred_classes and pred_scores must be tensors"
        )
    if classes.ndim != 1 or scores.ndim != 1 or masks.shape[1] != len(classes) or len(classes) != len(scores):
        raise ValueError("prediction fields must agree on query count")
    return {
        key: _clone_cpu(value)
        for key, value in prediction.items()
        if key in required
    }


def adapt_raw_local_prediction(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any] | None = None,
    *,
    stage: int | None = None,
) -> dict[str, Any]:
    """Restrict a full-scene prediction to one newest local stage."""

    copied = _prediction_copy(prediction)
    selector = _stage_selector(target, stage) if target is not None else None
    if selector is not None:
        if len(selector) != copied[_MASK_KEY].shape[0]:
            raise ValueError("prediction points and target stages must align")
        copied[_MASK_KEY] = copied[_MASK_KEY][selector].clone()
    copied[_MASK_KEY] = copied[_MASK_KEY].bool()
    return copied


def adapt_raw_local_target(
    target: Mapping[str, Any], *, stage: int | None = None
) -> dict[str, Any]:
    """Restrict a full-scene target to the same local stage as its prediction."""

    if _TARGET_MASK_KEY not in target or "labels" not in target:
        raise ValueError("target must contain masks and labels")
    masks = target[_TARGET_MASK_KEY]
    if not isinstance(masks, Tensor) or masks.ndim != 2:
        raise ValueError("target masks must have shape [G, N]")
    selector = _stage_selector(target, stage)
    copied = {
        key: _clone_cpu(value)
        for key, value in target.items()
        if key not in _STAGE_KEYS
    }
    if selector is not None:
        if len(selector) != masks.shape[1]:
            raise ValueError("target masks and stage labels must align")
        copied[_TARGET_MASK_KEY] = masks.detach().cpu().clone()[:, selector]
    else:
        copied[_TARGET_MASK_KEY] = masks.detach().cpu().clone()
    copied[_TARGET_MASK_KEY] = copied[_TARGET_MASK_KEY].bool()
    return copied


def adapt_raw_local_pair(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    stage: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if stage is None:
        selector = _stage_selector(target, None)
        stage = None if selector is None else int(torch.as_tensor(target[next(key for key in _STAGE_KEYS if key in target)]).max().item())
    return (
        adapt_raw_local_prediction(prediction, target, stage=stage),
        adapt_raw_local_target(target, stage=stage),
    )


def _as_int_tuple(values: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must have shape [K]")
        values = values.detach().cpu().tolist()
    result = tuple(values)
    if any(isinstance(item, bool) for item in result):
        raise ValueError(f"{name} cannot contain bool values")
    return result


def _stable_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__name__, repr(value))


@dataclass(frozen=True)
class _StagePrediction:
    stage: int
    masks: Tensor
    classes: tuple[Any, ...]
    scores: tuple[float, ...]
    track_ids: tuple[Hashable, ...]
    class_probs: Tensor | None = None


class IdentityAccumulator:
    """Dynamic, CPU-only identity/state accumulator for a single sequence."""

    def __init__(self) -> None:
        self._stages: dict[int, _StagePrediction] = {}

    @property
    def stages(self) -> tuple[int, ...]:
        return tuple(sorted(self._stages))

    @property
    def identities(self) -> tuple[Hashable, ...]:
        found: set[Hashable] = set()
        for stage in self._stages.values():
            found.update(stage.track_ids)
        return tuple(sorted(found, key=_stable_key))

    def add_stage(
        self,
        observation: Mapping[str, Any] | int,
        prediction: Mapping[str, Any] | None = None,
    ) -> None:
        if prediction is None:
            if not isinstance(observation, Mapping):
                raise ValueError("add_stage requires a stage prediction")
            prediction = observation
            if "stage" not in prediction:
                raise ValueError("stage prediction must contain stage")
            stage = int(prediction["stage"])
        else:
            stage = int(observation)
        if stage in self._stages:
            raise ValueError(f"stage {stage} was already added")
        normalized = _prediction_copy(prediction)
        masks = normalized[_MASK_KEY].bool().cpu().clone()
        query_count = masks.shape[1]
        track_ids = prediction.get("track_ids", tuple(range(query_count)))
        track_ids_tuple = _as_int_tuple(track_ids, name="track_ids")
        if len(track_ids_tuple) != query_count:
            raise ValueError("track_ids must agree on query count")
        if len(set(track_ids_tuple)) != len(track_ids_tuple):
            raise ValueError("track_ids must be unique within a stage")
        classes = _as_int_tuple(normalized["pred_classes"], name="pred_classes")
        scores = tuple(float(item) for item in normalized["pred_scores"].detach().cpu().tolist())
        class_probs = prediction.get("class_probs", prediction.get("class_prob"))
        if class_probs is not None:
            if not isinstance(class_probs, Tensor) or class_probs.ndim != 2 or class_probs.shape[0] != query_count:
                raise ValueError("class_probs must have shape [K, C]")
            if not torch.isfinite(class_probs).all().item():
                raise ValueError("class_probs must be finite")
            class_probs = class_probs.detach().cpu().clone().float()
        self._stages[stage] = _StagePrediction(
            stage=stage,
            masks=masks,
            classes=classes,
            scores=scores,
            track_ids=track_ids_tuple,
            class_probs=class_probs,
        )

    def snapshot(self, endpoint: int | None = None) -> IdentityAccumulator:
        snapshot = IdentityAccumulator()
        for stage in sorted(self._stages):
            if endpoint is not None and stage > int(endpoint):
                continue
            current = self._stages[stage]
            copied = {
                "stage": current.stage,
                _MASK_KEY: current.masks.clone(),
                "pred_classes": torch.tensor(current.classes),
                "pred_scores": torch.tensor(current.scores),
                "track_ids": current.track_ids,
            }
            if current.class_probs is not None:
                copied["class_probs"] = current.class_probs.clone()
            snapshot.add_stage(copied)
        return snapshot

    def _selected_stages(self, endpoint: int | None) -> list[_StagePrediction]:
        return [
            self._stages[stage]
            for stage in sorted(self._stages)
            if endpoint is None or stage <= int(endpoint)
        ]

    def _track_state(
        self, track_id: Hashable, stages: Sequence[_StagePrediction]
    ) -> tuple[Any, float]:
        observations = []
        for stage in stages:
            for index, current_id in enumerate(stage.track_ids):
                if current_id == track_id:
                    observations.append((stage, index))
        if not observations:
            raise KeyError(track_id)
        probs = [
            stage.class_probs[index]
            for stage, index in observations
            if stage.class_probs is not None
        ]
        if probs:
            mean_prob = torch.stack(probs).mean(dim=0)
            class_value = int(torch.argmax(mean_prob).item())
        else:
            class_value = observations[-1][0].classes[observations[-1][1]]
        score = sum(stage.scores[index] for stage, index in observations) / len(observations)
        return class_value, float(score)

    def build_prediction(self, endpoint: int | None = None) -> dict[str, Any]:
        stages = self._selected_stages(endpoint)
        if not stages:
            return {
                _MASK_KEY: torch.zeros((0, 0), dtype=torch.bool),
                "pred_classes": torch.zeros(0, dtype=torch.long),
                "pred_scores": torch.zeros(0, dtype=torch.float32),
                "track_ids": torch.zeros(0, dtype=torch.long),
            }
        identities = sorted(
            {track_id for stage in stages for track_id in stage.track_ids},
            key=_stable_key,
        )
        if not identities:
            return {
                _MASK_KEY: torch.zeros(
                    (sum(stage.masks.shape[0] for stage in stages), 0),
                    dtype=torch.bool,
                ),
                "pred_classes": torch.zeros(0, dtype=torch.long),
                "pred_scores": torch.zeros(0, dtype=torch.float32),
                "track_ids": torch.zeros(0, dtype=torch.long),
            }
        lookup = {track_id: column for column, track_id in enumerate(identities)}
        masks = torch.zeros(
            (sum(stage.masks.shape[0] for stage in stages), len(identities)),
            dtype=torch.bool,
        )
        offset = 0
        for stage in stages:
            for query, track_id in enumerate(stage.track_ids):
                masks[offset : offset + stage.masks.shape[0], lookup[track_id]] = stage.masks[:, query]
            offset += stage.masks.shape[0]
        classes, scores = zip(*(self._track_state(track_id, stages) for track_id in identities))
        if all(isinstance(track_id, int) and not isinstance(track_id, bool) for track_id in identities):
            track_tensor: Any = torch.tensor(identities, dtype=torch.long)
        else:
            track_tensor = tuple(identities)
        return {
            _MASK_KEY: masks,
            "pred_classes": torch.tensor(classes),
            "pred_scores": torch.tensor(scores, dtype=torch.float32),
            "track_ids": track_tensor,
        }


def build_online_endpoint_prediction(
    accumulator: IdentityAccumulator, *, endpoint: int
) -> dict[str, Any]:
    if not isinstance(accumulator, IdentityAccumulator):
        raise TypeError("accumulator must be an IdentityAccumulator")
    return accumulator.snapshot(endpoint).build_prediction()


def build_offline_reconstructed_prediction(
    accumulator: IdentityAccumulator,
) -> dict[str, Any]:
    if not isinstance(accumulator, IdentityAccumulator):
        raise TypeError("accumulator must be an IdentityAccumulator")
    return accumulator.snapshot().build_prediction()


def _mask_iou_matrix(gt_masks: Tensor, pred_masks: Tensor) -> Tensor:
    gt = gt_masks.detach().cpu().bool()
    pred = pred_masks.detach().cpu().bool()
    if gt.ndim != 2 or pred.ndim != 2 or gt.shape[1] != pred.shape[1]:
        raise ValueError("GT and prediction masks must be [N, P] with shared P")
    intersections = (gt[:, None, :] & pred[None, :, :]).sum(dim=2).float()
    unions = (gt[:, None, :] | pred[None, :, :]).sum(dim=2).float()
    return torch.where(unions > 0, intersections / unions, torch.zeros_like(unions))


def _normalize_iou_matrix(ious: Tensor | Sequence[Sequence[float]]) -> Tensor:
    if not isinstance(ious, Tensor) and isinstance(ious, Sequence) and not ious:
        return torch.empty((0, 0), dtype=torch.float32)
    matrix = ious.detach().cpu().float() if isinstance(ious, Tensor) else torch.as_tensor(ious, dtype=torch.float32)
    if matrix.ndim != 2:
        raise ValueError("IoU matrix must have shape [G, K]")
    if not torch.isfinite(matrix).all().item():
        raise ValueError("IoU matrix must be finite")
    return matrix


def _class_mask(
    gt_classes: Tensor | Sequence[Any] | None,
    pred_classes: Tensor | Sequence[Any] | None,
    shape: tuple[int, int],
) -> Tensor:
    if gt_classes is None or pred_classes is None:
        return torch.ones(shape, dtype=torch.bool)
    gt = torch.as_tensor(gt_classes).detach().cpu()
    pred = torch.as_tensor(pred_classes).detach().cpu()
    if gt.ndim != 1 or pred.ndim != 1 or tuple(gt.shape) != (shape[0],) or tuple(pred.shape) != (shape[1],):
        raise ValueError("class vectors must align with IoU matrix")
    return gt[:, None] == pred[None, :]


def global_hungarian_match(
    ious: Tensor | Sequence[Sequence[float]],
    pred_masks: Tensor | Sequence[Sequence[float]] | None = None,
    *,
    gt_classes: Tensor | Sequence[Any] | None = None,
    pred_classes: Tensor | Sequence[Any] | None = None,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Class-compatible matching: maximize cardinality, then total IoU.

    Dummy rows/columns allow legal edges to be left unmatched.  A cardinality
    bonus larger than any possible IoU sum makes the objective lexicographic.
    """

    if pred_masks is not None:
        gt_tensor = torch.as_tensor(ious)
        pred_tensor = torch.as_tensor(pred_masks)
        if gt_tensor.numel() == 0 or pred_tensor.numel() == 0:
            matrix = torch.empty((int(gt_tensor.shape[0]) if gt_tensor.ndim == 2 else 0, int(pred_tensor.shape[1]) if pred_tensor.ndim == 2 else 0))
        elif gt_tensor.ndim == 2 and pred_tensor.ndim == 2 and gt_tensor.shape[1] == pred_tensor.shape[0]:
            matrix = _mask_iou_matrix(gt_tensor, pred_tensor.transpose(0, 1))
        elif gt_tensor.ndim == 2 and pred_tensor.ndim == 2 and gt_tensor.shape[1] == pred_tensor.shape[1]:
            matrix = _mask_iou_matrix(gt_tensor, pred_tensor)
        else:
            raise ValueError("GT and prediction masks must share point dimension")
    else:
        matrix = _normalize_iou_matrix(ious)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    gt_count, pred_count = matrix.shape
    if not gt_count or not pred_count:
        return []
    legal = (matrix >= float(threshold)) & _class_mask(
        gt_classes, pred_classes, (gt_count, pred_count)
    )
    size = gt_count + pred_count
    weights = torch.zeros((size, size), dtype=torch.float64)
    bonus = float(min(gt_count, pred_count) + 1)
    weights[:gt_count, :pred_count] = torch.where(
        legal,
        bonus + matrix.double(),
        torch.zeros_like(matrix, dtype=torch.float64),
    )
    # The solver is deterministic for a fixed matrix; a tiny row-priority term
    # resolves equal-weight edges without affecting the lexicographic objective.
    for row in range(gt_count):
        for col in range(pred_count):
            if legal[row, col]:
                weights[row, col] += 1e-12 * ((gt_count - row) * (pred_count + 1) - col)
    rows, cols = linear_sum_assignment(-weights.numpy())
    result = [
        (int(row), int(col))
        for row, col in zip(rows, cols, strict=True)
        if row < gt_count and col < pred_count and bool(legal[row, col])
    ]
    return sorted(result)


def greedy_diagnostic_match(
    ious: Tensor | Sequence[Sequence[float]],
    *,
    gt_classes: Tensor | Sequence[Any] | None = None,
    pred_classes: Tensor | Sequence[Any] | None = None,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Legacy greedy assignment retained only for regression diagnostics."""

    matrix = _normalize_iou_matrix(ious)
    legal = (matrix >= float(threshold)) & _class_mask(
        gt_classes, pred_classes, tuple(matrix.shape)
    )
    used: set[int] = set()
    result: list[tuple[int, int]] = []
    for row in range(matrix.shape[0]):
        candidates = [
            col
            for col in range(matrix.shape[1])
            if bool(legal[row, col]) and col not in used
        ]
        if not candidates:
            continue
        col = min(candidates, key=lambda candidate: (-float(matrix[row, candidate]), candidate))
        used.add(col)
        result.append((row, col))
    return result


def match_instances_hungarian(
    gt_masks: Tensor,
    pred_masks: Tensor,
    gt_classes: Tensor | Sequence[Any] | None = None,
    pred_classes: Tensor | Sequence[Any] | None = None,
    *,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    return global_hungarian_match(
        _mask_iou_matrix(gt_masks, pred_masks),
        gt_classes=gt_classes,
        pred_classes=pred_classes,
        threshold=threshold,
    )


def _average_precision(
    y_true: Sequence[bool], scores: Sequence[float], gt_count: int
) -> float | None:
    if gt_count <= 0:
        return None
    if not scores:
        return 0.0
    order = sorted(range(len(scores)), key=lambda idx: (-float(scores[idx]), idx))
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for index in order:
        if y_true[index]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / gt_count)
    envelope = list(precisions)
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    previous = 0.0
    area = 0.0
    for precision, recall in zip(envelope, recalls, strict=True):
        area += precision * max(0.0, recall - previous)
        previous = recall
    return float(area)


def _single_frame_scores(
    prediction: Mapping[str, Any], target: Mapping[str, Any], threshold: float
) -> tuple[list[bool], list[float], int, int]:
    pred_masks = prediction[_MASK_KEY].detach().cpu().bool()
    pred_classes = prediction["pred_classes"].detach().cpu()
    pred_scores = prediction["pred_scores"].detach().cpu().float()
    gt_masks = target[_TARGET_MASK_KEY].detach().cpu().bool()
    gt_classes = target["labels"].detach().cpu()
    ious = _mask_iou_matrix(gt_masks, pred_masks.transpose(0, 1))
    matched: set[int] = set()
    y_true: list[bool] = []
    scores: list[float] = []
    for pred_index in sorted(range(pred_masks.shape[1]), key=lambda idx: (-float(pred_scores[idx]), idx)):
        compatible = [
            gt_index
            for gt_index in range(gt_masks.shape[0])
            if gt_index not in matched
            and bool(gt_classes[gt_index] == pred_classes[pred_index])
            and float(ious[gt_index, pred_index]) >= threshold
        ]
        if compatible:
            gt_index = max(compatible, key=lambda idx: (float(ious[idx, pred_index]), -idx))
            matched.add(gt_index)
            y_true.append(True)
        else:
            y_true.append(False)
        scores.append(float(pred_scores[pred_index]))
    return y_true, scores, int(gt_masks.shape[0]), len(matched)


def compute_raw_local_metrics(
    predictions: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Compute AP/REC from newest-stage local predictions only."""

    if isinstance(predictions, Mapping):
        predictions = [predictions]
    if isinstance(targets, Mapping):
        targets = [targets]
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    thresholds = {"AP50": 0.5, "AP25": 0.25}
    ap_values: dict[str, list[float]] = {key: [] for key in thresholds}
    rec_values: dict[str, list[tuple[int, int]]] = {key: [] for key in thresholds}
    all_ap: list[float] = []
    all_rec: dict[float, list[tuple[int, int]]] = {
        threshold: [] for threshold in _OFFICIAL_IOU_THRESHOLDS
    }
    for prediction, target in zip(predictions, targets, strict=True):
        prediction, target = adapt_raw_local_pair(prediction, target)
        for key, threshold in thresholds.items():
            y_true, scores, gt_count, matched = _single_frame_scores(prediction, target, threshold)
            ap = _average_precision(y_true, scores, gt_count)
            if ap is not None:
                ap_values[key].append(ap)
            rec_values[key].append((matched, gt_count))
        for threshold in _OFFICIAL_IOU_THRESHOLDS:
            y_true, scores, gt_count, matched = _single_frame_scores(
                prediction, target, threshold
            )
            del y_true, scores
            all_rec[threshold].append((matched, gt_count))
        ap_thresholds: list[float] = []
        for threshold in _OFFICIAL_IOU_THRESHOLDS:
            y_true, scores, count, _ = _single_frame_scores(prediction, target, threshold)
            value = _average_precision(y_true, scores, count)
            if value is not None:
                ap_thresholds.append(value)
        if ap_thresholds:
            all_ap.append(sum(ap_thresholds) / len(ap_thresholds))
    def mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0
    def recall(values: Sequence[tuple[int, int]]) -> float:
        denominator = sum(total for _, total in values)
        return float(sum(found for found, _ in values) / denominator) if denominator else 0.0

    result = {
        "AP": mean(all_ap),
        "AP50": mean(ap_values["AP50"]),
        "AP25": mean(ap_values["AP25"]),
        "REC": mean([recall(all_rec[threshold]) for threshold in _OFFICIAL_IOU_THRESHOLDS]),
        "REC50": recall(rec_values["AP50"]),
        "REC25": recall(rec_values["AP25"]),
    }
    result.update({f"raw_local_{key}": value for key, value in result.items()})
    return result


def _target_stages(targets: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(targets, Mapping):
        key = next((name for name in _STAGE_KEYS if name in targets), None)
        if key is None:
            return [adapt_raw_local_target(targets)]
        stages = torch.as_tensor(targets[key]).detach().cpu()
        return [adapt_raw_local_target(targets, stage=int(stage)) for stage in torch.unique(stages, sorted=True).tolist()]
    result: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        key = next((name for name in _STAGE_KEYS if name in target), None)
        if key is None or torch.unique(torch.as_tensor(target[key])).numel() <= 1:
            result.append(adapt_raw_local_target(target, stage=index if key is None else None))
        else:
            for stage in torch.unique(torch.as_tensor(target[key]), sorted=True).tolist():
                result.append(adapt_raw_local_target(target, stage=int(stage)))
    return result


def _temporal_scores(
    prediction: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    threshold: float,
) -> tuple[list[bool], list[float], int, int]:
    if not targets:
        return [], [], 0, 0
    pred_masks = prediction[_MASK_KEY].detach().cpu().bool()
    pred_classes = prediction["pred_classes"].detach().cpu()
    pred_scores = prediction["pred_scores"].detach().cpu().float()
    gt_count = targets[0]["masks"].shape[0]
    gt_labels = targets[0]["labels"].detach().cpu()
    iou_by_stage = []
    offset = 0
    for target in targets:
        point_count = target["masks"].shape[1]
        stage_masks = pred_masks[offset : offset + point_count]
        iou_by_stage.append(_mask_iou_matrix(target["masks"], stage_masks.transpose(0, 1)))
        offset += point_count
    temporal_ious = torch.stack(iou_by_stage).amin(dim=0) if iou_by_stage else torch.zeros((gt_count, pred_masks.shape[1]))
    matched: set[int] = set()
    y_true: list[bool] = []
    scores: list[float] = []
    for pred_index in sorted(range(pred_masks.shape[1]), key=lambda idx: (-float(pred_scores[idx]), idx)):
        compatible = [
            gt_index
            for gt_index in range(gt_count)
            if gt_index not in matched
            and bool(gt_labels[gt_index] == pred_classes[pred_index])
            and float(temporal_ious[gt_index, pred_index]) >= threshold
        ]
        if compatible:
            gt_index = max(compatible, key=lambda idx: (float(temporal_ious[idx, pred_index]), -idx))
            matched.add(gt_index)
            y_true.append(True)
        else:
            y_true.append(False)
        scores.append(float(pred_scores[pred_index]))
    return y_true, scores, gt_count, len(matched)


def _temporal_metric_values(
    prediction: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    values: dict[str, float] = {}
    ap_by_threshold: dict[float, float] = {}
    rec_by_threshold: dict[float, float] = {}
    for threshold in (0.5, 0.25):
        y_true, scores, gt_count, matched = _temporal_scores(prediction, targets, threshold)
        ap_by_threshold[threshold] = _average_precision(y_true, scores, gt_count) or 0.0
        rec_by_threshold[threshold] = matched / gt_count if gt_count else 0.0
    ap_values = []
    rec_values = []
    for threshold in _OFFICIAL_IOU_THRESHOLDS:
        y_true, scores, gt_count, _ = _temporal_scores(prediction, targets, threshold)
        ap_values.append(_average_precision(y_true, scores, gt_count) or 0.0)
        _, _, _, matched = _temporal_scores(prediction, targets, threshold)
        rec_values.append(matched / gt_count if gt_count else 0.0)
    values["t-mAP"] = float(sum(ap_values) / len(ap_values))
    values["t-mAP50"] = ap_by_threshold[0.5]
    values["t-mAP25"] = ap_by_threshold[0.25]
    values["t-REC"] = float(sum(rec_values) / len(rec_values))
    values["t-REC50"] = rec_by_threshold[0.5]
    values["t-REC25"] = rec_by_threshold[0.25]
    return values


def compute_endpoint_metrics(
    accumulator: IdentityAccumulator,
    targets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    endpoint: int,
) -> dict[str, float]:
    """Return strict-online endpoint metrics and a separately named offline diagnostic."""

    target_stages = _target_stages(targets)
    online_targets = target_stages[: int(endpoint) + 1]
    offline_targets = target_stages
    online = _temporal_metric_values(
        build_online_endpoint_prediction(accumulator, endpoint=endpoint), online_targets
    )
    offline = _temporal_metric_values(
        build_offline_reconstructed_prediction(accumulator), offline_targets
    )
    result = {f"online_{key}": value for key, value in online.items()}
    result.update({f"offline_reconstructed_{key}": value for key, value in offline.items()})
    result.update(
        {
            key.replace("t-", "t_"): value
            for key, value in result.items()
            if "t-" in key
        }
    )
    return result


def relative_retention(numerator: float, denominator: float) -> float | None:
    """Return a diagnostic ratio, explicitly representing a zero denominator."""

    if float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def compute_retention(numerator: float, denominator: float) -> float | None:
    return relative_retention(numerator, denominator)


# Short aliases keep the pure layer convenient for protocol and analysis code.
raw_local_metrics = compute_raw_local_metrics
strict_online_metrics = compute_endpoint_metrics
hungarian_diagnostic_match = global_hungarian_match
greedy_match = greedy_diagnostic_match


__all__ = [
    "FrozenRawObservation",
    "IdentityAccumulator",
    "adapt_raw_local_pair",
    "adapt_raw_local_prediction",
    "adapt_raw_local_target",
    "assert_shared_raw_predictions",
    "build_offline_reconstructed_prediction",
    "build_online_endpoint_prediction",
    "compute_endpoint_metrics",
    "compute_raw_local_metrics",
    "compute_retention",
    "freeze_raw_observation",
    "global_hungarian_match",
    "greedy_diagnostic_match",
    "greedy_match",
    "hungarian_diagnostic_match",
    "match_instances_hungarian",
    "observation_fingerprint",
    "raw_local_metrics",
    "raw_observation_fingerprint",
    "relative_retention",
    "strict_online_metrics",
]
