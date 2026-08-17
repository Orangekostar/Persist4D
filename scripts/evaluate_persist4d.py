#!/usr/bin/env python3
"""Evaluate fixed-capacity Persist4D identities over causal scan windows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_NAME = "persist4d_p5_single_memory"
SCHEMA_VERSION = 2
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / (
    "rescene4d_concerto_t2_repro.ckpt"
)
FORMAL_CHECKPOINT_REFERENCE = (
    "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt"
)
DEFAULT_HORIZONS = (2, 3, 4, 5)
OFFICIAL_FILTERED_SEQUENCE_COUNTS = {2: 154, 3: 120, 4: 75, 5: 43}
LOCAL_WINDOW = 2
RUNTIME_RECIPROCAL_RTOL = 1e-4

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "method",
        "source_commit",
        "source_tree_contract",
        "checkpoint",
        "settings",
        "legacy_parity",
        "horizons",
        "bounded_state",
        "conclusion",
        "errors",
    }
)
_HORIZON_KEYS = frozenset(
    {
        "T",
        "loaded_sequences",
        "persistent",
        "internal_baseline",
        "delta",
        "resources",
    }
)
_METRIC_KEYS = frozenset(
    {
        "t_mAP",
        "t_REC",
        "per_stage_AP",
        "matched_identity_observations",
        "identity_switches",
        "reactivation_events",
        "correct_reactivations",
        "reactivation_accuracy",
    }
)
_PERSISTENT_METRIC_KEYS = frozenset(
    {
        *_METRIC_KEYS,
        "rejected_births",
    }
)
_RESOURCE_KEYS = frozenset(
    {
        "peak_allocated_cuda_bytes",
        "mean_latency_ms",
        "throughput_sequences_per_second",
        "serialized_state_bytes",
    }
)
_SOURCE_TREE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_commit",
        "tracked_tree_clean",
        "index_clean",
        "allowed_untracked_outputs",
        "only_declared_outputs_untracked",
        "generation_head_unchanged",
    }
)
_LEGACY_PARITY_KEYS = frozenset(
    {
        "verified_by",
        "sample_count",
        "legacy_predictions_unchanged",
        "query_feature_shape",
    }
)
_CONCLUSION_KEYS = frozenset(
    {"label", "reason", "identity_improvements"}
)
_PERSISTENT_OUTPUT_KEYS = frozenset(
    {
        "persistent_slot_ids",
        "persistent_association_scores",
        "persistent_rejected_births",
    }
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(  # noqa: TRY004 - public validators use ValueError.
            f"{name} must be a positive integer"
        )
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def local_identity_ids(valid: Tensor, *, capacity: object) -> Tensor:
    """Map valid observations to stage-local query identities."""

    validated_capacity = _positive_integer(capacity, name="capacity")
    if not isinstance(valid, Tensor):
        raise ValueError("valid must be a tensor")  # noqa: TRY004
    if valid.ndim != 1:
        raise ValueError("valid must have shape [Q]")
    if valid.dtype != torch.bool:
        raise ValueError("valid must use bool dtype")
    if valid.shape[0] != validated_capacity:
        raise ValueError("valid query count must equal baseline capacity")
    query_ids = torch.arange(
        validated_capacity,
        dtype=torch.long,
        device=valid.device,
    )
    return torch.where(valid, query_ids, -torch.ones_like(query_ids))


@dataclass(frozen=True)
class SequenceAccumulator:
    """CPU-only persistent-slot prediction bookkeeping for one sequence."""

    capacity: int
    stage_masks: list[dict[int, Tensor]]
    class_prob_sum: Tensor
    class_prob_count: Tensor

    def __post_init__(self) -> None:
        capacity = _positive_integer(self.capacity, name="capacity")
        if capacity != self.capacity:
            object.__setattr__(self, "capacity", capacity)
        if not isinstance(self.stage_masks, list):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "stage_masks must be a list"
            )
        if not isinstance(self.class_prob_sum, Tensor):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "class_prob_sum must be a tensor"
            )
        if not isinstance(self.class_prob_count, Tensor):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "class_prob_count must be a tensor"
            )
        if (
            self.class_prob_sum.ndim != 2
            or self.class_prob_sum.shape[0] != capacity
            or self.class_prob_sum.shape[1] <= 0
        ):
            raise ValueError("class_prob_sum must have shape [capacity, C]")
        if self.class_prob_count.shape != (capacity,):
            raise ValueError("class_prob_count must have shape [capacity]")
        if self.class_prob_sum.device.type != "cpu":
            raise ValueError("class_prob_sum must remain on CPU")
        if self.class_prob_count.device.type != "cpu":
            raise ValueError("class_prob_count must remain on CPU")
        if not self.class_prob_sum.is_floating_point():
            raise ValueError("class_prob_sum must use a floating dtype")
        try:
            torch.iinfo(self.class_prob_count.dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError("class_prob_count must use an integer dtype") from error
        if not torch.isfinite(self.class_prob_sum).all().item():
            raise ValueError("class_prob_sum must contain only finite values")
        if torch.any(self.class_prob_count < 0).item():
            raise ValueError("class_prob_count must be non-negative")

    @classmethod
    def empty(cls, capacity: object, class_count: object) -> SequenceAccumulator:
        validated_capacity = _positive_integer(capacity, name="capacity")
        validated_class_count = _positive_integer(class_count, name="class_count")
        return cls(
            capacity=validated_capacity,
            stage_masks=[],
            class_prob_sum=torch.zeros(
                validated_capacity,
                validated_class_count,
                dtype=torch.float32,
            ),
            class_prob_count=torch.zeros(
                validated_capacity,
                dtype=torch.long,
            ),
        )

    @property
    def count(self) -> Tensor:
        """Compatibility alias for the per-slot observation count."""

        return self.class_prob_count

    def class_prob_mean(self) -> Tensor:
        result = torch.zeros_like(self.class_prob_sum)
        observed = self.class_prob_count > 0
        if torch.any(observed).item():
            result[observed] = self.class_prob_sum[observed] / (
                self.class_prob_count[observed, None].to(
                    dtype=self.class_prob_sum.dtype
                )
            )
        return result

    def add_stage(
        self,
        masks: Tensor | list[Tensor],
        class_prob: Tensor,
        slot_ids: Tensor,
    ) -> None:
        normalized_masks, query_count, mask_device = _validate_stage_masks(masks)
        if not isinstance(class_prob, Tensor):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "class_prob must be a tensor"
            )
        if class_prob.ndim != 2:
            raise ValueError("class_prob must have shape [Q, C]")
        if not class_prob.is_floating_point():
            raise ValueError("class_prob must use a floating dtype")
        if class_prob.shape[0] != query_count:
            raise ValueError("masks, class_prob, and slot_ids must share Q")
        if class_prob.shape[1] != self.class_prob_sum.shape[1]:
            raise ValueError("class_prob class dimension does not match accumulator")
        if class_prob.device != mask_device:
            raise ValueError("all stage inputs must use the same device")
        if not torch.isfinite(class_prob).all().item():
            raise ValueError("class_prob must contain only finite values")

        if not isinstance(slot_ids, Tensor):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "slot_ids must be a tensor"
            )
        if slot_ids.ndim != 1:
            raise ValueError("slot_ids must have shape [Q]")
        if slot_ids.shape[0] != query_count:
            raise ValueError("masks, class_prob, and slot_ids must share Q")
        try:
            torch.iinfo(slot_ids.dtype)
        except (TypeError, RuntimeError) as error:
            raise ValueError("slot_ids must use an integer dtype") from error
        if slot_ids.device != mask_device:
            raise ValueError("all stage inputs must use the same device")
        if torch.any(slot_ids < -1).item() or torch.any(
            slot_ids >= self.capacity
        ).item():
            raise ValueError("slot_ids values must be in range [-1, capacity)")

        valid_slots = slot_ids[slot_ids >= 0]
        if valid_slots.unique().numel() != valid_slots.numel():
            raise ValueError("a stage cannot assign one slot more than once")

        cpu_slots = slot_ids.detach().cpu().tolist()
        cpu_prob = class_prob.detach().to(
            device="cpu",
            dtype=self.class_prob_sum.dtype,
        )
        stage: dict[int, Tensor] = {}
        for query_index, slot in enumerate(cpu_slots):
            if slot < 0:
                continue
            stage[slot] = (
                normalized_masks[query_index].detach().bool().cpu().clone()
            )

        for query_index, slot in enumerate(cpu_slots):
            if slot < 0:
                continue
            self.class_prob_sum[slot].add_(cpu_prob[query_index])
            self.class_prob_count[slot].add_(1)
        self.stage_masks.append(stage)


def _accumulate_shared_stage(
    *,
    persistent: SequenceAccumulator,
    baseline: SequenceAccumulator,
    masks: Tensor | list[Tensor],
    class_prob: Tensor,
    persistent_slot_ids: Tensor,
    valid_observations: Tensor,
) -> Tensor:
    if persistent.capacity != baseline.capacity:
        raise ValueError("persistent and baseline capacities must match")
    if persistent.class_prob_sum.shape != baseline.class_prob_sum.shape:
        raise ValueError("persistent and baseline class dimensions must match")
    baseline_ids = local_identity_ids(
        valid_observations,
        capacity=baseline.capacity,
    )
    persistent.add_stage(masks, class_prob, persistent_slot_ids)
    baseline.add_stage(masks, class_prob, baseline_ids)
    return baseline_ids


def _validate_stage_masks(
    masks: Tensor | list[Tensor],
) -> tuple[list[Tensor], int, torch.device]:
    if isinstance(masks, Tensor):
        if masks.ndim != 2:
            raise ValueError("masks must have shape [Q, S]")
        if masks.shape[1] <= 0:
            raise ValueError("masks must have a positive stage-mask dimension")
        normalized = list(masks.unbind(dim=0))
        query_count = masks.shape[0]
        device = masks.device
    elif isinstance(masks, list):
        normalized = masks.copy()
        query_count = len(normalized)
        if not normalized:
            raise ValueError("masks list must contain at least one query mask")
        if not all(isinstance(mask, Tensor) for mask in normalized):
            raise ValueError("masks list entries must be tensors")
        if any(mask.ndim != 1 or mask.numel() == 0 for mask in normalized):
            raise ValueError("masks list entries must have shape [S]")
        stage_sizes = {mask.shape[0] for mask in normalized}
        if len(stage_sizes) != 1:
            raise ValueError("masks list entries must have the same shape")
        device = normalized[0].device
    else:
        raise ValueError(  # noqa: TRY004 - validation contract.
            "masks must be a [Q, S] tensor or list of Q tensors"
        )

    for mask in normalized:
        if mask.device != device:
            raise ValueError("all stage inputs must use the same device")
        if mask.is_complex():
            raise ValueError("masks must not use a complex dtype")
        if not torch.isfinite(mask).all().item():
            raise ValueError("masks must contain only finite values")
    return normalized, query_count, device


def identity_diagnostics(
    gt_ids_by_stage: Iterable[Iterable[object]],
    predicted_ids_by_stage: Iterable[Iterable[object]],
) -> dict[str, int | float | None]:
    try:
        gt_stages = tuple(gt_ids_by_stage)
        predicted_stages = tuple(predicted_ids_by_stage)
    except TypeError as error:
        raise ValueError("identity stages must be iterable") from error
    if len(gt_stages) != len(predicted_stages):
        raise ValueError("GT and predicted identity stage counts must match")

    previous: dict[int, tuple[int, int]] = {}
    switches = 0
    reactivations = 0
    correct_reactivations = 0
    observations = 0
    for stage_index, (gt_stage, predicted_stage) in enumerate(
        zip(gt_stages, predicted_stages, strict=True)
    ):
        try:
            gt_ids = tuple(gt_stage)
            predicted_ids = tuple(predicted_stage)
        except TypeError as error:
            raise ValueError("identity stages must contain iterable ID lists") from error
        if len(gt_ids) != len(predicted_ids):
            raise ValueError("GT and predicted identity lists must align")

        normalized_gt = [_identity_id(value, name="GT") for value in gt_ids]
        normalized_predicted = [
            _identity_id(value, name="predicted") for value in predicted_ids
        ]
        if len(set(normalized_gt)) != len(normalized_gt):
            raise ValueError("a GT identity cannot be duplicate within one stage")
        if len(set(normalized_predicted)) != len(normalized_predicted):
            raise ValueError(
                "a predicted identity cannot be duplicate within one stage"
            )

        for gt_id, predicted_id in zip(
            normalized_gt,
            normalized_predicted,
            strict=True,
        ):
            observations += 1
            if gt_id in previous:
                prior_id, prior_stage = previous[gt_id]
                switches += int(prior_id != predicted_id)
                if stage_index - prior_stage > 1:
                    reactivations += 1
                    correct_reactivations += int(prior_id == predicted_id)
            previous[gt_id] = (predicted_id, stage_index)

    return {
        "matched_identity_observations": observations,
        "identity_switches": switches,
        "reactivation_events": reactivations,
        "correct_reactivations": correct_reactivations,
        "reactivation_accuracy": (
            correct_reactivations / reactivations if reactivations else None
        ),
    }


def _identity_id(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} identity IDs must be non-negative integral nonbool"
        )
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} identity IDs must be non-negative")
    return normalized


@dataclass(frozen=True)
class _LegacyTensorSnapshot:
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
    content: Tensor


@dataclass(frozen=True)
class _LegacyArraySnapshot:
    dtype: Any
    shape: tuple[int, ...]
    content: Any


def _legacy_value_snapshot(value: object, *, path: str) -> object:
    if isinstance(value, (_LegacyTensorSnapshot, _LegacyArraySnapshot)):
        return value
    if isinstance(value, Tensor):
        return _LegacyTensorSnapshot(
            device=value.device,
            dtype=value.dtype,
            shape=tuple(value.shape),
            content=value.detach().cpu().clone(),
        )

    import numpy as np

    if isinstance(value, np.ndarray):
        return _LegacyArraySnapshot(
            dtype=value.dtype,
            shape=tuple(value.shape),
            content=value.copy(),
        )
    if isinstance(value, Mapping):
        snapshot: dict[object, object] = {}
        for key, item in value.items():
            if key is not None and type(key) not in {str, int, float, bool}:
                raise ValueError(
                    f"unsupported legacy parity value at {path}.<key>"
                )
            snapshot[key] = _legacy_value_snapshot(
                item,
                path=f"{path}.{key}",
            )
        return snapshot
    if type(value) is list:
        return [
            _legacy_value_snapshot(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is tuple:
        return tuple(
            _legacy_value_snapshot(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError(
        f"unsupported legacy parity value at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _require_legacy_value_equal(
    actual: object,
    expected: object,
    *,
    path: str,
) -> None:
    if type(actual) is not type(expected):
        raise RuntimeError(f"legacy prediction changed at {path}")
    if isinstance(expected, _LegacyTensorSnapshot):
        if (
            actual.device != expected.device
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.equal(actual.content, expected.content)
        ):
            raise RuntimeError(f"legacy prediction changed at {path}")
        return
    if isinstance(expected, _LegacyArraySnapshot):
        import numpy as np

        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not np.array_equal(actual.content, expected.content)
        ):
            raise RuntimeError(f"legacy prediction changed at {path}")
        return
    if isinstance(expected, Mapping):
        if tuple(actual) != tuple(expected):
            raise RuntimeError(f"legacy prediction changed at {path}")
        for key in expected:
            _require_legacy_value_equal(
                actual[key],
                expected[key],
                path=f"{path}.{key}",
            )
        return
    if type(expected) in {list, tuple}:
        if len(actual) != len(expected):
            raise RuntimeError(f"legacy prediction changed at {path}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _require_legacy_value_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if expected is None or type(expected) in {str, int, float, bool}:
        if actual == expected:
            return
        raise RuntimeError(f"legacy prediction changed at {path}")
    raise ValueError(f"unsupported legacy parity value at {path}")


def _legacy_parity_result(
    disabled_output: Mapping[str, object],
    enabled_output: Mapping[str, object],
    *,
    capacity: object,
) -> dict[str, object]:
    validated_capacity = _positive_integer(capacity, name="capacity")
    if not isinstance(disabled_output, Mapping) or not isinstance(
        enabled_output, Mapping
    ):
        raise RuntimeError(  # noqa: TRY004 - a parity gate execution failure.
            "legacy parity outputs must be mappings"
        )
    if _PERSISTENT_OUTPUT_KEYS.intersection(disabled_output):
        raise RuntimeError("disabled legacy output contains persistent fields")
    if not _PERSISTENT_OUTPUT_KEYS.issubset(enabled_output):
        raise RuntimeError("enabled output is missing persistent fields")
    enabled_base = {
        key: value
        for key, value in enabled_output.items()
        if key not in _PERSISTENT_OUTPUT_KEYS
    }
    if set(enabled_base) != {*disabled_output, "query_features"}:
        raise RuntimeError("legacy prediction keys changed")
    disabled_snapshot = _legacy_value_snapshot(
        disabled_output,
        path="disabled_output",
    )
    enabled_snapshot = _legacy_value_snapshot(
        enabled_base,
        path="enabled_output",
    )
    if not isinstance(disabled_snapshot, Mapping) or not isinstance(
        enabled_snapshot,
        Mapping,
    ):
        raise TypeError("legacy parity snapshots must be mappings")
    for key, expected in disabled_snapshot.items():
        _require_legacy_value_equal(
            enabled_snapshot[key],
            expected,
            path=key,
        )
    query_features = enabled_snapshot["query_features"]
    if (
        not isinstance(query_features, _LegacyTensorSnapshot)
        or len(query_features.shape) != 3
        or query_features.shape[0] != 1
        or query_features.shape[1] != validated_capacity
        or query_features.shape[2] <= 0
        or not query_features.content.is_floating_point()
        or not torch.isfinite(query_features.content).all().item()
    ):
        raise RuntimeError("query feature export failed the parity contract")
    return {
        "verified_by": "in_evaluator_fixed_t2_sample_toggle",
        "sample_count": 1,
        "legacy_predictions_unchanged": True,
        "query_feature_shape": list(query_features.shape),
    }


def _comparison_delta(
    persistent: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    persistent_stages = persistent["per_stage_AP"]
    baseline_stages = baseline["per_stage_AP"]
    return {
        "t_mAP": float(persistent["t_mAP"]) - float(baseline["t_mAP"]),
        "t_REC": float(persistent["t_REC"]) - float(baseline["t_REC"]),
        "per_stage_AP": {
            stage: float(persistent_stages[stage])
            - float(baseline_stages[stage])
            for stage in persistent_stages
        },
        "matched_identity_observations": int(
            persistent["matched_identity_observations"]
        )
        - int(baseline["matched_identity_observations"]),
        "identity_switches": int(persistent["identity_switches"])
        - int(baseline["identity_switches"]),
        "reactivation_events": int(persistent["reactivation_events"])
        - int(baseline["reactivation_events"]),
        "correct_reactivations": int(persistent["correct_reactivations"])
        - int(baseline["correct_reactivations"]),
        "reactivation_accuracy": (
            float(persistent["reactivation_accuracy"])
            - float(baseline["reactivation_accuracy"])
            if persistent["reactivation_accuracy"] is not None
            and baseline["reactivation_accuracy"] is not None
            else None
        ),
    }


def _derive_conclusion(
    horizons: Sequence[Mapping[str, object]],
    bounded_state: Mapping[str, object],
) -> dict[str, object]:
    improvements: list[str] = []
    if bounded_state.get("constant_shape") is True:
        for horizon in horizons:
            stage_count = horizon.get("T")
            if stage_count not in {4, 5}:
                continue
            persistent = horizon["persistent"]
            baseline = horizon["internal_baseline"]
            if float(persistent["t_REC"]) > float(baseline["t_REC"]):
                improvements.append(f"T{stage_count}:t_REC")
            same_observations = (
                persistent["matched_identity_observations"]
                == baseline["matched_identity_observations"]
            )
            if (
                same_observations
                and int(persistent["identity_switches"])
                < int(baseline["identity_switches"])
            ):
                improvements.append(f"T{stage_count}:identity_switches")
            persistent_accuracy = persistent["reactivation_accuracy"]
            baseline_accuracy = baseline["reactivation_accuracy"]
            if (
                same_observations
                and persistent["reactivation_events"]
                == baseline["reactivation_events"]
                and int(persistent["reactivation_events"]) > 0
                and persistent_accuracy is not None
                and baseline_accuracy is not None
                and float(persistent_accuracy) > float(baseline_accuracy)
            ):
                improvements.append(f"T{stage_count}:reactivation_accuracy")
    if improvements:
        return {
            "label": "P5_MVP_PASS",
            "reason": "bounded_execution_with_t4_t5_identity_improvement",
            "identity_improvements": improvements,
        }
    reason = (
        "bounded_execution_without_t4_t5_identity_improvement"
        if bounded_state.get("constant_shape") is True
        else "bounded_execution_not_proven"
    )
    return {
        "label": "P5_ASSOCIATION_DIAGNOSIS",
        "reason": reason,
        "identity_improvements": [],
    }


class _CliUsageError(ValueError):
    pass


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise _CliUsageError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capacity", type=int, default=100)
    return parser


def _output_hint(argv: Sequence[str]) -> Path | None:
    candidate: str | None = None
    for index, argument in enumerate(argv):
        if argument.startswith("--output="):
            candidate = argument.partition("=")[2]
        elif (
            argument == "--output"
            and index + 1 < len(argv)
            and not argv[index + 1].startswith("-")
        ):
            candidate = argv[index + 1]
    return Path(candidate) if candidate else None


def _failure_artifact(error: BaseException) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "method": METHOD_NAME,
        "conclusion": {
            "label": "P5_STREAMING_BLOCKED",
            "reason": "evaluation_failed",
            "identity_improvements": [],
        },
        "errors": [
            {
                "type": type(error).__name__,
                "code": _failure_code(error),
            }
        ],
    }


def _failure_code(error: BaseException) -> str:
    if isinstance(error, FileNotFoundError):
        return "missing_file"
    if isinstance(error, FileExistsError):
        return "output_exists"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, _CliUsageError):
        return "invalid_arguments"
    if isinstance(error, ValueError):
        return "invalid_input"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    if isinstance(error, OSError):
        return "io_error"
    return "evaluation_error"


def _json_text(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _publish_text_files_new(files_to_publish: Sequence[tuple[Path, str]]) -> None:
    if not files_to_publish:
        raise ValueError("at least one output file is required")
    resolved = [path.resolve() for path, _ in files_to_publish]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output paths must be distinct")

    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        for path, _ in files_to_publish:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing output: {path}")
        for path, content in files_to_publish:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_paths.append(Path(handle.name))
        for (path, _), temporary_path in zip(
            files_to_publish,
            temporary_paths,
            strict=True,
        ):
            os.link(temporary_path, path)
            published_paths.append(path)
    except BaseException:
        for path in reversed(published_paths):
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _write_failure_if_possible(output: Path, error: BaseException) -> None:
    if output.exists():
        return
    _publish_text_files_new([(output, _json_text(_failure_artifact(error)))])


def _validate_options(args: argparse.Namespace) -> None:
    if tuple(args.horizons) != DEFAULT_HORIZONS:
        raise ValueError("horizons must be exactly 2 3 4 5 for a complete run")
    if _positive_integer(args.capacity, name="capacity") != 100:
        raise ValueError("capacity must equal the fixed Q=100 query count")
    if args.markdown is not None and args.markdown.resolve() == args.output.resolve():
        raise ValueError("JSON and Markdown output paths must be distinct")
    if args.markdown is not None and args.markdown.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {args.markdown}"
        )
    args.checkpoint = _resolve_checkpoint(args.checkpoint)


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            f"{name} must be a mapping"
        )
    if set(value) != expected:
        raise ValueError(f"{name} has an invalid schema")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            f"{name} must be an integer"
        )
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def _require_signed_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    return int(value)


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            f"{name} must be numeric"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _require_unit_interval(value: object, *, name: str) -> float:
    normalized = _require_finite_number(value, name=name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return normalized


def _require_formal_checkpoint_reference(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            "checkpoint ref must be a string"
        )
    scheme, separator, payload = value.partition(":")
    segments = payload.split("/")
    if (
        separator != ":"
        or scheme != "repo"
        or not payload
        or payload.startswith("/")
        or "\\" in payload
        or any(segment in {"", ".", ".."} for segment in segments)
        or segments[0].casefold() in {"home", "mnt"}
        or PurePosixPath(payload).is_absolute()
        or PurePosixPath(payload).as_posix() != payload
    ):
        raise ValueError("checkpoint ref must be a canonical repository path")
    if value != FORMAL_CHECKPOINT_REFERENCE:
        raise ValueError("checkpoint ref is not the formal evaluation checkpoint")
    return value


def _validate_source_tree_contract(
    value: object,
    *,
    source_commit: str,
    args: argparse.Namespace,
) -> None:
    contract = _require_exact_keys(
        value,
        _SOURCE_TREE_KEYS,
        name="source_tree_contract",
    )
    if (
        _require_integer(
            contract["schema_version"],
            name="source_tree_contract.schema_version",
            minimum=1,
        )
        != 1
        or contract["status"] != "pass"
    ):
        raise ValueError("source_tree_contract must be passing schema 1")
    if contract["source_commit"] != source_commit:
        raise ValueError("source_tree_contract source_commit mismatch")
    for field in (
        "tracked_tree_clean",
        "index_clean",
        "only_declared_outputs_untracked",
        "generation_head_unchanged",
    ):
        if contract[field] is not True:
            raise ValueError(f"source_tree_contract.{field} must be true")
    allowed = contract["allowed_untracked_outputs"]
    if not isinstance(allowed, list) or any(
        not isinstance(path, str) for path in allowed
    ):
        raise ValueError(
            "source_tree_contract.allowed_untracked_outputs must be a list"
        )
    output_paths = [args.output]
    if args.markdown is not None:
        output_paths.append(args.markdown)
    expected = [
        f"repo:{path}"
        for path in _repository_output_paths(PROJECT_ROOT, output_paths)
    ]
    if allowed != expected:
        raise ValueError(
            "source_tree_contract allowed outputs do not match the CLI"
        )


def _validate_legacy_parity(value: object, *, capacity: int) -> None:
    parity = _require_exact_keys(
        value,
        _LEGACY_PARITY_KEYS,
        name="legacy_parity",
    )
    if parity["verified_by"] != "in_evaluator_fixed_t2_sample_toggle":
        raise ValueError("legacy_parity verification method is invalid")
    if (
        _require_integer(
            parity["sample_count"],
            name="legacy_parity.sample_count",
            minimum=1,
        )
        != 1
    ):
        raise ValueError("legacy_parity must verify exactly one sample")
    if parity["legacy_predictions_unchanged"] is not True:
        raise ValueError("legacy_parity must preserve every legacy prediction")
    shape = parity["query_feature_shape"]
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("legacy_parity query_feature_shape is invalid")
    dimensions = [
        _require_integer(item, name="query_feature_shape", minimum=1)
        for item in shape
    ]
    if dimensions[0] != 1 or dimensions[1] != capacity:
        raise ValueError("legacy_parity query_feature_shape is invalid")


def _maximum_identity_transitions(observations: int, horizon: int) -> int:
    if observations == 0:
        return 0
    minimum_tracks = (observations + horizon - 1) // horizon
    return observations - minimum_tracks


def _maximum_reactivation_events(observations: int, horizon: int) -> int:
    maximum_per_track = (horizon - 1) // 2
    if observations == 0 or maximum_per_track == 0:
        return 0
    minimum_tracks = (
        observations + maximum_per_track
    ) // (maximum_per_track + 1)
    return observations - minimum_tracks


def _validate_metric_block(
    value: object,
    *,
    horizon: int,
    loaded_sequences: int,
    capacity: int,
    name: str,
    persistent: bool,
) -> Mapping[str, object]:
    expected_keys = _PERSISTENT_METRIC_KEYS if persistent else _METRIC_KEYS
    metrics = _require_exact_keys(value, expected_keys, name=name)
    _require_unit_interval(metrics["t_mAP"], name=f"{name}.t_mAP")
    _require_unit_interval(metrics["t_REC"], name=f"{name}.t_REC")
    per_stage = metrics["per_stage_AP"]
    expected_stages = {str(stage) for stage in range(1, horizon + 1)}
    if not isinstance(per_stage, Mapping) or set(per_stage) != expected_stages:
        raise ValueError(f"{name}.per_stage_AP must contain every stage")
    for stage, stage_value in per_stage.items():
        _require_unit_interval(
            stage_value,
            name=f"{name}.per_stage_AP.{stage}",
        )
    observations = _require_integer(
        metrics["matched_identity_observations"],
        name=f"{name}.matched_identity_observations",
    )
    switches = _require_integer(
        metrics["identity_switches"],
        name=f"{name}.identity_switches",
    )
    reactivations = _require_integer(
        metrics["reactivation_events"],
        name=f"{name}.reactivation_events",
    )
    correct_reactivations = _require_integer(
        metrics["correct_reactivations"],
        name=f"{name}.correct_reactivations",
    )
    maximum_query_events = loaded_sequences * horizon * capacity
    if observations > maximum_query_events:
        raise ValueError(f"{name}.matched_identity_observations exceed query bound")
    if persistent:
        rejected_births = _require_integer(
            metrics["rejected_births"],
            name=f"{name}.rejected_births",
        )
        if rejected_births > maximum_query_events:
            raise ValueError(f"{name}.rejected_births exceed query bound")
    maximum_transitions = _maximum_identity_transitions(observations, horizon)
    maximum_reactivations = _maximum_reactivation_events(observations, horizon)
    if switches > maximum_transitions:
        raise ValueError(f"{name}.identity_switches are impossible")
    if reactivations > maximum_reactivations:
        raise ValueError(f"{name}.reactivation_events are impossible")
    if correct_reactivations > reactivations:
        raise ValueError(f"{name}.correct_reactivations are impossible")
    if reactivations - correct_reactivations > switches:
        raise ValueError(f"{name}.incorrect reactivations exceed switches")
    if switches + correct_reactivations > maximum_transitions:
        raise ValueError(f"{name}.identity transition counts are impossible")
    accuracy = metrics["reactivation_accuracy"]
    if reactivations == 0:
        if accuracy is not None:
            raise ValueError(
                f"{name}.reactivation_accuracy must be null without events"
            )
    else:
        if accuracy is None:
            raise ValueError(
                f"{name}.reactivation_accuracy is required with events"
            )
        normalized_accuracy = _require_unit_interval(
            accuracy,
            name=f"{name}.reactivation_accuracy",
        )
        if normalized_accuracy != correct_reactivations / reactivations:
            raise ValueError(
                f"{name}.reactivation_accuracy does not match counts"
            )
    return metrics


def _validate_delta(
    value: object,
    *,
    horizon: int,
    persistent: Mapping[str, object],
    baseline: Mapping[str, object],
) -> None:
    delta = _require_exact_keys(value, _METRIC_KEYS, name="delta")
    _require_finite_number(delta["t_mAP"], name="delta.t_mAP")
    _require_finite_number(delta["t_REC"], name="delta.t_REC")
    per_stage = delta["per_stage_AP"]
    expected_stages = {str(stage) for stage in range(1, horizon + 1)}
    if not isinstance(per_stage, Mapping) or set(per_stage) != expected_stages:
        raise ValueError("delta.per_stage_AP must contain every stage")
    for stage, stage_value in per_stage.items():
        _require_finite_number(stage_value, name=f"delta.per_stage_AP.{stage}")
    for field in (
        "matched_identity_observations",
        "identity_switches",
        "reactivation_events",
        "correct_reactivations",
    ):
        _require_signed_integer(delta[field], name=f"delta.{field}")
    if delta["reactivation_accuracy"] is not None:
        _require_finite_number(
            delta["reactivation_accuracy"],
            name="delta.reactivation_accuracy",
        )
    if dict(delta) != _comparison_delta(persistent, baseline):
        raise ValueError("delta must equal persistent minus internal_baseline")


def _validate_complete_artifact(
    artifact: object,
    *,
    args: argparse.Namespace,
) -> None:
    root = _require_exact_keys(artifact, _ROOT_KEYS, name="artifact root")
    if (
        _require_integer(
            root["schema_version"],
            name="schema_version",
            minimum=1,
        )
        != SCHEMA_VERSION
    ):
        raise ValueError("artifact schema_version must be 2")
    if root["status"] != "pass":
        raise ValueError("completed evaluation must have status=pass")
    if root["method"] != METHOD_NAME:
        raise ValueError("artifact method is invalid")
    if root["errors"] != []:
        raise ValueError("passing artifact errors must be empty")

    source_commit = root["source_commit"]
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ValueError("source_commit must be a lowercase 40-character Git hash")
    _validate_source_tree_contract(
        root["source_tree_contract"],
        source_commit=source_commit,
        args=args,
    )

    checkpoint = _require_exact_keys(
        root["checkpoint"],
        frozenset({"ref", "sha256"}),
        name="checkpoint",
    )
    _require_formal_checkpoint_reference(checkpoint["ref"])
    sha256 = checkpoint["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("checkpoint sha256 is invalid")

    settings = _require_exact_keys(
        root["settings"],
        frozenset(
            {
                "capacity",
                "local_window",
                "internal_baseline_identity",
                "shared_rescene_outputs",
            }
        ),
        name="settings",
    )
    capacity = _require_integer(
        settings["capacity"],
        name="settings.capacity",
        minimum=1,
    )
    local_window = _require_integer(
        settings["local_window"],
        name="settings.local_window",
        minimum=1,
    )
    if capacity != args.capacity or local_window != LOCAL_WINDOW:
        raise ValueError("artifact settings do not match the requested evaluation")
    if settings["internal_baseline_identity"] != "local_query_index":
        raise ValueError("artifact internal baseline identity is invalid")
    if settings["shared_rescene_outputs"] is not True:
        raise ValueError("baseline must reuse the persistent ReScene outputs")
    _validate_legacy_parity(root["legacy_parity"], capacity=capacity)

    horizons = root["horizons"]
    if not isinstance(horizons, list):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            "horizons must be a list"
        )
    if tuple(args.horizons) != DEFAULT_HORIZONS:
        raise ValueError("artifact validation requires horizons 2 3 4 5")
    if len(horizons) != len(DEFAULT_HORIZONS):
        raise ValueError("horizons must contain every requested horizon")
    serialized_sizes: list[int] = []
    for expected_horizon, item in zip(DEFAULT_HORIZONS, horizons, strict=True):
        horizon = _require_exact_keys(item, _HORIZON_KEYS, name="horizon")
        if (
            _require_integer(horizon["T"], name="T", minimum=1)
            != expected_horizon
        ):
            raise ValueError("horizons are incomplete or out of order")
        loaded_sequences = _require_integer(
            horizon["loaded_sequences"],
            name="loaded_sequences",
            minimum=1,
        )
        if loaded_sequences != OFFICIAL_FILTERED_SEQUENCE_COUNTS[expected_horizon]:
            raise ValueError(
                "loaded_sequences does not match the official filtered database"
            )
        persistent = _validate_metric_block(
            horizon["persistent"],
            horizon=expected_horizon,
            loaded_sequences=loaded_sequences,
            capacity=capacity,
            name="persistent",
            persistent=True,
        )
        baseline = _validate_metric_block(
            horizon["internal_baseline"],
            horizon=expected_horizon,
            loaded_sequences=loaded_sequences,
            capacity=capacity,
            name="internal_baseline",
            persistent=False,
        )
        _validate_delta(
            horizon["delta"],
            horizon=expected_horizon,
            persistent=persistent,
            baseline=baseline,
        )
        resources = _require_exact_keys(
            horizon["resources"],
            _RESOURCE_KEYS,
            name="resources",
        )
        _require_integer(
            resources["peak_allocated_cuda_bytes"],
            name="resources.peak_allocated_cuda_bytes",
            minimum=1,
        )
        serialized_state_bytes = _require_integer(
            resources["serialized_state_bytes"],
            name="resources.serialized_state_bytes",
            minimum=1,
        )
        mean_latency_ms = _require_finite_number(
            resources["mean_latency_ms"],
            name="resources.mean_latency_ms",
        )
        throughput = _require_finite_number(
            resources["throughput_sequences_per_second"],
            name="resources.throughput_sequences_per_second",
        )
        if mean_latency_ms <= 0.0:
            raise ValueError("mean_latency_ms must be positive")
        if throughput <= 0.0:
            raise ValueError("throughput_sequences_per_second must be positive")
        if not math.isclose(
            mean_latency_ms * throughput,
            1000.0,
            rel_tol=RUNTIME_RECIPROCAL_RTOL,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "mean_latency_ms and throughput_sequences_per_second "
                "must be reciprocal"
            )
        serialized_sizes.append(serialized_state_bytes)

    bounded = _require_exact_keys(
        root["bounded_state"],
        frozenset({"constant_shape", "maximum_state_bytes"}),
        name="bounded_state",
    )
    if bounded["constant_shape"] is not True:
        raise ValueError("passing evaluation must prove constant state shape")
    maximum_state_bytes = _require_integer(
        bounded["maximum_state_bytes"],
        name="maximum_state_bytes",
        minimum=1,
    )
    if maximum_state_bytes != max(serialized_sizes):
        raise ValueError("maximum_state_bytes does not match horizon measurements")
    conclusion = _require_exact_keys(
        root["conclusion"],
        _CONCLUSION_KEYS,
        name="conclusion",
    )
    if dict(conclusion) != _derive_conclusion(horizons, bounded):
        raise ValueError("conclusion must be derived from measured evidence")


def _render_markdown(artifact: Mapping[str, object]) -> str:
    conclusion = artifact["conclusion"]
    checkpoint = artifact["checkpoint"]
    source_contract = artifact["source_tree_contract"]
    parity = artifact["legacy_parity"]
    settings = artifact["settings"]
    improvements = conclusion["identity_improvements"]
    rendered_improvements = ", ".join(improvements) if improvements else "none"
    allowed_outputs = source_contract["allowed_untracked_outputs"]
    rendered_outputs = ", ".join(allowed_outputs) if allowed_outputs else "none"
    lines = [
        "# Persist4D P5 MVP Evaluation",
        "",
        (
            "Purpose: fixed-capacity streaming association diagnosis; metrics "
            "are not an official AP target."
        ),
        "",
        f"Status: `{artifact['status']}`",
        "",
        f"Conclusion: `{conclusion['label']}`",
        "",
        f"Reason: `{conclusion['reason']}`",
        "",
        f"Identity improvements: `{rendered_improvements}`",
        "",
        f"Source commit: `{artifact['source_commit']}`",
        "",
        f"Source tree contract: `{source_contract['status']}`",
        "",
        f"Allowed untracked outputs: `{rendered_outputs}`",
        "",
        f"Checkpoint reference: `{checkpoint['ref']}`",
        "",
        f"Checkpoint SHA-256: `{checkpoint['sha256']}`",
        "",
        f"Legacy predictions unchanged: `{str(parity['legacy_predictions_unchanged']).lower()}`",
        "",
        f"Legacy parity verification: `{parity['verified_by']}`",
        "",
        f"Query feature shape: `{parity['query_feature_shape']}`",
        "",
        f"Internal baseline identity: `{settings['internal_baseline_identity']}`",
        "",
        (
            "The internal no-memory baseline reuses each persistent run's same "
            "latest-stage valid ReScene observations, masks, and classifications; "
            "only the cross-stage identity is the local query index."
        ),
        "",
        "## Persistent And Baseline Metrics",
        "",
        (
            "| T | Sequences | Method | t-mAP | t-REC | Per-stage AP | "
            "Matched ID obs | ID switches | Reactivation events | "
            "Correct reactivations | Reactivation accuracy | Rejected births |"
        ),
        "|---:|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in artifact["horizons"]:
        for method in ("persistent", "internal_baseline"):
            metrics = horizon[method]
            per_stage = "; ".join(
                f"{stage}={metrics['per_stage_AP'][str(stage)]:.6f}"
                for stage in range(1, horizon["T"] + 1)
            )
            accuracy = metrics["reactivation_accuracy"]
            rendered_accuracy = (
                "n/a" if accuracy is None else f"{accuracy:.6f}"
            )
            rejected_births = (
                str(metrics["rejected_births"])
                if method == "persistent"
                else "n/a"
            )
            lines.append(
                f"| {horizon['T']} | {horizon['loaded_sequences']} | {method} | "
                f"{metrics['t_mAP']:.6f} | {metrics['t_REC']:.6f} | "
                f"{per_stage} | {metrics['matched_identity_observations']} | "
                f"{metrics['identity_switches']} | "
                f"{metrics['reactivation_events']} | "
                f"{metrics['correct_reactivations']} | {rendered_accuracy} | "
                f"{rejected_births} |"
            )
    lines.extend(
        [
            "",
            "## Differences",
            "",
            (
                "| T | Comparison | Delta t-mAP | Delta t-REC | "
                "Delta per-stage AP | Delta matched obs | Delta switches | "
                "Delta reactivation events | Delta correct reactivations | "
                "Delta reactivation accuracy |"
            ),
            "|---:|:---|---:|---:|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in artifact["horizons"]:
        delta = horizon["delta"]
        per_stage = "; ".join(
            f"{stage}={delta['per_stage_AP'][str(stage)]:+.6f}"
            for stage in range(1, horizon["T"] + 1)
        )
        accuracy = delta["reactivation_accuracy"]
        rendered_accuracy = "n/a" if accuracy is None else f"{accuracy:+.6f}"
        lines.append(
            f"| {horizon['T']} | delta (persistent - baseline) | "
            f"{delta['t_mAP']:+.6f} | {delta['t_REC']:+.6f} | {per_stage} | "
            f"{delta['matched_identity_observations']:+d} | "
            f"{delta['identity_switches']:+d} | "
            f"{delta['reactivation_events']:+d} | "
            f"{delta['correct_reactivations']:+d} | {rendered_accuracy} |"
        )
    lines.extend(
        [
            "",
            "## Resources And State",
            "",
            (
                "| T | Peak CUDA bytes | Mean latency (ms) | "
                "Throughput (seq/s) | State bytes |"
            ),
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in artifact["horizons"]:
        resources = horizon["resources"]
        lines.append(
            f"| {horizon['T']} | {resources['peak_allocated_cuda_bytes']} | "
            f"{resources['mean_latency_ms']:.6f} | "
            f"{resources['throughput_sequences_per_second']:.6f} | "
            f"{resources['serialized_state_bytes']} |"
        )
    bounded = artifact["bounded_state"]
    lines.extend(
        [
            "",
            (
                "The fixed-capacity state remained constant in shape across "
                "T=2/3/4/5."
            ),
            "",
            f"Constant state shape: `{str(bounded['constant_shape']).lower()}`",
            "",
            f"Maximum serialized state bytes: `{bounded['maximum_state_bytes']}`",
            "",
        ]
    )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_root: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("git rev-parse returned an invalid source commit")
    return commit


@dataclass(frozen=True)
class _SourceTreeGuard:
    repo_root: Path
    source_commit: str
    allowed_untracked_paths: tuple[str, ...]


def _git_paths(repo_root: Path, *arguments: str) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    return tuple(
        os.fsdecode(item)
        for item in completed.stdout.split(b"\0")
        if item
    )


def _repository_output_paths(
    repo_root: Path,
    output_paths: Iterable[Path],
) -> tuple[str, ...]:
    repository = repo_root.resolve()
    relative_paths: set[str] = set()
    for output_path in output_paths:
        candidate = Path(output_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            relative = candidate.resolve().relative_to(repository)
        except ValueError:
            continue
        if relative == Path("."):
            raise ValueError("an output path cannot be the repository root")
        relative_paths.add(relative.as_posix())
    return tuple(sorted(relative_paths))


def _git_index_flag_paths(repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-v", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    )
    flagged: list[str] = []
    for raw_record in completed.stdout.split(b"\0"):
        if not raw_record:
            continue
        record = os.fsdecode(raw_record)
        if len(record) < 3 or record[1] != " ":
            raise RuntimeError("git ls-files returned an invalid index record")
        tag = record[0]
        if tag == "S" or tag.islower():
            flagged.append(record[2:])
    return tuple(sorted(set(flagged)))


def _require_clean_source_snapshot(guard: _SourceTreeGuard) -> None:
    if git_commit(guard.repo_root) != guard.source_commit:
        raise RuntimeError("Git HEAD changed during evaluation")
    staged = _git_paths(
        guard.repo_root,
        "diff",
        "--cached",
        "--name-only",
        "-z",
        "--no-ext-diff",
    )
    unstaged = _git_paths(
        guard.repo_root,
        "diff",
        "--name-only",
        "-z",
        "--no-ext-diff",
    )
    index_flag_paths = _git_index_flag_paths(guard.repo_root)
    untracked = set(
        _git_paths(
            guard.repo_root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
    )
    if git_commit(guard.repo_root) != guard.source_commit:
        raise RuntimeError("Git HEAD changed during evaluation")
    if staged or unstaged:
        raise RuntimeError("tracked source changes are forbidden")
    if index_flag_paths:
        raise RuntimeError("hidden Git index flags are forbidden")
    disallowed = untracked.difference(guard.allowed_untracked_paths)
    if disallowed:
        raise RuntimeError("untracked files outside declared outputs are forbidden")


def _begin_source_tree_contract(
    *,
    repo_root: Path,
    output_paths: Iterable[Path],
) -> _SourceTreeGuard:
    repository = repo_root.resolve()
    guard = _SourceTreeGuard(
        repo_root=repository,
        source_commit=git_commit(repository),
        allowed_untracked_paths=_repository_output_paths(
            repository,
            output_paths,
        ),
    )
    _require_clean_source_snapshot(guard)
    return guard


def _source_tree_contract_payload(
    guard: _SourceTreeGuard,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": guard.source_commit,
        "tracked_tree_clean": True,
        "index_clean": True,
        "allowed_untracked_outputs": [
            f"repo:{path}" for path in guard.allowed_untracked_paths
        ],
        "only_declared_outputs_untracked": True,
        "generation_head_unchanged": True,
    }


def _finalize_source_tree_contract(
    guard: _SourceTreeGuard,
) -> dict[str, object]:
    _require_clean_source_snapshot(guard)
    return _source_tree_contract_payload(guard)


def _compose_runtime_config() -> tuple[Any, Any]:
    import hydra
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None,
        config_dir=str(PROJECT_ROOT / "conf"),
    ):
        config = compose(
            config_name="config_p2_rescene4d_concerto_t2",
            overrides=[
                "logging=local_csv",
                "+model.return_query_features=true",
                "general.gpus=1",
                "general.train_mode=false",
            ],
        )
        memory_config = compose(config_name="model/persist4d").persist4d
    if not hydra:
        raise AssertionError("Hydra composition did not initialize")
    return config, memory_config


def _resolve_checkpoint(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    formal = DEFAULT_CHECKPOINT
    try:
        candidate_mode = candidate.lstat().st_mode
        formal_mode = formal.lstat().st_mode
        resolved = candidate.resolve(strict=True)
        resolved_formal = formal.resolve(strict=True)
    except OSError as error:
        raise ValueError("formal checkpoint must be an existing regular file") from error
    if candidate.is_symlink() or formal.is_symlink():
        raise ValueError("formal checkpoint must not be a symbolic link")
    if not stat.S_ISREG(candidate_mode) or not stat.S_ISREG(formal_mode):
        raise ValueError("formal checkpoint must be a regular file")
    if resolved != resolved_formal:
        raise ValueError("checkpoint must be the formal repository checkpoint")
    return resolved_formal


def _validate_cuda_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must identify one CUDA device") from error
    if device.type != "cuda" or device.index is None:
        raise ValueError("device must identify one CUDA device, such as cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Persist4D evaluation")
    if device.index >= torch.cuda.device_count():
        raise ValueError("selected CUDA device is unavailable")
    torch.cuda.set_device(device)
    return device


def _load_system(config: Any, checkpoint: Path, device: torch.device) -> Any:
    from trainer.trainer import InstanceSegmentation

    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(  # noqa: TRY004 - checkpoint validation contract.
            "checkpoint must contain a full Lightning mapping"
        )
    state_dict = checkpoint_payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(  # noqa: TRY004 - checkpoint validation contract.
            "checkpoint is missing its full state_dict"
        )
    system = InstanceSegmentation(config)
    system.load_state_dict(state_dict, strict=True)
    system.to(device)
    system.eval()
    return system


def _move_data_to_device(data: Any, device: torch.device) -> Any:
    for key in list(data.keys()):
        if isinstance(data[key], Tensor):
            data[key] = data[key].to(device, non_blocking=True)
    return data


def _move_targets_to_device(
    targets: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for key, value in target.items()
        }
        for target in targets
    ]


def _segment_stages(target: Mapping[str, Any]) -> Tensor:
    point2segment = target.get("point2segment")
    temporal_stages = target.get("temporal_stages")
    if not isinstance(point2segment, Tensor) or not isinstance(
        temporal_stages, Tensor
    ):
        raise ValueError(  # noqa: TRY004 - validation contract.
            "target must contain tensor point2segment and temporal_stages"
        )
    if (
        point2segment.ndim != 1
        or temporal_stages.ndim != 1
        or point2segment.shape != temporal_stages.shape
        or point2segment.numel() == 0
    ):
        raise ValueError("target point2segment and temporal_stages must align")
    if torch.any(point2segment < 0).item() or torch.any(temporal_stages < 0).item():
        raise ValueError("segment and temporal stage indices must be non-negative")
    segment_count = int(point2segment.max().item()) + 1
    maximum_stage = int(temporal_stages.max().item())
    minimum = torch.full(
        (segment_count,),
        maximum_stage + 1,
        dtype=temporal_stages.dtype,
        device=temporal_stages.device,
    )
    maximum = torch.full(
        (segment_count,),
        -1,
        dtype=temporal_stages.dtype,
        device=temporal_stages.device,
    )
    minimum.scatter_reduce_(
        0,
        point2segment.long(),
        temporal_stages,
        reduce="amin",
        include_self=True,
    )
    maximum.scatter_reduce_(
        0,
        point2segment.long(),
        temporal_stages,
        reduce="amax",
        include_self=True,
    )
    if torch.any(minimum != maximum).item():
        raise ValueError("one segment cannot span more than one temporal stage")
    return minimum


def _latest_full_resolution_masks(
    system: Any,
    output: Mapping[str, Any],
    target: Mapping[str, Any],
    data: Any,
    *,
    latest_local_stage: int,
) -> Tensor:
    pred_masks = output.get("pred_masks")
    if not isinstance(pred_masks, list) or len(pred_masks) != 1:
        raise ValueError("streaming evaluation requires one pred_masks tensor")
    segment_logits = pred_masks[0]
    point2segment = target["point2segment"]
    if not isinstance(segment_logits, Tensor) or segment_logits.ndim != 2:
        raise ValueError("pred_masks[0] must have shape [S, Q]")
    if segment_logits.shape[0] <= int(point2segment.max().item()):
        raise ValueError("pred_masks do not cover every low-resolution segment")
    low_resolution_masks = (segment_logits > 0).float()[point2segment]
    full_masks = system._get_full_res_mask(
        low_resolution_masks,
        data.inverse_maps[0],
        data.target_full[0]["point2segment"],
    ).bool()
    full_stages = data.target_full[0]["temporal_stages"].detach().cpu()
    selector = full_stages == latest_local_stage
    if not torch.any(selector).item():
        raise ValueError("full-resolution target is missing the latest local stage")
    return full_masks[selector].transpose(0, 1).contiguous()


def _foreground_classes(class_prob: Tensor, background_class: int) -> Tensor:
    if not 0 <= background_class < class_prob.shape[-1]:
        raise ValueError("background_class is outside class probabilities")
    foreground = class_prob.clone()
    foreground[..., background_class] = -torch.inf
    return foreground.argmax(dim=-1)


def _match_stage_identities(
    *,
    gt_ids: Tensor,
    gt_classes: Tensor,
    gt_masks: Tensor,
    predicted_ids: Tensor,
    predicted_classes: Tensor,
    predicted_masks: Tensor,
    minimum_iou: float = 0.5,
) -> tuple[list[int], list[int]]:
    if gt_masks.ndim != 2 or predicted_masks.ndim != 2:
        raise ValueError("identity diagnostic masks must be two-dimensional")
    if gt_masks.shape[1] != predicted_masks.shape[1]:
        raise ValueError("GT and predicted masks must cover the same stage points")
    if (
        gt_ids.shape != gt_classes.shape
        or gt_ids.ndim != 1
        or gt_ids.shape[0] != gt_masks.shape[0]
    ):
        raise ValueError("GT identity diagnostic tensors must align")
    if (
        predicted_ids.shape != predicted_classes.shape
        or predicted_ids.ndim != 1
        or predicted_ids.shape[0] != predicted_masks.shape[0]
    ):
        raise ValueError("predicted identity diagnostic tensors must align")

    candidates: list[tuple[float, int, int]] = []
    for gt_index in range(gt_masks.shape[0]):
        for predicted_index in range(predicted_masks.shape[0]):
            if gt_classes[gt_index].item() != predicted_classes[predicted_index].item():
                continue
            intersection = torch.logical_and(
                gt_masks[gt_index],
                predicted_masks[predicted_index],
            ).sum().item()
            union = torch.logical_or(
                gt_masks[gt_index],
                predicted_masks[predicted_index],
            ).sum().item()
            iou = float(intersection / union) if union else 0.0
            if iou >= minimum_iou:
                candidates.append((-iou, gt_index, predicted_index))
    candidates.sort()
    used_gt: set[int] = set()
    used_predicted: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, gt_index, predicted_index in candidates:
        if gt_index in used_gt or predicted_index in used_predicted:
            continue
        used_gt.add(gt_index)
        used_predicted.add(predicted_index)
        matched.append((gt_index, predicted_index))
    matched.sort()
    return (
        [int(gt_ids[gt_index].item()) for gt_index, _ in matched],
        [int(predicted_ids[predicted_index].item()) for _, predicted_index in matched],
    )


def _state_signature(state: Any) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    signature = []
    for field in fields(state):
        value = getattr(state, field.name)
        if not isinstance(value, Tensor):
            raise ValueError(  # noqa: TRY004 - state validation contract.
                "persistent state fields must all be tensors"
            )
        signature.append((field.name, tuple(value.shape), str(value.dtype)))
    return tuple(signature)


def _serialized_state_bytes(state: Any) -> int:
    payload = {
        field.name: getattr(state, field.name).detach().cpu()
        for field in fields(state)
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def _accumulated_prediction(
    accumulator: SequenceAccumulator,
    stage_point_counts: Sequence[int],
    *,
    background_class: int,
    dataset: Any,
) -> dict[str, Tensor]:
    if len(stage_point_counts) != len(accumulator.stage_masks):
        raise ValueError("stage point counts must align with accumulated masks")
    observed_slots = torch.nonzero(
        accumulator.class_prob_count > 0,
        as_tuple=False,
    ).flatten()
    total_points = sum(stage_point_counts)
    masks = torch.zeros(
        total_points,
        observed_slots.numel(),
        dtype=torch.bool,
    )
    offset = 0
    for stage, point_count in zip(
        accumulator.stage_masks,
        stage_point_counts,
        strict=True,
    ):
        for column, slot_tensor in enumerate(observed_slots):
            slot = int(slot_tensor.item())
            if slot not in stage:
                continue
            mask = stage[slot]
            if mask.shape != (point_count,):
                raise ValueError("accumulated stage mask has an invalid point count")
            masks[offset : offset + point_count, column] = mask
        offset += point_count

    mean_prob = accumulator.class_prob_mean()[observed_slots]
    model_classes = _foreground_classes(mean_prob, background_class)
    foreground_prob = mean_prob.clone()
    foreground_prob[:, background_class] = -torch.inf
    scores = foreground_prob.amax(dim=1)
    raw_classes = dataset._remap_model_output(
        model_classes + int(dataset.label_offset)
    )
    return {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_classes": raw_classes,
    }


def _metric_target(target: Mapping[str, Any], dataset: Any) -> dict[str, Any]:
    copied = {
        key: value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        for key, value in target.items()
    }
    copied["labels"] = dataset._remap_model_output(
        copied["labels"] + int(dataset.label_offset)
    )
    return copied


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"metric adapter did not emit required key: {key}")
    value = metrics[key]
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"metric {key} must be scalar")
        value = value.detach().cpu().item()
    return _require_finite_number(value, name=key)


def _instantiate_metric(dataset: Any, horizon: int) -> Any:
    from stmetrics import (
        InstanceMetrics,
        LegacyAPEvaluator,
        SelectTimestepEvaluator,
        TemporalEvaluator,
    )

    dataset_spec = Path(dataset.data_dir[0]) / f"{dataset.dataset_name}.yaml"
    heads = [TemporalEvaluator(recall=True, aux="changes"), LegacyAPEvaluator()]
    heads.extend(SelectTimestepEvaluator(stage) for stage in range(horizon))
    return InstanceMetrics(
        dataset=str(dataset_spec),
        heads=heads,
        log_prefix="val",
        timestep_key="temporal_stages",
    )


def _summarize_method_metrics(
    metric: Any,
    *,
    horizon: int,
    identity_totals: Mapping[str, int],
    rejected_births: int | None = None,
) -> dict[str, object]:
    computed = metric.compute()
    reactivation_events = int(identity_totals["reactivation_events"])
    correct_reactivations = int(identity_totals["correct_reactivations"])
    result: dict[str, object] = {
        "t_mAP": _metric_value(computed, "val_mean_t-AP"),
        "t_REC": _metric_value(computed, "val_mean_t-REC"),
        "per_stage_AP": {
            str(stage): _metric_value(
                computed,
                f"val_mean_stage{stage}-AP",
            )
            for stage in range(1, horizon + 1)
        },
        **{key: int(value) for key, value in identity_totals.items()},
        "reactivation_accuracy": (
            correct_reactivations / reactivation_events
            if reactivation_events
            else None
        ),
    }
    if rejected_births is not None:
        result["rejected_births"] = int(rejected_births)
    return result


def _run_legacy_parity_step(
    streaming: Any,
    *,
    x: Any,
    point2segment: list[Tensor],
    raw_coordinates: Any,
    segment_stages: list[Tensor],
    state: Any,
    stage_index: int,
    device: torch.device,
    capacity: int,
) -> tuple[dict[str, object], Any, float, dict[str, object]]:
    import random

    import numpy as np

    base_model = streaming.base_model
    original_query_export = base_model.return_query_features
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()

    def seed_fixed_inference() -> None:
        random.seed(45)
        np.random.seed(45)
        torch.manual_seed(45)
        torch.cuda.manual_seed_all(45)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[device.index]):
            base_model.return_query_features = False
            seed_fixed_inference()
            with torch.inference_mode():
                disabled_output = base_model(
                    x,
                    point2segment,
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
            disabled_snapshot = _legacy_value_snapshot(
                disabled_output,
                path="disabled_output",
            )
            del disabled_output

            base_model.return_query_features = True
            seed_fixed_inference()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                enabled_output, next_state = streaming.forward_step(
                    x=x,
                    point2segment=point2segment,
                    raw_coordinates=raw_coordinates,
                    segment_stages=segment_stages,
                    state=state,
                    stage_index=stage_index,
                    is_eval=True,
                )
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            enabled_snapshot = {
                key: (
                    None
                    if key in _PERSISTENT_OUTPUT_KEYS
                    else _legacy_value_snapshot(
                        value,
                        path=f"enabled_output.{key}",
                    )
                )
                for key, value in enabled_output.items()
            }
            parity = _legacy_parity_result(
                disabled_snapshot,
                enabled_snapshot,
                capacity=capacity,
            )
    finally:
        base_model.return_query_features = original_query_export
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.use_deterministic_algorithms(
            deterministic_enabled,
            warn_only=deterministic_warn_only,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        torch.set_float32_matmul_precision(matmul_precision)
    return enabled_output, next_state, elapsed, parity


def _evaluate_horizon(
    *,
    system: Any,
    config: Any,
    memory_config: Any,
    horizon: int,
    capacity: int,
    device: torch.device,
) -> tuple[
    dict[str, object],
    list[tuple[tuple[str, tuple[int, ...], str], ...]],
    dict[str, object] | None,
]:
    import hydra
    from omegaconf import OmegaConf

    from datasets.streaming_sequence import causal_windows
    from models.persistent_memory import PersistentMemory, build_local_observation
    from models.streaming_rescene import StreamingReScene

    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    dataset_config.temporal_window = horizon
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    persistent_metric = _instantiate_metric(dataset, horizon)
    baseline_metric = _instantiate_metric(dataset, horizon)
    memory = PersistentMemory(
        capacity=capacity,
        class_weight=float(memory_config.class_weight),
        association_threshold=float(memory_config.association_threshold),
        update_rate=float(memory_config.update_rate),
        max_update_rate=float(memory_config.max_update_rate),
    )
    observation_settings = {
        "background_class": int(memory_config.background_class),
        "confidence_threshold": float(memory_config.confidence_threshold),
        "mask_threshold": float(memory_config.mask_threshold),
        "minimum_mask_support": int(memory_config.minimum_mask_support),
    }
    streaming = StreamingReScene(system.model, memory, observation_settings)
    streaming.to(device)
    streaming.eval()

    loaded_sequences = 0
    rejected_births = 0
    identity_totals = {
        method: {
            "matched_identity_observations": 0,
            "identity_switches": 0,
            "reactivation_events": 0,
            "correct_reactivations": 0,
        }
        for method in ("persistent", "internal_baseline")
    }
    sequence_latency_seconds: list[float] = []
    state_signatures: list[
        tuple[tuple[str, tuple[int, ...], str], ...]
    ] = []
    state_sizes: list[int] = []
    legacy_parity: dict[str, object] | None = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for sequence_index, scan_indices in enumerate(dataset.sequence_indices):
        persistent_accumulator: SequenceAccumulator | None = None
        baseline_accumulator: SequenceAccumulator | None = None
        state = None
        gt_ids_by_stage: dict[str, list[list[int]]] = {
            "persistent": [],
            "internal_baseline": [],
        }
        predicted_ids_by_stage: dict[str, list[list[int]]] = {
            "persistent": [],
            "internal_baseline": [],
        }
        stage_point_counts: list[int] = []
        forward_seconds = 0.0

        for global_stage, window in enumerate(causal_windows(scan_indices)):
            data = targets = output = next_state = None
            sample = segment_stages = raw_coordinates = None
            class_prob = slot_ids = full_masks = full_target = None
            observation = cpu_class_prob = cpu_slot_ids = None
            cpu_full_masks = cpu_valid = baseline_ids = None
            predicted_classes = None
            try:
                sample = dataset.load_scan_indices(
                    sequence_index,
                    window,
                    change_file=None,
                )
                data, targets, names = collate([sample])
                if len(targets) != 1 or list(names) != [dataset.sequence_names[sequence_index]]:
                    raise ValueError("validation collator changed the requested sequence")
                data = _move_data_to_device(data, device)
                targets = _move_targets_to_device(targets, device)
                segment_stages = _segment_stages(targets[0])
                latest_local_stage = int(segment_stages.max().item())
                raw_coordinates = system._process_raw_coordinates(data)

                verify_legacy_parity = (
                    horizon == 2
                    and sequence_index == 0
                    and tuple(window) == tuple(scan_indices)
                    and legacy_parity is None
                )
                if verify_legacy_parity:
                    (
                        output,
                        next_state,
                        elapsed,
                        legacy_parity,
                    ) = _run_legacy_parity_step(
                        streaming,
                        x=data,
                        point2segment=[targets[0]["point2segment"]],
                        raw_coordinates=raw_coordinates,
                        segment_stages=[segment_stages],
                        state=state,
                        stage_index=global_stage,
                        device=device,
                        capacity=capacity,
                    )
                    forward_seconds += elapsed
                else:
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.inference_mode():
                        output, next_state = streaming.forward_step(
                            x=data,
                            point2segment=[targets[0]["point2segment"]],
                            raw_coordinates=raw_coordinates,
                            segment_stages=[segment_stages],
                            state=state,
                            stage_index=global_stage,
                            is_eval=True,
                        )
                    torch.cuda.synchronize(device)
                    forward_seconds += time.perf_counter() - started

                observation = build_local_observation(
                    output,
                    [segment_stages],
                    latest_stage=latest_local_stage,
                    **observation_settings,
                )
                class_prob = observation.class_prob[0]
                slot_ids = output["persistent_slot_ids"][0]
                full_masks = _latest_full_resolution_masks(
                    system,
                    output,
                    targets[0],
                    data,
                    latest_local_stage=latest_local_stage,
                )
                if persistent_accumulator is None:
                    persistent_accumulator = SequenceAccumulator.empty(
                        capacity=capacity,
                        class_count=class_prob.shape[1],
                    )
                    baseline_accumulator = SequenceAccumulator.empty(
                        capacity=capacity,
                        class_count=class_prob.shape[1],
                    )
                cpu_class_prob = class_prob.detach().cpu()
                cpu_slot_ids = slot_ids.detach().cpu()
                cpu_full_masks = full_masks.detach().cpu()
                cpu_valid = observation.valid[0].detach().cpu()
                baseline_ids = _accumulate_shared_stage(
                    persistent=persistent_accumulator,
                    baseline=baseline_accumulator,
                    masks=cpu_full_masks,
                    class_prob=cpu_class_prob,
                    persistent_slot_ids=cpu_slot_ids,
                    valid_observations=cpu_valid,
                )
                stage_point_counts.append(cpu_full_masks.shape[1])
                rejected_births += int(
                    output["persistent_rejected_births"].sum().item()
                )

                full_target = data.target_full[0]
                full_stage_selector = (
                    full_target["temporal_stages"].detach().cpu()
                    == latest_local_stage
                )
                present_gt = full_target["masks"][:, full_stage_selector].any(dim=1)
                predicted_classes = _foreground_classes(
                    cpu_class_prob,
                    observation_settings["background_class"],
                )
                for method, predicted_identity_ids in (
                    ("persistent", cpu_slot_ids),
                    ("internal_baseline", baseline_ids),
                ):
                    valid_predictions = predicted_identity_ids >= 0
                    matched_gt, matched_predicted = _match_stage_identities(
                        gt_ids=full_target["ids"][present_gt].detach().cpu(),
                        gt_classes=full_target["labels"][present_gt]
                        .detach()
                        .cpu(),
                        gt_masks=full_target["masks"][present_gt][
                            :, full_stage_selector
                        ]
                        .detach()
                        .cpu()
                        .bool(),
                        predicted_ids=predicted_identity_ids[valid_predictions],
                        predicted_classes=predicted_classes[valid_predictions],
                        predicted_masks=cpu_full_masks[valid_predictions],
                    )
                    gt_ids_by_stage[method].append(matched_gt)
                    predicted_ids_by_stage[method].append(matched_predicted)

                state = next_state.detach()
                state_signatures.append(_state_signature(state))
                state_sizes.append(_serialized_state_bytes(state))
            finally:
                del (
                    data,
                    targets,
                    output,
                    next_state,
                    sample,
                    segment_stages,
                    raw_coordinates,
                    class_prob,
                    slot_ids,
                    full_masks,
                    full_target,
                    observation,
                    cpu_class_prob,
                    cpu_slot_ids,
                    cpu_full_masks,
                    cpu_valid,
                    baseline_ids,
                    predicted_classes,
                )

        if (
            persistent_accumulator is None
            or baseline_accumulator is None
            or state is None
        ):
            raise RuntimeError("sequence produced no streaming stages")
        for method in ("persistent", "internal_baseline"):
            diagnostic = identity_diagnostics(
                gt_ids_by_stage[method],
                predicted_ids_by_stage[method],
            )
            for key in identity_totals[method]:
                identity_totals[method][key] += int(diagnostic[key])

        full_data = full_targets = None
        persistent_prediction = baseline_prediction = target = None
        try:
            full_sample = dataset[sequence_index]
            full_data, full_targets, names = collate([full_sample])
            if len(full_targets) != 1 or list(names) != [dataset.sequence_names[sequence_index]]:
                raise ValueError("full metric target does not match the sequence")
            persistent_prediction = _accumulated_prediction(
                persistent_accumulator,
                stage_point_counts,
                background_class=observation_settings["background_class"],
                dataset=dataset,
            )
            baseline_prediction = _accumulated_prediction(
                baseline_accumulator,
                stage_point_counts,
                background_class=observation_settings["background_class"],
                dataset=dataset,
            )
            target = _metric_target(full_data.target_full[0], dataset)
            if any(
                prediction["pred_masks"].shape[0] != target["masks"].shape[1]
                for prediction in (persistent_prediction, baseline_prediction)
            ):
                raise ValueError("accumulated predictions do not align with metric target")
            persistent_metric.update([persistent_prediction], [target])
            baseline_metric.update([baseline_prediction], [target])
        finally:
            del (
                full_data,
                full_targets,
                persistent_prediction,
                baseline_prediction,
                target,
            )
        loaded_sequences += 1
        sequence_latency_seconds.append(forward_seconds)
        del persistent_accumulator, baseline_accumulator, state

    if loaded_sequences == 0 or not state_signatures or not state_sizes:
        raise RuntimeError(f"T={horizon} validation dataset produced no sequences")
    total_seconds = sum(sequence_latency_seconds)
    if total_seconds <= 0.0:
        raise RuntimeError("measured evaluation latency must be positive")
    persistent = _summarize_method_metrics(
        persistent_metric,
        horizon=horizon,
        identity_totals=identity_totals["persistent"],
        rejected_births=rejected_births,
    )
    baseline = _summarize_method_metrics(
        baseline_metric,
        horizon=horizon,
        identity_totals=identity_totals["internal_baseline"],
    )
    result = {
        "T": horizon,
        "loaded_sequences": loaded_sequences,
        "persistent": persistent,
        "internal_baseline": baseline,
        "delta": _comparison_delta(persistent, baseline),
        "resources": {
            "peak_allocated_cuda_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "mean_latency_ms": 1000.0 * total_seconds / loaded_sequences,
            "throughput_sequences_per_second": loaded_sequences / total_seconds,
            "serialized_state_bytes": max(state_sizes),
        },
    }
    return result, state_signatures, legacy_parity


def run_real_evaluation(args: argparse.Namespace) -> dict[str, object]:
    checkpoint = _resolve_checkpoint(args.checkpoint)
    source_commit = git_commit()
    checkpoint_sha256 = sha256_file(checkpoint)
    device = _validate_cuda_device(args.device)

    original_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        config, memory_config = _compose_runtime_config()
        system = _load_system(config, checkpoint, device)
        horizon_results = []
        signatures = []
        legacy_parity_results = []
        for horizon in args.horizons:
            result, horizon_signatures, legacy_parity = _evaluate_horizon(
                system=system,
                config=config,
                memory_config=memory_config,
                horizon=horizon,
                capacity=args.capacity,
                device=device,
            )
            horizon_results.append(result)
            signatures.extend(horizon_signatures)
            if legacy_parity is not None:
                legacy_parity_results.append(legacy_parity)
    finally:
        os.chdir(original_directory)

    if git_commit() != source_commit:
        raise RuntimeError("source commit changed during evaluation")
    constant_shape = bool(signatures) and len(set(signatures)) == 1
    if not constant_shape:
        raise RuntimeError("persistent state shape changed across the completed run")
    if len(legacy_parity_results) != 1:
        raise RuntimeError("T=2 legacy parity must run exactly once")
    maximum_state_bytes = max(
        int(horizon["resources"]["serialized_state_bytes"])
        for horizon in horizon_results
    )
    bounded_state = {
        "constant_shape": constant_shape,
        "maximum_state_bytes": maximum_state_bytes,
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "method": METHOD_NAME,
        "source_commit": source_commit,
        "checkpoint": {
            "ref": FORMAL_CHECKPOINT_REFERENCE,
            "sha256": checkpoint_sha256,
        },
        "settings": {
            "capacity": args.capacity,
            "local_window": LOCAL_WINDOW,
            "internal_baseline_identity": "local_query_index",
            "shared_rescene_outputs": True,
        },
        "legacy_parity": legacy_parity_results[0],
        "horizons": horizon_results,
        "bounded_state": bounded_state,
        "conclusion": _derive_conclusion(horizon_results, bounded_state),
        "errors": [],
    }
    return artifact


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[argparse.Namespace], Mapping[str, object]] | None = None,
) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    hinted_output = _output_hint(arguments)
    try:
        args = parser.parse_args(arguments)
    except _CliUsageError as error:
        parser.print_usage(sys.stderr)
        print(f"{parser.prog}: error: {error}", file=sys.stderr)
        if hinted_output is not None and not hinted_output.exists():
            try:
                _write_failure_if_possible(hinted_output, error)
            except Exception as write_error:  # noqa: BLE001 - CLI failure fallback.
                print(f"failed to write failure artifact: {write_error}", file=sys.stderr)
        return 2

    if args.output.exists():
        print(f"refusing to overwrite existing output: {args.output}", file=sys.stderr)
        return 2

    selected_runner = run_real_evaluation if runner is None else runner
    source_guard: _SourceTreeGuard | None = None
    published_paths: list[Path] = []
    try:
        _validate_options(args)
        if runner is None:
            declared_outputs = [args.output]
            if args.markdown is not None:
                declared_outputs.append(args.markdown)
            source_guard = _begin_source_tree_contract(
                repo_root=PROJECT_ROOT,
                output_paths=declared_outputs,
            )
        artifact = dict(selected_runner(args))
        if source_guard is not None:
            if artifact.get("source_commit") != source_guard.source_commit:
                raise RuntimeError(
                    "evaluator source commit differs from source tree guard"
                )
            artifact["source_tree_contract"] = _source_tree_contract_payload(
                source_guard
            )
        _validate_complete_artifact(artifact, args=args)
        files_to_publish: list[tuple[Path, str]] = []
        if args.markdown is not None:
            files_to_publish.append((args.markdown, _render_markdown(artifact)))
        files_to_publish.append((args.output, _json_text(artifact)))
        _publish_text_files_new(files_to_publish)
        published_paths = [path for path, _ in files_to_publish]
        if source_guard is not None:
            finalized = _finalize_source_tree_contract(source_guard)
            if finalized != artifact["source_tree_contract"]:
                raise RuntimeError("source tree contract changed during evaluation")
    except Exception as error:  # noqa: BLE001 - every runtime fault is persisted.
        for path in published_paths:
            path.unlink(missing_ok=True)
        try:
            _write_failure_if_possible(args.output, error)
        except Exception as write_error:  # noqa: BLE001 - final CLI fallback.
            print(f"failed to write failure artifact: {write_error}", file=sys.stderr)
        print(f"Persist4D evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
