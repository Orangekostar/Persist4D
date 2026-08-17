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
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_NAME = "persist4d_p5_single_memory"
SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / (
    "rescene4d_concerto_t2_repro.ckpt"
)
DEFAULT_HORIZONS = (2, 3, 4, 5)
LOCAL_WINDOW = 2

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "method",
        "source_commit",
        "checkpoint",
        "settings",
        "horizons",
        "bounded_state",
        "errors",
    }
)
_HORIZON_KEYS = frozenset(
    {
        "T",
        "loaded_sequences",
        "t_mAP",
        "t_REC",
        "per_stage_AP",
        "matched_identity_observations",
        "identity_switches",
        "reactivation_events",
        "correct_reactivations",
        "reactivation_accuracy",
        "rejected_births",
        "peak_allocated_cuda_bytes",
        "mean_latency_ms",
        "throughput_sequences_per_second",
        "serialized_state_bytes",
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
            stage[slot] = normalized_masks[query_index].detach().bool().cpu()

        for query_index, slot in enumerate(cpu_slots):
            if slot < 0:
                continue
            self.class_prob_sum[slot].add_(cpu_prob[query_index])
            self.class_prob_count[slot].add_(1)
        self.stage_masks.append(stage)


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
        "errors": [
            {
                "type": type(error).__name__,
                "message": str(error),
            }
        ],
    }


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
    _positive_integer(args.capacity, name="capacity")
    if args.markdown is not None and args.markdown.resolve() == args.output.resolve():
        raise ValueError("JSON and Markdown output paths must be distinct")
    if args.markdown is not None and args.markdown.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {args.markdown}"
        )


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


def _require_finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            f"{name} must be numeric"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_complete_artifact(
    artifact: object,
    *,
    args: argparse.Namespace,
) -> None:
    root = _require_exact_keys(artifact, _ROOT_KEYS, name="artifact root")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("artifact schema_version must be 1")
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

    checkpoint = _require_exact_keys(
        root["checkpoint"],
        frozenset({"ref", "sha256"}),
        name="checkpoint",
    )
    reference = checkpoint["ref"]
    if not isinstance(reference, str) or not reference.startswith(
        ("repo:", "external:")
    ):
        raise ValueError("checkpoint ref must be portable")
    sha256 = checkpoint["sha256"]
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("checkpoint sha256 is invalid")

    settings = _require_exact_keys(
        root["settings"],
        frozenset({"capacity", "local_window"}),
        name="settings",
    )
    if settings["capacity"] != args.capacity or settings["local_window"] != 2:
        raise ValueError("artifact settings do not match the requested evaluation")

    horizons = root["horizons"]
    if not isinstance(horizons, list):
        raise ValueError(  # noqa: TRY004 - artifact validation contract.
            "horizons must be a list"
        )
    if len(horizons) != len(args.horizons):
        raise ValueError("horizons must contain every requested horizon")
    serialized_sizes: list[int] = []
    for expected_horizon, item in zip(args.horizons, horizons, strict=True):
        horizon = _require_exact_keys(item, _HORIZON_KEYS, name="horizon")
        if horizon["T"] != expected_horizon:
            raise ValueError("horizons are incomplete or out of order")
        loaded = _require_integer(
            horizon["loaded_sequences"],
            name="loaded_sequences",
            minimum=1,
        )
        if loaded <= 0:
            raise ValueError("a passing horizon must load at least one sequence")
        _require_finite_number(horizon["t_mAP"], name="t_mAP")
        _require_finite_number(horizon["t_REC"], name="t_REC")
        per_stage = horizon["per_stage_AP"]
        if not isinstance(per_stage, Mapping) or set(per_stage) != {
            str(stage) for stage in range(1, expected_horizon + 1)
        }:
            raise ValueError("per_stage_AP must contain every stage")
        for value in per_stage.values():
            _require_finite_number(value, name="per_stage_AP")
        for name in (
            "matched_identity_observations",
            "identity_switches",
            "reactivation_events",
            "correct_reactivations",
            "rejected_births",
            "peak_allocated_cuda_bytes",
            "serialized_state_bytes",
        ):
            _require_integer(horizon[name], name=name)
        accuracy = horizon["reactivation_accuracy"]
        if accuracy is not None:
            normalized_accuracy = _require_finite_number(
                accuracy,
                name="reactivation_accuracy",
            )
            if not 0.0 <= normalized_accuracy <= 1.0:
                raise ValueError("reactivation_accuracy must be within [0, 1]")
        for name in ("mean_latency_ms", "throughput_sequences_per_second"):
            if _require_finite_number(horizon[name], name=name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        serialized_sizes.append(int(horizon["serialized_state_bytes"]))

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


def _render_markdown(artifact: Mapping[str, object]) -> str:
    lines = [
        "# Persist4D P5 Evaluation",
        "",
        f"Status: `{artifact['status']}`",
        "",
        (
            "| T | Sequences | t-mAP | t-REC | ID switches | Reactivation | "
            "Peak CUDA bytes | Mean latency (ms) | Throughput (seq/s) | "
            "State bytes |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in artifact["horizons"]:
        accuracy = horizon["reactivation_accuracy"]
        rendered_accuracy = "n/a" if accuracy is None else f"{accuracy:.6f}"
        lines.append(
            f"| {horizon['T']} | {horizon['loaded_sequences']} | "
            f"{horizon['t_mAP']:.6f} | {horizon['t_REC']:.6f} | "
            f"{horizon['identity_switches']} | {rendered_accuracy} | "
            f"{horizon['peak_allocated_cuda_bytes']} | "
            f"{horizon['mean_latency_ms']:.6f} | "
            f"{horizon['throughput_sequences_per_second']:.6f} | "
            f"{horizon['serialized_state_bytes']} |"
        )
    bounded = artifact["bounded_state"]
    lines.extend(
        [
            "",
            f"Constant state shape: `{bounded['constant_shape']}`",
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


def _checkpoint_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return "external:rescene4d_checkpoint"
    return f"repo:{relative.as_posix()}"


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
                "model.return_query_features=true",
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
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"required checkpoint does not exist: {_checkpoint_reference(resolved)}"
        )
    return resolved


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


def _evaluate_horizon(
    *,
    system: Any,
    config: Any,
    memory_config: Any,
    horizon: int,
    capacity: int,
    device: torch.device,
) -> tuple[dict[str, object], list[tuple[tuple[str, tuple[int, ...], str], ...]]]:
    import hydra
    from omegaconf import OmegaConf

    from datasets.streaming_sequence import causal_windows
    from models.persistent_memory import PersistentMemory
    from models.streaming_rescene import StreamingReScene

    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    dataset_config.temporal_window = horizon
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    metric = _instantiate_metric(dataset, horizon)
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
        "matched_identity_observations": 0,
        "identity_switches": 0,
        "reactivation_events": 0,
        "correct_reactivations": 0,
    }
    sequence_latency_seconds: list[float] = []
    state_signatures: list[
        tuple[tuple[str, tuple[int, ...], str], ...]
    ] = []
    state_sizes: list[int] = []

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for sequence_index, scan_indices in enumerate(dataset.sequence_indices):
        accumulator: SequenceAccumulator | None = None
        state = None
        gt_ids_by_stage: list[list[int]] = []
        predicted_ids_by_stage: list[list[int]] = []
        stage_point_counts: list[int] = []
        forward_seconds = 0.0

        for global_stage, window in enumerate(causal_windows(scan_indices)):
            data = targets = output = next_state = None
            sample = segment_stages = raw_coordinates = None
            class_prob = slot_ids = full_masks = full_target = None
            cpu_class_prob = cpu_slot_ids = None
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

                class_prob = output["pred_logits"][0].softmax(dim=-1)
                slot_ids = output["persistent_slot_ids"][0]
                full_masks = _latest_full_resolution_masks(
                    system,
                    output,
                    targets[0],
                    data,
                    latest_local_stage=latest_local_stage,
                )
                if accumulator is None:
                    accumulator = SequenceAccumulator.empty(
                        capacity=capacity,
                        class_count=class_prob.shape[1],
                    )
                cpu_class_prob = class_prob.detach().cpu()
                cpu_slot_ids = slot_ids.detach().cpu()
                accumulator.add_stage(full_masks, cpu_class_prob, cpu_slot_ids)
                stage_point_counts.append(full_masks.shape[1])
                rejected_births += int(
                    output["persistent_rejected_births"].sum().item()
                )

                full_target = data.target_full[0]
                full_stage_selector = (
                    full_target["temporal_stages"].detach().cpu()
                    == latest_local_stage
                )
                present_gt = full_target["masks"][:, full_stage_selector].any(dim=1)
                valid_predictions = cpu_slot_ids >= 0
                matched_gt, matched_predicted = _match_stage_identities(
                    gt_ids=full_target["ids"][present_gt].detach().cpu(),
                    gt_classes=full_target["labels"][present_gt].detach().cpu(),
                    gt_masks=full_target["masks"][present_gt][
                        :, full_stage_selector
                    ].detach().cpu().bool(),
                    predicted_ids=cpu_slot_ids[valid_predictions],
                    predicted_classes=_foreground_classes(
                        cpu_class_prob,
                        observation_settings["background_class"],
                    )[valid_predictions],
                    predicted_masks=full_masks[valid_predictions],
                )
                gt_ids_by_stage.append(matched_gt)
                predicted_ids_by_stage.append(matched_predicted)

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
                    cpu_class_prob,
                    cpu_slot_ids,
                )

        if accumulator is None or state is None:
            raise RuntimeError("sequence produced no streaming stages")
        diagnostic = identity_diagnostics(gt_ids_by_stage, predicted_ids_by_stage)
        for key in identity_totals:
            identity_totals[key] += int(diagnostic[key])

        full_data = full_targets = None
        try:
            full_sample = dataset[sequence_index]
            full_data, full_targets, names = collate([full_sample])
            if len(full_targets) != 1 or list(names) != [dataset.sequence_names[sequence_index]]:
                raise ValueError("full metric target does not match the sequence")
            prediction = _accumulated_prediction(
                accumulator,
                stage_point_counts,
                background_class=observation_settings["background_class"],
                dataset=dataset,
            )
            target = _metric_target(full_data.target_full[0], dataset)
            if prediction["pred_masks"].shape[0] != target["masks"].shape[1]:
                raise ValueError("accumulated predictions do not align with metric target")
            metric.update([prediction], [target])
        finally:
            del full_data, full_targets
        loaded_sequences += 1
        sequence_latency_seconds.append(forward_seconds)
        del accumulator, state

    if loaded_sequences == 0 or not state_signatures or not state_sizes:
        raise RuntimeError(f"T={horizon} validation dataset produced no sequences")
    metrics = metric.compute()
    t_map = _metric_value(metrics, "val_mean_t-AP")
    t_rec = _metric_value(metrics, "val_mean_t-REC")
    per_stage = {
        str(stage): _metric_value(metrics, f"val_mean_stage{stage}-AP")
        for stage in range(1, horizon + 1)
    }
    total_seconds = sum(sequence_latency_seconds)
    if total_seconds <= 0.0:
        raise RuntimeError("measured evaluation latency must be positive")
    reactivation_events = identity_totals["reactivation_events"]
    correct_reactivations = identity_totals["correct_reactivations"]
    result = {
        "T": horizon,
        "loaded_sequences": loaded_sequences,
        "t_mAP": t_map,
        "t_REC": t_rec,
        "per_stage_AP": per_stage,
        **identity_totals,
        "reactivation_accuracy": (
            correct_reactivations / reactivation_events
            if reactivation_events
            else None
        ),
        "rejected_births": rejected_births,
        "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "mean_latency_ms": 1000.0 * total_seconds / loaded_sequences,
        "throughput_sequences_per_second": loaded_sequences / total_seconds,
        "serialized_state_bytes": max(state_sizes),
    }
    return result, state_signatures


def run_real_evaluation(args: argparse.Namespace) -> dict[str, object]:
    device = _validate_cuda_device(args.device)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    source_commit = git_commit()
    checkpoint_sha256 = sha256_file(checkpoint)

    original_directory = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        config, memory_config = _compose_runtime_config()
        system = _load_system(config, checkpoint, device)
        horizon_results = []
        signatures = []
        for horizon in args.horizons:
            result, horizon_signatures = _evaluate_horizon(
                system=system,
                config=config,
                memory_config=memory_config,
                horizon=horizon,
                capacity=args.capacity,
                device=device,
            )
            horizon_results.append(result)
            signatures.extend(horizon_signatures)
    finally:
        os.chdir(original_directory)

    if git_commit() != source_commit:
        raise RuntimeError("source commit changed during evaluation")
    constant_shape = bool(signatures) and len(set(signatures)) == 1
    if not constant_shape:
        raise RuntimeError("persistent state shape changed across the completed run")
    maximum_state_bytes = max(
        int(horizon["serialized_state_bytes"]) for horizon in horizon_results
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "method": METHOD_NAME,
        "source_commit": source_commit,
        "checkpoint": {
            "ref": _checkpoint_reference(checkpoint),
            "sha256": checkpoint_sha256,
        },
        "settings": {"capacity": args.capacity, "local_window": LOCAL_WINDOW},
        "horizons": horizon_results,
        "bounded_state": {
            "constant_shape": constant_shape,
            "maximum_state_bytes": maximum_state_bytes,
        },
        "errors": [],
    }


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
    try:
        _validate_options(args)
        artifact = dict(selected_runner(args))
        _validate_complete_artifact(artifact, args=args)
        files_to_publish: list[tuple[Path, str]] = []
        if args.markdown is not None:
            files_to_publish.append((args.markdown, _render_markdown(artifact)))
        files_to_publish.append((args.output, _json_text(artifact)))
        _publish_text_files_new(files_to_publish)
    except Exception as error:  # noqa: BLE001 - every runtime fault is persisted.
        try:
            _write_failure_if_possible(args.output, error)
        except Exception as write_error:  # noqa: BLE001 - final CLI fallback.
            print(f"failed to write failure artifact: {write_error}", file=sys.stderr)
        print(f"Persist4D evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
