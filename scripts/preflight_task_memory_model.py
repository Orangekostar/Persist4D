#!/usr/bin/env python3
"""Run the real R1 load and empty-state TaskMemory parity gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scripts.task_memory_contracts import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = (
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
)
CHECKPOINT_BYTES = 754_813_672
PRETRAIN_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
PRETRAIN_BYTES = 433_987_358
SEED = 45
_PERSISTENT_KEYS = {
    "task_memory_lineage",
    "task_memory_read_diagnostics",
    "task_memory_route",
}


class TaskMemoryModelPreflightError(RuntimeError):
    """Raised when the real model adaptation fails a frozen gate."""


def _cpu_snapshot(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if type(value) is list:
        return [_cpu_snapshot(item) for item in value]
    if type(value) is tuple:
        return tuple(_cpu_snapshot(item) for item in value)
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TaskMemoryModelPreflightError(
        f"unsupported parity value at {type(value).__module__}.{type(value).__qualname__}"
    )


def _numeric_difference(left: Tensor, right: Tensor) -> float:
    if left.numel() == 0:
        return 0.0
    if left.is_floating_point() or left.is_complex():
        return float((left - right).abs().max().item())
    return 0.0 if torch.equal(left, right) else math.inf


def compare_tensor_trees(
    expected: object,
    actual: object,
    *,
    tolerance: float,
) -> dict[str, object]:
    """Compare nested raw outputs and return a compact per-array trace."""
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        raise TaskMemoryModelPreflightError("parity tolerance must be finite and nonnegative")
    records: list[dict[str, object]] = []

    def compare(left: object, right: object, path: str) -> None:
        if isinstance(left, Tensor):
            if (
                not isinstance(right, Tensor)
                or left.dtype != right.dtype
                or left.shape != right.shape
            ):
                raise TaskMemoryModelPreflightError(
                    f"parity structure differs at {path}"
                )
            difference = _numeric_difference(left, right)
            records.append(
                {
                    "dtype": str(left.dtype).removeprefix("torch."),
                    "exact": torch.equal(left, right),
                    "max_abs_error": difference,
                    "path": path,
                    "shape": list(left.shape),
                }
            )
            return
        if isinstance(left, np.ndarray):
            if (
                not isinstance(right, np.ndarray)
                or left.dtype != right.dtype
                or left.shape != right.shape
            ):
                raise TaskMemoryModelPreflightError(
                    f"parity structure differs at {path}"
                )
            exact = bool(np.array_equal(left, right))
            difference = (
                0.0
                if exact or left.size == 0
                else float(np.max(np.abs(left.astype(float) - right.astype(float))))
            )
            records.append(
                {
                    "dtype": str(left.dtype),
                    "exact": exact,
                    "max_abs_error": difference,
                    "path": path,
                    "shape": list(left.shape),
                }
            )
            return
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or set(left) != set(right):
                raise TaskMemoryModelPreflightError(
                    f"parity structure differs at {path}"
                )
            for key in sorted(left, key=str):
                compare(left[key], right[key], f"{path}.{key}")
            return
        if type(left) in {list, tuple}:
            if type(right) is not type(left) or len(left) != len(right):
                raise TaskMemoryModelPreflightError(
                    f"parity structure differs at {path}"
                )
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=True)
            ):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if type(left) is not type(right) or left != right:
            raise TaskMemoryModelPreflightError(f"parity value differs at {path}")

    compare(expected, actual, "output")
    max_abs_error = max((float(item["max_abs_error"]) for item in records), default=0.0)
    result = {
        "exact": all(bool(item["exact"]) for item in records),
        "max_abs_error": max_abs_error,
        "tensor_count": len(records),
        "tensors": records,
        "tolerance": float(tolerance),
        "within_tolerance": max_abs_error <= float(tolerance),
    }
    if not result["within_tolerance"]:
        raise TaskMemoryModelPreflightError("prediction exceeds parity tolerance")
    return result


def _portable_payload(payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    if any(marker in encoded for marker in ("/home/", "/mnt/", "file:")):
        raise TaskMemoryModelPreflightError("public preflight contains a local path")


def build_model_preflight_payloads(
    *,
    source_commit: str,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    load_audit: Mapping[str, object],
    samples: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the real gate observations and build self-hashed artifacts."""
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
        or isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise TaskMemoryModelPreflightError("preflight provenance is invalid")
    missing = load_audit.get("missing_keys")
    unexpected = load_audit.get("unexpected_keys")
    loaded = load_audit.get("loaded_key_count")
    if (
        isinstance(missing, (str, bytes))
        or not isinstance(missing, Sequence)
        or not missing
        or any(
            not isinstance(key, str) or not key.startswith("model.task_read.")
            for key in missing
        )
        or unexpected != []
        or isinstance(loaded, bool)
        or not isinstance(loaded, int)
        or loaded <= 0
    ):
        raise TaskMemoryModelPreflightError("strict R1 load audit differs")
    if (
        isinstance(samples, (str, bytes))
        or not isinstance(samples, Sequence)
        or [sample.get("stage") for sample in samples] != ["T1", "T2"]
    ):
        raise TaskMemoryModelPreflightError("parity gate requires fixed T1 and T2")
    for sample in samples:
        if (
            sample.get("query_shape") != [1, 100, 128]
            or sample.get("route_shape") != [1, 100]
            or sample.get("state_shape") != [1, 100, 128]
            or sample.get("read_invocations") != 1
            or sample.get("matched_queries") != 0
            or sample.get("read_output_norm") != 0.0
            or not sample.get("raw_parity", {}).get("within_tolerance")
            or not sample.get("official_parity", {}).get("within_tolerance")
        ):
            raise TaskMemoryModelPreflightError("empty-state parity sample differs")

    load_report: dict[str, object] = {
        "allowed_missing_prefixes": ["model.task_read."],
        "checkpoint": {
            "bytes": checkpoint_bytes,
            "reference": "external:r1_checkpoint",
            "sha256": checkpoint_sha256,
        },
        "loaded_key_count": loaded,
        "missing_keys": list(missing),
        "model_target": "models.Persist4DTaskMemory",
        "schema_version": "task-memory-r1-load-report-v1",
        "source_commit": source_commit,
        "status": "PASS",
        "unexpected_keys": [],
    }
    load_report["content_sha256"] = canonical_json_sha256(load_report)
    shape_trace: dict[str, object] = {
        "checkpoint_sha256": checkpoint_sha256,
        "determinism": {
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "seed": SEED,
            "tf32": False,
            "torch_deterministic_algorithms": True,
        },
        "model_target": "models.Persist4DTaskMemory",
        "read_contract": {
            "attention_branches_per_matched_query": 2,
            "hook": "after_first_complete_hlevel_pass",
            "null_value": "exact_zero",
            "route": "one_entity_slot_or_null",
        },
        "samples": [dict(sample) for sample in samples],
        "schema_version": "task-memory-query-shape-trace-v1",
        "source_commit": source_commit,
        "stages": ["T1", "T2"],
        "status": "PASS",
    }
    shape_trace["content_sha256"] = canonical_json_sha256(shape_trace)
    _portable_payload(load_report)
    _portable_payload(shape_trace)
    return load_report, shape_trace


def _seed_inference() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _run_stage_parity(
    *,
    system: object,
    stage_batch: object,
    class_mapper: object,
    device: torch.device,
    stage_name: str,
    sequence_id: str,
) -> dict[str, object]:
    from models.task_memory_state import TaskMemoryState
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import _frozen_inference_seed
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    data, targets, names = stage_batch.model_batch
    if len(targets) != 1 or len(names) != 1 or len(stage_batch.stage_meta) != 1:
        raise TaskMemoryModelPreflightError("parity stage must have batch size one")
    full_targets = getattr(data, "target_full", None)
    if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
        raise TaskMemoryModelPreflightError("parity stage lacks full target")
    data = _move_data_to_device(data, device)
    target = _move_targets_to_device(targets, device)[0]
    segment_stages = _segment_stages(target)
    latest_stage = int(segment_stages.max().item())
    raw_coordinates = system._process_raw_coordinates(data)
    forward_args = {
        "point2segment": [target["point2segment"]],
        "raw_coordinates": raw_coordinates,
        "is_eval": True,
    }

    system.model.task_memory_enabled = False
    with _frozen_inference_seed(SEED, device), torch.inference_mode():
        disabled_output = system.model(data, **forward_args)
    disabled_prediction = extract_official_task_prediction(
        system=system,
        output=disabled_output,
        target_low_resolution=target,
        target_full_resolution=full_targets[0],
        data=data,
        class_mapper=class_mapper,
        latest_stage_index=latest_stage,
    ).prediction()
    disabled_snapshot = _cpu_snapshot(disabled_output)
    del disabled_output
    gc.collect()
    torch.cuda.empty_cache()

    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=100,
        feature_dim=128,
        class_count=19,
        device=device,
        dtype=system.model.task_read.output_projection.weight.dtype,
        config=system.model.task_memory_config,
    )
    system.model.task_memory_enabled = True
    with _frozen_inference_seed(SEED, device), torch.inference_mode():
        enabled_output = system.model(
            data,
            **forward_args,
            task_state=state,
            stage_meta=stage_batch.stage_meta,
        )
    enabled_prediction = extract_official_task_prediction(
        system=system,
        output=enabled_output,
        target_low_resolution=target,
        target_full_resolution=full_targets[0],
        data=data,
        class_mapper=class_mapper,
        latest_stage_index=latest_stage,
    ).prediction()
    route = enabled_output["task_memory_route"]
    diagnostics = enabled_output["task_memory_read_diagnostics"]
    enabled_base = {
        key: value for key, value in enabled_output.items() if key not in _PERSISTENT_KEYS
    }
    enabled_snapshot = _cpu_snapshot(enabled_base)
    raw_parity = compare_tensor_trees(
        disabled_snapshot, enabled_snapshot, tolerance=0.0
    )
    official_parity = compare_tensor_trees(
        _cpu_snapshot(disabled_prediction),
        _cpu_snapshot(enabled_prediction),
        tolerance=0.0,
    )
    query_shape = list(enabled_snapshot["query_features"].shape)
    matched_queries = int((route.query_to_slot >= 0).sum().item())
    sample = {
        "matched_queries": matched_queries,
        "official_parity": official_parity,
        "query_shape": query_shape,
        "raw_parity": raw_parity,
        "read_invocations": 1,
        "read_output_norm": float(diagnostics["read_output_norm"]),
        "reference_id": stage_batch.stage_meta[0].reference_id,
        "route_shape": list(route.query_to_slot.shape),
        "scan_ids_in_window": list(stage_batch.stage_meta[0].scan_ids_in_window),
        "sequence_id": sequence_id,
        "stage": stage_name,
        "state_shape": list(state.embedding.shape),
    }
    del data, enabled_output, state
    gc.collect()
    torch.cuda.empty_cache()
    return sample


def run_preflight(
    *,
    data_root: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    output_root: Path,
    device_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute the fixed real T1/T2 adaptation gate and write public evidence."""
    import hydra
    from omegaconf import open_dict

    from datasets.task_memory_episode import (
        TaskMemoryEpisodeCollator,
        TaskMemoryEpisodeDataset,
        build_native_episode_masters,
    )
    from models.persist4d_task_memory import strict_load_r1_task_memory
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.preflight_task_memory_episode import (
        _rio_base_dataset,
        _role_by_reference,
        load_reference_by_scene,
    )
    from scripts.r1_downstream_context import (
        compose_r1_runtime_config,
        load_r1_contract,
        validate_external_file_identity,
        validate_r1_checkpoint_payload,
    )
    from scripts.run_task_memory_policy_baseline import (
        _atomic_json,
        _episode_specs,
        _git_head,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from trainer.trainer import InstanceSegmentation

    source_commit = _git_head()
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    pretrained_path = pretrained_path.expanduser().resolve(strict=True)
    validate_external_file_identity(
        checkpoint_path,
        expected_sha256=CHECKPOINT_SHA256,
        expected_bytes=CHECKPOINT_BYTES,
        require_digest_filename=True,
        label="R1 checkpoint",
    )
    validate_external_file_identity(
        pretrained_path,
        expected_sha256=PRETRAIN_SHA256,
        expected_bytes=PRETRAIN_BYTES,
        require_digest_filename=False,
        label="Concerto pretrain",
    )
    try:
        device = torch.device(device_name)
    except (TypeError, RuntimeError) as error:
        raise TaskMemoryModelPreflightError("device is invalid") from error
    if (
        device.type != "cuda"
        or device.index is None
        or not torch.cuda.is_available()
        or device.index >= torch.cuda.device_count()
    ):
        raise TaskMemoryModelPreflightError("preflight requires an available CUDA device")

    runtime_config, _ = compose_r1_runtime_config(pretrained_path)
    with open_dict(runtime_config.model):
        runtime_config.model._target_ = "models.Persist4DTaskMemory"
        runtime_config.model.task_memory_enabled = True
    with redirect_stdout(StringIO()):
        system = InstanceSegmentation(runtime_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = load_r1_contract(
        PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml"
    )
    validate_r1_checkpoint_payload(checkpoint, contract)
    load_audit = strict_load_r1_task_memory(system, checkpoint["state_dict"])
    del checkpoint
    system.to(device).eval()

    data_root = data_root.expanduser().resolve(strict=True)
    metadata_path = metadata_path.expanduser().resolve(strict=True)
    data_contract = json.loads(
        (PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    base = _rio_base_dataset(runtime_config, data_root=data_root, horizon=5)
    masters = tuple(
        master
        for master in build_native_episode_masters(
            base,
            reference_by_scene=load_reference_by_scene(metadata_path),
            role_by_reference=_role_by_reference(data_contract),
        )
        if master.role == "development" and len(master.scan_ids) == 5
    )
    if len(masters) != 47:
        raise TaskMemoryModelPreflightError("development H5 population differs")
    spec = _episode_specs(masters)[0]
    episode = TaskMemoryEpisodeDataset(base, (spec,), apply_augmentation=False)[0]
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(runtime_config.data.validation_collation)
    )
    batch = collator([episode])
    class_mapper = build_rio_class_mapper(base)
    samples = []
    with deterministic_inference_runtime(SEED, device):
        for stage_index, stage_name in enumerate(("T1", "T2")):
            _seed_inference()
            samples.append(
                _run_stage_parity(
                    system=system,
                    stage_batch=batch.stage_batches[stage_index],
                    class_mapper=class_mapper,
                    device=device,
                    stage_name=stage_name,
                    sequence_id=spec.source_sequence_id,
                )
            )
    load_report, shape_trace = build_model_preflight_payloads(
        source_commit=source_commit,
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_bytes=CHECKPOINT_BYTES,
        load_audit=load_audit,
        samples=samples,
    )
    output_root = output_root.expanduser().resolve()
    _atomic_json(output_root / "r1_load_report.json", load_report)
    _atomic_json(output_root / "query_shape_trace.json", shape_trace)
    return load_report, shape_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rio-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task_memory_retention_v2/implementation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_report, shape_trace = run_preflight(
        data_root=args.data_root,
        metadata_path=args.rio_metadata,
        checkpoint_path=args.checkpoint,
        pretrained_path=args.pretrained,
        output_root=args.output_root,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "load_report_sha256": load_report["content_sha256"],
                "query_shape_trace_sha256": shape_trace["content_sha256"],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TaskMemoryModelPreflightError",
    "build_model_preflight_payloads",
    "compare_tensor_trees",
    "run_preflight",
]
