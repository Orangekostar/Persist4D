#!/usr/bin/env python3
"""Run the resource-only and functional Sonata SSMOKE gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sonata_second_training import require_formal_authorization
from scripts.sonata_second_preflight import _compose_config
from utils.sonata_second_preflight import (
    SonataSecondPreflightError,
    build_sonata_source_tree_contract,
    canonical_sha256,
    file_sha256,
    validate_formal_resource_blocker,
)
from utils.sonata_weight_provenance import OFFICIAL_SONATA_WEIGHT_SPEC

PREFLIGHT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "preflight"
)
SMOKE_DIR = PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "smoke"
RESOURCE_BLOCKER_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "sonata_second_perception_v1"
    / "training"
    / "RESOURCE_BLOCKER.json"
)
TINY_T2_SAMPLE = "scene0112_00-scene0112_01"
RESOURCE_SAMPLE_INDICES = {"rio": 359, "scannet": 921}
MICROBATCH_CANDIDATES = (1, 2, 4, 8, 16)
TARGET_EFFECTIVE_BATCH = 32
SAFE_MEMORY_FRACTION = 0.90
TINY_OPTIMIZATION_STEPS = 4

BATCH_FIELDS = (
    "schema_version",
    "probe_scope",
    "microbatch_per_gpu",
    "gpu_count",
    "physical_global_batch",
    "status",
    "finite_loss",
    "finite_gradients",
    "loss",
    "peak_allocated_vram_mib",
    "peak_reserved_vram_mib",
    "memory_total_mib",
    "safe_headroom",
    "step_latency_seconds",
    "samples_per_second",
    "raw_points",
    "voxels",
    "validation_accuracy_inspected",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _seed(seed: int = 45) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _artifact_privacy(text: str) -> None:
    forbidden = (
        "/home/",
        "/mnt/shared/",
        "192.168.",
        "node107",
        "ssh://",
    )
    found = [token for token in forbidden if token in text]
    if found:
        raise ValueError("artifact contains private runtime identifiers")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True
    ) + "\n"
    _artifact_privacy(serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    _artifact_privacy(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SonataSecondPreflightError(f"{label} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise SonataSecondPreflightError(f"{label} must contain an object")
    return payload


def _name_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def validate_temporal_batch_contract(
    data: Any,
    targets: Sequence[Mapping[str, Any]],
    *,
    names: Sequence[str],
    expected_stage_counts: Sequence[int],
) -> dict[str, Any]:
    features = getattr(data, "features", None)
    coordinates = getattr(data, "coordinates", None)
    data_stages = getattr(data, "temporal_stages", None)
    if not isinstance(features, Tensor) or features.ndim != 2:
        raise ValueError("Sonata features must be a rank-2 tensor")
    if features.shape[1] != 9:
        raise ValueError("Sonata features must contain coord/color/normal")
    if not torch.isfinite(features).all().item():
        raise ValueError("Sonata coord/color/normal features must be finite")
    normals = features[:, 6:9]
    if normals.shape[1] != 3 or not torch.isfinite(normals).all().item():
        raise ValueError("Sonata normal features must be finite and three-dimensional")
    if (
        not isinstance(coordinates, Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 5
    ):
        raise ValueError("Sonata coordinates must have [batch,x,y,z,t]")
    if not isinstance(data_stages, (list, tuple)):
        raise TypeError("Sonata temporal stages must be split by true scene")
    if not (
        len(targets)
        == len(names)
        == len(expected_stage_counts)
        == len(data_stages)
    ):
        raise ValueError("Sonata temporal batch cardinalities differ")

    stage_values: list[list[int]] = []
    for index, (stages, target, expected_count) in enumerate(
        zip(data_stages, targets, expected_stage_counts)
    ):
        target_stages = target.get("temporal_stages")
        if not isinstance(stages, Tensor) or not isinstance(target_stages, Tensor):
            raise TypeError("Sonata temporal stages must be tensors")
        observed = sorted(int(value) for value in torch.unique(stages).tolist())
        target_values = sorted(
            int(value) for value in torch.unique(target_stages).tolist()
        )
        expected = list(range(int(expected_count)))
        if observed != expected or target_values != expected:
            raise ValueError(
                "future stage leakage or missing stage in the bounded training window"
            )
        scene_coordinates = coordinates[coordinates[:, 0].long() == index]
        coordinate_values = sorted(
            int(value) for value in torch.unique(scene_coordinates[:, -1]).tolist()
        )
        if coordinate_values != expected:
            raise ValueError("coordinate temporal stages differ from scene stages")
        point2segment = target.get("point2segment")
        if not isinstance(point2segment, Tensor) or (
            point2segment.numel() != target_stages.numel()
        ):
            raise ValueError("target temporal stages do not align to supervised points")
        stage_values.append(observed)

    return {
        "batch_size": len(targets),
        "sample_names": [str(name) for name in names],
        "feature_dimension": int(features.shape[1]),
        "normal_dimension": int(normals.shape[1]),
        "features_finite": True,
        "normals_finite": True,
        "stage_values": stage_values,
        "future_stage_leakage": False,
        "future_stage_definition": (
            "no target or coordinate stage outside each bounded collated window"
        ),
    }


def classify_sonata_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "frozen_encoder_embedding": [],
        "trainable_sonata_decoder": [],
        "trainable_rescene_decoder": [],
        "trainable_rescene_heads": [],
    }
    head_prefixes = (
        "model.mask_features_head.",
        "model.query_projection.",
        "model.np_feature_projection.",
        "model.mask_embed_head.",
        "model.class_embed_head.",
        "model.change_embed_head.",
    )
    for name, parameter in named_parameters:
        if name.startswith(
            ("model.backbone.model.enc.", "model.backbone.model.embedding.")
        ):
            groups["frozen_encoder_embedding"].append(name)
        elif name.startswith("model.backbone.model.dec.") and parameter.requires_grad:
            groups["trainable_sonata_decoder"].append(name)
        elif name.startswith(head_prefixes) and parameter.requires_grad:
            groups["trainable_rescene_heads"].append(name)
        elif (
            name.startswith("model.")
            and not name.startswith("model.backbone.")
            and parameter.requires_grad
        ):
            groups["trainable_rescene_decoder"].append(name)
    return groups


def summarize_gradients(
    parameters: Mapping[str, Tensor], names: Sequence[str]
) -> dict[str, Any]:
    norms: list[float] = []
    finite = True
    missing = 0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            missing += 1
            continue
        finite = finite and bool(torch.isfinite(gradient).all().item())
        norms.append(float(gradient.detach().float().norm().cpu()))
    return {
        "parameter_tensors": len(names),
        "parameter_name_sha256": _name_sha256(names),
        "missing_grad_tensors": missing,
        "finite": finite,
        "nonzero_grad_tensors": sum(value > 0 for value in norms),
        "maximum_grad_norm": max(norms, default=0.0),
    }


def select_batch_configuration(
    records: Sequence[Mapping[str, Any]],
    *,
    gpu_count: int,
    target_effective_batch: int = TARGET_EFFECTIVE_BATCH,
) -> dict[str, Any]:
    stable: list[Mapping[str, Any]] = []
    for record in records:
        microbatch = int(record["microbatch_per_gpu"])
        physical = microbatch * gpu_count
        memory_total = float(record["memory_total_mib"])
        peak = float(
            record.get("peak_reserved_vram_mib", record["peak_vram_mib"])
        )
        safe_headroom = peak <= memory_total * SAFE_MEMORY_FRACTION
        if (
            record.get("status") == "stable"
            and record.get("finite_loss") is True
            and record.get("finite_gradients") is True
            and safe_headroom
            and target_effective_batch % physical == 0
        ):
            stable.append(record)
    if not stable:
        raise ValueError("no stable physical batch divides the effective batch target")
    selected = max(
        stable,
        key=lambda item: (
            int(item["microbatch_per_gpu"]),
            float(item["samples_per_second"]),
        ),
    )
    microbatch = int(selected["microbatch_per_gpu"])
    physical = microbatch * gpu_count
    return {
        "gpu_count": gpu_count,
        "microbatch_per_gpu": microbatch,
        "physical_global_batch": physical,
        "accumulate_grad_batches": target_effective_batch // physical,
        "effective_global_batch": target_effective_batch,
        "selection_uses_validation_accuracy": False,
    }


def apply_formal_resource_blocker(
    records: Sequence[Mapping[str, Any]], blocker: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Supersede a short p95 probe with the stronger full-loader OOM evidence."""

    evidence = validate_formal_resource_blocker(blocker)
    failed_microbatch = evidence["failed_microbatch_per_gpu"]
    updated: list[dict[str, Any]] = []
    found = False
    for record in records:
        item = dict(record)
        if item.get("microbatch_per_gpu") == failed_microbatch:
            found = True
            item.update(
                {
                    "probe_scope": (
                        f"{item.get('probe_scope', 'resource_probe')}"
                        "+formal_epoch0_replay"
                    ),
                    "status": "oom_observed_formal_training",
                    "finite_loss": False,
                    "finite_gradients": False,
                    "loss": "",
                    "safe_headroom": False,
                }
            )
        updated.append(item)
    if not found:
        raise SonataSecondPreflightError(
            "resource probe omits the formally failed microbatch"
        )
    return updated, evidence


def validate_query_interface(
    output: Mapping[str, Any], *, expected_batch_size: int
) -> dict[str, Any]:
    logits = output.get("pred_logits")
    masks = output.get("pred_masks")
    queries = output.get("query_features")
    backbone = output.get("backbone_features")
    aux_outputs = output.get("aux_outputs")
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or logits.shape[0] != expected_batch_size
        or logits.shape[1] != 100
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("ReScene class output interface differs")
    if not isinstance(masks, list) or len(masks) != expected_batch_size:
        raise ValueError("ReScene mask output interface differs")
    if any(
        not isinstance(mask, Tensor)
        or mask.ndim != 2
        or mask.shape[1] != 100
        or not torch.isfinite(mask).all().item()
        for mask in masks
    ):
        raise ValueError("ReScene mask tensors differ")
    if (
        not isinstance(queries, Tensor)
        or list(queries.shape) != [expected_batch_size, 100, 128]
        or not torch.isfinite(queries).all().item()
    ):
        raise ValueError("Persist4D query feature interface differs")
    backbone_features = getattr(backbone, "F", None)
    if (
        not isinstance(backbone_features, Tensor)
        or backbone_features.ndim != 2
        or not torch.isfinite(backbone_features).all().item()
    ):
        raise ValueError("Sonata backbone output differs")
    if not isinstance(aux_outputs, list) or not aux_outputs:
        raise ValueError("ReScene auxiliary output hierarchy is empty")
    return {
        "pred_logits_shape": list(logits.shape),
        "mask_shapes": [list(mask.shape) for mask in masks],
        "query_feature_shape": list(queries.shape),
        "query_feature_dimension": int(queries.shape[-1]),
        "backbone_feature_shape": list(backbone_features.shape),
        "aux_output_count": len(aux_outputs),
        "all_outputs_finite": True,
        "official_output_extraction_compatible": True,
        "persist4d_observation_interface_compatible": True,
    }


def build_load_interface_contract(
    load_audit: Mapping[str, Any], *, decoder_parameter_tensor_count: int
) -> dict[str, Any]:
    loaded_encoder = load_audit.get("loaded_encoder_key_count")
    missing_decoder = load_audit.get("expected_decoder_missing_key_count")
    unexpected = load_audit.get("unexpected_keys")
    if (
        load_audit.get("gate") != "SW0-PASS"
        or not isinstance(loaded_encoder, int)
        or isinstance(loaded_encoder, bool)
        or loaded_encoder <= 0
        or not isinstance(missing_decoder, int)
        or isinstance(missing_decoder, bool)
        or missing_decoder <= 0
        or not isinstance(unexpected, list)
        or unexpected
        or decoder_parameter_tensor_count != missing_decoder
    ):
        raise ValueError("SS1 load audit and runtime decoder interface differ")
    return {
        "verified_weight_sha256": OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
        "load_gate": "SW0-PASS",
        "loaded_encoder_key_count": loaded_encoder,
        "missing_decoder_key_count": missing_decoder,
        "unexpected_key_count": len(unexpected),
        "decoder_parameter_tensor_count": decoder_parameter_tensor_count,
        "decoder_initialized": True,
    }


def validate_tiny_optimization(history: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in history]
    if len(values) < 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("tiny optimization requires at least three finite objectives")
    minimum = min(values[1:])
    if not minimum < values[0]:
        raise ValueError("tiny optimization did not decrease the objective")
    return {
        "steps": len(values),
        "initial_objective": values[0],
        "final_objective": values[-1],
        "minimum_after_initial": minimum,
        "minimum_over_initial_ratio": minimum / values[0],
        "passed": True,
    }


def _move_data_to_device(data: Any, device: torch.device) -> Any:
    for key in list(data.keys()):
        if isinstance(data[key], Tensor):
            data[key] = data[key].to(device, non_blocking=True)
    return data


def _move_targets_to_device(
    targets: Sequence[Mapping[str, Any]], device: torch.device
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


def _build_system(
    verified_weight: Path,
    training_output_dir: Path,
    device: torch.device,
    *,
    return_query_features: bool,
) -> tuple[Any, Any]:
    from omegaconf import open_dict

    from trainer.trainer import InstanceSegmentation

    cfg = _compose_config(verified_weight, training_output_dir)
    with open_dict(cfg):
        cfg.general.gpus = 1
        cfg.general.compile_model = False
        cfg.data.num_workers = 0
        cfg.model.return_query_features = return_query_features
    _seed()
    system = InstanceSegmentation(cfg).to(device)
    if not system.model.backbone.model.__class__.__module__.startswith("sonata"):
        raise RuntimeError("smoke runtime did not instantiate installed Sonata")
    return cfg, system


def _mixed_runtime(cfg: Any) -> tuple[Any, Any]:
    import hydra

    dataset = hydra.utils.instantiate(cfg.data.train_dataset)
    collator = hydra.utils.instantiate(cfg.data.train_collation)
    names = [child.dataset_name for child in dataset.datasets]
    if names != ["rio", "scannet"]:
        raise RuntimeError("resource probe mixed dataset order differs")
    return dataset, collator


def _materialize_resource_batch(
    dataset: Any,
    collator: Any,
    microbatch: int,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    samples = []
    sample_names = []
    expected_stage_counts = []
    raw_points = 0
    for batch_index in range(microbatch):
        child_index = batch_index % len(dataset.datasets)
        child = dataset.datasets[child_index]
        sample_index = RESOURCE_SAMPLE_INDICES[child.dataset_name]
        _seed(45 + batch_index)
        sample = child[sample_index]
        samples.append(sample)
        sample_names.append(str(child.sequence_names[sample_index]))
        expected_stage_counts.append(int(child.temporal_window))
        raw_points += int(sample[0].shape[0])
    _seed()
    data, targets, names = collator(samples)
    if list(names) != sample_names or len(targets) != microbatch:
        raise RuntimeError("resource probe collator changed fixed sample order")
    temporal = validate_temporal_batch_contract(
        data,
        targets,
        names=sample_names,
        expected_stage_counts=expected_stage_counts,
    )
    voxels = int(data.features.shape[0])
    data = _move_data_to_device(data, device)
    targets = _move_targets_to_device(targets, device)
    return data, targets, {
        "sample_names": sample_names,
        "expected_stage_counts": expected_stage_counts,
        "raw_points": raw_points,
        "voxels": voxels,
        "temporal_contract": temporal,
    }


def _forward_objective(
    system: Any, data: Any, targets: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], Mapping[str, Tensor], Tensor]:
    from trainer.trainer import aggregate_objective_loss

    output = system.forward(
        data,
        point2segment=[target["point2segment"] for target in targets],
        raw_coordinates=system._process_raw_coordinates(data),
        is_eval=False,
        targets=targets,
    )
    losses = system.criterion(output, targets, mask_type=system.mask_type)
    objective = aggregate_objective_loss(losses, system.criterion.weight_dict)
    return output, losses, objective


def _trainable_gradients_finite(system: Any) -> tuple[bool, bool]:
    gradients = [
        parameter.grad
        for parameter in system.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return (
        bool(gradients)
        and all(bool(torch.isfinite(value).all().item()) for value in gradients),
        any(bool(torch.count_nonzero(value).item()) for value in gradients),
    )


def _probe_batches(
    verified_weight: Path,
    training_output_dir: Path,
    device: torch.device,
    *,
    gpu_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg, system = _build_system(
        verified_weight,
        training_output_dir,
        device,
        return_query_features=False,
    )
    system.train()
    dataset, collator = _mixed_runtime(cfg)
    memory_total = float(torch.cuda.get_device_properties(device).total_memory / 1024**2)
    records: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for microbatch in MICROBATCH_CANDIDATES:
        physical = microbatch * gpu_count
        if TARGET_EFFECTIVE_BATCH % physical:
            continue
        system.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        data = targets = output = losses = objective = None
        metadata: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            data, targets, metadata = _materialize_resource_batch(
                dataset, collator, microbatch, device
            )
            compute_started = time.perf_counter()
            output, losses, objective = _forward_objective(system, data, targets)
            objective.backward()
            torch.cuda.synchronize(device)
            compute_elapsed = time.perf_counter() - compute_started
            finite_loss = bool(torch.isfinite(objective).all().item())
            finite_gradients, nonzero_gradients = _trainable_gradients_finite(system)
            peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
            safe_headroom = peak_reserved <= memory_total * SAFE_MEMORY_FRACTION
            status = (
                "stable"
                if finite_loss
                and finite_gradients
                and nonzero_gradients
                and safe_headroom
                else "unstable"
            )
            record = {
                "schema_version": 1,
                "probe_scope": "single_gpu_replica_real_p95_mixed_batch",
                "microbatch_per_gpu": microbatch,
                "gpu_count": gpu_count,
                "physical_global_batch": physical,
                "status": status,
                "finite_loss": finite_loss,
                "finite_gradients": finite_gradients and nonzero_gradients,
                "loss": float(objective.detach().cpu()),
                "peak_allocated_vram_mib": peak_allocated,
                "peak_reserved_vram_mib": peak_reserved,
                "peak_vram_mib": peak_reserved,
                "memory_total_mib": memory_total,
                "safe_headroom": safe_headroom,
                "step_latency_seconds": compute_elapsed,
                "samples_per_second": microbatch / compute_elapsed,
                "raw_points": metadata["raw_points"],
                "voxels": metadata["voxels"],
                "validation_accuracy_inspected": False,
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            torch.cuda.synchronize(device)
            record = {
                "schema_version": 1,
                "probe_scope": "single_gpu_replica_real_p95_mixed_batch",
                "microbatch_per_gpu": microbatch,
                "gpu_count": gpu_count,
                "physical_global_batch": physical,
                "status": "oom",
                "finite_loss": False,
                "finite_gradients": False,
                "loss": "",
                "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device)
                / 1024**2,
                "peak_reserved_vram_mib": torch.cuda.max_memory_reserved(device)
                / 1024**2,
                "peak_vram_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
                "memory_total_mib": memory_total,
                "safe_headroom": False,
                "step_latency_seconds": time.perf_counter() - started,
                "samples_per_second": 0.0,
                "raw_points": metadata.get("raw_points", ""),
                "voxels": metadata.get("voxels", ""),
                "validation_accuracy_inspected": False,
            }
        records.append(record)
        details[str(microbatch)] = {
            "sample_names": metadata.get("sample_names", []),
            "expected_stage_counts": metadata.get("expected_stage_counts", []),
        }
        del data, targets, output, losses, objective
        system.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if record["status"] != "stable":
            break
    del system, dataset, collator
    torch.cuda.empty_cache()
    return records, details


def _materialize_tiny_t2_batch(
    cfg: Any, device: torch.device
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    dataset, collator = _mixed_runtime(cfg)
    rio = dataset.datasets[0]
    try:
        sample_index = rio.sequence_names.index(TINY_T2_SAMPLE)
    except ValueError as error:
        raise RuntimeError("fixed supervised T=2 sample is absent") from error
    _seed()
    sample = rio[sample_index]
    _seed()
    data, targets, names = collator([sample])
    temporal = validate_temporal_batch_contract(
        data,
        targets,
        names=list(names),
        expected_stage_counts=[2],
    )
    metadata = {
        "dataset": "3RScan",
        "sample_name": TINY_T2_SAMPLE,
        "sample_reference": "repo:data/processed/rio/train_database.yaml",
        "raw_points": int(sample[0].shape[0]),
        "voxels": int(data.features.shape[0]),
        "supervised_instances": int(targets[0]["labels"].numel()),
        "temporal": temporal,
    }
    return (
        _move_data_to_device(data, device),
        _move_targets_to_device(targets, device),
        metadata,
    )


def _monitor_temporal_paths(system: Any) -> tuple[dict[str, int], dict[str, Any]]:
    backbone = system.model.backbone
    counters = {"temporal_overlay": 0, "temporal_pool_merge": 0}
    hierarchy: dict[str, Any] = {}
    serializations = []
    for serialization in backbone.decoder_serializations:
        if getattr(serialization, "__name__", "") != "temporal_overlay":
            serializations.append(serialization)
            continue

        def overlay(point: Any, original: Any = serialization) -> Any:
            counters["temporal_overlay"] += 1
            return original(point)

        serializations.append(overlay)
    backbone.decoder_serializations = serializations

    original_pool = backbone.temporal_pool_merge

    def temporal_pool(point: Any, *args: Any, **kwargs: Any) -> Any:
        counters["temporal_pool_merge"] += 1
        return original_pool(point, *args, **kwargs)

    backbone.temporal_pool_merge = temporal_pool
    original_forward = backbone.forward

    def monitored_forward(data: Any) -> Any:
        point, aux, coords = original_forward(data)
        hierarchy["level_count"] = len(aux)
        hierarchy["feature_dimensions"] = [int(item.F.shape[1]) for item in aux]
        hierarchy["point_counts"] = [int(item.F.shape[0]) for item in aux]
        hierarchy["coordinate_group_counts"] = [len(item) for item in coords]
        hierarchy["all_features_finite"] = all(
            bool(torch.isfinite(item.F).all().item()) for item in aux
        )
        return point, aux, coords

    backbone.forward = monitored_forward
    return counters, hierarchy


def _official_extraction(system: Any, output: Mapping[str, Any]) -> dict[str, Any]:
    predictions = system._get_predictions(output)
    selected = predictions[system.decoder_id]
    class_probabilities = selected["pred_logits"][0]
    mask_logits = selected["pred_masks"][0]
    scores, masks, classes, heatmap = system._get_mask_and_scores(
        class_probabilities,
        mask_logits,
        num_queries=100,
        num_classes=18,
        device=class_probabilities.device,
    )
    tensors = (scores, masks, classes, heatmap)
    if not all(isinstance(value, Tensor) for value in tensors):
        raise RuntimeError("official output extraction did not return tensors")
    if not torch.isfinite(scores).all().item() or not torch.isfinite(heatmap).all().item():
        raise RuntimeError("official output extraction returned non-finite values")
    if masks.shape[1] != scores.numel() or classes.shape != scores.shape:
        raise RuntimeError("official output extraction shapes differ")
    return {
        "candidate_count": int(scores.numel()),
        "mask_shape": list(masks.shape),
        "score_shape": list(scores.shape),
        "class_shape": list(classes.shape),
        "heatmap_shape": list(heatmap.shape),
        "finite": True,
    }


def _functional_smoke(
    verified_weight: Path,
    training_output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import hydra
    from omegaconf import OmegaConf

    cfg, system = _build_system(
        verified_weight,
        training_output_dir,
        device,
        return_query_features=True,
    )
    system.train()
    data, targets, sample = _materialize_tiny_t2_batch(cfg, device)
    counters, hierarchy = _monitor_temporal_paths(system)
    named_parameters = dict(system.named_parameters())
    groups = classify_sonata_parameters(named_parameters.items())
    if not all(groups.values()):
        raise RuntimeError("Sonata freeze/gradient parameter groups are incomplete")
    if any(
        named_parameters[name].requires_grad
        for name in groups["frozen_encoder_embedding"]
    ):
        raise RuntimeError("Sonata encoder or embedding is not frozen")
    trainable_groups = (
        "trainable_sonata_decoder",
        "trainable_rescene_decoder",
        "trainable_rescene_heads",
    )
    if any(
        not named_parameters[name].requires_grad
        for group in trainable_groups
        for name in groups[group]
    ):
        raise RuntimeError("Sonata decoder or ReScene task parameter is frozen")
    if bool(system.criterion.use_contrastive_loss) or hasattr(
        system.criterion, "contrastive_loss"
    ):
        raise RuntimeError("contrastive loss path is active")

    trainable = [parameter for parameter in system.parameters() if parameter.requires_grad]
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=trainable)
    scheduler_config = OmegaConf.create(
        OmegaConf.to_container(cfg.scheduler.scheduler, resolve=True)
    )
    scheduler_config.total_steps = 8
    scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
    history: list[float] = []
    first_losses: dict[str, float] = {}
    gradient_summaries: dict[str, Any] = {}
    interface: dict[str, Any] = {}
    extraction: dict[str, Any] = {}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for step in range(TINY_OPTIMIZATION_STEPS):
        _seed(45)
        optimizer.zero_grad(set_to_none=True)
        output, losses, objective = _forward_objective(system, data, targets)
        loss_keys = sorted(losses)
        if any("contrastive" in name for name in loss_keys):
            raise RuntimeError("contrastive loss term contributes to the objective")
        if not torch.isfinite(objective).all().item():
            raise RuntimeError("tiny optimization objective is non-finite")
        objective.backward()
        if step == 0:
            gradient_summaries = {
                group: summarize_gradients(named_parameters, names)
                for group, names in groups.items()
            }
            if gradient_summaries["frozen_encoder_embedding"][
                "nonzero_grad_tensors"
            ]:
                raise RuntimeError("frozen Sonata encoder received gradients")
            for group in trainable_groups:
                summary = gradient_summaries[group]
                if not summary["finite"] or not summary["nonzero_grad_tensors"]:
                    raise RuntimeError(f"{group} lacks finite nonzero gradients")
            interface = validate_query_interface(output, expected_batch_size=1)
            extraction = _official_extraction(system, output)
            first_losses = {
                name: float(value.detach().cpu())
                for name, value in sorted(losses.items())
            }
        optimizer.step()
        scheduler.step()
        history.append(float(objective.detach().cpu()))
        del output, losses, objective
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    optimization = validate_tiny_optimization(history)
    if counters["temporal_overlay"] <= 0 or counters["temporal_pool_merge"] <= 0:
        raise RuntimeError("required temporal sharing path was not executed")
    if (
        hierarchy.get("level_count", 0) < 4
        or not hierarchy.get("all_features_finite", False)
    ):
        raise RuntimeError("Sonata output feature hierarchy is invalid")

    load_audit = _load_json(
        PROJECT_ROOT
        / "artifacts"
        / "sonata_second_perception_v1"
        / "weight"
        / "sonata_load_key_audit.json",
        label="Sonata load-key audit",
    )
    decoder_parameters = groups["trainable_sonata_decoder"]
    gradient_contract = {
        "schema_version": 1,
        "status": "pass",
        "freeze_mode": "backbone_encoder",
        "frozen_encoder_eval": False,
        "groups": {
            group: {
                "parameter_names": names,
                **gradient_summaries[group],
            }
            for group, names in groups.items()
        },
        "encoder_embedding_requires_grad_false": True,
        "decoder_task_requires_grad_true": True,
        "frozen_encoder_gradients_absent": True,
        "decoder_head_gradients_finite_nonzero": True,
    }
    query_interface = {
        "schema_version": 1,
        "status": "pass",
        **interface,
        "official_extraction": extraction,
        "output_feature_hierarchy": hierarchy,
    }
    result = {
        "sample": sample,
        "load_interface": build_load_interface_contract(
            load_audit,
            decoder_parameter_tensor_count=len(decoder_parameters),
        ),
        "temporal_execution": {
            **counters,
            "decoder_serializations": ["standard", "temporal_overlay"],
            "temporal_masking": True,
            "contrastive_loss_active": False,
            "contrastive_loss_terms": [],
        },
        "objective": {
            "weighted": True,
            "class_mask_dice_weights": [2.0, 5.0, 2.0],
            "eos_coef": 0.2,
            "first_step_components": first_losses,
            "history": history,
            "tiny_optimization": optimization,
        },
        "elapsed_seconds": elapsed,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del system, data, targets, optimizer, scheduler
    torch.cuda.empty_cache()
    return result, gradient_contract, query_interface


def _hardware_contract(devices: Sequence[int]) -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    inventory = {}
    for line in query.splitlines():
        index, name, memory, driver = [value.strip() for value in line.split(",")]
        inventory[int(index)] = {
            "device_alias": f"device-{index}",
            "model": name,
            "memory_total_mib": int(memory),
            "driver_version": driver,
        }
    if any(device not in inventory for device in devices):
        raise RuntimeError("selected GPU is not visible")
    topology_text = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    topology_text = re.sub(r"\x1b\[[0-9;]*m", "", topology_text)
    rows = {
        int(parts[0][3:]): parts
        for line in topology_text.splitlines()
        if (parts := line.split()) and re.fullmatch(r"GPU\d+", parts[0])
    }
    links = []
    for left_index, left in enumerate(devices):
        for right in devices[left_index + 1 :]:
            links.append(
                {
                    "left": f"device-{left}",
                    "right": f"device-{right}",
                    "interconnect": rows[left][right + 1],
                }
            )
    nccl_version = torch.cuda.nccl.version()
    return {
        "visible_gpu_count": len(inventory),
        "selected_gpu_count": len(devices),
        "selected_devices": [f"device-{device}" for device in devices],
        "same_node": True,
        "devices": [inventory[device] for device in devices],
        "links": links,
        "selected_same_numa": len(devices) == 2
        and rows[devices[0]][-2] == rows[devices[1]][-2],
        "selected_numa_affinity": rows[devices[0]][-2],
        "selected_cpu_affinity": rows[devices[0]][-3],
        "nccl": {
            "version": ".".join(str(value) for value in nccl_version),
            "transport_selection": "automatic_same_node",
            "environment_overrides": {
                key: os.environ.get(key)
                for key in (
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                    "NCCL_IB_DISABLE",
                    "NCCL_SOCKET_IFNAME",
                )
            },
        },
    }


def _write_batch_artifacts(
    records: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    hardware: Mapping[str, Any],
    details: Mapping[str, Any],
) -> None:
    csv_path = PREFLIGHT_DIR / "batch_feasibility.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=BATCH_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    content = csv_path.read_text(encoding="utf-8")
    _artifact_privacy(content)
    topology = hardware["links"][0]["interconnect"]
    lines = [
            "# Sonata Batch Selection",
            "",
            "- Gate scope: resource feasibility only; validation accuracy was not inspected.",
            f"- Hardware: {selection['gpu_count']} x {hardware['devices'][0]['model']}",
            f"- Same node / same NUMA: yes / {'yes' if hardware['selected_same_numa'] else 'no'}",
            f"- Interconnect: `{topology}`",
            f"- Selected microbatch per GPU: {selection['microbatch_per_gpu']}",
            f"- Selected physical global batch: {selection['physical_global_batch']}",
            f"- Gradient accumulation: {selection['accumulate_grad_batches']}",
            f"- Effective global batch: {selection['effective_global_batch']}",
            "- Probe: one real forward/backward on fixed approximately 95th-percentile mixed-data samples.",
            "- Interpretation: this preserves the effective batch target but does not claim unpublished official physical-batch equivalence.",
    ]
    blocker = details.get("formal_resource_blocker")
    if isinstance(blocker, Mapping):
        lines.extend(
            [
                f"- Formal replay gate: `{blocker['gate']}`.",
                "- The p95 microbatch-4 result is superseded by two deterministic full-loader epoch-0 OOM replays.",
                f"- Resource blocker SHA-256: `{blocker['resource_blocker_sha256']}`.",
            ]
        )
    lines.extend(
        [
            "",
            f"Candidate sample bindings: `{canonical_sha256(details)}`.",
            "",
        ]
    )
    markdown = "\n".join(lines)
    _write_text(PREFLIGHT_DIR / "BATCH_SELECTION.md", markdown)


def _issue_smoke_results(
    *,
    result: Mapping[str, Any],
    gradient_path: Path,
    query_path: Path,
    batch_path: Path,
    hardware: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    source = build_sonata_source_tree_contract(PROJECT_ROOT, require_clean=True)
    preflight_authorization_path = PREFLIGHT_DIR / "preflight_authorization.json"
    authorization = _load_json(
        preflight_authorization_path, label="SP0 authorization"
    )
    bindings = {
        "source_commit": source["source_commit"],
        "source_tree_sha256": source["content_sha256"],
        "sp0_authorization_sha256": file_sha256(preflight_authorization_path),
        "sp0_authorization_payload_sha256": authorization[
            "authorization_sha256"
        ],
        "config_sha256": authorization["bindings"]["config_sha256"],
        "data_manifest_sha256": authorization["bindings"]["data_manifest_sha256"],
        "weight_sha256": OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
        "batch_feasibility_sha256": file_sha256(batch_path),
        "gradient_contract_sha256": file_sha256(gradient_path),
        "query_interface_sha256": file_sha256(query_path),
        "resource_blocker_sha256": file_sha256(RESOURCE_BLOCKER_PATH),
    }
    payload = {
        "schema_version": 1,
        "status": "pass",
        "gate": "SSMOKE-PASS",
        "issued_at": _utc_now(),
        "bindings": bindings,
        "hardware": hardware,
        "batch_selection": selection,
        "functional": dict(result),
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def validate_smoke_authorization(
    payload: Mapping[str, Any],
    *,
    source_tree_sha256: str,
    sp0_authorization_sha256: str,
    batch_feasibility_sha256: str,
    gradient_contract_sha256: str,
    query_interface_sha256: str,
    resource_blocker_sha256: str,
    expected_microbatch_per_gpu: int,
    expected_accumulation: int,
    expected_devices: Sequence[int],
) -> None:
    if payload.get("gate") != "SSMOKE-PASS" or payload.get("status") != "pass":
        raise SonataSecondPreflightError("SSMOKE gate is not passing")
    observed_hash = payload.get("authorization_sha256")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256", None)
    if observed_hash != canonical_sha256(unsigned):
        raise SonataSecondPreflightError("SSMOKE authorization payload hash differs")
    bindings = payload.get("bindings")
    expected_bindings = {
        "source_tree_sha256": source_tree_sha256,
        "sp0_authorization_sha256": sp0_authorization_sha256,
        "batch_feasibility_sha256": batch_feasibility_sha256,
        "gradient_contract_sha256": gradient_contract_sha256,
        "query_interface_sha256": query_interface_sha256,
        "resource_blocker_sha256": resource_blocker_sha256,
    }
    if not isinstance(bindings, Mapping) or any(
        bindings.get(key) != value for key, value in expected_bindings.items()
    ):
        raise SonataSecondPreflightError("SSMOKE authorization bindings differ")
    selection = payload.get("batch_selection")
    if not isinstance(selection, Mapping) or (
        selection.get("microbatch_per_gpu") != expected_microbatch_per_gpu
        or selection.get("accumulate_grad_batches") != expected_accumulation
        or selection.get("effective_global_batch") != TARGET_EFFECTIVE_BATCH
        or selection.get("selection_uses_validation_accuracy") is not False
    ):
        raise SonataSecondPreflightError("SSMOKE batch selection differs")
    hardware = payload.get("hardware")
    expected_aliases = [f"device-{device}" for device in expected_devices]
    if not isinstance(hardware, Mapping) or hardware.get("selected_devices") != expected_aliases:
        raise SonataSecondPreflightError("SSMOKE selected hardware differs")


def require_smoke_authorization(
    *,
    preflight_dir: Path = PREFLIGHT_DIR,
    smoke_dir: Path = SMOKE_DIR,
    expected_microbatch_per_gpu: int,
    expected_accumulation: int,
    expected_devices: Sequence[int],
) -> dict[str, Any]:
    source = build_sonata_source_tree_contract(PROJECT_ROOT, require_clean=True)
    payload = _load_json(smoke_dir / "smoke_results.json", label="SSMOKE results")
    validate_smoke_authorization(
        payload,
        source_tree_sha256=source["content_sha256"],
        sp0_authorization_sha256=file_sha256(
            preflight_dir / "preflight_authorization.json"
        ),
        batch_feasibility_sha256=file_sha256(
            preflight_dir / "batch_feasibility.csv"
        ),
        gradient_contract_sha256=file_sha256(
            smoke_dir / "gradient_contract.json"
        ),
        query_interface_sha256=file_sha256(smoke_dir / "query_interface.json"),
        resource_blocker_sha256=file_sha256(RESOURCE_BLOCKER_PATH),
        expected_microbatch_per_gpu=expected_microbatch_per_gpu,
        expected_accumulation=expected_accumulation,
        expected_devices=expected_devices,
    )
    return payload


def _write_smoke_report(payload: Mapping[str, Any]) -> None:
    functional = payload["functional"]
    temporal = functional["temporal_execution"]
    optimization = functional["objective"]["tiny_optimization"]
    selection = payload["batch_selection"]
    lines = [
        "# Sonata Second-Perception Smoke",
        "",
        f"- Gate: `{payload['gate']}`",
        f"- Source tree SHA-256: `{payload['bindings']['source_tree_sha256']}`",
        f"- Physical/effective batch: {selection['physical_global_batch']} / {selection['effective_global_batch']}",
        f"- Temporal-overlay calls: {temporal['temporal_overlay']}",
        f"- Temporal-mask calls: {temporal['temporal_pool_merge']}",
        "- Contrastive loss: disabled; no contrastive objective term observed.",
        "- Frozen encoder/embedding gradients: absent.",
        "- Sonata decoder and ReScene task gradients: finite and nonzero.",
        "- Query feature interface: `[1, 100, 128]`.",
        f"- Tiny optimization initial/minimum objective: {optimization['initial_objective']:.6f} / {optimization['minimum_after_initial']:.6f}",
        "",
        "This preflight-only optimization is a runtime sanity check, not model-selection evidence.",
        "",
    ]
    _write_text(SMOKE_DIR / "smoke_report.md", "\n".join(lines))


def run_smoke(
    *,
    weight_path: Path,
    training_output_dir: Path,
    devices: Sequence[int],
) -> dict[str, Any]:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise SonataSecondPreflightError("SSMOKE requires the repository root")
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("SSMOKE requires exactly two distinct selected devices")
    verified_weight = require_formal_authorization(
        weight_path=weight_path,
        training_output_dir=training_output_dir,
        artifact_dir=PREFLIGHT_DIR,
    )
    hardware = _hardware_contract(devices)
    device = torch.device("cuda", devices[0])
    torch.cuda.set_device(device)
    records, details = _probe_batches(
        verified_weight,
        training_output_dir,
        device,
        gpu_count=len(devices),
    )
    blocker = _load_json(RESOURCE_BLOCKER_PATH, label="formal resource blocker")
    records, blocker_evidence = apply_formal_resource_blocker(records, blocker)
    details["formal_resource_blocker"] = {
        **blocker_evidence,
        "resource_blocker_sha256": file_sha256(RESOURCE_BLOCKER_PATH),
    }
    selection = select_batch_configuration(records, gpu_count=len(devices))
    _write_batch_artifacts(records, selection, hardware, details)

    formal_cfg = _compose_config(verified_weight, training_output_dir)
    if (
        int(formal_cfg.data.batch_size) != selection["microbatch_per_gpu"]
        or int(formal_cfg.trainer.accumulate_grad_batches)
        != selection["accumulate_grad_batches"]
    ):
        raise RuntimeError(
            "resource-selected batch differs from the formal config; update and "
            "reauthorize SP0 before functional smoke"
        )

    functional, gradient_contract, query_interface = _functional_smoke(
        verified_weight, training_output_dir, device
    )
    gradient_path = SMOKE_DIR / "gradient_contract.json"
    query_path = SMOKE_DIR / "query_interface.json"
    _write_json(gradient_path, gradient_contract)
    _write_json(query_path, query_interface)
    result = _issue_smoke_results(
        result=functional,
        gradient_path=gradient_path,
        query_path=query_path,
        batch_path=PREFLIGHT_DIR / "batch_feasibility.csv",
        hardware=hardware,
        selection=selection,
    )
    _write_json(SMOKE_DIR / "smoke_results.json", result)
    _write_smoke_report(result)
    require_smoke_authorization(
        expected_microbatch_per_gpu=int(formal_cfg.data.batch_size),
        expected_accumulation=int(formal_cfg.trainer.accumulate_grad_batches),
        expected_devices=devices,
    )
    return result


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("devices must be comma-separated integers") from error
    if len(devices) != 2 or len(set(devices)) != 2 or any(value < 0 for value in devices):
        raise argparse.ArgumentTypeError("devices must identify two distinct GPUs")
    return devices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--devices", type=_parse_devices, default=(1, 2))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_only:
        cfg = _compose_config(args.weight_path, args.training_output_dir)
        payload = require_smoke_authorization(
            expected_microbatch_per_gpu=int(cfg.data.batch_size),
            expected_accumulation=int(cfg.trainer.accumulate_grad_batches),
            expected_devices=args.devices,
        )
        print(json.dumps({"gate": payload["gate"], "verified": True}))
        return 0
    result = run_smoke(
        weight_path=args.weight_path,
        training_output_dir=args.training_output_dir,
        devices=args.devices,
    )
    print(json.dumps({"gate": result["gate"], "verified": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
