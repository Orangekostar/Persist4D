#!/usr/bin/env python3
"""Compare decoder gradients for feasible physical-global batch groupings."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profile_temporal_scaling import (
    move_data_to_device,
    move_targets_to_device,
)
from scripts.run_p2_native_smoke import DEFAULT_CHECKPOINT, _forward_losses
from utils.rescene_objective_audit import compare_gradients
from utils.rescene_rootcause_preflight import canonical_sha256
from utils.rescene_runtime_audit import (
    IMPORTANT_GRADIENT_GROUPS,
    build_fixed_batch_panel,
    physical_batch_gradient_gate,
    summarize_physical_batch_runs,
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _compose_config() -> Any:
    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(config_name="config_rescene4d_concerto_rootcause")
    with open_dict(config):
        config.backbone.name = str(DEFAULT_CHECKPOINT.resolve())
        config.data.num_workers = 0
        config.data.pin_memory = False
    return config


def _sample_reference(dataset: Any, mixed_index: int) -> dict[str, object]:
    lower = 0
    for child in dataset.datasets:
        upper = lower + len(child)
        if mixed_index < upper:
            child_index = mixed_index - lower
            names = getattr(child, "sequence_names", None)
            sample_id = (
                str(names[child_index])
                if names is not None
                else Path(str(child.data[child_index]["filepath"])).stem
            )
            return {
                "dataset": str(child.dataset_name),
                "sample_id": sample_id,
                "sample_index": child_index,
                "mixed_dataset_index": mixed_index,
            }
        lower = upper
    raise IndexError(f"sample index {mixed_index} exceeds the mixed dataset")


def _build_panel(config: Any) -> dict[str, object]:
    dataset = hydra.utils.instantiate(config.data.train_dataset)
    draws = [int(value) for value in dataset.sampler]
    if len(draws) < 32:
        raise RuntimeError("active weighted sampler produced fewer than 32 samples")
    references = [_sample_reference(dataset, index) for index in draws[:32]]
    panel = build_fixed_batch_panel(references, seed=45)
    panel["weighted_stream_sha256"] = canonical_sha256(draws)
    return panel


def _parameter_group(name: str) -> str | None:
    if name.startswith("model.class_embed_head."):
        return "class_embed_head"
    if name.startswith("model.mask_embed_head."):
        return "mask_embed_head"
    if name.startswith("model.query_projection."):
        return "query_projection"
    if name.startswith("model.cross_attention.0.0."):
        return "first_cross_attention"
    if name.startswith("model.cross_attention.0.3."):
        return "last_cross_attention"
    if name.startswith("model.backbone.model.dec."):
        return "ptv3_decoder"
    return None


def _gradient_vectors(system: torch.nn.Module) -> dict[str, torch.Tensor]:
    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in IMPORTANT_GRADIENT_GROUPS
    }
    for name, parameter in system.named_parameters():
        group = _parameter_group(name)
        if group is None or not parameter.requires_grad:
            continue
        gradient = parameter.grad
        values[group].append(
            (
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
            ).float()
        )
    missing = [name for name, tensors in values.items() if not tensors]
    if missing:
        raise RuntimeError("missing gradient parameter groups: " + ", ".join(missing))
    return {name: torch.cat(tensors) for name, tensors in values.items()}


def _worker(
    rank: int,
    world_size: int,
    physical_global_batch: int,
    panel: dict[str, object],
    result_path: str,
    port: int,
    device_ids: tuple[int, int],
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device(f"cuda:{device_ids[rank]}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        _seed_all(45)
        config = _compose_config()
        from trainer.trainer import InstanceSegmentation

        system = InstanceSegmentation(config).to(device)
        system.train()
        dataset = hydra.utils.instantiate(config.data.train_dataset)
        collate = hydra.utils.instantiate(config.data.train_collation)
        local_batch = physical_global_batch // world_size
        microsteps = 32 // physical_global_batch
        system.zero_grad(set_to_none=True)
        component_sums: dict[str, float] = {}
        objective_sum = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for microstep in range(microsteps):
            global_start = microstep * physical_global_batch
            local_start = global_start + rank * local_batch
            selected = panel["samples"][local_start : local_start + local_batch]
            samples = []
            for reference in selected:
                _seed_all(int(reference["augmentation_seed"]))
                samples.append(dataset[int(reference["mixed_dataset_index"])])
            _seed_all(45 + microstep)
            data, targets, _ = collate(samples)
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            _, losses, breakdown = _forward_losses(system, data, targets)
            objective = breakdown["objective"] / microsteps
            objective.backward()
            objective_sum += float(objective.detach().cpu())
            for name, value in losses.items():
                component_sums[name] = component_sums.get(name, 0.0) + (
                    float(value.detach().cpu()) / microsteps
                )
            del data, targets, losses, breakdown, objective
        vectors = _gradient_vectors(system)
        for vector in vectors.values():
            dist.all_reduce(vector, op=dist.ReduceOp.SUM)
            vector.div_(world_size)
        scalar = torch.tensor(
            [objective_sum, time.perf_counter() - started],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(scalar[:1], op=dist.ReduceOp.SUM)
        scalar[0].div_(world_size)
        dist.all_reduce(scalar[1:], op=dist.ReduceOp.MAX)
        peak = torch.tensor(
            float(torch.cuda.max_memory_allocated(device) / 1024**2),
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        gathered_components: list[dict[str, float] | None] = [None] * world_size
        dist.all_gather_object(gathered_components, component_sums)
        if rank == 0:
            names = sorted(component_sums)
            mean_components = {
                name: sum(float(values[name]) for values in gathered_components) / world_size
                for name in names
            }
            torch.save(
                {
                    "metrics": {
                        "physical_global_batch": physical_global_batch,
                        "per_device_batch": local_batch,
                        "accumulation": microsteps,
                        "effective_global_batch": 32,
                        "feasible": True,
                        "objective_value": float(scalar[0].cpu()),
                        "loss_components": mean_components,
                        "peak_memory_mib": float(peak.cpu()),
                        "step_seconds": float(scalar[1].cpu()),
                        "gpu_model": torch.cuda.get_device_name(device),
                    },
                    "vectors": {name: value.cpu() for name, value in vectors.items()},
                },
                result_path,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _run_group(
    physical_global_batch: int,
    panel: dict[str, object],
    result_path: Path,
    device_ids: tuple[int, int],
) -> dict[str, object]:
    try:
        mp.spawn(
            _worker,
            args=(
                2,
                physical_global_batch,
                panel,
                str(result_path),
                _free_port(),
                device_ids,
            ),
            nprocs=2,
            join=True,
        )
    except Exception as error:
        message = str(error)
        if "out of memory" not in message.lower() and "OutOfMemory" not in message:
            raise
        return {
            "metrics": {
                "physical_global_batch": physical_global_batch,
                "accumulation": 32 // physical_global_batch,
                "effective_global_batch": 32,
                "feasible": False,
                "failure": "CUDAOutOfMemoryError",
            }
        }
    return torch.load(result_path, map_location="cpu", weights_only=True)


def _csv_bytes(rows: list[dict[str, object]]) -> bytes:
    fields = [
        "physical_global_batch",
        "accumulation",
        "parameter_group",
        "reference_norm",
        "candidate_norm",
        "cosine",
        "relative_norm_difference",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def run_audit(*, output_dir: Path, device_ids: tuple[int, int]) -> dict[str, object]:
    if len(set(device_ids)) != 2 or any(
        device < 0 or device >= torch.cuda.device_count() for device in device_ids
    ):
        raise ValueError("physical-batch audit requires two distinct CUDA devices")
    config = _compose_config()
    panel = _build_panel(config)
    _publish(output_dir / "fixed_batch_panel.json", _json_bytes(panel))
    results = []
    with tempfile.TemporaryDirectory(prefix="rescene-physical-batch-") as temporary:
        root = Path(temporary)
        for physical_batch in (4, 8, 16, 32):
            print(f"physical-global batch {physical_batch}", flush=True)
            result = _run_group(
                physical_batch,
                panel,
                root / f"batch-{physical_batch}.pt",
                device_ids,
            )
            results.append(result)
            if result["metrics"]["feasible"] is False:
                break
        reference = results[0]
        if reference["metrics"]["feasible"] is not True:
            raise RuntimeError("physical-global batch 4 reference was not feasible")
        rows = []
        gates = {}
        for result in results:
            metrics = result["metrics"]
            if metrics["feasible"] is not True:
                gates[str(metrics["physical_global_batch"])] = physical_batch_gradient_gate(
                    [], feasible=False
                )
                continue
            candidate_rows = []
            for group in IMPORTANT_GRADIENT_GROUPS:
                comparison = compare_gradients(
                    reference["vectors"][group], result["vectors"][group]
                )
                row = {
                    "physical_global_batch": metrics["physical_global_batch"],
                    "accumulation": metrics["accumulation"],
                    "parameter_group": group,
                    "reference_norm": comparison["left_norm"],
                    "candidate_norm": comparison["right_norm"],
                    "cosine": comparison["cosine"],
                    "relative_norm_difference": comparison[
                        "relative_norm_difference"
                    ],
                }
                rows.append(row)
                candidate_rows.append(row)
            gates[str(metrics["physical_global_batch"])] = physical_batch_gradient_gate(
                candidate_rows, feasible=True
            )
        runs = [result["metrics"] for result in results]
        resource_summary = summarize_physical_batch_runs(runs)
    authorized = [
        int(batch)
        for batch, gate in gates.items()
        if int(batch) > 4 and gate["authorized"]
    ]
    summary = {
        **resource_summary,
        "source_commit": _git_head(),
        "topology": {"world_size": 2, "cuda_device_indices": list(device_ids)},
        "fixed_panel_sha256": canonical_sha256(panel),
        "gradient_gates": gates,
        "gate": {
            "authorized": bool(authorized),
            "authorized_physical_global_batches": authorized,
        },
    }
    _publish(output_dir / "physical_batch_gradients.csv", _csv_bytes(rows))
    _publish(output_dir / "physical_batch_summary.json", _json_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--devices", default="1,2")
    args = parser.parse_args()
    device_ids = tuple(int(value) for value in args.devices.split(","))
    if len(device_ids) != 2:
        raise ValueError("--devices must contain exactly two CUDA indices")
    result = run_audit(output_dir=args.output_dir, device_ids=device_ids)
    print(json.dumps({"status": "pass", "gate": result["gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
