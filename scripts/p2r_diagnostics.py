"""Controlled P2R objective, encoder-mode, and microbatch diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/P2R"
P2_CHECKPOINT = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
P2_CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
PATHS = ("P2R-0", "P2R-A", "P2R-B")
COMPARISONS = (
    "frozen_encoder_mode",
    "objective_semantics",
    "microbatch_normalization",
)


def configured_objective(
    losses: Mapping[str, Tensor],
    weight_dict: Mapping[str, float],
    mode: str,
) -> Tensor:
    if mode == "weighted":
        from trainer.trainer import aggregate_objective_loss

        return aggregate_objective_loss(losses, weight_dict)
    if mode == "raw_sum":
        if not losses:
            raise ValueError("raw-sum objective requires loss terms")
        return sum(losses.values())
    raise ValueError("objective mode must be weighted or raw_sum")


def compare_gradients(left: Tensor, right: Tensor) -> dict[str, float]:
    left = left.detach().cpu().float().reshape(-1)
    right = right.detach().cpu().float().reshape(-1)
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("gradient vectors must be non-empty and aligned")
    if not torch.isfinite(left).all().item() or not torch.isfinite(right).all().item():
        raise ValueError("gradient vectors must be finite")
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    cosine = float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())
    return {
        "element_count": int(left.numel()),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "norm_ratio_right_over_left": (
            right_norm / left_norm if left_norm else math.inf
        ),
        "cosine_similarity": cosine,
        "max_abs_difference": float((left - right).abs().max().item()),
    }


def validate_diagnostic_report(
    report: Mapping[str, object],
) -> Mapping[str, object]:
    if report.get("schema_version") != 1 or report.get("status") != "pass":
        raise ValueError("diagnostic report header differs")
    paths = report.get("paths")
    comparisons = report.get("comparisons")
    if not isinstance(paths, Mapping) or set(paths) != set(PATHS):
        raise ValueError("diagnostic path coverage differs")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(COMPARISONS):
        raise ValueError("diagnostic comparison coverage differs")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _compose_system(*, device: torch.device, frozen_encoder_eval: bool):
    from omegaconf import open_dict

    from scripts.run_p2_native_smoke import (
        DEFAULT_CHECKPOINT,
        _compose_config,
        seed_everything,
    )
    from scripts.run_reviewer_closure_t3 import strict_load_adaptation_weights
    from trainer.trainer import InstanceSegmentation

    seed_everything(45)
    config = _compose_config(DEFAULT_CHECKPOINT)
    with open_dict(config):
        config.general.gpus = 1
        config.general.frozen_encoder_eval = frozen_encoder_eval
        config.general.p2_fail_closed_runtime = False
        config.data.num_workers = 0
    system = InstanceSegmentation(config).to(device)
    load = strict_load_adaptation_weights(
        system,
        P2_CHECKPOINT,
        expected_sha256=P2_CHECKPOINT_SHA256,
    )
    return config, system, load


def _gradient_vector(system: Any, names: Sequence[str]) -> Tensor:
    parameters = dict(system.named_parameters())
    values = []
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            raise RuntimeError(f"selected parameter has no gradient: {name}")
        values.append(gradient.detach().cpu().float().reshape(-1))
    if not values:
        raise RuntimeError("no parameters selected for gradient diagnostic")
    return torch.cat(values)


def _candidate_counts(
    output: Mapping[str, object], max_points: int
) -> dict[str, object]:
    segment_counts = []
    for layer in output.get("segment_features", []):
        if isinstance(layer, Tensor):
            count = int(layer.shape[-2]) if layer.ndim >= 2 else int(layer.numel())
        else:
            count = sum(int(value.shape[0]) for value in layer)
        segment_counts.append(count)
    aux_counts = []
    for layer in output.get("aux_features", []):
        features = getattr(layer, "decomposed_features", ())
        aux_counts.append(
            sum(min(int(value.shape[0]), max_points) for value in features)
        )
    return {
        "segment_candidates_per_layer": segment_counts,
        "aux_candidates_per_layer_after_cap": aux_counts,
    }


def _path_run(
    *,
    device: torch.device,
    frozen_encoder_eval: bool,
    objective_mode: str,
) -> tuple[dict[str, object], Tensor]:
    import gc

    from scripts.run_p2_native_smoke import (
        TINY_SAMPLE_NAME,
        _forward_losses,
        _gradient_summary,
        _make_optimizer_scheduler,
        _matching_quality,
        _materialize_named_train_batch,
        classify_parameters,
        seed_everything,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system, load = _compose_system(
        device=device, frozen_encoder_eval=frozen_encoder_eval
    )
    data, targets, sample, provenance = _materialize_named_train_batch(
        config, TINY_SAMPLE_NAME, device
    )
    groups = classify_parameters(system.named_parameters())
    selected_names = groups["trainable_rescene_heads"]
    optimizer, scheduler = _make_optimizer_scheduler(config, system, total_steps=2)
    initial_lr = float(optimizer.param_groups[0]["lr"])
    optimizer.zero_grad(set_to_none=True)
    system.train()
    seed_everything(45)
    output, losses, _ = _forward_losses(system, data, targets)
    objective = configured_objective(
        losses, system.criterion.weight_dict, objective_mode
    )
    quality = _matching_quality(system, output, targets)
    counts = _candidate_counts(output, int(system.criterion.max_points))
    objective.backward()
    gradient = _gradient_vector(system, selected_names)
    summaries = {
        group: _gradient_summary(system, names) for group, names in groups.items()
    }
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    record = {
        "frozen_encoder_eval": frozen_encoder_eval,
        "objective_mode": objective_mode,
        "encoder_training": bool(system.model.backbone.model.enc.training),
        "embedding_training": bool(system.model.backbone.model.embedding.training),
        "decoder_training": bool(system.model.backbone.model.dec.training),
        "objective": float(objective.detach().cpu()),
        "loss_components": {
            key: float(value.detach().cpu()) for key, value in sorted(losses.items())
        },
        "gradient_summaries": summaries,
        "selected_gradient_parameter_names": selected_names,
        "matching": quality,
        "contrastive_candidate_counts": counts,
        "optimizer_step_count": 1,
        "lr_trace": [initial_lr, float(optimizer.param_groups[0]["lr"])],
        "sample": sample,
        "input_provenance": provenance,
        "checkpoint_load": load,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del output, losses, objective, optimizer, scheduler, system, data, targets
    gc.collect()
    torch.cuda.empty_cache()
    return record, gradient


def _materialize_duplicate_batch(config: Any, device: torch.device, *, physical: bool):
    import hydra

    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )
    from scripts.run_p2_native_smoke import TINY_SAMPLE_NAME, seed_everything

    dataset = hydra.utils.instantiate(config.data.train_dataset)
    index = dataset.sequence_names.index(TINY_SAMPLE_NAME)
    seed_everything(45)
    sample = dataset[index]
    collate = hydra.utils.instantiate(config.data.train_collation)
    seed_everything(45)
    if physical:
        batches = [collate([copy.deepcopy(sample), copy.deepcopy(sample)])]
    else:
        batches = [
            collate([copy.deepcopy(sample)]),
            collate([copy.deepcopy(sample)]),
        ]
    result = []
    for data, targets, names in batches:
        result.append(
            (
                move_data_to_device(data, device),
                move_targets_to_device(targets, device),
                list(names),
            )
        )
    return result


def _batch_run(
    *, device: torch.device, physical: bool
) -> tuple[dict[str, object], Tensor]:
    import gc

    from scripts.run_p2_native_smoke import (
        _forward_losses,
        _gradient_summary,
        _make_optimizer_scheduler,
        _matching_quality,
        classify_parameters,
        seed_everything,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system, load = _compose_system(device=device, frozen_encoder_eval=False)
    batches = _materialize_duplicate_batch(config, device, physical=physical)
    groups = classify_parameters(system.named_parameters())
    selected_names = groups["trainable_rescene_heads"]
    optimizer, scheduler = _make_optimizer_scheduler(config, system, total_steps=2)
    initial_lr = float(optimizer.param_groups[0]["lr"])
    optimizer.zero_grad(set_to_none=True)
    system.eval()
    seed_everything(45)
    objective_values = []
    matching_count = 0
    segment_counts: list[int] = []
    aux_counts: list[int] = []
    point_counts = []
    for data, targets, names in batches:
        output, losses, _ = _forward_losses(system, data, targets)
        objective = configured_objective(
            losses, system.criterion.weight_dict, "weighted"
        )
        scale = 1.0 if physical else 1.0 / len(batches)
        (objective * scale).backward()
        objective_values.append(float(objective.detach().cpu()))
        matching_count += int(
            _matching_quality(system, output, targets)["matched_instances"]
        )
        counts = _candidate_counts(output, int(system.criterion.max_points))
        if not segment_counts:
            segment_counts = [0] * len(counts["segment_candidates_per_layer"])
            aux_counts = [0] * len(counts["aux_candidates_per_layer_after_cap"])
        segment_counts = [
            left + right
            for left, right in zip(
                segment_counts,
                counts["segment_candidates_per_layer"],
                strict=True,
            )
        ]
        aux_counts = [
            left + right
            for left, right in zip(
                aux_counts,
                counts["aux_candidates_per_layer_after_cap"],
                strict=True,
            )
        ]
        point_counts.append(int(data.features.shape[0]))
        del output, losses, objective
    gradient = _gradient_vector(system, selected_names)
    summaries = {
        group: _gradient_summary(system, names) for group, names in groups.items()
    }
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize(device)
    effective_objective = (
        objective_values[0]
        if physical
        else sum(objective_values) / len(objective_values)
    )
    record = {
        "mode": "physical_batch_2" if physical else "two_microbatches_accumulated",
        "sample_names": [name for _, _, names in batches for name in names],
        "batch_point_counts": point_counts,
        "effective_objective": effective_objective,
        "microbatch_objectives": objective_values,
        "gradient_summaries": summaries,
        "selected_gradient_parameter_names": selected_names,
        "hungarian_match_count": matching_count,
        "contrastive_candidate_counts": {
            "segment_candidates_per_layer": segment_counts,
            "aux_candidates_per_layer_after_cap": aux_counts,
        },
        "optimizer_step_count": 1,
        "lr_trace": [initial_lr, float(optimizer.param_groups[0]["lr"])],
        "checkpoint_load": load,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    del optimizer, scheduler, system, batches
    gc.collect()
    torch.cuda.empty_cache()
    return record, gradient


def _markdown(report: Mapping[str, object]) -> bytes:
    paths = report["paths"]
    comparisons = report["comparisons"]
    lines = [
        "# P2R Controlled Diagnostics",
        "",
        "- Status: `pass`",
        "- Checkpoint: `85ed1aba...131546e`",
        "- Seed: `45`",
        "- Scope: one real supervised T=2 sample; diagnostic, not G2 evidence",
        "",
        "| Path | Encoder mode | Objective | Scalar loss | Head grad norm | Matches |",
        "|---|---|---|---:|---:|---:|",
    ]
    for name in PATHS:
        row = paths[name]
        grad = row["gradient_summaries"]["trainable_rescene_heads"]
        lines.append(
            f"| {name} | {'eval' if not row['encoder_training'] else 'train'} | "
            f"{row['objective_mode']} | {row['objective']:.6f} | "
            f"{grad['max_grad_norm']:.6f} | {row['matching']['matched_instances']} |"
        )
    micro = comparisons["microbatch_normalization"]
    lines.extend(
        [
            "",
            "## Microbatch Diagnostic",
            "",
            f"- Physical objective: `{micro['physical']['effective_objective']:.6f}`",
            f"- Accumulated objective: `{micro['accumulated']['effective_objective']:.6f}`",
            f"- Selected-gradient cosine: `{micro['gradient']['cosine_similarity']:.9f}`",
            f"- Selected-gradient max absolute difference: `{micro['gradient']['max_abs_difference']:.6g}`",
            "- Physical-global-32: `hardware-infeasible, not executed`",
            "",
            "Differences are controlled diagnostics on one duplicated real sample. They",
            "do not by themselves explain the 6.861-point paper gap and do not authorize",
            "a full candidate without aligned task-metric pilots.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def run_diagnostics(
    *, device_index: int, output_root: Path = OUTPUT_ROOT
) -> Mapping[str, object]:
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    source_commit = _source_commit()
    baseline, baseline_gradient = _path_run(
        device=device,
        frozen_encoder_eval=False,
        objective_mode="weighted",
    )
    encoder_eval, encoder_eval_gradient = _path_run(
        device=device,
        frozen_encoder_eval=True,
        objective_mode="weighted",
    )
    raw_sum, raw_sum_gradient = _path_run(
        device=device,
        frozen_encoder_eval=False,
        objective_mode="raw_sum",
    )
    physical, physical_gradient = _batch_run(device=device, physical=True)
    accumulated, accumulated_gradient = _batch_run(device=device, physical=False)
    report = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": source_commit,
        "checkpoint_sha256": _sha256(P2_CHECKPOINT),
        "seed": 45,
        "scope": "single_real_t2_controlled_diagnostic_not_g2_evidence",
        "paths": {
            "P2R-0": baseline,
            "P2R-A": encoder_eval,
            "P2R-B": raw_sum,
        },
        "comparisons": {
            "frozen_encoder_mode": {
                "gradient": compare_gradients(baseline_gradient, encoder_eval_gradient),
                "only_changed_factor": "frozen_encoder_train_vs_eval",
            },
            "objective_semantics": {
                "gradient": compare_gradients(baseline_gradient, raw_sum_gradient),
                "only_changed_factor": "weighted_vs_public_raw_sum",
            },
            "microbatch_normalization": {
                "physical": physical,
                "accumulated": accumulated,
                "gradient": compare_gradients(physical_gradient, accumulated_gradient),
                "paper_physical_global_32_status": "hardware_infeasible_not_executed",
            },
        },
    }
    validate_diagnostic_report(report)
    payload = (
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    markdown = _markdown(report)
    _publish(output_root / "CONTROLLED_DIAGNOSTICS.json", payload)
    _publish(output_root / "CONTROLLED_DIAGNOSTICS.md", markdown)
    return {
        "status": "pass",
        "json_sha256": hashlib.sha256(payload).hexdigest(),
        "markdown_sha256": hashlib.sha256(markdown).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run_diagnostics(device_index=args.device, output_root=args.output_root)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compare_gradients",
    "configured_objective",
    "run_diagnostics",
    "validate_diagnostic_report",
]
