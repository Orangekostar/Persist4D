"""Run aligned one-factor P2R pilots and preregistered candidate selection."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
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

from scripts.p2r_diagnostics import (
    OUTPUT_ROOT,
    P2_CHECKPOINT,
    P2_CHECKPOINT_SHA256,
    _compose_system,
    configured_objective,
)

VARIANTS = ("P2R-0", "P2R-A", "P2R-B", "P2R-C")
PILOT_STEPS = 32
TRAIN_SAMPLE_COUNT = PILOT_STEPS * 2
VALIDATION_SAMPLE_COUNT = 24
PILOT_ROOT = OUTPUT_ROOT / "pilots"
METRIC_KEYS = {
    "t_mAP": "val_mean_t-AP",
    "t_mAP50": "val_mean_t-AP_50",
    "t_mAP25": "val_mean_t-AP_25",
    "overall_mAP": "val_mean_AP",
    "stage1_mAP": "val_mean_stage1-AP",
    "stage2_mAP": "val_mean_stage2-AP",
}


def stratified_indices(population_size: int, sample_size: int) -> tuple[int, ...]:
    if (
        isinstance(population_size, bool)
        or isinstance(sample_size, bool)
        or population_size <= 0
        or sample_size <= 0
        or sample_size > population_size
    ):
        raise ValueError("sample size must be within a positive population")
    if sample_size == 1:
        return (0,)
    return tuple(
        round(index * (population_size - 1) / (sample_size - 1))
        for index in range(sample_size)
    )


def choose_full_candidate(
    results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(results) != set(VARIANTS):
        raise ValueError("pilot result coverage differs")
    baseline = results["P2R-0"]["metrics"]
    if not isinstance(baseline, Mapping):
        raise TypeError("pilot baseline metrics are invalid")
    primary = ("t_mAP", "stage1_mAP", "stage2_mAP")
    dominating = []
    for variant in VARIANTS[1:]:
        metrics = results[variant]["metrics"]
        if not isinstance(metrics, Mapping):
            raise TypeError("pilot metrics are invalid")
        if all(float(metrics[key]) > float(baseline[key]) for key in primary):
            dominating.append(variant)
    selected = (
        max(dominating, key=lambda name: float(results[name]["metrics"]["t_mAP"]))
        if dominating
        else None
    )
    return {
        "rule": "strict_pareto_improvement_over_P2R-0_on_t_mAP_stage1_mAP_stage2_mAP_then_max_t_mAP",
        "authorized": selected is not None,
        "dominating_variants": dominating,
        "selected_variant": selected,
    }


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite pilot output: {path}")
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


def _variant_settings(variant: str) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown P2R pilot variant: {variant}")
    return {
        "frozen_encoder_eval": variant == "P2R-A",
        "objective_mode": "raw_sum" if variant == "P2R-B" else "weighted",
        "batch_mode": "microbatch_1x2" if variant == "P2R-C" else "physical_2",
    }


def _ordered_indices(dataset: Any, count: int) -> tuple[int, ...]:
    ordered = sorted(
        range(len(dataset)),
        key=lambda index: hashlib.sha256(
            str(dataset.sequence_names[index]).encode("utf-8")
        ).hexdigest(),
    )
    positions = stratified_indices(len(ordered), count)
    return tuple(ordered[position] for position in positions)


def _raw_train_samples(config: Any):
    import hydra

    from scripts.run_p2_native_smoke import seed_everything

    dataset = hydra.utils.instantiate(config.data.train_dataset)
    indices = _ordered_indices(dataset, TRAIN_SAMPLE_COUNT)
    samples = []
    names = []
    for position, index in enumerate(indices):
        seed_everything(45_000 + position)
        samples.append(dataset[index])
        names.append(str(dataset.sequence_names[index]))
    return dataset, samples, names


def _move_batch(batch: object, device: torch.device):
    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )

    data, targets, names = batch
    return (
        move_data_to_device(data, device),
        move_targets_to_device(targets, device),
        list(names),
    )


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        squared += float(parameter.grad.detach().float().square().sum().cpu())
    if not found:
        raise RuntimeError("pilot optimizer step has no gradients")
    return math.sqrt(squared)


def _train_pilot(
    *,
    system: Any,
    config: Any,
    variant: str,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[str]]:
    import hydra

    from scripts.run_p2_native_smoke import (
        _forward_losses,
        _make_optimizer_scheduler,
        seed_everything,
    )

    settings = _variant_settings(variant)
    _, samples, sample_names = _raw_train_samples(config)
    collate = hydra.utils.instantiate(config.data.train_collation)
    optimizer, scheduler = _make_optimizer_scheduler(
        config, system, total_steps=PILOT_STEPS
    )
    trainable = [
        parameter for parameter in system.parameters() if parameter.requires_grad
    ]
    history = []
    for step in range(PILOT_STEPS):
        pair = samples[step * 2 : step * 2 + 2]
        seed_everything(46_000 + step)
        if settings["batch_mode"] == "physical_2":
            batches = [collate([copy.deepcopy(item) for item in pair])]
        else:
            batches = [collate([copy.deepcopy(item)]) for item in pair]
        optimizer.zero_grad(set_to_none=True)
        system.train()
        objectives = []
        component_totals: dict[str, float] = {}
        point_count = 0
        for batch in batches:
            data, targets, _ = _move_batch(batch, device)
            output, losses, _ = _forward_losses(system, data, targets)
            objective = configured_objective(
                losses,
                system.criterion.weight_dict,
                str(settings["objective_mode"]),
            )
            (objective / len(batches)).backward()
            objectives.append(float(objective.detach().cpu()))
            point_count += int(data.features.shape[0])
            for name, value in losses.items():
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    value.detach().cpu()
                ) / len(batches)
            del output, losses, objective, data, targets
        gradient_norm = _gradient_norm(trainable)
        optimizer.step()
        scheduler.step()
        history.append(
            {
                "optimizer_step": step + 1,
                "objective": sum(objectives) / len(objectives),
                "gradient_norm": gradient_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "point_count": point_count,
                "loss_components": component_totals,
            }
        )
        print(
            f"{variant} pilot step {step + 1}/{PILOT_STEPS} "
            f"loss={history[-1]['objective']:.6f}",
            flush=True,
        )
    return history, sample_names


def _evaluate_pilot(
    *, system: Any, config: Any, device: torch.device
) -> tuple[dict[str, float], list[str]]:
    import gc

    import hydra

    from scripts.profile_temporal_scaling import (
        move_data_to_device,
        move_targets_to_device,
    )
    from scripts.run_p2_native_smoke import seed_everything

    dataset = hydra.utils.instantiate(config.data.validation_dataset)
    indices = _ordered_indices(dataset, VALIDATION_SAMPLE_COUNT)
    names = [str(dataset.sequence_names[index]) for index in indices]
    collate = hydra.utils.instantiate(config.data.validation_collation)
    system.validation_dataset = dataset
    system.instance_metric.reset()
    system.eval()
    with torch.inference_mode():
        for position, index in enumerate(indices):
            seed_everything(47_000 + position)
            sample = dataset[index]
            seed_everything(47_000 + position)
            data, targets, file_names = collate([sample])
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            output = system.forward(
                data,
                point2segment=[target["point2segment"] for target in targets],
                raw_coordinates=system._process_raw_coordinates(data),
                is_eval=True,
                targets=targets,
            )
            predictions = system._process_predictions(
                output=output,
                target_low_res=targets,
                target_full_res=data.target_full,
                inverse_maps=data.inverse_maps,
                file_names=list(file_names),
                full_res_coords=data.original_coordinates,
                original_colors=data.original_colors,
                original_normals=data.original_normals,
                raw_coords=None,
                idx=data.idx,
            )
            system.instance_metric.update(predictions, data.target_full)
            del output, predictions, data, targets
            if (position + 1) % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    computed = system.instance_metric.compute()
    metrics = {}
    for output_name, source_name in METRIC_KEYS.items():
        value = computed[source_name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise RuntimeError(f"pilot metric is invalid: {output_name}")
        metrics[output_name] = number
    system.instance_metric.reset()
    return metrics, names


def run_pilot(
    *, variant: str, device_index: int, output_root: Path = PILOT_ROOT
) -> Mapping[str, object]:
    import gc

    from scripts.run_p2_native_smoke import seed_everything

    settings = _variant_settings(variant)
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device)
    seed_everything(45)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config, system, load = _compose_system(
        device=device,
        frozen_encoder_eval=bool(settings["frozen_encoder_eval"]),
    )
    history, train_names = _train_pilot(
        system=system,
        config=config,
        variant=variant,
        device=device,
    )
    metrics, validation_names = _evaluate_pilot(
        system=system, config=config, device=device
    )
    torch.cuda.synchronize(device)
    result = {
        "schema_version": 1,
        "status": "pass",
        "variant": variant,
        "settings": settings,
        "source_commit": _source_commit(),
        "seed": 45,
        "initialization": "weights_only_from_frozen_P2_checkpoint",
        "checkpoint_sha256": _sha256(P2_CHECKPOINT),
        "checkpoint_load": load,
        "pilot_scope": "post_P2_32_step_controlled_finetune_not_G2_evidence",
        "optimizer_steps": PILOT_STEPS,
        "sample_exposures": TRAIN_SAMPLE_COUNT,
        "scan_stage_exposures": TRAIN_SAMPLE_COUNT * 2,
        "validation_sequence_count": VALIDATION_SAMPLE_COUNT,
        "train_sample_names": train_names,
        "validation_sample_names": validation_names,
        "metrics": metrics,
        "loss_curve": [row["objective"] for row in history],
        "lr_curve": [row["lr"] for row in history],
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    payload = (
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish(output_root / f"{variant}.json", payload)
    del system
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "status": "pass",
        "variant": variant,
        "metrics": metrics,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _read_result(path: Path) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        raise ValueError(f"pilot result is invalid: {path}")
    return value


def aggregate_pilots(
    *, pilot_root: Path = PILOT_ROOT, output_root: Path = OUTPUT_ROOT
) -> Mapping[str, object]:
    results = {
        variant: _read_result(pilot_root / f"{variant}.json") for variant in VARIANTS
    }
    binding_fields = (
        "source_commit",
        "seed",
        "checkpoint_sha256",
        "optimizer_steps",
        "sample_exposures",
        "scan_stage_exposures",
        "validation_sequence_count",
        "train_sample_names",
        "validation_sample_names",
    )
    baseline = results["P2R-0"]
    for result in results.values():
        if any(result[field] != baseline[field] for field in binding_fields):
            raise ValueError("pilot alignment binding differs")
    decision = choose_full_candidate(results)
    fields = (
        "variant",
        "changed_factor",
        "optimizer_steps",
        "sample_exposures",
        "validation_sequence_count",
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "overall_mAP",
        "stage1_mAP",
        "stage2_mAP",
        "loss_start",
        "loss_end",
        "lr_start",
        "lr_end",
        "elapsed_seconds",
        "peak_allocated_vram_mib",
    )
    changed = {
        "P2R-0": "control",
        "P2R-A": "frozen_encoder_eval",
        "P2R-B": "public_raw_sum_objective",
        "P2R-C": "microbatch_1x2_normalization",
    }
    rows = []
    for variant in VARIANTS:
        result = results[variant]
        metrics = result["metrics"]
        rows.append(
            {
                "variant": variant,
                "changed_factor": changed[variant],
                "optimizer_steps": result["optimizer_steps"],
                "sample_exposures": result["sample_exposures"],
                "validation_sequence_count": result["validation_sequence_count"],
                **metrics,
                "loss_start": result["loss_curve"][0],
                "loss_end": result["loss_curve"][-1],
                "lr_start": result["lr_curve"][0],
                "lr_end": result["lr_curve"][-1],
                "elapsed_seconds": result["elapsed_seconds"],
                "peak_allocated_vram_mib": result["peak_allocated_vram_mib"],
            }
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_payload = stream.getvalue().encode("utf-8")
    lines = [
        "# P2R Pilot Decision",
        "",
        "- Status: `pass`",
        f"- Optimizer steps per path: `{PILOT_STEPS}`",
        f"- Train sample exposures per path: `{TRAIN_SAMPLE_COUNT}`",
        f"- Validation subset: `{VALIDATION_SAMPLE_COUNT}` supervised T=2 sequences",
        "- Scope: post-P2 controlled fine-tune pilot; not official-like G2 evidence",
        f"- Full candidate authorized: `{str(decision['authorized']).lower()}`",
        f"- Selected variant: `{decision['selected_variant']}`",
        f"- Selection rule: `{decision['rule']}`",
        "",
        "| Variant | t-mAP | Overall mAP | Stage1 mAP | Stage2 mAP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {float(row['t_mAP']):.6f} | "
            f"{float(row['overall_mAP']):.6f} | {float(row['stage1_mAP']):.6f} | "
            f"{float(row['stage2_mAP']):.6f} |"
        )
    lines.extend(
        [
            "",
            "A full 450-epoch candidate is authorized only by the preregistered",
            "three-metric strict Pareto rule. Pairwise pilot differences are not",
            "official-like 154-sequence G2 results.",
            "",
        ]
    )
    markdown = "\n".join(lines).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "decision": decision,
        "source_commit": baseline["source_commit"],
        "checkpoint_sha256": P2_CHECKPOINT_SHA256,
        "pilot_result_sha256": {
            variant: _sha256(pilot_root / f"{variant}.json") for variant in VARIANTS
        },
        "outputs": {
            "PILOT_MATRIX.csv": hashlib.sha256(csv_payload).hexdigest(),
            "P2R_PILOT_DECISION.md": hashlib.sha256(markdown).hexdigest(),
        },
    }
    manifest_payload = (
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish(output_root / "PILOT_MATRIX.csv", csv_payload)
    _publish(output_root / "P2R_PILOT_DECISION.md", markdown)
    _publish(output_root / "pilot_manifest.json", manifest_payload)
    return {"status": "pass", "decision": decision, "row_count": len(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "aggregate"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pilot-root", type=Path, default=PILOT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.stage == "run":
        if args.variant is None:
            parser.error("run requires --variant")
        result = run_pilot(
            variant=args.variant,
            device_index=args.device,
            output_root=args.pilot_root,
        )
    else:
        result = aggregate_pilots(
            pilot_root=args.pilot_root, output_root=args.output_root
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_pilots",
    "choose_full_candidate",
    "run_pilot",
    "stratified_indices",
]
