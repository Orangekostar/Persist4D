#!/usr/bin/env python3
"""Profile official ReScene4D temporal scaling on supervised 3RScan windows."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
CONCERTO_CHECKPOINT_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
CONCERTO_CHECKPOINT_BYTES = 433_987_358
DEFAULT_CHECKPOINT = (
    Path.home() / ".cache" / "persist4d" / "concerto" / "concerto_base.pth"
)
DEFAULT_REFERENCE_SCENE = 219
BATCH_SEARCH_HORIZONS = (2, 4, 5)

CSV_REQUIRED_FIELDS = (
    "run_id",
    "mode",
    "T",
    "batch_size",
    "trial",
    "precision",
    "seed",
    "sequence_names",
    "num_points",
    "num_voxels",
    "peak_gpu_memory_mb",
    "wall_time_ms",
    "samples_per_second",
    "forward_backward_ms",
    "max_batch_size_without_oom",
    "oom_observed",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "torch_version",
    "cuda_version",
    "backbone",
    "voxel_size",
    "freeze_mode",
    "source_commit",
)

SUMMARY_METRICS = (
    "num_points",
    "num_voxels",
    "peak_gpu_memory_mb",
    "wall_time_ms",
    "samples_per_second",
    "forward_backward_ms",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_max_batch_size(
    can_run: Callable[[int], bool],
    maximum: int,
    *,
    attempts: MutableSequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Find the exact viable maximum with exponential then binary search.

    A successful candidate must complete twice. Only
    ``torch.cuda.OutOfMemoryError`` is interpreted as an observed OOM.
    """

    if maximum < 1:
        raise ValueError("maximum must be at least 1")

    import torch

    attempt_log = attempts if attempts is not None else []

    def candidate_succeeds(batch_size: int) -> bool:
        for repetition in (1, 2):
            try:
                outcome = can_run(batch_size)
            except torch.cuda.OutOfMemoryError as error:
                attempt_log.append(
                    {
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "outcome": "oom",
                        "exception": type(error).__name__,
                    }
                )
                return False
            if not outcome:
                attempt_log.append(
                    {
                        "batch_size": batch_size,
                        "repetition": repetition,
                        "outcome": "oom",
                        "exception": None,
                    }
                )
                return False
            attempt_log.append(
                {
                    "batch_size": batch_size,
                    "repetition": repetition,
                    "outcome": "success",
                    "exception": None,
                }
            )
        return True

    last_success = 0
    first_failure: int | None = None
    candidate = 1
    while candidate <= maximum:
        if candidate_succeeds(candidate):
            last_success = candidate
            if candidate == maximum:
                return {
                    "max_batch_size_without_oom": maximum,
                    "oom_observed": False,
                }
            candidate = min(maximum, candidate * 2)
        else:
            first_failure = candidate
            break

    if last_success == 0:
        return {"max_batch_size_without_oom": 0, "oom_observed": True}
    if first_failure is None:
        return {
            "max_batch_size_without_oom": last_success,
            "oom_observed": False,
        }

    low = last_success
    high = first_failure - 1
    while low < high:
        middle = (low + high + 1) // 2
        if candidate_succeeds(middle):
            low = middle
        else:
            high = middle - 1
    return {"max_batch_size_without_oom": low, "oom_observed": True}


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (int(row["T"]), str(row["mode"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (horizon, mode), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "T": horizon,
            "mode": mode,
            "trials": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = [_numeric(row.get(metric)) for row in group]
            clean = [value for value in values if value is not None]
            if metric == "samples_per_second" and not clean:
                clean = [
                    1000.0 * float(row.get("batch_size", 1)) / float(row["wall_time_ms"])
                    for row in group
                    if _numeric(row.get("wall_time_ms")) not in (None, 0.0)
                ]
            if not clean:
                continue
            summary[metric] = statistics.median(clean)
            summary[f"{metric}_min"] = min(clean)
            summary[f"{metric}_max"] = max(clean)

        if "samples_per_second" not in summary and "wall_time_ms" in summary:
            batch_sizes = [int(row.get("batch_size", 1)) for row in group]
            if len(set(batch_sizes)) == 1:
                throughput = 1000.0 * batch_sizes[0] / summary["wall_time_ms"]
                summary["samples_per_second"] = throughput
                summary["samples_per_second_min"] = throughput
                summary["samples_per_second_max"] = throughput

        for field in ("max_batch_size_without_oom", "oom_observed"):
            values = [row.get(field) for row in group if row.get(field) not in (None, "")]
            if values:
                summary[field] = values[0]
        summaries.append(summary)
    return summaries


def write_measurement_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    missing_by_row = []
    unexpected_by_row = []
    for index, row in enumerate(rows):
        missing = sorted(set(CSV_REQUIRED_FIELDS) - set(row))
        if missing:
            missing_by_row.append(f"row {index}: {', '.join(missing)}")
        unexpected = sorted(set(row) - set(CSV_REQUIRED_FIELDS))
        if unexpected:
            unexpected_by_row.append(f"row {index}: {', '.join(unexpected)}")
    if missing_by_row:
        raise ValueError("missing required CSV fields: " + "; ".join(missing_by_row))
    if unexpected_by_row:
        raise ValueError(
            "unexpected CSV fields: " + "; ".join(unexpected_by_row)
        )

    fieldnames = list(CSV_REQUIRED_FIELDS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_measurement_csv(path: Path) -> list[dict[str, Any]]:
    integer_fields = {
        "T",
        "batch_size",
        "trial",
        "seed",
        "num_points",
        "num_voxels",
        "max_batch_size_without_oom",
    }
    float_fields = {
        "peak_gpu_memory_mb",
        "wall_time_ms",
        "samples_per_second",
        "forward_backward_ms",
        "voxel_size",
    }
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for field in integer_fields:
                if row.get(field) not in (None, ""):
                    row[field] = int(row[field])
            for field in float_fields:
                if row.get(field) not in (None, ""):
                    row[field] = float(row[field])
            if row.get("oom_observed") not in (None, ""):
                row["oom_observed"] = row["oom_observed"].lower() == "true"
            rows.append(row)
    return rows


def select_reference_windows(
    sequence_database: Mapping[str, Mapping[str, Any]],
    dataset_sequence_names: Sequence[str],
    *,
    reference_scene: int,
    expected_count: int = 5,
) -> list[dict[str, Any]]:
    name_to_index = {name: index for index, name in enumerate(dataset_sequence_names)}
    selected = []
    for name, entry in sequence_database.items():
        if entry.get("type") != "validation":
            continue
        if int(entry.get("scene", -1)) != reference_scene:
            continue
        if name not in name_to_index:
            raise ValueError(f"sequence is absent from official validation loader: {name}")
        selected.append(
            {
                "sequence_name": name,
                "dataset_index": name_to_index[name],
                "sub_scenes": list(entry.get("sub_scenes", [])),
            }
        )
    selected.sort(key=lambda item: item["sequence_name"])
    if len(selected) != expected_count:
        raise ValueError(
            f"reference scene {reference_scene} must provide exactly {expected_count} "
            f"validation windows, found {len(selected)}"
        )
    return selected


def render_plots_from_csv(
    csv_path: Path,
    output_dir: Path,
    *,
    measurement_iterations: int,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = read_measurement_csv(csv_path)
    if not rows:
        raise ValueError("cannot render plots from an empty CSV")
    group_counts = Counter((int(row["T"]), str(row["mode"])) for row in rows)
    horizons = sorted({horizon for horizon, _ in group_counts})
    expected_groups = {
        (horizon, mode) for horizon in horizons for mode in ("inference", "training")
    }
    count_errors = [
        f"T={horizon}/mode={mode} expected {measurement_iterations}, "
        f"found {group_counts.get((horizon, mode), 0)}"
        for horizon, mode in sorted(expected_groups)
        if group_counts.get((horizon, mode), 0) != measurement_iterations
    ]
    count_errors.extend(
        f"unexpected T={horizon}/mode={mode} with {count} rows"
        for (horizon, mode), count in sorted(group_counts.items())
        if (horizon, mode) not in expected_groups
    )
    if count_errors:
        raise ValueError("measurement group count mismatch: " + "; ".join(count_errors))
    validated_measurement_count = group_counts[(horizons[0], "inference")]
    summaries = summarize_rows(rows)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    styles = {
        "inference": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
        "training": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    plot_specs = (
        (
            "peak_vram_vs_t.png",
            "peak_gpu_memory_mb",
            "Peak allocated GPU memory (MiB)",
            "ReScene4D peak VRAM vs temporal horizon",
        ),
        (
            "latency_vs_t.png",
            "wall_time_ms",
            "Synchronized wall time (ms / sequence)",
            "ReScene4D latency vs temporal horizon",
        ),
        (
            "throughput_vs_t.png",
            "samples_per_second",
            "Throughput (sequences / s)",
            "ReScene4D throughput vs temporal horizon",
        ),
    )
    plot_paths = []
    for filename, metric, ylabel, title in plot_specs:
        fig, axis = plt.subplots(figsize=(7.4, 4.6), dpi=200)
        fig.patch.set_facecolor("white")
        axis.set_facecolor("white")
        for mode in ("inference", "training"):
            mode_rows = sorted(
                (row for row in summaries if row["mode"] == mode),
                key=lambda row: row["T"],
            )
            if not mode_rows:
                continue
            x = np.asarray([row["T"] for row in mode_rows], dtype=float)
            y = np.asarray([row[metric] for row in mode_rows], dtype=float)
            low = np.asarray([row[f"{metric}_min"] for row in mode_rows], dtype=float)
            high = np.asarray([row[f"{metric}_max"] for row in mode_rows], dtype=float)
            style = styles[mode]
            axis.fill_between(x, low, high, color=style["color"], alpha=0.14, linewidth=0)
            axis.plot(
                x,
                y,
                label=mode.capitalize(),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6,
            )
        axis.set_xticks(horizons)
        axis.set_xlabel("Temporal horizon T")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=12, fontweight="semibold")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
        fig.text(
            0.5,
            0.012,
            f"Source: {Path(csv_path).name}; median and observed min-max, n={validated_measurement_count}",
            ha="center",
            fontsize=7.5,
            color="#666666",
        )
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        output_path = output_dir / filename
        fig.savefig(output_path, facecolor="white", metadata={"Software": "Persist4D profiler"})
        plt.close(fig)
        plot_paths.append(str(output_path))
    return {
        "summary": summaries,
        "plot_paths": plot_paths,
        "measurement_count": validated_measurement_count,
    }


def render_markdown(
    summaries: Sequence[Mapping[str, Any]],
    *,
    measurement_iterations: int,
    reference_scene: int,
    checkpoint_sha256: str,
    safety_margin_mb: int,
) -> str:
    lines = [
        "# ReScene4D Temporal Scaling on NVIDIA A40",
        "",
        (
            f"Official ReScene4D commit `{OFFICIAL_SOURCE_COMMIT}`; FP32 outer tensors "
            "(TF32-eligible matmul high); voxel size "
            f"0.02 m; validation reference scene `{reference_scene}`; "
            f"{measurement_iterations} measured trials per horizon/mode."
        ),
        "",
        "## Median Scaling (observed min-max)",
        "",
        "| Mode | T | Raw points | Voxels | Peak VRAM MiB | Wall ms | Sequences/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: (int(item["T"]), str(item["mode"]))):
        lines.append(
            "| {mode} | {T} | {points:.0f} [{points_min:.0f}, {points_max:.0f}] | "
            "{voxels:.0f} [{voxels_min:.0f}, {voxels_max:.0f}] | "
            "{memory:.1f} [{memory_min:.1f}, {memory_max:.1f}] | "
            "{wall:.1f} [{wall_min:.1f}, {wall_max:.1f}] | "
            "{throughput:.4f} [{throughput_min:.4f}, {throughput_max:.4f}] |".format(
                mode=row["mode"],
                T=row["T"],
                points=row["num_points"],
                points_min=row["num_points_min"],
                points_max=row["num_points_max"],
                voxels=row["num_voxels"],
                voxels_min=row["num_voxels_min"],
                voxels_max=row["num_voxels_max"],
                memory=row["peak_gpu_memory_mb"],
                memory_min=row["peak_gpu_memory_mb_min"],
                memory_max=row["peak_gpu_memory_mb_max"],
                wall=row["wall_time_ms"],
                wall_min=row["wall_time_ms_min"],
                wall_max=row["wall_time_ms_max"],
                throughput=row["samples_per_second"],
                throughput_min=row["samples_per_second_min"],
                throughput_max=row["samples_per_second_max"],
            )
        )

    lines.extend(
        [
            "",
            "## T=2/4/5 Maximum Batch Search",
            "",
            f"The search keeps a live {safety_margin_mb} MiB CUDA reserve while probing candidates.",
            "",
            "| Mode | T | Maximum successful batch | CUDA OOM observed | Search status |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    batch_rows = [
        row
        for row in summaries
        if row.get("max_batch_size_without_oom") not in (None, "")
    ]
    for result in sorted(batch_rows, key=lambda row: (int(row["T"]), str(row["mode"]))):
        oom_observed = result["oom_observed"]
        lines.append(
            f"| {result['mode']} | {result['T']} | {result['max_batch_size_without_oom']} | "
            f"{str(oom_observed).lower()} | "
            f"{'cuda_oom' if oom_observed else 'configured_cap_right_censored'} |"
        )

    lines.extend(
        [
            "",
            "## Measurement Contract",
            "",
            f"- Each T uses the five sorted cyclic windows from validation scene 0219; this run cycles over them for n={measurement_iterations} measured trials.",
            "- Inference uses `eval()` plus `inference_mode()`; training uses the official ReScene forward, SetCriterion, and backward path with the backbone encoder frozen.",
            "- Timings synchronize CUDA and exclude dataset loading, collation, and all host-to-device transfers; peak memory includes resident model, input, and target tensors.",
            "- The validation collator's train-mode GridSample is retained, with RNG seed 45 reset before every materialization.",
            "- Batch search repeats a freshly loaded median-point sequence and requires two consecutive successful operations per candidate.",
            "- A successful configured cap is right-censored and is not described as an observed OOM.",
            "- Precision uses FP32 outer tensors without autocast and `torch.set_float32_matmul_precision('high')`; TF32-eligible CUDA matmuls may use TF32, so these measurements are not strict IEEE FP32.",
            "- The unmodified Concerto FlashAttention path explicitly casts QKV to FP16 internally.",
            "",
            "## Model Limitation",
            "",
            (
                "The official complete ReScene checkpoint is not released. These resource measurements use the "
                f"pretrained Concerto encoder checkpoint `{checkpoint_sha256}` with the Concerto decoder and "
                "ReScene heads deterministically initialized at seed 45. They do not measure accuracy."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def repo_reference(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return f"repo:{path.relative_to(PROJECT_ROOT).as_posix()}"
    except ValueError:
        if path == DEFAULT_CHECKPOINT.resolve():
            return "local_cache:persist4d/concerto/concerto_base.pth"
        return f"external:{path.name}"


def validate_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Concerto checkpoint does not exist: {path}")
    byte_size = path.stat().st_size
    checksum = sha256_file(path)
    if byte_size != CONCERTO_CHECKPOINT_BYTES:
        raise ValueError(
            f"Concerto checkpoint size mismatch: {byte_size} != {CONCERTO_CHECKPOINT_BYTES}"
        )
    if checksum != CONCERTO_CHECKPOINT_SHA256:
        raise ValueError(
            f"Concerto checkpoint SHA-256 mismatch: {checksum} != {CONCERTO_CHECKPOINT_SHA256}"
        )
    return {
        "reference": repo_reference(path),
        "filename": path.name,
        "byte_size": byte_size,
        "sha256": checksum,
        "huggingface_revision": "c31f993a56129f2ba9c5d06a35957e3f05bff710",
        "license": "CC-BY-NC-4.0",
    }


def gpu_provenance(device_index: int) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = [part.strip() for part in result.stdout.strip().split(",")]
    else:
        parts = []
    return {
        "name": parts[0] if len(parts) > 0 else properties.name,
        "uuid": "redacted",
        "device_alias": f"device-{device_index}",
        "uuid_redacted": True,
        "driver_version": parts[1] if len(parts) > 1 else "unavailable",
        "total_memory_mib": (
            int(float(parts[2]))
            if len(parts) > 2
            else round(properties.total_memory / 1024**2)
        ),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device_index": device_index,
    }


def compose_official_runtime(
    *,
    horizon: int,
    processed_dir: Path,
    checkpoint: Path,
    seed: int,
    voxel_size: float,
    freeze_mode: str,
    device_index: int,
) -> tuple[Any, Any, Any, Any]:
    import hydra
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    seed_everything(seed)
    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "data/datasets=rio",
                "backbone=concerto",
                "model=rescene",
                "serialization=mixed",
            ],
        )

    processed_dir = Path(processed_dir).resolve()
    with open_dict(config):
        config.general.seed = seed
        config.general.freeze = freeze_mode
        config.general.gpus = 1
        config.general.compile_model = False
        config.data.voxel_size = voxel_size
        config.data.num_workers = 0
        config.data.test_batch_size = 1
        config.model.config.temporal_window = horizon
        config.backbone.name = str(Path(checkpoint).resolve())
        dataset_config = config.data.validation_dataset
        dataset_config.data_dir = str(processed_dir)
        dataset_config.label_db_filepath = str(processed_dir / "label_database.yaml")
        dataset_config.change_label_db_filepath = str(
            processed_dir / "change_label_database.yaml"
        )
        dataset_config.color_mean_std = str(processed_dir / "color_mean_std.yaml")
        dataset_config.temporal_window = horizon

    dataset = hydra.utils.instantiate(config.data.validation_dataset)
    collate = hydra.utils.instantiate(config.data.validation_collation)

    from trainer.trainer import InstanceSegmentation

    device = torch.device("cuda", device_index)
    system = InstanceSegmentation(config).to(device)
    return config, dataset, collate, system


def move_targets_to_device(targets: Sequence[Mapping[str, Any]], device: Any) -> list[dict[str, Any]]:
    import torch

    moved = []
    for target in targets:
        moved.append(
            {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in target.items()
            }
        )
    return moved


def move_data_to_device(data: Any, device: Any) -> Any:
    """Stage the same direct tensor fields moved by PointceptBackbone.encoder."""

    import torch

    for key in list(data.keys()):
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device, non_blocking=True)
    return data


def materialize_batch(
    dataset: Any,
    collate: Any,
    dataset_indices: Sequence[int],
    *,
    seed: int,
    horizon: int,
) -> tuple[Any, list[dict[str, Any]], list[str], int, int]:
    import numpy as np

    seed_everything(seed)
    samples = [dataset[index] for index in dataset_indices]
    for sample in samples:
        stages = sorted(int(stage) for stage in np.unique(sample[0][:, 3]).tolist())
        if stages != list(range(horizon)):
            raise ValueError(
                f"sample {sample[3]} has temporal stages {stages}, expected {list(range(horizon))}"
            )
    num_points = sum(int(sample[0].shape[0]) for sample in samples)
    data, targets, names = collate(samples)
    if not targets:
        raise ValueError(f"official collator returned no supervised targets for {names}")
    num_voxels = int(data.features.shape[0])
    return data, targets, list(names), num_points, num_voxels


def execute_operation(
    system: Any,
    data: Any,
    targets: Sequence[Mapping[str, Any]],
    *,
    mode: str,
) -> float | None:
    import torch

    if mode not in {"inference", "training"}:
        raise ValueError(f"unsupported profiling mode: {mode}")

    point2segment = [target["point2segment"] for target in targets]
    raw_coordinates = system._process_raw_coordinates(data)
    if mode == "inference":
        system.eval()
        with torch.inference_mode():
            system.forward(
                data,
                point2segment=point2segment,
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        return None

    system.train()
    system.zero_grad(set_to_none=True)
    output = system.forward(
        data,
        point2segment=point2segment,
        raw_coordinates=raw_coordinates,
        is_eval=False,
    )
    losses = system.criterion(output, targets, mask_type="segment_mask")
    loss = sum(losses.values())
    loss.backward()
    return float(loss.detach().cpu())


def measure_operation(
    system: Any,
    data: Any,
    targets: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.perf_counter()
    operation_start = time.perf_counter()
    loss_value = execute_operation(system, data, targets, mode=mode)
    torch.cuda.synchronize(device)
    operation_ms = (time.perf_counter() - operation_start) * 1000.0
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {
        "wall_time_ms": wall_ms,
        "forward_backward_ms": operation_ms,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "loss_value": loss_value,
    }


def cleanup_cuda(system: Any | None, *, empty_cache: bool) -> None:
    import torch

    if system is not None:
        system.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        if empty_cache:
            torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_mode(
    *,
    system: Any,
    dataset: Any,
    collate: Any,
    selected_windows: Sequence[Mapping[str, Any]],
    horizon: int,
    mode: str,
    warmup_iterations: int,
    measurement_iterations: int,
    seed: int,
    device: Any,
    run_id: str,
    hardware: Mapping[str, Any],
    voxel_size: float,
    freeze_mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for warmup in range(warmup_iterations):
        window = selected_windows[warmup % len(selected_windows)]
        data = targets = None
        try:
            data, targets, _, _, _ = materialize_batch(
                dataset,
                collate,
                [int(window["dataset_index"])],
                seed=seed,
                horizon=horizon,
            )
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            execute_operation(system, data, targets, mode=mode)
            import torch

            torch.cuda.synchronize(device)
        finally:
            del data, targets
            cleanup_cuda(system, empty_cache=False)

    import torch

    for trial in range(measurement_iterations):
        window = selected_windows[trial % len(selected_windows)]
        data = targets = None
        try:
            data, targets, names, num_points, num_voxels = materialize_batch(
                dataset,
                collate,
                [int(window["dataset_index"])],
                seed=seed,
                horizon=horizon,
            )
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            measured = measure_operation(
                system, data, targets, mode=mode, device=device
            )
            wall_ms = float(measured["wall_time_ms"])
            rows.append(
                {
                    "run_id": run_id,
                    "mode": mode,
                    "T": horizon,
                    "batch_size": 1,
                    "trial": trial,
                    "precision": "fp32",
                    "seed": seed,
                    "sequence_names": json.dumps(names, separators=(",", ":")),
                    "num_points": num_points,
                    "num_voxels": num_voxels,
                    "peak_gpu_memory_mb": float(measured["peak_gpu_memory_mb"]),
                    "wall_time_ms": wall_ms,
                    "samples_per_second": 1000.0 / wall_ms,
                    "forward_backward_ms": float(measured["forward_backward_ms"]),
                    "max_batch_size_without_oom": "",
                    "oom_observed": "",
                    "gpu_name": hardware["name"],
                    "gpu_uuid": hardware["uuid"],
                    "driver_version": hardware["driver_version"],
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "backbone": "concerto_base",
                    "voxel_size": voxel_size,
                    "freeze_mode": freeze_mode,
                    "source_commit": OFFICIAL_SOURCE_COMMIT,
                }
            )
        finally:
            del data, targets
            cleanup_cuda(system, empty_cache=False)
    return rows


def choose_median_point_window(
    dataset: Any, selected_windows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    candidates = []
    for window in selected_windows:
        seed_everything(seed)
        sample = dataset[int(window["dataset_index"])]
        candidates.append(
            (int(sample[0].shape[0]), str(window["sequence_name"]), dict(window))
        )
        del sample
    candidates.sort(key=lambda item: (item[0], item[1]))
    point_count, _, selected = candidates[len(candidates) // 2]
    selected["raw_point_count"] = point_count
    return selected


def search_max_batch(
    *,
    system: Any,
    dataset: Any,
    collate: Any,
    selected_window: Mapping[str, Any],
    horizon: int,
    mode: str,
    seed: int,
    device: Any,
    maximum: int,
    safety_margin_mb: int,
) -> dict[str, Any]:
    import torch

    attempts: list[dict[str, Any]] = []
    reserve_bytes = int(safety_margin_mb) * 1024 * 1024
    safety_reserve = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)

    def can_run(batch_size: int) -> bool:
        data = targets = None
        try:
            data, targets, _, _, _ = materialize_batch(
                dataset,
                collate,
                [int(selected_window["dataset_index"])] * batch_size,
                seed=seed,
                horizon=horizon,
            )
            data = move_data_to_device(data, device)
            targets = move_targets_to_device(targets, device)
            measure_operation(system, data, targets, mode=mode, device=device)
            return True
        finally:
            del data, targets
            cleanup_cuda(system, empty_cache=True)

    try:
        result = find_max_batch_size(can_run, maximum, attempts=attempts)
    finally:
        del safety_reserve
        cleanup_cuda(system, empty_cache=True)
    return {
        **result,
        "stop_reason": (
            "cuda_oom"
            if result["oom_observed"]
            else "configured_cap_right_censored"
        ),
        "configured_cap": maximum,
        "oom_safety_margin_mb": safety_margin_mb,
        "safety_reserve_allocated_bytes": reserve_bytes,
        "selected_sequence_name": selected_window["sequence_name"],
        "selected_raw_point_count": selected_window["raw_point_count"],
        "attempts": attempts,
    }


def model_provenance(system: Any) -> dict[str, Any]:
    parameters = list(system.model.parameters())
    backbone = system.model.backbone
    encoder = getattr(getattr(backbone, "model", None), "enc", None)
    return {
        "total_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in parameters if not parameter.requires_grad
        ),
        "total_parameter_tensors": len(parameters),
        "frozen_parameter_tensors": sum(
            not parameter.requires_grad for parameter in parameters
        ),
        "encoder_module_training_at_capture": (
            bool(encoder.training) if encoder is not None else None
        ),
        "complete_rescene_checkpoint_available": False,
        "initialized_components": {
            "concerto_encoder": "pretrained",
            "concerto_decoder": "seeded_random_initialization",
            "rescene_heads": "seeded_random_initialization",
        },
    }


def profile_horizon(
    *,
    horizon: int,
    processed_dir: Path,
    checkpoint: Path,
    settings: Mapping[str, Any],
    device: Any,
    hardware: Mapping[str, Any],
    run_id: str,
    skip_batch_search: bool,
) -> dict[str, Any]:
    config = dataset = collate = system = None
    try:
        config, dataset, collate, system = compose_official_runtime(
            horizon=horizon,
            processed_dir=processed_dir,
            checkpoint=checkpoint,
            seed=int(settings["seed"]),
            voxel_size=float(settings["voxel_size"]),
            freeze_mode=str(settings["freeze_mode"]),
            device_index=int(device.index or 0),
        )
        sequence_path = Path(processed_dir) / f"sequence_database_sliding_{horizon}.yaml"
        sequence_database = load_yaml(sequence_path)
        selected_windows = select_reference_windows(
            sequence_database,
            dataset.sequence_names,
            reference_scene=int(settings["reference_scene"]),
            expected_count=int(settings["profile_scenes"]),
        )
        for window in selected_windows:
            if len(str(window["sequence_name"]).split("-")) != horizon:
                raise ValueError(
                    f"selected window does not have T={horizon}: {window['sequence_name']}"
                )

        captured_model_provenance = model_provenance(system)
        rows = []
        for mode in ("inference", "training"):
            rows.extend(
                measure_mode(
                    system=system,
                    dataset=dataset,
                    collate=collate,
                    selected_windows=selected_windows,
                    horizon=horizon,
                    mode=mode,
                    warmup_iterations=int(settings["warmup_iterations"]),
                    measurement_iterations=int(settings["measurement_iterations"]),
                    seed=int(settings["seed"]),
                    device=device,
                    run_id=run_id,
                    hardware=hardware,
                    voxel_size=float(settings["voxel_size"]),
                    freeze_mode=str(settings["freeze_mode"]),
                )
            )

        searches: dict[str, Any] = {}
        if horizon in BATCH_SEARCH_HORIZONS and not skip_batch_search:
            median_window = choose_median_point_window(
                dataset, selected_windows, seed=int(settings["seed"])
            )
            for mode in ("inference", "training"):
                result = search_max_batch(
                    system=system,
                    dataset=dataset,
                    collate=collate,
                    selected_window=median_window,
                    horizon=horizon,
                    mode=mode,
                    seed=int(settings["seed"]),
                    device=device,
                    maximum=int(settings["max_batch_search"]),
                    safety_margin_mb=int(settings["oom_safety_margin_mb"]),
                )
                searches[f"{horizon}:{mode}"] = result
                for row in rows:
                    if row["mode"] == mode:
                        row["max_batch_size_without_oom"] = result[
                            "max_batch_size_without_oom"
                        ]
                        row["oom_observed"] = result["oom_observed"]

        return {
            "rows": rows,
            "batch_search": searches,
            "selected_windows": selected_windows,
            "sequence_database": {
                "reference": repo_reference(sequence_path),
                "sha256": sha256_file(sequence_path),
                "byte_size": sequence_path.stat().st_size,
            },
            "model": captured_model_provenance,
        }
    finally:
        del config, dataset, collate, system
        cleanup_cuda(None, empty_cache=True)


def artifact_hashes(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"sha256": sha256_file(path), "byte_size": path.stat().st_size}
        for path in paths
    }


def verify_official_source_tree() -> dict[str, Any]:
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", OFFICIAL_SOURCE_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    official_paths_unchanged = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            OFFICIAL_SOURCE_COMMIT,
            "--",
            "datasets",
            "models",
            "trainer",
            "main_instance_segmentation.py",
            "conf/backbone",
            "conf/data",
            "conf/loss",
            "conf/matcher",
            "conf/model",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    if not ancestor:
        raise RuntimeError(
            f"official source commit {OFFICIAL_SOURCE_COMMIT} is not an ancestor of HEAD"
        )
    if not official_paths_unchanged:
        raise RuntimeError("official model/data execution paths differ from the locked source commit")
    return {
        "official_commit": OFFICIAL_SOURCE_COMMIT,
        "official_commit_is_ancestor": ancestor,
        "official_execution_paths_unchanged": official_paths_unchanged,
        "runtime_head": current_head,
        "profiler_sha256": sha256_file(Path(__file__)),
    }


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    profile_config = load_yaml(args.config)
    settings = dict(profile_config)
    settings["horizons"] = list(args.horizons or profile_config["horizons"])
    settings["warmup_iterations"] = (
        profile_config["warmup_iterations"]
        if args.warmup_iterations is None
        else args.warmup_iterations
    )
    settings["measurement_iterations"] = (
        profile_config["measurement_iterations"]
        if args.measurement_iterations is None
        else args.measurement_iterations
    )
    settings["max_batch_search"] = (
        profile_config["max_batch_search"]
        if args.max_batch_search is None
        else args.max_batch_search
    )
    settings["reference_scene"] = args.reference_scene

    if settings["precision"] != "fp32":
        raise ValueError("P0/P1 profiler supports only the locked fp32 precision")
    if int(settings["profile_scenes"]) != 5:
        raise ValueError("the comparable scene0219 bucket requires profile_scenes=5")
    if int(settings["warmup_iterations"]) < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if int(settings["measurement_iterations"]) < 1:
        raise ValueError("measurement_iterations must be at least one")
    if any(int(horizon) not in (2, 3, 4, 5) for horizon in settings["horizons"]):
        raise ValueError("horizons must be drawn from 2, 3, 4, 5")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the official ReScene4D profiler")
    torch.set_float32_matmul_precision("high")

    source_verification = verify_official_source_tree()
    checkpoint_info = validate_checkpoint(args.checkpoint)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "profile_manifest.json"
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    device_index = int(profile_config["gpu_index"])
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    hardware = gpu_provenance(device_index)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_id": run_id,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_verification": source_verification,
        "configuration": {
            **settings,
            "config_reference": repo_reference(Path(args.config)),
            "processed_dir_reference": repo_reference(Path(args.processed_dir)),
            "explicitly_excluded_splits": ["train", "test"],
            "profile_split": "validation",
            "batch_search_skipped": bool(args.skip_batch_search),
            "precision_contract": {
                "outer_execution": "fp32_without_autocast",
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(
                    torch.backends.cuda.matmul.allow_tf32
                ),
                "upstream_concerto_flash_attention_qkv": "explicit_fp16_cast",
            },
        },
        "checkpoint": checkpoint_info,
        "hardware": hardware,
        "dependency_sources": {
            "concerto": "10a7d17cff4dddff028f1522c2e72de4c4515df7",
            "sonata": "18c09ff8d713494f78a8213792262b910977a65d",
            "detectron2": "b4a4a3bd136852dae5fb1de37978dee412653e31",
            "stmetrics": "640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f",
        },
        "horizons": {},
        "batch_search": {},
        "errors": [],
    }
    write_json(manifest, manifest_path)

    all_rows: list[dict[str, Any]] = []
    csv_path = output_dir / "re_scene4d_scaling.csv"
    for horizon in settings["horizons"]:
        result = profile_horizon(
            horizon=int(horizon),
            processed_dir=Path(args.processed_dir),
            checkpoint=Path(args.checkpoint),
            settings=settings,
            device=device,
            hardware=hardware,
            run_id=run_id,
            skip_batch_search=bool(args.skip_batch_search),
        )
        all_rows.extend(result.pop("rows"))
        manifest["batch_search"].update(result.pop("batch_search"))
        manifest["horizons"][str(horizon)] = result
        write_measurement_csv(all_rows, csv_path)
        write_json(manifest, manifest_path)

    plot_result = render_plots_from_csv(
        csv_path,
        output_dir,
        measurement_iterations=int(settings["measurement_iterations"]),
    )
    markdown_path = output_dir / "re_scene4d_scaling.md"
    markdown_path.write_text(
        render_markdown(
            plot_result["summary"],
            measurement_iterations=int(settings["measurement_iterations"]),
            reference_scene=int(settings["reference_scene"]),
            checkpoint_sha256=checkpoint_info["sha256"],
            safety_margin_mb=int(settings["oom_safety_margin_mb"]),
        ),
        encoding="utf-8",
    )
    plot_paths = [Path(path) for path in plot_result["plot_paths"]]
    manifest["status"] = "pass"
    manifest["measurement_rows"] = len(all_rows)
    manifest["artifacts"] = artifact_hashes([csv_path, markdown_path, *plot_paths])
    write_json(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile official ReScene4D temporal scaling on one NVIDIA A40."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "conf" / "profiling" / "p0_p1_a40.yaml",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "rio",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--measurement-iterations", type=int)
    parser.add_argument("--max-batch-search", type=int)
    parser.add_argument("--reference-scene", type=int, default=DEFAULT_REFERENCE_SCENE)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-batch-search", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "profiling",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.output_dir).resolve() / "profile_manifest.json"
    try:
        manifest = run_profile(args)
    except Exception as error:  # noqa: BLE001 - runtime failures must persist a blocked manifest.
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = {"schema_version": 1}
        else:
            manifest = {"schema_version": 1}
        manifest["status"] = "blocked"
        manifest.setdefault("errors", []).append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(manifest, manifest_path)
        print(f"profiling blocked; wrote {manifest_path}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(
        f"wrote {manifest['measurement_rows']} measurements to "
        f"{Path(args.output_dir).resolve()} (status=pass)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
