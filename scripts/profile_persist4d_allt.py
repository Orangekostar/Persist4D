#!/usr/bin/env python3
"""Profile frozen C2 and matched FH-adapt on a bounded Protocol-B subset."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1"
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "profile"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_TRAINING_ROOT = Path(
    "/mnt/shared/ww/persist4d-allt-task-superiority-v1/training/formal"
)
DEFAULT_SPLIT_MANIFEST = DEFAULT_ARTIFACT_ROOT / "split_manifest.json"
PROFILE_MODELS = ("C2", "FH-adapt")
REPORT_HORIZONS = (2, 3, 4, 5)
FROZEN_CHECKPOINTS = {
    "C2": "a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724",
    "FH-adapt": (
        "ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c"
    ),
}
SAMPLE_FIELDS = (
    "model",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "repeat",
    "latency_ms",
    "start_allocated_bytes",
    "peak_allocated_bytes",
    "incremental_peak_allocated_bytes",
    "window_voxel_points",
    "window_segments",
)
SUMMARY_FIELDS = (
    "model",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "measured_repeats",
    "median_latency_ms",
    "minimum_latency_ms",
    "maximum_latency_ms",
    "peak_allocated_bytes",
    "maximum_incremental_peak_allocated_bytes",
    "window_voxel_points",
    "window_segments",
)


class ProfileError(RuntimeError):
    """Raised when the bounded resource-profile contract is violated."""


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProfileError(f"{name} must be a sequence")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ProfileError(f"{name} must be finite and positive")
    return normalized


def _sequence_identity(sequence: object) -> tuple[str, str, str]:
    requests = _sequence(getattr(sequence, "stage_requests", None), name="stage requests")
    if not requests or not isinstance(requests[0], Mapping):
        raise ProfileError("profile sequence lacks its first request")
    request = requests[0]
    return (
        _text(request.get("reference_scene_id"), name="reference scene ID"),
        _text(request.get("master_sequence_id"), name="master sequence ID"),
        _text(request.get("order_id"), name="order ID"),
    )


def select_profile_sequences(sequences: Sequence[object]) -> tuple[object, ...]:
    """Select the first sorted canonical master in each Protocol-B cluster."""
    grouped: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for sequence in _sequence(sequences, name="profile population"):
        reference, master, order = _sequence_identity(sequence)
        if order == "canonical":
            grouped[reference].append((master, sequence))
    if len(grouped) != 6 or any(not values for values in grouped.values()):
        raise ProfileError("profile selection must cover six reference clusters")
    selected = []
    for reference in sorted(grouped):
        masters = grouped[reference]
        if len({master for master, _ in masters}) != len(masters):
            raise ProfileError("canonical profile population contains duplicate masters")
        selected.append(min(masters, key=lambda item: item[0])[1])
    return tuple(selected)


def validate_profile_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    models: Sequence[str] = PROFILE_MODELS,
    expected_reference_count: int = 6,
    horizons: Sequence[int] = REPORT_HORIZONS,
    repeats: int = 10,
) -> None:
    """Require an exact rectangular measured-sample profile."""
    values = _sequence(rows, name="profile samples")
    normalized_models = tuple(_text(value, name="profile model") for value in models)
    normalized_horizons = tuple(
        _integer(value, name="profile horizon", minimum=1) for value in horizons
    )
    expected_reference_count = _integer(
        expected_reference_count, name="reference count", minimum=1
    )
    repeats = _integer(repeats, name="measured repeats", minimum=1)
    observed: set[tuple[str, str, int, int]] = set()
    references: set[str] = set()
    masters: dict[str, str] = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise ProfileError("profile sample must be a mapping")
        model = _text(row.get("model"), name="profile model")
        reference = _text(row.get("reference_scene_id"), name="profile reference")
        master = _text(row.get("master_sequence_id"), name="profile master")
        order = _text(row.get("order_id"), name="profile order")
        horizon = _integer(row.get("T"), name="profile T", minimum=1)
        repeat = _integer(row.get("repeat"), name="profile repeat")
        if model not in normalized_models or order != "canonical":
            raise ProfileError("profile sample method or order differs")
        if horizon not in normalized_horizons or repeat >= repeats:
            raise ProfileError("profile sample horizon or repeat differs")
        if reference in masters and masters[reference] != master:
            raise ProfileError("profile reference uses more than one master")
        masters[reference] = master
        references.add(reference)
        key = (model, reference, horizon, repeat)
        if key in observed:
            raise ProfileError("profile coverage contains a duplicate sample")
        observed.add(key)
        _positive_float(row.get("latency_ms"), name="profile latency")
        for field in (
            "start_allocated_bytes",
            "peak_allocated_bytes",
            "incremental_peak_allocated_bytes",
            "window_voxel_points",
            "window_segments",
        ):
            _integer(row.get(field), name=field, minimum=int(field.startswith("window_")))
        if int(row["peak_allocated_bytes"]) < int(row["start_allocated_bytes"]):
            raise ProfileError("profile peak memory is below starting allocation")
        if int(row["incremental_peak_allocated_bytes"]) != (
            int(row["peak_allocated_bytes"]) - int(row["start_allocated_bytes"])
        ):
            raise ProfileError("profile incremental peak memory differs")
    expected = {
        (model, reference, horizon, repeat)
        for model in normalized_models
        for reference in references
        for horizon in normalized_horizons
        for repeat in range(repeats)
    }
    if (
        len(references) != expected_reference_count
        or len(masters) != expected_reference_count
        or observed != expected
    ):
        raise ProfileError("profile sample coverage differs")


def summarize_profile_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    models: Sequence[str] = PROFILE_MODELS,
    expected_reference_count: int = 6,
    horizons: Sequence[int] = REPORT_HORIZONS,
    repeats: int = 10,
) -> list[dict[str, object]]:
    validate_profile_samples(
        rows,
        models=models,
        expected_reference_count=expected_reference_count,
        horizons=horizons,
        repeats=repeats,
    )
    groups: dict[tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["model"]), str(row["reference_scene_id"]), int(row["T"]))].append(
            row
        )
    output = []
    for model in models:
        model_groups = sorted(key for key in groups if key[0] == model)
        for key in model_groups:
            samples = groups[key]
            latencies = [float(row["latency_ms"]) for row in samples]
            first = samples[0]
            for field in ("master_sequence_id", "order_id", "window_voxel_points", "window_segments"):
                if len({row[field] for row in samples}) != 1:
                    raise ProfileError(f"profile summary {field} differs within a cell")
            output.append(
                {
                    "model": key[0],
                    "reference_scene_id": key[1],
                    "master_sequence_id": first["master_sequence_id"],
                    "order_id": first["order_id"],
                    "T": key[2],
                    "measured_repeats": len(samples),
                    "median_latency_ms": statistics.median(latencies),
                    "minimum_latency_ms": min(latencies),
                    "maximum_latency_ms": max(latencies),
                    "peak_allocated_bytes": max(
                        int(row["peak_allocated_bytes"]) for row in samples
                    ),
                    "maximum_incremental_peak_allocated_bytes": max(
                        int(row["incremental_peak_allocated_bytes"])
                        for row in samples
                    ),
                    "window_voxel_points": first["window_voxel_points"],
                    "window_segments": first["window_segments"],
                }
            )
    return output


@dataclass(frozen=True)
class _PreparedInput:
    data: object
    target: Mapping[str, object]
    raw_coordinates: object
    stages: object
    latest_local_stage: int
    window_voxel_points: int
    window_segments: int


def _prepare_input(
    *,
    system: object,
    dataset: object,
    collate: object,
    request: Mapping[str, object],
    device: object,
) -> _PreparedInput:
    import torch

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )

    sample = dataset.load_scan_indices(
        int(request["context_index"]),
        tuple(request["inference_scan_indices"]),
        change_file=None,
    )
    data, targets, names = collate([sample])
    if list(names) != [request["master_sequence_id"]] or len(targets) != 1:
        raise ProfileError("profile collator changed the requested sequence")
    data = _move_data_to_device(data, device)
    targets = _move_targets_to_device(targets, device)
    target = targets[0]
    point2segment = target.get("point2segment")
    if not isinstance(point2segment, torch.Tensor) or point2segment.numel() == 0:
        raise ProfileError("profile target lacks point-to-segment mapping")
    stages = _segment_stages(target)
    latest = int(stages.max().item())
    return _PreparedInput(
        data=data,
        target=target,
        raw_coordinates=system._process_raw_coordinates(data),
        stages=stages,
        latest_local_stage=latest,
        window_voxel_points=int(point2segment.numel()),
        window_segments=int(point2segment.max().item()) + 1,
    )


def _c2_step(
    *,
    system: object,
    prepared: _PreparedInput,
    state: object,
    read_state: object,
) -> tuple[object, object]:
    import torch

    from models.persistent_memory import build_local_observation
    from trainer.persist4d_allt_trainer import build_detached_memory_read_state

    with torch.inference_mode():
        output = system(
            prepared.data,
            point2segment=[prepared.target["point2segment"]],
            raw_coordinates=prepared.raw_coordinates,
            is_eval=True,
            memory_read_state=read_state,
        )
        observation = build_local_observation(
            output,
            [prepared.stages],
            latest_stage=prepared.latest_local_stage,
            **system.observation_settings,
        )
        if state is None:
            state = system.persistent_memory.empty_state(observation)
        step = system.persistent_memory.step(
            observation,
            state,
            stage_index=int(state.stage_watermark.max().item()) + 1,
        )
        next_state = step.state.detach()
        next_read_state = build_detached_memory_read_state(next_state)
    return next_state, next_read_state


def _fh_forward(*, system: object, prepared: _PreparedInput) -> object:
    import torch

    with torch.inference_mode():
        return system(
            prepared.data,
            point2segment=[prepared.target["point2segment"]],
            raw_coordinates=prepared.raw_coordinates,
            is_eval=True,
            memory_read_state=None,
        )


def _measure(
    *,
    operation: Callable[[], object],
    device: object,
    warmups: int,
    repeats: int,
) -> list[dict[str, object]]:
    import torch

    torch.cuda.empty_cache()
    for _ in range(warmups):
        result = operation()
        torch.cuda.synchronize(device)
        del result
    samples = []
    for repeat in range(repeats):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start_allocated = int(torch.cuda.memory_allocated(device))
        started = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        del result
        samples.append(
            {
                "repeat": repeat,
                "latency_ms": elapsed_ms,
                "start_allocated_bytes": start_allocated,
                "peak_allocated_bytes": peak_allocated,
                "incremental_peak_allocated_bytes": max(
                    0, peak_allocated - start_allocated
                ),
            }
        )
    return samples


def _profile_c2_sequence(
    *,
    system: object,
    dataset: object,
    collate: object,
    sequence: object,
    device: object,
    horizons: Sequence[int],
    warmups: int,
    repeats: int,
) -> list[dict[str, object]]:
    requests = _sequence(sequence.stage_requests, name="C2 stage requests")
    reference, master, order = _sequence_identity(sequence)
    state = None
    read_state = None
    output = []
    for stage, request in enumerate(requests):
        horizon = stage + 1
        if horizon > max(horizons):
            break
        prepared = _prepare_input(
            system=system,
            dataset=dataset,
            collate=collate,
            request=request,
            device=device,
        )
        if horizon in horizons:
            measured = _measure(
                operation=lambda prepared=prepared, state=state, read_state=read_state: (
                    _c2_step(
                        system=system,
                        prepared=prepared,
                        state=state,
                        read_state=read_state,
                    )
                ),
                device=device,
                warmups=warmups,
                repeats=repeats,
            )
            output.extend(
                {
                    "model": "C2",
                    "reference_scene_id": reference,
                    "master_sequence_id": master,
                    "order_id": order,
                    "T": horizon,
                    **sample,
                    "window_voxel_points": prepared.window_voxel_points,
                    "window_segments": prepared.window_segments,
                }
                for sample in measured
            )
        if horizon < max(horizons):
            state, read_state = _c2_step(
                system=system,
                prepared=prepared,
                state=state,
                read_state=read_state,
            )
    return output


def _profile_fh_sequence(
    *,
    system: object,
    dataset: object,
    collate: object,
    sequence: object,
    device: object,
    horizons: Sequence[int],
    warmups: int,
    repeats: int,
) -> list[dict[str, object]]:
    reference, master, order = _sequence_identity(sequence)
    output = []
    for stage, request in enumerate(sequence.stage_requests):
        horizon = stage + 1
        if horizon not in horizons:
            continue
        prepared = _prepare_input(
            system=system,
            dataset=dataset,
            collate=collate,
            request=request,
            device=device,
        )
        measured = _measure(
            operation=lambda prepared=prepared: _fh_forward(
                system=system, prepared=prepared
            ),
            device=device,
            warmups=warmups,
            repeats=repeats,
        )
        output.extend(
            {
                "model": "FH-adapt",
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
                "T": horizon,
                **sample,
                "window_voxel_points": prepared.window_voxel_points,
                "window_segments": prepared.window_segments,
            }
            for sample in measured
        )
    return output


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise ProfileError("profile CSV fields differ")
        writer.writerow(row)
    return stream.getvalue().encode("ascii")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProfileError("cannot resolve source commit") from error
    if len(value) != 40:
        raise ProfileError("source commit is invalid")
    return value


def run_profile(
    *,
    c2_checkpoint: Path,
    fh_checkpoint: Path,
    device_name: str,
    pretrained: Path,
    metadata: Path,
    data_root: Path,
    split_manifest: Path,
    scratch_root: Path,
    output_root: Path,
    maximum_references: int | None = None,
    horizons: Sequence[int] = REPORT_HORIZONS,
    warmups: int = 5,
    repeats: int = 10,
) -> dict[str, object]:
    import torch

    from scripts.evaluate_persist4d_allt import (
        _file_sha256 as evaluation_file_sha256,
    )
    from scripts.evaluate_persist4d_allt import _load_system, _population
    from scripts.system_comparison_inference import deterministic_inference_runtime

    started = time.time()
    source_commit = _git_head()
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons) or any(
        value not in REPORT_HORIZONS for value in horizons
    ):
        raise ProfileError("profile horizons must be unique members of T2-T5")
    warmups = _integer(warmups, name="profile warmups")
    repeats = _integer(repeats, name="profile repeats", minimum=1)
    if not torch.cuda.is_available():
        raise ProfileError("CUDA is required for resource profiling")
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
        raise ProfileError("profile CUDA device is unavailable")
    device_label = torch.cuda.get_device_name(device)
    if "A40" not in device_label:
        raise ProfileError("formal resource profile requires an NVIDIA A40")
    checkpoint_paths = {"C2": c2_checkpoint, "FH-adapt": fh_checkpoint}
    for model, path in checkpoint_paths.items():
        resolved = path.expanduser().resolve()
        if resolved.is_symlink() or not resolved.is_file():
            raise ProfileError(f"{model} checkpoint is unavailable")
        if evaluation_file_sha256(resolved) != FROZEN_CHECKPOINTS[model]:
            raise ProfileError(f"{model} checkpoint hash differs")
        checkpoint_paths[model] = resolved
    rows = []
    selected_identities: dict[str, list[dict[str, str]]] = {}
    population_hashes = {}
    for model in PROFILE_MODELS:
        system, config, checkpoint_step, training_seed = _load_system(
            variant=model,
            checkpoint=checkpoint_paths[model],
            pretrained=pretrained.expanduser().resolve(),
            device=device,
            scratch=scratch_root.expanduser().resolve() / model,
        )
        if training_seed != 45 or checkpoint_step != (200 if model == "C2" else 100):
            raise ProfileError(f"{model} checkpoint metadata differs")
        window_mode = "local_pair" if model == "C2" else "full_history"
        dataset, collate, sequences, population_id, population_sha256 = _population(
            config=config,
            population="protocol_b",
            window_mode=window_mode,
            data_root=data_root.expanduser().resolve(),
            metadata=metadata.expanduser().resolve(),
            split_manifest=split_manifest.expanduser().resolve(),
        )
        if population_id != "protocol_b_43_masters_3_orders":
            raise ProfileError("profile population differs")
        selected = select_profile_sequences(sequences)
        if maximum_references is not None:
            maximum_references = _integer(
                maximum_references, name="maximum references", minimum=1
            )
            if maximum_references >= len(selected):
                raise ProfileError("maximum references must define a strict smoke subset")
            selected = selected[:maximum_references]
        selected_identities[model] = [
            {
                "reference_scene_id": _sequence_identity(item)[0],
                "master_sequence_id": _sequence_identity(item)[1],
                "order_id": _sequence_identity(item)[2],
            }
            for item in selected
        ]
        population_hashes[model] = population_sha256
        profiler = _profile_c2_sequence if model == "C2" else _profile_fh_sequence
        with deterministic_inference_runtime(45, device):
            for position, sequence in enumerate(selected, start=1):
                rows.extend(
                    profiler(
                        system=system,
                        dataset=dataset,
                        collate=collate,
                        sequence=sequence,
                        device=device,
                        horizons=horizons,
                        warmups=warmups,
                        repeats=repeats,
                    )
                )
                print(
                    f"[allt-profile] {model} {position}/{len(selected)} references",
                    flush=True,
                )
        del system, dataset, collate
        gc.collect()
        torch.cuda.empty_cache()
    if selected_identities["C2"] != selected_identities["FH-adapt"]:
        raise ProfileError("profile methods selected different masters")
    reference_count = len(selected_identities["C2"])
    validate_profile_samples(
        rows,
        expected_reference_count=reference_count,
        horizons=horizons,
        repeats=repeats,
    )
    summary = summarize_profile_samples(
        rows,
        expected_reference_count=reference_count,
        horizons=horizons,
        repeats=repeats,
    )
    sample_content = _csv_bytes(rows, SAMPLE_FIELDS)
    summary_content = _csv_bytes(summary, SUMMARY_FIELDS)
    _atomic_write(output_root / "samples.csv", sample_content)
    _atomic_write(output_root / "summary.csv", summary_content)
    run_summary = {
        "accuracy_role": "diagnostic_only_cannot_rescue_the_failed_accuracy_verdict",
        "checkpoints": FROZEN_CHECKPOINTS,
        "device_index": device.index,
        "device_name": device_label,
        "elapsed_seconds": time.time() - started,
        "evaluation_seed": 45,
        "excluded_scope": [
            "file_io",
            "collation",
            "host_to_device_transfer",
            "metric_computation",
            "C2_causal_preroll",
        ],
        "horizons": list(horizons),
        "included_scope": {
            "C2": "current model forward including memory read, observation extraction, and one B4 update",
            "FH-adapt": "one current full-prefix model forward",
        },
        "measured_repeats": repeats,
        "population_id": (
            "protocol_b_profile_6_reference_canonical"
            if maximum_references is None
            else f"smoke_protocol_b_profile_{reference_count}_reference_canonical"
        ),
        "protocol_b_population_sha256": population_hashes,
        "sample_row_count": len(rows),
        "samples_sha256": hashlib.sha256(sample_content).hexdigest(),
        "schema_version": 1,
        "selected_sequences": selected_identities["C2"],
        "source_commit": source_commit,
        "status": "pass",
        "summary_row_count": len(summary),
        "summary_sha256": hashlib.sha256(summary_content).hexdigest(),
        "warmup_repeats": warmups,
    }
    _atomic_json(output_root / "run_summary.json", run_summary)
    if _git_head() != source_commit:
        raise ProfileError("Git HEAD changed during resource profiling")
    return run_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c2-checkpoint",
        type=Path,
        default=DEFAULT_TRAINING_ROOT / "C2/update=0200.ckpt",
    )
    parser.add_argument(
        "--fh-checkpoint",
        type=Path,
        default=DEFAULT_TRAINING_ROOT / "FH-adapt/update=0100.ckpt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument(
        "--scratch-root", type=Path, default=Path("/tmp/persist4d-allt-profile")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--maximum-references", type=int)
    parser.add_argument("--horizons", type=int, nargs="+", default=REPORT_HORIZONS)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_profile(
        c2_checkpoint=args.c2_checkpoint,
        fh_checkpoint=args.fh_checkpoint,
        device_name=args.device,
        pretrained=args.pretrained,
        metadata=args.metadata,
        data_root=args.data_root,
        split_manifest=args.split_manifest,
        scratch_root=args.scratch_root,
        output_root=args.output_root,
        maximum_references=args.maximum_references,
        horizons=args.horizons,
        warmups=args.warmups,
        repeats=args.repeats,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "PROFILE_MODELS",
    "REPORT_HORIZONS",
    "SAMPLE_FIELDS",
    "SUMMARY_FIELDS",
    "ProfileError",
    "run_profile",
    "select_profile_sequences",
    "summarize_profile_samples",
    "validate_profile_samples",
]


if __name__ == "__main__":
    raise SystemExit(main())
