#!/usr/bin/env python3
"""Profile the locked Perception Gain deployment methods on live PB inputs."""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
DEFAULT_ASSETS = DEFAULT_EXTERNAL_ROOT / "assets.local.json"
HORIZONS = (2, 3, 4, 5)
WARMUP_REPEATS = 1
MEASURED_REPEATS = 3

SAMPLE_FIELDS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "repeat",
    "network_forward_ms",
    "state_update_ms",
    "materialize_ms",
    "end_to_end_ms",
    "prefix_cumulative_ms",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "cpu_rss_bytes",
    "resident_state_bytes",
    "soft_buffer_bytes",
    "archive_bytes",
    "window_voxel_points",
    "window_segments",
)


class ProfileError(RuntimeError):
    """Raised when the frozen live-profile contract is violated."""


@dataclass(frozen=True)
class ProfileMethod:
    method_id: str
    mode: str
    variant: str
    optimizer_update: int
    checkpoint_reference: str
    checkpoint_sha256: str
    refiner_enabled: bool = False
    refiner_update: int | None = None
    refiner_checkpoint_reference: str | None = None
    refiner_checkpoint_sha256: str | None = None


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileError(f"{name} must be finite and nonnegative")
    return result


def build_profile_methods(
    final_lock: Mapping[str, object],
) -> tuple[ProfileMethod, ...]:
    recipe = final_lock.get("recipe")
    if (
        final_lock.get("schema_version") != "perception-gain-final-lock-v1"
        or final_lock.get("status") != "LOCKED"
        or not isinstance(recipe, Mapping)
    ):
        raise ProfileError("final lock is unavailable")
    variant = _text(recipe.get("parent_variant"), name="parent variant")
    update = _integer(recipe.get("parent_optimizer_update"), name="parent update")
    parent_checkpoint = _text(recipe.get("parent_checkpoint"), name="parent checkpoint")
    parent_sha = _text(
        recipe.get("parent_checkpoint_sha256"), name="parent checkpoint SHA256"
    )
    if len(parent_sha) != 64:
        raise ProfileError("parent checkpoint SHA256 differs")
    refiner_enabled = recipe.get("refiner_enabled")
    if not isinstance(refiner_enabled, bool):
        raise ProfileError("refiner selection differs")
    refiner_update = recipe.get("refiner_optimizer_update")
    refiner_checkpoint = recipe.get("refiner_checkpoint")
    refiner_sha = recipe.get("refiner_checkpoint_sha256")
    if refiner_enabled and (
        not isinstance(refiner_update, int)
        or refiner_update <= 0
        or not isinstance(refiner_checkpoint, str)
        or not refiner_checkpoint
        or not isinstance(refiner_sha, str)
        or len(refiner_sha) != 64
    ):
        raise ProfileError("selected refiner identity differs")
    r1_sha = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
    methods = [
        ProfileMethod(
            "FH-R1-native", "native", "C0", 0, "external:r1_checkpoint", r1_sha
        ),
        ProfileMethod("D0-R1", "d0", "C0", 0, "external:r1_checkpoint", r1_sha),
        ProfileMethod(
            "FINAL",
            "d0",
            variant,
            update,
            parent_checkpoint,
            parent_sha,
            refiner_enabled,
            refiner_update if refiner_enabled else None,
            refiner_checkpoint if refiner_enabled else None,
            refiner_sha if refiner_enabled else None,
        ),
    ]
    if variant != "C0" or update != 0:
        methods.append(
            ProfileMethod(
                "FH-P*-native",
                "native",
                variant,
                update,
                parent_checkpoint,
                parent_sha,
            )
        )
    return tuple(methods)


def select_profile_specs(specs: Sequence[object]) -> tuple[object, ...]:
    grouped: dict[str, list[object]] = defaultdict(list)
    for spec in specs:
        reference = getattr(spec, "reference_id", None)
        sequence = getattr(spec, "source_sequence_id", None)
        context_index = getattr(spec, "context_index", None)
        draw_index = getattr(spec, "draw_index", None)
        canonical = (
            context_index % 3 == 0
            if isinstance(context_index, int)
            else draw_index == 0
        )
        if canonical and isinstance(reference, str) and isinstance(sequence, str):
            grouped[reference].append(spec)
    if len(grouped) != 6:
        raise ProfileError("profile selection must cover six PB references")
    selected = []
    for reference in sorted(grouped):
        candidates = grouped[reference]
        selected.append(min(candidates, key=lambda item: item.source_sequence_id))
    return tuple(selected)


def validate_profile_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str],
    repeats: int = MEASURED_REPEATS,
) -> None:
    method_ids = tuple(_text(value, name="profile method") for value in methods)
    repeats = _integer(repeats, name="profile repeats", minimum=1)
    observed: set[tuple[str, str, int, int]] = set()
    references: set[str] = set()
    masters: dict[str, str] = {}
    for row in rows:
        method = _text(row.get("method"), name="profile method")
        reference = _text(row.get("reference_scene_id"), name="profile reference")
        master = _text(row.get("master_sequence_id"), name="profile master")
        horizon = _integer(row.get("T"), name="profile horizon", minimum=1)
        repeat = _integer(row.get("repeat"), name="profile repeat")
        if (
            method not in method_ids
            or row.get("order_id") != "canonical"
            or horizon not in HORIZONS
            or repeat >= repeats
        ):
            raise ProfileError("profile row identity differs")
        key = (method, reference, horizon, repeat)
        if key in observed:
            raise ProfileError("profile coverage contains a duplicate row")
        observed.add(key)
        references.add(reference)
        if reference in masters and masters[reference] != master:
            raise ProfileError("profile reference uses multiple masters")
        masters[reference] = master
        for field in (
            "network_forward_ms",
            "state_update_ms",
            "materialize_ms",
            "end_to_end_ms",
            "prefix_cumulative_ms",
        ):
            _number(row.get(field), name=field)
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "cpu_rss_bytes",
            "resident_state_bytes",
            "soft_buffer_bytes",
            "archive_bytes",
            "window_voxel_points",
            "window_segments",
        ):
            _integer(row.get(field), name=field)
        if int(row["peak_reserved_bytes"]) < int(row["peak_allocated_bytes"]):
            raise ProfileError("profile reserved memory is below allocated memory")
        component_ms = sum(
            float(row[field])
            for field in ("network_forward_ms", "state_update_ms", "materialize_ms")
        )
        if float(row["end_to_end_ms"]) + 1e-6 < component_ms:
            raise ProfileError("profile end-to-end latency excludes a component")
    expected = {
        (method, reference, horizon, repeat)
        for method in method_ids
        for reference in references
        for horizon in HORIZONS
        for repeat in range(repeats)
    }
    if len(references) != 6 or observed != expected:
        raise ProfileError("profile coverage differs")


def summarize_profile_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    methods: Sequence[str],
    repeats: int = MEASURED_REPEATS,
) -> list[dict[str, object]]:
    validate_profile_rows(rows, methods=methods, repeats=repeats)
    grouped: dict[tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["method"]), str(row["reference_scene_id"]), int(row["T"]))
        ].append(row)
    output = []
    timing_fields = (
        "network_forward_ms",
        "state_update_ms",
        "materialize_ms",
        "end_to_end_ms",
        "prefix_cumulative_ms",
    )
    byte_fields = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "cpu_rss_bytes",
        "resident_state_bytes",
        "soft_buffer_bytes",
        "archive_bytes",
    )
    for method in methods:
        for key in sorted(value for value in grouped if value[0] == method):
            samples = grouped[key]
            row: dict[str, object] = {
                "method": key[0],
                "reference_scene_id": key[1],
                "master_sequence_id": samples[0]["master_sequence_id"],
                "order_id": "canonical",
                "T": key[2],
                "measured_repeats": len(samples),
            }
            for field in timing_fields:
                values = [float(sample[field]) for sample in samples]
                row[f"median_{field}"] = statistics.median(values)
                row[f"minimum_{field}"] = min(values)
                row[f"maximum_{field}"] = max(values)
            for field in byte_fields:
                row[f"maximum_{field}"] = max(int(sample[field]) for sample in samples)
            output.append(row)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise ProfileError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _external_path(
    reference: str, *, assets: Mapping[str, object], external_root: Path
) -> Path:
    if reference == "external:r1_checkpoint":
        value = assets.get("r1_checkpoint")
        if not isinstance(value, str):
            raise ProfileError("R1 checkpoint asset is unavailable")
        return Path(value).expanduser().resolve(strict=True)
    if not reference.startswith("external:"):
        raise ProfileError("profile checkpoint reference is not external")
    relative = reference.removeprefix("external:")
    path = (external_root / relative).resolve(strict=True)
    try:
        path.relative_to(external_root)
    except ValueError as error:
        raise ProfileError("profile checkpoint escapes external root") from error
    return path


def _storage_bytes(value: object) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_storage_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_storage_bytes(item) for item in value)
    tensors = getattr(value, "tensors", None)
    if callable(tensors):
        return sum(_storage_bytes(item) for item in tensors())
    return 0


def _current_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
    except (OSError, IndexError, ValueError) as error:
        raise ProfileError("cannot measure CPU RSS") from error
    return pages * os.sysconf("SC_PAGE_SIZE")


def _elapsed_ms(
    device: object, operation: Callable[[], object]
) -> tuple[object, float]:
    import torch

    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


class _TimedTransform:
    def __init__(self, transform: object, device: object) -> None:
        self.transform = transform
        self.device = device
        self.elapsed_ms = 0.0

    def observe_stage(self, prediction: object, stage_meta: object) -> None:
        self.transform.observe_stage(prediction, stage_meta)

    def __call__(self, request: object) -> object:
        result, elapsed = _elapsed_ms(self.device, lambda: self.transform(request))
        self.elapsed_ms += elapsed
        return result


def _load_system(
    method: ProfileMethod,
    *,
    assets: Mapping[str, object],
    external_root: Path,
    device: object,
) -> tuple[object, object, Path | None]:
    from scripts.perception_gain_evaluation import _load_evaluation_weights
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    pretrained = assets.get("concerto_pretrained")
    r1_checkpoint = assets.get("r1_checkpoint")
    if not isinstance(pretrained, str) or not isinstance(r1_checkpoint, str):
        raise ProfileError("profile model assets are unavailable")
    run_dir = external_root / "profile/runtime" / method.method_id
    config = compose_variant_config(
        method.variant, pretrained=Path(pretrained), run_dir=run_dir
    )
    config.model.return_query_features = True
    scorer = external_root / "training/scorer/model/update=0500.ckpt"
    scorer_checkpoint = scorer if method.variant == "Q-SEM" else None
    if method.variant == "Q-SEM":
        config.perception_training.scorer_checkpoint = str(scorer)
    checkpoint = (
        None
        if method.optimizer_update == 0
        else _external_path(
            method.checkpoint_reference, assets=assets, external_root=external_root
        )
    )
    system = PerceptionGainTrainer(config)
    _, observed_sha, _, _, _ = _load_evaluation_weights(
        system=system,
        variant=method.variant,
        optimizer_update=method.optimizer_update,
        r1_checkpoint=Path(r1_checkpoint),
        checkpoint=checkpoint,
        scorer_checkpoint=scorer_checkpoint,
    )
    if observed_sha != method.checkpoint_sha256:
        raise ProfileError(f"{method.method_id} checkpoint SHA256 differs")
    system.to(device).eval().requires_grad_(False)
    return system, config, scorer_checkpoint


def _profile_native_sequence(
    *,
    method: ProfileMethod,
    system: object,
    dataset: object,
    collator: object,
    spec: object,
    class_mapper: object,
    observation_settings: Mapping[str, object],
    device: object,
    repeat: int,
) -> list[dict[str, object]]:
    import torch

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
    )
    from scripts.system_comparison_inference import postprocess_full_history_output

    raw_samples = [
        dataset.load_scan_indices(
            spec.context_index, tuple(spec.scan_indices[:horizon]), change_file=None
        )
        for horizon in range(1, 6)
    ]
    cumulative = 0.0
    rows = []
    for horizon, sample in enumerate(raw_samples, start=1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        data, targets, names = collator([sample])
        if list(names) != ["-".join(spec.scan_ids[:horizon])]:
            raise ProfileError("native profile collator changed sequence identity")
        full_targets = getattr(data, "target_full", None)
        if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
            raise ProfileError("native profile lacks a full target")
        full_target = full_targets[0]
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target = targets[0]
        raw_coordinates = system._process_raw_coordinates(data)

        def forward(
            data: object = data,
            target: Mapping[str, object] = target,
            raw_coordinates: object = raw_coordinates,
        ) -> object:
            with torch.inference_mode():
                return system(
                    data,
                    point2segment=[target["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )

        output, network_ms = _elapsed_ms(device, forward)

        def materialize(
            output: object = output,
            target: Mapping[str, object] = target,
            full_target: Mapping[str, object] = full_target,
            data: object = data,
            horizon: int = horizon,
        ) -> object:
            return postprocess_full_history_output(
                system=system,
                output=output,
                target_low_resolution=target,
                target_full_resolution=full_target,
                data=data,
                horizon=horizon,
                class_mapper=class_mapper,
                **observation_settings,
            )

        _, materialize_ms = _elapsed_ms(
            device,
            materialize,
        )
        torch.cuda.synchronize(device)
        end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        cumulative += end_to_end_ms
        if horizon in HORIZONS:
            point2segment = target["point2segment"]
            rows.append(
                {
                    "method": method.method_id,
                    "reference_scene_id": spec.reference_id,
                    "master_sequence_id": spec.source_sequence_id,
                    "order_id": "canonical",
                    "T": horizon,
                    "repeat": repeat,
                    "network_forward_ms": network_ms,
                    "state_update_ms": 0.0,
                    "materialize_ms": materialize_ms,
                    "end_to_end_ms": end_to_end_ms,
                    "prefix_cumulative_ms": cumulative,
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "cpu_rss_bytes": _current_rss_bytes(),
                    "resident_state_bytes": 0,
                    "soft_buffer_bytes": 0,
                    "archive_bytes": 0,
                    "window_voxel_points": int(point2segment.numel()),
                    "window_segments": int(point2segment.max().item()) + 1,
                }
            )
        del data, targets, target, output
    return rows


def _profile_d0_sequence(
    *,
    method: ProfileMethod,
    system: object,
    base_collator: object,
    episode: object,
    spec: object,
    class_mapper: object,
    observation_settings: Mapping[str, object],
    refiner: object | None,
    device: object,
    repeat: int,
) -> list[dict[str, object]]:
    import torch

    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import _segment_stages
    from scripts.prepare_perception_refiner import advance_d0_identity
    from scripts.profile_task_memory import _prepare_sample
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_task_memory_controls import _window_observation
    from scripts.task_memory_output import LagOnePublisher
    from scripts.train_perception_refiner import CausalRefinerRevisionTransform

    timed_transform = None
    if refiner is not None:
        timed_transform = _TimedTransform(
            CausalRefinerRevisionTransform(
                system=system, refiner=refiner, device=device
            ),
            device,
        )
    publisher = LagOnePublisher(
        score_reducer="mean",
        iou_threshold=0.5,
        revision_mask_transform=timed_transform,
    )
    state = None
    cumulative = 0.0
    rows = []
    for stage_index, sample in enumerate(episode.stage_samples):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        prepared = _prepare_sample(
            sample=sample,
            spec=spec,
            stage_collator=base_collator,
            system=system,
            device=device,
        )

        def forward(prepared: object = prepared) -> object:
            with torch.inference_mode():
                return system(
                    prepared.data,
                    point2segment=[prepared.target_low["point2segment"]],
                    raw_coordinates=prepared.raw_coordinates,
                    is_eval=True,
                )

        output, network_ms = _elapsed_ms(device, forward)

        def update_state(
            output: object = output,
            prepared: object = prepared,
            prior_state: object = state,
        ) -> tuple[object, object]:
            local = build_local_observation(
                output,
                [_segment_stages(prepared.target_low)],
                latest_stage=prepared.latest_stage,
                **observation_settings,
            )
            observation = _window_observation(
                local_observation=local,
                output=output,
                segment_stages=_segment_stages(prepared.target_low),
                latest_stage=prepared.latest_stage,
                confidence_threshold=float(
                    observation_settings["confidence_threshold"]
                ),
                mask_threshold=float(observation_settings["mask_threshold"]),
                minimum_mask_support=int(observation_settings["minimum_mask_support"]),
            )
            return advance_d0_identity(
                observation=observation,
                state=prior_state,
                stage_meta=prepared.meta,
            )

        (state, identity_map), state_ms = _elapsed_ms(device, update_state)

        def materialize(
            output: object = output,
            prepared: object = prepared,
            identity_map: Mapping[int, object] = identity_map,
        ) -> object:
            prediction = extract_official_task_prediction(
                system=system,
                output=output,
                target_low_resolution=prepared.target_low,
                target_full_resolution=prepared.full_target,
                data=prepared.data,
                class_mapper=class_mapper,
                latest_stage_index=prepared.latest_stage,
                return_soft_evidence=refiner is not None,
            )
            return publisher.update(prediction, identity_map, prepared.meta)

        refiner_before = timed_transform.elapsed_ms if timed_transform else 0.0
        published, combined_materialize_ms = _elapsed_ms(device, materialize)
        refiner_ms = (
            timed_transform.elapsed_ms - refiner_before if timed_transform else 0.0
        )
        network_ms += refiner_ms
        materialize_ms = max(0.0, combined_materialize_ms - refiner_ms)
        torch.cuda.synchronize(device)
        end_to_end_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        cumulative += end_to_end_ms
        horizon = stage_index + 1
        if horizon in HORIZONS:
            point2segment = prepared.target_low["point2segment"]
            rows.append(
                {
                    "method": method.method_id,
                    "reference_scene_id": spec.reference_id,
                    "master_sequence_id": spec.source_sequence_id,
                    "order_id": "canonical",
                    "T": horizon,
                    "repeat": repeat,
                    "network_forward_ms": network_ms,
                    "state_update_ms": state_ms,
                    "materialize_ms": materialize_ms,
                    "end_to_end_ms": end_to_end_ms,
                    "prefix_cumulative_ms": cumulative,
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "cpu_rss_bytes": _current_rss_bytes(),
                    "resident_state_bytes": _storage_bytes(state),
                    "soft_buffer_bytes": int(published.accounting.lag1_buffer_bytes),
                    "archive_bytes": int(published.accounting.archive_payload_bytes),
                    "window_voxel_points": int(point2segment.numel()),
                    "window_segments": int(point2segment.max().item()) + 1,
                }
            )
        del prepared, output, published
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise ProfileError("profile CSV fields differ")
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_profile(
    *,
    final_lock_path: Path,
    assets_path: Path,
    external_root: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, object]:
    import hydra
    import torch
    import yaml

    from datasets.task_memory_episode import TaskMemoryEpisodeDataset
    from scripts.evaluate_persist4d import _validate_cuda_device
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.perception_gain_evaluation import (
        DATA_CONTRACT,
        build_role_base_dataset,
    )
    from scripts.perception_gain_foundation import _resolve_cache_assets
    from scripts.run_task_memory_policy_baseline import (
        PROTOCOL_B_POPULATION_ID,
        _episode_specs,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime

    started = time.perf_counter()
    final_lock_path = final_lock_path.expanduser().resolve(strict=True)
    assets_path = assets_path.expanduser().resolve(strict=True)
    external_root = external_root.expanduser().resolve(strict=True)
    final_lock = _read_json(final_lock_path)
    methods = build_profile_methods(final_lock)
    assets = _resolve_cache_assets(assets_path)
    device = _validate_cuda_device(device_name)
    device_label = torch.cuda.get_device_name(device)
    if "A40" not in device_label:
        raise ProfileError("formal profile requires an NVIDIA A40")
    p6a = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())
    settings = p6a["baselines"]["b4"]
    observation_settings = {
        "background_class": int(settings["background_class"]),
        "confidence_threshold": float(settings["confidence_threshold"]),
        "mask_threshold": float(settings["mask_threshold"]),
        "minimum_mask_support": int(settings["minimum_mask_support"]),
    }
    rows = []
    selected_identity: list[dict[str, object]] | None = None
    population_hash: str | None = None
    for method in methods:
        system, config, _ = _load_system(
            method, assets=assets, external_root=external_root, device=device
        )
        base = build_role_base_dataset(
            config=config,
            data_root=Path(str(assets["data_root"])),
            role="PB",
            horizon=5,
        )
        dataset, masters, current_population_hash = build_baseline_population(
            base,
            data_contract=_read_json(DATA_CONTRACT),
            metadata_path=Path(str(assets["rio_metadata"])),
            population_id=PROTOCOL_B_POPULATION_ID,
            protocol_b_manifest_path=PROJECT_ROOT
            / "artifacts/P6A/protocol_b_manifest.json",
        )
        specs = _episode_specs(masters)
        selected = select_profile_specs(specs)
        identity = [
            {
                "reference_scene_id": spec.reference_id,
                "master_sequence_id": spec.source_sequence_id,
                "order_id": "canonical",
                "context_index": spec.context_index,
            }
            for spec in selected
        ]
        if selected_identity is None:
            selected_identity = identity
            population_hash = current_population_hash
        elif (
            identity != selected_identity or current_population_hash != population_hash
        ):
            raise ProfileError("profile methods selected different PB inputs")
        base_collator = hydra.utils.instantiate(config.data.validation_collation)
        class_mapper = build_rio_class_mapper(dataset)
        refiner = None
        if method.refiner_enabled:
            from scripts.perception_refiner_evaluation import _load_refiner

            refiner_path = _external_path(
                _text(
                    method.refiner_checkpoint_reference,
                    name="refiner checkpoint reference",
                ),
                assets=assets,
                external_root=external_root,
            )
            refiner, identity_record = _load_refiner(
                refiner_path,
                optimizer_update=int(method.refiner_update),
                device=device,
            )
            if identity_record["sha256"] != method.refiner_checkpoint_sha256:
                raise ProfileError("profile refiner SHA256 differs")
        with deterministic_inference_runtime(45, device):
            for position, spec in enumerate(selected, start=1):
                episode = TaskMemoryEpisodeDataset(
                    dataset, (spec,), apply_augmentation=False
                )[0]
                for run_index in range(WARMUP_REPEATS + MEASURED_REPEATS):
                    measured_repeat = run_index - WARMUP_REPEATS
                    if method.mode == "native":
                        measured = _profile_native_sequence(
                            method=method,
                            system=system,
                            dataset=dataset,
                            collator=base_collator,
                            spec=spec,
                            class_mapper=class_mapper,
                            observation_settings=observation_settings,
                            device=device,
                            repeat=measured_repeat,
                        )
                    else:
                        measured = _profile_d0_sequence(
                            method=method,
                            system=system,
                            base_collator=base_collator,
                            episode=episode,
                            spec=spec,
                            class_mapper=class_mapper,
                            observation_settings=observation_settings,
                            refiner=refiner,
                            device=device,
                            repeat=measured_repeat,
                        )
                    if run_index >= WARMUP_REPEATS:
                        rows.extend(measured)
                print(
                    f"[perception-profile] {method.method_id} {position}/6",
                    flush=True,
                )
        del system, dataset, base, base_collator
        if refiner is not None:
            del refiner
        gc.collect()
        torch.cuda.empty_cache()
    method_ids = tuple(method.method_id for method in methods)
    validate_profile_rows(rows, methods=method_ids)
    summary = summarize_profile_rows(rows, methods=method_ids)
    sample_content = _csv_bytes(rows, SAMPLE_FIELDS)
    summary_fields = tuple(summary[0])
    summary_content = _csv_bytes(summary, summary_fields)
    output_root = output_root.expanduser().resolve()
    _atomic_write(output_root / "measurements.csv", sample_content)
    _atomic_write(output_root / "summary.csv", summary_content)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "perception-gain-profile-v1",
        "status": "PASS",
        "source_commit": source_commit,
        "device": device_label,
        "device_index": device.index,
        "evaluation_seed": 45,
        "warmup_full_sequences": WARMUP_REPEATS,
        "measured_full_sequences": MEASURED_REPEATS,
        "horizons": list(HORIZONS),
        "methods": [method.__dict__ for method in methods],
        "selected_sequences": selected_identity,
        "population_manifest_sha256": population_hash,
        "scope": {
            "end_to_end_includes": "in-memory sample collation, host-to-device transfer, live networks, D0 state, and usable output materialization",
            "end_to_end_excludes": "disk cold read and official metric computation",
            "network_forward": "encoder/decoder/scorer plus selected refiner calls",
            "state_update": "D0 association and resident state commit",
            "materialize": "full-resolution output reconstruction, publisher archive/buffer update, and required CPU transfer",
            "prefix_cumulative": "sum of measured online T1-through-current-stage end-to-end latency",
        },
        "measurement_rows": len(rows),
        "summary_rows": len(summary),
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "outputs": {
            "measurements.csv": {
                "bytes": len(sample_content),
                "sha256": __import__("hashlib").sha256(sample_content).hexdigest(),
            },
            "summary.csv": {
                "bytes": len(summary_content),
                "sha256": __import__("hashlib").sha256(summary_content).hexdigest(),
            },
        },
    }
    _atomic_write(
        output_root / "PROFILE_SUMMARY.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/perception_gain_v2.yaml")
    parser.add_argument(
        "--final-lock", type=Path, default=ARTIFACT_ROOT / "selection/FINAL_LOCK.json"
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=WARMUP_REPEATS)
    parser.add_argument("--repeats", type=int, default=MEASURED_REPEATS)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "resources")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup != WARMUP_REPEATS or args.repeats != MEASURED_REPEATS:
        raise SystemExit("profile requires --warmup 1 --repeats 3")
    if args.version == "v2":
        from scripts.perception_gain_v2 import load_config
        from scripts.perception_gain_v2_profile import run_profile_v2

        root = args.external_root or Path(os.environ.get(
            "PERSIST4D_GAIN_V2_ROOT", Path.home() / "persist4d_runs/perception_gain_v2"
        ))
        result = run_profile_v2(load_config(args.config), external_root=root)
        print(json.dumps(result, sort_keys=True))
        return int(result["status"] != "COMPLETE")
    result = run_profile(
        final_lock_path=args.final_lock,
        assets_path=args.assets,
        external_root=args.external_root or DEFAULT_EXTERNAL_ROOT,
        output_root=args.output,
        device_name=args.device,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
