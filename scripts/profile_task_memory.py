#!/usr/bin/env python3
"""Frozen profile-contract validation and TaskMemory resource summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

from scripts.profile_system_comparison import (
    PROTOCOL_MEASURED_REPEATS,
    PROTOCOL_WARMUP_REPEATS,
    ProfileUnit,
    build_profile_subset,
    measure_cuda_repeats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2"
DEFAULT_EXTERNAL_ROOT = Path("/mnt/shared/ww/persist4d-task-memory-retention-v2")

PROFILE_SCOPES = (
    "model_update",
    "end_to_end_update",
    "output_materialization",
    "cumulative_T1_to_T",
)


class ProfileError(RuntimeError):
    """Raised when profiling evidence violates the frozen contract."""


def _validate_profile_device_name(value: object) -> str:
    if not isinstance(value, str) or "A40" not in value.split():
        raise ProfileError("profile device must be an A40")
    return value


def validate_profile_contract(contract: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(contract, Mapping):
        raise ProfileError("profile contract must be a mapping")
    if contract.get("schema_version") != "task-memory-profile-v2":
        raise ProfileError("profile schema differs")
    if contract.get("device_count") != 1 or contract.get("device_type") != "A40":
        raise ProfileError("profile requires one A40")
    if contract.get("reference_units") != 6:
        raise ProfileError("profile requires six reference units")
    if contract.get("warmups") != 5:
        raise ProfileError("profile requires 5 warmups")
    if contract.get("repeats") != 10:
        raise ProfileError("profile requires 10 measured repeats")
    if contract.get("restore_cloned_state_each_repeat") is not True:
        raise ProfileError("profile must restore cloned state for each repeat")
    if tuple(contract.get("time_scopes", ())) != PROFILE_SCOPES:
        raise ProfileError("profile time scopes differ")
    return dict(contract)


def build_profile_units(
    protocol_manifest: Mapping[str, object],
) -> tuple[ProfileUnit, ...]:
    try:
        return build_profile_subset(protocol_manifest)
    except (TypeError, ValueError) as error:
        raise ProfileError(str(error)) from error


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProfileError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be finite and non-negative")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ProfileError(f"{name} must be finite and non-negative")
    return result


def summarize_measurements(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ProfileError("profile measurements must be a non-empty sequence")
    grouped: dict[tuple[str, str, str, int, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    snapshots: dict[tuple[str, str, str, int, str], set[str]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProfileError("profile measurement must be a mapping")
        method = row.get("method")
        reference = row.get("reference_id")
        master = row.get("master_sequence_id")
        scope = row.get("scope")
        if not all(
            isinstance(value, str) and value for value in (method, reference, master)
        ):
            raise ProfileError("profile measurement identity differs")
        if scope not in PROFILE_SCOPES:
            raise ProfileError("profile measurement scope differs")
        horizon = _integer(row.get("T"), name="profile T", minimum=2)
        if horizon not in (2, 3, 4, 5):
            raise ProfileError("profile horizon differs")
        repeat = _integer(row.get("repeat"), name="profile repeat")
        if repeat >= 10:
            raise ProfileError("profile repeat differs")
        snapshot = row.get("state_snapshot_sha256")
        if not isinstance(snapshot, str) or len(snapshot) != 64:
            raise ProfileError("profile state snapshot SHA256 differs")
        key = (str(method), str(reference), str(master), horizon, str(scope))
        grouped[key].append(row)
        snapshots[key].add(snapshot)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        repeats = {_integer(row.get("repeat"), name="profile repeat") for row in values}
        if repeats != set(range(10)) or len(snapshots[key]) != 1:
            raise ProfileError("profile requires ten repeats from one cloned state")
        latency = [_number(row.get("latency_ms"), name="latency_ms") for row in values]
        memory_fields = (
            "loaded_model_allocated_bytes",
            "peak_allocated_bytes",
            "incremental_peak_bytes",
            "peak_reserved_bytes",
            "cpu_state_bytes",
            "visual_state_bytes",
            "lag1_buffer_bytes",
            "archive_bytes",
        )
        memory = {
            field: max(_integer(row.get(field), name=field) for row in values)
            for field in memory_fields
        }
        method, reference, master, horizon, scope = key
        output.append(
            {
                "method": method,
                "reference_id": reference,
                "master_sequence_id": master,
                "T": horizon,
                "scope": scope,
                "measured_repeats": 10,
                "state_snapshot_sha256": next(iter(snapshots[key])),
                "median_latency_ms": statistics.median(latency),
                "mean_latency_ms": statistics.mean(latency),
                "std_latency_ms": statistics.pstdev(latency),
                "loaded_model_allocated_bytes": memory["loaded_model_allocated_bytes"],
                "absolute_peak_allocated_bytes": memory["peak_allocated_bytes"],
                "incremental_peak_bytes": memory["incremental_peak_bytes"],
                "peak_reserved_bytes": memory["peak_reserved_bytes"],
                "cpu_state_bytes": memory["cpu_state_bytes"],
                "visual_state_bytes": memory["visual_state_bytes"],
                "lag1_buffer_bytes": memory["lag1_buffer_bytes"],
                "archive_bytes": memory["archive_bytes"],
            }
        )
    methods = {row["method"] for row in output}
    references = {row["reference_id"] for row in output}
    expected = len(methods) * 6 * 4 * len(PROFILE_SCOPES)
    if len(references) != 6 or len(output) != expected:
        raise ProfileError("profile summary coverage differs")
    return output


def derive_resource_status(
    summary_rows: Sequence[Mapping[str, object]],
    *,
    candidate: str,
    baseline: str,
    permanent_state_bytes: int,
    state_budget_bytes: int,
) -> str:
    if permanent_state_bytes > state_budget_bytes:
        return "NO_ADVANTAGE"
    paired: dict[tuple[str, int], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in summary_rows:
        if (
            isinstance(row, Mapping)
            and row.get("scope") == "model_update"
            and row.get("T") in (4, 5)
            and row.get("method") in (candidate, baseline)
        ):
            key = (str(row.get("reference_id")), int(row["T"]))
            method = str(row["method"])
            if method in paired[key]:
                raise ProfileError("resource comparison contains duplicate cells")
            paired[key][method] = row
    if len(paired) != 12 or any(
        set(cell) != {candidate, baseline} for cell in paired.values()
    ):
        raise ProfileError("resource comparison requires paired T4/T5 six-unit cells")
    latency_wins = []
    peak_wins = []
    for horizon in (4, 5):
        horizon_cells = [
            cell for (_, value), cell in paired.items() if value == horizon
        ]
        candidate_latency = statistics.median(
            _number(cell[candidate].get("median_latency_ms"), name="candidate latency")
            for cell in horizon_cells
        )
        baseline_latency = statistics.median(
            _number(cell[baseline].get("median_latency_ms"), name="baseline latency")
            for cell in horizon_cells
        )
        candidate_peak = statistics.median(
            _integer(
                cell[candidate].get("absolute_peak_allocated_bytes"),
                name="candidate peak",
            )
            for cell in horizon_cells
        )
        baseline_peak = statistics.median(
            _integer(
                cell[baseline].get("absolute_peak_allocated_bytes"),
                name="baseline peak",
            )
            for cell in horizon_cells
        )
        latency_wins.append(candidate_latency < baseline_latency)
        peak_wins.append(candidate_peak < baseline_peak)
    if all(latency_wins) and all(peak_wins):
        return "ADVANTAGE"
    if any(latency_wins) or any(peak_wins):
        return "TRADEOFF"
    return "NO_ADVANTAGE"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_permanent_state_bytes(path: Path) -> int:
    source = path.expanduser().resolve(strict=True)
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ProfileError("state byte evidence cannot be read") from error
    matches = [
        row
        for row in rows
        if row.get("component") == "combined_state" and row.get("tensor") == "TOTAL"
    ]
    if len(matches) != 1 or matches[0].get("status") != "PASS":
        raise ProfileError("combined_state byte evidence differs")
    try:
        value = int(matches[0]["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError("combined_state byte evidence differs") from error
    return _integer(value, name="combined_state bytes", minimum=1)


def _storage_bytes(value: object) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_storage_bytes(item) for item in value)
    return 0


def _serialized_bytes(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _clone_state(
    state: object | None, *, device: torch.device | None = None
) -> object | None:
    if state is None:
        return None
    from models.task_memory_state import TaskMemoryState

    if not isinstance(state, TaskMemoryState):
        raise ProfileError("task state type differs")
    return TaskMemoryState(
        *(
            tensor.detach().to(device=device or tensor.device).clone()
            for tensor in state.tensors()
        ),
        config=state.config,
    )


def _clone_visual_state(
    state: object | None, *, device: torch.device | None = None
) -> object | None:
    if state is None:
        return None
    from models.object_visual_memory import ObjectVisualState

    if not isinstance(state, ObjectVisualState):
        raise ProfileError("visual state type differs")
    return ObjectVisualState(
        *(
            tensor.detach().to(device=device or tensor.device).clone()
            for tensor in state.tensors()
        )
    )


def _prepare_sample(
    *,
    sample: object,
    spec: object,
    stage_collator: object,
    system: object,
    device: torch.device,
) -> SimpleNamespace:
    from datasets.task_memory_episode import (
        TaskMemoryEpisodeSpec,
        TaskMemoryStageSample,
        _build_stage_meta,
        _clone_model_sample,
    )
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )

    if not isinstance(sample, TaskMemoryStageSample) or not isinstance(
        spec, TaskMemoryEpisodeSpec
    ):
        raise ProfileError("profile stage input type differs")
    if not callable(stage_collator):
        raise ProfileError("profile stage collator is unavailable")
    model_batch = stage_collator([_clone_model_sample(sample.model_sample)])
    if not isinstance(model_batch, tuple) or len(model_batch) != 3:
        raise ProfileError("profile collator output differs")
    data, targets, names = model_batch
    if not isinstance(targets, Sequence) or len(targets) != 1:
        raise ProfileError("profile target coverage differs")
    metas, _ = _build_stage_meta(
        point=data,
        targets=targets,
        stage_samples=(sample,),
        specs=(spec,),
    )
    expected_name = "-".join(metas[0].scan_ids_in_window)
    if list(names) != [expected_name]:
        raise ProfileError("profile collator changed sequence identity")
    full_targets = getattr(data, "target_full", None)
    if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
        raise ProfileError("profile input lacks one full-resolution target")
    full_target = full_targets[0]
    data = _move_data_to_device(data, device)
    targets = _move_targets_to_device(targets, device)
    target_low = targets[0]
    segment_stages = _segment_stages(target_low)
    latest_stage = int(segment_stages.max().item())
    raw_coordinates = system._process_raw_coordinates(data)
    point2segment = target_low.get("point2segment")
    if not isinstance(point2segment, Tensor):
        raise ProfileError("profile point2segment is unavailable")
    return SimpleNamespace(
        data=data,
        target_low=target_low,
        full_target=full_target,
        meta=metas[0],
        stage_meta=metas,
        latest_stage=latest_stage,
        raw_coordinates=raw_coordinates,
        point_count=int(point2segment.numel()),
        superpoint_count=int(torch.unique(point2segment).numel()),
    )


def _execute_prepared(
    *,
    prepared: SimpleNamespace,
    system: object,
    class_mapper: object,
    task_state: object | None,
    visual_state: object | None,
) -> tuple[object | None, object | None, dict[str, object], dict[str, object]]:
    from models.object_visual_memory import update_visual_state
    from models.task_memory_routing import EntityRoute
    from scripts.evaluate_task_memory import (
        _event_diagnostics,
        _runtime_state_sha256,
        _visual_stage_diagnostics,
        _visual_state_for_read,
        current_stage_target,
    )
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.task_memory_cache import build_stage_cache_record
    from trainer.task_memory_trainer import (
        build_final_prediction_observation,
        build_visual_slot_candidates,
        commit_task_memory_stage,
    )

    if not callable(class_mapper):
        raise ProfileError("profile class mapper is unavailable")
    state_enabled = bool(system.config.task_memory_training.state_enabled)
    visual_enabled = bool(system.config.task_memory_training.visual_enabled)
    state_before_sha256 = _runtime_state_sha256(task_state, visual_state)
    read_visual_state = (
        _visual_state_for_read(visual_state, content_control="native")
        if visual_state is not None
        else None
    )
    with torch.inference_mode():
        if state_enabled:
            output = system.model(
                prepared.data,
                point2segment=[prepared.target_low["point2segment"]],
                raw_coordinates=prepared.raw_coordinates,
                is_eval=True,
                task_state=task_state,
                task_visual_state=read_visual_state,
                stage_meta=prepared.stage_meta,
            )
        else:
            output = system.model(
                prepared.data,
                point2segment=[prepared.target_low["point2segment"]],
                raw_coordinates=prepared.raw_coordinates,
                is_eval=True,
            )
    official = extract_official_task_prediction(
        system=system,
        output=output,
        target_low_resolution=prepared.target_low,
        target_full_resolution=prepared.full_target,
        data=prepared.data,
        class_mapper=class_mapper,
        latest_stage_index=prepared.latest_stage,
    )
    current_target = current_stage_target(
        prepared.full_target, latest_local_stage=prepared.latest_stage
    )
    if state_enabled:
        route = output.get("task_memory_route")
        if not isinstance(route, EntityRoute) or task_state is None:
            raise ProfileError("profile task route/state is unavailable")
        observation = build_final_prediction_observation(
            output,
            prepared.stage_meta,
            background_class=int(system.model.task_background_class),
            confidence_threshold=float(system.model.task_confidence_threshold),
            mask_threshold=float(system.model.task_mask_threshold),
            minimum_mask_support=int(system.model.task_minimum_mask_support),
        )
        transition = commit_task_memory_stage(
            observation=observation,
            route=route,
            state=task_state,
            stage_meta=prepared.stage_meta,
        )
        events = _event_diagnostics(
            route=route,
            commit=transition.commit,
            state_before=task_state,
            official_prediction=official,
            target=current_target,
            class_mapper=class_mapper,
            stage_index=prepared.meta.absolute_stage_index,
        )
        next_state = transition.runtime_state
        if visual_enabled:
            if visual_state is None or read_visual_state is None:
                raise ProfileError("profile visual state is unavailable")
            diagnostics = output.get("task_memory_visual_read_diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ProfileError("profile visual diagnostics are unavailable")
            visual_diagnostics = _visual_stage_diagnostics(
                stored_state=visual_state,
                read_state=read_visual_state,
                read_diagnostics=diagnostics,
                absolute_stage_index=prepared.meta.absolute_stage_index,
                content_control="native",
            )
            next_visual = update_visual_state(
                visual_state,
                slot_candidates=build_visual_slot_candidates(
                    output,
                    prepared.stage_meta,
                    query_to_slot=transition.commit.query_to_slot,
                    background_class=int(system.model.task_background_class),
                    mask_threshold=float(system.model.task_mask_threshold),
                    capacity=int(system.model.task_memory_capacity),
                ),
                slot_generations=transition.commit.state.generations,
                policy=str(system.model.task_visual_policy),
            )
        else:
            visual_diagnostics = None
            next_visual = None
        identity_map = transition.commit.identity_map()
        state_after_sha256 = _runtime_state_sha256(next_state, next_visual)
    else:
        next_state = None
        next_visual = None
        visual_diagnostics = None
        identity_map = {}
        events = {
            "births": 0,
            "matched_births": 0,
            "matched_reactivations": 0,
            "reactivations": 0,
            "rejected_births": 0,
        }
        state_after_sha256 = _runtime_state_sha256(None, None)
    record = build_stage_cache_record(
        prediction=official,
        identity_map=identity_map,
        stage_meta=prepared.meta,
        target=current_target,
        state_before_sha256=state_before_sha256,
        state_after_sha256=state_after_sha256,
        event_diagnostics=events,
        visual_diagnostics=visual_diagnostics,
    )
    stats = {
        "point_count": prepared.point_count,
        "superpoint_count": prepared.superpoint_count,
        "query_count": int(output["pred_logits"].shape[1]),
        "occupied_count": (
            int(next_state.occupied.sum().item()) if next_state is not None else 0
        ),
    }
    return next_state, next_visual, record, stats


def _profile_rows(
    *,
    method: str,
    system: object,
    dataset: object,
    master: object,
    unit: ProfileUnit,
    class_mapper: object,
    device: torch.device,
    runtime_directory: Path,
    source_commit: str,
    loaded_model_allocated_bytes: int,
) -> list[dict[str, object]]:
    from datasets.task_memory_episode import TaskMemoryEpisodeDataset
    from scripts.evaluate_task_memory import _episode_spec, _runtime_state_sha256

    spec = _episode_spec(master, index=unit.context_index)
    episode = TaskMemoryEpisodeDataset(
        dataset,
        (spec,),
        apply_augmentation=False,
        window_mode=str(system.config.task_memory_training.window_mode),
    )[0]
    input_paths = []
    for stage, sample in enumerate(episode.stage_samples):
        path = runtime_directory / f"{method}-{unit.reference_scene_id}-{stage}.pt"
        torch.save({"sample": sample, "spec": spec}, path)
        input_paths.append(path)
    stage_collator = system.config.data.validation_collation
    import hydra

    collator = hydra.utils.instantiate(stage_collator)
    state = (
        system._new_task_state(1)
        if bool(system.config.task_memory_training.state_enabled)
        else None
    )
    visual_state = (
        system._new_visual_state(1)
        if bool(system.config.task_memory_training.visual_enabled)
        else None
    )
    initial_state_sha256 = _runtime_state_sha256(state, visual_state)
    snapshots = []
    records = []
    stats = []
    for sample in episode.stage_samples:
        snapshots.append(
            (
                _clone_state(state, device=torch.device("cpu")),
                _clone_visual_state(visual_state, device=torch.device("cpu")),
            )
        )
        prepared = _prepare_sample(
            sample=sample,
            spec=spec,
            stage_collator=collator,
            system=system,
            device=device,
        )
        state, visual_state, record, stage_stats = _execute_prepared(
            prepared=prepared,
            system=system,
            class_mapper=class_mapper,
            task_state=state,
            visual_state=visual_state,
        )
        records.append(record)
        stats.append(stage_stats)
    del prepared, state, visual_state
    precision = str(next(system.model.parameters()).dtype)
    rows = []
    for horizon in (2, 3, 4, 5):
        stage_index = horizon - 1
        snapshot_state, snapshot_visual = snapshots[stage_index]
        snapshot_sha = _runtime_state_sha256(snapshot_state, snapshot_visual)
        reference_record = records[stage_index]
        archive_bytes = len(_serialized_bytes(reference_record))
        lag1_bytes = _storage_bytes(reference_record.get("prediction"))
        task_state_bytes = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in snapshot_state.tensors()
            )
            if snapshot_state is not None
            else 0
        )
        visual_state_bytes = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in snapshot_visual.tensors()
            )
            if snapshot_visual is not None
            else 0
        )

        def model_factory(
            sample=episode.stage_samples[stage_index],
            task_snapshot=snapshot_state,
            visual_snapshot=snapshot_visual,
        ):
            prepared = _prepare_sample(
                sample=sample,
                spec=spec,
                stage_collator=collator,
                system=system,
                device=device,
            )
            current_state = _clone_state(task_snapshot, device=device)
            current_visual = _clone_visual_state(visual_snapshot, device=device)

            def operation():
                return _execute_prepared(
                    prepared=prepared,
                    system=system,
                    class_mapper=class_mapper,
                    task_state=current_state,
                    visual_state=current_visual,
                )

            return operation

        def end_to_end_factory(
            path=input_paths[stage_index],
            task_snapshot=snapshot_state,
            visual_snapshot=snapshot_visual,
        ):
            current_state = _clone_state(task_snapshot, device=device)
            current_visual = _clone_visual_state(visual_snapshot, device=device)

            def operation():
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                prepared = _prepare_sample(
                    sample=loaded["sample"],
                    spec=loaded["spec"],
                    stage_collator=collator,
                    system=system,
                    device=device,
                )
                result = _execute_prepared(
                    prepared=prepared,
                    system=system,
                    class_mapper=class_mapper,
                    task_state=current_state,
                    visual_state=current_visual,
                )
                return _serialized_bytes(result[2])

            return operation

        def materialization_factory(record=reference_record):
            return lambda: _serialized_bytes(record)

        def cumulative_factory(paths=tuple(input_paths[:horizon])):
            def operation():
                current_state = (
                    system._new_task_state(1)
                    if bool(system.config.task_memory_training.state_enabled)
                    else None
                )
                current_visual = (
                    system._new_visual_state(1)
                    if bool(system.config.task_memory_training.visual_enabled)
                    else None
                )
                outputs = []
                for path in paths:
                    loaded = torch.load(path, map_location="cpu", weights_only=False)
                    prepared = _prepare_sample(
                        sample=loaded["sample"],
                        spec=loaded["spec"],
                        stage_collator=collator,
                        system=system,
                        device=device,
                    )
                    current_state, current_visual, record, _ = _execute_prepared(
                        prepared=prepared,
                        system=system,
                        class_mapper=class_mapper,
                        task_state=current_state,
                        visual_state=current_visual,
                    )
                    outputs.append(_serialized_bytes(record))
                return outputs

            return operation

        factories = {
            "model_update": model_factory,
            "end_to_end_update": end_to_end_factory,
            "output_materialization": materialization_factory,
            "cumulative_T1_to_T": cumulative_factory,
        }
        for scope in PROFILE_SCOPES:
            state_hash = (
                initial_state_sha256 if scope == "cumulative_T1_to_T" else snapshot_sha
            )
            profile = measure_cuda_repeats(
                factories[scope],
                device=device,
                warmup_repeats=PROTOCOL_WARMUP_REPEATS,
                measured_repeats=PROTOCOL_MEASURED_REPEATS,
                enforce_protocol=True,
            )
            for repeat, (latency, allocated, reserved) in enumerate(
                zip(
                    profile.samples_ms,
                    profile.allocated_samples_bytes,
                    profile.reserved_samples_bytes,
                    strict=True,
                )
            ):
                rows.append(
                    {
                        "method": method,
                        "reference_id": unit.reference_scene_id,
                        "master_sequence_id": unit.master_sequence_id,
                        "order_id": unit.order_id,
                        "T": horizon,
                        "scope": scope,
                        "repeat": repeat,
                        "state_snapshot_sha256": state_hash,
                        "latency_ms": latency,
                        "loaded_model_allocated_bytes": loaded_model_allocated_bytes,
                        "peak_allocated_bytes": allocated,
                        "incremental_peak_bytes": max(
                            0, allocated - loaded_model_allocated_bytes
                        ),
                        "peak_reserved_bytes": reserved,
                        "cpu_state_bytes": task_state_bytes,
                        "visual_state_bytes": visual_state_bytes,
                        "lag1_buffer_bytes": lag1_bytes,
                        "archive_bytes": archive_bytes,
                        "point_count": stats[stage_index]["point_count"],
                        "superpoint_count": stats[stage_index]["superpoint_count"],
                        "occupied_count": stats[stage_index]["occupied_count"],
                        "query_count": stats[stage_index]["query_count"],
                        "precision": precision,
                        "source_commit": source_commit,
                    }
                )
            print(
                f"[task-memory-profile] {method} {unit.reference_scene_id} "
                f"T{horizon} {scope} complete",
                flush=True,
            )
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ProfileError("profile CSV cannot be empty")
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise ProfileError("profile CSV fields differ")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProfileError(f"profile output cannot be a symlink: {path}")
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


def run_profile(
    *,
    comparison_contract: Path,
    protocol_manifest: Path,
    data_contract: Path,
    variant_manifest: Path,
    data_root: Path,
    rio_metadata: Path,
    state_bytes_path: Path,
    pretrained: Path,
    external_root: Path,
    candidate_checkpoint: Path,
    fh_checkpoint: Path,
    device_name: str,
    output_root: Path,
) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.evaluate_task_memory import (
        PROTOCOL_B_POPULATION_ID,
        _build_protocol_b_population,
        _load_evaluation_system,
        _load_json,
        _resolve_variant_evaluation_identity,
        _rio_population_base,
    )
    from scripts.task_memory_contracts import canonical_json_sha256

    contract = _load_json(comparison_contract.expanduser().resolve(strict=True))
    unsigned = dict(contract)
    observed_hash = unsigned.pop("content_sha256", None)
    if observed_hash != canonical_json_sha256(unsigned):
        raise ProfileError("profile contract content SHA256 differs")
    validate_profile_contract(contract)
    protocol = _load_json(protocol_manifest.expanduser().resolve(strict=True))
    units = build_profile_units(protocol)
    data_contract_value = _load_json(data_contract.expanduser().resolve(strict=True))
    unsigned_data_contract = dict(data_contract_value)
    data_contract_content_sha256 = unsigned_data_contract.pop("content_sha256", None)
    if data_contract_content_sha256 != canonical_json_sha256(unsigned_data_contract):
        raise ProfileError("profile data contract content SHA256 differs")
    expected_protocol = data_contract_value.get("sources", {}).get("protocol_b", {})
    if not isinstance(expected_protocol, Mapping) or _file_sha256(
        protocol_manifest
    ) != expected_protocol.get("sha256"):
        raise ProfileError("profile Protocol-B identity differs")
    manifest = _load_json(variant_manifest.expanduser().resolve(strict=True))
    unsigned_manifest = dict(manifest)
    manifest_content_sha256 = unsigned_manifest.pop("content_sha256", None)
    if manifest_content_sha256 != canonical_json_sha256(unsigned_manifest):
        raise ProfileError("profile variant manifest content SHA256 differs")
    data_root = data_root.expanduser().resolve(strict=True)
    pretrained = pretrained.expanduser().resolve(strict=True)
    external_root = external_root.expanduser().resolve(strict=True)
    rio_metadata = rio_metadata.expanduser().resolve(strict=True)
    rio_metadata_sha256 = _file_sha256(rio_metadata)
    pretrained_sha256 = _file_sha256(pretrained)
    device = torch.device(device_name)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise ProfileError("profile requires an explicit CUDA device")
    device_label = _validate_profile_device_name(torch.cuda.get_device_name(device))
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked_status:
        raise ProfileError("profile requires a clean tracked source commit")
    output = []
    profiled_models = []
    runtime_parent = external_root / "profile_runtime"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    for variant, checkpoint in (
        ("M3-V-CORE", candidate_checkpoint),
        ("FH-CONT", fh_checkpoint),
    ):
        checkpoint = checkpoint.expanduser().resolve(strict=True)
        resolved, common, visual = _resolve_variant_evaluation_identity(
            variant=variant,
            manifest=manifest,
            external_root=external_root,
        )
        checkpoint_sha256 = _file_sha256(checkpoint)
        system, _ = _load_evaluation_system(
            variant=variant,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            pretrained=pretrained,
            external_root=external_root,
            device=device,
            expected_common_initialization_sha256=common,
            expected_common_visual_initialization_sha256=visual,
        )
        torch.cuda.empty_cache()
        loaded_model_allocated_bytes = int(torch.cuda.memory_allocated(device))
        base = _rio_population_base(
            system.config,
            data_root=data_root,
            horizon=5,
            population_id=PROTOCOL_B_POPULATION_ID,
        )
        wrapped, population = _build_protocol_b_population(base, protocol)
        class_mapper = build_rio_class_mapper(base)
        with tempfile.TemporaryDirectory(
            prefix=f"{variant}-", dir=runtime_parent
        ) as temporary:
            runtime_directory = Path(temporary)
            for unit in units:
                matches = [
                    master
                    for master in population
                    if master.reference_id == unit.reference_scene_id
                    and master.scan_ids == unit.visit_order
                ]
                if len(matches) != 1:
                    raise ProfileError("profile canonical unit cannot be resolved")
                output.extend(
                    _profile_rows(
                        method=variant,
                        system=system,
                        dataset=wrapped,
                        master=matches[0],
                        unit=unit,
                        class_mapper=class_mapper,
                        device=device,
                        runtime_directory=runtime_directory,
                        source_commit=source_commit,
                        loaded_model_allocated_bytes=loaded_model_allocated_bytes,
                    )
                )
        del system
        torch.cuda.empty_cache()
        profiled_models.append(
            {
                "variant": variant,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "resolved_config_sha256": resolved,
            }
        )
    summary = summarize_measurements(output)
    permanent_state_bytes = load_permanent_state_bytes(state_bytes_path)
    resource_status = derive_resource_status(
        summary,
        candidate="M3-V-CORE",
        baseline="FH-CONT",
        permanent_state_bytes=permanent_state_bytes,
        state_budget_bytes=2 * 1024 * 1024,
    )
    outputs = {
        "per_update_measurements.csv": _csv_bytes(output),
        "profile_units.csv": _csv_bytes(
            [
                {
                    "reference_id": unit.reference_scene_id,
                    "master_sequence_id": unit.master_sequence_id,
                    "order_id": unit.order_id,
                    "context_index": unit.context_index,
                    "scan_indices": " ".join(map(str, unit.scan_indices)),
                }
                for unit in units
            ]
        ),
        "profile_summary.csv": _csv_bytes(summary),
    }
    output_root = output_root.expanduser().resolve()
    for filename, content in outputs.items():
        _atomic_write(output_root / filename, content)
    result = {
        "schema_version": "task-memory-profile-run-v2",
        "status": "PASS",
        "resource_status": resource_status,
        "source_commit": source_commit,
        "profile_contract_sha256": _file_sha256(
            comparison_contract.expanduser().resolve(strict=True)
        ),
        "protocol_manifest_sha256": _file_sha256(
            protocol_manifest.expanduser().resolve(strict=True)
        ),
        "data_contract_sha256": _file_sha256(
            data_contract.expanduser().resolve(strict=True)
        ),
        "variant_manifest_sha256": _file_sha256(
            variant_manifest.expanduser().resolve(strict=True)
        ),
        "rio_metadata_sha256": rio_metadata_sha256,
        "pretrained_sha256": pretrained_sha256,
        "state_bytes_evidence_sha256": _file_sha256(
            state_bytes_path.expanduser().resolve(strict=True)
        ),
        "device": device_label,
        "warmups": PROTOCOL_WARMUP_REPEATS,
        "repeats": PROTOCOL_MEASURED_REPEATS,
        "measurement_rows": len(output),
        "summary_rows": len(summary),
        "permanent_state_bytes": permanent_state_bytes,
        "profiled_models": profiled_models,
        "outputs": [
            {"path": name, "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in sorted(outputs.items())
        ],
    }
    result["content_sha256"] = canonical_json_sha256(result)
    _atomic_write(
        output_root / "run_summary.json",
        (json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-contract",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "PROFILE_CONTRACT.json",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
    )
    parser.add_argument(
        "--data-contract",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "DATA_CONTRACT.json",
    )
    parser.add_argument(
        "--variant-manifest",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "training/variants.json",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--rio-metadata", type=Path, required=True)
    parser.add_argument(
        "--state-bytes",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "resources/state_bytes.csv",
    )
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--fh-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACT_ROOT / "resources"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.warmup != 5 or args.repeats != 10:
        raise SystemExit("profile requires --warmup 5 --repeats 10")
    result = run_profile(
        comparison_contract=args.comparison_contract,
        protocol_manifest=args.protocol_manifest,
        data_contract=args.data_contract,
        variant_manifest=args.variant_manifest,
        data_root=args.data_root,
        rio_metadata=args.rio_metadata,
        state_bytes_path=args.state_bytes,
        pretrained=args.pretrained,
        external_root=args.external_root,
        candidate_checkpoint=args.candidate_checkpoint,
        fh_checkpoint=args.fh_checkpoint,
        device_name=args.device,
        output_root=args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "PROFILE_SCOPES",
    "ProfileError",
    "build_profile_units",
    "derive_resource_status",
    "load_permanent_state_bytes",
    "summarize_measurements",
    "validate_profile_contract",
]


if __name__ == "__main__":
    raise SystemExit(main())
