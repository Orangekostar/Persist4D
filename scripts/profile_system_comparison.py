"""CUDA timing, VRAM, and state-size primitives for system comparison."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_WARMUP_REPEATS = 5
PROTOCOL_MEASURED_REPEATS = 10


class ProfilingError(ValueError):
    """Raised when profiling would violate the frozen measurement boundary."""


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProfilingError(f"{name} must be a nonempty string")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProfilingError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ProfileUnit:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    context_index: int
    context_scan_indices: tuple[int, ...]
    visit_order: tuple[str, ...]
    scan_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.reference_scene_id, name="reference_scene_id")
        _nonempty_string(self.master_sequence_id, name="master_sequence_id")
        if self.order_id != "canonical":
            raise ProfilingError("profile subset must use canonical order")
        _nonnegative_integer(self.context_index, name="context_index")
        if (
            len(self.context_scan_indices) != 5
            or len(set(self.context_scan_indices)) != 5
            or len(self.visit_order) != 5
            or len(set(self.visit_order)) != 5
            or len(self.scan_indices) != 5
            or len(set(self.scan_indices)) != 5
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*self.context_scan_indices, *self.scan_indices)
            )
            or any(not isinstance(value, str) or not value for value in self.visit_order)
        ):
            raise ProfilingError("profile unit must contain five unique scans")


def build_profile_subset(
    system_manifest: Mapping[str, object],
) -> tuple[ProfileUnit, ...]:
    if not isinstance(system_manifest, Mapping):
        raise ProfilingError("system manifest must be a mapping")
    protocol = system_manifest.get("protocol")
    masters = system_manifest.get("masters")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("order_variants")
        != ["canonical", "reverse", "sha256_seed45"]
        or isinstance(masters, (str, bytes))
        or not isinstance(masters, Sequence)
        or len(masters) != 43
    ):
        raise ProfilingError("system manifest does not expose exact Protocol-B")
    by_reference: dict[str, list[Mapping[str, object]]] = {}
    for master in masters:
        if not isinstance(master, Mapping):
            raise ProfilingError("system masters must contain mappings")
        reference = _nonempty_string(
            master.get("reference_scene_id"), name="reference_scene_id"
        )
        by_reference.setdefault(reference, []).append(master)
    if len(by_reference) != 6:
        raise ProfilingError("profile subset requires all six reference scenes")
    units: list[ProfileUnit] = []
    for reference in sorted(by_reference):
        selected = min(
            by_reference[reference], key=lambda item: str(item["master_sequence_id"])
        )
        orders = selected.get("orders")
        if not isinstance(orders, Mapping) or not isinstance(
            orders.get("canonical"), Mapping
        ):
            raise ProfilingError("profile master lacks canonical order")
        canonical = orders["canonical"]
        units.append(
            ProfileUnit(
                reference_scene_id=reference,
                master_sequence_id=_nonempty_string(
                    selected.get("master_sequence_id"), name="master_sequence_id"
                ),
                order_id="canonical",
                context_index=_nonnegative_integer(
                    selected.get("validation_index"), name="validation_index"
                ),
                context_scan_indices=tuple(
                    _nonnegative_integer(value, name="context scan index")
                    for value in selected.get("scan_indices", ())
                ),
                visit_order=tuple(canonical.get("visit_order", ())),
                scan_indices=tuple(
                    _nonnegative_integer(value, name="canonical scan index")
                    for value in canonical.get("scan_indices", ())
                ),
            )
        )
    return tuple(units)


@dataclass(frozen=True)
class CudaRepeatProfile:
    warmup_repeats: int
    measured_repeats: int
    samples_ms: tuple[float, ...]
    median_ms: float
    mean_ms: float
    std_ms: float
    allocated_samples_bytes: tuple[int, ...]
    reserved_samples_bytes: tuple[int, ...]
    peak_allocated_bytes: int
    peak_reserved_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "warmup_repeats": self.warmup_repeats,
            "measured_repeats": self.measured_repeats,
            "samples_ms": list(self.samples_ms),
            "median_ms": self.median_ms,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "allocated_samples_bytes": list(self.allocated_samples_bytes),
            "reserved_samples_bytes": list(self.reserved_samples_bytes),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


def measure_cuda_repeats(
    operation_factory: Callable[[], Callable[[], object]],
    *,
    device: torch.device,
    warmup_repeats: int = PROTOCOL_WARMUP_REPEATS,
    measured_repeats: int = PROTOCOL_MEASURED_REPEATS,
    enforce_protocol: bool = True,
    synchronize: Callable[[torch.device], object] = torch.cuda.synchronize,
    reset_peak: Callable[[torch.device], object] = torch.cuda.reset_peak_memory_stats,
    peak_allocated: Callable[[torch.device], int] = torch.cuda.max_memory_allocated,
    peak_reserved: Callable[[torch.device], int] = torch.cuda.max_memory_reserved,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> CudaRepeatProfile:
    """Measure only operations returned after untimed setup by the factory."""

    if not callable(operation_factory):
        raise ProfilingError("operation_factory must be callable")
    if device.type != "cuda" or device.index is None:
        raise ProfilingError("profiling device must identify one CUDA device")
    for name, value in (
        ("warmup_repeats", warmup_repeats),
        ("measured_repeats", measured_repeats),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ProfilingError(f"{name} must be a positive integer")
    if not isinstance(enforce_protocol, bool):
        raise ProfilingError("enforce_protocol must be boolean")
    if enforce_protocol and (
        warmup_repeats != PROTOCOL_WARMUP_REPEATS
        or measured_repeats != PROTOCOL_MEASURED_REPEATS
    ):
        raise ProfilingError("warmup/measured repeats must use the frozen 5+10 protocol")

    for _ in range(warmup_repeats):
        operation = operation_factory()
        if not callable(operation):
            raise ProfilingError("operation_factory must return a callable")
        synchronize(device)
        result = operation()
        synchronize(device)
        del result, operation

    samples: list[float] = []
    allocated: list[int] = []
    reserved: list[int] = []
    for _ in range(measured_repeats):
        operation = operation_factory()
        if not callable(operation):
            raise ProfilingError("operation_factory must return a callable")
        synchronize(device)
        reset_peak(device)
        started = clock_ns()
        result = operation()
        synchronize(device)
        ended = clock_ns()
        if ended < started:
            raise ProfilingError("monotonic clock moved backwards")
        elapsed_ms = (ended - started) / 1_000_000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise ProfilingError("measured latency is invalid")
        allocated_value = int(peak_allocated(device))
        reserved_value = int(peak_reserved(device))
        if allocated_value < 0 or reserved_value < 0:
            raise ProfilingError("CUDA peak memory must be non-negative")
        samples.append(elapsed_ms)
        allocated.append(allocated_value)
        reserved.append(reserved_value)
        del result, operation
    return CudaRepeatProfile(
        warmup_repeats=warmup_repeats,
        measured_repeats=measured_repeats,
        samples_ms=tuple(samples),
        median_ms=float(statistics.median(samples)),
        mean_ms=float(statistics.mean(samples)),
        std_ms=float(statistics.pstdev(samples)),
        allocated_samples_bytes=tuple(allocated),
        reserved_samples_bytes=tuple(reserved),
        peak_allocated_bytes=max(allocated),
        peak_reserved_bytes=max(reserved),
    )


def build_persistent_operation_factory(
    *,
    model_forward: Callable[[], object],
    tracker_factory: Callable[[], object],
    prior_observations: Sequence[object],
    current_stage: int,
) -> Callable[[], Callable[[], object]]:
    """Create fresh-state operations with causal tracker preroll outside timing."""

    if not callable(model_forward) or not callable(tracker_factory):
        raise ProfilingError("model_forward and tracker_factory must be callable")
    if isinstance(prior_observations, (str, bytes)) or not isinstance(
        prior_observations, Sequence
    ):
        raise ProfilingError("prior_observations must be a sequence")
    prior = tuple(prior_observations)
    if (
        isinstance(current_stage, bool)
        or not isinstance(current_stage, int)
        or current_stage != len(prior)
    ):
        raise ProfilingError("current_stage must immediately follow causal preroll")

    def operation_factory() -> Callable[[], object]:
        tracker = tracker_factory()
        step = getattr(tracker, "step", None)
        if not callable(step):
            raise ProfilingError("tracker must expose step()")
        for stage, observation in enumerate(prior):
            step(observation, stage_id=stage)

        def operation() -> object:
            observation = model_forward()
            return step(observation, stage_id=current_stage)

        return operation

    return operation_factory


def persistent_state_storage_bytes(state: object) -> int:
    tensors = getattr(state, "tensors", None)
    if not callable(tensors):
        raise ProfilingError("persistent state must expose tensors()")
    values = tensors()
    if not isinstance(values, tuple) or not values:
        raise ProfilingError("persistent state tensors must be a nonempty tuple")
    if any(not isinstance(value, Tensor) for value in values):
        raise ProfilingError("persistent state tensors must contain tensors")
    return sum(value.numel() * value.element_size() for value in values)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _move_frozen_observation(observation: object, device: torch.device) -> object:
    from scripts.p6a_association import FrozenObservation

    if not isinstance(observation, FrozenObservation):
        raise ProfilingError("cached preroll must contain FrozenObservation values")
    moved = FrozenObservation(
        features=observation.features.to(device),
        class_prob=observation.class_prob.to(device),
        confidence=observation.confidence.to(device),
        valid=observation.valid.to(device),
        latest_mask=tuple(mask.to(device) for mask in observation.latest_mask),
        mask_support=(
            None
            if observation.mask_support is None
            else observation.mask_support.to(device)
        ),
    )
    moved.validate()
    return moved


def run_system_profile(
    *,
    project_root: Path,
    device_name: str,
    metadata_path: Path,
) -> dict[str, object]:
    """Run the frozen six-cluster, T2-T5 5+10 CUDA profile."""

    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import (
        build_tracker_factories,
        cache_payload_to_frozen_observation,
        load_cached_protocol_sequences,
    )
    from scripts.run_system_comparison import (
        LOCAL_CACHE_MANIFEST,
        LOCAL_ENTRY_CACHE,
        SYSTEM_ROOT,
        _build_frozen_setup,
        _load_bound_inputs,
    )
    from scripts.system_comparison_analysis import _csv_bytes, _publish_exact
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        model_input_storage_bytes,
    )

    repository = Path(__file__).resolve().parents[1]
    if project_root.resolve() != repository:
        raise ProfilingError("project_root differs from the profiling repository")
    system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    if not isinstance(setup.device, torch.device) or setup.system is None:
        raise ProfilingError("profiling setup did not initialize CUDA")
    device = setup.device
    units = build_profile_subset(system_manifest)
    local_sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=LOCAL_ENTRY_CACHE,
        manifest_path=LOCAL_CACHE_MANIFEST,
        allowed_master_sequence_ids=tuple(unit.master_sequence_id for unit in units),
    )
    sequence_by_scope = {
        (sequence.master_sequence_id, sequence.order_id): sequence
        for sequence in local_sequences
    }
    settings = setup.p6a_config["baselines"]["b4"]
    tracker_factory = build_tracker_factories(setup.p6a_config)["B4"]

    def prepare(unit: ProfileUnit, indices: tuple[int, ...]) -> SimpleNamespace:
        sample = setup.dataset.load_scan_indices(
            unit.context_index,
            indices,
            change_file=None,
        )
        data, targets, names = setup.collate([sample])
        if list(names) != [unit.master_sequence_id] or len(targets) != 1:
            raise ProfilingError("profile collator changed sequence identity")
        target_full_values = _field(data, "target_full")
        if (
            isinstance(target_full_values, (str, bytes))
            or not isinstance(target_full_values, Sequence)
            or len(target_full_values) != 1
            or not isinstance(target_full_values[0], Mapping)
        ):
            raise ProfilingError("profile input lacks one full target")
        target_full = target_full_values[0]
        temporal_stages = target_full.get("temporal_stages")
        if not isinstance(temporal_stages, Tensor) or temporal_stages.ndim != 1:
            raise ProfilingError("profile target temporal stages are invalid")
        input_bytes = model_input_storage_bytes(data)
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target = targets[0]
        stages = _segment_stages(target)
        raw_coordinates = setup.system._process_raw_coordinates(data)
        return SimpleNamespace(
            data=data,
            target=target,
            stages=stages,
            raw_coordinates=raw_coordinates,
            point_count=int(temporal_stages.numel()),
            input_bytes=input_bytes,
        )

    rows: list[dict[str, object]] = []
    with deterministic_inference_runtime(
        int(setup.p6a_config["protocol_b"]["seed"]), device
    ):
        for unit in units:
            sequence = sequence_by_scope[(unit.master_sequence_id, "canonical")]
            cached_observations = tuple(
                _move_frozen_observation(
                    cache_payload_to_frozen_observation(payload), device
                )
                for payload in sequence.payloads
            )
            cumulative_points = {"FullHistory": 0, "Persist4D": 0}
            cumulative_bytes = {"FullHistory": 0, "Persist4D": 0}
            for horizon in (2, 3, 4, 5):
                for method in ("FullHistory", "Persist4D"):
                    base = {
                        "method": method,
                        "reference_scene_id": unit.reference_scene_id,
                        "master_sequence_id": unit.master_sequence_id,
                        "order_id": unit.order_id,
                        "horizon": horizon,
                    }
                    try:
                        torch.cuda.empty_cache()
                        if method == "FullHistory":
                            prepared = prepare(unit, unit.scan_indices[:horizon])

                            def operation_factory(prepared=prepared):
                                def operation(prepared=prepared):
                                    with torch.inference_mode():
                                        return setup.system(
                                            prepared.data,
                                            point2segment=[
                                                prepared.target["point2segment"]
                                            ],
                                            raw_coordinates=prepared.raw_coordinates,
                                            is_eval=True,
                                        )

                                return operation

                            update_scans = horizon
                            state_bytes = None
                            explicit_history_bytes = prepared.input_bytes
                        else:
                            prepared = prepare(
                                unit,
                                unit.scan_indices[horizon - 2 : horizon],
                            )
                            latest_stage = int(prepared.stages.max().item())

                            def model_forward(
                                prepared=prepared,
                                latest_stage=latest_stage,
                            ):
                                with torch.inference_mode():
                                    output = setup.system(
                                        prepared.data,
                                        point2segment=[
                                            prepared.target["point2segment"]
                                        ],
                                        raw_coordinates=prepared.raw_coordinates,
                                        is_eval=True,
                                    )
                                    return build_local_observation(
                                        output,
                                        [prepared.stages],
                                        latest_stage=latest_stage,
                                        background_class=int(
                                            settings["background_class"]
                                        ),
                                        confidence_threshold=float(
                                            settings["confidence_threshold"]
                                        ),
                                        mask_threshold=float(
                                            settings["mask_threshold"]
                                        ),
                                        minimum_mask_support=int(
                                            settings["minimum_mask_support"]
                                        ),
                                    )

                            operation_factory = build_persistent_operation_factory(
                                model_forward=model_forward,
                                tracker_factory=lambda unit_id=unit.master_sequence_id: tracker_factory(
                                    f"profile:{unit_id}"
                                ),
                                prior_observations=cached_observations[: horizon - 1],
                                current_stage=horizon - 1,
                            )
                            state_tracker = tracker_factory(
                                f"state:{unit.master_sequence_id}"
                            )
                            for stage, observation in enumerate(
                                cached_observations[:horizon]
                            ):
                                state_tracker.step(observation, stage_id=stage)
                            if state_tracker.state is None:
                                raise ProfilingError("B4 state was not initialized")
                            state_bytes = persistent_state_storage_bytes(
                                state_tracker.state
                            )
                            explicit_history_bytes = None
                            update_scans = 2
                        cumulative_points[method] += prepared.point_count
                        cumulative_bytes[method] += prepared.input_bytes
                        profile = measure_cuda_repeats(
                            operation_factory,
                            device=device,
                            warmup_repeats=PROTOCOL_WARMUP_REPEATS,
                            measured_repeats=PROTOCOL_MEASURED_REPEATS,
                            enforce_protocol=True,
                        )
                        rows.append(
                            {
                                **base,
                                "status": "pass",
                                "error_type": "",
                                "error_message": "",
                                "median_latency_ms": profile.median_ms,
                                "mean_latency_ms": profile.mean_ms,
                                "std_latency_ms": profile.std_ms,
                                "peak_allocated_bytes": profile.peak_allocated_bytes,
                                "peak_reserved_bytes": profile.peak_reserved_bytes,
                                "peak_allocated_mib": profile.peak_allocated_bytes
                                / (1024**2),
                                "peak_reserved_mib": profile.peak_reserved_bytes
                                / (1024**2),
                                "update_scan_count": update_scans,
                                "cumulative_scan_count": (
                                    horizon * (horizon + 1) // 2
                                    - 1
                                    if method == "FullHistory"
                                    else 2 * (horizon - 1)
                                ),
                                "update_point_count": prepared.point_count,
                                "cumulative_point_count": cumulative_points[method],
                                "model_input_bytes": prepared.input_bytes,
                                "cumulative_model_input_bytes": cumulative_bytes[method],
                                "persistent_state_bytes": state_bytes,
                                "explicit_history_input_bytes": explicit_history_bytes,
                            }
                        )
                    except Exception as error:  # noqa: BLE001 - preserve failed cell.
                        torch.cuda.empty_cache()
                        rows.append(
                            {
                                **base,
                                "status": "fail",
                                "error_type": type(error).__name__,
                                "error_message": str(error).replace("\n", " ")[:500],
                                "median_latency_ms": None,
                                "mean_latency_ms": None,
                                "std_latency_ms": None,
                                "peak_allocated_bytes": None,
                                "peak_reserved_bytes": None,
                                "peak_allocated_mib": None,
                                "peak_reserved_mib": None,
                                "update_scan_count": None,
                                "cumulative_scan_count": None,
                                "update_point_count": None,
                                "cumulative_point_count": None,
                                "model_input_bytes": None,
                                "cumulative_model_input_bytes": None,
                                "persistent_state_bytes": None,
                                "explicit_history_input_bytes": None,
                            }
                        )
    fields = (
        "method",
        "reference_scene_id",
        "master_sequence_id",
        "order_id",
        "horizon",
        "status",
        "error_type",
        "error_message",
        "median_latency_ms",
        "mean_latency_ms",
        "std_latency_ms",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "update_scan_count",
        "cumulative_scan_count",
        "update_point_count",
        "cumulative_point_count",
        "model_input_bytes",
        "cumulative_model_input_bytes",
        "persistent_state_bytes",
        "explicit_history_input_bytes",
    )
    rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            int(row["horizon"]),
            str(row["method"]),
        )
    )
    _publish_exact(
        SYSTEM_ROOT / "profile_results.csv",
        _csv_bytes(rows, fields),
    )
    failures = sum(row["status"] != "pass" for row in rows)
    result = {
        "schema_version": 1,
        "status": "pass" if failures == 0 else "fail",
        "source_commit": binding["source_commit"],
        "row_count": len(rows),
        "failure_count": failures,
        "warmup_repeats": PROTOCOL_WARMUP_REPEATS,
        "measured_repeats": PROTOCOL_MEASURED_REPEATS,
    }
    _publish_exact(
        SYSTEM_ROOT / "profile_summary.json",
        (
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    )
    if failures:
        raise ProfilingError(f"system profile retained {failures} failed cells")
    from scripts.system_comparison_analysis import run_statistical_analysis

    run_statistical_analysis(project_root=project_root)
    return result


__all__ = [
    "PROTOCOL_MEASURED_REPEATS",
    "PROTOCOL_WARMUP_REPEATS",
    "CudaRepeatProfile",
    "ProfileUnit",
    "ProfilingError",
    "build_persistent_operation_factory",
    "build_profile_subset",
    "measure_cuda_repeats",
    "persistent_state_storage_bytes",
    "run_system_profile",
]
