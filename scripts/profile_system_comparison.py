"""CUDA timing, VRAM, and state-size primitives for system comparison."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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
]
