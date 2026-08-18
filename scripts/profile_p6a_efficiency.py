"""Collect raw P6-A efficiency evidence under frozen Protocol B."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p6a_analysis import persistent_state_bytes
from scripts.p6a_association import B4PersistentTracker, FrozenObservation
from scripts.p6a_efficiency import build_efficiency_manifest

ORDER_IDS = ("canonical", "reverse", "sha256_seed45")
EFFICIENCY_CONFIG = {
    "warmup_per_group": 1,
    "measurements_per_unit": 1,
    "bootstrap_records": 129,
    "new_visit_records_per_horizon": 129,
    "full_history_records_per_horizon": 129,
    "total_records": 1161,
    "seed_per_sample": 45,
    "local_input": "latest_pair_plus_persistent_state",
    "full_history_input": "exact_common_prefix",
    "latency_boundary": "cuda_synchronized_model_forward_plus_cpu_tracker",
    "excluded_from_latency": [
        "dataset_load",
        "collate",
        "host_to_device",
        "metric_postprocess",
    ],
    "gpu_peak_memory": "torch_cuda_max_memory_allocated",
    "persistent_state_memory": "exact_tensor_storage_bytes",
}


def validate_efficiency_config(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("efficiency config must be a mapping")
    if dict(value) != EFFICIENCY_CONFIG:
        raise ValueError("efficiency config differs from the preregistered contract")


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be finite and non-negative")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class ModelMeasurement:
    latency_ms: float
    gpu_peak_memory_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "latency_ms",
            _nonnegative_float(self.latency_ms, name="latency_ms"),
        )
        object.__setattr__(
            self,
            "gpu_peak_memory_bytes",
            _positive_integer(
                self.gpu_peak_memory_bytes,
                name="gpu_peak_memory_bytes",
            ),
        )


@dataclass(frozen=True)
class TrackerMeasurement:
    latency_ms: float
    association_overhead_ms: float
    memory_update_overhead_ms: float
    persistent_state_bytes: int

    def __post_init__(self) -> None:
        latency = _nonnegative_float(self.latency_ms, name="latency_ms")
        association = _nonnegative_float(
            self.association_overhead_ms,
            name="association_overhead_ms",
        )
        update = _nonnegative_float(
            self.memory_update_overhead_ms,
            name="memory_update_overhead_ms",
        )
        if latency + 1e-9 < association + update:
            raise ValueError("tracker latency must cover association and memory update")
        object.__setattr__(self, "latency_ms", latency)
        object.__setattr__(self, "association_overhead_ms", association)
        object.__setattr__(self, "memory_update_overhead_ms", update)
        object.__setattr__(
            self,
            "persistent_state_bytes",
            _positive_integer(
                self.persistent_state_bytes,
                name="persistent_state_bytes",
            ),
        )


@dataclass(frozen=True)
class EfficiencyProfileUnit:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    context_index: int
    context_scan_indices: tuple[int, ...]
    scan_indices: tuple[int, ...]
    observations: tuple[FrozenObservation, ...]

    def __post_init__(self) -> None:
        _nonempty(self.reference_scene_id, name="reference_scene_id")
        _nonempty(self.master_sequence_id, name="master_sequence_id")
        if self.order_id not in ORDER_IDS:
            raise ValueError(f"order_id must be one of {ORDER_IDS}")
        if (
            isinstance(self.context_index, bool)
            or not isinstance(self.context_index, Integral)
            or self.context_index < 0
        ):
            raise ValueError("context_index must be a non-negative integer")
        for name, values in (
            ("context_scan_indices", self.context_scan_indices),
            ("scan_indices", self.scan_indices),
        ):
            if (
                len(values) != 5
                or len(set(values)) != 5
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, Integral)
                    or index < 0
                    for index in values
                )
            ):
                raise ValueError(
                    f"{name} must contain five unique non-negative integers"
                )
        if len(self.observations) != 5:
            raise ValueError("observations must contain five causal stages")
        for observation in self.observations:
            if not isinstance(observation, FrozenObservation):
                raise TypeError("observations must be FrozenObservation values")
            observation.validate()


def _validate_units(
    units: Sequence[EfficiencyProfileUnit],
) -> tuple[EfficiencyProfileUnit, ...]:
    if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
        raise TypeError("units must be a sequence")
    normalized = tuple(units)
    if len(normalized) != 129 or any(
        not isinstance(unit, EfficiencyProfileUnit) for unit in normalized
    ):
        raise ValueError("efficiency profiling requires exactly 129 units")
    by_master: dict[str, list[EfficiencyProfileUnit]] = {}
    for unit in normalized:
        by_master.setdefault(unit.master_sequence_id, []).append(unit)
    if len(by_master) != 43:
        raise ValueError("efficiency profiling requires exactly 43 masters")
    if len({unit.reference_scene_id for unit in normalized}) != 6:
        raise ValueError("efficiency profiling requires exactly 6 reference clusters")
    if any(
        len(master_units) != 3
        or {unit.order_id for unit in master_units} != set(ORDER_IDS)
        or len({unit.reference_scene_id for unit in master_units}) != 1
        for master_units in by_master.values()
    ):
        raise ValueError("each master must contain the three registered orders")
    return normalized


def _identity_fields(
    unit: EfficiencyProfileUnit,
    *,
    horizon: int,
    row_type: str,
) -> dict[str, object]:
    return {
        "reference_scene_id": unit.reference_scene_id,
        "master_sequence_id": unit.master_sequence_id,
        "order_id": unit.order_id,
        "T": horizon,
        "stage_id": 0 if row_type == "bootstrap" else horizon - 1,
        "row_type": row_type,
    }


def _method_call(
    callback: Callable[..., Any],
    *args: object,
    warmup: bool,
) -> Any:
    return callback(*args, warmup=warmup)


def collect_efficiency_records(
    units: Sequence[EfficiencyProfileUnit],
    *,
    measure_model: Callable[..., ModelMeasurement],
    measure_tracker: Callable[..., TrackerMeasurement],
) -> list[dict[str, object]]:
    """Measure nine groups with one unrecorded warmup per group."""

    normalized = _validate_units(units)
    first = normalized[0]
    records: list[dict[str, object]] = []

    _method_call(
        measure_model,
        first,
        first.scan_indices[:1],
        "bootstrap",
        1,
        warmup=True,
    )
    _method_call(measure_tracker, first, 1, warmup=True)
    for unit in normalized:
        model = _method_call(
            measure_model,
            unit,
            unit.scan_indices[:1],
            "bootstrap",
            1,
            warmup=False,
        )
        tracker = _method_call(measure_tracker, unit, 1, warmup=False)
        records.append(
            {
                **_identity_fields(unit, horizon=1, row_type="bootstrap"),
                "model_latency_ms": model.latency_ms,
                "tracker_latency_ms": tracker.latency_ms,
                "association_overhead_ms": tracker.association_overhead_ms,
                "memory_update_overhead_ms": tracker.memory_update_overhead_ms,
                "gpu_peak_memory_bytes": model.gpu_peak_memory_bytes,
                "persistent_state_bytes": tracker.persistent_state_bytes,
            }
        )

    for horizon in range(2, 6):
        local_indices = first.scan_indices[horizon - 2 : horizon]
        _method_call(
            measure_model,
            first,
            local_indices,
            "new_visit",
            horizon,
            warmup=True,
        )
        _method_call(measure_tracker, first, horizon, warmup=True)
        for unit in normalized:
            model = _method_call(
                measure_model,
                unit,
                unit.scan_indices[horizon - 2 : horizon],
                "new_visit",
                horizon,
                warmup=False,
            )
            tracker = _method_call(
                measure_tracker,
                unit,
                horizon,
                warmup=False,
            )
            records.append(
                {
                    **_identity_fields(
                        unit,
                        horizon=horizon,
                        row_type="new_visit",
                    ),
                    "model_latency_ms": model.latency_ms,
                    "tracker_latency_ms": tracker.latency_ms,
                    "association_overhead_ms": tracker.association_overhead_ms,
                    "memory_update_overhead_ms": tracker.memory_update_overhead_ms,
                    "gpu_peak_memory_bytes": model.gpu_peak_memory_bytes,
                    "persistent_state_bytes": tracker.persistent_state_bytes,
                }
            )

    for horizon in range(2, 6):
        _method_call(
            measure_model,
            first,
            first.scan_indices[:horizon],
            "full_history",
            horizon,
            warmup=True,
        )
        for unit in normalized:
            model = _method_call(
                measure_model,
                unit,
                unit.scan_indices[:horizon],
                "full_history",
                horizon,
                warmup=False,
            )
            records.append(
                {
                    **_identity_fields(
                        unit,
                        horizon=horizon,
                        row_type="full_history",
                    ),
                    "model_latency_ms": model.latency_ms,
                    "tracker_latency_ms": None,
                    "association_overhead_ms": None,
                    "memory_update_overhead_ms": None,
                    "gpu_peak_memory_bytes": model.gpu_peak_memory_bytes,
                    "persistent_state_bytes": None,
                }
            )
    return records


def measure_cuda_operation(
    operation: Callable[[], Any],
    *,
    device: torch.device,
    synchronize: Callable[[torch.device], object] = torch.cuda.synchronize,
    reset_peak: Callable[[torch.device], object] = torch.cuda.reset_peak_memory_stats,
    peak_memory: Callable[[torch.device], int] = torch.cuda.max_memory_allocated,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[Any, ModelMeasurement]:
    """Measure one CUDA operation after setup and before postprocessing."""

    if device.type != "cuda" or device.index is None:
        raise ValueError("device must identify one CUDA device")
    synchronize(device)
    reset_peak(device)
    start_ns = clock_ns()
    output = operation()
    synchronize(device)
    end_ns = clock_ns()
    if end_ns < start_ns:
        raise RuntimeError("monotonic clock moved backwards")
    return output, ModelMeasurement(
        latency_ms=(end_ns - start_ns) / 1_000_000.0,
        gpu_peak_memory_bytes=peak_memory(device),
    )


def _state_storage_bytes(state: object) -> int:
    tensors = getattr(state, "tensors", None)
    if not callable(tensors):
        raise TypeError("persistent state must expose tensors()")
    values = tensors()
    if not isinstance(values, tuple) or not values:
        raise ValueError("persistent state tensors must be a non-empty tuple")
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("persistent state values must be tensors")
    return sum(value.numel() * value.element_size() for value in values)


def measure_b4_tracker(
    observations: Sequence[FrozenObservation],
    *,
    horizon: int,
    tracker_settings: Mapping[str, object],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> TrackerMeasurement:
    """Time the final B4 transition after untimed causal cache pre-roll."""

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, Integral)
        or not 1 <= horizon <= len(observations)
    ):
        raise ValueError("horizon must select a non-empty observation prefix")
    required = {
        "capacity",
        "class_weight",
        "association_threshold",
        "update_rate",
    }
    if not isinstance(tracker_settings, Mapping) or not required.issubset(
        tracker_settings
    ):
        raise ValueError("tracker_settings are incomplete")
    tracker = B4PersistentTracker(
        sequence_id="efficiency-profile",
        capacity=int(tracker_settings["capacity"]),
        class_weight=float(tracker_settings["class_weight"]),
        association_threshold=float(tracker_settings["association_threshold"]),
        update_rate=float(tracker_settings["update_rate"]),
        max_update_rate=float(tracker_settings["update_rate"]),
    )
    for stage_id in range(int(horizon) - 1):
        tracker.step(observations[stage_id], stage_id=stage_id)
    timing: list[Mapping[str, float]] = []
    start_ns = clock_ns()
    result = tracker.step(
        observations[int(horizon) - 1],
        stage_id=int(horizon) - 1,
        timing_sink=timing.append,
    )
    end_ns = clock_ns()
    if end_ns < start_ns or len(timing) != 1:
        raise RuntimeError("B4 timing instrumentation did not publish one record")
    state = result.state_snapshot
    if state is None:
        raise RuntimeError("B4 transition did not publish state")
    actual_bytes = _state_storage_bytes(state)
    expected_bytes = persistent_state_bytes(
        state.capacity,
        state.feature_dim,
        state.class_count,
        batch_size=state.batch_size,
        dtype_bytes=state.embedding.element_size(),
        index_bytes=state.age.element_size(),
        bool_bytes=state.occupied.element_size(),
    )
    if actual_bytes != expected_bytes:
        raise RuntimeError("persistent state byte accounting is inconsistent")
    return TrackerMeasurement(
        latency_ms=(end_ns - start_ns) / 1_000_000.0,
        association_overhead_ms=float(timing[0]["association_overhead_ms"]),
        memory_update_overhead_ms=float(timing[0]["memory_update_overhead_ms"]),
        persistent_state_bytes=actual_bytes,
    )


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def build_efficiency_profile_units(
    protocol: object,
    sequences: Sequence[object],
    *,
    observation_loader: Callable[[Mapping[str, object]], FrozenObservation]
    | None = None,
) -> tuple[EfficiencyProfileUnit, ...]:
    """Join exact Protocol B scan orders with their frozen cache observations."""

    if observation_loader is None:
        from scripts.evaluate_persist4d_p6a import (
            cache_payload_to_frozen_observation,
        )

        observation_loader = cache_payload_to_frozen_observation
    masters = _field(protocol, "masters")
    variants = _field(protocol, "variants")
    if (
        isinstance(masters, (str, bytes))
        or not isinstance(masters, Sequence)
        or not isinstance(variants, Mapping)
    ):
        raise TypeError("protocol must expose masters and variants")
    master_by_id = {str(_field(master, "sequence_id")): master for master in masters}
    units: list[EfficiencyProfileUnit] = []
    for sequence in sequences:
        master_id = _nonempty(
            _field(sequence, "master_sequence_id"),
            name="master_sequence_id",
        )
        reference_id = _nonempty(
            _field(sequence, "reference_scene_id"),
            name="reference_scene_id",
        )
        order_id = _nonempty(_field(sequence, "order_id"), name="order_id")
        if master_id not in master_by_id:
            raise ValueError("cached sequence master is absent from Protocol B")
        master = master_by_id[master_id]
        if _field(master, "reference_scene_id") != reference_id:
            raise ValueError("cached sequence reference differs from Protocol B")
        by_order = variants.get(master_id)
        if not isinstance(by_order, Mapping) or order_id not in by_order:
            raise ValueError("cached sequence order is absent from Protocol B")
        raw_indices = _field(by_order[order_id], "scan_indices")
        raw_context_indices = _field(master, "scan_indices")
        raw_payloads = _field(sequence, "payloads")
        if (
            isinstance(raw_indices, (str, bytes))
            or not isinstance(raw_indices, Sequence)
            or isinstance(raw_context_indices, (str, bytes))
            or not isinstance(raw_context_indices, Sequence)
            or isinstance(raw_payloads, (str, bytes))
            or not isinstance(raw_payloads, Sequence)
        ):
            raise TypeError("Protocol B scans and cache payloads must be sequences")
        observations = []
        for payload in raw_payloads:
            if not isinstance(payload, Mapping):
                raise TypeError("cache payloads must be mappings")
            observations.append(observation_loader(payload))
        units.append(
            EfficiencyProfileUnit(
                reference_scene_id=reference_id,
                master_sequence_id=master_id,
                order_id=order_id,
                context_index=int(_field(master, "validation_index")),
                context_scan_indices=tuple(
                    int(index) for index in raw_context_indices
                ),
                scan_indices=tuple(int(index) for index in raw_indices),
                observations=tuple(observations),
            )
        )
    return _validate_units(units)


@dataclass
class RealModelProfiler:
    """Measure only ReScene forward on freshly prepared Protocol B inputs."""

    dataset: object
    collate: Callable[[list[object]], tuple[object, object, object]]
    system: object
    device: torch.device
    move_data: Callable[[object, torch.device], object]
    move_targets: Callable[[object, torch.device], object]
    measure_operation: Callable[..., tuple[object, ModelMeasurement]] = (
        measure_cuda_operation
    )
    seed: int = 45

    def __call__(
        self,
        unit: EfficiencyProfileUnit,
        scan_indices: tuple[int, ...],
        row_type: str,
        horizon: int,
        *,
        warmup: bool,
    ) -> ModelMeasurement:
        if row_type not in {"bootstrap", "new_visit", "full_history"}:
            raise ValueError("unsupported efficiency row_type")
        if isinstance(warmup, bool) is False:
            raise ValueError("warmup must be boolean")
        if horizon != len(scan_indices) and not (
            row_type == "new_visit" and len(scan_indices) == 2
        ):
            raise ValueError("scan_indices do not match the measured horizon")
        names = _field(self.dataset, "sequence_names")
        indices = _field(self.dataset, "sequence_indices")
        if (
            isinstance(names, (str, bytes))
            or not isinstance(names, Sequence)
            or unit.context_index >= len(names)
            or names[unit.context_index] != unit.master_sequence_id
            or isinstance(indices, (str, bytes))
            or not hasattr(indices, "__getitem__")
        ):
            raise ValueError("dataset context differs from the efficiency unit")
        if tuple(int(index) for index in indices[unit.context_index]) != tuple(
            unit.context_scan_indices
        ):
            raise ValueError("dataset scan indices differ from Protocol B master")

        from scripts.evaluate_persist4d_p6a import _frozen_inference_seed

        data = targets = output = None
        try:
            with _frozen_inference_seed(self.seed, self.device):
                sample = self.dataset.load_scan_indices(
                    unit.context_index,
                    scan_indices,
                    change_file=None,
                )
                data, targets, collated_names = self.collate([sample])
                if (
                    not isinstance(targets, Sequence)
                    or len(targets) != 1
                    or list(collated_names) != [unit.master_sequence_id]
                ):
                    raise ValueError("collator changed the requested efficiency sample")
                data = self.move_data(data, self.device)
                targets = self.move_targets(targets, self.device)
                target = targets[0]
                if not isinstance(target, Mapping) or not isinstance(
                    target.get("point2segment"), torch.Tensor
                ):
                    raise TypeError("collated target is missing point2segment")
                raw_coordinates = self.system._process_raw_coordinates(data)

                def forward() -> object:
                    with torch.inference_mode():
                        return self.system(
                            data,
                            point2segment=[target["point2segment"]],
                            raw_coordinates=raw_coordinates,
                            is_eval=True,
                        )

                output, measurement = self.measure_operation(
                    forward,
                    device=self.device,
                )
                if not isinstance(output, Mapping):
                    raise TypeError("ReScene output must be a mapping")
                return measurement
        finally:
            output = None
            targets = None
            data = None


def _regular_json(path: Path, *, name: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} cannot be decoded") from error
    if not isinstance(document, Mapping):
        raise TypeError(f"{name} must decode to a mapping")
    return document


@contextmanager
def _deterministic_inference_runtime():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    precision = torch.get_float32_matmul_precision()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cuda_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)


def run_real_efficiency_profile(
    *,
    cache_directory: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    output_path: Path | None,
    device_name: str,
) -> dict[str, object]:
    """Run the frozen 1161-observation P6-A efficiency profile."""

    import hydra
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import (
        _begin_source_tree_contract,
        _compose_runtime_config,
        _finalize_source_tree_contract,
        _load_system,
        _move_data_to_device,
        _move_targets_to_device,
        _resolve_checkpoint,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        EXPECTED_RESCENE_CHECKPOINT_SHA256,
        _cache_artifact_path,
        _external_cache_directory,
        _file_sha256,
        _frozen_protocol_bundle,
        _repository_path,
        build_cache_provenance,
        load_cached_protocol_sequences,
        publish_manifest_atomic,
    )

    cache_root = _external_cache_directory(cache_directory)
    output = _cache_artifact_path(
        cache_root,
        output_path,
        filename="efficiency_raw_manifest.json",
    )
    protocol_path = cache_root / "protocol_b_manifest.json"
    cache_manifest_path = cache_root / "cache_manifest.json"
    entries_path = cache_root / "entries"
    metadata = _repository_path(metadata_path)
    guard = _begin_source_tree_contract(
        repo_root=PROJECT_ROOT,
        output_paths=(output,),
    )

    protocol, expected_protocol_manifest, p6a_bytes = _frozen_protocol_bundle(
        metadata_path=metadata
    )
    stored_protocol_manifest = _regular_json(
        protocol_path,
        name="Protocol B manifest",
    )
    if dict(stored_protocol_manifest) != expected_protocol_manifest:
        raise ValueError("stored Protocol B manifest differs from frozen inputs")
    cache_manifest = _regular_json(cache_manifest_path, name="cache manifest")
    cache_provenance = cache_manifest.get("provenance")
    if not isinstance(cache_provenance, Mapping):
        raise TypeError("cache manifest provenance is missing")

    config, _memory_config = _compose_runtime_config()
    checkpoint = _resolve_checkpoint(checkpoint_path)
    if _file_sha256(checkpoint) != EXPECTED_RESCENE_CHECKPOINT_SHA256:
        raise ValueError("formal ReScene checkpoint SHA-256 differs from P6-A")
    runtime_bytes = OmegaConf.to_yaml(
        config,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")
    expected_cache_provenance = build_cache_provenance(
        source_commit=guard.source_commit,
        checkpoint_path=checkpoint,
        config_documents={"p6a": p6a_bytes, "runtime": runtime_bytes},
        protocol_manifest=expected_protocol_manifest,
    )
    if dict(cache_provenance) != expected_cache_provenance:
        raise ValueError("cache provenance differs from the current frozen run")

    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=entries_path,
        manifest_path=cache_manifest_path,
    )
    units = build_efficiency_profile_units(protocol, sequences)
    device = _validate_cuda_device(device_name)
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    dataset_config.temporal_window = 5
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    system = _load_system(config, checkpoint, device)
    p6a_config = yaml.safe_load(p6a_bytes)
    if not isinstance(p6a_config, Mapping):
        raise TypeError("P6-A config must be a mapping")
    baselines = p6a_config.get("baselines")
    if not isinstance(baselines, Mapping) or not isinstance(
        baselines.get("b4"), Mapping
    ):
        raise TypeError("P6-A B4 settings are missing")
    efficiency_config = p6a_config.get("efficiency")
    validate_efficiency_config(efficiency_config)
    tracker_settings = dict(baselines["b4"])
    seed = int(efficiency_config["seed_per_sample"])
    if seed != int(p6a_config["protocol_b"]["seed"]):
        raise ValueError("efficiency seed differs from Protocol B")
    model_profiler = RealModelProfiler(
        dataset=dataset,
        collate=collate,
        system=system,
        device=device,
        move_data=_move_data_to_device,
        move_targets=_move_targets_to_device,
        seed=seed,
    )

    def tracker_profiler(
        unit: EfficiencyProfileUnit,
        horizon: int,
        *,
        warmup: bool,
    ) -> TrackerMeasurement:
        if not isinstance(warmup, bool):
            raise TypeError("warmup must be boolean")
        return measure_b4_tracker(
            unit.observations,
            horizon=horizon,
            tracker_settings=tracker_settings,
        )

    try:
        with _deterministic_inference_runtime():
            records = collect_efficiency_records(
                units,
                measure_model=model_profiler,
                measure_tracker=tracker_profiler,
            )
        manifest = build_efficiency_manifest(
            records,
            source_commit=guard.source_commit,
            checkpoint_sha256=str(expected_cache_provenance["checkpoint_sha256"]),
            config_sha256=str(expected_cache_provenance["config_sha256"]),
            protocol_sha256=_file_sha256(protocol_path),
            cache_manifest_sha256=_file_sha256(cache_manifest_path),
        )
        _finalize_source_tree_contract(guard)
        publish_manifest_atomic(output, manifest)
        return manifest
    finally:
        del system
        torch.cuda.empty_cache()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile frozen P6-A local and full-history efficiency.",
    )
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = run_real_efficiency_profile(
        cache_directory=args.cache_directory,
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "record_count": manifest["coverage"]["record_count"],
                "records_sha256": manifest["records_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
