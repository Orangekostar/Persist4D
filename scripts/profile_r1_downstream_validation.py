"""Profile frozen R1 FullHistory and B4 deployment paths."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
DEFAULT_CACHE_ROOT = Path("/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache")
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_DATA_ROOT = Path("/home/ww/paper5")
METHODS = ("FullHistory", "B4")
HORIZONS = (2, 4, 5)
WARMUP_REPEATS = 5
MEASURED_REPEATS = 10


class R1ProfileError(ValueError):
    """Raised when R1 profiling differs from the frozen contract."""


def _cell(row: Mapping[str, object]) -> tuple[str, str, str, str, int]:
    try:
        result = (
            str(row["method"]),
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise R1ProfileError("profile labels are invalid") from error
    if (
        result[0] not in METHODS
        or not result[1]
        or not result[2]
        or result[3] != "canonical"
        or result[4] not in HORIZONS
        or row.get("warmup_repeats") != WARMUP_REPEATS
        or row.get("measured_repeats") != MEASURED_REPEATS
    ):
        raise R1ProfileError("profile labels or repeat contract differ")
    return result


def validate_profile_coverage(
    *,
    sample_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    if not sample_rows or not summary_rows:
        raise R1ProfileError("profile coverage is empty")
    summary_cells = [_cell(row) for row in summary_rows]
    references = {cell[1] for cell in summary_cells}
    masters = {cell[2] for cell in summary_cells}
    expected_cells = {
        (method, reference, master, "canonical", horizon)
        for reference, master in {
            (cell[1], cell[2]) for cell in summary_cells
        }
        for method in METHODS
        for horizon in HORIZONS
    }
    if (
        len(references) != 6
        or len(masters) != 6
        or len({(cell[1], cell[2]) for cell in summary_cells}) != 6
        or len(summary_cells) != len(set(summary_cells))
        or set(summary_cells) != expected_cells
    ):
        raise R1ProfileError("profile summary coverage differs")
    sample_cells = []
    for row in sample_rows:
        cell = _cell(row)
        index = row.get("sample_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise R1ProfileError("profile sample index is invalid")
        sample_cells.append((*cell, index))
    expected_samples = {
        (*cell, index)
        for cell in expected_cells
        for index in range(MEASURED_REPEATS)
    }
    if (
        len(sample_cells) != len(set(sample_cells))
        or set(sample_cells) != expected_samples
    ):
        raise R1ProfileError("profile sample coverage differs")
    return {
        "profile_unit_count": len(references),
        "summary_row_count": len(summary_rows),
        "sample_row_count": len(sample_rows),
    }


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _profile_observations(
    *,
    cache_root: Path,
    local_progress: Mapping[str, object],
    master_ids: Sequence[str],
) -> dict[str, tuple[object, ...]]:
    from scripts.evaluate_persist4d_p6a import (
        cache_payload_to_frozen_observation,
    )
    from scripts.p6a_cache import validate_cache_entry
    from scripts.system_comparison_v2_cache import observation_fingerprint

    records = local_progress.get("records")
    provenance = local_progress.get("provenance")
    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, Sequence)
        or not isinstance(provenance, Mapping)
    ):
        raise R1ProfileError("local cache progress is invalid")
    allowed = set(master_ids)
    grouped: dict[str, dict[int, object]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("key"), Mapping):
            raise R1ProfileError("local cache record is invalid")
        key = record["key"]
        master = str(key.get("master_sequence_id"))
        if master not in allowed or key.get("order_id") != "canonical":
            continue
        raw_entry = record.get("raw_entry")
        if not isinstance(raw_entry, Mapping):
            raise R1ProfileError("local cache raw entry is invalid")
        raw = validate_cache_entry(
            cache_root / "raw_predictions/entries" / str(raw_entry["filename"]),
            raw_entry,
            expected_provenance=provenance,
        )
        if observation_fingerprint(raw) != record.get(
            "raw_observation_fingerprint"
        ):
            raise R1ProfileError("profile observation fingerprint differs")
        stage = int(key["stage_index"])
        if stage in grouped.setdefault(master, {}):
            raise R1ProfileError("profile cache contains duplicate stages")
        grouped[master][stage] = cache_payload_to_frozen_observation(raw)
    if set(grouped) != allowed or any(
        set(stages) != set(range(5)) for stages in grouped.values()
    ):
        raise R1ProfileError("profile cache coverage differs")
    return {
        master: tuple(stages[index] for index in range(5))
        for master, stages in grouped.items()
    }


def run_profile(
    *,
    device_name: str,
    cache_root: Path,
    artifact_root: Path,
    contract_path: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    metadata_path: Path,
    data_root: Path,
) -> dict[str, object]:
    from models.persistent_memory import build_local_observation
    from scripts.analyze_r1_downstream_validation import (
        _atomic_write,
        _csv_bytes,
        _git_head,
        _validate_new_cache_binding,
    )
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import build_tracker_factories
    from scripts.profile_system_comparison import (
        _move_frozen_observation,
        build_persistent_operation_factory,
        build_profile_subset,
        measure_cuda_repeats,
        persistent_state_storage_bytes,
    )
    from scripts.r1_downstream_context import build_r1_setup
    from scripts.run_r1_downstream_validation import (
        _release_cuda,
        _require_clean_tracked_tree,
        validate_cache_execution,
    )
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        model_input_storage_bytes,
    )

    _require_clean_tracked_tree()
    validate_cache_execution(device_name)
    setup = None
    try:
        _compact, local_progress, _full_progress = _validate_new_cache_binding(
            cache_root=cache_root, artifact_root=artifact_root
        )
        setup = build_r1_setup(
            contract_path=contract_path,
            protocol_path=protocol_path,
            checkpoint_path=checkpoint_path,
            pretrained_path=pretrained_path,
            metadata_path=metadata_path,
            data_root=data_root,
            source_commit=_git_head(),
            device_name=device_name,
        )
        if not isinstance(setup.device, torch.device) or setup.system is None:
            raise R1ProfileError("profiling setup did not initialize CUDA")
        device = setup.device
        units = build_profile_subset(setup.protocol_manifest)
        observations = _profile_observations(
            cache_root=cache_root,
            local_progress=local_progress,
            master_ids=tuple(unit.master_sequence_id for unit in units),
        )
        settings = setup.p6a_config["baselines"]["b4"]
        tracker_factory = build_tracker_factories(setup.p6a_config)["B4"]

        def load_input(
            unit: object, indices: tuple[int, ...], *, move: bool
        ) -> SimpleNamespace:
            sample = setup.dataset.load_scan_indices(
                unit.context_index,
                indices,
                change_file=None,
            )
            data, targets, names = setup.collate([sample])
            if list(names) != [unit.master_sequence_id] or len(targets) != 1:
                raise R1ProfileError("profile collator changed sequence identity")
            target_full_values = _field(data, "target_full")
            if (
                isinstance(target_full_values, (str, bytes))
                or not isinstance(target_full_values, Sequence)
                or len(target_full_values) != 1
                or not isinstance(target_full_values[0], Mapping)
            ):
                raise R1ProfileError("profile input lacks one full target")
            temporal_stages = target_full_values[0].get("temporal_stages")
            if not isinstance(temporal_stages, Tensor) or temporal_stages.ndim != 1:
                raise R1ProfileError("profile temporal stages are invalid")
            point_count = int(temporal_stages.numel())
            input_bytes = model_input_storage_bytes(data)
            if not move:
                return SimpleNamespace(
                    point_count=point_count,
                    input_bytes=input_bytes,
                )
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
                point_count=point_count,
                input_bytes=input_bytes,
            )

        sample_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        with deterministic_inference_runtime(45, device):
            for unit_index, unit in enumerate(units, start=1):
                cached = tuple(
                    _move_frozen_observation(observation, device)
                    for observation in observations[unit.master_sequence_id]
                )
                input_stats = {}
                for horizon in range(2, 6):
                    for method in METHODS:
                        indices = (
                            unit.scan_indices[:horizon]
                            if method == "FullHistory"
                            else unit.scan_indices[horizon - 2 : horizon]
                        )
                        stats = load_input(unit, indices, move=False)
                        input_stats[(method, horizon)] = (
                            stats.point_count,
                            stats.input_bytes,
                        )
                for horizon in HORIZONS:
                    for method in METHODS:
                        torch.cuda.empty_cache()
                        indices = (
                            unit.scan_indices[:horizon]
                            if method == "FullHistory"
                            else unit.scan_indices[horizon - 2 : horizon]
                        )
                        prepared = load_input(unit, indices, move=True)
                        if method == "FullHistory":

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

                            persistent_bytes = None
                            explicit_history_bytes = prepared.input_bytes
                            update_scan_count = horizon
                        else:
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
                                    f"r1-profile:{unit_id}"
                                ),
                                prior_observations=cached[: horizon - 1],
                                current_stage=horizon - 1,
                            )
                            state_tracker = tracker_factory(
                                f"r1-profile-state:{unit.master_sequence_id}"
                            )
                            for stage, observation in enumerate(cached[:horizon]):
                                state_tracker.step(observation, stage_id=stage)
                            if state_tracker.state is None:
                                raise R1ProfileError("B4 profile state is unavailable")
                            persistent_bytes = persistent_state_storage_bytes(
                                state_tracker.state
                            )
                            explicit_history_bytes = None
                            update_scan_count = 2
                        profile = measure_cuda_repeats(
                            operation_factory,
                            device=device,
                            warmup_repeats=WARMUP_REPEATS,
                            measured_repeats=MEASURED_REPEATS,
                            enforce_protocol=True,
                        )
                        base = {
                            "method": method,
                            "reference_scene_id": unit.reference_scene_id,
                            "master_sequence_id": unit.master_sequence_id,
                            "order_id": "canonical",
                            "horizon": horizon,
                            "warmup_repeats": WARMUP_REPEATS,
                            "measured_repeats": MEASURED_REPEATS,
                        }
                        for sample_index, latency in enumerate(profile.samples_ms):
                            allocated = profile.allocated_samples_bytes[sample_index]
                            reserved = profile.reserved_samples_bytes[sample_index]
                            sample_rows.append(
                                {
                                    **base,
                                    "sample_index": sample_index,
                                    "latency_ms": latency,
                                    "peak_allocated_bytes": allocated,
                                    "peak_reserved_bytes": reserved,
                                    "peak_allocated_mib": allocated / (1024**2),
                                    "peak_reserved_mib": reserved / (1024**2),
                                }
                            )
                        cumulative_points = sum(
                            input_stats[(method, value)][0]
                            for value in range(2, horizon + 1)
                        )
                        cumulative_bytes = sum(
                            input_stats[(method, value)][1]
                            for value in range(2, horizon + 1)
                        )
                        summary_rows.append(
                            {
                                **base,
                                "median_latency_ms": profile.median_ms,
                                "mean_latency_ms": profile.mean_ms,
                                "std_latency_ms": profile.std_ms,
                                "peak_allocated_bytes": profile.peak_allocated_bytes,
                                "peak_reserved_bytes": profile.peak_reserved_bytes,
                                "peak_allocated_mib": profile.peak_allocated_bytes
                                / (1024**2),
                                "peak_reserved_mib": profile.peak_reserved_bytes
                                / (1024**2),
                                "update_scan_count": update_scan_count,
                                "cumulative_scan_count": (
                                    horizon * (horizon + 1) // 2 - 1
                                    if method == "FullHistory"
                                    else 2 * (horizon - 1)
                                ),
                                "update_point_count": prepared.point_count,
                                "cumulative_point_count": cumulative_points,
                                "model_input_bytes": prepared.input_bytes,
                                "cumulative_model_input_bytes": cumulative_bytes,
                                "persistent_state_bytes": persistent_bytes,
                                "explicit_history_input_bytes": explicit_history_bytes,
                                "measurement_boundary": (
                                    "cuda_synchronized_model_forward_plus_cpu_tracker"
                                    if method == "B4"
                                    else "cuda_synchronized_model_forward"
                                ),
                                "excluded": "io_collate_h2d_metrics_and_tracker_preroll",
                            }
                        )
                print(
                    f"[r1-profile] completed {unit_index}/{len(units)} units",
                    file=sys.stderr,
                    flush=True,
                )
        coverage = validate_profile_coverage(
            sample_rows=sample_rows,
            summary_rows=summary_rows,
        )
        profile_root = artifact_root / "profile"
        _atomic_write(profile_root / "samples.csv", _csv_bytes(sample_rows))
        _atomic_write(profile_root / "summary.csv", _csv_bytes(summary_rows))
        return {
            "status": "pass",
            "source_commit": _git_head(),
            "device": device_name,
            "coverage": coverage,
        }
    finally:
        if setup is not None:
            _release_cuda(setup)


def argument_parser() -> argparse.ArgumentParser:
    from scripts.run_r1_downstream_validation import (
        DEFAULT_CHECKPOINT,
        DEFAULT_CONTRACT,
        DEFAULT_PRETRAINED,
        DEFAULT_PROTOCOL,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    result = run_profile(
        device_name=arguments.device,
        cache_root=arguments.cache_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        protocol_path=arguments.protocol,
        checkpoint_path=arguments.checkpoint,
        pretrained_path=arguments.pretrained,
        metadata_path=arguments.metadata,
        data_root=arguments.data_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
