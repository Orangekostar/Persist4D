from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.p6a_analysis import persistent_state_bytes
from scripts.p6a_association import freeze_observation
from scripts.profile_p6a_efficiency import (
    EfficiencyProfileUnit,
    ModelMeasurement,
    RealModelProfiler,
    TrackerMeasurement,
    _argument_parser,
    build_efficiency_profile_units,
    collect_efficiency_records,
    measure_b4_tracker,
    measure_cuda_operation,
    run_real_efficiency_profile,
    validate_efficiency_config,
)

ORDERS = ("canonical", "reverse", "sha256_seed45")


def _observation(stage: int):
    return freeze_observation(
        {
            "features": torch.tensor([[1.0, float(stage)]], dtype=torch.float32),
            "class_prob": torch.tensor([[0.8, 0.2]], dtype=torch.float32),
            "confidence": torch.tensor([0.9], dtype=torch.float32),
            "valid": torch.tensor([True]),
            "latest_mask": torch.tensor([[1.0]], dtype=torch.float32),
        }
    )


def _units() -> tuple[EfficiencyProfileUnit, ...]:
    permutations = {
        "canonical": (0, 1, 2, 3, 4),
        "reverse": (4, 3, 2, 1, 0),
        "sha256_seed45": (2, 0, 4, 1, 3),
    }
    return tuple(
        EfficiencyProfileUnit(
            reference_scene_id=f"reference-{master_index % 6}",
            master_sequence_id=f"master-{master_index:02d}",
            order_id=order,
            context_index=master_index,
            context_scan_indices=tuple(
                master_index * 10 + stage for stage in range(5)
            ),
            scan_indices=tuple(
                master_index * 10 + stage for stage in permutations[order]
            ),
            observations=tuple(_observation(stage) for stage in range(5)),
        )
        for master_index in range(43)
        for order in ORDERS
    )


def test_collect_efficiency_records_has_exact_coverage_and_separate_warmups() -> None:
    model_calls: list[tuple[str, int, tuple[int, ...], bool]] = []
    tracker_calls: list[tuple[int, bool]] = []

    def measure_model(
        unit: EfficiencyProfileUnit,
        scan_indices: tuple[int, ...],
        row_type: str,
        horizon: int,
        *,
        warmup: bool,
    ) -> ModelMeasurement:
        model_calls.append((row_type, horizon, scan_indices, warmup))
        return ModelMeasurement(
            latency_ms=float(horizon),
            gpu_peak_memory_bytes=1000 + horizon,
        )

    def measure_tracker(
        unit: EfficiencyProfileUnit,
        horizon: int,
        *,
        warmup: bool,
    ) -> TrackerMeasurement:
        tracker_calls.append((horizon, warmup))
        return TrackerMeasurement(
            latency_ms=2.0,
            association_overhead_ms=0.5,
            memory_update_overhead_ms=0.75,
            persistent_state_bytes=63808,
        )

    records = collect_efficiency_records(
        _units(),
        measure_model=measure_model,
        measure_tracker=measure_tracker,
    )

    assert len(records) == 1161
    assert Counter((row["row_type"], row["T"]) for row in records) == {
        ("bootstrap", 1): 129,
        **{("new_visit", horizon): 129 for horizon in range(2, 6)},
        **{("full_history", horizon): 129 for horizon in range(2, 6)},
    }
    assert Counter(call[0:2] for call in model_calls if call[3]) == {
        ("bootstrap", 1): 1,
        **{("new_visit", horizon): 1 for horizon in range(2, 6)},
        **{("full_history", horizon): 1 for horizon in range(2, 6)},
    }
    assert Counter(horizon for horizon, warmup in tracker_calls if warmup) == {
        horizon: 1 for horizon in range(1, 6)
    }
    assert len(model_calls) == 1161 + 9
    assert len(tracker_calls) == 645 + 5

    first_t4_local = next(
        call for call in model_calls if call[0:2] == ("new_visit", 4) and not call[3]
    )
    first_t4_full = next(
        call for call in model_calls if call[0:2] == ("full_history", 4) and not call[3]
    )
    assert first_t4_local[2] == (2, 3)
    assert first_t4_full[2] == (0, 1, 2, 3)

    first_new_visit = next(row for row in records if row["row_type"] == "new_visit")
    assert first_new_visit["model_latency_ms"] == 2.0
    assert first_new_visit["tracker_latency_ms"] == 2.0
    assert first_new_visit["association_overhead_ms"] == 0.5
    assert first_new_visit["memory_update_overhead_ms"] == 0.75
    first_full = next(row for row in records if row["row_type"] == "full_history")
    assert first_full["tracker_latency_ms"] is None
    assert first_full["persistent_state_bytes"] is None


def test_measure_cuda_operation_synchronizes_around_only_the_operation() -> None:
    calls: list[str] = []
    clock = iter((2_000_000, 7_500_000))

    def operation() -> str:
        calls.append("operation")
        return "output"

    output, measurement = measure_cuda_operation(
        operation,
        device=torch.device("cuda:0"),
        synchronize=lambda _device: calls.append("synchronize"),
        reset_peak=lambda _device: calls.append("reset_peak"),
        peak_memory=lambda _device: 4096,
        clock_ns=lambda: next(clock),
    )

    assert output == "output"
    assert calls == ["synchronize", "reset_peak", "operation", "synchronize"]
    assert measurement == ModelMeasurement(
        latency_ms=5.5,
        gpu_peak_memory_bytes=4096,
    )


def test_measure_b4_tracker_uses_one_causal_final_transition() -> None:
    observations = tuple(_observation(stage) for stage in range(3))

    measurement = measure_b4_tracker(
        observations,
        horizon=3,
        tracker_settings={
            "capacity": 100,
            "class_weight": 0.25,
            "association_threshold": 0.5,
            "update_rate": 0.2,
        },
    )

    assert measurement.latency_ms >= (
        measurement.association_overhead_ms + measurement.memory_update_overhead_ms
    )
    assert measurement.persistent_state_bytes == persistent_state_bytes(100, 2, 2)


def _protocol_and_cached_sequences():
    masters = []
    variants = {}
    sequences = []
    for master_index in range(43):
        master_id = f"master-{master_index:02d}"
        reference_id = f"reference-{master_index % 6}"
        canonical = tuple(master_index * 10 + stage for stage in range(5))
        masters.append(
            SimpleNamespace(
                sequence_id=master_id,
                reference_scene_id=reference_id,
                validation_index=master_index,
                scan_indices=canonical,
            )
        )
        order_indices = {
            "canonical": canonical,
            "reverse": tuple(reversed(canonical)),
            "sha256_seed45": (
                canonical[2],
                canonical[0],
                canonical[4],
                canonical[1],
                canonical[3],
            ),
        }
        variants[master_id] = {}
        for order_id, scan_indices in order_indices.items():
            variants[master_id][order_id] = SimpleNamespace(scan_indices=scan_indices)
            sequences.append(
                SimpleNamespace(
                    reference_scene_id=reference_id,
                    master_sequence_id=master_id,
                    order_id=order_id,
                    payloads=tuple({"stage": stage} for stage in range(5)),
                )
            )
    return SimpleNamespace(masters=tuple(masters), variants=variants), tuple(sequences)


def test_build_efficiency_units_binds_protocol_order_to_cached_observations() -> None:
    protocol, sequences = _protocol_and_cached_sequences()

    units = build_efficiency_profile_units(
        protocol,
        sequences,
        observation_loader=lambda payload: _observation(payload["stage"]),
    )

    assert len(units) == 129
    reverse = next(
        unit
        for unit in units
        if unit.master_sequence_id == "master-00" and unit.order_id == "reverse"
    )
    assert reverse.context_index == 0
    assert reverse.context_scan_indices == (0, 1, 2, 3, 4)
    assert reverse.scan_indices == (4, 3, 2, 1, 0)
    assert len(reverse.observations) == 5


def test_real_model_profiler_keeps_setup_outside_measured_forward() -> None:
    calls: list[object] = []

    class Dataset:
        def __init__(self) -> None:
            self.sequence_names = ["master-00"]
            self.sequence_indices = [(0, 1, 2, 3, 4)]

        def load_scan_indices(self, context_index, scan_indices, *, change_file):
            calls.append(("load", context_index, scan_indices, change_file))
            return "sample"

    data = {"coord": torch.ones((2, 4))}
    target = {"point2segment": torch.tensor([0, 1])}

    def collate(samples):
        calls.append(("collate", tuple(samples)))
        return data, [target], ["master-00"]

    class System:
        def _process_raw_coordinates(self, received):
            calls.append(("raw", received))
            return torch.ones((2, 3))

        def __call__(self, received, **kwargs):
            calls.append(("forward", received, kwargs))
            return {"pred_logits": torch.ones((1, 1, 2))}

    def measured(operation, *, device):
        calls.append(("measure_begin", device))
        output = operation()
        calls.append("measure_end")
        return output, ModelMeasurement(1.25, 4096)

    profiler = RealModelProfiler(
        dataset=Dataset(),
        collate=collate,
        system=System(),
        device=torch.device("cpu"),
        move_data=lambda value, _device: value,
        move_targets=lambda value, _device: value,
        measure_operation=measured,
    )
    unit = EfficiencyProfileUnit(
        reference_scene_id="reference-0",
        master_sequence_id="master-00",
        order_id="reverse",
        context_index=0,
        context_scan_indices=(0, 1, 2, 3, 4),
        scan_indices=(4, 3, 2, 1, 0),
        observations=tuple(_observation(stage) for stage in range(5)),
    )

    result = profiler(
        unit,
        (2, 1),
        "new_visit",
        4,
        warmup=False,
    )

    assert result == ModelMeasurement(1.25, 4096)
    assert calls[0] == ("load", 0, (2, 1), None)
    assert [call[0] if isinstance(call, tuple) else call for call in calls] == [
        "load",
        "collate",
        "raw",
        "measure_begin",
        "forward",
        "measure_end",
    ]
    forward = next(
        call for call in calls if isinstance(call, tuple) and call[0] == "forward"
    )
    assert forward[2]["point2segment"] == [target["point2segment"]]
    assert forward[2]["is_eval"] is True


def test_efficiency_cli_defaults_raw_manifest_inside_external_cache(
    tmp_path: Path,
) -> None:
    args = _argument_parser().parse_args(
        ["--cache-directory", str(tmp_path), "--metadata", "metadata.json"]
    )

    assert args.output is None
    assert args.device == "cuda:0"


def test_efficiency_profiler_cli_help_works_outside_repository(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts/profile_p6a_efficiency.py"),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--cache-directory" in completed.stdout


def test_real_efficiency_profile_rejects_repository_cache_before_setup() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        run_real_efficiency_profile(
            cache_directory=Path("artifacts/P6A/cache"),
            metadata_path=Path("external/3RScan.json"),
            checkpoint_path=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
            output_path=None,
            device_name="cuda:0",
        )


def test_profiler_accepts_only_the_preregistered_efficiency_contract() -> None:
    config = yaml.safe_load(Path("conf/p6a/default.yaml").read_text())

    validate_efficiency_config(config["efficiency"])
    changed = dict(config["efficiency"])
    changed["warmup_per_group"] = 2
    with pytest.raises(ValueError, match="efficiency config"):
        validate_efficiency_config(changed)
