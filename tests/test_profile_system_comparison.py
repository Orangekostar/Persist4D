from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
import torch

from scripts.profile_system_comparison import (
    ProfilingError,
    build_persistent_operation_factory,
    build_profile_subset,
    measure_cuda_repeats,
    persistent_state_storage_bytes,
)
from scripts.system_comparison_protocol import build_system_comparison_manifest


def _system_manifest():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "artifacts/P6A/protocol_b_manifest.json"
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    return build_system_comparison_manifest(
        protocol_path,
        incumbent_binding={
            "status": "pass",
            "p6a_protocol_manifest_sha256": digest,
        },
    )


def test_profile_subset_uses_one_canonical_master_per_reference_scene() -> None:
    units = build_profile_subset(_system_manifest())

    assert len(units) == 6
    assert len({unit.reference_scene_id for unit in units}) == 6
    assert all(unit.order_id == "canonical" for unit in units)
    assert [unit.reference_scene_id for unit in units] == sorted(
        unit.reference_scene_id for unit in units
    )
    for unit in units:
        assert len(unit.visit_order) == 5
        assert len(unit.scan_indices) == 5
        assert len(unit.context_scan_indices) == 5


def test_cuda_timing_synchronizes_resets_and_excludes_warmups() -> None:
    events: list[str] = []
    clock_values = iter([0, 1_000_000, 2_000_000, 4_000_000, 5_000_000, 8_000_000])
    allocated_values = iter([100, 110, 120])
    reserved_values = iter([200, 220, 210])

    def operation_factory():
        events.append("factory")

        def operation():
            events.append("operation")

        return operation

    profile = measure_cuda_repeats(
        operation_factory,
        device=torch.device("cuda:0"),
        warmup_repeats=2,
        measured_repeats=3,
        enforce_protocol=False,
        synchronize=lambda device: events.append(f"sync:{device.index}"),
        reset_peak=lambda device: events.append(f"reset:{device.index}"),
        peak_allocated=lambda device: next(allocated_values),
        peak_reserved=lambda device: next(reserved_values),
        clock_ns=lambda: next(clock_values),
    )

    assert profile.samples_ms == (1.0, 2.0, 3.0)
    assert profile.median_ms == pytest.approx(2.0)
    assert profile.mean_ms == pytest.approx(2.0)
    assert profile.std_ms == pytest.approx((2 / 3) ** 0.5)
    assert profile.peak_allocated_bytes == 120
    assert profile.peak_reserved_bytes == 220
    assert events.count("factory") == 5
    assert events.count("operation") == 5
    assert events.count("reset:0") == 3
    assert events.count("sync:0") == 10
    assert events[:4] == ["factory", "sync:0", "operation", "sync:0"]


def test_cuda_timing_rejects_wrong_repeat_contract() -> None:
    with pytest.raises(ProfilingError, match="warmup|measured"):
        measure_cuda_repeats(
            lambda: lambda: None,
            device=torch.device("cuda:0"),
            warmup_repeats=4,
            measured_repeats=10,
        )


def test_persistent_operation_factory_resets_and_prerolls_each_repeat() -> None:
    events: list[tuple[str, int]] = []
    tracker_count = 0

    class Tracker:
        def __init__(self, tracker_id: int) -> None:
            self.tracker_id = tracker_id

        def step(self, observation, *, stage_id):
            events.append((f"tracker:{self.tracker_id}:{observation}", stage_id))
            return observation

    def tracker_factory():
        nonlocal tracker_count
        tracker_count += 1
        return Tracker(tracker_count)

    factory = build_persistent_operation_factory(
        model_forward=lambda: "current",
        tracker_factory=tracker_factory,
        prior_observations=("s1", "s2"),
        current_stage=2,
    )
    first = factory()
    second = factory()
    first()
    second()

    assert tracker_count == 2
    assert events == [
        ("tracker:1:s1", 0),
        ("tracker:1:s2", 1),
        ("tracker:2:s1", 0),
        ("tracker:2:s2", 1),
        ("tracker:1:current", 2),
        ("tracker:2:current", 2),
    ]


@dataclass
class _State:
    a: torch.Tensor
    b: torch.Tensor

    def tensors(self):
        return (self.a, self.b)


def test_persistent_state_bytes_are_exact_tensor_storage() -> None:
    state = _State(
        a=torch.zeros((3, 4), dtype=torch.float32),
        b=torch.zeros((5,), dtype=torch.int64),
    )
    assert persistent_state_storage_bytes(state) == 3 * 4 * 4 + 5 * 8

    with pytest.raises(ProfilingError, match="tensors"):
        persistent_state_storage_bytes(object())
