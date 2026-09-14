from __future__ import annotations

import pytest
import torch

from models.object_visual_memory import ObjectVisualState
from models.task_memory_state import TaskMemoryState
from scripts.profile_task_memory import (
    ProfileError,
    _clone_state,
    _clone_visual_state,
    _validate_profile_device_name,
    build_profile_units,
    derive_resource_status,
    load_permanent_state_bytes,
    summarize_measurements,
    validate_profile_contract,
)


def _contract() -> dict[str, object]:
    return {
        "schema_version": "task-memory-profile-v2",
        "device_count": 1,
        "device_type": "A40",
        "reference_units": 6,
        "warmups": 5,
        "repeats": 10,
        "restore_cloned_state_each_repeat": True,
        "time_scopes": [
            "model_update",
            "end_to_end_update",
            "output_materialization",
            "cumulative_T1_to_T",
        ],
    }


def _protocol() -> dict[str, object]:
    masters = []
    reference_counts = (8, 7, 7, 7, 7, 7)
    for reference, count in enumerate(reference_counts):
        for master in range(count):
            scans = [reference * 20 + master * 5 + index for index in range(5)]
            masters.append(
                {
                    "reference_scene_id": f"reference-{reference}",
                    "master_sequence_id": f"master-{reference}-{master}",
                    "validation_index": reference * 2 + master,
                    "scan_indices": scans,
                    "orders": {
                        "canonical": {
                            "visit_order": [f"scan-{value}" for value in scans],
                            "scan_indices": scans,
                        }
                    },
                }
            )
    return {
        "protocol": {"order_variants": ["canonical", "reverse", "sha256_seed45"]},
        "masters": masters,
    }


def test_profile_contract_and_units_are_frozen_to_six_canonical_references() -> None:
    contract = validate_profile_contract(_contract())
    units = build_profile_units(_protocol())

    assert contract["warmups"] == 5
    assert contract["repeats"] == 10
    assert len(units) == 6
    assert len({unit.reference_scene_id for unit in units}) == 6
    assert all(unit.order_id == "canonical" for unit in units)
    assert [unit.master_sequence_id for unit in units] == [
        f"master-{reference}-0" for reference in range(6)
    ]

    broken = _contract()
    broken["warmups"] = 4
    with pytest.raises(ProfileError, match="5 warmups"):
        validate_profile_contract(broken)


def test_profile_summary_keeps_scopes_memory_and_ten_cloned_repeats() -> None:
    rows = []
    for method, base in (("M3-V-CORE", 10.0), ("FH-CONT", 20.0)):
        for reference in range(6):
            for horizon in (2, 3, 4, 5):
                for scope in (
                    "model_update",
                    "end_to_end_update",
                    "output_materialization",
                    "cumulative_T1_to_T",
                ):
                    for repeat in range(10):
                        rows.append(
                            {
                                "method": method,
                                "reference_id": f"reference-{reference}",
                                "master_sequence_id": f"master-{reference}",
                                "T": horizon,
                                "scope": scope,
                                "repeat": repeat,
                                "state_snapshot_sha256": f"{reference:064x}",
                                "latency_ms": base + repeat,
                                "loaded_model_allocated_bytes": 100,
                                "peak_allocated_bytes": 180
                                if method == "M3-V-CORE"
                                else 240,
                                "incremental_peak_bytes": 80
                                if method == "M3-V-CORE"
                                else 140,
                                "peak_reserved_bytes": 300,
                                "cpu_state_bytes": 60,
                                "visual_state_bytes": 400,
                                "lag1_buffer_bytes": 500,
                                "archive_bytes": 600,
                            }
                        )

    summary = summarize_measurements(rows)

    assert len(summary) == 2 * 6 * 4 * 4
    assert all(row["measured_repeats"] == 10 for row in summary)
    assert all(row["median_latency_ms"] in (14.5, 24.5) for row in summary)
    assert all(row["absolute_peak_allocated_bytes"] in (180, 240) for row in summary)


def test_resource_advantage_requires_t4_t5_latency_peak_and_state_budget() -> None:
    summary = []
    for method, latency, peak in (
        ("M3-V-CORE", 10.0, 100),
        ("FH-CONT", 20.0, 200),
    ):
        for reference in range(6):
            for horizon in (4, 5):
                summary.append(
                    {
                        "method": method,
                        "reference_id": f"reference-{reference}",
                        "T": horizon,
                        "scope": "model_update",
                        "median_latency_ms": latency,
                        "absolute_peak_allocated_bytes": peak,
                    }
                )

    assert (
        derive_resource_status(
            summary,
            candidate="M3-V-CORE",
            baseline="FH-CONT",
            permanent_state_bytes=489_816,
            state_budget_bytes=2 * 1024 * 1024,
        )
        == "ADVANTAGE"
    )
    changed = 0
    for row in summary:
        if row["method"] == "M3-V-CORE" and row["T"] == 4 and changed < 4:
            row["median_latency_ms"] = 30.0
            changed += 1
    assert (
        derive_resource_status(
            summary,
            candidate="M3-V-CORE",
            baseline="FH-CONT",
            permanent_state_bytes=489_816,
            state_budget_bytes=2 * 1024 * 1024,
        )
        == "TRADEOFF"
    )


def test_permanent_state_bytes_comes_from_the_measured_combined_total(tmp_path) -> None:
    state_bytes = tmp_path / "state_bytes.csv"
    state_bytes.write_text(
        "component,tensor,bytes,status\n"
        "task_state,TOTAL,62616,PASS\n"
        "visual_state,TOTAL,427200,PASS\n"
        "combined_state,TOTAL,489816,PASS\n",
        encoding="ascii",
    )

    assert load_permanent_state_bytes(state_bytes) == 489_816

    state_bytes.write_text(
        "component,tensor,bytes,status\ncombined_state,TOTAL,489816,FAIL\n",
        encoding="ascii",
    )
    with pytest.raises(ProfileError, match="combined_state"):
        load_permanent_state_bytes(state_bytes)


def test_profile_snapshots_are_detached_clones_on_requested_device() -> None:
    task = TaskMemoryState.empty(
        batch_size=1,
        capacity=100,
        feature_dim=128,
        class_count=19,
        device="cpu",
        dtype=torch.float32,
    )
    visual = ObjectVisualState.empty(
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )

    cloned_task = _clone_state(task, device=torch.device("cpu"))
    cloned_visual = _clone_visual_state(visual, device=torch.device("cpu"))

    assert cloned_task is not None and cloned_task.embedding.device.type == "cpu"
    assert cloned_visual is not None and cloned_visual.features.device.type == "cpu"
    assert cloned_task.embedding.data_ptr() != task.embedding.data_ptr()
    assert cloned_visual.features.data_ptr() != visual.features.data_ptr()


def test_profile_device_is_frozen_to_a40() -> None:
    assert _validate_profile_device_name("NVIDIA A40") == "NVIDIA A40"
    with pytest.raises(ProfileError, match="A40"):
        _validate_profile_device_name("NVIDIA A100-SXM4-80GB")
