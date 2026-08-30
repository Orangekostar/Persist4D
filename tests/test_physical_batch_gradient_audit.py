from __future__ import annotations

import pytest

from utils.rescene_runtime_audit import (
    IMPORTANT_GRADIENT_GROUPS,
    build_fixed_batch_panel,
    physical_batch_gradient_gate,
    summarize_physical_batch_runs,
)


def _reference(group: str, cosine: float, relative: float) -> dict[str, object]:
    return {
        "parameter_group": group,
        "cosine": cosine,
        "relative_norm_difference": relative,
    }


def test_fixed_panel_requires_exactly_32_positioned_references() -> None:
    references = [
        {"dataset": "rio" if index % 2 == 0 else "scannet", "sample_index": index}
        for index in range(32)
    ]

    panel = build_fixed_batch_panel(references, seed=45)

    assert len(panel["samples"]) == 32
    assert [row["position"] for row in panel["samples"]] == list(range(32))
    assert [row["augmentation_seed"] for row in panel["samples"]] == [
        45 + index for index in range(32)
    ]
    with pytest.raises(ValueError, match="32"):
        build_fixed_batch_panel(references[:31], seed=45)


def test_physical_batch_gate_requires_two_important_group_differences() -> None:
    rows = [
        _reference(group, 0.99, 0.05) for group in IMPORTANT_GRADIENT_GROUPS
    ]
    rows[0] = _reference(IMPORTANT_GRADIENT_GROUPS[0], 0.97, 0.05)
    rows[1] = _reference(IMPORTANT_GRADIENT_GROUPS[1], 0.99, 0.11)

    gate = physical_batch_gradient_gate(rows, feasible=True)

    assert gate["authorized"] is True
    assert gate["triggered_group_count"] == 2
    assert gate["thresholds"] == {
        "minimum_cosine": 0.98,
        "maximum_relative_norm_difference": 0.1,
        "minimum_triggered_groups": 2,
    }
    assert physical_batch_gradient_gate(rows, feasible=False)["authorized"] is False


def test_physical_batch_summary_keeps_oom_as_infeasible() -> None:
    runs = [
        {
            "physical_global_batch": 4,
            "accumulation": 8,
            "feasible": True,
            "peak_memory_mib": 1000.0,
            "step_seconds": 2.0,
        },
        {
            "physical_global_batch": 8,
            "accumulation": 4,
            "feasible": False,
            "failure": "CUDAOutOfMemoryError",
        },
    ]

    summary = summarize_physical_batch_runs(runs)

    assert summary["reference_physical_global_batch"] == 4
    assert summary["runs"][1]["feasible"] is False
    assert "peak_memory_mib" not in summary["runs"][1]
    with pytest.raises(ValueError, match="OOM"):
        summarize_physical_batch_runs(
            [runs[0], {**runs[1], "peak_memory_mib": 2000.0}]
        )
