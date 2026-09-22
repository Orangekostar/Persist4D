from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.perception_gain_profile import (
    ProfileError,
    build_profile_methods,
    select_profile_specs,
    summarize_profile_rows,
    validate_profile_rows,
)


def _lock(*, variant: str = "C0", update: int = 0, refiner: bool = False):
    return {
        "schema_version": "perception-gain-final-lock-v1",
        "status": "LOCKED",
        "recipe": {
            "parent_variant": variant,
            "parent_optimizer_update": update,
            "parent_checkpoint": (
                "external:r1_checkpoint"
                if update == 0
                else f"external:training/{variant}/update={update:04d}.ckpt"
            ),
            "parent_checkpoint_sha256": "a" * 64,
            "refiner_enabled": refiner,
            "refiner_optimizer_update": 1000 if refiner else None,
            "refiner_checkpoint": (
                "external:training/refiner/model/update=1000.ckpt" if refiner else None
            ),
            "refiner_checkpoint_sha256": "b" * 64 if refiner else None,
        },
    }


def test_profile_methods_cover_r1_final_and_distinct_parent_native() -> None:
    baseline = build_profile_methods(_lock())
    assert [item.method_id for item in baseline] == [
        "FH-R1-native",
        "D0-R1",
        "FINAL",
    ]
    assert baseline[-1].variant == "C0"
    assert baseline[-1].optimizer_update == 0
    assert baseline[-1].refiner_enabled is False

    selected = build_profile_methods(_lock(variant="S-BAL", update=750, refiner=True))
    assert [item.method_id for item in selected] == [
        "FH-R1-native",
        "D0-R1",
        "FINAL",
        "FH-P*-native",
    ]
    assert selected[2].refiner_enabled is True
    assert selected[3].mode == "native"


def test_profile_selection_uses_first_canonical_master_per_reference() -> None:
    specs = []
    for reference in reversed(range(6)):
        for master, draw_index in ((reference + 10, 2), (reference, 0), (reference, 1)):
            specs.append(
                SimpleNamespace(
                    reference_id=f"reference-{reference}",
                    source_sequence_id=f"master-{master:02d}",
                    draw_index=draw_index,
                )
            )

    selected = select_profile_specs(specs)

    assert [item.reference_id for item in selected] == [
        f"reference-{index}" for index in range(6)
    ]
    assert [item.source_sequence_id for item in selected] == [
        f"master-{index:02d}" for index in range(6)
    ]
    assert all(item.draw_index == 0 for item in selected)


def _rows() -> list[dict[str, object]]:
    rows = []
    methods = ("FH-R1-native", "D0-R1", "FINAL")
    for method_index, method in enumerate(methods):
        for reference in range(6):
            cumulative = 0.0
            for horizon in range(1, 6):
                cumulative += 10.0 + horizon
                if horizon == 1:
                    continue
                for repeat in range(3):
                    rows.append(
                        {
                            "method": method,
                            "reference_scene_id": f"reference-{reference}",
                            "master_sequence_id": f"master-{reference}",
                            "order_id": "canonical",
                            "T": horizon,
                            "repeat": repeat,
                            "network_forward_ms": 5.0 + method_index + repeat,
                            "state_update_ms": 1.0,
                            "materialize_ms": 2.0,
                            "end_to_end_ms": 10.0 + horizon + repeat,
                            "prefix_cumulative_ms": cumulative + repeat,
                            "peak_allocated_bytes": 100 + repeat,
                            "peak_reserved_bytes": 200 + repeat,
                            "cpu_rss_bytes": 300 + repeat,
                            "resident_state_bytes": (
                                10 if method != "FH-R1-native" else 0
                            ),
                            "soft_buffer_bytes": 20 if method == "FINAL" else 0,
                            "archive_bytes": horizon * 30,
                            "window_voxel_points": 1000 + horizon,
                            "window_segments": 100 + horizon,
                        }
                    )
    return rows


def test_profile_rows_require_exact_rectangular_coverage_and_resources() -> None:
    rows = _rows()
    validate_profile_rows(rows, methods=("FH-R1-native", "D0-R1", "FINAL"))
    with pytest.raises(ProfileError, match="coverage"):
        validate_profile_rows(rows[:-1], methods=("FH-R1-native", "D0-R1", "FINAL"))

    invalid = [dict(row) for row in rows]
    invalid[0]["peak_reserved_bytes"] = 0
    with pytest.raises(ProfileError, match="reserved"):
        validate_profile_rows(invalid, methods=("FH-R1-native", "D0-R1", "FINAL"))


def test_profile_summary_reports_median_and_range_without_p99() -> None:
    summary = summarize_profile_rows(
        _rows(), methods=("FH-R1-native", "D0-R1", "FINAL")
    )
    assert len(summary) == 3 * 6 * 4
    row = summary[0]
    assert row["measured_repeats"] == 3
    assert row["median_end_to_end_ms"] == pytest.approx(13.0)
    assert row["minimum_end_to_end_ms"] == pytest.approx(12.0)
    assert row["maximum_end_to_end_ms"] == pytest.approx(14.0)
    assert not any("p99" in key for key in row)
