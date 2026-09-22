import pytest

from scripts.perception_gain_v2_profile import (
    ProfileError,
    profile_inventory,
    summarize_measurements,
)


def test_profile_deduplicates_an_unchanged_final_and_preserves_native_parent():
    methods = {
        name: {"method_id": name, "inference_identity": name}
        for name in ("r1", "fh", "p-fh")
    }
    lock = {
        "methods": methods,
        "aliases": {
            "R1-D0": "r1",
            "FINAL": "r1",
            "FH-R1-native": "fh",
            "FH-P-native": "p-fh",
        },
    }
    assert [row["method_id"] for row in profile_inventory(lock)] == ["fh", "r1", "p-fh"]


def test_profile_checks_all_components_and_fixed_population_before_summary():
    rows = []
    for reference in range(6):
        for repeat in range(3):
            for horizon in (2, 3, 4, 5):
                rows.append(
                    {
                        "method": "final",
                        "reference_scene_id": str(reference),
                        "master_sequence_id": str(reference),
                        "T": horizon,
                        "repeat": repeat,
                        "order_id": "canonical",
                        "input_prepare_ms": 1.0,
                        "network_forward_ms": 4.0,
                        "scorer_ms": 1.0,
                        "refiner_ms": 2.0,
                        "state_update_ms": 3.0,
                        "materialize_ms": 2.0,
                        "end_to_end_ms": 13.0,
                        "prefix_cumulative_ms": 13.0 * horizon,
                        "peak_allocated_bytes": 100,
                        "peak_reserved_bytes": 200,
                        "cpu_rss_bytes": 50,
                        "resident_state_bytes": 10,
                        "soft_buffer_bytes": 20,
                        "archive_bytes": 30,
                        "window_voxel_points": 4,
                        "window_segments": 2,
                    }
                )
    summary = summarize_measurements(rows, methods=["final"])
    assert len(summary) == 24
    assert all(row["median_refiner_ms"] == 2.0 for row in summary)
    with pytest.raises(ProfileError, match="coverage"):
        summarize_measurements(rows[:-1], methods=["final"])
    rows[0]["end_to_end_ms"] = 7.0
    with pytest.raises(ProfileError, match="component"):
        summarize_measurements(rows, methods=["final"])
