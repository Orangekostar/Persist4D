from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.profile_persist4d_allt import (
    ProfileError,
    select_profile_sequences,
    summarize_profile_samples,
    validate_profile_samples,
)


def _sequence(reference: int, master: int, order: str = "canonical") -> object:
    return SimpleNamespace(
        stage_requests=(
            {
                "reference_scene_id": f"reference-{reference}",
                "master_sequence_id": f"master-{master:02d}",
                "order_id": order,
            },
        )
    )


def test_profile_selection_uses_first_canonical_master_per_reference() -> None:
    sequences = []
    for reference in reversed(range(6)):
        sequences.extend(
            [
                _sequence(reference, reference + 10),
                _sequence(reference, reference),
                _sequence(reference, reference, "reverse"),
            ]
        )

    selected = select_profile_sequences(sequences)

    assert len(selected) == 6
    assert [item.stage_requests[0]["reference_scene_id"] for item in selected] == [
        f"reference-{index}" for index in range(6)
    ]
    assert [item.stage_requests[0]["master_sequence_id"] for item in selected] == [
        f"master-{index:02d}" for index in range(6)
    ]
    assert all(
        item.stage_requests[0]["order_id"] == "canonical" for item in selected
    )


def _samples() -> list[dict[str, object]]:
    rows = []
    for model_index, model in enumerate(("C2", "FH-adapt")):
        for reference in range(6):
            for horizon in range(2, 6):
                for repeat in range(10):
                    rows.append(
                        {
                            "model": model,
                            "reference_scene_id": f"reference-{reference}",
                            "master_sequence_id": f"master-{reference}",
                            "order_id": "canonical",
                            "T": horizon,
                            "repeat": repeat,
                            "latency_ms": float(repeat + model_index + 1),
                            "start_allocated_bytes": 100,
                            "peak_allocated_bytes": 200 + repeat,
                            "incremental_peak_allocated_bytes": 100 + repeat,
                            "window_voxel_points": 1000 + horizon,
                            "window_segments": 100 + horizon,
                        }
                    )
    return rows


def test_profile_samples_require_exact_repeat_coverage() -> None:
    rows = _samples()
    validate_profile_samples(rows)
    with pytest.raises(ProfileError, match="coverage"):
        validate_profile_samples(rows[:-1])


def test_profile_samples_require_identical_t2_inputs_between_methods() -> None:
    rows = _samples()
    changed = next(
        row
        for row in rows
        if row["model"] == "FH-adapt"
        and row["reference_scene_id"] == "reference-0"
        and row["T"] == 2
    )
    changed["window_segments"] = int(changed["window_segments"]) + 1

    with pytest.raises(ProfileError, match="T2 input"):
        validate_profile_samples(rows)


def test_profile_summary_is_deterministic_and_uses_measured_peaks() -> None:
    rows = _samples()
    first = summarize_profile_samples(rows)
    second = summarize_profile_samples(rows)

    assert first == second
    assert len(first) == 48
    c2 = first[0]
    assert c2["model"] == "C2"
    assert c2["median_latency_ms"] == pytest.approx(5.5)
    assert c2["minimum_latency_ms"] == pytest.approx(1.0)
    assert c2["maximum_latency_ms"] == pytest.approx(10.0)
    assert c2["peak_allocated_bytes"] == 209
    assert c2["maximum_incremental_peak_allocated_bytes"] == 109
    assert c2["measured_repeats"] == 10
