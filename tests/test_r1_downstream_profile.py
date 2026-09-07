from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch


def _profile():
    return importlib.import_module("scripts.profile_r1_downstream_validation")


def test_profile_coverage_requires_six_by_two_by_three_and_ten_samples() -> None:
    profile = _profile()
    samples = []
    summaries = []
    for reference_index in range(6):
        reference = f"cluster-{reference_index}"
        master = f"master-{reference_index}"
        for method in ("FullHistory", "B4"):
            for horizon in (2, 4, 5):
                base = {
                    "method": method,
                    "reference_scene_id": reference,
                    "master_sequence_id": master,
                    "order_id": "canonical",
                    "horizon": horizon,
                    "warmup_repeats": 5,
                    "measured_repeats": 10,
                }
                summaries.append(
                    {
                        **base,
                        "occupied_slots": 17 if method == "B4" else None,
                        "rejected_births": 2 if method == "B4" else None,
                    }
                )
                for sample_index in range(10):
                    samples.append({**base, "sample_index": sample_index})

    assert profile.validate_profile_coverage(
        sample_rows=samples, summary_rows=summaries
    ) == {
        "profile_unit_count": 6,
        "summary_row_count": 36,
        "sample_row_count": 360,
    }
    samples.pop()
    with pytest.raises(profile.R1ProfileError, match="sample coverage"):
        profile.validate_profile_coverage(
            sample_rows=samples, summary_rows=summaries
        )


def test_profile_coverage_requires_b4_state_fields() -> None:
    profile = _profile()
    summaries = []
    samples = []
    for reference_index in range(6):
        for method in ("FullHistory", "B4"):
            for horizon in (2, 4, 5):
                base = {
                    "method": method,
                    "reference_scene_id": f"cluster-{reference_index}",
                    "master_sequence_id": f"master-{reference_index}",
                    "order_id": "canonical",
                    "horizon": horizon,
                    "warmup_repeats": 5,
                    "measured_repeats": 10,
                }
                summaries.append(
                    {
                        **base,
                        "occupied_slots": None,
                        "rejected_births": None,
                    }
                )
                samples.extend(
                    {**base, "sample_index": sample_index}
                    for sample_index in range(10)
                )

    with pytest.raises(profile.R1ProfileError, match="state evidence"):
        profile.validate_profile_coverage(
            sample_rows=samples, summary_rows=summaries
        )


def test_b4_state_summary_uses_final_occupancy_and_cumulative_rejections() -> None:
    profile = _profile()
    steps = [
        SimpleNamespace(
            state_snapshot=SimpleNamespace(
                occupied=torch.tensor([[True, False, False]])
            ),
            rejected_births=(False, True),
        ),
        SimpleNamespace(
            state_snapshot=SimpleNamespace(
                occupied=torch.tensor([[True, True, False]])
            ),
            rejected_births=(True, True),
        ),
    ]

    assert profile.summarize_b4_state(steps) == {
        "occupied_slots": 2,
        "rejected_births": 3,
    }


def test_profile_coverage_rejects_noncanonical_or_wrong_repeat_contract() -> None:
    profile = _profile()
    row = {
        "method": "B4",
        "reference_scene_id": "cluster-0",
        "master_sequence_id": "master-0",
        "order_id": "reverse",
        "horizon": 4,
        "warmup_repeats": 5,
        "measured_repeats": 10,
    }
    with pytest.raises(profile.R1ProfileError, match="labels|contract"):
        profile.validate_profile_coverage(
            sample_rows=[{**row, "sample_index": index} for index in range(10)],
            summary_rows=[row],
        )
