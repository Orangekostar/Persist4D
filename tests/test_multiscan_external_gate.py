from __future__ import annotations

import pytest

from scripts.audit_multiscan_dataset import derive_multiscan_preflight_decision


def _decision(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "stable_identity_verified": True,
        "gap_events": 10,
        "gap_scenes": 3,
        "chronology_status": "DATASET_ORDER_ONLY",
        "ordered_revisit_protocol_allowed": True,
        "alignment_verified": True,
        "gt_leakage_impossible": True,
        "observation_coverage": 0.10,
    }
    values.update(overrides)
    return derive_multiscan_preflight_decision(**values)


def test_external_gate_passes_at_all_inclusive_thresholds() -> None:
    assert _decision()["decision"] == "MULTISCAN_FULL_EVAL_GO"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"gap_events": 9}, "MULTISCAN_GAP_FAIL"),
        ({"gap_scenes": 2}, "MULTISCAN_GAP_FAIL"),
        ({"alignment_verified": False}, "MULTISCAN_ALIGNMENT_FAIL"),
        ({"observation_coverage": 0.0999}, "MULTISCAN_COVERAGE_FAIL"),
        ({"stable_identity_verified": False}, "MULTISCAN_PROTOCOL_FAIL"),
        ({"gap_events": None}, "MULTISCAN_PROTOCOL_FAIL"),
        ({"chronology_status": "UNRESOLVED"}, "MULTISCAN_PROTOCOL_FAIL"),
        ({"gt_leakage_impossible": False}, "MULTISCAN_PROTOCOL_FAIL"),
    ],
)
def test_external_gate_fails_closed(
    overrides: dict[str, object], expected: str
) -> None:
    assert _decision(**overrides)["decision"] == expected
