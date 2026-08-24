from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.audit_tmap_protocol_shift import (
    ProtocolShiftError,
    build_metric_rows,
    build_population_rows,
    summarize_population,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/tmap_root_cause_v2"


def _prefix(sequence_id: str) -> dict[str, object]:
    return {
        "sequence_id": sequence_id,
        "scan_ids": sequence_id.split("-"),
    }


def _protocol() -> dict[str, object]:
    return {
        "masters": [
            {
                "master_sequence_id": "a-b-c-d-e",
                "reference_scene_id": "reference-a",
                "orders": {"canonical": {"prefixes": {"2": _prefix("a-b")}}},
            },
            {
                "master_sequence_id": "f-g-h-i-j",
                "reference_scene_id": "reference-b",
                "orders": {"canonical": {"prefixes": {"2": _prefix("f-g")}}},
            },
        ]
    }


def _metric_row(order: str, value: float) -> dict[str, str]:
    return {
        "method": "FullHistory",
        "order_id": order,
        "horizon": "2",
        "sequence_count": "43",
        "causal_prefix_t_mAP": str(value),
        "causal_prefix_t_mAP50": str(value + 0.1),
        "causal_prefix_t_mAP25": str(value + 0.2),
        "current_stage_AP": str(value + 0.3),
    }


def test_population_audit_requires_exact_ordered_pair_membership() -> None:
    rows = build_population_rows(
        _protocol(),
        {
            "a-b": {
                "type": "validation",
                "scene": 1,
                "sub_scenes": [0, 1],
                "filepath": "change_gt/validation/a-b.txt",
            },
            "g-f": {
                "type": "validation",
                "scene": 2,
                "sub_scenes": [1, 0],
                "filepath": "change_gt/validation/g-f.txt",
            },
        },
        expected_master_count=2,
    )

    assert [row["sequence_id"] for row in rows] == ["a-b", "f-g"]
    assert rows[0]["exact_ordered_pair_present"] is True
    assert rows[0]["official_like_split"] == "validation"
    assert rows[1]["exact_ordered_pair_present"] is False
    summary = summarize_population(rows)
    assert summary == {
        "requested_pair_count": 2,
        "exact_pair_count": 1,
        "missing_pair_count": 1,
        "full_exact_subset_identifiable": False,
    }


def test_metric_audit_extracts_preregistered_r0_to_r5_rows() -> None:
    per_order = [
        _metric_row("canonical", 0.20722658932209015),
        _metric_row("reverse", 0.21109874546527863),
        _metric_row("sha256_seed45", 0.1711808741092682),
    ]
    aggregate = [_metric_row("all", 0.19099636375904083)]

    rows = build_metric_rows(per_order, aggregate)

    assert [row["record_id"] for row in rows] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    assert rows[0]["t_mAP"] == pytest.approx(0.348)
    assert rows[1]["t_mAP"] == pytest.approx(0.27939)
    assert rows[2]["t_mAP"] == pytest.approx(0.20722658932209015)
    assert rows[5]["t_mAP"] == pytest.approx(0.19099636375904083)
    assert rows[0]["direct_comparison_group"] == "official_like_t2"
    assert rows[1]["direct_comparison_group"] == "official_like_t2"
    assert rows[2]["direct_comparison_group"] == "protocol_b_t2"
    assert rows[5]["order_scope"] == "three_order_pooled"


def test_metric_audit_fails_when_one_preregistered_order_is_missing() -> None:
    with pytest.raises(ProtocolShiftError, match="order coverage"):
        build_metric_rows(
            [
                _metric_row("canonical", 0.2),
                _metric_row("reverse", 0.2),
            ],
            [_metric_row("all", 0.2)],
        )


def test_checked_in_protocol_shift_artifacts_are_exact_and_explicit() -> None:
    with (ARTIFACT_ROOT / "protocol_shift_population.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        population = list(csv.DictReader(handle))
    with (ARTIFACT_ROOT / "protocol_shift_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        metrics = list(csv.DictReader(handle))
    report = (ARTIFACT_ROOT / "PROTOCOL_SHIFT_AUDIT.md").read_text(
        encoding="utf-8"
    )

    assert len(population) == 43
    assert sum(row["exact_ordered_pair_present"] == "true" for row in population) == 14
    assert sum(row["exact_ordered_pair_present"] == "false" for row in population) == 29
    assert [row["record_id"] for row in metrics] == [
        "R0",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
    ]
    assert "NOT IDENTIFIABLE FROM CURRENT ARTIFACTS" in report
    assert "34.8 and 19.10 are not directly comparable" in report
    assert "/home/" not in report
    assert "repo:data/processed/rio/sequence_database_sliding_2.yaml" in report
