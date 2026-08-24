from __future__ import annotations

import pytest

from scripts.build_rescan_per_scene_effects import (
    OUTPUT_FIELDS,
    build_per_scene_effect_rows,
)

METRICS = (
    "level_a_online_t_mAP",
    "level_a_online_t_REC",
    "level_b_observation_coverage",
    "level_b_gap_recovery_accuracy",
    "level_b_gap_recovery_recall",
    "level_b_normalized_id_switch_rate",
    "level_b_fragmentation_count",
    "level_b_merge_count",
)


def _row(scene_id: str, method_code: str, **values: str) -> dict[str, str]:
    row = {"scene_id": scene_id, "method_code": method_code}
    row.update({metric: values.get(metric, "0") for metric in METRICS})
    return row


def test_builds_high_and_low_priority_effects_with_relative_values() -> None:
    rows = build_per_scene_effect_rows(
        [
            _row(
                "scene_b",
                "B4",
                level_a_online_t_mAP="0.9",
                level_b_fragmentation_count="1",
            ),
            _row(
                "scene_b",
                "B2",
                level_a_online_t_mAP="0.5",
                level_b_fragmentation_count="3",
            ),
        ]
    )

    assert rows[0] == {
        "scene_id": "scene_b",
        "level": "level_a",
        "metric": "online_t_mAP",
        "higher_is_better": True,
        "comparator": "B2",
        "comparator_value": 0.5,
        "method": "B4",
        "method_value": 0.9,
        "effect": 0.4,
        "absolute_effect": 0.4,
        "relative_effect": 0.8,
    }
    fragmentation = rows[6]
    assert fragmentation["effect"] == -2.0
    assert fragmentation["absolute_effect"] == 2.0
    assert fragmentation["relative_effect"] == -2.0 / 3.0


def test_zero_comparator_has_empty_relative_effect() -> None:
    rows = build_per_scene_effect_rows(
        [
            _row("scene_a", "B2", level_b_gap_recovery_accuracy="0"),
            _row("scene_a", "B4", level_b_gap_recovery_accuracy="0.25"),
        ]
    )

    assert rows[3]["effect"] == 0.25
    assert rows[3]["relative_effect"] == ""


@pytest.mark.parametrize(
    ("case", "rows"),
    [
        (
            "missing B2",
            [_row("scene_a", "B4")],
        ),
        (
            "missing B4",
            [_row("scene_a", "B2")],
        ),
        (
            "missing B2",
            [_row("scene_a", "B1")],
        ),
        (
            "duplicate B2",
            [_row("scene_a", "B2"), _row("scene_a", "B2"), _row("scene_a", "B4")],
        ),
        (
            "duplicate B4",
            [_row("scene_a", "B2"), _row("scene_a", "B4"), _row("scene_a", "B4")],
        ),
        (
            "nonfinite",
            [_row("scene_a", "B2", level_a_online_t_mAP="nan"), _row("scene_a", "B4")],
        ),
    ],
)
def test_invalid_scene_rows_fail_closed(case: str, rows: list[dict[str, str]]) -> None:
    with pytest.raises(ValueError, match=case.split()[0]):
        build_per_scene_effect_rows(rows)


def test_output_order_is_scene_then_fixed_metric_order_and_fields_are_fixed() -> None:
    rows = build_per_scene_effect_rows(
        [
            _row("scene_z", "B2"),
            _row("scene_a", "B4"),
            _row("scene_z", "B4"),
            _row("scene_a", "B2"),
        ]
    )

    assert list(rows[0]) == list(OUTPUT_FIELDS)
    assert [row["scene_id"] for row in rows] == ["scene_a"] * 8 + ["scene_z"] * 8
    assert [row["metric"] for row in rows[:8]] == [
        "online_t_mAP",
        "online_t_REC",
        "observation_coverage",
        "gap_recovery_accuracy",
        "gap_recovery_recall",
        "normalized_id_switch_rate",
        "fragmentation_count",
        "merge_count",
    ]
