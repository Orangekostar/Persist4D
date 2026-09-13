from __future__ import annotations


def _curve_rows():
    values = {
        ("M3-BASE-CONT", 750): (0.50, 0.40, 0.30, 0.20),
        ("M3-V-LAST", 750): (0.49, 0.41, 0.31, 0.21),
        ("M3-V-CORE", 750): (0.49, 0.42, 0.32, 0.22),
        ("M3-BASE-CONT", 1500): (0.48, 0.38, 0.28, 0.18),
        ("M3-V-LAST", 1500): (0.47, 0.37, 0.27, 0.17),
        ("M3-V-CORE", 1500): (0.49, 0.39, 0.29, 0.19),
    }
    return [
        {
            "variant": variant,
            "update": update,
            "T": horizon,
            "t_mAP": tmap,
            "checkpoint_sha256": str(update // 750) * 64,
        }
        for (variant, update), tmaps in values.items()
        for horizon, tmap in zip(range(2, 6), tmaps, strict=True)
    ]


def test_m3_comparison_selects_positive_worst_case_before_higher_mean() -> None:
    from scripts.summarize_task_memory_m3 import build_m3_comparison_rows

    rows = build_m3_comparison_rows(_curve_rows())
    selected = [row for row in rows if row["selected_by_prespecified_rule"]]

    assert len(selected) == 1
    assert selected[0]["variant"] == "M3-V-CORE"
    assert selected[0]["update"] == 1500
    assert selected[0]["mean_t_mAP"] == 0.34
    assert abs(selected[0]["s_min_over_controls"] - 0.01) < 1e-12
    core_750 = next(
        row
        for row in rows
        if row["variant"] == "M3-V-CORE" and row["update"] == 750
    )
    assert abs(core_750["s_min_over_controls"] - (-0.01)) < 1e-12


def test_repeated_mean_ablation_uses_native_checkpoint_matched_deltas() -> None:
    from scripts.summarize_task_memory_m3 import build_content_ablation_rows

    native = {2: 0.50, 3: 0.40, 4: 0.30, 5: 0.20}
    repeated = {2: 0.49, 3: 0.38, 4: 0.27, 5: 0.16}
    rows = build_content_ablation_rows(
        native=native,
        repeated_mean=repeated,
        checkpoint_sha256="a" * 64,
        update=1500,
    )

    assert [row["T"] for row in rows] == [2, 3, 4, 5]
    assert rows[0]["delta_vs_native_t_mAP"] == -0.01
    assert abs(rows[-1]["delta_vs_native_t_mAP"] - (-0.04)) < 1e-12
    assert all(abs(row["native_four_t_mean"] - 0.35) < 1e-12 for row in rows)
    assert all(abs(row["repeated_mean_four_t_mean"] - 0.325) < 1e-12 for row in rows)


def test_memory_rows_preserve_measured_state_budget() -> None:
    from scripts.summarize_task_memory_m3 import build_memory_rows

    rows = build_memory_rows(
        {
            "budget_bytes": 2_097_152,
            "combined_state_bytes": 489_816,
            "representatives_per_entity": 8,
            "task_state_bytes": 62_616,
            "visual_state_bytes": 427_200,
        }
    )

    assert rows == [
        {
            "variant": "M3-BASE-CONT",
            "representatives_per_entity": 0,
            "task_state_bytes": 62_616,
            "visual_state_bytes": 0,
            "combined_state_bytes": 62_616,
            "budget_bytes": 2_097_152,
            "within_budget": True,
        },
        {
            "variant": "M3-V-LAST",
            "representatives_per_entity": 8,
            "task_state_bytes": 62_616,
            "visual_state_bytes": 427_200,
            "combined_state_bytes": 489_816,
            "budget_bytes": 2_097_152,
            "within_budget": True,
        },
        {
            "variant": "M3-V-CORE",
            "representatives_per_entity": 8,
            "task_state_bytes": 62_616,
            "visual_state_bytes": 427_200,
            "combined_state_bytes": 489_816,
            "budget_bytes": 2_097_152,
            "within_budget": True,
        },
    ]
