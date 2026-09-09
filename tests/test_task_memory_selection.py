from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.select_task_memory import (
    REQUIRED_M2_UPDATES,
    TaskMemorySelectionError,
    select_task_memory_checkpoint,
    selected_checkpoint_comparison,
    terminal_update_comparison,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for variant_index, variant in enumerate(("Q-INDEP", "Q-TALA")):
        for update in REQUIRED_M2_UPDATES:
            for baseline in ("R1", "FH-MATCH"):
                for horizon in (2, 3, 4, 5):
                    score = update / 100_000 + variant_index / 100
                    rows.append(
                        {
                            "variant": variant,
                            "update": update,
                            "checkpoint_sha256": f"{variant_index + update + 1:064x}",
                            "policy": "lag1",
                            "reducer": "mean",
                            "baseline": baseline,
                            "T": horizon,
                            "delta_t_mAP": score,
                            "median_update_latency_ms": 20.0 + variant_index,
                        }
                    )
    return rows


def test_selection_requires_all_registered_updates_and_all_t_cells() -> None:
    rows = _rows()
    missing_update = [
        row
        for row in rows
        if not (row["variant"] == "Q-INDEP" and row["update"] == 750)
    ]
    with pytest.raises(TaskMemorySelectionError, match="registered updates"):
        select_task_memory_checkpoint(missing_update)

    missing_horizon = [
        row
        for row in rows
        if not (
            row["variant"] == "Q-TALA"
            and row["update"] == 1500
            and row["baseline"] == "R1"
            and row["T"] == 4
        )
    ]
    with pytest.raises(TaskMemorySelectionError, match="all-T baseline cells"):
        select_task_memory_checkpoint(missing_horizon)


def test_selection_is_one_checkpoint_and_never_horizon_specific() -> None:
    selection = select_task_memory_checkpoint(_rows())

    assert selection["variant"] == "Q-TALA"
    assert selection["update"] == 3000
    assert selection["policy"] == "lag1"
    assert selection["reducer"] == "mean"
    assert selection["selection_horizons"] == [2, 3, 4, 5]
    assert selection["selection_baselines"] == ["FH-MATCH", "R1"]

    changed_policy = _rows()
    changed_policy[0]["policy"] = "commit0"
    with pytest.raises(TaskMemorySelectionError, match="single policy and reducer"):
        select_task_memory_checkpoint(changed_policy)


def test_selection_ties_use_mean_then_latency_then_earlier_update() -> None:
    rows = _rows()
    for row in rows:
        row["delta_t_mAP"] = 0.01
        row["median_update_latency_ms"] = 20.0
    selection = select_task_memory_checkpoint(rows)

    assert selection["variant"] == "Q-INDEP"
    assert selection["update"] == 0

    mean_tie_break = deepcopy(rows)
    for row in mean_tie_break:
        if row["variant"] == "Q-TALA" and row["update"] == 750:
            row["delta_t_mAP"] = 0.02 if row["T"] == 5 else 0.01
    assert select_task_memory_checkpoint(mean_tie_break)["update"] == 750

    latency_tie_break = deepcopy(rows)
    for row in latency_tie_break:
        if row["variant"] == "Q-TALA" and row["update"] == 1500:
            row["median_update_latency_ms"] = 19.0
    assert select_task_memory_checkpoint(latency_tie_break)["update"] == 1500


def test_terminal_and_selected_tables_are_kept_separate() -> None:
    rows = _rows()
    selection = select_task_memory_checkpoint(rows)

    terminal = terminal_update_comparison(rows)
    selected = selected_checkpoint_comparison(rows, selection)

    assert {row["update"] for row in terminal} == {3000}
    assert {row["variant"] for row in terminal} == {"Q-INDEP", "Q-TALA"}
    assert {row["update"] for row in selected} == {3000}
    assert {row["variant"] for row in selected} == {"Q-TALA"}
