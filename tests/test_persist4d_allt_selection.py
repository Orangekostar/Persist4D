from __future__ import annotations

from scripts.select_persist4d_allt import (
    build_selection_decision,
    evaluate_complementarity,
)


def _selection(*, t2: float, t3: float, t4: float, t5: float):
    return {
        "checkpoint_sha256": "a" * 64,
        "t_map_delta_by_horizon": {
            "2": t2,
            "3": t3,
            "4": t4,
            "5": t5,
        },
    }


def test_complementarity_requires_strict_short_and_long_horizon_gains() -> None:
    result = evaluate_complementarity(
        c1_selection=_selection(t2=0.01, t3=0.02, t4=-0.01, t5=-0.02),
        c2_selection=_selection(t2=-0.01, t3=-0.02, t4=0.03, t5=0.04),
    )

    assert result["status"] == "authorized_pending_training"
    assert result["c1_short_horizon_positive"] is True
    assert result["c2_long_horizon_positive"] is True


def test_complementarity_fails_closed_on_zero_or_negative_gain() -> None:
    result = evaluate_complementarity(
        c1_selection=_selection(t2=0.01, t3=0.0, t4=0.01, t5=0.01),
        c2_selection=_selection(t2=0.01, t3=0.01, t4=0.02, t5=0.03),
    )

    assert result["status"] == "gate_skipped"
    assert result["c1_short_horizon_positive"] is False
    assert result["c2_long_horizon_positive"] is True


def _rows(checkpoint: str, values: tuple[float, float, float, float]):
    return [
        {"checkpoint_sha256": checkpoint, "T": horizon, "t_mAP": value}
        for horizon, value in zip(range(2, 6), values, strict=True)
    ]


def test_selection_freezes_one_checkpoint_and_skips_unjustified_variants() -> None:
    checkpoints = {
        variant: {0: f"{index:064x}", 100: f"{index + 10:064x}"}
        for index, variant in enumerate(("C0", "C1", "C2", "FH-adapt"), start=1)
    }
    rows_by_variant = {
        "C0": [
            *_rows(checkpoints["C0"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C0"][100], (0.21, 0.21, 0.21, 0.21)),
        ],
        "C1": [
            *_rows(checkpoints["C1"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C1"][100], (0.25, 0.20, 0.21, 0.21)),
        ],
        "C2": [
            *_rows(checkpoints["C2"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C2"][100], (0.23, 0.23, 0.23, 0.23)),
        ],
        "FH-adapt": [
            *_rows(checkpoints["FH-adapt"][0], (0.21, 0.21, 0.21, 0.21)),
            *_rows(checkpoints["FH-adapt"][100], (0.22, 0.22, 0.22, 0.22)),
        ],
    }

    decision = build_selection_decision(
        rows_by_variant=rows_by_variant,
        checkpoint_steps=checkpoints,
    )

    assert decision["status"] == "frozen"
    assert decision["selected_candidate"]["variant"] == "C2"
    assert decision["selected_candidate"]["checkpoint_global_step"] == 100
    assert decision["selected_matched_rescene"]["variant"] == "FH-adapt"
    assert decision["conditional_variants"] == {
        "C3": "gate_skipped",
        "FH-L": "gate_skipped",
    }
    assert decision["protocol_b_used_for_selection"] is False


def test_selection_stays_pending_when_complementarity_authorizes_c3() -> None:
    checkpoints = {
        variant: {0: f"{index:064x}", 100: f"{index + 10:064x}"}
        for index, variant in enumerate(("C0", "C1", "C2", "FH-adapt"), start=1)
    }
    rows_by_variant = {
        "C0": [
            *_rows(checkpoints["C0"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C0"][100], (0.21, 0.21, 0.21, 0.21)),
        ],
        "C1": [
            *_rows(checkpoints["C1"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C1"][100], (0.23, 0.23, 0.20, 0.20)),
        ],
        "C2": [
            *_rows(checkpoints["C2"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C2"][100], (0.20, 0.20, 0.24, 0.24)),
        ],
        "FH-adapt": [
            *_rows(checkpoints["FH-adapt"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["FH-adapt"][100], (0.21, 0.21, 0.21, 0.21)),
        ],
    }

    decision = build_selection_decision(
        rows_by_variant=rows_by_variant,
        checkpoint_steps=checkpoints,
    )

    assert decision["status"] == "conditional_training_pending"
    assert decision["selected_candidate"] is None
    assert decision["conditional_variants"]["C3"] == "authorized_pending_training"


def test_update_zero_is_a_baseline_not_a_trainable_checkpoint_candidate() -> None:
    checkpoints = {
        variant: {0: f"{index:064x}", 100: f"{index + 10:064x}"}
        for index, variant in enumerate(("C0", "C1", "C2", "FH-adapt"), start=1)
    }
    rows_by_variant = {
        "C0": [
            *_rows(checkpoints["C0"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C0"][100], (0.30, 0.30, 0.30, 0.19)),
        ],
        "C1": [
            *_rows(checkpoints["C1"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C1"][100], (0.31, 0.31, 0.21, 0.21)),
        ],
        "C2": [
            *_rows(checkpoints["C2"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["C2"][100], (0.21, 0.21, 0.31, 0.31)),
        ],
        "FH-adapt": [
            *_rows(checkpoints["FH-adapt"][0], (0.20, 0.20, 0.20, 0.20)),
            *_rows(checkpoints["FH-adapt"][100], (0.21, 0.21, 0.21, 0.21)),
        ],
    }

    decision = build_selection_decision(
        rows_by_variant=rows_by_variant,
        checkpoint_steps=checkpoints,
    )

    assert len(decision["variant_rankings"]["C0"]) == 1
    assert decision["variant_rankings"]["C0"][0]["checkpoint_global_step"] == 100
