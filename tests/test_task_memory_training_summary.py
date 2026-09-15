from __future__ import annotations

import pytest

from scripts.summarize_task_memory_training import (
    TrainingSummaryError,
    checkpoint_identity,
    curve_snapshot,
)


def test_checkpoint_identity_distinguishes_selected_and_last() -> None:
    summary = {
        "variant": "Q-TALA",
        "status": "COMPLETE",
        "completed_global_step": 3000,
        "checkpoints": [
            {"name": "last.ckpt", "sha256": "a" * 64, "bytes": 20},
            {"name": "update=1500.ckpt", "sha256": "b" * 64, "bytes": 20},
            {"name": "update=3000.ckpt", "sha256": "a" * 64, "bytes": 20},
        ],
    }

    identity = checkpoint_identity(summary, selected_update=1500)

    assert identity["selected"]["sha256"] == "b" * 64
    assert identity["last"]["sha256"] == "a" * 64
    assert identity["last"]["optimizer_state_retained"] is True
    with pytest.raises(TrainingSummaryError, match="selected checkpoint"):
        checkpoint_identity(summary, selected_update=750)


def test_curve_snapshot_requires_one_four_horizon_checkpoint() -> None:
    rows = [
        {
            "phase": "M2",
            "variant": "Q-TALA",
            "update": 1500,
            "T": horizon,
            "t_mAP": 0.1 * horizon,
            "checkpoint_sha256": "b" * 64,
            "policy": "lag1",
            "reducer": "mean",
            "status": "PASS",
        }
        for horizon in (2, 3, 4, 5)
    ]

    snapshot = curve_snapshot(rows, variant="Q-TALA", update=1500)

    assert snapshot["mean_t_mAP"] == pytest.approx(0.35)
    assert snapshot["T5_t_mAP"] == pytest.approx(0.5)
    broken = [dict(row) for row in rows[:-1]]
    with pytest.raises(TrainingSummaryError, match="horizon coverage"):
        curve_snapshot(broken, variant="Q-TALA", update=1500)
