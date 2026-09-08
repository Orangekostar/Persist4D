from __future__ import annotations

import pytest

from scripts.publish_persist4d_allt import (
    DEFAULT_ARTIFACT_ROOT,
    TASK_METRICS,
    PublicationError,
    build_learning_curve_rows,
    build_memory_ablation_rows,
    build_variant_matrix,
)


def _evaluation_rows(
    model: str, checkpoint: str, base: float
) -> list[dict[str, object]]:
    return [
        {
            "population_id": "development_train_holdout_47_masters_canonical",
            "model": model,
            "checkpoint_sha256": checkpoint,
            "training_seed": 45,
            "evaluation_seed": 45,
            "method": "FullHistory" if model == "FH-adapt" else "B4",
            "reducer": "official" if model == "FH-adapt" else "mean",
            "T": horizon,
            **{
                metric: base + horizon / 100 + metric_index / 1000
                for metric_index, metric in enumerate(TASK_METRICS)
            },
            "local_current_AP": base,
            "num_master": 47,
            "num_order_units": 47,
            "num_reference_clusters": 8,
        }
        for horizon in range(2, 6)
    ]


def test_variant_matrix_records_complete_and_gate_skipped_states() -> None:
    matrix = build_variant_matrix(DEFAULT_ARTIFACT_ROOT)

    by_variant = {row["variant"]: row for row in matrix["variants"]}
    assert set(by_variant) == {"C0", "C1", "C2", "C3", "FH-adapt", "FH-L"}
    assert all(
        by_variant[variant]["execution_status"] == "COMPLETE"
        for variant in ("C0", "C1", "C2", "FH-adapt")
    )
    assert by_variant["C3"]["execution_status"] == "GATE_SKIPPED"
    assert by_variant["FH-L"]["execution_status"] == "GATE_SKIPPED"
    assert matrix["seed46_training_confirmation"] == "GATE_SKIPPED"
    assert matrix["optimizer_updates"] == 400
    assert matrix["effective_episode_batch"] == 8


def test_learning_curve_requires_five_updates_and_marks_frozen_selection() -> None:
    checkpoints = {update: f"{index:x}" * 64 for index, update in enumerate(range(0, 500, 100), 1)}
    evaluations = {
        update: _evaluation_rows("C2", checkpoint, update / 1000)
        for update, checkpoint in checkpoints.items()
    }

    curve = build_learning_curve_rows(
        "C2",
        evaluations,
        selected_step=200,
        selected_checkpoint=checkpoints[200],
    )

    assert len(curve) == 5
    assert [row["checkpoint_global_step"] for row in curve] == [0, 100, 200, 300, 400]
    assert [row["selected_update"] for row in curve] == [False, False, True, False, False]
    assert curve[2]["t_mAP_T2"] == pytest.approx(0.22)
    with pytest.raises(PublicationError, match="updates"):
        build_learning_curve_rows(
            "C2",
            {key: value for key, value in evaluations.items() if key != 400},
            selected_step=200,
            selected_checkpoint=checkpoints[200],
        )


def test_memory_ablation_has_three_diagnostic_policies_at_every_horizon() -> None:
    checkpoint = "a" * 64
    policies = {
        "all_occupied": _evaluation_rows("C2", checkpoint, 0.2),
        "disabled": _evaluation_rows("C2-memory-off", checkpoint, 0.1),
        "active_previous": _evaluation_rows("C2-previous-only", checkpoint, 0.15),
    }

    rows = build_memory_ablation_rows(policies, checkpoint_sha256=checkpoint)

    assert len(rows) == 12
    assert {(row["memory_read_policy"], row["T"]) for row in rows} == {
        (policy, horizon) for policy in policies for horizon in range(2, 6)
    }
    assert all(row["evidence_scope"] == "development_diagnostic" for row in rows)
    assert all(row["causal_claim"] == "not_established" for row in rows)
    disabled_t2 = next(
        row
        for row in rows
        if row["memory_read_policy"] == "disabled" and row["T"] == 2
    )
    assert disabled_t2["delta_t_mAP_vs_all_occupied"] == pytest.approx(-0.1)
