from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from scripts.publish_persist4d_allt import (
    COMPLETION_STATUS_KEYS,
    DEFAULT_ARTIFACT_ROOT,
    TASK_METRICS,
    PublicationError,
    _external_table,
    build_final_manifest,
    build_learning_curve_rows,
    build_memory_ablation_rows,
    build_variant_matrix,
    derive_final_statuses,
    primary_tmap_failure_cells,
    publish_final_package,
    validate_completion_inputs,
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


def test_completion_statuses_preserve_the_failed_scientific_verdict() -> None:
    statuses = derive_final_statuses(
        DEFAULT_ARTIFACT_ROOT, publication_status="NOT_ATTEMPTED"
    )

    assert tuple(statuses) == COMPLETION_STATUS_KEYS
    assert statuses == {
        "execution_status": "COMPLETE",
        "baseline_comparability": "MATCHED",
        "tmap_all_t_vs_R1": "FAIL",
        "tmap_all_t_vs_matched_rescene": "FAIL",
        "task_metrics_all_t": "FAIL",
        "seed_confirmation": "NOT_RUN",
        "independent_generalization": "NOT_ESTABLISHED",
        "publication_status": "NOT_ATTEMPTED",
    }
    assert primary_tmap_failure_cells(DEFAULT_ARTIFACT_ROOT) == {
        "C2_vs_R1_B4": [4, 5],
        "C2_vs_FH-adapt": [3, 4, 5],
    }


def test_final_manifest_hashes_reports_without_self_reference(tmp_path) -> None:
    report = tmp_path / "FINAL_REPORT.md"
    handoff = tmp_path / "HANDOFF.md"
    table = tmp_path / "results/all_t_metrics.csv"
    report.write_bytes(b"report\n")
    handoff.write_bytes(b"handoff\n")
    table.parent.mkdir()
    table.write_bytes(b"metric\n")
    statuses = {
        key: "COMPLETE" if key == "execution_status" else "NOT_ATTEMPTED"
        for key in COMPLETION_STATUS_KEYS
    }

    manifest = build_final_manifest(
        tmp_path,
        artifact_paths=(report, handoff, table),
        statuses=statuses,
        code_commit_at_run="a" * 40,
    )

    by_path = {row["path"]: row for row in manifest["artifacts"]}
    assert set(by_path) == {
        "FINAL_REPORT.md",
        "HANDOFF.md",
        "results/all_t_metrics.csv",
    }
    assert by_path["HANDOFF.md"]["sha256"] == hashlib.sha256(b"handoff\n").hexdigest()
    assert "FINAL_MANIFEST.json" not in by_path
    assert manifest["publication_commit"] == "resolve from remote HEAD"


def test_completion_validation_rejects_missing_required_artifacts(tmp_path) -> None:
    with pytest.raises(PublicationError, match="required completion artifact"):
        validate_completion_inputs(tmp_path)


def test_external_table_binds_selected_update_name_and_hash() -> None:
    table = _external_table(DEFAULT_ARTIFACT_ROOT)

    assert "training/formal/C0/update=0400.ckpt" in table
    assert "training/formal/C1/update=0100.ckpt" in table
    assert "training/formal/C2/update=0200.ckpt" in table
    assert "training/formal/FH-adapt/update=0100.ckpt" in table


def test_completion_package_is_deterministic_and_self_contained(tmp_path) -> None:
    artifact_root = tmp_path / "allt_task_superiority_v1"
    shutil.copytree(DEFAULT_ARTIFACT_ROOT, artifact_root)

    first = publish_final_package(
        artifact_root, code_commit_at_run="b" * 40
    )
    first_hashes = {row["path"]: row["sha256"] for row in first}
    second = publish_final_package(
        artifact_root, code_commit_at_run="b" * 40
    )

    assert first_hashes == {row["path"]: row["sha256"] for row in second}
    report = (artifact_root / "FINAL_REPORT.md").read_text(encoding="ascii")
    handoff = (artifact_root / "HANDOFF.md").read_text(encoding="ascii")
    manifest = json.loads(
        (artifact_root / "FINAL_MANIFEST.json").read_text(encoding="ascii")
    )
    assert manifest["statuses"] == derive_final_statuses(
        artifact_root, publication_status="NOT_ATTEMPTED"
    )
    assert "T4, T5" in report
    assert "T3, T4, T5" in report
    assert "Attempt coverage" in report
    assert "only 5/20" in handoff
    assert all(f"## {number}." in handoff for number in range(1, 19))
    assert "training/formal/C2/update=0200.ckpt" in handoff
    assert "--variant C2" in handoff
    assert "192.168.100.102:/mnt/data/shared" in handoff
    listed = {row["path"]: row for row in manifest["artifacts"]}
    assert "FINAL_REPORT.md" in listed
    assert "HANDOFF.md" in listed
    assert "TEST_REPORT.md" in listed
    assert "FINAL_MANIFEST.json" not in listed
    assert listed["HANDOFF.md"]["sha256"] == hashlib.sha256(
        (artifact_root / "HANDOFF.md").read_bytes()
    ).hexdigest()
