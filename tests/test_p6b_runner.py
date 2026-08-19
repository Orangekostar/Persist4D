from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.run_p6b_evaluation import (
    build_selection_document,
    build_source_tree_contract,
    compute_final_gate_results,
    load_selection_document,
    partition_cached_sequences,
)


class _Sequence:
    def __init__(self, reference: str, master: str, order: str = "canonical") -> None:
        self.reference_scene_id = reference
        self.master_sequence_id = master
        self.order_id = order


def _split() -> dict[str, object]:
    return {
        "tuning_reference_scene_ids": ["r0", "r1", "r2", "r3"],
        "heldout_reference_scene_ids": ["r4", "r5"],
        "tuning_master_sequence_ids": [f"m{i}" for i in range(32)],
        "heldout_master_sequence_ids": [f"h{i}" for i in range(11)],
    }


def _metric(method: str, horizon: int, *, switches: int = 10) -> dict[str, object]:
    return {
        "method": method,
        "T": f"T{horizon}",
        "t_mAP": 0.20,
        "t_REC": 0.30,
        "identity_switches": switches,
        "identity_switch_rate": switches / 100,
        "reactivation_accuracy": None if horizon == 2 else 0.80,
        "reactivation_recall": None if horizon == 2 else 0.40,
        "false_births": 3,
    }


def test_partition_keeps_tuning_and_heldout_clusters_disjoint() -> None:
    sequences = tuple(
        _Sequence(reference, master)
        for reference, master in zip(
            ["r0"] * 32 + ["r4"] * 11,
            [f"m{i}" for i in range(32)] + [f"h{i}" for i in range(11)],
            strict=True,
        )
    )

    tuning, heldout = partition_cached_sequences(sequences, _split())

    assert len(tuning) == 32 and len(heldout) == 11
    assert {item.reference_scene_id for item in tuning} == {"r0"}
    assert {item.reference_scene_id for item in heldout} == {"r4"}
    with pytest.raises(ValueError, match="registered split"):
        partition_cached_sequences((*sequences, _Sequence("unknown", "x")), _split())


def test_selection_document_binds_config_split_source_and_no_holdout_metrics(
    tmp_path: Path,
) -> None:
    document = build_selection_document(
        source_commit="a" * 40,
        split_manifest={**_split(), "sha256": "b" * 64},
        selected_config=P6BMemoryConfig(),
        ranking_key=(1.0, 2.0, 3.0, -0.4, -0.2, "{}"),
        baseline={"method": "B4"},
        candidate_rows=({"stage": "assignment"},),
        finalist_rows=({"stage": "assignment"},),
        selected_by_stage={"assignment": "p6b-id"},
        provenance={"cache_manifest_sha256": "c" * 64},
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_selection_document(path)

    assert loaded == document
    assert loaded["selected_config"] == asdict(P6BMemoryConfig())
    assert loaded["heldout_evaluated"] is False
    assert "heldout_results" not in loaded
    changed = dict(document)
    changed["selected_config_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="config.*SHA|SHA.*config"):
        load_selection_document(path)


def test_final_gates_compute_go_and_fail_closed_on_any_horizon_regression() -> None:
    baseline = [_metric("B4", horizon) for horizon in (2, 3, 4, 5)]
    candidate = [
        _metric("P6B", horizon, switches=(8 if horizon in (4, 5) else 10))
        for horizon in (2, 3, 4, 5)
    ]

    gates = compute_final_gate_results(
        baseline + candidate,
        evidence_complete=True,
        frozen_hashes_unchanged=True,
    )

    assert all(record["passed"] for record in gates.values())
    worse = [dict(row) for row in baseline + candidate]
    next(row for row in worse if row["method"] == "P6B" and row["T"] == "T5")[
        "identity_switch_rate"
    ] = 0.11
    stopped = compute_final_gate_results(
        worse, evidence_complete=True, frozen_hashes_unchanged=True
    )
    assert stopped["G6B-2"]["passed"] is False


def test_source_contract_rejects_tracked_and_untracked_dirty_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    assert build_source_tree_contract(tmp_path)["status"] == "pass"
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        build_source_tree_contract(tmp_path)
    subprocess.run(["git", "restore", "tracked.txt"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        build_source_tree_contract(tmp_path)
