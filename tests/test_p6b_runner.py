from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.p6b_protocol import (
    build_split_manifest,
    canonical_config_id,
    load_p6b_config,
)
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BClusterMetrics,
    P6BHorizonMetrics,
    candidate_ranking_key,
    run_staged_sweep,
)
from scripts.run_p6b_evaluation import (
    _candidate_sweep_rows,
    _config_sha256,
    _expected_selection_provenance,
    _join_per_sequence_rows,
    _paired_statistics,
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


def _sweep_metric(horizon: int, *, official: bool) -> P6BHorizonMetrics:
    cluster = P6BClusterMetrics(
        reference_scene_id="cluster-0",
        identity_switches=1,
        transition_opportunities=10,
        wrong_reactivations=1,
        predicted_reactivation_events=5,
        correct_reactivations=4,
        reactivation_attempts=5,
        gap_opportunities=8,
        false_births=1,
        births=5,
        rejected_births=1,
    )
    return P6BHorizonMetrics(
        horizon=horizon,
        identity_switches=1,
        transition_opportunities=10,
        wrong_reactivations=1,
        predicted_reactivation_events=5,
        correct_reactivations=4,
        reactivation_attempts=5,
        gap_opportunities=8,
        false_births=1,
        births=5,
        rejected_births=1,
        reactivation_accuracy=0.8,
        reactivation_recall=0.5,
        accepted_valid_observations=90,
        total_valid_observations=100,
        cluster_metrics=(cluster,),
        strict_online_tmap=0.2 if official else None,
        strict_online_trec=0.3 if official else None,
    )


def _sweep_candidate(
    config: P6BMemoryConfig, stage: str, *, official: bool
) -> P6BCandidateRow:
    return P6BCandidateRow(
        config=config,
        stage=stage,
        horizons=tuple(
            _sweep_metric(horizon, official=official) for horizon in (2, 3, 4, 5)
        ),
    )


def _association_row(
    method: str,
    reference: str,
    master: str,
    horizon: int,
    *,
    digest: str,
    switches: int,
) -> dict[str, object]:
    return {
        "method": method,
        "reference_scene_id": reference,
        "master_sequence_id": master,
        "order_id": "canonical",
        "prefix": horizon,
        "prediction_digest": digest,
        "id_switches": switches,
        "transition_opportunities": 10,
        "wrong_reactivations": 0 if horizon == 2 else switches,
        "predicted_reactivation_events": 0 if horizon == 2 else 4,
        "correct_reactivations": 0 if horizon == 2 else 4 - switches,
        "reactivation_attempts": 0 if horizon == 2 else 4,
        "gap_opportunities": 0 if horizon == 2 else 5,
        "false_births": switches,
        "births": 8,
        "rejected_births": 2,
        "reactivation_accuracy": None if horizon == 2 else (4 - switches) / 4,
        "reactivation_recall": None if horizon == 2 else (4 - switches) / 5,
    }


def _task_row(
    method: str,
    reference: str,
    master: str,
    horizon: int,
    *,
    digest: str,
) -> dict[str, object]:
    return {
        "method": method,
        "reference_scene_id": reference,
        "master_sequence_id": master,
        "order_id": "canonical",
        "T": f"T{horizon}",
        "t_mAP": 0.20 if method == "B4" else 0.21,
        "t_REC": 0.30 if method == "B4" else 0.31,
        "prediction_digest": digest,
    }


def _paired_sequence_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    association: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    for reference in ("heldout-a", "heldout-b"):
        for sequence_index in range(2):
            master = f"{reference}-master-{sequence_index}"
            digest = f"digest-{reference}-{sequence_index}"
            for horizon in (2, 3, 4, 5):
                for method, switches in (("B4", 2), ("P6B", 1)):
                    association.append(
                        _association_row(
                            method,
                            reference,
                            master,
                            horizon,
                            digest=digest,
                            switches=switches,
                        )
                    )
                    tasks.append(
                        _task_row(
                            method,
                            reference,
                            master,
                            horizon,
                            digest=digest,
                        )
                    )
    return association, tasks


def test_per_sequence_join_is_exact_and_recomputes_normalized_rates() -> None:
    association, tasks = _paired_sequence_fixture()

    rows = _join_per_sequence_rows(
        association,
        tasks,
        expected_sequence_count=4,
        expected_reference_scene_ids=("heldout-a", "heldout-b"),
    )

    assert len(rows) == 32
    row = next(item for item in rows if item["method"] == "P6B" and item["T"] == "T5")
    assert row["identity_switch_rate"] == pytest.approx(0.1)
    assert row["wrong_reactivation_rate"] == pytest.approx(0.25)
    assert row["false_birth_rate"] == pytest.approx(0.1)
    assert row["reactivation_accuracy"] == pytest.approx(0.75)
    assert row["reactivation_recall"] == pytest.approx(0.6)
    assert row["t_mAP"] == pytest.approx(0.21)
    assert row["prediction_digest"].startswith("digest-")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cross_digest"])
def test_per_sequence_join_rejects_inexact_evidence(mutation: str) -> None:
    association, tasks = _paired_sequence_fixture()
    if mutation == "missing":
        tasks.pop()
    elif mutation == "duplicate":
        tasks.append(dict(tasks[-1]))
    else:
        tasks[-1]["prediction_digest"] = "wrong-digest"

    with pytest.raises(ValueError, match="missing|duplicate|digest|population"):
        _join_per_sequence_rows(
            association,
            tasks,
            expected_sequence_count=4,
            expected_reference_scene_ids=("heldout-a", "heldout-b"),
        )


def test_paired_statistics_are_clustered_seeded_and_complete() -> None:
    association, tasks = _paired_sequence_fixture()
    rows = _join_per_sequence_rows(
        association,
        tasks,
        expected_sequence_count=4,
        expected_reference_scene_ids=("heldout-a", "heldout-b"),
    )

    first = _paired_statistics(rows, expected_sequence_count=4)
    second = _paired_statistics(rows, expected_sequence_count=4)

    assert first == second
    assert len(first) == 25
    idsw_t5 = next(
        item
        for item in first
        if item["metric"] == "identity_switch_rate" and item["T"] == "T5"
    )
    assert idsw_t5["n_clusters"] == 2
    assert idsw_t5["n_pairs"] == 4
    assert idsw_t5["n_bootstrap"] == 10_000
    assert idsw_t5["seed"] == 45
    assert idsw_t5["mean_delta"] == pytest.approx(-0.1)
    assert idsw_t5["std_delta"] == pytest.approx(0.0)
    assert idsw_t5["excluded_null_pairs"] == 0


def _valid_selection_document() -> dict[str, object]:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _sweep_candidate(protocol.base, "baseline", official=True)
    result = run_staged_sweep(
        protocol,
        baseline=baseline,
        fast_evaluator=lambda config, stage: _sweep_candidate(
            config, stage, official=False
        ),
        official_evaluator=lambda row: replace(
            row,
            horizons=tuple(
                replace(metric, strict_online_tmap=0.2, strict_online_trec=0.3)
                for metric in row.horizons
            ),
        ),
    )
    manifest = json.loads(
        Path("artifacts/P6A/protocol_b_manifest.json").read_text(encoding="utf-8")
    )
    split = build_split_manifest(manifest, seed=protocol.seed).to_mapping()
    return build_selection_document(
        source_commit="a" * 40,
        split_manifest=split,
        selected_config=result.selected.config,
        ranking_key=candidate_ranking_key(result.selected),
        baseline={"rows": _candidate_sweep_rows(baseline)},
        candidate_rows=tuple(
            row
            for candidate in result.candidate_rows
            for row in _candidate_sweep_rows(candidate)
        ),
        finalist_rows=tuple(
            row
            for candidate in result.finalist_rows
            for row in _candidate_sweep_rows(candidate)
        ),
        selected_by_stage={
            stage: candidate.config_id
            for stage, candidate in result.selected_by_stage.items()
        },
        provenance=_expected_selection_provenance(protocol, split),
    )


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
    document = _valid_selection_document()
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    loaded = load_selection_document(path)

    assert loaded == document
    assert loaded["selected_config"] == document["selected_config"]
    assert loaded["heldout_evaluated"] is False
    assert "heldout_results" not in loaded
    changed = dict(document)
    changed["selected_config_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="config.*SHA|SHA.*config"):
        load_selection_document(path)

    changed = deepcopy(document)
    changed_config = P6BMemoryConfig(
        **{
            **changed["selected_config"],
            "birth_confidence": 0.123,
        }
    )
    changed["selected_config"] = asdict(changed_config)
    changed["selected_config_id"] = canonical_config_id(changed_config)
    changed["selected_config_sha256"] = _config_sha256(changed_config)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="winner|ranking|selected"):
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
