from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_p6b_evaluation as p6b_runner
from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.p6b_protocol import (
    build_split_manifest,
    canonical_config_id,
    load_p6b_config,
)
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BClusterMetrics,
    P6BHorizonMetrics,
    P6BSequenceAssociationMetrics,
    candidate_ranking_key,
    run_staged_sweep,
)
from scripts.run_p6b_evaluation import (
    _argument_parser,
    _candidate_sweep_rows,
    _config_sha256,
    _expected_selection_provenance,
    _join_per_sequence_rows,
    _load_successful_heldout_attempt,
    _paired_statistics,
    _verification_ledger_from_outputs,
    build_selection_document,
    build_source_tree_contract,
    compute_final_gate_results,
    load_selection_document,
    partition_cached_sequences,
    recover_heldout_attempt,
    run_exactly_once_heldout,
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


@lru_cache(maxsize=1)
def _tuning_reference_scene_ids() -> tuple[str, ...]:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    manifest = json.loads(
        Path("artifacts/P6A/protocol_b_manifest.json").read_text(encoding="utf-8")
    )
    split = build_split_manifest(manifest, seed=protocol.seed)
    return tuple(sorted(split.tuning_reference_scene_ids))


@lru_cache(maxsize=1)
def _tuning_population_by_reference() -> dict[str, dict[str, object]]:
    _protocol, split = p6b_runner._selection_protocol()
    evidence = p6b_runner._split_population_evidence(split, partition="tuning")
    return {item["reference_scene_id"]: item for item in evidence["by_reference"]}


@lru_cache(maxsize=1)
def _tuning_identities_by_reference() -> dict[str, tuple[tuple[str, str, str], ...]]:
    _protocol, split = p6b_runner._selection_protocol()
    return {
        reference: tuple(
            sorted(
                (
                    assignment["reference_scene_id"],
                    assignment["master_sequence_id"],
                    order_id,
                )
                for assignment in split["assignments"]
                if assignment["partition"] == "tuning"
                and assignment["reference_scene_id"] == reference
                for order_id in assignment["order_ids"]
            )
        )
        for reference in split["tuning_reference_scene_ids"]
    }


def _sweep_metric(horizon: int, *, official: bool) -> P6BHorizonMetrics:
    population = _tuning_population_by_reference()
    identities = _tuning_identities_by_reference()
    clusters = tuple(
        P6BClusterMetrics(
            reference_scene_id=reference_scene_id,
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
            sequence_population_count=population[reference_scene_id]["count"],
            sequence_population_sha256=population[reference_scene_id]["sha256"],
            sequence_metrics=tuple(
                P6BSequenceAssociationMetrics(
                    reference_scene_id=identity[0],
                    master_sequence_id=identity[1],
                    order_id=identity[2],
                    identity_switches=1 if index == 0 else 0,
                    transition_opportunities=10 if index == 0 else 0,
                    wrong_reactivations=1 if index == 0 else 0,
                    predicted_reactivation_events=5 if index == 0 else 0,
                    correct_reactivations=4 if index == 0 else 0,
                    reactivation_attempts=5 if index == 0 else 0,
                    gap_opportunities=8 if index == 0 else 0,
                    false_births=1 if index == 0 else 0,
                    births=5 if index == 0 else 0,
                    rejected_births=1 if index == 0 else 0,
                )
                for index, identity in enumerate(identities[reference_scene_id])
            ),
            strict_online_tmap=0.2 if official else None,
            strict_online_trec=0.3 if official else None,
        )
        for reference_scene_id in _tuning_reference_scene_ids()
    )
    cluster_count = len(clusters)
    return P6BHorizonMetrics(
        horizon=horizon,
        identity_switches=cluster_count,
        transition_opportunities=10 * cluster_count,
        wrong_reactivations=cluster_count,
        predicted_reactivation_events=5 * cluster_count,
        correct_reactivations=4 * cluster_count,
        reactivation_attempts=5 * cluster_count,
        gap_opportunities=8 * cluster_count,
        false_births=cluster_count,
        births=5 * cluster_count,
        rejected_births=cluster_count,
        reactivation_accuracy=0.8,
        reactivation_recall=0.5,
        accepted_valid_observations=90,
        total_valid_observations=100,
        cluster_metrics=clusters,
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


def _paired_sequence_fixture() -> tuple[
    list[dict[str, object]], list[dict[str, object]]
]:
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
    assert idsw_t5["ci_low"] == pytest.approx(-0.1)
    assert idsw_t5["ci_high"] == pytest.approx(-0.1)
    assert idsw_t5["cluster_deltas"] == [
        {"reference_scene_id": "heldout-a", "delta": pytest.approx(-0.1)},
        {"reference_scene_id": "heldout-b", "delta": pytest.approx(-0.1)},
    ]
    assert idsw_t5["excluded_null_pairs"] == 0


def _attempt_inputs() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "selection": {
            "ref": "repo:artifacts/P6B_selection/selection.json",
            "sha256": "b" * 64,
        },
        "split_sha256": "c" * 64,
        "p6b_config_sha256": "d" * 64,
        "command": ["final-evaluate", "--protocol", "v2"],
    }


def test_exactly_once_attempt_is_durable_before_evaluation_and_immutable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "P6B_heldout"
    calls = 0

    def evaluate() -> dict[str, object]:
        nonlocal calls
        calls += 1
        assert root.is_dir()
        assert (
            json.loads((root / "attempt.json").read_text())["status"] == "in_progress"
        )
        assert not (root / "heldout_raw.json").exists()
        return {"status": "pass", "rows": [{"value": 1}]}

    raw = run_exactly_once_heldout(root, evaluator=evaluate, **_attempt_inputs())

    assert calls == 1
    assert raw["evaluation"] == {"status": "pass", "rows": [{"value": 1}]}
    attempt = json.loads((root / "attempt.json").read_text())
    assert attempt["status"] == "success"
    assert attempt["exit_status"] == 0
    expected_input_sha = hashlib.sha256(
        (json.dumps(_attempt_inputs(), sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()
    assert attempt["input_sha256"] == expected_input_sha
    assert (
        attempt["log_sha256"]
        == hashlib.sha256(
            (json.dumps(attempt["events"], sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest()
    )
    assert attempt["output"]["ref"] == "repo:artifacts/P6B_heldout/heldout_raw.json"
    assert len(attempt["output"]["sha256"]) == 64
    assert _load_successful_heldout_attempt(root) == raw
    assert _load_successful_heldout_attempt(root) == raw
    with pytest.raises(FileExistsError, match="already exists"):
        run_exactly_once_heldout(
            root,
            evaluator=lambda: pytest.fail("evaluator must not run twice"),
            **_attempt_inputs(),
        )


def test_failed_attempt_is_consumed_and_cannot_be_retried(tmp_path: Path) -> None:
    root = tmp_path / "P6B_heldout"

    def fail() -> dict[str, object]:
        raise RuntimeError("private details must not be persisted: /" + "home/user")

    with pytest.raises(RuntimeError, match="private details"):
        run_exactly_once_heldout(root, evaluator=fail, **_attempt_inputs())

    attempt = json.loads((root / "attempt.json").read_text())
    assert attempt["status"] == "failed"
    assert attempt["error_type"] == "RuntimeError"
    assert "home" not in json.dumps(attempt)
    assert not (root / "heldout_raw.json").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        run_exactly_once_heldout(
            root,
            evaluator=lambda: {"status": "pass"},
            **_attempt_inputs(),
        )
    with pytest.raises(ValueError, match="successful"):
        _load_successful_heldout_attempt(root)


def test_crash_after_raw_publication_recovers_without_evaluator_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "P6B_heldout"
    original_publish = p6b_runner._publish_json_durable
    calls = 0

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_before_attempt_replace(path, value, *, replace):
        if Path(path).name == "attempt.json" and replace:
            raise SimulatedProcessCrash
        return original_publish(path, value, replace=replace)

    def evaluate() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "pass", "rows": [{"value": 1}]}

    monkeypatch.setattr(
        p6b_runner, "_publish_json_durable", crash_before_attempt_replace
    )
    with pytest.raises(SimulatedProcessCrash):
        run_exactly_once_heldout(root, evaluator=evaluate, **_attempt_inputs())
    assert calls == 1
    assert json.loads((root / "attempt.json").read_text())["status"] == "in_progress"
    assert (root / "heldout_raw.json").is_file()

    monkeypatch.setattr(p6b_runner, "_publish_json_durable", original_publish)
    recovered = recover_heldout_attempt(root)

    assert calls == 1
    assert recovered["evaluation"]["rows"] == [{"value": 1}]
    attempt = json.loads((root / "attempt.json").read_text())
    assert attempt["status"] == "success"
    assert attempt["events"][-1]["event"] == "heldout_raw_recovered"
    assert _load_successful_heldout_attempt(root) == recovered


def test_post_raw_ledger_io_failure_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "P6B_heldout"
    original_publish = p6b_runner._publish_json_durable
    failed_once = False

    def fail_first_success_ledger(path, value, *, replace):
        nonlocal failed_once
        if Path(path).name == "attempt.json" and replace and not failed_once:
            failed_once = True
            raise OSError("simulated ledger write failure")
        return original_publish(path, value, replace=replace)

    monkeypatch.setattr(p6b_runner, "_publish_json_durable", fail_first_success_ledger)
    with pytest.raises(OSError, match="ledger write"):
        run_exactly_once_heldout(
            root,
            evaluator=lambda: {"status": "pass", "rows": [{"value": 1}]},
            **_attempt_inputs(),
        )

    attempt = json.loads((root / "attempt.json").read_text())
    assert attempt["status"] == "in_progress"
    assert (root / "heldout_raw.json").is_file()
    monkeypatch.setattr(p6b_runner, "_publish_json_durable", original_publish)
    assert recover_heldout_attempt(root)["evaluation"]["status"] == "pass"


def test_successful_attempt_loader_rejects_raw_tampering(tmp_path: Path) -> None:
    root = tmp_path / "P6B_heldout"
    run_exactly_once_heldout(
        root,
        evaluator=lambda: {"status": "pass", "rows": []},
        **_attempt_inputs(),
    )
    raw = json.loads((root / "heldout_raw.json").read_text())
    raw["evaluation"]["rows"].append({"tampered": True})
    (root / "heldout_raw.json").write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        _load_successful_heldout_attempt(root)


def test_cli_separates_heldout_evaluation_from_repeatable_packaging() -> None:
    parser = _argument_parser()
    evaluate = parser.parse_args(
        [
            "final-evaluate",
            "--cache-directory",
            "cache",
            "--metadata",
            "metadata.json",
            "--selection-root",
            "selection",
            "--output-root",
            "raw",
        ]
    )
    package = parser.parse_args(
        [
            "final-package",
            "--attempt-root",
            "raw",
            "--selection-root",
            "selection",
            "--output-root",
            "artifact",
        ]
    )

    assert evaluate.command == "final-evaluate"
    assert package.command == "final-package"
    assert not hasattr(package, "cache_directory")
    assert not hasattr(package, "metadata")
    with pytest.raises(SystemExit):
        parser.parse_args(["final"])


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
                replace(
                    metric,
                    cluster_metrics=tuple(
                        replace(
                            cluster,
                            strict_online_tmap=0.2,
                            strict_online_trec=0.3,
                        )
                        for cluster in metric.cluster_metrics
                    ),
                    strict_online_tmap=0.2,
                    strict_online_trec=0.3,
                )
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
        ranking_key=candidate_ranking_key(result.selected, baseline=baseline),
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
        verification_ledger=_verification_ledger_from_outputs(
            {
                "threshold_aware_total_score": (0, b"1 passed\n"),
                "gt_free_runtime_api": (0, b"1 passed\n"),
            }
        ),
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


def test_candidate_rows_compress_per_sequence_cluster_metrics() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    row = _candidate_sweep_rows(
        _sweep_candidate(protocol.base, "baseline", official=True)
    )[0]
    clusters = json.loads(row["cluster_metrics_json"])

    assert clusters
    assert all("sequence_metrics" not in cluster for cluster in clusters)
    assert all("sequence_metrics_evidence" in cluster for cluster in clusters)


def test_selection_document_stays_within_source_sized_budget() -> None:
    document = _valid_selection_document()
    payload = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")

    assert len(payload) <= 5 * 1024 * 1024


def test_selection_document_deduplicates_cluster_sequence_evidence() -> None:
    document = _valid_selection_document()
    registry = document["sequence_metric_evidence"]
    registered = {record["sha256"] for record in registry["records"]}
    referenced = set()
    for section in (document["baseline"]["rows"], document["candidate_rows"], document["finalist_rows"]):
        for row in section:
            for cluster in json.loads(row["cluster_metrics_json"]):
                assert "sequence_metrics_evidence" not in cluster
                referenced.add(cluster["sequence_metrics_evidence_sha256"])

    assert registry["schema_version"] == 1
    assert referenced == registered
    assert len(registered) < len(document["candidate_rows"])


def test_selection_publication_enforces_source_sized_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _valid_selection_document()
    output = tmp_path / "P6B_selection"
    monkeypatch.setattr(p6b_runner, "_MAX_SELECTION_DOCUMENT_BYTES", 1, raising=False)

    with pytest.raises(ValueError, match="selection document exceeds"):
        p6b_runner._publish_selection(output, document)

    assert not output.exists()


def test_selection_document_rejects_corrupt_cluster_sequence_evidence(
    tmp_path: Path,
) -> None:
    document = _valid_selection_document()
    evidence = document["sequence_metric_evidence"]["records"][0]["evidence"]
    evidence["records_zlib_base64"] += "AA=="
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="cluster metrics|sequence metric evidence"):
        load_selection_document(path)


def test_selection_document_rejects_heldout_cluster_metrics(tmp_path: Path) -> None:
    document = _valid_selection_document()
    row = document["candidate_rows"][0]
    clusters = json.loads(row["cluster_metrics_json"])
    clusters[0]["reference_scene_id"] = document["split_manifest"][
        "heldout_reference_scene_ids"
    ][0]
    clusters.sort(key=lambda item: item["reference_scene_id"])
    row["cluster_metrics_json"] = json.dumps(
        clusters, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="tuning cluster"):
        load_selection_document(path)


def test_selection_document_binds_frozen_valid_observation_denominators(
    tmp_path: Path,
) -> None:
    document = _valid_selection_document()
    document["candidate_rows"][0]["total_valid_observations"] += 1
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="frozen valid observation|frozen_b4_valid_observations",
    ):
        load_selection_document(path)


def test_selection_rows_record_and_validate_explicit_birth_denominators(
    tmp_path: Path,
) -> None:
    document = _valid_selection_document()
    row = document["candidate_rows"][0]

    assert row["true_births"] == row["births"] - row["false_births"]
    assert row["accepted_births"] == row["births"]
    assert row["valid_birth_opportunities"] == row["births"] + row["rejected_births"]
    assert row["frozen_b4_valid_observations"] == row["total_valid_observations"]

    row["valid_birth_opportunities"] += 1
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="candidate horizon metrics|birth opportunities|valid_birth_opportunities",
    ):
        load_selection_document(path)


def test_selection_rows_bind_exact_tuning_population(tmp_path: Path) -> None:
    document = _valid_selection_document()
    row = document["candidate_rows"][0]

    assert row["tuning_population_count"] == 96
    assert len(row["tuning_population_sha256"]) == 64
    expected = {
        item["reference_scene_id"]: item
        for item in document["tuning_population"]["by_reference"]
    }
    clusters = json.loads(row["cluster_metrics_json"])
    assert {
        cluster["reference_scene_id"]: {
            "reference_scene_id": cluster["reference_scene_id"],
            "count": cluster["sequence_population_count"],
            "sha256": cluster["sequence_population_sha256"],
        }
        for cluster in clusters
    } == expected

    clusters[0]["sequence_population_sha256"] = "0" * 64
    row["cluster_metrics_json"] = json.dumps(
        clusters, sort_keys=True, separators=(",", ":")
    )
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="tuning population|candidate horizon metrics"):
        load_selection_document(path)


@pytest.mark.parametrize(
    "mutation",
    ("candidate_grid", "finalist", "ranking", "population", "proof"),
)
def test_selection_validator_rejects_derived_evidence_tampering(
    tmp_path: Path, mutation: str
) -> None:
    document = _valid_selection_document()
    if mutation == "candidate_grid":
        del document["candidate_rows"][:4]
    elif mutation == "finalist":
        del document["finalist_rows"][:4]
    elif mutation == "ranking":
        document["ranking_key"][0] = float(document["ranking_key"][0]) + 1.0
    elif mutation == "population":
        document["candidate_rows"][0]["tuning_population_sha256"] = "0" * 64
    else:
        document["verification_ledger"]["proofs"][0]["exit_status"] = 1
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError):
        load_selection_document(path)


def test_final_evaluate_rejects_external_selection_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external = tmp_path / "selection"
    external.mkdir()
    (external / "selection.json").write_text("{}", encoding="utf-8")
    called = False

    def consume_attempt(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("held-out evaluator must not be called")

    monkeypatch.setattr(p6b_runner, "run_exactly_once_heldout", consume_attempt)
    monkeypatch.setattr(
        p6b_runner,
        "build_source_tree_contract",
        lambda *args, **kwargs: {"status": "pass", "source_commit": "a" * 40},
    )

    with pytest.raises(ValueError, match="canonical P6-B selection"):
        p6b_runner.run_final_evaluate(
            cache_directory=tmp_path / "cache",
            metadata_path=tmp_path / "metadata.json",
            selection_root=external,
            output_root=p6b_runner.PROJECT_ROOT / "artifacts/P6B_heldout",
            config_path=Path("conf/p6b/default.yaml"),
        )
    assert not called


def test_final_evaluate_rejects_noncanonical_config_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "repo"
    selection_root = project / "artifacts/P6B_selection"
    selection_root.mkdir(parents=True)
    selection_bytes = b"{}\n"
    (selection_root / "selection.json").write_bytes(selection_bytes)
    canonical_config = project / "conf/p6b/default.yaml"
    canonical_config.parent.mkdir(parents=True)
    canonical_config.write_text("seed: 45\n", encoding="utf-8")
    external_config = tmp_path / "equivalent.yaml"
    external_config.write_text("seed: 45\n# copied\n", encoding="utf-8")
    called = False

    def consume_attempt(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("held-out evaluator must not be called")

    monkeypatch.setattr(p6b_runner, "PROJECT_ROOT", project)
    monkeypatch.setattr(
        p6b_runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=selection_bytes, returncode=0),
    )
    monkeypatch.setattr(
        p6b_runner,
        "build_source_tree_contract",
        lambda *args, **kwargs: {"status": "pass", "source_commit": "a" * 40},
    )
    monkeypatch.setattr(
        p6b_runner,
        "load_selection_document",
        lambda path: {
            "source_commit": "a" * 40,
            "split_manifest": {"sha256": "b" * 64},
            "provenance": {
                "p6b_config_sha256": hashlib.sha256(
                    canonical_config.read_bytes()
                ).hexdigest()
            },
        },
    )
    monkeypatch.setattr(
        p6b_runner, "_validate_selection_source_boundary", lambda *args: None
    )
    monkeypatch.setattr(p6b_runner, "run_exactly_once_heldout", consume_attempt)

    with pytest.raises(ValueError, match="canonical P6-B config"):
        p6b_runner.run_final_evaluate(
            cache_directory=tmp_path / "cache",
            metadata_path=tmp_path / "metadata.json",
            selection_root=selection_root,
            output_root=project / "artifacts/P6B_heldout",
            config_path=external_config,
        )
    assert not called


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
        verification_proofs_passed=True,
    )

    assert all(record["passed"] for record in gates.values())
    worse = [dict(row) for row in baseline + candidate]
    next(row for row in worse if row["method"] == "P6B" and row["T"] == "T5")[
        "identity_switch_rate"
    ] = 0.11
    stopped = compute_final_gate_results(
        worse,
        evidence_complete=True,
        frozen_hashes_unchanged=True,
        verification_proofs_passed=True,
    )
    assert stopped["G6B-2"]["passed"] is False


def test_g6b1_requires_executed_verification_proofs() -> None:
    rows = [
        _metric(method, horizon) for method in ("B4", "P6B") for horizon in (2, 3, 4, 5)
    ]

    gates = compute_final_gate_results(
        rows,
        evidence_complete=True,
        frozen_hashes_unchanged=True,
        verification_proofs_passed=False,
    )

    assert not gates["G6B-1"]["passed"]


def test_official_evidence_population_uses_all_sequence_updates() -> None:
    import torch

    mask = torch.ones(120, dtype=torch.bool)
    prediction = {
        "pred_masks": mask[:, None],
        "pred_classes": torch.tensor([3]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": mask[None, :],
        "labels": torch.tensor([3]),
        "ids": torch.tensor([1]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(120, dtype=torch.long),
    }
    metric = OfficialMetricAccumulator(mode="strict_online")
    metric.update(prediction, target)
    rows = []
    for method in ("B4", "P6B"):
        for horizon in range(2, 6):
            for index in range(2):
                rows.append(
                    {
                        "method": method,
                        "reference_scene_id": f"ref-{index}",
                        "master_sequence_id": f"master-{index}",
                        "order_id": "canonical",
                        "T": f"T{horizon}",
                        "prediction_digest": str(index) * 64,
                        "state": metric.export_evidence(),
                    }
                )

    evidence = p6b_runner._official_metric_population_evidence(
        SimpleNamespace(per_sequence_metric_evidence=tuple(rows))
    )

    assert len(evidence) == 8
    assert {record["state"]["updates"] for record in evidence} == {2}


def test_source_contract_rejects_tracked_and_untracked_dirty_files(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
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
    untracked = tmp_path / "untracked.py"
    untracked.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        build_source_tree_contract(tmp_path)
    untracked.unlink()
    subprocess.run(
        ["git", "update-index", "--skip-worktree", "tracked.txt"],
        cwd=tmp_path,
        check=True,
    )
    try:
        with pytest.raises(ValueError, match="hidden index"):
            build_source_tree_contract(tmp_path)
    finally:
        subprocess.run(
            ["git", "update-index", "--no-skip-worktree", "tracked.txt"],
            cwd=tmp_path,
            check=True,
        )


def test_source_contract_can_allow_only_the_exact_heldout_evidence_prefix(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    heldout = tmp_path / "artifacts/P6B_heldout/attempt.json"
    heldout.parent.mkdir(parents=True)
    heldout.write_text("{}\n", encoding="utf-8")

    contract = build_source_tree_contract(
        tmp_path, allowed_dirty_prefixes=("artifacts/P6B_heldout/",)
    )

    assert contract["status"] == "pass"
    (tmp_path / "unexpected.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        build_source_tree_contract(
            tmp_path, allowed_dirty_prefixes=("artifacts/P6B_heldout/",)
        )
