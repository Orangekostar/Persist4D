from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from scripts.p6a_metrics import (
    OfficialMetricAccumulator,
    build_official_metric_population_evidence,
    recompute_official_metric_population_evidence,
)
from scripts.p6b_artifacts import (
    P6B_ARTIFACT_SCHEMA_VERSION,
    _validate_per_sequence,
    build_p6b_artifact_root,
    finalize_p6b_artifact,
    publish_p6b_artifact,
    render_p6b_bundle,
    validate_p6b_artifact,
)
from scripts.p6b_protocol import build_split_manifest, load_p6b_config
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BClusterMetrics,
    P6BHorizonMetrics,
    P6BSequenceAssociationMetrics,
    candidate_ranking_key,
    run_staged_sweep,
)
from scripts.run_p6b_evaluation import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_P5_SHA256,
    EXPECTED_P6A_SHA256,
    _candidate_sweep_rows,
    _expected_selection_provenance,
    _paired_statistics,
    _selection_protocol,
    _split_population_evidence,
    _verification_ledger_from_outputs,
    build_selection_document,
)


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
    _protocol, split = _selection_protocol()
    evidence = _split_population_evidence(split, partition="tuning")
    return {item["reference_scene_id"]: item for item in evidence["by_reference"]}


@lru_cache(maxsize=1)
def _tuning_identities_by_reference() -> dict[str, tuple[tuple[str, str, str], ...]]:
    _protocol, split = _selection_protocol()
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


def _metric(horizon: int, *, official: bool) -> P6BHorizonMetrics:
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


def _candidate(config, stage: str, *, official: bool) -> P6BCandidateRow:
    return P6BCandidateRow(
        config=config,
        stage=stage,
        horizons=tuple(_metric(horizon, official=official) for horizon in (2, 3, 4, 5)),
    )


@lru_cache(maxsize=1)
def _selection_cache() -> dict[str, object]:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate(protocol.base, "baseline", official=True)
    result = run_staged_sweep(
        protocol,
        baseline=baseline,
        fast_evaluator=lambda config, stage: _candidate(config, stage, official=False),
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


def _selection() -> dict[str, object]:
    return deepcopy(_selection_cache())


def _per_sequence(selection: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    heldout_assignments = [
        assignment
        for assignment in selection["split_manifest"]["assignments"]
        if assignment["partition"] == "heldout"
    ]
    for assignment in heldout_assignments:
        for order in assignment["order_ids"]:
            digest = hashlib.sha256(
                f"{assignment['master_sequence_id']}|{order}".encode()
            ).hexdigest()
            for horizon in range(2, 6):
                for method, switches in (("B4", 2), ("P6B", 1)):
                    predicted = 0 if horizon == 2 else 4
                    correct = 0 if horizon == 2 else 4 - switches
                    attempts = 0 if horizon == 2 else 4
                    gaps = 0 if horizon == 2 else 5
                    rows.append(
                        {
                            "method": method,
                            "reference_scene_id": assignment["reference_scene_id"],
                            "master_sequence_id": assignment["master_sequence_id"],
                            "order_id": order,
                            "T": f"T{horizon}",
                            "prediction_digest": digest,
                            "t_mAP": 1.0,
                            "t_REC": 1.0,
                            "identity_switches": switches,
                            "transition_opportunities": 10,
                            "identity_switch_rate": switches / 10,
                            "wrong_reactivations": 0 if horizon == 2 else switches,
                            "predicted_reactivation_events": predicted,
                            "wrong_reactivation_rate": (
                                None if not predicted else switches / predicted
                            ),
                            "correct_reactivations": correct,
                            "reactivation_attempts": attempts,
                            "gap_opportunities": gaps,
                            "reactivation_accuracy": (
                                None if not attempts else correct / attempts
                            ),
                            "reactivation_recall": None if not gaps else correct / gaps,
                            "false_births": switches,
                            "births": 8,
                            "rejected_births": 2,
                            "false_birth_rate": switches / 10,
                        }
                    )
    return sorted(
        rows,
        key=lambda row: (
            ("B4", "P6B").index(row["method"]),
            row["reference_scene_id"],
            row["master_sequence_id"],
            row["order_id"],
            row["T"],
        ),
    )


def _final_rows(per_sequence: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "identity_switches",
        "transition_opportunities",
        "wrong_reactivations",
        "predicted_reactivation_events",
        "correct_reactivations",
        "reactivation_attempts",
        "gap_opportunities",
        "false_births",
        "births",
        "rejected_births",
    )
    rows = []
    for method in ("B4", "P6B"):
        for horizon in ("T2", "T3", "T4", "T5"):
            scoped = [
                row
                for row in per_sequence
                if row["method"] == method and row["T"] == horizon
            ]
            counts = {field: sum(int(row[field]) for row in scoped) for field in fields}
            rows.append(
                {
                    "method": method,
                    "T": horizon,
                    "t_mAP": 0.20 if method == "B4" else 0.21,
                    "t_REC": 0.30 if method == "B4" else 0.31,
                    **counts,
                    "identity_switch_rate": counts["identity_switches"]
                    / counts["transition_opportunities"],
                    "wrong_reactivation_rate": (
                        None
                        if not counts["predicted_reactivation_events"]
                        else counts["wrong_reactivations"]
                        / counts["predicted_reactivation_events"]
                    ),
                    "reactivation_accuracy": (
                        None
                        if not counts["reactivation_attempts"]
                        else counts["correct_reactivations"]
                        / counts["reactivation_attempts"]
                    ),
                    "reactivation_recall": (
                        None
                        if not counts["gap_opportunities"]
                        else counts["correct_reactivations"]
                        / counts["gap_opportunities"]
                    ),
                    "false_birth_rate": counts["false_births"]
                    / (counts["births"] + counts["rejected_births"]),
                }
            )
    return rows


@lru_cache(maxsize=1)
def _official_metric_evidence() -> dict[str, object]:
    point_count = 120
    mask = torch.ones(point_count, dtype=torch.bool)
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
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }
    metric = OfficialMetricAccumulator(mode="strict_online")
    metric.update(prediction, target)
    return metric.export_evidence()


@lru_cache(maxsize=1)
def _partial_official_metric_evidence() -> dict[str, object]:
    point_count = 240
    first_mask = torch.zeros(point_count, dtype=torch.bool)
    first_mask[: point_count // 2] = True
    second_mask = ~first_mask
    prediction = {
        "pred_masks": first_mask[:, None],
        "pred_classes": torch.tensor([3]),
        "pred_scores": torch.tensor([0.9]),
    }
    target = {
        "masks": torch.stack((first_mask, second_mask)),
        "labels": torch.tensor([3, 3]),
        "ids": torch.tensor([1, 2]),
        "changes": torch.tensor([0, 0]),
        "temporal_stages": torch.zeros(point_count, dtype=torch.long),
    }
    metric = OfficialMetricAccumulator(mode="strict_online")
    metric.update(prediction, target)
    assert metric.compute()["online_t-mAP"] == 0.5
    return metric.export_evidence()


@lru_cache(maxsize=1)
def _root_cache() -> dict[str, object]:
    selection = _selection()
    per_sequence = _per_sequence(selection)
    failure_diagnostics = [
        {
            "method": row["method"],
            "reference_scene_id": row["reference_scene_id"],
            "master_sequence_id": row["master_sequence_id"],
            "order_id": row["order_id"],
            "T": row["T"],
            "prediction_digest": row["prediction_digest"],
            **{
                category: 0
                for category in (
                    *tuple(f"F{index}" for index in range(1, 8)),
                    "unclassified",
                )
            },
        }
        for row in per_sequence
    ]
    final_results = _final_rows(per_sequence)
    for row in final_results:
        row["t_mAP"] = 1.0
        row["t_REC"] = 1.0
    evaluation = {
        "status": "pass",
        "heldout_order_count": 33,
        "heldout_reference_scene_ids": selection["split_manifest"][
            "heldout_reference_scene_ids"
        ],
        "selected_config_id": selection["selected_config_id"],
        "selected_config_sha256": selection["selected_config_sha256"],
        "provenance": {
            "checkpoint": {
                "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
                "sha256": EXPECTED_CHECKPOINT_SHA256,
            },
            "p5": {
                "ref": "repo:artifacts/P5/persist4d_mvp_eval.json",
                "sha256": EXPECTED_P5_SHA256,
            },
            "p6a": {
                "ref": "repo:artifacts/P6A/p6a_eval.json",
                "sha256": EXPECTED_P6A_SHA256,
            },
            "p6a_protocol_manifest": {
                "ref": "repo:artifacts/P6A/protocol_b_manifest.json",
                "sha256": selection["provenance"]["p6a_protocol_manifest_sha256"],
            },
            "p6a_cache_manifest": {
                "ref": "external:p6a_cache_cee151a/cache_manifest.json",
                "sha256": selection["provenance"]["cache_manifest_sha256"],
            },
        },
        "final_results": final_results,
        "official_metric_evidence": [
            {
                "method": method,
                "T": horizon,
                "state": build_official_metric_population_evidence(
                    [
                        {
                            "reference_scene_id": row["reference_scene_id"],
                            "master_sequence_id": row["master_sequence_id"],
                            "order_id": row["order_id"],
                            "prediction_digest": row["prediction_digest"],
                            "state": deepcopy(_official_metric_evidence()),
                        }
                        for row in per_sequence
                        if row["method"] == method and row["T"] == horizon
                    ]
                ),
            }
            for method in ("B4", "P6B")
            for horizon in ("T2", "T3", "T4", "T5")
        ],
        "per_sequence_results": per_sequence,
        "failure_analysis": [
            {
                "method": method,
                "T": horizon,
                "failure_category": failure,
                "count": 0,
            }
            for method in ("B4", "P6B")
            for horizon in ("T2", "T3", "T4", "T5")
            for failure in (
                *tuple(f"F{index}" for index in range(1, 8)),
                "unclassified",
            )
        ],
        "failure_diagnostics": failure_diagnostics,
        "statistical_analysis": _paired_statistics(per_sequence),
    }
    selection_sha = hashlib.sha256(
        (
            json.dumps(selection, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()
    started_utc = "2026-08-21T00:00:00Z"
    command = ["final-evaluate", "--protocol", "P6B-v2"]
    selection_ref = {
        "ref": "repo:artifacts/P6B_selection/selection.json",
        "sha256": selection_sha,
    }
    inputs = {
        "source_commit": "b" * 40,
        "selection": selection_ref,
        "split_sha256": selection["split_manifest"]["sha256"],
        "p6b_config_sha256": selection["provenance"]["p6b_config_sha256"],
        "command": command,
    }
    attempt_id = hashlib.sha256(
        (
            json.dumps({**inputs, "started_utc": started_utc}, sort_keys=True, indent=2)
            + "\n"
        ).encode()
    ).hexdigest()
    raw = {
        "schema_version": 1,
        "protocol_version": 2,
        "attempt_id": attempt_id,
        "source_commit": inputs["source_commit"],
        "selection": selection_ref,
        "split_sha256": inputs["split_sha256"],
        "p6b_config_sha256": inputs["p6b_config_sha256"],
        "evaluation": evaluation,
    }
    raw_bytes = (
        json.dumps(raw, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    events = [
        {"event": "attempt_token_published", "utc": started_utc},
        {"event": "heldout_raw_published", "utc": "2026-08-21T01:00:00Z"},
    ]
    attempt = {
        "schema_version": 1,
        "protocol_version": 2,
        "attempt_id": attempt_id,
        "source_commit": raw["source_commit"],
        "selection": raw["selection"],
        "split_sha256": raw["split_sha256"],
        "p6b_config_sha256": raw["p6b_config_sha256"],
        "command": command,
        "input_sha256": hashlib.sha256(
            (json.dumps(inputs, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest(),
        "status": "success",
        "started_utc": started_utc,
        "ended_utc": "2026-08-21T01:00:00Z",
        "exit_status": 0,
        "error_type": None,
        "events": events,
        "log_sha256": hashlib.sha256(
            (json.dumps(events, sort_keys=True, indent=2) + "\n").encode()
        ).hexdigest(),
        "output": {
            "ref": "repo:artifacts/P6B_heldout/heldout_raw.json",
            "bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        },
    }
    return build_p6b_artifact_root(
        source_tree_contract={"status": "pass", "source_commit": "c" * 40},
        selection_document=selection,
        heldout_attempt=attempt,
        heldout_raw=raw,
    )


def _root() -> dict[str, object]:
    return deepcopy(_root_cache())


def test_finalize_validates_complete_schema_and_manifest_binding() -> None:
    root = finalize_p6b_artifact(_root())
    validate_p6b_artifact(root)
    files = render_p6b_bundle(root)

    assert root["schema_version"] == P6B_ARTIFACT_SCHEMA_VERSION == 2
    assert root["decision"] == "P6B_GO"
    assert root["source_lineage"] == {
        "schema_version": 1,
        "selection_source_commit": "a" * 40,
        "evaluation_source_commit": "b" * 40,
        "package_source_commit": "c" * 40,
        "selection_to_evaluation_allowed_prefix": "artifacts/P6B_selection/",
        "evaluation_to_package_allowed_prefix": "artifacts/P6B_heldout/",
    }
    assert set(files) >= {
        "p6b_eval.json",
        "P6B_GO_NOGO_REPORT.md",
        "artifact_manifest.json",
        "selected_config.yaml",
        "hyperparameter_sweep.csv",
        "final_results.csv",
        "per_sequence_results.csv",
        "failure_analysis.csv",
        "failure_diagnostics.csv",
        "statistical_analysis.json",
        "statistical_analysis.md",
        "execution_attempt.json",
        "heldout_raw.json",
        "figures/identity_comparison.svg",
    }
    records = {record["path"]: record for record in root["artifact_manifest"]}
    for path, record in records.items():
        assert record["bytes"] == len(files[path])
        assert record["sha256"] == hashlib.sha256(files[path]).hexdigest()


def test_finalize_accepts_json_object_key_reordering() -> None:
    finalized = finalize_p6b_artifact(json.loads(json.dumps(_root(), sort_keys=True)))
    validate_p6b_artifact(finalized)


def test_report_has_exact_sections_numeric_gates_and_terminal_decision() -> None:
    report = render_p6b_bundle(finalize_p6b_artifact(_root()))[
        "P6B_GO_NOGO_REPORT.md"
    ].decode()

    assert report.count("\n## ") == 11
    assert report.rstrip().endswith("P6B_GO")
    assert "relative reduction=" in report
    assert (
        "Sample SD"
        in render_p6b_bundle(finalize_p6b_artifact(_root()))[
            "statistical_analysis.md"
        ].decode()
    )
    assert "Only two held-out" in report
    assert "2026-08-21T00:00:00Z" in report
    assert "2026-08-21T01:00:00Z" in report
    assert "attempt_token_published" in report
    assert "heldout_raw_published" in report
    assert "B4 T2 total failures=0" in report
    assert "P6B T5 total failures=0" in report


def test_packaging_can_retry_without_mutating_heldout_raw(tmp_path: Path) -> None:
    root = _root()
    raw_before = json.dumps(
        root["heldout_raw"], sort_keys=True, separators=(",", ":")
    ).encode()
    invalid = deepcopy(root)
    invalid["heldout_raw"]["evaluation"]["per_sequence_results"].pop()
    _rebind_raw(invalid)

    with pytest.raises(
        ValueError,
        match="264|missing method/horizon|population differs",
    ):
        finalize_p6b_artifact(invalid)

    finalized = finalize_p6b_artifact(root)
    publish_p6b_artifact(tmp_path / "P6B", finalized)
    raw_after = json.dumps(
        root["heldout_raw"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert raw_after == raw_before


def test_artifact_rejects_heldout_population_outside_frozen_split() -> None:
    root = _root()
    for row in root["heldout_raw"]["evaluation"]["per_sequence_results"]:
        row["master_sequence_id"] = f"forged-{row['master_sequence_id']}"
    _rebind_raw(root)

    with pytest.raises(
        ValueError,
        match="held-out population|split assignment|outside held-out protocol",
    ):
        finalize_p6b_artifact(root)


def test_heldout_reactivation_counts_must_satisfy_core_identities() -> None:
    root = _root()
    evaluation = root["heldout_raw"]["evaluation"]
    row = evaluation["per_sequence_results"][1]
    row.update(
        wrong_reactivations=0,
        predicted_reactivation_events=4,
        correct_reactivations=3,
        wrong_reactivation_rate=0.0,
        reactivation_accuracy=3 / row["reactivation_attempts"],
        reactivation_recall=3 / row["gap_opportunities"],
    )
    selection = root["selection_document"]
    expected_units = {
        (
            assignment["reference_scene_id"],
            assignment["master_sequence_id"],
            order_id,
        )
        for assignment in selection["split_manifest"]["assignments"]
        if assignment["partition"] == "heldout"
        for order_id in assignment["order_ids"]
    }

    with pytest.raises(ValueError, match="predicted minus correct"):
        _validate_per_sequence(evaluation, expected_units)


def test_failure_analysis_is_recomputed_from_diagnostics() -> None:
    root = _root()
    root["heldout_raw"]["evaluation"]["failure_analysis"][0]["count"] = 999_999
    _rebind_raw(root)

    with pytest.raises(ValueError, match="failure_analysis.*diagnostic"):
        finalize_p6b_artifact(root)


def test_final_official_metrics_are_recomputed_from_sufficient_state() -> None:
    root = _root()
    evaluation = root["heldout_raw"]["evaluation"]
    assert len(evaluation["official_metric_evidence"]) == 8

    evaluation["final_results"][-1]["t_mAP"] = 0.0
    _rebind_raw(root)
    with pytest.raises(ValueError, match="official metric evidence"):
        finalize_p6b_artifact(root)


def test_official_metric_evidence_is_bound_to_heldout_identity_population() -> None:
    root = _root()
    evidence = root["heldout_raw"]["evaluation"]["official_metric_evidence"][0]
    _computed, records, _per_sequence = recompute_official_metric_population_evidence(
        evidence["state"]
    )
    altered = [deepcopy(record) for record in records]
    altered[0]["master_sequence_id"] = "forged-master"
    evidence["state"] = build_official_metric_population_evidence(altered)
    _rebind_raw(root)

    with pytest.raises(ValueError, match="held-out identity population"):
        finalize_p6b_artifact(root)


def test_official_metric_evidence_is_bound_to_each_per_sequence_metric() -> None:
    root = _root()
    evaluation = root["heldout_raw"]["evaluation"]
    evidence = evaluation["official_metric_evidence"][0]
    method, horizon = evidence["method"], evidence["T"]
    _computed, records, _per_sequence = recompute_official_metric_population_evidence(
        evidence["state"]
    )
    altered = [deepcopy(record) for record in records]
    altered[0]["state"] = deepcopy(_partial_official_metric_evidence())
    evidence["state"] = build_official_metric_population_evidence(altered)
    computed, _records, _per_sequence = recompute_official_metric_population_evidence(
        evidence["state"]
    )
    final = next(
        row
        for row in evaluation["final_results"]
        if row["method"] == method and row["T"] == horizon
    )
    final["t_mAP"] = computed["online_t-mAP"]
    final["t_REC"] = computed["online_t-REC"]
    _rebind_raw(root)

    with pytest.raises(ValueError, match="per-sequence official metric"):
        finalize_p6b_artifact(root)


def test_source_lineage_is_bound_to_embedded_documents() -> None:
    root = build_p6b_artifact_root(
        source_tree_contract={"status": "pass", "source_commit": "c" * 40},
        selection_document=_root()["selection_document"],
        heldout_attempt=_root()["heldout_attempt"],
        heldout_raw=_root()["heldout_raw"],
    )
    root["source_lineage"]["evaluation_source_commit"] = "d" * 40

    with pytest.raises(ValueError, match="source lineage"):
        finalize_p6b_artifact(root)


def test_claims_and_next_action_are_derived_from_gate_decision() -> None:
    root = _root()
    root["claims_supported"] = ["P6-B proves P7 and P8 are ready."]
    root["next_action"] = "Start P7/P8 immediately."

    with pytest.raises(ValueError, match="claims|next_action|narrative"):
        finalize_p6b_artifact(root)


def test_protocol_deviation_forces_g6b5_stop() -> None:
    root = _root()
    root["protocol_deviations"] = ["held-out protocol changed"]

    with pytest.raises(ValueError, match="G6B-5|gate_results|decision"):
        finalize_p6b_artifact(root)


def _rebind_raw(root: dict[str, object]) -> None:
    payload = (
        json.dumps(root["heldout_raw"], sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    root["heldout_attempt"]["output"].update(
        bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )


def _private_path_mutation(root: dict[str, object]) -> None:
    root["heldout_raw"]["evaluation"]["provenance"]["checkpoint"].update(
        ref="/" + "home/private/model.ckpt"
    )
    _rebind_raw(root)


def _count_mutation(root: dict[str, object]) -> None:
    root["heldout_raw"]["evaluation"]["per_sequence_results"][0].update(
        identity_switches=99
    )
    _rebind_raw(root)


def _statistic_mutation(root: dict[str, object]) -> None:
    root["heldout_raw"]["evaluation"]["statistical_analysis"][0].update(
        mean_delta=123.0
    )
    _rebind_raw(root)


def _cache_provenance_mutation(root: dict[str, object]) -> None:
    root["heldout_raw"]["evaluation"]["provenance"]["p6a_cache_manifest"].update(
        ref="external:other-cache/cache_manifest.json",
        sha256="9" * 64,
    )
    _rebind_raw(root)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda root: root.update(extra=True), "keys"),
        (lambda root: root.update(decision="P6B_STOP"), "decision"),
        (_private_path_mutation, "portable|private"),
        (_count_mutation, "identity switches|rate|sum|numerator"),
        (_statistic_mutation, "bootstrap recomputation"),
        (_cache_provenance_mutation, "frozen P6-A.*provenance|provenance.*frozen P6-A"),
        (
            lambda root: root["heldout_attempt"]["output"].update(sha256="0" * 64),
            "manifest binding",
        ),
        (
            lambda root: root["heldout_attempt"].update(input_sha256="0" * 64),
            "input SHA-256",
        ),
        (
            lambda root: root["heldout_attempt"].update(log_sha256="0" * 64),
            "event-log SHA-256",
        ),
        (lambda root: root.update(claims_supported=["SOTA"]), "claim"),
    ),
)
def test_artifact_rejects_tampered_raw_statistics_attempt_and_claims(
    mutation, match: str
) -> None:
    root = _root()
    mutation(root)
    with pytest.raises(ValueError, match=match):
        finalize_p6b_artifact(root)


def test_publish_is_atomic_and_refuses_existing_or_symlink_root(tmp_path: Path) -> None:
    root = finalize_p6b_artifact(_root())
    output = tmp_path / "P6B"
    publish_p6b_artifact(output, root)
    assert (output / "p6b_eval.json").is_file()

    with pytest.raises(FileExistsError):
        publish_p6b_artifact(output, root)
    link = tmp_path / "linked"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(FileExistsError):
        publish_p6b_artifact(link, root)
