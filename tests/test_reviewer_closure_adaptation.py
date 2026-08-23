from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import reviewer_closure_adaptation as adaptation
from scripts import run_reviewer_closure_adaptation as runner

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/reviewer_closure/phase_ii_evaluation.yaml"


def test_adaptation_runner_supports_direct_script_entrypoint(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/run_reviewer_closure_adaptation.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "{smoke,cache,finalize,evaluate,profile,gate}" in completed.stdout


def _full_key(index: int, horizon: int) -> dict[str, object]:
    scan_ids = [f"scan-{index}-{stage}" for stage in range(horizon)]
    return {
        "master_sequence_id": f"master-{index}",
        "reference_scene_id": f"reference-{index % 6}",
        "order_id": "canonical",
        "context_index": index,
        "context_scan_indices": list(range(index * 5, index * 5 + 5)),
        "horizon": horizon,
        "history_scan_ids": scan_ids,
        "scan_indices": list(range(index * 5, index * 5 + horizon)),
        "task_quality": "causal_prefix",
    }


def _sidecar_key(full: dict[str, object]) -> dict[str, object]:
    return {
        name: copy.deepcopy(full[name])
        for name in (
            "reference_scene_id",
            "master_sequence_id",
            "order_id",
            "horizon",
            "history_scan_ids",
            "scan_indices",
        )
    }


def test_phase_ii_config_freezes_statistics_and_gate_before_results() -> None:
    config = adaptation.load_phase_ii_evaluation_config(CONFIG_PATH)

    assert config["strongest_simple_tracker_method_id"] == "B2"
    assert config["evaluation_horizons"] == [2, 3, 4, 5]
    assert config["statistics"] == {
        "cluster_unit": "reference_scene_id",
        "cluster_count": 6,
        "bootstrap_replicates": 10_000,
        "seed": 45,
        "confidence_level": 0.95,
    }
    assert config["gate_ii"]["substantial_task_advantage"][
        "minimum_pooled_absolute_difference"
    ] == 0.01
    assert config["gate_ii"]["compute_no_material_disadvantage"][
        "maximum_ratio"
    ] == 1.10


def test_expected_adapted_keys_are_exact_t2_to_t5_subset() -> None:
    keys = [_full_key(index, horizon) for index in range(129) for horizon in range(1, 6)]

    selected = adaptation.expected_adapted_keys(keys)

    assert len(selected) == 516
    assert {int(key["horizon"]) for key in selected} == {2, 3, 4, 5}
    assert selected == [key for key in keys if int(key["horizon"]) >= 2]
    with pytest.raises(adaptation.AdaptationEvidenceError, match="coverage|516"):
        adaptation.expected_adapted_keys(keys[:-1])


def test_resume_reinfers_prediction_without_sidecar_and_rejects_orphans() -> None:
    expected = [_full_key(0, 2), _full_key(1, 2)]
    commit = "a" * 40
    prediction = {
        "key": expected[0],
        "content_sha256": "1" * 64,
    }

    pending = adaptation.validate_adapted_resume(
        expected_keys=expected,
        prediction_entries=[prediction],
        sidecar_entries=[],
        source_commit=commit,
    )
    assert pending == expected

    sidecar = {
        "key": _sidecar_key(expected[0]),
        "source_prediction_content_sha256": "1" * 64,
        "reference_prediction_content_sha256": "1" * 64,
        "sidecar_source_commit": commit,
    }
    pending = adaptation.validate_adapted_resume(
        expected_keys=expected,
        prediction_entries=[prediction],
        sidecar_entries=[sidecar],
        source_commit=commit,
    )
    assert pending == [expected[1]]

    with pytest.raises(adaptation.AdaptationEvidenceError, match="orphan|prediction"):
        adaptation.validate_adapted_resume(
            expected_keys=expected,
            prediction_entries=[],
            sidecar_entries=[sidecar],
            source_commit=commit,
        )


def _paired_rows() -> list[dict[str, object]]:
    rows = []
    for reference in range(6):
        for order in ("canonical", "reverse", "sha256_seed45"):
            for horizon in (4, 5):
                for method in ("FullHistoryAdaptedB2", "Persist4D"):
                    challenger = method == "FullHistoryAdaptedB2"
                    rows.append(
                        {
                            "method_id": method,
                            "reference_scene_id": f"reference-{reference}",
                            "master_sequence_id": f"master-{reference}",
                            "order_id": order,
                            "horizon": horizon,
                            "causal_prefix_t_mAP": 0.2 if challenger else 0.1,
                            "causal_prefix_t_REC": 0.3 if challenger else 0.2,
                            "deployment_id_switches": 2 if challenger else 1,
                            "identity_transition_opportunities": 10,
                            "fragmentation_count": 2 if challenger else 1,
                            "fragmentation_opportunities": 10,
                            "merge_count": 2 if challenger else 1,
                            "merge_opportunities": 10,
                            "gap_opportunities": 10,
                            "recovery_attempts": 10,
                            "correct_recoveries": 5 if challenger else 8,
                        }
                    )
    return rows


def test_paired_statistics_use_six_clusters_and_recompute_identity_rates() -> None:
    evidence = adaptation.paired_phase_ii_statistics(
        _paired_rows(),
        challenger_method_id="FullHistoryAdaptedB2",
        baseline_method_id="Persist4D",
    )

    bootstrap = {
        (row["metric"], row["horizon"]): row for row in evidence["bootstrap"]
    }
    task = bootstrap[("causal_prefix_t_mAP", 4)]
    switches = bootstrap[("normalized_id_switch_rate", 4)]
    gap = bootstrap[("gap_recovery_recall", 5)]
    assert task["cluster_count"] == 6
    assert task["difference"] == pytest.approx(0.1)
    assert task["ci_lower"] == pytest.approx(0.1)
    assert switches["challenger_mean"] == pytest.approx(0.2)
    assert switches["baseline_mean"] == pytest.approx(0.1)
    assert switches["difference"] == pytest.approx(0.1)
    assert gap["difference"] == pytest.approx(-0.3)
    assert len(evidence["order_robustness"]) == 4 * 2
    assert len(evidence["leave_one_scene_out"]) == 4 * 2 * 6


def _task_evidence(qualifying_horizons: set[int]) -> list[dict[str, object]]:
    rows = []
    for metric in ("causal_prefix_t_mAP", "causal_prefix_t_REC"):
        for horizon in (4, 5):
            qualifying = horizon in qualifying_horizons
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "pooled_difference": 0.02 if qualifying else 0.005,
                    "ci_lower": 0.01 if qualifying else -0.01,
                    "ci_upper": 0.03,
                    "order_consistent": qualifying,
                    "loso_consistent": qualifying,
                }
            )
    return rows


def _identity_evidence(*, persist_advantage: bool) -> list[dict[str, object]]:
    rows = []
    for metric in ("normalized_id_switch_rate", "gap_recovery_recall"):
        for horizon in (4, 5):
            difference = 0.05 if metric == "normalized_id_switch_rate" else -0.05
            if not persist_advantage:
                difference = -difference
            rows.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "difference": difference,
                    "ci_lower": difference - 0.01,
                    "ci_upper": difference + 0.01,
                    "order_consistent": True,
                    "loso_consistent": True,
                }
            )
    return rows


def _compute_evidence(ratio: float) -> list[dict[str, object]]:
    return [
        {
            "horizon": horizon,
            "latency_ratio": ratio,
            "peak_allocated_vram_ratio": ratio,
            "cumulative_scan_ratio": ratio,
        }
        for horizon in (4, 5)
    ]


@pytest.mark.parametrize(
    ("task_horizons", "persist_advantage", "compute_ratio", "expected"),
    (
        ({4, 5}, False, 1.05, "FULL_HISTORY_DOMINANT"),
        ({4}, True, 2.0, "ACCURACY_ADVANTAGE_BUT_COSTLY"),
        (set(), True, 2.0, "HORIZON_ROBUST"),
    ),
)
def test_gate_ii_classification_is_conjunctive_and_fail_closed(
    task_horizons: set[int],
    persist_advantage: bool,
    compute_ratio: float,
    expected: str,
) -> None:
    gate = adaptation.derive_gate_ii(
        task_evidence=_task_evidence(task_horizons),
        identity_evidence=_identity_evidence(
            persist_advantage=persist_advantage
        ),
        compute_evidence=_compute_evidence(compute_ratio),
        config=adaptation.load_phase_ii_evaluation_config(CONFIG_PATH),
    )

    assert gate["classification"] == expected
    assert gate["status"] == "pass"
    assert len(gate["content_sha256"]) == 64


def test_adapted_provenance_binds_checkpoint_config_protocol_and_commit(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "adapted.ckpt"
    checkpoint.write_bytes(b"adapted checkpoint")
    documents = {"runtime": b"a: 1\n", "phase_ii": b"schema: 1\n"}

    first = runner.build_adapted_provenance(
        checkpoint_path=checkpoint,
        source_commit="a" * 40,
        config_documents=documents,
        protocol_sha256="b" * 64,
    )
    second = runner.build_adapted_provenance(
        checkpoint_path=checkpoint,
        source_commit="a" * 40,
        config_documents=dict(reversed(list(documents.items()))),
        protocol_sha256="b" * 64,
    )

    assert first == second
    assert set(first) == {
        "source_commit",
        "checkpoint_sha256",
        "config_sha256",
        "protocol_sha256",
    }
    changed = runner.build_adapted_provenance(
        checkpoint_path=checkpoint,
        source_commit="a" * 40,
        config_documents={**documents, "phase_ii": b"schema: 2\n"},
        protocol_sha256="b" * 64,
    )
    assert changed["checkpoint_sha256"] == first["checkpoint_sha256"]
    assert changed["config_sha256"] != first["config_sha256"]


def test_adapted_batch_runs_one_forward_per_missing_sidecar() -> None:
    keys = [_full_key(0, 2), _full_key(1, 2)]
    commit = "a" * 40
    calls = []
    written_predictions = []
    written_sidecars = []

    class Producer:
        def produce_bundle(self, key):
            calls.append(key)
            digest = f"{len(calls):064x}"
            return SimpleNamespace(
                payload={"key": key, "content_sha256": digest},
                processed=SimpleNamespace(raw_observation={"digest": digest}),
            )

    def write_prediction(payload):
        written_predictions.append(payload)
        return {
            "key": payload["key"],
            "content_sha256": payload["content_sha256"],
        }

    def build_sidecar(*, key, raw_observation, prediction, source_commit):
        assert raw_observation["digest"] == prediction["content_sha256"]
        return {
            "key": _sidecar_key(key),
            "source_prediction_content_sha256": prediction["content_sha256"],
            "reference_prediction_content_sha256": prediction["content_sha256"],
            "sidecar_source_commit": source_commit,
        }

    def write_sidecar(payload):
        written_sidecars.append(payload)
        return payload

    result = runner.produce_adapted_batch(
        expected_keys=keys,
        existing_prediction_entries=[],
        existing_sidecar_entries=[],
        producer=Producer(),
        prediction_writer=write_prediction,
        sidecar_builder=build_sidecar,
        sidecar_writer=write_sidecar,
        source_commit=commit,
        smoke_only=False,
    )

    assert result == {
        "expected_count": 2,
        "reused_count": 0,
        "produced_count": 2,
    }
    assert calls == keys
    assert [row["key"] for row in written_predictions] == keys
    assert [row["key"] for row in written_sidecars] == [
        _sidecar_key(key) for key in keys
    ]


def test_adapted_sidecar_manifest_requires_exact_prediction_binding() -> None:
    keys = [_full_key(0, 2), _full_key(1, 2)]
    commit = "a" * 40
    prediction_entries = [
        {"key": key, "content_sha256": f"{index + 1:064x}"}
        for index, key in enumerate(keys)
    ]
    prediction_manifest = {
        "content_sha256": "f" * 64,
        "provenance": {
            "source_commit": commit,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "protocol_sha256": "d" * 64,
        },
        "entries": prediction_entries,
    }
    sidecars = [
        {
            "key": _sidecar_key(key),
            "content_sha256": f"{index + 10:064x}",
            "source_prediction_content_sha256": f"{index + 1:064x}",
            "reference_prediction_content_sha256": f"{index + 1:064x}",
            "sidecar_source_commit": commit,
        }
        for index, key in enumerate(keys)
    ]

    manifest = runner.build_adapted_sidecar_manifest(
        entries=sidecars,
        expected_keys=keys,
        prediction_manifest=prediction_manifest,
        source_commit=commit,
    )
    assert manifest["entry_count"] == 2
    assert manifest["prediction_manifest_content_sha256"] == "f" * 64
    assert len(manifest["content_sha256"]) == 64

    with pytest.raises(adaptation.AdaptationEvidenceError, match="coverage|exact"):
        runner.build_adapted_sidecar_manifest(
            entries=sidecars[:-1],
            expected_keys=keys,
            prediction_manifest=prediction_manifest,
            source_commit=commit,
        )


TASK_VALUES = {
    "causal_prefix_t_mAP": 0.1,
    "causal_prefix_t_mAP50": 0.2,
    "causal_prefix_t_mAP25": 0.3,
    "causal_prefix_t_REC": 0.4,
    "causal_prefix_t_REC50": 0.5,
    "causal_prefix_t_REC25": 0.6,
    "current_stage_AP": 0.7,
    "current_stage_AP50": 0.8,
    "current_stage_AP25": 0.9,
    "current_stage_REC": 1.0,
}
IDENTITY_VALUES = {
    "deployment_id_switches": 1,
    "identity_transition_opportunities": 2,
    "fragmentation_count": 1,
    "fragmentation_opportunities": 2,
    "merge_count": 1,
    "merge_opportunities": 2,
    "gap_opportunities": 2,
    "recovery_attempts": 2,
    "correct_recoveries": 1,
    "normalized_id_switch_rate": 0.5,
    "fragmentation_rate": 0.5,
    "merge_rate": 0.5,
    "gap_recovery_accuracy": 0.5,
    "gap_recovery_recall": 0.5,
}


def _formal_scope_rows() -> list[dict[str, object]]:
    rows = []
    for context in range(43):
        for order in ("canonical", "reverse", "sha256_seed45"):
            for horizon in (2, 3, 4, 5):
                rows.append(
                    {
                        "reference_scene_id": f"reference-{context % 6}",
                        "master_sequence_id": f"master-{context}",
                        "order_id": order,
                        "horizon": horizon,
                    }
                )
    return rows


def test_merge_phase_ii_rows_has_exact_five_method_coverage() -> None:
    scopes = _formal_scope_rows()
    adapted_tasks = [{**scope, **TASK_VALUES} for scope in scopes]
    adapted_identity = [
        {
            "method_id": method,
            **scope,
            **IDENTITY_VALUES,
        }
        for scope in scopes
        for method in ("FullHistoryNative", "B2")
    ]
    frozen = [
        {
            "method_id": method,
            "method": method,
            **scope,
            "tracker_initialization_horizon": 1 if method == "Persist4D" else 2,
            "task_metric_source": "frozen",
            **TASK_VALUES,
            **IDENTITY_VALUES,
        }
        for scope in scopes
        for method in ("FullHistoryNative", "B2", "Persist4D")
    ]

    rows = adaptation.merge_phase_ii_per_sequence(
        adapted_task_rows=adapted_tasks,
        adapted_identity_rows=adapted_identity,
        frozen_rows=frozen,
    )

    assert len(rows) == 2580
    assert {row["method_id"] for row in rows} == {
        "FullHistoryFrozenNative",
        "FullHistoryFrozenB2",
        "FullHistoryAdaptedNative",
        "FullHistoryAdaptedB2",
        "Persist4D",
    }
    adapted_b2 = next(
        row for row in rows if row["method_id"] == "FullHistoryAdaptedB2"
    )
    assert adapted_b2["task_metric_source"] == "adapted_checkpoint_cache"
    assert adapted_b2["tracker_initialization_horizon"] == 2
    assert adapted_b2["causal_prefix_t_mAP"] == 0.1
    assert adapted_b2["deployment_id_switches"] == 1


def test_aggregate_phase_ii_rows_keeps_pooled_task_and_recomputes_identity() -> None:
    scopes = _formal_scope_rows()
    method_ids = (
        "FullHistoryFrozenNative",
        "FullHistoryFrozenB2",
        "FullHistoryAdaptedNative",
        "FullHistoryAdaptedB2",
        "Persist4D",
    )
    per_sequence = [
        {
            "method_id": method,
            "method": method,
            **scope,
            "tracker_initialization_horizon": 1 if method == "Persist4D" else 2,
            "task_metric_source": "synthetic",
            **TASK_VALUES,
            **IDENTITY_VALUES,
        }
        for scope in scopes
        for method in method_ids
    ]
    task_rows = [
        {
            "method_id": method,
            "order_id": order,
            "horizon": horizon,
            "sequence_count": 129 if order == "all" else 43,
            "task_metric_source": "pooled_official",
            **{name: value + 0.01 for name, value in TASK_VALUES.items()},
        }
        for method in method_ids
        for order in ("all", "canonical", "reverse", "sha256_seed45")
        for horizon in (2, 3, 4, 5)
    ]

    summaries = adaptation.aggregate_phase_ii_results(
        per_sequence_rows=per_sequence,
        pooled_task_rows=task_rows,
    )

    assert len(summaries["results"]) == 20
    assert len(summaries["per_order"]) == 60
    row = next(
        value
        for value in summaries["results"]
        if value["method_id"] == "FullHistoryAdaptedB2" and value["horizon"] == 4
    )
    assert row["sequence_count"] == 129
    assert row["causal_prefix_t_mAP"] == pytest.approx(0.11)
    assert row["deployment_id_switches"] == 129
    assert row["identity_transition_opportunities"] == 258
    assert row["normalized_id_switch_rate"] == pytest.approx(0.5)


def test_phase_ii_compute_rows_use_six_clusters_median_latency_and_peak_vram() -> None:
    adapted_profile = []
    frozen_profile = []
    for reference in range(6):
        for horizon in (2, 3, 4, 5):
            base = {
                "reference_scene_id": f"reference-{reference}",
                "master_sequence_id": f"master-{reference}",
                "order_id": "canonical",
                "horizon": horizon,
                "status": "pass",
                "median_latency_ms": float(reference + horizon),
                "peak_allocated_mib": float(100 + reference + horizon),
                "peak_reserved_mib": float(200 + reference + horizon),
                "update_scan_count": horizon,
                "cumulative_scan_count": horizon * (horizon + 1) // 2 - 1,
                "update_point_count": 1000 + reference,
                "cumulative_point_count": 2000 + reference,
                "explicit_history_input_bytes": 3000 + reference,
            }
            adapted_profile.append({"method": "FullHistoryAdapted", **base})
            frozen_profile.extend(
                {"method": method, **base}
                for method in ("FullHistory", "Persist4D")
            )

    rows = adaptation.build_phase_ii_compute_rows(
        adapted_profile_rows=adapted_profile,
        frozen_profile_rows=frozen_profile,
    )

    assert len(rows) == 12
    adapted_t4 = next(
        row
        for row in rows
        if row["method_id"] == "FullHistoryAdapted" and row["horizon"] == 4
    )
    assert adapted_t4["profile_cluster_count"] == 6
    assert adapted_t4["median_latency_ms"] == pytest.approx(6.5)
    assert adapted_t4["peak_allocated_mib"] == pytest.approx(109.0)
    assert adapted_t4["cumulative_scans_processed"] == pytest.approx(9.0)


def test_gate_inputs_keep_pooled_task_and_directional_robustness_separate() -> None:
    result_rows = []
    for method, offset in (("FullHistoryAdaptedB2", 0.02), ("Persist4D", 0.0)):
        for horizon in (4, 5):
            result_rows.append(
                {
                    "method_id": method,
                    "horizon": horizon,
                    "causal_prefix_t_mAP": 0.1 + offset,
                    "causal_prefix_t_REC": 0.2 + offset,
                }
            )
    bootstrap = []
    order = []
    loso = []
    for metric in adaptation.STATISTICAL_METRICS:
        for horizon in (4, 5):
            identity = metric in adaptation.IDENTITY_STATISTICAL_METRICS
            difference = (
                0.05
                if metric == "normalized_id_switch_rate"
                else -0.05 if metric == "gap_recovery_recall" else 0.02
            )
            bootstrap.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "difference": difference,
                    "ci_lower": difference - 0.005,
                    "ci_upper": difference + 0.005,
                }
            )
            order.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "canonical_difference": difference,
                    "reverse_difference": difference,
                    "sha256_seed45_difference": difference,
                }
            )
            for reference in range(6):
                loso.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "dropped_reference_scene_id": f"reference-{reference}",
                        "difference": difference,
                        "identity": identity,
                    }
                )
    compute = [
        {
            "method_id": method,
            "horizon": horizon,
            "median_latency_ms": latency,
            "peak_allocated_mib": latency,
            "cumulative_scans_processed": scans,
        }
        for method, latency, scans in (
            ("FullHistoryAdapted", 100.0, 9.0),
            ("Persist4D", 100.0, 9.0),
        )
        for horizon in (4, 5)
    ]

    evidence = adaptation.build_gate_ii_evidence(
        result_rows=result_rows,
        bootstrap_rows=bootstrap,
        order_rows=order,
        loso_rows=loso,
        compute_rows=compute,
    )

    task = next(
        row
        for row in evidence["task"]
        if row["metric"] == "causal_prefix_t_mAP" and row["horizon"] == 4
    )
    identity = next(
        row
        for row in evidence["identity"]
        if row["metric"] == "normalized_id_switch_rate" and row["horizon"] == 4
    )
    assert task["pooled_difference"] == pytest.approx(0.02)
    assert task["order_consistent"] is True
    assert task["loso_consistent"] is True
    assert identity["order_consistent"] is True
    assert identity["loso_consistent"] is True
    assert evidence["compute"][0]["latency_ratio"] == pytest.approx(1.0)
