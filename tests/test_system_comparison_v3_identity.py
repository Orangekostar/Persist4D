from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.p6a_association import TrackStep
from scripts.system_comparison_v3_identity import (
    IDENTITY_COUNT_FIELDS,
    IDENTITY_RATE_FIELDS,
    aggregate_identity_rows,
    assert_identity_coverage,
    build_b4_minus_b2_cluster_effects,
    compare_b4_to_frozen,
    run_fresh_tracker_steps,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/identity"


def _metrics(*, switches: int = 1, transitions: int = 2) -> dict[str, object]:
    return {
        "deployment_id_switches": switches,
        "identity_transition_opportunities": transitions,
        "fragmentation_count": switches,
        "fragmentation_opportunities": transitions,
        "merge_count": 0,
        "merge_opportunities": transitions,
        "gap_opportunities": 2,
        "recovery_attempts": 1,
        "correct_recoveries": 1,
        "normalized_id_switch_rate": switches / transitions,
        "fragmentation_rate": switches / transitions,
        "merge_rate": 0.0,
        "gap_recovery_accuracy": 1.0,
        "gap_recovery_recall": 0.5,
    }


def _row(
    *,
    method: str,
    master: str,
    order: str,
    horizon: int,
    reference: str,
    switches: int = 1,
) -> dict[str, object]:
    return {
        "method": method,
        "reference_scene_id": reference,
        "master_sequence_id": master,
        "order_id": order,
        "horizon": horizon,
        **_metrics(switches=switches),
        "association_event_count": 10,
        "new_birth_count": 3,
        "false_birth_count": 1,
        "birth_rejected_count": 0,
    }


def test_fresh_tracker_runner_passes_observations_without_gt() -> None:
    observations = (object(), object())
    received = []

    class Tracker:
        def step(self, observation, *, stage_id):
            received.append((observation, stage_id))
            return TrackStep(
                method="B2",
                sequence_id="sequence-0",
                stage_id=stage_id,
                track_ids=(stage_id, None),
                matched_previous=(-1, -1),
                scores=(None, None),
                births=(True, False),
                valid=(True, True),
            )

    steps = run_fresh_tracker_steps(
        factory=lambda sequence_id: Tracker(),
        observations=observations,
        sequence_id="sequence-0",
    )

    assert received == [(observations[0], 0), (observations[1], 1)]
    assert [step.stage_id for step in steps] == [0, 1]


def test_fresh_tracker_runner_rejects_invalid_issued_ids() -> None:
    class Tracker:
        def step(self, observation, *, stage_id):
            return TrackStep(
                method="B2",
                sequence_id="sequence-0",
                stage_id=stage_id,
                track_ids=(-1, -1),
                matched_previous=(-1, -1),
                scores=(None, None),
                births=(True, True),
                valid=(True, True),
            )

    with pytest.raises(ValueError, match="issued IDs"):
        run_fresh_tracker_steps(
            factory=lambda sequence_id: Tracker(),
            observations=(object(),),
            sequence_id="sequence-0",
        )


def test_identity_coverage_requires_b2_b3_b4_129_sequences() -> None:
    methods = ("B2", "B3", "B4")
    orders = ("canonical", "reverse", "sha256_seed45")
    rows = [
        _row(
            method=method,
            master=f"master-{index:03d}",
            order=orders[index % 3],
            horizon=horizon,
            reference=f"reference-{index % 6}",
        )
        for method in methods
        for index in range(129)
        for horizon in (2, 3, 4, 5)
    ]
    assert assert_identity_coverage(rows)["row_count"] == 1548

    with pytest.raises(ValueError, match="coverage"):
        assert_identity_coverage(rows[:-1])


def test_b4_regression_is_exact_and_detects_drift() -> None:
    fresh = [
        _row(
            method="B4",
            master="master-0",
            order="canonical",
            horizon=2,
            reference="reference-0",
        )
    ]
    frozen = [
        {
            "method": "Persist4D",
            "master_sequence_id": "master-0",
            "order_id": "canonical",
            "horizon": "2",
            **{
                field: "" if fresh[0][field] is None else str(fresh[0][field])
                for field in (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS)
            },
        }
    ]
    assert compare_b4_to_frozen(fresh, frozen) == {
        "status": "pass_exact",
        "cell_count": 14,
        "row_count": 1,
        "max_abs_diff": 0.0,
    }

    fresh[0]["gap_recovery_recall"] = 0.5001
    with pytest.raises(ValueError, match="regression"):
        compare_b4_to_frozen(fresh, frozen)


def test_identity_aggregation_pools_counts_before_rates() -> None:
    rows = [
        _row(
            method="B2",
            master="master-0",
            order="canonical",
            horizon=5,
            reference="reference-0",
            switches=1,
        ),
        _row(
            method="B2",
            master="master-1",
            order="canonical",
            horizon=5,
            reference="reference-0",
            switches=2,
        ),
    ]
    rows[1]["identity_transition_opportunities"] = 8
    rows[1]["normalized_id_switch_rate"] = 0.25

    pooled = aggregate_identity_rows(rows)

    assert pooled["deployment_id_switches"] == 3
    assert pooled["identity_transition_opportunities"] == 10
    assert pooled["normalized_id_switch_rate"] == pytest.approx(0.3)
    assert pooled["false_birth_count"] == 2
    assert pooled["association_event_count"] == 20


def test_cluster_effects_show_all_six_paired_b4_minus_b2_differences() -> None:
    rows = []
    for index in range(6):
        reference = f"reference-{index}"
        for method, switches in (("B2", 2), ("B3", 1), ("B4", 0)):
            row = _row(
                method=method,
                master=f"cluster-{index}",
                order="all",
                horizon=5,
                reference=reference,
                switches=switches,
            )
            rows.append(row)

    effects = build_b4_minus_b2_cluster_effects(rows)

    assert len(effects) == 6
    assert {row["reference_scene_id"] for row in effects} == {
        f"reference-{index}" for index in range(6)
    }
    assert all(row["normalized_id_switch_rate_difference"] == -1.0 for row in effects)
    assert all(row["gap_recovery_recall_difference"] == 0.0 for row in effects)

    with pytest.raises(ValueError, match="six"):
        build_b4_minus_b2_cluster_effects(rows[:-3])


def test_identity_cli_help_runs_directly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/system_comparison_v3_identity.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--cache-root" in completed.stdout


def test_identity_artifact_contract_when_generated() -> None:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("identity artifacts have not been generated yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["gate_id0"]["status"] == "PASS"
    assert manifest["gate_id0"]["b4_frozen_regression"]["status"] == "pass_exact"
    assert manifest["coverage"] == {
        "aggregate_row_count": 48,
        "cluster_effect_row_count": 24,
        "horizons": [2, 3, 4, 5],
        "methods": ["B2", "B3", "B4"],
        "orders": ["canonical", "reverse", "sha256_seed45"],
        "per_cluster_row_count": 288,
        "per_sequence_row_count": 1548,
        "reference_cluster_count": 6,
        "sequence_count": 129,
    }
    assert manifest["channel_contract"]["task_and_identity_are_separate"] is True
    assert manifest["execution"]["regression_input"] == (
        "frozen_v1_query_observation_cache"
    )
    assert manifest["execution"]["v2_task_cache_used_for_identity"] is False

    for name, metadata in manifest["outputs"].items():
        content = (ARTIFACT_ROOT / name).read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    expected_rows = {
        "identity_per_sequence.csv": 1548,
        "identity_aggregate.csv": 48,
        "identity_per_cluster.csv": 288,
    }
    for name, count in expected_rows.items():
        with (ARTIFACT_ROOT / name).open(newline="", encoding="utf-8") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == count
