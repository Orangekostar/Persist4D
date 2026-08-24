import json

from scripts.final_evidence_capacity import (
    CapacityBootstrap,
    CapacityEvaluation,
    CapacityRobustness,
)
from scripts.run_final_capacity import build_capacity_artifact_payloads


def test_capacity_artifact_payloads_bind_tables_statistics_and_gate() -> None:
    evaluation = CapacityEvaluation(
        per_sequence_rows=(
            {
                "reference_scene_id": "scene-a",
                "capacity": 100,
                "horizon": 5,
                "causal_prefix_t_mAP": 0.5,
            },
        ),
        aggregate_rows=(
            {
                "capacity": 100,
                "horizon": 5,
                "sequence_count": 1,
                "causal_prefix_t_mAP": 0.5,
            },
        ),
    )
    bootstrap = CapacityBootstrap(
        effects=(
            {
                "capacity": 128,
                "reference_capacity": 100,
                "horizon": 5,
                "metric": "causal_prefix_t_REC",
                "effect": 0.0,
            },
        ),
        per_scene_effects=(
            {
                "capacity": 128,
                "reference_capacity": 100,
                "horizon": 5,
                "metric": "causal_prefix_t_REC",
                "reference_scene_id": "scene-a",
                "effect": 0.0,
            },
        ),
    )
    robustness = CapacityRobustness(
        robust_improvement=False,
        candidates=({"capacity": 128, "robust": False},),
    )

    payloads = build_capacity_artifact_payloads(
        evaluation=evaluation,
        bootstrap=bootstrap,
        robustness=robustness,
        gate="CAPACITY_100_OK",
        provenance={"source": "fixture"},
    )

    assert set(payloads) == {
        "capacity_aggregate.csv",
        "capacity_cluster_bootstrap.csv",
        "capacity_gate.json",
        "capacity_per_scene_effects.csv",
        "capacity_per_sequence.csv",
        "capacity_raw.json",
        "capacity_robustness.json",
    }
    assert b"scene-a" in payloads["capacity_per_sequence.csv"]
    assert json.loads(payloads["capacity_gate.json"])["classification"] == (
        "CAPACITY_100_OK"
    )
    assert json.loads(payloads["capacity_raw.json"])["provenance"] == {
        "source": "fixture"
    }
