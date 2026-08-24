from dataclasses import replace

import pytest
import torch

from scripts.final_evidence_capacity import (
    CAPACITY_GRID,
    assess_robust_capacity_improvement,
    build_class_mapper_from_label_document,
    build_protocol_from_reviewer_manifest,
    capacity_cluster_bootstrap,
    classify_capacity_gate,
    replay_capacity_grid,
    validate_capacity_replays,
)


def _observation(features: list[list[float]]) -> dict[str, torch.Tensor]:
    query_count = len(features)
    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "class_prob": torch.tensor(
            [[1.0, 0.0] for _ in range(query_count)], dtype=torch.float32
        ),
        "confidence": torch.ones(query_count, dtype=torch.float32),
        "valid": torch.ones(query_count, dtype=torch.bool),
    }


def test_capacity_grid_matches_the_preregistered_values() -> None:
    assert CAPACITY_GRID == (64, 100, 128, 160, 200)


def test_capacity_grid_replays_identical_observations_and_counts_births() -> None:
    observations = (
        _observation([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        _observation([[-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]]),
    )

    replays = replay_capacity_grid(
        observations,
        sequence_id="fixture",
        capacities=(2, 4),
        association_threshold=2.0,
    )

    assert [replay.capacity for replay in replays] == [2, 4]
    assert len({replay.observation_sha256 for replay in replays}) == 1
    assert [stage.occupied_count for stage in replays[0].stages] == [2, 2]
    assert [stage.birth_attempts for stage in replays[0].stages] == [3, 3]
    assert [stage.accepted_births for stage in replays[0].stages] == [2, 0]
    assert [stage.rejected_births for stage in replays[0].stages] == [1, 3]
    assert [stage.occupied_count for stage in replays[1].stages] == [3, 4]
    assert [stage.accepted_births for stage in replays[1].stages] == [3, 1]
    assert [stage.rejected_births for stage in replays[1].stages] == [0, 2]
    assert all(
        stage.active_count <= stage.occupied_count
        for replay in replays
        for stage in replay.stages
    )
    assert all(
        stage.dormant_count == stage.occupied_count - stage.active_count
        for replay in replays
        for stage in replay.stages
    )


def test_capacity_bootstrap_pairs_independent_scene_clusters() -> None:
    rows = []
    for scene, reference_value, candidate_value in (
        ("scene-a", 0.50, 0.60),
        ("scene-b", 0.40, 0.60),
    ):
        for capacity, value in ((100, reference_value), (128, candidate_value)):
            rows.append(
                {
                    "reference_scene_id": scene,
                    "capacity": capacity,
                    "horizon": 5,
                    "causal_prefix_t_REC": value,
                }
            )

    result = capacity_cluster_bootstrap(
        rows,
        reference_capacity=100,
        candidate_capacities=(128,),
        metrics=("causal_prefix_t_REC",),
        horizons=(5,),
        expected_cluster_count=2,
        replicates=100,
        seed=45,
    )

    assert result.effects[0]["cluster_count"] == 2
    assert result.effects[0]["effect"] == pytest.approx(0.15)
    assert result.effects[0]["candidate_mean"] == pytest.approx(0.60)
    assert result.effects[0]["reference_mean"] == pytest.approx(0.45)
    per_scene = {
        row["reference_scene_id"]: row["effect"] for row in result.per_scene_effects
    }
    assert per_scene == pytest.approx({"scene-a": 0.1, "scene-b": 0.2})


def test_capacity_robustness_requires_primary_gain_and_no_degradation() -> None:
    effects = [
        {
            "capacity": 128,
            "horizon": 5,
            "metric": "causal_prefix_t_REC",
            "effect": 0.02,
            "ci_lower": 0.01,
        }
    ]
    aggregate = [
        {
            "capacity": capacity,
            "horizon": horizon,
            "causal_prefix_t_mAP": 0.50,
            "normalized_id_switch_rate": 0.10,
        }
        for capacity in (100, 128)
        for horizon in (2, 3, 4, 5)
    ]

    decision = assess_robust_capacity_improvement(
        effects,
        aggregate,
        reference_capacity=100,
        candidate_capacities=(128,),
        primary_metrics=("causal_prefix_t_REC", "gap_recovery_recall"),
        minimum_absolute_improvement=0.01,
        maximum_t_map_drop=0.005,
        maximum_id_switch_increase=0.005,
    )

    assert decision.robust_improvement is True
    assert decision.candidates[0]["robust"] is True

    degraded = [dict(row) for row in aggregate]
    degraded[-1]["normalized_id_switch_rate"] = 0.11
    decision = assess_robust_capacity_improvement(
        effects,
        degraded,
        reference_capacity=100,
        candidate_capacities=(128,),
        primary_metrics=("causal_prefix_t_REC", "gap_recovery_recall"),
        minimum_absolute_improvement=0.01,
        maximum_t_map_drop=0.005,
        maximum_id_switch_increase=0.005,
    )

    assert decision.robust_improvement is False
    assert decision.candidates[0]["robust"] is False


def test_capacity_replay_state_bytes_match_the_real_state_tensors() -> None:
    replay = replay_capacity_grid(
        (_observation([[1.0, 0.0]]),),
        sequence_id="bytes",
        capacities=(64,),
    )[0]

    assert replay.stages[0].state_bytes == 2440


def test_capacity_replay_rejects_cross_capacity_observation_drift() -> None:
    replays = replay_capacity_grid(
        (_observation([[1.0, 0.0]]),),
        sequence_id="digest",
        capacities=(2, 4),
    )
    drifted = (replays[0], replace(replays[1], observation_sha256="0" * 64))

    with pytest.raises(ValueError, match="identical frozen observations"):
        validate_capacity_replays(drifted)


@pytest.mark.parametrize("capacities", [(), (0,), (64, 64), (64, True)])
def test_capacity_grid_rejects_invalid_capacity_sets(
    capacities: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="capacities"):
        replay_capacity_grid(
            (_observation([[1.0, 0.0]]),),
            sequence_id="invalid",
            capacities=capacities,
        )


def test_reviewer_manifest_reconstructs_exact_cache_protocol() -> None:
    document = {
        "protocol": {"order_variants": ["canonical", "reverse", "sha256_seed45"]},
        "masters": [
            {
                "master_sequence_id": "master-a",
                "reference_scene_id": "scene-a",
                "orders": {
                    "canonical": {"visit_order": ["a", "b", "c", "d", "e"]},
                    "reverse": {"visit_order": ["e", "d", "c", "b", "a"]},
                    "sha256_seed45": {"visit_order": ["b", "d", "a", "e", "c"]},
                },
            }
        ],
    }

    protocol = build_protocol_from_reviewer_manifest(document, expected_master_count=1)

    assert protocol["order_variants"] == (
        "canonical",
        "reverse",
        "sha256_seed45",
    )
    assert protocol["masters"] == (
        {"sequence_id": "master-a", "reference_scene_id": "scene-a"},
    )
    assert protocol["variants"]["master-a"]["canonical"]["scan_ids"] == (
        "a",
        "b",
        "c",
        "d",
        "e",
    )


def test_rio_class_mapper_uses_validation_order_after_frozen_offset() -> None:
    labels = {
        0: {"validation": False},
        1: {"validation": True},
        2: {"validation": True},
        3: {"validation": True},
        7: {"validation": True},
        11: {"validation": True},
    }

    mapper = build_class_mapper_from_label_document(
        labels, foreground_class_count=3, label_offset=2
    )

    assert [mapper(index) for index in range(3)] == [3, 7, 11]
    with pytest.raises(ValueError, match="outside"):
        mapper(3)


@pytest.mark.parametrize(
    ("robust_improvement", "development_split", "selected", "expected"),
    [
        (False, False, False, "CAPACITY_100_OK"),
        (True, False, False, "CAPACITY_SENSITIVITY_ONLY"),
        (True, True, False, "CAPACITY_SENSITIVITY_ONLY"),
        (True, True, True, "CAPACITY_CONFIG_REOPEN"),
    ],
)
def test_capacity_gate_respects_the_preexisting_selection_boundary(
    robust_improvement: bool,
    development_split: bool,
    selected: bool,
    expected: str,
) -> None:
    assert (
        classify_capacity_gate(
            robust_improvement=robust_improvement,
            preexisting_development_split=development_split,
            selected_without_final_tuning=selected,
            architecture_unchanged=True,
        )
        == expected
    )
