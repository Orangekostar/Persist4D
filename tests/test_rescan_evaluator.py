from __future__ import annotations

import numpy as np
import pytest
import torch

from datasets.rescan_adapter import IdentityAlternatives, RescanEvaluatorTarget
from scripts import evaluate_rescan_persist4d as rescan_evaluator
from scripts.evaluate_rescan_persist4d import (
    RescanEvaluationError,
    StageIdentityMatches,
    aggregate_rescan_results,
    build_rescan_stage_target,
    canonicalize_ambiguous_identities,
    derive_external_gate,
    evaluate_identity_sequence,
    evaluate_rescan_sequence,
    prepare_rescan_model_batch,
)


def _target() -> RescanEvaluatorTarget:
    return RescanEvaluatorTarget(
        scene_id="scene_a",
        capture_id="scene_a_0",
        class_ids=np.asarray([5, 5, 13, 13, 0], dtype=np.int32),
        instance_ids=np.asarray([4, 4, 9, 9, 1024], dtype=np.int32),
        identity_keys=(
            ("scene_a", 4),
            ("scene_a", 4),
            ("scene_a", 9),
            ("scene_a", 9),
            ("scene_a", 1024),
        ),
        ambiguities=IdentityAlternatives({4: (4, 7)}),
    )


def _label_map() -> dict[str, object]:
    return {
        "mappings": [
            {
                "source_class_id": 5,
                "target_class_id": 2,
                "status": "exact",
            },
            {
                "source_class_id": 13,
                "target_class_id": None,
                "status": "unsupported",
            },
        ]
    }


def test_stage_targets_keep_level_a_exact_and_level_b_all_official_instances() -> None:
    level_a = build_rescan_stage_target(_target(), _label_map(), level="A")
    level_b = build_rescan_stage_target(_target(), _label_map(), level="B")

    assert level_a.identity_ids.tolist() == [4]
    assert level_a.class_ids.tolist() == [2]
    assert level_a.masks.tolist() == [[True, True, False, False, False]]
    assert level_b.identity_ids.tolist() == [4, 9]
    assert level_b.class_ids.tolist() == [5, 13]
    assert level_b.masks.shape == (2, 5)
    assert level_a.accepted_identity_ids == ((4, 7),)


def test_collated_batch_is_sanitized_before_model_forward() -> None:
    data = {
        "features": torch.ones(2, 3),
        "labels": [torch.tensor([[1], [2]])],
        "original_labels": [np.asarray([[1], [2]])],
        "segment2label": [],
        "target_full": [
            {
                "point2segment": torch.tensor([0, 1]),
                "temporal_stages": torch.tensor([0, 0]),
                "ambiguities": None,
            }
        ],
    }
    targets = [
        {
            "point2segment": torch.tensor([0, 1]),
            "temporal_stages": torch.tensor([0, 0]),
        }
    ]

    prepared = prepare_rescan_model_batch(data, targets)

    assert set(prepared.target) == {"point2segment", "temporal_stages"}
    assert prepared.full_point2segment.tolist() == [0, 1]
    assert prepared.full_temporal_stages.tolist() == [0, 0]
    assert not {
        "labels",
        "original_labels",
        "segment2label",
        "target_full",
    }.intersection(prepared.data)


def test_collated_batch_rejects_any_nonempty_ambiguity_metadata() -> None:
    data = {
        "target_full": [
            {
                "point2segment": torch.tensor([0]),
                "temporal_stages": torch.tensor([0]),
                "ambiguities": {4: [4, 7]},
            }
        ]
    }
    targets = [
        {"point2segment": torch.tensor([0]), "temporal_stages": torch.tensor([0])}
    ]

    with pytest.raises(RescanEvaluationError, match="ambiguity"):
        prepare_rescan_model_batch(data, targets)


def test_official_ambiguity_assignment_avoids_penalizing_an_allowed_swap() -> None:
    canonical = canonicalize_ambiguous_identities(
        raw_identity_ids=(4, 7),
        predicted_track_ids=(70, 40),
        alternatives=IdentityAlternatives({4: (4, 7), 7: (7, 4)}),
        track_identity_history={40: 4, 70: 7},
    )

    assert canonical == (7, 4)


def test_identity_metrics_count_natural_gap_and_correct_recovery() -> None:
    stages = (
        StageIdentityMatches(
            raw_identity_ids=(4,),
            predicted_track_ids=(11,),
            alternatives=IdentityAlternatives({}),
        ),
        StageIdentityMatches(
            raw_identity_ids=(),
            predicted_track_ids=(),
            alternatives=IdentityAlternatives({}),
        ),
        StageIdentityMatches(
            raw_identity_ids=(4,),
            predicted_track_ids=(11,),
            alternatives=IdentityAlternatives({}),
        ),
    )

    metrics = evaluate_identity_sequence(stages)

    assert metrics["eligible_identity_count"] == 1
    assert metrics["gap_opportunities"] == 1
    assert metrics["recovery_attempts"] == 1
    assert metrics["correct_recoveries"] == 1
    assert metrics["gap_recovery_accuracy"] == 1.0
    assert metrics["gap_recovery_recall"] == 1.0
    assert metrics["normalized_id_switch_rate"] == 0.0


def _cache_entry(stage: int, mask: torch.Tensor) -> dict[str, object]:
    class_prob = torch.zeros((2, 19), dtype=torch.float32)
    class_prob[0, 2] = 0.9
    class_prob[0, 18] = 0.1
    class_prob[1, 18] = 1.0
    return {
        "key": {"stage_index": stage, "target_capture_id": f"scene_a_{stage}"},
        "observation": {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "class_prob": class_prob,
            "confidence": torch.tensor([0.9, 0.0]),
            "valid": torch.tensor([True, False]),
            "masks": torch.stack((mask, torch.zeros_like(mask))),
            "mask_support": torch.tensor([int(mask.sum()), 0]),
        },
    }


def _stage_target(stage: int, mask: torch.Tensor) -> RescanEvaluatorTarget:
    instance_ids = np.where(mask.numpy(), 4, 1024).astype(np.int32)
    class_ids = np.where(mask.numpy(), 5, 0).astype(np.int32)
    return RescanEvaluatorTarget(
        scene_id="scene_a",
        capture_id=f"scene_a_{stage}",
        class_ids=class_ids,
        instance_ids=instance_ids,
        identity_keys=tuple(("scene_a", int(value)) for value in instance_ids),
        ambiguities=IdentityAlternatives({}),
    )


def test_sequence_evaluation_fans_one_cache_into_all_frozen_trackers() -> None:
    first = torch.tensor([True, True, False, False])
    second = torch.tensor([False, False, True, True])

    result = evaluate_rescan_sequence(
        scene_id="scene_a",
        cache_entries=(_cache_entry(0, first), _cache_entry(1, second)),
        evaluator_targets=(_stage_target(0, first), _stage_target(1, second)),
        label_map=_label_map(),
    )

    assert set(result["methods"]) == {"B1", "B2", "B3", "B4"}
    for method in result["methods"].values():
        assert method["level_a"]["raw_local_AP50"] == 1.0
        assert method["level_a"]["online_t_mAP50"] == 1.0
        assert method["level_b"]["observation_coverage"] == 1.0
        assert method["level_b"]["normalized_id_switch_rate"] == 0.0


def test_scene_bootstrap_is_deterministic_and_sparse_gap_gate_is_inconclusive() -> None:
    scenes = []
    for scene_index, delta in enumerate((0.2, 0.1, -0.1)):
        scenes.append(
            {
                "scene_id": f"scene_{scene_index}",
                "methods": {
                    code: {
                        "method_name": code,
                        "level_a": {"online_t_mAP": 0.5},
                        "level_b": {
                            "observation_coverage": 0.8,
                            "gap_opportunities": 1,
                            "gap_recovery_recall": 0.4
                            + (delta if code == "B4" else 0.0),
                            "normalized_id_switch_rate": 0.1,
                        },
                    }
                    for code in ("B1", "B2", "B3", "B4")
                },
            }
        )

    first = aggregate_rescan_results(scenes, bootstrap_replicates=100, seed=45)
    second = aggregate_rescan_results(scenes, bootstrap_replicates=100, seed=45)
    gate = derive_external_gate(first, minimum_gap_opportunities=10)

    assert first == second
    assert first["population"]["scene_count"] == 3
    assert gate["classification"] == "EXTERNAL_INCONCLUSIVE"
    assert gate["observed_gap_opportunities"] == 3


def test_efficient_metrics_match_shared_reference_semantics() -> None:
    from scripts.p6a_metrics import (
        IdentityAccumulator,
        compute_endpoint_metrics,
        compute_raw_local_metrics,
    )

    stage_masks = torch.tensor(
        [[True, True, False, False], [False, False, True, False]]
    ).transpose(0, 1)
    prediction = {
        "stage": 0,
        "pred_masks": stage_masks,
        "pred_classes": torch.tensor([2, 4]),
        "pred_scores": torch.tensor([0.9, 0.6]),
        "track_ids": torch.tensor([10, 20]),
    }
    target = {
        "masks": torch.tensor([[True, True, False, False]]),
        "labels": torch.tensor([2]),
    }
    assert rescan_evaluator._efficient_raw_local_metrics(
        [prediction], [target]
    ) == pytest.approx(compute_raw_local_metrics([prediction], [target]))

    accumulator = IdentityAccumulator()
    accumulator.add_stage(prediction)
    temporal_target = {
        "masks": target["masks"],
        "labels": target["labels"],
        "ids": torch.tensor([4]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(4, dtype=torch.long),
    }
    assert rescan_evaluator._efficient_endpoint_metrics(
        accumulator, temporal_target, endpoint=0
    ) == pytest.approx(
        compute_endpoint_metrics(accumulator, temporal_target, endpoint=0)
    )
