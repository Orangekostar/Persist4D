from __future__ import annotations

import pytest
import torch

from scripts.p6a_metrics import build_online_endpoint_prediction
from scripts.reviewer_closure_decomposition import (
    OfficialTemporalCurveAccumulator,
    OfficialTemporalThresholdAccumulator,
    build_oracle_accumulator,
    classify_ceiling,
    classify_decomposition_failure,
    classify_observation_coverage,
)


def _temporal_pair(*, second_stage_support: int) -> tuple[dict, dict]:
    points_per_stage = 200
    target_mask = torch.ones((1, points_per_stage * 2), dtype=torch.bool)
    prediction_mask = torch.zeros((points_per_stage * 2, 1), dtype=torch.bool)
    prediction_mask[:points_per_stage, 0] = True
    prediction_mask[
        points_per_stage : points_per_stage + second_stage_support, 0
    ] = True
    return (
        {
            "pred_masks": prediction_mask,
            "pred_scores": torch.tensor([0.9]),
            "pred_classes": torch.tensor([3]),
        },
        {
            "masks": target_mask,
            "labels": torch.tensor([3]),
            "ids": torch.tensor([7]),
            "changes": torch.tensor([0]),
            "temporal_stages": torch.tensor(
                [0] * points_per_stage + [1] * points_per_stage
            ),
        },
    )


def _payload(stage: int, *, valid: bool = True) -> dict:
    points = 120
    class_prob = torch.zeros((1, 19), dtype=torch.float32)
    class_prob[0, 0] = 1.0
    return {
        "key": {"stage_index": stage},
        "observation": {
            "features": torch.ones((1, 4), dtype=torch.float32),
            "class_prob": class_prob,
            "confidence": torch.tensor([0.9]),
            "valid": torch.tensor([valid]),
            "masks": torch.ones((1, points), dtype=torch.bool),
            "mask_support": torch.tensor([points]),
        },
        "target": {
            "gt_ids": torch.tensor([11]),
            "gt_classes": torch.tensor([0]),
            "gt_masks": torch.ones((1, points), dtype=torch.bool),
        },
    }


def test_case_a_identity_perfect_but_high_iou_mask_fails() -> None:
    prediction, target = _temporal_pair(second_stage_support=120)
    ap50 = OfficialTemporalThresholdAccumulator(0.50)
    ap75 = OfficialTemporalThresholdAccumulator(0.75)
    ap50.update(prediction, target)
    ap75.update(prediction, target)

    assert ap50.compute() == 1.0
    assert ap75.compute() == 0.0


def test_official_curve_matches_single_threshold_semantics() -> None:
    prediction, target = _temporal_pair(second_stage_support=120)
    curve = OfficialTemporalCurveAccumulator((0.50, 0.75))
    curve.update(prediction, target)

    assert curve.compute() == {0.50: 1.0, 0.75: 0.0}


def test_case_b_oracle_recovers_fragmented_identity() -> None:
    payloads = (_payload(0), _payload(1))
    oracle = build_oracle_accumulator(
        payloads,
        sequence_id="synthetic",
        background_class=18,
    )
    prediction = build_online_endpoint_prediction(oracle, endpoint=1)

    assert prediction["track_ids"].tolist() == [11]
    metric_prediction = {
        **prediction,
        "pred_classes": torch.tensor([3]),
    }
    metric_target = {
        "masks": torch.ones((1, 240), dtype=torch.bool),
        "labels": torch.tensor([3]),
        "ids": torch.tensor([11]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.tensor([0] * 120 + [1] * 120),
    }
    metric = OfficialTemporalThresholdAccumulator(0.50)
    metric.update(metric_prediction, metric_target)
    assert metric.compute() == 1.0


def test_case_c_oracle_cannot_recover_absent_candidate() -> None:
    payloads = (_payload(0, valid=False), _payload(1, valid=False))
    oracle = build_oracle_accumulator(
        payloads,
        sequence_id="synthetic-miss",
        background_class=18,
    )
    prediction = build_online_endpoint_prediction(oracle, endpoint=1)

    assert prediction["pred_masks"].shape == (240, 0)
    categories = classify_observation_coverage(
        prediction_masks=torch.zeros((0, 120), dtype=torch.bool),
        prediction_classes=torch.empty(0, dtype=torch.long),
        valid=torch.empty(0, dtype=torch.bool),
        target_masks=torch.ones((1, 120), dtype=torch.bool),
        target_classes=torch.tensor([3]),
        threshold=0.50,
    )
    assert categories == ("no_candidate_observation",)


@pytest.mark.parametrize(
    ("event", "coverage", "expected"),
    [
        ({"failure_category": "F1"}, "no_candidate_observation", "local_observation_miss"),
        ({"failure_category": "F1"}, "wrong_class", "class_failure"),
        ({"failure_category": "F1"}, "insufficient_iou", "high_iou_mask_failure"),
        ({"failure_category": "F3"}, None, "identity_fragmentation"),
        ({"failure_category": "F4"}, None, "identity_merge"),
        ({"failure_category": "F5"}, None, "wrong_gap_recovery"),
        ({"failure_category": "F6"}, None, "class_failure"),
        ({"failure_category": "F7"}, None, "capacity_failure"),
        ({"failure_category": "F2"}, None, "unknown_unresolved"),
    ],
)
def test_failure_decomposition_keeps_an_explicit_unknown(
    event: dict, coverage: str | None, expected: str
) -> None:
    assert classify_decomposition_failure(event, coverage_category=coverage) == expected


def test_oracle_gate_requires_substantial_gain_and_gap_closure() -> None:
    assert (
        classify_ceiling(
            persistent={4: 0.10, 5: 0.08},
            full_history={4: 0.20, 5: 0.18},
            oracle={4: 0.16, 5: 0.09},
        )
        == "ASSOCIATION_CEILING"
    )
    assert (
        classify_ceiling(
            persistent={4: 0.10, 5: 0.08},
            full_history={4: 0.20, 5: 0.18},
            oracle={4: 0.13, 5: 0.10},
        )
        == "PERCEPTION_CEILING"
    )
