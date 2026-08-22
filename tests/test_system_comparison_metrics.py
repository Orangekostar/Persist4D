from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from scripts.system_comparison_metrics import (
    CausalPrefixPair,
    IdentityAssignmentUpdate,
    SystemMetricError,
    compute_causal_task_metrics,
    compute_deployment_identity_metrics,
    current_stage_pair,
    match_identity_update,
    validate_causal_prefix_pair,
)


def _causal_pair() -> CausalPrefixPair:
    return validate_causal_prefix_pair(
        prediction={
            "pred_masks": torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                ]
            ),
            "pred_scores": torch.tensor([0.9, 0.8]),
            "pred_classes": torch.tensor([10, 11]),
        },
        target={
            "masks": torch.tensor(
                [
                    [True, True, False, False],
                    [False, False, True, True],
                ]
            ),
            "labels": torch.tensor([10, 11]),
            "ids": torch.tensor([101, 202]),
            "changes": torch.tensor([0, 0]),
            "temporal_stages": torch.tensor([0, 1, 1, 1]),
        },
        horizon=2,
        observed_scan_ids=("scene0000_00", "scene0000_01"),
    )


def test_causal_prefix_pair_rejects_future_stage_and_scan_mismatch() -> None:
    pair = _causal_pair()
    future_target = dict(pair.target)
    future_target["temporal_stages"] = torch.tensor([0, 1, 1, 2])
    with pytest.raises(SystemMetricError, match="future|temporal"):
        validate_causal_prefix_pair(
            prediction=pair.prediction,
            target=future_target,
            horizon=2,
            observed_scan_ids=pair.observed_scan_ids,
        )

    with pytest.raises(SystemMetricError, match="scan IDs|horizon"):
        validate_causal_prefix_pair(
            prediction=pair.prediction,
            target=pair.target,
            horizon=2,
            observed_scan_ids=(*pair.observed_scan_ids, "scene0000_02"),
        )


def test_current_stage_pair_contains_only_latest_points_and_present_gt() -> None:
    pair = current_stage_pair(_causal_pair())

    assert pair.horizon == 1
    assert pair.observed_scan_ids == ("scene0000_01",)
    assert pair.prediction["pred_masks"].shape == (3, 2)
    assert pair.target["masks"].shape == (2, 3)
    assert pair.target["temporal_stages"].tolist() == [0, 0, 0]
    assert pair.target["ids"].tolist() == [101, 202]


class _FakeMetric:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.updates = []

    def update(self, prediction, target) -> None:
        self.updates.append((prediction, target))

    def compute(self):
        assert len(self.updates) == 1
        if self.mode == "strict_online":
            return {
                "online_t-mAP": 0.1,
                "online_t-mAP50": 0.2,
                "online_t-mAP25": 0.3,
                "online_t-REC": 0.4,
                "online_t-REC50": 0.5,
                "online_t-REC25": 0.6,
            }
        return {
            "raw_local_AP": 0.7,
            "raw_local_AP50": 0.8,
            "raw_local_AP25": 0.9,
            "raw_local_REC": 0.65,
            "raw_local_REC50": 0.75,
            "raw_local_REC25": 0.85,
        }


def test_task_metric_names_separate_prefix_quality_from_current_stage_ap() -> None:
    result = compute_causal_task_metrics(
        [_causal_pair()], metric_factory=_FakeMetric
    )

    assert result == {
        "causal_prefix_t_mAP": 0.1,
        "causal_prefix_t_mAP50": 0.2,
        "causal_prefix_t_mAP25": 0.3,
        "causal_prefix_t_REC": 0.4,
        "causal_prefix_t_REC50": 0.5,
        "causal_prefix_t_REC25": 0.6,
        "current_stage_AP": 0.7,
        "current_stage_AP50": 0.8,
        "current_stage_AP25": 0.9,
        "current_stage_REC": 0.65,
    }


def test_match_identity_update_uses_class_compatible_global_assignment() -> None:
    update = match_identity_update(
        horizon=2,
        gt_ids=torch.tensor([1, 2]),
        gt_classes=torch.tensor([10, 11]),
        gt_masks=torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        ),
        issued_ids=torch.tensor([100, 200]),
        pred_classes=torch.tensor([11, 10]),
        pred_masks=torch.tensor(
            [
                [False, True],
                [False, True],
                [True, False],
                [True, False],
            ]
        ),
        minimum_iou=0.5,
    )

    assert update.visible_gt_ids == (1, 2)
    assert update.assignments == {1: 200, 2: 100}


def test_deployment_identity_toy_case_counts_all_denominators() -> None:
    updates = [
        IdentityAssignmentUpdate(1, (1, 2), {1: 10, 2: 20}),
        IdentityAssignmentUpdate(2, (1, 2), {1: 10, 2: 21}),
        IdentityAssignmentUpdate(3, (1,), {1: 20}),
        IdentityAssignmentUpdate(4, (2,), {2: 20}),
        IdentityAssignmentUpdate(5, (1, 2), {1: 20, 2: 21}),
    ]
    result = compute_deployment_identity_metrics(updates)

    assert result["deployment_id_switches"] == 3
    assert result["identity_transition_opportunities"] == 4
    assert result["normalized_id_switch_rate"] == pytest.approx(0.75)
    assert result["fragmentation_count"] == 2
    assert result["fragmentation_opportunities"] == 6
    assert result["fragmentation_rate"] == pytest.approx(1 / 3)
    assert result["merge_count"] == 1
    assert result["merge_opportunities"] == 5
    assert result["merge_rate"] == pytest.approx(0.2)
    assert result["gap_opportunities"] == 2
    assert result["recovery_attempts"] == 2
    assert result["correct_recoveries"] == 1
    assert result["gap_recovery_accuracy"] == pytest.approx(0.5)
    assert result["gap_recovery_recall"] == pytest.approx(0.5)


def test_gap_recovery_recall_counts_unmatched_reappearance() -> None:
    result = compute_deployment_identity_metrics(
        [
            IdentityAssignmentUpdate(1, (7,), {7: 70}),
            IdentityAssignmentUpdate(2, (), {}),
            IdentityAssignmentUpdate(3, (7,), {}),
        ]
    )

    assert result["gap_opportunities"] == 1
    assert result["recovery_attempts"] == 0
    assert result["correct_recoveries"] == 0
    assert result["gap_recovery_accuracy"] is None
    assert result["gap_recovery_recall"] == 0.0


def test_identity_update_rejects_duplicate_issued_id_within_update() -> None:
    update = IdentityAssignmentUpdate(2, (1, 2), {1: 10, 2: 20})
    with pytest.raises(SystemMetricError, match="issued ID"):
        replace(update, assignments={1: 10, 2: 10})
