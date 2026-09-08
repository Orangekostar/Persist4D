from pathlib import Path

import pytest
import torch

from scripts.analyze_persist4d_allt import (
    AllTBaselineAccumulator,
    AllTBaselineError,
    plan_missing_cache_keys,
    validate_legacy_regression,
)
from scripts.system_comparison_metrics import validate_causal_prefix_pair


def _perfect_two_stage_pair():
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
            "pred_classes": torch.tensor([3, 4]),
        },
        target={
            "masks": torch.tensor(
                [
                    [True, True, False, False],
                    [False, False, True, True],
                ]
            ),
            "labels": torch.tensor([3, 4]),
            "ids": torch.tensor([101, 202]),
            "changes": torch.tensor([0, 0]),
            "temporal_stages": torch.tensor([0, 0, 1, 1]),
        },
        horizon=2,
        observed_scan_ids=("scene0000_00", "scene0000_01"),
    )


def test_allt_accumulator_uses_legacy_evaluator_on_the_entire_prefix() -> None:
    accumulator = AllTBaselineAccumulator(
        dataset_spec=Path("data/processed/rio/rio.yaml"), min_region_size=1
    )
    accumulator.update(_perfect_two_stage_pair())

    values = accumulator.compute()

    assert values["t_mAP"] == pytest.approx(1.0)
    assert values["t_mAP50"] == pytest.approx(1.0)
    assert values["t_mAP25"] == pytest.approx(1.0)
    assert values["t_REC"] == pytest.approx(1.0)
    assert values["prefix_overall_mAP"] == pytest.approx(1.0)
    assert values["local_current_AP"] == pytest.approx(1.0)


def test_legacy_regression_checks_only_preexisting_horizons() -> None:
    old = [
        {
            "method": "B4",
            "score_reducer": "mean",
            "order_id": "all",
            "horizon": "2",
            "causal_prefix_t_mAP": "0.21",
            "causal_prefix_t_mAP50": "0.31",
            "causal_prefix_t_mAP25": "0.41",
            "causal_prefix_t_REC": "0.51",
            "current_stage_AP": "0.61",
        }
    ]
    new = [
        {
            "method": "B4",
            "reducer": "mean",
            "T": 2,
            "t_mAP": 0.21,
            "t_mAP50": 0.31,
            "t_mAP25": 0.41,
            "t_REC": 0.51,
            "local_current_AP": 0.61,
        },
        {
            "method": "B4",
            "reducer": "mean",
            "T": 3,
            "t_mAP": 0.2,
            "t_mAP50": 0.3,
            "t_mAP25": 0.4,
            "t_REC": 0.5,
            "local_current_AP": 0.6,
        },
    ]

    assert validate_legacy_regression(new, old) == {
        "checked_cells": 1,
        "checked_values": 5,
        "status": "pass",
    }

    drifted = [dict(row) for row in new]
    drifted[0]["t_mAP"] = 0.2100001
    with pytest.raises(AllTBaselineError, match="regression"):
        validate_legacy_regression(drifted, old)


def test_missing_cache_plan_never_requests_existing_keys() -> None:
    expected = [("m0", "canonical", stage) for stage in range(1, 6)]
    observed = expected[:2] + expected[3:]

    assert plan_missing_cache_keys(expected, observed) == [
        ("m0", "canonical", 3)
    ]

    with pytest.raises(AllTBaselineError, match="unexpected"):
        plan_missing_cache_keys(expected, [*observed, ("other", "canonical", 1)])
