from __future__ import annotations

import copy

import pytest

from scripts.finalize_rescene_rootcause_full_evaluation import classify_full_result
from utils.rescene_rootcause_evaluation import RootCauseEvaluationError


def _rows(*, spatial_gain: float, tmap_gain: float):
    baseline = []
    candidate = []
    for seed, offset in zip((45, 46, 47), (0.0, 0.001, -0.001)):
        base = {
            "seed": seed,
            "t_mAP": 0.28 + offset,
            "t_mAP50": 0.45 + offset,
            "t_mAP25": 0.59 + offset,
            "overall_mAP": 0.37 + offset,
            "stage1_mAP": 0.42 + offset,
            "stage2_mAP": 0.43 + offset,
        }
        baseline.append(base)
        candidate.append(
            {
                **base,
                "t_mAP": base["t_mAP"] + tmap_gain,
                "t_mAP50": base["t_mAP50"] + tmap_gain,
                "t_mAP25": base["t_mAP25"] + tmap_gain,
                "overall_mAP": base["overall_mAP"] + spatial_gain,
                "stage1_mAP": base["stage1_mAP"] + spatial_gain,
                "stage2_mAP": base["stage2_mAP"] + spatial_gain,
            }
        )
    return candidate, baseline


def test_full_verdict_confirms_stable_all_metric_improvement() -> None:
    candidate, baseline = _rows(spatial_gain=0.02, tmap_gain=0.01)

    result = classify_full_result(candidate, baseline)

    assert result["verdict"] == "ROOTCAUSE-CONFIRMED"
    assert result["gates"]["paired_spatial_positive_all_seeds"] is True
    assert result["gates"]["stage1_mean_improved"] is True
    assert result["gates"]["stage2_mean_improved"] is True
    assert result["gates"]["overall_mean_improved"] is True
    assert result["gates"]["t_mAP_mean_improved"] is True


def test_full_verdict_labels_strong_local_without_rootcause_claim() -> None:
    candidate, baseline = _rows(spatial_gain=0.02, tmap_gain=0.01)

    result = classify_full_result(candidate, baseline, verdict_prefix="STRONG-LOCAL")

    assert result["verdict"] == "STRONG-LOCAL-CONFIRMED"
    assert result["verdict_prefix"] == "STRONG-LOCAL"


def test_full_verdict_is_partial_for_material_spatial_but_not_tmap_gain() -> None:
    candidate, baseline = _rows(spatial_gain=0.015, tmap_gain=-0.01)

    result = classify_full_result(candidate, baseline)

    assert result["verdict"] == "ROOTCAUSE-PARTIAL"
    assert result["paired_spatial_delta_mean"] == pytest.approx(0.015)
    assert result["gates"]["material_spatial_gain"] is True
    assert result["gates"]["t_mAP_mean_improved"] is False


def test_full_verdict_rejects_disappeared_or_unpaired_effect() -> None:
    candidate, baseline = _rows(spatial_gain=0.0, tmap_gain=0.01)
    result = classify_full_result(candidate, baseline)
    assert result["verdict"] == "ROOTCAUSE-NOT-CONFIRMED"

    malformed = copy.deepcopy(candidate)
    malformed[-1]["seed"] = 46
    with pytest.raises(RootCauseEvaluationError, match="seed matrix"):
        classify_full_result(malformed, baseline)
