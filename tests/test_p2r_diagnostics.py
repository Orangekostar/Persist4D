from __future__ import annotations

import pytest
import torch

from scripts.p2r_diagnostics import (
    compare_gradients,
    configured_objective,
    validate_diagnostic_report,
)


def test_configured_objective_distinguishes_weighted_and_raw_sum() -> None:
    losses = {
        "loss_ce": torch.tensor(2.0),
        "loss_mask": torch.tensor(3.0),
        "loss_segment_contrastive": torch.tensor(5.0),
        "loss_segment_contrastive_layer0": torch.tensor(7.0),
    }
    weights = {"loss_ce": 2.0, "loss_mask": 5.0}

    assert configured_objective(losses, weights, "weighted").item() == 24.0
    assert configured_objective(losses, weights, "raw_sum").item() == 17.0


def test_gradient_comparison_reports_norm_cosine_and_max_abs() -> None:
    result = compare_gradients(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))

    assert result["left_norm"] == pytest.approx(5**0.5)
    assert result["right_norm"] == pytest.approx(20**0.5)
    assert result["cosine_similarity"] == pytest.approx(1.0)
    assert result["max_abs_difference"] == pytest.approx(2.0)


def test_diagnostic_report_requires_all_controlled_paths() -> None:
    report = {
        "schema_version": 1,
        "status": "pass",
        "paths": {name: {} for name in ("P2R-0", "P2R-A", "P2R-B")},
        "comparisons": {
            "frozen_encoder_mode": {},
            "objective_semantics": {},
            "microbatch_normalization": {},
        },
    }

    assert validate_diagnostic_report(report) is report

    del report["comparisons"]["microbatch_normalization"]
    with pytest.raises(ValueError, match="comparison coverage"):
        validate_diagnostic_report(report)
