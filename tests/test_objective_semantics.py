from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from trainer.trainer import _configured_objective_loss, aggregate_objective_loss
from utils.rescene_objective_audit import (
    compare_gradients,
    eos_gradient_gate,
    objective_contribution_rows,
    optimized_objective,
)


def _losses() -> dict[str, torch.Tensor]:
    return {
        "loss_ce": torch.tensor(1.0),
        "loss_mask": torch.tensor(2.0),
        "loss_dice": torch.tensor(3.0),
        "loss_aux_contrastive": torch.tensor(4.0),
        "loss_aux_contrastive_layer_0": torch.tensor(5.0),
    }


def test_objective_rows_expose_upstream_duplication_and_local_weights() -> None:
    rows = objective_contribution_rows(
        _losses(), {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 2.0}
    )
    by_key = {row["loss_key"]: row for row in rows}

    assert by_key["loss_ce"]["upstream_multiplier"] == 1.0
    assert by_key["loss_ce"]["local_weighted_multiplier"] == 2.0
    assert by_key["loss_mask"]["local_weighted_contribution"] == 10.0
    assert by_key["loss_aux_contrastive"]["classification"] == "aggregate"
    assert by_key["loss_aux_contrastive_layer_0"]["classification"] == "diagnostic"
    assert by_key["loss_aux_contrastive_layer_0"]["upstream_included"] is True
    assert by_key["loss_aux_contrastive_layer_0"]["local_weighted_included"] is False


def test_raw_sum_is_exact_released_code_objective() -> None:
    losses = _losses()
    weights = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 2.0}

    assert optimized_objective(losses, weights, mode="raw_sum").item() == 15.0
    assert optimized_objective(losses, weights, mode="weighted").item() == 22.0
    assert aggregate_objective_loss(losses, weights).item() == 22.0


def test_trainer_rootcause_mode_isolated_from_legacy_flags() -> None:
    losses = _losses()
    weights = {"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 2.0}
    raw_module = SimpleNamespace(
        config=OmegaConf.create(
            {"general": {"rootcause_objective_mode": "raw_sum"}}
        ),
        criterion=SimpleNamespace(weight_dict=weights),
    )
    weighted_module = SimpleNamespace(
        config=OmegaConf.create(
            {"general": {"rootcause_objective_mode": "weighted"}}
        ),
        criterion=SimpleNamespace(weight_dict=weights),
    )
    legacy_module = SimpleNamespace(
        config=OmegaConf.create(
            {"general": {"p2_weighted_objective": True}}
        ),
        criterion=SimpleNamespace(weight_dict=weights),
    )

    assert _configured_objective_loss(raw_module, losses).item() == 15.0
    assert _configured_objective_loss(weighted_module, losses).item() == 22.0
    assert _configured_objective_loss(legacy_module, losses).item() == 22.0


def test_gradient_comparison_and_eos_gate_are_preregistered() -> None:
    aligned = compare_gradients(torch.tensor([1.0, 0.0]), torch.tensor([1.05, 0.0]))
    rotated = compare_gradients(torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.3]))

    assert aligned["cosine"] == pytest.approx(1.0)
    assert aligned["relative_norm_difference"] == pytest.approx(0.05)
    assert eos_gradient_gate(aligned)["authorized"] is False
    assert eos_gradient_gate(rotated)["authorized"] is True
    assert eos_gradient_gate(rotated)["thresholds"] == {
        "minimum_cosine": 0.98,
        "maximum_relative_norm_difference": 0.1,
    }


def test_gradient_comparison_rejects_nonfinite_or_misaligned() -> None:
    with pytest.raises(ValueError, match="aligned"):
        compare_gradients(torch.ones(2), torch.ones(3))
    with pytest.raises(ValueError, match="finite"):
        compare_gradients(torch.tensor([float("nan")]), torch.ones(1))
