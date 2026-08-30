from __future__ import annotations

import pytest
import torch
from torch import nn

from utils.rescene_runtime_audit import (
    disable_drop_path,
    encoder_stochasticity_gate,
    feature_repetition_statistics,
    stochastic_module_inventory,
)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob


def test_repetition_statistics_require_eight_aligned_finite_passes() -> None:
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    repeated = [base.clone() for _ in range(8)]

    stats = feature_repetition_statistics(repeated)

    assert stats["pass_count"] == 8
    assert stats["mean_cosine"] == pytest.approx(1.0)
    assert stats["relative_rms_deviation"] == 0.0
    assert stats["mean_feature_variance"] == 0.0
    assert encoder_stochasticity_gate({"level0": stats})["authorized"] is False
    with pytest.raises(ValueError, match="eight"):
        feature_repetition_statistics(repeated[:7])


def test_stochasticity_gate_uses_registered_or_threshold() -> None:
    base = torch.ones(4)
    repeated = [base.clone() for _ in range(8)]
    repeated[-1][0] = 0.5
    stats = feature_repetition_statistics(repeated)

    gate = encoder_stochasticity_gate({"decoder_level": stats})

    assert gate["authorized"] is True
    assert gate["minimum_cosine"] == 0.999
    assert gate["maximum_relative_rms_deviation"] == 0.001


def test_drop_path_disable_is_narrow_and_restores_state() -> None:
    model = nn.Sequential(DropPath(0.2), nn.Dropout(0.4), nn.BatchNorm1d(2))
    model.train()

    inventory = stochastic_module_inventory(model)
    assert {row["kind"] for row in inventory} == {
        "drop_path",
        "dropout",
        "normalization",
    }
    with disable_drop_path(model) as changed:
        assert changed == ["0"]
        assert model[0].drop_prob == 0.0
        assert model[1].p == 0.4
        assert model[1].training is True
        assert model[2].training is True
    assert model[0].drop_prob == 0.2
