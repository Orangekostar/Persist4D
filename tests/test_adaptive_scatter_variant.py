from __future__ import annotations

import torch

from models.scatter import AdaptiveScatter
from utils.rescene_strong_local import (
    materialize_strong_config,
    validate_strong_variant_isolation,
)


def test_adaptive_scatter_emits_group_rows_and_backpropagates() -> None:
    torch.manual_seed(0)
    scatter = AdaptiveScatter(scatter_type="adaptive", feat_dim=3)
    features = torch.tensor(
        [
            [0.2, 0.5, 0.8],
            [1.0, 1.3, 1.7],
            [-0.4, 0.7, 1.2],
            [1.5, -0.6, 0.9],
            [0.3, 1.1, -0.8],
        ],
        requires_grad=True,
    )
    group_index = torch.tensor([0, 0, 1, 1, 2])

    output = scatter(features, group_index)

    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    adaptive_parameters = list(scatter.parameters())
    assert adaptive_parameters
    assert all(parameter.grad is not None for parameter in adaptive_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in adaptive_parameters
    )
    assert any(
        torch.count_nonzero(parameter.grad) > 0
        for parameter in adaptive_parameters
    )


def test_a2_isolation_changes_only_scatter_switch() -> None:
    base = {
        "general": {
            "project_name": "base",
            "experiment_name": "base",
            "save_dir": "external:base",
        },
        "model": {"use_np_features": False, "scatter_type": "mean"},
    }
    observed = materialize_strong_config(
        base,
        variant="A2",
        output="external:strong/A2",
    )

    isolation = validate_strong_variant_isolation(
        base, observed, variant="A2"
    )
    assert isolation["changed_paths"] == ["model.scatter_type"]
    assert observed["model"]["scatter_type"] == "adaptive"
    assert observed["model"]["use_np_features"] is False
