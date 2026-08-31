from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import models.rescene as rescene_module
from models.rescene import ReScene
from utils.rescene_strong_local import (
    materialize_strong_config,
    validate_strong_variant_isolation,
)


class _IdentityPositionEncoding(nn.Module):
    def forward(
        self, coordinates: torch.Tensor, input_range: object = None
    ) -> torch.Tensor:
        del input_range
        return coordinates.permute(0, 2, 1)


def _minimal_model(*, use_np_features: bool) -> ReScene:
    model = ReScene.__new__(ReScene)
    nn.Module.__init__(model)
    model.non_parametric_queries = True
    model.num_queries = 2
    model.mask_dim = 3
    model.use_np_features = use_np_features
    model.pos_enc = _IdentityPositionEncoding()
    model.query_projection = nn.Identity()
    model.np_feature_projection = nn.Identity()
    return model


def _pcd_features() -> tuple[SimpleNamespace, list[list[torch.Tensor]]]:
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
    )
    features = torch.tensor(
        [
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [30.0, 31.0, 32.0],
            [40.0, 41.0, 42.0],
            [50.0, 51.0, 52.0],
        ]
    )
    pcd_features = SimpleNamespace(
        decomposed_coordinates=[coords], decomposed_features=[features]
    )
    return pcd_features, [[coords]]


def test_np_feature_switch_preserves_positions_and_selects_fps_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_fps(points: torch.Tensor, sample_count: int) -> torch.Tensor:
        assert points.shape == (1, 5, 3)
        assert sample_count == 2
        return torch.tensor([[3, 0]], dtype=torch.long, device=points.device)

    monkeypatch.setattr(rescene_module, "furthest_point_sample", fixed_fps)
    pcd_features, coords = _pcd_features()
    disabled = _minimal_model(use_np_features=False)
    enabled = _minimal_model(use_np_features=True)

    disabled_queries, disabled_pos, disabled_coords = disabled.initialize_queries(
        pcd_features, coords
    )
    enabled_queries, enabled_pos, enabled_coords = enabled.initialize_queries(
        pcd_features, coords
    )

    expected_indices = torch.tensor([3, 0])
    expected_coords = coords[-1][0][expected_indices].unsqueeze(0)
    expected_features = pcd_features.decomposed_features[0][
        expected_indices
    ].unsqueeze(0)
    for tensor in (disabled_queries, enabled_queries, disabled_pos, enabled_pos):
        assert tensor.shape == (1, 2, 3)
    assert torch.count_nonzero(disabled_queries) == 0
    torch.testing.assert_close(enabled_queries, expected_features)
    torch.testing.assert_close(disabled_coords, expected_coords)
    torch.testing.assert_close(enabled_coords, expected_coords)
    torch.testing.assert_close(disabled_pos, enabled_pos)


def test_a1_isolation_changes_only_np_feature_switch() -> None:
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
        variant="A1",
        output="external:strong/A1",
    )

    isolation = validate_strong_variant_isolation(
        base, observed, variant="A1"
    )
    assert isolation["changed_paths"] == ["model.use_np_features"]
    assert observed["model"]["use_np_features"] is True
    assert observed["model"]["scatter_type"] == "mean"
