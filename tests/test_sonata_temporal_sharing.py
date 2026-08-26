from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.pointcept import PointceptBackbone
from models.rescene import ReScene
from scripts.sonata_second_smoke import validate_temporal_batch_contract


class _Point(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def test_temporal_overlay_serializes_by_true_scene_and_restores_stage_batches() -> None:
    observed: list[torch.Tensor] = []

    def serialization(*, order: object, shuffle_orders: bool) -> None:
        del order, shuffle_orders
        observed.append(point.batch.clone())

    point = _Point(
        batch=torch.tensor([0, 1, 2, 3]),
        coord=torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 1],
            ]
        ),
        serialization=serialization,
    )
    owner = SimpleNamespace(
        model=SimpleNamespace(order=("z",), shuffle_orders=False)
    )

    result = PointceptBackbone.temporal_overlay(owner, point)

    assert result is point
    assert len(observed) == 1
    assert torch.equal(observed[0], torch.tensor([0, 0, 1, 1]))
    assert torch.equal(point.batch, torch.tensor([0, 1, 2, 3]))


def test_temporal_masking_executes_temporal_pool_merge() -> None:
    calls = {"temporal_pool_merge": 0}

    class Sparse:
        def __init__(self, features: torch.Tensor):
            self.F = features
            self.decomposed_features = [features]

    class Backbone:
        @staticmethod
        def sparse_from_sample(features: torch.Tensor, _sample: object) -> Sparse:
            return Sparse(features)

        @staticmethod
        def temporal_pool_merge(sparse: Sparse) -> Sparse:
            calls["temporal_pool_merge"] += 1
            return sparse

    owner = SimpleNamespace(backbone=Backbone(), temporal_masking=True)
    logits = torch.tensor([[[2.0], [-2.0]]])
    padding = torch.tensor([[False, False]])
    point2segment = [torch.tensor([0, 1])]

    result = ReScene.attn_mask(
        owner,
        logits,
        padding,
        sparse_coords=object(),
        point2segment=point2segment,
    )

    assert calls["temporal_pool_merge"] == 1
    assert len(result) == 1
    assert result[0].dtype == torch.bool


def test_real_batch_temporal_contract_rejects_future_or_nonfinite_features() -> None:
    data = SimpleNamespace(
        features=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.2, 0.3, 0.4, 0.0, 1.0, 0.0],
            ]
        ),
        temporal_stages=(torch.tensor([0, 1]),),
        coordinates=torch.tensor([[0, 0, 0, 0, 0], [0, 1, 0, 0, 1]]),
    )
    targets = [
        {
            "point2segment": torch.tensor([0, 1]),
            "temporal_stages": torch.tensor([0, 1]),
        }
    ]

    contract = validate_temporal_batch_contract(
        data,
        targets,
        names=["scene0112_00-scene0112_01"],
        expected_stage_counts=[2],
    )

    assert contract["feature_dimension"] == 9
    assert contract["normal_dimension"] == 3
    assert contract["stage_values"] == [[0, 1]]
    assert contract["future_stage_leakage"] is False

    targets[0]["temporal_stages"] = torch.tensor([0, 2])
    with pytest.raises(ValueError, match="future stage"):
        validate_temporal_batch_contract(
            data,
            targets,
            names=["scene0112_00-scene0112_01"],
            expected_stage_counts=[2],
        )

    targets[0]["temporal_stages"] = torch.tensor([0, 1])
    data.features[0, 7] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        validate_temporal_batch_contract(
            data,
            targets,
            names=["scene0112_00-scene0112_01"],
            expected_stage_counts=[2],
        )
