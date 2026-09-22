from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from models.criterion import SetCriterion, dice_loss, sigmoid_ce_loss
from models.perception_gain import (
    SemanticQueryScorer,
    derive_segment_stage_ids,
    select_semantic_query_indices,
    stage_aware_mask_losses,
)
from models.rescene import ReScene


def test_segment_stage_ids_are_integer_complete_and_never_mix_stages() -> None:
    assert derive_segment_stage_ids(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2, 2, 5, 5]),
    ).tolist() == [2, 5]

    with pytest.raises(ValueError, match="spans multiple stages"):
        derive_segment_stage_ids(
            torch.tensor([0, 0]),
            torch.tensor([1, 2]),
        )
    with pytest.raises(ValueError, match="contiguous"):
        derive_segment_stage_ids(
            torch.tensor([0, 2]),
            torch.tensor([1, 1]),
        )


def test_stage_mask_losses_preserve_single_stage_and_penalize_ghost_support() -> None:
    logits = torch.tensor([[2.0, -1.0, 3.0, 3.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    stages = torch.tensor([0, 0, 1, 1])

    balanced_mask, balanced_dice, debug = stage_aware_mask_losses(
        logits, target, stages, mode="balanced"
    )
    worst_mask, worst_dice, _ = stage_aware_mask_losses(
        logits, target, stages, mode="worst"
    )
    no_ghost_mask, no_ghost_dice, _ = stage_aware_mask_losses(
        torch.tensor([[2.0, -1.0, -3.0, -3.0]]),
        target,
        stages,
        mode="balanced",
    )
    single_mask, single_dice, _ = stage_aware_mask_losses(
        logits[:, :2], target[:, :2], torch.zeros(2, dtype=torch.long), mode="worst"
    )

    assert balanced_mask > no_ghost_mask
    assert balanced_dice > no_ghost_dice
    assert worst_mask >= balanced_mask
    assert worst_dice >= balanced_dice
    assert single_mask.detach().item() == pytest.approx(
        sigmoid_ce_loss(logits[:, :2], target[:, :2], 1).item()
    )
    assert single_dice.detach().item() == pytest.approx(
        dice_loss(logits[:, :2], target[:, :2], 1).item()
    )
    assert debug == {
        "matched_instances": 1,
        "valid_stages": 2,
        "ghost_instance_stages": 1,
        "sampled_support": 4,
    }


class _FixedMatcher:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty

    def __call__(self, outputs, targets, mask_type):
        assert mask_type == "segment_mask"
        if self.empty:
            empty = torch.empty(0, dtype=torch.long)
            return [(empty, empty) for _ in targets]
        return [
            (torch.arange(len(target["labels"])), torch.arange(len(target["labels"])))
            for target in targets
        ]


def _criterion(*, mode: str, empty: bool = False) -> SetCriterion:
    return SetCriterion(
        num_classes=3,
        matcher=_FixedMatcher(empty=empty),
        weight_dict={"loss_mask": 1.0, "loss_dice": 1.0},
        eos_coef=0.1,
        losses=["masks"],
        num_points=-1,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        class_weights=-1,
        num_changes=1,
        change_weights=torch.ones(1),
        mask_loss_mode=mode,
    )


def test_stage_loss_runs_on_aux_outputs_and_empty_matches_stay_connected() -> None:
    prediction = torch.tensor([[2.0], [-1.0], [3.0], [3.0]], requires_grad=True)
    aux_prediction = prediction.detach().clone().requires_grad_(True)
    outputs = {
        "pred_logits": torch.zeros((1, 1, 3)),
        "pred_masks": [prediction],
        "aux_outputs": [
            {
                "pred_logits": torch.zeros((1, 1, 3)),
                "pred_masks": [aux_prediction],
            }
        ],
    }
    targets = [
        {
            "labels": torch.tensor([1]),
            "segment_mask": torch.tensor([[True, False, False, False]]),
            "point2segment": torch.arange(4),
            "temporal_stages": torch.tensor([0, 0, 1, 1]),
        }
    ]

    legacy = _criterion(mode="legacy")(
        {"pred_logits": outputs["pred_logits"], "pred_masks": [prediction]},
        targets,
        "segment_mask",
    )
    losses = _criterion(mode="balanced")(outputs, targets, "segment_mask")

    expected_target = targets[0]["segment_mask"].float()
    assert legacy["loss_mask"].detach().item() == pytest.approx(
        sigmoid_ce_loss(prediction.T, expected_target, 1).item()
    )
    assert legacy["loss_dice"].detach().item() == pytest.approx(
        dice_loss(prediction.T, expected_target, 1).item()
    )
    assert set(losses) == {"loss_mask", "loss_dice", "loss_mask_0", "loss_dice_0"}
    assert losses["loss_mask"] == losses["loss_mask_0"]
    assert losses["loss_dice"] == losses["loss_dice_0"]

    empty_losses = _criterion(mode="worst", empty=True)(
        {"pred_logits": outputs["pred_logits"], "pred_masks": [prediction]},
        targets,
        "segment_mask",
    )
    total = sum(empty_losses.values())
    total.backward()
    assert math.isfinite(total.item()) and total.item() == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad).item() == 0


class _FeatureScore(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]


def test_semantic_query_selector_is_stage_balanced_stable_and_repeats_shortages() -> (
    None
):
    scorer = SemanticQueryScorer(feature_dim=128)
    assert [type(layer) for layer in scorer.network] == [
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.GELU,
        torch.nn.Linear,
    ]
    assert scorer.network[1].in_features == 128
    assert scorer.network[1].out_features == 64
    features = torch.tensor(
        [[0.9, 0.0], [0.8, 0.0], [0.7, 0.0], [0.6, 0.0], [0.5, 0.0], [0.4, 0.0]],
        requires_grad=True,
    )
    coordinates = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0, 2.0],
        ]
    )
    stages = torch.tensor([0, 0, 1, 1, 2, 2])

    indices, debug = select_semantic_query_indices(
        features,
        coordinates,
        stages,
        stable_keys=("a", "b", "c", "d", "e", "f"),
        scorer=_FeatureScore(),
        num_queries=8,
    )

    assert indices.tolist() == [0, 1, 0, 2, 3, 2, 4, 5]
    assert debug["stage_quotas"] == {0: 3, 1: 3, 2: 2}
    assert debug["repeat_count"] == 2
    assert indices.requires_grad is False

    model = ReScene.__new__(ReScene)
    torch.nn.Module.__init__(model)
    model.semantic_query_positioning = True
    model.semantic_query_fraction = 0.75
    model.semantic_query_scorer = _FeatureScore()
    model.num_queries = 8
    model.mask_dim = 4
    model.pos_enc = _CoordinateIdentity()
    model.query_projection = torch.nn.Identity()
    queries, query_pos, sampled = model.initialize_queries(
        SimpleNamespace(device=features.device),
        coords=[[coordinates]],
        semantic_features=[features],
        semantic_coordinates=[coordinates],
        semantic_stage_ids=[stages],
        semantic_stable_keys=[("a", "b", "c", "d", "e", "f")],
    )
    assert torch.count_nonzero(queries).item() == 0
    torch.testing.assert_close(sampled, coordinates[indices].unsqueeze(0))
    torch.testing.assert_close(query_pos, sampled)


def test_semantic_fps_batches_distance_queries_without_changing_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = torch.arange(60, dtype=torch.float32).unsqueeze(1)
    coordinates = torch.stack(
        (
            torch.linspace(0, 1, 60),
            torch.linspace(1, 0, 60),
            torch.arange(60, dtype=torch.float32).remainder(7) / 7,
            torch.zeros(60),
        ),
        dim=1,
    )
    stages = torch.zeros(60, dtype=torch.long)
    calls = 0
    original_cdist = torch.cdist

    def counted_cdist(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cdist(*args, **kwargs)

    monkeypatch.setattr(torch, "cdist", counted_cdist)
    indices, _ = select_semantic_query_indices(
        features,
        coordinates,
        stages,
        stable_keys=tuple(f"segment-{index:03d}" for index in range(60)),
        scorer=_FeatureScore(),
        num_queries=10,
        semantic_fraction=0.5,
    )

    assert indices.tolist() == [59, 41, 42, 46, 44, 0, 5, 16, 25, 20]
    assert calls <= 9


class _CoordinateIdentity(torch.nn.Module):
    def forward(self, coordinates: torch.Tensor, input_range=None) -> torch.Tensor:
        del input_range
        return coordinates.permute(0, 2, 1)
