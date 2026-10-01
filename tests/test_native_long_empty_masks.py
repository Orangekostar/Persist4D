import pytest
import torch
from torch.nn import functional as F

from models.criterion import (
    SetCriterion,
    dice_loss,
    infoNCE_chunked_loss,
    sigmoid_ce_loss,
)
from models.matcher import HungarianMatcher


@pytest.mark.parametrize("sample_fraction", [-1, 1.])
def test_empty_target_keeps_background_classification_and_zero_mask_gradient(sample_fraction):
    criterion = SetCriterion(
        num_classes=3, matcher=HungarianMatcher(num_points=sample_fraction),
        weight_dict={}, eos_coef=.2, losses=["labels", "masks"], num_points=sample_fraction,
        oversample_ratio=3., importance_sample_ratio=.75, class_weights=-1,
        num_changes=4, change_weights=[1.] * 4,
    )
    logits = torch.tensor([[[.2, -.3, .1], [.1, .4, -.2]]], requires_grad=True)
    masks = torch.randn(5, 2, requires_grad=True)
    auxiliary = torch.randn(5, 2, requires_grad=True)
    outputs = {"pred_logits": logits, "pred_masks": [masks],
               "aux_outputs": [{"pred_logits": logits, "pred_masks": [auxiliary]}]}
    target = {"labels": torch.empty(0, dtype=torch.long), "masks": torch.empty(0, 5)}
    losses = criterion(outputs, [target], mask_type="masks")
    expected = F.cross_entropy(logits.transpose(1, 2), torch.full((1, 2), 2), criterion.empty_weight)
    assert torch.equal(losses["loss_ce"], expected)
    assert losses["loss_ce"].item() > 0
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(value.item() == 0 for key, value in losses.items() if "mask" in key or "dice" in key)
    sum(losses.values()).backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert torch.equal(masks.grad, torch.zeros_like(masks))
    assert torch.equal(auxiliary.grad, torch.zeros_like(auxiliary))


def test_empty_sample_does_not_change_nonempty_sample_mask_objective():
    criterion = SetCriterion(
        num_classes=3, matcher=HungarianMatcher(), weight_dict={}, eos_coef=.2,
        losses=["labels", "masks"], num_points=-1, oversample_ratio=3.,
        importance_sample_ratio=.75, class_weights=-1, num_changes=4, change_weights=[1.] * 4,
    )
    empty = torch.randn(5, 2, requires_grad=True)
    occupied = torch.randn(5, 2, requires_grad=True)
    target = torch.tensor([[0., 1., 1., 0., 1.]])
    outputs = {"pred_masks": [empty, occupied]}
    targets = [{"masks": torch.empty(0, 5)}, {"masks": target}]
    indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
               (torch.tensor([1]), torch.tensor([0]))]
    result = criterion.loss_masks(outputs, targets, indices, 1., "masks")
    selected = occupied[:, 1].unsqueeze(0)
    assert torch.equal(result["loss_mask"], sigmoid_ce_loss(selected, target, 1))
    assert torch.equal(result["loss_dice"], dice_loss(selected, target, 1))
    sum(result.values()).backward()
    assert torch.equal(empty.grad, torch.zeros_like(empty))
    assert occupied.grad.abs().sum() > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_contrastive_loss_preserves_callers_tf32_policy(monkeypatch, enabled):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", enabled)
    features = torch.tensor([[1., .2], [.3, 1.], [.4, .5]], requires_grad=True)
    target = torch.tensor([[True, True, False], [False, False, True]])
    loss = infoNCE_chunked_loss(features, target, chunk_size=2, candidate_chunk_size=2)
    assert torch.backends.cuda.matmul.allow_tf32 is enabled
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(features.grad).all()
