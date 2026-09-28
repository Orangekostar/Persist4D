"""Protocol regressions: independent heads, stage aggregation and output isolation."""

from types import SimpleNamespace

import pytest
import torch

from models.short_module_heads import (
    MaskHead,
    Module,
    QualityHead,
    apply_module,
    build_head,
    describe_candidates,
    quality_inputs,
)


@pytest.mark.parametrize("value", ["Q1+M1", ["Q1", "M1"], True, "Q4"])
def test_module_rejects_combinations(value):
    with pytest.raises((ValueError, TypeError)):
        Module(value)


def test_quality_seeded_controls_and_monotone_gradients():
    q1 = build_head("Q1", feature_dim=7, query_dim=5, classes=3,
                    probability_dim=4, thresholds=(0.5, 0.75, 0.9), seed=45)
    q2 = build_head("Q2", feature_dim=7, query_dim=5, classes=3,
                    probability_dim=4, thresholds=(0.5, 0.75, 0.9), seed=45)
    q3 = build_head("Q3", feature_dim=7, query_dim=5, classes=3,
                    probability_dim=4, thresholds=(0.5, 0.75, 0.9), seed=45)
    for left, right in zip(q1.parameters(), q2.parameters()):
        assert torch.equal(left, right)
        assert left.data_ptr() != right.data_ptr()
    for left, right in zip(q1.trunk.parameters(), q3.trunk.parameters()):
        assert torch.equal(left, right)
    x = torch.randn(4, 46)
    p = q3.probabilities(x)
    assert p.shape == (4, 3)
    assert torch.all(p[:, 1:] < p[:, :-1])
    assert torch.equal(q3(x), p.mean(-1))
    torch.nn.functional.binary_cross_entropy(p, torch.ones_like(p)).backward()
    assert torch.all(q3.output.bias.grad != 0)
    assert not torch.equal(q1(x), torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_mask_zero_init_independence_and_negative_segment_repair():
    heads = [build_head(m, feature_dim=2, query_dim=3, classes=2,
                        probability_dim=3, thresholds=(0.5,), seed=45)
             for m in ("M0", "M1", "M2")]
    for other in heads[1:]:
        for left, right in zip(heads[0].parameters(), other.parameters()):
            assert torch.equal(left, right)
            assert left.data_ptr() != right.data_ptr()
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]])
    query = torch.ones(2, 3)
    h = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.]],
                      [[2., 3., 4., 5.], [5., 4., 3., 2.]]])
    onehot = torch.eye(2)
    stages = torch.tensor([0, 0, 1, 1])
    for head in heads:
        delta = head(features, query, h, onehot, stages)
        assert torch.equal(delta, torch.zeros(4, 2))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full((4, 2), -0.1) + delta, torch.ones(4, 2))
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.1)
        loss.backward()
        assert head.output.bias.grad.abs().sum() > 0
        optimizer.step()
        repaired = head(features, query, h, onehot, stages)
        assert torch.all(repaired > 0.1)
        assert torch.all(repaired.abs() <= 2)


def test_mask_stage_difference_is_only_pooling_and_h1_equivalent():
    shared = MaskHead(feature_dim=2, query_dim=3, classes=2, per_stage=False)
    staged = MaskHead(feature_dim=2, query_dim=3, classes=2, per_stage=True)
    with torch.no_grad():
        shared.output.weight.fill_(0.1)
    staged.load_state_dict(shared.state_dict())
    features = torch.randn(4, 2)
    query = torch.randn(2, 3)
    h = torch.randn(2, 1, 4)
    classes = torch.eye(2)
    stages = torch.zeros(4, dtype=torch.long)
    assert torch.equal(shared(features, query, h, classes, stages),
                       staged(features, query, h, classes, stages))
    h2 = torch.cat([h, h + 4], dim=1)
    stages[2:] = 1
    assert not torch.equal(shared(features, query, h2, classes, stages),
                           staged(features, query, h2, classes, stages))


def test_quality_rejects_invalid_threshold_order():
    with pytest.raises(ValueError):
        QualityHead(12, thresholds=(0.75, 0.5))


def test_descriptors_are_stage_local_and_do_not_weight_by_vertices():
    features = torch.tensor([[2., 4.], [6., 8.], [10., 12.], [20., 22.]],
                            requires_grad=True)
    logits = torch.tensor([[1., -1.], [-1., -1.], [1., -1.], [-1., -1.]])
    xyz = torch.tensor([[0., 0., 0.], [9., 0., 0.], [0., 0., 0.], [9., 0., 0.]])
    stages = torch.tensor([0, 0, 1, 1])
    h, support, probability = describe_candidates(
        features, logits, xyz, stages, ("a", "b", "c", "d"), horizon=2)
    assert h.shape == (2, 2, 4)
    assert torch.equal(h[0], torch.tensor([[2., 4., 4., 6.], [10., 12., 15., 17.]]))
    assert torch.equal(h[1, :, :2], torch.zeros(2, 2))
    assert torch.equal(support, torch.tensor([[.5, .5], [0., 0.]]))
    assert torch.equal(probability[1], torch.zeros(2))
    assert not h.requires_grad
    query = torch.tensor([[1., 3.], [2., 4.]])
    x = quality_inputs(query, h, torch.tensor([[.1, .9], [.8, .2]]),
                       torch.eye(2), torch.tensor([.3, .4]), support, probability)
    assert torch.equal(x[:, -1], torch.ones(2))
    assert torch.equal(x[:, 10:14], torch.tensor([[.1, .9, 1., 0.], [.8, .2, 0., 1.]]))
    h1, s1, p1 = describe_candidates(features[:2], logits[:2], xyz[:2], stages[:2],
                                    ("a", "b"), horizon=1)
    x1 = quality_inputs(query, h1, torch.ones(2, 2), torch.eye(2),
                        torch.ones(2), s1, p1)
    assert torch.equal(x1[:, 6:10], torch.zeros(2, 4))
    assert torch.equal(x1[:, -1], torch.full((2,), .5))


def test_inference_preserves_columns_and_zero_mask_uses_real_materialization():
    from scripts.rescene_task_postprocess import (
        OfficialTaskPrediction,
        OfficialTaskSoftEvidence,
    )
    from trainer.trainer import InstanceSegmentation

    system = SimpleNamespace(eval_on_segments=True)
    system._get_full_res_mask = InstanceSegmentation._get_full_res_mask.__get__(system)
    z = torch.tensor([[1., 1.], [-1., -1.], [1., 1.]])
    inverse = torch.tensor([0, 1, 2, 2])
    full_z = z[inverse]
    full_partition = torch.tensor([0, 0, 1, 1])
    evidence = OfficialTaskSoftEvidence(
        representation="FP32_SEGMENT_LOGITS", segment_logits=z, low_resolution_logits=z,
        full_resolution_logits=full_z, full_resolution_probabilities=full_z.sigmoid(),
        class_probabilities=torch.tensor([[.2, .7, .1], [.2, .7, .1]]),
        query_features=torch.tensor([[1., -1.], [1., -1.]]),
        segment_features=torch.tensor([[1., 2.], [2., 3.], [3., 4.]]),
        source_query_ids=torch.tensor([0, 0]), source_class_ids=torch.tensor([0, 1]),
        low_point2segment=torch.arange(3), voxel_inverse=inverse,
        full_point2segment=full_partition)
    masks = torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.bool)
    stages = torch.zeros(4, dtype=torch.long)
    parent = OfficialTaskPrediction(masks.clone(), torch.tensor([.2, .7]),
                                    torch.tensor([3, 4]), evidence.source_query_ids,
                                    evidence.source_class_ids, stages, 0, masks.clone(), evidence)
    desc = {"h": torch.ones(2, 1, 4), "support": torch.ones(2, 1),
            "foreground_probability": torch.ones(2, 1),
            "segment_stages": torch.zeros(3, dtype=torch.long), "classes": 2}
    b0 = apply_module("B0", None, parent, desc, system=system)
    assert torch.equal(b0["pred_masks"], masks)
    for arm in ("Q1", "M0"):
        head = build_head(arm, feature_dim=2, query_dim=2, classes=2,
                          probability_dim=3, thresholds=(.5,), seed=45)
        result = apply_module(arm, head, parent, desc, system=system)
        assert torch.equal(result["pred_classes"], torch.tensor([3, 4]))
        assert torch.equal(result["source_query_ids"], torch.tensor([0, 0]))
        assert torch.equal(result["source_class_ids"], torch.tensor([0, 1]))
        assert torch.equal(result["pred_masks"], masks)
        if arm == "M0":
            assert torch.equal(result["pred_scores"], torch.tensor([.2, .7]))
            # Corrupt only stored bool masks; real zero-residual reconstruction
            # must still derive its result from logits and the majority partition.
            parent.pred_masks.fill_(False)
            parent.latest_stage_masks.fill_(False)
            rebuilt = apply_module(arm, head, parent, desc, system=system)
            assert torch.equal(rebuilt["pred_masks"], masks)
        else:
            assert not torch.equal(result["pred_scores"], b0["pred_scores"])
