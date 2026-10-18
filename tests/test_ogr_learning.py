"""Focused OGR learning contracts; expectations use hand-computed examples."""

import torch


def test_pool_excludes_past_points_and_keeps_mask_gradient():
    from models.observation_grounded_modules import pool_observation

    features = torch.tensor([[99.], [2.], [8.], [999.]], requires_grad=True)
    masks = torch.zeros(4, 1, requires_grad=True)
    values, valid, mass = pool_observation(features, masks, torch.tensor([0., 3., 1., 0.]))
    torch.testing.assert_close(values, torch.tensor([[3.5]]))
    torch.testing.assert_close(mass, torch.tensor([2.]))
    assert valid.tolist() == [True]
    values.sum().backward()
    torch.testing.assert_close(masks.grad[:, 0], torch.tensor([0., -.5625, .5625, 0.]))
    assert features.grad is None
    values, valid, _ = pool_observation(features, masks, torch.zeros(4))
    assert not valid.any() and not values.any()


def test_identity_uses_shared_feature_domain_and_detaches_class_prior():
    from models.observation_grounded_modules import IdentityEvidenceHead

    head = IdentityEvidenceHead(dim=2)
    values = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    prototype = torch.tensor([[1., 0.], [0., 1.]])
    classes = torch.tensor([[.5, .5, 0.], [.5, .5, 0.]], requires_grad=True)
    history = torch.tensor([[.5, .5, 0.], [.5, .5, 0.]])
    evidence = head(values, torch.tensor([True, False]), prototype, classes, history)
    expected = torch.tensor([[10. + .25 * torch.log(torch.tensor(.5)),
                              .25 * torch.log(torch.tensor(.5)), 0.]])
    torch.testing.assert_close(evidence['logits'][:1], expected)
    torch.testing.assert_close(evidence['probabilities'][1], torch.tensor([0., 0., 1.]))
    evidence['probabilities'][0, 0].backward()
    assert values.grad is not None and classes.grad is None
    empty = head(values, torch.ones(2, dtype=torch.bool), prototype[:0], classes, history[:0])
    torch.testing.assert_close(empty['probabilities'], torch.ones(2, 1))


def test_feedback_zero_initialization_then_upstream_gradient_after_projection_update():
    from models.observation_grounded_modules import ObservationFeedback

    feedback = ObservationFeedback(seed=246)
    assert sum(p.numel() for p in feedback.parameters()) == 65792
    direct = torch.linspace(-2., 3., 256).reshape(2, 128).requires_grad_()
    history = torch.cos(direct.detach())
    source = torch.sin(direct.detach())
    assert torch.equal(feedback(direct, history, source), torch.zeros(2, 128))
    optimizer = torch.optim.AdamW(feedback.parameters(), lr=1e-3)
    feedback(direct, history, source).sum().backward()
    assert torch.count_nonzero(direct.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    direct.grad = None
    feedback(direct, history, source).square().sum().backward()
    assert direct.grad is not None and direct.grad.abs().sum() > 0
    assert feedback.input_layer.weight.grad.abs().sum() > 0


def test_fresh_four_arm_counts_and_empty_history_match_frozen_parent():
    from dataclasses import replace
    from types import SimpleNamespace
    from tests.test_revisit_prompt import _tiny_model, _history
    from tests.test_task_memory_routing import _meta
    from models.rescene import ReScene
    from models.observation_grounded import ObservationGroundedReScene
    from models.revisit_prompt_state import RevisitGeometryState
    from models.observation_evidence_state import ObservationEvidenceState

    torch.manual_seed(8)
    parent = _tiny_model(ReScene).eval()
    xyz = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]])
    features = torch.randn(2, 128)
    sample = SimpleNamespace(F=features, decomposed_features=[features],
                             decomposed_coordinates=[xyz])
    args = dict(point2segment=[torch.arange(2)], raw_coordinates=xyz, is_eval=True)
    meta = replace(_meta(0), local_stage_ids=torch.zeros(2, dtype=torch.long),
                   original_vertex_ids=(torch.arange(2),), scan_vertex_offsets=torch.tensor([0, 2]),
                   point2segment=torch.arange(2), segment_stage_ids=torch.zeros(2, dtype=torch.long),
                   voxel_inverse=torch.arange(2), full_resolution_point2segment=torch.arange(2))
    expected = parent(sample, **args)
    state = _history(0)
    for variant, count in [('R-TARGET', 84224), ('R-ID', 116993),
                           ('R-QFB', 182785), ('R-OBS', 182785)]:
        model = _tiny_model(ObservationGroundedReScene, ogr_variant=variant, mpr_seed=246)
        weights = model.state_dict()
        weights.update(parent.state_dict())
        model.load_state_dict(weights)
        model.train()
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == count
        assert not model.backbone.training and not model.mask_embed_head.training
        result = model(sample, **args, task_state=state,
                       geometry_state=RevisitGeometryState.empty(state), stage_meta=[meta],
                       observation_state=ObservationEvidenceState.empty(state))
        torch.testing.assert_close(result['pred_logits'], expected['pred_logits'], rtol=0, atol=0)
        torch.testing.assert_close(result['pred_masks'][0], expected['pred_masks'][0], rtol=0, atol=0)
        assert result['ogr_final'][0]['mass'].shape == (100,)
        assert not result['ogr_final'][0]['features'].requires_grad
        assert result['mpr_support_probabilities'].eq(1).all()


def _criterion():
    from models.criterion import SetCriterion
    from models.matcher import HungarianMatcher

    return SetCriterion(num_classes=3, matcher=HungarianMatcher(2., 5., 2., num_points=-1),
        weight_dict={}, eos_coef=.1, losses=['labels', 'masks'], num_points=-1,
        oversample_ratio=3., importance_sample_ratio=.75, class_weights=-1,
        num_changes=3, change_weights=[1., 1., 1.])


def _loss_example():
    logits = torch.tensor([[[4., 0., 0.], [0., 4., 0.], [1., 0., 0.], [0., 0., 3.]]], requires_grad=True)
    masks = torch.tensor([[4., -4., -1., -4.], [4., -4., -1., -4.],
                          [-4., 4., -1., -4.], [-4., 4., -1., -4.]], requires_grad=True)
    target = dict(labels=torch.tensor([0, 1]), masks=torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool))
    return dict(pred_logits=logits, pred_masks=[masks], aux_outputs=[dict(pred_logits=logits * .5, pred_masks=[masks * .5])]), [target]


def test_constrained_assignment_forces_target_once_and_removes_duplicate_from_all_losses():
    from models.observation_grounded_loss import AssignmentConstraints, ConstrainedTaskCriterion

    criterion = ConstrainedTaskCriterion.from_parent(_criterion())
    output, targets = _loss_example()
    # The actual frozen E1 disables change detection, so its key is present
    # with None, rather than absent. The reduced matcher must preserve that.
    output['pred_changes'] = None
    output['aux_outputs'][0]['pred_changes'] = None
    constraints = AssignmentConstraints(forced=((2, 0),), ignored=(0,))
    losses = criterion(output, targets, 'masks', constraints=constraints)
    assert len(criterion.assignments) == 2
    for assignment in criterion.assignments:
        assert dict(zip(*[v.tolist() for v in assignment[0]])) == {1: 1, 2: 0}
    sum(losses.values()).backward()
    assert output['pred_logits'].grad[0, 0].eq(0).all()
    assert output['pred_masks'][0].grad[:, 0].eq(0).all()
    assert output['pred_masks'][0].grad[:, 2].abs().sum() > 0


def test_unconstrained_loss_preserves_every_original_key_value_and_rng():
    from models.observation_grounded_loss import ConstrainedTaskCriterion

    parent = _criterion()
    wrapper = ConstrainedTaskCriterion.from_parent(parent)
    output, targets = _loss_example()
    torch.manual_seed(123)
    expected = parent(output, targets, 'masks')
    expected_rng = torch.get_rng_state()
    torch.manual_seed(123)
    actual = wrapper(output, targets, 'masks')
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], value) for key, value in expected.items())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    # An entirely ignored empty target is graph-connected zero, never NaN.
    output, targets = _loss_example()
    targets = [{key: value[:0] for key, value in targets[0].items()}]
    from models.observation_grounded_loss import AssignmentConstraints
    losses = wrapper(output, targets, 'masks', constraints=AssignmentConstraints(ignored=(0, 1, 2, 3)))
    assert all(torch.isfinite(value) and value == 0 for value in losses.values())
    sum(losses.values()).backward()
    assert output['pred_logits'].grad is not None


def test_group_identity_loss_excludes_unknown_columns_and_does_not_invent_null_labels():
    from models.observation_grounded_loss import grouped_identity_loss

    logits = torch.tensor([[1., 2., 100., 0.], [3., 2., 99., 0.]], requires_grad=True)
    loss, coverage = grouped_identity_loss(logits, ["A", "B"], ['A', 'A', None],
                                           valid=torch.ones(2, dtype=torch.bool))
    expected = torch.logsumexp(torch.tensor([1., 2., 0.]), 0) - torch.logsumexp(torch.tensor([1., 2.]), 0)
    torch.testing.assert_close(loss, expected)
    assert coverage['positive'] == 1 and coverage['unknown'] == 1 and coverage['null'] == 0
    loss.backward()
    assert logits.grad[:, 2].eq(0).all() and logits.grad[1].eq(0).all()


def test_optimizer_boundary_resume_reproduces_next_update_and_rng(tmp_path):
    import random
    import numpy as np
    from trainer.observation_grounded_trainer import save_resume, restore_resume
    from trainer.revisit_prompt_trainer import lr_scale

    def system():
        model = torch.nn.Module()
        model.mpr_prompt = torch.nn.Linear(2, 2)
        result = torch.nn.Module()
        result.model = model
        return result

    torch.manual_seed(20)
    uninterrupted = system()
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda cursor: lr_scale(cursor, 20))

    def update(module, opt, sched):
        opt.zero_grad(set_to_none=True)
        data = torch.randn(3, 2) + random.random() + float(np.random.random())
        module.model.mpr_prompt(data).square().sum().backward()
        opt.step()
        sched.step()

    update(uninterrupted, optimizer, scheduler)
    path = tmp_path / 'resume.pt'
    identity = dict(N=20, lambda_id=.1, data='bound', parent='bound', sources='bound')
    save_resume(path, uninterrupted, optimizer, scheduler, step=1, identity=identity)
    update(uninterrupted, optimizer, scheduler)
    expected_rng = torch.get_rng_state()
    expected_optimizer = optimizer.state_dict()
    resumed = system()
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda cursor: lr_scale(cursor, 20))
    saved = restore_resume(path, resumed, opt2, sched2, identity=identity)
    assert saved['next_draw'] == 4
    update(resumed, opt2, sched2)
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in uninterrupted.state_dict().items())
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert sched2.state_dict() == scheduler.state_dict()
    assert opt2.state_dict()['param_groups'] == expected_optimizer['param_groups']
