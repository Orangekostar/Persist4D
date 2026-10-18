"""Prediction-only state and complete route/commit contracts."""

from dataclasses import replace

import torch


def test_published_target_iou_follows_source_query_after_candidate_shuffle():
    from scripts.ogr_diagnostics import published_target_iou
    from types import SimpleNamespace

    truth = torch.tensor([True, True, False, False])
    official = SimpleNamespace(source_query_ids=torch.tensor([9, 2, 2]),
        pred_classes=torch.tensor([4, 3, 4]),
        latest_stage_masks=torch.tensor([[1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool))
    # Query9 has a perfect mask, but the designated query2/class4 has IoU1/3.
    assert published_target_iou(official, source_query=2, target=truth, target_class=4) == 1/3
    assert published_target_iou(official, source_query=8, target=truth, target_class=4) == 0.
    order = torch.tensor([2, 0, 1])
    shuffled = SimpleNamespace(source_query_ids=official.source_query_ids[order],
        pred_classes=official.pred_classes[order], latest_stage_masks=official.latest_stage_masks[:, order])
    assert published_target_iou(shuffled, source_query=2, target=truth, target_class=4) == 1/3


def test_shared_history_clone_preserves_episode_and_owns_independent_sidecars():
    from types import SimpleNamespace
    from scripts.ogr_runtime import ObservationEpisodeRuntime
    from scripts.ogr_diagnostics import clone_runtime

    spec = SimpleNamespace(input_id='fixed-input', reference_id='fixed-reference')
    runtime = ObservationEpisodeRuntime(spec, class_count=3, device='cpu')
    copied = clone_runtime(runtime)
    assert copied.input_id == spec.input_id and copied.reference_id == spec.reference_id
    copied.state.embedding[0, 0, 0] = 8.
    copied.geometry.geometry[0, 0, 0] = 9.
    copied.observation_state.prototype[0, 0, 0] = 10.
    assert runtime.state.embedding[0, 0, 0] == 0.
    assert runtime.geometry.geometry[0, 0, 0] == 0.
    assert runtime.observation_state.prototype[0, 0, 0] == 0.

from tests.test_task_memory_routing import _meta, _observation, _occupied_state


def test_partial_route_resolves_contention_with_dummy_and_preserves_ephemeral():
    from scripts.ogr_runtime import observation_route
    from models.task_memory_routing import commit_entities

    state = _occupied_state()
    observation = _observation([[1., 0.], [1., 0.], [0., 1.]])
    probabilities = torch.tensor([[.7, .1, .2], [.8, .1, .1], [.1, .4, .5]])
    route = observation_route(observation, state, [_meta(1)],
        [dict(probabilities=probabilities, slots=torch.tensor([0, 1]))])
    assert route.query_to_slot.tolist() == [[-1, 0, -1]]
    assert route.slot_to_query.tolist() == [[1, -1]]
    route.validate()
    commit = commit_entities(observation, route, state, [_meta(1)])
    assert commit.query_to_logical_id.tolist() == [[-1, 10, -1]]
    assert commit.rejected_births.tolist() == [[True, False, True]]
    assert torch.equal(commit.state.logical_ids, state.logical_ids)


def test_no_current_support_cannot_reclaim_identity_even_with_high_probability():
    from scripts.ogr_runtime import observation_route

    state = _occupied_state()
    observation = _observation([[1., 0.]], current=[False], previous=[True])
    route = observation_route(observation, state, [_meta(1)],
        [dict(probabilities=torch.tensor([[.99, .005, .005]]), slots=torch.tensor([0, 1]))])
    assert route.query_to_slot.tolist() == [[-1]]


def test_observation_state_uses_common_confidence_ema_and_preserves_unobserved_values():
    from models.observation_evidence_state import ObservationEvidenceState
    from models.task_memory_routing import route_entities, commit_entities

    state = _occupied_state()
    sidecar = ObservationEvidenceState.empty(state)
    # Test retained prototype in an occupied slot and an invalid occupied identity.
    sidecar.prototype[0, 0] = torch.tensor([1., 0.])
    sidecar.valid[0, 0] = True
    sidecar.quality[0, 0] = .8
    before = sidecar.prototype[0, 1].clone()
    observation = _observation([[1., 0.]], confidence=[.5])
    commit = commit_entities(observation, route_entities(observation, state, [_meta(1)]), state, [_meta(1)])
    values = torch.tensor([[0., 1.]], requires_grad=True)
    sidecar.update(commit, observation, [dict(values=values, valid=torch.tensor([True]))], state, 1)
    # confidence_ema: rate = .2 * .5 = .1, not the configured upper rate .2.
    expected = torch.tensor([.9, .1]) / torch.tensor(.82).sqrt()
    torch.testing.assert_close(sidecar.prototype[0, 0], expected)
    torch.testing.assert_close(sidecar.quality[0, 0], torch.tensor(.77))
    assert torch.equal(sidecar.prototype[0, 1], before)
    assert sidecar.last_updated.tolist() == [[1, -1]]
    assert sidecar.prototype.grad_fn is None and not sidecar.prototype.requires_grad
    sidecar.assert_synchronized(commit.state)
    assert sidecar.logical_ids.tolist() == [[10, 11]]
