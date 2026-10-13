"""The ten focused MPR-V1 contract groups; real GPU checks are campaign P0/P1."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


class _TinyBackbone(torch.nn.Module):
    PLANES = [128] * 9

    def forward(self, x):
        return x, [], [x.decomposed_coordinates]

    def sparse_from_sample(self, features, sample):
        return SimpleNamespace(F=features, decomposed_features=[features],
                               decomposed_coordinates=sample.decomposed_coordinates)


def _tiny_model(cls, *, non_parametric_queries=False, **extra):
    return cls(config=SimpleNamespace(backbone=_TinyBackbone()), hidden_dim=128,
               num_queries=100, num_heads=8, dim_feedforward=256,
               sample_sizes=[], shared_decoder=False, num_classes=19,
               num_decoders=1, dropout=0., pre_norm=False,
               positional_encoding_type="fourier", non_parametric_queries=non_parametric_queries,
               train_on_segments=True, normalize_pos_enc=True, use_level_embed=False,
               scatter_type="mean", hlevels=[], use_np_features=False, voxel_size=.02,
               max_sample_size=False, random_queries=False, gauss_scale=1.,
               random_query_both=False, random_normal=False, D=4, num_changes=5,
               temporal_masking=False, use_changes_loss=False, save_segment_info=False,
               return_query_features=True, **extra)


def test_T01_empty_history_preserves_real_parent_decoder_path():
    from models.rescene import ReScene
    from models.revisit_prompt import RevisitPromptReScene
    from models.revisit_prompt_state import RevisitGeometryState

    torch.manual_seed(8)
    parent = _tiny_model(ReScene).eval()
    for variant in ("M0", "M4"):
        adapted = _tiny_model(RevisitPromptReScene, mpr_variant=variant, mpr_seed=146).eval()
        state_dict = adapted.state_dict()
        state_dict.update(parent.state_dict())
        adapted.load_state_dict(state_dict, strict=True)
        coordinates = torch.tensor([[0., 0., 0., 0.], [1., 0., 0., 0.]])
        features = torch.randn(2, 128)
        x = SimpleNamespace(F=features, decomposed_features=[features],
                            decomposed_coordinates=[coordinates])
        arguments = dict(point2segment=[torch.arange(2)], raw_coordinates=coordinates,
                         is_eval=True)
        expected = parent(x, **arguments)
        result = adapted(x, **arguments, task_state=_history(0),
                         geometry_state=RevisitGeometryState.empty(_history(0)),
                         stage_meta=[SimpleNamespace(absolute_stage_index=0,
                                                     segment_stage_ids=torch.zeros(2, dtype=torch.long))])
        assert torch.equal(expected["pred_logits"], result["pred_logits"])
        assert torch.equal(expected["pred_masks"][0], result["pred_masks"][0])
        assert sum(p.numel() for p in adapted.parameters() if p.requires_grad) == (0 if variant == "M0" else 84224)


def _history(count=40):
    from models.task_memory_state import TaskMemoryState

    state = TaskMemoryState.empty(batch_size=1, capacity=100, feature_dim=128,
                                 class_count=19, device="cpu", dtype=torch.float32)
    occupied = state.occupied.clone()
    occupied[:, :count] = True
    logical = state.logical_ids.clone()
    logical[:, :count] = torch.arange(count)
    generations = state.generations.clone()
    generations[:, :count] = 0
    last_seen = state.last_seen.clone()
    last_seen[:, :count // 2] = 1
    last_seen[:, count // 2:count] = 0
    embedding = state.embedding.clone()
    embedding[:, :count] = torch.nn.functional.normalize(torch.randn(1, count, 128), dim=-1)
    classes = state.class_prob.clone()
    classes[:, :count, 0] = 1
    confidence = state.confidence.clone()
    confidence[:, :count] = .9
    return replace(state, occupied=occupied, logical_ids=logical, generations=generations,
                   last_seen=last_seen, embedding=embedding, class_prob=classes,
                   confidence=confidence, stage_watermark=torch.tensor([1]),
                   next_logical_id=torch.tensor([count]))


def test_T02_slot_budget_and_current_segment_geometry():
    from models.revisit_prompt_modules import select_history_slots, segment_context

    state = _history()
    slots = select_history_slots(state, 0, 2)
    assert slots.tolist() == list(range(16)) + list(range(20, 36))
    assert slots.numel() == 32
    assert select_history_slots(_history(0), 0, 2).numel() == 0
    raw = torch.tensor([[1., 0., 0., 0.], [3., 0., 0., 0.],
                        [10., 0., 0., 1.], [14., 0., 0., 1.]])
    context = segment_context(raw, torch.tensor([0, 0, 1, 1]),
                              torch.randn(3, 128), torch.tensor([False, False, True]),
                              SimpleNamespace(segment_stage_ids=torch.tensor([0, 1])))
    assert context["current_indices"].tolist() == [1]
    assert torch.equal(context["centers"], torch.tensor([[2., 0., 0.], [12., 0., 0.]]))


def test_T03_global_retrieval_needs_neither_proposals_nor_labels():
    from models.revisit_prompt_modules import PromptAdapter

    prompt = PromptAdapter(retrieval=True, seed=146)
    with torch.no_grad():
        for layer in [prompt.Wq, prompt.Wk]:
            layer.weight.copy_(torch.eye(128))
            layer.bias.zero_()
    source = torch.zeros(1, 128)
    source[0, 0] = 1
    features = torch.zeros(3, 128)
    features[0, 1] = 1
    features[1:, 0] = 1
    coords = torch.tensor([[0., 0., 0.], [100., 0., 0.], [101., 0., 0.]])
    result = prompt.retrieve(source, features, coords)
    assert result["anchor_indices"].tolist() == [1]  # similarity tie: original row
    assert result["neighbor_indices"][0].tolist() == [1, 2, 0]
    assert torch.equal(result["centers"], coords[1:2])
    assert result["logits"].requires_grad and result["z"].requires_grad
    result["z"].sum().backward()
    assert prompt.Wk.weight.grad is not None


def test_T04_geometry_and_unknown_auxiliary_labels():
    from models.revisit_prompt_modules import geometry6, SupportHead
    from models.revisit_prompt_loss import retrieval_loss, support_loss

    geometry, center, valid = geometry6(torch.tensor([[0., 0., 0.], [2., 0., 0.]]))
    expected = torch.tensor([0., 0., 1., 1., 0., 1.]).log1p()
    assert torch.allclose(geometry, expected)
    assert torch.equal(center, torch.tensor([1., 0., 0.])) and valid
    assert geometry6(torch.ones(1, 3))[0].eq(0).all()
    assert not geometry6(torch.empty(0, 3))[2]
    scores = torch.tensor([[1., 2.]], requires_grad=True)
    loss = retrieval_loss(scores, torch.tensor([[False, True]]))
    assert torch.allclose(loss, torch.logsumexp(scores[0], 0) - scores[0, 1])
    empty_loss = retrieval_loss(scores, torch.zeros_like(scores, dtype=torch.bool))
    assert empty_loss.item() == 0 and empty_loss.requires_grad
    probabilities = SupportHead(seed=146)(torch.randn(2, 272))
    assert torch.allclose(probabilities, torch.full((2,), torch.sigmoid(torch.tensor(2.)).item()))
    unknown = support_loss(probabilities, torch.tensor([-1, -1]))
    assert unknown.item() == 0 and unknown.requires_grad
    unknown.backward()


def test_T05_shared_parameters_and_zero_init_gradient_progression():
    from models.revisit_prompt_modules import PromptAdapter, SupportHead

    left = PromptAdapter(retrieval=True, seed=146)
    right = PromptAdapter(retrieval=True, seed=146)
    assert sum(p.numel() for p in left.parameters()) == 84224
    assert sum(p.numel() for p in PromptAdapter(retrieval=False, seed=146).parameters()) == 51200
    assert sum(p.numel() for p in SupportHead(seed=146).parameters()) == 17537
    assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
    decoder = torch.nn.Linear(128, 2).requires_grad_(False)
    before_parent = {k: v.clone() for k, v in decoder.state_dict().items()}
    optimizer = torch.optim.AdamW(left.parameters(), lr=1e-4)
    original = left.input_layer.weight.detach().clone()
    source, features, coords = torch.randn(2, 128), torch.randn(4, 128), torch.randn(4, 3)
    for step in range(3):
        optimizer.zero_grad()
        retrieved = left.retrieve(source, features, coords)
        delta = left(source, retrieved["z"], torch.zeros(2, 6),
                     retrieved["geometry"], torch.ones(2, 2))
        decoder(delta).square().mean().backward()
        if step == 0:
            assert left.input_layer.weight.grad.eq(0).all()
        optimizer.step()
    assert not torch.equal(original, left.input_layer.weight)
    assert all(torch.equal(v, decoder.state_dict()[k]) for k, v in before_parent.items())


def test_T10_parent_coverage_is_exact_and_new_names_are_whitelisted():
    from scripts.revisit_prompt_assets import load_parent_state

    model = torch.nn.Module()
    model.old = torch.nn.Linear(2, 2)
    model.mpr_prompt = torch.nn.Linear(2, 2)
    system = torch.nn.Module()
    system.model = model
    state = {k: v.clone() for k, v in system.state_dict().items()
             if not k.startswith("model.mpr_prompt.")}
    payload = dict(state_dict=state, omitted_encoder_parameter_names=[], included_encoder_buffers=[])
    result = load_parent_state(system, payload)
    assert result["loaded_task_entries"] == 2
    assert len(result["new_parameter_names"]) == 2
    with pytest.raises(ValueError, match="coverage"):
        load_parent_state(system, {**payload, "state_dict": {"model.old.weight": state["model.old.weight"]}})
    with pytest.raises(ValueError, match="dtype"):
        load_parent_state(system, {**payload, "state_dict": {k: v.double() for k, v in state.items()}})


def test_T02_uuid_draws_keep_scope_and_real_local_windows():
    from datasets.revisit_prompt_episode import RevisitEpisodeSpec, RevisitDrawPlan

    ids = tuple(f"00000000-0000-4000-8000-{i:012d}" for i in range(6))
    references = [dict(role="TRAIN", reference_id=ids[0], Tmax=6,
                       scan_ids=list(ids), scan_indices=list(range(10, 16)))]
    plan = RevisitDrawPlan(references, seed=146)
    records = plan.draw_update(0)
    assert [r["horizon"] for r in records] == [2, 3, 4, 5]
    assert records == RevisitDrawPlan(references, seed=146).draw_update(0)
    assert sum(r["horizon"] for r in records) == 14
    for record in records:
        spec = RevisitEpisodeSpec.from_record(record)
        assert spec.dataset_scope == "train"
        assert spec.stage_scan_ids[0] == (spec.scan_ids[0],)
        assert spec.stage_scan_ids[-1] == spec.scan_ids[-2:]
        assert len(set(spec.scan_ids)) == spec.horizon
        assert all(references[0]["scan_ids"][i-10] == u
                   for i, u in zip(spec.scan_indices, spec.scan_ids))
    with pytest.raises(ValueError, match="scope"):
        RevisitEpisodeSpec.from_record({**records[0], "role": "PB", "dataset_scope": "train"})


@pytest.mark.parametrize("source_dtype", [torch.int32, torch.int64])
def test_T08_prefix_class_semantics_override_window_first_point_without_future(source_dtype):
    from datasets.revisit_prompt_episode import causal_prefix_truth, project_causal_target
    from models.criterion import SetCriterion

    prefix_labels = torch.tensor([[4, 42, 0, 0], [7, 42, 0, 1], [8, 8, 0, 2]], dtype=source_dtype)
    common = causal_prefix_truth(prefix_labels[:2], torch.tensor([0, 1]),
                                 filter_out_classes=[0, 1, 255], label_offset=2)
    local_labels = torch.tensor([[7, 42, 0, 0], [255, 42, 0, 1], [8, 8, 0, 2]])
    result = project_causal_target(common, local_labels, torch.tensor([0, 1, 2]),
                                  torch.tensor([0, 1, 1]), segment_masks=True)
    assert result["ids"].tolist() == [42]
    assert result["labels"].tolist() == [2]  # first *causal prefix* point has class 4
    assert result["masks"].tolist() == [[True, True, False]]
    assert result["segment_mask"].tolist() == [[True, True, False]]
    criterion = SetCriterion(num_classes=19, matcher=None, weight_dict={}, eos_coef=.1,
        losses=[], num_points=-1, oversample_ratio=3, importance_sample_ratio=.75,
        class_weights=-1, num_changes=5, change_weights=[1.]*5)
    output = {"pred_logits":torch.zeros(1, 2, 19, requires_grad=True),
              "pred_changes":torch.zeros(1, 2, 5, requires_grad=True)}
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    loss = criterion.loss_labels(output, [result], matches, 1, "segment_mask")["loss_ce"]
    loss += criterion.loss_change_labels(output, [result], matches, 1, "segment_mask")["loss_changes_ce"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in output.values())
    assert all(common[key].dtype == result[key].dtype == torch.int64
               for key in ("ids", "labels", "changes"))
    empty = project_causal_target(common, local_labels[2:], torch.tensor([0]),
                                 torch.tensor([1]), segment_masks=True)
    assert empty["masks"].shape == (0, 1) and empty["segment_mask"].shape == (0, 1)
    assert all(empty[key].dtype == torch.int64 for key in ("ids", "labels", "changes"))


def test_T02_history_positions_use_reference_frame_across_window_shifts():
    from models.revisit_prompt import RevisitPromptReScene
    from models.revisit_prompt_state import RevisitGeometryState

    model = _tiny_model(RevisitPromptReScene, mpr_variant="M3", non_parametric_queries=True)
    state = _history(1)
    geometry = RevisitGeometryState.empty(state)
    geometry.valid[0, 0] = True
    geometry.centroid[0, 0] = torch.tensor([52., 0., 0.])
    raw = torch.tensor([[0., 0., 0., 0.], [2., 0., 0., 1.]])
    meta = SimpleNamespace(absolute_stage_index=2, segment_stage_ids=torch.tensor([0, 1]))
    model._mpr_state, model._mpr_geometry = state, geometry
    model._mpr_coordinate_offsets = torch.tensor([[50., 0., 0.]])
    _, _, sampled = model.before_first_mask(
        torch.zeros(1, 100, 128), torch.zeros(1, 100, 128), torch.zeros(1, 100, 4),
        dense_features=torch.randn(1, 2, 128), padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        raw_coordinates=raw, point2segment=[torch.arange(2)], stage_meta=[meta], coords=[[raw]])
    assert sampled[0, 0].tolist() == [2., 0., 0., 1.]
    assert model._mpr_contexts[0]["centers"].tolist() == [[50., 0., 0.], [52., 0., 0.]]


def _stage_meta(stage=0, episode="episode-A"):
    from datasets.task_memory_episode import StageMeta

    n = 1 if stage == 0 else 2
    stages = torch.arange(n)
    return StageMeta(reference_id="00000000-0000-4000-8000-000000000001",
        episode_id=episode, scan_ids_in_window=tuple(f"scan-{i}" for i in range(n)),
        absolute_stage_index=stage, local_stage_ids=stages,
        original_vertex_ids=tuple(torch.zeros(1, dtype=torch.long) for _ in range(n)),
        scan_vertex_offsets=torch.arange(n+1), point2segment=stages,
        segment_stage_ids=stages, augmentation_transform_id="identity",
        coordinate_frame_id="reference-frame", voxel_inverse=stages,
        full_resolution_point2segment=stages)


def _final_output(query=77, segments=1):
    features = torch.zeros(1, 100, 128)
    features[0, query, 0] = 1
    logits = torch.full((1, 100, 19), -20.)
    logits[:, :, 18] = 20.
    logits[0, query, 0], logits[0, query, 18] = 3., -3.
    masks = torch.full((segments, 100), -20.)
    masks[-1, query] = 20.
    return dict(query_features=features, pred_logits=logits, pred_masks=[masks],
                mpr_support_probabilities=torch.ones(1, 100))


def test_T06_final_commit_issues_ids_once_and_geometry_tracks_accepted_slot():
    from scripts.revisit_prompt_runtime import common_commit, empty_common_state
    from models.revisit_prompt_state import RevisitGeometryState

    state = empty_common_state(class_count=19, device="cpu")
    meta = _stage_meta()
    output = _final_output()
    observation, commit = common_commit(output, (meta,), state, background_class=18)
    assert commit.query_to_logical_id[0, 77].item() == 0
    assert commit.query_to_slot[0, 77].item() == 0
    assert state.stage_watermark.item() == -1 and commit.state.stage_watermark.item() == 0
    assert observation.current_supported[0, 77] and commit.births[0, 77]
    geometry = RevisitGeometryState.empty(state)
    geometry.update(commit, observation, output["pred_masks"],
                    [dict(current_indices=torch.tensor([0]), centers=torch.tensor([[9., 0., 0.]]))], 0)
    geometry.assert_synchronized(commit.state)
    assert geometry.centroid[0, 0].tolist() == [9., 0., 0.]
    with pytest.raises(ValueError, match="watermark|advance"):
        common_commit(output, (meta,), commit.state, background_class=18)
    geometry.generations[0, 0] = 3
    with pytest.raises(ValueError, match="identity|generation"):
        geometry.assert_synchronized(commit.state)
    # Separate episodes can issue local ID 0 independently; the state is never shared.
    _, separate = common_commit(output, (_stage_meta(episode="episode-B"),),
                                empty_common_state(class_count=19, device="cpu"), background_class=18)
    assert separate.query_to_logical_id[0, 77].item() == 0


def test_T07_support_scales_confidence_and_candidate_score_once_after_extraction():
    from scripts.revisit_prompt_runtime import common_commit, empty_common_state, scale_candidates
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    output = _final_output()
    original_logits = output["pred_logits"].clone()
    output["mpr_support_probabilities"][0, 77] = .2
    observation, commit = common_commit(output, (_stage_meta(),),
        empty_common_state(class_count=19, device="cpu"), background_class=18)
    assert not observation.valid[0, 77] and not commit.births.any()
    assert torch.equal(observation.class_prob, original_logits.softmax(-1))
    assert torch.equal(output["pred_logits"], original_logits)
    masks = torch.ones(1, 1, dtype=torch.bool)
    official = OfficialTaskPrediction(masks, torch.tensor([.8]), torch.tensor([1]),
        torch.tensor([77]), torch.tensor([0]), torch.tensor([0]), 0, masks)
    scaled = scale_candidates(official, output["mpr_support_probabilities"][0])
    assert scaled.pred_masks.shape[1] == 1  # no additional candidate deletion
    assert torch.allclose(scaled.pred_scores, torch.tensor([.16]))
    assert torch.equal(official.pred_scores, torch.tensor([.8]))


def test_T04_segment_supervision_ignores_absent_history_and_unknown_anchors():
    from models.revisit_prompt_loss import evidence_labels

    # Segment 0 has known identity 42; segment 1 has a purity tie; segment 2 is ignored.
    p2s = torch.tensor([0, 0, 1, 1, 2])
    ids = torch.tensor([42, 42, 42, 8, -1])
    positive, support = evidence_labels(p2s, ids, torch.tensor([0, 1, 2]),
        historical_ids=[42, 8, None], anchors=torch.tensor([0, 0, 2]),
        reliable_ids={42, 8})
    assert positive.tolist() == [[True, True, False], [False, True, False], [False, False, False]]
    assert support.tolist() == [1, 0, -1]
    _, support_tie = evidence_labels(p2s, ids, torch.tensor([0, 1, 2]),
        historical_ids=[42], anchors=torch.tensor([1]), reliable_ids={42, 8})
    assert support_tie.tolist() == [-1]


def test_T04_training_ledger_binds_only_unambiguous_actual_births():
    from models.revisit_prompt_loss import TrainingLabelLedger, bind_training_births
    from scripts.revisit_prompt_runtime import common_commit, empty_common_state

    meta, output = _stage_meta(), _final_output()
    _, commit = common_commit(output, (meta,), empty_common_state(class_count=19, device="cpu"),
                              background_class=18)
    target = dict(ids=torch.tensor([42]), labels=torch.tensor([0]),
                  masks=torch.ones(1, 1, dtype=torch.bool), temporal_stages=torch.tensor([0]),
                  ambiguities=[])
    stage = SimpleNamespace(stage_meta=(meta,), batch=(SimpleNamespace(target_full=[target]), [], []))
    ledger = TrainingLabelLedger()
    counts = bind_training_births(ledger, commit, output, stage)
    assert counts["bound_births"] == 1 and ledger.lookup(0, 0) == (meta.reference_id, 42)
    target["ids"] = torch.tensor([99])
    bind_training_births(ledger, replace(commit, births=torch.zeros_like(commit.births)), output, stage)
    assert ledger.lookup(0, 0)[1] == 42  # later observations cannot rewrite a birth label
    target["ambiguities"] = [[99, 100]]
    unbound = TrainingLabelLedger()
    assert bind_training_births(unbound, commit, output, stage)["bound_births"] == 0
    assert unbound.lookup(0, 0) is None
    target.update(ids=torch.empty(0, dtype=torch.long), labels=torch.empty(0, dtype=torch.long),
                  masks=torch.empty(0, 1, dtype=torch.bool), ambiguities=[])
    assert bind_training_births(unbound, commit, output, stage)["bound_births"] == 0


def test_T10_resume_restores_optimizer_scheduler_and_rng_without_parent_overwrite(tmp_path):
    import random
    import numpy as np
    from trainer.revisit_prompt_trainer import save_resume, restore_resume, lr_scale

    system = torch.nn.Module()
    system.model = torch.nn.Module()
    system.model.parent = torch.nn.Linear(2, 2).requires_grad_(False)
    system.model.mpr_prompt = torch.nn.Linear(2, 2)
    parent_before = {k: v.clone() for k, v in system.model.parent.state_dict().items()}
    optimizer = torch.optim.AdamW(system.model.mpr_prompt.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_scale(step, 400))
    system.model.mpr_prompt(torch.ones(1, 2)).sum().backward()
    optimizer.step(); scheduler.step()
    path = tmp_path/"resume.pt"
    identity = dict(parent_sha256="bound-parent", config_sha256="bound-config", N=400)
    save_resume(path, system, optimizer, scheduler, step=1, identity=identity)
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(2))
    with torch.no_grad(): system.model.mpr_prompt.weight.zero_()
    restored = restore_resume(path, system, optimizer, scheduler, identity=identity)
    assert restored["step"] == 1 and restored["next_draw"] == 4
    assert restored["scheduler"]["last_epoch"] == 1 and optimizer.state
    assert random.random() == expected_rng[0] and float(np.random.rand()) == expected_rng[1]
    assert torch.equal(torch.rand(2), expected_rng[2])
    assert all(torch.equal(v, system.model.parent.state_dict()[k]) for k,v in parent_before.items())
    assert lr_scale(0, 400) == .05 and lr_scale(19, 400) == 1 and lr_scale(399, 400) == .1
    with pytest.raises(ValueError, match="identity"):
        restore_resume(path, system, optimizer, scheduler, identity={**identity, "N": 800})


def test_T07_absolute_trajectory_adapter_preserves_class_expansion_and_ephemeral_candidates():
    from scripts.revisit_prompt_runtime import trajectory_sidecar
    from scripts.system_comparison_v2_inference import OfficialCandidateTrajectoryAccumulator
    from scripts.rescene_task_postprocess import OfficialTaskPrediction

    accumulator = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    provenance = dict(checkpoint_sha256="0"*64, config_hash="1"*64,
                      protocol_manifest_hash="2"*64, source_raw_observation_fingerprint="3"*64)
    first_masks = torch.ones(1, 2, dtype=torch.bool)
    first = OfficialTaskPrediction(first_masks, torch.tensor([.8, .3]), torch.tensor([1, 2]),
        torch.tensor([77, 77]), torch.tensor([0, 1]), torch.tensor([0]), 0, first_masks)
    track_ids = [None]*100
    track_ids[77] = 0
    accumulator.add_stage(trajectory_sidecar(first, _stage_meta(), history=["scan-0"], provenance=provenance),
                          SimpleNamespace(stage_id=0, track_ids=track_ids))
    second_masks = torch.ones(2, 2, dtype=torch.bool)
    second = OfficialTaskPrediction(second_masks, torch.tensor([.4, .2]), torch.tensor([1, 1]),
        torch.tensor([77, 88]), torch.tensor([0, 0]), torch.tensor([0, 1]), 1, second_masks[1:])
    accumulator.add_stage(trajectory_sidecar(second, _stage_meta(1), history=["scan-0", "scan-1"],
                                            provenance=provenance), SimpleNamespace(stage_id=1, track_ids=track_ids))
    prediction = accumulator.snapshot().prediction
    assert torch.allclose(prediction["pred_scores"], torch.tensor([.6, .3, .2]))
    assert prediction["pred_masks"].tolist() == [[True, True, False], [True, False, True]]


def test_T09_budget_charges_each_event_once_and_respects_phase_and_lifetime_caps(tmp_path):
    from scripts.revisit_prompt_campaign import available_budget
    import json

    plan = dict(budget=dict(prior_gpu_hours=10., lifetime_cap_gpu_hours=20.,
                           new_cap_gpu_hours=1., reserve_gpu_hours=8.),
                phase_caps_gpu_hours={"P0":.4,"P1":.2,"P2":.4,"P3":0.,"P4":0.,"P5":0.})
    event = dict(event_id="one-child",phase="P0",gpu_hours=.3)
    (tmp_path/"COST_LEDGER.jsonl").write_text(json.dumps(event)+"\n"+json.dumps(event)+"\n")
    assert available_budget(tmp_path, plan, phase="P0") == pytest.approx(.1)
    assert available_budget(tmp_path, plan, phase="P1") == pytest.approx(.2)
    assert available_budget(tmp_path, plan, phase="P2") == pytest.approx(.4)
    with pytest.raises(ValueError, match="authorized|lifetime|cap"):
        available_budget(tmp_path, {**plan,"budget":{**plan["budget"],"new_cap_gpu_hours":3.}}, phase="P0")


def test_T09_entrypoints_require_exact_run_plan_and_locked_confirmation():
    from scripts.revisit_prompt_assets import validate_run_plan, validate_evaluation_request

    assets = dict(budget={"new_cap_gpu_hours":78.})
    sources = {"model.py":"source-hash"}
    manifest = dict(counts={"CAL":{"1":31}, "SEL":{"1":39}, "PB":{"2":129}})
    plan = dict(status="LOCKED", N=400, checkpoint_steps=[100,200,300,400],
                budget=assets["budget"], numeric_sources=sources, expected_inputs=manifest["counts"])
    assert validate_run_plan(plan, assets, sources, manifest)["N"] == 400
    with pytest.raises(ValueError, match="source|numeric"):
        validate_run_plan({**plan,"numeric_sources":{}}, assets, sources, manifest)
    with pytest.raises(ValueError, match="budget"):
        validate_run_plan({**plan,"budget":{}}, assets, sources, manifest)
    validate_evaluation_request(variant="M4", seed=146, step=100, role="CAL", run_plan=plan)
    with pytest.raises(ValueError, match="checkpoint|step"):
        validate_evaluation_request(variant="M4", seed=146, step=0, role="CAL", run_plan=plan)
    endpoint = dict(variant="M4",adapter_seed=146,step=200)
    selection = dict(status="COMPLETE",control_star=dict(variant="M0",adapter_seed=146,step=0),
                     candidate_star=endpoint,candidate_descriptive_star=endpoint)
    validate_evaluation_request(variant="M4",seed=146,step=200,role="SEL",run_plan=plan,selection=selection)
    with pytest.raises(ValueError, match="locked|endpoint"):
        validate_evaluation_request(variant="M4",seed=146,step=100,role="PB",run_plan=plan,selection=selection)
    with pytest.raises(ValueError, match="lock|selection"):
        validate_evaluation_request(variant="M0",seed=146,step=0,role="PB",run_plan=plan)


def test_T08_evaluation_current_truth_retains_empty_targets_and_change_labels():
    from scripts.evaluate_revisit_prompt import current_metric_truth

    target = dict(ids=torch.tensor([7,8]),labels=torch.tensor([1,2]),
                  masks=torch.tensor([[True,False,False],[False,False,True]]),
                  changes=torch.tensor([3,4]),temporal_stages=torch.tensor([0,1,1]))
    current = current_metric_truth(target,1)
    assert current["ids"].tolist() == [8]
    assert current["changes"].tolist() == [4]
    assert current["masks"].tolist() == [[False,True]]
    assert current["temporal_stages"].tolist() == [0,0]
    empty = current_metric_truth(target,2)
    assert empty["masks"].shape == (0,0) and empty["ids"].numel() == 0


def test_T07_reference_states_repool_official_ap_instead_of_averaging_scene_ap():
    from pathlib import Path
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.evaluate_revisit_prompt import pool_official_states

    taxonomy = Path(__file__).resolve().parents[1]/"artifacts/native_long_postselection_v1/deployment/rio.yaml"
    pooled = OfficialMetricAccumulator(mode="strict_online",dataset_spec=taxonomy)
    references = []
    per_scene = []
    for objects, predicted in ((1,1),(3,1)):
        masks = torch.zeros(objects,objects*300,dtype=torch.bool)
        for i in range(objects):
            masks[i,i*150:(i+1)*150] = True
            masks[i,objects*150+i*150:objects*150+(i+1)*150] = True
        target = dict(ids=torch.arange(objects)+42,labels=torch.full((objects,),5),masks=masks,
                      changes=torch.zeros(objects,dtype=torch.long),
                      temporal_stages=torch.cat((torch.zeros(objects*150,dtype=torch.long),
                                                 torch.ones(objects*150,dtype=torch.long))))
        prediction = dict(pred_masks=masks[:predicted].T,pred_scores=torch.full((predicted,),.9),
                          pred_classes=torch.full((predicted,),5))
        metric = OfficialMetricAccumulator(mode="strict_online",dataset_spec=taxonomy)
        metric.update(prediction,target)
        pooled.update(prediction,target)
        references.append(metric.export_evidence())
        per_scene.append(metric.compute()["online_t-mAP"])
    result = pool_official_states(references,dataset_spec=taxonomy)
    assert result["online_t-mAP"] == pytest.approx(pooled.compute()["online_t-mAP"])
    assert abs(result["online_t-mAP"]-sum(per_scene)/2) > .05
    assert pool_official_states([references[0],references[0]],dataset_spec=taxonomy)["online_t-mAP"] == pytest.approx(per_scene[0])


def test_T05_private_adapter_initialization_preserves_cuda_and_cpu_generators(monkeypatch):
    from models.revisit_prompt_modules import PromptAdapter, SupportHead

    calls = []
    monkeypatch.setattr(torch.cuda,"manual_seed_all",lambda seed:calls.append(seed))
    before = torch.get_rng_state().clone()
    PromptAdapter(seed=146)
    SupportHead(seed=146)
    assert calls == []
    assert torch.equal(before,torch.get_rng_state())


def test_T09_cost_admission_uses_full_matrix_two_most_expensive_replicas_and_fixed_evaluation():
    from scripts.revisit_prompt_campaign import forecast_cost_plan

    budget = dict(new_cap_gpu_hours=78.)
    pilots = {f"M{i}":dict(update_seconds=float(i),startup_seconds=10.) for i in range(1,7)}
    evaluation = dict(seconds_per_stage=1.,seconds_per_input=.1,startup_seconds=10.)
    manifest = dict(counts={"CAL":{"1":31,"2":19,"3":14,"4":14,"5":6},
                           "SEL":{"1":39,"2":21,"3":21,"4":15,"5":15},
                           "PB":{"2":129,"3":129,"4":129,"5":129}})
    plan = forecast_cost_plan(budget,pilots,evaluation,manifest)
    assert plan["N"] == 1600 and plan["checkpoint_steps"] == [400,800,1200,1600]
    estimate = plan["training_estimates_gpu_hours"]["1600"]
    expected = 1.25*((sum(range(1,7))+6+5)*1600+8*10)/3600
    assert estimate == pytest.approx(expected)
    assert sum(plan["phase_caps_gpu_hours"].values()) == pytest.approx(78.)
    assert plan["worst_replication_variants"]["1600"] == ["M6","M5"]
    blocked = forecast_cost_plan(budget,{k:{**v,"update_seconds":1000.} for k,v in pilots.items()},evaluation,manifest)
    assert blocked["N"] is None and blocked["status"] == "BLOCKED_FULL_MATRIX_BUDGET"
    assert blocked["phase_caps_gpu_hours"]["P3"] == pytest.approx(plan["phase_caps_gpu_hours"]["P3"])
    assert blocked["phase_caps_gpu_hours"]["P4"] == pytest.approx(plan["phase_caps_gpu_hours"]["P4"])


def test_T08_empty_official_population_retains_input_count_and_reports_na():
    from pathlib import Path
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.evaluate_revisit_prompt import official_compute_nullable,pool_official_states

    taxonomy=Path(__file__).resolve().parents[1]/"artifacts/native_long_postselection_v1/deployment/rio.yaml"
    metric=OfficialMetricAccumulator(mode="raw_local",dataset_spec=taxonomy)
    metric.update(dict(pred_masks=torch.empty(200,0,dtype=torch.bool),pred_scores=torch.empty(0),
                       pred_classes=torch.empty(0,dtype=torch.long)),
                  dict(masks=torch.empty(0,200,dtype=torch.bool),ids=torch.empty(0,dtype=torch.long),
                       labels=torch.empty(0,dtype=torch.long),changes=torch.empty(0,dtype=torch.long),
                       temporal_stages=torch.zeros(200,dtype=torch.long)))
    assert official_compute_nullable(metric)["raw_local_AP"] is None
    evidence=metric.export_evidence()
    assert evidence["updates"] == 1
    assert pool_official_states([evidence],dataset_spec=taxonomy)["raw_local_AP"] is None


def test_T08_mask_diagnostic_counts_unbound_candidates_once_and_preserves_losses():
    from scripts.evaluate_revisit_prompt import current_mask_diagnostic

    target = dict(ids=torch.tensor([8,9]),labels=torch.tensor([5,5]),
                  masks=torch.tensor([[True,True,False,False],[False,False,True,True]]))
    # These candidates have no memory ID; duplicate masks cannot recover two GTs.
    prediction = dict(pred_masks=torch.tensor([[True,True],[True,True],[False,False],[False,False]]),
                      pred_classes=torch.tensor([5,5]),pred_scores=torch.tensor([.9,.8]))
    diagnosis = current_mask_diagnostic(prediction,target)
    assert diagnosis["matched_gt_ids"] == [8]
    assert diagnosis["unmatched_gt_ids"] == [9]
    assert diagnosis["gt_count"] == 2 and diagnosis["candidate_count"] == 2
    wrong_class = {**prediction,"pred_classes":torch.tensor([6,6])}
    assert current_mask_diagnostic(wrong_class,target)["matched_gt_ids"] == []


def test_T09_campaign_reuses_only_identical_completed_jobs(tmp_path,monkeypatch):
    import scripts.revisit_prompt_campaign as campaign
    from scripts.native_long_budget import write_json_atomic

    monkeypatch.setattr(campaign,"numeric_sources",lambda:{"model.py":"frozen"})
    calls = []
    monkeypatch.setattr(campaign,"charged_job",lambda *a,**k:calls.append(k))
    status = tmp_path/"evaluation/STATUS.json"
    write_json_atomic(status,dict(status="COMPLETE",numeric_sources={"model.py":"frozen"}))
    assets = dict(run_root=str(tmp_path))
    result = campaign.ensure_job(assets,["python","evaluate"],status_path=status,
        phase="P3",job_key="M4-quarter",complete_status="COMPLETE")
    assert result["status"] == "COMPLETE" and calls == []
    write_json_atomic(status,dict(status="COMPLETE",numeric_sources={"model.py":"different"}))
    with pytest.raises(ValueError,match="source"):
        campaign.ensure_job(assets,["python","evaluate"],status_path=status,
            phase="P3",job_key="M4-quarter",complete_status="COMPLETE")
    assert calls == []


def test_T09_campaign_stops_after_same_runtime_error_repeats(tmp_path,monkeypatch):
    import scripts.revisit_prompt_campaign as campaign
    monkeypatch.setattr(campaign,"numeric_sources",lambda:{})
    calls = []
    def fail(*args,**kwargs):
        calls.append(kwargs)
        raise RuntimeError("same data error")
    monkeypatch.setattr(campaign,"charged_job",fail)
    with pytest.raises(RuntimeError,match="repeated"):
        campaign.ensure_job(dict(run_root=str(tmp_path)),["python","train"],
            status_path=tmp_path/"train/STATUS.json",phase="P2",job_key="M5-s146",
            complete_status="COMPLETE",training=True)
    assert len(calls) == 2


def test_T09_final_verdict_requires_all_replication_roles_and_per_horizon_guard():
    from scripts.revisit_prompt_campaign import confirmation_verdict
    selection = dict(status="COMPLETE",replication=dict(eligible=True))
    control = dict(status="COMPLETE",S_long=.3,S_current=.4,
                   AP={str(h):.3 for h in range(1,6)},repeat_ranges=dict(S_long=.0001))
    candidate = dict(status="COMPLETE",S_long=.31,S_current=.4,
                     AP={str(h):.31 for h in range(1,6)},repeat_ranges=dict(S_long=.0001))
    pair = dict(control=control,candidate=candidate)
    incomplete = {"s146:SEL":pair,"s146:PB":pair}
    assert confirmation_verdict(selection,incomplete,replication_complete=True) == "INCONCLUSIVE"
    complete = {f"s{seed}:{role}":pair for seed in (146,147) for role in ("SEL","PB")}
    assert confirmation_verdict(selection,complete,replication_complete=True) == "PROMISING_REPLICATED_ADAPTER_SIGNAL"
    complete["s147:SEL"] = dict(control=control,candidate={**candidate,"AP":{**candidate["AP"],"2":.297}})
    assert confirmation_verdict(selection,complete,replication_complete=True) == "CONFIRMATION_NO_GO"
    rejected = dict(status="COMPLETE",replication=dict(eligible=False))
    assert confirmation_verdict(rejected,complete,replication_complete=True) == "SCREEN_NO_GO"
