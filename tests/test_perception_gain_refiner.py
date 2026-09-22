from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from datasets.task_memory_episode import StageMeta
from models.perception_gain import (
    CausalMaskRefiner,
    SoftOccurrence,
    pair_adjacent_soft_occurrences,
)
from scripts.rescene_task_postprocess import (
    OfficialTaskPrediction,
    OfficialTaskSoftEvidence,
)
from scripts.task_memory_output import (
    PublishedIdentity,
    RevisionCandidate,
    RevisionMaskRequest,
)


def _training_module():
    try:
        return importlib.import_module("scripts.train_perception_refiner")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception refiner trainer is unavailable: {error}")


def _preparation_module():
    try:
        return importlib.import_module("scripts.prepare_perception_refiner")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception refiner preparation is unavailable: {error}")


def _evaluation_module():
    try:
        return importlib.import_module("scripts.perception_refiner_evaluation")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception refiner evaluation is unavailable: {error}")


def test_additional_refiner_scores_only_each_native_terminal_horizon() -> None:
    evaluation = _evaluation_module()

    assert evaluation.scored_horizons_for_episode("ADDITIONAL", 2) == (2,)
    assert evaluation.scored_horizons_for_episode("ADDITIONAL", 4) == (4,)
    assert evaluation.scored_horizons_for_episode("PB", 5) == (2, 3, 4, 5)

    with pytest.raises(evaluation.RefinerEvaluationError, match="horizon"):
        evaluation.scored_horizons_for_episode("ADDITIONAL", 5)


def test_refiner_replication_clis_accept_isolated_outputs_and_seed() -> None:
    preparation = _preparation_module()
    training = _training_module()

    prepare_arguments = preparation._parser().parse_args(
        [
            "--variant",
            "C0",
            "--update",
            "3000",
            "--output-root",
            "/tmp/refiner-data",
            "--manifest-output",
            "/tmp/refiner-data.json",
        ]
    )
    train_arguments = training._parser().parse_args(
        [
            "--cache",
            "/tmp/refiner-data.pt",
            "--output",
            "/tmp/refiner-model",
            "--seed",
            "46",
        ]
    )

    assert prepare_arguments.output_root == Path("/tmp/refiner-data")
    assert prepare_arguments.manifest_output == Path("/tmp/refiner-data.json")
    assert train_arguments.seed == 46


def _occurrence(
    *,
    window: int,
    query: int,
    logical_id: int = 9,
    generation: int = 2,
    source_class_id: int = 4,
) -> SoftOccurrence:
    return SoftOccurrence(
        scan_id="scan-a",
        window_index=window,
        logical_id=logical_id,
        generation=generation,
        source_class_id=source_class_id,
        source_query_id=query,
        candidate_index=0,
    )


def test_soft_pairing_uses_exact_identity_class_and_unique_adjacent_old() -> None:
    new = _occurrence(window=3, query=77)
    valid_old = _occurrence(window=2, query=1)
    non_adjacent = _occurrence(window=1, query=2)
    wrong_generation = _occurrence(window=2, query=3, generation=3)

    decisions = pair_adjacent_soft_occurrences(
        old_occurrences=[valid_old, non_adjacent, wrong_generation],
        new_occurrences=[new],
    )

    assert decisions[0].old_index == 0
    assert decisions[0].reason == "PAIRED_UNIQUE_ADJACENT"
    duplicate = pair_adjacent_soft_occurrences(
        old_occurrences=[valid_old, _occurrence(window=2, query=99)],
        new_occurrences=[new],
    )
    assert duplicate[0].old_index is None
    assert duplicate[0].reason == "FALLBACK_MULTIPLE_ADJACENT_OLD"
    missing = pair_adjacent_soft_occurrences(
        old_occurrences=[non_adjacent],
        new_occurrences=[new],
    )
    assert missing[0].old_index is None
    assert missing[0].reason == "FALLBACK_NO_ADJACENT_OLD"


def test_refiner_is_135d_bounded_zero_noop_and_detaches_parent_inputs() -> None:
    refiner = CausalMaskRefiner()
    segment_count = 5
    features = torch.randn(segment_count, 128, requires_grad=True)
    old_logits = torch.randn(segment_count, requires_grad=True)
    new_logits = torch.randn(segment_count, requires_grad=True)
    old_score = torch.tensor(0.6, requires_grad=True)
    new_score = torch.tensor(0.8, requires_grad=True)

    refined, delta = refiner(
        new_features=features,
        old_logits=old_logits,
        new_logits=new_logits,
        old_score=old_score,
        new_score=new_score,
    )

    assert refiner.input_dim == 135
    assert refiner.feature_norm.elementwise_affine is False
    assert refined.shape == delta.shape == (segment_count,)
    assert torch.equal(refined, new_logits.detach())
    assert torch.count_nonzero(delta).item() == 0
    assert all(parameter.requires_grad for parameter in refiner.parameters())

    with torch.no_grad():
        refiner.output.weight.fill_(100.0)
        refiner.output.bias.fill_(100.0)
    refined, delta = refiner(
        new_features=features,
        old_logits=old_logits,
        new_logits=new_logits,
        old_score=old_score,
        new_score=new_score,
    )
    refined.sum().backward()

    assert torch.all(delta <= 2.0) and torch.all(delta >= -2.0)
    assert features.grad is None
    assert old_logits.grad is None
    assert new_logits.grad is None
    assert old_score.grad is None
    assert new_score.grad is None
    assert refiner.input.weight.grad is not None
    assert refiner.output.weight.grad is not None


def test_old_probabilities_align_by_vertex_id_then_new_segment_partition() -> None:
    training = _training_module()

    old_logits = training.align_old_probabilities_to_new_segments(
        old_vertex_ids=torch.tensor([30, 10, 20, 40]),
        old_point_probabilities=torch.tensor([0.8, 0.2, 0.6, 0.4]),
        new_vertex_ids=torch.tensor([10, 20, 30, 40]),
        new_low_point2segment=torch.tensor([0, 1, 2]),
        new_voxel_inverse=torch.tensor([0, 0, 1, 2]),
        new_segment_ids=torch.tensor([0, 1, 2]),
    )

    expected_probabilities = torch.tensor([0.4, 0.8, 0.4])
    torch.testing.assert_close(old_logits.sigmoid(), expected_probabilities)
    with pytest.raises(training.RefinerTrainingError, match="vertex"):
        training.align_old_probabilities_to_new_segments(
            old_vertex_ids=torch.tensor([10, 20]),
            old_point_probabilities=torch.tensor([0.2, 0.8]),
            new_vertex_ids=torch.tensor([10, 30]),
            new_low_point2segment=torch.tensor([0, 1]),
            new_voxel_inverse=torch.tensor([0, 1]),
            new_segment_ids=torch.tensor([0, 1]),
        )


def test_refiner_loss_and_schedule_match_frozen_contract() -> None:
    training = _training_module()
    refined_logits = torch.tensor([2.0, -1.0, 0.5], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0])
    delta = torch.tensor([0.2, -0.1, 0.0], requires_grad=True)

    loss, parts = training.refiner_candidate_loss(
        refined_logits=refined_logits,
        targets=targets,
        delta=delta,
    )

    loss.backward()
    assert set(parts) == {"bce", "dice", "delta_regularization"}
    assert loss.item() == pytest.approx(sum(parts.values()))
    assert refined_logits.grad is not None
    assert delta.grad is not None
    assert training.refiner_lr_multiplier(0) == pytest.approx(1 / 75)
    assert training.refiner_lr_multiplier(74) == pytest.approx(1.0)
    assert training.refiner_lr_multiplier(1499) == pytest.approx(0.1)


def test_refiner_sampling_is_reference_episode_candidate_uniform_and_exact_resume() -> (
    None
):
    training = _training_module()
    records = [
        {
            "reference_id": reference,
            "episode_id": episode,
            "candidate_id": f"{reference}-{episode}-{candidate}",
        }
        for reference, episode_count, candidate_count in (
            ("r0", 2, 3),
            ("r1", 1, 1),
            ("r2", 3, 2),
        )
        for episode_index in range(episode_count)
        for episode in (f"e{episode_index}",)
        for candidate in range(candidate_count)
    ]

    first = training.sample_refiner_candidates(records, count=9, seed=45, cursor=0)
    resumed = training.sample_refiner_candidates(records, count=4, seed=45, cursor=5)

    assert first[5:] == resumed
    assert {row["reference_id"] for row in first[:3]} == {"r0", "r1", "r2"}


def _training_record(reference: str, candidate: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(100 + candidate)
    segment_count = 6
    return {
        "reference_id": reference,
        "episode_id": f"episode-{candidate % 2}",
        "candidate_id": f"{reference}-{candidate}",
        "new_features": torch.randn(segment_count, 128, generator=generator),
        "old_logits": torch.randn(segment_count, generator=generator),
        "new_logits": torch.randn(segment_count, generator=generator),
        "old_score": 0.4,
        "new_score": 0.7,
        "target": torch.randint(0, 2, (segment_count,), generator=generator).float(),
    }


def test_refiner_training_exact_resume_matches_uninterrupted_cpu(
    tmp_path: Path,
) -> None:
    training = _training_module()
    records = tuple(
        _training_record(reference, candidate)
        for candidate, reference in enumerate(("r0", "r1", "r2", "r0", "r1", "r2"))
    )
    shard = tmp_path / "soft-training.pt"
    torch.save(records, shard)

    full = training.train_mask_refiner(
        cache_paths=(shard,),
        output_dir=tmp_path / "full",
        stop_after_updates=3,
        device="cpu",
        batch_candidates=2,
        maximum_segments=4,
    )
    partial = training.train_mask_refiner(
        cache_paths=(shard,),
        output_dir=tmp_path / "resume",
        stop_after_updates=1,
        device="cpu",
        batch_candidates=2,
        maximum_segments=4,
    )
    resumed = training.train_mask_refiner(
        cache_paths=(shard,),
        output_dir=tmp_path / "resume",
        stop_after_updates=3,
        resume=tmp_path / "resume/last.ckpt",
        device="cpu",
        batch_candidates=2,
        maximum_segments=4,
    )

    full_checkpoint = torch.load(
        tmp_path / "full/update=0003.ckpt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        tmp_path / "resume/update=0003.ckpt", map_location="cpu", weights_only=False
    )
    assert full["status"] == resumed["status"] == "SMOKE"
    assert partial["completed_updates"] == 1
    assert len(resumed["training_audit"]) == 2
    for name, value in full_checkpoint["refiner_state_dict"].items():
        torch.testing.assert_close(
            value, resumed_checkpoint["refiner_state_dict"][name]
        )


def _soft_prediction(
    *,
    segment_logits: torch.Tensor,
    source_query_id: int,
    source_class_id: int,
    temporal_stages: torch.Tensor,
) -> OfficialTaskPrediction:
    segment_count = segment_logits.shape[0]
    evidence = OfficialTaskSoftEvidence(
        representation="FP32_SEGMENT_LOGITS",
        segment_logits=segment_logits.float(),
        low_resolution_logits=segment_logits.float(),
        full_resolution_logits=segment_logits.float(),
        full_resolution_probabilities=segment_logits.float().sigmoid(),
        class_probabilities=torch.ones((1, 19), dtype=torch.float32) / 19,
        query_features=torch.zeros((1, 128), dtype=torch.float32),
        segment_features=torch.zeros((segment_count, 128), dtype=torch.float32),
        source_query_ids=torch.tensor([source_query_id]),
        source_class_ids=torch.tensor([source_class_id]),
        low_point2segment=torch.arange(segment_count),
        voxel_inverse=torch.arange(segment_count),
        full_point2segment=torch.arange(segment_count),
    )
    masks = segment_logits > 0
    latest_stage = int(temporal_stages.max().item())
    return OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.7]),
        pred_classes=torch.tensor([source_class_id]),
        source_query_ids=evidence.source_query_ids,
        source_class_ids=evidence.source_class_ids,
        temporal_stages=temporal_stages,
        latest_stage_index=latest_stage,
        latest_stage_masks=masks[temporal_stages == latest_stage],
        soft_evidence=evidence,
    )


def _stage_meta(
    *,
    absolute_stage: int,
    scan_ids: tuple[str, ...],
    vertex_ids: tuple[tuple[int, ...], ...],
) -> StageMeta:
    point_counts = [len(values) for values in vertex_ids]
    offsets = [0]
    stages = []
    for local_stage, count in enumerate(point_counts):
        offsets.append(offsets[-1] + count)
        stages.extend([local_stage] * count)
    point_count = offsets[-1]
    mapping = torch.arange(point_count)
    return StageMeta(
        reference_id="reference",
        episode_id="episode",
        scan_ids_in_window=scan_ids,
        absolute_stage_index=absolute_stage,
        local_stage_ids=torch.tensor(stages),
        original_vertex_ids=tuple(torch.tensor(values) for values in vertex_ids),
        scan_vertex_offsets=torch.tensor(offsets),
        point2segment=mapping,
        segment_stage_ids=torch.tensor(stages),
        augmentation_transform_id="fixed-transform",
        coordinate_frame_id="reference",
        voxel_inverse=mapping,
        full_resolution_point2segment=mapping.clone(),
    )


def test_revision_transform_pairs_frozen_identity_and_materializes_refined_logits() -> (
    None
):
    training = _training_module()
    system = type(
        "System",
        (),
        {"_get_full_res_mask": staticmethod(lambda masks, inverse, _: masks[inverse])},
    )()
    refiner = CausalMaskRefiner()
    transform = training.CausalRefinerRevisionTransform(
        system=system,
        refiner=refiner,
        device="cpu",
    )
    old_prediction = _soft_prediction(
        segment_logits=torch.tensor([[2.0], [-2.0]]),
        source_query_id=5,
        source_class_id=4,
        temporal_stages=torch.tensor([0, 0]),
    )
    new_prediction = _soft_prediction(
        segment_logits=torch.tensor([[-0.5], [-0.5], [1.0], [1.0]]),
        source_query_id=9,
        source_class_id=4,
        temporal_stages=torch.tensor([0, 0, 1, 1]),
    )
    old_meta = _stage_meta(
        absolute_stage=0,
        scan_ids=("scan-a",),
        vertex_ids=((20, 10),),
    )
    new_meta = _stage_meta(
        absolute_stage=1,
        scan_ids=("scan-a", "scan-b"),
        vertex_ids=((10, 20), (30, 40)),
    )
    transform.observe_stage(old_prediction, old_meta)
    transform.observe_stage(new_prediction, new_meta)
    identity = PublishedIdentity("track", 2, 4)
    request = RevisionMaskRequest(
        scan_id="scan-a",
        absolute_stage_index=1,
        source_vertex_ids=torch.tensor([10, 20]),
        target_vertex_ids=torch.tensor([20, 10]),
        augmentation_transform_id="fixed-transform",
        coordinate_frame_id="reference",
        old_candidates=(
            RevisionCandidate(0, identity, 5, 0.4, torch.tensor([True, False])),
        ),
        new_candidates=(
            RevisionCandidate(0, identity, 9, 0.7, torch.tensor([False, False])),
        ),
    )

    no_op = transform(request)
    assert no_op[0].tolist() == [False, False]
    with torch.no_grad():
        refiner.output.bias.fill_(100.0)
    refined = transform(request)
    assert refined[0].tolist() == [True, True]
    assert transform.last_audit["paired_candidate_count"] == 1
    assert transform.last_audit["fallback_candidate_count"] == 0
    assert len(transform.last_candidate_records) == 1
    record = transform.last_candidate_records[0]
    assert record["candidate_index"] == 0
    assert record["source_query_id"] == 9
    assert record["source_class_id"] == 4
    assert record["new_features"].shape == (2, 128)
    torch.testing.assert_close(record["new_logits"], torch.tensor([-0.5, -0.5]))

    wrong_class = RevisionMaskRequest(
        **{
            **request.__dict__,
            "old_candidates": (
                RevisionCandidate(
                    0,
                    PublishedIdentity("track", 2, 4),
                    5,
                    0.4,
                    torch.tensor([True, False]),
                ),
            ),
        }
    )
    old_prediction.soft_evidence.source_class_ids[0] = 8
    transform.observe_stage(old_prediction, old_meta)
    transform.observe_stage(new_prediction, new_meta)
    assert transform(wrong_class) == {}
    assert transform.last_audit["fallback_candidate_count"] == 1


def test_refiner_gt_matching_is_class_compatible_one_to_one_and_skips_ties() -> None:
    training = _training_module()
    matched = training.match_refiner_candidates_to_gt(
        candidate_masks=torch.tensor(
            [
                [True, False, True],
                [True, False, False],
                [False, True, True],
                [False, True, False],
            ]
        ),
        candidate_classes=torch.tensor([1, 1, 2]),
        gt_masks=torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        ),
        gt_classes=torch.tensor([1, 1]),
        iou_threshold=0.1,
    )

    assert matched["assignments"] == (0, 1, None)
    assert matched["ambiguous_candidate_indices"] == ()
    assert matched["matched_candidate_count"] == 2
    assert matched["unmatched_candidate_count"] == 1

    ambiguous = training.match_refiner_candidates_to_gt(
        candidate_masks=torch.tensor([[True, True], [True, True], [False, False]]),
        candidate_classes=torch.tensor([1, 1]),
        gt_masks=torch.tensor([[True, True, False]]),
        gt_classes=torch.tensor([1]),
        iou_threshold=0.1,
    )
    assert ambiguous["assignments"] == (None, None)
    assert ambiguous["ambiguous_candidate_indices"] == (0, 1)


def test_training_records_add_only_segment_targets_and_drop_ambiguous_candidates() -> (
    None
):
    training = _training_module()
    prediction = _soft_prediction(
        segment_logits=torch.tensor([[2.0], [-2.0], [1.0], [-1.0]]),
        source_query_id=3,
        source_class_id=4,
        temporal_stages=torch.tensor([0, 0, 1, 1]),
    )
    meta = _stage_meta(
        absolute_stage=1,
        scan_ids=("scan-a", "scan-b"),
        vertex_ids=((10, 20), (30, 40)),
    )
    pair_record = {
        "reference_id": "reference",
        "episode_id": "episode",
        "candidate_id": "candidate-0",
        "candidate_index": 0,
        "segment_ids": torch.tensor([0, 1]),
        "new_features": torch.zeros((2, 128)),
        "old_logits": torch.tensor([0.5, -0.5]),
        "new_logits": torch.tensor([2.0, -2.0]),
        "old_score": 0.4,
        "new_score": 0.7,
    }
    records, audit = training.build_refiner_training_records(
        pair_records=(pair_record,),
        new_prediction=prediction,
        full_target={
            "masks": torch.tensor([[True, False, False, False]]),
            "labels": torch.tensor([4]),
        },
        stage_meta=meta,
    )

    assert len(records) == 1
    assert records[0]["target"].tolist() == [1.0, 0.0]
    assert "masks" not in records[0]
    assert "labels" not in records[0]
    assert audit == {
        "paired_candidate_count": 1,
        "written_candidate_count": 1,
        "matched_candidate_count": 1,
        "empty_target_candidate_count": 0,
        "ambiguous_candidate_count": 0,
    }


def test_refiner_episode_inventory_uses_first_master_and_canonical_reverse_only() -> (
    None
):
    preparation = _preparation_module()
    inventory = preparation.build_refiner_episode_inventory(
        sequence_names=(
            "scene0002_00-scene0002_01-scene0002_02-scene0002_03",
            "scene0001_02-scene0001_01-scene0001_00",
            "scene0001_00-scene0001_01-scene0001_02",
            "scene0003_00-scene0003_01",
        ),
        sequence_indices=((20, 21, 22, 23), (12, 11, 10), (10, 11, 12), (30, 31)),
        reference_ids=("r2", "r1", "r1", "r3"),
        reference_limit=32,
        minimum_references=2,
    )

    assert inventory["status"] == "PASS"
    assert inventory["reference_count"] == 2
    assert len(inventory["episodes"]) == 4
    r1 = [row for row in inventory["episodes"] if row["reference_id"] == "r1"]
    assert [row["order_id"] for row in r1] == ["canonical", "reverse"]
    assert r1[0]["source_context_index"] == 2
    assert r1[0]["scan_ids"] == ("scene0001_00", "scene0001_01", "scene0001_02")
    assert r1[1]["scan_ids"] == tuple(reversed(r1[0]["scan_ids"]))
    assert r1[1]["scan_indices"] == tuple(reversed(r1[0]["scan_indices"]))


def test_new_only_is_invariant_to_old_evidence_after_nonzero_training_weights():
    torch.manual_seed(45)
    new_only = CausalMaskRefiner(input_mode="NEW_ONLY")
    pair = CausalMaskRefiner(input_mode="OLD_NEW")
    with torch.no_grad():
        new_only.output.weight.normal_(std=0.1)
    pair.load_state_dict(new_only.state_dict())
    inputs = {
        "new_features": torch.randn(6, 128),
        "new_logits": torch.randn(6),
        "new_score": 0.8,
    }
    first = new_only(**inputs, old_logits=torch.randn(6), old_score=0.1)[0]
    changed = new_only(
        **inputs, old_logits=torch.full((6,), float("nan")), old_score=999.0
    )[0]
    control = pair(**inputs, old_logits=inputs["new_logits"], old_score=0.8)[0]
    assert torch.equal(first, changed)
    assert torch.equal(first, control)


def test_v2_refiner_checkpoint_loads_mode_and_rejects_parent_mismatch(tmp_path):
    from scripts.perception_refiner_evaluation import (
        _load_refiner,
        RefinerEvaluationError,
    )

    training = _training_module()
    shard = tmp_path / "train.pt"
    torch.save((_training_record("r0", 0),), shard)
    binding = training.refiner_v2_binding(
        input_mode="NEW_ONLY",
        parent_recipe_hash="a" * 64,
        parent_weight_hash="b" * 64,
        cache_paths=(shard,),
    )
    summary = training.train_mask_refiner(
        cache_paths=(shard,),
        output_dir=tmp_path / "head",
        stop_after_updates=1,
        batch_candidates=1,
        maximum_segments=4,
        device="cpu",
        binding=binding,
    )
    path = tmp_path / "head/update=0000.ckpt"
    refiner, audit = _load_refiner(
        path, optimizer_update=0, device="cpu", expected_binding=binding
    )
    assert refiner.input_mode == "NEW_ONLY"
    assert audit["binding"] == binding
    assert len(summary["loss_curve"]) == 1
    with pytest.raises(RefinerEvaluationError, match="binding"):
        _load_refiner(
            path,
            optimizer_update=0,
            device="cpu",
            expected_binding={**binding, "parent_weight_hash": "c" * 64},
        )


def test_refiner_producer_advances_the_frozen_d0_last_identity_path() -> None:
    from models.task_memory_routing import PredictionObservation

    preparation = _preparation_module()
    observation = PredictionObservation(
        features=torch.tensor([[[1.0, 0.0]]]),
        class_prob=torch.tensor([[[0.8, 0.2]]]),
        confidence=torch.tensor([[0.9]]),
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[True]]),
        previous_supported=torch.tensor([[False]]),
    )
    state, identity_map = preparation.advance_d0_identity(
        observation=observation,
        state=None,
        stage_meta=_stage_meta(
            absolute_stage=0,
            scan_ids=("scan-a",),
            vertex_ids=((10,),),
        ),
    )

    assert state.config.update_mode == "last"
    assert state.config.class_weight == pytest.approx(0.25)
    assert state.config.association_threshold == pytest.approx(0.5)
    assert identity_map == {0: (0, 0)}

    second = PredictionObservation(
        features=torch.tensor([[[0.8, 0.6]]]),
        class_prob=torch.tensor([[[0.8, 0.2]]]),
        confidence=torch.tensor([[0.9]]),
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[True]]),
        previous_supported=torch.tensor([[True]]),
    )
    state, identity_map = preparation.advance_d0_identity(
        observation=second,
        state=state,
        stage_meta=_stage_meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            vertex_ids=((10,), (20,)),
        ),
    )

    assert identity_map == {0: (0, 0)}
    torch.testing.assert_close(state.embedding[0, 0], torch.tensor([0.8, 0.6]))


def test_refiner_evaluation_allows_only_mask_changes() -> None:
    evaluation = _evaluation_module()
    parent = {
        "pred_masks": torch.tensor([[True, False], [False, True]]),
        "pred_scores": torch.tensor([0.8, 0.6]),
        "pred_classes": torch.tensor([4, 7]),
    }
    refined = {
        **parent,
        "pred_masks": torch.tensor([[True, True], [False, True]]),
    }

    audit = evaluation.validate_mask_only_refiner_output(parent, refined)

    assert audit == {
        "candidate_count": 2,
        "changed_candidate_count": 1,
        "changed_point_count": 1,
    }
    with pytest.raises(evaluation.RefinerEvaluationError, match="scores or classes"):
        evaluation.validate_mask_only_refiner_output(
            parent,
            {**refined, "pred_scores": torch.tensor([0.9, 0.6])},
        )


def test_refiner_live_targets_are_appended_in_exact_causal_order() -> None:
    evaluation = _evaluation_module()
    raw = {
        "target": {
            "gt_ids": torch.tensor([7]),
            "gt_classes": torch.tensor([2]),
            "gt_masks": torch.tensor([[True, False]]),
        }
    }

    targets = evaluation.append_live_refiner_target((), raw, stage_index=0)

    assert len(targets) == 1
    assert targets[0]["gt_ids"].tolist() == [7]
    raw["target"]["gt_ids"][0] = 99
    assert targets[0]["gt_ids"].tolist() == [7]
    with pytest.raises(evaluation.RefinerEvaluationError, match="causal order"):
        evaluation.append_live_refiner_target(targets, raw, stage_index=2)
