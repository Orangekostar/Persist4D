from __future__ import annotations

import inspect

import torch

from datasets.task_memory_episode import StageMeta
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.task_memory_output import (
    CommitZeroPublisher,
    LagOnePublisher,
    PublishedIdentity,
)


def test_vertex_alignment_preserves_exact_reordering_and_independent_storage():
    from scripts.task_memory_output import _align_masks

    for source, target in (
        ([10, 20, 30], [10, 20, 30]),
        ([30, 10, 20], [10, 20, 30]),
        ([10, 10, 20], [10, 10, 20]),
    ):
        mask = torch.tensor([True, False, True])
        positions = {value: index for index, value in enumerate(source)}
        expected = mask[torch.tensor([positions[value] for value in target])]
        actual = _align_masks(
            source_vertex_ids=torch.tensor(source),
            target_vertex_ids=torch.tensor(target),
            mask=mask,
        )
        assert torch.equal(actual, expected)
        assert actual.data_ptr() != mask.data_ptr()


def _meta(
    *,
    absolute_stage: int,
    scan_ids: tuple[str, ...],
    point_counts: tuple[int, ...],
    vertex_ids: tuple[tuple[int, ...], ...] | None = None,
) -> StageMeta:
    if vertex_ids is None:
        vertex_ids = tuple(tuple(range(count)) for count in point_counts)
    offsets = [0]
    local_stages = []
    original_vertex_ids = []
    for local_stage, (count, ids) in enumerate(zip(point_counts, vertex_ids)):
        offsets.append(offsets[-1] + count)
        local_stages.extend([local_stage] * count)
        original_vertex_ids.append(torch.tensor(ids, dtype=torch.long))
    point_count = offsets[-1]
    point2segment = torch.arange(point_count, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=scan_ids,
        absolute_stage_index=absolute_stage,
        local_stage_ids=torch.tensor(local_stages, dtype=torch.long),
        original_vertex_ids=tuple(original_vertex_ids),
        scan_vertex_offsets=torch.tensor(offsets, dtype=torch.long),
        point2segment=point2segment,
        segment_stage_ids=torch.tensor(local_stages, dtype=torch.long),
        augmentation_transform_id="augmentation-0",
        coordinate_frame_id="reference-0",
        voxel_inverse=torch.arange(point_count, dtype=torch.long),
        full_resolution_point2segment=point2segment.clone(),
    )


def _prediction(
    *,
    masks: list[list[bool]],
    scores: list[float],
    classes: list[int],
    queries: list[int],
    stages: list[int],
) -> OfficialTaskPrediction:
    mask_tensor = torch.tensor(masks, dtype=torch.bool)
    stage_tensor = torch.tensor(stages, dtype=torch.long)
    latest_stage = max(stages)
    result = OfficialTaskPrediction(
        pred_masks=mask_tensor,
        pred_scores=torch.tensor(scores, dtype=torch.float32),
        pred_classes=torch.tensor(classes, dtype=torch.long),
        source_query_ids=torch.tensor(queries, dtype=torch.long),
        source_class_ids=torch.tensor(classes, dtype=torch.long),
        temporal_stages=stage_tensor,
        latest_stage_index=latest_stage,
        latest_stage_masks=mask_tensor[stage_tensor == latest_stage],
    )
    result.validate()
    return result


def _scan_masks(prefix, scan_id: str) -> dict[PublishedIdentity, torch.Tensor]:
    scan_index = prefix.scan_ids.index(scan_id)
    start = int(prefix.scan_vertex_offsets[scan_index].item())
    stop = int(prefix.scan_vertex_offsets[scan_index + 1].item())
    return {
        key: prefix.prediction["pred_masks"][start:stop, column]
        for column, key in enumerate(prefix.keys)
        if prefix.prediction["pred_masks"][start:stop, column].any().item()
    }


def test_lag1_revises_only_previous_scan_and_freezes_older_scans() -> None:
    publisher = LagOnePublisher(score_reducer="mean")
    identity = PublishedIdentity("track-7", 3, 4)

    first = publisher.update(
        _prediction(
            masks=[[True], [False], [False]],
            scores=[0.3],
            classes=[4],
            queries=[0],
            stages=[0, 0, 0],
        ),
        {0: ("track-7", 3)},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(3,)),
    )
    assert first.provisional_scan_id == "scan-a"
    assert first.archive == ()

    second = publisher.update(
        _prediction(
            masks=[
                [False],
                [True],
                [False],
                [True],
                [False],
            ],
            scores=[0.7],
            classes=[4],
            queries=[0],
            stages=[0, 0, 0, 1, 1],
        ),
        {0: ("track-7", 3)},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(3, 2),
        ),
    )
    frozen_a = _scan_masks(second, "scan-a")[identity].clone()
    assert frozen_a.tolist() == [False, True, False]
    assert second.provisional_scan_id == "scan-b"
    assert [record.scan_id for record in second.archive] == ["scan-a"]
    assert second.revision_log[-1].scan_id == "scan-a"

    third = publisher.update(
        _prediction(
            masks=[[False], [True], [True], [True]],
            scores=[0.9],
            classes=[4],
            queries=[0],
            stages=[0, 0, 1, 1],
        ),
        {0: ("track-7", 3)},
        _meta(
            absolute_stage=2,
            scan_ids=("scan-b", "scan-c"),
            point_counts=(2, 2),
        ),
    )

    assert torch.equal(_scan_masks(third, "scan-a")[identity], frozen_a)
    assert _scan_masks(third, "scan-b")[identity].tolist() == [False, True]
    assert [record.scan_id for record in third.archive] == ["scan-a", "scan-b"]
    assert [record.scan_id for record in third.revision_log] == ["scan-a", "scan-b"]
    assert third.provisional_scan_id == "scan-c"


def test_lag1_whole_stage_replacement_keeps_previous_only_candidates() -> None:
    publisher = LagOnePublisher()
    publisher.update(
        _prediction(
            masks=[[True, False], [False, True]],
            scores=[0.8, 0.7],
            classes=[2, 5],
            queries=[0, 1],
            stages=[0, 0],
        ),
        {0: ("kept", 0), 1: ("removed", 0)},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(2,)),
    )
    prefix = publisher.update(
        _prediction(
            masks=[
                [True, False],
                [False, False],
                [False, True],
                [False, False],
            ],
            scores=[0.6, 0.9],
            classes=[2, 7],
            queries=[0, 2],
            stages=[0, 0, 1, 1],
        ),
        {0: ("kept", 0), 2: ("current", 0)},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(2, 2),
        ),
    )

    scan_a = _scan_masks(prefix, "scan-a")
    scan_b = _scan_masks(prefix, "scan-b")
    assert PublishedIdentity("kept", 0, 2) in scan_a
    assert PublishedIdentity("kept", 0, 2) not in scan_b
    assert PublishedIdentity("removed", 0, 5) not in prefix.keys
    assert PublishedIdentity("current", 0, 7) in scan_b


def test_route_precedes_iou_fallback_and_reserves_conflicting_anchor() -> None:
    publisher = LagOnePublisher()
    first = publisher.update(
        _prediction(
            masks=[[True], [True], [False], [False]],
            scores=[0.8],
            classes=[3],
            queries=[10],
            stages=[0, 0, 0, 0],
        ),
        {},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(4,)),
    )
    old_identity = next(iter(_scan_masks(first, "scan-a")))

    second = publisher.update(
        _prediction(
            masks=[
                [True, True],
                [True, True],
                [False, False],
                [False, False],
                [True, False],
            ],
            scores=[0.9, 0.7],
            classes=[3, 3],
            queries=[1, 2],
            stages=[0, 0, 0, 0, 1],
        ),
        {1: ("routed", 5)},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(4, 1),
        ),
    )

    scan_a = _scan_masks(second, "scan-a")
    assert PublishedIdentity("routed", 5, 3) in scan_a
    assert old_identity not in scan_a
    assert len(scan_a) == 2
    assert second.revision_log[-1].route_conflicts == 1
    assert second.revision_log[-1].fallback_matches == 0


def test_iou_fallback_is_class_compatible_thresholded_and_stable_on_ties() -> None:
    publisher = LagOnePublisher(iou_threshold=0.5)
    first = publisher.update(
        _prediction(
            masks=[
                [True, True],
                [True, True],
                [False, False],
                [False, False],
            ],
            scores=[0.8, 0.7],
            classes=[3, 3],
            queries=[10, 20],
            stages=[0, 0, 0, 0],
        ),
        {},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(4,)),
    )
    old_keys = tuple(_scan_masks(first, "scan-a"))

    second = publisher.update(
        _prediction(
            masks=[
                [True, True, True, True],
                [True, True, False, True],
                [False, False, True, False],
                [False, False, True, False],
                [False, False, False, False],
            ],
            scores=[0.6, 0.5, 0.4, 0.3],
            classes=[3, 3, 3, 8],
            queries=[1, 2, 3, 4],
            stages=[0, 0, 0, 0, 1],
        ),
        {},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(4, 1),
        ),
    )
    scan_a = _scan_masks(second, "scan-a")

    assert old_keys[0] in scan_a
    assert old_keys[1] in scan_a
    assert [
        (candidate.source_query_id, candidate.identity)
        for candidate in second.archive[0].candidates[:2]
    ] == [(1, old_keys[0]), (2, old_keys[1])]
    assert second.revision_log[-1].fallback_matches == 2
    unmatched = [
        key for key in scan_a if key not in old_keys and key.class_id in {3, 8}
    ]
    assert {key.class_id for key in unmatched} == {3, 8}


def test_iou_fallback_accepts_the_frozen_half_overlap_threshold() -> None:
    publisher = LagOnePublisher(iou_threshold=0.5)
    first = publisher.update(
        _prediction(
            masks=[[True], [True], [False], [False]],
            scores=[0.8],
            classes=[3],
            queries=[4],
            stages=[0, 0, 0, 0],
        ),
        {},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(4,)),
    )
    old_identity = next(iter(_scan_masks(first, "scan-a")))

    second = publisher.update(
        _prediction(
            masks=[[True], [True], [True], [True], [False]],
            scores=[0.6],
            classes=[3],
            queries=[9],
            stages=[0, 0, 0, 0, 1],
        ),
        {},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(4, 1),
        ),
    )

    assert next(iter(_scan_masks(second, "scan-a"))) == old_identity
    assert second.revision_log[-1].fallback_matches == 1


def test_duplicate_route_keeps_best_candidate_and_emits_ephemeral_duplicate() -> None:
    publisher = LagOnePublisher()
    prefix = publisher.update(
        _prediction(
            masks=[[True, False], [False, True]],
            scores=[0.4, 0.9],
            classes=[6, 6],
            queries=[1, 2],
            stages=[0, 0],
        ),
        {1: ("shared", 2), 2: ("shared", 2)},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(2,)),
    )

    candidates = prefix.prediction["pred_masks"]
    routed = PublishedIdentity("shared", 2, 6)
    assert candidates[:, prefix.keys.index(routed)].tolist() == [False, True]
    assert len(prefix.keys) == 2
    assert sum(key.logical_id == "shared" for key in prefix.keys) == 1
    assert any(str(key.logical_id).startswith("ephemeral:") for key in prefix.keys)


def test_commit0_never_revises_past_and_lag1_reports_bounded_bytes() -> None:
    commit0 = CommitZeroPublisher()
    commit0.update(
        _prediction(
            masks=[[True], [False]],
            scores=[0.4],
            classes=[2],
            queries=[0],
            stages=[0, 0],
        ),
        {0: ("track", 0)},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(2,)),
    )
    prefix = commit0.update(
        _prediction(
            masks=[[False], [True], [True]],
            scores=[0.8],
            classes=[2],
            queries=[0],
            stages=[0, 0, 1],
        ),
        {0: ("track", 0)},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(2, 1),
        ),
    )
    identity = PublishedIdentity("track", 0, 2)
    assert _scan_masks(prefix, "scan-a")[identity].tolist() == [True, False]
    assert prefix.provisional_scan_id is None
    assert prefix.revision_log == ()

    lag1 = LagOnePublisher()
    lag1_prefix = lag1.update(
        _prediction(
            masks=[[True], [False], [True]],
            scores=[0.6],
            classes=[2],
            queries=[0],
            stages=[0, 0, 0],
        ),
        {0: ("track", 0)},
        _meta(absolute_stage=0, scan_ids=("scan-z",), point_counts=(3,)),
    )
    assert lag1_prefix.provisional_scan_id == "scan-z"
    assert lag1_prefix.accounting.archive_payload_bytes == 0
    assert lag1_prefix.accounting.lag1_buffer_bytes > 0
    assert lag1_prefix.accounting.materialized_output_bytes > 0


def test_publisher_contract_has_no_ground_truth_input() -> None:
    parameter_names = set(inspect.signature(LagOnePublisher.update).parameters)
    forbidden = {"target", "targets", "ground_truth", "gt", "labels"}
    assert parameter_names.isdisjoint(forbidden)
    assert tuple(inspect.signature(LagOnePublisher.update).parameters) == (
        "self",
        "prediction",
        "identity_map",
        "stage_meta",
    )


def test_lag1_revision_transform_changes_archive_only_after_identity_resolution() -> (
    None
):
    requests = []

    def transform(request):
        requests.append(request)
        assert request.scan_id == "scan-a"
        assert request.old_candidates[0].source_query_id == 5
        assert request.new_candidates[0].source_query_id == 9
        assert request.old_candidates[0].identity == request.new_candidates[0].identity
        return {0: torch.tensor([True, False, True])}

    parent = LagOnePublisher()
    refined = LagOnePublisher(revision_mask_transform=transform)
    first_prediction = _prediction(
        masks=[[True], [True], [False]],
        scores=[0.4],
        classes=[4],
        queries=[5],
        stages=[0, 0, 0],
    )
    first_meta = _meta(
        absolute_stage=0,
        scan_ids=("scan-a",),
        point_counts=(3,),
    )
    for publisher in (parent, refined):
        publisher.update(first_prediction, {5: ("track", 2)}, first_meta)

    second_prediction = _prediction(
        masks=[[False], [True], [False], [True], [False]],
        scores=[0.8],
        classes=[4],
        queries=[9],
        stages=[0, 0, 0, 1, 1],
    )
    second_meta = _meta(
        absolute_stage=1,
        scan_ids=("scan-a", "scan-b"),
        point_counts=(3, 2),
    )
    parent_prefix = parent.update(second_prediction, {9: ("track", 2)}, second_meta)
    refined_prefix = refined.update(second_prediction, {9: ("track", 2)}, second_meta)
    identity = PublishedIdentity("track", 2, 4)

    assert len(requests) == 1
    assert _scan_masks(parent_prefix, "scan-a")[identity].tolist() == [
        False,
        True,
        False,
    ]
    assert _scan_masks(refined_prefix, "scan-a")[identity].tolist() == [
        True,
        False,
        True,
    ]
    assert refined.revision_pair[1].candidates[0].mask.tolist() == [False, True, False]
    assert refined.current_dense_scan.candidates[0].mask.tolist() == [True, False]
    parent_current = parent.current_dense_scan.candidates[0]
    refined_current = refined.current_dense_scan.candidates[0]
    assert parent_current.candidate_index == refined_current.candidate_index
    assert parent_current.identity == refined_current.identity
    assert parent_current.source_query_id == refined_current.source_query_id
    assert parent_current.score == refined_current.score
    assert torch.equal(parent_current.mask, refined_current.mask)
    assert parent_prefix.keys == refined_prefix.keys
    torch.testing.assert_close(
        parent_prefix.prediction["pred_scores"],
        refined_prefix.prediction["pred_scores"],
    )


def test_lag1_revision_transform_can_restore_support_for_retained_candidate() -> None:
    publisher = LagOnePublisher(
        revision_mask_transform=lambda request: {
            request.new_candidates[0].candidate_index: torch.tensor([False, True])
        }
    )
    publisher.update(
        _prediction(
            masks=[[True], [False]],
            scores=[0.4],
            classes=[3],
            queries=[1],
            stages=[0, 0],
        ),
        {1: ("track", 0)},
        _meta(absolute_stage=0, scan_ids=("scan-a",), point_counts=(2,)),
    )
    prefix = publisher.update(
        _prediction(
            masks=[[False], [False], [True]],
            scores=[0.7],
            classes=[3],
            queries=[7],
            stages=[0, 0, 1],
        ),
        {7: ("track", 0)},
        _meta(
            absolute_stage=1,
            scan_ids=("scan-a", "scan-b"),
            point_counts=(2, 1),
        ),
    )

    identity = PublishedIdentity("track", 0, 3)
    assert _scan_masks(prefix, "scan-a")[identity].tolist() == [False, True]
