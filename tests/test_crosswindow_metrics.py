from __future__ import annotations

import torch

from datasets.task_memory_episode import StageMeta
from models.crosswindow_state import CrossWindowState
from models.overlap_entity_association import (
    A0_DEFAULT,
    associate,
    build_evidence,
)
from models.task_memory_routing import PredictionObservation
from scripts.crosswindow_cache import build_canonical_frame
from scripts.rescene_task_postprocess import OfficialTaskPrediction


def _frame_and_prediction():
    vertex_ids = torch.tensor([30, 10, 20], dtype=torch.long)
    points = torch.arange(3, dtype=torch.long)
    observation = PredictionObservation(
        features=torch.tensor([[[1.0, 0.0]]]),
        class_prob=torch.tensor([[[0.75, 0.25]]]),
        confidence=torch.tensor([[0.9]]),
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[True]]),
        previous_supported=torch.tensor([[False]]),
    )
    masks = torch.tensor([[True], [False], [True]], dtype=torch.bool)
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.8]),
        pred_classes=torch.tensor([1]),
        source_query_ids=torch.tensor([0]),
        source_class_ids=torch.tensor([1]),
        temporal_stages=torch.zeros(3, dtype=torch.long),
        latest_stage_index=0,
        latest_stage_masks=masks.clone(),
    )
    meta = StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=("scan-a",),
        absolute_stage_index=0,
        local_stage_ids=torch.zeros(3, dtype=torch.long),
        original_vertex_ids=(vertex_ids,),
        scan_vertex_offsets=torch.tensor([0, 3]),
        point2segment=points.clone(),
        segment_stage_ids=torch.zeros(3, dtype=torch.long),
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=points.clone(),
        full_resolution_point2segment=points.clone(),
    )
    frame = build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=observation,
        prediction=prediction,
        stage_meta=meta,
    )
    return frame, prediction


def test_publish_consumes_plan_without_reassignment() -> None:
    from scripts.replay_crosswindow_association import publish

    frame, _ = _frame_and_prediction()
    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    plan = associate(build_evidence(frame, state, None), A0_DEFAULT)

    prefix = publish(
        frame,
        plan,
        mask_selection="new",
        history_boundary=None,
    )

    assert prefix.identity_map == tuple(
        zip(
            plan.group_keys,
            zip(plan.entity_for_group, plan.generation_for_group, strict=True),
            strict=True,
        )
    )
    assert prefix.archive == ()
    assert prefix.provisional.vertex_ids.tolist() == [10, 20, 30]
    assert set(prefix.prediction) == {"pred_masks", "pred_scores", "pred_classes"}


class _MaskSumMetric:
    def __init__(self, mode: str) -> None:
        assert mode == "raw_local"
        self.value = 0.0

    def update(self, prediction, target) -> None:
        assert set(prediction) == {"pred_masks", "pred_scores", "pred_classes"}
        self.value = float(prediction["pred_masks"].sum().item())

    def compute(self):
        return {
            "raw_local_AP": self.value,
            "raw_local_AP50": self.value,
            "raw_local_AP25": self.value,
            "raw_local_REC": self.value,
        }


def test_raw_current_ap_is_association_invariant() -> None:
    from scripts.replay_crosswindow_association import evaluate_raw_current

    _, prediction = _frame_and_prediction()
    target = {
        "masks": torch.tensor([[True, False, True]]),
        "labels": torch.tensor([1]),
        "ids": torch.tensor([7]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.zeros(3, dtype=torch.long),
    }

    left = evaluate_raw_current(
        prediction,
        target,
        metric_factory=_MaskSumMetric,
    )
    right = evaluate_raw_current(
        prediction,
        target,
        published_prefix=object(),
        metric_factory=_MaskSumMetric,
    )

    assert left == right


def test_publish_freezes_archive_and_preserves_duplicate_class_candidates() -> None:
    from scripts.replay_crosswindow_association import publish

    frame, _ = _frame_and_prediction()
    duplicate = frame.candidates[0]
    frame = type(frame)(
        **{
            **frame.__dict__,
            "groups": (
                type(frame.groups[0])(
                    **{
                        **frame.groups[0].__dict__,
                        "candidate_indices": (0, 1),
                    }
                ),
            ),
            "candidates": (
                duplicate,
                type(duplicate)(
                    key=type(duplicate.key)(
                        **{**duplicate.key.__dict__, "candidate_index": 1}
                    ),
                    predicted_class_id=duplicate.predicted_class_id,
                    score=0.7,
                    slices=duplicate.slices,
                ),
            ),
        }
    )
    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    plan = associate(build_evidence(frame, state, None), A0_DEFAULT)

    prefix = publish(frame, plan, mask_selection="new", history_boundary=None)

    assert prefix.prediction["pred_masks"].shape[1] == 2
    assert prefix.keys[0] != prefix.keys[1]
    assert prefix.collision_fallback_count == 1


def test_canonical_target_and_prediction_share_point_order() -> None:
    from scripts.replay_crosswindow_association import canonicalize_stage_target

    frame, _ = _frame_and_prediction()
    target = {
        "gt_ids": torch.tensor([7]),
        "gt_classes": torch.tensor([1]),
        "gt_masks": torch.tensor([[True, False, True]]),
        "changes": torch.tensor([0]),
        "change_labels_valid": torch.tensor([True]),
        "change_label_semantics": "none",
        "gt_class_semantics": "model-space",
    }

    canonical = canonicalize_stage_target(
        target,
        original_vertex_ids=torch.tensor([30, 10, 20]),
    )

    assert canonical["gt_masks"].tolist() == [[False, True, True]]
    assert frame.candidates[0].slices[0].mask.tolist() == [False, True, True]


def test_publish_revision_does_not_mutate_prior_history_boundary() -> None:
    from models.overlap_entity_association import commit_observation
    from scripts.replay_crosswindow_association import publish

    first_frame, _ = _frame_and_prediction()
    state = CrossWindowState.empty(capacity=2, feature_dim=2, class_count=2)
    first_plan = associate(build_evidence(first_frame, state, None), A0_DEFAULT)
    state, committed = commit_observation(first_frame, state, first_plan)
    first_prefix = publish(
        first_frame,
        first_plan,
        mask_selection="new",
        history_boundary=None,
    )
    frozen_boundary = first_prefix.history_boundary

    previous_ids = torch.tensor([10, 20, 30], dtype=torch.long)
    current_ids = torch.tensor([60, 40, 50], dtype=torch.long)
    observation = PredictionObservation(
        features=torch.tensor([[[1.0, 0.0]]]),
        class_prob=torch.tensor([[[0.75, 0.25]]]),
        confidence=torch.tensor([[0.9]]),
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[True]]),
        previous_supported=torch.tensor([[True]]),
    )
    masks = torch.tensor(
        [[True], [False], [True], [False], [True], [True]], dtype=torch.bool
    )
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.6]),
        pred_classes=torch.tensor([1]),
        source_query_ids=torch.tensor([0]),
        source_class_ids=torch.tensor([1]),
        temporal_stages=torch.tensor([0, 0, 0, 1, 1, 1]),
        latest_stage_index=1,
        latest_stage_masks=masks[3:].clone(),
    )
    meta = StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=("scan-a", "scan-b"),
        absolute_stage_index=1,
        local_stage_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        original_vertex_ids=(previous_ids, current_ids),
        scan_vertex_offsets=torch.tensor([0, 3, 6]),
        point2segment=torch.arange(6),
        segment_stage_ids=torch.tensor([0, 0, 0, 1, 1, 1]),
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=torch.arange(6),
        full_resolution_point2segment=torch.arange(6),
    )
    second_frame = build_canonical_frame(
        producer_id="r1-b4",
        order_id="order-0",
        observation=observation,
        prediction=prediction,
        stage_meta=meta,
    )
    second_plan = associate(
        build_evidence(second_frame, state, committed.buffer), A0_DEFAULT
    )
    second_prefix = publish(
        second_frame,
        second_plan,
        mask_selection="new",
        history_boundary=frozen_boundary,
    )

    assert frozen_boundary.archive == ()
    assert frozen_boundary.provisional is first_prefix.provisional
    assert [scan.scan_id for scan in second_prefix.archive] == ["scan-a"]
    assert second_prefix.provisional.scan_id == "scan-b"
    assert [revision.scan_id for revision in second_prefix.revisions] == ["scan-a"]
