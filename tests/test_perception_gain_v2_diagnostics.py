import torch

from scripts.perception_gain_v2_diagnostics import (
    reconstruct_cached_target,
    repair_segment_counts,
)


def test_cached_occupancy_recovers_only_an_exact_class_compatible_target():
    masks = torch.tensor([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)
    labels = torch.tensor([3, 4])
    segments = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0.5, 0.5])
    result = reconstruct_cached_target(
        target,
        source_class_id=3,
        gt_masks=masks,
        gt_classes=labels,
        full_segment_ids=segments,
    )
    assert result == {"status": "UNIQUE_EXACT", "gt_index": 0}
    ambiguous = reconstruct_cached_target(
        target,
        source_class_id=3,
        gt_masks=masks,
        gt_classes=torch.tensor([3, 3]),
        full_segment_ids=segments,
    )
    assert ambiguous == {"status": "AMBIGUOUS_EXACT", "gt_index": None}
    missing = reconstruct_cached_target(
        torch.tensor([1.0, 0.0]),
        source_class_id=3,
        gt_masks=masks,
        gt_classes=labels,
        full_segment_ids=segments,
    )
    assert missing == {"status": "NO_EXACT_MATCH", "gt_index": None}


def test_empty_training_target_is_not_reassigned_to_a_scene_object():
    result = reconstruct_cached_target(
        torch.zeros(2),
        source_class_id=3,
        gt_masks=torch.ones((1, 4), dtype=torch.bool),
        gt_classes=torch.tensor([3]),
        full_segment_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert result == {"status": "EMPTY_TARGET", "gt_index": None}


def test_segment_diagnostics_separate_repairs_breakage_and_boundaries():
    values = repair_segment_counts(
        torch.tensor([-0.5, 0.5, 3.0, 2.0]),
        torch.tensor([0.5, -0.5, 2.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )
    assert values["segments"] == 4
    assert values["corrected_original_errors"] == 1
    assert values["broken_original_correct"] == 2
    assert values["changed_strictly_outside_bound"] == 0
    assert values["changed_at_exact_abs2"] == 1
