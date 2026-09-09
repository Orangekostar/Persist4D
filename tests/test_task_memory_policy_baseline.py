from __future__ import annotations

import torch

from datasets.task_memory_episode import NativeEpisodeMaster
from scripts.run_task_memory_policy_baseline import (
    _target_for_prefix,
    _validate_collated_stage_identity,
    classify_gap_event,
    select_diagnostic_masters,
)


def _master(reference: str, sequence: str, context_index: int) -> NativeEpisodeMaster:
    scan_ids = tuple(sequence.split("-"))
    return NativeEpisodeMaster(
        reference_id=reference,
        sequence_id=sequence,
        scan_ids=scan_ids,
        scan_indices=tuple(range(len(scan_ids))),
        role="development",
        context_index=context_index,
    )


def test_diagnostic_panel_round_robins_references_before_repeats() -> None:
    masters = (
        _master("ref-b", "scene0002_01-scene0002_02", 1),
        _master("ref-a", "scene0001_02-scene0001_03", 2),
        _master("ref-b", "scene0002_00-scene0002_01", 0),
        _master("ref-a", "scene0001_01-scene0001_02", 3),
        _master("ref-c", "scene0003_00-scene0003_01", 4),
    )

    selected = select_diagnostic_masters(masters, limit=4)

    assert [master.reference_id for master in selected] == [
        "ref-a",
        "ref-b",
        "ref-c",
        "ref-a",
    ]
    assert [master.sequence_id for master in selected] == [
        "scene0001_01-scene0001_02",
        "scene0002_00-scene0002_01",
        "scene0003_00-scene0003_01",
        "scene0001_02-scene0001_03",
    ]


def _target(ids: list[int]) -> dict[str, object]:
    return {
        "gt_ids": torch.tensor(ids, dtype=torch.long),
        "gt_classes": torch.zeros(len(ids), dtype=torch.long),
        "gt_masks": torch.ones((len(ids), 2), dtype=torch.bool),
        "changes": torch.zeros(len(ids), dtype=torch.long),
        "change_labels_valid": False,
        "change_label_semantics": (
            "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
        ),
        "gt_class_semantics": "rescene_model_index_0_based",
    }


def test_gap_event_requires_reappearance_after_an_absent_stage() -> None:
    targets = (_target([1]), _target([]), _target([1]))

    assert classify_gap_event(targets, horizon=2) == "no_gap_event"
    assert classify_gap_event(targets, horizon=3) == "gap_event"


def test_continuous_or_new_visibility_is_not_a_gap_event() -> None:
    continuous = (_target([1]), _target([1]), _target([1, 2]))
    new_only = (_target([]), _target([]), _target([4]))

    assert classify_gap_event(continuous, horizon=3) == "no_gap_event"
    assert classify_gap_event(new_only, horizon=3) == "no_gap_event"


def test_collated_stage_identity_uses_the_local_window_name() -> None:
    _validate_collated_stage_identity(
        names=["scene0001_01-scene0001_02"],
        scan_ids_in_window=("scene0001_01", "scene0001_02"),
    )


def test_prefix_target_supplies_contiguous_stage_keys() -> None:
    target = _target_for_prefix(
        (_target([1]), _target([]), _target([1])),
        horizon=3,
        class_mapper=lambda value: value,
    )

    assert target["temporal_stages"].tolist() == [0, 0, 1, 1, 2, 2]
    assert target["ids"].tolist() == [1]
    assert target["masks"].shape == (1, 6)
