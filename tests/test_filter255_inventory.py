from __future__ import annotations

import torch

from utils.rescene_data_audit import (
    LabelSample,
    filter255_supervision_contract,
    inventory_filter255,
    summarize_weighted_draws,
)


def test_filter255_inventory_counts_instances_points_and_dataset_fractions() -> None:
    samples = [
        LabelSample(
            dataset="rio",
            sample_id="r0",
            semantic_labels=torch.tensor([2, 2, 255, 255, 3, 3]),
            instance_ids=torch.tensor([10, 10, 20, 20, 30, 30]),
        ),
        LabelSample(
            dataset="scannet",
            sample_id="s0",
            semantic_labels=torch.tensor([255, 4, 4, 4]),
            instance_ids=torch.tensor([40, 50, 50, 50]),
        ),
    ]

    result = inventory_filter255(samples, excluded_classes=(0, 1))

    assert result["totals"]["target_instances"] == 5
    assert result["totals"]["target_points"] == 10
    assert result["totals"]["label255_instances"] == 2
    assert result["totals"]["label255_points"] == 3
    assert result["totals"]["instance_fraction"] == 0.4
    assert result["totals"]["point_fraction"] == 0.3
    assert result["gate"]["material"] is True
    assert {row["dataset"] for row in result["per_dataset"]} == {"rio", "scannet"}


def test_filter255_gate_uses_fixed_half_percent_threshold() -> None:
    sample = LabelSample(
        dataset="rio",
        sample_id="r0",
        semantic_labels=torch.tensor([255] + [2] * 999),
        instance_ids=torch.arange(1000),
    )

    result = inventory_filter255([sample], excluded_classes=(0, 1))

    assert result["gate"]["threshold"] == 0.005
    assert result["gate"]["material"] is False


def test_filter255_inventory_matches_first_point_instance_semantics() -> None:
    sample = LabelSample(
        dataset="rio",
        sample_id="mixed",
        semantic_labels=torch.tensor([255, 19, 19, 2, 255]),
        instance_ids=torch.tensor([7, 7, 7, 8, 8]),
    )

    result = inventory_filter255([sample], excluded_classes=(0, 1))

    assert result["totals"]["target_instances"] == 2
    assert result["totals"]["target_points"] == 5
    assert result["totals"]["label255_instances"] == 1
    assert result["totals"]["label255_points"] == 3


def test_label255_supervision_contract_matches_current_code_path() -> None:
    contract = filter255_supervision_contract(
        raw_ignore_label=255, label_offset=2, criterion_ignore_index=253
    )

    assert contract == {
        "raw_label": 255,
        "target_label": 253,
        "instance_semantic_resolution": "first_point_matching_training_collator",
        "target_construction_with_filter_255": "excluded",
        "target_construction_without_filter_255": "included",
        "classification": "ignored",
        "matcher_class_cost": "ignore_sentinel",
        "matcher_mask_cost": "included",
        "mask_and_dice": "included",
    }


def test_weighted_draw_summary_separates_replacement_duplicates() -> None:
    draws = [0, 4, 1, 4, 5, 0]
    result = summarize_weighted_draws(
        draws,
        dataset_sizes=(4, 2),
        dataset_names=("rio", "scannet"),
    )

    assert result["draw_count"] == 6
    assert result["dataset_draws"] == {"rio": 3, "scannet": 3}
    assert result["unique_sample_count"] == 4
    assert result["replacement_duplicate_count"] == 2
    assert result["replacement_duplicate_rate"] == 2 / 6
