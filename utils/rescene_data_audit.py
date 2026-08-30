"""Pure label and sampling audits for ReScene root-cause analysis."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabelSample:
    dataset: str
    sample_id: str
    semantic_labels: torch.Tensor
    instance_ids: torch.Tensor


def _sample_counts(
    sample: LabelSample, *, excluded_classes: set[int]
) -> dict[str, int]:
    semantics = sample.semantic_labels.detach().cpu().long().reshape(-1)
    instances = sample.instance_ids.detach().cpu().long().reshape(-1)
    if semantics.shape != instances.shape or semantics.numel() == 0:
        raise ValueError("label sample tensors must be non-empty and aligned")
    valid_points = instances != -1
    for excluded in excluded_classes:
        valid_points &= semantics != excluded
    target_instances = 0
    label255_instances = 0
    for instance_id in torch.unique(instances[valid_points]).tolist():
        selector = valid_points & (instances == int(instance_id))
        labels = torch.unique(semantics[selector])
        if labels.numel() != 1:
            raise ValueError("one target instance has inconsistent semantic labels")
        target_instances += 1
        if int(labels.item()) == 255:
            label255_instances += 1
    return {
        "target_instances": target_instances,
        "target_points": int(valid_points.sum().item()),
        "label255_instances": label255_instances,
        "label255_points": int((valid_points & (semantics == 255)).sum().item()),
    }


def _with_fractions(counts: dict[str, int]) -> dict[str, int | float]:
    instances = counts["target_instances"]
    points = counts["target_points"]
    return {
        **counts,
        "instance_fraction": (
            counts["label255_instances"] / instances if instances else 0.0
        ),
        "point_fraction": counts["label255_points"] / points if points else 0.0,
    }


def inventory_filter255(
    samples: Iterable[LabelSample], *, excluded_classes: Sequence[int] = (0, 1)
) -> dict[str, object]:
    """Count remapped ignore targets over the complete active sample database."""

    excluded = {int(value) for value in excluded_classes}
    per_dataset: dict[str, dict[str, int]] = {}
    totals = {
        "target_instances": 0,
        "target_points": 0,
        "label255_instances": 0,
        "label255_points": 0,
    }
    sample_count = 0
    for sample in samples:
        sample_count += 1
        if not sample.dataset or not sample.sample_id:
            raise ValueError("label sample identity is invalid")
        counts = _sample_counts(sample, excluded_classes=excluded)
        dataset_counts = per_dataset.setdefault(
            sample.dataset, {key: 0 for key in totals}
        )
        for key, value in counts.items():
            totals[key] += value
            dataset_counts[key] += value
    if sample_count == 0:
        raise ValueError("filter-255 inventory requires samples")
    total_row = _with_fractions(totals)
    threshold = 0.005
    return {
        "schema_version": 1,
        "sample_count": sample_count,
        "excluded_classes": sorted(excluded),
        "totals": total_row,
        "per_dataset": [
            {"dataset": name, **_with_fractions(per_dataset[name])}
            for name in sorted(per_dataset)
        ],
        "gate": {
            "threshold": threshold,
            "material": total_row["instance_fraction"] >= threshold
            or total_row["point_fraction"] >= threshold,
        },
    }


def filter255_supervision_contract(
    *, raw_ignore_label: int, label_offset: int, criterion_ignore_index: int
) -> dict[str, object]:
    """Describe the current target/matcher/loss path for remapped label 255."""

    target_label = raw_ignore_label - label_offset
    if target_label != criterion_ignore_index:
        raise ValueError("ignore label remapping differs from criterion sentinel")
    return {
        "raw_label": raw_ignore_label,
        "target_label": target_label,
        "target_construction_with_filter_255": "excluded",
        "target_construction_without_filter_255": "included",
        "classification": "ignored",
        "matcher_class_cost": "ignore_sentinel",
        "matcher_mask_cost": "included",
        "mask_and_dice": "included",
    }


def summarize_weighted_draws(
    draws: Sequence[int], *, dataset_sizes: Sequence[int], dataset_names: Sequence[str]
) -> dict[str, object]:
    """Summarize replacement draws without treating repeats as DDP overlap."""

    if len(dataset_sizes) != len(dataset_names) or not dataset_sizes:
        raise ValueError("dataset draw boundaries are invalid")
    sizes = tuple(int(value) for value in dataset_sizes)
    if any(value <= 0 for value in sizes):
        raise ValueError("dataset sizes must be positive")
    boundaries = []
    total = 0
    for size in sizes:
        total += size
        boundaries.append(total)
    counts = {str(name): 0 for name in dataset_names}
    normalized = []
    for draw in draws:
        if type(draw) is not int or not 0 <= draw < total:
            raise ValueError("weighted draw is outside the concatenated dataset")
        normalized.append(draw)
        for index, boundary in enumerate(boundaries):
            if draw < boundary:
                counts[str(dataset_names[index])] += 1
                break
    unique = len(set(normalized))
    duplicate_count = len(normalized) - unique
    return {
        "draw_count": len(normalized),
        "dataset_draws": counts,
        "dataset_draw_ratios": {
            name: count / len(normalized) if normalized else 0.0
            for name, count in counts.items()
        },
        "unique_sample_count": unique,
        "replacement_duplicate_count": duplicate_count,
        "replacement_duplicate_rate": (
            duplicate_count / len(normalized) if normalized else 0.0
        ),
    }
