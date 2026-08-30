"""Pure runtime diagnostics for samplers, stochasticity, and physical batches."""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch
from torch import nn

IMPORTANT_GRADIENT_GROUPS = (
    "class_embed_head",
    "mask_embed_head",
    "query_projection",
    "first_cross_attention",
    "last_cross_attention",
    "ptv3_decoder",
)


def serialize_sampler_chain(sampler: object) -> list[dict[str, object]]:
    """Serialize nested sampler wrappers without consuming their streams."""

    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    current: object | None = sampler
    while current is not None:
        identity = id(current)
        if identity in seen:
            raise ValueError("sampler wrapper chain contains a cycle")
        seen.add(identity)
        record: dict[str, object] = {
            "class": f"{type(current).__module__}.{type(current).__qualname__}"
        }
        rank = getattr(current, "rank", None)
        world_size = getattr(current, "num_replicas", None)
        if isinstance(rank, int):
            record["rank"] = rank
        if isinstance(world_size, int):
            record["world_size"] = world_size
        chain.append(record)
        nested = getattr(current, "sampler", None)
        if nested is None:
            wrapper_dataset = getattr(current, "dataset", None)
            nested = getattr(wrapper_dataset, "_sampler", None)
        current = nested if nested is not current else None
    return chain


def analyze_ddp_rank_streams(
    *,
    global_draws: Sequence[int],
    rank_draws: Mapping[int, Sequence[int]],
    rank_positions: Mapping[int, Sequence[int]] | None = None,
    world_size: int,
    minimum_draws_per_rank: int = 256,
) -> dict[str, object]:
    """Compare rank streams with exact positional shards of one global stream."""

    if type(world_size) is not int or world_size <= 1:
        raise ValueError("DDP world size must exceed one")
    if set(rank_draws) != set(range(world_size)):
        raise ValueError("DDP rank coverage differs")
    if any(len(rank_draws[rank]) < minimum_draws_per_rank for rank in rank_draws):
        raise ValueError(
            f"each DDP rank trace requires at least {minimum_draws_per_rank} draws"
        )
    if rank_positions is None:
        positions = {
            rank: list(range(rank, len(global_draws), world_size))
            for rank in range(world_size)
        }
    else:
        if set(rank_positions) != set(range(world_size)):
            raise ValueError("DDP rank-position coverage differs")
        positions = {
            rank: [int(position) for position in rank_positions[rank]]
            for rank in range(world_size)
        }
        if any(
            position < 0 or position >= len(global_draws)
            for values in positions.values()
            for position in values
        ):
            raise ValueError("DDP rank positions exceed the global stream")
    expected = {
        rank: [global_draws[position] for position in positions[rank]][
            : len(rank_draws[rank])
        ]
        for rank in range(world_size)
    }
    positional_mismatches = sum(
        observed != expected[rank][index]
        for rank in range(world_size)
        for index, observed in enumerate(rank_draws[rank])
        if index < len(expected[rank])
    )
    positional_mismatches += sum(
        max(0, len(rank_draws[rank]) - len(expected[rank]))
        for rank in range(world_size)
    )
    overlap = 0
    for left_rank in range(world_size):
        for right_rank in range(left_rank + 1, world_size):
            overlap += sum(
                (Counter(rank_draws[left_rank]) & Counter(rank_draws[right_rank])).values()
            )
    global_values = list(global_draws)
    replacement_duplicates = len(global_values) - len(set(global_values))
    correctly_sharded = positional_mismatches == 0
    observed_values = [
        value for rank in range(world_size) for value in rank_draws[rank]
    ]
    return {
        "world_size": world_size,
        "draws_per_rank": {
            str(rank): len(rank_draws[rank]) for rank in range(world_size)
        },
        "correctly_sharded": correctly_sharded,
        "positional_mismatch_count": positional_mismatches,
        "cross_rank_value_overlap_count": overlap,
        "cross_rank_value_overlap_rate": overlap / max(1, len(observed_values)),
        "cross_rank_value_overlap_is_sampler_bug": False,
        "observed_union_size": len(set(observed_values)),
        "replacement_duplicate_count": replacement_duplicates,
        "gate": "sampler_contract_pass" if correctly_sharded else "sampler_fix_required",
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_norm = left.norm()
    right_norm = right.norm()
    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def feature_repetition_statistics(passes: Sequence[torch.Tensor]) -> dict[str, object]:
    """Measure repeated-pass feature similarity at one decoder-relevant level."""

    if len(passes) < 8:
        raise ValueError("encoder stochasticity audit requires at least eight passes")
    flattened = [value.detach().cpu().float().reshape(-1) for value in passes]
    shape = flattened[0].shape
    if any(value.shape != shape or not torch.isfinite(value).all() for value in flattened):
        raise ValueError("repeated feature passes must be aligned and finite")
    base = flattened[0]
    cosines = [_cosine(base, value) for value in flattened[1:]]
    stacked = torch.stack(flattened)
    differences = stacked - base
    base_rms = float(torch.sqrt(torch.mean(base.square())).item())
    deviation_rms = float(torch.sqrt(torch.mean(differences.square())).item())
    relative_rms = deviation_rms / max(base_rms, torch.finfo(torch.float32).eps)
    return {
        "pass_count": len(flattened),
        "element_count": int(base.numel()),
        "mean_cosine": float(sum(cosines) / len(cosines)),
        "minimum_cosine": float(min(cosines)),
        "relative_rms_deviation": relative_rms,
        "mean_feature_variance": float(stacked.var(dim=0, unbiased=False).mean().item()),
    }


def encoder_stochasticity_gate(
    levels: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Authorize R3 if either registered stochasticity threshold is crossed."""

    if not levels:
        raise ValueError("encoder stochasticity levels are empty")
    minimum_cosine = 0.999
    maximum_relative = 0.001
    triggered = sorted(
        name
        for name, values in levels.items()
        if float(values["mean_cosine"]) < minimum_cosine
        or float(values["relative_rms_deviation"]) > maximum_relative
    )
    return {
        "authorized": bool(triggered),
        "triggered_levels": triggered,
        "minimum_cosine": minimum_cosine,
        "maximum_relative_rms_deviation": maximum_relative,
    }


def stochastic_module_inventory(module: nn.Module) -> list[dict[str, object]]:
    """Inventory stochastic and stateful normalization modules and modes."""

    rows: list[dict[str, object]] = []
    normalization_types = (
        nn.modules.batchnorm._BatchNorm,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.modules.instancenorm._InstanceNorm,
    )
    for name, child in module.named_modules():
        class_name = type(child).__name__.lower()
        if "droppath" in class_name or "stochasticdepth" in class_name:
            kind = "drop_path"
        elif isinstance(child, nn.modules.dropout._DropoutNd):
            kind = "dropout"
        elif isinstance(child, normalization_types):
            kind = "normalization"
        else:
            continue
        row: dict[str, object] = {
            "name": name,
            "class": f"{type(child).__module__}.{type(child).__qualname__}",
            "kind": kind,
            "training": bool(child.training),
        }
        for attribute in ("drop_prob", "p", "track_running_stats"):
            value = getattr(child, attribute, None)
            if isinstance(value, (bool, int, float)):
                row[attribute] = value
        rows.append(row)
    return rows


@contextmanager
def disable_drop_path(module: nn.Module) -> Iterator[list[str]]:
    """Temporarily zero DropPath probability without changing module modes."""

    changed: list[tuple[str, nn.Module, str, float]] = []
    for name, child in module.named_modules():
        class_name = type(child).__name__.lower()
        if "droppath" not in class_name and "stochasticdepth" not in class_name:
            continue
        for attribute in ("drop_prob", "p"):
            value = getattr(child, attribute, None)
            if isinstance(value, (int, float)):
                changed.append((name, child, attribute, float(value)))
                setattr(child, attribute, 0.0)
                break
    try:
        yield [name for name, _, _, _ in changed]
    finally:
        for _, child, attribute, value in changed:
            setattr(child, attribute, value)


def build_fixed_batch_panel(
    references: Sequence[Mapping[str, object]], *, seed: int
) -> dict[str, object]:
    """Freeze exactly 32 global sample references and per-sample RNG seeds."""

    if len(references) != 32:
        raise ValueError("physical-batch panel requires exactly 32 samples")
    if type(seed) is not int:
        raise ValueError("physical-batch panel seed is invalid")
    samples = []
    for position, reference in enumerate(references):
        if not isinstance(reference, Mapping) or not reference.get("dataset"):
            raise ValueError("physical-batch sample reference is invalid")
        samples.append(
            {
                "position": position,
                **copy.deepcopy(dict(reference)),
                "augmentation_seed": seed + position,
            }
        )
    return {"schema_version": 1, "seed": seed, "samples": samples}


def physical_batch_gradient_gate(
    rows: Sequence[Mapping[str, object]], *, feasible: bool
) -> dict[str, object]:
    """Authorize R2 only for a feasible grouping with two material group shifts."""

    thresholds = {
        "minimum_cosine": 0.98,
        "maximum_relative_norm_difference": 0.1,
        "minimum_triggered_groups": 2,
    }
    important = set(IMPORTANT_GRADIENT_GROUPS)
    triggered = sorted(
        {
            str(row["parameter_group"])
            for row in rows
            if str(row.get("parameter_group")) in important
            and (
                float(row["cosine"]) < thresholds["minimum_cosine"]
                or float(row["relative_norm_difference"])
                > thresholds["maximum_relative_norm_difference"]
            )
        }
    )
    return {
        "authorized": feasible
        and len(triggered) >= thresholds["minimum_triggered_groups"],
        "feasible": feasible,
        "triggered_groups": triggered,
        "triggered_group_count": len(triggered),
        "thresholds": thresholds,
    }


def summarize_physical_batch_runs(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate resource records while keeping OOM results non-numeric."""

    if not runs or runs[0].get("physical_global_batch") != 4 or not runs[0].get(
        "feasible"
    ):
        raise ValueError("physical-global batch 4 feasible reference is required")
    normalized = []
    for run in runs:
        record = copy.deepcopy(dict(run))
        feasible = record.get("feasible")
        if feasible is not True:
            failure = str(record.get("failure", ""))
            if "OOM" not in failure and "OutOfMemory" not in failure:
                raise ValueError("infeasible physical batch must record OOM")
            if any(
                name in record
                for name in ("peak_memory_mib", "step_seconds", "objective_value")
            ):
                raise ValueError("OOM physical batch cannot contain measurements")
        else:
            for name in ("peak_memory_mib", "step_seconds"):
                value = record.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError("feasible physical batch measurements are invalid")
        normalized.append(record)
    return {
        "schema_version": 1,
        "reference_physical_global_batch": 4,
        "runs": normalized,
    }
