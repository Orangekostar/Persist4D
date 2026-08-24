"""Controlled frozen-observation capacity replay for final evidence."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from models.persistent_memory import LocalInstanceObservation
from scripts.evaluate_persist4d_p6a import observation_content_digest
from scripts.p6a_association import (
    B4PersistentTracker,
    FrozenObservation,
    freeze_observation,
)

CAPACITY_GRID = (64, 100, 128, 160, 200)


def build_class_mapper_from_label_document(
    document: Mapping[object, object],
    *,
    foreground_class_count: int = 18,
    label_offset: int = 2,
) -> Callable[[int], int]:
    """Reconstruct the frozen RIO model-to-raw-label mapping."""

    if not isinstance(document, Mapping) or not document:
        raise ValueError("label document must be a non-empty mapping")
    for name, value in (
        ("foreground_class_count", foreground_class_count),
        ("label_offset", label_offset),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) < (1 if name == "foreground_class_count" else 0)
        ):
            raise ValueError(f"{name} is invalid")
    validation_ids = []
    for raw_id, metadata in document.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, Integral):
            raise ValueError("label IDs must be integers")
        if (
            not isinstance(metadata, Mapping)
            or type(metadata.get("validation")) is not bool
        ):
            raise ValueError("label metadata must contain a boolean validation field")
        if metadata["validation"]:
            validation_ids.append(int(raw_id))
    end = int(label_offset) + int(foreground_class_count)
    if end > len(validation_ids):
        raise ValueError("label document does not cover the frozen model classes")
    selected = tuple(validation_ids[int(label_offset) : end])

    def mapper(model_class: int) -> int:
        if (
            isinstance(model_class, bool)
            or not isinstance(model_class, Integral)
            or not 0 <= int(model_class) < len(selected)
        ):
            raise ValueError("foreground model class is outside the registered range")
        return selected[int(model_class)]

    return mapper


def build_protocol_from_reviewer_manifest(
    document: Mapping[str, object],
    *,
    expected_master_count: int = 43,
) -> dict[str, object]:
    """Reconstruct only the cache-key protocol recorded by reviewer closure."""

    if not isinstance(document, Mapping):
        raise ValueError("reviewer manifest must be a mapping")
    if (
        isinstance(expected_master_count, bool)
        or not isinstance(expected_master_count, Integral)
        or int(expected_master_count) <= 0
    ):
        raise ValueError("expected_master_count must be positive")
    protocol_metadata = document.get("protocol")
    masters_raw = document.get("masters")
    if not isinstance(protocol_metadata, Mapping) or not isinstance(
        masters_raw, Sequence
    ):
        raise ValueError("reviewer manifest protocol and masters are required")
    orders = protocol_metadata.get("order_variants")
    expected_orders = ("canonical", "reverse", "sha256_seed45")
    normalized_orders = (
        tuple(orders)
        if isinstance(orders, Sequence) and not isinstance(orders, (str, bytes))
        else ()
    )
    if normalized_orders != expected_orders:
        raise ValueError("reviewer manifest order variants differ")
    if len(masters_raw) != int(expected_master_count):
        raise ValueError("reviewer manifest master coverage differs")
    masters = []
    variants: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for raw in masters_raw:
        if not isinstance(raw, Mapping):
            raise ValueError("reviewer manifest masters must be mappings")
        master_id = raw.get("master_sequence_id")
        reference_id = raw.get("reference_scene_id")
        by_order = raw.get("orders")
        if (
            not isinstance(master_id, str)
            or not master_id
            or not isinstance(reference_id, str)
            or not reference_id
            or not isinstance(by_order, Mapping)
        ):
            raise ValueError("reviewer manifest master identity is invalid")
        if master_id in variants or set(by_order) != set(expected_orders):
            raise ValueError("reviewer manifest master/order coverage differs")
        masters.append({"sequence_id": master_id, "reference_scene_id": reference_id})
        variants[master_id] = {}
        for order in expected_orders:
            order_record = by_order[order]
            visit_order = (
                order_record.get("visit_order")
                if isinstance(order_record, Mapping)
                else None
            )
            if (
                isinstance(visit_order, (str, bytes))
                or not isinstance(visit_order, Sequence)
                or len(visit_order) != 5
                or any(not isinstance(scan, str) or not scan for scan in visit_order)
                or len(set(visit_order)) != 5
            ):
                raise ValueError("reviewer manifest visit order is invalid")
            variants[master_id][order] = {"scan_ids": tuple(visit_order)}
    if len({master["sequence_id"] for master in masters}) != len(masters):
        raise ValueError("reviewer manifest master IDs must be unique")
    return {
        "order_variants": expected_orders,
        "masters": tuple(masters),
        "variants": variants,
    }


def classify_capacity_gate(
    *,
    robust_improvement: bool,
    preexisting_development_split: bool,
    selected_without_final_tuning: bool,
    architecture_unchanged: bool,
) -> str:
    """Apply the preregistered capacity-selection boundary."""

    values = (
        robust_improvement,
        preexisting_development_split,
        selected_without_final_tuning,
        architecture_unchanged,
    )
    if any(type(value) is not bool for value in values):
        raise ValueError("capacity gate inputs must be booleans")
    if not architecture_unchanged:
        raise ValueError("capacity gate requires the frozen architecture")
    if not robust_improvement:
        return "CAPACITY_100_OK"
    if preexisting_development_split and selected_without_final_tuning:
        return "CAPACITY_CONFIG_REOPEN"
    return "CAPACITY_SENSITIVITY_ONLY"


@dataclass(frozen=True, slots=True)
class CapacityEvaluation:
    per_sequence_rows: tuple[dict[str, object], ...]
    aggregate_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CapacityBootstrap:
    effects: tuple[dict[str, object], ...]
    per_scene_effects: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class CapacityRobustness:
    robust_improvement: bool
    candidates: tuple[dict[str, object], ...]


def _cluster_metric_value(
    rows: Sequence[Mapping[str, object]], metric: str
) -> float | None:
    rate_counts = {
        "normalized_id_switch_rate": (
            "deployment_id_switches",
            "identity_transition_opportunities",
        ),
        "fragmentation_rate": (
            "fragmentation_count",
            "fragmentation_opportunities",
        ),
        "merge_rate": ("merge_count", "merge_opportunities"),
        "gap_recovery_accuracy": ("correct_recoveries", "recovery_attempts"),
        "gap_recovery_recall": ("correct_recoveries", "gap_opportunities"),
    }
    if metric in rate_counts:
        numerator_name, denominator_name = rate_counts[metric]
        numerator = sum(int(row[numerator_name]) for row in rows)
        denominator = sum(int(row[denominator_name]) for row in rows)
        return numerator / denominator if denominator else None
    values = [
        float(row[metric])
        for row in rows
        if row.get(metric) is not None and math.isfinite(float(row[metric]))
    ]
    return math.fsum(values) / len(values) if values else None


def capacity_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    reference_capacity: int,
    candidate_capacities: Sequence[object],
    metrics: Sequence[str],
    horizons: Sequence[object] = (2, 3, 4, 5),
    expected_cluster_count: int = 6,
    replicates: int = 10_000,
    seed: int = 45,
) -> CapacityBootstrap:
    """Pair capacity effects at the independent physical-scene level."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("capacity bootstrap requires result rows")
    candidates = _validated_capacities(candidate_capacities)
    if reference_capacity in candidates:
        raise ValueError("reference capacity must not be a candidate")
    if (
        isinstance(metrics, (str, bytes))
        or not isinstance(metrics, Sequence)
        or not metrics
        or any(not isinstance(metric, str) or not metric for metric in metrics)
    ):
        raise ValueError("bootstrap metrics must be non-empty names")
    horizon_values = tuple(int(value) for value in horizons)
    if horizon_values != tuple(sorted(set(horizon_values))) or any(
        value not in (2, 3, 4, 5) for value in horizon_values
    ):
        raise ValueError("bootstrap horizons must be unique T2-T5 values")
    for name, value in (
        ("expected_cluster_count", expected_cluster_count),
        ("replicates", replicates),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("bootstrap seed must be an integer")

    effect_rows: list[dict[str, object]] = []
    per_scene_rows: list[dict[str, object]] = []
    rng = random.Random(int(seed))
    for candidate in candidates:
        for horizon in horizon_values:
            reference_rows = [
                row
                for row in rows
                if int(row["capacity"]) == int(reference_capacity)
                and int(row["horizon"]) == horizon
            ]
            candidate_rows = [
                row
                for row in rows
                if int(row["capacity"]) == candidate and int(row["horizon"]) == horizon
            ]
            reference_scenes = {
                str(row["reference_scene_id"]) for row in reference_rows
            }
            candidate_scenes = {
                str(row["reference_scene_id"]) for row in candidate_rows
            }
            if reference_scenes != candidate_scenes or len(reference_scenes) != int(
                expected_cluster_count
            ):
                raise ValueError("capacity bootstrap scene coverage differs")
            for metric in metrics:
                paired_values = []
                for scene in sorted(reference_scenes):
                    reference_value = _cluster_metric_value(
                        [
                            row
                            for row in reference_rows
                            if row["reference_scene_id"] == scene
                        ],
                        metric,
                    )
                    candidate_value = _cluster_metric_value(
                        [
                            row
                            for row in candidate_rows
                            if row["reference_scene_id"] == scene
                        ],
                        metric,
                    )
                    effect = (
                        candidate_value - reference_value
                        if reference_value is not None and candidate_value is not None
                        else None
                    )
                    per_scene_rows.append(
                        {
                            "capacity": candidate,
                            "reference_capacity": int(reference_capacity),
                            "horizon": horizon,
                            "metric": metric,
                            "reference_scene_id": scene,
                            "reference_value": reference_value,
                            "candidate_value": candidate_value,
                            "effect": effect,
                        }
                    )
                    if effect is not None:
                        paired_values.append(
                            (float(reference_value), float(candidate_value), effect)
                        )
                reference_values = [value[0] for value in paired_values]
                candidate_values = [value[1] for value in paired_values]
                effects = [value[2] for value in paired_values]
                reference_mean = (
                    math.fsum(reference_values) / len(reference_values)
                    if reference_values
                    else None
                )
                candidate_mean = (
                    math.fsum(candidate_values) / len(candidate_values)
                    if candidate_values
                    else None
                )
                effect = math.fsum(effects) / len(effects) if effects else None
                bootstrap_means = []
                if effects:
                    for _ in range(int(replicates)):
                        bootstrap_means.append(
                            math.fsum(
                                effects[rng.randrange(len(effects))]
                                for _ in range(len(effects))
                            )
                            / len(effects)
                        )
                effect_rows.append(
                    {
                        "capacity": candidate,
                        "reference_capacity": int(reference_capacity),
                        "horizon": horizon,
                        "metric": metric,
                        "scene_coverage_count": len(reference_scenes),
                        "cluster_count": len(effects),
                        "bootstrap_replicates": int(replicates),
                        "seed": int(seed),
                        "reference_mean": reference_mean,
                        "candidate_mean": candidate_mean,
                        "effect": effect,
                        "relative_effect": (
                            effect / reference_mean
                            if effect is not None
                            and reference_mean is not None
                            and reference_mean != 0.0
                            else None
                        ),
                        "ci_lower": (
                            _linear_quantile(bootstrap_means, 0.025)
                            if bootstrap_means
                            else None
                        ),
                        "ci_upper": (
                            _linear_quantile(bootstrap_means, 0.975)
                            if bootstrap_means
                            else None
                        ),
                    }
                )
    return CapacityBootstrap(
        effects=tuple(effect_rows),
        per_scene_effects=tuple(per_scene_rows),
    )


def assess_robust_capacity_improvement(
    effects: Sequence[Mapping[str, object]],
    aggregate_rows: Sequence[Mapping[str, object]],
    *,
    reference_capacity: int,
    candidate_capacities: Sequence[object],
    primary_metrics: Sequence[str],
    minimum_absolute_improvement: float,
    maximum_t_map_drop: float,
    maximum_id_switch_increase: float,
) -> CapacityRobustness:
    """Apply the result-independent capacity robustness rule."""

    candidates = _validated_capacities(candidate_capacities)
    if not primary_metrics or any(
        not isinstance(metric, str) or not metric for metric in primary_metrics
    ):
        raise ValueError("primary metrics must be non-empty names")
    thresholds = (
        minimum_absolute_improvement,
        maximum_t_map_drop,
        maximum_id_switch_increase,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds):
        raise ValueError("robustness thresholds must be finite and non-negative")

    by_cell = {
        (int(row["capacity"]), int(row["horizon"])): row for row in aggregate_rows
    }
    horizons = (2, 3, 4, 5)
    if any(
        (capacity, horizon) not in by_cell
        for capacity in (int(reference_capacity), *candidates)
        for horizon in horizons
    ):
        raise ValueError("aggregate capacity coverage differs")
    candidate_rows = []
    for candidate in candidates:
        primary_evidence = [
            row
            for row in effects
            if int(row["capacity"]) == candidate
            and str(row["metric"]) in primary_metrics
            and row.get("effect") is not None
            and float(row["effect"]) >= minimum_absolute_improvement
            and row.get("ci_lower") is not None
            and float(row["ci_lower"]) > 0.0
        ]
        t_map_deltas = []
        id_switch_deltas = []
        non_degradation_defined = True
        for horizon in horizons:
            reference = by_cell[(int(reference_capacity), horizon)]
            comparison = by_cell[(candidate, horizon)]
            t_map_deltas.append(
                float(comparison["causal_prefix_t_mAP"])
                - float(reference["causal_prefix_t_mAP"])
            )
            reference_id = reference.get("normalized_id_switch_rate")
            comparison_id = comparison.get("normalized_id_switch_rate")
            if reference_id is None and comparison_id is None:
                continue
            if reference_id is None or comparison_id is None:
                non_degradation_defined = False
                continue
            id_switch_deltas.append(float(comparison_id) - float(reference_id))
        t_map_ok = min(t_map_deltas) >= -maximum_t_map_drop
        id_switch_ok = non_degradation_defined and (
            not id_switch_deltas or max(id_switch_deltas) <= maximum_id_switch_increase
        )
        robust = bool(primary_evidence) and t_map_ok and id_switch_ok
        candidate_rows.append(
            {
                "capacity": candidate,
                "primary_evidence_cell_count": len(primary_evidence),
                "worst_t_mAP_delta": min(t_map_deltas),
                "worst_id_switch_delta": (
                    max(id_switch_deltas) if id_switch_deltas else None
                ),
                "t_mAP_non_degradation": t_map_ok,
                "id_switch_non_degradation": id_switch_ok,
                "robust": robust,
            }
        )
    return CapacityRobustness(
        robust_improvement=any(bool(row["robust"]) for row in candidate_rows),
        candidates=tuple(candidate_rows),
    )


def evaluate_capacity_sequences(
    sequences: Sequence[object],
    *,
    capacities: Sequence[object] = CAPACITY_GRID,
    class_mapper: Callable[[int], int],
    background_class: int,
    metric_factory: Callable[[str], object],
    expected_sequence_count: int = 129,
    class_weight: float = 0.25,
    association_threshold: float = 0.5,
    update_rate: float = 0.2,
    max_update_rate: float = 0.2,
) -> CapacityEvaluation:
    """Evaluate the frozen Protocol B cache at every registered capacity."""

    from scripts.evaluate_persist4d_p6a import (
        CachedProtocolSequence,
        cache_payload_to_frozen_observation,
        prefix_causality_coordinator,
    )
    from scripts.system_comparison_analysis import (
        _persistent_identity_updates,
        _persistent_task_pair,
        aggregate_identity_metrics,
    )
    from scripts.system_comparison_metrics import (
        CausalTaskAccumulator,
        compute_causal_task_metrics,
        deployment_identity_metrics_by_horizon,
    )

    capacity_values = _validated_capacities(capacities)
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise ValueError("sequences must be a sequence")
    normalized_sequences = tuple(sequences)
    if (
        isinstance(expected_sequence_count, bool)
        or not isinstance(expected_sequence_count, Integral)
        or int(expected_sequence_count) <= 0
        or len(normalized_sequences) != int(expected_sequence_count)
    ):
        raise ValueError("sequence coverage differs from the registered protocol")
    if any(
        not isinstance(sequence, CachedProtocolSequence)
        for sequence in normalized_sequences
    ):
        raise ValueError("capacity evaluation requires cached Protocol B sequences")
    identities = {
        (sequence.master_sequence_id, sequence.order_id)
        for sequence in normalized_sequences
    }
    if len(identities) != len(normalized_sequences):
        raise ValueError("capacity evaluation contains duplicate sequence cells")
    if not callable(class_mapper) or not callable(metric_factory):
        raise ValueError("class mapper and metric factory must be callable")
    if (
        isinstance(background_class, bool)
        or not isinstance(background_class, Integral)
        or int(background_class) < 0
    ):
        raise ValueError("background_class must be a non-negative integer")

    horizons = (2, 3, 4, 5)
    task_accumulators = {
        (capacity, horizon): CausalTaskAccumulator(metric_factory=metric_factory)
        for capacity in capacity_values
        for horizon in horizons
    }
    per_sequence_rows: list[dict[str, object]] = []
    for sequence in normalized_sequences:
        sequence_id = f"{sequence.master_sequence_id}:{sequence.order_id}"
        frozen = tuple(
            cache_payload_to_frozen_observation(payload)
            for payload in sequence.payloads
        )
        capacity_replays = replay_capacity_grid(
            frozen,
            sequence_id=sequence_id,
            capacities=capacity_values,
            class_weight=class_weight,
            association_threshold=association_threshold,
            update_rate=update_rate,
            max_update_rate=max_update_rate,
        )
        for replay in capacity_replays:
            capacity = replay.capacity

            def tracker_factory(*, sequence_id: str) -> B4PersistentTracker:
                return B4PersistentTracker(
                    sequence_id=sequence_id,
                    capacity=capacity,
                    class_weight=class_weight,
                    association_threshold=association_threshold,
                    update_rate=update_rate,
                    max_update_rate=max_update_rate,
                )

            coordinated = prefix_causality_coordinator(
                sequence.payloads,
                {"B4": tracker_factory},
                endpoints=(1, 2, 3, 4),
                sequence_id=sequence_id,
                background_class=int(background_class),
            )
            if coordinated.content_digest != replay.observation_sha256:
                raise RuntimeError("capacity task replay observation digest differs")
            offline_steps = coordinated.offline_steps["B4"]
            _validate_timed_and_metric_replays(replay, offline_steps)
            identity_updates = _persistent_identity_updates(
                payloads=sequence.payloads,
                steps=offline_steps,
                class_mapper=class_mapper,
                background_class=int(background_class),
            )
            identity_by_horizon = deployment_identity_metrics_by_horizon(
                identity_updates
            )
            for horizon in horizons:
                pair = _persistent_task_pair(
                    payloads=sequence.payloads,
                    prediction=coordinated.online_predictions["B4"][horizon - 1],
                    horizon=horizon,
                    class_mapper=class_mapper,
                )
                task_accumulators[(capacity, horizon)].update(pair)
                task_metrics = compute_causal_task_metrics(
                    [pair], metric_factory=metric_factory
                )
                prefix = replay.stages[:horizon]
                occupied = [stage.occupied_count for stage in prefix]
                active = [stage.active_count for stage in prefix]
                dormant = [stage.dormant_count for stage in prefix]
                birth_opportunities = sum(stage.birth_attempts for stage in prefix)
                accepted_births = sum(stage.accepted_births for stage in prefix)
                rejected_births = sum(stage.rejected_births for stage in prefix)
                memory_update_latency = [
                    stage.memory_update_latency_ms for stage in prefix
                ]
                total_update_latency = [
                    stage.total_update_latency_ms for stage in prefix
                ]
                per_sequence_rows.append(
                    {
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "capacity": capacity,
                        "horizon": horizon,
                        "observation_sha256": replay.observation_sha256,
                        "peak_occupied_slots": max(occupied),
                        "mean_occupied_slots": math.fsum(occupied) / horizon,
                        "endpoint_occupied_slots": occupied[-1],
                        "peak_occupancy_ratio": max(occupied) / capacity,
                        "mean_occupancy_ratio": (
                            math.fsum(occupied) / horizon / capacity
                        ),
                        "peak_active_slots": max(active),
                        "mean_active_slots": math.fsum(active) / horizon,
                        "endpoint_active_slots": active[-1],
                        "peak_dormant_slots": max(dormant),
                        "mean_dormant_slots": math.fsum(dormant) / horizon,
                        "endpoint_dormant_slots": dormant[-1],
                        "birth_opportunities": birth_opportunities,
                        "accepted_births": accepted_births,
                        "rejected_births": rejected_births,
                        "birth_rejection_rate": (
                            rejected_births / birth_opportunities
                            if birth_opportunities
                            else None
                        ),
                        "state_bytes": prefix[-1].state_bytes,
                        "mean_memory_update_latency_ms": (
                            math.fsum(memory_update_latency) / horizon
                        ),
                        "mean_total_update_latency_ms": (
                            math.fsum(total_update_latency) / horizon
                        ),
                        **task_metrics,
                        **identity_by_horizon[horizon],
                    }
                )

    per_sequence_rows.sort(
        key=lambda row: (
            int(row["capacity"]),
            int(row["horizon"]),
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
        )
    )
    aggregate_rows = []
    for capacity in capacity_values:
        for horizon in horizons:
            selected = [
                row
                for row in per_sequence_rows
                if row["capacity"] == capacity and row["horizon"] == horizon
            ]
            if len(selected) != int(expected_sequence_count):
                raise RuntimeError("capacity aggregate coverage differs")
            state_bytes = {int(row["state_bytes"]) for row in selected}
            if len(state_bytes) != 1:
                raise RuntimeError("fixed-capacity state bytes differ across sequences")
            birth_opportunities = sum(
                int(row["birth_opportunities"]) for row in selected
            )
            accepted_births = sum(int(row["accepted_births"]) for row in selected)
            rejected_births = sum(int(row["rejected_births"]) for row in selected)
            aggregate_rows.append(
                {
                    "capacity": capacity,
                    "horizon": horizon,
                    "sequence_count": len(selected),
                    "reference_scene_count": len(
                        {str(row["reference_scene_id"]) for row in selected}
                    ),
                    **_distribution_fields(
                        selected,
                        source="peak_occupied_slots",
                        prefix="peak_occupied_slots",
                    ),
                    **_distribution_fields(
                        selected,
                        source="mean_occupied_slots",
                        prefix="mean_occupied_slots",
                    ),
                    **_distribution_fields(
                        selected,
                        source="peak_occupancy_ratio",
                        prefix="peak_occupancy_ratio",
                    ),
                    **_distribution_fields(
                        selected,
                        source="peak_active_slots",
                        prefix="peak_active_slots",
                    ),
                    **_distribution_fields(
                        selected,
                        source="peak_dormant_slots",
                        prefix="peak_dormant_slots",
                    ),
                    "birth_opportunities": birth_opportunities,
                    "accepted_births": accepted_births,
                    "rejected_births": rejected_births,
                    "birth_rejection_rate": (
                        rejected_births / birth_opportunities
                        if birth_opportunities
                        else None
                    ),
                    "state_bytes": next(iter(state_bytes)),
                    **_distribution_fields(
                        selected,
                        source="mean_memory_update_latency_ms",
                        prefix="memory_update_latency_ms",
                    ),
                    **_distribution_fields(
                        selected,
                        source="mean_total_update_latency_ms",
                        prefix="total_update_latency_ms",
                    ),
                    **task_accumulators[(capacity, horizon)].compute(),
                    **aggregate_identity_metrics(selected),
                }
            )
    return CapacityEvaluation(
        per_sequence_rows=tuple(per_sequence_rows),
        aggregate_rows=tuple(aggregate_rows),
    )


@dataclass(frozen=True, slots=True)
class CapacityStageReplay:
    stage_id: int
    capacity: int
    occupied_count: int
    active_count: int
    dormant_count: int
    birth_attempts: int
    accepted_births: int
    rejected_births: int
    birth_acceptance_rate: float | None
    state_bytes: int
    association_latency_ms: float
    memory_update_latency_ms: float
    memory_latency_ms: float
    total_update_latency_ms: float


@dataclass(frozen=True, slots=True)
class CapacityReplay:
    sequence_id: str
    capacity: int
    observation_sha256: str
    stages: tuple[CapacityStageReplay, ...]


def _validated_capacities(capacities: Sequence[object]) -> tuple[int, ...]:
    if isinstance(capacities, (str, bytes)) or not isinstance(capacities, Sequence):
        raise ValueError("capacities must be a non-empty sequence of unique integers")
    values = tuple(capacities)
    if (
        not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError("capacities must be a non-empty sequence of unique integers")
    return tuple(int(value) for value in values)


def _state_bytes(state: object) -> int:
    tensors = getattr(state, "tensors", None)
    if not callable(tensors):
        raise TypeError("capacity replay state must expose tensors")
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors()))


def replay_capacity_grid(
    observations: Sequence[
        FrozenObservation | LocalInstanceObservation | Mapping[str, Any]
    ],
    *,
    sequence_id: str,
    capacities: Sequence[object] = CAPACITY_GRID,
    class_weight: float = 0.25,
    association_threshold: float = 0.5,
    update_rate: float = 0.2,
    max_update_rate: float = 0.2,
) -> tuple[CapacityReplay, ...]:
    """Replay one immutable observation sequence at each requested capacity."""

    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be a non-empty string")
    capacity_values = _validated_capacities(capacities)
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise ValueError("observations must be a non-empty sequence")
    frozen = tuple(freeze_observation(observation) for observation in observations)
    if not frozen:
        raise ValueError("observations must be a non-empty sequence")
    observation_sha256 = observation_content_digest(frozen)

    replays = []
    for capacity in capacity_values:
        tracker = B4PersistentTracker(
            sequence_id=sequence_id,
            capacity=capacity,
            class_weight=class_weight,
            association_threshold=association_threshold,
            update_rate=update_rate,
            max_update_rate=max_update_rate,
        )
        stages = []
        for stage_id, observation in enumerate(frozen):
            timing_events: list[Mapping[str, float]] = []
            start_ns = time.perf_counter_ns()
            step = tracker.step(
                observation,
                stage_id=stage_id,
                timing_sink=timing_events.append,
            )
            total_update_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
            if len(timing_events) != 1:
                raise RuntimeError(
                    "capacity replay must record one memory timing event"
                )
            timing = timing_events[0]
            association_latency_ms = float(timing["association_overhead_ms"])
            memory_update_latency_ms = float(timing["memory_update_overhead_ms"])
            memory_latency_ms = association_latency_ms + memory_update_latency_ms
            state = step.state_snapshot
            occupied_count = int(state.occupied[0].sum().item())
            active_count = int(state.active[0].sum().item())
            accepted_births = sum(value is True for value in step.births)
            rejected_births = sum(value is True for value in step.rejected_births)
            birth_attempts = accepted_births + rejected_births
            stages.append(
                CapacityStageReplay(
                    stage_id=stage_id,
                    capacity=capacity,
                    occupied_count=occupied_count,
                    active_count=active_count,
                    dormant_count=occupied_count - active_count,
                    birth_attempts=birth_attempts,
                    accepted_births=accepted_births,
                    rejected_births=rejected_births,
                    birth_acceptance_rate=(
                        accepted_births / birth_attempts if birth_attempts else None
                    ),
                    state_bytes=_state_bytes(state),
                    association_latency_ms=association_latency_ms,
                    memory_update_latency_ms=memory_update_latency_ms,
                    memory_latency_ms=memory_latency_ms,
                    total_update_latency_ms=total_update_latency_ms,
                )
            )
        replays.append(
            CapacityReplay(
                sequence_id=sequence_id,
                capacity=capacity,
                observation_sha256=observation_sha256,
                stages=tuple(stages),
            )
        )
    result = tuple(replays)
    validate_capacity_replays(result)
    return result


def validate_capacity_replays(replays: Sequence[CapacityReplay]) -> None:
    """Fail closed on capacity, observation, state, or timing drift."""

    if not isinstance(replays, Sequence) or not replays:
        raise ValueError("capacity replays must be a non-empty sequence")
    if any(not isinstance(replay, CapacityReplay) for replay in replays):
        raise ValueError("capacity replays must contain CapacityReplay values")
    capacities = tuple(replay.capacity for replay in replays)
    _validated_capacities(capacities)
    sequence_ids = {replay.sequence_id for replay in replays}
    if len(sequence_ids) != 1:
        raise ValueError("capacity replays must use one sequence identity")
    observation_sha256 = {replay.observation_sha256 for replay in replays}
    if len(observation_sha256) != 1:
        raise ValueError("capacity replays must use identical frozen observations")
    digest = next(iter(observation_sha256))
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("capacity replay observation digest must be SHA-256")
    stage_count = len(replays[0].stages)
    if stage_count == 0 or any(len(replay.stages) != stage_count for replay in replays):
        raise ValueError("capacity replays must cover identical non-empty stages")
    for replay in replays:
        for stage_id, stage in enumerate(replay.stages):
            if stage.stage_id != stage_id or stage.capacity != replay.capacity:
                raise ValueError("capacity replay stages must be contiguous and bound")
            if not 0 <= stage.active_count <= stage.occupied_count <= replay.capacity:
                raise ValueError("capacity replay occupancy is invalid")
            if stage.dormant_count != stage.occupied_count - stage.active_count:
                raise ValueError("capacity replay dormant count is invalid")
            if stage.birth_attempts != stage.accepted_births + stage.rejected_births:
                raise ValueError("capacity replay birth accounting is invalid")
            expected_rate = (
                stage.accepted_births / stage.birth_attempts
                if stage.birth_attempts
                else None
            )
            if stage.birth_acceptance_rate != expected_rate:
                raise ValueError("capacity replay birth acceptance rate is invalid")
            if stage.state_bytes <= 0:
                raise ValueError("capacity replay state bytes must be positive")
            timings = (
                stage.association_latency_ms,
                stage.memory_update_latency_ms,
                stage.memory_latency_ms,
                stage.total_update_latency_ms,
            )
            if any(not math.isfinite(value) or value < 0 for value in timings):
                raise ValueError(
                    "capacity replay timings must be finite and non-negative"
                )
            if not math.isclose(
                stage.memory_latency_ms,
                stage.association_latency_ms + stage.memory_update_latency_ms,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("capacity replay memory timing is inconsistent")


def _validate_timed_and_metric_replays(
    replay: CapacityReplay,
    metric_steps: Sequence[object],
) -> None:
    if len(metric_steps) != len(replay.stages):
        raise RuntimeError("timed and metric capacity replays cover different stages")
    for recorded, step in zip(replay.stages, metric_steps, strict=True):
        state = getattr(step, "state_snapshot", None)
        if state is None:
            raise RuntimeError("metric replay step lacks a state snapshot")
        occupied = int(state.occupied[0].sum().item())
        active = int(state.active[0].sum().item())
        accepted_births = sum(value is True for value in step.births)
        rejected_births = sum(value is True for value in step.rejected_births)
        if (
            occupied != recorded.occupied_count
            or active != recorded.active_count
            or accepted_births != recorded.accepted_births
            or rejected_births != recorded.rejected_births
        ):
            raise RuntimeError("timed and metric capacity replay state differs")


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution_fields(
    rows: Sequence[Mapping[str, object]],
    *,
    source: str,
    prefix: str,
) -> dict[str, float]:
    values = [float(row[source]) for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"{source} distribution is invalid")
    return {
        f"{prefix}_mean": math.fsum(values) / len(values),
        f"{prefix}_median": _linear_quantile(values, 0.5),
        f"{prefix}_q25": _linear_quantile(values, 0.25),
        f"{prefix}_q75": _linear_quantile(values, 0.75),
        f"{prefix}_max": max(values),
    }


__all__ = (
    "CAPACITY_GRID",
    "CapacityBootstrap",
    "CapacityEvaluation",
    "CapacityReplay",
    "CapacityRobustness",
    "CapacityStageReplay",
    "assess_robust_capacity_improvement",
    "build_class_mapper_from_label_document",
    "build_protocol_from_reviewer_manifest",
    "capacity_cluster_bootstrap",
    "classify_capacity_gate",
    "evaluate_capacity_sequences",
    "replay_capacity_grid",
    "validate_capacity_replays",
)
