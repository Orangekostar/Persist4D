"""Causal replay, eligibility, and deterministic selection for P6-B."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

from scripts.evaluate_persist4d_p6a import (
    CachedProtocolSequence,
    build_association_events,
    cache_payload_to_frozen_observation,
    observation_content_digest,
)
from scripts.p6a_analysis import aggregate_event_metrics, validate_association_events
from scripts.p6a_association import FrozenObservation, TrackStep, freeze_observation
from scripts.p6b_association import P6BTracker, P6BTransitionDetails
from scripts.p6b_protocol import (
    P6BEligibility,
    P6BMemoryConfig,
    P6BProtocolConfig,
    canonical_config_id,
    canonical_config_json,
    expand_stage_configs,
    joint_neighbor_configs,
)


class P6BSweepError(ValueError):
    pass


_HORIZONS = (2, 3, 4, 5)
_ORDER_IDS = frozenset({"canonical", "reverse", "sha256_seed45"})


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise P6BSweepError(f"{name} must be a nonempty string")
    return value


def _finite_optional(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise P6BSweepError(f"{name} must be finite or None")
    number = float(value)
    if not math.isfinite(number):
        raise P6BSweepError(f"{name} must be finite or None")
    return number


@dataclass(frozen=True)
class P6BSweepSequence:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    observations: tuple[FrozenObservation, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.reference_scene_id, "reference_scene_id")
        _nonempty_string(self.master_sequence_id, "master_sequence_id")
        if self.order_id not in _ORDER_IDS:
            raise P6BSweepError("order_id must be a frozen P6-A order")
        if not isinstance(self.observations, tuple) or len(self.observations) != 5:
            raise P6BSweepError("observations must contain exactly five stages")
        frozen = []
        for observation in self.observations:
            if not isinstance(observation, FrozenObservation):
                raise P6BSweepError(
                    "observations must contain FrozenObservation values"
                )
            frozen.append(freeze_observation(observation))
        object.__setattr__(self, "observations", tuple(frozen))


@dataclass(frozen=True)
class P6BSequenceReplay:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    steps: tuple[TrackStep, ...]
    transitions: tuple[P6BTransitionDetails, ...]
    prefix_steps: Mapping[int, tuple[TrackStep, ...]]


@dataclass(frozen=True)
class P6BReplay:
    config: P6BMemoryConfig
    sequences: tuple[P6BSequenceReplay, ...]


def cached_sequences_to_sweep_sequences(
    sequences: Sequence[CachedProtocolSequence],
) -> tuple[P6BSweepSequence, ...]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise P6BSweepError("cached sequences must be a sequence")
    if not sequences:
        raise P6BSweepError("cached sequences must not be empty")
    identities: set[tuple[str, str]] = set()
    converted = []
    for sequence in sequences:
        if not isinstance(sequence, CachedProtocolSequence):
            raise P6BSweepError(
                "cached sequences must contain CachedProtocolSequence values"
            )
        identity = (sequence.master_sequence_id, sequence.order_id)
        if identity in identities:
            raise P6BSweepError("cached sequences contain a duplicate master/order")
        identities.add(identity)
        converted.append(
            P6BSweepSequence(
                reference_scene_id=sequence.reference_scene_id,
                master_sequence_id=sequence.master_sequence_id,
                order_id=sequence.order_id,
                observations=tuple(
                    cache_payload_to_frozen_observation(payload)
                    for payload in sequence.payloads
                ),
            )
        )
    return tuple(converted)


def derive_prefix_events(
    replay: P6BReplay,
    cached_sequences: Sequence[CachedProtocolSequence],
    *,
    background_class: int,
) -> Mapping[int, tuple[object, ...]]:
    if not isinstance(replay, P6BReplay):
        raise P6BSweepError("replay must be a P6BReplay")
    try:
        cached = {
            (sequence.master_sequence_id, sequence.order_id): sequence
            for sequence in cached_sequences
        }
    except AttributeError as error:
        raise P6BSweepError("cached source identity is invalid") from error
    if len(cached) != len(cached_sequences):
        raise P6BSweepError("cached sources contain duplicate identities")
    replay_identities = {
        (sequence.master_sequence_id, sequence.order_id)
        for sequence in replay.sequences
    }
    if set(cached) != replay_identities:
        raise P6BSweepError("cached source identities differ from replay identities")
    by_horizon: dict[int, list[object]] = {horizon: [] for horizon in _HORIZONS}
    for sequence_replay in replay.sequences:
        identity = (
            sequence_replay.master_sequence_id,
            sequence_replay.order_id,
        )
        source = cached[identity]
        digest = observation_content_digest(
            tuple(
                cache_payload_to_frozen_observation(payload)
                for payload in source.payloads
            )
        )
        for horizon in _HORIZONS:
            by_horizon[horizon].extend(
                build_association_events(
                    source.payloads[:horizon],
                    sequence_replay.prefix_steps[horizon],
                    method="P6B",
                    reference_scene_id=source.reference_scene_id,
                    master_sequence_id=source.master_sequence_id,
                    order_id=source.order_id,
                    prefix=horizon,
                    cache_digest=digest,
                    background_class=background_class,
                )
            )
    return MappingProxyType(
        {horizon: tuple(by_horizon[horizon]) for horizon in _HORIZONS}
    )


def extract_official_metrics(
    evaluation: object,
    *,
    method: str = "P6B",
) -> Mapping[int, Mapping[str, float]]:
    metric_blocks = getattr(evaluation, "metric_blocks", None)
    if not isinstance(metric_blocks, Mapping):
        raise P6BSweepError("evaluation must expose metric_blocks")
    strict_block = metric_blocks.get("strict")
    if not isinstance(method, str) or not method:
        raise P6BSweepError("official metric method must be nonempty")
    if not isinstance(strict_block, Mapping) or set(strict_block) != {method}:
        raise P6BSweepError(f"strict metric block must contain exact {method} method")
    method_block = strict_block[method]
    if not isinstance(method_block, Mapping) or set(method_block) != {
        f"T{horizon}" for horizon in _HORIZONS
    }:
        raise P6BSweepError("P6B strict metric block must contain exact T2-T5 keys")
    result: dict[int, Mapping[str, float]] = {}
    for horizon in _HORIZONS:
        values = method_block[f"T{horizon}"]
        if not isinstance(values, Mapping):
            raise P6BSweepError("P6B strict horizon metric must be a mapping")
        tmap = _finite_optional(values.get("online_t-mAP"), "online_t-mAP")
        trec = _finite_optional(values.get("online_t-REC"), "online_t-REC")
        if tmap is None or trec is None or not 0.0 <= tmap <= 1.0 or not 0.0 <= trec <= 1.0:
            raise P6BSweepError("P6B strict task metrics must be finite in [0, 1]")
        result[horizon] = MappingProxyType({"t_mAP": tmap, "t_REC": trec})
    return MappingProxyType(result)


def replay_configuration(
    sequences: Sequence[P6BSweepSequence],
    config: P6BMemoryConfig,
    *,
    allowed_reference_scene_ids: Sequence[str],
) -> P6BReplay:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise P6BSweepError("sequences must be a sequence")
    if not sequences:
        raise P6BSweepError("sequences must not be empty")
    if not isinstance(config, P6BMemoryConfig):
        raise P6BSweepError("config must be a P6BMemoryConfig")
    allowed = tuple(
        _nonempty_string(value, "allowed reference")
        for value in allowed_reference_scene_ids
    )
    if not allowed or len(set(allowed)) != len(allowed):
        raise P6BSweepError("allowed references must be nonempty and unique")
    allowed_set = set(allowed)
    identities: set[tuple[str, str]] = set()
    results = []
    for sequence in sequences:
        if not isinstance(sequence, P6BSweepSequence):
            raise P6BSweepError(
                "sequences must contain P6BSweepSequence values"
            )
        if sequence.reference_scene_id not in allowed_set:
            raise P6BSweepError(
                "sequence reference is heldout or absent from the allowed tuning split"
            )
        identity = (sequence.master_sequence_id, sequence.order_id)
        if identity in identities:
            raise P6BSweepError("sequences contain a duplicate master/order")
        identities.add(identity)
        tracker = P6BTracker(
            sequence_id=f"{sequence.master_sequence_id}:{sequence.order_id}",
            config=config,
        )
        steps = []
        transitions = []
        for stage, observation in enumerate(sequence.observations):
            steps.append(tracker.step(observation, stage_id=stage))
            if tracker.last_transition is None:
                raise RuntimeError("P6-B tracker omitted transition details")
            transitions.append(tracker.last_transition)
        frozen_steps = tuple(steps)
        results.append(
            P6BSequenceReplay(
                reference_scene_id=sequence.reference_scene_id,
                master_sequence_id=sequence.master_sequence_id,
                order_id=sequence.order_id,
                steps=frozen_steps,
                transitions=tuple(transitions),
                prefix_steps=MappingProxyType(
                    {horizon: frozen_steps[:horizon] for horizon in _HORIZONS}
                ),
            )
        )
    return P6BReplay(config=config, sequences=tuple(results))


def _aggregate_events(events: Iterable[object]) -> dict[str, int | float | None]:
    try:
        identity, reactivation = aggregate_event_metrics(events)
    except (TypeError, ValueError) as error:
        raise P6BSweepError(f"invalid association events: {error}") from error
    return {
        "identity_switches": int(identity["id_switches"]),
        "transition_opportunities": int(identity["transition_opportunities"]),
        "wrong_reactivations": int(reactivation["wrong_reactivations"]),
        "predicted_reactivation_events": int(
            reactivation["predicted_reactivation_events"]
        ),
        "correct_reactivations": int(reactivation["correct_reactivations"]),
        "reactivation_attempts": int(reactivation["reactivation_attempts"]),
        "gap_opportunities": int(reactivation["gap_opportunities"]),
        "false_births": int(identity["false_births"]),
        "births": int(identity["births"]),
        "rejected_births": int(identity["rejected_births"]),
        "reactivation_accuracy": reactivation["reactivation_accuracy"],
        "reactivation_recall": reactivation["reactivation_recall"],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class P6BClusterMetrics:
    reference_scene_id: str
    identity_switches: int
    transition_opportunities: int
    wrong_reactivations: int
    predicted_reactivation_events: int
    correct_reactivations: int
    reactivation_attempts: int
    gap_opportunities: int
    false_births: int
    births: int
    rejected_births: int

    def __post_init__(self) -> None:
        _nonempty_string(self.reference_scene_id, "reference_scene_id")
        for name in (
            "identity_switches",
            "transition_opportunities",
            "wrong_reactivations",
            "predicted_reactivation_events",
            "correct_reactivations",
            "reactivation_attempts",
            "gap_opportunities",
            "false_births",
            "births",
            "rejected_births",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise P6BSweepError(f"{name} must be a nonnegative integer")
        if self.identity_switches > self.transition_opportunities:
            raise P6BSweepError("identity switches exceed transition opportunities")
        if not (
            self.correct_reactivations
            <= self.reactivation_attempts
            <= self.gap_opportunities
        ):
            raise P6BSweepError("reactivation counts exceed their opportunities")
        if self.correct_reactivations > self.predicted_reactivation_events:
            raise P6BSweepError("correct reactivations exceed predicted events")
        if (
            self.wrong_reactivations
            != self.predicted_reactivation_events - self.correct_reactivations
        ):
            raise P6BSweepError("wrong reactivations differ from predicted minus correct")
        if self.false_births > self.births:
            raise P6BSweepError("false births exceed accepted births")

    @property
    def identity_switch_rate(self) -> float | None:
        return _ratio(self.identity_switches, self.transition_opportunities)

    @property
    def wrong_reactivation_rate(self) -> float | None:
        return _ratio(self.wrong_reactivations, self.predicted_reactivation_events)

    @property
    def false_birth_rate(self) -> float | None:
        return _ratio(self.false_births, self.births + self.rejected_births)


def cluster_event_metrics(events: Iterable[object]) -> tuple[P6BClusterMetrics, ...]:
    try:
        validated = validate_association_events(events)
    except (TypeError, ValueError) as error:
        raise P6BSweepError(f"invalid association events: {error}") from error
    grouped: dict[str, list[object]] = {}
    for event in validated:
        grouped.setdefault(event.reference_scene_id, []).append(event)
    result = []
    for reference_scene_id, cluster_events in sorted(grouped.items()):
        aggregate = _aggregate_events(cluster_events)
        result.append(
            P6BClusterMetrics(
                reference_scene_id=reference_scene_id,
                **{
                    name: int(aggregate[name])
                    for name in (
                        "identity_switches",
                        "transition_opportunities",
                        "wrong_reactivations",
                        "predicted_reactivation_events",
                        "correct_reactivations",
                        "reactivation_attempts",
                        "gap_opportunities",
                        "false_births",
                        "births",
                        "rejected_births",
                    )
                },
            )
        )
    if not result:
        raise P6BSweepError("cluster metrics require association events")
    return tuple(result)


def _observation_counts(
    replay: P6BReplay, horizon: int
) -> tuple[int, int]:
    accepted = 0
    total = 0
    for sequence in replay.sequences:
        for step in sequence.prefix_steps[horizon]:
            for valid, track_id in zip(step.valid, step.track_ids, strict=True):
                total += int(valid)
                accepted += int(valid and track_id is not None)
    return accepted, total


def build_candidate_row(
    replay: P6BReplay,
    *,
    stage: str,
    events_by_horizon: Mapping[int, Iterable[object]],
    official_metrics: Mapping[int, Mapping[str, object]] | None = None,
) -> P6BCandidateRow:
    if set(events_by_horizon) != set(_HORIZONS):
        raise P6BSweepError("events_by_horizon must contain exact T2-T5 keys")
    if official_metrics is not None and set(official_metrics) != set(_HORIZONS):
        raise P6BSweepError("official_metrics must contain exact T2-T5 keys")
    horizons = []
    for horizon in _HORIZONS:
        events = tuple(events_by_horizon[horizon])
        aggregate = _aggregate_events(events)
        clusters = cluster_event_metrics(events)
        accepted, total = _observation_counts(replay, horizon)
        if official_metrics is None:
            tmap = None
            trec = None
        else:
            block = official_metrics[horizon]
            if set(block) != {"t_mAP", "t_REC"}:
                raise P6BSweepError(
                    "official metric blocks must contain t_mAP and t_REC"
                )
            tmap = _finite_optional(block["t_mAP"], "t_mAP")
            trec = _finite_optional(block["t_REC"], "t_REC")
        horizons.append(
            P6BHorizonMetrics(
                horizon=horizon,
                identity_switches=int(aggregate["identity_switches"]),
                transition_opportunities=int(
                    aggregate["transition_opportunities"]
                ),
                wrong_reactivations=int(aggregate["wrong_reactivations"]),
                predicted_reactivation_events=int(
                    aggregate["predicted_reactivation_events"]
                ),
                correct_reactivations=int(aggregate["correct_reactivations"]),
                reactivation_attempts=int(aggregate["reactivation_attempts"]),
                gap_opportunities=int(aggregate["gap_opportunities"]),
                false_births=int(aggregate["false_births"]),
                births=int(aggregate["births"]),
                rejected_births=int(aggregate["rejected_births"]),
                reactivation_accuracy=aggregate["reactivation_accuracy"],
                reactivation_recall=aggregate["reactivation_recall"],
                accepted_valid_observations=accepted,
                total_valid_observations=total,
                cluster_metrics=clusters,
                strict_online_tmap=tmap,
                strict_online_trec=trec,
            )
        )
    return P6BCandidateRow(
        config=replay.config,
        stage=stage,
        horizons=tuple(horizons),
    )


@dataclass(frozen=True)
class P6BHorizonMetrics:
    horizon: int
    identity_switches: int
    transition_opportunities: int
    wrong_reactivations: int
    predicted_reactivation_events: int
    correct_reactivations: int
    reactivation_attempts: int
    gap_opportunities: int
    false_births: int
    births: int
    rejected_births: int
    reactivation_accuracy: float | None
    reactivation_recall: float | None
    accepted_valid_observations: int
    total_valid_observations: int
    cluster_metrics: tuple[P6BClusterMetrics, ...]
    strict_online_tmap: float | None = None
    strict_online_trec: float | None = None

    def __post_init__(self) -> None:
        if self.horizon not in _HORIZONS:
            raise P6BSweepError("horizon must be one of T2-T5")
        for name in (
            "identity_switches",
            "transition_opportunities",
            "wrong_reactivations",
            "predicted_reactivation_events",
            "correct_reactivations",
            "reactivation_attempts",
            "gap_opportunities",
            "false_births",
            "births",
            "rejected_births",
            "accepted_valid_observations",
            "total_valid_observations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise P6BSweepError(f"{name} must be a nonnegative integer")
        if self.accepted_valid_observations > self.total_valid_observations:
            raise P6BSweepError("accepted observations cannot exceed total observations")
        if not isinstance(self.cluster_metrics, tuple) or not self.cluster_metrics:
            raise P6BSweepError("cluster_metrics must be a nonempty tuple")
        if any(
            not isinstance(item, P6BClusterMetrics) for item in self.cluster_metrics
        ):
            raise P6BSweepError("cluster_metrics contain an invalid record")
        references = tuple(item.reference_scene_id for item in self.cluster_metrics)
        if references != tuple(sorted(set(references))):
            raise P6BSweepError("cluster_metrics must be uniquely sorted by reference")
        for name in (
            "identity_switches",
            "transition_opportunities",
            "wrong_reactivations",
            "predicted_reactivation_events",
            "correct_reactivations",
            "reactivation_attempts",
            "gap_opportunities",
            "false_births",
            "births",
            "rejected_births",
        ):
            if getattr(self, name) != sum(
                getattr(item, name) for item in self.cluster_metrics
            ):
                raise P6BSweepError(f"{name} differs from cluster totals")
        expected_accuracy = _ratio(
            self.correct_reactivations, self.reactivation_attempts
        )
        expected_recall = _ratio(self.correct_reactivations, self.gap_opportunities)
        for name, expected in (
            ("reactivation_accuracy", expected_accuracy),
            ("reactivation_recall", expected_recall),
        ):
            actual = getattr(self, name)
            if (actual is None) != (expected is None) or (
                actual is not None
                and expected is not None
                and not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise P6BSweepError(f"{name} differs from count-derived rate")
        for name in (
            "reactivation_accuracy",
            "reactivation_recall",
            "strict_online_tmap",
            "strict_online_trec",
        ):
            value = _finite_optional(getattr(self, name), name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise P6BSweepError(f"{name} must be within [0, 1]")

    @property
    def identity_switch_rate(self) -> float | None:
        return _ratio(self.identity_switches, self.transition_opportunities)

    @property
    def wrong_reactivation_rate(self) -> float | None:
        return _ratio(self.wrong_reactivations, self.predicted_reactivation_events)

    @property
    def false_birth_rate(self) -> float | None:
        return _ratio(self.false_births, self.births + self.rejected_births)

    def _cluster_mean(self, name: str) -> float | None:
        values = [
            getattr(item, name)
            for item in self.cluster_metrics
            if getattr(item, name) is not None
        ]
        return sum(values) / len(values) if values else None

    @property
    def cluster_mean_identity_switch_rate(self) -> float | None:
        return self._cluster_mean("identity_switch_rate")

    @property
    def cluster_mean_wrong_reactivation_rate(self) -> float | None:
        return self._cluster_mean("wrong_reactivation_rate")

    @property
    def cluster_mean_false_birth_rate(self) -> float | None:
        return self._cluster_mean("false_birth_rate")


@dataclass(frozen=True)
class P6BCandidateRow:
    config: P6BMemoryConfig
    stage: str
    horizons: tuple[P6BHorizonMetrics, ...]
    eligibility_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.config, P6BMemoryConfig):
            raise P6BSweepError("candidate config must be a P6BMemoryConfig")
        _nonempty_string(self.stage, "candidate stage")
        if not isinstance(self.horizons, tuple):
            raise P6BSweepError("candidate horizons must be a tuple")
        if tuple(item.horizon for item in self.horizons) != _HORIZONS:
            raise P6BSweepError("candidate horizons must be ordered T2-T5")
        if not isinstance(self.eligibility_reasons, tuple) or any(
            not isinstance(reason, str) or not reason
            for reason in self.eligibility_reasons
        ):
            raise P6BSweepError("eligibility reasons must be nonempty strings")

    @property
    def config_id(self) -> str:
        return canonical_config_id(self.config)

    @property
    def config_json(self) -> str:
        return canonical_config_json(self.config)

    @property
    def eligible(self) -> bool:
        return not self.eligibility_reasons

    def metric(self, horizon: int) -> P6BHorizonMetrics:
        if horizon not in _HORIZONS:
            raise P6BSweepError("horizon must be one of T2-T5")
        return self.horizons[horizon - 2]

    @property
    def official_metrics_complete(self) -> bool:
        return all(
            item.strict_online_tmap is not None
            and item.strict_online_trec is not None
            for item in self.horizons
        )


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _mean_required(values: Iterable[float | None], name: str) -> float:
    materialized = tuple(values)
    if not materialized or any(value is None for value in materialized):
        raise P6BSweepError(f"{name} requires a rate for every requested horizon")
    return sum(float(value) for value in materialized) / len(materialized)


def assess_candidate(
    candidate: P6BCandidateRow,
    *,
    baseline: P6BCandidateRow,
    eligibility: P6BEligibility,
    require_official_metrics: bool,
) -> P6BCandidateRow:
    reasons = []
    reactivation_horizons = (3, 4, 5)
    accuracy = _mean_present(
        candidate.metric(horizon).reactivation_accuracy
        for horizon in reactivation_horizons
    )
    if accuracy is None or accuracy < eligibility.minimum_reactivation_accuracy:
        reasons.append("reactivation_accuracy_below_minimum")
    candidate_recall = _mean_present(
        candidate.metric(horizon).reactivation_recall
        for horizon in reactivation_horizons
    )
    baseline_recall = _mean_present(
        baseline.metric(horizon).reactivation_recall
        for horizon in reactivation_horizons
    )
    if (
        candidate_recall is None
        or baseline_recall is None
        or baseline_recall - candidate_recall
        > eligibility.maximum_reactivation_recall_drop
    ):
        reasons.append("reactivation_recall_drop_exceeded")
    if any(
        candidate.metric(horizon).accepted_valid_observations
        < eligibility.minimum_valid_observation_ratio
        * baseline.metric(horizon).accepted_valid_observations
        for horizon in _HORIZONS
    ):
        reasons.append("valid_observation_ratio_below_minimum")
    if require_official_metrics:
        if not candidate.official_metrics_complete or not baseline.official_metrics_complete:
            reasons.append("official_task_metrics_missing")
        else:
            candidate_t2 = candidate.metric(2)
            baseline_t2 = baseline.metric(2)
            assert candidate_t2.strict_online_tmap is not None
            assert candidate_t2.strict_online_trec is not None
            assert baseline_t2.strict_online_tmap is not None
            assert baseline_t2.strict_online_trec is not None
            if (
                baseline_t2.strict_online_tmap - candidate_t2.strict_online_tmap
                > eligibility.maximum_t2_task_drop
                or baseline_t2.strict_online_trec - candidate_t2.strict_online_trec
                > eligibility.maximum_t2_task_drop
            ):
                reasons.append("t2_task_drop_exceeded")
    return replace(candidate, eligibility_reasons=tuple(reasons))


def candidate_ranking_key(row: P6BCandidateRow) -> tuple[float | str, ...]:
    if not row.official_metrics_complete:
        raise P6BSweepError("official task metrics are required for final ranking")
    t4 = row.metric(4)
    t5 = row.metric(5)
    reactivation_recall = _mean_present(
        row.metric(horizon).reactivation_recall for horizon in (3, 4, 5)
    )
    if reactivation_recall is None:
        raise P6BSweepError("reactivation recall is required for final ranking")
    task_values = (
        t4.strict_online_tmap,
        t4.strict_online_trec,
        t5.strict_online_tmap,
        t5.strict_online_trec,
    )
    if any(value is None for value in task_values):
        raise P6BSweepError("T4/T5 official task metrics are required")
    return (
        _mean_required(
            (
                row.metric(horizon).cluster_mean_identity_switch_rate
                for horizon in (4, 5)
            ),
            "identity-switch ranking",
        ),
        _mean_required(
            (
                row.metric(horizon).cluster_mean_wrong_reactivation_rate
                for horizon in (3, 4, 5)
            ),
            "wrong-reactivation ranking",
        ),
        _mean_required(
            (
                row.metric(horizon).cluster_mean_false_birth_rate
                for horizon in _HORIZONS
            ),
            "false-birth ranking",
        ),
        -reactivation_recall,
        -sum(float(value) for value in task_values if value is not None) / 4.0,
        row.config_json,
    )


def _coarse_objectives(row: P6BCandidateRow) -> tuple[float, ...]:
    recall = _mean_present(
        row.metric(horizon).reactivation_recall for horizon in (3, 4, 5)
    )
    if recall is None:
        recall = -1.0
    return (
        _mean_required(
            (
                row.metric(horizon).cluster_mean_identity_switch_rate
                for horizon in (4, 5)
            ),
            "identity-switch objective",
        ),
        _mean_required(
            (
                row.metric(horizon).cluster_mean_wrong_reactivation_rate
                for horizon in (3, 4, 5)
            ),
            "wrong-reactivation objective",
        ),
        _mean_required(
            (
                row.metric(horizon).cluster_mean_false_birth_rate
                for horizon in _HORIZONS
            ),
            "false-birth objective",
        ),
        -recall,
    )


def pareto_finalists(
    rows: Sequence[P6BCandidateRow],
    *,
    baseline: P6BCandidateRow,
    eligibility: P6BEligibility,
) -> tuple[P6BCandidateRow, ...]:
    assessed = [
        assess_candidate(
            row,
            baseline=baseline,
            eligibility=eligibility,
            require_official_metrics=False,
        )
        for row in rows
    ]
    eligible_rows = [row for row in assessed if row.eligible]
    frontier = []
    for candidate in eligible_rows:
        objective = _coarse_objectives(candidate)
        dominated = any(
            all(left <= right for left, right in zip(other_objective, objective, strict=True))
            and any(left < right for left, right in zip(other_objective, objective, strict=True))
            for other in eligible_rows
            if other is not candidate
            for other_objective in (_coarse_objectives(other),)
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda row: row.config_json))


@dataclass(frozen=True)
class P6BSelection:
    selected: P6BCandidateRow
    assessed_rows: tuple[P6BCandidateRow, ...]
    ranking_key: tuple[float | str, ...]


def select_final_candidate(
    rows: Sequence[P6BCandidateRow],
    *,
    baseline: P6BCandidateRow,
    eligibility: P6BEligibility,
) -> P6BSelection:
    if not rows:
        raise P6BSweepError("candidate rows must not be empty")
    assessed = tuple(
        assess_candidate(
            row,
            baseline=baseline,
            eligibility=eligibility,
            require_official_metrics=True,
        )
        for row in rows
    )
    eligible_rows = [row for row in assessed if row.eligible]
    if not eligible_rows:
        if any(
            "official_task_metrics_missing" in row.eligibility_reasons
            for row in assessed
        ):
            raise P6BSweepError(
                "official task metrics are required before freezing a P6-B config"
            )
        raise P6BSweepError("no P6-B candidate satisfies the eligibility gates")
    selected = min(eligible_rows, key=candidate_ranking_key)
    return P6BSelection(
        selected=selected,
        assessed_rows=assessed,
        ranking_key=candidate_ranking_key(selected),
    )


@dataclass(frozen=True)
class P6BStagedSweepResult:
    candidate_rows: tuple[P6BCandidateRow, ...]
    finalist_rows: tuple[P6BCandidateRow, ...]
    selected_by_stage: Mapping[str, P6BCandidateRow]
    selected: P6BCandidateRow


_STAGES = (
    "assignment",
    "reactivation",
    "class_compatibility",
    "consolidation",
    "birth_gate",
    "joint_neighbors",
)


def validate_staged_sweep_evidence(
    protocol: P6BProtocolConfig,
    *,
    baseline: P6BCandidateRow,
    candidate_rows: Sequence[P6BCandidateRow],
    finalist_rows: Sequence[P6BCandidateRow],
    selected_by_stage: Mapping[str, P6BCandidateRow],
    selected: P6BCandidateRow,
    ranking_key: Sequence[float | str],
) -> None:
    if not isinstance(protocol, P6BProtocolConfig):
        raise P6BSweepError("protocol must be a P6BProtocolConfig")
    if not baseline.official_metrics_complete:
        raise P6BSweepError("baseline must contain official task metrics")
    if tuple(selected_by_stage) != _STAGES:
        raise P6BSweepError("selected_by_stage must contain the exact stage order")
    if not candidate_rows or not finalist_rows:
        raise P6BSweepError("sweep evidence must contain candidates and finalists")

    incumbent = protocol.base
    for stage in _STAGES:
        expected_configs = (
            joint_neighbor_configs(incumbent, protocol.search)
            if stage == "joint_neighbors"
            else expand_stage_configs(incumbent, protocol.search, stage=stage)
        )
        candidates = tuple(row for row in candidate_rows if row.stage == stage)
        if tuple(row.config for row in candidates) != expected_configs:
            raise P6BSweepError(f"{stage} candidate grid differs from preregistration")
        if any(row.official_metrics_complete for row in candidates):
            raise P6BSweepError("coarse candidate rows cannot contain official metrics")
        reassessed = tuple(
            assess_candidate(
                row,
                baseline=baseline,
                eligibility=protocol.eligibility,
                require_official_metrics=False,
            )
            for row in candidates
        )
        if candidates != reassessed:
            raise P6BSweepError(f"{stage} candidate eligibility was not recomputed")
        frontier = pareto_finalists(
            candidates,
            baseline=baseline,
            eligibility=protocol.eligibility,
        )
        finalists = tuple(row for row in finalist_rows if row.stage == stage)
        if tuple(row.config for row in finalists) != tuple(
            row.config for row in frontier
        ):
            raise P6BSweepError(f"{stage} finalists differ from the Pareto frontier")
        selection = select_final_candidate(
            finalists,
            baseline=baseline,
            eligibility=protocol.eligibility,
        )
        if finalists != selection.assessed_rows:
            raise P6BSweepError(f"{stage} finalist eligibility was not recomputed")
        if selected_by_stage[stage] != selection.selected:
            raise P6BSweepError(f"{stage} selected winner differs from ranking")
        incumbent = selection.selected.config

    if any(row.stage not in _STAGES for row in (*candidate_rows, *finalist_rows)):
        raise P6BSweepError("sweep evidence contains an unknown stage")
    final = selected_by_stage[_STAGES[-1]]
    if selected != final:
        raise P6BSweepError("final selected candidate differs from joint winner")
    if tuple(ranking_key) != candidate_ranking_key(final):
        raise P6BSweepError("final ranking key differs from recomputed ranking")


def run_staged_sweep(
    protocol: P6BProtocolConfig,
    *,
    baseline: P6BCandidateRow,
    fast_evaluator: Callable[[P6BMemoryConfig, str], P6BCandidateRow],
    official_evaluator: Callable[[P6BCandidateRow], P6BCandidateRow],
) -> P6BStagedSweepResult:
    if not isinstance(protocol, P6BProtocolConfig):
        raise P6BSweepError("protocol must be a P6BProtocolConfig")
    if not baseline.official_metrics_complete:
        raise P6BSweepError("baseline must contain official task metrics")
    incumbent = protocol.base
    candidates: list[P6BCandidateRow] = []
    finalists_with_metrics: list[P6BCandidateRow] = []
    selected_by_stage: dict[str, P6BCandidateRow] = {}
    for stage in _STAGES:
        configs = (
            joint_neighbor_configs(incumbent, protocol.search)
            if stage == "joint_neighbors"
            else expand_stage_configs(incumbent, protocol.search, stage=stage)
        )
        stage_rows = []
        for config in configs:
            row = fast_evaluator(config, stage)
            if row.config != config or row.stage != stage:
                raise P6BSweepError("fast evaluator changed candidate identity")
            if row.official_metrics_complete:
                raise P6BSweepError("fast evaluator must not inject official metrics")
            stage_rows.append(row)
        assessed_stage_rows = tuple(
            assess_candidate(
                row,
                baseline=baseline,
                eligibility=protocol.eligibility,
                require_official_metrics=False,
            )
            for row in stage_rows
        )
        candidates.extend(assessed_stage_rows)
        frontier = pareto_finalists(
            stage_rows,
            baseline=baseline,
            eligibility=protocol.eligibility,
        )
        if not frontier:
            raise P6BSweepError(f"{stage} produced no eligible Pareto finalist")
        official_rows = []
        for finalist in frontier:
            official = official_evaluator(finalist)
            if official.config != finalist.config or official.stage != finalist.stage:
                raise P6BSweepError("official evaluator changed candidate identity")
            if not official.official_metrics_complete:
                raise P6BSweepError("official evaluator omitted task metrics")
            official_rows.append(official)
        selection = select_final_candidate(
            official_rows,
            baseline=baseline,
            eligibility=protocol.eligibility,
        )
        finalists_with_metrics.extend(selection.assessed_rows)
        incumbent = selection.selected.config
        selected_by_stage[stage] = selection.selected
    result = P6BStagedSweepResult(
        candidate_rows=tuple(candidates),
        finalist_rows=tuple(finalists_with_metrics),
        selected_by_stage=MappingProxyType(dict(selected_by_stage)),
        selected=selected_by_stage[_STAGES[-1]],
    )
    validate_staged_sweep_evidence(
        protocol,
        baseline=baseline,
        candidate_rows=result.candidate_rows,
        finalist_rows=result.finalist_rows,
        selected_by_stage=result.selected_by_stage,
        selected=result.selected,
        ranking_key=candidate_ranking_key(result.selected),
    )
    return result
