"""Pure P6-A identity, error, resource, and paired-statistics analysis.

The module deliberately has no dependency on a tracker, tensors, or a GPU.
Tracker adapters can materialize the typed records below and all scientific
aggregates can then be reconstructed from those records alone.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from random import Random
from statistics import mean, pstdev

FAILURE_CODES = ("F1", "F2", "F3", "F4", "F5", "F6", "F7")
FAILURE_LABELS = {
    "F1": "local_perception_miss",
    "F2": "association_miss",
    "F3": "identity_fragmentation",
    "F4": "identity_merge",
    "F5": "wrong_reactivation",
    "F6": "semantic_drift",
    "F7": "capacity_birth_failure",
}

_EVENT_KINDS = {"prediction", "gt_miss"}
_RESULT_ALIASES = {
    "match_correct": "active_correct",
    "correct_active_match": "active_correct",
    "match_wrong": "active_wrong",
    "wrong_active_match": "active_wrong",
    "gt_only_miss": "no_attempt",
    "miss": "no_attempt",
    "reactivation_correct": "reactivation_correct",
    "reactivation_wrong": "reactivation_wrong",
    "false_birth": "false_birth",
    "birth_rejected": "birth_rejected",
}
_SENTINEL_STRINGS = {"", "-1", "none", "null", "nan", "na"}


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")  # noqa: TRY004
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _optional_nonnegative_integer(
    value: object, *, name: str
) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, name=name)


def _optional_bool(value: object, *, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool or None")  # noqa: TRY004
    return value


def _identifier(value: object, *, name: str) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be bool")  # noqa: TRY004
    if isinstance(value, Integral):
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} uses a sentinel ID; use null instead")
        return result
    if isinstance(value, str):
        if value.casefold() in _SENTINEL_STRINGS:
            raise ValueError(f"{name} uses a sentinel ID; use null instead")
        return value
    raise ValueError(f"{name} must be a string, integer, or None")


def _text(value: object, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalise_event_kind(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("event_kind must be a string")  # noqa: TRY004
    normalized = value.casefold().replace("-", "_")
    if normalized in {"gt_only_miss", "gt_miss", "miss"}:
        normalized = "gt_miss"
    if normalized not in _EVENT_KINDS:
        raise ValueError("event_kind must be prediction or gt_miss")
    return normalized


def _normalise_result(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("association_result must be a non-empty string")
    normalized = value.casefold().replace("-", "_")
    return _RESULT_ALIASES.get(normalized, normalized)


def _normalise_row_type(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("row_type must be a string")  # noqa: TRY004
    normalized = value.casefold().replace("-", "_")
    return {
        "setup": "bootstrap",
        "bootstrap": "bootstrap",
        "per_new_visit": "new_visit",
        "new_visit": "new_visit",
        "update": "new_visit",
        "full_history": "full_history",
    }.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class AssociationEvent:
    """One prediction decision or one GT-only miss decision.

    ``None`` means that a field is not applicable. In particular, no
    negative/sentinel identity IDs are accepted. Optional evidence flags let
    an adapter state the exact diagnostic decision without coupling this
    module to tracker internals; missing flags are inferred where possible.
    """

    scene_id: str
    sequence_id: str
    stage_id: int
    event_kind: str = "prediction"
    reference_scene_id: str | None = None
    master_sequence_id: str | None = None
    prefix: int | None = None
    method: str | None = None
    query_id: str | int | None = None
    candidate_slot_id: str | int | None = None
    predicted_identity_id: str | int | None = None
    gt_entity_id: str | int | None = None
    association_correct: bool | None = None
    feature_similarity: float | None = None
    class_similarity: float | None = None
    total_score: float | None = None
    best_score: float | None = None
    second_best_score: float | None = None
    score_margin: float | None = None
    observation_confidence: float | None = None
    mask_support: float | None = None
    predicted_class: str | int | None = None
    class_entropy: float | None = None
    slot_age: int | None = None
    last_seen_stage: int | None = None
    gap_length: int | None = None
    slot_active: bool | None = None
    slot_occupied: bool | None = None
    association_result: str | None = None
    gt_present: bool | None = None
    prediction_present: bool | None = None
    transition_opportunity: bool | None = None
    id_switch: bool | None = None
    gap_opportunity: bool | None = None
    reactivation_attempt: bool | None = None
    reactivation_correct: bool | None = None
    new_birth: bool | None = None
    false_birth: bool | None = None
    reactivation: bool | None = None
    wrong_reactivation: bool | None = None
    local_observation_available: bool | None = None
    local_match_available: bool | None = None
    raw_local_match: bool | None = None
    raw_prediction_available: bool | None = None
    local_perception_miss: bool | None = None
    association_miss: bool | None = None
    association_attempted: bool | None = None
    identity_fragmentation: bool | None = None
    identity_merge: bool | None = None
    fragmentation: bool | None = None
    merge: bool | None = None
    semantic_drift: bool | None = None
    semantic_mismatch: bool | None = None
    capacity_failure: bool | None = None
    capacity_birth_failure: bool | None = None
    birth_rejected: bool | None = None
    is_failure: bool | None = None
    failure_category: str | None = None
    failure_code: str | None = None
    prediction_digest: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> AssociationEvent:
        aliases = {
            "row_type": "event_kind",
            "slot_id": "candidate_slot_id",
            "predicted_id": "predicted_identity_id",
            "gt_id": "gt_entity_id",
            "reactivation_wrong": "wrong_reactivation",
            "fragmentation": "identity_fragmentation",
            "merge": "identity_merge",
            "semantic_mismatch": "semantic_drift",
            "capacity_birth_failure": "capacity_failure",
        }
        values = dict(row)
        for source, target in aliases.items():
            if target not in values and source in values:
                values[target] = values[source]
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        _text(self.scene_id, name="scene_id")
        _text(self.sequence_id, name="sequence_id")
        _nonnegative_integer(self.stage_id, name="stage_id")
        event_kind = _normalise_event_kind(self.event_kind)
        _text(self.reference_scene_id, name="reference_scene_id", optional=True)
        _text(self.master_sequence_id, name="master_sequence_id", optional=True)
        _optional_nonnegative_integer(self.prefix, name="prefix")
        _text(self.method, name="method", optional=True)
        _identifier(self.query_id, name="query_id")
        _identifier(self.candidate_slot_id, name="candidate_slot_id")
        _identifier(self.predicted_identity_id, name="predicted_identity_id")
        _identifier(self.gt_entity_id, name="gt_entity_id")
        _identifier(self.predicted_class, name="predicted_class")

        for name in (
            "association_correct",
            "gt_present",
            "prediction_present",
            "transition_opportunity",
            "id_switch",
            "gap_opportunity",
            "reactivation_attempt",
            "reactivation_correct",
            "new_birth",
            "false_birth",
            "reactivation",
            "wrong_reactivation",
            "local_observation_available",
            "local_match_available",
            "raw_local_match",
            "raw_prediction_available",
            "local_perception_miss",
            "association_miss",
            "association_attempted",
            "identity_fragmentation",
            "identity_merge",
            "fragmentation",
            "merge",
            "semantic_drift",
            "semantic_mismatch",
            "capacity_failure",
            "capacity_birth_failure",
            "birth_rejected",
            "is_failure",
            "slot_active",
            "slot_occupied",
        ):
            _optional_bool(getattr(self, name), name=name)
        for name in (
            "feature_similarity",
            "class_similarity",
            "total_score",
            "best_score",
            "second_best_score",
            "score_margin",
            "observation_confidence",
            "mask_support",
            "class_entropy",
        ):
            _optional_finite(getattr(self, name), name=name)
        for name in ("slot_age", "last_seen_stage", "gap_length"):
            _optional_nonnegative_integer(getattr(self, name), name=name)
        result = _normalise_result(self.association_result)
        for name in ("failure_category", "failure_code"):
            value = getattr(self, name)
            if value is not None and value not in {*FAILURE_CODES, "unclassified"}:
                raise ValueError(f"{name} is not a P6-A category")
        if (
            self.failure_category is not None
            and self.failure_code is not None
            and self.failure_category != self.failure_code
        ):
            raise ValueError("failure_category and failure_code disagree")
        if self.prediction_digest is not None:
            _text(self.prediction_digest, name="prediction_digest")

        if event_kind == "gt_miss":
            if self.prediction_present is True:
                raise ValueError("gt_miss cannot have prediction_present=True")
            if self.query_id is not None or self.candidate_slot_id is not None:
                raise ValueError("gt_miss cannot contain prediction IDs")
            if self.gt_entity_id is None:
                raise ValueError("gt_miss requires gt_entity_id")
            if result not in {None, "no_attempt"}:
                raise ValueError("gt_miss must use the no_attempt result")
        if self.gt_present is True and self.gt_entity_id is None:
            raise ValueError("gt_present=True requires gt_entity_id")
        if self.prediction_present is False and self.predicted_identity_id is not None:
            raise ValueError("prediction_present=False cannot contain predicted identity")
        if self.id_switch is True and self.transition_opportunity is False:
            raise ValueError("id_switch requires a transition opportunity")
        if self.reactivation_correct is not None and not (
            self.reactivation_attempt is True or self.reactivation is True
        ):
            raise ValueError("reactivation_correct requires a reactivation attempt")
        if result == "reactivation_correct" and self.reactivation_correct is False:
            raise ValueError("reactivation result contradicts reactivation_correct")
        if result == "reactivation_wrong" and self.reactivation_correct is True:
            raise ValueError("reactivation result contradicts reactivation_correct")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_event(row: AssociationEvent | Mapping[str, object]) -> AssociationEvent:
    if isinstance(row, AssociationEvent):
        event = row
    elif isinstance(row, Mapping):
        event = AssociationEvent.from_mapping(row)
    else:
        raise ValueError(  # noqa: TRY004
            "association events must be AssociationEvent or mapping"
        )
    return replace(
        event,
        event_kind=_normalise_event_kind(event.event_kind),
        association_result=_normalise_result(event.association_result),
    )


def validate_association_events(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> tuple[AssociationEvent, ...]:
    """Validate a complete typed event table and reject duplicate decisions."""

    result: list[AssociationEvent] = []
    keys: set[tuple[object, ...]] = set()
    for index, raw in enumerate(events):
        event = _coerce_event(raw)
        event.validate()
        key = (
            event.scene_id,
            event.sequence_id,
            event.stage_id,
            event.event_kind,
            event.query_id,
            event.candidate_slot_id,
            event.gt_entity_id,
        )
        if key in keys:
            raise ValueError(f"duplicate association event at index {index}")
        keys.add(key)
        result.append(event)
    return tuple(result)


def _event_result(event: AssociationEvent) -> str | None:
    return _normalise_result(event.association_result)


def _events_have_explicit(events: Sequence[AssociationEvent], field: str) -> bool:
    return any(getattr(event, field) is not None for event in events)


def _group_by_gt(events: Sequence[AssociationEvent]) -> dict[tuple[str, str, object], list[AssociationEvent]]:
    groups: dict[tuple[str, str, object], list[AssociationEvent]] = {}
    for event in events:
        if event.gt_entity_id is None or event.gt_present is False:
            continue
        key = (event.scene_id, event.sequence_id, event.gt_entity_id)
        groups.setdefault(key, []).append(event)
    for group in groups.values():
        group.sort(key=lambda event: (event.stage_id, repr(event.query_id)))
    return groups


def _inferred_transition_counts(
    events: Sequence[AssociationEvent],
) -> tuple[int, int]:
    opportunities = 0
    switches = 0
    for group in _group_by_gt(events).values():
        observations = [
            event
            for event in group
            if event.predicted_identity_id is not None
            and event.prediction_present is not False
        ]
        for previous, current in pairwise(observations):
            if current.stage_id != previous.stage_id + 1:
                continue
            opportunities += 1
            switches += int(
                previous.predicted_identity_id != current.predicted_identity_id
            )
    return opportunities, switches


def _inferred_gap_opportunities(events: Sequence[AssociationEvent]) -> int:
    count = 0
    for group in _group_by_gt(events).values():
        observed_stages = sorted(
            {
                event.stage_id
                for event in group
                if event.predicted_identity_id is not None
                and event.prediction_present is not False
            }
        )
        count += sum(
            int(current - previous > 1)
            for previous, current in pairwise(observed_stages)
        )
    return count


def aggregate_identity_metrics(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct normalized identity metrics from the event table."""

    validated = validate_association_events(events)
    explicit_opportunities = sum(
        event.transition_opportunity is True for event in validated
    )
    explicit_switches = sum(event.id_switch is True for event in validated)
    if _events_have_explicit(validated, "transition_opportunity"):
        transition_opportunities = explicit_opportunities
    else:
        transition_opportunities, _ = _inferred_transition_counts(validated)
    if _events_have_explicit(validated, "id_switch"):
        id_switches = explicit_switches
    else:
        _, id_switches = _inferred_transition_counts(validated)

    active_correct = sum(
        _event_result(event) in {"active_correct", "correct"}
        and event.reactivation is not True
        for event in validated
    )
    active_wrong = sum(
        _event_result(event) in {"active_wrong", "wrong"}
        and event.reactivation is not True
        for event in validated
    )
    births = sum(
        event.new_birth is True or _event_result(event) in {"birth", "false_birth"}
        for event in validated
    )
    false_births = sum(
        event.false_birth is True
        or event.birth_rejected is True
        or _event_result(event) in {"false_birth", "birth_rejected"}
        for event in validated
    )
    fragmentation_count = sum(
        event.identity_fragmentation is True
        or _event_result(event) in {"fragmentation", "identity_fragmentation"}
        for event in validated
    )
    merge_count = sum(
        event.identity_merge is True
        or _event_result(event) in {"merge", "identity_merge"}
        for event in validated
    )
    return {
        "id_switches": int(id_switches),
        "transition_opportunities": int(transition_opportunities),
        "id_switch_rate": (
            id_switches / transition_opportunities
            if transition_opportunities
            else None
        ),
        "active_correct_matches": int(active_correct),
        "active_wrong_matches": int(active_wrong),
        "births": int(births),
        "false_births": int(false_births),
        "fragmentation_count": int(fragmentation_count),
        "merge_count": int(merge_count),
    }


def aggregate_reactivation_metrics(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct gap, attempt, correctness, and coverage metrics."""

    validated = validate_association_events(events)
    explicit_gap = _events_have_explicit(validated, "gap_opportunity")
    gap_opportunities = (
        sum(event.gap_opportunity is True for event in validated)
        if explicit_gap
        else _inferred_gap_opportunities(validated)
    )
    attempts = 0
    correct = 0
    wrong = 0
    for event in validated:
        result = _event_result(event)
        attempt = (
            event.reactivation_attempt
            if event.reactivation_attempt is not None
            else event.reactivation is True
            or result in {"reactivation_correct", "reactivation_wrong"}
        )
        if attempt:
            attempts += 1
        if event.reactivation_correct is True or result == "reactivation_correct":
            correct += 1
        if event.wrong_reactivation is True or (
            attempt
            and (
                event.reactivation_correct is False
                or result == "reactivation_wrong"
            )
        ):
            wrong += 1
    if correct + wrong > attempts:
        raise ValueError("reactivation correct/wrong events exceed attempts")
    if attempts > gap_opportunities:
        raise ValueError("reactivation attempts exceed gap opportunities")
    no_attempts = max(0, gap_opportunities - attempts)
    accuracy = correct / attempts if attempts else None
    precision = correct / attempts if attempts else None
    recall = correct / gap_opportunities if gap_opportunities else None
    coverage = attempts / gap_opportunities if gap_opportunities else None
    return {
        "gap_opportunities": int(gap_opportunities),
        "reactivation_attempts": int(attempts),
        "correct_reactivations": int(correct),
        "wrong_reactivations": int(wrong),
        "no_attempts": int(no_attempts),
        "reactivation_accuracy": accuracy,
        "reactivation_precision": precision,
        "reactivation_recall": recall,
        "reactivation_coverage": coverage,
    }


def aggregate_event_metrics(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the two aggregate blocks that the event table reconstructs."""

    validated = validate_association_events(events)
    return aggregate_identity_metrics(validated), aggregate_reactivation_metrics(validated)


def aggregate_metrics_by_sequence(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return event-derived identity/reactivation rows per scene/sequence."""

    validated = validate_association_events(events)
    groups: dict[tuple[str, str], list[AssociationEvent]] = {}
    for event in validated:
        groups.setdefault((event.scene_id, event.sequence_id), []).append(event)
    result: list[dict[str, object]] = []
    for (scene_id, sequence_id), group in sorted(groups.items()):
        identity, reactivation = aggregate_event_metrics(group)
        result.append(
            {
                "scene_id": scene_id,
                "sequence_id": sequence_id,
                **identity,
                **reactivation,
            }
        )
    return tuple(result)


def reconstruct_event_metrics(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> dict[str, object]:
    identity, reactivation = aggregate_event_metrics(events)
    return {**identity, **reactivation}


def _evidence_value(evidence: AssociationEvent | Mapping[str, object], *names: str) -> object:
    if isinstance(evidence, AssociationEvent):
        for name in names:
            value = getattr(evidence, name, None)
            if value is not None:
                return value
        return None
    for name in names:
        if name in evidence and evidence[name] is not None:
            return evidence[name]
    return None


def _truth(evidence: AssociationEvent | Mapping[str, object], *names: str) -> bool:
    return _evidence_value(evidence, *names) is True


def classify_failure(
    evidence: AssociationEvent | Mapping[str, object],
) -> str:
    """Assign exactly one deterministic primary failure code."""

    explicit = _evidence_value(evidence, "failure_category", "failure_code")
    if explicit in {*FAILURE_CODES, "unclassified"}:
        return str(explicit)
    result = _evidence_value(evidence, "association_result")
    result_text = result.casefold().replace("-", "_") if isinstance(result, str) else ""
    if (
        _evidence_value(
            evidence,
            "local_observation_available",
            "local_match_available",
            "raw_local_match",
            "raw_prediction_available",
        )
        is False
        or _truth(evidence, "local_perception_miss")
        or result_text in {"local_miss", "perception_miss", "local_perception_miss"}
    ):
        return "F1"
    if _truth(evidence, "association_miss") or result_text in {
        "association_miss",
        "association_wrong",
    }:
        return "F2"
    if _truth(evidence, "identity_fragmentation", "fragmentation") or result_text in {
        "fragmentation",
        "identity_fragmentation",
    }:
        return "F3"
    if _truth(evidence, "identity_merge", "merge") or result_text in {"merge", "identity_merge"}:
        return "F4"
    if (
        _truth(evidence, "wrong_reactivation")
        or (
            _truth(evidence, "reactivation")
            and _evidence_value(evidence, "reactivation_correct") is False
        )
        or result_text == "reactivation_wrong"
    ):
        return "F5"
    if _truth(evidence, "semantic_drift", "semantic_mismatch") or result_text in {
        "semantic_drift",
        "semantic_mismatch",
    }:
        return "F6"
    if (
        _truth(evidence, "capacity_failure", "capacity_birth_failure", "birth_rejected")
        or result_text
        in {"capacity_failure", "birth_rejected", "capacity_birth_failure", "false_birth"}
    ):
        return "F7"
    return "unclassified"


def classify_failures(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> tuple[str, ...]:
    validated = validate_association_events(events)
    return tuple(classify_failure(event) for event in validated)


def failure_breakdown(
    events: Iterable[AssociationEvent | Mapping[str, object]],
) -> dict[str, object]:
    validated = validate_association_events(events)
    failure_events = [
        event
        for event in validated
        if event.is_failure is True
        or event.association_correct is False
        or event.event_kind == "gt_miss"
    ]
    counts = {code: 0 for code in FAILURE_CODES}
    counts["unclassified"] = 0
    for event in failure_events:
        counts[classify_failure(event)] += 1
    total = len(failure_events)
    categorized = sum(counts[code] for code in FAILURE_CODES)
    return {
        "counts": counts,
        "total_failures": total,
        "categorized_failures": categorized,
        "explainability_share": categorized / total if total else None,
    }


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """Per-stage observability for bounded persistent state."""

    method: str
    horizon: int
    stage_id: int
    capacity: int
    birth_count: int = 0
    occupied_count: int = 0
    active_count: int = 0
    dormant_count: int | None = None
    rejected_births: int = 0
    persistent_state_bytes: int | None = None
    feature_dim: int | None = None
    class_count: int | None = None
    batch_size: int = 1

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> CapacitySnapshot:
        aliases = {
            "occupied_count_per_stage": "occupied_count",
            "active_count_per_stage": "active_count",
            "dormant_count_per_stage": "dormant_count",
            "peak_rejected_births": "rejected_births",
        }
        values = dict(row)
        for source, target in aliases.items():
            if target not in values and source in values:
                values[target] = values[source]
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        _text(self.method, name="method")
        _nonnegative_integer(self.horizon, name="horizon")
        _nonnegative_integer(self.stage_id, name="stage_id")
        capacity = _nonnegative_integer(self.capacity, name="capacity")
        if capacity == 0:
            raise ValueError("capacity must be positive")
        for name in (
            "birth_count",
            "occupied_count",
            "active_count",
            "rejected_births",
        ):
            _nonnegative_integer(getattr(self, name), name=name)
        _optional_nonnegative_integer(self.dormant_count, name="dormant_count")
        _optional_nonnegative_integer(
            self.persistent_state_bytes, name="persistent_state_bytes"
        )
        _optional_nonnegative_integer(self.feature_dim, name="feature_dim")
        _optional_nonnegative_integer(self.class_count, name="class_count")
        _nonnegative_integer(self.batch_size, name="batch_size")
        if self.batch_size == 0:
            raise ValueError("batch_size must be positive")
        if self.occupied_count > capacity:
            raise ValueError("occupied_count exceeds capacity")
        if self.active_count > self.occupied_count:
            raise ValueError("active_count must not exceed occupied_count")
        expected_dormant = self.occupied_count - self.active_count
        if self.dormant_count is not None and self.dormant_count != expected_dormant:
            raise ValueError("dormant_count must equal occupied_count-active_count")
        if (
            self.persistent_state_bytes is not None
            and self.feature_dim is not None
            and self.class_count is not None
            and self.persistent_state_bytes
            != persistent_state_bytes(
                capacity,
                self.feature_dim,
                self.class_count,
                batch_size=self.batch_size,
            )
        ):
            raise ValueError("persistent_state_bytes does not match capacity formula")


def persistent_state_bytes(
    capacity: int,
    feature_dim: int,
    class_count: int,
    *,
    batch_size: int = 1,
    dtype_bytes: int = 4,
    index_bytes: int = 8,
    bool_bytes: int = 1,
) -> int:
    """Return serialized bytes for the persistent state fields only.

    The fields match ``PersistentMemoryState``: embedding, class probability,
    confidence, two bool masks, two integer lifecycle fields, and one batch
    stage watermark. Offline tracks, masks, and evaluator bookkeeping are not
    included.
    """

    for name, value in (
        ("capacity", capacity),
        ("feature_dim", feature_dim),
        ("class_count", class_count),
        ("batch_size", batch_size),
        ("dtype_bytes", dtype_bytes),
        ("index_bytes", index_bytes),
        ("bool_bytes", bool_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    per_slot = (
        (feature_dim + class_count + 1) * dtype_bytes
        + 2 * bool_bytes
        + 2 * index_bytes
    )
    return int(batch_size * (capacity * per_slot + index_bytes))


expected_persistent_state_bytes = persistent_state_bytes


def audit_capacity(
    snapshots: Iterable[CapacitySnapshot | Mapping[str, object]],
    *,
    capacity: int | None = None,
) -> dict[str, object]:
    """Validate and aggregate capacity/state observations."""

    validated: list[CapacitySnapshot] = []
    for raw in snapshots:
        snapshot = (
            raw
            if isinstance(raw, CapacitySnapshot)
            else CapacitySnapshot.from_mapping(raw)
        )
        snapshot.validate()
        validated.append(snapshot)
    if not validated:
        raise ValueError("capacity audit requires at least one snapshot")
    capacities = {snapshot.capacity for snapshot in validated}
    if capacity is not None:
        _nonnegative_integer(capacity, name="capacity")
        capacities.add(capacity)
    if len(capacities) != 1:
        raise ValueError("capacity must be constant across snapshots")
    fixed_capacity = capacities.pop()
    occupied = [snapshot.occupied_count for snapshot in validated]
    active = [snapshot.active_count for snapshot in validated]
    dormant = [
        snapshot.dormant_count
        if snapshot.dormant_count is not None
        else snapshot.occupied_count - snapshot.active_count
        for snapshot in validated
    ]
    state_sizes = [
        snapshot.persistent_state_bytes
        for snapshot in validated
        if snapshot.persistent_state_bytes is not None
    ]
    return {
        "capacity": fixed_capacity,
        "birth_count": sum(snapshot.birth_count for snapshot in validated),
        "occupied_count_per_stage": {
            str(snapshot.stage_id): snapshot.occupied_count for snapshot in validated
        },
        "active_count_per_stage": {
            str(snapshot.stage_id): snapshot.active_count for snapshot in validated
        },
        "dormant_count_per_stage": {
            str(snapshot.stage_id): count
            for snapshot, count in zip(validated, dormant)
        },
        "peak_occupied": max(occupied),
        "peak_active": max(active),
        "peak_dormant": max(dormant),
        "occupancy_ratio": max(occupied) / fixed_capacity if fixed_capacity else None,
        "rejected_births": sum(snapshot.rejected_births for snapshot in validated),
        "persistent_state_bytes": max(state_sizes) if state_sizes else None,
        "bounded_state": bool(state_sizes) and len(set(state_sizes)) == 1,
    }


@dataclass(frozen=True, slots=True)
class EfficiencyRecord:
    """One explicitly typed setup or per-new-visit efficiency row."""

    method: str
    horizon: int
    stage_id: int
    row_type: str
    bootstrap_latency_ms: float | None = None
    new_visit_latency_ms: float | None = None
    association_overhead_ms: float | None = None
    memory_update_overhead_ms: float | None = None
    full_history_latency_ms: float | None = None
    gpu_peak_memory_bytes: int | None = None
    persistent_state_bytes: int | None = None
    reference_scene_id: str | None = None
    sequence_id: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> EfficiencyRecord:
        values = dict(row)
        aliases = {
            "type": "row_type",
            "latency_ms": "new_visit_latency_ms",
            "peak_working_memory_bytes": "gpu_peak_memory_bytes",
        }
        for source, target in aliases.items():
            if target not in values and source in values:
                values[target] = values[source]
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def validate(self) -> None:
        _text(self.method, name="method")
        _nonnegative_integer(self.horizon, name="horizon")
        _nonnegative_integer(self.stage_id, name="stage_id")
        row_type = _normalise_row_type(self.row_type)
        if row_type not in {"bootstrap", "new_visit", "full_history"}:
            raise ValueError("row_type must be bootstrap, new_visit, or full_history")
        for name in (
            "bootstrap_latency_ms",
            "new_visit_latency_ms",
            "association_overhead_ms",
            "memory_update_overhead_ms",
            "full_history_latency_ms",
        ):
            value = _optional_finite(getattr(self, name), name=name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("gpu_peak_memory_bytes", "persistent_state_bytes"):
            _optional_nonnegative_integer(getattr(self, name), name=name)
        _text(self.reference_scene_id, name="reference_scene_id", optional=True)
        _text(self.sequence_id, name="sequence_id", optional=True)
        if row_type == "bootstrap":
            if self.bootstrap_latency_ms is None:
                raise ValueError("bootstrap rows require bootstrap_latency_ms")
            if any(
                value is not None
                for value in (
                    self.new_visit_latency_ms,
                    self.association_overhead_ms,
                    self.memory_update_overhead_ms,
                    self.full_history_latency_ms,
                )
            ):
                raise ValueError("bootstrap rows cannot contain new_visit metrics")
        elif row_type == "new_visit":
            if self.new_visit_latency_ms is None:
                raise ValueError("new_visit rows require new_visit_latency_ms")
            if self.bootstrap_latency_ms is not None:
                raise ValueError("new_visit rows cannot contain bootstrap metrics")
        else:
            if self.full_history_latency_ms is None:
                raise ValueError("full_history rows require full_history_latency_ms")
            if any(
                value is not None
                for value in (
                    self.bootstrap_latency_ms,
                    self.new_visit_latency_ms,
                    self.association_overhead_ms,
                    self.memory_update_overhead_ms,
                )
            ):
                raise ValueError("full_history rows cannot contain setup/update metrics")


def _coerce_efficiency(row: EfficiencyRecord | Mapping[str, object]) -> EfficiencyRecord:
    if isinstance(row, EfficiencyRecord):
        record = row
    elif isinstance(row, Mapping):
        record = EfficiencyRecord.from_mapping(row)
    else:
        raise ValueError(  # noqa: TRY004
            "efficiency rows must be EfficiencyRecord or mapping"
        )
    return replace(record, row_type=_normalise_row_type(record.row_type))


def validate_efficiency_rows(
    rows: Iterable[EfficiencyRecord | Mapping[str, object]],
) -> tuple[EfficiencyRecord, ...]:
    result: list[EfficiencyRecord] = []
    for row in rows:
        record = _coerce_efficiency(row)
        record.validate()
        result.append(record)
    return tuple(result)


def _mean_optional_field(
    rows: Sequence[EfficiencyRecord], name: str
) -> float | None:
    values = [getattr(row, name) for row in rows if getattr(row, name) is not None]
    return mean(values) if values else None


def aggregate_efficiency(
    rows: Iterable[EfficiencyRecord | Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    validated = validate_efficiency_rows(rows)
    groups: dict[tuple[str, int, str], list[EfficiencyRecord]] = {}
    for row in validated:
        row_type = _normalise_row_type(row.row_type)
        groups.setdefault((row.method, row.horizon, row_type), []).append(row)
    result: list[dict[str, object]] = []
    for (method, horizon, row_type), group in sorted(groups.items()):
        memory = [row.gpu_peak_memory_bytes for row in group if row.gpu_peak_memory_bytes is not None]
        state = [row.persistent_state_bytes for row in group if row.persistent_state_bytes is not None]
        result.append(
            {
                "method": method,
                "horizon": horizon,
                "row_type": row_type,
                "count": len(group),
                "bootstrap_latency_ms": _mean_optional_field(group, "bootstrap_latency_ms"),
                "new_visit_latency_ms": _mean_optional_field(group, "new_visit_latency_ms"),
                "association_overhead_ms": _mean_optional_field(group, "association_overhead_ms"),
                "memory_update_overhead_ms": _mean_optional_field(group, "memory_update_overhead_ms"),
                "full_history_latency_ms": _mean_optional_field(group, "full_history_latency_ms"),
                "gpu_peak_memory_bytes": max(memory) if memory else None,
                "persistent_state_bytes": max(state) if state else None,
            }
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PairedMetricRecord:
    """One method/metric value in a common-prefix paired comparison."""

    reference_scene_id: str
    master_sequence_id: str
    prefix: int
    method: str
    metric: str
    value: float
    prediction_digest: str | None = None
    cache_digest: str | None = None
    order_id: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> PairedMetricRecord:
        values = dict(row)
        aliases = {
            "sequence_id": "master_sequence_id",
            "cache_fingerprint": "prediction_digest",
            "frozen_prediction_digest": "prediction_digest",
        }
        for source, target in aliases.items():
            if target not in values and source in values:
                values[target] = values[source]
        known = {field.name for field in fields(cls)}
        try:
            return cls(**{key: value for key, value in values.items() if key in known})
        except TypeError as error:
            raise ValueError("paired metric record is missing required fields") from error

    @property
    def effective_digest(self) -> str:
        if (
            self.prediction_digest is not None
            and self.cache_digest is not None
            and self.prediction_digest != self.cache_digest
        ):
            raise ValueError("prediction_digest and cache_digest disagree")
        digest = self.prediction_digest or self.cache_digest
        if not isinstance(digest, str) or not digest:
            raise ValueError("paired records require a prediction/cache digest")
        return digest

    def validate(self) -> None:
        _text(self.reference_scene_id, name="reference_scene_id")
        _text(self.master_sequence_id, name="master_sequence_id")
        _nonnegative_integer(self.prefix, name="prefix")
        _text(self.method, name="method")
        _text(self.metric, name="metric")
        _finite(self.value, name="value")
        _text(self.effective_digest, name="prediction_digest")
        _text(self.order_id, name="order_id", optional=True)


def _coerce_paired_rows(
    rows: Iterable[PairedMetricRecord | Mapping[str, object]],
    *,
    method: str,
    baseline_method: str,
    metric: str | None,
) -> tuple[PairedMetricRecord, ...]:
    result: list[PairedMetricRecord] = []
    for raw in rows:
        if isinstance(raw, PairedMetricRecord):
            record = raw
        elif isinstance(raw, Mapping):
            if "persisted_value" in raw or "baseline_value" in raw:
                if "persisted_value" not in raw or "baseline_value" not in raw:
                    raise ValueError("paired row must contain both method values")
                common = dict(raw)
                persisted = common.pop("persisted_value")
                baseline = common.pop("baseline_value")
                common.pop("method", None)
                common["metric"] = common.get("metric", metric or "metric")
                result.extend(
                    (
                        PairedMetricRecord(
                            method=method,
                            value=persisted,
                            **{
                                key: value
                                for key, value in common.items()
                                if key in {field.name for field in fields(PairedMetricRecord)}
                                and key not in {"value", "method"}
                            },
                        ),
                        PairedMetricRecord(
                            method=baseline_method,
                            value=baseline,
                            **{
                                key: value
                                for key, value in common.items()
                                if key in {field.name for field in fields(PairedMetricRecord)}
                                and key not in {"value", "method"}
                            },
                        ),
                    )
                )
                continue
            record = PairedMetricRecord.from_mapping(raw)
        else:
            raise ValueError(  # noqa: TRY004
                "paired rows must be PairedMetricRecord or mapping"
            )
        record.validate()
        if metric is not None and record.metric != metric:
            continue
        result.append(record)
    if not result:
        raise ValueError("paired bootstrap requires at least one metric row")
    return tuple(result)


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_cluster_bootstrap(
    rows: Iterable[PairedMetricRecord | Mapping[str, object]],
    *,
    method: str = "Persist4D",
    baseline_method: str = "EMA",
    metric: str | None = None,
    n_bootstrap: int = 10_000,
    seed: int = 45,
    confidence: float = 0.95,
    cluster_by: str = "reference_scene_id",
    cluster_key: str | None = None,
) -> dict[str, object]:
    """Run deterministic paired bootstrap over reference-scene clusters.

    A pair is exact only when master sequence, prefix, order, metric, and
    frozen prediction digest agree. Sampling sequence/window IDs as clusters
    is rejected because overlapping windows are not independent units.
    """

    if cluster_key is not None:
        cluster_by = cluster_key
    if cluster_by != "reference_scene_id":
        raise ValueError("paired bootstrap cluster must be reference_scene_id")
    _text(method, name="method")
    _text(baseline_method, name="baseline_method")
    if method == baseline_method:
        raise ValueError("method and baseline_method must differ")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")  # noqa: TRY004
    confidence = _finite(confidence, name="confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    records = _coerce_paired_rows(
        rows,
        method=method,
        baseline_method=baseline_method,
        metric=metric,
    )
    selected = [record for record in records if record.method in {method, baseline_method}]
    if not selected:
        raise ValueError("paired bootstrap has no selected methods")
    groups: dict[tuple[str, str, int, str | None, str], dict[str, list[PairedMetricRecord]]] = {}
    for record in selected:
        key = (
            record.reference_scene_id,
            record.master_sequence_id,
            record.prefix,
            record.order_id,
            record.metric,
        )
        groups.setdefault(key, {}).setdefault(record.method, []).append(record)
    pair_deltas: list[tuple[str, float, float, float]] = []
    for key, by_method in sorted(groups.items(), key=lambda item: repr(item[0])):
        if set(by_method) != {method, baseline_method}:
            raise ValueError("paired records have a missing method pair")
        for name in (method, baseline_method):
            if len(by_method[name]) != 1:
                raise ValueError("paired records contain a duplicate pair")
        left = by_method[method][0]
        right = by_method[baseline_method][0]
        if left.effective_digest != right.effective_digest:
            raise ValueError("paired records use different prediction caches")
        pair_deltas.append(
            (
                key[0],
                float(left.value) - float(right.value),
                float(left.value),
                float(right.value),
            )
        )
    clusters = sorted({item[0] for item in pair_deltas})
    if len(clusters) < 2:
        raise ValueError("paired cluster bootstrap requires at least two reference scenes")
    deltas_by_cluster: dict[str, list[float]] = {cluster: [] for cluster in clusters}
    for cluster, delta, _, _ in pair_deltas:
        deltas_by_cluster[cluster].append(delta)
    cluster_means = {
        cluster: mean(deltas_by_cluster[cluster]) for cluster in clusters
    }
    random = Random(int(seed))
    bootstrap_means: list[float] = []
    for _ in range(int(n_bootstrap)):
        sampled = [random.choice(clusters) for _ in clusters]
        bootstrap_means.append(mean(cluster_means[cluster] for cluster in sampled))
    alpha = (1.0 - confidence) / 2.0
    mean_delta = mean(delta for _, delta, _, _ in pair_deltas)
    baseline_mean = mean(value for _, _, _, value in pair_deltas)
    method_mean = mean(value for _, _, value, _ in pair_deltas)
    relative_reduction = (
        (baseline_mean - method_mean) / baseline_mean if baseline_mean else None
    )
    low = _percentile(bootstrap_means, alpha)
    high = _percentile(bootstrap_means, 1.0 - alpha)
    return {
        "method": method,
        "baseline_method": baseline_method,
        "metric": metric or selected[0].metric,
        "cluster_field": "reference_scene_id",
        "n_clusters": len(clusters),
        "n_pairs": len(pair_deltas),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "confidence": confidence,
        "method_mean": method_mean,
        "baseline_mean": baseline_mean,
        "mean_delta": mean_delta,
        "std_delta": pstdev(cluster_means.values()),
        "std": pstdev(cluster_means.values()),
        "mean": mean_delta,
        "ci_low": low,
        "ci_high": high,
        "delta_ci_low": low,
        "delta_ci_high": high,
        "ci95_low": low,
        "ci95_high": high,
        "relative_reduction": relative_reduction,
        "clusters": clusters,
    }


bootstrap_paired_clusters = paired_cluster_bootstrap


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Preregistered P6-A gate thresholds."""

    g6a1_relative_reduction: float = 0.20
    g6a1_ci_high: float = 0.0
    g6a2_accuracy: float = 0.70
    g6a2_recall: float = 0.25
    g6a3_abs_tolerance: float = 1e-12
    g6a4_t2_drop: float = 0.05
    g6a5_explainability: float = 0.90

    def validate(self) -> None:
        for name in (
            "g6a1_relative_reduction",
            "g6a2_accuracy",
            "g6a2_recall",
            "g6a3_abs_tolerance",
            "g6a4_t2_drop",
            "g6a5_explainability",
        ):
            value = _finite(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "g6a1_relative_reduction",
            "g6a2_accuracy",
            "g6a2_recall",
            "g6a3_abs_tolerance",
            "g6a4_t2_drop",
            "g6a5_explainability",
        ):
            if getattr(self, name) > 1.0:
                raise ValueError(f"{name} must not exceed 1")
        _finite(self.g6a1_ci_high, name="g6a1_ci_high")


def _horizon_value(values: object, horizon: int) -> object:
    if not isinstance(values, Mapping):
        return None
    return values.get(horizon, values.get(str(horizon)))


def _numeric_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return _finite(value, name="gate metric")
    except ValueError:
        return None


def _raw_values(values: object) -> list[float]:
    if isinstance(values, Mapping):
        output: list[float] = []
        for item in values.values():
            output.extend(_raw_values(item))
        return output
    if isinstance(values, (list, tuple)):
        output = []
        for item in values:
            output.extend(_raw_values(item))
        return output
    numeric = _numeric_value(values)
    return [] if numeric is None else [numeric]


def _gate_result(
    passed: bool,
    *,
    checks: Mapping[str, object],
    threshold: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "passed": bool(passed),
        "checks": dict(checks),
    }
    if threshold is not None:
        result["threshold"] = dict(threshold)
    return result


def _gate_g6a1(data: Mapping[str, object], config: GateConfig) -> dict[str, object]:
    paired = data.get("paired_idsw")
    checks: dict[str, object] = {}
    passed = True
    for horizon in (4, 5):
        entry = _horizon_value(paired, horizon)
        reduction = ci_high = None
        if isinstance(entry, Mapping):
            reduction = _numeric_value(entry.get("relative_reduction"))
            ci_high = _numeric_value(entry.get("ci_high", entry.get("delta_ci_high")))
        ok = (
            reduction is not None
            and ci_high is not None
            and reduction >= config.g6a1_relative_reduction
            and ci_high <= config.g6a1_ci_high
        )
        checks[f"T{horizon}"] = {
            "relative_reduction": reduction,
            "ci_high": ci_high,
            "passed": ok,
        }
        passed = passed and ok
    return _gate_result(
        passed,
        checks=checks,
        threshold={
            "relative_reduction": config.g6a1_relative_reduction,
            "ci_high_max": config.g6a1_ci_high,
        },
    )


def _gate_g6a2(
    data: Mapping[str, object], config: GateConfig, *, method: str, baseline: str
) -> dict[str, object]:
    reactivation = data.get("reactivation")
    method_values = reactivation.get(method) if isinstance(reactivation, Mapping) else None
    baseline_values = reactivation.get(baseline) if isinstance(reactivation, Mapping) else None
    checks: dict[str, object] = {}
    passed = True
    for horizon in (3, 4, 5):
        ours = _horizon_value(method_values, horizon)
        base = _horizon_value(baseline_values, horizon)
        ours_accuracy = ours_recall = base_accuracy = base_recall = None
        if isinstance(ours, Mapping):
            ours_accuracy = _numeric_value(ours.get("accuracy", ours.get("reactivation_accuracy")))
            ours_recall = _numeric_value(ours.get("recall", ours.get("reactivation_recall")))
        if isinstance(base, Mapping):
            base_accuracy = _numeric_value(base.get("accuracy", base.get("reactivation_accuracy")))
            base_recall = _numeric_value(base.get("recall", base.get("reactivation_recall")))
        improved = (
            (ours_accuracy is not None and base_accuracy is not None and ours_accuracy > base_accuracy)
            or (ours_recall is not None and base_recall is not None and ours_recall > base_recall)
        )
        ok = (
            ours_accuracy is not None
            and ours_recall is not None
            and ours_accuracy >= config.g6a2_accuracy
            and ours_recall >= config.g6a2_recall
            and improved
        )
        checks[f"T{horizon}"] = {
            "accuracy": ours_accuracy,
            "recall": ours_recall,
            "baseline_accuracy": base_accuracy,
            "baseline_recall": base_recall,
            "improved": improved,
            "passed": ok,
        }
        passed = passed and ok
    return _gate_result(
        passed,
        checks=checks,
        threshold={
            "accuracy_min": config.g6a2_accuracy,
            "recall_min": config.g6a2_recall,
            "must_improve": True,
        },
    )


def _gate_g6a3(data: Mapping[str, object], config: GateConfig) -> dict[str, object]:
    fingerprints = data.get("raw_prediction_fingerprints")
    fingerprint_values = []
    if isinstance(fingerprints, Mapping):
        fingerprint_values = list(fingerprints.values())
    fingerprint_equal = bool(fingerprint_values) and all(
        value == fingerprint_values[0] for value in fingerprint_values[1:]
    )
    raw = data.get("raw_local_ap", data.get("raw_local_metrics"))
    numeric_values = _raw_values(raw)
    if isinstance(raw, Mapping):
        method_arrays = [_raw_values(value) for value in raw.values()]
        comparable = bool(method_arrays) and all(
            len(array) == len(method_arrays[0]) for array in method_arrays
        )
        numeric_equal = comparable and all(
            abs(array[index] - method_arrays[0][index])
            <= config.g6a3_abs_tolerance
            for array in method_arrays[1:]
            for index in range(len(method_arrays[0]))
        )
    else:
        numeric_equal = bool(numeric_values) and max(numeric_values) - min(numeric_values) <= config.g6a3_abs_tolerance
    passed = (fingerprint_equal or not fingerprints) and (numeric_equal or not numeric_values)
    if not fingerprint_values and not numeric_values:
        passed = False
    return _gate_result(
        passed,
        checks={
            "fingerprints_equal": fingerprint_equal,
            "raw_metric_range": max(numeric_values) - min(numeric_values) if numeric_values else None,
            "numeric_equal": numeric_equal,
        },
        threshold={"absolute_tolerance": config.g6a3_abs_tolerance},
    )


def _task_value(values: object, method: str, horizon: int, metric: str) -> float | None:
    if not isinstance(values, Mapping):
        return None
    method_values = values.get(method)
    row = _horizon_value(method_values, horizon)
    if isinstance(row, Mapping):
        return _numeric_value(row.get(metric))
    return None


def _gate_g6a4(
    data: Mapping[str, object], config: GateConfig, *, method: str, baseline: str
) -> dict[str, object]:
    online = data.get("online_task", data.get("online_metrics"))
    checks: dict[str, object] = {}
    passed = True
    for metric in ("t_mAP", "t_REC"):
        ours = _task_value(online, method, 2, metric)
        base = _task_value(online, baseline, 2, metric)
        drop = None if ours is None or base is None else base - ours
        ok = drop is not None and drop <= config.g6a4_t2_drop + 1e-12
        checks[f"T2_{metric}"] = {"drop": drop, "passed": ok}
        passed = passed and ok
    positive_long = False
    for horizon in (4, 5):
        for metric in ("t_mAP", "t_REC"):
            ours = _task_value(online, method, horizon, metric)
            base = _task_value(online, baseline, horizon, metric)
            delta = None if ours is None or base is None else ours - base
            checks[f"T{horizon}_{metric}"] = {"delta": delta}
            positive_long = positive_long or (delta is not None and delta > 0)
    checks["positive_long_horizon_delta"] = positive_long
    passed = passed and positive_long
    return _gate_result(
        passed,
        checks=checks,
        threshold={"T2_drop_max": config.g6a4_t2_drop, "long_horizon_delta": ">0"},
    )


def _gate_g6a5(data: Mapping[str, object], config: GateConfig) -> dict[str, object]:
    share = _numeric_value(data.get("explainability_share"))
    counts = data.get("failure_counts", data.get("failure_breakdown"))
    if isinstance(counts, Mapping) and isinstance(counts.get("counts"), Mapping):
        counts = counts["counts"]
    if share is None and isinstance(counts, Mapping):
        values: dict[str, float] = {}
        for key, value in counts.items():
            numeric = _numeric_value(value)
            if numeric is not None:
                values[str(key)] = numeric
        total = sum(values.values())
        categorized = sum(values.get(code, 0.0) for code in FAILURE_CODES)
        share = categorized / total if total else None
    passed = share is not None and share >= config.g6a5_explainability
    return _gate_result(
        passed,
        checks={"explainability_share": share},
        threshold={"minimum": config.g6a5_explainability},
    )


def evaluate_gates(
    aggregates: Mapping[str, object],
    *,
    config: GateConfig | None = None,
    method: str = "Persist4D",
    strong_baseline: str = "EMA",
) -> dict[str, object]:
    """Apply preregistered G6A-1 through G6A-5 without tuning thresholds."""

    if not isinstance(aggregates, Mapping):
        raise ValueError("aggregates must be a mapping")  # noqa: TRY004
    config = config or GateConfig()
    config.validate()
    gates = {
        "G6A-1": _gate_g6a1(aggregates, config),
        "G6A-2": _gate_g6a2(
            aggregates, config, method=method, baseline=strong_baseline
        ),
        "G6A-3": _gate_g6a3(aggregates, config),
        "G6A-4": _gate_g6a4(
            aggregates, config, method=method, baseline=strong_baseline
        ),
        "G6A-5": _gate_g6a5(aggregates, config),
    }
    return {**gates, "overall_passed": all(gate["passed"] for gate in gates.values())}


apply_gates = evaluate_gates

# Discoverable compatibility names for artifact/rendering adapters.
AssociationEventRecord = AssociationEvent
EventRecord = AssociationEvent
CapacityRecord = CapacitySnapshot
EfficiencyRow = EfficiencyRecord
PairedRecord = PairedMetricRecord
validate_events = validate_association_events
event_table_metrics = reconstruct_event_metrics
identity_metrics = aggregate_identity_metrics
reactivation_metrics = aggregate_reactivation_metrics
per_sequence_metrics = aggregate_metrics_by_sequence
capacity_audit = audit_capacity
efficiency_metrics = aggregate_efficiency
cluster_bootstrap = paired_cluster_bootstrap
run_gates = evaluate_gates
