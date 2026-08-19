"""Preregistered split and search-space contracts for Persist4D P6-B."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import yaml

from models.persistent_memory_p6b import P6BMemoryConfig


class P6BProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class P6BSource:
    reference: str
    sha256: str


@dataclass(frozen=True)
class P6BSources:
    p6a_protocol_manifest: P6BSource
    p6a_cache_manifest: P6BSource


@dataclass(frozen=True)
class P6BSplitConfig:
    algorithm: str
    namespace: str
    tuning_cluster_count: int
    heldout_cluster_count: int
    expected_master_count: int
    expected_reference_scene_clusters: int
    order_variants: tuple[str, ...]


@dataclass(frozen=True)
class P6BSearchSpace:
    assignment_modes: tuple[str, ...]
    active_thresholds: tuple[float, ...]
    reactivation_thresholds: tuple[float, ...]
    reactivation_margins: tuple[float, ...]
    class_modes: tuple[str, ...]
    class_weights: tuple[float, ...]
    consolidation_confidences: tuple[float, ...]
    consolidation_margins: tuple[float, ...]
    birth_confidences: tuple[float, ...]
    birth_minimum_mask_supports: tuple[int, ...]
    birth_max_entropies: tuple[float | None, ...]


@dataclass(frozen=True)
class P6BEligibility:
    minimum_reactivation_accuracy: float
    maximum_reactivation_recall_drop: float
    minimum_valid_observation_ratio: float
    maximum_t2_task_drop: float


@dataclass(frozen=True)
class P6BProtocolConfig:
    schema_version: int
    seed: int
    sources: P6BSources
    split: P6BSplitConfig
    base: P6BMemoryConfig
    search: P6BSearchSpace
    eligibility: P6BEligibility
    ranking: tuple[str, ...]
    heldout_evaluation_count: int


@dataclass(frozen=True)
class P6BSplitAssignment:
    reference_scene_id: str
    master_sequence_id: str
    order_ids: tuple[str, ...]
    partition: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "reference_scene_id": self.reference_scene_id,
            "master_sequence_id": self.master_sequence_id,
            "order_ids": list(self.order_ids),
            "partition": self.partition,
        }


@dataclass(frozen=True)
class P6BSplitManifest:
    schema_version: int
    seed: int
    hash_algorithm: str
    hash_namespace: str
    tuning_reference_scene_ids: tuple[str, ...]
    heldout_reference_scene_ids: tuple[str, ...]
    tuning_master_sequence_ids: tuple[str, ...]
    heldout_master_sequence_ids: tuple[str, ...]
    assignments: tuple[P6BSplitAssignment, ...]
    sha256: str

    def to_mapping(self) -> dict[str, object]:
        return {
            **_split_payload(self),
            "sha256": self.sha256,
        }


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P6BProtocolError(f"{context} must be a mapping")
    return value


def _exact_fields(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise P6BProtocolError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise P6BProtocolError(
            f"{context} must be an integer of at least {minimum}"
        )
    return value


def _finite(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise P6BProtocolError(f"{context} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise P6BProtocolError(f"{context} must be finite")
    return number


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise P6BProtocolError(f"{context} must be a nonempty string")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P6BProtocolError(f"{context} must be a sequence")
    return value


def _tuple_strings(value: object, context: str) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{context} item") for item in _sequence(value, context)
    )
    if not result or len(set(result)) != len(result):
        raise P6BProtocolError(f"{context} must be nonempty and unique")
    return result


def _tuple_floats(value: object, context: str) -> tuple[float, ...]:
    result = tuple(
        _finite(item, f"{context} item") for item in _sequence(value, context)
    )
    if not result or len(set(result)) != len(result):
        raise P6BProtocolError(f"{context} must be nonempty and unique")
    return result


def _source(value: object, context: str) -> P6BSource:
    mapping = _mapping(value, context)
    _exact_fields(mapping, {"reference", "sha256"}, context)
    reference = _string(mapping["reference"], f"{context}.reference")
    if not reference.startswith(("repo:", "external:", "local_cache:")):
        raise P6BProtocolError(f"{context}.reference must be portable")
    sha256 = _string(mapping["sha256"], f"{context}.sha256")
    if len(sha256) != 64:
        raise P6BProtocolError(f"{context}.sha256 must contain 64 hex digits")
    try:
        int(sha256, 16)
    except ValueError as error:
        raise P6BProtocolError(
            f"{context}.sha256 must contain 64 hex digits"
        ) from error
    return P6BSource(reference=reference, sha256=sha256)


def load_p6b_config(path: str | Path) -> P6BProtocolConfig:
    source_path = Path(path)
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise P6BProtocolError(f"cannot read P6-B config: {error}") from error
    root = _mapping(payload, "config")
    _exact_fields(
        root,
        {
            "schema_version",
            "seed",
            "sources",
            "split",
            "base_memory",
            "search",
            "selection",
        },
        "config",
    )
    schema_version = _integer(root["schema_version"], "schema_version", minimum=1)
    if schema_version != 1:
        raise P6BProtocolError("schema_version must be 1")
    seed = _integer(root["seed"], "seed")

    sources_mapping = _mapping(root["sources"], "sources")
    _exact_fields(
        sources_mapping,
        {"p6a_protocol_manifest", "p6a_cache_manifest"},
        "sources",
    )
    sources = P6BSources(
        p6a_protocol_manifest=_source(
            sources_mapping["p6a_protocol_manifest"],
            "sources.p6a_protocol_manifest",
        ),
        p6a_cache_manifest=_source(
            sources_mapping["p6a_cache_manifest"],
            "sources.p6a_cache_manifest",
        ),
    )

    split_mapping = _mapping(root["split"], "split")
    split_fields = {
        "algorithm",
        "namespace",
        "tuning_cluster_count",
        "heldout_cluster_count",
        "expected_master_count",
        "expected_reference_scene_clusters",
        "order_variants",
    }
    _exact_fields(split_mapping, split_fields, "split")
    split = P6BSplitConfig(
        algorithm=_string(split_mapping["algorithm"], "split.algorithm"),
        namespace=_string(split_mapping["namespace"], "split.namespace"),
        tuning_cluster_count=_integer(
            split_mapping["tuning_cluster_count"],
            "split.tuning_cluster_count",
            minimum=1,
        ),
        heldout_cluster_count=_integer(
            split_mapping["heldout_cluster_count"],
            "split.heldout_cluster_count",
            minimum=1,
        ),
        expected_master_count=_integer(
            split_mapping["expected_master_count"],
            "split.expected_master_count",
            minimum=1,
        ),
        expected_reference_scene_clusters=_integer(
            split_mapping["expected_reference_scene_clusters"],
            "split.expected_reference_scene_clusters",
            minimum=1,
        ),
        order_variants=_tuple_strings(
            split_mapping["order_variants"], "split.order_variants"
        ),
    )
    if split.algorithm != "sha256" or split.namespace != "p6b":
        raise P6BProtocolError("split algorithm and namespace must be sha256/p6b")
    if (
        split.tuning_cluster_count + split.heldout_cluster_count
        != split.expected_reference_scene_clusters
    ):
        raise P6BProtocolError("split cluster counts must cover all references")

    base_mapping = _mapping(root["base_memory"], "base_memory")
    base_fields = {field.name for field in fields(P6BMemoryConfig)}
    _exact_fields(base_mapping, base_fields, "base_memory")
    try:
        base = P6BMemoryConfig(**dict(base_mapping))
    except (TypeError, ValueError) as error:
        raise P6BProtocolError(f"invalid base_memory: {error}") from error

    search_mapping = _mapping(root["search"], "search")
    search_fields = {field.name for field in fields(P6BSearchSpace)}
    _exact_fields(search_mapping, search_fields, "search")
    entropies = tuple(
        None if item is None else _finite(item, "birth_max_entropies item")
        for item in _sequence(
            search_mapping["birth_max_entropies"], "birth_max_entropies"
        )
    )
    supports = tuple(
        _integer(item, "birth_minimum_mask_supports item", minimum=1)
        for item in _sequence(
            search_mapping["birth_minimum_mask_supports"],
            "birth_minimum_mask_supports",
        )
    )
    search = P6BSearchSpace(
        assignment_modes=_tuple_strings(
            search_mapping["assignment_modes"], "assignment_modes"
        ),
        active_thresholds=_tuple_floats(
            search_mapping["active_thresholds"], "active_thresholds"
        ),
        reactivation_thresholds=_tuple_floats(
            search_mapping["reactivation_thresholds"],
            "reactivation_thresholds",
        ),
        reactivation_margins=_tuple_floats(
            search_mapping["reactivation_margins"], "reactivation_margins"
        ),
        class_modes=_tuple_strings(
            search_mapping["class_modes"], "class_modes"
        ),
        class_weights=_tuple_floats(
            search_mapping["class_weights"], "class_weights"
        ),
        consolidation_confidences=_tuple_floats(
            search_mapping["consolidation_confidences"],
            "consolidation_confidences",
        ),
        consolidation_margins=_tuple_floats(
            search_mapping["consolidation_margins"],
            "consolidation_margins",
        ),
        birth_confidences=_tuple_floats(
            search_mapping["birth_confidences"], "birth_confidences"
        ),
        birth_minimum_mask_supports=supports,
        birth_max_entropies=entropies,
    )
    for stage in (
        "assignment",
        "reactivation",
        "class_compatibility",
        "consolidation",
        "birth_gate",
    ):
        expand_stage_configs(base, search, stage=stage)

    selection = _mapping(root["selection"], "selection")
    _exact_fields(
        selection,
        {"eligibility", "ranking", "heldout_evaluation_count"},
        "selection",
    )
    eligibility_mapping = _mapping(selection["eligibility"], "eligibility")
    eligibility_fields = {field.name for field in fields(P6BEligibility)}
    _exact_fields(eligibility_mapping, eligibility_fields, "eligibility")
    eligibility = P6BEligibility(
        **{
            field: _finite(eligibility_mapping[field], f"eligibility.{field}")
            for field in eligibility_fields
        }
    )
    if not (
        0.0 <= eligibility.minimum_reactivation_accuracy <= 1.0
        and 0.0 <= eligibility.maximum_reactivation_recall_drop <= 1.0
        and 0.0 <= eligibility.minimum_valid_observation_ratio <= 1.0
        and eligibility.maximum_t2_task_drop >= 0.0
    ):
        raise P6BProtocolError("eligibility values are outside their valid ranges")
    ranking = _tuple_strings(selection["ranking"], "selection.ranking")
    heldout_count = _integer(
        selection["heldout_evaluation_count"],
        "selection.heldout_evaluation_count",
        minimum=1,
    )
    if heldout_count != 1:
        raise P6BProtocolError("heldout_evaluation_count must be exactly 1")
    return P6BProtocolConfig(
        schema_version=schema_version,
        seed=seed,
        sources=sources,
        split=split,
        base=base,
        search=search,
        eligibility=eligibility,
        ranking=ranking,
        heldout_evaluation_count=heldout_count,
    )


def canonical_config_json(config: P6BMemoryConfig) -> str:
    if not isinstance(config, P6BMemoryConfig):
        raise P6BProtocolError("config must be a P6BMemoryConfig")
    return json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_config_id(config: P6BMemoryConfig) -> str:
    digest = hashlib.sha256(canonical_config_json(config).encode("ascii")).hexdigest()
    return f"p6b-{digest[:16]}"


def _unique_sorted(configs: Sequence[P6BMemoryConfig]) -> tuple[P6BMemoryConfig, ...]:
    by_json = {canonical_config_json(config): config for config in configs}
    return tuple(by_json[key] for key in sorted(by_json))


def expand_stage_configs(
    base: P6BMemoryConfig,
    search: P6BSearchSpace,
    *,
    stage: str,
) -> tuple[P6BMemoryConfig, ...]:
    if not isinstance(base, P6BMemoryConfig):
        raise P6BProtocolError("base must be a P6BMemoryConfig")
    configs: list[P6BMemoryConfig] = []
    try:
        if stage == "assignment":
            configs = [
                replace(base, assignment_mode=mode)
                for mode in search.assignment_modes
            ]
        elif stage == "reactivation":
            configs = [
                replace(
                    base,
                    active_threshold=active,
                    reactivation_threshold=reactivation,
                    reactivation_margin=margin,
                )
                for active in search.active_thresholds
                for reactivation in search.reactivation_thresholds
                for margin in search.reactivation_margins
                if reactivation >= active
            ]
        elif stage == "class_compatibility":
            configs = [
                replace(base, class_mode=mode, class_weight=weight)
                for mode in search.class_modes
                for weight in search.class_weights
            ]
        elif stage == "consolidation":
            configs = [
                replace(
                    base,
                    consolidation_confidence=None,
                    consolidation_margin=None,
                )
            ] + [
                replace(
                    base,
                    consolidation_confidence=confidence,
                    consolidation_margin=margin,
                )
                for confidence in search.consolidation_confidences
                for margin in search.consolidation_margins
            ]
        elif stage == "birth_gate":
            configs = [
                replace(
                    base,
                    birth_confidence=confidence,
                    birth_minimum_mask_support=support,
                    birth_max_entropy=entropy,
                )
                for confidence in search.birth_confidences
                for support in search.birth_minimum_mask_supports
                for entropy in search.birth_max_entropies
            ]
        else:
            raise P6BProtocolError(f"unknown search stage: {stage}")
    except ValueError as error:
        raise P6BProtocolError(f"invalid {stage} search space: {error}") from error
    result = _unique_sorted(configs)
    if not result:
        raise P6BProtocolError(f"{stage} search space is empty")
    return result


def joint_neighbor_configs(
    selected: P6BMemoryConfig, search: P6BSearchSpace
) -> tuple[P6BMemoryConfig, ...]:
    grids: dict[str, tuple[object, ...]] = {
        "assignment_mode": search.assignment_modes,
        "active_threshold": search.active_thresholds,
        "reactivation_threshold": search.reactivation_thresholds,
        "reactivation_margin": search.reactivation_margins,
        "class_mode": search.class_modes,
        "class_weight": search.class_weights,
        "consolidation_confidence": search.consolidation_confidences,
        "consolidation_margin": search.consolidation_margins,
        "birth_confidence": search.birth_confidences,
        "birth_minimum_mask_support": search.birth_minimum_mask_supports,
        "birth_max_entropy": search.birth_max_entropies,
    }
    neighbors = [selected]
    for field_name, values in grids.items():
        current = getattr(selected, field_name)
        if current not in values:
            continue
        index = values.index(current)
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(values):
                continue
            try:
                neighbors.append(
                    replace(selected, **{field_name: values[neighbor_index]})
                )
            except ValueError:
                continue
    remaining = _unique_sorted(neighbors[1:])
    return (selected, *remaining)


def _split_payload(manifest: P6BSplitManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "seed": manifest.seed,
        "hash_algorithm": manifest.hash_algorithm,
        "hash_namespace": manifest.hash_namespace,
        "tuning_reference_scene_ids": list(
            manifest.tuning_reference_scene_ids
        ),
        "heldout_reference_scene_ids": list(
            manifest.heldout_reference_scene_ids
        ),
        "tuning_master_sequence_ids": list(manifest.tuning_master_sequence_ids),
        "heldout_master_sequence_ids": list(manifest.heldout_master_sequence_ids),
        "assignments": [assignment.to_mapping() for assignment in manifest.assignments],
    }


def _split_digest(manifest: P6BSplitManifest) -> str:
    serialized = json.dumps(
        _split_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def build_split_manifest(
    protocol_manifest: Mapping[str, object], *, seed: int = 45
) -> P6BSplitManifest:
    root = _mapping(protocol_manifest, "protocol_manifest")
    if root.get("schema_version") != "protocol-b-v1":
        raise P6BProtocolError("protocol_manifest schema must be protocol-b-v1")
    protocol = _mapping(root.get("protocol"), "protocol_manifest.protocol")
    if protocol.get("expected_master_count") != 43:
        raise P6BProtocolError("protocol_manifest must declare 43 masters")
    if protocol.get("expected_reference_scene_clusters") != 6:
        raise P6BProtocolError("protocol_manifest must declare six reference clusters")
    order_ids = _tuple_strings(
        protocol.get("order_variants"), "protocol_manifest order_variants"
    )
    if order_ids != ("canonical", "reverse", "sha256_seed45"):
        raise P6BProtocolError("protocol_manifest order variants differ from P6-A")
    masters = _sequence(root.get("masters"), "protocol_manifest.masters")
    if len(masters) != 43:
        raise P6BProtocolError("protocol_manifest must contain exactly 43 masters")
    master_rows: list[tuple[str, str]] = []
    for index, value in enumerate(masters):
        master = _mapping(value, f"masters[{index}]")
        master_id = _string(
            master.get("master_sequence_id"),
            f"masters[{index}].master_sequence_id",
        )
        reference_id = _string(
            master.get("reference_scene_id"),
            f"masters[{index}].reference_scene_id",
        )
        orders = _mapping(master.get("orders"), f"masters[{index}].orders")
        if tuple(orders) != order_ids or set(orders) != set(order_ids):
            raise P6BProtocolError("every master must contain all P6-A order variants")
        master_rows.append((master_id, reference_id))
    master_ids = [master_id for master_id, _ in master_rows]
    if len(set(master_ids)) != len(master_ids):
        raise P6BProtocolError("protocol_manifest contains duplicate master IDs")
    references = {reference_id for _, reference_id in master_rows}
    if len(references) != 6:
        raise P6BProtocolError("protocol_manifest must contain six reference clusters")
    seed_value = _integer(seed, "seed")
    ordered_references = tuple(
        sorted(
            references,
            key=lambda reference: (
                hashlib.sha256(
                    f"p6b|{seed_value}|{reference}".encode()
                ).hexdigest(),
                reference,
            ),
        )
    )
    tuning_references = ordered_references[:4]
    heldout_references = ordered_references[4:]
    assignments = tuple(
        P6BSplitAssignment(
            reference_scene_id=reference_id,
            master_sequence_id=master_id,
            order_ids=order_ids,
            partition=(
                "tuning" if reference_id in tuning_references else "heldout"
            ),
        )
        for master_id, reference_id in sorted(master_rows)
    )
    tuning_masters = tuple(
        assignment.master_sequence_id
        for assignment in assignments
        if assignment.partition == "tuning"
    )
    heldout_masters = tuple(
        assignment.master_sequence_id
        for assignment in assignments
        if assignment.partition == "heldout"
    )
    if len(tuning_masters) != 32 or len(heldout_masters) != 11:
        raise P6BProtocolError("cluster split must contain 32 tuning and 11 heldout masters")
    provisional = P6BSplitManifest(
        schema_version=1,
        seed=seed_value,
        hash_algorithm="sha256",
        hash_namespace="p6b",
        tuning_reference_scene_ids=tuning_references,
        heldout_reference_scene_ids=heldout_references,
        tuning_master_sequence_ids=tuning_masters,
        heldout_master_sequence_ids=heldout_masters,
        assignments=assignments,
        sha256="",
    )
    return replace(provisional, sha256=_split_digest(provisional))
