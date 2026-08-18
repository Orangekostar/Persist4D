"""Exact common-prefix Protocol B input construction and manifest validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_ID_PATTERN = re.compile(r"scene\d{4}_\d{2}\Z")
PORTABLE_REFERENCE_PREFIXES = ("repo:", "external:", "local_cache:")
DEFAULT_HORIZONS = (2, 3, 4, 5)
DEFAULT_ORDER_VARIANTS = ("canonical", "reverse", "sha256_seed45")


class ProtocolError(ValueError):
    """Raised when a Protocol B input or manifest violates its contract."""


@dataclass(frozen=True)
class OrderVariant:
    """One complete five-visit order with IDs and indices kept paired."""

    name: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]


@dataclass(frozen=True)
class MasterSequence:
    """A validated T5 validation master and its explicitly resolved scans."""

    sequence_id: str
    reference_scene_id: str
    scene: int
    split: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    validation_index: int
    database_index: int

    @property
    def master_sequence_id(self) -> str:
        return self.sequence_id


@dataclass(frozen=True)
class PrefixSequence:
    """A strict prefix of one complete master order."""

    master_sequence_id: str
    reference_scene_id: str
    sequence_id: str
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    horizon: int
    order: str
    validation_index: int
    split: str


@dataclass(frozen=True)
class ProtocolB:
    """Validated Protocol B graph used by evaluators and manifest renderers."""

    masters: tuple[MasterSequence, ...]
    variants: dict[str, dict[str, OrderVariant]]
    prefixes: dict[str, dict[str, dict[int, PrefixSequence]]]
    horizons: tuple[int, ...]
    seed: int
    split: str

    @property
    def order_variants(self) -> tuple[str, ...]:
        return DEFAULT_ORDER_VARIANTS

    @property
    def all_prefixes(self) -> tuple[PrefixSequence, ...]:
        return tuple(
            prefix
            for master in self.masters
            for order in self.order_variants
            for horizon in self.horizons
            for prefix in (self.prefixes[master.sequence_id][order][horizon],)
        )


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml(source: str | Path | Mapping[str, Any]) -> Any:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    try:
        with path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except OSError as error:
        raise ProtocolError(f"cannot load YAML source {path}: {error}") from error


def _load_json(source: str | Path | Sequence[Any] | Mapping[str, Any]) -> Any:
    if isinstance(source, (Mapping, list, tuple)):
        return source
    path = Path(source)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load JSON source {path}: {error}") from error


def _require_scan_id(scan_id: object, *, context: str) -> str:
    if not isinstance(scan_id, str) or SCAN_ID_PATTERN.fullmatch(scan_id) is None:
        raise ProtocolError(f"{context} contains an invalid scan ID: {scan_id!r}")
    return scan_id


def _normalize_scan_ids(scan_ids: Sequence[object], *, context: str) -> tuple[str, ...]:
    if isinstance(scan_ids, (str, bytes)):
        raise ProtocolError(f"{context} must be an explicit sequence of scan IDs")
    try:
        normalized = tuple(
            _require_scan_id(scan_id, context=context) for scan_id in scan_ids
        )
    except TypeError as error:
        raise ProtocolError(
            f"{context} must be an explicit sequence of scan IDs"
        ) from error
    if not normalized:
        raise ProtocolError(f"{context} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ProtocolError(f"{context} contains duplicate scan IDs")
    return normalized


def _require_nonnegative_index(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{context} must be a non-negative integer")
    return int(value)


def _record_scan_id(
    record: Mapping[str, Any], *, context: str, key: object = None
) -> str:
    explicit = record.get("scan_id", record.get("name"))
    if explicit is not None:
        scan_id = _require_scan_id(explicit, context=context)
    else:
        scene = record.get("scene")
        sub_scene = record.get("sub_scene", record.get("subscene"))
        if isinstance(scene, bool) or not isinstance(scene, int):
            raise ProtocolError(
                f"{context} lacks explicit scene/sub_scene identity; positional "
                "index assumptions are forbidden"
            )
        if isinstance(sub_scene, bool) or not isinstance(sub_scene, int):
            raise ProtocolError(
                f"{context} lacks explicit scene/sub_scene identity; positional "
                "index assumptions are forbidden"
            )
        scan_id = f"scene{scene:04d}_{sub_scene:02d}"
        _require_scan_id(scan_id, context=context)
    if key is not None:
        key_scan_id = _require_scan_id(key, context=f"{context} mapping key")
        if key_scan_id != scan_id:
            raise ProtocolError(
                f"{context} mapping key {key_scan_id!r} disagrees with record {scan_id!r}"
            )
    return scan_id


def _record_split(record: Mapping[str, Any]) -> str | None:
    value = record.get("split", record.get("type"))
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ProtocolError("scan metadata split must be a non-empty string")
    return "validation" if value == "val" else value


def _record_is_supervised(record: Mapping[str, Any]) -> bool:
    for key in ("supervised", "is_supervised"):
        if key in record:
            value = record[key]
            if not isinstance(value, bool):
                raise ProtocolError(f"scan metadata field {key!r} must be boolean")
            return value
    # Official validation_database.yaml has one instance GT path per scan.
    for key in (
        "instance_gt_filepath",
        "label_filepath",
        "labels_filepath",
        "ground_truth",
    ):
        value = record.get(key)
        if isinstance(value, str) and value not in {"", "None"}:
            return True
    return False


def _iter_scan_records(
    records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> list[tuple[object, Mapping[str, Any], int]]:
    if isinstance(records, Mapping):
        items = list(records.items())
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        items = [(None, record) for record in records]
    else:
        raise ProtocolError(
            "scan metadata must be mappings keyed by scan ID or a sequence of records"
        )

    normalized: list[tuple[object, Mapping[str, Any], int]] = []
    for position, (key, record) in enumerate(items):
        if not isinstance(record, Mapping):
            raise ProtocolError(
                f"scan metadata record {position} must be a mapping with explicit identity"
            )
        explicit_index = record.get("dataset_index", record.get("scan_index"))
        index = (
            _require_nonnegative_index(
                explicit_index, context=f"scan metadata record {position} index"
            )
            if explicit_index is not None
            else position
        )
        normalized.append((key, record, index))
    return normalized


def load_scan_indices(
    scan_ids: Sequence[object],
    scan_records: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    expected_split: str = "validation",
    require_supervised: bool = True,
    substitution_policy: str = "reject",
) -> tuple[int, ...]:
    """Resolve scan IDs through explicit metadata records.

    The returned indices are metadata indices, never indices inferred from a
    sequence position or silently substituted for a missing scan.
    """

    ids = _normalize_scan_ids(scan_ids, context="scan_ids")
    if substitution_policy != "reject":
        raise ProtocolError(
            "substitution policy must be 'reject'; scan substitution is forbidden"
        )
    if not isinstance(expected_split, str) or not expected_split:
        raise ProtocolError("expected_split must be a non-empty string")

    by_scan_id: dict[str, int] = {}
    for key, record, index in _iter_scan_records(scan_records):
        scan_id = _record_scan_id(
            record, context=f"scan metadata record {index}", key=key
        )
        if scan_id in by_scan_id:
            raise ProtocolError(f"scan metadata contains duplicate scan ID {scan_id!r}")
        split = _record_split(record)
        if split is not None and split != expected_split:
            raise ProtocolError(
                f"scan metadata record {scan_id!r} has wrong split {split!r}; "
                f"expected {expected_split!r}"
            )
        if require_supervised and not _record_is_supervised(record):
            raise ProtocolError(
                f"scan metadata record {scan_id!r} is not proven supervised"
            )
        by_scan_id[scan_id] = index

    missing = [scan_id for scan_id in ids if scan_id not in by_scan_id]
    if missing:
        raise ProtocolError(f"missing scan metadata for {missing!r}")
    return tuple(by_scan_id[scan_id] for scan_id in ids)


def _metadata_reference_ids(
    metadata: Sequence[Mapping[str, Any]] | Mapping[object, Mapping[str, Any]],
) -> dict[int, str]:
    if isinstance(metadata, Mapping):
        items = list(metadata.items())
    elif isinstance(metadata, Sequence) and not isinstance(metadata, (str, bytes)):
        items = list(enumerate(metadata))
    else:
        raise ProtocolError("3RScan metadata must be a sequence or mapping")

    references: dict[int, str] = {}
    for position, (key, record) in enumerate(items):
        if not isinstance(record, Mapping):
            raise ProtocolError(f"3RScan metadata record {position} must be a mapping")
        raw_scene = record.get("scene", key)
        if isinstance(raw_scene, bool) or not isinstance(raw_scene, int):
            raise ProtocolError(
                f"3RScan metadata record {position} lacks an integer scene index"
            )
        reference = record.get("reference", record.get("reference_scene_id"))
        if not isinstance(reference, str) or not reference:
            raise ProtocolError(
                f"3RScan metadata record {position} lacks reference UUID"
            )
        if raw_scene in references and references[raw_scene] != reference:
            raise ProtocolError(
                f"3RScan metadata has duplicate scene index {raw_scene}"
            )
        references[raw_scene] = reference
    return references


def _sequence_entries(source: Any) -> list[tuple[int, str, Mapping[str, Any]]]:
    if not isinstance(source, Mapping) or not source:
        raise ProtocolError("T5 sequence database must be a non-empty mapping")
    entries: list[tuple[int, str, Mapping[str, Any]]] = []
    for database_index, (sequence_id, record) in enumerate(source.items()):
        if not isinstance(sequence_id, str):
            raise ProtocolError("T5 sequence database keys must be strings")
        if not isinstance(record, Mapping):
            raise ProtocolError(f"T5 sequence {sequence_id!r} must map to a record")
        entries.append((database_index, sequence_id, record))
    return entries


def _validate_master_identity(
    sequence_id: str,
    record: Mapping[str, Any],
    *,
    expected_split: str,
) -> tuple[tuple[str, ...], int]:
    scan_ids = _normalize_scan_ids(
        sequence_id.split("-"), context=f"T5 sequence {sequence_id!r}"
    )
    if len(scan_ids) != 5:
        raise ProtocolError(
            f"T5 sequence {sequence_id!r} must contain exactly five scan IDs"
        )
    split = record.get("type", record.get("split"))
    if split != expected_split:
        raise ProtocolError(
            f"T5 sequence {sequence_id!r} has wrong split {split!r}; "
            f"expected {expected_split!r}"
        )
    scene = record.get("scene")
    sub_scenes = record.get("sub_scenes")
    if isinstance(scene, bool) or not isinstance(scene, int):
        raise ProtocolError(f"T5 sequence {sequence_id!r} lacks integer scene")
    if not isinstance(sub_scenes, list) or len(sub_scenes) != 5:
        raise ProtocolError(f"T5 sequence {sequence_id!r} must contain five sub_scenes")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in sub_scenes
    ):
        raise ProtocolError(f"T5 sequence {sequence_id!r} has invalid sub_scenes")
    expected_ids = tuple(
        f"scene{scene:04d}_{sub_scene:02d}" for sub_scene in sub_scenes
    )
    if len(set(expected_ids)) != len(expected_ids):
        raise ProtocolError(
            f"T5 sequence {sequence_id!r} contains duplicate sub_scenes"
        )
    if expected_ids != scan_ids:
        raise ProtocolError(
            f"T5 sequence {sequence_id!r} disagrees with scene/sub_scenes fields"
        )
    return scan_ids, scene


def load_t5_masters(
    sequence_database: str | Path | Mapping[str, Mapping[str, Any]],
    scan_metadata: str
    | Path
    | Sequence[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]],
    *,
    metadata_path: str
    | Path
    | Sequence[Mapping[str, Any]]
    | Mapping[object, Mapping[str, Any]]
    | None = None,
    expected_split: str = "validation",
    expected_master_count: int = 43,
    expected_cluster_count: int = 6,
    require_supervised: bool = True,
    substitution_policy: str = "reject",
) -> tuple[MasterSequence, ...]:
    """Load and validate all T5 masters from one database and one scan index source."""

    database = _load_yaml(sequence_database)
    records = (
        _load_yaml(scan_metadata)
        if isinstance(scan_metadata, (str, Path))
        else scan_metadata
    )
    entries = _sequence_entries(database)
    reference_ids = (
        _metadata_reference_ids(_load_json(metadata_path))
        if metadata_path is not None
        else {}
    )
    if (
        expected_cluster_count == 6
        and metadata_path is None
        and not all(record.get("reference_scene_id") for _, _, record in entries)
    ):
        raise ProtocolError(
            "six UUID reference-scene clusters require explicit 3RScan metadata"
        )

    masters: list[MasterSequence] = []
    for database_index, sequence_id, record in entries:
        if record.get("type", record.get("split")) != expected_split:
            continue
        scan_ids, scene = _validate_master_identity(
            sequence_id,
            record,
            expected_split=expected_split,
        )
        scan_indices = load_scan_indices(
            scan_ids,
            records,
            expected_split=expected_split,
            require_supervised=require_supervised,
            substitution_policy=substitution_policy,
        )
        reference_scene_id = record.get("reference_scene_id")
        if reference_scene_id is None:
            reference_scene_id = reference_ids.get(scene, f"scene{scene:04d}")
        if not isinstance(reference_scene_id, str) or not reference_scene_id:
            raise ProtocolError(
                f"T5 sequence {sequence_id!r} has invalid reference_scene_id"
            )
        masters.append(
            MasterSequence(
                sequence_id=sequence_id,
                reference_scene_id=reference_scene_id,
                scene=scene,
                split=expected_split,
                scan_ids=scan_ids,
                scan_indices=scan_indices,
                validation_index=len(masters),
                database_index=database_index,
            )
        )

    if not masters:
        raise ProtocolError(
            f"T5 sequence database has no masters in expected split {expected_split!r}"
        )
    if expected_master_count is not None and len(masters) != expected_master_count:
        raise ProtocolError(
            f"expected {expected_master_count} T5 validation masters, found {len(masters)}"
        )
    cluster_count = len({master.reference_scene_id for master in masters})
    if expected_cluster_count is not None and cluster_count != expected_cluster_count:
        raise ProtocolError(
            f"expected {expected_cluster_count} reference-scene clusters, found {cluster_count}"
        )
    return tuple(masters)


def _seeded_permutation(scan_ids: tuple[str, ...], *, seed: int) -> tuple[int, ...]:
    keyed = [
        (
            hashlib.sha256(f"{seed}:{scan_id}".encode()).hexdigest(),
            position,
        )
        for position, scan_id in enumerate(scan_ids)
    ]
    return tuple(position for _, position in sorted(keyed))


def _ordered_variant(
    name: str,
    scan_ids: tuple[str, ...],
    scan_indices: tuple[int, ...],
    positions: Sequence[int],
) -> OrderVariant:
    normalized_positions = tuple(positions)
    return OrderVariant(
        name=name,
        scan_ids=tuple(scan_ids[position] for position in normalized_positions),
        scan_indices=tuple(scan_indices[position] for position in normalized_positions),
    )


def build_order_variants(
    scan_ids: Sequence[object],
    scan_indices: Sequence[object],
    *,
    seed: int = 45,
) -> dict[str, OrderVariant]:
    """Build canonical, reverse, and SHA256-derived orders from one full order."""

    ids = _normalize_scan_ids(scan_ids, context="order scan_ids")
    if isinstance(scan_indices, (str, bytes)):
        raise ProtocolError("order scan_indices must be an explicit integer sequence")
    try:
        indices = tuple(
            _require_nonnegative_index(value, context="order scan_indices")
            for value in scan_indices
        )
    except TypeError as error:
        raise ProtocolError(
            "order scan_indices must be an explicit integer sequence"
        ) from error
    if len(ids) != len(indices):
        raise ProtocolError("order scan IDs and indices must have equal lengths")
    if len(set(indices)) != len(indices):
        raise ProtocolError("order scan_indices contains duplicate indices")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError("order seed must be an integer")
    if seed != 45:
        raise ProtocolError("Protocol B order seed is frozen to 45")
    return {
        "canonical": _ordered_variant("canonical", ids, indices, range(len(ids))),
        "reverse": _ordered_variant("reverse", ids, indices, reversed(range(len(ids)))),
        "sha256_seed45": _ordered_variant(
            "sha256_seed45",
            ids,
            indices,
            _seeded_permutation(ids, seed=seed),
        ),
    }


def derive_exact_prefixes(
    master: MasterSequence,
    order_name: str,
    variant: OrderVariant,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[int, PrefixSequence]:
    """Derive every horizon by slicing one already-complete order."""

    if variant.name != order_name:
        raise ProtocolError(
            f"order name {order_name!r} disagrees with variant {variant.name!r}"
        )
    if set(variant.scan_ids) != set(master.scan_ids) or len(variant.scan_ids) != 5:
        raise ProtocolError(
            "order variant must contain exactly the master's five scans"
        )
    expected_indices = dict(zip(master.scan_ids, master.scan_indices))
    if any(
        expected_indices[scan_id] != index
        for scan_id, index in zip(variant.scan_ids, variant.scan_indices)
    ):
        raise ProtocolError("order variant changed scan ID/index pairing")
    normalized_horizons = tuple(horizons)
    if normalized_horizons != tuple(sorted(set(normalized_horizons))):
        raise ProtocolError("horizons must be sorted and unique")
    if any(
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 2
        or horizon > 5
        for horizon in normalized_horizons
    ):
        raise ProtocolError("horizons must be integers in [2, 5]")
    return {
        horizon: PrefixSequence(
            master_sequence_id=master.sequence_id,
            reference_scene_id=master.reference_scene_id,
            sequence_id="-".join(variant.scan_ids[:horizon]),
            scan_ids=variant.scan_ids[:horizon],
            scan_indices=variant.scan_indices[:horizon],
            horizon=horizon,
            order=order_name,
            validation_index=master.validation_index,
            split=master.split,
        )
        for horizon in normalized_horizons
    }


def build_protocol_b(
    sequence_database: str | Path | Mapping[str, Mapping[str, Any]],
    scan_metadata: str
    | Path
    | Sequence[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]],
    *,
    metadata_path: str
    | Path
    | Sequence[Mapping[str, Any]]
    | Mapping[object, Mapping[str, Any]]
    | None = None,
    expected_split: str = "validation",
    expected_master_count: int = 43,
    expected_cluster_count: int = 6,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seed: int = 45,
    require_supervised: bool = True,
    substitution_policy: str = "reject",
) -> ProtocolB:
    """Construct Protocol B with one complete order per master and order variant."""

    normalized_horizons = tuple(horizons)
    masters = load_t5_masters(
        sequence_database,
        scan_metadata,
        metadata_path=metadata_path,
        expected_split=expected_split,
        expected_master_count=expected_master_count,
        expected_cluster_count=expected_cluster_count,
        require_supervised=require_supervised,
        substitution_policy=substitution_policy,
    )
    variants: dict[str, dict[str, OrderVariant]] = {}
    prefixes: dict[str, dict[str, dict[int, PrefixSequence]]] = {}
    for master in masters:
        master_variants = build_order_variants(
            master.scan_ids,
            master.scan_indices,
            seed=seed,
        )
        variants[master.sequence_id] = master_variants
        prefixes[master.sequence_id] = {
            name: derive_exact_prefixes(
                master,
                name,
                variant,
                horizons=normalized_horizons,
            )
            for name, variant in master_variants.items()
        }
    return ProtocolB(
        masters=masters,
        variants=variants,
        prefixes=prefixes,
        horizons=normalized_horizons,
        seed=seed,
        split=expected_split,
    )


def _portable_reference(path: str | Path, *, repository_root: str | Path) -> str:
    value = str(path)
    if value.startswith(PORTABLE_REFERENCE_PREFIXES):
        return value
    resolved = Path(path).resolve()
    root = Path(repository_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        if resolved.name == "3RScan.json":
            return "external:3RScan/3RScan.json"
        return f"external:{resolved.name}"
    return f"repo:{relative.as_posix()}"


def _source_descriptor(
    source: str | Path | None,
    *,
    role: str,
    repository_root: str | Path,
) -> dict[str, Any]:
    if source is None:
        return {"reference": f"external:provided/{role}", "sha256": None}
    path = Path(source)
    if not path.is_file():
        raise ProtocolError(f"source file for {role} does not exist: {path}")
    return {
        "reference": _portable_reference(path, repository_root=repository_root),
        "sha256": sha256_file(path),
    }


def _manifest_master(protocol: ProtocolB, master: MasterSequence) -> dict[str, Any]:
    orders: dict[str, Any] = {}
    for order_name in protocol.order_variants:
        variant = protocol.variants[master.sequence_id][order_name]
        orders[order_name] = {
            "visit_order": list(variant.scan_ids),
            "scan_indices": list(variant.scan_indices),
            "prefixes": {
                str(horizon): {
                    "sequence_id": protocol.prefixes[master.sequence_id][order_name][
                        horizon
                    ].sequence_id,
                    "scan_ids": list(
                        protocol.prefixes[master.sequence_id][order_name][
                            horizon
                        ].scan_ids
                    ),
                    "scan_indices": list(
                        protocol.prefixes[master.sequence_id][order_name][
                            horizon
                        ].scan_indices
                    ),
                }
                for horizon in protocol.horizons
            },
        }
    return {
        "master_sequence_id": master.sequence_id,
        "reference_scene_id": master.reference_scene_id,
        "scene": master.scene,
        "split": master.split,
        "validation_index": master.validation_index,
        "database_index": master.database_index,
        "scan_ids": list(master.scan_ids),
        "scan_indices": list(master.scan_indices),
        "visit_order": list(master.scan_ids),
        "orders": orders,
        "prefixes": {
            order_name: order_data["prefixes"]
            for order_name, order_data in orders.items()
        },
    }


def build_protocol_b_manifest(
    protocol: ProtocolB,
    *,
    sequence_database_path: str | Path | None = None,
    scan_metadata_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    source_manifest_path: str | Path | None = None,
    config_path: str | Path | None = None,
    repository_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Render a deterministic, portable Protocol B manifest payload."""

    manifest = {
        "schema_version": "protocol-b-v1",
        "protocol": {
            "name": "common-prefix",
            "split": protocol.split,
            "master_horizon": 5,
            "horizons": list(protocol.horizons),
            "expected_master_count": len(protocol.masters),
            "expected_reference_scene_clusters": len(
                {master.reference_scene_id for master in protocol.masters}
            ),
            "order_variants": list(protocol.order_variants),
            "seed": protocol.seed,
            "order_semantics": "metadata_order_only_no_timestamps",
            "substitution_policy": "reject",
            "scan_index_resolution": "explicit_scan_id_metadata_map",
            "require_supervised": True,
        },
        "sources": {
            "sequence_database": _source_descriptor(
                sequence_database_path,
                role="sequence_database",
                repository_root=repository_root,
            ),
            "scan_metadata": _source_descriptor(
                scan_metadata_path,
                role="validation_database",
                repository_root=repository_root,
            ),
            "metadata": _source_descriptor(
                metadata_path,
                role="3RScan.json",
                repository_root=repository_root,
            ),
            "source_manifest": _source_descriptor(
                source_manifest_path,
                role="source_manifest",
                repository_root=repository_root,
            ),
            "config": _source_descriptor(
                config_path,
                role="p6a/default.yaml",
                repository_root=repository_root,
            ),
        },
        "masters": [
            _manifest_master(protocol, master)
            for master in sorted(protocol.masters, key=lambda item: item.sequence_id)
        ],
    }
    validate_protocol_b_manifest(manifest)
    return manifest


def _validate_portable_sources(sources: object) -> None:
    if not isinstance(sources, Mapping):
        raise ProtocolError("manifest sources must be a mapping")
    for role, descriptor in sources.items():
        if not isinstance(descriptor, Mapping):
            raise ProtocolError(f"manifest source {role!r} must be a mapping")
        reference = descriptor.get("reference")
        if not isinstance(reference, str) or not reference.startswith(
            PORTABLE_REFERENCE_PREFIXES
        ):
            raise ProtocolError(f"manifest source {role!r} has non-portable reference")
        if (
            reference.startswith(("repo:/", "external:/", "local_cache:/"))
            or "/home/" in reference
            or "/Users/" in reference
            or "\\" in reference
            or "://" in reference
        ):
            raise ProtocolError(f"manifest source {role!r} contains a private path")
        digest = descriptor.get("sha256")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProtocolError(f"manifest source {role!r} has invalid SHA-256")


def validate_protocol_b_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if a serialized Protocol B manifest is not self-consistent."""

    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "protocol-b-v1"
    ):
        raise ProtocolError("manifest schema_version must be protocol-b-v1")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ProtocolError("manifest protocol must be a mapping")
    if protocol.get("master_horizon") != 5 or protocol.get("horizons") != [2, 3, 4, 5]:
        raise ProtocolError("manifest does not define exact T2-T5 horizons")
    if protocol.get("order_variants") != list(DEFAULT_ORDER_VARIANTS):
        raise ProtocolError(
            "manifest order variants are not canonical/reverse/sha256_seed45"
        )
    if protocol.get("substitution_policy") != "reject":
        raise ProtocolError("manifest substitution policy must be reject")
    _validate_portable_sources(manifest.get("sources"))
    masters = manifest.get("masters")
    if not isinstance(masters, list) or not masters:
        raise ProtocolError("manifest masters must be a non-empty list")
    expected_count = protocol.get("expected_master_count")
    if expected_count != len(masters):
        raise ProtocolError("manifest master count does not match records")
    master_ids: set[str] = set()
    cluster_ids: set[str] = set()
    for master in masters:
        if not isinstance(master, Mapping):
            raise ProtocolError("manifest master must be a mapping")
        master_id = master.get("master_sequence_id")
        if not isinstance(master_id, str) or master_id in master_ids:
            raise ProtocolError("manifest has duplicate or invalid master sequence ID")
        master_ids.add(master_id)
        reference_scene_id = master.get("reference_scene_id")
        if not isinstance(reference_scene_id, str) or not reference_scene_id:
            raise ProtocolError("manifest master lacks reference_scene_id")
        cluster_ids.add(reference_scene_id)
        scan_ids = _normalize_scan_ids(
            master.get("scan_ids", []), context="manifest master scan_ids"
        )
        scan_indices = tuple(master.get("scan_indices", []))
        if len(scan_ids) != 5 or len(scan_indices) != 5:
            raise ProtocolError("manifest master must contain five scans and indices")
        if master_id != "-".join(scan_ids) or master.get("visit_order") != list(
            scan_ids
        ):
            raise ProtocolError(
                "manifest master ID or canonical visit order is inconsistent"
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in scan_indices
        ):
            raise ProtocolError("manifest master has invalid scan indices")
        orders = master.get("orders")
        if not isinstance(orders, Mapping):
            raise ProtocolError("manifest master orders must be a mapping")
        top_level_prefixes = master.get("prefixes")
        if top_level_prefixes != {
            order_name: orders[order_name].get("prefixes")
            for order_name in DEFAULT_ORDER_VARIANTS
            if isinstance(orders.get(order_name), Mapping)
        }:
            raise ProtocolError(
                "manifest top-level prefixes disagree with order records"
            )
        for order_name in DEFAULT_ORDER_VARIANTS:
            order = orders.get(order_name)
            if not isinstance(order, Mapping):
                raise ProtocolError(f"manifest master lacks order {order_name!r}")
            order_ids = _normalize_scan_ids(
                order.get("visit_order", []),
                context=f"manifest {order_name} visit_order",
            )
            order_indices = tuple(order.get("scan_indices", []))
            if len(order_ids) != 5 or len(order_indices) != 5:
                raise ProtocolError(f"manifest {order_name} must contain five scans")
            expected_pairs = dict(zip(scan_ids, scan_indices))
            if any(
                expected_pairs.get(scan_id) != index
                for scan_id, index in zip(order_ids, order_indices)
            ):
                raise ProtocolError(
                    f"manifest {order_name} changed scan ID/index pairing"
                )
            if order_name == "canonical" and (
                order_ids != scan_ids or order_indices != scan_indices
            ):
                raise ProtocolError(
                    "manifest canonical order differs from the master order"
                )
            if order_name == "reverse" and (
                order_ids != tuple(reversed(scan_ids))
                or order_indices != tuple(reversed(scan_indices))
            ):
                raise ProtocolError("manifest reverse order is not the exact reverse")
            if order_name == "sha256_seed45":
                seeded_positions = _seeded_permutation(scan_ids, seed=45)
                if order_ids != tuple(
                    scan_ids[position] for position in seeded_positions
                ):
                    raise ProtocolError(
                        "manifest SHA256 seed45 order is not deterministic"
                    )
            prefixes = order.get("prefixes")
            if not isinstance(prefixes, Mapping):
                raise ProtocolError(f"manifest {order_name} prefixes must be a mapping")
            for horizon in DEFAULT_HORIZONS:
                prefix = prefixes.get(str(horizon))
                if not isinstance(prefix, Mapping):
                    raise ProtocolError(
                        f"manifest {order_name} lacks T{horizon} prefix"
                    )
                prefix_ids = tuple(prefix.get("scan_ids", []))
                prefix_indices = tuple(prefix.get("scan_indices", []))
                if (
                    prefix_ids != order_ids[:horizon]
                    or prefix_indices != order_indices[:horizon]
                ):
                    raise ProtocolError(
                        f"manifest {order_name} T{horizon} is not an exact full-order prefix"
                    )
                if prefix.get("sequence_id") != "-".join(prefix_ids):
                    raise ProtocolError(
                        f"manifest {order_name} T{horizon} has wrong sequence ID"
                    )
    if protocol.get("expected_reference_scene_clusters") != len(cluster_ids):
        raise ProtocolError(
            "manifest reference-scene cluster count does not match records"
        )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize manifest values using one stable JSON encoding."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def write_protocol_b_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    """Validate and write a byte-deterministic manifest."""

    validate_protocol_b_manifest(manifest)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(manifest) + b"\n")


# Descriptive aliases used by downstream P6-A runners.
render_protocol_b_manifest = build_protocol_b_manifest
validate_manifest = validate_protocol_b_manifest
