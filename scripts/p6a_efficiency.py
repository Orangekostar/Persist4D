"""Raw P6-A efficiency evidence manifest and deterministic CSV aggregation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real

from scripts.p6a_artifacts import CSV_COLUMN_SCHEMAS

SCHEMA_VERSION = 1
ORDER_IDS = ("canonical", "reverse", "sha256_seed45")
ROW_TYPES = ("bootstrap", "new_visit", "full_history")
RECORD_KEYS = (
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "stage_id",
    "row_type",
    "model_latency_ms",
    "tracker_latency_ms",
    "association_overhead_ms",
    "memory_update_overhead_ms",
    "gpu_peak_memory_bytes",
    "persistent_state_bytes",
)
PROVENANCE_KEYS = (
    "source_commit",
    "checkpoint_sha256",
    "config_sha256",
    "protocol_sha256",
    "cache_manifest_sha256",
)
COVERAGE_KEYS = (
    "record_count",
    "master_sequence_count",
    "reference_cluster_count",
    "order_variants",
    "by_row_type",
)
MANIFEST_KEYS = {
    "schema_version",
    "status",
    "provenance",
    "coverage",
    "records_sha256",
    "records",
}

_ROW_TYPE_ORDER = {row_type: index for index, row_type in enumerate(ROW_TYPES)}
_EXPECTED_GROUP_COUNTS = {
    ("bootstrap", 1): 129,
    **{("new_visit", horizon): 129 for horizon in range(2, 6)},
    **{("full_history", horizon): 129 for horizon in range(2, 6)},
}
_Row = dict[str, object]

__all__ = (
    "aggregate_efficiency_rows",
    "build_efficiency_manifest",
    "validate_efficiency_manifest",
)


def _exact_keys(value: object, expected: Sequence[str], *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{name} keys differ: missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha(value: object, *, name: str, length: int) -> str:
    text = _nonempty_string(value, name=name)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-{length * 4}")
    return text


def _finite_nonnegative(value: object, *, name: str, required: bool = True) -> object:
    if value is None:
        if required:
            raise ValueError(f"{name} must be finite and non-negative")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")  # noqa: TRY004
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive_integer(value: object, *, name: str, required: bool = True) -> object:
    if value is None:
        if required:
            raise ValueError(f"{name} must be a positive integer")
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    return int(value)


def _record_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _ROW_TYPE_ORDER[str(record["row_type"])],
        int(record["T"]),
        str(record["master_sequence_id"]),
        ORDER_IDS.index(str(record["order_id"])),
        str(record["reference_scene_id"]),
        int(record["stage_id"]),
    )


def _validate_record(raw: object, *, index: int) -> _Row:
    record = _exact_keys(raw, RECORD_KEYS, name=f"records[{index}]")
    normalized = {key: record[key] for key in RECORD_KEYS}
    _nonempty_string(normalized["reference_scene_id"], name="reference_scene_id")
    _nonempty_string(normalized["master_sequence_id"], name="master_sequence_id")
    order_id = _nonempty_string(normalized["order_id"], name="order_id")
    if order_id not in ORDER_IDS:
        raise ValueError(f"order_id must be one of {ORDER_IDS}")
    horizon = _integer(normalized["T"], name="T")
    stage_id = _integer(normalized["stage_id"], name="stage_id")
    row_type = normalized["row_type"]
    if not isinstance(row_type, str) or row_type not in ROW_TYPES:
        raise ValueError("row_type must be bootstrap, new_visit, or full_history")

    _finite_nonnegative(normalized["model_latency_ms"], name="model_latency_ms")
    _positive_integer(normalized["gpu_peak_memory_bytes"], name="gpu_peak_memory_bytes")

    if row_type == "bootstrap":
        if horizon != 1 or stage_id != 0:
            raise ValueError("bootstrap rows require T=1 and stage_id=0")
    else:
        if horizon not in (2, 3, 4, 5) or stage_id != horizon - 1:
            raise ValueError("new_visit and full_history rows require stage_id=T-1 for T2-T5")

    if row_type == "full_history":
        for field in (
            "tracker_latency_ms",
            "association_overhead_ms",
            "memory_update_overhead_ms",
            "persistent_state_bytes",
        ):
            if normalized[field] is not None:
                raise ValueError(f"full_history {field} must be None")
        return normalized

    tracker = _finite_nonnegative(
        normalized["tracker_latency_ms"], name="tracker_latency_ms"
    )
    association = _finite_nonnegative(
        normalized["association_overhead_ms"], name="association_overhead_ms"
    )
    memory_update = _finite_nonnegative(
        normalized["memory_update_overhead_ms"], name="memory_update_overhead_ms"
    )
    if float(tracker) + 1e-9 < float(association) + float(memory_update):
        raise ValueError("tracker_latency_ms must cover association and memory overhead")
    _positive_integer(
        normalized["persistent_state_bytes"], name="persistent_state_bytes"
    )
    return normalized


def _validate_records(records: object) -> list[_Row]:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    normalized = [_validate_record(record, index=index) for index, record in enumerate(records)]
    expected = set(_EXPECTED_GROUP_COUNTS)
    groups: dict[tuple[str, int], list[_Row]] = {}
    for record in normalized:
        groups.setdefault((str(record["row_type"]), int(record["T"])), []).append(record)
    if set(groups) != expected:
        raise ValueError("records do not cover the exact bootstrap/new_visit/full_history groups")
    for key, group in groups.items():
        if len(group) != _EXPECTED_GROUP_COUNTS[key]:
            raise ValueError(f"records coverage for {key} must contain exactly 129 rows")
        masters = {str(record["master_sequence_id"]) for record in group}
        references = {str(record["reference_scene_id"]) for record in group}
        if len(masters) != 43 or len(references) != 6:
            raise ValueError(f"records coverage for {key} must contain 43 masters and 6 references")
        by_master: dict[str, list[_Row]] = {}
        for record in group:
            by_master.setdefault(str(record["master_sequence_id"]), []).append(record)
        if any(
            {str(record["order_id"]) for record in master_rows} != set(ORDER_IDS)
            or len({str(record["reference_scene_id"]) for record in master_rows}) != 1
            for master_rows in by_master.values()
        ):
            raise ValueError(
                f"records coverage for {key} must contain canonical/reverse/sha256_seed45 per master"
            )
    ordered = sorted(normalized, key=_record_sort_key)
    if normalized != ordered:
        raise ValueError("records must be in canonical row type/T/master/order order")
    return normalized


def _coverage(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_row_type: dict[str, dict[str, int]] = {
        "bootstrap": {"T1": 129},
        "new_visit": {f"T{horizon}": 129 for horizon in range(2, 6)},
        "full_history": {f"T{horizon}": 129 for horizon in range(2, 6)},
    }
    return {
        "record_count": len(records),
        "master_sequence_count": 43,
        "reference_cluster_count": 6,
        "order_variants": list(ORDER_IDS),
        "by_row_type": by_row_type,
    }


def _records_digest(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(records),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_efficiency_manifest(
    records: Iterable[Mapping[str, object]],
    source_commit: str,
    checkpoint_sha256: str,
    config_sha256: str,
    protocol_sha256: str,
    cache_manifest_sha256: str,
) -> dict[str, object]:
    """Build and validate a canonical raw efficiency manifest."""

    raw_records = list(records)
    normalized = [_validate_record(record, index=index) for index, record in enumerate(raw_records)]
    normalized.sort(key=_record_sort_key)
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "provenance": {
            "source_commit": _sha(source_commit, name="source_commit", length=40),
            "checkpoint_sha256": _sha(
                checkpoint_sha256, name="checkpoint_sha256", length=64
            ),
            "config_sha256": _sha(config_sha256, name="config_sha256", length=64),
            "protocol_sha256": _sha(protocol_sha256, name="protocol_sha256", length=64),
            "cache_manifest_sha256": _sha(
                cache_manifest_sha256, name="cache_manifest_sha256", length=64
            ),
        },
        "coverage": _coverage(normalized),
        "records_sha256": _records_digest(normalized),
        "records": normalized,
    }
    validate_efficiency_manifest(manifest)
    return manifest


def validate_efficiency_manifest(manifest: object) -> None:
    """Fail closed on schema, provenance, coverage, or record drift."""

    root = _exact_keys(manifest, tuple(MANIFEST_KEYS), name="efficiency manifest")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported efficiency manifest schema_version")
    if root["status"] != "pass":
        raise ValueError("efficiency manifest status must be pass")
    provenance = _exact_keys(root["provenance"], PROVENANCE_KEYS, name="provenance")
    _sha(provenance["source_commit"], name="provenance.source_commit", length=40)
    for key in PROVENANCE_KEYS[1:]:
        _sha(provenance[key], name=f"provenance.{key}", length=64)
    records = _validate_records(root["records"])
    coverage = _exact_keys(root["coverage"], COVERAGE_KEYS, name="coverage")
    expected_coverage = _coverage(records)
    if dict(coverage) != expected_coverage:
        raise ValueError("efficiency manifest coverage is inconsistent")
    _sha(root["records_sha256"], name="records_sha256", length=64)
    if root["records_sha256"] != _records_digest(records):
        raise ValueError("records_sha256 is inconsistent")


def _mean(records: Sequence[Mapping[str, object]], field: str) -> float:
    values = [float(record[field]) for record in records]
    return sum(values) / len(values)


def _max_positive(records: Sequence[Mapping[str, object]], field: str) -> int:
    return max(int(record[field]) for record in records)


def _ordered_output_row(values: Mapping[str, object]) -> _Row:
    schema = CSV_COLUMN_SCHEMAS["efficiency_results.csv"]
    return {field: values[field] for field in schema}


def aggregate_efficiency_rows(
    manifest: Mapping[str, object],
) -> tuple[_Row, ...]:
    """Aggregate raw records into the 12 registered efficiency result rows."""

    validate_efficiency_manifest(manifest)
    records = manifest["records"]
    if not isinstance(records, list):  # pragma: no cover - validated above.
        raise TypeError("efficiency manifest records must be a list")
    by_group: dict[tuple[str, int], list[Mapping[str, object]]] = {}
    for record in records:
        if not isinstance(record, Mapping):  # pragma: no cover - validated above.
            raise TypeError("efficiency records must be mappings")
        by_group.setdefault((str(record["row_type"]), int(record["T"])), []).append(record)

    rows: list[_Row] = []
    bootstrap = by_group[("bootstrap", 1)]
    bootstrap_values = {
        "count": len(bootstrap),
        "bootstrap_latency_ms": _mean(bootstrap, "model_latency_ms"),
        "gpu_peak_memory_bytes": _max_positive(bootstrap, "gpu_peak_memory_bytes"),
        "persistent_state_bytes": _max_positive(bootstrap, "persistent_state_bytes"),
    }
    for horizon in range(2, 6):
        rows.append(
            _ordered_output_row(
                {
                    "method": "B4",
                    "T": horizon,
                    "stage_id": 0,
                    "row_type": "bootstrap",
                    **bootstrap_values,
                    "new_visit_latency_ms": None,
                    "association_overhead_ms": None,
                    "memory_update_overhead_ms": None,
                    "full_history_latency_ms": None,
                }
            )
        )

    for horizon in range(2, 6):
        group = by_group[("new_visit", horizon)]
        rows.append(
            _ordered_output_row(
                {
                    "method": "B4",
                    "T": horizon,
                    "stage_id": horizon - 1,
                    "row_type": "new_visit",
                    "count": len(group),
                    "bootstrap_latency_ms": None,
                    "new_visit_latency_ms": _mean(group, "model_latency_ms"),
                    "association_overhead_ms": _mean(group, "association_overhead_ms"),
                    "memory_update_overhead_ms": _mean(group, "memory_update_overhead_ms"),
                    "full_history_latency_ms": None,
                    "gpu_peak_memory_bytes": _max_positive(group, "gpu_peak_memory_bytes"),
                    "persistent_state_bytes": _max_positive(group, "persistent_state_bytes"),
                }
            )
        )

    for horizon in range(2, 6):
        group = by_group[("full_history", horizon)]
        rows.append(
            _ordered_output_row(
                {
                    "method": "full_history_rescene",
                    "T": horizon,
                    "stage_id": horizon - 1,
                    "row_type": "full_history",
                    "count": len(group),
                    "bootstrap_latency_ms": None,
                    "new_visit_latency_ms": None,
                    "association_overhead_ms": None,
                    "memory_update_overhead_ms": None,
                    "full_history_latency_ms": _mean(group, "model_latency_ms"),
                    "gpu_peak_memory_bytes": _max_positive(group, "gpu_peak_memory_bytes"),
                    "persistent_state_bytes": None,
                }
            )
        )
    return tuple(rows)
