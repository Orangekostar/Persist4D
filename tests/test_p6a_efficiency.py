from __future__ import annotations

import copy
import hashlib
import json
import random

import pytest

from scripts.p6a_artifacts import CSV_COLUMN_SCHEMAS
from scripts.p6a_efficiency import (
    aggregate_efficiency_rows,
    build_efficiency_manifest,
    validate_efficiency_manifest,
)

SOURCE_COMMIT = "a" * 40
CHECKPOINT_SHA256 = "b" * 64
CONFIG_SHA256 = "c" * 64
PROTOCOL_SHA256 = "d" * 64
CACHE_MANIFEST_SHA256 = "e" * 64
ORDERS = ("canonical", "reverse", "sha256_seed45")


def _record(
    row_type: str,
    horizon: int,
    master_index: int,
    order_id: str,
) -> dict[str, object]:
    reference_index = master_index % 6
    base = float(master_index + 1)
    record: dict[str, object] = {
        "reference_scene_id": f"reference-{reference_index}",
        "master_sequence_id": f"master-{master_index:02d}",
        "order_id": order_id,
        "T": horizon,
        "stage_id": 0 if row_type == "bootstrap" else horizon - 1,
        "row_type": row_type,
        "model_latency_ms": base + horizon,
        "tracker_latency_ms": 3.0,
        "association_overhead_ms": 1.0,
        "memory_update_overhead_ms": 1.0,
        "gpu_peak_memory_bytes": 100 + master_index,
        "persistent_state_bytes": 1000 + master_index,
    }
    if row_type == "full_history":
        record.update(
            {
                "tracker_latency_ms": None,
                "association_overhead_ms": None,
                "memory_update_overhead_ms": None,
                "persistent_state_bytes": None,
            }
        )
    return record


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for master_index in range(43):
        for order_id in ORDERS:
            records.append(_record("bootstrap", 1, master_index, order_id))
            for horizon in range(2, 6):
                records.append(_record("new_visit", horizon, master_index, order_id))
                records.append(_record("full_history", horizon, master_index, order_id))
    return records


def _manifest(records: list[dict[str, object]] | None = None) -> dict[str, object]:
    return build_efficiency_manifest(
        _records() if records is None else records,
        SOURCE_COMMIT,
        CHECKPOINT_SHA256,
        CONFIG_SHA256,
        PROTOCOL_SHA256,
        CACHE_MANIFEST_SHA256,
    )


def test_build_manifest_normalizes_records_and_binds_exact_provenance() -> None:
    records = _records()
    shuffled = copy.deepcopy(records)
    random.Random(45).shuffle(shuffled)

    manifest = _manifest(shuffled)

    assert set(manifest) == {
        "schema_version",
        "status",
        "provenance",
        "coverage",
        "records_sha256",
        "records",
    }
    assert manifest["schema_version"] == 1
    assert manifest["status"] == "pass"
    assert manifest["provenance"] == {
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "cache_manifest_sha256": CACHE_MANIFEST_SHA256,
    }
    assert manifest["coverage"] == {
        "record_count": 1161,
        "master_sequence_count": 43,
        "reference_cluster_count": 6,
        "order_variants": list(ORDERS),
        "by_row_type": {
            "bootstrap": {"T1": 129},
            "new_visit": {f"T{horizon}": 129 for horizon in range(2, 6)},
            "full_history": {f"T{horizon}": 129 for horizon in range(2, 6)},
        },
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            manifest["records"],
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["records_sha256"] == expected_digest
    validate_efficiency_manifest(manifest)


def test_manifest_validation_rejects_digest_provenance_and_record_schema_drift() -> None:
    manifest = _manifest()

    bad_digest = copy.deepcopy(manifest)
    bad_digest["records_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="records_sha256"):
        validate_efficiency_manifest(bad_digest)

    bad_provenance = copy.deepcopy(manifest)
    bad_provenance["provenance"]["config_sha256"] = "not-a-sha"
    with pytest.raises(ValueError, match="provenance"):
        validate_efficiency_manifest(bad_provenance)

    bad_record = copy.deepcopy(manifest)
    bad_record["records"][0].pop("row_type")
    with pytest.raises(ValueError, match="record"):
        validate_efficiency_manifest(bad_record)


def test_aggregate_efficiency_rows_reuses_bootstrap_raw_values_for_all_horizons() -> None:
    rows = aggregate_efficiency_rows(_manifest())

    assert len(rows) == 12
    assert tuple(rows[0]) == CSV_COLUMN_SCHEMAS["efficiency_results.csv"]
    assert [(row["method"], row["T"], row["row_type"]) for row in rows] == [
        *[("B4", horizon, "bootstrap") for horizon in range(2, 6)],
        *[("B4", horizon, "new_visit") for horizon in range(2, 6)],
        *[("full_history_rescene", horizon, "full_history") for horizon in range(2, 6)],
    ]

    bootstrap = [row for row in rows if row["row_type"] == "bootstrap"]
    assert all(row["stage_id"] == 0 and row["count"] == 129 for row in bootstrap)
    assert {row["bootstrap_latency_ms"] for row in bootstrap} == {26.0}
    assert {row["new_visit_latency_ms"] for row in bootstrap} == {None}
    assert {row["association_overhead_ms"] for row in bootstrap} == {None}
    assert {row["gpu_peak_memory_bytes"] for row in bootstrap} == {142}
    assert {row["persistent_state_bytes"] for row in bootstrap} == {1042}

    new_visit = [row for row in rows if row["row_type"] == "new_visit"]
    assert all(row["stage_id"] == row["T"] - 1 for row in new_visit)
    assert [row["new_visit_latency_ms"] for row in new_visit] == [27.0, 28.0, 29.0, 30.0]
    assert all(row["association_overhead_ms"] == 1.0 for row in new_visit)
    assert all(row["memory_update_overhead_ms"] == 1.0 for row in new_visit)
    assert all(row["count"] == 129 for row in new_visit)

    full_history = [row for row in rows if row["row_type"] == "full_history"]
    assert all(row["stage_id"] == row["T"] - 1 for row in full_history)
    assert [row["full_history_latency_ms"] for row in full_history] == [24.0, 25.0, 26.0, 27.0]
    assert all(row["persistent_state_bytes"] is None for row in full_history)
    assert all(row["gpu_peak_memory_bytes"] == 142 for row in full_history)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tracker_latency_ms", 1.0),
        ("association_overhead_ms", None),
        ("memory_update_overhead_ms", None),
        ("gpu_peak_memory_bytes", 0),
        ("persistent_state_bytes", 0),
        ("model_latency_ms", float("nan")),
    ],
)
def test_manifest_rejects_invalid_bootstrap_or_new_visit_measurements(
    field: str, value: object
) -> None:
    records = _records()
    records[0][field] = value

    with pytest.raises(ValueError):
        _manifest(records)


def test_manifest_rejects_full_history_tracker_components_or_state() -> None:
    records = _records()
    records[-1]["tracker_latency_ms"] = 1.0

    with pytest.raises(ValueError, match="full_history"):
        _manifest(records)


def test_manifest_requires_exact_coverage_and_tracker_decomposition() -> None:
    records = _records()
    records[0]["tracker_latency_ms"] = 1.0
    with pytest.raises(ValueError, match="tracker"):
        _manifest(records)

    records = _records()[:-1]
    with pytest.raises(ValueError, match="coverage|record"):
        _manifest(records)


def test_manifest_requires_one_master_reference_mapping_across_all_groups() -> None:
    records = _records()
    for record in records:
        if (
            record["row_type"] == "new_visit"
            and record["T"] == 2
            and record["master_sequence_id"] == "master-00"
        ):
            record["master_sequence_id"] = "master-99"

    with pytest.raises(ValueError, match="master/reference mapping"):
        _manifest(records)
