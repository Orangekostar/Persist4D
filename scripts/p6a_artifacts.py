"""Strict, deterministic, CPU-only evidence helpers for Persist4D P6-A.

The root payload is the only source of truth.  Renderers never read a file
from disk while constructing a bundle; publication verifies every rendered
byte against the root manifest before exposing any output.
"""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.p6a_cache import contains_windows_absolute_path

SCHEMA_VERSION = 2
ROOT_ARTIFACT_PATH = "p6a_eval.json"

ROOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "source_commit",
        "source_tree_contract",
        "p5_frozen_hashes",
        "protocol",
        "provenance",
        "methods",
        "horizons",
        "settings",
        "metric_blocks",
        "fingerprints",
        "analysis",
        "change_label_limitation",
        "derived_artifacts",
        "artifact_manifest",
        "gate_results",
        "claims_supported",
        "claims_not_supported",
        "next_action",
        "errors",
    }
)
SOURCE_TREE_KEYS = frozenset({"status", "source_commit"})
P5_FROZEN_VALUES = {
    "source_commit": "92bab01e93bacbc939606ec7c7f58d3f9b334fe6",
    "artifact_commit": "1380c4b9f37bec7933126ccc9bd70067de166f6f",
    "checkpoint_sha256": "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e",
    "json_sha256": "7da68910b0c0b43b5f04d8ae7d56543a460231c0616c62b2fb9485b88fd781a1",
    "markdown_sha256": "f2115bde732317e27aab8791dbe4744fcd2354b955ab8f1fe9338b0d351abe78",
}
P5_FROZEN_KEYS = frozenset(P5_FROZEN_VALUES)
PROTOCOL_KEYS = frozenset(
    {
        "name",
        "horizons",
        "master_sequence_count",
        "cluster_count",
        "order_count",
        "cache_entry_count",
    }
)
PROVENANCE_KEYS = frozenset({"checkpoint", "config", "dataset", "prediction_cache"})
PROVENANCE_RECORD_KEYS = frozenset({"ref", "sha256"})
METHOD_RECORD_KEYS = frozenset({"mode", "metric_block"})
HORIZON_IDS = ("T2", "T3", "T4", "T5")
HORIZON_SEQUENCE_COUNTS = {"T2": 129, "T3": 129, "T4": 129, "T5": 129}
METHOD_IDS = ("B0", "B0_sanity", "B1", "B2", "B3", "B4", "Oracle")
ONLINE_METHOD_IDS = METHOD_IDS[:-1]
METRIC_BLOCK_IDS = ("raw", "strict", "offline")
METRIC_FIELDS = (
    "AP",
    "AP50",
    "AP25",
    "REC",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "t_REC50",
    "t_REC25",
)
ANALYSIS_GROUPS = (
    "association",
    "error",
    "reactivation",
    "capacity",
    "efficiency",
    "statistical",
)
ANALYSIS_RECORD_KEYS = frozenset({"path", "rows", "status"})
CHANGE_LABEL_KEYS = frozenset({"available", "reason", "scope"})
GATE_RECORD_KEYS = frozenset({"passed", "evidence"})
GATE_IDS = tuple(f"G6A-{index}" for index in range(1, 6))
MANIFEST_RECORD_KEYS = frozenset({"path", "bytes", "sha256"})
DERIVED_KIND_IDS = ("csv", "json", "markdown", "svg", "yaml")
DERIVED_RECORD_KEYS = {
    "csv": frozenset({"columns", "rows"}),
    "json": frozenset({"text"}),
    "markdown": frozenset({"text"}),
    "svg": frozenset({"text"}),
    "yaml": frozenset({"text"}),
}
REQUIRED_CSV_PATHS = frozenset(
    {
        "baseline_results.csv",
        "strict_online_results.csv",
        "raw_local_results.csv",
        "per_sequence_results.csv",
        "association_events.csv",
        "error_breakdown.csv",
        "error_breakdown_T2.csv",
        "error_breakdown_T3.csv",
        "error_breakdown_T4.csv",
        "error_breakdown_T5.csv",
        "reactivation_audit.csv",
        "reactivation_score_distribution.csv",
        "reactivation_margin_distribution.csv",
        "reactivation_by_gap.csv",
        "capacity_audit.csv",
        "efficiency_results.csv",
    }
)
REQUIRED_JSON_PATHS = frozenset(
    {"protocol_b_manifest.json", "efficiency_raw_manifest.json"}
)
REQUIRED_MARKDOWN_PATHS = frozenset({"statistical_analysis.md"})
REQUIRED_YAML_PATHS = frozenset(
    {"configs/resolved_runtime.yaml", "configs/p6a_default.yaml"}
)
REQUIRED_SVG_PATHS = frozenset(
    {
        "figures/figure_a_identity.svg",
        "figures/figure_b_online_tmap.svg",
        "figures/figure_c_reactivation.svg",
        "figures/figure_d_failures.svg",
        "figures/figure_e_latency.svg",
    }
)
REPORT_PATH = "P6A_GO_NOGO_REPORT.md"

ONLINE_METHOD_SET = tuple(METHOD_IDS[:-1])
ALL_HORIZONS = tuple(HORIZON_IDS)
REACTIVATION_METHOD_SET = ("B1", "B2", "B3", "B4")
REACTIVATION_HORIZONS = ("T3", "T4", "T5")
FAILURE_CATEGORIES = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")
CAPACITY_METHOD = "B4"

CSV_COLUMN_SCHEMAS = {
    "baseline_results.csv": (
        "method",
        "T",
        "raw_AP",
        "online_t_mAP",
        "online_t_REC",
        "id_switch_rate",
        "reactivation_accuracy",
    ),
    "strict_online_results.csv": (
        "method",
        "T",
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "t_REC50",
        "t_REC25",
    ),
    "raw_local_results.csv": ("method", "T", "AP", "AP50", "AP25", "REC"),
    "per_sequence_results.csv": (
        "method",
        "reference_scene_id",
        "master_sequence_id",
        "scene_id",
        "sequence_id",
        "order_id",
        "prefix",
        "T",
        "prediction_digest",
        "id_switches",
        "transition_opportunities",
        "id_switch_rate",
        "active_correct_matches",
        "active_wrong_matches",
        "births",
        "false_births",
        "rejected_births",
        "fragmentation_count",
        "merge_count",
        "gap_opportunities",
        "reactivation_attempts",
        "predicted_reactivation_events",
        "correct_reactivations",
        "wrong_reactivations",
        "no_attempts",
        "reactivation_accuracy",
        "reactivation_precision",
        "reactivation_recall",
        "reactivation_coverage",
    ),
    "association_events.csv": (
        "event_id",
        "scene_id",
        "sequence_id",
        "reference_scene_id",
        "master_sequence_id",
        "order_id",
        "prefix",
        "method",
        "stage_id",
        "event_kind",
        "query_id",
        "candidate_slot_id",
        "predicted_identity_id",
        "gt_entity_id",
        "association_correct",
        "feature_similarity",
        "class_similarity",
        "total_score",
        "best_score",
        "second_best_score",
        "score_margin",
        "observation_confidence",
        "mask_support",
        "predicted_class",
        "class_entropy",
        "slot_age",
        "last_seen_stage",
        "gap_length",
        "slot_active",
        "slot_occupied",
        "association_result",
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
        "failure_category",
        "failure_code",
        "prediction_digest",
        "cache_digest",
    ),
    "error_breakdown.csv": ("method", "T", "category", "count", "share"),
    "reactivation_audit.csv": (
        "method",
        "T",
        "gap_opportunities",
        "reactivation_attempts",
        "correct_reactivations",
        "wrong_reactivations",
        "no_attempts",
        "reactivation_accuracy",
        "reactivation_precision",
        "reactivation_recall",
        "reactivation_coverage",
    ),
    "reactivation_score_distribution.csv": (
        "method",
        "T",
        "outcome",
        "bin_low",
        "bin_high",
        "count",
        "fraction",
    ),
    "reactivation_margin_distribution.csv": (
        "method",
        "T",
        "outcome",
        "bin_low",
        "bin_high",
        "count",
        "fraction",
    ),
    "reactivation_by_gap.csv": (
        "method",
        "T",
        "gap_length",
        "outcome",
        "count",
        "fraction",
    ),
    "capacity_audit.csv": (
        "method",
        "T",
        "stage_id",
        "capacity",
        "birth_count",
        "occupied_count",
        "active_count",
        "dormant_count",
        "peak_occupied",
        "peak_active",
        "peak_dormant",
        "occupancy_ratio",
        "rejected_births",
        "persistent_state_bytes",
    ),
    "efficiency_results.csv": (
        "method",
        "T",
        "stage_id",
        "row_type",
        "count",
        "bootstrap_latency_ms",
        "new_visit_latency_ms",
        "association_overhead_ms",
        "memory_update_overhead_ms",
        "full_history_latency_ms",
        "gpu_peak_memory_bytes",
        "persistent_state_bytes",
    ),
}
CSV_PRIMARY_KEYS = {
    "baseline_results.csv": ("method", "T"),
    "strict_online_results.csv": ("method", "T"),
    "raw_local_results.csv": ("method", "T"),
    "per_sequence_results.csv": (
        "method",
        "reference_scene_id",
        "master_sequence_id",
        "scene_id",
        "sequence_id",
        "order_id",
        "prefix",
        "T",
    ),
    "association_events.csv": ("event_id",),
    "error_breakdown.csv": ("method", "T", "category"),
    "reactivation_audit.csv": ("method", "T"),
    "reactivation_score_distribution.csv": (
        "method",
        "T",
        "outcome",
        "bin_low",
        "bin_high",
    ),
    "reactivation_margin_distribution.csv": (
        "method",
        "T",
        "outcome",
        "bin_low",
        "bin_high",
    ),
    "reactivation_by_gap.csv": ("method", "T", "gap_length", "outcome"),
    "capacity_audit.csv": ("method", "T", "stage_id"),
    "efficiency_results.csv": ("method", "T", "stage_id", "row_type"),
}
P6A_REPORT_SECTIONS = (
    "What was changed",
    "Why it was changed",
    "Experimental protocol",
    "Reproducibility binding",
    "Main results",
    "Statistical evidence",
    "Failure analysis",
    "What claims are supported",
    "What claims are NOT supported",
    "GO / NO-GO decision",
    "Exact next action",
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"(?:^|[^A-Za-z])GPU-[0-9A-Fa-f-]+")
_IPV4 = re.compile(r"(?<![0-9A-Za-z])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9A-Za-z])")
_IPV6 = re.compile(r"(?i)(?<![0-9A-F])(?:[0-9A-F]{1,4}:){2,}[0-9A-F:]+(?![0-9A-F])")
_PRIVATE_TEXT = (
    re.compile(r"/(?:home|Users|root|private|mnt)/"),
    _GPU_UUID,
    re.compile(r"ssh://"),
)


def _exact_keys(value: object, expected: frozenset[str] | set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{name} keys differ: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, name: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")  # noqa: TRY004
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    result = [_nonempty_string(item, name=f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_scalar_tree(value: object, *, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        normalized_path = value.replace("\\", "/")
        if (
            PurePosixPath(value).is_absolute()
            or contains_windows_absolute_path(value)
            or ".." in PurePosixPath(normalized_path).parts
        ):
            raise ValueError(f"{path} contains an absolute or traversing path")
        if any(pattern.search(value) for pattern in _PRIVATE_TEXT):
            raise ValueError(f"{path} contains private or non-portable text")
        if _IPV4.search(value) or _IPV6.search(value):
            candidate = _IPV4.search(value) or _IPV6.search(value)
            try:
                ipaddress.ip_address(candidate.group(0))
            except ValueError:
                pass
            else:
                raise ValueError(f"{path} contains an IP address")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} mapping keys must be strings")  # noqa: TRY004
            _validate_scalar_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_scalar_tree(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _validate_sha(value: object, *, name: str, length: int) -> str:
    digest = _nonempty_string(value, name=name)
    pattern = _HEX_40 if length == 40 else _HEX_64
    if pattern.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA{length * 4}")
    return digest


def _validate_reference(
    value: object, *, name: str, expected_prefix: str | None = None
) -> str:
    reference = _nonempty_string(value, name=name)
    scheme, separator, payload = reference.partition(":")
    if separator != ":" or scheme not in {"repo", "external", "local_cache"}:
        raise ValueError(f"{name} must use a portable reference prefix")
    if not payload or payload.startswith("/") or "\\" in payload:
        raise ValueError(f"{name} must be a relative portable reference")
    if ".." in PurePosixPath(payload).parts or "//" in payload:
        raise ValueError(f"{name} must not traverse parent directories")
    if expected_prefix is not None and not reference.startswith(expected_prefix):
        raise ValueError(f"{name} must start with {expected_prefix}")
    return reference


def _validate_relative_artifact_path(value: object, *, name: str) -> str:
    text = _nonempty_string(value, name=name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or "\\" in text
        or contains_windows_absolute_path(text)
        or path.parts[0] in {"P5", "P6B", "artifacts"}
    ):
        raise ValueError(f"{name} must be a safe relative P6A path")
    return path.as_posix()


def _horizon_token(value: object, *, name: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{name} must identify T2 through T5")  # noqa: TRY004
    if isinstance(value, int) and value in (2, 3, 4, 5):
        return f"T{value}"
    if isinstance(value, str) and value in HORIZON_IDS:
        return value
    raise ValueError(f"{name} must identify T2 through T5")


def _csv_schema_path(path: str) -> str:
    if re.fullmatch(r"error_breakdown_T[2345]\.csv", path):
        return "error_breakdown.csv"
    return path


def _csv_rows_with_schema(
    path: str, columns: object, rows: object
) -> list[Mapping[str, object]]:
    schema_path = _csv_schema_path(path)
    expected = CSV_COLUMN_SCHEMAS.get(schema_path)
    if expected is None:
        raise ValueError(f"{path} has no registered CSV schema")
    if not isinstance(columns, list) or tuple(columns) != expected:
        raise ValueError(f"{path} columns do not match the exact schema")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} rows must not be empty")
    validated: list[Mapping[str, object]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or tuple(raw_row) != expected:
            raise ValueError(f"{path} row {index} has schema drift")
        _validate_scalar_tree(raw_row, path=f"{path}.rows[{index}]")
        validated.append(raw_row)
    primary_key = CSV_PRIMARY_KEYS[schema_path]
    seen: set[tuple[object, ...]] = set()
    for index, row in enumerate(validated):
        key = tuple(row[field] for field in primary_key)
        if any(value is None for value in key):
            raise ValueError(f"{path} row {index} has a null primary-key field")
        try:
            duplicate = key in seen
        except TypeError as error:
            raise ValueError(
                f"{path} row {index} has an unhashable primary-key field"
            ) from error
        if duplicate:
            raise ValueError(f"{path} contains duplicate primary key")
        seen.add(key)
    return validated


def _unit_or_none(value: object, *, name: str) -> None:
    _finite_number(value, name=name, allow_none=True)
    if value is not None and not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


def _nonnegative_or_none(value: object, *, name: str) -> None:
    if value is None:
        return
    _integer(value, name=name, minimum=0)


def _nonnegative_integer(value: object, *, name: str) -> int:
    return _integer(value, name=name, minimum=0)


def _unit(value: object, *, name: str) -> None:
    _finite_number(value, name=name)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


def _validate_ratio(
    value: object, numerator: int, denominator: int, *, name: str
) -> None:
    expected = numerator / denominator if denominator else None
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be null when its denominator is zero")
    elif value is None or not math.isclose(float(value), expected, abs_tol=1e-9):
        raise ValueError(f"{name} does not match its count ratio")


def _validate_method_horizon_grid(
    path: str, rows: Sequence[Mapping[str, object]], methods: Sequence[str]
) -> None:
    actual = {
        (
            _nonempty_string(row["method"], name=f"{path}.method"),
            _horizon_token(row["T"], name=f"{path}.T"),
        )
        for row in rows
    }
    expected = {(method, horizon) for method in methods for horizon in HORIZON_IDS}
    if actual != expected:
        raise ValueError(f"{path} must contain exactly one row per method and horizon")


def _validate_metric_table(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    _validate_method_horizon_grid(path, rows, ONLINE_METHOD_SET)
    fields = CSV_COLUMN_SCHEMAS[path][2:]
    for row in rows:
        for field in fields:
            _unit_or_none(row[field], name=f"{path}.{field}")


def _validate_per_sequence_rows(
    path: str, rows: Sequence[Mapping[str, object]]
) -> None:
    counts: dict[tuple[object, str], int] = {}
    units: dict[tuple[object, str], set[tuple[object, ...]]] = {}
    for row in rows:
        method = _nonempty_string(row["method"], name=f"{path}.method")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        prefix = _integer(row["prefix"], name=f"{path}.prefix", minimum=2)
        if prefix != int(horizon[1]):
            raise ValueError(f"{path}.prefix does not match T")
        for field in (
            "reference_scene_id",
            "master_sequence_id",
            "scene_id",
            "sequence_id",
            "order_id",
        ):
            _nonempty_string(row[field], name=f"{path}.{field}")
        _validate_sha(row["prediction_digest"], name=f"{path}.prediction_digest", length=64)
        key = (method, horizon)
        counts[key] = counts.get(key, 0) + 1
        units.setdefault(key, set()).add(
            (
                row["reference_scene_id"],
                row["master_sequence_id"],
                row["scene_id"],
                row["sequence_id"],
                row["order_id"],
            )
        )
        if method not in ONLINE_METHOD_SET:
            raise ValueError(f"{path} contains an unsupported method")
        if row["order_id"] not in {"canonical", "reverse", "sha256_seed45"}:
            raise ValueError(f"{path} contains an unsupported order variant")
        for field in (
            "id_switches",
            "transition_opportunities",
            "active_correct_matches",
            "active_wrong_matches",
            "births",
            "false_births",
            "rejected_births",
            "fragmentation_count",
            "merge_count",
            "gap_opportunities",
            "reactivation_attempts",
            "predicted_reactivation_events",
            "correct_reactivations",
            "wrong_reactivations",
            "no_attempts",
        ):
            _nonnegative_integer(row[field], name=f"{path}.{field}")
        for field in (
            "id_switch_rate",
            "reactivation_accuracy",
            "reactivation_precision",
            "reactivation_recall",
            "reactivation_coverage",
        ):
            _unit_or_none(row[field], name=f"{path}.{field}")
        gap_opportunities = int(row["gap_opportunities"])
        attempts = int(row["reactivation_attempts"])
        correct = int(row["correct_reactivations"])
        wrong = int(row["wrong_reactivations"])
        if attempts > gap_opportunities or correct > attempts:
            raise ValueError(f"{path} contains impossible reactivation counts")
        if int(row["no_attempts"]) != gap_opportunities - attempts:
            raise ValueError(f"{path}.no_attempts does not match gap opportunities")
        predicted = int(row["predicted_reactivation_events"])
        if predicted < correct or predicted != correct + wrong:
            raise ValueError(f"{path}.predicted_reactivation_events is inconsistent")
        _validate_ratio(
            row["id_switch_rate"], int(row["id_switches"]),
            int(row["transition_opportunities"]), name=f"{path}.id_switch_rate",
        )
        _validate_ratio(
            row["reactivation_accuracy"], correct, attempts,
            name=f"{path}.reactivation_accuracy",
        )
        _validate_ratio(
            row["reactivation_precision"], correct, correct + wrong,
            name=f"{path}.reactivation_precision",
        )
        _validate_ratio(
            row["reactivation_recall"], correct, gap_opportunities,
            name=f"{path}.reactivation_recall",
        )
        _validate_ratio(
            row["reactivation_coverage"], attempts, gap_opportunities,
            name=f"{path}.reactivation_coverage",
        )
    expected = {
        (method, horizon): HORIZON_SEQUENCE_COUNTS[horizon]
        for method in ONLINE_METHOD_SET
        for horizon in HORIZON_IDS
    }
    if counts != expected:
        raise ValueError(f"{path} must contain exactly 129 prefix units per method/horizon")
    for key, unit_rows in units.items():
        master_ids = {unit[1] for unit in unit_rows}
        if len(master_ids) != 43:
            raise ValueError(f"{path} must contain 43 masters for {key}")
        if len({reference for reference, *_ in unit_rows}) != 6:
            raise ValueError(f"{path} must contain six reference-scene clusters for {key}")
        for master in master_ids:
            master_orders = {
                order
                for _, current_master, _, _, order in unit_rows
                if current_master == master
            }
            if master_orders != {"canonical", "reverse", "sha256_seed45"}:
                raise ValueError(f"{path} must contain three orders per master for {key}")


def _validate_error_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    if path.startswith("error_breakdown_T"):
        required_horizon = path[len("error_breakdown_") : -len(".csv")]
        rows_horizons = {
            _horizon_token(row["T"], name=f"{path}.T") for row in rows
        }
        if rows_horizons != {required_horizon}:
            raise ValueError(f"{path} contains rows for the wrong horizon")
    groups: dict[tuple[object, str], list[Mapping[str, object]]] = {}
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        category = row["category"]
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"{path} has an unknown failure category")
        _nonnegative_integer(row["count"], name=f"{path}.count")
        _unit(row["share"], name=f"{path}.share")
        groups.setdefault((row["method"], horizon), []).append(row)
    expected_groups = {
        (method, horizon)
        for method in ONLINE_METHOD_SET
        for horizon in HORIZON_IDS
        if not path.startswith("error_breakdown_T")
        or horizon == path[len("error_breakdown_") : -len(".csv")]
    }
    if set(groups) != expected_groups:
        raise ValueError(f"{path} must cover every method/horizon failure group")
    for group, group_rows in groups.items():
        if {row["category"] for row in group_rows} != set(FAILURE_CATEGORIES):
            raise ValueError(
                f"{path} must contain F1 through F7 and unclassified for {group}"
            )
        total_count = sum(int(row["count"]) for row in group_rows)
        expected_total = 1.0 if total_count else 0.0
        if not math.isclose(
            sum(float(row["share"]) for row in group_rows),
            expected_total,
            abs_tol=1e-9,
        ) or any(
            not math.isclose(
                float(row["share"]),
                int(row["count"]) / total_count if total_count else 0.0,
                abs_tol=1e-9,
            )
            for row in group_rows
        ):
            raise ValueError(f"{path} share does not match count for {group}")


def _validate_reactivation_audit(
    path: str, rows: Sequence[Mapping[str, object]]
) -> None:
    expected = {
        (method, horizon)
        for method in REACTIVATION_METHOD_SET
        for horizon in REACTIVATION_HORIZONS
    }
    actual = {
        (
            _nonempty_string(row["method"], name=f"{path}.method"),
            _horizon_token(row["T"], name=f"{path}.T"),
        )
        for row in rows
    }
    if actual != expected:
        raise ValueError(f"{path} must cover B1-B4 at T3-T5")
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        for field in (
            "gap_opportunities",
            "reactivation_attempts",
            "correct_reactivations",
            "wrong_reactivations",
            "no_attempts",
        ):
            _nonnegative_integer(row[field], name=f"{path}.{field}")
        for field in (
            "reactivation_accuracy",
            "reactivation_precision",
            "reactivation_recall",
            "reactivation_coverage",
        ):
            _unit_or_none(row[field], name=f"{path}.{field}")
        gap_opportunities = int(row["gap_opportunities"])
        attempts = int(row["reactivation_attempts"])
        correct = int(row["correct_reactivations"])
        wrong = int(row["wrong_reactivations"])
        no_attempts = int(row["no_attempts"])
        if attempts > gap_opportunities or correct > attempts:
            raise ValueError(f"{path} contains impossible reactivation counts")
        if no_attempts != gap_opportunities - attempts:
            raise ValueError(f"{path}.no_attempts does not match gap opportunities")
        _validate_ratio(
            row["reactivation_accuracy"], correct, attempts,
            name=f"{path}.reactivation_accuracy",
        )
        _validate_ratio(
            row["reactivation_precision"], correct, correct + wrong,
            name=f"{path}.reactivation_precision",
        )
        _validate_ratio(
            row["reactivation_recall"], correct, gap_opportunities,
            name=f"{path}.reactivation_recall",
        )
        _validate_ratio(
            row["reactivation_coverage"], attempts, gap_opportunities,
            name=f"{path}.reactivation_coverage",
        )


def _validate_reactivation_distribution(
    path: str, rows: Sequence[Mapping[str, object]]
) -> None:
    expected = {
        (method, horizon)
        for method in REACTIVATION_METHOD_SET
        for horizon in REACTIVATION_HORIZONS
    }
    groups: dict[tuple[object, str, object], list[Mapping[str, object]]] = {}
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        if row["method"] not in REACTIVATION_METHOD_SET or horizon not in REACTIVATION_HORIZONS:
            raise ValueError(f"{path} contains an unsupported method or horizon")
        if row["outcome"] not in ("correct", "wrong"):
            raise ValueError(f"{path} outcome must be correct or wrong")
        _finite_number(row["bin_low"], name=f"{path}.bin_low")
        _finite_number(row["bin_high"], name=f"{path}.bin_high")
        if float(row["bin_high"]) <= float(row["bin_low"]):
            raise ValueError(f"{path} bins must be increasing")
        _nonnegative_integer(row["count"], name=f"{path}.count")
        _unit(row["fraction"], name=f"{path}.fraction")
        groups.setdefault((row["method"], horizon, row["outcome"]), []).append(row)
    if {(method, horizon) for method, horizon, _ in groups} != expected:
        raise ValueError(f"{path} must cover B1-B4 at T3-T5")
    for group, group_rows in groups.items():
        ordered = sorted(group_rows, key=lambda row: (row["bin_low"], row["bin_high"]))
        total_count = sum(int(row["count"]) for row in ordered)
        expected_total = 1.0 if total_count else 0.0
        if not math.isclose(
            sum(float(row["fraction"]) for row in ordered),
            expected_total,
            abs_tol=1e-9,
        ) or any(
            not math.isclose(
                float(row["fraction"]),
                int(row["count"]) / total_count if total_count else 0.0,
                abs_tol=1e-9,
            )
            for row in ordered
        ):
            raise ValueError(f"{path} fractions do not match counts for {group}")
        for previous, current in pairwise(ordered):
            if previous["bin_high"] != current["bin_low"]:
                raise ValueError(f"{path} bins must be continuous")
    if {key[:2] for key in groups} != expected:
        raise ValueError(f"{path} must contain paired correct/wrong groups")
    if any(
        {outcome for method, horizon, outcome in groups if (method, horizon) == pair}
        != {"correct", "wrong"}
        for pair in expected
    ):
        raise ValueError(f"{path} must contain paired correct/wrong groups")


def _validate_reactivation_by_gap(
    path: str, rows: Sequence[Mapping[str, object]]
) -> None:
    expected = {
        (method, horizon)
        for method in REACTIVATION_METHOD_SET
        for horizon in REACTIVATION_HORIZONS
    }
    groups: dict[tuple[object, str, object], list[Mapping[str, object]]] = {}
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        if row["method"] not in REACTIVATION_METHOD_SET or horizon not in REACTIVATION_HORIZONS:
            raise ValueError(f"{path} contains an unsupported method or horizon")
        _nonnegative_or_none(row["gap_length"], name=f"{path}.gap_length")
        if row["outcome"] not in ("correct", "wrong"):
            raise ValueError(f"{path} outcome must be correct or wrong")
        _nonnegative_integer(row["count"], name=f"{path}.count")
        _unit(row["fraction"], name=f"{path}.fraction")
        groups.setdefault((row["method"], horizon, row["gap_length"]), []).append(row)
    if {(method, horizon) for method, horizon, _ in groups} != expected:
        raise ValueError(f"{path} must cover B1-B4 at T3-T5")
    for group, group_rows in groups.items():
        if {row["outcome"] for row in group_rows} != {"correct", "wrong"}:
            raise ValueError(f"{path} must contain paired outcomes for {group}")
        total_count = sum(int(row["count"]) for row in group_rows)
        expected_total = 1.0 if total_count else 0.0
        if not math.isclose(
            sum(float(row["fraction"]) for row in group_rows),
            expected_total,
            abs_tol=1e-9,
        ) or any(
            not math.isclose(
                float(row["fraction"]),
                int(row["count"]) / total_count if total_count else 0.0,
                abs_tol=1e-9,
            )
            for row in group_rows
        ):
            raise ValueError(f"{path} fractions do not match counts for {group}")


def _validate_capacity_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    stages: dict[tuple[object, str], set[int]] = {}
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        if row["method"] != CAPACITY_METHOD:
            raise ValueError(f"{path} only supports bounded state for {CAPACITY_METHOD}")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        stage = _integer(row["stage_id"], name=f"{path}.stage_id", minimum=0)
        if stage > 4:
            raise ValueError(f"{path}.stage_id must be between 0 and 4")
        if stage >= int(horizon[1:]):
            raise ValueError(f"{path}.stage_id must fall within its causal prefix")
        key = (row["method"], horizon)
        stages.setdefault(key, set()).add(stage)
        capacity = _integer(row["capacity"], name=f"{path}.capacity", minimum=1)
        occupied = _integer(row["occupied_count"], name=f"{path}.occupied_count", minimum=0)
        active = _integer(row["active_count"], name=f"{path}.active_count", minimum=0)
        dormant = _integer(row["dormant_count"], name=f"{path}.dormant_count", minimum=0)
        if occupied > capacity or active > occupied or dormant != occupied - active:
            raise ValueError(f"{path} has inconsistent capacity state counts")
        for field in (
            "birth_count",
            "peak_occupied",
            "peak_active",
            "peak_dormant",
            "rejected_births",
            "persistent_state_bytes",
        ):
            _nonnegative_integer(row[field], name=f"{path}.{field}")
        _unit(row["occupancy_ratio"], name=f"{path}.occupancy_ratio")
        if row["occupancy_ratio"] is not None and not math.isclose(
            float(row["occupancy_ratio"]), occupied / capacity, abs_tol=1e-9
        ):
            raise ValueError(f"{path}.occupancy_ratio does not match occupied/capacity")
    expected = {
        (CAPACITY_METHOD, horizon): set(range(int(horizon[1:])))
        for horizon in HORIZON_IDS
    }
    if stages != expected:
        raise ValueError(f"{path} must cover every stage in each causal prefix")


def _validate_efficiency_rows(path: str, rows: Sequence[Mapping[str, object]]) -> None:
    row_types = {"bootstrap", "new_visit", "full_history"}
    seen: set[tuple[object, object, object, object]] = set()
    observed_types: set[object] = set()
    for row in rows:
        _nonempty_string(row["method"], name=f"{path}.method")
        horizon = _horizon_token(row["T"], name=f"{path}.T")
        stage_id = _integer(row["stage_id"], name=f"{path}.stage_id", minimum=0)
        if stage_id > 4:
            raise ValueError(f"{path}.stage_id must be between 0 and 4")
        _nonempty_string(row["row_type"], name=f"{path}.row_type")
        if row["row_type"] not in row_types:
            raise ValueError(f"{path} has an unsupported row_type")
        if row["method"] not in METHOD_IDS and row["method"] != "full_history_rescene":
            raise ValueError(f"{path} contains an unsupported method")
        key = (row["method"], horizon, row["stage_id"], row["row_type"])
        if key in seen:
            raise ValueError(f"{path} contains duplicate stage timing rows")
        seen.add(key)
        observed_types.add(row["row_type"])
        _nonnegative_integer(row["count"], name=f"{path}.count")
        for field in (
            "bootstrap_latency_ms",
            "new_visit_latency_ms",
            "association_overhead_ms",
            "memory_update_overhead_ms",
            "full_history_latency_ms",
        ):
            _finite_number(row[field], name=f"{path}.{field}", allow_none=True)
            if row[field] is not None and float(row[field]) < 0:
                raise ValueError(f"{path}.{field} must be non-negative")
        for field in ("gpu_peak_memory_bytes", "persistent_state_bytes"):
            _nonnegative_or_none(row[field], name=f"{path}.{field}")
        bootstrap = row["bootstrap_latency_ms"]
        new_visit = row["new_visit_latency_ms"]
        association = row["association_overhead_ms"]
        memory_update = row["memory_update_overhead_ms"]
        full_history = row["full_history_latency_ms"]
        if row["row_type"] == "bootstrap":
            if bootstrap is None or any(
                value is not None
                for value in (new_visit, association, memory_update, full_history)
            ):
                raise ValueError(f"{path} bootstrap rows cannot contain visit metrics")
        elif row["row_type"] == "new_visit":
            if new_visit is None or bootstrap is not None or full_history is not None:
                raise ValueError(f"{path} new_visit rows cannot contain setup/full-history metrics")
        elif full_history is None or any(
            value is not None
            for value in (bootstrap, new_visit, association, memory_update)
        ):
            raise ValueError(f"{path} full_history rows cannot contain setup/update metrics")
    if observed_types != row_types:
        raise ValueError(f"{path} must contain bootstrap, new_visit, and full_history rows")


def _validate_association_event_rows(
    path: str, rows: Sequence[Mapping[str, object]]
) -> None:
    for row in rows:
        for field in (
            "event_id",
            "scene_id",
            "sequence_id",
            "reference_scene_id",
            "master_sequence_id",
            "order_id",
            "method",
            "event_kind",
            "prediction_digest",
            "cache_digest",
        ):
            _nonempty_string(row[field], name=f"{path}.{field}")
        _integer(row["prefix"], name=f"{path}.prefix", minimum=0)
        _integer(row["stage_id"], name=f"{path}.stage_id", minimum=0)
        _validate_sha(row["prediction_digest"], name=f"{path}.prediction_digest", length=64)
        _validate_sha(row["cache_digest"], name=f"{path}.cache_digest", length=64)
        if row["prediction_digest"] != row["cache_digest"]:
            raise ValueError(f"{path} prediction/cache digests must agree")
        if not isinstance(row["is_failure"], bool):
            raise ValueError(f"{path}.is_failure must be boolean")  # noqa: TRY004
        for field in (
            "association_correct",
            "slot_active",
            "slot_occupied",
            "new_birth",
            "reactivation",
            "reactivation_correct",
            "transition_opportunity",
            "id_switch",
            "gap_opportunity",
            "reactivation_attempt",
            "false_birth",
            "wrong_reactivation",
        ):
            if row[field] is not None and not isinstance(row[field], bool):
                raise ValueError(f"{path}.{field} must be boolean or null")
        for field in (
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
            _finite_number(row[field], name=f"{path}.{field}", allow_none=True)
        for field in ("slot_age", "last_seen_stage", "gap_length"):
            _nonnegative_or_none(row[field], name=f"{path}.{field}")
    try:
        from scripts.p6a_analysis import validate_association_events

        validate_association_events(rows)
    except Exception as error:
        raise ValueError(f"{path} failed association-event reconstruction validation") from error


def _validate_registered_csv(path: str, columns: object, rows: object) -> None:
    validated = _csv_rows_with_schema(path, columns, rows)
    if path in {"baseline_results.csv", "strict_online_results.csv", "raw_local_results.csv"}:
        _validate_metric_table(path, validated)
    elif path == "per_sequence_results.csv":
        _validate_per_sequence_rows(path, validated)
    elif path == "association_events.csv":
        _validate_association_event_rows(path, validated)
    elif path == "error_breakdown.csv" or path.startswith("error_breakdown_T"):
        _validate_error_rows(path, validated)
    elif path == "reactivation_audit.csv":
        _validate_reactivation_audit(path, validated)
    elif path in {
        "reactivation_score_distribution.csv",
        "reactivation_margin_distribution.csv",
    }:
        _validate_reactivation_distribution(path, validated)
    elif path == "reactivation_by_gap.csv":
        _validate_reactivation_by_gap(path, validated)
    elif path == "capacity_audit.csv":
        _validate_capacity_rows(path, validated)
    elif path == "efficiency_results.csv":
        _validate_efficiency_rows(path, validated)


def _validate_metric_record(value: object, *, name: str) -> None:
    record = _exact_keys(value, frozenset(METRIC_FIELDS), name=name)
    available = False
    for field in METRIC_FIELDS:
        item = record[field]
        _finite_number(item, name=f"{name}.{field}", allow_none=True)
        if item is not None:
            available = True
            if not 0.0 <= float(item) <= 1.0:
                raise ValueError(f"{name}.{field} must be within [0, 1]")
    if not available:
        raise ValueError(f"{name} must contain at least one metric")


def _validate_metric_blocks(value: object) -> None:
    blocks = _exact_keys(value, frozenset(METRIC_BLOCK_IDS), name="metric_blocks")
    for block_name in METRIC_BLOCK_IDS:
        expected_methods = ONLINE_METHOD_IDS if block_name != "offline" else METHOD_IDS
        methods = _exact_keys(
            blocks[block_name], frozenset(expected_methods), name=f"metric_blocks.{block_name}"
        )
        for method in expected_methods:
            horizons = _exact_keys(
                methods[method], frozenset(HORIZON_IDS),
                name=f"metric_blocks.{block_name}.{method}",
            )
            for horizon in HORIZON_IDS:
                _validate_metric_record(
                    horizons[horizon],
                    name=f"metric_blocks.{block_name}.{method}.{horizon}",
                )


def _validate_fingerprints(value: object) -> None:
    fingerprints = _exact_keys(value, frozenset({"prediction", "cache"}), name="fingerprints")
    for kind in ("prediction", "cache"):
        records = _exact_keys(
            fingerprints[kind], frozenset(METHOD_IDS), name=f"fingerprints.{kind}"
        )
        values = []
        for method in METHOD_IDS:
            values.append(
                _validate_sha(
                    records[method], name=f"fingerprints.{kind}.{method}", length=64
                )
            )
        if len(set(values)) != 1:
            raise ValueError(f"fingerprints.{kind} must bind one shared frozen value")


def _validate_derived_artifacts(value: object) -> None:
    derived = _exact_keys(value, frozenset(DERIVED_KIND_IDS), name="derived_artifacts")
    expected_paths = {
        "csv": REQUIRED_CSV_PATHS,
        "json": REQUIRED_JSON_PATHS,
        "markdown": REQUIRED_MARKDOWN_PATHS,
        "svg": REQUIRED_SVG_PATHS,
        "yaml": REQUIRED_YAML_PATHS,
    }
    for kind in DERIVED_KIND_IDS:
        records = derived[kind]
        if not isinstance(records, Mapping) or not records:
            raise ValueError(f"derived_artifacts.{kind} must not be empty")
        actual_paths = set(records)
        if actual_paths != set(expected_paths[kind]):
            raise ValueError(
                f"derived_artifacts.{kind} paths differ: "
                f"missing={sorted(set(expected_paths[kind]) - actual_paths)}, "
                f"extra={sorted(actual_paths - set(expected_paths[kind]))}"
            )
        suffix = {
            "csv": ".csv",
            "json": ".json",
            "markdown": ".md",
            "svg": ".svg",
            "yaml": ".yaml",
        }[kind]
        for raw_path, record in records.items():
            path = _validate_relative_artifact_path(
                raw_path, name=f"derived_artifacts.{kind} path"
            )
            if not path.endswith(suffix):
                raise ValueError(f"{path} has the wrong derived artifact kind")
            expected_keys = DERIVED_RECORD_KEYS[kind]
            normalized = _exact_keys(record, expected_keys, name=f"derived_artifacts.{path}")
            if kind == "csv":
                _validate_registered_csv(path, normalized["columns"], normalized["rows"])
            else:
                text = _nonempty_string(normalized["text"], name=f"{path}.text")
                if kind == "svg" and not text.lstrip().startswith("<svg"):
                    raise ValueError(f"{path} must contain an SVG root")
                if kind == "json":
                    try:
                        parsed = json.loads(text)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path} must contain valid JSON") from error
                    if path == "protocol_b_manifest.json":
                        try:
                            from scripts.p6a_protocol import (
                                validate_protocol_b_manifest,
                            )

                            validate_protocol_b_manifest(parsed)
                        except Exception as error:
                            raise ValueError(
                                f"{path} failed Protocol B manifest validation"
                            ) from error
                if kind == "yaml":
                    try:
                        import yaml

                        parsed_yaml = yaml.safe_load(text)
                    except Exception as error:
                        raise ValueError(f"{path} must contain safe YAML") from error
                    if not isinstance(parsed_yaml, Mapping):
                        raise ValueError(f"{path} must contain a YAML mapping")
                    _validate_scalar_tree(parsed_yaml, path=f"{path}.parsed")


def _validate_efficiency_aggregate_binding(
    derived: Mapping[str, object],
    root: Mapping[str, object],
) -> None:
    """Require the registered efficiency CSV to be an exact raw-manifest derivation."""

    raw_spec = derived["json"]["efficiency_raw_manifest.json"]
    csv_spec = derived["csv"]["efficiency_results.csv"]
    if not isinstance(raw_spec, Mapping) or not isinstance(csv_spec, Mapping):
        raise ValueError("efficiency derived artifacts must be mappings")  # noqa: TRY004
    try:
        raw_manifest = json.loads(raw_spec["text"])
        from scripts.p6a_efficiency import (
            aggregate_efficiency_rows,
            validate_efficiency_manifest,
        )

        validate_efficiency_manifest(raw_manifest)
        expected_rows = list(aggregate_efficiency_rows(raw_manifest))
    except Exception as error:
        raise ValueError("efficiency raw manifest cannot be aggregated") from error

    expected_columns = list(CSV_COLUMN_SCHEMAS["efficiency_results.csv"])
    if csv_spec["columns"] != expected_columns:
        raise ValueError("efficiency_results.csv columns are not the registered schema")
    actual_rows = csv_spec["rows"]
    if not isinstance(actual_rows, list) or len(actual_rows) != len(expected_rows):
        raise ValueError("efficiency_results.csv rows differ from raw manifest aggregation")
    for row_index, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
        if not isinstance(actual, Mapping) or tuple(actual) != tuple(expected):
            raise ValueError(
                f"efficiency_results.csv row {row_index} columns differ from raw manifest"
            )
        for field in expected_columns:
            actual_value = actual[field]
            expected_value = expected[field]
            if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                raise ValueError(
                    "efficiency_results.csv value differs from raw manifest "
                    f"at row {row_index}, column {field}"
                )

    yaml_specs = derived["yaml"]
    protocol_spec = derived["json"]["protocol_b_manifest.json"]
    provenance = root["provenance"]
    config_hasher = hashlib.sha256()
    for name, path in (
        ("p6a", "configs/p6a_default.yaml"),
        ("runtime", "configs/resolved_runtime.yaml"),
    ):
        content = yaml_specs[path]["text"].encode("utf-8")
        config_hasher.update(name.encode("utf-8") + b"\0")
        config_hasher.update(len(content).to_bytes(8, "big") + content)
    expected_provenance = {
        "source_commit": root["source_commit"],
        "checkpoint_sha256": provenance["checkpoint"]["sha256"],
        "config_sha256": config_hasher.hexdigest(),
        "protocol_sha256": hashlib.sha256(
            protocol_spec["text"].encode("utf-8")
        ).hexdigest(),
        "cache_manifest_sha256": provenance["prediction_cache"]["sha256"],
    }
    if raw_manifest["provenance"] != expected_provenance:
        raise ValueError("efficiency raw provenance differs from root evidence")


def _validate_manifest(value: object, *, derived: Mapping[str, object]) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("artifact_manifest must be a non-empty list")
    paths: list[str] = []
    for index, entry in enumerate(value):
        record = _exact_keys(entry, MANIFEST_RECORD_KEYS, name=f"artifact_manifest[{index}]")
        path = _validate_relative_artifact_path(
            record["path"], name=f"artifact_manifest[{index}].path"
        )
        _integer(record["bytes"], name=f"artifact_manifest[{index}].bytes", minimum=1)
        _validate_sha(record["sha256"], name=f"artifact_manifest[{index}].sha256", length=64)
        paths.append(path)
    if len(set(paths)) != len(paths) or paths != sorted(paths):
        raise ValueError("artifact_manifest paths must be unique and sorted")
    derived_paths = {
        path
        for kind in DERIVED_KIND_IDS
        for path in derived[kind]
    }
    expected_paths = derived_paths | {REPORT_PATH}
    if set(paths) != expected_paths:
        raise ValueError(
            f"artifact_manifest paths differ: missing={sorted(expected_paths - set(paths))}, "
            f"extra={sorted(set(paths) - expected_paths)}"
        )


def validate_root_artifact(artifact: object) -> None:
    """Validate the complete P6-A root contract and reject schema drift."""

    root = _exact_keys(artifact, ROOT_KEYS, name="artifact")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported P6-A schema_version")
    if root["status"] != "pass":
        raise ValueError("complete P6-A artifact status must be pass")
    _nonempty_string(root["run_id"], name="run_id")
    source_commit = _validate_sha(root["source_commit"], name="source_commit", length=40)

    source_tree = _exact_keys(root["source_tree_contract"], SOURCE_TREE_KEYS, name="source_tree_contract")
    if source_tree["status"] != "pass" or source_tree["source_commit"] != source_commit:
        raise ValueError("source_tree_contract must bind the passing source commit")

    frozen = _exact_keys(root["p5_frozen_hashes"], P5_FROZEN_KEYS, name="p5_frozen_hashes")
    for key, length in (
        ("source_commit", 40),
        ("artifact_commit", 40),
        ("checkpoint_sha256", 64),
        ("json_sha256", 64),
        ("markdown_sha256", 64),
    ):
        _validate_sha(frozen[key], name=f"p5_frozen_hashes.{key}", length=length)
    if dict(frozen) != P5_FROZEN_VALUES:
        raise ValueError("p5_frozen_hashes does not match the frozen P5 contract")
    frozen_checkpoint = frozen["checkpoint_sha256"]

    protocol = _exact_keys(root["protocol"], PROTOCOL_KEYS, name="protocol")
    if protocol["name"] != "exact_common_prefix_protocol_b":
        raise ValueError("protocol name must identify exact common-prefix Protocol B")
    if protocol["horizons"] != [2, 3, 4, 5]:
        raise ValueError("protocol horizons must be exactly [2, 3, 4, 5]")
    for field, expected in (
        ("master_sequence_count", 43),
        ("cluster_count", 6),
        ("order_count", 3),
        ("cache_entry_count", 645),
    ):
        if _integer(protocol[field], name=f"protocol.{field}", minimum=1) != expected:
            raise ValueError(f"protocol.{field} must be exactly {expected}")

    provenance = _exact_keys(root["provenance"], PROVENANCE_KEYS, name="provenance")
    expected_reference_prefix = {
        "checkpoint": "repo:checkpoints/",
        "config": "repo:conf/",
        "dataset": "repo:data/",
        "prediction_cache": "local_cache:",
    }
    expected_digest = {"checkpoint": frozen_checkpoint}
    for key in sorted(PROVENANCE_KEYS):
        record = _exact_keys(provenance[key], PROVENANCE_RECORD_KEYS, name=f"provenance.{key}")
        _validate_reference(
            record["ref"], name=f"provenance.{key}.ref", expected_prefix=expected_reference_prefix[key]
        )
        digest = _validate_sha(record["sha256"], name=f"provenance.{key}.sha256", length=64)
        if key in expected_digest and digest != expected_digest[key]:
            raise ValueError(f"provenance.{key}.sha256 must match P5 frozen hash")

    settings = _exact_keys(
        root["settings"], frozenset({"bootstrap_seed", "bootstrap_replicates"}), name="settings"
    )
    _integer(settings["bootstrap_seed"], name="settings.bootstrap_seed", minimum=0)
    _integer(settings["bootstrap_replicates"], name="settings.bootstrap_replicates", minimum=1)

    methods = _exact_keys(root["methods"], frozenset({"set", "oracle"}), name="methods")
    if methods["set"] != list(METHOD_IDS):
        raise ValueError("methods.set must be exactly B0, B0_sanity, B1-B4, Oracle")
    oracle = _exact_keys(methods["oracle"], METHOD_RECORD_KEYS, name="methods.oracle")
    if oracle["mode"] != "offline" or oracle["metric_block"] != "offline":
        raise ValueError("Oracle is allowed only in the offline metric block")

    horizons = _exact_keys(root["horizons"], frozenset(HORIZON_IDS), name="horizons")
    for horizon in HORIZON_IDS:
        record = _exact_keys(horizons[horizon], frozenset({"sequence_count"}), name=f"horizons.{horizon}")
        expected = HORIZON_SEQUENCE_COUNTS[horizon]
        if _integer(record["sequence_count"], name=f"horizons.{horizon}.sequence_count", minimum=1) != expected:
            raise ValueError(f"horizons.{horizon}.sequence_count must be exactly {expected}")

    _validate_metric_blocks(root["metric_blocks"])
    _validate_fingerprints(root["fingerprints"])

    analysis = _exact_keys(root["analysis"], frozenset(ANALYSIS_GROUPS), name="analysis")
    expected_analysis_paths = {
        "association": "association_events.csv",
        "error": "error_breakdown.csv",
        "reactivation": "reactivation_audit.csv",
        "capacity": "capacity_audit.csv",
        "efficiency": "efficiency_results.csv",
        "statistical": "statistical_analysis.md",
    }
    for group in ANALYSIS_GROUPS:
        record = _exact_keys(analysis[group], ANALYSIS_RECORD_KEYS, name=f"analysis.{group}")
        if record["status"] != "pass" or record["path"] != expected_analysis_paths[group]:
            raise ValueError(f"analysis.{group} must bind its required passing artifact")
        _integer(record["rows"], name=f"analysis.{group}.rows", minimum=1)

    limitation = _exact_keys(root["change_label_limitation"], CHANGE_LABEL_KEYS, name="change_label_limitation")
    if limitation["available"] is not False:
        raise ValueError("change_label_limitation.available must be false for P6-A")
    _nonempty_string(limitation["reason"], name="change_label_limitation.reason")
    _nonempty_string(limitation["scope"], name="change_label_limitation.scope")

    _validate_derived_artifacts(root["derived_artifacts"])
    _validate_efficiency_aggregate_binding(root["derived_artifacts"], root)
    _validate_manifest(root["artifact_manifest"], derived=root["derived_artifacts"])

    analysis_artifacts = root["derived_artifacts"]
    analysis_path_to_rows = {
        path: len(spec["rows"])
        for path, spec in analysis_artifacts["csv"].items()
    }
    analysis_path_to_rows.update(
        {path: 1 for kind in ("json", "markdown", "svg", "yaml") for path in analysis_artifacts[kind]}
    )
    for group in ANALYSIS_GROUPS:
        record = analysis[group]
        expected_rows = analysis_path_to_rows.get(record["path"])
        if expected_rows is None or record["rows"] != expected_rows:
            raise ValueError(f"analysis.{group}.rows does not match its derived artifact")

    gates = _exact_keys(root["gate_results"], frozenset(GATE_IDS), name="gate_results")
    for gate_id in GATE_IDS:
        gate = _exact_keys(gates[gate_id], GATE_RECORD_KEYS, name=gate_id)
        if not isinstance(gate["passed"], bool):
            raise ValueError(f"{gate_id}.passed must be boolean")  # noqa: TRY004
        _nonempty_string(gate["evidence"], name=f"{gate_id}.evidence")
    _string_list(root["claims_supported"], name="claims_supported")
    _string_list(root["claims_not_supported"], name="claims_not_supported")
    next_action = _nonempty_string(root["next_action"], name="next_action")
    if "p6b" in next_action.casefold() or "enter" in next_action.casefold() and "p6a" not in next_action.casefold():
        raise ValueError("next_action must stop at P6-A and require explicit continuation")
    _string_list(root["errors"], name="errors")
    _validate_scalar_tree(root)


def artifact_json_text(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    return json.dumps(artifact, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _markdown_list(values: object) -> str:
    items = _string_list(values, name="report list")
    if not items:
        return "None."
    return "\n".join(f"- {item}" for item in items)


def render_go_nogo_report(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    gates = artifact["gate_results"]
    decision = "P6A_GO" if all(gates[key]["passed"] for key in GATE_IDS) else "P6A_STOP"
    gate_lines = "\n".join(
        f"- {key}: {'PASS' if gates[key]['passed'] else 'FAIL'} - {gates[key]['evidence']}"
        for key in GATE_IDS
    )
    protocol = artifact["protocol"]
    p5 = artifact["p5_frozen_hashes"]
    analysis = artifact["analysis"]
    sections = {
        "What was changed": "Implemented the P6-A scientific evidence package with a frozen root payload.",
        "Why it was changed": "To isolate cross-stage association and state maintenance from frozen local perception.",
        "Experimental protocol": (
            f"Protocol B uses exactly {protocol['master_sequence_count']} masters, "
            f"{protocol['cluster_count']} reference-scene clusters, {protocol['order_count']} orders, "
            f"and {protocol['cache_entry_count']} cache entries at T=2/3/4/5."
        ),
        "Reproducibility binding": (
            f"P6-A source commit: `{artifact['source_commit']}`; P5 source commit: "
            f"`{p5['source_commit']}`; P5 artifact commit: `{p5['artifact_commit']}`; "
            f"P5 checkpoint SHA256: `{p5['checkpoint_sha256']}`."
        ),
        "Main results": gate_lines,
        "Statistical evidence": f"See `{analysis['statistical']['path']}`.",
        "Failure analysis": (
            f"Association: `{analysis['association']['path']}`; error: `{analysis['error']['path']}`; "
            f"reactivation: `{analysis['reactivation']['path']}`."
        ),
        "What claims are supported": _markdown_list(artifact["claims_supported"]),
        "What claims are NOT supported": _markdown_list(artifact["claims_not_supported"]),
        "GO / NO-GO decision": f"Decision: {decision}",
        "Exact next action": f"Exact next action: {artifact['next_action']}",
    }
    body = ["# Persist4D P6-A GO / NO-GO Report"]
    for section in P6A_REPORT_SECTIONS:
        body.extend(("", f"## {section}", "", sections[section]))
    return "\n".join(body) + "\n"


def render_csv(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]) -> str:
    normalized_columns = tuple(columns)
    if not normalized_columns or len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("CSV columns must be non-empty and unique")
    if any(not isinstance(column, str) or not column for column in normalized_columns):
        raise ValueError("CSV columns must be non-empty strings")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=normalized_columns, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    expected = set(normalized_columns)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"CSV row {index} does not match the exact columns")
        _validate_scalar_tree(row, path=f"rows[{index}]")
        writer.writerow(row)
    return output.getvalue()


def render_artifact_bundle(artifact: Mapping[str, object]) -> dict[str, bytes]:
    """Render every P6-A derived artifact from one validated root payload."""

    validate_root_artifact(artifact)
    derived = artifact["derived_artifacts"]
    rendered: dict[str, bytes] = {
        ROOT_ARTIFACT_PATH: artifact_json_text(artifact).encode("utf-8"),
        REPORT_PATH: render_go_nogo_report(artifact).encode("utf-8"),
    }
    for path, spec in sorted(derived["csv"].items()):
        rendered[path] = render_csv(spec["rows"], columns=spec["columns"]).encode("utf-8")
    for kind in ("json", "markdown", "svg", "yaml"):
        for path, spec in sorted(derived[kind].items()):
            rendered[path] = spec["text"].encode("utf-8")
    return dict(sorted(rendered.items()))


def _normalized_files(files: Mapping[str, str | bytes]) -> list[tuple[PurePosixPath, bytes]]:
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files must be a non-empty mapping")
    normalized: list[tuple[PurePosixPath, bytes]] = []
    for raw_path, content in files.items():
        relative = PurePosixPath(_validate_relative_artifact_path(raw_path, name="artifact output path"))
        if not isinstance(content, (str, bytes)):
            raise ValueError("artifact content must be text or bytes")  # noqa: TRY004
        payload = content.encode("utf-8") if isinstance(content, str) else content
        if not payload:
            raise ValueError(f"artifact {relative.as_posix()} must not be empty")
        normalized.append((relative, payload))
    normalized.sort(key=lambda item: item[0].as_posix())
    if len({path for path, _ in normalized}) != len(normalized):
        raise ValueError("artifact output paths must be unique")
    return normalized


def _check_existing_path_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"output path contains a symlink: {component}")
        if component.exists() and not component.is_dir():
            raise ValueError(f"output path component is not a directory: {component}")


def _validate_output_root(output_root: Path, targets: Sequence[Path]) -> None:
    if not isinstance(output_root, Path):
        raise ValueError("output_root must be a Path")  # noqa: TRY004
    _check_existing_path_components(output_root.parent)
    if output_root.is_symlink():
        raise ValueError("output_root must not be a symlink")
    if output_root.exists():
        raise FileExistsError("output_root must not already exist for atomic publication")
    for target in targets:
        _check_existing_path_components(target.parent)
        if target.is_symlink():
            raise ValueError(f"artifact output must not be a symlink: {target}")


def _manifest_map(artifact: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {entry["path"]: entry for entry in artifact["artifact_manifest"]}


def verify_artifact_manifest(
    artifact: Mapping[str, object], files: Mapping[str, str | bytes]
) -> bool:
    """Re-render and verify exact bytes, lengths, and SHA256 values."""

    validate_root_artifact(artifact)
    normalized = _normalized_files(files)
    actual = {path.as_posix(): payload for path, payload in normalized}
    rendered = render_artifact_bundle(artifact)
    if actual != rendered:
        missing = sorted(set(rendered) - set(actual))
        extra = sorted(set(actual) - set(rendered))
        raise ValueError(f"rendered files differ: missing={missing}, extra={extra}")
    manifest = _manifest_map(artifact)
    for path, payload in sorted(actual.items()):
        if path == ROOT_ARTIFACT_PATH:
            if payload != artifact_json_text(artifact).encode("utf-8"):
                raise ValueError("root JSON bytes differ from artifact_json_text")
            continue
        record = manifest[path]
        expected_digest = hashlib.sha256(payload).hexdigest()
        if record["bytes"] != len(payload) or record["sha256"] != expected_digest:
            raise ValueError(f"artifact manifest mismatch for {path}")
    return True


def publish_artifacts(
    output_root: Path,
    files: Mapping[str, str | bytes],
    *,
    artifact: Mapping[str, object] | None = None,
) -> list[Path]:
    """Stage a complete bundle and expose it with one directory replacement."""

    if artifact is None and isinstance(files, Mapping) and "schema_version" in files and "artifact_manifest" in files:
        return publish_root_artifact(output_root, files)  # type: ignore[arg-type]
    normalized = _normalized_files(files)
    targets = [output_root.joinpath(*relative.parts) for relative, _ in normalized]
    _validate_output_root(output_root, targets)

    if artifact is not None:
        rendered = render_artifact_bundle(artifact)
        verify_artifact_manifest(artifact, rendered)
        rendered_normalized = _normalized_files(rendered)
        if rendered_normalized != normalized:
            raise ValueError("supplied files do not equal root-payload rendering")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    _check_existing_path_components(output_root.parent)
    stage_root = Path(tempfile.mkdtemp(prefix=".p6a-stage-", dir=output_root.parent))
    staged: list[tuple[Path, Path]] = []
    try:
        for relative, payload in normalized:
            stage_target = stage_root.joinpath(*relative.parts)
            stage_target.parent.mkdir(parents=True, exist_ok=True)
            if stage_target.is_symlink() or not stage_target.parent.is_dir():
                raise ValueError("staging output is not a regular file path")
            stage_target.write_bytes(payload)
            if stage_target.is_symlink() or not stage_target.is_file():
                raise ValueError("staged artifact is not a regular file")
            staged.append((stage_target, output_root.joinpath(*relative.parts)))

        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError("output_root appeared during atomic publication")
        os.replace(stage_root, output_root)
        stage_root = None
        return [target for _, target in staged]
    finally:
        if stage_root is not None:
            shutil.rmtree(stage_root, ignore_errors=True)


def publish_root_artifact(output_root: Path, artifact: Mapping[str, object]) -> list[Path]:
    """Render and publish a complete root artifact as one verified bundle."""

    rendered = render_artifact_bundle(artifact)
    verify_artifact_manifest(artifact, rendered)
    return publish_artifacts(output_root, rendered, artifact=artifact)
