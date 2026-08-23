"""Pure evidence contracts and Gate II analysis for horizon adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.reviewer_closure_analysis import (
    IDENTITY_COUNT_FIELDS,
    IDENTITY_RATE_FIELDS,
    TASK_FIELDS,
    aggregate_identity_metrics,
)

HORIZONS = (2, 3, 4, 5)
GATE_HORIZONS = (4, 5)
ORDERS = ("canonical", "reverse", "sha256_seed45")
STATISTICAL_METRICS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_REC",
    "normalized_id_switch_rate",
    "gap_recovery_recall",
)
TASK_STATISTICAL_METRICS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_REC",
)
IDENTITY_STATISTICAL_METRICS = (
    "normalized_id_switch_rate",
    "gap_recovery_recall",
)
PHASE_II_METHOD_IDS = (
    "FullHistoryFrozenNative",
    "FullHistoryFrozenB2",
    "FullHistoryAdaptedNative",
    "FullHistoryAdaptedB2",
    "Persist4D",
)
PHASE_II_METHOD_NAMES = {
    "FullHistoryFrozenNative": "ReScene4D T2-Frozen (Native IDs)",
    "FullHistoryFrozenB2": "ReScene4D T2-Frozen + B2",
    "FullHistoryAdaptedNative": (
        "ReScene4D T2-to-T3 Horizon-Adapted (Native IDs)"
    ),
    "FullHistoryAdaptedB2": "ReScene4D T2-to-T3 Horizon-Adapted + B2",
    "Persist4D": "Persist4D Persistent Entity State",
}
PHASE_II_COMPUTE_METHOD_NAMES = {
    "FullHistoryFrozen": "ReScene4D Full-History (Frozen T2 Checkpoint)",
    "FullHistoryAdapted": "ReScene4D T2-to-T3 Horizon-Adapted",
    "Persist4D": "Persist4D Persistent Entity State",
}
_FROZEN_METHOD_MAP = {
    "FullHistoryNative": "FullHistoryFrozenNative",
    "B2": "FullHistoryFrozenB2",
    "Persist4D": "Persist4D",
}
_ADAPTED_METHOD_MAP = {
    "FullHistoryNative": "FullHistoryAdaptedNative",
    "B2": "FullHistoryAdaptedB2",
}


class AdaptationEvidenceError(ValueError):
    """Raised when Phase II evidence is incomplete or internally inconsistent."""


def _expected_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stage_name": "persist4d-reviewer-closure-phase-ii-evaluation",
        "paper_name": "ReScene4D T2-to-T3 Horizon-Adapted",
        "strongest_simple_tracker_method_id": "B2",
        "evaluation_horizons": [2, 3, 4, 5],
        "task_metrics": [
            "causal_prefix_t_mAP",
            "causal_prefix_t_mAP50",
            "causal_prefix_t_mAP25",
            "causal_prefix_t_REC",
            "current_stage_AP",
            "current_stage_REC",
        ],
        "identity_metrics": [
            "normalized_id_switch_rate",
            "fragmentation_rate",
            "merge_rate",
            "gap_recovery_accuracy",
            "gap_recovery_recall",
        ],
        "statistics": {
            "cluster_unit": "reference_scene_id",
            "cluster_count": 6,
            "bootstrap_replicates": 10_000,
            "seed": 45,
            "confidence_level": 0.95,
        },
        "gate_ii": {
            "substantial_task_advantage": {
                "compared_horizons": [4, 5],
                "required_metrics": [
                    "causal_prefix_t_mAP",
                    "causal_prefix_t_REC",
                ],
                "minimum_pooled_absolute_difference": 0.01,
                "require_cluster_ci_lower_above_zero": True,
                "require_all_orders_positive": True,
                "require_all_loso_positive": True,
            },
            "identity_gap_closed": {
                "compared_horizons": [4, 5],
                "required_metrics": [
                    "normalized_id_switch_rate",
                    "gap_recovery_recall",
                ],
                "rule": "no_robust_persist4d_advantage_in_any_required_cell",
            },
            "compute_no_material_disadvantage": {
                "compared_horizons": [4, 5],
                "required_ratios": [
                    "latency_ratio",
                    "peak_allocated_vram_ratio",
                    "cumulative_scan_ratio",
                ],
                "maximum_ratio": 1.10,
            },
            "classification_priority": [
                "FULL_HISTORY_DOMINANT",
                "ACCURACY_ADVANTAGE_BUT_COSTLY",
                "HORIZON_ROBUST",
            ],
        },
    }


def load_phase_ii_evaluation_config(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise AdaptationEvidenceError("Phase II evaluation config is unavailable")
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise AdaptationEvidenceError("cannot decode Phase II evaluation config") from error
    expected = _expected_config()
    if value != expected:
        raise AdaptationEvidenceError("Phase II evaluation config differs")
    return copy.deepcopy(expected)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _key_identity(value: Mapping[str, object]) -> str:
    return _canonical_json(dict(value)).decode("ascii")


def _scope(value: Mapping[str, object]) -> tuple[str, str, str]:
    try:
        reference = value["reference_scene_id"]
        master = value["master_sequence_id"]
        order = value["order_id"]
    except KeyError as error:
        raise AdaptationEvidenceError("adapted key identity is incomplete") from error
    if (
        not isinstance(reference, str)
        or not reference
        or not isinstance(master, str)
        or not master
        or order not in ORDERS
    ):
        raise AdaptationEvidenceError("adapted key identity is invalid")
    return reference, master, str(order)


def _horizon(value: Mapping[str, object]) -> int:
    horizon = value.get("horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 5:
        raise AdaptationEvidenceError("adapted key horizon is invalid")
    return horizon


def expected_adapted_keys(
    all_full_history_keys: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate the full 129 x T1-T5 population and return exact T2-T5 keys."""

    if (
        isinstance(all_full_history_keys, (str, bytes))
        or not isinstance(all_full_history_keys, Sequence)
        or len(all_full_history_keys) != 129 * 5
    ):
        raise AdaptationEvidenceError("adapted key coverage must contain 645 prefixes")
    keys = [copy.deepcopy(dict(key)) for key in all_full_history_keys]
    identities = [_key_identity(key) for key in keys]
    if len(set(identities)) != len(identities):
        raise AdaptationEvidenceError("adapted key coverage contains duplicates")
    by_scope: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for key in keys:
        by_scope[_scope(key)].add(_horizon(key))
    if len(by_scope) != 129 or any(values != set(range(1, 6)) for values in by_scope.values()):
        raise AdaptationEvidenceError("adapted key coverage lacks exact T1-T5 scopes")
    selected = [key for key in keys if _horizon(key) >= 2]
    if len(selected) != 516:
        raise AdaptationEvidenceError("adapted T2-T5 coverage must contain 516 prefixes")
    return selected


def _sidecar_key_from_full(value: Mapping[str, object]) -> dict[str, object]:
    required = (
        "reference_scene_id",
        "master_sequence_id",
        "order_id",
        "horizon",
        "history_scan_ids",
        "scan_indices",
    )
    try:
        return {name: copy.deepcopy(value[name]) for name in required}
    except KeyError as error:
        raise AdaptationEvidenceError("prediction key cannot bind a sidecar") from error


def validate_adapted_resume(
    *,
    expected_keys: Sequence[Mapping[str, object]],
    prediction_entries: Sequence[Mapping[str, object]],
    sidecar_entries: Sequence[Mapping[str, object]],
    source_commit: str,
) -> list[dict[str, object]]:
    """Return keys needing inference, including prediction-only interrupted pairs."""

    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise AdaptationEvidenceError("adapted resume source commit is invalid")
    expected = [copy.deepcopy(dict(key)) for key in expected_keys]
    expected_index = {_key_identity(key): key for key in expected}
    if not expected or len(expected_index) != len(expected):
        raise AdaptationEvidenceError("adapted resume expected keys are invalid")

    predictions: dict[str, Mapping[str, object]] = {}
    for entry in prediction_entries:
        key = entry.get("key")
        digest = entry.get("content_sha256")
        if not isinstance(key, Mapping) or not isinstance(digest, str) or len(digest) != 64:
            raise AdaptationEvidenceError("adapted prediction resume entry is invalid")
        identity = _key_identity(dict(key))
        if identity in predictions or identity not in expected_index:
            raise AdaptationEvidenceError("adapted prediction resume coverage differs")
        predictions[identity] = entry

    sidecars: dict[str, Mapping[str, object]] = {}
    for entry in sidecar_entries:
        key = entry.get("key")
        if not isinstance(key, Mapping):
            raise AdaptationEvidenceError("adapted sidecar resume entry is invalid")
        full_identity = next(
            (
                identity
                for identity, full_key in expected_index.items()
                if _sidecar_key_from_full(full_key) == dict(key)
            ),
            None,
        )
        if full_identity is None or full_identity in sidecars:
            raise AdaptationEvidenceError("adapted sidecar resume coverage differs")
        if full_identity not in predictions:
            raise AdaptationEvidenceError("orphan sidecar has no adapted prediction")
        prediction_digest = predictions[full_identity]["content_sha256"]
        if (
            entry.get("source_prediction_content_sha256") != prediction_digest
            or entry.get("reference_prediction_content_sha256") != prediction_digest
            or entry.get("sidecar_source_commit") != source_commit
        ):
            raise AdaptationEvidenceError("adapted sidecar prediction binding differs")
        sidecars[full_identity] = entry
    return [key for identity, key in expected_index.items() if identity not in sidecars]


def _result_scope(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    reference = row.get("reference_scene_id")
    master = row.get("master_sequence_id")
    order = row.get("order_id")
    horizon = row.get("horizon")
    if (
        not isinstance(reference, str)
        or not reference
        or not isinstance(master, str)
        or not master
        or order not in ORDERS
        or horizon not in HORIZONS
    ):
        raise AdaptationEvidenceError("Phase II result scope is invalid")
    return reference, master, str(order), int(horizon)


def _copy_fields(
    source: Mapping[str, object], fields: Sequence[str], *, name: str
) -> dict[str, object]:
    if any(field not in source for field in fields):
        raise AdaptationEvidenceError(f"{name} fields are incomplete")
    return {field: source[field] for field in fields}


def merge_phase_ii_per_sequence(
    *,
    adapted_task_rows: Sequence[Mapping[str, object]],
    adapted_identity_rows: Sequence[Mapping[str, object]],
    frozen_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Merge adapted task/identity evidence with three frozen deployments."""

    task_index: dict[tuple[str, str, str, int], Mapping[str, object]] = {}
    for row in adapted_task_rows:
        scope = _result_scope(row)
        if scope in task_index:
            raise AdaptationEvidenceError("adapted task results contain duplicates")
        _copy_fields(row, TASK_FIELDS, name="adapted task metric")
        task_index[scope] = row
    if len(task_index) != 129 * 4:
        raise AdaptationEvidenceError("adapted task result coverage differs")
    sequence_scopes = {scope[:3] for scope in task_index}
    if len(sequence_scopes) != 129 or {
        (*sequence, horizon)
        for sequence in sequence_scopes
        for horizon in HORIZONS
    } != set(task_index):
        raise AdaptationEvidenceError("adapted task prefix coverage differs")

    identity_index: dict[
        tuple[str, str, str, int, str], Mapping[str, object]
    ] = {}
    for row in adapted_identity_rows:
        scope = _result_scope(row)
        source_method = row.get("method_id")
        if source_method not in _ADAPTED_METHOD_MAP:
            raise AdaptationEvidenceError("adapted identity method differs")
        key = (*scope, str(source_method))
        if key in identity_index:
            raise AdaptationEvidenceError("adapted identity results contain duplicates")
        _copy_fields(
            row,
            (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS),
            name="adapted identity metric",
        )
        identity_index[key] = row
    expected_identity = {
        (*scope, method) for scope in task_index for method in _ADAPTED_METHOD_MAP
    }
    if set(identity_index) != expected_identity:
        raise AdaptationEvidenceError("adapted identity result coverage differs")

    frozen_index: dict[tuple[str, str, str, int, str], Mapping[str, object]] = {}
    for row in frozen_rows:
        scope = _result_scope(row)
        source_method = row.get("method_id")
        if source_method not in _FROZEN_METHOD_MAP:
            raise AdaptationEvidenceError("frozen Phase II method differs")
        key = (*scope, str(source_method))
        if key in frozen_index:
            raise AdaptationEvidenceError("frozen Phase II results contain duplicates")
        _copy_fields(row, TASK_FIELDS, name="frozen task metric")
        _copy_fields(
            row,
            (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS),
            name="frozen identity metric",
        )
        frozen_index[key] = row
    expected_frozen = {
        (*scope, method) for scope in task_index for method in _FROZEN_METHOD_MAP
    }
    if set(frozen_index) != expected_frozen:
        raise AdaptationEvidenceError("frozen Phase II result coverage differs")

    merged = []
    for scope in sorted(task_index):
        task = task_index[scope]
        for source_method, method_id in _ADAPTED_METHOD_MAP.items():
            identity = identity_index[(*scope, source_method)]
            merged.append(
                {
                    "method_id": method_id,
                    "method": PHASE_II_METHOD_NAMES[method_id],
                    "reference_scene_id": scope[0],
                    "master_sequence_id": scope[1],
                    "order_id": scope[2],
                    "horizon": scope[3],
                    "tracker_initialization_horizon": 2,
                    "task_metric_source": "adapted_checkpoint_cache",
                    **_copy_fields(task, TASK_FIELDS, name="adapted task metric"),
                    **_copy_fields(
                        identity,
                        (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS),
                        name="adapted identity metric",
                    ),
                }
            )
        for source_method, method_id in _FROZEN_METHOD_MAP.items():
            frozen = frozen_index[(*scope, source_method)]
            merged.append(
                {
                    **dict(frozen),
                    "method_id": method_id,
                    "method": PHASE_II_METHOD_NAMES[method_id],
                }
            )
    merged.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            str(row["method_id"]),
        )
    )
    return merged


def aggregate_phase_ii_results(
    *,
    per_sequence_rows: Sequence[Mapping[str, object]],
    pooled_task_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Combine official pooled task metrics with count-recomputed identity metrics."""

    cells: dict[tuple[str, str, str, int, str], Mapping[str, object]] = {}
    for row in per_sequence_rows:
        scope = _result_scope(row)
        method = row.get("method_id")
        if method not in PHASE_II_METHOD_IDS:
            raise AdaptationEvidenceError("Phase II per-sequence method differs")
        key = (*scope, str(method))
        if key in cells:
            raise AdaptationEvidenceError("Phase II per-sequence results contain duplicates")
        cells[key] = row
    if len(cells) != 129 * 4 * len(PHASE_II_METHOD_IDS):
        raise AdaptationEvidenceError("Phase II per-sequence coverage differs")

    task_index: dict[tuple[str, str, int], Mapping[str, object]] = {}
    valid_orders = ("all", *ORDERS)
    for row in pooled_task_rows:
        method = row.get("method_id")
        order = row.get("order_id")
        horizon = row.get("horizon")
        if (
            method not in PHASE_II_METHOD_IDS
            or order not in valid_orders
            or horizon not in HORIZONS
        ):
            raise AdaptationEvidenceError("pooled Phase II task cell differs")
        key = (str(method), str(order), int(horizon))
        if key in task_index:
            raise AdaptationEvidenceError("pooled Phase II task results contain duplicates")
        _copy_fields(row, TASK_FIELDS, name="pooled Phase II task metric")
        task_index[key] = row
    expected_tasks = {
        (method, order, horizon)
        for method in PHASE_II_METHOD_IDS
        for order in valid_orders
        for horizon in HORIZONS
    }
    if set(task_index) != expected_tasks:
        raise AdaptationEvidenceError("pooled Phase II task coverage differs")

    summaries = []
    for method in PHASE_II_METHOD_IDS:
        method_rows = [row for row in per_sequence_rows if row["method_id"] == method]
        method_name = {str(row.get("method")) for row in method_rows}
        initialization = {row.get("tracker_initialization_horizon") for row in method_rows}
        if len(method_name) != 1 or len(initialization) != 1:
            raise AdaptationEvidenceError("Phase II method metadata differs")
        for order in valid_orders:
            for horizon in HORIZONS:
                selected = [
                    row
                    for row in method_rows
                    if row["horizon"] == horizon
                    and (order == "all" or row["order_id"] == order)
                ]
                expected_count = 129 if order == "all" else 43
                task = task_index[(method, order, horizon)]
                if (
                    len(selected) != expected_count
                    or task.get("sequence_count") != expected_count
                ):
                    raise AdaptationEvidenceError("Phase II summary coverage differs")
                summaries.append(
                    {
                        "method_id": method,
                        "method": next(iter(method_name)),
                        "order_id": order,
                        "horizon": horizon,
                        "sequence_count": expected_count,
                        "tracker_initialization_horizon": next(iter(initialization)),
                        "task_metric_source": task.get("task_metric_source"),
                        **_copy_fields(
                            task, TASK_FIELDS, name="pooled Phase II task metric"
                        ),
                        **aggregate_identity_metrics(selected),
                    }
                )
    return {
        "results": [row for row in summaries if row["order_id"] == "all"],
        "per_order": [row for row in summaries if row["order_id"] != "all"],
    }


def build_phase_ii_compute_rows(
    *,
    adapted_profile_rows: Sequence[Mapping[str, object]],
    frozen_profile_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Aggregate the common six-cluster profile for three compute methods."""

    normalized: list[dict[str, object]] = []
    for source_rows, method_map in (
        (adapted_profile_rows, {"FullHistoryAdapted": "FullHistoryAdapted"}),
        (
            frozen_profile_rows,
            {"FullHistory": "FullHistoryFrozen", "Persist4D": "Persist4D"},
        ),
    ):
        for source in source_rows:
            source_method = source.get("method")
            if source_method not in method_map:
                raise AdaptationEvidenceError("Phase II profile method differs")
            if source.get("status") != "pass" or source.get("order_id") != "canonical":
                raise AdaptationEvidenceError("Phase II profile cell did not pass")
            scope = _result_scope(source)
            normalized.append(
                {
                    **dict(source),
                    "method_id": method_map[str(source_method)],
                    "_scope": scope,
                }
            )
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    seen = set()
    for row in normalized:
        method = str(row["method_id"])
        horizon = int(row["horizon"])
        scope = row["_scope"]
        identity = (method, *scope)
        if identity in seen:
            raise AdaptationEvidenceError("Phase II profile contains duplicate cells")
        seen.add(identity)
        grouped[(method, horizon)].append(row)
    expected = {
        (method, horizon)
        for method in PHASE_II_COMPUTE_METHOD_NAMES
        for horizon in HORIZONS
    }
    if set(grouped) != expected or any(len(rows) != 6 for rows in grouped.values()):
        raise AdaptationEvidenceError("Phase II profile coverage differs")

    def finite_values(
        rows: Sequence[Mapping[str, object]], field: str
    ) -> list[float]:
        return [_finite(row.get(field), name=field) for row in rows]

    def maximum_optional(
        rows: Sequence[Mapping[str, object]], field: str
    ) -> int | None:
        values = [row.get(field) for row in rows if row.get(field) is not None]
        if not values:
            return None
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise AdaptationEvidenceError(f"{field} must contain integer bytes")
        return max(int(value) for value in values)

    result = []
    for method, method_name in PHASE_II_COMPUTE_METHOD_NAMES.items():
        for horizon in HORIZONS:
            rows = grouped[(method, horizon)]
            references = {str(row["reference_scene_id"]) for row in rows}
            if len(references) != 6:
                raise AdaptationEvidenceError("Phase II profile lacks six clusters")
            result.append(
                {
                    "method_id": method,
                    "method": method_name,
                    "horizon": horizon,
                    "profile_cluster_count": 6,
                    "warmup_repeats": 5,
                    "measured_repeats": 10,
                    "scans_processed_per_update": float(
                        statistics.mean(finite_values(rows, "update_scan_count"))
                    ),
                    "cumulative_scans_processed": float(
                        statistics.mean(
                            finite_values(rows, "cumulative_scan_count")
                        )
                    ),
                    "median_latency_ms": float(
                        statistics.median(
                            finite_values(rows, "median_latency_ms")
                        )
                    ),
                    "peak_allocated_mib": max(
                        finite_values(rows, "peak_allocated_mib")
                    ),
                    "peak_reserved_mib": max(
                        finite_values(rows, "peak_reserved_mib")
                    ),
                    "mean_update_point_count": float(
                        statistics.mean(
                            finite_values(rows, "update_point_count")
                        )
                    ),
                    "mean_cumulative_point_count": float(
                        statistics.mean(
                            finite_values(rows, "cumulative_point_count")
                        )
                    ),
                    "historical_state_bytes": maximum_optional(
                        rows, "persistent_state_bytes"
                    ),
                    "explicit_history_input_bytes": maximum_optional(
                        rows, "explicit_history_input_bytes"
                    ),
                }
            )
    return result


def build_gate_ii_evidence(
    *,
    result_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    order_rows: Sequence[Mapping[str, object]],
    loso_rows: Sequence[Mapping[str, object]],
    compute_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Assemble preregistered Gate II inputs without mixing aggregation levels."""

    result_index = {
        (str(row.get("method_id")), int(row.get("horizon", -1))): row
        for row in result_rows
        if row.get("method_id") in {"FullHistoryAdaptedB2", "Persist4D"}
        and row.get("horizon") in GATE_HORIZONS
    }
    expected_results = {
        (method, horizon)
        for method in ("FullHistoryAdaptedB2", "Persist4D")
        for horizon in GATE_HORIZONS
    }
    if set(result_index) != expected_results:
        raise AdaptationEvidenceError("Gate II pooled result coverage differs")

    def metric_index(
        rows: Sequence[Mapping[str, object]], *, name: str
    ) -> dict[tuple[str, int], Mapping[str, object]]:
        index: dict[tuple[str, int], Mapping[str, object]] = {}
        for row in rows:
            metric = row.get("metric")
            horizon = row.get("horizon")
            if metric not in STATISTICAL_METRICS or horizon not in GATE_HORIZONS:
                raise AdaptationEvidenceError(f"{name} cell differs")
            key = (str(metric), int(horizon))
            if key in index:
                raise AdaptationEvidenceError(f"{name} contains duplicate cells")
            index[key] = row
        expected = {
            (metric, horizon)
            for metric in STATISTICAL_METRICS
            for horizon in GATE_HORIZONS
        }
        if set(index) != expected:
            raise AdaptationEvidenceError(f"{name} coverage differs")
        return index

    bootstrap = metric_index(bootstrap_rows, name="Gate II bootstrap")
    order = metric_index(order_rows, name="Gate II order evidence")
    loso: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in loso_rows:
        metric = row.get("metric")
        horizon = row.get("horizon")
        if metric not in STATISTICAL_METRICS or horizon not in GATE_HORIZONS:
            raise AdaptationEvidenceError("Gate II LOSO cell differs")
        loso[(str(metric), int(horizon))].append(row)
    if set(loso) != set(bootstrap) or any(len(rows) != 6 for rows in loso.values()):
        raise AdaptationEvidenceError("Gate II LOSO coverage differs")

    task_evidence = []
    identity_evidence = []
    order_fields = (
        "canonical_difference",
        "reverse_difference",
        "sha256_seed45_difference",
    )
    for metric in STATISTICAL_METRICS:
        for horizon in GATE_HORIZONS:
            key = (metric, horizon)
            bootstrap_row = bootstrap[key]
            order_differences = [
                _finite(order[key].get(field), name=field) for field in order_fields
            ]
            loso_differences = [
                _finite(row.get("difference"), name="LOSO difference")
                for row in loso[key]
            ]
            common = {
                "metric": metric,
                "horizon": horizon,
                "difference": _finite(
                    bootstrap_row.get("difference"), name="cluster difference"
                ),
                "ci_lower": _finite(
                    bootstrap_row.get("ci_lower"), name="cluster CI lower"
                ),
                "ci_upper": _finite(
                    bootstrap_row.get("ci_upper"), name="cluster CI upper"
                ),
            }
            if metric in TASK_STATISTICAL_METRICS:
                challenger = result_index[("FullHistoryAdaptedB2", horizon)]
                baseline = result_index[("Persist4D", horizon)]
                pooled_difference = _finite(
                    challenger.get(metric), name="challenger pooled task metric"
                ) - _finite(baseline.get(metric), name="baseline pooled task metric")
                task_evidence.append(
                    {
                        **common,
                        "pooled_difference": pooled_difference,
                        "order_consistent": all(value > 0 for value in order_differences),
                        "loso_consistent": all(value > 0 for value in loso_differences),
                    }
                )
            else:
                persist_direction = (
                    (lambda value: value > 0)
                    if metric == "normalized_id_switch_rate"
                    else (lambda value: value < 0)
                )
                identity_evidence.append(
                    {
                        **common,
                        "order_consistent": all(
                            persist_direction(value) for value in order_differences
                        ),
                        "loso_consistent": all(
                            persist_direction(value) for value in loso_differences
                        ),
                    }
                )

    compute_index = {
        (str(row.get("method_id")), int(row.get("horizon", -1))): row
        for row in compute_rows
        if row.get("method_id") in {"FullHistoryAdapted", "Persist4D"}
        and row.get("horizon") in GATE_HORIZONS
    }
    expected_compute = {
        (method, horizon)
        for method in ("FullHistoryAdapted", "Persist4D")
        for horizon in GATE_HORIZONS
    }
    if set(compute_index) != expected_compute:
        raise AdaptationEvidenceError("Gate II compute coverage differs")
    compute_evidence = []
    for horizon in GATE_HORIZONS:
        challenger = compute_index[("FullHistoryAdapted", horizon)]
        baseline = compute_index[("Persist4D", horizon)]

        def ratio(
            field: str,
            *,
            challenger: Mapping[str, object] = challenger,
            baseline: Mapping[str, object] = baseline,
        ) -> float:
            denominator = _finite(baseline.get(field), name=f"baseline {field}")
            if denominator <= 0:
                raise AdaptationEvidenceError(f"baseline {field} must be positive")
            return _finite(challenger.get(field), name=f"challenger {field}") / denominator

        compute_evidence.append(
            {
                "horizon": horizon,
                "latency_ratio": ratio("median_latency_ms"),
                "peak_allocated_vram_ratio": ratio("peak_allocated_mib"),
                "cumulative_scan_ratio": ratio("cumulative_scans_processed"),
            }
        )
    return {
        "task": task_evidence,
        "identity": identity_evidence,
        "compute": compute_evidence,
    }


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptationEvidenceError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise AdaptationEvidenceError(f"{name} must be finite")
    return result


def _metric_value(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    if metric in TASK_STATISTICAL_METRICS:
        return float(np.mean([_finite(row.get(metric), name=metric) for row in rows]))
    if metric in IDENTITY_STATISTICAL_METRICS:
        value = aggregate_identity_metrics(rows)[metric]
        if value is None:
            raise AdaptationEvidenceError(f"identity metric {metric} is missing")
        return float(value)
    raise AdaptationEvidenceError(f"unsupported Phase II metric: {metric}")


def _paired_cluster_values(
    rows: Sequence[Mapping[str, object]],
    *,
    challenger: str,
    baseline: str,
    metric: str,
    horizon: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    cells: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        method = row.get("method_id")
        if method not in {challenger, baseline} or row.get("horizon") != horizon:
            continue
        reference = row.get("reference_scene_id")
        master = row.get("master_sequence_id")
        order = row.get("order_id")
        if (
            not isinstance(reference, str)
            or not reference
            or not isinstance(master, str)
            or not master
            or order not in ORDERS
        ):
            raise AdaptationEvidenceError("Phase II paired row identity is invalid")
        grouped[(reference, str(method))].append(row)
        cells[(reference, str(method))].add((master, str(order)))
    references = sorted({reference for reference, _method in grouped})
    if len(references) != 6:
        raise AdaptationEvidenceError("Phase II statistics require six clusters")
    result = []
    for reference in references:
        challenger_rows = grouped.get((reference, challenger), [])
        baseline_rows = grouped.get((reference, baseline), [])
        if not challenger_rows or cells[(reference, challenger)] != cells[(reference, baseline)]:
            raise AdaptationEvidenceError("Phase II paired cluster coverage differs")
        challenger_value = _metric_value(challenger_rows, metric)
        baseline_value = _metric_value(baseline_rows, metric)
        result.append(
            {
                "reference_scene_id": reference,
                "pair_count": len(cells[(reference, challenger)]),
                "challenger_value": challenger_value,
                "baseline_value": baseline_value,
                "difference": challenger_value - baseline_value,
            }
        )
    return result


def _selected_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    horizon: int,
    references: set[str] | None = None,
    order: str | None = None,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row.get("method_id") == method
        and row.get("horizon") == horizon
        and (references is None or row.get("reference_scene_id") in references)
        and (order is None or row.get("order_id") == order)
    ]


def paired_phase_ii_statistics(
    rows: Sequence[Mapping[str, object]],
    *,
    challenger_method_id: str,
    baseline_method_id: str,
    replicates: int = 10_000,
    seed: int = 45,
) -> dict[str, list[dict[str, object]]]:
    if replicates != 10_000 or seed != 45:
        raise AdaptationEvidenceError("Phase II statistical settings differ")
    if not challenger_method_id or not baseline_method_id or challenger_method_id == baseline_method_id:
        raise AdaptationEvidenceError("Phase II paired methods are invalid")
    rng = np.random.default_rng(seed)
    bootstrap = []
    order_rows = []
    loso_rows = []
    for metric in STATISTICAL_METRICS:
        for horizon in GATE_HORIZONS:
            clusters = _paired_cluster_values(
                rows,
                challenger=challenger_method_id,
                baseline=baseline_method_id,
                metric=metric,
                horizon=horizon,
            )
            differences = np.asarray(
                [float(row["difference"]) for row in clusters], dtype=np.float64
            )
            indices = rng.integers(0, 6, size=(replicates, 6))
            samples = differences[indices].mean(axis=1)
            challenger_mean = float(
                np.mean([float(row["challenger_value"]) for row in clusters])
            )
            baseline_mean = float(
                np.mean([float(row["baseline_value"]) for row in clusters])
            )
            difference = challenger_mean - baseline_mean
            bootstrap.append(
                {
                    "challenger_method_id": challenger_method_id,
                    "baseline_method_id": baseline_method_id,
                    "metric": metric,
                    "horizon": horizon,
                    "cluster_count": 6,
                    "pair_count": sum(int(row["pair_count"]) for row in clusters),
                    "bootstrap_replicates": replicates,
                    "seed": seed,
                    "challenger_mean": challenger_mean,
                    "baseline_mean": baseline_mean,
                    "difference": difference,
                    "relative_difference": (
                        difference / abs(baseline_mean) if baseline_mean else None
                    ),
                    "ci_lower": float(np.quantile(samples, 0.025)),
                    "ci_upper": float(np.quantile(samples, 0.975)),
                }
            )
            per_order: dict[str, float] = {}
            for order in ORDERS:
                challenger_values = _selected_rows(
                    rows,
                    method=challenger_method_id,
                    horizon=horizon,
                    order=order,
                )
                baseline_values = _selected_rows(
                    rows,
                    method=baseline_method_id,
                    horizon=horizon,
                    order=order,
                )
                per_order[order] = _metric_value(challenger_values, metric) - _metric_value(
                    baseline_values, metric
                )
            order_rows.append(
                {
                    "challenger_method_id": challenger_method_id,
                    "baseline_method_id": baseline_method_id,
                    "metric": metric,
                    "horizon": horizon,
                    "canonical_difference": per_order["canonical"],
                    "reverse_difference": per_order["reverse"],
                    "sha256_seed45_difference": per_order["sha256_seed45"],
                    "sign_consistent": all(
                        value * difference > 0 for value in per_order.values()
                    ),
                }
            )
            references = {str(row["reference_scene_id"]) for row in clusters}
            for dropped in sorted(references):
                kept = references - {dropped}
                challenger_values = _selected_rows(
                    rows,
                    method=challenger_method_id,
                    horizon=horizon,
                    references=kept,
                )
                baseline_values = _selected_rows(
                    rows,
                    method=baseline_method_id,
                    horizon=horizon,
                    references=kept,
                )
                loso_difference = _metric_value(
                    challenger_values, metric
                ) - _metric_value(baseline_values, metric)
                loso_rows.append(
                    {
                        "challenger_method_id": challenger_method_id,
                        "baseline_method_id": baseline_method_id,
                        "metric": metric,
                        "horizon": horizon,
                        "dropped_reference_scene_id": dropped,
                        "remaining_cluster_count": 5,
                        "difference": loso_difference,
                        "sign_consistent": loso_difference * difference > 0,
                    }
                )
    return {
        "bootstrap": bootstrap,
        "order_robustness": order_rows,
        "leave_one_scene_out": loso_rows,
    }


def _index_exact(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str],
    name: str,
) -> dict[tuple[str, int], Mapping[str, object]]:
    result = {}
    for row in rows:
        metric = row.get("metric")
        horizon = row.get("horizon")
        key = (str(metric), int(horizon) if isinstance(horizon, int) else -1)
        if key in result:
            raise AdaptationEvidenceError(f"{name} contains duplicate cells")
        result[key] = row
    expected = {(metric, horizon) for metric in metrics for horizon in GATE_HORIZONS}
    if set(result) != expected:
        raise AdaptationEvidenceError(f"{name} coverage differs")
    return result


def derive_gate_ii(
    *,
    task_evidence: Sequence[Mapping[str, object]],
    identity_evidence: Sequence[Mapping[str, object]],
    compute_evidence: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    if dict(config) != _expected_config():
        raise AdaptationEvidenceError("Gate II config differs")
    task = _index_exact(
        task_evidence,
        metrics=TASK_STATISTICAL_METRICS,
        name="Gate II task evidence",
    )
    identity = _index_exact(
        identity_evidence,
        metrics=IDENTITY_STATISTICAL_METRICS,
        name="Gate II identity evidence",
    )
    threshold = float(
        config["gate_ii"]["substantial_task_advantage"][  # type: ignore[index]
            "minimum_pooled_absolute_difference"
        ]
    )
    qualifying_task_cells = []
    for key, row in task.items():
        qualifies = (
            _finite(row.get("pooled_difference"), name="pooled task difference")
            >= threshold
            and _finite(row.get("ci_lower"), name="task CI lower") > 0
            and bool(row.get("order_consistent"))
            and bool(row.get("loso_consistent"))
        )
        if qualifies:
            qualifying_task_cells.append({"metric": key[0], "horizon": key[1]})
    task_advantage_horizons = [
        horizon
        for horizon in GATE_HORIZONS
        if all(
            {"metric": metric, "horizon": horizon} in qualifying_task_cells
            for metric in TASK_STATISTICAL_METRICS
        )
    ]

    robust_persist_advantages = []
    for (metric, horizon), row in identity.items():
        difference = _finite(row.get("difference"), name="identity difference")
        lower = _finite(row.get("ci_lower"), name="identity CI lower")
        upper = _finite(row.get("ci_upper"), name="identity CI upper")
        direction = lower > 0 if metric == "normalized_id_switch_rate" else upper < 0
        if direction and bool(row.get("order_consistent")) and bool(
            row.get("loso_consistent")
        ):
            robust_persist_advantages.append(
                {"metric": metric, "horizon": horizon, "difference": difference}
            )
    identity_gap_closed = not robust_persist_advantages

    compute_by_horizon: dict[int, Mapping[str, object]] = {}
    for row in compute_evidence:
        horizon = row.get("horizon")
        if horizon not in GATE_HORIZONS or int(horizon) in compute_by_horizon:
            raise AdaptationEvidenceError("Gate II compute evidence coverage differs")
        compute_by_horizon[int(horizon)] = row
    if set(compute_by_horizon) != set(GATE_HORIZONS):
        raise AdaptationEvidenceError("Gate II compute evidence coverage differs")
    maximum_ratio = float(
        config["gate_ii"]["compute_no_material_disadvantage"][  # type: ignore[index]
            "maximum_ratio"
        ]
    )
    ratios = ("latency_ratio", "peak_allocated_vram_ratio", "cumulative_scan_ratio")
    compute_no_material_disadvantage = all(
        _finite(row.get(name), name=name) <= maximum_ratio
        for row in compute_by_horizon.values()
        for name in ratios
    )

    if (
        task_advantage_horizons == list(GATE_HORIZONS)
        and identity_gap_closed
        and compute_no_material_disadvantage
    ):
        classification = "FULL_HISTORY_DOMINANT"
    elif task_advantage_horizons:
        classification = "ACCURACY_ADVANTAGE_BUT_COSTLY"
    else:
        classification = "HORIZON_ROBUST"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "classification": classification,
        "task_advantage_horizons": task_advantage_horizons,
        "qualifying_task_cells": qualifying_task_cells,
        "identity_gap_closed": identity_gap_closed,
        "robust_persist4d_identity_advantages": robust_persist_advantages,
        "compute_no_material_disadvantage": compute_no_material_disadvantage,
        "thresholds": {
            "minimum_pooled_task_difference": threshold,
            "maximum_compute_ratio": maximum_ratio,
        },
    }
    result["content_sha256"] = _content_sha256(result)
    return result


__all__ = [
    "AdaptationEvidenceError",
    "derive_gate_ii",
    "expected_adapted_keys",
    "load_phase_ii_evaluation_config",
    "paired_phase_ii_statistics",
    "validate_adapted_resume",
]
