"""Phase-I statistics for the frozen Full-History tracker challenge."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE_I_ANALYSIS_CONFIG_PATH = (
    PROJECT_ROOT / "configs/reviewer_closure/phase_i_analysis.yaml"
)
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)
GATE_HORIZONS = (4, 5)
ELIGIBLE_TRACKERS = ("B1", "B2", "B3")
DIAGNOSTIC_TRACKER = "B4"
METHOD_IDS = (
    "FullHistoryNative",
    "B1",
    "B2",
    "B3",
    "B4",
    "Persist4D",
)
METHOD_NAMES = {
    "FullHistoryNative": "ReScene4D Full-History",
    "B1": "Pairwise Feature Association",
    "B2": "Pairwise Feature-Class Association",
    "B3": "EMA Temporal Association",
    "B4": "Full-History + Persistent-State Diagnostic",
    "Persist4D": "Persist4D Persistent Entity State",
}
TASK_FIELDS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_mAP50",
    "causal_prefix_t_mAP25",
    "causal_prefix_t_REC",
    "causal_prefix_t_REC50",
    "causal_prefix_t_REC25",
    "current_stage_AP",
    "current_stage_AP50",
    "current_stage_AP25",
    "current_stage_REC",
)
IDENTITY_COUNT_FIELDS = (
    "deployment_id_switches",
    "identity_transition_opportunities",
    "fragmentation_count",
    "fragmentation_opportunities",
    "merge_count",
    "merge_opportunities",
    "gap_opportunities",
    "recovery_attempts",
    "correct_recoveries",
)
IDENTITY_RATE_FIELDS = (
    "normalized_id_switch_rate",
    "fragmentation_rate",
    "merge_rate",
    "gap_recovery_accuracy",
    "gap_recovery_recall",
)
GATE_METRICS = ("normalized_id_switch_rate", "gap_recovery_recall")
_STRING_FIELDS = {
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
}
_INTEGER_FIELDS = {
    "horizon",
    "sequence_count",
    "update_scan_count",
    "update_point_count",
    "cumulative_scan_count",
    *IDENTITY_COUNT_FIELDS,
}


class ReviewerClosureAnalysisError(ValueError):
    """Raised when Phase-I evidence violates the frozen contract."""


def load_phase_i_analysis_config(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReviewerClosureAnalysisError(
            "Phase-I analysis config must be a regular file"
        )
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ReviewerClosureAnalysisError(
            "cannot decode Phase-I analysis config"
        ) from error
    expected = {
        "schema_version": 1,
        "stage_name": "persist4d-reviewer-closure-phase-i-analysis",
        "source_tracker_artifact_content_sha256": (
            "d6f2480e2787a943b5fd2a7f9c412a14c32bf0106bc23130f363fc72584efbf9"
        ),
        "strongest_simple_tracker": {
            "eligible_method_ids": ["B1", "B2", "B3"],
            "compared_horizons": [4, 5],
            "ranking": [
                "pooled_t4_t5_normalized_id_switch_rate_ascending",
                "pooled_t4_t5_gap_recovery_recall_descending",
                "method_id_ascending",
            ],
            "diagnostic_excluded_method_id": "B4",
        },
        "statistics": {
            "cluster_unit": "reference_scene_id",
            "cluster_count": 6,
            "bootstrap_replicates": 10000,
            "seed": 45,
            "confidence_level": 0.95,
        },
        "gate_i": {
            "compared_horizons": [4, 5],
            "identity_metrics": [
                "normalized_id_switch_rate",
                "gap_recovery_recall",
            ],
            "decision_rule": (
                "any_metric_any_horizon_persist4d_advantage_with_ci_order_and_loso"
            ),
            "requires_complete_six_cluster_population": True,
            "requires_ci_exclude_zero": True,
            "requires_order_sign_consistency": True,
            "requires_loso_sign_consistency": True,
        },
    }
    if value != expected:
        raise ReviewerClosureAnalysisError("Phase-I analysis config differs")
    return expected


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _nonempty_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReviewerClosureAnalysisError(f"{name} must be nonempty text")
    return value


def _count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReviewerClosureAnalysisError(f"{name} must be a non-negative integer")
    return value


def _finite_or_missing(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewerClosureAnalysisError(f"{name} must be finite or missing")
    result = float(value)
    if not math.isfinite(result):
        raise ReviewerClosureAnalysisError(f"{name} must be finite or missing")
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_identity_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ReviewerClosureAnalysisError("identity aggregation requires rows")
    totals = {
        field: sum(_count(row.get(field), name=field) for row in rows)
        for field in IDENTITY_COUNT_FIELDS
    }
    return {
        **totals,
        "normalized_id_switch_rate": _rate(
            totals["deployment_id_switches"],
            totals["identity_transition_opportunities"],
        ),
        "fragmentation_rate": _rate(
            totals["fragmentation_count"], totals["fragmentation_opportunities"]
        ),
        "merge_rate": _rate(totals["merge_count"], totals["merge_opportunities"]),
        "gap_recovery_accuracy": _rate(
            totals["correct_recoveries"], totals["recovery_attempts"]
        ),
        "gap_recovery_recall": _rate(
            totals["correct_recoveries"], totals["gap_opportunities"]
        ),
    }


def _load_json(path: str | Path, *, name: str) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReviewerClosureAnalysisError(f"{name} must be a regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewerClosureAnalysisError(f"cannot decode {name}") from error
    if not isinstance(value, dict):
        raise ReviewerClosureAnalysisError(f"{name} must be a mapping")
    return value


def load_tracker_raw_artifact(path: str | Path) -> dict[str, object]:
    artifact = _load_json(path, name="Full-History tracker artifact")
    if artifact.get("schema_version") != 1 or artifact.get("status") != "pass":
        raise ReviewerClosureAnalysisError("tracker artifact status differs")
    if artifact.get("content_sha256") != _content_sha256(artifact):
        raise ReviewerClosureAnalysisError("tracker artifact content hash differs")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != 43 * 3 * 4 * 5:
        raise ReviewerClosureAnalysisError("tracker artifact row count differs")
    cells: set[tuple[str, str, str, int]] = set()
    scopes: set[tuple[str, str, str]] = set()
    references: set[str] = set()
    for value in rows:
        if not isinstance(value, Mapping):
            raise ReviewerClosureAnalysisError("tracker row must be a mapping")
        method = _nonempty_text(value.get("method_id"), name="method_id")
        reference = _nonempty_text(
            value.get("reference_scene_id"), name="reference_scene_id"
        )
        master = _nonempty_text(
            value.get("master_sequence_id"), name="master_sequence_id"
        )
        order = _nonempty_text(value.get("order_id"), name="order_id")
        horizon = _count(value.get("horizon"), name="horizon")
        if (
            method not in METHOD_IDS[:-1]
            or order not in ORDERS
            or horizon not in HORIZONS
        ):
            raise ReviewerClosureAnalysisError("tracker row identity differs")
        if value.get("method") != METHOD_NAMES[method]:
            raise ReviewerClosureAnalysisError("tracker paper method name differs")
        for field in IDENTITY_COUNT_FIELDS:
            _count(value.get(field), name=field)
        for field in IDENTITY_RATE_FIELDS:
            _finite_or_missing(value.get(field), name=field)
        cell = (master, order, method, horizon)
        if cell in cells:
            raise ReviewerClosureAnalysisError("tracker artifact has duplicate cells")
        cells.add(cell)
        scopes.add((reference, master, order))
        references.add(reference)
    expected = {
        (master, order, method, horizon)
        for _reference, master, order in scopes
        for method in METHOD_IDS[:-1]
        for horizon in HORIZONS
    }
    if (
        cells != expected
        or len(scopes) != 43 * 3
        or len(references) != 6
        or artifact.get("sequence_count") != len(scopes)
        or artifact.get("row_count") != len(rows)
    ):
        raise ReviewerClosureAnalysisError("tracker artifact coverage differs")
    return artifact


def _read_typed_csv(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ReviewerClosureAnalysisError(f"required CSV is unavailable: {source}")
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReviewerClosureAnalysisError(f"cannot decode CSV: {source}") from error
    if not raw_rows:
        raise ReviewerClosureAnalysisError(f"required CSV is empty: {source}")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = {}
        for field, value in raw.items():
            if field in _STRING_FIELDS:
                row[field] = _nonempty_text(value, name=field)
            elif value == "":
                row[field] = None
            elif field in _INTEGER_FIELDS:
                try:
                    row[field] = int(value)
                except ValueError as error:
                    raise ReviewerClosureAnalysisError(
                        f"CSV field {field} is not an integer"
                    ) from error
            else:
                try:
                    row[field] = float(value)
                except ValueError as error:
                    raise ReviewerClosureAnalysisError(
                        f"CSV field {field} is not numeric"
                    ) from error
        rows.append(row)
    return rows


def _system_cell(row: Mapping[str, object]) -> tuple[str, str, str, int]:
    return (
        _nonempty_text(row.get("reference_scene_id"), name="reference_scene_id"),
        _nonempty_text(row.get("master_sequence_id"), name="master_sequence_id"),
        _nonempty_text(row.get("order_id"), name="order_id"),
        _count(row.get("horizon"), name="horizon"),
    )


def read_system_per_sequence_results(path: str | Path) -> list[dict[str, object]]:
    rows = _read_typed_csv(path)
    cells: set[tuple[str, str, str, str, int]] = set()
    scopes: set[tuple[str, str, str]] = set()
    for row in rows:
        method = _nonempty_text(row.get("method"), name="method")
        reference, master, order, horizon = _system_cell(row)
        if method not in {"FullHistory", "Persist4D"}:
            raise ReviewerClosureAnalysisError("system method coverage differs")
        if order not in ORDERS or horizon not in HORIZONS:
            raise ReviewerClosureAnalysisError("system prefix coverage differs")
        for field in TASK_FIELDS:
            _finite_or_missing(row.get(field), name=field)
        for field in IDENTITY_COUNT_FIELDS:
            _count(row.get(field), name=field)
        cell = (method, reference, master, order, horizon)
        if cell in cells:
            raise ReviewerClosureAnalysisError("system results contain duplicate cells")
        cells.add(cell)
        scopes.add((reference, master, order))
    expected = {
        (method, reference, master, order, horizon)
        for reference, master, order in scopes
        for method in ("FullHistory", "Persist4D")
        for horizon in HORIZONS
    }
    if cells != expected or len(scopes) != 43 * 3:
        raise ReviewerClosureAnalysisError("system per-sequence coverage differs")
    return rows


def _copy_metrics(
    source: Mapping[str, object], fields: Sequence[str]
) -> dict[str, object]:
    return {field: source.get(field) for field in fields}


def merge_phase_i_results(
    raw_artifact: Mapping[str, object],
    system_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    raw_rows = raw_artifact.get("rows")
    if not isinstance(raw_rows, list):
        raise ReviewerClosureAnalysisError("tracker artifact lacks rows")
    validated_system = read_system_per_sequence_results_from_rows(system_rows)
    by_system: dict[tuple[str, str, str, int, str], Mapping[str, object]] = {}
    for row in validated_system:
        reference, master, order, horizon = _system_cell(row)
        by_system[(reference, master, order, horizon, str(row["method"]))] = row
    merged: list[dict[str, object]] = []
    raw_scopes: set[tuple[str, str, str, int]] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ReviewerClosureAnalysisError("tracker row must be a mapping")
        scope = (
            str(raw["reference_scene_id"]),
            str(raw["master_sequence_id"]),
            str(raw["order_id"]),
            int(raw["horizon"]),
        )
        source = by_system.get((*scope, "FullHistory"))
        if source is None:
            raise ReviewerClosureAnalysisError("tracker and system scopes differ")
        raw_scopes.add(scope)
        merged.append(
            {
                "method_id": str(raw["method_id"]),
                "method": str(raw["method"]),
                "reference_scene_id": scope[0],
                "master_sequence_id": scope[1],
                "order_id": scope[2],
                "horizon": scope[3],
                "tracker_initialization_horizon": 2,
                "task_metric_source": "frozen_full_history_cache",
                **_copy_metrics(source, TASK_FIELDS),
                **_copy_metrics(raw, (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS)),
            }
        )
    system_scopes = {
        _system_cell(row) for row in validated_system if row["method"] == "FullHistory"
    }
    if raw_scopes != system_scopes:
        raise ReviewerClosureAnalysisError("tracker and system exact coverage differs")
    for row in validated_system:
        if row["method"] != "Persist4D":
            continue
        reference, master, order, horizon = _system_cell(row)
        merged.append(
            {
                "method_id": "Persist4D",
                "method": METHOD_NAMES["Persist4D"],
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
                "horizon": horizon,
                "tracker_initialization_horizon": 1,
                "task_metric_source": "frozen_persist4d_cache",
                **_copy_metrics(row, TASK_FIELDS),
                **_copy_metrics(row, (*IDENTITY_COUNT_FIELDS, *IDENTITY_RATE_FIELDS)),
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
    expected = {(*scope, method) for scope in system_scopes for method in METHOD_IDS}
    actual = {
        (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            str(row["method_id"]),
        )
        for row in merged
    }
    if actual != expected or len(actual) != len(merged):
        raise ReviewerClosureAnalysisError("merged Phase-I coverage differs")
    return merged


def read_system_per_sequence_results_from_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ReviewerClosureAnalysisError("system results must be a sequence")
    if len(rows) != 43 * 3 * 4 * 2:
        raise ReviewerClosureAnalysisError("system per-sequence row count differs")
    cells = set()
    scopes = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReviewerClosureAnalysisError("system result row must be a mapping")
        method = _nonempty_text(row.get("method"), name="method")
        reference, master, order, horizon = _system_cell(row)
        if method not in {"FullHistory", "Persist4D"}:
            raise ReviewerClosureAnalysisError("system method coverage differs")
        cell = (method, reference, master, order, horizon)
        if cell in cells:
            raise ReviewerClosureAnalysisError("system results contain duplicate cells")
        cells.add(cell)
        scopes.add((reference, master, order))
    expected = {
        (method, reference, master, order, horizon)
        for reference, master, order in scopes
        for method in ("FullHistory", "Persist4D")
        for horizon in HORIZONS
    }
    if cells != expected or len(scopes) != 43 * 3:
        raise ReviewerClosureAnalysisError("system per-sequence coverage differs")
    return list(rows)


def select_strongest_simple_tracker(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    ranking = []
    for method_id in ELIGIBLE_TRACKERS:
        selected = [
            row
            for row in rows
            if row.get("method_id") == method_id and row.get("horizon") in GATE_HORIZONS
        ]
        if not selected:
            raise ReviewerClosureAnalysisError(
                f"strongest-tracker selection lacks {method_id} rows"
            )
        metrics = aggregate_identity_metrics(selected)
        switch_rate = metrics["normalized_id_switch_rate"]
        gap_recall = metrics["gap_recovery_recall"]
        if switch_rate is None:
            raise ReviewerClosureAnalysisError("tracker selection lacks ID transitions")
        ranking.append(
            {
                "method_id": method_id,
                "method": METHOD_NAMES[method_id],
                "horizons": [4, 5],
                "normalized_id_switch_rate": switch_rate,
                "gap_recovery_recall": gap_recall,
                **{field: metrics[field] for field in IDENTITY_COUNT_FIELDS},
            }
        )
    ranking.sort(
        key=lambda row: (
            float(row["normalized_id_switch_rate"]),
            (
                -float(row["gap_recovery_recall"])
                if row["gap_recovery_recall"] is not None
                else math.inf
            ),
            str(row["method_id"]),
        )
    )
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "eligible_method_ids": list(ELIGIBLE_TRACKERS),
        "diagnostic_excluded_method_id": DIAGNOSTIC_TRACKER,
        "ranking_rule": [
            "pooled_t4_t5_normalized_id_switch_rate_ascending",
            "pooled_t4_t5_gap_recovery_recall_descending",
            "method_id_ascending",
        ],
        "selected_method_id": ranking[0]["method_id"],
        "ranking": ranking,
    }
    result["content_sha256"] = _content_sha256(result)
    return result


def _metric_expected(value: float, metric: str) -> bool:
    if metric == "normalized_id_switch_rate":
        return value < 0
    if metric == "gap_recovery_recall":
        return value > 0
    raise ReviewerClosureAnalysisError(f"unsupported Gate-I metric: {metric}")


def _cluster_pairs(
    rows: Sequence[Mapping[str, object]],
    *,
    tracker_method_id: str,
    metric: str,
    horizon: int,
) -> list[dict[str, object]]:
    if tracker_method_id not in ELIGIBLE_TRACKERS or metric not in GATE_METRICS:
        raise ReviewerClosureAnalysisError("paired analysis contract differs")
    by_cluster_method: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(
        list
    )
    pair_cells: dict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in rows:
        if row.get("method_id") not in {tracker_method_id, "Persist4D"}:
            continue
        if row.get("horizon") != horizon:
            continue
        reference = _nonempty_text(
            row.get("reference_scene_id"), name="reference_scene_id"
        )
        method = str(row["method_id"])
        by_cluster_method[(reference, method)].append(row)
        pair_cells[reference][method].add(
            (
                _nonempty_text(
                    row.get("master_sequence_id"), name="master_sequence_id"
                ),
                _nonempty_text(row.get("order_id"), name="order_id"),
            )
        )
    references = sorted({key[0] for key in by_cluster_method})
    if len(references) != 6:
        raise ReviewerClosureAnalysisError("paired analysis requires six clusters")
    result = []
    for reference in references:
        if set(pair_cells[reference]) != {tracker_method_id, "Persist4D"}:
            raise ReviewerClosureAnalysisError("paired method coverage differs")
        if (
            pair_cells[reference][tracker_method_id]
            != pair_cells[reference]["Persist4D"]
        ):
            raise ReviewerClosureAnalysisError("paired sequence coverage differs")
        tracker_metrics = aggregate_identity_metrics(
            by_cluster_method[(reference, tracker_method_id)]
        )
        persist_metrics = aggregate_identity_metrics(
            by_cluster_method[(reference, "Persist4D")]
        )
        tracker_value = tracker_metrics[metric]
        persist_value = persist_metrics[metric]
        result.append(
            {
                "reference_scene_id": reference,
                "pair_count": len(pair_cells[reference][tracker_method_id]),
                "tracker_value": (
                    None if tracker_value is None else float(tracker_value)
                ),
                "persist4d_value": (
                    None if persist_value is None else float(persist_value)
                ),
                "difference": (
                    None
                    if tracker_value is None or persist_value is None
                    else float(persist_value) - float(tracker_value)
                ),
            }
        )
    return result


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    tracker_method_id: str,
    metrics: Sequence[str] = GATE_METRICS,
    horizons: Sequence[int] = GATE_HORIZONS,
    replicates: int = 10_000,
    seed: int = 45,
) -> list[dict[str, object]]:
    if replicates != 10_000 or seed != 45:
        raise ReviewerClosureAnalysisError("bootstrap settings differ from protocol")
    rng = np.random.default_rng(seed)
    result = []
    for metric in metrics:
        for horizon in horizons:
            clusters = _cluster_pairs(
                rows,
                tracker_method_id=tracker_method_id,
                metric=metric,
                horizon=horizon,
            )
            finite_clusters = [row for row in clusters if row["difference"] is not None]
            if not finite_clusters:
                raise ReviewerClosureAnalysisError(
                    f"paired metric {metric} has no finite clusters"
                )
            tracker = np.asarray(
                [float(row["tracker_value"]) for row in finite_clusters],
                dtype=np.float64,
            )
            persist = np.asarray(
                [float(row["persist4d_value"]) for row in finite_clusters],
                dtype=np.float64,
            )
            differences = persist - tracker
            indices = rng.integers(
                0,
                len(finite_clusters),
                size=(replicates, len(finite_clusters)),
            )
            samples = differences[indices].mean(axis=1)
            tracker_mean = float(tracker.mean())
            persist_mean = float(persist.mean())
            difference = persist_mean - tracker_mean
            result.append(
                {
                    "tracker_method_id": tracker_method_id,
                    "metric": metric,
                    "horizon": horizon,
                    "reference_scene_count": len(clusters),
                    "cluster_count": len(finite_clusters),
                    "missing_cluster_count": len(clusters) - len(finite_clusters),
                    "pair_count": sum(
                        int(row["pair_count"]) for row in finite_clusters
                    ),
                    "bootstrap_replicates": replicates,
                    "seed": seed,
                    "tracker_mean": tracker_mean,
                    "persist4d_mean": persist_mean,
                    "difference": difference,
                    "relative_difference": (
                        difference / abs(tracker_mean) if tracker_mean else None
                    ),
                    "ci_lower": float(np.quantile(samples, 0.025)),
                    "ci_upper": float(np.quantile(samples, 0.975)),
                }
            )
    return result


def order_robustness(
    rows: Sequence[Mapping[str, object]],
    *,
    tracker_method_id: str,
    metrics: Sequence[str] = GATE_METRICS,
    horizons: Sequence[int] = GATE_HORIZONS,
) -> list[dict[str, object]]:
    result = []
    for metric in metrics:
        for horizon in horizons:
            differences = {}
            complete_by_order = {}
            for order in ORDERS:
                selected = [row for row in rows if row.get("order_id") == order]
                clusters = _cluster_pairs(
                    selected,
                    tracker_method_id=tracker_method_id,
                    metric=metric,
                    horizon=horizon,
                )
                finite_clusters = [
                    row for row in clusters if row["difference"] is not None
                ]
                if not finite_clusters:
                    raise ReviewerClosureAnalysisError(
                        f"order metric {metric} has no finite clusters"
                    )
                complete_by_order[order] = (
                    len(clusters) == 6 and len(finite_clusters) == 6
                )
                differences[order] = float(
                    np.mean([float(row["difference"]) for row in finite_clusters])
                )
            result.append(
                {
                    "tracker_method_id": tracker_method_id,
                    "metric": metric,
                    "horizon": horizon,
                    "canonical_difference": differences["canonical"],
                    "reverse_difference": differences["reverse"],
                    "sha256_seed45_difference": differences["sha256_seed45"],
                    "complete_cluster_population": all(complete_by_order.values()),
                    "expected_direction_consistent": all(
                        _metric_expected(value, metric)
                        for value in differences.values()
                    ),
                }
            )
    return result


def leave_one_scene_out(
    rows: Sequence[Mapping[str, object]],
    *,
    tracker_method_id: str,
    metrics: Sequence[str] = GATE_METRICS,
    horizons: Sequence[int] = GATE_HORIZONS,
) -> list[dict[str, object]]:
    result = []
    for metric in metrics:
        for horizon in horizons:
            clusters = _cluster_pairs(
                rows,
                tracker_method_id=tracker_method_id,
                metric=metric,
                horizon=horizon,
            )
            for dropped in clusters:
                kept = [
                    row
                    for row in clusters
                    if row["reference_scene_id"] != dropped["reference_scene_id"]
                    and row["difference"] is not None
                ]
                if not kept:
                    raise ReviewerClosureAnalysisError(
                        f"LOSO metric {metric} has no finite clusters"
                    )
                difference = float(np.mean([float(row["difference"]) for row in kept]))
                result.append(
                    {
                        "tracker_method_id": tracker_method_id,
                        "metric": metric,
                        "horizon": horizon,
                        "dropped_reference_scene_id": dropped["reference_scene_id"],
                        "remaining_cluster_count": len(kept),
                        "difference": difference,
                        "expected_direction_consistent": _metric_expected(
                            difference, metric
                        ),
                    }
                )
    return result


def derive_gate_i(
    *,
    tracker_method_id: str,
    bootstrap_rows: Sequence[Mapping[str, object]],
    order_rows: Sequence[Mapping[str, object]],
    loso_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if tracker_method_id not in ELIGIBLE_TRACKERS:
        raise ReviewerClosureAnalysisError("Gate-I tracker differs from selection")
    expected_keys = {
        (metric, horizon) for metric in GATE_METRICS for horizon in GATE_HORIZONS
    }
    bootstrap_keys = [
        (str(row.get("metric")), int(row.get("horizon", 0))) for row in bootstrap_rows
    ]
    order_keys = [
        (str(row.get("metric")), int(row.get("horizon", 0))) for row in order_rows
    ]
    if (
        len(bootstrap_keys) != len(expected_keys)
        or set(bootstrap_keys) != expected_keys
        or len(set(bootstrap_keys)) != len(bootstrap_keys)
        or len(order_keys) != len(expected_keys)
        or set(order_keys) != expected_keys
        or len(set(order_keys)) != len(order_keys)
        or any(
            row.get("tracker_method_id") != tracker_method_id
            for row in (*bootstrap_rows, *order_rows, *loso_rows)
        )
    ):
        raise ReviewerClosureAnalysisError("Gate-I evidence coverage differs")
    order_by_key = {
        (str(row["metric"]), int(row["horizon"])): row for row in order_rows
    }
    loso_by_key: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in loso_rows:
        loso_by_key[(str(row["metric"]), int(row["horizon"]))].append(row)
    if set(loso_by_key) != expected_keys or any(
        len(values) != 6
        or len({str(value.get("dropped_reference_scene_id")) for value in values}) != 6
        for values in loso_by_key.values()
    ):
        raise ReviewerClosureAnalysisError("Gate-I LOSO coverage differs")
    qualifying = []
    evidence = []
    for row in bootstrap_rows:
        metric = str(row["metric"])
        horizon = int(row["horizon"])
        key = (metric, horizon)
        ci_lower = float(row["ci_lower"])
        ci_upper = float(row["ci_upper"])
        complete_cluster_population = (
            int(row.get("reference_scene_count", 0)) == 6
            and int(row.get("cluster_count", 0)) == 6
            and int(row.get("missing_cluster_count", 0)) == 0
        )
        ci_advantage = (
            ci_upper < 0 if metric == "normalized_id_switch_rate" else ci_lower > 0
        )
        order_record = order_by_key[key]
        order_consistent = bool(
            order_record.get("complete_cluster_population")
            and order_record.get("expected_direction_consistent")
        )
        loso_values = loso_by_key.get(key, [])
        loso_consistent = len(loso_values) == 6 and all(
            value.get("remaining_cluster_count") == 5
            and bool(value.get("expected_direction_consistent"))
            for value in loso_values
        )
        record = {
            "metric": metric,
            "horizon": horizon,
            "complete_cluster_population": complete_cluster_population,
            "ci_advantage": ci_advantage,
            "order_consistent": order_consistent,
            "loso_consistent": loso_consistent,
        }
        evidence.append(record)
        if all(
            record[field]
            for field in (
                "complete_cluster_population",
                "ci_advantage",
                "order_consistent",
                "loso_consistent",
            )
        ):
            qualifying.append(record)
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "selected_tracker_method_id": tracker_method_id,
        "decision_rule": (
            "any_metric_any_horizon_persist4d_advantage_with_ci_order_and_loso"
        ),
        "classification": (
            "TRACKER_REJECTED" if qualifying else "TRACKER_EXPLAINS_IDENTITY"
        ),
        "qualifying_advantages": qualifying,
        "evidence": evidence,
    }
    result["content_sha256"] = _content_sha256(result)
    return result


def build_task_drift_rows(
    reference_metrics: Mapping[int, Mapping[str, object]],
    replay_metrics: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    if set(reference_metrics) != set(HORIZONS) or set(replay_metrics) != set(HORIZONS):
        raise ReviewerClosureAnalysisError("task drift horizon coverage differs")
    rows = []
    for horizon in HORIZONS:
        reference = reference_metrics[horizon]
        replay = replay_metrics[horizon]
        if set(reference) != set(TASK_FIELDS) or set(replay) != set(TASK_FIELDS):
            raise ReviewerClosureAnalysisError("task drift metric coverage differs")
        for metric in TASK_FIELDS:
            reference_value = _finite_or_missing(
                reference[metric], name=f"reference {metric}"
            )
            replay_value = _finite_or_missing(replay[metric], name=f"replay {metric}")
            if reference_value is None or replay_value is None:
                raise ReviewerClosureAnalysisError("task drift metrics must be finite")
            difference = replay_value - reference_value
            rows.append(
                {
                    "horizon": horizon,
                    "metric": metric,
                    "reference_value": reference_value,
                    "replay_value": replay_value,
                    "difference": difference,
                    "absolute_difference": abs(difference),
                    "relative_difference": (
                        difference / abs(reference_value) if reference_value else None
                    ),
                }
            )
    return rows


def _manifest_content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    supplied = payload.pop("content_sha256", None)
    digest = hashlib.sha256(
        (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if supplied != digest:
        raise ReviewerClosureAnalysisError("cache manifest content hash differs")
    return digest


def _validate_cache_manifest_file(path: str | Path) -> dict[str, object]:
    manifest = _load_json(path, name="Full-History cache manifest")
    _manifest_content_sha256(manifest)
    entries = manifest.get("entries")
    provenance = manifest.get("provenance")
    if not isinstance(entries, list) or not isinstance(provenance, Mapping):
        raise ReviewerClosureAnalysisError("cache manifest fields differ")
    from scripts.system_comparison_inference import build_full_history_cache_manifest

    rebuilt = build_full_history_cache_manifest(
        entries,
        expected_keys=[entry["key"] for entry in entries],
        expected_provenance=provenance,
    )
    if rebuilt != manifest:
        raise ReviewerClosureAnalysisError("cache manifest does not rebuild exactly")
    return manifest


def _key_digest(value: Mapping[str, object]) -> str:
    fields = (
        "reference_scene_id",
        "master_sequence_id",
        "order_id",
        "horizon",
        "scan_indices",
        "history_scan_ids",
    )
    if any(field not in value for field in fields):
        raise ReviewerClosureAnalysisError("cache key lacks shared prefix identity")
    return hashlib.sha256(
        _canonical_json_bytes({field: value[field] for field in fields})
    ).hexdigest()


def run_replay_task_drift_audit(
    *,
    reference_manifest_path: str | Path,
    replay_manifest_path: str | Path,
    replay_entry_root: str | Path,
    sidecar_manifest_path: str | Path,
    system_aggregate_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    reference_manifest = _validate_cache_manifest_file(reference_manifest_path)
    replay_manifest = _validate_cache_manifest_file(replay_manifest_path)
    if reference_manifest.get("entry_count") != 43 * 3 * 5:
        raise ReviewerClosureAnalysisError("reference cache coverage differs")
    if replay_manifest.get("entry_count") != 43 * 3 * 4:
        raise ReviewerClosureAnalysisError("replay cache coverage differs")
    reference_entries = {
        _key_digest(entry["key"]): entry
        for entry in reference_manifest["entries"]
        if int(entry["key"]["horizon"]) in HORIZONS
    }
    replay_entries = {
        _key_digest(entry["key"]): entry for entry in replay_manifest["entries"]
    }
    if set(reference_entries) != set(replay_entries) or len(replay_entries) != 516:
        raise ReviewerClosureAnalysisError("reference/replay O2-O5 keys differ")

    sidecar_manifest = _load_json(
        sidecar_manifest_path, name="Full-History sidecar manifest"
    )
    _manifest_content_sha256(sidecar_manifest)
    source_binding = sidecar_manifest.get("source_prediction_manifest")
    replay_binding = sidecar_manifest.get("replay_prediction_manifest")
    if (
        sidecar_manifest.get("entry_count") != 516
        or not isinstance(source_binding, Mapping)
        or not isinstance(replay_binding, Mapping)
        or source_binding.get("content_sha256") != reference_manifest["content_sha256"]
        or replay_binding.get("content_sha256") != replay_manifest["content_sha256"]
    ):
        raise ReviewerClosureAnalysisError("sidecar cache-manifest binding differs")
    sidecar_entries = sidecar_manifest.get("entries")
    if not isinstance(sidecar_entries, list):
        raise ReviewerClosureAnalysisError("sidecar manifest entries differ")
    sidecar_by_key = {_key_digest(entry["key"]): entry for entry in sidecar_entries}
    if set(sidecar_by_key) != set(replay_entries):
        raise ReviewerClosureAnalysisError("sidecar/replay keys differ")
    bitwise_equal_count = 0
    for identity, replay_entry in replay_entries.items():
        reference_entry = reference_entries[identity]
        sidecar_entry = sidecar_by_key[identity]
        if (
            sidecar_entry.get("source_prediction_content_sha256")
            != replay_entry["content_sha256"]
            or sidecar_entry.get("reference_prediction_content_sha256")
            != reference_entry["content_sha256"]
        ):
            raise ReviewerClosureAnalysisError("sidecar prediction binding differs")
        bitwise_equal_count += int(
            replay_entry["content_sha256"] == reference_entry["content_sha256"]
        )

    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import (
        CausalTaskAccumulator,
        causal_prefix_pair_from_payload,
    )

    accumulators = {horizon: CausalTaskAccumulator() for horizon in HORIZONS}
    replay_root = Path(replay_entry_root)
    for identity in sorted(replay_entries):
        entry = replay_entries[identity]
        payload = load_full_history_cache_entry(
            replay_root,
            entry,
            expected_provenance=replay_manifest["provenance"],
        )
        pair = causal_prefix_pair_from_payload(payload)
        accumulators[pair.horizon].update(pair)
    replay_metrics = {horizon: accumulators[horizon].compute() for horizon in HORIZONS}
    system_aggregate = _read_system_aggregate(Path(system_aggregate_path))
    reference_metrics = {
        horizon: {
            field: system_aggregate[("FullHistory", horizon)][field]
            for field in TASK_FIELDS
        }
        for horizon in HORIZONS
    }
    rows = build_task_drift_rows(reference_metrics, replay_metrics)
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "quantified",
        "policy": "frozen_reference_task_metrics_remain_primary",
        "reference_manifest_content_sha256": reference_manifest["content_sha256"],
        "replay_manifest_content_sha256": replay_manifest["content_sha256"],
        "sidecar_manifest_content_sha256": sidecar_manifest["content_sha256"],
        "entry_count": len(replay_entries),
        "bitwise_equal_prediction_count": bitwise_equal_count,
        "metric_row_count": len(rows),
        "maximum_absolute_difference": max(
            float(row["absolute_difference"]) for row in rows
        ),
    }
    summary["content_sha256"] = _content_sha256(summary)
    output = Path(output_root)
    _publish_exact(output / "full_history_replay_task_drift.csv", _csv_bytes(rows))
    _publish_exact(
        output / "full_history_replay_task_drift.json", _pretty_json_bytes(summary)
    )
    return summary


def _read_system_aggregate(path: Path) -> dict[tuple[str, int], Mapping[str, object]]:
    rows = _read_typed_csv(path)
    result = {}
    for row in rows:
        method = str(row.get("method"))
        order = str(row.get("order_id"))
        horizon = int(row.get("horizon", 0))
        if (
            method not in {"FullHistory", "Persist4D"}
            or order != "all"
            or horizon not in HORIZONS
        ):
            raise ReviewerClosureAnalysisError("system aggregate coverage differs")
        key = (method, horizon)
        if key in result:
            raise ReviewerClosureAnalysisError("system aggregate contains duplicates")
        result[key] = row
    if set(result) != {
        (method, horizon)
        for method in ("FullHistory", "Persist4D")
        for horizon in HORIZONS
    }:
        raise ReviewerClosureAnalysisError("system aggregate coverage differs")
    return result


def _aggregate_phase_i_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    system_aggregate: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    result = []
    for method_id in METHOD_IDS:
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["method_id"] == method_id and row["horizon"] == horizon
            ]
            if len(selected) != 43 * 3:
                raise ReviewerClosureAnalysisError("Phase-I aggregate coverage differs")
            source_method = "Persist4D" if method_id == "Persist4D" else "FullHistory"
            source = system_aggregate[(source_method, horizon)]
            result.append(
                {
                    "method_id": method_id,
                    "method": METHOD_NAMES[method_id],
                    "horizon": horizon,
                    "sequence_count": len(selected),
                    "task_metric_source": (
                        "frozen_persist4d_cache"
                        if method_id == "Persist4D"
                        else "frozen_full_history_cache"
                    ),
                    **_copy_metrics(source, TASK_FIELDS),
                    **aggregate_identity_metrics(selected),
                }
            )
    return result


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ReviewerClosureAnalysisError("cannot publish an empty CSV")
    fields = tuple(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ReviewerClosureAnalysisError("CSV row fields differ")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: "" if row[field] is None else row[field] for field in fields}
        )
    return stream.getvalue().encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _audit_markdown(
    *,
    raw: Mapping[str, object],
    selection: Mapping[str, object],
    aggregate: Sequence[Mapping[str, object]],
    bootstrap: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> bytes:
    selected = str(selection["selected_method_id"])
    selected_name = METHOD_NAMES[selected]
    lines = [
        "# Full-History Tracker Audit",
        "",
        f"Gate I: `{gate['classification']}`.",
        "",
        "## Frozen Scope",
        "",
        f"- Raw tracker artifact: `{raw['content_sha256']}`",
        "- Coverage: 43 masters x 3 orders x T2-T5.",
        "- Tracker observations begin at P2; Persist4D retains its frozen P1 initialization.",
        "- Task metrics are inherited unchanged from the frozen system-comparison caches; only identity assignment changes.",
        "- The separate replay task-drift CSV/JSON quantifies official-metric changes caused by audited cross-process CUDA sparse numerical variation; no post-hoc equivalence threshold is applied.",
        "",
        "## Strongest Simple Tracker",
        "",
        f"Selected `{selected}` ({selected_name}). The preregistered ranking minimizes pooled T4/T5 normalized ID-switch rate, then maximizes pooled gap-recovery recall, then uses method ID.",
        "B4 is diagnostic and was excluded from selection.",
        "",
        "## Replay Task Drift",
        "",
        "All 516 replay prediction content digests differ from the immutable reference predictions. `full_history_replay_task_drift.csv` reports signed, absolute, and relative official-metric drift; `full_history_replay_task_drift.json` binds both manifests. Frozen reference-cache task metrics remain primary.",
        "",
        "## Pooled Identity Results",
        "",
        "| Method | T | ID-switch rate | Gap recovery recall |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate:
        if int(row["horizon"]) not in GATE_HORIZONS:
            continue
        switch = row["normalized_id_switch_rate"]
        gap = row["gap_recovery_recall"]
        lines.append(
            f"| {row['method']} | {row['horizon']} | "
            f"{float(switch):.6f} | {float(gap):.6f} |"
        )
    lines.extend(
        (
            "",
            "## Paired Six-Cluster Evidence",
            "",
            "Differences are Persist4D minus the selected tracker. Negative is favorable for ID-switch rate; positive is favorable for gap recovery.",
            "",
            "| Metric | T | Clusters | Tracker | Persist4D | Difference | 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in bootstrap:
        lines.append(
            f"| {row['metric']} | {row['horizon']} | {row['cluster_count']}/{row['reference_scene_count']} | {float(row['tracker_mean']):.6f} | "
            f"{float(row['persist4d_mean']):.6f} | {float(row['difference']):+.6f} | "
            f"[{float(row['ci_lower']):+.6f}, {float(row['ci_upper']):+.6f}] |"
        )
    lines.extend(
        (
            "",
            "## Gate Decision",
            "",
            f"`{gate['classification']}` under the frozen CI + order + LOSO rule.",
            "",
        )
    )
    return ("\n".join(lines)).encode("utf-8")


def build_phase_i_artifacts(
    *,
    raw_path: str | Path,
    system_results_path: str | Path,
    output_root: str | Path,
) -> dict[str, object]:
    config = load_phase_i_analysis_config(PHASE_I_ANALYSIS_CONFIG_PATH)
    raw = load_tracker_raw_artifact(raw_path)
    if raw["content_sha256"] != config["source_tracker_artifact_content_sha256"]:
        raise ReviewerClosureAnalysisError("Phase-I source tracker artifact differs")
    system_path = Path(system_results_path)
    system_rows = read_system_per_sequence_results(system_path)
    rows = merge_phase_i_results(raw, system_rows)
    selection = select_strongest_simple_tracker(rows)
    selected = str(selection["selected_method_id"])
    bootstrap = paired_cluster_bootstrap(rows, tracker_method_id=selected)
    order = order_robustness(rows, tracker_method_id=selected)
    loso = leave_one_scene_out(rows, tracker_method_id=selected)
    gate = derive_gate_i(
        tracker_method_id=selected,
        bootstrap_rows=bootstrap,
        order_rows=order,
        loso_rows=loso,
    )
    system_aggregate = _read_system_aggregate(
        system_path.parent / "aggregate_results.csv"
    )
    aggregate = _aggregate_phase_i_rows(rows, system_aggregate=system_aggregate)
    output = Path(output_root)
    payloads = {
        "full_history_tracker_results.csv": _csv_bytes(rows),
        "full_history_tracker_aggregate.csv": _csv_bytes(aggregate),
        "full_history_tracker_cluster_bootstrap.csv": _csv_bytes(bootstrap),
        "full_history_tracker_loso.csv": _csv_bytes(loso),
        "full_history_tracker_order_robustness.csv": _csv_bytes(order),
        "full_history_tracker_selection.json": _pretty_json_bytes(selection),
        "gate_i.json": _pretty_json_bytes(gate),
        "FULL_HISTORY_TRACKER_AUDIT.md": _audit_markdown(
            raw=raw,
            selection=selection,
            aggregate=aggregate,
            bootstrap=bootstrap,
            gate=gate,
        ),
    }
    if gate["classification"] == "TRACKER_EXPLAINS_IDENTITY":
        payloads["TRIVIAL_TRACKER_CHALLENGE_REPORT.md"] = (
            "# Trivial Tracker Challenge Report\n\n"
            f"Gate I: `{gate['classification']}`. `{selected}` removes the preregistered robust Persist4D identity advantage.\n"
        ).encode("utf-8")
    for filename, payload in payloads.items():
        _publish_exact(output / filename, payload)
    return {
        "status": "pass",
        "row_count": len(rows),
        "selected_tracker_method_id": selected,
        "gate_i_classification": gate["classification"],
        "raw_content_sha256": raw["content_sha256"],
    }


__all__ = [
    "ORDERS",
    "TASK_FIELDS",
    "ReviewerClosureAnalysisError",
    "aggregate_identity_metrics",
    "build_task_drift_rows",
    "build_phase_i_artifacts",
    "derive_gate_i",
    "leave_one_scene_out",
    "load_phase_i_analysis_config",
    "load_tracker_raw_artifact",
    "merge_phase_i_results",
    "order_robustness",
    "paired_cluster_bootstrap",
    "read_system_per_sequence_results",
    "run_replay_task_drift_audit",
    "select_strongest_simple_tracker",
]
