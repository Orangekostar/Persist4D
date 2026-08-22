"""Paired cluster statistics and cached system-comparison evaluation."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

METHODS = ("FullHistory", "Persist4D")
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)


class AnalysisError(ValueError):
    """Raised when paired analysis inputs violate the frozen protocol."""


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnalysisError(f"{name} must be a nonempty string")
    return value


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in HORIZONS:
        raise AnalysisError("horizon must be one of T2-T5")
    return value


def _finite_or_missing(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{name} must be finite or explicitly missing")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{name} must be finite or explicitly missing")
    return result


def _identity(row: Mapping[str, object]) -> tuple[str, str, str, str, int]:
    method = _string(row.get("method"), name="method")
    if method not in METHODS:
        raise AnalysisError("method coverage differs from the frozen comparison")
    reference = _string(row.get("reference_scene_id"), name="reference_scene_id")
    master = _string(row.get("master_sequence_id"), name="master_sequence_id")
    order = _string(row.get("order_id"), name="order_id")
    if order not in ORDERS:
        raise AnalysisError("order coverage differs from the frozen comparison")
    return method, reference, master, order, _horizon(row.get("horizon"))


def validate_result_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_master_count: int = 43,
) -> dict[str, int]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise AnalysisError("result coverage must be a nonempty sequence")
    if (
        isinstance(expected_master_count, bool)
        or not isinstance(expected_master_count, int)
        or expected_master_count <= 0
    ):
        raise ValueError("expected_master_count must be positive")
    identities = [_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise AnalysisError("result coverage contains duplicate cells")
    masters = {(reference, master) for _, reference, master, _, _ in identities}
    references_by_master: dict[str, set[str]] = defaultdict(set)
    for _method, reference, master, _order, _horizon_value in identities:
        references_by_master[master].add(reference)
    if any(len(values) != 1 for values in references_by_master.values()):
        raise AnalysisError("master-to-reference coverage is inconsistent")
    expected = {
        (method, reference, master, order, horizon)
        for reference, master in masters
        for method in METHODS
        for order in ORDERS
        for horizon in HORIZONS
    }
    if len(masters) != expected_master_count or set(identities) != expected:
        raise AnalysisError("result coverage is not exact")
    references = {reference for reference, _master in masters}
    if len(references) != 6:
        raise AnalysisError("result coverage must contain six reference scenes")
    return {
        "method_count": len(METHODS),
        "master_count": len(masters),
        "reference_scene_count": len(references),
        "order_count": len(ORDERS),
        "horizon_count": len(HORIZONS),
        "row_count": len(rows),
    }


def _paired_values(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    horizon: int,
) -> list[dict[str, object]]:
    _string(metric, name="metric")
    horizon = _horizon(horizon)
    by_pair: dict[tuple[str, str, str], dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        method, reference, master, order, row_horizon = _identity(row)
        if row_horizon != horizon:
            continue
        by_pair[(reference, master, order)][method] = _finite_or_missing(
            row.get(metric), name=f"{metric} value"
        )
    if not by_pair or any(set(values) != set(METHODS) for values in by_pair.values()):
        raise AnalysisError("paired metric coverage differs between methods")
    result = []
    for (reference, master, order), values in sorted(by_pair.items()):
        full = values["FullHistory"]
        persistent = values["Persist4D"]
        if full is None or persistent is None:
            continue
        result.append(
            {
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
                "full_history": full,
                "persist4d": persistent,
                "difference": persistent - full,
            }
        )
    if not result:
        raise AnalysisError(f"paired metric {metric} has no finite observations")
    return result


def paired_cluster_values(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    horizon: int,
    expected_reference_scene_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    pairs = _paired_values(rows, metric=metric, horizon=horizon)
    by_reference: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pairs:
        by_reference[str(row["reference_scene_id"])].append(row)
    clusters = []
    for reference, values in sorted(by_reference.items()):
        full = float(np.mean([float(row["full_history"]) for row in values]))
        persistent = float(np.mean([float(row["persist4d"]) for row in values]))
        clusters.append(
            {
                "reference_scene_id": reference,
                "pair_count": len(values),
                "full_history_mean": full,
                "persist4d_mean": persistent,
                "difference": persistent - full,
            }
        )
    if expected_reference_scene_ids is None:
        if len(clusters) != 6:
            raise AnalysisError("paired cluster analysis requires six reference scenes")
    else:
        expected_references = tuple(
            _string(value, name="expected reference scene")
            for value in expected_reference_scene_ids
        )
        actual_references = {
            str(row["reference_scene_id"]) for row in clusters
        }
        if (
            len(expected_references) != 6
            or len(set(expected_references)) != 6
            or not actual_references <= set(expected_references)
        ):
            raise AnalysisError("paired cluster analysis reference scenes differ")
    if not clusters:
        raise AnalysisError("paired cluster analysis requires six reference scenes")
    return clusters


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str],
    horizons: Sequence[int] = HORIZONS,
    replicates: int = 10_000,
    seed: int = 45,
    expected_reference_scene_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    horizon_values = tuple(_horizon(value) for value in horizons)
    if not horizon_values or tuple(sorted(set(horizon_values))) != horizon_values:
        raise AnalysisError("bootstrap horizons must be ordered unique T2-T5 values")
    if replicates != 10_000 or seed != 45:
        raise AnalysisError("bootstrap must use 10,000 replicates and seed 45")
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence) or not metrics:
        raise AnalysisError("bootstrap metrics must be a nonempty sequence")
    rng = np.random.default_rng(seed)
    result = []
    for metric in metrics:
        for horizon in horizon_values:
            clusters = paired_cluster_values(
                rows,
                metric=metric,
                horizon=horizon,
                expected_reference_scene_ids=expected_reference_scene_ids,
            )
            full = np.asarray(
                [float(row["full_history_mean"]) for row in clusters], dtype=np.float64
            )
            persistent = np.asarray(
                [float(row["persist4d_mean"]) for row in clusters], dtype=np.float64
            )
            differences = persistent - full
            indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
            samples = differences[indices].mean(axis=1)
            full_mean = float(full.mean())
            persistent_mean = float(persistent.mean())
            difference = persistent_mean - full_mean
            result.append(
                {
                    "metric": metric,
                    "horizon": horizon,
                    "cluster_count": len(clusters),
                    "bootstrap_replicates": replicates,
                    "seed": seed,
                    "full_history_mean": full_mean,
                    "persist4d_mean": persistent_mean,
                    "difference": difference,
                    "relative_difference": (
                        difference / abs(full_mean) if full_mean != 0 else None
                    ),
                    "ci_lower": float(np.quantile(samples, 0.025)),
                    "ci_upper": float(np.quantile(samples, 0.975)),
                }
            )
    return result


def leave_one_scene_out(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str],
    horizons: Sequence[int] = HORIZONS,
    expected_reference_scene_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence) or not metrics:
        raise AnalysisError("LOSO metrics must be a nonempty sequence")
    result = []
    for metric in metrics:
        for horizon in horizons:
            clusters = paired_cluster_values(
                rows,
                metric=metric,
                horizon=horizon,
                expected_reference_scene_ids=expected_reference_scene_ids,
            )
            dropped_references = (
                tuple(sorted(expected_reference_scene_ids))
                if expected_reference_scene_ids is not None
                else tuple(
                    str(row["reference_scene_id"]) for row in clusters
                )
            )
            for dropped_reference in dropped_references:
                kept = [
                    row
                    for row in clusters
                    if row["reference_scene_id"] != dropped_reference
                ]
                if not kept:
                    raise AnalysisError("LOSO has no finite clusters after dropping")
                difference = float(
                    np.mean([float(row["difference"]) for row in kept])
                )
                result.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "dropped_reference_scene_id": dropped_reference,
                        "remaining_cluster_count": len(kept),
                        "difference": difference,
                    }
                )
    result.sort(
        key=lambda row: (
            str(row["metric"]),
            int(row["horizon"]),
            str(row["dropped_reference_scene_id"]),
        )
    )
    return result


def detect_order_direction_reversal(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    horizon: int,
) -> dict[str, object]:
    pairs = _paired_values(rows, metric=metric, horizon=horizon)
    by_order: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        by_order[str(row["order_id"])].append(float(row["difference"]))
    if set(by_order) != set(ORDERS):
        raise AnalysisError("order-direction analysis lacks exact order coverage")
    differences = {
        order: float(np.mean(by_order[order])) for order in ORDERS
    }
    nonzero_signs = {math.copysign(1.0, value) for value in differences.values() if value}
    return {
        "metric": metric,
        "horizon": horizon,
        "differences_by_order": differences,
        "direction_reversal": len(nonzero_signs) > 1,
    }


_IDENTITY_COUNT_FIELDS = (
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
_TASK_FIELDS = (
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
_IDENTITY_RATE_FIELDS = (
    "normalized_id_switch_rate",
    "fragmentation_rate",
    "merge_rate",
    "gap_recovery_accuracy",
    "gap_recovery_recall",
)


def _count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnalysisError(f"{name} must be a non-negative integer")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_identity_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise AnalysisError("identity aggregation requires nonempty rows")
    totals = {
        field: sum(_count(row.get(field), name=field) for row in rows)
        for field in _IDENTITY_COUNT_FIELDS
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


def _validate_profile_coverage(rows: Sequence[Mapping[str, object]]) -> None:
    identities = [_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise AnalysisError("profile coverage contains duplicate cells")
    references = {identity[1] for identity in identities}
    masters = {(identity[1], identity[2]) for identity in identities}
    expected = {
        (method, reference, master, "canonical", horizon)
        for reference, master in masters
        for method in METHODS
        for horizon in HORIZONS
    }
    if len(references) != 6 or len(masters) != 6 or set(identities) != expected:
        raise AnalysisError("profile coverage is not the exact six-cluster subset")


def build_statistical_tables(
    per_sequence_rows: Sequence[Mapping[str, object]],
    profile_rows: Sequence[Mapping[str, object]],
    *,
    expected_master_count: int = 43,
) -> dict[str, list[dict[str, object]]]:
    validate_result_coverage(
        per_sequence_rows,
        expected_master_count=expected_master_count,
    )
    _validate_profile_coverage(profile_rows)
    sources = (
        ("causal_prefix_t_mAP", per_sequence_rows),
        ("causal_prefix_t_REC", per_sequence_rows),
        ("normalized_id_switch_rate", per_sequence_rows),
        ("gap_recovery_recall", per_sequence_rows),
        ("median_latency_ms", profile_rows),
    )
    references = sorted(
        {_identity(row)[1] for row in per_sequence_rows}
    )
    bootstrap_rows: list[dict[str, object]] = []
    loso_rows: list[dict[str, object]] = []
    for metric, rows in sources:
        for horizon in HORIZONS:
            expected_references = (
                references if metric == "gap_recovery_recall" else None
            )
            try:
                bootstrap_rows.extend(
                    paired_cluster_bootstrap(
                        rows,
                        metrics=(metric,),
                        horizons=(horizon,),
                        replicates=10_000,
                        seed=45,
                        expected_reference_scene_ids=expected_references,
                    )
                )
                loso_rows.extend(
                    leave_one_scene_out(
                        rows,
                        metrics=(metric,),
                        horizons=(horizon,),
                        expected_reference_scene_ids=expected_references,
                    )
                )
            except AnalysisError as error:
                if "no finite observations" not in str(error):
                    raise
                bootstrap_rows.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "cluster_count": 0,
                        "bootstrap_replicates": 10_000,
                        "seed": 45,
                        "full_history_mean": None,
                        "persist4d_mean": None,
                        "difference": None,
                        "relative_difference": None,
                        "ci_lower": None,
                        "ci_upper": None,
                    }
                )
                loso_rows.extend(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "dropped_reference_scene_id": reference,
                        "remaining_cluster_count": 0,
                        "difference": None,
                    }
                    for reference in references
                )
    order_rows = []
    for metric, rows in sources[:-1]:
        for horizon in HORIZONS:
            try:
                reversal = detect_order_direction_reversal(
                    rows,
                    metric=metric,
                    horizon=horizon,
                )
                differences = reversal.pop("differences_by_order")
                order_rows.append(
                    {
                        **reversal,
                        **{
                            f"{order}_difference": differences[order]
                            for order in ORDERS
                        },
                    }
                )
            except AnalysisError as error:
                if "no finite observations" not in str(error):
                    raise
                order_rows.append(
                    {
                        "metric": metric,
                        "horizon": horizon,
                        "direction_reversal": None,
                        **{f"{order}_difference": None for order in ORDERS},
                    }
                )
    return {
        "cluster_bootstrap": bootstrap_rows,
        "leave_one_scene_out": loso_rows,
        "order_robustness": order_rows,
    }


def _map_classes(values: Tensor, mapper: object) -> Tensor:
    if not callable(mapper):
        raise AnalysisError("class mapper must be callable")
    return torch.tensor(
        [mapper(int(value)) for value in values.detach().cpu().long().tolist()],
        dtype=torch.long,
    )


def _persistent_task_pair(
    *,
    payloads: Sequence[Mapping[str, object]],
    prediction: Mapping[str, object],
    horizon: int,
    class_mapper: object,
) -> object:
    from scripts.evaluate_persist4d_p6a import build_temporal_target
    from scripts.system_comparison_metrics import validate_causal_prefix_pair

    target = build_temporal_target(payloads[:horizon])
    key = payloads[horizon - 1]["key"]
    return validate_causal_prefix_pair(
        prediction={
            "pred_masks": prediction["pred_masks"],
            "pred_scores": prediction["pred_scores"],
            "pred_classes": _map_classes(prediction["pred_classes"], class_mapper),
        },
        target={
            "masks": target["masks"],
            "labels": _map_classes(target["labels"], class_mapper),
            "ids": target["ids"],
            "changes": target["changes"],
            "temporal_stages": target["temporal_stages"],
        },
        horizon=horizon,
        observed_scan_ids=key["history_scan_ids"],
    )


def _persistent_identity_updates(
    *,
    payloads: Sequence[Mapping[str, object]],
    steps: Sequence[object],
    class_mapper: object,
    background_class: int,
) -> tuple[object, ...]:
    from scripts.evaluate_persist4d_p6a import stage_prediction_from_track_step
    from scripts.system_comparison_metrics import match_identity_update

    updates = []
    for stage, (payload, step) in enumerate(zip(payloads, steps, strict=True)):
        prediction = stage_prediction_from_track_step(
            payload,
            step,
            class_mapper=class_mapper,
            background_class=background_class,
        )
        target = payload["target"]
        track_ids = prediction["track_ids"]
        if not isinstance(track_ids, Tensor):
            raise AnalysisError("Persist4D issued IDs must be integer tensors")
        updates.append(
            match_identity_update(
                horizon=stage + 1,
                gt_ids=target["gt_ids"],
                gt_classes=_map_classes(target["gt_classes"], class_mapper),
                gt_masks=target["gt_masks"],
                issued_ids=track_ids,
                pred_classes=prediction["pred_classes"],
                pred_masks=prediction["pred_masks"],
                minimum_iou=0.5,
            )
        )
    return tuple(updates)


def _task_accumulators() -> dict[tuple[str, str, int], object]:
    from scripts.system_comparison_metrics import CausalTaskAccumulator

    return {
        (method, order, horizon): CausalTaskAccumulator()
        for method in METHODS
        for order in (*ORDERS, "all")
        for horizon in HORIZONS
    }


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


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise AnalysisError("CSV row fields differ from the artifact contract")
        writer.writerow({field: "" if row[field] is None else row[field] for field in fields})
    return stream.getvalue().encode("utf-8")


_CSV_STRING_FIELDS = {
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "metric",
    "status",
}
_CSV_OPTIONAL_STRING_FIELDS = {"error_type", "error_message"}
_CSV_INTEGER_FIELDS = {
    "horizon",
    "sequence_count",
    "update_scan_count",
    "update_point_count",
    "cumulative_scan_count",
    "cumulative_point_count",
    "model_input_bytes",
    "cumulative_model_input_bytes",
    "persistent_state_bytes",
    "explicit_history_input_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    *_IDENTITY_COUNT_FIELDS,
}


def _read_typed_csv(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise AnalysisError(f"required CSV is unavailable: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise AnalysisError(f"required CSV cannot be decoded: {path}") from error
    if not raw_rows:
        raise AnalysisError(f"required CSV is empty: {path}")
    rows = []
    for raw in raw_rows:
        row: dict[str, object] = {}
        for field, value in raw.items():
            if field in _CSV_OPTIONAL_STRING_FIELDS:
                row[field] = value
            elif field in _CSV_STRING_FIELDS:
                row[field] = _string(value, name=field)
            elif value == "":
                row[field] = None
            elif field in _CSV_INTEGER_FIELDS:
                try:
                    row[field] = int(value)
                except ValueError as error:
                    raise AnalysisError(f"CSV field {field} is not an integer") from error
            else:
                try:
                    row[field] = float(value)
                except ValueError as error:
                    raise AnalysisError(f"CSV field {field} is not numeric") from error
        rows.append(row)
    return rows


def run_statistical_analysis(*, project_root: Path) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    if project_root.resolve() != repository:
        raise AnalysisError("project_root differs from the analysis repository")
    from scripts.run_system_comparison import (
        SYSTEM_ROOT,
        _load_bound_inputs,
        oracle_attribution_required,
    )

    _system_manifest, binding = _load_bound_inputs()
    per_sequence = _read_typed_csv(SYSTEM_ROOT / "per_sequence_results.csv")
    profile = _read_typed_csv(SYSTEM_ROOT / "profile_results.csv")
    aggregate = _read_typed_csv(SYSTEM_ROOT / "aggregate_results.csv")
    tables = build_statistical_tables(per_sequence, profile)
    bootstrap_fields = (
        "metric",
        "horizon",
        "cluster_count",
        "bootstrap_replicates",
        "seed",
        "full_history_mean",
        "persist4d_mean",
        "difference",
        "relative_difference",
        "ci_lower",
        "ci_upper",
    )
    loso_fields = (
        "metric",
        "horizon",
        "dropped_reference_scene_id",
        "remaining_cluster_count",
        "difference",
    )
    order_fields = (
        "metric",
        "horizon",
        "direction_reversal",
        "canonical_difference",
        "reverse_difference",
        "sha256_seed45_difference",
    )
    for filename, rows, fields in (
        ("cluster_bootstrap.csv", tables["cluster_bootstrap"], bootstrap_fields),
        ("leave_one_scene_out.csv", tables["leave_one_scene_out"], loso_fields),
        ("order_robustness.csv", tables["order_robustness"], order_fields),
    ):
        _publish_exact(SYSTEM_ROOT / filename, _csv_bytes(rows, fields))
    tmap_rows = {
        int(row["horizon"]): row
        for row in tables["cluster_bootstrap"]
        if row["metric"] == "causal_prefix_t_mAP"
    }
    aggregate_tmap = {
        (str(row["method"]), int(row["horizon"])): row["causal_prefix_t_mAP"]
        for row in aggregate
    }
    if set(aggregate_tmap) != {
        (method, horizon)
        for method in METHODS
        for horizon in HORIZONS
    }:
        raise AnalysisError("aggregate task results lack exact system/horizon coverage")
    oracle_required = oracle_attribution_required(
        persist4d={
            f"T{horizon}": aggregate_tmap[("Persist4D", horizon)]
            for horizon in (4, 5)
        },
        full_history={
            f"T{horizon}": aggregate_tmap[("FullHistory", horizon)]
            for horizon in (4, 5)
        },
        paired_ci={
            f"T{horizon}": (
                tmap_rows[horizon]["ci_lower"],
                tmap_rows[horizon]["ci_upper"],
            )
            for horizon in (4, 5)
        },
        minimum_advantage=0.01,
    )
    summary = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": binding["source_commit"],
        "bootstrap_row_count": len(tables["cluster_bootstrap"]),
        "loso_row_count": len(tables["leave_one_scene_out"]),
        "order_robustness_row_count": len(tables["order_robustness"]),
        "oracle_attribution_required": oracle_required,
    }
    _publish_exact(
        SYSTEM_ROOT / "statistics_summary.json",
        (
            json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return summary


def run_oracle_attribution(
    *,
    project_root: Path,
    metadata_path: Path,
) -> dict[str, float]:
    """Run the existing GT-association diagnostic only after its trigger fires."""

    repository = Path(__file__).resolve().parents[1]
    if project_root.resolve() != repository:
        raise AnalysisError("project_root differs from the Oracle repository")
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        evaluate_cached_task_metrics,
        load_cached_protocol_sequences,
        normalize_official_metric_blocks,
    )
    from scripts.run_system_comparison import (
        LOCAL_CACHE_MANIFEST,
        LOCAL_ENTRY_CACHE,
        SYSTEM_ROOT,
        _build_frozen_setup,
        _load_bound_inputs,
    )

    _system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=LOCAL_ENTRY_CACHE,
        manifest_path=LOCAL_CACHE_MANIFEST,
    )
    factories = build_tracker_factories(setup.p6a_config)
    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories={"B4": factories["B4"]},
        class_mapper=build_rio_class_mapper(setup.dataset),
        background_class=int(setup.p6a_config["baselines"]["b4"]["background_class"]),
    )
    normalized = normalize_official_metric_blocks(evaluation.metric_blocks)
    oracle = normalized["offline"]["Oracle"]
    fields = (
        "method",
        "horizon",
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "t_REC50",
        "t_REC25",
        "diagnostic_semantics",
    )
    rows = [
        {
            "method": "Oracle",
            "horizon": horizon,
            "t_mAP": oracle[f"T{horizon}"]["t_mAP"],
            "t_mAP50": oracle[f"T{horizon}"]["t_mAP50"],
            "t_mAP25": oracle[f"T{horizon}"]["t_mAP25"],
            "t_REC": oracle[f"T{horizon}"]["t_REC"],
            "t_REC50": oracle[f"T{horizon}"]["t_REC50"],
            "t_REC25": oracle[f"T{horizon}"]["t_REC25"],
            "diagnostic_semantics": "GT-association post-hoc upper bound",
        }
        for horizon in HORIZONS
    ]
    _publish_exact(
        SYSTEM_ROOT / "oracle_attribution.csv",
        _csv_bytes(rows, fields),
    )
    return {f"T{horizon}": float(oracle[f"T{horizon}"]["t_mAP"]) for horizon in HORIZONS}


def _identity_fields(metrics: Mapping[str, object]) -> dict[str, object]:
    return {
        field: metrics[field]
        for field in (*_IDENTITY_COUNT_FIELDS, *_IDENTITY_RATE_FIELDS)
    }


def run_cached_system_evaluation(
    *,
    project_root: Path,
    metadata_path: Path,
) -> dict[str, object]:
    """Evaluate both frozen systems from validated caches without CUDA."""

    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        load_cached_protocol_sequences,
        prefix_causality_coordinator,
    )
    from scripts.run_system_comparison import (
        FULL_CACHE_MANIFEST,
        FULL_ENTRY_CACHE,
        LOCAL_CACHE_MANIFEST,
        LOCAL_ENTRY_CACHE,
        SYSTEM_ROOT,
        _build_frozen_setup,
        _load_bound_inputs,
        _run_incumbent_regression,
        _validated_full_manifest,
    )
    from scripts.system_comparison_inference import (
        load_full_history_cache_entry,
    )
    from scripts.system_comparison_metrics import (
        causal_prefix_pair_from_payload,
        compute_causal_task_metrics,
        deployment_identity_metrics_by_horizon,
        identity_updates_from_payloads,
    )

    repository = Path(__file__).resolve().parents[1]
    if project_root.resolve() != repository:
        raise AnalysisError("project_root differs from the evaluation repository")
    system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    incumbent_gate = _run_incumbent_regression(setup)
    local_sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=LOCAL_ENTRY_CACHE,
        manifest_path=LOCAL_CACHE_MANIFEST,
    )
    if len(local_sequences) != 43 * 3:
        raise AnalysisError("persistent cache must contain 129 exact sequences")
    full_manifest = _validated_full_manifest(
        system_manifest=system_manifest,
        provenance=setup.full_provenance,
    )
    if FULL_CACHE_MANIFEST.parent != FULL_ENTRY_CACHE.parent:
        raise AnalysisError("full-history cache layout differs")
    full_entries: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for entry in full_manifest["entries"]:
        key = entry["key"]
        identity = (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["horizon"]),
        )
        if identity in full_entries:
            raise AnalysisError("full-history manifest contains duplicate sequence cells")
        full_entries[identity] = entry

    class_mapper = build_rio_class_mapper(setup.dataset)
    factory = build_tracker_factories(setup.p6a_config)["B4"]
    background_class = int(
        setup.p6a_config["baselines"]["b4"]["background_class"]
    )
    task_accumulators = _task_accumulators()
    per_sequence_rows: list[dict[str, object]] = []

    for sequence in local_sequences:
        scope = (sequence.master_sequence_id, sequence.order_id)
        full_payloads = tuple(
            load_full_history_cache_entry(
                FULL_ENTRY_CACHE,
                full_entries[(*scope, horizon)],
                expected_provenance=setup.full_provenance,
            )
            for horizon in range(1, 6)
        )
        coordinated = prefix_causality_coordinator(
            sequence.payloads,
            {"B4": factory},
            endpoints=(1, 2, 3, 4),
            sequence_id=f"{sequence.master_sequence_id}:{sequence.order_id}",
            background_class=background_class,
        )
        persistent_updates = _persistent_identity_updates(
            payloads=sequence.payloads,
            steps=coordinated.offline_steps["B4"],
            class_mapper=class_mapper,
            background_class=background_class,
        )
        full_updates = identity_updates_from_payloads(full_payloads, minimum_iou=0.5)
        persistent_identity = deployment_identity_metrics_by_horizon(
            persistent_updates
        )
        full_identity = deployment_identity_metrics_by_horizon(full_updates)

        for horizon in HORIZONS:
            persistent_pair = _persistent_task_pair(
                payloads=sequence.payloads,
                prediction=coordinated.online_predictions["B4"][horizon - 1],
                horizon=horizon,
                class_mapper=class_mapper,
            )
            full_pair = causal_prefix_pair_from_payload(full_payloads[horizon - 1])
            for method, pair, identity_metrics in (
                ("Persist4D", persistent_pair, persistent_identity[horizon]),
                ("FullHistory", full_pair, full_identity[horizon]),
            ):
                task_metrics = compute_causal_task_metrics([pair])
                task_accumulators[(method, sequence.order_id, horizon)].update(pair)
                task_accumulators[(method, "all", horizon)].update(pair)
                if method == "FullHistory":
                    stats = full_payloads[horizon - 1]["input_stats"]
                    update_scans = int(stats["scan_count"])
                    update_points = int(stats["full_point_count"])
                    cumulative_scans = horizon * (horizon + 1) // 2 - 1
                else:
                    target = sequence.payloads[horizon - 1]["target"]
                    update_scans = len(
                        sequence.payloads[horizon - 1]["key"]["local_window_scan_ids"]
                    )
                    update_points = int(target["gt_masks"].shape[1])
                    cumulative_scans = 2 * (horizon - 1)
                per_sequence_rows.append(
                    {
                        "method": method,
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "horizon": horizon,
                        **task_metrics,
                        **_identity_fields(identity_metrics),
                        "update_scan_count": update_scans,
                        "update_point_count": update_points,
                        "cumulative_scan_count": cumulative_scans,
                    }
                )

    per_sequence_rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            str(row["method"]),
        )
    )
    coverage = validate_result_coverage(per_sequence_rows)
    aggregate_rows = []
    for order in (*ORDERS, "all"):
        for horizon in HORIZONS:
            for method in METHODS:
                selected = [
                    row
                    for row in per_sequence_rows
                    if row["method"] == method
                    and row["horizon"] == horizon
                    and (order == "all" or row["order_id"] == order)
                ]
                aggregate_rows.append(
                    {
                        "method": method,
                        "order_id": order,
                        "horizon": horizon,
                        "sequence_count": len(selected),
                        **task_accumulators[(method, order, horizon)].compute(),
                        **aggregate_identity_metrics(selected),
                    }
                )
    aggregate_rows.sort(
        key=lambda row: (
            str(row["order_id"]), int(row["horizon"]), str(row["method"])
        )
    )
    per_order_rows = [row for row in aggregate_rows if row["order_id"] != "all"]
    overall_rows = [row for row in aggregate_rows if row["order_id"] == "all"]

    sequence_fields = (
        "method",
        "reference_scene_id",
        "master_sequence_id",
        "order_id",
        "horizon",
        *_TASK_FIELDS,
        *_IDENTITY_COUNT_FIELDS,
        *_IDENTITY_RATE_FIELDS,
        "update_scan_count",
        "update_point_count",
        "cumulative_scan_count",
    )
    aggregate_fields = (
        "method",
        "order_id",
        "horizon",
        "sequence_count",
        *_TASK_FIELDS,
        *_IDENTITY_COUNT_FIELDS,
        *_IDENTITY_RATE_FIELDS,
    )
    outputs = {
        SYSTEM_ROOT / "per_sequence_results.csv": _csv_bytes(
            per_sequence_rows, sequence_fields
        ),
        SYSTEM_ROOT / "per_order_results.csv": _csv_bytes(
            per_order_rows, aggregate_fields
        ),
        SYSTEM_ROOT / "aggregate_results.csv": _csv_bytes(
            overall_rows, aggregate_fields
        ),
    }
    for path, payload in outputs.items():
        _publish_exact(path, payload)
    result = {
        "schema_version": 1,
        "status": "pass",
        "source_commit": binding["source_commit"],
        "coverage": coverage,
        "incumbent_regression": incumbent_gate,
        "per_order_row_count": len(per_order_rows),
        "aggregate_row_count": len(overall_rows),
    }
    _publish_exact(
        SYSTEM_ROOT / "cached_evaluation.json",
        (
            json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return result


__all__ = [
    "AnalysisError",
    "aggregate_identity_metrics",
    "build_statistical_tables",
    "detect_order_direction_reversal",
    "leave_one_scene_out",
    "paired_cluster_bootstrap",
    "paired_cluster_values",
    "run_cached_system_evaluation",
    "run_oracle_attribution",
    "run_statistical_analysis",
    "validate_result_coverage",
]
