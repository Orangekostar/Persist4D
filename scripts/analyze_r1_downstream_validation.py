#!/usr/bin/env python3
"""Analyze frozen R1 downstream validation caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
DEFAULT_CACHE_ROOT = Path("/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache")
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_DATA_ROOT = Path("/home/ww/paper5")
OLD_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/system_comparison_v2_full"
)
OLD_CACHE_MANIFEST = PROJECT_ROOT / "artifacts/system_comparison_v2/cache_manifest.json"
OLD_TASK_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/score_sensitivity"
OLD_IDENTITY_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/identity"
OLD_FULL_ROOT = PROJECT_ROOT / "artifacts/system_comparison_v2"

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
EVENT_COUNT_FIELDS = (
    "association_event_count",
    "new_birth_count",
    "false_birth_count",
    "birth_rejected_count",
)
RECOVERY_HORIZONS = (4, 5)
ORDERS = ("canonical", "reverse", "sha256_seed45")
REPORT_HORIZONS = (2, 4, 5)
TASK_BRANCHES = (
    ("FullHistory", "official"),
    *(
        (method, reducer)
        for method in ("B2", "B4")
        for reducer in ("mean", "latest", "max")
    ),
)
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
IDENTITY_RATE_FIELDS = (
    "normalized_id_switch_rate",
    "fragmentation_rate",
    "merge_rate",
    "recovery_attempt_coverage",
    "gap_recovery_accuracy",
    "gap_recovery_recall",
)
LOCAL_FIELDS = (
    "local_current_AP",
    "local_current_AP50",
    "local_current_AP25",
    "local_current_REC",
)


class R1AnalysisError(ValueError):
    """Raised when analysis inputs violate the frozen R1 contract."""


def resolve_metric_dataset_spec(data_root: Path) -> Path:
    root = data_root.expanduser().resolve(strict=True)
    specification = root / "data/processed/rio/rio.yaml"
    if not specification.is_file():
        raise R1AnalysisError("metric dataset spec is unavailable under data root")
    return specification.resolve(strict=True)


def _non_negative_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise R1AnalysisError(f"{field} must be a non-negative integer")
    return value


def _finite_metric(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R1AnalysisError(f"{field} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise R1AnalysisError(f"{field} must be finite")
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_identity_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    """Pool identity counts before deriving rates."""

    if not rows:
        raise R1AnalysisError("identity aggregation requires rows")
    totals = {
        field: sum(_non_negative_count(row.get(field), field=field) for row in rows)
        for field in (*IDENTITY_COUNT_FIELDS, *EVENT_COUNT_FIELDS)
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
        "recovery_attempt_coverage": _rate(
            totals["recovery_attempts"], totals["gap_opportunities"]
        ),
        "gap_recovery_accuracy": _rate(
            totals["correct_recoveries"], totals["recovery_attempts"]
        ),
        "gap_recovery_recall": _rate(
            totals["correct_recoveries"], totals["gap_opportunities"]
        ),
    }


def _row_scope(row: Mapping[str, object]) -> tuple[str, str, str]:
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
        raise R1AnalysisError("per-sequence scope labels differ")
    return reference, master, str(order)


def validate_per_sequence_coverage(
    *,
    task_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    local_rows: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Validate exact frozen Protocol-B analysis cells."""

    if not task_rows or not identity_rows or not local_rows:
        raise R1AnalysisError("per-sequence coverage is empty")
    scopes = {_row_scope(row) for row in local_rows}
    masters = {scope[1] for scope in scopes}
    references = {scope[0] for scope in scopes}
    if (
        len(scopes) != 129
        or len(masters) != 43
        or len(references) != 6
        or {scope[2] for scope in scopes} != set(ORDERS)
        or any(
            len({scope[0] for scope in scopes if scope[1] == master}) != 1
            for master in masters
        )
        or any(
            {scope[2] for scope in scopes if scope[1] == master} != set(ORDERS)
            for master in masters
        )
    ):
        raise R1AnalysisError("Protocol-B sequence coverage differs")

    local_cells = [(_row_scope(row), row.get("horizon")) for row in local_rows]
    expected_local = {
        (scope, horizon) for scope in scopes for horizon in REPORT_HORIZONS
    }
    if len(local_cells) != len(set(local_cells)) or set(local_cells) != expected_local:
        raise R1AnalysisError("local coverage differs")

    task_cells = [
        (
            _row_scope(row),
            row.get("horizon"),
            str(row.get("method")),
            str(row.get("score_reducer")),
        )
        for row in task_rows
    ]
    expected_task = {
        (scope, horizon, method, reducer)
        for scope in scopes
        for horizon in REPORT_HORIZONS
        for method, reducer in TASK_BRANCHES
    }
    if len(task_cells) != len(set(task_cells)) or set(task_cells) != expected_task:
        raise R1AnalysisError("task coverage differs")

    identity_cells = [
        (_row_scope(row), row.get("horizon"), str(row.get("method")))
        for row in identity_rows
    ]
    expected_identity = {
        (scope, horizon, method)
        for scope in scopes
        for horizon in REPORT_HORIZONS
        for method in ("B2", "B4")
    }
    if (
        len(identity_cells) != len(set(identity_cells))
        or set(identity_cells) != expected_identity
    ):
        raise R1AnalysisError("identity coverage differs")
    return {
        "master_count": len(masters),
        "sequence_count": len(scopes),
        "reference_cluster_count": len(references),
        "task_row_count": len(task_rows),
        "identity_row_count": len(identity_rows),
        "local_row_count": len(local_rows),
    }


def _unique_index(
    rows: Sequence[Mapping[str, object]],
    *,
    horizon: int,
    include_cluster: bool,
) -> dict[tuple[str, ...], Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("order_id") == "all" and row.get("horizon") == horizon
    ]
    result: dict[tuple[str, ...], Mapping[str, object]] = {}
    for row in selected:
        method = str(row.get("method"))
        if method not in {"B2", "B4"}:
            continue
        key = (
            (str(row.get("reference_scene_id")), method)
            if include_cluster
            else (method,)
        )
        if key in result:
            raise R1AnalysisError("recovery inputs contain duplicate cells")
        result[key] = row
    return result


def classify_recovery(
    aggregate_rows: Sequence[Mapping[str, object]],
    cluster_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Apply the frozen pooled-positive and four-of-six recovery rule."""

    decision: dict[str, object] = {}
    supported = True
    for horizon in RECOVERY_HORIZONS:
        aggregate = _unique_index(
            aggregate_rows, horizon=horizon, include_cluster=False
        )
        if set(aggregate) != {("B2",), ("B4",)}:
            raise R1AnalysisError(f"T{horizon} pooled recovery coverage is incomplete")
        b2 = _finite_metric(
            aggregate[("B2",)].get("gap_recovery_recall"),
            field="gap_recovery_recall",
        )
        b4 = _finite_metric(
            aggregate[("B4",)].get("gap_recovery_recall"),
            field="gap_recovery_recall",
        )
        clusters = _unique_index(
            cluster_rows, horizon=horizon, include_cluster=True
        )
        references = sorted({key[0] for key in clusters})
        if (
            len(references) != 6
            or set(clusters)
            != {(reference, method) for reference in references for method in ("B2", "B4")}
        ):
            raise R1AnalysisError(
                f"T{horizon} recovery decision requires six paired clusters"
            )
        deltas = []
        for reference in references:
            baseline = _finite_metric(
                clusters[(reference, "B2")].get("gap_recovery_recall"),
                field="gap_recovery_recall",
            )
            candidate = _finite_metric(
                clusters[(reference, "B4")].get("gap_recovery_recall"),
                field="gap_recovery_recall",
            )
            deltas.append(candidate - baseline)
        pooled_delta = b4 - b2
        positive_count = sum(delta > 0.0 for delta in deltas)
        horizon_supported = pooled_delta > 0.0 and positive_count >= 4
        supported = supported and horizon_supported
        decision[f"T{horizon}"] = {
            "pooled_b2": b2,
            "pooled_b4": b4,
            "pooled_b4_minus_b2": pooled_delta,
            "positive_cluster_count": positive_count,
            "valid_cluster_count": len(deltas),
            "supported": horizon_supported,
        }
    decision["status"] = (
        "RECOVERY_SUPPORTED" if supported else "RECOVERY_NOT_SUPPORTED"
    )
    return decision


def paired_cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    horizon: int,
    score_reducer: str | None,
    replicates: int,
    seed: int,
) -> dict[str, int | float | str]:
    """Bootstrap six paired B4-minus-B2 cluster effects."""

    if replicates != 10_000 or seed != 45:
        raise R1AnalysisError("paired bootstrap requires 10,000 replicates and seed 45")
    selected = [
        row
        for row in rows
        if row.get("order_id") == "all"
        and row.get("horizon") == horizon
        and (score_reducer is None or row.get("score_reducer") == score_reducer)
        and row.get("method") in {"B2", "B4"}
    ]
    index: dict[tuple[str, str], float] = {}
    for row in selected:
        key = (str(row.get("reference_scene_id")), str(row.get("method")))
        if key in index:
            raise R1AnalysisError("bootstrap inputs contain duplicate cells")
        index[key] = _finite_metric(row.get(metric), field=metric)
    references = sorted({key[0] for key in index})
    expected = {
        (reference, method) for reference in references for method in ("B2", "B4")
    }
    if len(references) != 6 or set(index) != expected:
        raise R1AnalysisError("paired bootstrap requires exactly six paired clusters")
    deltas = np.asarray(
        [index[(reference, "B4")] - index[(reference, "B2")] for reference in references],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(deltas), size=(replicates, len(deltas)))
    bootstrap = deltas[samples].mean(axis=1)
    return {
        "metric": metric,
        "horizon": horizon,
        "score_reducer": (
            score_reducer if score_reducer is not None else "not_applicable"
        ),
        "cluster_count": len(deltas),
        "positive_cluster_count": int(np.count_nonzero(deltas > 0.0)),
        "b4_minus_b2": float(deltas.mean()),
        "ci_lower": float(np.quantile(bootstrap, 0.025)),
        "ci_upper": float(np.quantile(bootstrap, 0.975)),
        "replicates": replicates,
        "seed": seed,
    }


def gap_event_rows(events: Sequence[object]) -> list[dict[str, object]]:
    """Keep only gap opportunities and attempted reactivations."""

    rows = []
    for event in events:
        if isinstance(event, Mapping):
            row = dict(event)
        elif is_dataclass(event) and not isinstance(event, type):
            row = asdict(event)
        else:
            raise R1AnalysisError("gap ledger events must be mappings or dataclasses")
        opportunity = row.get("gap_opportunity")
        attempt = row.get("reactivation_attempt")
        if not isinstance(opportunity, bool) or not isinstance(attempt, bool):
            raise R1AnalysisError("gap ledger flags must be boolean")
        if opportunity or attempt:
            row["attempt_coverage_event"] = attempt
            rows.append(row)
    return rows


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise R1AnalysisError(f"required JSON is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise R1AnalysisError(f"required JSON cannot be decoded: {path}") from error
    if not isinstance(value, Mapping):
        raise R1AnalysisError(f"required JSON must contain a mapping: {path}")
    return dict(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise R1AnalysisError(f"required CSV is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise R1AnalysisError(f"required CSV is empty: {path}")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise R1AnalysisError("CSV output rows must not be empty")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: "" if row.get(field) is None else row.get(field) for field in fields}
        for row in rows
    )
    return stream.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _git_head() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise R1AnalysisError("Git HEAD is invalid")
    return value


def _task_metric_block(accumulator: object) -> dict[str, float]:
    compute = getattr(accumulator, "compute", None)
    if not callable(compute):
        raise R1AnalysisError("task accumulator is invalid")
    values = compute()
    return {field: float(values[field]) for field in TASK_FIELDS}


def _local_metric_block(accumulator: object) -> dict[str, float]:
    compute = getattr(accumulator, "compute", None)
    if not callable(compute):
        raise R1AnalysisError("local accumulator is invalid")
    values = compute()
    return {
        "local_current_AP": float(values["raw_local_AP"]),
        "local_current_AP50": float(values["raw_local_AP50"]),
        "local_current_AP25": float(values["raw_local_AP25"]),
        "local_current_REC": float(values["raw_local_REC"]),
    }


def _task_accumulator(
    values: dict[tuple[object, ...], object],
    key: tuple[object, ...],
    *,
    dataset_spec: Path,
) -> object:
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.system_comparison_metrics import CausalTaskAccumulator

    if key not in values:
        values[key] = CausalTaskAccumulator(
            metric_factory=lambda mode: OfficialMetricAccumulator(
                mode=mode, dataset_spec=dataset_spec
            )
        )
    return values[key]


def _local_accumulator(
    values: dict[tuple[object, ...], object],
    key: tuple[object, ...],
    *,
    dataset_spec: Path,
) -> object:
    from scripts.p6a_metrics import OfficialMetricAccumulator

    if key not in values:
        values[key] = OfficialMetricAccumulator(
            mode="raw_local", dataset_spec=dataset_spec
        )
    return values[key]


def _single_task_metrics(pair: object, *, dataset_spec: Path) -> dict[str, float]:
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.system_comparison_metrics import CausalTaskAccumulator

    accumulator = CausalTaskAccumulator(
        metric_factory=lambda mode: OfficialMetricAccumulator(
            mode=mode, dataset_spec=dataset_spec
        )
    )
    accumulator.update(pair)
    return _task_metric_block(accumulator)


def _event_counts(events: Sequence[object], *, horizon: int) -> dict[str, int]:
    selected = []
    for event in events:
        stage = event.get("stage_id") if isinstance(event, Mapping) else getattr(event, "stage_id")
        if int(stage) < horizon:
            selected.append(event)

    def flag(event: object, name: str) -> bool:
        value = event.get(name) if isinstance(event, Mapping) else getattr(event, name)
        return value is True

    return {
        "association_event_count": len(selected),
        "new_birth_count": sum(flag(event, "new_birth") for event in selected),
        "false_birth_count": sum(flag(event, "false_birth") for event in selected),
        "birth_rejected_count": sum(
            flag(event, "birth_rejected") for event in selected
        ),
    }


def _identity_with_attempt_coverage(
    values: Mapping[str, object],
) -> dict[str, object]:
    result = dict(values)
    attempts = _non_negative_count(
        result.get("recovery_attempts"), field="recovery_attempts"
    )
    gaps = _non_negative_count(
        result.get("gap_opportunities"), field="gap_opportunities"
    )
    result["recovery_attempt_coverage"] = _rate(attempts, gaps)
    return result


def _validate_new_cache_binding(
    *, cache_root: Path, artifact_root: Path
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    compact = _read_json(artifact_root / "cache_manifest.json")
    local = _read_json(cache_root / "local_progress.json")
    full = _read_json(cache_root / "full_history_progress.json")
    if compact.get("status") != "pass":
        raise R1AnalysisError("compact R1 cache manifest did not pass")
    for progress, kind in ((local, "local"), (full, "full_history")):
        records = progress.get("records")
        if (
            progress.get("status") != "pass"
            or progress.get("cache_kind") != kind
            or isinstance(records, (str, bytes))
            or not isinstance(records, Sequence)
            or len(records) != 645
        ):
            raise R1AnalysisError(f"{kind} cache progress coverage differs")
    protocol_sha = compact.get("protocol_sha256")
    common = ("source_commit", "checkpoint_sha256", "config_sha256")
    for progress in (local, full):
        provenance = progress.get("provenance")
        if not isinstance(provenance, Mapping):
            raise R1AnalysisError("cache progress provenance is invalid")
        if any(provenance.get(field) != compact.get(field) for field in common):
            raise R1AnalysisError("cache progress provenance differs from manifest")
        if progress.get("protocol_manifest_sha256") != protocol_sha:
            raise R1AnalysisError("cache progress protocol differs from manifest")
    if (
        _canonical_sha256(local["records"])
        != compact.get("local", {}).get("records_sha256")
        or _canonical_sha256(full["records"])
        != compact.get("full_history", {}).get("records_sha256")
    ):
        raise R1AnalysisError("cache progress record digest differs from manifest")
    return compact, local, full


def _identity_tables(
    rows: Sequence[Mapping[str, object]], references: Sequence[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aggregate_rows = []
    cluster_rows = []
    for order in (*ORDERS, "all"):
        for horizon in REPORT_HORIZONS:
            for method in ("B2", "B4"):
                selected = [
                    row
                    for row in rows
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
                        **aggregate_identity_rows(selected),
                    }
                )
                for reference in references:
                    cluster = [
                        row
                        for row in selected
                        if row["reference_scene_id"] == reference
                    ]
                    if not cluster:
                        raise R1AnalysisError("identity cluster coverage is incomplete")
                    cluster_rows.append(
                        {
                            "method": method,
                            "reference_scene_id": reference,
                            "order_id": order,
                            "horizon": horizon,
                            "sequence_count": len(cluster),
                            **aggregate_identity_rows(cluster),
                            "inference_unit": "reference_scene_id",
                        }
                    )
    if len(aggregate_rows) != 24 or len(cluster_rows) != 144:
        raise R1AnalysisError("identity aggregate coverage differs")
    return aggregate_rows, cluster_rows


def _historical_replay_status(*, source_commit: str) -> dict[str, object]:
    progress_path = OLD_CACHE_ROOT / "cache_progress.json"
    progress = _read_json(progress_path)
    compact = _read_json(OLD_CACHE_MANIFEST)
    task_manifest = _read_json(OLD_TASK_ROOT / "manifest.json")
    identity_manifest = _read_json(OLD_IDENTITY_ROOT / "manifest.json")
    records = progress.get("records")
    compact_records = compact.get("records")
    metadata_exact = (
        progress.get("status") == "pass"
        and isinstance(records, Sequence)
        and not isinstance(records, (str, bytes))
        and len(records) == 645
        and compact.get("status") == "pass"
        and compact.get("entry_count") == 645
        and records == compact_records
        and _canonical_sha256(records) == compact.get("records_sha256")
        and progress.get("checkpoint_sha256") == compact.get("checkpoint_sha256")
        and progress.get("protocol_manifest_sha256")
        == compact.get("protocol_manifest_sha256")
        and progress.get("source_commit") == compact.get("source_commit")
        and progress.get("score_reducer") == compact.get("score_reducer")
    )
    raw_directory = OLD_CACHE_ROOT / "raw_predictions/entries"
    sidecar_directory = OLD_CACHE_ROOT / "task_sidecars/entries"
    available_raw = 0
    available_sidecars = 0
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw_entry = record.get("raw_entry")
            sidecar_entry = record.get("sidecar_entry")
            if isinstance(raw_entry, Mapping) and (
                raw_directory / str(raw_entry.get("filename"))
            ).is_file():
                available_raw += 1
            if isinstance(sidecar_entry, Mapping) and (
                sidecar_directory / str(sidecar_entry.get("filename"))
            ).is_file():
                available_sidecars += 1
    summaries_exact = (
        task_manifest.get("status") == "pass"
        and identity_manifest.get("status") == "pass"
        and task_manifest.get("inputs", {}).get("checkpoint_sha256")
        == compact.get("checkpoint_sha256")
        and task_manifest.get("inputs", {}).get("protocol_manifest_sha256")
        == compact.get("protocol_manifest_sha256")
        and identity_manifest.get("inputs", {}).get("checkpoint_sha256")
        == compact.get("checkpoint_sha256")
        and identity_manifest.get("inputs", {}).get("protocol_manifest_sha256")
        == compact.get("protocol_manifest_sha256")
    )
    raw_available = available_raw == 645 and available_sidecars == 645
    return {
        "schema_version": 1,
        "status": (
            "HISTORICAL_REPLAY_AVAILABLE"
            if metadata_exact and raw_available
            else "HISTORICAL_REPLAY_UNAVAILABLE"
        ),
        "non_blocking_for_r1": True,
        "analysis_source_commit": source_commit,
        "checkpoint_sha256": compact.get("checkpoint_sha256"),
        "protocol_sha256": compact.get("protocol_manifest_sha256"),
        "progress_metadata_status": "pass_exact" if metadata_exact else "fail",
        "raw_cache": {
            "expected_raw_entries": 645,
            "available_raw_entries": available_raw,
            "expected_sidecar_entries": 645,
            "available_sidecar_entries": available_sidecars,
            "status": "available" if raw_available else "unavailable",
        },
        "frozen_summary_comparison": {
            "status": "available" if summaries_exact else "unavailable",
            "task_aggregate_sha256": _file_sha256(
                OLD_TASK_ROOT / "aggregate.csv"
            ),
            "identity_aggregate_sha256": _file_sha256(
                OLD_IDENTITY_ROOT / "identity_aggregate.csv"
            ),
        },
        "reason": (
            "all referenced raw and sidecar entries are present"
            if raw_available
            else "progress metadata remains, but referenced raw/sidecar files are absent"
        ),
    }


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise R1AnalysisError("comparison metric must be finite or missing")
    return result


def _comparison_delta(current: object, old: object) -> float | None:
    current_value = _optional_float(current)
    old_value = _optional_float(old)
    if current_value is None or old_value is None:
        return None
    return current_value - old_value


def _checkpoint_regime_comparison(
    *,
    task_rows: Sequence[Mapping[str, object]],
    identity_rows: Sequence[Mapping[str, object]],
    local_rows: Sequence[Mapping[str, object]],
    old_checkpoint: str,
    r1_checkpoint: str,
) -> list[dict[str, object]]:
    old_score_rows = _read_csv(OLD_TASK_ROOT / "aggregate.csv")
    old_score = {
        (
            row["tracker"],
            row["score_reducer"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in old_score_rows
    }
    old_full_rows = _read_csv(OLD_FULL_ROOT / "aggregate_results.csv")
    old_full_rows.extend(_read_csv(OLD_FULL_ROOT / "per_order_results.csv"))
    old_full = {
        (row["method"], row["order_id"], int(row["horizon"])): row
        for row in old_full_rows
    }
    old_identity_rows = _read_csv(OLD_IDENTITY_ROOT / "identity_aggregate.csv")
    old_identity = {
        (row["method"], row["order_id"], int(row["horizon"])): row
        for row in old_identity_rows
    }
    rows: list[dict[str, object]] = []

    def append(
        *,
        channel: str,
        metric: str,
        method: str,
        reducer: str,
        order: str,
        horizon: int,
        old: object,
        current: object,
    ) -> None:
        rows.append(
            {
                "channel": channel,
                "metric": metric,
                "method": method,
                "score_reducer": reducer,
                "order_id": order,
                "horizon": horizon,
                "c_old_value": _optional_float(old),
                "r1_value": _optional_float(current),
                "r1_minus_c_old": _comparison_delta(current, old),
                "c_old_checkpoint_sha256": old_checkpoint,
                "r1_checkpoint_sha256": r1_checkpoint,
            }
        )

    old_score_names = {
        "current_stage_AP": "trajectory_current_slice_AP",
        "current_stage_AP50": "trajectory_current_slice_AP50",
        "current_stage_AP25": "trajectory_current_slice_AP25",
        "current_stage_REC": "trajectory_current_slice_REC",
    }
    for row in task_rows:
        method = str(row["method"])
        reducer = str(row["score_reducer"])
        order = str(row["order_id"])
        horizon = int(row["horizon"])
        if method == "FullHistory":
            old = old_full[(method, order, horizon)]
        else:
            old = old_score[(method, reducer, order, horizon)]
        for metric in TASK_FIELDS:
            old_name = metric if method == "FullHistory" else old_score_names.get(metric, metric)
            append(
                channel="task",
                metric=metric,
                method=method,
                reducer=reducer,
                order=order,
                horizon=horizon,
                old=old[old_name],
                current=row[metric],
            )

    identity_fields = (*IDENTITY_COUNT_FIELDS, *EVENT_COUNT_FIELDS, *IDENTITY_RATE_FIELDS)
    for row in identity_rows:
        method = str(row["method"])
        order = str(row["order_id"])
        horizon = int(row["horizon"])
        old = old_identity[(method, order, horizon)]
        for metric in identity_fields:
            if metric == "recovery_attempt_coverage":
                attempts = int(old["recovery_attempts"])
                gaps = int(old["gap_opportunities"])
                old_value = _rate(attempts, gaps)
            else:
                old_value = old[metric]
            append(
                channel="identity",
                metric=metric,
                method=method,
                reducer="not_applicable",
                order=order,
                horizon=horizon,
                old=old_value,
                current=row[metric],
            )

    old_local = {
        (row["order_id"], int(row["horizon"])): row
        for row in old_score_rows
        if row["tracker"] == "B4" and row["score_reducer"] == "mean"
    }
    for row in local_rows:
        order = str(row["order_id"])
        horizon = int(row["horizon"])
        old = old_local[(order, horizon)]
        for metric in LOCAL_FIELDS:
            append(
                channel="local_current",
                metric=metric,
                method="DirectLocalCurrent",
                reducer="not_applicable",
                order=order,
                horizon=horizon,
                old=old[metric],
                current=row[metric],
            )
    return rows


def run_analysis(
    *,
    cache_root: Path,
    artifact_root: Path,
    contract_path: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    metadata_path: Path,
    data_root: Path,
) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import (
        build_association_events,
        build_rio_class_mapper,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
        observation_content_digest,
    )
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.r1_downstream_context import build_r1_setup
    from scripts.run_r1_downstream_validation import _require_clean_tracked_tree
    from scripts.system_comparison_analysis import _persistent_identity_updates
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import (
        causal_prefix_pair_from_payload,
        compute_deployment_identity_metrics,
    )
    from scripts.system_comparison_v2_analysis import (
        build_v2_causal_pair,
        load_v2_sequences,
    )
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )
    from scripts.system_comparison_v3_identity import run_fresh_tracker_steps
    from scripts.system_comparison_v3_score_sensitivity import (
        assert_score_only_snapshots,
        build_local_current_pair,
    )

    _require_clean_tracked_tree()
    source_commit = _git_head()
    compact, local_progress, full_progress = _validate_new_cache_binding(
        cache_root=cache_root, artifact_root=artifact_root
    )
    setup = build_r1_setup(
        contract_path=contract_path,
        protocol_path=protocol_path,
        checkpoint_path=checkpoint_path,
        pretrained_path=pretrained_path,
        metadata_path=metadata_path,
        data_root=data_root,
        source_commit=source_commit,
        device_name=None,
    )
    metric_dataset_spec = resolve_metric_dataset_spec(data_root)
    local_manifest = {**local_progress, "entry_count": 645}
    sequences = load_v2_sequences(
        cache_manifest=local_manifest,
        cache_root=cache_root,
    )
    full_records = full_progress["records"]
    full_entries: dict[tuple[str, str, int], Mapping[str, object]] = {}
    for record in full_records:
        if not isinstance(record, Mapping) or not isinstance(record.get("key"), Mapping):
            raise R1AnalysisError("FullHistory progress record is invalid")
        key = record["key"]
        identity = (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["horizon"]),
        )
        if identity in full_entries:
            raise R1AnalysisError("FullHistory progress contains duplicate cells")
        full_entries[identity] = record
    if len(full_entries) != 645:
        raise R1AnalysisError("FullHistory progress coverage differs")

    class_mapper = build_rio_class_mapper(setup.dataset)
    factories = build_tracker_factories(setup.p6a_config)
    if not {"B2", "B4"} <= set(factories):
        raise R1AnalysisError("registered tracker factories are incomplete")
    background_class = int(setup.p6a_config["baselines"]["b4"]["background_class"])

    task_aggregate: dict[tuple[object, ...], object] = {}
    task_cluster: dict[tuple[object, ...], object] = {}
    task_counts: dict[tuple[object, ...], int] = defaultdict(int)
    task_cluster_counts: dict[tuple[object, ...], int] = defaultdict(int)
    local_aggregate: dict[tuple[object, ...], object] = {}
    local_cluster: dict[tuple[object, ...], object] = {}
    local_counts: dict[tuple[object, ...], int] = defaultdict(int)
    local_cluster_counts: dict[tuple[object, ...], int] = defaultdict(int)
    task_per_sequence: list[dict[str, object]] = []
    identity_per_sequence: list[dict[str, object]] = []
    local_per_sequence: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []
    score_only_checks = 0

    def update_task(
        *,
        pair: object,
        method: str,
        reducer: str,
        sequence: object,
        horizon: int,
    ) -> None:
        reference = str(getattr(sequence, "reference_scene_id"))
        order = str(getattr(sequence, "order_id"))
        for scope_order in (order, "all"):
            key = (method, reducer, scope_order, horizon)
            accumulator = _task_accumulator(
                task_aggregate, key, dataset_spec=metric_dataset_spec
            )
            accumulator.update(pair)
            task_counts[key] += 1
            cluster_key = (method, reducer, scope_order, horizon, reference)
            cluster_accumulator = _task_accumulator(
                task_cluster, cluster_key, dataset_spec=metric_dataset_spec
            )
            cluster_accumulator.update(pair)
            task_cluster_counts[cluster_key] += 1

    for sequence_index, sequence in enumerate(sequences, start=1):
        observations = tuple(
            cache_payload_to_frozen_observation(raw)
            for raw in sequence.raw_payloads
        )
        cache_digest = observation_content_digest(observations)
        sequence_id = f"{sequence.master_sequence_id}:{sequence.order_id}"
        steps_by_method = {
            method: run_fresh_tracker_steps(
                factory=factories[method],
                observations=observations,
                sequence_id=sequence_id,
            )
            for method in ("B2", "B4")
        }
        trajectories = {
            method: {
                reducer: OfficialCandidateTrajectoryAccumulator(
                    score_reducer=reducer
                )
                for reducer in ("mean", "latest", "max")
            }
            for method in ("B2", "B4")
        }

        for stage, (raw, sidecar) in enumerate(
            zip(sequence.raw_payloads, sequence.sidecars, strict=True)
        ):
            horizon = stage + 1
            snapshots_by_method = {}
            for method in ("B2", "B4"):
                snapshots = {}
                for reducer in ("mean", "latest", "max"):
                    trajectory = trajectories[method][reducer]
                    trajectory.add_stage(sidecar, steps_by_method[method][stage])
                    snapshots[reducer] = trajectory.snapshot()
                assert_score_only_snapshots(snapshots)
                score_only_checks += 1
                snapshots_by_method[method] = snapshots
            if horizon not in REPORT_HORIZONS:
                continue

            local_pair = build_local_current_pair(
                raw_payload=raw,
                sidecar=sidecar,
                class_mapper=class_mapper,
            )
            one_local = OfficialMetricAccumulator(
                mode="raw_local", dataset_spec=metric_dataset_spec
            )
            one_local.update(local_pair.prediction, local_pair.target)
            local_values = _local_metric_block(one_local)
            local_per_sequence.append(
                {
                    "scope": "sequence",
                    "method": "DirectLocalCurrent",
                    "reference_scene_id": sequence.reference_scene_id,
                    "master_sequence_id": sequence.master_sequence_id,
                    "order_id": sequence.order_id,
                    "horizon": horizon,
                    "sequence_count": 1,
                    **local_values,
                }
            )
            for scope_order in (sequence.order_id, "all"):
                local_key = (scope_order, horizon)
                local_acc = _local_accumulator(
                    local_aggregate,
                    local_key,
                    dataset_spec=metric_dataset_spec,
                )
                local_acc.update(local_pair.prediction, local_pair.target)
                local_counts[local_key] += 1
                cluster_key = (
                    scope_order,
                    horizon,
                    sequence.reference_scene_id,
                )
                cluster_acc = _local_accumulator(
                    local_cluster,
                    cluster_key,
                    dataset_spec=metric_dataset_spec,
                )
                cluster_acc.update(local_pair.prediction, local_pair.target)
                local_cluster_counts[cluster_key] += 1

            full_entry = full_entries[
                (sequence.master_sequence_id, sequence.order_id, horizon)
            ]
            full_payload = load_full_history_cache_entry(
                cache_root / "full_history/entries",
                full_entry,
                expected_provenance=full_progress["provenance"],
            )
            full_pair = causal_prefix_pair_from_payload(full_payload)
            full_values = _single_task_metrics(
                full_pair, dataset_spec=metric_dataset_spec
            )
            task_per_sequence.append(
                {
                    "method": "FullHistory",
                    "score_reducer": "official",
                    "reference_scene_id": sequence.reference_scene_id,
                    "master_sequence_id": sequence.master_sequence_id,
                    "order_id": sequence.order_id,
                    "horizon": horizon,
                    **{field: full_values[field] for field in TASK_FIELDS},
                }
            )
            update_task(
                pair=full_pair,
                method="FullHistory",
                reducer="official",
                sequence=sequence,
                horizon=horizon,
            )
            del full_payload, full_pair

            for method in ("B2", "B4"):
                for reducer in ("mean", "latest", "max"):
                    pair = build_v2_causal_pair(
                        snapshot=snapshots_by_method[method][reducer],
                        raw_payloads=sequence.raw_payloads[:horizon],
                        class_mapper=class_mapper,
                    )
                    values = _single_task_metrics(
                        pair, dataset_spec=metric_dataset_spec
                    )
                    task_per_sequence.append(
                        {
                            "method": method,
                            "score_reducer": reducer,
                            "reference_scene_id": sequence.reference_scene_id,
                            "master_sequence_id": sequence.master_sequence_id,
                            "order_id": sequence.order_id,
                            "horizon": horizon,
                            **{field: values[field] for field in TASK_FIELDS},
                        }
                    )
                    update_task(
                        pair=pair,
                        method=method,
                        reducer=reducer,
                        sequence=sequence,
                        horizon=horizon,
                    )

        for method in ("B2", "B4"):
            steps = steps_by_method[method]
            updates = _persistent_identity_updates(
                payloads=sequence.raw_payloads,
                steps=steps,
                class_mapper=class_mapper,
                background_class=background_class,
            )
            events = build_association_events(
                sequence.raw_payloads,
                steps,
                method=method,
                reference_scene_id=sequence.reference_scene_id,
                master_sequence_id=sequence.master_sequence_id,
                order_id=sequence.order_id,
                prefix=5,
                cache_digest=cache_digest,
                background_class=background_class,
            )
            ledger_rows.extend(gap_event_rows(events))
            for horizon in REPORT_HORIZONS:
                metrics = _identity_with_attempt_coverage(
                    compute_deployment_identity_metrics(updates[:horizon])
                )
                identity_per_sequence.append(
                    {
                        "method": method,
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "horizon": horizon,
                        **metrics,
                        **_event_counts(events, horizon=horizon),
                        "tracker_input": "frozen_query_observation_no_gt",
                        "identity_matching": "post_prediction_hungarian_iou_0.5",
                        "cache_digest": cache_digest,
                    }
                )
        if sequence_index % 5 == 0 or sequence_index == len(sequences):
            print(
                f"[r1-analysis] completed {sequence_index}/{len(sequences)} sequences",
                file=sys.stderr,
                flush=True,
            )

    coverage = validate_per_sequence_coverage(
        task_rows=task_per_sequence,
        identity_rows=identity_per_sequence,
        local_rows=local_per_sequence,
    )
    references = sorted({row["reference_scene_id"] for row in local_per_sequence})

    task_aggregate_rows = []
    for key, accumulator in sorted(task_aggregate.items()):
        method, reducer, order, horizon = key
        task_aggregate_rows.append(
            {
                "method": method,
                "score_reducer": reducer,
                "order_id": order,
                "horizon": horizon,
                "sequence_count": task_counts[key],
                **_task_metric_block(accumulator),
            }
        )
    task_cluster_rows = []
    for key, accumulator in sorted(task_cluster.items()):
        method, reducer, order, horizon, reference = key
        task_cluster_rows.append(
            {
                "method": method,
                "score_reducer": reducer,
                "reference_scene_id": reference,
                "order_id": order,
                "horizon": horizon,
                "sequence_count": task_cluster_counts[key],
                **_task_metric_block(accumulator),
                "inference_unit": "reference_scene_id",
            }
        )
    if len(task_aggregate_rows) != 84 or len(task_cluster_rows) != 504:
        raise R1AnalysisError("task aggregate coverage differs")

    local_aggregate_rows = []
    for key, accumulator in sorted(local_aggregate.items()):
        order, horizon = key
        local_aggregate_rows.append(
            {
                "scope": "aggregate",
                "method": "DirectLocalCurrent",
                "reference_scene_id": None,
                "master_sequence_id": None,
                "order_id": order,
                "horizon": horizon,
                "sequence_count": local_counts[key],
                **_local_metric_block(accumulator),
            }
        )
    local_cluster_rows = []
    for key, accumulator in sorted(local_cluster.items()):
        order, horizon, reference = key
        local_cluster_rows.append(
            {
                "scope": "cluster",
                "method": "DirectLocalCurrent",
                "reference_scene_id": reference,
                "master_sequence_id": None,
                "order_id": order,
                "horizon": horizon,
                "sequence_count": local_cluster_counts[key],
                **_local_metric_block(accumulator),
                "inference_unit": "reference_scene_id",
            }
        )
    if len(local_aggregate_rows) != 12 or len(local_cluster_rows) != 72:
        raise R1AnalysisError("local aggregate coverage differs")

    identity_aggregate_rows, identity_cluster_rows = _identity_tables(
        identity_per_sequence, references
    )
    recovery = classify_recovery(
        identity_aggregate_rows, identity_cluster_rows
    )
    sensitivity_rows = []
    for reducer in ("mean", "latest", "max"):
        for horizon in REPORT_HORIZONS:
            for metric in ("causal_prefix_t_mAP", "causal_prefix_t_REC"):
                sensitivity_rows.append(
                    {
                        "channel": "task",
                        "higher_is_better": True,
                        **paired_cluster_bootstrap(
                            task_cluster_rows,
                            metric=metric,
                            horizon=horizon,
                            score_reducer=reducer,
                            replicates=10_000,
                            seed=45,
                        ),
                    }
                )
    for metric, horizons, higher in (
        ("normalized_id_switch_rate", REPORT_HORIZONS, False),
        ("gap_recovery_recall", RECOVERY_HORIZONS, True),
    ):
        for horizon in horizons:
            sensitivity_rows.append(
                {
                    "channel": "identity",
                    "higher_is_better": higher,
                    **paired_cluster_bootstrap(
                        identity_cluster_rows,
                        metric=metric,
                        horizon=horizon,
                        score_reducer=None,
                        replicates=10_000,
                        seed=45,
                    ),
                }
            )

    historical = _historical_replay_status(source_commit=source_commit)
    comparison_rows = _checkpoint_regime_comparison(
        task_rows=task_aggregate_rows,
        identity_rows=identity_aggregate_rows,
        local_rows=local_aggregate_rows,
        old_checkpoint=str(setup.contract["historical_checkpoint"]["sha256"]),
        r1_checkpoint=str(setup.contract["checkpoint"]["sha256"]),
    )
    metrics_root = artifact_root / "metrics"
    outputs = {
        metrics_root / "local_current.csv": _csv_bytes(
            [*local_per_sequence, *local_aggregate_rows, *local_cluster_rows]
        ),
        metrics_root / "task_per_sequence.csv": _csv_bytes(task_per_sequence),
        metrics_root / "task_aggregate.csv": _csv_bytes(task_aggregate_rows),
        metrics_root / "task_per_cluster.csv": _csv_bytes(task_cluster_rows),
        metrics_root / "identity_per_sequence.csv": _csv_bytes(
            identity_per_sequence
        ),
        metrics_root / "identity_aggregate.csv": _csv_bytes(
            identity_aggregate_rows
        ),
        metrics_root / "identity_per_cluster.csv": _csv_bytes(
            identity_cluster_rows
        ),
        metrics_root / "gap_event_ledger.csv": _csv_bytes(ledger_rows),
        metrics_root / "score_sensitivity.csv": _csv_bytes(sensitivity_rows),
        metrics_root / "checkpoint_regime_comparison.csv": _csv_bytes(
            comparison_rows
        ),
    }
    for path, content in outputs.items():
        _atomic_write(path, content)
    _atomic_json(artifact_root / "historical_replay_status.json", historical)
    return {
        "status": "pass",
        "analysis_source_commit": source_commit,
        "checkpoint_sha256": compact["checkpoint_sha256"],
        "coverage": coverage,
        "task_aggregate_rows": len(task_aggregate_rows),
        "task_cluster_rows": len(task_cluster_rows),
        "identity_aggregate_rows": len(identity_aggregate_rows),
        "identity_cluster_rows": len(identity_cluster_rows),
        "gap_event_rows": len(ledger_rows),
        "score_only_snapshot_checks": score_only_checks,
        "recovery": recovery,
        "historical_replay_status": historical["status"],
    }


def argument_parser() -> argparse.ArgumentParser:
    from scripts.run_r1_downstream_validation import (
        DEFAULT_CHECKPOINT,
        DEFAULT_CONTRACT,
        DEFAULT_PRETRAINED,
        DEFAULT_PROTOCOL,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = argument_parser().parse_args(argv)
    result = run_analysis(
        cache_root=arguments.cache_root,
        artifact_root=arguments.artifact_root,
        contract_path=arguments.contract,
        protocol_path=arguments.protocol,
        checkpoint_path=arguments.checkpoint,
        pretrained_path=arguments.pretrained,
        metadata_path=arguments.metadata,
        data_root=arguments.data_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
