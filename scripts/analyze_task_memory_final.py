#!/usr/bin/env python3
"""Validate and summarize frozen TaskMemory V2 evaluation evidence."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2"
DEFAULT_LEGACY_BASELINE = (
    PROJECT_ROOT / "artifacts/allt_task_superiority_v1/baseline/all_t_metrics.csv"
)
DEFAULT_RESOURCE_SUMMARY = DEFAULT_ARTIFACT_ROOT / "resources/run_summary.json"
PROTOCOL_B_POPULATION_ID = "protocol_b_43_masters_3_orders"

REPORT_HORIZONS = (2, 3, 4, 5)
TASK_METRICS = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
)
ALL_T_EPSILON = 1e-6
_ROUNDING_TOLERANCE = 1e-12
_IDENTITY_COUNTS = (
    "correct_recoveries",
    "deployment_id_switches",
    "fragmentation_count",
    "fragmentation_opportunities",
    "gap_opportunities",
    "identity_transition_opportunities",
    "merge_count",
    "merge_opportunities",
    "recovery_attempts",
)
_IDENTITY_RATES = {
    "fragmentation_rate": ("fragmentation_count", "fragmentation_opportunities"),
    "gap_recovery_accuracy": ("correct_recoveries", "recovery_attempts"),
    "gap_recovery_attempt_coverage": ("recovery_attempts", "gap_opportunities"),
    "gap_recovery_recall": ("correct_recoveries", "gap_opportunities"),
    "merge_rate": ("merge_count", "merge_opportunities"),
    "normalized_id_switch_rate": (
        "deployment_id_switches",
        "identity_transition_opportunities",
    ),
}
_IDENTITY_EVENT_COUNTS = (
    "birth_count",
    "false_birth_count",
    "reactivation_count",
    "false_reactivation_count",
    "rejected_birth_count",
)
_IDENTITY_EVENT_RATES = {
    "false_birth_rate": ("false_birth_count", "birth_count"),
    "false_reactivation_rate": (
        "false_reactivation_count",
        "reactivation_count",
    ),
}
ALL_T_FIELDS = (
    "population_id",
    "evidence_scope",
    "variant",
    "checkpoint_sha256",
    "source_commit",
    "config_sha256",
    "training_seed",
    "evaluation_seed",
    "policy",
    "reducer",
    "window",
    "K",
    "r",
    "state_bytes",
    "T",
    *TASK_METRICS,
    "direct_current_AP",
    "reference_count",
    "master_count",
    "order_count",
)
PAIRED_DELTA_FIELDS = (
    "comparison",
    "T",
    "metric",
    "candidate_value",
    "baseline_value",
    "delta",
    "strict_positive",
)
RETENTION_FIELDS = (
    "variant",
    "T",
    "A_t_mAP",
    "R_t_mAP",
    "Dmax_t_mAP",
)
PER_REFERENCE_FIELDS = (
    "population_id",
    "variant",
    "checkpoint_sha256",
    "reference_id",
    "T",
    *TASK_METRICS,
    "direct_current_AP",
    "master_count",
    "order_count",
)
REFERENCE_DELTA_FIELDS = (
    "comparison",
    "reference_id",
    "T",
    "metric",
    "candidate_value",
    "baseline_value",
    "delta",
)
REFERENCE_BOOTSTRAP_FIELDS = (
    "metric",
    "T",
    "estimand",
    "reference_count",
    "mean_delta",
    "bootstrap_resamples",
    "bootstrap_seed",
    "interval_level",
    "interval_lower",
    "interval_upper",
)


class FinalAnalysisError(RuntimeError):
    """Raised when evidence does not satisfy the frozen analysis contract."""


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalAnalysisError(f"{name} must be a sequence")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalAnalysisError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise FinalAnalysisError(f"{name} must be an integer >= {minimum}")
    try:
        converted = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise FinalAnalysisError(f"{name} must be an integer >= {minimum}") from error
    if str(value).strip() != str(converted) or converted < minimum:
        raise FinalAnalysisError(f"{name} must be an integer >= {minimum}")
    return converted


def _rate(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise FinalAnalysisError(f"{name} must be a finite rate")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise FinalAnalysisError(f"{name} must be a finite rate") from error
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise FinalAnalysisError(f"{name} must be a finite rate")
    return converted


def _optional_rate(value: object, *, name: str) -> float | None:
    if value is None or value == "N/A" or value == "":
        return None
    return _rate(value, name=name)


def validate_primary_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_variant: str,
    expected_checkpoint_sha256: str,
    expected_policy: str = "lag1",
    expected_reducer: str = "mean",
) -> list[dict[str, object]]:
    """Require one frozen checkpoint/policy/reducer with exact T2-T5 coverage."""

    values = _sequence(rows, name="primary rows")
    expected_variant = _text(expected_variant, name="expected variant")
    expected_checkpoint_sha256 = _text(
        expected_checkpoint_sha256, name="expected checkpoint SHA256"
    )
    if len(expected_checkpoint_sha256) != 64:
        raise FinalAnalysisError("expected checkpoint SHA256 must have 64 characters")
    normalized = []
    seen = set()
    population_id: str | None = None
    for raw in values:
        if not isinstance(raw, Mapping):
            raise FinalAnalysisError("primary row must be a mapping")
        variant = raw.get("variant", raw.get("model"))
        if (
            variant != expected_variant
            or raw.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise FinalAnalysisError("primary checkpoint/variant differs")
        if (
            raw.get("policy") != expected_policy
            or raw.get("reducer") != expected_reducer
        ):
            raise FinalAnalysisError("primary policy/reducer differs")
        horizon = _integer(raw.get("T"), name="primary T", minimum=2)
        if horizon not in REPORT_HORIZONS or horizon in seen:
            raise FinalAnalysisError("primary horizon coverage differs")
        seen.add(horizon)
        observed_population = _text(raw.get("population_id"), name="population_id")
        if population_id is None:
            population_id = observed_population
        elif population_id != observed_population:
            raise FinalAnalysisError("primary population differs across horizons")
        row = dict(raw)
        row["variant"] = expected_variant
        row["T"] = horizon
        for metric in TASK_METRICS:
            row[metric] = _rate(raw.get(metric), name=metric)
        row["direct_current_AP"] = _rate(
            raw.get("direct_current_AP", raw.get("local_current_AP")),
            name="direct_current_AP",
        )
        for name, minimum in (
            ("training_seed", 0),
            ("evaluation_seed", 0),
            ("episode_count", 1),
            ("reference_count", 1),
            ("master_count", 1),
        ):
            value = (
                raw.get("order_count")
                if name == "episode_count" and raw.get(name) is None
                else raw.get(name)
            )
            row[name] = _integer(value, name=name, minimum=minimum)
        normalized.append(row)
    if seen != set(REPORT_HORIZONS):
        raise FinalAnalysisError("primary horizon coverage differs")
    invariants = (
        "population_id",
        "variant",
        "checkpoint_sha256",
        "training_seed",
        "evaluation_seed",
        "policy",
        "reducer",
        "episode_count",
        "reference_count",
        "master_count",
    )
    for field in invariants:
        if len({row[field] for row in normalized}) != 1:
            raise FinalAnalysisError(f"primary {field} differs across horizons")
    first = normalized[0]
    if (
        first["population_id"] != PROTOCOL_B_POPULATION_ID
        or first["training_seed"] != 45
        or first["evaluation_seed"] != 45
        or first["episode_count"] != 129
        or first["reference_count"] != 6
        or first["master_count"] != 43
    ):
        raise FinalAnalysisError(
            "primary rows differ from the frozen Protocol-B population"
        )
    return sorted(normalized, key=lambda row: int(row["T"]))


def _index_metric_rows(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> dict[int, Mapping[str, object]]:
    values = _sequence(rows, name=f"{name} rows")
    indexed = {}
    populations = set()
    for row in values:
        if not isinstance(row, Mapping):
            raise FinalAnalysisError(f"{name} row must be a mapping")
        horizon = _integer(row.get("T"), name=f"{name} T", minimum=2)
        if horizon not in REPORT_HORIZONS or horizon in indexed:
            raise FinalAnalysisError(f"{name} horizon coverage differs")
        populations.add(_text(row.get("population_id"), name=f"{name} population"))
        for metric in TASK_METRICS:
            _rate(row.get(metric), name=f"{name} {metric}")
        indexed[horizon] = row
    if set(indexed) != set(REPORT_HORIZONS) or len(populations) != 1:
        raise FinalAnalysisError(f"{name} horizon/population coverage differs")
    return indexed


def build_all_t_comparison(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    comparison_name: str,
    epsilon: float = ALL_T_EPSILON,
) -> dict[str, object]:
    """Build all 20 primary cells and strict all-T verdicts."""

    comparison_name = _text(comparison_name, name="comparison name")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise FinalAnalysisError("comparison epsilon must be non-negative")
    candidate = _index_metric_rows(candidate_rows, name="candidate")
    baseline = _index_metric_rows(baseline_rows, name="baseline")
    if {row["population_id"] for row in candidate.values()} != {
        row["population_id"] for row in baseline.values()
    }:
        raise FinalAnalysisError("candidate and baseline populations differ")
    output = []
    for horizon in REPORT_HORIZONS:
        for metric in TASK_METRICS:
            candidate_value = _rate(candidate[horizon][metric], name=metric)
            baseline_value = _rate(baseline[horizon][metric], name=metric)
            delta = candidate_value - baseline_value
            positive = delta - epsilon > _ROUNDING_TOLERANCE
            output.append(
                {
                    "comparison": comparison_name,
                    "T": horizon,
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "delta": delta,
                    "strict_positive": positive,
                }
            )
    positive_cells = sum(bool(row["strict_positive"]) for row in output)
    failed_tmap = [
        int(row["T"])
        for row in output
        if row["metric"] == "t_mAP" and not row["strict_positive"]
    ]
    tmap_deltas = [float(row["delta"]) for row in output if row["metric"] == "t_mAP"]
    return {
        "comparison": comparison_name,
        "epsilon": epsilon,
        "rows": output,
        "positive_cells": positive_cells,
        "total_cells": len(output),
        "tmap_all_t": "PASS" if not failed_tmap else "FAIL",
        "task_metrics_all_t": "PASS" if positive_cells == len(output) else "FAIL",
        "failed_tmap_horizons": failed_tmap,
        "minimum_tmap_delta": min(tmap_deltas),
        "mean_tmap_delta": sum(tmap_deltas) / len(tmap_deltas),
    }


def build_retention_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    indexed = _index_metric_rows(rows, name="retention")
    t2 = _rate(indexed[2]["t_mAP"], name="T2 t_mAP")
    if t2 <= 0:
        raise FinalAnalysisError("retention requires positive T2 t_mAP")
    absolute = {
        horizon: _rate(indexed[horizon]["t_mAP"], name="t_mAP")
        for horizon in REPORT_HORIZONS
    }
    dmax = max(t2 - value for value in absolute.values())
    variant = indexed[2].get("variant", indexed[2].get("model"))
    return [
        {
            "variant": variant,
            "T": horizon,
            "A_t_mAP": absolute[horizon],
            "R_t_mAP": absolute[horizon] / t2,
            "Dmax_t_mAP": dmax,
        }
        for horizon in REPORT_HORIZONS
    ]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_equal_reference_deltas(
    paired_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 45,
    resamples: int = 1000,
) -> list[dict[str, object]]:
    """Bootstrap six equal-weight reference deltas; this is not pooled-AP CI."""

    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or not 1 <= resamples <= 1000
    ):
        raise FinalAnalysisError("reference bootstrap requires 1-1000 resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FinalAnalysisError("reference bootstrap seed must be an integer")
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in _sequence(paired_rows, name="paired reference rows"):
        if not isinstance(row, Mapping):
            raise FinalAnalysisError("paired reference row must be a mapping")
        reference = _text(
            row.get("reference_id", row.get("reference_scene_id")),
            name="reference_id",
        )
        metric = _text(row.get("metric"), name="metric")
        horizon = _integer(row.get("T"), name="reference T", minimum=2)
        if metric not in TASK_METRICS or horizon not in (2, 5):
            raise FinalAnalysisError(
                "reference bootstrap is restricted to T2/T5 task metrics"
            )
        try:
            delta = float(row.get("delta"))
        except (TypeError, ValueError) as error:
            raise FinalAnalysisError("reference delta must be finite") from error
        if not math.isfinite(delta) or reference in grouped[(metric, horizon)]:
            raise FinalAnalysisError("reference delta coverage differs")
        grouped[(metric, horizon)][reference] = delta
    expected = {(metric, horizon) for metric in TASK_METRICS for horizon in (2, 5)}
    if set(grouped) != expected or any(len(values) != 6 for values in grouped.values()):
        raise FinalAnalysisError(
            "reference bootstrap requires six clusters in ten cells"
        )
    rng = random.Random(seed)
    output = []
    for metric, horizon in sorted(
        grouped, key=lambda key: (TASK_METRICS.index(key[0]), key[1])
    ):
        values = [
            grouped[(metric, horizon)][key]
            for key in sorted(grouped[(metric, horizon)])
        ]
        samples = [
            sum(values[rng.randrange(6)] for _ in range(6)) / 6
            for _ in range(resamples)
        ]
        output.append(
            {
                "metric": metric,
                "T": horizon,
                "estimand": "equal_reference_descriptive",
                "reference_count": 6,
                "mean_delta": sum(values) / 6,
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "interval_level": 0.95,
                "interval_lower": _quantile(samples, 0.025),
                "interval_upper": _quantile(samples, 0.975),
            }
        )
    return output


def validate_identity_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_variant: str,
    expected_checkpoint_sha256: str,
) -> list[dict[str, object]]:
    values = _sequence(rows, name="identity rows")
    normalized = []
    seen = set()
    for raw in values:
        if not isinstance(raw, Mapping):
            raise FinalAnalysisError("identity row must be a mapping")
        if (
            raw.get("variant", raw.get("model")) != expected_variant
            or raw.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise FinalAnalysisError("identity checkpoint/variant differs")
        if raw.get("policy") != "lag1":
            raise FinalAnalysisError("identity policy differs")
        if raw.get("identity_linker") != "lag1-route-then-class-iou-v1":
            raise FinalAnalysisError("identity linker differs")
        population = {
            "population_id": raw.get("population_id"),
            "episode_count": _integer(
                raw.get("episode_count"), name="identity episode count", minimum=1
            ),
            "reference_count": _integer(
                raw.get("reference_count"), name="identity reference count", minimum=1
            ),
            "master_count": _integer(
                raw.get("master_count"), name="identity master count", minimum=1
            ),
            "training_seed": _integer(
                raw.get("training_seed"), name="identity training seed"
            ),
            "evaluation_seed": _integer(
                raw.get("evaluation_seed"), name="identity evaluation seed"
            ),
            "visual_content_control": raw.get("visual_content_control"),
        }
        if population != {
            "population_id": PROTOCOL_B_POPULATION_ID,
            "episode_count": 129,
            "reference_count": 6,
            "master_count": 43,
            "training_seed": 45,
            "evaluation_seed": 45,
            "visual_content_control": "native",
        }:
            raise FinalAnalysisError("identity population differs")
        horizon = _integer(raw.get("T"), name="identity T", minimum=2)
        if horizon not in REPORT_HORIZONS or horizon in seen:
            raise FinalAnalysisError("identity horizon coverage differs")
        seen.add(horizon)
        row = dict(raw)
        row.update(population)
        row["T"] = horizon
        for field in _IDENTITY_COUNTS:
            row[field] = _integer(raw.get(field), name=field)
        for rate_name, (numerator_name, denominator_name) in _IDENTITY_RATES.items():
            observed = _optional_rate(raw.get(rate_name), name=rate_name)
            numerator = int(row[numerator_name])
            denominator = int(row[denominator_name])
            expected = numerator / denominator if denominator else None
            if expected is None:
                if observed is not None:
                    raise FinalAnalysisError(f"identity {rate_name} rate must be N/A")
            elif observed is None or not math.isclose(
                observed, expected, rel_tol=0, abs_tol=1e-12
            ):
                raise FinalAnalysisError(f"identity {rate_name} rate differs")
            row[rate_name] = observed
        normalized.append(row)
    if seen != set(REPORT_HORIZONS):
        raise FinalAnalysisError("identity horizon coverage differs")
    return sorted(normalized, key=lambda row: int(row["T"]))


def validate_identity_events(value: Mapping[str, object]) -> dict[str, object]:
    expected_fields = set(_IDENTITY_EVENT_COUNTS) | set(_IDENTITY_EVENT_RATES)
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise FinalAnalysisError("identity event fields differ")
    result = {
        field: _integer(value.get(field), name=field)
        for field in _IDENTITY_EVENT_COUNTS
    }
    for rate, (numerator_name, denominator_name) in _IDENTITY_EVENT_RATES.items():
        observed = _optional_rate(value.get(rate), name=rate)
        numerator = int(result[numerator_name])
        denominator = int(result[denominator_name])
        if numerator > denominator:
            raise FinalAnalysisError("identity event error count exceeds opportunity")
        expected = numerator / denominator if denominator else None
        if expected is None:
            if observed is not None:
                raise FinalAnalysisError("identity event rate must be N/A")
        elif observed is None or not math.isclose(
            observed, expected, rel_tol=0, abs_tol=1e-12
        ):
            raise FinalAnalysisError("identity event rate differs")
        result[rate] = observed
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalAnalysisError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FinalAnalysisError(f"JSON root must be an object: {path}")
    return value


def _validate_content_hash(value: Mapping[str, object], *, name: str) -> None:
    from scripts.task_memory_contracts import canonical_json_sha256

    unsigned = dict(value)
    observed = unsigned.pop("content_sha256", None)
    if observed != canonical_json_sha256(unsigned):
        raise FinalAnalysisError(f"{name} content SHA256 differs")


def load_resource_status(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.exists():
        return "NOT_MEASURED"
    value = _load_json(path)
    _validate_content_hash(value, name="resource profile")
    if value.get("status") != "PASS":
        raise FinalAnalysisError("resource profile did not pass")
    status = value.get("resource_status")
    if status not in {"ADVANTAGE", "TRADEOFF", "NO_ADVANTAGE"}:
        raise FinalAnalysisError("resource status differs")
    return str(status)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeError, csv.Error) as error:
        raise FinalAnalysisError(f"cannot read CSV: {path}") from error
    if not rows:
        raise FinalAnalysisError(f"CSV is empty: {path}")
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    if not rows or any(set(row) != set(fields) for row in rows):
        raise FinalAnalysisError("CSV rows differ from the required schema")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: "N/A" if row[field] is None else row[field] for field in fields}
        for row in rows
    )
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FinalAnalysisError(f"output cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(
        path,
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _verified_evaluation_manifest(evaluation_root: Path) -> dict[str, object]:
    manifest = _load_json(evaluation_root / "manifest.json")
    _validate_content_hash(manifest, name="evaluation manifest")
    population = manifest.get("population")
    if (
        manifest.get("status") != "PASS"
        or manifest.get("schema_version") != "task-memory-evaluation-run-v5"
        or not isinstance(population, Mapping)
        or population.get("id") != PROTOCOL_B_POPULATION_ID
        or population.get("reference_count") != 6
        or population.get("master_count") != 43
        or population.get("order_count") != 129
        or population.get("smoke") is not False
        or manifest.get("evaluation_seed") != 45
        or manifest.get("policies") != ["lag1", "commit0"]
        or manifest.get("reducers") != ["mean"]
        or manifest.get("visual_content_control") != "native"
    ):
        raise FinalAnalysisError("evaluation manifest is not complete Protocol-B")
    for name, expected_rows in (("metrics", 8), ("identity_metrics", 4)):
        record = manifest.get(name)
        if not isinstance(record, Mapping) or record.get("row_count") != expected_rows:
            raise FinalAnalysisError(f"evaluation manifest lacks {name}")
        logical = record.get("logical_reference")
        if not isinstance(logical, str) or not logical.startswith("repo:"):
            raise FinalAnalysisError(f"evaluation {name} reference differs")
        path = PROJECT_ROOT / logical.removeprefix("repo:")
        if (
            not path.is_file()
            or _file_sha256(path) != record.get("file_sha256")
            or path.stat().st_size != record.get("file_bytes")
        ):
            raise FinalAnalysisError(f"evaluation {name} artifact differs")
    return manifest


def _load_verified_cache_record(
    record: Mapping[str, object], cache_root: Path
) -> dict[str, object]:
    from scripts.task_memory_cache import cache_key_sha256, load_task_memory_cache

    filename = _text(record.get("filename"), name="cache filename")
    path = cache_root / filename
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("file_bytes")
        or _file_sha256(path) != record.get("file_sha256")
    ):
        raise FinalAnalysisError("evaluation cache file identity differs")
    payload = load_task_memory_cache(path)
    key = payload.get("key")
    if not isinstance(key, Mapping):
        raise FinalAnalysisError("evaluation cache key differs")
    if payload.get("content_sha256") != record.get(
        "content_sha256"
    ) or cache_key_sha256(key) != record.get("key_sha256"):
        raise FinalAnalysisError("evaluation cache content identity differs")
    if key.get("reference_id") != record.get("reference_id"):
        raise FinalAnalysisError("evaluation cache reference identity differs")
    return payload


def _cache_identity(
    manifest: Mapping[str, object], cache_root: Path
) -> tuple[str, str]:

    cache = manifest.get("cache")
    records = cache.get("records") if isinstance(cache, Mapping) else None
    if not isinstance(records, Sequence) or len(records) != 129:
        raise FinalAnalysisError("evaluation cache manifest coverage differs")
    first = records[0]
    if not isinstance(first, Mapping):
        raise FinalAnalysisError("evaluation cache record differs")
    payload = _load_verified_cache_record(first, cache_root)
    key = payload.get("key")
    if not isinstance(key, Mapping):
        raise FinalAnalysisError("evaluation cache key differs")
    return (
        _text(key.get("resolved_config_sha256"), name="resolved config SHA256"),
        _text(key.get("window_mode"), name="window mode"),
    )


def _state_shape(variant: str) -> tuple[int, int, int]:
    totals = {
        str(row.get("component")): _integer(
            row.get("bytes"), name="state bytes", minimum=1
        )
        for row in _read_csv_rows(DEFAULT_ARTIFACT_ROOT / "resources/state_bytes.csv")
        if row.get("tensor") == "TOTAL" and row.get("status") == "PASS"
    }
    if set(totals) != {"task_state", "visual_state", "combined_state"}:
        raise FinalAnalysisError("state byte evidence differs")
    if totals["task_state"] + totals["visual_state"] != totals["combined_state"]:
        raise FinalAnalysisError("combined state byte evidence differs")
    if variant == "M3-V-CORE":
        return 100, 8, totals["combined_state"]
    if variant == "M3-BASE-CONT":
        return 100, 0, totals["task_state"]
    if variant == "FH-CONT":
        return 0, 0, 0
    raise FinalAnalysisError(f"state shape is not registered for {variant}")


def load_evaluation_bundle(
    evaluation_root: Path,
    cache_root: Path,
    *,
    expected_variant: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    root = evaluation_root.expanduser().resolve()
    manifest = _verified_evaluation_manifest(root)
    if manifest.get("variant") != expected_variant:
        raise FinalAnalysisError("evaluation variant differs")
    checkpoint = _text(manifest.get("checkpoint_sha256"), name="checkpoint SHA256")
    source_commit = _text(manifest.get("source_commit"), name="source commit")
    config_sha256, window = _cache_identity(manifest, cache_root.expanduser().resolve())
    metrics_record = manifest["metrics"]
    identity_record = manifest["identity_metrics"]
    if not isinstance(metrics_record, Mapping) or not isinstance(
        identity_record, Mapping
    ):
        raise FinalAnalysisError("evaluation artifact records differ")
    metric_path = PROJECT_ROOT / str(metrics_record["logical_reference"]).removeprefix(
        "repo:"
    )
    identity_path = PROJECT_ROOT / str(
        identity_record["logical_reference"]
    ).removeprefix("repo:")
    selected_metrics = [
        row
        for row in _read_csv_rows(metric_path)
        if row.get("policy") == "lag1" and row.get("reducer") == "mean"
    ]
    primary = validate_primary_rows(
        selected_metrics,
        expected_variant=expected_variant,
        expected_checkpoint_sha256=checkpoint,
    )
    identity = validate_identity_rows(
        _read_csv_rows(identity_path),
        expected_variant=expected_variant,
        expected_checkpoint_sha256=checkpoint,
    )
    capacity, representatives, state_bytes = _state_shape(expected_variant)
    table = [
        {
            "population_id": PROTOCOL_B_POPULATION_ID,
            "evidence_scope": "new_protocol_b_inference",
            "variant": expected_variant,
            "checkpoint_sha256": checkpoint,
            "source_commit": source_commit,
            "config_sha256": config_sha256,
            "training_seed": row["training_seed"],
            "evaluation_seed": row["evaluation_seed"],
            "policy": row["policy"],
            "reducer": row["reducer"],
            "window": window,
            "K": capacity,
            "r": representatives,
            "state_bytes": state_bytes,
            "T": row["T"],
            **{metric: row[metric] for metric in TASK_METRICS},
            "direct_current_AP": row["direct_current_AP"],
            "reference_count": row["reference_count"],
            "master_count": row["master_count"],
            "order_count": row["episode_count"],
        }
        for row in primary
    ]
    events = manifest.get("identity_events")
    if not isinstance(events, Mapping):
        raise FinalAnalysisError("evaluation identity event summary differs")
    event_summary = validate_identity_events(events)
    identity = [
        {**row, "identity_event_scope": "full_protocol_run", **event_summary}
        for row in identity
    ]
    return table, identity, {"manifest": manifest, "identity_events": event_summary}


def load_legacy_baseline_rows(path: Path) -> list[dict[str, object]]:
    replay = _load_json(
        PROJECT_ROOT / "artifacts/allt_task_superiority_v1/baseline/replay_status.json"
    )
    _validate_content_hash(replay, name="legacy baseline replay")
    source_commit = _text(replay.get("source_commit"), name="legacy source commit")
    selected = []
    for row in _read_csv_rows(path.expanduser().resolve()):
        method = row.get("method")
        reducer = row.get("reducer")
        if method == "B4" and reducer == "mean":
            variant, policy, output_reducer = "B4-commit0", "commit0", "mean"
            capacity, representatives, state_bytes = 100, 0, 61_008
            window = "local_pair"
        elif method == "FullHistory" and reducer == "official":
            variant, policy, output_reducer = (
                "FH-R1-native",
                "full_history_native",
                "official",
            )
            capacity, representatives, state_bytes = 0, 0, 0
            window = "full_history"
        else:
            continue
        selected.append(
            {
                "population_id": PROTOCOL_B_POPULATION_ID,
                "evidence_scope": "verified_legacy_protocol_b_replay",
                "variant": variant,
                "checkpoint_sha256": _text(
                    row.get("checkpoint_sha256"), name="legacy checkpoint SHA256"
                ),
                "source_commit": source_commit,
                "config_sha256": "not_recorded_in_legacy_table",
                "training_seed": _integer(
                    row.get("training_seed"), name="training seed"
                ),
                "evaluation_seed": _integer(
                    row.get("evaluation_seed"), name="evaluation seed"
                ),
                "policy": policy,
                "reducer": output_reducer,
                "window": window,
                "K": capacity,
                "r": representatives,
                "state_bytes": state_bytes,
                "T": _integer(row.get("T"), name="legacy T", minimum=2),
                **{
                    metric: _rate(row.get(metric), name=metric)
                    for metric in TASK_METRICS
                },
                "direct_current_AP": _rate(
                    row.get("local_current_AP"), name="direct_current_AP"
                ),
                "reference_count": _integer(
                    row.get("num_reference_clusters"), name="reference count", minimum=1
                ),
                "master_count": _integer(
                    row.get("num_master"), name="master count", minimum=1
                ),
                "order_count": _integer(
                    row.get("num_order_units"), name="order count", minimum=1
                ),
            }
        )
    for variant, policy, reducer in (
        ("B4-commit0", "commit0", "mean"),
        ("FH-R1-native", "full_history_native", "official"),
    ):
        rows = [row for row in selected if row["variant"] == variant]
        if not rows:
            raise FinalAnalysisError(f"legacy baseline is unavailable: {variant}")
        validate_primary_rows(
            rows,
            expected_variant=variant,
            expected_checkpoint_sha256=str(rows[0]["checkpoint_sha256"]),
            expected_policy=policy,
            expected_reducer=reducer,
        )
    return selected


def build_reference_delta_rows(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    *,
    comparison_name: str,
) -> list[dict[str, object]]:
    def index(
        rows: Sequence[Mapping[str, object]], name: str
    ) -> dict[tuple[str, int], Mapping[str, object]]:
        indexed = {}
        for row in rows:
            reference = _text(row.get("reference_id"), name=f"{name} reference")
            horizon = _integer(row.get("T"), name=f"{name} T", minimum=2)
            key = (reference, horizon)
            if key in indexed:
                raise FinalAnalysisError(f"{name} reference metric cells duplicate")
            indexed[key] = row
        expected = {
            (reference, horizon)
            for reference in {key[0] for key in indexed}
            for horizon in REPORT_HORIZONS
        }
        if len({key[0] for key in indexed}) != 6 or set(indexed) != expected:
            raise FinalAnalysisError(f"{name} requires six-reference T2-T5 coverage")
        return indexed

    candidate = index(candidate_rows, "candidate")
    baseline = index(baseline_rows, "baseline")
    if set(candidate) != set(baseline):
        raise FinalAnalysisError("paired reference coverage differs")
    output = []
    for reference, horizon in sorted(candidate):
        if candidate[(reference, horizon)].get("order_count") != baseline[
            (reference, horizon)
        ].get("order_count"):
            raise FinalAnalysisError("paired reference order counts differ")
        for metric in TASK_METRICS:
            candidate_value = _rate(
                candidate[(reference, horizon)].get(metric), name=metric
            )
            baseline_value = _rate(
                baseline[(reference, horizon)].get(metric), name=metric
            )
            output.append(
                {
                    "comparison": comparison_name,
                    "reference_id": reference,
                    "T": horizon,
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "delta": candidate_value - baseline_value,
                }
            )
    return output


def validate_per_reference_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_variant: str,
    expected_checkpoint_sha256: str,
) -> list[dict[str, object]]:
    normalized = []
    indexed = set()
    for raw in _sequence(rows, name="per-reference rows"):
        if not isinstance(raw, Mapping):
            raise FinalAnalysisError("per-reference row must be a mapping")
        reference = _text(raw.get("reference_id"), name="reference_id")
        horizon = _integer(raw.get("T"), name="per-reference T", minimum=2)
        key = (reference, horizon)
        if horizon not in REPORT_HORIZONS or key in indexed:
            raise FinalAnalysisError("per-reference population differs")
        indexed.add(key)
        if (
            raw.get("population_id") != PROTOCOL_B_POPULATION_ID
            or raw.get("variant") != expected_variant
            or raw.get("checkpoint_sha256") != expected_checkpoint_sha256
        ):
            raise FinalAnalysisError("per-reference population differs")
        row = dict(raw)
        row["T"] = horizon
        for metric in TASK_METRICS:
            row[metric] = _rate(raw.get(metric), name=f"per-reference {metric}")
        row["direct_current_AP"] = _rate(
            raw.get("direct_current_AP"), name="per-reference direct current AP"
        )
        row["master_count"] = _integer(
            raw.get("master_count"), name="per-reference master count", minimum=1
        )
        row["order_count"] = _integer(
            raw.get("order_count"), name="per-reference order count", minimum=1
        )
        normalized.append(row)
    references = {reference for reference, _ in indexed}
    expected = {
        (reference, horizon) for reference in references for horizon in REPORT_HORIZONS
    }
    if len(references) != 6 or indexed != expected:
        raise FinalAnalysisError("per-reference population differs")
    for reference in references:
        reference_rows = [row for row in normalized if row["reference_id"] == reference]
        counts = {
            (int(row["master_count"]), int(row["order_count"]))
            for row in reference_rows
        }
        if len(counts) != 1 or any(orders != masters * 3 for masters, orders in counts):
            raise FinalAnalysisError("per-reference population differs")
    for horizon in REPORT_HORIZONS:
        horizon_rows = [row for row in normalized if row["T"] == horizon]
        if (
            sum(int(row["master_count"]) for row in horizon_rows) != 43
            or sum(int(row["order_count"]) for row in horizon_rows) != 129
        ):
            raise FinalAnalysisError("per-reference population differs")
    return sorted(normalized, key=lambda row: (str(row["reference_id"]), int(row["T"])))


def compute_per_reference_metrics(
    *,
    evaluation_root: Path,
    cache_root: Path,
    variant: str,
    class_mapper: object,
) -> list[dict[str, object]]:
    from scripts.task_memory_metrics import compute_cached_task_metrics

    if not callable(class_mapper):
        raise FinalAnalysisError("per-reference class mapper must be callable")
    manifest = _verified_evaluation_manifest(evaluation_root.expanduser().resolve())
    if manifest.get("variant") != variant:
        raise FinalAnalysisError("per-reference evaluation variant differs")
    cache = manifest.get("cache")
    records = cache.get("records") if isinstance(cache, Mapping) else None
    if not isinstance(records, Sequence) or len(records) != 129:
        raise FinalAnalysisError("per-reference cache coverage differs")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise FinalAnalysisError("per-reference cache record differs")
        reference = _text(record.get("reference_id"), name="cache reference")
        grouped[reference].append(record)
    if len(grouped) != 6 or any(len(values) % 3 for values in grouped.values()):
        raise FinalAnalysisError("per-reference Protocol-B grouping differs")
    checkpoint = _text(manifest.get("checkpoint_sha256"), name="checkpoint SHA256")
    output = []
    for reference in sorted(grouped):
        payloads = []
        for record in grouped[reference]:
            payload = _load_verified_cache_record(
                record, cache_root.expanduser().resolve()
            )
            key = payload.get("key")
            if not isinstance(key, Mapping) or key.get("reference_id") != reference:
                raise FinalAnalysisError("per-reference cache key differs")
            payloads.append(payload)
        metrics = compute_cached_task_metrics(
            payloads,
            reducers=("mean",),
            include_commit0=False,
            class_mapper=class_mapper,
        )
        selected = [
            row
            for row in metrics
            if row.get("policy") == "lag1" and row.get("reducer") == "mean"
        ]
        if len(selected) != 4:
            raise FinalAnalysisError("per-reference task metric coverage differs")
        for row in selected:
            output.append(
                {
                    "population_id": PROTOCOL_B_POPULATION_ID,
                    "variant": variant,
                    "checkpoint_sha256": checkpoint,
                    "reference_id": reference,
                    "T": int(row["T"]),
                    **{metric: float(row[metric]) for metric in TASK_METRICS},
                    "direct_current_AP": float(row["local_current_AP"]),
                    "master_count": len(grouped[reference]) // 3,
                    "order_count": len(grouped[reference]),
                }
            )
        del payloads
        gc.collect()
        print(
            f"[task-memory-final] {variant} reference {reference} complete", flush=True
        )
    return validate_per_reference_rows(
        output,
        expected_variant=variant,
        expected_checkpoint_sha256=checkpoint,
    )


def _class_mapper(data_root: Path, external_root: Path) -> object:
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.evaluate_task_memory import _rio_population_base
    from scripts.train_task_memory import compose_variant_config

    config = compose_variant_config(
        "M3-V-CORE",
        pretrained=external_root / "checkpoints/concerto_base.pth",
        run_dir=external_root / "analysis_runtime",
    )
    base = _rio_population_base(
        config,
        data_root=data_root.expanduser().resolve(strict=True),
        horizon=5,
        population_id=PROTOCOL_B_POPULATION_ID,
    )
    return build_rio_class_mapper(base)


def _retention_status(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> str:
    candidate_rows = build_retention_rows(candidate)
    baseline_rows = build_retention_rows(baseline)
    by_t = {int(row["T"]): row for row in baseline_rows}
    candidate_t2 = float(candidate_rows[0]["A_t_mAP"])
    baseline_t2 = float(baseline_rows[0]["A_t_mAP"])
    noninferior_start = candidate_t2 + ALL_T_EPSILON >= baseline_t2
    deltas = [
        float(row["R_t_mAP"]) - float(by_t[int(row["T"])]["R_t_mAP"])
        for row in candidate_rows[1:]
    ]
    if (
        noninferior_start
        and all(delta >= -ALL_T_EPSILON for delta in deltas)
        and any(delta > ALL_T_EPSILON for delta in deltas)
    ):
        return "IMPROVED"
    return "NOT_IMPROVED"


def analyze_final_results(
    *,
    candidate_evaluation: Path,
    candidate_cache: Path,
    base_evaluation: Path,
    base_cache: Path,
    fh_evaluation: Path,
    fh_cache: Path,
    legacy_baseline: Path,
    output_root: Path,
    data_root: Path | None = None,
    external_root: Path | None = None,
    candidate_reference_path: Path | None = None,
    fh_reference_path: Path | None = None,
    resource_summary: Path = DEFAULT_RESOURCE_SUMMARY,
) -> dict[str, object]:
    current = {}
    identities = []
    metadata = {}
    for variant, evaluation, cache in (
        ("M3-V-CORE", candidate_evaluation, candidate_cache),
        ("M3-BASE-CONT", base_evaluation, base_cache),
        ("FH-CONT", fh_evaluation, fh_cache),
    ):
        rows, identity, details = load_evaluation_bundle(
            evaluation, cache, expected_variant=variant
        )
        current[variant] = rows
        identities.extend(identity)
        metadata[variant] = details
    legacy = load_legacy_baseline_rows(legacy_baseline)
    by_legacy = {
        variant: [row for row in legacy if row["variant"] == variant]
        for variant in ("B4-commit0", "FH-R1-native")
    }
    all_metrics = [
        *by_legacy["B4-commit0"],
        *by_legacy["FH-R1-native"],
        *current["M3-BASE-CONT"],
        *current["FH-CONT"],
        *current["M3-V-CORE"],
    ]
    comparisons = {}
    delta_rows = []
    for baseline in (
        "B4-commit0",
        "FH-R1-native",
        "M3-BASE-CONT",
        "FH-CONT",
    ):
        baseline_rows = current.get(baseline, by_legacy.get(baseline))
        if baseline_rows is None:
            raise FinalAnalysisError(f"required baseline is unavailable: {baseline}")
        name = f"M3-V-CORE_vs_{baseline}"
        result = build_all_t_comparison(
            current["M3-V-CORE"], baseline_rows, comparison_name=name
        )
        comparisons[name] = {
            key: value for key, value in result.items() if key != "rows"
        }
        delta_rows.extend(result["rows"])
    retention = [
        row
        for variant in (
            "B4-commit0",
            "FH-R1-native",
            "M3-BASE-CONT",
            "FH-CONT",
            "M3-V-CORE",
        )
        for row in build_retention_rows(
            current.get(variant, by_legacy.get(variant)) or ()
        )
    ]
    per_reference = []
    reference_deltas = []
    cluster_effects = []
    if (candidate_reference_path is None) != (fh_reference_path is None):
        raise FinalAnalysisError("both precomputed reference tables are required")
    if candidate_reference_path is not None and fh_reference_path is not None:
        candidate_reference = validate_per_reference_rows(
            _read_csv_rows(candidate_reference_path.expanduser().resolve(strict=True)),
            expected_variant="M3-V-CORE",
            expected_checkpoint_sha256=str(
                current["M3-V-CORE"][0]["checkpoint_sha256"]
            ),
        )
        fh_reference = validate_per_reference_rows(
            _read_csv_rows(fh_reference_path.expanduser().resolve(strict=True)),
            expected_variant="FH-CONT",
            expected_checkpoint_sha256=str(current["FH-CONT"][0]["checkpoint_sha256"]),
        )
    elif data_root is not None and external_root is not None:
        mapper = _class_mapper(data_root, external_root)
        candidate_reference = compute_per_reference_metrics(
            evaluation_root=candidate_evaluation,
            cache_root=candidate_cache,
            variant="M3-V-CORE",
            class_mapper=mapper,
        )
        fh_reference = compute_per_reference_metrics(
            evaluation_root=fh_evaluation,
            cache_root=fh_cache,
            variant="FH-CONT",
            class_mapper=mapper,
        )
    else:
        candidate_reference = []
        fh_reference = []
    if candidate_reference and fh_reference:
        per_reference = [*candidate_reference, *fh_reference]
        reference_deltas = build_reference_delta_rows(
            candidate_reference,
            fh_reference,
            comparison_name="M3-V-CORE_vs_FH-CONT",
        )
        cluster_effects = bootstrap_equal_reference_deltas(
            [row for row in reference_deltas if row["T"] in (2, 5)],
            seed=45,
            resamples=1000,
        )
    output_root = output_root.expanduser().resolve()
    outputs = {
        "all_t_metrics.csv": _csv_bytes(all_metrics, ALL_T_FIELDS),
        "paired_deltas.csv": _csv_bytes(delta_rows, PAIRED_DELTA_FIELDS),
        "identity_counts.csv": _csv_bytes(identities, tuple(identities[0])),
        "retention.csv": _csv_bytes(retention, RETENTION_FIELDS),
    }
    if per_reference:
        outputs.update(
            {
                "per_reference_metrics.csv": _csv_bytes(
                    per_reference, PER_REFERENCE_FIELDS
                ),
                "per_reference_deltas.csv": _csv_bytes(
                    reference_deltas, REFERENCE_DELTA_FIELDS
                ),
                "reference_bootstrap.csv": _csv_bytes(
                    cluster_effects, REFERENCE_BOOTSTRAP_FIELDS
                ),
            }
        )
    for filename, content in outputs.items():
        _atomic_write(output_root / filename, content)
    matched = comparisons["M3-V-CORE_vs_FH-CONT"]
    status = {
        "EXECUTION": "PARTIAL",
        "TMAP_ALL_T_VS_R1": comparisons["M3-V-CORE_vs_B4-commit0"][
            "tmap_all_t"
        ],
        "TMAP_ALL_T_VS_MATCHED_FH": matched["tmap_all_t"],
        "TASK_METRICS_ALL_T": matched["task_metrics_all_t"],
        "RETENTION": _retention_status(current["M3-V-CORE"], current["FH-CONT"]),
        "RESOURCE": load_resource_status(resource_summary),
        "MECHANISM": "PARTIAL",
        "GENERALIZATION": "NOT_ESTABLISHED",
        "PUBLICATION": "NOT_ATTEMPTED",
    }
    verdict = {
        "schema_version": "task-memory-final-analysis-v2",
        "population_id": PROTOCOL_B_POPULATION_ID,
        "comparisons": comparisons,
        "status": status,
        "reference_analysis": {
            "status": "COMPLETE" if per_reference else "NOT_RUN",
            "interval_estimand": (
                "equal-reference descriptive interval; not pooled-AP confidence interval"
                if per_reference
                else None
            ),
        },
        "execution_limitations": [
            "final candidate independent-native evaluation not run",
            "not all preregistered long-memory controls were rerun on Protocol-B",
            "single training seed",
        ],
        "identity_events": {
            variant: details["identity_events"] for variant, details in metadata.items()
        },
        "outputs": [
            {"path": filename, "sha256": hashlib.sha256(content).hexdigest()}
            for filename, content in sorted(outputs.items())
        ],
    }
    from scripts.task_memory_contracts import canonical_json_sha256

    verdict["content_sha256"] = canonical_json_sha256(verdict)
    _atomic_json(output_root / "status.json", verdict)
    return verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = DEFAULT_ARTIFACT_ROOT / "evaluation/M5/protocol_b"
    cache = Path(
        "/mnt/shared/ww/persist4d-task-memory-retention-v2/evaluation_cache/M5/protocol_b"
    )
    parser.add_argument("--candidate-evaluation", type=Path, default=root / "M3-V-CORE")
    parser.add_argument("--candidate-cache", type=Path, default=cache / "M3-V-CORE")
    parser.add_argument("--base-evaluation", type=Path, default=root / "M3-BASE-CONT")
    parser.add_argument("--base-cache", type=Path, default=cache / "M3-BASE-CONT")
    parser.add_argument("--fh-evaluation", type=Path, default=root / "FH-CONT")
    parser.add_argument("--fh-cache", type=Path, default=cache / "FH-CONT")
    parser.add_argument("--legacy-baseline", type=Path, default=DEFAULT_LEGACY_BASELINE)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_ARTIFACT_ROOT / "final"
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--candidate-reference", type=Path)
    parser.add_argument("--fh-reference", type=Path)
    parser.add_argument(
        "--resource-summary", type=Path, default=DEFAULT_RESOURCE_SUMMARY
    )
    parser.add_argument("--reference-only", choices=("M3-V-CORE", "FH-CONT"))
    parser.add_argument("--reference-output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (args.data_root is None) != (args.external_root is None):
        raise SystemExit("--data-root and --external-root must be provided together")
    if args.reference_only is not None:
        if (
            args.data_root is None
            or args.external_root is None
            or args.reference_output is None
        ):
            raise SystemExit(
                "--reference-only requires --data-root, --external-root, and "
                "--reference-output"
            )
        evaluation, cache = {
            "M3-V-CORE": (args.candidate_evaluation, args.candidate_cache),
            "FH-CONT": (args.fh_evaluation, args.fh_cache),
        }[args.reference_only]
        rows = compute_per_reference_metrics(
            evaluation_root=evaluation,
            cache_root=cache,
            variant=args.reference_only,
            class_mapper=_class_mapper(args.data_root, args.external_root),
        )
        content = _csv_bytes(rows, PER_REFERENCE_FIELDS)
        _atomic_write(args.reference_output.expanduser().resolve(), content)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "variant": args.reference_only,
                    "rows": len(rows),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                sort_keys=True,
            )
        )
        return 0
    result = analyze_final_results(
        candidate_evaluation=args.candidate_evaluation,
        candidate_cache=args.candidate_cache,
        base_evaluation=args.base_evaluation,
        base_cache=args.base_cache,
        fh_evaluation=args.fh_evaluation,
        fh_cache=args.fh_cache,
        legacy_baseline=args.legacy_baseline,
        output_root=args.output_root,
        data_root=args.data_root,
        external_root=args.external_root,
        candidate_reference_path=args.candidate_reference,
        fh_reference_path=args.fh_reference,
        resource_summary=args.resource_summary,
    )
    print(json.dumps(result["status"], sort_keys=True))
    return 0


__all__ = [
    "ALL_T_EPSILON",
    "REPORT_HORIZONS",
    "TASK_METRICS",
    "FinalAnalysisError",
    "bootstrap_equal_reference_deltas",
    "build_all_t_comparison",
    "build_reference_delta_rows",
    "build_retention_rows",
    "compute_per_reference_metrics",
    "load_evaluation_bundle",
    "load_legacy_baseline_rows",
    "load_resource_status",
    "validate_identity_events",
    "validate_identity_rows",
    "validate_per_reference_rows",
    "validate_primary_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
