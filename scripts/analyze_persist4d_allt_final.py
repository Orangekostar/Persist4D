#!/usr/bin/env python3
"""Derive final per-reference and identity evidence from frozen All-T caches."""

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
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1"
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-allt-task-superiority-v1/evaluation_cache/"
    "protocol_b_43_masters_3_orders"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_ARTIFACT_ROOT / "results"
POPULATION_ID = "protocol_b_43_masters_3_orders"
REPORT_HORIZONS = (2, 3, 4, 5)
TASK_METRICS = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
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
    "gap_recovery_attempt_coverage",
)
FROZEN_MODELS = {
    "C2": {
        "checkpoint_sha256": (
            "a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724"
        ),
        "reducer": "mean",
        "window_mode": "local_pair",
    },
    "FH-adapt": {
        "checkpoint_sha256": (
            "ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c"
        ),
        "reducer": "official",
        "window_mode": "full_history",
    },
}
PER_REFERENCE_FIELDS = (
    "population_id",
    "reference_scene_id",
    "T",
    "metric",
    "candidate_model",
    "candidate_value",
    "baseline_model",
    "baseline_value",
    "delta",
    "candidate_sequence_count",
    "baseline_sequence_count",
)
IDENTITY_FIELDS = (
    "population_id",
    "model",
    "T",
    "sequence_count",
    *IDENTITY_COUNT_FIELDS,
    *IDENTITY_RATE_FIELDS,
)
CLUSTER_EFFECT_FIELDS = (
    "metric",
    "T",
    "estimand",
    "cluster_count",
    "equal_cluster_mean_delta",
    "bootstrap_resamples",
    "bootstrap_seed",
    "ci_level",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
)


class FinalAnalysisError(RuntimeError):
    """Raised when frozen caches cannot support the requested final analysis."""


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FinalAnalysisError(f"{name} must be a sequence")
    return value


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FinalAnalysisError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FinalAnalysisError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalAnalysisError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise FinalAnalysisError(f"{name} must be a finite rate")
    return normalized


def _index_metric_cells(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> tuple[str, dict[tuple[str, int], Mapping[str, object]]]:
    values = _sequence(rows, name=f"{name} metric rows")
    if not values:
        raise FinalAnalysisError(f"{name} metric rows are empty")
    if any(not isinstance(row, Mapping) for row in values):
        raise FinalAnalysisError(f"{name} metric row must be a mapping")
    models = {_nonempty(row.get("model"), name=f"{name} model") for row in values}
    if len(models) != 1:
        raise FinalAnalysisError(f"{name} metric rows must use one model")
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in values:
        reference = _nonempty(
            row.get("reference_scene_id"), name=f"{name} reference"
        )
        horizon = _integer(row.get("T"), name=f"{name} T", minimum=2)
        if horizon not in REPORT_HORIZONS:
            raise FinalAnalysisError(f"{name} horizon coverage differs")
        key = (reference, horizon)
        if key in indexed:
            raise FinalAnalysisError(f"{name} metric rows contain a duplicate cell")
        _integer(
            row.get("sequence_count"), name=f"{name} sequence count", minimum=1
        )
        for metric in TASK_METRICS:
            _rate(row.get(metric), name=f"{name} {metric}")
        indexed[key] = row
    references = {key[0] for key in indexed}
    expected = {
        (reference, horizon)
        for reference in references
        for horizon in REPORT_HORIZONS
    }
    if len(references) != 6 or set(indexed) != expected:
        raise FinalAnalysisError(f"{name} metric coverage differs")
    return next(iter(models)), indexed


def build_paired_reference_rows(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Pair official pooled-within-reference metrics in long form."""
    candidate_model, candidate = _index_metric_cells(
        candidate_rows, name="candidate"
    )
    baseline_model, baseline = _index_metric_cells(baseline_rows, name="baseline")
    if set(candidate) != set(baseline):
        raise FinalAnalysisError("candidate and baseline reference coverage differs")
    output = []
    for reference, horizon in sorted(candidate):
        candidate_row = candidate[(reference, horizon)]
        baseline_row = baseline[(reference, horizon)]
        candidate_count = _integer(
            candidate_row.get("sequence_count"),
            name="candidate sequence count",
            minimum=1,
        )
        baseline_count = _integer(
            baseline_row.get("sequence_count"),
            name="baseline sequence count",
            minimum=1,
        )
        if candidate_count != baseline_count:
            raise FinalAnalysisError("paired reference sequence coverage differs")
        for metric in TASK_METRICS:
            candidate_value = _rate(candidate_row.get(metric), name=metric)
            baseline_value = _rate(baseline_row.get(metric), name=metric)
            output.append(
                {
                    "population_id": POPULATION_ID,
                    "reference_scene_id": reference,
                    "T": horizon,
                    "metric": metric,
                    "candidate_model": candidate_model,
                    "candidate_value": candidate_value,
                    "baseline_model": baseline_model,
                    "baseline_value": baseline_value,
                    "delta": candidate_value - baseline_value,
                    "candidate_sequence_count": candidate_count,
                    "baseline_sequence_count": baseline_count,
                }
            )
    return output


def aggregate_identity_rows(
    sequence_rows: Sequence[Mapping[str, object]],
    *,
    expected_sequence_count: int = 129,
) -> list[dict[str, object]]:
    """Pool identity event counts before deriving rates and explicit N/A values."""
    from scripts.system_comparison_analysis import aggregate_identity_metrics

    rows = _sequence(sequence_rows, name="identity sequence rows")
    expected_sequence_count = _integer(
        expected_sequence_count, name="expected sequence count", minimum=1
    )
    groups: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    identities: dict[tuple[str, int], set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        if not isinstance(row, Mapping):
            raise FinalAnalysisError("identity sequence row must be a mapping")
        model = _nonempty(row.get("model"), name="identity model")
        if model not in FROZEN_MODELS:
            raise FinalAnalysisError("identity model differs from the frozen comparison")
        horizon = _integer(row.get("T"), name="identity T", minimum=2)
        if horizon not in REPORT_HORIZONS:
            raise FinalAnalysisError("identity horizon coverage differs")
        identity = (
            _nonempty(row.get("reference_scene_id"), name="identity reference"),
            _nonempty(row.get("master_sequence_id"), name="identity master"),
            _nonempty(row.get("order_id"), name="identity order"),
        )
        key = (model, horizon)
        if identity in identities[key]:
            raise FinalAnalysisError("identity rows contain a duplicate sequence cell")
        identities[key].add(identity)
        for field in IDENTITY_COUNT_FIELDS:
            _integer(row.get(field), name=field)
        groups[key].append(row)
    expected_groups = {
        (model, horizon) for model in FROZEN_MODELS for horizon in REPORT_HORIZONS
    }
    if set(groups) != expected_groups or any(
        len(group) != expected_sequence_count for group in groups.values()
    ):
        raise FinalAnalysisError("identity sequence coverage differs")
    output = []
    for model in FROZEN_MODELS:
        for horizon in REPORT_HORIZONS:
            aggregate = aggregate_identity_metrics(groups[(model, horizon)])
            output.append(
                {
                    "population_id": POPULATION_ID,
                    "model": model,
                    "T": horizon,
                    "sequence_count": expected_sequence_count,
                    **aggregate,
                }
            )
    return output


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise FinalAnalysisError("bootstrap sample is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_equal_cluster_effect(
    per_reference_rows: Sequence[Mapping[str, object]],
    *,
    seed: int = 45,
    resamples: int = 1000,
) -> list[dict[str, object]]:
    """Bootstrap six equal-weight reference deltas as descriptive evidence."""
    rows = _sequence(per_reference_rows, name="per-reference rows")
    seed = _integer(seed, name="bootstrap seed")
    resamples = _integer(resamples, name="bootstrap resamples", minimum=1)
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        metric = _nonempty(row.get("metric"), name="cluster metric")
        if metric not in TASK_METRICS:
            raise FinalAnalysisError("cluster metric coverage differs")
        horizon = _integer(row.get("T"), name="cluster T", minimum=2)
        reference = _nonempty(
            row.get("reference_scene_id"), name="cluster reference"
        )
        delta = row.get("delta")
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise FinalAnalysisError("cluster delta must be numeric")
        delta = float(delta)
        if not math.isfinite(delta):
            raise FinalAnalysisError("cluster delta must be finite")
        if reference in grouped[(metric, horizon)]:
            raise FinalAnalysisError("cluster rows contain a duplicate cell")
        grouped[(metric, horizon)][reference] = delta
    expected = {
        (metric, horizon) for metric in TASK_METRICS for horizon in REPORT_HORIZONS
    }
    if set(grouped) != expected or any(len(values) != 6 for values in grouped.values()):
        raise FinalAnalysisError("cluster effect coverage differs")
    generator = random.Random(seed)
    output = []
    for metric in TASK_METRICS:
        for horizon in REPORT_HORIZONS:
            values = [value for _, value in sorted(grouped[(metric, horizon)].items())]
            sampled_means = [
                sum(generator.choice(values) for _ in values) / len(values)
                for _ in range(resamples)
            ]
            output.append(
                {
                    "metric": metric,
                    "T": horizon,
                    "estimand": "equal_cluster_effect",
                    "cluster_count": len(values),
                    "equal_cluster_mean_delta": sum(values) / len(values),
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": seed,
                    "ci_level": 0.95,
                    "bootstrap_ci_lower": _quantile(sampled_means, 0.025),
                    "bootstrap_ci_upper": _quantile(sampled_means, 0.975),
                }
            )
    return output


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise FinalAnalysisError("value is not portable canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FinalAnalysisError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalAnalysisError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise FinalAnalysisError(f"JSON root must be an object: {path}")
    return value


def _git_head() -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FinalAnalysisError("cannot resolve source commit") from error
    if len(value) != 40:
        raise FinalAnalysisError("source commit is invalid")
    return value


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise FinalAnalysisError("CSV row fields differ")
        writer.writerow(
            {field: "" if row[field] is None else row[field] for field in fields}
        )
    return stream.getvalue().encode("ascii")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _validate_manifest(
    path: Path, *, model: str
) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    manifest = _load_json(path)
    specification = FROZEN_MODELS[model]
    unsigned = dict(manifest)
    expected_content = unsigned.pop("content_sha256", None)
    if expected_content != _canonical_json_sha256(unsigned):
        raise FinalAnalysisError(f"{model} cache manifest content hash differs")
    expected = {
        "status": "pass",
        "population_id": POPULATION_ID,
        "checkpoint_sha256": specification["checkpoint_sha256"],
        "entry_count": 129,
        "evaluation_seed": 45,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise FinalAnalysisError(f"{model} cache manifest {field} differs")
    reducers = manifest.get("score_reducers")
    if specification["reducer"] not in _sequence(reducers, name="score reducers"):
        raise FinalAnalysisError(f"{model} cache lacks its frozen reducer")
    records = _sequence(manifest.get("records"), name=f"{model} cache records")
    if len(records) != 129 or any(not isinstance(item, Mapping) for item in records):
        raise FinalAnalysisError(f"{model} cache record coverage differs")
    identities = {
        (
            record.get("reference_scene_id"),
            record.get("master_sequence_id"),
            record.get("order_id"),
        )
        for record in records
    }
    if len(identities) != 129:
        raise FinalAnalysisError(f"{model} cache records are not unique")
    return manifest, sorted(
        records,
        key=lambda item: (
            str(item["reference_scene_id"]),
            str(item["master_sequence_id"]),
            str(item["order_id"]),
        ),
    )


def _load_bundle(
    cache_directory: Path, record: Mapping[str, object]
) -> dict[str, object]:
    import torch

    from scripts.evaluate_persist4d_allt import (
        _validate_sequence_bundle,
        sequence_cache_key_sha256,
    )

    filename = _nonempty(record.get("filename"), name="cache filename")
    if Path(filename).name != filename:
        raise FinalAnalysisError("cache filename must be a plain name")
    key_sha256 = _nonempty(record.get("key_sha256"), name="cache key hash")
    if filename != f"{key_sha256}.pt":
        raise FinalAnalysisError("cache filename and key hash differ")
    path = cache_directory / filename
    expected_bytes = _integer(record.get("bytes"), name="cache bytes", minimum=1)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
        raise FinalAnalysisError(f"cache file metadata differs: {filename}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise FinalAnalysisError(f"cache file cannot be loaded: {filename}") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("key"), Mapping):
        raise FinalAnalysisError("cache bundle root differs")
    if sequence_cache_key_sha256(value["key"]) != key_sha256:
        raise FinalAnalysisError("cache bundle key hash differs")
    bundle = _validate_sequence_bundle(value, expected_key=value["key"])
    for field in ("reference_scene_id", "master_sequence_id", "order_id"):
        if bundle["key"].get(field) != record.get(field):
            raise FinalAnalysisError(f"cache record {field} differs")
    return bundle


def _c2_identity_updates(bundle: Mapping[str, object]) -> tuple[object, ...]:
    import torch

    from scripts.evaluate_persist4d_p6a import (
        cache_payload_to_frozen_observation,
        stage_prediction_from_track_step,
    )
    from scripts.p6a_association import B4PersistentTracker
    from scripts.system_comparison_metrics import match_identity_update

    raw_payloads = _sequence(bundle.get("raw_payloads"), name="C2 raw payloads")
    if len(raw_payloads) != 5:
        raise FinalAnalysisError("C2 identity replay requires exact T1-T5 payloads")
    key = bundle["key"]
    tracker = B4PersistentTracker(
        sequence_id=str(key["master_sequence_id"]),
        capacity=100,
        class_weight=0.25,
        association_threshold=0.5,
        update_rate=0.2,
        max_update_rate=0.2,
    )
    updates = []
    for stage, raw in enumerate(raw_payloads):
        if not isinstance(raw, Mapping):
            raise FinalAnalysisError("C2 raw payload is invalid")
        target = raw.get("target")
        if not isinstance(target, Mapping) or target.get(
            "gt_class_semantics"
        ) != "rescene_model_index_0_based":
            raise FinalAnalysisError("C2 target class semantics differ")
        step = tracker.step(
            cache_payload_to_frozen_observation(raw), stage_id=stage
        )
        prediction = stage_prediction_from_track_step(raw, step, class_mapper=None)
        issued_ids = prediction["track_ids"]
        if not isinstance(issued_ids, torch.Tensor):
            raise FinalAnalysisError("C2 replay issued IDs are not integer tensors")
        updates.append(
            match_identity_update(
                horizon=stage + 1,
                gt_ids=target["gt_ids"],
                gt_classes=target["gt_classes"],
                gt_masks=target["gt_masks"],
                issued_ids=issued_ids,
                pred_classes=prediction["pred_classes"],
                pred_masks=prediction["pred_masks"],
                minimum_iou=0.5,
            )
        )
    return tuple(updates)


def _analyze_model_cache(
    *,
    model: str,
    manifest_path: Path,
    cache_directory: Path,
    dataset_spec: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
    from scripts.evaluate_persist4d_allt import _unpack_pair
    from scripts.system_comparison_metrics import (
        deployment_identity_metrics_by_horizon,
        identity_updates_from_payloads,
    )

    manifest, records = _validate_manifest(manifest_path, model=model)
    references = sorted({str(record["reference_scene_id"]) for record in records})
    if len(references) != 6:
        raise FinalAnalysisError(f"{model} reference coverage differs")
    accumulators = {
        (reference, horizon): AllTBaselineAccumulator(
            dataset_spec=dataset_spec, include_task=True
        )
        for reference in references
        for horizon in REPORT_HORIZONS
    }
    sequence_counts: dict[str, int] = defaultdict(int)
    identity_sequence_rows = []
    reducer = str(FROZEN_MODELS[model]["reducer"])
    for position, record in enumerate(records, start=1):
        bundle = _load_bundle(cache_directory, record)
        key = bundle["key"]
        reference = str(key["reference_scene_id"])
        sequence_counts[reference] += 1
        for horizon in REPORT_HORIZONS:
            accumulators[(reference, horizon)].update(
                _unpack_pair(bundle["pairs"][reducer][str(horizon)])
            )
        updates = (
            _c2_identity_updates(bundle)
            if model == "C2"
            else identity_updates_from_payloads(bundle["full_history_payloads"])
        )
        for horizon, values in deployment_identity_metrics_by_horizon(updates).items():
            identity_sequence_rows.append(
                {
                    "model": model,
                    "reference_scene_id": reference,
                    "master_sequence_id": str(key["master_sequence_id"]),
                    "order_id": str(key["order_id"]),
                    "T": horizon,
                    **{
                        field: int(values[field]) for field in IDENTITY_COUNT_FIELDS
                    },
                }
            )
        del bundle, updates
        gc.collect()
        if position % 5 == 0 or position == len(records):
            print(
                f"[allt-final-analysis] {model} {position}/{len(records)}",
                flush=True,
            )
    metric_rows = []
    for reference in references:
        for horizon in REPORT_HORIZONS:
            metric_rows.append(
                {
                    "model": model,
                    "reference_scene_id": reference,
                    "T": horizon,
                    "sequence_count": sequence_counts[reference],
                    **accumulators[(reference, horizon)].compute(),
                }
            )
    metadata = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "cache_manifest_sha256": _file_sha256(manifest_path),
        "cache_manifest_content_sha256": manifest["content_sha256"],
        "cache_directory": manifest["cache_directory"],
        "entry_count": len(records),
        "reducer": reducer,
        "source_commit_at_inference": manifest["source_commit"],
    }
    return metric_rows, identity_sequence_rows, metadata


def analyze_final_caches(
    *,
    c2_manifest: Path,
    c2_cache_directory: Path,
    fh_manifest: Path,
    fh_cache_directory: Path,
    dataset_spec: Path,
    output_root: Path,
) -> dict[str, object]:
    """Run the complete cache-only final analysis and publish compact outputs."""
    source_commit = _git_head()
    model_rows: dict[str, list[dict[str, object]]] = {}
    identity_rows = []
    metadata = {}
    for model, manifest_path, cache_directory in (
        ("C2", c2_manifest, c2_cache_directory),
        ("FH-adapt", fh_manifest, fh_cache_directory),
    ):
        metrics, identities, model_metadata = _analyze_model_cache(
            model=model,
            manifest_path=manifest_path.expanduser().resolve(),
            cache_directory=cache_directory.expanduser().resolve(),
            dataset_spec=dataset_spec.expanduser().resolve(),
        )
        model_rows[model] = metrics
        identity_rows.extend(identities)
        metadata[model] = model_metadata
    paired = build_paired_reference_rows(model_rows["C2"], model_rows["FH-adapt"])
    identity = aggregate_identity_rows(identity_rows)
    cluster = bootstrap_equal_cluster_effect(paired, seed=45, resamples=1000)
    outputs = {
        "per_reference.csv": _csv_bytes(paired, PER_REFERENCE_FIELDS),
        "identity_counts.csv": _csv_bytes(identity, IDENTITY_FIELDS),
        "cluster_effects.csv": _csv_bytes(cluster, CLUSTER_EFFECT_FIELDS),
    }
    for filename, content in outputs.items():
        _atomic_write(output_root / filename, content)
    manifest = {
        "analysis_scope": {
            "cluster_uncertainty": (
                "1000 fixed-seed bootstrap resamples of six equal-weight reference "
                "deltas; descriptive only, not a pooled-AP confidence interval"
            ),
            "identity": (
                "fresh T1-T5 recomputation; C2 replays prediction-only B4 with "
                "model-index class labels and FH-adapt uses frozen full-history IDs"
            ),
            "task_metrics": (
                "official metrics pooled within each reference; sequence AP values "
                "are never averaged to represent pooled AP"
            ),
        },
        "coverage": {
            "cluster_effect_rows": len(cluster),
            "identity_rows": len(identity),
            "models": list(FROZEN_MODELS),
            "order_units_per_model": 129,
            "per_reference_rows": len(paired),
            "reference_clusters": 6,
        },
        "dataset_spec_sha256": _file_sha256(dataset_spec.expanduser().resolve()),
        "models": metadata,
        "outputs": [
            {"path": filename, "sha256": hashlib.sha256(content).hexdigest()}
            for filename, content in outputs.items()
        ],
        "population_id": POPULATION_ID,
        "schema_version": 1,
        "source_commit": source_commit,
        "status": "pass",
    }
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    _atomic_json(output_root / "final_analysis_manifest.json", manifest)
    if _git_head() != source_commit:
        raise FinalAnalysisError("Git HEAD changed during final cache analysis")
    return manifest


def _parser() -> argparse.ArgumentParser:
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c2-manifest",
        type=Path,
        default=(
            DEFAULT_ARTIFACT_ROOT
            / "evaluation/protocol_b/C2/update=0200/cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--c2-cache-directory",
        type=Path,
        default=(
            DEFAULT_CACHE_ROOT
            / "C2"
            / str(FROZEN_MODELS["C2"]["checkpoint_sha256"])
        ),
    )
    parser.add_argument(
        "--fh-manifest",
        type=Path,
        default=(
            DEFAULT_ARTIFACT_ROOT
            / "evaluation/protocol_b/FH-adapt/update=0100/cache_manifest.json"
        ),
    )
    parser.add_argument(
        "--fh-cache-directory",
        type=Path,
        default=(
            DEFAULT_CACHE_ROOT
            / "FH-adapt"
            / str(FROZEN_MODELS["FH-adapt"]["checkpoint_sha256"])
        ),
    )
    parser.add_argument(
        "--dataset-spec",
        type=Path,
        default=resolve_metric_dataset_spec(PROJECT_ROOT),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = analyze_final_caches(
        c2_manifest=args.c2_manifest,
        c2_cache_directory=args.c2_cache_directory,
        fh_manifest=args.fh_manifest,
        fh_cache_directory=args.fh_cache_directory,
        dataset_spec=args.dataset_spec,
        output_root=args.output_root,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "CLUSTER_EFFECT_FIELDS",
    "IDENTITY_FIELDS",
    "PER_REFERENCE_FIELDS",
    "TASK_METRICS",
    "FinalAnalysisError",
    "aggregate_identity_rows",
    "analyze_final_caches",
    "bootstrap_equal_cluster_effect",
    "build_paired_reference_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
