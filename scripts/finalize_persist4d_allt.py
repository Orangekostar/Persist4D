#!/usr/bin/env python3
"""Finalize frozen Persist4D all-T evidence without changing selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1"
FROZEN_SOURCE_COMMIT = "d288af93cefc7cf8aaabb9b541b92539782c019c"
FINAL_POPULATION = "protocol_b_43_masters_3_orders"

REPORT_HORIZONS = (2, 3, 4, 5)
TASK_METRICS = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
)
RESULT_FIELDS = (
    "population_id",
    "model",
    "checkpoint_sha256",
    "training_seed",
    "evaluation_seed",
    "method",
    "reducer",
    "T",
    *TASK_METRICS,
    "local_current_AP",
    "num_master",
    "num_order_units",
    "num_reference_clusters",
)
INTEGER_RESULT_FIELDS = (
    "training_seed",
    "evaluation_seed",
    "T",
    "num_master",
    "num_order_units",
    "num_reference_clusters",
)
FLOAT_RESULT_FIELDS = (*TASK_METRICS, "local_current_AP")


class FinalizationError(RuntimeError):
    pass


def _canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise FinalizationError("value is not canonical portable JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FinalizationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot load JSON: {path}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"JSON root must be an object: {path}")
    return value


def _validated_self_hash(document: Mapping[str, object], *, name: str) -> None:
    expected = document.get("content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise FinalizationError(f"{name} lacks its content hash")
    unsigned = dict(document)
    del unsigned["content_sha256"]
    if _canonical_json_sha256(unsigned) != expected:
        raise FinalizationError(f"{name} content hash differs")


def _read_result_rows(path: Path) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
                raise FinalizationError("metric CSV fields differ")
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise FinalizationError(f"cannot load metric CSV: {path}") from error
    if not raw_rows:
        raise FinalizationError("metric CSV is empty")
    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        try:
            for field in INTEGER_RESULT_FIELDS:
                row[field] = int(str(row[field]))
            for field in FLOAT_RESULT_FIELDS:
                row[field] = float(str(row[field]))
        except (TypeError, ValueError) as error:
            raise FinalizationError("metric CSV contains invalid numbers") from error
        if any(not math.isfinite(float(row[field])) for field in FLOAT_RESULT_FIELDS):
            raise FinalizationError("metric CSV contains non-finite values")
        rows.append(row)
    return rows


def _expected_forward_count(*, window_mode: str, sequence_count: int) -> int:
    if window_mode == "local_pair":
        return 5 * sequence_count
    if window_mode == "full_history":
        return 15 * sequence_count
    raise FinalizationError("evaluation window mode differs")


def validate_evaluation_artifacts(
    *,
    evaluation_root: Path,
    expected_variant: str,
    expected_checkpoint: str,
    expected_step: int,
    expected_source_commit: str,
    expected_reducers: Sequence[str],
) -> list[dict[str, object]]:
    metrics_path = evaluation_root / "all_t_metrics.csv"
    summary = _load_json(evaluation_root / "run_summary.json")
    manifest = _load_json(evaluation_root / "cache_manifest.json")
    reducers = tuple(expected_reducers)
    expected_window = "full_history" if reducers == ("official",) else "local_pair"
    expected_method = "FullHistory" if expected_window == "full_history" else "B4"
    expected_summary = {
        "status": "pass",
        "variant": expected_variant,
        "model": expected_variant,
        "checkpoint_sha256": expected_checkpoint,
        "checkpoint_global_step": expected_step,
        "source_commit": expected_source_commit,
        "population_id": FINAL_POPULATION,
        "sequence_count": 129,
        "new_sequence_count": 129,
        "reused_sequence_count": 0,
        "training_seed": 45,
        "evaluation_seed": 45,
        "window_mode": expected_window,
        "reducers": list(reducers),
        "metric_row_count": 4 * len(reducers),
        "model_forward_count": _expected_forward_count(
            window_mode=expected_window, sequence_count=129
        ),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            label = "source commit" if field == "source_commit" else field
            raise FinalizationError(f"evaluation {label} differs")
    if summary.get("metric_sha256") != _file_sha256(metrics_path):
        raise FinalizationError("evaluation metric hash differs")

    _validated_self_hash(manifest, name="cache manifest")
    expected_manifest = {
        "status": "pass",
        "checkpoint_sha256": expected_checkpoint,
        "source_commit": expected_source_commit,
        "population_id": FINAL_POPULATION,
        "entry_count": 129,
        "reused_entry_count": 0,
        "evaluation_seed": 45,
        "score_reducers": list(reducers),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise FinalizationError(f"cache manifest {field} differs")
    records = manifest.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FinalizationError("cache manifest records are invalid")
    if len(records) != 129:
        raise FinalizationError("cache manifest coverage differs")
    filenames = set()
    keys = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise FinalizationError("cache manifest record is invalid")
        filename = record.get("filename")
        key = record.get("key_sha256")
        if (
            not isinstance(filename, str)
            or not isinstance(key, str)
            or filename != f"{key}.pt"
            or not isinstance(record.get("bytes"), int)
            or int(record["bytes"]) <= 0
            or not isinstance(record.get("file_sha256"), str)
            or len(str(record["file_sha256"])) != 64
        ):
            raise FinalizationError("cache manifest record fields differ")
        filenames.add(filename)
        keys.add(key)
    if len(filenames) != 129 or len(keys) != 129:
        raise FinalizationError("cache manifest contains duplicate records")

    rows = _read_result_rows(metrics_path)
    if len(rows) != 4 * len(reducers):
        raise FinalizationError("metric row coverage differs")
    seen = set()
    for row in rows:
        if (
            row["population_id"] != FINAL_POPULATION
            or row["model"] != expected_variant
            or row["checkpoint_sha256"] != expected_checkpoint
            or row["training_seed"] != 45
            or row["evaluation_seed"] != 45
            or row["method"] != expected_method
            or row["reducer"] not in reducers
            or row["num_master"] != 43
            or row["num_order_units"] != 129
            or row["num_reference_clusters"] != 6
        ):
            raise FinalizationError("metric row binding differs")
        key = (row["reducer"], row["T"])
        if key in seen:
            raise FinalizationError("metric rows contain duplicate cells")
        seen.add(key)
    expected_cells = {
        (reducer, horizon) for reducer in reducers for horizon in REPORT_HORIZONS
    }
    if seen != expected_cells:
        raise FinalizationError("metric rows do not cover exact T2-T5")
    return rows


def _index_rows(
    rows: Sequence[Mapping[str, object]], *, name: str
) -> dict[int, Mapping[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise FinalizationError(f"{name} rows must be a sequence")
    checkpoints = {row.get("checkpoint_sha256") for row in rows}
    if len(checkpoints) != 1:
        raise FinalizationError(f"{name} must use one checkpoint for every horizon")
    indexed: dict[int, Mapping[str, object]] = {}
    for row in rows:
        horizon = row.get("T")
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise FinalizationError(f"{name} horizon is invalid")
        if horizon in indexed:
            raise FinalizationError(f"{name} contains a duplicate horizon")
        for metric in TASK_METRICS:
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise FinalizationError(f"{name} {metric} is invalid")
        indexed[horizon] = row
    if tuple(sorted(indexed)) != REPORT_HORIZONS:
        raise FinalizationError(f"{name} must cover exactly T2-T5")
    return indexed


def build_delta_rows(
    *,
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    candidate: str,
    baseline: str,
    evidence_scope: str,
) -> list[dict[str, object]]:
    if not candidate or not baseline or not evidence_scope:
        raise FinalizationError("delta identities must be non-empty")
    candidate_by_horizon = _index_rows(candidate_rows, name="candidate")
    baseline_by_horizon = _index_rows(baseline_rows, name="baseline")
    output = []
    for metric in TASK_METRICS:
        for horizon in REPORT_HORIZONS:
            delta = float(candidate_by_horizon[horizon][metric]) - float(
                baseline_by_horizon[horizon][metric]
            )
            output.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "T": horizon,
                    "raw_delta": delta,
                    "delta_pp": 100.0 * delta,
                    "positive": delta > 0.0,
                    "evidence_scope": evidence_scope,
                }
            )
    return output


def build_verdict(delta_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if isinstance(delta_rows, (str, bytes)) or not isinstance(delta_rows, Sequence):
        raise FinalizationError("delta rows must be a sequence")
    indexed: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in delta_rows:
        metric = row.get("metric")
        horizon = row.get("T")
        if metric not in TASK_METRICS or horizon not in REPORT_HORIZONS:
            raise FinalizationError("delta row metric cell is invalid")
        key = (str(metric), int(horizon))
        if key in indexed or type(row.get("positive")) is not bool:
            raise FinalizationError("delta rows contain duplicate or invalid cells")
        indexed[key] = row
    expected = {
        (metric, horizon) for metric in TASK_METRICS for horizon in REPORT_HORIZONS
    }
    if set(indexed) != expected:
        raise FinalizationError("delta rows must cover all 20 task cells")
    failed = [
        {"metric": metric, "T": horizon}
        for metric in TASK_METRICS
        for horizon in REPORT_HORIZONS
        if not indexed[(metric, horizon)]["positive"]
    ]
    tmap_pass = all(indexed[("t_mAP", horizon)]["positive"] for horizon in REPORT_HORIZONS)
    return {
        "failed_cells": failed,
        "strict_zero_tolerance": True,
        "task_metrics_all_t": "PASS" if not failed else "FAIL",
        "tmap_all_t": "PASS" if tmap_pass else "FAIL",
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FinalizationError(f"output cannot be a symlink: {path}")
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


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    content = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _atomic_write(path, content)


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=tuple(fieldnames),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def _validate_frozen_selection(path: Path) -> dict[str, object]:
    selection = _load_json(path)
    _validated_self_hash(selection, name="frozen selection")
    candidate = selection.get("selected_candidate")
    matched = selection.get("selected_matched_rescene")
    if (
        selection.get("status") != "frozen"
        or selection.get("protocol_b_used_for_selection") is not False
        or selection.get("protocol_b_evaluation")
        != "authorized_after_selection_freeze"
        or selection.get("training_seed") != 45
        or selection.get("evaluation_seed") != 45
        or selection.get("seed46_training_confirmation") != "gate_skipped"
        or not isinstance(candidate, Mapping)
        or candidate.get("variant") != "C2"
        or candidate.get("checkpoint_global_step") != 200
        or not isinstance(matched, Mapping)
        or matched.get("variant") != "FH-adapt"
        or matched.get("checkpoint_global_step") != 100
    ):
        raise FinalizationError("frozen selection contract differs")
    return selection


def _validate_baseline(artifact_root: Path) -> list[dict[str, object]]:
    contract = _load_json(artifact_root / "run_contract.json")
    replay = _load_json(artifact_root / "baseline/replay_status.json")
    _validated_self_hash(contract, name="run contract")
    _validated_self_hash(replay, name="baseline replay")
    checkpoint = contract.get("input_identities", {}).get("r1_checkpoint", {})
    if not isinstance(checkpoint, Mapping):
        raise FinalizationError("baseline checkpoint contract is invalid")
    checkpoint_sha256 = checkpoint.get("sha256")
    if (
        replay.get("status") != "pass"
        or replay.get("checkpoint_sha256") != checkpoint_sha256
        or replay.get("model_forward_count") != 0
        or replay.get("new_prediction_count") != 0
        or replay.get("derived_metric_rows") != 28
    ):
        raise FinalizationError("baseline replay binding differs")
    rows = _read_result_rows(artifact_root / "baseline/all_t_metrics.csv")
    expected_groups = {
        (method, reducer)
        for method, reducers in (
            ("B2", ("mean", "latest", "max")),
            ("B4", ("mean", "latest", "max")),
            ("FullHistory", ("official",)),
        )
        for reducer in reducers
    }
    cells = set()
    for row in rows:
        if (
            row["population_id"] != "protocol_b_129_units"
            or row["model"] != "R1_epoch390"
            or row["checkpoint_sha256"] != checkpoint_sha256
            or row["training_seed"] != 45
            or row["evaluation_seed"] != 45
            or row["num_master"] != 43
            or row["num_order_units"] != 129
            or row["num_reference_clusters"] != 6
        ):
            raise FinalizationError("baseline metric binding differs")
        cell = (row["method"], row["reducer"], row["T"])
        if cell in cells:
            raise FinalizationError("baseline contains duplicate metric cells")
        cells.add(cell)
    expected_cells = {
        (method, reducer, horizon)
        for method, reducer in expected_groups
        for horizon in REPORT_HORIZONS
    }
    if cells != expected_cells:
        raise FinalizationError("baseline metric coverage differs")
    return rows


def _select_rows(
    rows: Sequence[Mapping[str, object]], *, method: str, reducer: str
) -> list[Mapping[str, object]]:
    selected = [
        row
        for row in rows
        if row.get("method") == method and row.get("reducer") == reducer
    ]
    if len(selected) != len(REPORT_HORIZONS):
        raise FinalizationError(f"{method}/{reducer} metric coverage differs")
    return selected


def _comparison_summary(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    verdict = build_verdict(rows)
    tmap_rows = [row for row in rows if row["metric"] == "t_mAP"]
    return {
        **verdict,
        "minimum_t_map_delta": min(float(row["raw_delta"]) for row in tmap_rows),
        "t_map_delta_by_horizon": {
            str(row["T"]): float(row["raw_delta"])
            for row in sorted(tmap_rows, key=lambda value: int(value["T"]))
        },
    }


def validate_external_cache_files(
    *,
    cache_root: Path,
    evaluation_root: Path,
    variant: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    manifest = _load_json(evaluation_root / "cache_manifest.json")
    records = manifest["records"]
    if not isinstance(records, Sequence):
        raise FinalizationError("cache manifest records are invalid")
    directory = cache_root / FINAL_POPULATION / variant / checkpoint_sha256
    total_bytes = 0
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise FinalizationError("cache manifest record is invalid")
        path = directory / str(record["filename"])
        try:
            stat = path.stat()
        except OSError as error:
            raise FinalizationError(f"cache file is unavailable: {path}") from error
        if path.is_symlink() or not path.is_file() or stat.st_size != record["bytes"]:
            raise FinalizationError(f"cache file identity differs: {path}")
        if _file_sha256(path) != record["file_sha256"]:
            raise FinalizationError(f"cache file hash differs: {path}")
        total_bytes += stat.st_size
        if index % 10 == 0 or index == len(records):
            print(
                f"[allt-finalize] {variant} cache {index}/{len(records)}",
                flush=True,
            )
    return {
        "bytes": total_bytes,
        "entry_count": len(records),
        "status": "pass",
    }


def finalize(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    cache_root: Path | None = None,
    expected_source_commit: str = FROZEN_SOURCE_COMMIT,
) -> dict[str, object]:
    artifact_root = artifact_root.expanduser().resolve()
    selection_path = artifact_root / "selection/FROZEN_SELECTION.json"
    selection = _validate_frozen_selection(selection_path)
    candidate = selection["selected_candidate"]
    matched = selection["selected_matched_rescene"]
    if not isinstance(candidate, Mapping) or not isinstance(matched, Mapping):
        raise FinalizationError("frozen selections are invalid")
    candidate_checkpoint = str(candidate["checkpoint_sha256"])
    matched_checkpoint = str(matched["checkpoint_sha256"])
    candidate_root = artifact_root / "evaluation/protocol_b/C2/update=0200"
    matched_root = artifact_root / "evaluation/protocol_b/FH-adapt/update=0100"
    candidate_rows = validate_evaluation_artifacts(
        evaluation_root=candidate_root,
        expected_variant="C2",
        expected_checkpoint=candidate_checkpoint,
        expected_step=200,
        expected_source_commit=expected_source_commit,
        expected_reducers=("mean", "latest", "max"),
    )
    matched_rows = validate_evaluation_artifacts(
        evaluation_root=matched_root,
        expected_variant="FH-adapt",
        expected_checkpoint=matched_checkpoint,
        expected_step=100,
        expected_source_commit=expected_source_commit,
        expected_reducers=("official",),
    )
    baseline_rows = _validate_baseline(artifact_root)

    comparison_specs = (
        (
            "C2_vs_R1_B4",
            _select_rows(candidate_rows, method="B4", reducer="mean"),
            _select_rows(baseline_rows, method="B4", reducer="mean"),
            "C2_mean",
            "R1_B4_mean",
        ),
        (
            "C2_vs_R1_FullHistory",
            _select_rows(candidate_rows, method="B4", reducer="mean"),
            _select_rows(baseline_rows, method="FullHistory", reducer="official"),
            "C2_mean",
            "R1_FullHistory",
        ),
        (
            "C2_vs_FH-adapt",
            _select_rows(candidate_rows, method="B4", reducer="mean"),
            _select_rows(matched_rows, method="FullHistory", reducer="official"),
            "C2_mean",
            "FH-adapt",
        ),
        (
            "FH-adapt_vs_R1_FullHistory",
            _select_rows(matched_rows, method="FullHistory", reducer="official"),
            _select_rows(baseline_rows, method="FullHistory", reducer="official"),
            "FH-adapt",
            "R1_FullHistory",
        ),
    )
    delta_rows: list[dict[str, object]] = []
    comparison_rows: dict[str, list[dict[str, object]]] = {}
    for name, proposed, reference, proposed_name, reference_name in comparison_specs:
        rows = build_delta_rows(
            candidate_rows=proposed,
            baseline_rows=reference,
            candidate=proposed_name,
            baseline=reference_name,
            evidence_scope=FINAL_POPULATION,
        )
        for row in rows:
            row["comparison"] = name
        comparison_rows[name] = rows
        delta_rows.extend(rows)

    sensitivity_rows: list[dict[str, object]] = []
    for reducer in ("mean", "latest", "max"):
        rows = build_delta_rows(
            candidate_rows=_select_rows(
                candidate_rows, method="B4", reducer=reducer
            ),
            baseline_rows=_select_rows(
                baseline_rows, method="B4", reducer=reducer
            ),
            candidate=f"C2_{reducer}",
            baseline=f"R1_B4_{reducer}",
            evidence_scope=FINAL_POPULATION,
        )
        for row in rows:
            row["reducer"] = reducer
        sensitivity_rows.extend(rows)

    comparisons = {
        name: _comparison_summary(rows) for name, rows in comparison_rows.items()
    }
    primary = comparisons["C2_vs_FH-adapt"]
    verdict = {
        "comparisons": comparisons,
        "evaluation_source_commit": expected_source_commit,
        "goal": "C2 strictly exceeds matched FH-adapt in t_mAP at T2-T5",
        "goal_status": primary["tmap_all_t"],
        "independent_generalization": "NOT_ESTABLISHED",
        "primary_comparison": "C2_vs_FH-adapt",
        "protocol_b_used_for_selection": False,
        "schema_version": 1,
        "strict_zero_tolerance": True,
    }
    verdict["content_sha256"] = _canonical_json_sha256(verdict)

    output_root = artifact_root / "results"
    combined_rows = sorted(
        [*baseline_rows, *candidate_rows, *matched_rows],
        key=lambda row: (
            str(row["model"]),
            str(row["method"]),
            str(row["reducer"]),
            int(row["T"]),
        ),
    )
    delta_fields = (
        "comparison",
        "candidate",
        "baseline",
        "metric",
        "T",
        "raw_delta",
        "delta_pp",
        "positive",
        "evidence_scope",
    )
    sensitivity_fields = (
        "reducer",
        "candidate",
        "baseline",
        "metric",
        "T",
        "raw_delta",
        "delta_pp",
        "positive",
        "evidence_scope",
    )
    confirmation_rows = (
        {
            "training_seed": 45,
            "status": "complete",
            "role": "formal_selection_and_final_evaluation",
            "reason": "frozen_primary_run",
        },
        {
            "training_seed": 46,
            "status": "gate_skipped",
            "role": "training_confirmation",
            "reason": "development_candidate_vs_matched_rescene_not_all_t_positive",
        },
    )
    outputs = {
        "all_t_metrics.csv": _csv_bytes(combined_rows, fieldnames=RESULT_FIELDS),
        "deltas_all_metrics.csv": _csv_bytes(delta_rows, fieldnames=delta_fields),
        "score_sensitivity.csv": _csv_bytes(
            sensitivity_rows, fieldnames=sensitivity_fields
        ),
        "run_confirmation.csv": _csv_bytes(
            confirmation_rows,
            fieldnames=("training_seed", "status", "role", "reason"),
        ),
    }
    for name, content in outputs.items():
        _atomic_write(output_root / name, content)
    _atomic_json(output_root / "verdict.json", verdict)

    cache_validation: dict[str, object]
    if cache_root is None:
        cache_validation = {"status": "not_requested"}
    else:
        resolved_cache = cache_root.expanduser().resolve()
        cache_validation = {
            "C2": validate_external_cache_files(
                cache_root=resolved_cache,
                evaluation_root=candidate_root,
                variant="C2",
                checkpoint_sha256=candidate_checkpoint,
            ),
            "FH-adapt": validate_external_cache_files(
                cache_root=resolved_cache,
                evaluation_root=matched_root,
                variant="FH-adapt",
                checkpoint_sha256=matched_checkpoint,
            ),
            "status": "pass",
        }

    input_paths = (
        selection_path,
        artifact_root / "baseline/all_t_metrics.csv",
        artifact_root / "baseline/replay_status.json",
        candidate_root / "all_t_metrics.csv",
        candidate_root / "cache_manifest.json",
        candidate_root / "run_summary.json",
        matched_root / "all_t_metrics.csv",
        matched_root / "cache_manifest.json",
        matched_root / "run_summary.json",
    )
    manifest = {
        "cache_validation": cache_validation,
        "evaluation_source_commit": expected_source_commit,
        "finalizer_sha256": _file_sha256(Path(__file__).resolve()),
        "inputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _file_sha256(path),
            }
            for path in input_paths
        ],
        "outputs": [
            {"path": name, "sha256": _file_sha256(output_root / name)}
            for name in (*outputs, "verdict.json")
        ],
        "schema_version": 1,
        "status": "pass",
    }
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    _atomic_json(output_root / "evidence_manifest.json", manifest)
    return verdict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize frozen Persist4D all-T Protocol-B evidence."
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--expected-source-commit", default=FROZEN_SOURCE_COMMIT
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    verdict = finalize(
        artifact_root=arguments.artifact_root,
        cache_root=arguments.cache_root,
        expected_source_commit=arguments.expected_source_commit,
    )
    print(json.dumps(verdict, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
