#!/usr/bin/env python3
"""Validate, summarize, and gate formal ReScene decoder diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import statistics
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_evaluation import RootCauseEvaluationError
from utils.rescene_rootcause_preflight import canonical_sha256

DIAGNOSTIC_MODES = (
    "query_initialization",
    "query_conflicts",
    "attention_mask_recall",
    "superpoint_features",
)
THRESHOLDS = {
    "superpoint_within_variance_poor": 0.05,
    "superpoint_nearest_margin_weak": 0.10,
    "query_competed_fraction_substantial": 0.25,
    "query_count_per_gt_substantial": 2.0,
    "early_attention_allowed_fraction_low": 0.50,
    "early_attention_severe_recall": 0.25,
    "early_attention_severe_fraction": 0.25,
    "attention_reset_fraction_frequent": 0.10,
}


def _float(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("diagnostic summary value is invalid") from error
    if not math.isfinite(value):
        raise RootCauseEvaluationError("diagnostic summary value is invalid")
    return value


def _int(row: Mapping[str, Any], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("diagnostic summary value is invalid") from error
    return value


def _bool(row: Mapping[str, Any], field: str) -> bool:
    value = row.get(field)
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise RootCauseEvaluationError("diagnostic summary value is invalid")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RootCauseEvaluationError("diagnostic summary population is empty")
    return statistics.mean(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise RootCauseEvaluationError("diagnostic quantile population is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_decoder_diagnostics(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, object]:
    """Compute preregistered descriptive summaries and evidence-only gates."""

    if set(tables) != set(DIAGNOSTIC_MODES) or any(
        not rows for rows in tables.values()
    ):
        raise RootCauseEvaluationError("diagnostic table matrix differs")

    query_rows = tables["query_initialization"]
    query_scenes = [
        row for row in query_rows if row.get("record_type") == "scene_summary"
    ]
    query_instances = [
        row for row in query_rows if row.get("record_type") == "gt_instance"
    ]
    if not query_scenes or not query_instances:
        raise RootCauseEvaluationError("query initialization table differs")
    query_summary: dict[str, object] = {
        "scene_count": len(query_scenes),
        "gt_instance_count": len(query_instances),
        "foreground_query_fraction_mean": _mean(
            [_float(row, "foreground_query_fraction") for row in query_scenes]
        ),
        "background_query_fraction_mean": _mean(
            [_float(row, "background_query_fraction") for row in query_scenes]
        ),
        "gt_instance_coverage_mean": _mean(
            [_float(row, "gt_instance_coverage") for row in query_scenes]
        ),
        "query_content_norm_mean": _mean(
            [_float(row, "query_content_norm_mean") for row in query_scenes]
        ),
        "query_content_zero_fraction_mean": _mean(
            [_float(row, "query_content_zero_fraction") for row in query_scenes]
        ),
    }
    for size_bin in ("small_lt100", "medium_100_999", "large_ge1000"):
        rows = [row for row in query_instances if row.get("size_bin") == size_bin]
        query_summary[f"{size_bin}_instance_count"] = len(rows)
        query_summary[f"{size_bin}_coverage"] = (
            _mean([float(_bool(row, "covered_by_fps_query")) for row in rows])
            if rows
            else None
        )

    conflict_rows = tables["query_conflicts"]
    feeding_rows = [row for row in conflict_rows if _bool(row, "feeds_next_attention")]
    if not feeding_rows:
        raise RootCauseEvaluationError("query conflict table differs")
    conflict_summary = {
        "row_count": len(conflict_rows),
        "attention_feeding_row_count": len(feeding_rows),
        "competed_active_query_fraction_mean": _mean(
            [_float(row, "competed_active_query_fraction") for row in feeding_rows]
        ),
        "mean_queries_per_gt_iou25_mean": _mean(
            [_float(row, "mean_queries_per_gt_iou25") for row in feeding_rows]
        ),
        "gt_coverage_iou25_mean": _mean(
            [_float(row, "gt_coverage_iou25") for row in feeding_rows]
        ),
        "gt_coverage_iou50_mean": _mean(
            [_float(row, "gt_coverage_iou50") for row in feeding_rows]
        ),
        "query_utilization_iou25_mean": _mean(
            [_float(row, "query_utilization_iou25") for row in feeding_rows]
        ),
        "competing_query_pairwise_iou_mean": _mean(
            [_float(row, "competing_query_pairwise_iou_mean") for row in feeding_rows]
        ),
    }

    attention_rows = tables["attention_mask_recall"]
    first_layer = min(_int(row, "decoder_prediction_layer") for row in attention_rows)
    early_rows = [
        row
        for row in attention_rows
        if _int(row, "decoder_prediction_layer") == first_layer
    ]
    reset_groups: dict[tuple[str, int], float] = {}
    for row in attention_rows:
        key = (str(row["file_name"]), _int(row, "decoder_prediction_layer"))
        value = _float(row, "post_sample_reset_fraction")
        if key in reset_groups and not math.isclose(
            reset_groups[key], value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RootCauseEvaluationError("attention reset table differs")
        reset_groups[key] = value
    early_allowed = [_float(row, "allowed_gt_fraction") for row in early_rows]
    attention_summary = {
        "row_count": len(attention_rows),
        "earliest_layer": first_layer,
        "earliest_allowed_gt_fraction_mean": _mean(early_allowed),
        "earliest_severe_recall_fraction": _mean(
            [
                float(value < THRESHOLDS["early_attention_severe_recall"])
                for value in early_allowed
            ]
        ),
        "allowed_gt_fraction_mean_all_layers": _mean(
            [_float(row, "allowed_gt_fraction") for row in attention_rows]
        ),
        "post_sample_reset_fraction_mean": _mean(list(reset_groups.values())),
    }

    superpoint_rows = tables["superpoint_features"]
    margins = [
        _float(row, "nearest_instance_cosine_margin")
        for row in superpoint_rows
        if row.get("nearest_instance_cosine_margin") not in (None, "")
    ]
    if not margins:
        raise RootCauseEvaluationError("superpoint margin population is empty")
    superpoint_summary = {
        "gt_instance_count": len(superpoint_rows),
        "within_variance_mean": _mean(
            [_float(row, "within_instance_feature_variance") for row in superpoint_rows]
        ),
        "nearest_margin_mean": _mean(margins),
        "nearest_margin_p25": _quantile(margins, 0.25),
        "segment_purity_mean": _mean(
            [_float(row, "mean_segment_purity") for row in superpoint_rows]
        ),
        "gt_instances_per_segment_mean": _mean(
            [_float(row, "mean_gt_instances_per_segment") for row in superpoint_rows]
        ),
        "segments_per_gt_mean": _mean(
            [_float(row, "segments_per_gt") for row in superpoint_rows]
        ),
    }

    superpoint_evidence = (
        superpoint_summary["within_variance_mean"]
        >= THRESHOLDS["superpoint_within_variance_poor"]
        or superpoint_summary["nearest_margin_p25"]
        <= THRESHOLDS["superpoint_nearest_margin_weak"]
    )
    conflict_evidence = (
        conflict_summary["competed_active_query_fraction_mean"]
        >= THRESHOLDS["query_competed_fraction_substantial"]
        and conflict_summary["mean_queries_per_gt_iou25_mean"]
        >= THRESHOLDS["query_count_per_gt_substantial"]
    )
    starvation_evidence = (
        attention_summary["earliest_allowed_gt_fraction_mean"]
        <= THRESHOLDS["early_attention_allowed_fraction_low"]
        or attention_summary["earliest_severe_recall_fraction"]
        >= THRESHOLDS["early_attention_severe_fraction"]
        or attention_summary["post_sample_reset_fraction_mean"]
        >= THRESHOLDS["attention_reset_fraction_frequent"]
    )
    return {
        "query_initialization": query_summary,
        "query_conflicts": conflict_summary,
        "attention_mask_recall": attention_summary,
        "superpoint_features": superpoint_summary,
        "thresholds": THRESHOLDS,
        "gates": {
            "A1": {
                "authorized": True,
                "reason": "required first ReScene-native structural experiment after SD0",
            },
            "A2": {
                "diagnostic_evidence_pass": superpoint_evidence,
                "authorized": False,
                "status": (
                    "pending_A1_result" if superpoint_evidence else "gate_skipped"
                ),
            },
            "query_competition_design": {
                "supported": conflict_evidence,
                "implementation_authorized": False,
            },
            "attention_relaxation_design": {
                "supported": starvation_evidence,
                "implementation_authorized": False,
            },
        },
    }


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _file_identity(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.lstat()
    except OSError as error:
        raise RootCauseEvaluationError(
            "diagnostic finalization input is unavailable"
        ) from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise RootCauseEvaluationError(
            "diagnostic finalization input changed while hashing"
        )
    return {"bytes": size, "sha256": digest.hexdigest()}


def _load_formal_tables(
    input_directory: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    from utils.rescene_rootcause_diagnostic_runtime import MODE_FIELDS

    tables = {}
    sources = {}
    bindings = {
        field: set()
        for field in (
            "variant",
            "completed_epoch",
            "source_commit",
            "checkpoint_sha256",
            "checkpoint_manifest_sha256",
            "variant_authorization_sha256",
        )
    }
    for mode in DIAGNOSTIC_MODES:
        csv_path = input_directory / f"{mode}.csv"
        manifest_path = input_directory / f"{mode}.manifest.json"
        manifest = _load_json(manifest_path, name="diagnostic manifest")
        content_sha256 = manifest.get("content_sha256")
        unsigned = dict(manifest)
        unsigned.pop("content_sha256", None)
        contract = manifest.get("contract")
        if (
            not isinstance(content_sha256, str)
            or canonical_sha256(unsigned) != content_sha256
            or manifest.get("status") != "pass"
            or manifest.get("scope") != "official_like_t2"
            or manifest.get("mode") != mode
            or manifest.get("seed") != 45
            or manifest.get("validation_sequence_count") != 154
            or not isinstance(contract, Mapping)
            or canonical_sha256(
                {key: value for key, value in contract.items() if key != "sha256"}
            )
            != contract.get("sha256")
        ):
            raise RootCauseEvaluationError("diagnostic manifest binding differs")
        csv_identity = _file_identity(csv_path)
        if csv_identity["sha256"] != manifest.get("csv_sha256"):
            raise RootCauseEvaluationError("diagnostic CSV hash differs")
        try:
            with csv_path.open(encoding="ascii", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != tuple(MODE_FIELDS[mode]):
                    raise RootCauseEvaluationError("diagnostic CSV schema differs")
                rows = [dict(row) for row in reader]
        except OSError as error:
            raise RootCauseEvaluationError("diagnostic CSV is unreadable") from error
        if (
            len(rows) != manifest.get("row_count")
            or len({row.get("file_name") for row in rows}) != 154
        ):
            raise RootCauseEvaluationError("diagnostic CSV coverage differs")
        tables[mode] = rows
        sources[mode] = {
            "csv": csv_identity,
            "manifest": _file_identity(manifest_path),
            "contract_sha256": contract["sha256"],
            "manifest_content_sha256": content_sha256,
        }
        for field, values in bindings.items():
            values.add(manifest.get(field))
    if any(len(values) != 1 for values in bindings.values()):
        raise RootCauseEvaluationError("diagnostic cross-mode binding differs")
    return tables, {
        "sources": sources,
        "bindings": {field: next(iter(values)) for field, values in bindings.items()},
    }


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _markdown(summary: Mapping[str, Any]) -> bytes:
    gates = summary["gates"]
    lines = [
        "# ReScene Decoder Diagnostics",
        "",
        "Scope: project diagnostics on the official-like 154-sequence T2 split.",
        "These values are not CompetitorFormer, LaSSM, or Relation3D metrics.",
        "",
        "## Query Initialization",
        "",
        f"- Foreground FPS fraction: `{summary['query_initialization']['foreground_query_fraction_mean']}`",
        f"- GT instance coverage: `{summary['query_initialization']['gt_instance_coverage_mean']}`",
        f"- Zero query-content fraction: `{summary['query_initialization']['query_content_zero_fraction_mean']}`",
        "",
        "## Evidence Gates",
        "",
        f"- A1 authorized: `{str(gates['A1']['authorized']).lower()}`",
        f"- A2 diagnostic evidence: `{str(gates['A2']['diagnostic_evidence_pass']).lower()}`",
        f"- A2 status: `{gates['A2']['status']}`",
        f"- Query-competition design supported: `{str(gates['query_competition_design']['supported']).lower()}`",
        f"- Attention-relaxation design supported: `{str(gates['attention_relaxation_design']['supported']).lower()}`",
        "",
        "A2 and both high-risk implementations remain unauthorized at SD0.",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite diagnostic summary")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    tables, provenance = _load_formal_tables(arguments.input_dir)
    summary = summarize_decoder_diagnostics(tables)
    summary.update(
        {
            "schema_version": 1,
            "status": "pass",
            "experiment": "rescene_task_learning_root_cause_v1",
            "provenance": provenance,
        }
    )
    summary["content_sha256"] = canonical_sha256(summary)
    _publish(arguments.output_json, _json_bytes(summary))
    _publish(arguments.output_report, _markdown(summary))
    print(
        json.dumps(
            {
                "A2_status": summary["gates"]["A2"]["status"],
                "content_sha256": summary["content_sha256"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
