#!/usr/bin/env python3
"""Finalize three-seed RC4 evaluation against the frozen Concerto baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    METRIC_NAMES,
    RootCauseEvaluationError,
    validate_checkpoint_manifest_binding,
)
from utils.rescene_rootcause_preflight import canonical_sha256

FROZEN_CONCERTO_CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)


def _metrics(row: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for name in METRIC_NAMES:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise RootCauseEvaluationError(
                "full evaluation metric is invalid"
            ) from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RootCauseEvaluationError("full evaluation metric is invalid")
        result[name] = value
    return result


def _normalized_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    try:
        seeds = [int(row["seed"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("full evaluation seed matrix differs") from error
    if len(seeds) != 3 or set(seeds) != set(EVALUATION_SEEDS):
        raise RootCauseEvaluationError("full evaluation seed matrix differs")
    return {int(row["seed"]): _metrics(row) for row in rows}


def _summary(rows: Mapping[int, Mapping[str, float]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"seed_count": len(rows)}
    for metric in METRIC_NAMES:
        values = [rows[seed][metric] for seed in EVALUATION_SEEDS]
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = statistics.stdev(values)
    spatial = [
        (rows[seed]["stage1_mAP"] + rows[seed]["stage2_mAP"]) / 2.0
        for seed in EVALUATION_SEEDS
    ]
    result["SpatialStageMean_mean"] = statistics.mean(spatial)
    result["SpatialStageMean_std"] = statistics.stdev(spatial)
    return result


def classify_full_result(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    verdict_prefix: str = "ROOTCAUSE",
) -> dict[str, object]:
    """Apply the registered RC4 scientific verdict without Persist4D inputs."""

    if verdict_prefix not in {"ROOTCAUSE", "STRONG-LOCAL"}:
        raise RootCauseEvaluationError("full evaluation verdict prefix is invalid")
    candidate = _normalized_rows(candidate_rows)
    baseline = _normalized_rows(baseline_rows)
    candidate_summary = _summary(candidate)
    baseline_summary = _summary(baseline)
    paired_spatial_deltas = [
        (candidate[seed]["stage1_mAP"] + candidate[seed]["stage2_mAP"]) / 2.0
        - (baseline[seed]["stage1_mAP"] + baseline[seed]["stage2_mAP"]) / 2.0
        for seed in EVALUATION_SEEDS
    ]
    delta_mean = statistics.mean(paired_spatial_deltas)
    gates = {
        "stage1_mean_improved": candidate_summary["stage1_mAP_mean"]
        > baseline_summary["stage1_mAP_mean"],
        "stage2_mean_improved": candidate_summary["stage2_mAP_mean"]
        > baseline_summary["stage2_mAP_mean"],
        "overall_mean_improved": candidate_summary["overall_mAP_mean"]
        > baseline_summary["overall_mAP_mean"],
        "t_mAP_mean_improved": candidate_summary["t_mAP_mean"]
        > baseline_summary["t_mAP_mean"],
        "paired_spatial_positive_all_seeds": all(
            value > 0.0 for value in paired_spatial_deltas
        ),
        "material_spatial_gain": delta_mean >= 0.01,
    }
    confirmed = all(
        gates[name]
        for name in (
            "stage1_mean_improved",
            "stage2_mean_improved",
            "overall_mean_improved",
            "t_mAP_mean_improved",
            "paired_spatial_positive_all_seeds",
        )
    )
    verdict = (
        f"{verdict_prefix}-CONFIRMED"
        if confirmed
        else f"{verdict_prefix}-PARTIAL"
        if gates["material_spatial_gain"]
        else f"{verdict_prefix}-NOT-CONFIRMED"
    )
    return {
        "verdict": verdict,
        "verdict_prefix": verdict_prefix,
        "gates": gates,
        "paired_spatial_deltas": paired_spatial_deltas,
        "paired_spatial_delta_mean": delta_mean,
        "candidate_summary": candidate_summary,
        "baseline_summary": baseline_summary,
        "selection_used_persist4d": False,
    }


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _validate_hash(payload: Mapping[str, Any], field: str, *, name: str) -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError(f"{name} hash differs")


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
            "full evaluation input is unavailable"
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
        raise RootCauseEvaluationError("full evaluation input changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _candidate_runs(
    paths: Sequence[Path],
    *,
    variant: str,
    completed_epoch: int,
    authorization_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_sha256: str,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows = []
    sources = {}
    provenance = {
        field: set()
        for field in ("source_commit", "contract_sha256", "evaluation_config_sha256")
    }
    for path in paths:
        run = _load_json(path, name="full official-like run")
        if (
            run.get("status") != "pass"
            or run.get("scope") != "official_like_t2"
            or run.get("variant") != variant
            or run.get("completed_epoch") != completed_epoch
            or run.get("validation_sequence_count") != 154
            or run.get("variant_authorization_sha256") != authorization_sha256
            or run.get("checkpoint_manifest_sha256") != checkpoint_manifest_sha256
            or run.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise RootCauseEvaluationError("full official-like run binding differs")
        metrics = _metrics(run.get("metrics", {}))
        spatial = (metrics["stage1_mAP"] + metrics["stage2_mAP"]) / 2.0
        try:
            reported_spatial = float(run["SpatialStageMean"])
        except (KeyError, TypeError, ValueError) as error:
            raise RootCauseEvaluationError(
                "full official-like metric binding differs"
            ) from error
        if not math.isclose(spatial, reported_spatial, rel_tol=0.0, abs_tol=1e-12):
            raise RootCauseEvaluationError("full official-like metric binding differs")
        rows.append({"seed": run.get("seed"), **metrics})
        if path.name in sources:
            raise RootCauseEvaluationError("duplicate full evaluation source name")
        sources[path.name] = _file_identity(path)
        for field, values in provenance.items():
            values.add(run.get(field))
    _normalized_rows(rows)
    if any(
        len(values) != 1
        or not _is_lower_hex(next(iter(values)), 40 if field == "source_commit" else 64)
        for field, values in provenance.items()
    ):
        raise RootCauseEvaluationError("full official-like provenance differs")
    return rows, sources


def _baseline_rows(
    path: Path, *, start_state: Mapping[str, Any]
) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader if row.get("model") == "concerto"]
    except OSError as error:
        raise RootCauseEvaluationError(
            "frozen Concerto baseline is unreadable"
        ) from error
    if any(
        row.get("checkpoint_sha256") != FROZEN_CONCERTO_CHECKPOINT_SHA256
        or row.get("validation_sequence_count") != "154"
        for row in rows
    ):
        raise RootCauseEvaluationError("frozen Concerto baseline binding differs")
    result = [{"seed": row["seed"], **_metrics(row)} for row in rows]
    normalized = _normalized_rows(result)
    summary = _summary(normalized)
    frozen = start_state.get("local_metrics", {}).get("concerto_reimplementation")
    if not isinstance(frozen, Mapping) or any(
        not math.isclose(
            float(summary[f"{metric}_mean"]),
            float(frozen[metric]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for metric in ("t_mAP", "overall_mAP", "stage1_mAP", "stage2_mAP")
    ):
        raise RootCauseEvaluationError("frozen Concerto baseline summary differs")
    return result


def build_full_evaluation_outputs(
    *,
    run_paths: Sequence[Path],
    authorization_path: Path,
    checkpoint_manifest_path: Path,
    full_training_manifest_path: Path,
    baseline_path: Path,
    start_state_path: Path,
    study_kind: str = "rootcause",
) -> dict[str, bytes]:
    if study_kind not in {"rootcause", "strong_local"}:
        raise RootCauseEvaluationError("full evaluation study kind is invalid")
    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_hash(authorization, "authorization_sha256", name="variant authorization")
    training = _load_json(full_training_manifest_path, name="full-training manifest")
    _validate_hash(training, "content_sha256", name="full-training manifest")
    checkpoint = _load_json(checkpoint_manifest_path, name="checkpoint manifest")
    _validate_hash(checkpoint, "content_sha256", name="checkpoint manifest")
    variant = validate_checkpoint_manifest_binding(
        checkpoint, authorization=authorization
    )
    if (
        checkpoint.get("stage") != "full_candidate"
        or checkpoint["full_training"]["manifest_sha256"] != training["content_sha256"]
        or training.get("variant") != variant
        or training["selection"]["selected_epoch"]
        != checkpoint["checkpoint"]["selected_epoch"]
        or training["selection"]["selected_checkpoint_sha256"]
        != checkpoint["checkpoint"]["sha256"]
    ):
        raise RootCauseEvaluationError("full evaluation manifest binding differs")
    candidate, run_sources = _candidate_runs(
        run_paths,
        variant=variant,
        completed_epoch=checkpoint["checkpoint"]["selected_epoch"],
        authorization_sha256=authorization["authorization_sha256"],
        checkpoint_manifest_sha256=checkpoint["content_sha256"],
        checkpoint_sha256=checkpoint["checkpoint"]["sha256"],
    )
    start_state = _load_json(start_state_path, name="start state")
    baseline = _baseline_rows(baseline_path, start_state=start_state)
    verdict_prefix = "ROOTCAUSE" if study_kind == "rootcause" else "STRONG-LOCAL"
    model_prefix = (
        "rootcause_full" if study_kind == "rootcause" else "strong_local_full"
    )
    result = classify_full_result(candidate, baseline, verdict_prefix=verdict_prefix)
    result.update(
        {
            "schema_version": 1,
            "status": "pass",
            "variant": variant,
            "selected_completed_epoch": checkpoint["checkpoint"]["selected_epoch"],
            "selected_checkpoint_sha256": checkpoint["checkpoint"]["sha256"],
            "full_training_manifest_sha256": training["content_sha256"],
            "checkpoint_manifest_sha256": checkpoint["content_sha256"],
            "baseline_checkpoint_sha256": FROZEN_CONCERTO_CHECKPOINT_SHA256,
            "experiment": (
                "rescene_task_learning_root_cause_v1"
                if study_kind == "rootcause"
                else "rescene_strong_local_v1"
            ),
        }
    )
    result["content_sha256"] = canonical_sha256(result)

    per_seed_fields = (
        "model",
        "seed",
        *METRIC_NAMES,
        "SpatialStageMean",
        "paired_spatial_delta",
        "checkpoint_sha256",
    )
    by_candidate = _normalized_rows(candidate)
    by_baseline = _normalized_rows(baseline)
    per_seed_rows = []
    for model, rows, checkpoint_sha in (
        (
            "frozen_concerto_reimplementation",
            by_baseline,
            FROZEN_CONCERTO_CHECKPOINT_SHA256,
        ),
        (f"{model_prefix}_{variant}", by_candidate, checkpoint["checkpoint"]["sha256"]),
    ):
        for seed in EVALUATION_SEEDS:
            spatial = (rows[seed]["stage1_mAP"] + rows[seed]["stage2_mAP"]) / 2.0
            baseline_spatial = (
                by_baseline[seed]["stage1_mAP"] + by_baseline[seed]["stage2_mAP"]
            ) / 2.0
            per_seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    **rows[seed],
                    "SpatialStageMean": spatial,
                    "paired_spatial_delta": spatial - baseline_spatial,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )
    summary_fields = (
        "model",
        "seed_count",
        *(
            f"{metric}_{statistic}"
            for metric in (*METRIC_NAMES, "SpatialStageMean")
            for statistic in ("mean", "std")
        ),
        "paired_spatial_delta_mean",
    )
    summary_rows = []
    for model, summary, delta in (
        (
            "frozen_concerto_reimplementation",
            result["baseline_summary"],
            0.0,
        ),
        (
            f"{model_prefix}_{variant}",
            result["candidate_summary"],
            result["paired_spatial_delta_mean"],
        ),
    ):
        summary_rows.append(
            {
                field: model
                if field == "model"
                else delta
                if field == "paired_spatial_delta_mean"
                else summary.get(field, "")
                for field in summary_fields
            }
        )
    provenance: dict[str, object] = {
        "schema_version": 1,
        "run_sources": run_sources,
        "baseline_source": _file_identity(baseline_path),
        "start_state_source": _file_identity(start_state_path),
        "result_content_sha256": result["content_sha256"],
    }
    provenance["content_sha256"] = canonical_sha256(provenance)
    return {
        "official_like_per_seed.csv": _csv_bytes(per_seed_rows, per_seed_fields),
        "official_like_summary.csv": _csv_bytes(summary_rows, summary_fields),
        "ROOT_CAUSE_FULL_VERDICT.json": _json_bytes(result),
        "ROOT_CAUSE_FULL_VERDICT.md": _verdict_markdown(result),
        "FULL_EVALUATION_PROVENANCE.json": _json_bytes(provenance),
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _verdict_markdown(result: Mapping[str, Any]) -> bytes:
    title = (
        "ReScene-Strong Full Verdict"
        if result.get("verdict_prefix") == "STRONG-LOCAL"
        else "Root-Cause Full Verdict"
    )
    lines = [
        f"# {title}",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        f"Paired mean SpatialStageMean delta: `{result['paired_spatial_delta_mean']}`",
        "",
        "## Gates",
        "",
        *[
            f"- `{name}`: `{'PASS' if passed else 'FAIL'}`"
            for name, passed in result["gates"].items()
        ],
        "",
        "The comparison uses the frozen three-seed Concerto reimplementation.",
        "Persist4D metrics were not used.",
        "",
    ]
    return "\n".join(lines).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite full-evaluation output")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--full-training-manifest", type=Path, required=True)
    parser.add_argument("--baseline-per-seed", type=Path, required=True)
    parser.add_argument("--start-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    outputs = build_full_evaluation_outputs(
        run_paths=arguments.runs,
        authorization_path=arguments.authorization,
        checkpoint_manifest_path=arguments.checkpoint_manifest,
        full_training_manifest_path=arguments.full_training_manifest,
        baseline_path=arguments.baseline_per_seed,
        start_state_path=arguments.start_state,
    )
    for name, payload in outputs.items():
        _publish(arguments.output_dir / name, payload)
    verdict = json.loads(outputs["ROOT_CAUSE_FULL_VERDICT.json"])
    print(
        json.dumps(
            {
                "content_sha256": verdict["content_sha256"],
                "status": "pass",
                "verdict": verdict["verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
