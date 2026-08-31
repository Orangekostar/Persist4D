#!/usr/bin/env python3
"""Validate and summarize formal ReScene root-cause learning curves."""

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
from typing import Any

from utils.rescene_rootcause_evaluation import (
    RootCauseEvaluationError,
    validate_candidate_binding,
)
from utils.rescene_rootcause_preflight import canonical_sha256

VALIDATION_EPOCHS = (15, 30, 45, 60, 75, 90)
METRIC_FIELDS = {
    "stage1_mAP": "val_mean_stage1-AP",
    "stage2_mAP": "val_mean_stage2-AP",
    "overall_mAP": "val_mean_AP",
    "t_mAP": "val_mean_t-AP",
    "t_mAP50": "val_mean_t-AP_50",
    "t_mAP25": "val_mean_t-AP_25",
}
CSV_FIELDS = (
    "variant",
    "completed_epoch",
    "optimizer_step",
    "train_log_step",
    *METRIC_FIELDS,
    "SpatialStageMean",
    "candidate_id",
    "config_sha256",
    "variant_authorization_sha256",
    "metrics_csv_sha256",
)


def _load_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _file_identity(path: Path) -> dict[str, object]:
    try:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        if size != path.stat().st_size:
            raise OSError
    except OSError as error:
        raise RootCauseEvaluationError("learning-curve source is unavailable") from error
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_authorization(authorization: Mapping[str, Any]) -> None:
    expected = authorization.get("authorization_sha256")
    unsigned = dict(authorization)
    unsigned.pop("authorization_sha256", None)
    if (
        authorization.get("status") != "authorized"
        or not isinstance(expected, str)
        or canonical_sha256(unsigned) != expected
    ):
        raise RootCauseEvaluationError("variant authorization hash differs")


def _number(row: Mapping[str, str], field: str, *, unit_interval: bool) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("learning-curve metric is invalid") from error
    if (
        not math.isfinite(value)
        or (unit_interval and not 0.0 <= value <= 1.0)
        or (not unit_interval and value < 0.0)
    ):
        raise RootCauseEvaluationError("learning-curve metric is invalid")
    return value


def _metrics_version(path: Path) -> int:
    name = path.parent.name
    if not name.startswith("version_"):
        raise RootCauseEvaluationError("learning-curve logger version is invalid")
    try:
        version = int(name.removeprefix("version_"))
    except ValueError as error:
        raise RootCauseEvaluationError(
            "learning-curve logger version is invalid"
        ) from error
    if version < 0:
        raise RootCauseEvaluationError("learning-curve logger version is invalid")
    return version


def _validation_rows(
    metrics_paths: Sequence[Path],
) -> tuple[
    dict[int, dict[str, str]],
    dict[int, dict[str, object]],
    dict[str, dict[str, object]],
]:
    if not metrics_paths:
        raise RootCauseEvaluationError("learning-curve CSV is unreadable")
    by_epoch: dict[int, dict[str, str]] = {}
    identities_by_epoch: dict[int, dict[str, object]] = {}
    sources: dict[str, dict[str, object]] = {}
    versions: set[int] = set()
    for metrics_path in metrics_paths:
        version = _metrics_version(metrics_path)
        if version in versions:
            raise RootCauseEvaluationError("duplicate learning-curve logger version")
        versions.add(version)
        identity = _file_identity(metrics_path)
        source_name = Path(*metrics_path.parts[-3:]).as_posix()
        sources[source_name] = identity
        try:
            with metrics_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise RootCauseEvaluationError(
                        "learning-curve CSV has no header"
                    )
                required = {"epoch", "step", *METRIC_FIELDS.values()}
                if not required.issubset(reader.fieldnames):
                    raise RootCauseEvaluationError(
                        "learning-curve CSV schema differs"
                    )
                rows = [
                    dict(row)
                    for row in reader
                    if row.get("val_mean_stage1-AP") not in (None, "")
                ]
        except OSError as error:
            raise RootCauseEvaluationError(
                "learning-curve CSV is unreadable"
            ) from error
        for row in rows:
            try:
                completed_epoch = int(row["epoch"]) + 1
            except (TypeError, ValueError) as error:
                raise RootCauseEvaluationError(
                    "validation epoch is invalid"
                ) from error
            if completed_epoch in by_epoch:
                raise RootCauseEvaluationError("duplicate validation checkpoint")
            by_epoch[completed_epoch] = row
            identities_by_epoch[completed_epoch] = identity
    if set(by_epoch) != set(VALIDATION_EPOCHS):
        raise RootCauseEvaluationError("standard validation checkpoints differ")
    return by_epoch, identities_by_epoch, sources


def summarize_learning_curves(
    *,
    run_directories: Mapping[str, Path],
    authorization_path: Path,
) -> dict[str, object]:
    """Build exact standard-validation rows and R0-relative lead decisions."""

    authorization = _load_object(authorization_path, name="variant authorization")
    _validate_authorization(authorization)
    selected = authorization.get("selected_variants")
    variants = tuple(run_directories)
    if (
        not variants
        or "R0" not in variants
        or not isinstance(selected, list)
        or any(variant not in selected for variant in variants)
    ):
        raise RootCauseEvaluationError("learning-curve variants are not authorized")
    output_rows: list[dict[str, object]] = []
    sources: dict[str, dict[str, object]] = {}
    spatial_by_variant: dict[str, dict[int, float]] = {}
    for variant, run_directory in run_directories.items():
        candidate = _load_object(
            run_directory / ".rootcause_candidate.json", name="candidate record"
        )
        validate_candidate_binding(
            variant=variant, authorization=authorization, candidate=candidate
        )
        metrics_paths = sorted(
            run_directory.glob("local_metrics/version_*/metrics.csv"),
            key=_metrics_version,
        )
        validation_rows, identities_by_epoch, metric_sources = _validation_rows(
            metrics_paths
        )
        spatial_by_variant[variant] = {}
        for completed_epoch in VALIDATION_EPOCHS:
            source = validation_rows[completed_epoch]
            try:
                train_log_step = int(source["step"])
            except (TypeError, ValueError) as error:
                raise RootCauseEvaluationError("validation step is invalid") from error
            expected_log_step = completed_epoch * 66 - 1
            if train_log_step != expected_log_step:
                raise RootCauseEvaluationError("validation step differs")
            metrics = {
                output_name: _number(source, input_name, unit_interval=True)
                for output_name, input_name in METRIC_FIELDS.items()
            }
            spatial = (metrics["stage1_mAP"] + metrics["stage2_mAP"]) / 2.0
            spatial_by_variant[variant][completed_epoch] = spatial
            output_rows.append(
                {
                    "variant": variant,
                    "completed_epoch": completed_epoch,
                    "optimizer_step": completed_epoch * 66,
                    "train_log_step": train_log_step,
                    **metrics,
                    "SpatialStageMean": spatial,
                    "candidate_id": candidate["candidate_id"],
                    "config_sha256": candidate["config_sha256"],
                    "variant_authorization_sha256": candidate[
                        "variant_authorization_sha256"
                    ],
                    "metrics_csv_sha256": identities_by_epoch[completed_epoch][
                        "sha256"
                    ],
                }
            )
        aggregate_sha256 = (
            next(iter(metric_sources.values()))["sha256"]
            if len(metric_sources) == 1
            else canonical_sha256(metric_sources)
        )
        sources[variant] = {
            "candidate_id": candidate["candidate_id"],
            "metrics_csv_bytes": sum(
                int(identity["bytes"]) for identity in metric_sources.values()
            ),
            "metrics_csv_sha256": aggregate_sha256,
            "metrics_csv_sources": metric_sources,
        }
    control = spatial_by_variant["R0"]
    validation_leads = {
        variant: {
            epoch: spatial_by_variant[variant][epoch] > control[epoch]
            for epoch in (75, 90)
        }
        for variant in variants
        if variant != "R0"
    }
    return {
        "rows": output_rows,
        "sources": sources,
        "validation_leads": validation_leads,
        "authorization_sha256": authorization["authorization_sha256"],
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("ascii")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite learning-curve output")
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


def _run_argument(value: str) -> tuple[str, Path]:
    variant, separator, path = value.partition("=")
    if not separator or not variant or not path:
        raise argparse.ArgumentTypeError("run must use VARIANT=PATH")
    return variant, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_argument, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_directories = dict(arguments.run)
    if len(run_directories) != len(arguments.run):
        raise RootCauseEvaluationError("run variants must be unique")
    result = summarize_learning_curves(
        run_directories=run_directories,
        authorization_path=arguments.authorization,
    )
    _publish(arguments.output, _csv_bytes(result["rows"]))
    provenance = {key: value for key, value in result.items() if key != "rows"}
    _publish(arguments.provenance_output, _json_bytes(provenance))
    print(json.dumps(provenance, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
