#!/usr/bin/env python3
"""Finalize one ReScene-Strong short curve and its full-run gate."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rescene_rootcause_checkpoint import _stable_file_identity
from scripts.finalize_rescene_rootcause_training import PER_SEED_FIELDS
from scripts.summarize_rescene_rootcause_curves import (
    CSV_FIELDS as LEARNING_CURVE_FIELDS,
)
from scripts.summarize_rescene_rootcause_curves import (
    METRIC_FIELDS,
    VALIDATION_EPOCHS,
    _metrics_version,
    _number,
    _validation_rows,
)
from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    METRIC_NAMES,
    RootCauseEvaluationError,
    validate_candidate_binding,
    validate_checkpoint_manifest_binding,
)
from utils.rescene_rootcause_preflight import canonical_sha256
from utils.rescene_strong_local import decide_strong_result

SHORT_EPOCHS = (60, 90)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RootCauseEvaluationError(f"{name} is not readable JSON") from error
    if not isinstance(value, dict):
        raise RootCauseEvaluationError(f"{name} must contain an object")
    return value


def _validate_signed(payload: Mapping[str, Any], *, field: str, name: str) -> None:
    expected = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError(f"{name} content hash differs")


def _bound_identity(
    authorization: Mapping[str, Any], *, name: str, path: Path
) -> dict[str, object]:
    upstream = authorization.get("upstream_evidence")
    expected = upstream.get(name) if isinstance(upstream, Mapping) else None
    observed = _stable_file_identity(path)
    if observed != expected:
        raise RootCauseEvaluationError(f"{name} evidence identity differs")
    return observed


def _read_csv(path: Path, *, fields: Sequence[str]) -> list[dict[str, str]]:
    try:
        with path.open(encoding="ascii", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise RootCauseEvaluationError("strong-local CSV schema differs")
            return [dict(row) for row in reader]
    except OSError as error:
        raise RootCauseEvaluationError("strong-local CSV is unreadable") from error


def _root_curve_rows(
    *,
    path: Path,
    base_variant: str,
    authorization: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[int, float]]:
    _bound_identity(authorization, name="root_learning_curves", path=path)
    rows = _read_csv(path, fields=LEARNING_CURVE_FIELDS)
    selected = [row for row in rows if row["variant"] == base_variant]
    try:
        by_epoch = {int(row["completed_epoch"]): row for row in selected}
    except (TypeError, ValueError) as error:
        raise RootCauseEvaluationError("root-cause learning curve differs") from error
    if len(selected) != len(VALIDATION_EPOCHS) or set(by_epoch) != set(
        VALIDATION_EPOCHS
    ):
        raise RootCauseEvaluationError("root-cause learning curve differs")
    spatial = {}
    for epoch, row in by_epoch.items():
        try:
            value = float(row["SpatialStageMean"])
        except (TypeError, ValueError) as error:
            raise RootCauseEvaluationError(
                "root-cause learning curve differs"
            ) from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RootCauseEvaluationError("root-cause learning curve differs")
        spatial[epoch] = value
    return [by_epoch[epoch] for epoch in VALIDATION_EPOCHS], spatial


def _strong_curve_rows(
    *, run_directory: Path, authorization: Mapping[str, Any], variant: str
) -> tuple[list[dict[str, object]], dict[int, float], dict[str, object]]:
    candidate_path = run_directory / ".rootcause_candidate.json"
    candidate = _load_json(candidate_path, name="strong-local candidate")
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
    rows = []
    spatial = {}
    for completed_epoch in VALIDATION_EPOCHS:
        source = validation_rows[completed_epoch]
        try:
            train_log_step = int(source["step"])
        except (TypeError, ValueError) as error:
            raise RootCauseEvaluationError(
                "strong-local validation step differs"
            ) from error
        if train_log_step != completed_epoch * 66 - 1:
            raise RootCauseEvaluationError("strong-local validation step differs")
        metrics = {
            output_name: _number(source, input_name, unit_interval=True)
            for output_name, input_name in METRIC_FIELDS.items()
        }
        spatial[completed_epoch] = (metrics["stage1_mAP"] + metrics["stage2_mAP"]) / 2.0
        rows.append(
            {
                "variant": variant,
                "completed_epoch": completed_epoch,
                "optimizer_step": completed_epoch * 66,
                "train_log_step": train_log_step,
                **metrics,
                "SpatialStageMean": spatial[completed_epoch],
                "val_loss": _number(source, "val_loss", unit_interval=False),
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
    return (
        rows,
        spatial,
        {
            "candidate": _stable_file_identity(candidate_path),
            "metrics": metric_sources,
        },
    )


def _metric_row(row: Mapping[str, str]) -> dict[str, float]:
    try:
        metrics = {name: float(row[name]) for name in METRIC_NAMES}
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("official-like metric row is invalid") from error
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in metrics.values()
    ):
        raise RootCauseEvaluationError("official-like metric row is invalid")
    return metrics


def _root_runs(
    *,
    path: Path,
    epoch: int,
    base_variant: str,
    authorization: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> tuple[dict[int, dict[str, float]], list[dict[str, str]]]:
    name = f"root_official_like_epoch{epoch}"
    _bound_identity(authorization, name=name, path=path)
    rows = _read_csv(path, fields=PER_SEED_FIELDS)
    selected = [
        row
        for row in rows
        if row["variant"] == base_variant and int(row["completed_epoch"]) == epoch
    ]
    try:
        by_seed = {int(row["seed"]): _metric_row(row) for row in selected}
    except (TypeError, ValueError) as error:
        raise RootCauseEvaluationError("root official-like seed differs") from error
    if len(selected) != len(EVALUATION_SEEDS) or set(by_seed) != set(EVALUATION_SEEDS):
        raise RootCauseEvaluationError("root official-like seed matrix differs")
    comparison = decide_strong_result(
        variant=authorization["selected_variants"][0],
        base_runs=by_seed,
        variant_runs=by_seed,
        validation_leads={75: True, 90: True},
        contract_integrity=True,
    )["base_summary"]
    expected = decision[f"epoch{epoch}_summary"][base_variant]
    for field, value in comparison.items():
        if field == "seed_count":
            valid = value == expected.get(field)
        else:
            valid = math.isclose(
                float(value), float(expected.get(field)), rel_tol=0.0, abs_tol=1e-12
            )
        if not valid:
            raise RootCauseEvaluationError("root official-like summary differs")
    return by_seed, selected


def _strong_runs(
    *,
    evaluation_root: Path,
    epoch: int,
    variant: str,
    authorization: Mapping[str, Any],
) -> tuple[dict[int, dict[str, float]], list[dict[str, object]], dict[str, object]]:
    directory = evaluation_root / variant / f"epoch{epoch:03d}"
    manifest_path = directory / "checkpoint_manifest.json"
    manifest = _load_json(manifest_path, name="strong-local checkpoint manifest")
    _validate_signed(
        manifest, field="content_sha256", name="strong-local checkpoint manifest"
    )
    observed_variant = validate_checkpoint_manifest_binding(
        manifest, authorization=authorization
    )
    if observed_variant != variant or manifest["checkpoint"]["selected_epoch"] != epoch:
        raise RootCauseEvaluationError("strong-local checkpoint binding differs")
    by_seed = {}
    output_rows = []
    sources: dict[str, object] = {
        "checkpoint_manifest": _stable_file_identity(manifest_path)
    }
    for seed in EVALUATION_SEEDS:
        path = directory / f"seed{seed}.json"
        run = _load_json(path, name="strong-local official-like run")
        if (
            run.get("status") != "pass"
            or run.get("scope") != "official_like_t2"
            or run.get("variant") != variant
            or run.get("completed_epoch") != epoch
            or run.get("seed") != seed
            or run.get("validation_sequence_count") != 154
            or run.get("variant_authorization_sha256")
            != authorization["authorization_sha256"]
            or run.get("checkpoint_manifest_sha256") != manifest["content_sha256"]
            or run.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]
        ):
            raise RootCauseEvaluationError("strong-local official-like binding differs")
        metrics = _metric_row(run["metrics"])
        spatial = (metrics["stage1_mAP"] + metrics["stage2_mAP"]) / 2.0
        if not math.isclose(
            spatial, float(run.get("SpatialStageMean")), rel_tol=0.0, abs_tol=1e-12
        ):
            raise RootCauseEvaluationError("strong-local spatial metric differs")
        by_seed[seed] = metrics
        output_rows.append(
            {
                "variant": variant,
                "completed_epoch": epoch,
                "seed": seed,
                **metrics,
                "SpatialStageMean": spatial,
                "validation_sequence_count": 154,
                "checkpoint_sha256": run["checkpoint_sha256"],
                "elapsed_seconds": run["elapsed_seconds"],
            }
        )
        sources[f"seed{seed}"] = _stable_file_identity(path)
    return by_seed, output_rows, sources


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


def build_strong_outputs(
    *,
    authorization_path: Path,
    root_decision_path: Path,
    root_learning_curves_path: Path,
    root_epoch60_path: Path,
    root_epoch90_path: Path,
    run_directory: Path,
    evaluation_root: Path,
) -> dict[str, bytes]:
    authorization = _load_json(authorization_path, name="strong-local authorization")
    _validate_signed(
        authorization,
        field="authorization_sha256",
        name="strong-local authorization",
    )
    if authorization.get("status") != "authorized":
        raise RootCauseEvaluationError("strong-local authorization is inactive")
    _bound_identity(authorization, name="short_decision", path=root_decision_path)
    root_decision = _load_json(root_decision_path, name="root-cause short decision")
    _validate_signed(
        root_decision, field="content_sha256", name="root-cause short decision"
    )
    variant = authorization["selected_variants"][0]
    base_variant = authorization["base_variant"]
    root_curve_rows, root_spatial = _root_curve_rows(
        path=root_learning_curves_path,
        base_variant=base_variant,
        authorization=authorization,
    )
    strong_curve_rows, strong_spatial, curve_sources = _strong_curve_rows(
        run_directory=run_directory,
        authorization=authorization,
        variant=variant,
    )
    validation_leads = {
        epoch: strong_spatial[epoch] > root_spatial[epoch] for epoch in (75, 90)
    }
    per_seed_rows = []
    comparisons = {}
    evaluation_sources = {}
    for epoch, root_path in (
        (60, root_epoch60_path),
        (90, root_epoch90_path),
    ):
        base_runs, root_rows = _root_runs(
            path=root_path,
            epoch=epoch,
            base_variant=base_variant,
            authorization=authorization,
            decision=root_decision,
        )
        strong_runs, strong_rows, sources = _strong_runs(
            evaluation_root=evaluation_root,
            epoch=epoch,
            variant=variant,
            authorization=authorization,
        )
        comparison = decide_strong_result(
            variant=variant,
            base_runs=base_runs,
            variant_runs=strong_runs,
            validation_leads=validation_leads,
            contract_integrity=True,
        )
        comparisons[epoch] = comparison
        per_seed_rows.extend(root_rows)
        per_seed_rows.extend(strong_rows)
        evaluation_sources[f"epoch{epoch}"] = sources
    decision = {
        **comparisons[90],
        "experiment": "rescene_strong_local_v1",
        "base_variant": base_variant,
        "selected_variant": (
            variant if comparisons[90]["full_training_authorized"] else None
        ),
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "validation_leads": validation_leads,
        "epoch60_comparison": comparisons[60],
        "epoch90_comparison": comparisons[90],
    }
    decision["content_sha256"] = canonical_sha256(decision)
    provenance = {
        "schema_version": 1,
        "experiment": "rescene_strong_local_v1",
        "authorization": _stable_file_identity(authorization_path),
        "root_decision": _stable_file_identity(root_decision_path),
        "root_learning_curves": _stable_file_identity(root_learning_curves_path),
        "root_epoch60": _stable_file_identity(root_epoch60_path),
        "root_epoch90": _stable_file_identity(root_epoch90_path),
        "strong_curve_sources": curve_sources,
        "strong_evaluation_sources": evaluation_sources,
        "decision_content_sha256": decision["content_sha256"],
    }
    provenance["content_sha256"] = canonical_sha256(provenance)
    report = (
        "# ReScene-Strong Verdict\n\n"
        f"Variant: `{variant}`\n\n"
        f"Base root-cause semantics: `{base_variant}`\n\n"
        f"Status: `{decision['full_training_status']}`\n\n"
        f"Mean paired spatial delta: `{decision['paired_spatial_delta_mean']}`\n\n"
        "Persist4D metrics were not used.\n"
    ).encode("ascii")
    return {
        "learning_curves.csv": _csv_bytes(
            [*root_curve_rows, *strong_curve_rows], LEARNING_CURVE_FIELDS
        ),
        "official_like_per_seed.csv": _csv_bytes(per_seed_rows, PER_SEED_FIELDS),
        "STRONG_LOCAL_VERDICT.json": _json_bytes(decision),
        "STRONG_LOCAL_VERDICT.md": report,
        "STRONG_LOCAL_PROVENANCE.json": _json_bytes(provenance),
    }


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite strong-local output")
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
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--root-decision", type=Path, required=True)
    parser.add_argument("--root-learning-curves", type=Path, required=True)
    parser.add_argument("--root-epoch60", type=Path, required=True)
    parser.add_argument("--root-epoch90", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    outputs = build_strong_outputs(
        authorization_path=arguments.authorization,
        root_decision_path=arguments.root_decision,
        root_learning_curves_path=arguments.root_learning_curves,
        root_epoch60_path=arguments.root_epoch60,
        root_epoch90_path=arguments.root_epoch90,
        run_directory=arguments.run_dir,
        evaluation_root=arguments.evaluation_root,
    )
    for name, payload in outputs.items():
        _publish(arguments.output_dir / name, payload)
    decision = json.loads(outputs["STRONG_LOCAL_VERDICT.json"])
    print(
        json.dumps(
            {
                "content_sha256": decision["content_sha256"],
                "status": decision["full_training_status"],
                "variant": decision["variant"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
