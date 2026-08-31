#!/usr/bin/env python3
"""Finalize matched ReScene root-cause short curves and the RC4 gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.summarize_rescene_rootcause_curves import (
    CSV_FIELDS as LEARNING_CURVE_FIELDS,
)
from scripts.summarize_rescene_rootcause_curves import (
    summarize_learning_curves,
)
from utils.rescene_rootcause_evaluation import (
    EVALUATION_SEEDS,
    METRIC_NAMES,
    RootCauseEvaluationError,
    decide_full_candidate,
    summarize_epoch_runs,
    validate_candidate_binding,
    validate_checkpoint_manifest_binding,
)
from utils.rescene_rootcause_preflight import canonical_sha256

SHORT_EPOCHS = (60, 90)
PER_SEED_FIELDS = (
    "variant",
    "completed_epoch",
    "seed",
    *METRIC_NAMES,
    "SpatialStageMean",
    "validation_sequence_count",
    "checkpoint_sha256",
    "elapsed_seconds",
)
ROOTCAUSE_SUMMARY_FIELDS = (
    "completed_epoch",
    "variant",
    "seed_count",
    *(
        f"{metric}_{statistic}"
        for metric in (*METRIC_NAMES, "SpatialStageMean")
        for statistic in ("mean", "std")
    ),
    "paired_spatial_delta_mean",
    "paired_spatial_positive_seed_count",
)


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
        raise RootCauseEvaluationError("short-curve source is unavailable") from error
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
        raise RootCauseEvaluationError("short-curve source changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _validate_content_hash(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not isinstance(expected, str) or canonical_sha256(unsigned) != expected:
        raise RootCauseEvaluationError(f"{name} content hash differs")


def _validate_authorization(payload: Mapping[str, Any]) -> None:
    expected = payload.get("authorization_sha256")
    unsigned = dict(payload)
    unsigned.pop("authorization_sha256", None)
    if (
        payload.get("status") != "authorized"
        or not isinstance(expected, str)
        or canonical_sha256(unsigned) != expected
    ):
        raise RootCauseEvaluationError("variant authorization hash differs")


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


def _per_seed_row(run: Mapping[str, Any]) -> dict[str, object]:
    metrics = run["metrics"]
    return {
        "variant": run["variant"],
        "completed_epoch": run["completed_epoch"],
        "seed": run["seed"],
        **metrics,
        "SpatialStageMean": (
            float(metrics["stage1_mAP"]) + float(metrics["stage2_mAP"])
        )
        / 2.0,
        "validation_sequence_count": run["validation_sequence_count"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "elapsed_seconds": run["elapsed_seconds"],
    }


def _summary_rows(
    summaries: Mapping[int, Mapping[str, Mapping[str, object]]],
    variants: Sequence[str],
) -> list[dict[str, object]]:
    rows = []
    for epoch in SHORT_EPOCHS:
        for variant in variants:
            record = summaries[epoch][variant]
            rows.append(
                {
                    field: (
                        epoch
                        if field == "completed_epoch"
                        else variant
                        if field == "variant"
                        else record.get(field, "")
                    )
                    for field in ROOTCAUSE_SUMMARY_FIELDS
                }
            )
    return rows


def _decision_markdown(decision: Mapping[str, Any]) -> bytes:
    selected = decision["selected_variant"]
    lines = [
        "# ReScene Root-Cause Short-Curve Decision",
        "",
        f"Status: `{decision['full_training_status']}`",
        f"Selected full candidate: `{selected if selected is not None else 'none'}`",
        "",
        "Primary metric: epoch-90 mean `SpatialStageMean`.",
        "Persist4D metrics were not used.",
        "",
        "## Gates",
        "",
    ]
    for variant, record in decision["decisions"].items():
        lines.append(f"### {variant}")
        lines.append("")
        for gate, passed in record["gates"].items():
            lines.append(f"- `{gate}`: `{'PASS' if passed else 'FAIL'}`")
        lines.extend(
            [
                "",
                f"All gates: `{'PASS' if record['all_gates_pass'] else 'FAIL'}`",
                "",
            ]
        )
    if decision["infeasible_variants"]:
        lines.extend(["## Runtime-Infeasible Variants", ""])
        for variant, record in decision["infeasible_variants"].items():
            lines.append(f"- `{variant}`: `{record['reason_code']}`")
        lines.append("")
    return ("\n".join(lines).rstrip() + "\n").encode("ascii")


def _validate_infeasible_variants(
    *,
    infeasible_variants: Mapping[str, Path],
    authorization: Mapping[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, object]]]:
    records: dict[str, dict[str, str]] = {}
    sources: dict[str, dict[str, object]] = {}
    variants = authorization["variants"]
    initialization = authorization["initialization"]
    for variant, path in infeasible_variants.items():
        payload = _load_json(path, name="runtime infeasibility record")
        expected = {
            "status": "infeasible",
            "variant": variant,
            "reason_code": "full_dataset_cuda_oom",
            "variant_authorization_sha256": authorization["authorization_sha256"],
            "config_sha256": variants[variant]["config_sha256"],
            "common_initialization_sha256": initialization["common_state"]["sha256"],
            "replacement_variant": None,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RootCauseEvaluationError("runtime infeasibility binding differs")
        candidate_id = payload.get("candidate_id")
        failed_batch_index = payload.get("failed_batch_index")
        attempts = payload.get("attempts")
        if (
            not isinstance(candidate_id, str)
            or len(candidate_id) != 64
            or not isinstance(failed_batch_index, int)
            or failed_batch_index < 0
            or not isinstance(attempts, list)
            or len(attempts) < 2
        ):
            raise RootCauseEvaluationError("runtime infeasibility record is incomplete")
        sources[f"{variant}/INFEASIBLE.json"] = _file_identity(path)
        evidence_root = path.resolve().parent.parent
        allocators: set[str] = set()
        for index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, Mapping):
                raise RootCauseEvaluationError("runtime infeasibility attempt is invalid")
            allocator = attempt.get("allocator")
            relative = attempt.get("evidence")
            if (
                not isinstance(allocator, str)
                or allocator in allocators
                or not isinstance(relative, str)
            ):
                raise RootCauseEvaluationError("runtime infeasibility attempt is invalid")
            allocators.add(allocator)
            evidence = (evidence_root / relative).resolve()
            try:
                evidence.relative_to(evidence_root)
            except ValueError as error:
                raise RootCauseEvaluationError(
                    "runtime infeasibility evidence escapes its root"
                ) from error
            failure = _load_json(evidence, name="runtime failure evidence")
            candidate_path = evidence.parent / ".rootcause_candidate.json"
            candidate = _load_json(candidate_path, name="failed candidate record")
            validate_candidate_binding(
                variant=variant, authorization=authorization, candidate=candidate
            )
            launch_log = evidence.parent / f"{variant}.launch.log"
            launch_identity = _file_identity(launch_log)
            if (
                failure.get("status") != "failed"
                or failure.get("variant") != variant
                or failure.get("candidate_id") != candidate_id
                or failure.get("candidate_id") != candidate.get("candidate_id")
                or failure.get("config_sha256") != expected["config_sha256"]
                or failure.get("common_initialization_sha256")
                != expected["common_initialization_sha256"]
                or failure.get("failed_batch_index") != failed_batch_index
                or failure.get("launch_log_sha256") != launch_identity["sha256"]
                or attempt.get("launch_log_sha256") != launch_identity["sha256"]
            ):
                raise RootCauseEvaluationError(
                    "runtime infeasibility evidence binding differs"
                )
            prefix = f"{variant}/attempt{index}"
            sources[f"{prefix}/FAILURE.json"] = _file_identity(evidence)
            sources[f"{prefix}/.rootcause_candidate.json"] = _file_identity(
                candidate_path
            )
            sources[f"{prefix}/{variant}.launch.log"] = launch_identity
        records[variant] = {
            "candidate_id": candidate_id,
            "reason_code": expected["reason_code"],
        }
    return records, sources


def _evaluation_matrix(
    *,
    evaluation_root: Path,
    variants: Sequence[str],
    authorization: Mapping[str, Any],
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, dict[str, dict[str, object]]],
    dict[str, dict[str, object]],
]:
    runs_by_epoch: dict[int, list[dict[str, Any]]] = {}
    summaries: dict[int, dict[str, dict[str, object]]] = {}
    sources: dict[str, dict[str, object]] = {}
    for epoch in SHORT_EPOCHS:
        epoch_runs = []
        for variant in variants:
            directory = evaluation_root / variant / f"epoch{epoch:03d}"
            manifest_path = directory / "checkpoint_manifest.json"
            manifest = _load_json(manifest_path, name="checkpoint manifest")
            _validate_content_hash(manifest, name="checkpoint manifest")
            observed_variant = validate_checkpoint_manifest_binding(
                manifest, authorization=authorization
            )
            if (
                observed_variant != variant
                or manifest["checkpoint"]["selected_epoch"] != epoch
            ):
                raise RootCauseEvaluationError("checkpoint manifest binding differs")
            sources[f"{variant}/epoch{epoch:03d}/checkpoint_manifest.json"] = (
                _file_identity(manifest_path)
            )
            for seed in EVALUATION_SEEDS:
                run_path = directory / f"seed{seed}.json"
                run = _load_json(run_path, name="official-like run")
                if (
                    run.get("variant") != variant
                    or run.get("completed_epoch") != epoch
                    or run.get("seed") != seed
                    or run.get("variant_authorization_sha256")
                    != authorization["authorization_sha256"]
                    or run.get("checkpoint_manifest_sha256")
                    != manifest["content_sha256"]
                    or run.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]
                ):
                    raise RootCauseEvaluationError("official-like run binding differs")
                sources[f"{variant}/epoch{epoch:03d}/seed{seed}.json"] = _file_identity(
                    run_path
                )
                epoch_runs.append(run)
        summaries[epoch] = summarize_epoch_runs(
            epoch_runs, variants=variants, completed_epoch=epoch
        )
        runs_by_epoch[epoch] = epoch_runs
    return runs_by_epoch, summaries, sources


def build_short_curve_outputs(
    *,
    run_directories: Mapping[str, Path],
    infeasible_variants: Mapping[str, Path] | None = None,
    authorization_path: Path,
    evaluation_root: Path,
) -> dict[str, bytes]:
    """Validate all RC3 evidence and render immutable compact outputs."""

    variants = tuple(run_directories)
    infeasible_variants = infeasible_variants or {}
    if not variants or variants[0] != "R0" or len(variants) != len(set(variants)):
        raise RootCauseEvaluationError("variants must be unique and start with R0")
    authorization = _load_json(authorization_path, name="variant authorization")
    _validate_authorization(authorization)
    authorized_variants = authorization.get("selected_variants")
    if not isinstance(authorized_variants, list):
        raise RootCauseEvaluationError("variant authorization is invalid")
    resolved_variants = set(variants) | set(infeasible_variants)
    if (
        set(variants) & set(infeasible_variants)
        or resolved_variants != set(authorized_variants)
        or [variant for variant in authorized_variants if variant in run_directories]
        != list(variants)
        or "R0" in infeasible_variants
    ):
        raise RootCauseEvaluationError("finalized variants differ from authorization")

    infeasible_records, infeasible_sources = _validate_infeasible_variants(
        infeasible_variants=infeasible_variants,
        authorization=authorization,
    )

    curves = summarize_learning_curves(
        run_directories=run_directories,
        authorization_path=authorization_path,
    )
    runs_by_epoch, summaries, evaluation_sources = _evaluation_matrix(
        evaluation_root=evaluation_root,
        variants=variants,
        authorization=authorization,
    )
    contract_integrity = {variant: True for variant in variants}
    gate = decide_full_candidate(
        summaries[90],
        validation_leads=curves["validation_leads"],
        contract_integrity=contract_integrity,
    )
    selected = gate["selected_variant"]
    decision: dict[str, object] = {
        "schema_version": 1,
        **gate,
        "experiment": "rescene_task_learning_root_cause_v1",
        "completed_epoch": 90,
        "full_training_authorized": selected is not None,
        "full_training_status": "authorized"
        if selected is not None
        else "gate_skipped",
        "validation_leads": curves["validation_leads"],
        "completed_variants": list(variants),
        "infeasible_variants": infeasible_records,
        "contract_integrity": contract_integrity,
        "epoch60_summary": summaries[60],
        "epoch90_summary": summaries[90],
        "variant_authorization_sha256": authorization["authorization_sha256"],
    }
    decision["content_sha256"] = canonical_sha256(decision)

    per_seed_by_epoch = {
        epoch: [
            _per_seed_row(run)
            for run in sorted(
                runs_by_epoch[epoch], key=lambda row: (row["variant"], row["seed"])
            )
        ]
        for epoch in SHORT_EPOCHS
    }
    combined_per_seed = [
        row for epoch in SHORT_EPOCHS for row in per_seed_by_epoch[epoch]
    ]
    provenance: dict[str, object] = {
        "schema_version": 1,
        "experiment": "rescene_task_learning_root_cause_v1",
        "variant_authorization": _file_identity(authorization_path),
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "learning_curve_sources": curves["sources"],
        "evaluation_sources": evaluation_sources,
        "infeasible_sources": infeasible_sources,
        "decision_content_sha256": decision["content_sha256"],
    }
    provenance["content_sha256"] = canonical_sha256(provenance)

    return {
        "learning_curves.csv": _csv_bytes(curves["rows"], LEARNING_CURVE_FIELDS),
        "official_like_epoch60.csv": _csv_bytes(per_seed_by_epoch[60], PER_SEED_FIELDS),
        "official_like_epoch90.csv": _csv_bytes(per_seed_by_epoch[90], PER_SEED_FIELDS),
        "rootcause_per_seed.csv": _csv_bytes(combined_per_seed, PER_SEED_FIELDS),
        "rootcause_summary.csv": _csv_bytes(
            _summary_rows(summaries, variants), ROOTCAUSE_SUMMARY_FIELDS
        ),
        "ROOTCAUSE_SHORT_DECISION.json": _json_bytes(decision),
        "ROOTCAUSE_SHORT_DECISION.md": _decision_markdown(decision),
        "ROOTCAUSE_SHORT_PROVENANCE.json": _json_bytes(provenance),
    }


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite short-curve output")
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


def _run_argument(value: str) -> tuple[str, Path]:
    variant, separator, path = value.partition("=")
    if not separator or not variant or not path:
        raise argparse.ArgumentTypeError("run must use VARIANT=PATH")
    return variant, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_run_argument, required=True)
    parser.add_argument("--infeasible", action="append", type=_run_argument, default=[])
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_directories = dict(arguments.run)
    infeasible_variants = dict(arguments.infeasible)
    if len(run_directories) != len(arguments.run):
        raise RootCauseEvaluationError("run variants must be unique")
    if len(infeasible_variants) != len(arguments.infeasible):
        raise RootCauseEvaluationError("infeasible variants must be unique")
    outputs = build_short_curve_outputs(
        run_directories=run_directories,
        infeasible_variants=infeasible_variants,
        authorization_path=arguments.authorization,
        evaluation_root=arguments.evaluation_root,
    )
    for name, payload in outputs.items():
        _publish(arguments.output_dir / name, payload)
    decision = json.loads(outputs["ROOTCAUSE_SHORT_DECISION.json"])
    print(
        json.dumps(
            {
                "content_sha256": decision["content_sha256"],
                "selected_variant": decision["selected_variant"],
                "status": decision["full_training_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
