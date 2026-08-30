#!/usr/bin/env python3
"""Finalize one authorized 450-epoch ReScene root-cause candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_evaluation import (
    METRIC_NAMES,
    RootCauseEvaluationError,
    build_full_checkpoint_manifest,
    validate_candidate_binding,
    validate_checkpoint_payload,
    validate_full_checkpoint_payload,
)
from utils.rescene_rootcause_preflight import canonical_sha256

VALIDATION_EPOCHS = tuple(range(15, 451, 15))
METRIC_FIELDS = {
    "stage1_mAP": "val_mean_stage1-AP",
    "stage2_mAP": "val_mean_stage2-AP",
    "overall_mAP": "val_mean_AP",
    "t_mAP": "val_mean_t-AP",
    "t_mAP50": "val_mean_t-AP_50",
    "t_mAP25": "val_mean_t-AP_25",
}
LEARNING_CURVE_FIELDS = (
    "completed_epoch",
    "optimizer_step",
    "train_log_step",
    *METRIC_NAMES,
    "SpatialStageMean",
    "val_loss",
    "metrics_csv_sha256",
)
BEST_CHECKPOINT = re.compile(
    r"^epoch=(?P<epoch>\d+)-val_mean_t-AP="
    r"(?P<score>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:-v\d+)?\.ckpt$"
)


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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        raise RootCauseEvaluationError("full-training input is unavailable") from error
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
        raise RootCauseEvaluationError("full-training input changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _number(row: Mapping[str, str], field: str, *, unit_interval: bool) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RootCauseEvaluationError("full validation metric is invalid") from error
    if (
        not math.isfinite(value)
        or (unit_interval and not 0.0 <= value <= 1.0)
        or (not unit_interval and value < 0.0)
    ):
        raise RootCauseEvaluationError("full validation metric is invalid")
    return value


def _source_name(path: Path) -> str:
    parts = path.parts
    if "local_metrics" in parts:
        return Path(*parts[parts.index("local_metrics") :]).as_posix()
    return path.name


def read_full_validation_trajectory(
    metrics_paths: Sequence[Path],
) -> dict[str, object]:
    """Join resumed CSVLogger versions into the exact 30-point trajectory."""

    if not metrics_paths:
        raise RootCauseEvaluationError("full validation metrics are missing")
    by_epoch: dict[int, dict[str, object]] = {}
    sources: dict[str, dict[str, object]] = {}
    for path in metrics_paths:
        identity = _file_identity(path)
        source_name = _source_name(path)
        if source_name in sources:
            raise RootCauseEvaluationError("duplicate validation source name")
        sources[source_name] = identity
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"epoch", "step", "val_loss", *METRIC_FIELDS.values()}
                if reader.fieldnames is None or not required.issubset(
                    reader.fieldnames
                ):
                    raise RootCauseEvaluationError("full validation CSV schema differs")
                rows = [
                    dict(row)
                    for row in reader
                    if row.get("val_mean_stage1-AP") not in (None, "")
                ]
        except OSError as error:
            raise RootCauseEvaluationError(
                "full validation CSV is unreadable"
            ) from error
        for source in rows:
            try:
                completed_epoch = int(source["epoch"]) + 1
                train_log_step = int(source["step"])
            except (TypeError, ValueError) as error:
                raise RootCauseEvaluationError(
                    "full validation boundary is invalid"
                ) from error
            if completed_epoch in by_epoch:
                raise RootCauseEvaluationError("duplicate full validation boundary")
            if completed_epoch not in VALIDATION_EPOCHS:
                raise RootCauseEvaluationError("unexpected full validation boundary")
            if train_log_step != completed_epoch * 66 - 1:
                raise RootCauseEvaluationError("full validation step differs")
            metrics = {
                output: _number(source, input_name, unit_interval=True)
                for output, input_name in METRIC_FIELDS.items()
            }
            by_epoch[completed_epoch] = {
                "completed_epoch": completed_epoch,
                "optimizer_step": completed_epoch * 66,
                "train_log_step": train_log_step,
                **metrics,
                "SpatialStageMean": (metrics["stage1_mAP"] + metrics["stage2_mAP"])
                / 2.0,
                "val_loss": _number(source, "val_loss", unit_interval=False),
                "metrics_csv_sha256": identity["sha256"],
            }
    if set(by_epoch) != set(VALIDATION_EPOCHS):
        raise RootCauseEvaluationError(
            "full run requires exactly 30 validation boundaries"
        )
    return {
        "rows": [by_epoch[epoch] for epoch in VALIDATION_EPOCHS],
        "sources": sources,
    }


def select_full_checkpoint(
    validation_rows: Sequence[Mapping[str, Any]],
    checkpoint_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Require one Top-1 checkpoint and one exact epoch-450 completion state."""

    if len(validation_rows) != 30 or {
        row.get("completed_epoch") for row in validation_rows
    } != set(VALIDATION_EPOCHS):
        raise RootCauseEvaluationError("full validation trajectory differs")
    best = [
        record
        for record in checkpoint_records
        if record.get("role") == "best_validation"
    ]
    full = [
        record
        for record in checkpoint_records
        if record.get("role") == "exact_full_boundary"
    ]
    if len(best) != 1 or len(full) != 1:
        raise RootCauseEvaluationError("full checkpoint inventory differs")
    selected = best[0]
    boundary = full[0]
    best_score = max(float(row["t_mAP"]) for row in validation_rows)
    selected_row = next(
        (
            row
            for row in validation_rows
            if row["completed_epoch"] == selected.get("completed_epoch")
        ),
        None,
    )
    if (
        selected_row is None
        or selected.get("selected_step") != int(selected["completed_epoch"]) * 66
        or not math.isclose(
            float(selected.get("selection_metric_exact", float("nan"))),
            best_score,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        or not math.isclose(
            float(selected_row["t_mAP"]),
            best_score,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise RootCauseEvaluationError(
            "checkpoint does not match the highest validation val_mean_t-AP"
        )
    if (
        boundary.get("completed_epoch") != 450
        or boundary.get("selected_step") != 29_700
    ):
        raise RootCauseEvaluationError("exact full-budget checkpoint differs")
    return {
        "monitor": "val_mean_t-AP",
        "mode": "max",
        "validation_event_count": 30,
        "selected_epoch": selected["completed_epoch"],
        "selected_step": selected["selected_step"],
        "selection_metric_exact": selected["selection_metric_exact"],
        "selected_checkpoint_name": selected.get("filename"),
        "selected_checkpoint_sha256": selected["sha256"],
        "selected_checkpoint_bytes": selected["bytes"],
        "full_budget_checkpoint_sha256": boundary["sha256"],
        "full_budget_checkpoint_bytes": boundary["bytes"],
    }


def build_full_training_manifest(
    *,
    variant: str,
    candidate_id: str,
    authorization_sha256: str,
    config_sha256: str,
    decision: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    selection: Mapping[str, Any],
    validation_sources: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Bind authorization, exact resume, complete budget, and local selection."""

    if (
        not variant
        or any(
            not _is_sha256(value)
            for value in (candidate_id, authorization_sha256, config_sha256)
        )
        or decision.get("selected_variant") != variant
        or decision.get("full_training_authorized") is not True
        or not _is_sha256(decision.get("content_sha256"))
        or resume_plan.get("variant") != variant
        or resume_plan.get("candidate_id") != candidate_id
        or resume_plan.get("runtime_selector_exact_match") is not True
        or resume_plan.get("completed_epoch") != 90
        or resume_plan.get("selected_step") != 5_940
        or not _is_sha256(resume_plan.get("content_sha256"))
    ):
        raise RootCauseEvaluationError("full-training resume binding differs")
    required_selection = {
        "monitor": "val_mean_t-AP",
        "mode": "max",
        "validation_event_count": 30,
    }
    selected_epoch = selection.get("selected_epoch")
    if (
        any(selection.get(key) != value for key, value in required_selection.items())
        or not isinstance(selected_epoch, int)
        or isinstance(selected_epoch, bool)
        or selected_epoch not in VALIDATION_EPOCHS
        or selection.get("selected_step") != selected_epoch * 66
        or not _is_sha256(selection.get("selected_checkpoint_sha256"))
        or not _is_sha256(selection.get("full_budget_checkpoint_sha256"))
        or any(
            not isinstance(selection.get(field), int)
            or isinstance(selection.get(field), bool)
            or selection[field] <= 0
            for field in (
                "selected_checkpoint_bytes",
                "full_budget_checkpoint_bytes",
            )
        )
    ):
        raise RootCauseEvaluationError("full-training selection contract differs")
    if any(
        not isinstance(source, Mapping)
        or not isinstance(source.get("bytes"), int)
        or source["bytes"] <= 0
        or not _is_sha256(source.get("sha256"))
        for source in validation_sources.values()
    ):
        raise RootCauseEvaluationError("full-training validation sources differ")
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment": decision.get("experiment", "rescene_task_learning_root_cause_v1"),
        "stage": "full_candidate",
        "variant": variant,
        "candidate_id": candidate_id,
        "variant_authorization_sha256": authorization_sha256,
        "config_sha256": config_sha256,
        "short_decision_sha256": decision["content_sha256"],
        "resume_plan_sha256": resume_plan["content_sha256"],
        "budget": {
            "resumed_from_completed_epoch": 90,
            "resumed_from_optimizer_step": 5_940,
            "completed_epoch": 450,
            "optimizer_steps": 29_700,
            "validation_event_count": 30,
        },
        "selection": dict(selection),
        "validation_sources": {
            name: dict(source) for name, source in validation_sources.items()
        },
        "selection_used_persist4d": False,
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def _callback_score(payload: Mapping[str, Any]) -> float:
    callbacks = payload.get("callbacks")
    states = (
        [
            state
            for state in callbacks.values()
            if isinstance(state, Mapping)
            and state.get("monitor") == "val_mean_t-AP"
            and "best_model_score" in state
        ]
        if isinstance(callbacks, Mapping)
        else []
    )
    if len(states) != 1:
        raise RootCauseEvaluationError(
            "checkpoint lacks one val_mean_t-AP callback state"
        )
    value = states[0]["best_model_score"]
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RootCauseEvaluationError("checkpoint callback score is invalid")
        value = value.detach().cpu().item()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RootCauseEvaluationError("checkpoint callback score is invalid")
    return float(value)


def inspect_full_checkpoints(
    *,
    run_directory: Path,
    variant: str,
    authorization: Mapping[str, Any],
    expected_state_dict_entries: int = 798,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    from scripts.evaluate_rescene_rootcause_checkpoint import (
        _checkpoint_training_config,
    )

    best_paths = [
        path
        for path in run_directory.glob("*.ckpt")
        if BEST_CHECKPOINT.fullmatch(path.name)
    ]
    full_path = run_directory / "epoch=450.ckpt"
    if len(best_paths) != 1 or not full_path.is_file():
        raise RootCauseEvaluationError("full checkpoint files differ")
    records = []
    facts_by_role = {}
    for role, path in (
        ("best_validation", best_paths[0]),
        ("exact_full_boundary", full_path),
    ):
        identity = _file_identity(path)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RootCauseEvaluationError("full checkpoint is unreadable") from error
        if not isinstance(payload, Mapping):
            raise RootCauseEvaluationError("full checkpoint payload is invalid")
        completed_epoch = int(payload.get("epoch", -2)) + 1
        if role == "best_validation":
            facts = validate_full_checkpoint_payload(
                payload,
                completed_epoch=completed_epoch,
                expected_state_dict_entries=expected_state_dict_entries,
            )
        else:
            facts = validate_checkpoint_payload(
                payload,
                completed_epoch=450,
                expected_state_dict_entries=expected_state_dict_entries,
            )
        portable_config = _checkpoint_training_config(
            payload, variant=variant, authorization=authorization
        )
        facts["training_config_sha256"] = canonical_sha256(portable_config)
        record: dict[str, object] = {
            "filename": path.name,
            "role": role,
            **identity,
            "completed_epoch": facts["selected_epoch"],
            "selected_step": facts["selected_step"],
            "state_dict_entry_count": facts["state_dict_entry_count"],
            "optimizer_state_count": facts["optimizer_state_count"],
            "scheduler_state_count": facts["scheduler_state_count"],
        }
        if role == "best_validation":
            match = BEST_CHECKPOINT.fullmatch(path.name)
            assert match is not None
            exact_score = _callback_score(payload)
            if int(match.group("epoch")) != completed_epoch - 1 or float(
                match.group("score")
            ) != float(f"{exact_score:.3f}"):
                raise RootCauseEvaluationError(
                    "best checkpoint filename differs from callback state"
                )
            record["selection_metric_exact"] = exact_score
            record["selection_metric_filename_rounded"] = float(match.group("score"))
        records.append(record)
        facts_by_role[role] = facts
    return records, facts_by_role


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


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise RootCauseEvaluationError("refusing to overwrite full-training output")
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


def _report(manifest: Mapping[str, Any]) -> bytes:
    selection = manifest["selection"]
    title = (
        "ReScene-Strong Full Training Report"
        if manifest.get("experiment") == "rescene_strong_local_v1"
        else "ReScene Root-Cause Full Training Report"
    )
    return (
        "\n".join(
            [
                f"# {title}",
                "",
                "- Status: `pass`",
                f"- Variant: `{manifest['variant']}`",
                "- Resume boundary: completed epoch `90`, optimizer step `5940`",
                "- Full boundary: completed epoch `450`, optimizer step `29700`",
                "- Selection: highest local `val_mean_t-AP`",
                f"- Selected completed epoch: `{selection['selected_epoch']}`",
                f"- Exact selection metric: `{selection['selection_metric_exact']}`",
                f"- Selected checkpoint SHA256: `{selection['selected_checkpoint_sha256']}`",
                "- Persist4D metrics used for selection: `false`",
                "",
            ]
        )
    ).encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--resume-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)

    authorization = _load_json(arguments.authorization, name="variant authorization")
    _validate_hash(authorization, "authorization_sha256", name="variant authorization")
    candidate = _load_json(
        arguments.run_directory / ".rootcause_candidate.json",
        name="candidate record",
    )
    validate_candidate_binding(
        variant=arguments.variant,
        authorization=authorization,
        candidate=candidate,
    )
    decision = _load_json(arguments.decision, name="short-curve decision")
    _validate_hash(decision, "content_sha256", name="short-curve decision")
    resume_plan = _load_json(arguments.resume_plan, name="full-resume plan")
    _validate_hash(resume_plan, "content_sha256", name="full-resume plan")
    metrics_paths = sorted(
        arguments.run_directory.glob("local_metrics/version_*/metrics.csv")
    )
    trajectory = read_full_validation_trajectory(metrics_paths)
    records, facts = inspect_full_checkpoints(
        run_directory=arguments.run_directory,
        variant=arguments.variant,
        authorization=authorization,
        expected_state_dict_entries=authorization["variants"][arguments.variant].get(
            "expected_state_dict_entries", 798
        ),
    )
    selection = select_full_checkpoint(trajectory["rows"], records)
    manifest = build_full_training_manifest(
        variant=arguments.variant,
        candidate_id=candidate["candidate_id"],
        authorization_sha256=authorization["authorization_sha256"],
        config_sha256=candidate["config_sha256"],
        decision=decision,
        resume_plan=resume_plan,
        selection=selection,
        validation_sources=trajectory["sources"],
    )
    selected_record = next(
        record for record in records if record["role"] == "best_validation"
    )
    selected_manifest = build_full_checkpoint_manifest(
        variant=arguments.variant,
        completed_epoch=int(selection["selected_epoch"]),
        authorization=authorization,
        candidate=candidate,
        file_identity={
            "bytes": selected_record["bytes"],
            "sha256": selected_record["sha256"],
        },
        checkpoint_facts=facts["best_validation"],
        full_training_manifest_sha256=manifest["content_sha256"],
        full_training_completed_epoch=450,
    )
    inventory_fields = (
        "filename",
        "role",
        "sha256",
        "bytes",
        "completed_epoch",
        "selected_step",
        "selection_metric_exact",
        "selection_metric_filename_rounded",
        "state_dict_entry_count",
        "optimizer_state_count",
        "scheduler_state_count",
    )
    inventory_rows = [
        {field: record.get(field, "") for field in inventory_fields}
        for record in records
    ]
    outputs = {
        "learning_curve.csv": _csv_bytes(trajectory["rows"], LEARNING_CURVE_FIELDS),
        "checkpoint_inventory.csv": _csv_bytes(inventory_rows, inventory_fields),
        "FULL_TRAINING_MANIFEST.json": _json_bytes(manifest),
        "selected_checkpoint_manifest.json": _json_bytes(selected_manifest),
        "FULL_TRAINING_REPORT.md": _report(manifest),
    }
    for name, payload in outputs.items():
        _publish(arguments.output_dir / name, payload)
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["content_sha256"],
                "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
                "selected_epoch": selection["selected_epoch"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
