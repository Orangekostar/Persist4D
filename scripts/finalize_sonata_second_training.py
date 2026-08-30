#!/usr/bin/env python3
"""Finalize portable SS4 evidence after the full Sonata run completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_sonata_second_training import (
    CANDIDATE_RECORD_NAME,
    _candidate_contract,
    _load_json,
)
from utils.sonata_second_preflight import (
    build_sonata_source_tree_contract,
    canonical_sha256,
    file_sha256,
)
from utils.sonata_training_evidence import RUNTIME_EVENTS_NAME

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "training"
)
_BEST_CHECKPOINT_RE = re.compile(
    r"^epoch=(?P<epoch>\d+)-val_mean_t-AP="
    r"(?P<score>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:-v\d+)?\.ckpt$"
)
_LAST_CHECKPOINT_RE = re.compile(r"^last(?:-v\d+)?\.ckpt$")


class SonataTrainingFinalizationError(RuntimeError):
    """Raised when an incomplete run is presented as completed SS4 evidence."""


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _read_runtime_events(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
        lines = raw.decode("ascii").splitlines()
        events = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SonataTrainingFinalizationError(
            "runtime events are missing or malformed"
        ) from error
    if not events or any(not isinstance(event, dict) for event in events):
        raise SonataTrainingFinalizationError("runtime events are empty or invalid")
    portable = b"".join(
        (
            json.dumps(
                event,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        for event in events
    )
    forbidden = (b"/home/", b"/mnt/shared/", b"192.168.")
    if any(value in portable for value in forbidden):
        raise SonataTrainingFinalizationError("runtime events contain private paths")
    return events, portable


def _completed_epochs(
    events: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    steps_per_epoch: int,
    samples_per_epoch: int,
) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    for event in events:
        epoch = event.get("completed_epoch")
        if event.get("event") == "epoch_completed" and isinstance(epoch, int):
            by_epoch[epoch] = dict(event)
    if set(by_epoch) != set(range(epochs)):
        raise SonataTrainingFinalizationError(
            f"formal run requires exactly {epochs} completed epochs"
        )
    ordered = [by_epoch[epoch] for epoch in range(epochs)]
    reconstructed_samples = 0
    reconstructed_stages = 0
    for epoch, event in enumerate(ordered):
        expected_steps = (epoch + 1) * steps_per_epoch
        if event.get("optimizer_steps") != expected_steps:
            raise SonataTrainingFinalizationError(
                "optimizer-step trajectory does not match the frozen budget"
            )
        if event.get("samples_seen_epoch") != samples_per_epoch:
            raise SonataTrainingFinalizationError(
                "sample trajectory does not match the frozen epoch sampler"
            )
        stage_count = event.get("stage_observations_epoch")
        if not isinstance(stage_count, int) or stage_count < samples_per_epoch:
            raise SonataTrainingFinalizationError(
                "stage-observation trajectory is incomplete"
            )
        raw_samples_total = event.get("samples_seen_total")
        raw_stages_total = event.get("stage_observations_total")
        if not isinstance(raw_samples_total, int) or not isinstance(
            raw_stages_total, int
        ):
            raise SonataTrainingFinalizationError(
                "raw cumulative observation totals are invalid"
            )
        reconstructed_samples += samples_per_epoch
        reconstructed_stages += stage_count
        event["samples_seen_total_raw"] = raw_samples_total
        event["stage_observations_total_raw"] = raw_stages_total
        event["samples_seen_total_reconstructed"] = reconstructed_samples
        event["stage_observations_total_reconstructed"] = reconstructed_stages
    final_steps = epochs * steps_per_epoch
    if not any(
        event.get("event") == "fit_completed"
        and event.get("completed_epoch") == epochs - 1
        and event.get("optimizer_steps") == final_steps
        for event in events
    ):
        raise SonataTrainingFinalizationError("full-budget fit completion is absent")
    return ordered


def _checkpoint_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SonataTrainingFinalizationError(
            "training checkpoint is not a regular file"
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise SonataTrainingFinalizationError(
            f"training checkpoint is unreadable: {path.name}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise SonataTrainingFinalizationError("training checkpoint is not a mapping")
    if not isinstance(checkpoint.get("state_dict"), Mapping) or not checkpoint[
        "state_dict"
    ]:
        raise SonataTrainingFinalizationError("checkpoint model state is incomplete")
    if not isinstance(checkpoint.get("optimizer_states"), list) or not checkpoint[
        "optimizer_states"
    ]:
        raise SonataTrainingFinalizationError("checkpoint optimizer state is incomplete")
    if not isinstance(checkpoint.get("lr_schedulers"), list) or not checkpoint[
        "lr_schedulers"
    ]:
        raise SonataTrainingFinalizationError("checkpoint scheduler state is incomplete")
    epoch = checkpoint.get("epoch")
    global_step = checkpoint.get("global_step")
    if not isinstance(epoch, int) or not isinstance(global_step, int):
        raise SonataTrainingFinalizationError("checkpoint budget metadata is invalid")
    best_match = _BEST_CHECKPOINT_RE.fullmatch(path.name)
    callbacks = checkpoint.get("callbacks")
    selection_states = (
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
    if len(selection_states) != 1:
        raise SonataTrainingFinalizationError(
            "checkpoint lacks a unique val_mean_t-AP callback state"
        )
    exact_score = selection_states[0]["best_model_score"]
    if isinstance(exact_score, torch.Tensor):
        if exact_score.numel() != 1:
            raise SonataTrainingFinalizationError(
                "checkpoint selection metric is not scalar"
            )
        exact_score = exact_score.detach().cpu().item()
    if (
        isinstance(exact_score, bool)
        or not isinstance(exact_score, (int, float))
        or not math.isfinite(float(exact_score))
    ):
        raise SonataTrainingFinalizationError("checkpoint selection metric is invalid")
    exact_score = float(exact_score)
    filename_score = float(best_match.group("score")) if best_match else None
    if best_match and (
        epoch != int(best_match.group("epoch"))
        or filename_score != float(f"{exact_score:.3f}")
    ):
        raise SonataTrainingFinalizationError(
            "checkpoint filename does not match its exact callback score"
        )
    role = "best_validation" if best_match else "last"
    return {
        "reference": f"external:sonata_training_output/{path.name}",
        "filename": path.name,
        "role": role,
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
        "epoch": epoch,
        "global_step": global_step,
        "selection_metric_name": "val_mean_t-AP",
        "selection_metric_exact": exact_score,
        "selection_metric_filename_rounded": filename_score,
        "state_dict_entry_count": len(checkpoint["state_dict"]),
        "optimizer_state_count": len(checkpoint["optimizer_states"]),
        "scheduler_state_count": len(checkpoint["lr_schedulers"]),
    }


def _checkpoint_inventory(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    paths = sorted(output_dir.glob("*.ckpt"), key=lambda path: path.name)
    recognized = [
        path
        for path in paths
        if _BEST_CHECKPOINT_RE.fullmatch(path.name)
        or _LAST_CHECKPOINT_RE.fullmatch(path.name)
    ]
    if len(recognized) != len(paths) or not recognized:
        raise SonataTrainingFinalizationError("checkpoint inventory is incomplete")
    records = [_checkpoint_record(path) for path in recognized]
    best = [record for record in records if record["role"] == "best_validation"]
    last = [record for record in records if record["role"] == "last"]
    if len(best) != 1 or not last:
        raise SonataTrainingFinalizationError(
            "best-validation and last checkpoint evidence are required"
        )
    last_is_full_budget = any(record["global_step"] == 29_700 for record in last)
    last_is_top1_alias = any(
        record["sha256"] == best[0]["sha256"] for record in last
    )
    if not last_is_full_budget and not last_is_top1_alias:
        raise SonataTrainingFinalizationError(
            "last checkpoint is neither full-budget nor a Top-1 alias"
        )
    return records, {
        "last_is_full_budget": last_is_full_budget,
        "last_is_top1_byte_identical_alias": last_is_top1_alias,
    }


def _checkpoint_selection_contract(
    records: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validation_scores: list[tuple[int, float]] = []
    for event in completed:
        epoch = int(event["completed_epoch"])
        if (epoch + 1) % 15 != 0:
            continue
        metrics = event.get("metrics")
        score = metrics.get("val_mean_t-AP") if isinstance(metrics, Mapping) else None
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise SonataTrainingFinalizationError(
                "validation checkpoint metric trajectory is incomplete"
            )
        validation_scores.append((epoch, float(score)))
    if len(validation_scores) != 30:
        raise SonataTrainingFinalizationError(
            "formal run requires exactly 30 validation metric points"
        )
    best_records = [record for record in records if record["role"] == "best_validation"]
    if len(best_records) != 1:
        raise SonataTrainingFinalizationError(
            "formal run requires one best-validation checkpoint"
        )
    record = best_records[0]
    best_score = max(score for _, score in validation_scores)
    selected_epoch_score = dict(validation_scores).get(int(record["epoch"]))
    exact_score = float(record["selection_metric_exact"])
    if (
        selected_epoch_score is None
        or not math.isclose(exact_score, best_score, rel_tol=0.0, abs_tol=1.0e-6)
        or not math.isclose(
            selected_epoch_score,
            best_score,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise SonataTrainingFinalizationError(
            "checkpoint does not match the highest validation val_mean_t-AP"
        )
    return {
        "monitor": "val_mean_t-AP",
        "mode": "max",
        "validation_event_count": len(validation_scores),
        "selected_epoch": int(record["epoch"]),
        "selection_metric_exact": exact_score,
        "selection_metric_filename_rounded": record[
            "selection_metric_filename_rounded"
        ],
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("ascii")


def _learning_curve(events: Sequence[Mapping[str, Any]]) -> bytes:
    metric_names = sorted(
        {
            str(name)
            for event in events
            for name in event.get("metrics", {})
            if isinstance(event.get("metrics"), Mapping)
        }
    )
    base_fields = [
        "epoch",
        "validation_performed",
        "optimizer_steps",
        "samples_seen_epoch",
        "samples_seen_total_raw",
        "samples_seen_total_reconstructed",
        "stage_observations_epoch",
        "stage_observations_total_raw",
        "stage_observations_total_reconstructed",
        "learning_rate",
        "peak_allocated_vram_mib",
        "peak_reserved_vram_mib",
        "process_wall_clock_seconds",
    ]
    rows = []
    for event in events:
        validation_performed = (int(event["completed_epoch"]) + 1) % 15 == 0
        row = {
            "epoch": event["completed_epoch"],
            "validation_performed": str(validation_performed).lower(),
            **{
                field: event.get(field)
                for field in base_fields
                if field not in {"epoch", "validation_performed"}
            },
        }
        metrics = event.get("metrics", {})
        row.update(
            {
                name: (
                    ""
                    if name.startswith("val_") and not validation_performed
                    else metrics.get(name, "")
                )
                for name in metric_names
            }
        )
        rows.append(row)
    return _csv_bytes(rows, [*base_fields, *metric_names])


def _checkpoint_csv(records: Sequence[Mapping[str, Any]]) -> bytes:
    fields = [
        "reference",
        "filename",
        "role",
        "sha256",
        "byte_size",
        "epoch",
        "global_step",
        "selection_metric_name",
        "selection_metric_exact",
        "selection_metric_filename_rounded",
        "state_dict_entry_count",
        "optimizer_state_count",
        "scheduler_state_count",
    ]
    return _csv_bytes(records, fields)


def _observed_wall_clock(events: Sequence[Mapping[str, Any]]) -> float | None:
    timestamps = []
    for event in events:
        value = event.get("timestamp_utc")
        if not isinstance(value, str):
            continue
        try:
            timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)).total_seconds()


def _runtime_reconciliation(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, int | None]:
    interruption_count = 0
    resume_launch_count = 0
    raw_resume_launch_count = 0
    raw_fit_interrupted_count = 0
    zero_checkpoint_resume_seen = False
    correction_version: int | None = None
    termination_events = {
        "storage_migration_started",
        "checkpoint_callback_path_repaired",
    }
    for event in events:
        event_name = event.get("event")
        if event_name == "launch_authorized" and event.get("launch_mode") == "resume":
            raw_resume_launch_count += 1
            checkpoint_count = event.get("checkpoint_count")
            if isinstance(checkpoint_count, int) and checkpoint_count > 0:
                resume_launch_count += 1
            elif checkpoint_count == 0:
                zero_checkpoint_resume_seen = True
            else:
                raise SonataTrainingFinalizationError(
                    "resume launch checkpoint count is invalid"
                )
        if event_name == "fit_interrupted":
            raw_fit_interrupted_count += 1
            interruption_count += 1
        elif event_name in termination_events and isinstance(
            event.get("termination"), str
        ):
            interruption_count += 1
        if event_name != "runtime_evidence_semantics_correction":
            continue
        version = event.get("correction_version")
        if not isinstance(version, int) or version <= 0:
            raise SonataTrainingFinalizationError(
                "runtime correction version is invalid"
            )
        if correction_version is None:
            if version != 1:
                raise SonataTrainingFinalizationError(
                    "runtime correction chain must start at version 1"
                )
        elif (
            version != correction_version + 1
            or event.get("supersedes_correction_version") != correction_version
        ):
            raise SonataTrainingFinalizationError(
                "runtime correction chain is not contiguous"
            )
        if (
            event.get("expected_checkpoint_resume_count") != resume_launch_count
            or event.get("expected_infrastructure_interruption_count")
            != interruption_count
        ):
            raise SonataTrainingFinalizationError(
                "runtime correction counts do not match their event prefix"
            )
        if (
            version == 1
            and zero_checkpoint_resume_seen
            and (
                event.get("basis_checkpoint_count") != 0
                or event.get("original_launch_mode") != "resume"
                or event.get("corrected_launch_mode") != "retry_before_checkpoint"
            )
        ):
            raise SonataTrainingFinalizationError(
                "zero-checkpoint launch correction is invalid"
            )
        correction_version = version
    if zero_checkpoint_resume_seen and correction_version is None:
        raise SonataTrainingFinalizationError(
            "zero-checkpoint resume launch lacks a semantics correction"
        )
    return {
        "interruption_count": interruption_count,
        "resume_launch_count": resume_launch_count,
        "correction_version": correction_version,
        "raw_fit_interrupted_count": raw_fit_interrupted_count,
        "raw_resume_launch_count": raw_resume_launch_count,
    }


def finalize_training(
    *,
    training_output_dir: Path,
    artifact_dir: Path,
    expected_candidate: Mapping[str, Any],
    source_snapshot_manifest: Mapping[str, Any],
    hardware: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(training_output_dir)
    destination = Path(artifact_dir)
    candidate = _load_json(
        output_dir / CANDIDATE_RECORD_NAME,
        name="Sonata candidate record",
    )
    if candidate != dict(expected_candidate):
        raise SonataTrainingFinalizationError("candidate contract mismatch")
    bindings = candidate.get("bindings", {})
    training_source = source_snapshot_manifest.get("training_source")
    evidence_producer = source_snapshot_manifest.get("evidence_producer")
    if (
        source_snapshot_manifest.get("schema_version") != 2
        or source_snapshot_manifest.get("status") != "pass"
        or not isinstance(training_source, Mapping)
        or not isinstance(evidence_producer, Mapping)
        or training_source.get("status") != "pass"
        or evidence_producer.get("status") != "pass"
        or training_source.get("source_commit") != bindings.get("source_commit")
        or training_source.get("content_sha256") != bindings.get("source_tree_sha256")
    ):
        raise SonataTrainingFinalizationError("source snapshot binding mismatch")
    producer_commit = evidence_producer.get("source_commit")
    producer_sha256 = evidence_producer.get("content_sha256")
    if (
        not isinstance(producer_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", producer_commit)
        or not isinstance(producer_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", producer_sha256)
    ):
        raise SonataTrainingFinalizationError(
            "evidence producer source identity is invalid"
        )

    recipe = candidate.get("recipe", {})
    epochs = int(recipe.get("epochs", -1))
    steps_per_epoch = int(recipe.get("optimizer_steps_per_epoch", -1))
    samples_per_epoch = int(recipe.get("samples_per_epoch", -1))
    if (epochs, steps_per_epoch, samples_per_epoch) != (450, 66, 2112):
        raise SonataTrainingFinalizationError("candidate budget contract mismatch")
    events, portable_events = _read_runtime_events(output_dir / RUNTIME_EVENTS_NAME)
    completed = _completed_epochs(
        events,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        samples_per_epoch=samples_per_epoch,
    )
    checkpoints, checkpoint_semantics = _checkpoint_inventory(output_dir)
    checkpoint_selection = _checkpoint_selection_contract(checkpoints, completed)
    target_names = (
        "TRAINING_MANIFEST.json",
        "TRAINING_REPORT.md",
        "learning_curve.csv",
        "checkpoint_inventory.csv",
        "runtime_events.jsonl",
        "source_snapshot_manifest.json",
    )
    if any((destination / name).exists() for name in target_names):
        raise SonataTrainingFinalizationError("SS4 training artifacts already exist")

    learning_curve = _learning_curve(completed)
    checkpoint_csv = _checkpoint_csv(checkpoints)
    source_bytes = _json_bytes(source_snapshot_manifest)
    runtime_reconciliation = _runtime_reconciliation(events)
    stage_observations = sum(
        int(event["stage_observations_epoch"]) for event in completed
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "stage": "SS4",
        "candidate_id": candidate["candidate_id"],
        "bindings": bindings,
        "recipe": recipe,
        "budget": {
            "completed_epochs": epochs,
            "optimizer_steps": epochs * steps_per_epoch,
            "samples_seen": epochs * samples_per_epoch,
            "stage_observations_seen": stage_observations,
        },
        "runtime": {
            "observed_wall_clock_seconds": _observed_wall_clock(events),
            **runtime_reconciliation,
            "peak_allocated_vram_mib": max(
                float(event.get("peak_allocated_vram_mib", 0.0))
                for event in completed
            ),
            "peak_reserved_vram_mib": max(
                float(event.get("peak_reserved_vram_mib", 0.0))
                for event in completed
            ),
            "hardware": dict(hardware or {}),
        },
        "checkpoint_semantics": checkpoint_semantics,
        "checkpoint_selection": checkpoint_selection,
        "checkpoints": checkpoints,
        "artifacts": {
            "learning_curve_sha256": hashlib.sha256(learning_curve).hexdigest(),
            "checkpoint_inventory_sha256": hashlib.sha256(
                checkpoint_csv
            ).hexdigest(),
            "runtime_events_sha256": hashlib.sha256(portable_events).hexdigest(),
            "source_snapshot_manifest_sha256": hashlib.sha256(
                source_bytes
            ).hexdigest(),
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    report = "\n".join(
        [
            "# Sonata Second-Perception SS4 Training Report",
            "",
            "- Status: PASS",
            f"- Candidate: `{candidate['candidate_id']}`",
            "- Seed / epochs: `45 / 450`",
            "- Effective global batch: `32`",
            f"- Optimizer steps: `{epochs * steps_per_epoch}`",
            f"- Samples seen: `{epochs * samples_per_epoch}`",
            f"- Stage observations seen: `{stage_observations}`",
            f"- Infrastructure interruptions: `{manifest['runtime']['interruption_count']}`",
            f"- Resume launches: `{manifest['runtime']['resume_launch_count']}`",
            "- Checkpoint selection remains frozen to highest validation `val_mean_t-AP`.",
            "- No Persist4D, Protocol-B, B2, B3, or B4 input was used for selection.",
            "",
        ]
    ).encode("ascii")
    _publish(destination / "learning_curve.csv", learning_curve)
    _publish(destination / "checkpoint_inventory.csv", checkpoint_csv)
    _publish(destination / "runtime_events.jsonl", portable_events)
    _publish(destination / "source_snapshot_manifest.json", source_bytes)
    _publish(destination / "TRAINING_MANIFEST.json", _json_bytes(manifest))
    _publish(destination / "TRAINING_REPORT.md", report)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--devices", default="1,2")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        devices = tuple(int(value) for value in args.devices.split(","))
    except ValueError as error:
        raise SonataTrainingFinalizationError("devices are invalid") from error
    smoke = _load_json(
        PROJECT_ROOT
        / "artifacts/sonata_second_perception_v1/smoke/smoke_results.json",
        name="Sonata smoke authorization",
    )
    expected_candidate = _candidate_contract(
        artifact_dir=(
            PROJECT_ROOT / "artifacts/sonata_second_perception_v1/preflight"
        ),
        smoke_authorization=smoke,
        devices=devices,
    )
    training_source = build_sonata_source_tree_contract(
        PROJECT_ROOT,
        require_clean=False,
        revision=expected_candidate["bindings"]["source_commit"],
    )
    evidence_producer = build_sonata_source_tree_contract(
        PROJECT_ROOT, require_clean=True
    )
    source_snapshot = {
        "schema_version": 2,
        "status": "pass",
        "training_source": training_source,
        "evidence_producer": evidence_producer,
    }
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            training_source["source_commit"],
            "HEAD",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SonataTrainingFinalizationError(
            "authorized source commit is not an ancestor of HEAD"
        )
    environment = _load_json(
        PROJECT_ROOT
        / "artifacts/sonata_second_perception_v1/preflight/environment_manifest.json",
        name="Sonata environment manifest",
    )
    manifest = finalize_training(
        training_output_dir=args.training_output_dir,
        artifact_dir=args.artifact_dir,
        expected_candidate=expected_candidate,
        source_snapshot_manifest=source_snapshot,
        hardware=environment.get("hardware", {}),
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "optimizer_steps": manifest["budget"]["optimizer_steps"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
