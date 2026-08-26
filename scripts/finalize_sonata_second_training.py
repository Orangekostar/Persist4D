#!/usr/bin/env python3
"""Finalize portable SS4 evidence after the full Sonata run completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
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
    role = "best_validation" if best_match else "last"
    return {
        "reference": f"external:sonata_training_output/{path.name}",
        "filename": path.name,
        "role": role,
        "sha256": file_sha256(path),
        "byte_size": path.stat().st_size,
        "epoch": epoch,
        "global_step": global_step,
        "val_mean_t_ap": float(best_match.group("score")) if best_match else None,
        "state_dict_entry_count": len(checkpoint["state_dict"]),
        "optimizer_state_count": len(checkpoint["optimizer_states"]),
        "scheduler_state_count": len(checkpoint["lr_schedulers"]),
    }


def _checkpoint_inventory(output_dir: Path) -> list[dict[str, Any]]:
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
    if max(record["global_step"] for record in last) != 29_700:
        raise SonataTrainingFinalizationError(
            "last checkpoint does not contain the full training budget"
        )
    return records


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
        "optimizer_steps",
        "samples_seen_epoch",
        "stage_observations_epoch",
        "learning_rate",
        "peak_allocated_vram_mib",
        "peak_reserved_vram_mib",
        "process_wall_clock_seconds",
    ]
    rows = []
    for event in events:
        row = {
            "epoch": event["completed_epoch"],
            **{field: event.get(field) for field in base_fields if field != "epoch"},
        }
        metrics = event.get("metrics", {})
        row.update({name: metrics.get(name, "") for name in metric_names})
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
        "val_mean_t_ap",
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
    if (
        source_snapshot_manifest.get("source_commit")
        != bindings.get("source_commit")
        or source_snapshot_manifest.get("content_sha256")
        != bindings.get("source_tree_sha256")
    ):
        raise SonataTrainingFinalizationError("source snapshot binding mismatch")

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
    checkpoints = _checkpoint_inventory(output_dir)
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
            "interruption_count": sum(
                event.get("event") == "fit_interrupted" for event in events
            ),
            "resume_launch_count": sum(
                event.get("event") == "launch_authorized"
                and event.get("launch_mode") == "resume"
                for event in events
            ),
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
    source_snapshot = build_sonata_source_tree_contract(
        PROJECT_ROOT, require_clean=True
    )
    source_snapshot["source_commit"] = expected_candidate["bindings"][
        "source_commit"
    ]
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            source_snapshot["source_commit"],
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
