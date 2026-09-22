"""Resumable command-line driver for Persist4D Perception Gain V1."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.crosswindow_cache import ASSET_KEYS, resolve_assets
from scripts.perception_gain_data import build_data_roles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/perception_gain_v1.yaml"
RUN_STATE_SCHEMA = "perception-gain-run-state-v1"
STAGES = (
    "bind",
    "foundation",
    "scorer",
    "perception_pilot",
    "perception_full",
    "perception_select",
    "refinement",
    "final_lock",
    "replication",
    "confirm",
    "profile",
    "report",
    "publish",
)
PIPELINE = ("bootstrap", *STAGES)
_RUNTIME_CONTRACT = {
    "batch_size_per_gpu": 2,
    "cpu_workers": 8,
    "devices": 2,
    "gradient_accumulation": 8,
    "precision": "32-true",
    "train_seed": 45,
    "eval_seed": 45,
}
PILOT_VARIANTS = ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")
PILOT_UPDATES = (250, 750)
PILOT_ENDPOINT = 750
PILOT_LOGICAL_UNITS = 23
FULL_UPDATES = (0, 250, 750, 1500, 2250, 3000)
FULL_ENDPOINT = 3000
SEL_LOGICAL_UNITS = 24
REFINER_UPDATES = (0, 500, 1000, 1500)


class CampaignError(RuntimeError):
    """Raised when campaign identities, state, or frozen settings differ."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_campaign_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CampaignError(f"cannot read campaign config: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "perception-gain-v1"
    ):
        raise CampaignError("campaign config schema differs")
    for section in (
        "identity",
        "paths",
        "runtime",
        "budget",
        "training",
        "scorer",
        "refiner",
        "evaluation",
    ):
        if not isinstance(value.get(section), dict):
            raise CampaignError(f"campaign config lacks {section}")
    if value["runtime"] != _RUNTIME_CONTRACT:
        raise CampaignError("runtime contract differs")
    budget = value["budget"]
    category_total = sum(
        int(budget[name])
        for name in (
            "foundation_and_evaluation_gpu_hours",
            "perception_training_gpu_hours",
            "refinement_gpu_hours",
            "profiling_gpu_hours",
            "smoke_and_recovery_gpu_hours",
        )
    )
    if int(budget.get("total_gpu_hours", -1)) != 192 or category_total != 192:
        raise CampaignError("GPU-hour budget differs")
    return value


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def new_run_state(
    *, parent: str, instruction_sha: str, config_sha: str, data_sha: str
) -> dict[str, object]:
    identity = {
        "parent": parent,
        "instruction_sha256": instruction_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
    }
    if not _hex(parent, 40) or any(
        not _hex(value, 64) for value in (instruction_sha, config_sha, data_sha)
    ):
        raise CampaignError("run-state identities must be Git/SHA256 hex strings")
    return {
        "schema_version": RUN_STATE_SCHEMA,
        "identity": identity,
        "stage": "bootstrap",
        "stage_status": "PASS",
        "completed_stages": ["bootstrap"],
        "budget_used": {
            "foundation_and_evaluation_gpu_hours": 0.0,
            "perception_training_gpu_hours": 0.0,
            "refinement_gpu_hours": 0.0,
            "profiling_gpu_hours": 0.0,
            "smoke_and_recovery_gpu_hours": 0.0,
            "new_cache_bytes": 0,
        },
        "failures": [],
        "next_command": (
            "python -m scripts.perception_gain_campaign run --config "
            "configs/perception_gain_v1.yaml --external-root "
            '"$PERSIST4D_PERCEPTION_RUN_ROOT" --through bind --resume'
        ),
    }


def load_resume_state(
    path: Path,
    *,
    parent: str,
    instruction_sha: str,
    config_sha: str,
    data_sha: str,
) -> dict[str, Any]:
    state = _read_json(path)
    expected = {
        "parent": parent,
        "instruction_sha256": instruction_sha,
        "config_sha256": config_sha,
        "data_sha256": data_sha,
    }
    if state.get("schema_version") != RUN_STATE_SCHEMA:
        raise CampaignError("run-state schema differs")
    if state.get("identity") != expected:
        raise CampaignError("resume identity differs")
    return state


def pending_stages(completed_stages: Sequence[str], *, through: str) -> tuple[str, ...]:
    if through not in STAGES:
        raise CampaignError(f"unknown campaign stage: {through}")
    completed = tuple(completed_stages)
    if completed != PIPELINE[: len(completed)]:
        raise CampaignError("completed stages must be an exact prefix")
    endpoint = PIPELINE.index(through) + 1
    if len(completed) >= endpoint:
        return ()
    return PIPELINE[len(completed) : endpoint]


def build_partial_finalization_state(
    state: Mapping[str, object], *, interruption: Mapping[str, object]
) -> dict[str, object]:
    completed = state.get("completed_stages")
    failure = interruption.get("failure")
    progress = interruption.get("resume_progress")
    resource = interruption.get("resource_accounting")
    if (
        isinstance(completed, (str, bytes))
        or not isinstance(completed, Sequence)
        or tuple(completed) != PIPELINE[: len(completed)]
        or "perception_pilot" in completed
        or interruption.get("schema_version")
        != "perception-gain-training-interruption-v1"
        or interruption.get("status") != "RESUMABLE_EXTERNAL_INTERRUPTION"
        or not isinstance(failure, Mapping)
        or not isinstance(progress, Mapping)
        or not isinstance(resource, Mapping)
        or not isinstance(interruption.get("checkpoint"), str)
        or not _hex(interruption.get("checkpoint_sha256"), 64)
        or not isinstance(progress.get("completed_optimizer_updates"), int)
        or isinstance(resource.get("accounted_gpu_hours"), bool)
        or not isinstance(resource.get("accounted_gpu_hours"), (int, float))
        or float(resource["accounted_gpu_hours"]) < 0.0
    ):
        raise CampaignError("partial finalization interruption evidence differs")
    record = {
        "stage": "perception_pilot",
        "status": "RESUMABLE_EXTERNAL_INTERRUPTION",
        "reason": failure.get("reason"),
        "external_dependency": failure.get("external_dependency"),
        "external_dependency_status": failure.get("external_dependency_status"),
        "resume_checkpoint": interruption["checkpoint"],
        "resume_checkpoint_sha256": interruption["checkpoint_sha256"],
        "completed_optimizer_updates": progress["completed_optimizer_updates"],
        "completed_arm_updates": {
            "C0": 750,
            "S-BAL": progress["completed_optimizer_updates"],
        },
        "accounted_gpu_hours": float(resource["accounted_gpu_hours"]),
    }
    failures = [
        dict(item)
        for item in state.get("failures", ())
        if isinstance(item, Mapping) and item.get("stage") != "perception_pilot"
    ]
    return {
        **dict(state),
        "execution_status": "PARTIAL_WITH_BLOCKERS",
        "stage": "perception_pilot",
        "stage_status": "BLOCKED",
        "failures": [*failures, record],
        "partial_delivery": {
            "status": "PREPARING",
            "uncompleted_stages": list(PIPELINE[len(completed) :]),
            "report": "PENDING",
            "publish": "PENDING",
        },
        "next_command": state.get("next_command"),
    }


def reconcile_partial_costs(
    state: Mapping[str, object],
    *,
    artifact_root: Path,
    external_root: Path,
    interruption: Mapping[str, object],
) -> dict[str, object]:
    if "partial_costs" in state:
        return dict(state)
    budget = state.get("budget_used")
    resource = interruption.get("resource_accounting")
    c0_summary = _read_json(external_root / "training/C0/run_summary.json")
    if (
        not isinstance(budget, Mapping)
        or not isinstance(resource, Mapping)
        or c0_summary.get("status") != "COMPLETE"
        or c0_summary.get("variant") != "C0"
        or c0_summary.get("completed_global_step") != 750
        or isinstance(c0_summary.get("gpu_hours"), bool)
        or not isinstance(c0_summary.get("gpu_hours"), (int, float))
        or isinstance(resource.get("accounted_gpu_hours"), bool)
        or not isinstance(resource.get("accounted_gpu_hours"), (int, float))
    ):
        raise CampaignError("partial training cost evidence differs")
    evaluations = []
    for path in sorted(
        (artifact_root / "training/pilot/evaluation/cal").glob("*/*.json")
    ):
        summary = _read_json(path)
        gpu_hours = summary.get("gpu_hours")
        if (
            summary.get("status") != "PASS"
            or isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or float(gpu_hours) < 0.0
        ):
            raise CampaignError("partial evaluation cost evidence differs")
        evaluations.append(
            {
                "path": path.relative_to(artifact_root).as_posix(),
                "gpu_hours": float(gpu_hours),
            }
        )
    if not evaluations:
        raise CampaignError("partial evaluation cost evidence is unavailable")
    c0_gpu_hours = float(c0_summary["gpu_hours"])
    interrupted_gpu_hours = float(resource["accounted_gpu_hours"])
    evaluation_gpu_hours = sum(row["gpu_hours"] for row in evaluations)
    updated_budget = dict(budget)
    updated_budget["perception_training_gpu_hours"] = (
        float(updated_budget.get("perception_training_gpu_hours", 0.0))
        + c0_gpu_hours
        + interrupted_gpu_hours
    )
    updated_budget["foundation_and_evaluation_gpu_hours"] = (
        float(updated_budget.get("foundation_and_evaluation_gpu_hours", 0.0))
        + evaluation_gpu_hours
    )
    return {
        **dict(state),
        "budget_used": updated_budget,
        "partial_costs": {
            "C0_training_gpu_hours": c0_gpu_hours,
            "S_BAL_interrupted_gpu_hours": interrupted_gpu_hours,
            "CAL_evaluation_gpu_hours": evaluation_gpu_hours,
            "CAL_evaluations": evaluations,
        },
    }


def _external_artifact_path(external_root: Path, reference: object) -> Path:
    if not isinstance(reference, str) or not reference.startswith("external:"):
        raise CampaignError("scorer artifact reference is invalid")
    relative = Path(reference.removeprefix("external:"))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignError("scorer artifact reference escapes the run root")
    return external_root.resolve() / relative


def validate_scorer_stage(
    *,
    data_manifest: Mapping[str, object],
    training_summary: Mapping[str, object],
    external_root: Path,
) -> dict[str, object]:
    if (
        data_manifest.get("status") != "PASS"
        or data_manifest.get("completed_pair_count") != 64
        or not isinstance(data_manifest.get("reference_count"), int)
        or int(data_manifest["reference_count"]) < 8
    ):
        raise CampaignError("scorer data coverage is incomplete")
    shards = data_manifest.get("shards")
    if (
        isinstance(shards, (str, bytes))
        or not isinstance(shards, Sequence)
        or not shards
    ):
        raise CampaignError("scorer data shards are unavailable")
    record_count = 0
    cache_bytes = 0
    for record in shards:
        if not isinstance(record, Mapping):
            raise CampaignError("scorer shard record is invalid")
        path = _external_artifact_path(external_root, record.get("external_reference"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise CampaignError("scorer shard hash or byte count differs")
        record_count += int(record.get("record_count", 0))
        cache_bytes += path.stat().st_size
    if record_count != 64:
        raise CampaignError("scorer shard record coverage differs")

    if (
        training_summary.get("status") != "COMPLETE"
        or training_summary.get("completed_updates") != 500
        or training_summary.get("checkpoint") != "update=0500.ckpt"
    ):
        raise CampaignError("scorer training endpoint is incomplete")
    checkpoint_reference = "external:training/scorer/model/update=0500.ckpt"
    checkpoint = _external_artifact_path(external_root, checkpoint_reference)
    if not checkpoint.is_file() or _sha256(checkpoint) != training_summary.get(
        "checkpoint_sha256"
    ):
        raise CampaignError("scorer checkpoint hash differs")
    data_gpu_hours = data_manifest.get("gpu_hours")
    training_gpu_hours = training_summary.get("gpu_hours")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) < 0.0
        for value in (data_gpu_hours, training_gpu_hours)
    ):
        raise CampaignError("scorer GPU-hour evidence is invalid")
    return {
        "status": "PASS",
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": training_summary["checkpoint_sha256"],
        "data_pair_count": 64,
        "reference_count": int(data_manifest["reference_count"]),
        "training_updates": 500,
        "gpu_hours": float(data_gpu_hours) + float(training_gpu_hours),
        "cache_bytes": cache_bytes,
    }


def validate_cached_bind_stage(
    binding: Mapping[str, object],
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    r1 = binding.get("r1")
    if (
        binding.get("schema_version") != "perception-gain-foundation-binding-v1"
        or binding.get("status") != "PASS"
        or not isinstance(r1, Mapping)
        or r1.get("checkpoint_sha256") != checkpoint_sha256
        or r1.get("checkpoint_bytes") != checkpoint_bytes
    ):
        raise CampaignError("cached bind evidence differs")
    return dict(binding)


def validate_cached_foundation_stage(
    *,
    corrected_e1: Mapping[str, object],
    live_smoke: Mapping[str, object],
    diagnostic_panel: Mapping[str, object],
    native_smoke: Mapping[str, object],
) -> dict[str, object]:
    records = (corrected_e1, live_smoke, diagnostic_panel, native_smoke)
    if any(value.get("status") != "PASS" for value in records):
        raise CampaignError("cached foundation evidence is incomplete")
    gpu_values = (
        live_smoke.get("gpu_hours"),
        diagnostic_panel.get("gpu_hours"),
        native_smoke.get("gpu_hours"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) < 0.0
        for value in gpu_values
    ):
        raise CampaignError("cached foundation evidence has invalid GPU accounting")
    return {
        "corrected_e1": dict(corrected_e1),
        "diagnostic_panel": dict(diagnostic_panel),
        "live_smoke": dict(live_smoke),
        "native_smoke": dict(native_smoke),
        "gpu_hours": sum(float(value) for value in gpu_values),
        "status": "PASS",
    }


def validate_perception_pilot_stage(
    *,
    training_summaries: Mapping[str, Mapping[str, object]],
    evaluation_summaries: Mapping[tuple[str, int], Mapping[str, object]],
    baseline_metrics: Mapping[object, object],
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import select_pilot_promotions

    if set(training_summaries) != set(PILOT_VARIANTS) or set(evaluation_summaries) != {
        (variant, update) for variant in PILOT_VARIANTS for update in PILOT_UPDATES
    }:
        raise CampaignError("pilot arm or evaluation coverage differs")

    training_gpu_hours = 0.0
    evaluation_gpu_hours = 0.0
    checkpoint_records: dict[tuple[str, int], Mapping[str, object]] = {}
    training_rows = []
    for variant in PILOT_VARIANTS:
        summary = training_summaries[variant]
        progress = summary.get("progress")
        gpu_hours = summary.get("gpu_hours")
        checkpoints = summary.get("checkpoints")
        if (
            summary.get("status") != "COMPLETE"
            or summary.get("variant") != variant
            or summary.get("completed_global_step") != PILOT_ENDPOINT
            or not isinstance(progress, Mapping)
            or progress.get("schema_version") != "perception-gain-exact-resume-v1"
            or progress.get("completed_local_batches") != 6000
            or progress.get("completed_optimizer_updates") != PILOT_ENDPOINT
            or progress.get("next_global_draw_index") != 24000
            or isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or float(gpu_hours) < 0.0
            or isinstance(checkpoints, (str, bytes))
            or not isinstance(checkpoints, Sequence)
        ):
            raise CampaignError(f"pilot training endpoint differs: {variant}")
        by_name = {
            record.get("name"): record
            for record in checkpoints
            if isinstance(record, Mapping)
        }
        required_names = {"last.ckpt", "update=0250.ckpt", "update=0750.ckpt"}
        if not required_names.issubset(by_name):
            raise CampaignError(f"pilot checkpoints are incomplete: {variant}")
        for name in sorted(required_names):
            record = by_name[name]
            path = external_root / "training" / variant / name
            digest = record.get("sha256")
            size = record.get("bytes")
            if (
                not path.is_file()
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or path.stat().st_size != size
                or not _hex(digest, 64)
            ):
                raise CampaignError(
                    f"pilot checkpoint record differs: {variant}/{name}"
                )
        for update in PILOT_UPDATES:
            checkpoint_records[(variant, update)] = by_name[f"update={update:04d}.ckpt"]
        training_gpu_hours += float(gpu_hours)
        training_rows.append(
            {
                "variant": variant,
                "completed_optimizer_updates": PILOT_ENDPOINT,
                "gpu_hours": float(gpu_hours),
            }
        )

    candidates = []
    evaluation_rows = []
    for variant in PILOT_VARIANTS:
        for update in PILOT_UPDATES:
            summary = evaluation_summaries[(variant, update)]
            candidate = summary.get("candidate")
            parity = summary.get("t2_same_forward_parity")
            gpu_hours = summary.get("gpu_hours")
            record = checkpoint_records[(variant, update)]
            expected_reference = f"external:training/{variant}/update={update:04d}.ckpt"
            if (
                summary.get("status") != "PASS"
                or summary.get("variant") != variant
                or summary.get("optimizer_update") != update
                or summary.get("data_role") != "CAL"
                or summary.get("expected_logical_unit_count") != PILOT_LOGICAL_UNITS
                or summary.get("completed_logical_unit_count") != PILOT_LOGICAL_UNITS
                or not isinstance(candidate, Mapping)
                or candidate.get("method_id") != variant
                or candidate.get("optimizer_update") != update
                or candidate.get("coverage_status") != "COMPLETE"
                or candidate.get("expected_logical_units") != PILOT_LOGICAL_UNITS
                or candidate.get("completed_logical_units") != PILOT_LOGICAL_UNITS
                or candidate.get("checkpoint") != expected_reference
                or candidate.get("checkpoint_sha256") != record.get("sha256")
                or not isinstance(parity, Mapping)
                or parity.get("status") != "PASS"
                or isinstance(gpu_hours, bool)
                or not isinstance(gpu_hours, (int, float))
                or float(gpu_hours) < 0.0
            ):
                raise CampaignError(
                    f"pilot evaluation checkpoint identity differs: {variant}/{update}"
                )
            candidates.append(dict(candidate))
            evaluation_gpu_hours += float(gpu_hours)
            evaluation_rows.append(
                {
                    "variant": variant,
                    "optimizer_update": update,
                    "gpu_hours": float(gpu_hours),
                }
            )

    selection = select_pilot_promotions(
        candidates,
        baseline_metrics=baseline_metrics,
    )
    return {
        "schema_version": "perception-gain-pilot-stage-v1",
        "status": "PASS",
        "pilot_endpoint": PILOT_ENDPOINT,
        "training": training_rows,
        "evaluations": evaluation_rows,
        "training_gpu_hours": training_gpu_hours,
        "evaluation_gpu_hours": evaluation_gpu_hours,
        "selection": selection,
    }


def validate_perception_full_stage(
    *,
    completed_full_arms: Sequence[str],
    training_summaries: Mapping[str, Mapping[str, object]],
    evaluation_summaries: Mapping[tuple[str, int], Mapping[str, object]],
    baseline_metrics: Mapping[object, object],
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import (
        compare_candidate,
        select_cal_checkpoints,
    )

    arms = tuple(completed_full_arms)
    if (
        not 1 <= len(arms) <= 3
        or arms[0] != "C0"
        or len(set(arms)) != len(arms)
        or any(variant not in PILOT_VARIANTS for variant in arms)
        or set(training_summaries) != set(arms)
        or set(evaluation_summaries)
        != {(variant, update) for variant in arms for update in FULL_UPDATES}
    ):
        raise CampaignError("full-training arm or evaluation coverage differs")

    checkpoint_records: dict[tuple[str, int], Mapping[str, object]] = {}
    training_rows = []
    training_gpu_hours = 0.0
    required_names = {
        "last.ckpt",
        *(f"update={update:04d}.ckpt" for update in FULL_UPDATES if update),
    }
    for variant in arms:
        summary = training_summaries[variant]
        progress = summary.get("progress")
        checkpoints = summary.get("checkpoints")
        gpu_hours = summary.get("gpu_hours")
        if (
            summary.get("status") != "COMPLETE"
            or summary.get("variant") != variant
            or summary.get("completed_global_step") != FULL_ENDPOINT
            or not isinstance(progress, Mapping)
            or progress.get("schema_version") != "perception-gain-exact-resume-v1"
            or progress.get("completed_local_batches") != 24000
            or progress.get("completed_optimizer_updates") != FULL_ENDPOINT
            or progress.get("next_global_draw_index") != 96000
            or isinstance(checkpoints, (str, bytes))
            or not isinstance(checkpoints, Sequence)
            or isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or float(gpu_hours) < 0.0
        ):
            raise CampaignError(f"full training endpoint differs: {variant}")
        by_name = {
            record.get("name"): record
            for record in checkpoints
            if isinstance(record, Mapping)
        }
        if not required_names.issubset(by_name):
            raise CampaignError(f"full checkpoints are incomplete: {variant}")
        for name in sorted(required_names):
            record = by_name[name]
            path = external_root / "training" / variant / name
            size = record.get("bytes")
            if (
                not path.is_file()
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or path.stat().st_size != size
                or not _hex(record.get("sha256"), 64)
            ):
                raise CampaignError(f"full checkpoint record differs: {variant}/{name}")
        for update in FULL_UPDATES:
            if update:
                checkpoint_records[(variant, update)] = by_name[
                    f"update={update:04d}.ckpt"
                ]
        training_gpu_hours += float(gpu_hours)
        training_rows.append(
            {
                "variant": variant,
                "completed_optimizer_updates": FULL_ENDPOINT,
                "gpu_hours": float(gpu_hours),
            }
        )

    candidates = []
    candidates_by_identity: dict[tuple[str, int], dict[str, object]] = {}
    evaluation_gpu_hours = 0.0
    incremental_evaluation_gpu_hours = 0.0
    for variant in arms:
        for update in FULL_UPDATES:
            summary = evaluation_summaries[(variant, update)]
            candidate = summary.get("candidate")
            parity = summary.get("t2_same_forward_parity")
            gpu_hours = summary.get("gpu_hours")
            if (
                summary.get("status") != "PASS"
                or summary.get("variant") != variant
                or summary.get("optimizer_update") != update
                or summary.get("data_role") != "CAL"
                or summary.get("expected_logical_unit_count") != PILOT_LOGICAL_UNITS
                or summary.get("completed_logical_unit_count") != PILOT_LOGICAL_UNITS
                or not isinstance(candidate, Mapping)
                or candidate.get("method_id") != variant
                or candidate.get("optimizer_update") != update
                or candidate.get("coverage_status") != "COMPLETE"
                or candidate.get("expected_logical_units") != PILOT_LOGICAL_UNITS
                or candidate.get("completed_logical_units") != PILOT_LOGICAL_UNITS
                or not _hex(candidate.get("checkpoint_sha256"), 64)
                or not isinstance(candidate.get("checkpoint"), str)
                or not candidate.get("checkpoint")
                or not isinstance(parity, Mapping)
                or parity.get("status") != "PASS"
                or isinstance(gpu_hours, bool)
                or not isinstance(gpu_hours, (int, float))
                or float(gpu_hours) < 0.0
            ):
                raise CampaignError(
                    f"full CAL evaluation identity differs: {variant}/{update}"
                )
            if update:
                record = checkpoint_records[(variant, update)]
                expected_reference = (
                    f"external:training/{variant}/update={update:04d}.ckpt"
                )
                if candidate.get("checkpoint") != expected_reference or candidate.get(
                    "checkpoint_sha256"
                ) != record.get("sha256"):
                    raise CampaignError(
                        f"full CAL checkpoint identity differs: {variant}/{update}"
                    )
            frozen_candidate = dict(candidate)
            candidates.append(frozen_candidate)
            candidates_by_identity[(variant, update)] = frozen_candidate
            evaluation_gpu_hours += float(gpu_hours)
            if update not in PILOT_UPDATES:
                incremental_evaluation_gpu_hours += float(gpu_hours)

    selected = select_cal_checkpoints(
        candidates,
        baseline_metrics=baseline_metrics,
        completed_full_arms=arms,
    )
    paired = []
    for variant in arms[1:]:
        for update in FULL_UPDATES:
            row = compare_candidate(
                candidates_by_identity[(variant, update)],
                baseline_metrics=candidates_by_identity[("C0", update)]["metrics"],
            )
            paired.append(row)
    return {
        "schema_version": "perception-gain-full-stage-v1",
        "status": "PASS",
        "completed_full_arms": list(arms),
        "training": training_rows,
        "training_gpu_hours": training_gpu_hours,
        "evaluation_gpu_hours": evaluation_gpu_hours,
        "incremental_evaluation_gpu_hours": incremental_evaluation_gpu_hours,
        "d0_r1_baseline": candidates_by_identity[("C0", 0)],
        "cal_checkpoint_selection": selected,
        "paired_vs_c0": paired,
    }


def validate_perception_selection_stage(
    *,
    cal_selection: Mapping[str, object],
    evaluation_summaries: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import select_perception_parent

    arms = cal_selection.get("completed_full_arms")
    selected = cal_selection.get("cal_checkpoint_selection")
    if (
        cal_selection.get("status") != "PASS"
        or isinstance(arms, (str, bytes))
        or not isinstance(arms, Sequence)
        or not arms
        or arms[0] != "C0"
        or len(set(arms)) != len(arms)
        or not isinstance(selected, Mapping)
        or set(selected) != set(arms)
        or set(evaluation_summaries) != {"R1", *arms}
    ):
        raise CampaignError("CAL selection is unavailable for SEL")

    candidates: list[dict[str, object]] = []
    evaluation_rows = []
    unique_costs: dict[tuple[str, int], float] = {}
    baseline_candidate: dict[str, object] | None = None
    for label in ("R1", *arms):
        summary = evaluation_summaries[label]
        candidate = summary.get("candidate")
        expected = None if label == "R1" else selected[label]
        expected_variant = "C0" if label == "R1" else label
        expected_update = (
            0
            if label == "R1"
            else (
                expected.get("optimizer_update")
                if isinstance(expected, Mapping)
                else None
            )
        )
        parity = summary.get("t2_same_forward_parity")
        gpu_hours = summary.get("gpu_hours")
        if label != "R1" and (
            not isinstance(expected, Mapping)
            or not isinstance(candidate, Mapping)
            or any(
                candidate.get(field) != expected.get(field)
                for field in (
                    "method_id",
                    "optimizer_update",
                    "checkpoint",
                    "checkpoint_sha256",
                )
            )
        ):
            raise CampaignError(f"SEL CAL-selected identity differs: {label}")
        if (
            not isinstance(expected_update, int)
            or summary.get("status") != "PASS"
            or summary.get("variant") != expected_variant
            or summary.get("optimizer_update") != expected_update
            or summary.get("data_role") != "SEL"
            or summary.get("expected_logical_unit_count") != SEL_LOGICAL_UNITS
            or summary.get("completed_logical_unit_count") != SEL_LOGICAL_UNITS
            or not isinstance(candidate, Mapping)
            or candidate.get("method_id") != expected_variant
            or candidate.get("optimizer_update") != expected_update
            or candidate.get("coverage_status") != "COMPLETE"
            or candidate.get("expected_logical_units") != SEL_LOGICAL_UNITS
            or candidate.get("completed_logical_units") != SEL_LOGICAL_UNITS
            or not _hex(candidate.get("checkpoint_sha256"), 64)
            or not isinstance(candidate.get("metrics"), Mapping)
            or not isinstance(parity, Mapping)
            or parity.get("status") != "PASS"
            or isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or float(gpu_hours) < 0.0
        ):
            raise CampaignError(f"SEL evaluation identity differs: {label}")
        if label == "R1":
            if candidate.get("checkpoint") != "external:r1_checkpoint":
                raise CampaignError("SEL R1 checkpoint identity differs")
            baseline_candidate = dict(candidate)
        else:
            candidates.append(dict(candidate))
        identity = (expected_variant, expected_update)
        unique_costs.setdefault(identity, float(gpu_hours))
        evaluation_rows.append(
            {
                "label": label,
                "variant": expected_variant,
                "optimizer_update": expected_update,
                "gpu_hours": float(gpu_hours),
            }
        )

    if baseline_candidate is None:
        raise CampaignError("SEL R1 baseline is unavailable")
    selection = select_perception_parent(
        candidates,
        d0_metrics=baseline_candidate["metrics"],
        c0_method_id="C0",
    )
    parent_candidate = selection.get("selected")
    if parent_candidate is None:
        parent_candidate = {
            **baseline_candidate,
            "method_id": "R1",
            "source_variant": "C0",
        }
        parent_cal_candidate = cal_selection.get("d0_r1_baseline")
        if not isinstance(parent_cal_candidate, Mapping):
            raise CampaignError("CAL R1 baseline is unavailable")
    else:
        parent_candidate = {
            **parent_candidate,
            "source_variant": selection["selected_method_id"],
        }
        parent_cal_candidate = selected[selection["selected_method_id"]]
    lock = {
        "schema_version": "perception-lock-v1",
        "status": "LOCKED",
        "selection_status": selection["status"],
        "selected_method_id": selection["selected_method_id"],
        "parent_candidate": parent_candidate,
        "parent_cal_candidate": parent_cal_candidate,
        "d0_r1_baseline": baseline_candidate,
        "cal_checkpoint_selection": dict(selected),
        "sel_candidates": candidates,
        "data_role": "SEL",
        "output_policy": "D0/lag1/mean",
        "selection_thresholds": {
            "base": {"S_min": -0.001, "S_mean": 0.005, "S_long": 0.0},
            "new_vs_c0": {"S_mean": 0.002, "S_min": -0.002},
        },
    }
    return {
        "schema_version": "perception-gain-selection-stage-v1",
        "status": "PASS",
        "evaluations": evaluation_rows,
        "evaluation_gpu_hours": sum(unique_costs.values()),
        "selection": selection,
        "lock": lock,
    }


def validate_refinement_stage(
    *,
    perception_lock: Mapping[str, object],
    data_manifest: Mapping[str, object],
    training_summary: Mapping[str, object],
    cal_evaluations: Mapping[int, Mapping[str, object]],
    sel_evaluation: Mapping[str, object],
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import (
        compare_candidate,
        rank_candidates,
        select_refiner,
    )

    parent = perception_lock.get("parent_candidate")
    parent_cal = perception_lock.get("parent_cal_candidate")
    if (
        perception_lock.get("schema_version") != "perception-lock-v1"
        or perception_lock.get("status") != "LOCKED"
        or not isinstance(parent, Mapping)
        or not isinstance(parent_cal, Mapping)
    ):
        raise CampaignError("perception lock is unavailable for refinement")
    parent_variant = parent.get("source_variant", parent.get("method_id"))
    parent_update = parent.get("optimizer_update")
    parent_checkpoint = parent.get("checkpoint")
    parent_sha = parent.get("checkpoint_sha256")
    if (
        parent_variant not in PILOT_VARIANTS
        or not isinstance(parent_update, int)
        or not isinstance(parent_checkpoint, str)
        or not _hex(parent_sha, 64)
        or not isinstance(parent.get("metrics"), Mapping)
        or not isinstance(parent_cal.get("metrics"), Mapping)
    ):
        raise CampaignError("locked perception parent identity differs")

    data_gpu_hours = data_manifest.get("gpu_hours")
    shards = data_manifest.get("shards")
    if (
        data_manifest.get("status") != "PASS"
        or data_manifest.get("variant") != parent_variant
        or data_manifest.get("optimizer_update") != parent_update
        or data_manifest.get("checkpoint") != parent_checkpoint
        or data_manifest.get("checkpoint_sha256") != parent_sha
        or not isinstance(data_manifest.get("selected_reference_count"), int)
        or int(data_manifest["selected_reference_count"]) < 8
        or data_manifest.get("completed_reference_count")
        != data_manifest.get("selected_reference_count")
        or not isinstance(data_manifest.get("candidate_count"), int)
        or int(data_manifest["candidate_count"]) <= 0
        or isinstance(shards, (str, bytes))
        or not isinstance(shards, Sequence)
        or not shards
        or isinstance(data_gpu_hours, bool)
        or not isinstance(data_gpu_hours, (int, float))
        or float(data_gpu_hours) < 0.0
    ):
        raise CampaignError("refiner data manifest differs")
    cache_bytes = 0
    record_count = 0
    for record in shards:
        if not isinstance(record, Mapping):
            raise CampaignError("refiner data shard record differs")
        path = _external_artifact_path(external_root, record.get("external_reference"))
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("sha256")
        ):
            raise CampaignError("refiner data shard identity differs")
        cache_bytes += path.stat().st_size
        record_count += int(record.get("record_count", 0))
    if cache_bytes != data_manifest.get(
        "cache_bytes"
    ) or record_count != data_manifest.get("candidate_count"):
        raise CampaignError("refiner data cache coverage differs")

    training_gpu_hours = training_summary.get("gpu_hours")
    if (
        training_summary.get("status") != "COMPLETE"
        or training_summary.get("completed_updates") != 1500
        or training_summary.get("checkpoint") != "update=1500.ckpt"
        or training_summary.get("reference_count")
        != data_manifest.get("completed_reference_count")
        or training_summary.get("candidate_count")
        != data_manifest.get("candidate_count")
        or isinstance(training_gpu_hours, bool)
        or not isinstance(training_gpu_hours, (int, float))
        or float(training_gpu_hours) < 0.0
    ):
        raise CampaignError("refiner training endpoint differs")
    checkpoint_records: dict[int, dict[str, object]] = {}
    for update in REFINER_UPDATES:
        path = external_root / "training/refiner/model" / f"update={update:04d}.ckpt"
        if not path.is_file():
            raise CampaignError(f"refiner checkpoint is unavailable: {update}")
        checkpoint_records[update] = {
            "checkpoint": f"external:training/refiner/model/update={update:04d}.ckpt",
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    if checkpoint_records[1500]["sha256"] != training_summary.get("checkpoint_sha256"):
        raise CampaignError("refiner final checkpoint identity differs")

    if set(cal_evaluations) != set(REFINER_UPDATES):
        raise CampaignError("refiner CAL checkpoint coverage differs")

    def validated_evaluation(
        summary: Mapping[str, object], *, role: str, update: int
    ) -> tuple[dict[str, object], float]:
        logical_units = PILOT_LOGICAL_UNITS if role == "CAL" else SEL_LOGICAL_UNITS
        candidate = summary.get("candidate")
        evaluated_parent = summary.get("parent")
        refiner = summary.get("refiner")
        parent_metrics = summary.get("parent_metrics")
        audit = summary.get("mask_change_audit")
        gpu_hours = summary.get("gpu_hours")
        record = checkpoint_records[update]
        if (
            summary.get("status") != "PASS"
            or summary.get("data_role") != role
            or summary.get("expected_logical_unit_count") != logical_units
            or summary.get("completed_logical_unit_count") != logical_units
            or not isinstance(evaluated_parent, Mapping)
            or evaluated_parent.get("variant") != parent_variant
            or evaluated_parent.get("optimizer_update") != parent_update
            or evaluated_parent.get("checkpoint") != parent_checkpoint
            or evaluated_parent.get("checkpoint_sha256") != parent_sha
            or not isinstance(refiner, Mapping)
            or refiner.get("updates") != update
            or refiner.get("checkpoint") != record["checkpoint"]
            or refiner.get("sha256") != record["sha256"]
            or not isinstance(candidate, Mapping)
            or candidate.get("method_id") != "R-REFINE"
            or candidate.get("optimizer_update") != update
            or candidate.get("checkpoint") != record["checkpoint"]
            or candidate.get("checkpoint_sha256") != record["sha256"]
            or candidate.get("coverage_status") != "COMPLETE"
            or candidate.get("expected_logical_units") != logical_units
            or candidate.get("completed_logical_units") != logical_units
            or not isinstance(candidate.get("metrics"), Mapping)
            or not isinstance(parent_metrics, Mapping)
            or not isinstance(audit, Mapping)
            or isinstance(gpu_hours, bool)
            or not isinstance(gpu_hours, (int, float))
            or float(gpu_hours) < 0.0
        ):
            raise CampaignError(f"refiner {role} evaluation identity differs: {update}")
        expected_parent_metrics = (
            parent_cal["metrics"] if role == "CAL" else parent["metrics"]
        )
        parent_probe = {**dict(candidate), "metrics": parent_metrics}
        parent_comparison = compare_candidate(
            parent_probe, baseline_metrics=expected_parent_metrics
        )
        if any(
            abs(float(value)) > 1e-12 for value in parent_comparison["deltas"].values()
        ):
            raise CampaignError(f"refiner {role} parent metrics differ: {update}")
        if update == 0 and (
            audit.get("changed_candidate_count") != 0
            or audit.get("changed_point_count") != 0
        ):
            raise CampaignError("refiner update 0 is not an exact no-op")
        return dict(candidate), float(gpu_hours)

    cal_candidates = []
    evaluation_gpu_hours = 0.0
    for update in REFINER_UPDATES:
        candidate, gpu_hours = validated_evaluation(
            cal_evaluations[update], role="CAL", update=update
        )
        cal_candidates.append(candidate)
        evaluation_gpu_hours += gpu_hours
    ranked = rank_candidates(cal_candidates, baseline_metrics=parent_cal["metrics"])
    cal_selected = ranked[0]
    selected_update = int(cal_selected["optimizer_update"])
    sel_candidate, sel_gpu_hours = validated_evaluation(
        sel_evaluation, role="SEL", update=selected_update
    )
    evaluation_gpu_hours += sel_gpu_hours
    selection = select_refiner(sel_candidate, parent_metrics=parent["metrics"])
    return {
        "schema_version": "perception-gain-refinement-stage-v1",
        "status": "PASS",
        "parent": dict(parent),
        "data_manifest": dict(data_manifest),
        "training": dict(training_summary),
        "cal_candidates": ranked,
        "cal_selected": cal_selected,
        "sel_candidate": sel_candidate,
        "selection": selection,
        "refinement_gpu_hours": float(data_gpu_hours) + float(training_gpu_hours),
        "evaluation_gpu_hours": evaluation_gpu_hours,
        "cache_bytes": cache_bytes,
        "checkpoint_records": checkpoint_records,
    }


def normalize_confirmation_evaluation(
    summary: Mapping[str, object],
    *,
    method_id: str,
    population_id: str,
    physical_evaluation_id: str,
    source_method: str | None = None,
) -> dict[str, object]:
    contracts = {
        "PB-129": {
            "role": "PB",
            "horizons": (2, 3, 4, 5),
            "units": 129,
            "prefixes": 516,
            "population": {
                str(horizon): {
                    "reference_count": 6,
                    "logical_unit_count": 129,
                }
                for horizon in (2, 3, 4, 5)
            },
        },
        "LOCAL-T2-154": {
            "role": None,
            "horizons": (2,),
            "units": 154,
            "prefixes": 154,
            "population": {"2": {"reference_count": 41, "logical_unit_count": 154}},
        },
        "ADDITIONAL-native-terminal": {
            "role": "ADDITIONAL",
            "horizons": (2, 3, 4),
            "units": 220,
            "prefixes": 220,
            "population": {
                "2": {"reference_count": 40, "logical_unit_count": 111},
                "3": {"reference_count": 23, "logical_unit_count": 77},
                "4": {"reference_count": 8, "logical_unit_count": 32},
            },
        },
    }
    contract = contracts.get(population_id)
    schema = summary.get("schema_version")
    if (
        contract is None
        or not isinstance(method_id, str)
        or not method_id
        or not isinstance(physical_evaluation_id, str)
        or not physical_evaluation_id
        or summary.get("status") not in {"PASS", "PARTIAL"}
        or not _hex(summary.get("population_manifest_sha256"), 64)
        or (
            contract["role"] is not None
            and summary.get("data_role") != contract["role"]
        )
    ):
        raise CampaignError("confirmation evaluation identity differs")

    horizons = contract["horizons"]
    metric_rows = summary.get("metric_rows")
    rows = (
        [dict(row) for row in metric_rows if isinstance(row, Mapping)]
        if isinstance(metric_rows, Sequence)
        and not isinstance(metric_rows, (str, bytes))
        else []
    )
    if source_method is not None:
        rows = [row for row in rows if row.get("method") == source_method]
    rows = [{**row, "method": method_id} for row in rows]

    if schema == "perception-gain-native-evaluation-v1":
        completed_by_horizon = summary.get("completed_units_by_horizon")
        if not isinstance(completed_by_horizon, Mapping):
            raise CampaignError("native confirmation coverage differs")
        completed_sets = []
        for horizon in horizons:
            values = completed_by_horizon.get(str(horizon))
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise CampaignError("native confirmation horizon coverage differs")
            completed_sets.append(set(values))
        completed_units = len(
            set.union(*completed_sets)
            if population_id == "ADDITIONAL-native-terminal"
            else set.intersection(*completed_sets)
        )
        completed_prefixes = summary.get("completed_prefix_count")
        expected_units = summary.get("expected_logical_unit_count")
        expected_prefixes = summary.get("expected_prefix_count")
        population = summary.get("population_by_horizon")
        metrics_source = None
    elif schema == "perception-gain-checkpoint-evaluation-v1":
        candidate = summary.get("candidate")
        if not isinstance(candidate, Mapping):
            raise CampaignError("D0 confirmation candidate differs")
        metrics_source = candidate.get("metrics")
        completed_units = summary.get("completed_logical_unit_count")
        expected_units = summary.get("expected_logical_unit_count")
        completed_prefixes = int(completed_units) * len(horizons)
        expected_prefixes = int(expected_units) * len(horizons)
        population = contract["population"]
    elif schema == "perception-gain-local-t2-v1":
        metrics_payload = summary.get("metrics")
        if not isinstance(metrics_payload, Mapping):
            raise CampaignError("LOCAL-T2 confirmation metrics differ")
        metrics_source = {"2": metrics_payload.get("t_mAP")}
        completed_units = summary.get("validation_sequence_count")
        expected_units = contract["units"]
        completed_prefixes = completed_units
        expected_prefixes = contract["prefixes"]
        population = contract["population"]
        rows = [
            {
                "method": method_id,
                "reference": "all",
                "T": 2,
                "t_mAP": metrics_payload.get("t_mAP"),
                "logical_unit_count": completed_units,
                "reference_count": 41,
            }
        ]
    elif schema == "perception-refiner-evaluation-v1":
        if source_method not in {"PARENT", "R-REFINE"}:
            raise CampaignError("refiner confirmation source method differs")
        candidate = summary.get("candidate")
        metrics_source = (
            summary.get("parent_metrics")
            if source_method == "PARENT"
            else candidate.get("metrics") if isinstance(candidate, Mapping) else None
        )
        completed_units = summary.get("completed_logical_unit_count")
        expected_units = summary.get("expected_logical_unit_count")
        completed_prefixes = int(completed_units) * (
            len(horizons) if population_id == "PB-129" else 1
        )
        expected_prefixes = int(expected_units) * (
            len(horizons) if population_id == "PB-129" else 1
        )
        population = summary.get("population_by_horizon")
    else:
        raise CampaignError("confirmation evaluation schema differs")

    if metrics_source is None:
        pooled = {
            str(row.get("T")): row.get("t_mAP")
            for row in rows
            if row.get("reference") == "all"
        }
        metrics_source = pooled
    if not isinstance(metrics_source, Mapping):
        raise CampaignError("confirmation metrics are unavailable")
    metrics = {
        str(horizon): metrics_source.get(str(horizon), metrics_source.get(horizon))
        for horizon in horizons
    }
    if (
        expected_units != contract["units"]
        or expected_prefixes != contract["prefixes"]
        or isinstance(completed_units, bool)
        or not isinstance(completed_units, int)
        or isinstance(completed_prefixes, bool)
        or not isinstance(completed_prefixes, int)
        or population != contract["population"]
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
            for value in metrics.values()
        )
        or isinstance(summary.get("gpu_hours"), bool)
        or not isinstance(summary.get("gpu_hours"), (int, float))
    ):
        raise CampaignError("confirmation evaluation contract differs")
    return {
        "status": "MEASURED",
        "method_id": method_id,
        "population_id": population_id,
        "population_hash": summary["population_manifest_sha256"],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "coverage": {
            "expected_units": expected_units,
            "completed_units": completed_units,
            "expected_prefixes": expected_prefixes,
            "completed_prefixes": completed_prefixes,
        },
        "population_by_horizon": dict(population),
        "physical_evaluation_id": physical_evaluation_id,
        "gpu_hours": float(summary["gpu_hours"]),
        "metric_rows": rows,
    }


def build_replication_decision(
    *,
    final_lock: Mapping[str, object],
    estimated_perception_gpu_hours: float,
    estimated_refinement_gpu_hours: float,
    remaining_perception_gpu_hours: float,
    remaining_refinement_gpu_hours: float,
) -> dict[str, object]:
    recipe = final_lock.get("recipe")
    numbers = (
        estimated_perception_gpu_hours,
        estimated_refinement_gpu_hours,
        remaining_perception_gpu_hours,
        remaining_refinement_gpu_hours,
    )
    if (
        final_lock.get("status") != "LOCKED"
        or not isinstance(recipe, Mapping)
        or not isinstance(recipe.get("parent_optimizer_update"), int)
        or not isinstance(recipe.get("refiner_enabled"), bool)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0.0
            for value in numbers
        )
    ):
        raise CampaignError("replication decision inputs differ")
    needs_perception = int(recipe["parent_optimizer_update"]) > 0
    needs_refiner = bool(recipe["refiner_enabled"])
    scopes = [
        name
        for name, enabled in (
            ("perception_seed46", needs_perception),
            ("refiner_seed46", needs_refiner),
        )
        if enabled
    ]
    if not scopes:
        return {
            "decision": "SKIP",
            "replication_status": "NOT_APPLICABLE",
            "replication_scope": [],
            "reason": "locked recipe contains no newly trained deployed component",
        }
    affordable = float(estimated_perception_gpu_hours) <= float(
        remaining_perception_gpu_hours
    ) and float(estimated_refinement_gpu_hours) <= float(remaining_refinement_gpu_hours)
    return {
        "decision": "RUN_REQUIRED" if affordable else "SKIP",
        "replication_status": None if affordable else "NOT_RUN_BUDGET",
        "replication_scope": scopes,
        "estimated_perception_gpu_hours": float(estimated_perception_gpu_hours),
        "estimated_refinement_gpu_hours": float(estimated_refinement_gpu_hours),
        "remaining_perception_gpu_hours": float(remaining_perception_gpu_hours),
        "remaining_refinement_gpu_hours": float(remaining_refinement_gpu_hours),
        "reason": (
            None if affordable else "optional second seed exceeds category budget"
        ),
    }


def build_confirmation_summary(
    *,
    pb: Mapping[str, Mapping[str, object]],
    local_t2: Mapping[str, Mapping[str, object]],
    additional: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    required = {
        "PB": {
            "methods": {
                "FH-R1-native",
                "D0-R1",
                "C0-best-D0",
                "FH-P*-native",
                "P*-D0",
                "FINAL",
            },
            "horizons": (2, 3, 4, 5),
            "units": 129,
            "prefixes": 516,
            "population": {
                str(horizon): {
                    "reference_count": 6,
                    "logical_unit_count": 129,
                }
                for horizon in (2, 3, 4, 5)
            },
        },
        "LOCAL-T2": {
            "methods": {"R1-native", "C0-native", "P*-native"},
            "horizons": (2,),
            "units": 154,
            "prefixes": 154,
            "population": {"2": {"reference_count": 41, "logical_unit_count": 154}},
        },
        "ADDITIONAL": {
            "methods": {"FH-R1-native", "P*-D0", "FINAL"},
            "horizons": (2, 3, 4),
            "units": 220,
            "prefixes": 220,
            "population": {
                "2": {"reference_count": 40, "logical_unit_count": 111},
                "3": {"reference_count": 23, "logical_unit_count": 77},
                "4": {"reference_count": 8, "logical_unit_count": 32},
            },
        },
    }
    groups = {"PB": pb, "LOCAL-T2": local_t2, "ADDITIONAL": additional}
    complete: dict[tuple[str, str], bool] = {}
    physical_costs: dict[str, float] = {}
    for group_name, methods in groups.items():
        contract = required[group_name]
        if set(methods) != contract["methods"]:
            raise CampaignError(f"{group_name} confirmation methods differ")
        population_ids = set()
        population_hashes = set()
        for method_id, row in methods.items():
            metrics = row.get("metrics")
            coverage = row.get("coverage")
            population = row.get("population_by_horizon")
            physical_id = row.get("physical_evaluation_id")
            gpu_hours = row.get("gpu_hours")
            if (
                row.get("status") not in {"MEASURED", "ALIASED"}
                or row.get("method_id") != method_id
                or not isinstance(row.get("population_id"), str)
                or not row.get("population_id")
                or not _hex(row.get("population_hash"), 64)
                or not isinstance(metrics, Mapping)
                or set(metrics) != {str(value) for value in contract["horizons"]}
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0.0 <= float(value) <= 1.0
                    for value in metrics.values()
                )
                or not isinstance(coverage, Mapping)
                or coverage.get("expected_units") != contract["units"]
                or coverage.get("expected_prefixes") != contract["prefixes"]
                or not isinstance(coverage.get("completed_units"), int)
                or not 0 <= int(coverage["completed_units"]) <= contract["units"]
                or not isinstance(coverage.get("completed_prefixes"), int)
                or not 0 <= int(coverage["completed_prefixes"]) <= contract["prefixes"]
                or population != contract["population"]
                or not isinstance(physical_id, str)
                or not physical_id
                or isinstance(gpu_hours, bool)
                or not isinstance(gpu_hours, (int, float))
                or float(gpu_hours) < 0.0
            ):
                raise CampaignError(
                    f"{group_name} confirmation result differs: {method_id}"
                )
            population_ids.add(row["population_id"])
            population_hashes.add(row["population_hash"])
            prior_cost = physical_costs.setdefault(physical_id, float(gpu_hours))
            if abs(prior_cost - float(gpu_hours)) > 1e-12:
                raise CampaignError("aliased confirmation GPU costs differ")
            complete[(group_name, method_id)] = (
                coverage["completed_units"] == contract["units"]
                and coverage["completed_prefixes"] == contract["prefixes"]
            )
        if len(population_ids) != 1 or len(population_hashes) != 1:
            raise CampaignError(f"{group_name} confirmation populations differ")

    def all_t_strictly_higher(
        candidate: Mapping[str, object], baseline: Mapping[str, object]
    ) -> bool:
        return all(
            float(candidate[str(horizon)]) > float(baseline[str(horizon)]) + 1e-6
            for horizon in (2, 3, 4, 5)
        )

    pb_comparable = all(
        complete[("PB", method)] for method in ("FH-R1-native", "D0-R1", "FINAL")
    )
    local_comparable = all(
        complete[("LOCAL-T2", method)]
        for method in ("R1-native", "C0-native", "P*-native")
    )
    final_pb = pb["FINAL"]["metrics"]
    d0_pb = pb["D0-R1"]["metrics"]
    native_pb = pb["FH-R1-native"]["metrics"]
    pb_vs_native = all_t_strictly_higher(final_pb, native_pb) if pb_comparable else None
    pb_vs_d0 = all_t_strictly_higher(final_pb, d0_pb) if pb_comparable else None
    local_gain = (
        float(local_t2["P*-native"]["metrics"]["2"])
        > float(local_t2["R1-native"]["metrics"]["2"]) + 1e-6
        if local_comparable
        else None
    )
    local_delta_c0 = (
        float(local_t2["P*-native"]["metrics"]["2"])
        - float(local_t2["C0-native"]["metrics"]["2"])
        if local_comparable
        else None
    )
    d0_deltas = {
        str(horizon): float(final_pb[str(horizon)]) - float(d0_pb[str(horizon)])
        for horizon in (2, 3, 4, 5)
    }
    recommend_final = bool(
        pb_vs_native is True
        and sum(d0_deltas.values()) / 4.0 > 0.0
        and min(d0_deltas.values()) >= -0.001
    )
    all_complete = all(complete.values())
    return {
        "schema_version": "perception-gain-confirmation-v1",
        "status": "PASS",
        "execution_status": "COMPLETE" if all_complete else "PARTIAL_WITH_BLOCKERS",
        "populations": {
            name: {
                "population_id": next(
                    iter({row["population_id"] for row in rows.values()})
                ),
                "population_hash": next(
                    iter({row["population_hash"] for row in rows.values()})
                ),
                "population_by_horizon": required[name]["population"],
            }
            for name, rows in groups.items()
        },
        "methods": {name: dict(rows) for name, rows in groups.items()},
        "comparisons": {
            "local_t2_gain": local_gain,
            "local_t2_vs_c0_delta": local_delta_c0,
            "pb_all_t_vs_native_fh": pb_vs_native,
            "pb_all_t_vs_d0": pb_vs_d0,
            "pb_final_vs_d0_deltas": d0_deltas if pb_comparable else None,
            "comparison_status": (
                "COMPLETE"
                if pb_comparable and local_comparable
                else "INCOMPLETE_COVERAGE"
            ),
        },
        "deployment": {
            "recommended_method": "FINAL" if recommend_final else "D0-R1",
            "final_selected": recommend_final,
        },
        "gpu_hours": sum(physical_costs.values()),
    }


def build_final_lock(
    *,
    perception_lock: Mapping[str, object],
    refinement: Mapping[str, object],
    campaign_identity: Mapping[str, object],
    campaign_config_sha256: str,
    data_roles_sha256: str,
) -> dict[str, object]:
    parent = perception_lock.get("parent_candidate")
    cal_selected = refinement.get("cal_selected")
    selection = refinement.get("selection")
    parent_commit = campaign_identity.get("parent_commit")
    if (
        perception_lock.get("schema_version") != "perception-lock-v1"
        or perception_lock.get("status") != "LOCKED"
        or refinement.get("schema_version") != "perception-gain-refinement-stage-v1"
        or refinement.get("status") != "PASS"
        or not isinstance(parent, Mapping)
        or parent.get("source_variant") not in PILOT_VARIANTS
        or not isinstance(parent.get("optimizer_update"), int)
        or not isinstance(parent.get("checkpoint"), str)
        or not _hex(parent.get("checkpoint_sha256"), 64)
        or not isinstance(cal_selected, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(selection.get("enabled"), bool)
        or not _hex(parent_commit, 40)
        or not _hex(campaign_config_sha256, 64)
        or not _hex(data_roles_sha256, 64)
    ):
        raise CampaignError("final lock inputs differ from the frozen campaign")
    enabled = bool(selection["enabled"])
    selected_refiner = selection.get("selected")
    if enabled and (
        not isinstance(selected_refiner, Mapping)
        or selected_refiner.get("optimizer_update")
        != cal_selected.get("optimizer_update")
        or selected_refiner.get("checkpoint") != cal_selected.get("checkpoint")
        or selected_refiner.get("checkpoint_sha256")
        != cal_selected.get("checkpoint_sha256")
    ):
        raise CampaignError("selected refiner identity differs from CAL lock")
    return {
        "schema_version": "perception-gain-final-lock-v1",
        "status": "LOCKED",
        "campaign_identity": dict(campaign_identity),
        "campaign_config_sha256": campaign_config_sha256,
        "data_roles_sha256": data_roles_sha256,
        "perception_selection_status": perception_lock.get("selection_status"),
        "refiner_selection_status": selection.get("status"),
        "recipe": {
            "parent_method_id": perception_lock.get("selected_method_id"),
            "parent_variant": parent["source_variant"],
            "parent_optimizer_update": parent["optimizer_update"],
            "parent_checkpoint": parent["checkpoint"],
            "parent_checkpoint_sha256": parent["checkpoint_sha256"],
            "association": "D0",
            "output_policy": "lag1",
            "score_reducer": "mean",
            "state_capacity": 100,
            "window_size": 2,
            "refiner_enabled": enabled,
            "refiner_optimizer_update": (
                selected_refiner["optimizer_update"] if enabled else None
            ),
            "refiner_checkpoint": (selected_refiner["checkpoint"] if enabled else None),
            "refiner_checkpoint_sha256": (
                selected_refiner["checkpoint_sha256"] if enabled else None
            ),
            "eval_seed": 45,
        },
        "perception_lock": dict(perception_lock),
        "evaluated_refiner_candidate": dict(cal_selected),
        "selected_refiner": dict(selected_refiner) if enabled else None,
        "refiner_sel_comparison": selection.get("comparison"),
        "selection_thresholds": {
            "S_min": -0.001,
            "S_mean": 0.003,
            "S_long": 0.0,
        },
    }


def _run_stage_process(
    command: Sequence[str],
    *,
    command_label: str,
    stage: str,
    device: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    execution_log: Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    started = dt.datetime.now(dt.timezone.utc)
    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    process = subprocess.Popen(
        list(command),
        cwd=PROJECT_ROOT,
        env=child_environment,
    )
    return_code = process.wait()
    _append_jsonl(
        execution_log,
        {
            "command": command_label,
            "cwd": "repo:.",
            "device": device,
            "elapsed_seconds": (
                dt.datetime.now(dt.timezone.utc) - started
            ).total_seconds(),
            "exit_code": return_code,
            "inputs": list(inputs),
            "outputs": list(outputs),
            "pid": process.pid,
            "stage": stage,
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    if return_code != 0:
        raise CampaignError(f"campaign subprocess failed: {command_label}")


def _run_perception_pilot_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    roles_path = artifact_root / "DATA_ROLES.json"
    scorer_checkpoint = external_root / "training/scorer/model/update=0500.ckpt"
    training_summaries: dict[str, Mapping[str, object]] = {}
    evaluation_summaries: dict[tuple[str, int], Mapping[str, object]] = {}

    for variant in PILOT_VARIANTS:
        run_dir = external_root / "training" / variant
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            resume = run_dir / "last.ckpt"
            command = [
                sys.executable,
                "-m",
                "scripts.train_perception_gain",
                "--variant",
                variant,
                "--assets",
                str(assets_path),
                "--roles",
                str(roles_path),
                "--external-root",
                str(external_root),
                "--stop-after-updates",
                str(PILOT_ENDPOINT),
            ]
            command_label = (
                "python -m scripts.train_perception_gain "
                f"--variant {variant} --assets external:assets.local.json "
                "--roles repo:artifacts/perception_gain_v1/DATA_ROLES.json "
                f"--external-root external:. --stop-after-updates {PILOT_ENDPOINT}"
            )
            inputs = ["external:r1_checkpoint", "external:data_root"]
            if resume.is_file():
                command.extend(("--resume", str(resume)))
                command_label += " --resume external:training/{}/last.ckpt".format(
                    variant
                )
                inputs.append(f"external:training/{variant}/last.ckpt")
            environment = None
            if variant == "Q-SEM":
                if not scorer_checkpoint.is_file():
                    raise CampaignError("Q-SEM scorer checkpoint is unavailable")
                environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
                inputs.append("external:training/scorer/model/update=0500.ckpt")
            _run_stage_process(
                command,
                command_label=command_label,
                stage="perception_pilot",
                device="cuda:0,cuda:1",
                inputs=inputs,
                outputs=[f"external:training/{variant}/run_summary.json"],
                execution_log=execution_log,
                environment=environment,
            )
        training_summaries[variant] = _read_json(summary_path)

    baseline_path = artifact_root / "training/pilot/evaluation/cal/C0/update=0000.json"
    if not baseline_path.is_file():
        _run_stage_process(
            (
                sys.executable,
                "-m",
                "scripts.perception_gain_evaluation",
                "--mode",
                "evaluate",
                "--variant",
                "C0",
                "--update",
                "0",
                "--role",
                "CAL",
                "--assets",
                str(assets_path),
                "--roles",
                str(roles_path),
                "--external-root",
                str(external_root),
                "--device",
                "cuda:0",
            ),
            command_label=(
                "python -m scripts.perception_gain_evaluation --mode evaluate "
                "--variant C0 --update 0 --role CAL "
                "--assets external:assets.local.json "
                "--roles repo:artifacts/perception_gain_v1/DATA_ROLES.json "
                "--external-root external:. --device cuda:0"
            ),
            stage="perception_pilot",
            device="cuda:0",
            inputs=["external:r1_checkpoint", "external:data_root"],
            outputs=[
                "repo:artifacts/perception_gain_v1/training/pilot/evaluation/"
                "cal/C0/update=0000.json"
            ],
            execution_log=execution_log,
        )
    baseline = _read_json(baseline_path)
    baseline_candidate = baseline.get("candidate")
    if (
        baseline.get("status") != "PASS"
        or not isinstance(baseline_candidate, Mapping)
        or baseline_candidate.get("method_id") != "C0"
        or baseline_candidate.get("optimizer_update") != 0
        or baseline_candidate.get("coverage_status") != "COMPLETE"
        or not isinstance(baseline_candidate.get("metrics"), Mapping)
    ):
        raise CampaignError("pilot baseline evaluation differs")

    for variant in PILOT_VARIANTS:
        environment = None
        scorer_arguments: tuple[str, ...] = ()
        if variant == "Q-SEM":
            environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
            scorer_arguments = ("--scorer-checkpoint", str(scorer_checkpoint))
        for update in PILOT_UPDATES:
            output_path = (
                artifact_root
                / "training/pilot/evaluation/cal"
                / variant
                / f"update={update:04d}.json"
            )
            checkpoint = (
                external_root / "training" / variant / f"update={update:04d}.ckpt"
            )
            if not output_path.is_file():
                _run_stage_process(
                    (
                        sys.executable,
                        "-m",
                        "scripts.perception_gain_evaluation",
                        "--mode",
                        "evaluate",
                        "--variant",
                        variant,
                        "--update",
                        str(update),
                        "--role",
                        "CAL",
                        "--checkpoint",
                        str(checkpoint),
                        *scorer_arguments,
                        "--assets",
                        str(assets_path),
                        "--roles",
                        str(roles_path),
                        "--external-root",
                        str(external_root),
                        "--device",
                        "cuda:0",
                    ),
                    command_label=(
                        "python -m scripts.perception_gain_evaluation "
                        f"--mode evaluate --variant {variant} --update {update} "
                        "--role CAL "
                        f"--checkpoint external:training/{variant}/"
                        f"update={update:04d}.ckpt --device cuda:0"
                    ),
                    stage="perception_pilot",
                    device="cuda:0",
                    inputs=[
                        f"external:training/{variant}/update={update:04d}.ckpt",
                        "external:data_root",
                    ],
                    outputs=[
                        "repo:artifacts/perception_gain_v1/training/pilot/"
                        f"evaluation/cal/{variant}/update={update:04d}.json"
                    ],
                    execution_log=execution_log,
                    environment=environment,
                )
            evaluation_summaries[(variant, update)] = _read_json(output_path)

    result = validate_perception_pilot_stage(
        training_summaries=training_summaries,
        evaluation_summaries=evaluation_summaries,
        baseline_metrics=baseline_candidate["metrics"],
        external_root=external_root,
    )
    _atomic_json(artifact_root / "selection/PILOT_SELECTION.json", result)
    return result


def _run_perception_full_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    pilot = _read_json(artifact_root / "selection/PILOT_SELECTION.json")
    selection = pilot.get("selection")
    arms = (
        tuple(selection.get("full_training_arms", ()))
        if isinstance(selection, Mapping)
        else ()
    )
    if pilot.get("status") != "PASS" or not 1 <= len(arms) <= 3 or arms[0] != "C0":
        raise CampaignError("pilot selection is unavailable for full training")

    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    roles_path = artifact_root / "DATA_ROLES.json"
    scorer_checkpoint = external_root / "training/scorer/model/update=0500.ckpt"
    training_summaries: dict[str, Mapping[str, object]] = {}
    evaluation_summaries: dict[tuple[str, int], Mapping[str, object]] = {}

    for variant in arms:
        run_dir = external_root / "training" / variant
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            raise CampaignError(f"pilot training summary is unavailable: {variant}")
        summary = _read_json(summary_path)
        completed = summary.get("completed_global_step")
        if completed != FULL_ENDPOINT:
            if (
                summary.get("status") != "COMPLETE"
                or summary.get("variant") != variant
                or completed != PILOT_ENDPOINT
            ):
                raise CampaignError(f"full resume point differs: {variant}")
            resume = run_dir / "last.ckpt"
            if not resume.is_file():
                raise CampaignError(f"full resume checkpoint is unavailable: {variant}")
            command = (
                sys.executable,
                "-m",
                "scripts.train_perception_gain",
                "--variant",
                variant,
                "--assets",
                str(assets_path),
                "--roles",
                str(roles_path),
                "--external-root",
                str(external_root),
                "--stop-after-updates",
                str(FULL_ENDPOINT),
                "--resume",
                str(resume),
            )
            environment = None
            inputs = [
                f"external:training/{variant}/last.ckpt",
                "external:data_root",
            ]
            if variant == "Q-SEM":
                if not scorer_checkpoint.is_file():
                    raise CampaignError("Q-SEM scorer checkpoint is unavailable")
                environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
                inputs.append("external:training/scorer/model/update=0500.ckpt")
            _run_stage_process(
                command,
                command_label=(
                    "python -m scripts.train_perception_gain "
                    f"--variant {variant} --assets external:assets.local.json "
                    "--roles repo:artifacts/perception_gain_v1/DATA_ROLES.json "
                    f"--external-root external:. --stop-after-updates {FULL_ENDPOINT} "
                    f"--resume external:training/{variant}/last.ckpt"
                ),
                stage="perception_full",
                device="cuda:0,cuda:1",
                inputs=inputs,
                outputs=[f"external:training/{variant}/run_summary.json"],
                execution_log=execution_log,
                environment=environment,
            )
            summary = _read_json(summary_path)
        training_summaries[variant] = summary

    for variant in arms:
        environment = None
        scorer_arguments: tuple[str, ...] = ()
        if variant == "Q-SEM":
            environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
            scorer_arguments = ("--scorer-checkpoint", str(scorer_checkpoint))
        for update in FULL_UPDATES:
            pilot_path = (
                artifact_root
                / "training/pilot/evaluation/cal"
                / variant
                / f"update={update:04d}.json"
            )
            full_path = (
                artifact_root
                / "training/full/evaluation/cal"
                / variant
                / f"update={update:04d}.json"
            )
            output_path = pilot_path if pilot_path.is_file() else full_path
            if not output_path.is_file():
                checkpoint_arguments: tuple[str, ...] = ()
                inputs = ["external:data_root"]
                if update:
                    checkpoint = (
                        external_root
                        / "training"
                        / variant
                        / f"update={update:04d}.ckpt"
                    )
                    checkpoint_arguments = ("--checkpoint", str(checkpoint))
                    inputs.append(
                        f"external:training/{variant}/update={update:04d}.ckpt"
                    )
                else:
                    inputs.append("external:r1_checkpoint")
                _run_stage_process(
                    (
                        sys.executable,
                        "-m",
                        "scripts.perception_gain_evaluation",
                        "--mode",
                        "evaluate",
                        "--variant",
                        variant,
                        "--update",
                        str(update),
                        "--role",
                        "CAL",
                        *checkpoint_arguments,
                        *scorer_arguments,
                        "--assets",
                        str(assets_path),
                        "--roles",
                        str(roles_path),
                        "--external-root",
                        str(external_root),
                        "--output",
                        str(output_path),
                        "--device",
                        "cuda:0",
                    ),
                    command_label=(
                        "python -m scripts.perception_gain_evaluation "
                        f"--mode evaluate --variant {variant} --update {update} "
                        "--role CAL --output repo:artifacts/perception_gain_v1/"
                        f"training/full/evaluation/cal/{variant}/"
                        f"update={update:04d}.json --device cuda:0"
                    ),
                    stage="perception_full",
                    device="cuda:0",
                    inputs=inputs,
                    outputs=[
                        "repo:artifacts/perception_gain_v1/training/full/"
                        f"evaluation/cal/{variant}/update={update:04d}.json"
                    ],
                    execution_log=execution_log,
                    environment=environment,
                )
            evaluation_summaries[(variant, update)] = _read_json(output_path)

    baseline = evaluation_summaries[("C0", 0)].get("candidate")
    if not isinstance(baseline, Mapping) or not isinstance(
        baseline.get("metrics"), Mapping
    ):
        raise CampaignError("full CAL baseline differs")
    result = validate_perception_full_stage(
        completed_full_arms=arms,
        training_summaries=training_summaries,
        evaluation_summaries=evaluation_summaries,
        baseline_metrics=baseline["metrics"],
        external_root=external_root,
    )
    _atomic_json(artifact_root / "selection/CAL_CHECKPOINT_SELECTION.json", result)
    return result


def _run_perception_select_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import write_immutable_lock

    cal_selection = _read_json(
        artifact_root / "selection/CAL_CHECKPOINT_SELECTION.json"
    )
    selected = cal_selection.get("cal_checkpoint_selection")
    arms = cal_selection.get("completed_full_arms")
    if (
        cal_selection.get("status") != "PASS"
        or not isinstance(selected, Mapping)
        or isinstance(arms, (str, bytes))
        or not isinstance(arms, Sequence)
        or set(selected) != set(arms)
    ):
        raise CampaignError("CAL checkpoint selection is unavailable")

    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    roles_path = artifact_root / "DATA_ROLES.json"
    scorer_checkpoint = external_root / "training/scorer/model/update=0500.ckpt"
    requests = [("R1", "C0", 0)]
    for variant in arms:
        candidate = selected[variant]
        if not isinstance(candidate, Mapping):
            raise CampaignError(f"CAL checkpoint selection differs: {variant}")
        requests.append((variant, variant, candidate.get("optimizer_update")))

    evaluation_summaries: dict[str, Mapping[str, object]] = {}
    completed: dict[tuple[str, int], Mapping[str, object]] = {}
    for label, variant, update in requests:
        if not isinstance(update, int):
            raise CampaignError(f"CAL checkpoint update differs: {label}")
        identity = (variant, update)
        if identity in completed:
            evaluation_summaries[label] = completed[identity]
            continue
        output_path = (
            artifact_root
            / "selection/evaluation/sel"
            / label
            / f"update={update:04d}.json"
        )
        scorer_arguments: tuple[str, ...] = ()
        environment = None
        inputs = ["external:data_root"]
        if variant == "Q-SEM":
            if not scorer_checkpoint.is_file():
                raise CampaignError("Q-SEM scorer checkpoint is unavailable")
            scorer_arguments = ("--scorer-checkpoint", str(scorer_checkpoint))
            environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
            inputs.append("external:training/scorer/model/update=0500.ckpt")
        checkpoint_arguments: tuple[str, ...] = ()
        if update:
            candidate = selected[label]
            checkpoint = _external_artifact_path(
                external_root, candidate.get("checkpoint")
            )
            checkpoint_arguments = ("--checkpoint", str(checkpoint))
            inputs.append(str(candidate["checkpoint"]))
        else:
            inputs.append("external:r1_checkpoint")
        if not output_path.is_file():
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.perception_gain_evaluation",
                    "--mode",
                    "evaluate",
                    "--variant",
                    variant,
                    "--update",
                    str(update),
                    "--role",
                    "SEL",
                    *checkpoint_arguments,
                    *scorer_arguments,
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.perception_gain_evaluation --mode evaluate "
                    f"--variant {variant} --update {update} --role SEL "
                    "--output repo:artifacts/perception_gain_v1/selection/"
                    f"evaluation/sel/{label}/update={update:04d}.json --device cuda:0"
                ),
                stage="perception_select",
                device="cuda:0",
                inputs=inputs,
                outputs=[
                    "repo:artifacts/perception_gain_v1/selection/evaluation/sel/"
                    f"{label}/update={update:04d}.json"
                ],
                execution_log=execution_log,
                environment=environment,
            )
        summary = _read_json(output_path)
        completed[identity] = summary
        evaluation_summaries[label] = summary

    result = validate_perception_selection_stage(
        cal_selection=cal_selection,
        evaluation_summaries=evaluation_summaries,
    )
    _atomic_json(artifact_root / "selection/SEL_SELECTION.json", result)
    write_immutable_lock(
        artifact_root / "selection/PERCEPTION_LOCK.json", result["lock"]
    )
    return result


def _run_refinement_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import rank_candidates

    perception_lock = _read_json(artifact_root / "selection/PERCEPTION_LOCK.json")
    parent = perception_lock.get("parent_candidate")
    parent_cal = perception_lock.get("parent_cal_candidate")
    if not isinstance(parent, Mapping) or not isinstance(parent_cal, Mapping):
        raise CampaignError("perception parent is unavailable for refinement")
    parent_variant = parent.get("source_variant")
    parent_update = parent.get("optimizer_update")
    if parent_variant not in PILOT_VARIANTS or not isinstance(parent_update, int):
        raise CampaignError("perception parent identity differs")

    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    roles_path = artifact_root / "DATA_ROLES.json"
    scorer_checkpoint = external_root / "training/scorer/model/update=0500.ckpt"
    scorer_arguments: tuple[str, ...] = ()
    environment = None
    common_inputs = ["external:data_root"]
    if parent_variant == "Q-SEM":
        if not scorer_checkpoint.is_file():
            raise CampaignError("Q-SEM scorer checkpoint is unavailable")
        scorer_arguments = ("--scorer-checkpoint", str(scorer_checkpoint))
        environment = {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
        common_inputs.append("external:training/scorer/model/update=0500.ckpt")
    parent_checkpoint_arguments: tuple[str, ...] = ()
    if parent_update:
        parent_checkpoint = _external_artifact_path(
            external_root, parent.get("checkpoint")
        )
        parent_checkpoint_arguments = ("--parent-checkpoint", str(parent_checkpoint))
        preparation_checkpoint_arguments = ("--checkpoint", str(parent_checkpoint))
        common_inputs.append(str(parent["checkpoint"]))
    else:
        preparation_checkpoint_arguments = ()
        common_inputs.append("external:r1_checkpoint")

    data_manifest_path = artifact_root / "training/refiner/DATA_MANIFEST.json"
    if not data_manifest_path.is_file():
        _run_stage_process(
            (
                sys.executable,
                "-m",
                "scripts.prepare_perception_refiner",
                "--variant",
                parent_variant,
                "--update",
                str(parent_update),
                *preparation_checkpoint_arguments,
                *scorer_arguments,
                "--assets",
                str(assets_path),
                "--roles",
                str(roles_path),
                "--external-root",
                str(external_root),
                "--device",
                "cuda:0",
            ),
            command_label=(
                "python -m scripts.prepare_perception_refiner "
                f"--variant {parent_variant} --update {parent_update} "
                "--assets external:assets.local.json --roles "
                "repo:artifacts/perception_gain_v1/DATA_ROLES.json "
                "--external-root external:. --device cuda:0"
            ),
            stage="refinement",
            device="cuda:0",
            inputs=common_inputs,
            outputs=[
                "repo:artifacts/perception_gain_v1/training/refiner/DATA_MANIFEST.json",
                "external:training/refiner/data/",
            ],
            execution_log=execution_log,
            environment=environment,
        )
    data_manifest = _read_json(data_manifest_path)
    shard_records = data_manifest.get("shards")
    if isinstance(shard_records, (str, bytes)) or not isinstance(
        shard_records, Sequence
    ):
        raise CampaignError("refiner data shards are unavailable")
    cache_paths = [
        _external_artifact_path(external_root, record.get("external_reference"))
        for record in shard_records
        if isinstance(record, Mapping)
    ]
    if not cache_paths:
        raise CampaignError("refiner training cache is unavailable")

    model_root = external_root / "training/refiner/model"
    training_summary_path = model_root / "run_summary.json"
    training_summary = (
        _read_json(training_summary_path) if training_summary_path.is_file() else None
    )
    if (
        not isinstance(training_summary, Mapping)
        or training_summary.get("completed_updates") != 1500
    ):
        resume = model_root / "last.ckpt"
        resume_arguments = ("--resume", str(resume)) if resume.is_file() else ()
        _run_stage_process(
            (
                sys.executable,
                "-m",
                "scripts.train_perception_refiner",
                "--cache",
                *(str(path) for path in cache_paths),
                "--output",
                str(model_root),
                "--stop-after-updates",
                "1500",
                *resume_arguments,
                "--device",
                "cuda:0",
            ),
            command_label=(
                "python -m scripts.train_perception_refiner --cache "
                "external:training/refiner/data/soft-*.pt --output "
                "external:training/refiner/model --stop-after-updates 1500 "
                + (
                    "--resume external:training/refiner/model/last.ckpt "
                    if resume_arguments
                    else ""
                )
                + "--device cuda:0"
            ),
            stage="refinement",
            device="cuda:0",
            inputs=[
                *(str(record["external_reference"]) for record in shard_records),
            ],
            outputs=["external:training/refiner/model/run_summary.json"],
            execution_log=execution_log,
        )
        training_summary = _read_json(training_summary_path)

    cal_evaluations: dict[int, Mapping[str, object]] = {}
    for update in REFINER_UPDATES:
        refiner_checkpoint = model_root / f"update={update:04d}.ckpt"
        output_path = (
            artifact_root
            / "training/refiner/evaluation/cal"
            / f"update={update:04d}.json"
        )
        if not output_path.is_file():
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.perception_refiner_evaluation",
                    "--parent-variant",
                    parent_variant,
                    "--parent-update",
                    str(parent_update),
                    *parent_checkpoint_arguments,
                    *scorer_arguments,
                    "--refiner-update",
                    str(update),
                    "--refiner-checkpoint",
                    str(refiner_checkpoint),
                    "--role",
                    "CAL",
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.perception_refiner_evaluation "
                    f"--parent-variant {parent_variant} --parent-update {parent_update} "
                    f"--refiner-update {update} --refiner-checkpoint external:training/"
                    f"refiner/model/update={update:04d}.ckpt --role CAL --device cuda:0"
                ),
                stage="refinement",
                device="cuda:0",
                inputs=[
                    *common_inputs,
                    f"external:training/refiner/model/update={update:04d}.ckpt",
                ],
                outputs=[
                    "repo:artifacts/perception_gain_v1/training/refiner/"
                    f"evaluation/cal/update={update:04d}.json"
                ],
                execution_log=execution_log,
                environment=environment,
            )
        cal_evaluations[update] = _read_json(output_path)

    if not isinstance(parent_cal.get("metrics"), Mapping):
        raise CampaignError("perception parent CAL metrics are unavailable")
    ranked = rank_candidates(
        [summary["candidate"] for summary in cal_evaluations.values()],
        baseline_metrics=parent_cal["metrics"],
    )
    selected_update = int(ranked[0]["optimizer_update"])
    selected_refiner_checkpoint = model_root / f"update={selected_update:04d}.ckpt"
    sel_output = (
        artifact_root
        / "training/refiner/evaluation/sel"
        / f"update={selected_update:04d}.json"
    )
    if not sel_output.is_file():
        _run_stage_process(
            (
                sys.executable,
                "-m",
                "scripts.perception_refiner_evaluation",
                "--parent-variant",
                parent_variant,
                "--parent-update",
                str(parent_update),
                *parent_checkpoint_arguments,
                *scorer_arguments,
                "--refiner-update",
                str(selected_update),
                "--refiner-checkpoint",
                str(selected_refiner_checkpoint),
                "--role",
                "SEL",
                "--assets",
                str(assets_path),
                "--roles",
                str(roles_path),
                "--external-root",
                str(external_root),
                "--output",
                str(sel_output),
                "--device",
                "cuda:0",
            ),
            command_label=(
                "python -m scripts.perception_refiner_evaluation "
                f"--parent-variant {parent_variant} --parent-update {parent_update} "
                f"--refiner-update {selected_update} --refiner-checkpoint "
                "external:training/refiner/model/"
                f"update={selected_update:04d}.ckpt --role SEL --device cuda:0"
            ),
            stage="refinement",
            device="cuda:0",
            inputs=[
                *common_inputs,
                "external:training/refiner/model/" f"update={selected_update:04d}.ckpt",
            ],
            outputs=[
                "repo:artifacts/perception_gain_v1/training/refiner/"
                f"evaluation/sel/update={selected_update:04d}.json"
            ],
            execution_log=execution_log,
            environment=environment,
        )
    sel_evaluation = _read_json(sel_output)
    result = validate_refinement_stage(
        perception_lock=perception_lock,
        data_manifest=data_manifest,
        training_summary=training_summary,
        cal_evaluations=cal_evaluations,
        sel_evaluation=sel_evaluation,
        external_root=external_root,
    )
    _atomic_json(artifact_root / "selection/REFINER_SELECTION.json", result)
    return result


def _run_final_lock_stage(
    *, artifact_root: Path, config_path: Path, config: Mapping[str, object]
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import write_immutable_lock

    perception_lock = _read_json(artifact_root / "selection/PERCEPTION_LOCK.json")
    refinement = _read_json(artifact_root / "selection/REFINER_SELECTION.json")
    identity = config.get("identity")
    if not isinstance(identity, Mapping):
        raise CampaignError("campaign identity is unavailable for final lock")
    lock = build_final_lock(
        perception_lock=perception_lock,
        refinement=refinement,
        campaign_identity=identity,
        campaign_config_sha256=_sha256(config_path),
        data_roles_sha256=_sha256(artifact_root / "DATA_ROLES.json"),
    )
    write_immutable_lock(artifact_root / "selection/FINAL_LOCK.json", lock)
    return {"status": "PASS", "lock": lock}


def _run_replication_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
    state: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    from scripts.perception_gain_evaluation import compare_candidate

    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json")
    recipe = lock.get("recipe")
    budget = state.get("budget_used")
    limits = config.get("budget")
    if (
        not isinstance(recipe, Mapping)
        or not isinstance(budget, Mapping)
        or not isinstance(limits, Mapping)
    ):
        raise CampaignError("replication inputs are unavailable")

    def training_cost(variant: str, endpoint: int) -> float:
        cost = 0.0
        pilot = state.get("perception_pilot")
        full = state.get("perception_full")
        if isinstance(pilot, Mapping):
            rows = pilot.get("training")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                match = next(
                    (
                        row
                        for row in rows
                        if isinstance(row, Mapping) and row.get("variant") == variant
                    ),
                    None,
                )
                if isinstance(match, Mapping):
                    cost += (
                        float(match.get("gpu_hours", 0.0)) * min(endpoint, 750) / 750
                    )
        if endpoint > 750 and isinstance(full, Mapping):
            rows = full.get("training")
            if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                match = next(
                    (
                        row
                        for row in rows
                        if isinstance(row, Mapping) and row.get("variant") == variant
                    ),
                    None,
                )
                if isinstance(match, Mapping):
                    cost += float(match.get("gpu_hours", 0.0)) * (endpoint - 750) / 2250
        return cost

    parent_variant = str(recipe.get("parent_variant"))
    parent_update = int(recipe.get("parent_optimizer_update", -1))
    replication_variants = (
        tuple(dict.fromkeys(("C0", parent_variant))) if parent_update > 0 else ()
    )
    estimated_perception = sum(
        training_cost(variant, parent_update) for variant in replication_variants
    )
    refinement = state.get("refinement")
    estimated_refinement = (
        float(refinement.get("refinement_gpu_hours", 0.0))
        if bool(recipe.get("refiner_enabled")) and isinstance(refinement, Mapping)
        else 0.0
    )
    remaining_perception = max(
        0.0,
        float(limits["perception_training_gpu_hours"])
        - float(budget.get("perception_training_gpu_hours", 0.0)),
    )
    remaining_refinement = max(
        0.0,
        float(limits["refinement_gpu_hours"])
        - float(budget.get("refinement_gpu_hours", 0.0)),
    )
    decision = build_replication_decision(
        final_lock=lock,
        estimated_perception_gpu_hours=estimated_perception,
        estimated_refinement_gpu_hours=estimated_refinement,
        remaining_perception_gpu_hours=remaining_perception,
        remaining_refinement_gpu_hours=remaining_refinement,
    )
    output_path = artifact_root / "confirmation/REPLICATION_SUMMARY.json"
    if decision["decision"] == "SKIP":
        result = {
            "schema_version": "perception-gain-replication-v1",
            "status": "PASS",
            **decision,
            "components": [],
            "perception_gpu_hours": 0.0,
            "refinement_gpu_hours": 0.0,
        }
        _atomic_json(output_path, result)
        return result

    roles_path = artifact_root / "DATA_ROLES.json"
    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    scorer_checkpoint = external_root / "training/scorer/model/update=0500.ckpt"
    summaries: dict[str, Mapping[str, object]] = {}
    evaluations: dict[str, Mapping[str, object]] = {}
    for variant in replication_variants:
        run_subdir = Path("replication/seed46") / variant
        run_dir = external_root / "training" / run_subdir
        summary_path = run_dir / "run_summary.json"
        if (
            not summary_path.is_file()
            or _read_json(summary_path).get("completed_global_step") != parent_update
        ):
            environment = (
                {"PERSIST4D_Q_SEM_SCORER": str(scorer_checkpoint)}
                if variant == "Q-SEM"
                else None
            )
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.train_perception_gain",
                    "--variant",
                    variant,
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--train-seed",
                    "46",
                    "--run-subdir",
                    str(run_subdir),
                    "--stop-after-updates",
                    str(parent_update),
                ),
                command_label=(
                    "python -m scripts.train_perception_gain "
                    f"--variant {variant} --train-seed 46 --run-subdir "
                    f"{run_subdir} --stop-after-updates {parent_update}"
                ),
                stage="replication",
                device="cuda:0,cuda:1",
                inputs=["external:r1_checkpoint", "external:data_root"],
                outputs=[f"external:training/{run_subdir}/run_summary.json"],
                execution_log=execution_log,
                environment=environment,
            )
        summary = _read_json(summary_path)
        checkpoint = run_dir / f"update={parent_update:04d}.ckpt"
        evaluation_path = (
            artifact_root
            / "confirmation/replication/seed46/sel"
            / f"{variant}-update={parent_update:04d}.json"
        )
        if not evaluation_path.is_file():
            scorer_arguments = (
                ("--scorer-checkpoint", str(scorer_checkpoint))
                if variant == "Q-SEM"
                else ()
            )
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.perception_gain_evaluation",
                    "--mode",
                    "evaluate",
                    "--variant",
                    variant,
                    "--update",
                    str(parent_update),
                    "--role",
                    "SEL",
                    "--checkpoint",
                    str(checkpoint),
                    *scorer_arguments,
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output",
                    str(evaluation_path),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.perception_gain_evaluation --mode evaluate "
                    f"--variant {variant} --update {parent_update} --role SEL "
                    "--checkpoint external:training/replication/seed46/..."
                ),
                stage="replication",
                device="cuda:0",
                inputs=[f"external:training/{run_subdir}/{checkpoint.name}"],
                outputs=[str(evaluation_path.relative_to(PROJECT_ROOT))],
                execution_log=execution_log,
            )
        summaries[variant] = summary
        evaluations[variant] = _read_json(evaluation_path)

    components = []
    perception_gpu_hours = sum(
        float(value["gpu_hours"]) for value in summaries.values()
    )
    if parent_update > 0:
        parent_candidate = evaluations[parent_variant].get("candidate")
        if not isinstance(parent_candidate, Mapping):
            raise CampaignError("seed46 parent evaluation differs")
        if parent_variant == "C0":
            baseline = lock["perception_lock"].get("d0_r1_baseline")
            paired_with = "R1"
        else:
            baseline = evaluations["C0"].get("candidate")
            paired_with = "C0-seed46"
        if not isinstance(baseline, Mapping) or not isinstance(
            baseline.get("metrics"), Mapping
        ):
            raise CampaignError("seed46 perception baseline differs")
        comparison = compare_candidate(
            parent_candidate, baseline_metrics=baseline["metrics"]
        )
        supported = comparison["S_mean"] > 0.0 and comparison["S_min"] >= -0.002
        components.append(
            {
                "component": "perception",
                "status": "SUPPORTED" if supported else "MIXED",
                "paired_with": paired_with,
                "optimizer_update": parent_update,
                "comparison": comparison,
            }
        )

    refinement_gpu_hours = 0.0
    if bool(recipe.get("refiner_enabled")):
        parent_checkpoint = (
            external_root
            / "training/replication/seed46"
            / parent_variant
            / f"update={parent_update:04d}.ckpt"
            if parent_update
            else None
        )
        data_root = external_root / "training/replication/seed46/refiner/data"
        data_manifest_path = (
            artifact_root / "confirmation/replication/seed46/refiner/DATA_MANIFEST.json"
        )
        if not data_manifest_path.is_file():
            parent_checkpoint_arguments = (
                ("--checkpoint", str(parent_checkpoint)) if parent_checkpoint else ()
            )
            scorer_arguments = (
                ("--scorer-checkpoint", str(scorer_checkpoint))
                if parent_variant == "Q-SEM"
                else ()
            )
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.prepare_perception_refiner",
                    "--variant",
                    parent_variant,
                    "--update",
                    str(parent_update),
                    *parent_checkpoint_arguments,
                    *scorer_arguments,
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output-root",
                    str(data_root),
                    "--manifest-output",
                    str(data_manifest_path),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.prepare_perception_refiner "
                    f"--variant {parent_variant} --update {parent_update} "
                    "--output-root external:training/replication/seed46/refiner/data"
                ),
                stage="replication",
                device="cuda:0",
                inputs=["external:data_root"],
                outputs=[str(data_manifest_path.relative_to(PROJECT_ROOT))],
                execution_log=execution_log,
            )
        data_manifest = _read_json(data_manifest_path)
        shards = data_manifest.get("shards")
        if isinstance(shards, (str, bytes)) or not isinstance(shards, Sequence):
            raise CampaignError("seed46 refiner shards differ")
        cache_paths = tuple(
            _external_artifact_path(external_root, row.get("external_reference"))
            for row in shards
            if isinstance(row, Mapping)
        )
        model_root = external_root / "training/replication/seed46/refiner/model"
        refiner_update = int(recipe["refiner_optimizer_update"])
        refiner_summary_path = model_root / "run_summary.json"
        if not refiner_summary_path.is_file():
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.train_perception_refiner",
                    "--cache",
                    *(str(path) for path in cache_paths),
                    "--output",
                    str(model_root),
                    "--stop-after-updates",
                    str(refiner_update),
                    "--seed",
                    "46",
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.train_perception_refiner --seed 46 "
                    f"--stop-after-updates {refiner_update}"
                ),
                stage="replication",
                device="cuda:0",
                inputs=[str(row.get("external_reference")) for row in shards],
                outputs=[
                    "external:training/replication/seed46/refiner/model/run_summary.json"
                ],
                execution_log=execution_log,
            )
        refiner_summary = _read_json(refiner_summary_path)
        refiner_eval_path = (
            artifact_root
            / "confirmation/replication/seed46/refiner/SEL_EVALUATION.json"
        )
        if not refiner_eval_path.is_file():
            parent_arguments = (
                ("--parent-checkpoint", str(parent_checkpoint))
                if parent_checkpoint
                else ()
            )
            scorer_arguments = (
                ("--scorer-checkpoint", str(scorer_checkpoint))
                if parent_variant == "Q-SEM"
                else ()
            )
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.perception_refiner_evaluation",
                    "--parent-variant",
                    parent_variant,
                    "--parent-update",
                    str(parent_update),
                    *parent_arguments,
                    *scorer_arguments,
                    "--refiner-update",
                    str(refiner_update),
                    "--refiner-checkpoint",
                    str(model_root / f"update={refiner_update:04d}.ckpt"),
                    "--role",
                    "SEL",
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output",
                    str(refiner_eval_path),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.perception_refiner_evaluation "
                    f"--parent-variant {parent_variant} --parent-update {parent_update} "
                    f"--refiner-update {refiner_update} --role SEL"
                ),
                stage="replication",
                device="cuda:0",
                inputs=["external:training/replication/seed46/refiner/model/..."],
                outputs=[str(refiner_eval_path.relative_to(PROJECT_ROOT))],
                execution_log=execution_log,
            )
        refiner_eval = _read_json(refiner_eval_path)
        candidate = refiner_eval.get("candidate")
        parent_metrics = refiner_eval.get("parent_metrics")
        if not isinstance(candidate, Mapping) or not isinstance(
            parent_metrics, Mapping
        ):
            raise CampaignError("seed46 refiner evaluation differs")
        comparison = compare_candidate(candidate, baseline_metrics=parent_metrics)
        supported = comparison["S_mean"] > 0.0 and comparison["S_min"] >= -0.002
        components.append(
            {
                "component": "refiner",
                "status": "SUPPORTED" if supported else "MIXED",
                "paired_with": "seed46-parent",
                "optimizer_update": refiner_update,
                "comparison": comparison,
            }
        )
        refinement_gpu_hours = float(data_manifest.get("gpu_hours", 0.0)) + float(
            refiner_summary.get("gpu_hours", 0.0)
        )

    replication_status = (
        "SUPPORTED"
        if components and all(row["status"] == "SUPPORTED" for row in components)
        else "MIXED"
    )
    result = {
        "schema_version": "perception-gain-replication-v1",
        "status": "PASS",
        **decision,
        "replication_status": replication_status,
        "components": components,
        "perception_gpu_hours": perception_gpu_hours,
        "refinement_gpu_hours": refinement_gpu_hours,
    }
    _atomic_json(output_path, result)
    return result


def _run_confirmation_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    lock = _read_json(artifact_root / "selection/FINAL_LOCK.json")
    recipe = lock.get("recipe")
    perception_lock = lock.get("perception_lock")
    if (
        lock.get("schema_version") != "perception-gain-final-lock-v1"
        or lock.get("status") != "LOCKED"
        or not isinstance(recipe, Mapping)
        or not isinstance(perception_lock, Mapping)
    ):
        raise CampaignError("FINAL_LOCK is unavailable for confirmation")
    c0_candidates = perception_lock.get("cal_checkpoint_selection")
    c0 = c0_candidates.get("C0") if isinstance(c0_candidates, Mapping) else None
    if not isinstance(c0, Mapping):
        raise CampaignError("C0-best checkpoint is unavailable for confirmation")
    roles_path = artifact_root / "DATA_ROLES.json"
    execution_log = artifact_root / "EXECUTION_LOG.jsonl"
    raw_root = artifact_root / "confirmation/raw"
    raw_cache: dict[tuple[object, ...], dict[str, object]] = {}

    def identity(
        variant: str, update: int, checkpoint: object, checkpoint_sha256: object
    ) -> dict[str, object]:
        if (
            variant not in PILOT_VARIANTS
            or update not in FULL_UPDATES
            or not isinstance(checkpoint, str)
            or not checkpoint
            or not _hex(checkpoint_sha256, 64)
        ):
            raise CampaignError("confirmation checkpoint identity differs")
        return {
            "variant": variant,
            "update": update,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
        }

    campaign_identity = lock.get("campaign_identity")
    if not isinstance(campaign_identity, Mapping):
        raise CampaignError("campaign identity is unavailable for confirmation")
    r1 = identity(
        "C0",
        0,
        "external:r1_checkpoint",
        campaign_identity.get("r1_checkpoint_sha256"),
    )
    c0_identity = identity(
        "C0",
        int(c0.get("optimizer_update", -1)),
        c0.get("checkpoint"),
        c0.get("checkpoint_sha256"),
    )
    parent = identity(
        str(recipe.get("parent_variant")),
        int(recipe.get("parent_optimizer_update", -1)),
        recipe.get("parent_checkpoint"),
        recipe.get("parent_checkpoint_sha256"),
    )

    def model_arguments(model: Mapping[str, object]) -> tuple[str, ...]:
        arguments: list[str] = []
        if int(model["update"]):
            arguments.extend(
                (
                    "--checkpoint",
                    str(_external_artifact_path(external_root, model["checkpoint"])),
                )
            )
        if model["variant"] == "Q-SEM":
            arguments.extend(
                (
                    "--scorer-checkpoint",
                    str(external_root / "training/scorer/model/update=0500.ckpt"),
                )
            )
        return tuple(arguments)

    def ensure_evaluation(
        *,
        kind: str,
        model: Mapping[str, object],
        role: str,
        extra: Sequence[str] = (),
    ) -> tuple[dict[str, object], str]:
        key = (
            kind,
            model["variant"],
            model["update"],
            model["checkpoint_sha256"],
            role,
            *extra,
        )
        physical_id = ":".join(str(value) for value in key)
        if key in raw_cache:
            return raw_cache[key], physical_id
        output = (
            raw_root
            / role.lower()
            / f"{hashlib.sha256(physical_id.encode()).hexdigest()[:16]}.json"
        )
        if kind == "native":
            module = "scripts.perception_gain_native_evaluation"
            command = (
                sys.executable,
                "-m",
                module,
                "--variant",
                str(model["variant"]),
                "--update",
                str(model["update"]),
                "--role",
                role,
                *model_arguments(model),
                *(extra or ()),
            )
        elif kind == "d0":
            module = "scripts.perception_gain_evaluation"
            command = (
                sys.executable,
                "-m",
                module,
                "--mode",
                "evaluate",
                "--variant",
                str(model["variant"]),
                "--update",
                str(model["update"]),
                "--role",
                role,
                *model_arguments(model),
            )
        elif kind == "local":
            module = "scripts.perception_gain_local_evaluation"
            command = (
                sys.executable,
                "-m",
                module,
                "--variant",
                str(model["variant"]),
                "--update",
                str(model["update"]),
                *model_arguments(model),
            )
        else:
            raise CampaignError("confirmation evaluation kind differs")
        if not output.is_file():
            shared = (
                "--assets",
                str(assets_path),
                "--external-root",
                str(external_root),
                "--output",
                str(output),
                "--device",
                "cuda:0",
            )
            if kind != "local":
                shared = (*shared, "--roles", str(roles_path))
            _run_stage_process(
                (*command, *shared),
                command_label=(
                    f"python -m {module} --variant {model['variant']} "
                    f"--update {model['update']} --role {role} --device cuda:0"
                ),
                stage="confirm",
                device="cuda:0",
                inputs=[str(model["checkpoint"]), "external:data_root"],
                outputs=[str(output.relative_to(PROJECT_ROOT))],
                execution_log=execution_log,
            )
        summary = _read_json(output)
        if (
            summary.get("variant") != model["variant"]
            or summary.get("optimizer_update") != model["update"]
            or summary.get("checkpoint_sha256") != model["checkpoint_sha256"]
        ):
            raise CampaignError("confirmation output checkpoint differs")
        raw_cache[key] = summary
        return summary, physical_id

    pb: dict[str, dict[str, object]] = {}
    for method_id, model in (
        ("FH-R1-native", r1),
        ("FH-P*-native", parent),
    ):
        raw, physical = ensure_evaluation(kind="native", model=model, role="PB")
        pb[method_id] = normalize_confirmation_evaluation(
            raw,
            method_id=method_id,
            population_id="PB-129",
            physical_evaluation_id=physical,
        )
    for method_id, model in (("D0-R1", r1), ("C0-best-D0", c0_identity)):
        raw, physical = ensure_evaluation(kind="d0", model=model, role="PB")
        pb[method_id] = normalize_confirmation_evaluation(
            raw,
            method_id=method_id,
            population_id="PB-129",
            physical_evaluation_id=physical,
        )

    refiner_enabled = bool(recipe.get("refiner_enabled"))
    refiner_update = int(recipe["refiner_optimizer_update"]) if refiner_enabled else 0
    refiner_reference = (
        recipe.get("refiner_checkpoint")
        if refiner_enabled
        else "external:training/refiner/model/update=0000.ckpt"
    )
    refiner_checkpoint = _external_artifact_path(external_root, refiner_reference)
    parent_arguments = tuple(
        value for value in model_arguments(parent) if value not in {"--checkpoint"}
    )
    if int(parent["update"]):
        parent_arguments = (
            "--parent-checkpoint",
            str(_external_artifact_path(external_root, parent["checkpoint"])),
            *(
                (
                    "--scorer-checkpoint",
                    str(external_root / "training/scorer/model/update=0500.ckpt"),
                )
                if parent["variant"] == "Q-SEM"
                else ()
            ),
        )

    def ensure_refiner(role: str) -> tuple[dict[str, object], str]:
        physical_id = (
            f"refiner:{parent['variant']}:{parent['update']}:"
            f"{parent['checkpoint_sha256']}:{refiner_update}:{role}"
        )
        output = (
            raw_root
            / role.lower()
            / f"{hashlib.sha256(physical_id.encode()).hexdigest()[:16]}.json"
        )
        if not output.is_file():
            _run_stage_process(
                (
                    sys.executable,
                    "-m",
                    "scripts.perception_refiner_evaluation",
                    "--parent-variant",
                    str(parent["variant"]),
                    "--parent-update",
                    str(parent["update"]),
                    *parent_arguments,
                    "--refiner-update",
                    str(refiner_update),
                    "--refiner-checkpoint",
                    str(refiner_checkpoint),
                    "--role",
                    role,
                    "--assets",
                    str(assets_path),
                    "--roles",
                    str(roles_path),
                    "--external-root",
                    str(external_root),
                    "--output",
                    str(output),
                    "--device",
                    "cuda:0",
                ),
                command_label=(
                    "python -m scripts.perception_refiner_evaluation "
                    f"--parent-variant {parent['variant']} --parent-update "
                    f"{parent['update']} --refiner-update {refiner_update} "
                    f"--role {role} --device cuda:0"
                ),
                stage="confirm",
                device="cuda:0",
                inputs=[str(parent["checkpoint"]), str(refiner_reference)],
                outputs=[str(output.relative_to(PROJECT_ROOT))],
                execution_log=execution_log,
            )
        summary = _read_json(output)
        evaluated_parent = summary.get("parent")
        evaluated_refiner = summary.get("refiner")
        if (
            not isinstance(evaluated_parent, Mapping)
            or evaluated_parent.get("checkpoint_sha256") != parent["checkpoint_sha256"]
            or not isinstance(evaluated_refiner, Mapping)
            or evaluated_refiner.get("updates") != refiner_update
        ):
            raise CampaignError("confirmation refiner identity differs")
        return summary, physical_id

    pb_refiner, pb_refiner_physical = ensure_refiner("PB")
    pb["P*-D0"] = normalize_confirmation_evaluation(
        pb_refiner,
        method_id="P*-D0",
        population_id="PB-129",
        physical_evaluation_id=pb_refiner_physical,
        source_method="PARENT",
    )
    if refiner_enabled:
        pb["FINAL"] = normalize_confirmation_evaluation(
            pb_refiner,
            method_id="FINAL",
            population_id="PB-129",
            physical_evaluation_id=pb_refiner_physical,
            source_method="R-REFINE",
        )
    else:
        pb["FINAL"] = {**pb["P*-D0"], "status": "ALIASED", "method_id": "FINAL"}

    local_t2 = {}
    for method_id, model in (
        ("R1-native", r1),
        ("C0-native", c0_identity),
        ("P*-native", parent),
    ):
        raw, physical = ensure_evaluation(kind="local", model=model, role="LOCAL-T2")
        local_t2[method_id] = normalize_confirmation_evaluation(
            raw,
            method_id=method_id,
            population_id="LOCAL-T2-154",
            physical_evaluation_id=physical,
        )

    additional = {}
    additional_r1, additional_r1_physical = ensure_evaluation(
        kind="native",
        model=r1,
        role="ADDITIONAL",
        extra=("--horizons", "2", "3", "4"),
    )
    additional["FH-R1-native"] = normalize_confirmation_evaluation(
        additional_r1,
        method_id="FH-R1-native",
        population_id="ADDITIONAL-native-terminal",
        physical_evaluation_id=additional_r1_physical,
    )
    additional_refiner, additional_refiner_physical = ensure_refiner("ADDITIONAL")
    additional["P*-D0"] = normalize_confirmation_evaluation(
        additional_refiner,
        method_id="P*-D0",
        population_id="ADDITIONAL-native-terminal",
        physical_evaluation_id=additional_refiner_physical,
        source_method="PARENT",
    )
    if refiner_enabled:
        additional["FINAL"] = normalize_confirmation_evaluation(
            additional_refiner,
            method_id="FINAL",
            population_id="ADDITIONAL-native-terminal",
            physical_evaluation_id=additional_refiner_physical,
            source_method="R-REFINE",
        )
    else:
        additional["FINAL"] = {
            **additional["P*-D0"],
            "status": "ALIASED",
            "method_id": "FINAL",
        }
    result = build_confirmation_summary(pb=pb, local_t2=local_t2, additional=additional)
    _atomic_json(artifact_root / "confirmation/PB_METHODS.json", {"methods": pb})
    _atomic_json(
        artifact_root / "confirmation/LOCAL_T2_METHODS.json",
        {"methods": local_t2},
    )
    _atomic_json(
        artifact_root / "confirmation/ADDITIONAL_METHODS.json",
        {"methods": additional},
    )
    _atomic_json(artifact_root / "confirmation/CONFIRMATION_SUMMARY.json", result)
    return result


def _run_profile_stage(
    *,
    artifact_root: Path,
    assets_path: Path,
    external_root: Path,
) -> dict[str, object]:
    output = artifact_root / "resources/PROFILE_SUMMARY.json"
    _run_stage_process(
        [
            sys.executable,
            "-m",
            "scripts.perception_gain_profile",
            "--final-lock",
            str(artifact_root / "selection/FINAL_LOCK.json"),
            "--assets",
            str(assets_path),
            "--external-root",
            str(external_root),
            "--device",
            "cuda:0",
            "--warmup",
            "1",
            "--repeats",
            "3",
            "--output",
            str(artifact_root / "resources"),
        ],
        command_label="live PB profile: 1 full-sequence warmup + 3 measured repeats",
        stage="profile",
        device="cuda:0",
        inputs=[
            "selection/FINAL_LOCK.json",
            "external:assets.local.json",
            "artifacts/P6A/protocol_b_manifest.json",
        ],
        outputs=[
            "resources/measurements.csv",
            "resources/summary.csv",
            "resources/PROFILE_SUMMARY.json",
        ],
        execution_log=artifact_root / "EXECUTION_LOG.jsonl",
    )
    result = _read_json(output)
    if (
        result.get("schema_version") != "perception-gain-profile-v1"
        or result.get("status") != "PASS"
        or result.get("warmup_full_sequences") != 1
        or result.get("measured_full_sequences") != 3
        or result.get("horizons") != [2, 3, 4, 5]
        or result.get("measurement_rows") != len(result.get("methods", [])) * 6 * 4 * 3
    ):
        raise CampaignError("profile output differs from the frozen contract")
    return result


def _run_report_stage(
    *,
    artifact_root: Path,
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_reporting import generate_delivery

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return generate_delivery(
        project_root=PROJECT_ROOT,
        artifact_root=artifact_root,
        external_root=external_root,
        experiment_commit=source_commit,
        publication_phase="PREPARE",
    )


def _run_publish_stage(
    *,
    artifact_root: Path,
    external_root: Path,
) -> dict[str, object]:
    from scripts.perception_gain_publish import publish

    return publish(artifact_root=artifact_root, external_root=external_root)


def _external_root(value: str | None) -> Path:
    configured = value or os.environ.get("PERSIST4D_PERCEPTION_RUN_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.home() / "persist4d_runs/perception_gain_v1").resolve()
    )


def _verified_input(
    path_value: str | None,
    *,
    logical_reference: str,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, object]:
    path = Path(path_value).expanduser().resolve() if path_value else None
    if path is None or not path.exists():
        raise CampaignError(f"required input is unavailable: {logical_reference}")
    result: dict[str, object] = {
        "logical_reference": logical_reference,
        "kind": "directory" if path.is_dir() else "file",
        "status": "VERIFIED",
    }
    if path.is_file():
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256(path)
        if expected_bytes is not None and observed_bytes != expected_bytes:
            raise CampaignError(f"input byte count differs: {logical_reference}")
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise CampaignError(f"input SHA256 differs: {logical_reference}")
        result.update({"bytes": observed_bytes, "sha256": observed_sha256})
    return result


def _code_bindings() -> str:
    return """# Perception Gain V1 Code Bindings

| Existing path | Symbol | Current role | Perception Gain change / experiment |
|---|---|---|---|
| `models/criterion.py` | `SetCriterion.loss_masks`, `SetCriterion.forward` | matched mask BCE/Dice and aux loss | opt-in stage aggregation for S-BAL/S-WORST |
| `models/matcher.py` | `HungarianMatcher.forward` | fixed one-to-one matching | unchanged; regression checked |
| `trainer/trainer.py` | `_configured_objective_loss`, `InstanceSegmentation` | raw-sum objective/model setup | narrow perception trainer reuse |
| `conf/config_rescene4d_concerto_rootcause.yaml` | root R1 config | R1 data/model semantics | resolved values frozen in RUN_CONFIG |
| `models/rescene.py` | `initialize_queries`, `aggregate_features`, `forward`, `attn_mask` | query/decoder path | Q-SEM and A-OPEN opt-in paths |
| `datasets/task_memory_episode.py` | `StageMeta`, `TaskMemoryEpisodeCollator` | causal stage/vertex mapping | reused by pair/scorer/refiner data |
| `scripts/crosswindow_cache.py` | `resolve_assets`, `align_mask`, `build_canonical_frame` | asset and canonical alignment | reused without old-output overwrite |
| `scripts/diagnose_crosswindow_failures.py` | `_published_match_by_gt`, `diagnose_candidate_coverage` | E1 diagnostic | tau-specific official trace |
| `scripts/rescene_task_postprocess.py` | `extract_official_task_prediction` | official bool prediction | optional real soft sidecar |
| `scripts/task_memory_output.py` | lag-one publishers | D0 identity/output | fixed D0 path; optional refiner insertion |
| `scripts/replay_crosswindow_association.py` | CrossWindow replay | D0 association | unchanged association contract |
| `scripts/run_task_memory_controls.py` | R1 prediction producer | cached/live D0 source | checkpoint/config parameterization |
| `scripts/system_comparison_inference.py` | native FH inference | full-prefix baseline | live checkpoint runner reuse |
| `scripts/p6a_metrics.py` | `OfficialMetricAccumulator` | official local/temporal metrics | optional match trace only |
| `scripts/system_comparison_metrics.py` | `CausalTaskAccumulator` | causal aggregate metrics | reused for all formal scores |
| `scripts/train_task_memory.py` | checkpoint/asset helpers | prior fixed training runner | helpers reused, variants unchanged |
| `trainer/task_memory_trainer.py` | episode schedule/resume helpers | prior state trainer | metadata/scheduler patterns reused |
| `scripts/profile_task_memory.py` | live timing primitives | measured resource path | adapted for locked final methods |

Line locations are recorded against fixed parent `6ef77620aa20926311eff3124a794a6ca2e32727`; final changed line locations are refreshed during report generation.
"""


def _bootstrap(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    artifact_root = PROJECT_ROOT / config["paths"]["artifact_root"]
    instruction = PROJECT_ROOT / config["paths"]["instruction"]
    data_contract = PROJECT_ROOT / config["paths"]["task_data_contract"]
    if _sha256(instruction) != config["identity"]["instruction_sha256"]:
        raise CampaignError("execution instruction SHA differs")
    if not data_contract.is_file():
        raise CampaignError("task data contract is unavailable")
    external_root = _external_root(args.external_root)
    external_root.mkdir(parents=True, exist_ok=True)
    locator_path = (
        Path(args.assets_from).expanduser().resolve()
        if getattr(args, "assets_from", None)
        else PROJECT_ROOT / config["paths"]["task_asset_locator"]
    )
    fallback = _read_json(locator_path) if locator_path.is_file() else {}
    explicit = {
        key: getattr(args, key)
        for key in ASSET_KEYS
        if key != "external_run_root" and getattr(args, key, None) is not None
    }
    resolution = resolve_assets(
        explicit=explicit,
        environ=os.environ,
        fallback=fallback,
    )
    assets = dict(resolution.values)
    sources = dict(resolution.sources)
    required_assets = (
        "r1_checkpoint",
        "concerto_pretrained",
        "data_root",
        "rio_metadata",
        "metric_dataset_spec",
    )
    if any(assets.get(name) is None for name in required_assets):
        missing = [name for name in required_assets if assets.get(name) is None]
        raise CampaignError(f"required assets are unresolved: {missing}")
    input_records = {
        "r1_checkpoint": _verified_input(
            assets["r1_checkpoint"],
            logical_reference="external:r1_checkpoint",
            expected_sha256=config["identity"]["r1_checkpoint_sha256"],
            expected_bytes=int(config["identity"]["r1_checkpoint_bytes"]),
        ),
        "concerto_pretrained": _verified_input(
            assets["concerto_pretrained"],
            logical_reference="external:concerto_pretrained",
            expected_sha256=config["identity"]["concerto_sha256"],
        ),
        "data_root": _verified_input(
            assets["data_root"], logical_reference="external:data_root"
        ),
        "rio_metadata": _verified_input(
            assets["rio_metadata"], logical_reference="external:rio_metadata"
        ),
        "metric_dataset_spec": _verified_input(
            assets["metric_dataset_spec"],
            logical_reference="external:metric_dataset_spec",
        ),
        "task_data_contract": _verified_input(
            str(data_contract), logical_reference=config["paths"]["task_data_contract"]
        ),
    }
    cross_roles_path = PROJECT_ROOT / config["paths"]["crosswindow_data_roles"]
    cross_roles = _read_json(cross_roles_path)
    task_contract = _read_json(data_contract)
    additional = task_contract.get("roles", {}).get(
        "additional_native_reference_ids", []
    )
    roles = build_data_roles(
        task_contract,
        cross_roles,
        local_t2_reference_ids=additional,
    )
    _atomic_json(
        artifact_root / "DATA_ROLES.json",
        {
            "schema_version": "perception-gain-data-roles-v1",
            "roles": roles,
            "physical_reference_counts": {
                name: len(values)
                for name, values in roles.items()
                if name != "removed_train_overlap"
            },
            "exposure": {
                "TRAIN": "new_training_authorized",
                "CAL": "development_historically_exposed",
                "SEL": "development_historically_exposed",
                "PB": "final_historical_benchmark_exposed",
                "LOCAL-T2": "historical_official_like_validation",
                "ADDITIONAL": "final_native_validation",
            },
            "sources": {
                "task_data_contract": {
                    "logical_reference": config["paths"]["task_data_contract"],
                    "sha256": _sha256(data_contract),
                },
                "crosswindow_data_roles": {
                    "logical_reference": config["paths"]["crosswindow_data_roles"],
                    "sha256": _sha256(cross_roles_path),
                },
            },
        },
    )
    run_config = {
        "schema_version": "perception-gain-run-config-v1",
        "resolved_config": config,
        "inputs": input_records,
        "asset_resolution_sources": sources,
        "external_assets_file": "external:assets.local.json",
        "pilot_config_frozen": True,
    }
    _atomic_json(artifact_root / "RUN_CONFIG.json", run_config)
    _atomic_text(artifact_root / "CODE_BINDINGS.md", _code_bindings())
    local_assets = {key: value for key, value in assets.items() if value is not None}
    local_assets["perception_run_root"] = str(external_root)
    _atomic_json(external_root / "assets.local.json", local_assets)
    state = new_run_state(
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract),
    )
    _atomic_json(artifact_root / "RUN_STATE.json", state)
    _atomic_json(
        external_root / "campaign.local.json",
        {
            "artifact_root": str(artifact_root.resolve()),
            "config": str(config_path),
            "external_root": str(external_root),
        },
    )
    _append_jsonl(
        artifact_root / "EXECUTION_LOG.jsonl",
        {
            "command": "bootstrap",
            "cwd": "repo:.",
            "exit_code": 0,
            "stage": "bootstrap",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"status": "PASS", "stage": "bootstrap"}, sort_keys=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    config = load_campaign_config(Path(args.config).resolve())
    state = _read_json(
        PROJECT_ROOT / config["paths"]["artifact_root"] / "RUN_STATE.json"
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def _finalize_partial(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    artifact_root = PROJECT_ROOT / config["paths"]["artifact_root"]
    data_contract = PROJECT_ROOT / config["paths"]["task_data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract),
    )
    blocker = (
        Path(args.blocker).expanduser().resolve()
        if args.blocker
        else artifact_root / "training/S-BAL/INTERRUPTION.json"
    )
    interruption = _read_json(blocker)
    state = build_partial_finalization_state(state, interruption=interruption)
    external_root = _external_root(args.external_root)
    state = reconcile_partial_costs(
        state,
        artifact_root=artifact_root,
        external_root=external_root,
        interruption=interruption,
    )
    _atomic_json(state_path, state)
    _append_jsonl(
        artifact_root / "EXECUTION_LOG.jsonl",
        {
            "command": "finalize-partial --blocker repo:artifacts/perception_gain_v1/training/S-BAL/INTERRUPTION.json",
            "elapsed_seconds": 0.0,
            "exit_code": 0,
            "inputs": ["training/S-BAL/INTERRUPTION.json"],
            "outputs": [
                "FINAL_REPORT.md",
                "HANDOFF.md",
                "ARTIFACT_MANIFEST.json",
                "RELEASE_PLAN.json",
            ],
            "stage": "partial_finalization",
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    report = _run_report_stage(
        artifact_root=artifact_root,
        external_root=external_root,
    )
    state["partial_delivery"] = {
        **state["partial_delivery"],
        "report": "PASS",
        "publish": "RUNNING",
    }
    _atomic_json(state_path, state)
    publication = _run_publish_stage(
        artifact_root=artifact_root,
        external_root=external_root,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "execution_status": "PARTIAL_WITH_BLOCKERS",
                "report": report,
                "publication": publication,
            },
            sort_keys=True,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    from scripts.perception_gain_foundation import (
        run_bind,
        run_corrected_e1,
        run_diagnostic_panel,
        run_live_smoke,
    )

    config_path = Path(args.config).resolve()
    config = load_campaign_config(config_path)
    artifact_root = PROJECT_ROOT / config["paths"]["artifact_root"]
    data_contract = PROJECT_ROOT / config["paths"]["task_data_contract"]
    state_path = artifact_root / "RUN_STATE.json"
    state = load_resume_state(
        state_path,
        parent=config["identity"]["parent_commit"],
        instruction_sha=config["identity"]["instruction_sha256"],
        config_sha=_sha256(config_path),
        data_sha=_sha256(data_contract),
    )
    completed = state.get("completed_stages")
    if not isinstance(completed, list):
        raise CampaignError("run state lacks completed_stages")
    external_root = _external_root(args.external_root)
    assets_path = external_root / "assets.local.json"
    stages_to_run = pending_stages(completed, through=args.through)
    for stage in stages_to_run:
        started = dt.datetime.now(dt.timezone.utc)
        if stage == "bind":
            cached_binding = artifact_root / "foundation/R1_BINDING.json"
            if cached_binding.is_file():
                result = validate_cached_bind_stage(
                    _read_json(cached_binding),
                    checkpoint_sha256=config["identity"]["r1_checkpoint_sha256"],
                    checkpoint_bytes=int(config["identity"]["r1_checkpoint_bytes"]),
                )
            else:
                result = run_bind(assets_path=assets_path)
        elif stage == "foundation":
            evidence_paths = {
                "corrected_e1": artifact_root / "foundation/e1/status.json",
                "live_smoke": artifact_root / "foundation/LIVE_SMOKE.json",
                "diagnostic_panel": artifact_root
                / "foundation/diagnostic_panel/status.json",
                "native_smoke": artifact_root / "foundation/NATIVE_SMOKE.json",
            }
            if all(path.is_file() for path in evidence_paths.values()):
                result = validate_cached_foundation_stage(
                    **{name: _read_json(path) for name, path in evidence_paths.items()}
                )
            else:
                from scripts.perception_gain_native_evaluation import (
                    run_native_checkpoint_evaluation,
                )

                e1 = run_corrected_e1(assets_path=assets_path)
                live = run_live_smoke(
                    assets_path=assets_path,
                    external_root=external_root,
                    device_name="cuda:0",
                )
                diagnostic = run_diagnostic_panel(
                    assets_path=assets_path,
                    device_name="cuda:0",
                )
                native = run_native_checkpoint_evaluation(
                    variant="C0",
                    optimizer_update=0,
                    role="CAL",
                    checkpoint=None,
                    scorer_checkpoint=None,
                    horizons=(2, 5),
                    unit_limit=1,
                    assets_path=assets_path,
                    roles_path=artifact_root / "DATA_ROLES.json",
                    external_root=external_root,
                    output_path=evidence_paths["native_smoke"],
                    device_name="cuda:0",
                )
                result = validate_cached_foundation_stage(
                    corrected_e1=e1,
                    live_smoke=live,
                    diagnostic_panel=diagnostic,
                    native_smoke=native,
                )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["gpu_hours"])
        elif stage == "scorer":
            from scripts.prepare_perception_scorer import run as prepare_scorer_data
            from scripts.train_perception_gain import train_semantic_scorer

            data_manifest_path = artifact_root / "training/scorer/DATA_MANIFEST.json"
            training_summary_path = artifact_root / "training/scorer/RUN_SUMMARY.json"
            if data_manifest_path.is_file():
                data_manifest = _read_json(data_manifest_path)
            else:
                data_manifest = prepare_scorer_data(
                    assets_path=assets_path,
                    roles_path=artifact_root / "DATA_ROLES.json",
                    external_root=external_root,
                    device_name="cuda:0",
                )
            if training_summary_path.is_file():
                training_summary = _read_json(training_summary_path)
            else:
                shard_paths = tuple(
                    _external_artifact_path(
                        external_root, record.get("external_reference")
                    )
                    for record in data_manifest.get("shards", [])
                    if isinstance(record, Mapping)
                )
                training_summary = train_semantic_scorer(
                    cache_paths=shard_paths,
                    output_dir=external_root / "training/scorer/model",
                    summary_output=training_summary_path,
                    stop_after_updates=500,
                    device="cuda:0",
                )
            result = validate_scorer_stage(
                data_manifest=data_manifest,
                training_summary=training_summary,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["perception_training_gpu_hours"] = float(
                budget.get("perception_training_gpu_hours", 0.0)
            ) + float(result["gpu_hours"])
            budget["new_cache_bytes"] = int(budget.get("new_cache_bytes", 0)) + int(
                result["cache_bytes"]
            )
        elif stage == "perception_pilot":
            result = _run_perception_pilot_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["perception_training_gpu_hours"] = float(
                budget.get("perception_training_gpu_hours", 0.0)
            ) + float(result["training_gpu_hours"])
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["evaluation_gpu_hours"])
        elif stage == "perception_full":
            result = _run_perception_full_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["perception_training_gpu_hours"] = float(
                budget.get("perception_training_gpu_hours", 0.0)
            ) + float(result["training_gpu_hours"])
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["incremental_evaluation_gpu_hours"])
        elif stage == "perception_select":
            result = _run_perception_select_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["evaluation_gpu_hours"])
        elif stage == "refinement":
            result = _run_refinement_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["refinement_gpu_hours"] = float(
                budget.get("refinement_gpu_hours", 0.0)
            ) + float(result["refinement_gpu_hours"])
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["evaluation_gpu_hours"])
            budget["new_cache_bytes"] = int(budget.get("new_cache_bytes", 0)) + int(
                result["cache_bytes"]
            )
        elif stage == "final_lock":
            result = _run_final_lock_stage(
                artifact_root=artifact_root,
                config_path=config_path,
                config=config,
            )
        elif stage == "replication":
            result = _run_replication_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
                state=state,
                config=config,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["perception_training_gpu_hours"] = float(
                budget.get("perception_training_gpu_hours", 0.0)
            ) + float(result["perception_gpu_hours"])
            budget["refinement_gpu_hours"] = float(
                budget.get("refinement_gpu_hours", 0.0)
            ) + float(result["refinement_gpu_hours"])
        elif stage == "confirm":
            result = _run_confirmation_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["foundation_and_evaluation_gpu_hours"] = float(
                budget.get("foundation_and_evaluation_gpu_hours", 0.0)
            ) + float(result["gpu_hours"])
        elif stage == "profile":
            result = _run_profile_stage(
                artifact_root=artifact_root,
                assets_path=assets_path,
                external_root=external_root,
            )
            budget = state.get("budget_used")
            if not isinstance(budget, dict):
                raise CampaignError("run state lacks budget accounting")
            budget["profiling_gpu_hours"] = float(
                budget.get("profiling_gpu_hours", 0.0)
            ) + float(result["gpu_hours"])
        elif stage == "report":
            result = _run_report_stage(
                artifact_root=artifact_root,
                external_root=external_root,
            )
        elif stage == "publish":
            result = _run_publish_stage(
                artifact_root=artifact_root,
                external_root=external_root,
            )
        else:
            raise CampaignError(f"stage execution is not implemented yet: {stage}")
        if str(result.get("status", "")).upper() != "PASS":
            raise CampaignError(f"campaign stage did not pass: {stage}")
        completed.append(stage)
        state.update(
            {
                "completed_stages": completed,
                "stage": stage,
                "stage_status": "PASS",
                stage: result,
                "next_command": (
                    "python -m scripts.perception_gain_campaign run --config "
                    "configs/perception_gain_v1.yaml --external-root "
                    '"$PERSIST4D_PERCEPTION_RUN_ROOT" --through '
                    f"{STAGES[min(STAGES.index(stage) + 1, len(STAGES) - 1)]} --resume"
                ),
            }
        )
        _atomic_json(state_path, state)
        _append_jsonl(
            artifact_root / "EXECUTION_LOG.jsonl",
            {
                "command": f"run --through {args.through}",
                "elapsed_seconds": (
                    dt.datetime.now(dt.timezone.utc) - started
                ).total_seconds(),
                "exit_code": 0,
                "stage": stage,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "stage": state["stage"],
                "resumed": bool(args.resume),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "status"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default=str(DEFAULT_CONFIG))
        command.add_argument("--external-root")
        if name == "bootstrap":
            command.add_argument("--assets-from")
            for key in ASSET_KEYS:
                if key != "external_run_root":
                    command.add_argument(f"--{key.replace('_', '-')}", dest=key)
    run = subparsers.add_parser("run")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--external-root")
    run.add_argument("--through", choices=STAGES, required=True)
    run.add_argument("--resume", action="store_true")
    partial = subparsers.add_parser("finalize-partial")
    partial.add_argument("--config", default=str(DEFAULT_CONFIG))
    partial.add_argument("--external-root")
    partial.add_argument("--blocker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    if args.command == "status":
        return _status(args)
    if args.command == "run":
        return _run(args)
    if args.command == "finalize-partial":
        return _finalize_partial(args)
    raise CampaignError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "STAGES",
    "CampaignError",
    "build_partial_finalization_state",
    "load_campaign_config",
    "load_resume_state",
    "main",
    "new_run_state",
    "pending_stages",
    "validate_scorer_stage",
]
