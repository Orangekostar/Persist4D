#!/usr/bin/env python3
"""Deterministic selection rules for the Perception-Gain v1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any

HORIZONS = (2, 3, 4, 5)
RANK_TOLERANCE = 1e-6
PILOT_UPDATES = frozenset((250, 750))
CAL_UPDATES = frozenset((0, 250, 750, 1500, 2250, 3000))
VARIANTS = ("C0", "S-BAL", "S-WORST", "Q-SEM", "A-OPEN")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_ASSETS = Path("/home/ww/persist4d_runs/perception_gain_v1/assets.local.json")
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
DEFAULT_ROLES = ARTIFACT_ROOT / "DATA_ROLES.json"
DATA_CONTRACT = PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
R1_CHECKPOINT_SHA256 = (
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
)
R1_CHECKPOINT_BYTES = 754_813_672
BASE_MANIFEST = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/baseline/cache_manifest.json"
)
SUPPLEMENT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/task_memory_retention_v2/baseline/control_observation_manifest.json"
)


class PerceptionEvaluationError(ValueError):
    """Raised when evaluation inputs violate the frozen protocol."""


@dataclass(frozen=True)
class LivePopulationUnit:
    logical_unit_id: str
    reference_id: str
    sequence_id: str
    spec_index: int


@dataclass(frozen=True)
class AdditionalPopulationSlice:
    horizon: int
    base_dataset: object
    episode_specs: tuple[object, ...]
    units: tuple[LivePopulationUnit, ...]


def evaluation_population_source(role: str) -> str:
    if role == "PB":
        return "validation"
    if role in {"CAL", "SEL"}:
        return "training"
    if role == "ADDITIONAL":
        return "native_validation_slices"
    raise PerceptionEvaluationError("evaluation population role is invalid")


def build_role_base_dataset(
    *, config: object, data_root: Path, role: str, horizon: int = 5
) -> object:
    source = evaluation_population_source(role)
    if source == "validation":
        from scripts.evaluate_task_memory import _rio_population_base
        from scripts.run_task_memory_policy_baseline import PROTOCOL_B_POPULATION_ID

        return _rio_population_base(
            config,
            data_root=data_root,
            horizon=horizon,
            population_id=PROTOCOL_B_POPULATION_ID,
        )
    if source == "training":
        from scripts.preflight_task_memory_episode import _rio_base_dataset

        return _rio_base_dataset(config, data_root=data_root, horizon=horizon)
    raise PerceptionEvaluationError("ADDITIONAL requires horizon-specific slices")


def select_live_population_units(
    *,
    role: str,
    role_references: Sequence[str],
    episode_specs: Sequence[object],
    expected_logical_units: int,
) -> tuple[LivePopulationUnit, ...]:
    if (
        not isinstance(role, str)
        or not role
        or isinstance(role_references, (str, bytes))
        or not role_references
        or len(set(role_references)) != len(role_references)
        or isinstance(expected_logical_units, bool)
        or not isinstance(expected_logical_units, int)
        or expected_logical_units <= 0
    ):
        raise PerceptionEvaluationError("live population coverage request is invalid")
    selected = set(role_references)
    units = []
    seen = set()
    for spec_index, spec in enumerate(episode_specs):
        reference = getattr(spec, "reference_id", None)
        sequence = getattr(spec, "source_sequence_id", None)
        if reference not in selected:
            continue
        if not isinstance(sequence, str) or not sequence:
            raise PerceptionEvaluationError("live population sequence identity differs")
        seen.add(reference)
        units.append(
            LivePopulationUnit(
                logical_unit_id=f"{role}:{len(units):05d}",
                reference_id=reference,
                sequence_id=sequence,
                spec_index=spec_index,
            )
        )
    if seen != selected or len(units) != expected_logical_units:
        raise PerceptionEvaluationError(
            "live population coverage differs: "
            f"references={len(seen)}/{len(selected)}, "
            f"units={len(units)}/{expected_logical_units}"
        )
    return tuple(units)


def build_additional_population_slices(
    *,
    config: object,
    data_root: Path,
    metadata_path: Path,
    data_contract: Mapping[str, object],
    role_references: Sequence[str],
) -> tuple[AdditionalPopulationSlice, ...]:
    from datasets.task_memory_episode import (
        TaskMemoryEpisodeSpec,
        build_native_episode_masters,
    )
    from scripts.evaluate_task_memory import (
        NATIVE_POPULATION_ID,
        _rio_population_base,
    )
    from scripts.preflight_task_memory_episode import (
        _role_by_reference,
        load_reference_by_scene,
    )

    expected = {2: (111, 40), 3: (77, 23), 4: (32, 8)}
    allowed = set(role_references)
    slices = []
    for horizon, (expected_units, expected_references) in expected.items():
        base = _rio_population_base(
            config,
            data_root=data_root,
            horizon=horizon,
            population_id=NATIVE_POPULATION_ID,
        )
        masters = tuple(
            sorted(
                (
                    master
                    for master in build_native_episode_masters(
                        base,
                        reference_by_scene=load_reference_by_scene(metadata_path),
                        role_by_reference=_role_by_reference(data_contract),
                    )
                    if master.role == NATIVE_POPULATION_ID
                    and len(master.scan_ids) == horizon
                    and master.reference_id in allowed
                ),
                key=lambda value: (value.reference_id, value.sequence_id),
            )
        )
        references = tuple(sorted({master.reference_id for master in masters}))
        if len(masters) != expected_units or len(references) != expected_references:
            raise PerceptionEvaluationError(
                f"ADDITIONAL T{horizon} population coverage differs"
            )
        specs = tuple(
            TaskMemoryEpisodeSpec.from_master(
                master,
                horizon=horizon,
                augmentation_seed=45,
                draw_index=index,
                bucket=f"T{horizon}",
            )
            for index, master in enumerate(masters)
        )
        units = select_live_population_units(
            role=f"ADDITIONAL-T{horizon}",
            role_references=references,
            episode_specs=specs,
            expected_logical_units=expected_units,
        )
        slices.append(
            AdditionalPopulationSlice(
                horizon=horizon,
                base_dataset=base,
                episode_specs=specs,
                units=units,
            )
        )
    return tuple(slices)


def summarize_additional_population_slices(
    slices: Sequence[AdditionalPopulationSlice],
) -> dict[str, object]:
    expected = {2: (111, 40), 3: (77, 23), 4: (32, 8)}
    if tuple(item.horizon for item in slices) != tuple(expected):
        raise PerceptionEvaluationError("ADDITIONAL horizons differ")
    manifest_rows = []
    population_by_horizon = {}
    logical_ids = set()
    for item in slices:
        unit_count, reference_count = expected[item.horizon]
        references = {unit.reference_id for unit in item.units}
        if len(item.units) != unit_count or len(references) != reference_count:
            raise PerceptionEvaluationError(
                f"ADDITIONAL T{item.horizon} population coverage differs"
            )
        population_by_horizon[str(item.horizon)] = {
            "reference_count": len(references),
            "logical_unit_count": len(item.units),
        }
        for unit in item.units:
            if unit.logical_unit_id in logical_ids:
                raise PerceptionEvaluationError(
                    "ADDITIONAL logical units are not unique"
                )
            logical_ids.add(unit.logical_unit_id)
            try:
                spec = item.episode_specs[unit.spec_index]
                scan_ids = list(spec.scan_ids)
                scan_indices = list(spec.scan_indices)
                context_index = int(spec.context_index)
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                raise PerceptionEvaluationError(
                    "ADDITIONAL episode identity differs"
                ) from error
            if len(scan_ids) != item.horizon or len(scan_indices) != item.horizon:
                raise PerceptionEvaluationError(
                    "ADDITIONAL episode horizon identity differs"
                )
            manifest_rows.append(
                {
                    "T": item.horizon,
                    "logical_unit_id": unit.logical_unit_id,
                    "reference_id": unit.reference_id,
                    "sequence_id": unit.sequence_id,
                    "context_index": context_index,
                    "scan_ids": scan_ids,
                    "scan_indices": scan_indices,
                }
            )
    encoded = json.dumps(
        manifest_rows, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        "population_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "population_by_horizon": population_by_horizon,
        "expected_logical_unit_count": len(logical_ids),
        "expected_prefix_count": len(logical_ids),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PerceptionEvaluationError(f"cannot decode JSON input: {path}") from error
    if not isinstance(value, dict):
        raise PerceptionEvaluationError(f"JSON input must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_pooled_d0_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_reference_count: int,
    expected_logical_unit_count: int,
) -> dict[int, float]:
    """Extract the four official pooled D0 t-mAP values with exact coverage."""

    selected = [
        row
        for row in rows
        if row.get("method") == "D0" and row.get("reference") == "all"
    ]
    by_horizon: dict[int, float] = {}
    for row in selected:
        horizon = row.get("T")
        value = row.get("t_mAP")
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or horizon not in HORIZONS
            or horizon in by_horizon
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            or row.get("reference_count") != expected_reference_count
            or row.get("logical_unit_count") != expected_logical_unit_count
        ):
            raise PerceptionEvaluationError("D0 pooled metric coverage is invalid")
        by_horizon[horizon] = float(value)
    if set(by_horizon) != set(HORIZONS):
        raise PerceptionEvaluationError("D0 pooled metric coverage is incomplete")
    return by_horizon


def select_d0_metric_rows(
    rows: Sequence[Mapping[str, object]], *, data_role: str
) -> list[dict[str, object]]:
    if not isinstance(data_role, str) or not data_role:
        raise PerceptionEvaluationError("D0 metric row role is invalid")
    return [
        {**dict(row), "data_role": data_role}
        for row in rows
        if row.get("method") == "D0"
    ]


def build_checkpoint_candidate(
    *,
    variant: str,
    optimizer_update: int,
    checkpoint_reference: str,
    checkpoint_sha256: str,
    metrics: Mapping[Any, Any],
    expected_logical_units: int,
    completed_logical_units: int,
    new_parameter_count: int,
) -> dict[str, object]:
    if variant not in VARIANTS:
        raise PerceptionEvaluationError("checkpoint candidate variant is invalid")
    if (
        isinstance(optimizer_update, bool)
        or not isinstance(optimizer_update, int)
        or optimizer_update not in CAL_UPDATES
        or not isinstance(checkpoint_reference, str)
        or not checkpoint_reference
        or not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
        or isinstance(expected_logical_units, bool)
        or not isinstance(expected_logical_units, int)
        or expected_logical_units <= 0
        or isinstance(completed_logical_units, bool)
        or not isinstance(completed_logical_units, int)
        or not 0 <= completed_logical_units <= expected_logical_units
        or isinstance(new_parameter_count, bool)
        or not isinstance(new_parameter_count, int)
        or new_parameter_count < 0
    ):
        raise PerceptionEvaluationError("checkpoint candidate identity is invalid")
    return {
        "method_id": variant,
        "optimizer_update": optimizer_update,
        "new_parameter_count": new_parameter_count,
        "metrics": _normalise_metrics(metrics, field="metrics"),
        "coverage_status": (
            "COMPLETE"
            if completed_logical_units == expected_logical_units
            else "INCOMPLETE"
        ),
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha256,
        "expected_logical_units": expected_logical_units,
        "completed_logical_units": completed_logical_units,
    }


def build_live_cache_key(
    *,
    master_sequence_id: str,
    reference_scene_id: str,
    stage_index: int,
    scan_ids: Sequence[str],
    local_window_scan_ids: Sequence[str],
) -> dict[str, object]:
    return {
        "master_sequence_id": master_sequence_id,
        "reference_scene_id": reference_scene_id,
        "order_id": "canonical",
        "stage_index": stage_index,
        "history_scan_ids": list(scan_ids[: stage_index + 1]),
        "local_window_scan_ids": list(local_window_scan_ids),
    }


def build_live_provenance(
    *,
    source_commit: str,
    checkpoint_sha256: str,
    config_sha256: str,
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    from scripts.task_memory_contracts import canonical_json_sha256

    return {
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "dataset_sha256": canonical_json_sha256(list(episodes)),
    }


def build_live_replay_payload(
    *,
    reference_id: str,
    sequence_id: str,
    episode_id: str,
    scan_ids: Sequence[str],
    stages: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    required = {"observation", "prediction", "stage_meta", "target"}
    if (
        any(
            not isinstance(value, str) or not value
            for value in (reference_id, sequence_id, episode_id)
        )
        or isinstance(scan_ids, (str, bytes))
        or not isinstance(scan_ids, Sequence)
        or len(scan_ids) != 5
        or len(set(scan_ids)) != 5
        or any(not isinstance(value, str) or not value for value in scan_ids)
        or isinstance(stages, (str, bytes))
        or not isinstance(stages, Sequence)
        or len(stages) != 5
        or any(
            not isinstance(stage, Mapping) or set(stage) != required for stage in stages
        )
    ):
        raise PerceptionEvaluationError("live replay requires exact five stages")
    return (
        {
            "episode": {
                "episode_id": episode_id,
                "reference_id": reference_id,
                "scan_ids": list(scan_ids),
                "sequence_id": sequence_id,
            },
            "stages": [
                {"stage_meta": stage["stage_meta"], "target": stage["target"]}
                for stage in stages
            ],
        },
        {
            "stages": [
                {
                    "observation": stage["observation"],
                    "prediction": stage["prediction"],
                    "base_overlap": None,
                }
                for stage in stages
            ]
        },
    )


def _normalise_metrics(metrics: Mapping[Any, Any], *, field: str) -> dict[int, float]:
    normalised: dict[int, float] = {}
    for horizon in HORIZONS:
        candidates = (horizon, str(horizon), f"T{horizon}")
        present = [key for key in candidates if key in metrics]
        if len(present) != 1:
            raise PerceptionEvaluationError(
                f"{field} must contain exactly one value for horizon T{horizon}"
            )
        value = float(metrics[present[0]])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise PerceptionEvaluationError(
                f"{field}[T{horizon}] must be finite and within [0, 1]"
            )
        normalised[horizon] = value
    return normalised


def _candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, int]:
    method_id = candidate.get("method_id")
    update = candidate.get("optimizer_update")
    if not isinstance(method_id, str) or not method_id:
        raise PerceptionEvaluationError(
            "candidate method_id must be a non-empty string"
        )
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise PerceptionEvaluationError(
            "candidate optimizer_update must be a non-negative integer"
        )
    return method_id, update


def compare_candidate(
    candidate: Mapping[str, Any],
    *,
    baseline_metrics: Mapping[Any, Any],
) -> dict[str, Any]:
    """Attach Protocol-A delta scores to one checkpoint candidate."""

    _candidate_identity(candidate)
    baseline = _normalise_metrics(baseline_metrics, field="baseline_metrics")
    metrics = _normalise_metrics(
        candidate.get("metrics", {}), field="candidate.metrics"
    )
    row = dict(candidate)
    row["metrics"] = metrics
    if candidate.get("coverage_status") != "COMPLETE":
        row.update(
            {
                "comparison_status": "INCOMPLETE_COVERAGE",
                "deltas": None,
                "S_mean": None,
                "S_long": None,
                "S_min": None,
            }
        )
        return row

    deltas = {horizon: metrics[horizon] - baseline[horizon] for horizon in HORIZONS}
    row.update(
        {
            "comparison_status": "COMPLETE",
            "deltas": deltas,
            "S_mean": sum(deltas.values()) / len(HORIZONS),
            "S_long": (deltas[4] + deltas[5]) / 2.0,
            "S_min": min(deltas.values()),
        }
    )
    return row


def _rank_compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    left_complete = left.get("comparison_status") == "COMPLETE"
    right_complete = right.get("comparison_status") == "COMPLETE"
    if left_complete != right_complete:
        return -1 if left_complete else 1

    if left_complete:
        for field in ("S_mean", "S_long", "S_min"):
            difference = float(left[field]) - float(right[field])
            if abs(difference) > RANK_TOLERANCE:
                return -1 if difference > 0.0 else 1

    left_parameters = left.get("new_parameter_count", 0)
    right_parameters = right.get("new_parameter_count", 0)
    if (
        isinstance(left_parameters, bool)
        or not isinstance(left_parameters, int)
        or left_parameters < 0
        or isinstance(right_parameters, bool)
        or not isinstance(right_parameters, int)
        or right_parameters < 0
    ):
        raise PerceptionEvaluationError(
            "new_parameter_count must be a non-negative integer"
        )
    if left_parameters != right_parameters:
        return -1 if left_parameters < right_parameters else 1

    left_method, left_update = _candidate_identity(left)
    right_method, right_update = _candidate_identity(right)
    if left_update != right_update:
        return -1 if left_update < right_update else 1
    return (left_method > right_method) - (left_method < right_method)


def rank_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[Any, Any],
) -> list[dict[str, Any]]:
    compared = [
        compare_candidate(candidate, baseline_metrics=baseline_metrics)
        for candidate in candidates
    ]
    identities = [_candidate_identity(row) for row in compared]
    if len(identities) != len(set(identities)):
        raise PerceptionEvaluationError("duplicate candidate method/update identity")
    return sorted(compared, key=cmp_to_key(_rank_compare))


def _best_by_method(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[Any, Any],
    allowed_updates: frozenset[int],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        method_id, update = _candidate_identity(candidate)
        if update not in allowed_updates:
            raise PerceptionEvaluationError(
                f"unexpected optimizer update {update} for method {method_id}"
            )
        grouped.setdefault(method_id, []).append(candidate)

    selected: dict[str, dict[str, Any]] = {}
    for method_id, rows in grouped.items():
        complete = [
            row
            for row in rank_candidates(rows, baseline_metrics=baseline_metrics)
            if row["comparison_status"] == "COMPLETE"
        ]
        if not complete:
            raise PerceptionEvaluationError(
                f"method {method_id} has no complete candidate"
            )
        selected[method_id] = complete[0]
    return selected


def select_pilot_promotions(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[Any, Any],
) -> dict[str, Any]:
    selected = _best_by_method(
        candidates,
        baseline_metrics=baseline_metrics,
        allowed_updates=PILOT_UPDATES,
    )
    if "C0" not in selected:
        raise PerceptionEvaluationError("pilot selection requires C0")

    new_arms = [row for method, row in selected.items() if method != "C0"]
    ranked_new = sorted(new_arms, key=cmp_to_key(_rank_compare))
    promoted = [
        row for row in ranked_new if row["S_mean"] >= -0.01 and row["S_min"] >= -0.02
    ][:2]
    if promoted:
        status = "TWO_ARM_GATE" if len(promoted) == 2 else "ONE_ARM_GATE"
    elif ranked_new and ranked_new[0]["S_mean"] >= -0.03:
        promoted = ranked_new[:1]
        status = "ONE_ARM_FALLBACK"
    else:
        status = "NO_NEW_ARM"

    methods = ["C0", *(row["method_id"] for row in promoted)]
    return {
        "promotion_status": status,
        "full_training_arms": methods,
        "selected_pilot_points": selected,
        "resume_updates": {method: 750 for method in methods},
    }


def select_cal_checkpoints(
    candidates: Sequence[Mapping[str, Any]],
    *,
    baseline_metrics: Mapping[Any, Any],
    completed_full_arms: Sequence[str],
) -> dict[str, dict[str, Any]]:
    selected = _best_by_method(
        candidates,
        baseline_metrics=baseline_metrics,
        allowed_updates=CAL_UPDATES,
    )
    required = tuple(completed_full_arms)
    if len(required) != len(set(required)):
        raise PerceptionEvaluationError("completed_full_arms contains duplicates")
    missing = [method for method in required if method not in selected]
    unexpected = [method for method in selected if method not in required]
    if missing or unexpected:
        raise PerceptionEvaluationError(
            f"CAL method mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {method: selected[method] for method in required}


def _passes_base_gate(row: Mapping[str, Any]) -> bool:
    return (
        row["comparison_status"] == "COMPLETE"
        and row["S_min"] >= -0.001
        and row["S_mean"] >= 0.005
        and row["S_long"] >= 0.0
    )


def select_perception_parent(
    candidates: Sequence[Mapping[str, Any]],
    *,
    d0_metrics: Mapping[Any, Any],
    c0_method_id: str,
) -> dict[str, Any]:
    ranked = rank_candidates(candidates, baseline_metrics=d0_metrics)
    c0_rows = [row for row in ranked if row["method_id"] == c0_method_id]
    if len(c0_rows) != 1:
        raise PerceptionEvaluationError(
            "SEL selection requires exactly one C0 candidate"
        )
    c0 = c0_rows[0]

    eligible_new: list[dict[str, Any]] = []
    for row in ranked:
        if row["method_id"] == c0_method_id or not _passes_base_gate(row):
            continue
        versus_c0 = compare_candidate(row, baseline_metrics=c0["metrics"])
        if (
            versus_c0["comparison_status"] == "COMPLETE"
            and versus_c0["S_mean"] >= 0.002
            and versus_c0["S_min"] >= -0.002
        ):
            eligible_new.append(row)

    if eligible_new:
        selected = sorted(eligible_new, key=cmp_to_key(_rank_compare))[0]
        return {
            "status": "NEW_PERCEPTION_SELECTED",
            "selected_method_id": selected["method_id"],
            "selected": selected,
        }
    if _passes_base_gate(c0):
        return {
            "status": "CONTINUATION_ONLY",
            "selected_method_id": c0_method_id,
            "selected": c0,
        }
    return {"status": "KEEP_R1", "selected_method_id": "R1", "selected": None}


def select_refiner(
    candidate: Mapping[str, Any],
    *,
    parent_metrics: Mapping[Any, Any],
) -> dict[str, Any]:
    comparison = compare_candidate(candidate, baseline_metrics=parent_metrics)
    enabled = (
        comparison["comparison_status"] == "COMPLETE"
        and comparison["S_min"] >= -0.001
        and comparison["S_mean"] >= 0.003
        and comparison["S_long"] >= 0.0
    )
    return {
        "status": "REFINER_SELECTED" if enabled else "KEEP_PARENT",
        "enabled": enabled,
        "selected": comparison if enabled else None,
        "comparison": comparison,
    }


def _external_reference(path: Path, *, external_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(external_root.expanduser().resolve())
    except ValueError:
        return f"external:{resolved.name}"
    return f"external:{relative.as_posix()}"


def _load_evaluation_weights(
    *,
    system: object,
    variant: str,
    optimizer_update: int,
    r1_checkpoint: Path,
    checkpoint: Path | None,
    scorer_checkpoint: Path | None,
) -> tuple[dict[str, object], str, str, int, list[dict[str, object]]]:
    import torch

    from trainer.perception_gain_trainer import (
        load_frozen_semantic_scorer,
        strict_load_r1_perception,
    )

    weight_sources = [
        {
            "role": "r1_initialization",
            "sha256": R1_CHECKPOINT_SHA256,
            "bytes": R1_CHECKPOINT_BYTES,
        }
    ]
    if optimizer_update == 0:
        if checkpoint is not None:
            raise PerceptionEvaluationError(
                "update 0 must use the frozen R1 initialization"
            )
        if (
            not r1_checkpoint.is_file()
            or r1_checkpoint.stat().st_size != R1_CHECKPOINT_BYTES
            or _sha256(r1_checkpoint) != R1_CHECKPOINT_SHA256
        ):
            raise PerceptionEvaluationError("R1 checkpoint identity differs")
        r1_payload = torch.load(r1_checkpoint, map_location="cpu", weights_only=False)
        r1_state = (
            r1_payload.get("state_dict") if isinstance(r1_payload, Mapping) else None
        )
        if not isinstance(r1_state, Mapping):
            raise PerceptionEvaluationError("R1 checkpoint lacks state_dict")
        audit = strict_load_r1_perception(
            system,
            r1_state,
            allow_semantic_scorer=variant == "Q-SEM",
        )
        identity = R1_CHECKPOINT_SHA256
        checkpoint_reference = "external:r1_checkpoint"
        if variant == "Q-SEM":
            if scorer_checkpoint is None or not scorer_checkpoint.is_file():
                raise PerceptionEvaluationError(
                    "Q-SEM update 0 requires the frozen scorer"
                )
            scorer_audit = load_frozen_semantic_scorer(system, str(scorer_checkpoint))
            scorer_audit = {
                **scorer_audit,
                "path": "external:training/scorer/model/update=0500.ckpt",
            }
            scorer_sha = _sha256(scorer_checkpoint)
            weight_sources.append(
                {
                    "role": "semantic_scorer",
                    "sha256": scorer_sha,
                    "bytes": scorer_checkpoint.stat().st_size,
                }
            )
            audit = {**audit, "scorer": scorer_audit}
            identity = hashlib.sha256(
                f"{identity}:Q-SEM:{scorer_sha}".encode("ascii")
            ).hexdigest()
            checkpoint_reference = (
                "external:r1_checkpoint+external:training/scorer/model/update=0500.ckpt"
            )
    else:
        if checkpoint is None or not checkpoint.is_file():
            raise PerceptionEvaluationError(
                "trained evaluation checkpoint is unavailable"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise PerceptionEvaluationError("trained checkpoint lacks state_dict")
        try:
            incompatible = system.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise PerceptionEvaluationError(
                f"trained checkpoint tensor contract differs: {error}"
            ) from error
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise PerceptionEvaluationError("trained checkpoint load was not strict")
        identity = _sha256(checkpoint)
        checkpoint_reference = str(checkpoint)
        audit = {
            "loaded_key_count": len(state),
            "missing_keys": [],
            "unexpected_keys": [],
        }
        weight_sources.append(
            {
                "role": "trained_checkpoint",
                "sha256": identity,
                "bytes": checkpoint.stat().st_size,
            }
        )

    new_parameter_count = (
        sum(
            parameter.numel()
            for name, parameter in system.named_parameters()
            if name.startswith("model.semantic_query_scorer.")
        )
        if variant == "Q-SEM"
        else 0
    )
    return audit, identity, checkpoint_reference, new_parameter_count, weight_sources


def run_checkpoint_evaluation(
    *,
    variant: str,
    optimizer_update: int,
    role: str,
    checkpoint: Path | None,
    scorer_checkpoint: Path | None,
    assets_path: Path = DEFAULT_ASSETS,
    roles_path: Path = DEFAULT_ROLES,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    output_path: Path | None = None,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    """Stream one real checkpoint through the fixed CAL or SEL D0 protocol."""

    import hydra
    import torch
    import yaml
    from omegaconf import OmegaConf

    from datasets.task_memory_episode import (
        TaskMemoryEpisodeCollator,
        TaskMemoryEpisodeDataset,
    )
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _latest_full_resolution_masks,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        build_rio_class_mapper,
        cache_payload_from_inference,
    )
    from scripts.perception_gain_foundation import (
        _metric_class_mapping,
        _resolve_cache_assets,
    )
    from scripts.replay_crosswindow_association import E0ReplayAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.run_task_memory_controls import (
        _window_observation,
        observation_payload,
    )
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        PROTOCOL_B_POPULATION_ID,
        _episode_specs,
        _meta_payload,
        _prediction_payload,
        _stage_target,
        _validate_collated_stage_identity,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.task_memory_contracts import canonical_json_sha256
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import PerceptionGainTrainer

    if variant not in VARIANTS or role not in {"CAL", "SEL", "PB"}:
        raise PerceptionEvaluationError("evaluation variant or data role is invalid")
    if optimizer_update not in CAL_UPDATES:
        raise PerceptionEvaluationError(
            "evaluation update is outside the fixed CAL schedule"
        )
    assets = _resolve_cache_assets(assets_path)
    roles_payload = _load_json(roles_path).get("roles")
    references = roles_payload.get(role) if isinstance(roles_payload, Mapping) else None
    if (
        isinstance(references, (str, bytes))
        or not isinstance(references, Sequence)
        or not references
    ):
        raise PerceptionEvaluationError(f"{role} role is unavailable")
    device = _validate_cuda_device(device_name)
    run_dir = (
        external_root
        / "evaluation"
        / role.lower()
        / variant
        / f"update={optimizer_update:04d}"
    )
    config = compose_variant_config(
        variant,
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
    )
    config.model.return_query_features = True
    if variant == "Q-SEM" and scorer_checkpoint is not None:
        config.perception_training.scorer_checkpoint = str(scorer_checkpoint)
    base_dataset = build_role_base_dataset(
        config=config,
        data_root=Path(assets["data_root"]),
        role=role,
        horizon=5,
    )
    population_id = (
        PROTOCOL_B_POPULATION_ID if role == "PB" else DEVELOPMENT_POPULATION_ID
    )
    base_dataset, masters, population_manifest_sha256 = build_baseline_population(
        base_dataset,
        data_contract=_load_json(DATA_CONTRACT),
        metadata_path=Path(assets["rio_metadata"]),
        population_id=population_id,
        protocol_b_manifest_path=PROJECT_ROOT
        / "artifacts/P6A/protocol_b_manifest.json",
    )
    episode_specs = _episode_specs(masters)
    units = select_live_population_units(
        role=role,
        role_references=references,
        episode_specs=episode_specs,
        expected_logical_units={"CAL": 23, "SEL": 24, "PB": 129}[role],
    )

    system = PerceptionGainTrainer(config)
    load_audit, checkpoint_sha, checkpoint_reference, new_parameters, weight_sources = (
        _load_evaluation_weights(
            system=system,
            variant=variant,
            optimizer_update=optimizer_update,
            r1_checkpoint=Path(assets["r1_checkpoint"]),
            checkpoint=checkpoint,
            scorer_checkpoint=scorer_checkpoint,
        )
    )
    if checkpoint is not None:
        checkpoint_reference = _external_reference(
            checkpoint, external_root=external_root
        )
    system.to(device).eval().requires_grad_(False)
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(config.data.validation_collation)
    )
    class_mapper = build_rio_class_mapper(base_dataset)
    p6a = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())
    settings = p6a["baselines"]["b4"]
    observation_settings = {
        "background_class": int(settings["background_class"]),
        "confidence_threshold": float(settings["confidence_threshold"]),
        "mask_threshold": float(settings["mask_threshold"]),
        "minimum_mask_support": int(settings["minimum_mask_support"]),
    }
    config_sha = canonical_json_sha256(OmegaConf.to_container(config, resolve=True))
    provenance = build_live_provenance(
        source_commit="6ef77620aa20926311eff3124a794a6ca2e32727",
        checkpoint_sha256=checkpoint_sha,
        config_sha256=config_sha,
        episodes=tuple(
            {
                "reference_id": episode_specs[unit.spec_index].reference_id,
                "sequence_id": episode_specs[unit.spec_index].source_sequence_id,
                "scan_ids": list(episode_specs[unit.spec_index].scan_ids),
            }
            for unit in units
        ),
    )
    accumulator = E0ReplayAccumulator(
        dataset_spec=str(Path(assets["metric_dataset_spec"])),
        class_mapping=_metric_class_mapping(Path(assets["metric_dataset_spec"])),
        checkpoint_sha256=checkpoint_sha,
        source_commit=provenance["source_commit"],
        index_trigger_count=0,
    )
    completed_units = []
    incomplete_units = []
    completed_references = set()
    started = time.perf_counter()
    with deterministic_inference_runtime(45, device):
        for index, unit in enumerate(units, start=1):
            spec = episode_specs[unit.spec_index]
            try:
                episode = TaskMemoryEpisodeDataset(
                    base_dataset, (spec,), apply_augmentation=False
                )[0]
                with _frozen_inference_seed(45, device):
                    batch = collator([episode])
                    live_stages = []
                    for stage_batch in batch.stage_batches:
                        data, targets, names = stage_batch.model_batch
                        meta = stage_batch.stage_meta[0]
                        _validate_collated_stage_identity(
                            names=names, scan_ids_in_window=meta.scan_ids_in_window
                        )
                        full_targets = getattr(data, "target_full", None)
                        if (
                            not isinstance(full_targets, Sequence)
                            or len(full_targets) != 1
                        ):
                            raise PerceptionEvaluationError(
                                "live evaluation stage lacks full target"
                            )
                        full_target = full_targets[0]
                        data = _move_data_to_device(data, device)
                        targets = _move_targets_to_device(targets, device)
                        target = targets[0]
                        segment_stages = _segment_stages(target)
                        latest_stage = int(segment_stages.max().item())
                        raw_coordinates = system._process_raw_coordinates(data)
                        with torch.inference_mode():
                            output = system(
                                data,
                                point2segment=[target["point2segment"]],
                                raw_coordinates=raw_coordinates,
                                is_eval=True,
                            )
                        local = build_local_observation(
                            output,
                            [segment_stages],
                            latest_stage=latest_stage,
                            **observation_settings,
                        )
                        window = _window_observation(
                            local_observation=local,
                            output=output,
                            segment_stages=segment_stages,
                            latest_stage=latest_stage,
                            confidence_threshold=observation_settings[
                                "confidence_threshold"
                            ],
                            mask_threshold=observation_settings["mask_threshold"],
                            minimum_mask_support=observation_settings[
                                "minimum_mask_support"
                            ],
                        )
                        current_masks = _latest_full_resolution_masks(
                            system,
                            output,
                            target,
                            data,
                            latest_local_stage=latest_stage,
                        )
                        raw = cache_payload_from_inference(
                            key=build_live_cache_key(
                                master_sequence_id=spec.source_sequence_id,
                                reference_scene_id=spec.reference_id,
                                stage_index=meta.absolute_stage_index,
                                scan_ids=spec.scan_ids,
                                local_window_scan_ids=meta.scan_ids_in_window,
                            ),
                            provenance=provenance,
                            observation=local,
                            full_masks=current_masks,
                            full_target=full_target,
                            latest_local_stage=latest_stage,
                        )
                        prediction = extract_official_task_prediction(
                            system=system,
                            output=output,
                            target_low_resolution=target,
                            target_full_resolution=full_target,
                            data=data,
                            class_mapper=class_mapper,
                            latest_stage_index=latest_stage,
                        )
                        live_stages.append(
                            {
                                "observation": observation_payload(window),
                                "prediction": _prediction_payload(prediction),
                                "stage_meta": _meta_payload(meta),
                                "target": _stage_target(raw),
                            }
                        )
                        del (
                            data,
                            targets,
                            output,
                            local,
                            window,
                            current_masks,
                            prediction,
                        )
                    base, supplement = build_live_replay_payload(
                        reference_id=spec.reference_id,
                        sequence_id=spec.source_sequence_id,
                        episode_id=spec.episode_id,
                        scan_ids=spec.scan_ids,
                        stages=live_stages,
                    )
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as error:
                incomplete_units.append(
                    {
                        "logical_unit_id": unit.logical_unit_id,
                        "reference_id": unit.reference_id,
                        "sequence_id": unit.sequence_id,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
                torch.cuda.empty_cache()
                continue
            accumulator.update(
                logical_unit_id=unit.logical_unit_id,
                base=base,
                supplement=supplement,
            )
            completed_units.append(unit.logical_unit_id)
            completed_references.add(unit.reference_id)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "role": role,
                        "total": len(units),
                        "update": optimizer_update,
                        "variant": variant,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del base, episode, supplement
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not completed_units:
        first_failure = incomplete_units[0] if incomplete_units else None
        raise PerceptionEvaluationError(
            f"checkpoint evaluation completed no units; first_failure={first_failure}"
        )
    replay = accumulator.finalize()
    metrics = extract_pooled_d0_metrics(
        replay["metric_rows"],
        expected_reference_count=len(completed_references),
        expected_logical_unit_count=len(completed_units),
    )
    candidate = build_checkpoint_candidate(
        variant=variant,
        optimizer_update=optimizer_update,
        checkpoint_reference=checkpoint_reference,
        checkpoint_sha256=checkpoint_sha,
        metrics=metrics,
        expected_logical_units=len(units),
        completed_logical_units=len(completed_units),
        new_parameter_count=new_parameters,
    )
    if replay["status"] != "PASS":
        candidate["coverage_status"] = "INCOMPLETE"
    selected_rows = select_d0_metric_rows(replay["metric_rows"], data_role=role)
    status = (
        "PASS"
        if candidate["coverage_status"] == "COMPLETE" and not incomplete_units
        else "PARTIAL"
    )
    summary = {
        "schema_version": "perception-gain-checkpoint-evaluation-v1",
        "status": status,
        "variant": variant,
        "optimizer_update": optimizer_update,
        "data_role": role,
        "output_policy": "D0/lag1/mean",
        "official_metric": "pooled_t_mAP",
        "checkpoint": checkpoint_reference,
        "checkpoint_sha256": checkpoint_sha,
        "weight_sources": weight_sources,
        "resolved_config_sha256": config_sha,
        "population_manifest_sha256": population_manifest_sha256,
        "evaluation_input_mode": "LIVE_LOCAL_DATASET",
        "roles_sha256": _sha256(roles_path),
        "expected_reference_count": len(references),
        "completed_reference_count": len(completed_references),
        "expected_logical_unit_count": len(units),
        "completed_logical_unit_count": len(completed_units),
        "completed_units": completed_units,
        "incomplete_units": incomplete_units,
        "candidate": candidate,
        "metric_rows": selected_rows,
        "t2_same_forward_parity": replay["t2_parity"],
        "load_audit": load_audit,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "source_sha256": _sha256(Path(__file__)),
    }
    if output_path is None:
        output_path = (
            ARTIFACT_ROOT
            / "training/pilot/evaluation"
            / role.lower()
            / variant
            / f"update={optimizer_update:04d}.json"
        )
    _atomic_json(output_path, summary)
    del system
    torch.cuda.empty_cache()
    return summary


def write_immutable_lock(
    path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    destination = Path(path)
    serialisable = json.loads(json.dumps(dict(payload), allow_nan=False))
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PerceptionEvaluationError(
                f"cannot read immutable lock {destination}: {error}"
            ) from error
        if existing != serialisable:
            raise PerceptionEvaluationError(
                f"immutable lock {destination} already exists with different content"
            )
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(serialisable, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError:
        return write_immutable_lock(destination, serialisable)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return serialisable


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lock", "evaluate"), default="lock")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--update", type=int)
    parser.add_argument("--role", choices=("CAL", "SEL", "PB"), default="CAL")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scorer-checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    if arguments.mode == "evaluate":
        if arguments.variant is None or arguments.update is None:
            parser.error("evaluate mode requires --variant and --update")
        run_checkpoint_evaluation(
            variant=arguments.variant,
            optimizer_update=arguments.update,
            role=arguments.role,
            checkpoint=arguments.checkpoint,
            scorer_checkpoint=arguments.scorer_checkpoint,
            assets_path=arguments.assets,
            roles_path=arguments.roles,
            external_root=arguments.external_root,
            output_path=arguments.output,
            device_name=arguments.device,
        )
        return 0
    if arguments.lock is None or arguments.payload is None:
        parser.error("lock mode requires --lock and --payload")
    payload = json.loads(arguments.payload.read_text(encoding="utf-8"))
    write_immutable_lock(arguments.lock, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
