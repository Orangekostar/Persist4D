#!/usr/bin/env python3
"""Publish compact artifacts and final handoff for Persist4D All-T."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1"
FORMAL_VARIANTS = ("C0", "C1", "C2", "FH-adapt")
ALL_VARIANTS = ("C0", "C1", "C2", "C3", "FH-adapt", "FH-L")
UPDATES = (0, 100, 200, 300, 400)
HORIZONS = (2, 3, 4, 5)
TASK_METRICS = (
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
)
POLICIES = ("all_occupied", "disabled", "active_previous")
POLICY_MODELS = {
    "all_occupied": "C2",
    "disabled": "C2-memory-off",
    "active_previous": "C2-previous-only",
}
POLICY_LABELS = {
    "all_occupied": "selected_candidate_main",
    "disabled": "memory_read_disabled",
    "active_previous": "previous_stage_active_slots_only",
}
LEARNING_CURVE_FIELDS = (
    "variant",
    "checkpoint_global_step",
    "checkpoint_sha256",
    "reducer",
    "t_mAP_T2",
    "t_mAP_T3",
    "t_mAP_T4",
    "t_mAP_T5",
    "minimum_t_mAP",
    "mean_t_mAP",
    "selected_update",
)
ABLATION_FIELDS = (
    "population_id",
    "model",
    "checkpoint_sha256",
    "memory_read_policy",
    "diagnostic_label",
    "evidence_scope",
    "causal_claim",
    "reducer",
    "T",
    *TASK_METRICS,
    *(f"delta_{metric}_vs_all_occupied" for metric in TASK_METRICS),
)


class PublicationError(RuntimeError):
    """Raised when a compact artifact cannot be derived without guessing."""


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"JSON root must be an object: {path}")
    return value


def _canonical_json_sha256(value: object) -> str:
    try:
        content = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise PublicationError("value is not portable canonical JSON") from error
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PublicationError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PublicationError(f"{name} must be a sequence")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PublicationError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublicationError(f"{name} must be an integer >= {minimum}")
    return value


def _rate(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublicationError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise PublicationError(f"{name} must be a finite rate")
    return normalized


def _validate_self_hash(value: Mapping[str, object], *, name: str) -> None:
    unsigned = dict(value)
    expected = unsigned.pop("content_sha256", None)
    if expected != _canonical_json_sha256(unsigned):
        raise PublicationError(f"{name} content hash differs")


def _selection_rows(artifact_root: Path) -> dict[str, dict[str, object]]:
    path = artifact_root / "selection/selection_trace.csv"
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicationError("selection trace cannot be decoded") from error
    selected = {}
    for row in raw_rows:
        if row.get("selected_within_variant") != "True":
            continue
        variant = _text(row.get("variant"), name="selection variant")
        if variant in selected:
            raise PublicationError("selection trace has duplicate selected variants")
        try:
            step = int(str(row["checkpoint_global_step"]))
        except (KeyError, ValueError) as error:
            raise PublicationError("selected checkpoint step is invalid") from error
        selected[variant] = {
            "checkpoint_global_step": step,
            "checkpoint_sha256": _text(
                row.get("checkpoint_sha256"), name="selected checkpoint hash"
            ),
        }
    if set(selected) != set(FORMAL_VARIANTS):
        raise PublicationError("selection trace does not select every formal variant")
    return selected


def build_variant_matrix(artifact_root: Path) -> dict[str, object]:
    """Build a compact run/gate matrix from formal summaries and frozen gates."""
    budget = _load_json(artifact_root / "budget_and_schedule.json")
    _validate_self_hash(budget, name="budget and schedule")
    selection = _load_json(artifact_root / "selection/FROZEN_SELECTION.json")
    _validate_self_hash(selection, name="frozen selection")
    selected = _selection_rows(artifact_root)
    architecture = {
        "C0": (False, False, "local_pair"),
        "C1": (True, False, "local_pair"),
        "C2": (False, True, "local_pair"),
        "C3": (True, True, "local_pair"),
        "FH-adapt": (False, False, "full_history"),
        "FH-L": (True, False, "full_history"),
    }
    variants = []
    for variant in ALL_VARIANTS:
        local_enhancement, memory_read, window_mode = architecture[variant]
        common = {
            "local_enhancement_enabled": local_enhancement,
            "memory_read_enabled": memory_read,
            "variant": variant,
            "window_mode": window_mode,
        }
        if variant in FORMAL_VARIANTS:
            run_root = artifact_root / f"training/formal/{variant}"
            summary = _load_json(run_root / "run_summary.json")
            plan = _load_json(run_root / "run_plan.json")
            checkpoint_manifest = _load_json(run_root / "checkpoint_manifest.json")
            if (
                summary.get("variant") != variant
                or summary.get("execution_status") != "COMPLETE"
                or summary.get("smoke") is not False
                or summary.get("completed_global_step") != 400
                or summary.get("optimizer_updates_requested") != 400
                or plan.get("optimizer_updates") != 400
                or plan.get("seed") != 45
                or plan.get("window_mode") != window_mode
                or checkpoint_manifest.get("variant") != variant
            ):
                raise PublicationError(f"{variant} formal training evidence differs")
            checkpoint_rows = _sequence(
                checkpoint_manifest.get("checkpoints"),
                name=f"{variant} checkpoint manifest",
            )
            expected_names = {"last.ckpt", *(f"update={step:04d}.ckpt" for step in UPDATES)}
            if {row.get("name") for row in checkpoint_rows if isinstance(row, Mapping)} != expected_names:
                raise PublicationError(f"{variant} checkpoint coverage differs")
            variants.append(
                {
                    **common,
                    "checkpoint_external_reference": checkpoint_manifest.get(
                        "external_reference"
                    ),
                    "completed_global_step": 400,
                    "elapsed_seconds": summary.get("elapsed_seconds"),
                    "execution_status": "COMPLETE",
                    "gpu_hours": summary.get("gpu_hours"),
                    "gradient_audit_status": (
                        "pass"
                        if not summary["gradient_audit"]["frozen_gradient_names"]
                        and not summary["gradient_audit"]["missing_optimizer_parameters"]
                        and not summary["gradient_audit"]["nonfinite_gradient_names"]
                        else "fail"
                    ),
                    "learning_curve": (
                        f"repo:artifacts/allt_task_superiority_v1/training/formal/"
                        f"{variant}/learning_curve.csv"
                    ),
                    "selected_checkpoint_global_step": selected[variant][
                        "checkpoint_global_step"
                    ],
                    "selected_checkpoint_sha256": selected[variant][
                        "checkpoint_sha256"
                    ],
                    "total_parameter_count": summary.get("total_parameter_count"),
                    "trainable_parameter_count": summary.get(
                        "trainable_parameter_count"
                    ),
                    "training_plan": summary.get("plan"),
                }
            )
        else:
            status = _load_json(
                artifact_root / f"training/formal/{variant}/status.json"
            )
            if (
                status.get("status") != "gate_skipped"
                or status.get("variant") != variant
            ):
                raise PublicationError(f"{variant} gate evidence differs")
            variants.append(
                {
                    **common,
                    "execution_status": "GATE_SKIPPED",
                    "reason": _text(status.get("reason"), name=f"{variant} reason"),
                }
            )
    result = {
        "devices": budget.get("devices"),
        "effective_episode_batch": (
            int(budget["physical_episode_batch_per_gpu"])
            * int(budget["devices"])
            * int(budget["gradient_accumulation"])
        ),
        "evaluation_updates": budget.get("evaluation_updates"),
        "gradient_accumulation": budget.get("gradient_accumulation"),
        "optimizer_updates": budget.get("optimizer_updates"),
        "physical_episode_batch_per_gpu": budget.get(
            "physical_episode_batch_per_gpu"
        ),
        "schema_version": 1,
        "seed46_reason": (
            "development candidate did not strictly exceed matched FH-adapt at every T"
        ),
        "seed46_training_confirmation": "GATE_SKIPPED",
        "training_seed": selection.get("training_seed"),
        "variants": variants,
    }
    result["content_sha256"] = _canonical_json_sha256(result)
    return result


def _index_evaluation_rows(
    rows: Sequence[Mapping[str, object]], *, model: str
) -> tuple[str, dict[int, Mapping[str, object]]]:
    values = _sequence(rows, name=f"{model} evaluation rows")
    indexed = {}
    checkpoints = set()
    reducers = set()
    for row in values:
        if not isinstance(row, Mapping) or row.get("model") != model:
            raise PublicationError(f"{model} evaluation model differs")
        horizon = _integer(row.get("T"), name=f"{model} T", minimum=2)
        if horizon in indexed:
            raise PublicationError(f"{model} evaluation has duplicate horizons")
        if horizon not in HORIZONS:
            raise PublicationError(f"{model} evaluation horizon differs")
        checkpoint = _text(
            row.get("checkpoint_sha256"), name=f"{model} checkpoint"
        )
        checkpoints.add(checkpoint)
        reducers.add(_text(row.get("reducer"), name=f"{model} reducer"))
        for metric in TASK_METRICS:
            _rate(row.get(metric), name=f"{model} {metric}")
        indexed[horizon] = row
    if set(indexed) != set(HORIZONS) or len(checkpoints) != 1 or len(reducers) != 1:
        raise PublicationError(f"{model} evaluation coverage differs")
    return next(iter(reducers)), indexed


def build_learning_curve_rows(
    variant: str,
    evaluations: Mapping[int, Sequence[Mapping[str, object]]],
    *,
    selected_step: int,
    selected_checkpoint: str,
) -> list[dict[str, object]]:
    """Collapse five development evaluations into a checkpoint-level curve."""
    if set(evaluations) != set(UPDATES):
        raise PublicationError("learning curve updates differ")
    if selected_step not in UPDATES:
        raise PublicationError("selected learning-curve update differs")
    output = []
    for update in UPDATES:
        reducer, rows = _index_evaluation_rows(evaluations[update], model=variant)
        checkpoints = {str(row["checkpoint_sha256"]) for row in rows.values()}
        checkpoint = next(iter(checkpoints))
        values = [float(rows[horizon]["t_mAP"]) for horizon in HORIZONS]
        selected = update == selected_step
        if selected and checkpoint != selected_checkpoint:
            raise PublicationError("selected learning-curve checkpoint differs")
        output.append(
            {
                "variant": variant,
                "checkpoint_global_step": update,
                "checkpoint_sha256": checkpoint,
                "reducer": reducer,
                **{
                    f"t_mAP_T{horizon}": float(rows[horizon]["t_mAP"])
                    for horizon in HORIZONS
                },
                "minimum_t_mAP": min(values),
                "mean_t_mAP": sum(values) / len(values),
                "selected_update": selected,
            }
        )
    return output


def build_memory_ablation_rows(
    evaluations: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    checkpoint_sha256: str,
) -> list[dict[str, object]]:
    """Build a three-policy development diagnostic without a causal claim."""
    if set(evaluations) != set(POLICIES):
        raise PublicationError("memory ablation policies differ")
    indexed = {}
    for policy in POLICIES:
        model = POLICY_MODELS[policy]
        reducer, rows = _index_evaluation_rows(evaluations[policy], model=model)
        if reducer != "mean" or any(
            row["checkpoint_sha256"] != checkpoint_sha256 for row in rows.values()
        ):
            raise PublicationError("memory ablation checkpoint or reducer differs")
        indexed[policy] = rows
    output = []
    for policy in POLICIES:
        for horizon in HORIZONS:
            current = indexed[policy][horizon]
            main = indexed["all_occupied"][horizon]
            output.append(
                {
                    "population_id": current["population_id"],
                    "model": POLICY_MODELS[policy],
                    "checkpoint_sha256": checkpoint_sha256,
                    "memory_read_policy": policy,
                    "diagnostic_label": POLICY_LABELS[policy],
                    "evidence_scope": "development_diagnostic",
                    "causal_claim": "not_established",
                    "reducer": "mean",
                    "T": horizon,
                    **{metric: float(current[metric]) for metric in TASK_METRICS},
                    **{
                        f"delta_{metric}_vs_all_occupied": (
                            float(current[metric]) - float(main[metric])
                        )
                        for metric in TASK_METRICS
                    },
                }
            )
    return output


def _read_evaluation_csv(path: Path) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicationError(f"evaluation CSV cannot be decoded: {path}") from error
    rows = []
    for raw in raw_rows:
        row: dict[str, object] = dict(raw)
        try:
            for field in (
                "training_seed",
                "evaluation_seed",
                "T",
                "num_master",
                "num_order_units",
                "num_reference_clusters",
            ):
                row[field] = int(raw[field])
            for field in (*TASK_METRICS, "local_current_AP"):
                row[field] = float(raw[field])
        except (KeyError, ValueError) as error:
            raise PublicationError(f"evaluation CSV values differ: {path}") from error
        rows.append(row)
    return rows


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise PublicationError("published CSV fields differ")
        writer.writerow(row)
    return stream.getvalue().encode("ascii")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _validate_diagnostic(
    artifact_root: Path, *, policy: str, directory: str
) -> list[dict[str, object]]:
    root = artifact_root / f"evaluation/ablation/{directory}/update=0200"
    metrics_path = root / "all_t_metrics.csv"
    summary = _load_json(root / "run_summary.json")
    manifest = _load_json(root / "cache_manifest.json")
    _validate_self_hash(manifest, name=f"{policy} cache manifest")
    expected = {
        "checkpoint_sha256": (
            "a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724"
        ),
        "evaluation_seed": 45,
        "memory_read_policy": policy,
        "model": POLICY_MODELS[policy],
        "population_id": "development_train_holdout_47_masters_canonical",
        "reducers": ["mean"],
        "sequence_count": 47,
        "status": "pass",
        "training_seed": 45,
        "variant": "C2",
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise PublicationError(f"{policy} diagnostic summary {field} differs")
    if summary.get("metric_sha256") != _file_sha256(metrics_path):
        raise PublicationError(f"{policy} diagnostic metric hash differs")
    if (
        manifest.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or manifest.get("memory_read_policy") != policy
        or manifest.get("entry_count") != 47
        or manifest.get("status") != "pass"
        or manifest.get("module_config", {}).get(
            "evaluation_memory_read_policy"
        ) != policy
    ):
        raise PublicationError(f"{policy} diagnostic cache binding differs")
    return _read_evaluation_csv(metrics_path)


def publish_compact_exports(artifact_root: Path) -> list[dict[str, str]]:
    """Write the variant matrix, four learning curves, and memory ablation."""
    matrix = build_variant_matrix(artifact_root)
    matrix_path = artifact_root / "training/variant_matrix.json"
    _atomic_json(matrix_path, matrix)
    selected = _selection_rows(artifact_root)
    outputs = [matrix_path]
    for variant in FORMAL_VARIANTS:
        evaluations = {
            update: _read_evaluation_csv(
                artifact_root
                / f"evaluation/development/{variant}/update={update:04d}/all_t_metrics.csv"
            )
            for update in UPDATES
        }
        rows = build_learning_curve_rows(
            variant,
            evaluations,
            selected_step=int(selected[variant]["checkpoint_global_step"]),
            selected_checkpoint=str(selected[variant]["checkpoint_sha256"]),
        )
        path = artifact_root / f"training/formal/{variant}/learning_curve.csv"
        _atomic_write(path, _csv_bytes(rows, LEARNING_CURVE_FIELDS))
        outputs.append(path)
    main = _read_evaluation_csv(
        artifact_root / "evaluation/development/C2/update=0200/all_t_metrics.csv"
    )
    diagnostics = {
        "all_occupied": main,
        "disabled": _validate_diagnostic(
            artifact_root, policy="disabled", directory="C2-memory-off"
        ),
        "active_previous": _validate_diagnostic(
            artifact_root, policy="active_previous", directory="C2-previous-only"
        ),
    }
    ablation = build_memory_ablation_rows(
        diagnostics,
        checkpoint_sha256=(
            "a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724"
        ),
    )
    ablation_path = artifact_root / "results/memory_read_ablation.csv"
    _atomic_write(ablation_path, _csv_bytes(ablation, ABLATION_FIELDS))
    outputs.append(ablation_path)
    return [
        {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": _file_sha256(path),
        }
        for path in outputs
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = publish_compact_exports(args.artifact_root.expanduser().resolve())
    print(json.dumps({"outputs": outputs, "status": "pass"}, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "TASK_METRICS",
    "PublicationError",
    "build_learning_curve_rows",
    "build_memory_ablation_rows",
    "build_variant_matrix",
    "publish_compact_exports",
]


if __name__ == "__main__":
    raise SystemExit(main())
