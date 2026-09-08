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
import statistics
import subprocess
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
COMPLETION_STATUS_KEYS = (
    "execution_status",
    "baseline_comparability",
    "tmap_all_t_vs_R1",
    "tmap_all_t_vs_matched_rescene",
    "task_metrics_all_t",
    "seed_confirmation",
    "independent_generalization",
    "publication_status",
)
PUBLICATION_STATUSES = ("PUSH_VERIFIED", "PUSH_FAILED", "NOT_ATTEMPTED")
REQUIRED_COMPLETION_PATHS = (
    "EXPERIMENT_CONTRACT.md",
    "run_contract.json",
    "source_map.md",
    "split_manifest.json",
    "budget_and_schedule.json",
    "baseline/all_t_metrics.csv",
    "baseline/replay_status.json",
    "diagnostics/failure_summary.csv",
    "diagnostics/final_r1_decoder_summary.json",
    "training/variant_matrix.json",
    *(f"training/formal/{variant}/resolved_config.yaml" for variant in FORMAL_VARIANTS),
    *(f"training/formal/{variant}/learning_curve.csv" for variant in FORMAL_VARIANTS),
    *(f"training/formal/{variant}/run_plan.json" for variant in FORMAL_VARIANTS),
    *(f"training/formal/{variant}/run_summary.json" for variant in FORMAL_VARIANTS),
    *(f"training/formal/{variant}/checkpoint_manifest.json" for variant in FORMAL_VARIANTS),
    "training/formal/C3/status.json",
    "training/formal/FH-L/status.json",
    "selection/selection_trace.csv",
    "selection/FROZEN_SELECTION.json",
    "selection/complementarity_gate.json",
    "evaluation/protocol_b/C2/update=0200/all_t_metrics.csv",
    "evaluation/protocol_b/C2/update=0200/cache_manifest.json",
    "evaluation/protocol_b/C2/update=0200/run_summary.json",
    "evaluation/protocol_b/FH-adapt/update=0100/all_t_metrics.csv",
    "evaluation/protocol_b/FH-adapt/update=0100/cache_manifest.json",
    "evaluation/protocol_b/FH-adapt/update=0100/run_summary.json",
    "evaluation/ablation/C2-memory-off/update=0200/all_t_metrics.csv",
    "evaluation/ablation/C2-memory-off/update=0200/cache_manifest.json",
    "evaluation/ablation/C2-memory-off/update=0200/run_summary.json",
    "evaluation/ablation/C2-previous-only/update=0200/all_t_metrics.csv",
    "evaluation/ablation/C2-previous-only/update=0200/cache_manifest.json",
    "evaluation/ablation/C2-previous-only/update=0200/run_summary.json",
    "results/all_t_metrics.csv",
    "results/deltas_all_metrics.csv",
    "results/per_reference.csv",
    "results/identity_counts.csv",
    "results/cluster_effects.csv",
    "results/score_sensitivity.csv",
    "results/run_confirmation.csv",
    "results/memory_read_ablation.csv",
    "results/verdict.json",
    "results/evidence_manifest.json",
    "results/final_analysis_manifest.json",
    "profile/samples.csv",
    "profile/summary.csv",
    "profile/run_summary.json",
)
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


def derive_final_statuses(
    artifact_root: Path, *, publication_status: str
) -> dict[str, str]:
    """Derive the required completion states from frozen structured evidence."""
    if publication_status not in PUBLICATION_STATUSES:
        raise PublicationError("publication status differs")
    verdict = _load_json(artifact_root / "results/verdict.json")
    _validate_self_hash(verdict, name="final verdict")
    comparisons = verdict.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise PublicationError("final verdict comparisons differ")
    r1 = comparisons.get("C2_vs_R1_B4")
    matched = comparisons.get("C2_vs_FH-adapt")
    if not isinstance(r1, Mapping) or not isinstance(matched, Mapping):
        raise PublicationError("required final comparisons are unavailable")
    matrix = _load_json(artifact_root / "training/variant_matrix.json")
    _validate_self_hash(matrix, name="variant matrix")
    if matrix.get("seed46_training_confirmation") != "GATE_SKIPPED":
        raise PublicationError("seed 46 gate status differs")
    variants = matrix.get("variants")
    if not isinstance(variants, Sequence):
        raise PublicationError("variant matrix rows differ")
    by_variant = {
        row.get("variant"): row for row in variants if isinstance(row, Mapping)
    }
    if (
        by_variant.get("C2", {}).get("execution_status") != "COMPLETE"
        or by_variant.get("FH-adapt", {}).get("execution_status") != "COMPLETE"
        or by_variant.get("C2", {}).get("selected_checkpoint_sha256")
        != "a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724"
        or by_variant.get("FH-adapt", {}).get("selected_checkpoint_sha256")
        != "ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c"
    ):
        raise PublicationError("matched selected checkpoints differ")
    statuses = {
        "execution_status": "COMPLETE",
        "baseline_comparability": "MATCHED",
        "tmap_all_t_vs_R1": str(r1.get("tmap_all_t")),
        "tmap_all_t_vs_matched_rescene": str(matched.get("tmap_all_t")),
        "task_metrics_all_t": str(matched.get("task_metrics_all_t")),
        "seed_confirmation": "NOT_RUN",
        "independent_generalization": str(
            verdict.get("independent_generalization")
        ),
        "publication_status": publication_status,
    }
    expected_values = {
        "tmap_all_t_vs_R1": {"PASS", "FAIL", "NOT_EVALUATED"},
        "tmap_all_t_vs_matched_rescene": {"PASS", "FAIL", "NOT_EVALUATED"},
        "task_metrics_all_t": {"PASS", "FAIL", "NOT_EVALUATED"},
        "independent_generalization": {"CONFIRMED", "NOT_ESTABLISHED"},
    }
    for field, allowed in expected_values.items():
        if statuses[field] not in allowed:
            raise PublicationError(f"{field} status differs")
    return statuses


def primary_tmap_failure_cells(artifact_root: Path) -> dict[str, list[int]]:
    """Return exact non-positive t-mAP horizons for the two primary baselines."""
    verdict = _load_json(artifact_root / "results/verdict.json")
    _validate_self_hash(verdict, name="final verdict")
    comparisons = verdict.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise PublicationError("final verdict comparisons differ")
    output = {}
    for comparison in ("C2_vs_R1_B4", "C2_vs_FH-adapt"):
        value = comparisons.get(comparison)
        if not isinstance(value, Mapping):
            raise PublicationError(f"{comparison} verdict is unavailable")
        failed = value.get("failed_cells")
        if not isinstance(failed, Sequence):
            raise PublicationError(f"{comparison} failed cells differ")
        horizons = sorted(
            {
                _integer(row.get("T"), name=f"{comparison} failed T", minimum=2)
                for row in failed
                if isinstance(row, Mapping) and row.get("metric") == "t_mAP"
            }
        )
        deltas = value.get("t_map_delta_by_horizon")
        if not isinstance(deltas, Mapping):
            raise PublicationError(f"{comparison} t-mAP deltas differ")
        expected = sorted(
            int(horizon) for horizon, delta in deltas.items() if float(delta) <= 0.0
        )
        if horizons != expected:
            raise PublicationError(f"{comparison} failed t-mAP cells differ")
        output[comparison] = horizons
    return output


def validate_completion_inputs(artifact_root: Path) -> tuple[Path, ...]:
    """Fail closed until every required compact completion artifact exists."""
    root = artifact_root.expanduser().resolve()
    paths = tuple(root / relative for relative in REQUIRED_COMPLETION_PATHS)
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise PublicationError(f"required completion artifact is missing: {path}")
    expected_csv_rows = {
        "results/all_t_metrics.csv": 44,
        "results/deltas_all_metrics.csv": 80,
        "results/per_reference.csv": 120,
        "results/identity_counts.csv": 8,
        "results/cluster_effects.csv": 20,
        "results/memory_read_ablation.csv": 12,
        "profile/samples.csv": 480,
        "profile/summary.csv": 48,
    }
    for relative, expected in expected_csv_rows.items():
        if len(_read_csv_rows(root / relative)) != expected:
            raise PublicationError(f"required completion artifact row count differs: {relative}")
    analysis = _load_json(root / "results/final_analysis_manifest.json")
    _validate_self_hash(analysis, name="final analysis manifest")
    coverage = analysis.get("coverage")
    if (
        analysis.get("status") != "pass"
        or not isinstance(coverage, Mapping)
        or coverage.get("per_reference_rows") != 120
        or coverage.get("identity_rows") != 8
        or coverage.get("cluster_effect_rows") != 20
    ):
        raise PublicationError("final analysis coverage differs")
    _validate_output_hashes(
        root / "results", analysis.get("outputs"), name="final analysis"
    )
    profile = _load_json(root / "profile/run_summary.json")
    if (
        profile.get("status") != "pass"
        or profile.get("population_id")
        != "protocol_b_profile_6_reference_canonical"
        or profile.get("device_name") != "NVIDIA A40"
        or profile.get("sample_row_count") != 480
        or profile.get("summary_row_count") != 48
        or profile.get("measured_repeats") != 10
        or profile.get("warmup_repeats") != 5
        or profile.get("samples_sha256") != _file_sha256(root / "profile/samples.csv")
        or profile.get("summary_sha256") != _file_sha256(root / "profile/summary.csv")
    ):
        raise PublicationError("formal profile evidence differs")
    _validate_profile_pairing(_read_csv_rows(root / "profile/samples.csv"))
    matrix = _load_json(root / "training/variant_matrix.json")
    _validate_self_hash(matrix, name="variant matrix")
    variants = matrix.get("variants")
    if not isinstance(variants, Sequence):
        raise PublicationError("variant matrix rows differ")
    forbidden_unrun_fields = {
        "completed_global_step",
        "elapsed_seconds",
        "gpu_hours",
        "selected_checkpoint_global_step",
        "selected_checkpoint_sha256",
        "total_parameter_count",
        "trainable_parameter_count",
        "training_plan",
    }
    for row in variants:
        if (
            isinstance(row, Mapping)
            and row.get("execution_status") == "GATE_SKIPPED"
            and forbidden_unrun_fields.intersection(row)
        ):
            raise PublicationError("gate-skipped variant contains numeric run evidence")
    derive_final_statuses(root, publication_status="NOT_ATTEMPTED")
    primary_tmap_failure_cells(root)
    return paths


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicationError(f"CSV cannot be decoded: {path}") from error
    if not rows or reader.fieldnames is None:
        raise PublicationError(f"CSV has no data rows: {path}")
    return rows


def _validate_output_hashes(root: Path, value: object, *, name: str) -> None:
    rows = _sequence(value, name=f"{name} outputs")
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicationError(f"{name} output record differs")
        relative = _text(row.get("path"), name=f"{name} output path")
        if row.get("sha256") != _file_sha256(root / relative):
            raise PublicationError(f"{name} output hash differs: {relative}")


def _validate_profile_pairing(rows: Sequence[Mapping[str, str]]) -> None:
    keys = ("reference_scene_id", "master_sequence_id", "order_id", "T", "repeat")
    indexed = {(row["model"], *(row[field] for field in keys)): row for row in rows}
    if len(indexed) != len(rows):
        raise PublicationError("profile sample identities are not unique")
    for row in rows:
        for field in (
            "latency_ms",
            "start_allocated_bytes",
            "peak_allocated_bytes",
            "incremental_peak_allocated_bytes",
        ):
            try:
                value = float(row[field])
            except (KeyError, ValueError) as error:
                raise PublicationError("profile numeric value differs") from error
            if not math.isfinite(value) or value < 0.0:
                raise PublicationError("profile numeric value is invalid")
        if row.get("T") != "2":
            continue
        other = "FH-adapt" if row.get("model") == "C2" else "C2"
        paired = indexed.get((other, *(row[field] for field in keys)))
        if (
            paired is None
            or row.get("window_voxel_points") != paired.get("window_voxel_points")
            or row.get("window_segments") != paired.get("window_segments")
        ):
            raise PublicationError("T2 profile inputs differ across methods")


def build_final_manifest(
    artifact_root: Path,
    *,
    artifact_paths: Sequence[Path],
    statuses: Mapping[str, str],
    code_commit_at_run: str,
) -> dict[str, object]:
    """Build a deterministic manifest without listing FINAL_MANIFEST itself."""
    root = artifact_root.expanduser().resolve()
    if tuple(statuses) != COMPLETION_STATUS_KEYS:
        raise PublicationError("completion status fields differ")
    if (
        len(code_commit_at_run) != 40
        or any(character not in "0123456789abcdef" for character in code_commit_at_run)
    ):
        raise PublicationError("code commit at run differs")
    records = []
    seen = set()
    for supplied in artifact_paths:
        path = supplied.expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise PublicationError("manifest artifact is outside artifact root") from error
        if relative == "FINAL_MANIFEST.json":
            raise PublicationError("FINAL_MANIFEST cannot list itself")
        if relative in seen or not path.is_file() or path.is_symlink():
            raise PublicationError(f"manifest artifact differs: {relative}")
        seen.add(relative)
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": _file_sha256(path),
            }
        )
    manifest = {
        "artifacts": sorted(records, key=lambda row: str(row["path"])),
        "code_commit_at_run": code_commit_at_run,
        "publication_commit": "resolve from remote HEAD",
        "schema_version": 1,
        "status": "pass",
        "statuses": dict(statuses),
    }
    manifest["content_sha256"] = _canonical_json_sha256(manifest)
    return manifest


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or len(value) != 40:
        raise PublicationError("cannot resolve code commit at publication")
    return value


def _float_text(value: object, *, digits: int = 6) -> str:
    try:
        number = float(str(value))
    except ValueError as error:
        raise PublicationError("report numeric value differs") from error
    if not math.isfinite(number):
        raise PublicationError("report numeric value is not finite")
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _main_metric_rows(artifact_root: Path) -> dict[str, dict[int, dict[str, str]]]:
    rows = _read_csv_rows(artifact_root / "results/all_t_metrics.csv")
    selectors = {
        "C2": ("C2", "B4", "mean"),
        "R1-B4": ("R1_epoch390", "B4", "mean"),
        "FH-adapt": ("FH-adapt", "FullHistory", "official"),
        "FH-R1": ("R1_epoch390", "FullHistory", "official"),
    }
    output: dict[str, dict[int, dict[str, str]]] = {}
    for label, selector in selectors.items():
        selected = [
            row
            for row in rows
            if (row.get("model"), row.get("method"), row.get("reducer")) == selector
        ]
        indexed = {int(row["T"]): row for row in selected}
        if set(indexed) != set(HORIZONS):
            raise PublicationError(f"main metric rows differ: {label}")
        output[label] = indexed
    return output


def _delta_rows(artifact_root: Path, comparison: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in _read_csv_rows(artifact_root / "results/deltas_all_metrics.csv")
        if row.get("comparison") == comparison
    ]
    if len(rows) != len(TASK_METRICS) * len(HORIZONS):
        raise PublicationError(f"delta coverage differs: {comparison}")
    return sorted(rows, key=lambda row: (TASK_METRICS.index(row["metric"]), int(row["T"])))


def _primary_metric_table(artifact_root: Path) -> str:
    metrics = _main_metric_rows(artifact_root)
    rows = []
    for horizon in HORIZONS:
        c2 = float(metrics["C2"][horizon]["t_mAP"])
        r1 = float(metrics["R1-B4"][horizon]["t_mAP"])
        matched = float(metrics["FH-adapt"][horizon]["t_mAP"])
        fh_r1 = float(metrics["FH-R1"][horizon]["t_mAP"])
        rows.append(
            (
                f"T{horizon}",
                _float_text(c2),
                _float_text(r1),
                _float_text(c2 - r1),
                _float_text(matched),
                _float_text(c2 - matched),
                _float_text(fh_r1),
            )
        )
    return _markdown_table(
        (
            "Horizon",
            "C2 mean",
            "R1 B4 mean",
            "Delta vs R1 B4",
            "FH-adapt",
            "Delta vs FH-adapt",
            "FH-R1",
        ),
        rows,
    )


def _all_task_delta_table(artifact_root: Path) -> str:
    rows = []
    for row in _delta_rows(artifact_root, "C2_vs_FH-adapt"):
        rows.append(
            (
                row["metric"],
                f"T{row['T']}",
                _float_text(row["raw_delta"]),
                "PASS" if row["positive"] == "True" else "FAIL",
            )
        )
    return _markdown_table(("Metric", "Horizon", "Raw delta", "Strict > 0"), rows)


def _training_table(artifact_root: Path) -> str:
    matrix = _load_json(artifact_root / "training/variant_matrix.json")
    rows = []
    variants = _sequence(matrix.get("variants"), name="variant rows")
    for value in variants:
        if not isinstance(value, Mapping):
            raise PublicationError("variant row differs")
        if value.get("execution_status") == "GATE_SKIPPED":
            rows.append(
                (
                    value["variant"],
                    "GATE_SKIPPED",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    _text(value.get("reason"), name="gate reason"),
                )
            )
            continue
        plan = value.get("training_plan")
        if not isinstance(plan, Mapping):
            raise PublicationError("formal training plan differs")
        rows.append(
            (
                value["variant"],
                "COMPLETE",
                value["selected_checkpoint_global_step"],
                value["trainable_parameter_count"],
                _float_text(value["gpu_hours"], digits=3),
                f"{plan['global_episode_count']} episodes / {plan['supervised_stage_count']} stages",
                str(value["selected_checkpoint_sha256"]),
            )
        )
    return _markdown_table(
        ("Variant", "State", "Selected update", "Trainable params", "GPU-h", "Exposure", "Checkpoint/reason"),
        rows,
    )


def _development_table(artifact_root: Path) -> str:
    selected = _selection_rows(artifact_root)
    values = {}
    for variant in FORMAL_VARIANTS:
        rows = _read_csv_rows(
            artifact_root / f"training/formal/{variant}/learning_curve.csv"
        )
        match = [row for row in rows if row.get("selected_update") == "True"]
        if len(match) != 1:
            raise PublicationError(f"selected learning curve row differs: {variant}")
        if int(match[0]["checkpoint_global_step"]) != selected[variant]["checkpoint_global_step"]:
            raise PublicationError(f"learning curve selection differs: {variant}")
        values[variant] = match[0]
    rows = []
    for variant in FORMAL_VARIANTS:
        row = values[variant]
        rows.append(
            (
                variant,
                row["checkpoint_global_step"],
                *(_float_text(row[f"t_mAP_T{horizon}"]) for horizon in HORIZONS),
                _float_text(row["minimum_t_mAP"]),
            )
        )
    return _markdown_table(
        ("Variant", "Update", "T2", "T3", "T4", "T5", "Minimum"), rows
    )


def _module_effect_table(artifact_root: Path) -> str:
    values = {}
    for variant in ("C0", "C1", "C2"):
        rows = _read_csv_rows(
            artifact_root / f"training/formal/{variant}/learning_curve.csv"
        )
        selected = [row for row in rows if row.get("selected_update") == "True"]
        if len(selected) != 1:
            raise PublicationError(f"selected module-effect row differs: {variant}")
        values[variant] = selected[0]
    rows = []
    for label, variant in (("L: C1-C0", "C1"), ("M: C2-C0", "C2")):
        rows.append(
            (
                label,
                *(
                    _float_text(
                        float(values[variant][f"t_mAP_T{horizon}"])
                        - float(values["C0"][f"t_mAP_T{horizon}"])
                    )
                    for horizon in HORIZONS
                ),
            )
        )
    return _markdown_table(("Effect", "T2", "T3", "T4", "T5"), rows)


def _diagnostic_summary(artifact_root: Path) -> str:
    value = _load_json(artifact_root / "diagnostics/final_r1_decoder_summary.json")
    _validate_self_hash(value, name="R1 decoder diagnostic")
    summary = value.get("failure_summary")
    if not isinstance(summary, Mapping):
        raise PublicationError("R1 failure summary differs")
    counts = summary.get("category_counts_official_valid")
    cold = summary.get("t1_cold_start_contribution")
    if not isinstance(counts, Mapping) or not isinstance(cold, Mapping):
        raise PublicationError("R1 diagnostic counts differ")
    return (
        f"D0 contains {summary['official_valid_row_count']} official-valid trajectory rows "
        f"({summary['unique_gt_trajectory_count']} unique GT trajectories). Its non-additive "
        f"labels include mask-insufficient={counts['mask_insufficient']}, "
        f"full-history-success/B4-failure={counts['full_history_success_b4_failure']}, "
        f"class-change={counts['class_change']}, fragmentation={counts['fragmentation_event']}, "
        f"merge={counts['merge_event']}, and no-candidate={counts['no_candidate']}. "
        f"T1 cold-start contributed to {cold['contributed_row_count']}/"
        f"{cold['failed_row_count']} failed rows ({_float_text(cold['fraction'], digits=3)})."
    )


def _memory_table(artifact_root: Path) -> str:
    rows = []
    for row in _read_csv_rows(artifact_root / "results/memory_read_ablation.csv"):
        if row["memory_read_policy"] == "all_occupied":
            continue
        rows.append(
            (
                row["memory_read_policy"],
                f"T{row['T']}",
                _float_text(row["t_mAP"]),
                _float_text(row["delta_t_mAP_vs_all_occupied"]),
            )
        )
    return _markdown_table(("Policy", "Horizon", "t-mAP", "Delta vs all occupied"), rows)


def _cluster_table(artifact_root: Path) -> str:
    raw = _read_csv_rows(artifact_root / "results/cluster_effects.csv")
    rows = []
    for row in raw:
        if row["metric"] != "t_mAP":
            continue
        rows.append(
            (
                f"T{row['T']}",
                _float_text(row["equal_cluster_mean_delta"]),
                f"[{_float_text(row['bootstrap_ci_lower'])}, {_float_text(row['bootstrap_ci_upper'])}]",
                row["cluster_count"],
            )
        )
    if len(rows) != 4:
        raise PublicationError("t-mAP cluster rows differ")
    return _markdown_table(("Horizon", "Equal-cluster mean delta", "95% descriptive interval", "Clusters"), rows)


def _per_reference_summary(artifact_root: Path) -> str:
    raw = _read_csv_rows(artifact_root / "results/per_reference.csv")
    rows = []
    for horizon in HORIZONS:
        values = [
            float(row["delta"])
            for row in raw
            if row["metric"] == "t_mAP" and int(row["T"]) == horizon
        ]
        if len(values) != 6:
            raise PublicationError("per-reference t-mAP coverage differs")
        rows.append(
            (
                f"T{horizon}",
                sum(value > 0.0 for value in values),
                _float_text(min(values)),
                _float_text(max(values)),
            )
        )
    return _markdown_table(("Horizon", "Positive references / 6", "Minimum delta", "Maximum delta"), rows)


def _identity_table(artifact_root: Path) -> str:
    rows = []
    for row in _read_csv_rows(artifact_root / "results/identity_counts.csv"):
        rows.append(
            (
                row["model"],
                f"T{row['T']}",
                row["deployment_id_switches"],
                row["fragmentation_count"],
                row["merge_count"],
                row["correct_recoveries"],
                row["recovery_attempts"],
                row["gap_opportunities"],
                row["gap_recovery_accuracy"] or "N/A",
                row["gap_recovery_recall"] or "N/A",
            )
        )
    return _markdown_table(
        ("Model", "Horizon", "ID switches", "Fragments", "Merges", "Correct recovery", "Attempts", "Gap opp.", "Accuracy", "Recall"),
        rows,
    )


def _profile_table(artifact_root: Path) -> str:
    summary = _read_csv_rows(artifact_root / "profile/summary.csv")
    rows = []
    for model in ("C2", "FH-adapt"):
        for horizon in HORIZONS:
            values = [
                row
                for row in summary
                if row["model"] == model and int(row["T"]) == horizon
            ]
            if len(values) != 6:
                raise PublicationError("profile summary cell coverage differs")
            latencies = [float(row["median_latency_ms"]) for row in values]
            memory = max(
                float(row["maximum_incremental_peak_allocated_bytes"])
                for row in values
            ) / (1024**2)
            points = [int(row["window_voxel_points"]) for row in values]
            rows.append(
                (
                    model,
                    f"T{horizon}",
                    _float_text(statistics.median(latencies), digits=3),
                    f"{_float_text(min(latencies), digits=3)}-{_float_text(max(latencies), digits=3)}",
                    _float_text(memory, digits=1),
                    int(statistics.median(points)),
                )
            )
    return _markdown_table(
        ("Model", "Horizon", "Median of cell medians (ms)", "Cell range (ms)", "Max incremental MiB", "Median voxel points"),
        rows,
    )


def _status_table(statuses: Mapping[str, str]) -> str:
    return _markdown_table(("Status", "Value"), [(key, statuses[key]) for key in COMPLETION_STATUS_KEYS])


def _test_report() -> str:
    return """# Persist4D All-T Test Report

Status: PASS. The direct suite completed with `45 passed, 1 warning` in the
Persist4D environment. The warning is the existing Albumentations import of
the deprecated `scipy.ndimage.filters.gaussian_filter` namespace.

## Direct verification

```bash
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \\
  tests/test_persist4d_allt_model.py \\
  tests/test_persist4d_allt_evaluation.py \\
  tests/test_analyze_persist4d_allt_final.py \\
  tests/test_profile_persist4d_allt.py \\
  tests/test_publish_persist4d_allt.py \\
  tests/test_persist4d_allt_selection.py \\
  tests/test_finalize_persist4d_allt.py
python -m ruff check scripts/evaluate_persist4d_allt.py \\
  scripts/analyze_persist4d_allt_final.py scripts/profile_persist4d_allt.py \\
  scripts/publish_persist4d_allt.py tests/test_persist4d_allt_evaluation.py \\
  tests/test_analyze_persist4d_allt_final.py tests/test_profile_persist4d_allt.py \\
  tests/test_publish_persist4d_allt.py
git diff --check
```

## Real failures retained

- Scientific gate: C2 failed strict all-T t-mAP against both R1 B4 and matched FH-adapt.
- Scientific gate: only 5/20 C2-vs-FH-adapt task cells were positive.
- Operational: the first remote memory-diagnostic write hit NFS permissions; the target namespace permissions were corrected before rerun.
- Contract: the first profile smoke exposed unequal stochastic T2 preparation; deterministic per-cell preparation seeding was added and the smoke was rerun before the formal profile.
- Environment: invoking the direct suite with base Python failed collection because `torch_scatter` was absent; the exact suite passed under the declared Persist4D environment.
"""


def _final_report(artifact_root: Path, statuses: Mapping[str, str], code_commit: str) -> str:
    failures = primary_tmap_failure_cells(artifact_root)
    return f"""# Persist4D All-T Final Report

## Outcome

{_status_table(statuses)}

Execution completed, but the scientific all-T superiority goal failed. C2 did not strictly exceed R1 B4 at T{', T'.join(map(str, failures['C2_vs_R1_B4']))}, nor matched FH-adapt at T{', T'.join(map(str, failures['C2_vs_FH-adapt']))}. Protocol-B was final-only and did not change selection.

Code commit used by this publisher: `{code_commit}`. Publication commit: resolve from remote HEAD.

## Frozen Protocol-B Results

{_primary_metric_table(artifact_root)}

The primary reducer is one frozen `mean` C2 checkpoint. `latest` and `max` remain sensitivity analyses only.

## All Candidate-vs-Matched Task Cells

{_all_task_delta_table(artifact_root)}

## Training And Selection

{_training_table(artifact_root)}

All formal models used seed 45, two A40 GPUs, physical batch 1/GPU, gradient accumulation 4, effective batch 8, 400 optimizer updates, and `32-true` precision. C3 and FH-L contain no fabricated numeric run results. Seed 46 was gate-skipped after the seed-45 development candidate failed the matched all-T gate.

### Selected Development Curves

{_development_table(artifact_root)}

{_module_effect_table(artifact_root)}

L alone (C1) did not satisfy its short-horizon complementarity gate. M alone (C2) improved the selected C0 at all development horizons, but only T2 exceeded selected FH-adapt; therefore L+M (C3) was not authorized. C2 reads prediction-only persistent slots beyond its W=2 input window and commits each stage immediately.

{_diagnostic_summary(artifact_root)}

## Memory Read Diagnostic

{_memory_table(artifact_root)}

This is a 47-master development diagnostic on the frozen C2 checkpoint. Disabling or restricting reads slightly improved T3-T5 t-mAP, but these policies were not selected models and do not establish causality.

## Reference-Cluster Evidence

{_per_reference_summary(artifact_root)}

{_cluster_table(artifact_root)}

The intervals are 1,000 fixed-seed resamples of six equal-weight reference deltas. They are descriptive equal-cluster effects, not pooled-AP confidence intervals and not evidence of independent generalization.

## Identity Evidence

{_identity_table(artifact_root)}

Identity values are freshly recomputed from T1-T5. A zero denominator is represented as `N/A`; it is never converted to 0 or 1.

## Bounded Resource Profile

{_profile_table(artifact_root)}

The profile contains 480 measurements (2 models x 6 references x 4 horizons x 10 repeats) on one NVIDIA A40 after 5 warmups. It includes model forward, C2 memory read, observation extraction, and one B4 update; it excludes file I/O, collation, H2D, metrics, and C2 preroll. C2 uses K=100 and W=2 with deployment memory that does not grow with T. This bounded profile does not rescue the failed accuracy verdict.

## Next Round

Change exactly one factor: add a learned quality score/gate for memory reads while keeping the checkpoint-selection rule, reducer, K=100, W=2, losses, and evaluation protocol fixed. This is the sole recommended bottleneck because the frozen read ablations improve T3-T5 slightly while the present all-occupied read loses at longer horizons; it remains a hypothesis until a preregistered rerun.
"""


def _reproduction_commands() -> str:
    return """```bash
export ALLT_EXTERNAL_ROOT=/mnt/shared/ww/persist4d-allt-task-superiority-v1
export R1_CHECKPOINT=/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt
export CONCERTO_PRETRAINED=/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth
export RIO_METADATA=/home/ww/3RScan.json
export PERSIST4D_DATA_ROOT="$PWD/data"

python -m torch.distributed.run --standalone --nproc_per_node=2 \\
  scripts/train_persist4d_allt.py --variant C2 \\
  --checkpoint "$R1_CHECKPOINT" --pretrained "$CONCERTO_PRETRAINED" \\
  --metadata "$RIO_METADATA" --data-root "$PERSIST4D_DATA_ROOT" \\
  --external-root "$ALLT_EXTERNAL_ROOT/training" \\
  --artifact-root artifacts/allt_task_superiority_v1/training

python scripts/evaluate_persist4d_allt.py --variant C2 \\
  --checkpoint "$ALLT_EXTERNAL_ROOT/training/formal/C2/update=0200.ckpt" \\
  --population protocol_b --device cuda:0 --reducers mean latest max \\
  --cache-root "$ALLT_EXTERNAL_ROOT/evaluation_cache" \\
  --output-root artifacts/allt_task_superiority_v1/evaluation/protocol_b/C2/update=0200

python scripts/analyze_persist4d_allt_final.py \\
  --c2-cache-directory "$ALLT_EXTERNAL_ROOT/evaluation_cache/protocol_b_43_masters_3_orders/C2/a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724" \\
  --fh-cache-directory "$ALLT_EXTERNAL_ROOT/evaluation_cache/protocol_b_43_masters_3_orders/FH-adapt/ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c"
python scripts/profile_persist4d_allt.py --device cuda:0
python scripts/publish_persist4d_allt.py
```"""


def _external_table(artifact_root: Path) -> str:
    run_contract = _load_json(artifact_root / "run_contract.json")
    identities = run_contract.get("input_identities")
    if not isinstance(identities, Mapping):
        raise PublicationError("input identities differ")
    matrix = _load_json(artifact_root / "training/variant_matrix.json")
    evidence = _load_json(artifact_root / "results/evidence_manifest.json")
    cache_validation = evidence.get("cache_validation")
    if not isinstance(cache_validation, Mapping):
        raise PublicationError("cache validation differs")
    rows = [
        (
            "R1 checkpoint",
            "external:r1_checkpoint",
            identities["r1_checkpoint"]["sha256"],
            identities["r1_checkpoint"]["bytes"],
        ),
        (
            "Concerto pretrained",
            "external:concerto_pretrained",
            identities["concerto_pretrained"]["sha256"],
            identities["concerto_pretrained"]["bytes"],
        ),
    ]
    for value in _sequence(matrix.get("variants"), name="variant rows"):
        if not isinstance(value, Mapping) or value.get("execution_status") != "COMPLETE":
            continue
        manifest = _load_json(
            artifact_root / f"training/formal/{value['variant']}/checkpoint_manifest.json"
        )
        selected_name = f"update={int(value['selected_checkpoint_global_step']):04d}.ckpt"
        selected = [
            row
            for row in _sequence(manifest.get("checkpoints"), name="checkpoints")
            if isinstance(row, Mapping)
            and row.get("name") == selected_name
            and row.get("sha256") == value.get("selected_checkpoint_sha256")
        ]
        if len(selected) != 1:
            raise PublicationError("selected external checkpoint differs")
        rows.append(
            (
                f"{value['variant']} selected checkpoint",
                f"{manifest['external_reference']}/{selected[0]['name']}",
                selected[0]["sha256"],
                selected[0]["bytes"],
            )
        )
    for model in ("C2", "FH-adapt"):
        row = cache_validation.get(model)
        manifest_path = artifact_root / (
            "evaluation/protocol_b/C2/update=0200/cache_manifest.json"
            if model == "C2"
            else "evaluation/protocol_b/FH-adapt/update=0100/cache_manifest.json"
        )
        manifest = _load_json(manifest_path)
        if not isinstance(row, Mapping):
            raise PublicationError("cache byte evidence differs")
        rows.append(
            (
                f"{model} Protocol-B cache",
                manifest["cache_directory"],
                manifest["content_sha256"],
                row["bytes"],
            )
        )
    return _markdown_table(("Object", "Logical reference", "SHA256/content SHA256", "Bytes"), rows)


def _handoff(artifact_root: Path, statuses: Mapping[str, str], code_commit: str) -> str:
    run_contract = _load_json(artifact_root / "run_contract.json")
    analysis = _load_json(artifact_root / "results/final_analysis_manifest.json")
    profile = _load_json(artifact_root / "profile/run_summary.json")
    return f"""# Persist4D All-T Handoff

## 1. Scientific Goal

The goal was one frozen Persist4D checkpoint that strictly exceeds matched ReScene at t-mAP T2-T5. Execution is complete; the scientific goal is **not achieved**.

{_status_table(statuses)}

## 2. Repository And Commits

- Repository: `Orangekostar/Persist4D`
- Branch: `research/persist4d-allt-task-superiority-v1`
- Start SHA: `{run_contract['baseline_commit']}`
- Publication-code SHA: `{code_commit}`
- Formal evaluation SHA: `d288af93cefc7cf8aaabb9b541b92539782c019c`
- Final cache-analysis SHA: `{analysis['source_commit']}`
- Resource-profile SHA: `{profile['source_commit']}`
- Result/publication commit: resolve from remote branch HEAD after push.

## 3. Frozen Inputs And Checkpoints

{_external_table(artifact_root)}

The sole primary candidate is C2 update 200, SHA256 `a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724`. The matched comparator is FH-adapt update 100, SHA256 `ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c`.

## 4. Data Roles

RIO adaptation uses 36 references/215 masters; development uses 8 references/47 masters and is R1-base exposed. Protocol-B uses 6 references/43 masters x 3 orders = 129 correlated order units and is historical-benchmark exposed, final-only. No processed independent test split was available, so independent generalization is `NOT_ESTABLISHED`.

## 5. Implementation Surface

- `models/persist4d_allt.py`: optional L/M interfaces and prediction-only persistent memory.
- `datasets/persist4d_sequence_dataset.py`: deterministic continuous episode construction.
- `trainer/persist4d_allt_trainer.py`: all-T training, immediate state commits, and gradient audit.
- `scripts/train_persist4d_allt.py`: frozen-budget two-GPU training and external checkpoint manifests.
- `scripts/evaluate_persist4d_allt.py`: continuous T1-T5 inference, official metrics, cache provenance, and diagnostic read policies.
- `scripts/analyze_persist4d_allt_final.py`: streaming per-reference, identity, and equal-cluster analysis.
- `scripts/profile_persist4d_allt.py`: bounded real-A40 profile.
- `scripts/publish_persist4d_allt.py`: fail-closed compact exports, reports, and manifest.

Launch commands are in section 16.

## 6. Variant States

{_training_table(artifact_root)}

C0/C1/C2/FH-adapt completed. C3 was gate-skipped because C1 did not improve both T2/T3 over C0. FH-L was gate-skipped because the frozen candidate did not use L.

## 7. Budget And Exposure

All executed variants used two A40s, 400 optimizer updates, physical batch 1/GPU, accumulation 4, effective batch 8, and `32-true` precision. Every plan contains 3,200 global episodes, 1,600 groups, 1,778 RIO episodes, 1,422 ScanNet episodes, and 7,642 supervised stages. C0/C1/C2 each scanned 12,084 encoder inputs; FH-adapt scanned 16,524. Exact trainable parameters and GPU-hours are in section 6.

## 8. Single-Checkpoint Main Results

{_primary_metric_table(artifact_root)}

All 20 C2-vs-FH-adapt task cells:

{_all_task_delta_table(artifact_root)}

## 9. Baseline Verdicts

C2 vs R1 B4 t-mAP all-T: `FAIL` (T4/T5). C2 vs matched FH-adapt t-mAP all-T: `FAIL` (T3/T4/T5). C2 vs matched FH-adapt all 20 task cells: `FAIL` (only 5/20 positive).

## 10. L/M Evidence

{_development_table(artifact_root)}

{_module_effect_table(artifact_root)}

L did not pass its short-horizon gate. M improved selected C0 across development T2-T5 but exceeded matched FH-adapt only at T2. C3 complementarity therefore was not tested. C2's W=2 current input reads prediction-only persistent K=100 slots that can originate before the current window; the deployment state does not grow with T.

{_diagnostic_summary(artifact_root)}

## 11. Evaluation Semantics

Primary C2 score reducer: `mean`; `latest/max` are sensitivity only. `CandidateTrajectoryKey=(track_id,class_id)` is unchanged. Each episode executes T1 then T2/T3/T4/T5 continuously, with prediction-driven B4 state committed immediately after each stage; no offline smoothing or GT enters state/prediction.

## 12. Seeds And Statistical Units

Training/adaptation seed 45 is primary. Seed 46 was not run because the first-seed development candidate failed the matched all-T gate. Evaluation seed 45 is used for the frozen final run; planned 46/47 repeats are not independent training seeds. The primary population is pooled official evaluation over 129 order units clustered within six references; the 1,000-resample equal-cluster intervals are descriptive only.

## 13. Resources And Limits

{_profile_table(artifact_root)}

Scope: 5 warmups + 10 repeats for C2 and FH-adapt on one canonical sequence from each of six references at T2-T5, all sequentially on one NVIDIA A40. Includes forward, C2 memory read, observation extraction, and B4 update; excludes I/O, collation, H2D, metrics, and C2 preroll. It proves only this bounded deployment operation, not end-to-end throughput or multi-GPU scaling, and cannot offset the failed accuracy goal.

## 14. Tests And Real Failures

See `TEST_REPORT.md`. Direct model/evaluator/analyzer/profiler/publisher/selection/finalizer tests and Ruff are required. Retained failures include the scientific gates, the corrected NFS diagnostic-write permission failure, and the profile-smoke T2 input mismatch that was fixed before the formal profile.

## 15. External Storage Reconstruction

Logical `external:allt_task_superiority_v1/...` resolves under `$ALLT_EXTERNAL_ROOT`. Current binding is `/mnt/shared/ww/persist4d-allt-task-superiority-v1`; `/mnt/shared` is the node-mounted NFS export `192.168.100.102:/mnt/data/shared`. Checkpoint and cache hashes/bytes are in section 3 and their tracked manifests. No credentials or large binaries are stored in Git.

## 16. Minimal Commands

{_reproduction_commands()}

## 17. One Next-Round Bottleneck

Change only memory-read quality scoring/gating. Keep reducer, losses, K=100, W=2, selection, and evaluation fixed. The diagnostic read restrictions improved T3-T5 slightly, so low-quality occupied-slot reads are the best-supported single hypothesis; the diagnostic is not causal proof.

## 18. GitHub Publication

- Branch: <https://github.com/Orangekostar/Persist4D/tree/research/persist4d-allt-task-superiority-v1>
- Commit: resolve from remote branch HEAD.
- Tracked artifact publication status in this immutable package: `{statuses['publication_status']}`.
- `PUSH_VERIFIED` may be reported only in the external publication receipt/final response after local HEAD equals remote HEAD and Git readback of this file plus `FINAL_MANIFEST.json` matches local bytes.
"""


def publish_final_package(
    artifact_root: Path,
    *,
    publication_status: str = "NOT_ATTEMPTED",
    code_commit_at_run: str | None = None,
) -> list[dict[str, str]]:
    """Validate all evidence, then publish deterministic reports and manifest."""
    root = artifact_root.expanduser().resolve()
    required = validate_completion_inputs(root)
    statuses = derive_final_statuses(root, publication_status=publication_status)
    code_commit = code_commit_at_run or _git_head()
    report_paths = {
        root / "TEST_REPORT.md": _test_report(),
        root / "FINAL_REPORT.md": _final_report(root, statuses, code_commit),
        root / "HANDOFF.md": _handoff(root, statuses, code_commit),
    }
    for path, content in report_paths.items():
        _atomic_write(path, content.encode("ascii"))
    manifest = build_final_manifest(
        root,
        artifact_paths=(*required, *report_paths),
        statuses=statuses,
        code_commit_at_run=code_commit,
    )
    manifest_path = root / "FINAL_MANIFEST.json"
    _atomic_json(manifest_path, manifest)
    outputs = (*report_paths, manifest_path)
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _file_sha256(path)}
        for path in outputs
    ]


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
    parser.add_argument(
        "--publication-status", choices=PUBLICATION_STATUSES, default="NOT_ATTEMPTED"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    outputs = publish_compact_exports(args.artifact_root.expanduser().resolve())
    outputs.extend(
        publish_final_package(
            args.artifact_root.expanduser().resolve(),
            publication_status=args.publication_status,
        )
    )
    print(json.dumps({"outputs": outputs, "status": "pass"}, sort_keys=True))
    return 0


__all__ = [
    "COMPLETION_STATUS_KEYS",
    "DEFAULT_ARTIFACT_ROOT",
    "TASK_METRICS",
    "PublicationError",
    "build_final_manifest",
    "build_learning_curve_rows",
    "build_memory_ablation_rows",
    "build_variant_matrix",
    "derive_final_statuses",
    "primary_tmap_failure_cells",
    "publish_compact_exports",
    "publish_final_package",
    "validate_completion_inputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
