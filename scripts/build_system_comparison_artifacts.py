"""Build and verify the lightweight system-comparison artifact package."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml

ALLOWED_CLASSIFICATIONS = (
    "SYSTEM_LOCK",
    "SYSTEM_PARETO_LOCK",
    "ASSOCIATION_LIMITED",
    "REPRESENTATION_LIMITED",
)


class ArtifactError(ValueError):
    """Raised when final evidence is missing, inconsistent, or noncanonical."""


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"{name} must be finite")
    return result


def classify_system_outcome(
    *,
    persist4d_tmap: Mapping[str, object],
    full_history_tmap: Mapping[str, object],
    paired_ci: Mapping[str, Sequence[float]],
    identity_advantage: bool,
    compute_advantage: bool,
    meaningful_advantage: float,
    noninferiority_tolerance: float = 0.0,
    oracle_tmap: Mapping[str, object] | None = None,
    oracle_within: float = 0.01,
    oracle_closure_fraction: float = 0.8,
) -> dict[str, object]:
    if not isinstance(identity_advantage, bool) or not isinstance(
        compute_advantage, bool
    ):
        raise ArtifactError("system advantages must be boolean")
    for name, value in (
        ("meaningful_advantage", meaningful_advantage),
        ("noninferiority_tolerance", noninferiority_tolerance),
        ("oracle_within", oracle_within),
        ("oracle_closure_fraction", oracle_closure_fraction),
    ):
        if not math.isfinite(value) or value < 0:
            raise ArtifactError(f"{name} must be finite and non-negative")
    persistent = {
        horizon: _finite(persist4d_tmap.get(horizon), name=f"Persist4D {horizon}")
        for horizon in ("T4", "T5")
    }
    full = {
        horizon: _finite(
            full_history_tmap.get(horizon), name=f"Full-History {horizon}"
        )
        for horizon in ("T4", "T5")
    }
    meaningful_horizons = []
    for horizon in ("T4", "T5"):
        interval = paired_ci.get(horizon)
        if (
            isinstance(interval, (str, bytes))
            or not isinstance(interval, Sequence)
            or len(interval) != 2
        ):
            raise ArtifactError(f"paired CI is invalid at {horizon}")
        upper = _finite(interval[1], name=f"paired CI upper {horizon}")
        if full[horizon] - persistent[horizon] >= meaningful_advantage and upper < 0:
            meaningful_horizons.append(horizon)

    if (
        identity_advantage
        and compute_advantage
        and all(
            persistent[horizon] + noninferiority_tolerance >= full[horizon]
            for horizon in ("T4", "T5")
        )
    ):
        return {
            "classification": "SYSTEM_LOCK",
            "reason": "non-inferior long-horizon task quality with identity and compute advantages",
            "oracle_required": False,
        }
    if identity_advantage and compute_advantage and not meaningful_horizons:
        return {
            "classification": "SYSTEM_PARETO_LOCK",
            "reason": "no meaningful Full-History accuracy advantage and Persist4D improves identity and compute",
            "oracle_required": False,
        }
    if not meaningful_horizons:
        raise ArtifactError(
            "outcome is indeterminate without identity and compute advantages"
        )
    if oracle_tmap is None:
        raise ArtifactError("meaningful Full-History advantage requires Oracle evidence")
    oracle_closes = []
    closure_by_horizon = {}
    for horizon in meaningful_horizons:
        oracle = _finite(oracle_tmap.get(horizon), name=f"Oracle {horizon}")
        deficit = full[horizon] - persistent[horizon]
        closure = (oracle - persistent[horizon]) / deficit
        closure_by_horizon[horizon] = closure
        oracle_closes.append(
            abs(full[horizon] - oracle) <= oracle_within
            or closure >= oracle_closure_fraction
        )
    association_limited = all(oracle_closes)
    return {
        "classification": (
            "ASSOCIATION_LIMITED" if association_limited else "REPRESENTATION_LIMITED"
        ),
        "reason": (
            "Oracle association closes the meaningful Full-History deficit"
            if association_limited
            else "Oracle association does not close the meaningful Full-History deficit"
        ),
        "oracle_required": True,
        "oracle_closure_fraction_by_horizon": closure_by_horizon,
    }


def render_final_report(
    *,
    source_commit: str,
    classification: str,
    answers: Sequence[str],
    evidence_files: Sequence[str],
) -> str:
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ArtifactError("report source commit must be a lowercase SHA-1")
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise ArtifactError("report classification is not preregistered")
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence) or len(answers) != 10:
        raise ArtifactError("final report must contain exactly 10 answers")
    if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ArtifactError("final report answers must be nonempty")
    questions = (
        "What is the checkpoint training horizon?",
        "Are T3-T5 Full-History results zero-shot extensions?",
        "How does Full-History task quality change with T?",
        "How does Persist4D task quality change with T?",
        "Which system has better deployment identity stability?",
        "Which system has better Gap Identity Recovery?",
        "Which system has better per-new-visit latency scaling?",
        "Which system has better peak VRAM scaling?",
        "Does Persist4D form an accuracy/identity/compute Pareto advantage?",
        "What should happen next?",
    )
    lines = [
        "# Full-History vs Persistent-State System Comparison",
        "",
        "Method A: ReScene4D Full-History (Frozen T2 Checkpoint).",
        "Method B: Persist4D Persistent-State.",
        "",
        f"Source commit: `{source_commit}`",
        f"Final classification: `{classification}`",
        "",
    ]
    for index, (question, answer) in enumerate(
        zip(questions, answers, strict=True), start=1
    ):
        lines.extend([f"## Q{index}. {question}", "", answer.strip(), ""])
    lines.extend(["## Evidence", ""])
    for path in evidence_files:
        if not isinstance(path, str) or not path:
            raise ArtifactError("report evidence paths must be nonempty strings")
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            (
                "The comparison uses the frozen T2 checkpoint, exact Protocol-B "
                "prefixes, all three preregistered orders, and reference-scene "
                "clusters as the independent statistical units."
            ),
            "",
        ]
    )
    report = "\n".join(lines)
    validate_final_report(report)
    return report


def validate_final_report(text: str) -> dict[str, object]:
    if not isinstance(text, str) or not text:
        raise ArtifactError("final report must be nonempty text")
    headings = re.findall(r"^## Q([1-9]|10)\. ", text, flags=re.MULTILINE)
    if headings != [str(index) for index in range(1, 11)]:
        raise ArtifactError("final report does not answer exact Q1-Q10")
    classifications = [
        label
        for label in ALLOWED_CLASSIFICATIONS
        if re.search(rf"Final classification: `{label}`", text)
    ]
    if len(classifications) != 1:
        raise ArtifactError("final report must contain exactly one classification")
    if "ReScene4D Full-History (Frozen T2 Checkpoint)" not in text:
        raise ArtifactError("final report lacks the frozen T2 method name")
    return {"answer_count": 10, "classification": classifications[0]}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_manifest(
    root: Path,
    *,
    required_paths: Sequence[str],
    source_commit: str,
) -> dict[str, object]:
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise ArtifactError("artifact source commit must be a lowercase SHA-1")
    if isinstance(required_paths, (str, bytes)) or not isinstance(
        required_paths, Sequence
    ):
        raise ArtifactError("required artifact paths must be a sequence")
    normalized = list(required_paths)
    if len(set(normalized)) != len(required_paths):
        raise ArtifactError("required artifact paths contain duplicates")
    artifacts = []
    for raw in normalized:
        if not isinstance(raw, str) or not raw:
            raise ArtifactError("artifact path must be a nonempty string")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != raw:
            raise ArtifactError("artifact path must be normalized and relative")
        path = root / Path(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ArtifactError(f"required artifact is missing: {raw}")
        artifacts.append(
            {
                "path": raw,
                "byte_size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": source_commit,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _load_b3_rows(project_root: Path) -> list[dict[str, object]]:
    strict_path = project_root / "artifacts/P6A/strict_online_results.csv"
    baseline_path = project_root / "artifacts/P6A/baseline_results.csv"
    try:
        with strict_path.open("r", encoding="utf-8", newline="") as handle:
            strict = [row for row in csv.DictReader(handle) if row["method"] == "B3"]
        with baseline_path.open("r", encoding="utf-8", newline="") as handle:
            baseline = [row for row in csv.DictReader(handle) if row["method"] == "B3"]
    except (OSError, UnicodeError, csv.Error, KeyError) as error:
        raise ArtifactError("frozen B3 evidence cannot be decoded") from error
    if len(strict) != 4 or len(baseline) != 4:
        raise ArtifactError("frozen B3 evidence lacks exact T2-T5 coverage")
    if (
        {int(row["T"]) for row in strict} != {2, 3, 4, 5}
        or {int(row["T"]) for row in baseline} != {2, 3, 4, 5}
    ):
        raise ArtifactError("frozen B3 horizons differ from T2-T5")
    baseline_by_horizon = {int(row["T"]): row for row in baseline}
    return [
        {
            "method": "B3",
            "T": int(row["T"]),
            "t_mAP": float(row["t_mAP"]),
            "t_REC": float(row["t_REC"]),
            "id_switch_rate": float(
                baseline_by_horizon[int(row["T"])]["id_switch_rate"]
            ),
        }
        for row in strict
    ]


def _trend(
    rows: Mapping[tuple[str, int], Mapping[str, object]],
    method: str,
    metric: str,
    *,
    percent: bool = False,
) -> str:
    values = []
    for horizon in (2, 3, 4, 5):
        value = rows[(method, horizon)].get(metric)
        if value is None:
            rendered = "NA"
        else:
            number = _finite(value, name=f"{method} {metric} T{horizon}")
            rendered = f"{100 * number:.2f}%" if percent else f"{number:.4f}"
        values.append(f"T{horizon}={rendered}")
    return ", ".join(values)


def run_build_all_artifacts(
    *,
    project_root: Path,
    metadata_path: Path,
) -> dict[str, object]:
    from scripts.run_system_comparison import SYSTEM_ROOT, _load_bound_inputs
    from scripts.system_comparison_analysis import (
        _csv_bytes,
        _publish_exact,
        _read_typed_csv,
        run_oracle_attribution,
    )
    from scripts.system_comparison_figures import (
        build_table_a,
        build_table_b,
        render_required_figures,
    )

    repository = Path(__file__).resolve().parents[1]
    if project_root.resolve() != repository:
        raise ArtifactError("project_root differs from the artifact repository")
    _system_manifest, binding = _load_bound_inputs()
    aggregate = _read_typed_csv(SYSTEM_ROOT / "aggregate_results.csv")
    profile = _read_typed_csv(SYSTEM_ROOT / "profile_results.csv")
    bootstrap = _read_typed_csv(SYSTEM_ROOT / "cluster_bootstrap.csv")
    b3_rows = _load_b3_rows(repository)
    table_a = build_table_a(aggregate, b3_rows)
    table_b = build_table_b(profile)

    table_a_fields = (
        "method_id",
        "method",
        "history_strategy",
        "horizon",
        "causal_prefix_t_mAP",
        "causal_prefix_t_REC",
        "normalized_id_switch_rate",
        "gap_recovery_accuracy",
        "gap_recovery_recall",
    )
    table_b_fields = (
        "method_id",
        "method",
        "horizon",
        "profile_cluster_count",
        "scans_processed_per_update",
        "cumulative_scans_processed",
        "median_latency_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "mean_update_point_count",
        "mean_cumulative_point_count",
        "historical_state_bytes",
        "explicit_history_input_bytes",
    )
    for filename, rows, fields in (
        ("table_a_system_comparison.csv", table_a, table_a_fields),
        ("table_b_compute_scaling.csv", table_b, table_b_fields),
        ("system_results.csv", table_a, table_a_fields),
        ("efficiency_results.csv", table_b, table_b_fields),
    ):
        _publish_exact(SYSTEM_ROOT / filename, _csv_bytes(rows, fields))

    aggregate_by = {
        (str(row["method"]), int(row["horizon"])): row for row in aggregate
    }
    if set(aggregate_by) != {
        (method, horizon)
        for method in ("FullHistory", "Persist4D")
        for horizon in (2, 3, 4, 5)
    }:
        raise ArtifactError("aggregate results lack exact system/horizon coverage")
    identity_fields = (
        "method",
        "horizon",
        "deployment_id_switches",
        "identity_transition_opportunities",
        "normalized_id_switch_rate",
        "fragmentation_count",
        "fragmentation_opportunities",
        "fragmentation_rate",
        "merge_count",
        "merge_opportunities",
        "merge_rate",
    )
    gap_fields = (
        "method",
        "horizon",
        "gap_opportunities",
        "recovery_attempts",
        "correct_recoveries",
        "gap_recovery_accuracy",
        "gap_recovery_recall",
    )
    identity_rows = [
        {field: row[field] for field in identity_fields}
        for row in aggregate
    ]
    gap_rows = [{field: row[field] for field in gap_fields} for row in aggregate]
    _publish_exact(
        SYSTEM_ROOT / "identity_results.csv",
        _csv_bytes(identity_rows, identity_fields),
    )
    _publish_exact(
        SYSTEM_ROOT / "gap_recovery_results.csv",
        _csv_bytes(gap_rows, gap_fields),
    )

    table_b_by = {
        (str(row["method_id"]), int(row["horizon"])): row for row in table_b
    }
    cumulative_rows = []
    for method in ("FullHistory", "Persist4D"):
        cumulative_latency = 0.0
        for horizon in (2, 3, 4, 5):
            row = table_b_by[(method, horizon)]
            cumulative_latency += float(row["median_latency_ms"])
            cumulative_rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "cumulative_median_latency_ms": cumulative_latency,
                    "cumulative_scans_processed": row[
                        "cumulative_scans_processed"
                    ],
                    "mean_cumulative_point_count": row[
                        "mean_cumulative_point_count"
                    ],
                }
            )
    cumulative_fields = (
        "method",
        "horizon",
        "cumulative_median_latency_ms",
        "cumulative_scans_processed",
        "mean_cumulative_point_count",
    )
    _publish_exact(
        SYSTEM_ROOT / "cumulative_compute.csv",
        _csv_bytes(cumulative_rows, cumulative_fields),
    )
    render_required_figures(table_a, table_b, SYSTEM_ROOT / "figures")

    tmap_bootstrap = {
        int(row["horizon"]): row
        for row in bootstrap
        if row["metric"] == "causal_prefix_t_mAP"
        and int(row["horizon"]) in (4, 5)
    }
    if set(tmap_bootstrap) != {4, 5}:
        raise ArtifactError("t-mAP bootstrap evidence lacks T4/T5")
    persist_tmap = {
        f"T{horizon}": aggregate_by[("Persist4D", horizon)][
            "causal_prefix_t_mAP"
        ]
        for horizon in (4, 5)
    }
    full_tmap = {
        f"T{horizon}": aggregate_by[("FullHistory", horizon)][
            "causal_prefix_t_mAP"
        ]
        for horizon in (4, 5)
    }
    paired_ci = {
        f"T{horizon}": (
            tmap_bootstrap[horizon]["ci_lower"],
            tmap_bootstrap[horizon]["ci_upper"],
        )
        for horizon in (4, 5)
    }
    identity_advantage = all(
        _finite(
            aggregate_by[("Persist4D", horizon)]["normalized_id_switch_rate"],
            name="Persist4D IDSW rate",
        )
        < _finite(
            aggregate_by[("FullHistory", horizon)]["normalized_id_switch_rate"],
            name="Full-History IDSW rate",
        )
        for horizon in (4, 5)
    )
    gap_advantage = all(
        _finite(
            aggregate_by[("Persist4D", horizon)]["gap_recovery_recall"],
            name="Persist4D gap recovery recall",
        )
        > _finite(
            aggregate_by[("FullHistory", horizon)]["gap_recovery_recall"],
            name="Full-History gap recovery recall",
        )
        for horizon in (4, 5)
    )
    compute_advantage = all(
        float(table_b_by[("Persist4D", horizon)]["median_latency_ms"])
        < float(table_b_by[("FullHistory", horizon)]["median_latency_ms"])
        and float(table_b_by[("Persist4D", horizon)]["mean_update_point_count"])
        < float(table_b_by[("FullHistory", horizon)]["mean_update_point_count"])
        for horizon in (4, 5)
    )
    vram_advantage = all(
        float(table_b_by[("Persist4D", horizon)]["peak_allocated_mib"])
        < float(table_b_by[("FullHistory", horizon)]["peak_allocated_mib"])
        for horizon in (4, 5)
    )
    config = yaml.safe_load(
        (repository / "configs/system_comparison/persist4d_incumbent.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(config, Mapping) or not isinstance(
        config.get("decision"), Mapping
    ):
        raise ArtifactError("incumbent decision configuration is invalid")
    decision = config["decision"]
    classifier_args = {
        "persist4d_tmap": persist_tmap,
        "full_history_tmap": full_tmap,
        "paired_ci": paired_ci,
        "identity_advantage": identity_advantage,
        "compute_advantage": compute_advantage,
        "noninferiority_tolerance": float(
            decision["system_lock_task_noninferiority_tolerance"]
        ),
        "meaningful_advantage": float(
            decision["meaningful_full_history_tmap_advantage"]
        ),
        "oracle_within": float(decision["oracle_within_full_history_tmap"]),
        "oracle_closure_fraction": float(
            decision["oracle_deficit_closure_fraction"]
        ),
    }
    oracle_tmap = None
    try:
        outcome = classify_system_outcome(**classifier_args)
    except ArtifactError as error:
        if "requires Oracle evidence" not in str(error):
            raise
        oracle_tmap = run_oracle_attribution(
            project_root=repository,
            metadata_path=metadata_path,
        )
        outcome = classify_system_outcome(
            **classifier_args,
            oracle_tmap=oracle_tmap,
        )

    answers = (
        "The formal checkpoint was trained and validated at temporal horizon T2.",
        "Yes. Full-History T3/T4/T5 are zero-shot temporal-horizon extensions of the frozen T2 checkpoint.",
        "Full-History causal-prefix t-mAP: "
        + _trend(aggregate_by, "FullHistory", "causal_prefix_t_mAP")
        + ".",
        "Persist4D causal-prefix t-mAP: "
        + _trend(aggregate_by, "Persist4D", "causal_prefix_t_mAP")
        + ".",
        ("Persist4D" if identity_advantage else "Full-History or neither system")
        + " has the lower T4/T5 normalized deployment ID-switch rate. Full-History: "
        + _trend(
            aggregate_by,
            "FullHistory",
            "normalized_id_switch_rate",
            percent=True,
        )
        + "; Persist4D: "
        + _trend(
            aggregate_by,
            "Persist4D",
            "normalized_id_switch_rate",
            percent=True,
        )
        + ".",
        ("Persist4D" if gap_advantage else "Full-History or neither system")
        + " has the higher T4/T5 Gap Identity Recovery recall. Full-History: "
        + _trend(
            aggregate_by, "FullHistory", "gap_recovery_recall", percent=True
        )
        + "; Persist4D: "
        + _trend(
            aggregate_by, "Persist4D", "gap_recovery_recall", percent=True
        )
        + ".",
        ("Persist4D" if compute_advantage else "Full-History or neither system")
        + " has the better T4/T5 per-new-visit latency and point-processing scaling. Median latency values are in table_b_compute_scaling.csv.",
        ("Persist4D" if vram_advantage else "Full-History or neither system")
        + " has the lower T4/T5 peak allocated VRAM. Peak allocated/reserved VRAM is reported in table_b_compute_scaling.csv and remains separate from state/input bytes.",
        (
            "Yes. Persist4D forms the preregistered system-level Pareto result: "
            if outcome["classification"]
            in {"SYSTEM_LOCK", "SYSTEM_PARETO_LOCK"}
            else "No. Persist4D does not form the preregistered system-level Pareto result: "
        )
        + str(outcome["reason"])
        + ".",
        {
            "SYSTEM_LOCK": "Freeze the method, complete published baselines, run external validation, and write the paper.",
            "SYSTEM_PARETO_LOCK": "Freeze the method as a Pareto result; do not add modules, then run external validation and write the paper.",
            "ASSOCIATION_LIMITED": "Audit a geometry-aware matcher using predicted object evidence; do not integrate it without a separate feasibility gate.",
            "REPRESENTATION_LIMITED": "Audit memory-conditioned query perception; do not implement it without a separate feasibility gate.",
        }[str(outcome["classification"])],
    )
    evidence_files = (
        "REScene_FULL_HISTORY_CODE_AUDIT.md",
        "FULL_HISTORY_DETERMINISM_AUDIT.md",
        "table_a_system_comparison.csv",
        "table_b_compute_scaling.csv",
        "cluster_bootstrap.csv",
        "leave_one_scene_out.csv",
        "order_robustness.csv",
    )
    if oracle_tmap is not None:
        evidence_files = (*evidence_files, "oracle_attribution.csv")
    report = render_final_report(
        source_commit=str(binding["source_commit"]),
        classification=str(outcome["classification"]),
        answers=answers,
        evidence_files=evidence_files,
    )
    _publish_exact(
        SYSTEM_ROOT / "SYSTEM_COMPARISON_GO_NOGO_REPORT.md",
        report.encode("utf-8"),
    )

    required = [
        "REScene_FULL_HISTORY_CODE_AUDIT.md",
        "FULL_HISTORY_DETERMINISM_AUDIT.md",
        "system_comparison_manifest.json",
        "reproducibility_binding.json",
        "full_history_predictions/manifest.json",
        "persistent_predictions/manifest.json",
        "per_sequence_results.csv",
        "per_order_results.csv",
        "aggregate_results.csv",
        "cached_evaluation.json",
        "profile_results.csv",
        "profile_summary.json",
        "statistics_summary.json",
        "system_results.csv",
        "identity_results.csv",
        "gap_recovery_results.csv",
        "efficiency_results.csv",
        "cumulative_compute.csv",
        "cluster_bootstrap.csv",
        "leave_one_scene_out.csv",
        "order_robustness.csv",
        "table_a_system_comparison.csv",
        "table_b_compute_scaling.csv",
        "figures/figure_1_task_quality.svg",
        "figures/figure_2_identity_stability.svg",
        "figures/figure_3_gap_recovery.svg",
        "figures/figure_4_latency_scaling.svg",
        "figures/figure_5_peak_vram.svg",
        "figures/figure_6_accuracy_compute_pareto.svg",
        "SYSTEM_COMPARISON_GO_NOGO_REPORT.md",
    ]
    if oracle_tmap is not None:
        required.append("oracle_attribution.csv")
    manifest = build_artifact_manifest(
        SYSTEM_ROOT,
        required_paths=required,
        source_commit=str(binding["source_commit"]),
    )
    _publish_exact(
        SYSTEM_ROOT / "artifact_manifest.json",
        (
            json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return {
        "status": "pass",
        "classification": outcome["classification"],
        "artifact_count": manifest["artifact_count"],
        "oracle_attribution_run": oracle_tmap is not None,
    }


__all__ = [
    "ALLOWED_CLASSIFICATIONS",
    "ArtifactError",
    "build_artifact_manifest",
    "classify_system_outcome",
    "render_final_report",
    "run_build_all_artifacts",
    "validate_final_report",
]
