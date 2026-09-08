from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.evaluate_persist4d_allt import (
    rank_development_checkpoints,
    validate_result_rows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1"
REQUIRED_VARIANTS = ("C0", "C1", "C2", "FH-adapt")
EXPECTED_POPULATION = "development_train_holdout_47_masters_canonical"
INTEGER_FIELDS = {
    "training_seed",
    "evaluation_seed",
    "T",
    "num_master",
    "num_order_units",
    "num_reference_clusters",
}
FLOAT_FIELDS = {
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "local_current_AP",
}
TRACE_FIELDS = (
    "variant",
    "rank",
    "selected_within_variant",
    "checkpoint_global_step",
    "checkpoint_sha256",
    "baseline",
    "minimum_t_map_delta",
    "mean_t_map",
    "all_t_positive",
    "t2_delta",
    "t3_delta",
    "t4_delta",
    "t5_delta",
)


class SelectionError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot load JSON evidence: {path}") from error
    if not isinstance(payload, dict):
        raise SelectionError(f"JSON evidence must be an object: {path}")
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(path, content)


def _repo_ref(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise SelectionError("selection inputs must be inside the repository") from error
    return f"repo:{relative.as_posix()}"


def _load_metric_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="ascii", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise SelectionError(f"cannot load metric evidence: {path}") from error
    rows: list[dict[str, Any]] = []
    try:
        for raw in raw_rows:
            row: dict[str, Any] = dict(raw)
            for field in INTEGER_FIELDS:
                row[field] = int(row[field])
            for field in FLOAT_FIELDS:
                row[field] = float(row[field])
                if not math.isfinite(row[field]):
                    raise ValueError(field)
            rows.append(row)
        validate_result_rows(rows)
    except (KeyError, TypeError, ValueError) as error:
        raise SelectionError(f"invalid metric evidence: {path}") from error
    return rows


def _checkpoint_steps(
    *, manifest: Mapping[str, Any], variant: str, expected_steps: tuple[int, ...]
) -> dict[int, str]:
    if manifest.get("variant") != variant:
        raise SelectionError(f"{variant} training manifest variant differs")
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise SelectionError(f"{variant} checkpoint records are missing")
    steps: dict[int, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise SelectionError(f"{variant} checkpoint record must be an object")
        match = re.fullmatch(r"update=(\d{4})\.ckpt", str(record.get("name")))
        if match is None:
            continue
        step = int(match.group(1))
        checkpoint = record.get("sha256")
        size = record.get("bytes")
        if not isinstance(checkpoint, str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint) is None:
            raise SelectionError(f"{variant} checkpoint SHA256 is invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise SelectionError(f"{variant} checkpoint byte count is invalid")
        if step in steps:
            raise SelectionError(f"{variant} contains a duplicate checkpoint step")
        steps[step] = checkpoint
    if tuple(sorted(steps)) != expected_steps:
        raise SelectionError(f"{variant} checkpoint schedule differs")
    return steps


def _validate_cache_manifest(
    *,
    manifest: dict[str, Any],
    variant: str,
    checkpoint: str,
    reducer: str,
    source_commit: str,
) -> None:
    content_sha256 = manifest.pop("content_sha256", None)
    actual_content_sha256 = _canonical_json_sha256(manifest)
    manifest["content_sha256"] = content_sha256
    records = manifest.get("records")
    filenames = (
        [record.get("filename") for record in records if isinstance(record, Mapping)]
        if isinstance(records, list)
        else []
    )
    checks = (
        content_sha256 == actual_content_sha256,
        manifest.get("status") == "pass",
        manifest.get("checkpoint_sha256") == checkpoint,
        manifest.get("population_id") == EXPECTED_POPULATION,
        manifest.get("evaluation_seed") == 45,
        manifest.get("score_reducers") == [reducer],
        manifest.get("source_commit") == source_commit,
        manifest.get("entry_count") == 47,
        isinstance(records, list) and len(records) == 47,
        len(filenames) == len(set(filenames)) == 47,
    )
    if not all(checks):
        raise SelectionError(f"{variant} cache manifest binding failed")


def load_variant_evidence(
    *,
    artifact_root: Path,
    variant: str,
    expected_steps: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, str], list[dict[str, Any]]]:
    training_root = artifact_root / "training/formal" / variant
    checkpoint_path = training_root / "checkpoint_manifest.json"
    training_summary_path = training_root / "run_summary.json"
    checkpoint_manifest = _load_json(checkpoint_path)
    training_summary = _load_json(training_summary_path)
    steps = _checkpoint_steps(
        manifest=checkpoint_manifest,
        variant=variant,
        expected_steps=expected_steps,
    )
    if (
        training_summary.get("variant") != variant
        or training_summary.get("execution_status") != "COMPLETE"
        or training_summary.get("completed_global_step") != expected_steps[-1]
        or training_summary.get("optimizer_updates_requested") != expected_steps[-1]
    ):
        raise SelectionError(f"{variant} formal training is incomplete")

    expected_reducer = "official" if variant.startswith("FH-") else "mean"
    expected_method = "FullHistory" if variant.startswith("FH-") else "B4"
    rows: list[dict[str, Any]] = []
    sources = [
        {"ref": _repo_ref(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        {"ref": _repo_ref(training_summary_path), "sha256": _sha256(training_summary_path)},
    ]
    for step in expected_steps:
        evaluation_root = (
            artifact_root / "evaluation/development" / variant / f"update={step:04d}"
        )
        metric_path = evaluation_root / "all_t_metrics.csv"
        cache_path = evaluation_root / "cache_manifest.json"
        summary_path = evaluation_root / "run_summary.json"
        metric_rows = _load_metric_rows(metric_path)
        cache_manifest = _load_json(cache_path)
        summary = _load_json(summary_path)
        checkpoint = steps[step]
        source_commit = summary.get("source_commit")
        expected_row_identity = all(
            row["population_id"] == EXPECTED_POPULATION
            and row["model"] == variant
            and row["checkpoint_sha256"] == checkpoint
            and row["training_seed"] == 45
            and row["evaluation_seed"] == 45
            and row["method"] == expected_method
            and row["reducer"] == expected_reducer
            for row in metric_rows
        )
        summary_checks = (
            summary.get("status") == "pass",
            summary.get("variant") == variant,
            summary.get("model") == variant,
            summary.get("checkpoint_global_step") == step,
            summary.get("checkpoint_sha256") == checkpoint,
            summary.get("population_id") == EXPECTED_POPULATION,
            summary.get("training_seed") == 45,
            summary.get("evaluation_seed") == 45,
            summary.get("reducers") == [expected_reducer],
            summary.get("sequence_count") == 47,
            summary.get("metric_row_count") == 4,
            summary.get("metric_sha256") == _sha256(metric_path),
            isinstance(source_commit, str)
            and re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None,
        )
        if len(metric_rows) != 4 or not expected_row_identity or not all(summary_checks):
            raise SelectionError(f"{variant} update={step:04d} evaluation binding failed")
        _validate_cache_manifest(
            manifest=cache_manifest,
            variant=variant,
            checkpoint=checkpoint,
            reducer=expected_reducer,
            source_commit=source_commit,
        )
        rows.extend(metric_rows)
        for path in (metric_path, cache_path, summary_path):
            sources.append({"ref": _repo_ref(path), "sha256": _sha256(path)})
    return rows, steps, sources


def _strictly_positive_deltas(
    selection: Mapping[str, Any], horizons: tuple[str, ...]
) -> bool:
    deltas = selection.get("t_map_delta_by_horizon")
    if not isinstance(deltas, Mapping):
        return False

    for horizon in horizons:
        value = deltas.get(horizon)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if value <= 0.0:
            return False
    return True


def evaluate_complementarity(
    *, c1_selection: Mapping[str, Any], c2_selection: Mapping[str, Any]
) -> dict[str, Any]:
    c1_short_positive = _strictly_positive_deltas(c1_selection, ("2", "3"))
    c2_long_positive = _strictly_positive_deltas(c2_selection, ("4", "5"))
    authorized = c1_short_positive and c2_long_positive

    return {
        "status": "authorized_pending_training" if authorized else "gate_skipped",
        "c1_checkpoint_sha256": c1_selection.get("checkpoint_sha256"),
        "c2_checkpoint_sha256": c2_selection.get("checkpoint_sha256"),
        "c1_short_horizon_positive": c1_short_positive,
        "c2_long_horizon_positive": c2_long_positive,
        "strict_zero_tolerance": True,
    }


def _checkpoint_rows(
    rows: list[Mapping[str, Any]], checkpoint_sha256: str
) -> list[Mapping[str, Any]]:
    selected = [
        row for row in rows if row.get("checkpoint_sha256") == checkpoint_sha256
    ]
    if len(selected) != 4:
        raise ValueError("checkpoint must have exactly four development rows")
    return selected


def _rank_variant(
    *,
    variant: str,
    rows: list[Mapping[str, Any]],
    baseline_rows: list[Mapping[str, Any]],
    steps: Mapping[int, str],
) -> list[dict[str, Any]]:
    step_by_checkpoint = {checkpoint: step for step, checkpoint in steps.items()}
    if len(step_by_checkpoint) != len(steps):
        raise ValueError(f"{variant} checkpoint identities must be unique")
    candidate_checkpoints = {
        checkpoint for step, checkpoint in steps.items() if step > 0
    }
    candidate_rows = [
        row for row in rows if row.get("checkpoint_sha256") in candidate_checkpoints
    ]
    ranked = rank_development_checkpoints(candidate_rows, baseline_rows)
    for position, row in enumerate(ranked, start=1):
        checkpoint = str(row["checkpoint_sha256"])
        if checkpoint not in step_by_checkpoint:
            raise ValueError(f"{variant} metric checkpoint is absent from training manifest")
        row["variant"] = variant
        row["checkpoint_global_step"] = step_by_checkpoint[checkpoint]
        row["rank"] = position
    return ranked


def build_selection_decision(
    *,
    rows_by_variant: Mapping[str, list[Mapping[str, Any]]],
    checkpoint_steps: Mapping[str, Mapping[int, str]],
) -> dict[str, Any]:
    required = ("C0", "C1", "C2", "FH-adapt")
    if any(variant not in rows_by_variant for variant in required):
        raise ValueError("required development variant is missing")
    if any(variant not in checkpoint_steps for variant in required):
        raise ValueError("required checkpoint manifest is missing")

    p0_sha = checkpoint_steps["C0"].get(0)
    fh_r1_sha = checkpoint_steps["FH-adapt"].get(0)
    if p0_sha is None or fh_r1_sha is None:
        raise ValueError("update=0000 baselines are required")
    p0_rows = _checkpoint_rows(rows_by_variant["C0"], p0_sha)
    fh_r1_rows = _checkpoint_rows(rows_by_variant["FH-adapt"], fh_r1_sha)

    rankings: dict[str, list[dict[str, Any]]] = {}
    for variant in ("C0", "C1", "C2"):
        rankings[variant] = _rank_variant(
            variant=variant,
            rows=rows_by_variant[variant],
            baseline_rows=p0_rows,
            steps=checkpoint_steps[variant],
        )
    rankings["FH-adapt"] = _rank_variant(
        variant="FH-adapt",
        rows=rows_by_variant["FH-adapt"],
        baseline_rows=fh_r1_rows,
        steps=checkpoint_steps["FH-adapt"],
    )

    c0_selected_rows = _checkpoint_rows(
        rows_by_variant["C0"], str(rankings["C0"][0]["checkpoint_sha256"])
    )
    module_selections: dict[str, dict[str, Any]] = {}
    for variant in ("C1", "C2"):
        selected_rows = _checkpoint_rows(
            rows_by_variant[variant],
            str(rankings[variant][0]["checkpoint_sha256"]),
        )
        comparison = rank_development_checkpoints(selected_rows, c0_selected_rows)[0]
        module_selections[variant] = {
            **comparison,
            "variant": variant,
            "checkpoint_global_step": rankings[variant][0]["checkpoint_global_step"],
        }

    complementarity = evaluate_complementarity(
        c1_selection=module_selections["C1"],
        c2_selection=module_selections["C2"],
    )
    conditional = {
        "C3": complementarity["status"],
        "FH-L": "gate_skipped",
    }
    if complementarity["status"] == "authorized_pending_training" and "C3" not in rows_by_variant:
        return {
            "status": "conditional_training_pending",
            "protocol_b_used_for_selection": False,
            "selected_candidate": None,
            "selected_matched_rescene": None,
            "variant_rankings": rankings,
            "module_selections_vs_c0": module_selections,
            "complementarity_gate": complementarity,
            "conditional_variants": conditional,
        }

    local_variants = ["C0", "C1", "C2"]
    if "C3" in rows_by_variant:
        if "C3" not in checkpoint_steps:
            raise ValueError("C3 metrics lack a checkpoint manifest")
        rankings["C3"] = _rank_variant(
            variant="C3",
            rows=rows_by_variant["C3"],
            baseline_rows=p0_rows,
            steps=checkpoint_steps["C3"],
        )
        local_variants.append("C3")
        conditional["C3"] = "complete"

    local_selected_rows = []
    variant_by_checkpoint: dict[str, str] = {}
    selected_rank_by_checkpoint: dict[str, dict[str, Any]] = {}
    for variant in local_variants:
        selected = rankings[variant][0]
        checkpoint = str(selected["checkpoint_sha256"])
        local_selected_rows.extend(_checkpoint_rows(rows_by_variant[variant], checkpoint))
        variant_by_checkpoint[checkpoint] = variant
        selected_rank_by_checkpoint[checkpoint] = selected
    local_ranking = rank_development_checkpoints(local_selected_rows, p0_rows)
    local_checkpoint = str(local_ranking[0]["checkpoint_sha256"])
    local_variant = variant_by_checkpoint[local_checkpoint]
    selected_candidate = {
        **local_ranking[0],
        "variant": local_variant,
        "checkpoint_global_step": selected_rank_by_checkpoint[local_checkpoint][
            "checkpoint_global_step"
        ],
    }

    if local_variant in {"C1", "C3"} and "FH-L" not in rows_by_variant:
        conditional["FH-L"] = "authorized_pending_training"
        return {
            "status": "conditional_training_pending",
            "protocol_b_used_for_selection": False,
            "selected_candidate": None,
            "selected_matched_rescene": None,
            "variant_rankings": rankings,
            "module_selections_vs_c0": module_selections,
            "complementarity_gate": complementarity,
            "conditional_variants": conditional,
        }

    matched_variants = ["FH-adapt"]
    if local_variant in {"C1", "C3"}:
        if "FH-L" not in checkpoint_steps:
            raise ValueError("FH-L metrics lack a checkpoint manifest")
        rankings["FH-L"] = _rank_variant(
            variant="FH-L",
            rows=rows_by_variant["FH-L"],
            baseline_rows=fh_r1_rows,
            steps=checkpoint_steps["FH-L"],
        )
        matched_variants.append("FH-L")
        conditional["FH-L"] = "complete"

    matched_selected_rows = []
    matched_variant_by_checkpoint: dict[str, str] = {}
    matched_rank_by_checkpoint: dict[str, dict[str, Any]] = {}
    for variant in matched_variants:
        selected = rankings[variant][0]
        checkpoint = str(selected["checkpoint_sha256"])
        matched_selected_rows.extend(_checkpoint_rows(rows_by_variant[variant], checkpoint))
        matched_variant_by_checkpoint[checkpoint] = variant
        matched_rank_by_checkpoint[checkpoint] = selected
    matched_ranking = rank_development_checkpoints(matched_selected_rows, fh_r1_rows)
    matched_checkpoint = str(matched_ranking[0]["checkpoint_sha256"])
    matched_variant = matched_variant_by_checkpoint[matched_checkpoint]
    selected_matched = {
        **matched_ranking[0],
        "variant": matched_variant,
        "checkpoint_global_step": matched_rank_by_checkpoint[matched_checkpoint][
            "checkpoint_global_step"
        ],
    }

    candidate_rows = _checkpoint_rows(rows_by_variant[local_variant], local_checkpoint)
    baseline_rows = _checkpoint_rows(
        rows_by_variant[matched_variant], matched_checkpoint
    )
    development_comparison = rank_development_checkpoints(
        candidate_rows, baseline_rows
    )[0]
    return {
        "status": "frozen",
        "protocol_b_used_for_selection": False,
        "selected_candidate": selected_candidate,
        "selected_matched_rescene": selected_matched,
        "development_comparison": development_comparison,
        "variant_rankings": rankings,
        "local_candidate_ranking": local_ranking,
        "matched_rescene_ranking": matched_ranking,
        "module_selections_vs_c0": module_selections,
        "complementarity_gate": complementarity,
        "conditional_variants": conditional,
    }


def _metrics_by_horizon(
    rows: list[Mapping[str, Any]], checkpoint: str
) -> dict[str, float]:
    return {
        str(row["T"]): float(row["t_mAP"])
        for row in _checkpoint_rows(rows, checkpoint)
    }


def _trace_bytes(decision: Mapping[str, Any]) -> bytes:
    output = []
    for variant in sorted(decision["variant_rankings"]):
        baseline = "FH-R1" if variant.startswith("FH-") else "P0"
        for row in decision["variant_rankings"][variant]:
            deltas = row["t_map_delta_by_horizon"]
            output.append(
                {
                    "variant": variant,
                    "rank": row["rank"],
                    "selected_within_variant": row["rank"] == 1,
                    "checkpoint_global_step": row["checkpoint_global_step"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "baseline": baseline,
                    "minimum_t_map_delta": row["minimum_t_map_delta"],
                    "mean_t_map": row["mean_t_map"],
                    "all_t_positive": row["all_t_positive"],
                    "t2_delta": deltas["2"],
                    "t3_delta": deltas["3"],
                    "t4_delta": deltas["4"],
                    "t5_delta": deltas["5"],
                }
            )
    text_output = []
    class _ListWriter:
        def write(self, value: str) -> int:
            text_output.append(value)
            return len(value)

    writer = csv.DictWriter(_ListWriter(), fieldnames=TRACE_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)
    return "".join(text_output).encode("ascii")


def _conditional_status_payload(
    *, variant: str, status: str, decision: Mapping[str, Any]
) -> dict[str, Any]:
    if variant == "C3":
        reason = (
            "C1 did not strictly improve both T2 and T3 versus selected C0, "
            "so the preregistered complementarity gate failed closed."
        )
        evidence = decision["complementarity_gate"]
    else:
        reason = (
            "The frozen Persist4D candidate does not use L, so matched FH-L "
            "training is not authorized."
        )
        selected = decision.get("selected_candidate")
        evidence = {
            "selected_candidate_variant": (
                selected.get("variant") if isinstance(selected, Mapping) else None
            )
        }
    return {
        "evidence": evidence,
        "reason": reason,
        "schema_version": 1,
        "status": status,
        "variant": variant,
    }


def run_selection(*, artifact_root: Path, output_root: Path) -> dict[str, Any]:
    budget_path = artifact_root / "budget_and_schedule.json"
    budget = _load_json(budget_path)
    raw_steps = budget.get("evaluation_updates")
    if (
        not isinstance(raw_steps, list)
        or any(isinstance(step, bool) or not isinstance(step, int) for step in raw_steps)
        or tuple(raw_steps) != tuple(sorted(set(raw_steps)))
        or not raw_steps
        or raw_steps[0] != 0
    ):
        raise SelectionError("evaluation checkpoint schedule is invalid")
    expected_steps = tuple(raw_steps)

    rows_by_variant: dict[str, list[Mapping[str, Any]]] = {}
    checkpoint_steps: dict[str, Mapping[int, str]] = {}
    sources = [{"ref": _repo_ref(budget_path), "sha256": _sha256(budget_path)}]
    for variant in REQUIRED_VARIANTS:
        rows, steps, variant_sources = load_variant_evidence(
            artifact_root=artifact_root,
            variant=variant,
            expected_steps=expected_steps,
        )
        rows_by_variant[variant] = rows
        checkpoint_steps[variant] = steps
        sources.extend(variant_sources)

    decision = build_selection_decision(
        rows_by_variant=rows_by_variant,
        checkpoint_steps=checkpoint_steps,
    )
    if decision["status"] != "frozen":
        status_path = output_root / "selection_status.json"
        _atomic_json(
            status_path,
            {
                "decision": decision,
                "schema_version": 1,
                "status": decision["status"],
            },
        )
        return decision

    trace_path = output_root / "selection_trace.csv"
    trace_content = _trace_bytes(decision)
    _atomic_write(trace_path, trace_content)
    complementarity_path = output_root / "complementarity_gate.json"
    _atomic_json(complementarity_path, decision["complementarity_gate"])

    candidate = dict(decision["selected_candidate"])
    candidate_variant = candidate["variant"]
    candidate_checkpoint = candidate["checkpoint_sha256"]
    candidate["t_map_by_horizon"] = _metrics_by_horizon(
        rows_by_variant[candidate_variant], candidate_checkpoint
    )
    matched = dict(decision["selected_matched_rescene"])
    matched_variant = matched["variant"]
    matched_checkpoint = matched["checkpoint_sha256"]
    matched["t_map_by_horizon"] = _metrics_by_horizon(
        rows_by_variant[matched_variant], matched_checkpoint
    )
    comparison = dict(decision["development_comparison"])
    comparison["baseline_checkpoint_sha256"] = matched_checkpoint
    comparison["candidate_variant"] = candidate_variant
    comparison["baseline_variant"] = matched_variant
    comparison["all_t_positive"] = all(
        value > 0.0 for value in comparison["t_map_delta_by_horizon"].values()
    )

    frozen = {
        "conditional_variants": decision["conditional_variants"],
        "development_candidate_vs_matched_rescene": comparison,
        "development_population_id": EXPECTED_POPULATION,
        "evaluation_seed": 45,
        "input_evidence": sorted(sources, key=lambda row: row["ref"]),
        "primary_metric": "causal_prefix_t_mAP",
        "protocol_b_evaluation": "authorized_after_selection_freeze",
        "protocol_b_used_for_selection": False,
        "ranking": [
            "maximum minimum T2-T5 t-mAP delta",
            "maximum mean T2-T5 t-mAP",
            "checkpoint SHA256 lexical order",
        ],
        "schema_version": 1,
        "seed46_training_confirmation": (
            "authorized_pending_training"
            if comparison["all_t_positive"]
            else "gate_skipped"
        ),
        "selected_candidate": candidate,
        "selected_matched_rescene": matched,
        "selection_trace": {
            "ref": _repo_ref(trace_path),
            "sha256": hashlib.sha256(trace_content).hexdigest(),
        },
        "status": "frozen",
        "strict_zero_tolerance": True,
        "training_seed": 45,
    }
    frozen["content_sha256"] = _canonical_json_sha256(frozen)
    frozen_path = output_root / "FROZEN_SELECTION.json"
    _atomic_json(frozen_path, frozen)

    for variant in ("C3", "FH-L"):
        status = decision["conditional_variants"][variant]
        if status == "gate_skipped":
            _atomic_json(
                artifact_root / "training/formal" / variant / "status.json",
                _conditional_status_payload(
                    variant=variant,
                    status=status,
                    decision=decision,
                ),
            )
    return frozen


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate development evidence and freeze Persist4D all-T selection."
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT / "selection",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_selection(
        artifact_root=args.artifact_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(result, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
