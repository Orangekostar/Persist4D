"""Validate and publish the compact R1 downstream evidence closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
CHECKPOINT_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
PROTOCOL_SHA256 = "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
UPSTREAM_PATHS = (
    "CODE_MAP.md",
    "EXPERIMENT_CONTRACT.md",
    "runtime_config.yaml",
    "input_manifest.json",
    "smoke_and_parity.json",
    "cache_manifest.json",
    "historical_replay_status.json",
    "metrics/local_current.csv",
    "metrics/task_per_sequence.csv",
    "metrics/task_aggregate.csv",
    "metrics/task_per_cluster.csv",
    "metrics/identity_per_sequence.csv",
    "metrics/identity_aggregate.csv",
    "metrics/identity_per_cluster.csv",
    "metrics/gap_event_ledger.csv",
    "metrics/score_sensitivity.csv",
    "metrics/checkpoint_regime_comparison.csv",
    "profile/samples.csv",
    "profile/summary.csv",
)


class R1FinalizationError(ValueError):
    """Raised when final R1 evidence is incomplete or inconsistent."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R1FinalizationError(f"invalid JSON evidence: {path.name}") from error
    if not isinstance(value, Mapping):
        raise R1FinalizationError(f"JSON evidence must be a mapping: {path.name}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise R1FinalizationError(f"missing CSV evidence: {path.name}") from error
    if not rows:
        raise R1FinalizationError(f"CSV evidence is empty: {path.name}")
    return rows


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise R1FinalizationError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise R1FinalizationError(f"{field} must be finite")
    return result


def derive_final_statuses(
    *, recovery: Mapping[str, object], historical: Mapping[str, object]
) -> dict[str, str]:
    recovery_status = recovery.get("status")
    if recovery_status not in {"RECOVERY_SUPPORTED", "RECOVERY_NOT_SUPPORTED"}:
        raise R1FinalizationError("recovery status is invalid")
    historical_status = historical.get("status")
    if historical_status == "HISTORICAL_REPLAY_AVAILABLE":
        old_new = "HISTORICAL_REPLAY_AVAILABLE"
    elif historical_status == "HISTORICAL_REPLAY_UNAVAILABLE":
        summary = historical.get("frozen_summary_comparison")
        if not isinstance(summary, Mapping) or summary.get("status") != "available":
            raise R1FinalizationError("frozen summary comparison is unavailable")
        old_new = "FROZEN_SUMMARY_COMPARISON_ONLY"
    else:
        raise R1FinalizationError("historical replay status is invalid")
    return {
        "execution": "EXECUTION_COMPLETE",
        "candidate": "R1_FROZEN_CHECKPOINT_EVALUATED",
        "recovery": str(recovery_status),
        "task": "TASK_EVIDENCE_COMPLETE",
        "resource": "RESOURCE_PROFILE_COMPLETE",
        "old_new": old_new,
        "publication": (
            "RECOVERY_CLAIM_AUTHORIZED"
            if recovery_status == "RECOVERY_SUPPORTED"
            else "RECOVERY_CLAIM_NOT_AUTHORIZED"
        ),
    }


def build_final_manifest(
    *,
    source_commit: str,
    statuses: Mapping[str, str],
    checkpoint_sha256: str,
    protocol_sha256: str,
    upstream_payloads: Mapping[str, bytes],
    output_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise R1FinalizationError("source commit is invalid")
    expected_statuses = {
        "execution",
        "candidate",
        "recovery",
        "task",
        "resource",
        "old_new",
        "publication",
    }
    if set(statuses) != expected_statuses:
        raise R1FinalizationError("final status coverage differs")
    if len(checkpoint_sha256) != 64 or len(protocol_sha256) != 64:
        raise R1FinalizationError("frozen identity is invalid")
    if not upstream_payloads or not output_payloads:
        raise R1FinalizationError("final manifest inputs cannot be empty")
    recovery_supported = statuses["recovery"] == "RECOVERY_SUPPORTED"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": "persist4d_r1_downstream_validation_v1",
        "status": "complete",
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256,
        "statuses": dict(statuses),
        "upstream_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(upstream_payloads.items())
        },
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(output_payloads.items())
        },
        "authorized_claims": [
            "The frozen R1 checkpoint was evaluated on all 129 Protocol-B sequence-order units.",
            (
                "B4 satisfies the preregistered R1 gap-recovery rule against B2 at T4 and T5."
                if recovery_supported
                else "B4 does not satisfy the preregistered R1 gap-recovery rule against B2 at both T4 and T5."
            ),
            "The six-unit A40 resource profile is descriptive for this frozen implementation.",
        ],
        "forbidden_claims": [
            "causal attribution of old-versus-R1 downstream differences without historical raw replay",
            "generalization beyond the frozen Protocol-B data and implementation",
            "state-of-the-art or production deployment readiness",
        ],
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def _expected_task_cells() -> set[tuple[str, str, str, int]]:
    cells = {
        ("FullHistory", "official", order, horizon)
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    }
    cells.update(
        (method, reducer, order, horizon)
        for method in ("B2", "B4")
        for reducer in ("mean", "latest", "max")
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    )
    return cells


def _validate_exact_cells(
    rows: Sequence[Mapping[str, str]],
    *,
    expected: set[tuple[object, ...]],
    fields: Sequence[str],
    name: str,
) -> None:
    cells: list[tuple[object, ...]] = []
    for row in rows:
        values: list[object] = []
        for field in fields:
            value: object = row.get(field)
            if field == "horizon":
                try:
                    value = int(str(value))
                except ValueError as error:
                    raise R1FinalizationError(f"{name} horizon is invalid") from error
            values.append(value)
        cells.append(tuple(values))
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise R1FinalizationError(f"{name} coverage differs")


def _validate_evidence(artifact_root: Path) -> dict[str, object]:
    input_manifest = _read_json(artifact_root / "input_manifest.json")
    cache_manifest = _read_json(artifact_root / "cache_manifest.json")
    smoke = _read_json(artifact_root / "smoke_and_parity.json")
    for value, name in (
        (input_manifest, "input"),
        (cache_manifest, "cache"),
        (smoke, "smoke"),
    ):
        if value.get("status") != "pass":
            raise R1FinalizationError(f"{name} evidence did not pass")
    if (
        input_manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        or cache_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or input_manifest.get("protocol", {}).get("sha256") != PROTOCOL_SHA256
        or cache_manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or cache_manifest.get("local", {}).get("entry_count") != 645
        or cache_manifest.get("full_history", {}).get("entry_count") != 645
        or smoke.get("pair_count") != 3
    ):
        raise R1FinalizationError("frozen input or cache binding differs")

    metrics_root = artifact_root / "metrics"
    tables = {
        name: _read_csv(metrics_root / f"{name}.csv")
        for name in (
            "local_current",
            "task_per_sequence",
            "task_aggregate",
            "task_per_cluster",
            "identity_per_sequence",
            "identity_aggregate",
            "identity_per_cluster",
            "gap_event_ledger",
            "score_sensitivity",
            "checkpoint_regime_comparison",
        )
    }
    expected_lengths = {
        "local_current": 471,
        "task_per_sequence": 2709,
        "task_aggregate": 84,
        "task_per_cluster": 504,
        "identity_per_sequence": 774,
        "identity_aggregate": 24,
        "identity_per_cluster": 144,
        "score_sensitivity": 23,
    }
    for name, expected in expected_lengths.items():
        if len(tables[name]) != expected:
            raise R1FinalizationError(f"{name} row count differs")
    if not tables["gap_event_ledger"] or not tables["checkpoint_regime_comparison"]:
        raise R1FinalizationError("event or old-new evidence is empty")
    _validate_exact_cells(
        tables["task_aggregate"],
        expected=_expected_task_cells(),
        fields=("method", "score_reducer", "order_id", "horizon"),
        name="task aggregate",
    )
    expected_identity_cells = {
        (method, order, horizon)
        for method in ("B2", "B4")
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    }
    _validate_exact_cells(
        tables["identity_aggregate"],
        expected=expected_identity_cells,
        fields=("method", "order_id", "horizon"),
        name="identity aggregate",
    )

    def normalize_horizons(
        rows: Sequence[Mapping[str, str]],
    ) -> list[dict[str, object]]:
        return [{**row, "horizon": int(row["horizon"])} for row in rows]

    local_sequence_rows = [
        row for row in tables["local_current"] if row.get("scope") == "sequence"
    ]
    from scripts.analyze_r1_downstream_validation import (
        classify_recovery,
        validate_per_sequence_coverage,
    )

    coverage = validate_per_sequence_coverage(
        task_rows=normalize_horizons(tables["task_per_sequence"]),
        identity_rows=normalize_horizons(tables["identity_per_sequence"]),
        local_rows=normalize_horizons(local_sequence_rows),
    )
    if coverage != {
        "master_count": 43,
        "sequence_count": 129,
        "reference_cluster_count": 6,
        "task_row_count": 2709,
        "identity_row_count": 774,
        "local_row_count": 387,
    }:
        raise R1FinalizationError("per-sequence evidence coverage differs")

    references = {
        row["reference_scene_id"] for row in tables["identity_per_cluster"]
    }
    if len(references) != 6 or "" in references:
        raise R1FinalizationError("cluster identity coverage differs")
    _validate_exact_cells(
        tables["task_per_cluster"],
        expected={
            (*cell, reference)
            for cell in _expected_task_cells()
            for reference in references
        },
        fields=(
            "method",
            "score_reducer",
            "order_id",
            "horizon",
            "reference_scene_id",
        ),
        name="task cluster",
    )
    _validate_exact_cells(
        tables["identity_per_cluster"],
        expected={
            (*cell, reference)
            for cell in expected_identity_cells
            for reference in references
        },
        fields=("method", "order_id", "horizon", "reference_scene_id"),
        name="identity cluster",
    )
    normalized_identity = [
        {
            **row,
            "horizon": int(row["horizon"]),
            "gap_recovery_recall": (
                None
                if row.get("gap_recovery_recall", "") == ""
                else _finite(row["gap_recovery_recall"], field="gap_recovery_recall")
            ),
        }
        for row in tables["identity_aggregate"]
    ]
    normalized_clusters = [
        {
            **row,
            "horizon": int(row["horizon"]),
            "gap_recovery_recall": (
                None
                if row.get("gap_recovery_recall", "") == ""
                else _finite(row["gap_recovery_recall"], field="gap_recovery_recall")
            ),
        }
        for row in tables["identity_per_cluster"]
    ]
    recovery = classify_recovery(normalized_identity, normalized_clusters)
    historical = _read_json(artifact_root / "historical_replay_status.json")

    from scripts.profile_r1_downstream_validation import validate_profile_coverage

    samples = _read_csv(artifact_root / "profile/samples.csv")
    summaries = _read_csv(artifact_root / "profile/summary.csv")

    def normalize_profile(row: Mapping[str, str], *, sample: bool) -> dict[str, object]:
        normalized: dict[str, object] = {
            **row,
            "horizon": int(row["horizon"]),
            "warmup_repeats": int(row["warmup_repeats"]),
            "measured_repeats": int(row["measured_repeats"]),
        }
        if sample:
            normalized["sample_index"] = int(row["sample_index"])
        return normalized

    coverage = validate_profile_coverage(
        sample_rows=[normalize_profile(row, sample=True) for row in samples],
        summary_rows=[normalize_profile(row, sample=False) for row in summaries],
    )
    return {
        "tables": tables,
        "profile_rows": summaries,
        "profile_coverage": coverage,
        "recovery": recovery,
        "historical": historical,
        "input_manifest": input_manifest,
        "cache_manifest": cache_manifest,
    }


def _select_task(
    rows: Sequence[Mapping[str, str]], method: str, reducer: str, horizon: int
) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("score_reducer") == reducer
        and row.get("order_id") == "all"
        and row.get("horizon") == str(horizon)
    ]
    if len(selected) != 1:
        raise R1FinalizationError("task summary cell is unavailable")
    return selected[0]


def _resource_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result = []
    for method in ("FullHistory", "B4"):
        for horizon in (2, 4, 5):
            selected = [
                row
                for row in rows
                if row.get("method") == method and row.get("horizon") == str(horizon)
            ]
            if len(selected) != 6:
                raise R1FinalizationError("resource summary coverage differs")
            result.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "cluster_median_latency_ms": statistics.median(
                        _finite(row["median_latency_ms"], field="median_latency_ms")
                        for row in selected
                    ),
                    "max_peak_allocated_mib": max(
                        _finite(row["peak_allocated_mib"], field="peak_allocated_mib")
                        for row in selected
                    ),
                }
            )
    return result


def _render_report(
    *, evidence: Mapping[str, object], statuses: Mapping[str, str], source_commit: str
) -> bytes:
    recovery = evidence["recovery"]
    tables = evidence["tables"]
    resources = _resource_summary(evidence["profile_rows"])
    lines = [
        "# Persist4D R1 Downstream Validation Final Report",
        "",
        f"- Generation commit: `{source_commit}`",
        f"- R1 checkpoint: `{CHECKPOINT_SHA256}` (completed epoch 390, step 25740)",
        f"- Protocol-B: `{PROTOCOL_SHA256}`; 43 masters, 3 orders, 129 sequence-order units, 6 reference-scene clusters",
        f"- Execution: `{statuses['execution']}`",
        f"- Recovery: `{statuses['recovery']}`",
        f"- Historical comparison: `{statuses['old_new']}`",
        f"- Publication: `{statuses['publication']}`",
        "",
        "## Recovery Decision",
        "",
        "| Horizon | B2 recall | B4 recall | B4-B2 | Positive clusters | Pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for horizon in (4, 5):
        row = recovery[f"T{horizon}"]
        lines.append(
            f"| T{horizon} | {row['pooled_b2']:.6f} | {row['pooled_b4']:.6f} | {row['pooled_b4_minus_b2']:+.6f} | {row['positive_cluster_count']}/6 | {'yes' if row['supported'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The decision is the frozen pooled-positive and at-least-four-of-six-positive-clusters rule at both T4 and T5.",
            "",
            "## Task Evidence",
            "",
            "Pooled official causal-prefix metrics across all three registered orders:",
            "",
            "| Method | Reducer | Horizon | t-mAP | t-REC |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for method, reducer in (("FullHistory", "official"), ("B2", "mean"), ("B4", "mean")):
        for horizon in (2, 4, 5):
            row = _select_task(tables["task_aggregate"], method, reducer, horizon)
            lines.append(
                f"| {method} | {reducer} | T{horizon} | {_finite(row['causal_prefix_t_mAP'], field='t-mAP'):.6f} | {_finite(row['causal_prefix_t_REC'], field='t-REC'):.6f} |"
            )
    lines.extend(
        [
            "",
            "Mean/latest/max reducer results, per-sequence tables, six-cluster tables, identity counts, event ledger, and seed-45 10,000-replicate paired bootstrap results are in `metrics/`.",
            "",
            "## Resource Evidence",
            "",
            "Latency is the median of the six per-unit medians. Peak allocation is the maximum observed unit peak.",
            "",
            "| Method | Horizon | Latency (ms) | Peak allocated (MiB) |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in resources:
        lines.append(
            f"| {row['method']} | T{row['horizon']} | {row['cluster_median_latency_ms']:.3f} | {row['max_peak_allocated_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The A40 profile uses the first canonical master per cluster, 5 warmups and 10 measured repeats. It includes model forward and B4 tracking, while excluding I/O, collation, H2D, metric scoring, and tracker preroll.",
            "",
            "## Evidence Boundary",
            "",
            f"Historical raw replay is `{evidence['historical']['status']}`. Frozen C-old summaries remain available for descriptive checkpoint-regime comparison, but they do not support causal attribution of old-versus-R1 changes.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _render_handoff(
    *, statuses: Mapping[str, str], recovery: Mapping[str, object], source_commit: str
) -> bytes:
    lines = [
        "# R1 Downstream Validation Handoff",
        "",
        "## Result",
        "",
        f"The experiment is `{statuses['execution']}` and the registered recovery decision is `{statuses['recovery']}`.",
        f"T4 B4-B2 gap-recovery recall: `{recovery['T4']['pooled_b4_minus_b2']:+.6f}` with `{recovery['T4']['positive_cluster_count']}/6` positive clusters.",
        f"T5 B4-B2 gap-recovery recall: `{recovery['T5']['pooled_b4_minus_b2']:+.6f}` with `{recovery['T5']['positive_cluster_count']}/6` positive clusters.",
        "",
        "## Frozen Identities",
        "",
        f"- Source commit used for finalization: `{source_commit}`",
        f"- R1 checkpoint SHA256: `{CHECKPOINT_SHA256}`",
        f"- Protocol-B SHA256: `{PROTOCOL_SHA256}`",
        "- Runtime: one A40, float32, batch size 1, seed 45",
        "",
        "## Evidence Layout",
        "",
        "- `metrics/`: task, direct-local, identity, event, sensitivity, and old-new tables",
        "- `profile/`: all 360 measured samples and 36 six-unit summary cells",
        "- `FINAL_REPORT.md`: reviewer-facing compact interpretation",
        "- `FINAL_MANIFEST.json`: authoritative SHA256 inventory and claim boundary",
        "",
        "## Verification",
        "",
        "```bash",
        "PERSIST4D_PYTHON=/home/ww/miniconda3/envs/persist4d/bin/python",
        "$PERSIST4D_PYTHON -m pytest -q tests/test_r1_downstream_*.py",
        "$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py",
        "git diff --check",
        "git status --short",
        "```",
        "",
        "Large tensor caches are not in Git. Their compact identity is `cache_manifest.json`; the local external reference is `/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache`.",
        "",
        "## Claim Boundary",
        "",
        f"Publication status is `{statuses['publication']}`. Historical comparison is `{statuses['old_new']}`; do not make causal old-versus-R1 claims without reconstructing the missing C-old raw cache.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite final artifact: {path.name}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize(*, artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> Mapping[str, object]:
    from scripts.run_r1_downstream_validation import _require_clean_tracked_tree

    _require_clean_tracked_tree()
    evidence = _validate_evidence(artifact_root)
    statuses = derive_final_statuses(
        recovery=evidence["recovery"], historical=evidence["historical"]
    )
    source_commit = _git("rev-parse", "HEAD")
    report = _render_report(
        evidence=evidence, statuses=statuses, source_commit=source_commit
    )
    handoff = _render_handoff(
        statuses=statuses,
        recovery=evidence["recovery"],
        source_commit=source_commit,
    )
    output_payloads = {"FINAL_REPORT.md": report, "HANDOFF.md": handoff}
    upstream_payloads = {
        name: (artifact_root / name).read_bytes() for name in UPSTREAM_PATHS
    }
    manifest = build_final_manifest(
        source_commit=source_commit,
        statuses=statuses,
        checkpoint_sha256=CHECKPOINT_SHA256,
        protocol_sha256=PROTOCOL_SHA256,
        upstream_payloads=upstream_payloads,
        output_payloads=output_payloads,
    )
    output_payloads["FINAL_MANIFEST.json"] = _json_payload(manifest)
    for name, payload in output_payloads.items():
        _publish(artifact_root / name, payload)
    return {
        "status": "complete",
        "source_commit": source_commit,
        "statuses": statuses,
        "upstream_file_count": len(upstream_payloads),
        "output_file_count": len(output_payloads),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            finalize(artifact_root=arguments.artifact_root),
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
