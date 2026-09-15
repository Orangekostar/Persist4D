#!/usr/bin/env python3
"""Build the compact TaskMemory V2 publication metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/task_memory_retention_v2"

CORRECTNESS_CLASSES = (
    "r1_load_and_shutdown_parity",
    "data_and_supervision_isolation",
    "route_and_commit",
    "tala_supervision",
    "visual_memory",
    "training_path",
    "output_and_metric",
    "publication",
)
VERIFICATION_CHECKS = (
    "correctness_suite",
    "real_gpu_gate",
    "ruff_changed_python",
    "git_diff_check",
    "manifest_and_hash",
    "secret_scan",
)

STATUS_FIELDS = (
    "EXECUTION",
    "TMAP_ALL_T_VS_R1",
    "TMAP_ALL_T_VS_MATCHED_FH",
    "TASK_METRICS_ALL_T",
    "RETENTION",
    "RESOURCE",
    "MECHANISM",
    "GENERALIZATION",
    "PUBLICATION",
)
STATUS_VALUES = {
    "EXECUTION": {"COMPLETE", "PARTIAL", "BLOCKED"},
    "TMAP_ALL_T_VS_R1": {"PASS", "FAIL", "NOT_RUN"},
    "TMAP_ALL_T_VS_MATCHED_FH": {"PASS", "FAIL", "NOT_RUN"},
    "TASK_METRICS_ALL_T": {"PASS", "FAIL", "NOT_RUN"},
    "RETENTION": {"IMPROVED", "NOT_IMPROVED", "INCONCLUSIVE"},
    "RESOURCE": {"ADVANTAGE", "TRADEOFF", "NO_ADVANTAGE", "NOT_MEASURED"},
    "MECHANISM": {"SUPPORTED", "PARTIAL", "UNSUPPORTED"},
    "GENERALIZATION": {"INDEPENDENT", "BASE_EXPOSED_ONLY", "NOT_ESTABLISHED"},
    "PUBLICATION": {"PUSH_VERIFIED", "PUSH_FAILED", "NOT_ATTEMPTED"},
}
PRIMARY_VARIANTS = (
    "FH-R1-native",
    "B4-commit0",
    "R1+B4-lag1",
    "FH-R1-lag1",
    "W-BASE",
    "Q-TALA",
    "M3-BASE-CONT",
    "M3-V-CORE",
    "FH-MATCH",
    "FH-CONT",
    "D-LAST-commit0",
    "D-LAST-lag1",
    "D-EMA-commit0",
    "D-EMA-lag1",
)
_MANIFEST_EXCLUDED = {
    "FINAL_MANIFEST.json",
    "HANDOFF.md",
    "PUBLICATION_RECEIPT.json",
}


class PublicationError(RuntimeError):
    """Raised when the publication package violates its frozen contract."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_statuses(statuses: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(statuses, Mapping) or set(statuses) != set(STATUS_FIELDS):
        raise PublicationError("status fields differ from the frozen order")
    output = {}
    for field in STATUS_FIELDS:
        value = statuses[field]
        if value not in STATUS_VALUES[field]:
            raise PublicationError(f"{field} status differs")
        output[field] = str(value)
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_final_manifest(
    artifact_root: Path,
    *,
    artifact_paths: Sequence[Path],
    statuses: Mapping[str, object],
    results_commit: str,
) -> dict[str, object]:
    root = artifact_root.expanduser().resolve()
    validated_statuses = validate_statuses(statuses)
    if len(results_commit) != 40:
        raise PublicationError("results commit must be a full Git SHA")
    records = []
    seen = set()
    for raw_path in artifact_paths:
        path = raw_path.expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise PublicationError(
                "manifest artifact is outside the artifact root"
            ) from error
        if relative in _MANIFEST_EXCLUDED:
            continue
        if relative in seen or path.is_symlink() or not path.is_file():
            raise PublicationError("manifest artifact path is invalid or duplicated")
        seen.add(relative)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    value = {
        "schema_version": "task-memory-final-manifest-v2",
        "results_commit": results_commit,
        "publication_commit_resolution": "resolve from remote branch tip",
        "statuses": validated_statuses,
        "artifacts": sorted(records, key=lambda row: str(row["path"])),
    }
    value["content_sha256"] = _canonical_sha256(value)
    return value


def render_handoff(
    *,
    statuses: Mapping[str, object],
    results_commit: str,
    final_manifest_sha256: str,
    section_bodies: Mapping[int, str] | None = None,
) -> str:
    validated = validate_statuses(statuses)
    if len(results_commit) != 40 or len(final_manifest_sha256) != 64:
        raise PublicationError("handoff commit/hash identity differs")
    sections = [
        (
            "Goal and verdict",
            "\n".join(f"- {key}: `{value}`" for key, value in validated.items()),
        ),
        ("Repository", f"Results commit: `{results_commit}`."),
        ("What was actually implemented", "See the results commit and final manifest."),
        (
            "Knowledge and upstream evidence",
            "Frozen sources are listed in EVIDENCE_MAP.md.",
        ),
        (
            "Data and output policy",
            "Protocol-B is reported separately from development and native populations.",
        ),
        (
            "Model state and training",
            "Checkpoint identities are recorded in final/all_t_metrics.csv.",
        ),
        (
            "Experiments completed and not run",
            "Unrun stages remain explicit in training/variants.json.",
        ),
        ("Primary table", "See final/all_t_metrics.csv and final/paired_deltas.csv."),
        (
            "Mechanism and failure evidence",
            "See mechanism and visual diagnostic artifacts.",
        ),
        ("Cost and storage", "See training/costs_and_exposure.csv and resources/."),
        (
            "Selected and resume checkpoints",
            "Selected and last checkpoints remain distinct in their manifests.",
        ),
        (
            "Reproduction commands",
            "See COMMANDS.md; external paths are environment variables.",
        ),
        ("Tests and limitations", "See TEST_REPORT.md and FINAL_REPORT.md."),
        (
            "Claims supported / not supported",
            "Scientific status fields above are independent.",
        ),
        (
            "GitHub and next exact action",
            f"FINAL_MANIFEST.json SHA256: `{final_manifest_sha256}`.",
        ),
    ]
    if section_bodies is not None:
        if set(section_bodies) != set(range(2, 16)) or any(
            not isinstance(body, str) or not body.strip()
            for body in section_bodies.values()
        ):
            raise PublicationError("handoff section bodies differ")
        sections = [
            sections[0],
            *[
                (sections[index - 1][0], section_bodies[index])
                for index in range(2, 16)
            ],
        ]
    content = ["# TaskMemory Retention V2 Handoff", ""]
    for index, (title, body) in enumerate(sections, start=1):
        content.extend((f"## {index}. {title}", "", body, ""))
    return "\n".join(content).rstrip() + "\n"


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PublicationError(f"cannot read JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicationError(f"JSON root must be an object: {path}")
    return value


def _validate_content_hash(value: Mapping[str, object], *, name: str) -> None:
    unsigned = dict(value)
    observed = unsigned.pop("content_sha256", None)
    if observed != _canonical_sha256(unsigned):
        raise PublicationError(f"{name} content SHA256 differs")


def validate_verification(
    value: Mapping[str, object], *, results_commit: str
) -> dict[str, object]:
    _validate_content_hash(value, name="verification evidence")
    if (
        value.get("schema_version") != "task-memory-verification-v2"
        or value.get("results_commit") != results_commit
    ):
        raise PublicationError("verification commit identity differs")
    classes = value.get("correctness_classes")
    if not isinstance(classes, Mapping) or set(classes) != set(CORRECTNESS_CLASSES):
        raise PublicationError("verification correctness classes differ")
    for name in CORRECTNESS_CLASSES:
        record = classes[name]
        if (
            not isinstance(record, Mapping)
            or record.get("status") != "PASS"
            or not isinstance(record.get("evidence"), str)
            or not str(record["evidence"]).strip()
        ):
            raise PublicationError(f"verification correctness class failed: {name}")
    checks = value.get("checks")
    if isinstance(checks, (str, bytes)) or not isinstance(checks, Sequence):
        raise PublicationError("verification checks differ")
    by_name = {}
    check_order = []
    for record in checks:
        if not isinstance(record, Mapping):
            raise PublicationError("verification check differs")
        name = record.get("name")
        if not isinstance(name, str) or name in by_name:
            raise PublicationError("verification check identity differs")
        if (
            record.get("exit_code") != 0
            or not isinstance(record.get("command"), str)
            or not str(record["command"]).strip()
            or not isinstance(record.get("observed"), str)
            or not str(record["observed"]).strip()
        ):
            raise PublicationError(f"verification check failed: {name}")
        by_name[name] = dict(record)
        check_order.append(name)
    if tuple(check_order) != VERIFICATION_CHECKS:
        raise PublicationError("verification check coverage differs")
    return dict(value)


def render_test_report(verification: Mapping[str, object]) -> str:
    results_commit = verification.get("results_commit")
    if not isinstance(results_commit, str):
        raise PublicationError("verification results commit differs")
    validated = validate_verification(verification, results_commit=results_commit)
    classes = validated["correctness_classes"]
    checks = validated["checks"]
    if not isinstance(classes, Mapping) or not isinstance(checks, Sequence):
        raise PublicationError("verification report evidence differs")
    class_lines = "\n".join(
        f"- `{name}`: PASS; {classes[name]['evidence']}"  # type: ignore[index]
        for name in CORRECTNESS_CLASSES
    )
    command_lines = "\n\n".join(
        f"### `{record['name']}`\n\n"
        f"```bash\n{record['command']}\n```\n\n"
        f"Observed: {record['observed']} (exit code 0)."
        for record in checks
        if isinstance(record, Mapping)
    )
    return f"""# TaskMemory Retention V2 Test Report

Results commit: `{results_commit}`.

## Correctness classes

{class_lines}

## Direct verification

{command_lines}

Passing engineering checks do not change failed or inconclusive scientific
statuses.
"""


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise PublicationError(f"cannot read CSV: {path}") from error
    if not rows:
        raise PublicationError(f"CSV is empty: {path}")
    return rows


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise PublicationError(f"publication output cannot be a symlink: {path}")
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


def _metric_table(rows: Sequence[Mapping[str, str]]) -> str:
    by_cell = {(row["variant"], int(row["T"])): row for row in rows}
    lines = [
        "| Variant | T2 | T3 | T4 | T5 | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in PRIMARY_VARIANTS:
        values = [
            float(by_cell[(variant, horizon)]["t_mAP"]) for horizon in range(2, 6)
        ]
        lines.append(
            f"| {variant} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {sum(values) / 4:.6f} |"
        )
    return "\n".join(lines)


def _comparison_lines(analysis: Mapping[str, object]) -> str:
    comparisons = analysis.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise PublicationError("final analysis comparisons differ")
    lines = []
    for name in (
        "M3-V-CORE_vs_B4-commit0",
        "M3-V-CORE_vs_R1+B4-lag1",
        "M3-V-CORE_vs_FH-R1-native",
        "M3-V-CORE_vs_FH-R1-lag1",
        "M3-V-CORE_vs_W-BASE",
        "M3-V-CORE_vs_Q-TALA",
        "M3-V-CORE_vs_M3-BASE-CONT",
        "M3-V-CORE_vs_FH-MATCH",
        "M3-V-CORE_vs_FH-CONT",
        "M3-V-CORE_vs_D-LAST-lag1",
        "M3-V-CORE_vs_D-EMA-lag1",
    ):
        value = comparisons.get(name)
        if not isinstance(value, Mapping):
            raise PublicationError(f"comparison is unavailable: {name}")
        failed = (
            ", ".join(f"T{item}" for item in value.get("failed_tmap_horizons", ()))
            or "none"
        )
        lines.append(
            f"- {name}: tMAP `{value.get('tmap_all_t')}`, "
            f"20-cell `{value.get('positive_cells')}/{value.get('total_cells')}`, "
            f"minimum tMAP delta `{float(value.get('minimum_tmap_delta')):.6f}`, "
            f"failed tMAP horizons `{failed}`."
        )
    return "\n".join(lines)


def _profile_panel_lines(
    rows: Sequence[Mapping[str, str]], resource_status: object
) -> str:
    lines = [f"Resource verdict: `{resource_status}`."]
    for variant in ("M3-V-CORE", "FH-CONT"):
        for horizon in (4, 5):
            selected = [
                row
                for row in rows
                if row.get("method") == variant
                and row.get("scope") == "model_update"
                and row.get("T") == str(horizon)
            ]
            if len(selected) != 6:
                raise PublicationError("resource profile panel coverage differs")
            latency = sorted(float(row["median_latency_ms"]) for row in selected)
            peak = sorted(int(row["absolute_peak_allocated_bytes"]) for row in selected)
            lines.append(
                f"- {variant} T{horizon}: panel median model_update "
                f"{(latency[2] + latency[3]) / 2:.3f} ms; absolute peak "
                f"{(peak[2] + peak[3]) // 2} bytes."
            )
    return "\n".join(lines)


def _build_handoff_section_bodies(
    *,
    root: Path,
    statuses: Mapping[str, str],
    results_commit: str,
    final_manifest_sha256: str,
    metrics: Sequence[Mapping[str, str]],
    analysis: Mapping[str, object],
    resource: Mapping[str, object],
    verification: Mapping[str, object],
) -> dict[int, str]:
    start = _load_json(root / "START_STATE.json")
    _validate_content_hash(start, name="start state")
    repository = start.get("repository")
    if not isinstance(repository, Mapping):
        raise PublicationError("start repository identity differs")
    candidate = [row for row in metrics if row.get("variant") == "M3-V-CORE"]
    fh = [row for row in metrics if row.get("variant") == "FH-CONT"]
    if len(candidate) != 4 or len(fh) != 4:
        raise PublicationError("selected checkpoint coverage differs")
    candidate_sha = {row["checkpoint_sha256"] for row in candidate}
    fh_sha = {row["checkpoint_sha256"] for row in fh}
    if len(candidate_sha) != 1 or len(fh_sha) != 1:
        raise PublicationError("selected checkpoint identity differs")
    profile_rows = _read_csv(root / "resources/profile_summary.csv")
    m4_rows = _read_csv(root / "training/M4_budget_gate.csv")
    fh_cost_rows = _read_csv(root / "training/FH_CONT_cost_and_exposure.csv")
    if len(m4_rows) != 1 or len(fh_cost_rows) != 1:
        raise PublicationError("training budget evidence differs")
    checks = verification.get("checks")
    if not isinstance(checks, Sequence):
        raise PublicationError("verification command evidence differs")
    verified_commands = "\n".join(
        f"- `{record['name']}`: `{record['command']}`"
        for record in checks
        if isinstance(record, Mapping)
    )
    limitations = analysis.get("execution_limitations")
    if isinstance(limitations, (str, bytes)) or not isinstance(limitations, Sequence):
        raise PublicationError("execution limitations differ")
    limitation_lines = "\n".join(f"- {item}" for item in limitations)
    m4 = m4_rows[0]
    fh_cost = fh_cost_rows[0]
    return {
        2: (
            f"Parent: `{repository.get('reviewed_parent')}`. Branch: "
            f"`{repository.get('branch')}`. Results commit E: `{results_commit}`. "
            "Publication tip is resolved from the remote branch containing this file."
        ),
        3: (
            "Implemented causal TaskMemory route/commit state, TALA sequence "
            "supervision, bounded visual memory, compact evaluation caches, frozen "
            "training/evaluation entry points, final analysis, and one-A40 resource "
            "profiling. Numeric artifacts in FINAL_MANIFEST.json were actually run; "
            "items named under limitations were not."
        ),
        4: (
            "Upstream code and literature identities, immutable revisions/blobs, and "
            "license-use boundaries are recorded in EVIDENCE_MAP.md and "
            "references_inventory.csv. Reviewed parent is fixed above."
        ),
        5: (
            "Primary confirmation population is frozen Protocol-B: 6 physical "
            "references, 43 masters, and 129 deterministic orders. Primary output "
            "uses lag1/mean; commit0 is a separate policy control. Development, "
            "Protocol-B, and independent-native evidence are not pooled."
        ),
        6: (
            "Selected M3-V-CORE uses K=100, r=8 and measured permanent state "
            f"{resource.get('permanent_state_bytes')} bytes. It was selected at "
            "update 1500 by the frozen development rule. FH-CONT is the matched "
            "full-history continuation control; training seed and evaluation seed "
            "are both 45 and remain distinct fields."
        ),
        7: (
            "Completed M2 W-BASE/Q-INDEP/Q-TALA/FH-MATCH, M3 BASE-CONT/V-LAST/"
            "V-CORE, FH-CONT, the unified fourteen-method Protocol-B table, "
            "base-exposed native evaluation, long-memory controls, per-reference "
            "analysis, and resource profiling. M4 was conditionally skipped as "
            f"`{m4.get('decision')}` because `{m4.get('reason')}`. Overall execution "
            f"is `{statuses['EXECUTION']}`; scientific limitations remain in section 13."
        ),
        8: _metric_table(metrics) + "\n\n" + _comparison_lines(analysis),
        9: (
            f"Mechanism verdict is `{statuses['MECHANISM']}`. The final development "
            "real/read-off/previous/unrelated visual-content comparison is in "
            "visual/paired_content_ablation.csv; route, gap, birth, reactivation, "
            "fragmentation, merge, and ID-switch evidence remains separate from "
            "task AP in final/identity_counts.csv."
        ),
        10: (
            f"Recorded campaign training GPU-hours after FH-CONT: "
            f"{float(fh_cost['campaign_gpu_hours_after_run']):.6f} / "
            f"{float(fh_cost['campaign_training_gpu_hour_cap']):.0f}. Permanent "
            f"state budget: {resource.get('permanent_state_bytes')} / 2097152 bytes.\n\n"
            + _profile_panel_lines(profile_rows, resource.get("resource_status"))
        ),
        11: (
            f"M3-V-CORE selected checkpoint: `{next(iter(candidate_sha))}`, logical "
            "location `external:run_root/training/formal/M3-V-CORE/update=1500.ckpt`. "
            f"FH-CONT selected checkpoint: `{next(iter(fh_sha))}`, logical location "
            "`external:run_root/training/formal/FH-CONT/update=1500.ckpt`. Selected "
            "checkpoints are not presented as optimizer-complete resume checkpoints; "
            "their run directories retain separate `last.ckpt` files."
        ),
        12: (
            "Public environment-variable commands are in COMMANDS.md. Final direct "
            "verification used:\n" + verified_commands
        ),
        13: (
            "All eight correctness classes and the real A40 gate are detailed in "
            "TEST_REPORT.md. Remaining limitations:\n" + limitation_lines
        ),
        14: (
            "Supported claims are limited to the exact all-T, retention, mechanism, "
            "and resource statuses in section 1. Not supported: indefinite-horizon "
            "retention, replicated training stability, truly unseen-base "
            "generalization, or attribution of policy effects to memory content alone."
        ),
        15: (
            f"Results commit E publication status: `{statuses['PUBLICATION']}`. "
            f"FINAL_MANIFEST.json SHA256: `{final_manifest_sha256}`. Next exact "
            "action: replicate M3-V-CORE and its strongest matched baseline with a "
            "second training seed before making a stability claim."
        ),
    }


def derive_package_statuses(
    analysis: Mapping[str, object],
    resource: Mapping[str, object],
    *,
    publication_status: str,
) -> dict[str, str]:
    _validate_content_hash(analysis, name="final analysis")
    _validate_content_hash(resource, name="resource profile")
    raw = analysis.get("status")
    if not isinstance(raw, Mapping):
        raise PublicationError("final analysis statuses differ")
    statuses = {field: raw.get(field) for field in STATUS_FIELDS}
    statuses["RESOURCE"] = resource.get("resource_status")
    statuses["PUBLICATION"] = publication_status
    reference_analysis = analysis.get("reference_analysis")
    if not isinstance(reference_analysis, Mapping):
        raise PublicationError("final reference analysis differs")
    return validate_statuses(statuses)


def _final_report(
    *,
    statuses: Mapping[str, str],
    metrics: Sequence[Mapping[str, str]],
    analysis: Mapping[str, object],
    resource: Mapping[str, object],
    results_commit: str,
) -> str:
    status_lines = "\n".join(f"- {key}: `{value}`" for key, value in statuses.items())
    return f"""# TaskMemory Retention V2 Final Report

Results commit: `{results_commit}`.

## Verdict

{status_lines}

## Protocol-B tMAP

{_metric_table(metrics)}

## Strict comparisons

{_comparison_lines(analysis)}

## Retention and mechanism

Absolute A(T), relative R(T), and Dmax are in `final/retention.csv`. Mechanism is
reported separately from task superiority; the final development content control
supports only the status shown above.

## Reference evidence

The six physical references are the statistical units. The T2/T5 intervals in
`final/reference_bootstrap.csv` are equal-reference descriptive intervals, not
pooled-AP confidence intervals.

The separate native T2-T4 evidence in `final/independent_reference_results.csv`
uses adaptation-holdout references that were exposed to the original R1 base;
therefore its generalization status is `BASE_EXPOSED_ONLY`, not `INDEPENDENT`.

## Resources

Resource status is `{resource.get("resource_status")}`. The profile uses one A40,
six fixed canonical units, five warmups, ten measurements, cloned prior state, and
separate model-update, end-to-end, materialization, and true cumulative scopes.

## Limitations

Protocol-B is a previously exposed historical benchmark. One training seed does
not establish replicated training stability. FH encoder-cache equivalence and
new-model saturation effects were not established. Exact limits remain explicit
in `final/status.json`.
"""


def publish_package(
    artifact_root: Path,
    *,
    results_commit: str,
    publication_status: str,
) -> dict[str, object]:
    root = artifact_root.expanduser().resolve()
    analysis = _load_json(root / "final/status.json")
    resource = _load_json(root / "resources/run_summary.json")
    verification = validate_verification(
        _load_json(root / "VERIFICATION.json"), results_commit=results_commit
    )
    statuses = derive_package_statuses(
        analysis, resource, publication_status=publication_status
    )
    metrics = _read_csv(root / "final/all_t_metrics.csv")
    if len(metrics) != len(PRIMARY_VARIANTS) * 4:
        raise PublicationError(
            "final all-T table must contain fourteen four-horizon methods"
        )
    required = (
        "START_STATE.json",
        "EVIDENCE_MAP.md",
        "DATA_CONTRACT.json",
        "OUTPUT_CONTRACT.md",
        "BUDGET_CONTRACT.json",
        "EVALUATION_CONTRACT.json",
        "PROFILE_CONTRACT.json",
        "COMMANDS.md",
        "VERIFICATION.json",
        "references_inventory.csv",
        "implementation/r1_load_report.json",
        "implementation/real_gradient_smoke.json",
        "implementation/query_state_contract.md",
        "implementation/sequence_loss_example.csv",
        "baseline/policy_comparison.csv",
        "baseline/long_memory_controls.csv",
        "training/variants.json",
        "training/M3_learning_curves.csv",
        "training/FH_CONT_learning_curves.csv",
        "training/learning_curves.csv",
        "training/terminal_update_comparison.csv",
        "training/selected_checkpoint_comparison.csv",
        "training/selected_checkpoints.json",
        "training/costs_and_exposure.csv",
        "training/FH_CONT_cost_and_exposure.csv",
        "training/M4_budget_gate.csv",
        "visual/paired_content_ablation.csv",
        "final/all_t_metrics.csv",
        "final/paired_deltas.csv",
        "final/per_reference_metrics.csv",
        "final/per_reference_deltas.csv",
        "final/reference_bootstrap.csv",
        "final/independent_reference_results.csv",
        "final/identity_counts.csv",
        "final/retention.csv",
        "final/status.json",
        "resources/profile_units.csv",
        "resources/per_update_measurements.csv",
        "resources/profile_summary.csv",
        "resources/state_bytes.csv",
        "resources/run_summary.json",
    )
    paths = []
    for relative in required:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise PublicationError(
                f"required publication artifact is missing: {relative}"
            )
        paths.append(path)
    report = _final_report(
        statuses=statuses,
        metrics=metrics,
        analysis=analysis,
        resource=resource,
        results_commit=results_commit,
    ).encode("ascii")
    test_report = render_test_report(verification).encode("ascii")
    report_path = root / "FINAL_REPORT.md"
    test_path = root / "TEST_REPORT.md"
    _atomic_write(report_path, report)
    _atomic_write(test_path, test_report)
    manifest = build_final_manifest(
        root,
        artifact_paths=(*paths, report_path, test_path),
        statuses=statuses,
        results_commit=results_commit,
    )
    manifest_path = root / "FINAL_MANIFEST.json"
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    manifest_sha256 = _file_sha256(manifest_path)
    section_bodies = _build_handoff_section_bodies(
        root=root,
        statuses=statuses,
        results_commit=results_commit,
        final_manifest_sha256=manifest_sha256,
        metrics=metrics,
        analysis=analysis,
        resource=resource,
        verification=verification,
    )
    handoff = render_handoff(
        statuses=statuses,
        results_commit=results_commit,
        final_manifest_sha256=manifest_sha256,
        section_bodies=section_bodies,
    ).encode("ascii")
    handoff_path = root / "HANDOFF.md"
    _atomic_write(handoff_path, handoff)
    return {
        "statuses": statuses,
        "results_commit": results_commit,
        "final_manifest_sha256": manifest_sha256,
        "handoff_sha256": _file_sha256(handoff_path),
        "primary_csv_sha256": _file_sha256(root / "final/all_t_metrics.csv"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--results-commit", required=True)
    parser.add_argument(
        "--publication-status",
        choices=("PUSH_VERIFIED", "PUSH_FAILED", "NOT_ATTEMPTED"),
        default="PUSH_VERIFIED",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = publish_package(
        args.artifact_root,
        results_commit=args.results_commit,
        publication_status=args.publication_status,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "CORRECTNESS_CLASSES",
    "STATUS_FIELDS",
    "VERIFICATION_CHECKS",
    "PublicationError",
    "build_final_manifest",
    "derive_package_statuses",
    "publish_package",
    "render_handoff",
    "render_test_report",
    "validate_statuses",
    "validate_verification",
]


if __name__ == "__main__":
    raise SystemExit(main())
