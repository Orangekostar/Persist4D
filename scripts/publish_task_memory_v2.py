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
) -> str:
    validated = validate_statuses(statuses)
    if len(results_commit) != 40 or len(final_manifest_sha256) != 64:
        raise PublicationError("handoff commit/hash identity differs")
    sections = (
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
    )
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
    wanted = ("R1+B4", "FH-R1", "M3-BASE-CONT", "FH-CONT", "M3-V-CORE")
    by_cell = {(row["variant"], int(row["T"])): row for row in rows}
    lines = [
        "| Variant | T2 | T3 | T4 | T5 | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in wanted:
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
        "M3-V-CORE_vs_R1+B4",
        "M3-V-CORE_vs_FH-R1",
        "M3-V-CORE_vs_M3-BASE-CONT",
        "M3-V-CORE_vs_FH-CONT",
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
            f"failed tMAP horizons `{failed}`."
        )
    return "\n".join(lines)


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

## Resources

Resource status is `{resource.get("resource_status")}`. The profile uses one A40,
six fixed canonical units, five warmups, ten measurements, cloned prior state, and
separate model-update, end-to-end, materialization, and true cumulative scopes.

## Limitations

Protocol-B is a previously exposed historical benchmark. One training seed does
not establish replicated training stability. Unrun variants and independent-data
limits remain explicit in `final/status.json` and `training/variants.json`.
"""


def _test_report(results_commit: str) -> str:
    return f"""# TaskMemory Retention V2 Test Report

Results commit: `{results_commit}`.

The direct verification commands and their observed results are recorded in the
results commit. Passing engineering checks do not change failed or inconclusive
scientific statuses.
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
    statuses = derive_package_statuses(
        analysis, resource, publication_status=publication_status
    )
    metrics = _read_csv(root / "final/all_t_metrics.csv")
    if len(metrics) != 20:
        raise PublicationError(
            "final all-T table must contain five four-horizon models"
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
        "references_inventory.csv",
        "implementation/r1_load_report.json",
        "implementation/real_gradient_smoke.json",
        "implementation/query_state_contract.md",
        "implementation/sequence_loss_example.csv",
        "baseline/policy_comparison.csv",
        "training/variants.json",
        "training/M3_learning_curves.csv",
        "training/FH_CONT_learning_curves.csv",
        "training/costs_and_exposure.csv",
        "training/FH_CONT_cost_and_exposure.csv",
        "training/M4_budget_gate.csv",
        "visual/paired_content_ablation.csv",
        "final/all_t_metrics.csv",
        "final/paired_deltas.csv",
        "final/per_reference_metrics.csv",
        "final/per_reference_deltas.csv",
        "final/reference_bootstrap.csv",
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
    test_report = _test_report(results_commit).encode("ascii")
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
    handoff = render_handoff(
        statuses=statuses,
        results_commit=results_commit,
        final_manifest_sha256=manifest_sha256,
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
    "STATUS_FIELDS",
    "PublicationError",
    "build_final_manifest",
    "derive_package_statuses",
    "publish_package",
    "render_handoff",
    "validate_statuses",
]


if __name__ == "__main__":
    raise SystemExit(main())
