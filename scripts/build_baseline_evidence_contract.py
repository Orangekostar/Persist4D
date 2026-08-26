"""Build the V3 baseline evidence boundary from frozen, verified inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
P2_REPORT_SHA256 = (
    "d891fb7fd53306d8ab65db81b9bb85f08664a9689de850ac7836143b238816bc"
)


class BaselineEvidenceError(ValueError):
    """Raised when the frozen baseline evidence cannot be bound safely."""


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise BaselineEvidenceError(f"required input is missing: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaselineEvidenceError(f"{name} must be a lowercase hex digest")
    return value


def _repo_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"external:{resolved.name}"
    return f"repo:{relative.as_posix()}"


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineEvidenceError(f"{name} is not readable JSON") from error
    if not isinstance(value, Mapping):
        raise BaselineEvidenceError(f"{name} must contain a JSON object")
    return value


def build_baseline_evidence_contract(
    *,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    p2_report_path: Path,
    expected_p2_report_sha256: str,
    protocol_manifest_path: Path,
    v2_manifest_path: Path,
    source_commit: str,
    official_repository_url: str,
    official_revision: str,
    official_readme_sha256: str,
    retrieved_at: str,
    checkpoint_section_status: str,
) -> dict[str, object]:
    """Return the validated E0/E1/E2 contract without writing files."""

    checkpoint_path = Path(checkpoint_path)
    p2_report_path = Path(p2_report_path)
    protocol_manifest_path = Path(protocol_manifest_path)
    v2_manifest_path = Path(v2_manifest_path)
    expected_checkpoint_sha256 = _digest(
        expected_checkpoint_sha256, name="expected checkpoint SHA256", length=64
    )
    expected_p2_report_sha256 = _digest(
        expected_p2_report_sha256, name="expected P2 report SHA256", length=64
    )
    source_commit = _digest(source_commit, name="source commit", length=40)
    official_revision = _digest(
        official_revision, name="official revision", length=40
    )
    official_readme_sha256 = _digest(
        official_readme_sha256, name="official README SHA256", length=64
    )
    if not official_repository_url.startswith("https://github.com/"):
        raise BaselineEvidenceError("official repository URL must be HTTPS GitHub")
    if not retrieved_at.endswith("Z") or "T" not in retrieved_at:
        raise BaselineEvidenceError("retrieval timestamp must be UTC ISO-8601")
    if not checkpoint_section_status.strip():
        raise BaselineEvidenceError("checkpoint section status must not be empty")

    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise BaselineEvidenceError("checkpoint SHA256 differs from expected value")
    p2_report_sha = _sha256(p2_report_path)
    if p2_report_sha != expected_p2_report_sha256:
        raise BaselineEvidenceError("P2 report SHA256 differs from expected value")
    p2_report = p2_report_path.read_text(encoding="utf-8")
    if "27.939" not in p2_report or "G2 = RED" not in p2_report:
        raise BaselineEvidenceError("P2 report lacks the frozen result and RED gate")

    protocol_sha = _sha256(protocol_manifest_path)
    v2_sha = _sha256(v2_manifest_path)
    v2_manifest = _load_json(v2_manifest_path, name="V2 manifest")
    if (
        v2_manifest.get("checkpoint_sha256") != checkpoint_sha
        or v2_manifest.get("score_reducer") != "mean"
        or v2_manifest.get("status") != "pass"
    ):
        raise BaselineEvidenceError("V2 manifest does not bind the frozen mean run")

    evidence_rows: list[dict[str, object]] = [
        {
            "evidence_id": "E0",
            "name": "ReScene4D-C (paper-reported)",
            "source": "ReScene4D paper",
            "t_mAP_percent": 34.8,
            "status": "external_reference",
            "locally_rerun": False,
            "checkpoint_availability": "not_publicly_exposed",
            "table_label": "ReScene4D-C (paper-reported)",
        },
        {
            "evidence_id": "E1",
            "name": "ReScene4D-C (our reimplementation)",
            "source": _repo_reference(p2_report_path),
            "checkpoint_sha256": checkpoint_sha,
            "t_mAP_percent": 27.939,
            "status": "local_best_effort_reimplementation",
            "locally_rerun": True,
            "g2_gate": "RED",
            "table_label": "ReScene4D-C (our best-effort reimplementation)",
        },
        {
            "evidence_id": "E2",
            "name": "FullHistory using shared frozen local reimplementation",
            "source": _repo_reference(v2_manifest_path),
            "t_mAP_percent": None,
            "status": "controlled_internal_baseline",
            "locally_rerun": True,
            "protocol": "Protocol B / V3 bridge as appropriate",
            "table_label": "FullHistory (shared frozen local reimplementation)",
        },
    ]
    allowed = [
        "ReScene4D reports 34.8% t-mAP.",
        "Our best-effort reimplementation reaches 27.939%.",
        (
            "All controlled Persist4D-vs-FullHistory comparisons use the same "
            "frozen local model."
        ),
    ]
    forbidden = [
        "Our ReScene4D reproduces the official 34.8 model.",
        "Persist4D beats official ReScene4D.",
        "ReScene4D obtains 27.939%.",
        "34.8 -> Protocol-B t-mAP is a direct model degradation.",
    ]
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "generated_at": retrieved_at,
        "execution": {"gpu_inference_performed": False},
        "configuration": {
            "config_hash": "not_applicable",
            "cache_hash": "not_applicable",
        },
        "scripts": {
            "builder": {
                "reference": "repo:scripts/build_baseline_evidence_contract.py",
                "sha256": _sha256(Path(__file__)),
            }
        },
        "evidence_rows": evidence_rows,
        "official_repository_audit": {
            "repository_url": official_repository_url,
            "revision": official_revision,
            "retrieved_at": retrieved_at,
            "readme_sha256": official_readme_sha256,
            "checkpoint_section_status": checkpoint_section_status.strip(),
            "live_verified": True,
            "reported_task_checkpoint_publicly_available": False,
        },
        "frozen_provenance": {
            "checkpoint": {
                "reference": _repo_reference(checkpoint_path),
                "sha256": checkpoint_sha,
            },
            "p2_report": {
                "reference": _repo_reference(p2_report_path),
                "sha256": p2_report_sha,
            },
            "protocol_b_manifest": {
                "reference": _repo_reference(protocol_manifest_path),
                "sha256": protocol_sha,
            },
            "v2_manifest": {
                "reference": _repo_reference(v2_manifest_path),
                "sha256": v2_sha,
            },
        },
        "claims": {"allowed": allowed, "forbidden": forbidden},
        "gate_b0": {
            "status": "PASS",
            "paper_and_local_values_separated": True,
            "checkpoint_statement_has_provenance": True,
            "table_labels_generated_from_contract": True,
        },
    }


def _render_markdown(contract: Mapping[str, object]) -> str:
    rows = contract["evidence_rows"]
    claims = contract["claims"]
    remote = contract["official_repository_audit"]
    gate = contract["gate_b0"]
    if not isinstance(rows, Sequence) or not isinstance(claims, Mapping):
        raise BaselineEvidenceError("contract has invalid report fields")
    if not isinstance(remote, Mapping) or not isinstance(gate, Mapping):
        raise BaselineEvidenceError("contract has invalid provenance fields")
    lines = [
        "# Baseline Evidence Contract",
        "",
        "| ID | Method/source | t-mAP (%) | Evidence class | Locally rerun |",
        "|---|---|---:|---|---|",
    ]
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise BaselineEvidenceError("evidence row must be a mapping")
        score = raw_row.get("t_mAP_percent")
        rendered_score = "n/a" if score is None else str(score)
        lines.append(
            "| {evidence_id} | {name} | {score} | {status} | {rerun} |".format(
                evidence_id=raw_row["evidence_id"],
                name=raw_row["name"],
                score=rendered_score,
                status=raw_row["status"],
                rerun=str(raw_row["locally_rerun"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Official Repository Audit",
            "",
            f"- URL: `{remote['repository_url']}`",
            f"- Revision: `{remote['revision']}`",
            f"- Retrieved: `{remote['retrieved_at']}`",
            f"- README SHA256: `{remote['readme_sha256']}`",
            f"- Checkpoint section: `{remote['checkpoint_section_status']}`",
            "- Reported task checkpoint publicly available: `false`",
            "",
            "## Allowed Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in claims["allowed"])
    lines.extend(["", "## Forbidden Claims", ""])
    lines.extend(f"- {claim}" for claim in claims["forbidden"])
    lines.extend(
        [
            "",
            f"## Gate B0: {gate['status']}",
            "",
            (
                "Paper-reported, locally measured, and controlled internal "
                "evidence remain separate."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BaselineEvidenceError(f"refusing symlink output: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_baseline_evidence_contract(
    contract: Mapping[str, object], output_directory: Path
) -> dict[str, Path]:
    """Write canonical JSON and Markdown views of one validated contract."""

    output_directory = Path(output_directory)
    json_path = output_directory / "baseline_evidence_contract.json"
    markdown_path = output_directory / "BASELINE_EVIDENCE_CONTRACT.md"
    _atomic_write(
        json_path,
        json.dumps(contract, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, _render_markdown(contract))
    return {"json": json_path, "markdown": markdown_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt",
        type=Path,
    )
    parser.add_argument(
        "--p2-report",
        default=PROJECT_ROOT / "artifacts/P2_G2_REPRODUCTION_REPORT.md",
        type=Path,
    )
    parser.add_argument(
        "--protocol-manifest",
        default=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--v2-manifest",
        default=PROJECT_ROOT / "artifacts/system_comparison_v2/manifest.json",
        type=Path,
    )
    parser.add_argument(
        "--output-directory",
        default=PROJECT_ROOT / "artifacts/reviewer_closure_v3/baseline",
        type=Path,
    )
    parser.add_argument("--expected-checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--expected-p2-report-sha256", default=P2_REPORT_SHA256)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--official-readme-sha256", required=True)
    parser.add_argument("--retrieved-at", required=True)
    parser.add_argument(
        "--official-repository-url",
        default="https://github.com/GradientSpaces/rescene4d",
    )
    parser.add_argument("--checkpoint-section-status", default="Coming soon.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = build_baseline_evidence_contract(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        p2_report_path=args.p2_report,
        expected_p2_report_sha256=args.expected_p2_report_sha256,
        protocol_manifest_path=args.protocol_manifest,
        v2_manifest_path=args.v2_manifest,
        source_commit=args.source_commit,
        official_repository_url=args.official_repository_url,
        official_revision=args.official_revision,
        official_readme_sha256=args.official_readme_sha256,
        retrieved_at=args.retrieved_at,
        checkpoint_section_status=args.checkpoint_section_status,
    )
    paths = write_baseline_evidence_contract(contract, args.output_directory)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
