#!/usr/bin/env python3
"""Build and verify the final ReScene root-cause handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_rootcause_preflight import canonical_sha256

EXPERIMENT = "rescene_task_learning_root_cause_v1"
BRANCH = "research/persist4d-rescene-task-learning-root-cause-v1"
MANIFEST_NAME = "FINAL_MANIFEST.json"

HANDOFF_SECTION_TITLES = (
    "Repository / branch / start SHA / end SHA / remote SHA",
    "Scientific question",
    "External evidence frozen",
    "Current Concerto/Sonata baseline metrics",
    "Exact files changed",
    "Root-cause hypotheses audited",
    "DDP sampler verdict",
    "Label-255 verdict",
    "Encoder stochasticity verdict",
    "Physical-batch gradient verdict",
    "Short-curve variants actually run",
    "Common initialization SHA",
    "Scheduler/full-trajectory verification",
    "Epoch-60/90 results",
    "Full-run authorization decision",
    "Full candidate result if run",
    "Query initialization diagnostics",
    "Query conflict diagnostics",
    "Mask-attention recall diagnostics",
    "Strong-local variants actually run",
    "Best local checkpoint",
    "Exact claims supported",
    "Exact claims NOT supported",
    "Remaining risks",
    "All artifact hashes",
    "Test/lint results",
    "External files not in Git and how to locate them by hash",
    "Exact reproduction commands",
    "Recommended next stage",
    "GitHub branch URL / PR URL",
)

REQUIRED_ARTIFACTS = (
    "START_STATE.md",
    "START_STATE.json",
    "EXTERNAL_EVIDENCE.md",
    "LITERATURE_EVIDENCE.md",
    "CODE_AUDIT.md",
    "REPRODUCTION_GAP_CONTRACT.md",
    "FINAL_REPORT.md",
    "HANDOFF.md",
    "audit/UPSTREAM_LOCAL_DIFF.md",
    "audit/upstream_local_diff.json",
    "audit/LOSS_SEMANTICS.md",
    "audit/DATA_SEMANTICS.md",
    "audit/RUNTIME_SEMANTICS.md",
    "audit/filter255_inventory.csv",
    "audit/ddp_sampler_rank_trace.csv",
    "audit/ddp_sampler_summary.json",
    "audit/encoder_stochasticity.csv",
    "audit/encoder_stochasticity_summary.json",
    "audit/physical_batch_gradients.csv",
    "audit/physical_batch_summary.json",
    "initialization/COMMON_INITIALIZATION.json",
    "short_curves/VARIANT_CONTRACT.md",
    "short_curves/variant_manifest.json",
    "short_curves/learning_curves.csv",
    "short_curves/official_like_epoch60.csv",
    "short_curves/official_like_epoch90.csv",
    "short_curves/rootcause_per_seed.csv",
    "short_curves/rootcause_summary.csv",
    "short_curves/ROOTCAUSE_SHORT_DECISION.md",
    "full_candidate/FULL_TRAINING_REPORT.md",
    "full_candidate/selected_checkpoint_manifest.json",
    "full_candidate/official_like_per_seed.csv",
    "full_candidate/official_like_summary.csv",
    "full_candidate/ROOT_CAUSE_FULL_VERDICT.md",
    "decoder_diagnostics/query_initialization.csv",
    "decoder_diagnostics/query_conflicts.csv",
    "decoder_diagnostics/attention_mask_recall.csv",
    "decoder_diagnostics/superpoint_features.csv",
    "decoder_diagnostics/DECODER_DIAGNOSTICS.md",
    "strong_local/variant_manifest.json",
    "strong_local/learning_curves.csv",
    "strong_local/official_like_per_seed.csv",
    "strong_local/STRONG_LOCAL_VERDICT.md",
)

FULL_CANDIDATE_ARTIFACTS = frozenset(
    path for path in REQUIRED_ARTIFACTS if path.startswith("full_candidate/")
)
PRIVATE_PATTERNS = (
    (re.compile(r"(?:^|[\s`\"'=])/(?:home|mnt)/"), "private path"),
    (re.compile(r"\bww@[\w.-]+\b"), "private username or host"),
    (re.compile(r"\b[\w.-]+@(?:192\.168\.|10\.|172\.(?:1[6-9]|2\d|3[01])\d*\.)"), "private host"),
    (re.compile(r"\bhf_[A-Za-z0-9]{10,}\b"), "Hugging Face token"),
    (re.compile(r"\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\s*="), "token assignment"),
    (re.compile(r"BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY"), "private key"),
)
FORBIDDEN_SUFFIXES = {".ckpt", ".pth", ".pt", ".tar", ".gz", ".zip"}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HANDOFF_HEADING_RE = re.compile(r"^## ([1-9]|[12][0-9]|30)\. (.+)$", re.MULTILINE)


class FinalArtifactError(RuntimeError):
    """Raised when the final evidence package is incomplete or mutable."""


def _identity(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode):
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(block)
                digest.update(block)
        after = path.lstat()
    except OSError as error:
        raise FinalArtifactError("artifact is unavailable or not regular") from error
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or size != after.st_size:
        raise FinalArtifactError("artifact changed while hashing")
    return {"bytes": size, "sha256": digest.hexdigest()}


def _files(root: Path) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise FinalArtifactError("artifact root is unavailable")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            raise FinalArtifactError(f"artifact is not a regular file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise FinalArtifactError(f"large binary is forbidden in Git: {relative}")
        files[relative] = path
    return files


def _validate_repository(repository: Mapping[str, object]) -> None:
    expected_refs = {
        "branch": BRANCH,
        "head_reference": f"refs/heads/{BRANCH}",
        "remote_reference": f"refs/remotes/origin/{BRANCH}",
    }
    if any(repository.get(key) != value for key, value in expected_refs.items()):
        raise FinalArtifactError("repository binding differs")
    if any(
        not isinstance(repository.get(field), str)
        or COMMIT_RE.fullmatch(str(repository[field])) is None
        for field in ("start_commit", "evidence_commit")
    ):
        raise FinalArtifactError("repository commit is invalid")


def _validate_external_files(records: object) -> list[Mapping[str, object]]:
    if not isinstance(records, list) or not records:
        raise FinalArtifactError("external file inventory is empty")
    required = {
        "logical_name",
        "external_reference",
        "sha256",
        "bytes",
        "creating_commit",
        "config_sha256",
        "selected_epoch",
        "selected_step",
    }
    seen: set[str] = set()
    validated = []
    for value in records:
        if not isinstance(value, Mapping) or not required.issubset(value):
            raise FinalArtifactError("external file record is incomplete")
        reference = value["external_reference"]
        if (
            not isinstance(value["logical_name"], str)
            or not value["logical_name"]
            or not isinstance(reference, str)
            or not reference.startswith("external:")
            or reference in seen
            or not isinstance(value["sha256"], str)
            or SHA256_RE.fullmatch(value["sha256"]) is None
            or not isinstance(value["bytes"], int)
            or value["bytes"] <= 0
            or not isinstance(value["creating_commit"], str)
            or COMMIT_RE.fullmatch(value["creating_commit"]) is None
            or not isinstance(value["config_sha256"], str)
            or SHA256_RE.fullmatch(value["config_sha256"]) is None
            or not isinstance(value["selected_epoch"], int)
            or value["selected_epoch"] < 0
            or not isinstance(value["selected_step"], int)
            or value["selected_step"] < 0
        ):
            raise FinalArtifactError("external file record is invalid")
        seen.add(reference)
        validated.append(value)
    return validated


def _validate_full_candidate(files: Mapping[str, Path]) -> str:
    present = FULL_CANDIDATE_ARTIFACTS & files.keys()
    status_path = files.get("full_candidate/STATUS.json")
    if present == FULL_CANDIDATE_ARTIFACTS and status_path is None:
        return "completed"
    if present or status_path is None:
        raise FinalArtifactError("full-candidate artifacts are incomplete")
    try:
        payload = json.loads(status_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalArtifactError("full-candidate status is invalid") from error
    if not isinstance(payload, dict):
        raise FinalArtifactError("full-candidate status is invalid")
    expected_hash = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if (
        payload.get("status") != "gate_skipped"
        or not isinstance(payload.get("reason"), str)
        or not payload["reason"]
        or not isinstance(payload.get("upstream_gate"), str)
        or not payload["upstream_gate"]
        or not isinstance(expected_hash, str)
        or canonical_sha256(unsigned) != expected_hash
    ):
        raise FinalArtifactError("full-candidate gate skip is invalid")
    return "gate_skipped"


def _validate_required(files: Mapping[str, Path]) -> str:
    missing = set(REQUIRED_ARTIFACTS) - files.keys()
    full_status = _validate_full_candidate(files)
    if full_status == "gate_skipped":
        missing -= FULL_CANDIDATE_ARTIFACTS
    if missing:
        raise FinalArtifactError(
            "required artifacts are missing: " + ", ".join(sorted(missing))
        )
    return full_status


def _validate_handoff(path: Path) -> None:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise FinalArtifactError("HANDOFF is not portable ASCII") from error
    sections = [
        (int(index), title) for index, title in HANDOFF_HEADING_RE.findall(text)
    ]
    expected = list(enumerate(HANDOFF_SECTION_TITLES, start=1))
    if sections != expected:
        raise FinalArtifactError("HANDOFF sections differ")


def _principal_outcome(path: Path) -> str:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise FinalArtifactError("FINAL_REPORT is not portable ASCII") from error
    outcomes = re.findall(r"^Principal outcome: `?(TLRC-(?:GREEN|YELLOW|RED))`?$", text, re.MULTILINE)
    if len(outcomes) != 1:
        raise FinalArtifactError("FINAL_REPORT principal outcome differs")
    return outcomes[0]


def _validate_privacy(files: Mapping[str, Path]) -> None:
    for relative, path in files.items():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise FinalArtifactError(
                f"artifact is not UTF-8 text: {relative}"
            ) from error
        for pattern, description in PRIVATE_PATTERNS:
            if pattern.search(text):
                raise FinalArtifactError(f"{description} found in {relative}")


def build_final_manifest(
    *,
    artifact_root: Path,
    repository: Mapping[str, object],
    external_files: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Hash the complete portable tree, excluding the manifest itself."""

    _validate_repository(repository)
    _validate_external_files(list(external_files))
    files = _files(artifact_root)
    files.pop(MANIFEST_NAME, None)
    _validate_required(files)
    _validate_handoff(files["HANDOFF.md"])
    principal_outcome = _principal_outcome(files["FINAL_REPORT.md"])
    _validate_privacy(files)
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "experiment": EXPERIMENT,
        "principal_outcome": principal_outcome,
        "repository": dict(repository),
        "artifacts": {
            relative: _identity(path) for relative, path in sorted(files.items())
        },
        "external_files": [dict(record) for record in external_files],
    }
    payload["content_sha256"] = canonical_sha256(payload)
    return payload


def verify_final_artifacts(artifact_root: Path) -> dict[str, object]:
    """Verify completeness, privacy, and every identity in FINAL_MANIFEST."""

    files = _files(artifact_root)
    manifest_path = files.get(MANIFEST_NAME)
    if manifest_path is None:
        raise FinalArtifactError("FINAL_MANIFEST.json is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalArtifactError("FINAL_MANIFEST.json is invalid") from error
    if not isinstance(manifest, dict):
        raise FinalArtifactError("FINAL_MANIFEST.json is invalid")
    expected_hash = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if (
        manifest.get("status") != "pass"
        or manifest.get("experiment") != EXPERIMENT
        or not isinstance(expected_hash, str)
        or canonical_sha256(unsigned) != expected_hash
    ):
        raise FinalArtifactError("final manifest content hash differs")
    repository = manifest.get("repository")
    if not isinstance(repository, Mapping):
        raise FinalArtifactError("repository binding differs")
    _validate_repository(repository)
    external_files = _validate_external_files(manifest.get("external_files"))

    files_without_manifest = dict(files)
    files_without_manifest.pop(MANIFEST_NAME)
    full_status = _validate_required(files_without_manifest)
    _validate_handoff(files_without_manifest["HANDOFF.md"])
    principal_outcome = _principal_outcome(files_without_manifest["FINAL_REPORT.md"])
    if manifest.get("principal_outcome") != principal_outcome:
        raise FinalArtifactError("manifest principal outcome differs")
    _validate_privacy(files)

    identities = manifest.get("artifacts")
    if not isinstance(identities, Mapping) or set(identities) != set(
        files_without_manifest
    ):
        raise FinalArtifactError("manifest artifact inventory differs")
    for relative, path in files_without_manifest.items():
        if identities[relative] != _identity(path):
            raise FinalArtifactError(f"artifact identity differs: {relative}")
    return {
        "status": "pass",
        "principal_outcome": principal_outcome,
        "full_candidate_status": full_status,
        "artifact_count": len(files_without_manifest),
        "external_file_count": len(external_files),
        "handoff_section_count": len(HANDOFF_SECTION_TITLES),
        "manifest_content_sha256": expected_hash,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    arguments = parser.parse_args(argv)
    result = verify_final_artifacts(arguments.artifact_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
