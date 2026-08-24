"""Verify the repository-resident final-evidence artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


class EvidenceVerificationError(RuntimeError):
    """Raised when a final-evidence binding does not verify."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceVerificationError(f"invalid {field}")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceVerificationError(f"invalid {field}")
    return value


def validate_full_history_contract(
    cache_manifest: Mapping[str, object],
    raw_result: Mapping[str, object],
    cache_manifest_sha256: str,
) -> None:
    """Fail closed unless the committed Full-History run is self-consistent."""
    if cache_manifest.get("status") != "pass":
        raise EvidenceVerificationError("Full-History cache status changed")
    provenance = _mapping(cache_manifest.get("provenance"), "cache provenance")
    if provenance.get("history_strategy") != "full_history":
        raise EvidenceVerificationError("Full-History strategy changed")
    entries = _sequence(cache_manifest.get("entries"), "cache entries")
    if cache_manifest.get("entry_count") != len(entries):
        raise EvidenceVerificationError("Full-History entry count changed")

    filenames: set[str] = set()
    digests: set[str] = set()
    for entry_value in entries:
        entry = _mapping(entry_value, "cache entry")
        key = _mapping(entry.get("key"), "cache key")
        filename = entry.get("filename")
        digest = entry.get("sha256")
        if not isinstance(filename, str) or filename in filenames:
            raise EvidenceVerificationError("duplicate Full-History cache filename")
        if not isinstance(digest, str) or len(digest) != 64 or digest in digests:
            raise EvidenceVerificationError("invalid Full-History cache digest")
        filenames.add(filename)
        digests.add(digest)
        if key.get("history_strategy") != "full_history":
            raise EvidenceVerificationError("Full-History cache key changed")
        stage_index = key.get("stage_index")
        capture_ids = _sequence(
            key.get("local_capture_ids"), "expanding capture history"
        )
        if (
            isinstance(stage_index, bool)
            or not isinstance(stage_index, int)
            or stage_index < 0
            or len(capture_ids) != stage_index + 1
            or not capture_ids
            or capture_ids[-1] != key.get("target_capture_id")
        ):
            raise EvidenceVerificationError("invalid expanding capture history")

    raw_provenance = _mapping(raw_result.get("provenance"), "raw provenance")
    if raw_provenance.get("cache_manifest_sha256") != cache_manifest_sha256:
        raise EvidenceVerificationError("Full-History cache manifest binding changed")
    for field in (
        "checkpoint_sha256",
        "dataset_content_sha256",
        "evaluator_sha256",
    ):
        if raw_provenance.get(field) != provenance.get(field):
            raise EvidenceVerificationError("Full-History runtime provenance changed")
    gate = _mapping(raw_result.get("external_gate"), "Full-History external gate")
    if gate.get("classification") != "EXTERNAL_INCONCLUSIVE":
        raise EvidenceVerificationError("Full-History classification changed")


def verify(root: Path, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen":
        raise EvidenceVerificationError("manifest status must be frozen")
    checked = 0
    for entry in manifest.get("repository_files", []):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise EvidenceVerificationError(f"unsafe manifest path: {relative}")
        path = root / relative
        if not path.is_file():
            raise EvidenceVerificationError(f"missing artifact: {relative}")
        if path.stat().st_size != entry["size_bytes"]:
            raise EvidenceVerificationError(f"size mismatch: {relative}")
        if _sha256(path) != entry["sha256"]:
            raise EvidenceVerificationError(f"SHA256 mismatch: {relative}")
        checked += 1

    gate_path = root / "artifacts/final_evidence/external_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("classification") != "EXTERNAL_INCONCLUSIVE":
        raise EvidenceVerificationError("external gate classification changed")
    final_report = (
        root / "artifacts/final_evidence/FINAL_PAPER_EVIDENCE_REPORT.md"
    ).read_text(encoding="utf-8")
    if "`PAPER_READY_INTERNAL_ONLY`" not in final_report:
        raise EvidenceVerificationError("final paper classification is missing")

    artifact_root = root / "artifacts/final_evidence"
    spot_check = (artifact_root / "RESCAN_EVENT_SPOT_CHECK.md").read_text(
        encoding="utf-8"
    )
    if "`EVENT_SPOT_CHECK_PASS`" not in spot_check:
        raise EvidenceVerificationError("ReScan event spot check is missing")

    effects_path = artifact_root / "external/rescan_per_scene_effects.csv"
    with effects_path.open(encoding="utf-8", newline="") as handle:
        effect_rows = list(csv.DictReader(handle))
    if len(effect_rows) != 104 or len({row["scene_id"] for row in effect_rows}) != 13:
        raise EvidenceVerificationError("ReScan per-scene effects changed")

    full_history_manifest_path = (
        artifact_root / "external/rescan_full_history_cache_manifest.json"
    )
    full_history_manifest = json.loads(
        full_history_manifest_path.read_text(encoding="utf-8")
    )
    full_history_raw = json.loads(
        (artifact_root / "external/rescan_full_history_raw.json").read_text(
            encoding="utf-8"
        )
    )
    validate_full_history_contract(
        full_history_manifest,
        full_history_raw,
        _sha256(full_history_manifest_path),
    )
    if (
        full_history_manifest.get("scene_count") != 13
        or full_history_manifest.get("entry_count") != 45
    ):
        raise EvidenceVerificationError("formal Full-History population changed")

    external_inputs = _mapping(manifest.get("external_inputs"), "external inputs")
    full_history_binding = _mapping(
        external_inputs.get("rescan_full_history_observation_cache"),
        "Full-History cache binding",
    )
    if full_history_binding.get("manifest_sha256") != _sha256(
        full_history_manifest_path
    ):
        raise EvidenceVerificationError("final manifest Full-History binding changed")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/final_evidence/final_evidence_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    checked = verify(root, manifest_path)
    print(f"FINAL_EVIDENCE_OK files={checked}")


if __name__ == "__main__":
    main()
