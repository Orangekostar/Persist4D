"""Verify the repository-resident final-evidence artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


class EvidenceVerificationError(RuntimeError):
    """Raised when a final-evidence binding does not verify."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
