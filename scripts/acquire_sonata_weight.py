"""Acquire, freeze, and audit the official Sonata encoder weight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sonata_weight_provenance import (
    OFFICIAL_SONATA_WEIGHT_SPEC,
    SonataLoadKeyError,
    SonataWeightProvenanceError,
    audit_sonata_checkpoint_load,
    build_sonata_weight_manifest,
    validate_official_sonata_remote_metadata,
    validate_sonata_weight_manifest,
)

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "weight"
)
SONATA_CODE_REVISION = "18c09ff8d713494f78a8213792262b910977a65d"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _remote_metadata_url(endpoint: str) -> str:
    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    repo_id = urllib.parse.quote(spec.repo_id, safe="/")
    revision = urllib.parse.quote(spec.revision, safe="")
    return (
        f"{endpoint.rstrip('/')}/api/models/{repo_id}/revision/{revision}"
        "?blobs=true"
    )


def _fetch_remote_metadata(endpoint: str) -> dict[str, Any]:
    request = urllib.request.Request(
        _remote_metadata_url(endpoint),
        headers={"User-Agent": "Persist4D-Sonata-Provenance/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except Exception as error:
        raise SonataWeightProvenanceError(
            "official Sonata remote metadata could not be retrieved"
        ) from error
    if not isinstance(payload, dict):
        raise SonataWeightProvenanceError("remote metadata response is not an object")
    return payload


def _copy_verified_snapshot(source: Path, output: Path) -> None:
    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    source = Path(source).resolve()
    output = Path(output)
    if output.exists() or output.is_symlink():
        existing = build_sonata_weight_manifest(
            output,
            spec=spec,
            acquired_at=_utc_now(),
            download_source="https://huggingface.co/facebook/sonata",
            require_official=True,
        )
        validate_sonata_weight_manifest(
            existing,
            output,
            spec=spec,
            require_official=True,
        )
        return
    if not source.is_file():
        raise SonataWeightProvenanceError("downloaded cache object is not a file")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        hasher = hashlib.sha256()
        copied_bytes = 0
        with source.open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            for block in iter(lambda: source_handle.read(1024 * 1024), b""):
                output_handle.write(block)
                hasher.update(block)
                copied_bytes += len(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if copied_bytes != spec.bytes or hasher.hexdigest() != spec.sha256:
            raise SonataWeightProvenanceError(
                "downloaded cache object differs from the official LFS object"
            )
        temporary.chmod(0o444)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_report(
    path: Path,
    *,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    audit_metadata: dict[str, Any],
) -> None:
    source = manifest["source"]
    local_file = manifest["local_file"]
    lines = [
        "# Sonata Weight Provenance",
        "",
        f"- Gate: `{audit['gate']}`",
        f"- Repository: `{source['repo_id']}`",
        f"- Immutable revision: `{source['revision']}`",
        f"- Filename: `{source['filename']}`",
        f"- SHA-256: `{local_file['sha256']}`",
        f"- Bytes: {local_file['bytes']}",
        f"- License: `{source['license']}`",
        f"- Acquired: `{manifest['acquired_at']}`",
        f"- Local reference: `{local_file['reference']}`",
        f"- Sonata code revision: `{SONATA_CODE_REVISION}`",
        "",
        "## Load-Key Audit",
        "",
        f"- Checkpoint keys: {audit['checkpoint_key_count']}",
        f"- Model keys: {audit['model_key_count']}",
        f"- Loaded keys: {audit['loaded_key_count']}",
        f"- Loaded encoder/embedding keys: {audit['loaded_encoder_key_count']}",
        (
            "- Expected train-from-scratch decoder missing keys: "
            f"{audit['expected_decoder_missing_key_count']}"
        ),
        f"- Unexpected keys: {len(audit['unexpected_keys'])}",
        f"- Resolved `enc_mode`: `{audit_metadata['resolved_config_enc_mode']}`",
        "",
        "The local file is an immutable regular-file snapshot. The official",
        "weight is encoder-only; only `dec.*` parameters are initialized from",
        "scratch. No critical `embedding.*` or `enc.*` parameter is missing.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def acquire_and_audit(
    *,
    output_path: Path,
    artifact_dir: Path,
    cache_dir: Path,
    endpoint: str,
) -> dict[str, Path]:
    remote_payload = _fetch_remote_metadata(endpoint)
    remote = validate_official_sonata_remote_metadata(remote_payload)
    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    cached_path = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        repo_type="model",
        revision=spec.revision,
        cache_dir=str(cache_dir),
        endpoint=endpoint,
    )
    _copy_verified_snapshot(Path(cached_path), Path(output_path))

    acquired_at = _utc_now()
    manifest = build_sonata_weight_manifest(
        output_path,
        spec=spec,
        acquired_at=acquired_at,
        download_source=f"{endpoint.rstrip('/')}/{spec.repo_id}",
        require_official=True,
    )
    manifest["remote_metadata"] = remote
    manifest["implementation"] = {
        "script_ref": "repo:scripts/acquire_sonata_weight.py",
        "script_sha256": _sha256(Path(__file__)),
        "utility_ref": "repo:utils/sonata_weight_provenance.py",
        "utility_sha256": _sha256(
            PROJECT_ROOT / "utils" / "sonata_weight_provenance.py"
        ),
        "sonata_code_ref": "external:github/facebookresearch/sonata",
        "sonata_code_revision": SONATA_CODE_REVISION,
    }
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "sonata_weight_manifest.json"
    _write_json(manifest_path, manifest)

    try:
        audit, audit_metadata = audit_sonata_checkpoint_load(
            output_path,
            expected_sha256=spec.sha256,
        )
    except SonataLoadKeyError as error:
        failure = {
            "schema_version": 1,
            "status": "fail",
            "gate": "SW0-FAIL",
            "weight_sha256": spec.sha256,
            "weight_manifest_sha256": _sha256(manifest_path),
            "error": str(error),
        }
        _write_json(artifact_dir / "sonata_load_key_audit.json", failure)
        raise
    audit["weight_manifest_sha256"] = _sha256(manifest_path)
    audit["metadata"] = audit_metadata
    audit["sonata_code_revision"] = SONATA_CODE_REVISION
    audit_path = artifact_dir / "sonata_load_key_audit.json"
    _write_json(audit_path, audit)

    report_path = artifact_dir / "SONATA_WEIGHT_PROVENANCE.md"
    _write_report(
        report_path,
        manifest=manifest,
        audit=audit,
        audit_metadata=audit_metadata,
    )
    return {
        "weight": Path(output_path),
        "manifest": manifest_path,
        "load_key_audit": audit_path,
        "report": report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = acquire_and_audit(
        output_path=args.output_path,
        artifact_dir=args.artifact_dir,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
