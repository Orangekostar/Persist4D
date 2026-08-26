from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from utils.sonata_weight_provenance import (
    OFFICIAL_SONATA_WEIGHT_SPEC,
    SonataWeightProvenanceError,
    SonataWeightSpec,
    build_sonata_weight_manifest,
    validate_official_sonata_remote_metadata,
    validate_sonata_weight_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fixture_spec(path: Path) -> SonataWeightSpec:
    payload = path.read_bytes()
    return SonataWeightSpec(
        repo_id="facebook/sonata",
        revision="a" * 40,
        filename="sonata.pth",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        license="CC-BY-NC-4.0",
    )


def _official_remote_metadata() -> dict[str, object]:
    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    return {
        "id": spec.repo_id,
        "sha": spec.revision,
        "private": False,
        "gated": False,
        "siblings": [
            {
                "rfilename": spec.filename,
                "size": spec.bytes,
                "lfs": {"sha256": spec.sha256, "size": spec.bytes},
            }
        ],
    }


def test_official_spec_is_immutable_and_provenance_complete() -> None:
    assert OFFICIAL_SONATA_WEIGHT_SPEC.repo_id == "facebook/sonata"
    assert OFFICIAL_SONATA_WEIGHT_SPEC.revision == (
        "df99897472c09f91ba9288da0a034aacffc0b010"
    )
    assert OFFICIAL_SONATA_WEIGHT_SPEC.filename == "sonata.pth"
    assert OFFICIAL_SONATA_WEIGHT_SPEC.sha256 == (
        "c5ced5acdae30d1c469713398073a866e25e6e414e23feed5dc025373657ac50"
    )
    assert OFFICIAL_SONATA_WEIGHT_SPEC.bytes == 434_008_287
    assert OFFICIAL_SONATA_WEIGHT_SPEC.license == "CC-BY-NC-4.0"


def test_acquisition_script_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/acquire_sonata_weight.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output-path" in result.stdout


def test_official_remote_metadata_matches_frozen_lfs_object() -> None:
    observed = validate_official_sonata_remote_metadata(_official_remote_metadata())

    assert observed == {
        "repo_id": OFFICIAL_SONATA_WEIGHT_SPEC.repo_id,
        "revision": OFFICIAL_SONATA_WEIGHT_SPEC.revision,
        "filename": OFFICIAL_SONATA_WEIGHT_SPEC.filename,
        "sha256": OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
        "bytes": OFFICIAL_SONATA_WEIGHT_SPEC.bytes,
        "gated": False,
        "private": False,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha", "b" * 40, "revision"),
        ("private", True, "private"),
        ("gated", "manual", "gated"),
    ],
)
def test_remote_metadata_identity_or_access_mismatch_fails(
    field: str,
    value: object,
    match: str,
) -> None:
    metadata = _official_remote_metadata()
    metadata[field] = value

    with pytest.raises(SonataWeightProvenanceError, match=match):
        validate_official_sonata_remote_metadata(metadata)


def test_remote_metadata_lfs_hash_mismatch_fails() -> None:
    metadata = _official_remote_metadata()
    metadata["siblings"][0]["lfs"]["sha256"] = "0" * 64

    with pytest.raises(SonataWeightProvenanceError, match="LFS SHA256"):
        validate_official_sonata_remote_metadata(metadata)


def test_manifest_binds_regular_file_hash_bytes_and_remote_revision(
    tmp_path: Path,
) -> None:
    weight = tmp_path / "sonata.pth"
    weight.write_bytes(b"verified-sonata-fixture")
    spec = _fixture_spec(weight)

    manifest = build_sonata_weight_manifest(
        weight,
        spec=spec,
        acquired_at="2026-08-26T08:00:00Z",
        download_source="https://huggingface.co/facebook/sonata",
    )

    assert manifest["status"] == "pass"
    assert manifest["gate"] == "weight-provenance-pass"
    assert manifest["source"] == {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.filename,
        "download_source": "https://huggingface.co/facebook/sonata",
        "license": spec.license,
    }
    assert manifest["local_file"] == {
        "reference": f"external:sonata_weight/{spec.sha256}",
        "sha256": spec.sha256,
        "bytes": spec.bytes,
        "regular_file": True,
        "symlink": False,
    }
    validate_sonata_weight_manifest(manifest, weight, spec=spec)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("repo_id", "someone/sonata", "repo_id"),
        ("revision", "main", "immutable revision"),
        ("filename", "renamed.pth", "filename"),
    ],
)
def test_formal_manifest_rejects_nonofficial_source_identity(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    weight = tmp_path / "sonata.pth"
    weight.write_bytes(b"not-the-real-weight")
    spec = replace(OFFICIAL_SONATA_WEIGHT_SPEC, **{field: value})

    with pytest.raises(SonataWeightProvenanceError, match=match):
        build_sonata_weight_manifest(
            weight,
            spec=spec,
            acquired_at="2026-08-26T08:00:00Z",
            download_source="https://huggingface.co/facebook/sonata",
            require_official=True,
        )


def test_anonymous_same_name_file_is_rejected(tmp_path: Path) -> None:
    weight = tmp_path / "sonata.pth"
    weight.write_bytes(b"anonymous")

    with pytest.raises(SonataWeightProvenanceError, match="SHA256"):
        build_sonata_weight_manifest(
            weight,
            spec=OFFICIAL_SONATA_WEIGHT_SPEC,
            acquired_at="2026-08-26T08:00:00Z",
            download_source="https://huggingface.co/facebook/sonata",
        )


def test_symlink_weight_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.pth"
    target.write_bytes(b"fixture")
    weight = tmp_path / "sonata.pth"
    weight.symlink_to(target)

    with pytest.raises(SonataWeightProvenanceError, match="regular file"):
        build_sonata_weight_manifest(
            weight,
            spec=_fixture_spec(target),
            acquired_at="2026-08-26T08:00:00Z",
            download_source="https://huggingface.co/facebook/sonata",
        )


def test_stale_manifest_is_rejected_after_file_changes(tmp_path: Path) -> None:
    weight = tmp_path / "sonata.pth"
    weight.write_bytes(b"first")
    spec = _fixture_spec(weight)
    manifest = build_sonata_weight_manifest(
        weight,
        spec=spec,
        acquired_at="2026-08-26T08:00:00Z",
        download_source="https://huggingface.co/facebook/sonata",
    )
    weight.write_bytes(b"second")

    with pytest.raises(SonataWeightProvenanceError, match="mismatch"):
        validate_sonata_weight_manifest(manifest, weight, spec=spec)
