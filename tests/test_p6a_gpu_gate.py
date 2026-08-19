from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.evaluate_persist4d import _compose_runtime_config, _resolve_checkpoint
from scripts.evaluate_persist4d_p6a import (
    EXPECTED_RESCENE_CHECKPOINT_SHA256,
    _file_sha256,
    _frozen_protocol_bundle,
    build_cache_provenance,
    load_cached_protocol_sequences,
)
from scripts.p6a_artifacts import (
    P5_FROZEN_VALUES,
    render_artifact_bundle,
    validate_root_artifact,
    verify_artifact_manifest,
)
from scripts.p6a_builder import _expected_cache_keys
from scripts.p6a_cache import validate_cache_manifest
from scripts.p6a_efficiency import validate_efficiency_manifest
from scripts.run_p6a_evaluation import _canonical_json_mapping, _runtime_config_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "P6A"
VERIFY_REAL = os.environ.get("P6A_VERIFY_GPU_ARTIFACTS") == "1"
PRIVATE_MARKERS = (
    "/home/",
    "/Users/",
    "/mnt/",
    "\\Users\\",
    "CUDA_VISIBLE_DEVICES",
    "CONCERTO_CHECKPOINT",
)
UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"
)


def _assert_only_protocol_reference_uuids(
    serialized: str, allowed: set[str]
) -> None:
    assert set(UUID_PATTERN.findall(serialized)) == allowed


def test_uuid_privacy_allowlist_rejects_undeclared_identifiers() -> None:
    allowed = {"10b17940-3938-2467-8a7a-958300ba83d3"}
    serialized = " ".join((*allowed, "12345678-1234-1234-1234-123456789abc"))

    with pytest.raises(AssertionError):
        _assert_only_protocol_reference_uuids(serialized, allowed)


def _required_external_path(name: str) -> Path:
    raw = os.environ.get(name)
    assert raw, f"set {name} for the opt-in P6-A artifact gate"
    return Path(raw).expanduser().resolve()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.mark.skipif(
    not VERIFY_REAL,
    reason="set P6A_VERIFY_GPU_ARTIFACTS=1 after the real P6-A run",
)
def test_real_p6a_root_artifact_renders_exactly_from_one_validated_payload() -> None:
    root_path = ARTIFACT_ROOT / "p6a_eval.json"
    assert root_path.is_file() and not root_path.is_symlink()
    artifact = _canonical_json_mapping(root_path, name="P6-A root artifact")

    validate_root_artifact(artifact)
    rendered = render_artifact_bundle(artifact)
    verify_artifact_manifest(artifact, rendered)
    actual_paths = {
        path.relative_to(ARTIFACT_ROOT).as_posix(): path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
    }
    assert set(actual_paths) == set(rendered)
    for relative, expected in rendered.items():
        path = actual_paths[relative]
        assert not path.is_symlink()
        assert path.read_bytes() == expected


@pytest.mark.skipif(
    not VERIFY_REAL,
    reason="set P6A_VERIFY_GPU_ARTIFACTS=1 after the real P6-A run",
)
def test_real_p6a_external_cache_and_efficiency_evidence_are_complete() -> None:
    cache_root = _required_external_path("P6A_CACHE_DIRECTORY")
    metadata = _required_external_path("P6A_3RSCAN_METADATA")
    artifact = _canonical_json_mapping(
        ARTIFACT_ROOT / "p6a_eval.json",
        name="P6-A root artifact",
    )
    protocol, fresh_protocol, p6a_bytes = _frozen_protocol_bundle(
        metadata_path=metadata
    )
    stored_protocol = _canonical_json_mapping(
        cache_root / "protocol_b_manifest.json",
        name="Protocol B manifest",
    )
    cache_manifest_path = cache_root / "cache_manifest.json"
    cache_manifest = _canonical_json_mapping(
        cache_manifest_path,
        name="cache manifest",
    )
    efficiency_path = cache_root / "efficiency_raw_manifest.json"
    efficiency = _canonical_json_mapping(
        efficiency_path,
        name="efficiency manifest",
    )
    assert stored_protocol == fresh_protocol

    config, _memory = _compose_runtime_config()
    runtime_text = _runtime_config_text(config)
    checkpoint = _resolve_checkpoint(
        Path("checkpoints/rescene4d_concerto_t2_repro.ckpt")
    )
    expected_provenance = build_cache_provenance(
        source_commit=artifact["source_commit"],
        checkpoint_path=checkpoint,
        config_documents={
            "p6a": p6a_bytes,
            "runtime": runtime_text.encode("utf-8"),
        },
        protocol_manifest=fresh_protocol,
    )
    validate_cache_manifest(
        cache_manifest,
        expected_keys=_expected_cache_keys(fresh_protocol),
        expected_provenance=expected_provenance,
    )
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_root / "entries",
        manifest_path=cache_manifest_path,
    )
    assert len(sequences) == 129
    assert sum(len(sequence.payloads) for sequence in sequences) == 645

    validate_efficiency_manifest(efficiency)
    assert efficiency["coverage"]["record_count"] == 1161
    assert efficiency["provenance"] == {
        "source_commit": artifact["source_commit"],
        "checkpoint_sha256": EXPECTED_RESCENE_CHECKPOINT_SHA256,
        "config_sha256": expected_provenance["config_sha256"],
        "protocol_sha256": _file_sha256(cache_root / "protocol_b_manifest.json"),
        "cache_manifest_sha256": _file_sha256(cache_manifest_path),
    }
    embedded = artifact["derived_artifacts"]["json"]
    assert json.loads(embedded["protocol_b_manifest.json"]["text"]) == stored_protocol
    assert json.loads(embedded["efficiency_raw_manifest.json"]["text"]) == efficiency


@pytest.mark.skipif(
    not VERIFY_REAL,
    reason="set P6A_VERIFY_GPU_ARTIFACTS=1 after the real P6-A run",
)
def test_real_p6a_evidence_is_source_bound_private_and_keeps_p5_frozen() -> None:
    root_path = ARTIFACT_ROOT / "p6a_eval.json"
    artifact = _canonical_json_mapping(root_path, name="P6-A root artifact")
    source_commit = artifact["source_commit"]
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    changed = set(
        _git("diff", "--name-only", f"{source_commit}..HEAD").splitlines()
    )
    assert changed
    assert all(path.startswith("artifacts/P6A/") for path in changed)
    assert _git("status", "--porcelain", "--untracked-files=all") == ""

    checkpoint = _resolve_checkpoint(
        Path("checkpoints/rescene4d_concerto_t2_repro.ckpt")
    )
    assert _file_sha256(checkpoint) == EXPECTED_RESCENE_CHECKPOINT_SHA256
    p5_json = PROJECT_ROOT / "artifacts/P5/persist4d_mvp_eval.json"
    p5_markdown = PROJECT_ROOT / "artifacts/P5/persist4d_mvp_eval.md"
    assert _file_sha256(p5_json) == P5_FROZEN_VALUES["json_sha256"]
    assert _file_sha256(p5_markdown) == P5_FROZEN_VALUES["markdown_sha256"]

    serialized = b"".join(
        path.read_bytes()
        for path in sorted(ARTIFACT_ROOT.rglob("*"))
        if path.is_file()
    ).decode("ascii")
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", serialized)
    protocol_manifest = _canonical_json_mapping(
        ARTIFACT_ROOT / "protocol_b_manifest.json",
        name="Protocol B manifest",
    )
    reference_ids = {
        master["reference_scene_id"] for master in protocol_manifest["masters"]
    }
    assert len(reference_ids) == 6
    _assert_only_protocol_reference_uuids(serialized, reference_ids)
    assert all(marker not in serialized for marker in PRIVATE_MARKERS)
    assert hashlib.sha256(root_path.read_bytes()).hexdigest()
