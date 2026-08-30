from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.rescene_rootcause_preflight import (
    RootCauseContractError,
    build_external_file_manifest,
    canonical_sha256,
    portable_reference,
    validate_exact_bindings,
    validate_portable_payload,
)


def test_canonical_sha256_is_order_independent_and_ascii() -> None:
    left = {"nested": {"beta": 2, "alpha": "value"}, "items": [3, 1]}
    right = {"items": [3, 1], "nested": {"alpha": "value", "beta": 2}}

    assert canonical_sha256(left) == canonical_sha256(right)
    assert len(canonical_sha256(left)) == 64
    with pytest.raises((TypeError, ValueError)):
        canonical_sha256({"invalid": float("nan")})


def test_portable_reference_requires_namespace_and_sha256() -> None:
    digest = "a" * 64

    assert portable_reference("checkpoint/rootcause_common", digest) == (
        f"external:checkpoint/rootcause_common/{digest}"
    )
    with pytest.raises(RootCauseContractError, match="namespace"):
        portable_reference("../private", digest)
    with pytest.raises(RootCauseContractError, match="SHA-256"):
        portable_reference("checkpoint/rootcause_common", "short")


def test_external_manifest_records_content_without_private_path(tmp_path: Path) -> None:
    checkpoint = tmp_path / "private" / "state.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"rootcause-state")

    manifest = build_external_file_manifest(
        checkpoint,
        logical_name="rootcause_common_initial_state",
        reference="external:checkpoint/rootcause_common/" + "b" * 64,
        creating_commit="1" * 40,
        config_sha256="2" * 64,
        upstream_checkpoint_sha256="3" * 64,
        selected_epoch=90,
        selected_step=5940,
    )

    assert manifest["logical_name"] == "rootcause_common_initial_state"
    assert manifest["bytes"] == len(b"rootcause-state")
    assert len(manifest["sha256"]) == 64
    assert str(tmp_path) not in json.dumps(manifest, sort_keys=True)
    validate_portable_payload(manifest)


def test_external_manifest_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"state")
    link = tmp_path / "link.pt"
    link.symlink_to(source)

    with pytest.raises(RootCauseContractError, match="regular"):
        build_external_file_manifest(
            link,
            logical_name="state",
            reference="external:checkpoint/state/" + "a" * 64,
            creating_commit="1" * 40,
            config_sha256="2" * 64,
            upstream_checkpoint_sha256="3" * 64,
        )


def test_exact_bindings_fail_closed_on_missing_extra_or_drift() -> None:
    expected = {
        "source_commit": "1" * 40,
        "data": {"rio": "2" * 64, "scannet": "3" * 64},
        "runtime": {"torch": "2.6.0+cu126", "lightning": "2.6.5"},
    }

    assert validate_exact_bindings(expected, expected) == expected
    for changed in (
        {"source_commit": "1" * 40, "data": expected["data"]},
        {**expected, "unexpected": True},
        {**expected, "source_commit": "4" * 40},
    ):
        with pytest.raises(RootCauseContractError, match="bindings"):
            validate_exact_bindings(changed, expected)


@pytest.mark.parametrize(
    "value",
    [
        "/private/checkpoint.pt",
        "file:///private/checkpoint.pt",
        "external:checkpoint/192.0.2.1/state",
    ],
)
def test_portable_payload_rejects_private_or_network_locations(value: str) -> None:
    with pytest.raises(RootCauseContractError, match="portable"):
        validate_portable_payload({"reference": value})
