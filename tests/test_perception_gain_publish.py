from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.perception_gain_publish import (
    PublicationError,
    build_publication_receipt,
    choose_release_tag,
    validate_release_assets,
)


def test_release_tag_uses_smallest_unused_suffix_without_moving_existing_tag() -> None:
    assert choose_release_tag(set()) == "persist4d-perception-gain-v1"
    assert (
        choose_release_tag({"persist4d-perception-gain-v1"})
        == "persist4d-perception-gain-v1-r2"
    )
    assert (
        choose_release_tag(
            {"persist4d-perception-gain-v1", "persist4d-perception-gain-v1-r2"}
        )
        == "persist4d-perception-gain-v1-r3"
    )


def test_release_asset_validation_uses_exact_bytes_and_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"weights")
    plan = [
        {
            "role": "selected_parent",
            "planned_name": "selected-parent.ckpt",
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    ]
    validated = validate_release_assets(plan)
    assert validated[0]["path"] == checkpoint

    plan[0]["bytes"] = 1
    try:
        validate_release_assets(plan)
    except PublicationError as error:
        assert "bytes" in str(error)
    else:
        raise AssertionError("mismatched release bytes were accepted")


def test_publication_receipt_distinguishes_verified_release_from_code_only() -> None:
    code_only = build_publication_receipt(
        publication_commit="a" * 40,
        remote_branch_sha="a" * 40,
        tag="persist4d-perception-gain-v1",
        remote_tag_sha="a" * 40,
        branch_url="https://github.com/o/r/tree/branch",
        commit_url="https://github.com/o/r/commit/" + "a" * 40,
        handoff_url="https://github.com/o/r/blob/" + "a" * 40 + "/HANDOFF.md",
        report_url="https://github.com/o/r/blob/" + "a" * 40 + "/FINAL_REPORT.md",
        release=None,
        assets=(),
        verified_remote_files=("HANDOFF.md", "confirmation/CONFIRMATION_SUMMARY.json"),
    )
    assert code_only["publication_status"] == "CODE_ONLY"
    assert code_only["release_url"] is None

    released = build_publication_receipt(
        publication_commit="a" * 40,
        remote_branch_sha="a" * 40,
        tag="persist4d-perception-gain-v1",
        remote_tag_sha="a" * 40,
        branch_url="https://github.com/o/r/tree/branch",
        commit_url="https://github.com/o/r/commit/" + "a" * 40,
        handoff_url="https://github.com/o/r/blob/" + "a" * 40 + "/HANDOFF.md",
        report_url="https://github.com/o/r/blob/" + "a" * 40 + "/FINAL_REPORT.md",
        release={"url": "https://github.com/o/r/releases/tag/t", "id": 1},
        assets=({"name": "model.ckpt", "bytes": 7, "sha256": "b" * 64},),
        verified_remote_files=("HANDOFF.md", "confirmation/CONFIRMATION_SUMMARY.json"),
    )
    assert released["publication_status"] == "VERIFIED"
    assert released["verification_level"] == "REMOTE_METADATA"
