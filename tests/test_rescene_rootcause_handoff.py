from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_rescene_rootcause_handoff import (
    HANDOFF_SECTION_TITLES,
    REQUIRED_ARTIFACTS,
    FinalArtifactError,
    build_final_manifest,
    verify_final_artifacts,
)


def _handoff() -> str:
    lines = ["# ReScene Task-Learning Root-Cause Handoff", ""]
    for index, title in enumerate(HANDOFF_SECTION_TITLES, start=1):
        lines.extend([f"## {index}. {title}", "", "Verified evidence.", ""])
    return "\n".join(lines)


def _external_file() -> dict[str, object]:
    return {
        "logical_name": "R0 epoch-90 checkpoint",
        "external_reference": "external:checkpoint/" + "a" * 64,
        "sha256": "a" * 64,
        "bytes": 754_813_736,
        "creating_commit": "b" * 40,
        "config_sha256": "c" * 64,
        "selected_epoch": 90,
        "selected_step": 5_940,
    }


def _artifact_tree(tmp_path: Path, *, full_skipped: bool = False) -> Path:
    root = tmp_path / "artifacts" / "rescene_task_learning_root_cause_v1"
    for relative in REQUIRED_ARTIFACTS:
        if full_skipped and relative.startswith("full_candidate/"):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "HANDOFF.md":
            path.write_text(_handoff(), encoding="ascii")
        elif path.name == "FINAL_REPORT.md":
            path.write_text(
                "# Final Report\n\nPrincipal outcome: `TLRC-YELLOW`\n",
                encoding="ascii",
            )
        elif path.suffix == ".json":
            path.write_text('{"status": "pass"}\n', encoding="ascii")
        elif path.suffix == ".csv":
            path.write_text("status\npass\n", encoding="ascii")
        else:
            path.write_text("# Evidence\n\nVerified.\n", encoding="ascii")
    if full_skipped:
        status = {
            "schema_version": 1,
            "status": "gate_skipped",
            "reason": "no short-curve candidate passed every gate",
            "upstream_gate": "ROOTCAUSE_SHORT_DECISION",
        }
        from utils.rescene_rootcause_preflight import canonical_sha256

        status["content_sha256"] = canonical_sha256(status)
        path = root / "full_candidate" / "STATUS.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status), encoding="ascii")
    return root


def _write_manifest(root: Path) -> None:
    manifest = build_final_manifest(
        artifact_root=root,
        repository={
            "branch": "research/persist4d-rescene-task-learning-root-cause-v1",
            "start_commit": "1" * 40,
            "evidence_commit": "2" * 40,
            "head_reference": (
                "refs/heads/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
            "remote_reference": (
                "refs/remotes/origin/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
        },
        external_files=[_external_file()],
    )
    (root / "FINAL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def test_final_artifact_verifier_hashes_tree_and_accepts_complete_handoff(
    tmp_path: Path,
) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)

    result = verify_final_artifacts(root)

    assert result["status"] == "pass"
    assert result["principal_outcome"] == "TLRC-YELLOW"
    assert result["handoff_section_count"] == 30
    assert result["external_file_count"] == 1
    assert result["artifact_count"] == len(
        [path for path in root.rglob("*") if path.is_file()]
    ) - 1


def test_final_artifact_verifier_accepts_signed_full_gate_skip(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path, full_skipped=True)
    _write_manifest(root)

    result = verify_final_artifacts(root)

    assert result["full_candidate_status"] == "gate_skipped"


def test_final_manifest_accepts_pretraining_external_state(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    external = _external_file()
    external.update(
        {
            "logical_name": "common initialization state",
            "selected_epoch": 0,
            "selected_step": 0,
        }
    )

    manifest = build_final_manifest(
        artifact_root=root,
        repository={
            "branch": "research/persist4d-rescene-task-learning-root-cause-v1",
            "start_commit": "1" * 40,
            "evidence_commit": "2" * 40,
            "head_reference": (
                "refs/heads/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
            "remote_reference": (
                "refs/remotes/origin/research/"
                "persist4d-rescene-task-learning-root-cause-v1"
            ),
        },
        external_files=[external],
    )

    assert manifest["external_files"][0]["selected_step"] == 0


def test_final_artifact_verifier_rejects_private_paths(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    (root / "CODE_AUDIT.md").write_text(
        "checkpoint: /home/researcher/run/model.ckpt\n", encoding="ascii"
    )

    with pytest.raises(FinalArtifactError, match="private path"):
        verify_final_artifacts(root)


def test_final_artifact_verifier_rejects_changed_artifact(tmp_path: Path) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    with (root / "CODE_AUDIT.md").open("a", encoding="ascii") as handle:
        handle.write("changed after manifest\n")

    with pytest.raises(FinalArtifactError, match="identity differs"):
        verify_final_artifacts(root)


def test_final_artifact_verifier_rejects_missing_handoff_section(
    tmp_path: Path,
) -> None:
    root = _artifact_tree(tmp_path)
    _write_manifest(root)
    handoff = root / "HANDOFF.md"
    handoff.write_text(
        handoff.read_text(encoding="ascii").replace(
            "## 17. Query initialization diagnostics\n", ""
        ),
        encoding="ascii",
    )
    with pytest.raises(FinalArtifactError, match="HANDOFF sections differ"):
        verify_final_artifacts(root)
