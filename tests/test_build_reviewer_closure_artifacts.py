from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import build_reviewer_closure_artifacts as builder

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts/reviewer_closure"
SYSTEM_TREE = "398fe87e1d40d67e61399fd893f02dc5f5f6b7ad"
SOURCE_COMMIT = "a" * 40


def test_real_reviewer_closure_package_builds_complete_final_manifest() -> None:
    manifest = builder.build_final_evidence_manifest(
        artifact_root=ARTIFACT_ROOT,
        repo_root=REPO_ROOT,
        source_commit=SOURCE_COMMIT,
        expected_system_tree=SYSTEM_TREE,
    )

    assert manifest["status"] == "pass"
    assert manifest["final_classification"] == "FINAL_LOCK"
    assert manifest["gates"] == {
        "phase_i": "TRACKER_REJECTED",
        "phase_ii": "HORIZON_ROBUST",
        "phase_iii": "PERCEPTION_CEILING",
        "phase_iv": "not_triggered",
    }
    assert manifest["immutable_system_comparison_tree"] == SYSTEM_TREE
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["content_sha256"] == builder.content_sha256(manifest)
    assert manifest["required_artifact_count"] == len(builder.REQUIRED_ARTIFACTS)
    assert set(builder.REQUIRED_ARTIFACTS).issubset(manifest["artifacts"])
    builder.verify_final_evidence_manifest(
        manifest,
        artifact_root=ARTIFACT_ROOT,
        repo_root=REPO_ROOT,
    )


def test_final_report_requires_exact_sections_and_one_classification() -> None:
    report = (ARTIFACT_ROOT / "FINAL_METHOD_LOCK_REPORT.md").read_text(encoding="utf-8")

    assert builder.validate_final_report(report) == "FINAL_LOCK"

    missing_section = report.replace("## 8. Statistical robustness", "### Statistics")
    with pytest.raises(builder.FinalEvidenceError, match="exactly 12 sections"):
        builder.validate_final_report(missing_section)

    conflicting = report.replace(
        "`FINAL_LOCK`",
        "`FINAL_LOCK` and `FINAL_PARETO_LOCK`",
    )
    with pytest.raises(builder.FinalEvidenceError, match="exactly one classification"):
        builder.validate_final_report(conflicting)


def test_real_t3_smoke_evidence_proves_one_step_and_strict_reload() -> None:
    smoke = json.loads(
        (ARTIFACT_ROOT / "t3_smoke_report.json").read_text(encoding="utf-8")
    )

    builder.validate_t3_smoke_evidence(smoke)

    nonfinite = copy.deepcopy(smoke)
    nonfinite["losses"]["loss_ce"] = float("nan")
    with pytest.raises(builder.FinalEvidenceError, match="finite losses"):
        builder.validate_t3_smoke_evidence(nonfinite)

    no_reload = copy.deepcopy(smoke)
    no_reload["checkpoint_reload"]["strict"] = False
    with pytest.raises(builder.FinalEvidenceError, match="checkpoint reload"):
        builder.validate_t3_smoke_evidence(no_reload)


def test_gate_classifications_reject_unknown_or_failed_states() -> None:
    gate_i = {"status": "pass", "classification": "TRACKER_REJECTED"}
    gate_ii = {"status": "pass", "classification": "HORIZON_ROBUST"}
    phase_iii = {
        "status": "pass",
        "ceiling_classification": "PERCEPTION_CEILING",
    }

    assert builder.validate_gate_classifications(gate_i, gate_ii, phase_iii) == (
        "TRACKER_REJECTED",
        "HORIZON_ROBUST",
        "PERCEPTION_CEILING",
    )

    invalid = copy.deepcopy(gate_ii)
    invalid["classification"] = "UNKNOWN"
    with pytest.raises(builder.FinalEvidenceError, match="Gate-II classification"):
        builder.validate_gate_classifications(gate_i, invalid, phase_iii)

    failed = copy.deepcopy(phase_iii)
    failed["status"] = "failed"
    with pytest.raises(builder.FinalEvidenceError, match="Phase III status"):
        builder.validate_gate_classifications(gate_i, gate_ii, failed)


def test_final_manifest_publication_is_atomic_idempotent_and_refuses_conflict(
    tmp_path: Path,
) -> None:
    manifest = builder.build_final_evidence_manifest(
        artifact_root=ARTIFACT_ROOT,
        repo_root=REPO_ROOT,
        source_commit=SOURCE_COMMIT,
        expected_system_tree=SYSTEM_TREE,
    )
    output = tmp_path / "final_evidence_manifest.json"

    builder.publish_final_evidence_manifest(output, manifest)
    builder.publish_final_evidence_manifest(output, manifest)

    changed = copy.deepcopy(manifest)
    changed["source_commit"] = "b" * 40
    changed["content_sha256"] = builder.content_sha256(changed)
    with pytest.raises(FileExistsError, match="different content"):
        builder.publish_final_evidence_manifest(output, changed)
