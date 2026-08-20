from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.p6b_artifacts import (
    P6B_ARTIFACT_SCHEMA_VERSION,
    finalize_p6b_artifact,
    publish_p6b_artifact,
    render_p6b_bundle,
    validate_p6b_artifact,
)


def _row(stage: str = "assignment", *, config_id: str = "p6b-a") -> dict[str, object]:
    return {
        "config_id": config_id,
        "config_json": '{"active_threshold":0.5}',
        "stage": stage,
        "T": "T4",
        "identity_switches": 8,
        "wrong_reactivations": 2,
        "false_births": 3,
        "reactivation_accuracy": 0.8,
        "reactivation_recall": 0.4,
        "accepted_valid_observations": 90,
        "total_valid_observations": 100,
        "strict_online_tmap": 0.2,
        "strict_online_trec": 0.3,
        "eligible": True,
        "eligibility_reasons": "",
    }


def _root() -> dict[str, object]:
    config = {"active_threshold": 0.5}
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    sweep_rows = []
    for stage in (
        "assignment",
        "reactivation",
        "class_compatibility",
        "consolidation",
        "birth_gate",
        "joint_neighbors",
    ):
        sweep_rows.append(_row(stage, config_id=f"p6b-{stage}"))
    return {
        "schema_version": P6B_ARTIFACT_SCHEMA_VERSION,
        "status": "pass",
        "decision": "P6B_GO",
        "source_commit": "a" * 40,
        "source_tree_contract": {"status": "pass", "source_commit": "a" * 40},
        "provenance": {
            "checkpoint": {"ref": "repo:checkpoints/model.ckpt", "sha256": "b" * 64},
            "p5": {"ref": "repo:artifacts/P5/persist4d_mvp_eval.json", "sha256": "c" * 64},
            "p6a": {"ref": "repo:artifacts/P6A/p6a_eval.json", "sha256": "d" * 64},
            "p6a_protocol_manifest": {"ref": "repo:artifacts/P6A/protocol_b_manifest.json", "sha256": "e" * 64},
            "p6a_cache_manifest": {"ref": "external:p6a_cache/cache_manifest.json", "sha256": "f" * 64},
        },
        "split_manifest": {
            "schema_version": 1,
            "seed": 45,
            "hash_algorithm": "sha256",
            "hash_namespace": "p6b|45|reference",
            "tuning_reference_scene_ids": ["r0", "r1", "r2", "r3"],
            "heldout_reference_scene_ids": ["r4", "r5"],
            "tuning_master_sequence_ids": [f"m{i}" for i in range(32)],
            "heldout_master_sequence_ids": [f"h{i}" for i in range(11)],
            "assignments": [],
            "sha256": "1" * 64,
        },
        "selection": {
            "config_id": "p6b-selected",
            "config_sha256": config_sha256,
            "config": config,
            "ranking_key": [8.0, 2.0, 3.0, -0.4, -0.25, "{}"],
            "tuning_reference_scene_ids": ["r0", "r1", "r2", "r3"],
        },
        "sweep_rows": sweep_rows,
        "final_results": [
            {"method": method, "T": horizon, "t_mAP": 0.2, "t_REC": 0.3, "identity_switches": 8, "identity_switch_rate": 0.1, "reactivation_accuracy": 0.8, "reactivation_recall": 0.4, "false_births": 3}
            for method in ("B4", "P6B")
            for horizon in ("T2", "T3", "T4", "T5")
        ],
        "per_sequence_results": [
            {"method": "P6B", "reference_scene_id": "r4", "master_sequence_id": "h0", "order_id": "canonical", "T": "T5", "identity_switches": 1, "transition_opportunities": 10, "identity_switch_rate": 0.1, "wrong_reactivations": 1, "false_births": 2, "reactivation_accuracy": 0.8, "reactivation_recall": 0.4}
        ],
        "failure_analysis": [
            {"method": "P6B", "T": "T5", "failure_category": "F3", "count": 2}
        ],
        "gate_results": {
            f"G6B-{index}": {"passed": True, "evidence": "fixture"}
            for index in range(1, 6)
        },
        "claims_supported": ["P6-B improves held-out identity continuity."],
        "claims_not_supported": ["P6-B is not a SOTA claim."],
        "next_action": "Freeze P6-B and plan P7 separately.",
        "artifact_manifest": [],
    }


def test_finalize_validates_exact_schema_and_manifest_binding() -> None:
    root = finalize_p6b_artifact(_root())
    validate_p6b_artifact(root)
    files = render_p6b_bundle(root)

    assert set(files) >= {
        "p6b_eval.json",
        "P6B_GO_NOGO_REPORT.md",
        "artifact_manifest.json",
        "selected_config.yaml",
        "final_results.csv",
        "figures/identity_comparison.svg",
    }
    records = {record["path"]: record for record in root["artifact_manifest"]}
    for path, record in records.items():
        assert record["bytes"] == len(files[path])
        assert record["sha256"] == hashlib.sha256(files[path]).hexdigest()
    assert json.loads(files["p6b_eval.json"])["decision"] == "P6B_GO"


def test_finalize_accepts_json_object_key_reordering() -> None:
    root = json.loads(json.dumps(_root(), sort_keys=True))

    finalized = finalize_p6b_artifact(root)

    validate_p6b_artifact(finalized)


def test_report_has_exact_eleven_sections_and_terminal_decision() -> None:
    report = render_p6b_bundle(finalize_p6b_artifact(_root()))[
        "P6B_GO_NOGO_REPORT.md"
    ].decode()

    assert report.count("\n## ") == 11
    assert report.rstrip().endswith("P6B_GO")
    assert "1. What was changed" in report
    assert "11. Exact next action" in report


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda root: root.update(extra=True), "keys"),
        (lambda root: root.update(decision="GO"), "decision"),
        (lambda root: root["provenance"]["checkpoint"].update(ref="/" + "home/private/model.ckpt"), "portable"),
        (lambda root: root["split_manifest"].update(heldout_reference_scene_ids=["r3", "r5"]), "overlap"),
        (lambda root: root.update(claims_supported=["SOTA"]), "claim"),
    ),
)
def test_artifact_rejects_schema_privacy_leakage_and_unsupported_claims(
    mutation, match: str
) -> None:
    root = _root()
    mutation(root)
    with pytest.raises(ValueError, match=match):
        finalize_p6b_artifact(root)


def test_publish_is_atomic_and_refuses_existing_or_symlink_root(tmp_path: Path) -> None:
    root = finalize_p6b_artifact(_root())
    output = tmp_path / "P6B"
    publish_p6b_artifact(output, root)
    assert (output / "p6b_eval.json").is_file()

    with pytest.raises(FileExistsError):
        publish_p6b_artifact(output, root)
    link = tmp_path / "linked"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(FileExistsError):
        publish_p6b_artifact(link, root)
