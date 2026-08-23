from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts import reviewer_closure_protocol as protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/reviewer_closure/protocol.yaml"
MODULE_PATH = REPO_ROOT / "scripts/reviewer_closure_protocol.py"
SYSTEM_MANIFEST_PATH = (
    REPO_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
)
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"


def _api(name: str):
    value = getattr(protocol, name, None)
    assert value is not None, f"missing reviewer-closure protocol API: {name}"
    return value


def test_reviewer_closure_protocol_files_exist() -> None:
    assert CONFIG_PATH.is_file()
    assert MODULE_PATH.is_file()


def test_config_freezes_lineage_tree_sidecar_and_statistics() -> None:
    config = _api("load_reviewer_closure_config")(CONFIG_PATH)

    assert config["source_commits"] == {
        "completed_branch_head": "b2414e3b2e89a990ee42a368caf6784eb27f8f01",
        "system_report_source": "575acc12fbd63f38fc3c16578914b25c2fed8584",
        "official_rescene": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
    }
    assert config["baseline"] == {
        "artifact_tree": "398fe87e1d40d67e61399fd893f02dc5f5f6b7ad",
        "classification": "SYSTEM_PARETO_LOCK",
    }
    assert config["sidecar"]["schema_version"] == "full-history-observations-v2"
    assert config["sidecar"]["horizons"] == [2, 3, 4, 5]
    assert config["statistics"] == {
        "cluster_unit": "reference_scene_id",
        "cluster_count": 6,
        "bootstrap_replicates": 10000,
        "seed": 45,
        "confidence_level": 0.95,
    }


def test_binding_matches_frozen_files_and_exact_commit_lineage() -> None:
    binding = _api("validate_reviewer_closure_binding")(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )

    assert binding["status"] == "pass"
    assert binding["completed_branch_head"] == (
        "b2414e3b2e89a990ee42a368caf6784eb27f8f01"
    )
    assert binding["system_report_source"] == (
        "575acc12fbd63f38fc3c16578914b25c2fed8584"
    )
    assert binding["completed_branch_parent"] == binding["system_report_source"]
    assert binding["system_comparison_artifact_tree"] == (
        "398fe87e1d40d67e61399fd893f02dc5f5f6b7ad"
    )
    assert binding["checkpoint_sha256"] == (
        "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
    )
    assert binding["system_manifest_content_sha256"] == (
        "8eb702d000b5d25c11249d06763faefcdb28c61b516774e9e296d77e57fd78ac"
    )
    assert binding["system_comparison_worktree_status"] == "clean"


def test_config_rejects_a_different_existing_official_rescene_commit(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["source_commits"]["official_rescene"] = config["source_commits"][
        "system_report_source"
    ]
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    error = _api("ReviewerClosureProtocolError")
    with pytest.raises(error, match="source_commits"):
        _api("load_reviewer_closure_config")(path)


def test_binding_rejects_tampered_source_hash(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["sources"]["system_report"]["sha256"] = "0" * 64
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    error = _api("ReviewerClosureProtocolError")
    with pytest.raises(error, match="system_report.*SHA256"):
        _api("validate_reviewer_closure_binding")(
            path,
            repo_root=REPO_ROOT,
            checkpoint_path=CHECKPOINT_PATH,
        )


def _manifest() -> dict[str, object]:
    binding = _api("validate_reviewer_closure_binding")(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    return _api("build_reviewer_closure_manifest")(
        CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
        binding=binding,
    )


def test_manifest_is_exact_protocol_copy_with_paper_facing_methods() -> None:
    manifest = _manifest()
    validated = _api("validate_reviewer_closure_manifest")(
        manifest,
        config_path=CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
    )
    source = json.loads(SYSTEM_MANIFEST_PATH.read_text(encoding="utf-8"))

    assert validated["comparison_prefix_count"] == 43 * 3 * 4
    assert validated["masters"] == source["masters"]
    assert len(validated["masters"]) == 43
    assert len({row["reference_scene_id"] for row in validated["masters"]}) == 6
    assert validated["protocol"]["order_variants"] == [
        "canonical",
        "reverse",
        "sha256_seed45",
    ]
    names = [row["name"] for row in validated["phase_i_methods"]]
    assert names == [
        "ReScene4D Full-History (Frozen T2 Checkpoint)",
        "Pairwise Feature Association",
        "Pairwise Feature-Class Association",
        "EMA Temporal Association",
        "Persist4D Persistent Entity State",
        "Full-History + Persistent-State Diagnostic",
    ]
    assert not any(name in {"B1", "B2", "B3"} for name in names)


def test_sidecar_keys_cover_only_exact_o2_to_o5_prefixes() -> None:
    keys = _api("full_history_observation_keys")(_manifest())

    assert len(keys) == 43 * 3 * 4
    assert len({json.dumps(row, sort_keys=True) for row in keys}) == len(keys)
    assert {row["horizon"] for row in keys} == {2, 3, 4, 5}
    assert {row["order_id"] for row in keys} == {
        "canonical",
        "reverse",
        "sha256_seed45",
    }
    for row in keys:
        assert row["horizon"] == len(row["scan_indices"])
        assert row["horizon"] == len(row["history_scan_ids"])


def test_manifest_rejects_master_prefix_mutation() -> None:
    tampered = copy.deepcopy(_manifest())
    tampered["masters"][0]["orders"]["canonical"]["prefixes"]["4"][
        "scan_ids"
    ][-1] = "scene9999_99"

    error = _api("ReviewerClosureProtocolError")
    with pytest.raises(error, match="digest|masters|prefix"):
        _api("validate_reviewer_closure_manifest")(
            tampered,
            config_path=CONFIG_PATH,
            system_manifest_path=SYSTEM_MANIFEST_PATH,
        )


def test_manifest_rejects_semantic_prefix_mutation_with_recomputed_digest() -> None:
    tampered = copy.deepcopy(_manifest())
    tampered["masters"][0]["orders"]["canonical"]["prefixes"]["4"][
        "scan_ids"
    ][-1] = "scene9999_99"
    tampered.pop("content_sha256")
    canonical = (
        json.dumps(
            tampered,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    tampered["content_sha256"] = hashlib.sha256(canonical).hexdigest()

    error = _api("ReviewerClosureProtocolError")
    with pytest.raises(error, match="masters|prefix"):
        _api("validate_reviewer_closure_manifest")(
            tampered,
            config_path=CONFIG_PATH,
            system_manifest_path=SYSTEM_MANIFEST_PATH,
        )


def test_sidecar_entries_are_ignored_but_manifest_is_trackable() -> None:
    entry = "artifacts/reviewer_closure/full_history_observations_v2/entries/a.pt"
    manifest = "artifacts/reviewer_closure/full_history_observations_v2/manifest.json"

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", entry], cwd=REPO_ROOT, check=False
    )
    trackable = subprocess.run(
        ["git", "check-ignore", "-q", manifest], cwd=REPO_ROOT, check=False
    )
    assert ignored.returncode == 0
    assert trackable.returncode == 1
