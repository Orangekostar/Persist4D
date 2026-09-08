import hashlib
import json
from pathlib import Path

import pytest

from scripts.persist4d_allt_contract import (
    AllTContractError,
    apply_budget_amendment,
    build_reference_split,
    build_s0_documents,
    build_split_manifest,
    canonical_json_sha256,
    require_file_identity,
    validate_role_assignments,
    write_s0_documents,
)


def test_canonical_json_hash_rejects_nonfinite_values() -> None:
    assert canonical_json_sha256({"b": 2, "a": 1}) == hashlib.sha256(
        b'{"a":1,"b":2}'
    ).hexdigest()
    with pytest.raises(AllTContractError, match="finite"):
        canonical_json_sha256({"metric": float("nan")})


def test_file_identity_fails_closed_on_size_or_digest_drift(tmp_path: Path) -> None:
    source = tmp_path / "checkpoint.ckpt"
    source.write_bytes(b"abc")

    identity = require_file_identity(
        source,
        expected_bytes=3,
        expected_sha256=(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ),
        label="R1 checkpoint",
    )
    assert identity == {
        "bytes": 3,
        "sha256": (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        ),
        "status": "pass",
    }

    with pytest.raises(AllTContractError, match="byte size"):
        require_file_identity(
            source,
            expected_bytes=4,
            expected_sha256=identity["sha256"],
            label="R1 checkpoint",
        )
    with pytest.raises(AllTContractError, match="SHA256"):
        require_file_identity(
            source,
            expected_bytes=3,
            expected_sha256="0" * 64,
            label="R1 checkpoint",
        )


def test_reference_split_is_deterministic_and_reference_disjoint() -> None:
    split = build_reference_split(
        ["ref-c", "ref-a", "ref-b"], development_count=1, seed=45
    )

    assert split["development_reference_ids"] == ["ref-b"]
    assert split["adaptation_reference_ids"] == ["ref-a", "ref-c"]
    assert split["assignment_by_reference"] == {
        "ref-a": "adaptation",
        "ref-b": "development",
        "ref-c": "adaptation",
    }
    assert split["hash_namespace"] == "allt-v1:45:"
    validate_role_assignments(
        split,
        protocol_reference_ids=["final-only"],
        processed_test_available=False,
    )


def test_role_validation_rejects_final_or_cross_role_contamination() -> None:
    split = {
        "development_reference_ids": ["same"],
        "adaptation_reference_ids": ["same"],
        "assignment_by_reference": {"same": "development"},
    }
    with pytest.raises(AllTContractError, match="disjoint"):
        validate_role_assignments(
            split,
            protocol_reference_ids=["final"],
            processed_test_available=False,
        )

    clean = build_reference_split(["train-a", "train-b"], development_count=1)
    with pytest.raises(AllTContractError, match="Protocol-B"):
        validate_role_assignments(
            clean,
            protocol_reference_ids=[clean["development_reference_ids"][0]],
            processed_test_available=False,
        )
    with pytest.raises(AllTContractError, match="processed test"):
        validate_role_assignments(
            clean,
            protocol_reference_ids=["final"],
            processed_test_available=True,
        )


def test_budget_can_be_amended_once_before_formal_training() -> None:
    provisional = {
        "status": "provisional_pre_throughput",
        "amendment_count": 0,
        "formal_training_started": False,
        "optimizer_updates": 4000,
        "physical_episode_batch_per_gpu": 1,
        "gradient_accumulation": 4,
        "devices": 2,
        "effective_episode_batch": 8,
    }
    frozen = apply_budget_amendment(
        provisional,
        changes={
            "gradient_accumulation": 8,
            "devices": 1,
            "optimizer_updates": 3000,
        },
        reason="one-GPU throughput preflight",
        throughput={"episodes_per_hour": 72.0, "peak_allocated_bytes": 1234},
    )

    assert frozen["status"] == "frozen_before_formal_training"
    assert frozen["amendment_count"] == 1
    assert frozen["effective_episode_batch"] == 8
    assert frozen["evaluation_updates"] == [0, 750, 1500, 2250, 3000]
    assert frozen["amendment"]["before"]["devices"] == 2
    assert frozen["amendment"]["after"]["devices"] == 1

    with pytest.raises(AllTContractError, match="already frozen"):
        apply_budget_amendment(
            frozen,
            changes={"optimizer_updates": 3000},
            reason="second change",
            throughput={"episodes_per_hour": 80.0},
        )


def test_budget_rejects_changes_after_formal_training() -> None:
    budget = {
        "status": "provisional_pre_throughput",
        "amendment_count": 0,
        "formal_training_started": True,
        "optimizer_updates": 4000,
        "physical_episode_batch_per_gpu": 1,
        "gradient_accumulation": 4,
        "devices": 2,
        "effective_episode_batch": 8,
    }
    with pytest.raises(AllTContractError, match="formal training"):
        apply_budget_amendment(
            budget,
            changes={"devices": 1, "gradient_accumulation": 8},
            reason="too late",
            throughput={"episodes_per_hour": 72.0},
        )


def test_contract_payload_is_portable_json() -> None:
    split = build_reference_split(["r0", "r1"], development_count=1)
    encoded = json.dumps(split, allow_nan=False, sort_keys=True)
    assert "/home/" not in encoded
    assert "192.168." not in encoded


def test_split_manifest_assigns_every_master_from_its_reference() -> None:
    masters = [
        {"sequence_id": "a-0", "reference_scene_id": "ref-a"},
        {"sequence_id": "a-1", "reference_scene_id": "ref-a"},
        {"sequence_id": "b-0", "reference_scene_id": "ref-b"},
        {"sequence_id": "c-0", "reference_scene_id": "ref-c"},
    ]
    manifest = build_split_manifest(
        masters,
        protocol_reference_ids=["final-a"],
        development_count=1,
        expected_reference_count=3,
        expected_master_count=4,
    )

    assert manifest["reference_counts"] == {
        "adaptation": 2,
        "development": 1,
        "protocol_b_final": 1,
    }
    assert manifest["master_counts"] == {"adaptation": 3, "development": 1}
    assert manifest["master_assignments"] == [
        {"role": "adaptation", "sequence_id": "a-0"},
        {"role": "adaptation", "sequence_id": "a-1"},
        {"role": "development", "sequence_id": "b-0"},
        {"role": "adaptation", "sequence_id": "c-0"},
    ]
    assert manifest["independent_generalization"] == "NOT_ESTABLISHED"
    assert manifest["processed_test_available"] is False
    assert manifest["content_sha256"] == canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "content_sha256"}
    )


def test_split_manifest_rejects_population_drift() -> None:
    masters = [{"sequence_id": "a-0", "reference_scene_id": "ref-a"}]
    with pytest.raises(AllTContractError, match="master count"):
        build_split_manifest(
            masters,
            protocol_reference_ids=["final"],
            development_count=1,
            expected_reference_count=2,
            expected_master_count=2,
        )
    with pytest.raises(AllTContractError, match="reference count"):
        build_split_manifest(
            masters,
            protocol_reference_ids=["final"],
            development_count=1,
            expected_reference_count=2,
            expected_master_count=1,
        )


def test_s0_documents_are_complete_portable_and_idempotent(tmp_path: Path) -> None:
    split = build_split_manifest(
        [
            {"sequence_id": "train-a", "reference_scene_id": "ref-a"},
            {"sequence_id": "train-b", "reference_scene_id": "ref-b"},
        ],
        protocol_reference_ids=["final"],
        development_count=1,
        expected_reference_count=2,
        expected_master_count=2,
    )
    identities = {
        "r1_checkpoint": {"bytes": 3, "sha256": "a" * 64, "status": "pass"},
        "protocol_b": {"bytes": 4, "sha256": "b" * 64, "status": "pass"},
        "concerto_pretrained": {
            "bytes": 5,
            "sha256": "c" * 64,
            "status": "pass",
        },
        "rio_metadata": {"bytes": 6, "sha256": "d" * 64, "status": "pass"},
    }
    documents = build_s0_documents(
        baseline_commit="2" * 40,
        identities=identities,
        split_manifest=split,
        cache_counts={"local": 645, "full_history": 645},
    )

    assert set(documents) == {
        "EXPERIMENT_CONTRACT.md",
        "budget_and_schedule.json",
        "run_contract.json",
        "source_map.md",
        "split_manifest.json",
    }
    run_contract = documents["run_contract.json"]
    assert run_contract["formal_candidate_metrics_started"] is False
    assert run_contract["report_horizons"] == [2, 3, 4, 5]
    assert run_contract["required_variants"] == ["C0", "C1", "C2", "FH-adapt"]
    assert documents["budget_and_schedule.json"]["status"] == (
        "provisional_pre_throughput"
    )
    serialized = json.dumps(documents, allow_nan=False, sort_keys=True)
    assert "/home/" not in serialized
    assert "/mnt/" not in serialized
    assert "192.168." not in serialized

    write_s0_documents(tmp_path, documents)
    write_s0_documents(tmp_path, documents)
    assert json.loads((tmp_path / "run_contract.json").read_text()) == run_contract

    changed = dict(documents)
    changed["run_contract.json"] = {**run_contract, "training_seed": 46}
    with pytest.raises(AllTContractError, match="differs"):
        write_s0_documents(tmp_path, changed)
