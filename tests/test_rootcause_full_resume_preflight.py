from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.prepare_rescene_rootcause_full_resume import (
    archive_conflicting_last_checkpoint,
    validate_exact_resume_evidence,
)
from utils.rescene_rootcause_evaluation import (
    RootCauseEvaluationError,
    build_checkpoint_manifest,
)
from utils.rescene_rootcause_preflight import canonical_sha256


def _authorization() -> dict[str, object]:
    payload = {
        "status": "authorized",
        "source_commit": "1" * 40,
        "selected_variants": ["R0", "R1"],
        "variants": {
            "R0": {"config_sha256": "2" * 64},
            "R1": {"config_sha256": "3" * 64},
        },
        "initialization": {
            "common_state": {"sha256": "4" * 64},
            "pretrained": {"sha256": "5" * 64},
        },
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def _candidate(authorization: dict[str, object]) -> dict[str, object]:
    payload = {
        "variant": "R1",
        "source_commit": authorization["source_commit"],
        "variant_authorization_sha256": authorization["authorization_sha256"],
        "config_sha256": authorization["variants"]["R1"]["config_sha256"],
        "common_initialization_sha256": "4" * 64,
        "pretrained_sha256": "5" * 64,
    }
    payload["candidate_id"] = canonical_sha256(payload)
    return payload


def _evidence(tmp_path):
    authorization = _authorization()
    candidate = _candidate(authorization)
    run = tmp_path / "R1"
    run.mkdir()
    exact = run / "epoch=090.ckpt"
    exact.write_bytes(b"exact-epoch-90")
    identity = {
        "bytes": exact.stat().st_size,
        "sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
    }
    manifest = build_checkpoint_manifest(
        variant="R1",
        completed_epoch=90,
        authorization=authorization,
        candidate=candidate,
        file_identity=identity,
        checkpoint_facts={
            "selected_epoch": 90,
            "selected_step": 5940,
            "training_config_sha256": authorization["variants"]["R1"]["config_sha256"],
        },
    )
    decision = {
        "schema_version": 1,
        "status": "pass",
        "selected_variant": "R1",
        "full_training_authorized": True,
        "full_training_status": "authorized",
        "variant_authorization_sha256": authorization["authorization_sha256"],
    }
    decision["content_sha256"] = canonical_sha256(decision)
    paths = {}
    for name, value in (
        ("authorization", authorization),
        ("candidate", candidate),
        ("manifest", manifest),
        ("decision", decision),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths, exact


def _migrated_evidence(tmp_path):
    original = _authorization()
    adapted = copy.deepcopy(original)
    adapted["runtime"] = {
        "gpu_count": 2,
        "gpu_models": ["NVIDIA A40", "NVIDIA A40"],
        "torch": "2.6.0+cu126",
    }
    original["runtime"] = {
        "gpu_count": 3,
        "gpu_models": ["NVIDIA A40", "NVIDIA A40", "NVIDIA A40"],
        "torch": "2.6.0+cu126",
    }
    original["authorization_sha256"] = canonical_sha256(
        {key: value for key, value in original.items() if key != "authorization_sha256"}
    )
    adapted["authorization_sha256"] = canonical_sha256(
        {key: value for key, value in adapted.items() if key != "authorization_sha256"}
    )
    candidate = _candidate(adapted)
    run = tmp_path / "R1"
    run.mkdir()
    exact = run / "epoch=090.ckpt"
    exact.write_bytes(b"migrated-exact-epoch-90")
    identity = {
        "bytes": exact.stat().st_size,
        "sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
    }
    manifest = build_checkpoint_manifest(
        variant="R1",
        completed_epoch=90,
        authorization=adapted,
        candidate=candidate,
        file_identity=identity,
        checkpoint_facts={
            "selected_epoch": 90,
            "selected_step": 5940,
            "training_config_sha256": adapted["variants"]["R1"]["config_sha256"],
        },
    )
    decision = {
        "schema_version": 1,
        "status": "pass",
        "selected_variant": "R1",
        "full_training_authorized": True,
        "full_training_status": "authorized",
        "variant_authorization_sha256": original["authorization_sha256"],
    }
    decision["content_sha256"] = canonical_sha256(decision)
    migration = {
        "schema_version": 1,
        "status": "pass",
        "scientific_config_changed": False,
        "changed_fields": ["runtime.gpu_count", "runtime.gpu_models"],
        "original_root_authorization_sha256": original["authorization_sha256"],
        "adapted_root_authorization_sha256": adapted["authorization_sha256"],
        "original_runtime": original["runtime"],
        "adapted_runtime": adapted["runtime"],
    }
    paths = {}
    for name, value in (
        ("authorization", adapted),
        ("decision_authorization", original),
        ("candidate", candidate),
        ("manifest", manifest),
        ("decision", decision),
        ("migration", migration),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    return paths, exact


def test_exact_epoch90_resume_preflight_binds_selected_checkpoint(tmp_path) -> None:
    paths, exact = _evidence(tmp_path)

    plan = validate_exact_resume_evidence(
        variant="R1",
        authorization_path=paths["authorization"],
        candidate_path=paths["candidate"],
        decision_path=paths["decision"],
        checkpoint_manifest_path=paths["manifest"],
        exact_checkpoint_path=exact,
        selected_resume_checkpoint=exact,
    )

    assert plan["status"] == "pass"
    assert plan["variant"] == "R1"
    assert plan["completed_epoch"] == 90
    assert plan["selected_step"] == 5940
    assert plan["expected_state_dict_entries"] == 798
    assert (
        plan["exact_checkpoint_sha256"]
        == hashlib.sha256(exact.read_bytes()).hexdigest()
    )
    assert len(plan["content_sha256"]) == 64


def test_exact_resume_accepts_runtime_only_migration_of_authorization(tmp_path) -> None:
    paths, exact = _migrated_evidence(tmp_path)

    plan = validate_exact_resume_evidence(
        variant="R1",
        authorization_path=paths["authorization"],
        candidate_path=paths["candidate"],
        decision_path=paths["decision"],
        checkpoint_manifest_path=paths["manifest"],
        exact_checkpoint_path=exact,
        selected_resume_checkpoint=exact,
        decision_authorization_path=paths["decision_authorization"],
        runtime_migration_provenance_path=paths["migration"],
    )

    assert plan["status"] == "pass"
    assert (
        plan["variant_authorization_sha256"]
        == json.loads(paths["authorization"].read_text())["authorization_sha256"]
    )
    assert (
        plan["decision_authorization_sha256"]
        == json.loads(paths["decision_authorization"].read_text())[
            "authorization_sha256"
        ]
    )
    assert plan["runtime_migration"]["changed_fields"] == [
        "runtime.gpu_count",
        "runtime.gpu_models",
    ]


def test_exact_resume_rejects_incomplete_runtime_migration_bridge(tmp_path) -> None:
    paths, exact = _migrated_evidence(tmp_path)

    with pytest.raises(RootCauseEvaluationError, match="bridge is incomplete"):
        validate_exact_resume_evidence(
            variant="R1",
            authorization_path=paths["authorization"],
            candidate_path=paths["candidate"],
            decision_path=paths["decision"],
            checkpoint_manifest_path=paths["manifest"],
            exact_checkpoint_path=exact,
            selected_resume_checkpoint=exact,
            decision_authorization_path=paths["decision_authorization"],
        )


def test_exact_resume_rejects_scientific_change_in_runtime_migration(tmp_path) -> None:
    paths, exact = _migrated_evidence(tmp_path)
    authorization = json.loads(paths["authorization"].read_text())
    authorization["variants"]["R1"]["config_sha256"] = "9" * 64
    authorization["authorization_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
    )
    paths["authorization"].write_text(json.dumps(authorization), encoding="utf-8")
    candidate = _candidate(authorization)
    paths["candidate"].write_text(json.dumps(candidate), encoding="utf-8")
    manifest = build_checkpoint_manifest(
        variant="R1",
        completed_epoch=90,
        authorization=authorization,
        candidate=candidate,
        file_identity={
            "bytes": exact.stat().st_size,
            "sha256": hashlib.sha256(exact.read_bytes()).hexdigest(),
        },
        checkpoint_facts={
            "selected_epoch": 90,
            "selected_step": 5940,
            "training_config_sha256": "9" * 64,
        },
    )
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    provenance = json.loads(paths["migration"].read_text())
    provenance["adapted_root_authorization_sha256"] = authorization[
        "authorization_sha256"
    ]
    provenance["adapted_runtime"] = authorization["runtime"]
    paths["migration"].write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(
        RootCauseEvaluationError, match="changed more than GPU inventory"
    ):
        validate_exact_resume_evidence(
            variant="R1",
            authorization_path=paths["authorization"],
            candidate_path=paths["candidate"],
            decision_path=paths["decision"],
            checkpoint_manifest_path=paths["manifest"],
            exact_checkpoint_path=exact,
            selected_resume_checkpoint=exact,
            decision_authorization_path=paths["decision_authorization"],
            runtime_migration_provenance_path=paths["migration"],
        )


def test_exact_epoch90_resume_preflight_rejects_last_checkpoint_selection(
    tmp_path,
) -> None:
    paths, exact = _evidence(tmp_path)
    last = exact.parent / "last.ckpt"
    last.write_bytes(b"same-boundary-different-callback-state")

    with pytest.raises(RootCauseEvaluationError, match="exact epoch-90"):
        validate_exact_resume_evidence(
            variant="R1",
            authorization_path=paths["authorization"],
            candidate_path=paths["candidate"],
            decision_path=paths["decision"],
            checkpoint_manifest_path=paths["manifest"],
            exact_checkpoint_path=exact,
            selected_resume_checkpoint=last,
        )


def test_archive_conflict_is_recoverable_and_limited_to_last_checkpoint(
    tmp_path,
) -> None:
    _, exact = _evidence(tmp_path)
    last = exact.parent / "last.ckpt"
    last.write_bytes(b"last-at-epoch-90")
    archive = exact.parent / "pre_full_resume_checkpoints"

    record = archive_conflicting_last_checkpoint(
        exact_checkpoint_path=exact,
        selected_checkpoint_path=last,
        selected_checkpoint_facts={"selected_epoch": 90, "selected_step": 5940},
        archive_directory=archive,
    )

    assert not last.exists()
    assert (archive / "last.ckpt").read_bytes() == b"last-at-epoch-90"
    assert record["recoverable"] is True
    assert record["original_name"] == "last.ckpt"

    unrelated = exact.parent / "metric.ckpt"
    unrelated.write_bytes(b"metric")
    with pytest.raises(RootCauseEvaluationError, match="last-checkpoint"):
        archive_conflicting_last_checkpoint(
            exact_checkpoint_path=exact,
            selected_checkpoint_path=unrelated,
            selected_checkpoint_facts={"selected_epoch": 90, "selected_step": 5940},
            archive_directory=archive,
        )
