from __future__ import annotations

import hashlib

import pytest


def _finalizer():
    from scripts import finalize_r1_downstream_validation

    return finalize_r1_downstream_validation


def test_final_statuses_preserve_recovery_and_historical_boundaries() -> None:
    finalizer = _finalizer()

    statuses = finalizer.derive_final_statuses(
        recovery={"status": "RECOVERY_SUPPORTED"},
        historical={
            "status": "HISTORICAL_REPLAY_UNAVAILABLE",
            "frozen_summary_comparison": {"status": "available"},
        },
        publication_status="LOCAL_ONLY",
    )

    assert statuses == {
        "execution_status": "COMPLETE",
        "candidate_semantics": "PASS",
        "recovery_evidence": "SUPPORTED",
        "temporal_task_result": "B2_AND_FULLHISTORY_REPORTED_WITH_REDUCER_SENSITIVITY",
        "resource_evidence": "SUPPORTED_IN_PROFILE",
        "old_new_comparability": "HISTORICAL_REFERENCE_ONLY",
        "publication_status": "LOCAL_ONLY",
    }


def test_final_statuses_do_not_authorize_failed_recovery() -> None:
    finalizer = _finalizer()

    statuses = finalizer.derive_final_statuses(
        recovery={"status": "RECOVERY_NOT_SUPPORTED"},
        historical={
            "status": "HISTORICAL_REPLAY_AVAILABLE",
            "frozen_summary_comparison": {"status": "available"},
        },
        publication_status="PUSH_VERIFIED",
    )

    assert statuses["recovery_evidence"] == "NOT_SUPPORTED"
    assert statuses["old_new_comparability"] == "REPLAY_COMPATIBLE"
    assert statuses["publication_status"] == "PUSH_VERIFIED"


def test_final_statuses_reject_unknown_or_incomplete_evidence() -> None:
    finalizer = _finalizer()

    with pytest.raises(finalizer.R1FinalizationError, match="recovery status"):
        finalizer.derive_final_statuses(
            recovery={"status": "unexpected"},
            historical={"status": "HISTORICAL_REPLAY_UNAVAILABLE"},
            publication_status="LOCAL_ONLY",
        )
    with pytest.raises(finalizer.R1FinalizationError, match="summary comparison"):
        finalizer.derive_final_statuses(
            recovery={"status": "RECOVERY_SUPPORTED"},
            historical={
                "status": "HISTORICAL_REPLAY_UNAVAILABLE",
                "frozen_summary_comparison": {"status": "unavailable"},
            },
            publication_status="LOCAL_ONLY",
        )
    with pytest.raises(finalizer.R1FinalizationError, match="publication status"):
        finalizer.derive_final_statuses(
            recovery={"status": "RECOVERY_SUPPORTED"},
            historical={
                "status": "HISTORICAL_REPLAY_UNAVAILABLE",
                "frozen_summary_comparison": {"status": "available"},
            },
            publication_status="unknown",
        )


def test_final_manifest_closes_every_upstream_and_output_byte() -> None:
    finalizer = _finalizer()
    upstream = {
        "metrics/task_aggregate.csv": b"task\n",
        "profile/summary.csv": b"profile\n",
    }
    outputs = {"FINAL_REPORT.md": b"report\n", "HANDOFF.md": b"handoff\n"}
    statuses = {
        "execution_status": "COMPLETE",
        "candidate_semantics": "PASS",
        "recovery_evidence": "SUPPORTED",
        "temporal_task_result": "B2_AND_FULLHISTORY_REPORTED_WITH_REDUCER_SENSITIVITY",
        "resource_evidence": "SUPPORTED_IN_PROFILE",
        "old_new_comparability": "HISTORICAL_REFERENCE_ONLY",
        "publication_status": "LOCAL_ONLY",
    }

    manifest = finalizer.build_final_manifest(
        source_commit="a" * 40,
        statuses=statuses,
        checkpoint_sha256="b" * 64,
        protocol_sha256="c" * 64,
        upstream_payloads=upstream,
        output_payloads=outputs,
    )

    assert manifest["statuses"] == statuses
    assert manifest["upstream_sha256"] == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in upstream.items()
    }
    assert manifest["output_sha256"] == {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in outputs.items()
    }
    assert len(manifest["content_sha256"]) == 64


def test_smoke_contract_requires_six_clusters_and_all_129_cache_pairs() -> None:
    finalizer = _finalizer()
    smoke = {
        "status": "pass",
        "pair_count": 6,
        "repeated_input_count": 2,
        "pairs": [
            {
                "reference_scene_id": f"cluster-{index}",
                "repeat_count": 2 if index < 2 else 1,
                "t2_observation_parity": "pass",
                "t2_candidate_parity": "pass",
            }
            for index in range(6)
        ],
        "query_feature_export_parity": {
            "status": "pass",
            "legacy_predictions_unchanged": True,
        },
        "fresh_state_gt_isolation": {"status": "pass"},
        "t2_cache_parity": {
            "status": "pass",
            "unit_count": 129,
            "pass_count": 129,
            "fail_count": 0,
        },
        "local_current_invariance": {"status": "pass"},
    }

    assert finalizer.validate_smoke_contract(smoke)["status"] == "pass"
    smoke["pair_count"] = 3
    with pytest.raises(finalizer.R1FinalizationError, match="six clusters"):
        finalizer.validate_smoke_contract(smoke)


def test_handoff_contract_requires_14_numbered_sections_and_exact_status_keys() -> None:
    finalizer = _finalizer()
    status_lines = "\n".join(
        f"{key}: value" for key in finalizer.REQUIRED_STATUS_KEYS
    )
    body = "\n".join(
        ["# Persist4D R1 Downstream Validation Handoff", status_lines]
        + [
            f"## {index}. {title}"
            for index, title in enumerate(
                finalizer.REQUIRED_HANDOFF_SECTIONS, start=1
            )
        ]
    ).encode()

    assert finalizer.validate_handoff_contract(body)["section_count"] == 14
    with pytest.raises(finalizer.R1FinalizationError, match="handoff section"):
        finalizer.validate_handoff_contract(body.replace(b"## 14.", b"## 15."))
