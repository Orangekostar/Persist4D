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
    )

    assert statuses == {
        "execution": "EXECUTION_COMPLETE",
        "candidate": "R1_FROZEN_CHECKPOINT_EVALUATED",
        "recovery": "RECOVERY_SUPPORTED",
        "task": "TASK_EVIDENCE_COMPLETE",
        "resource": "RESOURCE_PROFILE_COMPLETE",
        "old_new": "FROZEN_SUMMARY_COMPARISON_ONLY",
        "publication": "RECOVERY_CLAIM_AUTHORIZED",
    }


def test_final_statuses_do_not_authorize_failed_recovery() -> None:
    finalizer = _finalizer()

    statuses = finalizer.derive_final_statuses(
        recovery={"status": "RECOVERY_NOT_SUPPORTED"},
        historical={
            "status": "HISTORICAL_REPLAY_AVAILABLE",
            "frozen_summary_comparison": {"status": "available"},
        },
    )

    assert statuses["old_new"] == "HISTORICAL_REPLAY_AVAILABLE"
    assert statuses["publication"] == "RECOVERY_CLAIM_NOT_AUTHORIZED"


def test_final_statuses_reject_unknown_or_incomplete_evidence() -> None:
    finalizer = _finalizer()

    with pytest.raises(finalizer.R1FinalizationError, match="recovery status"):
        finalizer.derive_final_statuses(
            recovery={"status": "unexpected"},
            historical={"status": "HISTORICAL_REPLAY_UNAVAILABLE"},
        )
    with pytest.raises(finalizer.R1FinalizationError, match="summary comparison"):
        finalizer.derive_final_statuses(
            recovery={"status": "RECOVERY_SUPPORTED"},
            historical={
                "status": "HISTORICAL_REPLAY_UNAVAILABLE",
                "frozen_summary_comparison": {"status": "unavailable"},
            },
        )


def test_final_manifest_closes_every_upstream_and_output_byte() -> None:
    finalizer = _finalizer()
    upstream = {
        "metrics/task_aggregate.csv": b"task\n",
        "profile/summary.csv": b"profile\n",
    }
    outputs = {"FINAL_REPORT.md": b"report\n", "HANDOFF.md": b"handoff\n"}
    statuses = {
        "execution": "EXECUTION_COMPLETE",
        "candidate": "R1_FROZEN_CHECKPOINT_EVALUATED",
        "recovery": "RECOVERY_SUPPORTED",
        "task": "TASK_EVIDENCE_COMPLETE",
        "resource": "RESOURCE_PROFILE_COMPLETE",
        "old_new": "FROZEN_SUMMARY_COMPARISON_ONLY",
        "publication": "RECOVERY_CLAIM_AUTHORIZED",
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
