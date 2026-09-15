from __future__ import annotations

import hashlib
import json

import pytest

from scripts.publish_task_memory_v2 import (
    CORRECTNESS_CLASSES,
    FORMAL_PROTOCOL_B_ARTIFACTS,
    STATUS_FIELDS,
    VERIFICATION_CHECKS,
    PublicationError,
    build_final_manifest,
    derive_package_statuses,
    render_handoff,
    render_test_report,
    validate_statuses,
    validate_verification,
)


def test_publication_requires_formal_protocol_b_controls_and_cache_manifest() -> None:
    assert set(FORMAL_PROTOCOL_B_ARTIFACTS) == {
        "evaluation/M5/protocol_b/R1-B4-policy/cache_manifest.json",
        "evaluation/M5/protocol_b/R1-B4-policy/policy_comparison.csv",
        "evaluation/M5/protocol_b/baseline/control_observation_manifest.json",
        "evaluation/M5/protocol_b/baseline/long_memory_controls.csv",
    }


def _statuses() -> dict[str, str]:
    return {
        "EXECUTION": "COMPLETE",
        "TMAP_ALL_T_VS_R1": "PASS",
        "TMAP_ALL_T_VS_MATCHED_FH": "FAIL",
        "TASK_METRICS_ALL_T": "FAIL",
        "RETENTION": "IMPROVED",
        "RESOURCE": "TRADEOFF",
        "MECHANISM": "PARTIAL",
        "GENERALIZATION": "BASE_EXPOSED_ONLY",
        "PUBLICATION": "NOT_ATTEMPTED",
    }


def _with_content_hash(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return result


def _verification(results_commit: str = "a" * 40) -> dict[str, object]:
    return _with_content_hash(
        {
            "schema_version": "task-memory-verification-v2",
            "results_commit": results_commit,
            "correctness_classes": {
                name: {"status": "PASS", "evidence": "pytest node selection"}
                for name in CORRECTNESS_CLASSES
            },
            "checks": [
                {
                    "name": name,
                    "command": f"verify-{name}",
                    "exit_code": 0,
                    "observed": f"{name} passed",
                }
                for name in VERIFICATION_CHECKS
            ],
        }
    )


def test_statuses_are_independent_and_closed_vocabularies() -> None:
    assert tuple(validate_statuses(_statuses())) == STATUS_FIELDS
    broken = _statuses()
    broken["RESOURCE"] = "PASS"
    with pytest.raises(PublicationError, match="RESOURCE"):
        validate_statuses(broken)


def test_manifest_excludes_itself_handoff_and_publication_receipt(tmp_path) -> None:
    result = tmp_path / "final/all_t_metrics.csv"
    report = tmp_path / "FINAL_REPORT.md"
    manifest = tmp_path / "FINAL_MANIFEST.json"
    handoff = tmp_path / "HANDOFF.md"
    receipt = tmp_path / "PUBLICATION_RECEIPT.json"
    result.parent.mkdir()
    result.write_bytes(b"metric\n")
    report.write_bytes(b"report\n")
    manifest.write_bytes(b"self\n")
    handoff.write_bytes(b"handoff\n")
    receipt.write_bytes(b"receipt\n")

    value = build_final_manifest(
        tmp_path,
        artifact_paths=(result, report, manifest, handoff, receipt),
        statuses=_statuses(),
        results_commit="a" * 40,
    )

    by_path = {row["path"]: row for row in value["artifacts"]}
    assert set(by_path) == {"final/all_t_metrics.csv", "FINAL_REPORT.md"}
    assert (
        by_path["final/all_t_metrics.csv"]["sha256"]
        == hashlib.sha256(b"metric\n").hexdigest()
    )
    assert "content_sha256" in value


def test_handoff_has_exact_required_fifteen_sections_and_manifest_hash() -> None:
    text = render_handoff(
        statuses=_statuses(),
        results_commit="a" * 40,
        final_manifest_sha256="b" * 64,
    )

    assert sum(line.startswith("## ") for line in text.splitlines()) == 15
    assert "`" + "a" * 40 + "`" in text
    assert "`" + "b" * 64 + "`" in text
    assert "PUSH_VERIFIED" not in text


def test_status_derivation_rejects_malformed_reference_analysis() -> None:
    analysis = _with_content_hash(
        {"status": _statuses(), "reference_analysis": "complete"}
    )
    resource = _with_content_hash({"status": "PASS", "resource_status": "ADVANTAGE"})

    with pytest.raises(PublicationError, match="reference analysis"):
        derive_package_statuses(
            analysis,
            resource,
            publication_status="NOT_ATTEMPTED",
        )


def test_status_derivation_does_not_hide_unrun_experiments() -> None:
    statuses = _statuses()
    statuses["EXECUTION"] = "PARTIAL"
    analysis = _with_content_hash(
        {
            "status": statuses,
            "reference_analysis": {"status": "COMPLETE"},
        }
    )
    resource = _with_content_hash({"status": "PASS", "resource_status": "ADVANTAGE"})

    derived = derive_package_statuses(
        analysis,
        resource,
        publication_status="PUSH_VERIFIED",
    )

    assert derived["EXECUTION"] == "PARTIAL"


def test_verification_and_test_report_require_observed_passing_commands() -> None:
    verification = validate_verification(_verification(), results_commit="a" * 40)
    report = render_test_report(verification)

    assert "verify-real_gpu_gate" in report
    assert "real_gpu_gate passed" in report
    assert all(name in report for name in CORRECTNESS_CLASSES)

    failed = _verification()
    failed["checks"][0]["exit_code"] = 1  # type: ignore[index]
    failed = _with_content_hash(
        {key: value for key, value in failed.items() if key != "content_sha256"}
    )
    with pytest.raises(PublicationError, match="verification check"):
        validate_verification(failed, results_commit="a" * 40)


def test_handoff_accepts_concrete_body_for_every_required_section() -> None:
    details = {index: f"concrete evidence {index}" for index in range(2, 16)}
    text = render_handoff(
        statuses=_statuses(),
        results_commit="a" * 40,
        final_manifest_sha256="b" * 64,
        section_bodies=details,
    )

    assert all(f"concrete evidence {index}" in text for index in range(2, 16))
