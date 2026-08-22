from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.build_system_comparison_artifacts import (
    ALLOWED_CLASSIFICATIONS,
    ArtifactError,
    build_artifact_manifest,
    classify_system_outcome,
    render_final_report,
    validate_final_report,
)


def test_outcome_classifier_emits_exactly_one_preregistered_label() -> None:
    common = {
        "persist4d_tmap": {"T4": 0.10, "T5": 0.09},
        "full_history_tmap": {"T4": 0.10, "T5": 0.08},
        "paired_ci": {"T4": (-0.01, 0.01), "T5": (-0.01, 0.01)},
        "identity_advantage": True,
        "compute_advantage": True,
        "meaningful_advantage": 0.01,
    }
    assert classify_system_outcome(**common)["classification"] == "SYSTEM_LOCK"

    noninferior = {
        **common,
        "persist4d_tmap": {"T4": 0.099, "T5": 0.089},
        "noninferiority_tolerance": 0.002,
    }
    assert (
        classify_system_outcome(**noninferior)["classification"]
        == "SYSTEM_LOCK"
    )

    pareto = {
        **common,
        "full_history_tmap": {"T4": 0.105, "T5": 0.095},
    }
    assert (
        classify_system_outcome(**pareto)["classification"]
        == "SYSTEM_PARETO_LOCK"
    )

    deficit = {
        **common,
        "full_history_tmap": {"T4": 0.12, "T5": 0.11},
        "paired_ci": {"T4": (-0.03, -0.01), "T5": (-0.03, -0.01)},
    }
    assert (
        classify_system_outcome(
            **deficit,
            oracle_tmap={"T4": 0.115, "T5": 0.105},
        )["classification"]
        == "ASSOCIATION_LIMITED"
    )
    assert (
        classify_system_outcome(
            **deficit,
            oracle_tmap={"T4": 0.101, "T5": 0.091},
        )["classification"]
        == "REPRESENTATION_LIMITED"
    )


def test_final_report_answers_all_ten_questions_and_names_zero_shot_status() -> None:
    answers = [
        "T2.",
        "Yes, T3-T5 are zero-shot temporal-horizon extensions.",
        "Full-History trend is evidence-bound.",
        "Persist4D trend is evidence-bound.",
        "Deployment identity comparison is reported.",
        "Gap Identity Recovery comparison is reported.",
        "Per-new-visit latency scaling is reported.",
        "Peak VRAM scaling is reported.",
        "The Pareto decision follows the preregistered rule.",
        "Lock the method and proceed to external validation.",
    ]
    report = render_final_report(
        source_commit="a" * 40,
        classification="SYSTEM_PARETO_LOCK",
        answers=answers,
        evidence_files=("table_a_system_comparison.csv", "table_b_compute_scaling.csv"),
    )
    validated = validate_final_report(report)

    assert validated["classification"] == "SYSTEM_PARETO_LOCK"
    assert validated["answer_count"] == 10
    assert sum(label in report for label in ALLOWED_CLASSIFICATIONS) == 1
    assert "Frozen T2 Checkpoint" in report

    with pytest.raises(ArtifactError, match="10"):
        render_final_report(
            source_commit="a" * 40,
            classification="SYSTEM_LOCK",
            answers=answers[:-1],
            evidence_files=(),
        )


def test_artifact_manifest_hashes_every_required_lightweight_file(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "figures" / "second.svg"
    second.parent.mkdir()
    first.write_bytes(b"a,b\n1,2\n")
    second.write_bytes(b"<svg/>\n")
    manifest = build_artifact_manifest(
        tmp_path,
        required_paths=("first.csv", "figures/second.svg"),
        source_commit="b" * 40,
    )

    assert manifest["artifact_count"] == 2
    assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert manifest["artifacts"][1]["path"] == "figures/second.svg"

    second.unlink()
    with pytest.raises(ArtifactError, match="missing"):
        build_artifact_manifest(
            tmp_path,
            required_paths=("first.csv", "figures/second.svg"),
            source_commit="b" * 40,
        )
