from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from scripts.reviewer_closure_figures import (
    FigureEvidenceError,
    render_phase_iii_figures,
    render_strong_baseline_identity_scaling,
)


def _iou_rows() -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "horizon": horizon,
            "iou_threshold": threshold,
            "temporal_ap": 0.30 - threshold / 5 - (horizon - 4) / 50
            - (0.01 if method == "Persist4D" else 0.0),
            "aggregation": "pooled class-macro official stmetrics",
            "sequence_count": 129,
        }
        for method in ("FullHistory", "Persist4D")
        for horizon in (4, 5)
        for threshold in (
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
        )
    ]


def _coverage_rows() -> list[dict[str, object]]:
    fractions = {
        "no_candidate_observation": 0.10,
        "wrong_class": 0.15,
        "insufficient_iou": 0.20,
        "associable": 0.55,
    }
    return [
        {
            "method": method,
            "horizon": horizon,
            "iou_threshold": threshold,
            "category": category,
            "count": round(1000 * fraction),
            "total_gt_entity_stages": 1000,
            "fraction": fraction,
        }
        for method in ("FullHistory", "Persist4D")
        for horizon in (2, 3, 4, 5)
        for threshold in (0.25, 0.50, 0.75)
        for category, fraction in fractions.items()
    ]


def _failure_rows() -> list[dict[str, object]]:
    categories = (
        "local_observation_miss",
        "class_failure",
        "high_iou_mask_failure",
        "identity_fragmentation",
        "identity_merge",
        "wrong_gap_recovery",
        "capacity_failure",
        "unknown_unresolved",
    )
    return [
        {
            "method": "Persist4D",
            "horizon": horizon,
            "category": category,
            "count": 125,
            "total_failure_events": 1000,
            "fraction": 0.125,
            "operational_definition": f"measured definition for {category}",
        }
        for horizon in (4, 5)
        for category in categories
    ]


def _oracle_rows() -> list[dict[str, object]]:
    semantics = {
        "FullHistory": "frozen primary system metric",
        "Persist4D": "frozen primary system metric",
        "Oracle": (
            "P6-A offline GT-ID readout; unmatched candidates retained; "
            "masks/classes unchanged"
        ),
    }
    return [
        {
            "method": method,
            "horizon": horizon,
            "t_mAP": 0.30 - horizon / 40
            - ({"FullHistory": 0.0, "Persist4D": 0.01, "Oracle": 0.12}[method]),
            "diagnostic_semantics": semantics[method],
        }
        for horizon in (2, 3, 4, 5)
        for method in ("FullHistory", "Oracle", "Persist4D")
    ]


def _tracker_rows() -> list[dict[str, object]]:
    offsets = {"FullHistoryNative": 0.80, "B2": 0.10, "Persist4D": 0.07}
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "sequence_count": 129,
            "normalized_id_switch_rate": (
                None
                if horizon == 2 and method != "Persist4D"
                else offsets[method] + horizon / 100
            ),
        }
        for method in ("FullHistoryNative", "B2", "Persist4D")
        for horizon in (2, 3, 4, 5)
    ]


def test_phase_iii_figures_are_traceable_accessible_and_multiformat(
    tmp_path: Path,
) -> None:
    paths = render_phase_iii_figures(
        _iou_rows(),
        _coverage_rows(),
        _failure_rows(),
        _oracle_rows(),
        tmp_path,
    )

    assert len(paths) == 15
    assert {path.suffix for path in paths} == {".svg", ".pdf", ".png"}
    assert {path.stem for path in paths} == {
        "iou_threshold_curve",
        "observation_coverage",
        "failure_decomposition",
        "oracle_association_gain",
        "performance_decomposition",
    }
    for path in paths:
        assert path.stat().st_size > 500
    for path in (candidate for candidate in paths if candidate.suffix == ".svg"):
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        title = root.find("{http://www.w3.org/2000/svg}title")
        description = root.find("{http://www.w3.org/2000/svg}desc")
        assert title is not None and (title.text or "").strip()
        assert description is not None and "Source:" in (description.text or "")
        text = path.read_text(encoding="utf-8")
        assert all(line == line.rstrip() for line in text.splitlines())
        assert "gradient" not in text.lower()
        if path.stem in {
            "iou_threshold_curve",
            "oracle_association_gain",
            "performance_decomposition",
        }:
            assert "#0072b2" in text.lower()
            assert "#d55e00" in text.lower()


def test_identity_scaling_handles_t2_as_not_applicable(tmp_path: Path) -> None:
    paths = render_strong_baseline_identity_scaling(_tracker_rows(), tmp_path)

    assert [path.name for path in paths] == [
        "strong_baseline_identity_scaling.svg",
        "strong_baseline_identity_scaling.pdf",
        "strong_baseline_identity_scaling.png",
    ]
    svg = paths[0].read_text(encoding="utf-8")
    assert "Full-History native" in svg
    assert "B2 feature + class" in svg
    assert "Persist4D" in svg
    assert "T2 initialization: no ID-transition rate for Full-History/B2" in svg


def test_rendering_rejects_incomplete_or_inconsistent_evidence(tmp_path: Path) -> None:
    with pytest.raises(FigureEvidenceError, match="IoU sweep"):
        render_phase_iii_figures(
            _iou_rows()[:-1],
            _coverage_rows(),
            _failure_rows(),
            _oracle_rows(),
            tmp_path,
        )

    bad_coverage = _coverage_rows()
    bad_coverage[0] = {**bad_coverage[0], "fraction": 0.2}
    with pytest.raises(FigureEvidenceError, match="sum to one"):
        render_phase_iii_figures(
            _iou_rows(),
            bad_coverage,
            _failure_rows(),
            _oracle_rows(),
            tmp_path,
        )
