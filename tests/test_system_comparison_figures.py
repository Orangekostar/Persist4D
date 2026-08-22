from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.system_comparison_figures import (
    build_table_a,
    build_table_b,
    render_required_figures,
)


def _aggregate_rows() -> list[dict[str, object]]:
    rows = []
    for method, offset in (("FullHistory", 0.0), ("Persist4D", 0.01)):
        for horizon in (2, 3, 4, 5):
            rows.append(
                {
                    "method": method,
                    "order_id": "all",
                    "horizon": horizon,
                    "causal_prefix_t_mAP": 0.2 - horizon / 100 + offset,
                    "causal_prefix_t_REC": 0.3 - horizon / 100 + offset,
                    "normalized_id_switch_rate": horizon / 20
                    - (0.05 if method == "Persist4D" else 0.0),
                    "gap_recovery_accuracy": None if horizon == 2 else 0.5 + offset,
                    "gap_recovery_recall": None if horizon == 2 else 0.4 + offset,
                }
            )
    return rows


def _b3_rows() -> list[dict[str, object]]:
    return [
        {"method": "B3", "T": horizon, "t_mAP": 0.1, "t_REC": 0.2}
        for horizon in (2, 3, 4, 5)
    ]


def _profile_rows() -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "reference_scene_id": f"reference-{cluster}",
            "master_sequence_id": f"master-{cluster}",
            "order_id": "canonical",
            "horizon": horizon,
            "status": "pass",
            "median_latency_ms": 10.0 * horizon
            * (1.5 if method == "FullHistory" else 1.0),
            "peak_allocated_mib": 1000.0 + horizon * 100
            * (2 if method == "FullHistory" else 1),
            "peak_reserved_mib": 1200.0 + horizon * 100
            * (2 if method == "FullHistory" else 1),
            "update_scan_count": horizon if method == "FullHistory" else 2,
            "cumulative_scan_count": (
                horizon * (horizon + 1) // 2
                if method == "FullHistory"
                else 1 + 2 * (horizon - 1)
            ),
            "update_point_count": 1000 * horizon,
            "cumulative_point_count": 2000 * horizon,
            "model_input_bytes": 10000 * horizon,
            "cumulative_model_input_bytes": 20000 * horizon,
            "persistent_state_bytes": 4096 if method == "Persist4D" else None,
            "explicit_history_input_bytes": (
                10000 * horizon if method == "FullHistory" else None
            ),
        }
        for cluster in range(6)
        for horizon in (2, 3, 4, 5)
        for method in ("FullHistory", "Persist4D")
    ]


def test_table_a_and_b_have_exact_method_horizon_coverage() -> None:
    table_a = build_table_a(_aggregate_rows(), _b3_rows())
    table_b = build_table_b(_profile_rows())

    assert len(table_a) == 3 * 4
    assert {(row["method_id"], row["horizon"]) for row in table_a} == {
        (method, horizon)
        for method in ("FullHistory", "B3", "Persist4D")
        for horizon in (2, 3, 4, 5)
    }
    assert len(table_b) == 2 * 4
    assert all(row["profile_cluster_count"] == 6 for row in table_b)
    full = next(
        row
        for row in table_b
        if row["method_id"] == "FullHistory" and row["horizon"] == 5
    )
    persistent = next(
        row
        for row in table_b
        if row["method_id"] == "Persist4D" and row["horizon"] == 5
    )
    assert full["historical_state_bytes"] is None
    assert full["explicit_history_input_bytes"] is not None
    assert persistent["historical_state_bytes"] == 4096
    assert persistent["explicit_history_input_bytes"] is None


def test_six_svg_figures_are_valid_traceable_and_accessible(tmp_path: Path) -> None:
    table_a = build_table_a(_aggregate_rows(), _b3_rows())
    table_b = build_table_b(_profile_rows())
    paths = render_required_figures(table_a, table_b, tmp_path)

    assert [path.name for path in paths] == [
        "figure_1_task_quality.svg",
        "figure_2_identity_stability.svg",
        "figure_3_gap_recovery.svg",
        "figure_4_latency_scaling.svg",
        "figure_5_peak_vram.svg",
        "figure_6_accuracy_compute_pareto.svg",
    ]
    for path in paths:
        root = ET.parse(path).getroot()
        assert root.tag.endswith("svg")
        assert root.attrib["viewBox"] == "0 0 720 420"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        description = root.find("{http://www.w3.org/2000/svg}desc")
        assert description is not None and "Source:" in (description.text or "")
        assert "zero-shot" in (description.text or "")
        text = path.read_text(encoding="utf-8")
        assert "#0072B2" in text
        assert "#D55E00" in text
        assert "gradient" not in text.lower()
        assert "font-size=\"10\"" not in text

    gap_root = ET.parse(tmp_path / "figure_3_gap_recovery.svg").getroot()
    vertical_axis = next(
        element
        for element in gap_root.iter("{http://www.w3.org/2000/svg}line")
        if element.attrib.get("x1") == "96.0"
        and element.attrib.get("x2") == "96.0"
    )
    assert float(vertical_axis.attrib["y1"]) > 102.0

    vram_root = ET.parse(tmp_path / "figure_5_peak_vram.svg").getroot()
    vram_text = list(vram_root.iter("{http://www.w3.org/2000/svg}text"))
    y_ticks = [
        element
        for element in vram_text
        if element.attrib.get("text-anchor") == "end"
    ]
    assert len(y_ticks) == 5
    assert {element.attrib.get("x") for element in y_ticks} == {"86"}
