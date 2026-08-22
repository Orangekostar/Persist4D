"""Publication tables and editable SVGs for the system comparison."""

from __future__ import annotations

import html
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

METHODS = ("FullHistory", "Persist4D")
HORIZONS = (2, 3, 4, 5)
FULL_COLOR = "#0072B2"
PERSIST_COLOR = "#D55E00"
NEUTRAL_COLOR = "#666666"


class FigureError(ValueError):
    """Raised when a table or figure lacks measured source evidence."""


def _finite_or_missing(value: object, *, name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FigureError(f"{name} must be finite or missing")
    result = float(value)
    if not math.isfinite(result):
        raise FigureError(f"{name} must be finite or missing")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FigureError(f"{name} must be a non-negative integer")
    return value


def build_table_a(
    aggregate_rows: Sequence[Mapping[str, object]],
    b3_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_cell: dict[tuple[str, int], Mapping[str, object]] = {}
    for row in aggregate_rows:
        method = row.get("method")
        horizon = row.get("horizon")
        if method not in METHODS or horizon not in HORIZONS or row.get("order_id") != "all":
            continue
        cell = (str(method), int(horizon))
        if cell in by_cell:
            raise FigureError("Table A aggregate inputs contain duplicate cells")
        by_cell[cell] = row
    if set(by_cell) != {
        (method, horizon) for method in METHODS for horizon in HORIZONS
    }:
        raise FigureError("Table A aggregate inputs lack exact method/horizon coverage")
    b3_by_horizon = {}
    for row in b3_rows:
        horizon = row.get("T")
        if row.get("method") != "B3" or horizon not in HORIZONS:
            continue
        if horizon in b3_by_horizon:
            raise FigureError("Table A B3 inputs contain duplicate horizons")
        b3_by_horizon[int(horizon)] = row
    if set(b3_by_horizon) != set(HORIZONS):
        raise FigureError("Table A B3 inputs lack T2-T5 coverage")

    metadata = {
        "FullHistory": (
            "ReScene4D Full-History (Frozen T2 Checkpoint)",
            "Reprocess exact observed prefix",
        ),
        "B3": ("EMA Temporal Association", "Local pair + unbounded EMA IDs"),
        "Persist4D": (
            "Persist4D Persistent-State",
            "Local pair + bounded entity state",
        ),
    }
    result = []
    for method in ("FullHistory", "B3", "Persist4D"):
        for horizon in HORIZONS:
            display, strategy = metadata[method]
            if method == "B3":
                source = b3_by_horizon[horizon]
                tmap = _finite_or_missing(source.get("t_mAP"), name="B3 t_mAP")
                trec = _finite_or_missing(source.get("t_REC"), name="B3 t_REC")
                idsw = _finite_or_missing(
                    source.get("id_switch_rate"), name="B3 id switch rate"
                )
                gap_accuracy = None
                gap = None
            else:
                source = by_cell[(method, horizon)]
                tmap = _finite_or_missing(
                    source.get("causal_prefix_t_mAP"), name="t_mAP"
                )
                trec = _finite_or_missing(
                    source.get("causal_prefix_t_REC"), name="t_REC"
                )
                idsw = _finite_or_missing(
                    source.get("normalized_id_switch_rate"), name="IDSW rate"
                )
                gap = _finite_or_missing(
                    source.get("gap_recovery_recall"), name="gap recovery recall"
                )
                gap_accuracy = _finite_or_missing(
                    source.get("gap_recovery_accuracy"), name="gap recovery accuracy"
                )
            if tmap is None or trec is None:
                raise FigureError("Table A task metrics cannot be missing")
            result.append(
                {
                    "method_id": method,
                    "method": display,
                    "history_strategy": strategy,
                    "horizon": horizon,
                    "causal_prefix_t_mAP": tmap,
                    "causal_prefix_t_REC": trec,
                    "normalized_id_switch_rate": idsw,
                    "gap_recovery_accuracy": gap_accuracy,
                    "gap_recovery_recall": gap,
                }
            )
    return result


def _mean(values: Sequence[object], *, name: str) -> float:
    normalized = [_finite_or_missing(value, name=name) for value in values]
    if any(value is None for value in normalized):
        raise FigureError(f"{name} cannot be missing")
    return float(statistics.mean(value for value in normalized if value is not None))


def _maximum_optional(values: Sequence[object], *, name: str) -> int | None:
    normalized = [None if value in (None, "") else _integer(value, name=name) for value in values]
    present = [value for value in normalized if value is not None]
    return max(present) if present else None


def build_table_b(
    profile_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    identities = set()
    for row in profile_rows:
        method = row.get("method")
        horizon = row.get("horizon")
        if method not in METHODS or horizon not in HORIZONS:
            raise FigureError("Table B profile cell is outside the frozen protocol")
        if row.get("order_id") != "canonical" or row.get("status") != "pass":
            raise FigureError("Table B requires passing canonical profile rows")
        identity = (
            str(method),
            str(row.get("reference_scene_id")),
            str(row.get("master_sequence_id")),
            int(horizon),
        )
        if identity in identities:
            raise FigureError("Table B profile inputs contain duplicate cells")
        identities.add(identity)
        grouped[(str(method), int(horizon))].append(row)
    if set(grouped) != {
        (method, horizon) for method in METHODS for horizon in HORIZONS
    } or any(len(rows) != 6 for rows in grouped.values()):
        raise FigureError("Table B requires six clusters for every method/horizon")

    display_names = {
        "FullHistory": "ReScene4D Full-History (Frozen T2 Checkpoint)",
        "Persist4D": "Persist4D Persistent-State",
    }
    result = []
    for method in METHODS:
        for horizon in HORIZONS:
            rows = grouped[(method, horizon)]
            state_bytes = _maximum_optional(
                [row.get("persistent_state_bytes") for row in rows],
                name="persistent state bytes",
            )
            explicit_bytes = _maximum_optional(
                [row.get("explicit_history_input_bytes") for row in rows],
                name="explicit history input bytes",
            )
            if method == "FullHistory" and (state_bytes is not None or explicit_bytes is None):
                raise FigureError("Full-History must report explicit input, not state bytes")
            if method == "Persist4D" and (state_bytes is None or explicit_bytes is not None):
                raise FigureError("Persist4D must report bounded state, not explicit history")
            result.append(
                {
                    "method_id": method,
                    "method": display_names[method],
                    "horizon": horizon,
                    "profile_cluster_count": 6,
                    "scans_processed_per_update": _mean(
                        [row.get("update_scan_count") for row in rows],
                        name="update scan count",
                    ),
                    "cumulative_scans_processed": _mean(
                        [row.get("cumulative_scan_count") for row in rows],
                        name="cumulative scan count",
                    ),
                    "median_latency_ms": float(
                        statistics.median(
                            _finite_or_missing(
                                row.get("median_latency_ms"), name="latency"
                            )
                            for row in rows
                        )
                    ),
                    "peak_allocated_mib": max(
                        _finite_or_missing(
                            row.get("peak_allocated_mib"), name="peak allocated MiB"
                        )
                        for row in rows
                    ),
                    "peak_reserved_mib": max(
                        _finite_or_missing(
                            row.get("peak_reserved_mib"), name="peak reserved MiB"
                        )
                        for row in rows
                    ),
                    "mean_update_point_count": _mean(
                        [row.get("update_point_count") for row in rows],
                        name="update point count",
                    ),
                    "mean_cumulative_point_count": _mean(
                        [row.get("cumulative_point_count") for row in rows],
                        name="cumulative point count",
                    ),
                    "historical_state_bytes": state_bytes,
                    "explicit_history_input_bytes": explicit_bytes,
                }
            )
    return result


def _publish_exact(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"figure output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"figure output contains different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _marker(x: float, y: float, *, color: str, method: str) -> str:
    if method == "FullHistory":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
    return (
        f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" '
        f'fill="{color}"/>'
    )


def _line_svg(
    *,
    title: str,
    ylabel: str,
    source: str,
    series: Sequence[Mapping[str, object]],
) -> str:
    values = [
        float(value)
        for item in series
        for value in item["values"]
        if value is not None
    ]
    if not values:
        raise FigureError(f"figure {title} has no finite data")
    lower = min(0.0, min(values))
    upper = max(values)
    span = upper - lower
    upper = upper + (0.12 * span if span else 1.0)
    left, right, bottom = 96.0, 682.0, 350.0
    top = 72.0 if len(series) <= 2 else 48.0 + (len(series) - 1) * 18.0 + 24.0

    def x_coord(horizon: int) -> float:
        return left + (horizon - 2) * (right - left) / 3

    def y_coord(value: float) -> float:
        return bottom - (value - lower) * (bottom - top) / (upper - lower)

    escaped_title = html.escape(title)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        f"<title>{escaped_title}</title>",
        (
            f"<desc>Source: {html.escape(source)}. Frozen T2 checkpoint; T3-T5 "
            "are zero-shot temporal-horizon extensions.</desc>"
        ),
        '<rect width="720" height="420" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222222" font-size="12">',
        f'<text x="{left}" y="30" font-size="16" font-weight="700">{escaped_title}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222222"/>',
    ]
    for tick in range(5):
        value = lower + tick * (upper - lower) / 4
        y = y_coord(value)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#D9D9D9"/>',
                f'<text x="86" y="{y + 4:.2f}" text-anchor="end">{value:.3g}</text>',
            ]
        )
    for horizon in HORIZONS:
        x = x_coord(horizon)
        elements.append(
            f'<text x="{x:.2f}" y="372" text-anchor="middle">T{horizon}</text>'
        )
    elements.extend(
        [
            '<text x="380" y="402" text-anchor="middle">Temporal horizon</text>',
            f'<text x="18" y="215" text-anchor="middle" transform="rotate(-90 18 215)">{html.escape(ylabel)}</text>',
        ]
    )
    for series_index, item in enumerate(series):
        method = str(item["method"])
        color = str(item["color"])
        dash = ' stroke-dasharray="7 4"' if item.get("dashed") else ""
        points = []
        for horizon, value in zip(HORIZONS, item["values"], strict=True):
            if value is not None:
                points.append((x_coord(horizon), y_coord(float(value))))
        if len(points) >= 2:
            path = " ".join(
                ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
                for index, (x, y) in enumerate(points)
            )
            elements.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4"{dash}/>'
            )
        elements.extend(
            _marker(x, y, color=color, method=method) for x, y in points
        )
        legend_y = 48 + series_index * 18
        elements.append(
            f'<line x1="430" y1="{legend_y}" x2="452" y2="{legend_y}" stroke="{color}" stroke-width="2.4"{dash}/>'
        )
        elements.append(_marker(441, legend_y, color=color, method=method))
        elements.append(
            f'<text x="460" y="{legend_y + 4}">{html.escape(str(item["label"]))}</text>'
        )
    elements.extend(["</g>", "</svg>", ""])
    return "\n".join(elements)


def _pareto_svg(table_a: Sequence[Mapping[str, object]], table_b: Sequence[Mapping[str, object]]) -> str:
    quality = {
        (row["method_id"], row["horizon"]): float(row["causal_prefix_t_mAP"])
        for row in table_a
        if row["method_id"] in METHODS
    }
    efficiency = {
        (row["method_id"], row["horizon"]): float(row["median_latency_ms"])
        for row in table_b
    }
    points = [
        (method, horizon, efficiency[(method, horizon)], quality[(method, horizon)])
        for method in METHODS
        for horizon in HORIZONS
    ]
    x_values = [point[2] for point in points]
    y_values = [point[3] for point in points]
    x_min, x_max = min(x_values) * 0.9, max(x_values) * 1.1
    y_min, y_max = min(0.0, min(y_values) * 0.9), max(y_values) * 1.12
    left, right, top, bottom = 78.0, 682.0, 72.0, 350.0

    def x_coord(value: float) -> float:
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def y_coord(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="420" viewBox="0 0 720 420">',
        "<title>Accuracy-Compute Pareto</title>",
        (
            "<desc>Source: table_a_system_comparison.csv and "
            "table_b_compute_scaling.csv. Frozen T2 checkpoint; T3-T5 are "
            "zero-shot temporal-horizon extensions.</desc>"
        ),
        '<rect width="720" height="420" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222222" font-size="12">',
        '<text x="78" y="30" font-size="16" font-weight="700">Accuracy-Compute Pareto</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222222"/>',
        '<text x="380" y="402" text-anchor="middle">Median update latency (ms)</text>',
        '<text x="18" y="215" text-anchor="middle" transform="rotate(-90 18 215)">Causal-prefix t-mAP</text>',
    ]
    for tick in range(5):
        x_value = x_min + tick * (x_max - x_min) / 4
        y_value = y_min + tick * (y_max - y_min) / 4
        x, y = x_coord(x_value), y_coord(y_value)
        elements.extend(
            [
                f'<text x="{x:.2f}" y="372" text-anchor="middle">{x_value:.3g}</text>',
                f'<text x="68" y="{y + 4:.2f}" text-anchor="end">{y_value:.3g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#D9D9D9"/>',
            ]
        )
    for method, horizon, latency, quality_value in points:
        color = FULL_COLOR if method == "FullHistory" else PERSIST_COLOR
        x, y = x_coord(latency), y_coord(quality_value)
        elements.append(_marker(x, y, color=color, method=method))
        elements.append(
            f'<text x="{x + 7:.2f}" y="{y - 7:.2f}" fill="{color}">{"FH" if method == "FullHistory" else "P"}-T{horizon}</text>'
        )
    elements.extend(["</g>", "</svg>", ""])
    return "\n".join(elements)


def render_required_figures(
    table_a: Sequence[Mapping[str, object]],
    table_b: Sequence[Mapping[str, object]],
    output_directory: Path,
) -> tuple[Path, ...]:
    table_a_by = {(row["method_id"], row["horizon"]): row for row in table_a}
    table_b_by = {(row["method_id"], row["horizon"]): row for row in table_b}
    if len(table_a_by) != 12 or len(table_b_by) != 8:
        raise FigureError("figure inputs lack exact table coverage")

    def values(table, method: str, metric: str):
        return [table[(method, horizon)][metric] for horizon in HORIZONS]

    figures = (
        (
            "figure_1_task_quality.svg",
            _line_svg(
                title="Task Quality vs Temporal Horizon",
                ylabel="Causal-prefix t-mAP",
                source="table_a_system_comparison.csv",
                series=(
                    {"method": "FullHistory", "label": "Full-History", "color": FULL_COLOR, "values": values(table_a_by, "FullHistory", "causal_prefix_t_mAP")},
                    {"method": "Persist4D", "label": "Persist4D", "color": PERSIST_COLOR, "values": values(table_a_by, "Persist4D", "causal_prefix_t_mAP")},
                ),
            ),
        ),
        (
            "figure_2_identity_stability.svg",
            _line_svg(
                title="Deployment Identity Stability",
                ylabel="Normalized ID switch rate",
                source="table_a_system_comparison.csv",
                series=(
                    {"method": "FullHistory", "label": "Full-History", "color": FULL_COLOR, "values": values(table_a_by, "FullHistory", "normalized_id_switch_rate")},
                    {"method": "Persist4D", "label": "Persist4D", "color": PERSIST_COLOR, "values": values(table_a_by, "Persist4D", "normalized_id_switch_rate")},
                ),
            ),
        ),
        (
            "figure_3_gap_recovery.svg",
            _line_svg(
                title="Gap Identity Recovery",
                ylabel="Recovery rate",
                source="aggregate_results.csv",
                series=tuple(
                    {
                        "method": method,
                        "label": f"{'Full-History' if method == 'FullHistory' else 'Persist4D'} {label}",
                        "color": FULL_COLOR if method == "FullHistory" else PERSIST_COLOR,
                        "values": values(table_a_by, method, metric),
                        "dashed": label == "Accuracy",
                    }
                    for method in METHODS
                    for label, metric in (
                        ("Accuracy", "gap_recovery_accuracy"),
                        ("Recall", "gap_recovery_recall"),
                    )
                ),
            ),
        ),
        (
            "figure_4_latency_scaling.svg",
            _line_svg(
                title="Per-New-Visit Latency Scaling",
                ylabel="Median latency (ms/update)",
                source="table_b_compute_scaling.csv",
                series=(
                    {"method": "FullHistory", "label": "Full-History", "color": FULL_COLOR, "values": values(table_b_by, "FullHistory", "median_latency_ms")},
                    {"method": "Persist4D", "label": "Persist4D", "color": PERSIST_COLOR, "values": values(table_b_by, "Persist4D", "median_latency_ms")},
                ),
            ),
        ),
        (
            "figure_5_peak_vram.svg",
            _line_svg(
                title="Peak GPU Memory Scaling",
                ylabel="Peak allocated VRAM (MiB)",
                source="table_b_compute_scaling.csv",
                series=(
                    {"method": "FullHistory", "label": "Full-History", "color": FULL_COLOR, "values": values(table_b_by, "FullHistory", "peak_allocated_mib")},
                    {"method": "Persist4D", "label": "Persist4D", "color": PERSIST_COLOR, "values": values(table_b_by, "Persist4D", "peak_allocated_mib")},
                ),
            ),
        ),
        (
            "figure_6_accuracy_compute_pareto.svg",
            _pareto_svg(table_a, table_b),
        ),
    )
    paths = []
    for filename, svg in figures:
        path = output_directory / filename
        _publish_exact(path, svg)
        paths.append(path)
    return tuple(paths)


__all__ = [
    "FigureError",
    "build_table_a",
    "build_table_b",
    "render_required_figures",
]
