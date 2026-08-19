"""Deterministic publication figures for Persist4D P6-B evidence."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence

_METHODS = ("B4", "P6B")
_HORIZONS = ("T2", "T3", "T4", "T5")


def _metric_rows(
    rows: Sequence[Mapping[str, object]], metric: str, *, allow_t2_none: bool
) -> dict[tuple[str, str], float | None]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("figure rows must be a sequence")
    result: dict[tuple[str, str], float | None] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("figure rows must contain mappings")
        method = row.get("method")
        horizon = row.get("T")
        key = (method, horizon)
        if method not in _METHODS or horizon not in _HORIZONS or key in result:
            raise ValueError("figure rows must contain exact method/horizon pairs")
        value = row.get(metric)
        if value is None and allow_t2_none and horizon == "T2":
            result[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{metric} must be finite")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{metric} must be finite in [0, 1]")
        result[key] = number
    expected = {(method, horizon) for method in _METHODS for horizon in _HORIZONS}
    if set(result) != expected:
        raise ValueError("figure rows must contain exact method/horizon pairs")
    return result


def _render_grouped_bars(
    values: Mapping[tuple[str, str], float | None], *, title: str, ylabel: str
) -> str:
    width, height = 720, 360
    left, top, plot_width, plot_height = 70, 50, 610, 245
    colors = {"B4": "#4C78A8", "P6B": "#E45756"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>'
        )
    group_width = plot_width / len(_HORIZONS)
    bar_width = 34
    for group, horizon in enumerate(_HORIZONS):
        center = left + group_width * (group + 0.5)
        parts.append(
            f'<text x="{center:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{horizon}</text>'
        )
        for offset, method in enumerate(_METHODS):
            value = values[(method, horizon)]
            if value is None:
                continue
            x = center + (offset - 0.5) * (bar_width + 8) - bar_width / 2
            bar_height = plot_height * value
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="{colors[method]}"/>'
            )
    for index, method in enumerate(_METHODS):
        x = left + 175 + index * 135
        parts.extend(
            (
                f'<rect x="{x}" y="325" width="14" height="14" fill="{colors[method]}"/>',
                f'<text x="{x + 21}" y="337" font-family="sans-serif" font-size="12">{method}</text>',
            )
        )
    parts.append(
        f'<text transform="translate(17 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_identity_figure(rows: Sequence[Mapping[str, object]]) -> str:
    return _render_grouped_bars(
        _metric_rows(rows, "identity_switch_rate", allow_t2_none=False),
        title="Held-out identity continuity",
        ylabel="ID switch rate",
    )


def render_reactivation_figure(rows: Sequence[Mapping[str, object]]) -> str:
    return _render_grouped_bars(
        _metric_rows(rows, "reactivation_accuracy", allow_t2_none=True),
        title="Held-out dormant reactivation",
        ylabel="React. accuracy",
    )
