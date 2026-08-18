"""Deterministic, dependency-free SVG renderers for P6-A Figures A-E."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from html import escape
from itertools import pairwise
from numbers import Real
from typing import Any

METHOD_ORDER = ("b0", "b0_sanity", "b1", "b2", "b3", "b4", "oracle")
BASELINE_METHOD_ORDER = ("b0", "b0_sanity", "b1", "b2", "b3", "b4")
REACTIVATION_METHOD_ORDER = ("b1", "b2", "b3", "b4")
REACTIVATION_HORIZONS = (3, 4, 5)
LATENCY_METHOD_ORDER = ("b4", "full_history_rescene")
HORIZONS = (2, 3, 4, 5)
OUTCOMES = ("correct", "wrong")
REACTIVATION_METRICS = ("best_score", "score_margin")
FAILURE_CATEGORIES = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")
PHASES = ("bootstrap", "new_visit")
TOLERANCE = 1e-9

_PALETTE = {
    "b0": "#000000",
    "b0_sanity": "#E69F00",
    "b1": "#56B4E9",
    "b2": "#009E73",
    "b3": "#F0E442",
    "b4": "#0072B2",
    "oracle": "#D55E00",
    "full_history_rescene": "#CC79A7",
}
_LINE_STYLES = {
    "b0": "",
    "b0_sanity": "6 3",
    "b1": "2 3",
    "b2": "10 3 2 3",
    "b3": "1 3",
    "b4": "8 2",
    "oracle": "14 3 2 3",
    "full_history_rescene": "4 4",
}
_MARKERS = {
    "b0": "circle",
    "b0_sanity": "square",
    "b1": "triangle",
    "b2": "diamond",
    "b3": "cross",
    "b4": "plus",
    "oracle": "star",
    "full_history_rescene": "diamond",
}
_CATEGORY_FILLS = (
    "#1A1A1A",
    "#3D3D3D",
    "#5C5C5C",
    "#7A7A7A",
    "#999999",
    "#B8B8B8",
    "#D6D6D6",
    "#F2F2F2",
)
_CATEGORY_DASHES = (
    "",
    "6 2",
    "2 2",
    "8 2 2 2",
    "1 2",
    "10 2",
    "4 2 1 2",
    "3 2 1 2",
)

_SCHEMA_A = frozenset(("method_id", "horizon", "id_switch_rate"))
_SCHEMA_B = frozenset(("method_id", "horizon", "online_t_mAP"))
_SCHEMA_C = frozenset(
    (
        "method_id",
        "horizon",
        "outcome",
        "metric",
        "bin_low",
        "bin_high",
        "count",
        "fraction",
    )
)
_SCHEMA_D = frozenset(("method_id", "horizon", "category", "count", "share"))
_SCHEMA_E = frozenset(("method_id", "horizon", "phase", "latency_ms"))


def _as_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(rows, (str, bytes, Mapping)):
        raise ValueError(  # noqa: TRY004
            "rows must be a non-empty iterable of mappings"
        )
    try:
        materialized = tuple(rows)
    except TypeError as error:
        raise ValueError("rows must be a non-empty iterable of mappings") from error
    if not materialized:
        raise ValueError("rows cannot be empty")
    if any(not isinstance(row, Mapping) for row in materialized):
        raise ValueError("every row must be a mapping")
    return materialized


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")  # noqa: TRY004
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _horizon(value: Any, *, horizons: Sequence[int] = HORIZONS) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in horizons:
        choices = ", ".join(str(horizon) for horizon in horizons)
        raise ValueError(f"horizon must be one of {choices}")
    return int(value)


def _method(value: Any, *, methods: Sequence[str] = METHOD_ORDER) -> str:
    if not isinstance(value, str) or value not in methods:
        raise ValueError(f"method_id must be one of {', '.join(methods)}")
    return value


def _count(value: Any, *, name: str = "count") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return int(value)


def _fraction(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return result


def _check_columns(row: Mapping[str, Any], schema: frozenset[str]) -> None:
    if frozenset(row.keys()) != schema:
        raise ValueError("row columns must exactly match the figure schema")


def _base_records(
    rows: Iterable[Mapping[str, Any]],
    schema: frozenset[str],
    value_name: str,
    *,
    value_validator: Callable[[Any], Any] = _finite,
) -> list[dict[str, Any]]:
    materialized = _as_rows(rows)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in materialized:
        _check_columns(row, schema)
        method = _method(row["method_id"])
        horizon = _horizon(row["horizon"])
        key = (method, horizon)
        if key in seen:
            raise ValueError("duplicate primary key")
        seen.add(key)
        records.append(
            {
                "method_id": method,
                "horizon": horizon,
                value_name: value_validator(row[value_name], name=value_name),
            }
        )
    expected = {
        (method, horizon)
        for method in BASELINE_METHOD_ORDER
        for horizon in HORIZONS
    }
    missing = expected - seen
    if missing:
        raise ValueError("missing method/horizon group")
    oracle_seen = {key for key in seen if key[0] == "oracle"}
    oracle_expected = {("oracle", horizon) for horizon in HORIZONS}
    if oracle_seen and oracle_seen != oracle_expected:
        raise ValueError("missing oracle method/horizon group")
    return sorted(records, key=lambda record: (METHOD_ORDER.index(record["method_id"]), record["horizon"]))


def _c_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = _as_rows(rows)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str, float, float]] = set()
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in materialized:
        _check_columns(row, _SCHEMA_C)
        method = _method(row["method_id"], methods=REACTIVATION_METHOD_ORDER)
        horizon = _horizon(row["horizon"], horizons=REACTIVATION_HORIZONS)
        outcome = row["outcome"]
        metric = row["metric"]
        if outcome not in OUTCOMES:
            raise ValueError("outcome must be correct or wrong")
        if metric not in REACTIVATION_METRICS:
            raise ValueError("metric must be best_score or score_margin")
        bin_low = _finite(row["bin_low"], name="bin_low")
        bin_high = _finite(row["bin_high"], name="bin_high")
        if bin_high <= bin_low:
            raise ValueError("bin_high must be greater than bin_low")
        key = (method, horizon, outcome, metric, bin_low, bin_high)
        if key in seen:
            raise ValueError("duplicate primary key")
        seen.add(key)
        group_key = (method, horizon, outcome, metric)
        record = {
            "method_id": method,
            "horizon": horizon,
            "outcome": outcome,
            "metric": metric,
            "bin_low": bin_low,
            "bin_high": bin_high,
            "count": _count(row["count"]),
            "fraction": _fraction(row["fraction"], name="fraction"),
        }
        groups.setdefault(group_key, []).append(record)
        records.append(record)
    group_keys = set(groups)
    for method, horizon, outcome, metric in group_keys:
        counterpart = (
            method,
            horizon,
            "wrong" if outcome == "correct" else "correct",
            metric,
        )
        if counterpart not in group_keys:
            raise ValueError("correct and wrong reactivation groups must be paired")
    for group_records in groups.values():
        ordered = sorted(group_records, key=lambda record: (record["bin_low"], record["bin_high"]))
        total_count = sum(record["count"] for record in ordered)
        total = sum(record["fraction"] for record in group_records)
        expected_total = 1.0 if total_count else 0.0
        if not math.isclose(
            total, expected_total, abs_tol=TOLERANCE, rel_tol=0.0
        ) or any(
            not math.isclose(
                record["fraction"],
                record["count"] / total_count if total_count else 0.0,
                abs_tol=TOLERANCE,
                rel_tol=0.0,
            )
            for record in ordered
        ):
            raise ValueError("reactivation fractions must match counts")
        for previous, current in pairwise(ordered):
            if previous["bin_high"] != current["bin_low"]:
                raise ValueError("reactivation bins must be continuous")
    return sorted(
        records,
        key=lambda record: (
            METHOD_ORDER.index(record["method_id"]),
            record["horizon"],
            OUTCOMES.index(record["outcome"]),
            REACTIVATION_METRICS.index(record["metric"]),
            record["bin_low"],
            record["bin_high"],
        ),
    )


def _d_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = _as_rows(rows)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in materialized:
        _check_columns(row, _SCHEMA_D)
        method = _method(row["method_id"])
        horizon = _horizon(row["horizon"])
        category = row["category"]
        if category not in FAILURE_CATEGORIES:
            raise ValueError("category must be F1 through F7 or unclassified")
        key = (method, horizon, category)
        if key in seen:
            raise ValueError("duplicate primary key")
        seen.add(key)
        record = {
            "method_id": method,
            "horizon": horizon,
            "category": category,
            "count": _count(row["count"]),
            "share": _fraction(row["share"], name="share"),
        }
        groups.setdefault((method, horizon), []).append(record)
        records.append(record)
    for group_records in groups.values():
        categories = {record["category"] for record in group_records}
        if categories != set(FAILURE_CATEGORIES):
            raise ValueError("missing failure category group")
        total = sum(record["share"] for record in group_records)
        if not math.isclose(total, 1.0, abs_tol=TOLERANCE, rel_tol=0.0):
            raise ValueError("failure share must close to 1")
    b4_keys = {(method, horizon) for method, horizon, _ in seen if method == "b4"}
    if b4_keys != {("b4", horizon) for horizon in HORIZONS}:
        raise ValueError("missing b4 failure horizon group")
    return sorted(
        records,
        key=lambda record: (
            METHOD_ORDER.index(record["method_id"]),
            record["horizon"],
            FAILURE_CATEGORIES.index(record["category"]),
        ),
    )


def _e_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    materialized = _as_rows(rows)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in materialized:
        _check_columns(row, _SCHEMA_E)
        method = _method(row["method_id"], methods=LATENCY_METHOD_ORDER)
        horizon = _horizon(row["horizon"])
        phase = row["phase"]
        if phase not in PHASES:
            raise ValueError("phase must be bootstrap or new_visit")
        if method == "full_history_rescene" and phase == "bootstrap":
            raise ValueError("full_history_rescene cannot have bootstrap latency")
        latency = _finite(row["latency_ms"], name="latency_ms")
        if latency < 0:
            raise ValueError("latency_ms must be >= 0")
        key = (method, horizon, phase)
        if key in seen:
            raise ValueError("duplicate primary key")
        seen.add(key)
        records.append(
            {
                "method_id": method,
                "horizon": horizon,
                "phase": phase,
                "latency_ms": latency,
            }
        )
    expected = {
        ("b4", horizon, phase)
        for horizon in HORIZONS
        for phase in PHASES
    }
    expected.update(
        ("full_history_rescene", horizon, "new_visit") for horizon in HORIZONS
    )
    missing = expected - seen
    if missing:
        raise ValueError("missing latency method/horizon phase group")
    return sorted(
        records,
        key=lambda record: (
            LATENCY_METHOD_ORDER.index(record["method_id"]),
            record["horizon"],
            PHASES.index(record["phase"]),
        ),
    )


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    return format(float(value), ".12g")


def _esc(value: Any) -> str:
    return escape(str(value), quote=True)


def _text(x: float, y: float, value: Any, *, size: int = 13, anchor: str = "start") -> str:
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{size}" '
        f'text-anchor="{_esc(anchor)}">{_esc(value)}</text>'
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = "#333333",
    width: float = 1.0,
    dash: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" '
        f'y2="{_fmt(y2)}" stroke="{stroke}" stroke-width="{_fmt(width)}"'
        f'{dash_attr}/>'
    )


def _marker(x: float, y: float, method: str, *, size: float = 4.5) -> str:
    color = _PALETTE[method]
    shape = _MARKERS[method]
    if shape == "circle":
        return f'<circle cx="{_fmt(x)}" cy="{_fmt(y)}" r="{_fmt(size)}" fill="{color}"/>'
    if shape == "square":
        origin = size
        return f'<rect x="{_fmt(x - origin)}" y="{_fmt(y - origin)}" width="{_fmt(2 * origin)}" height="{_fmt(2 * origin)}" fill="{color}"/>'
    if shape == "triangle":
        return f'<path d="M {_fmt(x)} {_fmt(y - size)} L {_fmt(x + size)} {_fmt(y + size)} L {_fmt(x - size)} {_fmt(y + size)} Z" fill="{color}"/>'
    if shape == "diamond":
        return f'<path d="M {_fmt(x)} {_fmt(y - size)} L {_fmt(x + size)} {_fmt(y)} L {_fmt(x)} {_fmt(y + size)} L {_fmt(x - size)} {_fmt(y)} Z" fill="{color}"/>'
    if shape == "cross":
        return _line(x - size, y - size, x + size, y + size, stroke=color, width=2) + _line(x - size, y + size, x + size, y - size, stroke=color, width=2)
    if shape == "plus":
        return _line(x - size, y, x + size, y, stroke=color, width=2) + _line(x, y - size, x, y + size, stroke=color, width=2)
    return (
        _line(x - size, y, x + size, y, stroke=color, width=2)
        + _line(x, y - size, x, y + size, stroke=color, width=2)
        + _line(x - size * 0.7, y - size * 0.7, x + size * 0.7, y + size * 0.7, stroke=color, width=1.5)
        + _line(x - size * 0.7, y + size * 0.7, x + size * 0.7, y - size * 0.7, stroke=color, width=1.5)
    )


def _svg_start(letter: str, title: str, description: str) -> list[str]:
    title_id = f"figure-{letter.lower()}-title"
    desc_id = f"figure-{letter.lower()}-description"
    return [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-labelledby="{title_id} {desc_id}" viewBox="0 0 960 600">'
        ),
        '<title id="' + title_id + '">' + _esc(title) + "</title>",
        '<desc id="' + desc_id + '">' + _esc(description) + "</desc>",
        '<rect x="0" y="0" width="960" height="600" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#222222">',
    ]


def _svg_end(parts: list[str]) -> str:
    parts.extend(("</g>", "</svg>"))
    return "".join(parts)


def _legend(
    parts: list[str],
    *,
    x: float = 680,
    y: float = 82,
    phase: bool = False,
    methods: Sequence[str] = METHOD_ORDER,
) -> None:
    for method_index, method in enumerate(methods):
        row_y = y + method_index * 25
        parts.append(
            _line(
                x,
                row_y,
                x + 28,
                row_y,
                stroke=_PALETTE[method],
                width=2,
                dash=_LINE_STYLES[method],
            )
        )
        parts.append(_marker(x + 14, row_y, method, size=3.5))
        parts.append(_text(x + 38, row_y + 5, method))
    if phase:
        phase_y = y + len(METHOD_ORDER) * 25 + 12
        parts.append(_line(x, phase_y, x + 28, phase_y, stroke="#444444", dash="6 3"))
        parts.append(_text(x + 38, phase_y + 5, "bootstrap"))
        parts.append(_line(x, phase_y + 22, x + 28, phase_y + 22, stroke="#444444"))
        parts.append(_text(x + 38, phase_y + 27, "new_visit"))


def _x_positions(horizons: Sequence[int]) -> dict[int, float]:
    if not horizons:
        return {}
    if len(horizons) == 1:
        return {horizons[0]: 340.0}
    return {
        horizon: 80.0 + index * (590.0 - 80.0) / (len(horizons) - 1)
        for index, horizon in enumerate(horizons)
    }


def _axes(
    parts: list[str],
    *,
    y_label: str,
    x_label: str = "Horizon (T)",
    horizons: Sequence[int] = HORIZONS,
) -> None:
    parts.append(_line(70, 500, 610, 500, stroke="#333333"))
    parts.append(_line(70, 70, 70, 500, stroke="#333333"))
    for horizon, x in _x_positions(horizons).items():
        parts.append(_line(x, 500, x, 506, stroke="#333333"))
        parts.append(_text(x, 527, horizon, anchor="middle"))
    parts.append(_text(340, 566, x_label, anchor="middle"))
    parts.append(
        '<text x="20" y="290" font-size="13" text-anchor="middle" '
        'transform="rotate(-90 20 290)">' + _esc(y_label) + "</text>"
    )


def _scale(values: Sequence[float], low: float = 0.0, high: float = 1.0) -> Callable[[float], float]:
    minimum = min(values, default=0.0)
    maximum = max(values, default=1.0)
    if math.isclose(minimum, maximum, abs_tol=1e-15, rel_tol=0.0):
        pad = max(abs(minimum) * 0.1, 1.0)
        minimum -= pad
        maximum += pad
    return lambda value: high - (float(value) - minimum) / (maximum - minimum) * (high - low)


def _series_points(records: Sequence[Mapping[str, Any]], value_name: str) -> dict[str, list[tuple[int, float]]]:
    grouped = {method: [] for method in METHOD_ORDER}
    for record in records:
        grouped[record["method_id"]].append(
            (record["horizon"], float(record[value_name]))
        )
    return {method: sorted(values) for method, values in grouped.items()}


def _render_line_figure(
    letter: str,
    title: str,
    description: str,
    records: Sequence[Mapping[str, Any]],
    value_name: str,
    y_label: str,
) -> str:
    parts = _svg_start(letter, title, description)
    _axes(parts, y_label=y_label)
    grouped = _series_points(records, value_name)
    methods = [method for method in METHOD_ORDER if grouped[method]]
    scale = _scale([value for values in grouped.values() for _, value in values])
    x_by_horizon = _x_positions(HORIZONS)
    for method in methods:
        points = [(x_by_horizon[horizon], scale(value)) for horizon, value in grouped[method]]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {_fmt(x)} {_fmt(y)}"
            for index, (x, y) in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{_PALETTE[method]}" '
            f'stroke-width="2" stroke-dasharray="{_LINE_STYLES[method]}"/>'
        )
        for x, y in points:
            parts.append(_marker(x, y, method))
    _legend(parts, methods=methods)
    return _svg_end(parts)


def render_figure_a_identity(rows: Iterable[Mapping[str, Any]]) -> str:
    records = _base_records(rows, _SCHEMA_A, "id_switch_rate")
    return _render_line_figure(
        "A",
        "Figure A: Identity & ID-Switch Rate",
        "Identity switch rate by association method and horizon.",
        records,
        "id_switch_rate",
        "ID-switch rate (fraction)",
    )


def render_figure_b_online_tmap(rows: Iterable[Mapping[str, Any]]) -> str:
    records = _base_records(rows, _SCHEMA_B, "online_t_mAP")
    return _render_line_figure(
        "B",
        "Figure B: Online t-mAP",
        "Strict-online temporal mean average precision by method and horizon.",
        records,
        "online_t_mAP",
        "Online t-mAP",
    )


def render_figure_c_reactivation(rows: Iterable[Mapping[str, Any]]) -> str:
    records = _c_records(rows)
    parts = _svg_start(
        "C",
        "Figure C: Reactivation & Score-Bin Distributions",
        "Reactivation outcome distributions over score bins.",
    )
    _axes(
        parts,
        y_label="Fraction",
        x_label="Horizon (T)",
        horizons=REACTIVATION_HORIZONS,
    )
    panels = {
        ("correct", "best_score"): (80, 92),
        ("correct", "score_margin"): (370, 92),
        ("wrong", "best_score"): (80, 320),
        ("wrong", "score_margin"): (370, 320),
    }
    group_keys = {
        (
            record["method_id"],
            record["horizon"],
            record["outcome"],
            record["metric"],
        )
        for record in records
    }
    for (outcome, metric), (panel_x, panel_y) in panels.items():
        parts.append(_text(panel_x, panel_y, f"{outcome} / {metric}", size=12))
        for method_index, method in enumerate(REACTIVATION_METHOD_ORDER):
            present = any(
                (method, horizon, outcome, metric) in group_keys
                for horizon in REACTIVATION_HORIZONS
            )
            if not present:
                continue
            parts.append(_text(panel_x - 8, panel_y + 28 + method_index * 27, method, size=8, anchor="end"))
            for horizon_index, horizon in enumerate(REACTIVATION_HORIZONS):
                group = [
                    record
                    for record in records
                    if record["method_id"] == method
                    and record["horizon"] == horizon
                    and record["outcome"] == outcome
                    and record["metric"] == metric
                ]
                if not group:
                    continue
                group = sorted(group, key=lambda record: (record["bin_low"], record["bin_high"]))
                cell_x = panel_x + horizon_index * 67
                cell_y = panel_y + 18 + method_index * 27
                width = 58 / len(group)
                for bin_index, record in enumerate(group):
                    height = float(record["fraction"]) * 19
                    parts.append(
                        f'<rect x="{_fmt(cell_x + bin_index * width)}" y="{_fmt(cell_y - height)}" '
                        f'width="{_fmt(width - 1)}" height="{_fmt(height)}" '
                        f'fill="{_PALETTE[method]}" stroke="#333333" stroke-width="0.4"/>'
                    )
                parts.append(_text(cell_x - 3, cell_y + 11, horizon, size=8, anchor="end"))
    _legend(parts, x=680, y=92, methods=REACTIVATION_METHOD_ORDER)
    return _svg_end(parts)


def render_figure_d_failures(rows: Iterable[Mapping[str, Any]]) -> str:
    records = _d_records(rows)
    parts = _svg_start(
        "D",
        "Figure D: Failure Composition",
        "Exclusive F1 through F7 and unclassified failure composition by method and horizon.",
    )
    parts.append(_text(360, 38, "Horizon (T)", anchor="middle"))
    parts.append(_text(70, 38, "Method", anchor="start"))
    parts.append(_text(390, 568, "Failure share (stacked composition)", anchor="middle"))
    group_map = {
        (record["method_id"], record["horizon"], record["category"]): record
        for record in records
    }
    group_keys = sorted(
        {(record["method_id"], record["horizon"]) for record in records},
        key=lambda key: (METHOD_ORDER.index(key[0]), key[1]),
    )
    for group_index, (method, horizon) in enumerate(group_keys):
        y = 70 + group_index * 15
        parts.append(_text(70, y + 5, f"{method} / T{horizon}", size=9, anchor="end"))
        x = 100.0
        for category_index, category in enumerate(FAILURE_CATEGORIES):
            record = group_map[(method, horizon, category)]
            width = float(record["share"]) * 490
            dash = _CATEGORY_DASHES[category_index]
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(
                f'<rect x="{_fmt(x)}" y="{_fmt(y - 8)}" width="{_fmt(width)}" height="10" '
                f'fill="{_CATEGORY_FILLS[category_index]}" stroke="#222222" stroke-width="0.5"{dash_attr}/>'
            )
            x += width
    for category_index, category in enumerate(FAILURE_CATEGORIES):
        x = 100 + category_index * 73
        parts.append(
            f'<rect x="{_fmt(x)}" y="{_fmt(550)}" width="12" height="10" '
            f'fill="{_CATEGORY_FILLS[category_index]}" stroke="#222222" stroke-width="0.5"/>'
        )
        parts.append(_text(x + 17, 560, category, size=10))
    return _svg_end(parts)


def render_figure_e_latency(rows: Iterable[Mapping[str, Any]]) -> str:
    records = _e_records(rows)
    parts = _svg_start(
        "E",
        "Figure E: Latency Trends",
        "Bootstrap and per-new-visit latency trends by method and horizon.",
    )
    _axes(parts, y_label="Latency (ms)")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {
        (method, phase): [] for method in LATENCY_METHOD_ORDER for phase in PHASES
    }
    for record in records:
        grouped[(record["method_id"], record["phase"])].append(
            (record["horizon"], float(record["latency_ms"]))
        )
    scale = _scale([value for values in grouped.values() for _, value in values])
    x_by_horizon = _x_positions(HORIZONS)
    for method in LATENCY_METHOD_ORDER:
        for phase in PHASES:
            points = [
                (x_by_horizon[horizon], scale(value))
                for horizon, value in sorted(grouped[(method, phase)])
            ]
            if not points:
                continue
            path = " ".join(
                ("M" if index == 0 else "L") + f" {_fmt(x)} {_fmt(y)}"
                for index, (x, y) in enumerate(points)
            )
            dash = _LINE_STYLES[method]
            if phase == "bootstrap":
                dash = "6 3" if not dash else dash
            parts.append(
                f'<path d="{path}" fill="none" stroke="{_PALETTE[method]}" '
                f'stroke-width="2" stroke-dasharray="{dash}"/>'
            )
            for x, y in points:
                parts.append(_marker(x, y, method, size=3.5 if phase == "new_visit" else 2.8))
    _legend(parts, phase=True, methods=LATENCY_METHOD_ORDER)
    return _svg_end(parts)


__all__ = [
    "render_figure_a_identity",
    "render_figure_b_online_tmap",
    "render_figure_c_reactivation",
    "render_figure_d_failures",
    "render_figure_e_latency",
]
