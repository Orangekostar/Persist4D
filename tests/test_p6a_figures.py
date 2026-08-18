from __future__ import annotations

from itertools import product

import pytest

from scripts.p6a_figures import (
    render_figure_a_identity,
    render_figure_b_online_tmap,
    render_figure_c_reactivation,
    render_figure_d_failures,
    render_figure_e_latency,
)

METHODS = ("b0", "b0_sanity", "b1", "b2", "b3", "b4", "oracle")
HORIZONS = (2, 3, 4, 5)


def _rows_a() -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "id_switch_rate": 0.01 * (index + horizon),
        }
        for index, method in enumerate(METHODS)
        for horizon in HORIZONS
    ]


def _rows_b() -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "online_t_mAP": 0.1 + 0.01 * (index + horizon),
        }
        for index, method in enumerate(METHODS)
        for horizon in HORIZONS
    ]


def _rows_c() -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "outcome": outcome,
            "metric": metric,
            "bin_low": float(bin_index) / 2,
            "bin_high": float(bin_index + 1) / 2,
            "count": 10 if bin_index == 0 else 5,
            "fraction": 2 / 3 if bin_index == 0 else 1 / 3,
        }
        for method, horizon, outcome, metric, bin_index in product(
            METHODS,
            HORIZONS,
            ("correct", "wrong"),
            ("best_score", "score_margin"),
            (0, 1),
        )
    ]


def _rows_d() -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "category": f"F{category}",
            "count": category,
            "share": 1 / 7,
        }
        for method, horizon, category in product(METHODS, HORIZONS, range(1, 8))
    ]


def _rows_e() -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "phase": phase,
            "latency_ms": float(index + horizon),
        }
        for index, method in enumerate(METHODS)
        for horizon, phase in product(HORIZONS, ("bootstrap", "new_visit"))
    ]


@pytest.mark.parametrize(
    ("renderer", "rows", "title"),
    [
        (render_figure_a_identity, _rows_a, "Figure A"),
        (render_figure_b_online_tmap, _rows_b, "Figure B"),
        (render_figure_c_reactivation, _rows_c, "Figure C"),
        (render_figure_d_failures, _rows_d, "Figure D"),
        (render_figure_e_latency, _rows_e, "Figure E"),
    ],
)
def test_renderers_return_accessible_fixed_svg_and_are_order_independent(
    renderer, rows, title: str
) -> None:
    source = rows()
    rendered = renderer(source)
    reversed_rendered = renderer(list(reversed(source)))

    assert rendered == reversed_rendered
    assert rendered.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert 'viewBox="0 0 960 600"' in rendered
    assert '<rect x="0" y="0" width="960" height="600" fill="#FFFFFF"/>' in rendered
    assert 'role="img"' in rendered
    assert "aria-labelledby=" in rendered
    assert f">{title}:" in rendered
    assert "Horizon" in rendered
    assert all(method in rendered for method in METHODS)
    assert "/home/" not in rendered
    assert "timestamp" not in rendered.lower()


@pytest.mark.parametrize(
    "renderer_rows",
    [
        (render_figure_a_identity, _rows_a),
        (render_figure_b_online_tmap, _rows_b),
        (render_figure_c_reactivation, _rows_c),
        (render_figure_d_failures, _rows_d),
        (render_figure_e_latency, _rows_e),
    ],
)
def test_renderers_reject_empty_missing_and_extra_columns(renderer_rows) -> None:
    renderer, rows_factory = renderer_rows
    rows = rows_factory()

    with pytest.raises(ValueError, match="empty"):
        renderer([])

    missing = dict(rows[0])
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="columns"):
        renderer([missing])

    extra = dict(rows[0])
    extra["unexpected"] = 1
    with pytest.raises(ValueError, match="columns"):
        renderer([extra])


@pytest.mark.parametrize(
    ("renderer", "rows_factory", "field", "value"),
    [
        (render_figure_a_identity, _rows_a, "id_switch_rate", float("nan")),
        (render_figure_b_online_tmap, _rows_b, "online_t_mAP", float("inf")),
        (render_figure_e_latency, _rows_e, "latency_ms", -1.0),
    ],
)
def test_line_renderers_reject_nonfinite_or_invalid_values(
    renderer, rows_factory, field: str, value: float
) -> None:
    rows = rows_factory()
    rows[0][field] = value
    with pytest.raises(ValueError):
        renderer(rows)


def test_renderers_reject_unknown_method_invalid_horizon_and_duplicate_keys() -> None:
    rows = _rows_a()
    rows[0]["method_id"] = "new_method"
    with pytest.raises(ValueError, match="method"):
        render_figure_a_identity(rows)

    rows = _rows_a()
    rows[0]["horizon"] = 6
    with pytest.raises(ValueError, match="horizon"):
        render_figure_a_identity(rows)

    rows = _rows_a()
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        render_figure_a_identity(rows)


def test_line_renderers_reject_missing_method_horizon_groups() -> None:
    rows = _rows_b()
    rows = [row for row in rows if not (row["method_id"] == "oracle" and row["horizon"] == 5)]
    with pytest.raises(ValueError, match="missing"):
        render_figure_b_online_tmap(rows)

    rows = _rows_e()
    rows = [
        row
        for row in rows
        if not (
            row["method_id"] == "b3"
            and row["horizon"] == 4
            and row["phase"] == "bootstrap"
        )
    ]
    with pytest.raises(ValueError, match="missing"):
        render_figure_e_latency(rows)


def test_reactivation_bins_and_failure_shares_must_close() -> None:
    rows = _rows_c()
    rows[0]["fraction"] = 0.5
    with pytest.raises(ValueError, match="fraction"):
        render_figure_c_reactivation(rows)

    rows = _rows_d()
    rows[0]["share"] = 0.2
    with pytest.raises(ValueError, match="share"):
        render_figure_d_failures(rows)


def test_reactivation_rejects_missing_bin_group_and_duplicate_bin_key() -> None:
    rows = _rows_c()
    rows = [
        row
        for row in rows
        if not (
            row["method_id"] == "b0"
            and row["horizon"] == 2
            and row["outcome"] == "correct"
            and row["metric"] == "best_score"
            and row["bin_low"] == 0.5
        )
    ]
    with pytest.raises(ValueError, match="bin|fraction"):
        render_figure_c_reactivation(rows)

    rows = _rows_c()
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError, match="duplicate"):
        render_figure_c_reactivation(rows)


def test_svg_text_is_escaped_by_renderer_helpers() -> None:
    rows = _rows_a()
    rendered = render_figure_a_identity(rows)
    assert "&lt;" not in rendered
    assert "&amp;" in rendered
