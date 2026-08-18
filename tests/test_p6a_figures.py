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

BASELINE_METHODS = ("b0", "b0_sanity", "b1", "b2", "b3", "b4")
METHODS = BASELINE_METHODS + ("oracle",)
C_METHODS = ("b1", "b2", "b3", "b4")
C_HORIZONS = (3, 4, 5)
HORIZONS = (2, 3, 4, 5)
E_METHODS = ("b4", "full_history_rescene")
FAILURE_CATEGORIES = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")


def _rows_a(*, include_oracle: bool = True) -> list[dict[str, object]]:
    methods = METHODS if include_oracle else BASELINE_METHODS
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "id_switch_rate": 0.01 * (index + horizon),
        }
        for index, method in enumerate(methods)
        for horizon in HORIZONS
    ]


def _rows_b(*, include_oracle: bool = True) -> list[dict[str, object]]:
    methods = METHODS if include_oracle else BASELINE_METHODS
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "online_t_mAP": 0.1 + 0.01 * (index + horizon),
        }
        for index, method in enumerate(methods)
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
            "count": 2 if bin_index == 0 else 1,
            "fraction": 2 / 3 if bin_index == 0 else 1 / 3,
        }
        for method, horizon, outcome, metric, bin_index in product(
            C_METHODS,
            C_HORIZONS,
            ("correct", "wrong"),
            ("best_score", "score_margin"),
            (0, 1),
        )
    ]


def _rows_d() -> list[dict[str, object]]:
    return [
        {
            "method_id": "b4",
            "horizon": horizon,
            "category": category,
            "count": category_index,
            "share": 1 / len(FAILURE_CATEGORIES),
        }
        for horizon, (category_index, category) in product(
            HORIZONS, enumerate(FAILURE_CATEGORIES, start=1)
        )
    ]


def _rows_d_group(method: str, horizon: int) -> list[dict[str, object]]:
    return [
        {
            "method_id": method,
            "horizon": horizon,
            "category": category,
            "count": category_index,
            "share": 1 / len(FAILURE_CATEGORIES),
        }
        for category_index, category in enumerate(FAILURE_CATEGORIES, start=1)
    ]


def _rows_e() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for phase in ("bootstrap", "new_visit"):
            rows.append(
                {
                    "method_id": "b4",
                    "horizon": horizon,
                    "phase": phase,
                    "latency_ms": float(horizon),
                }
            )
        rows.append(
            {
                "method_id": "full_history_rescene",
                "horizon": horizon,
                "phase": "new_visit",
                "latency_ms": float(horizon + 1),
            }
        )
    return rows


@pytest.mark.parametrize(
    ("renderer", "rows", "title", "methods"),
    [
        (render_figure_a_identity, _rows_a, "Figure A", METHODS),
        (render_figure_b_online_tmap, _rows_b, "Figure B", METHODS),
        (render_figure_c_reactivation, _rows_c, "Figure C", C_METHODS),
        (render_figure_d_failures, _rows_d, "Figure D", ("b4",)),
        (render_figure_e_latency, _rows_e, "Figure E", E_METHODS),
    ],
)
def test_renderers_return_accessible_fixed_svg_and_are_order_independent(
    renderer, rows, title: str, methods: tuple[str, ...]
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
    assert all(method in rendered for method in methods)
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


def test_line_renderers_require_base_grid_but_oracle_is_optional_and_complete() -> None:
    assert render_figure_a_identity(_rows_a(include_oracle=False))
    assert render_figure_b_online_tmap(_rows_b(include_oracle=False))

    rows = _rows_a(include_oracle=False)
    rows.append({"method_id": "oracle", "horizon": 2, "id_switch_rate": 0.1})
    with pytest.raises(ValueError, match="missing"):
        render_figure_a_identity(rows)


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


def test_line_renderers_reject_missing_required_groups() -> None:
    rows = _rows_b(include_oracle=False)
    rows = [row for row in rows if not (row["method_id"] == "b3" and row["horizon"] == 4)]
    with pytest.raises(ValueError, match="missing"):
        render_figure_b_online_tmap(rows)

    rows = _rows_e()
    rows = [
        row
        for row in rows
        if not (
            row["method_id"] == "full_history_rescene"
            and row["horizon"] == 4
        )
    ]
    with pytest.raises(ValueError, match="missing"):
        render_figure_e_latency(rows)


def test_reactivation_uses_only_measured_methods_and_horizons() -> None:
    rendered = render_figure_c_reactivation(_rows_c())
    assert "b0" not in rendered
    assert "T2" not in rendered
    assert "N/A" not in rendered


def test_reactivation_requires_paired_correct_and_wrong_groups() -> None:
    rows = [
        row
        for row in _rows_c()
        if not (
            row["method_id"] == "b4"
            and row["horizon"] == 5
            and row["outcome"] == "wrong"
            and row["metric"] == "score_margin"
        )
    ]
    with pytest.raises(ValueError, match="paired|outcome"):
        render_figure_c_reactivation(rows)


def test_reactivation_figure_preserves_an_empty_outcome_group() -> None:
    rows = _rows_c()
    for row in rows:
        if row["outcome"] == "wrong":
            row["count"] = 0
            row["fraction"] = 0.0

    rendered = render_figure_c_reactivation(rows)

    assert "wrong / best_score" in rendered


@pytest.mark.parametrize("bad_low", (0.6, 0.4))
def test_reactivation_bins_must_be_continuous_without_gap_or_overlap(bad_low: float) -> None:
    rows = _rows_c()
    target = next(
        row
        for row in rows
        if row["method_id"] == "b1"
        and row["horizon"] == 3
        and row["outcome"] == "correct"
        and row["metric"] == "best_score"
        and row["bin_low"] == 0.5
    )
    target["bin_low"] = bad_low
    with pytest.raises(ValueError, match="bin"):
        render_figure_c_reactivation(rows)


def test_reactivation_fractions_and_counts_must_have_positive_closed_mass() -> None:
    rows = _rows_c()
    for row in rows:
        if (
            row["method_id"] == "b1"
            and row["horizon"] == 3
            and row["outcome"] == "correct"
            and row["metric"] == "best_score"
        ):
            row["count"] = 0
    with pytest.raises(ValueError, match="count"):
        render_figure_c_reactivation(rows)

    rows = _rows_c()
    rows[0]["fraction"] = 0.5
    with pytest.raises(ValueError, match="fraction"):
        render_figure_c_reactivation(rows)


def test_reactivation_rejects_unsupported_method_or_horizon_and_duplicate_bin_key() -> None:
    rows = _rows_c()
    rows[0]["method_id"] = "b0"
    with pytest.raises(ValueError, match="method"):
        render_figure_c_reactivation(rows)

    rows = _rows_c()
    rows[0]["horizon"] = 2
    with pytest.raises(ValueError, match="horizon"):
        render_figure_c_reactivation(rows)

    rows = _rows_c()
    rows[-1] = dict(rows[-2])
    with pytest.raises(ValueError, match="duplicate"):
        render_figure_c_reactivation(rows)


def test_failure_composition_allows_partial_methods_but_requires_b4_t2_to_t5() -> None:
    rows = _rows_d() + _rows_d_group("b2", 3)
    rendered = render_figure_d_failures(rows)
    assert "b4 / T2" in rendered
    assert "b2 / T3" in rendered
    assert "b2 / T2" not in rendered

    rows = [row for row in rows if not (row["method_id"] == "b4" and row["horizon"] == 5)]
    with pytest.raises(ValueError, match="b4|missing"):
        render_figure_d_failures(rows)


def test_failure_shares_must_close_and_categories_must_be_complete() -> None:
    rows = _rows_d()
    rows[0]["share"] = 0.2
    with pytest.raises(ValueError, match="share"):
        render_figure_d_failures(rows)

    rows = _rows_d()
    rows = [
        row
        for row in rows
        if not (row["method_id"] == "b4" and row["horizon"] == 2 and row["category"] == "F7")
    ]
    with pytest.raises(ValueError, match="category|missing"):
        render_figure_d_failures(rows)


def test_latency_requires_b4_phases_and_full_history_new_visit_only() -> None:
    rendered = render_figure_e_latency(_rows_e())
    assert "full_history_rescene" in rendered

    rows = _rows_e()
    rows.append(
        {
            "method_id": "full_history_rescene",
            "horizon": 2,
            "phase": "bootstrap",
            "latency_ms": 1.0,
        }
    )
    with pytest.raises(ValueError, match="bootstrap|phase"):
        render_figure_e_latency(rows)


def test_latency_rejects_legacy_methods() -> None:
    rows = _rows_e()
    rows[0]["method_id"] = "b0"
    with pytest.raises(ValueError, match="method"):
        render_figure_e_latency(rows)


def test_svg_text_is_escaped_by_renderer_helpers() -> None:
    rows = _rows_a()
    rendered = render_figure_a_identity(rows)
    assert "&lt;" not in rendered
    assert "&amp;" in rendered
