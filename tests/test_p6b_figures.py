from __future__ import annotations

import pytest

from scripts.p6b_figures import render_identity_figure, render_reactivation_figure


def _results() -> list[dict[str, object]]:
    return [
        {"method": method, "T": horizon, "identity_switch_rate": rate, "reactivation_accuracy": accuracy}
        for method, delta in (("B4", 0.0), ("P6B", -0.01))
        for horizon, rate, accuracy in (
            ("T2", 0.10 + delta, None),
            ("T3", 0.11 + delta, 0.75),
            ("T4", 0.12 + delta, 0.76),
            ("T5", 0.13 + delta, 0.77),
        )
    ]


def test_figures_are_deterministic_portable_svg() -> None:
    first = render_identity_figure(_results())
    second = render_identity_figure(_results())
    reactivation = render_reactivation_figure(_results())

    assert first == second
    assert first.startswith("<svg") and first.endswith("</svg>\n")
    assert "B4" in first and "P6B" in first
    assert "React. accuracy" in reactivation
    assert "/home/" not in first and "/mnt/" not in reactivation


def test_figure_rejects_missing_method_horizon_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="exact"):
        render_identity_figure(_results()[:-1])
    invalid = _results()
    invalid[0] = {**invalid[0], "identity_switch_rate": float("nan")}
    with pytest.raises(ValueError, match="finite"):
        render_identity_figure(invalid)
