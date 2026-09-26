"""The fixed V2 development gates; unavailable coverage never becomes a score."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import cmp_to_key

TOLERANCE = 1e-6
HORIZONS = (2, 3, 4, 5)


def metrics(candidate: Mapping) -> dict[int, float] | None:
    if candidate.get("coverage_status") != "COMPLETE":
        return None
    values = candidate.get("metrics", {})
    try:
        result = {
            t: float(values[t] if t in values else values[str(t)]) for t in HORIZONS
        }
    except (KeyError, TypeError, ValueError):
        return None
    return (
        result
        if all(math.isfinite(v) and 0 <= v <= 1 for v in result.values())
        else None
    )


def compare(candidate: Mapping, baseline: Mapping) -> dict:
    left, right = metrics(candidate), metrics(baseline)
    if left is None or right is None:
        return {
            **candidate,
            "comparison_status": "INCOMPLETE_COVERAGE",
            "deltas": None,
            "S_mean": None,
            "S_long": None,
            "S_min": None,
        }
    delta = {t: left[t] - right[t] for t in HORIZONS}
    return {
        **candidate,
        "comparison_status": "COMPLETE",
        "deltas": delta,
        "S_mean": sum(delta.values()) / 4,
        "S_long": (delta[4] + delta[5]) / 2,
        "S_min": min(delta.values()),
    }


def _compare_rank(left: Mapping, right: Mapping) -> int:
    for field in ("S_mean", "S_long", "S_min"):
        difference = left[field] - right[field]
        if abs(difference) > TOLERANCE:
            return -1 if difference > 0 else 1
    for field in ("new_parameter_count", "tie_update"):
        a = left.get(
            field, left.get("optimizer_update", 0) if field == "tie_update" else 0
        )
        b = right.get(
            field, right.get("optimizer_update", 0) if field == "tie_update" else 0
        )
        if a != b:
            return -1 if a < b else 1
    a, b = str(left["method_id"]), str(right["method_id"])
    return (a > b) - (a < b)


def rank(candidates: Sequence[Mapping], baseline: Mapping) -> list[dict]:
    rows = [compare(value, baseline) for value in candidates]
    return sorted(
        (row for row in rows if row["comparison_status"] == "COMPLETE"),
        key=cmp_to_key(_compare_rank),
    )


def gate(
    row: Mapping, *, mean: float, minimum: float, long: float | None = 0.0
) -> bool:
    return (
        row.get("comparison_status") == "COMPLETE"
        and row["S_mean"] >= mean - TOLERANCE
        and row["S_min"] >= minimum - TOLERANCE
        and (long is None or row["S_long"] >= long - TOLERANCE)
    )


def select_repair(*, parent: Mapping, new: Mapping, pair: Mapping) -> dict:
    new_parent, pair_parent, pair_new = (
        compare(new, parent),
        compare(pair, parent),
        compare(pair, new),
    )
    new_ok = gate(new_parent, mean=0.003, minimum=-0.001)
    pair_ok = gate(pair_parent, mean=0.003, minimum=-0.001)
    extra_ok = gate(pair_new, mean=0.001, minimum=-0.002)
    mode = "PAIR" if pair_ok and extra_ok else "NEW" if new_ok else "KEEP_PARENT"
    return {
        "selected_mode": mode,
        "base_positive": new_ok or pair_ok,
        "history_evidence_supported": pair_ok and extra_ok,
        "NEW-parent": new_parent,
        "PAIR-parent": pair_parent,
        "PAIR-NEW": pair_new,
    }


def strict_all_t(candidate: Mapping, baseline: Mapping) -> bool | None:
    compared = compare(candidate, baseline)
    if compared["comparison_status"] != "COMPLETE":
        return None
    return all(value > TOLERANCE for value in compared["deltas"].values())


def select_learning_rate(
    probes: Mapping[str, Sequence[Mapping]], baseline: Mapping, *, comparable_high: bool
) -> dict:
    selected = {}
    for label in ("H", "L"):
        ranked = rank(
            [
                row
                for row in probes.get(label, ())
                if row["optimizer_update"] in {250, 750}
            ],
            baseline,
        )
        selected[label] = ranked[0] if ranked else None
    choice = "L"
    if comparable_high and selected["H"] is not None and selected["L"] is not None:
        for field in ("S_mean", "S_long", "S_min"):
            difference = selected["H"][field] - selected["L"][field]
            if abs(difference) > TOLERANCE:
                choice = "H" if difference > 0 else "L"
                break
    return {
        "learning_rate_label": choice,
        "probes": selected,
        "high_comparable": comparable_high,
        "tie_preference": "L",
    }


def select_full_promotion(
    candidates: Sequence[Mapping],
    baseline: Mapping,
    *,
    control_by_method: Mapping[str, str],
) -> dict:
    grouped = {}
    for row in candidates:
        grouped.setdefault(row["method_id"], []).append(row)
    best = {
        name: rows[0]
        for name, values in grouped.items()
        if (rows := rank(values, baseline))
    }
    controls = set(control_by_method.values())
    missing_controls = {
        name: control
        for name, control in control_by_method.items()
        if name in best and control not in best
    }
    new = rank(
        [
            row
            for name, row in best.items()
            if name in control_by_method and name not in missing_controls
        ],
        baseline,
    )
    eligible = [row for row in new if gate(row, mean=-0.01, minimum=-0.02, long=None)]
    chosen, reason = (eligible[0], "PILOT_GATE") if eligible else (None, "NO_NEW_ARM")
    if chosen is None and new and new[0]["S_mean"] >= -0.03 - TOLERANCE:
        by_step = {
            row["optimizer_update"]: compare(row, baseline)
            for row in grouped[new[0]["method_id"]]
        }
        if all(
            step in by_step and by_step[step]["comparison_status"] == "COMPLETE"
            for step in (250, 750)
        ) and by_step[750]["S_mean"] >= by_step[250]["S_mean"] - 0.005 - TOLERANCE:
            chosen, reason = new[0], "EXPLORATORY_FALLBACK"
    controls_ranked = rank(
        [row for name, row in best.items() if name in controls], baseline
    )
    c0 = next(
        (row for row in controls_ranked if gate(row, mean=0.005, minimum=-0.001)), None
    )
    if chosen is not None:
        methods = [control_by_method[chosen["method_id"]], chosen["method_id"]]
    elif c0 is not None and gate(c0, mean=0.005, minimum=-0.001):
        methods, reason = [c0["method_id"]], "C0_ONLY_GATE"
    else:
        methods = []
    return {
        "full_training_recipes": methods,
        "reason": reason,
        "pilot_best": best,
        "missing_controls": missing_controls,
        "resume_update": 750,
        "endpoint": 3000,
    }


def select_perception(
    candidates: Sequence[Mapping], baseline: Mapping, *, c0_id: str
) -> dict:
    ranked = rank(candidates, baseline)
    c0 = next((row for row in ranked if row["method_id"] == c0_id), None)
    eligible = []
    for row in ranked:
        if (
            row["method_id"] != c0_id
            and c0 is not None
            and gate(row, mean=0.005, minimum=-0.001)
        ):
            control = compare(row, c0)
            if gate(control, mean=0.002, minimum=-0.002, long=None):
                eligible.append({**row, "versus_c0": control})
    selected = (
        eligible[0]
        if eligible
        else (
            c0
            if c0 is not None and gate(c0, mean=0.005, minimum=-0.001)
            else dict(baseline)
        )
    )
    return {
        "selected": selected,
        "evaluated": ranked,
        "fallback": selected["method_id"] == baseline["method_id"],
    }


def select_final(candidates: Sequence[Mapping], baseline: Mapping) -> dict:
    if len(candidates) > 6:
        raise ValueError("V2 finite final candidate inventory exceeds six")
    ranked = rank(candidates, baseline)
    eligible = [
        row
        for row in ranked
        if row.get("component_gate_passed", False)
        and gate(row, mean=0.003, minimum=-0.001)
    ]
    chosen = eligible[0] if eligible else compare(baseline, baseline)
    size = (
        "TARGET_REACHED"
        if chosen["S_mean"] is not None
        and chosen["S_mean"] >= 0.005 - TOLERANCE
        and chosen["S_long"] >= 0.01 - TOLERANCE
        else "MODEST_GAIN" if eligible else "NO_CONFIRMED_GAIN"
    )
    return {"selected": chosen, "ranked": ranked, "development_result_size": size}
