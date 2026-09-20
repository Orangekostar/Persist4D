"""Final CrossWindow quality and resource gates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


class CrossWindowProfileError(ValueError):
    """Raised when final quality or profiling evidence is malformed."""


def _tmap(values: Mapping[int, float], *, name: str) -> dict[int, float]:
    if set(values) != {2, 3, 4, 5}:
        raise CrossWindowProfileError(f"{name} must cover T2-T5")
    result = {}
    for horizon, value in values.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise CrossWindowProfileError(f"{name} values must be finite")
        result[horizon] = float(value)
    return result


def evaluate_final_status(
    *,
    candidate_tmap: Mapping[int, float],
    native_fh_tmap: Mapping[int, float] | None,
    resource_status: str,
    epsilon: float = 1e-6,
) -> dict[str, object]:
    candidate = _tmap(candidate_tmap, name="candidate t_mAP")
    if resource_status not in {"PASS", "FAIL", "UNCONFIRMED"}:
        raise CrossWindowProfileError("resource status differs")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or epsilon < 0
    ):
        raise CrossWindowProfileError("epsilon must be finite and non-negative")
    if native_fh_tmap is None:
        native = None
        deltas = {horizon: None for horizon in candidate}
        tmap_status = "UNCONFIRMED"
    else:
        native = _tmap(native_fh_tmap, name="native FH t_mAP")
        deltas = {
            horizon: candidate[horizon] - native[horizon] for horizon in candidate
        }
        tmap_status = (
            "PASS"
            if all(float(value) > float(epsilon) for value in deltas.values())
            else "FAIL"
        )
    return {
        "schema_version": "crosswindow-final-status-v1",
        "candidate_t_mAP": candidate,
        "native_FH_t_mAP": native,
        "delta_vs_native_FH": deltas,
        "epsilon": float(epsilon),
        "TMAP_ALL_T_STATUS": tmap_status,
        "TMAP_ALL_T_PASS": tmap_status == "PASS",
        "RESOURCE_STATUS": resource_status,
        "RESOURCE_PASS": resource_status == "PASS",
        "JOINT_GOAL_PASS": tmap_status == "PASS" and resource_status == "PASS",
    }


def evaluate_resource_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    native_method: str = "FH-R1-native",
    candidate_method: str = "frozen-final-candidate",
    required_reduction: float = 0.05,
) -> dict[str, object]:
    if not rows:
        return {
            "status": "UNCONFIRMED",
            "reason": "profiling rows are unavailable",
            "checks": {},
        }
    index = {
        (str(row.get("method")), int(row.get("T", -1))): row for row in rows
    }
    required = [
        (method, horizon)
        for method in (native_method, candidate_method)
        for horizon in (4, 5)
    ]
    if any(
        key not in index or index[key].get("status") != "MEASURED"
        for key in required
    ):
        return {
            "status": "UNCONFIRMED",
            "reason": "native and candidate T4-T5 real-forward profiles are required",
            "checks": {},
        }
    checks = {}
    for horizon in (4, 5):
        native = index[(native_method, horizon)]
        candidate = index[(candidate_method, horizon)]
        for field in ("model_update_ms", "cumulative_ms"):
            checks[f"T{horizon}_{field}"] = float(candidate[field]) <= (
                1.0 - required_reduction
            ) * float(native[field])
        checks[f"T{horizon}_end_to_end_ms"] = float(
            candidate["end_to_end_ms"]
        ) <= float(native["end_to_end_ms"])
        checks[f"T{horizon}_allocated_peak_bytes"] = int(
            candidate["allocated_peak_bytes"]
        ) <= int(native["allocated_peak_bytes"])
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "reason": "" if all(checks.values()) else "one or more resource gates failed",
        "checks": checks,
    }


__all__ = [
    "CrossWindowProfileError",
    "evaluate_final_status",
    "evaluate_resource_gate",
]
