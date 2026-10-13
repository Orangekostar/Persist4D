"""Lock MPR-V1 CAL endpoints from completed repeated evaluation rows."""

import argparse
import csv
import hashlib
import json
import math
import numbers
import sys
from pathlib import Path

from scripts.native_long_budget import write_json_atomic


VARIANTS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
TRAINED_VARIANTS = VARIANTS[1:]
HORIZONS = (1, 2, 3, 4, 5)
REPEATS = (0, 1, 2)
TOLERANCE = 1e-6
GUARD_DELTA = 0.002
REPLICATION_LONG_DELTA = 0.005
REPLICATION_CURRENT_DELTA = 0.0
REPLICATION_AP_DELTA = -0.002
REPLICATION_RANGE = 0.002


def _is_int(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_unit(value):
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _as_expected_inputs(run_lock):
    expected = run_lock.get("expected_inputs", {}).get("CAL")
    if not isinstance(expected, dict):
        raise ValueError("run_lock.expected_inputs.CAL must be a mapping")
    result = {}
    for horizon in HORIZONS:
        value = expected.get(str(horizon), expected.get(horizon))
        if not _is_int(value) or value <= 0:
            raise ValueError(f"CAL expected_inputs T{horizon} must be a positive integer")
        result[horizon] = int(value)
    return result


def _validate_run_lock(run_lock):
    if not isinstance(run_lock, dict):
        raise ValueError("run_lock must be a mapping")
    n = run_lock.get("N")
    if not _is_int(n) or n not in (400, 800, 1200, 1600):
        raise ValueError("run_lock.N must be one of 400, 800, 1200, 1600")
    steps = run_lock.get("checkpoint_steps")
    if not isinstance(steps, list) or len(steps) != 4 or any(
        not _is_int(step) or step <= 0 for step in steps
    ):
        raise ValueError("run_lock.checkpoint_steps must contain four positive integers")
    if steps != [n // 4, n // 2, 3 * n // 4, n]:
        raise ValueError("run_lock.checkpoint_steps must be N/4, N/2, 3N/4, N")
    parameters = run_lock.get("trainable_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("run_lock.trainable_parameters must be a mapping")
    normalized_parameters = {}
    for variant in VARIANTS:
        value = parameters.get(variant)
        if not _is_int(value) or value < 0:
            raise ValueError(f"missing or invalid parameter count for {variant}")
        normalized_parameters[variant] = int(value)
    replication_variants = run_lock.get("replication_variants")
    if replication_variants is None:
        replication_variants = list(TRAINED_VARIANTS)
    elif (
        not isinstance(replication_variants, list)
        or not replication_variants
        or len(set(replication_variants)) != len(replication_variants)
        or any(variant not in TRAINED_VARIANTS for variant in replication_variants)
    ):
        raise ValueError("run_lock.replication_variants must list unique trained variants")
    return {
        "N": int(n),
        "checkpoint_steps": [int(step) for step in steps],
        "expected_inputs": _as_expected_inputs(run_lock),
        "trainable_parameters": normalized_parameters,
        "replication_variants": list(replication_variants),
    }


def _add_reason(reasons, reason):
    if reason not in reasons:
        reasons.append(reason)


def _row_seed(row):
    value = row.get("adapter_seed") if isinstance(row, dict) else None
    return value if _is_int(value) else None


def _row_step(row):
    value = row.get("step") if isinstance(row, dict) else None
    return value if _is_int(value) else None


def _row_matches_scope(row, *, variant, source_seed, allowed_steps):
    return (
        isinstance(row, dict)
        and row.get("variant") == variant
        and row.get("role") == "CAL"
        and _row_seed(row) == source_seed
        and _row_step(row) in allowed_steps
    )


def _point_key(row):
    return (
        int(row["step"]),
        row.get("repeat_index"),
        row.get("horizon"),
    )


def _validate_point(row, expected_inputs):
    reasons = []
    repeat = row.get("repeat_index")
    horizon = row.get("horizon")
    if not _is_int(repeat) or repeat not in REPEATS:
        _add_reason(reasons, "INVALID_REPEAT_INDEX")
    if not _is_int(horizon) or horizon not in HORIZONS:
        _add_reason(reasons, "INVALID_HORIZON")
    expected = expected_inputs.get(horizon) if _is_int(horizon) else None
    observed_expected = row.get("expected_inputs")
    observed_completed = row.get("completed_inputs")
    if not _is_int(observed_expected) or observed_expected <= 0:
        _add_reason(reasons, "INVALID_EXPECTED_INPUTS")
    elif observed_expected != expected:
        _add_reason(reasons, "EXPECTED_INPUTS_MISMATCH")
    if not _is_int(observed_completed) or observed_completed != expected:
        _add_reason(reasons, "COMPLETED_INPUTS_MISMATCH")
    if row.get("status") != "COMPLETE":
        _add_reason(reasons, "STATUS_NOT_COMPLETE")
    if not _is_finite_unit(row.get("AP")):
        _add_reason(reasons, "INVALID_AP")
    if not _is_finite_unit(row.get("current_AP")):
        _add_reason(reasons, "INVALID_CURRENT_AP")
    return reasons


def _source_seed(rows, *, seed):
    """Use the frozen M0 seed when selecting a seed147 replication matrix."""
    if seed == 147:
        if any(
            _row_matches_scope(row, variant="M0", source_seed=146, allowed_steps={0})
            for row in rows
        ):
            return 146
    return seed


def _complete_variant(rows, variant, *, source_seed, allowed_steps, expected_inputs):
    reasons = []
    expected_keys = {
        (step, repeat, horizon)
        for step in allowed_steps
        for repeat in REPEATS
        for horizon in HORIZONS
    }
    scoped = [
        row
        for row in rows
        if _row_matches_scope(
            row,
            variant=variant,
            source_seed=source_seed,
            allowed_steps=set(allowed_steps),
        )
    ]
    by_key = {}
    for row in scoped:
        key = _point_key(row)
        by_key.setdefault(key, []).append(row)

    valid_points = {}
    for key in sorted(expected_keys):
        candidates = by_key.get(key, [])
        if not candidates:
            _add_reason(reasons, f"MISSING_POINT:{key[0]}:R{key[1]}:T{key[2]}")
            continue
        if len(candidates) != 1:
            _add_reason(reasons, f"DUPLICATE_POINT:{key[0]}:R{key[1]}:T{key[2]}")
            continue
        row = candidates[0]
        point_reasons = _validate_point(row, expected_inputs)
        for reason in point_reasons:
            _add_reason(reasons, f"{reason}:{key[0]}:R{key[1]}:T{key[2]}")
        if not point_reasons:
            valid_points[key] = row
    return valid_points if not reasons else None, reasons


def _mean(values):
    return sum(float(value) for value in values) / len(values)


def _endpoint(variant, source_seed, step, points, parameters):
    ap = {}
    current = {}
    ap_ranges = {}
    repeat_long = []
    for horizon in HORIZONS:
        values = [points[(step, repeat, horizon)]["AP"] for repeat in REPEATS]
        current_values = [
            points[(step, repeat, horizon)]["current_AP"] for repeat in REPEATS
        ]
        ap[f"T{horizon}"] = _mean(values)
        current[f"T{horizon}"] = _mean(current_values)
        ap_ranges[f"T{horizon}"] = max(values) - min(values)
    for repeat in REPEATS:
        repeat_long.append(
            _mean([points[(step, repeat, horizon)]["AP"] for horizon in (3, 4, 5)])
        )
    return {
        "variant": variant,
        "adapter_seed": source_seed,
        "step": step,
        "parameters": parameters,
        "AP": ap,
        "current_AP": current,
        "S_long": _mean([ap[f"T{horizon}"] for horizon in (3, 4, 5)]),
        "S_current": _mean([current[f"T{horizon}"] for horizon in (3, 4, 5)]),
        "S_long_range": max(repeat_long) - min(repeat_long),
        "AP_ranges": ap_ranges,
        "guard_pass": False,
    }


def _metric_compare(left, right):
    for left_value, right_value in (
        (left["S_long"], right["S_long"]),
        (left["S_current"], right["S_current"]),
        (left["AP"]["T2"], right["AP"]["T2"]),
    ):
        difference = round(left_value - right_value, 12)
        if difference > TOLERANCE:
            return 1
        if difference < -TOLERANCE:
            return -1
    return 0


def _better_within_variant(left, right):
    comparison = _metric_compare(left, right)
    if comparison:
        return left if comparison > 0 else right
    return left if left["step"] < right["step"] else right


def _better_cross_variant(left, right):
    comparison = _metric_compare(left, right)
    if comparison:
        return left if comparison > 0 else right
    if left["parameters"] != right["parameters"]:
        return left if left["parameters"] < right["parameters"] else right
    return left if left["variant"] < right["variant"] else right


def _best(endpoints):
    winner = None
    for endpoint in endpoints:
        winner = endpoint if winner is None else _better_cross_variant(winner, endpoint)
    return winner


def _replication_decision(result, run_lock):
    reasons = []
    complete = result["status"] == "COMPLETE"
    control = result["control_star"]
    candidate = result["candidate_star"]
    if not complete:
        reasons.append("EXPERIMENT_INCOMPLETE")
    if control is None:
        reasons.append("NO_GUARD_PASSING_CONTROL")
    if candidate is None:
        reasons.append("NO_GUARD_PASSING_CANDIDATE")
    if control is not None and candidate is not None:
        delta_long = candidate["S_long"] - control["S_long"]
        delta_current = candidate["S_current"] - control["S_current"]
        if delta_long < REPLICATION_LONG_DELTA:
            reasons.append("DELTA_S_LONG_BELOW_0.005")
        if delta_current < REPLICATION_CURRENT_DELTA:
            reasons.append("DELTA_S_CURRENT_NEGATIVE")
        for horizon in (3, 4, 5):
            delta = candidate["AP"][f"T{horizon}"] - control["AP"][f"T{horizon}"]
            if delta < REPLICATION_AP_DELTA:
                reasons.append(f"DELTA_AP_T{horizon}_BELOW_-0.002")
        if control["S_long_range"] > REPLICATION_RANGE:
            reasons.append("CONTROL_S_LONG_RANGE_ABOVE_0.002")
        if candidate["S_long_range"] > REPLICATION_RANGE:
            reasons.append("CANDIDATE_S_LONG_RANGE_ABOVE_0.002")
        if delta_long <= 2.0 * max(
            candidate["S_long_range"], control["S_long_range"]
        ):
            reasons.append("DELTA_S_LONG_NOT_GREATER_THAN_TWICE_RANGE")
    eligible = not reasons
    variants_to_train = []
    if eligible:
        variants_to_train.append(candidate["variant"])
        if control["variant"] != "M0":
            variants_to_train.append(control["variant"])
    return {
        "eligible": eligible,
        "reasons": reasons,
        "variants_to_train": variants_to_train,
        "seed": 147,
        "N": run_lock["N"],
    }


def select_cal_rows(rows, run_lock, *, seed=146):
    """Select complete CAL endpoints and determine whether seed147 is allowed."""
    if seed not in (146, 147):
        raise ValueError("seed must be 146 or 147")
    normalized_lock = _validate_run_lock(run_lock)
    locked_control = None
    locked_candidate = None
    if seed == 147:
        if "replication_variants" not in run_lock:
            raise ValueError("seed147 requires explicitly locked replication families")
        families = normalized_lock["replication_variants"]
        candidates = [variant for variant in families if variant in {"M4", "M5", "M6"}]
        controls = [variant for variant in families if variant in {"M1", "M2", "M3"}]
        if len(candidates) != 1 or len(controls) > 1:
            raise ValueError("seed147 families must contain one candidate and at most one trained control")
        locked_candidate = candidates[0]
        locked_control = controls[0] if controls else "M0"
    rows = list(rows)
    source_m0_seed = _source_seed(rows, seed=seed)
    required_variants = (
        list(VARIANTS)
        if seed == 146
        else ["M0", *normalized_lock["replication_variants"]]
    )
    required_variants = list(dict.fromkeys(required_variants))

    variant_status = {}
    elimination_reasons = {}
    selected_by_variant = {}
    complete_endpoints = {}
    for variant in VARIANTS:
        if variant not in required_variants:
            variant_status[variant] = "NOT_REQUIRED"
            elimination_reasons[variant] = ["SEED147_REPLICATION_SCOPE"]
            selected_by_variant[variant] = None
            continue
        source_seed = source_m0_seed if variant == "M0" else seed
        steps = [0] if variant == "M0" else normalized_lock["checkpoint_steps"]
        points, reasons = _complete_variant(
            rows,
            variant,
            source_seed=source_seed,
            allowed_steps=steps,
            expected_inputs=normalized_lock["expected_inputs"],
        )
        if points is None:
            variant_status[variant] = "INCOMPLETE"
            elimination_reasons[variant] = reasons
            selected_by_variant[variant] = None
            continue
        endpoints = [
            _endpoint(
                variant,
                source_seed,
                step,
                points,
                normalized_lock["trainable_parameters"][variant],
            )
            for step in steps
        ]
        winner = endpoints[0]
        for endpoint in endpoints[1:]:
            winner = _better_within_variant(winner, endpoint)
        variant_status[variant] = "COMPLETE"
        elimination_reasons[variant] = []
        selected_by_variant[variant] = winner
        complete_endpoints[variant] = winner

    m0 = complete_endpoints.get("M0")
    for variant, endpoint in complete_endpoints.items():
        if variant == "M0":
            endpoint["guard_pass"] = True
        elif m0 is not None:
            endpoint["guard_pass"] = (
                endpoint["AP"]["T1"] >= m0["AP"]["T1"] - GUARD_DELTA
                and endpoint["AP"]["T2"] >= m0["AP"]["T2"] - GUARD_DELTA
            )
        else:
            endpoint["guard_pass"] = False
        if not endpoint["guard_pass"]:
            _add_reason(elimination_reasons[variant], "GUARD_FAILED")
        elif variant != "M0" and m0 is None:
            _add_reason(elimination_reasons[variant], "NO_COMPLETE_CONTROL")

    endpoints = list(complete_endpoints.values())
    best_overall = _best(endpoints)
    control_star = _best(
        endpoint
        for variant, endpoint in complete_endpoints.items()
        if variant in {"M0", "M1", "M2", "M3"} and endpoint["guard_pass"]
    )
    candidate_star = _best(
        endpoint
        for variant, endpoint in complete_endpoints.items()
        if variant in {"M4", "M5", "M6"} and endpoint["guard_pass"]
    )
    candidate_descriptive_star = _best(
        endpoint
        for variant, endpoint in complete_endpoints.items()
        if variant in {"M4", "M5", "M6"}
    )
    if seed == 147:
        # A negative replication must not replace the main seed's locked family.
        # Guard outcomes remain recorded separately for the final paired verdict.
        control_star = complete_endpoints.get(locked_control)
        candidate_star = complete_endpoints.get(locked_candidate)
        candidate_descriptive_star = candidate_star

    required_complete = all(
        variant_status.get(variant) == "COMPLETE" for variant in required_variants
    )
    status = "COMPLETE" if required_complete else "EXPERIMENT_INCOMPLETE"
    result = {
        "schema": 1,
        "status": status,
        "seed": seed,
        "selected_by_variant": selected_by_variant,
        "variant_status": variant_status,
        "elimination_reasons": elimination_reasons,
        "best_overall": best_overall,
        "control_star": control_star,
        "candidate_star": candidate_star,
        "candidate_descriptive_star": candidate_descriptive_star,
        "family_selection_frozen": seed == 147,
    }
    if seed == 146:
        result["replication"] = _replication_decision(result, normalized_lock)
    else:
        result["replication"] = {
            "eligible": False,
            "reasons": ["SEED147_SELECTION_ONLY"],
            "variants_to_train": [],
            "seed": 147,
            "N": normalized_lock["N"],
        }
    return result


def _csv_value(row, name, converter):
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        return converter(value)
    except (TypeError, ValueError):
        return value


def _read_results_csv(path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = []
        for row in csv.DictReader(stream):
            parsed = dict(row)
            for field in ("adapter_seed", "step", "repeat_index", "horizon", "expected_inputs", "completed_inputs"):
                parsed[field] = _csv_value(parsed, field, int)
            for field in ("AP", "current_AP"):
                parsed[field] = _csv_value(parsed, field, float)
            rows.append(parsed)
    return rows


def _error_payload(seed, message):
    return {
        "schema": 1,
        "status": "INPUT_ERROR",
        "seed": seed,
        "error": message,
        "selected_by_variant": {variant: None for variant in VARIANTS},
        "variant_status": {variant: "INCOMPLETE" for variant in VARIANTS},
        "elimination_reasons": {variant: ["INPUT_ERROR"] for variant in VARIANTS},
        "best_overall": None,
        "control_star": None,
        "candidate_star": None,
        "candidate_descriptive_star": None,
        "replication": {
            "eligible": False,
            "reasons": ["INPUT_ERROR"],
            "variants_to_train": [],
            "seed": 147,
            "N": None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Select MPR-V1 CAL checkpoints from repeated CSV results."
    )
    parser.add_argument(
        "--results",
        default="artifacts/revisit_prompt_v1/CAL_ALL.csv",
        help="CAL results CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--run-lock",
        default="artifacts/revisit_prompt_v1/RUN_LOCK.json",
        help="run lock JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        help="output lock; defaults to CAL_SELECTION_LOCK or REPLICATION_LOCK under the public artifact directory",
    )
    parser.add_argument("--main-selection-lock",default="artifacts/revisit_prompt_v1/CAL_SELECTION_LOCK.json",
                        help="frozen main family qualification for seed147")
    parser.add_argument("--seed", type=int, choices=(146, 147), default=146)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the selection without writing output",
    )
    args = parser.parse_args(argv)
    output = Path(args.output or f"artifacts/revisit_prompt_v1/{'REPLICATION_LOCK' if args.seed == 147 else 'CAL_SELECTION_LOCK'}.json")
    try:
        results_path = Path(args.results)
        lock_path = Path(args.run_lock)
        if not results_path.is_file():
            raise FileNotFoundError(f"results file not found: {results_path}")
        if not lock_path.is_file():
            raise FileNotFoundError(f"run lock file not found: {lock_path}")
        rows = _read_results_csv(results_path)
        with lock_path.open("r", encoding="utf-8") as stream:
            run_lock = json.load(stream)
        main_path = None
        if args.seed == 147:
            main_path = Path(args.main_selection_lock)
            with main_path.open("r",encoding="utf-8") as stream:
                main_selection = json.load(stream)
            replication = main_selection.get("replication",{})
            if main_selection.get("status") != "COMPLETE" or not replication.get("eligible"):
                raise ValueError("seed147 requires qualified frozen main selection families")
            run_lock = {**run_lock,"replication_variants":replication["variants_to_train"]}
        result = select_cal_rows(rows, run_lock, seed=args.seed)
        result["input_identities"] = {"run_lock_sha256":hashlib.sha256(lock_path.read_bytes()).hexdigest(),
                                      "results_sha256":hashlib.sha256(results_path.read_bytes()).hexdigest(),
                                      "results_snapshot":f"CAL_SELECTION_INPUT_s{args.seed}.csv",
                                      "main_selection_sha256":hashlib.sha256(main_path.read_bytes()).hexdigest() if main_path else None}
        if output.is_file():
            existing = json.loads(output.read_text())
            if existing.get("status") == "COMPLETE" and existing != result:
                raise ValueError("completed selection lock is frozen; preserve its original results")
        rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if not args.dry_run:
            snapshot = output.parent/result["input_identities"]["results_snapshot"]
            snapshot.parent.mkdir(parents=True,exist_ok=True)
            temporary = snapshot.with_suffix(".tmp")
            temporary.write_bytes(results_path.read_bytes())
            temporary.replace(snapshot)
            write_json_atomic(output, result)
        print(rendered)
        return 0
    except Exception as exc:
        result = _error_payload(args.seed, str(exc))
        if not args.dry_run:
            try:
                error_path = output.with_name(output.stem+".entry-error.json") if output.exists() else output
                write_json_atomic(error_path, result)
            except Exception as write_error:
                result["output_error"] = str(write_error)
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

