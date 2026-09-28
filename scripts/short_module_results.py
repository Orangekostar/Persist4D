"""Fixed-step replication and mechanism comparisons for independent screened heads."""

import csv
import json
import time
from pathlib import Path

from scripts.short_module_evaluation import (
    CONTROLS,
    TOLERANCE,
    build_shortlist,
    evaluate_point,
    write_csv,
)
from scripts.short_module_screen import ARTIFACTS, read_json, write_json
from scripts.short_module_training import load_training_records, train_arm


def replication_requests(rows: list[dict]) -> dict[str, list[int]]:
    requests = {}
    for row in rows:
        if row["classification"] not in ("MODEST_GAIN", "TARGET_GAIN"):
            continue
        arm, step = row["module"], int(row["selected_step"])
        requests.setdefault(arm, set()).add(step)
        if arm in CONTROLS:
            requests.setdefault(CONTROLS[arm], set()).add(step)
    return {arm: sorted(steps) for arm, steps in sorted(requests.items())}


def read_screening() -> list[dict]:
    with (ARTIFACTS / "selection/MODULE_SCREENING.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("seed", "actual_updates", "selected_step", "positive_SEL_references"):
            row[field] = int(row[field])
        for field in ("SEL_T2_AP", "delta_T2_B0", "SEL_T1_AP", "delta_T1_B0"):
            row[field] = float(row[field]) if row[field] else None
        for field in ("seed46_T2_delta_B0", "seed46_T1_delta_B0"):
            if field in row:
                row[field] = float(row[field]) if row[field] else None
        if "geometry_signal_only" in row:
            row["geometry_signal_only"] = row["geometry_signal_only"] == "True"
    return rows


def replicate(root: Path, *, device: str) -> dict:
    rows = read_screening()
    requests = replication_requests(rows)
    plan = {"seed": 46, "fixed_updates": requests, "epoch_search": False,
            "schedule_horizon": 1500, "maximum_updates_per_arm": 1500}
    plan_path = ARTIFACTS / "selection/REPLICATION_PLAN.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("replication request changed after fixed-point registration")
    write_json(plan_path, plan)
    started = time.perf_counter()
    results, failures = {}, {}
    records = load_training_records(root, read_json(root / "EXPORT_INDEX.json")) if requests else None
    for arm, steps in requests.items():
        try:
            resolved = read_json(ARTIFACTS / "RUN_CONFIG.json")
            consumed = sum(json.loads(line)["gpu_hours"] for line in
                           (ARTIFACTS / "BUDGET_LEDGER.jsonl").read_text().splitlines())
            remaining = min(192., resolved["prior_gpu_hours"] + 48.) - consumed - (time.perf_counter() - started) / 3600
            observed = read_json(ARTIFACTS / "training" / arm / "checkpoint_manifest.json")
            predicted_hours = (observed["training_seconds"] * max(steps) / 1500 + 120 * len(steps)) / 3600
            if remaining < predicted_hours + 1.:
                failures[arm] = "NOT_REPLICATED_BUDGET: preserve profile/reload/publication allowance"
                continue
            train_arm(arm, seed=46, updates=max(steps), root=root, device=device, training_records=records)
            for step in steps:
                results[f"{arm}:{step}"] = evaluate_point(root, module=arm, step=step, seed=46,
                                                          role="SEL", device=device)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
            failures[arm] = f"{type(error).__name__}: {error}"
    baseline = read_json(ARTIFACTS / "BASELINE.json")["SEL"]
    by_reference = []
    for row in rows:
        second = results.get(f"{row['module']}:{row['selected_step']}")
        row["seed46_T2_delta_B0"] = None
        row["seed46_T1_delta_B0"] = None
        if second is not None and second["status"] == "COMPLETE":
            d2 = second["metrics"]["T2"] - baseline["metrics"]["T2"]
            d1 = second["metrics"]["T1"] - baseline["metrics"]["T1"]
            row["seed46_T2_delta_B0"] = d2
            row["seed46_T1_delta_B0"] = d1
            if row["classification"] in ("MODEST_GAIN", "TARGET_GAIN"):
                if d2 > TOLERANCE and d1 >= -.002 - TOLERANCE:
                    row["evidence_level"] = ("DEVELOPMENT_REPLICATED" if row["positive_SEL_references"] >= 3
                                             else "REFERENCE_MIXED")
                else:
                    row["evidence_level"] = "MIXED_SEED"
        else:
            row["evidence_level"] = "NOT_REPLICATED"
    for key, result in results.items():
        arm, step = key.split(":")
        base_refs = {(r["H"], r["reference_id"]): r["AP"] for r in baseline["by_reference"]}
        for r in result["by_reference"]:
            by_reference.append({"module": arm, "seed": 46, "step": int(step), **r,
                                 "delta_B0": None if r["AP"] is None else r["AP"] - base_refs[r["H"], r["reference_id"]]})
    write_csv(ARTIFACTS / "evaluation/SEED46_BY_REFERENCE.csv", by_reference)
    write_csv(ARTIFACTS / "selection/MODULE_SCREENING.csv", rows)
    write_json(ARTIFACTS / "selection/SHORTLIST.json", build_shortlist(rows))
    summary = {"status": "COMPLETE" if not failures else "PARTIAL", "plan": plan,
               "results": results, "failures": failures, "elapsed_seconds": time.perf_counter() - started}
    write_json(ARTIFACTS / "selection/REPLICATION_RESULTS.json", summary)
    mechanism_comparisons(root, summary)
    return summary


def mechanism_comparisons(root: Path, replication: dict | None = None) -> list[dict]:
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    cal = {}
    for evidence in lock["evidence"]:
        result = read_json(Path(evidence["path"]))
        cal[result["module"], result["step"]] = result
    selected_sel = read_json(ARTIFACTS / "selection/SCREEN_RESULTS.json")
    rows = []

    def add(scope, seed, arm, control, left, right):
        rows.append({"scope": scope, "seed": seed, "module": arm, "control": control,
                     "module_step": left["step"], "control_step": right["step"],
                     "T2_delta": left["metrics"]["T2"] - right["metrics"]["T2"],
                     "T1_delta": left["metrics"]["T1"] - right["metrics"]["T1"],
                     "interpretation": "Mechanism contrast; engineering benefit still requires positive delta against B0"})

    for arm, control in CONTROLS.items():
        a, b = lock["selected_updates"][arm], lock["selected_updates"][control]
        add("CAL_individually_selected", 45, arm, control, cal[arm, a], cal[control, b])
        if selected_sel[arm]["status"] == selected_sel[control]["status"] == "COMPLETE":
            add("SEL_individually_selected", 45, arm, control, selected_sel[arm], selected_sel[control])
        for step in (500, 1000, 1500):
            add("CAL_same_step", 45, arm, control, cal[arm, step], cal[control, step])
        if replication:
            second = replication["results"]
            for step in replication["plan"]["fixed_updates"].get(arm, []):
                if f"{arm}:{step}" in second and f"{control}:{step}" in second:
                    left, right = second[f"{arm}:{step}"], second[f"{control}:{step}"]
                    if left["status"] == right["status"] == "COMPLETE":
                        add("SEL_seed46_fixed_same_step", 46, arm, control, left, right)
    write_csv(ARTIFACTS / "evaluation/PAIRED_MECHANISM_DELTAS.csv", rows)
    return rows
