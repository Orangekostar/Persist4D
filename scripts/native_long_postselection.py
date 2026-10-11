"""Read-only analysis of the completed, locked native-long cluster evidence."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.native_long_budget import write_json_atomic

HORIZONS = tuple(f"T{h}" for h in range(1, 6))
METRICS = (*HORIZONS, "S_long")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_result(row, expected):
    if (row["status"] != "COMPLETE" or row["completed"] != expected
            or row["expected"] != expected or set(expected) != set(HORIZONS)
            or any(n <= 0 for n in expected.values())):
        raise ValueError("evaluation requires complete official denominators")
    values = row["metrics"]
    if (set(values) != set(METRICS)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values())):
        raise ValueError("evaluation requires finite official metrics")
    if abs(values["S_long"] - sum(values[f"T{h}"] for h in (3, 4, 5)) / 3) > 1e-12:
        raise ValueError("S_long must be the mean of the three pooled horizon APs")


def classify_delta(delta):
    if set(delta) != set(METRICS) or any(not math.isfinite(v) for v in delta.values()):
        raise ValueError("comparison requires all finite horizon deltas")
    if abs(delta["S_long"] - sum(delta[f"T{h}"] for h in (3, 4, 5)) / 3) > 1e-12:
        raise ValueError("comparison S_long differs from horizon deltas")
    regressed = [h for h in HORIZONS if delta[h] < -.002]
    positive = delta["S_long"] > 1e-6
    status = "NO_LONG_GAIN" if not positive else (
        "LONG_TRADEOFF" if regressed else "PROVISIONAL_LONG_GAIN")
    return {"status": status, "regressed_horizons": regressed,
            "positive_long_signal": positive, "eligible": positive and not regressed,
            "development_target_0_5pp_met": delta["S_long"] >= .005,
            "statistically_confirmed": False}


def module_shortlist(comparisons):
    rows = []
    for row in comparisons:
        if row["arm"] not in {"E2", "E3"}:
            continue
        if row["control"] != "E1" or row["step"] != row["control_step"]:
            raise ValueError("module attribution requires a same-step E1 control")
        rows.append({**row, **classify_delta(row["delta"])})
    if len({r["arm"] for r in rows}) != len(rows):
        raise ValueError("duplicate module comparison")
    eligible = sorted([r for r in rows if r["eligible"]], key=lambda r: (
        -r["delta"]["S_long"], -min(r["delta"][h] for h in HORIZONS),
        -r["delta"]["T2"], r["added_parameters"], r["arm"]))
    return {"modules": rows, "ranked_eligible": [r["arm"] for r in eligible],
            "selected_module": eligible[0]["arm"] if eligible else None,
            "selection_basis": "locked SEL pooled metrics; execution instruction section 9.4",
            "reference_diagnostics_used_for_selection": False}


def replication_budget(training_costs, module, *, remaining, reserve):
    if any(not math.isfinite(x) or x < 0 for x in (remaining, reserve)):
        raise ValueError("remaining budget and reserve must be finite and nonnegative")
    if module is None:
        return {"status": "NO_ELIGIBLE_MODULE", "authorized_to_start": False}
    costs = [training_costs[arm] for arm in ("E1", module)]
    if any(not math.isfinite(x) or x <= 0 for x in costs):
        raise ValueError("replication requires actual full training costs")
    forecast = sum(costs) * 1.25
    fits = forecast + reserve <= remaining
    return {"status": "BUDGET_FEASIBLE_PENDING_EXECUTION" if fits else "NOT_REPLICATED_BUDGET",
            "authorized_to_start": fits, "arms": ["E1", module], "training_seed": 46,
            "updates_per_arm": 29700, "measured_seed45_training_gpu_hours": sum(costs),
            "forecast_pair_gpu_hours": forecast, "forecast_multiplier": 1.25,
            "remaining_gpu_hours": remaining, "reserve_gpu_hours": reserve,
            "forecast_excludes_new_CAL_selection": True}


def reference_deltas(candidate, control):
    def index(row):
        result = {}
        for value in row["by_reference"]:
            key = value["horizon"], value["reference_id"]
            if (key in result or value["completed"] != value["expected"]
                    or value["expected"] <= 0 or not math.isfinite(value["AP"])
                    or not 0 <= value["AP"] <= 1):
                raise ValueError("reference population is incomplete or invalid")
            result[key] = value
        return result

    left, right = index(candidate), index(control)
    if left.keys() != right.keys():
        raise ValueError("reference populations differ")
    rows, long = [], collections.defaultdict(list)
    for (h, ref), value in sorted(left.items()):
        base = right[h, ref]
        if value["expected"] != base["expected"]:
            raise ValueError("paired reference denominators differ")
        delta = value["AP"] - base["AP"]
        rows.append({"horizon": h, "reference_id": ref, "inputs": value["expected"],
                     "AP": value["AP"], "control_AP": base["AP"], "delta": delta})
        if h >= 3:
            long[ref].append((h, delta))
    reference_rows = [{"reference_id": ref, "available_horizons": [h for h, _ in values],
                       "mean_available_delta": sum(d for _, d in values) / len(values)}
                      for ref, values in sorted(long.items())]
    signs = {name: 0 for name in ("positive", "zero", "negative")}
    for row in reference_rows:
        delta = row["mean_available_delta"]
        signs["positive" if delta > 1e-6 else "negative" if delta < -1e-6 else "zero"] += 1
    return {"horizon_rows": rows, "long_reference_rows": reference_rows,
            "long_reference_signs": signs,
            "estimand": "per physical reference, mean delta over its available T3–T5; not pooled S_long"}


def compare(candidate, control, kind):
    if candidate["identity"]["input_records"] != control["identity"]["input_records"]:
        raise ValueError("paired ordered input populations differ")
    if candidate["identity"]["evaluation_seed"] != control["identity"]["evaluation_seed"]:
        raise ValueError("paired evaluation seeds differ")
    return {"arm": candidate["arm"], "step": candidate["step"], "control": control["arm"],
            "control_step": control["step"], "comparison_kind": kind,
            "delta": {k: candidate["metrics"][k] - control["metrics"][k] for k in METRICS},
            "reference_diagnostics": reference_deltas(candidate, control)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    from models.native_long_modules import QueryConditionedMaskFeedback

    files = [root / name for name in ("cluster/STATUS.json", "POPULATION.json",
             "selection/BUDGET_LOCK.json", "selection/CAL_LOCK.json",
             "cluster/diagnostics/SEED45_TRAIN_CAL_SEL_COMPLETE_20261010.json")]
    before = {str(p): sha256(p) for p in files}
    status, population, budget, lock, proof = map(read_json, files)
    if (proof["status"] != "ALL_FOUR_SEED45_TRAIN_CAL_AND_LOCKED_SEL_COMPLETE_VERIFIED"
            or lock != proof["CAL_LOCK"] or sha256(files[3]) != proof["CAL_LOCK_sha256"]
            or status["live_gpu_hours"] != 0):
        raise ValueError("final audit/lock differs or GPU work remains active")
    arm_rows = status["snapshots"]
    if set(arm_rows) != {"E0", "E1", "E2", "E3"}:
        raise ValueError("all four trajectories are required")
    lookup, table, training_costs, evidence = {}, [], {}, []
    for job in budget["jobs"]:
        arm = job["arm"]
        snapshot = arm_rows[arm]
        if (snapshot["status"] != "FULL_U_COMPLETE" or snapshot["updates"] != 29700
                or snapshot.get("active") or snapshot.get("stale")
                or snapshot["job_id"] != job["job_id"] or snapshot["identity"] != job["identity"]):
            raise ValueError("completed trajectory identity differs")
        if sorted(r["step"] for r in snapshot["cal"]) != list(range(2970, 29701, 2970)):
            raise ValueError("ten complete CAL endpoints required per arm")
        selected = {lock["selected_updates"][arm]}
        if arm == "E1":
            selected.update(lock["selected_updates"][a] for a in ("E2", "E3"))
        if sorted(r["step"] for r in snapshot["sel"]) != sorted(selected):
            raise ValueError("SEL endpoints differ from immutable requests")
        for role in ("CAL", "SEL"):
            for row in snapshot[role.lower()]:
                validate_result(row, population["evaluation_counts"][role])
                if row["role"] != role or row["arm"] != arm or row["seed"] != 45:
                    raise ValueError("evaluation identity differs")
                records = sorted(population["evaluation_inputs"][role],
                                 key=lambda r: (r["horizon"], r["input_id"]))
                if row["identity"]["input_records"] != records:
                    raise ValueError("evaluation input order differs")
                references = collections.Counter((r["horizon"], r["reference_id"]) for r in records)
                if ({(r["horizon"], r["reference_id"]): r["expected"] for r in row["by_reference"]}
                        != references or len(row["by_reference"]) != len(references)):
                    raise ValueError("evaluation reference coverage differs")
                reference_deltas(row, row)
                if len(row["evidence"]) != 5:
                    raise ValueError("five horizon evidence files required")
                for h, item in enumerate(row["evidence"], 1):
                    name = f"{role}-{row['step']:05d}-T{h}-official-state.json"
                    if Path(item["path"]).name != name:
                        raise ValueError("official evidence name differs")
                    path = root / "cluster/evidence" / arm / "seed45" / name
                    if sha256(path) != item["sha256"]:
                        raise ValueError("official evidence SHA differs")
                    evidence.append({"arm": arm, "role": role, "step": row["step"],
                                     "path": str(path), "sha256": item["sha256"]})
                lookup[arm, role, row["step"]] = row
                table.append({"arm": arm, "role": role, "step": row["step"],
                              "training_seed": 45, "inputs": sum(row["completed"].values()),
                              **row["metrics"], "checkpoint_sha256": row["identity"]["checkpoint_sha256"]})
        training_costs[arm] = sum(e["gpu_hours"] for e in snapshot["events"] if e["phase"].startswith("train:"))
    comparisons = []
    for arm in ("E1", "E2", "E3"):
        step = lock["selected_updates"][arm]
        control = "E0" if arm == "E1" else "E1"
        control_step = lock["selected_updates"][control] if arm == "E1" else step
        row = compare(lookup[arm, "SEL", step], lookup[control, "SEL", control_step],
                      "CAL_SELECTED_BASELINES" if arm == "E1" else "SAME_STEP_MODULE")
        row["added_parameters"] = sum(p.numel() for p in QueryConditionedMaskFeedback().parameters()) if arm == "E3" else 0
        row.update(classify_delta(row["delta"]))
        comparisons.append(row)
    for arm in ("E2", "E3"):
        for control in ("E0", "E1"):
            comparisons.append(compare(lookup[arm, "SEL", lock["selected_updates"][arm]],
                                       lookup[control, "SEL", lock["selected_updates"][control]],
                                       "CAL_SELECTED_ENGINEERING"))
    for step in range(2970, 29701, 2970):
        for arm, control in (("E1", "E0"), ("E2", "E1"), ("E3", "E1")):
            comparisons.append(compare(lookup[arm, "CAL", step], lookup[control, "CAL", step],
                                       "CAL_SAME_STEP_DEVELOPMENT"))
    shortlist = module_shortlist([r for r in comparisons if r["comparison_kind"] == "SAME_STEP_MODULE"])
    remaining = budget["lifetime_cap_gpu_hours"] - status["lifetime_gpu_hours"]
    replication = replication_budget(training_costs, shortlist["selected_module"],
                                     remaining=remaining, reserve=budget["reserve_gpu_hours"])
    formal = {"status": "PLANNED_NOT_RUN", "training_seed": 45, "evaluation_seed": 9001,
              "selected_module": shortlist["selected_module"], "rows": [
                  {"arm": a, "step": lock["selected_updates"][a], "kind": "CAL_SELECTED"}
                  for a in ("E0", "E1", *([shortlist["selected_module"]] if shortlist["selected_module"] else []))],
              "populations": ["PB", "LOCAL-T2", "ADDITIONAL"],
              "population_materialization": "PENDING_EXACT_MANIFEST_AND_TRAIN_DISJOINTNESS_AUDIT",
              "CAL_LOCK_sha256": sha256(files[3]), "post_hoc_checkpoint_reselection_allowed": False}
    if shortlist["selected_module"]:
        step = lock["selected_updates"][shortlist["selected_module"]]
        if step != lock["selected_updates"]["E1"]:
            formal["rows"].append({"arm": "E1", "step": step, "kind": "MATCHED_STEP_CONTROL"})
    if before != {str(p): sha256(p) for p in files}:
        raise ValueError("source evidence changed during analysis")
    output.mkdir(parents=True, exist_ok=True)
    payload = {"status": "SEED45_POSTSELECTION_ANALYZED_CONFIRMATION_PENDING", "CAL_LOCK": lock,
               "source_sha256": before, "official_evidence_verified": evidence,
               "units": "AP and delta in 0–1; percentages and percentage points multiply by 100",
               "comparisons": comparisons, "shortlist": shortlist, "replication": replication,
               "formal_plan": formal, "budget_gpu_hours": status["lifetime_gpu_hours"],
               "runtime_status": "RUNTIME_CONDITIONAL", "cvpr_readiness": "NOT_READY"}
    write_json_atomic(output / "ANALYSIS.json", payload)
    write_json_atomic(output / "SHORTLIST.json", shortlist)
    write_json_atomic(output / "FORMAL_PLAN.json", formal)
    write_csv(output / "CAL_ALL.csv", [r for r in table if r["role"] == "CAL"])
    write_csv(output / "SEL_LOCKED.csv", [r for r in table if r["role"] == "SEL"])
    write_csv(output / "PAIRED_DELTAS.csv", [{k: v for k, v in r.items() if k in (
        "arm", "step", "control", "control_step", "comparison_kind")} | r["delta"] for r in comparisons])
    write_csv(output / "BY_REFERENCE.csv", [{"arm": r["arm"], "step": r["step"],
        "control": r["control"], "control_step": r["control_step"], "comparison_kind": r["comparison_kind"],
        **item} for r in comparisons for item in r["reference_diagnostics"]["horizon_rows"]])
    report = "# Native-long seed45 选后证据分析\n\n"
    report += "状态：NOT_READY_FOR_CVPR。训练完成不代表正式确认、独立训练复验或持久实体系统收益已验证。\n\n"
    report += "| 比较 | 步数 | ΔS_long (pp) | ΔT1 (pp) | ΔT2 (pp) | 状态 | 长时 reference 正/零/负 |\n|---|---|---:|---:|---:|---|---|\n"
    for row in comparisons[:3]:
        d, signs = row["delta"], row["reference_diagnostics"]["long_reference_signs"]
        report += (f"| {row['arm']}−{row['control']} | {row['step']}/{row['control_step']} | "
                   f"{100*d['S_long']:+.4f} | {100*d['T1']:+.4f} | {100*d['T2']:+.4f} | {row['status']} | "
                   f"{signs['positive']}/{signs['zero']}/{signs['negative']} |\n")
    report += "\n上述模块对照使用相同优化步数；不是与 E1 最佳检查点比较。各自 CAL 选点的工程比较：\n\n"
    report += "| 比较 | 各自步数 | ΔS_long (pp) | ΔT2 (pp) |\n|---|---|---:|---:|\n"
    for row in comparisons[3:7]:
        report += (f"| {row['arm']}−{row['control']} | {row['step']}/{row['control_step']} | "
                   f"{100*row['delta']['S_long']:+.4f} | {100*row['delta']['T2']:+.4f} |\n")
    report += ("\nreference 统计只平均该场景实际存在的 T3–T5，缺失长度不填零；"
               "它不是官方 pooled S_long，也不能用于构造其置信区间或代替独立训练种子。\n\n")
    report += f"按原协议进入正式确认的模块：{shortlist['selected_module']}。正式人口和污染审计尚待执行。\n\n"
    report += (f"预算已用 {status['lifetime_gpu_hours']:.6f}/1800 GPUh，余 {remaining:.6f}。"
               f"seed46 状态：{replication['status']}；配对完整训练预估 "
               f"{replication.get('forecast_pair_gpu_hours', 0):.2f} GPUh（另留8 GPUh），不启动注定超额的训练。\n\n")
    report += ("已核验40个CAL和6个SEL结果的分母、输入顺序及230份本地官方指标证据SHA。"
               "尚需正式确认、新进程波动测试、真实profile、独立复验和论文主张整合；"
               "新感知实验不能直接支持旧固定实体记忆系统的增益。\n")
    (output / "REPORT.md").write_text(report)
    return {"status": payload["status"], "report": str(output / "REPORT.md"),
            "selected_module": shortlist["selected_module"], "replication": replication,
            "evidence_files_verified": len(evidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
