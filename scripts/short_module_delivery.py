"""Result presentation and bounded Git/Release delivery for this experiment only."""

import ast
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

from scripts.short_module_evaluation import (
    ARMS,
    CONTROLS,
    build_cal_lock,
    build_shortlist,
    write_csv,
)
from scripts.short_module_results import read_screening
from scripts.short_module_screen import (
    ARTIFACTS,
    PROJECT,
    read_json,
    sha256,
    write_json,
)

BRANCH = "research/rescene-short-module-screen-v1"
TAG = "rescene-short-module-screen-v1"
REPOSITORY = "Orangekostar/Persist4D"


def _csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_experiment(root: Path) -> dict:
    """Check concrete experiment artifacts; the primary review also covers semantics."""
    from scripts.p6a_metrics import recompute_official_metric_evidence

    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise ValueError(f"delivery evidence failed: {name}")

    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    index = read_json(root / "EXPORT_INDEX.json")
    cache = Path(index["cache"])
    identity = read_json(cache / "IDENTITY.json")
    archived_data = ARTIFACTS / "recovery/short_module_data.before_regularization_fix.py.txt"
    for source, digest in identity["sources"].items():
        if Path(source).name == "short_module_data.py" and sha256(Path(source)) != digest:
            require("historical_label_source", archived_data.exists() and sha256(archived_data) == digest)
            old = {node.name: ast.dump(node) for node in ast.parse(archived_data.read_text()).body
                   if isinstance(node, ast.FunctionDef)}
            current = {node.name: ast.dump(node) for node in ast.parse(Path(source).read_text()).body
                       if isinstance(node, ast.FunctionDef)}
            require("cache_label_functions_unchanged", all(old[name] == current[name] for name in
                    ("_order", "select_population", "quality_labels", "geometry_assignment", "low_segment_targets")))
        else:
            require("producer_source:" + source, sha256(Path(source)) == digest)
    require("full_export", index["status"] == "COMPLETE" and
            index["completed_unique_inputs"] == index["expected_unique_inputs"] ==
            len({r["input_id"] for r in population["records"]}))
    for entry in index["entries"]:
        audit, record = entry["audit"], entry["record"]
        require(f"native_input:{entry['input_id']}", audit["parent_frozen"] and audit["parent_eval"] and
                [Path(p).stem for p in audit["actual_loaded_paths"]] ==
                [scan.removeprefix("scene") for scan in record["scan_ids"]] and
                len(audit["actual_loaded_paths"]) == record["horizon"])
        for kind in ("prediction", "targets"):
            require(f"cache_bytes:{entry['input_id']}:{kind}",
                    sha256(cache / entry[kind]["file"]) == entry[kind]["sha256"])
    require("native_equivalence", any(all((e["audit"].get("native_equality") or {}).values()) and
            bool(e["audit"].get("native_equality")) for e in index["entries"]))
    baseline = read_json(ARTIFACTS / "BASELINE.json")
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    points = []
    for evidence in lock["evidence"]:
        require("CAL_hash:" + evidence["path"], sha256(Path(evidence["path"])) == evidence["sha256"])
        points.append(read_json(Path(evidence["path"])))
    require("all_CAL_points", {(r["module"], r["step"]) for r in points} ==
            {(arm, step) for arm in ARMS for step in (0, 500, 1000, 1500)})
    require("CAL_selection_recomputed", build_cal_lock(points)["selected_updates"] == lock["selected_updates"])
    selected = read_json(ARTIFACTS / "selection/SCREEN_RESULTS.json")
    require("all_SEL_locked", set(selected) == set(ARMS) and
            all(selected[a]["step"] == lock["selected_updates"][a] for a in ARMS))
    points.extend(selected.values())
    points.extend(baseline.values())
    replication = read_json(ARTIFACTS / "selection/REPLICATION_RESULTS.json")
    from scripts.short_module_results import replication_requests

    requests = replication_requests(read_screening())
    require("fixed_replication_plan", replication["plan"]["fixed_updates"] == requests)
    require("replication_disposition", replication["status"] == "COMPLETE" or
            all("NOT_REPLICATED_BUDGET" in reason for reason in replication["failures"].values()))
    points.extend(replication["results"].values())
    for result in points:
        if result["module"] != "B0":
            checkpoint = root / "training" / result["module"] / f"seed{result['seed']}" / f"update={result['step']:04d}.pt"
            require("evaluated_head:" + result["result_path"], sha256(checkpoint) == result["head_sha256"])
        for horizon in (1, 2):
            evidence_path = Path(result["result_path"]).with_name(
                f"{result['role']}-{result['step']:04d}-H{horizon}-metric-state.json")
            recomputed = recompute_official_metric_evidence(read_json(evidence_path))
            value = recomputed["raw_local_AP" if horizon == 1 else "online_t-mAP"]
            require(f"metric_recomputed:{evidence_path}", value == result["metrics"][f"T{horizon}"])
        expected = {str(h): population["counts"][result["role"]][f"T{h}"] for h in (1, 2)}
        require(f"complete_metric:{result['module']}:{result['role']}:{result['step']}",
                result["status"] == "COMPLETE" and result["expected"] == result["completed"] == expected
                and all(value is not None for value in result["metrics"].values())
                and len(result["by_reference"]) == 8)
        if result["module"].startswith("Q"):
            require("Q_geometry:" + result["result_path"], result["isolation"]["mask_unchanged"])
        if result["module"].startswith("M"):
            require("M_scores:" + result["result_path"], result["isolation"]["score_unchanged"])
            if result["step"] == 0:
                require("M_zero:" + result["module"], result["isolation"]["mask_unchanged"])
    manifests = {arm: read_json(ARTIFACTS / "training" / arm / "checkpoint_manifest.json") for arm in ARMS}
    require("common_sampling", len({m["sample_plan_sha256"] for m in manifests.values()}) == 1)
    require("same_M_initialization", len({manifests[a]["initial_parameter_sha256"] for a in ("M0", "M1", "M2")}) == 1)
    require("same_Q12_initialization", manifests["Q1"]["initial_parameter_sha256"] == manifests["Q2"]["initial_parameter_sha256"])
    for arm, meta in manifests.items():
        require("trained:" + arm, meta["optimizer_updates"] == 1500 and not meta["parent_loaded_in_training"]
                and len(meta["supervised_gradient_checks"]) == 2 and
                all(r["head_changed"] and not r["parent_in_optimizer"] for r in meta["supervised_gradient_checks"]))
        require("checkpoints:" + arm, {c["step"] for c in meta["checkpoints"]} == {0, 500, 1000, 1500} and
                all(sha256(Path(c["path"])) == c["sha256"] for c in meta["checkpoints"]))
        rows = _csv(ARTIFACTS / "training" / arm / "metrics.csv")
        require("training_log:" + arm, len(rows) == 15 and all(
            abs(float(r["loss"]) - sum(float(r[k]) for k in ("bce", "dice", "regularization", "quality"))) < 1e-6
            and sum(float(r[k]) for k in ("positive", "negative", "unknown")) == 1600 for r in rows))
    for arm, steps in requests.items():
        if arm in replication["failures"]:
            continue
        second = read_json(ARTIFACTS / "training" / arm / "seed46/checkpoint_manifest.json")
        require("seed46_fresh:" + arm, second["seed"] == 46 and second["optimizer_updates"] == max(steps)
                and second["schedule_horizon"] == 1500 and second["initial_parameter_sha256"] !=
                manifests[arm]["initial_parameter_sha256"] and not second["parent_loaded_in_training"])
        require("seed46_fixed_points:" + arm, all(f"{arm}:{step}" in replication["results"] for step in steps))
    profile = read_json(ARTIFACTS / "resources/PROFILE_SUMMARY.json")
    require("full_profile", profile["status"] == "COMPLETE" and profile["measurement_rows"] == 112
            and profile["warmup_rows"] == 28 and len(profile["reload_checks"]) == 6 and
            all(all(r["equal"].values()) for r in profile["reload_checks"]))
    ledger = [json.loads(line) for line in (ARTIFACTS / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
    require("budget", sum(r["gpu_hours"] for r in ledger) <= 192 and
            sum(r["gpu_hours"] for r in ledger if r["scope"] == "ROUND1") <= 48)
    result = {"status": "VERIFIED", "checks": checks,
              "scope": "Artifact and numeric invariants; does not substitute for primary scientific/source review"}
    write_json(ARTIFACTS / "validation/EXPERIMENT_CHECKS.json", result)
    return result


def verify_code() -> dict:
    sources = [PROJECT / "models/short_module_heads.py", *sorted((PROJECT / "scripts").glob("short_module_*.py")),
               *sorted((PROJECT / "tests").glob("test_short_module_*.py"))]
    identity = {str(p.relative_to(PROJECT)): sha256(p) for p in sources}
    path = ARTIFACTS / "validation/CODE_CHECKS.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("sources") == identity and previous.get("status") == "VERIFIED":
            return previous
    tests = [str(p.relative_to(PROJECT)) for p in sources if p.parent.name == "tests"]
    commands = [[sys.executable, "-m", "pytest", "-q", *tests, "tests/test_rescene_task_postprocess.py"],
                ["/home/ww/miniconda3/bin/ruff", "check", *identity],
                ["git", "diff", "--check"]]
    results = []
    for command in commands:
        started = time.monotonic()
        completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, check=False,
                                   timeout=1200)
        results.append({"command": command, "exit_code": completed.returncode,
                        "elapsed_seconds": time.monotonic() - started,
                        "output": completed.stdout + completed.stderr})
        if completed.returncode:
            write_json(path, {"status": "FAILED", "sources": identity, "commands": results})
            raise ValueError("relevant code verification failed: " + results[-1]["output"])
    result = {"status": "VERIFIED", "sources": identity, "commands": results}
    write_json(path, result)
    return result


def export_public_heads(root: Path) -> dict:
    heads = {}
    for arm in ARMS:
        (ARTIFACTS / "training" / arm).mkdir(parents=True, exist_ok=True)
        records = []
        for source in sorted((root / "training" / arm).glob("seed*/update=*.pt")):
            saved = torch.load(source, map_location="cpu", weights_only=False)
            meta = saved["metadata"]
            if meta["module"] != arm or sum(t.numel() for t in saved["head"].values()) != meta["parameter_count"]:
                raise ValueError("public head contains mismatched state")
            destination = ARTIFACTS / "training" / arm / "checkpoints" / f"seed{meta['seed']}-update{meta['optimizer_updates']:04d}.pt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                torch.save({"head": saved["head"], "metadata": meta}, destination)
            public = torch.load(destination, map_location="cpu", weights_only=False)
            if public["metadata"] != meta or not all(torch.equal(public["head"][key], value)
                                                    for key, value in saved["head"].items()):
                raise ValueError("existing public checkpoint differs from evaluated training state")
            if destination.stat().st_size > 20 * 1024**2:
                raise ValueError("head exceeds the authorized Git size limit")
            records.append({"seed": meta["seed"], "step": meta["optimizer_updates"],
                            "path": str(destination.relative_to(PROJECT)),
                            "sha256": sha256(destination), "bytes": destination.stat().st_size})
        heads[arm] = records
        mode = "score_only" if arm.startswith("Q") else "mask_only"
        (ARTIFACTS / "training" / arm / "README.md").write_text(
            f"# {arm} standalone head\n\nMode: `{mode}`. Exactly one runtime module; no combinations.\n\n"
            "`deployment.pt` is the seed45 CAL-selected head. `checkpoints/` contains every saved small head, including negative controls and fixed-step seed46 controls when run. Optimizer states are intentionally excluded from these public inference bundles; external recovery checkpoints retain them.\n\n"
            "Load `head` with `models.short_module_heads.build_head` using `metadata.dimensions`, thresholds and seed, then call `apply_module` with native parent prediction/descriptor and the original materialization system. No GT input is accepted. Metadata binds the frozen R1 SHA, supported real H1/H2 inputs, selected update, descriptor and postprocess. R1/Concerto weights and data are not redistributed.\n\n"
            "A score head's update0 is not an identity transform. A mask head's update0 is zero residual through the real materializer. Disable through B0. Same-slot heads are alternatives, not stackable.\n")
    write_json(ARTIFACTS / "PUBLIC_HEADS.json", heads)
    return heads


def report(root: Path) -> dict:
    required = ("BASELINE.json", "selection/CAL_LOCK.json", "selection/SCREEN_RESULTS.json",
                "selection/REPLICATION_RESULTS.json", "resources/PROFILE_SUMMARY.json", "diagnostics/SUMMARY.json")
    missing = [name for name in required if not (ARTIFACTS / name).exists()]
    screen_path = ARTIFACTS / "selection/SCREEN_RESULTS.json"
    if screen_path.exists() and (set(read_json(screen_path)) != set(ARMS) or
                                any(r["status"] != "COMPLETE" for r in read_json(screen_path).values())):
        missing.append("selection/SCREEN_RESULTS.json:INCOMPLETE")
    for name in ("resources/PROFILE_SUMMARY.json", "diagnostics/SUMMARY.json"):
        if (ARTIFACTS / name).exists() and read_json(ARTIFACTS / name).get("status") != "COMPLETE":
            missing.append(name + ":INCOMPLETE")
    ledger = [json.loads(line) for line in (ARTIFACTS / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
    cost = {"cumulative_gpu_hours": sum(row["gpu_hours"] for row in ledger),
            "round_gpu_hours": sum(row["gpu_hours"] for row in ledger if row["scope"] == "ROUND1"),
            "prior_gpu_hours": sum(row["gpu_hours"] for row in ledger if row["scope"] == "PRIOR")}
    if missing:
        heads = export_public_heads(root)
        population_path = ARTIFACTS / "SHORT_POPULATION.json"
        counts = read_json(population_path)["counts"] if population_path.exists() else {}
        partial_rows = [{"module": arm, "status": "NOT_RUN_OR_BLOCKED", "CAL_T1_AP": None,
                         "CAL_T2_AP": None, "SEL_T1_AP": None, "SEL_T2_AP": None,
                         "expected_CAL_T1": counts.get("CAL", {}).get("T1"),
                         "expected_CAL_T2": counts.get("CAL", {}).get("T2"),
                         "expected_SEL_T1": counts.get("SEL", {}).get("T1"),
                         "expected_SEL_T2": counts.get("SEL", {}).get("T2"),
                         "available_saved_heads": len(heads[arm])} for arm in ARMS]
        available = read_json(screen_path) if screen_path.exists() else {}
        for row in partial_rows:
            if row["module"] in available:
                point = available[row["module"]]
                row.update(status=point["status"], SEL_T1_AP=point["metrics"]["T1"],
                           SEL_T2_AP=point["metrics"]["T2"])
        write_csv(ARTIFACTS / "MODULE_SCREENING.csv", partial_rows)
        write_json(ARTIFACTS / "SHORTLIST.json", {**build_shortlist([]), "status": "PARTIAL"})
        summary = {"scientific_status": "PARTIAL", "missing": missing, "cost": cost}
        (ARTIFACTS / "FINAL_REPORT.md").write_text(
            "# ReScene round-one single-module screen — partial\n\n"
            "No completion or positive-module claim is supported yet. Missing stages/artifacts:\n\n"
            + "\n".join(f"- `{name}`" for name in missing)
            + f"\n\nMeasured cumulative cost: {cost['cumulative_gpu_hours']:.6f} GPU hours. Resume the fixed run; do not shrink population denominators.\n")
        write_json(ARTIFACTS / "REPORT_STATUS.json", summary)
        return summary
    heads = export_public_heads(root)
    verify_experiment(root)
    rows = read_screening()
    baseline = read_json(ARTIFACTS / "BASELINE.json")
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    cal = {}
    for evidence in lock["evidence"]:
        result = read_json(Path(evidence["path"]))
        cal[result["module"], result["step"]] = result
    selected_sel = read_json(ARTIFACTS / "selection/SCREEN_RESULTS.json")
    replication = read_json(ARTIFACTS / "selection/REPLICATION_RESULTS.json")
    all_points = [*baseline.values(), *cal.values(), *selected_sel.values(), *replication["results"].values()]
    baseline_refs = {(role, row["H"], row["reference_id"]): row["AP"]
                     for role, result in baseline.items() for row in result["by_reference"]}
    write_csv(ARTIFACTS / "evaluation/BY_REFERENCE.csv", [
        {"module": result["module"], "seed": result["seed"], "step": result["step"],
         "role": result["role"], **row,
         "delta_B0": row["AP"] - baseline_refs[result["role"], row["H"], row["reference_id"]]
         if row["AP"] is not None else None}
        for result in all_points for row in result["by_reference"]])
    write_csv(ARTIFACTS / "evaluation/STAGE_AP_GIVEN_T2.csv", [
        {"module": result["module"], "seed": result["seed"], "step": result["step"], "role": result["role"],
         "status": result["status"], **result["stage_AP_given_T2"]} for result in all_points])
    write_csv(ARTIFACTS / "evaluation/SEED46_FIXED_CHECKPOINTS.csv", [
        {"module": result["module"], "seed": 46, "step": result["step"], "status": result["status"],
         "T1_AP": result["metrics"]["T1"], "T2_AP": result["metrics"]["T2"]}
        for result in replication["results"].values()])
    profile = _csv(ARTIFACTS / "resources/PROFILE.csv")
    transitions = _csv(ARTIFACTS / "diagnostics/GEOMETRY_TRANSITIONS.csv")
    for row in rows:
        arm = row["module"]
        row["status"] = selected_sel[arm]["status"]
        measured = [r for r in profile if r["module"] == arm and r["warmup"] == "False"]
        geometry = [r for r in transitions if r["module"] == arm and r["role"] == "SEL" and float(r["threshold"]) == .5]
        selected = cal[arm, row["selected_step"]]
        row["CAL_T2_AP"] = selected["metrics"]["T2"]
        row["CAL_T1_AP"] = selected["metrics"]["T1"]
        row["mechanism_control"] = CONTROLS.get(arm)
        row["mechanism_SEL_T2_delta"] = (row["SEL_T2_AP"] - selected_sel[CONTROLS[arm]]["metrics"]["T2"]
                                         if arm in CONTROLS else None)
        row["geometry_repairs_at050"] = sum(int(r["repaired_tracks"]) for r in geometry)
        row["geometry_breaks_at050"] = sum(int(r["broken_tracks"]) for r in geometry)
        row["joint_pass_delta_at050"] = sum(int(r["joint_pass_module"]) - int(r["joint_pass_B0"]) for r in geometry)
        row["median_end_to_end_seconds"] = statistics.median(float(r["end_to_end_seconds"]) for r in measured)
        row["peak_allocated_GiB"] = max(int(r["cuda_peak_allocated_bytes"]) for r in measured) / 1024**3
        row["peak_reserved_GiB"] = max(int(r["cuda_peak_reserved_bytes"]) for r in measured) / 1024**3
        worst_delta = sum(float(r["mean_worst_stage_iou_delta"] or 0) * int(r["original_assigned_tracks"]) for r in geometry)
        row["geometry_signal_only"] = (row["delta_T2_B0"] <= 1e-6
                                       and (row["joint_pass_delta_at050"] > 0 or worst_delta > 1e-6))
        row["geometry_classification"] = "GEOMETRY_SIGNAL_ONLY" if row["geometry_signal_only"] else "NONE"
    measured_base = [r for r in profile if r["module"] == "B0" and r["warmup"] == "False"]
    base_row = {key: None for key in rows[0]}
    base_row.update(module="B0", slot="parent", seed=45, actual_updates=0, selected_step=0, status="COMPLETE",
                    CAL_T2_AP=baseline["CAL"]["metrics"]["T2"], CAL_T1_AP=baseline["CAL"]["metrics"]["T1"],
                    SEL_T2_AP=baseline["SEL"]["metrics"]["T2"], SEL_T1_AP=baseline["SEL"]["metrics"]["T1"],
                    delta_T2_B0=0., delta_T1_B0=0., classification="BASELINE", evidence_level="FROZEN_R1",
                    median_end_to_end_seconds=statistics.median(float(r["end_to_end_seconds"]) for r in measured_base),
                    peak_allocated_GiB=max(int(r["cuda_peak_allocated_bytes"]) for r in measured_base) / 1024**3,
                    peak_reserved_GiB=max(int(r["cuda_peak_reserved_bytes"]) for r in measured_base) / 1024**3)
    write_csv(ARTIFACTS / "evaluation/MAIN_RESULTS.csv", [base_row, *rows])
    write_csv(ARTIFACTS / "selection/MODULE_SCREENING.csv", rows)
    shortlist = build_shortlist(rows)
    write_json(ARTIFACTS / "selection/SHORTLIST.json", shortlist)
    write_csv(ARTIFACTS / "MODULE_SCREENING.csv", rows)
    write_json(ARTIFACTS / "SHORTLIST.json", shortlist)
    scientific_status = "COMPLETE" if any(r["classification"] in ("MODEST_GAIN", "TARGET_GAIN") for r in rows) else "NO_POSITIVE_MODULES"
    population = read_json(ARTIFACTS / "SHORT_POPULATION.json")
    text = ["# ReScene round-one single-module screen", "", f"Scientific status: **{scientific_status}**.", "",
            "Six separate heads were trained from the same frozen R1 features. No combination or default deployment change was executed. Results are development screening, not a formal test or a claim to exceed official ReScene.", "",
            "| Arm | Step | CAL T2 | SEL T2 | ΔT2 vs B0 | SEL T1 | ΔT1 vs B0 | Seed46 ΔT2 | Positive refs | Evidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in [base_row, *rows]:
        second = r.get("seed46_T2_delta_B0")
        second_text = f"{second:+.6f}" if second is not None else "—"
        text.append(f"| {r['module']} | {r['selected_step']} | {r['CAL_T2_AP']:.6f} | {r['SEL_T2_AP']:.6f} | {r['delta_T2_B0']:+.6f} | {r['SEL_T1_AP']:.6f} | {r['delta_T1_B0']:+.6f} | {second_text} | {r['positive_SEL_references'] if r['positive_SEL_references'] is not None else '—'} | {r['evidence_level']} |")
    for r in shortlist["provisional"]:
        if r["evidence_level"] == "MIXED_SEED":
            text += ["", f"{r['module']}'s seed45 signal did not replicate: seed46 ΔT2={r['seed46_T2_delta_B0']:+.6f}. It remains a labeled single-seed signal, not a confirmed positive module. Its seed45 reference consistency is {r['positive_SEL_references']}/4."]
    text += ["", "AP and deltas use the [0,1] scale. Positive reference counts describe consistency, not statistical significance.", "",
             f"Confirmed independent shortlist: {', '.join(r['module'] for r in shortlist['confirmed']) or 'empty'}. Provisional positive signals: {', '.join(r['module'] for r in shortlist['provisional']) or 'empty'}. Same-slot candidates are **ALTERNATIVES_NOT_STACKABLE**.", "",
             "## Population and protocol", "",
             f"Population counts: `{json.dumps(population['counts'], sort_keys=True)}`. Every T1 input is a separately loaded and forwarded single scan. T2 stage AP is reported separately in evaluation evidence; it is not T1. TRAIN is physically disjoint from CAL/SEL and previously exposed PB/LOCAL/ADDITIONAL; R1 pretraining and prior development exposure remain disclosed.", "",
             "All seed45 arms have 1500 updates. CAL selects only 500/1000/1500 after the full trajectory; update0 is diagnostic. All six selections were locked before head SEL scoring. Seed46 uses fixed seed45 steps and direct controls with a 1500-step schedule horizon. No epoch search or positive-seed replacement was performed.", "",
             "Q1/Q2 have 95,873 parameters and identical seeded initialization; Q3 has 96,393 (+520) and the same seeded trunk. Labels use the same temporal-best GT, strict official > thresholds, and pre-disambiguation pairwise records. Output is a quality ranking score, not an established calibrated deployment probability. M0/M1/M2 each have 68,225 parameters and identical fresh initial states; M1 changes only unmatched supervision and M2 changes only stage aggregation. M0/M1 queries already contain temporal context.", "",
             "## Mechanism and geometry", "",
             "[Paired contrasts](evaluation/PAIRED_MECHANISM_DELTAS.csv) contain each arm's individually selected CAL/SEL point, same-step CAL contrasts, and fixed same-step seed46 SEL controls when available. Same-step seed45 diagnostics use CAL so SEL is not expanded into another checkpoint search. Engineering gain is always measured against B0; beating a degraded control alone is insufficient.", "",
             "[Candidate diagnostics](diagnostics/CANDIDATE_FAILURES.csv), [joint/separate counts](diagnostics/JOINT_SEPARATE_COUNTS.csv), [appearance/disappearance](diagnostics/APPEAR_DISAPPEAR.csv), and [geometry transitions](diagnostics/GEOMETRY_TRANSITIONS.csv) disclose real denominators. Joint/separate geometry is not a realizable AP upper bound. Original-track diagnostics use the frozen class-compatible one-to-one assignment with concat IoU≥0.10 and one GT identity across stages. Score-only geometry is unchanged; any score-matching recall change is not new geometry.", "",
             "## Cost and reproducibility", "",
             f"This round consumed **{cost['round_gpu_hours']:.6f} GPU hours**; cumulative usage is **{cost['cumulative_gpu_hours']:.6f}** including **{cost['prior_gpu_hours']:.6f}** inherited usage. All process reservations, shared parent export and failures/retries are charged. Limits are 48 incremental/192 cumulative GPU hours. [Profile](resources/PROFILE.csv) contains a fresh full parent forward for B0 and every selected module on four fixed SEL pairs, one warmup plus three measurements each; metric/hash/reload work is outside timed intervals.", "",
             "[Main results](evaluation/MAIN_RESULTS.csv) include actual updates, both APs/deltas, mechanism deltas, geometry repairs/breaks, measured latency and VRAM. [Public head inventory](PUBLIC_HEADS.json) binds all small saved states; each selected deployment bundle passed a real-input reload equality check. Original R1/Concerto, raw data and GT are not redistributed.", "",
             "The existing data staging uses hardlinks and null per-file hashes; byte-immutability is not claimed. Where ambiguity metadata is absent, labels report AMBIGUITY_METADATA_UNAVAILABLE rather than claiming all real ambiguity was resolved. FP32, evaluation seed45, TF32-off and the recorded 2-thread/cuBLAS environment are fixed. The [regularization correction](recovery/REGULARIZATION_FIX.md) preserves and charges discarded M attempts; Q/M0 log replay verified exact saved parameter equality.", "",
             "## Delivery", "",
             "Git publication, tag and actual asset availability are recorded in HANDOFF.md and the external publication receipt. Release authentication was unavailable at initial preflight; small heads remain directly distributable through Git. Publication status is separate from scientific status. No deployment was replaced and no future combination was executed.", ""]
    (ARTIFACTS / "FINAL_REPORT.md").write_text("\n".join(text))
    summary = {"scientific_status": scientific_status, "cost": cost,
               "confirmed": [r["module"] for r in shortlist["confirmed"]],
               "provisional": [r["module"] for r in shortlist["provisional"]],
               "public_head_count": sum(len(r) for r in heads.values())}
    write_json(ARTIFACTS / "REPORT_STATUS.json", summary)
    return summary


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=PROJECT, text=True,
                          capture_output=True, check=True).stdout.strip()


def _metric_asset(root: Path, experiment_commit: str) -> dict:
    points = []
    lock = read_json(ARTIFACTS / "selection/CAL_LOCK.json")
    points.extend(Path(e["path"]) for e in lock["evidence"])
    for name in ("BASELINE.json", "selection/SCREEN_RESULTS.json"):
        points.extend(Path(r["result_path"]) for r in read_json(ARTIFACTS / name).values())
    points.extend(Path(r["result_path"]) for r in read_json(ARTIFACTS / "selection/REPLICATION_RESULTS.json")["results"].values())
    path = root / "publication" / f"official-metric-evidence-{experiment_commit[:12]}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for point in sorted(set(points)):
            result = read_json(point)
            prefix = f"{result['module']}/seed{result['seed']}"
            archive.write(point, f"{prefix}/{point.name}")
            for horizon in (1, 2):
                evidence = point.with_name(f"{result['role']}-{result['step']:04d}-H{horizon}-metric-state.json")
                archive.write(evidence, f"{prefix}/{evidence.name}")
    if path.stat().st_size > 1024**3:
        raise ValueError("metric evidence asset exceeds1 GiB")
    return {"path": str(path), "planned_name": path.name, "sha256": sha256(path),
            "bytes": path.stat().st_size, "content": "Official sufficient metric states and result records; no data, GT masks or parent weights"}


def publish(root: Path) -> dict:
    from scripts.perception_gain_v2_publication import GitHubReleaseClient

    if _git("branch", "--show-current") != BRANCH:
        raise ValueError("refusing publication outside the dedicated branch")
    state_path = root / "publication/PUBLISH_STATE.json"
    receipt_path = root / "publication/PUBLICATION_RECEIPT.json"
    client = GitHubReleaseClient.authorized()
    if receipt_path.exists():
        previous = read_json(receipt_path)
        if (_git("rev-parse", "HEAD") == previous["publication_commit_B"]
                and (previous["publication_status"] == "VERIFIED" or client is None)):
            return previous
    if state_path.exists():
        state = read_json(state_path)
    else:
        summary = read_json(ARTIFACTS / "REPORT_STATUS.json")
        if summary["scientific_status"] not in ("COMPLETE", "NO_POSITIVE_MODULES", "PARTIAL"):
            raise ValueError("delivery requires an explicit scientific status")
        verify_code()
        if summary["scientific_status"] != "PARTIAL":
            verify_experiment(root)
        source_paths = ["models/short_module_heads.py", "configs/short_module_screen_v1.yaml",
                        "docs/0928/ReScene_Round1_Single_Module",
                        "docs/superpowers/plans/2026-09-28-short-module-screen.md"]
        source_paths += [f"scripts/short_module_{name}.py" for name in
                         ("data", "native", "screen", "training", "evaluation", "results", "diagnostics", "profile", "delivery")]
        source_paths += [f"tests/test_short_module_{name}.py" for name in
                         ("heads", "data", "native", "screen", "training", "evaluation", "results", "diagnostics")]
        _git("add", "--", *source_paths)
        _git("add", "-f", "--", "artifacts/short_module_screen_v1")
        # csv.DictWriter deliberately uses the standard CRLF CSV dialect.
        # Recognize CR as line termination while retaining whitespace checks.
        _git("-c", "core.whitespace=cr-at-eol", "diff", "--cached", "--check")
        _git("commit", "-m", "Record ReScene independent single-module experiments and small heads")
        experiment = _git("rev-parse", "HEAD")
        tag, revision = TAG, 1
        while _git("ls-remote", "origin", f"refs/tags/{tag}") or subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"], cwd=PROJECT, check=False).returncode == 0:
            revision += 1
            tag = f"{TAG}-r{revision}"
        state = {"experiment_commit_A": experiment, "tag": tag,
                 "assets": [_metric_asset(root, experiment)] if summary["scientific_status"] != "PARTIAL" else [],
                 "missing_scientific_artifacts": summary.get("missing", [])}
        write_json(state_path, state)
    if "publication_commit_B" not in state:
        handoff = ["# ReScene round-one handoff", "",
                   f"Experiment commit A: `{state['experiment_commit_A']}`. Branch: `{BRANCH}`. Planned immutable tag: `{state['tag']}`.", "",
                   "Scientific results: [FINAL_REPORT.md](FINAL_REPORT.md). All small head checkpoints and selected deployment bundles are in [PUBLIC_HEADS.json](PUBLIC_HEADS.json). No combinations or default model replacement occurred.", "",
                   "This handoff/manifest is commit B. B is intentionally absent from its own content; the external publication receipt records A/B/tag and verifies actual remote references.", "",
                   f"Publication availability: {'Release authorization available; upload/download verification is pending the external receipt.' if client else 'CODE_ONLY: Git carries code, complete result tables and small heads. No existing authorized Release credential was available.'}", "",
                   "Required external metric-evidence asset:", ""]
        for asset in state["assets"]:
            handoff.append(f"- `{asset['planned_name']}` — {asset['bytes']} bytes, SHA256 `{asset['sha256']}`; {'pending upload verification' if client else 'not available from Release; local verified copy exists' }.")
        handoff += ["", "After existing GitHub Release authorization is restored, rerun:", "",
                    "```bash", f"cd {PROJECT}",
                    f"/home/ww/miniconda3/envs/persist4d/bin/python -m scripts.short_module_screen publish --root {root} --resume",
                    "```", "", "No token should be pasted into logs or supplied to another application. Parent weights, raw data and GT are not redistributed.", ""]
        (ARTIFACTS / "HANDOFF.md").write_text("\n".join(handoff))
        files = []
        for path in sorted(ARTIFACTS.rglob("*")):
            if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
                files.append({"path": str(path.relative_to(PROJECT)), "sha256": sha256(path), "bytes": path.stat().st_size})
        write_json(ARTIFACTS / "ARTIFACT_MANIFEST.json", {"experiment_commit_A": state["experiment_commit_A"],
                   "files": files, "release_assets": state["assets"], "self_hash_excluded": True})
        _git("add", "-f", "--", "artifacts/short_module_screen_v1/HANDOFF.md", "artifacts/short_module_screen_v1/ARTIFACT_MANIFEST.json")
        _git("commit", "-m", "Freeze single-module delivery manifest and handoff")
        state["publication_commit_B"] = _git("rev-parse", "HEAD")
        write_json(state_path, state)
    commit = state["publication_commit_B"]
    if _git("rev-parse", "HEAD") != commit:
        raise ValueError("HEAD changed after publication snapshot")
    _git("push", "origin", f"HEAD:refs/heads/{BRANCH}")
    local_tag = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{state['tag']}"],
                               cwd=PROJECT, capture_output=True, text=True, check=False)
    if local_tag.returncode:
        _git("tag", state["tag"], commit)
    elif local_tag.stdout.strip() != commit:
        raise ValueError("refusing to move existing local tag")
    _git("push", "origin", f"refs/tags/{state['tag']}")
    remote = _git("ls-remote", "origin", f"refs/heads/{BRANCH}", f"refs/tags/{state['tag']}")
    refs = {line.split()[1]: line.split()[0] for line in remote.splitlines()}
    if refs != {f"refs/heads/{BRANCH}": commit, f"refs/tags/{state['tag']}": commit}:
        raise ValueError("remote branch/tag SHA verification failed")
    release_url, availability = None, []
    if client is not None:
        response = client.session.get(client.root + f"/releases/tags/{state['tag']}", timeout=60)
        release = (response.json() if response.status_code == 200 else client.create_draft(
            tag=state["tag"], commit=commit, notes="Independent ReScene single-module development screen. See FINAL_REPORT.md; no combinations or deployment replacement."))
        existing = {a["name"]: a for a in client.list_assets(release["id"])}
        for asset in state["assets"]:
            if asset["planned_name"] not in existing:
                client.upload(release, asset)
            remote_asset = next(a for a in client.list_assets(release["id"]) if a["name"] == asset["planned_name"])
            digest, count = hashlib.sha256(), 0
            for block in client.download(remote_asset):
                digest.update(block)
                count += len(block)
            if digest.hexdigest() != asset["sha256"] or count != asset["bytes"]:
                raise ValueError("Release asset roundtrip bytes/digest mismatch")
            availability.append({"name": asset["planned_name"], "available": True, "sha256": digest.hexdigest(),
                                 "bytes": count, "url": remote_asset["browser_download_url"]})
        release = client.publish(release["id"])
        release_url = release["html_url"]
    else:
        availability = [{"name": a["planned_name"], "available": False, "local_verified": sha256(Path(a["path"])) == a["sha256"]}
                        for a in state["assets"]]
    receipt = {**state, "branch": BRANCH, "repository": REPOSITORY, "remote_refs": refs,
               "publication_status": "VERIFIED" if client and not state.get("missing_scientific_artifacts") else "CODE_ONLY", "release_url": release_url,
               "asset_availability": availability, "small_heads_in_Git": True,
               "scientific_status": read_json(ARTIFACTS / "REPORT_STATUS.json")["scientific_status"],
               "recovery_command": f"python -m scripts.short_module_screen publish --root {root} --resume"}
    write_json(receipt_path, receipt)
    return receipt
