"""Verify completed controlled runs and generate evidence-bounded comparison tables."""

import csv
import json
from pathlib import Path

import torch

from scripts.p6a_metrics import recompute_official_metric_evidence
from scripts.rescene_code_first_audit import OLD_PUBLIC, OLD_ROOT, PUBLIC, ROOT
from scripts.short_module_eval_identity import evaluation_identity
from scripts.short_module_identity import identity_digest
from scripts.short_module_screen import read_json, sha256, write_json


def verify_noaug_inputs():
    original = read_json(OLD_ROOT / "EXPORT_INDEX.json")
    current = read_json(ROOT / "noaug/EXPORT_INDEX.json")
    identity = read_json(Path(current["cache"]) / "IDENTITY.json")
    if identity["apply_training_augmentation"] is not False or any("train" not in mode for mode in identity["dataset_modes"].values()):
        raise ValueError("noaug must disable augmentation while retaining train discovery mode")
    old_entries = {r["input_id"]: r for r in original["entries"] if r["record"]["role"] in ("CAL", "SEL")}
    if {r["input_id"] for r in current["entries"]} != set(old_entries) or len(current["entries"]) != 94:
        raise ValueError("noaug input population changed")

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return isinstance(b, (tuple, list)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    for entry in current["entries"]:
        old = old_entries[entry["input_id"]]
        if entry["record"] != old["record"] or entry["audit"]["actual_loaded_paths"] != old["audit"]["actual_loaded_paths"]:
            raise ValueError("noaug changed scan identity or scan order")
        if not entry["audit"]["parent_eval"] or not entry["audit"]["parent_frozen"]:
            raise ValueError("noaug parent was not frozen/eval")
        a_path = Path(original["cache"]) / old["targets"]["file"]
        b_path = Path(current["cache"]) / entry["targets"]["file"]
        if sha256(a_path) != old["targets"]["sha256"] or sha256(b_path) != entry["targets"]["sha256"]:
            raise ValueError("target shard content changed")
        a = torch.load(a_path, map_location="cpu", weights_only=False)
        b = torch.load(b_path, map_location="cpu", weights_only=False)
        for field in ("target", "semantic_labels"):
            if not equal(a[field], b[field]):
                raise ValueError(f"noaug target/semantic/ambiguity mismatch: {entry['input_id']}/{field}")
    result = {"status": "COMPLETE", "inputs": 94, "same_records_and_loaded_scan_order": True,
              "same_entire_target_including_ambiguities_and_semantic_labels": True,
              "parent_frozen_and_eval": True}
    write_json(PUBLIC / "NOAUG_INPUT_PARITY.json", result)
    return result


def verify_result(row):
    if row["status"] != "COMPLETE" or row["completed"] != row["expected"] or row["incomplete"]:
        raise ValueError("incomplete metric population")
    path = Path(row["result_path"])
    if read_json(path) != row:
        raise ValueError("comparison is not identical to persisted result")
    root = path.parents[4]
    index = read_json(root / "EXPORT_INDEX.json")
    native_identity = read_json(Path(index["cache"]) / "IDENTITY.json")
    if identity_digest(native_identity) != index["cache_identity_sha256"]:
        raise ValueError("native materialization configuration/identity content changed")
    head = (Path(row["checkpoint_root"]) / "training" / row["module"] / f"seed{row['seed']}"
            / f"update={row['step']:04d}.pt") if row["module"] != "B0" else None
    identity = evaluation_identity(index=index, head_path=head,
                                   dataset_spec=read_json(root / "assets.local.json")["metric_dataset_spec"])
    if identity_digest(identity) != row["evaluator_sha256"]:
        raise ValueError(f"stale evaluation dependency: {path}")
    for h in (1, 2):
        evidence_path = path.with_name(f"{row['role']}-{row['step']:04d}-H{h}-metric-state.json")
        evidence = read_json(evidence_path)
        recomputed = recompute_official_metric_evidence(evidence)
        value = recomputed["raw_local_AP" if h == 1 else "online_t-mAP"]
        if abs(value - row["metrics"][f"T{h}"]) > 1e-8:
            raise ValueError(f"metric state does not reproduce reported AP: {evidence_path}")
    return {"path": str(path), "sha256": sha256(path), "recomputed_metric_states": 2}


def main():
    if sha256(PUBLIC / "SHORT_POPULATION.json") != sha256(OLD_PUBLIC / "SHORT_POPULATION.json"):
        raise ValueError("controlled evaluation population/order changed")
    q = read_json(PUBLIC / "Q_MATCHED_COMPARISON.json")
    m = read_json(PUBLIC / "M_MATCHED_COMPARISON.json")
    noaug = read_json(PUBLIC / "NOAUG_FIXED_HEADS.json")
    if {(r["role"], r["arm"]) for r in q} != {(role, arm) for role in ("CAL", "SEL") for arm in ("Q1", "Q2", "Q3")} or len(q) != 6:
        raise ValueError("Q matched comparison incomplete")
    if {(r["role"], r["arm"]) for r in m} != {(role, arm) for role in ("CAL", "SEL") for arm in ("M0", "M1")} or len(m) != 4:
        raise ValueError("conditional M comparison incomplete")
    if {(r["role"], r["module"]) for r in noaug} != {(role, arm) for role in ("CAL", "SEL") for arm in ("B0", "Q1", "Q2", "Q3", "M0", "M1", "M2")} or len(noaug) != 14:
        raise ValueError("original-head noaug comparison incomplete")
    unique = {}
    for row in [r[k] for r in q for k in ("baseline", "original", "repaired")] + [r[k] for r in m for k in ("original", "expanded_eligibility")] + noaug:
        unique[row["result_path"]] = row
    for row in [r["original"] for r in q + m] + noaug:
        if row["module"] == "B0":
            continue
        manifest = read_json(OLD_PUBLIC / "training" / row["module"] / "checkpoint_manifest.json")
        checkpoint = next(c for c in manifest["checkpoints"] if c["step"] == row["step"])
        if row["head_sha256"] != checkpoint["sha256"]:
            raise ValueError("original locked head bytes differ from historical manifest")
    verified = [verify_result(row) for row in unique.values()]
    input_parity = verify_noaug_inputs()
    relabel = read_json(PUBLIC / "RELABEL_AUDIT.json")
    if relabel["R1_forwards"] != 0 or any(r["legacy_M_assignment_changes"] for r in relabel["groups"]):
        raise ValueError("Q repair changed the parent inputs or M assignments")
    training = {}
    for arm in ("Q1", "Q2", "Q3", "M0", "M1"):
        old = read_json(OLD_PUBLIC / "training" / arm / "checkpoint_manifest.json")
        new = read_json(PUBLIC / "training" / arm / "checkpoint_manifest.json")
        for field in ("base_sha256", "sample_plan_sha256", "initial_parameter_sha256", "dimensions", "thresholds", "seed"):
            if old[field] != new[field]:
                raise ValueError(f"controlled training invariant changed: {arm}/{field}")
        if new["optimizer_updates"] != 1500 or new["status"] != "COMPLETE":
            raise ValueError("training did not reach original 1500 updates")
        for checkpoint in new["checkpoints"]:
            if sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
                raise ValueError("trained checkpoint content changed")
        run_root = Path(new["checkpoints"][0]["path"]).parents[3]
        plan = read_json(run_root / "training/sample_plan_seed45.json")
        if identity_digest(plan) != new["sample_plan_sha256"]:
            raise ValueError("saved training draws differ from recorded sample plan identity")
        training[arm] = {"same_initialization_and_sample_plan": True, "updates": 1500,
                         "label_identity_sha256": new["label_identity_sha256"]}
    historical = {}
    with (OLD_PUBLIC / "evaluation/CAL_ALL_CHECKPOINTS.csv").open() as stream:
        for row in csv.DictReader(stream):
            historical["CAL", row["module"], int(row["step"])] = {h: float(row[f"{h}_AP"]) for h in ("T1", "T2")}
    with (OLD_PUBLIC / "evaluation/SEL_FIXED_CHECKPOINTS.csv").open() as stream:
        for row in csv.DictReader(stream):
            historical["SEL", row["module"], int(row["selected_step"])] = {h: float(row[f"SEL_{h}_AP"]) for h in ("T1", "T2")}
    old_base = read_json(OLD_PUBLIC / "BASELINE.json")
    for row in q + m:
        expected = historical[row["role"], row["arm"], row["locked_step"]]
        if any(abs(row["original"]["metrics"][h] - expected[h]) > 1e-8 for h in ("T1", "T2")):
            raise ValueError("historical locked head metric was not reproduced")
    for row in q:
        if row["baseline"]["metrics"] != old_base[row["role"]]["metrics"]:
            raise ValueError("historical B0 metric was not reproduced")
    events = [json.loads(line) for line in (PUBLIC / "BUDGET_LEDGER.jsonl").read_text().splitlines()]
    if len({r["event_id"] for r in events}) != len(events):
        raise ValueError("duplicate budget event")
    spent = sum(r["gpu_hours"] for r in events)
    inherited = read_json(PUBLIC / "BUDGET_CONTRACT.json")["inherited_gpu_hours"]
    if spent > 4 or inherited + spent > 192:
        raise ValueError("campaign budget exceeded")
    report = {"status": "CONTROLLED_RUNS_VERIFIED", "metric_results": verified, "training": training,
              "noaug_input_parity": input_parity,
              "cost": {"campaign_gpu_hours": spent, "cumulative_gpu_hours": inherited + spent,
                       "failed_or_interrupted_commands_included": True},
              "manual_requirement_review": "REQUIREMENT_REVIEW.md"}
    write_json(PUBLIC / "VERIFIED_RESULTS.json", report)
    lines = ["# 0929 code-first audit", "", "Fixed-input label repair and independent original-head input-condition verification.",
             "No checkpoint reselection, Q+M combination, or method grid was performed.", "",
             "## Fixed-input repair (AP percentage points)", "",
             "| Role | Arm | Original T1 | Updated T1 | ΔT1 | Original T2 | Updated T2 | ΔT2 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in q + m:
        old = row["original"]["metrics"]
        new = row["repaired" if row["arm"].startswith("Q") else "expanded_eligibility"]["metrics"]
        values = [old["T1"], new["T1"], new["T1"] - old["T1"], old["T2"], new["T2"], new["T2"] - old["T2"]]
        lines.append(f"| {row['role']} | {row['arm']} | " + " | ".join(f"{100 * v:.4f}" for v in values) + " |")
    lines += ["", "M0/M1 are a separate expanded-eligibility control; the Q comparison preserved all original M assignments.",
              "", "## Original locked heads without training augmentation", "",
              "| Role | Arm | T1 AP (%) | ΔT1 vs own B0 (pp) | T2 AP (%) | ΔT2 vs own B0 (pp) |",
              "|---|---|---:|---:|---:|---:|"]
    bases = {r["role"]: r["metrics"] for r in noaug if r["module"] == "B0"}
    for row in noaug:
        value, base = row["metrics"], bases[row["role"]]
        values = [value["T1"], value["T1"] - base["T1"], value["T2"], value["T2"] - base["T2"]]
        lines.append(f"| {row['role']} | {row['module']} | " + " | ".join(f"{100 * v:.4f}" for v in values) + " |")
    lines += ["", "Historical results remain fixed-augmentation development screening. Neither condition is untouched test evidence.",
              "Newly repaired heads were tested at seed45 only; these results do not establish replicated gains.",
              "The score cross-tab demonstrates uplift of affected candidates, not sole causation of historical AP degradation.",
              "Repair effects are the measured matched differences above; no claim of general method invalidity follows.",
              "", f"Measured campaign cost, including interrupted evaluation: {spent:.6f} GPU-hours; cumulative {inherited + spent:.6f}.",
              "", "Source artifacts: RELABEL_AUDIT.json, ORIGINAL_Q_SCORE_CROSSTAB.json, Q_MATCHED_COMPARISON.json,",
              "M_MATCHED_COMPARISON.json, NOAUG_FIXED_HEADS.json, VERIFIED_RESULTS.json and TEST_NOTES.md.", ""]
    (PUBLIC / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(report["cost"], indent=2))


if __name__ == "__main__":
    main()
