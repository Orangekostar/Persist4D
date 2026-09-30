"""Evidence-bound reporting for the empty budget-authorized retraining plan."""

import csv
import json
import zipfile

from omegaconf import OmegaConf

from scripts.native_long_campaign import (
    ARTIFACTS,
    PROJECT,
    code_identity,
    compose_config,
)
from scripts.short_module_screen import read_json, sha256, write_json


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("do not fabricate an empty result CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_report(root):
    state = read_json(root / "RUN_STATE.json")
    if any(entry["seed45_updates"] for entry in state["arms"].values()):
        raise ValueError("budget-limited development report cannot overwrite trained results; complete trained-result stage integration first")
    population = read_json(root / "POPULATION.json")
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    events = [json.loads(line) for line in (root / "COST_LEDGER.jsonl").read_text().splitlines()]
    campaign_hours = sum(r["gpu_hours"] for r in events)
    lifetime = plan["prior"]["gpu_hours"] + campaign_hours
    current_plan = {**plan, "actual_after_preflight": {"campaign_gpu_hours": campaign_hours,
                   "lifetime_gpu_hours": lifetime, "remaining_gpu_hours": 192 - lifetime}}
    write_json(root / "RESOURCE_PLAN.json", current_plan)
    write_json(ARTIFACTS / "RESOURCE_PLAN.json", current_plan)
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root)
    (ARTIFACTS / "CONFIG_RESOLVED.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    costs = read_json(root / "resources/COST_MEASUREMENTS.json")
    arm_rows = []
    for arm, entry in state["arms"].items():
        arm_rows.append({"arm": arm, "seed": 45, "optimizer_updates": entry["seed45_updates"],
                         "required_full_updates": 29700, "status": entry["status"],
                         "initialization": "new encoder-only common seed45", "T1_AP": None,
                         "T2_t_AP": None, "T3_t_AP": None, "T4_t_AP": None, "T5_t_AP": None,
                         "delta_vs_E0": None, "delta_vs_E1": None,
                         "curriculum": "ScanNet T1 + rio T2" if arm == "E0" else "ScanNet T1 + rio T2-T5",
                         "sampling": "stage-stratified" if arm == "E2" else "native capped",
                         "mask_feature_feedback": arm == "E3", "training_seed46": "NOT_REPLICATED"})
    write_csv(ARTIFACTS / "training/ARM_STATUS.csv", arm_rows)
    coverage = {"status": "NOT_RUN_BUDGET" if not plan["full_arms"] else "CHECK_RESULTS",
                "arms": {arm: {role: {f"T{h}": {"completed": 0,
                    "expected": population["evaluation_counts"][role][f"T{h}"], "AP": None}
                    for h in range(1, 6)} for role in ("CAL", "SEL")} for arm in state["arms"]}}
    write_json(ARTIFACTS / "evaluation/COVERAGE.json", coverage)
    for arm in state["arms"]:
        if not state["arms"][arm]["seed45_updates"]:
            write_json(ARTIFACTS / "training" / arm / "seed45/checkpoint_manifest.json", {
                "arm": arm, "seed": 45, "updates": 0, "status": state["arms"][arm]["status"],
                "checkpoints": [], "temporary_preflight_updates_are_not_training": True})
            cfg = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm)
            write_json(ARTIFACTS / "training" / arm / "seed45/resolved_config.json",
                       OmegaConf.to_container(cfg, resolve=True))
    role_counts = read_json(PROJECT / "artifacts/perception_gain_v2/INPUT_MANIFEST.json")["role_counts"]
    write_json(ARTIFACTS / "confirmation/COVERAGE.json", {"status": "NOT_RUN_NO_TRAINED_MODELS",
        "formal_sources": {"INPUT_MANIFEST.json": sha256(PROJECT / "artifacts/perception_gain_v2/INPUT_MANIFEST.json"),
                           "DATA_ROLES.json": sha256(PROJECT / "artifacts/perception_gain_v2/DATA_ROLES.json")},
        "reference_counts": {r: role_counts[r] for r in ("PB", "LOCAL-T2", "ADDITIONAL")},
        "completed": 0, "input_denominators": None,
        "reason": "no lockable trained checkpoint; formal input materialization and evaluation not executed"})
    write_json(ARTIFACTS / "selection/SHORTLIST.json", {"status": "NO_TRAINED_COMPARISON",
        "confirmed": [], "provisional": [], "gain": None, "replication": "NOT_REPLICATED",
        "reason": "empty budget-authorized full trajectory set"})
    write_json(ARTIFACTS / "selection/LOCK_STATUS.json", {"CAL": "NOT_LOCKED_NO_TRAINED_TRAJECTORIES",
        "SEL": "NOT_RUN_BUDGET", "FORMAL": "NOT_LOCKED_NO_TRAINED_MODELS"})
    sampling, feedback = [], []
    for measurement in costs["measurements"]:
        for row in measurement["sampling"]:
            sampling.append({"scope": "TEMPORARY_PREFLIGHT", "arm": measurement["arm"],
                "horizon": measurement["horizon"], "repeat": measurement["repeat"],
                "execution_stage": row["execution_stage"], "available": row["available"],
                "selected": row["selected"], "stages": json.dumps(row["stages"]),
                "counts": json.dumps(row["counts"]), "stratified": row["stratified"]})
        if measurement["arm"] == "E3":
            feedback.append({"scope": "TEMPORARY_PREFLIGHT", "horizon": measurement["horizon"],
                "repeat": measurement["repeat"], "update_norm": measurement["feature_update"][0]["update_norm"],
                "qkv_grad_norm": measurement["gradients"]["model.native_long_feedback.attention.in_proj_weight"],
                "out_proj_grad_norm": measurement["gradients"]["model.native_long_feedback.attention.out_proj.weight"],
                "task_head_grad_norm": measurement["gradients"]["model.mask_embed_head.0.weight"]})
    write_csv(ARTIFACTS / "diagnostics/SAMPLING.csv", sampling)
    write_csv(ARTIFACTS / "diagnostics/FEATURE_UPDATE.csv", feedback)
    write_json(ARTIFACTS / "diagnostics/COVERAGE.json", {"sampling": "REAL_PREFLIGHT_ONLY",
        "feature_feedback": "REAL_PREFLIGHT_ONLY", "nested_prefix_errors": "NOT_RUN_NO_TRAINED_MODELS",
        "selected_model_order_AP": "NOT_RUN_NO_SELECTED_MODEL", "profile": "NOT_RUN_NO_TRAINED_MODELS"})
    evidence = sorted((ARTIFACTS / "evidence").glob("*_official_state.json"))
    if not evidence:
        raise ValueError("real official metric sufficient statistics are missing")
    with zipfile.ZipFile(ARTIFACTS / "evidence/official_metric_states.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in evidence:
            archive.write(path, arcname=path.name)
    audit = read_json(ARTIFACTS / "runtime/ORDER_AUDIT.json")
    runtime_note = (
        "# Native Runtime Note\n\n"
        f"Status: {audit['status']}; repeats within each fixed A/BA/ABA protocol are identical. "
        "Input, inverse/segment mappings and embedding are identical across orders. "
        "The first localized difference is encoder enc0.block0.cpe.0 sparse convolution. "
        "A Native sparse-algorithm probe also retained history sensitivity; that algorithm is not enabled.\n\n"
        "Use a fresh process and sorted (horizon,input_id) evaluation order, per-input seed hash(9001,input_id), "
        "FP32, TF32 off, highest matmul precision. No 94-input replay. Training keeps frozen encoder modules in train mode.\n\n"
        "The initialized TRAIN A probe yields t-AP=0 in A, BA and ABA, but BA changes candidate lineage. "
        "This single-input diagnostic does not resolve trained-model accuracy or confirm any gain. "
        "Official recall can be undefined at initialization; no NaN was replaced by zero.\n\n"
        "Largest complete T5: batch2 OOM both before and with exact non-reentrant F chunk checkpointing; "
        "batch1 complete backward/update passed. All arms use batch1/rank, world2, accumulation16. "
        "The fixed OneCycle horizon and effective batch32 remain unchanged.\n"
        "\nResource deviation: two early CPU resume fixtures called a legacy all-device RNG helper and "
        "initialized three visible GPU contexts. Conservatively charged 3 cards times their full 5.52s/6.28s "
        "process durations. The new hook captures only its rank-local GPU, and final CPU checks hide CUDA.\n"
    )
    (ARTIFACTS / "RUNTIME_NOTE.md").write_text(runtime_note)
    bindings = (
        "# Code Bindings\n\n"
        "| Requirement | Implementation | Actual evidence |\n|---|---|---|\n"
        "| Encoder-only/task-fresh | models/pointcept.py::_load_state_dict | SOURCE_AND_INITIALIZATION.json |\n"
        "| Real population/draw/RNG | datasets/native_long_dataset.py | POPULATION.json; DATA_CONTENT_BINDING.json |\n"
        "| LOW/FULL inverse correctness | datasets/pointcept_utils.py | initialization/REAL_MIXED_BATCH.json; collation regression |\n"
        "| Fixed-count stage sampling | models/native_long_modules.py; models/rescene.py | diagnostics/SAMPLING.csv |\n"
        "| Late query-conditioned F | models/native_long_modules.py; ReScene.update_mask_features | REAL_FEEDBACK_IDENTITY.json; FEATURE_UPDATE.csv |\n"
        "| Boundary resume | trainer/native_long_trainer.py | actual Lightning split-resume fixture, draw/LR/tensor equality |\n"
        "| raw_sum/12aux/backward | scripts/native_long_runtime.py::training_measurement | resources/COST_MEASUREMENTS.json |\n"
        "| Budget/no doomed training | native_long_campaign.freeze_cost_plan | RESOURCE_PLAN.json; RUN_STATE.json |\n"
        "| Checkpoint CAL/SEL | scripts/native_long_execution.py | selection regression only; real trajectories not run |\n"
        "| Exact task/buffer loader | scripts/native_long_assets.py | initialization/TASK_BUNDLE_RELOAD_AUDIT.json |\n\n"
        "Seed46 pair execution, formal-confirmation integration and selected-model profiling remain unexecuted "
        "and are not certified by this budget-limited delivery. The current authorized empty-plan CLI is exercised.\n"
    )
    (ARTIFACTS / "CODE_BINDINGS.md").write_text(bindings)
    requirements = [
        (0, "PARTIAL_BUDGET", "Four independent implementations; zero authorized full trajectories, no gain claim."),
        (1, "PASS_BINDING", "Exact base, mandatory historical materials and fixed encoder/R1 SHA; isolated branch."),
        (2, "PASS_DEVELOPMENT", "Code bindings and actual preflight evidence; training effects remain unmeasured."),
        (3, "PASS_POPULATION", "361 TRAIN, 8 CAL, 8 SEL; unique scans/global IDs/paired draws, T5 available."),
        (4, "PASS_INITIALIZATION", "New seed45 common state, encoder-only, frozen train mode, raw_sum scalar/gradient and 12aux."),
        (5, "RUNTIME_CONDITIONAL", "Six fresh-process A/BA/ABA probes; history sensitivity; complete largest backward and OOM fallback."),
        (6, "PASS_PREFLIGHT_ONLY", "S activates all 12 T5 capped layers with fixed totals; extras/T1/no-cap/padding covered."),
        (7, "PASS_PREFLIGHT_ONLY", "Initial identity, two-step real gradients, exactly later four outputs/final mask change; chunk/checkpoint parity."),
        (8, "PARTIAL_BUDGET", "U/scheduler/global batch retained; actual Lightning split-resume fixture passed, native full training not run."),
        (9, "NOT_RUN_BUDGET", "No CAL lock/SEL/seed46/formal confirmation; exact internal expected denominators and null scores."),
        (10, "PASS_PLAN_WITH_FIXED_DEVIATION", "Prior/failures/CPU-test CUDA overcapture charged; serial IO and reference-size/curriculum integrated; empty mode locked before results."),
        (11, "PARTIAL_PREFLIGHT", "Actual temporary S/F diagnostics; trained nested failures/order AP/full deployment profile deferred."),
        (12, "PASS_DEVELOPMENT_REVIEW", "Focused regression/lint/compile/diff checks recorded separately; no scientific PASS inferred from tests."),
        (13, "PARTIAL_STAGE_SUPPORT", "run/status/report/publish current budget path tested; seed45 Lightning/CAL/SEL path provided but not real-run validated; later positive-plan stages need validation."),
        (14, "CODE_ONLY", "A/B/tag and verified Git receipt; F initialization is small Git asset, task initialization package remains local without Release authorization."),
        (15, "PASS_EVIDENCE_BOUNDARY", "S/F are unconfirmed engineering hypotheses; no reproduction/official-performance claim."),
    ]
    review = "# Requirement Review\n\nPrimary-agent review against EXECUTION_INSTRUCTION.md sections 0-15.\n\n"
    review += "| Section | Status | Evidence and boundary |\n|---|---|---|\n"
    review += "".join(f"| {number} | {status} | {detail} |\n" for number, status, detail in requirements)
    review += "\nThis is a budget-limited development delivery, not completion of the four-arm retraining study. "
    review += "No full training, CAL/SEL, replication, formal confirmation or trained-model profile is marked PASS.\n"
    (ARTIFACTS / "REQUIREMENT_REVIEW.md").write_text(review)
    report = (
        "# Native Long Retrain Result\n\n"
        "Scientific status: **PARTIAL_FULL_RETRAIN_BUDGET**. Publication: **CODE_ONLY**. "
        "No trained accuracy result or module gain is available.\n\n"
        "| Arm | Actual optimizer updates | Status | T1-T5 official AP | Net gain |\n|---|---:|---|---|---|\n"
        + "".join(f"| {r['arm']} | {r['optimizer_updates']} / 29700 | {r['status']} | Not evaluated | Not measured |\n" for r in arm_rows)
        + "\nAll four share a new seed45 task initialization with the fixed pretrained embedding/encoder. "
        "E0 uses short T2; E1 long curriculum; E2 long+S; E3 long+F. No combination or old task warm start. "
        "Temporary optimizer updates only measure feasibility/cost and do not count toward these trajectories.\n\n"
        f"Budget mode: **{plan['mode']}**, authorized full arms `{plan['full_arms']}`. "
        f"Prior {plan['prior']['gpu_hours']:.9f} GPUh; this campaign {campaign_hours:.9f} GPUh; "
        f"lifetime {lifetime:.9f}/192 GPUh; current remainder {192 - lifetime:.9f} GPUh. "
        "At least 8 GPUh is reserved.\n\n"
        "Full forecasts include the single 1.25 training margin and separate CAL proxy:\n\n"
        + "".join(f"- {arm}: {hours:.2f} GPUh.\n" for arm, hours in plan["forecast"]["costs_gpu_hours"].items())
        + "\nEven E0 plus the reserve exceeds the available lifetime budget, so section10 forbids starting it. "
        "Forecasts are finite measurements with approximate point-count scaling, not promised runtimes.\n\n"
        "Recommended configuration from single-A40 preflight: two A40, batch1/rank, accumulation16, effective batch32, 0 workers, "
        "FP32/TF32 off, voxel0.02, Q100, U29700, AdamW5e-4 and fixed full OneCycle. "
        "Batch2 remained OOM after exact F checkpointing; largest T5 batch1 used "
        f"{costs['feasibility'][-1]['peak_allocated_bytes'] / 1024**3:.2f} GiB allocated at peak. Full native DDP has not been exercised.\n\n"
        "Population is TRAIN361/CAL8/SEL8. Complete expected input counts T1..T5 are "
        f"CAL `{list(population['evaluation_counts']['CAL'].values())}`, SEL `{list(population['evaluation_counts']['SEL'].values())}`. "
        "All completed counts are zero for main evaluation; see evaluation/COVERAGE.json. "
        "No seed46 pair, formal lock, PB/LOCAL/ADDITIONAL accuracy, trained nested-prefix diagnosis or deployment profile exists.\n\n"
        "Real development evidence: corrected mixed-H FULL inverse-map slicing; exact common tensors; "
        "F initial identity and late output changes; zero first qkv gradient then nonzero second gradient; "
        "S fixed-cap stage quotas; raw_sum scalar/gradient equality; actual largest backward and IO timing. "
        "Native status remains RUNTIME_CONDITIONAL. The initialized TRAIN A diagnostic has zero AP in all orders "
        "while BA changes candidate lineage; this is not evidence of a trained-model gain.\n\n"
        f"New common state SHA: `{source['initialization']['45']['sha256']}`. R1 is historical-only and "
        "was not reevaluated as a common-runtime bridge, so no historical AP comparison is claimed.\n\n"
        "Git includes the F initialization state and compressed official initialized TRAIN-A sufficient statistics. "
        "No trained model was generated. The genuine nonencoder initialization task package is local-only; "
        "Release authorization was unavailable at the single startup check. Full common state and raw traces remain local. "
        "No raw data/GT or original pretrained checkpoint is redistributed. "
        "See HANDOFF.md, ARTIFACT_MANIFEST.json and the external publication receipt for exact delivery.\n"
    )
    (ARTIFACTS / "FINAL_REPORT.md").write_text(report)
    state.update(stage="BUDGET_LIMITED_DEVELOPMENT_DELIVERY", scientific_status="PARTIAL_FULL_RETRAIN_BUDGET",
                 publication_status="CODE_ONLY", code=code_identity(), actual_campaign_gpu_hours=campaign_hours,
                 lifetime_gpu_hours=lifetime, required_full_training_completed=False)
    write_json(root / "RUN_STATE.json", state)
    write_json(ARTIFACTS / "RUN_STATE.json", state)
    return {"report": str(ARTIFACTS / "FINAL_REPORT.md"), "state": state}
