"""Bound native retraining campaign; every stage reports actual evidence."""

import argparse
import collections
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import yaml

from datasets.native_long_dataset import assign_development_roles, evaluation_inputs
from scripts.native_long_budget import budget_cap
from scripts.short_module_screen import read_json, sha256, write_json

PROJECT = Path(__file__).resolve().parents[1]
BASE = "c6e9d01edfe2b3c832374a424fc45e44bdadfb68"
ARTIFACTS = Path(os.environ.get("RESCENE_NATIVE_LONG_ARTIFACTS",
                               PROJECT / "artifacts/native_long_retrain_v1")).resolve()


def choose_budget_mode(costs, *, remaining, reserve):
    c0, c1, cs, cf = (costs[arm] for arm in ("E0", "E1", "E2", "E3"))
    candidates = (
        ("FULL_FOUR", c0 + c1 + cs + cf, ["E0", "E1", "E2", "E3"], []),
        ("SCREEN_ONE", c0 + c1 + .4 * (cs + cf) + .6 * max(cs, cf),
         ["E0", "E1"], ["E2", "E3"]),
        ("PRIORITY_F", c0 + c1 + cf, ["E0", "E1", "E3"], []),
    )
    for mode, cost, arms, pilot in candidates:
        if cost + reserve <= remaining:
            return {"mode": mode, "forecast_training_gpu_hours": cost,
                    "full_arms": arms, "pilot_arms": pilot, "reserve_gpu_hours": reserve}
    arms = ["E0"] if c0 + reserve <= remaining else []
    return {"mode": "BASELINE_RECOVERY_ONLY", "full_arms": arms, "pilot_arms": [],
            "forecast_training_gpu_hours": c0 if arms else 0., "reserve_gpu_hours": reserve,
            "scientific_status": "PARTIAL_FULL_RETRAIN_BUDGET"}


def forecast_costs(measured, io, population):
    import numpy as np

    from datasets.native_long_dataset import horizon_weights

    batch = measured["per_rank_batch"]
    base, points, overhead_s, overhead_f = {}, {}, 0., {}
    for h in range(1, 6):
        rows = [r for r in measured["measurements"] if r["arm"] == "E1" and r["horizon"] == h]
        times = [r["seconds"] for r in rows if not r["raw_sum_audit"]]
        if not times:
            raise ValueError("no untampered native timing for horizon")
        base[h] = float(np.mean(times)) / batch
        points[h] = float(np.mean([r["full_points"] for r in rows])) / batch
        base[h] += float(np.mean([r["total_seconds"] for r in io["rows"] if r["horizon"] == h])) / batch
    s = [r["seconds"] for r in measured["measurements"] if r["arm"] == "E2"]
    e1_5 = [r["seconds"] for r in measured["measurements"] if r["arm"] == "E1" and r["horizon"] == 5]
    overhead_s = max(0., float(np.mean(s) - np.mean(e1_5))) / batch
    for h in (2, 5):
        f = [r["seconds"] for r in measured["measurements"] if r["arm"] == "E3" and r["horizon"] == h]
        baseline = [r["seconds"] for r in measured["measurements"] if r["arm"] == "E1"
                    and r["horizon"] == h and not r["raw_sum_audit"]]
        overhead_f[h] = max(0., float(np.mean(f) - np.mean(baseline))) / batch
    groups = {int(h): item for h, item in measured["point_distributions"].items()}
    train = [r for r in population["references"] if r["role"] == "TRAIN"]
    mass = {h: 0. for h in range(1, 6)}
    mass[1] = 4 / 9
    point_mass = {h: 0. for h in mass}
    point_mass[1] = mass[1] * groups[1]["mean_full_points"]
    # Integrate the prescribed curriculum after per-reference unavailable-H normalization.
    for update in np.linspace(0, 29700, 601, endpoint=False) + 29700 / 1202:
        weights = horizon_weights(float(update))
        for ref in train:
            available = range(2, min(5, ref["Tmax"]) + 1)
            denominator = sum(weights[h] for h in available)
            for h in available:
                probability = 5 / 9 / 601 / len(train) * weights[h] / denominator
                mass[h] += probability
                point_mass[h] += probability * h * io["mean_scan_points_by_train_reference"][ref["reference_id"]]
    scale = {h: groups[h]["mean_full_points"] / points[h] for h in groups}
    common_t1 = base[1] * point_mass[1] / points[1]
    per_draw = {"E0": common_t1 + 5 / 9 * base[2] * scale[2]}
    per_draw["E1"] = common_t1 + sum(base[h] * point_mass[h] / points[h] for h in range(2, 6))
    per_draw["E2"] = per_draw["E1"] + sum(overhead_s * point_mass[h] / points[h] for h in range(2, 6))
    per_draw["E3"] = per_draw["E1"] + sum(
        (overhead_f[2] + (h - 2) / 3 * (overhead_f[5] - overhead_f[2])) * point_mass[h] / points[h]
        for h in range(2, 6)) + mass[1] * overhead_f[2] * scale[1]
    training = {arm: seconds * 950400 / 3600 * 1.25 for arm, seconds in per_draw.items()}
    cal_proxy = {arm: 10 * sum(population["evaluation_counts"]["CAL"][f"T{h}"]
                              * (base[h] + (overhead_s if arm == "E2" else max(overhead_f.values())
                                            if arm == "E3" else 0.)) * scale[h] / 3600
                              for h in range(1, 6)) for arm in training}
    return {"costs_gpu_hours": {arm: training[arm] + cal_proxy[arm] for arm in training},
            "training_gpu_hours_with_single_1_25_margin": training,
            "cal_gpu_hours_separate_conservative_full_step_proxy": cal_proxy,
            "per_draw_gpu_occupancy_seconds_including_serial_IO": per_draw,
            "integrated_horizon_mass": mass, "mean_to_measured_point_scaling": scale,
            "integrated_points_per_draw_by_horizon": point_mass,
            "limitations": ["finite median native inputs; linear full-point scaling is approximate",
                            "two ranks may have imbalance; 25% training margin applied once",
                            "raw_sum gradient audit timing excluded from speed estimate"]}


def freeze_cost_plan(root):
    measured = read_json(root / "resources/COST_MEASUREMENTS.json")
    memory = read_json(root / "resources/CHECKPOINT_MEMORY_PROBE.json")
    if memory["status"] == "PASS" and measured["per_rank_batch"] == 1:
        raise ValueError("batch2 checkpoint success requires a batch2 cost measurement before planning")
    # JSON stores horizon keys as strings; compare that same representation on resume.
    forecast = json.loads(json.dumps(forecast_costs(
        measured, read_json(root / "resources/IO_MEASUREMENTS.json"),
        read_json(root / "POPULATION.json")), allow_nan=False))
    plan = read_json(root / "RESOURCE_PLAN.json")
    used = sum(json.loads(line)["gpu_hours"] for line in (root / "COST_LEDGER.jsonl").read_text().splitlines())
    remaining = budget_cap(plan) - plan["prior"]["gpu_hours"] - used
    selected = choose_budget_mode(forecast["costs_gpu_hours"], remaining=remaining,
                                  reserve=plan.get("reserve_gpu_hours", 8.))
    frozen = {**plan, **selected, "forecast": forecast, "current_campaign_gpu_hours": used,
              "remaining_at_plan_gpu_hours": remaining,
              "physical_batch": {"world_size": measured["world_size_planned"],
                                 "per_rank_batch": measured["per_rank_batch"],
                                 "accumulation": measured["accumulation"], "effective": 32},
              "frozen_before_training_results": True,
              "measurement_sha256": sha256(root / "resources/COST_MEASUREMENTS.json"),
              "IO_measurement_sha256": sha256(root / "resources/IO_MEASUREMENTS.json")}
    path = root / "selection/BUDGET_LOCK.json"
    if path.exists():
        prior = read_json(path)
        if any(prior.get(k) != frozen.get(k) for k in ("mode", "full_arms", "pilot_arms", "forecast", "physical_batch")):
            state = read_json(root / "RUN_STATE.json")
            if any(r["seed45_updates"] for r in state["arms"].values()) or (root / "evaluation/CAL_ALL.json").exists():
                raise ValueError("budget lock changed; reconcile real overruns explicitly")
            frozen["pre_result_forecast_correction"] = {
                "prior_lock_sha256": sha256(path),
                "reason": "preserve reference-size/curriculum correlation in metadata cost integration",
                "training_updates_seen": 0, "selection_scores_seen": 0}
        else:
            frozen = prior
    write_json(path, frozen)
    write_json(root / "RESOURCE_PLAN.json", frozen)
    write_json(ARTIFACTS / "RESOURCE_PLAN.json", frozen)
    state = read_json(root / "RUN_STATE.json")
    state.update(stage="BUDGET_FROZEN", budget_mode=selected["mode"])
    if not selected["full_arms"] and not selected["pilot_arms"]:
        state["scientific_status"] = "PARTIAL_FULL_RETRAIN_BUDGET"
        for entry in state["arms"].values():
            entry.update(status="NOT_RUN_BUDGET", seed45_updates=0)
    write_json(root / "RUN_STATE.json", state)
    write_json(ARTIFACTS / "RUN_STATE.json", state)
    return frozen


def compose_config(path, *, root, arm="E0", seed=45):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    path = Path(path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        config = compose(config_name=path.stem)
    with open_dict(config):
        config.general.seed = seed
        config.native_long.arm = arm
        config.native_long.runtime_root = str(root)
        config.general.save_dir = str(root / "training" / arm / f"seed{seed}")
        config.model.native_long_sampling = arm == "E2"
        config.model.native_long_feedback = arm == "E3"
        config.native_long.population_file = str(root / "POPULATION.json")
        worker = root / "WORKER_SPEC.json"
        if worker.exists():
            spec = read_json(worker)
            config.native_long.data_root = spec["data_root"]
            config.backbone.name = spec["encoder"]
        config.instance_metric.dataset = str(Path(config.native_long.data_root) / "processed/rio/rio.yaml")
        lock = root / "selection/BUDGET_LOCK.json"
        if lock.exists():
            physical = read_json(lock)["physical_batch"]
            config.general.gpus = physical["world_size"]
            config.data.batch_size = physical["per_rank_batch"]
            config.trainer.accumulate_grad_batches = physical["accumulation"]
            config.native_long.feedback_checkpoint = False
    OmegaConf.resolve(config)
    if (config.scheduler.scheduler.total_steps != 29700
            or config.general.rootcause_objective_mode != "raw_sum"
            or config.backbone.pretrained_scope != "encoder_only"
            or config.trainer.max_steps != 29700):
        raise ValueError("training identity differs from the fixed full protocol")
    return config


def code_identity():
    paths = [PROJECT / p for p in ("models/rescene.py", "models/pointcept.py",
             "models/native_long_modules.py", "models/criterion.py", "models/matcher.py",
             "datasets/native_long_dataset.py", "datasets/semseg.py", "datasets/pointcept_utils.py",
             "trainer/trainer.py", "trainer/native_long_trainer.py", "scripts/native_long_campaign.py",
             "scripts/native_long_budget.py", "scripts/native_long_cluster.py",
             "scripts/native_long_runtime.py", "scripts/native_long_execution.py", "scripts/native_long_assets.py",
             "scripts/rescene_task_postprocess.py", "scripts/p6a_metrics.py", "scripts/evaluate_persist4d_p6a.py",
             "conf/config_native_long_retrain.yaml")]
    return {str(p.relative_to(PROJECT)): sha256(p) for p in paths if p.is_file()}


def bind(root, config, base):
    actual = subprocess.check_output(["git", "rev-parse", f"{base}^{{commit}}"],
                                     cwd=PROJECT, text=True).strip()
    if actual != BASE:
        raise ValueError("base differs from the required native retrain source")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=PROJECT, check=True)
    assets = read_json(Path.home() / "persist4d_runs/perception_gain_v2/assets.local.json")
    binding = root / "SOURCE_AND_INITIALIZATION.json"
    existing = read_json(binding) if binding.exists() else {}
    weights = {}
    for name, expected, size in (
        ("concerto_pretrained", "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07", 433987358),
        ("r1_checkpoint", "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd", 754813672),
    ):
        path = Path(assets[name])
        stat = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        prior = existing.get("weights", {}).get(name, {})
        digest = prior.get("sha256") if prior.get("stat") == stat else sha256(path)
        if digest != expected or stat["bytes"] != size:
            raise ValueError(f"bound weight mismatch: {name}")
        weights[name] = {"path": str(path), "sha256": digest, "stat": stat,
                         "purpose": "pretrained_encoder" if name == "concerto_pretrained" else "historical_only"}
    sources = {}
    for package in ("concerto", "sonata", "stmetrics", "torch", "pytorch_lightning"):
        spec = importlib.util.find_spec(package)
        if spec is None:
            raise RuntimeError(f"required installed library unavailable: {package}")
        sources[package] = {"source": spec.origin, "source_sha256": sha256(Path(spec.origin))}
        if package in {"concerto", "sonata", "stmetrics"}:
            folder = Path(spec.origin).parent
            sources[package]["python_source_sha256"] = {
                str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*.py"))}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "pytorch-lightning")}
    clip_path = PROJECT / "artifacts/rescene_task_learning_root_cause_v1/audit/runtime_migration_2gpu/root_variant_manifest.json"
    clip = float(read_json(clip_path)["variants"]["R1"]["resolved_config"]["trainer"]["gradient_clip_val"])
    if float(config.trainer.gradient_clip_val) != clip:
        raise ValueError("gradient clipping differs from the full R1 resolved recipe")
    result = {"base_commit": BASE, "code": code_identity(), "weights": weights,
              "installed_libraries": sources, "versions": versions,
              "initialization": existing.get("initialization", {}),
              "gradient_clip_val": clip,
              "gradient_clip_source": {"file": str(clip_path.relative_to(PROJECT)), "sha256": sha256(clip_path),
                                       "field": "variants.R1.resolved_config.trainer.gradient_clip_val"},
              "assets": assets}
    write_json(binding, result)
    write_json(ARTIFACTS / binding.name, result)
    return assets


def bind_data_content(root, config):
    from datasets.native_long_dataset import build_native_datasets

    destination = root / "DATA_CONTENT_BINDING.json"
    prior = read_json(destination) if destination.exists() else {"files": []}
    by_path = {row["path"]: row for row in prior["files"]}
    files = set()
    for dataset in build_native_datasets(config, config.native_long.data_root, augmentation=False).values():
        for row in dataset.data:
            for key in ("filepath", "raw_instance_filepath"):
                if key in row and Path(row[key]).is_file():
                    files.add(Path(row[key]))
    rows = []
    for path in sorted(files):
        stat = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        previous = by_path.get(str(path))
        digest = previous["sha256"] if previous and previous["stat"] == stat else sha256(path)
        if previous and previous["sha256"] != digest:
            raise ValueError("bound training input/labels changed; new trajectory and GT audit required")
        rows.append({"path": str(path), "sha256": digest, "stat": stat})
    payload = {"schema": 1, "policy": "whole-file content SHA; cached only for unchanged bytes/mtime", "files": rows}
    write_json(destination, payload)
    write_json(ARTIFACTS / destination.name, payload)
    return payload


def verify_data_content(root):
    for row in read_json(root / "DATA_CONTENT_BINDING.json")["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size != row["stat"]["bytes"]:
            raise ValueError("bound data input disappeared or changed bytes")
        if path.stat().st_mtime_ns != row["stat"]["mtime_ns"] and sha256(path) != row["sha256"]:
            raise ValueError("changed native labels/input invalidates the checkpoint and prediction cache")


def freeze_population(root, assets):
    from scripts.qp_mn_binding import ledger_total

    data_root = Path(assets["data_root"])
    db_path = data_root / "processed/rio/train_database.yaml"
    rows = yaml.safe_load(db_path.read_text())
    metadata = read_json(Path(assets["rio_metadata"]))
    aliases = {}
    for index, row in enumerate(rows):
        scan = Path(row["raw_filepath"]).parent.name
        if scan in aliases:
            raise ValueError("processed TRAIN database contains duplicate physical scans")
        path = Path(row["filepath"].replace("../../", ""))
        if not path.is_absolute():
            path = data_root / path.relative_to("data")
        aliases[scan] = {"index": index, "path": path, "alias": Path(row["instance_gt_filepath"]).stem}
    role_path = PROJECT / "artifacts/perception_gain_v2/DATA_ROLES.json"
    old_roles = read_json(role_path)["roles"]
    excluded = collections.defaultdict(list)
    for role in ("CAL", "SEL", "PB", "LOCAL-T2", "ADDITIONAL"):
        for reference in old_roles[role]:
            excluded[reference].append("historical:" + role)
    audit_path = PROJECT / "artifacts/perception_gain_v2/foundation/LOCAL_POPULATION_AUDIT.json"
    for reference in read_json(audit_path)["validation_reference_ids"]:
        excluded[reference].append("historical:LOCAL_VALIDATION")
    candidates, rejected = [], []
    for reference in metadata:
        ref = reference["reference"]
        reasons = list(excluded.get(ref, []))
        if reference["type"] != "train":
            reasons.append("official:" + reference["type"])
        scans = [ref] + [r["reference"] for r in reference["scans"]]
        if len(scans) != len(set(scans)):
            raise ValueError("official metadata repeats a scan within reference")
        missing = [s for s in scans if s not in aliases or not aliases[s]["path"].is_file()]
        legal = [s for s in scans if s not in missing]
        if len(legal) < 2:
            reasons.append("fewer_than_two_readable_scans")
        if reasons:
            rejected.append({"reference_id": ref, "reasons": reasons, "unavailable_scans": missing})
            continue
        candidates.append({"reference_id": ref, "official_split": "train", "Tmax": len(legal),
                           "scan_ids": legal, "scan_aliases": [aliases[s]["alias"] for s in legal],
                           "scan_indices": [aliases[s]["index"] for s in legal],
                           "unavailable_scans": missing,
                           "ambiguity_source": "sequence_database_sliding_2.yaml; reference metadata",
                           "gt_identity_source": "semseg.v2.json annotation id/objectId, preserved processed column11"})
    records = assign_development_roles(candidates)
    counts = {role: {"references": sum(r["role"] == role for r in records),
                     "Tmax_groups": dict(collections.Counter(min(5, r["Tmax"]) for r in records
                                                              if r["role"] == role))}
              for role in ("TRAIN", "CAL", "SEL")}
    legal_long = all(counts[r]["Tmax_groups"].get(5, 0) >= 1 for r in counts)
    legal_long &= all(counts[r]["references"] >= 4 for r in ("CAL", "SEL"))
    inputs = {role: evaluation_inputs(records, role) for role in ("CAL", "SEL")}
    payload = {"status": "READY" if legal_long else "BLOCKED_LONG_POPULATION", "references": records,
               "counts": counts, "evaluation_inputs": inputs, "excluded": rejected,
               "historical_role_overlap": {k: v for k, v in excluded.items() if len(v) > 1},
               "exposure": "new task holdouts; historical encoder/research exposure is not excluded",
               "sources": {"official_metadata": sha256(Path(assets["rio_metadata"])),
                           "processed_train_database": sha256(db_path), "old_roles": sha256(role_path),
                           "local_validation_audit": sha256(audit_path)},
               "evaluation_counts": {role: {f"T{h}": sum(r["horizon"] == h for r in inputs[role])
                                             for h in range(1, 6)} for role in inputs}}
    destination = root / "POPULATION.json"
    if (destination.exists() and read_json(destination) != payload
            and any((root / "training").glob("*/seed*/last.ckpt"))):
        raise ValueError("population changed after training; a new trajectory is required")
    write_json(destination, payload)
    write_json(ARTIFACTS / destination.name, payload)
    live = PROJECT.parent
    # All histories share event IDs; inherited subtotals must not be added twice.
    paths = [live / "persist4d-perception-gain-v2/artifacts/perception_gain_v2/budget/LEDGER.jsonl",
             live / "rescene-short-module-screen-v1/artifacts/short_module_screen_v1/BUDGET_LEDGER.jsonl",
             live / "rescene-code-first-audit-v1/artifacts/rescene_code_first_audit_v1/BUDGET_LEDGER.jsonl",
             live / "rescene-qp-mn-targeted-v1/artifacts/qp_mn_targeted_v1/BUDGET_LEDGER.jsonl"]
    prior = ledger_total(paths, subtotal_ids={"inherited-v2-final"})
    plan = {"lifetime_cap_gpu_hours": 192., "prior": prior,
            "remaining_before_preflight_gpu_hours": 192. - prior["gpu_hours"],
            "reserve_confirmation_gpu_hours": 8., "forecast_multiplier": 1.25,
            "mode": "PENDING_REAL_COST_MEASUREMENT"}
    lock_path = root / "selection/BUDGET_LOCK.json"
    if lock_path.exists():
        frozen = read_json(lock_path)
        if frozen["prior"] != prior:
            raise ValueError("historical budget ledger changed after the plan lock")
        plan = frozen
    write_json(root / "RESOURCE_PLAN.json", plan)
    write_json(ARTIFACTS / "RESOURCE_PLAN.json", plan)
    return payload


def prepare(root, config, base):
    from omegaconf import OmegaConf

    root.mkdir(parents=True, exist_ok=True)
    assets = bind(root, config, base)
    population = freeze_population(root, assets)
    bind_data_content(root, config)
    (ARTIFACTS / "CONFIG_RESOLVED.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    if not (root / "ACCESS_CHECK.json").exists():
        from scripts.perception_gain_v2_publication import GitHubReleaseClient

        client = GitHubReleaseClient.authorized()
        write_json(root / "ACCESS_CHECK.json", {"release_authorized": client is not None,
                   "git_remote": "git@github.com:Orangekostar/Persist4D.git"})
        if client:
            client.session.close()
    state = {"stage": "PREPARED", "scientific_status": "IN_PROGRESS",
             "population_status": population["status"], "code": code_identity(),
             "arms": {arm: {"seed45_updates": 0, "status": "NOT_STARTED"} for arm in ("E0", "E1", "E2", "E3")}}
    if (root / "RUN_STATE.json").exists():
        state = {**state, **read_json(root / "RUN_STATE.json")}
    state["code"] = code_identity()
    write_json(root / "RUN_STATE.json", state)
    write_json(ARTIFACTS / "RUN_STATE.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "report", "publish"))
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("RESCENE_NATIVE_LONG_ROOT",
                                                    Path.home() / "persist4d_runs/native_long_retrain_v1")))
    parser.add_argument("--config", type=Path, default=PROJECT / "conf/config_native_long_retrain.yaml")
    parser.add_argument("--base-commit", default=BASE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--upload-assets", action="store_true",
                        help="explicitly retry an initialization-asset upload with existing Release authorization")
    parser.add_argument("--through", choices=("prepare", "preflight", "report", "publish"), default="publish")
    args = parser.parse_args()
    root = args.root.resolve()
    if root == Path.home() or root == PROJECT or root.name in {
            "perception_gain_v2", "qp_mn_targeted_v1", "short_module_screen_v1", "rescene_code_first_audit_v1"}:
        raise ValueError("refusing historical/protected runtime root")
    lock = root / "selection/BUDGET_LOCK.json"
    if lock.exists() and read_json(lock).get("execution_mode") == "SSH_CLUSTER":
        from scripts.native_long_cluster import cluster_command

        if args.command == "publish":
            raise ValueError("cluster development requires its own current review; old empty-plan publisher is historical")
        print(json.dumps(cluster_command(root, args.command, through=args.through), indent=2))
        return
    if args.command == "status":
        print(json.dumps(read_json(root / "RUN_STATE.json"), indent=2))
        return
    config = compose_config(args.config, root=root)
    if args.command == "run":
        state = prepare(root, config, args.base_commit)
        print(json.dumps(state, indent=2), flush=True)
        if args.through != "prepare":
            from scripts.native_long_runtime import run_preflight

            run_preflight(root, config, resume=args.resume)
            freeze_cost_plan(root)
        if args.through in {"report", "publish"}:
            from scripts.native_long_execution import execute_authorized_plan
            from scripts.native_long_report import generate_report

            execute_authorized_plan(root)
            print(json.dumps(generate_report(root), indent=2), flush=True)
        if args.through == "publish":
            from scripts.native_long_publication import publish

            print(json.dumps(publish(root, upload_assets=args.upload_assets), indent=2), flush=True)
    elif args.command == "report":
        from scripts.native_long_report import generate_report

        print(json.dumps(generate_report(root), indent=2))
    elif args.command == "publish":
        from scripts.native_long_publication import publish

        print(json.dumps(publish(root, upload_assets=args.upload_assets), indent=2))


if __name__ == "__main__":
    main()
