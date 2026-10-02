"""Native checkpoint evaluation and optimizer-boundary Lightning execution."""

import math
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from datasets.native_long_dataset import NativeLongCollator, isolated_rng, stable_seed
from scripts.native_long_budget import remaining_budget
from scripts.native_long_budget import write_json_atomic as write_json
from scripts.native_long_campaign import (
    ARTIFACTS,
    PROJECT,
    code_identity,
    compose_config,
    verify_data_content,
)
from scripts.native_long_runtime import (
    build_native_datasets,
    configure_numeric,
    load_samples,
    move_batch,
    system_from_common,
)
from scripts.short_module_screen import append_event, read_json, sha256


def select_cal_points(rows, arms):
    selected = {}
    for arm in arms:
        points = sorted([r for r in rows if r["arm"] == arm and r["seed"] == 45], key=lambda r: r["step"])
        horizons = {f"T{h}" for h in range(1, 6)}
        if not points or any(r["role"] != "CAL" or r["status"] != "COMPLETE"
                             or set(r["expected"]) != horizons or r["completed"] != r["expected"]
                             or any(n < 1 for n in r["expected"].values())
                             or not horizons <= set(r["metrics"])
                             or any(not math.isfinite(v) or not 0 <= v <= 1 for v in r["metrics"].values())
                             for r in points):
            raise ValueError("selection requires a complete, finite official CAL population")
        if any(abs(r["metrics"]["S_long"] - sum(r["metrics"][f"T{h}"] for h in (3, 4, 5)) / 3) > 1e-12
               for r in points):
            raise ValueError("S_long differs from the three pooled official horizon AP values")
        best = points[0]
        keys = ("T2", "T1") if arm == "E0" else ("S_long", "T2", "T1")
        for row in points[1:]:
            for key in keys:
                delta = row["metrics"][key] - best["metrics"][key]
                if abs(delta) > 1e-6:
                    if delta > 0:
                        best = row
                    break
        selected[arm] = best["step"]
    return {"schema": 1, "seed": 45, "selected_updates": selected,
            "all_planned_trajectories_stopped_before_SEL": True, "tolerance": 1e-6}


def rank_pilot(rows):
    lookup = {(r["arm"], r["step"]): r["metrics"] for r in rows if r["status"] == "COMPLETE"}
    ranked = []
    for arm in ("E2", "E3"):
        deltas = [{k: lookup[arm, step][k] - lookup["E1", step][k]
                   for k in ("S_long", "T2")} for step in (8910, 11880)]
        ranked.append({"arm": arm, "guard": all(d["T2"] >= -.002 for d in deltas),
                       "delta_long": sum(d["S_long"] for d in deltas) / 2,
                       "delta_T2": sum(d["T2"] for d in deltas) / 2})
    return sorted(ranked, key=lambda r: (-r["guard"], -r["delta_long"], -r["delta_T2"],
                                       r["arm"] != "E3", r["arm"]))


def native_training_identity(root, config):
    verify_data_content(root)
    data = read_json(root / "DATA_CONTENT_BINDING.json")
    return {"schema": 1, "config": OmegaConf.to_container(config, resolve=True), "code": code_identity(),
            "population_sha256": sha256(root / "POPULATION.json"),
            "data_content_sha256": data.get("portable_sha256", sha256(root / "DATA_CONTENT_BINDING.json")),
            "initialization_sha256": read_json(root / "SOURCE_AND_INITIALIZATION.json")["initialization"][
                str(config.general.seed)]["sha256"]}


def evaluate_checkpoint(root, arm, seed, step, role="CAL"):
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    configure_numeric()
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm, seed=seed)
    if role == "SEL":
        lock = read_json(root / "selection/CAL_LOCK.json")
        allowed = {lock["selected_updates"][arm]}
        if arm == "E1":
            allowed.update(lock["selected_updates"].values())
        if step not in allowed:
            raise ValueError("SEL checkpoint is not in the immutable CAL selection/control lock")
    path = root / "training" / arm / f"seed{seed}" / f"update={step:05d}.ckpt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    identity = native_training_identity(root, config)
    if saved["native_long_resume"]["identity"] != identity or saved["global_step"] != step:
        raise ValueError("checkpoint identity/step differs from current native trajectory")
    records = sorted(read_json(root / "POPULATION.json")["evaluation_inputs"][role],
                     key=lambda r: (r["horizon"], r["input_id"]))
    dependencies = {"training": identity, "checkpoint_sha256": sha256(path), "input_records": records,
                    "evaluation_seed": 9001, "native_process_protocol": "fresh process; sorted H,input_id",
                    "metric_sources": read_json(root / "SOURCE_AND_INITIALIZATION.json")["installed_libraries"]["stmetrics"]}
    result_path = root / "evaluation" / arm / f"seed{seed}/{role}-{step:05d}.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["identity"] == dependencies and previous["status"] == "COMPLETE" and all(
                sha256(Path(e["path"])) == e["sha256"] for e in previous["evidence"]):
            return previous
    system, _ = system_from_common(root, config)
    system.load_state_dict(saved["state_dict"], strict=True)
    del saved
    system.to("cuda:0").eval()
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    collate = NativeLongCollator(config, training=False)
    mapper = build_rio_class_mapper(datasets["rio"])

    def metric(h):
        return OfficialMetricAccumulator(mode="raw_local" if h == 1 else "strict_online",
                                         dataset_spec=config.instance_metric.dataset)

    pooled, references = {h: metric(h) for h in range(1, 6)}, {}
    expected = {f"T{h}": sum(r["horizon"] == h for r in records) for h in range(1, 6)}
    result = {"arm": arm, "seed": seed, "step": step, "role": role, "status": "INCOMPLETE",
              "completed": {k: 0 for k in expected}, "expected": expected, "identity": dependencies,
              "metrics": {}, "evidence": [], "by_reference": []}
    for record in records:
        h, ref = record["horizon"], record["reference_id"]
        with isolated_rng(stable_seed(9001, record["input_id"]), cuda_devices=(0,)):
            data, targets, _ = move_batch(collate(load_samples(datasets, [record])), torch.device("cuda:0"))
            with torch.inference_mode():
                output = system(data, point2segment=[targets[0]["point2segment"]],
                                raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            prediction = extract_official_task_prediction(
                system=system, output=output, target_low_resolution=targets[0],
                target_full_resolution=data.target_full[0], data=data, class_mapper=mapper,
                latest_stage_index=h - 1).prediction()
            target = _metric_target(data.target_full[0], datasets["rio"])
        pooled[h].update(prediction, target)
        references.setdefault((h, ref), metric(h)).update(prediction, target)
        result["completed"][f"T{h}"] += 1
        write_json(result_path, result)
    if result["completed"] != expected or any(n < 1 for n in expected.values()):
        raise ValueError("official evaluation population incomplete")
    for h, accumulator in pooled.items():
        key = "raw_local_AP" if h == 1 else "online_t-mAP"
        result["metrics"][f"T{h}"] = accumulator.compute()[key]
        evidence = result_path.with_name(f"{role}-{step:05d}-T{h}-official-state.json")
        write_json(evidence, accumulator.export_evidence())
        result["evidence"].append({"path": str(evidence), "sha256": sha256(evidence)})
    result["metrics"]["S_long"] = sum(result["metrics"][f"T{h}"] for h in (3, 4, 5)) / 3
    for (h, ref), accumulator in sorted(references.items()):
        result["by_reference"].append({"horizon": h, "reference_id": ref,
            "completed": accumulator._updates, "expected": sum(r["horizon"] == h and r["reference_id"] == ref for r in records),
            "AP": accumulator.compute()["raw_local_AP" if h == 1 else "online_t-mAP"]})
    result["status"] = "COMPLETE"
    write_json(result_path, result)
    return result


def charged_process(root, command, *, phase, cards, max_gpu_hours=None):
    active_path = root / "ACTIVE_PROCESS.json"
    if active_path.exists():
        raise RuntimeError("unreconciled process reservation; inspect its PID and ledger before resuming")
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    remaining = remaining_budget(root, plan)
    if max_gpu_hours is not None:
        remaining = min(remaining, max_gpu_hours)
    if remaining <= 0:
        raise RuntimeError("lifetime GPU budget exhausted")
    started = time.perf_counter()
    event_id = f"native-long:{phase}:{time.time_ns()}"
    reservation = {"event_id": event_id, "phase": phase, "cards": cards,
                   "started_unix": time.time(), "limit_gpu_hours": remaining}
    write_json(active_path, reservation)
    exit_code = -1
    try:
        process = subprocess.Popen(command, cwd=PROJECT, start_new_session=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1" if cards == 2 else "0",
            "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "RESCENE_NATIVE_PROCESS_GPUH_LIMIT": str(remaining),
            "RESCENE_NATIVE_PROCESS_STARTED_MONOTONIC": str(started)})
        try:
            write_json(active_path, {**reservation, "pid": process.pid})
            exit_code = process.wait(timeout=remaining * 3600 / cards)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    finally:
        seconds = time.perf_counter() - started
        event = {"event_id": event_id, "phase": phase, "seconds": seconds,
                 "cards": cards, "gpu_hours": cards * seconds / 3600, "exit_code": exit_code}
        append_event(root / "COST_LEDGER.jsonl", event)
        append_event(ARTIFACTS / "COST_LEDGER.jsonl", event)
        active_path.unlink(missing_ok=True)
    if exit_code:
        raise RuntimeError(f"{phase} failed; actual GPU reservation recorded")


class UpdateBoundary(pl.Callback):
    def __init__(self, root, arm, seed, endpoint):
        self.root, self.arm, self.seed, self.endpoint = root, arm, seed, endpoint
        self.last_step = -1
        progress = root / "TRAIN_PROGRESS.json"
        self.checkpoint_step = read_json(progress).get("checkpoint_updates", 0) if progress.exists() else 0
        self.started = float(os.environ.get("RESCENE_NATIVE_PROCESS_STARTED_MONOTONIC", time.perf_counter()))
        limit = os.environ.get("RESCENE_NATIVE_PROCESS_GPUH_LIMIT")
        self.gpu_hour_limit = float(limit) if limit else None

    def on_train_start(self, trainer, module):
        # The restored step precedes every accumulated batch of the next update.
        self.last_step = trainer.global_step

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.last_step or not step:
            return
        self.last_step = step
        directory = self.root / "training" / self.arm / f"seed{self.seed}"
        budget_stop = (self.gpu_hour_limit is not None and
                       trainer.world_size * (time.perf_counter() - self.started) / 3600
                       >= max(0., self.gpu_hour_limit - .05))
        if self.gpu_hour_limit is not None and torch.distributed.is_initialized():
            stop = torch.tensor(int(budget_stop), device=module.device)
            torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
            budget_stop = bool(stop.item())
        if step % 990 == 0 or step == self.endpoint or budget_stop:
            directory.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(directory / "last.ckpt")
            trainer.save_checkpoint(directory / f"update={step:05d}.ckpt")
            self.checkpoint_step = step
        if trainer.is_global_zero:
            write_json(self.root / "TRAIN_PROGRESS.json", {"arm": self.arm, "seed": self.seed,
                       "updates": step, "checkpoint_updates": self.checkpoint_step,
                       "status": "BUDGET_STOP" if budget_stop else "TRAINING"})
        if budget_stop:
            trainer.should_stop = True
        if step % 100 == 0 and trainer.is_global_zero:
            append_event(directory / "diagnostics.jsonl", {"step": step,
                "sampling": module.model.native_long_sampling_stats,
                "feedback": module.model.native_long_feedback_stats})


def train_process(root, arm, seed, endpoint):
    from pytorch_lightning.loggers import CSVLogger
    from pytorch_lightning.strategies import DDPStrategy

    configure_numeric()
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm, seed=seed)
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    if arm not in plan["full_arms"] + plan["pilot_arms"] or seed != 45:
        raise ValueError("training is outside the frozen budget authorization")
    system, _ = system_from_common(root, config)
    system.native_identity = native_training_identity(root, config)
    directory = root / "training" / arm / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "resolved_config.json", OmegaConf.to_container(config, resolve=True))
    world = plan["physical_batch"]["world_size"]
    trainer = pl.Trainer(accelerator="gpu", devices=world, max_steps=endpoint, max_epochs=-1,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches, precision="32-true",
        gradient_clip_val=1., num_sanity_val_steps=0, use_distributed_sampler=False,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(hours=8)) if world == 2 else "auto",
        callbacks=[UpdateBoundary(root, arm, seed, endpoint)], enable_checkpointing=False,
        logger=CSVLogger(str(directory), name="native_metrics"), enable_progress_bar=False)
    checkpoint = directory / "last.ckpt"
    trainer.fit(system, ckpt_path=str(checkpoint) if checkpoint.exists() else None, weights_only=False)


def ddp_preflight_process(root, arm):
    from pytorch_lightning.strategies import DDPStrategy

    configure_numeric()
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm)
    system, _ = system_from_common(root, config)
    system.native_identity = native_training_identity(root, config)
    trainer = pl.Trainer(accelerator="gpu", devices=2, max_steps=2, max_epochs=-1,
        accumulate_grad_batches=16, precision="32-true", gradient_clip_val=1.,
        num_sanity_val_steps=0, use_distributed_sampler=False,
        strategy=DDPStrategy(find_unused_parameters=True, timeout=timedelta(minutes=5)),
        enable_checkpointing=False, logger=False, enable_progress_bar=False)
    trainer.fit(system)
    if trainer.is_global_zero:
        write_json(root / "DDP_PREFLIGHT.json", {"status": "PASS", "optimizer_updates": trainer.global_step,
            "official_training_updates": 0, "code": code_identity(), "world_size": 2,
            "effective_batch": 32, "scheduler_total_steps": 29700})


def checkpoint_manifest(root, arm):
    directory = root / "training" / arm / "seed45"
    checkpoints = [{"step": int(p.stem.split("=")[1]), "path": str(p), "bytes": p.stat().st_size,
                    "sha256": sha256(p)} for p in sorted(directory.glob("update=*.ckpt"))]
    maximum = max((r["step"] for r in checkpoints), default=0)
    payload = {"arm": arm, "seed": 45, "updates": maximum, "full_updates": 29700,
               "status": "FULL_U_COMPLETE" if maximum == 29700 else "PARTIAL", "checkpoints": checkpoints}
    write_json(ARTIFACTS / "training" / arm / "seed45/checkpoint_manifest.json", payload)
    write_json(ARTIFACTS / "training" / arm / "seed45/resolved_config.json",
               read_json(directory / "resolved_config.json"))
    state = read_json(root / "RUN_STATE.json")
    state["arms"][arm].update(seed45_updates=maximum, status=payload["status"])
    write_json(root / "RUN_STATE.json", state)
    write_json(ARTIFACTS / "RUN_STATE.json", state)
    return payload


def execute_authorized_plan(root):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    if plan.get("execution_mode") == "SSH_CLUSTER":
        from scripts.native_long_cluster import start_cluster

        return start_cluster(root)
    arms = plan["full_arms"] + plan["pilot_arms"]
    if not arms:
        return {"status": "NOT_RUN_BUDGET", "authorized_arms": [], "full_updates": 29700}
    rows = []
    def run_points(arm, points):
        # CAL is evaluated at each optimizer endpoint in a new single-GPU process.
        for endpoint in points:
            charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "train",
                "--root", str(root), "--arm", arm, "--step", str(endpoint)], phase=f"train:{arm}:{endpoint}",
                cards=plan["physical_batch"]["world_size"])
            checkpoint_manifest(root, arm)
            charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "evaluate",
                "--root", str(root), "--arm", arm, "--step", str(endpoint), "--role", "CAL"],
                phase=f"CAL:{arm}:{endpoint}", cards=1)
            rows.append(read_json(root / "evaluation" / arm / f"seed45/CAL-{endpoint:05d}.json"))
    if plan["mode"] == "SCREEN_ONE":
        for arm in arms:
            run_points(arm, range(2970, 11881, 2970))
        ranked = rank_pilot(rows)
        winner = ranked[0]["arm"]
        write_json(root / "selection/PILOT_CONTINUATION.json", {"ranking": ranked, "continued": winner,
                   "uses_SEL": False, "selected_before_remaining_training": True})
        for arm in ("E0", "E1", winner):
            run_points(arm, range(14850, 29701, 2970))
    else:
        for arm in arms:
            run_points(arm, range(2970, 29701, 2970))
    write_json(ARTIFACTS / "evaluation/CAL_ALL.json", {"rows": rows})
    lock = select_cal_points(rows, arms)
    lock_path = root / "selection/CAL_LOCK.json"
    if lock_path.exists() and read_json(lock_path) != lock:
        raise ValueError("CAL lock is immutable")
    write_json(lock_path, lock)
    write_json(ARTIFACTS / "selection/CAL_LOCK.json", lock)
    for arm, step in lock["selected_updates"].items():
        requests = {(arm, step)}
        if arm in {"E2", "E3"}:
            requests.add(("E1", step))
        for method, update in sorted(requests):
            charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "evaluate",
                "--root", str(root), "--arm", method, "--step", str(update), "--role", "SEL"],
                phase=f"SEL:{method}:{update}", cards=1)
    return {"status": "SEED45_SEL_COMPLETE", "lock": lock}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "evaluate", "preflight"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("E0", "E1", "E2", "E3"), required=True)
    parser.add_argument("--step", type=int, default=29700)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--role", choices=("CAL", "SEL"), default="CAL")
    args = parser.parse_args()
    if args.phase == "train":
        train_process(args.root, args.arm, args.seed, args.step)
    elif args.phase == "evaluate":
        evaluate_checkpoint(args.root, args.arm, args.seed, args.step, args.role)
    else:
        ddp_preflight_process(args.root, args.arm)


if __name__ == "__main__":
    main()
