"""Single-GPU frozen-parent adaptation; formal settings come only from RUN_LOCK."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from scripts.native_long_budget import ledger_events, write_json_atomic
from scripts.revisit_prompt_assets import ARTIFACTS, local_assets, read_json, sha256, public_error
from scripts.revisit_prompt_assets import (
    build_system, configuration_digest, configure_runtime, numeric_sources,
    parent_tensor_digest, read_run_plan, require_charged_child, validate_replication,
)


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
        stream.flush()
        os.fsync(stream.fileno())


def train(args, assets):
    controlled = args.pilot or args.smoke
    if controlled:
        if args.seed != 146:
            raise ValueError("preflight/pilots use seed146; formal training never loads their weights")
        n, checkpoints, phase = 400, [], "P0" if args.smoke else "P1"
        updates, plan = 3 if args.smoke else 8, None
    else:
        plan = read_run_plan(assets)
        n, checkpoints, phase = plan["N"], plan["checkpoint_steps"], "P2"
        updates = n
        if args.seed == 147:
            validate_replication(args.variant, read_json(assets["selection_lock"]))
    request = dict(variant=args.variant, adapter_seed=args.seed, formal_N=None if controlled else n,
                   planned_updates=updates, checkpoint_steps=checkpoints, phase=phase,
                   pilot=args.pilot,smoke=args.smoke, status="DRY_RUN_NO_TRAINING")
    if args.dry_run:
        return request
    require_charged_child(phase)
    import torch
    from datasets.revisit_prompt_episode import RevisitDrawPlan, RevisitEpisodeLoader
    from trainer.revisit_prompt_trainer import adapter_state, restore_resume, save_resume

    configure_runtime(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("MPR training requires exactly one visible CUDA device")
    root = Path(assets["run_root"])
    folder = root/("preflight/gradients" if args.smoke else "pilots" if args.pilot else "trajectories")/f"{args.variant}-s{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    sources = numeric_sources()
    system, config, load_audit = build_system(assets, variant=args.variant, seed=args.seed)
    identity = dict(schema=1, parent_seed=45, parent_sha256=assets["hashes"]["parent"],
        encoder_sha256=assets["hashes"]["encoder"], taxonomy_sha256=assets["hashes"]["taxonomy"],
        variant=args.variant, adapter_seed=args.seed, N=n, pilot=args.pilot,smoke=args.smoke,
        population_sha256=assets["hashes"]["population"], numeric_sources=sources,
        config_sha256=configuration_digest(config, assets),
        run_lock_sha256=sha256(root/"RUN_LOCK.json") if plan else None)
    parent_before = parent_tensor_digest(system)
    initial = adapter_state(system)
    from models.revisit_prompt_modules import tensor_summary
    from scripts.revisit_prompt_assets import digest
    identity["initialization_sha256"] = digest({k:tensor_summary(v) for k,v in initial.items()})
    if plan and identity["initialization_sha256"] != plan["initialization_hashes"][f"{args.variant}-s{args.seed}"]:
        raise ValueError("adapter initialization differs from the frozen run lock")
    system.to("cuda:0")
    optimizer, scheduler = system.make_optimizer(n)
    latest = folder/"resume-latest.pt"
    step = 0
    population = read_json(assets["population_path"])
    draws = RevisitDrawPlan(population["references"], seed=args.seed)
    draw_rows = [r for cursor in range(updates) for r in draws.draw_update(cursor)]
    draw_bytes = "".join(json.dumps(r, sort_keys=True)+"\n" for r in draw_rows)
    draw_path = root/("smoke_draws" if args.smoke else "pilot_draws" if args.pilot else "draws")/f"seed{args.seed}"/"DRAW_PLAN.jsonl"
    draw_path.parent.mkdir(parents=True, exist_ok=True)
    if draw_path.exists() and draw_path.read_text() != draw_bytes:
        raise ValueError("same-seed draw plan differs across trajectories")
    if not draw_path.exists():
        temporary = draw_path.with_suffix(".tmp")
        temporary.write_text(draw_bytes)
        os.replace(temporary, draw_path)
    identity["draw_plan_sha256"] = sha256(draw_path)
    if plan and identity["draw_plan_sha256"] != plan["draw_hashes"][str(args.seed)]:
        raise ValueError("full training draw identity differs from the frozen run lock")
    # Include the draw identity before creating or validating the actual resume.
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError("requested trajectory has no completed resume boundary")
        saved = restore_resume(latest, system, optimizer, scheduler, identity=identity)
        step = saved["step"]
    else:
        if latest.exists() or (folder/"TRAINING.jsonl").exists():
            raise ValueError("trajectory already exists; use --resume without overwriting its evidence")
        save_resume(latest, system, optimizer, scheduler, step=0, identity=identity)
    loader = RevisitEpisodeLoader(config, assets["data_root"], read_json(assets["formal_population_path"]))
    stop = {"requested":False}
    previous_handler = signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    durations = []
    previous_steps = [row["step"] for row in ledger_events(folder/"TRAINING.jsonl")]
    previous_reported_step = max(previous_steps,default=0)
    status = {**request, "status":"RUNNING", "step":step, "identity":identity,
              "restored_step":step,"previous_reported_step":previous_reported_step,
              "load_audit":load_audit, "internal_attention_dtype":"original FlashAttention FP16 where required"}
    write_json_atomic(folder/"STATUS.json", status)
    started = time.perf_counter()
    try:
        while step < updates:
            deadline = float(os.environ.get("MPR_JOB_DEADLINE_MONOTONIC", "inf"))
            predicted = max(durations)*1.25 if durations else 0.
            if stop["requested"] or deadline-time.perf_counter() <= predicted+15.:
                status["status"] = "STOPPED_AT_OPTIMIZER_BOUNDARY"
                break
            before = time.perf_counter()
            record = system.optimize_update(loader, draw_rows[step*4:(step+1)*4], optimizer, scheduler,
                                            device="cuda:0", capture_gradients=step < 3 or args.pilot)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-before
            step += 1
            durations.append(seconds)
            _append(folder/"TRAINING.jsonl", dict(**record, step=step, variant=args.variant,
                adapter_seed=args.seed, parent_seed=45, seconds=seconds, pilot=args.pilot,smoke=args.smoke,
                replayed_update=step<=previous_reported_step,
                process_gpu_hours=(time.perf_counter()-float(os.environ["MPR_JOB_STARTED_MONOTONIC"]))/3600,
                status="OPTIMIZER_BOUNDARY_COMPLETE", attempt=os.environ["MPR_CHARGED_EVENT_ID"]))
            if controlled or step % 50 == 0 or step in checkpoints or step == updates:
                save_resume(latest, system, optimizer, scheduler, step=step, identity=identity)
            if step in checkpoints:
                system.save_delta(folder/f"delta-{step:06d}.pt", assets=assets, step=step,
                                  numeric_sources=sources, training_identity=identity)
            status.update(step=step, elapsed_seconds=time.perf_counter()-started)
            write_json_atomic(folder/"STATUS.json", status)
        save_resume(latest, system, optimizer, scheduler, step=step, identity=identity)
        parent_after = parent_tensor_digest(system)
        if parent_before != parent_after:
            raise ValueError("an original frozen parent parameter or buffer changed")
        final = adapter_state(system)
        changes = {key: not torch.equal(initial[key], final[key]) for key in initial}
        status.update(step=step, parent_tensor_sha256=parent_after, frozen_parent_unchanged=True,
                      adapter_tensor_changed=changes, seconds_per_update=durations,
                      elapsed_seconds=time.perf_counter()-started)
        if step == updates:
            status["status"] = ("SMOKE_COMPLETE_NOT_A_FORMAL_MODEL" if args.smoke else
                                "PILOT_COMPLETE_NOT_A_FORMAL_MODEL" if args.pilot else "COMPLETE")
        write_json_atomic(folder/"STATUS.json", status)
        return status
    except BaseException as error:
        status.update(status="FAILED", step=step, error_type=type(error).__name__, error=str(error))
        write_json_atomic(folder/"STATUS.json", status)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=[f"M{i}" for i in range(1,7)])
    parser.add_argument("--seed", type=int, default=146, choices=(146,147))
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="charged P1: eight updates, discarded before formal training")
    mode.add_argument("--smoke",action="store_true",help="charged P0: three gradient-check updates, never a formal model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    assets = None
    try:
        assets = local_assets()
        result = train(args, assets)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] != "STOPPED_AT_OPTIMIZER_BOUNDARY" else 75
    except Exception as error:
        write_json_atomic(ARTIFACTS/"TRAIN_ENTRY_STATUS.json", dict(status="FAILED",variant=args.variant,
            adapter_seed=args.seed,error_type=type(error).__name__,error=public_error(error,assets)))
        print(f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
