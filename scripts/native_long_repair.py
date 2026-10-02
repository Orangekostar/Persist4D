"""Audited repair of stopped trajectories; never discard checkpoints or costs."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from scripts.native_long_budget import ledger_events, merge_events
from scripts.native_long_budget import write_json_atomic as write_json
from scripts.native_long_campaign import PROJECT, code_identity
from scripts.native_long_cluster import (
    digest,
    exclusive,
    remote_module,
    rsync,
    ssh,
    validate_snapshot,
    worker_snapshot,
)
from scripts.short_module_screen import append_event, read_json, sha256

REPAIR_FILES = {"scripts/native_long_execution.py", "scripts/native_long_cluster.py",
                "scripts/native_long_campaign.py", "scripts/native_long_repair.py"}


def build_repair_plan(previous, snapshots, code):
    original = previous["jobs"][0]["identity"]["code"]
    changed = {p for p in set(original) | set(code) if original.get(p) != code.get(p)}
    if not changed or not changed <= REPAIR_FILES or set(original) - set(code):
        raise ValueError("repair only permits reviewed execution/control code changes")
    result = copy.deepcopy(previous)
    jobs = []
    for old in previous["jobs"]:
        row = validate_snapshot(snapshots[old["arm"]], old)
        if (old["identity"]["code"] != original or row["status"] != "FAILED"
                or row.get("stale") or row.get("active") or row.get("sel")
                or not 0 < row["updates"] < old["endpoint"]):
            raise ValueError("repair requires every trajectory failed, stopped and checkpointed")
        job = {k: copy.deepcopy(v) for k, v in old.items() if k != "job_id"}
        job["identity"]["code"] = code
        job["repair"] = {"previous_job_id": old["job_id"], "resume_update": row["updates"],
                         "carried_events": {e["event_id"]: e["job_id"] for e in row["events"]}}
        job["job_id"] = digest(job)
        jobs.append(job)
    result["jobs"] = jobs
    result["trained_repair"] = {"kind": "EXECUTION_CONTROL_METADATA_REPAIR",
                               "changed_files": sorted(changed),
                               "recompute_CAL": True, "preserve_training_state_and_costs": True}
    return result


def checkpoint_state_digest(checkpoint):
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    payload = dict(checkpoint)
    payload["native_long_resume"] = dict(checkpoint["native_long_resume"])
    identity = dict(payload["native_long_resume"]["identity"])
    identity.pop("code")
    payload["native_long_resume"]["identity"] = identity
    hasher = hashlib.sha256()

    def visit(value):
        if OmegaConf.is_config(value):
            hasher.update(f"omegaconf:{type(value).__name__}:".encode())
            visit(OmegaConf.to_container(value, resolve=False))
        elif isinstance(value, torch.Tensor):
            hasher.update(f"tensor:{value.dtype}:{tuple(value.shape)}:".encode())
            hasher.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            hasher.update(f"array:{value.dtype}:{value.shape}:".encode())
            hasher.update(value.tobytes())
        elif isinstance(value, Mapping):
            hasher.update(b"mapping:")
            for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(value[key])
            if hasattr(value, "_metadata"):
                hasher.update(b"metadata:")
                visit(value._metadata)
            hasher.update(b"end:")
        elif isinstance(value, (tuple, list)):
            hasher.update(f"{type(value).__name__}:{len(value)}:".encode())
            for item in value:
                visit(item)
        elif isinstance(value, type):
            hasher.update(f"type:{value.__module__}.{value.__qualname__}:".encode())
        elif isinstance(value, (str, bytes, bool, int, float)) or value is None:
            encoded = repr(value).encode()
            hasher.update(f"{type(value).__name__}:{len(encoded)}:".encode() + encoded)
        else:
            raise TypeError(f"unsupported checkpoint value: {type(value)}")

    visit(payload)
    return hasher.hexdigest()


def migrate_checkpoint(source, destination, *, expected_identity, desired_identity, step):
    import torch

    before = {k: v for k, v in expected_identity.items() if k != "code"}
    after = {k: v for k, v in desired_identity.items() if k != "code"}
    if before != after:
        raise ValueError("repair cannot change checkpoint config/data/initialization")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    resume = saved["native_long_resume"]
    if resume["identity"] != expected_identity:
        raise ValueError("original checkpoint identity differs from the repair request")
    if (saved["global_step"] != step or resume["schema"] != 1
            or resume["next_global_draw"] != step * 32 or len(resume["rng_by_rank"]) != 2
            or any(s["last_epoch"] != step or s["total_steps"] != 29700
                   for s in saved["lr_schedulers"])):
        raise ValueError("checkpoint optimizer/draw/scheduler/rank boundary is invalid")
    state_sha = checkpoint_state_digest(saved)
    resume["identity"] = desired_identity
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".ckpt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(saved, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if checkpoint_state_digest(reloaded) != state_sha or reloaded["native_long_resume"]["identity"] != desired_identity:
            raise ValueError("checkpoint training payload changed during metadata migration")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"step": step, "next_global_draw": step * 32, "state_sha256": state_sha,
            "original_sha256": sha256(source), "migrated_sha256": sha256(destination)}


def assert_stopped(root):
    if (root / "ACTIVE_PROCESS.json").exists() or (root / "selection/CAL_LOCK.json").exists():
        raise ValueError("repair refuses active reservations or selected trajectories")
    path = root / "WORKER_PID.json"
    pid = read_json(path).get("pid", 0) if path.exists() else 0
    cmd = Path("/proc") / str(pid) / "cmdline"
    if pid and cmd.exists() and str(root).encode() in cmd.read_bytes():
        raise ValueError("repair refuses a running worker")


def worker_repair(root, request, *, prepare_only=False):
    import torch

    with exclusive(root / "worker.lock"):
        assert_stopped(root)
        spec = read_json(root / "WORKER_SPEC.json")
        arm = spec["arm"]
        old = next(j for j in request["previous"]["jobs"] if j["arm"] == arm)
        new = next(j for j in request["plan"]["jobs"] if j["arm"] == arm)
        archive = root / "repairs" / request["repair_id"]
        if spec["job_id"] not in {old["job_id"], new["job_id"]}:
            raise ValueError("worker job differs from the reviewed repair")
        prepared = archive / "PREPARED.json"
        applied = archive / "APPLIED.json"
        if applied.exists():
            if code_identity() != new["identity"]["code"] or spec["job_id"] != new["job_id"]:
                raise ValueError("applied repair code/spec changed")
            return read_json(applied)
        step = new["repair"]["resume_update"]
        directory = root / "training" / arm / "seed45"
        names = ("last.ckpt", f"update={step:05d}.ckpt")
        if not prepared.exists():
            if spec["job_id"] != old["job_id"] or code_identity() != old["identity"]["code"]:
                raise ValueError("original worker code/spec differs before repair preparation")
            snapshot = validate_snapshot(worker_snapshot(root), old)
            if snapshot != request["snapshots"][arm]:
                raise ValueError("worker evidence changed after repair review")
            write_json(root / "REPAIR_PENDING.json", {"repair_id": request["repair_id"]})
            archive.mkdir(parents=True, exist_ok=True)
            for name in ("WORKER_SPEC.json", "WORKER_STATE.json", "TRAIN_PROGRESS.json",
                         "SOURCE_AND_INITIALIZATION.json", "COST_LEDGER.jsonl", "DDP_PREFLIGHT.json", "worker.log"):
                path = root / name
                if path.exists():
                    shutil.copy2(path, archive / name)
            for name in request["plan"]["trained_repair"]["changed_files"]:
                path = PROJECT / name
                if path.exists() and name in old["identity"]["code"]:
                    backup = archive / "code" / name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
            receipts = {}
            for name in names:
                target = archive / name
                shutil.copy2(directory / name, target)
                saved = torch.load(target, map_location="cpu", weights_only=False)
                identity = saved["native_long_resume"]["identity"]
                if any(identity[k] != v for k, v in old["identity"].items() if v is not None):
                    raise ValueError("checkpoint identity differs from the frozen worker job")
                receipts[name] = {"sha256": sha256(target), "state_sha256": checkpoint_state_digest(saved),
                                  "identity": identity}
                if saved["global_step"] != step:
                    raise ValueError("checkpoint step differs from reviewed progress")
            if receipts[names[0]]["state_sha256"] != receipts[names[1]]["state_sha256"]:
                raise ValueError("last checkpoint differs from committed checkpoint")
            for row in snapshot["cal"]:
                if row["step"] == step and row["identity"]["checkpoint_sha256"] != receipts[names[1]]["sha256"]:
                    raise ValueError("committed checkpoint SHA differs from completed CAL evidence")
            write_json(prepared, {"repair_id": request["repair_id"], "checkpoints": receipts,
                                  "snapshot": snapshot})
        if prepare_only:
            return read_json(prepared)
        if code_identity() != new["identity"]["code"]:
            raise ValueError("repaired worker source SHA differs from the reviewed code")
        from scripts.native_long_campaign import compose_config
        from scripts.native_long_execution import native_training_identity

        config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=root, arm=arm)
        desired = native_training_identity(root, config)
        receipts = {}
        for name, before in read_json(prepared)["checkpoints"].items():
            source = archive / name
            if sha256(source) != before["sha256"]:
                raise ValueError("archived original checkpoint changed")
            receipts[name] = migrate_checkpoint(source, directory / name,
                expected_identity=before["identity"], desired_identity=desired, step=step)
            if receipts[name]["state_sha256"] != before["state_sha256"]:
                raise ValueError("checkpoint numerical state differs from the repair archive")
        evaluation = root / "evaluation"
        if evaluation.exists():
            evaluation.rename(archive / "evaluation")
        events = ledger_events(archive / "COST_LEDGER.jsonl")
        tagged = [{"job_id": old["job_id"], **event} for event in events]
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
            for event in tagged:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, root / "COST_LEDGER.jsonl")
        original_spec = read_json(archive / "WORKER_SPEC.json")
        write_json(root / "WORKER_SPEC.json", {**original_spec, **new})
        receipt = {"repair_id": request["repair_id"], "old_job_id": old["job_id"],
                   "new_job_id": new["job_id"], "checkpoints": receipts,
                   "preserved_gpu_hours": sum(e["gpu_hours"] for e in tagged),
                   "archived_CAL_count": len(read_json(prepared)["snapshot"]["cal"])}
        write_json(applied, receipt)
        (root / "REPAIR_PENDING.json").unlink(missing_ok=True)
        return receipt


def repair_cluster(root):
    with exclusive(root / "cluster/controller.lock"):
        if (root / "selection/CAL_LOCK.json").exists():
            raise ValueError("repair refuses a selected controller trajectory")
        monitor = root / "cluster/MONITOR_PID.json"
        pid = read_json(monitor).get("pid", 0) if monitor.exists() else 0
        cmd = Path("/proc") / str(pid) / "cmdline"
        if pid and cmd.exists() and str(root).encode() in cmd.read_bytes():
            raise ValueError("stop the controller monitor before a reviewed repair")
        path = root / "selection/BUDGET_LOCK.json"
        previous = read_json(path)
        latest = root / "cluster/LATEST_TRAINED_REPAIR.json"
        current = code_identity()
        pending = read_json(latest) if latest.exists() else None
        if pending and pending["plan"]["jobs"][0]["identity"]["code"] == current:
            request = pending
            if pending.get("status") == "COMPLETE":
                if previous != pending["plan"]:
                    raise ValueError("completed repair controller lock changed")
                return pending
        else:
            snapshots = {j["arm"]: json.loads(ssh(j, remote_module(j, "snapshot"))) for j in previous["jobs"]}
            repaired = build_repair_plan(previous, snapshots, current)
            old_sha = sha256(path)
            repaired["previous_budget_lock_sha256"] = old_sha
            repair_id = digest({"old_budget_sha256": old_sha, "new_code": current})
            request = {"schema": 1, "repair_id": repair_id, "status": "PREPARING", "previous": previous,
                       "plan": repaired, "snapshots": snapshots,
                       "supersedes_pending_repair_id": pending["repair_id"] if pending else None}
            archive = root / "cluster/repairs" / repair_id
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archive / "BUDGET_LOCK.json")
            for name in ("SOURCE_AND_INITIALIZATION.json", "RUN_STATE.json", "COST_LEDGER.jsonl", "RESOURCE_PLAN.json"):
                original = root / name
                if original.exists():
                    shutil.copy2(original, archive / name)
            for name in ("STATUS.json", "snapshots", "MONITOR_PID.json", "monitor.log"):
                original = root / "cluster" / name
                if original.is_dir():
                    shutil.copytree(original, archive / name)
                elif original.exists():
                    shutil.copy2(original, archive / name)
            write_json(latest, request)
        archive = root / "cluster/repairs" / request["repair_id"]
        request_path = archive / "REQUEST.json"
        write_json(request_path, request)
        for old in request["previous"]["jobs"]:
            remote = f"{old['user']}@{old['host']}"
            rsync(["-az", str(PROJECT / "scripts/native_long_repair.py"),
                   f"{remote}:{shlex.quote(old['repo'])}/scripts/"])
            rsync(["-az", str(request_path), f"{remote}:{shlex.quote(old['root'])}/REPAIR_REQUEST.json"])
            command = remote_module(old, "snapshot")
            command[-4:] = ["scripts.native_long_repair", "prepare-worker", "--root", old["root"]]
            ssh(old, command, timeout=300)
        receipts = {}
        for job in request["plan"]["jobs"]:
            remote = f"{job['user']}@{job['host']}"
            for name in request["plan"]["trained_repair"]["changed_files"]:
                rsync(["-az", str(PROJECT / name),
                       f"{remote}:{shlex.quote(str(Path(job['repo']) / Path(name).parent))}/"])
            command = remote_module(job, "snapshot")
            command[-4:] = ["scripts.native_long_repair", "apply-worker", "--root", job["root"]]
            receipts[job["arm"]] = json.loads(ssh(job, command, timeout=300))
        events = merge_events([ledger_events(root / "COST_LEDGER.jsonl"),
                               *[r["events"] for r in request["snapshots"].values()]])
        existing = {e["event_id"] for e in ledger_events(root / "COST_LEDGER.jsonl")}
        for event in events:
            if event["event_id"] not in existing:
                append_event(root / "COST_LEDGER.jsonl", event)
        history = root / "selection/budget_history" / f"{request['plan']['previous_budget_lock_sha256']}.json"
        history.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive / "BUDGET_LOCK.json", history)
        for name in ("STATUS.json", "snapshots"):
            old = root / "cluster" / name
            if old.exists():
                old.rename(archive / f"retired-{name}")
        write_json(path, request["plan"])
        write_json(root / "RESOURCE_PLAN.json", request["plan"])
        state = read_json(root / "RUN_STATE.json")
        state.update(stage="CLUSTER_REPAIRED", code=current)
        for job in request["plan"]["jobs"]:
            state["arms"][job["arm"]].update(status="REPAIRED_READY", seed45_updates=job["repair"]["resume_update"])
        write_json(root / "RUN_STATE.json", state)
        request.update(status="COMPLETE", worker_receipts=receipts,
                       carried_campaign_gpu_hours=sum(e["gpu_hours"] for e in events))
        write_json(latest, request)
        write_json(archive / "COMPLETE.json", request)
        return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("repair", "prepare-worker", "apply-worker"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "repair":
        result = repair_cluster(root)
    else:
        result = worker_repair(root, read_json(root / "REPAIR_REQUEST.json"),
                               prepare_only=args.command == "prepare-worker")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
