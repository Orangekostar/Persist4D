"""Disjoint native training jobs, SSH transport and controller-only selection."""

import argparse
import fcntl
import hashlib
import ipaddress
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import yaml

from scripts.native_long_budget import budget_cap, ledger_events, merge_events
from scripts.native_long_budget import write_json_atomic as write_json
from scripts.native_long_campaign import PROJECT, code_identity
from scripts.short_module_screen import append_event, read_json, sha256

ARMS = ("E0", "E1", "E2", "E3")
ARTIFACTS = PROJECT / "artifacts/native_long_cluster_v1"


class LockBusyError(RuntimeError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@contextmanager
def exclusive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise LockBusyError(f"another controller/worker holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_config(config):
    nodes = config["nodes"]
    if len({n["host"] for n in nodes}) != len(nodes):
        raise ValueError("cluster hosts must be unique")
    if sorted(n["arm"] for n in nodes if n.get("arm") is not None) != list(ARMS):
        raise ValueError("assign exactly one host to each of E0-E3")
    if sum(n["gpus"] for n in nodes) > config["max_concurrent_gpus"]:
        raise ValueError("cluster exceeds its concurrent GPU limit")
    if config["reserve_gpu_hours"] < 8:
        raise ValueError("reserve at least eight GPU hours")
    for node in nodes:
        ipaddress.ip_address(node["host"])
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", node["user"]) or node["gpus"] != 2:
            raise ValueError("native cluster nodes require a valid user and two GPUs")
        for key in ("repo", "root", "data_root", "encoder", "common"):
            path = Path(node[key])
            if not path.is_absolute() or ".." in path.parts or any(c in str(path) for c in "\n\r\0"):
                raise ValueError("remote paths must be absolute and normalized")
        if "native_long_cluster_v1" not in Path(node["root"]).parts:
            raise ValueError("worker root must be an isolated native_long_cluster_v1 directory")


def build_cluster_plan(previous, config, *, current_gpu_hours, population_sha256,
                       initialization_sha256, code, data_content_sha256=None):
    validate_config(config)
    cap = budget_cap({"lifetime_cap_gpu_hours": config["lifetime_budget_gpu_hours"]})
    available = cap - previous["prior"]["gpu_hours"] - current_gpu_hours
    costs = previous["forecast"]["costs_gpu_hours"]
    if any(not math.isfinite(costs[a]) or costs[a] <= 0 for a in ARMS):
        raise ValueError("all four complete forecasts must be finite and positive")
    headroom = available - config["reserve_gpu_hours"] - sum(costs[a] for a in ARMS)
    if headroom < 0:
        raise ValueError("budget cannot cover four complete arms and the reserve")
    share = math.floor(headroom / 4 * 1e6) / 1e6
    identity = {"population_sha256": population_sha256,
                "initialization_sha256": initialization_sha256, "code": code,
                "data_content_sha256": data_content_sha256}
    jobs = []
    for arm in ARMS:
        node = next(n for n in config["nodes"] if n.get("arm") == arm)
        job = {**node, "root": str(Path(node["root"]) / arm / "seed45"), "cards": 2,
               "seed": 45, "endpoint": 29700, "quota_gpu_hours": costs[arm] + share,
               "identity": identity}
        job["job_id"] = digest(job)
        jobs.append(job)
    return {**previous, "mode": "FULL_FOUR", "full_arms": list(ARMS), "pilot_arms": [],
            "execution_mode": "SSH_CLUSTER", "lifetime_cap_gpu_hours": cap,
            "reserve_gpu_hours": config["reserve_gpu_hours"], "cluster_config": config,
            "cluster_config_sha256": digest(config), "jobs": jobs,
            "physical_batch": {"world_size": 2, "per_rank_batch": 1, "accumulation": 16, "effective": 32},
            "current_campaign_gpu_hours": current_gpu_hours, "remaining_at_plan_gpu_hours": available,
            "forecast_training_gpu_hours": sum(costs[a] for a in ARMS),
            "remaining_before_preflight_gpu_hours": cap - previous["prior"]["gpu_hours"],
            "authorization": {"source": "user_2026-10-01_adjust_budget_and_scheduler",
                              "selected_lifetime_cap_gpu_hours": cap,
                              "supersedes_lifetime_cap_gpu_hours": budget_cap(previous),
                              "maximum_concurrent_gpus": config["max_concurrent_gpus"]}}


def failed_replacement_snapshots(root, previous, config):
    validate_config(config)
    if config["lifetime_budget_gpu_hours"] != budget_cap(previous):
        raise ValueError("failed recovery must preserve the lifetime cap")
    monitor_path = root / "cluster/MONITOR_PID.json"
    if monitor_path.exists():
        pid = read_json(monitor_path).get("pid", 0)
        proc = Path("/proc") / str(pid) / "cmdline"
        if pid and proc.exists() and str(root).encode() in proc.read_bytes():
            raise ValueError("stop the old controller monitor before recovery")
    snapshots = {}
    for job in previous["jobs"]:
        node = next(n for n in config["nodes"] if n.get("arm") == job["arm"])
        new_root = Path(node["root"]) / job["arm"] / "seed45"
        for old, new in ((Path(job["root"]), new_root), (Path(job["repo"]), Path(node["repo"]))):
            if old == new or old in new.parents or new in old.parents:
                raise ValueError("failed recovery requires separate code and worker roots")
        if any(node[k] != job[k] for k in ("host", "user", "python", "data_root", "encoder", "common")):
            raise ValueError("failed recovery must preserve bound hosts, environment and assets")
        row = json.loads(ssh(job, remote_module(job, "snapshot")))
        if (row.get("status") != "FAILED" or row.get("stale") or row.get("updates") != 0
                or row.get("active") is not None or row.get("cal") or row.get("sel")):
            raise ValueError("replacement requires failed, inactive, uncheckpointed trajectories without evaluation")
        validate_snapshot(row, job)
        guard = ("import json;from pathlib import Path;"
                 f"root=Path({job['root']!r});new=Path({str(new_root)!r});"
                 "pid=json.loads((root/'WORKER_PID.json').read_text()).get('pid',0);"
                 "proc=Path('/proc')/str(pid)/'cmdline';"
                 "alive=pid>0 and proc.exists() and str(root).encode() in proc.read_bytes();"
                 "assert not alive and not (root/'ACTIVE_PROCESS.json').exists() "
                 "and not any(root.rglob('*.ckpt')),'old worker is alive, reserved or checkpointed';"
                 "assert not new.exists() or not any(new.iterdir()),'replacement worker root is not empty'")
        ssh(job, ["python3", "-c", guard])
        snapshots[job["arm"]] = row
    return snapshots


def amend_budget(root, config, *, artifacts=ARTIFACTS, code=None, amend_unstarted=False,
                 replace_failed_uncheckpointed=False):
    with exclusive(root / "cluster/controller.lock"):
        path = root / "selection/BUDGET_LOCK.json"
        previous = read_json(path)
        identity = code_identity() if code is None else code
        recovery_snapshots = None
        if previous.get("execution_mode") == "SSH_CLUSTER":
            changed = previous["cluster_config_sha256"] != digest(config) or previous["jobs"][0]["identity"]["code"] != identity
            if not changed:
                return previous
            if replace_failed_uncheckpointed:
                recovery_snapshots = failed_replacement_snapshots(root, previous, config)
            elif not amend_unstarted:
                raise ValueError("cluster lock/config/code changed; create an explicit reviewed amendment")
            validate_config(config)
            for job in previous["jobs"] if recovery_snapshots is None else []:
                program = ("import json;from pathlib import Path;"
                           f"root=Path({job['root']!r});pid=root/'WORKER_PID.json';progress=root/'TRAIN_PROGRESS.json';"
                           "number=json.loads(pid.read_text()).get('pid',0) if pid.exists() else 0;"
                           "cmd=Path('/proc')/str(number)/'cmdline';"
                           "active=number>0 and cmd.exists() and str(root).encode() in cmd.read_bytes();"
                           "updates=json.loads(progress.read_text())['updates'] if progress.exists() else 0;"
                           "ledger=root/'COST_LEDGER.jsonl';"
                           "assert not active and not (root/'ACTIVE_PROCESS.json').exists() and updates==0 "
                           "and (not ledger.exists() or not ledger.read_text().strip()),"
                           "'cannot amend a worker after work or an unresolved reservation'")
                ssh(job, ["python3", "-c", program])
        state = read_json(root / "RUN_STATE.json")
        if any(r["seed45_updates"] for r in state["arms"].values()) or (root / "selection/CAL_LOCK.json").exists():
            raise ValueError("cannot amend a trained/selected trajectory as a fresh cluster plan")
        old_events = ledger_events(root / "COST_LEDGER.jsonl")
        events = merge_events([old_events, *[r["events"] for r in (recovery_snapshots or {}).values()]])
        portable_sha = portable_data(root)["sha256"] if (root / "DATA_CONTENT_BINDING.json").exists() else None
        if recovery_snapshots is not None:
            old_identity = previous["jobs"][0]["identity"]
            if (sha256(root / "POPULATION.json") != old_identity["population_sha256"]
                    or portable_sha != old_identity["data_content_sha256"]
                    or read_json(root / "SOURCE_AND_INITIALIZATION.json")["initialization"]["45"]["sha256"]
                    != old_identity["initialization_sha256"]):
                raise ValueError("failed recovery must preserve population, data and common initialization")
        plan = build_cluster_plan(previous, config, current_gpu_hours=sum(r["gpu_hours"] for r in events),
                                  population_sha256=sha256(root / "POPULATION.json"),
                                  initialization_sha256=read_json(root / "SOURCE_AND_INITIALIZATION.json")[
                                      "initialization"]["45"]["sha256"], code=identity,
                                  data_content_sha256=portable_sha)
        old_sha = sha256(path)
        archive = root / f"selection/budget_history/{old_sha}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
        plan["previous_budget_lock_sha256"] = old_sha
        if recovery_snapshots is not None:
            failure = root / f"cluster/failures/{old_sha}"
            failure.mkdir(parents=True, exist_ok=True)
            for name in ("SOURCE_AND_INITIALIZATION.json", "RUN_STATE.json", "COST_LEDGER.jsonl"):
                shutil.copy2(root / name, failure / name)
            write_json(failure / "WORKER_SNAPSHOTS.json", recovery_snapshots)
            for name in ("snapshots", "STATUS.json", "READINESS.json", "STAGING.json", "DISPATCH.json",
                         "DEPLOYMENT_READINESS.json", "DEPLOYMENT_OBSERVATION.json", "readiness",
                         "MONITOR_PID.json", "monitor.log"):
                old = root / "cluster" / name
                if old.exists():
                    old.rename(failure / name)
            existing_ids = {r["event_id"] for r in old_events}
            for event in events:
                if event["event_id"] not in existing_ids:
                    append_event(root / "COST_LEDGER.jsonl", event)
            plan["recovery"] = {"kind": "REPLACE_ALL_FAILED_UNCHECKPOINTED",
                                "previous_job_ids": {j["arm"]: j["job_id"] for j in previous["jobs"]},
                                "failure_archive": str(failure), "carried_campaign_gpu_hours": sum(r["gpu_hours"] for r in events)}
            write_json(failure / "RECOVERY.json", plan["recovery"])
            write_json(root / "cluster/LATEST_RECOVERY.json", plan["recovery"])
        write_json(path, plan)
        write_json(root / "RESOURCE_PLAN.json", plan)
        write_json(artifacts / "BUDGET_LOCK.json", plan)
        write_json(artifacts / "PREVIOUS_BUDGET_LOCK.json", previous)
        state.update(stage="CLUSTER_PLANNED", budget_mode="FULL_FOUR",
                     scientific_status="NOT_RUN_CLUSTER_READY_PENDING", code=identity,
                     lifetime_cap_gpu_hours=budget_cap(plan))
        for row in state["arms"].values():
            row.update(status="READY_CLUSTER", seed45_updates=0)
        write_json(root / "RUN_STATE.json", state)
        write_json(artifacts / "RUN_STATE.json", state)
        return plan


def validate_snapshot(snapshot, job):
    if (snapshot["job_id"] != job["job_id"] or snapshot["identity"] != job["identity"]
            or snapshot["arm"] != job["arm"]):
        raise ValueError("worker identity differs from the immutable cluster job")
    if not 0 <= snapshot["updates"] <= 29700:
        raise ValueError("worker optimizer position is invalid")
    if any(event.get("job_id") != job["job_id"] for event in snapshot["events"]):
        raise ValueError("worker ledger contains an event from a different job")
    for row in snapshot.get("cal", []) + snapshot.get("sel", []):
        training = row["identity"]["training"]
        keys = ["code", "population_sha256", "initialization_sha256"]
        if job["identity"].get("data_content_sha256") is not None:
            keys.append("data_content_sha256")
        if any(training[k] != job["identity"][k] for k in keys):
            raise ValueError("worker evaluation identity differs from its job")
    return snapshot


def selection_from_snapshots(plan, snapshots, population):
    from scripts.native_long_execution import select_cal_points

    rows = []
    completed, failed = [], []
    for job in plan["jobs"]:
        snapshot = validate_snapshot(snapshots[job["arm"]], job)
        if snapshot["status"] == "FAILED" and not snapshot.get("stale") and not snapshot.get("active"):
            failed.append(job["arm"])
            continue
        if snapshot.get("stale") or snapshot["updates"] != 29700 or snapshot["status"] != "FULL_U_COMPLETE":
            raise ValueError("CAL lock requires every planned full trajectory")
        completed.append(job["arm"])
        if {r["step"] for r in snapshot["cal"]} != set(range(2970, 29701, 2970)) or len(snapshot["cal"]) != 10:
            raise ValueError("all ten complete CAL endpoints are required")
        for row in snapshot["cal"]:
            if (row["arm"] != job["arm"] or row["seed"] != 45
                    or row["expected"] != population["evaluation_counts"]["CAL"]):
                raise ValueError("CAL identity/population differs from controller population")
        rows.extend(snapshot["cal"])
    if not completed:
        raise ValueError("no completed full trajectory is available")
    return {**select_cal_points(rows, completed), "excluded_failed_arms": failed,
            "cluster_job_ids": {j["arm"]: j["job_id"] for j in plan["jobs"]},
            "population_sha256": plan["jobs"][0]["identity"]["population_sha256"]}


def sel_comparisons(plan, snapshots, population, lock):
    lookup = {}
    for job in plan["jobs"]:
        snapshot = snapshots.get(job["arm"])
        if not snapshot or snapshot.get("stale") or "events" not in snapshot:
            return None
        validate_snapshot(snapshot, job)
        for row in snapshot.get("sel", []):
            if (row["role"] != "SEL" or row["seed"] != 45 or row["status"] != "COMPLETE"
                    or row["arm"] != job["arm"] or row["expected"] != population["evaluation_counts"]["SEL"]
                    or row["completed"] != row["expected"] or any(n < 1 for n in row["expected"].values())):
                raise ValueError("SEL reporting requires complete official denominators")
            values = row["metrics"]
            if (not {"S_long", *[f"T{h}" for h in range(1, 6)]} <= set(values)
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values())
                    or abs(values["S_long"] - sum(values[f"T{h}"] for h in (3, 4, 5)) / 3) > 1e-12):
                raise ValueError("SEL metrics must be finite and use pooled official S_long")
            lookup[job["arm"], row["step"]] = values
    if any((arm, step) not in lookup for arm, step in lock["selected_updates"].items()):
        return None
    comparisons = []
    for arm in ("E1", "E2", "E3"):
        if arm not in lock["selected_updates"]:
            continue
        step = lock["selected_updates"][arm]
        control = "E0" if arm == "E1" else "E1"
        if control not in lock["selected_updates"]:
            continue
        control_step = lock["selected_updates"]["E0"] if arm == "E1" else step
        if (arm, step) not in lookup or (control, control_step) not in lookup:
            return None
        comparisons.append({"arm": arm, "step": step, "control": control, "control_step": control_step,
            "delta": {key: lookup[arm, step][key] - lookup[control, control_step][key]
                      for key in ("T1", "T2", "T3", "T4", "T5", "S_long")},
            "status": "SEED45_ONLY_UNCONFIRMED"})
    return comparisons


def ssh_transport():
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ControlMaster=auto",
            "-o", "ControlPersist=600", "-o", "ControlPath=/tmp/rescene-native-long-%C"]


def ssh(node, argv, *, timeout=60):
    return subprocess.run([*ssh_transport(),
                           f"{node['user']}@{node['host']}", shlex.join([str(a) for a in argv])],
                          check=True, capture_output=True, text=True, timeout=timeout).stdout


def rsync(argv, **kwargs):
    return subprocess.run(["rsync", "-e", shlex.join(ssh_transport()), *argv], check=True, **kwargs)


def remote_module(job, phase, *args):
    env = ["env", f"PYTHONPATH={remote_pythonpath(job)}",
           f"RESCENE_NATIVE_LONG_ARTIFACTS={job['root']}/artifacts", "OMP_NUM_THREADS=2",
           "MKL_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2"]
    return [*env, job["python"], "-m", "scripts.native_long_cluster", phase,
            "--root", job["root"], *args]


def remote_pythonpath(node):
    return ":".join([node["repo"], *[f"{node['repo']}/third_party/{name}"
                    for name in ("concerto", "sonata", "stmetrics", "detectron2")]])


def probe_node(node):
    program = (
        "import csv,io,json,importlib.util,importlib.metadata,subprocess;from pathlib import Path;"
        f"node=json.loads({json.dumps(json.dumps(node))});"
        "packages=['torch','pytorch_lightning','spconv','concerto','sonata','stmetrics'];"
        "missing=[p for p in packages if importlib.util.find_spec(p) is None];"
        "paths={k:Path(node[k]).exists() for k in ['repo','data_root','encoder','common']};"
        "gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',"
        "'--format=csv,noheader,nounits'],text=True);"
        "cards=list(csv.reader(io.StringIO(gpu)));"
        "hardware=len(cards)>=2 and all('A40' in r[1] and float(r[2])>=43000 and float(r[3])<=512 and float(r[4])==0 for r in cards[:2]);"
        "spec=Path(node['root'])/'WORKER_SPEC.json';"
        "staged=spec.exists() and json.loads(spec.read_text()).get('job_id')==node.get('job_id');"
        "print(json.dumps({'paths':paths,'missing_packages':missing,'gpu':gpu,'hardware_ready':hardware,"
        "'staged':staged,'ready':not missing and all(paths.values()) and hardware and staged}))"
    )
    try:
        return {"host": node["host"], "ssh": "PASS", **json.loads(ssh(node,
            ["env", f"PYTHONPATH={remote_pythonpath(node)}", node["python"], "-c", program]))}
    except (subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"host": node["host"], "ssh": "FAILED", "ready": False, "error": str(error),
                "stderr": getattr(error, "stderr", None)}


def probe_cluster(root):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    with ThreadPoolExecutor(max_workers=6) as pool:
        assigned = {j["host"]: j for j in plan["jobs"]}
        nodes = list(pool.map(probe_node, [assigned.get(n["host"], n) for n in plan["cluster_config"]["nodes"]]))
    payload = {"status": "READY" if all(n["ready"] for n in nodes if n["host"] in assigned) else "PREPARATION_REQUIRED",
               "nodes": nodes, "long_training_started": False}
    write_json(root / "cluster/READINESS.json", payload)
    write_json(ARTIFACTS / "READINESS.json", payload)
    return payload


def portable_data(root):
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    data_root = Path(source["assets"]["data_root"])
    rows = {}
    for row in read_json(root / "DATA_CONTENT_BINDING.json")["files"]:
        path = Path(row["path"])
        try:
            relative = path.relative_to(data_root)
        except ValueError:
            relative = Path("external_gt") / digest(str(path))[:24] / path.name
        rows[str(relative)] = {"bytes": row["stat"]["bytes"], "sha256": row["sha256"], "source": str(path)}
    for path in sorted((data_root / "processed").rglob("*.yaml")):
        rows[str(path.relative_to(data_root))] = {"bytes": path.stat().st_size, "sha256": sha256(path), "source": str(path)}
    return {"files": rows, "sha256": digest(rows)}


def native_dependency_sources(root):
    import importlib.util

    libraries = read_json(root / "SOURCE_AND_INITIALIZATION.json")["installed_libraries"]
    sources = {name: Path(libraries[name]["source"]).parent.parent
               for name in ("concerto", "sonata", "stmetrics")}
    sources["detectron2"] = Path(importlib.util.find_spec("detectron2").origin).parent.parent
    sources["pointnet2"] = PROJECT / "third_party/pointnet2"
    return sources


def stage_cluster(root, *, with_data=False):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    source = read_json(root / "SOURCE_AND_INITIALIZATION.json")
    data = portable_data(root)
    dependencies = native_dependency_sources(root)
    outcomes = {}
    with exclusive(root / "cluster/controller.lock"):
        for job in plan["jobs"]:
            try:
                if job["identity"]["code"] != code_identity():
                    raise ValueError("staged code differs from locked code")
                if data["sha256"] != job["identity"]["data_content_sha256"]:
                    raise ValueError("controller data/metadata changed after the cluster lock")
                remote = f"{job['user']}@{job['host']}"
                guard = ("import json;from pathlib import Path;"
                         f"root=Path({job['root']!r});progress=root/'TRAIN_PROGRESS.json';pid=root/'WORKER_PID.json';"
                         "updates=json.loads(progress.read_text())['updates'] if progress.exists() else 0;"
                         "number=json.loads(pid.read_text()).get('pid',0) if pid.exists() else 0;"
                         "cmd=Path('/proc')/str(number)/'cmdline';"
                         "active=number>0 and cmd.exists() and str(root).encode() in cmd.read_bytes();"
                         "assert not active and updates==0,'cannot stage over an active/trained worker'")
                ssh(job, ["python3", "-c", guard])
                ssh(job, ["mkdir", "-p", job["repo"], f"{job['repo']}/third_party", job["root"], str(Path(job["encoder"]).parent),
                          str(Path(job["common"]).parent)])
                rsync(["-az", "--safe-links", "--exclude=__pycache__", "--exclude=*.pyc",
                    *[str(PROJECT / d) for d in ("models", "datasets", "trainer", "utils", "scripts", "conf")],
                    f"{remote}:{shlex.quote(job['repo'])}/"])
                rsync(["-az", "--safe-links", "--exclude=.git", "--exclude=__pycache__", "--exclude=*.pyc",
                    *[str(path) for path in dependencies.values()],
                    f"{remote}:{shlex.quote(job['repo'])}/third_party/"])
                for name, local in (("encoder", source["weights"]["concerto_pretrained"]["path"]),
                                    ("common", source["initialization"]["45"]["path"])):
                    rsync(["-az", local, f"{remote}:{shlex.quote(job[name])}"])
                if with_data:
                    ssh(job, ["mkdir", "-p", job["data_root"]])
                    bundle = root / "cluster/data_bundle"
                    for relative, row in data["files"].items():
                        link = bundle / relative
                        link.parent.mkdir(parents=True, exist_ok=True)
                        if not link.is_symlink():
                            link.symlink_to(row["source"])
                    rsync(["-azL", str(bundle) + "/", f"{remote}:{shlex.quote(job['data_root'])}/"])
                spec = {**job, "data": data, "population": read_json(root / "POPULATION.json"),
                        "source": {"initialization": source["initialization"],
                                   "weights": source["weights"], "versions": source["versions"],
                                   "installed_libraries": source["installed_libraries"]}}
                path = root / f"cluster/specs/{job['arm']}.json"
                write_json(path, spec)
                rsync(["-az", str(path), f"{remote}:{shlex.quote(job['root'])}/WORKER_SPEC.json"])
                outcomes[job["arm"]] = {"status": "STAGED", "job_id": job["job_id"]}
            except (subprocess.SubprocessError, AssertionError, ValueError) as error:
                outcomes[job["arm"]] = {"status": "STAGE_FAILED", "error": str(error)}
    write_json(root / "cluster/STAGING.json", outcomes)
    return outcomes


def bootstrap_worker(root):
    spec = read_json(root / "WORKER_SPEC.json")
    if spec["identity"]["code"] != code_identity():
        raise ValueError("worker code differs from paired controller code")
    if digest(spec["data"]["files"]) != spec["data"]["sha256"] or spec["data"]["sha256"] != spec["identity"]["data_content_sha256"]:
        raise ValueError("portable data manifest differs from the locked controller data")
    import importlib.metadata
    import importlib.util

    for package, version in spec["source"]["versions"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"worker version mismatch: {package}")
    for name in ("concerto", "sonata", "stmetrics"):
        folder = Path(importlib.util.find_spec(name).origin).parent
        actual = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*.py"))}
        if actual != spec["source"]["installed_libraries"][name]["python_source_sha256"]:
            raise ValueError(f"worker numerical library source differs: {name}")
    rows = []
    for relative, expected in spec["data"]["files"].items():
        path = Path(spec["data_root"]) / relative
        if ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("invalid portable data path")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"worker data content mismatch: {relative}")
        rows.append({"path": str(path), "sha256": expected["sha256"],
                     "stat": {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}})
    for key, expected in (("encoder", spec["source"]["weights"]["concerto_pretrained"]["sha256"]),
                          ("common", spec["identity"]["initialization_sha256"])):
        if sha256(Path(spec[key])) != expected:
            raise ValueError(f"worker {key} SHA differs")
    write_json(root / "POPULATION.json", spec["population"])
    if sha256(root / "POPULATION.json") != spec["identity"]["population_sha256"]:
        raise ValueError("worker population differs from controller")
    write_json(root / "DATA_CONTENT_BINDING.json", {"files": rows, "portable_sha256": spec["data"]["sha256"]})
    source = spec["source"]
    source["initialization"]["45"]["path"] = spec["common"]
    source["weights"]["concerto_pretrained"]["path"] = spec["encoder"]
    source["code"] = code_identity()
    write_json(root / "SOURCE_AND_INITIALIZATION.json", source)
    plan = {"mode": "FULL_FOUR", "full_arms": [spec["arm"]], "pilot_arms": [],
            "lifetime_cap_gpu_hours": spec["quota_gpu_hours"], "prior": {"gpu_hours": 0.},
            "physical_batch": {"world_size": 2, "per_rank_batch": 1, "accumulation": 16, "effective": 32}}
    write_json(root / "selection/BUDGET_LOCK.json", plan)
    write_json(root / "RESOURCE_PLAN.json", plan)
    if not (root / "RUN_STATE.json").exists():
        write_json(root / "RUN_STATE.json", {"arms": {spec["arm"]: {"seed45_updates": 0, "status": "READY"}}})
    (root / "COST_LEDGER.jsonl").touch(exist_ok=True)
    return spec


def worker_snapshot(root):
    spec = read_json(root / "WORKER_SPEC.json")
    status = read_json(root / "WORKER_STATE.json") if (root / "WORKER_STATE.json").exists() else {"status": "NOT_STARTED"}
    progress = read_json(root / "TRAIN_PROGRESS.json") if (root / "TRAIN_PROGRESS.json").exists() else {"updates": 0}
    events = [{**r, "job_id": spec["job_id"]} for r in ledger_events(root / "COST_LEDGER.jsonl")]
    cal, sel = [], []
    for role, rows in (("CAL", cal), ("SEL", sel)):
        for path in sorted((root / "evaluation" / spec["arm"] / "seed45").glob(f"{role}-?????.json")):
            row = read_json(path)
            if row["status"] == "COMPLETE":
                if row["identity"]["training"]["code"] != spec["identity"]["code"]:
                    raise ValueError("worker evaluation used different code")
                for evidence in row["evidence"]:
                    if sha256(Path(evidence["path"])) != evidence["sha256"]:
                        raise ValueError("worker official metric evidence changed")
                rows.append(row)
    active = read_json(root / "ACTIVE_PROCESS.json") if (root / "ACTIVE_PROCESS.json").exists() else None
    live = 0.
    if active and active["event_id"] not in {r["event_id"] for r in events}:
        live = max(0., time.time() - active["started_unix"]) * active["cards"] / 3600
    return {"job_id": spec["job_id"], "identity": spec["identity"], "arm": spec["arm"],
            "status": status["status"], "updates": progress.get("checkpoint_updates", progress["updates"]),
            "observed_optimizer_updates": progress["updates"], "events": events,
            "cal": cal, "sel": sel, "active": active, "live_gpu_hours": live,
            "error": status.get("error")}


def run_worker(root, *, evaluate_only=False):
    from scripts.native_long_execution import charged_process, checkpoint_manifest

    with exclusive(root / "worker.lock"):
        state_path = root / "WORKER_STATE.json"
        stage = "SEL" if evaluate_only else "TRAIN_CAL"
        write_json(state_path, {"status": "SEL_RUNNING" if evaluate_only else "TRAIN_CAL_RUNNING", "stage": stage})
        try:
            spec = bootstrap_worker(root)
            if evaluate_only:
                requests = read_json(root / "selection/SEL_REQUESTS.json")["steps"]
                for step in requests:
                    charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "evaluate",
                        "--root", str(root), "--arm", spec["arm"], "--step", str(step), "--role", "SEL"],
                        phase=f"SEL:{spec['arm']}:{step}", cards=1)
            else:
                proof = root / "DDP_PREFLIGHT.json"
                if not proof.exists() or read_json(proof).get("code") != spec["identity"]["code"]:
                    charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "preflight",
                        "--root", str(root), "--arm", spec["arm"]], phase=f"DDP_PREFLIGHT:{spec['arm']}",
                        cards=2, max_gpu_hours=.2)
                if read_json(proof)["optimizer_updates"] != 2 or read_json(proof)["status"] != "PASS":
                    raise ValueError("actual two-GPU native DDP preflight did not pass")
                for endpoint in range(2970, 29701, 2970):
                    progress = read_json(root / "TRAIN_PROGRESS.json") if (root / "TRAIN_PROGRESS.json").exists() else {"updates": 0}
                    if progress.get("checkpoint_updates", progress["updates"]) < endpoint:
                        charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "train",
                            "--root", str(root), "--arm", spec["arm"], "--step", str(endpoint)],
                            phase=f"train:{spec['arm']}:{endpoint}", cards=2)
                        manifest = checkpoint_manifest(root, spec["arm"])
                        if manifest["updates"] < endpoint:
                            raise RuntimeError("worker stopped at a budget boundary before the endpoint")
                    charged_process(root, [sys.executable, "-m", "scripts.native_long_execution", "evaluate",
                        "--root", str(root), "--arm", spec["arm"], "--step", str(endpoint), "--role", "CAL"],
                        phase=f"CAL:{spec['arm']}:{endpoint}", cards=1)
            write_json(state_path, {"status": "FULL_U_COMPLETE", "stage": stage,
                       "SEL_steps": requests if evaluate_only else []})
        except BaseException as error:
            write_json(state_path, {"status": "FAILED", "error": str(error)})
            raise
        finally:
            write_json(root / "RESULT.json", worker_snapshot(root))


def launch_worker(job, *, evaluate_only=False):
    command = remote_module(job, "worker-sel" if evaluate_only else "worker")
    program = f"""import fcntl,json,subprocess
from pathlib import Path
root=Path({job['root']!r})
command={command!r}
spec=json.loads((root/'WORKER_SPEC.json').read_text())
assert spec['job_id']=={job['job_id']!r},'staged worker job differs'
with (root/'launch.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    path=root/'WORKER_PID.json'
    old=json.loads(path.read_text()) if path.exists() else {{}}
    pid=old.get('pid',0)
    proc=Path('/proc')/str(pid)/'cmdline'
    alive=pid>0 and proc.exists() and str(root).encode() in proc.read_bytes()
    phase={'worker-sel' if evaluate_only else 'worker'!r}
    state_path=root/'WORKER_STATE.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {{}}
    requests=json.loads((root/'selection/SEL_REQUESTS.json').read_text())['steps'] if phase=='worker-sel' else []
    done=state.get('status')=='FULL_U_COMPLETE' and (phase=='worker' or state.get('stage')=='SEL' and state.get('SEL_steps')==requests)
    if alive:
        assert old.get('phase')==phase,'worker is busy in a different phase'
        row={{**old,'status':'ALREADY_RUNNING'}}
    elif done:
        row={{'status':'ALREADY_COMPLETE'}}
    else:
        with (root/'worker.log').open('a') as log:
            process=subprocess.Popen(command,cwd={job['repo']!r},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row={{'pid':process.pid,'phase':phase,'status':'LAUNCHED'}}
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(row))
        temporary.replace(path)
    print(json.dumps(row))
"""
    return json.loads(ssh(job, ["python3", "-c", program]))


def start_cluster(root):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    if any(j["identity"]["code"] != code_identity() for j in plan["jobs"]):
        raise ValueError("current code differs from reviewed cluster lock")
    outcomes = {}
    def start(job):
        readiness = probe_node(job)
        if not readiness["ready"]:
            return job["arm"], {"status": "BLOCKED_PREPARATION", "readiness": readiness}
        try:
            return job["arm"], {"status": "LAUNCHED_OR_RUNNING", **launch_worker(job)}
        except subprocess.SubprocessError as error:
            return job["arm"], {"status": "LAUNCH_FAILED", "error": str(error)}
    with exclusive(root / "cluster/controller.lock"), ThreadPoolExecutor(max_workers=4) as pool:
        outcomes.update(pool.map(start, plan["jobs"]))
        write_json(root / "cluster/DISPATCH.json", outcomes)
    return outcomes


def collect_evidence(root, job, snapshot):
    scope = Path(job["root"]) / "evaluation" / job["arm"] / "seed45"
    for row in snapshot.get("cal", []) + snapshot.get("sel", []):
        for evidence in row.get("evidence", []):
            remote = Path(evidence["path"])
            if remote.parent != scope or not re.fullmatch(r"(?:CAL|SEL)-\d{5}-T[1-5]-official-state\.json", remote.name):
                raise ValueError("official evidence path is outside the worker scope")
            local = root / "cluster/evidence" / job["arm"] / "seed45" / remote.name
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or sha256(local) != evidence["sha256"]:
                rsync(["-az", f"{job['user']}@{job['host']}:{shlex.quote(str(remote))}", str(local)], timeout=60)
            if sha256(local) != evidence["sha256"]:
                raise ValueError("collected official evidence SHA differs from the worker snapshot")


def collect_cluster(root):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    snapshots = {}
    with exclusive(root / "cluster/controller.lock"):
        for job in plan["jobs"]:
            try:
                row = json.loads(ssh(job, remote_module(job, "snapshot")))
                validate_snapshot(row, job)
                collect_evidence(root, job, row)
                write_json(root / f"cluster/snapshots/{job['arm']}.json", row)
                snapshots[job["arm"]] = row
            except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as error:
                previous = root / f"cluster/snapshots/{job['arm']}.json"
                if previous.exists():
                    row = validate_snapshot(read_json(previous), job)
                    if row.get("active"):
                        row["live_gpu_hours"] = max(0., time.time() - row["active"]["started_unix"]) * row["active"]["cards"] / 3600
                    snapshots[job["arm"]] = {**row, "stale": True, "retrieval_error": str(error)}
                else:
                    snapshots[job["arm"]] = {"status": "UNAVAILABLE", "error": str(error)}
        events = merge_events([ledger_events(root / "COST_LEDGER.jsonl"),
                               *[r["events"] for r in snapshots.values() if "events" in r]])
        payload = {"snapshots": snapshots, "events": events,
                   "lifetime_gpu_hours": plan["prior"]["gpu_hours"] + sum(r["gpu_hours"] for r in events),
                   "live_gpu_hours": sum(r.get("live_gpu_hours", 0.) for r in snapshots.values()),
                   "lifetime_cap_gpu_hours": budget_cap(plan)}
        write_json(root / "cluster/STATUS.json", payload)
    return payload


def finalize_cluster(root, *, payload=None):
    payload = collect_cluster(root) if payload is None else payload
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    population_sha = sha256(root / "POPULATION.json")
    if population_sha != plan["jobs"][0]["identity"]["population_sha256"]:
        raise ValueError("controller population changed after planning")
    with exclusive(root / "cluster/controller.lock"):
        path = root / "selection/CAL_LOCK.json"
        if path.exists():
            lock = read_json(path)
            if (lock["cluster_job_ids"] != {j["arm"]: j["job_id"] for j in plan["jobs"]}
                    or lock["population_sha256"] != population_sha):
                raise ValueError("controller CAL lock belongs to a different cluster/population")
        else:
            lock = selection_from_snapshots(plan, payload["snapshots"], read_json(root / "POPULATION.json"))
            write_json(path, lock)
        write_json(ARTIFACTS / "CAL_LOCK.json", lock)
        outcomes = {}
        for job in plan["jobs"]:
            if job["arm"] not in lock["selected_updates"]:
                continue
            snapshot = payload["snapshots"].get(job["arm"], {})
            if snapshot.get("stale") or "events" not in snapshot:
                outcomes[job["arm"]] = {"status": "UNAVAILABLE"}
                continue
            validate_snapshot(snapshot, job)
            steps = {lock["selected_updates"][job["arm"]]}
            if job["arm"] == "E1":
                steps.update(lock["selected_updates"][a] for a in ("E2", "E3") if a in lock["selected_updates"])
            directory = root / f"cluster/selection/{job['arm']}"
            write_json(directory / "CAL_LOCK.json", lock)
            write_json(directory / "SEL_REQUESTS.json", {"steps": sorted(steps)})
            try:
                ssh(job, ["mkdir", "-p", str(Path(job["root"]) / "selection")])
                rsync(["-az", str(directory) + "/", f"{job['user']}@{job['host']}:{shlex.quote(job['root'])}/selection/"])
                outcomes[job["arm"]] = launch_worker(job, evaluate_only=True)
            except (subprocess.SubprocessError, OSError) as error:
                outcomes[job["arm"]] = {"status": "SEL_DISPATCH_FAILED", "error": str(error)}
        write_json(root / "cluster/SEL_DISPATCH.json", outcomes)
    return {"lock": lock, "SEL_dispatch": outcomes}


def cluster_report(root):
    plan = read_json(root / "selection/BUDGET_LOCK.json")
    status_path = root / "cluster/STATUS.json"
    status = read_json(status_path) if status_path.exists() else {"snapshots": {}, "live_gpu_hours": 0.,
        "lifetime_gpu_hours": plan["prior"]["gpu_hours"] + sum(r["gpu_hours"] for r in ledger_events(root / "COST_LEDGER.jsonl"))}
    rows = []
    for job in plan["jobs"]:
        snapshot = status["snapshots"].get(job["arm"], {"status": "NOT_STARTED", "updates": 0, "sel": []})
        rows.append({"arm": job["arm"], "host": job["host"], "updates": snapshot.get("updates", 0),
                     "status": snapshot["status"], "quota_gpu_hours": job["quota_gpu_hours"], "SEL": snapshot.get("sel", [])})
    lock_path = root / "selection/CAL_LOCK.json"
    comparisons = sel_comparisons(plan, status["snapshots"], read_json(root / "POPULATION.json"),
                                  read_json(lock_path)) if lock_path.exists() else None
    payload = {"mode": plan["mode"], "execution_mode": "SSH_CLUSTER", "arms": rows,
               "lifetime_cap_gpu_hours": budget_cap(plan), "lifetime_gpu_hours": status["lifetime_gpu_hours"],
               "live_gpu_hours": status.get("live_gpu_hours", 0.),
               "remaining_gpu_hours": budget_cap(plan) - status["lifetime_gpu_hours"] - status.get("live_gpu_hours", 0.),
               "reserve_gpu_hours": plan["reserve_gpu_hours"],
               "scientific_status": "SEED45_SEL_COMPLETE_CONFIRMATION_PENDING" if comparisons is not None else "IN_PROGRESS_OR_NOT_STARTED",
               "seed46_formal_profile": "NOT_RUN", "gain": comparisons}
    write_json(ARTIFACTS / "STATUS.json", payload)
    report = "# Native Long Cluster\n\nBudget authorization: user request2026-10-01.\n\n"
    report += f"Lifetime cap {budget_cap(plan):.2f} GPUh; accounted {status['lifetime_gpu_hours']:.6f} GPUh.\n\n"
    report += "| Arm | Host | Committed updates | Status | GPUh allocation |\n|---|---|---:|---|---:|\n"
    report += "".join(f"| {r['arm']} | {r['host']} | {r['updates']} /29700 | {r['status']} | {r['quota_gpu_hours']:.6f} |\n" for r in rows)
    report += "\nGlobal batch32, per-rank batch1, world2, accumulation16; all arms use the same seed45 common state and population. "
    report += "No trained gain is claimed until complete locked SEL comparisons are available. Seed46, formal confirmation and profiling are separate future stages. "
    report += "Historical budget-limited results in artifacts/native_long_retrain_v1 remain a snapshot of the previous192 GPUh plan.\n"
    if comparisons is not None:
        report += "\n| Arm | Control | Arm step | Control step | Delta S_long | Delta T2 |\n|---|---|---:|---:|---:|---:|\n"
        report += "".join(f"| {r['arm']} | {r['control']} | {r['step']} | {r['control_step']} | {r['delta']['S_long']:.6f} | {r['delta']['T2']:.6f} |\n" for r in comparisons)
    (ARTIFACTS / "REPORT.md").write_text(report)
    return payload


def cluster_command(root, command, *, through="publish"):
    if command in {"status", "report"}:
        return cluster_report(root)
    if through == "prepare":
        return stage_cluster(root)
    if through == "preflight":
        return probe_cluster(root)
    return start_cluster(root)


def watch_cluster(root, *, interval=30):
    while True:
        try:
            status = collect_cluster(root)
        except LockBusyError:
            time.sleep(interval)
            continue
        report = cluster_report(root)
        if report["gain"] is not None:
            return report
        plan = read_json(root / "selection/BUDGET_LOCK.json")
        snapshots = status["snapshots"]
        if all(r.get("status") in {"UNAVAILABLE", "NOT_STARTED"} for r in snapshots.values()):
            return {**report, "status": "NO_RUNNING_WORKERS_CHECK_READINESS"}
        lock_path = root / "selection/CAL_LOCK.json"
        if lock_path.exists():
            selected = read_json(lock_path)["selected_updates"]
            if any(snapshots.get(a, {}).get("status") == "FAILED" for a in selected):
                return {**report, "status": "SEL_FAILED_CHECK_WORKER_LOGS"}
            finalize_cluster(root, payload=status)
        elif all(snapshots.get(a, {}).get("status") in {"FULL_U_COMPLETE", "FAILED"}
               and not snapshots[a].get("stale") and not snapshots[a].get("active") for a in ARMS):
            finalize_cluster(root, payload=status)
        if status["lifetime_gpu_hours"] + status["live_gpu_hours"] >= budget_cap(plan):
            return {**report, "status": "BUDGET_EXHAUSTED"}
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "probe", "stage", "start", "watch", "collect", "finalize", "report", "snapshot", "worker", "worker-sel"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "conf/native_long_cluster.yaml")
    parser.add_argument("--with-data", action="store_true")
    parser.add_argument("--amend-unstarted", action="store_true")
    parser.add_argument("--replace-failed-uncheckpointed", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "plan":
        result = amend_budget(root, yaml.safe_load(args.config.read_text()), amend_unstarted=args.amend_unstarted,
                              replace_failed_uncheckpointed=args.replace_failed_uncheckpointed)
    elif args.command == "probe":
        result = probe_cluster(root)
    elif args.command == "stage":
        result = stage_cluster(root, with_data=args.with_data)
    elif args.command == "start":
        result = start_cluster(root)
    elif args.command == "collect":
        result = collect_cluster(root)
    elif args.command == "watch":
        result = watch_cluster(root)
    elif args.command == "finalize":
        result = finalize_cluster(root)
    elif args.command == "report":
        result = cluster_report(root)
    elif args.command == "snapshot":
        result = worker_snapshot(root)
    else:
        run_worker(root, evaluate_only=args.command == "worker-sel")
        result = worker_snapshot(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
