import copy
import json

import pytest


def cluster_config():
    return {"lifetime_budget_gpu_hours": 1800., "reserve_gpu_hours": 8.,
            "max_concurrent_gpus": 12,
            "nodes": [{"host": f"192.168.100.{101+i}", "user": "pluto", "gpus": 2,
                       "arm": f"E{i}" if i < 4 else None, "python": "python3",
                       "repo": "/home/pluto/native_long_cluster_v1/code",
                       "root": "/home/pluto/native_long_cluster_v1/runtime",
                       "data_root": "/home/pluto/native_long_cluster_v1/data",
                       "encoder": "/home/pluto/native_long_cluster_v1/assets/encoder.pth",
                       "common": "/home/pluto/native_long_cluster_v1/assets/common.pt"}
                      for i in range(6)]}


def previous_plan():
    return {"lifetime_cap_gpu_hours": 192., "prior": {"gpu_hours": 80.73503875023967},
            "full_arms": [], "pilot_arms": [], "mode": "BASELINE_RECOVERY_ONLY",
            "forecast": {"costs_gpu_hours": {"E0": 380., "E1": 400., "E2": 410., "E3": 420.}},
            "physical_batch": {"world_size": 2, "per_rank_batch": 1, "accumulation": 16, "effective": 32}}


def planned():
    from scripts.native_long_cluster import build_cluster_plan

    return build_cluster_plan(previous_plan(), cluster_config(), current_gpu_hours=.2182,
                              population_sha256="population", initialization_sha256="common",
                              code={"model.py": "code"})


def test_budget_amendment_authorizes_four_disjoint_two_gpu_jobs():
    plan = planned()
    assert plan["mode"] == "FULL_FOUR"
    assert plan["full_arms"] == ["E0", "E1", "E2", "E3"]
    assert plan["lifetime_cap_gpu_hours"] == 1800.
    assert len({j["host"] for j in plan["jobs"]}) == 4
    assert all(j["endpoint"] == 29700 and j["cards"] == 2 for j in plan["jobs"])
    assert plan["physical_batch"] == previous_plan()["physical_batch"]
    total = sum(j["quota_gpu_hours"] for j in plan["jobs"])
    assert total + plan["prior"]["gpu_hours"] + .2182 + 8 <= 1800. + 1e-9
    assert planned() == plan


def test_cluster_rejects_duplicate_hosts_and_inadequate_budget():
    from scripts.native_long_cluster import build_cluster_plan

    config = cluster_config()
    config["nodes"][1]["host"] = config["nodes"][0]["host"]
    with pytest.raises(ValueError, match="unique"):
        build_cluster_plan(previous_plan(), config, current_gpu_hours=.2,
                           population_sha256="p", initialization_sha256="i", code={})
    config = cluster_config()
    config["lifetime_budget_gpu_hours"] = 192.
    with pytest.raises(ValueError, match="four complete"):
        build_cluster_plan(previous_plan(), config, current_gpu_hours=.2,
                           population_sha256="p", initialization_sha256="i", code={})


def test_budget_amendment_archives_original_and_is_idempotent(tmp_path):
    from scripts.native_long_cluster import amend_budget
    from scripts.short_module_screen import sha256, write_json

    root = tmp_path / "runtime"
    write_json(root / "selection/BUDGET_LOCK.json", previous_plan())
    write_json(root / "RESOURCE_PLAN.json", previous_plan())
    write_json(root / "POPULATION.json", {"references": []})
    write_json(root / "SOURCE_AND_INITIALIZATION.json", {"initialization": {"45": {"sha256": "common"}}})
    write_json(root / "RUN_STATE.json", {"arms": {a: {"seed45_updates": 0} for a in ("E0", "E1", "E2", "E3")}})
    (root / "COST_LEDGER.jsonl").write_text(json.dumps({"event_id": "preflight", "gpu_hours": .2}) + "\n")
    old = sha256(root / "selection/BUDGET_LOCK.json")
    amend_budget(root, cluster_config(), artifacts=tmp_path / "artifacts", code={})
    before = (root / "selection/BUDGET_LOCK.json").read_bytes()
    amend_budget(root, cluster_config(), artifacts=tmp_path / "artifacts", code={})
    assert (root / "selection/BUDGET_LOCK.json").read_bytes() == before
    assert sha256(root / f"selection/budget_history/{old}.json") == old


def test_ledger_merge_deduplicates_and_rejects_conflicting_events():
    from scripts.native_long_cluster import merge_events

    row = {"event_id": "node-E0:1", "gpu_hours": 2., "cards": 2}
    assert merge_events([[row], [copy.deepcopy(row)]]) == [row]
    with pytest.raises(ValueError, match="conflicting"):
        merge_events([[row], [{**row, "gpu_hours": 3.}]])
    with pytest.raises(ValueError, match="finite"):
        merge_events([[{**row, "gpu_hours": float("nan")}]] )


def complete_snapshot(job):
    counts = {f"T{h}": h for h in range(1, 6)}
    return {"job_id": job["job_id"], "identity": job["identity"],
            "arm": job["arm"], "status": "FULL_U_COMPLETE", "updates": 29700,
            "events": [], "cal": [{"arm": job["arm"], "seed": 45, "role": "CAL",
                "step": step, "status": "COMPLETE", "expected": counts, "completed": counts,
                "identity": {"training": job["identity"]},
                "metrics": {**{f"T{h}": .1 for h in range(1, 6)}, "S_long": .1}}
                for step in range(2970, 29701, 2970)]}


def test_controller_requires_all_full_trajectories_and_complete_cal():
    from scripts.native_long_cluster import selection_from_snapshots

    plan = planned()
    snapshots = {j["arm"]: complete_snapshot(j) for j in plan["jobs"]}
    population = {"evaluation_counts": {"CAL": {f"T{h}": h for h in range(1, 6)}}}
    lock = selection_from_snapshots(plan, snapshots, population)
    assert lock["selected_updates"] == {a: 2970 for a in snapshots}
    broken = copy.deepcopy(snapshots)
    broken["E2"]["cal"][0]["completed"]["T5"] -= 1
    with pytest.raises(ValueError, match="CAL"):
        selection_from_snapshots(plan, broken, population)
    broken = copy.deepcopy(snapshots)
    broken["E3"]["updates"] = 11880
    with pytest.raises(ValueError, match="full trajectory"):
        selection_from_snapshots(plan, broken, population)
    broken = copy.deepcopy(snapshots)
    broken["E1"]["identity"]["initialization_sha256"] = "other"
    with pytest.raises(ValueError, match="identity"):
        selection_from_snapshots(plan, broken, population)


def test_start_does_not_launch_unprepared_nodes(tmp_path, monkeypatch):
    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    plan = planned()
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    monkeypatch.setattr(cluster, "code_identity", lambda: plan["jobs"][0]["identity"]["code"])
    monkeypatch.setattr(cluster, "probe_node", lambda job: {"ready": False})
    monkeypatch.setattr(cluster, "launch_worker", lambda job: pytest.fail("unprepared node launched"))
    result = cluster.start_cluster(tmp_path)
    assert len(result) == 4
    assert all(row["status"] == "BLOCKED_PREPARATION" for row in result.values())


def test_completed_worker_launch_is_idempotent(tmp_path, monkeypatch):
    import subprocess
    import sys

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    job = {**planned()["jobs"][0], "root": str(tmp_path)}
    write_json(tmp_path / "WORKER_SPEC.json", job)
    write_json(tmp_path / "WORKER_STATE.json", {"status": "FULL_U_COMPLETE", "stage": "TRAIN_CAL"})
    monkeypatch.setattr(cluster, "ssh", lambda node, argv: subprocess.check_output([sys.executable, *argv[1:]], text=True))
    assert cluster.launch_worker(job) == {"status": "ALREADY_COMPLETE"}
    assert not (tmp_path / "WORKER_PID.json").exists()


def test_offline_collection_retains_charged_events_and_cannot_select(tmp_path, monkeypatch):
    import subprocess

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    plan = planned()
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    for job in plan["jobs"]:
        row = complete_snapshot(job)
        row["events"] = [{"event_id": job["arm"], "job_id": job["job_id"], "gpu_hours": 10.}]
        write_json(tmp_path / f"cluster/snapshots/{job['arm']}.json", row)
    def offline(*args):
        raise subprocess.CalledProcessError(255, ["ssh"])
    monkeypatch.setattr(cluster, "ssh", offline)
    result = cluster.collect_cluster(tmp_path)
    assert result["lifetime_gpu_hours"] == plan["prior"]["gpu_hours"] + 40.
    assert all(row["stale"] for row in result["snapshots"].values())
    with pytest.raises(ValueError, match="full trajectory"):
        cluster.selection_from_snapshots(plan, result["snapshots"], {"evaluation_counts": {"CAL": {}}})


def test_sel_gain_uses_e1_at_the_module_selected_step():
    from scripts.native_long_cluster import sel_comparisons

    plan = planned()
    snapshots = {j["arm"]: complete_snapshot(j) for j in plan["jobs"]}
    counts = {f"T{h}": h for h in range(1, 6)}
    lock = {"selected_updates": {"E0": 2970, "E1": 2970, "E2": 5940, "E3": 8910}}
    for arm, snapshot in snapshots.items():
        steps = [2970, 5940, 8910] if arm == "E1" else [lock["selected_updates"][arm]]
        snapshot["sel"] = [{**copy.deepcopy(snapshot["cal"][0]), "role": "SEL", "step": s,
            "metrics": {**{f"T{h}": s / 100000 for h in range(1, 6)}, "S_long": s / 100000}}
            for s in steps]
    result = sel_comparisons(plan, snapshots, {"evaluation_counts": {"SEL": counts}}, lock)
    assert result[1]["control_step"] == 5940 and result[1]["delta"]["S_long"] == 0.
    snapshots["E1"]["sel"] = snapshots["E1"]["sel"][:1]
    assert sel_comparisons(plan, snapshots, {"evaluation_counts": {"SEL": counts}}, lock) is None


def test_failed_module_does_not_block_completed_baseline_selection():
    from scripts.native_long_cluster import selection_from_snapshots

    plan = planned()
    snapshots = {j["arm"]: complete_snapshot(j) for j in plan["jobs"]}
    snapshots["E2"].update(status="FAILED", updates=990, cal=[])
    population = {"evaluation_counts": {"CAL": {f"T{h}": h for h in range(1, 6)}}}
    lock = selection_from_snapshots(plan, snapshots, population)
    assert lock["excluded_failed_arms"] == ["E2"]
    assert set(lock["selected_updates"]) == {"E0", "E1", "E3"}


def test_portable_data_preserves_external_annotation_content(tmp_path):
    from scripts.native_long_cluster import portable_data
    from scripts.short_module_screen import sha256, write_json

    data = tmp_path / "data"
    path = data / "processed/rio/train/scan.npy"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"processed vertices")
    external = tmp_path / "original-scans/scan/semseg.v2.json"
    external.parent.mkdir(parents=True)
    external.write_text("{}")
    write_json(tmp_path / "SOURCE_AND_INITIALIZATION.json", {"assets": {"data_root": str(data)}})
    write_json(tmp_path / "DATA_CONTENT_BINDING.json", {"files": [
        {"path": str(p), "sha256": sha256(p), "stat": {"bytes": p.stat().st_size}} for p in (path, external)]})
    manifest = portable_data(tmp_path)
    assert manifest["files"]["processed/rio/train/scan.npy"]["sha256"] == sha256(path)
    row = next(r for name, r in manifest["files"].items() if name.startswith("external_gt/"))
    assert row["sha256"] == sha256(external) and row["source"] == str(external)


def test_worker_config_uses_remote_metric_dataset(tmp_path):
    from scripts.native_long_campaign import PROJECT, compose_config
    from scripts.short_module_screen import write_json

    write_json(tmp_path / "WORKER_SPEC.json", {"data_root": "/remote/data", "encoder": "/remote/encoder.pth"})
    config = compose_config(PROJECT / "conf/config_native_long_retrain.yaml", root=tmp_path)
    assert config.instance_metric.dataset == "/remote/data/processed/rio/rio.yaml"
    assert config.backbone.name == "/remote/encoder.pth"


def test_collected_evidence_is_hashed_and_scoped(tmp_path, monkeypatch):
    from pathlib import Path

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import sha256

    job = planned()["jobs"][0]
    remote = Path(job["root"]) / "evaluation/E0/seed45/CAL-02970-T1-official-state.json"
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"official":true}')
    snapshot = {"cal": [{"evidence": [{"path": str(remote), "sha256": sha256(fixture)}]}], "sel": []}
    def transfer(argv, **kwargs):
        Path(argv[-1]).write_bytes(fixture.read_bytes())
    monkeypatch.setattr(cluster.subprocess, "run", transfer)
    cluster.collect_evidence(tmp_path, job, snapshot)
    local = tmp_path / "cluster/evidence/E0/seed45" / remote.name
    assert local.read_bytes() == fixture.read_bytes()
    snapshot["cal"][0]["evidence"][0]["sha256"] = "incorrect"
    with pytest.raises(ValueError, match="evidence SHA"):
        cluster.collect_evidence(tmp_path, job, snapshot)
    snapshot["cal"][0]["evidence"][0]["path"] = "/unrelated/private.json"
    with pytest.raises(ValueError, match="scope"):
        cluster.collect_evidence(tmp_path, job, snapshot)


def test_worker_resumes_uncommitted_optimizer_endpoint(tmp_path, monkeypatch):
    from scripts import native_long_cluster as cluster
    from scripts import native_long_execution as execution
    from scripts.short_module_screen import read_json, write_json

    job = planned()["jobs"][0]
    write_json(tmp_path / "DDP_PREFLIGHT.json", {"code": job["identity"]["code"],
               "status": "PASS", "optimizer_updates": 2})
    write_json(tmp_path / "TRAIN_PROGRESS.json", {"updates": 2970, "checkpoint_updates": 990})
    monkeypatch.setattr(cluster, "bootstrap_worker", lambda root: job)
    monkeypatch.setattr(cluster, "worker_snapshot", lambda root: {})
    monkeypatch.setattr(execution, "checkpoint_manifest", lambda root, arm: {
        "updates": read_json(root / "TRAIN_PROGRESS.json")["checkpoint_updates"]})
    phases = []
    def charge(root, command, *, phase, cards):
        phases.append(phase)
        if phase.startswith("train:"):
            step = int(command[-1])
            write_json(root / "TRAIN_PROGRESS.json", {"updates": step, "checkpoint_updates": step})
    monkeypatch.setattr(execution, "charged_process", charge)
    cluster.run_worker(tmp_path)
    assert phases[0] == "train:E0:2970"


def test_explicit_unstarted_amendment_checks_every_previous_worker(tmp_path, monkeypatch):
    import subprocess

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    plan = planned()
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    write_json(tmp_path / "RUN_STATE.json", {"arms": {a: {"seed45_updates": 0} for a in cluster.ARMS}})
    write_json(tmp_path / "POPULATION.json", {})
    write_json(tmp_path / "SOURCE_AND_INITIALIZATION.json", {"initialization": {"45": {"sha256": "common"}}})
    changed = cluster_config()
    changed["nodes"][0]["python"] = "/remote/env/bin/python"
    before = (tmp_path / "selection/BUDGET_LOCK.json").read_bytes()
    def busy(*args):
        raise subprocess.CalledProcessError(1, ["ssh"])
    monkeypatch.setattr(cluster, "ssh", busy)
    with pytest.raises(subprocess.CalledProcessError):
        cluster.amend_budget(tmp_path, changed, artifacts=tmp_path / "artifacts",
                             code=plan["jobs"][0]["identity"]["code"], amend_unstarted=True)
    assert (tmp_path / "selection/BUDGET_LOCK.json").read_bytes() == before
    checked = []
    monkeypatch.setattr(cluster, "ssh", lambda node, argv: checked.append(node["arm"]) or "")
    result = cluster.amend_budget(tmp_path, changed, artifacts=tmp_path / "artifacts",
                                  code=plan["jobs"][0]["identity"]["code"], amend_unstarted=True)
    assert checked == list(cluster.ARMS)
    assert result["jobs"][0]["python"] == "/remote/env/bin/python"


def test_partial_sel_dispatch_retries_without_reselecting_cal(tmp_path, monkeypatch):
    import subprocess

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import sha256, write_json

    plan = planned()
    write_json(tmp_path / "POPULATION.json", {"evaluation_counts": {"CAL": {f"T{h}": h for h in range(1, 6)}}})
    for job in plan["jobs"]:
        job["identity"]["population_sha256"] = sha256(tmp_path / "POPULATION.json")
    snapshots = {j["arm"]: complete_snapshot(j) for j in plan["jobs"]}
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    monkeypatch.setattr(cluster, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(cluster, "collect_cluster", lambda root: {"snapshots": snapshots})
    fail_once = [True]
    def transfer(node, argv):
        if node["arm"] == "E0" and fail_once:
            fail_once.pop()
            raise subprocess.CalledProcessError(255, ["ssh"])
        return ""
    monkeypatch.setattr(cluster, "ssh", transfer)
    monkeypatch.setattr(cluster.subprocess, "run", lambda *a, **k: None)
    launches = []
    monkeypatch.setattr(cluster, "launch_worker", lambda job, **kwargs: launches.append(job["arm"]) or {"status": "LAUNCHED"})
    first = cluster.finalize_cluster(tmp_path)
    assert first["SEL_dispatch"]["E0"]["status"] == "SEL_DISPATCH_FAILED"
    assert launches == ["E1", "E2", "E3"]
    before = (tmp_path / "selection/CAL_LOCK.json").read_bytes()
    snapshots["E1"]["status"] = "SEL_RUNNING"
    second = cluster.finalize_cluster(tmp_path)
    assert second["SEL_dispatch"]["E0"]["status"] == "LAUNCHED"
    assert (tmp_path / "selection/CAL_LOCK.json").read_bytes() == before


def test_remote_pythonpath_resolves_staged_native_dependencies():
    from scripts.native_long_cluster import remote_module

    job = planned()["jobs"][0]
    command = remote_module(job, "worker")
    value = next(p.split("=", 1)[1] for p in command if p.startswith("PYTHONPATH="))
    paths = value.split(":")
    assert paths[0] == job["repo"]
    assert all(f"{job['repo']}/third_party/{name}" in paths
               for name in ("concerto", "sonata", "stmetrics", "detectron2"))


def test_native_dependency_sources_use_bound_library_trees(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    libraries = {name: {"source": str(tmp_path / "bound" / name / name / "__init__.py")}
                 for name in ("concerto", "sonata", "stmetrics")}
    write_json(tmp_path / "SOURCE_AND_INITIALIZATION.json", {"installed_libraries": libraries})
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: SimpleNamespace(
        origin=str(tmp_path / "bound" / name / name / "__init__.py")))
    sources = cluster.native_dependency_sources(tmp_path)
    assert set(sources) == {"concerto", "sonata", "stmetrics", "detectron2", "pointnet2"}
    assert sources["concerto"] == tmp_path / "bound/concerto"
    assert sources["detectron2"] == tmp_path / "bound/detectron2"
    assert sources["pointnet2"] == cluster.PROJECT / "third_party/pointnet2"


def test_monitor_retries_when_another_controller_holds_lock(tmp_path, monkeypatch):
    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    write_json(tmp_path / "selection/BUDGET_LOCK.json", planned())
    calls, slept = [], []
    def collect(root):
        calls.append(1)
        if len(calls) == 1:
            with (cluster.exclusive(root / "cluster/controller.lock"),
                  cluster.exclusive(root / "cluster/controller.lock")):
                pytest.fail("concurrent controller lock acquired")
        return {"snapshots": {a: {"status": "NOT_STARTED"} for a in cluster.ARMS}}
    monkeypatch.setattr(cluster, "collect_cluster", collect)
    monkeypatch.setattr(cluster, "cluster_report", lambda root: {"gain": None})
    monkeypatch.setattr(cluster.time, "sleep", slept.append)
    result = cluster.watch_cluster(tmp_path, interval=1)
    assert result["status"] == "NO_RUNNING_WORKERS_CHECK_READINESS"
    assert len(calls) == 2 and slept == [1]


def test_ssh_and_rsync_reuse_the_same_cluster_connection(monkeypatch):
    from types import SimpleNamespace

    from scripts import native_long_cluster as cluster

    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="ok")
    monkeypatch.setattr(cluster.subprocess, "run", run)
    cluster.ssh(planned()["jobs"][0], ["true"])
    assert "ControlMaster=auto" in commands[0]
    assert "ControlPath=/tmp/rescene-native-long-%C" in commands[0]
    cluster.rsync(["-az", "local", "remote:path"])
    transport = commands[1][commands[1].index("-e") + 1]
    assert "ControlMaster=auto" in transport and "ControlPath=/tmp/rescene-native-long-%C" in transport


def test_worker_starts_in_staged_code_directory(tmp_path, monkeypatch):
    import subprocess
    import sys
    import time
    from pathlib import Path

    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import write_json

    job = {**planned()["jobs"][0], "repo": str(tmp_path / "code"), "root": str(tmp_path / "runtime")}
    root, repo = Path(job["root"]), Path(job["repo"])
    repo.mkdir()
    write_json(root / "WORKER_SPEC.json", job)
    receipt = tmp_path / "child-cwd.txt"
    command = [sys.executable, "-c",
               f"from pathlib import Path;Path({str(receipt)!r}).write_text(str(Path.cwd()))"]
    monkeypatch.setattr(cluster, "remote_module", lambda *args: command)
    monkeypatch.setattr(cluster, "ssh", lambda node, argv: subprocess.run(
        argv, cwd=tmp_path, capture_output=True, text=True, check=True).stdout)
    assert cluster.launch_worker(job)["status"] == "LAUNCHED"
    deadline = time.monotonic() + 5
    while not receipt.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert receipt.read_text() == str(repo)


def failed_recovery_fixture(tmp_path, monkeypatch):
    from scripts import native_long_cluster as cluster
    from scripts.short_module_screen import append_event, sha256, write_json

    write_json(tmp_path / "POPULATION.json", {"references": []})
    write_json(tmp_path / "SOURCE_AND_INITIALIZATION.json", {"initialization": {"45": {"sha256": "common"}}})
    write_json(tmp_path / "RUN_STATE.json", {"arms": {a: {"seed45_updates": 0} for a in cluster.ARMS}})
    plan = planned()
    for job in plan["jobs"]:
        job["identity"]["population_sha256"] = sha256(tmp_path / "POPULATION.json")
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    append_event(tmp_path / "COST_LEDGER.jsonl", {"event_id": "controller", "gpu_hours": .2})
    snapshots = {}
    for job in plan["jobs"]:
        row = complete_snapshot(job)
        row.update(status="FAILED", updates=0, observed_optimizer_updates=2, active=None,
                   cal=[], sel=[], events=[{"event_id": job["arm"], "job_id": job["job_id"], "gpu_hours": 1.}])
        snapshots[job["arm"]] = row
        write_json(tmp_path / f"cluster/snapshots/{job['arm']}.json", row)
    write_json(tmp_path / "cluster/STATUS.json", {"snapshots": snapshots})
    config = cluster_config()
    for node in config["nodes"]:
        node.update(root=node["root"] + "-retry", repo=node["repo"] + "-retry")
    monkeypatch.setattr(cluster, "ssh", lambda job, argv: json.dumps(snapshots[job["arm"]]) if argv[0] == "env" else "")
    monkeypatch.setattr(cluster, "portable_data", lambda root: {"sha256": plan["jobs"][0]["identity"]["data_content_sha256"]})
    return cluster, plan, config, snapshots


def test_failed_uncheckpointed_replacement_preserves_costs_and_history(tmp_path, monkeypatch):
    from scripts.native_long_budget import ledger_events
    from scripts.short_module_screen import sha256

    cluster, previous, config, _ = failed_recovery_fixture(tmp_path, monkeypatch)
    before = (tmp_path / "selection/BUDGET_LOCK.json").read_bytes()
    old_sha = sha256(tmp_path / "selection/BUDGET_LOCK.json")
    replacement = cluster.amend_budget(tmp_path, config, artifacts=tmp_path / "artifacts",
                                      code={"x": "fixed"}, replace_failed_uncheckpointed=True)
    assert replacement["current_campaign_gpu_hours"] == pytest.approx(4.2)
    events = ledger_events(tmp_path / "COST_LEDGER.jsonl")
    assert {e["event_id"] for e in events} == {"controller", *cluster.ARMS}
    assert (tmp_path / f"selection/budget_history/{old_sha}.json").read_bytes() == before
    archive = tmp_path / f"cluster/failures/{old_sha}"
    assert (archive / "snapshots/E0.json").exists() and (archive / "STATUS.json").exists()
    assert not (tmp_path / "cluster/snapshots").exists()
    assert not (tmp_path / "cluster/STATUS.json").exists()
    assert replacement["recovery"]["previous_job_ids"] == {j["arm"]: j["job_id"] for j in previous["jobs"]}
    quotas = sum(j["quota_gpu_hours"] for j in replacement["jobs"])
    assert quotas + replacement["reserve_gpu_hours"] + previous["prior"]["gpu_hours"] + 4.2 <= 1800
    assert cluster.amend_budget(tmp_path, config, artifacts=tmp_path / "artifacts", code={"x": "fixed"},
                                replace_failed_uncheckpointed=True) == replacement


@pytest.mark.parametrize("unsafe", ["running", "checkpoint", "active", "evidence", "stale"])
def test_failed_replacement_rejects_unsafe_trajectory(tmp_path, monkeypatch, unsafe):
    cluster, _, config, snapshots = failed_recovery_fixture(tmp_path, monkeypatch)
    row = snapshots["E0"]
    if unsafe == "running":
        row["status"] = "TRAIN_CAL_RUNNING"
    elif unsafe == "checkpoint":
        row["updates"] = 990
    elif unsafe == "active":
        row["active"] = {}
    elif unsafe == "evidence":
        row["cal"] = [{"uncompleted": True}]
    else:
        row["stale"] = True
    before = (tmp_path / "selection/BUDGET_LOCK.json").read_bytes()
    with pytest.raises(ValueError, match="failed.*uncheckpointed"):
        cluster.amend_budget(tmp_path, config, artifacts=tmp_path / "artifacts",
                             code={"x": "fixed"}, replace_failed_uncheckpointed=True)
    assert (tmp_path / "selection/BUDGET_LOCK.json").read_bytes() == before


def test_failed_replacement_requires_separate_code_and_worker_roots(tmp_path, monkeypatch):
    cluster, _, _, _ = failed_recovery_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="separate"):
        cluster.amend_budget(tmp_path, cluster_config(), artifacts=tmp_path / "artifacts",
                             code={"x": "fixed"}, replace_failed_uncheckpointed=True)
