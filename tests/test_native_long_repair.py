import copy
import json

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from scripts.native_long_budget import write_json_atomic as write_json
from tests.test_native_long_cluster import complete_snapshot, planned


def repair_inputs():
    old = planned()
    old_code = {"models/rescene.py": "model", "scripts/native_long_execution.py": "old"}
    for job in old["jobs"]:
        job["identity"]["code"] = old_code
    rows = {j["arm"]: copy.deepcopy(complete_snapshot(j)) for j in old["jobs"]}
    for job in old["jobs"]:
        rows[job["arm"]].update(status="FAILED", updates=2970, active=None)
        rows[job["arm"]]["events"] = [
            {"event_id": job["arm"], "job_id": job["job_id"], "gpu_hours": 37.}]
    return old, rows, {**old_code, "scripts/native_long_execution.py": "fixed"}


def test_repair_preserves_budget_positions_and_cost_provenance():
    from scripts.native_long_cluster import validate_snapshot
    from scripts.native_long_repair import build_repair_plan

    old, rows, code = repair_inputs()
    repaired = build_repair_plan(old, rows, code)
    assert repaired["prior"] == old["prior"]
    assert repaired["lifetime_cap_gpu_hours"] == old["lifetime_cap_gpu_hours"]
    assert repaired["physical_batch"] == old["physical_batch"]
    for before, after in zip(old["jobs"], repaired["jobs"]):
        assert after["quota_gpu_hours"] == before["quota_gpu_hours"]
        assert after["root"] == before["root"]
        assert after["job_id"] != before["job_id"]
        assert after["repair"]["resume_update"] == 2970
        row = copy.deepcopy(rows[before["arm"]])
        row.update(job_id=after["job_id"], identity=after["identity"], cal=[])
        validate_snapshot(row, after)
        row["events"][0]["event_id"] = "not-carried"
        with pytest.raises(ValueError, match="ledger"):
            validate_snapshot(row, after)


@pytest.mark.parametrize("problem", ["running", "active", "stale", "model", "data", "removed"])
def test_repair_rejects_unsafe_or_numerical_changes(problem):
    from scripts.native_long_repair import build_repair_plan

    old, rows, code = repair_inputs()
    if problem == "running":
        rows["E0"]["status"] = "TRAIN_CAL_RUNNING"
    elif problem == "active":
        rows["E0"]["active"] = {"pid": 123}
    elif problem == "stale":
        rows["E0"]["stale"] = True
    elif problem == "model":
        code["models/rescene.py"] = "changed"
    elif problem == "data":
        rows["E0"]["identity"]["population_sha256"] = "changed"
    else:
        code.pop("models/rescene.py")
    with pytest.raises(ValueError):
        build_repair_plan(old, rows, code)


def test_checkpoint_metadata_migration_preserves_every_training_state(tmp_path):
    from scripts.native_long_repair import checkpoint_state_digest, migrate_checkpoint

    identity = {"code": {"execution": "old"}, "config": {"batch": 32}}
    saved = {"global_step": 2970, "state_dict": {"weight": torch.randn(3, 4)},
             "hyper_parameters": OmegaConf.create({"config": {"batch": 32, "layers": [1, 2]}}),
             "hparams_type": DictConfig,
             "optimizer_states": [{"state": {0: {"step": torch.tensor(2970.),
                                                   "exp_avg": torch.randn(3, 4)}}}],
             "lr_schedulers": [{"total_steps": 29700, "last_epoch": 2970}],
             "loops": {"completed": 2970}, "callbacks": {},
             "native_long_resume": {"schema": 1, "next_global_draw": 95040,
                 "identity": identity, "rng_by_rank": [
                     {"numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
                     for _ in range(2)]}}
    source, target = tmp_path / "old.ckpt", tmp_path / "new.ckpt"
    torch.save(saved, source)
    original_bytes = source.read_bytes()
    desired = {**identity, "code": {"execution": "fixed"}}
    receipt = migrate_checkpoint(source, target, expected_identity=identity,
                                 desired_identity=desired, step=2970)
    migrated = torch.load(target, weights_only=False)
    assert source.read_bytes() == original_bytes
    assert migrated["native_long_resume"]["identity"] == desired
    assert checkpoint_state_digest(saved) == checkpoint_state_digest(migrated) == receipt["state_sha256"]
    damaged = copy.deepcopy(migrated)
    damaged["optimizer_states"][0]["state"][0]["exp_avg"][0, 0] += 1
    assert checkpoint_state_digest(damaged) != receipt["state_sha256"]
    with pytest.raises(ValueError, match="identity"):
        migrate_checkpoint(source, target, expected_identity=desired, desired_identity=identity, step=2970)
    with pytest.raises(ValueError, match="config"):
        migrate_checkpoint(source, target, expected_identity=identity,
                           desired_identity={**desired, "config": {"batch": 16}}, step=2970)


def test_worker_snapshot_does_not_reassign_historical_costs(tmp_path):
    from scripts.native_long_cluster import worker_snapshot

    job = planned()["jobs"][0]
    write_json(tmp_path / "WORKER_SPEC.json", job)
    event = {"event_id": "past", "job_id": "previous", "gpu_hours": 2.}
    (tmp_path / "COST_LEDGER.jsonl").write_text(json.dumps(event) + "\n")
    assert worker_snapshot(tmp_path)["events"] == [event]


def test_watch_reports_all_failed_without_attempting_selection(tmp_path, monkeypatch):
    from scripts import native_long_cluster as cluster

    plan, snapshots, _ = repair_inputs()
    write_json(tmp_path / "selection/BUDGET_LOCK.json", plan)
    monkeypatch.setattr(cluster, "collect_cluster", lambda root: {
        "snapshots": snapshots, "lifetime_gpu_hours": 230., "live_gpu_hours": 0.})
    monkeypatch.setattr(cluster, "cluster_report", lambda root: {"gain": None})
    monkeypatch.setattr(cluster, "finalize_cluster", lambda *a, **k: pytest.fail("failed jobs selected"))
    assert cluster.watch_cluster(tmp_path)["status"] == "ALL_TRAINING_FAILED_CHECK_WORKER_LOGS"


@pytest.mark.parametrize("path", ["ACTIVE_PROCESS.json", "selection/CAL_LOCK.json"])
def test_repair_refuses_unresolved_or_selected_worker(tmp_path, path):
    from scripts.native_long_repair import assert_stopped

    write_json(tmp_path / path, {})
    with pytest.raises(ValueError, match="active reservations or selected"):
        assert_stopped(tmp_path)


def test_worker_repair_archives_evidence_and_is_idempotent(tmp_path, monkeypatch):
    from scripts import native_long_campaign as campaign
    from scripts import native_long_execution as execution
    from scripts import native_long_repair as repair
    from scripts.native_long_budget import ledger_events
    from scripts.native_long_cluster import worker_snapshot
    from scripts.short_module_screen import sha256

    old, _, code = repair_inputs()
    job = old["jobs"][0]
    write_json(tmp_path / "WORKER_SPEC.json", job)
    write_json(tmp_path / "WORKER_STATE.json", {"status": "FAILED"})
    write_json(tmp_path / "TRAIN_PROGRESS.json", {"updates": 2970, "checkpoint_updates": 2970})
    identity = {**job["identity"], "config": {"batch": 32}}
    saved = {"global_step": 2970, "state_dict": {"weight": torch.ones(2)},
             "lr_schedulers": [{"last_epoch": 2970, "total_steps": 29700}],
             "native_long_resume": {"schema": 1, "next_global_draw": 95040,
                                     "identity": identity, "rng_by_rank": [{}, {}]}}
    directory = tmp_path / "training/E0/seed45"
    directory.mkdir(parents=True)
    for name in ("last.ckpt", "update=02970.ckpt"):
        torch.save(saved, directory / name)
    original_sha = sha256(directory / "last.ckpt")
    (tmp_path / "COST_LEDGER.jsonl").write_text(json.dumps({"event_id": "past", "gpu_hours": 3.}) + "\n")
    row = {"arm": "E0", "seed": 45, "role": "CAL", "step": 2970, "status": "COMPLETE",
           "identity": {"training": identity, "checkpoint_sha256": sha256(directory / "update=02970.ckpt")},
           "evidence": []}
    write_json(tmp_path / "evaluation/E0/seed45/CAL-02970.json", row)
    snapshots = {"E0": worker_snapshot(tmp_path)}
    old["jobs"] = [job]
    new = repair.build_repair_plan(old, snapshots, code)
    request = {"repair_id": "test", "previous": old, "plan": new, "snapshots": snapshots}
    actual_code = [job["identity"]["code"]]
    monkeypatch.setattr(repair, "code_identity", lambda: actual_code[0])
    monkeypatch.setattr(repair, "PROJECT", tmp_path / "code")
    monkeypatch.setattr(campaign, "compose_config", lambda *a, **kw: None)
    desired = {**identity, "code": code}
    monkeypatch.setattr(execution, "native_training_identity", lambda *a: desired)
    repair.worker_repair(tmp_path, request, prepare_only=True)
    assert (tmp_path / "REPAIR_PENDING.json").exists()
    actual_code[0] = code
    receipt = repair.worker_repair(tmp_path, request)
    assert receipt["preserved_gpu_hours"] == 3.
    assert receipt["archived_CAL_count"] == 1
    assert not (tmp_path / "REPAIR_PENDING.json").exists()
    assert not (tmp_path / "evaluation").exists()
    assert sha256(tmp_path / "repairs/test/last.ckpt") == original_sha
    before = sha256(directory / "last.ckpt")
    assert repair.worker_repair(tmp_path, request) == receipt
    assert sha256(directory / "last.ckpt") == before
    assert ledger_events(tmp_path / "COST_LEDGER.jsonl")[0]["job_id"] == job["job_id"]
    current = worker_snapshot(tmp_path)
    repair.validate_snapshot(current, new["jobs"][0])
