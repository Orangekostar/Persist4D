import json
import subprocess
import sys

import pytest


def setup_budget(tmp_path, monkeypatch, cap):
    from scripts import native_long_execution as execution
    from scripts.short_module_screen import write_json

    monkeypatch.setattr(execution, "ARTIFACTS", tmp_path / "artifacts")
    write_json(tmp_path / "selection/BUDGET_LOCK.json",
               {"lifetime_cap_gpu_hours": cap, "prior": {"gpu_hours": 200.}})
    (tmp_path / "COST_LEDGER.jsonl").write_text("")
    return execution


def test_process_uses_amended_cap_and_records_actual_reservation(tmp_path, monkeypatch):
    execution = setup_budget(tmp_path, monkeypatch, 1800.)
    execution.charged_process(tmp_path, [sys.executable, "-c", "pass"], phase="fixture", cards=2)
    row = json.loads((tmp_path / "COST_LEDGER.jsonl").read_text())
    assert row["cards"] == 2 and row["exit_code"] == 0 and row["gpu_hours"] > 0
    assert not (tmp_path / "ACTIVE_PROCESS.json").exists()


def test_exhausted_budget_never_starts_process(tmp_path, monkeypatch):
    execution = setup_budget(tmp_path, monkeypatch, 192.)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("process launched"))
    with pytest.raises(RuntimeError, match="exhausted"):
        execution.charged_process(tmp_path, ["unused"], phase="fixture", cards=2)


def test_timeout_stops_process_group_and_charges_failure(tmp_path, monkeypatch):
    execution = setup_budget(tmp_path, monkeypatch, 1800.)
    with pytest.raises(subprocess.TimeoutExpired):
        execution.charged_process(tmp_path, [sys.executable, "-c", "import time; time.sleep(5)"],
                                  phase="fixture-timeout", cards=1, max_gpu_hours=.0001)
    row = json.loads((tmp_path / "COST_LEDGER.jsonl").read_text())
    assert row["exit_code"] == -1 and row["gpu_hours"] > 0
    assert not (tmp_path / "ACTIVE_PROCESS.json").exists()


def test_unreconciled_reservation_blocks_duplicate_gpu_process(tmp_path, monkeypatch):
    from scripts.short_module_screen import write_json

    execution = setup_budget(tmp_path, monkeypatch, 1800.)
    write_json(tmp_path / "ACTIVE_PROCESS.json", {"event_id": "orphan", "pid": 123})
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("duplicate process launched"))
    with pytest.raises(RuntimeError, match="unreconciled"):
        execution.charged_process(tmp_path, ["unused"], phase="fixture", cards=2)


def test_optimizer_stop_includes_process_startup_time(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from scripts import native_long_execution as execution

    monkeypatch.setenv("RESCENE_NATIVE_PROCESS_GPUH_LIMIT", ".1")
    monkeypatch.setenv("RESCENE_NATIVE_PROCESS_STARTED_MONOTONIC", "1")
    ticks = iter((90., 110.))
    monkeypatch.setattr(execution.time, "perf_counter", lambda: next(ticks))
    saved = []
    trainer = SimpleNamespace(global_step=1, world_size=2, is_global_zero=True,
                              save_checkpoint=saved.append, should_stop=False)
    module = SimpleNamespace(device=torch.device("cpu"))
    callback = execution.UpdateBoundary(tmp_path, "E0", 45, 2970)
    callback.on_train_batch_end(trainer, module, None, None, 0)
    assert trainer.should_stop and len(saved) == 2
