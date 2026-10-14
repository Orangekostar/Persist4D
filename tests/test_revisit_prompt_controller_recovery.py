"""Controller resource recovery preserves failed dispatches and completed models."""

import json

import pytest

from scripts import continue_revisit_prompt_formal as controller
from scripts import revisit_prompt_campaign as campaign
from scripts.revisit_prompt_assets import digest, numeric_sources


def _resource_blocked_run(tmp_path, monkeypatch, *, recovery):
    assets = dict(run_root=str(tmp_path), python="unused-gpu-worker", physical_gpu_index=1)
    if recovery:
        assets["dispatch_job_keys"] = {"M3": "train:M3:s146:resource-recovery1"}
    sources = numeric_sources()
    for variant in ("M1", "M2"):
        folder = tmp_path / "trajectories" / f"{variant}-s146"
        folder.mkdir(parents=True)
        (folder / "STATUS.json").write_text(json.dumps(dict(status="COMPLETE", numeric_sources=sources)))
        (folder / "delta-000400.pt").write_bytes(b"completed-model-must-stay-unchanged")
    journal = tmp_path / "jobs" / (digest("train:M3:s146") + ".json")
    journal.parent.mkdir()
    journal.write_text(json.dumps(dict(job_key="train:M3:s146", attempts=[
        dict(status="FAILED", error_fingerprint="gpu-busy", error="GPU occupied"),
        dict(status="FAILED", error_fingerprint="gpu-busy", error="GPU occupied")
    ])))
    original = journal.read_bytes()
    monkeypatch.setattr(controller, "local_assets", lambda: assets)
    monkeypatch.setattr(controller, "read_run_plan", lambda _: dict(N=400))
    monkeypatch.setattr(controller, "evaluate_endpoint", lambda *_: None)
    monkeypatch.setattr(controller, "fresh_m0_admission", lambda _: dict(status="ADMITTED"))
    monkeypatch.setattr(controller, "run_campaign", lambda *_, **__: dict(
        experiment_status="COMPLETE", scientific_status="INCONCLUSIVE"))

    # Only the external GPU child is substituted. Real ensure_job still reads,
    # rejects/reuses statuses and appends a distinct job journal on completion.
    def gpu_child(root, command, *, phase, job_key):
        variant = command[command.index("--variant") + 1]
        folder = root / "trajectories" / f"{variant}-s146"
        folder.mkdir(parents=True)
        (folder / "STATUS.json").write_text(json.dumps(dict(status="COMPLETE", numeric_sources=sources)))
        return dict(event_id="owned-child:" + job_key)

    monkeypatch.setattr(campaign, "charged_job", gpu_child)
    return journal, original


def test_explicit_resource_recovery_continues_without_resetting_retry_history(tmp_path, monkeypatch):
    journal, original = _resource_blocked_run(tmp_path, monkeypatch, recovery=True)
    assert controller.main() == 0
    assert journal.read_bytes() == original
    for variant in ("M1", "M2"):
        assert (tmp_path / "trajectories" / f"{variant}-s146" / "delta-000400.pt").read_bytes() == b"completed-model-must-stay-unchanged"
    for variant in ("M3", "M4", "M5", "M6"):
        status = json.loads((tmp_path / "trajectories" / f"{variant}-s146" / "STATUS.json").read_text())
        assert status["status"] == "COMPLETE"
    recovery = json.loads((tmp_path / "jobs" / (digest("train:M3:s146:resource-recovery1") + ".json")).read_text())
    assert recovery["attempts"][0]["status"] == "COMPLETE"
    assert json.loads((tmp_path / "CONTROLLER_STATUS.json").read_text())["stage"] == "FINISHED"


def test_without_explicit_recovery_repeated_failure_remains_stopped(tmp_path, monkeypatch):
    journal, original = _resource_blocked_run(tmp_path, monkeypatch, recovery=False)
    with pytest.raises(RuntimeError, match="same runtime error repeated"):
        controller.main()
    assert journal.read_bytes() == original
    assert not (tmp_path / "trajectories" / "M3-s146").exists()
    assert json.loads((tmp_path / "CONTROLLER_STATUS.json").read_text())["status"] == "FAILED"
