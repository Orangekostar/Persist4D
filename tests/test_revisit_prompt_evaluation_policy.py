"""Evaluation policies must not silently reuse an incompatible numerical run."""
import json

import pytest

from scripts import revisit_prompt_evaluation_policy as policy


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def locked_policy(tmp_path, monkeypatch):
    project, root, public = tmp_path / "project", tmp_path / "run", tmp_path / "public"
    root.mkdir(); public.mkdir(); project.mkdir()
    monkeypatch.setattr(policy, "PROJECT", project)
    monkeypatch.setattr(policy, "ARTIFACTS", public)
    sources = {"frozen.py": "original-training-source"}
    monkeypatch.setattr(policy, "numeric_sources", lambda: sources)
    for name in policy.POLICY_FILES:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pinned policy code: " + name)
    (root / "RUN_LOCK.json").write_bytes(b"immutable original training lock")
    (root / "COST_LEDGER.jsonl").write_text("")
    (root / "trajectories/M1-s146").mkdir(parents=True)
    (root / "trajectories/M1-s146/delta-000400.pt").write_bytes(b"unchanged completed weights")
    assets = dict(run_root=str(root), python="owned-python", budget={"new_cap_gpu_hours":100.})
    lock = dict(status="LOCKED", policy_id=policy.POLICY_ID, namespace=policy.NAMESPACE,
                training_numeric_sources=sources,
                training_run_lock_sha256=policy.sha256(root / "RUN_LOCK.json"),
                policy_sources={name:policy.sha256(project / name) for name in policy.POLICY_FILES})
    for p in (root / policy.LOCK_NAME, public / policy.LOCK_NAME):
        _write(p, lock)
    return assets, root, public, lock


def test_namespace_reuses_weights_and_one_cost_ledger_without_copying(tmp_path, monkeypatch):
    assets, root, _, _ = locked_policy(tmp_path, monkeypatch)
    lock = policy.load_policy(assets)
    scoped = policy.prepare_namespace(assets, lock)
    namespace = policy.evaluation_root(assets, lock)
    assert scoped["budget"] == assets["budget"]
    assert (namespace / "RUN_LOCK.json").resolve() == root / "RUN_LOCK.json"
    assert (namespace / "COST_LEDGER.jsonl").resolve() == root / "COST_LEDGER.jsonl"
    assert (namespace / "trajectories/M1-s146/delta-000400.pt").read_bytes() == b"unchanged completed weights"
    assert not (namespace / "evaluation").exists()


def test_changed_policy_source_is_rejected_before_job_launch(tmp_path, monkeypatch):
    assets, _, _, _ = locked_policy(tmp_path, monkeypatch)
    (policy.PROJECT / policy.POLICY_FILES[0]).write_text("modified after freeze")
    with pytest.raises(ValueError, match="policy source identity"):
        policy.load_policy(assets)


def test_training_source_registry_stays_exact(tmp_path, monkeypatch):
    assets, _, _, _ = locked_policy(tmp_path, monkeypatch)
    monkeypatch.setattr(policy, "numeric_sources", lambda: {"frozen.py":"changed"})
    with pytest.raises(ValueError, match="training numerical source"):
        policy.load_policy(assets)


def test_runtime_and_public_policy_locks_must_match(tmp_path, monkeypatch):
    assets, _, public, _ = locked_policy(tmp_path, monkeypatch)
    (public / policy.LOCK_NAME).write_text("{}")
    with pytest.raises(ValueError, match="runtime/public evaluation policy"):
        policy.load_policy(assets)


def test_original_result_without_policy_marker_is_rejected(tmp_path, monkeypatch):
    assets, _, _, lock = locked_policy(tmp_path, monkeypatch)
    result = dict(status="COMPLETE", request=dict(numeric_sources=lock["training_numeric_sources"]))
    with pytest.raises(ValueError, match="evaluation policy identity"):
        policy.validate_result(result, policy.policy_identity(assets, lock), lock)


def test_result_marker_binds_the_policy_and_preserves_training_sources(tmp_path, monkeypatch):
    assets, _, _, lock = locked_policy(tmp_path, monkeypatch)
    identity = policy.policy_identity(assets, lock)
    result = dict(status="COMPLETE", evaluation_policy=identity,
                  request=dict(evaluation_policy=identity, numeric_sources=lock["training_numeric_sources"]))
    policy.validate_result(result, identity, lock)
    result["request"]["evaluation_policy"] = {**identity, "policy_sha256":"different policy"}
    with pytest.raises(ValueError, match="evaluation policy identity"):
        policy.validate_result(result, identity, lock)


def test_annotation_does_not_modify_official_metric_evidence(tmp_path):
    writes = []
    identity = dict(policy_id="stable-mean-v1", policy_sha256="policy")
    writer = policy.annotated_writer(lambda path, value: writes.append(value), identity)
    evidence = dict(schema_version=1, sha256="official-state", states=[])
    writer(tmp_path / "official-long-T3.json", evidence)
    assert writes[-1] == evidence
    result = dict(status="COMPLETE", request=dict(numeric_sources={"frozen.py":"same"}))
    writer(tmp_path / "RESULTS.json", result)
    assert writes[-1]["request"]["evaluation_policy"] == identity
    assert writes[-1]["request"]["numeric_sources"] == result["request"]["numeric_sources"]
    assert "evaluation_policy" not in result


def test_real_ensure_job_routes_evaluation_to_namespace_and_original_cost_pool(tmp_path, monkeypatch):
    from scripts import revisit_prompt_campaign as campaign
    assets, root, _, lock = locked_policy(tmp_path, monkeypatch)
    identity = policy.policy_identity(assets, lock)
    monkeypatch.setattr(campaign, "numeric_sources", lambda: lock["training_numeric_sources"])
    calls = []
    namespace = policy.evaluation_root(assets, lock)
    status_path = namespace / "evaluation/M1-s146-u000100/CAL/repeat0/STATUS.json"
    original_path = root / "evaluation/M1-s146-u000100/CAL/repeat0/STATUS.json"
    _write(original_path, dict(status="COMPLETE", numeric_sources=lock["training_numeric_sources"]))
    original_bytes = original_path.read_bytes()

    def external_child(bound_root, command, **kwargs):
        calls.append((bound_root, command, kwargs))
        _write(status_path, dict(status="COMPLETE", numeric_sources=lock["training_numeric_sources"],
                                 evaluation_policy=identity))
        return dict(event_id="owned-policy-evaluation")

    monkeypatch.setattr(campaign, "charged_job", external_child)
    with policy.campaign_policy(assets, lock) as scoped:
        command = campaign.child_command(scoped, "scripts.evaluate_revisit_prompt", "--role", "CAL")
        campaign.ensure_job(scoped, command, status_path=status_path, phase="P3",
                            job_key="eval:M1:s146:u100:CAL:r0", complete_status="COMPLETE")
    assert len(calls) == 1
    assert str(calls[0][0]) == str(root)
    assert calls[0][1][2] == "scripts.evaluate_revisit_prompt_policy"
    assert calls[0][2]["job_key"] == "policy:stable-mean-v1:eval:M1:s146:u100:CAL:r0"
    assert original_path.read_bytes() == original_bytes
    assert list((namespace / "jobs").glob("*.json"))


def test_completed_unmarked_namespace_status_is_not_reused(tmp_path, monkeypatch):
    from scripts import revisit_prompt_campaign as campaign
    assets, _, _, lock = locked_policy(tmp_path, monkeypatch)
    path = policy.evaluation_root(assets, lock) / "evaluation/M0-s146-u000000/CAL/repeat0/STATUS.json"
    _write(path, dict(status="COMPLETE", numeric_sources=lock["training_numeric_sources"]))
    with policy.campaign_policy(assets, lock) as scoped:
        with pytest.raises(ValueError, match="completed child evaluation policy identity"):
            campaign.ensure_job(scoped, ["unused"], status_path=path, phase="P1",
                                job_key="eval:M0:s146:u0:CAL:r0", complete_status="COMPLETE")
