"""Native inference must preserve C3 weights and keep execution policies apart."""
import json

import pytest
import torch
from torch import nn
from spconv import constants
from spconv.core import ConvAlgo
from spconv.pytorch import SubMConv3d

from models.revisit_prompt_native_numerics import (
    evaluation_builders, pin_native_convolutions,
)
from scripts import revisit_prompt_native_policy as policy


def sparse_model():
    return nn.Sequential(SubMConv3d(2, 3, 3, algo=ConvAlgo.MaskImplicitGemm))


def test_native_pin_preserves_all_weight_values_layouts_and_grad_flags():
    model = sparse_model()
    before = {k: (v.clone(), v.stride(), v.data_ptr()) for k, v in model.state_dict().items()}
    grads = [p.requires_grad for p in model.parameters()]
    assert pin_native_convolutions(model) == ["0"]
    assert model[0].algo == ConvAlgo.Native
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k][0])
        assert v.stride() == before[k][1] and v.data_ptr() == before[k][2]
    assert [p.requires_grad for p in model.parameters()] == grads


@pytest.mark.parametrize("flag,reason", [("ALL_WEIGHT_IS_KRSC", "KRSC"), ("SPCONV_ALLOW_TF32", "TF32")])
def test_unsupported_runtime_rejected_before_any_algorithm_changes(monkeypatch, flag, reason):
    model = sparse_model()
    monkeypatch.setattr(constants, flag, flag == "SPCONV_ALLOW_TF32")
    with pytest.raises(ValueError, match=reason):
        pin_native_convolutions(model)
    assert model[0].algo == ConvAlgo.MaskImplicitGemm


def test_missing_sparse_parent_is_rejected():
    with pytest.raises(ValueError, match="sparse convolution"):
        pin_native_convolutions(nn.Linear(2, 3))


def test_builders_pin_after_loading_and_restore_even_on_failure(monkeypatch):
    from scripts import revisit_prompt_assets as assets
    from scripts import evaluate_revisit_prompt as evaluation
    built, loaded = sparse_model(), sparse_model()

    def build(*args, **kwargs):
        assert built[0].algo == ConvAlgo.MaskImplicitGemm
        return built, "original-config", "original-audit"

    def load(*args, **kwargs):
        assert loaded[0].algo == ConvAlgo.MaskImplicitGemm
        return loaded

    monkeypatch.setattr(assets, "build_system", build)
    monkeypatch.setattr(evaluation, "build_system", build)
    monkeypatch.setattr(assets, "load_revisit_bundle", load)
    with pytest.raises(RuntimeError, match="owned failure"):
        with evaluation_builders():
            assert evaluation.build_system({}) == (built, "original-config", "original-audit")
            # P5 imports these builders locally from assets.
            assert assets.load_revisit_bundle("immutable-delta") is loaded
            assert built[0].algo == loaded[0].algo == ConvAlgo.Native
            raise RuntimeError("owned failure")
    assert assets.build_system is evaluation.build_system is build
    assert assets.load_revisit_bundle is load


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def locked(tmp_path, monkeypatch):
    root, public, project = (tmp_path / name for name in ("run", "public", "project"))
    root.mkdir(); public.mkdir(); project.mkdir()
    monkeypatch.setattr(policy, "PROJECT", project)
    monkeypatch.setattr(policy, "ARTIFACTS", public)
    sources = {"training.py": "original"}
    monkeypatch.setattr(policy, "numeric_sources", lambda: sources)
    monkeypatch.setattr(policy, "runtime_identity", lambda: {"runtime": "original"})
    for name in policy.POLICY_FILES:
        p = project / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("frozen: " + name)
    (root / "RUN_LOCK.json").write_text("original training lock")
    (root / "COST_LEDGER.jsonl").write_text("")
    (root / "trajectories").mkdir()
    (root / "p5").mkdir()
    (root / "p5/PANEL.json").write_text('{"input_ids":["fixed-panel"]}')
    lock = dict(status="LOCKED", policy_id=policy.POLICY_ID, namespace=policy.NAMESPACE,
                training_numeric_sources=sources, training_run_lock_sha256=policy.sha256(root / "RUN_LOCK.json"),
                policy_sources={name: policy.sha256(project / name) for name in policy.POLICY_FILES},
                inference_runtime={"runtime": "original"})
    for p in (root / policy.LOCK_NAME, public / policy.LOCK_NAME):
        write(p, lock)
    return dict(run_root=str(root), python="owned-python", budget={"cap":100}), root, lock


def test_runtime_or_source_drift_is_rejected(locked, monkeypatch):
    assets, _, _ = locked
    policy.load_policy(assets)
    monkeypatch.setattr(policy, "runtime_identity", lambda: {"runtime": "changed"})
    with pytest.raises(ValueError, match="inference runtime"):
        policy.load_policy(assets)
    monkeypatch.setattr(policy, "runtime_identity", lambda: {"runtime": "original"})
    (policy.PROJECT / policy.POLICY_FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="policy source identity"):
        policy.load_policy(assets)


def test_native_namespace_shares_training_ledger_but_isolates_p5(locked):
    assets, root, lock = locked
    scoped = policy.prepare_namespace(assets, lock)
    namespace = policy.evaluation_root(assets, lock)
    assert scoped["budget"] == assets["budget"]
    assert (namespace / "trajectories").resolve() == root / "trajectories"
    assert (namespace / "COST_LEDGER.jsonl").resolve() == root / "COST_LEDGER.jsonl"
    assert not (namespace / "p5").is_symlink()
    assert (namespace / "p5/PANEL.json").read_bytes() == (root / "p5/PANEL.json").read_bytes()
    (namespace / "p5/PANEL.json").write_text('{"input_ids":["different"]}')
    with pytest.raises(ValueError, match="P5 panel"):
        policy.prepare_namespace(assets, lock)


def test_v1_result_cannot_be_reused_and_native_children_charge_original_pool(locked, monkeypatch):
    from scripts import revisit_prompt_campaign as campaign
    assets, root, lock = locked
    identity = policy.policy_identity(assets, lock)
    old = dict(status="COMPLETE", evaluation_policy={"policy_id":"stable-mean-v1"},
               request=dict(numeric_sources=lock["training_numeric_sources"]))
    with pytest.raises(ValueError, match="evaluation policy identity"):
        policy.validate_result(old, identity, lock)
    calls = []
    status_path = policy.evaluation_root(assets, lock) / "evaluation/M0-s146-u000000/CAL/repeat0/STATUS.json"
    monkeypatch.setattr(campaign, "numeric_sources", lambda: lock["training_numeric_sources"])

    def charged(bound_root, command, **kwargs):
        calls.append((bound_root, command, kwargs))
        write(status_path, dict(status="COMPLETE", numeric_sources=lock["training_numeric_sources"],
                                evaluation_policy=identity))
        return {"event_id":"owned-native-event"}

    monkeypatch.setattr(campaign, "charged_job", charged)
    with policy.campaign_policy(assets, lock) as scoped:
        command = campaign.child_command(scoped, "scripts.evaluate_revisit_prompt", "--role", "CAL")
        campaign.ensure_job(scoped, command, status_path=status_path, phase="P1",
                            job_key="eval:M0:s146:u0:CAL:r0", complete_status="COMPLETE")
        p5 = campaign.child_command(scoped, "scripts.revisit_prompt_campaign", "p5", "--mode", "profile")
    assert str(calls[0][0]) == str(root)
    assert calls[0][1][2] == p5[2] == "scripts.evaluate_revisit_prompt_native"
    assert calls[0][2]["job_key"] == "policy:native-stable-mean-v2:eval:M0:s146:u0:CAL:r0"


def test_baseline_alignment_uses_actual_fresh_arrays_without_erasing_old_evidence(tmp_path):
    from scripts.continue_revisit_prompt_native import align_runtime_baseline
    p = tmp_path / "RUNTIME_SUMMARY.json"
    old = dict(M0_CAL_repeat_AP=[[.2,.3,.4]], epsilon_run_pp=9., numerical_status="NONBITWISE_RUNTIME_OBSERVED")
    write(p, old)
    admission = dict(status="ADMITTED", epsilon_run_pp=0., AP_by_repeat=[{str(h): .1*h for h in range(1,6)} for _ in range(3)])
    identity = {"policy_id":policy.POLICY_ID, "policy_sha256":"new"}
    align_runtime_baseline(p, admission, identity)
    v = json.loads(p.read_text())
    assert v["M0_CAL_repeat_AP"] == [[.1*h]*3 for h in range(1,6)]
    assert v["evaluation_policy"] == identity and v["epsilon_run_pp"] == 0.
    assert v["numerical_status"] == "NONBITWISE_RUNTIME_OBSERVED"
    backup = p.with_name("RUNTIME_SUMMARY_BEFORE_NATIVE_V2.json")
    assert json.loads(backup.read_text()) == old
