"""Centroid-v3 uses one original cost pool and never reuses v2 result rows."""
import json

import pytest

from scripts import revisit_prompt_centroid_policy as policy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def locked(tmp_path, monkeypatch):
    root, public, project = (tmp_path/name for name in ('run','public','project'))
    for p in (root,public,project): p.mkdir()
    monkeypatch.setattr(policy, 'ARTIFACTS', public)
    monkeypatch.setattr(policy, 'PROJECT', project)
    monkeypatch.setattr(policy, 'numeric_sources', lambda:{'training.py':'original'})
    monkeypatch.setattr(policy, 'runtime_identity', lambda:{'runtime':'original'})
    for name in policy.POLICY_FILES:
        p = project/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('frozen source: '+name)
    (root/'RUN_LOCK.json').write_text('original C3 lock')
    (root/'COST_LEDGER.jsonl').write_text('')
    (root/'trajectories').mkdir()
    lock = dict(status='LOCKED', policy_id=policy.POLICY_ID, namespace=policy.NAMESPACE,
        training_numeric_sources={'training.py':'original'}, training_run_lock_sha256=policy.sha256(root/'RUN_LOCK.json'),
        inference_runtime={'runtime':'original'}, policy_sources={name:policy.sha256(project/name) for name in policy.POLICY_FILES})
    for p in (root/policy.LOCK_NAME,public/policy.LOCK_NAME): write(p,lock)
    return dict(run_root=str(root),python='owned-python'), root, public, lock


def test_version_namespace_and_source_registry_are_separate(locked, monkeypatch):
    assets, root, public, lock = locked
    policy.load_policy(assets)
    assert policy.POLICY_ID == 'native-stable-centroids-v3'
    assert 'models/revisit_prompt_centroid_numerics.py' in policy.POLICY_FILES
    scoped = policy.prepare_namespace(assets, lock)
    n = policy.evaluation_root(assets, lock)
    assert (n/'COST_LEDGER.jsonl').resolve() == root/'COST_LEDGER.jsonl'
    assert (n/'trajectories').resolve() == root/'trajectories'
    assert not (n/'p5').is_symlink()
    assert not (n/'evaluation').exists()
    (policy.PROJECT/'models/revisit_prompt_centroid_numerics.py').write_text('changed after freeze')
    with pytest.raises(ValueError, match='policy source identity'):
        policy.load_policy(assets)


def test_v2_lock_and_v2_result_markers_fail_closed(locked):
    assets, root, public, lock = locked
    identity = policy.policy_identity(assets,lock)
    result = dict(evaluation_policy={'policy_id':'native-stable-mean-v2'},request={})
    with pytest.raises(ValueError,match='evaluation policy identity'):
        policy.validate_result(result,identity,lock)
    old = {**lock,'policy_id':'native-stable-mean-v2'}
    for p in (root/policy.LOCK_NAME,public/policy.LOCK_NAME): write(p,old)
    with pytest.raises(ValueError,match='status or namespace'):
        policy.load_policy(assets)


def test_v3_children_route_to_original_ledger_and_new_entry(locked,monkeypatch):
    from scripts import revisit_prompt_campaign as campaign
    assets,root,_,lock = locked
    identity = policy.policy_identity(assets,lock)
    path = policy.evaluation_root(assets,lock)/'evaluation/M1-s146-u000100/CAL/repeat0/STATUS.json'
    monkeypatch.setattr(campaign,'numeric_sources',lambda:lock['training_numeric_sources'])
    calls=[]

    def charged(bound_root,command,**kwargs):
        calls.append((bound_root,command,kwargs))
        write(path,dict(status='COMPLETE',numeric_sources=lock['training_numeric_sources'],evaluation_policy=identity))
        return {'event_id':'owned-v3-child'}

    monkeypatch.setattr(campaign,'charged_job',charged)
    with policy.campaign_policy(assets,lock) as scoped:
        command=campaign.child_command(scoped,'scripts.evaluate_revisit_prompt','--role','CAL')
        campaign.ensure_job(scoped,command,status_path=path,phase='P3',job_key='eval:M1:s146:u100:CAL:r0',complete_status='COMPLETE')
        p5=campaign.child_command(scoped,'scripts.revisit_prompt_campaign','p5')
    assert str(calls[0][0])==str(root)
    assert calls[0][1][2]==p5[2]=='scripts.evaluate_revisit_prompt_centroids'
    assert calls[0][2]['job_key']=='policy:native-stable-centroids-v3:eval:M1:s146:u100:CAL:r0'


@pytest.mark.parametrize('mode',['CAL','p5'])
def test_entry_applies_and_restores_both_inference_helpers(monkeypatch,tmp_path,mode):
    from scripts import evaluate_revisit_prompt_centroids as entry
    from scripts import revisit_prompt_assets as assets_module
    from scripts import revisit_prompt_campaign as campaign
    from scripts import evaluate_revisit_prompt as evaluation
    from models import revisit_prompt as prompt
    from models import revisit_prompt_numerics as means
    from models.revisit_prompt_centroid_numerics import stable_segment_context
    legacy=campaign if mode=='p5' else evaluation
    identity=dict(policy_id=policy.POLICY_ID,policy_sha256='locked-v3')
    scoped=dict(run_root=str(tmp_path/'v3'))
    monkeypatch.setattr(entry,'local_assets',lambda:{'run_root':str(tmp_path)})
    monkeypatch.setattr(entry,'load_policy',lambda assets:{})
    monkeypatch.setattr(entry,'policy_identity',lambda assets,lock:identity)
    monkeypatch.setattr(entry,'prepare_namespace',lambda assets,lock:scoped)
    monkeypatch.setattr(means,'install_evaluation_policy',lambda:None)
    writes=[]
    writer=lambda path,payload:writes.append(payload)
    monkeypatch.setattr(legacy,'write_json_atomic',writer)
    original_assets,original_build,original_context=legacy.local_assets,assets_module.build_system,prompt.segment_context

    def run(args):
        assert legacy.local_assets()==scoped
        assert prompt.segment_context is stable_segment_context
        assert assets_module.build_system is not original_build
        legacy.write_json_atomic(tmp_path/'RESULTS.json',dict(status='COMPLETE',request={}))
        return 0

    monkeypatch.setattr(legacy,'main',run)
    assert entry.main(['p5'] if mode=='p5' else ['--role','CAL'])==0
    assert writes[-1]['evaluation_policy']==writes[-1]['request']['evaluation_policy']==identity
    assert legacy.local_assets is original_assets and legacy.write_json_atomic is writer
    assert assets_module.build_system is original_build and prompt.segment_context is original_context


def test_controller_alignment_preserves_v2_summary(tmp_path):
    from scripts.continue_revisit_prompt_centroids import align_runtime_baseline
    path=tmp_path/'RUNTIME_SUMMARY.json'
    old=dict(evaluation_policy={'policy_id':'native-stable-mean-v2'},M0_CAL_repeat_AP=[[.2,.3,.4]],numerical_status='NONBITWISE_RUNTIME_OBSERVED',inherited_summary_before_native_sha256='v2-history-digest')
    write(path,old)
    previous=path.read_bytes()
    admission=dict(status='ADMITTED',epsilon_run_pp=0.,AP_by_repeat=[{str(h):.1*h for h in range(1,6)} for _ in range(3)])
    write(tmp_path/'EVALUATION_M0_ADMISSION_V3.json',admission)
    identity=dict(policy_id=policy.POLICY_ID,policy_sha256='v3')
    align_runtime_baseline(path,admission,identity)
    actual=json.loads(path.read_text())
    assert actual['evaluation_policy']==identity
    assert actual['M0_CAL_repeat_AP']==[[.1*h]*3 for h in range(1,6)]
    assert actual['fresh_policy_admission']=='EVALUATION_M0_ADMISSION_V3.json'
    assert (tmp_path/'RUNTIME_SUMMARY_BEFORE_CENTROIDS_V3.json').read_bytes()==previous
    assert actual['inherited_summary_before_centroids_sha256']==policy.sha256(tmp_path/'RUNTIME_SUMMARY_BEFORE_CENTROIDS_V3.json')
    assert actual['inherited_summary_before_native_sha256']=='v2-history-digest'
    assert actual['numerical_status']=='NONBITWISE_RUNTIME_OBSERVED'
