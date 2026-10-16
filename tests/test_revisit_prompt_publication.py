"""Publication identity and frozen inference sources must survive final delivery."""
import json

import pytest

from scripts import publish_revisit_prompt as publication


@pytest.fixture
def delivery(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    public = project / 'artifacts/revisit_prompt_v1'
    public.mkdir(parents=True)
    root = tmp_path / 'run'
    root.mkdir()
    monkeypatch.setattr(publication, 'PROJECT', project)
    monkeypatch.setattr(publication, 'ARTIFACTS', public)
    monkeypatch.setattr(publication, 'NUMERIC_FILES', ())
    assets = dict(run_root=str(root))
    (public / 'STATUS.json').write_text(json.dumps(dict(
        publication_ready=True, prompt_audit_execution_complete=True)))
    for name in ('REPORT.md', 'PROMPT_AUDIT.md'):
        (public / name).write_text('completed review')
    (public / 'MODEL_MANIFEST.json').write_text('{"assets": []}')
    return assets, project, public, root


def test_explicit_new_publication_identity_is_used_in_preview(delivery):
    assets, _, _, root = delivery
    result = publication.publish(assets, execute=False,
        branch='research/formal-run', tag='formal-run-v1')
    assert result == dict(status='DRY_RUN_NOT_PUBLISHED', branch='research/formal-run',
        tag='formal-run-v1', publication_ready=True, missing=[])
    assert not (root / 'publication/PUBLICATION_PENDING.json').exists()


def test_wrong_origin_is_rejected_before_publication_mutations(delivery, monkeypatch):
    assets, _, _, root = delivery

    def git(*args, **kwargs):
        if args == ('branch', '--show-current'):
            return publication.BRANCH
        if args == ('remote', 'get-url', 'origin'):
            return 'git@github.com:someone-else/Persist4D.git'
        pytest.fail(f'Publication mutated before validating the remote: {args}')

    monkeypatch.setattr(publication, 'git', git)
    with pytest.raises(ValueError, match='Orangekostar/Persist4D'):
        publication.publish(assets, execute=True)
    assert not (root / 'publication/PUBLICATION_PENDING.json').exists()


def test_frozen_inference_sources_are_included_and_validated(delivery):
    _, project, public, _ = delivery
    source = project / 'models/frozen_inference.py'
    source.parent.mkdir()
    source.write_text('frozen inference implementation')
    lock = public / 'EVALUATION_POLICY_V4.json'
    lock.write_text(json.dumps(dict(policy_id='ordered-sparse-reference-v4',
        policy_sources={'models/frozen_inference.py': publication.sha256(source)})))
    (public / 'STATUS.json').write_text(json.dumps(dict(evaluation_policy=dict(
        policy_id='ordered-sparse-reference-v4', policy_sha256=publication.sha256(lock)))))
    assert source in publication.phase_a_files()
    source.write_text('changed after evaluation')
    with pytest.raises(ValueError, match='source identity'):
        publication.phase_a_files()
