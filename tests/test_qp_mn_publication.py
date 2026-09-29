import importlib


def test_artifact_manifest_hashes_content_without_self_reference(tmp_path):
    api = importlib.import_module('scripts.qp_mn_publication')
    artifact = tmp_path / 'artifacts'
    artifact.mkdir()
    (artifact / 'result.csv').write_text('AP\n0.7\n')
    (artifact / 'ARTIFACT_MANIFEST.json').write_text('old manifest')
    result = api.artifact_manifest(artifact)
    assert [r['path'] for r in result['files']] == ['result.csv']
    assert result['files'][0]['bytes'] == len(b'AP\n0.7\n')
    assert result['files'][0]['sha256'] == api.hashlib.sha256(b'AP\n0.7\n').hexdigest()
