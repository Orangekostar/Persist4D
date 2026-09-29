"""Explicit scoped Git delivery, immutable new tag, and external byte receipt."""
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.qp_mn_binding import PROJECT
from scripts.short_module_identity import file_digest
from scripts.short_module_screen import read_json, write_json


def git(*args, binary=False):
    return subprocess.check_output(['git', *args], cwd=PROJECT, text=not binary)


def artifact_manifest(directory):
    directory = Path(directory)
    rows = [{'path': str(p.relative_to(directory)), 'sha256': file_digest(p), 'bytes': p.stat().st_size}
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json']
    binaries = [r for r in rows if Path(r['path']).suffix in ('.pt', '.zip')]
    if any(r['bytes'] > 20 * 1024**2 for r in binaries) or sum(r['bytes'] for r in binaries) > 50 * 1024**2:
        raise ValueError('required binary assets exceed direct Git budget; use authorized Release')
    return {'schema': 1, 'self_excluded': True, 'files': rows,
            'binary_bytes': sum(r['bytes'] for r in binaries), 'binary_count': len(binaries)}


def verify_publication(c, receipt):
    refs = dict(line.split()[::-1] for line in git('ls-remote', 'origin',
                f"refs/heads/{receipt['branch']}", f"refs/tags/{receipt['tag']}").splitlines())
    expected = receipt['commit_B']
    if refs.get(f"refs/heads/{receipt['branch']}") != expected or refs.get(f"refs/tags/{receipt['tag']}") != expected:
        raise ValueError('remote branch/tag do not match delivery B')
    git('fetch', '--no-tags', 'origin', f"refs/heads/{receipt['branch']}")
    directory = c.artifacts.relative_to(PROJECT)
    manifest = read_json(c.artifacts / 'ARTIFACT_MANIFEST.json')
    manifest_blob = git('cat-file', 'blob', f'{expected}:{directory}/ARTIFACT_MANIFEST.json', binary=True)
    if hashlib.sha256(manifest_blob).hexdigest() != receipt['artifact_manifest_sha256']:
        raise ValueError('committed artifact manifest differs from publication receipt')
    assets = []
    for row in manifest['files']:
        path = str(directory / row['path'])
        content = git('cat-file', 'blob', f'{expected}:{path}', binary=True)
        if hashlib.sha256(content).hexdigest() != row['sha256'] or len(content) != row['bytes']:
            raise ValueError(f'actual Git blob bytes differ: {path}')
        assets.append({**row, 'git_path': path, 'verified_commit': expected})
    # A is an actual ancestor; the original repaired base travels with the branch.
    git('merge-base', '--is-ancestor', c.config['base_commit'], expected)
    git('merge-base', '--is-ancestor', receipt['commit_A'], expected)
    return {**receipt, 'status': 'GIT_COMPLETE', 'release_status': 'NOT_REQUIRED',
            'verified_utc': datetime.now(timezone.utc).isoformat(), 'verified_assets': assets,
            'remote_refs': refs, 'repaired_base_available': True}


def publish(c):
    if not c.artifacts.is_relative_to(PROJECT):
        raise ValueError('publication artifact root must be inside the current worktree')
    branch = git('branch', '--show-current').strip()
    if branch != c.config['publication']['branch']:
        raise ValueError('refusing to publish a different branch')
    audit = read_json(c.artifacts / 'REQUIREMENT_REVIEW.json')
    if audit['scientific_review_status'] != 'PASS' or not audit['evidence']:
        raise ValueError('primary requirement review is not complete')
    for path, digest in audit['evidence'].items():
        if file_digest(path) != digest:
            raise ValueError(f'reviewed evidence changed: {path}')
    for required in ('FINAL_REPORT.md', 'REQUIREMENT_REVIEW.md', 'resources/PROFILE.csv',
                     'evidence/official_metric_states.zip', 'selection/SHORTLIST.json'):
        if not (c.artifacts / required).is_file():
            raise ValueError(f'missing required delivery: {required}')
    artifact_manifest(c.artifacts)
    scoped = [PROJECT / 'models/qp_mn_heads.py', PROJECT / 'configs/qp_mn_targeted_v1.yaml', c.artifacts,
              PROJECT / 'docs/0929/ReScene_QP_MN_Targeted_Execution',
              PROJECT / 'docs/superpowers/specs/2026-09-29-qp-mn-design.md',
              PROJECT / 'docs/superpowers/plans/2026-09-29-qp-mn.md']
    scoped += sorted((PROJECT / 'scripts').glob('qp_mn_*.py'))
    scoped += sorted((PROJECT / 'tests').glob('test_qp_mn_*.py'))
    paths = [str(p.relative_to(PROJECT)) for p in scoped]
    receipt_path = c.root / 'publication/PUBLICATION_RECEIPT.json'
    if receipt_path.exists() and not git('status', '--porcelain', '--', *paths).strip():
        result = verify_publication(c, read_json(receipt_path))
        write_json(receipt_path, result)
        return result
    # Never include an unrelated staged path in our commit.
    already_staged = git('diff', '--cached', '--name-only').splitlines()
    if any(not any(p == scope or p.startswith(scope + '/') for scope in paths) for p in already_staged):
        raise ValueError('unrelated staged user changes must not enter the experiment commit')
    git('add', '--', *paths)
    if git('diff', '--cached', '--name-only').strip():
        git('commit', '-m', 'Complete targeted Q-P and M-N experiments and evidence')
    commit_a = git('rev-parse', 'HEAD').strip()
    base_tag = c.config['publication']['tag']
    local_tags = set(git('tag', '--list').splitlines())
    remote_tags = {line.split()[1].removeprefix('refs/tags/') for line in git('ls-remote', '--tags', 'origin').splitlines()}
    tag, suffix = base_tag, 2
    while tag in local_tags or tag in remote_tags:
        tag = f'{base_tag}-r{suffix}'
        suffix += 1
    shortlist = read_json(c.artifacts / 'selection/SHORTLIST.json')
    handoff = (
        '# Q-P / M-N handoff\n\n'
        f'Experiment commit A: `{commit_a}`. Branch: `{branch}`. New delivery tag: `{tag}`.\n\n'
        f'Repaired base: `{c.config["base_commit"]}`. Runtime: `{c.root}`.\n\n'
        'Read `FINAL_REPORT.md`, `REQUIREMENT_REVIEW.md`, and `resources/NATIVE_RUNTIME_NOTE.md`. '
        'All five selected single-module bundles are under `training/<arm>/seed45/deployment.pt`; '
        'all 22 inference checkpoints are under `training/<arm>/seed<seed>/checkpoints/`. '
        'They require the separately held original R1, Concerto and legal dataset assets. '
        'Neither raw parent weights, raw data nor raw GT are redistributed.\n\n'
        f'Confirmed development modules: {[r["module"] for r in shortlist["confirmed"]]}. '
        f'Provisional engineering candidates: {[r["module"] for r in shortlist["provisional"]]}. '
        'Q-P and M-N do not beat B0. M-C is positive in both tested seeds but fails the '
        'seed45 reference-consistency gate. Keep B0 as the default; no combinations or formal test claims.\n\n'
        'Commands (existing persist4d environment):\n\n'
        '```bash\n'
        'python -m scripts.qp_mn_campaign status\n'
        'python -m scripts.qp_mn_campaign run --base-commit 465f37a --resume\n'
        'python -m scripts.qp_mn_campaign report\n'
        'python -m scripts.qp_mn_campaign publish --resume\n'
        '```\n\n'
        'The artifact manifest excludes itself and contains no self-referential B hash. '
        'The external `publication/PUBLICATION_RECEIPT.json` under the runtime root records '
        'actual A/B/tag and byte-verified remote delivery after the push. '
        'Git carries all required small assets; no Release is required.\n'
    )
    (c.artifacts / 'HANDOFF.md').write_text(handoff)
    write_json(c.artifacts / 'ARTIFACT_MANIFEST.json', artifact_manifest(c.artifacts))
    git('add', '--', str((c.artifacts / 'HANDOFF.md').relative_to(PROJECT)),
        str((c.artifacts / 'ARTIFACT_MANIFEST.json').relative_to(PROJECT)))
    git('commit', '-m', 'Record targeted experiment handoff and artifact manifest')
    commit_b = git('rev-parse', 'HEAD').strip()
    git('push', 'origin', f'HEAD:refs/heads/{branch}')
    git('tag', tag, commit_b)
    git('push', 'origin', f'refs/tags/{tag}')
    receipt = {'commit_A': commit_a, 'commit_B': commit_b, 'branch': branch, 'tag': tag,
               'scientific_status': audit['scientific_status'],
               'artifact_manifest_sha256': file_digest(c.artifacts / 'ARTIFACT_MANIFEST.json')}
    result = verify_publication(c, receipt)
    write_json(receipt_path, result)
    return result
