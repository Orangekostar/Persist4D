"""Package only audited official metric states; exclude point-level data and weights."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def official_bytes(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def validate_state(state):
    if set(state) != {'mode', 'schema_version', 'sha256', 'states', 'updates'}:
        raise ValueError('not an official sufficient-state-only payload')
    core = {k: v for k, v in state.items() if k != 'sha256'}
    if hashlib.sha256(official_bytes(core)).hexdigest() != state['sha256']:
        raise ValueError('official state content digest differs')


def run(root, destination):
    parent = root / 'cluster/postselection'
    destination.mkdir(parents=True, exist_ok=True)
    audits = {name: read_json(parent / (name + '.json')) for name in
              ('FORMAL_AUDIT', 'BRIDGE_AUDIT', 'MECHANISM_AUDIT', 'MECHANISM_REPLAY_AUDIT')}
    if any(a['status'] != 'COMPLETE' for a in audits.values()):
        raise ValueError('all metric audits must complete before packaging')
    entries = []

    def add(path, member, expected_sha):
        if sha256(path) != expected_sha:
            raise ValueError('audited metric source changed: ' + str(path))
        state = read_json(path)
        validate_state(state)
        entries.append((member, path.read_bytes(), {'source_sha256': expected_sha,
            'official_state_sha256': state['sha256'], 'updates': state['updates'], 'mode': state['mode']}))

    project = Path(__file__).resolve().parents[1]
    analysis = read_json(project / 'artifacts/native_long_postselection_v1/ANALYSIS.json')
    for item in analysis['official_evidence_verified']:
        path = Path(item['path'])
        add(path, 'training-evaluation/' + str(path.relative_to(root / 'cluster/evidence')), item['sha256'])
    if len(entries) != 230:
        raise ValueError('training evaluation coverage differs')
    formal = []
    for source in audits['FORMAL_AUDIT']['source_receipts']:
        path = Path(source['path'])
        if sha256(path) != source['sha256']:
            raise ValueError('formal audited receipt changed')
        row = read_json(path)
        for item in row['evidence'] + row['reference_evidence']:
            state_path = path.parent / Path(item['path']).name
            member = 'formal/' + str(state_path.relative_to(parent / 'collected'))
            formal.append(member)
            add(state_path, member, item['sha256'])
    if len(formal) != 596:
        raise ValueError('formal evaluation coverage differs')
    for source in audits['BRIDGE_AUDIT']['source_receipts']:
        path = Path(source['path'])
        if path.parent.name == 'official':
            add(path, 'bridge/' + str(path.relative_to(parent / 'collected/E3/memory-bridge')), source['sha256'])
    mechanism = audits['MECHANISM_AUDIT']
    for row in mechanism['pooled_metrics'] + mechanism['by_reference']:
        name = row['official_file']
        add(parent / 'collected/E0/mechanism-audit/official' / name,
            'mechanism/' + name, row['official_file_sha256'])
    for source in audits['MECHANISM_REPLAY_AUDIT']['source_receipts']:
        if source['repeat'] == 0:
            continue
        path = parent / 'collected/E0/mechanism-replay' / (source['name'] + '.json')
        if sha256(path) != source['sha256']:
            raise ValueError('repeat receipt changed')
        state = read_json(path)['official_state']
        validate_state(state)
        entries.append(('runtime-repeat/' + path.name, official_bytes(state), {
            'source_sha256': source['sha256'], 'official_state_sha256': state['sha256'],
            'updates': state['updates'], 'mode': state['mode']}))
    counts = {group: sum(name.startswith(group + '/') for name, _, _ in entries)
              for group in ('training-evaluation', 'formal', 'bridge', 'mechanism', 'runtime-repeat')}
    if counts != {'training-evaluation': 230, 'formal': 596, 'bridge': 184, 'mechanism': 75, 'runtime-repeat': 8}:
        raise ValueError('official package coverage differs: ' + str(counts))
    if len({name for name, _, _ in entries}) != len(entries):
        raise ValueError('duplicate archive member')
    archive = destination / 'official_metric_states.zip'
    manifest = {'schema': 1, 'scope': 'Official sufficient statistics only; no raw coordinates, point masks or model weights',
                'counts': counts, 'files': [], 'audits': {name: sha256(parent / (name + '.json')) for name in audits}}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
        for name, payload, metadata in sorted(entries):
            stream.writestr(name, payload)
            manifest['files'].append({'member': name, 'bytes': len(payload),
                'sha256': hashlib.sha256(payload).hexdigest(), **metadata})
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None or set(stream.namelist()) != {r['member'] for r in manifest['files']}:
            raise ValueError('archive member or CRC check failed')
        for row in manifest['files']:
            data = stream.read(row['member'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('archive roundtrip differs')
    manifest.update(status='VERIFIED', archive=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive))
    write_json_atomic(destination / 'OFFICIAL_METRIC_STATES_MANIFEST.json', manifest)
    return {k: manifest[k] for k in ('status', 'counts', 'bytes', 'sha256')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root, args.destination))
