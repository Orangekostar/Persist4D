"""Small inference-only checkpoints and independently verified metric statistics."""
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import torch

from scripts.p6a_metrics import recompute_official_metric_evidence
from scripts.qp_mn_evaluation import load_head
from scripts.short_module_evaluation import write_csv
from scripts.short_module_identity import file_digest
from scripts.short_module_screen import read_json, write_json


def verify_profile(context):
    c = context
    summary = read_json(c.artifacts / 'resources/PROFILE_SUMMARY.json')
    rows = list(csv.DictReader((c.artifacts / 'resources/PROFILE.csv').open()))
    modules = ('B0', 'Q2_F', 'Q_A0', 'Q_P', 'M_C', 'M_N')
    expected = {(arm, r['input_id'], repeat) for arm in modules
                for r in summary['selected_records'] for repeat in range(4)}
    observed = {(r['module'], r['input_id'], int(r['repeat'])) for r in rows}
    if len(rows) != 96 or len(expected) != 96 or observed != expected:
        raise ValueError('profile lacks the full prescribed input/module/repetition grid')
    if any((r['warmup'] == 'True') != (int(r['repeat']) == 0) or float(r['parent_network_seconds']) <= 0
           or float(r['end_to_end_seconds']) <= float(r['parent_network_seconds']) for r in rows):
        raise ValueError('profile timing or warmup scope differs')
    if {r['module'] for r in summary['reload_checks']} != set(modules[1:]):
        raise ValueError('selected head reload evidence missing')
    for row in summary['reload_checks']:
        bundle = c.artifacts / 'training' / row['module'] / 'seed45/deployment.pt'
        if file_digest(bundle) != row['bundle_sha256'] or not all(row['equal'].values()):
            raise ValueError('profile reload bundle changed or disagrees')
    zero = read_json(c.artifacts / 'resources/ZERO_STEP_NATIVE_PARITY.json')
    if zero['status'] != 'PASS' or not all(zero['native_NOAUG_cache_equal'].values()):
        raise ValueError('native zero-step check did not bind to the cache')
    for row in zero['rows']:
        expected_equal = {k: not (row['module'] == 'Q_A0' and k == 'pred_scores') for k in row['same_as_B0']}
        if row['same_as_B0'] != expected_equal:
            raise ValueError('native zero-step slot invariant differs')
    setup = read_json(c.artifacts / 'resources/PROFILE_SETUP_REPLAY.json')
    if ({r['input_id'] for r in setup['rows']} != {e['input_id'] for e in c.eval_index['entries']}
            or not all(all(r['equal'].values()) for r in setup['rows'])):
        raise ValueError('native setup replay is not exact and complete')
    result = {'status': 'PASS', 'forwards': len(rows), 'timed': 72, 'warmup': 24,
              'reloads': len(summary['reload_checks']), 'setup_inputs': len(setup['rows']),
              'profile_sha256': file_digest(c.artifacts / 'resources/PROFILE.csv'),
              'runtime_limitations': 'resources/NATIVE_RUNTIME_NOTE.md'}
    write_json(c.artifacts / 'resources/PROFILE_VERIFICATION.json', result)
    return result


def portable_checkpoint(raw, destination, *, mode, label_identity):
    raw, destination = Path(raw), Path(destination)
    saved = torch.load(raw, map_location='cpu', weights_only=False)
    metadata = {**saved['metadata'], 'source_module': saved['metadata']['module'], 'module': mode,
                'label_identity_sha256': label_identity, 'geometry_policy': 'frozen-v1',
                'source_checkpoint_sha256': file_digest(raw),
                'input_schema': 'unchanged quality_inputs / shared MaskHead; prediction-only',
                'single_module_only': True}
    portable = {'head': saved['head'], 'metadata': metadata,
                'slot': 'score_only' if mode.startswith('Q') else 'mask_only',
                'inference_api': 'scripts.qp_mn_adapter.apply_module'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = torch.load(destination, map_location='cpu', weights_only=False)
        if existing['metadata'] != metadata or set(existing) != set(portable) or any(
                not torch.equal(value, existing['head'][key]) for key, value in saved['head'].items()):
            raise ValueError(f'portable checkpoint changed: {destination}')
    else:
        torch.save(portable, destination)
    if destination.stat().st_size > 20 * 1024**2:
        raise ValueError('head exceeds small Git asset limit')
    return {'path': str(destination), 'bytes': destination.stat().st_size, 'sha256': file_digest(destination),
            'source_checkpoint_sha256': file_digest(raw)}


def package_heads(context):
    c, rows = context, []
    for manifest_path in sorted((c.artifacts / 'training').glob('*/seed*/checkpoint_manifest.json')):
        manifest = read_json(manifest_path)
        arm, seed = manifest_path.parent.parent.name, int(manifest_path.parent.name.removeprefix('seed'))
        for point in manifest['checkpoints']:
            load_head(c, arm, point['step'], seed, 'cpu')
            destination = manifest_path.parent / 'checkpoints' / f"update={point['step']:04d}.pt"
            rows.append({'module': arm, 'seed': seed, 'step': point['step'],
                         **portable_checkpoint(point['path'], destination, mode=arm,
                                               label_identity=c.train_index['label_identity_sha256'])})
        log_path = manifest_path.parent / 'metrics.jsonl'
        if log_path.exists():
            write_csv(manifest_path.parent / 'metrics.csv', [json.loads(line) for line in log_path.read_text().splitlines() if line])
    write_json(c.artifacts / 'training/PORTABLE_CHECKPOINTS.json', {'status': 'COMPLETE', 'rows': rows})
    return rows


def package_metric_evidence(context):
    c = context
    verified = read_json(c.artifacts / 'evaluation/VERIFIED_METRIC_STATES.json')
    files = {row['path']: row['sha256'] for row in verified['rows']}
    diagnostics = read_json(c.artifacts / 'diagnostics/DIAGNOSTICS.json')
    assisted = []
    for row in diagnostics['assisted_official']:
        path = Path(row['evidence_path'])
        if file_digest(path) != row['evidence_sha256']:
            raise ValueError('diagnostic metric evidence changed')
        state = read_json(path)
        actual = recompute_official_metric_evidence(state)['raw_local_AP' if row['H'] == 1 else 'online_t-mAP']
        if abs(actual - row['official_AP']) > 1e-8 or state['updates'] != row['full_input_count']:
            raise ValueError('diagnostic AP/count recomputation mismatch')
        assisted.append({**row, 'recomputed': actual})
        files[str(path)] = row['evidence_sha256']
    write_json(c.artifacts / 'diagnostics/VERIFIED_ASSISTED_METRICS.json', {'status': 'PASS', 'rows': assisted})
    destination = c.artifacts / 'evidence/official_metric_states.zip'
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for number, (path, digest) in enumerate(sorted(files.items())):
            path = Path(path)
            if file_digest(path) != digest:
                raise ValueError('metric file changed after verification')
            state = read_json(path)
            # Official export is sufficient statistics, never point masks/GT.
            if 'schema_version' not in state and 'schema' not in state:
                raise ValueError('unrecognized official metric evidence')
            name = f'{number:03d}-{path.name}'
            info = ZipInfo(name, date_time=(2026, 9, 29, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
            entries.append({'archive_path': name, 'source_path': str(path), 'sha256': digest, 'bytes': path.stat().st_size})
    if destination.stat().st_size > 20 * 1024**2:
        raise ValueError('metric package requires the explicit larger-asset publication path')
    result = {'path': str(destination), 'sha256': file_digest(destination), 'bytes': destination.stat().st_size,
              'main_metric_states': len(verified['rows']), 'diagnostic_metric_states': len(assisted), 'entries': entries}
    write_json(c.artifacts / 'evidence/OFFICIAL_METRIC_PACKAGE.json', result)
    return result
