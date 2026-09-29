"""Bind repaired assets and explicit output roots before any targeted experiment."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from scripts.qp_mn_adapter import validate_output_paths
from scripts.short_module_identity import (
    assert_sources_current,
    file_digest,
    identity_digest,
)
from scripts.short_module_screen import append_event, read_json, write_json

PROJECT = Path(__file__).resolve().parents[1]


def ledger_total(paths, *, subtotal_ids):
    events, sources = {}, []
    for path in map(Path, paths):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        sources.append({'path': str(path), 'sha256': file_digest(path), 'rows': len(rows)})
        for row in rows:
            key = row['event_id']
            if key in subtotal_ids:
                continue
            if key in events and events[key]['gpu_hours'] != row['gpu_hours']:
                raise ValueError(f'conflicting ledger event: {key}')
            events[key] = row
    return {'gpu_hours': sum(r['gpu_hours'] for r in events.values()),
            'unique_events': len(events), 'sources': sources,
            'excluded_subtotals': sorted(subtotal_ids), 'event_ids': sorted(events)}


def verify_index_files(index, *, fields=('prediction', 'targets')):
    if index['status'] != 'COMPLETE':
        raise ValueError('incomplete input export')
    cache = Path(index['cache'])
    seen = set()
    for entry in index['entries']:
        if entry['input_id'] in seen:
            raise ValueError('duplicate input identity')
        seen.add(entry['input_id'])
        for field in fields:
            path = cache / entry[field]['file']
            if file_digest(path) != entry[field]['sha256']:
                raise ValueError(f'changed input content: {entry["input_id"]} {field}')
    return len(seen)


@dataclass
class Context:
    config: dict
    root: Path
    artifacts: Path
    repaired: Path
    historical: Path
    train_index: dict
    eval_index: dict
    resolved: dict
    assets: dict

    @classmethod
    def load(cls, config_path, *, root=None, artifact_root=None):
        cfg = yaml.safe_load(Path(config_path).read_text())
        runtime = Path(root or cfg['runtime_root']).resolve()
        public = Path(artifact_root or cfg['artifact_root'])
        public = (PROJECT / public).resolve() if not public.is_absolute() else public.resolve()
        repaired, historical = Path(cfg['repaired_root']).resolve(), Path(cfg['historical_root']).resolve()
        validate_output_paths(runtime, public, [repaired, historical,
                              PROJECT / 'artifacts/rescene_code_first_audit_v1',
                              PROJECT / 'artifacts/short_module_screen_v1'])
        return cls(cfg, runtime, public, repaired, historical,
                   read_json(repaired / 'EXPORT_INDEX.json'), read_json(repaired / 'noaug/EXPORT_INDEX.json'),
                   read_json(PROJECT / 'artifacts/rescene_code_first_audit_v1/RUN_CONFIG.json'),
                   read_json(repaired / 'assets.local.json'))

    @property
    def dimensions(self):
        return self.resolved['parent_dimensions']

    @property
    def population(self):
        return [entry['record'] for entry in self.train_index['entries']]


def bind(context: Context, *, base_commit):
    c = context
    base = subprocess.check_output(['git', 'rev-parse', f'{base_commit}^{{commit}}'], cwd=PROJECT, text=True).strip()
    if base != c.config['base_commit']:
        raise ValueError('requested base is not the repaired contract')
    subprocess.run(['git', 'merge-base', '--is-ancestor', base, 'HEAD'], cwd=PROJECT, check=True)
    assets = {}
    for key, expected in [('r1_checkpoint', c.config['parent_sha256']),
                          ('concerto_pretrained', c.config['concerto_sha256'])]:
        path = Path(c.assets[key])
        actual = file_digest(path)
        if actual != expected:
            raise ValueError(f'parent asset mismatch: {key}')
        assets[key] = {'path': str(path), 'sha256': actual, 'bytes': path.stat().st_size}
    if assets['r1_checkpoint']['bytes'] != c.config['parent_bytes']:
        raise ValueError('R1 byte size mismatch')
    if c.train_index['label_identity']['geometry_policy'] != 'frozen-v1':
        raise ValueError('M assignment must remain frozen-v1')
    assert_sources_current(c.train_index['label_identity'])
    if identity_digest(c.train_index['label_identity']) != c.train_index['label_identity_sha256']:
        raise ValueError('label identity payload mismatch')
    noaug_identity = read_json(Path(c.eval_index['cache']) / 'IDENTITY.json')
    assert_sources_current(noaug_identity)
    if c.eval_index.get('condition') != 'no-training-augmentation':
        raise ValueError('main evaluation must use NOAUG_NATIVE')
    counts = {}
    for role in ('TRAIN', 'CAL', 'SEL'):
        records = [r for r in c.population if r['role'] == role]
        counts[role] = {'references': len({r['reference_id'] for r in records}),
                        **{f'T{h}': sum(r['horizon'] == h for r in records) for h in (1, 2)}}
    if counts != {'TRAIN': {'references': 64, 'T1': 199, 'T2': 188},
                  'CAL': {'references': 4, 'T1': 23, 'T2': 23},
                  'SEL': {'references': 4, 'T1': 24, 'T2': 24}}:
        raise ValueError(f'population contract differs: {counts}')
    if {e['input_id'] for e in c.eval_index['entries']} != {r['input_id'] for r in c.population if r['role'] != 'TRAIN'}:
        raise ValueError('NOAUG evaluation population differs')
    for index in (c.train_index, c.eval_index):
        verify_index_files(index)
        for e in index['entries']:
            r, audit = e['record'], e['audit']
            if len(r['scan_ids']) != r['horizon'] or len(audit['actual_loaded_paths']) != r['horizon']:
                raise ValueError('native scan count differs from observed horizon')
            if not audit['parent_eval'] or not audit['parent_frozen'] or audit['dimensions'] != c.dimensions:
                raise ValueError('native parent freeze or dimensions differ')
    first = next(e for e in c.train_index['entries'] if e['record']['role'] == 'TRAIN')
    payload = torch.load(Path(c.train_index['cache']) / first['prediction']['file'], map_location='cpu', weights_only=False)
    soft = payload['parent']['soft']
    actual_dims = {'feature_dim': soft['segment_features'].shape[1], 'query_dim': soft['query_features'].shape[1],
                   'probability_dim': soft['class_probabilities'].shape[1], 'classes': payload['descriptor']['classes']}
    if actual_dims != c.dimensions:
        raise ValueError('actual parent tensor dimensions differ')
    prior = ledger_total([PROJECT / 'artifacts/perception_gain_v2/budget/LEDGER.jsonl',
                          PROJECT / 'artifacts/short_module_screen_v1/BUDGET_LEDGER.jsonl',
                          PROJECT / 'artifacts/rescene_code_first_audit_v1/BUDGET_LEDGER.jsonl'],
                         subtotal_ids={'inherited-v2-final'})
    budget = {**prior, 'new_cap_gpu_hours': min(c.config['budget']['new_gpu_hours'],
              c.config['budget']['cumulative_gpu_hours'] - prior['gpu_hours']),
              'reserve_gpu_hours': c.config['budget']['reserve_gpu_hours']}
    if budget['new_cap_gpu_hours'] < budget['reserve_gpu_hours']:
        raise ValueError('insufficient budget to reserve completion')
    reports = {name: file_digest(PROJECT / 'artifacts/rescene_code_first_audit_v1' / name)
               for name in ('CONCLUSION.md', 'REQUIREMENT_REVIEW.md', 'RELABEL_AUDIT.json',
                            'NOAUG_INPUT_PARITY.json', 'TEST_NOTES.md')}
    binding = {'base_commit': base, 'reports': reports, 'assets': assets, 'counts': counts,
               'actual_dimensions': actual_dims, 'training_index_sha256': identity_digest(c.train_index),
               'evaluation_index_sha256': identity_digest(c.eval_index),
               'known_test_exclusion': 'Existing repo_reference symlink serialization; no model/label/metric effect',
               'previous_exposure': 'R1/development and historical SEL exposure; not untouched test data'}
    contract = {'quality_target': 'quality.temporal', 'quality_weight': 'quality.valid (binary)',
                'quality_schema': 'quality-threshold-validity-v2', 'shape_usable': 'quality.geometry_valid',
                'geometry_policy': 'frozen-v1', 'assignment': 'assignment entity id >=0',
                'shape_targets': 'segment_targets', 'shape_weights': 'segment_weights',
                'label_identity_sha256': c.train_index['label_identity_sha256'],
                'train_prediction_identity': c.train_index['cache_identity_sha256'],
                'evaluation_condition': 'NOAUG_NATIVE',
                'evaluation_prediction_identity': c.eval_index['cache_identity_sha256'],
                'dimensions': actual_dims, 'thresholds': c.resolved['official_thresholds']}
    refs = sorted({r['reference_id'] for r in c.population if r['role'] == 'TRAIN'},
                  key=lambda x: (hashlib.sha256(x.encode()).hexdigest(), x))[:8]
    panel = []
    for ref in refs:
        rows = [r for r in c.population if r['role'] == 'TRAIN' and r['reference_id'] == ref]
        pairs = sorted([r for r in rows if r['horizon'] == 2],
                       key=lambda r: (hashlib.sha256(r['input_id'].encode()).hexdigest(), r['input_id']))[:2]
        scans = {s for r in pairs for s in r['scan_ids']}
        panel.extend(pairs + [r for r in rows if r['horizon'] == 1 and r['scan_ids'][0] in scans])
    for name, value in [('BASE_BINDING.json', binding), ('REPAIRED_CONTRACT.json', contract),
                        ('RESOLVED_CONFIG.json', {**c.config, 'artifact_root': str(c.artifacts),
                                                 'runtime_root': str(c.root), 'dimensions': actual_dims}),
                        ('BUDGET_CONTRACT.json', budget),
                        ('INPUT_MANIFEST.json', {'counts': counts, 'training': c.train_index, 'evaluation': c.eval_index}),
                        ('diagnostics/TRAIN_PANEL.json', {'rule': 'SHA256(reference_id), then SHA256(input_id); no outcome selection',
                                                          'references': refs, 'records': panel})]:
        path = c.artifacts / name
        if path.exists() and read_json(path) != value:
            raise ValueError(f'bound contract changed: {name}')
        write_json(path, value)
    c.root.mkdir(parents=True, exist_ok=True)
    write_json(c.root / 'BINDING.json', binding)
    return binding


@contextmanager
def gpu_scope(context, *, device, stage):
    """Charge complete device reservation, including failed attempts, once per stage."""
    c = context
    count = int(str(device).startswith('cuda'))
    contract = read_json(c.artifacts / 'BUDGET_CONTRACT.json')
    ledger = c.artifacts / 'BUDGET_LEDGER.jsonl'
    used = ledger_total([ledger], subtotal_ids=set())['gpu_hours'] if ledger.exists() else 0.
    if count and contract['new_cap_gpu_hours'] - used <= contract['reserve_gpu_hours']:
        raise ValueError('completion reserve reached; no further exploration permitted')
    if count:
        gpu = str(device).split(':')[-1]
        memory = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                                          '--format=csv,noheader,nounits'], text=True).strip()
        if int(memory) > 0:
            raise ValueError(f'GPU {gpu} is not idle ({memory} MiB); do not evict other jobs')
    utc = datetime.now(timezone.utc).isoformat()
    event = {'event_id': f'{stage}:{utc}', 'stage': stage, 'device': str(device), 'gpu_count': count,
             'pid': os.getpid(), 'start_utc': utc, 'status': 'RUNNING',
             'measurement': 'whole stage process GPU reservation walltime including failures'}
    running = c.root / 'reservations' / f'{stage}-{os.getpid()}.json'
    write_json(running, event)
    started = time.perf_counter()
    try:
        yield
        event['status'] = 'COMPLETE'
    except BaseException:
        event['status'] = 'FAILED'
        raise
    finally:
        event.update(elapsed_seconds=time.perf_counter() - started)
        event['gpu_hours'] = event['elapsed_seconds'] * count / 3600
        event['end_utc'] = datetime.now(timezone.utc).isoformat()
        append_event(ledger, event)
        write_json(running, event)


def reuse_q2(context):
    """Import the completed corrected bridge, never resume its optimizer under a new formula."""
    from models.qp_mn_heads import build_head
    from scripts.short_module_training import build_sample_plan, parameter_digest

    c = context
    source = read_json(PROJECT / 'artifacts/rescene_code_first_audit_v1/training/Q2/checkpoint_manifest.json')
    counts = {e['input_id']: e['audit']['candidate_count'] for e in c.train_index['entries']}
    plan = build_sample_plan(c.population, counts, seed=45)
    head = build_head('Q2_F', **c.dimensions, seed=45)
    initial = parameter_digest(head)
    checkpoints = []
    if {r['step'] for r in source['checkpoints']} != {0, 500, 1000, 1500}:
        raise ValueError('Q2 bridge lacks all required states')
    for row in source['checkpoints']:
        path = Path(row['path'])
        if file_digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Q2 checkpoint content differs from repaired manifest')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        meta = saved['metadata']
        expected = {'module': 'Q2', 'seed': 45, 'optimizer_updates': row['step'],
                    'base_sha256': c.config['parent_sha256'], 'dimensions': c.dimensions,
                    'cache_identity_sha256': c.train_index['cache_identity_sha256'],
                    'label_identity_sha256': c.train_index['label_identity_sha256'],
                    'initial_parameter_sha256': initial, 'sample_plan_sha256': identity_digest(plan),
                    'schedule_horizon': 1500}
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError('Q2 bridge contract differs')
        for name, digest in meta['training_sources'].items():
            if file_digest(name) != digest:
                raise ValueError(f'Q2 source changed: {name}')
        for group in saved['optimizer']['param_groups']:
            if group['betas'] != (.9, .999) or group['eps'] != 1e-8 or group['weight_decay'] != .0001:
                raise ValueError('Q2 optimizer contract differs')
        head.load_state_dict(saved['head'], strict=True)
        if row['step'] == 0 and parameter_digest(head) != initial:
            raise ValueError('Q2 initial weights differ')
        if row['step'] and any(int(s['step']) != row['step'] for s in saved['optimizer']['state'].values()):
            raise ValueError('Q2 optimizer state step differs')
        destination = c.root / 'training/Q2_F/seed45' / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_digest(destination) != row['sha256']:
            raise ValueError('existing imported Q2 checkpoint differs')
        if not destination.exists():
            shutil.copyfile(path, destination)
        checkpoints.append({**row, 'source_path': str(path), 'path': str(destination)})
    manifest = {**source, 'module': 'Q2_F', 'source_module': 'Q2', 'source_run': str(c.repaired),
                'updates_reused': 1500, 'updates_new': 0, 'checkpoints': checkpoints,
                'optimizer_resume_allowed': False, 'status': 'COMPLETE',
                'equivalence_probe': 'tests/test_qp_mn_real_contract.py::test_real_q2_bridge_loss_and_gradient_equivalence'}
    write_json(c.artifacts / 'training/Q2_F/seed45/checkpoint_manifest.json', manifest)
    write_json(c.artifacts / 'training/Q2_F/seed45/resolved_config.json', {
        'formula': 'corrected Q2 sigmoid weighted MSE; immutable completed bridge',
        'source_manifest_sha256': file_digest(PROJECT / 'artifacts/rescene_code_first_audit_v1/training/Q2/checkpoint_manifest.json'),
        'sample_plan_sha256': identity_digest(plan), 'training': c.config['training']})
    return manifest
