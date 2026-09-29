"""NOAUG native official evaluation with an all-arm CAL lock and fixed SEL endpoints."""
import math
import time
from pathlib import Path

import torch

from models.qp_mn_heads import Module, build_head
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.qp_mn_adapter import apply_module
from scripts.qp_mn_binding import PROJECT, verify_index_files
from scripts.short_module_eval_identity import evaluation_identity
from scripts.short_module_evaluation import materialization_system
from scripts.short_module_identity import (
    assert_sources_current,
    file_digest,
    identity_digest,
    source_identity,
)
from scripts.short_module_native import unpack_prediction
from scripts.short_module_screen import read_json, write_json
from scripts.system_comparison_inference import unpack_bool_matrix

ARMS = ('Q2_F', 'Q_A0', 'Q_P', 'M_C', 'M_N')
TOLERANCE = 1e-6


def build_cal_lock(rows):
    selected = {}
    for arm in ARMS:
        points = [r for r in rows if r['module'] == arm and r['seed'] == 45
                  and r['step'] in (500, 1000, 1500) and r['role'] == 'CAL']
        if len(points) != 3 or {r['step'] for r in points} != {500, 1000, 1500} or any(
            r['status'] != 'COMPLETE' or any(r['metrics'][k] is None or not math.isfinite(r['metrics'][k])
                                           for k in ('T1', 'T2')) for r in points):
            raise ValueError(f'{arm} lacks complete CAL trajectory')
        best = min(points, key=lambda r: r['step'])
        for row in sorted(points, key=lambda r: r['step']):
            d2, d1 = (row['metrics'][k] - best['metrics'][k] for k in ('T2', 'T1'))
            if d2 > TOLERANCE or abs(d2) <= TOLERANCE and d1 > TOLERANCE:
                best = row
        selected[arm] = best['step']
    return {'schema': 'qp-mn-cal-lock-v1', 'seed': 45, 'selected_updates': selected,
            'condition': 'NOAUG_NATIVE', 'all_arms_locked_before_SEL': True,
            'selection': 'T2 descending, T1 descending, earlier step; tolerance1e-6'}


def validate_sel_request(lock, *, module, step, seed, replication=None):
    Module(module)
    if not lock or set(lock['selected_updates']) != set(ARMS):
        raise ValueError('SEL requires the complete all-arm CAL lock')
    if module == 'B0' and step == 0:
        return
    if seed == 45 and step in (lock['selected_updates'][module], 1500):
        return
    if seed == 46 and replication and step in replication.get(module, []):
        return
    raise ValueError('SEL point is neither locked selection, fixed endpoint nor predeclared replication')


def classify_gain(d2, d1):
    if d2 < -TOLERANCE:
        return 'NEGATIVE'
    if abs(d2) <= TOLERANCE:
        return 'NO_GAIN'
    if d1 is None or d1 < -.002 - TOLERANCE:
        return 'T2_ONLY_TRADEOFF'
    return 'TARGET_MAGNITUDE' if d2 >= .005 - TOLERANCE else 'POSITIVE_SINGLE_SEED'


def load_head(context, module, step, seed, device):
    c, module = context, Module(module)
    if module is Module.B0:
        if step != 0:
            raise ValueError('B0 has no checkpoint step')
        return None, None
    manifest = read_json(c.artifacts / 'training' / module.value / f'seed{seed}/checkpoint_manifest.json')
    row = next(r for r in manifest['checkpoints'] if r['step'] == step)
    path = Path(row['path'])
    if file_digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
        raise ValueError('checkpoint bytes differ from bound manifest')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    meta = saved['metadata']
    expected_mode = 'Q2' if module is Module.Q2_F and manifest.get('source_module') == 'Q2' else module.value
    expected = {'module': expected_mode, 'seed': seed, 'optimizer_updates': step,
                'base_sha256': c.config['parent_sha256'], 'dimensions': c.dimensions,
                'label_identity_sha256': c.train_index['label_identity_sha256']}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('checkpoint parent, labels, module, dimensions or step differ')
    if 'training_identity' in meta:
        identity = meta['training_identity']
        assert_sources_current(identity)
        if identity['input_index_sha256'] != identity_digest(c.train_index) or identity['training'] != c.config['training']:
            raise ValueError('training input or recipe identity differs')
        if identity_digest(read_json(c.root / 'training' / f'sample_plan_seed{seed}.json')) != meta['sample_plan_sha256']:
            raise ValueError('training sample plan differs')
    else:
        if module is not Module.Q2_F or manifest.get('updates_reused') != 1500:
            raise ValueError('unverified legacy checkpoint import')
        assert_sources_current({'sources': meta['training_sources']})
    head = build_head(module, **c.dimensions, seed=seed, thresholds=tuple(c.resolved['official_thresholds']))
    head.load_state_dict(saved['head'], strict=True)
    return head.to(device).eval(), path


def evaluate_point(context, *, module, step, seed=45, role='CAL', device='cpu'):
    c = context
    Module(module)
    if role not in ('CAL', 'SEL'):
        raise ValueError('main metrics only use CAL/SEL')
    if role == 'SEL':
        path = c.artifacts / 'selection/CAL_LOCK.json'
        replication_path = c.artifacts / 'selection/REPLICATION_PLAN.json'
        validate_sel_request(read_json(path) if path.exists() else None, module=module, step=step, seed=seed,
                             replication=read_json(replication_path)['steps'] if replication_path.exists() else None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    index = c.eval_index
    entries = [e for e in index['entries'] if e['record']['role'] == role]
    verify_index_files({**index, 'entries': entries})
    head, head_path = load_head(c, module, step, seed, device)
    dependencies = evaluation_identity(index=index, head_path=head_path, dataset_spec=c.assets['metric_dataset_spec'])
    dependencies['sources'].update(source_identity([Path(__file__), PROJECT / 'scripts/qp_mn_adapter.py',
                                                   PROJECT / 'models/qp_mn_heads.py']))
    dependencies['targeted_mode'] = module
    identity = identity_digest(dependencies)
    result_path = c.root / 'evaluation' / identity[:16] / module / f'seed{seed}/{role}-{step:04d}.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['evaluation_identity'] == dependencies and result['status'] == 'COMPLETE' and all(
            Path(r['path']).is_file() and file_digest(r['path']) == r['sha256'] for r in result['evidence']):
            return result
    def metric(h):
        return OfficialMetricAccumulator(mode='raw_local' if h == 1 else 'strict_online',
                                          dataset_spec=c.assets['metric_dataset_spec'], min_region_size=100)
    pooled = {h: metric(h) for h in (1, 2)}
    references = {}
    completed = {1: 0, 2: 0}
    system = materialization_system(index)
    isolation = {'mask_unchanged': True, 'score_unchanged': True, 'class_lineage_unchanged': True}
    started = time.perf_counter()
    for entry in entries:
        record = entry['record']
        saved = torch.load(Path(index['cache']) / entry['prediction']['file'], map_location='cpu', weights_only=False)
        parent = unpack_prediction(saved['parent'])
        target = torch.load(Path(index['cache']) / entry['targets']['file'], map_location='cpu', weights_only=False)['target']
        target['masks'] = unpack_bool_matrix(target['masks'])
        prediction = apply_module(module, head, parent, saved['descriptor'], system=system)
        for name in ('pred_classes', 'source_query_ids', 'source_class_ids'):
            if not torch.equal(prediction[name], getattr(parent, name)):
                raise ValueError('module changed retained class or candidate lineage')
        same_mask = torch.equal(prediction['pred_masks'], parent.pred_masks)
        same_score = torch.equal(prediction['pred_scores'], parent.pred_scores)
        if module.startswith('Q') and not same_mask or module.startswith('M') and not same_score:
            raise ValueError('single-module output isolation violated')
        isolation['mask_unchanged'] &= same_mask
        isolation['score_unchanged'] &= same_score
        h, ref = record['horizon'], record['reference_id']
        pooled[h].update(prediction, target)
        if (h, ref) not in references:
            references[h, ref] = metric(h)
        references[h, ref].update(prediction, target)
        completed[h] += 1
    expected = {h: sum(e['record']['horizon'] == h for e in entries) for h in (1, 2)}
    if completed != expected or not all(completed.values()):
        raise ValueError('official metric population incomplete')
    values = {f'T{h}': pooled[h].compute()['raw_local_AP' if h == 1 else 'online_t-mAP'] for h in (1, 2)}
    by_reference, evidence = [], []
    for (h, ref), accumulator in sorted(references.items()):
        by_reference.append({'reference_id': ref, 'H': h, 'completed': accumulator._updates,
                             'expected': sum(e['record']['horizon'] == h and e['record']['reference_id'] == ref for e in entries),
                             'AP': accumulator.compute()['raw_local_AP' if h == 1 else 'online_t-mAP']})
    for h, accumulator in pooled.items():
        path = result_path.with_name(f'{role}-{step:04d}-H{h}-metric-state.json')
        write_json(path, accumulator.export_evidence())
        evidence.append({'H': h, 'path': str(path), 'sha256': file_digest(path)})
    result = {'module': module, 'seed': seed, 'step': step, 'role': role, 'condition': 'NOAUG_NATIVE',
              'status': 'COMPLETE', 'metrics': values, 'by_reference': by_reference, 'evidence': evidence,
              'completed': completed, 'expected': expected, 'isolation': isolation,
              'head_sha256': file_digest(head_path) if head_path else None, 'evaluation_identity': dependencies,
              'elapsed_seconds': time.perf_counter() - started, 'result_path': str(result_path)}
    write_json(result_path, result)
    print(f'{module} seed{seed} step{step} {role}: {values}', flush=True)
    return result


def evaluate_cal(context, *, device='cpu'):
    c = context
    # Every trajectory must exist through 1500 before any selection is made.
    for arm in ARMS:
        load_head(c, arm, 1500, 45, 'cpu')
    baseline = evaluate_point(c, module='B0', step=0, role='CAL', device=device)
    rows = [evaluate_point(c, module=arm, step=step, device=device)
            for arm in ARMS for step in (500, 1000, 1500)]
    write_json(c.artifacts / 'evaluation/CAL_ALL.json', {'baseline': baseline, 'rows': rows})
    lock = build_cal_lock(rows)
    lock['evidence'] = [{'path': r['result_path'], 'sha256': file_digest(r['result_path'])} for r in rows]
    path = c.artifacts / 'selection/CAL_LOCK.json'
    if path.exists() and read_json(path) != lock:
        raise ValueError('CAL lock is immutable; do not reselect after SEL')
    write_json(path, lock)
    return lock


def evaluate_sel(context, *, device='cpu'):
    c = context
    lock = read_json(c.artifacts / 'selection/CAL_LOCK.json')
    baseline = evaluate_point(c, module='B0', step=0, role='SEL', device=device)
    selected, endpoint = [], []
    for arm in ARMS:
        selected.append(evaluate_point(c, module=arm, step=lock['selected_updates'][arm], role='SEL', device=device))
        endpoint.append(evaluate_point(c, module=arm, step=1500, role='SEL', device=device))
    write_json(c.artifacts / 'evaluation/SEL_LOCKED.json', {'baseline': baseline, 'rows': selected})
    write_json(c.artifacts / 'evaluation/SEL_ENDPOINT_1500.json', {'baseline': baseline, 'rows': endpoint})
    return selected
