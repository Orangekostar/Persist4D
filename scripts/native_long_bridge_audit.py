"""Audit trained memory comparisons, identity events and physical-reference uncertainty."""

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_confirmation_audit import recompute
from scripts.native_long_memory_bridge import METHODS, digest, pool_identity
from scripts.native_long_postselection import read_json, sha256

RATES = {'normalized_id_switch_rate': ('deployment_id_switches', 'identity_transition_opportunities'),
         'fragmentation_rate': ('fragmentation_count', 'fragmentation_opportunities'),
         'merge_rate': ('merge_count', 'merge_opportunities'),
         'gap_recovery_accuracy': ('correct_recoveries', 'recovery_attempts'),
         'gap_recovery_recall': ('correct_recoveries', 'gap_opportunities')}


def paired_count_bootstrap(candidate, control, numerator, denominator, *, replicates=10000):
    import numpy as np

    def index(rows):
        result = {r['reference_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate physical reference')
        return result

    a, b = index(candidate), index(control)
    if a.keys() != b.keys() or len(a) < 2:
        raise ValueError('paired physical reference coverage differs')
    refs = sorted(a)
    counts = np.array([[a[r][numerator], a[r][denominator], b[r][numerator], b[r][denominator]]
                       for r in refs], dtype=np.int64)
    if (counts < 0).any() or (counts[:, 0] > counts[:, 1]).any() or (counts[:, 2] > counts[:, 3]).any():
        raise ValueError('identity counts have invalid denominators')
    total = counts.sum(0)
    point = float(total[0] / total[1] - total[2] / total[3]) if total[1] and total[3] else None
    draws = np.random.default_rng(45).integers(0, len(refs), (replicates, len(refs)))
    sampled = counts[draws].sum(1)
    valid = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
    rows = sampled[valid]
    delta = rows[:, 0] / rows[:, 1] - rows[:, 2] / rows[:, 3]
    return {'point_delta': point, 'ci95': np.quantile(delta, [.025, .975]).tolist() if len(delta) else None,
            'reference_ids': refs, 'clusters': len(refs), 'replicates': replicates,
            'valid_replicates': int(valid.sum()), 'seed': 45,
            'estimand': 'difference of pooled event-count rates; paired resampling of physical references'}


def restore_identity_updates(rows):
    from scripts.system_comparison_metrics import IdentityAssignmentUpdate

    return [IdentityAssignmentUpdate(horizon=r['horizon'], visible_gt_ids=tuple(r['visible_gt_ids']),
            assignments={int(k): v for k, v in r['assignments'].items()}) for r in rows]


def verify_identity_rows(actual, expected):
    if any(actual[k] != value for k, value in expected.items()):
        raise ValueError('identity aggregate differs from saved assignment events')


def verify_state(evidence, updates, mode):
    core = {k: v for k, v in evidence.items() if k != 'sha256'}
    hashed = hashlib.sha256((json.dumps(core, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    if (hashed != evidence['sha256'] or evidence['schema_version'] != 1
            or evidence['updates'] != updates or evidence['mode'] != mode):
        raise ValueError('official state digest, mode or denominator differs')


def run(directory):
    from scripts.system_comparison_metrics import compute_deployment_identity_metrics

    project = Path(__file__).resolve().parents[1]
    plan_path = directory / 'BRIDGE_PLAN.json'
    plan = read_json(plan_path)
    if sha256(directory / 'FORMAL_POPULATION.json') != plan['formal_population_sha256']:
        raise ValueError('bridge formal population changed')
    if sha256(directory / 'FORMAL_LOCK.json') != plan['formal_lock_sha256']:
        raise ValueError('bridge checkpoint selection changed')
    if sha256(directory / 'BRIDGE_CPU_ALIGNMENT.json') != plan['cpu_alignment_sha256']:
        raise ValueError('bridge preflight changed')
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('bridge measurement source changed: ' + name)
    manifest = read_json(directory / 'FORMAL_POPULATION.json')
    units = sorted((r for r in manifest['records']['PB'] if r['horizon'] == 5), key=lambda r: r['source_unit'])
    if len(units) != 129 or [u['input_id'] for u in units] != plan['unit_input_ids']:
        raise ValueError('bridge fixed PB units differ')
    ref_counts = collections.Counter(u['reference_id'] for u in units)
    if len(ref_counts) != 6:
        raise ValueError('bridge physical reference population differs')
    dataset_spec = Path(manifest['source_data_root']) / 'processed/rio/rio.yaml'
    taxonomy = next(r for r in manifest['data_files'] if r['relative_path'] == 'processed/rio/rio.yaml')
    if sha256(dataset_spec) != taxonomy['sha256']:
        raise ValueError('bridge evaluator taxonomy changed')
    preflight = read_json(directory / 'BRIDGE_CPU_ALIGNMENT.json')
    drift_expected = {(r['unit'], r['horizon'], d['id']) for r in preflight['rows'] for d in r['window_label_drift']}
    result = {'status': 'INCOMPLETE', 'plan_sha256': sha256(plan_path), 'rows': [], 'pending': [],
              'source_receipts': [], 'official_states_verified': 0, 'pooled_states_recomputed': 0,
              'runtime_status': 'RUNTIME_CONDITIONAL', 'independent_training_seeds': 1,
              'physical_reference_clusters': 6, 'statistical_scope': 'identity-rate bootstrap only; no pooled AP CI inferred from reference AP'}
    for selected in plan['rows']:
        name = f"{selected['arm']}-{selected['step']:05d}"
        folder = directory / 'collected/E3/memory-bridge' / name
        path = folder / 'RESULT.json'
        if not path.exists() or read_json(path)['status'] != 'COMPLETE':
            result['pending'].append(name)
            continue
        row = read_json(path)
        if (row['completed_units'] != 129 or row['expected_units'] != 129 or row['forward_count'] != 1032
                or row['shared_prefixes'] != 258 or len(row['unit_evidence']) != 129
                or (row['arm'], row['step']) != (selected['arm'], selected['step'])
                or row['identity']['bridge_plan_sha256'] != sha256(plan_path)
                or row['identity']['bundle_sha256'] != selected['bundle_sha256']
                or row['identity']['checkpoint_sha256'] != selected['checkpoint_sha256']):
            raise ValueError('bridge completion or weight/plan identity differs: ' + name)
        result['source_receipts'].append({'path': str(path), 'sha256': sha256(path)})
        identities, times, capacity, drifts = [], [], [], set()
        for index, (unit, entry) in enumerate(zip(units, row['unit_evidence'])):
            filename = f'{index:03d}.json'
            unit_path = folder / 'units' / filename
            if Path(entry['path']).name != filename or sha256(unit_path) != entry['sha256']:
                raise ValueError('bridge unit evidence order/SHA differs')
            saved = read_json(unit_path)
            if (saved['unit'] != unit['source_unit'] or saved['reference_id'] != unit['reference_id']
                    or not saved['shared_T1_T2_exact'] or not saved['local_candidates_shared_B2_B4']
                    or saved['evaluation_truth'] != 'official_causal_full_prefix_first_observed_point'):
                raise ValueError('bridge unit population or shared-candidate contract differs')
            result['source_receipts'].append({'path': str(unit_path), 'sha256': entry['sha256']})
            forwards = [t for t in saved['timings'] if 'network_seconds' in t]
            if [(t['horizon'], t['context']) for t in forwards] != [(h, mode) for h in range(1, 6)
                    for mode in (('full',) if h <= 2 else ('full', 'local'))]:
                raise ValueError('bridge forward execution order differs')
            forward_map = {(t['horizon'], t['context']): t for t in forwards}
            for t in forwards:
                h = t['horizon']
                scans = unit['physical_scan_ids'][:h]
                if t['context'] == 'local':
                    scans = scans[-2:]
                if t['consumed_scans'] != scans or t['seed_key'] != digest(unit['physical_scan_ids'][:h]):
                    raise ValueError('bridge forward used a different/future context or seed')
                if any(not math.isfinite(t[k]) or t[k] < 0 for k in
                       ('network_seconds', 'io_collate_H2D_seconds', 'postprocess_seconds')):
                    raise ValueError('invalid bridge timing')
            for h in (3, 4, 5):
                if forward_map[h, 'full']['current_points_sha256'] != forward_map[h, 'local']['current_points_sha256']:
                    raise ValueError('bridge current point/annotation alignment differs')
            local_rows = {(r['method'], r['horizon']): r for r in saved['identity_rows']}
            if len(local_rows) != 12 or len(saved['identity_rows']) != 12:
                raise ValueError('bridge identity horizon coverage differs')
            for method in METHODS:
                updates = restore_identity_updates(saved['identity_updates'][method])
                if [x.horizon for x in updates] != list(range(1, 6)):
                    raise ValueError('bridge saved identity events omit visits')
                for h in range(2, 6):
                    expected = compute_deployment_identity_metrics(updates[:h])
                    actual = local_rows[method, h]
                    if actual['unit'] != unit['source_unit'] or actual['reference_id'] != unit['reference_id']:
                        raise ValueError('bridge identity event scope differs')
                    verify_identity_rows(actual, expected)
                    identities.append(actual)
                for h in range(1, 6):
                    context = 'full' if method == 'FH-B2' or h <= 2 else 'local'
                    forward = forward_map[h, context]
                    tracker = next(t for t in saved['timings'] if t.get('method') == method and t['horizon'] == h)
                    times.append({'method': method, 'horizon': h, 'reference_id': unit['reference_id'],
                        'network_seconds': forward['network_seconds'], 'tracker_seconds': tracker['tracker_seconds'],
                        'network_plus_tracker_seconds': forward['network_seconds'] + tracker['tracker_seconds'],
                        'io_collate_H2D_seconds': forward['io_collate_H2D_seconds'],
                        'postprocess_seconds': forward['postprocess_seconds'],
                        'cuda_peak_allocated_bytes': forward['cuda_peak_allocated_bytes']})
            if [r['horizon'] for r in saved['capacities']] != list(range(1, 6)):
                raise ValueError('bridge capacity stages differ')
            for c in saved['capacities']:
                if not 0 <= c['active'] <= c['occupied'] <= 100 or c['state_bytes'] <= 0:
                    raise ValueError('bridge state capacity is invalid')
                capacity.append(c)
            drifts.update((unit['source_unit'], d['horizon'], d['id']) for d in saved['window_label_drift'])
        if drifts != drift_expected:
            raise ValueError('bridge annotation differences do not match full CPU audit')
        metrics = {(r['method'], r['horizon']): r for r in row['metrics']}
        refs = {(r['method'], r['horizon'], r['reference_id']): r for r in row['by_reference']}
        raw = {(r['context'], r['horizon']): r for r in row['raw_current_metrics']}
        if (len(metrics) != len(row['metrics']) or len(refs) != len(row['by_reference'])
                or len(raw) != len(row['raw_current_metrics'])
                or len(metrics) != 12 or len(refs) != 72 or len(raw) != 8):
            raise ValueError('bridge pooled/reference metric coverage differs')
        evidence = {Path(e['path']).name: e for e in row['official_evidence']}
        expected_states = {}
        for method in METHODS:
            for h in range(2, 6):
                actual = metrics[method, h]
                if actual['units'] != 129:
                    raise ValueError('bridge pooled metric denominator differs')
                verify_identity_rows(actual, pool_identity([r for r in identities if (r['method'], r['horizon']) == (method, h)]))
                expected_states[f'{method}-T{h}.json'] = ('strict_online', 129, actual, True)
                for ref, count in ref_counts.items():
                    value = refs[method, h, ref]
                    if value['units'] != count:
                        raise ValueError('bridge reference denominator differs')
                    verify_identity_rows(value, pool_identity([r for r in identities
                        if (r['method'], r['horizon'], r['reference_id']) == (method, h, ref)]))
                    expected_states[f'{method}-T{h}-{ref}.json'] = ('strict_online', count, value, False)
        for context in ('full', 'local'):
            for h in range(2, 6):
                expected_states[f'raw-{context}-T{h}.json'] = ('raw_local', 129, raw[context, h], True)
        if evidence.keys() != expected_states.keys() or len(evidence) != len(row['official_evidence']):
            raise ValueError('bridge official sufficient-state coverage differs')
        for filename, (mode, count, value, calculate) in expected_states.items():
            state_path = folder / 'official' / filename
            if sha256(state_path) != evidence[filename]['sha256']:
                raise ValueError('bridge official state bytes differ')
            state = read_json(state_path)
            verify_state(state, count, mode)
            result['official_states_verified'] += 1
            if calculate:
                computed = recompute(state, dataset_spec)
                if any(abs(computed[k] - value[k]) > 1e-12 for k in computed):
                    raise ValueError('bridge independently recomputed official metric differs')
                result['pooled_states_recomputed'] += 1
            result['source_receipts'].append({'path': str(state_path), 'sha256': evidence[filename]['sha256']})
        profile = []
        for method in METHODS:
            for h in range(1, 6):
                samples = [t for t in times if (t['method'], t['horizon']) == (method, h)]
                summary = {'method': method, 'horizon': h, 'inputs': len(samples)}
                for key in ('network_seconds', 'tracker_seconds', 'network_plus_tracker_seconds',
                            'io_collate_H2D_seconds', 'postprocess_seconds', 'cuda_peak_allocated_bytes'):
                    summary[key] = {'mean': statistics.mean(s[key] for s in samples),
                                    'median': statistics.median(s[key] for s in samples),
                                    'max': max(s[key] for s in samples)}
                profile.append(summary)
        comparisons = []
        for h in range(2, 6):
            a = [refs['LOCAL-B4', h, ref] for ref in ref_counts]
            b = [refs['LOCAL-B2', h, ref] for ref in ref_counts]
            comparisons.append({'horizon': h, 'candidate': 'LOCAL-B4', 'control': 'LOCAL-B2',
                'metric_delta': {k: metrics['LOCAL-B4', h][k] - metrics['LOCAL-B2', h][k]
                                 for k in metrics['LOCAL-B4', h] if k.startswith('online_')},
                'identity_rate_deltas': {rate: paired_count_bootstrap(a, b, num, den)
                                        for rate, (num, den) in RATES.items()}})
        result['rows'].append({'arm': selected['arm'], 'step': selected['step'], 'metrics': row['metrics'],
            'by_reference': row['by_reference'], 'raw_current_metrics': row['raw_current_metrics'],
            'local_memory_vs_strong_pair': comparisons, 'profile': profile,
            'window_annotation_differences': len(drifts),
            'capacity': {'max_occupied': max(c['occupied'] for c in capacity),
                         'max_active': max(c['active'] for c in capacity),
                         'state_bytes': sorted({c['state_bytes'] for c in capacity}),
                         'rejected_births': sum(c['rejected_births'] for c in capacity)}})
    if not result['pending']:
        state = read_json(directory / 'collected/E3/BRIDGE_STATE.json')
        if (state['status'] != 'COMPLETE' or state['plan_sha256'] != sha256(plan_path)
                or state['completed_jobs'] != [f"{r['arm']}-{r['step']:05d}" for r in plan['rows']]):
            raise ValueError('bridge queue completion differs')
        result['status'] = 'COMPLETE'
    write_json_atomic(directory / 'BRIDGE_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    result = run(parser.parse_args().directory)
    print({k: result[k] for k in ('status', 'pending', 'official_states_verified', 'pooled_states_recomputed')})
