"""Verify collected formal results and recompute pooled AP from sufficient states."""

import argparse
import base64
import collections
import hashlib
import json
import math
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, reference_deltas, sha256


def verify_state_content(evidence, updates):
    core = {k: v for k, v in evidence.items() if k != 'sha256'}
    actual = hashlib.sha256((json.dumps(core, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    if (actual != evidence['sha256'] or evidence['schema_version'] != 1
            or evidence['mode'] != 'strict_online' or evidence['updates'] != updates):
        raise ValueError('official sufficient-state digest or denominator differs')


def recompute(evidence, dataset_spec):
    import numpy as np
    import torch

    from scripts.p6a_metrics import OfficialMetricAccumulator

    accumulator = OfficialMetricAccumulator(mode=evidence['mode'], dataset_spec=dataset_spec)
    head = accumulator._metric.heads[0]
    if len(evidence['states']) != len(head.metric_state) or {s['name'] for s in evidence['states']} != set(head.metric_state):
        raise ValueError('official sufficient-state names differ from evaluator')
    for state in evidence['states']:
        array = np.frombuffer(base64.b64decode(state['data'], validate=True),
                              dtype=np.dtype(state['dtype'].removeprefix('torch.'))).copy().reshape(state['shape'])
        tensor = torch.from_numpy(array)
        if state['kind'] not in ('list', 'tensor'):
            raise ValueError('unknown official state kind')
        setattr(head, state['name'], [tensor] if state['kind'] == 'list' else tensor)
    accumulator._updates = head._update_count = accumulator._metric._update_count = evidence['updates']
    return accumulator.compute()


def run(directory):
    import stmetrics

    project = Path(__file__).resolve().parents[1]
    metric_source = Path(stmetrics.__file__).resolve().parent
    manifest_path, lock_path = directory / 'FORMAL_POPULATION.json', directory / 'FORMAL_LOCK.json'
    manifest, lock = read_json(manifest_path), read_json(lock_path)
    if sha256(manifest_path) != lock['FORMAL_POPULATION_sha256']:
        raise ValueError('formal population lock differs')
    dataset_spec = Path(manifest['source_data_root']) / 'processed/rio/rio.yaml'
    data_binding = next(r for r in manifest['data_files'] if r['relative_path'] == 'processed/rio/rio.yaml')
    if sha256(dataset_spec) != data_binding['sha256']:
        raise ValueError('official metric taxonomy bytes differ')
    result = {'status': 'INCOMPLETE', 'formal_lock_sha256': sha256(lock_path), 'rows': [],
              'pending': [], 'comparisons': [], 'official_states_verified': 0, 'pooled_states_recomputed': 0,
              'runtime_status': 'RUNTIME_CONDITIONAL', 'source_receipts': []}
    index = {}
    for selected in lock['rows']:
        arm, step = selected['arm'], selected['step']
        receipt = read_json(directory / 'assets' / f'{arm}-{step:05d}.json')
        for role in ('PB', 'LOCAL-T2', 'ADDITIONAL'):
            path = directory / 'collected' / arm / 'evaluation' / arm / 'seed45' / f'{role}-{step:05d}.json'
            if not path.exists() or read_json(path)['status'] != 'COMPLETE':
                result['pending'].append({'arm': arm, 'step': step, 'role': role})
                continue
            row = read_json(path)
            identity = row['identity']
            records = manifest['records'][role]
            expected = dict(collections.Counter(f"T{r['horizon']}" for r in records))
            if (row['completed'] != expected or row['expected'] != expected or row['seed'] != 45
                    or (row['arm'], row['step'], row['role']) != (arm, step, role)
                    or identity['training'] != receipt['training_identity']
                    or identity['checkpoint_sha256'] != receipt['checkpoint_sha256']
                    or identity['formal_lock_sha256'] != sha256(lock_path)
                    or identity['formal_population_sha256'] != sha256(manifest_path)
                    or identity['input_ids'] != [r['input_id'] for r in records]
                    or identity['evaluation_seed'] != 9001
                    or identity['native_process_protocol'] != manifest['native_process_protocol']):
                raise ValueError('formal identity/order/coverage differs: ' + str(path))
            for name, expected_sha in identity['producer_sha256'].items():
                if sha256(project / 'scripts' / name) != expected_sha:
                    raise ValueError('formal producer changed: ' + name)
            for name, expected_sha in identity['metric_sources']['python_source_sha256'].items():
                if sha256(metric_source / name) != expected_sha:
                    raise ValueError('official evaluator changed: ' + name)
            ref_counts = collections.Counter((r['horizon'], r['reference_id']) for r in records)
            actual_refs = {(r['horizon'], r['reference_id']): r for r in row['by_reference']}
            if len(actual_refs) != len(row['by_reference']) or actual_refs.keys() != ref_counts.keys():
                raise ValueError('formal reference population differs')
            for key, ref in actual_refs.items():
                if ref['completed'] != ref_counts[key] or ref['expected'] != ref_counts[key]:
                    raise ValueError('formal reference denominator differs')
            if len(row['evidence']) != len(expected) or len(row['reference_evidence']) != len(ref_counts):
                raise ValueError('formal official evidence coverage differs')
            if {(r['horizon'], r['reference_id']) for r in row['reference_evidence']} != ref_counts.keys():
                raise ValueError('formal reference evidence has duplicate or missing keys')
            for item in (*row['evidence'], *row['reference_evidence']):
                evidence_path = path.with_name(Path(item['path']).name)
                if sha256(evidence_path) != item['sha256']:
                    raise ValueError('collected official evidence SHA differs')
                evidence = read_json(evidence_path)
                horizon = item.get('horizon') or int(evidence_path.name.split('-T')[1].split('-')[0])
                count = ref_counts[horizon, item['reference_id']] if 'reference_id' in item else expected[f'T{horizon}']
                verify_state_content(evidence, count)
                result['official_states_verified'] += 1
                if 'reference_id' not in item:
                    values = recompute(evidence, dataset_spec)
                    if any(abs(values[k] - row['official_metrics'][f'T{horizon}'][k]) > 1e-12 for k in values):
                        raise ValueError('independently recomputed pooled metric differs')
                    if abs(values['online_t-mAP'] - row['metrics'][f'T{horizon}']) > 1e-12:
                        raise ValueError('pooled AP summary differs')
                    result['pooled_states_recomputed'] += 1
            for value in row['metrics'].values():
                if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError('formal metric is not finite in [0,1]')
            long = sum(row['metrics'][f'T{h}'] for h in (3, 4, 5)) / 3 if role == 'PB' else None
            if row['metrics']['S_long'] != long:
                raise ValueError('formal S_long is not the registered three-horizon mean')
            result['rows'].append({'arm': arm, 'step': step, 'role': role, 'kind': selected['kind'],
                                    'inputs': sum(expected.values()), 'metrics': row['metrics'],
                                    'official_metrics': row['official_metrics']})
            result['source_receipts'].append({'path': str(path), 'sha256': sha256(path)})
            index[arm, step, role] = row
    for role in ('PB', 'LOCAL-T2', 'ADDITIONAL'):
        for candidate, control, label in ((('E1', 23760), ('E0', 5940), 'selected_E1_vs_E0'),
            (('E3', 2970), ('E1', 2970), 'matched_step_E3_vs_E1'),
            (('E3', 2970), ('E1', 23760), 'selected_E3_vs_best_E1')):
            if (*candidate, role) not in index or (*control, role) not in index:
                continue
            a, b = index[(*candidate, role)], index[(*control, role)]
            result['comparisons'].append({'comparison': label, 'role': role,
                'delta': {k: a['metrics'][k] - b['metrics'][k] for k in a['metrics'] if a['metrics'][k] is not None},
                'reference_diagnostics': reference_deltas(a, b)})
    if not result['pending']:
        result['status'] = 'COMPLETE'
    write_json_atomic(directory / 'FORMAL_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    row = run(parser.parse_args().directory)
    print({k: row[k] for k in ('status', 'pending', 'official_states_verified', 'pooled_states_recomputed')})
