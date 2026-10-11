"""Audit trained profile and fresh-process A versus B-then-A runtime evidence."""

import argparse
import statistics
from pathlib import Path

import torch

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def tensor_delta(a, b):
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('runtime evidence must contain finite tensors')
    row = {'shape_equal': a.shape == b.shape, 'dtype_equal': a.dtype == b.dtype,
           'left_shape': list(a.shape), 'right_shape': list(b.shape)}
    row['exact'] = row['shape_equal'] and row['dtype_equal'] and torch.equal(a, b)
    if row['shape_equal']:
        row['different_fraction'] = float((a != b).double().mean()) if a.numel() else 0.
        row['max_abs'] = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    return row


def run(directory):
    plan_path = directory / 'RUNTIME_PLAN.json'
    plan = read_json(plan_path)
    state = read_json(directory / 'collected/E0/RUNTIME_STATE.json')
    folder = directory / 'collected/E0/runtime'
    if state['status'] != 'COMPLETE' or len(state['completed_jobs']) != 12 or state['plan_sha256'] != sha256(plan_path):
        raise ValueError('runtime queue or frozen plan is incomplete')
    if {p.stem for p in folder.glob('*.json')} != set(state['completed_jobs']):
        raise ValueError('collected runtime jobs differ from the completed queue')
    result = {'status': 'VERIFIED', 'runtime_status': 'RUNTIME_CONDITIONAL', 'plan_sha256': sha256(plan_path),
              'gpu_identity': state['gpu_identity'], 'source_receipts': [], 'profile': [], 'comparisons': [],
              'profile_scope': 'five fixed CAL inputs; one warmup and three measurements per input; one A40',
              'order_scope': 'fixed CAL T2 A, fixed CAL T5 B; fresh processes, two repeats per order and arm'}
    tensors = {}
    for name in state['completed_jobs']:
        path = folder / (name + '.json')
        row = read_json(path)
        expected = next(r for r in plan['rows'] if (r['arm'], r['step']) == (row['arm'], row['step']))
        if (row['status'] != 'COMPLETE' or row['bundle_sha256'] != expected['sha256']
                or row['checkpoint_sha256'] != expected['checkpoint_sha256']
                or row['producer_sha256'] != plan['producer_sha256']
                or row['population_sha256'] != plan['population_sha256']
                or row['records'] != plan['records'] or row['evaluation_seed'] != 9001):
            raise ValueError('runtime row identity differs: ' + name)
        result['source_receipts'].append({'path': str(path), 'sha256': sha256(path)})
        if row['mode'] == 'profile':
            keys = [(r['horizon'], r['repeat']) for r in row['rows']]
            if sorted(keys) != [(h, r) for h in range(1, 6) for r in range(4)]:
                raise ValueError('profile repeats/horizons differ')
            for h in range(1, 6):
                rows = [r for r in row['rows'] if r['horizon'] == h and not r['warmup']]
                if len(rows) != 3 or any(r['warmup'] != (r['repeat'] == 0) for r in row['rows']):
                    raise ValueError('profile warmup boundary differs')
                summary = {'arm': row['arm'], 'step': row['step'], 'horizon': h, 'measurements': len(rows)}
                for key in ('network_seconds', 'feedback_cuda_seconds', 'input_io_collate_seconds',
                            'H2D_seconds', 'postprocess_seconds', 'end_to_end_seconds'):
                    summary[key + '_mean'] = statistics.mean(r[key] for r in rows)
                    summary[key + '_median'] = statistics.median(r[key] for r in rows)
                for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes'):
                    summary[key + '_max'] = max(r[key] for r in rows)
                result['profile'].append(summary)
        elif row['mode'] == 'order':
            evidence_path = folder / Path(row['tensor_evidence']['path']).name
            if sha256(evidence_path) != row['tensor_evidence']['sha256']:
                raise ValueError('runtime tensor evidence SHA differs')
            evidence = torch.load(evidence_path, map_location='cpu', weights_only=True)
            if evidence['metrics'] != row['metrics']:
                raise ValueError('runtime AP JSON and tensor evidence differ')
            result['source_receipts'].append({'path': str(evidence_path), 'sha256': sha256(evidence_path)})
            tensors[name] = evidence
        else:
            raise ValueError('unexpected runtime job mode')
    for arm in ('E1', 'E3'):
        for left, right, kind in (('A-0', 'A-1', 'fresh_repeat_A'), ('BA-0', 'BA-1', 'fresh_repeat_BA'),
                                  ('A-0', 'BA-0', 'process_order_repeat0'), ('A-1', 'BA-1', 'process_order_repeat1')):
            a, b = (tensors[f'{arm}-02970-order-{suffix}'] for suffix in (left, right))
            compared = {key: tensor_delta(a[key], b[key]) for key in
                        ('input_coord', 'input_feat', 'inverse', 'point2segment', 'final_logits', 'final_masks')}
            predictions = {key: tensor_delta(a['prediction'][key], b['prediction'][key]) for key in a['prediction']}
            result['comparisons'].append({'arm': arm, 'left': left, 'right': right, 'kind': kind,
                'tensors': compared, 'predictions': predictions,
                'input_exact': all(compared[k]['exact'] for k in ('input_coord', 'input_feat', 'inverse', 'point2segment')),
                'output_exact': all(compared[k]['exact'] for k in ('final_logits', 'final_masks')),
                'official_prediction_exact': all(r['exact'] for r in predictions.values()),
                'left_metrics': a['metrics'], 'right_metrics': b['metrics'],
                'metric_delta_right_minus_left': {k: b['metrics'][k] - a['metrics'][k] for k in a['metrics']}})
    result['order_dependence_observed'] = any(r['input_exact'] and not r['output_exact']
        for r in result['comparisons'] if r['kind'].startswith('process_order'))
    result['repeat_variation_observed'] = any(not r['output_exact']
        for r in result['comparisons'] if r['kind'].startswith('fresh_repeat'))
    write_json_atomic(directory / 'RUNTIME_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    result = run(parser.parse_args().directory)
    print({k: result[k] for k in ('status', 'runtime_status', 'order_dependence_observed', 'repeat_variation_observed')})
