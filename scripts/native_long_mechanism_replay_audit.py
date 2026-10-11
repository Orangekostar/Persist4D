"""Verify every fixed repeat and compare all original/repeat pairs, without replacement."""

import argparse
from itertools import combinations
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_mechanism import nullable_single_input_metrics
from scripts.native_long_mechanism_audit import pair_alignment
from scripts.native_long_mechanism_replay import validate_plan
from scripts.native_long_postselection import read_json, sha256


def run(directory, dataset_spec):
    import torch

    from scripts.native_long_deployment_validation import tensor_difference
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.system_comparison_inference import _tensor_sha256

    parent_plan_path = directory / 'MECHANISM_PLAN.json'
    replay = directory / 'mechanism-replay-v1/postselection'
    plan_path = replay / 'MECHANISM_PLAN.json'
    original, plan = read_json(parent_plan_path), read_json(plan_path)
    validate_plan(plan, original)
    if (sha256(parent_plan_path) != plan['parent_plan_sha256']
            or sha256(dataset_spec) != plan['metric_spec_sha256']
            or sha256(replay / 'MECHANISM_CPU_PREFLIGHT.json') != plan['cpu_preflight_sha256']):
        raise ValueError('repeat parent, taxonomy or CPU proof differs')
    project = Path(__file__).resolve().parents[1]
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('repeat source differs: ' + name)
    state = read_json(directory / 'MECHANISM_REPLAY_STATE.json')
    if (state['status'] != 'COMPLETE' or state['plan_sha256'] != sha256(plan_path)
            or len(state['completed_jobs']) != 8
            or set(state['completed_jobs']) != {j['name'] for j in plan['jobs']}):
        raise ValueError('repeat queue incomplete')
    originals = [j for j in original['jobs'] if j['name'] in {r['original_job'] for r in plan['jobs']}]
    result = {'status': 'COMPLETE', 'parent_plan_sha256': sha256(parent_plan_path),
              'plan_sha256': sha256(plan_path), 'audit_producer_sha256': sha256(__file__),
              'runtime_status': 'RUNTIME_CONDITIONAL', 'rows_verified': 0, 'comparisons': [],
              'source_receipts': [], 'scope': 'Two diagnostically flagged CAL inputs, original plus two repeats per mode; '
              'all pairwise comparisons retained; no replacement of original outcomes or independent training replication'}
    cached = {}
    for job in originals + plan['jobs']:
        is_repeat = 'original_job' in job
        folder = replay if is_repeat else directory
        path = folder / 'mechanism' / (job['name'] + '.json')
        row = read_json(path)
        record = plan['records'][job['record_index']]
        expected_plan_sha = sha256(plan_path if is_repeat else parent_plan_path)
        if (row['status'] != 'COMPLETE' or row['job'] != job or row['record'] != record
                or row['plan_sha256'] != expected_plan_sha or row['evaluation_seed'] != 9001
                or sha256(path.with_suffix('.pt')) != row['tensor_evidence']['sha256']):
            raise ValueError('repeat job or tensor receipt differs')
        tensors = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        if (len(tensors['stages']) != 13
                or [{k: _tensor_sha256(v) for k, v in stage.items()} for stage in tensors['stages']] != row['stage_sha256']
                or {k: _tensor_sha256(v) for k, v in tensors['target'].items()} != row['target_sha256']):
            raise ValueError('repeat stage or target bytes differ')
        accumulator = OfficialMetricAccumulator(mode='strict_online', dataset_spec=dataset_spec)
        accumulator.update(tensors['prediction'], tensors['target'])
        if accumulator.export_evidence() != row['official_state'] or nullable_single_input_metrics(accumulator) != row['metrics']:
            raise ValueError('repeat official state or metric recomputation differs')
        result['source_receipts'].append({'name': job['name'], 'sha256': sha256(path),
            'tensor_sha256': row['tensor_evidence']['sha256'], 'record_index': job['record_index'],
            'mode': job['mode'], 'repeat': job.get('repeat', 0), 'metrics': row['metrics']})
        result['rows_verified'] += 1
        cached[job['name']] = (row, tensors)
    for index in (21, 25):
        rows = [v for v in cached.values() if v[0]['job']['record_index'] == index]
        if len(rows) != 6:
            raise ValueError('repeat comparison coverage differs')
        for (left, a), (right, b) in combinations(rows, 2):
            alignment = pair_alignment(left, right)
            result['comparisons'].append({'record_index': index,
                'left': left['job']['name'], 'right': right['job']['name'],
                'same_mode': left['job']['mode'] == right['job']['mode'], 'alignment': alignment,
                'stages': [{k: tensor_difference(x[k], y[k]) for k in ('logits', 'masks')}
                           for x, y in zip(a['stages'], b['stages'])],
                'prediction': {k: tensor_difference(a['prediction'][k], b['prediction'][k]) for k in a['prediction']},
                'official_state_exact': left['official_state'] == right['official_state'],
                'metrics_exact': left['metrics'] == right['metrics'],
                'metric_left_minus_right': {k: left['metrics'][k] - right['metrics'][k]
                    if left['metrics'][k] is not None and right['metrics'][k] is not None else None
                    for k in left['metrics']}})
    result['summary'] = {}
    for same_mode in (True, False):
        rows = [r for r in result['comparisons'] if r['same_mode'] == same_mode]
        result['summary']['same_mode' if same_mode else 'cross_mode'] = {
            'pairs': len(rows), 'upstream_different_pairs': sum(not r['alignment']['upstream_auxiliary_exact'] for r in rows),
            'feedback_input_different_pairs': sum(not r['alignment']['feedback_input_exact'] for r in rows),
            'official_state_different_pairs': sum(not r['official_state_exact'] for r in rows),
            'metrics_different_pairs': sum(not r['metrics_exact'] for r in rows)}
    write_json_atomic(directory / 'MECHANISM_REPLAY_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--dataset-spec', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.directory, args.dataset_spec)
    print({k: result[k] for k in ('status', 'rows_verified', 'summary')})
