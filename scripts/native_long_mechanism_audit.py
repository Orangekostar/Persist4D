"""Recompute trained CAL diagnostics without changing weights or selecting checkpoints."""

import argparse
import collections
import math
from itertools import pairwise
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_mechanism import (
    candidate_diagnostics,
    nullable_single_input_metrics,
)
from scripts.native_long_postselection import read_json, sha256


def pair_alignment(active, off):
    """Keep runtime-confounded pairs without treating them as isolated interventions."""
    result = {'input_exact': active['input_sha256'] == off['input_sha256'],
              'target_exact': active['target_sha256'] == off['target_sha256']}
    if (not all(result.values()) or len(active['feedback']) != 1 or len(off['feedback']) != 1
            or len(active['stage_sha256']) != 13 or len(off['stage_sha256']) != 13):
        raise ValueError('feedback pair changed an input, target or capture count')
    result['feedback_input_exact'] = active['feedback'][0]['input_sha256'] == off['feedback'][0]['input_sha256']
    result['computed_feedback_exact'] = active['feedback'] == off['feedback']
    result['changed_upstream_stages'] = [i for i in range(8)
        if active['stage_sha256'][i] != off['stage_sha256'][i]]
    result['upstream_auxiliary_exact'] = not result['changed_upstream_stages']
    result['attribution_eligible'] = all(result[k] for k in (
        'input_exact', 'target_exact', 'feedback_input_exact',
        'computed_feedback_exact', 'upstream_auxiliary_exact'))
    result['status'] = 'ALIGNED' if result['attribution_eligible'] else 'RUNTIME_CONFOUNDED'
    return result


def candidate_delta(active, control):
    a, b = ({r['gt_id']: r for r in rows} for rows in (active, control))
    if (len(a) != len(active) or len(b) != len(control) or a.keys() != b.keys()
            or any(a[k]['gt_class'] != b[k]['gt_class'] for k in a)):
        raise ValueError('paired candidate ground truth differs')
    return {'gt_objects': len(a), 'matched_with_feedback': sum(r['matched'] for r in a.values()),
            'matched_without_feedback': sum(r['matched'] for r in b.values()),
            'new_high_iou_objects': sorted(k for k in a if a[k]['matched'] and not b[k]['matched']),
            'lost_high_iou_objects': sorted(k for k in a if b[k]['matched'] and not a[k]['matched']),
            'preserved_high_iou_objects': sorted(k for k in a if a[k]['matched'] and b[k]['matched']),
            'high_iou_candidate_count_delta': sum(r['high_iou_candidates'] for r in active)
            - sum(r['high_iou_candidates'] for r in control)}


def nested_candidate_trace(rows):
    if len(rows) != 5:
        raise ValueError('nested diagnostic requires all five causal prefixes')
    indices = [{r['gt_id']: r for r in visit} for visit in rows]
    if any(len(index) != len(visit) for index, visit in zip(indices, rows)):
        raise ValueError('duplicate current GT instance')
    result = []
    for identity in sorted(set().union(*(r.keys() for r in indices))):
        values = [r.get(identity) for r in indices]
        visible = [(h, r) for h, r in enumerate(values, 1) if r is not None]
        adjacent = [(a, b) for a, b in pairwise(values)
                    if a is not None and b is not None and a['gt_class'] == b['gt_class']]
        hits = [(a, b) for a, b in adjacent if a['matched'] and b['matched']]
        result.append({'gt_id': identity,
            'statuses': ['absent' if r is None else ('matched' if r['matched'] else 'missed') for r in values],
            'classes': [r['gt_class'] if r else None for r in values],
            'best_query_ids': [r['best_query_id'] if r else None for r in values],
            'class_candidate_counts': [r['class_candidates'] if r else None for r in values],
            'high_iou_candidate_counts': [r['high_iou_candidates'] if r else None for r in values],
            'first_visible_horizon': visible[0][0],
            'first_observed_failure_horizon': next((h for h, r in visible if not r['matched']), None),
            'class_change_count': sum(a['gt_class'] != b['gt_class'] for (_, a), (_, b) in pairwise(visible)),
            'lost_after_visible_hit': sum(a['matched'] and not b['matched'] for a, b in adjacent),
            'recovered_after_visible_miss': sum(not a['matched'] and b['matched'] for a, b in adjacent),
            'matched_query_transition_opportunities': len(hits),
            'matched_query_changes': sum(a['best_query_id'] != b['best_query_id'] for a, b in hits)})
    return result


def summarize_sampling(rows):
    result = {'calls': len(rows), 'cap_triggered_calls': 0, 'available_stage_opportunities': 0,
              'available_past_stage_opportunities': 0, 'uniform_missing': 0, 'stratified_missing': 0,
              'uniform_missing_past': 0, 'stratified_missing_past': 0}
    spreads = {'uniform': [], 'stratified': []}
    for row in rows:
        capacities = row['available_per_stage']
        if (sum(capacities) != row['available'] or any(c <= 0 for c in capacities)
                or row['selected'] != min(row['cap'], row['available'])
                or len(capacities) != len(row['stages']) or row['stages'] != sorted(set(row['stages']))
                or row['cap_triggered'] != (row['available'] > row['cap'])):
            raise ValueError('sampling availability or cap differs')
        result['cap_triggered_calls'] += row['cap_triggered']
        result['available_stage_opportunities'] += len(capacities)
        result['available_past_stage_opportunities'] += max(0, len(capacities) - 1)
        for mode in ('uniform', 'stratified'):
            counts = row[mode]['counts']
            missing = sum(c == 0 for c in counts)
            if (len(counts) != len(capacities) or sum(counts) != row['selected']
                    or any(c < 0 or c > n for c, n in zip(counts, capacities))
                    or missing != row[mode]['missing_available_stages']):
                raise ValueError('sampling counts exceed availability or change the budget')
            if mode == 'stratified' and row['selected'] >= len(capacities) and missing:
                raise ValueError('stratified sampling omitted an available stage despite sufficient quota')
            result[mode + '_missing'] += missing
            result[mode + '_missing_past'] += sum(c == 0 for c in counts[:-1])
            if row['selected']:
                spreads[mode].append((max(counts) - min(counts)) / row['selected'])
    result['balance_scope'] = 'mean per-call max-minus-min selected stage share; calls are repeated measurements'
    for mode, values in spreads.items():
        result[mode + '_mean_stage_share_spread'] = sum(values) / len(values) if values else None
    return result


def validate_gradients(row):
    groups = {'feedback': 'model.native_long_feedback.', 'mask_head': 'model.mask_embed_head.',
              'cross_attention': 'model.cross_attention.', 'self_attention': 'model.self_attention.',
              'decoder_ffn': 'model.ffn_attention.'}
    if (not all(row[k] is True for k in ('no_optimizer_step', 'parameters_unchanged', 'frozen_encoder_gradients_absent'))
            or row['auxiliary_count'] != 12 or row['feedback_checkpoint_chunks']
            or any(not math.isfinite(v) or v < 0 for v in row['gradient_norms'].values())
            or not math.isfinite(row['total_loss'])):
        raise ValueError('trained gradient receipt violates the fixed objective/parameter contract')
    for name, prefix in groups.items():
        norms = [v for k, v in row['gradient_norms'].items() if k.startswith(prefix)]
        expected = {'tensor_count': len(norms), 'l2': sum(v*v for v in norms) ** .5}
        recorded = row['gradient_groups'][name]
        # JSON sorts parameter names, changing only the floating summation order.
        if (expected['tensor_count'] != recorded['tensor_count'] or not expected['l2'] > 0
                or not math.isclose(expected['l2'], recorded['l2'], rel_tol=1e-12, abs_tol=0.)):
            raise ValueError('trained gradient group is missing, zero or incorrectly summarized')


def run(directory, dataset_spec):
    import torch

    from scripts.native_long_deployment_validation import tensor_difference
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.system_comparison_inference import _tensor_sha256

    project = Path(__file__).resolve().parents[1]
    plan_path = directory / 'MECHANISM_PLAN.json'
    plan, plan_sha = read_json(plan_path), sha256(plan_path)
    if (sha256(dataset_spec) != plan['metric_spec_sha256']
            or sha256(directory / 'MECHANISM_CPU_PREFLIGHT.json') != plan['cpu_preflight_sha256']):
        raise ValueError('mechanism taxonomy or CPU population proof differs')
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('mechanism measurement source differs: ' + name)
    expected_modes = {('E0', 5940, 'normal'), ('E1', 23760, 'normal'), ('E1', 2970, 'normal'),
                      ('E3', 2970, 'normal'), ('E3', 2970, 'off')}
    expected = {(*key, i) for key in expected_modes for i in range(30)}
    expected.update({('E3', 2970, 'gradient', i) for i in (1, 4)})
    job_keys = {(j['arm'], j['step'], j['mode'], j['record_index']) for j in plan['jobs']}
    if len(plan['jobs']) != 152 or job_keys != expected or len(plan['records']) != 30:
        raise ValueError('mechanism fixed job coverage differs')
    state_path = directory / 'MECHANISM_STATE.json'
    state = read_json(state_path)
    complete = set(state['completed_jobs'])
    if (state['plan_sha256'] != plan_sha or len(complete) != len(state['completed_jobs'])
            or not complete.issubset({j['name'] for j in plan['jobs']})):
        raise ValueError('mechanism worker state differs from plan')
    result = {'status': 'INCOMPLETE', 'plan_sha256': plan_sha, 'pending': [], 'rows_verified': 0,
              'source_receipts': [{'path': str(state_path), 'sha256': sha256(state_path)}],
              'gradient_probes': [], 'pooled_metrics': [], 'by_reference': [],
              'feedback_pairs': [], 'nested_prefix_traces': [], 'sampling': [],
              'runtime_status': 'RUNTIME_CONDITIONAL', 'scope': 'fixed CAL explanatory panel; two physical references, one training seed',
              'candidate_metric_scope': 'class-compatible best IoU >= 0.5 per visible GT; not official AP or one-to-one/persistent-ID matching',
              'query_change_scope': 'raw query-slot change between consecutive matched, same-class visible visits; not a tracker identity error'}
    result['absence_scope'] = 'absent means missing from the current official GT target; physical absence and target filtering are not distinguished'
    folder = directory / 'mechanism'
    accumulators, ref_accumulators, counts = {}, {}, collections.Counter()
    cached = {}
    for job in plan['jobs']:
        if job['name'] not in complete:
            result['pending'].append(job['name'])
            continue
        path = folder / (job['name'] + '.json')
        row = read_json(path)
        record = plan['records'][job['record_index']]
        if (row['status'] != 'COMPLETE' or row['job'] != job or row['record'] != record
                or row['plan_sha256'] != plan_sha or row['evaluation_seed'] != 9001):
            raise ValueError('mechanism job, weights, population or seed differs')
        result['source_receipts'].append({'path': str(path), 'sha256': sha256(path)})
        if job['mode'] == 'gradient':
            validate_gradients(row)
            result['gradient_probes'].append({k: row[k] for k in ('job', 'gradient_groups', 'total_loss',
                'no_optimizer_step', 'parameters_unchanged', 'frozen_encoder_gradients_absent', 'peak_allocated_bytes')})
        else:
            tensor_path = path.with_suffix('.pt')
            if sha256(tensor_path) != row['tensor_evidence']['sha256']:
                raise ValueError('mechanism tensor artifact bytes differ')
            tensors = torch.load(tensor_path, map_location='cpu', weights_only=True)
            if (len(tensors['stages']) != 13
                    or [{k: _tensor_sha256(v) for k, v in stage.items()} for stage in tensors['stages']] != row['stage_sha256']
                    or {k: _tensor_sha256(v) for k, v in tensors['target'].items()} != row['target_sha256']):
                raise ValueError('mechanism stage/target tensors differ from receipt')
            prediction, target = tensors['prediction'], tensors['target']
            current = target['temporal_stages'] == record['horizon'] - 1
            latest_prediction = {**prediction, 'pred_masks': prediction['pred_masks'][current]}
            latest_target = {**target, 'masks': target['masks'][:, current]}
            if (candidate_diagnostics(prediction, target) != row['prefix_candidates']
                    or candidate_diagnostics(latest_prediction, latest_target) != row['current_candidates']):
                raise ValueError('candidate metrics differ from saved prediction and ground truth')
            single = OfficialMetricAccumulator(mode='strict_online', dataset_spec=dataset_spec)
            single.update(prediction, target)
            if single.export_evidence() != row['official_state'] or nullable_single_input_metrics(single) != row['metrics']:
                raise ValueError('single-input official state differs from prediction/target recomputation')
            key = (job['arm'], job['step'], job['mode'], record['horizon'])
            ref_key = (*key, record['reference_id'])
            for mapping, group in ((accumulators, key), (ref_accumulators, ref_key)):
                if group not in mapping:
                    mapping[group] = OfficialMetricAccumulator(mode='strict_online', dataset_spec=dataset_spec)
                mapping[group].update(prediction, target)
                counts[group] += 1
            if job['arm'] == 'E1' and job['step'] == 23760 and record['horizon'] == 5:
                if len(row['sampling']) != 12:
                    raise ValueError('mechanism sampling trace omitted decoder stages')
                result['sampling'].append({'input_id': record['input_id'], 'reference_id': record['reference_id'],
                                            'counts': summarize_sampling(row['sampling']), 'stages': row['sampling']})
            elif row['sampling']:
                raise ValueError('sampling probe ran outside its fixed panel')
            cached[job['arm'], job['step'], job['mode'], job['record_index']] = row
            del tensors, single
        result['rows_verified'] += 1
    for mapping, output in ((accumulators, 'pooled_metrics'), (ref_accumulators, 'by_reference')):
        for key, accumulator in sorted(mapping.items()):
            arm, step, mode, horizon, *refs = key
            expected_count = sum(r['horizon'] == horizon and (not refs or r['reference_id'] == refs[0]) for r in plan['records'])
            if counts[key] != expected_count:
                continue
            evidence = accumulator.export_evidence()
            filename = '-'.join(map(str, key)) + '.json'
            path = directory / 'mechanism-audit' / 'official' / filename
            write_json_atomic(path, evidence)
            result[output].append({'arm': arm, 'step': step, 'mode': mode, 'horizon': horizon,
                'reference_id': refs[0] if refs else None, 'inputs': counts[key],
                'metrics': nullable_single_input_metrics(accumulator), 'official_state_sha256': evidence['sha256'],
                'official_file': filename, 'official_file_sha256': sha256(path)})
    for index, record in enumerate(plan['records']):
        active, off = cached.get(('E3', 2970, 'normal', index)), cached.get(('E3', 2970, 'off', index))
        if active is None or off is None:
            continue
        alignment = pair_alignment(active, off)
        a, b = [torch.load(folder / (r['job']['name'] + '.pt'), map_location='cpu', weights_only=True) for r in (active, off)]
        stages = [{k: tensor_difference(x[k], y[k]) for k in ('logits', 'masks')}
                  for x, y in zip(a['stages'], b['stages'])]
        result['feedback_pairs'].append({'input_id': record['input_id'], 'panel_unit': record['panel_unit'],
            'reference_id': record['reference_id'], 'horizon': record['horizon'],
            'record_index': index, 'alignment': alignment,
            'upstream_auxiliary_stages_exact': 8 - len(alignment['changed_upstream_stages']), 'stages': stages,
            'computed_feedback_norm': active['feedback'][0]['update_norm'], 'off_applied_feedback_norm': 0.,
            'prefix': candidate_delta(active['prefix_candidates'], off['prefix_candidates']),
            'current': candidate_delta(active['current_candidates'], off['current_candidates'])})
        del a, b
    confounded = [p['record_index'] for p in result['feedback_pairs'] if not p['alignment']['attribution_eligible']]
    result['counterfactual_verdict'] = ('PARTIALLY_CONFOUNDED' if confounded else
                                       ('ALIGNED_COMPLETE_PANEL' if len(result['feedback_pairs']) == 30 else 'INCOMPLETE'))
    result['pair_counts'] = {'measured': len(result['feedback_pairs']), 'expected': 30,
                            'attribution_eligible': len(result['feedback_pairs']) - len(confounded),
                            'runtime_confounded': len(confounded), 'confounded_record_indices': confounded}
    result['pooled_feedback_comparison_scope'] = ('All original inputs are retained, including confounded pairs; '
        'pooled AP differences describe the full intervention runs and do not isolate feedback causality. '
        'Alignment is a diagnostic condition, not a new selection rule or independent training replication.')
    for arm, step, mode in sorted(expected_modes):
        for unit in sorted({r['panel_unit'] for r in plan['records']}):
            indices = [i for i, r in enumerate(plan['records']) if r['panel_unit'] == unit]
            rows = [cached.get((arm, step, mode, i)) for i in indices]
            if any(row is None for row in rows):
                continue
            if [r['record']['horizon'] for r in rows] != list(range(1, 6)):
                raise ValueError('nested prefix ordering differs')
            result['nested_prefix_traces'].append({'arm': arm, 'step': step, 'mode': mode,
                'panel_unit': unit, 'reference_id': rows[0]['record']['reference_id'],
                'objects': nested_candidate_trace([row['current_candidates'] for row in rows])})
    result['sampling_total'] = summarize_sampling([s for row in result['sampling'] for s in row['stages']])
    if not result['pending']:
        if (state['status'] != 'COMPLETE' or len(result['pooled_metrics']) != 25
                or len(result['by_reference']) != 50 or len(result['gradient_probes']) != 2
                or len(result['feedback_pairs']) != 30 or len(result['nested_prefix_traces']) != 30
                or len(result['sampling']) != 6):
            raise ValueError('mechanism queue or audit output coverage is incomplete')
        result['status'] = 'COMPLETE'
    result['audit_producer_sha256'] = sha256(__file__)
    write_json_atomic(directory / 'MECHANISM_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--dataset-spec', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.directory, args.dataset_spec)
    print({k: result[k] if k != 'pending' else len(result[k]) for k in ('status', 'rows_verified', 'pending')})
