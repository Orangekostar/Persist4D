"""Fixed seed replication, paired comparisons and independently recomputed evidence."""
import csv
from collections import defaultdict
from pathlib import Path

from scripts.p6a_metrics import recompute_official_metric_evidence
from scripts.qp_mn_binding import gpu_scope
from scripts.qp_mn_evaluation import ARMS, classify_gain, evaluate_point, load_head
from scripts.qp_mn_training import train_arm
from scripts.short_module_evaluation import write_csv
from scripts.short_module_identity import file_digest
from scripts.short_module_screen import read_json, write_json
from scripts.short_module_training import load_training_records

POSITIVE = {'POSITIVE_SINGLE_SEED', 'TARGET_MAGNITUDE'}


def build_replication_plan(sel, lock):
    steps, candidates = defaultdict(set), []
    if {r['module'] for r in sel['rows']} != set(ARMS) or any(r['status'] != 'COMPLETE' for r in sel['rows']):
        raise ValueError('complete locked SEL results required for replication planning')
    baseline = sel['baseline']['metrics']
    for row in sel['rows']:
        arm = row['module']
        if row['seed'] != 45 or row['role'] != 'SEL' or row['step'] != lock['selected_updates'][arm]:
            raise ValueError('replication trigger must use seed45 locked SEL points')
        d2, d1 = (row['metrics'][k] - baseline[k] for k in ('T2', 'T1'))
        if classify_gain(d2, d1) in POSITIVE:
            candidates.append(arm)
            steps[arm].add(row['step'])
            if arm in ('Q_P', 'M_N'):
                control = {'Q_P': 'Q_A0', 'M_N': 'M_C'}[arm]
                steps[control].update((row['step'], lock['selected_updates'][control]))
    return {'seed': 46, 'schedule_horizon': 1500, 'candidates': sorted(candidates),
            'steps': {a: sorted(v) for a, v in sorted(steps.items())},
            'rule': 'all protected positives; methods paired at method selected step plus at most control own locked point',
            'seed46_selection_allowed': False}


def prepare_replication(context):
    c = context
    sel = read_json(c.artifacts / 'evaluation/SEL_LOCKED.json')
    lock = read_json(c.artifacts / 'selection/CAL_LOCK.json')
    plan = build_replication_plan(sel, lock)
    plan['source_sel_sha256'] = file_digest(c.artifacts / 'evaluation/SEL_LOCKED.json')
    plan['source_lock_sha256'] = file_digest(c.artifacts / 'selection/CAL_LOCK.json')
    path = c.artifacts / 'selection/REPLICATION_PLAN.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('replication plan is immutable; no second-seed reselection')
    write_json(path, plan)
    return plan


def replicate(context, *, device='cuda:0'):
    c = context
    plan = prepare_replication(c)
    rows = []
    pending = {}
    for arm, points in plan['steps'].items():
        manifest_path = c.artifacts / 'training' / arm / 'seed46/checkpoint_manifest.json'
        manifest = read_json(manifest_path) if manifest_path.exists() else {'checkpoints': []}
        if not set(points) <= {r['step'] for r in manifest['checkpoints']}:
            pending[arm] = points
        else:
            for point in points:
                load_head(c, arm, point, 46, 'cpu')
    if pending:
        records = load_training_records(c.root, c.train_index)
        with gpu_scope(c, device=device, stage='replicate-seed46-training'):
            for arm, points in pending.items():
                train_arm(c, arm, seed=46, updates=max(points), device=device, records=records)
            del records
            # The process retains its CUDA context during CPU scoring, so charge
            # that reservation too. The stage subprocess exits immediately after.
            rows = _replication_evaluation(c, plan)
    else:
        rows = _replication_evaluation(c, plan)
    result = {'status': 'COMPLETE' if rows else 'NOT_APPLICABLE', 'plan': plan, 'rows': rows}
    write_json(c.artifacts / 'evaluation/SEED46_FIXED.json', result)
    return result


def _replication_evaluation(context, plan):
    return [evaluate_point(context, module=arm, step=step, seed=46, role=role, device='cpu')
            for arm, points in plan['steps'].items() for step in points for role in ('CAL', 'SEL')]


def comparison_tables(context):
    c = context
    cal = read_json(c.artifacts / 'evaluation/CAL_ALL.json')
    sel = read_json(c.artifacts / 'evaluation/SEL_LOCKED.json')
    endpoint = read_json(c.artifacts / 'evaluation/SEL_ENDPOINT_1500.json')
    lock = read_json(c.artifacts / 'selection/CAL_LOCK.json')
    replication_file = c.artifacts / 'evaluation/SEED46_FIXED.json'
    replication = read_json(replication_file) if replication_file.exists() else {'rows': []}
    table, by_reference, paired = [], [], []
    def flatten(row, baseline):
        return {'module': row['module'], 'seed': row['seed'], 'step': row['step'], 'role': row['role'],
                'condition': 'NOAUG_NATIVE', 'status': row['status'], **row['metrics'],
                'delta_T1_B0': row['metrics']['T1']-baseline['metrics']['T1'],
                'delta_T2_B0': row['metrics']['T2']-baseline['metrics']['T2'], 'result_path': row['result_path']}
    for name, collection in [('CAL_ALL', cal), ('SEL_LOCKED', sel), ('SEL_ENDPOINT_1500', endpoint)]:
        write_csv(c.artifacts / f'evaluation/{name}.csv', [flatten(r, collection['baseline']) for r in [collection['baseline'], *collection['rows']]])
    for collection in (cal, sel, endpoint):
        for row in [collection['baseline'], *collection['rows']]:
            for ref in row['by_reference']:
                by_reference.append({'module': row['module'], 'seed': row['seed'], 'step': row['step'],
                                     'role': row['role'], **ref})
    for row in replication['rows']:
        by_reference.extend([{'module': row['module'], 'seed': row['seed'], 'step': row['step'],
                              'role': row['role'], **r} for r in row['by_reference']])
    by_reference = list({tuple(sorted(r.items())): r for r in by_reference}.values())
    write_csv(c.artifacts / 'evaluation/BY_REFERENCE.csv', by_reference)
    base_refs = {r['reference_id']: r['AP'] for r in sel['baseline']['by_reference'] if r['H'] == 2}
    for row in sel['rows']:
        arm = row['module']
        cal_row = next(r for r in cal['rows'] if r['module'] == arm and r['step'] == row['step'])
        d2, d1 = (row['metrics'][k]-sel['baseline']['metrics'][k] for k in ('T2', 'T1'))
        classification = classify_gain(d2, d1)
        positive_refs = sum(r['AP'] - base_refs[r['reference_id']] > 1e-6 for r in row['by_reference'] if r['H'] == 2)
        second = next((r for r in replication['rows'] if r['module'] == arm and r['role'] == 'SEL' and r['step'] == row['step']), None)
        d2_second = second['metrics']['T2']-sel['baseline']['metrics']['T2'] if second else None
        d1_second = second['metrics']['T1']-sel['baseline']['metrics']['T1'] if second else None
        level = 'NOT_REPLICATED' if classification in POSITIVE else 'NEGATIVE_OR_TRADEOFF'
        if second and classification in POSITIVE:
            if d2_second <= 1e-6:
                level = 'MIXED_SEED'
            elif d1_second < -.002 - 1e-6:
                level = 'SECOND_SEED_T1_TRADEOFF'
            elif positive_refs >= 3:
                level = 'DEVELOPMENT_REPLICATED'
            else:
                level = 'POSITIVE_TWO_SEED_REFERENCE_INCONSISTENT'
        manifest = read_json(c.artifacts / 'training' / arm / 'seed45/checkpoint_manifest.json')
        table.append({'module': arm, 'selected_step': row['step'], 'updates_new': manifest['updates_new'],
                      'updates_reused': manifest['updates_reused'], 'CAL_T1': cal_row['metrics']['T1'],
                      'CAL_T2': cal_row['metrics']['T2'], 'SEL_T1': row['metrics']['T1'], 'SEL_T2': row['metrics']['T2'],
                      'delta_T1_B0': d1, 'delta_T2_B0': d2, 'classification': classification,
                      'target_magnitude': d2 >= .005-1e-6 and classification in POSITIVE,
                      'positive_references': positive_refs, 'reference_count': len(base_refs),
                      'seed46_delta_T1': d1_second, 'seed46_delta_T2': d2_second, 'evidence_level': level})
    for method, control in [('Q_P','Q_A0'), ('Q_P','Q2_F'), ('M_N','M_C')]:
        for role, rows in [('CAL',cal['rows']),('SEL',sel['rows'])]:
            a = next(r for r in rows if r['module'] == method and r['step'] == lock['selected_updates'][method])
            b = next(r for r in rows if r['module'] == control and r['step'] == lock['selected_updates'][control])
            paired.append({'comparison': f'{method}-{control}', 'role': role, 'kind': 'own_CAL_selected_points',
                           'seed': 45, 'method_step': a['step'], 'control_step': b['step'],
                           **{f'delta_{k}': a['metrics'][k]-b['metrics'][k] for k in ('T1','T2')}})
        for role, rows, steps in [('CAL',cal['rows'],(500,1000,1500)),('SEL',endpoint['rows'],(1500,))]:
            for step in steps:
                a, b = [next(r for r in rows if r['module'] == arm and r['step'] == step) for arm in (method,control)]
                paired.append({'comparison': f'{method}-{control}', 'role': role, 'kind': 'same_step', 'seed':45,
                               'method_step':step, 'control_step':step,
                               **{f'delta_{k}': a['metrics'][k]-b['metrics'][k] for k in ('T1','T2')}})
        step = lock['selected_updates'][method]
        second = [next((r for r in replication['rows'] if r['module'] == arm and r['step'] == step and r['role'] == 'SEL'), None)
                  for arm in (method,control)]
        if all(second):
            paired.append({'comparison':f'{method}-{control}', 'role':'SEL', 'kind':'same_locked_step', 'seed':46,
                           'method_step':step, 'control_step':step,
                           **{f'delta_{k}':second[0]['metrics'][k]-second[1]['metrics'][k] for k in ('T1','T2')}})
    write_csv(c.artifacts / 'evaluation/PAIRED_DELTAS.csv', paired)
    baseline_row = {'module': 'B0', 'selected_step': 0, 'updates_new': 0, 'updates_reused': 0,
                    'CAL_T1': cal['baseline']['metrics']['T1'], 'CAL_T2': cal['baseline']['metrics']['T2'],
                    'SEL_T1': sel['baseline']['metrics']['T1'], 'SEL_T2': sel['baseline']['metrics']['T2'],
                    'delta_T1_B0': 0., 'delta_T2_B0': 0., 'classification': 'BASELINE',
                    'target_magnitude': False, 'seed46_delta_T1': None, 'seed46_delta_T2': None,
                    'positive_references': 0, 'reference_count': len(base_refs), 'evidence_level': 'FROZEN_BASELINE'}
    main_table = [baseline_row, *table]
    profile_path = c.artifacts / 'resources/PROFILE.csv'
    profile_rows = list(csv.DictReader(profile_path.open())) if profile_path.exists() else []
    for row in main_table:
        measured = [r for r in profile_rows if r['module'] == row['module'] and r['warmup'] == 'False']
        row['latency_seconds_mean'] = sum(float(r['end_to_end_seconds']) for r in measured) / len(measured) if measured else None
        row['peak_allocated_bytes'] = max(int(r['cuda_peak_allocated_bytes']) for r in measured) if measured else None
        row['timed_forwards'] = len(measured)
        control = {'Q_P': 'Q_A0', 'M_N': 'M_C'}.get(row['module'])
        comparison = next((r for r in paired if r['comparison'] == f"{row['module']}-{control}"
                           and r['role'] == 'SEL' and r['kind'] == 'own_CAL_selected_points'), None)
        row['mechanism_control'] = control
        row['mechanism_delta_T2_selected'] = comparison['delta_T2'] if comparison else None
    write_csv(c.artifacts / 'evaluation/MAIN_TABLE.csv', main_table)
    shortlist = {'confirmed':[r for r in table if r['evidence_level']=='DEVELOPMENT_REPLICATED'],
                 'provisional':[r for r in table if r['classification'] in POSITIVE and r['evidence_level']!='DEVELOPMENT_REPLICATED'],
                 'negative_or_tradeoff':[r for r in table if r['classification'] not in POSITIVE],
                 'all_rows':table, 'formal_test_claim':False, 'combinations_allowed':False,
                 'previous_SEL_exposure':True}
    write_json(c.artifacts / 'selection/SHORTLIST.json', shortlist)
    return shortlist


def verify_metric_states(context):
    c = context
    points = {}
    for name in ('CAL_ALL','SEL_LOCKED','SEL_ENDPOINT_1500','SEED46_FIXED'):
        collection = read_json(c.artifacts / f'evaluation/{name}.json')
        rows = collection['rows'] + ([collection['baseline']] if 'baseline' in collection else [])
        points.update({r['result_path']:r for r in rows})
    verified = []
    for row in points.values():
        for item in row['evidence']:
            state = read_json(Path(item['path']))
            if file_digest(item['path']) != item['sha256']:
                raise ValueError('metric evidence file changed')
            h = item['H']
            expected = row['expected'].get(str(h), row['expected'].get(h))
            actual = recompute_official_metric_evidence(state)['raw_local_AP' if h==1 else 'online_t-mAP']
            if state['updates'] != expected or abs(actual-row['metrics'][f'T{h}']) > 1e-8:
                raise ValueError('independent official metric/count mismatch')
            verified.append({'result':row['result_path'], 'H':h, 'recomputed':actual, 'updates':expected, **item})
    result = {'status':'PASS', 'unique_complete_results':len(points), 'metric_states':len(verified), 'rows':verified}
    write_json(c.artifacts / 'evaluation/VERIFIED_METRIC_STATES.json', result)
    return result
