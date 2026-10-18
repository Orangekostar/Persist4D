"""Evidence-bound common-N selection and the predeclared 44-job campaign."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import ARTIFACTS, BudgetJournal, digest, numeric_sources, read_json, sha256
from scripts.ogr_policy import (
    CONTROLS, VARIANTS, Point, evaluation_jobs, identity_weight, select_control,
    select_endpoint, select_n,
)


PARAMETERS = {'M0': 0, 'M4': 84224, 'R-TARGET': 84224, 'R-ID': 116993,
              'R-QFB': 182785, 'R-OBS': 182785, 'FH-COMMIT0': 0}


def journal_for(assets):
    return BudgetJournal(assets['run_root'], prior=assets['budget']['prior_gpu_hours'],
                         external_reserved=assets['budget']['external_reserved_gpu_hours'])


def training_costs(timed, n):
    costs = {variant: 1.25*(startup+n*duration)/3600.
             for variant, (startup, duration) in timed.items()}
    return dict(first=sum(costs.values()),
                replication=costs['R-OBS']+max(costs[variant] for variant in CONTROLS),
                by_variant=costs)


def evaluation_forecast(assets):
    """No CAL scores: structural timings and the exact input-stage census only."""
    root = Path(assets['run_root'])
    manifest = read_json(ARTIFACTS / 'DATA_MANIFEST.json')
    costs, bindings = {}, []
    for method in ('R-OBS', 'FH-COMMIT0'):
        for role in ('CAL', 'PB'):
            path = root / 'timing' / f'{method}-s246-u000000' / role / 'RESULTS.json'
            result = read_json(path)
            status = read_json(path.with_name('STATUS.json'))
            if (result['status'] != 'COMPLETE' or status['status'] != 'COMPLETE'
                    or result['request']['timing_only'] is not True or result['rows']
                    or len(result['episodes']) != 6):
                raise ValueError('six complete, score-free structural inputs are required')
            stages = [row['end_to_end_seconds'] for episode in result['episodes'] for row in episode['timings']]
            if not stages or any(not math.isfinite(value) or value <= 0 for value in stages):
                raise ValueError('actual stage timings must be positive and finite')
            import numpy as np
            p90 = float(np.percentile(stages, 90))
            # FH timing inputs have T3 prefixes; reserve linear point growth to T5.
            # This is an explicit conservative forecast, not a measured T5 latency.
            prefix_factor = 5./min(episode['horizon'] for episode in result['episodes']) if method == 'FH-COMMIT0' else 1.
            costs[method, role] = dict(p90_stage_seconds=p90, prefix_forecast_factor=prefix_factor,
                stage_seconds=p90*prefix_factor, startup_seconds=status['startup_seconds'])
            bindings.append(dict(method=method, role=role,
                path=path.relative_to(root).as_posix(), sha256=sha256(path), input_ids=result['request']['input_ids']))
    forecasts = {}
    for method in ('R-OBS', 'FH-COMMIT0'):
        for role in ('CAL', 'SEL', 'PB'):
            # SEL has no score-free panel of its own; use the slower measured panel.
            source = costs[method, role] if role in ('CAL', 'PB') else max(
                (costs[method, key] for key in ('CAL', 'PB')), key=lambda row: row['stage_seconds'])
            seconds = source['startup_seconds']+manifest['stage_forwards'][role]*source['stage_seconds']
            forecasts[method, role] = 1.5*seconds/3600.
    # 26 CAL, eight ordinary SEL/PB, plus the two full-prefix controls.
    total = (26*forecasts['R-OBS', 'CAL']+8*forecasts['R-OBS', 'SEL']
             +8*forecasts['R-OBS', 'PB']+forecasts['FH-COMMIT0', 'SEL']+forecasts['FH-COMMIT0', 'PB'])
    return dict(gpu_hours=total, factor=1.5, jobs=44,
        stage_count=26*197+9*279+9*1806,
        method_role_gpu_hours={f'{method}:{role}': value for (method, role), value in forecasts.items()},
        measured_costs={f'{method}:{role}': value for (method, role), value in costs.items()},
        timing_evidence=bindings,
        limitation='SEL uses slower CAL/PB timing; FH T3 timing is scaled for T5 prefix point growth; no AP was read')


def calibrated_identity_weight(assets):
    path = Path(assets['run_root']) / 'probe/R-ID-s246/TRAINING.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    selected = [row for row in rows if 13 <= row['step'] <= 20]
    task, identity = [], []
    for row in selected:
        for stage in row['stages']:
            probe = stage.get('gradient_probe')
            if probe is None:
                continue
            left, right = probe.get('task'), probe.get('identity')
            if (isinstance(left, (float, int)) and isinstance(right, (float, int))
                    and math.isfinite(left) and math.isfinite(right) and left > 0 and right > 0):
                task.append(float(left))
                identity.append(float(right))
    if task:
        left, right = statistics.median(task), statistics.median(identity)
        return dict(status='CALIBRATED_TRAIN_ONLY', lambda_id=identity_weight(left, right),
                    task_median=left, identity_median=right, valid_stage_pairs=len(task),
                    updates=[13, 14, 15, 16, 17, 18, 19, 20], evidence_sha256=sha256(path))
    return dict(status='UNCALIBRATED', lambda_id=.1, valid_stage_pairs=0,
        reason='No positive finite paired gradient norm in the fixed last eight TRAIN probe updates',
        evidence_sha256=sha256(path), requires_bounded_next_train_probe=True)


def verified_probe(assets, variant):
    path = Path(assets['run_root']) / 'probe' / f'{variant}-s246' / 'STATUS.json'
    status = read_json(path)
    identity = status.get('identity', {})
    if (status.get('status') != 'COMPLETE' or status.get('step') != 20 or status.get('mode') != 'probe'
            or status.get('variant') != variant or status.get('adapter_seed') != 246
            or identity.get('parent') != assets['hashes']['parent']
            or identity.get('data') != assets['hashes']['population']
            or identity.get('encoder') != assets['hashes']['encoder']
            or status.get('frozen_parent_unchanged') is not True):
        raise ValueError('completed probe identity or update denominator differs')
    from datasets.revisit_prompt_episode import RevisitDrawPlan
    population = read_json(assets['population_path'])
    generator = RevisitDrawPlan(population['references'], seed=246)
    records = [record for cursor in range(20) for record in generator.draw_update(cursor)]
    if identity.get('draw_sha256') != digest(records):
        raise ValueError('completed probe deterministic TRAIN draws differ')
    sources = numeric_sources(require_complete=False)
    differences = [name for name in sources if name.startswith(('models/', 'datasets/', 'trainer/'))
                   and identity['sources'].get(name) != sources[name]]
    if differences:
        reconciliation = read_json(ARTIFACTS / 'PROBE_SOURCE_RECONCILIATION.json')
        for name in differences:
            row = reconciliation['approved_nonformal_changes'].get(name)
            if not row or row['before_sha256'] != identity['sources'].get(name) or row['after_sha256'] != sources[name]:
                raise ValueError('probe numerical source changed without its exact bounded reconciliation: '+name)
    return status


def expected_evaluations(assets, plan, selection=None, replication=None):
    """Keep all 44 rows visible, including unbound and blocked work."""
    control = (selection or {}).get('control', {}).get('point', {}).get('variant')
    # A template value supplies enumeration only; it must never label an unbound control.
    n = plan['N'] or 400
    jobs = evaluation_jobs(n, control or 'R-TARGET')
    rows = []
    for index, job in enumerate(jobs):
        item = dict(job_id=index+1, **job, status='PENDING', actual_step=None)
        if job['seed'] == 247 and job['method'] != 'R-OBS' and control is None:
            item['method'] = 'CONTROL_STAR_UNBOUND'
        if plan['N'] is None:
            item.update(status='BUDGET_BLOCKED', reason='NO_FEASIBLE_COMMON_N')
            if job['method'] in VARIANTS:
                item['step'] = f'{job["step"]}/400*N' if isinstance(job['step'], int) else job['step']
        elif isinstance(job['step'], int):
            item['actual_step'] = job['step']
        else:
            locked = selection if job['seed'] == 246 else replication
            endpoint = (locked or {}).get('endpoints', {}).get(item['method'])
            if endpoint:
                item['actual_step'] = endpoint['point']['step']
        rows.append(item)
    result = dict(schema=1, expected_jobs=44, complete_jobs=0, jobs=rows,
                  N=plan['N'], control=control, conditional_confirm_jobs=[])
    write_json_atomic(ARTIFACTS / 'EXPECTED_EVALUATIONS.json', result)
    return result


def lock_plan(assets, config):
    path = ARTIFACTS / 'RUN_PLAN.json'
    sources = numeric_sources()
    if path.exists():
        plan = read_json(path)
        if plan['numeric_sources'] != sources or plan['data_manifest_sha256'] != sha256(ARTIFACTS / 'DATA_MANIFEST.json'):
            raise ValueError('frozen plan identity differs; never silently relock N or lambda')
        return plan
    root = Path(assets['run_root'])
    timed, probes = {}, []
    # Adding delivery modules does not invalidate measured model timings. Every
    # pre-existing numeric implementation used by each probe remains bound.
    numeric_paths = {name for name in sources if name.startswith(('models/', 'datasets/', 'trainer/'))}
    for variant in VARIANTS:
        status_path = root / 'probe' / f'{variant}-s246' / 'STATUS.json'
        status = verified_probe(assets, variant)
        identity = status['identity']
        if (status['status'] != 'COMPLETE' or status['step'] != 20 or status['mode'] != 'probe'
                or status['variant'] != variant or status['adapter_seed'] != 246
                or identity['parent'] != assets['hashes']['parent'] or identity['data'] != assets['hashes']['population']
                or status['frozen_parent_unchanged'] is not True):
            raise ValueError('all four actual, identity-bound 20-update probes are required')
        # The source-intervention metadata correction affects only diagnostics;
        # ordinary probe numerical behavior is unchanged and is recorded below.
        differences = [name for name in numeric_paths if identity['sources'].get(name) != sources[name]]
        # verified_probe already requires an exact original/final digest pair for
        # the bounded diagnostic-only correction; arbitrary model edits fail.
        timed[variant] = (status['startup_seconds'], status['timed_p90_seconds'])
        probes.append(dict(variant=variant, path=status_path.relative_to(root).as_posix(), sha256=sha256(status_path),
            nontraining_probe=True, source_changes=differences,
            source_change_scope='source permutation diagnostic and provenance only; inactive in probes' if differences else None))
    calibration = calibrated_identity_weight(assets)
    forecast = evaluation_forecast(assets)
    budget = journal_for(assets).export(ARTIFACTS)
    if budget['active_jobs']:
        raise ValueError('settle probe/timing reservations before locking the total cost forecast')
    # D, F and I totals include settled work once and reserve their remaining caps.
    nontraining = forecast['gpu_hours']+2.+4.+4.
    n = select_n(timed, available=budget['initial_available_gpu_hours'],
                 nontraining_forecast=nontraining, evaluation_forecast=forecast['gpu_hours'])
    projections = {str(value): training_costs(timed, value) for value in (1200, 800, 400)}
    from datasets.revisit_prompt_episode import RevisitDrawPlan
    population = read_json(assets['population_path'])
    draws = {}
    if n is not None:
        for seed in (246, 247):
            generator = RevisitDrawPlan(population['references'], seed=seed)
            records = [record for cursor in range(n) for record in generator.draw_update(cursor)]
            draw_path = root / 'draws' / f'seed-{seed}-N{n}.json'
            write_json_atomic(draw_path, records)
            draws[str(seed)] = dict(sha256=digest(records), file_sha256=sha256(draw_path),
                                    episodes=len(records), stage_forwards=14*n)
    plan = dict(schema=1, status='LOCKED' if n else 'NO_FEASIBLE_COMMON_N',
        experiment_status='PLANNED' if n else 'BUDGET_BLOCKED', N=n,
        lambda_id=calibration['lambda_id'], lambda_calibration=calibration,
        checkpoint_steps=[n//4, n//2, 3*n//4, n] if n else [],
        adapter_seeds=[246, 247], parent_seed=45, parent_sha256=assets['hashes']['parent'],
        data_manifest_sha256=sha256(ARTIFACTS / 'DATA_MANIFEST.json'), numeric_sources=sources,
        measured_training_costs={key: dict(startup_seconds=value[0], p90_update_seconds=value[1])
                                 for key, value in timed.items()},
        projections=projections, evaluation_forecast=forecast,
        nontraining_forecast_gpu_hours=nontraining, budget_at_lock=budget,
        probe_evidence=probes, draws=draws, CAL_scores_read_before_lock=False,
        remaining_nontraining_caps_included_once={'D': 2., 'F': 4., 'I': 4.},
        blocked_reasons=(['evaluation forecast exceeds the fixed 12 GPUh phase cap']
            if forecast['gpu_hours'] > 12 else ['no allowed N meets all phase and total caps']) if n is None else [])
    write_json_atomic(path, plan)
    expected_evaluations(assets, plan)
    return plan


def evaluation_folder(assets, variant, seed, step, role):
    return Path(assets['run_root']) / 'evaluation' / f'{variant}-s{seed}-u{step:06d}' / role


def verified_evaluation(assets, variant, seed, step, role, *, allow_missing=False):
    folder = evaluation_folder(assets, variant, seed, step, role)
    path = folder / 'RESULTS.json'
    if not path.is_file():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    result, status = read_json(path), read_json(folder / 'STATUS.json')
    records = read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records'][role]
    inputs = {record['input_id']: (record['reference_id'], record['horizon']) for record in records}
    episodes = {row['input_id']: (row['reference_id'], row['horizon']) for row in result['episodes']}
    request = result['request']
    if (result['status'] != 'COMPLETE' or status['status'] != 'COMPLETE'
            or (request['variant'], request['adapter_seed'], request['step'], request['role']) != (variant, seed, step, role)
            or request['timing_only'] or inputs != episodes or len(episodes) != len(result['episodes'])
            or result['numeric_sources'] != read_json(ARTIFACTS / 'RUN_PLAN.json')['numeric_sources']):
        raise ValueError('evaluation identity, source, or full input denominator differs')
    for row in result['rows']:
        if row['status'] != 'COMPLETE' or row['completed_inputs'] != row['expected_inputs']:
            raise ValueError('incomplete horizon cannot enter selection')
        for key in ('AP', 'current_AP'):
            if row[key] is None or not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError('undefined/nonfinite AP cannot enter selection')
    for row in (*result['references'], *result['official_evidence']):
        evidence = folder / row.get('evidence', row.get('path', ''))
        if sha256(evidence) != row['sha256']:
            raise ValueError('official sufficient-state bytes differ')
    return result


def point_from_evaluation(assets, variant, seed, step):
    result = verified_evaluation(assets, variant, seed, step, 'CAL')
    return Point(variant=variant, seed=seed, step=step,
        ap={row['horizon']: row['AP'] for row in result['rows']},
        current={row['horizon']: row['current_AP'] for row in result['rows']}, parameters=PARAMETERS[variant])


def validate_evaluation_request(args, assets):
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    if plan['status'] != 'LOCKED' or plan['numeric_sources'] != numeric_sources():
        raise ValueError('evaluation requires the unchanged frozen numerical plan')
    if args.role == 'CONFIRM':
        confirm = read_json(ARTIFACTS / 'CONFIRM_STATUS.json')
        if confirm['status'] != 'AVAILABLE':
            raise ValueError('CONFIRM exposure and scope are not independently verified')
    if args.variant == 'M0':
        if args.step != 0:
            raise ValueError('M0 has no adapted step')
    elif args.variant == 'M4':
        if args.step != 200 or args.seed != 146:
            raise ValueError('the named baseline is original M4 seed146 step200 only')
    elif args.variant == 'FH-COMMIT0':
        if args.step != 0 or args.role not in ('SEL', 'PB'):
            raise ValueError('FH-COMMIT0 is a fixed full-prefix SEL/PB control')
    elif args.step not in plan['checkpoint_steps'] or args.seed not in (246, 247):
        raise ValueError('unregistered checkpoint or adapter seed')
    if args.seed == 247 and args.variant in VARIANTS:
        first = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')
        if args.variant not in ('R-OBS', first['control']['point']['variant']):
            raise ValueError('replication is restricted to OBS and the CAL-locked control family')
    if args.role != 'CAL':
        lock_name = 'REPLICATION_LOCK.json' if args.seed == 247 and args.variant in VARIANTS else 'CAL_SELECTION_LOCK.json'
        selection = read_json(ARTIFACTS / lock_name)
        if selection['status'] != 'LOCKED':
            raise ValueError('all first-seed CAL selections and control must be locked before SEL/PB')
        if args.variant in VARIANTS and args.step != selection['endpoints'][args.variant]['point']['step']:
            raise ValueError('SEL/PB may evaluate only the independently CAL-selected step')
    return True


def run_evaluation(assets, variant, seed, step, role):
    if verified_evaluation(assets, variant, seed, step, role, allow_missing=True) is not None:
        return dict(status='REUSED_EXACT_COMPLETE', variant=variant, seed=seed, step=step, role=role)
    from scripts.ogr_campaign import charged_launch
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    method = 'FH-COMMIT0' if variant == 'FH-COMMIT0' else 'R-OBS'
    reserve = plan['evaluation_forecast']['method_role_gpu_hours'][f'{method}:{role}']
    return charged_launch(assets, job_key=f'eval:{variant}:s{seed}:u{step}:{role}', phase='E_EVALUATION',
        reservation=max(.05, reserve), worker_args=['--worker', 'evaluation', '--variant', variant,
            '--seed', str(seed), '--step', str(step), '--role', role])


def run_training_wave(assets, variants, seed):
    from scripts.ogr_campaign import charged_launch, idle_gpus
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    pending, results = [], []
    for variant in variants:
        folder = Path(assets['run_root']) / 'trajectories' / f'{variant}-s{seed}'
        status = read_json(folder / 'STATUS.json') if (folder / 'STATUS.json').is_file() else None
        if status:
            identity = status.get('identity', {})
            if (identity.get('sources') != plan['numeric_sources'] or identity.get('N') != plan['N']
                    or identity.get('lambda_id') != plan['lambda_id'] or identity.get('variant') != variant
                    or identity.get('adapter_seed') != seed or identity.get('draw_sha256') != plan['draws'][str(seed)]['sha256']):
                raise ValueError('existing formal trajectory differs from the locked training identity')
            if status['status'] == 'COMPLETE' and status['step'] == plan['N']:
                results.append(dict(status='REUSED_EXACT_COMPLETE', variant=variant, seed=seed))
                continue
        pending.append((variant, status))
    while pending:
        cards = idle_gpus()[:2]
        if not cards:
            raise RuntimeError('no idle authorized GPU for the next training wave')
        wave, pending = pending[:len(cards)], pending[len(cards):]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = []
            for (variant, status), card in zip(wave, cards):
                measured = plan['measured_training_costs'][variant]
                remaining = plan['N']-(status or {}).get('step', 0)
                reserve = 1.25*(measured['startup_seconds']+remaining*measured['p90_update_seconds'])/3600.
                arguments = ['--worker', 'train', '--mode', 'formal', '--variant', variant, '--seed', str(seed)]
                if status:
                    arguments.append('--resume')
                attempt = sum(event['job_key'].startswith(f'train:{variant}:s{seed}:')
                    for event in read_json(Path(assets['run_root']) / 'BUDGET_STATE.json')['events'])
                futures.append(executor.submit(charged_launch, assets,
                    job_key=f'train:{variant}:s{seed}:attempt{attempt}',
                    phase='T_FIRST' if seed == 246 else 'T_REPLICATION', reservation=reserve, gpu=card, worker_args=arguments))
            wave_result = [future.result() for future in futures]
        results.extend(wave_result)
        if any(row['exit_code'] for row in wave_result):
            raise RuntimeError('training stopped; preserve the exact optimizer boundary and resolve the reported error')
    return results


def lock_endpoints(assets, seed, variants):
    name = 'CAL_SELECTION_LOCK.json' if seed == 246 else 'REPLICATION_LOCK.json'
    path = ARTIFACTS / name
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    if path.exists():
        result = read_json(path)
        if result['run_plan_sha256'] != sha256(ARTIFACTS / 'RUN_PLAN.json'):
            raise ValueError('selection lock is bound to another plan')
        return result
    baseline = point_from_evaluation(assets, 'M0', 246, 0)
    selected = {variant: select_endpoint([point_from_evaluation(assets, variant, seed, step)
        for step in plan['checkpoint_steps']], baseline, n=plan['N'], variant=variant, seed=seed) for variant in variants}
    result = dict(schema=1, status='LOCKED', adapter_seed=seed,
        run_plan_sha256=sha256(ARTIFACTS / 'RUN_PLAN.json'),
        data_manifest_sha256=plan['data_manifest_sha256'],
        endpoints={variant: asdict(value) for variant, value in selected.items()},
        selection_role='CAL', test_scores_used=False, parent_seed=45)
    if seed == 246:
        result['control'] = asdict(select_control({variant: selected[variant] for variant in CONTROLS}))
    else:
        result['control_family'] = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')['control']['point']['variant']
    write_json_atomic(path, result)
    return result


def run_locked_campaign(assets, config):
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    if plan['status'] == 'NO_FEASIBLE_COMMON_N':
        expected_evaluations(assets, plan)
        return dict(status='BUDGET_BLOCKED', reason='NO_FEASIBLE_COMMON_N', formal_updates=0)
    if plan['status'] != 'LOCKED' or plan['numeric_sources'] != numeric_sources():
        raise ValueError('run requires an unchanged, frozen implementation')
    if plan['lambda_calibration']['status'] != 'CALIBRATED_TRAIN_ONLY':
        raise ValueError('resolve the bounded TRAIN-only identity gradient calibration before formal training')
    # The finite new-path numeric gate precedes any formal optimizer update.
    from scripts.ogr_diagnostics import compare_numeric_panel
    from scripts.ogr_campaign import charged_launch
    for replicate in (0, 1):
        panel_path = Path(assets['run_root']) / 'numeric/new-path' / f'process-{replicate}/RESULTS.json'
        if not panel_path.is_file():
            launched = charged_launch(assets, job_key=f'check:new-path-repeat{replicate}',
                phase='I_INTEGRATION_REPAIR', reservation=.3,
                worker_args=['--worker', 'diagnostics', '--panel', 'repeat', '--replicate', str(replicate)])
            if launched['exit_code']:
                raise RuntimeError('the finite new-path process failed before formal training')
        panel = read_json(panel_path)
        core = {name: value for name, value in plan['numeric_sources'].items()
                if name.startswith(('models/', 'datasets/', 'trainer/'))}
        if any(panel.get('numeric_sources', {}).get(name) != value for name, value in core.items()):
            raise ValueError('new-path numeric evidence used a different model, data or training source')
    if compare_numeric_panel(assets)['status'] != 'FINITE_PANEL_WITHIN_LIMIT':
        raise ValueError('the finite new-path numeric gate is confounded; no formal training may start')
    run_training_wave(assets, VARIANTS, 246)
    run_evaluation(assets, 'M0', 246, 0, 'CAL')
    run_evaluation(assets, 'M4', 146, 200, 'CAL')
    for variant in VARIANTS:
        for step in plan['checkpoint_steps']:
            run_evaluation(assets, variant, 246, step, 'CAL')
    first = lock_endpoints(assets, 246, VARIANTS)
    control = first['control']['point']['variant']
    # This is mandatory even when the first seed is scientifically negative.
    run_training_wave(assets, ('R-OBS', control), 247)
    for variant in ('R-OBS', control):
        for step in plan['checkpoint_steps']:
            run_evaluation(assets, variant, 247, step, 'CAL')
    replication = lock_endpoints(assets, 247, ('R-OBS', control))
    for seed, selection in ((246, first), (247, replication)):
        for variant, endpoint in selection['endpoints'].items():
            for role in ('SEL', 'PB'):
                run_evaluation(assets, variant, seed, endpoint['point']['step'], role)
    for variant, seed, step in (('M0', 246, 0), ('M4', 146, 200), ('FH-COMMIT0', 246, 0)):
        for role in ('SEL', 'PB'):
            run_evaluation(assets, variant, seed, step, role)
    from scripts.ogr_diagnostics import run_final_panels
    from scripts.ogr_statistics import compute_statistics
    run_final_panels(assets)
    statistics_result = compute_statistics(assets)
    expected_evaluations(assets, plan, first, replication)
    return dict(status='CORE_COMPLETE', statistics=statistics_result)
