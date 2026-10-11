"""Bounded repeats of the two originally confounded CAL inputs, preserving originals."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.native_long_budget import (
    ledger_events,
    merge_events,
    remaining_budget,
    write_json_atomic,
)
from scripts.native_long_postselection import read_json, sha256


def validate_plan(plan, original):
    expected = {(i, mode, repeat) for i in (21, 25) for mode in ('normal', 'off') for repeat in (1, 2)}
    actual = {(j['record_index'], j['mode'], j['repeat']) for j in plan['jobs']}
    if (len(plan['jobs']) != 8 or actual != expected or len({j['name'] for j in plan['jobs']}) != 8
            or plan['total_ceiling_gpu_hours'] != .3 or plan['records'] != original['records']):
        raise ValueError('repeat coverage, ceiling or population differs')
    for key in ('data_files', 'population_sha256', 'metric_spec_sha256', 'cpu_preflight_sha256'):
        if plan[key] != original[key]:
            raise ValueError('repeat changes original data/proof binding: ' + key)
    if any(plan['code'].get(k) != v for k, v in original['code'].items()):
        raise ValueError('repeat changes original numerical source')
    for job in plan['jobs']:
        source = next(j for j in original['jobs'] if j['name'] == job['original_job'])
        if (job['arm'], job['step']) != ('E3', 2970) or any(
                job.get(k) != v for k, v in source.items() if k != 'name'):
            raise ValueError('repeat changes original checkpoint or intervention')


def run(root):
    from scripts.native_long_cluster import exclusive
    from scripts.native_long_execution import charged_process

    parent = root / 'postselection'
    replay_root = parent / 'mechanism-replay-v1'
    replay_parent = replay_root / 'postselection'
    state_path = parent / 'MECHANISM_REPLAY_STATE.json'
    if state_path.exists():
        raise RuntimeError('repeat queue exists; refusing to overwrite')
    plan_path = replay_parent / 'MECHANISM_PLAN.json'
    plan = read_json(plan_path)
    validate_plan(plan, read_json(parent / 'MECHANISM_PLAN.json'))
    if sha256(parent / 'MECHANISM_PLAN.json') != plan['parent_plan_sha256']:
        raise ValueError('original mechanism plan differs')
    state = {'status': 'STARTING', 'pid': os.getpid(), 'started_unix': time.time(),
             'plan_sha256': sha256(plan_path), 'completed_jobs': [], 'queue_size': 8}
    write_json_atomic(state_path, state)
    try:
        with exclusive(root / 'worker.lock'):
            if (read_json(root / 'WORKER_SPEC.json')['arm'] != 'E0'
                    or any(read_json(parent / n)['status'] != 'COMPLETE'
                           for n in ('STATE.json', 'RUNTIME_STATE.json', 'MECHANISM_STATE.json'))
                    or (root / 'ACTIVE_PROCESS.json').exists()):
                raise RuntimeError('repeat requires released E0 worker')
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True, timeout=15).strip():
                raise RuntimeError('repeat GPU host occupied')
            previous = {e['event_id'] for e in ledger_events(root / 'COST_LEDGER.jsonl')}
            for job in plan['jobs']:
                events = merge_events([ledger_events(root / 'COST_LEDGER.jsonl')])
                spent = sum(e['gpu_hours'] for e in events if e['event_id'] not in previous)
                allowance = min(.3 - spent,
                    remaining_budget(root, read_json(root / 'selection/BUDGET_LOCK.json')) - .1)
                if allowance <= 0:
                    raise RuntimeError('repeat bounded allowance exhausted')
                state.update(status='RUNNING', phase='mechanism:' + job['name'],
                             updated_unix=time.time(), billed_gpu_hours=spent)
                write_json_atomic(state_path, state)
                charged_process(root, [sys.executable, '-m', 'scripts.native_long_mechanism',
                    '--root', str(replay_root), '--job', job['name']], phase=state['phase'], cards=1,
                    max_gpu_hours=min(.15, allowance))
                result = read_json(replay_parent / 'mechanism' / (job['name'] + '.json'))
                if (result['status'] != 'COMPLETE' or result['job'] != job
                        or result['plan_sha256'] != state['plan_sha256']):
                    raise ValueError('repeat returned inconsistent evidence')
                state['completed_jobs'].append(job['name'])
                write_json_atomic(state_path, state)
            state.update(status='COMPLETE', phase=None, completed_unix=time.time())
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), completed_unix=time.time())
        raise
    finally:
        write_json_atomic(state_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(run(parser.parse_args().root)['status'])
