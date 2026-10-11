"""Bounded, charged queue of fresh-process CAL mechanism diagnostics."""

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


def run(root):
    from scripts.native_long_cluster import exclusive
    from scripts.native_long_execution import charged_process

    parent = root / 'postselection'
    state_path = parent / 'MECHANISM_STATE.json'
    if state_path.exists():
        raise RuntimeError('mechanism queue already exists; inspect before retrying')
    plan = read_json(parent / 'MECHANISM_PLAN.json')
    state = {'status': 'STARTING', 'pid': os.getpid(), 'started_unix': time.time(),
             'plan_sha256': sha256(parent / 'MECHANISM_PLAN.json'), 'completed_jobs': [],
             'queue_size': len(plan['jobs'])}
    write_json_atomic(state_path, state)
    try:
        with exclusive(root / 'worker.lock'):
            if (read_json(root / 'WORKER_SPEC.json')['arm'] != 'E0'
                    or read_json(parent / 'STATE.json')['status'] != 'COMPLETE'
                    or read_json(parent / 'RUNTIME_STATE.json')['status'] != 'COMPLETE'
                    or (root / 'ACTIVE_PROCESS.json').exists()):
                raise RuntimeError('mechanism requires the released E0 formal/runtime worker')
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'], text=True, timeout=15).strip():
                raise RuntimeError('mechanism GPU host has other compute processes')
            project = Path(__file__).resolve().parents[1]
            for name, expected in plan['code'].items():
                if sha256(project / name) != expected:
                    raise ValueError('mechanism source/config differs: ' + name)
            if len(plan['jobs']) != 152 or len({j['name'] for j in plan['jobs']}) != 152:
                raise ValueError('mechanism queue coverage differs')
            previous = {e['event_id'] for e in ledger_events(root / 'COST_LEDGER.jsonl')}
            state['gpu_identity'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True, timeout=15)
            for job in plan['jobs']:
                current = merge_events([ledger_events(root / 'COST_LEDGER.jsonl')])
                spent = sum(e['gpu_hours'] for e in current if e['event_id'] not in previous)
                allowance = min(plan['total_ceiling_gpu_hours'] - spent,
                                remaining_budget(root, read_json(root / 'selection/BUDGET_LOCK.json')) - .1)
                if allowance <= 0:
                    raise RuntimeError('mechanism bounded allowance exhausted')
                state.update(status='RUNNING', phase='mechanism:' + job['name'],
                             updated_unix=time.time(), billed_gpu_hours=spent)
                write_json_atomic(state_path, state)
                charged_process(root, [sys.executable, '-m', 'scripts.native_long_mechanism',
                    '--root', str(root), '--job', job['name']], phase=state['phase'], cards=1,
                    max_gpu_hours=min(.15, allowance))
                result = read_json(parent / 'mechanism' / (job['name'] + '.json'))
                if (result['status'] != 'COMPLETE' or result['job'] != job
                        or result['plan_sha256'] != state['plan_sha256']):
                    raise RuntimeError('mechanism returned incomplete or different evidence')
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
    result = run(parser.parse_args().root)
    print(result['status'])
