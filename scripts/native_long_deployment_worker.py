"""Queue trained deployment checks after formal evaluation releases its worker."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.native_long_budget import remaining_budget, write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def execute(root, state):
    from scripts.native_long_execution import charged_process

    directory = root / 'postselection'
    plan = read_json(directory / 'RUNTIME_PLAN.json')
    formal = read_json(directory / 'FORMAL_LOCK.json')
    producer = Path(__file__).with_name('native_long_deployment_validation.py')
    if (sha256(producer) != plan['producer_sha256']
            or sha256(root / 'POPULATION.json') != plan['population_sha256']
            or sha256(directory / 'FORMAL_LOCK.json') != plan['formal_lock_sha256']
            or [(r['arm'], r['step']) for r in plan['rows']] != [(r['arm'], r['step']) for r in formal['rows']]):
        raise ValueError('runtime plan/code/population differs from the frozen formal rows')
    if (root / 'ACTIVE_PROCESS.json').exists():
        raise RuntimeError('unreconciled GPU reservation')
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=15)
    if query.stdout.strip():
        raise RuntimeError('runtime GPU host is occupied: ' + query.stdout.strip())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=15)
    state.update(status='RUNNING', gpu_identity=gpu.stdout, plan_sha256=sha256(directory / 'RUNTIME_PLAN.json'))
    jobs = []
    for row in plan['rows']:
        bundle = directory / 'runtime-assets' / row['bundle_name']
        receipt = read_json(bundle.with_suffix('.json'))
        if sha256(bundle) != row['sha256'] or receipt['checkpoint_sha256'] != row['checkpoint_sha256']:
            raise ValueError('runtime bundle or checkpoint binding differs')
        jobs.append((row, 'profile', None, 0))
    for row in plan['rows']:
        if row['arm'] in ('E1', 'E3') and row['step'] == 2970:
            for order in ('A', 'BA'):
                for repeat in range(2):
                    jobs.append((row, 'order', order, repeat))
    state['queue_size'] = len(jobs)
    for row, mode, order, repeat in jobs:
        name = f"{row['arm']}-{row['step']:05d}-{mode}-{order or 'CAL'}-{repeat}"
        output = directory / 'runtime' / (name + '.json')
        state.update(phase='postruntime:' + name, updated_unix=time.time())
        write_json_atomic(directory / 'RUNTIME_STATE.json', state)
        allowance = remaining_budget(root, read_json(root / 'selection/BUDGET_LOCK.json')) - .1
        if allowance <= 0:
            raise RuntimeError('worker quota cannot fund deployment validation')
        command = [sys.executable, '-m', 'scripts.native_long_deployment_validation', '--mode', mode,
                   '--root', str(root), '--bundle', str(directory / 'runtime-assets' / row['bundle_name']),
                   '--output', str(output)]
        if order:
            command += ['--order', order]
        charged_process(root, command, phase=state['phase'], cards=1, max_gpu_hours=min(1., allowance))
        result = read_json(output)
        if result['status'] != 'COMPLETE':
            raise RuntimeError('deployment process returned incomplete evidence')
        state['completed_jobs'].append(name)
        write_json_atomic(directory / 'RUNTIME_STATE.json', state)
    state.update(status='COMPLETE', phase=None, completed_unix=time.time())


def run(root):
    from scripts.native_long_cluster import LockBusyError, exclusive

    directory = root / 'postselection'
    if (directory / 'RUNTIME_STATE.json').exists():
        raise RuntimeError('runtime queue already exists; inspect before retrying')
    state = {'status': 'WAITING_FOR_FORMAL', 'pid': os.getpid(), 'started_unix': time.time(),
             'completed_jobs': []}
    write_json_atomic(directory / 'RUNTIME_STATE.json', state)
    deadline = time.monotonic() + 6 * 3600
    try:
        while True:
            formal_state = read_json(directory / 'STATE.json')
            if formal_state['status'] == 'FAILED':
                raise RuntimeError('formal evaluation failed; runtime queue remains stopped')
            if formal_state['status'] == 'COMPLETE':
                try:
                    with exclusive(root / 'worker.lock'):
                        execute(root, state)
                    break
                except LockBusyError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError('formal evaluation did not release the runtime host within six hours')
            time.sleep(30)
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), completed_unix=time.time())
        raise
    finally:
        write_json_atomic(directory / 'RUNTIME_STATE.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
