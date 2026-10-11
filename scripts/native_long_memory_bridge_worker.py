"""Run the fixed trained-memory bridge after this node's formal evaluation."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.native_long_budget import remaining_budget, write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def execute(root, data_root, state):
    from scripts.native_long_execution import charged_process

    parent = root / 'postselection'
    project = Path(__file__).resolve().parents[1]
    plan = read_json(parent / 'BRIDGE_PLAN.json')
    preflight = read_json(parent / 'BRIDGE_CPU_ALIGNMENT.json')
    if (sha256(parent / 'BRIDGE_CPU_ALIGNMENT.json') != plan['cpu_alignment_sha256']
            or preflight['status'] != 'PASS' or preflight['units'] != 129 or preflight['checks'] != 645
            or preflight['bridge_sha256'] != plan['code']['scripts/native_long_memory_bridge.py']):
        raise ValueError('bridge lacks the bound complete CPU point/truth alignment audit')
    if read_json(root / 'WORKER_SPEC.json')['arm'] != 'E3':
        raise ValueError('bridge is assigned only to the E3 worker quota')
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('bridge code/config SHA differs: ' + name)
    if (root / 'ACTIVE_PROCESS.json').exists():
        raise RuntimeError('unreconciled GPU reservation')
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=15)
    if query.stdout.strip():
        raise RuntimeError('bridge GPU host is occupied: ' + query.stdout.strip())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=15)
    state.update(status='RUNNING', queue_size=len(plan['rows']), gpu_identity=gpu.stdout,
                 plan_sha256=sha256(parent / 'BRIDGE_PLAN.json'))
    for row in plan['rows']:
        name = f"{row['arm']}-{row['step']:05d}"
        bundle = parent / 'bridge-assets' / row['bundle_name']
        if sha256(bundle) != row['bundle_sha256']:
            raise ValueError('bridge bundle SHA differs')
        state.update(phase='memory-bridge:' + name, updated_unix=time.time())
        write_json_atomic(parent / 'BRIDGE_STATE.json', state)
        allowance = remaining_budget(root, read_json(root / 'selection/BUDGET_LOCK.json')) - .1
        if allowance <= 0:
            raise RuntimeError('worker quota cannot fund memory bridge')
        destination = parent / 'memory-bridge' / name
        charged_process(root, [sys.executable, '-m', 'scripts.native_long_memory_bridge',
            '--root', str(root), '--data-root', str(data_root), '--bundle', str(bundle),
            '--directory', str(destination)], phase=state['phase'], cards=1,
            max_gpu_hours=min(2., allowance))
        result = read_json(destination / 'RESULT.json')
        if result['status'] != 'COMPLETE' or result['completed_units'] != 129:
            raise RuntimeError('memory bridge returned incomplete evidence')
        state['completed_jobs'].append(name)
        write_json_atomic(parent / 'BRIDGE_STATE.json', state)
    state.update(status='COMPLETE', phase=None, completed_unix=time.time())


def run(root, data_root):
    from scripts.native_long_cluster import LockBusyError, exclusive

    parent = root / 'postselection'
    if (parent / 'BRIDGE_STATE.json').exists():
        raise RuntimeError('bridge queue already exists; inspect before retrying')
    state = {'status': 'WAITING_FOR_FORMAL', 'pid': os.getpid(), 'started_unix': time.time(),
             'completed_jobs': []}
    write_json_atomic(parent / 'BRIDGE_STATE.json', state)
    deadline = time.monotonic() + 6 * 3600
    try:
        while True:
            formal = read_json(parent / 'STATE.json')
            if formal['status'] == 'FAILED':
                raise RuntimeError('formal evaluation failed; memory bridge remains stopped')
            if formal['status'] == 'COMPLETE':
                try:
                    with exclusive(root / 'worker.lock'):
                        execute(root, data_root, state)
                    break
                except LockBusyError:
                    pass
            if time.monotonic() >= deadline:
                raise TimeoutError('formal evaluation did not release bridge host within six hours')
            time.sleep(30)
    except BaseException as error:
        state.update(status='FAILED', error=repr(error), completed_unix=time.time())
        raise
    finally:
        write_json_atomic(parent / 'BRIDGE_STATE.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.data_root)
