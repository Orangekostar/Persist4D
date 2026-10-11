"""Run locked formal evaluations under the original worker lock and GPU ledger."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from scripts.native_long_budget import (
    ledger_events,
    remaining_budget,
    write_json_atomic,
)


def read(path):
    return json.loads(path.read_text())


def process_alive(pid):
    try:
        return bool(pid and (Path('/proc') / str(pid) / 'cmdline').read_bytes())
    except OSError:
        return False


def probe(root):
    directory = root / 'postselection'
    state = read(directory / 'STATE.json') if (directory / 'STATE.json').exists() else None
    active = read(root / 'ACTIVE_PROCESS.json') if (root / 'ACTIVE_PROCESS.json').exists() else None
    results = []
    for path in sorted((directory / 'evaluation').glob('*/*/*-?????.json')):
        row = read(path)
        results.append({key: row.get(key) for key in
                        ('arm', 'step', 'role', 'status', 'completed', 'expected', 'elapsed_seconds')})
    events = ledger_events(root / 'COST_LEDGER.jsonl')
    used = sum(row['gpu_hours'] for row in events)
    live = (max(0., time.time() - active['started_unix']) * active['cards'] / 3600
            if active and active['event_id'] not in {r['event_id'] for r in events} else 0.)
    return {'time_unix': time.time(), 'state': state, 'active': active,
            'worker_alive': process_alive((state or {}).get('pid')),
            'active_alive': process_alive((active or {}).get('pid')),
            'accounted_gpu_hours': used, 'live_gpu_hours': live, 'results': results,
            'formal_events': [r for r in events if r['phase'].startswith('formal:')]}


def run(root, data_root):
    from scripts.native_long_cluster import exclusive
    from scripts.native_long_execution import charged_process

    directory = root / 'postselection'
    with exclusive(root / 'worker.lock'):
        if (root / 'ACTIVE_PROCESS.json').exists():
            raise RuntimeError('existing process reservation requires reconciliation')
        spec = read(root / 'WORKER_SPEC.json')
        lock = read(directory / 'FORMAL_LOCK.json')
        rows = [r for r in lock['rows'] if r['arm'] == spec['arm']]
        if not rows or lock['training_seed'] != spec['seed']:
            raise ValueError('worker is not selected for formal confirmation')
        plan = read(root / 'selection/BUDGET_LOCK.json')
        state = {'status': 'RUNNING', 'pid': os.getpid(), 'arm': spec['arm'],
                 'started_unix': time.time(), 'completed_jobs': [], 'queue_size': len(rows) * 3}
        write_json_atomic(directory / 'STATE.json', state)
        try:
            for row in rows:
                for role in ('PB', 'LOCAL-T2', 'ADDITIONAL'):
                    phase = f"formal:{row['arm']}:{row['step']}:{role}"
                    state.update(phase=phase, updated_unix=time.time())
                    write_json_atomic(directory / 'STATE.json', state)
                    # Leave an additional per-worker margin inside its existing
                    # quota. The controller's global reserve is separate.
                    allowance = remaining_budget(root, plan) - .1
                    if allowance <= 0:
                        raise RuntimeError('worker quota cannot fund formal evaluation')
                    charged_process(root, [sys.executable, '-m', 'scripts.native_long_confirmation_inference',
                        '--root', str(root), '--directory', str(directory), '--data-root', str(data_root),
                        '--arm', row['arm'], '--step', str(row['step']), '--role', role],
                        phase=phase, cards=1, max_gpu_hours=min(2., allowance))
                    state['completed_jobs'].append(phase)
            state.update(status='COMPLETE', phase=None, completed_unix=time.time())
        except BaseException as error:
            state.update(status='FAILED', error=repr(error), completed_unix=time.time())
            raise
        finally:
            write_json_atomic(directory / 'STATE.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    if not args.probe and args.data_root is None:
        parser.error('--data-root is required for execution')
    print(json.dumps(probe(args.root) if args.probe else run(args.root, args.data_root)))


if __name__ == '__main__':
    main()
