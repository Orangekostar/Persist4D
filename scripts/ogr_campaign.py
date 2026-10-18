"""Budgeted OGR resolution, diagnostics, probes, locked training and evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import (
    ARTIFACTS, PROJECT, BudgetJournal, build_system, configure_runtime, digest,
    local_assets, numeric_sources, parent_tensor_digest, read_json, resolve_assets, sha256,
)


VARIANTS = ('R-TARGET', 'R-ID', 'R-QFB', 'R-OBS')


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def require_charged_worker():
    event_id = os.environ.get('OGR_CHARGED_EVENT_ID', '')
    if not event_id.startswith('ogr:'):
        raise ValueError('OGR GPU execution requires a charged campaign child')
    return event_id


def idle_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    return [int(row.split(',')[0]) for row in output.splitlines()
            if int(row.split(',')[1]) <= 64 and int(row.split(',')[2]) == 0]


def charged_launch(assets, *, job_key, phase, reservation, worker_args, gpu=None):
    journal = BudgetJournal(assets['run_root'], prior=assets['budget']['prior_gpu_hours'],
                            external_reserved=assets['budget']['external_reserved_gpu_hours'])
    active = {row['physical_gpu_index'] for row in journal.snapshot()['active_jobs']}
    available = [index for index in idle_gpus() if index not in active]
    if gpu is not None:
        available = [index for index in available if index == gpu]
    if not available:
        raise RuntimeError('no idle authorized physical GPU; no other process will be stopped')
    gpu = available[0]
    event = journal.reserve(job_key, phase, reservation, gpu=gpu)
    started = time.perf_counter()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OGR_CHARGED_EVENT_ID=event,
        OGR_DEADLINE_MONOTONIC=str(started + reservation * 3600 - 30),
        OGR_STARTED_MONOTONIC=str(started), PYTHONPATH=os.pathsep.join(assets['pythonpath']))
    log = Path(assets['run_root']) / 'logs' / (job_key.replace(':', '_').replace('/', '_') + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [assets['python'], '-u', '-m', 'scripts.ogr_campaign', *worker_args]
    process, code = None, 1
    try:
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            append_jsonl(Path(assets['run_root']) / 'PROCESS_EVENTS.jsonl',
                dict(event_id=event, job_key=job_key, pid=process.pid, physical_gpu_index=gpu,
                     started_unix=time.time(), log=log.relative_to(assets['run_root']).as_posix(),
                     worker_arguments=worker_args))
            try:
                code = process.wait(timeout=max(1., reservation * 3600 - 20))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait(timeout=5)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        journal.finish(event, seconds=time.perf_counter()-started, exit_code=code)
        journal.export(ARTIFACTS)
    return dict(job_key=job_key, event_id=event, exit_code=code, physical_gpu_index=gpu,
                status='COMPLETE' if code == 0 else 'FAILED')


def train_worker(args, assets):
    event = require_charged_worker()
    import numpy as np
    import torch
    from datasets.observation_grounded_episode import ObservationEpisodeLoader
    from datasets.revisit_prompt_episode import RevisitDrawPlan
    from scripts.revisit_prompt_assets import configuration_digest
    from trainer.observation_grounded_trainer import adapter_state, save_resume, restore_resume

    configure_runtime(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('OGR requires exactly one visible CUDA device')
    formal = args.mode == 'formal'
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json') if formal else None
    if formal and plan['status'] != 'LOCKED':
        raise ValueError('formal OGR training requires a locked RUN_PLAN')
    n = plan['N'] if formal else 20
    updates = n if formal else 2 if args.mode == 'integration' else 20
    suffix = f'-attempt{args.attempt}' if args.attempt else ''
    folder = Path(assets['run_root']) / ('trajectories' if formal else args.mode) / f'{args.variant}-s{args.seed}{suffix}'
    folder.mkdir(parents=True, exist_ok=True)
    status = dict(status='STARTING', variant=args.variant, adapter_seed=args.seed, mode=args.mode,
                  planned_updates=updates, step=0, event_id=event, started_unix=time.time())
    write_json_atomic(folder / 'STATUS.json', status)
    try:
        sources = numeric_sources(require_complete=formal)
        if formal and sources != plan['numeric_sources']:
            raise ValueError('formal source identity differs from frozen RUN_PLAN')
        system, config, audit = build_system(assets, variant=args.variant, seed=args.seed)
        before = parent_tensor_digest(system)
        initial = adapter_state(system)
        population = read_json(assets['population_path'])
        draws = RevisitDrawPlan(population['references'], seed=args.seed)
        records = [record for cursor in range(updates) for record in draws.draw_update(cursor)]
        draw_sha = digest(records)
        lambda_id = plan['lambda_id'] if formal else .1
        identity = dict(parent=assets['hashes']['parent'], encoder=assets['hashes']['encoder'],
            taxonomy=assets['hashes']['taxonomy'], data=assets['hashes']['population'],
            variant=args.variant, adapter_seed=args.seed, mode=args.mode, N=n, lambda_id=lambda_id,
            draw_sha256=draw_sha, sources=sources, config_sha256=configuration_digest(config, assets))
        system.to('cuda:0')
        optimizer, scheduler = system.make_optimizer(n)
        latest = folder / 'resume-latest.pt'
        step = 0
        if args.resume:
            step = restore_resume(latest, system, optimizer, scheduler, identity=identity)['step']
        elif latest.exists() or (folder / 'TRAINING.jsonl').exists():
            raise ValueError('trajectory already exists; exact resume is required')
        else:
            save_resume(latest, system, optimizer, scheduler, step=0, identity=identity)
        loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
        durations = []
        stopping = {'requested': False}
        old_handler = signal.signal(signal.SIGTERM, lambda *_: stopping.update(requested=True))
        startup = time.perf_counter()-float(os.environ['OGR_STARTED_MONOTONIC'])
        status.update(status='RUNNING', step=step, load_audit=audit, identity=identity, startup_seconds=startup,
                      numeric_policy='ogr-frozen-backbone-v4-derived-v1',
                      internal_attention_dtype='inherited Concerto FP16 FlashAttention where required')
        write_json_atomic(folder / 'STATUS.json', status)
        try:
            while step < updates:
                deadline = float(os.environ['OGR_DEADLINE_MONOTONIC'])
                allowance = 1.25 * max(durations, default=0.) + 20
                if stopping['requested'] or time.perf_counter() + allowance >= deadline:
                    break
                started = time.perf_counter()
                record = system.optimize_update(loader, records[step*4:(step+1)*4], optimizer, scheduler,
                    device='cuda:0', lambda_id=lambda_id,
                    gradient_probe=args.mode == 'probe' and args.variant == 'R-ID' and step >= 12,
                    capture_gradients=args.mode == 'integration' or step == 1)
                torch.cuda.synchronize()
                duration = time.perf_counter()-started
                durations.append(duration)
                step += 1
                append_jsonl(folder / 'TRAINING.jsonl', dict(**record, step=step, seconds=duration,
                    variant=args.variant, adapter_seed=args.seed, mode=args.mode, event_id=event))
                if not formal or step % 50 == 0 or step in plan['checkpoint_steps'] or step == updates:
                    save_resume(latest, system, optimizer, scheduler, step=step, identity=identity)
                if formal and step in plan['checkpoint_steps']:
                    system.save_delta(folder / f'delta-{step:06d}.pt', assets=assets, step=step,
                                       numeric_sources=sources, training_identity=identity)
                status.update(step=step, episode_count=4 * step, stage_count=14 * step,
                              updated_unix=time.time(), last_update_seconds=duration)
                write_json_atomic(folder / 'STATUS.json', status)
                print(json.dumps({key: status[key] for key in ('variant', 'mode', 'step', 'last_update_seconds')}), flush=True)
            save_resume(latest, system, optimizer, scheduler, step=step, identity=identity)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
        after = parent_tensor_digest(system)
        if before != after:
            raise ValueError('frozen parent parameter or buffer bytes changed')
        final = adapter_state(system)
        status.update(status='COMPLETE' if step == updates else 'STOPPED_AT_OPTIMIZER_BOUNDARY', step=step,
            frozen_parent_unchanged=True, parent_tensor_sha256=after, seconds_per_update=durations,
            trainable_tensor_changed={key: not torch.equal(initial[key], final[key]) for key in initial},
            timed_p90_seconds=float(np.percentile(durations[4:], 90)) if len(durations) > 4 else None,
            updated_unix=time.time(), formal_model=formal)
        write_json_atomic(folder / 'STATUS.json', status)
        return status
    except BaseException as error:
        status.update(status='FAILED', error_type=type(error).__name__, error=str(error), updated_unix=time.time())
        write_json_atomic(folder / 'STATUS.json', status)
        raise


def probe(assets, *, integration_only=False, attempt=0):
    root = Path(assets['run_root'])
    if integration_only:
        if attempt and not (root / 'repairs/NULL_PRED_CHANGES.json').is_file():
            raise ValueError('a specific recorded repair is required before the one integration retry')
        return charged_launch(assets, job_key='integration:R-OBS:s246' + (f':fixed-retry{attempt}' if attempt else ''),
            phase='I_INTEGRATION_REPAIR', reservation=.5,
            worker_args=['--worker', 'train', '--mode', 'integration', '--variant', 'R-OBS', '--attempt', str(attempt)])
    result, pending = [], []
    for variant in VARIANTS:
        status_path = root / 'probe' / f'{variant}-s246' / 'STATUS.json'
        if status_path.is_file() and read_json(status_path)['status'] == 'COMPLETE':
            from scripts.ogr_selection import verified_probe
            result.append(verified_probe(assets, variant))
            continue
        pending.append(variant)
    while pending:
        gpus = idle_gpus()[:2]
        if not gpus:
            raise RuntimeError('probe is waiting for an idle GPU')
        wave, pending = pending[:len(gpus)], pending[len(gpus):]
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [executor.submit(charged_launch, assets, job_key=f'probe:{variant}:s246',
                phase='I_INTEGRATION_REPAIR', reservation=1., gpu=gpu,
                worker_args=['--worker', 'train', '--mode', 'probe', '--variant', variant])
                for variant, gpu in zip(wave, gpus)]
            wave_result = [future.result() for future in futures]
        result.extend(wave_result)
        if any(row['status'] != 'COMPLETE' for row in wave_result):
            break
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=('resolve', 'diagnose', 'probe', 'lock', 'run', 'finalize'))
    parser.add_argument('--config', default='conf/observation_grounded_v1/default.yaml')
    parser.add_argument('--worker', choices=('train', 'evaluation', 'diagnostics'))
    parser.add_argument('--mode', choices=('integration', 'probe', 'formal'), default='formal')
    parser.add_argument('--variant', choices=(*VARIANTS, 'M0', 'M4', 'FH-COMMIT0'), default='R-OBS')
    parser.add_argument('--seed', type=int, choices=(146, 246, 247), default=246)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', type=int, choices=(0, 1), default=0,
                        help='one separately recorded attempt after a specific repair')
    parser.add_argument('--integration-only', action='store_true')
    parser.add_argument('--role', choices=('CAL', 'SEL', 'PB', 'CONFIRM'), default='CAL')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--timing-only', action='store_true', help='six structural inputs, no AP computation')
    parser.add_argument('--panel', choices=('p1', 'mechanism', 'profile', 'repeat', 'causality'), default='p1')
    parser.add_argument('--replicate', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    if args.seed == 146 and not (args.worker == 'evaluation' and args.variant == 'M4'):
        parser.error('seed 146 is reserved for the named original M4-200 evaluation baseline')
    if args.action is None and args.worker is None:
        parser.error('an action or a charged internal worker is required')
    import yaml
    config = yaml.safe_load((PROJECT / args.config).read_text())
    if args.action == 'resolve':
        assets = resolve_assets(config)
        result = {'status': 'ASSETS_AND_PRIOR_BOUND', 'budget': assets['budget']}
    else:
        assets = local_assets()
        if args.worker == 'train':
            result = train_worker(args, assets)
        elif args.worker == 'evaluation':
            from scripts.ogr_evaluation import evaluation_worker
            result = evaluation_worker(args, assets)
        elif args.worker == 'diagnostics' or args.action == 'diagnose':
            from scripts.ogr_diagnostics import diagnose
            result = diagnose(args, assets)
        elif args.action == 'probe':
            result = probe(assets, integration_only=args.integration_only, attempt=args.attempt)
        elif args.action == 'lock':
            from scripts.ogr_selection import lock_plan
            result = lock_plan(assets, config)
        elif args.action == 'run':
            from scripts.ogr_selection import run_locked_campaign
            result = run_locked_campaign(assets, config)
        elif args.action == 'finalize':
            from scripts.ogr_delivery import finalize
            result = finalize(assets, config)
    reported = ({key: result[key] for key in ('status', 'variant', 'mode', 'step') if key in result}
                if args.worker and isinstance(result, dict) else result)
    print(json.dumps(reported, sort_keys=True, allow_nan=False))
    return 75 if isinstance(result, dict) and result.get('status') == 'STOPPED_AT_OPTIMIZER_BOUNDARY' else 0


if __name__ == '__main__':
    raise SystemExit(main())
