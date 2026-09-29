"""Thin, explicit-root orchestration for the fixed Q-P / M-N campaign."""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.qp_mn_binding import PROJECT, Context, bind, gpu_scope, reuse_q2
from scripts.short_module_identity import (
    assert_sources_current,
    file_digest,
    identity_digest,
    source_identity,
)
from scripts.short_module_screen import append_event, read_json, write_json

STAGES = ('bind', 'adapt-export', 'B0', 'train-pairs', 'CAL-lock', 'SEL',
          'replicate', 'diagnostics', 'probes', 'profile', 'report', 'publish')
STAGE_SOURCES = {
    'bind': ('qp_mn_binding.py',), 'adapt-export': ('qp_mn_binding.py',),
    'B0': ('qp_mn_evaluation.py',), 'train-pairs': ('qp_mn_training.py',),
    'CAL-lock': ('qp_mn_evaluation.py',), 'SEL': ('qp_mn_evaluation.py',),
    'replicate': ('qp_mn_results.py', 'qp_mn_training.py', 'qp_mn_evaluation.py'),
    'diagnostics': ('qp_mn_diagnostics.py',), 'probes': ('qp_mn_probes.py',),
    'profile': ('qp_mn_profile.py',), 'report': ('qp_mn_results.py', 'qp_mn_delivery.py'),
    'publish': ('qp_mn_publication.py',),
}


def receipt_valid(receipt, identity):
    return (receipt.get('identity') == identity and bool(receipt.get('outputs'))
            and all(Path(p).is_file() and file_digest(p) == digest for p, digest in receipt['outputs'].items()))


def stage_identity(c, stage):
    sources = [PROJECT / 'scripts' / name for name in STAGE_SOURCES[stage]]
    sources += [PROJECT / 'scripts/qp_mn_adapter.py', PROJECT / 'models/qp_mn_heads.py']
    identity = {'schema': 1, 'stage': stage, 'config': c.config,
                'root': str(c.root), 'artifacts': str(c.artifacts), 'sources': source_identity(sources),
                'train_index': identity_digest(c.train_index), 'eval_index': identity_digest(c.eval_index)}
    if stage not in ('bind', 'adapt-export', 'B0', 'train-pairs'):
        identity['checkpoints'] = {str(p): file_digest(p) for p in sorted((c.artifacts / 'training').glob('*/seed*/checkpoint_manifest.json'))}
    if stage in ('SEL', 'replicate', 'diagnostics', 'probes', 'profile', 'report', 'publish'):
        identity['cal_lock'] = file_digest(c.artifacts / 'selection/CAL_LOCK.json')
    if stage == 'probes':
        identity['diagnostics'] = file_digest(c.artifacts / 'diagnostics/DIAGNOSTICS.json')
    return identity


def record_stage(c, stage, identity, outputs, *, status='COMPLETE', error=None):
    receipt = {'stage': stage, 'status': status, 'utc': datetime.now(timezone.utc).isoformat(),
               'identity': identity, 'outputs': {str(p): file_digest(p) for p in outputs if Path(p).is_file()},
               'error': error}
    write_json(c.root / 'stages' / f'{stage}.json', receipt)
    if stage != 'publish':
        append_event(c.artifacts / 'EXECUTION_LOG.jsonl', {k: v for k, v in receipt.items() if k not in ('identity', 'outputs')})
    return receipt


def stage_outputs(c, stage):
    if stage == 'B0':
        return sorted((c.root / 'evaluation').glob('*/B0/seed45/CAL-0000.json'))
    if stage == 'publish':
        return [c.root / 'publication/PUBLICATION_RECEIPT.json']
    patterns = {
        'bind': ['BASE_BINDING.json', 'REPAIRED_CONTRACT.json', 'INPUT_MANIFEST.json', 'RESOLVED_CONFIG.json'],
        'adapt-export': ['training/Q2_F/seed45/checkpoint_manifest.json'],
        'B0': [], 'train-pairs': ['training/*/seed45/checkpoint_manifest.json', 'training/*/seed45/metrics.jsonl'],
        'CAL-lock': ['evaluation/CAL_ALL.json', 'selection/CAL_LOCK.json'],
        'SEL': ['evaluation/SEL_LOCKED.json', 'evaluation/SEL_ENDPOINT_1500.json'],
        'replicate': ['selection/REPLICATION_PLAN.json', 'evaluation/SEED46_FIXED.json'],
        'diagnostics': ['diagnostics/DIAGNOSTICS.json', 'diagnostics/*.csv', 'diagnostics/evidence/*.json'],
        'probes': ['diagnostics/PROBES.json', 'diagnostics/probes/*/*'],
        'profile': ['resources/PROFILE*.json', 'resources/PROFILE.csv', 'resources/ZERO_STEP_NATIVE_PARITY.json', 'training/*/seed45/deployment.pt'],
        'report': ['evaluation/MAIN_TABLE.csv', 'selection/SHORTLIST.json', 'evidence/*', 'training/PORTABLE_CHECKPOINTS.json'],
        'publish': [],
    }
    return sorted({p for pattern in patterns[stage] for p in c.artifacts.glob(pattern) if p.is_file()})


def _verify_training(c, arm, seed, points):
    from scripts.qp_mn_evaluation import load_head
    manifest_path = c.artifacts / 'training' / arm / f'seed{seed}/checkpoint_manifest.json'
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if not set(points) <= {p['step'] for p in manifest['checkpoints']}:
        return False
    for point in points:
        load_head(c, arm, point, seed, 'cpu')
    return True


def _existing_probe(c):
    path = c.artifacts / 'diagnostics/PROBES.json'
    if not path.exists():
        return False
    result = read_json(path)
    diagnostics = read_json(c.artifacts / 'diagnostics/DIAGNOSTICS.json')
    for name, trigger in [('Q_MEMORIZATION', diagnostics['q_memorization_trigger']),
                          ('M_FIT_DIAGNOSTIC', diagnostics['m_fit_trigger'])]:
        row = result[name]
        if not trigger:
            if row['status'] != 'NOT_APPLICABLE':
                raise ValueError('probe trigger differs; do not silently reuse')
            continue
        assert_sources_current(row)
        if row['status'] != 'DIAGNOSTIC_ONLY' or row['seed'] != 145 or row['label_identity_sha256'] != c.train_index['label_identity_sha256']:
            raise ValueError('probe identity differs')
        if name == 'M_FIT_DIAGNOSTIC':
            expected = [(r['input_id'], r['retained_index']) for r in diagnostics['m_fit_candidates']]
            observed = [(r['input_id'], r['candidates'][0]) for r in row['fixed_candidates']]
            if observed != expected or row['updates'] != 500:
                raise ValueError('probe fixed candidates or horizon differ')
        for checkpoint in row['checkpoints']:
            if file_digest(checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('probe checkpoint changed')
    return True


def execute_stage(c, stage, *, device, base_commit):
    from scripts.qp_mn_evaluation import evaluate_cal, evaluate_point, evaluate_sel
    if stage == 'bind':
        bind(c, base_commit=base_commit)
    elif stage == 'adapt-export':
        reuse_q2(c)
    elif stage == 'B0':
        evaluate_point(c, module='B0', step=0, role='CAL')
    elif stage == 'train-pairs':
        from scripts.qp_mn_training import train_arm
        from scripts.short_module_training import load_training_records
        pending = [arm for arm in ('Q_A0', 'Q_P', 'M_C', 'M_N') if not _verify_training(c, arm, 45, (0, 500, 1000, 1500))]
        if pending:
            records = load_training_records(c.root, c.train_index)
            with gpu_scope(c, device=device, stage='train-pairs'):
                for arm in pending:
                    train_arm(c, arm, seed=45, updates=1500, device=device, records=records)
    elif stage == 'CAL-lock':
        evaluate_cal(c)
    elif stage == 'SEL':
        evaluate_sel(c)
    elif stage == 'replicate':
        from scripts.qp_mn_results import replicate
        replicate(c, device=device)
    elif stage == 'diagnostics':
        from scripts.qp_mn_diagnostics import run_diagnostics
        run_diagnostics(c)
    elif stage == 'probes':
        from scripts.qp_mn_probes import run_probes
        if not _existing_probe(c):
            run_probes(c, device=device)
    elif stage == 'profile':
        from scripts.qp_mn_profile import profile
        with gpu_scope(c, device=device, stage='profile'):
            profile(c, device=device)
    elif stage == 'report':
        from scripts.qp_mn_delivery import package_heads, package_metric_evidence
        from scripts.qp_mn_results import comparison_tables, verify_metric_states
        comparison_tables(c)
        verify_metric_states(c)
        package_heads(c)
        package_metric_evidence(c)
    elif stage == 'publish':
        from scripts.qp_mn_publication import publish
        publish(c)
    return stage_outputs(c, stage)


def status(c):
    stages = {}
    for stage in STAGES:
        path = c.root / 'stages' / f'{stage}.json'
        if not path.exists():
            stages[stage] = 'UNVERIFIED'
            continue
        receipt = read_json(path)
        try:
            valid = receipt_valid(receipt, stage_identity(c, stage))
        except (FileNotFoundError, ValueError):
            valid = False
        stages[stage] = receipt['status'] if valid else 'STALE_OR_INCOMPLETE'
    result = {'root': str(c.root), 'artifacts': str(c.artifacts), 'stages': stages,
              'scientific_status': 'EXPERIMENTS_COMPLETE_REVIEW_REQUIRED' if all(stages[s] == 'COMPLETE' for s in STAGES[:-1]) else 'PARTIAL',
              'publication_receipt': str(c.root / 'publication/PUBLICATION_RECEIPT.json')}
    review_path = c.artifacts / 'REQUIREMENT_REVIEW.json'
    if result['scientific_status'] != 'PARTIAL' and review_path.exists():
        review = read_json(review_path)
        if (review.get('scientific_review_status') == 'PASS' and review.get('evidence')
                and all(Path(p).is_file() and file_digest(p) == digest for p, digest in review['evidence'].items())):
            result['scientific_status'] = review['scientific_status']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'status', 'report', 'publish'))
    parser.add_argument('--config', default=str(PROJECT / 'configs/qp_mn_targeted_v1.yaml'))
    parser.add_argument('--root')
    parser.add_argument('--artifact-root')
    parser.add_argument('--base-commit', default='465f37a')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--through', choices=STAGES, default='publish')
    parser.add_argument('--_stage', choices=STAGES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    c = Context.load(args.config, root=args.root, artifact_root=args.artifact_root)
    if args.command == 'status':
        import json
        print(json.dumps(status(c), indent=2))
        return
    if args._stage:
        import torch
        torch.set_num_threads(2)
        identity = stage_identity(c, args._stage)
        try:
            outputs = execute_stage(c, args._stage, device=args.device, base_commit=args.base_commit)
            record_stage(c, args._stage, stage_identity(c, args._stage), outputs)
        except Exception as error:
            record_stage(c, args._stage, identity, [], status='FAILED', error=str(error))
            raise
        return
    stages = STAGES[:STAGES.index(args.through) + 1] if args.command == 'run' else (args.command,)
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for stage in stages:
        # Core stages always inspect their own live identities. Only expensive
        # fixed diagnostics/profile reuse a receipt bound to inputs and outputs.
        receipt_path = c.root / 'stages' / f'{stage}.json'
        if (args.resume and stage in ('diagnostics', 'profile') and receipt_path.exists()
                and receipt_valid(read_json(receipt_path), stage_identity(c, stage))):
            continue
        command = [sys.executable, '-m', 'scripts.qp_mn_campaign', 'run', '--_stage', stage,
                   '--config', str(Path(args.config).resolve()), '--root', str(c.root),
                   '--artifact-root', str(c.artifacts), '--base-commit', args.base_commit, '--device', args.device]
        completed = subprocess.run(command, cwd=PROJECT, env=env, check=False)
        if stage != 'publish':
            write_json(c.artifacts / 'RUN_STATE.json', status(c))
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
