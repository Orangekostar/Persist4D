"""Head-only training with immutable source/input identity and paired sample plans."""
import json
import random
import time

import numpy as np
import torch

from models.qp_mn_heads import Module, build_head
from scripts.qp_mn_adapter import batch_loss
from scripts.qp_mn_binding import PROJECT
from scripts.short_module_identity import file_digest, identity_digest, source_identity
from scripts.short_module_screen import append_event, read_json, write_json
from scripts.short_module_training import (
    build_sample_plan,
    load_training_records,
    parameter_digest,
)
from trainer.perception_gain_trainer import perception_lr_multiplier


def training_identity(context, module, seed, plan):
    sources = [PROJECT / p for p in (
        'models/qp_mn_heads.py', 'models/short_module_heads.py', 'scripts/qp_mn_adapter.py',
        'scripts/qp_mn_training.py', 'scripts/short_module_training.py',
        'scripts/short_module_data.py', 'trainer/perception_gain_trainer.py')]
    return {'schema': 'qp-mn-training-v1', 'module': Module(module).value, 'seed': seed,
            'parent_sha256': context.config['parent_sha256'], 'dimensions': context.dimensions,
            'input_index_sha256': identity_digest(context.train_index),
            'label_identity_sha256': context.train_index['label_identity_sha256'],
            'sample_plan_sha256': identity_digest(plan), 'sources': source_identity(sources),
            'training': context.config['training']}


def save_checkpoint(path, head, optimizer, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                'optimizer': optimizer.state_dict(), 'metadata': metadata,
                'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                'python_rng': random.getstate(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}, temporary)
    temporary.replace(path)


def train_arm(context, module, *, seed, updates, device, records=None):
    c, module = context, Module(module)
    if module is Module.B0 or seed not in (45, 46) or not 1 <= updates <= 1500:
        raise ValueError('invalid targeted training request')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    cfg = c.config['training']
    if cfg['schedule_horizon'] != 1500:
        raise ValueError('scheduler horizon must remain 1500 even for locked replication')
    counts = {e['input_id']: e['audit']['candidate_count'] for e in c.train_index['entries']}
    plan = build_sample_plan(c.population, counts, seed=seed)
    plan_path = c.root / 'training' / f'sample_plan_seed{seed}.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('paired sample plan changed')
    write_json(plan_path, plan)
    identity = training_identity(c, module, seed, plan)
    directory = c.root / 'training' / module.value / f'seed{seed}'
    public = c.artifacts / 'training' / module.value / f'seed{seed}'
    head = build_head(module, **c.dimensions, seed=seed, thresholds=tuple(c.resolved['official_thresholds']))
    initial = parameter_digest(head)
    head.to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                                  eps=cfg['epsilon'], weight_decay=cfg['weight_decay'])
    metadata = {'module': module.value, 'seed': seed, 'dimensions': c.dimensions,
                'base_sha256': c.config['parent_sha256'], 'thresholds': c.resolved['official_thresholds'],
                'initial_parameter_sha256': initial, 'sample_plan_sha256': identity_digest(plan),
                'training_identity': identity, 'optimizer_updates': 0, 'schedule_horizon': 1500,
                'label_identity_sha256': c.train_index['label_identity_sha256'],
                'geometry_policy': 'frozen-v1', 'supervised_gradient_checks': [],
                'parent_loaded_in_training': False, 'parameter_count': sum(p.numel() for p in head.parameters()),
                'updates_reused': 0, 'updates_new': 0}
    last = directory / 'last.pt'
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        if saved['metadata']['training_identity'] != identity:
            raise ValueError('resume identity differs from current sources/inputs/loss')
        metadata = saved['metadata']
        if metadata['optimizer_updates'] > updates:
            raise ValueError('cannot resume backwards to an earlier stop')
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if saved['cuda_rng'] and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    else:
        save_checkpoint(directory / 'update=0000.pt', head, optimizer, metadata)
    write_json(public / 'resolved_config.json', {'identity': identity, 'initial_parameter_sha256': initial})
    records = records if records is not None else load_training_records(c.root, c.train_index)
    start = metadata['optimizer_updates']
    started, window, window_count = time.perf_counter(), {}, 0
    for offset in range(start, updates):
        lr = cfg['learning_rate'] * perception_lr_multiplier(offset, total_updates=1500,
                    warmup_updates=cfg['warmup_updates'], minimum_fraction=cfg['minimum_lr_fraction'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, detail, _ = batch_loss(module, head, records, plan[offset], classes=c.dimensions['classes'], device=device)
        if not torch.isfinite(loss):
            raise ValueError(f'nonfinite training loss at update {offset + 1}')
        check = detail['Npos'] > 0 and len(metadata['supervised_gradient_checks']) < 2
        before = {k: v.detach().clone() for k, v in head.named_parameters()} if check else {}
        loss.backward()
        trunk_norm = sum(float(p.grad.square().sum()) for p in head.trunk.parameters() if p.grad is not None) ** .5 if check else None
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        if check:
            changed = {k: not torch.equal(before[k], p) for k, p in head.named_parameters()}
            if not any(changed.values()) or norm.item() <= 0:
                raise ValueError('supervised batch failed to update head')
            metadata['supervised_gradient_checks'].append({'update': offset + 1,
                'preclip_gradient_norm': norm.item(), 'trunk_gradient_norm': trunk_norm,
                'changed_parameters': changed, 'parent_in_optimizer': False,
                'trunk_weight_decay_can_change_parameters_even_when_initial_gradient_zero': True})
        metadata['optimizer_updates'] = offset + 1
        metadata['updates_new'] = offset + 1
        if offset + 1 == 20:
            metadata['first20_seconds'] = time.perf_counter() - started
            metadata['forecast_1500_seconds'] = metadata['first20_seconds'] * 75
        detail.update(loss=loss.detach().item(), preclip_gradient_norm=norm.item(),
                      clipped_updates=int(norm.item() > cfg['clip_norm']))
        for key, value in detail.items():
            window[key] = window.get(key, 0) + value
        window_count += 1
        if (offset + 1) % cfg['last_every'] == 0 or offset + 1 == updates:
            row = {'update': offset + 1, 'count_updates': window_count,
                   'sums': window, 'lr': lr, 'elapsed_seconds': time.perf_counter() - started}
            append_event(public / 'metrics.jsonl', row)
            save_checkpoint(last, head, optimizer, metadata)
            print(json.dumps({'module': module.value, 'seed': seed, **row}), flush=True)
            window, window_count = {}, 0
        if offset + 1 in cfg['checkpoints'] or offset + 1 == updates:
            save_checkpoint(directory / f'update={offset + 1:04d}.pt', head, optimizer, metadata)
    checkpoints = [{'step': int(p.stem.split('=')[1]), 'path': str(p),
                    'sha256': file_digest(p), 'bytes': p.stat().st_size}
                   for p in sorted(directory.glob('update=*.pt'))]
    manifest = {**metadata, 'checkpoints': checkpoints, 'status': 'COMPLETE',
                'training_seconds_this_call': time.perf_counter() - started}
    write_json(public / 'checkpoint_manifest.json', manifest)
    return manifest
