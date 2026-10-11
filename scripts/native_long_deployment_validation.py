"""Trained bundle roundtrips and fixed-CAL deployment measurements."""

import argparse
import json
import time
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def fixed_cal_inputs(population):
    records = population['evaluation_inputs']['CAL']
    if any(not r['input_id'].startswith('CAL:') for r in records):
        raise ValueError('runtime diagnostics must use CAL inputs')
    if {r['horizon'] for r in records} != set(range(1, 6)):
        raise ValueError('runtime diagnostics require all five horizons')
    return [min((r for r in records if r['horizon'] == h), key=lambda r: r['input_id'])
            for h in range(1, 6)]


def tensor_difference(left, right):
    import torch

    if left.shape != right.shape:
        return {'same_shape': False, 'left_shape': list(left.shape), 'right_shape': list(right.shape)}
    delta = (left.double() - right.double()).abs()
    return {'same_shape': True, 'exact': torch.equal(left, right),
            'max_abs': float(delta.max()) if delta.numel() else 0.,
            'mean_abs': float(delta.mean()) if delta.numel() else 0.,
            'changed_fraction': float((left != right).double().mean()) if delta.numel() else 0.}


def export(root, arm, step, destination):
    import torch

    from scripts.native_long_assets import export_task_bundle, load_task_bundle
    from scripts.native_long_campaign import PROJECT, compose_config
    from scripts.native_long_confirmation_inference import (
        validate_checkpoint,
        validate_request,
    )
    from scripts.native_long_execution import native_training_identity
    from scripts.native_long_runtime import configure_numeric, system_from_common

    started = time.monotonic()
    configure_numeric()
    directory = root / 'postselection'
    lock, manifest = (read_json(directory / name) for name in ('FORMAL_LOCK.json', 'FORMAL_POPULATION.json'))
    validate_request(lock, manifest, arm, step, 'PB')
    if sha256(directory / 'FORMAL_POPULATION.json') != lock['FORMAL_POPULATION_sha256']:
        raise ValueError('formal population SHA differs')
    config = compose_config(PROJECT / 'conf/config_native_long_retrain.yaml', root=root, arm=arm, seed=45)
    checkpoint = root / 'training' / arm / 'seed45' / f'update={step:05d}.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = native_training_identity(root, config)
    validate_checkpoint(saved, identity, step)
    system, _ = system_from_common(root, config)
    system.load_state_dict(saved['state_dict'], strict=True)
    del saved
    if destination.exists():
        raise ValueError('refusing to overwrite a trained bundle')
    receipt = export_task_bundle(root, system, step=step, destination=destination)
    restored = load_task_bundle(destination)
    original, reloaded = system.state_dict(), restored.state_dict()
    if original.keys() != reloaded.keys() or any(not torch.equal(v, reloaded[k]) for k, v in original.items()):
        raise ValueError('trained bundle tensor roundtrip differs')
    receipt.update(status='EXACT_TRAINED_TENSOR_RELOAD', checkpoint_sha256=sha256(checkpoint),
                   training_identity=identity, tensor_entries=len(original),
                   formal_lock_sha256=sha256(directory / 'FORMAL_LOCK.json'),
                   export_producer_sha256=sha256(__file__), cpu_seconds=time.monotonic() - started)
    write_json_atomic(destination.with_suffix('.json'), receipt)
    return receipt


def run(root, bundle, output, mode, order):
    import torch

    from datasets.native_long_dataset import (
        NativeLongCollator,
        build_native_datasets,
        isolated_rng,
        stable_seed,
    )
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.native_long_assets import load_task_bundle
    from scripts.native_long_runtime import configure_numeric, load_samples, move_batch
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    started = time.perf_counter()
    configure_numeric()
    receipt = read_json(bundle.with_suffix('.json'))
    if receipt['status'] != 'EXACT_TRAINED_TENSOR_RELOAD' or sha256(bundle) != receipt['sha256']:
        raise ValueError('bundle lacks a valid exact trained tensor roundtrip')
    if receipt['metadata']['population_sha256'] != sha256(root / 'POPULATION.json'):
        raise ValueError('deployment CAL population differs from training')
    if output.exists():
        raise ValueError('refusing to overwrite deployment evidence')
    population = read_json(root / 'POPULATION.json')
    records = fixed_cal_inputs(population)
    system = load_task_bundle(bundle)
    config = system.config
    device = torch.device('cuda:0')
    system.to(device).eval()
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    collate = NativeLongCollator(config, training=False)
    mapper = build_rio_class_mapper(datasets['rio'])
    result = {'status': 'INCOMPLETE', 'mode': mode, 'order': order,
              'arm': receipt['metadata']['arm'], 'step': receipt['metadata']['step'],
              'bundle_sha256': receipt['sha256'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
              'population_sha256': sha256(root / 'POPULATION.json'),
              'producer_sha256': sha256(__file__), 'evaluation_seed': 9001,
              'device': torch.cuda.get_device_name(device), 'records': records,
              'setup_seconds': time.perf_counter() - started, 'rows': []}
    feedback_events, handles = [], []
    if mode == 'profile' and getattr(system.model, 'native_long_feedback', None) is not None:
        def before(module, inputs):
            pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            pair[0].record()
            feedback_events.append(pair)

        def after(module, inputs, output):
            feedback_events[-1][1].record()

        handles = [system.model.native_long_feedback.register_forward_pre_hook(before),
                   system.model.native_long_feedback.register_forward_hook(after)]

    def forward(record, capture=False):
        feedback_events.clear()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with isolated_rng(stable_seed(9001, record['input_id']), cuda_devices=(0,)):
            begin = time.perf_counter()
            batch = collate(load_samples(datasets, [record]))
            prepared = time.perf_counter()
            data, targets, _ = move_batch(batch, device)
            torch.cuda.synchronize(device)
            moved = time.perf_counter()
            with torch.inference_mode():
                outputs = system(data, point2segment=[targets[0]['point2segment']],
                                 raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            torch.cuda.synchronize(device)
            forwarded = time.perf_counter()
            prediction = extract_official_task_prediction(system=system, output=outputs,
                target_low_resolution=targets[0], target_full_resolution=data.target_full[0],
                data=data, class_mapper=mapper, latest_stage_index=record['horizon'] - 1).prediction()
            torch.cuda.synchronize(device)
            finished = time.perf_counter()
        timing = {'input_id': record['input_id'], 'horizon': record['horizon'],
                  'input_io_collate_seconds': prepared - begin, 'H2D_seconds': moved - prepared,
                  'network_seconds': forwarded - moved, 'postprocess_seconds': finished - forwarded,
                  'end_to_end_seconds': finished - begin,
                  'feedback_cuda_seconds': sum(a.elapsed_time(b) for a, b in feedback_events) / 1000,
                  'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                  'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        evidence = None
        if capture:
            # Tensor copies and official metric calculation are outside profile timing.
            accumulator = OfficialMetricAccumulator(mode='strict_online', dataset_spec=config.instance_metric.dataset)
            accumulator.update(prediction, _metric_target(data.target_full[0], datasets['rio']))
            evidence = {'final_logits': outputs['pred_logits'].detach().cpu().clone(),
                        'final_masks': outputs['pred_masks'][0].detach().cpu().clone(),
                        'input_coord': data['coord'].detach().cpu().clone(),
                        'input_feat': data['feat'].detach().cpu().clone(),
                        'inverse': data.inverse_maps[0].detach().cpu().clone(),
                        'point2segment': targets[0]['point2segment'].detach().cpu().clone(),
                        'prediction': {k: v.detach().cpu().clone() for k, v in prediction.items()},
                        'metrics': accumulator.compute(), 'official_state': accumulator.export_evidence()}
        return timing, evidence

    try:
        if mode == 'profile':
            for record in records:
                for repeat in range(4):
                    timing, _ = forward(record)
                    timing.update(repeat=repeat, warmup=repeat == 0)
                    result['rows'].append(timing)
                    write_json_atomic(output, result)
        else:
            if order not in ('A', 'BA'):
                raise ValueError('order probe must use A or BA')
            chosen = {'A': records[1], 'B': records[4]}
            evidence = None
            for letter in order:
                timing, evidence = forward(chosen[letter], capture=letter == 'A')
                result['rows'].append(timing)
            result['metrics'] = evidence['metrics']
            evidence_path = output.with_suffix('.pt')
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(evidence, evidence_path)
            result['tensor_evidence'] = {'path': str(evidence_path), 'sha256': sha256(evidence_path)}
    finally:
        for handle in handles:
            handle.remove()
    result.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - started)
    write_json_atomic(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'profile', 'order'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('E0', 'E1', 'E3'))
    parser.add_argument('--step', type=int)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--order', choices=('A', 'BA'), default='A')
    args = parser.parse_args()
    if args.mode == 'export':
        if args.arm is None or args.step is None:
            parser.error('export requires --arm and --step')
        result = export(args.root, args.arm, args.step, args.bundle)
    else:
        if args.output is None:
            parser.error('profile/order requires --output')
        result = run(args.root, args.bundle, args.output, args.mode, args.order)
    print(json.dumps({k: result.get(k) for k in ('status', 'arm', 'step', 'mode', 'metrics', 'sha256')}))


if __name__ == '__main__':
    main()
