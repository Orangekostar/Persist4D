"""CAL-locked negative-arm export and portable, byte-bound task-bundle loading."""

import argparse
import time
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def validate_cal_selection(lock, arm, step, population_sha256):
    if (lock.get('schema') != 1 or lock.get('seed') != 45
            or lock.get('all_planned_trajectories_stopped_before_SEL') is not True
            or lock.get('population_sha256') != population_sha256
            or lock.get('selected_updates', {}).get(arm) != step):
        raise ValueError('export does not match the stopped, seed-45 CAL selection')


def portable_config(payload, encoder, dataset_spec, dataset_spec_sha256):
    from omegaconf import OmegaConf

    encoder, dataset_spec = Path(encoder).resolve(), Path(dataset_spec).resolve()
    if sha256(encoder) != payload['metadata']['pretrained_encoder_sha256']:
        raise ValueError('separate encoder bytes differ from the trained asset')
    if sha256(dataset_spec) != dataset_spec_sha256:
        raise ValueError('metric taxonomy bytes differ from the delivery manifest')
    config = OmegaConf.create(payload['config'])
    config.backbone.name = str(encoder)
    config.model.config.backbone.name = str(encoder)
    config.instance_metric.dataset = str(dataset_spec)
    return config


def load_portable_bundle(path, *, encoder, dataset_spec, dataset_spec_sha256):
    import torch

    from datasets.native_long_dataset import isolated_rng
    from scripts.native_long_assets import ENCODER_PREFIXES
    from scripts.native_long_campaign import code_identity
    from trainer.native_long_trainer import NativeLongTrainer

    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['schema'] != 1 or payload['metadata']['code'] != code_identity():
        raise ValueError('task bundle schema/numeric code identity differs')
    config = portable_config(payload, encoder, dataset_spec, dataset_spec_sha256)
    with isolated_rng(config.general.seed):
        system = NativeLongTrainer(config)
    current = system.state_dict()
    omitted = {k for k, _ in system.named_parameters() if k.startswith(ENCODER_PREFIXES)}
    buffers = {k for k, _ in system.named_buffers() if k.startswith(ENCODER_PREFIXES)}
    if (set(payload['omitted_encoder_parameter_names']) != omitted
            or set(payload['included_encoder_buffers']) != buffers
            or set(payload['state_dict']) != set(current) - omitted):
        raise ValueError('bundle omitted a task parameter or mutable encoder buffer')
    current.update(payload['state_dict'])
    system.load_state_dict(current, strict=True)
    return system


def export_cal_locked(root, arm, step, destination, cal_lock_sha256):
    import torch

    from scripts.native_long_assets import export_task_bundle, load_task_bundle
    from scripts.native_long_campaign import PROJECT, compose_config
    from scripts.native_long_confirmation_inference import validate_checkpoint
    from scripts.native_long_execution import native_training_identity
    from scripts.native_long_runtime import configure_numeric, system_from_common

    started = time.monotonic()
    lock_path = root / 'selection/CAL_LOCK.json'
    if sha256(lock_path) != cal_lock_sha256:
        raise ValueError('remote CAL lock differs from the controller selection')
    validate_cal_selection(read_json(lock_path), arm, step, sha256(root / 'POPULATION.json'))
    if (root / 'ACTIVE_PROCESS.json').exists():
        raise ValueError('CPU export requires the training node to be inactive')
    if destination.exists() or destination.with_suffix('.json').exists():
        raise ValueError('refusing to overwrite a deployment bundle')
    configure_numeric()
    config = compose_config(PROJECT / 'conf/config_native_long_retrain.yaml', root=root, arm=arm, seed=45)
    checkpoint = root / 'training' / arm / 'seed45' / f'update={step:05d}.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    identity = native_training_identity(root, config)
    validate_checkpoint(saved, identity, step)
    system, _ = system_from_common(root, config)
    system.load_state_dict(saved['state_dict'], strict=True)
    del saved
    receipt = export_task_bundle(root, system, step=step, destination=destination)
    restored = load_task_bundle(destination)
    original, reloaded = system.state_dict(), restored.state_dict()
    if original.keys() != reloaded.keys() or any(not torch.equal(v, reloaded[k]) for k, v in original.items()):
        raise ValueError('CAL-locked trained bundle tensor roundtrip differs')
    receipt.update(status='EXACT_TRAINED_TENSOR_RELOAD', checkpoint_sha256=sha256(checkpoint),
                   training_identity=identity, tensor_entries=len(original),
                   cal_lock_sha256=cal_lock_sha256, export_producer_sha256=sha256(__file__),
                   cpu_seconds=time.monotonic() - started)
    write_json_atomic(destination.with_suffix('.json'), receipt)
    return receipt


def audit_portable_assets(directory, root, encoder, dataset_spec):
    import torch

    from scripts.native_long_assets import ENCODER_PREFIXES

    started = time.monotonic()
    source = read_json(root / 'SOURCE_AND_INITIALIZATION.json')
    initial_path = Path(source['initialization']['45']['path'])
    if sha256(initial_path) != source['initialization']['45']['sha256']:
        raise ValueError('local encoder reference initialization changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)['state_dict']
    plan = read_json(directory / 'MECHANISM_PLAN.json')
    spec_sha = plan['metric_spec_sha256']
    rows = []
    for arm, step in (('E0', 5940), ('E1', 23760), ('E1', 2970), ('E2', 29700), ('E3', 2970)):
        path = directory / 'assets' / f'{arm}-{step:05d}.pt'
        receipt = read_json(path.with_suffix('.json'))
        if sha256(path) != receipt['sha256'] or receipt['status'] != 'EXACT_TRAINED_TENSOR_RELOAD':
            raise ValueError('trained export receipt differs')
        system = load_portable_bundle(path, encoder=encoder, dataset_spec=dataset_spec,
                                      dataset_spec_sha256=spec_sha)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        omitted = set(payload['omitted_encoder_parameter_names'])
        expected = {**{k: initial[k] for k in omitted}, **payload['state_dict']}
        actual = system.state_dict()
        if expected.keys() != actual.keys() or any(not torch.equal(v, actual[k]) for k, v in expected.items()):
            raise ValueError('portable task/buffer/frozen-encoder reconstruction differs')
        rows.append({'arm': arm, 'step': step, 'bundle_sha256': receipt['sha256'],
                     'checkpoint_sha256': receipt['checkpoint_sha256'], 'tensor_entries': len(actual),
                     'frozen_encoder_parameter_entries': len(omitted),
                     'mutable_encoder_buffers': sum(k.startswith(ENCODER_PREFIXES) for k, _ in system.named_buffers()),
                     'status': 'EXACT_TENSOR_RELOAD_WITH_RELOCATED_ENCODER_AND_TAXONOMY'})
        del system, payload, actual, expected
    result = {'status': 'VERIFIED', 'rows': rows, 'encoder_sha256': sha256(encoder),
              'taxonomy_sha256': spec_sha, 'loader_sha256': sha256(__file__),
              'reference_initialization_sha256': sha256(initial_path),
              'scope': 'all trained task parameters/buffers plus all frozen encoder parameters; CPU tensors, no runtime-invariance claim',
              'cpu_seconds': time.monotonic() - started}
    write_json_atomic(directory / 'PORTABLE_ASSET_AUDIT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('export', 'audit'), required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('E0', 'E1', 'E2', 'E3'))
    parser.add_argument('--step', type=int)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--cal-lock-sha256')
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--encoder', type=Path)
    parser.add_argument('--dataset-spec', type=Path)
    args = parser.parse_args()
    if args.mode == 'export':
        result = export_cal_locked(args.root, args.arm, args.step, args.destination, args.cal_lock_sha256)
    else:
        result = audit_portable_assets(args.directory, args.root, args.encoder, args.dataset_spec)
    print({'status': result['status'], 'rows': len(result.get('rows', []))})
