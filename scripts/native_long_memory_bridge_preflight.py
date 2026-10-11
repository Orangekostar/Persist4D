"""CPU audit of shared point order and common truth for every fixed PB prefix."""

import argparse
import time
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def run(directory, data_root):
    import torch
    from omegaconf import OmegaConf

    from datasets.native_long_dataset import NativeLongCollator
    from scripts.evaluate_persist4d_p6a import build_temporal_target
    from scripts.native_long_campaign import code_identity
    from scripts.native_long_confirmation_inference import (
        formal_dataset,
        formal_samples,
    )
    from scripts.native_long_memory_bridge import (
        assert_same_temporal_truth,
        context_record,
        current_target,
        project_common_truth,
    )
    from scripts.p6a_cache import CHANGE_LABEL_SEMANTICS
    from scripts.system_comparison_inference import _tensor_sha256

    started = time.time()
    manifest = read_json(directory / 'FORMAL_POPULATION.json')
    receipt = directory / 'assets/E0-05940.json'
    config = OmegaConf.create(read_json(receipt)['training_identity']['config'])
    dataset = formal_dataset(config, data_root, manifest)
    collate = NativeLongCollator(config, training=False)
    units = sorted((r for r in manifest['records']['PB'] if r['horizon'] == 5), key=lambda r: r['source_unit'])
    result = {'status': 'INCOMPLETE', 'rows': [], 'formal_population_sha256': sha256(directory / 'FORMAL_POPULATION.json'),
              'config_receipt_sha256': sha256(receipt), 'code': code_identity(),
              'producer_sha256': sha256(__file__),
              'bridge_sha256': sha256(Path(__file__).with_name('native_long_memory_bridge.py'))}
    output = directory / 'BRIDGE_CPU_ALIGNMENT.json'
    unit, h = None, None
    write_json_atomic(output, result)
    try:
        for index, unit in enumerate(units):
            payloads = []
            for h in range(1, 6):
                targets, proofs, point_ids = [], [], []
                for mode in ('full', 'local'):
                    record = context_record(unit, h, mode)
                    data, _, _ = collate(formal_samples(dataset, record))
                    target = data.target_full[0]
                    targets.append(current_target(target, record['horizon']))
                    selector = (target['temporal_stages'] == record['horizon'] - 1).numpy()
                    labels = torch.from_numpy(data.original_labels[0][selector, :2].copy())
                    coords = torch.from_numpy(data.original_coordinates[0][selector, :3].copy())
                    proofs.append({'coordinates': _tensor_sha256(coords), 'semantic_instance': _tensor_sha256(labels)})
                    point_ids.append(labels[:, 1])
                    if mode == 'full':
                        full = {k: target[k] for k in ('ids', 'labels', 'masks', 'temporal_stages')}
                        full['changes'] = torch.zeros_like(full['ids'])
                if proofs[0] != proofs[1]:
                    raise ValueError('full/pair current raw point order or annotations differ')
                aligned = project_common_truth(targets[0], point_ids[1])
                payloads.append({'key': {'stage_index': h - 1}, 'target': {
                    'gt_ids': aligned['ids'], 'gt_classes': aligned['labels'], 'gt_masks': aligned['masks'],
                    'changes': torch.zeros_like(aligned['ids']), 'change_labels_valid': False,
                    'change_label_semantics': CHANGE_LABEL_SEMANTICS, 'gt_class_semantics': 'rescene_model_index_0_based'}})
                assert_same_temporal_truth(full, build_temporal_target(payloads))
                a, b = (dict(zip(t['ids'].tolist(), t['labels'].tolist())) for t in targets)
                drift = [{'id': i, 'full_label': a.get(i), 'local_label': b.get(i)}
                         for i in sorted(a.keys() | b.keys()) if a.get(i) != b.get(i)]
                result['rows'].append({'unit': unit['source_unit'], 'horizon': h, 'window_label_drift': drift,
                                       'masks_shape': list(aligned['masks'].shape), 'current_points_sha256': proofs[0]})
            if (index + 1) % 20 == 0:
                print(f'CPU alignment {index + 1}/{len(units)}', flush=True)
        if len(units) != 129 or len(result['rows']) != 645:
            raise ValueError('CPU bridge coverage differs')
        result.update(status='PASS', units=len(units), checks=len(result['rows']),
                      annotation_differences=sum(len(r['window_label_drift']) for r in result['rows']))
    except BaseException as error:
        result.update(status='FAILED', error=repr(error), failed_unit=(unit or {}).get('source_unit'), failed_horizon=h)
        raise
    finally:
        result['elapsed_seconds'] = time.time() - started
        write_json_atomic(output, result)
    return {k: result[k] for k in ('status', 'units', 'checks', 'annotation_differences', 'elapsed_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.directory, args.data_root))
