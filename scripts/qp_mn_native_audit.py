"""Read-only native/cache replay diagnostic; never changes experiment inputs."""
import hashlib
import os
from pathlib import Path

import torch

from scripts.qp_mn_binding import Context, gpu_scope
from scripts.short_module_native import NativeSession, unpack_prediction
from scripts.short_module_screen import build_datasets, native_config, write_json


def differences(left, right):
    rows = {}
    for key, value in left.items():
        other = right[key]
        if not isinstance(value, torch.Tensor):
            continue
        equal = torch.equal(value, other)
        row = {'equal': equal, 'shape': list(value.shape), 'other_shape': list(other.shape)}
        if value.shape == other.shape:
            row['different_values'] = int((value != other).sum())
            row['max_absolute_difference'] = float((value.double() - other.double()).abs().max()) if value.numel() else 0.
        rows[key] = row
    return rows


def audit(c, device='cuda:0'):
    config = native_config(c.root, c.assets)
    datasets = build_datasets(config, c.assets)
    for dataset in datasets.values():
        dataset.apply_training_augmentation = False
    session = NativeSession(config=config, datasets=datasets, assets=c.assets, device=device)
    groups = {}
    for record in c.population:
        if record['role'] == 'SEL' and record['horizon'] == 2:
            groups.setdefault(record['reference_id'], []).append(record)
    record = min(groups[min(groups)], key=lambda r: (hashlib.sha256(r['input_id'].encode()).hexdigest(), r['input_id']))
    entry = next(e for e in c.eval_index['entries'] if e['input_id'] == record['input_id'])
    cached = torch.load(Path(c.eval_index['cache']) / entry['prediction']['file'], weights_only=False)
    results = []
    previous = None
    traces = {}
    handles = []
    def trace(name):
        def hook(module, args, result):
            if isinstance(result, dict):
                traces[name] = {k: v.detach().cpu().clone() for k, v in result.items()
                                if isinstance(v, torch.Tensor) and k in ('feat', 'coord', 'grid_coord', 'serialized_order')}
        return hook
    if os.environ.get('QPMN_TRACE') == '1':
        for name, module in session.system.named_modules():
            if name.endswith('embedding') or any(name.endswith(f'.{kind}{i}') for kind in ('enc', 'dec') for i in range(5)):
                handles.append(module.register_forward_hook(trace(name)))
        session.produce(record)
        before = dict(traces)
    if os.environ.get('QPMN_REPLAY_EXPORT_PREFIX') == '1':
        for index, warm_entry in enumerate(c.eval_index['entries']):
            if index >= int(os.environ.get('QPMN_PREFIX_LIMIT', '94')):
                break
            if warm_entry['input_id'] == record['input_id']:
                break
            warm, _, _ = session.produce(warm_entry['record'])
            warm_cache = torch.load(Path(c.eval_index['cache']) / warm_entry['prediction']['file'], weights_only=False)
            comparison = differences(unpack_prediction(warm['parent']).prediction(), unpack_prediction(warm_cache['parent']).prediction())
            print('PREFIX', warm_entry['input_id'], {k: v['equal'] for k, v in comparison.items()}, flush=True)
    for repeat in range(3):
        current, _, audit_row = session.produce(record, compare_native=True)
        prediction = unpack_prediction(current['parent']).prediction()
        results.append({'repeat': repeat, 'audit': audit_row,
                        'cache_prediction': differences(prediction, unpack_prediction(cached['parent']).prediction()),
                        'cache_soft': differences(current['parent']['soft'], cached['parent']['soft']),
                        'cache_coordinates': differences({'coordinates': current['coordinates']}, {'coordinates': cached['coordinates']}),
                        'previous_prediction': differences(prediction, previous) if previous else None})
        previous = prediction
    if os.environ.get('QPMN_TRACE') == '1':
        results.append({'trace_differences': {k: differences(traces[k], before[k]) for k in traces}})
    condition = 'controlled' if os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' else 'uncontrolled'
    if os.environ.get('QPMN_REPLAY_EXPORT_PREFIX') == '1':
        condition += '_prefix' + os.environ.get('QPMN_PREFIX_LIMIT', 'all')
    write_json(c.artifacts / f'resources/NATIVE_REPLAY_AUDIT_{condition}.json', {
        'input_id': record['input_id'], 'rows': results,
        'environment': {k: os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}})
    return results


if __name__ == '__main__':
    torch.set_num_threads(2)
    context = Context.load('configs/qp_mn_targeted_v1.yaml')
    with gpu_scope(context, device='cuda:0', stage='native-replay-audit'):
        print(audit(context))
