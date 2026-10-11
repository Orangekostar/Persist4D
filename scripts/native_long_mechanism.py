"""Fresh-process trained mechanism probes on a fixed nested CAL panel."""

import argparse
import hashlib
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256


def build_panel(population):
    cal = population['evaluation_inputs']['CAL']
    if any(not r['input_id'].startswith('CAL:') for r in cal):
        raise ValueError('mechanism panel must use CAL only')
    units = sorted((r for r in cal if r['horizon'] == 5), key=lambda r: r['input_id'])
    if len(units) != 6 or len({r['reference_id'] for r in units}) != 2:
        raise ValueError('mechanism panel requires all six CAL T5 inputs and two references')
    records = []
    for unit in units:
        if len(unit['scan_ids']) != 5 or len(set(unit['scan_ids'])) != 5:
            raise ValueError('CAL T5 input is not five distinct actual visits')
        for h in range(1, 6):
            identity = json.dumps([unit['input_id'], unit['scan_ids'][:h]], sort_keys=True).encode()
            records.append({**unit, 'horizon': h, 'scan_ids': unit['scan_ids'][:h],
                            'scan_indices': unit['scan_indices'][:h], 'panel_unit': unit['input_id'],
                            'input_id': 'CAL-MECHANISM:' + hashlib.sha256(identity).hexdigest()})
    return records


@contextmanager
def feedback_capture(block, mode, rows):
    from scripts.system_comparison_inference import _tensor_sha256

    if mode not in ('normal', 'off') or (mode == 'off' and block is None):
        raise ValueError('invalid trained feedback intervention')
    if block is None:
        yield
        return

    def capture(module, inputs, output):
        queries, features, padding = inputs
        update = (output.detach() - features.detach()).norm()
        norm = features.detach().norm()
        rows.append({'input_sha256': {k: _tensor_sha256(v) for k, v in
                                     zip(('queries', 'features', 'padding'), inputs)},
                     'update_norm': float(update), 'feature_norm': float(norm),
                     'relative_update_norm': float(update / norm) if norm > 0 else None,
                     'query_count': queries.shape[1], 'valid_segments': int((~padding).sum())})
        return features if mode == 'off' else output

    handle = block.register_forward_hook(capture)
    try:
        yield
    finally:
        handle.remove()


def candidate_diagnostics(prediction, target):
    import torch

    masks, classes = prediction['pred_masks'].cpu().bool().T, prediction['pred_classes'].cpu()
    query_ids = prediction['source_query_ids'].cpu()
    rows = []
    for identity, label, mask in zip(target['ids'].cpu(), target['labels'].cpu(), target['masks'].cpu().bool()):
        if not mask.any():
            continue
        allowed = torch.where(classes == label)[0]
        candidates = masks[allowed]
        union = (candidates | mask).sum(1)
        iou = (candidates & mask).sum(1).float() / union.clamp_min(1)
        best = int(iou.argmax()) if iou.numel() else None
        score = float(iou[best]) if best is not None else 0.
        rows.append({'gt_id': int(identity), 'gt_class': int(label), 'points': int(mask.sum()),
                     'class_candidates': allowed.numel(), 'best_iou': score,
                     'best_query_id': int(query_ids[allowed[best]]) if score > 0 else None,
                     'high_iou_candidates': int((iou >= .5).sum()), 'matched': score >= .5})
    return rows


def capture_sampler(original_sample, sampling):
    def capture_sample(decomposed_feat, curr_sample_max=None, is_eval=False, extra=None,
                       stage_ids=None, sampling_stage_idx=None):
        if stage_ids is not None and sampling_stage_idx is not None:
            cap = stage_ids[0].numel() if curr_sample_max is None else curr_sample_max
            sampling.append((sampling_stage_idx, cap, stage_ids[0].detach().clone()))
        return original_sample(decomposed_feat, curr_sample_max, is_eval, extra,
                               stage_ids, sampling_stage_idx)

    return capture_sample


def sampling_panel(captured, input_id):
    import torch

    from datasets.native_long_dataset import stable_seed
    from models.native_long_modules import stage_stratified_indices
    from scripts.system_comparison_inference import _tensor_sha256

    result = []
    for execution, cap, stages in captured:
        available = stages.numel()
        count = min(cap, available)
        labels, capacities = torch.unique(stages, sorted=True, return_counts=True)
        seed = stable_seed(45, 'mechanism-sampling', input_id, execution)
        row = {'execution_stage': execution, 'cap': cap, 'available': available,
               'selected': count, 'cap_triggered': available > cap, 'stages': labels.tolist(),
               'available_per_stage': capacities.tolist(), 'stage_ids_sha256': _tensor_sha256(stages)}
        for name in ('uniform', 'stratified'):
            generator = torch.Generator(device=stages.device).manual_seed(seed)
            if available <= cap:
                indices = torch.arange(available, device=stages.device)
            elif name == 'uniform':
                indices = torch.randperm(available, device=stages.device, generator=generator)[:count]
            else:
                indices = stage_stratified_indices(stages, count, generator=generator)
            counts = [int((stages[indices] == label).sum()) for label in labels]
            row[name] = {'counts': counts, 'missing_available_stages': sum(x == 0 for x in counts),
                         'indices_sha256': _tensor_sha256(indices)}
        result.append(row)
    return result


def nullable_single_input_metrics(accumulator):
    """Keep undefined one-input AP missing; pooled states still retain the input."""
    raw = accumulator._metric.compute()
    suffixes = {'mAP': 'AP', 'mAP50': 'AP_50', 'mAP25': 'AP_25',
                'REC': 'REC', 'REC50': 'REC_50', 'REC25': 'REC_25'}
    result = {}
    for output, source in suffixes.items():
        value = float(raw['val_mean_t-' + source])
        result['online_t-' + output] = value if math.isfinite(value) else None
    return result


def run(root, job_name):
    import torch

    from datasets.native_long_dataset import (
        NativeLongCollator,
        build_native_datasets,
        isolated_rng,
        stable_seed,
    )
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.native_long_assets import ENCODER_PREFIXES, load_task_bundle
    from scripts.native_long_runtime import configure_numeric, load_samples, move_batch
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.system_comparison_inference import _tensor_sha256

    started = time.perf_counter()
    parent = root / 'postselection'
    plan = read_json(parent / 'MECHANISM_PLAN.json')
    job = next(j for j in plan['jobs'] if j['name'] == job_name)
    record = plan['records'][job['record_index']]
    project = Path(__file__).resolve().parents[1]
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('mechanism source changed: ' + name)
    if sha256(root / 'POPULATION.json') != plan['population_sha256']:
        raise ValueError('mechanism CAL population changed')
    if build_panel(read_json(root / 'POPULATION.json')) != plan['records']:
        raise ValueError('mechanism nested panel differs from CAL source')
    bundle = parent / 'runtime-assets' / job['bundle_name']
    receipt = read_json(bundle.with_suffix('.json'))
    if (sha256(bundle) != job['bundle_sha256'] or sha256(bundle.with_suffix('.json')) != job['receipt_sha256']
            or receipt['status'] != 'EXACT_TRAINED_TENSOR_RELOAD'
            or receipt['checkpoint_sha256'] != job['checkpoint_sha256']
            or (receipt['metadata']['arm'], receipt['metadata']['step']) != (job['arm'], job['step'])):
        raise ValueError('mechanism trained bundle identity differs')
    output = parent / 'mechanism' / (job_name + '.json')
    if output.exists() or output.with_suffix('.pt').exists():
        raise ValueError('refusing to overwrite mechanism evidence')
    configure_numeric()
    system = load_task_bundle(bundle)
    device = torch.device('cuda:0')
    system.to(device).eval()
    config = system.config
    for item in plan['data_files']:
        path = Path(config.native_long.data_root) / item['relative_path']
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('mechanism CAL data bytes changed: ' + item['relative_path'])
    if sha256(Path(config.instance_metric.dataset)) != plan['metric_spec_sha256']:
        raise ValueError('mechanism metric taxonomy changed')
    datasets = build_native_datasets(config, config.native_long.data_root, augmentation=False)
    for scan, index in zip(record['scan_ids'], record['scan_indices']):
        if Path(datasets['rio'].data[index]['raw_filepath']).parent.name != scan:
            raise ValueError('mechanism physical CAL scan/index differs')
    collate = NativeLongCollator(config, training=False)
    mapper = build_rio_class_mapper(datasets['rio'])
    feedback = system.model.native_long_feedback
    captures, sampling = [], []
    result = {'status': 'INCOMPLETE', 'job': job, 'record': record,
              'plan_sha256': sha256(parent / 'MECHANISM_PLAN.json'),
              'runtime_status': 'RUNTIME_CONDITIONAL', 'evaluation_seed': 9001,
              'setup_seconds': time.perf_counter() - started}
    write_json_atomic(output, result)
    with isolated_rng(stable_seed(9001, record['input_id']), cuda_devices=(0,)):
        batch = collate(load_samples(datasets, [record]))
        data, targets, _ = move_batch(batch, device)
        if job['mode'] == 'gradient':
            from trainer.trainer import _configured_objective_loss

            system.train()
            system.model.native_long_draw_ids = [record['input_id']]
            feedback.checkpoint_chunks = bool(config.native_long.get('feedback_checkpoint', False))
            before = {k: _tensor_sha256(p) for k, p in system.named_parameters()}
            outputs = system(data, point2segment=[targets[0]['point2segment']],
                             raw_coordinates=system._process_raw_coordinates(data))
            losses = system.criterion(outputs, targets, mask_type=system.mask_type)
            total = _configured_objective_loss(system, losses)
            if not torch.equal(total, sum(losses.values())) or len(outputs['aux_outputs']) != 12:
                raise ValueError('trained gradient probe objective/auxiliary count differs')
            total.backward()
            gradients = {k: float(p.grad.detach().norm()) for k, p in system.named_parameters() if p.grad is not None}
            if any(not torch.isfinite(p.grad).all() for p in system.parameters() if p.grad is not None):
                raise ValueError('trained gradient probe produced nonfinite gradients')
            if any(k.startswith(ENCODER_PREFIXES) for k in gradients):
                raise ValueError('frozen encoder acquired gradients')
            if any(_tensor_sha256(p) != before[k] for k, p in system.named_parameters()):
                raise ValueError('gradient-only probe changed model parameters')
            groups = {'feedback': 'model.native_long_feedback.', 'mask_head': 'model.mask_embed_head.',
                      'cross_attention': 'model.cross_attention.', 'self_attention': 'model.self_attention.',
                      'decoder_ffn': 'model.ffn_attention.'}
            result.update(gradient_groups={g: {'tensor_count': sum(k.startswith(prefix) for k in gradients),
                'l2': sum(v*v for k, v in gradients.items() if k.startswith(prefix)) ** .5}
                for g, prefix in groups.items()}, gradient_norms=gradients,
                loss_terms={k: float(v.detach()) for k, v in losses.items()}, total_loss=float(total.detach()),
                no_optimizer_step=True, parameters_unchanged=True, frozen_encoder_gradients_absent=True,
                feedback_checkpoint_chunks=feedback.checkpoint_chunks, auxiliary_count=12)
        else:
            original_sample = system.model.sample_and_batch_features
            if job['arm'] == 'E1' and job['step'] == 23760 and record['horizon'] == 5:
                system.model.sample_and_batch_features = capture_sampler(original_sample, sampling)
            try:
                with torch.inference_mode(), feedback_capture(feedback, job['mode'], captures):
                    outputs = system(data, point2segment=[targets[0]['point2segment']],
                                     raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            finally:
                system.model.sample_and_batch_features = original_sample
            if len(outputs['aux_outputs']) != 12 or (feedback is not None and len(captures) != 1):
                raise ValueError('trained forward changed feedback/auxiliary execution count')
            official = extract_official_task_prediction(system=system, output=outputs,
                target_low_resolution=targets[0], target_full_resolution=data.target_full[0], data=data,
                class_mapper=mapper, latest_stage_index=record['horizon'] - 1)
            metric_target = _metric_target(data.target_full[0], datasets['rio'])
            target = {k: metric_target[k].cpu() for k in ('ids', 'labels', 'masks', 'changes', 'temporal_stages')}
            prediction = {k: v.cpu() for k, v in official.prediction().items()}
            current = {**target, 'masks': target['masks'][:, target['temporal_stages'] == record['horizon'] - 1]}
            accumulator = OfficialMetricAccumulator(mode='strict_online', dataset_spec=config.instance_metric.dataset)
            accumulator.update(prediction, target)
            tensor_path = output.with_suffix('.pt')
            tensors = {'stages': [{'logits': r['pred_logits'].detach().cpu().clone(),
                                  'masks': r['pred_masks'][0].detach().cpu().clone()}
                                 for r in [*outputs['aux_outputs'], outputs]],
                       'prediction': prediction, 'target': target}
            torch.save(tensors, tensor_path)
            result.update(feedback=captures, sampling=sampling_panel(sampling, record['input_id']),
                stage_sha256=[{k: _tensor_sha256(v) for k, v in r.items()} for r in tensors['stages']],
                input_sha256={k: _tensor_sha256(v) for k, v in
                              {'coord': data['coord'], 'feat': data['feat'], 'inverse': data.inverse_maps[0],
                               'point2segment': targets[0]['point2segment']}.items()},
                target_sha256={k: _tensor_sha256(v) for k, v in target.items()},
                prefix_candidates=candidate_diagnostics(prediction, target),
                current_candidates=candidate_diagnostics(official.prediction(latest_only=True), current),
                metrics=nullable_single_input_metrics(accumulator), official_state=accumulator.export_evidence(),
                tensor_evidence={'path': str(tensor_path), 'sha256': sha256(tensor_path)})
    torch.cuda.synchronize()
    result.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json_atomic(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    row = run(args.root, args.job)
    print(json.dumps({'status': row['status'], 'job': row['job']['name']}))
