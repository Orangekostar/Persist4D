"""Fixed trained-perceptor comparison of full history, pair association and memory."""

import argparse
import collections
import dataclasses
import hashlib
import json
import time
from pathlib import Path

from scripts.native_long_budget import write_json_atomic
from scripts.native_long_postselection import read_json, sha256

METHODS = ('FH-B2', 'LOCAL-B2', 'LOCAL-B4')
COUNTS = ('deployment_id_switches', 'identity_transition_opportunities', 'fragmentation_count',
          'fragmentation_opportunities', 'merge_count', 'merge_opportunities', 'gap_opportunities',
          'recovery_attempts', 'correct_recoveries')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def context_record(unit, horizon, mode):
    if mode not in ('full', 'local') or horizon not in range(1, 6):
        raise ValueError('bridge context must be causal full/local T1–T5')
    start = max(0, horizon - 2) if mode == 'local' else 0
    result = {**unit, 'horizon': horizon - start}
    for key in ('scan_ids', 'scan_indices', 'physical_scan_ids'):
        result[key] = unit[key][start:horizon]
    result['input_id'] = 'memory-bridge:' + digest(result['physical_scan_ids'])
    result['draw_seed_key'] = digest(unit['physical_scan_ids'][:horizon])
    return result


def pool_identity(rows):
    totals = {key: sum(r[key] for r in rows) for key in COUNTS}
    for rate, numerator, denominator in (
        ('normalized_id_switch_rate', 'deployment_id_switches', 'identity_transition_opportunities'),
        ('fragmentation_rate', 'fragmentation_count', 'fragmentation_opportunities'),
        ('merge_rate', 'merge_count', 'merge_opportunities'),
        ('gap_recovery_accuracy', 'correct_recoveries', 'recovery_attempts'),
        ('gap_recovery_recall', 'correct_recoveries', 'gap_opportunities'),
    ):
        totals[rate] = totals[numerator] / totals[denominator] if totals[denominator] else None
    return totals


def current_target(target, horizon):
    masks = target['masks'][:, target['temporal_stages'] == horizon - 1].cpu().bool()
    present = masks.any(dim=1)
    ids = target['ids'].cpu()[present]
    order = ids.argsort()
    return {'ids': ids[order], 'labels': target['labels'].cpu()[present][order],
            'masks': masks[present][order]}


def audit_current_alignment(full, local):
    import torch

    if any(not torch.equal(full[k], local[k]) for k in ('ids', 'masks')):
        raise ValueError('full/local current GT identity or point-order alignment differs')
    return [{'id': int(identity), 'full_label': int(a), 'local_label': int(b)}
            for identity, a, b in zip(full['ids'], full['labels'], local['labels']) if a != b]


def project_common_truth(common, local_point_ids):
    projected = {'ids': common['ids'].clone(), 'labels': common['labels'].clone(),
                 'masks': common['ids'][:, None] == local_point_ids[None, :]}
    audit_current_alignment(common, projected)
    return projected


def assert_same_temporal_truth(full, accumulated):
    import torch

    if not torch.equal(full['temporal_stages'], accumulated['temporal_stages']):
        raise ValueError('committed local point stages differ from full-prefix target')
    a, b = full['ids'].argsort(), accumulated['ids'].argsort()
    if any(not torch.equal(full[k][a], accumulated[k][b]) for k in ('ids', 'labels', 'masks', 'changes')):
        raise ValueError('committed local truth differs from common full-prefix target')


def build_local_payload(unit, h, observation, current, provenance):
    import torch

    from scripts.p6a_cache import (
        CHANGE_LABEL_SEMANTICS,
        SCHEMA_VERSION,
        validate_cache_payload,
    )

    master, order = unit['source_unit'].rsplit(':', 1)
    payload = {'schema_version': SCHEMA_VERSION, 'provenance': provenance,
               'key': {'master_sequence_id': master, 'reference_scene_id': unit['reference_id'],
                       'order_id': order, 'stage_index': h - 1,
                       'history_scan_ids': unit['scan_ids'][:h],
                       'local_window_scan_ids': unit['scan_ids'][max(0, h - 2):h]},
               'observation': observation,
               'target': {'gt_ids': current['ids'], 'gt_classes': current['labels'],
                          'gt_masks': current['masks'], 'changes': torch.zeros_like(current['ids']),
                          'change_labels_valid': False, 'change_label_semantics': CHANGE_LABEL_SEMANTICS,
                          'gt_class_semantics': 'rescene_model_index_0_based'}}
    validate_cache_payload(payload)
    return payload


def run(root, data_root, bundle, directory):
    import torch
    import yaml

    from datasets.native_long_dataset import (
        NativeLongCollator,
        isolated_rng,
        stable_seed,
    )
    from scripts.evaluate_persist4d_p6a import (
        _remap_metric_target,
        build_rio_class_mapper,
        build_temporal_target,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
    )
    from scripts.native_long_assets import load_task_bundle
    from scripts.native_long_confirmation_inference import (
        formal_dataset,
        formal_samples,
    )
    from scripts.native_long_runtime import configure_numeric, move_batch
    from scripts.p6a_cache import validate_cache_payload
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from scripts.rescene_task_postprocess import extract_official_task_prediction
    from scripts.system_comparison_inference import (
        _tensor_sha256,
        full_history_content_sha256,
        postprocess_full_history_output,
    )
    from scripts.system_comparison_metrics import (
        compute_deployment_identity_metrics,
        match_identity_update,
    )
    from scripts.system_comparison_v2_cache import build_task_sidecar
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )

    started = time.perf_counter()
    configure_numeric()
    project = Path(__file__).resolve().parents[1]
    parent = root / 'postselection'
    plan = read_json(parent / 'BRIDGE_PLAN.json')
    manifest = read_json(parent / 'FORMAL_POPULATION.json')
    receipt = read_json(bundle.with_suffix('.json'))
    selected = next((r for r in plan['rows'] if r['bundle_sha256'] == sha256(bundle)), None)
    if (selected is None or receipt['status'] != 'EXACT_TRAINED_TENSOR_RELOAD'
            or sha256(bundle.with_suffix('.json')) != selected['receipt_sha256']
            or receipt['checkpoint_sha256'] != selected['checkpoint_sha256']
            or receipt['metadata']['arm'] != selected['arm'] or receipt['metadata']['step'] != selected['step']):
        raise ValueError('trained bundle differs from fixed bridge row')
    if sha256(parent / 'FORMAL_POPULATION.json') != plan['formal_population_sha256']:
        raise ValueError('bridge formal population SHA differs')
    if sha256(parent / 'FORMAL_LOCK.json') != plan['formal_lock_sha256']:
        raise ValueError('bridge fixed checkpoint selection differs')
    for name, expected in plan['code'].items():
        if sha256(project / name) != expected:
            raise ValueError('bridge code/config SHA differs: ' + name)
    for item in manifest['data_files']:
        path = data_root / item['relative_path']
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError('bridge staged data changed')
    if (directory / 'RESULT.json').exists():
        raise ValueError('refusing to overwrite bridge evidence')
    units = sorted((r for r in manifest['records']['PB'] if r['horizon'] == 5), key=lambda r: r['source_unit'])
    if [r['input_id'] for r in units] != plan['unit_input_ids'] or len(units) != 129:
        raise ValueError('bridge requires all fixed 129 PB units')
    settings = yaml.safe_load((project / 'conf/p6a/default.yaml').read_text())
    factories = build_tracker_factories(settings)
    observation_settings = {key: settings['baselines']['b4'][key] for key in
                            ('background_class', 'confidence_threshold', 'mask_threshold', 'minimum_mask_support')}
    system = load_task_bundle(bundle)
    system.model.return_query_features = True
    config = system.config
    system.to('cuda:0').eval()
    dataset = formal_dataset(config, data_root, manifest)
    collate = NativeLongCollator(config, training=False)
    mapper = build_rio_class_mapper(dataset)
    provenance = {'source_commit': plan['source_commit'], 'checkpoint_sha256': receipt['checkpoint_sha256'],
                  'config_sha256': digest(receipt['training_identity']['config']),
                  'dataset_sha256': plan['formal_population_sha256']}
    identity = {'bridge_plan_sha256': sha256(parent / 'BRIDGE_PLAN.json'), 'bundle_sha256': sha256(bundle),
                'checkpoint_sha256': receipt['checkpoint_sha256'], 'native_runtime_status': 'RUNTIME_CONDITIONAL'}
    result = {'status': 'INCOMPLETE', 'arm': selected['arm'], 'step': selected['step'], 'identity': identity,
              'completed_units': 0, 'expected_units': len(units), 'forward_count': 0, 'shared_prefixes': 0,
              'unit_evidence': [], 'setup_seconds': time.perf_counter() - started}
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / 'RESULT.json', result)

    def metric(mode='strict_online'):
        return OfficialMetricAccumulator(mode=mode, dataset_spec=config.instance_metric.dataset)

    pooled = {(method, h): metric() for method in METHODS for h in range(2, 6)}
    raw = {(context, h): metric('raw_local') for context in ('full', 'local') for h in range(2, 6)}
    refs, identity_rows = {}, []

    def infer(unit, h, mode):
        record = context_record(unit, h, mode)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with isolated_rng(stable_seed(9001, record['draw_seed_key']), cuda_devices=(0,)):
            begin = time.perf_counter()
            data, targets, _ = move_batch(collate(formal_samples(dataset, record)), torch.device('cuda:0'))
            torch.cuda.synchronize()
            prepared = time.perf_counter()
            with torch.inference_mode():
                outputs = system(data, point2segment=[targets[0]['point2segment']],
                                 raw_coordinates=system._process_raw_coordinates(data), is_eval=True)
            torch.cuda.synchronize()
            forwarded = time.perf_counter()
            processed = postprocess_full_history_output(system=system, output=outputs,
                target_low_resolution=targets[0], target_full_resolution=data.target_full[0], data=data,
                horizon=record['horizon'], class_mapper=mapper, **observation_settings)
            official = extract_official_task_prediction(system=system, output=outputs,
                target_low_resolution=targets[0], target_full_resolution=data.target_full[0], data=data,
                class_mapper=mapper, latest_stage_index=record['horizon'] - 1)
            torch.cuda.synchronize()
            finished = time.perf_counter()
        full_target = data.target_full[0]
        current = current_target(full_target, record['horizon'])
        selector = (full_target['temporal_stages'].cpu() == record['horizon'] - 1).numpy()
        point_labels = torch.from_numpy(data.original_labels[0][selector, :2].copy())
        point_coords = torch.from_numpy(data.original_coordinates[0][selector, :3].copy())
        # This is a latest-pair cache only. Full-prefix observations remain in
        # their explicitly labelled processed result and never masquerade as it.
        payload = None
        if mode == 'local' or h <= 2:
            payload = build_local_payload(unit, h, processed.raw_observation, current, provenance)
        result['forward_count'] += 1
        timings = {'horizon': h, 'context': mode, 'consumed_scans': record['physical_scan_ids'],
                   'seed_key': record['draw_seed_key'],
                   'input_tensors_sha256': {k: _tensor_sha256(data[k]) for k in
                                           ('coord', 'feat', 'grid_coord', 'offset')
                                           if isinstance(data.get(k), torch.Tensor)},
                   'raw_observation_sha256': processed.observation_fingerprints,
                   'official_current_sha256': {k: _tensor_sha256(v) for k, v in
                                               official.prediction(latest_only=True).items()},
                   'current_target_sha256': {k: _tensor_sha256(v) for k, v in current.items()},
                   'current_points_sha256': {'coordinates': _tensor_sha256(point_coords),
                                             'semantic_and_instance': _tensor_sha256(point_labels)},
                   'io_collate_H2D_seconds': prepared - begin, 'network_seconds': forwarded - prepared,
                   'postprocess_seconds': finished - forwarded,
                   'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        return processed, official, payload, timings, current, point_labels[:, 1]

    for index, unit in enumerate(units):
        unit_id = unit['source_unit']
        trackers = {method: factories['B4' if method == 'LOCAL-B4' else 'B2'](unit_id) for method in METHODS}
        trajectories = {method: OfficialCandidateTrajectoryAccumulator(score_reducer='mean')
                        for method in METHODS if method.startswith('LOCAL')}
        updates = {method: [] for method in METHODS}
        payloads, timings, capacities, unit_rows, label_drift = [], [], [], [], []
        for h in range(1, 6):
            full = infer(unit, h, 'full')
            local = full if h <= 2 else infer(unit, h, 'local')
            if h <= 2:
                result['shared_prefixes'] += 1
            timings.append(full[3])
            if h > 2:
                timings.append(local[3])
            full_current = current_target(full[0].target, h)
            local_current = current_target(local[0].target, min(h, 2))
            if full[3]['current_points_sha256'] != local[3]['current_points_sha256']:
                raise ValueError('current raw point coordinates/annotations differ across contexts')
            full_labels = dict(zip(full_current['ids'].tolist(), full_current['labels'].tolist()))
            local_labels = dict(zip(local_current['ids'].tolist(), local_current['labels'].tolist()))
            label_drift.extend({'horizon': h, 'id': i, 'full_label': full_labels.get(i),
                                'local_label': local_labels.get(i)}
                               for i in sorted(full_labels.keys() | local_labels.keys())
                               if full_labels.get(i) != local_labels.get(i))
            # The official collator uses the first point in the observed prefix.
            # Use that same causal evaluation truth for all methods. This never
            # enters the network or the association observations.
            aligned = project_common_truth(full[4], local[5])
            local[2]['target'].update(gt_ids=aligned['ids'], gt_classes=aligned['labels'],
                                     gt_masks=aligned['masks'], changes=torch.zeros_like(aligned['ids']))
            validate_cache_payload(local[2])
            payloads.append(local[2])
            common_target = _remap_metric_target(build_temporal_target(payloads), mapper)
            assert_same_temporal_truth(full[0].target, common_target)
            sidecar = build_task_sidecar(raw_cache_payload=local[2], official_prediction=local[1],
                                         protocol_manifest_sha256=plan['formal_population_sha256'])
            sidecar_sha = full_history_content_sha256(sidecar)
            for method in METHODS:
                processed = full[0] if method == 'FH-B2' else local[0]
                obs = cache_payload_to_frozen_observation({'observation': processed.raw_observation})
                before = time.perf_counter()
                step = trackers[method].step(obs, stage_id=h - 1)
                timings.append({'horizon': h, 'method': method, 'tracker_seconds': time.perf_counter() - before})
                query_ids = [q for q, value in enumerate(step.track_ids) if value is not None]
                probabilities = processed.raw_observation['class_prob'][query_ids].clone()
                probabilities[:, 18] = -torch.inf
                prediction_classes = torch.tensor([mapper(int(c)) for c in probabilities.argmax(dim=-1)], dtype=torch.long)
                updates[method].append(match_identity_update(horizon=h,
                    gt_ids=full_current['ids'], gt_classes=full_current['labels'], gt_masks=full_current['masks'],
                    issued_ids=torch.tensor([step.track_ids[q] for q in query_ids], dtype=torch.long),
                    pred_classes=prediction_classes,
                    pred_masks=processed.raw_observation['masks'][query_ids].T.contiguous(), minimum_iou=.5))
                if method in trajectories:
                    trajectories[method].add_stage(sidecar, step)
                if method == 'LOCAL-B4':
                    state = trackers[method].state
                    tensors = [v for v in vars(state).values() if isinstance(v, torch.Tensor)]
                    capacities.append({'horizon': h, 'state_bytes': sum(t.numel() * t.element_size() for t in tensors),
                                       'occupied': int(state.occupied.sum()), 'active': int(state.active.sum()),
                                       'births': sum(step.births), 'rejected_births': sum(step.rejected_births)})
                if h >= 2:
                    if method == 'FH-B2':
                        prediction, target = full[0].task_prediction, full[0].target
                    else:
                        prediction = trajectories[method].snapshot().prediction
                        target = common_target
                    pooled[method, h].update(prediction, target)
                    key = (method, h, unit['reference_id'])
                    if key not in refs:
                        refs[key] = metric()
                    refs[key].update(prediction, target)
                    row = {'method': method, 'horizon': h, 'reference_id': unit['reference_id'],
                           'unit': unit_id, **compute_deployment_identity_metrics(updates[method])}
                    identity_rows.append(row)
                    unit_rows.append(row)
            if full_history_content_sha256(sidecar) != sidecar_sha:
                raise ValueError('association changed shared official candidates')
            timings.append({'horizon': h, 'shared_local_sidecar_sha256': sidecar_sha})
            if h >= 2:
                raw_target = {**full_current, 'changes': torch.zeros_like(full_current['ids']),
                              'temporal_stages': torch.zeros(full_current['masks'].shape[1], dtype=torch.long)}
                for context, value in (('full', full), ('local', local)):
                    raw[context, h].update(value[1].prediction(latest_only=True), raw_target)
            write_json_atomic(directory / 'PROGRESS.json', {'unit_index': index, 'unit': unit_id, 'horizon': h,
                                                            'forward_count': result['forward_count'], 'time_unix': time.time()})
        evidence = {'unit': unit_id, 'reference_id': unit['reference_id'], 'timings': timings,
                    'identity_rows': unit_rows, 'capacities': capacities,
                    'window_label_drift': label_drift,
                    'evaluation_truth': 'official_causal_full_prefix_first_observed_point',
                    'identity_updates': {m: [dataclasses.asdict(x) for x in rows] for m, rows in updates.items()},
                    'shared_T1_T2_exact': True, 'local_candidates_shared_B2_B4': True}
        path = directory / 'units' / f'{index:03d}.json'
        write_json_atomic(path, evidence)
        result['unit_evidence'].append({'path': str(path), 'sha256': sha256(path)})
        result['completed_units'] += 1
        write_json_atomic(directory / 'RESULT.json', result)
    result['metrics'], result['official_evidence'], result['by_reference'] = [], [], []
    for (method, h), accumulator in pooled.items():
        if accumulator._updates != 129:
            raise ValueError('bridge pooled official denominator differs')
        rows = [r for r in identity_rows if r['method'] == method and r['horizon'] == h]
        result['metrics'].append({'method': method, 'horizon': h, 'units': 129,
                                  **accumulator.compute(), **pool_identity(rows)})
        path = directory / 'official' / f'{method}-T{h}.json'
        write_json_atomic(path, accumulator.export_evidence())
        result['official_evidence'].append({'path': str(path), 'sha256': sha256(path)})
    counts = collections.Counter(unit['reference_id'] for unit in units)
    for (method, h, ref), accumulator in sorted(refs.items()):
        if accumulator._updates != counts[ref]:
            raise ValueError('bridge reference denominator differs')
        rows = [r for r in identity_rows if r['method'] == method and r['horizon'] == h and r['reference_id'] == ref]
        result['by_reference'].append({'method': method, 'horizon': h, 'reference_id': ref,
                                       'units': counts[ref], **accumulator.compute(), **pool_identity(rows)})
        path = directory / 'official' / f'{method}-T{h}-{ref}.json'
        write_json_atomic(path, accumulator.export_evidence())
        result['official_evidence'].append({'path': str(path), 'sha256': sha256(path)})
    result['raw_current_metrics'] = []
    for (context, h), accumulator in raw.items():
        if accumulator._updates != 129:
            raise ValueError('bridge raw current denominator differs')
        result['raw_current_metrics'].append({'context': context, 'horizon': h, **accumulator.compute()})
        path = directory / 'official' / f'raw-{context}-T{h}.json'
        write_json_atomic(path, accumulator.export_evidence())
        result['official_evidence'].append({'path': str(path), 'sha256': sha256(path)})
    if result['forward_count'] != 1032 or result['shared_prefixes'] != 258:
        raise ValueError('bridge forward coverage differs')
    result.update(status='COMPLETE', elapsed_seconds=time.perf_counter() - started)
    write_json_atomic(directory / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data-root', 'bundle', 'directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    row = run(args.root, args.data_root, args.bundle, args.directory)
    print(json.dumps({k: row[k] for k in ('status', 'arm', 'step', 'completed_units', 'forward_count')}))
