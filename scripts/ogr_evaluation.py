"""Own-history OGR deployment with original candidate and pooled-AP semantics."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import ARTIFACTS, build_system, digest, frozen_numeric_policy, read_json, sha256
from scripts.evaluate_revisit_prompt import (
    current_metric_truth, current_mask_diagnostic, official_compute_nullable, pool_official_states,
)


def small_observation_fingerprint(observation):
    """Bind final query observations; intentionally do not copy full mask tensors."""
    result = hashlib.sha256()
    for name in ('features', 'class_prob', 'confidence', 'valid', 'current_supported', 'previous_supported'):
        value = getattr(observation, name).detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def replay_episode(system, loader, record, *, provenance, measure_only=False, capture_panel=False):
    import torch
    from datasets.observation_grounded_episode import ObservationEpisodeSpec
    from scripts.evaluate_persist4d import _metric_target
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.ogr_runtime import ObservationEpisodeRuntime, trajectory_sidecar
    from scripts.system_comparison_metrics import match_identity_update, compute_deployment_identity_metrics
    from scripts.system_comparison_v2_inference import OfficialCandidateTrajectoryAccumulator

    spec = ObservationEpisodeSpec.from_record(record,
        window_policy='full_prefix' if record.get('method') == 'FH-COMMIT0' else 'local_pair')
    dataset = loader.dataset(spec.dataset_scope)
    mapper = build_rio_class_mapper(dataset, foreground_class_count=system.model.num_classes-1)
    runtime = ObservationEpisodeRuntime(spec, class_count=system.model.num_classes, device='cuda:0')
    trajectory = OfficialCandidateTrajectoryAccumulator(score_reducer='mean') if not measure_only else None
    identities, structures, timing, masks_diagnostics, discrete, panel_tensors = [], [], [], [], [], []
    iterator = iter(loader.stages(spec))
    started_episode = time.perf_counter()
    for stage_index in range(spec.horizon):
        started = time.perf_counter()
        stage = next(iterator).to('cuda:0')
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        output = runtime.forward(system, stage)
        torch.cuda.synchronize()
        forwarded = time.perf_counter()
        finished = runtime.finish(system, output, stage, class_mapper=mapper, measure=True)
        torch.cuda.synchronize()
        published = time.perf_counter()
        meta = stage.stage_meta[0]
        current = meta.local_stage_ids == len(meta.scan_ids_in_window)-1
        if not measure_only:
            fingerprint = small_observation_fingerprint(finished['observation'])
            source_order = record.get('source_unit', '').rsplit(':', 1)[-1]
            order = source_order if source_order in ('canonical', 'reverse', 'sha256_seed45') else 'canonical'
            sidecar = trajectory_sidecar(finished['official'], meta, history=runtime.history,
                provenance={**provenance, 'source_raw_observation_fingerprint': fingerprint}, order_id=order)
            trajectory.add_stage(sidecar, finished['track_step'])
            target = _metric_target(stage.common_truth, dataset)
            current_target = current_metric_truth(target, stage_index)
            candidate = finished['official'].prediction(latest_only=True)
            current_prediction = {key: candidate[key] for key in ('pred_masks', 'pred_scores', 'pred_classes')}
            query_ids = (finished['commit'].query_to_logical_id[0] >= 0).nonzero(as_tuple=True)[0]
            classes = finished['observation'].class_prob[0, query_ids, :-1].argmax(-1).cpu()
            classes = torch.tensor([mapper(int(value)) for value in classes], dtype=torch.long)
            prediction_masks = (output['pred_masks'][0][meta.full_resolution_point2segment][:, query_ids] >= 0)
            identities.append(match_identity_update(horizon=stage_index+1, gt_ids=current_target['ids'],
                gt_classes=current_target['labels'], gt_masks=current_target['masks'],
                issued_ids=finished['commit'].query_to_logical_id[0, query_ids].cpu(),
                pred_classes=classes, pred_masks=prediction_masks[current].cpu(), minimum_iou=.5))
            masks_diagnostics.append(dict(stage=stage_index, **current_mask_diagnostic(current_prediction, current_target)))
        structures.append(dict(stage=stage_index, window_scan_ids=list(meta.scan_ids_in_window),
            points=meta.local_stage_ids.numel(), segments=meta.segment_stage_ids.numel(),
            current_points=int(current.sum()), occupied_slots=int(runtime.state.occupied.sum()),
            state_bytes=runtime.state_nbytes()))
        if capture_panel:
            from models.revisit_prompt_modules import tensor_summary
            discrete.append(dict(stage=stage_index,
                points=tensor_summary(stage.batch[0].coordinates),
                point2segment=tensor_summary(meta.point2segment),
                full_resolution_mapping=tensor_summary(meta.full_resolution_point2segment),
                route=finished['route'].query_to_slot.cpu().tolist(),
                issued_ids=finished['commit'].query_to_logical_id.cpu().tolist(),
                source_query_ids=finished['official'].source_query_ids.tolist(),
                logits=tensor_summary(output['pred_logits']), masks=tensor_summary(output['pred_masks'][0])))
            panel_tensors.append(dict(stage=stage_index, logits=output['pred_logits'].detach().cpu(),
                                      masks=output['pred_masks'][0].detach().cpu()))
        timing.append(dict(stage=stage_index, io_seconds=loaded-started, forward_seconds=forwarded-loaded,
            commit_postprocess_seconds=published-forwarded, **finished['timings'],
            end_to_end_seconds=time.perf_counter()-started))
        del output, finished, stage
    result = dict(structures=structures, timings=timing, seconds=time.perf_counter()-started_episode,
                  discrete=discrete, fingerprint_scope='final query observation only; dense masks are not copied for hashing')
    if capture_panel:
        result['panel_tensors'] = panel_tensors
    if not measure_only:
        snapshot = trajectory.snapshot()
        if snapshot.prediction['pred_masks'].shape[0] != target['masks'].shape[1]:
            raise ValueError('OGR committed current vertices differ from causal full-prefix target')
        result.update(prediction=snapshot.prediction, target=target, current_prediction=current_prediction,
            current_target=current_target, identity=compute_deployment_identity_metrics(identities),
            identity_assignments=[dict(horizon=update.horizon, visible_gt_ids=list(update.visible_gt_ids),
                                       assignments=update.assignments) for update in identities],
            current_mask_diagnostics=masks_diagnostics)
    return result


def load_endpoint(assets, *, variant, seed, step):
    import torch
    from scripts.ogr_assets import load_ogr_bundle, PROJECT
    from trainer.observation_grounded_trainer import load_adapter_state

    if variant in ('M0', 'FH-COMMIT0'):
        system, config, audit = build_system(assets, variant=variant, seed=246)
        return system, config, assets['hashes']['parent'], audit
    if variant == 'M4':
        # Original M4-200 is a named baseline only; R arms never load this delta.
        previous = PROJECT.parent / 'persist4d-mpr-v1-formal-20261014'
        manifest = read_json(previous / 'artifacts/revisit_prompt_v1/MODEL_MANIFEST.json')
        row = next(row for row in manifest['assets'] if row.get('variant') == 'M4'
                   and row.get('step') == 200 and row.get('kind') == 'delta')
        path = previous / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('original M4-200 baseline bytes differ')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        system, config, audit = build_system(assets, variant='M4', seed=146)
        load_adapter_state(system, payload['state_dict'])
        return system, config, row['sha256'], audit
    path = Path(assets['run_root']) / 'trajectories' / f'{variant}-s{seed}' / f'delta-{step:06d}.pt'
    system = load_ogr_bundle(path, assets=assets)
    return system, system.config, sha256(path), None


def evaluation_worker(args, assets):
    from scripts.ogr_campaign import require_charged_worker
    event = require_charged_worker()
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.observation_grounded_episode import ObservationEpisodeLoader
    from scripts.ogr_assets import configure_runtime, numeric_sources
    from scripts.revisit_prompt_assets import configuration_digest
    from scripts.native_long_memory_bridge import pool_identity
    from scripts.p6a_metrics import OfficialMetricAccumulator

    configure_runtime(9001)
    timing_only = args.timing_only
    if not timing_only:
        from scripts.ogr_selection import validate_evaluation_request
        validate_evaluation_request(args, assets)
    manifest = read_json(ARTIFACTS / 'DATA_MANIFEST.json')
    inputs = sorted(manifest['records'][args.role], key=lambda record: (record['horizon'], record['input_id']))
    if timing_only:
        inputs = [record for record in inputs if record['horizon'] >= 3][:6]
    folder = Path(assets['run_root']) / ('timing' if timing_only else 'evaluation') / f'{args.variant}-s{args.seed}-u{args.step:06d}' / args.role
    if (folder / 'STATUS.json').exists():
        raise ValueError('evaluation evidence already exists; never overwrite an attempt')
    folder.mkdir(parents=True, exist_ok=True)
    request = dict(variant=args.variant, adapter_seed=args.seed, step=args.step, role=args.role,
                   input_ids=[record['input_id'] for record in inputs], timing_only=timing_only, event_id=event)
    status = dict(status='STARTING', request=request, completed_inputs=0)
    write_json_atomic(folder / 'STATUS.json', status)
    try:
        if timing_only:
            system, config, audit = build_system(assets, variant=args.variant, seed=args.seed)
            checkpoint_sha = '0' * 64
        else:
            system, config, checkpoint_sha, audit = load_endpoint(assets, variant=args.variant, seed=args.seed, step=args.step)
        system.to('cuda:0').eval()
        loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
        provenance = dict(checkpoint_sha256=checkpoint_sha, config_hash=configuration_digest(config, assets),
                          protocol_manifest_hash=sha256(ARTIFACTS / 'DATA_MANIFEST.json'))
        request.update(checkpoint_sha256=checkpoint_sha,
                       data_manifest_sha256=provenance['protocol_manifest_hash'])
        def metric(horizon, kind):
            return OfficialMetricAccumulator(mode='raw_local' if horizon == 1 or kind == 'current' else 'strict_online',
                dataset_spec=assets['taxonomy_path'], min_region_size=int(config.instance_metric.get('min_region_size', 100)))
        expected = Counter(record['horizon'] for record in inputs)
        pooled = {(horizon, kind): metric(horizon, kind) for horizon in expected for kind in ('long', 'current')} if not timing_only else {}
        by_reference, episodes = {}, []
        status.update(status='RUNNING', startup_seconds=time.perf_counter()-float(os.environ['OGR_STARTED_MONOTONIC']))
        write_json_atomic(folder / 'STATUS.json', status)
        with frozen_numeric_policy(), torch.no_grad():
            for record in inputs:
                record = {**record, 'method': args.variant}
                with isolated_rng(stable_seed(9001, record['input_id']), cuda_devices=(0,)):
                    episode = replay_episode(system, loader, record, provenance=provenance, measure_only=timing_only)
                horizon, reference = record['horizon'], record['reference_id']
                if not timing_only:
                    for kind in ('long', 'current'):
                        prediction = episode['prediction' if kind == 'long' else 'current_prediction']
                        target = episode['target' if kind == 'long' else 'current_target']
                        pooled[horizon, kind].update(prediction, target)
                        key = horizon, kind, reference
                        if key not in by_reference:
                            by_reference[key] = metric(horizon, kind)
                        by_reference[key].update(prediction, target)
                episodes.append(dict(input_id=record['input_id'], reference_id=reference, horizon=horizon,
                    **{key: value for key, value in episode.items() if key not in ('prediction', 'target', 'current_prediction', 'current_target')}))
                status.update(completed_inputs=len(episodes), last_input_id=record['input_id'], updated_unix=time.time())
                write_json_atomic(folder / 'STATUS.json', status)
        rows, references, official = [], [], []
        if not timing_only:
            for horizon in sorted(expected):
                long_metrics = official_compute_nullable(pooled[horizon, 'long'])
                current_metrics = official_compute_nullable(pooled[horizon, 'current'])
                rows.append(dict(variant=args.variant, adapter_seed=args.seed, step=args.step, role=args.role,
                    horizon=horizon, status='COMPLETE', AP=long_metrics['raw_local_AP' if horizon == 1 else 'online_t-mAP'],
                    current_AP=current_metrics['raw_local_AP'], expected_inputs=expected[horizon],
                    completed_inputs=expected[horizon], official_metrics=long_metrics, official_current_metrics=current_metrics,
                    identity=pool_identity([episode['identity'] for episode in episodes if episode['horizon'] == horizon])))
                for kind in ('long', 'current'):
                    path = folder / f'official-{kind}-T{horizon}.json'
                    write_json_atomic(path, pooled[horizon, kind].export_evidence())
                    official.append(dict(path=path.name, sha256=sha256(path), horizon=horizon, kind=kind))
            for (horizon, kind, reference), accumulator in sorted(by_reference.items()):
                path = folder / 'references' / f'{reference}-T{horizon}-{kind}.json'
                write_json_atomic(path, accumulator.export_evidence())
                references.append(dict(reference_id=reference, horizon=horizon, kind=kind,
                    metrics=official_compute_nullable(accumulator), evidence=path.relative_to(folder).as_posix(), sha256=sha256(path)))
        write_json_atomic(folder / 'RESULTS.json', dict(status='COMPLETE', request=request, rows=rows,
            references=references, official_evidence=official, episodes=episodes, load_audit=audit,
            numeric_sources=numeric_sources(require_complete=not timing_only)))
        status.update(status='COMPLETE', completed_inputs=len(episodes))
        write_json_atomic(folder / 'STATUS.json', status)
        return status
    except BaseException as error:
        status.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        write_json_atomic(folder / 'STATUS.json', status)
        raise
