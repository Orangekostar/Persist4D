"""Fixed-order shared-history diagnostics; labels never enter prediction state."""

from collections import Counter
from contextlib import contextmanager
import csv
from dataclasses import fields
import json
from pathlib import Path
import time

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import ARTIFACTS, digest, read_json, sha256


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    columns = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: 'NA' if row.get(key) is None else json.dumps(row[key], sort_keys=True)
                if isinstance(row.get(key), (dict, list, tuple)) else row.get(key) for key in columns})


def published_target_iou(official, *, source_query, target, target_class):
    """Only candidates descended from this query with the correct class count."""
    selected = ((official.source_query_ids == source_query) & (official.pred_classes == target_class)).nonzero(as_tuple=True)[0]
    best = 0.
    for column in selected.tolist():
        prediction = official.latest_stage_masks[:, column].cpu().bool()
        intersection = int((prediction & target.cpu()).sum())
        union = int((prediction | target.cpu()).sum())
        best = max(best, intersection/max(1, union))
    return best


def clone_runtime(runtime):
    from models.task_memory_state import TaskMemoryState
    from scripts.ogr_runtime import ObservationEpisodeRuntime

    clone = ObservationEpisodeRuntime(runtime.spec, class_count=runtime.state.class_count,
                                      device=runtime.state.embedding.device)
    clone.state = TaskMemoryState(*(tensor.clone() for tensor in runtime.state.tensors()), config=runtime.state.config)
    clone.geometry = runtime.geometry.clone()
    clone.observation_state = runtime.observation_state.clone()
    clone.history = list(runtime.history)
    return clone


def prediction_state_fingerprint(runtime):
    """Small decision state only; no dense point features or labels are hashed."""
    import hashlib
    import torch

    result = hashlib.sha256()
    for sidecar in (runtime.state, runtime.geometry, runtime.observation_state):
        for tensor in sidecar.tensors():
            value = tensor.detach().cpu().contiguous()
            result.update(str((tuple(value.shape), value.dtype)).encode())
            result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    result.update(json.dumps(runtime.history).encode())
    return result.hexdigest()


def reliable_segments(stage):
    import torch
    from models.task_memory_supervision import ambiguous_identity_ids

    meta, target = stage.stage_meta[0], stage.batch[0].target_full[0]
    current = meta.local_stage_ids.cpu() == len(meta.scan_ids_in_window)-1
    p2s = meta.full_resolution_point2segment.cpu()[current]
    vertex_ids = stage.loss_labels.cpu()[current, 1].long()
    reliable = set(target['ids'][target['masks'][:, current].any(1)].tolist())
    reliable -= set(ambiguous_identity_ids(target.get('ambiguities')))
    unique, inverse = vertex_ids.unique(sorted=True, return_inverse=True)
    count = meta.segment_stage_ids.numel()
    histogram = torch.bincount(p2s*unique.numel()+inverse,
        minlength=count*unique.numel()).reshape(count, -1)
    totals = histogram.sum(1)
    maximum, index = histogram.max(1)
    majority = unique[index]
    unambiguous = (histogram == maximum[:, None]).sum(1) == 1
    purity = maximum.float()/totals.clamp_min(1)
    return dict(current=current, point2segment=p2s, vertex_ids=vertex_ids,
        reliable=reliable, totals=totals, histogram=histogram, identity_columns={int(identity): i for i, identity in enumerate(unique)},
        majority=majority, unambiguous=unambiguous, purity=purity, target=target)


def target_flow(output, finished, stage, runtime, ledger, *, method, class_mapper, intervention='NONE'):
    import torch

    meta = stage.stage_meta[0]
    truth = reliable_segments(stage)
    target = truth['target']
    classes = output['pred_logits'][0, :, :-1].argmax(-1).detach().cpu()
    full_masks = (output['pred_masks'][0].detach().cpu()[truth['point2segment']] >= 0)
    rows, targets = [], {}
    for retrieval in output['mpr_retrieval']:
        slots = retrieval['source_slots'].cpu().tolist()
        current_segments = retrieval['current_segment_indices'].cpu()
        anchors = retrieval['anchor_segment_indices'].cpu()
        collisions = Counter(anchors.tolist())
        scores = retrieval['logits'].detach().cpu()
        for query, slot in enumerate(slots):
            logical = int(runtime.state.logical_ids[0, slot])
            generation = int(runtime.state.generations[0, slot])
            identity = ledger.lookup(logical, generation)
            bound = identity is not None and identity[0] == meta.reference_id
            identity_id = int(identity[1]) if bound else None
            evaluable = bound and identity_id in truth['reliable']
            positive = ((truth['majority'] == identity_id) & truth['unambiguous'] & (truth['purity'] >= .5)
                        & (truth['totals'] > 0)) if evaluable else torch.zeros_like(truth['unambiguous'])
            positive_segments = current_segments[positive[current_segments]]
            top = scores[query].argsort(descending=True, stable=True)[:5]
            top_segments = current_segments[top]
            anchor = int(anchors[query])
            neighbors = current_segments[retrieval['neighbor_indices'][query].cpu()]
            denominator = int(truth['totals'][neighbors].sum())
            purity = None
            if evaluable:
                column = truth['identity_columns'][identity_id]
                purity = float(truth['histogram'][neighbors, column].sum())/max(1, denominator)
            raw_iou = best_iou = published_iou = None
            target_label = None
            if evaluable:
                index = int((target['ids'] == identity_id).nonzero(as_tuple=True)[0][0])
                mask = target['masks'][index, truth['current']].cpu().bool()
                target_label = int(target['labels'][index])
                intersection = (full_masks & mask[:, None]).sum(0).float()
                union = (full_masks | mask[:, None]).sum(0).clamp_min(1)
                ious = intersection/union
                raw_iou = float(ious[query])
                same_class = classes == target_label
                best_iou = float(ious[same_class].max()) if same_class.any() else 0.
                published_iou = published_target_iou(finished['official'], source_query=query,
                    target=mask, target_class=class_mapper(target_label))
                targets[query] = dict(instance_id=identity_id, mask=mask, label=target_label,
                    positive_segments=positive_segments, source_slot=slot)
            kept = bool((finished['official'].source_query_ids == query).any())
            row = dict(method=method, context='SHARED_M0_HISTORY_SINGLE_TERMINAL_STEP', intervention=intervention,
                input_id=runtime.spec.input_id, reference_id=meta.reference_id, horizon=runtime.spec.horizon,
                scan_uuid=meta.scan_ids_in_window[-1], absolute_stage=meta.absolute_stage_index,
                source_slot=slot, source_logical_id=logical, source_generation=generation, source_query_id=query,
                history_binding='RELIABLE' if bound else 'UNKNOWN', gt_instance_id=identity_id,
                current_target_evaluable=evaluable, target_label=target_label,
                positive_segment_count=int(positive_segments.numel()), anchor_segment=anchor,
                top1_segment=int(top_segments[0]), top5_segments=top_segments.tolist(),
                top1_hit=bool(positive[anchor]) if evaluable else None,
                top5_hit=bool(positive[top_segments].any()) if evaluable else None,
                point_weighted_neighborhood_purity=purity, anchor_prompt_count=collisions[anchor],
                raw_target_IoU=raw_iou, raw_sameclass_best_IoU=best_iou,
                official_retained=kept, published_target_IoU=published_iou,
                published_logical_id=int(finished['commit'].query_to_logical_id[0, query]) if kept else None,
                confidence=float(runtime.state.confidence[0, slot]), last_seen=int(runtime.state.last_seen[0, slot]),
                canonical_entity_event=False)
            rows.append(row)
    grouped = {}
    for row in rows:
        if row['current_target_evaluable']:
            grouped.setdefault(row['gt_instance_id'], []).append(row)
    for entries in grouped.values():
        canonical = min(entries, key=lambda row: (-row['confidence'], -row['last_seen'],
            row['source_logical_id'], row['source_generation'], row['source_slot']))
        canonical['canonical_entity_event'] = True
    return rows, targets, truth


@contextmanager
def anchor_reference(model, *, query, global_segment, current_segments):
    """Replace one retrieved location using the original nearest32 rule."""
    import torch
    from models.revisit_prompt_modules import nearest_indices, geometry6

    original = model.mpr_prompt.retrieve
    local = int((current_segments.cpu() == global_segment).nonzero(as_tuple=True)[0][0])
    def retrieve(source, features, centers):
        result = original(source, features, centers)
        neighbors = nearest_indices(centers, centers[local])
        replacements = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in result.items()}
        replacements['anchor_indices'][query] = local
        replacements['neighbor_indices'][query] = neighbors
        replacements['centers'][query] = centers[local]
        weights = result['logits'][query, neighbors].softmax(-1)
        replacements['z'][query] = (weights[:, None]*features[neighbors]).sum(0)
        replacements['geometry'][query] = geometry6(centers[neighbors])[0]
        replacements['geometry_valid'][query] = True
        return replacements
    model.mpr_prompt.retrieve = retrieve
    try:
        yield
    finally:
        model.mpr_prompt.retrieve = original


def shared_history_worker(args, assets, *, final=False):
    from scripts.ogr_campaign import require_charged_worker
    event = require_charged_worker()
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.observation_grounded_episode import ObservationEpisodeLoader, ObservationEpisodeSpec
    from models.revisit_prompt_loss import TrainingLabelLedger, bind_training_births
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.ogr_assets import build_system, configure_runtime, frozen_numeric_policy, numeric_sources
    from scripts.ogr_evaluation import load_endpoint
    from scripts.ogr_runtime import ObservationEpisodeRuntime

    configure_runtime(9001)
    records = sorted([record for record in read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records']['CAL']
                      if record['horizon'] in (3, 4, 5)], key=lambda row: row['input_id'])
    if len(records) != 34:
        raise ValueError('the shared-history panel requires exactly 34 fixed inputs')
    output_root = Path(assets['run_root']) / ('mechanism' if final else 'diagnostics')
    if (output_root / 'STATUS.json').exists():
        raise ValueError('diagnostic evidence exists; never overwrite an attempt')
    output_root.mkdir(parents=True, exist_ok=True)
    m0, config, _ = build_system(assets, variant='M0', seed=246)
    m0.to('cuda:0').eval()
    if final:
        selection = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')
        endpoints = {variant: selection['endpoints'][variant]['point']['step'] for variant in ('R-ID', 'R-QFB', 'R-OBS')}
    else:
        endpoints = {'M4': 200}
    # Only one endpoint is resident alongside M0; no extra GPU is consumed.
    loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
    all_rows, anchors, interventions, completed = [], [], [], []
    shared_state_bindings = {}
    status = dict(status='RUNNING', event_id=event, completed_inputs=0, expected_inputs=34,
                  scope='SHARED_M0_HISTORY_SINGLE_TERMINAL_STEP', final_endpoints=final)
    write_json_atomic(output_root / 'STATUS.json', status)
    try:
        for variant, step in endpoints.items():
            system, _, checkpoint_sha, _ = load_endpoint(assets, variant=variant, seed=146 if variant == 'M4' else 246, step=step)
            system.to('cuda:0').eval()
            for record in records:
                if time.perf_counter()+30 >= float(__import__('os').environ['OGR_DEADLINE_MONOTONIC']):
                    status.update(status='INCOMPLETE_BUDGET', last_method=variant)
                    break
                spec = ObservationEpisodeSpec.from_record(record)
                runtime = ObservationEpisodeRuntime(spec, class_count=m0.model.num_classes, device='cuda:0')
                ledger = TrainingLabelLedger()
                mapper = build_rio_class_mapper(loader.dataset(spec.dataset_scope), foreground_class_count=m0.model.num_classes-1)
                with frozen_numeric_policy(), torch.no_grad():
                    for stage in loader.stages(spec):
                        stage = stage.to('cuda:0')
                        terminal = stage.stage_meta[0].absolute_stage_index == spec.horizon-1
                        if terminal:
                            state_sha = prediction_state_fingerprint(runtime)
                            previous = shared_state_bindings.setdefault(spec.input_id, state_sha)
                            if state_sha != previous:
                                raise ValueError('the shared M0 pre-step decision state differs between endpoints')
                            rng = stable_seed(9001, spec.input_id, 'shared-terminal')
                            endpoint_runtime = clone_runtime(runtime)
                            with isolated_rng(rng, cuda_devices=(0,)):
                                output = endpoint_runtime.forward(system, stage)
                                finished = endpoint_runtime.finish(system, output, stage, class_mapper=mapper)
                            # Use the pre-step state for source binding, never the intervened commit.
                            rows, targets, truth = target_flow(output, finished, stage, runtime, ledger,
                                method='M4-200' if variant == 'M4' else variant, class_mapper=mapper)
                            all_rows.extend(rows)
                            if not final:
                                choices = sorted([row for row in rows if row['current_target_evaluable']
                                    and row['positive_segment_count'] > 0], key=lambda row:
                                    (row['source_logical_id'], row['source_generation'], row['source_slot']))[:2]
                                if not choices:
                                    anchors.append(dict(input_id=spec.input_id, method='M4-200', status='NA_NO_RELIABLE_POSITIVE_PROMPT',
                                        diagnostic='GT_ASSISTED_DIAGNOSTIC', raw_target_IoU=None, published_target_IoU=None))
                                for row in choices:
                                    query = row['source_query_id']
                                    candidates = targets[query]['positive_segments'].tolist()
                                    anchor = min(candidates, key=lambda segment: (-float(truth['purity'][segment]), segment))
                                    intervention_runtime = clone_runtime(runtime)
                                    current_segments = output['mpr_retrieval'][0]['current_segment_indices']
                                    with isolated_rng(rng, cuda_devices=(0,)), anchor_reference(system.model,
                                            query=query, global_segment=anchor, current_segments=current_segments):
                                        changed = intervention_runtime.forward(system, stage)
                                        changed_finish = intervention_runtime.finish(system, changed, stage, class_mapper=mapper)
                                    changed_rows, _, _ = target_flow(changed, changed_finish, stage, runtime, ledger,
                                        method='M4-200', class_mapper=mapper, intervention='GT_ASSISTED_ANCHOR')
                                    found = next(value for value in changed_rows if value['source_query_id'] == query)
                                    anchors.append(dict(**found, status='COMPLETE', diagnostic='GT_ASSISTED_DIAGNOSTIC',
                                        chosen_anchor=anchor, paired_raw_target_IoU=row['raw_target_IoU'],
                                        paired_published_target_IoU=row['published_target_IoU'],
                                        delta_raw_target_IoU=found['raw_target_IoU']-row['raw_target_IoU'],
                                        delta_published_target_IoU=found['published_target_IoU']-row['published_target_IoU']))
                                    del changed, changed_finish
                            elif variant == 'R-OBS':
                                interventions.extend(single_step_interventions(system, runtime, stage, ledger, mapper, rows, rng))
                            del output, finished
                        with isolated_rng(stable_seed(9001, spec.input_id, stage.stage_meta[0].absolute_stage_index), cuda_devices=(0,)):
                            own = runtime.forward(m0, stage)
                            own_finished = runtime.finish(m0, own, stage, publish=False)
                        bind_training_births(ledger, own_finished['commit'], own, stage)
                        del own, own_finished, stage
                completed.append(dict(method=variant, input_id=spec.input_id, checkpoint_sha256=checkpoint_sha))
                status.update(completed_inputs=len(completed), last_input_id=spec.input_id, last_method=variant)
                write_json_atomic(output_root / 'STATUS.json', status)
            del system
            torch.cuda.empty_cache()
        write_csv(output_root / 'TARGET_FLOW.csv', all_rows)
        write_csv(output_root / 'ANCHOR_REFERENCE.csv', anchors, fieldnames=None if anchors else ['input_id', 'status'])
        write_csv(output_root / 'TARGET_INTERVENTIONS.csv', interventions,
                  fieldnames=None if interventions else ['input_id', 'method', 'status', 'reason'])
        summary = dict(status='COMPLETE' if len(completed) == 34*len(endpoints) else 'INCOMPLETE_BUDGET',
            completed=completed, expected=34*len(endpoints), selected_prompt_rows=len(all_rows),
            reliable_current_rows=sum(row['current_target_evaluable'] for row in all_rows),
            positive_segment_rows=sum(row['positive_segment_count'] > 0 for row in all_rows),
            canonical_entity_events=sum(row['canonical_entity_event'] for row in all_rows),
            anchor_interventions=sum(row.get('status') == 'COMPLETE' for row in anchors),
            model_input_contains_label_ledger=False, history_source='M0_PREDICTIONS',
            context='SHARED_M0_HISTORY_SINGLE_TERMINAL_STEP', event_id=event,
            shared_state_bindings=shared_state_bindings, numeric_sources=numeric_sources())
        write_json_atomic(output_root / 'RESULTS.json', summary)
        status.update(status=summary['status'])
        write_json_atomic(output_root / 'STATUS.json', status)
        if not final:
            for name in ('TARGET_FLOW.csv', 'ANCHOR_REFERENCE.csv'):
                (ARTIFACTS / name).write_bytes((output_root / name).read_bytes())
        return summary
    except BaseException as error:
        status.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        write_json_atomic(output_root / 'STATUS.json', status)
        raise


def single_step_interventions(system, runtime, stage, ledger, mapper, baseline, rng):
    import torch
    from datasets.native_long_dataset import isolated_rng

    results = []
    try:
        system.model.ogr_feedback_disabled = True
        clone = clone_runtime(runtime)
        with isolated_rng(rng, cuda_devices=(0,)):
            output = clone.forward(system, stage)
            finished = clone.finish(system, output, stage, class_mapper=mapper)
        rows, _, _ = target_flow(output, finished, stage, runtime, ledger, method='R-OBS',
                                 class_mapper=mapper, intervention='FEEDBACK_OFF')
        before = {row['source_query_id']: row for row in baseline}
        for row in rows:
            paired = before[row['source_query_id']]
            row['delta_published_target_IoU'] = (row['published_target_IoU']-paired['published_target_IoU']
                if row['published_target_IoU'] is not None and paired['published_target_IoU'] is not None else None)
        results.extend(rows)
    finally:
        system.model.ogr_feedback_disabled = False
    choices = sorted([row for row in baseline if row['current_target_evaluable'] and row['canonical_entity_event']],
        key=lambda row: (row['source_logical_id'], row['source_generation'], row['source_slot']))
    pair = next(((left, right) for i, left in enumerate(choices) for right in choices[i+1:]
        if left['target_label'] == right['target_label'] and left['gt_instance_id'] != right['gt_instance_id']), None)
    if pair is None:
        results.append(dict(input_id=runtime.spec.input_id, method='R-OBS', intervention='SAME_CLASS_SOURCE_SWAP',
                            status='NA_NO_RELIABLE_SAME_CLASS_PAIR'))
        return results
    left, right = pair
    selected = system.model.mpr_prompt is not None and len(baseline)
    permutation = torch.arange(selected, device=runtime.state.embedding.device)
    a, b = left['source_query_id'], right['source_query_id']
    permutation[a], permutation[b] = permutation[b].clone(), permutation[a].clone()
    clone = clone_runtime(runtime)
    with isolated_rng(rng, cuda_devices=(0,)):
        output = clone.forward(system, stage, history_permutation=[permutation])
        finished = clone.finish(system, output, stage, class_mapper=mapper)
    rows, _, _ = target_flow(output, finished, stage, runtime, ledger, method='R-OBS',
        class_mapper=mapper, intervention='SAME_CLASS_SOURCE_SWAP')
    for row in rows:
        row['status'] = 'COMPLETE'
        row['swapped_query_pair'] = [a, b]
        row['logical_commit_state_permuted'] = False
    results.extend(rows)
    return results


def diagnose(args, assets):
    if args.worker == 'diagnostics':
        if args.panel in ('p1', 'mechanism'):
            return shared_history_worker(args, assets, final=args.panel == 'mechanism')
        if args.panel == 'profile':
            return profile_worker(args, assets)
        if args.panel == 'repeat':
            return numeric_panel_worker(args, assets)
        if args.panel == 'causality':
            return causality_worker(args, assets)
        raise ValueError('unknown diagnostic panel')
    from scripts.ogr_campaign import charged_launch
    root = Path(assets['run_root']) / 'diagnostics'
    if args.attempt:
        repair = read_json(Path(assets['run_root']) / 'repairs/MISSING_EPISODE_SPEC.json')
        if repair['same_error_retry_limit'] != 1 or not (root / 'attempt0-failed/STATUS.json').is_file():
            raise ValueError('the one diagnostic retry requires its exact preserved failure and repair')
    if (root / 'RESULTS.json').is_file():
        result = read_json(root / 'RESULTS.json')
        if result['status'] == 'COMPLETE' and result['expected'] == 34 and len(result['completed']) == 34:
            return dict(status='REUSED_COMPLETE', evidence_sha256=sha256(root / 'RESULTS.json'))
        raise ValueError('existing diagnostic is incomplete; review its specific failure before another attempt')
    suffix = ':fixed-retry1' if args.attempt else ''
    return charged_launch(assets, job_key='diagnostics:P1:shared-M0:M4-200'+suffix, phase='D_DIAGNOSTIC',
        reservation=1.5, worker_args=['--worker', 'diagnostics', '--panel', 'p1', '--attempt', str(args.attempt)])


def run_final_panels(assets):
    from scripts.ogr_campaign import charged_launch
    from scripts.ogr_selection import journal_for
    from scripts.ogr_assets import numeric_sources
    root = Path(assets['run_root'])
    results = []

    def reusable(path):
        if not path.is_file():
            return False
        value = read_json(path)
        if value.get('status') != 'COMPLETE' or value.get('numeric_sources') != numeric_sources():
            raise ValueError('an existing final panel is incomplete or has a different frozen source')
        return True

    for panel, phase, reserve in (('mechanism', 'D_DIAGNOSTIC', .4), ('profile', 'F_PROFILE_CONFIRM', 1.5)):
        folder = 'mechanism' if panel == 'mechanism' else 'profile'
        if reusable(root / folder / 'RESULTS.json'):
            results.append(dict(status='REUSED_EXACT_COMPLETE', panel=panel))
            continue
        result = charged_launch(assets, job_key=f'panel:{panel}', phase=phase, reservation=reserve,
            worker_args=['--worker', 'diagnostics', '--panel', panel])
        results.append(result)
        if result['exit_code']:
            raise RuntimeError('a required final panel failed; preserve evidence and diagnose the specific error')
    for replicate in (0, 1):
        if reusable(root / 'numeric/final' / f'process-{replicate}/RESULTS.json'):
            results.append(dict(status='REUSED_EXACT_COMPLETE', panel='repeat', replicate=replicate))
            continue
        result = charged_launch(assets, job_key=f'panel:repeat:final:{replicate}', phase='I_INTEGRATION_REPAIR',
            reservation=.3, worker_args=['--worker', 'diagnostics', '--panel', 'repeat', '--mode', 'formal',
                                         '--replicate', str(replicate)])
        results.append(result)
        if result['exit_code']:
            raise RuntimeError('a final numeric process failed; preserve its attempt')
    compare_numeric_panel(assets, final=True)
    journal_for(assets).export(ARTIFACTS)
    return results


@contextmanager
def measured_forward_components(model):
    """Scoped synchronized wall timings; nested regional time is reported separately."""
    import torch

    totals = dict(backbone_seconds=0., prompt_hook_seconds=0., feedback_hook_seconds=0., regional_seconds=0.)
    originals = {}
    for owner, name, key in ((model.backbone, 'forward', 'backbone_seconds'),
        (model, 'before_first_mask', 'prompt_hook_seconds'),
        (model, 'after_decoder_stage', 'feedback_hook_seconds'), (model, '_regional', 'regional_seconds')):
        original = getattr(owner, name)
        originals[owner, name] = original
        def wrapped(*args, _original=original, _key=key, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = _original(*args, **kwargs)
            torch.cuda.synchronize()
            totals[_key] += time.perf_counter()-started
            return value
        setattr(owner, name, wrapped)
    try:
        yield totals
    finally:
        for (owner, name), original in originals.items():
            setattr(owner, name, original)


def profile_worker(args, assets):
    from scripts.ogr_campaign import require_charged_worker
    event = require_charged_worker()
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.observation_grounded_episode import ObservationEpisodeLoader, ObservationEpisodeSpec
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.ogr_assets import configure_runtime, frozen_numeric_policy, numeric_sources
    from scripts.ogr_evaluation import load_endpoint
    from scripts.ogr_runtime import ObservationEpisodeRuntime

    configure_runtime(9001)
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    if plan['status'] != 'LOCKED':
        return dict(status='NOT_RUN_NO_FORMAL_ENDPOINTS', event_id=event)
    selection = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')
    panel = sorted([record for record in read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records']['CAL']
                    if record['horizon'] >= 3], key=lambda row: row['input_id'])[:6]
    endpoints = [('M0', 246, 0), ('M4', 146, 200)]+[(variant, 246, selection['endpoints'][variant]['point']['step'])
        for variant in ('R-TARGET', 'R-ID', 'R-QFB', 'R-OBS')]+[('FH-COMMIT0', 246, 0)]
    rows = []
    for variant, seed, step in endpoints:
        system, config, _, _ = load_endpoint(assets, variant=variant, seed=seed, step=step)
        system.to('cuda:0').eval()
        loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
        for repeat in range(4):
            for record in panel:
                policy = 'full_prefix' if variant == 'FH-COMMIT0' else 'local_pair'
                spec = ObservationEpisodeSpec.from_record(record, window_policy=policy)
                runtime = ObservationEpisodeRuntime(spec, class_count=system.model.num_classes, device='cuda:0')
                mapper = build_rio_class_mapper(loader.dataset(spec.dataset_scope), foreground_class_count=system.model.num_classes-1)
                iterator = iter(loader.stages(spec))
                with frozen_numeric_policy(), torch.no_grad(), isolated_rng(stable_seed(9001, spec.input_id), cuda_devices=(0,)):
                    episode_start = time.perf_counter()
                    for stage_index in range(spec.horizon):
                        started = time.perf_counter()
                        stage = next(iterator).to('cuda:0')
                        torch.cuda.synchronize()
                        loaded = time.perf_counter()
                        torch.cuda.reset_peak_memory_stats()
                        with measured_forward_components(system.model) as components:
                            output = runtime.forward(system, stage)
                        torch.cuda.synchronize()
                        forwarded = time.perf_counter()
                        finished = runtime.finish(system, output, stage, class_mapper=mapper, measure=True)
                        torch.cuda.synchronize()
                        if repeat:
                            rows.append(dict(method=variant, adapter_seed=seed, step=step, repeat=repeat,
                                input_id=spec.input_id, horizon=spec.horizon, stage=stage_index,
                                io_seconds=loaded-started, forward_seconds=forwarded-loaded,
                                **components, **finished['timings'],
                                prompt_readout_feedback_seconds=components['prompt_hook_seconds']+components['feedback_hook_seconds'],
                                stage_seconds=time.perf_counter()-started, episode_cumulative_seconds=time.perf_counter()-episode_start,
                                allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
                                **{f'{key}_state_bytes': value for key, value in runtime.state_nbytes().items()},
                                occupied_slots=int(runtime.state.occupied.sum()), history_output_storage='not retained in decision-state profile',
                                logging='status only; no dense tensor fingerprints', physical_gpu=__import__('os').environ['CUDA_VISIBLE_DEVICES']))
                        del output, finished, stage
        del system
        torch.cuda.empty_cache()
    write_csv(ARTIFACTS / 'PROFILE.csv', rows)
    result = dict(status='COMPLETE', event_id=event, configurations=7, warmup_passes=1, measured_passes=3,
        panel_inputs=[record['input_id'] for record in panel], rows=len(rows),
        timing_scope='synchronized GPU wall time; regional_seconds is nested, not additive with hooks',
        numeric_sources=numeric_sources())
    write_json_atomic(Path(assets['run_root']) / 'profile/RESULTS.json', result)
    return result


def numeric_panel_worker(args, assets):
    from scripts.ogr_campaign import require_charged_worker
    event = require_charged_worker()
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.observation_grounded_episode import ObservationEpisodeLoader
    from scripts.ogr_assets import build_system, configure_runtime, frozen_numeric_policy, numeric_sources
    from scripts.ogr_evaluation import load_endpoint, replay_episode
    from scripts.evaluate_revisit_prompt import official_compute_nullable
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from trainer.observation_grounded_trainer import load_adapter_state

    configure_runtime(9001)
    final = args.mode == 'formal'
    methods = ['R-OBS']
    selection = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json') if final else None
    if final:
        methods.append(selection['control']['point']['variant'])
    panel = sorted([record for record in read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records']['CAL']
                    if record['horizon'] >= 3], key=lambda row: row['input_id'])[:6]
    root = Path(assets['run_root']) / 'numeric' / ('final' if final else 'new-path') / f'process-{args.replicate}'
    if root.exists():
        raise ValueError('finite numeric panel attempt already exists')
    root.mkdir(parents=True)
    results = []
    for method in methods:
        if final:
            step = selection['endpoints'][method]['point']['step']
            system, config, checkpoint_sha, _ = load_endpoint(assets, variant=method, seed=246, step=step)
        else:
            path = Path(assets['run_root']) / 'probe/R-OBS-s246/resume-latest.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['step'] != 20 or payload['identity']['mode'] != 'probe':
                raise ValueError('new-path repeat requires the real 20-update TRAIN probe')
            system, config, _ = build_system(assets, variant='R-OBS', seed=246)
            load_adapter_state(system, payload['state_dict'])
            checkpoint_sha, step = sha256(path), 20
        system.to('cuda:0').eval()
        loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
        accumulators, episodes, tensors = {}, [], []
        provenance = dict(checkpoint_sha256=checkpoint_sha, config_hash=digest(str(config)),
                          protocol_manifest_hash=sha256(ARTIFACTS / 'DATA_MANIFEST.json'))
        with frozen_numeric_policy(), torch.no_grad():
            for record in panel:
                with isolated_rng(stable_seed(9001, record['input_id']), cuda_devices=(0,)):
                    episode = replay_episode(system, loader, record, provenance=provenance, capture_panel=True)
                horizon = record['horizon']
                if horizon not in accumulators:
                    accumulators[horizon] = OfficialMetricAccumulator(mode='strict_online', dataset_spec=assets['taxonomy_path'])
                accumulators[horizon].update(episode['prediction'], episode['target'])
                episodes.append(dict(input_id=record['input_id'], discrete=episode['discrete']))
                tensors.append(dict(input_id=record['input_id'], stages=episode['panel_tensors']))
        metrics = {str(horizon): official_compute_nullable(metric) for horizon, metric in accumulators.items()}
        official_evidence = []
        for horizon, accumulator in accumulators.items():
            path = root / f'{method}-T{horizon}.official.json'
            write_json_atomic(path, accumulator.export_evidence())
            official_evidence.append(dict(path=path.name, horizon=horizon, sha256=sha256(path)))
        tensor_path = root / f'{method}-panel-tensors.pt'
        torch.save(tensors, tensor_path)
        results.append(dict(method=method, step=step, checkpoint_sha256=checkpoint_sha, metrics=metrics, episodes=episodes,
            panel_tensors=dict(path=tensor_path.name, sha256=sha256(tensor_path)), official_evidence=official_evidence))
        del system
        torch.cuda.empty_cache()
    result = dict(status='COMPLETE', event_id=event, process=args.replicate, final=final,
        scope='finite six-input panel; not full CAL selection', results=results, numeric_sources=numeric_sources(require_complete=False))
    write_json_atomic(root / 'RESULTS.json', result)
    return dict(status='COMPLETE', process=args.replicate, final=final)


def compare_numeric_panel(assets, *, final=False):
    import torch
    root = Path(assets['run_root']) / 'numeric' / ('final' if final else 'new-path')
    panels = [read_json(root / f'process-{index}/RESULTS.json') for index in (0, 1)]
    if any(panel.get('status') != 'COMPLETE' or panel.get('final') != final for panel in panels):
        raise ValueError('two complete processes from the same finite-panel phase are required')
    if panels[0]['numeric_sources'] != panels[1]['numeric_sources']:
        raise ValueError('finite-panel processes used different frozen source identities')
    comparisons, confounded = [], False
    for left, right in zip(panels[0]['results'], panels[1]['results'], strict=True):
        if any(left[key] != right[key] for key in ('method', 'step', 'checkpoint_sha256')):
            raise ValueError('finite-panel endpoint identity differs between processes')
        discrete_equal = True
        tensor_hash_differences = 0
        for a, b in zip(left['episodes'], right['episodes'], strict=True):
            if a['input_id'] != b['input_id']:
                raise ValueError('numeric panel fixed input identity differs')
            for x, y in zip(a['discrete'], b['discrete'], strict=True):
                structural = ('stage', 'points', 'point2segment', 'full_resolution_mapping', 'route', 'issued_ids', 'source_query_ids')
                discrete_equal &= all(x[key] == y[key] for key in structural)
                tensor_hash_differences += int(x['logits'] != y['logits'])+int(x['masks'] != y['masks'])
        ranges = {}
        for horizon in left['metrics']:
            a = left['metrics'][horizon]['online_t-mAP']
            b = right['metrics'][horizon]['online_t-mAP']
            ranges[horizon] = None if a is None or b is None else 100*abs(a-b)
        accepted = discrete_equal and all(value is not None and value <= .2 for value in ranges.values())
        confounded |= not accepted
        left_tensors = torch.load(root / 'process-0' / left['panel_tensors']['path'], map_location='cpu', weights_only=True)
        right_tensors = torch.load(root / 'process-1' / right['panel_tensors']['path'], map_location='cpu', weights_only=True)
        differences = []
        for a, b in zip(left_tensors, right_tensors, strict=True):
            for x, y in zip(a['stages'], b['stages'], strict=True):
                differences.append(dict(input_id=a['input_id'], stage=x['stage'],
                    logits_max_abs=float((x['logits']-y['logits']).abs().max()),
                    mask_logits_max_abs=float((x['masks']-y['masks']).abs().max())))
        comparisons.append(dict(method=left['method'], discrete_equal=discrete_equal,
            tensor_summary_differences=tensor_hash_differences, AP_range_pp=ranges, accepted=accepted,
            actual_tensor_differences=differences,
            tensor_difference_scope='actual class and segment-mask logit maximum absolute differences on the fixed panel'))
    result = dict(status='NUMERICALLY_CONFOUNDED' if confounded else 'FINITE_PANEL_WITHIN_LIMIT',
        final=final, comparisons=comparisons, full_global_determinism_claim=False,
        process_bindings=[sha256(root / f'process-{index}/RESULTS.json') for index in (0, 1)])
    write_json_atomic(ARTIFACTS / ('FINAL_NUMERIC_PANEL.json' if final else 'NEW_PATH_NUMERIC_PANEL.json'), result)
    return result


def causality_worker(args, assets):
    """One real pair and one empty/unknown path, with label-only mutation."""
    from scripts.ogr_campaign import require_charged_worker
    event = require_charged_worker()
    import copy
    import torch
    from datasets.native_long_dataset import isolated_rng, stable_seed
    from datasets.observation_grounded_episode import ObservationEpisodeLoader, ObservationEpisodeSpec
    from scripts.ogr_assets import build_system, configure_runtime, frozen_numeric_policy
    from scripts.ogr_runtime import ObservationEpisodeRuntime

    configure_runtime(9001)
    system, config, _ = build_system(assets, variant='R-OBS', seed=246)
    system.to('cuda:0').eval()
    loader = ObservationEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
    record = min((row for row in read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records']['CAL'] if row['horizon'] == 2),
                 key=lambda row: row['input_id'])
    spec = ObservationEpisodeSpec.from_record(record)
    stages = list(loader.stages(spec))
    stage = stages[-1].to('cuda:0')
    original = ObservationEpisodeRuntime(spec, class_count=system.model.num_classes, device='cuda:0')
    # Build a real, prediction-only nonempty history before permuting any labels.
    with frozen_numeric_policy(), torch.no_grad():
        first = stages[0].to('cuda:0')
        output = original.forward(system, first)
        original.finish(system, output, first, publish=False)
        changed = copy.deepcopy(stage)
        for target in (*changed.batch[1], *changed.batch[0].target_full, changed.common_truth):
            for name in ('ids', 'labels', 'masks', 'segment_mask'):
                if name in target:
                    target[name] = target[name].flip(0)
        changed.loss_labels[:, 1] = changed.loss_labels[:, 1].flip(0)
        with isolated_rng(stable_seed(9001, 'causality'), cuda_devices=(0,)):
            left = clone_runtime(original).forward(system, stage)
        with isolated_rng(stable_seed(9001, 'causality'), cuda_devices=(0,)):
            right = clone_runtime(original).forward(system, changed)
        equal = (torch.equal(left['pred_logits'], right['pred_logits'])
                 and torch.equal(left['pred_masks'][0], right['pred_masks'][0]))
        if not equal:
            raise ValueError('prediction changed after label-only permutation')
        empty = ObservationEpisodeRuntime(spec, class_count=system.model.num_classes, device='cuda:0')
        unknown = clone_runtime(original)
        unknown.observation_state.valid.zero_()
        empty_result = empty.forward(system, first)
        unknown_result = unknown.forward(system, stage)
    meta = stage.stage_meta[0]
    result = dict(status='PASS', event_id=event, input_id=spec.input_id,
        real_pair_nonempty_gt=bool(stage.batch[0].target_full[0]['ids'].numel()),
        prediction_invariant_to_gt_permutation=equal, real_empty_history_checked=True,
        unknown_correspondence_all_null=bool(unknown_result['ogr_final'][0]['probabilities'][:, -1].eq(1).all()),
        points=meta.local_stage_ids.numel(), full_point2segment_count=meta.full_resolution_point2segment.numel(),
        current_points=int((meta.local_stage_ids == 1).sum()), scan_uuids=list(meta.scan_ids_in_window),
        occupied_history_slots=int(original.state.occupied.sum()))
    if not result['real_pair_nonempty_gt'] or not result['unknown_correspondence_all_null']:
        raise ValueError('causality check did not exercise the required real paths')
    write_json_atomic(ARTIFACTS / 'REAL_DATA_CAUSALITY.json', result)
    return result
