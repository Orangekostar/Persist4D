"""Fixed TRAIN/CAL explanations; GT-assisted outputs never enter deployment/selection."""
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.qp_mn_adapter import apply_module, batch_loss
from scripts.qp_mn_evaluation import load_head
from scripts.rescene_task_postprocess import materialize_segment_logits
from scripts.short_module_data import (
    geometry_assignment,
    low_segment_targets,
    quality_labels,
)
from scripts.short_module_evaluation import materialization_system, write_csv
from scripts.short_module_identity import file_digest, identity_digest, source_identity
from scripts.short_module_native import unpack_prediction
from scripts.short_module_screen import read_json, write_json
from scripts.short_module_training import build_sample_plan, load_training_records
from scripts.system_comparison_inference import unpack_bool_matrix


def ranking_counts(target, parent, scores, valid, classes):
    rows = []
    for cls in sorted(set(classes[valid].tolist())):
        keep = valid & (classes == cls)
        y, before, after = target[keep], parent[keep], scores[keep]
        difference = y[:, None] - y[None, :]
        pairs = torch.triu(torch.ones_like(difference, dtype=torch.bool), diagonal=1) & (difference != 0)
        original = (before[:, None] - before[None, :]) * difference
        current = (after[:, None] - after[None, :]) * difference
        rows.append({'class_id': cls, 'comparable_pairs': int(pairs.sum()),
                     'inversions': int((pairs & (current < 0)).sum()),
                     'ties': int((pairs & (current == 0)).sum()),
                     'original_correct_destroyed': int((pairs & (original > 0) & (current < 0)).sum()),
                     'original_correct_tied': int((pairs & (original > 0) & (current == 0)).sum())})
    return rows


def gt_quality_scores(parent, labels):
    known = torch.tensor([s in ('MATCHED', 'VALID_NEGATIVE') for s in labels['status']])
    return torch.where(known, labels['temporal'], parent)


def loose_reachable(lower, upper, original, gt):
    if (lower & ~original).any() or (original & ~upper).any():
        raise ValueError('native materialization bounds do not contain original output')
    return lower | (gt & upper & ~lower)


def quantiles(value):
    if not value.numel():
        return {f'q{p}': None for p in (0, 10, 25, 50, 75, 90, 100)}
    return {f'q{p}': float(torch.quantile(value.float(), p / 100)) for p in (0, 10, 25, 50, 75, 90, 100)}


def ordinary_iou(mask, target):
    return float((mask & target).sum() / (mask | target).sum().clamp_min(1))


def diagnostic_inputs(context):
    c = context
    panel = read_json(c.artifacts / 'diagnostics/TRAIN_PANEL.json')
    ids = {r['input_id'] for r in panel['records']}
    sources = [(c.train_index, [e for e in c.train_index['entries'] if e['input_id'] in ids]),
               (c.eval_index, [e for e in c.eval_index['entries'] if e['record']['role'] == 'CAL'])]
    for index, entries in sources:
        system = materialization_system(index)
        for entry in entries:
            saved = torch.load(Path(index['cache']) / entry['prediction']['file'], map_location='cpu', weights_only=False)
            data = torch.load(Path(index['cache']) / entry['targets']['file'], map_location='cpu', weights_only=False)
            target = {**data['target'], 'masks': unpack_bool_matrix(data['target']['masks'])}
            parent = unpack_prediction(saved['parent'])
            labels = data.get('quality')
            if labels is None:
                labels = quality_labels(parent.prediction(), target, dataset_spec=Path(c.assets['metric_dataset_spec']))
            assignment = data.get('assignment')
            if assignment is None:
                assignment = geometry_assignment(parent.prediction(), target, labels, candidate_keys=saved['candidate_keys'])
            yield entry['record'], saved, parent, target, data, labels, assignment, system


def gradient_diagnostics(context, records, lock):
    c = context
    counts = {e['input_id']: e['audit']['candidate_count'] for e in c.train_index['entries']}
    # Longer prefix only supplies the prescribed nearest later diagnostic batch;
    # no optimizer step beyond the 1500 main trajectory is executed.
    plan = build_sample_plan(c.population, counts, seed=45, updates=2000)
    selected = []
    for anchor in (1, 500, 1500):
        for index in range(anchor - 1, len(plan)):
            draws = plan[index]
            positives = sum(int((records[d['input_id']]['quality']['geometry_valid'][d['candidates']]
                                & (records[d['input_id']]['assignment'][d['candidates']] >= 0)).sum())
                            for d in draws if (records[d['input_id']]['weights'] > 0).any())
            if positives:
                selected.append((anchor, index + 1, draws))
                break
        else:
            raise ValueError('no supervised diagnostic batch in declared finite plan prefix')
    rows = []
    for arm in ('M_C', 'M_N'):
        head = load_head(c, arm, lock['selected_updates'][arm], 45, 'cpu')[0]
        for requested, actual, draws in selected:
            for formula in ('M_C', 'M_N'):
                _, detail, parts = batch_loss(formula, head, records, draws, classes=c.dimensions['classes'], device='cpu')
                norms = {}
                for part in ('shape', 'regularization'):
                    grads = torch.autograd.grad(parts[part], tuple(head.parameters()), retain_graph=True)
                    norms[part] = sum(float(g.square().sum()) for g in grads) ** .5
                rows.append({'head': arm, 'step': lock['selected_updates'][arm], 'loss_formula': formula,
                             'requested_batch': requested, 'actual_batch': actual, **detail,
                             'shape_gradient_norm': norms['shape'], 'regularizer_gradient_norm': norms['regularization'],
                             'shape_to_regularizer_ratio': norms['shape'] / norms['regularization'] if norms['regularization'] else None,
                             'optimizer_updates': 0})
    write_csv(c.artifacts / 'diagnostics/M_GRADIENT_COMPONENTS.csv', rows)
    write_json(c.artifacts / 'diagnostics/M_GRADIENT_BATCHES.json', [
        {'requested': req, 'actual': act, 'draws': draws} for req, act, draws in selected])
    return rows


def run_diagnostics(context):
    c = context
    lock = read_json(c.artifacts / 'selection/CAL_LOCK.json')
    records = load_training_records(c.root, c.train_index)
    train_y = torch.cat([r['quality']['temporal'][r['quality']['valid']] for r in records.values()])
    constant = float(train_y.mean())
    plan = read_json(c.root / 'training/sample_plan_seed45.json')
    occurrences = Counter()
    for draws in plan:
        for draw in draws:
            r = records[draw['input_id']]
            for i in draw['candidates']:
                if r['quality']['geometry_valid'][i] and r['assignment'][i] >= 0 and (r['weights'] > 0).any():
                    occurrences[draw['input_id'], i] += 1
    quality_heads = {(arm, step): load_head(c, arm, step, 45, 'cpu')[0]
                     for arm in ('Q2_F', 'Q_A0', 'Q_P') for step in (0, 500, 1000, 1500)}
    mask_heads = {arm: load_head(c, arm, lock['selected_updates'][arm], 45, 'cpu')[0]
                  for arm in ('M_C', 'M_N')}
    aggregated = defaultdict(list)
    ranking, duplicates, shape_rows, reach_rows = [], [], [], []
    assisted = {}
    def metric(role, h, name):
        key = role, h, name
        if key not in assisted:
            assisted[key] = OfficialMetricAccumulator(mode='raw_local' if h == 1 else 'strict_online',
                            dataset_spec=c.assets['metric_dataset_spec'], min_region_size=100)
        return assisted[key]
    input_counts = Counter()
    with torch.no_grad():
        for record, saved, parent, target, data, labels, assignment, system in diagnostic_inputs(c):
            role, identity, horizon = record['role'], record['input_id'], record['horizon']
            input_counts[role, horizon] += 1
            common = {'role': role, 'reference_id': record['reference_id'], 'input_id': identity, 'H': horizon}
            parent_scores, y, valid = parent.pred_scores, labels['temporal'], labels['valid']
            partial = torch.tensor([v is not None and 0 < v < 1 for v in labels['ignore_proportion']])
            entities = Counter(g for g, status in zip(labels['gt_ids'], labels['status']) if g is not None and status == 'MATCHED')
            predictions = {('B0', 0): parent_scores}
            for (arm, step), head in quality_heads.items():
                predictions[arm, step] = apply_module(arm, head, parent, saved['descriptor'], system=system)['pred_scores']
            for (arm, step), scores in predictions.items():
                aggregated[role, arm, step].append((scores.clone(), y.clone(), valid.clone(), parent_scores.clone(), partial.clone()))
                ranking.extend([{**common, 'module': arm, 'step': step, **r} for r in ranking_counts(y, parent_scores, scores, valid, parent.pred_classes)])
                for i, entity in enumerate(labels['gt_ids']):
                    if entity is None or entities[entity] < 2:
                        continue
                    same_class = parent.pred_classes == parent.pred_classes[i]
                    duplicates.append({**common, 'module': arm, 'step': step, 'retained_index': i,
                        'source_query_id': int(parent.source_query_ids[i]), 'source_class_id': int(parent.source_class_ids[i]),
                        'entity_id': entity, 'duplicates_for_entity': entities[entity], 'target': float(y[i]),
                        'score': float(scores[i]), 'parent_score': float(parent_scores[i]),
                        'class_rank': 1 + int((same_class & (scores > scores[i])).sum()),
                        'parent_class_rank': 1 + int((same_class & (parent_scores > parent_scores[i])).sum())})
            metric(role, horizon, 'B0').update(parent.prediction(), target)
            metric(role, horizon, 'GT_QUALITY_SCORE_DIAGNOSTIC').update(
                {**parent.prediction(), 'pred_scores': gt_quality_scores(parent_scores, labels)}, target)
            soft, desc = parent.soft_evidence, saved['descriptor']
            logits = soft.segment_logits
            lower = materialize_segment_logits(system=system, evidence=soft, segment_logits=logits - 2)
            upper = materialize_segment_logits(system=system, evidence=soft, segment_logits=logits + 2)
            gt_columns = parent.pred_masks.clone()
            for i, entity in enumerate(assignment.tolist()):
                if entity >= 0:
                    gt_columns[:, i] = target['masks'][target['ids'] == entity].any(0)
            loose = loose_reachable(lower, upper, parent.pred_masks, gt_columns)
            metric(role, horizon, 'GT_LOOSE_MASK_DIAGNOSTIC').update({**parent.prediction(), 'pred_masks': loose}, target)
            full_low = soft.low_point2segment[soft.voxel_inverse]
            known = (data['semantic_labels'] >= 0) & (data['semantic_labels'] != 255)
            mask_predictions, deltas = {}, {}
            for arm, head in mask_heads.items():
                delta = head(soft.segment_features, soft.query_features, desc['h'],
                             F.one_hot(soft.source_class_ids, desc['classes']).float(), desc['segment_stages'])
                deltas[arm] = delta
                mask_predictions[arm] = materialize_segment_logits(system=system, evidence=soft, segment_logits=logits + delta)
            for i, entity in enumerate(assignment.tolist()):
                row = {**common, 'retained_index': i, 'source_query_id': int(parent.source_query_ids[i]),
                       'source_class_id': int(parent.source_class_ids[i]), 'assigned_gt': entity if entity >= 0 else None,
                       'shape_usable': bool(labels['geometry_valid'][i]),
                       'supervision_occurrences': occurrences[identity, i] if role == 'TRAIN' else 0,
                       'original_iou': None, 'loose_iou': None, 'loose_gain': None,
                       'strict_strong_error_segments': None, 'valid_segments': None,
                       'strict_strong_error_points': None, 'raw_error_known_points': None,
                       'original_correct_known_points': None, 'original_error_known_points': None,
                       'boundary_segments': int((logits[:, i].abs() == 2).sum())}
                if entity >= 0:
                    gt = gt_columns[:, i]
                    original_iou = ordinary_iou(parent.pred_masks[:, i], gt)
                    loose_iou = ordinary_iou(loose[:, i], gt)
                    if loose_iou + 1e-7 < original_iou:
                        raise ValueError('loose ordinary IoU must not be below original')
                    segment_y, weights = low_segment_targets(low_point2segment=soft.low_point2segment,
                        voxel_inverse=soft.voxel_inverse, gt_mask=gt, semantic_labels=data['semantic_labels'],
                        segment_count=logits.shape[0])
                    wrong_segment = ((logits[:, i] > 0) != (segment_y > .5)) & (weights > 0)
                    strong = logits[:, i].abs() > 2
                    wrong_points = ((logits[full_low, i] > 0) != gt) & known
                    row.update(original_iou=original_iou, loose_iou=loose_iou, loose_gain=loose_iou-original_iou,
                        strict_strong_error_segments=int((wrong_segment & strong).sum()), valid_segments=int((weights > 0).sum()),
                        strict_strong_error_points=int((wrong_points & strong[full_low]).sum()),
                        raw_error_known_points=int(wrong_points.sum()),
                        original_correct_known_points=int(((parent.pred_masks[:, i] == gt) & known).sum()),
                        original_error_known_points=int(((parent.pred_masks[:, i] != gt) & known).sum()))
                reach_rows.append(row)
                for arm, delta in deltas.items():
                    new_mask = mask_predictions[arm][:, i]
                    new_iou = ordinary_iou(new_mask, gt_columns[:, i]) if entity >= 0 else None
                    shape_rows.append({**row, 'module': arm, 'step': lock['selected_updates'][arm],
                        'delta_mean': float(delta[:, i].mean()), 'delta_abs_mean': float(delta[:, i].abs().mean()),
                        **{f'delta_{k}': v for k, v in quantiles(delta[:, i]).items()},
                        'raw_sign_flips': int(((logits[:, i] > 0) != (logits[:, i] + delta[:, i] > 0)).sum()),
                        'materialized_changed_points': int((new_mask != parent.pred_masks[:, i]).sum()),
                        'fp32_saturated_residual_segments': int((delta[:, i].abs() == 2).sum()),
                        'new_iou': new_iou,
                        'repaired_at_075': entity >= 0 and row['original_iou'] <= .75 < new_iou,
                        'broken_at_075': entity >= 0 and new_iou <= .75 < row['original_iou']})
            print(f'diagnostics {role} H{horizon} {identity[:10]}', flush=True)
    summary = []
    for (role, arm, step), chunks in sorted(aggregated.items()):
        scores, y, valid, parent, partial = [torch.cat([r[j] for r in chunks]) for j in range(5)]
        ranks = [r for r in ranking if (r['role'], r['module'], r['step']) == (role, arm, step)]
        summary.append({'role': role, 'module': arm, 'step': step, 'all_candidates': scores.numel(),
            'weighted_count': int(valid.sum()), 'weighted_mse': float((scores[valid]-y[valid]).square().mean()),
            'train_mean_constant': constant, 'constant_mse': float((constant-y[valid]).square().mean()),
            'partial_ignore_count': int(partial.sum()), 'partial_ignore_up': int((partial & (scores > parent)).sum()),
            'partial_ignore_down': int((partial & (scores < parent)).sum()),
            **quantiles(scores), **{key: sum(r[key] for r in ranks) for key in (
                'comparable_pairs', 'inversions', 'ties', 'original_correct_destroyed', 'original_correct_tied')}})
    assisted_rows = []
    for (role, h, name), accumulator in sorted(assisted.items()):
        evidence = c.artifacts / 'diagnostics/evidence' / f'{role}-H{h}-{name}.json'
        write_json(evidence, accumulator.export_evidence())
        assisted_rows.append({'role': role, 'H': h, 'diagnostic': name, 'full_input_count': accumulator._updates,
            'official_AP': accumulator.compute()['raw_local_AP' if h == 1 else 'online_t-mAP'],
            'status': 'DIAGNOSTIC_ONLY', 'evidence_path': str(evidence), 'evidence_sha256': file_digest(evidence)})
    directory = c.artifacts / 'diagnostics'
    for name, rows in [('QUALITY_RANKING.csv', summary), ('QUALITY_CLASS_RANKING.csv', ranking),
                       ('DUPLICATE_POSITIONS.csv', duplicates), ('SHAPE_SUPERVISION.csv', shape_rows),
                       ('RESIDUAL_REACHABILITY.csv', reach_rows), ('GT_ASSISTED_OFFICIAL.csv', assisted_rows)]:
        write_csv(directory / name, rows)
    gradients = gradient_diagnostics(c, records, lock)
    selected_q = [next(r for r in summary if r['role'] == 'TRAIN' and r['module'] == arm
                       and r['step'] == lock['selected_updates'][arm]) for arm in ('Q_A0', 'Q_P')]
    q_trigger = all(r['weighted_mse'] >= r['constant_mse'] for r in selected_q)
    eligible_m = [r for r in reach_rows if r['role'] == 'TRAIN' and r['shape_usable'] and
                  r['assigned_gt'] is not None and r['original_iou'] < .9 and r['loose_gain'] >= .05]
    cal = read_json(c.artifacts / 'evaluation/CAL_ALL.json')
    mn = next(r for r in cal['rows'] if r['module'] == 'M_N' and r['step'] == lock['selected_updates']['M_N'])
    m_gain = mn['metrics']['T2'] - cal['baseline']['metrics']['T2']
    m_trigger = m_gain <= 1e-6 and len(eligible_m) >= 8
    result = {'status': 'BASE_DIAGNOSTICS_COMPLETE', 'scope': 'TRAIN panel and full NOAUG CAL only',
        'input_counts': {f'{role}_H{h}': n for (role, h), n in input_counts.items()},
        'train_constant': constant, 'train_constant_fit_count': train_y.numel(), 'quality': summary,
        'assisted_official': assisted_rows, 'shape_rows': len(shape_rows), 'reachability_rows': len(reach_rows),
        'gradient_rows': len(gradients), 'q_memorization_trigger': q_trigger,
        'q_memorization_status': 'REQUIRED' if q_trigger else 'NOT_APPLICABLE',
        'm_fit_trigger': m_trigger, 'm_fit_status': 'REQUIRED' if m_trigger else 'NOT_APPLICABLE',
        'm_fit_eligible_candidates': len(eligible_m), 'm_selected_cal_gain': m_gain,
        'm_fit_candidates': sorted(eligible_m, key=lambda r: (r['input_id'], r['source_query_id'], r['source_class_id'], r['retained_index']))[:16],
        'identity': {'sources': source_identity([Path(__file__)]), 'cal_lock_sha256': file_digest(c.artifacts / 'selection/CAL_LOCK.json'),
                     'train_index_sha256': identity_digest(c.train_index), 'eval_index_sha256': identity_digest(c.eval_index)}}
    write_json(directory / 'DIAGNOSTICS.json', result)
    return result
