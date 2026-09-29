"""Opt-in probes against immutable real repaired records, never new training."""
import json
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from models.qp_mn_heads import build_head
from models.short_module_heads import quality_inputs
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.qp_mn_adapter import batch_loss, segment_samples
from scripts.short_module_data import geometry_loss, quality_labels
from scripts.short_module_native import unpack_prediction
from scripts.short_module_training import build_sample_plan, load_training_records
from scripts.system_comparison_inference import unpack_bool_matrix

ROOT = Path(os.environ.get('RESCENE_REPAIRED_ROOT', '/nonexistent'))
pytestmark = pytest.mark.skipif(not ROOT.is_dir(), reason='set RESCENE_REPAIRED_ROOT for real contract probes')


@pytest.fixture(scope='module')
def real():
    index = json.loads((ROOT / 'EXPORT_INDEX.json').read_text())
    records = load_training_records(ROOT, index)
    plan = build_sample_plan([e['record'] for e in index['entries']],
                             {e['input_id']: e['audit']['candidate_count'] for e in index['entries']}, seed=45)
    return index, records, plan


def test_real_batch_mc_legacy_value_and_parameter_gradients(real):
    _, records, plan = real
    checkpoint = torch.load(ROOT.parent / 'short_module_screen_v1/training/M1/seed45/update=1000.pt',
                            map_location='cpu', weights_only=False)
    dimensions = checkpoint['metadata']['dimensions']
    head = build_head('M_C', **dimensions, seed=45)
    head.load_state_dict(checkpoint['head'])
    # First actually supervised batch, plus the prescribed later plan positions.
    for anchor in (0, 499, 1499):
        for draws in plan[anchor:]:
            new, detail, _ = batch_loss('M_C', head, records, draws, classes=dimensions['classes'], device='cpu')
            if detail['Npos']:
                break
        else:
            continue
        losses = []
        for draw in draws:
            r = records[draw['input_id']]
            selected = torch.tensor(draw['candidates'])
            d = r['descriptor']
            delta = head(r['features'], r['query'][selected], d['h'][selected],
                         F.one_hot(r['class_ids'][selected], dimensions['classes']).float(), d['segment_stages'])
            losses.append(geometry_loss('M1', logits=r['logits'][:, selected] + delta, delta=delta,
                targets=r['targets'][:, selected], weights=r['weights'], segment_stages=d['segment_stages'],
                matched=r['assignment'][selected] >= 0, usable=r['quality']['geometry_valid'][selected],
                sample_indices=segment_samples(r, draw, 'cpu')))
        legacy = torch.stack(losses).mean()
        torch.testing.assert_close(new, legacy, atol=1e-7, rtol=1e-6)
        for a, b in zip(torch.autograd.grad(new, head.parameters()),
                        torch.autograd.grad(legacy, head.parameters())):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


def test_real_q2_bridge_loss_and_gradient_equivalence(real):
    _, records, plan = real
    checkpoint = torch.load(ROOT / 'training/Q2/seed45/update=1000.pt', map_location='cpu', weights_only=False)
    dimensions = checkpoint['metadata']['dimensions']
    head = build_head('Q2_F', **dimensions, seed=45)
    head.load_state_dict(checkpoint['head'])
    for draws in (plan[0], plan[499], plan[1499]):
        new, _, _ = batch_loss('Q2_F', head, records, draws, classes=dimensions['classes'], device='cpu')
        xs, ys, valid = [], [], []
        for draw in draws:
            r = records[draw['input_id']]
            selected = torch.tensor(draw['candidates'])
            d = r['descriptor']
            xs.append(quality_inputs(r['query'][selected], d['h'][selected], r['class_probabilities'][selected],
                                     F.one_hot(r['class_ids'][selected], dimensions['classes']).float(),
                                     r['scores'][selected], d['support'][selected], d['foreground_probability'][selected]))
            ys.append(r['quality']['temporal'][selected])
            valid.append(r['quality']['valid'][selected])
        x, y, v = torch.cat(xs), torch.cat(ys), torch.cat(valid)
        legacy = F.mse_loss(head(x[v]), y[v]) if v.any() else head(x).sum() * 0
        torch.testing.assert_close(new, legacy, atol=1e-7, rtol=1e-6)
        for a, b in zip(torch.autograd.grad(new, head.parameters()),
                        torch.autograd.grad(legacy, head.parameters())):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)


def test_real_quality_score_column_invariance_and_official_affine_ranking(real):
    index, _, _ = real
    entry = next(e for e in index['entries'] if e['record']['role'] == 'TRAIN' and e['record']['horizon'] == 2)
    pred = torch.load(Path(index['cache']) / entry['prediction']['file'], map_location='cpu', weights_only=False)
    saved = torch.load(Path(index['cache']) / entry['targets']['file'], map_location='cpu', weights_only=False)
    target = dict(saved['target'])
    target['masks'] = unpack_bool_matrix(target['masks'])
    prediction = unpack_prediction(pred['parent']).prediction()
    assets = json.loads((ROOT / 'assets.local.json').read_text())
    spec = Path(assets['metric_dataset_spec'])
    labels = quality_labels(prediction, target, dataset_spec=spec)
    order = torch.arange(prediction['pred_scores'].numel() - 1, -1, -1)
    changed = {**prediction, 'pred_masks': prediction['pred_masks'][:, order],
               'pred_classes': prediction['pred_classes'][order],
               'pred_scores': torch.linspace(-2., 2., len(order))}
    other = quality_labels(changed, target, dataset_spec=spec)
    for key in ('temporal', 'concat', 'valid', 'threshold_valid', 'geometry_valid'):
        assert torch.equal(other[key], labels[key][order])
        assert torch.equal(labels[key], saved['quality'][key])
    values = []
    for scores in (changed['pred_scores'], 2 * changed['pred_scores'] + 4):
        metric = OfficialMetricAccumulator(mode='strict_online', dataset_spec=spec, min_region_size=100)
        metric.update({**changed, 'pred_scores': scores}, target)
        values.append(metric.compute()['online_t-mAP'])
    assert values[0] == pytest.approx(values[1], abs=1e-8)
