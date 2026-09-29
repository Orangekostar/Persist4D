"""Exercise real prediction isolation, fixed draws and scope validation."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from models.qp_mn_heads import build_head


def api():
    return importlib.import_module('scripts.qp_mn_adapter')


def test_raw_quality_prediction_changes_only_score_without_refiltering():
    module = api()
    parent = SimpleNamespace(
        pred_scores=torch.tensor([.2, .9]),
        soft_evidence=SimpleNamespace(query_features=torch.randn(2, 3),
                                     class_probabilities=torch.randn(2, 3),
                                     source_class_ids=torch.tensor([0, 1])))
    original = {'pred_scores': parent.pred_scores, 'pred_masks': torch.tensor([[True, False]]),
                    'pred_classes': torch.tensor([1, 2]), 'source_query_ids': torch.tensor([6, 7])}
    parent.prediction = lambda: dict(original)
    desc = {'h': torch.randn(2, 1, 4), 'classes': 2, 'support': torch.ones(2, 1),
                'foreground_probability': torch.ones(2, 1)}
    head = build_head('Q_P', feature_dim=2, query_dim=3, classes=2, probability_dim=3, seed=45)
    with torch.no_grad():
        head.output.bias.fill_(-2)
    result = module.apply_module('Q_P', head, parent, desc, system=None)
    assert torch.equal(result['pred_scores'], torch.tensor([-1.8, -1.1]))
    for key in ('pred_masks', 'pred_classes', 'source_query_ids'):
        assert torch.equal(result[key], original[key])
    with pytest.raises(ValueError):
        module.apply_module('Q_A0', head, parent, desc, system=None)


def test_segment_samples_are_shared_stagewise_and_reproducible():
    r = {'descriptor': {'segment_stages': torch.tensor([0, 1, 0, 1])},
             'weights': torch.tensor([1., 0., 2., 3.])}
    draw = {'segment_seed': 45, 'horizon': 2}
    a = api().segment_samples(r, draw, 'cpu')
    b = api().segment_samples(r, draw, 'cpu')
    assert set(a[0].tolist()) == {0, 2} and a[1].tolist() == [3]
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_output_context_rejects_historical_runtime_alias(tmp_path):
    old = tmp_path / 'old'
    old.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError):
        api().validate_output_paths(alias, tmp_path / 'public', [old])


def test_batch_quality_uses_one_denominator_across_records():
    head = build_head('Q_A0', feature_dim=2, query_dim=3, classes=2, probability_dim=3, seed=45)
    records = {}
    draws = []
    for i in range(4):
        records[str(i)] = {'query': torch.zeros(4, 3), 'class_ids': torch.zeros(4, dtype=torch.long),
                              'class_probabilities': torch.ones(4, 3), 'scores': torch.ones(4),
                              'descriptor': {'h': torch.zeros(4, 1, 4), 'support': torch.ones(4, 1),
                                              'foreground_probability': torch.ones(4, 1)},
                              'quality': {'temporal': torch.full((4,), float(i + 1)),
                                           'valid': torch.tensor([True] + [i == 0] * 3)}}
        draws.append({'input_id': str(i), 'candidates': [0, 1, 2, 3]})
    loss, detail, parts = api().batch_loss('Q_A0', head, records, draws, classes=2, device='cpu')
    assert loss.item() == pytest.approx(33 / 7)
    assert detail['Npos'] == 7 and detail['B'] == 16
    loss.backward()
    assert head.output.bias.grad.item() == pytest.approx(-26 / 7)
    assert parts['shape'] is None
