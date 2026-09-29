import importlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.qp_mn_binding import PROJECT


def test_training_resume_reproduces_uninterrupted_and_rejects_changed_input(tmp_path):
    train = importlib.import_module('scripts.qp_mn_training').train_arm
    config = yaml.safe_load((PROJECT / 'configs/qp_mn_targeted_v1.yaml').read_text())
    records, entries = {}, []
    for horizon in (1, 2):
        key = str(horizon)
        records[key] = {'query': torch.ones(4, 3), 'class_ids': torch.zeros(4, dtype=torch.long),
                        'class_probabilities': torch.ones(4, 3), 'scores': torch.full((4,), .7),
                        'descriptor': {'h': torch.ones(4, horizon, 4), 'support': torch.ones(4, horizon),
                                       'foreground_probability': torch.ones(4, horizon)},
                        'quality': {'temporal': torch.tensor([.1, .3, .5, .7]),
                                    'valid': torch.ones(4, dtype=torch.bool)}}
        entries.append({'input_id': key, 'audit': {'candidate_count': 4},
                        'record': {'input_id': key, 'role': 'TRAIN', 'horizon': horizon, 'reference_id': 'ref'}})
    def context(name):
        return SimpleNamespace(config=config, root=tmp_path / name, artifacts=tmp_path / (name + '-public'),
            dimensions={'feature_dim': 2, 'query_dim': 3, 'classes': 2, 'probability_dim': 3},
            resolved={'official_thresholds': [.5]}, train_index={'entries': entries, 'label_identity_sha256': 'label',
            'cache_identity_sha256': 'pred'}, population=[e['record'] for e in entries])
    a, b = context('split'), context('full')
    train(a, 'Q_P', seed=45, updates=1, device='cpu', records=records)
    train(a, 'Q_P', seed=45, updates=2, device='cpu', records=records)
    train(b, 'Q_P', seed=45, updates=2, device='cpu', records=records)
    p = 'training/Q_P/seed45/update=0002.pt'
    split = torch.load(a.root / p, weights_only=False)
    full = torch.load(b.root / p, weights_only=False)
    assert all(torch.equal(v, full['head'][k]) for k, v in split['head'].items())
    assert len(split['metadata']['supervised_gradient_checks']) == 2
    assert split['metadata']['schedule_horizon'] == 1500
    assert 'torch_rng' in split and 'optimizer' in split
    a.train_index['label_identity_sha256'] = 'changed'
    with pytest.raises(ValueError, match='resume identity'):
        train(a, 'Q_P', seed=45, updates=3, device='cpu', records=records)
