import importlib

import torch


def test_portable_head_excludes_optimizer_and_keeps_exact_parameters(tmp_path):
    api = importlib.import_module('scripts.qp_mn_delivery')
    raw, public = tmp_path / 'raw.pt', tmp_path / 'public.pt'
    torch.save({'head': {'weight': torch.tensor([1., 2.])},
                'metadata': {'module': 'Q2', 'optimizer_updates': 500},
                'optimizer': {'secret_large_state': torch.ones(100)}, 'torch_rng': torch.ones(2)}, raw)
    receipt = api.portable_checkpoint(raw, public, mode='Q2_F', label_identity='labels')
    saved = torch.load(public, weights_only=False)
    assert set(saved) == {'head', 'metadata', 'slot', 'inference_api'}
    assert saved['metadata']['module'] == 'Q2_F'
    assert saved['metadata']['source_module'] == 'Q2'
    assert saved['metadata']['label_identity_sha256'] == 'labels'
    assert torch.equal(saved['head']['weight'], torch.tensor([1., 2.]))
    assert receipt['bytes'] == public.stat().st_size
