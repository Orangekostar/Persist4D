from types import SimpleNamespace

import pytest
import torch


def test_encoder_only_load_is_strict_and_leaves_task_fresh():
    from models.pointcept import PointceptBackbone

    model = torch.nn.Module()
    model.embedding = torch.nn.Linear(3, 4)
    model.enc = torch.nn.Linear(4, 4)
    model.dec = torch.nn.Linear(4, 3)
    initial_task = {k: v.clone() for k, v in model.dec.state_dict().items()}
    source = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    wrapper = SimpleNamespace(model=model, pretrained_scope="encoder_only")
    PointceptBackbone._load_state_dict(wrapper, {"state_dict": source})
    assert all(torch.equal(v, torch.ones_like(v)) for v in model.enc.state_dict().values())
    assert all(torch.equal(v, initial_task[k]) for k, v in model.dec.state_dict().items())
    assert wrapper.pretrained_load_audit["excluded_pretrained_keys"] == ["dec.bias", "dec.weight"]
    del source["enc.weight"]
    with pytest.raises(ValueError, match="coverage"):
        PointceptBackbone._load_state_dict(wrapper, {"state_dict": source})
