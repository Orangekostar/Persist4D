"""Stable centers retain the frozen context contract and inference-only scope."""
from types import SimpleNamespace

import pytest
import torch

from models import revisit_prompt_centroid_numerics as numerics


def inputs():
    p2s = torch.tensor([2, 0, 2, 1, 0])
    stages = torch.tensor([0, 1, 1])
    xyz = torch.tensor([[4.,6.,8.], [1.,2.,3.], [6.,8.,10.], [9.,10.,11.], [3.,4.,5.]])
    raw = torch.cat((xyz, stages[p2s, None]), 1)
    return raw, p2s, torch.arange(512.).reshape(4,128), torch.tensor([False,False,False,True]), SimpleNamespace(segment_stage_ids=stages)


def test_unsorted_segments_keep_context_fields_and_original_inputs():
    values = inputs()
    copies = [v.clone() for v in values[:4]]
    legacy = numerics.original_segment_context(*values)
    actual = numerics.stable_segment_context(*values)
    torch.testing.assert_close(actual['centers'], torch.tensor([[2.,3.,4.], [9.,10.,11.], [5.,7.,9.]]), rtol=0, atol=0)
    assert actual.keys() == legacy.keys()
    for key in actual.keys()-{'centers'}:
        if isinstance(actual[key], torch.Tensor):
            assert torch.equal(actual[key], legacy[key])
        else:
            assert actual[key] == legacy[key]
    for original, before in zip(values, copies):
        assert torch.equal(original, before)
    assert actual['centers'].dtype == values[0].dtype
    assert actual['features'].data_ptr() == values[2].data_ptr()


def test_original_accumulation_variation_is_discarded(monkeypatch):
    original = numerics.original_segment_context
    calls = []

    def varying(*args):
        result = original(*args)
        calls.append(len(calls)+1)
        result['centers'] = result['centers'] + len(calls)*.1
        return result

    monkeypatch.setattr(numerics, 'original_segment_context', varying)
    outputs = [numerics.stable_segment_context(*inputs()) for _ in range(3)]
    assert len(calls) == 3
    assert all(torch.equal(x['centers'], outputs[0]['centers']) for x in outputs[1:])
    torch.testing.assert_close(outputs[0]['centers'], torch.tensor([[2.,3.,4.], [9.,10.,11.], [5.,7.,9.]]), rtol=0, atol=0)


@pytest.mark.parametrize('case,reason', [('ids','contiguous'), ('padding','dense feature'), ('time','StageMeta')])
def test_original_validation_remains_active(case, reason):
    raw, p2s, features, padding, meta = inputs()
    if case == 'ids': p2s[3] = 2
    if case == 'padding': padding[-1] = False
    if case == 'time': raw[0,3] = 4
    with pytest.raises(ValueError, match=reason):
        numerics.stable_segment_context(raw, p2s, features, padding, meta)


def test_empty_context_keeps_geometry_shape():
    result = numerics.stable_segment_context(torch.zeros(0,4), torch.zeros(0,dtype=torch.long),
        torch.zeros(3,128), torch.ones(3,dtype=torch.bool), SimpleNamespace(segment_stage_ids=torch.zeros(0,dtype=torch.long)))
    assert result['centers'].shape == (0,3)
    assert result['features'].shape == (0,128)


def test_inference_context_restores_original_on_failure():
    from models import revisit_prompt as prompt
    from models import revisit_prompt_modules as modules
    original, training_original = prompt.segment_context, modules.segment_context
    with pytest.raises(RuntimeError, match='owned failure'):
        with numerics.evaluation_centroids():
            assert prompt.segment_context is numerics.stable_segment_context
            assert modules.segment_context is training_original
            raise RuntimeError('owned failure')
    assert prompt.segment_context is original
    assert modules.segment_context is training_original
