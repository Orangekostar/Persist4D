"""Catch accidental sigmoid/skip changes and shape/regularizer denominator coupling."""
import importlib

import pytest
import torch

from scripts.short_module_data import geometry_loss


def api():
    return importlib.import_module('models.qp_mn_heads')


def test_paired_quality_initialization_and_unrestricted_scores():
    m = api()
    dims = {'feature_dim': 2, 'query_dim': 3, 'classes': 2, 'probability_dim': 3, 'seed': 45}
    a, p = m.build_head('Q_A0', **dims), m.build_head('Q_P', **dims)
    assert all(torch.equal(v, p.state_dict()[k]) for k, v in a.state_dict().items())
    x, parent = torch.randn(4, 22), torch.tensor([.1, .3, .5, .7])
    assert torch.equal(p(x, parent), parent)
    assert torch.equal(a(x, parent), torch.zeros(4))
    with torch.no_grad():
        a.output.bias.fill_(-2)
        p.output.bias.fill_(2)
    assert torch.equal(a(x, parent), torch.full((4,), -2.))
    assert torch.equal(p(x, parent), parent + 2)


def test_quality_weighted_batch_and_empty_connected_gradient():
    m = api()
    scores = torch.tensor([1., 3., 100.], requires_grad=True)
    loss = m.quality_loss(scores, torch.zeros(3), torch.tensor([1., 3., 0.]))
    assert loss.item() == 7
    assert torch.equal(torch.autograd.grad(loss, scores)[0], torch.tensor([.5, 4.5, 0.]))
    empty = m.quality_loss(scores, torch.zeros(3), torch.zeros(3))
    assert empty.item() == 0
    assert torch.equal(torch.autograd.grad(empty, scores)[0], torch.zeros(3))


def fixture(segments, delta_value, *, supervised=True, sampled=True):
    delta = torch.full((segments, 4), delta_value, requires_grad=True)
    return {'logits': delta + .2, 'delta': delta, 'targets': torch.ones(segments, 4),
                'weights': torch.ones(segments), 'segment_stages': torch.zeros(segments, dtype=torch.long),
                'matched': torch.tensor([True, False, True, False]),
                'usable': torch.tensor([supervised, True, False, False]),
                'sample_indices': [torch.arange(segments) if sampled else torch.empty(0, dtype=torch.long)]}


def test_mc_equals_legacy_m1_value_and_gradient_and_mn_only_scales_shape():
    m = api()
    records = [fixture(2, .1), fixture(9, .3), fixture(3, .2, sampled=False),
               fixture(5, .4, supervised=False)]
    terms = [m.shape_terms(**r) for r in records]
    c, detail = m.mask_loss('M_C', terms)
    n, n_detail = m.mask_loss('M_N', terms)
    legacy = torch.stack([geometry_loss('M1', **r) for r in records]).mean()
    assert torch.allclose(c, legacy, atol=1e-7)
    deltas = [r['delta'] for r in records]
    for x, y in zip(torch.autograd.grad(c, deltas, retain_graph=True),
                    torch.autograd.grad(legacy, deltas, retain_graph=True)):
        assert torch.allclose(x, y, atol=1e-7)
    assert detail['B'] == 16 and detail['Npos'] == 2
    assert detail['matched_without_sampled_points'] == 1
    # Equal weight per record, even though records have unequal segment counts.
    expected_reg = .01 * (.01 + .09 + .04 + .16) / 4
    assert detail['regularization'] == pytest.approx(expected_reg)
    assert n_detail['regularization'] == detail['regularization']
    assert n.item() - expected_reg == pytest.approx(8 * (c.item() - expected_reg))
    # The no-supervision records receive identical regularization-only gradients.
    cg = torch.autograd.grad(c, deltas, retain_graph=True)
    ng = torch.autograd.grad(n, deltas)
    assert torch.equal(cg[2], ng[2]) and torch.equal(cg[3], ng[3])
    assert cg[3].abs().sum() > 0


def test_mn_zero_positive_and_all_positive():
    m = api()
    r = fixture(3, .4, sampled=False)
    loss, detail = m.mask_loss('M_N', [m.shape_terms(**r)])
    assert detail['Npos'] == 0 and loss.item() == pytest.approx(.0016)
    assert torch.autograd.grad(loss, r['delta'])[0].abs().sum() > 0
    r = fixture(3, .4)
    r['matched'][:] = True
    r['usable'][:] = True
    terms = [m.shape_terms(**r)]
    assert torch.equal(m.mask_loss('M_C', terms)[0], m.mask_loss('M_N', terms)[0])


def test_unknown_and_combined_modes_rejected():
    for name in ('Q1', 'M1', 'Q_P+M_N', 'unknown'):
        with pytest.raises(ValueError):
            api().Module(name)
