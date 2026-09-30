import copy

import torch


def test_non_reentrant_feedback_checkpoint_preserves_output_and_gradients():
    from models.native_long_modules import QueryConditionedMaskFeedback

    torch.manual_seed(45)
    plain = QueryConditionedMaskFeedback(chunk_size=7)
    torch.nn.init.normal_(plain.output_projection.weight, std=.01)
    torch.nn.init.normal_(plain.ffn[2].weight, std=.01)
    checked = copy.deepcopy(plain)
    checked.checkpoint_chunks = True
    q = torch.randn(2, 100, 128)
    f = torch.randn(2, 17, 128)
    pad = torch.zeros(2, 17, dtype=torch.bool)
    a, b = plain(q, f, pad), checked(q, f, pad)
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert all(torch.equal(p.grad, dict(checked.named_parameters())[name].grad)
               for name, p in plain.named_parameters())


def test_stage_quota_capacity_and_remainder():
    from models.native_long_modules import stage_stratified_indices

    stages = torch.tensor([0] + [1] * 7 + [2] * 9)
    generator = torch.Generator().manual_seed(45)
    idx = stage_stratified_indices(stages, 8, generator=generator)
    assert idx.numel() == idx.unique().numel() == 8
    assert torch.bincount(stages[idx], minlength=3).tolist() == [1, 4, 3]
    assert stages[idx].tolist() == sorted(stages[idx].tolist())


def test_stage_sampler_never_advances_global_rng():
    from models.native_long_modules import stage_stratified_indices

    torch.manual_seed(91)
    before = torch.get_rng_state().clone()
    stage_stratified_indices(torch.arange(3).repeat_interleave(10), 8,
                             generator=torch.Generator().manual_seed(45))
    assert torch.equal(before, torch.get_rng_state())


def test_feedback_identity_padding_and_two_update_gradients():
    from models.native_long_modules import QueryConditionedMaskFeedback

    torch.manual_seed(45)
    block = QueryConditionedMaskFeedback(chunk_size=3)
    f = torch.randn(2, 7, 128, requires_grad=True)
    q = torch.randn(2, 100, 128, requires_grad=True)
    padding = torch.tensor([[False] * 7, [False] * 4 + [True] * 3])
    f.data[padding] = 0
    initial = block(q, f, padding)
    assert torch.equal(initial, f)
    optimizer = torch.optim.AdamW(block.parameters(), lr=.001)
    target = torch.randn_like(f)
    (initial - target).square().mean().backward()
    assert block.output_projection.weight.grad.abs().sum() > 0
    assert block.ffn[2].weight.grad.abs().sum() > 0
    assert block.attention.in_proj_weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    result = block(q, f, padding)
    assert torch.count_nonzero(result[padding]) == 0
    assert not torch.equal(result, f)
    (result - target).square().mean().backward()
    assert block.attention.in_proj_weight.grad.abs().sum() > 0
    assert q.grad.abs().sum() > 0
    assert f.grad.abs().sum() > 0


def test_feedback_chunk_equivalence_and_query_dependence():
    from models.native_long_modules import QueryConditionedMaskFeedback

    torch.manual_seed(45)
    chunked = QueryConditionedMaskFeedback(chunk_size=3)
    torch.nn.init.normal_(chunked.output_projection.weight, std=.02)
    torch.nn.init.normal_(chunked.ffn[2].weight, std=.02)
    whole = copy.deepcopy(chunked)
    whole.chunk_size = 1000
    f = torch.randn(2, 11, 128)
    q = torch.randn(2, 100, 128)
    padding = torch.zeros(2, 11, dtype=torch.bool)
    assert torch.allclose(chunked(q, f, padding), whole(q, f, padding), atol=1e-6, rtol=1e-6)
    assert not torch.allclose(chunked(q, f, padding), chunked(q * 2, f, padding))
