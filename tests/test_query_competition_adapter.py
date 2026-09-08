import torch

from models.query_competition_adapter import QueryCompetitionAdapter


def _competition_inputs():
    queries = torch.randn(1, 100, 128)
    class_logits = torch.full((1, 100, 19), -8.0)
    class_logits[:, :, 18] = 8.0
    class_logits[0, 0, 0] = 12.0
    class_logits[0, 1, 0] = 10.0
    mask_logits = torch.full((1, 4, 100), -8.0)
    mask_logits[0, :3, 0] = 12.0
    mask_logits[0, :3, 1] = 8.0
    padding_mask = torch.zeros(1, 4, dtype=torch.bool)
    point2segment = [torch.tensor([0, 0, 1, 2, 3])]
    return queries, class_logits, mask_logits, padding_mask, point2segment


def _forward(adapter, inputs):
    queries, class_logits, mask_logits, padding_mask, point2segment = inputs
    return adapter(
        queries,
        class_logits=class_logits,
        mask_logits=mask_logits,
        padding_mask=padding_mask,
        point2segment=point2segment,
    )


def test_qcl_adapter_initialization_is_exact_identity_without_rng_use() -> None:
    adapter = QueryCompetitionAdapter()
    inputs = _competition_inputs()
    before = torch.random.get_rng_state().clone()

    output = _forward(adapter, inputs)

    assert torch.equal(output, inputs[0])
    assert torch.equal(torch.random.get_rng_state(), before)


def test_qcl_adapter_updates_only_lower_ranked_competing_queries() -> None:
    adapter = QueryCompetitionAdapter()
    with torch.no_grad():
        adapter.message_projection.weight.copy_(torch.eye(128))
        adapter.message_projection.bias.zero_()
        adapter.output_projection.weight.copy_(torch.eye(128))
        adapter.output_projection.bias.zero_()
        adapter.gate_projection.weight.zero_()
        adapter.gate_projection.bias.zero_()
    inputs = _competition_inputs()

    output = _forward(adapter, inputs)

    assert torch.equal(output[:, 0], inputs[0][:, 0])
    assert not torch.equal(output[:, 1], inputs[0][:, 1])
    assert torch.equal(output[:, 2:], inputs[0][:, 2:])
    assert adapter.last_diagnostics["competing_query_count"] == 1


def test_qcl_adapter_reaches_inner_parameters_after_two_updates() -> None:
    adapter = QueryCompetitionAdapter()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    inputs = _competition_inputs()

    for _ in range(2):
        optimizer.zero_grad()
        output = _forward(adapter, inputs)
        output[:, 1].sum().backward()
        optimizer.step()

    assert adapter.output_projection.weight.grad is not None
    assert torch.count_nonzero(adapter.output_projection.weight.grad) > 0
    assert adapter.message_projection.weight.grad is not None
    assert torch.count_nonzero(adapter.message_projection.weight.grad) > 0


def test_qcl_adapter_bounds_support_without_random_sampling() -> None:
    adapter = QueryCompetitionAdapter(support_budget=3)
    inputs = list(_competition_inputs())
    inputs[2] = inputs[2].repeat(1, 3, 1)
    inputs[3] = torch.zeros(1, 12, dtype=torch.bool)
    inputs[4] = None

    _forward(adapter, tuple(inputs))

    assert adapter.last_diagnostics["sampled_support_count"] == 3
