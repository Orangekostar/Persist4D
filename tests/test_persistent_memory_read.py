import pytest
import torch

from models.persistent_memory_read import (
    DetachedMemoryReadState,
    MemoryReadError,
    PersistentMemoryRead,
)


def _state(*, batch: int = 2) -> DetachedMemoryReadState:
    embeddings = torch.linspace(-1.0, 1.0, batch * 100 * 128).reshape(
        batch, 100, 128
    )
    occupied = torch.zeros(batch, 100, dtype=torch.bool)
    occupied[0, :3] = True
    active = torch.zeros_like(occupied)
    active[0, :2] = True
    confidence = torch.full((batch, 100), 0.75)
    last_seen = torch.arange(100).repeat(batch, 1)
    return DetachedMemoryReadState(
        embeddings=embeddings,
        occupied_mask=occupied,
        active_mask=active,
        confidence=confidence,
        last_seen=last_seen,
    )


def test_state_is_detached_and_has_no_ground_truth_surface() -> None:
    source = torch.randn(1, 100, 128, requires_grad=True)
    state = DetachedMemoryReadState(
        embeddings=source,
        occupied_mask=torch.ones(1, 100, dtype=torch.bool),
    )

    assert state.embeddings.requires_grad is False
    assert state.embeddings.grad_fn is None
    assert state.embeddings.data_ptr() != source.data_ptr()
    assert not hasattr(state, "gt_ids")
    assert not hasattr(state, "gt_masks")


@pytest.mark.parametrize(
    ("queries", "embeddings", "occupied"),
    [
        (torch.zeros(1, 99, 128), torch.zeros(1, 100, 128), torch.zeros(1, 100)),
        (torch.zeros(1, 100, 127), torch.zeros(1, 100, 128), torch.zeros(1, 100)),
        (torch.zeros(2, 100, 128), torch.zeros(1, 100, 128), torch.zeros(1, 100)),
        (torch.zeros(1, 100, 128), torch.zeros(1, 99, 128), torch.zeros(1, 99)),
    ],
)
def test_read_rejects_noncontract_shapes(
    queries: torch.Tensor, embeddings: torch.Tensor, occupied: torch.Tensor
) -> None:
    module = PersistentMemoryRead()
    state = DetachedMemoryReadState(
        embeddings=embeddings,
        occupied_mask=occupied.bool(),
    )

    with pytest.raises(MemoryReadError):
        module(queries, state)


def test_zero_output_initialization_is_exact_identity_and_finite() -> None:
    torch.manual_seed(3)
    module = PersistentMemoryRead()
    queries = torch.randn(2, 100, 128)

    output = module(queries, _state())

    assert torch.equal(output, queries)
    assert torch.count_nonzero(module.output_projection.weight) == 0
    assert torch.count_nonzero(module.output_projection.bias) == 0
    assert torch.count_nonzero(module.query_projection.weight) > 0
    assert torch.isfinite(output).all()
    assert all(
        value is None or torch.isfinite(torch.tensor(value))
        for value in module.last_diagnostics.values()
    )


def test_empty_sample_stays_identity_after_adapter_learns() -> None:
    torch.manual_seed(5)
    module = PersistentMemoryRead()
    torch.nn.init.eye_(module.output_projection.weight)
    torch.nn.init.zeros_(module.output_projection.bias)
    queries = torch.randn(2, 100, 128)

    output = module(queries, _state())

    assert not torch.equal(output[0], queries[0])
    assert torch.equal(output[1], queries[1])
    diagnostics = module.last_diagnostics
    assert diagnostics["active_attention_mass"] > 0.0
    assert diagnostics["dormant_attention_mass"] > 0.0
    assert diagnostics["null_attention_mass"] > 0.0
    assert diagnostics["occupied_slot_count"] == 3


def test_explicit_null_keeps_mixed_empty_attention_safe() -> None:
    module = PersistentMemoryRead()
    torch.nn.init.normal_(module.output_projection.weight)
    queries = torch.randn(2, 100, 128)
    output = module(queries, _state())

    assert torch.isfinite(output).all()
    assert torch.equal(output[1], queries[1])
    assert module.last_diagnostics["empty_sample_count"] == 1


def test_read_parameters_receive_gradients_after_two_optimizer_updates() -> None:
    torch.manual_seed(7)
    module = PersistentMemoryRead()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    queries = torch.randn(1, 100, 128)
    target = torch.randn_like(queries)
    state = _state(batch=1)

    loss = (module(queries, state) - target).square().mean()
    loss.backward()
    assert module.output_projection.weight.grad.abs().sum() > 0
    assert module.query_projection.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    loss = (module(queries, state) - target).square().mean()
    loss.backward()

    assert module.query_projection.weight.grad.abs().sum() > 0
    assert module.key_projection.weight.grad.abs().sum() > 0
    assert module.value_projection.weight.grad.abs().sum() > 0
    assert module.gate_projection.weight.grad.abs().sum() > 0
    assert torch.isfinite(module.query_projection.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_memory_read_cuda_contract() -> None:
    module = PersistentMemoryRead().cuda()
    queries = torch.randn(1, 100, 128, device="cuda")
    state = DetachedMemoryReadState(
        embeddings=torch.randn(1, 100, 128, device="cuda"),
        occupied_mask=torch.ones(1, 100, dtype=torch.bool, device="cuda"),
    )

    output = module(queries, state)
    output.sum().backward()

    assert output.is_cuda
    assert torch.isfinite(output).all()
    assert module.output_projection.weight.grad is not None
