from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.persist4d_allt import (
    Persist4DAllT,
    Persist4DModelError,
    strict_load_r1_with_named_adapters,
)
from models.persistent_memory_read import DetachedMemoryReadState


class _RecordingRead(nn.Module):
    def __init__(self, trace=None):
        super().__init__()
        self.inputs = []
        self.trace = trace

    def forward(self, queries, state):
        if self.trace is not None:
            self.trace.append("M")
        self.inputs.append((queries.detach().clone(), state))
        return queries + 2.0


class _RecordingLocal(nn.Module):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def forward(self, queries, **context):
        self.trace.append("L")
        self.context = context
        return queries + 1.0


def _hook_model(*, with_local: bool = False) -> Persist4DAllT:
    model = Persist4DAllT.__new__(Persist4DAllT)
    nn.Module.__init__(model)
    model.hlevels = [0, 1, 2, 3]
    model.memory_read_enabled = True
    model._local_memory_trace = []
    model.memory_read = _RecordingRead(model._local_memory_trace)
    model._memory_read_state = DetachedMemoryReadState(
        embeddings=torch.zeros(1, 100, 128),
        occupied_mask=torch.ones(1, 100, dtype=torch.bool),
    )
    model._memory_read_call_count = 0
    model._local_enhancement_call_count = 0
    model.local_enhancement = (
        _RecordingLocal(model._local_memory_trace) if with_local else None
    )
    model.mask_module = lambda queries, _features: (
        torch.zeros(1, 100, 19),
        None,
        torch.zeros(1, 2, 100),
    )
    return model


def test_memory_read_occurs_once_after_first_complete_hlevel_pass() -> None:
    model = _hook_model()
    queries = torch.zeros(1, 100, 128)

    for execution_stage in range(8):
        queries = model.after_decoder_stage(
            queries,
            execution_stage_idx=execution_stage,
            shared_parameter_idx=0,
        )

    assert model._memory_read_call_count == 1
    assert len(model.memory_read.inputs) == 1
    assert torch.equal(queries, torch.full_like(queries, 2.0))


def test_local_enhancement_precedes_the_single_memory_read() -> None:
    model = _hook_model(with_local=True)
    original = torch.zeros(1, 100, 128)

    output = model.after_decoder_stage(
        original,
        execution_stage_idx=3,
        shared_parameter_idx=0,
        decoder_features=torch.zeros(1, 2, 128),
        decoder_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        point2segment=None,
    )

    assert model._local_memory_trace == ["L", "M"]
    assert torch.equal(model.memory_read.inputs[0][0], original + 1.0)
    assert torch.equal(output, original + 3.0)
    assert set(model.local_enhancement.context) == {
        "class_logits",
        "mask_logits",
        "padding_mask",
        "point2segment",
    }
    assert model._local_enhancement_call_count == 1


def test_disabled_allt_hook_is_identity_without_rng_consumption() -> None:
    model = Persist4DAllT.__new__(Persist4DAllT)
    nn.Module.__init__(model)
    model.hlevels = [0, 1, 2, 3]
    model.memory_read_enabled = False
    model.memory_read = None
    model.local_enhancement = None
    model._memory_read_state = None
    model._memory_read_call_count = 0
    queries = torch.randn(1, 100, 128)
    before = torch.random.get_rng_state().clone()

    output = model.after_decoder_stage(
        queries,
        execution_stage_idx=3,
        shared_parameter_idx=0,
    )

    assert output is queries
    assert torch.equal(torch.random.get_rng_state(), before)


class _TinyAllTSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.base = nn.Linear(2, 2)
        self.model.memory_read = nn.Linear(2, 2)


def test_strict_r1_load_allows_only_exact_named_adapter_keys() -> None:
    source = _TinyAllTSystem()
    old_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if not key.startswith("model.memory_read.")
    }
    target = _TinyAllTSystem().eval()

    audit = strict_load_r1_with_named_adapters(
        target,
        old_state,
        allowed_missing_prefixes=("model.memory_read.",),
    )

    assert audit == {
        "loaded_key_count": 2,
        "missing_keys": ["model.memory_read.bias", "model.memory_read.weight"],
        "unexpected_keys": [],
    }
    assert target.training is False


def test_strict_r1_load_rejects_base_missing_and_unexpected_keys() -> None:
    system = _TinyAllTSystem()
    with pytest.raises(Persist4DModelError, match="missing keys"):
        strict_load_r1_with_named_adapters(
            system,
            {},
            allowed_missing_prefixes=("model.memory_read.",),
        )
    with pytest.raises(Persist4DModelError, match="unexpected keys"):
        strict_load_r1_with_named_adapters(
            system,
            {
                "model.base.weight": system.model.base.weight.detach().clone(),
                "model.base.bias": system.model.base.bias.detach().clone(),
                "unexpected": torch.tensor(1.0),
            },
            allowed_missing_prefixes=("model.memory_read.",),
        )


def test_forward_rejects_memory_state_when_module_is_disabled() -> None:
    model = Persist4DAllT.__new__(Persist4DAllT)
    nn.Module.__init__(model)
    model.memory_read_enabled = False
    model.memory_read = None
    model.local_enhancement = None
    state = DetachedMemoryReadState.empty(1, device="cpu", dtype=torch.float32)

    with pytest.raises(Persist4DModelError, match="disabled"):
        model.forward(SimpleNamespace(), memory_read_state=state)
