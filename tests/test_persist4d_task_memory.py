from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from datasets.task_memory_episode import StageMeta
from models.persist4d_allt import Persist4DAllT
from models.persist4d_task_memory import (
    Persist4DTaskMemory,
    TaskMemoryModelError,
    strict_load_r1_task_memory,
)
from models.rescene import ReScene
from models.task_memory_read import TaskMemoryRead
from models.task_memory_routing import PredictionObservation, route_entities
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState
from scripts.preflight_task_memory_model import (
    TaskMemoryModelPreflightError,
    build_model_preflight_payloads,
    compare_tensor_trees,
)


def _meta(stage: int = 1) -> StageMeta:
    scan_ids = ("scan-0",) if stage == 0 else ("scan-0", "scan-1")
    count = len(scan_ids)
    indices = torch.arange(count, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=scan_ids,
        absolute_stage_index=stage,
        local_stage_ids=indices,
        original_vertex_ids=tuple(torch.tensor([0]) for _ in scan_ids),
        scan_vertex_offsets=torch.arange(count + 1, dtype=torch.long),
        point2segment=indices,
        segment_stage_ids=indices,
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=indices,
        full_resolution_point2segment=indices,
    )


def _state(*, occupied: bool = True) -> TaskMemoryState:
    state = TaskMemoryState.empty(
        batch_size=1,
        capacity=100,
        feature_dim=128,
        class_count=19,
        device="cpu",
        dtype=torch.float32,
        config=TaskMemoryConfig(association_threshold=-1.0),
    )
    if not occupied:
        return state
    embedding = state.embedding.clone()
    embedding[0, 0, 0] = 1.0
    class_prob = state.class_prob.clone()
    class_prob[0, 0, 0] = 1.0
    confidence = state.confidence.clone()
    confidence[0, 0] = 0.9
    occupied_mask = state.occupied.clone()
    occupied_mask[0, 0] = True
    return replace(
        state,
        embedding=embedding,
        class_prob=class_prob,
        confidence=confidence,
        occupied=occupied_mask,
        active=occupied_mask.clone(),
        last_seen=torch.cat(
            [torch.zeros(1, 1, dtype=torch.long), state.last_seen[:, 1:]], dim=1
        ),
        stage_watermark=torch.tensor([0]),
        logical_ids=torch.cat(
            [torch.zeros(1, 1, dtype=torch.long), state.logical_ids[:, 1:]], dim=1
        ),
        generations=torch.cat(
            [torch.zeros(1, 1, dtype=torch.long), state.generations[:, 1:]], dim=1
        ),
        next_logical_id=torch.tensor([1]),
    )


def _route(state: TaskMemoryState):
    features = torch.zeros(1, 100, 128)
    features[0, 0, 0] = 1.0
    class_prob = torch.zeros(1, 100, 19)
    class_prob[0, 0, 0] = 1.0
    valid = torch.zeros(1, 100, dtype=torch.bool)
    valid[0, 0] = True
    observation = PredictionObservation(
        features=features,
        class_prob=class_prob,
        confidence=valid.to(torch.float32),
        valid=valid,
        current_supported=valid.clone(),
        previous_supported=torch.zeros_like(valid),
    )
    return route_entities(observation, state, [_meta()])


def _identity_read() -> TaskMemoryRead:
    read = TaskMemoryRead()
    identity = torch.eye(128)
    with torch.no_grad():
        for layer in (read.query_projection, read.key_projection, read.value_projection):
            layer.weight.copy_(identity)
            layer.bias.zero_()
        read.output_projection.weight.copy_(identity)
        read.output_projection.bias.zero_()
        read.gate_projection.weight.zero_()
        read.gate_projection.bias.zero_()
        read.null_key.zero_()
        read.null_key[0] = -1.0
    return read


def test_task_read_has_frozen_shape_scale_and_zero_initial_residual() -> None:
    read = TaskMemoryRead()
    assert read.hidden_dim == 128
    assert read.num_queries == read.capacity == 100
    assert read.attention_scale.item() == pytest.approx(math.sqrt(128))
    assert torch.count_nonzero(read.output_projection.weight) == 0
    assert torch.count_nonzero(read.output_projection.bias) == 0

    state = _state()
    route = _route(state)
    queries = torch.randn(1, 100, 128)
    output = read(queries, state, route)

    assert torch.equal(output, queries)


def test_task_read_uses_only_routed_slot_and_preserves_discovery_queries() -> None:
    read = _identity_read()
    state = _state()
    route = _route(state)
    queries = torch.zeros(1, 100, 128)
    queries[0, 0, 0] = 1.0
    queries[0, 1, 1] = 1.0

    output = read(queries, state, route)

    assert output[0, 0, 0] > queries[0, 0, 0]
    assert torch.equal(output[:, 1:], queries[:, 1:])
    assert read.last_diagnostics["matched_query_count"] == 1
    assert read.last_diagnostics["slot_win_count"] == 1
    assert read.last_diagnostics["unassigned_query_count"] == 99


def test_task_read_null_winner_has_strict_zero_residual_and_clamped_scale() -> None:
    read = _identity_read()
    state = _state()
    route = _route(state)
    queries = torch.zeros(1, 100, 128)
    queries[0, 0, 0] = -1.0
    with torch.no_grad():
        read.attention_scale.fill_(100.0)

    output = read(queries, state, route)

    assert torch.equal(output, queries)
    assert read.last_diagnostics["attention_scale"] == 64.0
    assert read.last_diagnostics["null_win_count"] == 1
    assert read.last_diagnostics["read_output_norm"] == 0.0


def test_task_read_empty_state_preserves_all_queries() -> None:
    read = _identity_read()
    state = _state(occupied=False)
    route = _route(state)
    queries = torch.randn(1, 100, 128)

    output = read(queries, state, route)

    assert torch.equal(output, queries)
    assert read.last_diagnostics["matched_query_count"] == 0
    assert read.last_diagnostics["unassigned_query_count"] == 100


class _RecordingRead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = []
        self.last_diagnostics = {"status": "recorded"}

    def forward(self, queries, state, route):
        self.calls.append((queries.detach().clone(), state, route))
        return queries + 2.0


class _RecordingHead:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, queries, _features):
        self.calls += 1
        logits = torch.full((1, 100, 19), -8.0)
        logits[0, 0, 0] = 8.0
        masks = torch.full((1, 2, 100), -8.0)
        masks[0, :, 0] = 8.0
        return logits, None, masks


def _hook_model(*, enabled: bool = True) -> Persist4DTaskMemory:
    model = Persist4DTaskMemory.__new__(Persist4DTaskMemory)
    nn.Module.__init__(model)
    model.hlevels = [0, 1, 2, 3]
    model.num_queries = 100
    model.mask_dim = 128
    model.num_classes = 19
    model.decoder_norm = nn.Identity()
    model.task_memory_enabled = enabled
    model.task_memory_capacity = 100
    model.task_memory_config = TaskMemoryConfig(association_threshold=-1.0)
    model.task_background_class = 18
    model.task_confidence_threshold = 0.5
    model.task_mask_threshold = 0.5
    model.task_minimum_mask_support = 1
    model.task_read = _RecordingRead() if enabled else None
    model.mask_module = _RecordingHead()
    model._task_state = _state() if enabled else None
    model._task_stage_meta = (_meta(),) if enabled else None
    model._task_route = None
    model._task_pre_observation = None
    model._task_read_call_count = 0
    return model


def test_model_routes_and_reads_once_after_first_complete_hlevel_pass() -> None:
    model = _hook_model()
    queries = torch.zeros(1, 100, 128)
    queries[0, 0, 0] = 1.0
    padding = torch.zeros(1, 2, dtype=torch.bool)

    for execution_stage in range(8):
        queries = model.after_decoder_stage(
            queries,
            execution_stage_idx=execution_stage,
            shared_parameter_idx=0,
            decoder_features=torch.zeros(1, 2, 128),
            decoder_padding_mask=padding,
            point2segment=None,
        )

    assert model.mask_module.calls == 1
    assert model._task_read_call_count == 1
    assert len(model.task_read.calls) == 1
    assert model._task_route.query_to_slot[0, 0].item() == 0
    assert torch.equal(queries[0, 0], torch.full((128,), 2.0).index_add(0, torch.tensor([0]), torch.tensor([1.0])))


def test_disabled_hook_is_identity_and_does_not_call_prediction_head() -> None:
    model = _hook_model(enabled=False)
    queries = torch.randn(1, 100, 128)
    before = torch.random.get_rng_state().clone()

    output = model.after_decoder_stage(
        queries,
        execution_stage_idx=3,
        shared_parameter_idx=0,
        decoder_features=torch.zeros(1, 2, 128),
        decoder_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    assert output is queries
    assert model.mask_module.calls == 0
    assert torch.equal(torch.random.get_rng_state(), before)


def test_pre_observation_keeps_confident_previous_only_query_valid() -> None:
    model = _hook_model()
    queries = torch.zeros(1, 100, 128)
    logits = torch.full((1, 100, 19), -8.0)
    logits[0, 0, 0] = 8.0
    masks = torch.full((1, 2, 100), -8.0)
    masks[0, 0, 0] = 8.0

    observation = model._build_pre_observation(
        queries,
        logits,
        masks,
        torch.zeros(1, 2, dtype=torch.bool),
        [_meta()],
    )

    assert observation.previous_supported[0, 0].item() is True
    assert observation.current_supported[0, 0].item() is False
    assert observation.valid[0, 0].item() is True


def test_forward_returns_route_lineage_and_clears_temporary_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _hook_model()
    marker = torch.tensor(7.0)

    def fake_forward(self, _x, **_kwargs):
        queries = torch.zeros(1, 100, 128)
        queries[0, 0, 0] = 1.0
        self.after_decoder_stage(
            queries,
            execution_stage_idx=3,
            shared_parameter_idx=0,
            decoder_features=torch.zeros(1, 2, 128),
            decoder_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
        return {"raw_marker": marker}

    monkeypatch.setattr(ReScene, "forward", fake_forward)
    output = model.forward(object(), task_state=_state(), stage_meta=[_meta()])

    assert output["raw_marker"] is marker
    assert output["task_memory_route"].query_to_slot[0, 0].item() == 0
    assert output["task_memory_lineage"]["prior_logical_id"][0, 0].item() == 0
    assert output["task_memory_read_diagnostics"] == {"status": "recorded"}
    assert model._task_state is None
    assert model._task_stage_meta is None
    assert model._task_route is None
    assert model._task_pre_observation is None


def test_forward_clears_temporary_state_after_base_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _hook_model()

    def fail_forward(_self, _x, **_kwargs):
        raise RuntimeError("base failure")

    monkeypatch.setattr(ReScene, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="base failure"):
        model.forward(object(), task_state=_state(), stage_meta=[_meta()])

    assert model._task_state is None
    assert model._task_stage_meta is None
    assert model._task_route is None
    assert model._task_pre_observation is None


def test_task_model_inherits_rescene_directly_without_old_c2_read() -> None:
    assert issubclass(Persist4DTaskMemory, ReScene)
    assert not issubclass(Persist4DTaskMemory, Persist4DAllT)
    assert "memory_read" not in Persist4DTaskMemory.__dict__


class _TinyTaskSystem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.base = nn.Linear(2, 2)
        self.model.task_read = nn.Linear(2, 2)


def test_strict_r1_load_allows_only_task_read_prefix() -> None:
    source = _TinyTaskSystem()
    r1_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if not key.startswith("model.task_read.")
    }
    target = _TinyTaskSystem()

    audit = strict_load_r1_task_memory(target, r1_state)

    assert audit == {
        "loaded_key_count": 2,
        "missing_keys": ["model.task_read.bias", "model.task_read.weight"],
        "unexpected_keys": [],
    }
    with pytest.raises(TaskMemoryModelError, match="missing keys"):
        strict_load_r1_task_memory(target, {})


def test_real_parity_helpers_require_exact_t1_t2_prediction_trees() -> None:
    disabled = {
        "pred_logits": torch.tensor([[[1.0, 2.0]]]),
        "pred_masks": [torch.tensor([[3.0]])],
        "aux_outputs": [{"pred_logits": torch.tensor([[[4.0]]])}],
    }
    exact = compare_tensor_trees(disabled, disabled, tolerance=0.0)

    assert exact["exact"] is True
    assert exact["max_abs_error"] == 0.0
    assert exact["tensor_count"] == 3
    tensors_by_path = {item["path"]: item for item in exact["tensors"]}
    assert tensors_by_path["output.pred_logits"]["shape"] == [1, 1, 2]

    changed = {
        **disabled,
        "pred_masks": [torch.tensor([[3.001]])],
    }
    with pytest.raises(TaskMemoryModelPreflightError, match="parity tolerance"):
        compare_tensor_trees(disabled, changed, tolerance=0.0)

    load_report, shape_trace = build_model_preflight_payloads(
        source_commit="a" * 40,
        checkpoint_sha256="b" * 64,
        checkpoint_bytes=754_813_672,
        load_audit={
            "loaded_key_count": 798,
            "missing_keys": ["model.task_read.null_key"],
            "unexpected_keys": [],
        },
        samples=[
            {
                "stage": "T1",
                "reference_id": "reference-0",
                "sequence_id": "scan-0-scan-1",
                "scan_ids_in_window": ["scan-0"],
                "raw_parity": exact,
                "official_parity": exact,
                "query_shape": [1, 100, 128],
                "route_shape": [1, 100],
                "state_shape": [1, 100, 128],
                "read_invocations": 1,
                "matched_queries": 0,
                "read_output_norm": 0.0,
            },
            {
                "stage": "T2",
                "reference_id": "reference-0",
                "sequence_id": "scan-0-scan-1",
                "scan_ids_in_window": ["scan-0", "scan-1"],
                "raw_parity": exact,
                "official_parity": exact,
                "query_shape": [1, 100, 128],
                "route_shape": [1, 100],
                "state_shape": [1, 100, 128],
                "read_invocations": 1,
                "matched_queries": 0,
                "read_output_norm": 0.0,
            },
        ],
    )

    assert load_report["status"] == shape_trace["status"] == "PASS"
    assert load_report["allowed_missing_prefixes"] == ["model.task_read."]
    assert shape_trace["stages"] == ["T1", "T2"]
    assert "/home/" not in str(load_report)
