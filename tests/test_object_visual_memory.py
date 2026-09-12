from __future__ import annotations

from dataclasses import replace

import torch

from datasets.task_memory_episode import StageMeta
from models.object_visual_memory import (
    ObjectVisualRead,
    ObjectVisualState,
    VisualCandidate,
    select_visual_representatives,
    update_visual_state,
)
from trainer.task_memory_trainer import build_visual_slot_candidates


def _candidate(keys: list[int], qualities: list[float], features: list[list[float]]) -> list[VisualCandidate]:
    return [
        VisualCandidate(
            feature=torch.tensor(feature, dtype=torch.float32),
            quality=torch.tensor(quality, dtype=torch.float32),
            source_stage=stage,
            source_key=key,
        )
        for stage, (key, quality, feature) in enumerate(zip(keys, qualities, features))
    ]


def test_visual_state_is_fixed_shape_and_bounded() -> None:
    state = ObjectVisualState.empty(
        batch_size=2,
        capacity=100,
        representatives=8,
        feature_dim=128,
        device="cpu",
        dtype=torch.float32,
    )
    state.validate()
    assert state.features.shape == (2, 100, 8, 128)
    assert state.valid.shape == (2, 100, 8)
    assert state.storage_bytes <= 2 * 1024 * 1024


def test_v_last_is_stable_and_prefers_quality_then_coverage() -> None:
    candidates = _candidate(
        [7, 7, 6, 5],
        [0.9, 0.8, 0.95, 0.7],
        [[1.0, 0.0], [1.0, 0.01], [-1.0, 0.0], [0.0, 1.0]],
    )
    selected = select_visual_representatives(candidates, policy="V-LAST", limit=3)
    assert [item.source_key for item in selected] == [6, 7, 5]


def test_v_core_deduplicates_and_keeps_two_recent_representatives() -> None:
    old = _candidate(
        [1, 2, 3],
        [0.9, 0.8, 0.7],
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    current = _candidate(
        [2, 4, 5],
        [0.99, 0.6, 0.95],
        [[0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]],
    )
    selected = select_visual_representatives(
        old + current, policy="V-CORE", limit=4, recent_keys={4, 5}
    )
    keys = [item.source_key for item in selected]
    assert len(keys) == len(set(keys))
    assert {4, 5} <= set(keys)
    assert 2 in keys


def test_update_visual_state_uses_generation_and_zeroes_stale_representatives() -> None:
    state = ObjectVisualState.empty(
        batch_size=1,
        capacity=100,
        representatives=8,
        feature_dim=128,
        device="cpu",
        dtype=torch.float32,
    )
    features = state.features.clone()
    valid = state.valid.clone()
    quality = state.quality.clone()
    stages = state.source_stage.clone()
    keys = state.source_key.clone()
    generations = state.generations.clone()
    features[0, 0, 0, :2] = torch.tensor([3.0, 4.0])
    valid[0, 0, 0] = True
    quality[0, 0, 0] = 0.8
    stages[0, 0, 0] = 1
    keys[0, 0, 0] = 1001
    generations[0, 0] = 2
    state = replace(
        state,
        features=features,
        valid=valid,
        quality=quality,
        source_stage=stages,
        source_key=keys,
        generations=generations,
    )
    updated = update_visual_state(
        state,
        slot_candidates=[
            [[VisualCandidate(torch.nn.functional.pad(torch.tensor([1.0, 0.0]), (0, 126)), torch.tensor(0.9), 2, 2002)]]
            + [[] for _ in range(99)]
        ],
        slot_generations=torch.tensor([[3] + [-1] * 99]),
        policy="V-LAST",
    )
    assert updated.valid[0, 0, 0].item() is True
    assert updated.source_key[0, 0, 0].item() == 2002
    assert updated.generations[0, 0].item() == 3
    assert updated.valid[0, 0].sum().item() == 1
    assert torch.equal(updated.features[0, 1], torch.zeros_like(updated.features[0, 1]))


def test_visual_read_has_zero_residual_when_no_valid_representatives() -> None:
    read = ObjectVisualRead()
    state = ObjectVisualState.empty(
        batch_size=1,
        capacity=100,
        representatives=8,
        feature_dim=128,
        device="cpu",
        dtype=torch.float32,
    )
    queries = torch.randn(1, 100, 128)
    query_to_slot = torch.full((1, 100), -1, dtype=torch.long)
    generations = torch.full((1, 100), -1, dtype=torch.long)
    output = read(queries, state, query_to_slot, generations)
    assert torch.equal(output, queries)
    assert read.last_diagnostics["valid_read_count"] == 0


def test_visual_read_null_winner_has_strict_zero_residual() -> None:
    read = ObjectVisualRead()
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
        read.null_key[0] = 1.0
    state = ObjectVisualState.empty(
        batch_size=1,
        device="cpu",
        dtype=torch.float32,
    )
    features = state.features.clone()
    valid = state.valid.clone()
    qualities = state.quality.clone()
    stages = state.source_stage.clone()
    keys = state.source_key.clone()
    generations = state.generations.clone()
    features[0, 0, 0, 0] = -1.0
    valid[0, 0, 0] = True
    qualities[0, 0, 0] = 1.0
    stages[0, 0, 0] = 0
    keys[0, 0, 0] = 1
    generations[0, 0] = 0
    state = replace(
        state,
        features=features,
        valid=valid,
        quality=qualities,
        source_stage=stages,
        source_key=keys,
        generations=generations,
    )
    queries = torch.zeros(1, 100, 128)
    queries[0, 0, 0] = 1.0
    route = torch.full((1, 100), -1, dtype=torch.long)
    route[0, 0] = 0
    query_generations = torch.full((1, 100), -1, dtype=torch.long)
    query_generations[0, 0] = 0
    output = read(queries, state, route, query_generations)
    assert torch.equal(output, queries)
    assert read.last_diagnostics["null_win_count"] == 1


def test_v_last_uses_normalized_cosine_coverage_after_first_quality_pick() -> None:
    candidates = _candidate(
        [1, 2, 3],
        [1.0, 0.99, 0.1],
        [[1.0, 0.0], [10.0, 0.1], [-1.0, 0.0]],
    )
    selected = select_visual_representatives(candidates, policy="V-LAST", limit=2)
    assert [item.source_key for item in selected] == [1, 3]


def test_visual_candidates_use_current_prediction_rows_only() -> None:
    features = torch.zeros(1, 3, 128)
    features[0, 0, 0] = 10.0
    features[0, 1, 1] = 2.0
    features[0, 2, 2] = 3.0
    masks = torch.full((3, 100), -8.0)
    masks[:, 0] = torch.tensor([8.0, 2.0, 1.0])
    logits = torch.full((1, 100, 19), -8.0)
    logits[0, 0, 0] = 8.0
    route = torch.full((1, 100), -1, dtype=torch.long)
    route[0, 0] = 3
    meta = StageMeta(
        reference_id="reference",
        episode_id="episode",
        scan_ids_in_window=("old", "new"),
        absolute_stage_index=4,
        local_stage_ids=torch.tensor([0, 1, 1]),
        original_vertex_ids=(torch.tensor([0]), torch.tensor([0, 1])),
        scan_vertex_offsets=torch.tensor([0, 1, 3]),
        point2segment=torch.arange(3),
        segment_stage_ids=torch.tensor([0, 1, 1]),
        augmentation_transform_id="identity",
        coordinate_frame_id="reference",
        voxel_inverse=torch.arange(3),
        full_resolution_point2segment=torch.arange(3),
    )
    slots = build_visual_slot_candidates(
        {
            "pred_logits": logits,
            "pred_masks": [masks],
            "task_memory_visual_features": features,
            "task_memory_visual_padding_mask": torch.zeros(1, 3, dtype=torch.bool),
        },
        [meta],
        query_to_slot=route,
        background_class=18,
        mask_threshold=0.5,
    )
    candidates = slots[0][3]
    assert len(candidates) == 2
    assert all(item.source_stage == 4 for item in candidates)
    assert all(item.feature[0].item() == 0.0 for item in candidates)
