from __future__ import annotations

import pytest
import torch

from datasets.task_memory_episode import StageMeta
from scripts.run_task_memory_controls import (
    ControlRunnerError,
    _window_observation,
    observation_payload,
    prediction_observation_from_payload,
    run_control_trajectory,
    validate_observation_supplement,
)


def _meta(stage: int) -> StageMeta:
    scan_ids = ("scan-0",) if stage == 0 else (f"scan-{stage - 1}", f"scan-{stage}")
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


def _payload(feature: tuple[float, float], *, confidence: float = 0.8) -> dict:
    return {
        "features": torch.tensor([[feature]], dtype=torch.float32),
        "class_prob": torch.tensor([[[0.7, 0.3]]], dtype=torch.float32),
        "confidence": torch.tensor([[confidence]], dtype=torch.float32),
        "valid": torch.tensor([[True]]),
        "current_supported": torch.tensor([[True]]),
        "previous_supported": torch.tensor([[False]]),
    }


def _prediction() -> dict:
    return {
        "pred_masks": torch.tensor([[True]]),
        "pred_scores": torch.tensor([0.8]),
        "pred_classes": torch.tensor([1]),
        "source_query_ids": torch.tensor([0]),
        "source_class_ids": torch.tensor([1]),
        "temporal_stages": torch.tensor([0]),
        "latest_stage_index": 0,
        "latest_stage_masks": torch.tensor([[True]]),
    }


def _supplement_stage(feature: tuple[float, float]) -> dict:
    return {
        "observation": _payload(feature),
        "prediction": _prediction(),
        "base_overlap": {
            "exact": True,
            "generated_candidate_count": 1,
            "base_candidate_count": 1,
            "common_candidate_count": 1,
            "score_max_abs": 0.0,
            "aligned_mask_iou_mean": 1.0,
        },
    }


def test_observation_payload_is_detached_prediction_only_round_trip() -> None:
    raw = _payload((1.0, 0.0))
    observation = prediction_observation_from_payload(raw)
    payload = observation_payload(observation)

    assert all(not value.requires_grad for value in payload.values())
    assert not any("gt" in key or "target" in key for key in payload)
    restored = prediction_observation_from_payload(payload)
    for field in payload:
        assert torch.equal(getattr(restored, field), payload[field])


def test_observation_supplement_is_bound_to_base_cache_and_has_no_labels() -> None:
    supplement = {
        "schema_version": "task-memory-control-observations-v2",
        "provenance": {"source_commit": "a" * 40},
        "base_cache": {
            "filename": "episode.pt",
            "sha256": "b" * 64,
            "sequence_id": "scan-0-scan-1",
            "reference_id": "reference-0",
        },
        "stages": [
            _supplement_stage((1.0, 0.0)),
            _supplement_stage((0.0, 1.0)),
        ],
    }
    validated = validate_observation_supplement(
        supplement,
        expected_provenance={"source_commit": "a" * 40},
        expected_base_cache={
            "filename": "episode.pt",
            "sha256": "b" * 64,
            "sequence_id": "scan-0-scan-1",
            "reference_id": "reference-0",
        },
        expected_stage_count=2,
    )
    assert len(validated["stages"]) == 2

    wrong = dict(supplement)
    wrong["base_cache"] = dict(supplement["base_cache"], sha256="c" * 64)
    with pytest.raises(ControlRunnerError, match="base cache"):
        validate_observation_supplement(
            wrong,
            expected_provenance={"source_commit": "a" * 40},
            expected_base_cache=supplement["base_cache"],
            expected_stage_count=2,
        )

    labeled = dict(supplement)
    labeled_stage = _supplement_stage((1.0, 0.0))
    labeled_stage["observation"] = dict(
        labeled_stage["observation"], gt_ids=torch.tensor([1])
    )
    labeled["stages"] = [labeled_stage]
    with pytest.raises(ControlRunnerError, match="prediction-only"):
        validate_observation_supplement(
            labeled,
            expected_provenance={"source_commit": "a" * 40},
            expected_base_cache=supplement["base_cache"],
            expected_stage_count=1,
        )


def test_control_trajectory_uses_same_routes_but_distinct_update_rules() -> None:
    observations = [
        prediction_observation_from_payload(_payload((1.0, 0.0))),
        prediction_observation_from_payload(_payload((0.8, 0.6), confidence=0.5)),
    ]
    metas = [_meta(0), _meta(1)]

    last = run_control_trajectory(observations, metas, update_mode="last", capacity=2)
    ema = run_control_trajectory(
        observations, metas, update_mode="fixed_ema", capacity=2
    )

    assert last.identity_maps == ema.identity_maps == ({0: (0, 0)}, {0: (0, 0)})
    assert last.diagnostics.inherited_routes == 1
    assert ema.diagnostics.inherited_routes == 1
    assert last.diagnostics.births == ema.diagnostics.births == 1
    assert not torch.equal(last.final_state.embedding, ema.final_state.embedding)
    assert last.final_state.embedding[0, 0].tolist() == pytest.approx([0.8, 0.6])
    assert ema.final_state.embedding[0, 0].tolist() == pytest.approx(
        [0.9922779, 0.1240347]
    )


def test_previous_only_observation_is_not_promoted_to_active() -> None:
    first = prediction_observation_from_payload(_payload((1.0, 0.0)))
    second_payload = _payload((1.0, 0.0))
    second_payload = dict(
        second_payload,
        valid=torch.tensor([[True]]),
        current_supported=torch.tensor([[False]]),
        previous_supported=torch.tensor([[True]]),
    )
    second = prediction_observation_from_payload(second_payload)

    trajectory = run_control_trajectory(
        [first, second], [_meta(0), _meta(1)], update_mode="last", capacity=1
    )

    assert trajectory.identity_maps[-1] == {0: (0, 0)}
    assert trajectory.diagnostics.previous_only_queries == 1
    assert trajectory.diagnostics.dormant_routes == 1
    assert not trajectory.final_state.active.any().item()
    assert trajectory.final_state.last_seen.tolist() == [[0]]


def test_window_observation_keeps_confident_previous_only_query_valid() -> None:
    local = prediction_observation_from_payload(
        {
            "features": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            "class_prob": torch.tensor([[[0.7, 0.3], [0.2, 0.8]]]),
            "confidence": torch.tensor([[0.8, 0.4]]),
            "valid": torch.tensor([[False, False]]),
            "current_supported": torch.tensor([[False, True]]),
            "previous_supported": torch.tensor([[True, False]]),
        }
    )
    logits = torch.tensor(
        [
            [8.0, -8.0],
            [8.0, -8.0],
            [-8.0, 8.0],
            [-8.0, 8.0],
        ]
    )

    observation = _window_observation(
        local_observation=local,
        output={"pred_masks": [logits]},
        segment_stages=torch.tensor([0, 0, 1, 1]),
        latest_stage=1,
        confidence_threshold=0.5,
        mask_threshold=0.5,
        minimum_mask_support=1,
    )

    assert observation.current_supported.tolist() == [[False, True]]
    assert observation.previous_supported.tolist() == [[True, False]]
    assert observation.valid.tolist() == [[True, False]]
