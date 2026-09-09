from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import pytest
import torch

from datasets.task_memory_episode import StageMeta
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.task_memory_cache import (
    CACHE_LIMIT_BYTES,
    TaskMemoryCacheError,
    build_episode_cache_payload,
    build_evaluation_cache_key,
    build_stage_cache_record,
    cache_key_sha256,
    load_task_memory_cache,
    prediction_from_cache_record,
    write_task_memory_cache,
)
from scripts.task_memory_metrics import (
    aggregate_identity_event_diagnostics,
    compute_cached_task_metrics,
    replay_lag1_prefixes,
)


def _meta(stage: int, scan_ids: tuple[str, ...]) -> StageMeta:
    point_count = len(scan_ids)
    local_stages = torch.arange(point_count, dtype=torch.long)
    return StageMeta(
        reference_id="reference-0",
        episode_id="episode-0",
        scan_ids_in_window=scan_ids,
        absolute_stage_index=stage,
        local_stage_ids=local_stages,
        original_vertex_ids=tuple(torch.tensor([0]) for _ in scan_ids),
        scan_vertex_offsets=torch.arange(point_count + 1, dtype=torch.long),
        point2segment=torch.arange(point_count, dtype=torch.long),
        segment_stage_ids=local_stages.clone(),
        augmentation_transform_id="identity-v1",
        coordinate_frame_id="reference-0",
        voxel_inverse=torch.arange(point_count, dtype=torch.long),
        full_resolution_point2segment=torch.arange(point_count, dtype=torch.long),
    )


def _prediction(stage: int, point_count: int) -> OfficialTaskPrediction:
    temporal_stages = (
        torch.tensor([0], dtype=torch.long)
        if stage == 0
        else torch.tensor([0, 1], dtype=torch.long)
    )
    assert temporal_stages.numel() == point_count
    masks = torch.ones((point_count, 1), dtype=torch.bool)
    prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.25 + stage / 10], dtype=torch.float32),
        pred_classes=torch.tensor([3], dtype=torch.long),
        source_query_ids=torch.tensor([0], dtype=torch.long),
        source_class_ids=torch.tensor([3], dtype=torch.long),
        temporal_stages=temporal_stages,
        latest_stage_index=0 if stage == 0 else 1,
        latest_stage_masks=masks[-1:],
    )
    prediction.validate()
    return prediction


def _target(identity: int) -> dict[str, object]:
    return {
        "gt_ids": torch.tensor([identity], dtype=torch.long),
        "gt_classes": torch.tensor([3], dtype=torch.long),
        "gt_masks": torch.ones((1, 1), dtype=torch.bool),
        "changes": torch.zeros(1, dtype=torch.long),
        "change_labels_valid": False,
        "change_label_semantics": (
            "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
        ),
        "gt_class_semantics": "rescene_model_index_0_based",
    }


def _key(horizon: int) -> dict[str, object]:
    return build_evaluation_cache_key(
        population_id="development-smoke",
        reference_id="reference-0",
        episode_id="episode-0",
        history_scan_ids=tuple(f"scan-{index}" for index in range(horizon)),
        window_mode="local_pair",
        checkpoint_sha256="1" * 64,
        resolved_config_sha256="2" * 64,
        data_contract_sha256="3" * 64,
        state_contract_sha256="4" * 64,
        initial_state_sha256="0" * 64,
        output_policy="lag1-v1",
        postprocess_sha256="5" * 64,
        evaluation_seed=45,
    )


def _episode(horizon: int) -> dict[str, object]:
    stages = []
    state_before = "0" * 64
    for stage in range(horizon):
        state_after = f"{stage + 1:064x}"
        scans = (
            ("scan-0",)
            if stage == 0
            else (f"scan-{stage - 1}", f"scan-{stage}")
        )
        stages.append(
            build_stage_cache_record(
                prediction=_prediction(stage, len(scans)),
                identity_map={0: (7, 0)},
                stage_meta=_meta(stage, scans),
                target=_target(7),
                state_before_sha256=state_before,
                state_after_sha256=state_after,
                event_diagnostics={
                    "births": int(stage == 0),
                    "matched_births": int(stage == 0),
                    "reactivations": 0,
                    "matched_reactivations": 0,
                    "rejected_births": 0,
                },
            )
        )
        state_before = state_after
    return build_episode_cache_payload(key=_key(horizon), stages=stages)


def _full_history_episode(horizon: int) -> dict[str, object]:
    key = _key(horizon)
    key["window_mode"] = "full_history"
    stages = []
    state_before = "0" * 64
    for stage in range(horizon):
        scans = tuple(f"scan-{index}" for index in range(stage + 1))
        masks = torch.ones((stage + 1, 1), dtype=torch.bool)
        temporal_stages = torch.arange(stage + 1, dtype=torch.long)
        prediction = OfficialTaskPrediction(
            pred_masks=masks,
            pred_scores=torch.tensor([0.5], dtype=torch.float32),
            pred_classes=torch.tensor([3]),
            source_query_ids=torch.tensor([0]),
            source_class_ids=torch.tensor([3]),
            temporal_stages=temporal_stages,
            latest_stage_index=stage,
            latest_stage_masks=masks[-1:],
        )
        state_after = f"{stage + 1:064x}"
        stages.append(
            build_stage_cache_record(
                prediction=prediction,
                identity_map={0: (7, 0)},
                stage_meta=_meta(stage, scans),
                target=_target(7),
                state_before_sha256=state_before,
                state_after_sha256=state_after,
                event_diagnostics={
                    "births": int(stage == 0),
                    "matched_births": int(stage == 0),
                    "reactivations": 0,
                    "matched_reactivations": 0,
                    "rejected_births": 0,
                },
            )
        )
        state_before = state_after
    return build_episode_cache_payload(key=key, stages=stages)


def test_cache_key_binds_history_model_config_state_policy_and_postprocess() -> None:
    key = _key(3)
    baseline = cache_key_sha256(key)
    mutations = {
        "history_scan_ids": ["scan-0", "scan-1", "scan-x"],
        "checkpoint_sha256": "a" * 64,
        "resolved_config_sha256": "b" * 64,
        "state_contract_sha256": "c" * 64,
        "initial_state_sha256": "e" * 64,
        "output_policy": "commit0-v1",
        "postprocess_sha256": "d" * 64,
    }

    for field, value in mutations.items():
        changed = deepcopy(key)
        changed[field] = value
        assert cache_key_sha256(changed) != baseline


def test_stage_cache_uses_one_lossless_mask_payload_for_all_reducers() -> None:
    source = _prediction(1, 2)
    stage = build_stage_cache_record(
        prediction=source,
        identity_map={0: (7, 0)},
        stage_meta=_meta(1, ("scan-0", "scan-1")),
        target=_target(7),
        state_before_sha256="1" * 64,
        state_after_sha256="2" * 64,
        event_diagnostics={
            "births": 0,
            "matched_births": 0,
            "reactivations": 0,
            "matched_reactivations": 0,
            "rejected_births": 0,
        },
    )

    restored = prediction_from_cache_record(stage)
    assert torch.equal(restored.pred_masks, source.pred_masks)
    assert stage["prediction"]["pred_masks"]["encoding"] == (
        "numpy-packbits-little-v1"
    )
    assert "reducer" not in repr(stage)


@pytest.mark.parametrize("horizon", range(1, 6))
def test_episode_cache_requires_continuous_empty_state_trajectory(horizon: int) -> None:
    payload = _episode(horizon)

    assert len(payload["stages"]) == horizon
    assert payload["stages"][0]["state_before_sha256"] == "0" * 64
    for previous, current in zip(payload["stages"], payload["stages"][1:]):
        assert previous["state_after_sha256"] == current["state_before_sha256"]

    broken = deepcopy(payload)
    if horizon > 1:
        broken["stages"][1]["state_before_sha256"] = "f" * 64
        with pytest.raises(TaskMemoryCacheError, match="state trajectory"):
            build_episode_cache_payload(key=broken["key"], stages=broken["stages"])


def test_cache_write_is_atomic_and_rejects_the_40_gib_budget(tmp_path: Path) -> None:
    payload = _episode(2)

    with pytest.raises(TaskMemoryCacheError, match="cache cap"):
        write_task_memory_cache(tmp_path, payload, max_total_bytes=1)
    assert list(tmp_path.iterdir()) == []

    record = write_task_memory_cache(
        tmp_path, payload, max_total_bytes=CACHE_LIMIT_BYTES
    )
    loaded = load_task_memory_cache(
        tmp_path / record["filename"], expected_key=payload["key"]
    )
    assert loaded["content_sha256"] == payload["content_sha256"]


class _MetricSpy:
    point_counts: ClassVar[list[int]] = []

    def __init__(self) -> None:
        self.count = 0

    def update(self, pair) -> None:
        self.point_counts.append(pair.prediction["pred_masks"].shape[0])
        self.count += 1

    def compute(self) -> dict[str, float]:
        assert self.count > 0
        return {
            "prefix_overall_mAP": 0.11,
            "local_current_AP": 0.22,
            "t_REC": 0.33,
            "t_mAP": 0.44,
            "t_mAP25": 0.55,
            "t_mAP50": 0.66,
        }


def test_lag1_replay_versions_full_prefix_and_reuses_for_three_reducers() -> None:
    payload = _episode(5)
    replay = replay_lag1_prefixes(
        payload,
        reducers=("mean", "latest", "max"),
        class_mapper=lambda value: value,
    )

    assert set(replay) == {"mean", "latest", "max"}
    for reducer, prefixes in replay.items():
        assert [item.pair.horizon for item in prefixes] == [1, 2, 3, 4, 5]
        assert prefixes[-1].pair.prediction["pred_masks"].shape[0] == 5
        assert prefixes[-1].pair.target["temporal_stages"].tolist() == [0, 1, 2, 3, 4]
        assert prefixes[-1].revision_versions == {
            "scan-0": 1,
            "scan-1": 1,
            "scan-2": 1,
            "scan-3": 1,
            "scan-4": 0,
        }
        assert prefixes[-1].pair.prediction["pred_scores"].numel() == 1
        assert reducer == prefixes[-1].score_reducer

    _MetricSpy.point_counts = []
    rows = compute_cached_task_metrics(
        [payload],
        reducers=("mean", "latest", "max"),
        class_mapper=lambda value: value,
        accumulator_factory=_MetricSpy,
    )
    assert len(rows) == 12
    assert set(_MetricSpy.point_counts) == {2, 3, 4, 5}
    assert all(row["prefix_overall_mAP"] == 0.11 for row in rows)


def test_full_history_cache_replays_only_last_two_scans_into_lag1_publisher() -> None:
    payload = _full_history_episode(5)

    replay = replay_lag1_prefixes(
        payload,
        reducers=("mean",),
        class_mapper=lambda value: value,
    )["mean"]

    assert [item.pair.prediction["pred_masks"].shape[0] for item in replay] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert replay[-1].revision_versions["scan-3"] == 1
    assert replay[-1].revision_versions["scan-4"] == 0


def test_identity_event_diagnostics_keep_na_denominators_and_error_counts() -> None:
    empty = aggregate_identity_event_diagnostics([])
    assert empty["false_birth_rate"] is None
    assert empty["false_reactivation_rate"] is None

    result = aggregate_identity_event_diagnostics(
        [
            {
                "births": 3,
                "matched_births": 2,
                "reactivations": 2,
                "matched_reactivations": 1,
                "rejected_births": 4,
            },
            {
                "births": 1,
                "matched_births": 1,
                "reactivations": 0,
                "matched_reactivations": 0,
                "rejected_births": 1,
            },
        ]
    )
    assert result["false_birth_count"] == 1
    assert result["false_birth_rate"] == pytest.approx(0.25)
    assert result["false_reactivation_count"] == 1
    assert result["false_reactivation_rate"] == pytest.approx(0.5)
    assert result["rejected_birth_count"] == 5
