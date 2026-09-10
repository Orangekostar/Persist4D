from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from omegaconf import OmegaConf

import scripts.evaluate_task_memory as evaluation_module
from datasets.task_memory_episode import NativeEpisodeMaster, StageMeta
from models.task_memory_state import TaskMemoryConfig
from scripts.evaluate_task_memory import (
    COMMON_POPULATION_ID,
    NATIVE_POPULATION_ID,
    _compute_population_rows,
    _csv_bytes,
    _episode_spec,
    _population_horizons,
    _retention_rows_for_population,
    _rio_population_base,
    _state_contract_sha256,
    select_development_masters,
)
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
    build_retention_rows,
    compute_cached_identity_metrics,
    compute_cached_task_metrics,
    replay_commit0_prefixes,
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


def _master(
    *, reference_id: str, horizon: int, role: str
) -> NativeEpisodeMaster:
    scan_ids = tuple(f"scene0001_{index:02d}" for index in range(horizon))
    return NativeEpisodeMaster(
        reference_id=reference_id,
        sequence_id="-".join(scan_ids),
        scan_ids=scan_ids,
        scan_indices=tuple(range(horizon)),
        role=role,
        context_index=0,
    )


def test_population_selection_separates_common_h5_and_additional_native_h2_h4() -> None:
    masters = (
        _master(reference_id="common", horizon=5, role="development"),
        _master(reference_id="native-2", horizon=2, role="additional_native_refs"),
        _master(reference_id="native-3", horizon=3, role="additional_native_refs"),
        _master(reference_id="native-4", horizon=4, role="additional_native_refs"),
        _master(reference_id="adaptation", horizon=3, role="adaptation"),
    )

    common = select_development_masters(
        masters,
        population_id=COMMON_POPULATION_ID,
        smoke_master_count=None,
    )
    native = select_development_masters(
        masters,
        population_id=NATIVE_POPULATION_ID,
        smoke_master_count=None,
    )

    assert [(master.reference_id, len(master.scan_ids)) for master in common] == [
        ("common", 5)
    ]
    assert [(master.reference_id, len(master.scan_ids)) for master in native] == [
        ("native-2", 2),
        ("native-3", 3),
        ("native-4", 4),
    ]


def test_native_episode_spec_uses_the_real_master_horizon() -> None:
    master = _master(
        reference_id="native-3", horizon=3, role="additional_native_refs"
    )

    spec = _episode_spec(master, index=7)

    assert spec.horizon == 3
    assert spec.bucket == "T3"
    assert spec.scan_ids == master.scan_ids


def test_native_population_does_not_form_a_cross_population_retention_curve() -> None:
    metric_rows = [
        {"T": horizon, "policy": "lag1", "reducer": "mean", "t_mAP": value}
        for horizon, value in ((2, 0.5), (3, 0.4), (4, 0.3))
    ]

    rows, status = _retention_rows_for_population(
        metric_rows,
        population_id=NATIVE_POPULATION_ID,
    )

    assert rows == []
    assert status == "NOT_APPLICABLE_VARYING_NATIVE_POPULATION"


def test_native_population_loads_each_real_horizon_and_counts_only_terminals() -> None:
    assert _population_horizons(COMMON_POPULATION_ID) == (5,)
    assert _population_horizons(NATIVE_POPULATION_ID) == (2, 3, 4)

    task_rows, identity_rows = _compute_population_rows(
        [_episode(horizon) for horizon in (2, 3, 4)],
        population_id=NATIVE_POPULATION_ID,
        reducers=("mean", "latest", "max"),
        class_mapper=lambda value: value,
        accumulator_factory=_MetricSpy,
    )

    assert len(task_rows) == 12
    assert {(row["T"], row["episode_count"]) for row in task_rows} == {
        (2, 1),
        (3, 1),
        (4, 1),
    }
    assert [row["T"] for row in identity_rows] == [2, 3, 4]


def test_native_population_uses_the_validation_dataset_source(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "data": {
                "train_dataset": {"sentinel": "train"},
                "validation_dataset": {
                    "_target_": "types.SimpleNamespace",
                    "data_dir": "data/processed/rio",
                    "mode": "validation",
                    "sentinel": "validation",
                    "temporal_window": 2,
                },
            }
        }
    )

    dataset = _rio_population_base(
        config,
        data_root=tmp_path,
        horizon=4,
        population_id=NATIVE_POPULATION_ID,
    )

    assert dataset.sentinel == "validation"
    assert dataset.mode == "validation"
    assert dataset.temporal_window == 4
    assert dataset.data_dir == str(tmp_path / "processed/rio")


def test_native_smoke_panel_covers_h2_h3_h4() -> None:
    masters = tuple(
        _master(
            reference_id=reference_id,
            horizon=horizon,
            role="additional_native_refs",
        )
        for reference_id in ("native-a", "native-b", "native-c")
        for horizon in (2, 3, 4)
    )

    smoke = select_development_masters(
        masters,
        population_id=NATIVE_POPULATION_ID,
        smoke_master_count=3,
    )

    assert {len(master.scan_ids) for master in smoke} == {2, 3, 4}
    assert len({master.reference_id for master in smoke}) == 3


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
        output_policy="commit0+lag1-v1",
        postprocess_sha256="5" * 64,
        evaluation_seed=45,
    )


def test_state_contract_hash_uses_the_runtime_task_memory_config() -> None:
    system = SimpleNamespace(
        config=SimpleNamespace(
            task_memory_training=SimpleNamespace(state_enabled=True)
        ),
        model=SimpleNamespace(
            task_memory_capacity=100,
            task_memory_config=TaskMemoryConfig(
                association_threshold=0.55,
                class_weight=0.25,
                max_update_rate=0.2,
                update_rate=0.2,
            ),
        ),
    )

    observed = _state_contract_sha256(system)

    assert isinstance(observed, str)
    assert len(observed) == 64
    system.model.task_memory_config = TaskMemoryConfig(update_mode="last")
    assert _state_contract_sha256(system) != observed


def test_evaluation_csv_writes_zero_denominators_as_na() -> None:
    encoded = _csv_bytes([{"count": 0, "rate": None}]).decode("utf-8")

    assert encoded == "count,rate\n0,N/A\n"
    with pytest.raises(Exception, match="scalar"):
        _csv_bytes([{"invalid": {"nested": True}}])


def test_evaluation_runtime_applies_the_frozen_seed_and_device(monkeypatch) -> None:
    events = []

    @contextmanager
    def fake_runtime(seed, device):
        events.append(("enter", seed, device))
        yield
        events.append(("exit", seed, device))

    monkeypatch.setattr(
        evaluation_module, "deterministic_inference_runtime", fake_runtime
    )
    device = torch.device("cuda:0")

    with evaluation_module._evaluation_runtime(device):
        events.append(("body",))

    assert events == [
        ("enter", 45, device),
        ("body",),
        ("exit", 45, device),
    ]


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


def test_shared_forward_cache_adds_commit0_mean_without_repeating_forward() -> None:
    payload = _episode(5)

    commit0 = replay_commit0_prefixes(
        payload,
        reducers=("mean",),
        class_mapper=lambda value: value,
    )["mean"]
    assert commit0[-1].revision_versions == {
        "scan-0": 0,
        "scan-1": 0,
        "scan-2": 0,
        "scan-3": 0,
        "scan-4": 0,
    }
    assert commit0[-1].accounting.lag1_buffer_bytes == 0

    _MetricSpy.point_counts = []
    rows = compute_cached_task_metrics(
        [payload],
        reducers=("mean", "latest", "max"),
        include_commit0=True,
        class_mapper=lambda value: value,
        accumulator_factory=_MetricSpy,
    )

    assert len(rows) == 16
    assert {(row["policy"], row["reducer"]) for row in rows} == {
        ("commit0", "mean"),
        ("lag1", "mean"),
        ("lag1", "latest"),
        ("lag1", "max"),
    }
    assert len(_MetricSpy.point_counts) == 16


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


def test_one_query_can_publish_separate_class_preserving_trajectories() -> None:
    prediction = OfficialTaskPrediction(
        pred_masks=torch.ones((1, 2), dtype=torch.bool),
        pred_scores=torch.tensor([0.7, 0.6]),
        pred_classes=torch.tensor([3, 4]),
        source_query_ids=torch.tensor([0, 0]),
        source_class_ids=torch.tensor([3, 4]),
        temporal_stages=torch.tensor([0]),
        latest_stage_index=0,
        latest_stage_masks=torch.ones((1, 2), dtype=torch.bool),
    )
    stage = build_stage_cache_record(
        prediction=prediction,
        identity_map={0: (7, 0)},
        stage_meta=_meta(0, ("scan-0",)),
        target=_target(7),
        state_before_sha256="0" * 64,
        state_after_sha256="1" * 64,
        event_diagnostics={
            "births": 1,
            "matched_births": 1,
            "reactivations": 0,
            "matched_reactivations": 0,
            "rejected_births": 0,
        },
    )
    payload = build_episode_cache_payload(key=_key(1), stages=[stage])

    prefix = replay_lag1_prefixes(
        payload, reducers=("mean",), class_mapper=lambda value: value
    )["mean"][0]

    assert prefix.pair.prediction["pred_classes"].tolist() == [3, 4]


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


def test_cached_identity_metrics_reuse_published_ids_and_keep_na_rates() -> None:
    rows = compute_cached_identity_metrics(
        [_episode(5)], class_mapper=lambda value: value
    )

    assert [row["T"] for row in rows] == [2, 3, 4, 5]
    assert rows[0]["identity_transition_opportunities"] == 1
    assert rows[-1]["identity_transition_opportunities"] == 4
    assert all(row["deployment_id_switches"] == 0 for row in rows)
    assert all(row["normalized_id_switch_rate"] == 0.0 for row in rows)
    assert all(row["gap_recovery_accuracy"] is None for row in rows)
    assert all(row["gap_recovery_attempt_coverage"] is None for row in rows)


def test_retention_uses_each_reducer_t2_and_keeps_zero_denominator_na() -> None:
    rows = []
    for reducer, values in {
        "mean": (0.5, 0.4, 0.3, 0.25),
        "max": (0.0, 0.1, 0.2, 0.3),
    }.items():
        for horizon, value in zip((2, 3, 4, 5), values, strict=True):
            rows.append(
                {
                    "T": horizon,
                    "policy": "lag1",
                    "reducer": reducer,
                    "t_mAP": value,
                }
            )

    retention = build_retention_rows(rows)

    mean_t5 = next(
        row for row in retention if row["reducer"] == "mean" and row["T"] == 5
    )
    max_t5 = next(
        row for row in retention if row["reducer"] == "max" and row["T"] == 5
    )
    assert mean_t5["relative_t_mAP_retention"] == pytest.approx(0.5)
    assert max_t5["relative_t_mAP_retention"] is None
