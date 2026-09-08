from __future__ import annotations

import copy
import threading

import pytest

from scripts.evaluate_persist4d_allt import (
    AllTEvaluationError,
    _checkpoint_identity,
    _compact_metric_bundle,
    _compute_metric_values,
    _new_model_forward_count,
    _resolve_metric_reducers,
    build_sequence_cache_key,
    build_stage_requests,
    rank_development_checkpoints,
    sequence_cache_key_sha256,
    validate_result_rows,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _stage_requests(window_mode: str = "local_pair") -> tuple[dict[str, object], ...]:
    return build_stage_requests(
        master_sequence_id="scene0000_00-scene0000_01-scene0000_02-scene0000_03-scene0000_04",
        reference_scene_id="reference-0",
        order_id="canonical",
        scan_ids=tuple(f"scene0000_0{index}" for index in range(5)),
        scan_indices=(10, 11, 12, 13, 14),
        context_index=7,
        window_mode=window_mode,
    )


def test_stage_requests_preserve_one_continuous_episode() -> None:
    local = _stage_requests("local_pair")
    assert [request["stage_index"] for request in local] == list(range(5))
    assert [request["history_scan_ids"] for request in local] == [
        ["scene0000_00"],
        ["scene0000_00", "scene0000_01"],
        ["scene0000_00", "scene0000_01", "scene0000_02"],
        ["scene0000_00", "scene0000_01", "scene0000_02", "scene0000_03"],
        [
            "scene0000_00",
            "scene0000_01",
            "scene0000_02",
            "scene0000_03",
            "scene0000_04",
        ],
    ]
    assert [request["inference_scan_indices"] for request in local] == [
        [10],
        [10, 11],
        [11, 12],
        [12, 13],
        [13, 14],
    ]

    full = _stage_requests("full_history")
    assert [request["inference_scan_indices"] for request in full] == [
        [10],
        [10, 11],
        [10, 11, 12],
        [10, 11, 12, 13],
        [10, 11, 12, 13, 14],
    ]


def test_sequence_cache_key_binds_checkpoint_modules_seed_and_prefixes() -> None:
    key = build_sequence_cache_key(
        checkpoint_sha256=SHA_A,
        module_config_sha256=SHA_B,
        population_id="development_v1",
        evaluation_seed=45,
        postprocess_version="official-rescene-v1",
        stage_requests=_stage_requests(),
    )
    original = sequence_cache_key_sha256(key)
    assert len(original) == 64
    for field, value in (
        ("checkpoint_sha256", SHA_C),
        ("module_config_sha256", SHA_C),
        ("evaluation_seed", 46),
        ("postprocess_version", "official-rescene-v2"),
    ):
        changed = copy.deepcopy(key)
        changed[field] = value
        assert sequence_cache_key_sha256(changed) != original
    changed_prefix = copy.deepcopy(key)
    changed_prefix["stage_requests"] = list(
        build_stage_requests(
            master_sequence_id=key["master_sequence_id"],
            reference_scene_id=key["reference_scene_id"],
            order_id=key["order_id"],
            scan_ids=(*tuple(f"scene0000_0{index}" for index in range(4)), "scene9999_00"),
            scan_indices=(10, 11, 12, 13, 14),
            context_index=7,
            window_mode=key["window_mode"],
        )
    )
    assert sequence_cache_key_sha256(changed_prefix) != original


def _metric_row(checkpoint: str, horizon: int, value: float) -> dict[str, object]:
    return {
        "checkpoint_sha256": checkpoint,
        "T": horizon,
        "t_mAP": value,
    }


def test_development_selection_maximizes_worst_horizon_then_mean() -> None:
    baseline = [
        _metric_row(SHA_C, horizon, 0.20) for horizon in range(2, 6)
    ]
    candidates = [
        *[
            _metric_row(SHA_A, horizon, value)
            for horizon, value in zip(range(2, 6), (0.21, 0.21, 0.21, 0.21), strict=True)
        ],
        *[
            _metric_row(SHA_B, horizon, value)
            for horizon, value in zip(range(2, 6), (0.40, 0.40, 0.40, 0.205), strict=True)
        ],
    ]
    ranked = rank_development_checkpoints(candidates, baseline)
    assert [row["checkpoint_sha256"] for row in ranked] == [SHA_A, SHA_B]
    assert ranked[0]["minimum_t_map_delta"] == pytest.approx(0.01)
    assert ranked[0]["mean_t_map"] == pytest.approx(0.21)


def test_development_selection_rejects_per_horizon_checkpoint_substitution() -> None:
    baseline = [_metric_row(SHA_C, horizon, 0.2) for horizon in range(2, 6)]
    incomplete = [_metric_row(SHA_A, horizon, 0.3) for horizon in (2, 3, 4)]
    with pytest.raises(AllTEvaluationError, match="exactly T2-T5"):
        rank_development_checkpoints(incomplete, baseline)


def test_result_rows_require_one_checkpoint_to_cover_every_horizon() -> None:
    common = {
        "population_id": "development_v1",
        "model": "C2",
        "checkpoint_sha256": SHA_A,
        "training_seed": 45,
        "evaluation_seed": 45,
        "method": "B4",
        "reducer": "mean",
        "t_mAP": 0.1,
        "t_mAP50": 0.2,
        "t_mAP25": 0.3,
        "t_REC": 0.4,
        "prefix_overall_mAP": 0.5,
        "local_current_AP": 0.6,
        "num_master": 8,
        "num_order_units": 8,
        "num_reference_clusters": 2,
    }
    rows = [{**common, "T": horizon} for horizon in range(2, 6)]
    validate_result_rows(rows)
    with pytest.raises(AllTEvaluationError, match="exactly T2-T5"):
        validate_result_rows(rows[:-1])


def test_forward_count_reports_only_new_method_specific_work() -> None:
    assert _new_model_forward_count(
        window_mode="local_pair", sequence_count=3, reused_count=1
    ) == 10
    assert _new_model_forward_count(
        window_mode="full_history", sequence_count=3, reused_count=1
    ) == 30
    assert _new_model_forward_count(
        window_mode="local_pair", sequence_count=3, reused_count=3
    ) == 0


def test_metric_reducers_can_limit_intermediate_checkpoint_work() -> None:
    assert _resolve_metric_reducers(window_mode="local_pair", requested=None) == (
        "mean",
        "latest",
        "max",
    )
    assert _resolve_metric_reducers(
        window_mode="local_pair", requested=("mean",)
    ) == ("mean",)
    assert _resolve_metric_reducers(
        window_mode="full_history", requested=None
    ) == ("official",)
    with pytest.raises(AllTEvaluationError, match="not available"):
        _resolve_metric_reducers(
            window_mode="full_history", requested=("mean",)
        )
    with pytest.raises(AllTEvaluationError, match="unique"):
        _resolve_metric_reducers(
            window_mode="local_pair", requested=("mean", "mean")
        )


def test_checkpoint_identity_uses_bound_training_metadata() -> None:
    assert _checkpoint_identity(
        {
            "global_step": 0,
            "allt_metadata": {"training_seed": 45, "variant": "C0"},
        },
        variant="C0",
    ) == (0, 45)
    assert _checkpoint_identity(
        {
            "global_step": 100,
            "hyper_parameters": {
                "general": {"seed": 46},
                "allt_training": {"variant": "C1"},
            },
        },
        variant="C1",
    ) == (100, 46)
    with pytest.raises(AllTEvaluationError, match="variant metadata differs"):
        _checkpoint_identity(
            {
                "global_step": 100,
                "hyper_parameters": {
                    "general": {"seed": 45},
                    "allt_training": {"variant": "C2"},
                },
            },
            variant="C1",
        )


def test_metric_accumulators_compute_concurrently_in_stable_order() -> None:
    barrier = threading.Barrier(2)

    class Accumulator:
        def __init__(self, value: float) -> None:
            self.value = value

        def compute(self) -> dict[str, float]:
            barrier.wait(timeout=5.0)
            return {"t_mAP": self.value}

    accumulators = {
        ("mean", 2): Accumulator(0.2),
        ("mean", 3): Accumulator(0.3),
    }
    values = _compute_metric_values(
        accumulators,
        keys=(("mean", 3), ("mean", 2)),
        workers=2,
    )
    assert list(values) == [("mean", 3), ("mean", 2)]
    assert values[("mean", 2)] == {"t_mAP": 0.2}
    with pytest.raises(AllTEvaluationError, match="workers"):
        _compute_metric_values(
            accumulators,
            keys=(("mean", 2),),
            workers=0,
        )


def test_metric_bundle_retains_only_requested_reducers() -> None:
    key = {"master_sequence_id": "master-0"}
    mean_pairs = {"2": {"prediction": "mean"}}
    compact = _compact_metric_bundle(
        {
            "key": key,
            "pairs": {
                "mean": mean_pairs,
                "latest": {"2": {"prediction": "latest"}},
                "max": {"2": {"prediction": "max"}},
            },
            "raw_payloads": ["large"],
            "sidecars": ["large"],
        },
        reducers=("mean",),
    )
    assert compact == {"key": key, "pairs": {"mean": mean_pairs}}
