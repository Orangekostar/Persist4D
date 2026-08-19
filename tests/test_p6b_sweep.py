from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.evaluate_persist4d_p6a import CachedProtocolSequence
from scripts.p6a_analysis import AssociationEvent
from scripts.p6a_association import FrozenObservation
from scripts.p6b_association import P6BTracker
from scripts.p6b_protocol import load_p6b_config
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BHorizonMetrics,
    P6BSweepError,
    P6BSweepSequence,
    assess_candidate,
    build_candidate_row,
    cached_sequences_to_sweep_sequences,
    candidate_ranking_key,
    derive_prefix_events,
    extract_official_metrics,
    pareto_finalists,
    replay_configuration,
    run_staged_sweep,
    select_final_candidate,
)


def _observation(feature: tuple[float, float]) -> FrozenObservation:
    return FrozenObservation(
        features=torch.tensor([feature], dtype=torch.float64),
        class_prob=torch.tensor([[0.98, 0.01, 0.01]], dtype=torch.float64),
        confidence=torch.tensor([0.99], dtype=torch.float64),
        valid=torch.tensor([True]),
        latest_mask=(torch.full((1, 4), 10.0, dtype=torch.float64),),
    )


def _sequence(reference: str = "tune") -> P6BSweepSequence:
    return P6BSweepSequence(
        reference_scene_id=reference,
        master_sequence_id=f"master-{reference}",
        order_id="canonical",
        observations=tuple(_observation((1.0, 0.0)) for _ in range(5)),
    )


def _cached_payload(stage: int) -> dict[str, object]:
    point_count = stage + 3
    masks = torch.zeros((1, point_count), dtype=torch.bool)
    masks[0, 0] = True
    return {
        "schema_version": 3,
        "key": {
            "master_sequence_id": "master-cache",
            "reference_scene_id": "tune-cache",
            "order_id": "canonical",
            "stage_index": stage,
            "history_scan_ids": [
                f"scene0001_{index:02d}" for index in range(stage + 1)
            ],
            "local_window_scan_ids": [
                f"scene0001_{index:02d}"
                for index in range(max(0, stage - 1), stage + 1)
            ],
        },
        "provenance": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
        },
        "observation": {
            "features": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "class_prob": torch.tensor([[0.01, 0.98, 0.01]]),
            "confidence": torch.tensor([0.99]),
            "valid": torch.tensor([True]),
            "masks": masks,
            "mask_support": torch.tensor([1]),
            "local_query_ids": torch.tensor([0]),
        },
        "target": {
            "gt_ids": torch.tensor([10]),
            "gt_classes": torch.tensor([1]),
            "gt_masks": masks.clone(),
            "changes": torch.tensor([0]),
            "change_labels_valid": False,
            "change_label_semantics": (
                "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
            ),
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }


def _cached_sequence() -> CachedProtocolSequence:
    return CachedProtocolSequence(
        reference_scene_id="tune-cache",
        master_sequence_id="master-cache",
        order_id="canonical",
        payloads=tuple(_cached_payload(stage) for stage in range(5)),
    )


def _memory_config() -> P6BMemoryConfig:
    return P6BMemoryConfig(
        capacity=2,
        background_class=2,
        birth_minimum_mask_support=1,
        birth_max_entropy=None,
        consolidation_margin=None,
    )


def _horizon(
    horizon: int,
    *,
    switches: int = 10,
    wrong: int = 4,
    false_births: int = 3,
    accuracy: float = 0.75,
    recall: float = 0.30,
    accepted: int = 90,
    total: int = 100,
    tmap: float | None = 0.20,
    trec: float | None = 0.30,
) -> P6BHorizonMetrics:
    return P6BHorizonMetrics(
        horizon=horizon,
        identity_switches=switches,
        wrong_reactivations=wrong,
        false_births=false_births,
        reactivation_accuracy=accuracy,
        reactivation_recall=recall,
        accepted_valid_observations=accepted,
        total_valid_observations=total,
        strict_online_tmap=tmap,
        strict_online_trec=trec,
    )


def _candidate(
    name_offset: float = 0.0,
    *,
    overrides: dict[int, dict[str, object]] | None = None,
    official: bool = True,
) -> P6BCandidateRow:
    config = replace(
        _memory_config(),
        active_threshold=_memory_config().active_threshold + name_offset,
    )
    values = []
    for horizon in (2, 3, 4, 5):
        kwargs = dict((overrides or {}).get(horizon, {}))
        if not official:
            kwargs.update(tmap=None, trec=None)
        values.append(_horizon(horizon, **kwargs))
    return P6BCandidateRow(
        config=config,
        stage="fixture",
        horizons=tuple(values),
    )


def test_one_t5_replay_exposes_identical_causal_prefix_steps() -> None:
    sequence = _sequence()
    replay = replay_configuration(
        (sequence,),
        _memory_config(),
        allowed_reference_scene_ids=("tune",),
    )

    assert len(replay.sequences) == 1
    sequence_replay = replay.sequences[0]
    assert set(sequence_replay.prefix_steps) == {2, 3, 4, 5}
    assert all(
        sequence_replay.prefix_steps[horizon]
        == sequence_replay.steps[:horizon]
        for horizon in (2, 3, 4, 5)
    )
    assert [step.track_ids for step in sequence_replay.steps] == [(0,)] * 5

    for horizon in (2, 3, 4, 5):
        tracker = P6BTracker(sequence_id="manual", config=_memory_config())
        manual = tuple(
            tracker.step(observation, stage_id=stage)
            for stage, observation in enumerate(sequence.observations[:horizon])
        )
        assert [step.track_ids for step in manual] == [
            step.track_ids for step in sequence_replay.prefix_steps[horizon]
        ]


def test_replay_rejects_heldout_or_mixed_cluster_input() -> None:
    with pytest.raises(P6BSweepError, match="heldout|allowed"):
        replay_configuration(
            (_sequence("tune"), _sequence("heldout")),
            _memory_config(),
            allowed_reference_scene_ids=("tune",),
        )


def test_candidate_eligibility_enforces_reactivation_observation_and_t2_gates() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()
    candidate = _candidate(
        0.01,
        overrides={
            2: {"accepted": 80, "tmap": 0.17},
            3: {"accuracy": 0.60, "recall": 0.20},
            4: {"accuracy": 0.60, "recall": 0.20},
            5: {"accuracy": 0.60, "recall": 0.20},
        },
    )

    assessed = assess_candidate(
        candidate,
        baseline=baseline,
        eligibility=protocol.eligibility,
        require_official_metrics=True,
    )

    assert not assessed.eligible
    assert set(assessed.eligibility_reasons) == {
        "reactivation_accuracy_below_minimum",
        "reactivation_recall_drop_exceeded",
        "valid_observation_ratio_below_minimum",
        "t2_task_drop_exceeded",
    }


def test_ineligible_candidate_cannot_win_even_with_better_switch_count() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()
    eligible = _candidate(0.01, overrides={4: {"switches": 8}, 5: {"switches": 8}})
    ineligible = _candidate(
        0.02,
        overrides={
            3: {"accuracy": 0.1},
            4: {"accuracy": 0.1, "switches": 0},
            5: {"accuracy": 0.1, "switches": 0},
        },
    )

    selection = select_final_candidate(
        (ineligible, eligible),
        baseline=baseline,
        eligibility=protocol.eligibility,
    )

    assert selection.selected.config == eligible.config
    assert len(selection.assessed_rows) == 2


def test_missing_official_task_metrics_cannot_be_frozen() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")

    with pytest.raises(P6BSweepError, match="official"):
        select_final_candidate(
            (_candidate(0.01, official=False),),
            baseline=_candidate(),
            eligibility=protocol.eligibility,
        )


def test_ranking_key_uses_all_preregistered_criteria_in_order() -> None:
    row = _candidate(
        overrides={
            2: {"false_births": 8},
            3: {"wrong": 7, "recall": 0.20},
            4: {
                "switches": 12,
                "wrong": 6,
                "false_births": 5,
                "recall": 0.30,
                "tmap": 0.40,
                "trec": 0.50,
            },
            5: {
                "switches": 14,
                "wrong": 5,
                "false_births": 4,
                "recall": 0.40,
                "tmap": 0.30,
                "trec": 0.40,
            },
        }
    )

    key = candidate_ranking_key(row)

    assert key[:5] == pytest.approx((13.0, 18.0, 20.0, -0.30, -0.40))
    assert key[5] == row.config_json


def test_pareto_finalists_exclude_dominated_and_ineligible_rows() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()
    strong = _candidate(
        0.01,
        overrides={4: {"switches": 8}, 5: {"switches": 8}},
        official=False,
    )
    dominated = _candidate(
        0.02,
        overrides={4: {"switches": 12}, 5: {"switches": 12}},
        official=False,
    )
    ineligible = _candidate(
        0.03,
        overrides={3: {"accuracy": 0.1}, 4: {"accuracy": 0.1}, 5: {"accuracy": 0.1}},
        official=False,
    )

    finalists = pareto_finalists(
        (dominated, ineligible, strong),
        baseline=replace(
            baseline,
            horizons=tuple(
                replace(item, strict_online_tmap=None, strict_online_trec=None)
                for item in baseline.horizons
            ),
        ),
        eligibility=protocol.eligibility,
    )

    assert [row.config for row in finalists] == [strong.config]


def test_candidate_row_aggregates_event_and_acceptance_metrics() -> None:
    replay = replay_configuration(
        (_sequence(),),
        _memory_config(),
        allowed_reference_scene_ids=("tune",),
    )
    events = {
        horizon: (
            AssociationEvent(
                event_id=f"P6B:tune:canonical:T{horizon}:1:0",
                scene_id="master-tune",
                sequence_id="master-tune:canonical",
                reference_scene_id="tune",
                master_sequence_id="master-tune",
                order_id="canonical",
                prefix=horizon,
                method="P6B",
                stage_id=1,
                query_id=0,
                candidate_slot_id=0,
                predicted_identity_id=0,
                gt_entity_id=0,
                association_correct=True,
                association_result="reactivation_correct",
                gt_present=True,
                prediction_present=True,
                transition_opportunity=True,
                id_switch=horizon == 5,
                gap_opportunity=True,
                reactivation_attempt=True,
                reactivation_correct=True,
                new_birth=False,
                false_birth=False,
                reactivation=True,
                wrong_reactivation=False,
                is_failure=False,
                prediction_digest="a" * 64,
                cache_digest="a" * 64,
            ),
        )
        for horizon in (2, 3, 4, 5)
    }

    row = build_candidate_row(
        replay,
        stage="fixture",
        events_by_horizon=events,
        official_metrics={
            horizon: {"t_mAP": 0.1 * horizon, "t_REC": 0.05 * horizon}
            for horizon in (2, 3, 4, 5)
        },
    )

    assert row.metric(5).identity_switches == 1
    assert row.metric(5).wrong_reactivations == 0
    assert row.metric(5).reactivation_accuracy == 1.0
    assert row.metric(5).reactivation_recall == 1.0
    assert row.metric(5).accepted_valid_observations == 5
    assert row.metric(5).total_valid_observations == 5
    assert row.metric(4).strict_online_tmap == pytest.approx(0.4)
    assert row.metric(4).strict_online_trec == pytest.approx(0.2)


def test_cached_sequences_convert_and_derive_exact_prefix_events() -> None:
    cached = _cached_sequence()
    sequences = cached_sequences_to_sweep_sequences((cached,))
    replay = replay_configuration(
        sequences,
        _memory_config(),
        allowed_reference_scene_ids=("tune-cache",),
    )

    events = derive_prefix_events(replay, (cached,), background_class=2)

    assert tuple(events) == (2, 3, 4, 5)
    assert len(events[2]) == 2
    assert len(events[5]) == 5
    assert {event.prefix for event in events[4]} == {4}
    assert {event.method for event in events[5]} == {"P6B"}
    assert len({event.cache_digest for event in events[5]}) == 1


def test_cached_conversion_rejects_duplicate_or_mismatched_replay_sources() -> None:
    cached = _cached_sequence()
    with pytest.raises(P6BSweepError, match="duplicate"):
        cached_sequences_to_sweep_sequences((cached, cached))

    replay = replay_configuration(
        cached_sequences_to_sweep_sequences((cached,)),
        _memory_config(),
        allowed_reference_scene_ids=("tune-cache",),
    )
    with pytest.raises(P6BSweepError, match="source|identity"):
        derive_prefix_events(replay, (), background_class=2)


def test_official_metrics_are_extracted_only_from_exact_p6b_strict_blocks() -> None:
    evaluation = SimpleNamespace(
        metric_blocks={
            "strict": {
                "P6B": {
                    f"T{horizon}": {
                        "online_t-mAP": horizon / 20,
                        "online_t-mAP50": 0.1,
                        "online_t-mAP25": 0.2,
                        "online_t-REC": horizon / 10,
                        "online_t-REC50": 0.3,
                        "online_t-REC25": 0.4,
                    }
                    for horizon in (2, 3, 4, 5)
                }
            }
        }
    )

    metrics = extract_official_metrics(evaluation)

    assert metrics[2] == {"t_mAP": pytest.approx(0.1), "t_REC": pytest.approx(0.2)}
    assert metrics[5] == {"t_mAP": pytest.approx(0.25), "t_REC": pytest.approx(0.5)}
    evaluation.metric_blocks["strict"]["Other"] = evaluation.metric_blocks["strict"].pop(
        "P6B"
    )
    with pytest.raises(P6BSweepError, match="P6B"):
        extract_official_metrics(evaluation)


def test_staged_sweep_preserves_candidates_and_selects_official_incumbents() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()

    def fast_evaluator(config: P6BMemoryConfig, stage: str) -> P6BCandidateRow:
        return replace(_candidate(official=False), config=config, stage=stage)

    def official_evaluator(row: P6BCandidateRow) -> P6BCandidateRow:
        return replace(
            row,
            horizons=tuple(
                replace(item, strict_online_tmap=0.20, strict_online_trec=0.30)
                for item in row.horizons
            ),
        )

    result = run_staged_sweep(
        protocol,
        baseline=baseline,
        fast_evaluator=fast_evaluator,
        official_evaluator=official_evaluator,
    )

    assert tuple(result.selected_by_stage) == (
        "assignment",
        "reactivation",
        "class_compatibility",
        "consolidation",
        "birth_gate",
        "joint_neighbors",
    )
    counts = {
        stage: sum(row.stage == stage for row in result.candidate_rows)
        for stage in result.selected_by_stage
    }
    assert counts["assignment"] == 2
    assert counts["reactivation"] == 48
    assert counts["class_compatibility"] == 6
    assert counts["consolidation"] == 10
    assert counts["birth_gate"] == 64
    assert counts["joint_neighbors"] > 1
    assert result.finalist_rows
    assert all(row.official_metrics_complete for row in result.finalist_rows)
    assert all(isinstance(row.eligibility_reasons, tuple) for row in result.candidate_rows)
    assert result.selected.official_metrics_complete
