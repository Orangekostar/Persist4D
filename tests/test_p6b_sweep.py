from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import lcm
from types import SimpleNamespace

import pytest
import torch

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.evaluate_persist4d_p6a import (
    CachedProtocolSequence,
    cache_payload_to_frozen_observation,
)
from scripts.p6a_analysis import AssociationEvent
from scripts.p6a_association import FrozenObservation
from scripts.p6b_association import P6BTracker
from scripts.p6b_protocol import load_p6b_config
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BClusterMetrics,
    P6BHorizonMetrics,
    P6BSequenceAssociationMetrics,
    P6BSweepError,
    P6BSweepSequence,
    _sequence_population_binding,
    assess_candidate,
    attach_cluster_task_metrics,
    build_candidate_row,
    cached_sequences_to_sweep_sequences,
    candidate_ranking_key,
    cluster_event_metrics,
    derive_prefix_events,
    extract_official_metrics,
    pareto_finalists,
    replay_configuration,
    run_staged_sweep,
    select_final_candidate,
    validate_staged_sweep_evidence,
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


def test_cached_mask_support_is_not_reinterpreted_as_zero_one_logits() -> None:
    payload = _cached_payload(stage=0)
    observation = cache_payload_to_frozen_observation(payload)
    tracker = P6BTracker(
        sequence_id="cached-support",
        config=replace(_memory_config(), birth_minimum_mask_support=2),
    )

    tracker.step(observation, stage_id=0)

    assert observation.mask_support is not None
    assert observation.mask_support.tolist() == [1]
    assert tracker.last_transition is not None
    assert tracker.last_transition.birth_mask_support == (1,)
    assert tracker.last_transition.rejected_birth_support == (True,)


def _horizon(
    horizon: int,
    *,
    switches: int = 10,
    transitions: int = 100,
    wrong: int = 4,
    false_births: int = 3,
    accuracy: float = 0.75,
    recall: float = 0.30,
    accepted: int = 90,
    total: int = 100,
    tmap: float | None = 0.20,
    trec: float | None = 0.30,
) -> P6BHorizonMetrics:
    accuracy_fraction = Fraction(str(accuracy))
    recall_fraction = Fraction(str(recall))
    correct = lcm(accuracy_fraction.numerator, recall_fraction.numerator)
    attempts = correct * accuracy_fraction.denominator // accuracy_fraction.numerator
    gaps = correct * recall_fraction.denominator // recall_fraction.numerator
    predicted = correct + wrong
    births = max(10, false_births)
    population = (("cluster-0", "cluster-0-master", "canonical"),)
    population_count, population_sha = _sequence_population_binding(population)
    sequence_metrics = (
        P6BSequenceAssociationMetrics(
            reference_scene_id="cluster-0",
            master_sequence_id="cluster-0-master",
            order_id="canonical",
            identity_switches=switches,
            transition_opportunities=transitions,
            wrong_reactivations=wrong,
            predicted_reactivation_events=predicted,
            correct_reactivations=correct,
            reactivation_attempts=attempts,
            gap_opportunities=gaps,
            false_births=false_births,
            births=births,
            rejected_births=2,
        ),
    )
    cluster = P6BClusterMetrics(
        reference_scene_id="cluster-0",
        identity_switches=switches,
        transition_opportunities=transitions,
        wrong_reactivations=wrong,
        predicted_reactivation_events=predicted,
        correct_reactivations=correct,
        reactivation_attempts=attempts,
        gap_opportunities=gaps,
        false_births=false_births,
        births=births,
        rejected_births=2,
        sequence_population_count=population_count,
        sequence_population_sha256=population_sha,
        sequence_metrics=sequence_metrics,
        strict_online_tmap=tmap,
        strict_online_trec=trec,
    )
    return P6BHorizonMetrics(
        horizon=horizon,
        identity_switches=switches,
        transition_opportunities=transitions,
        wrong_reactivations=wrong,
        predicted_reactivation_events=predicted,
        correct_reactivations=correct,
        reactivation_attempts=attempts,
        gap_opportunities=gaps,
        false_births=false_births,
        births=births,
        rejected_births=2,
        reactivation_accuracy=accuracy,
        reactivation_recall=recall,
        accepted_valid_observations=accepted,
        total_valid_observations=total,
        cluster_metrics=(cluster,),
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
        sequence_replay.prefix_steps[horizon] == sequence_replay.steps[:horizon]
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
            3: {"accuracy": 0.1, "recall": 0.05},
            4: {"accuracy": 0.1, "recall": 0.05, "switches": 0},
            5: {"accuracy": 0.1, "recall": 0.05, "switches": 0},
        },
    )

    selection = select_final_candidate(
        (ineligible, eligible),
        baseline=baseline,
        eligibility=protocol.eligibility,
    )

    assert selection.selected.config == eligible.config
    assert len(selection.assessed_rows) == 2


def test_selection_api_rejects_free_form_gate_deferrals() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate(
        overrides={
            horizon: {"accepted": 100, "total": 100}
            for horizon in (2, 3, 4, 5)
        }
    )
    candidate = _candidate(
        0.01,
        overrides={
            2: {"accepted": 80, "total": 100, "tmap": 0.10},
            3: {"accepted": 80, "total": 100},
            4: {"accepted": 80, "total": 100},
            5: {"accepted": 80, "total": 100},
        },
    )

    with pytest.raises(TypeError):
        select_final_candidate(
            (candidate,),
            baseline=baseline,
            eligibility=protocol.eligibility,
            deferred_eligibility_reasons=frozenset(
                {
                    "valid_observation_ratio_below_minimum",
                    "t2_task_drop_exceeded",
                }
            ),
        )


def test_stage_policy_never_defers_t2_or_final_stage_gates() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate(
        overrides={
            horizon: {"accepted": 100, "total": 100}
            for horizon in (2, 3, 4, 5)
        }
    )
    observation_drop = _candidate(
        0.01,
        overrides={
            horizon: {"accepted": 80, "total": 100}
            for horizon in (2, 3, 4, 5)
        },
    )
    task_drop = _candidate(
        0.02,
        overrides={2: {"accepted": 100, "total": 100, "tmap": 0.10}},
    )

    provisional = select_final_candidate(
        (replace(observation_drop, stage="assignment"),),
        baseline=baseline,
        eligibility=protocol.eligibility,
        stage="assignment",
    )
    assert provisional.selected.eligibility_reasons == (
        "valid_observation_ratio_below_minimum",
    )
    for stage in ("birth_gate", "joint_neighbors"):
        with pytest.raises(P6BSweepError, match="eligibility"):
            select_final_candidate(
                (replace(observation_drop, stage=stage),),
                baseline=baseline,
                eligibility=protocol.eligibility,
                stage=stage,
            )
    for stage in ("assignment", "birth_gate", "joint_neighbors"):
        with pytest.raises(P6BSweepError, match="eligibility"):
            select_final_candidate(
                (replace(task_drop, stage=stage),),
                baseline=baseline,
                eligibility=protocol.eligibility,
                stage=stage,
            )


def test_stage_policy_rejects_rows_claiming_a_different_stage() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate(
        overrides={
            horizon: {"accepted": 100, "total": 100}
            for horizon in (2, 3, 4, 5)
        }
    )
    mismatched = replace(
        _candidate(
            0.01,
            overrides={
                horizon: {"accepted": 80, "total": 100}
                for horizon in (2, 3, 4, 5)
            },
        ),
        stage="joint_neighbors",
    )

    with pytest.raises(P6BSweepError, match="stage"):
        select_final_candidate(
            (mismatched,),
            baseline=baseline,
            eligibility=protocol.eligibility,
            stage="assignment",
        )
    with pytest.raises(P6BSweepError, match="stage"):
        pareto_finalists(
            (
                replace(
                    mismatched,
                    horizons=tuple(
                        replace(
                            metric,
                            strict_online_tmap=None,
                            strict_online_trec=None,
                        )
                        for metric in mismatched.horizons
                    ),
                ),
            ),
            baseline=baseline,
            eligibility=protocol.eligibility,
            stage="assignment",
        )


def test_missing_official_task_metrics_cannot_be_frozen() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")

    with pytest.raises(P6BSweepError, match="official"):
        select_final_candidate(
            (_candidate(0.01, official=False),),
            baseline=_candidate(),
            eligibility=protocol.eligibility,
        )


def test_official_metric_completeness_includes_cluster_task_fields() -> None:
    complete = _candidate()
    absent = _candidate(official=False)
    partial_metric = absent.metric(4)
    partial = replace(
        absent,
        horizons=tuple(
            replace(
                partial_metric,
                cluster_metrics=tuple(
                    replace(
                        cluster,
                        strict_online_tmap=0.2,
                        strict_online_trec=0.3,
                    )
                    for cluster in partial_metric.cluster_metrics
                ),
            )
            if metric.horizon == 4
            else metric
            for metric in absent.horizons
        ),
    )

    assert complete.official_metrics_complete is True
    assert complete.official_metrics_absent is False
    assert absent.official_metrics_complete is False
    assert absent.official_metrics_absent is True
    assert partial.official_metrics_complete is False
    assert partial.official_metrics_absent is False


def test_ranking_key_uses_all_preregistered_criteria_in_order() -> None:
    baseline = _candidate()
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

    key = candidate_ranking_key(row, baseline=baseline)

    assert key[:5] == pytest.approx(
        (
            sum(
                row.metric(horizon).cluster_mean_identity_switch_rate
                - baseline.metric(horizon).cluster_mean_identity_switch_rate
                for horizon in (4, 5)
            )
            / 2,
            sum(
                row.metric(horizon).wrong_reactivation_rate
                - baseline.metric(horizon).wrong_reactivation_rate
                for horizon in (3, 4, 5)
            )
            / 3,
            sum(
                row.metric(horizon).false_birth_rate
                - baseline.metric(horizon).false_birth_rate
                for horizon in (2, 3, 4, 5)
            )
            / 4,
            -sum(
                row.metric(horizon).reactivation_recall
                - baseline.metric(horizon).reactivation_recall
                for horizon in (3, 4, 5)
            )
            / 3,
            -sum(
                candidate_value - baseline_value
                for horizon in (4, 5)
                for candidate_value, baseline_value in (
                    (
                        row.metric(horizon).strict_online_tmap,
                        baseline.metric(horizon).strict_online_tmap,
                    ),
                    (
                        row.metric(horizon).strict_online_trec,
                        baseline.metric(horizon).strict_online_trec,
                    ),
                )
            )
            / 4,
        )
    )
    assert key[5] == row.config_json


def test_ranking_key_records_paired_deltas_against_frozen_b4() -> None:
    baseline = _candidate()
    candidate = _candidate(
        0.01,
        overrides={
            4: {"switches": 8, "transitions": 100},
            5: {"switches": 9, "transitions": 100},
        },
    )

    key = candidate_ranking_key(candidate, baseline=baseline)

    expected = (
        sum(
            candidate.metric(horizon).cluster_mean_identity_switch_rate
            - baseline.metric(horizon).cluster_mean_identity_switch_rate
            for horizon in (4, 5)
        )
        / 2
    )
    assert key[0] == pytest.approx(expected)


def _cluster(
    reference: str,
    *,
    correct: int,
    gaps: int,
    tmap: float,
    trec: float,
    identities: tuple[tuple[str, str, str], ...] | None = None,
) -> P6BClusterMetrics:
    population = identities or ((reference, f"{reference}-0", "canonical"),)
    population_count, population_sha = _sequence_population_binding(population)
    sequence_metrics = tuple(
        P6BSequenceAssociationMetrics(
            reference_scene_id=identity[0],
            master_sequence_id=identity[1],
            order_id=identity[2],
            identity_switches=1 if index == 0 else 0,
            transition_opportunities=10 if index == 0 else 0,
            wrong_reactivations=0,
            predicted_reactivation_events=correct if index == 0 else 0,
            correct_reactivations=correct if index == 0 else 0,
            reactivation_attempts=correct if index == 0 else 0,
            gap_opportunities=gaps if index == 0 else 0,
            false_births=1 if index == 0 else 0,
            births=10 if index == 0 else 0,
            rejected_births=0,
        )
        for index, identity in enumerate(sorted(population))
    )
    return P6BClusterMetrics(
        reference_scene_id=reference,
        identity_switches=1,
        transition_opportunities=10,
        wrong_reactivations=0,
        predicted_reactivation_events=correct,
        correct_reactivations=correct,
        reactivation_attempts=correct,
        gap_opportunities=gaps,
        false_births=1,
        births=10,
        rejected_births=0,
        sequence_population_count=population_count,
        sequence_population_sha256=population_sha,
        sequence_metrics=sequence_metrics,
        strict_online_tmap=tmap,
        strict_online_trec=trec,
    )


def _with_clusters(
    row: P6BCandidateRow,
    horizon: int,
    clusters: tuple[P6BClusterMetrics, ...],
    *,
    tmap: float,
    trec: float,
) -> P6BCandidateRow:
    metric = row.metric(horizon)
    correct = sum(item.correct_reactivations for item in clusters)
    gaps = sum(item.gap_opportunities for item in clusters)
    replacement = replace(
        metric,
        identity_switches=sum(item.identity_switches for item in clusters),
        transition_opportunities=sum(
            item.transition_opportunities for item in clusters
        ),
        wrong_reactivations=0,
        predicted_reactivation_events=correct,
        correct_reactivations=correct,
        reactivation_attempts=correct,
        gap_opportunities=gaps,
        false_births=sum(item.false_births for item in clusters),
        births=sum(item.births for item in clusters),
        rejected_births=0,
        reactivation_accuracy=1.0,
        reactivation_recall=correct / gaps,
        cluster_metrics=clusters,
        strict_online_tmap=tmap,
        strict_online_trec=trec,
    )
    return replace(
        row,
        horizons=tuple(
            replacement if item.horizon == horizon else item for item in row.horizons
        ),
    )


def test_ranking_uses_paired_reference_cluster_recall_and_task_score() -> None:
    baseline = _candidate()
    candidate = _candidate(0.01)
    for horizon in (3, 4, 5):
        baseline = _with_clusters(
            baseline,
            horizon,
            (
                _cluster("a", correct=1, gaps=10, tmap=0.2, trec=0.2),
                _cluster("b", correct=9, gaps=10, tmap=0.8, trec=0.8),
            ),
            tmap=0.90,
            trec=0.90,
        )
        candidate = _with_clusters(
            candidate,
            horizon,
            (
                _cluster("a", correct=4, gaps=5, tmap=0.4, trec=0.4),
                _cluster("b", correct=6, gaps=15, tmap=0.8, trec=0.8),
            ),
            tmap=0.10,
            trec=0.10,
        )

    key = candidate_ranking_key(candidate, baseline=baseline)

    assert key[3] == pytest.approx(-0.1)
    assert key[4] == pytest.approx(-0.1)


def test_pareto_shortlist_uses_reference_cluster_recall() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()
    candidate = _candidate(0.01)
    for horizon in (3, 4, 5):
        baseline = _with_clusters(
            baseline,
            horizon,
            (
                _cluster("a", correct=1, gaps=10, tmap=0.2, trec=0.2),
                _cluster("b", correct=9, gaps=10, tmap=0.8, trec=0.8),
            ),
            tmap=0.5,
            trec=0.5,
        )
        candidate = _with_clusters(
            candidate,
            horizon,
            (
                _cluster("a", correct=4, gaps=5, tmap=0.2, trec=0.2),
                _cluster("b", correct=6, gaps=15, tmap=0.8, trec=0.8),
            ),
            tmap=0.5,
            trec=0.5,
        )

    finalists = pareto_finalists(
        (baseline, candidate), baseline=baseline, eligibility=protocol.eligibility
    )

    assert [row.config_id for row in finalists] == [candidate.config_id]


def test_cluster_task_metrics_are_averaged_within_reference_scene() -> None:
    clusters = (
        _cluster(
            "a",
            correct=1,
            gaps=2,
            tmap=0.0,
            trec=0.0,
            identities=(("a", "a-0", "canonical"), ("a", "a-1", "reverse")),
        ),
        _cluster("b", correct=1, gaps=2, tmap=0.0, trec=0.0),
    )
    rows = (
        {
            "method": "P6B",
            "reference_scene_id": "a",
            "master_sequence_id": "a-0",
            "order_id": "canonical",
            "T": "T4",
            "t_mAP": 0.2,
            "t_REC": 0.4,
            "prediction_digest": "1" * 64,
        },
        {
            "method": "P6B",
            "reference_scene_id": "a",
            "master_sequence_id": "a-1",
            "order_id": "reverse",
            "T": "T4",
            "t_mAP": 0.4,
            "t_REC": 0.6,
            "prediction_digest": "2" * 64,
        },
        {
            "method": "P6B",
            "reference_scene_id": "b",
            "master_sequence_id": "b-0",
            "order_id": "canonical",
            "T": "T4",
            "t_mAP": 0.8,
            "t_REC": 1.0,
            "prediction_digest": "3" * 64,
        },
    )

    enriched = attach_cluster_task_metrics(
        clusters,
        rows,
        method="P6B",
        horizon=4,
        expected_sequence_count=3,
    )

    assert enriched[0].strict_online_tmap == pytest.approx(0.3)
    assert enriched[0].strict_online_trec == pytest.approx(0.5)
    assert enriched[1].strict_online_tmap == pytest.approx(0.8)
    assert enriched[1].strict_online_trec == pytest.approx(1.0)
    with pytest.raises(P6BSweepError, match="population"):
        attach_cluster_task_metrics(
            clusters,
            rows[:-1],
            method="P6B",
            horizon=4,
            expected_sequence_count=3,
        )


def test_ranking_prefers_lower_normalized_rate_over_lower_raw_count() -> None:
    baseline = _candidate()
    lower_rate = _candidate(
        0.01,
        overrides={
            4: {"switches": 10, "transitions": 100},
            5: {"switches": 10, "transitions": 100},
        },
    )
    lower_count_but_worse_rate = _candidate(
        0.02,
        overrides={
            4: {"switches": 8, "transitions": 40},
            5: {"switches": 8, "transitions": 40},
        },
    )

    assert candidate_ranking_key(lower_rate, baseline=baseline) < candidate_ranking_key(
        lower_count_but_worse_rate, baseline=baseline
    )


def test_horizon_rates_are_recomputed_from_explicit_denominators() -> None:
    metric = _horizon(
        5,
        switches=3,
        transitions=12,
        wrong=2,
        false_births=4,
    )

    assert metric.identity_switch_rate == pytest.approx(3 / 12)
    assert metric.wrong_reactivation_rate == pytest.approx(
        2 / metric.predicted_reactivation_events
    )
    assert metric.false_birth_rate == pytest.approx(4 / (metric.births + 2))
    assert metric.cluster_mean_identity_switch_rate == pytest.approx(3 / 12)
    assert metric.true_births == metric.births - metric.false_births
    assert metric.accepted_births == metric.births
    assert metric.valid_birth_opportunities == metric.births + metric.rejected_births
    assert metric.frozen_b4_valid_observations == metric.total_valid_observations


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
        overrides={
            3: {"accuracy": 0.1, "recall": 0.05},
            4: {"accuracy": 0.1, "recall": 0.05},
            5: {"accuracy": 0.1, "recall": 0.05},
        },
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
    assert row.metric(5).frozen_b4_valid_observations == 5
    assert row.metric(5).accepted_births == row.metric(5).births
    assert row.metric(5).true_births == (
        row.metric(5).births - row.metric(5).false_births
    )
    assert row.metric(5).valid_birth_opportunities == (
        row.metric(5).births + row.metric(5).rejected_births
    )
    assert row.metric(4).strict_online_tmap == pytest.approx(0.4)
    assert row.metric(4).strict_online_trec == pytest.approx(0.2)


def test_cluster_metrics_include_explicit_zero_event_population_rows() -> None:
    event = AssociationEvent(
        event_id="P6B:tune:canonical:T2:1:0",
        scene_id="master-a",
        sequence_id="master-a:canonical",
        reference_scene_id="tune",
        master_sequence_id="master-a",
        order_id="canonical",
        prefix=2,
        method="P6B",
        stage_id=1,
        query_id=0,
        candidate_slot_id=0,
        predicted_identity_id=0,
        gt_entity_id=0,
        association_correct=True,
        association_result="active_correct",
        gt_present=True,
        prediction_present=True,
        transition_opportunity=True,
        id_switch=False,
        gap_opportunity=False,
        reactivation_attempt=False,
        reactivation_correct=None,
        new_birth=False,
        false_birth=False,
        reactivation=False,
        wrong_reactivation=False,
        is_failure=False,
        prediction_digest="a" * 64,
        cache_digest="a" * 64,
    )

    (cluster,) = cluster_event_metrics(
        (event,),
        population_identities=(
            ("tune", "master-a", "canonical"),
            ("tune", "master-b", "reverse"),
        ),
    )

    assert [
        (row.master_sequence_id, row.order_id) for row in cluster.sequence_metrics
    ] == [("master-a", "canonical"), ("master-b", "reverse")]
    zero = cluster.sequence_metrics[1]
    assert (
        sum(
            getattr(zero, field)
            for field in (
                "identity_switches",
                "transition_opportunities",
                "wrong_reactivations",
                "predicted_reactivation_events",
                "correct_reactivations",
                "reactivation_attempts",
                "gap_opportunities",
                "false_births",
                "births",
                "rejected_births",
            )
        )
        == 0
    )


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
    evaluation.metric_blocks["strict"]["Other"] = evaluation.metric_blocks[
        "strict"
    ].pop("P6B")
    with pytest.raises(P6BSweepError, match="P6B"):
        extract_official_metrics(evaluation)


def test_staged_sweep_defers_future_component_gates_until_their_stage() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate(
        overrides={
            horizon: {"accepted": 100, "total": 100}
            for horizon in (2, 3, 4, 5)
        }
    )

    def fast_evaluator(config: P6BMemoryConfig, stage: str) -> P6BCandidateRow:
        accepted = 100 if stage in {"birth_gate", "joint_neighbors"} else 89
        overrides = {
            horizon: {"accepted": accepted, "total": 100}
            for horizon in (2, 3, 4, 5)
        }
        if stage == "assignment":
            for horizon in (3, 4, 5):
                overrides[horizon].update(accuracy=0.60, recall=0.20)
        return replace(
            _candidate(overrides=overrides, official=False),
            config=config,
            stage=stage,
        )

    def official_evaluator(row: P6BCandidateRow) -> P6BCandidateRow:
        return replace(
            row,
            horizons=tuple(
                replace(
                    metric,
                    cluster_metrics=tuple(
                        replace(
                            cluster,
                            strict_online_tmap=0.20,
                            strict_online_trec=0.30,
                        )
                        for cluster in metric.cluster_metrics
                    ),
                    strict_online_tmap=0.20,
                    strict_online_trec=0.30,
                )
                for metric in row.horizons
            ),
        )

    result = run_staged_sweep(
        protocol,
        baseline=baseline,
        fast_evaluator=fast_evaluator,
        official_evaluator=official_evaluator,
    )

    assert set(result.selected_by_stage["assignment"].eligibility_reasons) == {
        "reactivation_accuracy_below_minimum",
        "reactivation_recall_drop_exceeded",
        "valid_observation_ratio_below_minimum",
    }
    assert result.selected_by_stage["reactivation"].eligibility_reasons == (
        "valid_observation_ratio_below_minimum",
    )
    assert result.selected_by_stage["birth_gate"].eligible
    assert result.selected.eligible


def test_staged_sweep_preserves_candidates_and_selects_official_incumbents() -> None:
    protocol = load_p6b_config("conf/p6b/default.yaml")
    baseline = _candidate()

    def fast_evaluator(config: P6BMemoryConfig, stage: str) -> P6BCandidateRow:
        return replace(_candidate(official=False), config=config, stage=stage)

    def official_evaluator(row: P6BCandidateRow) -> P6BCandidateRow:
        return replace(
            row,
            horizons=tuple(
                replace(
                    item,
                    cluster_metrics=tuple(
                        replace(
                            cluster,
                            strict_online_tmap=0.20,
                            strict_online_trec=0.30,
                        )
                        for cluster in item.cluster_metrics
                    ),
                    strict_online_tmap=0.20,
                    strict_online_trec=0.30,
                )
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
    assert all(
        isinstance(row.eligibility_reasons, tuple) for row in result.candidate_rows
    )
    assert result.selected.official_metrics_complete
    validate_staged_sweep_evidence(
        protocol,
        baseline=baseline,
        candidate_rows=result.candidate_rows,
        finalist_rows=result.finalist_rows,
        selected_by_stage=result.selected_by_stage,
        selected=result.selected,
        ranking_key=candidate_ranking_key(result.selected, baseline=baseline),
    )

    nonwinner = next(
        row
        for row in result.finalist_rows
        if row != result.selected_by_stage[row.stage]
    )
    drifted_horizons = tuple(
        replace(
            metric,
            identity_switches=metric.identity_switches + 1,
            cluster_metrics=(
                replace(
                    metric.cluster_metrics[0],
                    identity_switches=metric.cluster_metrics[0].identity_switches + 1,
                    sequence_metrics=(
                        replace(
                            metric.cluster_metrics[0].sequence_metrics[0],
                            identity_switches=metric.cluster_metrics[0]
                            .sequence_metrics[0]
                            .identity_switches
                            + 1,
                        ),
                        *metric.cluster_metrics[0].sequence_metrics[1:],
                    ),
                ),
                *metric.cluster_metrics[1:],
            ),
        )
        if metric.horizon == 4
        else metric
        for metric in nonwinner.horizons
    )
    drifted = replace(nonwinner, horizons=drifted_horizons)
    finalists = tuple(
        drifted if row is nonwinner else row for row in result.finalist_rows
    )
    with pytest.raises(P6BSweepError, match="association evidence"):
        validate_staged_sweep_evidence(
            protocol,
            baseline=baseline,
            candidate_rows=result.candidate_rows,
            finalist_rows=finalists,
            selected_by_stage=result.selected_by_stage,
            selected=result.selected,
            ranking_key=candidate_ranking_key(result.selected, baseline=baseline),
        )

    tampered = dict(result.selected_by_stage)
    tampered["assignment"] = replace(
        tampered["assignment"],
        config=replace(tampered["assignment"].config, active_threshold=0.123),
    )
    with pytest.raises(P6BSweepError, match="assignment.*winner|winner.*assignment"):
        validate_staged_sweep_evidence(
            protocol,
            baseline=baseline,
            candidate_rows=result.candidate_rows,
            finalist_rows=result.finalist_rows,
            selected_by_stage=tampered,
            selected=result.selected,
            ranking_key=candidate_ranking_key(result.selected, baseline=baseline),
        )
