from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest
import torch

from models.persistent_memory import LocalInstanceObservation, PersistentMemory
from scripts.p6a_association import (
    AssociationDiagnostics,
    B0SanityTracker,
    B0StageUniqueTracker,
    B1FeatureTracker,
    B2FeatureClassTracker,
    B3EmaTracker,
    B4PersistentTracker,
    IdentityNamespace,
    OracleStageTarget,
    TrackStep,
    fan_out_observation,
    foreground_normalized_class_prob,
    freeze_observation,
    run_b0,
    run_b0_sanity,
    run_b1,
    run_b2,
    run_b3,
    run_baseline,
    run_oracle_posthoc,
    threshold_aware_hungarian,
)


def _observation(
    features: list[list[float]],
    class_prob: list[list[float]] | None = None,
    *,
    valid: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    feature_tensor = torch.tensor(features, dtype=torch.float64)
    query_count = feature_tensor.shape[0]
    if class_prob is None:
        class_prob = [[1.0, 0.0] for _ in range(query_count)]
    class_tensor = torch.tensor(class_prob, dtype=torch.float64)
    if valid is None:
        valid = [True] * query_count
    return {
        "features": feature_tensor,
        "class_prob": class_tensor,
        "confidence": torch.ones(query_count, dtype=torch.float64),
        "valid": torch.tensor(valid, dtype=torch.bool),
    }


def _mask_observation(
    features: list[list[float]],
    class_prob: list[list[float]],
    mask_logits: list[list[float]],
    *,
    valid: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    observation = _observation(features, class_prob, valid=valid)
    observation["latest_mask"] = torch.tensor(mask_logits, dtype=torch.float64)
    return observation


def _ids(step: TrackStep) -> tuple[object, ...]:
    return step.track_ids


def test_observation_fanout_deep_copies_inputs_and_each_method() -> None:
    source = _observation([[1.0, 0.0]])
    frozen = freeze_observation(source)
    fanout = fan_out_observation(source, ["b0", "b1"])

    assert frozen.features is not source["features"]
    assert fanout["b0"].features is not fanout["b1"].features
    source["features"][0, 0] = 99.0
    fanout["b0"].features[0, 1] = 77.0
    assert frozen.features.tolist() == [[1.0, 0.0]]
    assert fanout["b1"].features.tolist() == [[1.0, 0.0]]
    with pytest.raises(FrozenInstanceError):
        frozen.features = source["features"]


def test_namespace_resets_per_method_and_sequence_without_capacity_limit() -> None:
    namespace = IdentityNamespace()
    assert namespace.next_id("b1", "sequence-a") == 0
    assert namespace.next_id("b1", "sequence-a") == 1
    assert namespace.next_id("b2", "sequence-a") == 0
    assert namespace.next_id("b1", "sequence-b") == 0
    namespace.reset("b1", "sequence-a")
    assert namespace.next_id("b1", "sequence-a") == 0


def test_b0_stage_unique_ids_are_not_local_query_ids() -> None:
    tracker = B0StageUniqueTracker(sequence_id="scene")
    first = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=0)
    second = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=1)

    assert _ids(first) == ((0, 0), (0, 1))
    assert _ids(second) == ((1, 0), (1, 1))
    assert first.method == "B0"
    assert first.scores == (None, None)
    assert first.rejected_births == (False, False)


def test_b0_sanity_explicitly_reuses_local_query_index() -> None:
    tracker = B0SanityTracker(sequence_id="scene")
    first = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=0)
    second = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=1)

    assert _ids(first) == (0, 1)
    assert _ids(second) == (0, 1)
    assert first.method == "B0-sanity"
    assert first.scores == (None, None)
    assert first.rejected_births == (False, False)


def test_b1_is_threshold_aware_before_hungarian_assignment() -> None:
    tracker = B1FeatureTracker(sequence_id="scene", feature_threshold=0.8)
    tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=0)

    # The globally highest unrestricted assignment would consume query 0 with
    # the wrong previous track and leave a high-quality edge unmatched.  The
    # low edge must be forbidden before Hungarian optimization.
    second = tracker.step(
        _observation([[0.75, 0.6614378], [0.0, 1.0]]), stage_id=1
    )

    assert _ids(second) == (2, 1)
    assert second.matched_previous == (-1, 1)
    assert second.births == (True, False)


def test_threshold_aware_hungarian_breaks_equal_scores_by_low_indices() -> None:
    score = torch.ones((3, 4), dtype=torch.float64)

    assert threshold_aware_hungarian(score, 1.0) == (
        (0, 0),
        (1, 1),
        (2, 2),
    )


def test_b1_handles_q_not_equal_to_dynamic_identity_count() -> None:
    steps = run_b1(
        [
            _observation([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            _observation([[1.0, 0.0]]),
            _observation(
                [[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]
            ),
        ],
        sequence_id="scene",
    )

    assert _ids(steps[0]) == (0, 1, 2)
    assert _ids(steps[1]) == (0,)
    assert _ids(steps[2]) == (3, 4, 5, 0)
    assert steps[2].matched_previous == (-1, -1, -1, 0)


def test_b2_uses_foreground_renormalized_class_similarity() -> None:
    probabilities = torch.tensor(
        [[[0.90, 0.05, 0.05], [0.10, 0.45, 0.45]]], dtype=torch.float64
    )
    normalized = foreground_normalized_class_prob(probabilities, background_class=0)
    torch.testing.assert_close(
        normalized,
        torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.float64),
    )

    tracker = B2FeatureClassTracker(
        sequence_id="scene", feature_threshold=-1.0, class_weight=1.0
    )
    tracker.step(
        _observation(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        stage_id=0,
    )
    second = tracker.step(
        _observation(
            [[1.0, 0.0], [0.8, 0.6]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ),
        stage_id=1,
    )
    assert _ids(second) == (1, 0)
    assert second.matched_previous == (1, 0)


def test_b2_lambda_zero_degenerates_to_b1() -> None:
    stages = [
        _observation(
            [[1.0, 0.0], [0.8, 0.6]],
            [[0.99, 0.01], [0.01, 0.99]],
        ),
        _observation(
            [[0.8, 0.6], [1.0, 0.0]],
            [[0.01, 0.99], [0.99, 0.01]],
        ),
    ]
    b1 = run_b1(stages, sequence_id="scene", feature_threshold=0.5)
    b2 = run_b2(
        stages,
        sequence_id="scene",
        feature_threshold=0.5,
        class_weight=0.0,
    )
    assert _ids(b2[1]) == _ids(b1[1])
    assert b2[1].matched_previous == b1[1].matched_previous


def test_b3_updates_only_active_previous_stage_ema_and_births_after_gap() -> None:
    tracker = B3EmaTracker(
        sequence_id="scene", feature_threshold=0.5, class_weight=0.0, update_rate=0.2
    )
    first = tracker.step(_observation([[1.0, 0.0]]), stage_id=0)
    second = tracker.step(_observation([[0.0, 1.0]]), stage_id=1)
    assert _ids(first) == (0,)
    assert _ids(second) == (1,)
    torch.testing.assert_close(
        tracker.prototypes[1], torch.tensor([0.0, 1.0], dtype=torch.float64)
    )
    assert 0 not in tracker.prototypes

    # No observation at stage 2 means no active prototype to reactivate.
    tracker.step(_observation([[0.0, 0.0]], valid=[False]), stage_id=2)
    fourth = tracker.step(_observation([[0.0, 1.0]]), stage_id=3)
    assert _ids(fourth) == (2,)
    assert fourth.births == (True,)


def test_b3_ema_value_is_exact_and_does_not_mutate_input() -> None:
    tracker = B3EmaTracker(
        sequence_id="scene", feature_threshold=-1.0, class_weight=0.0, update_rate=0.25
    )
    tracker.step(_observation([[2.0, 0.0]]), stage_id=0)
    source = _observation([[0.0, 2.0]])
    original = source["features"].clone()
    tracker.step(source, stage_id=1)
    torch.testing.assert_close(
        tracker.prototypes[0], torch.tensor([1.5, 0.5], dtype=torch.float64)
    )
    torch.testing.assert_close(source["features"], original)


def test_gap_forces_birth_for_feature_tracker() -> None:
    tracker = B1FeatureTracker(sequence_id="scene", feature_threshold=-1.0)
    tracker.step(_observation([[1.0, 0.0]]), stage_id=0)
    gap = tracker.step(_observation([[1.0, 0.0]]), stage_id=2)
    assert _ids(gap) == (1,)
    assert gap.matched_previous == (-1,)


def test_track_step_is_frozen() -> None:
    step = run_b0([_observation([[1.0, 0.0]])], sequence_id="scene")[0]
    with pytest.raises(FrozenInstanceError):
        step.method = "changed"


def test_run_helpers_reset_namespace_per_sequence_and_method() -> None:
    stages = [_observation([[1.0, 0.0]])]
    assert _ids(run_b1(stages, sequence_id="a")[0]) == (0,)
    assert _ids(run_b1(stages, sequence_id="a")[0]) == (0,)
    assert _ids(run_b2(stages, sequence_id="a")[0]) == (0,)


def test_input_observation_is_unchanged_by_all_baselines() -> None:
    stages = [_observation([[1.0, 0.0], [0.0, 1.0]])]
    before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in stages[0].items()
    }
    for run in (run_b0, run_b0_sanity, run_b1, run_b2, run_b3):
        run(stages, sequence_id="scene")
    for key, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(stages[0][key], value)


def test_track_step_scores_use_typed_none_for_invalid_and_births() -> None:
    steps = run_b1(
        [
            _observation([[1.0, 0.0]], valid=[False]),
            _observation([[1.0, 0.0]]),
        ],
        sequence_id="scene",
    )

    assert steps[0].scores == (None,)
    assert steps[1].scores == (None,)
    assert steps[0].rejected_births == (False,)
    assert all(score is None or isinstance(score, float) for step in steps for score in step.scores)


def test_b4_matches_direct_frozen_p5_step_and_exposes_detached_snapshot() -> None:
    source = _mask_observation(
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [1.0, 0.0]],
        [[10.0, -10.0], [-10.0, 10.0]],
    )
    tracker = B4PersistentTracker(
        sequence_id="scene",
        capacity=2,
        class_weight=0.25,
        association_threshold=0.5,
        update_rate=0.2,
        max_update_rate=0.2,
    )
    result = tracker.step(source, stage_id=0)

    frozen = freeze_observation(source)
    p5_observation = LocalInstanceObservation(
        features=frozen.features.unsqueeze(0),
        class_prob=frozen.class_prob.unsqueeze(0),
        confidence=frozen.confidence.unsqueeze(0),
        latest_mask=[frozen.latest_mask[0]],
        valid=frozen.valid.unsqueeze(0),
    )
    memory = PersistentMemory(
        capacity=2,
        class_weight=0.25,
        association_threshold=0.5,
        update_rate=0.2,
        max_update_rate=0.2,
    )
    direct_state = memory.empty_state(p5_observation)
    direct = memory.step(p5_observation, direct_state, stage_index=0)

    assert result.track_ids == (0, 1)
    assert result.scores == (None, None)
    assert result.rejected_births == (False, False)
    assert result.state_snapshot is not None
    for actual, expected in zip(
        result.state_snapshot.tensors(), direct.state.tensors(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

    result.state_snapshot.embedding[0, 0, 0] = 99.0
    assert tracker.state is not None
    assert tracker.state.embedding[0, 0, 0].item() != 99.0


def test_b4_has_dynamic_q_and_reports_capacity_rejection() -> None:
    tracker = B4PersistentTracker(sequence_id="scene", capacity=1)
    result = tracker.step(
        _mask_observation(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[10.0], [10.0], [10.0]],
        ),
        stage_id=0,
    )

    assert result.track_ids == (0, None, None)
    assert result.rejected_births == (False, True, True)
    assert result.state_snapshot is not None
    assert result.state_snapshot.capacity == 1


def test_b4_dormant_reactivation_reuses_slot_without_birth() -> None:
    tracker = B4PersistentTracker(sequence_id="scene", capacity=1)
    first = _mask_observation([[1.0, 0.0]], [[1.0, 0.0]], [[10.0]])
    missing = _mask_observation(
        [[1.0, 0.0]], [[1.0, 0.0]], [[10.0]], valid=[False]
    )
    recovered = _mask_observation([[1.0, 0.0]], [[1.0, 0.0]], [[10.0]])

    assert tracker.step(first, stage_id=0).track_ids == (0,)
    assert tracker.step(missing, stage_id=1).track_ids == (None,)
    result = tracker.step(recovered, stage_id=2)

    assert result.track_ids == (0,)
    assert result.births == (False,)
    assert result.matched_previous == (-1,)
    assert result.scores[0] is not None


def test_b4_reset_clears_state_and_slot_namespace() -> None:
    tracker = B4PersistentTracker(sequence_id="scene", capacity=1)
    source = _mask_observation([[1.0, 0.0]], [[1.0, 0.0]], [[10.0]])
    assert tracker.step(source, stage_id=0).track_ids == (0,)
    tracker.reset()
    assert tracker.state is None
    assert tracker.step(source, stage_id=5).track_ids == (0,)


def test_b4_direct_step_signature_has_no_gt_argument() -> None:
    for tracker_type in (
        B0StageUniqueTracker,
        B0SanityTracker,
        B1FeatureTracker,
        B2FeatureClassTracker,
        B3EmaTracker,
        B4PersistentTracker,
    ):
        parameters = inspect.signature(tracker_type.step).parameters
        assert "gt" not in parameters
        assert "target" not in parameters


def _oracle_target() -> OracleStageTarget:
    return OracleStageTarget(
        gt_ids=(101, 7001),
        classes=(1, 2),
        masks=torch.tensor(
            [[True, True, False, False], [False, False, True, True]]
        ),
    )


def test_oracle_is_posthoc_class_compatible_and_deterministic() -> None:
    observation = _mask_observation(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        [[10.0, 10.0, -10.0, -10.0], [-10.0, -10.0, 10.0, 10.0], [10.0, -10.0, 10.0, -10.0]],
    )
    targets = [_oracle_target()]
    first = run_oracle_posthoc([observation], targets, sequence_id="scene")
    second = run_oracle_posthoc([observation], targets, sequence_id="scene")

    assert first == second
    assert first[0].track_ids == (101, 7001, ("Oracle", 0, 2))
    assert first[0].scores[0] == pytest.approx(1.0)
    assert first[0].scores[1] == pytest.approx(1.0)
    assert first[0].scores[2] is None
    assert first[0].state_snapshot is None


def test_oracle_clones_targets_and_observations_and_never_updates_tracker_state() -> None:
    observation = _mask_observation(
        [[1.0, 0.0]], [[0.0, 1.0, 0.0]], [[10.0, 10.0, -10.0, -10.0]]
    )
    target = _oracle_target()
    original_feature = observation["features"].clone()
    original_mask = target.masks.clone()
    result = run_oracle_posthoc([observation], [target], sequence_id="scene")

    assert torch.equal(observation["features"], original_feature)
    assert torch.equal(target.masks, original_mask)
    assert result[0].track_ids[0] == 101


def test_run_baseline_supports_b4_but_rejects_oracle_dispatch() -> None:
    observation = _mask_observation([[1.0, 0.0]], [[1.0, 0.0]], [[10.0]])
    assert run_baseline("b4", [observation], sequence_id="scene")[0].track_ids == (0,)
    with pytest.raises(ValueError, match="B4"):
        run_baseline("oracle", [observation], sequence_id="scene")


def test_association_diagnostics_are_typed_and_query_aligned() -> None:
    observations = [
        _observation([[1.0, 0.0], [0.0, 1.0]]),
        _observation([[0.8, 0.6], [0.0, 1.0]]),
    ]
    for step in run_b1(observations, sequence_id="scene") + run_b2(
        observations, sequence_id="scene"
    ) + run_b3(observations, sequence_id="scene"):
        diagnostics = step.diagnostics
        assert isinstance(diagnostics, AssociationDiagnostics)
        assert diagnostics.query_count == step.query_count
        for field in diagnostics.per_query_fields():
            assert len(field) == step.query_count


def test_b2_diagnostics_decompose_total_and_report_margin() -> None:
    tracker = B2FeatureClassTracker(
        sequence_id="scene", feature_threshold=-1.0, class_weight=0.25
    )
    tracker.step(
        _observation(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        stage_id=0,
    )
    result = tracker.step(
        _observation(
            [[0.8, 0.6], [0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ),
        stage_id=1,
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.selected_candidate_identity[0] == 0
    assert diagnostics.chosen_class_similarity[0] == pytest.approx(1.0)
    assert diagnostics.chosen_total_score[0] == pytest.approx(
        diagnostics.chosen_feature_similarity[0]
        + 0.25 * diagnostics.chosen_class_similarity[0]
    )
    assert diagnostics.best_score[0] is not None
    assert diagnostics.second_best_score[0] is not None
    assert diagnostics.score_margin[0] == pytest.approx(
        diagnostics.best_score[0] - diagnostics.second_best_score[0]
    )


def test_global_assignment_can_select_a_non_best_candidate() -> None:
    tracker = B1FeatureTracker(sequence_id="scene", feature_threshold=0.0)
    tracker.step(
        _observation([[1.0, 0.0], [0.9, 0.435889894]]), stage_id=0
    )
    result = tracker.step(
        _observation([[1.0, 0.0], [0.8, -0.6]]), stage_id=1
    )
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.selected_candidate_identity[0] == 1
    assert diagnostics.best_candidate_identity[0] == 0
    assert diagnostics.selected_candidate_identity[0] != diagnostics.best_candidate_identity[0]


def test_low_threshold_birth_keeps_best_candidate_diagnostic() -> None:
    tracker = B1FeatureTracker(sequence_id="scene", feature_threshold=0.95)
    tracker.step(_observation([[1.0, 0.0]]), stage_id=0)
    result = tracker.step(_observation([[0.8, 0.6]]), stage_id=1)
    diagnostics = result.diagnostics
    assert result.births == (True,)
    assert diagnostics is not None
    assert diagnostics.selected_candidate_identity == (None,)
    assert diagnostics.best_candidate_identity == (0,)
    assert diagnostics.chosen_total_score == (None,)
    assert diagnostics.best_score[0] == pytest.approx(0.8)
    assert diagnostics.second_best_score == (None,)


def test_b4_diagnostics_report_active_match_and_gap_reactivation_state() -> None:
    tracker = B4PersistentTracker(sequence_id="scene", capacity=1)
    source = _mask_observation([[1.0, 0.0]], [[1.0, 0.0]], [[10.0]])
    missing = _mask_observation(
        [[1.0, 0.0]], [[1.0, 0.0]], [[10.0]], valid=[False]
    )

    first = tracker.step(source, stage_id=0)
    active = tracker.step(source, stage_id=1)
    tracker.step(missing, stage_id=2)
    recovered = tracker.step(source, stage_id=3)

    assert first.diagnostics is not None
    assert first.diagnostics.selected_candidate_identity == (None,)
    active_diagnostics = active.diagnostics
    assert active_diagnostics is not None
    assert active_diagnostics.selected_candidate_identity == (0,)
    assert active_diagnostics.slot_occupied == (True,)
    assert active_diagnostics.slot_active == (True,)
    assert active_diagnostics.slot_age == (0,)
    assert active_diagnostics.last_seen_stage == (0,)
    assert active_diagnostics.reactivation == (False,)

    recovered_diagnostics = recovered.diagnostics
    assert recovered_diagnostics is not None
    assert recovered_diagnostics.selected_candidate_identity == (0,)
    assert recovered_diagnostics.reactivation == (True,)
    assert recovered_diagnostics.slot_active == (False,)
    assert recovered_diagnostics.slot_occupied == (True,)
    assert recovered_diagnostics.slot_age == (2,)
    assert recovered_diagnostics.last_seen_stage == (1,)


def test_diagnostics_do_not_change_existing_track_step_fields() -> None:
    result = run_b1(
        [_observation([[1.0, 0.0]]), _observation([[1.0, 0.0]])],
        sequence_id="scene",
    )
    assert result[0].method == "B1"
    assert result[0].sequence_id == "scene"
    assert result[0].stage_id == 0
    assert result[1].track_ids == (0,)
    assert result[1].matched_previous == (0,)
    assert result[1].scores == (1.0,)
    assert result[1].births == (False,)
    assert result[1].valid == (True,)
    assert result[1].rejected_births == (False,)
