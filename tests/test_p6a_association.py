from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from scripts.p6a_association import (
    B0SanityTracker,
    B0StageUniqueTracker,
    B1FeatureTracker,
    B2FeatureClassTracker,
    B3EmaTracker,
    IdentityNamespace,
    TrackStep,
    fan_out_observation,
    foreground_normalized_class_prob,
    freeze_observation,
    run_b0,
    run_b0_sanity,
    run_b1,
    run_b2,
    run_b3,
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


def test_b0_sanity_explicitly_reuses_local_query_index() -> None:
    tracker = B0SanityTracker(sequence_id="scene")
    first = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=0)
    second = tracker.step(_observation([[1.0, 0.0], [0.0, 1.0]]), stage_id=1)

    assert _ids(first) == (0, 1)
    assert _ids(second) == (0, 1)
    assert first.method == "B0-sanity"


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
