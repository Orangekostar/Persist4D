from __future__ import annotations

import pytest
import torch

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.p6a_association import B4PersistentTracker, TrackStep
from scripts.p6b_association import P6BTracker, P6BTransitionDetails


def _observation(
    features: list[list[float]],
    class_prob: list[list[float]],
    *,
    confidence: list[float] | None = None,
    masks: list[list[float]] | None = None,
    valid: list[bool] | None = None,
) -> dict[str, torch.Tensor]:
    query_count = len(features)
    return {
        "features": torch.tensor(features, dtype=torch.float64),
        "class_prob": torch.tensor(class_prob, dtype=torch.float64),
        "confidence": torch.tensor(
            confidence if confidence is not None else [1.0] * query_count,
            dtype=torch.float64,
        ),
        "latest_mask": torch.tensor(
            masks
            if masks is not None
            else [[10.0, 10.0, 10.0] for _ in range(query_count)],
            dtype=torch.float64,
        ),
        "valid": torch.tensor(
            valid if valid is not None else [True] * query_count,
            dtype=torch.bool,
        ),
    }


def _config() -> P6BMemoryConfig:
    return P6BMemoryConfig(
        capacity=3,
        active_threshold=0.50,
        reactivation_threshold=0.85,
        reactivation_margin=0.10,
        class_weight=0.0,
        background_class=2,
        consolidation_confidence=0.90,
        consolidation_margin=0.10,
        birth_confidence=0.80,
        birth_minimum_mask_support=2,
        birth_max_entropy=0.50,
    )


def _five_stage_run() -> tuple[P6BTracker, list[TrackStep], list[P6BTransitionDetails]]:
    tracker = P6BTracker(sequence_id="synthetic", config=_config())
    stages = [
        _observation(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]],
            confidence=[0.99, 0.99],
            masks=[[10.0, 10.0, 10.0], [10.0, -10.0, -10.0]],
        ),
        _observation(
            [[0.98, 0.198997487, 0.0], [0.0, 1.0, 0.0]],
            [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]],
            confidence=[0.95, 0.99],
        ),
        _observation(
            [[0.0, 1.0, 0.0]],
            [[0.01, 0.98, 0.01]],
            confidence=[0.99],
        ),
        _observation(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.98, 0.01, 0.01], [0.98, 0.01, 0.01]],
            confidence=[0.80, 0.99],
        ),
        _observation(
            [[-0.577350269, -0.577350269, -0.577350269]],
            [[0.98, 0.01, 0.01]],
            confidence=[0.99],
        ),
    ]
    steps: list[TrackStep] = []
    transitions: list[P6BTransitionDetails] = []
    for stage_id, observation in enumerate(stages):
        steps.append(tracker.step(observation, stage_id=stage_id))
        assert tracker.last_transition is not None
        transitions.append(tracker.last_transition)
    return tracker, steps, transitions


def test_p6b_tracker_exposes_exact_five_stage_transition_contract() -> None:
    _, steps, transitions = _five_stage_run()
    reactivation_score = 0.9992805337461891

    assert [step.method for step in steps] == ["P6B"] * 5
    assert [step.sequence_id for step in steps] == ["synthetic"] * 5
    assert steps[0].track_ids == (0, None)
    assert steps[0].births == (True, False)
    assert steps[0].rejected_births == (False, True)
    assert transitions[0].rejected_birth_support == (False, True)

    assert steps[1].track_ids == (0, 1)
    assert steps[1].matched_previous == (0, -1)
    assert steps[1].births == (False, True)
    assert transitions[1].consolidated == (True, False)

    assert steps[2].track_ids == (1,)
    assert steps[2].matched_previous == (1,)
    assert steps[2].births == (False,)

    assert steps[3].track_ids == (0, 2)
    assert steps[3].matched_previous == (-1, -1)
    assert steps[3].scores[0] == pytest.approx(reactivation_score, abs=1e-12)
    assert steps[3].scores[1] is None
    assert steps[3].births == (False, True)
    assert transitions[3].reactivations == (True, False)
    assert transitions[3].consolidated == (False, False)

    assert steps[4].track_ids == (None,)
    assert steps[4].births == (False,)
    assert steps[4].rejected_births == (True,)
    assert transitions[4].rejected_birth_capacity == (True,)


def test_p6b_diagnostics_are_query_aligned_and_report_dormant_match() -> None:
    _, steps, _ = _five_stage_run()
    diagnostics = steps[3].diagnostics
    reactivation_score = 0.9992805337461891

    assert diagnostics is not None
    assert diagnostics.query_count == 2
    assert diagnostics.selected_candidate_identity == (0, None)
    assert diagnostics.best_candidate_identity == (0, 0)
    assert diagnostics.chosen_feature_similarity[0] == pytest.approx(
        reactivation_score, abs=1e-12
    )
    assert diagnostics.chosen_feature_similarity[1] is None
    assert diagnostics.chosen_total_score == (
        pytest.approx(reactivation_score, abs=1e-12),
        None,
    )
    assert diagnostics.best_score == (
        pytest.approx(reactivation_score, abs=1e-12),
        pytest.approx(0.0),
    )
    assert diagnostics.second_best_score == (
        pytest.approx(0.0),
        pytest.approx(0.0),
    )
    assert diagnostics.score_margin == (
        pytest.approx(reactivation_score, abs=1e-12),
        pytest.approx(0.0),
    )
    assert diagnostics.slot_age == (2, 2)
    assert diagnostics.last_seen_stage == (1, 1)
    assert diagnostics.slot_active == (False, False)
    assert diagnostics.slot_occupied == (True, True)
    assert diagnostics.reactivation == (True, False)


def test_p6b_state_snapshot_is_detached_from_tracker_state() -> None:
    tracker, steps, _ = _five_stage_run()
    snapshot = steps[3].state_snapshot

    assert snapshot is not None
    assert tracker.state is not None
    expected = tracker.state.embedding[0, 0, 0].item()
    snapshot.embedding[0, 0, 0] = 99.0
    assert tracker.state.embedding[0, 0, 0].item() == expected


def test_p6b_reset_clears_state_and_can_change_sequence() -> None:
    tracker, _, _ = _five_stage_run()

    tracker.reset(sequence_id="next")

    assert tracker.sequence_id == "next"
    assert tracker.state is None
    assert tracker.last_transition is None
    first = tracker.step(
        _observation(
            [[1.0, 0.0, 0.0]],
            [[0.98, 0.01, 0.01]],
        ),
        stage_id=0,
    )
    assert first.track_ids == (0,)


def test_importing_and_running_p6b_does_not_change_frozen_b4() -> None:
    source = _observation(
        [[1.0, 0.0, 0.0]],
        [[0.98, 0.01, 0.01]],
    )
    before = B4PersistentTracker(
        sequence_id="before",
        capacity=1,
        class_weight=0.25,
        association_threshold=0.5,
    ).step(source, stage_id=0)

    P6BTracker(sequence_id="p6b", config=_config()).step(source, stage_id=0)

    after = B4PersistentTracker(
        sequence_id="after",
        capacity=1,
        class_weight=0.25,
        association_threshold=0.5,
    ).step(source, stage_id=0)
    assert before.track_ids == after.track_ids == (0,)
    assert before.births == after.births == (True,)
    assert before.rejected_births == after.rejected_births == (False,)
    assert before.state_snapshot is not None
    assert after.state_snapshot is not None
    for before_tensor, after_tensor in zip(
        before.state_snapshot.tensors(),
        after.state_snapshot.tensors(),
        strict=True,
    ):
        torch.testing.assert_close(before_tensor, after_tensor)


def test_p6b_tracker_runtime_api_has_no_ground_truth_inputs() -> None:
    tracker = P6BTracker(sequence_id="scene", config=_config())

    with pytest.raises(TypeError, match="ground_truth"):
        tracker.step(
            _observation(
                [[1.0, 0.0, 0.0]],
                [[0.98, 0.01, 0.01]],
            ),
            stage_id=0,
            ground_truth=torch.tensor([1]),  # type: ignore[call-arg]
        )


def test_uniform_foreground_entropy_stays_within_theoretical_range() -> None:
    tracker = P6BTracker(
        sequence_id="entropy",
        config=P6BMemoryConfig(
            capacity=1,
            background_class=18,
            birth_minimum_mask_support=1,
            birth_max_entropy=0.5,
        ),
    )
    class_prob = [[1.0 / 18.0] * 18 + [0.0]]

    step = tracker.step(
        _observation(
            [[1.0, 0.0, 0.0]],
            class_prob,
            confidence=[0.99],
        ),
        stage_id=0,
    )

    assert step.rejected_births == (True,)
    assert tracker.last_transition is not None
    assert tracker.last_transition.birth_entropy == (1.0,)
