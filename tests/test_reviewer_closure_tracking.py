from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts import reviewer_closure_tracking as tracking
from scripts.p6a_association import (
    B1FeatureTracker,
    B2FeatureClassTracker,
    B3EmaTracker,
    B4PersistentTracker,
    freeze_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/reviewer_closure_tracking.py"


def _api(name: str):
    value = getattr(tracking, name, None)
    assert value is not None, f"missing reviewer-closure tracking API: {name}"
    return value


def _key(horizon: int) -> dict[str, object]:
    scans = ["S1", "S2", "S3", "S4", "S5"]
    return {
        "reference_scene_id": "reference-1",
        "master_sequence_id": "master-1",
        "order_id": "canonical",
        "horizon": horizon,
        "history_scan_ids": scans[:horizon],
        "scan_indices": list(range(horizon)),
    }


def _stage(
    horizon: int,
    *,
    visible: bool,
    feature: tuple[float, float] = (1.0, 0.0),
) -> object:
    valid = torch.tensor([visible, False])
    masks = torch.tensor(
        [[visible, visible, False], [False, False, False]], dtype=torch.bool
    )
    observation = freeze_observation(
        {
            "features": torch.tensor([feature, (0.0, 1.0)]),
            "class_prob": torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]),
            "confidence": torch.tensor([0.9, 0.8]),
            "valid": valid,
            "latest_mask": masks.float(),
            "mask_support": masks.sum(dim=1),
        }
    )
    return _api("FullHistoryTrackerStage")(
        key=_key(horizon),
        observation=observation,
        local_query_ids=torch.tensor([7, 11]),
        gt_ids=torch.tensor([101]),
        gt_classes=torch.tensor([20]),
        gt_masks=torch.tensor([[visible, visible, False]], dtype=torch.bool),
        native_issued_ids=(
            torch.tensor([7]) if visible else torch.empty(0, dtype=torch.long)
        ),
        pred_classes=(
            torch.tensor([20]) if visible else torch.empty(0, dtype=torch.long)
        ),
        pred_masks=(
            masks[:1].transpose(0, 1).contiguous()
            if visible
            else torch.empty((3, 0), dtype=torch.bool)
        ),
    )


def _sequence(*, master: str = "master-1") -> object:
    stages = tuple(
        _stage(horizon, visible=horizon in {2, 5}) for horizon in range(2, 6)
    )
    if master != "master-1":
        adjusted = []
        for stage in stages:
            key = copy.deepcopy(stage.key)
            key["master_sequence_id"] = master
            adjusted.append(
                _api("FullHistoryTrackerStage")(
                    key=key,
                    observation=stage.observation,
                    local_query_ids=stage.local_query_ids,
                    gt_ids=stage.gt_ids,
                    gt_classes=stage.gt_classes,
                    gt_masks=stage.gt_masks,
                    native_issued_ids=stage.native_issued_ids,
                    pred_classes=stage.pred_classes,
                    pred_masks=stage.pred_masks,
                )
            )
        stages = tuple(adjusted)
    return _api("FullHistoryTrackerSequence")(stages=stages)


def _factories() -> dict[str, object]:
    return {
        "B1": lambda sequence_id: B1FeatureTracker(
            sequence_id=sequence_id, feature_threshold=0.5, background_class=2
        ),
        "B2": lambda sequence_id: B2FeatureClassTracker(
            sequence_id=sequence_id,
            feature_threshold=0.5,
            class_weight=0.25,
            background_class=2,
        ),
        "B3": lambda sequence_id: B3EmaTracker(
            sequence_id=sequence_id,
            feature_threshold=0.5,
            class_weight=0.25,
            background_class=2,
            update_rate=0.2,
        ),
        "B4": lambda sequence_id: B4PersistentTracker(
            sequence_id=sequence_id,
            capacity=8,
            association_threshold=0.5,
            class_weight=0.25,
            update_rate=0.2,
            max_update_rate=0.2,
        ),
    }


def test_tracking_module_exists() -> None:
    assert MODULE_PATH.is_file()


def test_real_manifests_load_one_exact_o2_to_o5_sequence() -> None:
    sequence = next(_api("iter_full_history_tracker_sequences")())

    assert [stage.horizon for stage in sequence.stages] == [2, 3, 4, 5]
    assert sequence.reference_scene_id in {
        "10b17940-3938-2467-8a7a-958300ba83d3",
        "137a8158-1db5-2cc0-8003-31c12610471e",
        "280d8ebb-6cc6-2788-9153-98959a2da801",
        "5630cfcf-12bf-2860-8784-83d28a611a83",
        "8eabc45f-5af7-2f32-8528-640861d2a135",
        "ddc73797-765b-241a-9e2c-097c5989baf6",
    }
    assert all(
        stage.observation.features.shape == (100, 128) for stage in sequence.stages
    )


def test_tracking_manifest_validation_rejects_sidecar_tampering() -> None:
    reviewer = json.loads(
        (
            REPO_ROOT / "artifacts/reviewer_closure/reviewer_closure_manifest.json"
        ).read_text()
    )
    sidecar = json.loads(
        (
            REPO_ROOT
            / "artifacts/reviewer_closure/full_history_observations_v2/manifest.json"
        ).read_text()
    )
    replay = json.loads(
        (
            REPO_ROOT
            / "artifacts/reviewer_closure/full_history_replay_v2/manifest.json"
        ).read_text()
    )
    changed = copy.deepcopy(sidecar)
    changed["entries"][0]["source_prediction_content_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="replay prediction|content"):
        _api("validate_full_history_tracking_manifests")(
            reviewer_manifest=reviewer,
            replay_manifest=replay,
            sidecar_manifest=changed,
        )


def test_runner_reuses_frozen_trackers_without_changing_step_results() -> None:
    sequence = _api("FullHistoryTrackerSequence")(
        stages=tuple(_stage(horizon, visible=True) for horizon in range(2, 6))
    )
    factory = _factories()["B2"]
    observed = _api("run_full_history_tracker")(sequence, factory, method_id="B2")

    direct = factory(sequence.sequence_id)
    expected = tuple(
        direct.step(stage.observation, stage_id=stage.horizon)
        for stage in sequence.stages
    )
    assert [step.track_ids for step in observed] == [
        step.track_ids for step in expected
    ]
    assert [step.matched_previous for step in observed] == [
        step.matched_previous for step in expected
    ]


def test_sequence_rejects_future_or_non_nested_prefix() -> None:
    stages = list(_sequence().stages)
    changed = copy.deepcopy(stages[1].key)
    changed["history_scan_ids"][-1] = "future-S5"
    stages[1] = _api("FullHistoryTrackerStage")(
        key=changed,
        observation=stages[1].observation,
        local_query_ids=stages[1].local_query_ids,
        gt_ids=stages[1].gt_ids,
        gt_classes=stages[1].gt_classes,
        gt_masks=stages[1].gt_masks,
        native_issued_ids=stages[1].native_issued_ids,
        pred_classes=stages[1].pred_classes,
        pred_masks=stages[1].pred_masks,
    )
    with pytest.raises(ValueError, match="nested|future|prefix"):
        _api("FullHistoryTrackerSequence")(stages=tuple(stages))


def test_trackers_reset_per_sequence_and_keep_independent_namespaces() -> None:
    results = _api("evaluate_full_history_tracker_sequences")(
        [_sequence(master="master-a"), _sequence(master="master-b")],
        tracker_factories=_factories(),
    )
    for master in ("master-a", "master-b"):
        first_ids = {
            method: updates[0].assignments[101]
            for (result_master, order, method), updates in results.updates.items()
            if result_master == master
            and method != "FullHistoryNative"
            and order == "canonical"
        }
        assert set(first_ids.values()) == {0}


def test_frozen_factory_and_raw_artifact_use_registered_paper_names() -> None:
    factories = _api("build_full_history_tracker_factories")()
    assert tuple(factories) == ("B1", "B2", "B3", "B4")
    assert factories["B1"]("scope").feature_threshold == pytest.approx(0.5)
    assert factories["B3"]("scope").update_rate == pytest.approx(0.2)

    result = _api("evaluate_full_history_tracker_sequences")(
        [_sequence()], tracker_factories=_factories()
    )
    artifact = _api("build_full_history_tracking_artifact")(
        result,
        reviewer_manifest_content_sha256="1" * 64,
        replay_manifest_content_sha256="2" * 64,
        sidecar_manifest_content_sha256="3" * 64,
        tracker_config_sha256="4" * 64,
        source_commit="5" * 40,
        expected_sequence_count=1,
    )

    assert artifact["sequence_count"] == 1
    assert artifact["row_count"] == 5 * 4
    assert {row["method"] for row in artifact["rows"]} == {
        "ReScene4D Full-History",
        "Pairwise Feature Association",
        "Pairwise Feature-Class Association",
        "EMA Temporal Association",
        "Full-History + Persistent-State Diagnostic",
    }
    assert artifact["content_sha256"] == _api("tracking_content_sha256")(artifact)


def test_gap_case_measures_pairwise_ema_failure_and_persistent_recovery() -> None:
    result = _api("evaluate_full_history_tracker_sequences")(
        [_sequence()], tracker_factories=_factories()
    )

    for method in ("B1", "B2", "B3"):
        metrics = result.per_sequence_metrics[("master-1", "canonical", method, 5)]
        assert metrics["gap_opportunities"] == 1
        assert metrics["recovery_attempts"] == 1
        assert metrics["correct_recoveries"] == 0
        assert metrics["gap_recovery_recall"] == 0.0

    diagnostic = result.per_sequence_metrics[("master-1", "canonical", "B4", 5)]
    assert diagnostic["gap_opportunities"] == 1
    assert diagnostic["correct_recoveries"] == 1
    assert diagnostic["gap_recovery_accuracy"] == 1.0
    assert diagnostic["gap_recovery_recall"] == 1.0
