from __future__ import annotations

import pytest
import torch

from scripts.p6a_association import TrackStep
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
)


def _sidecar(
    *,
    stage: int,
    masks: torch.Tensor,
    scores: list[float],
    classes: list[int],
    queries: list[int],
) -> dict[str, object]:
    history = [f"scene0000_0{index}" for index in range(stage + 1)]
    return {
        "schema_version": 1,
        "key": {
            "master_sequence_id": "master-0",
            "reference_scene_id": "reference-0",
            "order_id": "canonical",
            "stage_index": stage,
            "history_scan_ids": history,
            "local_window_scan_ids": history[-1:] if stage == 0 else history[-2:],
        },
        "provenance": {
            "checkpoint_sha256": "a" * 64,
            "config_hash": "b" * 64,
            "protocol_manifest_hash": "c" * 64,
            "source_raw_observation_fingerprint": "d" * 64,
        },
        "task_prediction": {
            "pred_masks": masks.bool(),
            "pred_scores": torch.tensor(scores),
            "pred_classes": torch.tensor(classes),
            "source_query_ids": torch.tensor(queries),
            "source_class_ids": torch.tensor(classes),
            "latest_stage_index": stage,
        },
    }


def _step(stage: int, track_ids: tuple[object, ...]) -> TrackStep:
    count = len(track_ids)
    return TrackStep(
        method="B4",
        sequence_id="master-0:canonical",
        stage_id=stage,
        track_ids=track_ids,
        matched_previous=(-1,) * count,
        scores=(None,) * count,
        births=(False,) * count,
        valid=(True,) * count,
    )


def test_same_persistent_track_and_class_forms_one_mean_score_trajectory() -> None:
    accumulator = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    accumulator.add_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True]]),
            scores=[0.8],
            classes=[10],
            queries=[0],
        ),
        _step(0, (3, None)),
    )
    first = accumulator.snapshot()
    accumulator.add_stage(
        _sidecar(
            stage=1,
            masks=torch.tensor([[False], [True]]),
            scores=[0.6],
            classes=[10],
            queries=[1],
        ),
        _step(1, (None, 3)),
    )
    second = accumulator.snapshot()

    assert first.prediction["pred_masks"].tolist() == [[True]]
    assert second.prediction["pred_masks"].tolist() == [
        [True],
        [False],
        [True],
    ]
    assert second.prediction["pred_scores"].tolist() == pytest.approx([0.7])
    assert second.prediction["pred_classes"].tolist() == [10]
    assert second.keys[0].kind == "persistent"
    assert second.keys[0].persistent_track_id == 3


def test_same_track_different_official_classes_are_not_merged() -> None:
    accumulator = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    accumulator.add_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True, True]]),
            scores=[0.8, 0.4],
            classes=[10, 11],
            queries=[0, 0],
        ),
        _step(0, (7,)),
    )

    snapshot = accumulator.snapshot()
    assert snapshot.prediction["pred_masks"].shape == (1, 2)
    assert snapshot.prediction["pred_classes"].tolist() == [10, 11]
    assert [key.persistent_track_id for key in snapshot.keys] == [7, 7]


def test_unmatched_official_candidate_is_kept_as_stage_local_ephemeral() -> None:
    accumulator = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    accumulator.add_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True]]),
            scores=[0.9],
            classes=[10],
            queries=[1],
        ),
        _step(0, (4, None)),
    )
    accumulator.add_stage(
        _sidecar(
            stage=1,
            masks=torch.tensor([[True], [False]]),
            scores=[0.7],
            classes=[10],
            queries=[1],
        ),
        _step(1, (4, None)),
    )

    snapshot = accumulator.snapshot()
    assert len(snapshot.keys) == 2
    assert all(key.kind == "ephemeral" for key in snapshot.keys)
    assert [key.stage_index for key in snapshot.keys] == [0, 1]
    assert snapshot.prediction["pred_masks"].tolist() == [
        [True, False],
        [False, True],
        [False, False],
    ]


def test_later_stage_does_not_rewrite_committed_masks() -> None:
    accumulator = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
    accumulator.add_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True], [False]]),
            scores=[0.9],
            classes=[10],
            queries=[0],
        ),
        _step(0, (1,)),
    )
    committed = accumulator.snapshot().prediction["pred_masks"].clone()
    accumulator.add_stage(
        _sidecar(
            stage=1,
            masks=torch.tensor([[False], [True]]),
            scores=[0.5],
            classes=[10],
            queries=[0],
        ),
        _step(1, (1,)),
    )

    assert torch.equal(
        accumulator.snapshot().prediction["pred_masks"][:2], committed
    )
