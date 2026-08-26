from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.system_comparison_v3_oracle_identity import (
    OracleCandidateTrajectoryAccumulator,
    OracleIdentityError,
    OracleStageTarget,
    freeze_official_candidate_stage,
    link_oracle_identities,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/oracle_identity"


def _sidecar(
    *,
    stage: int,
    masks: torch.Tensor,
    scores: list[float],
    classes: list[int],
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
            "local_window_scan_ids": history[-2:],
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
            "source_query_ids": torch.arange(len(scores)),
            "source_class_ids": torch.tensor(classes),
            "latest_stage_index": stage,
        },
    }


def test_gt_is_required_only_after_official_candidates_are_frozen() -> None:
    sidecar = _sidecar(
        stage=0,
        masks=torch.tensor([[True], [False]]),
        scores=[0.8],
        classes=[7],
    )
    stage = freeze_official_candidate_stage(sidecar)

    assert stage.stage_index == 0
    assert stage.predicted_classes.tolist() == [7]
    with pytest.raises(OracleIdentityError, match="GT target is required"):
        link_oracle_identities(stage, None)


def test_oracle_keeps_candidate_masks_classes_scores_and_predicted_class() -> None:
    sidecars = (
        _sidecar(
            stage=0,
            masks=torch.tensor([[True], [False]]),
            scores=[0.8],
            classes=[7],
        ),
        _sidecar(
            stage=1,
            masks=torch.tensor([[False], [True]]),
            scores=[0.4],
            classes=[7],
        ),
    )
    originals = [
        {
            name: value.clone()
            for name, value in sidecar["task_prediction"].items()
            if isinstance(value, torch.Tensor)
        }
        for sidecar in sidecars
    ]
    accumulator = OracleCandidateTrajectoryAccumulator()
    for sidecar in sidecars:
        stage = freeze_official_candidate_stage(sidecar)
        target = OracleStageTarget(
            gt_ids=torch.tensor([42]),
            gt_masks=stage.predicted_masks.transpose(0, 1),
        )
        accumulator.add_stage(stage, link_oracle_identities(stage, target))

    snapshot = accumulator.snapshot()

    assert snapshot.prediction["pred_masks"].tolist() == [[True], [False], [False], [True]]
    assert snapshot.prediction["pred_scores"].tolist() == pytest.approx([0.6])
    assert snapshot.prediction["pred_classes"].tolist() == [7]
    assert snapshot.keys[0].oracle_gt_id == 42
    assert snapshot.keys[0].predicted_class_id == 7
    for sidecar, original in zip(sidecars, originals, strict=True):
        for name, value in original.items():
            assert torch.equal(sidecar["task_prediction"][name], value)


def test_oracle_matching_uses_iou_not_gt_semantics() -> None:
    stage = freeze_official_candidate_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor(
                [
                    [True, False],
                    [True, False],
                    [False, True],
                    [False, True],
                ]
            ),
            scores=[0.9, 0.7],
            classes=[5, 9],
        )
    )
    target = OracleStageTarget(
        gt_ids=torch.tensor([100, 200]),
        gt_masks=torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        ),
    )

    assert link_oracle_identities(stage, target, iou_threshold=0.5) == (100, 200)


def test_unmatched_official_candidate_remains_stage_local_ephemeral() -> None:
    stage = freeze_official_candidate_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True], [False]]),
            scores=[0.3],
            classes=[11],
        )
    )
    target = OracleStageTarget(
        gt_ids=torch.tensor([8]),
        gt_masks=torch.tensor([[False, True]]),
    )
    links = link_oracle_identities(stage, target)
    accumulator = OracleCandidateTrajectoryAccumulator()
    accumulator.add_stage(stage, links)
    key = accumulator.snapshot().keys[0]

    assert links == (None,)
    assert key.kind == "ephemeral"
    assert key.stage_index == 0
    assert key.candidate_index == 0
    assert key.oracle_gt_id is None


def test_oracle_rejects_changed_candidate_linkage_shape() -> None:
    stage = freeze_official_candidate_stage(
        _sidecar(
            stage=0,
            masks=torch.tensor([[True], [False]]),
            scores=[0.8],
            classes=[7],
        )
    )
    accumulator = OracleCandidateTrajectoryAccumulator()

    with pytest.raises(OracleIdentityError, match="one linkage per candidate"):
        accumulator.add_stage(stage, ())


def test_oracle_cli_help_runs_directly() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/system_comparison_v3_oracle_identity.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--cache-root" in completed.stdout


def test_oracle_artifact_contract_when_generated() -> None:
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Oracle-ID artifacts have not been generated yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["gate_or0"] == {
        "status": "PASS",
        "gt_access": "post_prediction_linkage_only",
        "iou_threshold": 0.5,
        "candidate_fields_unchanged": True,
        "predicted_class_retained": True,
        "unmatched_candidates_ephemeral": True,
    }
    assert manifest["execution"]["gpu_inference_performed"] is False
    assert manifest["execution"]["score_reducer"] == "mean"
    assert manifest["coverage"] == {
        "sequence_count": 129,
        "methods": ["FullHistory", "B2", "B4", "Oracle-ID"],
        "orders": ["canonical", "reverse", "sha256_seed45"],
        "horizons": [2, 3, 4, 5],
        "per_sequence_row_count": 2064,
        "aggregate_row_count": 64,
    }
    for name, metadata in manifest["outputs"].items():
        path = ARTIFACT_ROOT / name
        content = path.read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]
    for name, expected in (("oracle_per_sequence.csv", 2064), ("oracle_aggregate.csv", 64)):
        with (ARTIFACT_ROOT / name).open(newline="", encoding="utf-8") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == expected
