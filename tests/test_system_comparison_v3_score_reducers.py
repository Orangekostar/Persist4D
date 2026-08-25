from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.p6a_association import TrackStep
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
    V2InferenceError,
)
from scripts.system_comparison_v3_score_sensitivity import (
    assert_local_current_invariance,
    assert_score_only_snapshots,
    build_local_current_pair,
)

REDUCERS = ("mean", "latest", "max")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/score_sensitivity"


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


def _accumulators():
    return {
        reducer: OfficialCandidateTrajectoryAccumulator(score_reducer=reducer)
        for reducer in REDUCERS
    }


def _add(accumulators, sidecar, step):
    for accumulator in accumulators.values():
        accumulator.add_stage(sidecar, step)


def test_only_registered_score_reducers_are_accepted():
    for reducer in REDUCERS:
        assert (
            OfficialCandidateTrajectoryAccumulator(score_reducer=reducer).score_reducer
            == reducer
        )
    with pytest.raises(V2InferenceError, match="score reducer"):
        OfficialCandidateTrajectoryAccumulator(score_reducer="median")


def test_single_occurrence_and_ephemeral_scores_are_identical():
    accumulators = _accumulators()
    sidecar = _sidecar(
        stage=0,
        masks=torch.tensor([[True, False], [False, True]]),
        scores=[0.8, 0.3],
        classes=[10, 11],
        queries=[0, 1],
    )
    _add(accumulators, sidecar, _step(0, (7, None)))

    snapshots = {name: value.snapshot() for name, value in accumulators.items()}
    reference = snapshots["mean"]
    for snapshot in snapshots.values():
        assert snapshot.prediction["pred_scores"].tolist() == pytest.approx([0.8, 0.3])
        assert torch.equal(
            snapshot.prediction["pred_masks"], reference.prediction["pred_masks"]
        )
        assert torch.equal(
            snapshot.prediction["pred_classes"],
            reference.prediction["pred_classes"],
        )
        assert snapshot.keys == reference.keys
    assert reference.keys[1].kind == "ephemeral"


def test_multiple_occurrences_change_scores_only():
    accumulators = _accumulators()
    _add(
        accumulators,
        _sidecar(
            stage=0,
            masks=torch.tensor([[True], [False]]),
            scores=[0.8],
            classes=[10],
            queries=[0],
        ),
        _step(0, (7,)),
    )
    _add(
        accumulators,
        _sidecar(
            stage=1,
            masks=torch.tensor([[False], [True]]),
            scores=[0.4],
            classes=[10],
            queries=[0],
        ),
        _step(1, (7,)),
    )
    snapshots = {name: value.snapshot() for name, value in accumulators.items()}

    assert snapshots["mean"].prediction["pred_scores"].tolist() == pytest.approx([0.6])
    assert snapshots["latest"].prediction["pred_scores"].tolist() == pytest.approx(
        [0.4]
    )
    assert snapshots["max"].prediction["pred_scores"].tolist() == pytest.approx([0.8])
    reference = snapshots["mean"]
    for snapshot in snapshots.values():
        assert torch.equal(
            snapshot.prediction["pred_masks"], reference.prediction["pred_masks"]
        )
        assert torch.equal(
            snapshot.prediction["pred_classes"],
            reference.prediction["pred_classes"],
        )
        assert snapshot.keys == reference.keys


def test_gap_reactivation_has_no_future_score_leakage():
    accumulators = _accumulators()
    _add(
        accumulators,
        _sidecar(
            stage=0,
            masks=torch.tensor([[True]]),
            scores=[0.9],
            classes=[10],
            queries=[0],
        ),
        _step(0, (7,)),
    )
    _add(
        accumulators,
        _sidecar(
            stage=1,
            masks=torch.tensor([[True]]),
            scores=[0.2],
            classes=[11],
            queries=[0],
        ),
        _step(1, (None,)),
    )
    before_reactivation = {
        name: value.snapshot() for name, value in accumulators.items()
    }
    for snapshot in before_reactivation.values():
        assert snapshot.prediction["pred_scores"][0].item() == pytest.approx(0.9)

    _add(
        accumulators,
        _sidecar(
            stage=2,
            masks=torch.tensor([[False], [True]]),
            scores=[0.3],
            classes=[10],
            queries=[0],
        ),
        _step(2, (7,)),
    )
    snapshots = {name: value.snapshot() for name, value in accumulators.items()}
    assert snapshots["mean"].prediction["pred_scores"][0].item() == pytest.approx(0.6)
    assert snapshots["latest"].prediction["pred_scores"][0].item() == pytest.approx(0.3)
    assert snapshots["max"].prediction["pred_scores"][0].item() == pytest.approx(0.9)
    for snapshot in snapshots.values():
        ephemeral = [
            index for index, key in enumerate(snapshot.keys) if key.kind == "ephemeral"
        ]
        assert len(ephemeral) == 1
        assert snapshot.prediction["pred_scores"][ephemeral[0]].item() == pytest.approx(
            0.2
        )


def test_score_only_snapshot_validator_rejects_mask_or_key_drift():
    accumulators = _accumulators()
    for stage, score in ((0, 0.9), (1, 0.3)):
        _add(
            accumulators,
            _sidecar(
                stage=stage,
                masks=torch.tensor([[True]]),
                scores=[score],
                classes=[10],
                queries=[0],
            ),
            _step(stage, (7,)),
        )
    snapshots = {name: value.snapshot() for name, value in accumulators.items()}
    assert_score_only_snapshots(snapshots)

    changed = dict(snapshots)
    changed_snapshot = changed["latest"]
    changed_snapshot.prediction["pred_masks"][0, 0] = False
    with pytest.raises(ValueError, match="masks"):
        assert_score_only_snapshots(changed)


def test_direct_local_current_pair_uses_sidecar_scores_without_tracker():
    sidecar = _sidecar(
        stage=1,
        masks=torch.tensor([[True], [False], [True]]),
        scores=[0.73],
        classes=[8],
        queries=[0],
    )
    raw = {
        "key": sidecar["key"],
        "target": {
            "gt_ids": torch.tensor([4]),
            "gt_classes": torch.tensor([8]),
            "gt_masks": torch.tensor([[True, False, True]]),
            "changes": torch.tensor([0]),
            "change_labels_valid": False,
            "change_label_semantics": "unavailable_all_static_placeholder",
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }

    pair = build_local_current_pair(
        raw_payload=raw,
        sidecar=sidecar,
        class_mapper=lambda value: value + 100,
    )

    assert pair.prediction["pred_scores"].tolist() == pytest.approx([0.73])
    assert pair.prediction["pred_classes"].tolist() == [8]
    assert pair.target["labels"].tolist() == [108]
    assert pair.target["ids"].tolist() == [4]
    assert pair.target["temporal_stages"].tolist() == [0, 0, 0]


def test_local_current_invariance_is_exact_across_trackers_and_reducers():
    rows = [
        {
            "reference_scene_id": "reference-0",
            "master_sequence_id": "master-0",
            "order_id": "canonical",
            "horizon": 2,
            "tracker": tracker,
            "score_reducer": reducer,
            "local_sidecar_sha256": "a" * 64,
            "local_current_AP": 0.4,
            "local_current_AP50": 0.5,
            "local_current_AP25": 0.6,
            "local_current_REC": 0.7,
        }
        for tracker in ("B2", "B3", "B4")
        for reducer in REDUCERS
    ]
    assert assert_local_current_invariance(rows)["status"] == "pass_exact"

    rows[-1]["local_current_AP"] = 0.4000001
    with pytest.raises(ValueError, match="local-current"):
        assert_local_current_invariance(rows)


def test_score_sensitivity_cli_help_runs_directly():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/system_comparison_v3_score_sensitivity.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--cache-root" in completed.stdout


def test_score_sensitivity_artifact_contract_when_generated():
    manifest_path = ARTIFACT_ROOT / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("score-sensitivity artifacts have not been generated yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "pass"
    assert manifest["gate_ev0"]["status"] == "PASS"
    assert manifest["gate_ev0"]["primary_reducer"] == "mean"
    assert manifest["gate_ev0"]["mean_v2_regression_max_abs_diff"] <= 1e-12
    assert manifest["gate_ev0"]["score_only_snapshot_status"] == "pass_exact"
    assert manifest["gate_ev0"]["local_current_invariance"]["status"] == ("pass_exact")
    assert len(manifest["inputs"]["checkpoint_sha256"]) == 64
    assert len(manifest["inputs"]["config_sha256"]) == 64
    assert len(manifest["inputs"]["protocol_manifest_sha256"]) == 64
    assert len(manifest["inputs"]["cache_records_sha256"]) == 64
    assert manifest["execution"]["gpu_inference_performed"] is False
    assert manifest["coverage"] == {
        "aggregate_row_count": 144,
        "horizons": [2, 3, 4, 5],
        "orders": ["canonical", "reverse", "sha256_seed45"],
        "per_cluster_row_count": 648,
        "per_sequence_row_count": 4644,
        "reducers": ["mean", "latest", "max"],
        "reference_cluster_count": 6,
        "sequence_count": 129,
        "trackers": ["B2", "B3", "B4"],
    }

    for name, metadata in manifest["outputs"].items():
        path = ARTIFACT_ROOT / name
        content = path.read_bytes()
        assert len(content) == metadata["bytes"]
        assert hashlib.sha256(content).hexdigest() == metadata["sha256"]

    expected_rows = {
        "per_sequence.csv": 4644,
        "aggregate.csv": 144,
        "per_cluster.csv": 648,
    }
    for name, count in expected_rows.items():
        with (ARTIFACT_ROOT / name).open(newline="", encoding="utf-8") as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == count
