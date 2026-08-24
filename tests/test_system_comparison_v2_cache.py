from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from scripts.p6a_cache import validate_cache_payload
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.system_comparison_v2_cache import (
    build_task_sidecar,
    load_task_sidecar,
    observation_fingerprint,
    task_sidecar_digest,
    validate_task_sidecar,
    write_task_sidecar,
)


def _raw_payload() -> dict[str, object]:
    masks = torch.tensor(
        [[True, True, False], [False, True, True]], dtype=torch.bool
    )
    return {
        "schema_version": 3,
        "key": {
            "master_sequence_id": "scene0001_00-scene0001_01",
            "reference_scene_id": "reference-1",
            "order_id": "canonical",
            "stage_index": 1,
            "history_scan_ids": ["scene0001_00", "scene0001_01"],
            "local_window_scan_ids": ["scene0001_00", "scene0001_01"],
        },
        "provenance": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
        },
        "observation": {
            "features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "class_prob": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]),
            "confidence": torch.tensor([0.8, 0.7]),
            "valid": torch.tensor([True, True]),
            "masks": masks,
            "mask_support": torch.tensor([2, 2], dtype=torch.long),
            "local_query_ids": torch.tensor([0, 1], dtype=torch.long),
        },
        "target": {
            "gt_ids": torch.tensor([10, 20], dtype=torch.long),
            "gt_classes": torch.tensor([0, 1], dtype=torch.long),
            "gt_masks": masks.clone(),
            "changes": torch.tensor([0, 0], dtype=torch.long),
            "change_labels_valid": False,
            "change_label_semantics": (
                "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
            ),
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }


def _prediction() -> OfficialTaskPrediction:
    masks = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
            [False, True, False],
        ],
        dtype=torch.bool,
    )
    stages = torch.tensor([0, 1, 1, 1], dtype=torch.long)
    return OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=torch.tensor([0.9, 0.8, 0.4]),
        pred_classes=torch.tensor([3, 4, 3]),
        source_query_ids=torch.tensor([0, 1, 0]),
        source_class_ids=torch.tensor([0, 1, 0]),
        temporal_stages=stages,
        latest_stage_index=1,
        latest_stage_masks=masks[stages == 1],
    )


def test_task_sidecar_preserves_official_candidate_lineage() -> None:
    raw = _raw_payload()
    sidecar = build_task_sidecar(
        raw_cache_payload=raw,
        official_prediction=_prediction(),
        protocol_manifest_sha256="5" * 64,
    )

    validate_task_sidecar(sidecar)
    assert sidecar["schema_version"] == 1
    assert sidecar["key"] == raw["key"]
    task = sidecar["task_prediction"]
    assert task["pred_masks"].shape == (3, 3)
    assert task["pred_scores"].tolist() == pytest.approx([0.9, 0.8, 0.4])
    assert task["pred_classes"].tolist() == [3, 4, 3]
    assert task["source_query_ids"].tolist() == [0, 1, 0]
    assert task["source_class_ids"].tolist() == [0, 1, 0]
    assert task["latest_stage_index"] == 1
    assert sidecar["provenance"]["source_raw_observation_fingerprint"] == (
        observation_fingerprint(raw)
    )


def test_sidecar_does_not_mutate_p6a_schema_v3() -> None:
    raw = _raw_payload()
    before = copy.deepcopy(raw)
    before_digest = observation_fingerprint(raw)

    build_task_sidecar(
        raw_cache_payload=raw,
        official_prediction=_prediction(),
        protocol_manifest_sha256="5" * 64,
    )

    validate_cache_payload(raw)
    assert raw["schema_version"] == 3
    assert observation_fingerprint(raw) == before_digest
    assert raw.keys() == before.keys()
    for name, value in raw["observation"].items():
        assert torch.equal(value, before["observation"][name])


def test_sidecar_validation_rejects_candidate_shape_drift() -> None:
    sidecar = build_task_sidecar(
        raw_cache_payload=_raw_payload(),
        official_prediction=_prediction(),
        protocol_manifest_sha256="5" * 64,
    )
    sidecar["task_prediction"]["source_query_ids"] = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="candidate"):
        validate_task_sidecar(sidecar)


def test_task_sidecar_atomic_roundtrip_is_content_addressed(tmp_path: Path) -> None:
    sidecar = build_task_sidecar(
        raw_cache_payload=_raw_payload(),
        official_prediction=_prediction(),
        protocol_manifest_sha256="5" * 64,
    )

    entry = write_task_sidecar(tmp_path, sidecar)
    loaded = load_task_sidecar(tmp_path / entry["filename"])

    assert entry["content_sha256"] == task_sidecar_digest(sidecar)
    assert task_sidecar_digest(loaded) == entry["content_sha256"]
    assert entry["filename"] == f"{entry['content_sha256']}.pt"
    assert write_task_sidecar(tmp_path, sidecar) == entry
