from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from scripts.system_comparison_inference import (
    ProcessedFullHistory,
    build_full_history_payload,
    full_history_observation_fingerprints,
)
from scripts.system_comparison_v2_cache import task_sidecar_digest
from scripts.system_comparison_v2_parity import (
    T2ParityError,
    compare_t2_task_predictions,
    summarize_t2_rows,
)


def _full_payload() -> dict[str, object]:
    task_masks = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
            [False, True],
        ]
    )
    observation = {
        "features": torch.ones((2, 3)),
        "class_prob": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        "confidence": torch.tensor([0.8, 0.8]),
        "valid": torch.tensor([True, True]),
        "masks": task_masks[1:].T.contiguous(),
        "mask_support": torch.tensor([1, 2]),
        "local_query_ids": torch.tensor([0, 1]),
    }
    processed = ProcessedFullHistory(
        task_prediction={
            "pred_masks": task_masks,
            "pred_scores": torch.tensor([0.9, 0.8]),
            "pred_classes": torch.tensor([10, 11]),
        },
        identity_prediction={
            "pred_masks": task_masks[1:],
            "pred_scores": torch.tensor([0.8, 0.8]),
            "pred_classes": torch.tensor([10, 11]),
            "issued_ids": torch.tensor([0, 1]),
        },
        target={
            "masks": torch.tensor(
                [
                    [True, True, False, False],
                    [False, False, True, True],
                ]
            ),
            "labels": torch.tensor([10, 11]),
            "ids": torch.tensor([101, 202]),
            "changes": torch.tensor([0, 0]),
            "temporal_stages": torch.tensor([0, 1, 1, 1]),
        },
        raw_observation=observation,
        observation_fingerprints=full_history_observation_fingerprints(
            observation
        ),
    )
    return build_full_history_payload(
        key={
            "master_sequence_id": "scene0000_00-scene0000_01",
            "reference_scene_id": "reference-0",
            "order_id": "canonical",
            "context_index": 0,
            "context_scan_indices": [0, 1, 2, 3, 4],
            "horizon": 2,
            "history_scan_ids": ["scene0000_00", "scene0000_01"],
            "scan_indices": [0, 1],
            "task_quality": True,
        },
        provenance={
            "source_commit": "a" * 40,
            "checkpoint_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "protocol_sha256": "d" * 64,
        },
        processed=processed,
        input_stats={
            "scan_count": 2,
            "full_point_count": 4,
            "low_resolution_point_count": 2,
            "segment_count": 2,
            "model_input_bytes": 128,
            "scan_point_counts": [1, 3],
        },
    )


def _sidecar() -> dict[str, object]:
    return {
        "schema_version": 1,
        "key": {
            "master_sequence_id": "scene0000_00-scene0000_01",
            "reference_scene_id": "reference-0",
            "order_id": "canonical",
            "stage_index": 1,
            "history_scan_ids": ["scene0000_00", "scene0000_01"],
            "local_window_scan_ids": ["scene0000_00", "scene0000_01"],
        },
        "provenance": {
            "checkpoint_sha256": "b" * 64,
            "config_hash": "c" * 64,
            "protocol_manifest_hash": "d" * 64,
            "source_raw_observation_fingerprint": "e" * 64,
        },
        "task_prediction": {
            "pred_masks": torch.tensor(
                [[True, False], [False, True], [False, True]]
            ),
            "pred_scores": torch.tensor([0.9, 0.8]),
            "pred_classes": torch.tensor([10, 11]),
            "source_query_ids": torch.tensor([0, 1]),
            "source_class_ids": torch.tensor([10, 11]),
            "latest_stage_index": 1,
        },
    }


def _metric(prediction, target):
    assert prediction["pred_masks"].shape[0] == target["masks"].shape[1]
    return {"raw_local_AP": 0.75}


def test_exact_t2_task_predictions_pass_all_parity_checks() -> None:
    full_payload = _full_payload()
    sidecar = _sidecar()
    row = compare_t2_task_predictions(
        full_payload=full_payload,
        local_sidecar=sidecar,
        full_history_content_sha256=full_payload["content_sha256"],
        sidecar_content_sha256=task_sidecar_digest(sidecar),
        metric_function=_metric,
    )

    assert row["candidate_count_full"] == row["candidate_count_local"] == 2
    assert row["masks_equal"] is True
    assert row["classes_equal"] is True
    assert row["scores_allclose"] is True
    assert row["score_max_abs_diff"] == pytest.approx(0.0)
    assert row["AP_abs_diff"] == pytest.approx(0.0)
    assert row["parity_pass"] is True


def test_score_difference_fails_parity_without_hiding_ap() -> None:
    full_payload = _full_payload()
    sidecar = _sidecar()
    sidecar["task_prediction"]["pred_scores"][0] += 1e-3

    row = compare_t2_task_predictions(
        full_payload=full_payload,
        local_sidecar=sidecar,
        full_history_content_sha256=full_payload["content_sha256"],
        sidecar_content_sha256=task_sidecar_digest(sidecar),
        metric_function=_metric,
    )

    assert row["score_max_abs_diff"] == pytest.approx(1e-3, abs=1e-7)
    assert row["scores_allclose"] is False
    assert row["parity_pass"] is False


def test_t2_key_mismatch_is_rejected() -> None:
    full_payload = _full_payload()
    sidecar = deepcopy(_sidecar())
    sidecar["key"]["history_scan_ids"][1] = "scene9999_00"
    sidecar["key"]["local_window_scan_ids"][1] = "scene9999_00"

    with pytest.raises(T2ParityError, match="keys|history"):
        compare_t2_task_predictions(
            full_payload=full_payload,
            local_sidecar=sidecar,
            full_history_content_sha256=full_payload["content_sha256"],
            sidecar_content_sha256=task_sidecar_digest(sidecar),
            metric_function=_metric,
        )


def test_summary_requires_exact_unique_coverage_and_all_pass() -> None:
    full_payload = _full_payload()
    sidecar = _sidecar()
    row = compare_t2_task_predictions(
        full_payload=full_payload,
        local_sidecar=sidecar,
        full_history_content_sha256=full_payload["content_sha256"],
        sidecar_content_sha256=task_sidecar_digest(sidecar),
        metric_function=_metric,
    )

    summary = summarize_t2_rows([row], expected_unit_count=1)
    assert summary["status"] == "pass"
    assert summary["unit_count"] == 1

    with pytest.raises(T2ParityError, match="coverage"):
        summarize_t2_rows([row], expected_unit_count=2)

    failed = dict(row, parity_pass=False)
    summary = summarize_t2_rows([failed], expected_unit_count=1)
    assert summary["status"] == "fail"
