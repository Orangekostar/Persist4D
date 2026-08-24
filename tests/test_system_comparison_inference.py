from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.system_comparison_inference import (
    FullHistoryCacheError,
    assert_t2_observation_regression,
    build_full_history_payload,
    full_history_cache_keys,
    load_full_history_cache_entry,
    normalize_temporal_stages,
    pack_bool_matrix,
    postprocess_full_history_output,
    unpack_bool_matrix,
    validate_full_history_cache_key,
    validate_full_history_dataset_context,
    validate_full_history_payload,
    write_full_history_cache_entry,
)
from scripts.system_comparison_protocol import build_system_comparison_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "artifacts/P6A/protocol_b_manifest.json"


def _system_manifest() -> dict[str, object]:
    digest = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    return build_system_comparison_manifest(
        PROTOCOL_PATH,
        incumbent_binding={
            "status": "pass",
            "p6a_protocol_manifest_sha256": digest,
        },
    )


def test_bool_matrix_pack_round_trip_handles_empty_dimensions() -> None:
    for value in (
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.empty((4, 0), dtype=torch.bool),
        torch.empty((0, 7), dtype=torch.bool),
    ):
        packed = pack_bool_matrix(value)
        restored = unpack_bool_matrix(packed)
        assert restored.dtype == torch.bool
        assert restored.shape == value.shape
        assert torch.equal(restored, value)


def test_cache_keys_cover_identity_initialization_and_exact_task_prefixes() -> None:
    keys = full_history_cache_keys(_system_manifest())

    assert len(keys) == 43 * 3 * 5
    assert len({json.dumps(row, sort_keys=True) for row in keys}) == len(keys)
    assert sum(row["task_quality"] for row in keys) == 43 * 3 * 4
    for row in keys:
        validated = validate_full_history_cache_key(row)
        assert validated["horizon"] == len(validated["history_scan_ids"])
        assert validated["horizon"] == len(validated["scan_indices"])
        assert validated["task_quality"] is (validated["horizon"] >= 2)


def test_cache_key_rejects_noncausal_history() -> None:
    key = full_history_cache_keys(_system_manifest())[1]
    tampered = dict(key)
    tampered["history_scan_ids"] = [*key["history_scan_ids"], "scene9999_99"]

    with pytest.raises(FullHistoryCacheError, match="horizon|history"):
        validate_full_history_cache_key(tampered)


def test_dataset_context_accepts_real_numpy_sequence_index_container() -> None:
    key = full_history_cache_keys(_system_manifest())[1]
    dataset = SimpleNamespace(
        sequence_names=[key["master_sequence_id"]],
        sequence_indices=np.asarray([key["context_scan_indices"]]),
    )
    adjusted = dict(key)
    adjusted["context_index"] = 0

    validate_full_history_dataset_context(dataset, adjusted)


def test_temporal_stage_normalization_accepts_integer_valued_collator_float() -> None:
    stages = normalize_temporal_stages(torch.tensor([0.0, 1.0, 1.0]))
    assert stages.dtype == torch.long
    assert stages.tolist() == [0, 1, 1]

    with pytest.raises(FullHistoryCacheError, match="integer stage"):
        normalize_temporal_stages(torch.tensor([0.0, 1.5]))


def test_deterministic_runtime_sets_cublas_workspace_before_cuda() -> None:
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


class _FakeSystem:
    model = SimpleNamespace(train_on_segments=True)

    def _get_predictions(self, output):
        return [
            {
                "pred_logits": output["pred_logits"].softmax(dim=-1)[..., :-1],
                "pred_masks": output["pred_masks"],
            }
        ]

    def _get_batch_masks(self, prediction, bid, target_low_res):
        return prediction[0]["pred_masks"][bid][target_low_res[bid]["point2segment"]]

    def _get_mask_and_scores(
        self,
        mask_cls,
        mask_pred,
        num_queries=100,
        num_classes=18,
        device=None,
        return_lineage=False,
    ):
        del mask_cls, num_queries, num_classes, device
        masks = (mask_pred > 0).float()
        result = (
            torch.tensor([0.9, 0.8]),
            masks,
            torch.tensor([0, 1]),
            mask_pred.sigmoid(),
        )
        if return_lineage:
            return (
                *result,
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
            )
        return result

    def _get_full_res_mask(
        self, mask, inverse_map, point2segment_full, is_heatmap=False
    ):
        del point2segment_full, is_heatmap
        return mask[inverse_map]

    def _filter_and_sort_predictions(self, masks, scores, classes, heatmap):
        return classes, masks, scores.numpy(), heatmap


def _processed_fixture():
    output = {
        "pred_logits": torch.tensor([[[4.0, 0.0, -2.0], [0.0, 4.0, -2.0]]]),
        "pred_masks": [
            torch.tensor(
                [
                    [2.0, -2.0],
                    [2.0, -2.0],
                    [-2.0, 2.0],
                ]
            )
        ],
        "query_features": torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    }
    target_low = {
        "point2segment": torch.tensor([0, 1, 2]),
        "temporal_stages": torch.tensor([0, 1, 1]),
    }
    target_full = {
        "point2segment": torch.tensor([0, 1, 2, 3]),
        "temporal_stages": torch.tensor([0, 1, 1, 1]),
        "masks": torch.tensor(
            [[True, False, False, False], [False, True, True, True]]
        ),
        "labels": torch.tensor([0, 1]),
        "ids": torch.tensor([101, 202]),
        "changes": torch.tensor([0, 0]),
    }
    data = SimpleNamespace(inverse_maps=[torch.tensor([0, 1, 1, 2])])
    return postprocess_full_history_output(
        system=_FakeSystem(),
        output=output,
        target_low_resolution=target_low,
        target_full_resolution=target_full,
        data=data,
        horizon=2,
        class_mapper=lambda value: value + 10,
        background_class=2,
        confidence_threshold=0.5,
        mask_threshold=0.5,
        minimum_mask_support=1,
    )


def test_postprocess_keeps_official_task_output_separate_from_raw_query_ids() -> None:
    processed = _processed_fixture()

    assert processed.task_prediction["pred_masks"].shape == (4, 2)
    assert processed.task_prediction["pred_classes"].tolist() == [10, 11]
    assert processed.identity_prediction["issued_ids"].tolist() == [0, 1]
    assert processed.identity_prediction["pred_masks"].shape == (3, 2)
    assert processed.identity_prediction["pred_classes"].tolist() == [10, 11]
    assert processed.target["temporal_stages"].tolist() == [0, 1, 1, 1]
    assert processed.target["changes"].tolist() == [0, 0]
    assert set(processed.observation_fingerprints) == {
        "features",
        "class_prob",
        "confidence",
        "valid",
        "masks",
        "mask_support",
        "local_query_ids",
        "combined",
    }


def _payload() -> dict[str, object]:
    key = full_history_cache_keys(_system_manifest())[1]
    processed = _processed_fixture()
    return build_full_history_payload(
        key=key,
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
            "low_resolution_point_count": 3,
            "segment_count": 3,
            "model_input_bytes": 128,
            "scan_point_counts": [1, 3],
        },
    )


def test_full_history_payload_is_causal_and_self_consistent() -> None:
    payload = validate_full_history_payload(_payload())

    assert unpack_bool_matrix(payload["task_prediction"]["pred_masks"]).shape == (
        4,
        2,
    )
    assert unpack_bool_matrix(payload["identity_prediction"]["pred_masks"]).shape == (
        3,
        2,
    )
    assert unpack_bool_matrix(payload["target"]["masks"]).shape == (2, 4)

    future = copy.deepcopy(payload)
    future["target"]["temporal_stages"][-1] = 2
    with pytest.raises(FullHistoryCacheError, match="future|temporal"):
        validate_full_history_payload(future)


def test_cache_entry_is_atomic_reusable_and_refuses_mismatch(tmp_path: Path) -> None:
    payload = _payload()
    first = write_full_history_cache_entry(tmp_path, payload)
    second = write_full_history_cache_entry(tmp_path, payload)

    assert first == second
    loaded = load_full_history_cache_entry(tmp_path, first)
    assert loaded["content_sha256"] == payload["content_sha256"]
    changed = copy.deepcopy(payload)
    changed["task_prediction"]["pred_scores"][0] += 0.01
    with pytest.raises(FileExistsError, match="different content"):
        write_full_history_cache_entry(tmp_path, changed)
    assert not list(tmp_path.glob(".*.tmp"))


def test_t2_regression_uses_complete_local_observation_fingerprint() -> None:
    payload = _payload()
    processed = _processed_fixture()
    identity_masks = processed.identity_prediction["all_query_masks"].transpose(0, 1)
    local_payload = {
        "key": {
            "stage_index": 1,
            "history_scan_ids": payload["key"]["history_scan_ids"],
            "local_window_scan_ids": payload["key"]["history_scan_ids"],
        },
        "observation": {
            "features": processed.raw_observation["features"],
            "class_prob": processed.raw_observation["class_prob"],
            "confidence": processed.raw_observation["confidence"],
            "valid": processed.raw_observation["valid"],
            "masks": identity_masks,
            "mask_support": processed.raw_observation["mask_support"],
            "local_query_ids": torch.arange(identity_masks.shape[0]),
        },
    }
    assert_t2_observation_regression(payload, local_payload)

    local_payload["observation"]["features"] = (
        local_payload["observation"]["features"].clone()
    )
    local_payload["observation"]["features"][0, 0] += 1
    with pytest.raises(FullHistoryCacheError, match="T2 observation"):
        assert_t2_observation_regression(payload, local_payload)
