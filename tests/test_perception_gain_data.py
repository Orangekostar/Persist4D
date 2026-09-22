from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch


def _data_module():
    try:
        return importlib.import_module("scripts.perception_gain_data")
    except ModuleNotFoundError as error:
        pytest.fail(f"perception data module is unavailable: {error}")


def test_data_roles_preserve_frozen_evaluation_and_remove_train_overlap() -> None:
    data = _data_module()
    task_contract = {
        "roles": {
            "adaptation_reference_ids": ["train-b", "shared", "train-a"],
            "additional_native_reference_ids": ["additional"],
            "protocol_b_reference_ids": ["pb", "shared"],
        }
    }
    crosswindow_roles = {
        "roles": {
            "DEV-CAL": ["cal"],
            "DEV-SEL": ["sel"],
            "additional_native_refs": ["additional"],
        }
    }
    roles = data.build_data_roles(
        task_contract,
        crosswindow_roles,
        local_t2_reference_ids=("local",),
    )
    assert roles == {
        "TRAIN": ["train-a", "train-b"],
        "CAL": ["cal"],
        "SEL": ["sel"],
        "PB": ["pb", "shared"],
        "LOCAL-T2": ["local"],
        "ADDITIONAL": ["additional"],
        "removed_train_overlap": ["shared"],
    }
    data.validate_role_disjointness(roles)

    roles["TRAIN"].append("cal")
    with pytest.raises(data.PerceptionDataError, match="TRAIN overlaps evaluation"):
        data.validate_role_disjointness(roles)


def test_scorer_inventory_labels_and_sampling_are_reference_first() -> None:
    data = _data_module()
    pairs = [
        {"reference_id": reference, "pair_id": f"{reference}-p{index}"}
        for reference in ("r0", "r1", "r2")
        for index in range(3)
    ]

    inventory = data.select_scorer_pairs(pairs, limit=5, minimum_references=3)

    assert inventory["status"] == "PASS"
    assert inventory["reference_count"] == 3
    assert len(inventory["pairs"]) == 5
    assert len({row["reference_id"] for row in inventory["pairs"][:3]}) == 3
    labels = data.segment_foreground_targets(
        point2segment=torch.tensor([0, 0, 0, 1, 1, 2]),
        semantic_labels=torch.tensor([3, 3, 1, 1, 255, 255]),
        thing_class_ids=(3,),
        stuff_class_ids=(1,),
        ignore_class_ids=(255,),
    )
    torch.testing.assert_close(labels["targets"], torch.tensor([2 / 3, 0.0, 0.0]))
    assert labels["valid"].tolist() == [True, True, False]
    assert labels["valid_point_counts"].tolist() == [3, 1, 0]
    segments = {"r0": ("a", "b"), "r1": ("c",), "r2": ("d", "e")}
    first = data.sample_reference_segments(segments, count=9, seed=45, cursor=0)
    second = data.sample_reference_segments(segments, count=9, seed=45, cursor=0)
    assert first == second
    assert {reference for reference, _ in first[:3]} == set(segments)


def test_refiner_inventory_and_soft_shards_enforce_real_coverage_and_no_gt(
    tmp_path: Path,
) -> None:
    data = _data_module()
    selected = data.select_refiner_references(
        {
            "r0": ("a", "b", "c"),
            "r1": ("d", "e"),
            "r2": ("f", "g", "h", "i"),
        },
        minimum_references=2,
    )
    assert selected == {
        "references": selected["references"],
        "reference_count": 2,
        "status": "PASS",
    }
    assert set(selected["references"]) == {"r0", "r2"}

    payload = {
        "reference_id": "r0",
        "soft": {"segment_logits": torch.tensor([0.0, 1.0])},
    }
    written = data.write_soft_sidecar_shard(
        tmp_path,
        shard_id=0,
        records=[payload],
        maximum_cache_bytes=1024 * 1024,
    )
    assert written["bytes"] == data.sidecar_cache_bytes(tmp_path)
    assert len(written["sha256"]) == 64
    with pytest.raises(data.PerceptionDataError, match="GT field"):
        data.write_soft_sidecar_shard(
            tmp_path,
            shard_id=1,
            records=[{"soft": payload, "gt_masks": torch.ones(1)}],
            maximum_cache_bytes=1024 * 1024,
        )
    with pytest.raises(data.PerceptionDataError, match="cache limit"):
        data.write_soft_sidecar_shard(
            tmp_path,
            shard_id=1,
            records=[payload],
            maximum_cache_bytes=written["bytes"],
        )


def test_scorer_record_aligns_features_coordinates_stages_and_soft_targets() -> None:
    data = _data_module()
    record = data.build_scorer_record(
        reference_id="ref-a",
        pair_id="scene0001_00-scene0001_01",
        segment_features=torch.arange(3 * 128, dtype=torch.float32).reshape(3, 128),
        point2segment=torch.tensor([0, 0, 1, 1, 2]),
        semantic_labels=torch.tensor([2, 2, 0, 2, 255]),
        raw_coordinates=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 1.0],
                [0.0, 4.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
        temporal_stages=torch.tensor([0, 0, 1, 1, 1]),
        thing_class_ids=tuple(range(2, 20)),
        stuff_class_ids=(0, 1),
        ignore_class_ids=(255,),
    )

    assert set(record) == {
        "pair_id",
        "reference_id",
        "segment_coordinates",
        "segment_features",
        "segment_stage_ids",
        "targets",
        "valid",
    }
    torch.testing.assert_close(
        record["segment_coordinates"],
        torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        ),
    )
    assert record["segment_stage_ids"].tolist() == [0, 1, 1]
    torch.testing.assert_close(record["targets"], torch.tensor([1.0, 0.5, 0.0]))
    assert record["valid"].tolist() == [True, True, False]
