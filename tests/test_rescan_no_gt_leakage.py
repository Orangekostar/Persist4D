from __future__ import annotations

import numpy as np
import pytest

from datasets.rescan_adapter import (
    RescanAdapterError,
    RescanPointCloud,
    assert_no_gt_leakage,
    split_inference_and_evaluation,
)


def _cloud() -> RescanPointCloud:
    return RescanPointCloud(
        xyz=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        rgb=np.asarray([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        radius=np.asarray([0.01, 0.01], dtype=np.float32),
        class_ids=np.asarray([5, 5], dtype=np.int32),
        instance_ids=np.asarray([4, 4], dtype=np.int32),
    )


def test_inference_payload_contains_no_object_ground_truth() -> None:
    split = split_inference_and_evaluation(
        _cloud(), scene_id="scene_a", capture_id="scene_a_0"
    )

    assert_no_gt_leakage(split.inference.as_mapping())
    assert set(split.inference.as_mapping()) == {
        "xyz",
        "normals",
        "rgb",
        "geometric_segment_ids",
    }
    np.testing.assert_array_equal(split.inference.geometric_segment_ids, [0, 1])
    np.testing.assert_array_equal(split.target.class_ids, [5, 5])
    np.testing.assert_array_equal(split.target.instance_ids, [4, 4])


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "class_idx",
        "class_ids",
        "instance_idx",
        "instance_ids",
        "stable_identity",
        "ambiguities",
        "object_transform",
        "target_full",
    ],
)
def test_no_gt_leakage_guard_fails_closed(forbidden_key: str) -> None:
    with pytest.raises(RescanAdapterError, match="ground-truth leakage"):
        assert_no_gt_leakage({"xyz": np.zeros((1, 3)), forbidden_key: [1]})
