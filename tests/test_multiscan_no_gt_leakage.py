from __future__ import annotations

import numpy as np
import pytest

from datasets.multiscan_adapter import (
    MultiScanAdapterError,
    MultiScanEvaluatorTarget,
    MultiScanInferenceInput,
    assert_no_gt_leakage,
)


def test_multiscan_inference_contract_contains_geometry_only_fields() -> None:
    inference = MultiScanInferenceInput(
        xyz=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        normals=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        rgb=np.asarray([[10, 20, 30]], dtype=np.uint8),
        geometric_segment_ids=np.asarray([4], dtype=np.int64),
    )
    target = MultiScanEvaluatorTarget(
        scene_id="scene_00069",
        scan_id="scene_00069_00",
        class_ids=np.asarray([6], dtype=np.int32),
        instance_ids=np.asarray([2], dtype=np.int32),
        stable_object_ids=np.asarray([17], dtype=np.int32),
    )

    assert set(inference.as_mapping()) == {
        "xyz",
        "normals",
        "rgb",
        "geometric_segment_ids",
    }
    assert target.stable_object_ids.tolist() == [17]
    assert "stable_object_ids" not in inference.as_mapping()


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "objectId",
        "object_id",
        "stable_object_ids",
        "instance_ids",
        "instance_gt",
        "class_ids",
        "semantic_gt",
        "partId",
        "mobilityType",
        "gt_obb",
        "gt_correspondence",
    ],
)
def test_multiscan_gt_leakage_guard_fails_closed(forbidden_key: str) -> None:
    with pytest.raises(MultiScanAdapterError, match="ground-truth leakage"):
        assert_no_gt_leakage(
            {
                "xyz": np.zeros((1, 3), dtype=np.float32),
                "nested": {forbidden_key: [1]},
            }
        )
