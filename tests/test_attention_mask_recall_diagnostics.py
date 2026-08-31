from __future__ import annotations

import pytest
import torch

from utils.rescene_rootcause_diagnostics import attention_mask_records


def _target() -> dict[str, torch.Tensor]:
    return {
        "ids": torch.tensor([10, 20]),
        "labels": torch.tensor([2, 3]),
        "masks": torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
        "point2segment": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }


def _prediction() -> dict[str, object]:
    segment_logits = torch.tensor(
        [
            [8.0, 7.0, -8.0],
            [8.0, 7.0, -8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, 8.0],
        ]
    )
    class_logits = torch.tensor(
        [
            [4.0, 0.0, -2.0],
            [3.0, 0.0, -2.0],
            [0.0, 4.0, -2.0],
        ]
    )
    layer = {
        "pred_masks": [segment_logits],
        "pred_logits": class_logits.unsqueeze(0),
        "pred_changes": None,
    }
    return {
        "pred_masks": [segment_logits],
        "pred_logits": class_logits.unsqueeze(0),
        "pred_changes": None,
        "aux_outputs": [layer],
    }


def test_attention_mask_recall_binds_gt_and_reset_diagnostics() -> None:
    rows = attention_mask_records(
        file_name="scene000",
        output=_prediction(),
        target=_target(),
        reset_counts=[{"reset_count": 2, "query_count": 3}],
    )

    assert len(rows) == 2
    assert {row["gt_instance_id"] for row in rows} == {10, 20}
    assert all(row["allowed_gt_fraction"] == pytest.approx(1.0) for row in rows)
    assert all(row["masked_gt_fraction"] == pytest.approx(0.0) for row in rows)
    assert all(
        row["post_sample_all_masked_reset_count"] == 2 for row in rows
    )
    assert all(
        row["post_sample_reset_fraction"] == pytest.approx(2 / 3)
        for row in rows
    )
