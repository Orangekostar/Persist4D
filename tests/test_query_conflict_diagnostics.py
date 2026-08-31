from __future__ import annotations

import pytest
import torch

from utils.rescene_rootcause_diagnostic_runtime import RootCauseDiagnosticCollector
from utils.rescene_rootcause_diagnostics import query_conflict_records


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


def test_query_conflicts_cover_every_prediction_layer() -> None:
    rows = query_conflict_records(
        file_name="scene000", output=_prediction(), target=_target()
    )

    assert len(rows) == 2
    assert [row["decoder_prediction_layer"] for row in rows] == [0, 1]
    assert [row["feeds_next_attention"] for row in rows] == [True, False]
    for row in rows:
        assert row["gt_coverage_iou25"] == pytest.approx(1.0)
        assert row["gt_coverage_iou50"] == pytest.approx(1.0)
        assert row["mean_queries_per_gt_iou25"] == pytest.approx(1.5)
        assert row["competed_active_query_fraction"] == pytest.approx(1 / 3)
        assert row["distinct_gt_covered_iou25"] == 2
    assert rows[-1]["feeds_next_attention"] is False


def test_diagnostic_collector_captures_direct_system_forward() -> None:
    class System(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Identity()

        def forward(self, *, targets):
            return _prediction()

        def validation_step(self, batch, batch_idx):
            return self.forward(targets=batch[1])

    system = System()
    collector = RootCauseDiagnosticCollector("query_conflicts")
    collector.install(system)

    batch = (None, [_target()], ["scene000"])
    system.validation_step(batch, 0)

    assert collector.sequence_count == 1
    assert len(collector.rows) == 2
