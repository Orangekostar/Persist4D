from __future__ import annotations

import pytest
import torch

from utils.rescene_rootcause_diagnostic_runtime import RootCauseDiagnosticCollector
from utils.rescene_rootcause_diagnostics import (
    attention_mask_records,
    query_conflict_records,
    query_initialization_records,
    superpoint_feature_records,
)
from utils.rescene_rootcause_evaluation import RootCauseEvaluationError


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
        "segment_mask": torch.tensor(
            [[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.bool
        ),
        "point2segment": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
    }


def _prediction() -> dict[str, list[torch.Tensor] | torch.Tensor | None]:
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


def test_query_initialization_reports_scene_and_instance_coverage() -> None:
    rows = query_initialization_records(
        file_name="scene000",
        sampled_indices=torch.tensor([0, 3, 4, 7]),
        query_content_norms=torch.zeros(4),
        target=_target(),
    )

    scene = rows[0]
    instances = rows[1:]
    assert scene["record_type"] == "scene_summary"
    assert scene["foreground_query_fraction"] == pytest.approx(0.75)
    assert scene["background_query_fraction"] == pytest.approx(0.25)
    assert scene["gt_instance_coverage"] == pytest.approx(1.0)
    assert scene["query_content_zero_fraction"] == pytest.approx(1.0)
    assert [row["query_count"] for row in instances] == [1, 2]
    assert [row["size_points"] for row in instances] == [3, 4]


def test_query_conflicts_report_redundancy_at_every_prediction_layer() -> None:
    rows = query_conflict_records(
        file_name="scene000", output=_prediction(), target=_target()
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["decoder_prediction_layer"] == 0
    assert first["feeds_next_attention"] is True
    assert first["gt_coverage_iou25"] == pytest.approx(1.0)
    assert first["gt_coverage_iou50"] == pytest.approx(1.0)
    assert first["mean_queries_per_gt_iou25"] == pytest.approx(1.5)
    assert first["competed_active_query_fraction"] == pytest.approx(1 / 3)
    assert first["distinct_gt_covered_iou25"] == 2
    assert rows[-1]["feeds_next_attention"] is False


def test_attention_mask_records_bind_matches_and_real_reset_counts() -> None:
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
    assert all(row["post_sample_all_masked_reset_count"] == 2 for row in rows)
    assert all(row["post_sample_reset_fraction"] == pytest.approx(2 / 3) for row in rows)


def test_superpoint_features_report_variance_margin_and_purity() -> None:
    features = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
    )
    rows = superpoint_feature_records(
        file_name="scene000",
        segment_features=features,
        target=_target(),
    )

    assert len(rows) == 2
    assert all(row["segments_per_gt"] == 2 for row in rows)
    assert [row["mean_segment_purity"] for row in rows] == pytest.approx([0.75, 1.0])
    assert all(row["mean_gt_instances_per_segment"] == pytest.approx(1.0) for row in rows)
    assert all(row["nearest_instance_cosine_margin"] > 0.8 for row in rows)
    assert all(row["within_instance_feature_variance"] < 0.01 for row in rows)


def test_decoder_diagnostics_reject_tensor_contract_mismatch() -> None:
    target = _target()
    target["point2segment"] = torch.tensor([0, 1])
    with pytest.raises(RootCauseEvaluationError, match="diagnostic tensor"):
        query_conflict_records(
            file_name="scene000", output=_prediction(), target=target
        )


class _FakeDiagnosticModel(torch.nn.Module):
    def initialize_queries(self, *, pcd_features, coords):
        queries = torch.zeros(1, 3, 2)
        sampled = coords[-1][0][torch.tensor([0, 3, 7])].unsqueeze(0)
        return queries, torch.zeros_like(queries), sampled

    def sample_and_batch_features(self, *args, **kwargs):
        attention = torch.tensor(
            [[[True, False, False], [True, False, True]]], dtype=torch.bool
        )
        return torch.zeros(1), attention, torch.zeros(1), torch.zeros(1)

    def forward(self, data, point2segment=None, raw_coordinates=None, is_eval=False):
        coordinates = [torch.arange(8).float().unsqueeze(1)]
        self.initialize_queries(pcd_features=object(), coords=[coordinates])
        self.sample_and_batch_features(
            torch.zeros(1), extra=[torch.zeros(1), torch.zeros(1)]
        )
        output = _prediction()
        output["segment_features"] = [
            [torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])]
        ]
        return output


class _FakeDiagnosticSystem(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeDiagnosticModel()

    def forward(self, data, point2segment=None, raw_coordinates=None, targets=None):
        return self.model(data, point2segment, raw_coordinates)

    def validation_step(self, batch, batch_index):
        data, targets, _ = batch
        return self(data, targets=targets)


@pytest.mark.parametrize(
    "mode",
    (
        "query_initialization",
        "query_conflicts",
        "attention_mask_recall",
        "superpoint_features",
    ),
)
def test_diagnostic_collector_uses_real_module_hook_path(mode: str) -> None:
    system = _FakeDiagnosticSystem()
    collector = RootCauseDiagnosticCollector(mode)
    collector.install(system)

    system.validation_step((object(), [_target()], ["scene000"]), 0)

    assert collector.sequence_count == 1
    assert collector.rows
    assert {row["file_name"] for row in collector.rows} == {"scene000"}
