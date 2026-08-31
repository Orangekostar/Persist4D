from __future__ import annotations

import pytest
import torch

from utils.rescene_rootcause_diagnostics import query_initialization_records


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


def test_query_initialization_reports_scene_and_gt_coverage() -> None:
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
    assert scene["gt_instance_count"] == 2
    assert scene["gt_instance_coverage"] == pytest.approx(1.0)
    assert scene["query_content_zero_fraction"] == pytest.approx(1.0)
    assert [row["query_count"] for row in instances] == [1, 2]
    assert [row["covered_by_fps_query"] for row in instances] == [True, True]
