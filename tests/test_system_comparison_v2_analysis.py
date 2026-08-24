from __future__ import annotations

import torch

from scripts.system_comparison_v2_analysis import build_v2_causal_pair
from scripts.system_comparison_v2_inference import V2TrajectorySnapshot


def test_v2_causal_pair_maps_target_classes_without_remapping_candidates(
    monkeypatch,
) -> None:
    target = {
        "masks": torch.tensor([[True, True, True]]),
        "labels": torch.tensor([5]),
        "ids": torch.tensor([9]),
        "changes": torch.tensor([0]),
        "temporal_stages": torch.tensor([0, 1, 1]),
    }
    monkeypatch.setattr(
        "scripts.system_comparison_v2_analysis.build_temporal_target",
        lambda payloads: target,
    )
    snapshot = V2TrajectorySnapshot(
        prediction={
            "pred_masks": torch.tensor([[True], [True], [True]]),
            "pred_scores": torch.tensor([0.8]),
            "pred_classes": torch.tensor([105]),
        },
        keys=(),
        stage_count=2,
        score_reducer="mean",
    )
    raws = (
        {"key": {"history_scan_ids": ["scene0000_00"]}},
        {
            "key": {
                "history_scan_ids": ["scene0000_00", "scene0000_01"]
            }
        },
    )

    pair = build_v2_causal_pair(
        snapshot=snapshot,
        raw_payloads=raws,
        class_mapper=lambda value: value + 100,
    )

    assert pair.horizon == 2
    assert pair.prediction["pred_classes"].tolist() == [105]
    assert pair.target["labels"].tolist() == [105]
    assert pair.observed_scan_ids == ("scene0000_00", "scene0000_01")
