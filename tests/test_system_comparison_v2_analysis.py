from __future__ import annotations

import csv

import torch

import scripts.system_comparison_v2_analysis as analysis
from scripts.system_comparison_v2_analysis import build_v2_causal_pair
from scripts.system_comparison_v2_inference import V2TrajectorySnapshot


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_old_row_maps_combines_pooled_and_per_order_aggregates(
    tmp_path, monkeypatch
) -> None:
    methods = ("FullHistory", "Persist4D")
    orders = ("canonical", "reverse", "seed")
    horizons = (2, 3, 4, 5)
    sequence_rows = [
        {
            "method": method,
            "master_sequence_id": f"sequence-{index:03d}",
            "order_id": orders[index % len(orders)],
            "horizon": horizon,
        }
        for method in methods
        for index in range(129)
        for horizon in horizons
    ]
    pooled_rows = [
        {"method": method, "order_id": "all", "horizon": horizon}
        for method in methods
        for horizon in horizons
    ]
    per_order_rows = [
        {"method": method, "order_id": order, "horizon": horizon}
        for method in methods
        for order in orders
        for horizon in horizons
    ]
    _write_csv(tmp_path / "per_sequence_results.csv", sequence_rows)
    _write_csv(tmp_path / "aggregate_results.csv", pooled_rows)
    _write_csv(tmp_path / "per_order_results.csv", per_order_rows)
    monkeypatch.setattr(analysis, "SYSTEM_ROOT", tmp_path)

    _sequence_map, aggregate_map = analysis._old_row_maps()

    assert len(aggregate_map) == 32
    assert ("FullHistory", "canonical", 2) in aggregate_map
    assert ("Persist4D", "all", 5) in aggregate_map


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
