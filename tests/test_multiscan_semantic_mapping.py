from __future__ import annotations

import csv
from pathlib import Path

import pytest

from datasets.multiscan_adapter import MultiScanAdapterError
from scripts.audit_multiscan_dataset import build_multiscan_label_map

OFFICIAL_MAP = Path(
    "/mnt/shared/ww/persist4d-multiscan/official-code/"
    "dataset/benchmark/object_semantic_label_map.csv"
)


def test_official_multiscan_taxonomy_has_eleven_exact_rescene_classes() -> None:
    mapping = build_multiscan_label_map(OFFICIAL_MAP)

    assert mapping["source_class_count"] == 20
    assert mapping["status_counts"] == {
        "exact": 11,
        "defensible": 0,
        "ambiguous": 0,
        "unsupported": 9,
    }
    exact = {
        row["source_class_name"]: row["target_class_id"]
        for row in mapping["mappings"]
        if row["status"] == "exact"
    }
    assert exact == {
        "cabinet": 0,
        "bed": 1,
        "chair": 2,
        "sofa": 3,
        "table": 4,
        "door": 5,
        "window": 6,
        "curtain": 11,
        "refrigerator": 12,
        "toilet": 14,
        "sink": 15,
    }
    unsupported = {
        row["source_class_name"]
        for row in mapping["mappings"]
        if row["status"] == "unsupported"
    }
    assert unsupported == {
        "floor",
        "ceiling",
        "wall",
        "microwave",
        "pillow",
        "tv_monitor",
        "trash_can",
        "suitcase",
        "backpack",
    }
    assert all(
        row["status"] in {"exact", "defensible", "ambiguous", "unsupported"}
        for row in mapping["mappings"]
    )


def test_semantic_map_rejects_one_id_with_conflicting_names(tmp_path: Path) -> None:
    path = tmp_path / "conflict.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["objectName", "objectSemanticName", "objectSemanticId"])
        writer.writerow(["chair", "chair", 6])
        writer.writerow(["seat", "sofa", 6])

    with pytest.raises(MultiScanAdapterError, match="semantic ID"):
        build_multiscan_label_map(path)
