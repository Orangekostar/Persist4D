from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = ROOT / "artifacts/final_evidence/rescan_to_rescene_label_map.json"


def test_rescan_label_map_covers_the_complete_official_class_range() -> None:
    document = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
    mappings = document["mappings"]

    assert document["source_taxonomy"] == "ReScan NYU40 class_idx"
    assert document["target_taxonomy"] == "frozen ReScene ScanNet18 foreground output"
    assert [entry["source_class_id"] for entry in mappings] == list(range(41))
    assert all(
        entry["status"] in {"exact", "reasonable", "ambiguous", "unsupported"}
        for entry in mappings
    )
    assert all(entry["mapping_evidence"] for entry in mappings)


def test_rescan_label_map_freezes_exact_model_output_indices() -> None:
    document = json.loads(LABEL_MAP.read_text(encoding="utf-8"))
    mappings = {entry["source_class_id"]: entry for entry in document["mappings"]}
    exact_source_ids = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]

    assert [
        mappings[source_id]["target_class_id"] for source_id in exact_source_ids
    ] == list(range(18))
    assert all(
        mappings[source_id]["status"] == "exact" for source_id in exact_source_ids
    )
    assert all(
        mappings[source_id]["status"] == "unsupported"
        for source_id in set(range(41)) - set(exact_source_ids)
    )
    assert all(
        mappings[source_id]["target_class_id"] is None
        for source_id in set(range(41)) - set(exact_source_ids)
    )
