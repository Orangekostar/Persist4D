"""Strict MultiScan metadata and no-GT-leakage adapter contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_SCAN_ID = re.compile(r"(scene_\d{5})_(\d{2})\Z")
_OFFICIAL_SPLITS = frozenset({"train", "val", "test", "unassigned"})
_SELECTED_RULE = "all physical scenes with >= 3 scans"


class MultiScanAdapterError(ValueError):
    """Raised when official MultiScan data violates the frozen contract."""


def parse_multiscan_scan_id(value: object) -> tuple[str, int]:
    """Parse an exact official ``scene_xxxxx_xx`` identifier."""
    if not isinstance(value, str):
        raise MultiScanAdapterError("MultiScan scan ID must be a string")
    match = _SCAN_ID.fullmatch(value)
    if match is None:
        raise MultiScanAdapterError(f"invalid MultiScan scan ID: {value!r}")
    return match.group(1), int(match.group(2))


def _selected_scene_list_sha256(scenes: list[dict[str, object]]) -> str:
    selected = [scene for scene in scenes if int(scene["number_of_scans"]) >= 3]
    payload = (
        json.dumps(
            selected,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_multiscan_inventory(path: str | Path) -> dict[str, object]:
    """Build the official physical-scene inventory and frozen T>=3 subset."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MultiScanAdapterError("MultiScan split CSV must be a regular file")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != {"scanId", "split"}:
            raise MultiScanAdapterError(
                "MultiScan split CSV must contain exact official columns"
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise MultiScanAdapterError("MultiScan split CSV must not be empty")

    seen: set[str] = set()
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    split_counts: Counter[str] = Counter()
    for row in raw_rows:
        scan_id = row.get("scanId")
        scene_id, scan_index = parse_multiscan_scan_id(scan_id)
        if scan_id in seen:
            raise MultiScanAdapterError(f"duplicate MultiScan scan ID: {scan_id}")
        seen.add(scan_id)
        raw_split = row.get("split")
        if not isinstance(raw_split, str):
            raise MultiScanAdapterError("official split must be a string")
        split = raw_split.strip() or "unassigned"
        if split not in _OFFICIAL_SPLITS:
            raise MultiScanAdapterError(f"unknown official split: {split!r}")
        grouped[scene_id].append((scan_index, scan_id, split))
        split_counts[split] += 1

    scenes: list[dict[str, object]] = []
    for scene_id in sorted(grouped):
        rows = sorted(grouped[scene_id])
        indices = [row[0] for row in rows]
        if indices != list(range(len(rows))):
            raise MultiScanAdapterError(
                f"scan indices must be contiguous for {scene_id}"
            )
        splits = {row[2] for row in rows}
        if len(splits) != 1:
            raise MultiScanAdapterError(
                f"physical scene has mixed official split values: {scene_id}"
            )
        scenes.append(
            {
                "scene_id": scene_id,
                "scan_ids": [row[1] for row in rows],
                "official_split": next(iter(splits)),
                "number_of_scans": len(rows),
            }
        )

    length_distribution = Counter(int(scene["number_of_scans"]) for scene in scenes)
    selected_scenes = [scene for scene in scenes if int(scene["number_of_scans"]) >= 3]
    return {
        "schema_version": 1,
        "status": "pass",
        "scan_count": len(seen),
        "scene_count": len(scenes),
        "split_scan_counts": {
            split: split_counts[split]
            for split in ("train", "val", "test", "unassigned")
        },
        "temporal_length_distribution": {
            str(length): count for length, count in sorted(length_distribution.items())
        },
        "threshold_scene_counts": {
            str(threshold): sum(
                int(scene["number_of_scans"]) >= threshold for scene in scenes
            )
            for threshold in (3, 4, 5)
        },
        "selected_rule": _SELECTED_RULE,
        "selected_scene_count": len(selected_scenes),
        "selected_scan_count": sum(
            int(scene["number_of_scans"]) for scene in selected_scenes
        ),
        "selected_scene_ids": [str(scene["scene_id"]) for scene in selected_scenes],
        "selected_scan_ids": [
            str(scan_id) for scene in selected_scenes for scan_id in scene["scan_ids"]
        ],
        "selected_scene_list_sha256": _selected_scene_list_sha256(scenes),
        "scenes": scenes,
    }


__all__ = [
    "MultiScanAdapterError",
    "build_multiscan_inventory",
    "parse_multiscan_scan_id",
]
