from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from datasets.multiscan_adapter import (
    MultiScanAdapterError,
    build_multiscan_inventory,
    parse_multiscan_scan_id,
)
from scripts.audit_multiscan_dataset import build_inventory_artifacts

OFFICIAL_CSV = Path(
    "/mnt/shared/ww/persist4d-multiscan/official-code/dataset/benchmark/scans_split.csv"
)
EXPECTED_SCENE_FIELDS = {
    "scene_id",
    "scan_ids",
    "official_split",
    "number_of_scans",
}


def _write_inventory_csv(
    path: Path,
    rows: list[tuple[str, str]],
) -> Path:
    lines = ["scanId,split"]
    lines.extend(f"{scan_id},{split}" for scan_id, split in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_parse_multiscan_scan_id_returns_numeric_scene_and_temporal_index() -> None:
    assert parse_multiscan_scan_id("scene_00104_08") == ("scene_00104", 8)


@pytest.mark.parametrize(
    "scan_id",
    [
        "",
        "scene_00104_8",
        "scene_0104_08",
        "scene_00104_aa",
        "scene_00104_100",
        "scene_00104_08x",
        "scene_00104_08_extra",
        None,
        104,
    ],
)
def test_parse_multiscan_scan_id_rejects_malformed_values(scan_id: object) -> None:
    with pytest.raises(MultiScanAdapterError):
        parse_multiscan_scan_id(scan_id)


def test_build_multiscan_inventory_is_sorted_and_selects_t_at_least_three(
    tmp_path: Path,
) -> None:
    rows = [
        ("scene_00010_02", "train"),
        ("scene_00002_02", "val"),
        ("scene_00001_01", ""),
        ("scene_00002_00", "val"),
        ("scene_00010_00", "train"),
        ("scene_00001_00", ""),
        ("scene_00002_01", "val"),
        ("scene_00010_01", "train"),
    ]
    shuffled_path = _write_inventory_csv(tmp_path / "shuffled.csv", rows)
    ordered_path = _write_inventory_csv(
        tmp_path / "ordered.csv",
        sorted(rows, key=lambda row: parse_multiscan_scan_id(row[0])),
    )

    shuffled = build_multiscan_inventory(shuffled_path)
    ordered = build_multiscan_inventory(ordered_path)

    assert shuffled == ordered
    assert shuffled["scan_count"] == 8
    assert shuffled["scene_count"] == 3
    assert shuffled["split_scan_counts"] == {
        "train": 3,
        "val": 3,
        "test": 0,
        "unassigned": 2,
    }
    assert shuffled["temporal_length_distribution"] == {"2": 1, "3": 2}
    assert shuffled["threshold_scene_counts"] == {"3": 2, "4": 0, "5": 0}
    assert shuffled["selected_rule"] == "all physical scenes with >= 3 scans"
    assert shuffled["selected_scene_count"] == 2
    assert shuffled["selected_scan_count"] == 6
    assert shuffled["selected_scene_ids"] == ["scene_00002", "scene_00010"]
    assert shuffled["selected_scan_ids"] == [
        "scene_00002_00",
        "scene_00002_01",
        "scene_00002_02",
        "scene_00010_00",
        "scene_00010_01",
        "scene_00010_02",
    ]
    assert re.fullmatch(r"[0-9a-f]{64}", shuffled["selected_scene_list_sha256"])
    assert [scene["scene_id"] for scene in shuffled["scenes"]] == [
        "scene_00001",
        "scene_00002",
        "scene_00010",
    ]
    assert shuffled["scenes"][0] == {
        "scene_id": "scene_00001",
        "scan_ids": ["scene_00001_00", "scene_00001_01"],
        "official_split": "unassigned",
        "number_of_scans": 2,
    }
    assert all(set(scene) == EXPECTED_SCENE_FIELDS for scene in shuffled["scenes"])


def test_build_multiscan_inventory_rejects_duplicate_scan_ids(tmp_path: Path) -> None:
    path = _write_inventory_csv(
        tmp_path / "duplicate.csv",
        [
            ("scene_00001_00", "train"),
            ("scene_00001_00", "train"),
        ],
    )

    with pytest.raises(MultiScanAdapterError, match="duplicate"):
        build_multiscan_inventory(path)


def test_build_multiscan_inventory_rejects_mixed_official_splits(
    tmp_path: Path,
) -> None:
    path = _write_inventory_csv(
        tmp_path / "mixed_split.csv",
        [
            ("scene_00001_00", "train"),
            ("scene_00001_01", "val"),
        ],
    )

    with pytest.raises(MultiScanAdapterError, match="split"):
        build_multiscan_inventory(path)


def test_build_multiscan_inventory_requires_official_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing_split.csv"
    path.write_text("scanId\nscene_00001_00\n", encoding="utf-8")

    with pytest.raises(MultiScanAdapterError, match="columns"):
        build_multiscan_inventory(path)


def test_build_multiscan_inventory_rejects_noncontiguous_scan_indices(
    tmp_path: Path,
) -> None:
    path = _write_inventory_csv(
        tmp_path / "noncontiguous.csv",
        [
            ("scene_00001_00", "train"),
            ("scene_00001_02", "train"),
        ],
    )

    with pytest.raises(MultiScanAdapterError, match="contiguous"):
        build_multiscan_inventory(path)


def test_build_multiscan_inventory_rejects_unknown_split(tmp_path: Path) -> None:
    path = _write_inventory_csv(
        tmp_path / "unknown_split.csv",
        [("scene_00001_00", "development")],
    )

    with pytest.raises(MultiScanAdapterError, match="official split"):
        build_multiscan_inventory(path)


def test_official_multiscan_inventory_matches_frozen_counts() -> None:
    inventory = build_multiscan_inventory(OFFICIAL_CSV)

    assert inventory["schema_version"] == 1
    assert inventory["status"] == "pass"
    assert inventory["scan_count"] == 273
    assert inventory["scene_count"] == 117
    assert inventory["split_scan_counts"] == {
        "train": 174,
        "val": 42,
        "test": 41,
        "unassigned": 16,
    }
    assert inventory["temporal_length_distribution"] == {
        "1": 16,
        "2": 78,
        "3": 9,
        "4": 5,
        "5": 6,
        "6": 1,
        "9": 2,
    }
    assert inventory["threshold_scene_counts"] == {"3": 23, "4": 14, "5": 9}
    assert inventory["selected_rule"] == "all physical scenes with >= 3 scans"
    assert inventory["selected_scene_count"] == 23
    assert inventory["selected_scan_count"] == 101
    assert inventory["selected_scene_ids"][0] == "scene_00069"
    assert inventory["selected_scene_ids"][-1] == "scene_00116"
    assert len(inventory["selected_scene_list_sha256"]) == 64
    assert (
        inventory["selected_scene_list_sha256"]
        == inventory["selected_scene_list_sha256"].lower()
    )
    assert re.fullmatch(r"[0-9a-f]{64}", inventory["selected_scene_list_sha256"])
    assert all(set(scene) == EXPECTED_SCENE_FIELDS for scene in inventory["scenes"])


def test_inventory_artifacts_are_deterministic_and_bind_the_frozen_subset(
    tmp_path: Path,
) -> None:
    scans_split = _write_inventory_csv(
        tmp_path / "scans_split.csv",
        [
            ("scene_00002_02", "val"),
            ("scene_00001_00", "train"),
            ("scene_00002_00", "val"),
            ("scene_00002_01", "val"),
        ],
    )
    binding = {
        "schema_version": 1,
        "status": "pass",
        "persist4d": {"commit": "a" * 40},
        "multiscan": {"commit": "b" * 40},
    }
    output = tmp_path / "artifacts"

    paths = build_inventory_artifacts(
        scans_split_path=scans_split,
        output_directory=output,
        reproducibility_binding=binding,
    )

    assert set(paths) == {
        "repro_bindings.json",
        "reproducibility_binding.json",
        "multiscan_inventory.json",
        "longitudinal_subset_manifest.json",
    }
    assert (
        paths["repro_bindings.json"].read_bytes()
        == paths["reproducibility_binding.json"].read_bytes()
    )
    subset = json.loads(paths["longitudinal_subset_manifest.json"].read_text())
    assert subset == {
        "collection_name": "MultiScan Longitudinal Zero-Shot Collection",
        "scene_count": 1,
        "scan_count": 3,
        "schema_version": 1,
        "scenes": [
            {
                "number_of_scans": 3,
                "official_split": "val",
                "scan_ids": [
                    "scene_00002_00",
                    "scene_00002_01",
                    "scene_00002_02",
                ],
                "scene_id": "scene_00002",
            }
        ],
        "selected_rule": "all physical scenes with >= 3 scans",
        "selected_scene_list_sha256": json.loads(
            paths["multiscan_inventory.json"].read_text()
        )["selected_scene_list_sha256"],
        "status": "pass",
    }

    second = build_inventory_artifacts(
        scans_split_path=scans_split,
        output_directory=output,
        reproducibility_binding=binding,
    )
    assert {name: path.read_bytes() for name, path in paths.items()} == {
        name: path.read_bytes() for name, path in second.items()
    }
