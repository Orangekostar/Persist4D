from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scripts.audit_rescan_dataset import (
    RescanDatasetAuditError,
    build_rescan_dataset_manifest,
    write_rescan_dataset_manifest,
)


def _write_capture(
    root: Path,
    scene_id: str,
    temporal_index: int,
    instance_ids: tuple[int, ...],
) -> None:
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("radius", "f4"),
        ("class_idx", "i4"),
        ("instance_idx", "i4"),
    ]
    vertices = np.zeros(len(instance_ids), dtype=dtype)
    vertices["x"] = np.arange(len(instance_ids), dtype=np.float32)
    vertices["nz"] = 1.0
    vertices["red"] = 10
    vertices["green"] = 20
    vertices["blue"] = 30
    vertices["radius"] = 0.01
    vertices["class_idx"] = 5
    vertices["instance_idx"] = instance_ids
    capture_id = f"{scene_id}_{temporal_index}"
    segmentation = root / scene_id / "gt_segmentation"
    segmentation.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
    ).write(segmentation / f"{capture_id}.ply")
    for modality, suffix in (("color", ".h264"), ("depth", ".depth")):
        directory = root / scene_id / modality
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{capture_id}{suffix}").write_bytes(
            f"{modality}-{capture_id}".encode("ascii")
        )


def test_dataset_manifest_hashes_files_and_counts_natural_gap_events(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene_list.txt").write_text("scene_a\n", encoding="utf-8")
    (tmp_path / "nyu40_classes.txt").write_text("chair 5\n", encoding="utf-8")
    _write_capture(tmp_path, "scene_a", 0, (4, 5))
    _write_capture(tmp_path, "scene_a", 1, (5,))
    _write_capture(tmp_path, "scene_a", 2, (4, 5))
    (tmp_path / "scene_a/gt_segmentation/scene_a_2.txt").write_text(
        "4 | 4 7\n", encoding="utf-8"
    )

    manifest = build_rescan_dataset_manifest(tmp_path)

    assert manifest["status"] == "pass"
    assert manifest["summary"] == {
        "ambiguity_file_count": 1,
        "capture_count": 3,
        "encountered_class_ids": [5],
        "file_count": 12,
        "gap_opportunity_count": 1,
        "scene_count": 1,
        "stable_identity_count": 2,
        "stable_object_identity_count": 2,
        "object_identity_excluded_source_class_ids": [1, 2, 22],
        "semantic_inconsistent_identity_count": 0,
    }
    assert manifest["chronology"]["status"] == "official_index_order"
    assert manifest["scenes"][0]["capture_ids"] == [
        "scene_a_0",
        "scene_a_1",
        "scene_a_2",
    ]
    assert manifest["scenes"][0]["gap_opportunities"] == [
        {
            "absent_capture_ids": ["scene_a_1"],
            "identity": 4,
            "left_capture_id": "scene_a_0",
            "right_capture_id": "scene_a_2",
        }
    ]
    assert len(manifest["dataset_content_sha256"]) == 64
    assert all(len(record["sha256"]) == 64 for record in manifest["files"])
    assert "/home/" not in json.dumps(manifest, sort_keys=True)


def test_dataset_manifest_requires_matching_color_and_depth_captures(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene_list.txt").write_text("scene_a\n", encoding="utf-8")
    (tmp_path / "nyu40_classes.txt").write_text("chair 5\n", encoding="utf-8")
    _write_capture(tmp_path, "scene_a", 0, (4,))
    (tmp_path / "scene_a/depth/scene_a_0.depth").unlink()

    with pytest.raises(RescanDatasetAuditError, match="depth"):
        build_rescan_dataset_manifest(tmp_path)


def test_dataset_manifest_excludes_instances_outside_official_0_to_255_range(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene_list.txt").write_text("scene_a\n", encoding="utf-8")
    (tmp_path / "nyu40_classes.txt").write_text("chair 5\n", encoding="utf-8")
    _write_capture(tmp_path, "scene_a", 0, (4, 256, 1024))

    manifest = build_rescan_dataset_manifest(tmp_path)

    assert manifest["scenes"][0]["stable_identity_ids"] == [4]
    assert manifest["scenes"][0]["stable_object_identity_ids"] == [4]


def test_dataset_manifest_publication_is_canonical_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "manifest.json"
    payload = {"schema_version": 1, "status": "pass"}

    write_rescan_dataset_manifest(output, payload)
    first = output.read_bytes()
    write_rescan_dataset_manifest(output, payload)

    assert output.read_bytes() == first
    assert first.endswith(b"\n")
    with pytest.raises(FileExistsError, match="different"):
        write_rescan_dataset_manifest(output, {**payload, "status": "changed"})
