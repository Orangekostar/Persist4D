from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from datasets.rescan_adapter import (
    RescanAdapterError,
    RescanTemporalDataset,
    discover_rescan_dataset,
    geometric_voxel_segments,
    parse_rescan_ambiguities,
    read_rescan_ply,
    split_inference_and_evaluation,
)

VERTEX_DTYPE = [
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


def _write_ply(
    path: Path,
    *,
    dtype: list[tuple[str, str]] = VERTEX_DTYPE,
    instance_ids: tuple[int, ...] = (4, 9),
) -> None:
    vertices = np.zeros(len(instance_ids), dtype=dtype)
    for index, instance_id in enumerate(instance_ids):
        for field, value in {
            "x": 1.0 + index,
            "y": 2.0 + index,
            "z": 3.0 + index,
            "nx": 0.0,
            "ny": 0.0,
            "nz": 1.0,
            "red": 10 + index,
            "green": 20 + index,
            "blue": 30 + index,
            "radius": 0.01,
            "class_idx": 5,
            "instance_idx": instance_id,
        }.items():
            if field in vertices.dtype.names:
                vertices[field][index] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
    ).write(path)


def test_rescan_ply_parser_preserves_all_official_vertex_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene_a_0.ply"
    _write_ply(path)

    cloud = read_rescan_ply(path)

    np.testing.assert_allclose(cloud.xyz, [[1, 2, 3], [2, 3, 4]])
    np.testing.assert_allclose(cloud.normals, [[0, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(cloud.rgb, [[10, 20, 30], [11, 21, 31]])
    np.testing.assert_allclose(cloud.radius, [0.01, 0.01])
    np.testing.assert_array_equal(cloud.class_ids, [5, 5])
    np.testing.assert_array_equal(cloud.instance_ids, [4, 9])
    assert cloud.xyz.dtype == np.float32
    assert cloud.rgb.dtype == np.uint8
    assert cloud.class_ids.dtype == np.int32


def test_rescan_ply_parser_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.ply"
    _write_ply(
        path,
        dtype=[field for field in VERTEX_DTYPE if field[0] != "instance_idx"],
    )

    with pytest.raises(RescanAdapterError, match="instance_idx"):
        read_rescan_ply(path)


def test_rescan_dataset_order_is_scene_list_then_numeric_capture_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene_list.txt").write_text("scene_b\nscene_a\n", encoding="utf-8")
    for relative in (
        "scene_b/gt_segmentation/scene_b_2.ply",
        "scene_b/gt_segmentation/scene_b_0.ply",
        "scene_b/gt_segmentation/scene_b_1.ply",
        "scene_a/gt_segmentation/scene_a_1.ply",
        "scene_a/gt_segmentation/scene_a_0.ply",
    ):
        _write_ply(tmp_path / relative)

    inventory = discover_rescan_dataset(tmp_path)

    assert [scene.scene_id for scene in inventory.scenes] == ["scene_b", "scene_a"]
    assert [capture.capture_id for capture in inventory.scenes[0].captures] == [
        "scene_b_0",
        "scene_b_1",
        "scene_b_2",
    ]
    assert [capture.temporal_index for capture in inventory.scenes[0].captures] == [
        0,
        1,
        2,
    ]


def test_stable_instance_ids_become_scene_scoped_evaluator_identities(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "scene_a_0.ply"
    second_path = tmp_path / "scene_a_1.ply"
    _write_ply(first_path, instance_ids=(4, 9))
    _write_ply(second_path, instance_ids=(4, 12))

    first = split_inference_and_evaluation(
        read_rescan_ply(first_path), scene_id="scene_a", capture_id="scene_a_0"
    )
    second = split_inference_and_evaluation(
        read_rescan_ply(second_path), scene_id="scene_a", capture_id="scene_a_1"
    )

    assert first.target.identity_keys[0] == ("scene_a", 4)
    assert second.target.identity_keys[0] == ("scene_a", 4)
    assert first.target.identity_keys[1] != second.target.identity_keys[1]


def test_official_ambiguity_alternatives_accept_only_registered_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scene_a_2.txt"
    path.write_text("4 | 4 7\n9 | 9 12\n", encoding="utf-8")

    alternatives = parse_rescan_ambiguities(path)

    assert alternatives.accepts(4, 4)
    assert alternatives.accepts(4, 7)
    assert not alternatives.accepts(4, 8)
    assert alternatives.accepts(3, 3)
    assert not alternatives.accepts(3, 4)


def test_ambiguity_parser_rejects_a_row_without_its_base_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("4 | 7 8\n", encoding="utf-8")

    with pytest.raises(RescanAdapterError, match="base identity"):
        parse_rescan_ambiguities(path)


def test_geometric_voxel_segments_depend_only_on_xyz() -> None:
    xyz = np.asarray(
        [[0.00, 0.00, 0.00], [0.04, 0.04, 0.04], [0.11, 0.00, 0.00]],
        dtype=np.float32,
    )

    segments = geometric_voxel_segments(xyz, voxel_size_m=0.1)

    np.testing.assert_array_equal(segments, [0, 0, 1])
    assert segments.flags.writeable is False


def test_temporal_dataset_builds_gt_free_scene_local_windows(tmp_path: Path) -> None:
    (tmp_path / "scene_list.txt").write_text("scene_a\n", encoding="utf-8")
    for temporal_index in range(3):
        _write_ply(
            tmp_path / "scene_a" / "gt_segmentation" / f"scene_a_{temporal_index}.ply",
            instance_ids=(4, 9),
        )

    dataset = RescanTemporalDataset(tmp_path, geometry_segment_size_m=0.1)
    sample = dataset.load_scan_indices(0, (1, 2), change_file=None)

    coordinates, features, labels, name = sample[:4]
    assert dataset.sequence_names == ("scene_a",)
    assert dataset.sequence_indices == ((0, 1, 2),)
    assert name == "scene_a"
    assert coordinates.shape == (4, 4)
    np.testing.assert_array_equal(coordinates[:, 3], [0, 0, 1, 1])
    assert features.shape == (4, 9)
    assert labels.shape == (4, 4)
    np.testing.assert_array_equal(labels[:, :3], [[2, 0, 0]] * 4)
    assert sample[8] is None
    assert labels[:2, -1].max() < labels[2:, -1].min()

    targets = dataset.evaluator_targets((1, 2))
    assert [target.capture_id for target in targets] == ["scene_a_1", "scene_a_2"]
    np.testing.assert_array_equal(targets[0].instance_ids, [4, 9])


def test_real_official_rescan_sample_when_mounted() -> None:
    import os

    root = os.environ.get("PERSIST4D_RESCAN_ROOT")
    if root is None:
        pytest.skip("official ReScan package is not mounted")
    dataset = RescanTemporalDataset(root, geometry_segment_size_m=0.1)

    sample = dataset.load_scan_indices(0, dataset.sequence_indices[0][:1])

    assert sample[0].shape[0] > 100_000
    assert sample[0].shape[1] == 4
    assert sample[2].shape == (sample[0].shape[0], 4)
    assert sample[2][:, -1].max() < sample[0].shape[0]
