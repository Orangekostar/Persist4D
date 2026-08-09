import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from datasets.semseg import SemanticSegmentationDataset


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _make_three_scan_fixture(tmp_path: Path) -> tuple[Path, np.ndarray]:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    database = []
    scan_names = []
    for scan_index in range(3):
        scan_name = f"scene0000_{scan_index:02d}"
        scan_names.append(scan_name)
        points = np.zeros((2, 12), dtype=np.float32)
        points[:, :3] = np.array(
            [[scan_index, 0.0, 0.0], [scan_index, 1.0, 0.0]],
            dtype=np.float32,
        )
        points[:, 3:6] = [10.0, 20.0, 30.0]
        points[:, 6:9] = [0.0, 0.0, 1.0]
        points[:, 9] = [0, 1]
        points[:, 10] = 1
        points[:, 11] = scan_index + 1

        point_path = processed_dir / f"{scan_name}.npy"
        np.save(point_path, points)
        instance_path = processed_dir / f"{scan_name}.txt"
        instance_path.write_text("0 0\n", encoding="utf-8")
        database.append(
            {
                "filepath": str(point_path),
                "instance_gt_filepath": str(instance_path),
                "file_len": len(points),
            }
        )

    labels = {
        label: {
            "name": f"class-{label}",
            "color": [label, label, label],
            "validation": label != 0,
        }
        for label in range(21)
    }
    change_labels = {
        label: {
            "name": f"change-{label}",
            "color": [label, label, label],
            "validation": True,
        }
        for label in range(6)
    }
    _write_yaml(processed_dir / "validation_database.yaml", database)
    _write_yaml(processed_dir / "label_database.yaml", labels)
    _write_yaml(processed_dir / "change_label_database.yaml", change_labels)
    _write_yaml(
        processed_dir / "color_mean_std.yaml",
        {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    )

    raw_changes = np.array(
        [[10, 100], [11, 101], [12, 102], [13, 103], [14, 104], [15, 105]],
        dtype=np.int64,
    )
    change_path = processed_dir / "changes.txt"
    np.savetxt(change_path, raw_changes, fmt="%d")
    sequence_name = "-".join(scan_names)
    _write_yaml(
        processed_dir / "sequence_database_sliding_3.yaml",
        {
            sequence_name: {
                "scene": 0,
                "sub_scenes": [0, 1, 2],
                "type": "validation",
                "ambiguities": [],
                "nonrigid": [[], []],
                "rigid": [[], []],
                "removed": [[], []],
                "added": [[], []],
                "filepath": str(change_path),
            }
        },
    )
    return processed_dir, raw_changes


def _make_dataset(processed_dir: Path) -> SemanticSegmentationDataset:
    return SemanticSegmentationDataset(
        dataset_name="rio",
        data_dir=str(processed_dir),
        label_db_filepath=str(processed_dir / "label_database.yaml"),
        change_label_db_filepath=str(processed_dir / "change_label_database.yaml"),
        color_mean_std=str(processed_dir / "color_mean_std.yaml"),
        mode="validation",
        add_colors=True,
        add_normals=False,
        add_raw_coordinates=False,
        add_instance=True,
        num_labels=20,
        num_changes=-1,
        filter_out_classes=[0, 1],
        label_offset=2,
        temporal_window=3,
    )


def test_official_loader_projects_three_scan_changes_to_first_transition(
    tmp_path: Path,
) -> None:
    processed_dir, raw_changes = _make_three_scan_fixture(tmp_path)

    coordinates, features, labels, *_ = _make_dataset(processed_dir)[0]

    assert set(coordinates[:, 3].astype(int)) == {0, 1, 2}
    assert coordinates.shape == (6, 4)
    assert features.shape == (6, 3)
    assert labels.shape == (6, 4)
    np.testing.assert_array_equal(labels[:, 2], raw_changes[:, 0])


def test_audit_split_records_loader_shapes_and_projection(tmp_path: Path) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_temporal_loader.py"
    assert script.exists(), "temporal loader audit script has not been implemented"
    audit = importlib.import_module("scripts.audit_temporal_loader")

    result = audit.audit_split(
        processed_dir=processed_dir,
        horizon=3,
        split="validation",
        sample_limit=5,
        seed=45,
    )

    assert result["database_count"] == 1
    assert result["loader_sequence_count"] == 1
    assert result["success_count"] == 1
    assert result["exceptions"] == []
    sample = result["samples"][0]
    assert sample["coordinate_shape"] == [6, 4]
    assert sample["feature_shape"] == [6, 3]
    assert sample["label_shape"] == [6, 4]
    assert sample["temporal_stages"] == [0, 1, 2]
    assert sample["raw_change_shape"] == [6, 2]
    assert sample["raw_change_dim"] == 2
    assert sample["projected_change_dim"] == 1


def test_script_entrypoint_works_outside_project_directory(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_temporal_loader.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Audit the official ReScene4D temporal loader" in result.stdout
    assert "--samples-per-split" in result.stdout
