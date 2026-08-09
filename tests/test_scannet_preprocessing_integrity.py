from pathlib import Path

import numpy as np
import yaml

from datasets.preprocessing.scannet_preprocessing import ScannetPreprocessing


def test_known_label_bug_filter_keeps_npy_gt_and_database_in_sync(
    tmp_path: Path,
) -> None:
    save_dir = tmp_path / "scannet"
    database = []
    for scene, sub_scene, wrong_instance in (
        (270, 0, 50),
        (270, 2, 50),
        (384, 0, 149),
    ):
        npy_path = save_dir / "train" / f"{scene:04d}_{sub_scene:02d}.npy"
        gt_path = (
            save_dir
            / "instance_gt"
            / "train"
            / f"scene{scene:04d}_{sub_scene:02d}.txt"
        )
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        points = np.zeros((3, 12), dtype=np.float32)
        points[:, 3:6] = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
        points[:, 10] = 3
        points[:, 11] = [0, wrong_instance, 1]
        np.save(npy_path, points)
        gt_path.write_text(
            f"3001\n{3000 + wrong_instance + 1}\n3002\n",
            encoding="utf-8",
        )
        database.append(
            {
                "scene": scene,
                "sub_scene": sub_scene,
                "filepath": str(npy_path),
                "instance_gt_filepath": str(gt_path),
                "file_len": 3,
                "color_mean": [0.0, 0.0, 0.0],
                "color_std": [0.0, 0.0, 0.0],
            }
        )
    database_path = save_dir / "train_database.yaml"
    database_path.write_text(
        yaml.safe_dump(database, sort_keys=False),
        encoding="utf-8",
    )
    processor = ScannetPreprocessing.__new__(ScannetPreprocessing)
    processor.save_dir = save_dir
    processor.scannet200 = False

    processor.fix_bugs_in_labels()

    updated_database = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    for record in updated_database:
        points = np.load(record["filepath"])
        gt = np.loadtxt(record["instance_gt_filepath"], dtype=np.int64, ndmin=1)
        assert points.shape == (2, 12)
        assert gt.tolist() == [3001, 3002]
        assert record["file_len"] == 2
        expected_colors = points[:, 3:6] / 255.0
        assert np.allclose(record["color_mean"], expected_colors.mean(axis=0))
        assert np.allclose(
            record["color_std"],
            np.square(expected_colors).mean(axis=0),
        )
