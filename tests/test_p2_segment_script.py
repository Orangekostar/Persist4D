import errno
from pathlib import Path

from datasets.preprocessing import segment_script


def test_run_segmentation_moves_output_across_filesystems(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ply_path = tmp_path / "scene0707_00_vh_clean_2.ply"
    ply_path.write_bytes(b"ply")
    output_path = tmp_path / "external" / "scene0707_00_vh_clean_2.0.010000.segs.json"

    def fake_run(command, *, check):
        assert check is True
        source = Path(command[1]).with_suffix(".0.010000.segs.json")
        source.write_text('{"segIndices": [0]}\n', encoding="utf-8")

    def raise_exdev(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(segment_script.subprocess, "run", fake_run)
    monkeypatch.setattr(segment_script.shutil.os, "rename", raise_exdev)

    assert segment_script.run_segmentation(ply_path, output_path) is True
    assert output_path.read_text(encoding="utf-8") == '{"segIndices": [0]}\n'
    assert not ply_path.with_suffix(".0.010000.segs.json").exists()


def test_process_scannet_test_can_rerun_all_100_test_scenes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "scannet"
    save_dir = tmp_path / "segments"
    git_repo = tmp_path / "ScanNet"
    benchmark = git_repo / "Tasks" / "Benchmark"
    benchmark.mkdir(parents=True)
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    (benchmark / "scannetv2_test.txt").write_text(
        "\n".join(scenes) + "\n", encoding="utf-8"
    )
    for scene in scenes:
        mesh = data_dir / "test" / scene / f"{scene}_vh_clean_2.ply"
        mesh.parent.mkdir(parents=True)
        mesh.write_bytes(b"ply")

    def fake_run_segmentation(ply_path, output_path):
        del ply_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}\n", encoding="utf-8")
        return True

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        segment_script,
        "run_segmentation",
        fake_run_segmentation,
    )

    segment_script.process_scannet_test(data_dir, save_dir, git_repo)
    segment_script.process_scannet_test(data_dir, save_dir, git_repo)

    assert len(list(save_dir.glob("*.segs.json"))) == 100
    assert not any((tmp_path / "temp_ply").iterdir())
