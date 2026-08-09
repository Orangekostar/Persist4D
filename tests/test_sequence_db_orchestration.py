import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import build_rscan_sequence_dbs as orchestration


def test_script_entrypoint_imports_project_modules() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_rscan_sequence_dbs.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Build and inventory official 3RScan" in result.stdout


def _entry(
    *,
    scene: int,
    sub_scenes: list[int],
    split: str,
    filepath: str,
) -> dict[str, object]:
    transitions = len(sub_scenes) - 1
    return {
        "scene": scene,
        "sub_scenes": sub_scenes,
        "type": split,
        "ambiguities": [],
        "nonrigid": [[] for _ in range(transitions)],
        "rigid": [[] for _ in range(transitions)],
        "removed": [[] for _ in range(transitions)],
        "added": [[] for _ in range(transitions)],
        "filepath": filepath,
    }


def _key(scene: int, sub_scenes: list[int]) -> str:
    return "-".join(f"scene{scene:04d}_{sub_scene:02d}" for sub_scene in sub_scenes)


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_build_one_delegates_to_official_builder(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    expected = tmp_path / "sequence_database_sliding_3.yaml"

    def fake_main(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(expected)

    monkeypatch.setattr(orchestration.official_sequence_builder, "main", fake_main)

    result = orchestration.build_one(
        data_dir=tmp_path / "raw",
        metadata=tmp_path / "3RScan.json",
        processed_dir=tmp_path,
        horizon=3,
        sequence_type="sliding",
        scannet200=False,
    )

    assert result == expected
    assert calls == [
        {
            "data_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path),
            "metadata_file": str(tmp_path / "3RScan.json"),
            "sequence_type": "sliding",
            "sequence_length": 3,
            "scannet200": False,
        }
    ]


def test_build_all_delegates_every_horizon_and_records_inventory(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[int] = []
    metadata = tmp_path / "3RScan.json"
    metadata.write_text("[]\n", encoding="utf-8")
    change_file = tmp_path / "change.txt"
    change_file.write_text("0\n", encoding="utf-8")

    def fake_build_one(**kwargs: object) -> Path:
        horizon = int(kwargs["horizon"])
        calls.append(horizon)
        path = tmp_path / f"sequence_database_sliding_{horizon}.yaml"
        sub_scenes = list(range(horizon))
        _write_yaml(
            path,
            {
                _key(1, sub_scenes): _entry(
                    scene=1,
                    sub_scenes=sub_scenes,
                    split="train",
                    filepath=str(change_file),
                )
            },
        )
        return path

    monkeypatch.setattr(orchestration, "build_one", fake_build_one)

    manifest = orchestration.build_all(
        data_dir=tmp_path / "raw",
        metadata=metadata,
        processed_dir=tmp_path,
        horizons=[2, 3, 4, 5],
        source_commit="fb2fe42",
    )

    assert calls == [2, 3, 4, 5]
    assert manifest["official_seed"] == 45
    assert manifest["scan_order_semantics"] == "metadata_order_only_no_timestamps"
    assert [item["sequence_length"] for item in manifest["databases"]] == [2, 3, 4, 5]
    assert all(item["sequence_count"] == 1 for item in manifest["databases"])
    assert all(item["count_by_split"] == {"test": 0, "train": 1, "validation": 0} for item in manifest["databases"])


def test_build_all_keeps_runtime_paths_but_emits_portable_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_root = tmp_path / "private-home" / "paper5"
    processed_dir = project_root / "data" / "processed" / "rio"
    dataset_root = tmp_path / "private-datasets" / "3RScan"
    data_dir = dataset_root / "scans"
    metadata = dataset_root / "3RScan.json"
    processed_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    metadata.write_text("[]\n", encoding="utf-8")
    change_file = processed_dir / "changes.txt"
    change_file.write_text("0\n", encoding="utf-8")
    runtime_calls: list[dict[str, object]] = []

    def fake_build_one(**kwargs: object) -> Path:
        runtime_calls.append(kwargs)
        horizon = int(kwargs["horizon"])
        path = processed_dir / f"sequence_database_sliding_{horizon}.yaml"
        sub_scenes = list(range(horizon))
        _write_yaml(
            path,
            {
                _key(1, sub_scenes): _entry(
                    scene=1,
                    sub_scenes=sub_scenes,
                    split="train",
                    filepath=str(change_file),
                )
            },
        )
        return path

    monkeypatch.setattr(orchestration, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(orchestration, "build_one", fake_build_one)

    manifest = orchestration.build_all(
        data_dir=data_dir,
        metadata=metadata,
        processed_dir=processed_dir,
        horizons=[2],
    )

    assert runtime_calls[0]["data_dir"] == data_dir
    assert runtime_calls[0]["metadata"] == metadata
    assert runtime_calls[0]["processed_dir"] == processed_dir
    assert manifest["data_dir"] == "external:3RScan/scans"
    assert manifest["metadata"]["path"] == "external:3RScan/3RScan.json"
    assert manifest["processed_dir"] == "repo:data/processed/rio"
    assert manifest["databases"][0]["path"] == (
        "repo:data/processed/rio/sequence_database_sliding_2.yaml"
    )
    assert str(tmp_path) not in json.dumps(manifest)


def test_checked_in_manifest_contains_no_personal_absolute_paths() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "artifacts" / "data_audit" / "sequence_db_manifest.json"

    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    assert manifest["data_dir"] == "external:3RScan/scans"
    assert manifest["metadata"]["path"] == "external:3RScan/3RScan.json"
    assert manifest["processed_dir"] == "repo:data/processed/rio"
    assert all(
        database["path"].startswith("repo:data/processed/rio/")
        for database in manifest["databases"]
    )
    assert "/home/" not in manifest_text
    assert "/Users/" not in manifest_text


def test_inventory_counts_test_path_limitations(tmp_path: Path) -> None:
    existing_train = tmp_path / "train.txt"
    existing_validation = tmp_path / "validation.txt"
    existing_train.touch()
    existing_validation.touch()
    database = {
        _key(1, [0, 1]): _entry(
            scene=1,
            sub_scenes=[0, 1],
            split="train",
            filepath=str(existing_train),
        ),
        _key(2, [0, 1]): _entry(
            scene=2,
            sub_scenes=[0, 1],
            split="validation",
            filepath=str(existing_validation),
        ),
        _key(3, [0, 1]): _entry(
            scene=3,
            sub_scenes=[0, 1],
            split="test",
            filepath="None",
        ),
        _key(4, [0, 1]): _entry(
            scene=4,
            sub_scenes=[0, 1],
            split="test",
            filepath=str(tmp_path / "missing-test.txt"),
        ),
    }
    path = tmp_path / "sequence_database_sliding_2.yaml"
    _write_yaml(path, database)

    inventory = orchestration.inventory_database(path, horizon=2)

    assert inventory["count_by_split"] == {"test": 2, "train": 1, "validation": 1}
    assert inventory["unresolved_filepath_count_by_split"] == {
        "test": 1,
        "train": 0,
        "validation": 0,
    }
    assert inventory["nonexistent_filepath_count_by_split"] == {
        "test": 1,
        "train": 0,
        "validation": 0,
    }
    assert inventory["unique_reference_scene_count"] == 4


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "top level must be a mapping"),
        ({}, "database is empty"),
        (
            {
                _key(1, [0, 1, 0]): _entry(
                    scene=1,
                    sub_scenes=[0, 1, 0],
                    split="train",
                    filepath="unused",
                )
            },
            "repeated scan ID",
        ),
        (
            {
                _key(1, [0, 1]): _entry(
                    scene=1,
                    sub_scenes=[0, 1],
                    split="train",
                    filepath="unused",
                )
            },
            "expected 3 scan IDs",
        ),
    ],
)
def test_inventory_rejects_invalid_database(
    tmp_path: Path, payload: object, message: str
) -> None:
    path = tmp_path / "database.yaml"
    _write_yaml(path, payload)

    with pytest.raises(ValueError, match=message):
        orchestration.inventory_database(path, horizon=3)


def test_inventory_rejects_missing_required_field(tmp_path: Path) -> None:
    change_file = tmp_path / "change.txt"
    change_file.touch()
    entry = _entry(
        scene=1,
        sub_scenes=[0, 1],
        split="train",
        filepath=str(change_file),
    )
    del entry["removed"]
    path = tmp_path / "database.yaml"
    _write_yaml(path, {_key(1, [0, 1]): entry})

    with pytest.raises(ValueError, match="missing required fields: removed"):
        orchestration.inventory_database(path, horizon=2)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_inventory_requires_resolved_existing_supervised_paths(
    tmp_path: Path, split: str
) -> None:
    path = tmp_path / "database.yaml"
    _write_yaml(
        path,
        {
            _key(1, [0, 1]): _entry(
                scene=1,
                sub_scenes=[0, 1],
                split=split,
                filepath="None",
            )
        },
    )

    with pytest.raises(ValueError, match=f"{split} sequence has unresolved filepath"):
        orchestration.inventory_database(path, horizon=2)


def test_manifest_serialization_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = {"z": 1, "a": {"y": 2, "b": 3}}

    orchestration.write_manifest(path, manifest)
    first = path.read_bytes()
    orchestration.write_manifest(path, manifest)

    assert path.read_bytes() == first
    assert json.loads(first) == manifest
    assert first.endswith(b"\n")
