import importlib
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from datasets.semseg import SemanticSegmentationDataset


def _load_audit_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_temporal_loader.py"
    assert script.exists(), "temporal loader audit script has not been implemented"
    return importlib.import_module("scripts.audit_temporal_loader")


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


def _make_known_empty_rio_fixture(tmp_path: Path) -> Path:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "validation_database.yaml"
    database = yaml.safe_load(database_path.read_text(encoding="utf-8"))

    empty_scan_id = "0171_01"
    empty_scan_name = "scene0171_01"
    empty_point_path = processed_dir / "rio" / "validation" / f"{empty_scan_id}.npy"
    empty_instance_path = (
        processed_dir
        / "rio"
        / "instance_gt"
        / "validation"
        / f"{empty_scan_name}.txt"
    )
    empty_point_path.parent.mkdir(parents=True)
    empty_instance_path.parent.mkdir(parents=True)
    Path(database[2]["filepath"]).rename(empty_point_path)
    Path(database[2]["instance_gt_filepath"]).rename(empty_instance_path)
    database[2]["filepath"] = str(empty_point_path)
    database[2]["instance_gt_filepath"] = str(empty_instance_path)
    _write_yaml(database_path, database)

    changes = np.zeros(4, dtype=np.int64)
    change_path = processed_dir / "known_empty_changes.txt"
    np.savetxt(change_path, changes, fmt="%d")
    _write_yaml(
        processed_dir / "sequence_database_sliding_2.yaml",
        {
            "scene0000_00-scene0000_01": {
                "type": "validation",
                "ambiguities": [],
                "filepath": str(change_path),
            },
            f"scene0000_01-{empty_scan_name}": {
                "type": "validation",
                "ambiguities": [],
                "filepath": str(change_path),
            },
        },
    )
    return processed_dir


def _make_dataset(
    processed_dir: Path,
    *,
    fail_closed: bool = True,
    temporal_window: int = 3,
    known_empty_scan_policy: str | None = None,
) -> SemanticSegmentationDataset:
    kwargs = {
        "dataset_name": "rio",
        "data_dir": str(processed_dir),
        "label_db_filepath": str(processed_dir / "label_database.yaml"),
        "change_label_db_filepath": str(processed_dir / "change_label_database.yaml"),
        "color_mean_std": str(processed_dir / "color_mean_std.yaml"),
        "mode": "validation",
        "add_colors": True,
        "add_normals": False,
        "add_raw_coordinates": False,
        "add_instance": True,
        "num_labels": 20,
        "num_changes": -1,
        "filter_out_classes": [0, 1],
        "label_offset": 2,
        "temporal_window": temporal_window,
        "fail_closed": fail_closed,
    }
    if known_empty_scan_policy is not None:
        kwargs["known_empty_scan_policy"] = known_empty_scan_policy
    return SemanticSegmentationDataset(**kwargs)


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


def test_known_empty_scan_error_policy_fails_before_returning_sample_zero(
    tmp_path: Path,
) -> None:
    processed_dir = _make_known_empty_rio_fixture(tmp_path)
    problem_sequence = "scene0000_01-scene0171_01"

    with pytest.raises(
        ValueError,
        match=rf"0171_01.*{problem_sequence}",
    ):
        _make_dataset(
            processed_dir,
            fail_closed=True,
            temporal_window=2,
            known_empty_scan_policy="error",
        )


def test_official_known_empty_scan_substitution_is_observable(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    processed_dir = _make_known_empty_rio_fixture(tmp_path)
    dataset = _make_dataset(
        processed_dir,
        fail_closed=False,
        temporal_window=2,
    )

    with caplog.at_level(logging.WARNING, logger="datasets.semseg"):
        result = dataset[1]

    assert result[3] == "scene0000_00-scene0000_01"
    assert result[7] == 0
    assert dataset.known_empty_scan_substitution_count == 1
    record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "known_empty_scan_substitution"
    )
    assert record.policy == "official_substitute"
    assert record.scene_source == "rio"
    assert record.scene_id == "0171_01"
    assert record.sequence_name == "scene0000_01-scene0171_01"
    assert record.requested_index == 1
    assert record.substitute_index == 0
    assert record.substitution_count == 1


@pytest.mark.parametrize("invalid_database", [{}, [], None, [{"type": "validation"}]])
def test_temporal_sequence_database_requires_a_non_empty_mapping(
    tmp_path: Path,
    invalid_database,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "sequence_database_sliding_3.yaml"
    _write_yaml(database_path, invalid_database)

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(database_path))}.*non-empty mapping",
    ):
        _make_dataset(processed_dir)


def test_temporal_sequence_database_requires_requested_mode(
    tmp_path: Path,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "sequence_database_sliding_3.yaml"
    database = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    for entry in database.values():
        entry["type"] = "train"
    _write_yaml(database_path, database)

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(database_path))}.*mode 'validation'",
    ):
        _make_dataset(processed_dir)


@pytest.mark.parametrize(
    "invalid_sequence",
    [
        "scene0000_00",
        "scene0000_00-scene0000_01",
        "scene0000_00-scene0000_01-scene0000_02-scene0000_03",
    ],
)
def test_temporal_sequence_length_mismatch_has_database_and_sequence_context(
    tmp_path: Path,
    invalid_sequence: str,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "sequence_database_sliding_3.yaml"
    database = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    entry = next(iter(database.values()))
    _write_yaml(database_path, {invalid_sequence: entry})

    with pytest.raises(
        ValueError,
        match=rf"{re.escape(str(database_path))}.*{invalid_sequence}.*expected 3",
    ):
        _make_dataset(processed_dir)


def test_missing_temporal_sequence_database_raises_with_path(
    tmp_path: Path,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "sequence_database_sliding_3.yaml"
    database_path.unlink()

    with pytest.raises(FileNotFoundError, match=re.escape(str(database_path))):
        _make_dataset(processed_dir)


def test_unknown_temporal_scan_has_database_sequence_and_name_context(
    tmp_path: Path,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    database_path = processed_dir / "sequence_database_sliding_3.yaml"
    database = yaml.safe_load(database_path.read_text(encoding="utf-8"))
    entry = next(iter(database.values()))
    missing_name = "scene9999_99"
    bad_sequence = f"scene0000_00-{missing_name}-scene0000_02"
    _write_yaml(database_path, {bad_sequence: entry})

    with pytest.raises(
        KeyError,
        match=(
            rf"{re.escape(str(database_path))}.*{bad_sequence}.*{missing_name}"
        ),
    ):
        _make_dataset(processed_dir)


def test_audit_split_records_loader_shapes_and_projection(tmp_path: Path) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    audit = _load_audit_module()

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


def test_repo_reference_serializes_repository_paths_without_absolute_prefix() -> None:
    audit = _load_audit_module()
    processed_dir = audit.PROJECT_ROOT / "data" / "processed" / "rio"

    assert audit.repo_reference(processed_dir) == "repo:data/processed/rio"
    assert audit.repo_reference(
        processed_dir / "sequence_database_sliding_3.yaml"
    ) == "repo:data/processed/rio/sequence_database_sliding_3.yaml"


def test_committed_audit_uses_repo_references_without_personal_paths() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "data_audit"
        / "temporal_loader_audit.json"
    )
    serialized = artifact_path.read_text(encoding="utf-8")
    artifact = yaml.safe_load(serialized)

    assert artifact["configuration"]["processed_dir"] == "repo:data/processed/rio"
    for record in artifact["audits"]:
        assert record["database_path"].startswith("repo:")
        for sample in record["samples"]:
            assert sample["change_filepath"].startswith("repo:")
    assert "/home/" not in serialized
    assert "/Users/" not in serialized


def test_missing_split_writes_blocked_artifact_before_nonzero_exit(
    tmp_path: Path,
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_temporal_loader.py"
    output = tmp_path / "blocked.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--processed-dir",
            str(processed_dir),
            "--output",
            str(output),
            "--horizons",
            "3",
            "--splits",
            "train",
            "--samples-per-split",
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    artifact = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert artifact["status"] == "blocked"
    assert artifact["totals"]["failures"] == 1
    assert artifact["audits"][0]["exceptions"][0]["type"] == "SystemExit"


def test_database_loader_count_mismatch_blocks_audit(
    monkeypatch, tmp_path: Path
) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    audit = _load_audit_module()

    class CountMismatchDataset:
        def __len__(self) -> int:
            return 2

    monkeypatch.setattr(audit, "make_dataset", lambda *_args, **_kwargs: CountMismatchDataset())

    artifact = audit.run_audit(
        processed_dir=processed_dir,
        horizons=[3],
        splits=["validation"],
        sample_limit=5,
        seed=45,
    )

    assert artifact["status"] == "blocked"
    assert artifact["totals"]["failures"] == 1
    audit_record = artifact["audits"][0]
    assert audit_record["success_count"] == 0
    assert audit_record["validation_errors"][0]["code"] == "database_loader_count_mismatch"


def test_false_projection_blocks_audit(monkeypatch, tmp_path: Path) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    audit = _load_audit_module()
    original_inspect_sample = audit.inspect_sample

    def inspect_with_false_projection(*args, **kwargs):
        sample = original_inspect_sample(*args, **kwargs)
        sample["projection_matches_official_rule"] = False
        return sample

    monkeypatch.setattr(audit, "inspect_sample", inspect_with_false_projection)

    artifact = audit.run_audit(
        processed_dir=processed_dir,
        horizons=[3],
        splits=["validation"],
        sample_limit=1,
        seed=45,
    )

    assert artifact["status"] == "blocked"
    assert artifact["totals"]["failures"] == 1
    audit_record = artifact["audits"][0]
    assert audit_record["success_count"] == 0
    assert audit_record["validation_errors"][0]["errors"] == [
        {
            "code": "projection_mismatch",
            "expected": True,
            "actual": False,
        }
    ]


def test_audit_records_unambiguous_generator_provenance(tmp_path: Path) -> None:
    processed_dir, _ = _make_three_scan_fixture(tmp_path)
    audit = _load_audit_module()

    artifact = audit.run_audit(
        processed_dir=processed_dir,
        horizons=[3],
        splits=["validation"],
        sample_limit=1,
        seed=45,
    )

    script_path = Path(audit.__file__).resolve()
    assert artifact["official_source_commit"] == (
        "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
    )
    assert artifact["generator_git_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=script_path.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert isinstance(artifact["generator_dirty"], bool)
    assert artifact["source_hashes"]["scripts/audit_temporal_loader.py"] == (
        audit.sha256_file(script_path)
    )
    assert "git_commit" not in artifact


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
