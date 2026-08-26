from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from utils.sonata_second_preflight import (
    SonataSecondPreflightError,
    build_sonata_data_manifest,
    build_sonata_source_tree_contract,
    build_sonata_training_semantics,
    canonical_sha256,
    directory_content_manifest,
    issue_sonata_preflight_authorization,
    portable_resolved_config,
    validate_sonata_preflight_authorization,
)


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rio = tmp_path / "private" / "rio"
    scannet = tmp_path / "private" / "scannet"
    taxonomy = {
        1: {"name": "wall", "validation": True},
        2: {"name": "floor", "validation": True},
        3: {"name": "chair", "validation": True},
    }
    for root in (rio, scannet):
        _write_yaml(root / "label_database.yaml", taxonomy)
        _write_yaml(
            root / "color_mean_std.yaml",
            {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        )
        for split in ("train", "validation"):
            array_path = root / split / f"{split}_00.npy"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            points = np.zeros((3, 12), dtype=np.float32)
            points[:, 6:9] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            np.save(array_path, points)
        _write_yaml(root / "train_database.yaml", [{"filepath": "train/train_00.npy"}])
        _write_yaml(
            root / "validation_database.yaml",
            [{"filepath": "validation/validation_00.npy"}],
        )
    test_path = scannet / "test" / "test_00.npy"
    test_path.parent.mkdir(parents=True)
    np.save(test_path, np.zeros((2, 10), dtype=np.float32))
    _write_yaml(scannet / "test_database.yaml", [{"filepath": "test/test_00.npy"}])
    _write_yaml(
        rio / "sequence_database_sliding_2.yaml",
        [{"scene": "scene0000_00-scene0000_01", "scan_indices": [0, 1]}],
    )
    return rio, scannet


def _bindings() -> dict[str, str]:
    return {
        "source_tree_sha256": "1" * 64,
        "source_commit": "2" * 40,
        "config_sha256": "3" * 64,
        "weight_manifest_sha256": "4" * 64,
        "data_manifest_sha256": "5" * 64,
        "environment_manifest_sha256": "6" * 64,
        "training_semantics_sha256": "7" * 64,
    }


def test_portable_resolved_config_removes_machine_paths() -> None:
    payload = {
        "backbone": {"name": "/private/weights/sonata.pth"},
        "general": {"save_dir": "/private/training/output"},
        "data": {"voxel_size": 0.02},
    }

    portable = portable_resolved_config(
        payload,
        expected_weight_path=Path("/private/weights/sonata.pth"),
        expected_output_dir=Path("/private/training/output"),
        weight_sha256="a" * 64,
    )

    assert portable["backbone"]["name"] == "external:sonata_verified_input/" + "a" * 64
    assert portable["general"]["save_dir"] == "external:sonata_training_output"
    serialized = json.dumps(portable, sort_keys=True)
    assert "/private" not in serialized


def test_data_manifest_binds_content_splits_taxonomy_and_finite_normals(
    tmp_path: Path,
) -> None:
    rio, scannet = _dataset_fixture(tmp_path)
    expected_rio = directory_content_manifest(rio)
    expected_scannet = directory_content_manifest(scannet)

    manifest = build_sonata_data_manifest(
        rio,
        scannet,
        expected_rio=expected_rio,
        expected_scannet=expected_scannet,
        expected_database_counts={
            "rio": {"train": 1, "validation": 1, "t2_sequences": 1},
            "scannet": {"train": 1, "validation": 1, "test": 1},
        },
    )

    assert manifest["status"] == "pass"
    assert manifest["datasets"]["rio"]["content"] == expected_rio
    assert manifest["datasets"]["scannet"]["content"] == expected_scannet
    assert manifest["datasets"]["rio"]["normals"]["finite"] is True
    assert manifest["datasets"]["rio"]["normals"]["dimension"] == 3
    assert manifest["datasets"]["scannet"]["normals"]["finite"] is True
    assert manifest["taxonomy"]["status"] == "match"
    serialized = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in serialized


def test_data_manifest_rejects_nonfinite_normals(tmp_path: Path) -> None:
    rio, scannet = _dataset_fixture(tmp_path)
    array_path = next(rio.glob("train/*.npy"))
    points = np.load(array_path)
    points[0, 7] = np.nan
    np.save(array_path, points)

    with pytest.raises(SonataSecondPreflightError, match="non-finite normals"):
        build_sonata_data_manifest(
            rio,
            scannet,
            expected_rio=directory_content_manifest(rio),
            expected_scannet=directory_content_manifest(scannet),
            expected_database_counts={
                "rio": {"train": 1, "validation": 1, "t2_sequences": 1},
                "scannet": {"train": 1, "validation": 1, "test": 1},
            },
        )


def test_data_manifest_rejects_content_or_t2_asset_mismatch(tmp_path: Path) -> None:
    rio, scannet = _dataset_fixture(tmp_path)
    expected_rio = directory_content_manifest(rio)
    expected_scannet = directory_content_manifest(scannet)
    (rio / "sequence_database_sliding_2.yaml").unlink()

    with pytest.raises(SonataSecondPreflightError, match="content manifest mismatch"):
        build_sonata_data_manifest(
            rio,
            scannet,
            expected_rio=expected_rio,
            expected_scannet=expected_scannet,
            expected_database_counts={},
        )


def test_authorization_binds_every_upstream_hash_and_validates() -> None:
    bindings = _bindings()
    authorization = issue_sonata_preflight_authorization(
        **bindings,
        issued_at="2026-08-26T08:00:00Z",
        max_age_seconds=86400,
    )

    assert authorization["gate"] == "SP0-PASS"
    assert authorization["bindings"] == bindings
    assert authorization["authorization_sha256"] == canonical_sha256(
        {key: value for key, value in authorization.items() if key != "authorization_sha256"}
    )
    validate_sonata_preflight_authorization(
        authorization,
        expected_bindings=bindings,
        now=datetime(2026, 8, 26, 8, 30, tzinfo=timezone.utc),
    )


def test_authorization_rejects_tampering_staleness_and_binding_drift() -> None:
    bindings = _bindings()
    authorization = issue_sonata_preflight_authorization(
        **bindings,
        issued_at="2026-08-26T08:00:00Z",
        max_age_seconds=60,
    )
    tampered = json.loads(json.dumps(authorization))
    tampered["bindings"]["config_sha256"] = "8" * 64

    with pytest.raises(SonataSecondPreflightError, match="payload hash"):
        validate_sonata_preflight_authorization(
            tampered,
            expected_bindings=bindings,
            now=datetime(2026, 8, 26, 8, 0, 30, tzinfo=timezone.utc),
        )
    with pytest.raises(SonataSecondPreflightError, match="stale"):
        validate_sonata_preflight_authorization(
            authorization,
            expected_bindings=bindings,
            now=datetime(2026, 8, 26, 8, 2, tzinfo=timezone.utc),
        )
    drifted = dict(bindings)
    drifted["data_manifest_sha256"] = "9" * 64
    with pytest.raises(SonataSecondPreflightError, match="bindings"):
        validate_sonata_preflight_authorization(
            authorization,
            expected_bindings=drifted,
            now=datetime(2026, 8, 26, 8, 0, 30, tzinfo=timezone.utc),
        )


def test_source_tree_contract_requires_committed_scoped_files(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "conf").mkdir(parents=True)
    (repository / "conf" / "config.yaml").write_text("value: 1\n", encoding="utf-8")
    (repository / "train.py").write_text("print('train')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repository, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)

    contract = build_sonata_source_tree_contract(
        repository,
        scopes=("conf", "train.py"),
        require_clean=True,
    )

    assert contract["status"] == "pass"
    assert contract["file_count"] == 2
    assert len(contract["content_sha256"]) == 64
    assert all(entry["ref"].startswith("repo:") for entry in contract["files"])

    (repository / "train.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(SonataSecondPreflightError, match="uncommitted"):
        build_sonata_source_tree_contract(
            repository,
            scopes=("conf", "train.py"),
            require_clean=True,
        )


def test_training_semantics_are_derived_from_the_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SONATA_CHECKPOINT", "/verified/sonata.pth")
    monkeypatch.setenv("SONATA_OUTPUT_DIR", "/training/sonata-second")
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[1] / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.2"):
        cfg = compose(config_name="config_rescene4d_sonata_second")

    semantics = build_sonata_training_semantics(
        cfg,
        config_sha256="a" * 64,
        weight_manifest_sha256="b" * 64,
        load_key_audit_sha256="c" * 64,
    )

    assert semantics["status"] == "pass"
    assert semantics["effective_global_batch"] == 32
    assert semantics["optimizer"]["target"] == "torch.optim.AdamW"
    assert semantics["optimizer"]["lr"] == 5e-4
    assert semantics["scheduler"]["target"] == "torch.optim.lr_scheduler.OneCycleLR"
    assert semantics["scheduler"]["interval"] == "step"
    assert semantics["objective"] == {
        "implementation": "trainer.trainer.aggregate_objective_loss",
        "weighted": True,
        "contrastive": False,
        "class_mask_dice_weights": [2.0, 5.0, 2.0],
        "eos_coef": 0.2,
    }
    assert semantics["checkpoint_selection"]["monitor"] == "val_mean_t-AP"


@pytest.mark.parametrize(
    "script",
    ["scripts/sonata_second_preflight.py", "scripts/run_sonata_second_training.py"],
)
def test_sonata_preflight_clis_support_direct_execution(script: str) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_training_launcher_rejects_missing_authorization(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_sonata_second_training.py",
            "--weight-path",
            str(tmp_path / "missing.pth"),
            "--training-output-dir",
            str(tmp_path / "training"),
            "--artifact-dir",
            str(tmp_path / "missing-artifacts"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "authorization" in result.stderr.lower()
