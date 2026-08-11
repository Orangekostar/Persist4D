#!/usr/bin/env python3
"""Audit the P2 ReScene4D-C reproduction contract and ScanNet data gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.p2_preflight import (
    P2_KNOWN_EMPTY_RIO_SCAN_ID,
    P2_KNOWN_EMPTY_RIO_SEQUENCES,
    P2_KNOWN_EMPTY_SCANNET_SCAN_IDS,
    P2_PREFLIGHT_SCHEMA_VERSION,
    P2_RIO_SEQUENCE_DATABASE_REF,
    P2_RIO_SEQUENCE_DATABASE_SHA256,
    P2_TRAINING_CONTRACT_SCHEMA_VERSION,
    P2_TRAINING_SEMANTIC_SHA256,
    SCANNET_OFFICIAL_COMMIT,
    SCANNET_OFFICIAL_REPOSITORY_REF,
    SCANNET_SPLIT_SHA256,
    build_p2_input_manifest,
    build_p2_runtime_environment_contract,
    build_p2_runtime_source_contract,
    build_p2_source_tree_contract,
    build_scannet_official_split_identity,
    issue_formal_authorization,
    not_issued_authorization,
    p2_training_semantic_sha256,
    validate_p2_training_config_contract,
)

OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
P2_TRAINING_CONTRACT_FIX_COMMIT = "3c6b11a3af600aa98c93128361c2ecb4900ea186"
P2_RUNTIME_SAFETY_FIX_COMMIT = "52781187a236b7114cf067b59540120a9aebe8fe"
CONCERTO_REVISION = "c31f993a56129f2ba9c5d06a35957e3f05bff710"
CONCERTO_SHA256 = "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
CONCERTO_BYTES = 433_987_358
SEQUENCE_DB_SHA256 = P2_RIO_SEQUENCE_DATABASE_SHA256

ADAMW_IMPLICIT_DEFAULTS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "amsgrad": False,
}
ONECYCLE_IMPLICIT_DEFAULTS = {
    "pct_start": 0.3,
    "anneal_strategy": "cos",
    "cycle_momentum": True,
    "base_momentum": 0.85,
    "max_momentum": 0.95,
    "div_factor": 25.0,
    "final_div_factor": 10000.0,
    "three_phase": False,
}
FROZEN_ENCODER_RUNTIME = {
    "parameters_require_grad": False,
    "module_mode": "train",
    "concerto_drop_path_rate": 0.3,
    "decoder_and_head_trainable": True,
}

OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
SPLIT_FILES = {
    "train": "scannetv2_train.txt",
    "validation": "scannetv2_val.txt",
    "test": "scannetv2_test.txt",
}
DATABASE_FILES = {
    "train": "train_database.yaml",
    "validation": "validation_database.yaml",
    "test": "test_database.yaml",
}
NYU40_INSTANCE_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
NYU40_INSTANCE_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

REPRODUCTION_TARGET: dict[str, Any] = {
    "schema_version": 1,
    "stage": "P2",
    "baseline": "ReScene4D-C T=2",
    "official_source_ref": "external:arxiv/2601.11508v2",
    "official_code_ref": "external:github/GradientSpaces/rescene4d",
    "official_source_commit": OFFICIAL_SOURCE_COMMIT,
    "model": {
        "backbone": "Concerto",
        "encoder": "PTv3 frozen",
        "decoder": "train from scratch",
        "queries": 100,
        "query_initialization": "FPS non-parametric",
        "temporal_window": 2,
        "contrastive": True,
        "st_serialization": True,
        "st_masking": False,
    },
    "data": {
        "mix": [
            {"dataset": "3RScan", "temporal_window": 2, "weight": 1.0},
            {"dataset": "ScanNet", "temporal_window": 1, "weight": 0.8},
        ],
        "voxel_size_m": 0.02,
        "taxonomy": "NYU40 18-class instance subset",
        "sequence_database": "sliding T=2",
    },
    "training": {
        "epochs": 450,
        "effective_batch_size": 32,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "max_lr": 0.0005,
        "loss_weights": {"class": 2.0, "mask_bce": 5.0, "dice": 2.0},
        "no_object_weight": 0.2,
        "seed": 45,
        "precision": "32-true",
    },
    "local_recommended_topology": {
        "gpus": 2,
        "batch_per_gpu": 4,
        "gradient_accumulation": 4,
        "physical_global_batch": 8,
        "effective_batch": 32,
    },
    "checkpoint": {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "revision": CONCERTO_REVISION,
        "sha256": CONCERTO_SHA256,
    },
    "reproduction_choices": {
        "sequence_database": {
            "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml",
            "sha256": SEQUENCE_DB_SHA256,
        },
        "frozen_encoder_runtime": FROZEN_ENCODER_RUNTIME,
        "adamw_implicit_defaults": ADAMW_IMPLICIT_DEFAULTS,
        "onecycle_implicit_defaults": ONECYCLE_IMPLICIT_DEFAULTS,
        "augmentation_exactness": "repository recipe; paper exact transform list unreported",
    },
}


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _yaml_text(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=False)


def _read_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_ref(path: Path, external_role: str) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except (OSError, ValueError):
        return f"external:{external_role}"
    return f"repo:{relative.as_posix()}"


def _audit_model_checkpoint(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint_path = Path(checkpoint_path).expanduser()
    default_path = (
        Path.home() / ".cache" / "persist4d" / "concerto" / "concerto_base.pth"
    )
    reference = (
        "local_cache:persist4d/concerto/concerto_base.pth"
        if checkpoint_path == default_path
        else _portable_ref(checkpoint_path, "concerto_checkpoint")
    )
    errors: list[dict[str, Any]] = []
    observed_bytes: int | None = None
    observed_sha256: str | None = None
    if not checkpoint_path.is_file():
        errors.append(_error("model_checkpoint_missing"))
    else:
        try:
            observed_bytes = checkpoint_path.stat().st_size
            observed_sha256 = _sha256_file(checkpoint_path)
        except OSError:
            errors.append(_error("model_checkpoint_unreadable"))
        if observed_bytes is not None and observed_bytes != CONCERTO_BYTES:
            errors.append(
                _error(
                    "model_checkpoint_size_mismatch",
                    expected=CONCERTO_BYTES,
                    observed=observed_bytes,
                )
            )
        if observed_sha256 != CONCERTO_SHA256:
            errors.append(_error("model_checkpoint_sha256_mismatch"))
    return (
        {
            "reference": reference,
            "expected_sha256": CONCERTO_SHA256,
            "observed_sha256": observed_sha256,
            "expected_byte_size": CONCERTO_BYTES,
            "observed_byte_size": observed_bytes,
            "status": "pass" if not errors else "fail",
        },
        errors,
    )


def _join_ref(root_ref: str, relative: str) -> str:
    return f"{root_ref.rstrip('/')}/{relative.lstrip('/')}"


def _git_revision(path: Path, fallback: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    revision = result.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else fallback


def _error(code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, **details}


def _audit_known_empty_scan_substitutions(
    p2_config: Any | None,
    sequence_database_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = sequence_database_path or (
        REPO_ROOT / "data" / "processed" / "rio" / "sequence_database_sliding_2.yaml"
    )
    observed_sha256 = _sha256_file(path)
    errors: list[dict[str, Any]] = []
    fail_closed: dict[str, Any] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    policies: dict[str, Any] = {
        "train": None,
        "validation": None,
        "test": None,
    }
    temporal_windows: dict[str, Any] = {
        "train": None,
        "validation": None,
        "test": None,
    }

    if p2_config is None:
        errors.append(_error("known_empty_scan_p2_config_unavailable"))
    else:
        try:
            train = p2_config.data.train_dataset
            validation = p2_config.data.validation_dataset
            test = p2_config.data.test_dataset
            split_configs = {
                "train": train,
                "validation": validation,
                "test": test,
            }
            for split, dataset_config in split_configs.items():
                fail_closed[split] = dataset_config.get("fail_closed")
                policies[split] = dataset_config.get("known_empty_scan_policy")
                if fail_closed[split] is not True:
                    errors.append(
                        _error(
                            "known_empty_scan_fail_closed_config_mismatch",
                            split=split,
                        )
                    )
                if policies[split] != "official_substitute":
                    errors.append(
                        _error(
                            "known_empty_scan_policy_config_mismatch",
                            split=split,
                        )
                    )

            rio_children = [
                child
                for child in train.datasets
                if child.get("dataset_name") == "rio"
            ]
            if len(rio_children) != 1:
                errors.append(_error("known_empty_scan_train_rio_config_mismatch"))
            else:
                temporal_windows["train"] = rio_children[0].get(
                    "temporal_window"
                )
            temporal_windows["validation"] = validation.get("temporal_window")
            temporal_windows["test"] = test.get("temporal_window")
            if any(value != 2 for value in temporal_windows.values()):
                errors.append(
                    _error("known_empty_scan_temporal_window_config_mismatch")
                )
            if validation.get("dataset_name") != "rio":
                errors.append(
                    _error(
                        "known_empty_scan_dataset_config_mismatch",
                        split="validation",
                    )
                )
            if test.get("dataset_name") != "rio":
                errors.append(
                    _error(
                        "known_empty_scan_dataset_config_mismatch",
                        split="test",
                    )
                )
        except (AttributeError, KeyError, TypeError):
            errors.append(_error("known_empty_scan_p2_config_invalid"))

    if observed_sha256 is None:
        errors.append(_error("rio_sequence_database_missing"))
    elif observed_sha256 != P2_RIO_SEQUENCE_DATABASE_SHA256:
        errors.append(_error("rio_sequence_database_sha256_mismatch"))

    affected_sequences: list[str] = []
    try:
        sequence_database = _read_yaml(path)
    except (OSError, yaml.YAMLError):
        errors.append(_error("rio_sequence_database_unreadable"))
    else:
        if not isinstance(sequence_database, Mapping):
            errors.append(_error("rio_sequence_database_invalid"))
        else:
            known_scan = f"scene{P2_KNOWN_EMPTY_RIO_SCAN_ID}"
            affected_sequences = sorted(
                str(name)
                for name in sequence_database
                if known_scan in str(name).split("-")
            )
            if affected_sequences != P2_KNOWN_EMPTY_RIO_SEQUENCES:
                errors.append(_error("known_empty_scan_sequences_mismatch"))

    common_policy = (
        "official_substitute"
        if set(policies.values()) == {"official_substitute"}
        else "config_mismatch"
    )
    return (
        {
            "status": "pass" if not errors else "fail",
            "dataset": "rio",
            "temporal_window": 2,
            "known_empty_scan_id": P2_KNOWN_EMPTY_RIO_SCAN_ID,
            "policy": common_policy,
            "sequence_database_ref": (
                P2_RIO_SEQUENCE_DATABASE_REF
                if path.resolve()
                == (
                    REPO_ROOT
                    / "data"
                    / "processed"
                    / "rio"
                    / "sequence_database_sliding_2.yaml"
                ).resolve()
                else _portable_ref(path, "rio_sequence_database")
            ),
            "expected_sequence_database_sha256": (
                P2_RIO_SEQUENCE_DATABASE_SHA256
            ),
            "observed_sequence_database_sha256": observed_sha256,
            "fail_closed": fail_closed,
            "affected_sequences": affected_sequences,
            "scannet_known_empty_scan_ids": P2_KNOWN_EMPTY_SCANNET_SCAN_IDS,
        },
        errors,
    )


def _audit_formal_data_roots(
    p2_config: Any | None,
    processed_scannet_dir: Path,
    rio_processed_dir: Path,
    *,
    raw_scannet_dir: Path | None = None,
    split_dir: Path | None = None,
    test_segments_dir: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    expected_paths: dict[str, Path] = {}
    try:
        if p2_config is None:
            raise TypeError
        children = p2_config.data.train_dataset.datasets
        by_name = {str(child.dataset_name): child for child in children}
        if set(by_name) != {"rio", "scannet"}:
            raise ValueError
        for name in ("scannet", "rio"):
            path = Path(str(by_name[name].data_dir)).expanduser()
            expected_paths[name] = (
                path if path.is_absolute() else REPO_ROOT / path
            ).resolve()
        expected_paths.update(
            {
                "raw_scannet": (
                    REPO_ROOT / "data" / "raw" / "scannet" / "scannet"
                ).resolve(),
                "split_metadata": (
                    REPO_ROOT
                    / "third_party"
                    / "ScanNet"
                    / "Tasks"
                    / "Benchmark"
                ).resolve(),
                "test_segments": (
                    REPO_ROOT / "data" / "raw" / "scannet_test_segments"
                ).resolve(),
            }
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        errors.append(_error("formal_data_root_config_invalid"))

    observed_paths = {
        "scannet": Path(processed_scannet_dir).expanduser().resolve(),
        "rio": Path(rio_processed_dir).expanduser().resolve(),
        "raw_scannet": Path(
            raw_scannet_dir
            or REPO_ROOT / "data" / "raw" / "scannet" / "scannet"
        ).expanduser().resolve(),
        "split_metadata": Path(
            split_dir
            or REPO_ROOT
            / "third_party"
            / "ScanNet"
            / "Tasks"
            / "Benchmark"
        ).expanduser().resolve(),
        "test_segments": Path(
            test_segments_dir
            or REPO_ROOT / "data" / "raw" / "scannet_test_segments"
        ).expanduser().resolve(),
    }
    for name in (
        "raw_scannet",
        "scannet",
        "rio",
        "split_metadata",
        "test_segments",
    ):
        expected = expected_paths.get(name)
        if expected is not None and observed_paths[name] != expected:
            errors.append(_error(f"formal_{name}_data_root_mismatch"))

    expected_refs = {
        name: _portable_ref(path, f"configured_{name}_processed")
        for name, path in expected_paths.items()
    }
    observed_refs = {
        name: (
            expected_refs[name]
            if name in expected_paths and path == expected_paths[name]
            else _portable_ref(path, f"audited_{name}_processed")
        )
        for name, path in observed_paths.items()
    }
    return (
        {
            "status": "pass" if not errors else "fail",
            "expected": expected_refs,
            "observed": observed_refs,
        },
        errors,
    )


def _scene_error(code: str, split: str, scenes: Sequence[str]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "split": split,
        "missing_scene_count": len(scenes),
        "scenes": list(scenes[:10]),
    }
    if len(scenes) == 1:
        details["scene"] = scenes[0]
    return _error(code, **details)


def _scene_from_record(record: Mapping[str, Any]) -> str | None:
    try:
        return f"scene{int(record['scene']):04d}_{int(record['sub_scene']):02d}"
    except (KeyError, TypeError, ValueError):
        pass
    instance_path = record.get("instance_gt_filepath")
    if instance_path:
        match = re.search(r"scene\d{4}_\d{2}", Path(str(instance_path)).stem)
        if match:
            return match.group(0)
    filepath = record.get("filepath")
    if filepath:
        match = re.search(r"(?:scene)?(\d{4}_\d{2})", Path(str(filepath)).stem)
        if match:
            return f"scene{match.group(1)}"
    return None


def _resolve_record_path(value: Any, processed_dir: Path, split: str) -> Path | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_split_metadata(
    split_dir: Path,
    expected_counts: Mapping[str, int],
) -> tuple[dict[str, list[str]], dict[str, Any], list[dict[str, Any]]]:
    scenes_by_split: dict[str, list[str]] = {}
    records: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    split_root_ref = _portable_ref(split_dir, "scannet_splits")

    for split in ("train", "validation", "test"):
        filename = SPLIT_FILES[split]
        path = split_dir / filename
        expected = int(expected_counts[split])
        observed_sha256 = None
        read_failed = False
        if path.is_file():
            try:
                before = path.stat()
                payload = path.read_bytes()
                after = path.stat()
                if (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                ) != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise OSError("split changed while reading")
                observed_sha256 = hashlib.sha256(payload).hexdigest()
                scenes = [
                    line.strip()
                    for line in payload.decode("utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeError):
                scenes = []
                read_failed = True
                errors.append(_error("split_file_unreadable", split=split))
        else:
            scenes = []
            read_failed = True
            errors.append(_error("split_file_missing", split=split))
        unique_scenes = list(dict.fromkeys(scenes))
        valid_names = all(re.fullmatch(r"scene\d{4}_\d{2}", scene) for scene in scenes)
        status = "fail" if read_failed else "pass"
        if len(scenes) != expected:
            errors.append(
                _error(
                    "split_count_mismatch",
                    split=split,
                    expected=expected,
                    observed=len(scenes),
                )
            )
            status = "fail"
        if len(unique_scenes) != len(scenes):
            errors.append(_error("split_duplicate_scene", split=split))
            status = "fail"
        if not valid_names:
            errors.append(_error("split_scene_name_invalid", split=split))
            status = "fail"
        scenes_by_split[split] = unique_scenes
        records[split] = {
            "source_ref": _join_ref(split_root_ref, filename),
            "expected": expected,
            "observed": len(scenes),
            "unique": len(unique_scenes),
            "observed_sha256": observed_sha256,
            "status": status,
        }
    scene_owners: dict[str, list[str]] = {}
    for split, scenes in scenes_by_split.items():
        for scene in scenes:
            scene_owners.setdefault(scene, []).append(split)
    overlap = {
        scene: splits
        for scene, splits in scene_owners.items()
        if len(splits) > 1
    }
    if overlap:
        errors.append(
            _error(
                "split_cross_partition_overlap",
                overlap_count=len(overlap),
                scenes=sorted(overlap)[:10],
            )
        )
        for split, record in records.items():
            if any(split in splits for splits in overlap.values()):
                record["status"] = "fail"
    return scenes_by_split, records, errors


def _audit_official_split_identity(
    split_dir: Path,
    *,
    split_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if split_records is None:
        identity = build_scannet_official_split_identity(split_dir=split_dir)
    else:
        try:
            observed_commit = subprocess.run(
                ["git", "-C", str(split_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            observed_commit = None
        files = {}
        for split, filename in SPLIT_FILES.items():
            record = split_records.get(split, {})
            observed_sha256 = record.get("observed_sha256")
            observed_count = record.get("observed")
            files[split] = {
                "reference": record.get(
                    "source_ref",
                    _join_ref(
                        _portable_ref(split_dir, "scannet_splits"),
                        filename,
                    ),
                ),
                "expected_sha256": SCANNET_SPLIT_SHA256[split],
                "observed_sha256": observed_sha256,
                "expected_scene_count": OFFICIAL_SPLIT_COUNTS[split],
                "observed_scene_count": observed_count,
                "status": (
                    "pass"
                    if observed_sha256 == SCANNET_SPLIT_SHA256[split]
                    and observed_count == OFFICIAL_SPLIT_COUNTS[split]
                    else "fail"
                ),
            }
        identity = {
            "status": (
                "pass"
                if observed_commit == SCANNET_OFFICIAL_COMMIT
                and all(record["status"] == "pass" for record in files.values())
                else "fail"
            ),
            "repository_ref": SCANNET_OFFICIAL_REPOSITORY_REF,
            "expected_commit": SCANNET_OFFICIAL_COMMIT,
            "observed_commit": observed_commit,
            "files": files,
        }
    errors: list[dict[str, Any]] = []
    if identity.get("observed_commit") != SCANNET_OFFICIAL_COMMIT:
        errors.append(_error("scannet_official_commit_mismatch"))
    for split, record in identity.get("files", {}).items():
        if record.get("observed_sha256") != record.get("expected_sha256"):
            errors.append(
                _error("official_split_sha256_mismatch", split=split)
            )
        if record.get("observed_scene_count") != record.get(
            "expected_scene_count"
        ):
            errors.append(
                _error("official_split_scene_count_mismatch", split=split)
            )
    return identity, errors


def _audit_semantic_snapshot_stability(
    *,
    initial_input_manifest: Mapping[str, Any],
    initial_split_identity: Mapping[str, Any],
    processed_scannet_dir: Path,
    rio_processed_dir: Path,
    split_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    final_input_manifest = build_p2_input_manifest(
        scannet_root=processed_scannet_dir,
        rio_root=rio_processed_dir,
    )
    final_split_identity = build_scannet_official_split_identity(
        split_dir=split_dir
    )
    errors: list[dict[str, Any]] = []
    if final_input_manifest != initial_input_manifest:
        errors.append(_error("processed_input_changed_during_semantic_audit"))
    if final_split_identity != initial_split_identity:
        errors.append(_error("official_split_changed_during_semantic_audit"))
    return final_input_manifest, final_split_identity, errors


def _audit_source_contract_stability(
    *,
    initial_source_tree_contract: Mapping[str, Any],
    initial_runtime_source_contract: Mapping[str, Any],
    initial_runtime_environment_contract: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    final_source_tree_contract = build_p2_source_tree_contract()
    final_runtime_source_contract = build_p2_runtime_source_contract()
    final_runtime_environment_contract = build_p2_runtime_environment_contract()
    errors: list[dict[str, Any]] = []
    if final_source_tree_contract != initial_source_tree_contract:
        errors.append(_error("source_tree_changed_during_audit"))
    if final_runtime_source_contract != initial_runtime_source_contract:
        errors.append(_error("runtime_source_changed_during_audit"))
    if final_runtime_environment_contract != initial_runtime_environment_contract:
        errors.append(_error("runtime_environment_changed_during_audit"))
    return (
        final_source_tree_contract,
        final_runtime_source_contract,
        final_runtime_environment_contract,
        errors,
    )


def _initial_input_manifest_allowed(
    *,
    source_tree: Mapping[str, Any],
    runtime_source: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> bool:
    return all(
        contract.get("status") == "pass"
        for contract in (source_tree, runtime_source, runtime_environment)
    )


def _raw_scene_candidates(raw_dir: Path, split: str, scene: str) -> list[Path]:
    if split == "test":
        folders = ("scans_test", "test")
    elif split == "validation":
        folders = ("scans", "val", "validation")
    else:
        folders = ("scans", "train")
    return [*(raw_dir / folder / scene for folder in folders), raw_dir / scene]


def _required_raw_assets(scene: str, split: str) -> list[str]:
    assets = [f"{scene}_vh_clean_2.ply"]
    if split != "test":
        assets.extend(
            [
                f"{scene}_vh_clean_2.labels.ply",
                f"{scene}_vh_clean_2.0.010000.segs.json",
                f"{scene}.aggregation.json",
                f"{scene}.txt",
            ]
        )
    return assets


def _audit_raw_assets(
    raw_dir: Path,
    test_segments_dir: Path,
    scenes_by_split: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    raw_root_ref = _portable_ref(raw_dir, "scannet_raw")
    segments_root_ref = _portable_ref(test_segments_dir, "scannet_test_segments")
    expected_scene_count = sum(len(scenes) for scenes in scenes_by_split.values())
    complete_scene_count = 0
    missing_asset_count = 0
    by_split: dict[str, Any] = {}

    if not raw_dir.is_dir():
        errors.append(_error("scannet_raw_root_missing"))
    label_map = raw_dir / "scannetv2-labels.combined.tsv"
    if not label_map.is_file():
        errors.append(_error("scannet_label_map_missing"))

    for split in ("train", "validation", "test"):
        missing_scenes = 0
        split_missing_assets = 0
        missing_examples: list[dict[str, Any]] = []
        for scene in scenes_by_split[split]:
            candidates = _raw_scene_candidates(raw_dir, split, scene)
            scene_dir = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
            missing = [
                filename
                for filename in _required_raw_assets(scene, split)
                if not (scene_dir / filename).is_file()
            ]
            if split == "test":
                segment_filename = f"{scene}_vh_clean_2.0.010000.segs.json"
                if not (test_segments_dir / segment_filename).is_file():
                    missing.append(segment_filename)
            if missing:
                missing_scenes += 1
                split_missing_assets += len(missing)
                if len(missing_examples) < 10:
                    relative_parent = scene_dir.relative_to(raw_dir).as_posix()
                    example_refs = [
                        _join_ref(
                            segments_root_ref if split == "test" and filename.endswith("segs.json") else raw_root_ref,
                            filename
                            if split == "test" and filename.endswith("segs.json")
                            else f"{relative_parent}/{filename}",
                        )
                        for filename in missing
                    ]
                    missing_examples.append(
                        {"scene": scene, "missing_asset_refs": example_refs}
                    )
            else:
                complete_scene_count += 1
        if missing_scenes:
            errors.append(
                _error(
                    "raw_scene_assets_incomplete",
                    split=split,
                    missing_scene_count=missing_scenes,
                    missing_asset_count=split_missing_assets,
                )
            )
        missing_asset_count += split_missing_assets
        by_split[split] = {
            "expected_scene_count": len(scenes_by_split[split]),
            "complete_scene_count": len(scenes_by_split[split]) - missing_scenes,
            "missing_scene_count": missing_scenes,
            "missing_asset_count": split_missing_assets,
            "missing_examples": missing_examples,
            "status": "pass" if missing_scenes == 0 else "fail",
        }

    return (
        {
            "root_ref": raw_root_ref,
            "test_segments_ref": segments_root_ref,
            "label_map_ref": _join_ref(raw_root_ref, "scannetv2-labels.combined.tsv"),
            "label_map_present": label_map.is_file(),
            "expected_scene_count": expected_scene_count,
            "complete_scene_count": complete_scene_count,
            "missing_asset_count": missing_asset_count,
            "by_split": by_split,
            "status": "pass" if complete_scene_count == expected_scene_count and label_map.is_file() else "fail",
        },
        errors,
    )


def _audit_taxonomy(
    processed_dir: Path,
    dataset_name: str = "scannet",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    label_db_path = processed_dir / "label_database.yaml"
    metric_path = processed_dir / f"{dataset_name}.yaml"
    expected_validation_items = [
        (1, "wall"),
        (2, "floor"),
        *list(zip(NYU40_INSTANCE_IDS, NYU40_INSTANCE_LABELS)),
    ]
    validation_items: list[tuple[int, str | None]] = []
    metric_ids: list[Any] = []
    metric_labels: list[str] = []
    metric_name: str | None = None

    try:
        labels = _read_yaml(label_db_path)
        if not isinstance(labels, Mapping):
            raise TypeError
        for class_id, entry in labels.items():
            if type(class_id) is not int:
                errors.append(
                    _error("label_database_validation_id_type_invalid")
                )
                continue
            if not isinstance(entry, Mapping):
                errors.append(_error("label_database_entry_invalid"))
                continue
            validation = entry.get("validation")
            if type(validation) is not bool:
                errors.append(
                    _error("label_database_validation_flag_type_invalid")
                )
                continue
            if validation:
                name = entry.get("name")
                if isinstance(name, str):
                    normalized_name = name.replace("_", " ")
                else:
                    normalized_name = None
                validation_items.append((class_id, normalized_name))
    except (OSError, TypeError, yaml.YAMLError):
        errors.append(_error("label_database_invalid_or_missing"))

    try:
        metric = _read_yaml(metric_path)
        if not isinstance(metric, Mapping):
            raise TypeError
        metric_name_value = metric.get("name")
        metric_id_values = metric.get("valid_class_ids")
        metric_label_values = metric.get("class_labels")
        if not isinstance(metric_name_value, str):
            raise TypeError
        if not isinstance(metric_id_values, list):
            raise TypeError
        if any(type(value) is not int for value in metric_id_values):
            errors.append(_error("metric_class_id_type_invalid"))
        if not isinstance(metric_label_values, list) or any(
            not isinstance(value, str) for value in metric_label_values
        ):
            errors.append(_error("metric_class_label_type_invalid"))
        metric_name = metric_name_value
        metric_ids = list(metric_id_values)
        metric_labels = list(metric_label_values)
    except (OSError, TypeError, yaml.YAMLError):
        errors.append(_error("metric_taxonomy_invalid_or_missing"))

    if dict(validation_items) != dict(expected_validation_items):
        errors.append(
            _error(
                "label_database_validation_mapping_mismatch",
                expected_count=len(expected_validation_items),
                observed_count=len(validation_items),
            )
        )
    elif validation_items != expected_validation_items:
        errors.append(_error("label_database_validation_order_mismatch"))
    if metric_ids != NYU40_INSTANCE_IDS:
        errors.append(_error("metric_class_ids_mismatch"))
    if metric_labels != NYU40_INSTANCE_LABELS:
        errors.append(_error("metric_class_labels_mismatch"))
    if metric_name != dataset_name:
        errors.append(_error("metric_dataset_name_mismatch"))

    status = "pass" if not errors else "fail"
    return (
        {
            "status": status,
            "name": metric_name,
            "valid_class_ids": metric_ids,
            "class_labels": metric_labels,
            "class_count": len(metric_ids),
        },
        errors,
    )


def _load_unique_yaml_mapping(path: Path) -> Mapping[str, Any]:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.YAMLError(f"duplicate YAML key: {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    if not isinstance(loaded, Mapping):
        raise TypeError("expected a YAML mapping")
    return loaded


def _audit_rio_record_paths(
    rio_processed_dir: Path,
    *,
    validate_content: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind every P2-active RIO record to its canonical processed asset."""
    errors: list[dict[str, Any]] = []
    database_record_counts: dict[str, int] = {}
    records_by_split: dict[str, list[Mapping[str, Any]]] = {}
    scenes_by_split: dict[str, set[str]] = {}
    seen_npy: set[Path] = set()
    seen_gt: set[Path] = set()
    supervision_observed: list[bool] = []
    scene_supervision: dict[str, bool] = {}
    expected_repo_prefix: Path | None
    try:
        expected_repo_prefix = rio_processed_dir.resolve().relative_to(
            REPO_ROOT.resolve()
        )
    except ValueError:
        expected_repo_prefix = None

    for split, expected_count in (("train", 1178), ("validation", 157)):
        database_path = rio_processed_dir / f"{split}_database.yaml"
        try:
            loaded = _read_yaml(database_path)
            if not isinstance(loaded, list) or not all(
                isinstance(record, Mapping) for record in loaded
            ):
                raise TypeError
            records = list(loaded)
        except (OSError, TypeError, yaml.YAMLError):
            records = []
            errors.append(_error("rio_database_invalid", split=split))
        records_by_split[split] = records
        database_record_counts[split] = len(records)
        if len(records) != expected_count:
            errors.append(
                _error(
                    "rio_database_count_mismatch",
                    split=split,
                    expected=expected_count,
                    observed=len(records),
                )
            )

        split_scenes: set[str] = set()
        for record in records:
            scene = _scene_from_record(record)
            if scene is None or scene in split_scenes:
                errors.append(_error("rio_database_scene_invalid", split=split))
                continue
            split_scenes.add(scene)
            numeric_stem = scene.removeprefix("scene")
            npy_path: Path | None = None
            instance_path: Path | None = None
            path_specs = (
                (
                    "filepath",
                    rio_processed_dir / split,
                    numeric_stem,
                    ".npy",
                    "rio_processed_npy",
                    seen_npy,
                ),
                (
                    "instance_gt_filepath",
                    rio_processed_dir / "instance_gt" / split,
                    scene,
                    ".txt",
                    "rio_instance_gt",
                    seen_gt,
                ),
            )
            for field, root, stem, suffix, code_prefix, seen in path_specs:
                value = record.get(field)
                lexical_path = Path(value) if isinstance(value, str) else None
                if lexical_path is None:
                    errors.append(_error(f"{code_prefix}_path_invalid", split=split))
                    continue
                if lexical_path.is_absolute() or ".." in lexical_path.parts:
                    resolved = lexical_path.resolve()
                    errors.append(
                        _error(f"{code_prefix}_path_noncanonical", split=split)
                    )
                else:
                    resolved = (REPO_ROOT / lexical_path).resolve()
                if field == "filepath":
                    npy_path = resolved
                else:
                    instance_path = resolved
                expected_root = root.resolve()
                if not resolved.is_relative_to(expected_root):
                    errors.append(
                        _error(f"{code_prefix}_path_outside_split", split=split)
                    )
                expected_lexical = (
                    expected_repo_prefix / root.relative_to(rio_processed_dir) / f"{stem}{suffix}"
                    if expected_repo_prefix is not None
                    else None
                )
                if expected_lexical is not None and lexical_path != expected_lexical:
                    errors.append(
                        _error(f"{code_prefix}_path_noncanonical", split=split)
                    )
                if resolved.stem != stem or resolved.suffix != suffix:
                    errors.append(
                        _error(f"{code_prefix}_scene_stem_mismatch", split=split)
                    )
                if resolved in seen:
                    errors.append(_error(f"{code_prefix}_path_reused", split=split))
                seen.add(resolved)
                if not resolved.is_file() or resolved.is_symlink():
                    errors.append(_error(f"{code_prefix}_missing", split=split))
            if (
                validate_content
                and npy_path is not None
                and instance_path is not None
                and npy_path.is_file()
                and instance_path.is_file()
                and not any(
                    error.get("split") == split
                    and error["code"].endswith(
                        ("path_outside_split", "path_noncanonical")
                    )
                    for error in errors
                )
            ):
                supervision_count_before = len(supervision_observed)
                content_codes = _validate_processed_record_assets(
                    record,
                    npy_path,
                    instance_path,
                    split,
                    allow_known_empty_supervision=True,
                    instance_gt_offset=0,
                    allow_unlabeled_instance_gt=True,
                    supervision_observed=supervision_observed,
                )
                errors.extend(
                    _error(f"rio_{code}", split=split, scene=scene)
                    for code in sorted(content_codes)
                )
                if len(supervision_observed) > supervision_count_before:
                    scene_supervision[scene] = supervision_observed[-1]
        scenes_by_split[split] = split_scenes

    combined_path = rio_processed_dir / "train_validation_database.yaml"
    try:
        combined = _read_yaml(combined_path)
    except (OSError, yaml.YAMLError):
        combined = None
    if combined != records_by_split["train"] + records_by_split["validation"]:
        errors.append(_error("rio_train_validation_database_mismatch"))

    sequence_path = rio_processed_dir / "sequence_database_sliding_2.yaml"
    try:
        sequences = _load_unique_yaml_mapping(sequence_path)
    except (OSError, TypeError, yaml.YAMLError):
        sequences = {}
        errors.append(_error("rio_sequence_database_invalid"))
    if len(sequences) != 1482:
        errors.append(
            _error(
                "rio_sequence_database_count_mismatch",
                expected=1482,
                observed=len(sequences),
            )
        )

    seen_change_paths: set[Path] = set()
    sequence_counts = {"train": 0, "validation": 0}
    endpoint_counts = {
        split: {scene: [0, 0] for scene in scenes}
        for split, scenes in scenes_by_split.items()
    }
    observed_unsupervised_sequences: list[str] = []
    for sequence, entry in sequences.items():
        if not isinstance(entry, Mapping) or entry.get("type") not in sequence_counts:
            continue
        split = str(entry["type"])
        sequence_counts[split] += 1
        names = str(sequence).split("-")
        if (
            len(names) != 2
            or any(not re.fullmatch(r"scene\d{4}_\d{2}", name) for name in names)
            or any(name not in scenes_by_split[split] for name in names)
        ):
            errors.append(_error("rio_sequence_endpoints_invalid", split=split))
        else:
            for index, name in enumerate(names):
                endpoint_counts[split][name][index] += 1
            try:
                expected_scene = int(names[0][5:9])
                expected_sub_scenes = [int(name[10:12]) for name in names]
            except (ValueError, IndexError):
                expected_scene = None
                expected_sub_scenes = None
            if (
                type(entry.get("scene")) is not int
                or entry.get("scene") != expected_scene
                or entry.get("sub_scenes") != expected_sub_scenes
            ):
                errors.append(_error("rio_sequence_metadata_mismatch", split=split))
            if validate_content and not any(
                scene_supervision.get(name, False) for name in names
            ):
                observed_unsupervised_sequences.append(str(sequence))

        filepath = entry.get("filepath")
        lexical_path = Path(filepath) if isinstance(filepath, str) else None
        if lexical_path is None:
            errors.append(_error("rio_change_gt_path_invalid", split=split))
            continue
        if lexical_path.is_absolute() or ".." in lexical_path.parts:
            resolved_change = lexical_path.resolve()
            errors.append(_error("rio_change_gt_path_noncanonical", split=split))
        else:
            resolved_change = (REPO_ROOT / lexical_path).resolve()
        change_root = (rio_processed_dir / "change_gt" / split).resolve()
        if not resolved_change.is_relative_to(change_root):
            errors.append(_error("rio_change_gt_path_outside_root", split=split))
        if expected_repo_prefix is not None:
            expected_change = (
                expected_repo_prefix / "change_gt" / split / f"{sequence}.txt"
            )
            if lexical_path != expected_change:
                errors.append(_error("rio_change_gt_path_noncanonical", split=split))
        if resolved_change.stem != str(sequence) or resolved_change.suffix != ".txt":
            errors.append(_error("rio_change_gt_scene_stem_mismatch", split=split))
        if resolved_change in seen_change_paths:
            errors.append(_error("rio_change_gt_path_reused", split=split))
        seen_change_paths.add(resolved_change)
        if not resolved_change.is_file() or resolved_change.is_symlink():
            errors.append(_error("rio_change_gt_missing", split=split))

    for split, expected_count in (("train", 1178), ("validation", 157)):
        if sequence_counts[split] != expected_count:
            errors.append(_error("rio_sequence_split_count_mismatch", split=split))
        if any(counts != [1, 1] for counts in endpoint_counts[split].values()):
            errors.append(_error("rio_sequence_endpoint_coverage_mismatch", split=split))

    if validate_content and not any(supervision_observed):
        errors.append(_error("rio_dataset_instance_supervision_empty"))
    if validate_content and observed_unsupervised_sequences:
        errors.append(
            _error(
                "rio_active_sequence_supervision_empty",
                sequence_count=len(observed_unsupervised_sequences),
                sequences=observed_unsupervised_sequences,
            )
        )
    rio_content_errors = any(
        error["code"].startswith("rio_processed_")
        or error["code"]
        in {
            "rio_dataset_instance_supervision_empty",
            "rio_active_sequence_supervision_empty",
        }
        for error in errors
    )

    return (
        {
            "status": "pass" if not errors else "fail",
            "database_record_counts": database_record_counts,
            "sequence_record_count": len(sequences),
            "content_validation": (
                "pass"
                if validate_content and not rio_content_errors
                else "not_run_diagnostic"
                if not validate_content
                else "fail"
            ),
            "supervised_record_count": (
                sum(supervision_observed) if validate_content else None
            ),
            "unsupervised_sequences": (
                observed_unsupervised_sequences if validate_content else None
            ),
        },
        errors,
    )


def _is_integer_column(values: np.ndarray, *, minimum: int, maximum: int) -> bool:
    return bool(
        values.size
        and np.isfinite(values).all()
        and np.equal(values, np.rint(values)).all()
        and values.min() >= minimum
        and values.max() <= maximum
    )


def _read_instance_gt(path: Path) -> np.ndarray:
    values = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return np.asarray(values, dtype=np.int64)


def _validate_processed_record_assets(
    record: Mapping[str, Any],
    npy_path: Path,
    instance_path: Path | None,
    split: str,
    *,
    allow_known_empty_supervision: bool = False,
    instance_gt_offset: int = 1,
    allow_unlabeled_instance_gt: bool = False,
    supervision_observed: list[bool] | None = None,
) -> set[str]:
    error_codes: set[str] = set()
    file_len = record.get("file_len")
    if type(file_len) is not int or file_len < 1:
        error_codes.add("processed_npy_file_len_invalid")
        file_len = None
    try:
        points = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError):
        return {"processed_npy_load_failed"}

    expected_columns = 10 if split == "test" else 12
    if (
        points.ndim != 2
        or points.shape[0] < 1
        or points.shape[1] != expected_columns
    ):
        error_codes.add("processed_npy_shape_invalid")
        return error_codes
    if points.dtype != np.dtype(np.float32):
        error_codes.add("processed_npy_dtype_invalid")
        return error_codes
    if file_len is not None and points.shape[0] != file_len:
        error_codes.add("processed_npy_file_len_mismatch")
    try:
        finite = bool(np.isfinite(points).all())
    except TypeError:
        error_codes.add("processed_npy_dtype_invalid")
        return error_codes
    if not finite:
        error_codes.add("processed_npy_nonfinite")
        return error_codes

    segments = np.asarray(points[:, 9])
    if not _is_integer_column(
        segments,
        minimum=0,
        maximum=max(points.shape[0] - 1, 0),
    ):
        error_codes.add("processed_npy_integer_columns_invalid")
        error_codes.add("processed_npy_value_range_invalid")

    if split == "test":
        return error_codes

    semantic_labels = np.asarray(points[:, 10])
    instance_labels = np.asarray(points[:, 11])
    semantic_values = np.rint(semantic_labels).astype(np.int64)
    semantic_ids_valid = np.isin(semantic_values, range(41)).all()
    if (
        not _is_integer_column(semantic_labels, minimum=-1, maximum=255)
        or not semantic_ids_valid
        or not _is_integer_column(instance_labels, minimum=-1, maximum=999)
    ):
        error_codes.add("processed_npy_integer_columns_invalid")
        error_codes.add("processed_npy_value_range_invalid")
        return error_codes
    if not np.any((semantic_labels > 0) & (semantic_labels != 255)):
        error_codes.add("processed_npy_supervision_empty")
    effective_instance_supervision = np.isin(
        semantic_values,
        NYU40_INSTANCE_IDS,
    ) & (instance_labels >= 0)
    if supervision_observed is not None:
        supervision_observed.append(bool(np.any(effective_instance_supervision)))
    if (
        not allow_known_empty_supervision
        and not np.any(effective_instance_supervision)
    ):
        error_codes.add("processed_npy_instance_supervision_empty")

    if instance_path is None or not instance_path.is_file():
        return error_codes
    try:
        instance_gt = _read_instance_gt(instance_path)
    except (OSError, UnicodeError, ValueError, OverflowError):
        error_codes.add("processed_instance_gt_invalid")
        return error_codes
    if file_len is not None and len(instance_gt) != file_len:
        error_codes.add("processed_instance_gt_length_mismatch")
    if len(instance_gt) != points.shape[0]:
        error_codes.add("processed_instance_gt_length_mismatch")
        return error_codes
    minimum_instance_gt = -1 if allow_unlabeled_instance_gt else 0
    if len(instance_gt) == 0 or np.any(instance_gt < minimum_instance_gt):
        error_codes.add("processed_instance_gt_invalid")
        return error_codes
    if allow_unlabeled_instance_gt:
        valid_negative = (
            (instance_gt == -1)
            & (semantic_values == 0)
            & (instance_labels == -1)
        )
        if np.any((instance_gt < 0) & ~valid_negative):
            error_codes.add("processed_instance_gt_invalid")
            return error_codes
    if not np.any(instance_gt > 0):
        error_codes.add("processed_instance_gt_supervision_empty")
    expected_gt = (
        np.rint(semantic_labels).astype(np.int64) * 1000
        + np.rint(instance_labels).astype(np.int64)
        + instance_gt_offset
    )
    if not np.array_equal(instance_gt, expected_gt):
        error_codes.add("processed_instance_gt_npy_mismatch")
    return error_codes


def _audit_processed_assets(
    processed_dir: Path,
    scenes_by_split: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    processed_root_ref = _portable_ref(processed_dir, "scannet_processed")
    if not processed_dir.is_dir():
        errors.append(_error("scannet_processed_root_missing"))

    required_global = ("label_database.yaml", "color_mean_std.yaml", "scannet.yaml")
    global_files = {
        name: {
            "reference": _join_ref(processed_root_ref, name),
            "present": (processed_dir / name).is_file(),
        }
        for name in required_global
    }
    for name, record in global_files.items():
        if not record["present"]:
            errors.append(_error("processed_global_file_missing", name=name))

    database_scene_count = 0
    npy_scene_count = 0
    instance_gt_scene_count = 0
    by_split: dict[str, Any] = {}
    seen_npy_paths: dict[Path, str] = {}
    seen_instance_paths: dict[Path, str] = {}
    for split in ("train", "validation", "test"):
        database_path = processed_dir / DATABASE_FILES[split]
        expected_scenes = list(scenes_by_split[split])
        database: list[Mapping[str, Any]] = []
        if database_path.is_file():
            try:
                loaded = _read_yaml(database_path)
                if not isinstance(loaded, list) or not all(isinstance(item, Mapping) for item in loaded):
                    raise TypeError
                database = loaded
            except (OSError, TypeError, yaml.YAMLError):
                errors.append(_error("processed_database_invalid", split=split))
        else:
            errors.append(_error("processed_database_missing", split=split))

        scene_records: dict[str, Mapping[str, Any]] = {}
        duplicate_scenes: set[str] = set()
        for record in database:
            scene = _scene_from_record(record)
            if scene is None:
                continue
            if scene in scene_records:
                duplicate_scenes.add(scene)
            scene_records[scene] = record
        if duplicate_scenes:
            errors.append(
                _error(
                    "processed_database_duplicate_scene",
                    split=split,
                    count=len(duplicate_scenes),
                )
            )
        if len(database) != len(expected_scenes):
            errors.append(
                _error(
                    "processed_database_count_mismatch",
                    split=split,
                    expected=len(expected_scenes),
                    observed=len(database),
                )
            )

        split_db_count = 0
        split_npy_count = 0
        split_instance_count = 0
        missing_database_scenes: list[str] = []
        missing_npy_scenes: list[str] = []
        missing_instance_scenes: list[str] = []
        invalid_scenes_by_code: dict[str, list[str]] = {}
        missing_examples: list[dict[str, Any]] = []
        for scene in expected_scenes:
            record = scene_records.get(scene)
            if record is None:
                missing_database_scenes.append(scene)
                if len(missing_examples) < 10:
                    missing_examples.append({"scene": scene, "missing": ["database_record"]})
                continue
            split_db_count += 1
            npy_path = _resolve_record_path(record.get("filepath"), processed_dir, split)
            instance_path = None
            if split != "test":
                instance_path = _resolve_record_path(
                    record.get("instance_gt_filepath"),
                    processed_dir / "instance_gt",
                    split,
                )
            validation_codes: set[str] = set()
            processed_ref = None
            if npy_path is not None:
                resolved_npy = npy_path.resolve()
                expected_npy_root = (processed_dir / split).resolve()
                expected_npy_stem = scene.removeprefix("scene")
                if not resolved_npy.is_relative_to(expected_npy_root):
                    validation_codes.add("processed_npy_path_outside_split")
                if resolved_npy.stem != expected_npy_stem:
                    validation_codes.add("processed_npy_scene_stem_mismatch")
                if resolved_npy in seen_npy_paths:
                    validation_codes.add("processed_npy_path_reused")
                else:
                    seen_npy_paths[resolved_npy] = scene
                try:
                    processed_ref = processed_dir.resolve().relative_to(
                        REPO_ROOT.resolve()
                    )
                except ValueError:
                    processed_ref = None
                if processed_ref is not None and Path(
                    str(record.get("filepath"))
                ) != processed_ref / split / f"{expected_npy_stem}.npy":
                    validation_codes.add("processed_npy_path_noncanonical")

            if split != "test" and instance_path is not None:
                resolved_instance = instance_path.resolve()
                expected_instance_root = (
                    processed_dir / "instance_gt" / split
                ).resolve()
                if not resolved_instance.is_relative_to(expected_instance_root):
                    validation_codes.add(
                        "processed_instance_gt_path_outside_split"
                    )
                if resolved_instance.stem != scene:
                    validation_codes.add(
                        "processed_instance_gt_scene_stem_mismatch"
                    )
                if resolved_instance in seen_instance_paths:
                    validation_codes.add("processed_instance_gt_path_reused")
                else:
                    seen_instance_paths[resolved_instance] = scene
                if processed_ref is not None and Path(
                    str(record.get("instance_gt_filepath"))
                ) != processed_ref / "instance_gt" / split / f"{scene}.txt":
                    validation_codes.add(
                        "processed_instance_gt_path_noncanonical"
                    )

            if npy_path is None or not npy_path.is_file() or npy_path.suffix != ".npy":
                missing_npy_scenes.append(scene)
                if len(missing_examples) < 10:
                    missing_examples.append({"scene": scene, "missing": ["npy"]})
            else:
                if not any(
                    code.endswith("path_outside_split")
                    for code in validation_codes
                ):
                    validation_codes.update(
                        _validate_processed_record_assets(
                            record,
                            npy_path,
                            instance_path,
                            split,
                            allow_known_empty_supervision=(
                                scene in {"scene0154_00", "scene0636_00"}
                            ),
                        )
                    )
                if not any(code.startswith("processed_npy_") for code in validation_codes):
                    split_npy_count += 1
                if validation_codes and len(missing_examples) < 10:
                    missing_examples.append(
                        {"scene": scene, "invalid": sorted(validation_codes)}
                    )
            for code in validation_codes:
                invalid_scenes_by_code.setdefault(code, []).append(scene)
            if split != "test":
                if instance_path is None or not instance_path.is_file():
                    missing_instance_scenes.append(scene)
                    if len(missing_examples) < 10:
                        missing_examples.append({"scene": scene, "missing": ["instance_gt"]})
                elif (
                    npy_path is not None
                    and npy_path.is_file()
                    and npy_path.suffix == ".npy"
                    and not validation_codes
                ):
                    split_instance_count += 1

        if missing_database_scenes:
            errors.append(
                _scene_error("processed_database_scene_missing", split, missing_database_scenes)
            )
        if missing_npy_scenes:
            errors.append(_scene_error("processed_npy_missing", split, missing_npy_scenes))
        if missing_instance_scenes:
            errors.append(
                _scene_error("processed_instance_gt_missing", split, missing_instance_scenes)
            )
        for code, scenes in sorted(invalid_scenes_by_code.items()):
            errors.append(_scene_error(code, split, scenes))

        database_scene_count += split_db_count
        npy_scene_count += split_npy_count
        instance_gt_scene_count += split_instance_count
        expected_instance_count = 0 if split == "test" else len(expected_scenes)
        split_pass = (
            len(database) == len(expected_scenes)
            and split_db_count == len(expected_scenes)
            and split_npy_count == len(expected_scenes)
            and split_instance_count == expected_instance_count
            and not duplicate_scenes
        )
        by_split[split] = {
            "database_ref": _join_ref(processed_root_ref, DATABASE_FILES[split]),
            "expected_scene_count": len(expected_scenes),
            "database_record_count": len(database),
            "database_scene_count": split_db_count,
            "npy_scene_count": split_npy_count,
            "instance_gt_scene_count": split_instance_count,
            "missing_examples": missing_examples,
            "status": "pass" if split_pass else "fail",
        }

    taxonomy, taxonomy_errors = _audit_taxonomy(processed_dir)
    errors.extend(taxonomy_errors)
    expected_total = sum(len(scenes) for scenes in scenes_by_split.values())
    processed = {
        "root_ref": processed_root_ref,
        "global_files": global_files,
        "expected_scene_count": expected_total,
        "database_scene_count": database_scene_count,
        "npy_scene_count": npy_scene_count,
        "instance_gt_scene_count": instance_gt_scene_count,
        "by_split": by_split,
        "status": "pass" if not errors else "fail",
    }
    return processed, taxonomy, errors


def _augmentation_snapshot() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, filename in (
        ("image", "albumentations_aug.yaml"),
        ("volume", "volumentations_aug.yaml"),
    ):
        path = REPO_ROOT / "conf" / "augmentation" / filename
        try:
            config = _read_yaml(path)
            transforms = config["transform"]["transforms"]
            output[key] = {
                "reference": f"repo:conf/augmentation/{filename}",
                "schema_version": str(config.get("__version__", "unreported")),
                "transforms": [
                    str(transform.get("__class_fullname__", "unknown")).rsplit(".", 1)[-1]
                    for transform in transforms
                ],
            }
        except (OSError, TypeError, KeyError, yaml.YAMLError):
            output[key] = {
                "reference": f"repo:conf/augmentation/{filename}",
                "schema_version": "unresolved",
                "transforms": [],
            }
    return output


def _compose_config_snapshot() -> tuple[
    dict[str, Any],
    dict[str, Any],
    bool,
    str | None,
    Any | None,
]:
    try:
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base="1.2"):
            p2 = compose(config_name="config_p2_rescene4d_concerto_t2")
            base = compose(config_name="config_base_instance_segmentation")

        def snapshot(cfg: Any) -> dict[str, Any]:
            datasets = cfg.data.train_dataset.datasets
            return {
                "backbone": "Concerto" if cfg.backbone.model_lib == "concerto" else str(cfg.backbone.model_lib),
                "backbone_checkpoint_config": {
                    "name": str(cfg.backbone.name),
                    "repo_id": str(cfg.backbone.repo_id),
                },
                "encoder_freeze": str(cfg.general.freeze),
                "decoder_trainability": "trainable",
                "frozen_encoder_runtime": FROZEN_ENCODER_RUNTIME
                if str(cfg.general.freeze) == "backbone_encoder"
                else {
                    "parameters_require_grad": True,
                    "module_mode": "train",
                    "drop_path_rate": 0.3,
                },
                "num_queries": int(cfg.model.num_queries),
                "non_parametric_queries": bool(cfg.model.non_parametric_queries),
                "query_initialization": "FPS non-parametric"
                if cfg.model.non_parametric_queries
                else "learned",
                "temporal_window": {
                    str(dataset.dataset_name): int(dataset.temporal_window)
                    for dataset in datasets
                },
                "contrastive": bool(cfg.loss.contrastive_loss),
                "st_serialization": list(cfg.backbone.decoder_serializations),
                "st_masking": bool(cfg.model.temporal_masking),
                "voxel_size_m": float(cfg.data.voxel_size),
                "loss_weights": {
                    "class": float(cfg.matcher.cost_class),
                    "mask_bce": float(cfg.matcher.cost_mask),
                    "dice": float(cfg.matcher.cost_dice),
                },
                "no_object_weight": float(cfg.loss.eos_coef),
                "optimizer": str(cfg.optimizer._target_).rsplit(".", 1)[-1],
                "scheduler": str(cfg.scheduler.scheduler._target_).rsplit(".", 1)[-1],
                "max_lr": float(cfg.scheduler.scheduler.max_lr),
                "epochs": int(cfg.trainer.max_epochs),
                "gpus": int(cfg.general.gpus),
                "batch_per_gpu": int(cfg.data.batch_size),
                "gradient_accumulation": int(cfg.trainer.accumulate_grad_batches),
                "effective_batch_size": int(cfg.general.gpus)
                * int(cfg.data.batch_size)
                * int(cfg.trainer.accumulate_grad_batches),
                "precision": str(cfg.trainer.get("precision", "not explicit")),
                "dataset_mix": [
                    {
                        "dataset": "3RScan" if dataset.dataset_name == "rio" else "ScanNet",
                        "temporal_window": int(dataset.temporal_window),
                        "weight": float(weight),
                    }
                    for dataset, weight in zip(datasets, cfg.data.train_dataset.weights)
                ],
                "adamw_implicit_defaults": ADAMW_IMPLICIT_DEFAULTS,
                "onecycle_implicit_defaults": ONECYCLE_IMPLICIT_DEFAULTS,
                "augmentations": _augmentation_snapshot(),
                "evaluation_taxonomy": NYU40_INSTANCE_IDS,
                "sequence_database": {
                    "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml",
                    "sha256": _sha256_file(
                        REPO_ROOT
                        / "data"
                        / "processed"
                        / "rio"
                        / "sequence_database_sliding_2.yaml"
                    ),
                    "mode": "sliding",
                    "temporal_window": 2,
                },
                "metric_config": "repo:conf/metrics/tmap.yaml",
                "seed": None if cfg.general.seed is None else int(cfg.general.seed),
            }

        return snapshot(p2), snapshot(base), True, None, p2
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - preserve partial evidence.
        return {}, {}, False, type(exc).__name__, None


def _setting(official: Any, reproduction: Any, repository_default: Any = None, status: str | None = None) -> dict[str, Any]:
    if status is None:
        status = "match" if official == reproduction else "deviation"
    return {
        "official": official,
        "reproduction": reproduction,
        "repository_default": repository_default,
        "status": status,
    }


def _build_config_diff(
    reproduction: Mapping[str, Any],
    repository_default: Mapping[str, Any],
    composed: bool,
    error_type: str | None,
    model_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    get = reproduction.get
    base = repository_default.get
    official_mix = REPRODUCTION_TARGET["data"]["mix"]
    official_temporal = {"rio": 2, "scannet": 1}
    checkpoint_verified = model_checkpoint.get("status") == "pass"
    if checkpoint_verified:
        checkpoint_choice = {
            "reference": model_checkpoint.get("reference"),
            "revision": CONCERTO_REVISION,
            "sha256": model_checkpoint.get("observed_sha256"),
            "license": "CC-BY-NC-4.0",
        }
    else:
        checkpoint_choice = {
            "reference": model_checkpoint.get("reference"),
            "revision": CONCERTO_REVISION,
            "expected_sha256": model_checkpoint.get("expected_sha256"),
            "observed_sha256": model_checkpoint.get("observed_sha256"),
            "expected_byte_size": model_checkpoint.get("expected_byte_size"),
            "observed_byte_size": model_checkpoint.get("observed_byte_size"),
            "verification_status": model_checkpoint.get("status", "fail"),
            "license": "CC-BY-NC-4.0",
        }
    frozen_runtime = get("frozen_encoder_runtime")
    if isinstance(frozen_runtime, Mapping):
        frozen_runtime = {
            "parameters_require_grad": frozen_runtime.get("parameters_require_grad"),
            "module_mode": frozen_runtime.get("module_mode"),
            "drop_path_rate": frozen_runtime.get("concerto_drop_path_rate"),
            "decoder_and_head_trainable": frozen_runtime.get("decoder_and_head_trainable"),
        }
    settings = {
        "backbone": _setting("Concerto", get("backbone"), base("backbone")),
        "backbone_checkpoint": _setting(
            {
                "encoder": "Concerto pretrained",
                "exact_revision": "not reported",
                "exact_weight_hash": "not reported",
            },
            checkpoint_choice if composed else None,
            base("backbone_checkpoint_config"),
            "verified_reproduction_choice"
            if composed and checkpoint_verified
            else "unverified",
        ),
        "encoder_freeze": _setting("backbone_encoder", get("encoder_freeze"), base("encoder_freeze")),
        "decoder_trainability": _setting("trainable", get("decoder_trainability"), base("decoder_trainability")),
        "frozen_encoder_runtime": _setting(
            {
                "parameters_require_grad": False,
                "module_mode": "not reported",
                "drop_path_rate": "not reported",
            },
            frozen_runtime,
            base("frozen_encoder_runtime"),
            "repository_behavior_risk" if frozen_runtime else "unverified",
        ),
        "num_queries": _setting(100, get("num_queries"), base("num_queries")),
        "query_initialization": _setting(
            "FPS non-parametric", get("query_initialization"), base("query_initialization")
        ),
        "temporal_window": _setting(official_temporal, get("temporal_window"), base("temporal_window")),
        "contrastive": _setting(True, get("contrastive"), base("contrastive")),
        "st_serialization": _setting(
            ["standard", "temporal_overlay"],
            get("st_serialization"),
            base("st_serialization"),
        ),
        "st_masking": _setting(False, get("st_masking"), base("st_masking")),
        "voxel_size_m": _setting(0.02, get("voxel_size_m"), base("voxel_size_m")),
        "loss_weights": _setting(
            {"class": 2.0, "mask_bce": 5.0, "dice": 2.0},
            get("loss_weights"),
            base("loss_weights"),
        ),
        "no_object_weight": _setting(0.2, get("no_object_weight"), base("no_object_weight")),
        "optimizer": _setting("AdamW", get("optimizer"), base("optimizer")),
        "scheduler": _setting("OneCycleLR", get("scheduler"), base("scheduler")),
        "adamw_implicit_defaults": _setting(
            "not reported",
            get("adamw_implicit_defaults"),
            base("adamw_implicit_defaults"),
            "verified_reproduction_choice"
            if get("adamw_implicit_defaults")
            else "unverified",
        ),
        "onecycle_implicit_defaults": _setting(
            "not reported",
            get("onecycle_implicit_defaults"),
            base("onecycle_implicit_defaults"),
            "verified_reproduction_choice"
            if get("onecycle_implicit_defaults")
            else "unverified",
        ),
        "max_lr": _setting(0.0005, get("max_lr"), base("max_lr")),
        "epochs": _setting(450, get("epochs"), base("epochs")),
        "effective_batch_size": _setting(
            32, get("effective_batch_size"), base("effective_batch_size")
        ),
        "precision": _setting(
            "not reported",
            get("precision"),
            base("precision"),
            "explicit_reproduction_choice" if get("precision") == "32-true" else "unverified",
        ),
        "dataset_mix": _setting(official_mix, get("dataset_mix"), base("dataset_mix")),
        "augmentations": _setting(
            {
                "sequence_scope": "same rotation/scaling across the registered sequence",
                "exact_transform_list": "not reported",
                "serialized_config_versions": "not reported",
            },
            get("augmentations"),
            base("augmentations"),
            "paper_exactness_unverified" if get("augmentations") else "unverified",
        ),
        "evaluation_taxonomy": _setting(
            NYU40_INSTANCE_IDS, get("evaluation_taxonomy"), base("evaluation_taxonomy")
        ),
        "sequence_database": _setting(
            {
                "construction": "randomly ordered sliding windows",
                "temporal_window": 2,
                "exact_yaml_hash": "not reported",
            },
            get("sequence_database"),
            base("sequence_database"),
            "match"
            if get("sequence_database", {}).get("sha256") == SEQUENCE_DB_SHA256
            else "deviation",
        ),
        "metric_config": _setting(
            "t-mAP + overall AP + stage AP",
            get("metric_config"),
            base("metric_config"),
            "match" if get("metric_config") else "unverified",
        ),
        "seed": _setting(45, get("seed"), base("seed")),
    }
    implementation_fixes = [
        {
            "id": "weighted_segmentation_objective",
            "classification": "local_paper_alignment_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_TRAINING_CONTRACT_FIX_COMMIT,
            "upstream_behavior": "training and validation returned sum(losses.values()) even though criterion.weight_dict was constructed",
            "upstream_semantic_effect": "configured class/mask_bce/dice weights 2/5/2 were not applied to the optimized objective; effective weights were 1/1/1",
            "local_behavior": "training and validation use criterion.weight_dict, applying 2/5/2 to final and auxiliary segmentation losses",
            "paper_alignment": "restores the reported class/mask BCE/Dice objective ratio 2/5/2",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/trainer/trainer.py",
                "repo:trainer/trainer.py",
                "repo:tests/test_p2_training_contract.py",
            ],
        },
        {
            "id": "contrastive_diagnostic_deduplication",
            "classification": "local_paper_alignment_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_TRAINING_CONTRACT_FIX_COMMIT,
            "upstream_behavior": "criterion emitted aggregate and per-layer contrastive values, and raw loss summation optimized every emitted value",
            "upstream_semantic_effect": "aggregate contrastive losses and their per-layer diagnostic losses were both summed, double-counting the contrastive objective",
            "local_behavior": "aggregate contrastive objectives are optimized exactly once; per-layer values remain logging-only diagnostics",
            "paper_alignment": "counts each aggregate temporal contrastive loss exactly once while preserving per-layer observability",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/models/criterion.py",
                "external:github/GradientSpaces/rescene4d@fb2fe42/trainer/trainer.py",
                "repo:trainer/trainer.py",
                "repo:tests/test_p2_training_contract.py",
            ],
        },
        {
            "id": "hydra_contrastive_override_order",
            "classification": "local_paper_alignment_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_TRAINING_CONTRACT_FIX_COMMIT,
            "upstream_behavior": "the optional contrastive config was composed before set_criterion in the Hydra defaults list",
            "upstream_semantic_effect": "loss/contrastive=infoNCE was overwritten by the later set_criterion default, leaving loss.contrastive_loss=false",
            "local_behavior": "set_criterion composes first and loss/contrastive=infoNCE composes afterward, resolving loss.contrastive_loss=true",
            "paper_alignment": "ensures the paper-required temporal contrastive loss is active",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/conf/config_base_instance_segmentation.yaml",
                "repo:conf/config_base_instance_segmentation.yaml",
                "repo:tests/test_p2_training_contract.py",
            ],
        },
        {
            "id": "fail_closed_dataset_sequence_mix_validation",
            "classification": "local_reproduction_safety_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_RUNTIME_SAFETY_FIX_COMMIT,
            "upstream_behavior": "missing split databases called bare exit(), temporal database exceptions were printed and swallowed, and zero-length child datasets were skipped",
            "upstream_semantic_effect": "a missing or empty ScanNet child could terminate with status 0 or be omitted from sampling, while an unknown temporal scan could leave zero-filled indices",
            "local_behavior": "every configured split database and temporal sequence mapping must be non-empty; every sequence scan must resolve; every mixed child must be non-empty; dataset and weight lengths must match and weights must be finite and positive",
            "safety_effect": "prevents missing ScanNet from degrading the required mix to RIO-only and prevents unknown temporal scans from becoming zero indices",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/datasets/semseg.py",
                "external:github/GradientSpaces/rescene4d@fb2fe42/datasets/multi_dataset.py",
                "repo:datasets/semseg.py",
                "repo:datasets/multi_dataset.py",
                "repo:tests/test_p2_data_failure.py",
                "repo:tests/test_temporal_loader.py",
            ],
        },
        {
            "id": "ddp_batch_contract_consensus",
            "classification": "local_reproduction_safety_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_RUNTIME_SAFETY_FIX_COMMIT,
            "upstream_behavior": "training returned None for empty targets or the recognized single-point cross-attention error; evaluation returned 0.0 or None for empty coordinates or that error, without cross-rank consensus",
            "upstream_semantic_effect": "asymmetric rank-local skipping could desynchronize optimizer and scheduler progress or hang later DDP collectives",
            "local_behavior": "staged input, recursive output, criterion, objective, and evaluation consensus cover train, validation, and test microbatches; a pre-optimizer gradient-finiteness consensus prevents parameter updates after non-finite gradients",
            "collective_contract": {
                "normal_train_microbatch": {
                    "safety_int32_max_all_reduce_count": 3,
                    "criterion_float_num_masks_all_reduce_count": 1,
                    "total_all_reduce_count": 4,
                    "all_gather_object_count": 0,
                },
                "normal_validation_microbatch": {
                    "safety_int32_max_all_reduce_count": 4,
                    "criterion_float_num_masks_all_reduce_count": 1,
                    "total_all_reduce_count": 5,
                    "all_gather_object_count": 0,
                },
                "normal_test_microbatch": {
                    "safety_int32_max_all_reduce_count": 3,
                    "criterion_float_num_masks_all_reduce_count": 0,
                    "total_all_reduce_count": 3,
                    "all_gather_object_count": 0,
                },
                "covered_stage_failure": {
                    "additional_all_gather_object_count": 1,
                },
                "train_optimizer_step_accumulation_4": {
                    "safety_int32_max_all_reduce_count": 13,
                    "optimizer_gradient_int32_max_all_reduce_count": 1,
                    "criterion_float_num_masks_all_reduce_count": 4,
                    "total_all_reduce_count": 17,
                },
            },
            "performance_cost": "three blocking scalar int32 MAX all_reduce operations per normal train DDP microbatch, four per validation microbatch, and three per test microbatch; train accumulation=4 costs twelve microbatch safety all-reduces, one optimizer-gradient safety all-reduce, and four criterion float num_masks all-reduces per optimizer step (17 total); all_gather_object adds one call only on a covered stage failure",
            "coverage_boundary": "consensus covers input, recursive model output, criterion, objective, evaluation metadata and metrics, plus pre-optimizer gradient finiteness; exceptions outside those covered stages retain native behavior",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/trainer/trainer.py",
                "repo:trainer/trainer.py",
                "repo:tests/test_p2_ddp_batch_contract.py",
            ],
        },
        {
            "id": "full_state_checkpoint_resume_selection",
            "classification": "local_reproduction_safety_fix",
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_RUNTIME_SAFETY_FIX_COMMIT,
            "upstream_behavior": "resume selection checked deserialization of last or last-epoch and a filename-metric fallback, then allowed starting from scratch when candidates were corrupt",
            "upstream_semantic_effect": "weights-only or partial checkpoints could pass the load check, stale last metadata could win, and an all-corrupt checkpoint directory could silently reset training",
            "local_behavior": "statically validates required Lightning full-state fields, selects the latest valid candidate by checkpoint epoch/global_step and numeric filename version, and refuses to start from scratch when checkpoint files exist but all are invalid",
            "restore_boundary": "static validation is not a real Lightning restore; trainer.fit is attempted once with the selected checkpoint and does not automatically retry another candidate after a Lightning restore failure",
            "startup_cost": "each checkpoint candidate is deserialized on CPU once during static selection",
            "evidence_refs": [
                "external:github/GradientSpaces/rescene4d@fb2fe42/main_instance_segmentation.py",
                "repo:main_instance_segmentation.py",
                "repo:tests/test_p2_checkpoint_selection.py",
            ],
        },
    ]
    return {
        "schema_version": 1,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "config_ref": "repo:conf/config_p2_rescene4d_concerto_t2.yaml",
        "base_config_ref": "repo:conf/config_base_instance_segmentation.yaml",
        "config_composed": composed,
        "compose_error_type": error_type,
        "settings": settings,
        "reproduction_code_relation": {
            "official_code_commit": OFFICIAL_SOURCE_COMMIT,
            "local_fix_commit": P2_TRAINING_CONTRACT_FIX_COMMIT,
            "runtime_safety_fix_commit": P2_RUNTIME_SAFETY_FIX_COMMIT,
            "official_code_used_unchanged": False,
            "status": "local_alignment_and_safety_patch_set",
        },
        "implementation_fixes": implementation_fixes,
        "declared_deviations": [
            {
                "setting": "backbone_checkpoint",
                "reason": "The paper does not report the exact Concerto revision or weight hash; the reproduction pins and verifies one licensed artifact.",
            },
            {
                "setting": "frozen_encoder_runtime",
                "reason": "Encoder parameters have requires_grad=false, but the encoder module remains in train mode and Concerto drop_path=0.3 remains active.",
            },
            {
                "setting": "adamw_implicit_defaults",
                "reason": "The paper reports AdamW but not betas, epsilon, weight decay, or AMSGrad; PyTorch 2.6 defaults are locked as a reproduction choice.",
            },
            {
                "setting": "onecycle_implicit_defaults",
                "reason": "The paper reports OneCycle and max LR only; PyTorch 2.6 curve and momentum defaults are locked as a reproduction choice.",
            },
            {
                "setting": "augmentations",
                "reason": "The paper reports shared sequence rotation/scaling but not the exact transform list; repository color, scale, and three-axis rotation transforms are retained with serialized config versions recorded.",
            },
            {
                "setting": "precision",
                "reason": "Official precision is not reported; reproduction locks FP32.",
            },
            {
                "setting": "hardware_topology",
                "reason": "Local recommendation is 2 A40 GPUs with accumulation; official training used 8 H100 GPUs.",
            },
        ],
    }


def _source_ref_from_manifest(entry: Mapping[str, Any], fallback: str) -> str:
    repository_url = str(entry.get("repository_url", ""))
    match = re.search(r"github\.com/([^?#]+)", repository_url)
    if not match:
        return fallback
    repository = match.group(1).rstrip("/").removesuffix(".git")
    return f"external:github/{repository}"


def _package_version(
    candidates: Sequence[str], fallback: str = "unknown"
) -> str:
    for distribution in candidates:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return fallback


def _format_cudnn_version(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value // 10000}.{(value % 10000) // 100}.{value % 100}"


def _format_nccl_version(value: Any) -> str:
    if isinstance(value, tuple):
        return ".".join(str(component) for component in value)
    if isinstance(value, int):
        return f"{value // 10000}.{(value % 10000) // 100}.{value % 100}"
    return "unknown"


def _measure_gpu(fallback: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = [
            [part.strip() for part in line.split(",")]
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        if rows and all(len(row) == 3 for row in rows):
            names = {row[0] for row in rows}
            memories = {int(float(row[1])) for row in rows}
            drivers = {row[2] for row in rows}
            if len(names) == len(memories) == len(drivers) == 1:
                return {
                    "count": len(rows),
                    "model": names.pop(),
                    "memory_mib": memories.pop(),
                    "driver": drivers.pop(),
                }
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return {
        key: value
        for key, value in fallback.items()
        if key in {"count", "model", "memory_mib", "driver"}
    }


def _measure_runtime_environment(source_runtime: Mapping[str, Any]) -> dict[str, Any]:
    source_packages = source_runtime.get("runtime_packages", {})
    if not isinstance(source_packages, Mapping):
        source_packages = {}
    packages = {
        "pytorch_lightning": _package_version(("pytorch-lightning", "lightning")),
        "hydra_core": _package_version(("hydra-core",)),
        "spconv": _package_version(
            ("spconv-cu126", "spconv"), str(source_packages.get("spconv-cu126", "unknown"))
        ),
        "flash_attn": _package_version(
            ("flash-attn",), str(source_packages.get("flash_attn", "unknown"))
        ),
        "torch_scatter": _package_version(
            ("torch-scatter",), str(source_packages.get("torch_scatter", "unknown"))
        ),
        "sonata": _package_version(
            ("sonata",), str(source_packages.get("sonata", "unknown"))
        ),
        "detectron2": _package_version(
            ("detectron2",), str(source_packages.get("detectron2", "unknown"))
        ),
        "concerto": _package_version(("concerto",)),
        "stmetrics": _package_version(
            ("stmetrics",), str(source_packages.get("stmetrics", "unknown"))
        ),
        "wandb": _package_version(("wandb",)),
    }
    torch_version = str(source_runtime.get("torch", "unknown"))
    cuda_version = str(source_runtime.get("cuda", "unknown"))
    cudnn_version = "unknown"
    nccl_version = "unknown"
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda)
        cudnn_version = _format_cudnn_version(torch.backends.cudnn.version())
        nccl_version = _format_nccl_version(torch.cuda.nccl.version())
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    fallback_gpu = source_runtime.get("gpu", {})
    if not isinstance(fallback_gpu, Mapping):
        fallback_gpu = {}
    return {
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda": cuda_version,
        "cudnn": cudnn_version,
        "nccl": nccl_version,
        "runtime_packages": packages,
        "gpu": _measure_gpu(fallback_gpu),
    }


def _build_environment_manifest(
    model_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = REPO_ROOT / "artifacts" / "environment" / "source_manifest.json"
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        source = {}
    runtime = source.get("runtime_environment", {})
    third_party_source = source.get("third_party_sources", {})
    fallback_commits = {
        "concerto": "10a7d17cff4dddff028f1522c2e72de4c4515df7",
        "sonata": "18c09ff8d713494f78a8213792262b910977a65d",
        "detectron2": "b4a4a3bd136852dae5fb1de37978dee412653e31",
        "stmetrics": "640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f",
        "scannet_tools": "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c",
    }
    paths = {
        "concerto": REPO_ROOT / "third_party" / "concerto",
        "sonata": REPO_ROOT / "third_party" / "sonata",
        "detectron2": REPO_ROOT / "third_party" / "detectron2",
        "stmetrics": REPO_ROOT / "third_party" / "stmetrics",
        "scannet_tools": REPO_ROOT / "third_party" / "ScanNet",
    }
    fallback_refs = {
        "concerto": "external:github/Pointcept/Concerto",
        "sonata": "external:github/facebookresearch/sonata",
        "detectron2": "external:github/facebookresearch/detectron2",
        "stmetrics": "external:github/GradientSpaces/stmetrics",
        "scannet_tools": "external:github/ScanNet/ScanNet",
    }
    third_party = {
        name: {
            "source_ref": _source_ref_from_manifest(
                third_party_source.get(name, {}), fallback_refs[name]
            ),
            "commit": _git_revision(path, fallback_commits[name]),
            **(
                {"license": third_party_source[name]["license"]}
                if name in third_party_source and "license" in third_party_source[name]
                else {}
            ),
        }
        for name, path in paths.items()
    }
    return {
        "schema_version": 1,
        "manifest_source_ref": "repo:artifacts/environment/source_manifest.json",
        "workspace": {
            "source_ref": "repo:.",
            "commit": _git_revision(REPO_ROOT, "unknown"),
            "branch": "research/p2-rescene4d-t2-repro",
        },
        "official_source": {
            "source_ref": "external:github/GradientSpaces/rescene4d",
            "paper_ref": "external:arxiv/2601.11508v2",
            "commit": OFFICIAL_SOURCE_COMMIT,
        },
        "third_party_sources": third_party,
        "model_weights": {
            "concerto": {
                "reference": model_checkpoint.get("reference"),
                "revision": CONCERTO_REVISION,
                "expected_sha256": model_checkpoint.get("expected_sha256"),
                "observed_sha256": model_checkpoint.get("observed_sha256"),
                "expected_byte_size": model_checkpoint.get("expected_byte_size"),
                "observed_byte_size": model_checkpoint.get("observed_byte_size"),
                "status": model_checkpoint.get("status", "fail"),
                "license": source.get("model_weights", {})
                .get("concerto", {})
                .get("license", "CC-BY-NC-4.0"),
            },
            "rescene": {
                "complete_checkpoint_included": False,
                "upstream_status": "Coming soon",
            },
        },
        "runtime_environment": _measure_runtime_environment(runtime),
    }


def _instantiate_real_mix(
    processed_scannet_dir: Path,
    rio_processed_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import hydra
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base="1.2"):
            config = compose(config_name="config_p2_rescene4d_concerto_t2")
        with open_dict(config):
            train_config = config.data.train_dataset
            train_config.image_augmentations_path = str(
                REPO_ROOT / "conf" / "augmentation" / "albumentations_aug.yaml"
            )
            train_config.volume_augmentations_path = str(
                REPO_ROOT / "conf" / "augmentation" / "volumentations_aug.yaml"
            )
            rio_config, scannet_config = train_config.datasets
            rio_config.data_dir = str(rio_processed_dir)
            rio_config.label_db_filepath = str(rio_processed_dir / "label_database.yaml")
            rio_config.color_mean_std = str(rio_processed_dir / "color_mean_std.yaml")
            scannet_config.data_dir = str(processed_scannet_dir)
            scannet_config.label_db_filepath = str(processed_scannet_dir / "label_database.yaml")
            scannet_config.color_mean_std = str(processed_scannet_dir / "color_mean_std.yaml")
        dataset = hydra.utils.instantiate(config.data.train_dataset)
        result = {
            "attempted": True,
            "status": "pass",
            "implementation": f"{dataset.__class__.__module__}.{dataset.__class__.__name__}",
            "dataset_names": [child.dataset_name for child in dataset.datasets],
            "dataset_sizes": [len(child) for child in dataset.datasets],
            "weights": [float(weight) for weight in dataset.weights],
            "temporal_windows": [int(child.temporal_window) for child in dataset.datasets],
            "sampler": dataset.sampler.__class__.__name__,
        }
        expected = {
            "implementation": "datasets.multi_dataset.MultiDataset",
            "dataset_names": ["rio", "scannet"],
            "weights": [1.0, 0.8],
            "temporal_windows": [2, 1],
            "sampler": "WeightedRandomSampler",
        }
        if any(result[key] != value for key, value in expected.items()) or any(
            size <= 0 for size in result["dataset_sizes"]
        ):
            result["status"] = "fail"
            return result, [_error("mixed_dataset_contract_mismatch")]
        return result, []
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - audit third-party construction.
        return (
            {
                "attempted": True,
                "status": "fail",
                "exception_type": type(exc).__name__,
            },
            [_error("mixed_dataset_instantiation_failed", exception_type=type(exc).__name__)],
        )


def _config_audit_markdown(diff: Mapping[str, Any], preflight: Mapping[str, Any]) -> str:
    lines = [
        "# P2 ReScene4D-C T=2 Configuration Audit",
        "",
        f"Data gate: `{preflight['status']}`",
        f"Formal P2 training authorized: `{str(preflight['formal_p2_training_authorized']).lower()}`",
        "",
        "| Setting | Official target | P2 reproduction | Repository default | Status |",
        "|---|---|---|---|---|",
    ]
    for name, record in diff["settings"].items():
        values = [record[key] for key in ("official", "reproduction", "repository_default")]
        rendered = [json.dumps(value, sort_keys=True, ensure_ascii=True) for value in values]
        lines.append(
            f"| `{name}` | `{rendered[0]}` | `{rendered[1]}` | `{rendered[2]}` | `{record['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Code-Level Paper Alignment Fixes",
            "",
            f"This reproduction is not an unchanged checkout of official code commit `{diff['reproduction_code_relation']['official_code_commit']}`. Local commit `{diff['reproduction_code_relation']['local_fix_commit']}` applies the following paper-alignment fixes:",
            "",
            "- Weighted segmentation objective: upstream training and validation used `sum(losses.values())`, so configured 2/5/2 matcher weights did not scale the optimized losses and effective weights were `1/1/1`. The local reducer applies `criterion.weight_dict` to final and auxiliary segmentation losses.",
            "- Contrastive diagnostic deduplication: upstream raw summation included aggregate contrastive values and their per-layer diagnostics. The local reducer keeps per-layer values for logging while aggregate contrastive objectives are optimized exactly once.",
            "- Hydra contrastive override order: upstream composed `loss/contrastive=infoNCE` before `set_criterion`, so the latter restored `loss.contrastive_loss=false`. The local defaults order composes the optional contrastive override last and resolves it to true.",
            "",
            "## Local Reproduction Safety Fixes",
            "",
            f"These are local reproduction safety fixes, not paper-alignment loss fixes, and are not unchanged behavior from official code commit `{diff['reproduction_code_relation']['official_code_commit']}`. They are bound to local runtime safety commit `{diff['reproduction_code_relation']['runtime_safety_fix_commit']}`.",
            "",
            "- Fail-closed data validation: split and temporal databases, sequence scan references, every configured mixed child, and sampling weights are validated before sampling. This prevents missing ScanNet from degrading the required mix to RIO-only and prevents an unknown temporal scan from retaining zero indices.",
            "- DDP batch-contract consensus: covered input, recursive output, criterion, objective, evaluation, and gradient failures raise across ranks instead of returning `None` or updating non-finite parameters. The normal path adds three scalar int32 MAX all-reduces per train microbatch, four per validation and three per test microbatch. At train accumulation=4, this is twelve microbatch safety plus one optimizer-gradient safety and four criterion float num_masks all-reduces per optimizer step (17 total); all_gather_object only on a covered failure. This is a deliberate performance cost.",
            "- Full-state checkpoint selection: candidates receive static Lightning state validation, latest selection uses checkpoint epoch/global_step plus numeric filename version metadata, and an all-corrupt directory refuses a silent fresh start. Static validation is not a real Lightning restore; `trainer.fit` does not automatically retry another candidate after a restore failure.",
            "",
            "## Data Gate Evidence",
            "",
            f"- Split metadata: `{preflight['split_metadata_status']}`; expected train/validation/test = 1201/312/100 by default.",
            f"- Raw ScanNet assets: `{preflight['raw_assets']['status']}`.",
            f"- Processed DB/NPY assets: `{preflight['processed_assets']['status']}`.",
            f"- NYU40 18-class taxonomy: `{preflight['class_taxonomy']['status']}`.",
            f"- Real 3RScan + ScanNet mix instantiation: `{preflight['mix_instantiation']['status']}`.",
            "",
            "The precision choice (`32-true`) is explicit because the official paper does not report training precision.",
            "",
            "## Reproduction Choices And Risks",
            "",
            "- Frozen encoder parameters use `requires_grad=false`, while the encoder module remains in train mode; Concerto `drop_path=0.3` is therefore an explicit runtime risk.",
            "- Exact Concerto revision, AdamW defaults, OneCycle defaults, precision, and transform list are verified reproduction choices because the paper does not report them completely.",
        ]
    )
    for deviation in diff["declared_deviations"]:
        lines.append(f"- `{deviation['setting']}`: {deviation['reason']}")
    lines.append("")
    return "\n".join(lines)


def _blocked_markdown(preflight: Mapping[str, Any]) -> str:
    if preflight["status"] == "diagnostic_pass":
        counts = preflight["expected_split_counts"]
        return "\n".join(
            [
                "# P2 Formal Training Blocked",
                "",
                "Status: `BLOCKED_DIAGNOSTIC_ONLY`",
                "",
                "The injected dataset passed the selected diagnostic checks, but this is a diagnostic-only run and cannot authorize formal P2 training.",
                "",
                f"Selected diagnostic counts: {counts['train']}/{counts['validation']}/{counts['test']}.",
                "Formal authorization requires the official 1201/312/100 ScanNet split counts.",
                "",
                "The formal training blocker remains in place.",
                "",
            ]
        )
    raw = preflight["raw_assets"]
    processed = preflight["processed_assets"]
    counts = preflight["expected_split_counts"]
    error_codes = sorted({entry["code"] for entry in preflight["errors"]})
    rio_unsupervised_sequences = preflight.get("rio_path_integrity", {}).get(
        "unsupervised_sequences"
    ) or []
    return "\n".join(
        [
            "# P2 ScanNet Prerequisite Blocked",
            "",
            "Status: `BLOCKED_MISSING_SCANNET`",
            "",
            "Official ReScene4D-C training requires the authorized ScanNet v2 release mixed with 3RScan. The current gate does not permit a 3RScan-only run to be labeled an official reproduction.",
            "",
            "## Exact Missing Prerequisite",
            "",
            f"- Official split coverage must be train={counts['train']}, validation={counts['validation']}, test={counts['test']}.",
            f"- Raw asset coverage is {raw['complete_scene_count']}/{raw['expected_scene_count']} scenes; missing assets={raw['missing_asset_count']}.",
            f"- Processed DB coverage is {processed['database_scene_count']}/{processed['expected_scene_count']} scenes.",
            f"- Processed NPY coverage is {processed['npy_scene_count']}/{processed['expected_scene_count']} scenes.",
            f"- NYU40 instance taxonomy status is `{preflight['class_taxonomy']['status']}`.",
            f"- Real mixed-dataset instantiation status is `{preflight['mix_instantiation']['status']}`.",
            "- RIO active T=2 sequences without an NYU40-18 instance: "
            f"{len(rio_unsupervised_sequences)}"
            + (
                f" (`{', '.join(rio_unsupervised_sequences)}`)"
                if rio_unsupervised_sequences
                else "."
            ),
            f"- Blocking error codes: `{', '.join(error_codes)}`.",
            "",
            "Required inputs must be obtained under the official ScanNet terms, preprocessed into the repository schema, and then re-audited. Unauthorized mirrors are not an acceptable substitute.",
            "",
            "## Not Executed As Official Reproduction",
            "",
            "- formal topology benchmark",
            "- official smoke test",
            "- official tiny overfit",
            "- formal 450-epoch training",
            "- formal checkpoint",
            "- G2 metrics",
            "",
            "No formal training or metric verdict is recorded while this prerequisite remains blocked.",
            "",
        ]
    )


def _validate_artifact_privacy(rendered: Mapping[str, str]) -> None:
    combined = "\n".join(rendered.values())
    linux_home = "/" + "ho" + "me" + "/"
    macos_home = "/" + "Us" + "ers" + "/"
    forbidden = [
        re.escape(linux_home),
        re.escape(macos_home),
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        re.escape("GPU" + "-"),
    ]
    for pattern in forbidden:
        if re.search(pattern, combined):
            raise ValueError(f"artifact privacy validation failed: {pattern}")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _invalidate_formal_preflight(output_dir: Path) -> None:
    invalid = {
        "schema_version": P2_PREFLIGHT_SCHEMA_VERSION,
        "status": "audit_in_progress",
        "formal_p2_training_authorized": False,
        "errors": [{"code": "audit_in_progress"}],
        "authorization": not_issued_authorization("audit_in_progress"),
    }
    _atomic_write_text(
        output_dir / "scannet_preflight.json",
        _json_text(invalid),
    )


def run_audit(
    *,
    raw_scannet_dir: Path,
    processed_scannet_dir: Path,
    split_dir: Path,
    test_segments_dir: Path,
    rio_processed_dir: Path,
    output_dir: Path,
    expected_split_counts: Mapping[str, int],
) -> int:
    _invalidate_formal_preflight(output_dir)
    source_tree_contract = build_p2_source_tree_contract()
    runtime_source_contract = build_p2_runtime_source_contract()
    runtime_environment_contract = build_p2_runtime_environment_contract()
    (
        reproduction,
        repository_default,
        composed,
        compose_error,
        p2_config,
    ) = _compose_config_snapshot()
    selected_counts = {
        split: int(expected_split_counts[split])
        for split in ("train", "validation", "test")
    }
    official_counts = selected_counts == OFFICIAL_SPLIT_COUNTS
    config_contract_errors = validate_p2_training_config_contract(p2_config)
    try:
        observed_semantic_sha256 = p2_training_semantic_sha256(p2_config)
    except Exception:  # noqa: BLE001 - malformed config is represented below.
        observed_semantic_sha256 = None
    config_contract = {
        "schema_version": P2_TRAINING_CONTRACT_SCHEMA_VERSION,
        "status": "pass" if not config_contract_errors else "fail",
        "errors": config_contract_errors,
        "expected_semantic_sha256": P2_TRAINING_SEMANTIC_SHA256,
        "observed_semantic_sha256": observed_semantic_sha256,
    }
    config_audit_errors = [
        _error("p2_config_contract_mismatch", field=error)
        for error in config_contract_errors
    ]
    known_empty_substitutions, known_empty_errors = (
        _audit_known_empty_scan_substitutions(p2_config)
    )
    data_root_bindings, data_root_errors = _audit_formal_data_roots(
        p2_config,
        processed_scannet_dir,
        rio_processed_dir,
        raw_scannet_dir=raw_scannet_dir,
        split_dir=split_dir,
        test_segments_dir=test_segments_dir,
    )
    if data_root_errors and not official_counts:
        data_root_bindings["status"] = "diagnostic_override"
    if p2_config is None:
        model_checkpoint = {
            "reference": "external:concerto_checkpoint",
            "expected_sha256": CONCERTO_SHA256,
            "observed_sha256": None,
            "expected_byte_size": CONCERTO_BYTES,
            "observed_byte_size": None,
            "status": "fail",
        }
        checkpoint_errors = [_error("model_checkpoint_config_unavailable")]
    else:
        checkpoint_path = Path(str(p2_config.backbone.name)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        model_checkpoint, checkpoint_errors = _audit_model_checkpoint(
            checkpoint_path
        )

    scenes_by_split, split_records, split_errors = _read_split_metadata(
        split_dir, expected_split_counts
    )
    official_split_identity, official_split_errors = (
        _audit_official_split_identity(
            split_dir,
            split_records=split_records,
        )
    )
    if official_split_errors and not official_counts:
        official_split_identity["status"] = "diagnostic_override"
    raw_assets, raw_errors = _audit_raw_assets(
        raw_scannet_dir, test_segments_dir, scenes_by_split
    )
    initial_input_manifest = None
    snapshot_start_errors: list[dict[str, Any]] = []
    if official_counts and not any(
        (
            checkpoint_errors,
            config_audit_errors,
            known_empty_errors,
            data_root_errors,
            official_split_errors,
            split_errors,
            raw_errors,
        )
    ) and _initial_input_manifest_allowed(
        source_tree=source_tree_contract,
        runtime_source=runtime_source_contract,
        runtime_environment=runtime_environment_contract,
    ):
        initial_input_manifest = build_p2_input_manifest(
            scannet_root=processed_scannet_dir,
            rio_root=rio_processed_dir,
        )
        if initial_input_manifest.get("status") != "pass":
            snapshot_start_errors.append(
                _error("p2_input_manifest_precheck_failed")
            )
    processed_assets, taxonomy, processed_errors = _audit_processed_assets(
        processed_scannet_dir, scenes_by_split
    )
    rio_taxonomy, rio_taxonomy_errors = _audit_taxonomy(
        rio_processed_dir,
        "rio",
    )
    rio_path_integrity, rio_path_errors = _audit_rio_record_paths(
        rio_processed_dir,
        validate_content=official_counts,
    )
    errors = [
        *checkpoint_errors,
        *config_audit_errors,
        *known_empty_errors,
        *(
            [
                _error(
                    "local_source_tree_contract_failed",
                    errors=source_tree_contract.get("errors", []),
                    disallowed_committed_paths=source_tree_contract.get(
                        "disallowed_committed_paths", []
                    ),
                    disallowed_dirty_paths=source_tree_contract.get(
                        "disallowed_dirty_paths", []
                    ),
                )
            ]
            if official_counts and source_tree_contract.get("status") != "pass"
            else []
        ),
        *(
            [
                _error(
                    "runtime_source_contract_failed",
                    errors=runtime_source_contract.get("errors", []),
                )
            ]
            if official_counts and runtime_source_contract.get("status") != "pass"
            else []
        ),
        *(
            [
                _error(
                    "runtime_environment_contract_failed",
                    errors=runtime_environment_contract.get("errors", []),
                )
            ]
            if official_counts
            and runtime_environment_contract.get("status") != "pass"
            else []
        ),
        *(data_root_errors if official_counts else []),
        *(official_split_errors if official_counts else []),
        *(rio_taxonomy_errors if official_counts else []),
        *(rio_path_errors if official_counts else []),
        *snapshot_start_errors,
        *split_errors,
        *raw_errors,
        *processed_errors,
    ]

    if errors:
        mix = {"attempted": False, "status": "blocked_prerequisites"}
    else:
        mix, mix_errors = _instantiate_real_mix(processed_scannet_dir, rio_processed_dir)
        errors.extend(mix_errors)

    if official_counts and not errors and mix.get("status") == "pass":
        if initial_input_manifest is None:
            errors.append(_error("p2_input_manifest_precheck_missing"))
            input_manifest = {
                "schema_version": 1,
                "status": "fail",
            }
        else:
            (
                input_manifest,
                official_split_identity,
                snapshot_errors,
            ) = _audit_semantic_snapshot_stability(
                initial_input_manifest=initial_input_manifest,
                initial_split_identity=official_split_identity,
                processed_scannet_dir=processed_scannet_dir,
                rio_processed_dir=rio_processed_dir,
                split_dir=split_dir,
            )
            errors.extend(snapshot_errors)
        if input_manifest.get("status") != "pass":
            errors.append(_error("p2_input_manifest_failed"))
    elif official_counts:
        input_manifest = {
            "schema_version": 1,
            "status": "blocked_prerequisites",
        }
    else:
        input_manifest = {
            "schema_version": 1,
            "status": "diagnostic_not_issued",
        }

    if official_counts:
        (
            source_tree_contract,
            runtime_source_contract,
            runtime_environment_contract,
            source_stability_errors,
        ) = _audit_source_contract_stability(
            initial_source_tree_contract=source_tree_contract,
            initial_runtime_source_contract=runtime_source_contract,
            initial_runtime_environment_contract=runtime_environment_contract,
        )
        errors.extend(source_stability_errors)

    audit_passed = not errors and mix.get("status") == "pass"
    formal_authorized = audit_passed and official_counts and composed
    diagnostic_pass = audit_passed and not official_counts
    split_status = "pass" if not split_errors else "fail"
    if formal_authorized:
        status = "pass"
    elif diagnostic_pass:
        status = "diagnostic_pass"
    elif not composed:
        status = "blocked_p2_config"
    else:
        status = "blocked_missing_scannet"
    preflight = {
        "schema_version": P2_PREFLIGHT_SCHEMA_VERSION,
        "status": status,
        "formal_p2_training_authorized": formal_authorized,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "local_source_commit": source_tree_contract.get("source_commit"),
        "source_tree_contract": source_tree_contract,
        "runtime_source_contract": runtime_source_contract,
        "runtime_environment_contract": runtime_environment_contract,
        "expected_split_counts": selected_counts,
        "split_metadata_status": split_status,
        "split_metadata": split_records,
        "official_split_identity": official_split_identity,
        "model_checkpoint": model_checkpoint,
        "raw_assets": raw_assets,
        "processed_assets": processed_assets,
        "class_taxonomy": taxonomy,
        "rio_class_taxonomy": rio_taxonomy,
        "known_empty_scan_substitutions": known_empty_substitutions,
        "data_root_bindings": data_root_bindings,
        "rio_path_integrity": rio_path_integrity,
        "input_manifest": input_manifest,
        "config_contract": config_contract,
        "mix_instantiation": mix,
        "errors": errors,
    }
    if formal_authorized:
        issue_formal_authorization(preflight, p2_config)
    else:
        if diagnostic_pass:
            authorization_reason = "non_official_expected_split_counts"
        elif not composed:
            authorization_reason = "p2_config_not_composed"
        else:
            authorization_reason = "formal_prerequisites_failed"
        preflight["authorization"] = not_issued_authorization(
            authorization_reason
        )

    config_diff = _build_config_diff(
        reproduction,
        repository_default,
        composed,
        compose_error,
        model_checkpoint,
    )
    environment = _build_environment_manifest(model_checkpoint)
    rendered = {
        "config_audit.md": _config_audit_markdown(config_diff, preflight),
        "environment_manifest.json": _json_text(environment),
        "reproduction_target.yaml": _yaml_text(REPRODUCTION_TARGET),
        "official_vs_repro_config_diff.json": _json_text(config_diff),
        "scannet_preflight.json": _json_text(preflight),
    }
    if not formal_authorized:
        rendered["BLOCKED_MISSING_SCANNET.md"] = _blocked_markdown(preflight)
    _validate_artifact_privacy(rendered)

    blocked_path = output_dir / "BLOCKED_MISSING_SCANNET.md"
    for filename, content in rendered.items():
        if filename != "scannet_preflight.json":
            _atomic_write_text(output_dir / filename, content)
    if formal_authorized and blocked_path.exists():
        blocked_path.unlink()
    _atomic_write_text(
        output_dir / "scannet_preflight.json",
        rendered["scannet_preflight.json"],
    )
    return 0 if formal_authorized or diagnostic_pass else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-scannet-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "scannet" / "scannet",
    )
    parser.add_argument(
        "--processed-scannet-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "scannet",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "third_party" / "ScanNet" / "Tasks" / "Benchmark",
    )
    parser.add_argument(
        "--test-segments-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "scannet_test_segments",
    )
    parser.add_argument(
        "--rio-processed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "rio",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "P2"
    )
    parser.add_argument("--expected-train", type=int, default=1201)
    parser.add_argument("--expected-validation", type=int, default=312)
    parser.add_argument("--expected-test", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_audit(
        raw_scannet_dir=args.raw_scannet_dir,
        processed_scannet_dir=args.processed_scannet_dir,
        split_dir=args.split_dir,
        test_segments_dir=args.test_segments_dir,
        rio_processed_dir=args.rio_processed_dir,
        output_dir=args.output_dir,
        expected_split_counts={
            "train": args.expected_train,
            "validation": args.expected_validation,
            "test": args.expected_test,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
