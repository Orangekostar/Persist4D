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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
P2_TRAINING_CONTRACT_FIX_COMMIT = "3c6b11a3af600aa98c93128361c2ecb4900ea186"
P2_RUNTIME_SAFETY_FIX_COMMIT = "973629172cc01ae0998bc785ac0ea2979756b72c"
CONCERTO_REVISION = "c31f993a56129f2ba9c5d06a35957e3f05bff710"
CONCERTO_SHA256 = "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
CONCERTO_BYTES = 433_987_358
SEQUENCE_DB_SHA256 = "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"

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
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path]
    candidates.extend(
        [
            processed_dir / split / path.name,
            processed_dir / path,
        ]
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


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
        if path.is_file():
            scenes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            scenes = []
            errors.append(_error("split_file_missing", split=split))
        unique_scenes = list(dict.fromkeys(scenes))
        valid_names = all(re.fullmatch(r"scene\d{4}_\d{2}", scene) for scene in scenes)
        status = "pass"
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
            "status": status,
        }
    return scenes_by_split, records, errors


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


def _audit_taxonomy(processed_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    label_db_path = processed_dir / "label_database.yaml"
    metric_path = processed_dir / "scannet.yaml"
    expected_validation_ids = {1, 2, *NYU40_INSTANCE_IDS}
    validation_ids: set[int] = set()
    metric_ids: list[int] = []
    metric_labels: list[str] = []
    metric_name: str | None = None

    try:
        labels = _read_yaml(label_db_path)
        if not isinstance(labels, Mapping):
            raise TypeError
        validation_ids = {
            int(class_id)
            for class_id, entry in labels.items()
            if isinstance(entry, Mapping) and bool(entry.get("validation"))
        }
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        errors.append(_error("label_database_invalid_or_missing"))

    try:
        metric = _read_yaml(metric_path)
        if not isinstance(metric, Mapping):
            raise TypeError
        metric_name = str(metric.get("name"))
        metric_ids = [int(value) for value in metric.get("valid_class_ids", [])]
        metric_labels = [str(value) for value in metric.get("class_labels", [])]
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        errors.append(_error("metric_taxonomy_invalid_or_missing"))

    if validation_ids and validation_ids != expected_validation_ids:
        errors.append(
            _error(
                "label_database_validation_ids_mismatch",
                expected_count=len(expected_validation_ids),
                observed_count=len(validation_ids),
            )
        )
    if metric_ids and metric_ids != NYU40_INSTANCE_IDS:
        errors.append(_error("metric_class_ids_mismatch"))
    if metric_labels and metric_labels != NYU40_INSTANCE_LABELS:
        errors.append(_error("metric_class_labels_mismatch"))
    if metric_name is not None and metric_name != "scannet":
        errors.append(_error("metric_dataset_name_mismatch"))

    status = "pass" if not errors else "fail"
    return (
        {
            "status": status,
            "name": metric_name,
            "valid_class_ids": metric_ids,
            "class_count": len(metric_ids),
        },
        errors,
    )


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
            if npy_path is not None and npy_path.is_file() and npy_path.suffix == ".npy":
                split_npy_count += 1
            else:
                missing_npy_scenes.append(scene)
                if len(missing_examples) < 10:
                    missing_examples.append({"scene": scene, "missing": ["npy"]})
            if split != "test":
                instance_path = _resolve_record_path(
                    record.get("instance_gt_filepath"), processed_dir / "instance_gt", split
                )
                if instance_path is not None and instance_path.is_file():
                    split_instance_count += 1
                else:
                    missing_instance_scenes.append(scene)
                    if len(missing_examples) < 10:
                        missing_examples.append({"scene": scene, "missing": ["instance_gt"]})

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


def _compose_config_snapshot() -> tuple[dict[str, Any], dict[str, Any], bool, str | None]:
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

        return snapshot(p2), snapshot(base), True, None
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - preserve partial audit evidence.
        return {}, {}, False, type(exc).__name__


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
) -> dict[str, Any]:
    get = reproduction.get
    base = repository_default.get
    official_mix = REPRODUCTION_TARGET["data"]["mix"]
    official_temporal = {"rio": 2, "scannet": 1}
    checkpoint_choice = {
        "reference": "local_cache:persist4d/concerto/concerto_base.pth",
        "revision": CONCERTO_REVISION,
        "sha256": CONCERTO_SHA256,
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
            "verified_reproduction_choice" if composed else "unverified",
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
            "local_behavior": "preflight and post-forward consensus cover train, validation, and test microbatches and raise a contextual RuntimeError across ranks instead of returning None",
            "collective_contract": {
                "normal_ddp_microbatch": {
                    "int32_max_all_reduce_count": 2,
                    "all_gather_object_count": 0,
                },
                "covered_preflight_failure": {
                    "int32_max_all_reduce_count": 1,
                    "all_gather_object_count": 1,
                },
                "covered_forward_failure": {
                    "int32_max_all_reduce_count": 2,
                    "all_gather_object_count": 1,
                },
            },
            "performance_cost": "two blocking scalar int32 MAX all_reduce operations per normal DDP microbatch, or eight per optimizer step at accumulation=4; all_gather_object is failure-only",
            "coverage_boundary": "consensus covers declared preflight violations and the recognized single-point cross-attention RuntimeError; unrelated rank-local exceptions are not converted",
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
    except (ImportError, RuntimeError, AttributeError):
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


def _build_environment_manifest() -> dict[str, Any]:
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
                "reference": "local_cache:persist4d/concerto/concerto_base.pth",
                "revision": CONCERTO_REVISION,
                "sha256": CONCERTO_SHA256,
                "byte_size": CONCERTO_BYTES,
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
            "- DDP batch-contract consensus: covered preflight and forward failures raise across ranks instead of returning `None`. The normal path adds two scalar int32 MAX all-reduces per normal DDP microbatch, or eight per optimizer step at accumulation=4, and all_gather_object only on a covered failure. This is a deliberate performance cost.",
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
    raw = preflight["raw_assets"]
    processed = preflight["processed_assets"]
    counts = preflight["expected_split_counts"]
    error_codes = sorted({entry["code"] for entry in preflight["errors"]})
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
    scenes_by_split, split_records, split_errors = _read_split_metadata(
        split_dir, expected_split_counts
    )
    raw_assets, raw_errors = _audit_raw_assets(
        raw_scannet_dir, test_segments_dir, scenes_by_split
    )
    processed_assets, taxonomy, processed_errors = _audit_processed_assets(
        processed_scannet_dir, scenes_by_split
    )
    errors = [*split_errors, *raw_errors, *processed_errors]

    if errors:
        mix = {"attempted": False, "status": "blocked_prerequisites"}
    else:
        mix, mix_errors = _instantiate_real_mix(processed_scannet_dir, rio_processed_dir)
        errors.extend(mix_errors)

    formal_authorized = not errors and mix.get("status") == "pass"
    split_status = "pass" if not split_errors else "fail"
    preflight = {
        "schema_version": 1,
        "status": "pass" if formal_authorized else "blocked_missing_scannet",
        "formal_p2_training_authorized": formal_authorized,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "expected_split_counts": {
            split: int(expected_split_counts[split])
            for split in ("train", "validation", "test")
        },
        "split_metadata_status": split_status,
        "split_metadata": split_records,
        "raw_assets": raw_assets,
        "processed_assets": processed_assets,
        "class_taxonomy": taxonomy,
        "mix_instantiation": mix,
        "errors": errors,
    }

    reproduction, repository_default, composed, compose_error = _compose_config_snapshot()
    config_diff = _build_config_diff(
        reproduction, repository_default, composed, compose_error
    )
    environment = _build_environment_manifest()
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

    output_dir.mkdir(parents=True, exist_ok=True)
    blocked_path = output_dir / "BLOCKED_MISSING_SCANNET.md"
    if formal_authorized and blocked_path.exists():
        blocked_path.unlink()
    for filename, content in rendered.items():
        (output_dir / filename).write_text(content, encoding="utf-8")
    return 0 if formal_authorized else 2


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
