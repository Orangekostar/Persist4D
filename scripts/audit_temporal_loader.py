#!/usr/bin/env python3
"""Audit the official temporal dataset loader on supervised 3RScan splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.semseg import SemanticSegmentationDataset


DEFAULT_HORIZONS = (2, 3, 4, 5)
DEFAULT_SPLITS = ("train", "validation")
OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str) -> str:
    path = Path(value.replace("../../", ""))
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def repo_reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return f"repo:{resolved.relative_to(PROJECT_ROOT).as_posix()}"
    except ValueError:
        return f"external:{resolved.name}"


def make_dataset(processed_dir: Path, horizon: int, split: str) -> SemanticSegmentationDataset:
    dataset = SemanticSegmentationDataset(
        dataset_name="rio",
        data_dir=str(processed_dir),
        label_db_filepath=str(processed_dir / "label_database.yaml"),
        change_label_db_filepath=str(processed_dir / "change_label_database.yaml"),
        color_mean_std=str(processed_dir / "color_mean_std.yaml"),
        mode=split,
        add_colors=True,
        add_normals=False,
        add_raw_coordinates=False,
        add_instance=True,
        num_labels=20,
        num_changes=-1,
        filter_out_classes=[0, 1],
        label_offset=2,
        temporal_window=horizon,
        image_augmentations_path=None,
        volume_augmentations_path=None,
    )
    for record in dataset.data:
        record["filepath"] = resolve_project_path(str(record["filepath"]))
        record["instance_gt_filepath"] = resolve_project_path(
            str(record["instance_gt_filepath"])
        )
    dataset.change_files = [
        None if path is None or str(path) == "None" else resolve_project_path(str(path))
        for path in dataset.change_files
    ]
    return dataset


def select_indices(count: int, limit: int, seed: int, horizon: int, split: str) -> list[int]:
    sample_count = min(count, limit)
    selection_seed = seed + horizon * 100 + (0 if split == "train" else 1)
    return sorted(random.Random(selection_seed).sample(range(count), sample_count))


def inspect_sample(
    dataset: SemanticSegmentationDataset,
    dataset_index: int,
    seed: int,
) -> dict[str, Any]:
    sample_seed = seed + dataset_index
    random.seed(sample_seed)
    np.random.seed(sample_seed)

    change_file = dataset.change_files[dataset_index]
    if change_file is None:
        raise ValueError("supervised sequence has no change-label filepath")
    raw_changes = np.genfromtxt(change_file, dtype=int)
    loaded = dataset[dataset_index]
    coordinates, features, labels, sequence_name = loaded[:4]
    projected_changes = (
        labels[:, 2]
        if labels.ndim == 2 and labels.shape[1] > 2
        else np.asarray(None)
    )
    expected_projection = raw_changes[:, 0] if raw_changes.ndim == 2 else raw_changes
    temporal_stages = (
        sorted(int(stage) for stage in np.unique(coordinates[:, 3]).tolist())
        if coordinates.ndim == 2 and coordinates.shape[1] > 3
        else []
    )

    return {
        "dataset_index": dataset_index,
        "sample_seed": sample_seed,
        "sequence_name": sequence_name,
        "change_filepath": repo_reference(Path(change_file)),
        "coordinate_shape": list(coordinates.shape),
        "feature_shape": list(features.shape),
        "label_shape": list(labels.shape),
        "temporal_stages": temporal_stages,
        "raw_change_shape": list(raw_changes.shape),
        "raw_change_dim": int(raw_changes.ndim),
        "projected_change_dim": int(projected_changes.ndim),
        "projection_matches_official_rule": bool(
            np.array_equal(projected_changes, expected_projection)
        ),
    }


def validate_sample_record(sample: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []

    def require(code: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            errors.append({"code": code, "expected": expected, "actual": actual})

    coordinate_shape = sample["coordinate_shape"]
    feature_shape = sample["feature_shape"]
    label_shape = sample["label_shape"]
    point_count = coordinate_shape[0] if len(coordinate_shape) == 2 else None

    require("coordinate_rank", 2, len(coordinate_shape))
    require(
        "coordinate_columns",
        4,
        coordinate_shape[1] if len(coordinate_shape) == 2 else None,
    )
    require(
        "feature_point_count",
        point_count,
        feature_shape[0] if len(feature_shape) >= 1 else None,
    )
    require(
        "label_point_count",
        point_count,
        label_shape[0] if len(label_shape) >= 1 else None,
    )
    require(
        "label_columns",
        4,
        label_shape[1] if len(label_shape) == 2 else None,
    )
    require("temporal_stages", list(range(horizon)), sample["temporal_stages"])

    expected_raw_shape = (
        [point_count] if horizon == 2 else [point_count, horizon - 1]
    )
    require("raw_change_shape", expected_raw_shape, sample["raw_change_shape"])
    require("raw_change_dim", 1 if horizon == 2 else 2, sample["raw_change_dim"])
    require("projected_change_dim", 1, sample["projected_change_dim"])
    require(
        "projection_mismatch",
        True,
        sample["projection_matches_official_rule"],
    )
    return errors


def audit_split(
    *,
    processed_dir: Path,
    horizon: int,
    split: str,
    sample_limit: int,
    seed: int,
) -> dict[str, Any]:
    processed_dir = Path(processed_dir).resolve()
    database_path = processed_dir / f"sequence_database_sliding_{horizon}.yaml"
    database = load_yaml(database_path)
    if not isinstance(database, dict):
        raise ValueError(f"sequence database must be a mapping: {database_path}")

    result: dict[str, Any] = {
        "horizon": horizon,
        "split": split,
        "database_path": repo_reference(database_path),
        "database_sha256": sha256_file(database_path),
        "database_count": sum(
            entry.get("type") == split for entry in database.values()
        ),
        "loader_sequence_count": None,
        "sample_indices": [],
        "success_count": 0,
        "failure_count": 0,
        "samples": [],
        "exceptions": [],
        "validation_errors": [],
    }

    try:
        dataset = make_dataset(processed_dir, horizon, split)
        result["loader_sequence_count"] = len(dataset)
    except (Exception, SystemExit) as error:
        result["failure_count"] = 1
        result["exceptions"].append(
            {
                "phase": "dataset_initialization",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        return result

    if result["loader_sequence_count"] != result["database_count"]:
        result["failure_count"] = 1
        result["validation_errors"].append(
            {
                "phase": "dataset_validation",
                "code": "database_loader_count_mismatch",
                "expected": result["database_count"],
                "actual": result["loader_sequence_count"],
            }
        )
        return result

    indices = select_indices(len(dataset), sample_limit, seed, horizon, split)
    result["sample_indices"] = indices
    for dataset_index in indices:
        try:
            sample = inspect_sample(dataset, dataset_index, seed)
            result["samples"].append(sample)
            validation_errors = validate_sample_record(sample, horizon)
            if validation_errors:
                result["failure_count"] += 1
                result["validation_errors"].append(
                    {
                        "phase": "sample_validation",
                        "dataset_index": dataset_index,
                        "sequence_name": sample["sequence_name"],
                        "errors": validation_errors,
                    }
                )
            else:
                result["success_count"] += 1
        except (Exception, SystemExit) as error:
            result["failure_count"] += 1
            result["exceptions"].append(
                {
                    "phase": "sample_load",
                    "dataset_index": dataset_index,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )
    return result


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return True if result.returncode != 0 else bool(result.stdout.strip())


def run_audit(
    *,
    processed_dir: Path,
    horizons: Sequence[int],
    splits: Sequence[str],
    sample_limit: int,
    seed: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    top_level_exceptions: list[dict[str, Any]] = []
    for horizon in horizons:
        for split in splits:
            try:
                records.append(
                    audit_split(
                        processed_dir=processed_dir,
                        horizon=horizon,
                        split=split,
                        sample_limit=sample_limit,
                        seed=seed,
                    )
                )
            except (Exception, SystemExit) as error:
                top_level_exceptions.append(
                    {
                        "horizon": horizon,
                        "split": split,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )

    failure_count = sum(record["failure_count"] for record in records)
    failure_count += len(top_level_exceptions)
    loader_source = PROJECT_ROOT / "datasets" / "semseg.py"
    profile_config = PROJECT_ROOT / "conf" / "profiling" / "p0_p1_a40.yaml"
    generator_source = PROJECT_ROOT / "scripts" / "audit_temporal_loader.py"
    return {
        "schema_version": 2,
        "status": "pass" if failure_count == 0 else "blocked",
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "generator_git_commit": git_commit(),
        "generator_dirty": git_dirty(),
        "source_hashes": {
            "datasets/semseg.py": sha256_file(loader_source),
            "conf/profiling/p0_p1_a40.yaml": sha256_file(profile_config),
            "scripts/audit_temporal_loader.py": sha256_file(generator_source),
        },
        "configuration": {
            "processed_dir": repo_reference(Path(processed_dir)),
            "horizons": list(horizons),
            "audited_splits": list(splits),
            "explicitly_excluded_splits": ["test"],
            "sequence_type": "sliding",
            "samples_per_split": sample_limit,
            "seed": seed,
            "dataset": {
                "num_labels": 20,
                "num_changes": -1,
                "filter_out_classes": [0, 1],
                "label_offset": 2,
                "add_colors": True,
                "add_normals": False,
                "add_instance": True,
            },
        },
        "sequence_order_semantics": "metadata_order_only_no_timestamps",
        "semantic_limitations": [
            {
                "code": "metadata_order_is_not_chronology",
                "detail": (
                    "3RScan metadata contains no timestamps; sequence position records "
                    "metadata order only and is not asserted to be real chronology."
                ),
            },
            {
                "code": "first_transition_change_projection",
                "detail": (
                    "For T>2, raw change GT has shape (N, T-1), but the official "
                    "loader projects it to raw_changes[:, 0]; downstream labels retain "
                    "only the first transition."
                ),
            },
            {
                "code": "test_split_excluded",
                "detail": "The test split is excluded because local test point clouds and GT are unavailable.",
            },
        ],
        "audits": records,
        "top_level_exceptions": top_level_exceptions,
        "totals": {
            "requested_audits": len(horizons) * len(splits),
            "completed_audits": len(records),
            "loaded_samples": sum(record["success_count"] for record in records),
            "failures": failure_count,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the official ReScene4D temporal loader on T=2..5."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "rio",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "data_audit" / "temporal_loader_audit.json",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument(
        "--splits", choices=DEFAULT_SPLITS, nargs="+", default=list(DEFAULT_SPLITS)
    )
    parser.add_argument(
        "--samples-per-split",
        "--sample-limit",
        dest="samples_per_split",
        type=int,
        default=5,
    )
    parser.add_argument("--seed", type=int, default=45)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_split < 1:
        raise SystemExit("--samples-per-split must be at least 1")
    if any(horizon < 2 for horizon in args.horizons):
        raise SystemExit("--horizons values must be at least 2")

    artifact = run_audit(
        processed_dir=args.processed_dir,
        horizons=args.horizons,
        splits=args.splits,
        sample_limit=args.samples_per_split,
        seed=args.seed,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} (status={artifact['status']})")
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
