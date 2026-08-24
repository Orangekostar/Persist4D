#!/usr/bin/env python3
"""Audit raw ReScan coordinate compatibility without fitting transforms."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from datasets.rescan_adapter import discover_rescan_dataset, read_rescan_ply


class RescanCoordinateAuditError(ValueError):
    pass


def _voxel_representatives(xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise RescanCoordinateAuditError("xyz must have finite shape [N, 3]")
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise RescanCoordinateAuditError("voxel_size_m must be positive")
    grid = np.floor(points / voxel_size_m).astype(np.int64)
    _, indices = np.unique(grid, axis=0, return_index=True)
    return np.ascontiguousarray(points[np.sort(indices)])


def coordinate_pair_metrics(
    first_xyz: np.ndarray,
    second_xyz: np.ndarray,
    *,
    voxel_size_m: float = 0.05,
) -> dict[str, float | int]:
    first = _voxel_representatives(first_xyz, voxel_size_m)
    second = _voxel_representatives(second_xyz, voxel_size_m)
    if not first.size or not second.size:
        raise RescanCoordinateAuditError("coordinate audit inputs must be nonempty")
    first_to_second = cKDTree(second).query(first, k=1, workers=1)[0]
    second_to_first = cKDTree(first).query(second, k=1, workers=1)[0]
    symmetric = np.concatenate((first_to_second, second_to_first))
    return {
        "first_voxel_point_count": int(first.shape[0]),
        "second_voxel_point_count": int(second.shape[0]),
        "first_to_second_median_nn_m": float(np.median(first_to_second)),
        "second_to_first_median_nn_m": float(np.median(second_to_first)),
        "symmetric_median_nn_m": float(np.median(symmetric)),
        "symmetric_p90_nn_m": float(np.quantile(symmetric, 0.9)),
        "symmetric_overlap_fraction_at_10cm": float(np.mean(symmetric <= 0.1)),
        "symmetric_overlap_fraction_at_20cm": float(np.mean(symmetric <= 0.2)),
    }


def build_coordinate_audit(
    dataset_root: Path,
    *,
    voxel_size_m: float = 0.05,
) -> dict[str, object]:
    inventory = discover_rescan_dataset(dataset_root)
    scenes = []
    all_pairs = []
    for scene in inventory.scenes:
        pairs = []
        previous = None
        for capture in scene.captures:
            cloud = read_rescan_ply(capture.ply_path)
            if previous is not None:
                metrics = coordinate_pair_metrics(
                    previous[1], cloud.xyz, voxel_size_m=voxel_size_m
                )
                record = {
                    "first_capture_id": previous[0],
                    "second_capture_id": capture.capture_id,
                    **metrics,
                }
                pairs.append(record)
                all_pairs.append(record)
            previous = (capture.capture_id, cloud.xyz)
        scenes.append({"scene_id": scene.scene_id, "adjacent_pairs": pairs})
    medians = [float(pair["symmetric_median_nn_m"]) for pair in all_pairs]
    overlaps = [float(pair["symmetric_overlap_fraction_at_10cm"]) for pair in all_pairs]
    return {
        "schema_version": 1,
        "status": "pass",
        "audit_scope": "raw_xyz_geometry_only_no_transform_fitting",
        "model_input_transform_applied": False,
        "ground_truth_identity_or_object_pose_used": False,
        "voxel_size_m": voxel_size_m,
        "summary": {
            "scene_count": len(scenes),
            "adjacent_pair_count": len(all_pairs),
            "median_of_pair_symmetric_median_nn_m": float(np.median(medians)),
            "maximum_pair_symmetric_median_nn_m": float(np.max(medians)),
            "median_pair_overlap_fraction_at_10cm": float(np.median(overlaps)),
            "minimum_pair_overlap_fraction_at_10cm": float(np.min(overlaps)),
        },
        "interpretation_rule": (
            "raw frames are usable when every adjacent pair has geometric overlap; "
            "metrics are descriptive and no alignment is estimated"
        ),
        "scenes": scenes,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = build_coordinate_audit(
        arguments.dataset_root, voxel_size_m=arguments.voxel_size_m
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
