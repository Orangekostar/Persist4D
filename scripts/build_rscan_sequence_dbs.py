#!/usr/bin/env python3
"""Build and inventory official 3RScan sliding sequence databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.preprocessing import build_rscan_sequence_db as official_sequence_builder


OFFICIAL_SEED = 45
OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
SPLITS = ("test", "train", "validation")
SUPERVISED_SPLITS = frozenset(("train", "validation"))
REQUIRED_FIELDS = frozenset(
    (
        "added",
        "ambiguities",
        "filepath",
        "nonrigid",
        "removed",
        "rigid",
        "scene",
        "sub_scenes",
        "type",
    )
)
SCAN_ID_PATTERN = re.compile(r"scene\d{4}_\d{2}")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_one(
    *,
    data_dir: str | Path,
    metadata: str | Path,
    processed_dir: str | Path,
    horizon: int,
    sequence_type: str = "sliding",
    scannet200: bool = False,
) -> Path:
    """Delegate one database build to the official ReScene4D entry point."""
    output = official_sequence_builder.main(
        data_dir=str(data_dir),
        processed_dir=str(processed_dir),
        metadata_file=str(metadata),
        sequence_type=sequence_type,
        sequence_length=horizon,
        scannet200=scannet200,
    )
    return Path(output)


def _empty_split_counts() -> dict[str, int]:
    return {split: 0 for split in SPLITS}


def _scan_ids_from_key(key: object, *, horizon: int) -> list[str]:
    if not isinstance(key, str):
        raise ValueError("sequence database keys must be strings")
    scan_ids = key.split("-")
    if len(scan_ids) != horizon:
        raise ValueError(
            f"sequence {key!r} has {len(scan_ids)} scan IDs; expected {horizon} scan IDs"
        )
    if any(SCAN_ID_PATTERN.fullmatch(scan_id) is None for scan_id in scan_ids):
        raise ValueError(f"sequence {key!r} contains an invalid scan ID")
    if len(set(scan_ids)) != len(scan_ids):
        raise ValueError(f"sequence {key!r} contains a repeated scan ID")
    return scan_ids


def _validate_entry(
    key: str,
    entry: object,
    scan_ids: list[str],
    *,
    horizon: int,
) -> tuple[str, int, str | None]:
    if not isinstance(entry, dict):
        raise ValueError(f"sequence {key!r} entry must be a mapping")

    missing = sorted(REQUIRED_FIELDS - entry.keys())
    if missing:
        raise ValueError(
            f"sequence {key!r} is missing required fields: {', '.join(missing)}"
        )

    scene = entry["scene"]
    sub_scenes = entry["sub_scenes"]
    split = entry["type"]
    if not isinstance(scene, int):
        raise ValueError(f"sequence {key!r} field 'scene' must be an integer")
    if not isinstance(sub_scenes, list) or len(sub_scenes) != horizon:
        raise ValueError(
            f"sequence {key!r} field 'sub_scenes' must contain {horizon} values"
        )
    if any(not isinstance(sub_scene, int) for sub_scene in sub_scenes):
        raise ValueError(f"sequence {key!r} sub-scene IDs must be integers")
    expected_scan_ids = [
        f"scene{scene:04d}_{sub_scene:02d}" for sub_scene in sub_scenes
    ]
    if expected_scan_ids != scan_ids:
        raise ValueError(f"sequence {key!r} key disagrees with scene/sub_scenes fields")
    if split not in SPLITS:
        raise ValueError(f"sequence {key!r} has unsupported split {split!r}")

    transition_count = horizon - 1
    for field in ("added", "nonrigid", "removed", "rigid"):
        value = entry[field]
        if not isinstance(value, list) or len(value) != transition_count:
            raise ValueError(
                f"sequence {key!r} field {field!r} must contain "
                f"{transition_count} transitions"
            )

    filepath = entry["filepath"]
    if filepath is None or filepath == "" or filepath == "None":
        normalized_filepath = None
    elif isinstance(filepath, str):
        normalized_filepath = filepath
    else:
        raise ValueError(f"sequence {key!r} field 'filepath' must be a string or null")
    return split, scene, normalized_filepath


def inventory_database(path: str | Path, *, horizon: int) -> dict[str, Any]:
    """Validate one official YAML database and return its inventory."""
    database_path = Path(path)
    if not database_path.is_file():
        raise FileNotFoundError(f"sequence database does not exist: {database_path}")
    with database_path.open(encoding="utf-8") as database_file:
        database = yaml.safe_load(database_file)

    if not isinstance(database, dict):
        raise ValueError(f"{database_path}: top level must be a mapping")
    if not database:
        raise ValueError(f"{database_path}: database is empty")

    count_by_split: Counter[str] = Counter()
    unresolved_by_split: Counter[str] = Counter()
    nonexistent_by_split: Counter[str] = Counter()
    reference_scenes: set[int] = set()

    for key, entry in database.items():
        scan_ids = _scan_ids_from_key(key, horizon=horizon)
        split, scene, filepath = _validate_entry(
            key, entry, scan_ids, horizon=horizon
        )
        count_by_split[split] += 1
        reference_scenes.add(scene)

        if filepath is None:
            unresolved_by_split[split] += 1
            if split in SUPERVISED_SPLITS:
                raise ValueError(f"{split} sequence has unresolved filepath: {key}")
            continue

        if not Path(filepath).is_file():
            nonexistent_by_split[split] += 1
            if split in SUPERVISED_SPLITS:
                raise ValueError(f"{split} sequence filepath does not exist: {filepath}")

    return {
        "path": str(database_path.resolve()),
        "sha256": sha256_file(database_path),
        "bytes": database_path.stat().st_size,
        "sequence_length": horizon,
        "sequence_count": len(database),
        "count_by_split": {
            split: count_by_split[split] for split in SPLITS
        },
        "unique_reference_scene_count": len(reference_scenes),
        "unresolved_filepath_count_by_split": {
            split: unresolved_by_split[split] for split in SPLITS
        },
        "nonexistent_filepath_count_by_split": {
            split: nonexistent_by_split[split] for split in SPLITS
        },
    }


def build_all(
    *,
    data_dir: str | Path,
    processed_dir: str | Path,
    horizons: Iterable[int],
    metadata: str | Path | None = None,
    sequence_type: str = "sliding",
    scannet200: bool = False,
    source_commit: str = OFFICIAL_SOURCE_COMMIT,
) -> dict[str, Any]:
    """Build and inventory every requested official sequence database."""
    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    metadata = Path(metadata) if metadata is not None else data_dir / "3RScan.json"
    requested_horizons = list(horizons)
    if not requested_horizons or any(horizon < 2 for horizon in requested_horizons):
        raise ValueError("horizons must contain integers greater than or equal to 2")
    if len(set(requested_horizons)) != len(requested_horizons):
        raise ValueError("horizons must not contain duplicates")
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {metadata}")

    databases = []
    for horizon in requested_horizons:
        database_path = build_one(
            data_dir=data_dir,
            metadata=metadata,
            processed_dir=processed_dir,
            horizon=horizon,
            sequence_type=sequence_type,
            scannet200=scannet200,
        )
        databases.append(inventory_database(database_path, horizon=horizon))

    return {
        "schema_version": 1,
        "official_builder": (
            "datasets.preprocessing.build_rscan_sequence_db.main"
        ),
        "official_source_commit": source_commit,
        "official_seed": OFFICIAL_SEED,
        "sequence_type": sequence_type,
        "scannet200": scannet200,
        "scan_order_semantics": "metadata_order_only_no_timestamps",
        "data_dir": str(data_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "metadata": {
            "path": str(metadata.resolve()),
            "sha256": sha256_file(metadata),
        },
        "supervised_splits": sorted(SUPERVISED_SPLITS),
        "test_split_limitation": (
            "Official test entries are retained for provenance but are not loadable "
            "without hidden test assets; profiling uses train/validation only."
        ),
        "databases": databases,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic JSON manifest."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inventory official 3RScan sliding sequence databases."
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--horizons", required=True, nargs="+", type=int)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sequence-type", default="sliding", choices=("sliding",))
    scannet_group = parser.add_mutually_exclusive_group()
    scannet_group.add_argument("--scannet200", action="store_true")
    scannet_group.add_argument("--no-scannet200", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scannet200 = args.scannet200 and not args.no_scannet200
    manifest = build_all(
        data_dir=args.data_dir,
        metadata=args.metadata,
        processed_dir=args.processed_dir,
        horizons=args.horizons,
        sequence_type=args.sequence_type,
        scannet200=scannet200,
    )
    write_manifest(args.manifest, manifest)


if __name__ == "__main__":
    main()
