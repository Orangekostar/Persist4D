"""Inventory and bind the official ReScan dataset without importing raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np

from datasets.rescan_adapter import (
    RescanAdapterError,
    discover_rescan_dataset,
    parse_rescan_ambiguities,
    read_rescan_ply,
)


class RescanDatasetAuditError(ValueError):
    """Raised when the official package is incomplete or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _content_digest(records: Sequence[Mapping[str, object]]) -> str:
    payload = json.dumps(
        list(records),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _regular_files(root: Path) -> tuple[Path, ...]:
    result = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RescanDatasetAuditError(f"dataset contains a symlink: {path}")
        if path.is_file():
            result.append(path)
    return tuple(sorted(result, key=lambda path: path.relative_to(root).as_posix()))


def _modality_capture_ids(
    root: Path,
    *,
    scene_id: str,
    modality: str,
    suffix: str,
) -> set[str]:
    directory = root / scene_id / modality
    if directory.is_symlink() or not directory.is_dir():
        raise RescanDatasetAuditError(f"scene {scene_id} lacks {modality}")
    paths = tuple(directory.glob(f"*{suffix}"))
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise RescanDatasetAuditError(f"scene {scene_id} has invalid {modality} files")
    return {path.name[: -len(suffix)] for path in paths}


def _gap_opportunities(
    capture_ids: Sequence[str],
    identities_by_capture: Sequence[set[int]],
) -> list[dict[str, object]]:
    all_identities = sorted(set().union(*identities_by_capture))
    gaps = []
    for identity in all_identities:
        visible = [
            index
            for index, identities in enumerate(identities_by_capture)
            if identity in identities
        ]
        for left, right in pairwise(visible):
            if right - left <= 1:
                continue
            gaps.append(
                {
                    "identity": identity,
                    "left_capture_id": capture_ids[left],
                    "right_capture_id": capture_ids[right],
                    "absent_capture_ids": list(capture_ids[left + 1 : right]),
                }
            )
    return gaps


def build_rescan_dataset_manifest(
    dataset_root: str | Path,
    *,
    archive_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise RescanDatasetAuditError("dataset root must be a regular directory")
    try:
        inventory = discover_rescan_dataset(root)
    except RescanAdapterError as error:
        raise RescanDatasetAuditError(str(error)) from error
    class_file = root / "nyu40_classes.txt"
    if class_file.is_symlink() or not class_file.is_file():
        raise RescanDatasetAuditError("dataset root lacks nyu40_classes.txt")

    files = []
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )

    scenes = []
    encountered_classes: set[int] = set()
    stable_identity_count = 0
    stable_object_identity_count = 0
    ambiguity_file_count = 0
    gap_count = 0
    semantic_inconsistent_identity_count = 0
    for scene in inventory.scenes:
        capture_ids = [capture.capture_id for capture in scene.captures]
        expected_ids = set(capture_ids)
        for modality, suffix in (("color", ".h264"), ("depth", ".depth")):
            observed = _modality_capture_ids(
                root,
                scene_id=scene.scene_id,
                modality=modality,
                suffix=suffix,
            )
            if observed != expected_ids:
                raise RescanDatasetAuditError(
                    f"scene {scene.scene_id} {modality} capture ids differ: "
                    f"missing={sorted(expected_ids - observed)}, "
                    f"extra={sorted(observed - expected_ids)}"
                )

        identities_by_capture = []
        object_identities_by_capture = []
        classes_by_identity: dict[int, set[int]] = {}
        capture_records = []
        for capture in scene.captures:
            try:
                cloud = read_rescan_ply(capture.ply_path)
            except RescanAdapterError as error:
                raise RescanDatasetAuditError(str(error)) from error
            class_ids = sorted(int(value) for value in set(cloud.class_ids.tolist()))
            encountered_classes.update(class_ids)
            valid_identity_mask = (
                (cloud.instance_ids >= 0)
                & (cloud.instance_ids < 256)
                & (cloud.class_ids > 0)
            )
            identities = {
                int(value) for value in cloud.instance_ids[valid_identity_mask].tolist()
            }
            identities_by_capture.append(identities)
            object_mask = valid_identity_mask & ~np.isin(cloud.class_ids, [1, 2, 22])
            object_identities = {
                int(value) for value in cloud.instance_ids[object_mask].tolist()
            }
            object_identities_by_capture.append(object_identities)
            for identity in identities:
                identity_classes = {
                    int(value)
                    for value in cloud.class_ids[
                        valid_identity_mask & (cloud.instance_ids == identity)
                    ].tolist()
                }
                classes_by_identity.setdefault(identity, set()).update(identity_classes)
            ambiguity_rows: dict[str, list[int]] = {}
            if capture.ambiguity_path is not None:
                ambiguity_file_count += 1
                try:
                    ambiguity = parse_rescan_ambiguities(capture.ambiguity_path)
                except RescanAdapterError as error:
                    raise RescanDatasetAuditError(str(error)) from error
                ambiguity_rows = {
                    str(base): list(values)
                    for base, values in ambiguity.alternatives.items()
                }
            capture_records.append(
                {
                    "capture_id": capture.capture_id,
                    "temporal_index": capture.temporal_index,
                    "ply_reference": (
                        "external:rescan/dataset/"
                        + capture.ply_path.relative_to(root).as_posix()
                    ),
                    "point_count": int(cloud.xyz.shape[0]),
                    "class_ids": class_ids,
                    "instance_ids": sorted(identities),
                    "ambiguity_alternatives": ambiguity_rows,
                    "xyz_min": [float(value) for value in cloud.xyz.min(axis=0)],
                    "xyz_max": [float(value) for value in cloud.xyz.max(axis=0)],
                }
            )
        stable_identities = sorted(set().union(*identities_by_capture))
        semantic_inconsistent_ids = sorted(
            identity
            for identity, values in classes_by_identity.items()
            if len(values) != 1
        )
        semantic_inconsistent_identity_count += len(semantic_inconsistent_ids)
        stable_object_identities = sorted(set().union(*object_identities_by_capture))
        stable_identity_count += len(stable_identities)
        stable_object_identity_count += len(stable_object_identities)
        gaps = _gap_opportunities(capture_ids, object_identities_by_capture)
        gap_count += len(gaps)
        scenes.append(
            {
                "scene_id": scene.scene_id,
                "capture_ids": capture_ids,
                "stable_identity_ids": stable_identities,
                "stable_object_identity_ids": stable_object_identities,
                "identity_source_class_ids": {
                    str(identity): sorted(classes_by_identity[identity])
                    for identity in stable_identities
                },
                "semantic_inconsistent_identity_ids": semantic_inconsistent_ids,
                "gap_opportunities": gaps,
                "captures": capture_records,
            }
        )

    archive = None
    if archive_path is not None:
        archive_source = Path(archive_path)
        if archive_source.is_symlink() or not archive_source.is_file():
            raise RescanDatasetAuditError("archive must be a regular file")
        archive = {
            "external_reference": "external:rescan/rescan_dataset.zip",
            "bytes": archive_source.stat().st_size,
            "sha256": _sha256_file(archive_source),
        }

    return {
        "schema_version": 1,
        "status": "pass",
        "source": {
            "dataset_reference": "external:rescan/dataset",
            "official_code_commit": "f45283be31119e9bd955d40bc159b1774dfed092",
            "official_download_url": (
                "https://rescan.cs.princeton.edu/assets/rescan_dataset.zip"
            ),
            "archive": archive,
        },
        "chronology": {
            "status": "official_index_order",
            "ordering_rule": "scene_list order; contiguous numeric capture suffix",
            "evidence": [
                "official pipeline sorts gt_segmentation PLY basenames",
                "official paper and supplement define captures as t0..tn",
                "package capture suffixes are contiguous 0..n",
            ],
        },
        "coordinate_metadata": {
            "scene_transform_file_count": 0,
            "object_transform_input_allowed": False,
            "status": "requires_geometry_audit",
        },
        "summary": {
            "scene_count": len(scenes),
            "capture_count": sum(len(scene["capture_ids"]) for scene in scenes),
            "file_count": len(files),
            "ambiguity_file_count": ambiguity_file_count,
            "stable_identity_count": stable_identity_count,
            "stable_object_identity_count": stable_object_identity_count,
            "object_identity_excluded_source_class_ids": [1, 2, 22],
            "semantic_inconsistent_identity_count": (
                semantic_inconsistent_identity_count
            ),
            "gap_opportunity_count": gap_count,
            "encountered_class_ids": sorted(encountered_classes),
        },
        "dataset_content_sha256": _content_digest(files),
        "files": files,
        "scenes": scenes,
    }


def write_rescan_dataset_manifest(
    path: str | Path,
    payload: Mapping[str, object],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical_json_bytes(payload)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError("manifest path is not a regular file")
        if output.read_bytes() == content:
            return
        raise FileExistsError("manifest already contains different content")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = build_rescan_dataset_manifest(
        arguments.dataset_root,
        archive_path=arguments.archive,
    )
    write_rescan_dataset_manifest(arguments.output, manifest)
    print(json.dumps(manifest["summary"], allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
