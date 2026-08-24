"""Audit the official MultiScan inventory and freeze the T>=3 collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets.multiscan_adapter import (
    MultiScanAdapterError,
    build_multiscan_inventory,
    parse_multiscan_scan_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "487080cf31266f1572257e2aca36767e074b68b6"
MULTISCAN_COMMIT = "697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605"
DATA_REPOSITORY_REVISION = "c62c9aad850a8638ac3e42605926a65707600125"
_MAPPING_STATUSES = ("exact", "defensible", "ambiguous", "unsupported")


def _json_bytes(value: Mapping[str, object]) -> bytes:
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


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise MultiScanAdapterError(f"refusing to overwrite different artifact: {path}")
    path.write_bytes(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def build_chronology_audit(inventory: Mapping[str, object]) -> dict[str, object]:
    """Freeze official scan-index order without claiming acquisition chronology."""
    if inventory.get("status") != "pass":
        raise MultiScanAdapterError("chronology audit requires a passing inventory")
    digest = inventory.get("selected_scene_list_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise MultiScanAdapterError("chronology audit requires the selected-list hash")
    raw_scenes = inventory.get("scenes")
    if not isinstance(raw_scenes, list) or not raw_scenes:
        raise MultiScanAdapterError("chronology audit requires inventory scenes")
    scene_orders = []
    for raw_scene in raw_scenes:
        if not isinstance(raw_scene, Mapping):
            raise MultiScanAdapterError("chronology inventory scene must be a mapping")
        scene_id = raw_scene.get("scene_id")
        raw_scan_ids = raw_scene.get("scan_ids")
        if not isinstance(scene_id, str) or not isinstance(raw_scan_ids, list):
            raise MultiScanAdapterError("chronology inventory scene is malformed")
        ordered_scan_ids = []
        prior_index = -1
        for scan_id in raw_scan_ids:
            observed_scene, scan_index = parse_multiscan_scan_id(scan_id)
            if observed_scene != scene_id or scan_index <= prior_index:
                raise MultiScanAdapterError("inventory scan order is inconsistent")
            ordered_scan_ids.append(scan_id)
            prior_index = scan_index
        scene_orders.append(
            {"scene_id": scene_id, "ordered_scan_ids": ordered_scan_ids}
        )
    return {
        "schema_version": 1,
        "status": "DATASET_ORDER_ONLY",
        "selected_scene_list_sha256": digest,
        "ordering_rule": "numeric scan suffix within each physical scene",
        "physical_chronology_proven": False,
        "ordered_revisit_protocol_allowed": True,
        "claim_boundary": (
            "this is not proven physical chronology; causal claims are limited "
            "to the frozen dataset-index order"
        ),
        "evidence": [
            {
                "source": "official scans_split.csv and dataset naming documentation",
                "finding": "the final two digits are defined as the scan index",
            },
            {
                "source": "official acquired-data schema",
                "finding": (
                    "scan metadata has no cross-capture acquisition timestamp; "
                    "JSONL timestamps are frame-level ARKit session timestamps"
                ),
            },
            {
                "source": "official alignment schema",
                "finding": (
                    "reference_scan_alignment identifies a geometric reference, "
                    "not capture chronology"
                ),
            },
        ],
        "scene_orders": scene_orders,
    }


def _frozen_rescene_classes(path: Path) -> dict[str, int]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise MultiScanAdapterError("frozen ReScene label map must be a regular file")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MultiScanAdapterError("cannot parse frozen ReScene label map") from error
    mappings = document.get("mappings") if isinstance(document, dict) else None
    if not isinstance(mappings, list):
        raise MultiScanAdapterError("frozen ReScene label map lacks mappings")
    target_classes: dict[str, int] = {}
    target_ids: set[int] = set()
    for raw_mapping in mappings:
        if not isinstance(raw_mapping, dict) or raw_mapping.get("status") != "exact":
            continue
        name = raw_mapping.get("target_class_name")
        target_id = raw_mapping.get("target_class_id")
        if (
            not isinstance(name, str)
            or isinstance(target_id, bool)
            or not isinstance(target_id, int)
            or name in target_classes
            or target_id in target_ids
        ):
            raise MultiScanAdapterError("frozen ReScene exact class rows are invalid")
        target_classes[name] = target_id
        target_ids.add(target_id)
    if len(target_classes) != 18 or target_ids != set(range(18)):
        raise MultiScanAdapterError("frozen ReScene taxonomy must contain 18 classes")
    return target_classes


def build_multiscan_label_map(
    semantic_map_path: str | Path,
    *,
    frozen_rescene_map_path: str | Path = (
        PROJECT_ROOT / "artifacts/final_evidence/rescan_to_rescene_label_map.json"
    ),
) -> dict[str, object]:
    """Map official MultiScan semantic classes conservatively to ReScene18."""
    source = Path(semantic_map_path)
    if source.is_symlink() or not source.is_file():
        raise MultiScanAdapterError("MultiScan semantic map must be a regular file")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"objectName", "objectSemanticName", "objectSemanticId"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise MultiScanAdapterError("MultiScan semantic map columns differ")
        raw_rows = list(reader)
    if not raw_rows:
        raise MultiScanAdapterError("MultiScan semantic map must not be empty")

    names_by_id: dict[int, set[str]] = defaultdict(set)
    object_names_by_id: dict[int, set[str]] = defaultdict(set)
    seen_object_names: set[str] = set()
    for row in raw_rows:
        object_name = row.get("objectName")
        semantic_name = row.get("objectSemanticName")
        raw_id = row.get("objectSemanticId")
        if not object_name or not semantic_name or not raw_id:
            raise MultiScanAdapterError("MultiScan semantic map contains empty fields")
        try:
            semantic_id = int(raw_id)
        except ValueError as error:
            raise MultiScanAdapterError(
                "MultiScan semantic ID must be an integer"
            ) from error
        if semantic_id <= 0 or object_name in seen_object_names:
            raise MultiScanAdapterError("MultiScan semantic map contains duplicate IDs")
        seen_object_names.add(object_name)
        names_by_id[semantic_id].add(semantic_name)
        object_names_by_id[semantic_id].add(object_name)
    if any(len(names) != 1 for names in names_by_id.values()):
        raise MultiScanAdapterError("one MultiScan semantic ID has conflicting names")

    target_path = Path(frozen_rescene_map_path)
    target_classes = _frozen_rescene_classes(target_path)
    mappings = []
    for semantic_id in sorted(names_by_id):
        semantic_name = next(iter(names_by_id[semantic_id]))
        target_id = target_classes.get(semantic_name)
        status = "exact" if target_id is not None else "unsupported"
        evidence = (
            "exact_official_semantic_name_and_frozen_target_name"
            if status == "exact"
            else (
                "excluded_structural_class"
                if semantic_name in {"wall", "floor", "ceiling"}
                else "absent_from_frozen_foreground_taxonomy"
            )
        )
        mappings.append(
            {
                "source_class_id": semantic_id,
                "source_class_name": semantic_name,
                "source_object_names": sorted(object_names_by_id[semantic_id]),
                "status": status,
                "target_class_id": target_id,
                "target_class_name": semantic_name if target_id is not None else None,
                "mapping_evidence": evidence,
            }
        )
    status_counts = Counter(row["status"] for row in mappings)
    return {
        "schema_version": 1,
        "source_taxonomy": "official MultiScan objectSemanticId",
        "target_taxonomy": "frozen ReScene ScanNet18 foreground output",
        "source_class_count": len(mappings),
        "status_counts": {
            status: status_counts[status] for status in _MAPPING_STATUSES
        },
        "primary_class_aware_status": "exact_only",
        "provenance": {
            "source_semantic_map_sha256": _sha256_file(source),
            "target_label_map_reference": (
                "repo:artifacts/final_evidence/rescan_to_rescene_label_map.json"
            ),
            "target_label_map_sha256": _sha256_file(target_path),
        },
        "mappings": mappings,
    }


def build_inventory_artifacts(
    *,
    scans_split_path: Path,
    output_directory: Path,
    reproducibility_binding: Mapping[str, object],
) -> dict[str, Path]:
    """Publish deterministic source bindings, inventory, and frozen subset."""
    if (
        reproducibility_binding.get("schema_version") != 1
        or reproducibility_binding.get("status") != "pass"
    ):
        raise MultiScanAdapterError("reproducibility binding must pass schema v1")
    inventory = build_multiscan_inventory(scans_split_path)
    selected_ids = set(inventory["selected_scene_ids"])
    subset_scenes = [
        scene for scene in inventory["scenes"] if scene["scene_id"] in selected_ids
    ]
    subset = {
        "collection_name": "MultiScan Longitudinal Zero-Shot Collection",
        "scene_count": inventory["selected_scene_count"],
        "scan_count": inventory["selected_scan_count"],
        "schema_version": 1,
        "scenes": subset_scenes,
        "selected_rule": inventory["selected_rule"],
        "selected_scene_list_sha256": inventory["selected_scene_list_sha256"],
        "status": "pass",
    }
    payloads = {
        "repro_bindings.json": _json_bytes(reproducibility_binding),
        "reproducibility_binding.json": _json_bytes(reproducibility_binding),
        "multiscan_inventory.json": _json_bytes(inventory),
        "longitudinal_subset_manifest.json": _json_bytes(subset),
    }
    paths = {name: output_directory / name for name in payloads}
    for name, payload in payloads.items():
        _publish_exact(paths[name], payload)
    return paths


def _reproducibility_binding(
    *,
    root: Path,
    multiscan_repository: Path,
    scans_split_path: Path,
    semantic_map_path: Path,
    checkpoint_path: Path,
    data_repository_revision: str,
) -> dict[str, object]:
    source_binding = json.loads(
        (root / "artifacts/final_evidence/source_binding.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_inputs = source_binding["frozen_inputs"]
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != frozen_inputs["checkpoint_sha256"]:
        raise MultiScanAdapterError("frozen checkpoint SHA256 differs")
    if _git(multiscan_repository, "rev-parse", "HEAD") != MULTISCAN_COMMIT:
        raise MultiScanAdapterError("official MultiScan checkout commit differs")
    return {
        "schema_version": 1,
        "status": "pass",
        "persist4d": {
            "branch": "research/persist4d-multiscan-preflight",
            "frozen_commit": FROZEN_COMMIT,
            "frozen_tree": _git(root, "rev-parse", f"{FROZEN_COMMIT}^{{tree}}"),
            "generation_commit": _git(root, "rev-parse", "HEAD"),
            "final_evidence_tree": _git(
                root, "rev-parse", f"{FROZEN_COMMIT}:artifacts/final_evidence"
            ),
            "reviewer_closure_tree": _git(
                root, "rev-parse", f"{FROZEN_COMMIT}:artifacts/reviewer_closure"
            ),
        },
        "frozen_runtime": {
            "checkpoint_reference": (
                "repo-ignored:checkpoints/rescene4d_concerto_t2_repro.ckpt"
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "external_config_reference": "repo:configs/final_evidence/rescan.yaml",
            "external_config_sha256": _sha256_file(
                root / "configs/final_evidence/rescan.yaml"
            ),
            "rescene_config_reference": (
                "repo:configs/system_comparison/persist4d_incumbent.yaml"
            ),
            "rescene_config_sha256": _sha256_file(
                root / "configs/system_comparison/persist4d_incumbent.yaml"
            ),
        },
        "multiscan": {
            "repository": "https://github.com/smartscenes/multiscan",
            "commit": MULTISCAN_COMMIT,
            "tree": _git(multiscan_repository, "rev-parse", "HEAD^{tree}"),
            "scans_split_sha256": _sha256_file(scans_split_path),
            "semantic_map_sha256": _sha256_file(semantic_map_path),
            "data_repository": "https://huggingface.co/datasets/3dlg-hcvc/MultiScan",
            "data_repository_revision": data_repository_revision,
            "data_license": "CC-BY-NC-4.0",
            "release_file_access": "gated_unauthenticated",
        },
        "environment": source_binding["environment"],
    }


def _parser() -> argparse.ArgumentParser:
    official = Path("/mnt/shared/ww/persist4d-multiscan/official-code")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--multiscan-repository", type=Path, default=official)
    parser.add_argument(
        "--scans-split",
        type=Path,
        default=official / "dataset/benchmark/scans_split.csv",
    )
    parser.add_argument(
        "--semantic-map",
        type=Path,
        default=official / "dataset/benchmark/object_semantic_label_map.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument(
        "--data-repository-revision",
        default=DATA_REPOSITORY_REVISION,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts/multiscan",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    binding = _reproducibility_binding(
        root=arguments.root.resolve(),
        multiscan_repository=arguments.multiscan_repository.resolve(),
        scans_split_path=arguments.scans_split.resolve(),
        semantic_map_path=arguments.semantic_map.resolve(),
        checkpoint_path=arguments.checkpoint.resolve(),
        data_repository_revision=arguments.data_repository_revision,
    )
    paths = build_inventory_artifacts(
        scans_split_path=arguments.scans_split.resolve(),
        output_directory=arguments.output_directory.resolve(),
        reproducibility_binding=binding,
    )
    inventory = json.loads(paths["multiscan_inventory.json"].read_text())
    print(
        "MULTISCAN_INVENTORY_OK "
        f"scans={inventory['scan_count']} scenes={inventory['scene_count']} "
        f"selected_scenes={inventory['selected_scene_count']} "
        f"selected_scans={inventory['selected_scan_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_inventory_artifacts"]
