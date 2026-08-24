"""Audit the official MultiScan inventory and freeze the T>=3 collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets.multiscan_adapter import (
    MultiScanAdapterError,
    build_multiscan_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "487080cf31266f1572257e2aca36767e074b68b6"
MULTISCAN_COMMIT = "697bc9ec86fb7d34d47cb4cdbddcfc3c7f18c605"
DATA_REPOSITORY_REVISION = "c62c9aad850a8638ac3e42605926a65707600125"


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
