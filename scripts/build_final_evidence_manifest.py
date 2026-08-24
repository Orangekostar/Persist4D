"""Build the content-addressed manifest for the final paper evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPOSITORY_SOURCES = (
    "configs/final_evidence/capacity.yaml",
    "configs/final_evidence/rescan.yaml",
    "scripts/audit_rescan_coordinates.py",
    "scripts/audit_rescan_dataset.py",
    "scripts/build_final_evidence_manifest.py",
    "scripts/build_final_paper_evidence.py",
    "scripts/build_rescan_per_scene_effects.py",
    "scripts/evaluate_rescan_persist4d.py",
    "scripts/final_capacity_figures.py",
    "scripts/final_paper_figures.py",
    "scripts/rescan_protocol.py",
    "scripts/verify_final_evidence.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", *args), cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def build(root: Path, output_path: Path) -> dict[str, object]:
    artifact_root = root / "artifacts/final_evidence"
    repository_paths = [
        path
        for path in artifact_root.rglob("*")
        if path.is_file() and path.resolve() != output_path.resolve()
    ]
    repository_paths.extend(root / path for path in REPOSITORY_SOURCES)
    missing = [path for path in repository_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final-evidence source: {missing[0]}")

    dataset_manifest = json.loads(
        (artifact_root / "external/rescan_dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rescan_source = json.loads(
        (artifact_root / "external/rescan_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    living_source = json.loads(
        (artifact_root / "external/livingscenes_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_binding = json.loads(
        (artifact_root / "source_binding.json").read_text(encoding="utf-8")
    )
    rescan_raw = json.loads(
        (artifact_root / "external/rescan_raw.json").read_text(encoding="utf-8")
    )
    full_history_manifest_path = (
        artifact_root / "external/rescan_full_history_cache_manifest.json"
    )
    full_history_manifest = json.loads(
        full_history_manifest_path.read_text(encoding="utf-8")
    )
    full_history_raw = json.loads(
        (artifact_root / "external/rescan_full_history_raw.json").read_text(
            encoding="utf-8"
        )
    )
    local_provenance = rescan_raw["provenance"]
    full_history_provenance = full_history_raw["provenance"]
    generation_commit = _git(root, "rev-parse", "HEAD")
    manifest = {
        "classifications": {
            "architecture": "FINAL_LOCK",
            "capacity": "CAPACITY_100_OK",
            "external_validation": "EXTERNAL_INCONCLUSIVE",
            "final_paper_evidence": "PAPER_READY_INTERNAL_ONLY",
            "livingscenes": "NOT_RUN",
            "official_rescan_method": "RESCAN_METHOD_NOT_REPRODUCED",
        },
        "external_inputs": {
            "checkpoint": {
                "external_reference": "repo-ignored:checkpoints/rescene4d_concerto_t2_repro.ckpt",
                "sha256": "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e",
            },
            "livingscenes": {
                "commit": living_source["repository"]["commit"],
                "commit_tree": living_source["repository"]["commit_tree"],
                "weight_sha256": living_source["weights"]["sha256"],
                "weight_size_bytes": living_source["weights"]["size_bytes"],
            },
            "rescan": {
                "archive_sha256": "6985096973085de6c3f9b1463022edbbddc19f5bcb2c6536fb42f8e74ba28f9a",
                "archive_size_bytes": rescan_source["dataset_download"][
                    "content_length_bytes"
                ],
                "dataset_content_sha256": dataset_manifest["dataset_content_sha256"],
                "official_code_commit": rescan_source["repository"]["commit"],
                "official_code_tree": rescan_source["repository"]["commit_tree"],
            },
            "rescan_observation_cache": {
                "external_reference": "external:rescan/persist4d-cache",
                "manifest_sha256": local_provenance["cache_manifest_sha256"],
            },
            "rescan_full_history_observation_cache": {
                "entry_count": full_history_manifest["entry_count"],
                "external_reference": ("external:rescan/full-history-cache-v1"),
                "manifest_sha256": _sha256(full_history_manifest_path),
                "size_bytes": sum(
                    int(entry["bytes"]) for entry in full_history_manifest["entries"]
                ),
            },
        },
        "generation_base": {
            "branch": _git(root, "branch", "--show-current"),
            "commit": generation_commit,
            "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        },
        "frozen_runtime": {
            "checkpoint_sha256": "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e",
            "config_sha256": _sha256(root / "configs/final_evidence/rescan.yaml"),
            "current_evaluator_sha256": _sha256(
                root / "scripts/evaluate_rescan_persist4d.py"
            ),
            "current_source_commit": generation_commit,
            "dataset_manifest_sha256": _sha256(
                artifact_root / "external/rescan_dataset_manifest.json"
            ),
            "evaluator_sha256": local_provenance["evaluator_sha256"],
            "label_map_sha256": _sha256(
                artifact_root / "rescan_to_rescene_label_map.json"
            ),
            "source_commit": local_provenance["source_commit"],
        },
        "full_history_runtime": {
            "cache_manifest_sha256": full_history_provenance["cache_manifest_sha256"],
            "checkpoint_sha256": full_history_provenance["checkpoint_sha256"],
            "dataset_content_sha256": full_history_provenance["dataset_content_sha256"],
            "evaluator_sha256": full_history_provenance["evaluator_sha256"],
            "history_strategy": full_history_manifest["provenance"]["history_strategy"],
            "source_commit": full_history_provenance["source_commit"],
        },
        "repository_files": [
            _entry(root, path) for path in sorted(set(repository_paths))
        ],
        "reviewer_closure_binding": source_binding["reviewer_closure"],
        "schema_version": 1,
        "status": "frozen",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final_evidence/final_evidence_manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output
    manifest = build(root, output)
    print(f"FINAL_EVIDENCE_MANIFEST files={len(manifest['repository_files'])}")


if __name__ == "__main__":
    main()
