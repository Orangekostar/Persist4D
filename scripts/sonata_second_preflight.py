"""Build the formal Sonata second-perception SP0 authorization package."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.sonata_second_preflight import (
    SONATA_BRANCH,
    SONATA_CONFIG_NAME,
    SONATA_START_COMMIT,
    SonataSecondPreflightError,
    build_sonata_data_manifest,
    build_sonata_environment_manifest,
    build_sonata_source_tree_contract,
    build_sonata_training_semantics,
    canonical_sha256,
    file_sha256,
    issue_sonata_preflight_authorization,
    portable_resolved_config,
    validate_sonata_training_config_contract,
)
from utils.sonata_weight_provenance import (
    OFFICIAL_SONATA_WEIGHT_SPEC,
    build_sonata_weight_manifest,
    validate_sonata_weight_manifest,
)

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "preflight"
)
SS1_ARTIFACT_DIR = (
    PROJECT_ROOT / "artifacts" / "sonata_second_perception_v1" / "weight"
)
EXPECTED_RIO_CONTENT = {
    "content_sha256": "bf1dc30493ae453d4202f3a0ef9ca28d35c8123df880e21770aca460e7f997f7",
    "file_count": 5855,
    "total_bytes": 5964404592,
}
EXPECTED_SCANNET_CONTENT = {
    "content_sha256": "6147418b6378d5b5b41cfd1082f336f1dabbc8d400989af3eee988c89b08676a",
    "file_count": 3134,
    "total_bytes": 12579495583,
}
EXPECTED_DATABASE_COUNTS = {
    "rio": {"train": 1178, "validation": 157, "t2_sequences": 1482},
    "scannet": {"train": 1201, "validation": 312, "test": 100},
}
P2_SCANNET_PREFLIGHT_SHA256 = (
    "8a6f0e07af0e5c566d027489cc24a7d60347c3a1a19a0bd21629464cd9cd7bd0"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SonataSecondPreflightError(f"{name} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise SonataSecondPreflightError(f"{name} must contain an object")
    return payload


def _require_repository_state() -> None:
    if Path.cwd().resolve() != PROJECT_ROOT.resolve():
        raise SonataSecondPreflightError("preflight must run from the repository root")
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if branch != SONATA_BRANCH:
        raise SonataSecondPreflightError(
            f"formal Sonata branch mismatch: expected {SONATA_BRANCH}, got {branch}"
        )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SONATA_START_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if ancestry.returncode != 0:
        raise SonataSecondPreflightError("authorized start commit is not an ancestor")


def _snapshot_weight(source: Path, training_output_dir: Path) -> Path:
    spec = OFFICIAL_SONATA_WEIGHT_SPEC
    snapshot = (
        Path(training_output_dir)
        / ".verified_inputs"
        / f"sonata-{spec.sha256}.pth"
    )
    if snapshot.exists() or snapshot.is_symlink():
        manifest = build_sonata_weight_manifest(
            snapshot,
            spec=spec,
            acquired_at=_utc_now(),
            download_source="https://huggingface.co/facebook/sonata",
            require_official=True,
        )
        validate_sonata_weight_manifest(
            manifest,
            snapshot,
            spec=spec,
            require_official=True,
        )
        return snapshot.resolve()

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{snapshot.name}.", suffix=".tmp", dir=snapshot.parent
    )
    temporary = Path(temporary_name)
    try:
        with Path(source).open("rb") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(source_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, snapshot)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = build_sonata_weight_manifest(
        snapshot,
        spec=spec,
        acquired_at=_utc_now(),
        download_source="https://huggingface.co/facebook/sonata",
        require_official=True,
    )
    validate_sonata_weight_manifest(
        manifest,
        snapshot,
        spec=spec,
        require_official=True,
    )
    return snapshot.resolve()


def _compose_config(weight_path: Path, training_output_dir: Path):
    previous_weight = os.environ.get("SONATA_CHECKPOINT")
    previous_output = os.environ.get("SONATA_OUTPUT_DIR")
    os.environ["SONATA_CHECKPOINT"] = str(weight_path)
    os.environ["SONATA_OUTPUT_DIR"] = str(training_output_dir)
    try:
        with initialize_config_dir(
            config_dir=str(PROJECT_ROOT / "conf"), version_base="1.2"
        ):
            cfg = compose(config_name=SONATA_CONFIG_NAME)
            OmegaConf.resolve(cfg)
            return cfg
    finally:
        if previous_weight is None:
            os.environ.pop("SONATA_CHECKPOINT", None)
        else:
            os.environ["SONATA_CHECKPOINT"] = previous_weight
        if previous_output is None:
            os.environ.pop("SONATA_OUTPUT_DIR", None)
        else:
            os.environ["SONATA_OUTPUT_DIR"] = previous_output


def _weight_flash_attn_active(path: Path) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SonataSecondPreflightError("verified weight config is unreadable") from error
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict):
        raise SonataSecondPreflightError("verified weight lacks a config mapping")
    return bool(config.get("enable_flash", False))


def _inspect_mixed_runtime(cfg: Any) -> dict[str, Any]:
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    dataset = hydra.utils.instantiate(cfg.data.train_dataset)
    if len(getattr(dataset, "datasets", [])) != 2:
        raise SonataSecondPreflightError("real mixed dataset has the wrong child count")
    if not isinstance(getattr(dataset, "sampler", None), WeightedRandomSampler):
        raise SonataSecondPreflightError("real mixed dataset sampler is not active")
    children = list(dataset.datasets)
    if [getattr(child, "dataset_name", None) for child in children] != [
        "rio",
        "scannet",
    ]:
        raise SonataSecondPreflightError("real mixed dataset identities mismatch")
    if [getattr(child, "temporal_window", None) for child in children] != [2, 1]:
        raise SonataSecondPreflightError("real mixed temporal windows mismatch")
    samples = [child[0] for child in children]
    collator = hydra.utils.instantiate(cfg.data.train_collation)
    data, targets, names = collator(samples)
    features = getattr(data, "features", None)
    coordinates = getattr(data, "coordinates", None)
    temporal_stages = getattr(data, "temporal_stages", None)
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or features.shape[1] != 9
        or not bool(torch.isfinite(features).all().item())
    ):
        raise SonataSecondPreflightError("real collated Sonata features are invalid")
    if (
        not isinstance(coordinates, torch.Tensor)
        or coordinates.ndim != 2
        or coordinates.shape[1] != 5
    ):
        raise SonataSecondPreflightError("real collated T2 coordinates are invalid")
    if not isinstance(targets, list) or len(targets) != 2:
        raise SonataSecondPreflightError("real collated targets are invalid")
    if not isinstance(temporal_stages, (list, tuple)) or len(temporal_stages) != 2:
        raise SonataSecondPreflightError("real temporal stage tensors are invalid")
    observed_stage_counts = [
        int(torch.unique(stages).numel()) for stages in temporal_stages
    ]
    if observed_stage_counts != [2, 1]:
        raise SonataSecondPreflightError(
            f"real collated temporal stages mismatch: {observed_stage_counts}"
        )
    return {
        "status": "pass",
        "implementation": "datasets.multi_dataset.MultiDataset",
        "sampler": type(dataset.sampler).__name__,
        "dataset_names": [child.dataset_name for child in children],
        "dataset_sizes": [len(child) for child in children],
        "weights": list(dataset.weights),
        "sampler_num_samples": int(dataset.sampler.num_samples),
        "sampler_seed": dataset.sampler_seed,
        "temporal_windows": [child.temporal_window for child in children],
        "collated_sample_names": [str(name) for name in names],
        "collated_point_count": int(features.shape[0]),
        "collated_feature_dimension": int(features.shape[1]),
        "collated_coordinate_dimension": int(coordinates.shape[1]),
        "collated_temporal_stage_counts": observed_stage_counts,
        "collated_features_finite": True,
    }


def _write_report(
    path: Path,
    *,
    source: dict[str, Any],
    data: dict[str, Any],
    semantics: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    mixed = data["mixed_runtime"]
    lines = [
        "# Sonata Second-Perception Preflight",
        "",
        f"- Gate: `{authorization['gate']}`",
        f"- Authorized source commit: `{source['source_commit']}`",
        f"- Source tree SHA-256: `{source['content_sha256']}`",
        f"- Data manifest SHA-256: `{authorization['bindings']['data_manifest_sha256']}`",
        f"- Effective global batch: {semantics['effective_global_batch']}",
        f"- Precision: `{semantics['precision']}`",
        f"- Mixed dataset sizes: `{mixed['dataset_sizes']}`",
        f"- Mixed sampling weights: `{mixed['weights']}`",
        f"- Real collated feature dimension: {mixed['collated_feature_dimension']}",
        f"- Real temporal stage counts: `{mixed['collated_temporal_stage_counts']}`",
        "",
        "The authorization is valid only while its source, canonical resolved",
        "config, SS1 weight manifest, data, runtime environment, and training",
        "semantics hashes remain unchanged and the age limit has not expired.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_preflight(
    *,
    weight_path: Path,
    training_output_dir: Path,
    artifact_dir: Path,
    max_age_seconds: int,
) -> dict[str, Path]:
    _require_repository_state()
    source_contract = build_sonata_source_tree_contract(
        PROJECT_ROOT, require_clean=True
    )
    if source_contract["branch"] != SONATA_BRANCH:
        raise SonataSecondPreflightError("source contract branch mismatch")

    ss1_manifest_path = SS1_ARTIFACT_DIR / "sonata_weight_manifest.json"
    load_audit_path = SS1_ARTIFACT_DIR / "sonata_load_key_audit.json"
    ss1_manifest = _load_json(ss1_manifest_path, name="SS1 weight manifest")
    load_audit = _load_json(load_audit_path, name="SS1 load-key audit")
    validate_sonata_weight_manifest(
        ss1_manifest,
        weight_path,
        spec=OFFICIAL_SONATA_WEIGHT_SPEC,
        require_official=True,
    )
    if (
        load_audit.get("gate") != "SW0-PASS"
        or load_audit.get("weight_sha256") != OFFICIAL_SONATA_WEIGHT_SPEC.sha256
        or load_audit.get("weight_manifest_sha256") != file_sha256(ss1_manifest_path)
    ):
        raise SonataSecondPreflightError("SS1 load-key audit is not bound to SW0")

    verified_weight = _snapshot_weight(weight_path, training_output_dir)
    cfg = _compose_config(verified_weight, training_output_dir)
    config_errors = validate_sonata_training_config_contract(
        cfg,
        expected_weight_path=verified_weight,
        expected_output_dir=training_output_dir,
    )
    if config_errors:
        raise SonataSecondPreflightError("; ".join(config_errors))
    portable_config = portable_resolved_config(
        cfg,
        expected_weight_path=verified_weight,
        expected_output_dir=training_output_dir,
        weight_sha256=OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
    )
    config_sha256 = canonical_sha256(portable_config)

    data_manifest = build_sonata_data_manifest(
        PROJECT_ROOT / "data" / "processed" / "rio",
        PROJECT_ROOT / "data" / "processed" / "scannet",
        expected_rio=EXPECTED_RIO_CONTENT,
        expected_scannet=EXPECTED_SCANNET_CONTENT,
        expected_database_counts=EXPECTED_DATABASE_COUNTS,
    )
    p2_preflight_path = PROJECT_ROOT / "artifacts" / "P2" / "scannet_preflight.json"
    if file_sha256(p2_preflight_path) != P2_SCANNET_PREFLIGHT_SHA256:
        raise SonataSecondPreflightError("frozen P2 data provenance artifact changed")
    data_manifest["provenance"] = {
        "frozen_p2_scannet_preflight_ref": (
            "repo:artifacts/P2/scannet_preflight.json"
        ),
        "frozen_p2_scannet_preflight_sha256": P2_SCANNET_PREFLIGHT_SHA256,
    }
    data_manifest["mixed_runtime"] = _inspect_mixed_runtime(cfg)

    flash_attn_active = _weight_flash_attn_active(verified_weight)
    environment = build_sonata_environment_manifest(
        source_tree_contract=source_contract,
        flash_attn_active=flash_attn_active,
    )
    weight_manifest_sha256 = file_sha256(ss1_manifest_path)
    load_key_audit_sha256 = file_sha256(load_audit_path)
    semantics = build_sonata_training_semantics(
        cfg,
        config_sha256=config_sha256,
        weight_manifest_sha256=weight_manifest_sha256,
        load_key_audit_sha256=load_key_audit_sha256,
    )
    semantics["verified_weight"] = {
        "reference": "external:sonata_verified_input/"
        + OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
        "sha256": OFFICIAL_SONATA_WEIGHT_SPEC.sha256,
        "bytes": OFFICIAL_SONATA_WEIGHT_SPEC.bytes,
        "regular_file": True,
        "symlink": False,
    }

    artifact_dir = Path(artifact_dir)
    resolved_json = artifact_dir / "resolved_config.json"
    resolved_yaml = artifact_dir / "resolved_config.yaml"
    environment_path = artifact_dir / "environment_manifest.json"
    data_path = artifact_dir / "data_manifest.json"
    semantics_path = artifact_dir / "training_semantics.json"
    _write_json(resolved_json, portable_config)
    _write_yaml(resolved_yaml, portable_config)
    _write_json(environment_path, environment)
    _write_json(data_path, data_manifest)
    _write_json(semantics_path, semantics)

    authorization = issue_sonata_preflight_authorization(
        source_tree_sha256=source_contract["content_sha256"],
        source_commit=source_contract["source_commit"],
        config_sha256=config_sha256,
        weight_manifest_sha256=weight_manifest_sha256,
        data_manifest_sha256=file_sha256(data_path),
        environment_manifest_sha256=file_sha256(environment_path),
        training_semantics_sha256=file_sha256(semantics_path),
        issued_at=_utc_now(),
        max_age_seconds=max_age_seconds,
    )
    authorization["repository"] = {
        "branch": SONATA_BRANCH,
        "authorized_start_commit": SONATA_START_COMMIT,
        "source_commit": source_contract["source_commit"],
        "start_is_ancestor": True,
        "repository_root": "repo:.",
    }
    payload = dict(authorization)
    payload.pop("authorization_sha256")
    authorization["authorization_sha256"] = canonical_sha256(payload)
    authorization_path = artifact_dir / "preflight_authorization.json"
    _write_json(authorization_path, authorization)
    report_path = artifact_dir / "preflight_report.md"
    _write_report(
        report_path,
        source=source_contract,
        data=data_manifest,
        semantics=semantics,
        authorization=authorization,
    )
    return {
        "resolved_config_json": resolved_json,
        "resolved_config_yaml": resolved_yaml,
        "environment_manifest": environment_path,
        "data_manifest": data_path,
        "training_semantics": semantics_path,
        "authorization": authorization_path,
        "report": report_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-path", type=Path, required=True)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--max-age-seconds", type=int, default=86400)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = run_preflight(
        weight_path=args.weight_path,
        training_output_dir=args.training_output_dir,
        artifact_dir=args.artifact_dir,
        max_age_seconds=args.max_age_seconds,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
