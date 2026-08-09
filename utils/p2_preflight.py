"""Shared identity and freshness checks for formal P2 training authorization."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
P2_CONFIG_NAME = "config_p2_rescene4d_concerto_t2"
P2_CONFIG_REF = "repo:conf/config_p2_rescene4d_concerto_t2.yaml"
P2_TARGET = "rescene4d_concerto_t2"
P2_EXPERIMENT_NAME = "rescene4d_concerto_t2_repro"
P2_SAVE_DIR = "checkpoints/rescene4d_concerto_t2_repro"
P2_PREFLIGHT_MAX_AGE_SECONDS = 24 * 60 * 60
P2_PREFLIGHT_SCHEMA_VERSION = 2
P2_AUTHORIZATION_SCHEMA_VERSION = 1
P2_TRAINING_CONTRACT_SCHEMA_VERSION = 1
OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
SCANNET_OFFICIAL_COMMIT = "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c"
SCANNET_OFFICIAL_REPOSITORY_REF = "external:github/ScanNet/ScanNet"
SCANNET_SPLIT_FILES = {
    "train": "scannetv2_train.txt",
    "validation": "scannetv2_val.txt",
    "test": "scannetv2_test.txt",
}
SCANNET_SPLIT_SHA256 = {
    "train": "96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8",
    "validation": "d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083",
    "test": "0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0",
}
P2_CONCERTO_CHECKPOINT_SHA256 = (
    "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"
)
P2_CONCERTO_CHECKPOINT_BYTES = 433_987_358
P2_RIO_SEQUENCE_DATABASE_REF = (
    "repo:data/processed/rio/sequence_database_sliding_2.yaml"
)
P2_RIO_SEQUENCE_DATABASE_SHA256 = (
    "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"
)
P2_KNOWN_EMPTY_RIO_SCAN_ID = "0171_01"
P2_KNOWN_EMPTY_RIO_SEQUENCES = [
    "scene0171_00-scene0171_01",
    "scene0171_01-scene0171_02",
]
P2_KNOWN_EMPTY_SCANNET_SCAN_IDS = ["scene0154_00", "scene0636_00"]
P2_TRAINING_SEMANTIC_SHA256 = (
    "4e6532a02bb67e1c1a9f990010d1ba89f4d40d596b9790f91b79ff70566565bc"
)
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


def _portable_root_ref(path: Path, repo_root: Path, role: str) -> str:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except (OSError, ValueError):
        return f"external:{role}"
    return f"repo:{relative.as_posix()}"


def _sha256_file_stable(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise OSError("input changed while hashing")
    return after.st_size, digest.hexdigest()


def _directory_content_manifest(root: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(root)
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda path: path.as_posix()):
        if candidate.is_symlink():
            raise OSError("symbolic links are forbidden in formal input roots")
        if not candidate.is_file():
            continue
        resolved_file = candidate.resolve(strict=True)
        if not resolved_file.is_relative_to(resolved_root):
            raise OSError("input file escapes its formal root")
        byte_size, sha256 = _sha256_file_stable(resolved_file)
        entries.append(
            {
                "path": resolved_file.relative_to(resolved_root).as_posix(),
                "byte_size": byte_size,
                "sha256": sha256,
            }
        )
    if not entries:
        raise ValueError("formal input root is empty")
    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["byte_size"] for entry in entries),
        "content_sha256": _canonical_sha256(entries),
    }


def build_p2_input_manifest(
    *,
    scannet_root: str | Path | None = None,
    rio_root: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash all processed training inputs into a compact deterministic manifest."""
    repository = Path(repo_root or REPO_ROOT).resolve()
    roots = {
        "scannet": Path(
            scannet_root or repository / "data" / "processed" / "scannet"
        ),
        "rio": Path(rio_root or repository / "data" / "processed" / "rio"),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass",
        "roots": {
            name: _portable_root_ref(path, repository, f"{name}_processed")
            for name, path in roots.items()
        },
    }
    errors: list[str] = []
    for name, root in roots.items():
        try:
            manifest[name] = _directory_content_manifest(root)
        except (OSError, ValueError):
            manifest[name] = {
                "file_count": 0,
                "total_bytes": 0,
                "content_sha256": None,
            }
            errors.append(f"{name}_input_manifest_failed")
    if errors:
        manifest["status"] = "fail"
        manifest["errors"] = errors
    return manifest


def build_scannet_official_split_identity(
    *,
    split_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(repo_root or REPO_ROOT).resolve()
    root = Path(
        split_dir
        or repository / "third_party" / "ScanNet" / "Tasks" / "Benchmark"
    )
    files: dict[str, Any] = {}
    status = "pass"
    for split, filename in SCANNET_SPLIT_FILES.items():
        path = root / filename
        observed_sha256: str | None = None
        scene_count = 0
        try:
            _, observed_sha256 = _sha256_file_stable(path)
            scenes = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            scene_count = len(scenes)
        except (OSError, UnicodeError):
            status = "fail"
        if (
            observed_sha256 != SCANNET_SPLIT_SHA256[split]
            or scene_count != OFFICIAL_SPLIT_COUNTS[split]
        ):
            status = "fail"
        files[split] = {
            "reference": (
                "repo:third_party/ScanNet/Tasks/Benchmark/" + filename
                if root.resolve()
                == (
                    repository
                    / "third_party"
                    / "ScanNet"
                    / "Tasks"
                    / "Benchmark"
                ).resolve()
                else _portable_root_ref(path, repository, "scannet_split")
            ),
            "expected_sha256": SCANNET_SPLIT_SHA256[split],
            "observed_sha256": observed_sha256,
            "expected_scene_count": OFFICIAL_SPLIT_COUNTS[split],
            "observed_scene_count": scene_count,
            "status": (
                "pass"
                if observed_sha256 == SCANNET_SPLIT_SHA256[split]
                and scene_count == OFFICIAL_SPLIT_COUNTS[split]
                else "fail"
            ),
        }
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        commit = None
    if commit != SCANNET_OFFICIAL_COMMIT:
        status = "fail"
    return {
        "status": status,
        "repository_ref": SCANNET_OFFICIAL_REPOSITORY_REF,
        "expected_commit": SCANNET_OFFICIAL_COMMIT,
        "observed_commit": commit,
        "files": files,
    }


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def p2_training_config_sha256(cfg: Any) -> str:
    """Hash the complete resolved training config, including the fixed gate path."""
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True)
    else:
        payload = copy.deepcopy(cfg)
    if not isinstance(payload, dict):
        raise TypeError("P2 training config must resolve to a mapping")
    return _canonical_sha256(payload)


def p2_training_semantic_sha256(cfg: Any) -> str:
    """Hash the fixed P2 behavior while normalizing the verified weight location."""
    payload = _resolved_config_payload(cfg)
    for backbone in (
        payload.get("backbone"),
        payload.get("model", {}).get("config", {}).get("backbone"),
    ):
        if isinstance(backbone, dict):
            backbone["name"] = "<verified-local-concerto-checkpoint>"
    return _canonical_sha256(payload)


def artifact_payload_sha256(artifact: Mapping[str, Any]) -> str:
    """Hash all artifact fields except the digest field itself."""
    payload = copy.deepcopy(dict(artifact))
    authorization = payload.get("authorization")
    if isinstance(authorization, dict):
        authorization.pop("artifact_payload_sha256", None)
    return _canonical_sha256(payload)


def issue_formal_authorization(
    preflight: dict[str, Any],
    cfg: Any,
    *,
    now: datetime | None = None,
) -> None:
    issued_at = now or datetime.now(timezone.utc)
    if issued_at.tzinfo is None:
        raise ValueError("formal authorization timestamp must be timezone-aware")
    issued_at = issued_at.astimezone(timezone.utc)
    preflight["authorization"] = {
        "schema_version": P2_AUTHORIZATION_SCHEMA_VERSION,
        "status": "issued",
        "config_ref": P2_CONFIG_REF,
        "config_sha256": p2_training_config_sha256(cfg),
        "expected_split_counts": dict(OFFICIAL_SPLIT_COUNTS),
        "issued_at_utc": issued_at.isoformat().replace("+00:00", "Z"),
        "max_age_seconds": P2_PREFLIGHT_MAX_AGE_SECONDS,
    }
    preflight["authorization"]["artifact_payload_sha256"] = (
        artifact_payload_sha256(preflight)
    )


def not_issued_authorization(reason: str) -> dict[str, Any]:
    return {
        "schema_version": P2_AUTHORIZATION_SCHEMA_VERSION,
        "status": "not_issued",
        "reason": reason,
    }


def _exact_counts(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(OFFICIAL_SPLIT_COUNTS)
        and all(type(value[key]) is int for key in OFFICIAL_SPLIT_COUNTS)
        and dict(value) == OFFICIAL_SPLIT_COUNTS
    )


def _resolved_config_payload(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True)
    else:
        payload = copy.deepcopy(cfg)
    if not isinstance(payload, dict):
        raise TypeError("P2 training config must resolve to a mapping")
    return payload


def _config_path_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def validate_p2_training_config_contract(cfg: Any) -> list[str]:
    """Return semantic deviations from the fixed formal P2 training profile."""
    try:
        payload = _resolved_config_payload(cfg)
    except Exception as exc:  # noqa: BLE001 - malformed config must fail closed.
        return [f"training_config unavailable:{type(exc).__name__}"]

    errors: list[str] = []
    expected_paths: dict[str, Any] = {
        "p2_preflight.target": P2_TARGET,
        "p2_preflight.artifact_path": (
            "artifacts/P2/scannet_preflight.json"
        ),
        "aux_metric": None,
        "general.train_mode": True,
        "general.seed": 45,
        "general.checkpoint": None,
        "general.backbone_checkpoint": None,
        "general.freeze": "backbone_encoder",
        "general.gpus": 2,
        "general.project_name": P2_EXPERIMENT_NAME,
        "general.workspace": None,
        "general.experiment_name": P2_EXPERIMENT_NAME,
        "general.save_dir": P2_SAVE_DIR,
        "general.p2_weighted_objective": True,
        "general.p2_fail_closed_runtime": True,
        "data.batch_size": 4,
        "data.train_dataloader.batch_size": 4,
        "data.voxel_size": 0.02,
        "data.train_dataset._target_": (
            "datasets.multi_dataset.MultiDataset.from_config"
        ),
        "data.train_dataset.weights": [1.0, 0.8],
        "data.train_dataset.filter_out_classes": [0, 1, 255],
        "data.train_dataset.fail_closed": True,
        "data.train_dataset.known_empty_scan_policy": "official_substitute",
        "data.train_dataset.epoch_sample_multiple": 32,
        "data.train_dataset.sampler_seed": 45,
        "data.validation_dataset._target_": (
            "datasets.semseg.SemanticSegmentationDataset"
        ),
        "data.validation_dataset.dataset_name": "rio",
        "data.validation_dataset.data_dir": "data/processed/rio",
        "data.validation_dataset.temporal_window": 2,
        "data.validation_dataset.filter_out_classes": [0, 1, 255],
        "data.validation_dataset.fail_closed": True,
        "data.validation_dataset.known_empty_scan_policy": (
            "official_substitute"
        ),
        "data.test_dataset._target_": (
            "datasets.semseg.SemanticSegmentationDataset"
        ),
        "data.test_dataset.dataset_name": "rio",
        "data.test_dataset.data_dir": "data/processed/rio",
        "data.test_dataset.temporal_window": 2,
        "data.test_dataset.filter_out_classes": [0, 1, 255],
        "data.test_dataset.fail_closed": True,
        "data.test_dataset.known_empty_scan_policy": "official_substitute",
        "backbone._target_": "models.PointceptBackbone",
        "backbone.model_lib": "concerto",
        "backbone.decoder_serializations": [
            "standard",
            "temporal_overlay",
        ],
        "model.num_queries": 100,
        "model.non_parametric_queries": True,
        "model.random_query_both": False,
        "model.random_normal": False,
        "model.random_queries": False,
        "model.temporal_masking": False,
        "model.config.temporal_window": 2,
        "loss.eos_coef": 0.2,
        "loss.contrastive_loss": True,
        "loss.contrastive_loss_type": "infoNCE",
        "matcher.cost_class": 2.0,
        "matcher.cost_mask": 5.0,
        "matcher.cost_dice": 2.0,
        "optimizer._target_": "torch.optim.AdamW",
        "optimizer.lr": 0.0005,
        "optimizer.betas": [0.9, 0.999],
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.01,
        "optimizer.amsgrad": False,
        "scheduler.scheduler._target_": (
            "torch.optim.lr_scheduler.OneCycleLR"
        ),
        "scheduler.scheduler.max_lr": 0.0005,
        "scheduler.scheduler.epochs": 450,
        "scheduler.scheduler.total_steps": -1,
        "scheduler.scheduler.pct_start": 0.3,
        "scheduler.scheduler.anneal_strategy": "cos",
        "scheduler.scheduler.cycle_momentum": True,
        "scheduler.scheduler.base_momentum": 0.85,
        "scheduler.scheduler.max_momentum": 0.95,
        "scheduler.scheduler.div_factor": 25.0,
        "scheduler.scheduler.final_div_factor": 10000.0,
        "scheduler.scheduler.three_phase": False,
        "scheduler.scheduler.last_epoch": -1,
        "scheduler.pytorch_lightning_params.interval": "step",
        "trainer.max_epochs": 450,
        "trainer.accumulate_grad_batches": 4,
        "trainer.precision": "32-true",
        "trainer.strategy": "ddp_find_unused_parameters_true",
    }
    for path, expected in expected_paths.items():
        try:
            observed = _config_path_value(payload, path)
        except KeyError:
            errors.append(f"training_config.{path} is missing")
            continue
        if observed != expected or (
            isinstance(expected, bool) and observed is not expected
        ):
            errors.append(f"training_config.{path} mismatch")

    expected_datasets = [
        {
            "target": "datasets.semseg.SemanticSegmentationDataset",
            "dataset_name": "rio",
            "data_dir": "data/processed/rio",
            "label_db_filepath": "data/processed/rio/label_database.yaml",
            "color_mean_std": "data/processed/rio/color_mean_std.yaml",
            "temporal_window": 2,
        },
        {
            "target": "datasets.semseg.SemanticSegmentationDataset",
            "dataset_name": "scannet",
            "data_dir": "data/processed/scannet",
            "label_db_filepath": "data/processed/scannet/label_database.yaml",
            "color_mean_std": "data/processed/scannet/color_mean_std.yaml",
            "temporal_window": 1,
        },
    ]
    if payload.get("data", {}).get("train_dataset", {}).get(
        "datasets"
    ) != expected_datasets:
        errors.append("training_config.data.train_dataset.datasets mismatch")

    expected_logging = [
        {
            "_target_": "pytorch_lightning.loggers.CSVLogger",
            "save_dir": P2_SAVE_DIR,
            "name": "local_metrics",
        }
    ]
    if payload.get("logging") != expected_logging:
        errors.append("training_config.logging mismatch")

    expected_callbacks = [
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": "val_mean_t-AP",
            "mode": "max",
            "save_top_k": 1,
            "save_last": True,
            "dirpath": P2_SAVE_DIR,
            "filename": "epoch={epoch:03d}-val_mean_t-AP={val_mean_t-AP:.3f}",
            "every_n_epochs": 1,
            "save_on_train_epoch_end": False,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": None,
            "save_top_k": -1,
            "save_last": False,
            "dirpath": P2_SAVE_DIR,
            "filename": "periodic-epoch={epoch:03d}",
            "every_n_epochs": 25,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
            "monitor": None,
            "save_top_k": -1,
            "save_last": False,
            "dirpath": "checkpoints",
            "filename": P2_EXPERIMENT_NAME,
            "every_n_epochs": 450,
            "save_on_train_epoch_end": True,
            "save_weights_only": False,
            "auto_insert_metric_name": False,
            "enable_version_counter": False,
        },
        {
            "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
        },
    ]
    if payload.get("callbacks") != expected_callbacks:
        errors.append("training_config.callbacks mismatch")
    observed_semantic_sha256 = p2_training_semantic_sha256(payload)
    if observed_semantic_sha256 != P2_TRAINING_SEMANTIC_SHA256:
        errors.append(
            "training_config.semantic_sha256 mismatch: expected "
            f"{P2_TRAINING_SEMANTIC_SHA256}, got {observed_semantic_sha256}"
        )
    return errors


def _validate_split_metadata(artifact: Mapping[str, Any], errors: list[str]) -> None:
    if artifact.get("split_metadata_status") != "pass":
        errors.append("split_metadata_status is not pass")
    records = artifact.get("split_metadata")
    if not isinstance(records, Mapping):
        errors.append("split_metadata is missing")
        return
    for split, expected in OFFICIAL_SPLIT_COUNTS.items():
        record = records.get(split)
        if not isinstance(record, Mapping):
            errors.append(f"split_metadata.{split} is missing")
            continue
        for field in ("expected", "observed", "unique"):
            if record.get(field) != expected:
                errors.append(f"split_metadata.{split}.{field} mismatch")
        if record.get("status") != "pass":
            errors.append(f"split_metadata.{split}.status is not pass")


def _expected_official_split_identity() -> dict[str, Any]:
    return {
        "status": "pass",
        "repository_ref": SCANNET_OFFICIAL_REPOSITORY_REF,
        "expected_commit": SCANNET_OFFICIAL_COMMIT,
        "observed_commit": SCANNET_OFFICIAL_COMMIT,
        "files": {
            split: {
                "reference": (
                    "repo:third_party/ScanNet/Tasks/Benchmark/"
                    + SCANNET_SPLIT_FILES[split]
                ),
                "expected_sha256": SCANNET_SPLIT_SHA256[split],
                "observed_sha256": SCANNET_SPLIT_SHA256[split],
                "expected_scene_count": OFFICIAL_SPLIT_COUNTS[split],
                "observed_scene_count": OFFICIAL_SPLIT_COUNTS[split],
                "status": "pass",
            }
            for split in OFFICIAL_SPLIT_COUNTS
        },
    }


def _validate_official_split_identity(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    if artifact.get("official_split_identity") != (
        _expected_official_split_identity()
    ):
        errors.append("official_split_identity mismatch")


def _validate_input_manifest(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    manifest = artifact.get("input_manifest")
    if not isinstance(manifest, Mapping):
        errors.append("input_manifest is missing")
        return
    if manifest.get("schema_version") != 1:
        errors.append("input_manifest.schema_version mismatch")
    if manifest.get("status") != "pass":
        errors.append("input_manifest.status is not pass")
    if manifest.get("roots") != {
        "scannet": "repo:data/processed/scannet",
        "rio": "repo:data/processed/rio",
    }:
        errors.append("input_manifest.roots mismatch")
    for dataset in ("scannet", "rio"):
        record = manifest.get(dataset)
        if not isinstance(record, Mapping):
            errors.append(f"input_manifest.{dataset} is missing")
            continue
        if type(record.get("file_count")) is not int or record["file_count"] < 1:
            errors.append(f"input_manifest.{dataset}.file_count invalid")
        if type(record.get("total_bytes")) is not int or record["total_bytes"] < 1:
            errors.append(f"input_manifest.{dataset}.total_bytes invalid")
        sha256 = record.get("content_sha256")
        if not isinstance(sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", sha256
        ):
            errors.append(f"input_manifest.{dataset}.content_sha256 invalid")


def _validate_model_checkpoint(artifact: Mapping[str, Any], errors: list[str]) -> None:
    checkpoint = artifact.get("model_checkpoint")
    if not isinstance(checkpoint, Mapping):
        errors.append("model_checkpoint is missing")
        return
    expected = {
        "expected_sha256": P2_CONCERTO_CHECKPOINT_SHA256,
        "observed_sha256": P2_CONCERTO_CHECKPOINT_SHA256,
        "expected_byte_size": P2_CONCERTO_CHECKPOINT_BYTES,
        "observed_byte_size": P2_CONCERTO_CHECKPOINT_BYTES,
        "status": "pass",
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            errors.append(f"model_checkpoint.{field} mismatch")
    reference = checkpoint.get("reference")
    if not isinstance(reference, str) or not reference.startswith(
        ("repo:", "external:", "local_cache:")
    ):
        errors.append("model_checkpoint.reference invalid")


def _validate_asset_summaries(artifact: Mapping[str, Any], errors: list[str]) -> None:
    expected_total = sum(OFFICIAL_SPLIT_COUNTS.values())
    expected_instances = (
        OFFICIAL_SPLIT_COUNTS["train"] + OFFICIAL_SPLIT_COUNTS["validation"]
    )
    raw = artifact.get("raw_assets")
    if not isinstance(raw, Mapping):
        errors.append("raw_assets is missing")
    else:
        expected_raw = {
            "status": "pass",
            "expected_scene_count": expected_total,
            "complete_scene_count": expected_total,
            "missing_asset_count": 0,
        }
        for field, expected in expected_raw.items():
            if raw.get(field) != expected:
                errors.append(f"raw_assets.{field} mismatch")

    processed = artifact.get("processed_assets")
    if not isinstance(processed, Mapping):
        errors.append("processed_assets is missing")
        return
    expected_processed = {
        "status": "pass",
        "expected_scene_count": expected_total,
        "database_scene_count": expected_total,
        "npy_scene_count": expected_total,
        "instance_gt_scene_count": expected_instances,
    }
    for field, expected in expected_processed.items():
        if processed.get(field) != expected:
            errors.append(f"processed_assets.{field} mismatch")
    by_split = processed.get("by_split")
    if not isinstance(by_split, Mapping):
        errors.append("processed_assets.by_split is missing")
        return
    for split, expected in OFFICIAL_SPLIT_COUNTS.items():
        record = by_split.get(split)
        if not isinstance(record, Mapping):
            errors.append(f"processed_assets.by_split.{split} is missing")
            continue
        expected_record = {
            "expected_scene_count": expected,
            "database_record_count": expected,
            "database_scene_count": expected,
            "npy_scene_count": expected,
            "instance_gt_scene_count": 0 if split == "test" else expected,
            "status": "pass",
        }
        for field, expected_value in expected_record.items():
            if record.get(field) != expected_value:
                errors.append(
                    f"processed_assets.by_split.{split}.{field} mismatch"
                )


def _validate_taxonomy_and_mix(artifact: Mapping[str, Any], errors: list[str]) -> None:
    expected_taxonomy = {
        "status": "pass",
        "valid_class_ids": NYU40_INSTANCE_IDS,
        "class_labels": NYU40_INSTANCE_LABELS,
        "class_count": len(NYU40_INSTANCE_IDS),
    }
    for artifact_field, dataset_name in (
        ("class_taxonomy", "scannet"),
        ("rio_class_taxonomy", "rio"),
    ):
        taxonomy = artifact.get(artifact_field)
        if not isinstance(taxonomy, Mapping):
            errors.append(f"{artifact_field} is missing")
            continue
        for field, expected in {
            **expected_taxonomy,
            "name": dataset_name,
        }.items():
            if taxonomy.get(field) != expected:
                errors.append(f"{artifact_field}.{field} mismatch")

    mix = artifact.get("mix_instantiation")
    if not isinstance(mix, Mapping):
        errors.append("mix_instantiation is missing")
        return
    expected_mix = {
        "attempted": True,
        "status": "pass",
        "implementation": "datasets.multi_dataset.MultiDataset",
        "dataset_names": ["rio", "scannet"],
        "weights": [1.0, 0.8],
        "temporal_windows": [2, 1],
        "sampler": "WeightedRandomSampler",
    }
    for field, expected in expected_mix.items():
        if mix.get(field) != expected:
            errors.append(f"mix_instantiation.{field} mismatch")
    sizes = mix.get("dataset_sizes")
    if (
        not isinstance(sizes, list)
        or len(sizes) != 2
        or sizes != [1178, OFFICIAL_SPLIT_COUNTS["train"]]
    ):
        errors.append("mix_instantiation.dataset_sizes mismatch")


def _validate_known_empty_substitutions(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    evidence = artifact.get("known_empty_scan_substitutions")
    expected = {
        "status": "pass",
        "dataset": "rio",
        "temporal_window": 2,
        "known_empty_scan_id": P2_KNOWN_EMPTY_RIO_SCAN_ID,
        "policy": "official_substitute",
        "sequence_database_ref": P2_RIO_SEQUENCE_DATABASE_REF,
        "expected_sequence_database_sha256": P2_RIO_SEQUENCE_DATABASE_SHA256,
        "observed_sequence_database_sha256": P2_RIO_SEQUENCE_DATABASE_SHA256,
        "fail_closed": {
            "train": True,
            "validation": True,
            "test": True,
        },
        "affected_sequences": P2_KNOWN_EMPTY_RIO_SEQUENCES,
        "scannet_known_empty_scan_ids": P2_KNOWN_EMPTY_SCANNET_SCAN_IDS,
    }
    if not isinstance(evidence, Mapping):
        errors.append("known_empty_scan_substitutions is missing")
        return
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            errors.append(
                f"known_empty_scan_substitutions.{field} mismatch"
            )


def _validate_data_root_bindings(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    bindings = artifact.get("data_root_bindings")
    expected_roots = {
        "raw_scannet": "repo:data/raw/scannet/scannet",
        "scannet": "repo:data/processed/scannet",
        "rio": "repo:data/processed/rio",
        "split_metadata": "repo:third_party/ScanNet/Tasks/Benchmark",
        "test_segments": "repo:data/raw/scannet_test_segments",
    }
    expected = {
        "status": "pass",
        "expected": expected_roots,
        "observed": expected_roots,
    }
    if not isinstance(bindings, Mapping):
        errors.append("data_root_bindings is missing")
        return
    for field, expected_value in expected.items():
        if bindings.get(field) != expected_value:
            errors.append(f"data_root_bindings.{field} mismatch")


def _validate_rio_path_integrity(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    evidence = artifact.get("rio_path_integrity")
    expected = {
        "status": "pass",
        "database_record_counts": {"train": 1178, "validation": 157},
        "sequence_record_count": 1482,
        "content_validation": "pass",
    }
    if not isinstance(evidence, Mapping):
        errors.append("rio_path_integrity is missing")
        return
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            errors.append(f"rio_path_integrity.{field} mismatch")
    supervised_record_count = evidence.get("supervised_record_count")
    if (
        type(supervised_record_count) is not int
        or supervised_record_count < 1
        or supervised_record_count > 1335
    ):
        errors.append("rio_path_integrity.supervised_record_count invalid")
    if evidence.get("unsupervised_sequences") != []:
        errors.append("rio_path_integrity.unsupervised_sequences mismatch")


def _parse_issued_at(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _authorization_age_error(
    issued_at: datetime,
    checked_at: datetime,
) -> str | None:
    if checked_at.tzinfo is None:
        raise ValueError("authorization check timestamp must be timezone-aware")
    age_seconds = (
        checked_at.astimezone(timezone.utc) - issued_at
    ).total_seconds()
    if age_seconds < -300:
        return "authorization.issued_at_utc is in the future"
    if age_seconds > P2_PREFLIGHT_MAX_AGE_SECONDS:
        return "authorization is stale"
    return None


def validate_p2_preflight_authorization(
    cfg: Any,
    artifact: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> list[str]:
    errors = validate_p2_training_config_contract(cfg)
    if artifact.get("schema_version") != P2_PREFLIGHT_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if artifact.get("formal_p2_training_authorized") is not True:
        errors.append("formal_p2_training_authorized is not true")
    if artifact.get("status") != "pass":
        errors.append("status is not pass")
    if artifact.get("official_source_commit") != OFFICIAL_SOURCE_COMMIT:
        errors.append("official_source_commit mismatch")
    if artifact.get("errors") != []:
        errors.append("errors is not empty")
    config_contract = artifact.get("config_contract")
    if not isinstance(config_contract, Mapping):
        errors.append("config_contract is missing")
    else:
        if config_contract.get("status") != "pass":
            errors.append("config_contract.status is not pass")
        if config_contract.get("errors") != []:
            errors.append("config_contract.errors is not empty")
        expected_contract = {
            "schema_version": P2_TRAINING_CONTRACT_SCHEMA_VERSION,
            "expected_semantic_sha256": P2_TRAINING_SEMANTIC_SHA256,
            "observed_semantic_sha256": P2_TRAINING_SEMANTIC_SHA256,
        }
        for field, expected in expected_contract.items():
            if config_contract.get(field) != expected:
                errors.append(f"config_contract.{field} mismatch")
    if not _exact_counts(artifact.get("expected_split_counts")):
        errors.append("expected_split_counts are not official")
    _validate_split_metadata(artifact, errors)
    _validate_official_split_identity(artifact, errors)
    _validate_model_checkpoint(artifact, errors)
    _validate_asset_summaries(artifact, errors)
    _validate_taxonomy_and_mix(artifact, errors)
    _validate_known_empty_substitutions(artifact, errors)
    _validate_data_root_bindings(artifact, errors)
    _validate_rio_path_integrity(artifact, errors)
    _validate_input_manifest(artifact, errors)

    authorization = artifact.get("authorization")
    if not isinstance(authorization, Mapping):
        errors.append("authorization is missing")
        return errors
    if authorization.get("schema_version") != P2_AUTHORIZATION_SCHEMA_VERSION:
        errors.append("authorization.schema_version mismatch")
    if authorization.get("status") != "issued":
        errors.append("authorization.status is not issued")
    if authorization.get("config_ref") != P2_CONFIG_REF:
        errors.append("authorization.config_ref mismatch")
    if not _exact_counts(authorization.get("expected_split_counts")):
        errors.append("authorization.expected_split_counts are not official")
    if authorization.get("max_age_seconds") != P2_PREFLIGHT_MAX_AGE_SECONDS:
        errors.append("authorization.max_age_seconds mismatch")
    expected_payload_sha = artifact_payload_sha256(artifact)
    if authorization.get("artifact_payload_sha256") != expected_payload_sha:
        errors.append("authorization.artifact_payload_sha256 mismatch")
    try:
        config_sha = p2_training_config_sha256(cfg)
    except Exception as exc:  # noqa: BLE001 - unresolvable config must fail closed.
        errors.append(f"authorization.config_sha256 unavailable:{type(exc).__name__}")
    else:
        if authorization.get("config_sha256") != config_sha:
            errors.append("authorization.config_sha256 mismatch")

    issued_at = _parse_issued_at(authorization.get("issued_at_utc"))
    if issued_at is None:
        errors.append("authorization.issued_at_utc invalid")
    else:
        checked_at = now or datetime.now(timezone.utc)
        age_error = _authorization_age_error(issued_at, checked_at)
        if age_error is not None:
            errors.append(age_error)
    current_inputs_revalidated = False
    if not errors:
        current_inputs_revalidated = True
        current_split_identity = build_scannet_official_split_identity()
        if current_split_identity != artifact.get("official_split_identity"):
            errors.append("current official_split_identity mismatch")
        current_input_manifest = build_p2_input_manifest()
        if current_input_manifest != artifact.get("input_manifest"):
            errors.append("current input_manifest mismatch")
    if current_inputs_revalidated and issued_at is not None:
        final_checked_at = now or datetime.now(timezone.utc)
        final_age_error = _authorization_age_error(issued_at, final_checked_at)
        if final_age_error is not None and final_age_error not in errors:
            errors.append(final_age_error)
    return errors


def require_p2_preflight_authorization(
    cfg: Any,
    *,
    artifact_path: str | Path | None = None,
    now: datetime | None = None,
) -> Path:
    if artifact_path is None:
        marker = cfg.get("p2_preflight") if hasattr(cfg, "get") else None
        artifact_path = (
            marker.get("artifact_path")
            if isinstance(marker, Mapping) and marker.get("artifact_path")
            else REPO_ROOT / "artifacts" / "P2" / "scannet_preflight.json"
        )
    path = Path(str(artifact_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise RuntimeError(f"P2 preflight artifact is missing: {path}")
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"P2 preflight artifact is unreadable: {type(exc).__name__}"
        ) from exc
    if not isinstance(artifact, Mapping):
        raise RuntimeError(  # noqa: TRY004 - keep authorization failures uniform.
            "P2 preflight artifact root is not a mapping"
        )
    errors = validate_p2_preflight_authorization(cfg, artifact, now=now)
    if errors:
        raise RuntimeError(
            "P2 preflight authorization rejected: " + "; ".join(errors)
        )
    return path
