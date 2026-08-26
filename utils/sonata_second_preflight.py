"""Formal preflight contracts for Sonata second-perception training."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import stat
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omegaconf import OmegaConf

SONATA_CONFIG_NAME = "config_rescene4d_sonata_second"
SONATA_TARGET = "sonata_second_perception"
SONATA_EXPERIMENT_NAME = "rescene4d_sonata_second_perception"
SONATA_BRANCH = "research/persist4d-sonata-second-perception-v1"
SONATA_START_COMMIT = "e5d7f4e96fedc76c0c6d414ab293f54909c61df3"
SONATA_SOURCE_SCOPES = (
    "conf",
    "datasets",
    "models",
    "trainer",
    "utils",
    "scripts",
    "main_instance_segmentation.py",
)


class SonataSecondPreflightError(ValueError):
    """Raised when formal Sonata training cannot be authorized."""


def canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _stable_file_hash(path: Path) -> tuple[int, str]:
    path = Path(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise SonataSecondPreflightError(f"input file is unavailable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SonataSecondPreflightError(f"input is not a regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        observed_bytes = 0
        hasher = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                observed_bytes += len(block)
                hasher.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise SonataSecondPreflightError(f"input changed while hashing: {path}")
    return observed_bytes, hasher.hexdigest()


def directory_content_manifest(root: Path) -> dict[str, Any]:
    """Hash all regular files below a data root without exposing its path."""

    root = Path(root)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise SonataSecondPreflightError("formal data root is unavailable") from error
    if not resolved_root.is_dir():
        raise SonataSecondPreflightError("formal data root is not a directory")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if candidate.is_symlink():
            raise SonataSecondPreflightError(
                "symbolic links are forbidden inside formal data roots"
            )
        if not candidate.is_file():
            continue
        resolved_file = candidate.resolve(strict=True)
        if not resolved_file.is_relative_to(resolved_root):
            raise SonataSecondPreflightError("formal data file escapes its root")
        byte_size, sha256 = _stable_file_hash(resolved_file)
        entries.append(
            {
                "path": resolved_file.relative_to(resolved_root).as_posix(),
                "byte_size": byte_size,
                "sha256": sha256,
            }
        )
    if not entries:
        raise SonataSecondPreflightError("formal data root is empty")
    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["byte_size"] for entry in entries),
        "content_sha256": canonical_sha256(entries),
    }


def _git_output(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SonataSecondPreflightError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout


def build_sonata_source_tree_contract(
    repository: Path,
    *,
    scopes: tuple[str, ...] = SONATA_SOURCE_SCOPES,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Hash committed training source and reject scoped worktree changes."""

    repository = Path(repository).resolve()
    if require_clean:
        status = _git_output(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *scopes,
        )
        if status.strip():
            raise SonataSecondPreflightError(
                "training source tree contains uncommitted scoped files"
            )
    tracked_output = subprocess.run(
        ["git", "ls-files", "-z", "--", *scopes],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if tracked_output.returncode != 0:
        raise SonataSecondPreflightError("git ls-files failed for training source")
    paths = [
        Path(value.decode("utf-8"))
        for value in tracked_output.stdout.split(b"\0")
        if value
    ]
    if not paths:
        raise SonataSecondPreflightError("training source scope is empty")
    files: list[dict[str, Any]] = []
    for relative in sorted(paths, key=lambda value: value.as_posix()):
        byte_size, sha256 = _stable_file_hash(repository / relative)
        files.append(
            {
                "ref": f"repo:{relative.as_posix()}",
                "bytes": byte_size,
                "sha256": sha256,
            }
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": _git_output(repository, "rev-parse", "HEAD").strip(),
        "branch": _git_output(
            repository, "rev-parse", "--abbrev-ref", "HEAD"
        ).strip(),
        "scopes": list(scopes),
        "file_count": len(files),
        "content_sha256": canonical_sha256(files),
        "files": files,
    }


def _load_yaml(path: Path, *, name: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SonataSecondPreflightError(f"{name} is not readable YAML") from error


def _database_count(path: Path, *, name: str) -> int:
    payload = _load_yaml(path, name=name)
    if not isinstance(payload, (list, dict)):
        raise SonataSecondPreflightError(f"{name} must contain records")
    return len(payload)


def _normal_array_manifest(root: Path) -> dict[str, Any]:
    array_paths = sorted(root.rglob("*.npy"), key=lambda value: value.as_posix())
    if not array_paths:
        raise SonataSecondPreflightError("processed dataset has no NPY arrays")
    total_rows = 0
    minimum_columns: int | None = None
    dtypes: set[str] = set()
    for path in array_paths:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise SonataSecondPreflightError("processed NPY array is unreadable") from error
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] < 9:
            raise SonataSecondPreflightError(
                "processed NPY array lacks three normal columns"
            )
        total_rows += int(array.shape[0])
        minimum_columns = (
            int(array.shape[1])
            if minimum_columns is None
            else min(minimum_columns, int(array.shape[1]))
        )
        dtypes.add(str(array.dtype))
        for start in range(0, array.shape[0], 1_000_000):
            normals = np.asarray(array[start : start + 1_000_000, 6:9])
            if normals.shape[1] != 3 or not bool(np.isfinite(normals).all()):
                raise SonataSecondPreflightError(
                    "processed NPY array contains non-finite normals"
                )
    return {
        "array_count": len(array_paths),
        "row_count": total_rows,
        "minimum_column_count": minimum_columns,
        "dimension": 3,
        "dtypes": sorted(dtypes),
        "finite": True,
    }


def _taxonomy(path: Path, *, name: str) -> dict[str, Any]:
    payload = _load_yaml(path, name=name)
    if not isinstance(payload, dict):
        raise SonataSecondPreflightError(f"{name} must contain a mapping")
    validated = [
        (int(class_id), value.get("name"))
        for class_id, value in payload.items()
        if isinstance(value, dict) and value.get("validation") is True
    ]
    if not validated or any(not isinstance(label, str) for _, label in validated):
        raise SonataSecondPreflightError(f"{name} has no valid taxonomy")
    validated.sort()
    return {
        "class_ids": [class_id for class_id, _ in validated],
        "class_labels": [label for _, label in validated],
        "class_count": len(validated),
        "sha256": canonical_sha256(validated),
    }


def build_sonata_data_manifest(
    rio_root: Path,
    scannet_root: Path,
    *,
    expected_rio: Mapping[str, Any],
    expected_scannet: Mapping[str, Any],
    expected_database_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Verify processed assets required by the fixed 3RScan/ScanNet mix."""

    rio_root = Path(rio_root)
    scannet_root = Path(scannet_root)
    observed_content = {
        "rio": directory_content_manifest(rio_root),
        "scannet": directory_content_manifest(scannet_root),
    }
    expected_content = {
        "rio": dict(expected_rio),
        "scannet": dict(expected_scannet),
    }
    for name in ("rio", "scannet"):
        if observed_content[name] != expected_content[name]:
            raise SonataSecondPreflightError(
                f"{name} content manifest mismatch"
            )

    databases = {
        "rio": {
            "train": _database_count(
                rio_root / "train_database.yaml", name="RIO train database"
            ),
            "validation": _database_count(
                rio_root / "validation_database.yaml",
                name="RIO validation database",
            ),
            "t2_sequences": _database_count(
                rio_root / "sequence_database_sliding_2.yaml",
                name="RIO T2 sequence database",
            ),
        },
        "scannet": {
            "train": _database_count(
                scannet_root / "train_database.yaml",
                name="ScanNet train database",
            ),
            "validation": _database_count(
                scannet_root / "validation_database.yaml",
                name="ScanNet validation database",
            ),
            "test": _database_count(
                scannet_root / "test_database.yaml", name="ScanNet test database"
            ),
        },
    }
    for dataset_name, expected_counts in expected_database_counts.items():
        if databases.get(dataset_name) != dict(expected_counts):
            raise SonataSecondPreflightError(
                f"{dataset_name} database counts mismatch"
            )

    rio_taxonomy = _taxonomy(
        rio_root / "label_database.yaml", name="RIO label database"
    )
    scannet_taxonomy = _taxonomy(
        scannet_root / "label_database.yaml", name="ScanNet label database"
    )
    if rio_taxonomy != scannet_taxonomy:
        raise SonataSecondPreflightError("RIO and ScanNet taxonomies differ")

    return {
        "schema_version": 1,
        "status": "pass",
        "datasets": {
            "rio": {
                "root_ref": "repo:data/processed/rio",
                "content": observed_content["rio"],
                "databases": databases["rio"],
                "normals": _normal_array_manifest(rio_root),
                "label_database_ref": "repo:data/processed/rio/label_database.yaml",
                "t2_sequence_database_ref": (
                    "repo:data/processed/rio/sequence_database_sliding_2.yaml"
                ),
            },
            "scannet": {
                "root_ref": "external:data_root/scannet/"
                + observed_content["scannet"]["content_sha256"],
                "content": observed_content["scannet"],
                "databases": databases["scannet"],
                "normals": _normal_array_manifest(scannet_root),
                "label_database_ref": "external:scannet_processed/label_database.yaml",
            },
        },
        "taxonomy": {**rio_taxonomy, "status": "match"},
        "mix_contract": {
            "datasets": ["rio", "scannet"],
            "temporal_windows": [2, 1],
            "sampling_weights": [1.0, 0.8],
        },
    }


def portable_resolved_config(
    payload: Any,
    *,
    expected_weight_path: Path,
    expected_output_dir: Path,
    weight_sha256: str,
) -> dict[str, Any]:
    """Return the fully resolved config with private runtime roots canonicalized."""

    config = _resolved_config_payload(payload)
    if not _paths_equal(config.get("backbone", {}).get("name"), expected_weight_path):
        raise SonataSecondPreflightError("resolved backbone weight path mismatch")
    if not _paths_equal(config.get("general", {}).get("save_dir"), expected_output_dir):
        raise SonataSecondPreflightError("resolved training output path mismatch")
    if len(weight_sha256) != 64:
        raise SonataSecondPreflightError("weight SHA256 is invalid")
    weight_reference = f"external:sonata_verified_input/{weight_sha256}"
    output_reference = "external:sonata_training_output"

    def replace_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: replace_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace_paths(item) for item in value]
        if isinstance(value, str) and _paths_equal(value, expected_weight_path):
            return weight_reference
        if isinstance(value, str) and _paths_equal(value, expected_output_dir):
            return output_reference
        return value

    return replace_paths(config)


def build_sonata_training_semantics(
    cfg: Any,
    *,
    config_sha256: str,
    weight_manifest_sha256: str,
    load_key_audit_sha256: str,
) -> dict[str, Any]:
    """Derive the optimizer, objective, batch, and selection contract."""

    payload = _resolved_config_payload(cfg)
    for value, name in (
        (config_sha256, "config SHA256"),
        (weight_manifest_sha256, "weight manifest SHA256"),
        (load_key_audit_sha256, "load-key audit SHA256"),
    ):
        _digest(value, name=name, length=64)
    checkpoint_callbacks = [
        callback
        for callback in payload.get("callbacks", [])
        if isinstance(callback, Mapping)
        and callback.get("_target_")
        == "pytorch_lightning.callbacks.ModelCheckpoint"
    ]
    if len(checkpoint_callbacks) != 1:
        raise SonataSecondPreflightError(
            "training semantics require exactly one checkpoint callback"
        )
    checkpoint = checkpoint_callbacks[0]
    try:
        world_size = int(_path_value(payload, "general.gpus"))
        physical_batch = int(_path_value(payload, "data.batch_size"))
        accumulation = int(
            _path_value(payload, "trainer.accumulate_grad_batches")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise SonataSecondPreflightError("training batch semantics are invalid") from error
    effective_batch = world_size * physical_batch * accumulation
    return {
        "schema_version": 1,
        "status": "pass",
        "upstream": {
            "config_sha256": config_sha256,
            "weight_manifest_sha256": weight_manifest_sha256,
            "load_key_audit_sha256": load_key_audit_sha256,
        },
        "seed": _path_value(payload, "general.seed"),
        "epochs": _path_value(payload, "trainer.max_epochs"),
        "precision": _path_value(payload, "trainer.precision"),
        "world_size": world_size,
        "physical_batch_per_device": physical_batch,
        "accumulate_grad_batches": accumulation,
        "effective_global_batch": effective_batch,
        "optimizer": {
            "target": _path_value(payload, "optimizer._target_"),
            "lr": _path_value(payload, "optimizer.lr"),
            "betas": _path_value(payload, "optimizer.betas"),
            "eps": _path_value(payload, "optimizer.eps"),
            "weight_decay": _path_value(payload, "optimizer.weight_decay"),
            "amsgrad": _path_value(payload, "optimizer.amsgrad"),
        },
        "scheduler": {
            "target": _path_value(payload, "scheduler.scheduler._target_"),
            "max_lr": _path_value(payload, "scheduler.scheduler.max_lr"),
            "interval": _path_value(
                payload, "scheduler.pytorch_lightning_params.interval"
            ),
            "total_steps": _path_value(payload, "scheduler.scheduler.total_steps"),
        },
        "objective": {
            "implementation": "trainer.trainer.aggregate_objective_loss",
            "weighted": bool(
                _path_value(payload, "general.sonata_weighted_objective")
            ),
            "contrastive": bool(_path_value(payload, "loss.contrastive_loss")),
            "class_mask_dice_weights": [
                _path_value(payload, "matcher.cost_class"),
                _path_value(payload, "matcher.cost_mask"),
                _path_value(payload, "matcher.cost_dice"),
            ],
            "eos_coef": _path_value(payload, "loss.eos_coef"),
        },
        "freeze": {
            "mode": _path_value(payload, "general.freeze"),
            "frozen_encoder_eval": _path_value(
                payload, "general.frozen_encoder_eval"
            ),
        },
        "checkpoint_selection": {
            "monitor": checkpoint.get("monitor"),
            "mode": checkpoint.get("mode"),
            "save_top_k": checkpoint.get("save_top_k"),
            "save_last": checkpoint.get("save_last"),
            "persist4d_inputs_allowed": False,
        },
    }


def _package_version(distribution: str, module_name: str | None = None) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        if module_name is None:
            raise SonataSecondPreflightError(
                f"required package is unavailable: {distribution}"
            ) from None
        module = importlib.import_module(module_name)
        value = getattr(module, "__version__", None)
        if not isinstance(value, str) or not value:
            raise SonataSecondPreflightError(
                f"required package version is unavailable: {distribution}"
            )
        return value


def _module_repository_revision(module_name: str) -> str:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise SonataSecondPreflightError(
            f"module source is unavailable: {module_name}"
        )
    candidate = Path(module_file).resolve().parent
    while candidate != candidate.parent:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=candidate,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        candidate = candidate.parent
    raise SonataSecondPreflightError(
        f"module repository revision is unavailable: {module_name}"
    )


def build_sonata_environment_manifest(
    *,
    source_tree_contract: Mapping[str, Any],
    flash_attn_active: bool,
) -> dict[str, Any]:
    """Record the exact runtime and verified third-party source revisions."""

    import torch

    source_sha256 = source_tree_contract.get("content_sha256")
    source_commit = source_tree_contract.get("source_commit")
    _digest(source_sha256, name="source tree SHA256", length=64)
    _digest(source_commit, name="source commit", length=40)
    sonata_revision = _module_repository_revision("sonata")
    stmetrics_revision = _module_repository_revision("stmetrics")
    expected_revisions = {
        "sonata": "18c09ff8d713494f78a8213792262b910977a65d",
        "stmetrics": "640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f",
    }
    observed_revisions = {
        "sonata": sonata_revision,
        "stmetrics": stmetrics_revision,
    }
    if observed_revisions != expected_revisions:
        raise SonataSecondPreflightError("third-party source revision mismatch")

    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    gpu_memory = [
        int(torch.cuda.get_device_properties(index).total_memory)
        for index in range(torch.cuda.device_count())
    ]
    if not gpu_names:
        raise SonataSecondPreflightError("formal Sonata runtime has no CUDA device")
    return {
        "schema_version": 1,
        "status": "pass",
        "source": {
            "branch": source_tree_contract.get("branch"),
            "commit": source_commit,
            "tree_sha256": source_sha256,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pytorch_lightning": _package_version("pytorch-lightning"),
            "spconv": _package_version("spconv-cu126", "spconv"),
            "torch_scatter": _package_version("torch-scatter", "torch_scatter"),
            "flash_attn": {
                "version": _package_version("flash-attn", "flash_attn"),
                "active": bool(flash_attn_active),
            },
            "sonata": _package_version("sonata", "sonata"),
            "stmetrics": _package_version("stmetrics", "stmetrics"),
        },
        "hardware": {
            "cuda_device_count": len(gpu_names),
            "gpu_names": gpu_names,
            "gpu_total_memory_bytes": gpu_memory,
        },
        "third_party_sources": {
            "sonata": {
                "ref": "external:github/facebookresearch/sonata",
                "revision": sonata_revision,
            },
            "stmetrics": {
                "ref": "external:github/GradientSpaces/stmetrics",
                "revision": stmetrics_revision,
            },
            "rescene4d": {
                "ref": "external:github/GradientSpaces/rescene4d",
                "revision": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
                "local_start_commit": SONATA_START_COMMIT,
            },
            "scannet_tools": {
                "ref": "external:github/ScanNet/ScanNet",
                "revision": "3830fce7f8b2e48ef047ef7fd76ea5f62903f51c",
            },
        },
    }


def file_sha256(path: Path) -> str:
    """Return a stable SHA256 for a required regular file."""

    return _stable_file_hash(Path(path))[1]


def _digest(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SonataSecondPreflightError(f"{name} must be lowercase hexadecimal")
    return value


def issue_sonata_preflight_authorization(
    *,
    source_tree_sha256: str,
    source_commit: str,
    config_sha256: str,
    weight_manifest_sha256: str,
    data_manifest_sha256: str,
    environment_manifest_sha256: str,
    training_semantics_sha256: str,
    issued_at: str,
    max_age_seconds: int,
) -> dict[str, Any]:
    bindings = {
        "source_tree_sha256": _digest(
            source_tree_sha256, name="source tree SHA256", length=64
        ),
        "source_commit": _digest(source_commit, name="source commit", length=40),
        "config_sha256": _digest(config_sha256, name="config SHA256", length=64),
        "weight_manifest_sha256": _digest(
            weight_manifest_sha256, name="weight manifest SHA256", length=64
        ),
        "data_manifest_sha256": _digest(
            data_manifest_sha256, name="data manifest SHA256", length=64
        ),
        "environment_manifest_sha256": _digest(
            environment_manifest_sha256,
            name="environment manifest SHA256",
            length=64,
        ),
        "training_semantics_sha256": _digest(
            training_semantics_sha256,
            name="training semantics SHA256",
            length=64,
        ),
    }
    try:
        parsed = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise SonataSecondPreflightError("authorization timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SonataSecondPreflightError("authorization timestamp must be UTC")
    if not isinstance(max_age_seconds, int) or max_age_seconds <= 0:
        raise SonataSecondPreflightError("authorization max age must be positive")
    payload = {
        "schema_version": 1,
        "status": "pass",
        "gate": "SP0-PASS",
        "issued_at": issued_at,
        "max_age_seconds": max_age_seconds,
        "bindings": bindings,
    }
    payload["authorization_sha256"] = canonical_sha256(payload)
    return payload


def validate_sonata_preflight_authorization(
    authorization: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    now: datetime | None = None,
) -> None:
    if not isinstance(authorization, Mapping):
        raise SonataSecondPreflightError("authorization must be an object")
    payload = dict(authorization)
    observed_sha256 = payload.pop("authorization_sha256", None)
    if observed_sha256 != canonical_sha256(payload):
        raise SonataSecondPreflightError("authorization payload hash mismatch")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "pass"
        or payload.get("gate") != "SP0-PASS"
    ):
        raise SonataSecondPreflightError("authorization gate is not SP0-PASS")
    if payload.get("bindings") != dict(expected_bindings):
        raise SonataSecondPreflightError("authorization bindings mismatch")
    try:
        issued_at = datetime.fromisoformat(
            str(payload.get("issued_at", "")).replace("Z", "+00:00")
        )
        max_age = int(payload["max_age_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise SonataSecondPreflightError("authorization age contract is invalid") from error
    current = now or datetime.now(timezone.utc)
    age_seconds = (current - issued_at).total_seconds()
    if age_seconds < 0 or age_seconds > max_age:
        raise SonataSecondPreflightError("authorization is stale")


def _resolved_config_payload(cfg: Any) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        payload = OmegaConf.to_container(cfg, resolve=True)
    else:
        payload = copy.deepcopy(cfg)
    if not isinstance(payload, dict):
        raise TypeError("Sonata training config must resolve to a mapping")
    return payload


def _path_value(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


def _paths_equal(left: object, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def validate_sonata_training_config_contract(
    cfg: Any,
    *,
    expected_weight_path: Path,
    expected_output_dir: Path,
) -> list[str]:
    """Return every deviation from the frozen Sonata primary recipe."""

    try:
        payload = _resolved_config_payload(cfg)
    except Exception as error:  # noqa: BLE001 - malformed configs fail closed.
        return [f"training_config unavailable:{type(error).__name__}"]

    expected: dict[str, Any] = {
        "sonata_second_preflight.target": SONATA_TARGET,
        "sonata_second_preflight.artifact_path": (
            "artifacts/sonata_second_perception_v1/preflight/"
            "preflight_authorization.json"
        ),
        "aux_metric": None,
        "general.train_mode": True,
        "general.seed": 45,
        "general.freeze": "backbone_encoder",
        "general.frozen_encoder_eval": False,
        "general.gpus": 2,
        "general.project_name": SONATA_EXPERIMENT_NAME,
        "general.workspace": None,
        "general.experiment_name": SONATA_EXPERIMENT_NAME,
        "general.sonata_weighted_objective": True,
        "general.sonata_fail_closed_runtime": True,
        "data.batch_size": 4,
        "data.train_dataloader.batch_size": 4,
        "data.voxel_size": 0.02,
        "data.train_dataset._target_": (
            "datasets.multi_dataset.MultiDataset.from_config"
        ),
        "data.train_dataset.weights": [1.0, 0.8],
        "data.train_dataset.filter_out_classes": [0, 1, 255],
        "data.train_dataset.exclude_unsupervised_sequences": True,
        "data.train_dataset.fail_closed": True,
        "data.train_dataset.known_empty_scan_policy": "official_substitute",
        "data.train_dataset.epoch_sample_multiple": 32,
        "data.train_dataset.sampler_seed": 45,
        "data.validation_dataset.temporal_window": 2,
        "data.validation_dataset.filter_out_classes": [0, 1, 255],
        "data.validation_dataset.exclude_unsupervised_sequences": True,
        "data.validation_dataset.fail_closed": True,
        "data.validation_dataset.known_empty_scan_policy": "official_substitute",
        "data.test_dataset.temporal_window": 2,
        "data.test_dataset.filter_out_classes": [0, 1, 255],
        "data.test_dataset.exclude_unsupervised_sequences": True,
        "data.test_dataset.fail_closed": True,
        "data.test_dataset.known_empty_scan_policy": "official_substitute",
        "backbone._target_": "models.PointceptBackbone",
        "backbone.repo_id": "facebook/sonata",
        "backbone.model_lib": "sonata",
        "backbone.custom_config.enc_mode": False,
        "backbone.decoder_serializations": ["standard", "temporal_overlay"],
        "model.num_queries": 100,
        "model.non_parametric_queries": True,
        "model.random_query_both": False,
        "model.random_normal": False,
        "model.random_queries": False,
        "model.temporal_masking": True,
        "model.config.temporal_window": 2,
        "loss.contrastive_loss": False,
        "loss.eos_coef": 0.2,
        "matcher.cost_class": 2.0,
        "matcher.cost_mask": 5.0,
        "matcher.cost_dice": 2.0,
        "optimizer._target_": "torch.optim.AdamW",
        "optimizer.lr": 0.0005,
        "optimizer.betas": [0.9, 0.999],
        "optimizer.eps": 1e-8,
        "optimizer.weight_decay": 0.01,
        "optimizer.amsgrad": False,
        "scheduler.scheduler._target_": "torch.optim.lr_scheduler.OneCycleLR",
        "scheduler.scheduler.max_lr": 0.0005,
        "scheduler.pytorch_lightning_params.interval": "step",
        "trainer.max_epochs": 450,
        "trainer.accumulate_grad_batches": 4,
        "trainer.precision": "32-true",
    }
    errors: list[str] = []
    for path, expected_value in expected.items():
        try:
            observed = _path_value(payload, path)
        except KeyError:
            errors.append(f"training_config.{path} missing")
            continue
        if observed != expected_value:
            errors.append(
                f"training_config.{path} mismatch: expected "
                f"{expected_value!r}, got {observed!r}"
            )

    try:
        weight_name = _path_value(payload, "backbone.name")
    except KeyError:
        errors.append("training_config.backbone.name missing")
    else:
        if not _paths_equal(weight_name, expected_weight_path):
            errors.append("training_config.backbone.name mismatch")
    try:
        save_dir = _path_value(payload, "general.save_dir")
    except KeyError:
        errors.append("training_config.general.save_dir missing")
    else:
        if not _paths_equal(save_dir, expected_output_dir):
            errors.append("training_config.general.save_dir mismatch")

    datasets = payload.get("data", {}).get("train_dataset", {}).get("datasets")
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
    if datasets != expected_datasets:
        errors.append("training_config.data.train_dataset.datasets mismatch")

    try:
        effective_batch = (
            int(_path_value(payload, "general.gpus"))
            * int(_path_value(payload, "data.batch_size"))
            * int(_path_value(payload, "trainer.accumulate_grad_batches"))
        )
    except (KeyError, TypeError, ValueError):
        effective_batch = None
    if effective_batch != 32:
        errors.append(
            f"training_config.effective_global_batch mismatch: got {effective_batch!r}"
        )

    callbacks = payload.get("callbacks")
    checkpoint_callbacks = [
        callback
        for callback in callbacks or []
        if isinstance(callback, Mapping)
        and callback.get("_target_")
        == "pytorch_lightning.callbacks.ModelCheckpoint"
    ]
    if len(checkpoint_callbacks) != 1:
        errors.append("training_config.callbacks checkpoint count mismatch")
    else:
        checkpoint = checkpoint_callbacks[0]
        required_callback = {
            "monitor": "val_mean_t-AP",
            "mode": "max",
            "save_top_k": 1,
            "save_last": True,
        }
        for field, expected_value in required_callback.items():
            if checkpoint.get(field) != expected_value:
                errors.append(f"training_config.callbacks.{field} mismatch")
        monitor = str(checkpoint.get("monitor", "")).lower()
        if any(name in monitor for name in ("persist", "b2", "b3", "b4", "protocol")):
            errors.append("training_config.callbacks uses a forbidden monitor")
    return errors
