"""Shared identity and freshness checks for formal P2 training authorization."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
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
P2_PREFLIGHT_SCHEMA_VERSION = 4
P2_AUTHORIZATION_SCHEMA_VERSION = 1
P2_TRAINING_CONTRACT_SCHEMA_VERSION = 1
P2_SOURCE_TREE_CONTRACT_SCHEMA_VERSION = 1
P2_ALLOWED_SOURCE_DIRTY_PREFIXES = ("artifacts/P2/",)
P2_RUNTIME_SOURCE_CONTRACT_SCHEMA_VERSION = 1
P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION = 2
P2_RUNTIME_ENVIRONMENT_VERSIONS = {
    "python": "3.10.20",
    "torch": "2.6.0+cu126",
    "cuda": "12.6",
    "cudnn": "9.5.1",
    "nccl": "2.21.5",
    "pytorch_lightning": "2.6.5",
    "hydra_core": "1.3.4",
    "omegaconf": "2.3.1",
    "spconv": "2.3.8",
    "cumm": "0.7.11",
    "flash_attn": "2.8.3",
    "torch_scatter": "2.1.2+pt26cu126",
    "pointnet2": "0.0.0",
    "cuda_runtime_package": "12.6.77",
    # Prefix the four-component package version so privacy scanners cannot
    # mistake it for an IPv4 address.
    "cudnn_package": "v9.5.1.17",
    "nccl_package": "2.21.5",
}
P2_RUNTIME_SOURCE_REPOSITORIES = {
    "concerto": {
        "relative_root": "third_party/concerto",
        "module": "concerto",
        "expected_commit": "10a7d17cff4dddff028f1522c2e72de4c4515df7",
    },
    "sonata": {
        "relative_root": "third_party/sonata",
        "module": "sonata",
        "expected_commit": "18c09ff8d713494f78a8213792262b910977a65d",
    },
    "detectron2": {
        "relative_root": "third_party/detectron2",
        "module": "detectron2",
        "expected_commit": "b4a4a3bd136852dae5fb1de37978dee412653e31",
        "native_extensions": {
            "detectron2._C": {
                "relative_path": (
                    "detectron2/_C.cpython-310-x86_64-linux-gnu.so"
                ),
                "expected_byte_size": 1_291_352,
                "expected_sha256": (
                    "4f66dbe809bfcba0015f71d26ded0c1922e60bad0d92d5f63ecb9e300ae5cba8"
                ),
            }
        },
    },
    "stmetrics": {
        "relative_root": "third_party/stmetrics",
        "module": "stmetrics",
        "expected_commit": "640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f",
    },
}
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
P2_SCANNET_SEQUENCE_FILTER_COUNTS = {
    "train": {
        "sequence_count": OFFICIAL_SPLIT_COUNTS["train"],
        "excluded_count": len(P2_KNOWN_EMPTY_SCANNET_SCAN_IDS),
        "retained_count": OFFICIAL_SPLIT_COUNTS["train"]
        - len(P2_KNOWN_EMPTY_SCANNET_SCAN_IDS),
    }
}
P2_TRAINING_SEMANTIC_SHA256 = (
    "ce6be7b458d8371202ce19aed82e2f22a2417faa4f746f483b0fef0921f2f526"
)
NYU40_INSTANCE_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
P2_RIO_SEQUENCE_FILTER_COUNTS = {
    "train": {
        "sequence_count": 1178,
        "excluded_count": 4,
        "retained_count": 1174,
    },
    "validation": {
        "sequence_count": 157,
        "excluded_count": 3,
        "retained_count": 154,
    },
    "test": {
        "sequence_count": 157,
        "excluded_count": 3,
        "retained_count": 154,
    },
}
P2_RIO_SEQUENCE_FILTER_SHA256 = (
    "476c6e8819c6dce9ac9284d0e6dcf9ff65d117e1bff6647850dd950004162073"
)
P2_FORMAL_EPOCH_SAMPLE_MULTIPLE = 32
P2_FORMAL_DATASET_WEIGHTS = (1.0, 0.8)
P2_FORMAL_RAW_SAMPLER_NUM_SAMPLES = int(
    P2_RIO_SEQUENCE_FILTER_COUNTS["train"]["retained_count"]
    * (1 + sum(P2_FORMAL_DATASET_WEIGHTS[1:]) / P2_FORMAL_DATASET_WEIGHTS[0])
)
P2_FORMAL_SAMPLER_NUM_SAMPLES = (
    P2_FORMAL_RAW_SAMPLER_NUM_SAMPLES
    - P2_FORMAL_RAW_SAMPLER_NUM_SAMPLES % P2_FORMAL_EPOCH_SAMPLE_MULTIPLE
)
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


def _lexical_root_ref(path: Path, repo_root: Path, role: str) -> str:
    """Encode a configured path without resolving repository symlinks."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        relative = candidate.relative_to(repo_root)
    except ValueError:
        return f"external:{role}"
    return f"repo:{relative.as_posix()}"


def _resolved_root_ref(path: Path, repo_root: Path, role: str) -> str:
    """Encode resolved identity while keeping external paths private."""
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(repo_root.resolve())
    except (OSError, ValueError):
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()
        return f"external:{role}/{digest}"
    return f"repo:{relative.as_posix()}"


P2_DATA_ROOT_RELATIVE_PATHS = {
    "raw_scannet": Path("data/raw/scannet/scannet"),
    "scannet": Path("data/processed/scannet"),
    "rio": Path("data/processed/rio"),
    "split_metadata": Path("third_party/ScanNet/Tasks/Benchmark"),
    "test_segments": Path("data/raw/scannet_test_segments"),
}


def p2_data_root_reference_contract(
    *, repo_root: str | Path | None = None
) -> dict[str, dict[str, str]]:
    repository = Path(repo_root or REPO_ROOT).resolve()
    paths = {
        name: repository / relative
        for name, relative in P2_DATA_ROOT_RELATIVE_PATHS.items()
    }
    return {
        "expected": {
            name: _lexical_root_ref(path, repository, f"configured_{name}")
            for name, path in paths.items()
        },
        "expected_resolved": {
            name: _resolved_root_ref(path, repository, f"data_root/{name}")
            for name, path in paths.items()
        },
    }


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


def _git_nul_paths(repo_root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted({path for path in result.stdout.split("\0") if path})


def _git_index_flag_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-v", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    flagged = []
    for record in result.stdout.split("\0"):
        if not record:
            continue
        if len(record) < 3 or record[1] != " ":
            raise ValueError("invalid git ls-files output")
        tag = record[0]
        if tag == "S" or tag.islower():
            flagged.append(record[2:])
    return sorted(set(flagged))


def _git_blob_oid(content: bytes, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _stable_worktree_blob(path: Path, expected_mode: str) -> tuple[str, bytes]:
    before = path.lstat()
    if expected_mode == "120000":
        if not path.is_symlink():
            raise OSError("tracked symlink changed type")
        content = os.readlink(path).encode("utf-8", errors="surrogateescape")
        observed_mode = "120000"
    else:
        if expected_mode not in {"100644", "100755"}:
            raise OSError("unsupported tracked file mode")
        if path.is_symlink() or not path.is_file():
            raise OSError("tracked file changed type")
        content = path.read_bytes()
        observed_mode = "100755" if before.st_mode & 0o111 else "100644"
    after = path.lstat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_mode,
    )
    if identity_before != identity_after:
        raise OSError("tracked file changed while hashing")
    return observed_mode, content


def _git_tracked_tree_contract(
    repo_root: Path,
    commit: str,
    *,
    exclude_allowed_paths: bool = False,
) -> tuple[str, str]:
    tree = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", commit],
        capture_output=True,
        check=True,
    ).stdout
    algorithm = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--show-object-format"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if algorithm not in hashlib.algorithms_available:
        raise ValueError("unsupported git object format")

    expected_entries = []
    observed_entries = []
    for raw_record in tree.split(b"\0"):
        if not raw_record:
            continue
        header, raw_path = raw_record.split(b"\t", 1)
        mode, object_type, object_id = header.decode("ascii").split(" ")
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if object_type != "blob" or "\n" in path or "\r" in path:
            raise ValueError("unsupported tracked tree entry")
        if exclude_allowed_paths and _source_path_is_allowed(path):
            continue
        expected_entry = {
            "path": path,
            "mode": mode,
            "object_id": object_id,
        }
        expected_entries.append(expected_entry)
        try:
            observed_mode, content = _stable_worktree_blob(repo_root / path, mode)
            observed_object_id = _git_blob_oid(content, algorithm)
        except OSError:
            observed_mode = "missing"
            observed_object_id = None
        observed_entries.append(
            {
                "path": path,
                "mode": observed_mode,
                "object_id": observed_object_id,
            }
        )
    return _canonical_sha256(expected_entries), _canonical_sha256(observed_entries)


def _source_path_is_allowed(path: str) -> bool:
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in P2_ALLOWED_SOURCE_DIRTY_PREFIXES
    )


def build_p2_source_tree_contract(
    *,
    source_commit: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Bind formal evidence to a commit while allowing only P2 artifact changes."""
    repository = Path(repo_root or REPO_ROOT).resolve()
    errors: list[str] = []
    observed_head: str | None = None
    committed_paths: list[str] = []
    dirty_paths: list[str] = []
    index_flag_paths: list[str] = []
    expected_tracked_tree_sha256: str | None = None
    observed_tracked_tree_sha256: str | None = None
    try:
        head_result = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        observed_head = head_result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{40}", observed_head):
            errors.append("invalid_observed_head")
        pinned_commit = source_commit or observed_head
        if not isinstance(pinned_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", pinned_commit
        ):
            errors.append("invalid_source_commit")
        else:
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "merge-base",
                    "--is-ancestor",
                    pinned_commit,
                    observed_head or "HEAD",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if ancestor.returncode != 0:
                errors.append("source_commit_is_not_an_ancestor")
            else:
                committed_paths = _git_nul_paths(
                    repository,
                    "diff",
                    "--name-only",
                    "-z",
                    "--no-ext-diff",
                    f"{pinned_commit}..{observed_head}",
                )

        dirty_paths = sorted(
            set(
                _git_nul_paths(
                    repository,
                    "diff",
                    "--name-only",
                    "-z",
                    "--no-ext-diff",
                )
            )
            | set(
                _git_nul_paths(
                    repository,
                    "diff",
                    "--cached",
                    "--name-only",
                    "-z",
                    "--no-ext-diff",
                )
            )
            | set(
                _git_nul_paths(
                    repository,
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                )
            )
        )
        index_flag_paths = _git_index_flag_paths(repository)
        if isinstance(pinned_commit, str) and re.fullmatch(
            r"[0-9a-f]{40}", pinned_commit
        ):
            (
                expected_tracked_tree_sha256,
                observed_tracked_tree_sha256,
            ) = _git_tracked_tree_contract(
                repository,
                pinned_commit,
                exclude_allowed_paths=True,
            )
    except (OSError, ValueError, subprocess.CalledProcessError):
        pinned_commit = source_commit
        errors.append("git_state_unavailable")

    disallowed_committed = [
        path for path in committed_paths if not _source_path_is_allowed(path)
    ]
    disallowed_dirty = [
        path for path in dirty_paths if not _source_path_is_allowed(path)
    ]
    if disallowed_committed:
        errors.append("non_artifact_commits_since_source")
    if disallowed_dirty:
        errors.append("non_artifact_worktree_changes")
    if index_flag_paths:
        errors.append("index_flags_not_clean")
    if (
        expected_tracked_tree_sha256 is None
        or observed_tracked_tree_sha256 is None
    ):
        errors.append("tracked_tree_unavailable")
    elif expected_tracked_tree_sha256 != observed_tracked_tree_sha256:
        errors.append("tracked_tree_mismatch")
    return {
        "schema_version": P2_SOURCE_TREE_CONTRACT_SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "source_commit": pinned_commit,
        "observed_head": observed_head,
        "allowed_dirty_prefixes": list(P2_ALLOWED_SOURCE_DIRTY_PREFIXES),
        "committed_paths_since_source": committed_paths,
        "dirty_paths": dirty_paths,
        "index_flag_paths": index_flag_paths,
        "expected_tracked_tree_sha256": expected_tracked_tree_sha256,
        "observed_tracked_tree_sha256": observed_tracked_tree_sha256,
        "disallowed_committed_paths": disallowed_committed,
        "disallowed_dirty_paths": disallowed_dirty,
        "errors": errors,
    }


def build_p2_runtime_source_contract(
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Pin editable runtime dependencies to clean nested repositories."""
    repository = Path(repo_root or REPO_ROOT).resolve()
    records: dict[str, Any] = {}
    contract_errors: list[str] = []
    for name, definition in sorted(P2_RUNTIME_SOURCE_REPOSITORIES.items()):
        relative_root = str(definition["relative_root"])
        module_name = str(definition["module"])
        expected_commit = str(definition["expected_commit"])
        configured_root = repository / relative_root
        root = configured_root.resolve()
        errors: list[str] = []
        observed_commit: str | None = None
        dirty_paths: list[str] = []
        index_flag_paths: list[str] = []
        expected_tracked_tree_sha256: str | None = None
        observed_tracked_tree_sha256: str | None = None
        module_origin_ref: str | None = None
        native_extension_records: dict[str, Any] = {}

        if (
            configured_root.is_symlink()
            or not root.is_dir()
            or not root.is_relative_to(repository)
        ):
            errors.append("repository_root_invalid")
        else:
            try:
                top_level = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                observed_commit = subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                if Path(top_level).resolve() != root:
                    errors.append("repository_root_mismatch")
                dirty_paths = sorted(
                    set(
                        _git_nul_paths(
                            root,
                            "diff",
                            "--name-only",
                            "-z",
                            "--no-ext-diff",
                        )
                    )
                    | set(
                        _git_nul_paths(
                            root,
                            "diff",
                            "--cached",
                            "--name-only",
                            "-z",
                            "--no-ext-diff",
                        )
                    )
                    | set(
                        _git_nul_paths(
                            root,
                            "ls-files",
                            "--others",
                            "--exclude-standard",
                            "-z",
                        )
                    )
                )
                index_flag_paths = _git_index_flag_paths(root)
                (
                    expected_tracked_tree_sha256,
                    observed_tracked_tree_sha256,
                ) = _git_tracked_tree_contract(root, expected_commit)
            except (OSError, ValueError, subprocess.CalledProcessError):
                errors.append("git_state_unavailable")

        if observed_commit != expected_commit:
            errors.append("commit_mismatch")
        if dirty_paths:
            errors.append("worktree_not_clean")
        if index_flag_paths:
            errors.append("index_flags_not_clean")
        if (
            expected_tracked_tree_sha256 is None
            or observed_tracked_tree_sha256 is None
        ):
            errors.append("tracked_tree_unavailable")
        elif expected_tracked_tree_sha256 != observed_tracked_tree_sha256:
            errors.append("tracked_tree_mismatch")

        try:
            spec = importlib.util.find_spec(module_name)
            origin_value = None if spec is None else spec.origin
            if not isinstance(origin_value, str):
                raise TypeError
            origin = Path(origin_value).resolve(strict=True)
            expected_origin = (
                root / Path(*module_name.split(".")) / "__init__.py"
            ).resolve(strict=True)
            if not origin.is_file() or origin != expected_origin:
                raise ValueError
            module_origin_ref = _portable_root_ref(
                origin,
                repository,
                f"{name}_module",
            )
        except (ImportError, ModuleNotFoundError, OSError, TypeError, ValueError):
            errors.append("module_origin_mismatch")

        native_extensions = definition.get("native_extensions", {})
        if not isinstance(native_extensions, Mapping):
            errors.append("native_extension_contract_invalid")
            native_extensions = {}
        for extension_module, extension_contract in sorted(
            native_extensions.items()
        ):
            observed_byte_size = None
            observed_sha256 = None
            origin_ref = None
            extension_status = "fail"
            expected_extension = (
                extension_contract
                if isinstance(extension_contract, Mapping)
                else {}
            )
            try:
                if not isinstance(extension_contract, Mapping):
                    raise TypeError
                relative_extension_path = extension_contract.get("relative_path")
                if not isinstance(relative_extension_path, str):
                    raise TypeError
                expected_extension_origin = (
                    root / relative_extension_path
                ).resolve(strict=True)
                extension_spec = importlib.util.find_spec(extension_module)
                extension_origin_value = (
                    None if extension_spec is None else extension_spec.origin
                )
                if not isinstance(extension_origin_value, str):
                    raise TypeError
                extension_origin = Path(extension_origin_value).resolve(strict=True)
                if (
                    not extension_origin.is_file()
                    or extension_origin != expected_extension_origin
                ):
                    raise ValueError
                observed_byte_size, observed_sha256 = _sha256_file_stable(
                    extension_origin
                )
                origin_ref = _portable_root_ref(
                    extension_origin,
                    repository,
                    f"{name}_native_extension",
                )
                if (
                    observed_byte_size == expected_extension.get("expected_byte_size")
                    and observed_sha256 == expected_extension.get("expected_sha256")
                ):
                    extension_status = "pass"
                else:
                    errors.append("native_extension_mismatch")
            except (
                ImportError,
                ModuleNotFoundError,
                OSError,
                TypeError,
                ValueError,
            ):
                errors.append("native_extension_mismatch")
            native_extension_records[extension_module] = {
                "origin_ref": origin_ref,
                "expected_byte_size": expected_extension.get("expected_byte_size"),
                "observed_byte_size": observed_byte_size,
                "expected_sha256": expected_extension.get("expected_sha256"),
                "observed_sha256": observed_sha256,
                "status": extension_status,
            }

        records[name] = {
            "reference": _portable_root_ref(root, repository, f"{name}_source"),
            "module": module_name,
            "expected_commit": expected_commit,
            "observed_commit": observed_commit,
            "module_origin_ref": module_origin_ref,
            "dirty_paths": dirty_paths,
            "index_flag_paths": index_flag_paths,
            "expected_tracked_tree_sha256": expected_tracked_tree_sha256,
            "observed_tracked_tree_sha256": observed_tracked_tree_sha256,
            "native_extensions": native_extension_records,
            "status": "pass" if not errors else "fail",
            "errors": errors,
        }
        contract_errors.extend(f"{name}:{error}" for error in errors)

    return {
        "schema_version": P2_RUNTIME_SOURCE_CONTRACT_SCHEMA_VERSION,
        "status": "pass" if not contract_errors else "fail",
        "repositories": records,
        "errors": contract_errors,
    }


def _environment_ref(path: Path, prefix: Path) -> str:
    resolved = path.resolve(strict=True)
    return f"env:{resolved.relative_to(prefix).as_posix()}"


def _native_binary_manifest(paths: list[Path], prefix: Path) -> dict[str, Any]:
    candidates: set[Path] = set()
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(prefix):
            raise ValueError("runtime binary escapes environment prefix")
        if resolved.is_file():
            candidates.add(resolved)
            continue
        if not resolved.is_dir():
            raise ValueError("runtime binary root is invalid")
        for candidate in resolved.rglob("*"):
            if candidate.is_file() and ".so" in candidate.name:
                candidates.add(candidate.resolve(strict=True))
    if not candidates:
        raise ValueError("runtime component has no native files")

    entries = []
    for candidate in sorted(candidates, key=lambda item: item.as_posix()):
        if not candidate.is_relative_to(prefix):
            raise ValueError("runtime binary escapes environment prefix")
        byte_size, sha256 = _sha256_file_stable(candidate)
        entries.append(
            {
                "path": candidate.relative_to(prefix).as_posix(),
                "byte_size": byte_size,
                "sha256": sha256,
            }
        )
    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["byte_size"] for entry in entries),
        "content_sha256": _canonical_sha256(entries),
    }


def _python_source_manifest(root: Path, prefix: Path) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir() or not resolved_root.is_relative_to(prefix):
        raise ValueError("runtime Python source root is invalid")
    entries = []
    for candidate in sorted(resolved_root.rglob("*.py")):
        if candidate.is_symlink():
            raise ValueError("runtime Python source cannot be a symbolic link")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("runtime Python source escapes package root")
        byte_size, sha256 = _sha256_file_stable(resolved)
        entries.append(
            {
                "path": resolved.relative_to(prefix).as_posix(),
                "byte_size": byte_size,
                "sha256": sha256,
            }
        )
    if not entries:
        raise ValueError("runtime Python package has no source files")
    return {
        "file_count": len(entries),
        "total_bytes": sum(entry["byte_size"] for entry in entries),
        "content_sha256": _canonical_sha256(entries),
    }


def _runtime_module_origin(
    module_name: str,
    expected_path: Path,
    prefix: Path,
) -> str:
    spec = importlib.util.find_spec(module_name)
    origin_value = None if spec is None else spec.origin
    if not isinstance(origin_value, str):
        raise TypeError(module_name)
    origin = Path(origin_value).resolve(strict=True)
    if origin != expected_path.resolve(strict=True):
        raise ValueError(f"unexpected module origin for {module_name}")
    return _environment_ref(origin, prefix)


def _format_runtime_cudnn_version(value: int | None) -> str:
    if not isinstance(value, int):
        return "unknown"
    return f"{value // 10000}.{value % 10000 // 100}.{value % 100}"


def _format_runtime_nccl_version(value: Any) -> str:
    if not isinstance(value, tuple) or not value or any(
        not isinstance(part, int) for part in value
    ):
        return "unknown"
    return ".".join(str(part) for part in value)


def _runtime_environment_versions() -> dict[str, str]:
    import torch

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": _format_runtime_cudnn_version(torch.backends.cudnn.version()),
        "nccl": _format_runtime_nccl_version(torch.cuda.nccl.version()),
        "pytorch_lightning": importlib.metadata.version("pytorch-lightning"),
        "hydra_core": importlib.metadata.version("hydra-core"),
        "omegaconf": importlib.metadata.version("omegaconf"),
        "spconv": importlib.metadata.version("spconv-cu126"),
        "cumm": importlib.metadata.version("cumm-cu126"),
        "flash_attn": importlib.metadata.version("flash-attn"),
        "torch_scatter": importlib.metadata.version("torch-scatter"),
        "pointnet2": importlib.metadata.version("pointnet2"),
        "cuda_runtime_package": importlib.metadata.version(
            "nvidia-cuda-runtime-cu12"
        ),
        "cudnn_package": (
            "v" + importlib.metadata.version("nvidia-cudnn-cu12")
        ),
        "nccl_package": importlib.metadata.version("nvidia-nccl-cu12"),
    }


def build_p2_runtime_environment_contract() -> dict[str, Any]:
    """Bind formal P2 authorization to the active CUDA/PyTorch runtime."""
    prefix = Path(sys.prefix).resolve()
    site_packages = prefix / "lib" / "python3.10" / "site-packages"
    errors: list[str] = []
    try:
        versions = _runtime_environment_versions()
    except (
        AttributeError,
        ImportError,
        importlib.metadata.PackageNotFoundError,
        OSError,
        RuntimeError,
    ):
        versions = {}
        errors.append("runtime_versions_unavailable")
    if versions != P2_RUNTIME_ENVIRONMENT_VERSIONS:
        errors.append("runtime_versions_mismatch")

    component_definitions = {
        "python": {
            "modules": {},
            "origins": [Path(sys.executable).resolve()],
            "native_paths": [Path(sys.executable).resolve()],
        },
        "torch": {
            "modules": {
                "torch": site_packages / "torch" / "__init__.py",
            },
            "origins": [],
            "native_paths": [site_packages / "torch"],
        },
        "spconv": {
            "modules": {
                "spconv": site_packages / "spconv" / "__init__.py",
            },
            "origins": [],
            "native_paths": [
                site_packages / "spconv",
                site_packages / "spconv_cu126.libs",
            ],
        },
        "cumm": {
            "modules": {
                "cumm": site_packages / "cumm" / "__init__.py",
            },
            "origins": [],
            "native_paths": [
                site_packages / "cumm",
                site_packages / "cumm_cu126.libs",
            ],
        },
        "flash_attn": {
            "modules": {
                "flash_attn": site_packages / "flash_attn" / "__init__.py",
                "flash_attn_2_cuda": (
                    site_packages
                    / "flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
                ),
            },
            "origins": [],
            "native_paths": [
                site_packages
                / "flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so"
            ],
        },
        "torch_scatter": {
            "modules": {
                "torch_scatter": (
                    site_packages / "torch_scatter" / "__init__.py"
                ),
            },
            "origins": [],
            "native_paths": [site_packages / "torch_scatter"],
        },
        "pointnet2": {
            "modules": {
                "pointnet2._ext": (
                    site_packages
                    / "pointnet2"
                    / "_ext.cpython-310-x86_64-linux-gnu.so"
                ),
            },
            "origins": [],
            "native_paths": [
                site_packages
                / "pointnet2"
                / "_ext.cpython-310-x86_64-linux-gnu.so"
            ],
        },
        "nvidia_cuda_libraries": {
            "modules": {},
            "origins": [site_packages / "nvidia"],
            "native_paths": [site_packages / "nvidia"],
        },
        "pytorch_lightning": {
            "modules": {
                "pytorch_lightning": (
                    site_packages / "pytorch_lightning" / "__init__.py"
                ),
            },
            "origins": [],
            "python_source_root": site_packages / "pytorch_lightning",
        },
        "hydra": {
            "modules": {
                "hydra": site_packages / "hydra" / "__init__.py",
            },
            "origins": [],
            "python_source_root": site_packages / "hydra",
        },
        "omegaconf": {
            "modules": {
                "omegaconf": site_packages / "omegaconf" / "__init__.py",
            },
            "origins": [],
            "python_source_root": site_packages / "omegaconf",
        },
    }
    components = {}
    for name, definition in component_definitions.items():
        component_errors = []
        origin_refs = []
        try:
            origin_refs.extend(
                _runtime_module_origin(module, path, prefix)
                for module, path in definition["modules"].items()
            )
            origin_refs.extend(
                _environment_ref(path, prefix) for path in definition["origins"]
            )
            native_paths = definition.get("native_paths")
            native_manifest = (
                _native_binary_manifest(list(native_paths), prefix)
                if native_paths is not None
                else None
            )
            source_root = definition.get("python_source_root")
            python_source_manifest = (
                _python_source_manifest(Path(source_root), prefix)
                if source_root is not None
                else None
            )
            if native_manifest is None and python_source_manifest is None:
                raise ValueError("runtime component has no content binding")
        except (ImportError, OSError, TypeError, ValueError):
            native_manifest = {
                "file_count": 0,
                "total_bytes": 0,
                "content_sha256": None,
            }
            python_source_manifest = None
            component_errors.append("runtime_component_unavailable")
        component = {
            "status": "pass" if not component_errors else "fail",
            "origin_refs": sorted(origin_refs),
            "errors": component_errors,
        }
        if native_manifest is not None:
            component["native_manifest"] = native_manifest
        if python_source_manifest is not None:
            component["python_source_manifest"] = python_source_manifest
        components[name] = component
        errors.extend(f"{name}:{error}" for error in component_errors)

    try:
        pointops_spec = importlib.util.find_spec("pointops")
    except (ImportError, ModuleNotFoundError, ValueError):
        pointops_spec = None
    optional_modules = {
        "pointops": {
            "required": False,
            "status": "absent" if pointops_spec is None else "present_not_required",
        }
    }
    return {
        "schema_version": P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "versions": versions,
        "components": components,
        "optional_modules": optional_modules,
        "errors": errors,
    }


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
            name: _resolved_root_ref(path, repository, f"data_root/{name}")
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
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise OSError("split changed while reading")
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            scenes = [
                line.strip()
                for line in payload.decode("utf-8").splitlines()
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
        "data.batch_size": 2,
        "data.train_dataloader.batch_size": 2,
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
        "data.validation_dataset._target_": (
            "datasets.semseg.SemanticSegmentationDataset"
        ),
        "data.validation_dataset.dataset_name": "rio",
        "data.validation_dataset.data_dir": "data/processed/rio",
        "data.validation_dataset.temporal_window": 2,
        "data.validation_dataset.filter_out_classes": [0, 1, 255],
        "data.validation_dataset.exclude_unsupervised_sequences": True,
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
        "data.test_dataset.exclude_unsupervised_sequences": True,
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
        "trainer.accumulate_grad_batches": 8,
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
            "save_on_train_epoch_end": True,
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
    expected_roots = p2_data_root_reference_contract()["expected_resolved"]
    expected_roots = {
        name: expected_roots[name] for name in ("scannet", "rio")
    }
    if manifest.get("roots") != expected_roots:
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


def _validate_tracked_tree_binding(
    record: Mapping[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    if record.get("index_flag_paths") != []:
        errors.append(f"{prefix}.index_flag_paths mismatch")
    expected_sha256 = record.get("expected_tracked_tree_sha256")
    observed_sha256 = record.get("observed_tracked_tree_sha256")
    if (
        not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        or observed_sha256 != expected_sha256
    ):
        errors.append(f"{prefix}.tracked_tree_sha256 mismatch")


def _validate_source_tree_contract(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> str | None:
    source_commit = artifact.get("local_source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", source_commit
    ):
        errors.append("local_source_commit invalid")
        source_commit = None

    contract = artifact.get("source_tree_contract")
    if not isinstance(contract, Mapping):
        errors.append("source_tree_contract is missing")
        return source_commit
    expected = {
        "schema_version": P2_SOURCE_TREE_CONTRACT_SCHEMA_VERSION,
        "status": "pass",
        "source_commit": source_commit,
        "observed_head": source_commit,
        "allowed_dirty_prefixes": list(P2_ALLOWED_SOURCE_DIRTY_PREFIXES),
        "committed_paths_since_source": [],
        "disallowed_committed_paths": [],
        "disallowed_dirty_paths": [],
        "errors": [],
    }
    for field, expected_value in expected.items():
        if contract.get(field) != expected_value:
            errors.append(f"source_tree_contract.{field} mismatch")
    dirty_paths = contract.get("dirty_paths")
    if not isinstance(dirty_paths, list) or any(
        not isinstance(path, str) or not _source_path_is_allowed(path)
        for path in dirty_paths
    ):
        errors.append("source_tree_contract.dirty_paths invalid")
    _validate_tracked_tree_binding(contract, "source_tree_contract", errors)
    return source_commit


def _validate_runtime_source_contract(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> Mapping[str, Any] | None:
    contract = artifact.get("runtime_source_contract")
    if not isinstance(contract, Mapping):
        errors.append("runtime_source_contract is missing")
        return None
    if contract.get("schema_version") != P2_RUNTIME_SOURCE_CONTRACT_SCHEMA_VERSION:
        errors.append("runtime_source_contract.schema_version mismatch")
    if contract.get("status") != "pass":
        errors.append("runtime_source_contract.status is not pass")
    if contract.get("errors") != []:
        errors.append("runtime_source_contract.errors is not empty")

    repositories = contract.get("repositories")
    if not isinstance(repositories, Mapping) or set(repositories) != set(
        P2_RUNTIME_SOURCE_REPOSITORIES
    ):
        errors.append("runtime_source_contract.repositories mismatch")
        return contract
    for name, definition in P2_RUNTIME_SOURCE_REPOSITORIES.items():
        record = repositories.get(name)
        if not isinstance(record, Mapping):
            errors.append(f"runtime_source_contract.{name} is missing")
            continue
        relative_root = str(definition["relative_root"])
        expected = {
            "reference": f"repo:{relative_root}",
            "module": definition["module"],
            "expected_commit": definition["expected_commit"],
            "observed_commit": definition["expected_commit"],
            "dirty_paths": [],
            "status": "pass",
            "errors": [],
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                errors.append(
                    f"runtime_source_contract.{name}.{field} mismatch"
                )
        _validate_tracked_tree_binding(
            record,
            f"runtime_source_contract.{name}",
            errors,
        )
        origin_ref = record.get("module_origin_ref")
        expected_origin_ref = (
            f"repo:{relative_root}/{str(definition['module']).replace('.', '/')}/"
            "__init__.py"
        )
        if origin_ref != expected_origin_ref:
            errors.append(
                f"runtime_source_contract.{name}.module_origin_ref mismatch"
            )
        native_records = record.get("native_extensions")
        native_contracts = definition.get("native_extensions", {})
        if not isinstance(native_records, Mapping) or set(native_records) != set(
            native_contracts
        ):
            errors.append(
                f"runtime_source_contract.{name}.native_extensions mismatch"
            )
            continue
        for extension_module, extension_contract in native_contracts.items():
            native_record = native_records.get(extension_module)
            if not isinstance(native_record, Mapping):
                errors.append(
                    "runtime_source_contract."
                    f"{name}.{extension_module} is missing"
                )
                continue
            expected_native = {
                "expected_byte_size": extension_contract["expected_byte_size"],
                "observed_byte_size": extension_contract["expected_byte_size"],
                "expected_sha256": extension_contract["expected_sha256"],
                "observed_sha256": extension_contract["expected_sha256"],
                "status": "pass",
            }
            for field, expected_value in expected_native.items():
                if native_record.get(field) != expected_value:
                    errors.append(
                        "runtime_source_contract."
                        f"{name}.{extension_module}.{field} mismatch"
                    )
            native_origin_ref = native_record.get("origin_ref")
            expected_native_origin_ref = (
                f"repo:{relative_root}/{extension_contract['relative_path']}"
            )
            if native_origin_ref != expected_native_origin_ref:
                errors.append(
                    "runtime_source_contract."
                    f"{name}.{extension_module}.origin_ref mismatch"
                )
    return contract


def _validate_runtime_environment_contract(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> Mapping[str, Any] | None:
    contract = artifact.get("runtime_environment_contract")
    prefix = "runtime_environment_contract"
    if not isinstance(contract, Mapping):
        errors.append(f"{prefix} is missing")
        return None
    if contract.get("schema_version") != (
        P2_RUNTIME_ENVIRONMENT_CONTRACT_SCHEMA_VERSION
    ):
        errors.append(f"{prefix}.schema_version mismatch")
    if contract.get("status") != "pass":
        errors.append(f"{prefix}.status is not pass")
    if contract.get("errors") != []:
        errors.append(f"{prefix}.errors is not empty")
    if contract.get("versions") != P2_RUNTIME_ENVIRONMENT_VERSIONS:
        errors.append(f"{prefix}.versions mismatch")

    expected_components = {
        "pytorch_lightning",
        "hydra",
        "omegaconf",
        "python",
        "torch",
        "spconv",
        "cumm",
        "flash_attn",
        "torch_scatter",
        "pointnet2",
        "nvidia_cuda_libraries",
    }
    components = contract.get("components")
    if not isinstance(components, Mapping) or set(components) != expected_components:
        errors.append(f"{prefix}.components mismatch")
        return contract
    python_source_components = {"pytorch_lightning", "hydra", "omegaconf"}
    for name, record in components.items():
        component_prefix = f"{prefix}.{name}"
        if not isinstance(record, Mapping):
            errors.append(f"{component_prefix} is missing")
            continue
        if record.get("status") != "pass" or record.get("errors") != []:
            errors.append(f"{component_prefix}.status mismatch")
        origin_refs = record.get("origin_refs")
        if (
            not isinstance(origin_refs, list)
            or not origin_refs
            or any(
                not isinstance(origin, str) or not origin.startswith("env:")
                for origin in origin_refs
            )
        ):
            errors.append(f"{component_prefix}.origin_refs invalid")
        manifest_field = (
            "python_source_manifest"
            if name in python_source_components
            else "native_manifest"
        )
        manifest = record.get(manifest_field)
        if not isinstance(manifest, Mapping):
            errors.append(f"{component_prefix}.{manifest_field} invalid")
            continue
        if (
            type(manifest.get("file_count")) is not int
            or manifest["file_count"] < 1
            or type(manifest.get("total_bytes")) is not int
            or manifest["total_bytes"] < 1
            or not isinstance(manifest.get("content_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest["content_sha256"])
        ):
            errors.append(f"{component_prefix}.{manifest_field} invalid")

    optional_modules = contract.get("optional_modules")
    pointops = (
        optional_modules.get("pointops")
        if isinstance(optional_modules, Mapping)
        else None
    )
    if (
        not isinstance(pointops, Mapping)
        or pointops.get("required") is not False
        or pointops.get("status") not in {"absent", "present_not_required"}
    ):
        errors.append(f"{prefix}.optional_modules invalid")
    return contract


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
        "sampler_num_samples": P2_FORMAL_SAMPLER_NUM_SAMPLES,
        "epoch_sample_multiple": P2_FORMAL_EPOCH_SAMPLE_MULTIPLE,
    }
    for field, expected in expected_mix.items():
        if mix.get(field) != expected:
            errors.append(f"mix_instantiation.{field} mismatch")
    sizes = mix.get("dataset_sizes")
    if (
        not isinstance(sizes, list)
        or len(sizes) != 2
        or sizes
        != [
            P2_RIO_SEQUENCE_FILTER_COUNTS["train"]["retained_count"],
            P2_SCANNET_SEQUENCE_FILTER_COUNTS["train"]["retained_count"],
        ]
    ):
        errors.append("mix_instantiation.dataset_sizes mismatch")


def _validate_unsupervised_sequence_filter(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> None:
    evidence = artifact.get("unsupervised_sequence_filter")
    if not isinstance(evidence, Mapping):
        errors.append("unsupervised_sequence_filter is missing")
        return
    expected_header = {
        "schema_version": 1,
        "status": "pass",
        "enabled": True,
        "source": "real_npy",
        "taxonomy_label_ids": NYU40_INSTANCE_IDS,
    }
    for field, expected in expected_header.items():
        if evidence.get(field) != expected:
            errors.append(f"unsupervised_sequence_filter.{field} mismatch")

    by_split = evidence.get("by_split")
    if not isinstance(by_split, Mapping):
        errors.append("unsupervised_sequence_filter.by_split is missing")
        return
    if set(by_split) != set(P2_RIO_SEQUENCE_FILTER_COUNTS):
        errors.append("unsupervised_sequence_filter.by_split keys mismatch")
        return

    names_by_split: dict[str, list[str]] = {}
    for split, expected_counts in P2_RIO_SEQUENCE_FILTER_COUNTS.items():
        record = by_split.get(split)
        if not isinstance(record, Mapping):
            errors.append(f"unsupervised_sequence_filter.by_split.{split} is missing")
            continue
        for field, expected in expected_counts.items():
            if record.get(field) != expected:
                errors.append(
                    f"unsupervised_sequence_filter.by_split.{split}.{field} mismatch"
                )
        names = record.get("excluded_sequences")
        if not isinstance(names, list) or any(type(name) is not str for name in names):
            errors.append(
                f"unsupervised_sequence_filter.by_split.{split}.excluded_sequences invalid"
            )
            continue
        if names != sorted(set(names)):
            errors.append(
                f"unsupervised_sequence_filter.by_split.{split}.excluded_sequences ordering mismatch"
            )
        if len(names) != expected_counts["excluded_count"]:
            errors.append(
                f"unsupervised_sequence_filter.by_split.{split}.excluded_sequences count mismatch"
            )
        names_by_split[split] = names

    if set(names_by_split) == set(P2_RIO_SEQUENCE_FILTER_COUNTS):
        observed_digest = _canonical_sha256(names_by_split)
        if observed_digest != P2_RIO_SEQUENCE_FILTER_SHA256:
            errors.append("unsupervised_sequence_filter.sequence_name_sha256 mismatch")
    if evidence.get("sequence_name_sha256") != P2_RIO_SEQUENCE_FILTER_SHA256:
        errors.append("unsupervised_sequence_filter.sequence_name_sha256 mismatch")

    path_integrity = artifact.get("rio_path_integrity")
    if isinstance(path_integrity, Mapping):
        if path_integrity.get("unsupervised_sequences") != []:
            errors.append("rio_path_integrity.unsupervised_sequences mismatch")
        if path_integrity.get("excluded_unsupervised_sequences") != names_by_split:
            errors.append(
                "rio_path_integrity.excluded_unsupervised_sequences mismatch"
            )
        if path_integrity.get("filtered_sequence_counts") != {
            "train": P2_RIO_SEQUENCE_FILTER_COUNTS["train"]["retained_count"],
            "validation": P2_RIO_SEQUENCE_FILTER_COUNTS["validation"]["retained_count"],
        }:
            errors.append("rio_path_integrity.filtered_sequence_counts mismatch")


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
    root_contract = p2_data_root_reference_contract()
    expected = {
        "status": "pass",
        "expected": root_contract["expected"],
        "observed": root_contract["expected"],
        "expected_resolved": root_contract["expected_resolved"],
        "observed_resolved": root_contract["expected_resolved"],
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
    source_commit = _validate_source_tree_contract(artifact, errors)
    runtime_source_contract = _validate_runtime_source_contract(artifact, errors)
    runtime_environment_contract = _validate_runtime_environment_contract(
        artifact,
        errors,
    )
    _validate_asset_summaries(artifact, errors)
    _validate_taxonomy_and_mix(artifact, errors)
    _validate_unsupervised_sequence_filter(artifact, errors)
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
    first_source_contract: Mapping[str, Any] | None = None
    first_runtime_source_contract: Mapping[str, Any] | None = None
    first_runtime_environment_contract: Mapping[str, Any] | None = None
    first_split_identity: Mapping[str, Any] | None = None
    first_input_manifest: Mapping[str, Any] | None = None
    if not errors:
        current_inputs_revalidated = True
        first_source_contract = build_p2_source_tree_contract(
            source_commit=source_commit
        )
        if first_source_contract.get("status") != "pass":
            errors.append("current source_tree_contract is not pass")
        first_runtime_source_contract = build_p2_runtime_source_contract()
        if first_runtime_source_contract != runtime_source_contract:
            errors.append("current runtime_source_contract mismatch")
        first_runtime_environment_contract = (
            build_p2_runtime_environment_contract()
        )
        if first_runtime_environment_contract != runtime_environment_contract:
            errors.append("current runtime_environment_contract mismatch")
        first_split_identity = build_scannet_official_split_identity()
        if first_split_identity != artifact.get("official_split_identity"):
            errors.append("current official_split_identity mismatch")
        first_input_manifest = build_p2_input_manifest()
        if first_input_manifest != artifact.get("input_manifest"):
            errors.append("current input_manifest mismatch")
    if current_inputs_revalidated and issued_at is not None:
        final_source_contract = build_p2_source_tree_contract(
            source_commit=source_commit
        )
        final_runtime_source_contract = build_p2_runtime_source_contract()
        final_runtime_environment_contract = (
            build_p2_runtime_environment_contract()
        )
        final_split_identity = build_scannet_official_split_identity()
        final_input_manifest = build_p2_input_manifest()
        if final_source_contract.get("status") != "pass":
            errors.append("final source_tree_contract is not pass")
        elif (
            first_source_contract is not None
            and final_source_contract.get("observed_head")
            != first_source_contract.get("observed_head")
        ):
            errors.append("source HEAD changed during authorization")
        if final_runtime_source_contract != first_runtime_source_contract:
            errors.append("runtime source changed during authorization")
        if (
            final_runtime_environment_contract
            != first_runtime_environment_contract
        ):
            errors.append("runtime environment changed during authorization")
        if final_split_identity != first_split_identity:
            errors.append("official split changed during authorization")
        if final_input_manifest != first_input_manifest:
            errors.append("training inputs changed during authorization")
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
