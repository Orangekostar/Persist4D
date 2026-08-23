"""Frozen protocol bindings for the Persist4D reviewer-closure experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.system_comparison_protocol import (
    ProtocolBindingError,
    sha256_file,
    validate_system_comparison_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ORDERS = ("canonical", "reverse", "sha256_seed45")
_HORIZONS = (2, 3, 4, 5)
_SOURCE_NAMES = {
    "checkpoint",
    "protocol_b_manifest",
    "system_config",
    "system_manifest",
    "reproducibility_binding",
    "system_report",
    "system_artifact_manifest",
}
_PHASE_I_METHODS = [
    {
        "id": "full_history_native",
        "name": "ReScene4D Full-History (Frozen T2 Checkpoint)",
    },
    {"id": "pairwise_feature", "name": "Pairwise Feature Association"},
    {
        "id": "pairwise_feature_class",
        "name": "Pairwise Feature-Class Association",
    },
    {"id": "ema_temporal", "name": "EMA Temporal Association"},
    {
        "id": "persist4d",
        "name": "Persist4D Persistent Entity State",
    },
    {
        "id": "full_history_persistent_state_diagnostic",
        "name": "Full-History + Persistent-State Diagnostic",
    },
]


class ReviewerClosureProtocolError(ValueError):
    """Raised when reviewer-closure inputs differ from their frozen binding."""


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReviewerClosureProtocolError(f"{name} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ReviewerClosureProtocolError(
            f"{name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ReviewerClosureProtocolError(f"{name} must be a 40-digit commit")
    return value


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReviewerClosureProtocolError(f"{name} must be a regular file")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name=name)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewerClosureProtocolError(f"cannot decode {name}") from error


def _resolve_reference(reference: str, *, repo_root: Path) -> Path:
    if not reference.startswith("repo:"):
        raise ReviewerClosureProtocolError("source references must use repo: paths")
    relative = Path(reference.removeprefix("repo:"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReviewerClosureProtocolError("source reference escapes repository")
    path = repo_root / relative
    if path.is_symlink() or not path.is_file():
        raise ReviewerClosureProtocolError(
            f"bound source is not a regular file: {reference}"
        )
    return path.resolve()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReviewerClosureProtocolError(
            "cannot verify reviewer-closure git binding"
        ) from error
    return completed.stdout.strip()


def load_reviewer_closure_config(path: str | Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ReviewerClosureProtocolError("cannot load reviewer-closure config") from error
    config = _mapping(payload, name="reviewer-closure config")
    _exact_keys(
        config,
        {
            "schema_version",
            "stage_name",
            "source_commits",
            "baseline",
            "sources",
            "sidecar",
            "statistics",
            "gate_i",
        },
        name="reviewer-closure config",
    )
    if config["schema_version"] != 1 or config["stage_name"] != "persist4d-reviewer-closure":
        raise ReviewerClosureProtocolError("reviewer-closure schema or stage differs")

    commits = _mapping(config["source_commits"], name="source_commits")
    _exact_keys(
        commits,
        {"completed_branch_head", "system_report_source", "official_rescene"},
        name="source_commits",
    )
    config["source_commits"] = {
        name: _commit(value, name=f"source_commits.{name}")
        for name, value in commits.items()
    }
    if config["source_commits"] != {
        "completed_branch_head": "b2414e3b2e89a990ee42a368caf6784eb27f8f01",
        "system_report_source": "575acc12fbd63f38fc3c16578914b25c2fed8584",
        "official_rescene": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
    }:
        raise ReviewerClosureProtocolError("source_commits differ from frozen binding")
    baseline = _mapping(config["baseline"], name="baseline")
    if baseline != {
        "artifact_tree": "398fe87e1d40d67e61399fd893f02dc5f5f6b7ad",
        "classification": "SYSTEM_PARETO_LOCK",
    }:
        raise ReviewerClosureProtocolError("immutable system-comparison baseline differs")

    sources = _mapping(config["sources"], name="sources")
    _exact_keys(sources, _SOURCE_NAMES, name="sources")
    normalized_sources: dict[str, dict[str, str]] = {}
    for name in sorted(_SOURCE_NAMES):
        source = _mapping(sources[name], name=f"sources.{name}")
        _exact_keys(source, {"reference", "sha256"}, name=f"sources.{name}")
        reference = source["reference"]
        digest = source["sha256"]
        if not isinstance(reference, str) or not reference.startswith("repo:"):
            raise ReviewerClosureProtocolError(f"sources.{name}.reference is invalid")
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            raise ReviewerClosureProtocolError(f"sources.{name}.sha256 is invalid")
        normalized_sources[name] = {"reference": reference, "sha256": digest}
    config["sources"] = normalized_sources

    sidecar = _mapping(config["sidecar"], name="sidecar")
    _exact_keys(
        sidecar,
        {
            "schema_version",
            "horizons",
            "tensor_fields",
            "identity_fields",
            "provenance_fields",
        },
        name="sidecar",
    )
    if (
        sidecar["schema_version"] != "full-history-observations-v2"
        or sidecar["horizons"] != list(_HORIZONS)
        or sidecar["tensor_fields"]
        != [
            "features",
            "class_prob",
            "confidence",
            "valid",
            "current_stage_masks",
            "mask_support",
            "local_query_ids",
        ]
        or sidecar["identity_fields"]
        != [
            "reference_scene_id",
            "master_sequence_id",
            "order_id",
            "horizon",
            "scan_indices",
        ]
        or sidecar["provenance_fields"]
        != [
            "source_prediction_content_sha256",
            "checkpoint_sha256",
            "config_sha256",
            "source_commit",
        ]
    ):
        raise ReviewerClosureProtocolError("sidecar contract differs")

    statistics = _mapping(config["statistics"], name="statistics")
    if statistics != {
        "cluster_unit": "reference_scene_id",
        "cluster_count": 6,
        "bootstrap_replicates": 10000,
        "seed": 45,
        "confidence_level": 0.95,
    }:
        raise ReviewerClosureProtocolError("statistics contract differs")
    gate = _mapping(config["gate_i"], name="gate_i")
    if gate != {
        "compared_horizons": [4, 5],
        "identity_metrics": [
            "normalized_deployment_id_switch_rate",
            "gap_identity_recovery",
        ],
        "decision_rule": "any_metric_any_horizon_persist4d_advantage_with_paired_ci",
        "requires_ci_exclude_zero": True,
    }:
        raise ReviewerClosureProtocolError("Gate I contract differs")
    return config


def validate_reviewer_closure_binding(
    config_path: str | Path,
    *,
    repo_root: str | Path = PROJECT_ROOT,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(repo_root).resolve()
    config_source = Path(config_path)
    config = load_reviewer_closure_config(config_source)
    resolved: dict[str, Path] = {}
    for name, source in config["sources"].items():
        path = _resolve_reference(source["reference"], repo_root=repository)
        if sha256_file(path) != source["sha256"]:
            raise ReviewerClosureProtocolError(
                f"{name} SHA256 differs from frozen binding"
            )
        resolved[name] = path
    if checkpoint_path is not None:
        supplied = Path(checkpoint_path)
        if supplied.is_symlink() or not supplied.is_file():
            raise ReviewerClosureProtocolError("checkpoint_path must be a regular file")
        if supplied.resolve() != resolved["checkpoint"]:
            raise ReviewerClosureProtocolError("checkpoint_path differs from binding")

    commits = config["source_commits"]
    completed = commits["completed_branch_head"]
    report_source = commits["system_report_source"]
    official = commits["official_rescene"]
    for commit in (completed, report_source, official):
        _git(repository, "cat-file", "-e", f"{commit}^{{commit}}")
    parent = _git(repository, "rev-parse", f"{completed}^")
    if parent != report_source:
        raise ReviewerClosureProtocolError("completed branch parent differs from report source")
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", completed, "HEAD"],
            cwd=repository,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise ReviewerClosureProtocolError(
            "current branch does not descend from completed comparison"
        ) from error
    artifact_tree = _git(
        repository, "rev-parse", f"{completed}:artifacts/system_comparison"
    )
    if artifact_tree != config["baseline"]["artifact_tree"]:
        raise ReviewerClosureProtocolError("system-comparison artifact tree differs")
    old_artifact_status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "artifacts/system_comparison",
    )
    if old_artifact_status:
        raise ReviewerClosureProtocolError(
            "system-comparison artifacts have worktree changes"
        )

    report = resolved["system_report"].read_text(encoding="utf-8")
    if f"Source commit: `{report_source}`" not in report:
        raise ReviewerClosureProtocolError("system report source commit differs")
    if "Final classification: `SYSTEM_PARETO_LOCK`" not in report:
        raise ReviewerClosureProtocolError("system report classification differs")
    system_manifest = _load_json(resolved["system_manifest"], name="system manifest")
    if system_manifest.get("content_sha256") != (
        "8eb702d000b5d25c11249d06763faefcdb28c61b516774e9e296d77e57fd78ac"
    ):
        raise ReviewerClosureProtocolError("system manifest content digest differs")
    try:
        validate_system_comparison_manifest(
            system_manifest,
            source_protocol_path=resolved["protocol_b_manifest"],
        )
    except ProtocolBindingError as error:
        raise ReviewerClosureProtocolError(str(error)) from error
    return {
        "status": "pass",
        "reviewer_config_sha256": sha256_file(config_source),
        "completed_branch_head": completed,
        "completed_branch_parent": parent,
        "system_report_source": report_source,
        "official_rescene_commit": official,
        "system_comparison_artifact_tree": artifact_tree,
        "system_comparison_worktree_status": "clean",
        "checkpoint_sha256": config["sources"]["checkpoint"]["sha256"],
        "system_config_sha256": config["sources"]["system_config"]["sha256"],
        "protocol_sha256": config["sources"]["protocol_b_manifest"]["sha256"],
        "system_manifest_file_sha256": config["sources"]["system_manifest"]["sha256"],
        "system_manifest_content_sha256": system_manifest["content_sha256"],
        "baseline_classification": config["baseline"]["classification"],
    }


def build_reviewer_closure_manifest(
    config_path: str | Path,
    *,
    system_manifest_path: str | Path,
    binding: Mapping[str, object],
) -> dict[str, Any]:
    config = load_reviewer_closure_config(config_path)
    if binding.get("status") != "pass":
        raise ReviewerClosureProtocolError("reviewer-closure binding has not passed")
    if binding.get("reviewer_config_sha256") != sha256_file(config_path):
        raise ReviewerClosureProtocolError("reviewer-closure config digest differs")
    source_path = Path(system_manifest_path)
    source = _load_json(source_path, name="system manifest")
    if sha256_file(source_path) != config["sources"]["system_manifest"]["sha256"]:
        raise ReviewerClosureProtocolError("system manifest file SHA256 differs")
    if source.get("content_sha256") != binding.get("system_manifest_content_sha256"):
        raise ReviewerClosureProtocolError("system manifest content binding differs")
    manifest: dict[str, Any] = {
        "schema_version": "reviewer-closure-v1",
        "name": "Persist4D Reviewer Closure",
        "binding": copy.deepcopy(dict(binding)),
        "source_system_manifest": {
            "reference": "repo:artifacts/system_comparison/system_comparison_manifest.json",
            "file_sha256": sha256_file(source_path),
            "content_sha256": source["content_sha256"],
        },
        "protocol": copy.deepcopy(source["protocol"]),
        "sources": copy.deepcopy(source["sources"]),
        "masters": copy.deepcopy(source["masters"]),
        "phase_i_methods": copy.deepcopy(_PHASE_I_METHODS),
        "sidecar": copy.deepcopy(config["sidecar"]),
        "statistics": copy.deepcopy(config["statistics"]),
        "gate_i": copy.deepcopy(config["gate_i"]),
        "comparison_prefix_count": 43 * 3 * 4,
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    return manifest


def _validate_prefixes(manifest: Mapping[str, object]) -> None:
    masters = manifest.get("masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise ReviewerClosureProtocolError("masters must be a sequence")
    if len(masters) != 43:
        raise ReviewerClosureProtocolError("master coverage differs")
    clusters: set[str] = set()
    count = 0
    for raw_master in masters:
        master = _mapping(raw_master, name="master")
        clusters.add(str(master.get("reference_scene_id")))
        orders = _mapping(master.get("orders"), name="master.orders")
        if tuple(orders) != _ORDERS:
            raise ReviewerClosureProtocolError("order coverage differs")
        for order_name in _ORDERS:
            order = _mapping(orders[order_name], name="order")
            visit_order = order.get("visit_order")
            scan_indices = order.get("scan_indices")
            prefixes = _mapping(order.get("prefixes"), name="prefixes")
            if (
                not isinstance(visit_order, list)
                or not isinstance(scan_indices, list)
                or len(visit_order) != 5
                or len(scan_indices) != 5
            ):
                raise ReviewerClosureProtocolError("order scan coverage differs")
            for horizon in _HORIZONS:
                prefix = _mapping(prefixes.get(str(horizon)), name="prefix")
                if (
                    prefix.get("scan_ids") != visit_order[:horizon]
                    or prefix.get("scan_indices") != scan_indices[:horizon]
                ):
                    raise ReviewerClosureProtocolError("master prefix differs")
                count += 1
    if len(clusters) != 6 or count != 43 * 3 * 4:
        raise ReviewerClosureProtocolError("cluster or prefix coverage differs")


def validate_reviewer_closure_manifest(
    value: Mapping[str, object],
    *,
    config_path: str | Path,
    system_manifest_path: str | Path,
) -> dict[str, Any]:
    manifest = _mapping(value, name="reviewer-closure manifest")
    if manifest.get("content_sha256") != _content_sha256(manifest):
        raise ReviewerClosureProtocolError("reviewer-closure content digest differs")
    config = load_reviewer_closure_config(config_path)
    source_path = Path(system_manifest_path)
    source = _load_json(source_path, name="system manifest")
    if manifest.get("schema_version") != "reviewer-closure-v1":
        raise ReviewerClosureProtocolError("reviewer-closure schema differs")
    if manifest.get("masters") != source.get("masters"):
        raise ReviewerClosureProtocolError("reviewer-closure masters differ")
    if manifest.get("protocol") != source.get("protocol"):
        raise ReviewerClosureProtocolError("reviewer-closure protocol differs")
    if manifest.get("phase_i_methods") != _PHASE_I_METHODS:
        raise ReviewerClosureProtocolError("Phase I method contract differs")
    if manifest.get("sidecar") != config["sidecar"]:
        raise ReviewerClosureProtocolError("sidecar manifest contract differs")
    if manifest.get("statistics") != config["statistics"]:
        raise ReviewerClosureProtocolError("statistics manifest contract differs")
    if manifest.get("gate_i") != config["gate_i"]:
        raise ReviewerClosureProtocolError("Gate I manifest contract differs")
    if manifest.get("comparison_prefix_count") != 43 * 3 * 4:
        raise ReviewerClosureProtocolError("comparison prefix count differs")
    source_binding = _mapping(
        manifest.get("source_system_manifest"), name="source_system_manifest"
    )
    if source_binding.get("file_sha256") != sha256_file(source_path):
        raise ReviewerClosureProtocolError("source system manifest digest differs")
    if source_binding.get("content_sha256") != source.get("content_sha256"):
        raise ReviewerClosureProtocolError("source system content digest differs")
    binding = _mapping(manifest.get("binding"), name="binding")
    if (
        binding.get("status") != "pass"
        or binding.get("reviewer_config_sha256") != sha256_file(config_path)
        or binding.get("system_comparison_artifact_tree")
        != config["baseline"]["artifact_tree"]
    ):
        raise ReviewerClosureProtocolError("reviewer-closure binding differs")
    _validate_prefixes(manifest)
    return manifest


def full_history_observation_keys(
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    _validate_prefixes(manifest)
    keys: list[dict[str, object]] = []
    for master in manifest["masters"]:
        for order_name in _ORDERS:
            order = master["orders"][order_name]
            for horizon in _HORIZONS:
                prefix = order["prefixes"][str(horizon)]
                keys.append(
                    {
                        "reference_scene_id": master["reference_scene_id"],
                        "master_sequence_id": master["master_sequence_id"],
                        "order_id": order_name,
                        "horizon": horizon,
                        "history_scan_ids": copy.deepcopy(prefix["scan_ids"]),
                        "scan_indices": copy.deepcopy(prefix["scan_indices"]),
                    }
                )
    return keys


__all__ = [
    "ReviewerClosureProtocolError",
    "build_reviewer_closure_manifest",
    "full_history_observation_keys",
    "load_reviewer_closure_config",
    "validate_reviewer_closure_binding",
    "validate_reviewer_closure_manifest",
]
