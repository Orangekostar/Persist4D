"""Frozen incumbent and exact Protocol-B bindings for system comparison."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ORDERS = ("canonical", "reverse", "sha256_seed45")
_HORIZONS = (2, 3, 4, 5)
_METRICS = ("t_mAP", "t_mAP50", "t_mAP25", "t_REC", "t_REC50", "t_REC25")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class ProtocolBindingError(ValueError):
    """Raised when frozen system-comparison inputs differ from their binding."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolBindingError(f"{name} must be a mapping")
    return dict(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ProtocolBindingError(
            f"{name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolBindingError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolBindingError(f"{name} must be finite")
    return number


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ProtocolBindingError(f"{name} must be a lowercase 40-digit commit")
    return value


def _source(value: object, *, name: str) -> dict[str, str]:
    source = _mapping(value, name=name)
    _exact_keys(source, {"reference", "sha256"}, name=name)
    reference = source["reference"]
    digest = source["sha256"]
    if not isinstance(reference, str) or not reference:
        raise ProtocolBindingError(f"{name}.reference must be nonempty")
    if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
        raise ProtocolBindingError(f"{name}.sha256 must contain 64 lowercase hex digits")
    return {"reference": reference, "sha256": digest}


def load_incumbent_config(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    try:
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ProtocolBindingError(f"cannot load incumbent config: {error}") from error
    root = _mapping(payload, name="incumbent config")
    _exact_keys(
        root,
        {
            "schema_version",
            "stage_name",
            "method",
            "memory",
            "source_commits",
            "sources",
            "reference_metrics",
            "profile",
            "statistics",
            "decision",
        },
        name="incumbent config",
    )
    if root["schema_version"] != 1:
        raise ProtocolBindingError("incumbent config schema_version must be 1")
    if root["stage_name"] != "full-history-vs-persistent-state-system-comparison":
        raise ProtocolBindingError("incumbent config stage_name differs")

    method = _mapping(root["method"], name="method")
    if method != {
        "id": "B4",
        "name": "Persist4D Persistent-State",
        "implementation": "frozen_p5_persist4d",
    }:
        raise ProtocolBindingError("method is not the frozen P6-A B4 incumbent")

    expected_memory: dict[str, int | float] = {
        "capacity": 100,
        "association_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
        "update_rate": 0.2,
        "max_update_rate": 0.2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }
    memory = _mapping(root["memory"], name="memory")
    if memory != expected_memory:
        raise ProtocolBindingError("memory differs from the frozen P6-A B4 settings")

    commits = _mapping(root["source_commits"], name="source_commits")
    _exact_keys(commits, {"p6a", "implementation_base"}, name="source_commits")
    _commit(commits["p6a"], name="source_commits.p6a")
    _commit(commits["implementation_base"], name="source_commits.implementation_base")

    sources = _mapping(root["sources"], name="sources")
    source_names = {
        "p6a_config",
        "checkpoint",
        "p6a_report",
        "p6a_strict_results",
        "p6a_protocol_manifest",
    }
    _exact_keys(sources, source_names, name="sources")
    root["sources"] = {
        name: _source(sources[name], name=f"sources.{name}")
        for name in sorted(source_names)
    }

    references = _mapping(root["reference_metrics"], name="reference_metrics")
    _exact_keys(references, {f"T{value}" for value in _HORIZONS}, name="reference_metrics")
    for horizon in _HORIZONS:
        values = _mapping(references[f"T{horizon}"], name=f"reference_metrics.T{horizon}")
        _exact_keys(values, set(_METRICS), name=f"reference_metrics.T{horizon}")
        references[f"T{horizon}"] = {
            metric: _finite(values[metric], name=f"reference_metrics.T{horizon}.{metric}")
            for metric in _METRICS
        }
    root["reference_metrics"] = references

    profile = _mapping(root["profile"], name="profile")
    if profile != {
        "warmup_repeats": 5,
        "measured_repeats": 10,
        "subset_rule": "first_master_per_reference_scene_canonical",
    }:
        raise ProtocolBindingError("profile settings differ from the frozen protocol")
    statistics = _mapping(root["statistics"], name="statistics")
    if statistics != {
        "cluster_unit": "reference_scene_id",
        "bootstrap_replicates": 10000,
        "seed": 45,
    }:
        raise ProtocolBindingError("statistics settings differ from the frozen protocol")
    decision = _mapping(root["decision"], name="decision")
    _exact_keys(
        decision,
        {
            "system_lock_task_noninferiority_tolerance",
            "meaningful_full_history_tmap_advantage",
            "meaningful_advantage_requires_negative_ci",
            "oracle_within_full_history_tmap",
            "oracle_deficit_closure_fraction",
        },
        name="decision",
    )
    if decision["meaningful_advantage_requires_negative_ci"] is not True:
        raise ProtocolBindingError("meaningful advantage must require a negative paired CI")
    for key in set(decision) - {"meaningful_advantage_requires_negative_ci"}:
        value = _finite(decision[key], name=f"decision.{key}")
        if not 0.0 <= value <= 1.0:
            raise ProtocolBindingError(f"decision.{key} must be within [0, 1]")
    return root


def _resolve_reference(reference: str, *, repo_root: Path) -> Path:
    if reference.startswith("repo:"):
        relative = Path(reference.removeprefix("repo:"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolBindingError("repo reference must remain inside the repository")
        path = repo_root / relative
    else:
        path = Path(reference)
    if path.is_symlink() or not path.is_file():
        raise ProtocolBindingError(f"bound source is not a regular file: {reference}")
    return path.resolve()


def _verify_commit_exists(repo_root: Path, commit: str, *, name: str) -> None:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProtocolBindingError(f"{name} is unavailable in repository history") from error


def _reference_b4_rows(path: Path) -> dict[str, dict[str, float]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ProtocolBindingError("cannot parse P6-A strict result rows") from error
    selected: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("method") != "B4":
            continue
        horizon = row.get("T")
        key = f"T{horizon}"
        if key in selected or key not in {f"T{value}" for value in _HORIZONS}:
            raise ProtocolBindingError("P6-A strict results contain invalid B4 horizons")
        selected[key] = {
            metric: _finite(float(row[metric]), name=f"P6-A {key}.{metric}")
            for metric in _METRICS
        }
    if set(selected) != {f"T{value}" for value in _HORIZONS}:
        raise ProtocolBindingError("P6-A strict results lack exact B4 T2-T5 rows")
    return selected


def validate_incumbent_binding(
    config_path: str | Path,
    *,
    repo_root: str | Path = PROJECT_ROOT,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    repository = Path(repo_root).resolve()
    config_source = Path(config_path)
    config = load_incumbent_config(config_source)
    commits = config["source_commits"]
    for name, commit in commits.items():
        _verify_commit_exists(repository, commit, name=f"source_commits.{name}")

    resolved: dict[str, Path] = {}
    for name, source in config["sources"].items():
        path = _resolve_reference(source["reference"], repo_root=repository)
        observed = sha256_file(path)
        if observed != source["sha256"]:
            raise ProtocolBindingError(f"{name} SHA256 differs from frozen binding")
        resolved[name] = path
    if checkpoint_path is not None:
        supplied = Path(checkpoint_path)
        if supplied.is_symlink() or not supplied.is_file():
            raise ProtocolBindingError("checkpoint_path must be a regular file")
        if supplied.resolve() != resolved["checkpoint"]:
            raise ProtocolBindingError("checkpoint_path differs from frozen binding")

    observed_metrics = _reference_b4_rows(resolved["p6a_strict_results"])
    for horizon, expected in config["reference_metrics"].items():
        for metric, expected_value in expected.items():
            if not math.isclose(
                observed_metrics[horizon][metric],
                expected_value,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ProtocolBindingError(
                    f"P6-A B4 reference metric differs: {horizon}.{metric}"
                )

    return {
        "status": "pass",
        "incumbent_config_sha256": sha256_file(config_source),
        "p6a_source_commit": commits["p6a"],
        "implementation_base_commit": commits["implementation_base"],
        "checkpoint_sha256": config["sources"]["checkpoint"]["sha256"],
        "p6a_config_sha256": config["sources"]["p6a_config"]["sha256"],
        "p6a_report_sha256": config["sources"]["p6a_report"]["sha256"],
        "p6a_strict_results_sha256": config["sources"]["p6a_strict_results"]["sha256"],
        "p6a_protocol_manifest_sha256": config["sources"]["p6a_protocol_manifest"]["sha256"],
        "reference_metrics": copy.deepcopy(observed_metrics),
    }


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProtocolBindingError("source Protocol-B manifest must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolBindingError("cannot decode source Protocol-B manifest") from error
    return _mapping(payload, name="source Protocol-B manifest")


def _validate_source_protocol(protocol: Mapping[str, object]) -> None:
    if protocol.get("schema_version") != "protocol-b-v1":
        raise ProtocolBindingError("source Protocol-B schema differs")
    config = _mapping(protocol.get("protocol"), name="source protocol")
    if config.get("horizons") != list(_HORIZONS):
        raise ProtocolBindingError("source Protocol-B horizons differ")
    if config.get("order_variants") != list(_ORDERS):
        raise ProtocolBindingError("source Protocol-B orders differ")
    if config.get("expected_master_count") != 43:
        raise ProtocolBindingError("source Protocol-B master count differs")
    masters = protocol.get("masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise ProtocolBindingError("source Protocol-B masters must be a sequence")
    if len(masters) != 43:
        raise ProtocolBindingError("source Protocol-B must contain 43 masters")
    references: set[str] = set()
    prefix_count = 0
    for master_index, raw_master in enumerate(masters):
        master = _mapping(raw_master, name=f"master {master_index}")
        reference = master.get("reference_scene_id")
        if not isinstance(reference, str) or not reference:
            raise ProtocolBindingError("master reference_scene_id is invalid")
        references.add(reference)
        orders = _mapping(master.get("orders"), name=f"master {master_index}.orders")
        if tuple(orders) != _ORDERS:
            raise ProtocolBindingError("master orders differ from preregistration")
        for order_name in _ORDERS:
            order = _mapping(orders[order_name], name=f"master {master_index}.{order_name}")
            scan_ids = order.get("visit_order")
            scan_indices = order.get("scan_indices")
            prefixes = _mapping(order.get("prefixes"), name="order prefixes")
            if (
                not isinstance(scan_ids, list)
                or not isinstance(scan_indices, list)
                or len(scan_ids) != 5
                or len(scan_indices) != 5
                or set(prefixes) != {str(value) for value in _HORIZONS}
            ):
                raise ProtocolBindingError("Protocol-B order coverage differs")
            for horizon in _HORIZONS:
                prefix = _mapping(prefixes[str(horizon)], name="prefix")
                if (
                    prefix.get("scan_ids") != scan_ids[:horizon]
                    or prefix.get("scan_indices") != scan_indices[:horizon]
                ):
                    raise ProtocolBindingError("Protocol-B is not an exact prefix")
                prefix_count += 1
    if len(references) != 6 or prefix_count != 43 * 3 * 4:
        raise ProtocolBindingError("Protocol-B cluster or prefix coverage differs")


def build_system_comparison_manifest(
    protocol_path: str | Path,
    *,
    incumbent_binding: Mapping[str, object],
) -> dict[str, Any]:
    path = Path(protocol_path)
    protocol = _load_protocol(path)
    _validate_source_protocol(protocol)
    source_digest = sha256_file(path)
    if incumbent_binding.get("status") != "pass":
        raise ProtocolBindingError("incumbent binding has not passed")
    if incumbent_binding.get("p6a_protocol_manifest_sha256") != source_digest:
        raise ProtocolBindingError("incumbent Protocol-B hash differs")
    manifest: dict[str, Any] = {
        "schema_version": "system-comparison-v1",
        "name": "Full-History vs Persistent-State System Comparison",
        "source_protocol": {
            "reference": "repo:artifacts/P6A/protocol_b_manifest.json",
            "sha256": source_digest,
            "schema_version": protocol["schema_version"],
        },
        "incumbent_binding": copy.deepcopy(dict(incumbent_binding)),
        "systems": [
            {
                "id": "full_history",
                "name": "ReScene4D Full-History (Frozen T2 Checkpoint)",
                "history_strategy": "reprocess_exact_observed_prefix",
            },
            {
                "id": "persistent_state",
                "name": "Persist4D Persistent-State",
                "history_strategy": "latest_pair_plus_bounded_entity_state",
            },
        ],
        "protocol": {
            **copy.deepcopy(protocol["protocol"]),
            "identity_initialization_horizon": 1,
            "task_quality_horizons": list(_HORIZONS),
        },
        "sources": copy.deepcopy(protocol["sources"]),
        "masters": copy.deepcopy(protocol["masters"]),
        "comparison_prefix_count": 43 * 3 * 4,
        "identity_initialization_count": 43 * 3,
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    return manifest


def validate_system_comparison_manifest(
    value: Mapping[str, object],
    *,
    source_protocol_path: str | Path,
) -> dict[str, Any]:
    manifest = _mapping(value, name="system comparison manifest")
    if manifest.get("content_sha256") != _content_sha256(manifest):
        raise ProtocolBindingError("system comparison content digest differs")
    source_path = Path(source_protocol_path)
    source = _load_protocol(source_path)
    _validate_source_protocol(source)
    if manifest.get("schema_version") != "system-comparison-v1":
        raise ProtocolBindingError("system comparison schema differs")
    source_binding = _mapping(manifest.get("source_protocol"), name="source_protocol")
    if source_binding.get("sha256") != sha256_file(source_path):
        raise ProtocolBindingError("system comparison source Protocol-B hash differs")
    if manifest.get("masters") != source.get("masters"):
        raise ProtocolBindingError("system comparison masters differ from source")
    if manifest.get("comparison_prefix_count") != 43 * 3 * 4:
        raise ProtocolBindingError("system comparison prefix count differs")
    if manifest.get("identity_initialization_count") != 43 * 3:
        raise ProtocolBindingError("identity initialization count differs")
    synthetic_source = {
        "schema_version": "protocol-b-v1",
        "protocol": {
            key: value
            for key, value in _mapping(manifest.get("protocol"), name="protocol").items()
            if key not in {"identity_initialization_horizon", "task_quality_horizons"}
        },
        "masters": manifest["masters"],
    }
    _validate_source_protocol(synthetic_source)
    return manifest


def write_canonical_json(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    payload = _canonical_bytes(value)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ProtocolBindingError",
    "build_system_comparison_manifest",
    "load_incumbent_config",
    "sha256_file",
    "validate_incumbent_binding",
    "validate_system_comparison_manifest",
    "write_canonical_json",
]
