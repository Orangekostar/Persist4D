"""Strict, deterministic, CPU-only evidence helpers for Persist4D P6-A.

The root payload is the only source of truth.  Renderers never read a file
from disk while constructing a bundle; publication verifies every rendered
byte against the root manifest before exposing any output.
"""

from __future__ import annotations

import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 2

ROOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_id",
        "source_commit",
        "source_tree_contract",
        "p5_frozen_hashes",
        "protocol",
        "provenance",
        "methods",
        "horizons",
        "settings",
        "metric_blocks",
        "fingerprints",
        "analysis",
        "change_label_limitation",
        "derived_artifacts",
        "artifact_manifest",
        "gate_results",
        "claims_supported",
        "claims_not_supported",
        "next_action",
        "errors",
    }
)
SOURCE_TREE_KEYS = frozenset({"status", "source_commit"})
P5_FROZEN_KEYS = frozenset(
    {"git_commit", "checkpoint_sha256", "config_sha256", "dataset_sha256"}
)
PROTOCOL_KEYS = frozenset(
    {
        "name",
        "horizons",
        "master_sequence_count",
        "cluster_count",
        "order_count",
        "cache_entry_count",
    }
)
PROVENANCE_KEYS = frozenset({"checkpoint", "config", "dataset", "prediction_cache"})
PROVENANCE_RECORD_KEYS = frozenset({"ref", "sha256"})
METHOD_RECORD_KEYS = frozenset({"mode", "metric_block"})
HORIZON_IDS = ("T2", "T3", "T4", "T5")
HORIZON_SEQUENCE_COUNTS = {"T2": 154, "T3": 120, "T4": 75, "T5": 43}
METHOD_IDS = ("B0", "B0_sanity", "B1", "B2", "B3", "B4", "Oracle")
ONLINE_METHOD_IDS = METHOD_IDS[:-1]
METRIC_BLOCK_IDS = ("raw", "strict", "offline")
METRIC_FIELDS = (
    "AP",
    "AP50",
    "AP25",
    "REC",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
)
ANALYSIS_GROUPS = (
    "association",
    "error",
    "reactivation",
    "capacity",
    "efficiency",
    "statistical",
)
ANALYSIS_RECORD_KEYS = frozenset({"path", "rows", "status"})
CHANGE_LABEL_KEYS = frozenset({"available", "reason", "scope"})
GATE_RECORD_KEYS = frozenset({"passed", "evidence"})
GATE_IDS = tuple(f"G6A-{index}" for index in range(1, 6))
MANIFEST_RECORD_KEYS = frozenset({"path", "bytes", "sha256"})
DERIVED_KIND_IDS = ("csv", "json", "markdown", "svg")
DERIVED_RECORD_KEYS = {
    "csv": frozenset({"columns", "rows"}),
    "json": frozenset({"text"}),
    "markdown": frozenset({"text"}),
    "svg": frozenset({"text"}),
}
REQUIRED_CSV_PATHS = frozenset(
    {
        "baseline_results.csv",
        "strict_online_results.csv",
        "raw_local_results.csv",
        "per_sequence_results.csv",
        "association_events.csv",
        "error_breakdown.csv",
        "reactivation_audit.csv",
        "capacity_audit.csv",
        "efficiency_results.csv",
    }
)
REQUIRED_JSON_PATHS = frozenset({"protocol_b_manifest.json"})
REQUIRED_MARKDOWN_PATHS = frozenset({"statistical_analysis.md"})
REQUIRED_SVG_PATHS = frozenset(
    {
        "figures/figure_a_identity.svg",
        "figures/figure_b_online_tmap.svg",
        "figures/figure_c_reactivation.svg",
        "figures/figure_d_failures.svg",
        "figures/figure_e_latency.svg",
    }
)
REPORT_PATH = "P6A_GO_NOGO_REPORT.md"
P6A_REPORT_SECTIONS = (
    "What was changed",
    "Why it was changed",
    "Experimental protocol",
    "Reproducibility binding",
    "Main results",
    "Statistical evidence",
    "Failure analysis",
    "What claims are supported",
    "What claims are NOT supported",
    "GO / NO-GO decision",
    "Exact next action",
)

_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_GPU_UUID = re.compile(r"(?:^|[^A-Za-z])GPU-[0-9A-Fa-f-]+")
_IPV4 = re.compile(r"(?<![0-9A-Za-z])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9A-Za-z])")
_IPV6 = re.compile(r"(?i)(?<![0-9A-F])(?:[0-9A-F]{1,4}:){2,}[0-9A-F:]+(?![0-9A-F])")
_PRIVATE_TEXT = (
    re.compile(r"/(?:home|Users|root|private|mnt)/"),
    _GPU_UUID,
    re.compile(r"ssh://"),
    _WINDOWS_ABSOLUTE,
)


def _exact_keys(value: object, expected: frozenset[str] | set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(value)
    if actual != set(expected):
        raise ValueError(
            f"{name} keys differ: missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, name: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    result = [_nonempty_string(item, name=f"{name} item") for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _validate_scalar_tree(value: object, *, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        if PurePosixPath(value).is_absolute() or _WINDOWS_ABSOLUTE.search(value):
            raise ValueError(f"{path} contains an absolute path")
        if any(pattern.search(value) for pattern in _PRIVATE_TEXT):
            raise ValueError(f"{path} contains private or non-portable text")
        if _IPV4.search(value) or _IPV6.search(value):
            candidate = _IPV4.search(value) or _IPV6.search(value)
            try:
                ipaddress.ip_address(candidate.group(0))
            except ValueError:
                pass
            else:
                raise ValueError(f"{path} contains an IP address")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} mapping keys must be strings")
            _validate_scalar_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_scalar_tree(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _validate_sha(value: object, *, name: str, length: int) -> str:
    digest = _nonempty_string(value, name=name)
    pattern = _HEX_40 if length == 40 else _HEX_64
    if pattern.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA{length * 4}")
    return digest


def _validate_reference(
    value: object, *, name: str, expected_prefix: str | None = None
) -> str:
    reference = _nonempty_string(value, name=name)
    scheme, separator, payload = reference.partition(":")
    if separator != ":" or scheme not in {"repo", "external", "local_cache"}:
        raise ValueError(f"{name} must use a portable reference prefix")
    if not payload or payload.startswith("/") or "\\" in payload:
        raise ValueError(f"{name} must be a relative portable reference")
    if ".." in PurePosixPath(payload).parts or "//" in payload:
        raise ValueError(f"{name} must not traverse parent directories")
    if expected_prefix is not None and not reference.startswith(expected_prefix):
        raise ValueError(f"{name} must start with {expected_prefix}")
    return reference


def _validate_relative_artifact_path(value: object, *, name: str) -> str:
    text = _nonempty_string(value, name=name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or "\\" in text
        or _WINDOWS_ABSOLUTE.search(text)
        or path.parts[0] in {"P5", "P6B", "artifacts"}
    ):
        raise ValueError(f"{name} must be a safe relative P6A path")
    return path.as_posix()


def _validate_metric_record(value: object, *, name: str) -> None:
    record = _exact_keys(value, frozenset(METRIC_FIELDS), name=name)
    available = False
    for field in METRIC_FIELDS:
        item = record[field]
        _finite_number(item, name=f"{name}.{field}", allow_none=True)
        if item is not None:
            available = True
            if not 0.0 <= float(item) <= 1.0:
                raise ValueError(f"{name}.{field} must be within [0, 1]")
    if not available:
        raise ValueError(f"{name} must contain at least one metric")


def _validate_metric_blocks(value: object) -> None:
    blocks = _exact_keys(value, frozenset(METRIC_BLOCK_IDS), name="metric_blocks")
    for block_name in METRIC_BLOCK_IDS:
        expected_methods = ONLINE_METHOD_IDS if block_name != "offline" else METHOD_IDS
        methods = _exact_keys(
            blocks[block_name], frozenset(expected_methods), name=f"metric_blocks.{block_name}"
        )
        for method in expected_methods:
            horizons = _exact_keys(
                methods[method], frozenset(HORIZON_IDS),
                name=f"metric_blocks.{block_name}.{method}",
            )
            for horizon in HORIZON_IDS:
                _validate_metric_record(
                    horizons[horizon],
                    name=f"metric_blocks.{block_name}.{method}.{horizon}",
                )


def _validate_fingerprints(value: object) -> None:
    fingerprints = _exact_keys(value, frozenset({"prediction", "cache"}), name="fingerprints")
    for kind in ("prediction", "cache"):
        records = _exact_keys(
            fingerprints[kind], frozenset(METHOD_IDS), name=f"fingerprints.{kind}"
        )
        values = []
        for method in METHOD_IDS:
            values.append(
                _validate_sha(
                    records[method], name=f"fingerprints.{kind}.{method}", length=64
                )
            )
        if len(set(values)) != 1:
            raise ValueError(f"fingerprints.{kind} must bind one shared frozen value")


def _validate_derived_artifacts(value: object) -> None:
    derived = _exact_keys(value, frozenset(DERIVED_KIND_IDS), name="derived_artifacts")
    expected_paths = {
        "csv": REQUIRED_CSV_PATHS,
        "json": REQUIRED_JSON_PATHS,
        "markdown": REQUIRED_MARKDOWN_PATHS,
        "svg": REQUIRED_SVG_PATHS,
    }
    for kind in DERIVED_KIND_IDS:
        records = derived[kind]
        if not isinstance(records, Mapping) or not records:
            raise ValueError(f"derived_artifacts.{kind} must not be empty")
        actual_paths = set(records)
        if actual_paths != set(expected_paths[kind]):
            raise ValueError(
                f"derived_artifacts.{kind} paths differ: "
                f"missing={sorted(set(expected_paths[kind]) - actual_paths)}, "
                f"extra={sorted(actual_paths - set(expected_paths[kind]))}"
            )
        suffix = ".csv" if kind == "csv" else ".json" if kind == "json" else ".md" if kind == "markdown" else ".svg"
        for raw_path, record in records.items():
            path = _validate_relative_artifact_path(
                raw_path, name=f"derived_artifacts.{kind} path"
            )
            if not path.endswith(suffix):
                raise ValueError(f"{path} has the wrong derived artifact kind")
            expected_keys = DERIVED_RECORD_KEYS[kind]
            normalized = _exact_keys(record, expected_keys, name=f"derived_artifacts.{path}")
            if kind == "csv":
                columns = normalized["columns"]
                if (
                    not isinstance(columns, list)
                    or not columns
                    or any(not isinstance(column, str) or not column for column in columns)
                    or len(set(columns)) != len(columns)
                ):
                    raise ValueError(f"{path} columns must be a non-empty unique list")
                rows = normalized["rows"]
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{path} rows must not be empty")
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping) or set(row) != set(columns):
                        raise ValueError(f"{path} row {index} has schema drift")
                    _validate_scalar_tree(row, path=f"{path}.rows[{index}]")
                render_csv(rows, columns=columns)
            else:
                text = _nonempty_string(normalized["text"], name=f"{path}.text")
                if kind == "svg" and not text.lstrip().startswith("<svg"):
                    raise ValueError(f"{path} must contain an SVG root")
                if kind == "json":
                    try:
                        json.loads(text)
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"{path} must contain valid JSON") from error


def _validate_manifest(value: object, *, derived: Mapping[str, object]) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("artifact_manifest must be a non-empty list")
    paths: list[str] = []
    for index, entry in enumerate(value):
        record = _exact_keys(entry, MANIFEST_RECORD_KEYS, name=f"artifact_manifest[{index}]")
        path = _validate_relative_artifact_path(
            record["path"], name=f"artifact_manifest[{index}].path"
        )
        _integer(record["bytes"], name=f"artifact_manifest[{index}].bytes", minimum=1)
        _validate_sha(record["sha256"], name=f"artifact_manifest[{index}].sha256", length=64)
        paths.append(path)
    if len(set(paths)) != len(paths) or paths != sorted(paths):
        raise ValueError("artifact_manifest paths must be unique and sorted")
    derived_paths = {
        path
        for kind in DERIVED_KIND_IDS
        for path in derived[kind]
    }
    expected_paths = derived_paths | {REPORT_PATH}
    if set(paths) != expected_paths:
        raise ValueError(
            f"artifact_manifest paths differ: missing={sorted(expected_paths - set(paths))}, "
            f"extra={sorted(set(paths) - expected_paths)}"
        )


def validate_root_artifact(artifact: object) -> None:
    """Validate the complete P6-A root contract and reject schema drift."""

    root = _exact_keys(artifact, ROOT_KEYS, name="artifact")
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported P6-A schema_version")
    if root["status"] != "pass":
        raise ValueError("complete P6-A artifact status must be pass")
    _nonempty_string(root["run_id"], name="run_id")
    source_commit = _validate_sha(root["source_commit"], name="source_commit", length=40)

    source_tree = _exact_keys(root["source_tree_contract"], SOURCE_TREE_KEYS, name="source_tree_contract")
    if source_tree["status"] != "pass" or source_tree["source_commit"] != source_commit:
        raise ValueError("source_tree_contract must bind the passing source commit")

    frozen = _exact_keys(root["p5_frozen_hashes"], P5_FROZEN_KEYS, name="p5_frozen_hashes")
    if _validate_sha(frozen["git_commit"], name="p5_frozen_hashes.git_commit", length=40) != source_commit:
        raise ValueError("p5_frozen_hashes.git_commit must equal source_commit")
    frozen_checkpoint = _validate_sha(frozen["checkpoint_sha256"], name="p5_frozen_hashes.checkpoint_sha256", length=64)
    frozen_config = _validate_sha(frozen["config_sha256"], name="p5_frozen_hashes.config_sha256", length=64)
    frozen_dataset = _validate_sha(frozen["dataset_sha256"], name="p5_frozen_hashes.dataset_sha256", length=64)

    protocol = _exact_keys(root["protocol"], PROTOCOL_KEYS, name="protocol")
    if protocol["name"] != "exact_common_prefix_protocol_b":
        raise ValueError("protocol name must identify exact common-prefix Protocol B")
    if protocol["horizons"] != [2, 3, 4, 5]:
        raise ValueError("protocol horizons must be exactly [2, 3, 4, 5]")
    for field, expected in (
        ("master_sequence_count", 43),
        ("cluster_count", 6),
        ("order_count", 3),
        ("cache_entry_count", 645),
    ):
        if _integer(protocol[field], name=f"protocol.{field}", minimum=1) != expected:
            raise ValueError(f"protocol.{field} must be exactly {expected}")

    provenance = _exact_keys(root["provenance"], PROVENANCE_KEYS, name="provenance")
    expected_reference_prefix = {
        "checkpoint": "repo:checkpoints/",
        "config": "repo:conf/",
        "dataset": "repo:data/",
        "prediction_cache": "repo:artifacts/P6A/",
    }
    expected_digest = {
        "checkpoint": frozen_checkpoint,
        "config": frozen_config,
        "dataset": frozen_dataset,
    }
    for key in sorted(PROVENANCE_KEYS):
        record = _exact_keys(provenance[key], PROVENANCE_RECORD_KEYS, name=f"provenance.{key}")
        _validate_reference(
            record["ref"], name=f"provenance.{key}.ref", expected_prefix=expected_reference_prefix[key]
        )
        digest = _validate_sha(record["sha256"], name=f"provenance.{key}.sha256", length=64)
        if key in expected_digest and digest != expected_digest[key]:
            raise ValueError(f"provenance.{key}.sha256 must match P5 frozen hash")

    settings = _exact_keys(
        root["settings"], frozenset({"bootstrap_seed", "bootstrap_replicates"}), name="settings"
    )
    _integer(settings["bootstrap_seed"], name="settings.bootstrap_seed", minimum=0)
    _integer(settings["bootstrap_replicates"], name="settings.bootstrap_replicates", minimum=1)

    methods = _exact_keys(root["methods"], frozenset({"set", "oracle"}), name="methods")
    if methods["set"] != list(METHOD_IDS):
        raise ValueError("methods.set must be exactly B0, B0_sanity, B1-B4, Oracle")
    oracle = _exact_keys(methods["oracle"], METHOD_RECORD_KEYS, name="methods.oracle")
    if oracle["mode"] != "offline" or oracle["metric_block"] != "offline":
        raise ValueError("Oracle is allowed only in the offline metric block")

    horizons = _exact_keys(root["horizons"], frozenset(HORIZON_IDS), name="horizons")
    for horizon in HORIZON_IDS:
        record = _exact_keys(horizons[horizon], frozenset({"sequence_count"}), name=f"horizons.{horizon}")
        expected = HORIZON_SEQUENCE_COUNTS[horizon]
        if _integer(record["sequence_count"], name=f"horizons.{horizon}.sequence_count", minimum=1) != expected:
            raise ValueError(f"horizons.{horizon}.sequence_count must be exactly {expected}")

    _validate_metric_blocks(root["metric_blocks"])
    _validate_fingerprints(root["fingerprints"])

    analysis = _exact_keys(root["analysis"], frozenset(ANALYSIS_GROUPS), name="analysis")
    expected_analysis_paths = {
        "association": "association_events.csv",
        "error": "error_breakdown.csv",
        "reactivation": "reactivation_audit.csv",
        "capacity": "capacity_audit.csv",
        "efficiency": "efficiency_results.csv",
        "statistical": "statistical_analysis.md",
    }
    for group in ANALYSIS_GROUPS:
        record = _exact_keys(analysis[group], ANALYSIS_RECORD_KEYS, name=f"analysis.{group}")
        if record["status"] != "pass" or record["path"] != expected_analysis_paths[group]:
            raise ValueError(f"analysis.{group} must bind its required passing artifact")
        _integer(record["rows"], name=f"analysis.{group}.rows", minimum=1)

    limitation = _exact_keys(root["change_label_limitation"], CHANGE_LABEL_KEYS, name="change_label_limitation")
    if limitation["available"] is not False:
        raise ValueError("change_label_limitation.available must be false for P6-A")
    _nonempty_string(limitation["reason"], name="change_label_limitation.reason")
    _nonempty_string(limitation["scope"], name="change_label_limitation.scope")

    _validate_derived_artifacts(root["derived_artifacts"])
    _validate_manifest(root["artifact_manifest"], derived=root["derived_artifacts"])

    gates = _exact_keys(root["gate_results"], frozenset(GATE_IDS), name="gate_results")
    for gate_id in GATE_IDS:
        gate = _exact_keys(gates[gate_id], GATE_RECORD_KEYS, name=gate_id)
        if not isinstance(gate["passed"], bool):
            raise ValueError(f"{gate_id}.passed must be boolean")
        _nonempty_string(gate["evidence"], name=f"{gate_id}.evidence")
    _string_list(root["claims_supported"], name="claims_supported")
    _string_list(root["claims_not_supported"], name="claims_not_supported")
    next_action = _nonempty_string(root["next_action"], name="next_action")
    if "p6b" in next_action.casefold() or "enter" in next_action.casefold() and "p6a" not in next_action.casefold():
        raise ValueError("next_action must stop at P6-A and require explicit continuation")
    _string_list(root["errors"], name="errors")
    _validate_scalar_tree(root)


def artifact_json_text(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    return json.dumps(artifact, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _markdown_list(values: object) -> str:
    items = _string_list(values, name="report list")
    if not items:
        return "None."
    return "\n".join(f"- {item}" for item in items)


def render_go_nogo_report(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    gates = artifact["gate_results"]
    decision = "P6A_GO" if all(gates[key]["passed"] for key in GATE_IDS) else "P6A_STOP"
    gate_lines = "\n".join(
        f"- {key}: {'PASS' if gates[key]['passed'] else 'FAIL'} - {gates[key]['evidence']}"
        for key in GATE_IDS
    )
    protocol = artifact["protocol"]
    p5 = artifact["p5_frozen_hashes"]
    analysis = artifact["analysis"]
    sections = {
        "What was changed": "Implemented the P6-A scientific evidence package with a frozen root payload.",
        "Why it was changed": "To isolate cross-stage association and state maintenance from frozen local perception.",
        "Experimental protocol": (
            f"Protocol B uses exactly {protocol['master_sequence_count']} masters, "
            f"{protocol['cluster_count']} reference-scene clusters, {protocol['order_count']} orders, "
            f"and {protocol['cache_entry_count']} cache entries at T=2/3/4/5."
        ),
        "Reproducibility binding": (
            f"Source commit: `{artifact['source_commit']}`; P5 checkpoint SHA256: `{p5['checkpoint_sha256']}`; "
            f"config SHA256: `{p5['config_sha256']}`; dataset SHA256: `{p5['dataset_sha256']}`."
        ),
        "Main results": gate_lines,
        "Statistical evidence": f"See `{analysis['statistical']['path']}`.",
        "Failure analysis": (
            f"Association: `{analysis['association']['path']}`; error: `{analysis['error']['path']}`; "
            f"reactivation: `{analysis['reactivation']['path']}`."
        ),
        "What claims are supported": _markdown_list(artifact["claims_supported"]),
        "What claims are NOT supported": _markdown_list(artifact["claims_not_supported"]),
        "GO / NO-GO decision": f"Decision: {decision}",
        "Exact next action": f"Exact next action: {artifact['next_action']}",
    }
    body = ["# Persist4D P6-A GO / NO-GO Report"]
    for section in P6A_REPORT_SECTIONS:
        body.extend(("", f"## {section}", "", sections[section]))
    return "\n".join(body) + "\n"


def render_csv(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]) -> str:
    normalized_columns = tuple(columns)
    if not normalized_columns or len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("CSV columns must be non-empty and unique")
    if any(not isinstance(column, str) or not column for column in normalized_columns):
        raise ValueError("CSV columns must be non-empty strings")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=normalized_columns, extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    expected = set(normalized_columns)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"CSV row {index} does not match the exact columns")
        _validate_scalar_tree(row, path=f"rows[{index}]")
        writer.writerow(row)
    return output.getvalue()


def render_artifact_bundle(artifact: Mapping[str, object]) -> dict[str, bytes]:
    """Render every P6-A derived artifact from one validated root payload."""

    validate_root_artifact(artifact)
    derived = artifact["derived_artifacts"]
    rendered: dict[str, bytes] = {REPORT_PATH: render_go_nogo_report(artifact).encode("utf-8")}
    for path, spec in sorted(derived["csv"].items()):
        rendered[path] = render_csv(spec["rows"], columns=spec["columns"]).encode("utf-8")
    for kind in ("json", "markdown", "svg"):
        for path, spec in sorted(derived[kind].items()):
            rendered[path] = spec["text"].encode("utf-8")
    return dict(sorted(rendered.items()))


def _normalized_files(files: Mapping[str, str | bytes]) -> list[tuple[PurePosixPath, bytes]]:
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files must be a non-empty mapping")
    normalized: list[tuple[PurePosixPath, bytes]] = []
    for raw_path, content in files.items():
        relative = PurePosixPath(_validate_relative_artifact_path(raw_path, name="artifact output path"))
        if not isinstance(content, (str, bytes)):
            raise ValueError("artifact content must be text or bytes")
        payload = content.encode("utf-8") if isinstance(content, str) else content
        if not payload:
            raise ValueError(f"artifact {relative.as_posix()} must not be empty")
        normalized.append((relative, payload))
    normalized.sort(key=lambda item: item[0].as_posix())
    if len({path for path, _ in normalized}) != len(normalized):
        raise ValueError("artifact output paths must be unique")
    return normalized


def _check_existing_path_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"output path contains a symlink: {component}")
        if component.exists() and not component.is_dir():
            raise ValueError(f"output path component is not a directory: {component}")


def _validate_output_root(output_root: Path, targets: Sequence[Path]) -> None:
    if not isinstance(output_root, Path):
        raise ValueError("output_root must be a Path")
    _check_existing_path_components(output_root.parent)
    if output_root.is_symlink():
        raise ValueError("output_root must not be a symlink")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output_root must be a directory")
    for target in targets:
        _check_existing_path_components(target.parent)
        if target.is_symlink():
            raise ValueError(f"artifact output must not be a symlink: {target}")


def _manifest_map(artifact: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {entry["path"]: entry for entry in artifact["artifact_manifest"]}


def verify_artifact_manifest(
    artifact: Mapping[str, object], files: Mapping[str, str | bytes]
) -> bool:
    """Re-render and verify exact bytes, lengths, and SHA256 values."""

    validate_root_artifact(artifact)
    normalized = _normalized_files(files)
    actual = {path.as_posix(): payload for path, payload in normalized}
    rendered = render_artifact_bundle(artifact)
    if actual != rendered:
        missing = sorted(set(rendered) - set(actual))
        extra = sorted(set(actual) - set(rendered))
        raise ValueError(f"rendered files differ: missing={missing}, extra={extra}")
    manifest = _manifest_map(artifact)
    for path, payload in sorted(actual.items()):
        record = manifest[path]
        expected_digest = hashlib.sha256(payload).hexdigest()
        if record["bytes"] != len(payload) or record["sha256"] != expected_digest:
            raise ValueError(f"artifact manifest mismatch for {path}")
    return True


def publish_artifacts(
    output_root: Path,
    files: Mapping[str, str | bytes],
    *,
    artifact: Mapping[str, object] | None = None,
) -> list[Path]:
    """Stage, verify, and atomically publish new P6-A files.

    Existing paths are never replaced.  If any individual publication fails,
    files already published by this call are removed and the staging directory
    is cleaned in the ``finally`` block.
    """

    if artifact is None and isinstance(files, Mapping) and "schema_version" in files and "artifact_manifest" in files:
        return publish_root_artifact(output_root, files)  # type: ignore[arg-type]
    normalized = _normalized_files(files)
    targets = [output_root.joinpath(*relative.parts) for relative, _ in normalized]
    _validate_output_root(output_root, targets)

    if artifact is not None:
        rendered = render_artifact_bundle(artifact)
        verify_artifact_manifest(artifact, rendered)
        rendered_normalized = _normalized_files(rendered)
        if rendered_normalized != normalized:
            raise ValueError("supplied files do not equal root-payload rendering")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    _check_existing_path_components(output_root.parent)
    stage_root = Path(tempfile.mkdtemp(prefix=".p6a-stage-", dir=output_root.parent))
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for relative, payload in normalized:
            stage_target = stage_root.joinpath(*relative.parts)
            stage_target.parent.mkdir(parents=True, exist_ok=True)
            if stage_target.is_symlink() or not stage_target.parent.is_dir():
                raise ValueError("staging output is not a regular file path")
            stage_target.write_bytes(payload)
            if stage_target.is_symlink() or not stage_target.is_file():
                raise ValueError("staged artifact is not a regular file")
            staged.append((stage_target, output_root.joinpath(*relative.parts)))

        existing = []
        for _, target in staged:
            if target.is_symlink():
                raise ValueError(f"artifact output must not be a symlink: {target}")
            if target.exists():
                existing.append(target)
        if existing:
            raise FileExistsError(f"refusing to overwrite existing artifact: {existing[0]}")

        output_root.mkdir(parents=True, exist_ok=True)
        _check_existing_path_components(output_root)
        for stage_target, target in staged:
            target.parent.mkdir(parents=True, exist_ok=True)
            _check_existing_path_components(target.parent)
            if target.is_symlink() or target.exists():
                raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
            os.replace(stage_target, target)
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"published artifact is not a regular file: {target}")
            published.append(target)
        return [target for _, target in staged]
    except Exception:
        for target in reversed(published):
            if target.is_file() and not target.is_symlink():
                target.unlink()
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def publish_root_artifact(output_root: Path, artifact: Mapping[str, object]) -> list[Path]:
    """Render and publish a complete root artifact as one verified bundle."""

    rendered = render_artifact_bundle(artifact)
    verify_artifact_manifest(artifact, rendered)
    return publish_artifacts(output_root, rendered, artifact=artifact)
