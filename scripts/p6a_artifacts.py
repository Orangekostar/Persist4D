"""Strict, deterministic artifact helpers for Persist4D P6-A."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
ROOT_KEYS = {
    "schema_version",
    "status",
    "protocol",
    "run_id",
    "source_commit",
    "source_tree_contract",
    "provenance",
    "settings",
    "artifact_manifest",
    "gate_results",
    "claims_supported",
    "claims_not_supported",
    "next_action",
    "errors",
}
PROTOCOL_KEYS = {
    "name",
    "horizons",
    "reference_scene_count",
    "master_sequence_count",
}
PROVENANCE_KEYS = {"checkpoint", "config", "dataset", "prediction_cache"}
PROVENANCE_RECORD_KEYS = {"ref", "sha256"}
GATE_RECORD_KEYS = {"passed", "evidence"}
GATE_IDS = tuple(f"G6A-{index}" for index in range(1, 6))
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
_PRIVATE_TEXT = (
    re.compile(r"/(?:home|Users)/"),
    re.compile(r"(?:^|[^A-Za-z])GPU-[0-9A-Fa-f-]+"),
    re.compile(r"ssh://"),
)


def _exact_keys(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} must be a mapping"
        )
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_list(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(  # noqa: TRY004 - validation contract.
            f"{name} must be a list"
        )
    return [_nonempty_string(item, name=f"{name} item") for item in value]


def _validate_scalar_tree(value: object, *, path: str = "root") -> None:
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _PRIVATE_TEXT):
            raise ValueError(f"{path} contains private or non-portable text")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(  # noqa: TRY004 - validation contract.
                    f"{path} mapping keys must be strings"
                )
            _validate_scalar_tree(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_scalar_tree(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported type {type(value).__name__}")


def _validate_reference(
    value: object, *, name: str, expected_prefix: str | None = None
) -> str:
    reference = _nonempty_string(value, name=name)
    if not reference.startswith(("repo:", "external:", "local_cache:")):
        raise ValueError(f"{name} must use a portable reference prefix")
    if ".." in PurePosixPath(reference.split(":", 1)[1]).parts:
        raise ValueError(f"{name} must not traverse parent directories")
    if expected_prefix is not None and not reference.startswith(expected_prefix):
        raise ValueError(f"{name} must start with {expected_prefix}")
    return reference


def validate_root_artifact(artifact: object) -> None:
    """Validate a complete P6-A root artifact and reject schema drift."""

    root = _exact_keys(artifact, ROOT_KEYS, name="artifact")
    if (
        _integer(root["schema_version"], name="schema_version", minimum=1)
        != SCHEMA_VERSION
    ):
        raise ValueError("unsupported P6-A schema_version")
    if root["status"] != "pass":
        raise ValueError("complete P6-A artifact status must be pass")
    _nonempty_string(root["run_id"], name="run_id")
    source_commit = _nonempty_string(root["source_commit"], name="source_commit")
    if _HEX_40.fullmatch(source_commit) is None:
        raise ValueError("source_commit must be a lowercase 40-character SHA1")

    protocol = _exact_keys(root["protocol"], PROTOCOL_KEYS, name="protocol")
    if protocol["name"] != "exact_common_prefix_protocol_b":
        raise ValueError("protocol name must identify exact common-prefix Protocol B")
    if protocol["horizons"] != [2, 3, 4, 5]:
        raise ValueError("protocol horizons must be exactly [2, 3, 4, 5]")
    _integer(protocol["reference_scene_count"], name="reference_scene_count", minimum=1)
    _integer(protocol["master_sequence_count"], name="master_sequence_count", minimum=1)

    source_tree = root["source_tree_contract"]
    if not isinstance(source_tree, Mapping) or source_tree.get("status") != "pass":
        raise ValueError("source_tree_contract must be a passing mapping")

    provenance = _exact_keys(root["provenance"], PROVENANCE_KEYS, name="provenance")
    expected_reference_prefix = {
        "checkpoint": "repo:checkpoints/",
        "config": "repo:conf/",
        "dataset": "repo:data/",
        "prediction_cache": "repo:artifacts/P6A/",
    }
    for key in sorted(PROVENANCE_KEYS):
        record = _exact_keys(
            provenance[key], PROVENANCE_RECORD_KEYS, name=f"provenance.{key}"
        )
        _validate_reference(
            record["ref"],
            name=f"provenance.{key}.ref",
            expected_prefix=expected_reference_prefix[key],
        )
        digest = _nonempty_string(record["sha256"], name=f"provenance.{key}.sha256")
        if _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"provenance.{key}.sha256 must be lowercase SHA256")

    if not isinstance(root["settings"], Mapping):
        raise ValueError(  # noqa: TRY004 - validation contract.
            "settings must be a mapping"
        )
    if not isinstance(root["artifact_manifest"], list):
        raise ValueError(  # noqa: TRY004 - validation contract.
            "artifact_manifest must be a list"
        )
    gates = _exact_keys(root["gate_results"], set(GATE_IDS), name="gate_results")
    for gate_id in GATE_IDS:
        gate = _exact_keys(gates[gate_id], GATE_RECORD_KEYS, name=gate_id)
        if not isinstance(gate["passed"], bool):
            raise ValueError(  # noqa: TRY004 - validation contract.
                f"{gate_id}.passed must be boolean"
            )
        _nonempty_string(gate["evidence"], name=f"{gate_id}.evidence")
    _string_list(root["claims_supported"], name="claims_supported")
    _string_list(root["claims_not_supported"], name="claims_not_supported")
    _nonempty_string(root["next_action"], name="next_action")
    if not isinstance(root["errors"], list):
        raise ValueError(  # noqa: TRY004 - validation contract.
            "errors must be a list"
        )
    _validate_scalar_tree(root)


def artifact_json_text(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    return (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _markdown_list(values: object) -> str:
    items = _string_list(values, name="report list")
    if not items:
        return "None."
    return "\n".join(f"- {item}" for item in items)


def render_go_nogo_report(artifact: Mapping[str, object]) -> str:
    validate_root_artifact(artifact)
    gates = artifact["gate_results"]
    decision = (
        "P6A_GO" if all(gates[key]["passed"] for key in GATE_IDS) else "P6A_NO_GO"
    )
    gate_lines = "\n".join(
        f"- {key}: {'PASS' if gates[key]['passed'] else 'FAIL'} - {gates[key]['evidence']}"
        for key in GATE_IDS
    )
    protocol = artifact["protocol"]
    sections = {
        "What was changed": "Implemented the P6-A scientific evaluation package.",
        "Why it was changed": "To isolate association quality from frozen local perception.",
        "Experimental protocol": (
            f"Protocol B uses {protocol['master_sequence_count']} master sequences from "
            f"{protocol['reference_scene_count']} reference scenes at T=2/3/4/5."
        ),
        "Reproducibility binding": (
            f"Source commit: `{artifact['source_commit']}`; run: `{artifact['run_id']}`."
        ),
        "Main results": gate_lines,
        "Statistical evidence": "See the bound statistical analysis artifact.",
        "Failure analysis": "See the bound F1-F7 error decomposition artifacts.",
        "What claims are supported": _markdown_list(artifact["claims_supported"]),
        "What claims are NOT supported": _markdown_list(
            artifact["claims_not_supported"]
        ),
        "GO / NO-GO decision": f"Decision: {decision}",
        "Exact next action": f"Exact next action: {artifact['next_action']}",
    }
    body = ["# Persist4D P6-A GO / NO-GO Report"]
    for section in P6A_REPORT_SECTIONS:
        body.extend(("", f"## {section}", "", sections[section]))
    return "\n".join(body) + "\n"


def render_csv(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]) -> str:
    normalized_columns = tuple(columns)
    if not normalized_columns or len(set(normalized_columns)) != len(
        normalized_columns
    ):
        raise ValueError("CSV columns must be non-empty and unique")
    if any(not isinstance(column, str) or not column for column in normalized_columns):
        raise ValueError("CSV columns must be non-empty strings")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=normalized_columns,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    expected = set(normalized_columns)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != expected:
            raise ValueError(f"CSV row {index} does not match the exact columns")
        _validate_scalar_tree(row, path=f"rows[{index}]")
        writer.writerow(row)
    return output.getvalue()


def _relative_output_path(value: object) -> PurePosixPath:
    text = _nonempty_string(value, name="artifact output path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.parts[0] in {"P5", "artifacts"}:
        raise ValueError("artifact output must be relative to artifacts/P6A")
    return path


def publish_artifacts(
    output_root: Path, files: Mapping[str, str | bytes]
) -> list[Path]:
    """Publish new P6-A files with same-filesystem atomic replacements."""

    if not isinstance(output_root, Path):
        raise ValueError(  # noqa: TRY004 - validation contract.
            "output_root must be a Path"
        )
    if not isinstance(files, Mapping) or not files:
        raise ValueError("files must be a non-empty mapping")
    normalized: list[tuple[PurePosixPath, bytes]] = []
    for raw_path, content in files.items():
        relative = _relative_output_path(raw_path)
        if not isinstance(content, (str, bytes)):
            raise ValueError(  # noqa: TRY004 - validation contract.
                "artifact content must be text or bytes"
            )
        payload = content.encode("utf-8") if isinstance(content, str) else content
        normalized.append((relative, payload))
    normalized.sort(key=lambda item: item[0].as_posix())
    if len({path for path, _ in normalized}) != len(normalized):
        raise ValueError("artifact output paths must be unique")

    targets = [
        (output_root.joinpath(*relative.parts), payload)
        for relative, payload in normalized
    ]
    existing = [
        target for target, _ in targets if target.exists() or target.is_symlink()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing artifact: {existing[0]}")

    temporary: list[tuple[Path, Path]] = []
    try:
        for target, payload in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.append((temporary_path, target))
        for temporary_path, target in temporary:
            os.replace(temporary_path, target)
        return [target for target, _ in targets]
    finally:
        for temporary_path, _ in temporary:
            temporary_path.unlink(missing_ok=True)
