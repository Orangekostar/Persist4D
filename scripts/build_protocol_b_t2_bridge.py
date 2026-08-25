"""Build the exact canonical Protocol-B T2 bridge and prove PB0 parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p6a_protocol import validate_protocol_b_manifest

DEFAULT_PROTOCOL_MANIFEST = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
DEFAULT_T2_AUDIT = PROJECT_ROOT / "artifacts/tmap_root_cause_v2/PROTOCOL_SHIFT_AUDIT.md"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/protocol_bridge"
CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
TRANSITION_FIELDS = ("added", "nonrigid", "removed", "rigid")
T2_AUDIT_BINDING = re.compile(
    r"^- `(?P<reference>repo:[^`]*sequence_database_sliding_2\.yaml)`: "
    r"`(?P<sha256>[0-9a-f]{64})`$",
    re.MULTILINE,
)


class BridgeBuildError(ValueError):
    """Raised when an exact T2 bridge cannot be proven from frozen inputs."""


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise BridgeBuildError(f"required file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BridgeBuildError(f"{name} must be a lowercase hex digest")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeBuildError(f"{name} must be a mapping")
    return value


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BridgeBuildError(f"{name} must be a sequence")
    return value


def _load_json(path: Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeBuildError(f"cannot load {name}: {path}") from error
    return _mapping(value, name=name)


def _load_yaml(path: Path, *, name: str) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BridgeBuildError(f"cannot load {name}: {path}") from error


def _resolve_repo_reference(reference: object, *, name: str) -> Path:
    if not isinstance(reference, str) or not reference.startswith("repo:"):
        raise BridgeBuildError(f"{name} must be a repo: reference")
    relative = Path(reference.removeprefix("repo:"))
    if relative.is_absolute() or ".." in relative.parts:
        raise BridgeBuildError(f"{name} contains an unsafe repository path")
    path = PROJECT_ROOT / relative
    if not path.is_file():
        raise BridgeBuildError(f"{name} does not resolve to a file: {reference}")
    return path


def _repo_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"external:{resolved.name}"
    return f"repo:{relative.as_posix()}"


def _runtime_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def extract_first_transition_target(
    t5_change_target: np.ndarray, scan_point_counts: Sequence[int]
) -> np.ndarray:
    """Extract T2 rows and transition zero from one exact T5 change target."""

    target = np.asarray(t5_change_target)
    if target.ndim != 2 or target.shape[1] != 4:
        raise BridgeBuildError("T5 change target must have four transition columns")
    if len(scan_point_counts) != 5 or any(
        isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0
        for count in scan_point_counts
    ):
        raise BridgeBuildError("scan point counts must contain five positive integers")
    counts = tuple(int(count) for count in scan_point_counts)
    if target.shape[0] != sum(counts):
        raise BridgeBuildError("T5 change target row count differs from scan points")
    extracted = target[: counts[0] + counts[1], 0]
    if not np.issubdtype(extracted.dtype, np.integer) and not np.all(
        np.equal(extracted, np.floor(extracted))
    ):
        raise BridgeBuildError("change target must contain integer labels")
    return extracted.astype(np.int64, copy=True)


def derive_exact_t2_record(
    *,
    master_sequence_id: str,
    canonical_scan_ids: Sequence[str],
    source_record: Mapping[str, object],
    output_change_path: str,
) -> dict[str, object]:
    """Derive one T2 YAML record without substituting scans or transitions."""

    scans = tuple(canonical_scan_ids)
    if len(scans) != 5 or master_sequence_id != "-".join(scans):
        raise BridgeBuildError("canonical scan IDs must exactly equal the T5 master")
    record = _mapping(source_record, name=f"T5 record {master_sequence_id}")
    scene = record.get("scene")
    sub_scenes = record.get("sub_scenes")
    if isinstance(scene, bool) or not isinstance(scene, int):
        raise BridgeBuildError("T5 source record must contain an integer scene")
    if not isinstance(sub_scenes, list) or len(sub_scenes) != 5:
        raise BridgeBuildError("T5 source record must contain five sub-scenes")
    expected = tuple(f"scene{scene:04d}_{int(value):02d}" for value in sub_scenes)
    if expected != scans:
        raise BridgeBuildError("T5 record sub-scenes differ from canonical scans")
    if record.get("type") != "validation":
        raise BridgeBuildError("bridge source record must be validation")
    ambiguities = record.get("ambiguities")
    if not isinstance(ambiguities, list):
        raise BridgeBuildError("T5 source ambiguities must be a list")
    result: dict[str, object] = {
        "ambiguities": ambiguities,
        "filepath": output_change_path,
        "scene": scene,
        "sub_scenes": list(sub_scenes[:2]),
        "type": "validation",
    }
    for field in TRANSITION_FIELDS:
        transitions = record.get(field)
        if not isinstance(transitions, list) or len(transitions) != 4:
            raise BridgeBuildError(f"T5 field {field} must contain four transitions")
        result[field] = transitions[:1]
    return {key: result[key] for key in sorted(result)}


def _metadata_index(records: object) -> tuple[dict[str, Mapping[str, Any]], dict[str, int]]:
    entries = _sequence(records, name="validation scan metadata")
    by_id: dict[str, Mapping[str, Any]] = {}
    indices: dict[str, int] = {}
    for index, value in enumerate(entries):
        record = _mapping(value, name=f"validation scan metadata[{index}]")
        scene = record.get("scene")
        sub_scene = record.get("sub_scene")
        if (
            isinstance(scene, bool)
            or not isinstance(scene, int)
            or isinstance(sub_scene, bool)
            or not isinstance(sub_scene, int)
        ):
            raise BridgeBuildError("validation scan metadata lacks explicit identity")
        scan_id = f"scene{scene:04d}_{sub_scene:02d}"
        if scan_id in by_id:
            raise BridgeBuildError(f"duplicate validation scan metadata: {scan_id}")
        point_count = record.get("file_len")
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count <= 0:
            raise BridgeBuildError(f"validation scan {scan_id} has invalid file_len")
        for field in ("filepath", "instance_gt_filepath"):
            path = record.get(field)
            if not isinstance(path, str) or not (PROJECT_ROOT / path).is_file():
                raise BridgeBuildError(
                    f"validation scan {scan_id} lacks supervised {field}"
                )
        by_id[scan_id] = record
        indices[scan_id] = index
    return by_id, indices


def _scan_arrays(
    scan_id: str,
    record: Mapping[str, Any],
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if scan_id in cache:
        return cache[scan_id]
    points = np.load(PROJECT_ROOT / record["filepath"], allow_pickle=False)
    if points.ndim != 2 or points.shape[1] < 12:
        raise BridgeBuildError(f"processed scan {scan_id} must have at least 12 columns")
    if points.shape[0] != record["file_len"]:
        raise BridgeBuildError(f"processed scan {scan_id} row count differs")
    semantic = points[:, 10].astype(np.int64, copy=True)
    point_instances = points[:, 11].astype(np.int64, copy=True)
    encoded_instances = np.loadtxt(
        PROJECT_ROOT / record["instance_gt_filepath"], dtype=np.int64
    )
    encoded_instances = np.asarray(encoded_instances).reshape(-1)
    normalized_instances = np.where(
        encoded_instances < 0, encoded_instances, encoded_instances % 1000
    )
    if normalized_instances.shape != point_instances.shape or not np.array_equal(
        normalized_instances, point_instances
    ):
        raise BridgeBuildError(f"instance GT differs from processed scan {scan_id}")
    cache[scan_id] = (semantic, normalized_instances.copy())
    return cache[scan_id]


def _target_components(
    scan_ids: Sequence[str],
    scan_metadata: Mapping[str, Mapping[str, Any]],
    change_labels: np.ndarray,
    scan_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    if len(scan_ids) != 2:
        raise BridgeBuildError("T2 target requires exactly two ordered scans")
    semantics = []
    instances = []
    stages = []
    counts = []
    for stage, scan_id in enumerate(scan_ids):
        if scan_id not in scan_metadata:
            raise BridgeBuildError(f"scan metadata is missing {scan_id}")
        semantic, instance = _scan_arrays(scan_id, scan_metadata[scan_id], scan_cache)
        semantics.append(semantic)
        instances.append(instance)
        counts.append(int(semantic.shape[0]))
        stages.append(np.full(semantic.shape[0], stage, dtype=np.int64))
    changes = np.asarray(change_labels).reshape(-1).astype(np.int64, copy=False)
    if changes.shape[0] != sum(counts):
        raise BridgeBuildError("T2 change labels differ from T2 point count")
    return {
        "scan_ids": tuple(scan_ids),
        "point_counts": tuple(counts),
        "semantic_labels": np.concatenate(semantics),
        "instance_gt": np.concatenate(instances),
        "temporal_stages": np.concatenate(stages),
        "change_labels": changes.copy(),
    }


def _record_scan_ids(record: Mapping[str, Any], *, horizon: int) -> tuple[str, ...]:
    scene = record.get("scene")
    sub_scenes = record.get("sub_scenes")
    if isinstance(scene, bool) or not isinstance(scene, int):
        raise BridgeBuildError("sequence record has invalid scene")
    if not isinstance(sub_scenes, list) or len(sub_scenes) != horizon:
        raise BridgeBuildError("sequence record has invalid sub-scenes")
    return tuple(f"scene{scene:04d}_{int(value):02d}" for value in sub_scenes)


def _read_change(path: Path, *, columns: int) -> np.ndarray:
    try:
        value = np.genfromtxt(path, dtype=np.int64)
    except OSError as error:
        raise BridgeBuildError(f"cannot load change target: {path}") from error
    value = np.asarray(value)
    if columns == 4:
        if value.ndim != 2 or value.shape[1] != 4:
            raise BridgeBuildError("T5 change target must have four columns")
        return value
    if value.ndim == 1:
        return value.reshape(-1)
    if value.ndim == 2 and value.shape[1] == 1:
        return value[:, 0]
    raise BridgeBuildError("T2 change target must contain one transition column")


def _source_path(record: Mapping[str, Any], *, name: str) -> Path:
    value = record.get("filepath")
    if not isinstance(value, str) or value in {"", "None"}:
        raise BridgeBuildError(f"{name} has no supervised change target")
    path = PROJECT_ROOT / value
    if not path.is_file():
        raise BridgeBuildError(f"{name} change target is missing: {value}")
    return path


def _official_t2_binding(audit_path: Path) -> tuple[Path, str]:
    try:
        report = audit_path.read_text(encoding="utf-8")
    except OSError as error:
        raise BridgeBuildError("cannot read frozen T2 audit") from error
    match = T2_AUDIT_BINDING.search(report)
    if match is None:
        raise BridgeBuildError("frozen T2 audit lacks the database binding")
    path = _resolve_repo_reference(match.group("reference"), name="official T2 database")
    expected = match.group("sha256")
    if _sha256(path) != expected:
        raise BridgeBuildError("official T2 database differs from frozen audit")
    return path, expected


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise BridgeBuildError(f"refusing symlink output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise BridgeBuildError("bridge inventory must not be empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _change_collection_hash(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _report(
    *, gate: Mapping[str, object], sources: Mapping[str, Mapping[str, object]]
) -> str:
    lines = [
        "# Protocol-B Exact T2 Bridge Build Report",
        "",
        "## Construction",
        "",
        "Each record is the exact canonical first-two-scan prefix of one registered",
        "T5 master. Its target contains only those point rows and transition column 0.",
        "No missing pair is replaced by another pair or by reverse order.",
        "",
        "## Frozen Sources",
        "",
    ]
    for name, descriptor in sources.items():
        lines.append(
            f"- `{name}`: `{descriptor['reference']}` / `{descriptor['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## PB0",
            "",
            f"- Canonical prefixes: `{gate['canonical_prefix_count']}/43`",
            f"- Existing T2 overlaps: `{gate['overlap_count']}/14`",
            f"- Exact overlap parity: `{gate['overlap_parity_count']}/14`",
            f"- Pair substitutions: `{gate['pair_substitution_count']}`",
            f"- Reverse substitutions: `{gate['reverse_pair_substitution_count']}`",
            f"- Future-stage leakage: `{gate['future_stage_leakage_count']}`",
            f"- Validation and supervised: `{gate['validation_supervised_count']}/43`",
            "",
            f"Gate PB0: **{gate['status']}**.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_protocol_b_t2_bridge(
    *,
    protocol_manifest_path: Path = DEFAULT_PROTOCOL_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    source_commit: str,
    t2_audit_path: Path = DEFAULT_T2_AUDIT,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build all 43 bridge records, publish artifacts, and return the manifest."""

    source_commit = _digest(source_commit, name="source commit", length=40)
    protocol_manifest_path = Path(protocol_manifest_path)
    output_root = Path(output_root)
    t2_audit_path = Path(t2_audit_path)
    protocol = _load_json(protocol_manifest_path, name="Protocol-B manifest")
    try:
        validate_protocol_b_manifest(protocol)
    except (TypeError, ValueError) as error:
        raise BridgeBuildError("Protocol-B manifest validation failed") from error
    sources = _mapping(protocol.get("sources"), name="Protocol-B sources")
    t5_descriptor = _mapping(sources.get("sequence_database"), name="T5 source")
    scan_descriptor = _mapping(sources.get("scan_metadata"), name="scan metadata source")
    t5_path = _resolve_repo_reference(t5_descriptor.get("reference"), name="T5 source")
    scan_path = _resolve_repo_reference(
        scan_descriptor.get("reference"), name="scan metadata source"
    )
    for descriptor, path, name in (
        (t5_descriptor, t5_path, "T5 source"),
        (scan_descriptor, scan_path, "scan metadata source"),
    ):
        if descriptor.get("sha256") != _sha256(path):
            raise BridgeBuildError(f"{name} differs from Protocol-B provenance")
    official_t2_path, official_t2_sha = _official_t2_binding(t2_audit_path)

    t5_database = _mapping(_load_yaml(t5_path, name="T5 database"), name="T5 database")
    official_t2 = _mapping(
        _load_yaml(official_t2_path, name="official T2 database"),
        name="official T2 database",
    )
    scan_records, scan_indices = _metadata_index(
        _load_yaml(scan_path, name="validation scan metadata")
    )
    masters = _sequence(protocol.get("masters"), name="Protocol-B masters")
    if len(masters) != 43:
        raise BridgeBuildError("Protocol-B manifest must contain 43 masters")

    bridge_database: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    change_paths: list[Path] = []
    scan_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    overlap_parity = 0
    validation_supervised = 0
    for master_index, value in enumerate(masters):
        master = _mapping(value, name=f"Protocol-B master[{master_index}]")
        master_id = master.get("master_sequence_id")
        if not isinstance(master_id, str) or master_id not in t5_database:
            raise BridgeBuildError("Protocol-B master is absent from source T5 database")
        orders = _mapping(master.get("orders"), name=f"master[{master_index}].orders")
        canonical = _mapping(orders.get("canonical"), name="canonical order")
        canonical_scans = tuple(
            str(item)
            for item in _sequence(canonical.get("visit_order"), name="canonical visits")
        )
        if len(canonical_scans) != 5 or master_id != "-".join(canonical_scans):
            raise BridgeBuildError("canonical order differs from its T5 master")
        prefix = _mapping(
            _mapping(canonical.get("prefixes"), name="canonical prefixes").get("2"),
            name="canonical T2 prefix",
        )
        prefix_scans = tuple(
            str(item) for item in _sequence(prefix.get("scan_ids"), name="T2 scans")
        )
        sequence_id = prefix.get("sequence_id")
        expected_prefix = canonical_scans[:2]
        if prefix_scans != expected_prefix or sequence_id != "-".join(expected_prefix):
            raise BridgeBuildError("T2 prefix is not the exact canonical slice")
        expected_indices = tuple(scan_indices[scan_id] for scan_id in canonical_scans)
        manifest_indices = tuple(
            int(item)
            for item in _sequence(canonical.get("scan_indices"), name="scan indices")
        )
        if expected_indices != manifest_indices:
            raise BridgeBuildError("scan metadata indices differ from Protocol-B")
        source_record = _mapping(t5_database[master_id], name=f"T5 record {master_id}")
        source_scans = _record_scan_ids(source_record, horizon=5)
        if source_scans != canonical_scans:
            raise BridgeBuildError("source T5 record differs from canonical scans")
        counts = tuple(int(scan_records[scan_id]["file_len"]) for scan_id in canonical_scans)
        t5_change = _read_change(_source_path(source_record, name=master_id), columns=4)
        bridge_change = extract_first_transition_target(t5_change, counts)
        change_path = output_root / "bridge_change_gt" / f"{sequence_id}.txt"
        change_bytes = "".join(f"{int(label)}\n" for label in bridge_change).encode("ascii")
        _atomic_write(change_path, change_bytes)
        change_paths.append(change_path)
        runtime_change_path = _runtime_reference(change_path)
        bridge_record = derive_exact_t2_record(
            master_sequence_id=master_id,
            canonical_scan_ids=canonical_scans,
            source_record=source_record,
            output_change_path=runtime_change_path,
        )
        bridge_database[str(sequence_id)] = bridge_record

        derived = _target_components(
            prefix_scans, scan_records, bridge_change, scan_cache
        )
        supervised = all(
            (PROJECT_ROOT / scan_records[scan_id]["filepath"]).is_file()
            and (PROJECT_ROOT / scan_records[scan_id]["instance_gt_filepath"]).is_file()
            for scan_id in prefix_scans
        )
        if source_record.get("type") == "validation" and supervised:
            validation_supervised += 1
        overlap = str(sequence_id) in official_t2
        parity: dict[str, bool | None] = {
            "scan_ids": None,
            "point_counts": None,
            "instance_gt": None,
            "semantic_labels": None,
            "temporal_stages": None,
            "change_labels": None,
        }
        if overlap:
            official_record = _mapping(
                official_t2[str(sequence_id)], name=f"official T2 {sequence_id}"
            )
            official_scans = _record_scan_ids(official_record, horizon=2)
            official_change = _read_change(
                _source_path(official_record, name=f"official T2 {sequence_id}"),
                columns=1,
            )
            official = _target_components(
                official_scans, scan_records, official_change, scan_cache
            )
            parity = {
                "scan_ids": derived["scan_ids"] == official["scan_ids"],
                "point_counts": derived["point_counts"] == official["point_counts"],
                "instance_gt": np.array_equal(
                    derived["instance_gt"], official["instance_gt"]
                ),
                "semantic_labels": np.array_equal(
                    derived["semantic_labels"], official["semantic_labels"]
                ),
                "temporal_stages": np.array_equal(
                    derived["temporal_stages"], official["temporal_stages"]
                ),
                "change_labels": np.array_equal(
                    derived["change_labels"], official["change_labels"]
                ),
            }
            if not all(parity.values()):
                failed = [name for name, passed in parity.items() if not passed]
                raise BridgeBuildError(
                    f"official T2 parity failed for {sequence_id}: {failed}"
                )
            overlap_parity += 1
        rows.append(
            {
                "master_sequence_id": master_id,
                "reference_scene_id": master.get("reference_scene_id"),
                "sequence_id": sequence_id,
                "scan_id_1": prefix_scans[0],
                "scan_id_2": prefix_scans[1],
                "scan_index_1": expected_indices[0],
                "scan_index_2": expected_indices[1],
                "point_count": int(bridge_change.shape[0]),
                "split": source_record.get("type"),
                "exact_ordered_pair": "true",
                "validation_supervised": _bool(supervised),
                "pair_substituted": "false",
                "reverse_pair_substituted": "false",
                "future_stage_leakage": "false",
                "overlap_official_t2": _bool(overlap),
                **{
                    f"parity_{name}": "" if passed is None else _bool(passed)
                    for name, passed in parity.items()
                },
                "bridge_change_sha256": _sha256(change_path),
            }
        )

    overlap_count = sum(row["overlap_official_t2"] == "true" for row in rows)
    gate = {
        "status": "PASS",
        "canonical_prefix_count": len(rows),
        "overlap_count": overlap_count,
        "overlap_parity_count": overlap_parity,
        "pair_substitution_count": 0,
        "reverse_pair_substitution_count": 0,
        "future_stage_leakage_count": 0,
        "validation_supervised_count": validation_supervised,
    }
    if (
        len(rows) != 43
        or overlap_count != 14
        or overlap_parity != 14
        or validation_supervised != 43
    ):
        raise BridgeBuildError(f"PB0 coverage gate failed: {gate}")

    source_bindings: dict[str, dict[str, object]] = {
        "protocol_b_manifest": {
            "reference": _repo_reference(protocol_manifest_path),
            "sha256": _sha256(protocol_manifest_path),
        },
        "source_t5_database": {
            "reference": _repo_reference(t5_path),
            "sha256": _sha256(t5_path),
        },
        "validation_scan_metadata": {
            "reference": _repo_reference(scan_path),
            "sha256": _sha256(scan_path),
        },
        "official_sliding_t2_database": {
            "reference": _repo_reference(official_t2_path),
            "sha256": official_t2_sha,
        },
        "frozen_t2_inventory_audit": {
            "reference": _repo_reference(t2_audit_path),
            "sha256": _sha256(t2_audit_path),
        },
    }
    yaml_path = output_root / "sequence_database_protocol_b_exact_t2.yaml"
    inventory_path = output_root / "bridge_inventory.csv"
    report_path = output_root / "BRIDGE_BUILD_REPORT.md"
    _atomic_write(
        yaml_path,
        yaml.safe_dump(bridge_database, sort_keys=False).encode("utf-8"),
    )
    _atomic_write(inventory_path, _csv_bytes(rows))
    _atomic_write(report_path, _report(gate=gate, sources=source_bindings).encode())

    config_descriptor = _mapping(sources.get("config"), name="Protocol-B config source")
    script_paths = {
        "builder": Path(__file__).resolve(),
        "test": PROJECT_ROOT / "tests/test_protocol_b_t2_bridge.py",
    }
    outputs = {
        path.relative_to(output_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (yaml_path, inventory_path, report_path)
    }
    outputs["bridge_change_gt"] = {
        "file_count": len(change_paths),
        "sha256": _change_collection_hash(change_paths, output_root),
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": generated_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_used": False,
        "config_hash": config_descriptor.get("sha256"),
        "protocol_hash": _sha256(protocol_manifest_path),
        "cache_hash": None,
        "cache_used": False,
        "hardware": {"gpu_inference": False},
        "construction": {
            "order": "canonical",
            "horizon": 2,
            "change_target": (
                "T5 rows for canonical scans 1-2, first transition column only"
            ),
            "pair_substitution": "forbidden",
            "future_stage_access": "forbidden",
        },
        "sources": source_bindings,
        "scripts": {
            name: {"reference": _repo_reference(path), "sha256": _sha256(path)}
            for name, path in script_paths.items()
        },
        "gate_pb0": gate,
        "outputs": outputs,
    }
    _atomic_write(
        output_root / "bridge_manifest.json",
        (json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--t2-audit", type=Path, default=DEFAULT_T2_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_protocol_b_t2_bridge(
        protocol_manifest_path=args.protocol_manifest,
        output_root=args.output_root,
        source_commit=args.source_commit,
        t2_audit_path=args.t2_audit,
    )
    print(json.dumps(manifest["gate_pb0"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
