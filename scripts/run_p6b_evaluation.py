"""Leakage-free sweep and held-out evaluation entry point for Persist4D P6-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.evaluate_persist4d_p6a import (
    _frozen_protocol_bundle,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
    evaluate_cached_task_metrics,
    load_cached_protocol_sequences,
)
from scripts.p6a_analysis import (
    PairedMetricRecord,
    aggregate_event_metrics,
    aggregate_metrics_by_sequence,
    failure_breakdown,
    paired_cluster_bootstrap,
)
from scripts.p6a_metrics import build_official_metric_population_evidence
from scripts.p6b_artifacts import (
    build_p6b_artifact_root,
    finalize_p6b_artifact,
    publish_p6b_artifact,
)
from scripts.p6b_association import P6BTracker
from scripts.p6b_protocol import (
    build_split_manifest,
    canonical_config_id,
    canonical_config_json,
    load_p6b_config,
)
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BHorizonMetrics,
    P6BSweepError,
    attach_cluster_task_metrics,
    build_candidate_row,
    cached_sequences_to_sweep_sequences,
    candidate_ranking_key,
    candidate_stage_eligible,
    cluster_event_metrics,
    cluster_metrics_from_payload,
    cluster_metrics_to_payload,
    derive_prefix_events,
    extract_official_metrics,
    replay_configuration,
    run_staged_sweep,
    stage_eligibility_policy,
    validate_staged_sweep_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA_VERSION = 2
EXPECTED_CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
EXPECTED_P5_SHA256 = "7da68910b0c0b43b5f04d8ae7d56543a460231c0616c62b2fb9485b88fd781a1"
EXPECTED_P6A_SHA256 = "bffc32fde402396258ed750943101bd8acb6318bc2526ea8f99a9ec42dbe9399"
_HORIZONS = (2, 3, 4, 5)
_MAX_SELECTION_DOCUMENT_BYTES = 8 * 1024 * 1024
_PROTOCOL_PROOFS = (
    (
        "threshold_aware_total_score",
        "tests/test_p6b_memory.py::test_threshold_aware_assignment_maximizes_score_before_cardinality",
    ),
    (
        "gt_free_runtime_api",
        "tests/test_p6b_association.py::test_p6b_tracker_runtime_api_has_no_ground_truth_inputs",
    ),
)
_STAGES = (
    "assignment",
    "reactivation",
    "class_compatibility",
    "consolidation",
    "birth_gate",
    "joint_neighbors",
)
_SWEEP_ROW_KEYS = frozenset(
    {
        "config_id",
        "config_json",
        "stage",
        "T",
        "identity_switches",
        "transition_opportunities",
        "identity_switch_rate",
        "wrong_reactivations",
        "predicted_reactivation_events",
        "correct_reactivations",
        "reactivation_attempts",
        "gap_opportunities",
        "wrong_reactivation_rate",
        "false_births",
        "true_births",
        "births",
        "accepted_births",
        "rejected_births",
        "valid_birth_opportunities",
        "false_birth_rate",
        "cluster_metrics_json",
        "reactivation_accuracy",
        "reactivation_recall",
        "accepted_valid_observations",
        "total_valid_observations",
        "frozen_b4_valid_observations",
        "tuning_population_count",
        "tuning_population_sha256",
        "strict_online_tmap",
        "strict_online_trec",
        "full_eligible",
        "stage_eligible",
        "eligibility_reasons",
    }
)
_SELECTION_PROVENANCE_KEYS = frozenset(
    {
        "p6a_protocol_manifest_sha256",
        "cache_manifest_sha256",
        "split_sha256",
        "p6b_config_sha256",
        "gt_free_inference_test_sha256",
        "gt_free_inference_status",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_commit",
        "source_tree_contract",
        "provenance",
        "split_manifest",
        "selected_config_id",
        "selected_config_sha256",
        "selected_config",
        "ranking_key",
        "baseline",
        "candidate_rows",
        "finalist_rows",
        "selected_by_stage",
        "heldout_evaluated",
        "tuning_population",
        "sequence_metric_evidence",
        "stage_eligibility_policy",
        "verification_ledger",
    }
)
HELDOUT_ATTEMPT_SCHEMA_VERSION = 1
HELDOUT_RAW_SCHEMA_VERSION = 1
_PRIVATE_PATH_MARKERS = ("/home/", "/mnt/", "/Users/")
_HELDOUT_ATTEMPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "attempt_id",
        "source_commit",
        "selection",
        "split_sha256",
        "p6b_config_sha256",
        "command",
        "input_sha256",
        "status",
        "started_utc",
        "ended_utc",
        "exit_status",
        "error_type",
        "events",
        "log_sha256",
        "output",
    }
)
_HELDOUT_RAW_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "attempt_id",
        "source_commit",
        "selection",
        "split_sha256",
        "p6b_config_sha256",
        "evaluation",
    }
)


def _git(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _log_event(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_json_durable(
    path: Path, value: Mapping[str, object], *, replace: bool
) -> bytes:
    destination = Path(path)
    payload = _canonical_json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _json_without_constants(path: Path) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value} is forbidden")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} cannot be decoded") from error


def _validate_hex_digest(value: object, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is invalid")
    return value


def _validate_heldout_inputs(
    *,
    source_commit: object,
    selection: object,
    split_sha256: object,
    p6b_config_sha256: object,
    command: object,
) -> dict[str, object]:
    _validate_hex_digest(source_commit, length=40, name="source_commit")
    if not isinstance(selection, Mapping) or set(selection) != {"ref", "sha256"}:
        raise ValueError("selection input differs from the schema")
    selection_ref = selection["ref"]
    if selection_ref != "repo:artifacts/P6B_selection/selection.json":
        raise ValueError("selection input must use the frozen repository reference")
    _validate_hex_digest(selection["sha256"], length=64, name="selection.sha256")
    _validate_hex_digest(split_sha256, length=64, name="split_sha256")
    _validate_hex_digest(p6b_config_sha256, length=64, name="p6b_config_sha256")
    if (
        isinstance(command, (str, bytes))
        or not isinstance(command, Sequence)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ValueError("command must be a nonempty text sequence")
    result = {
        "source_commit": source_commit,
        "selection": dict(selection),
        "split_sha256": split_sha256,
        "p6b_config_sha256": p6b_config_sha256,
        "command": list(command),
    }
    serialized = json.dumps(result, sort_keys=True, allow_nan=False)
    if any(marker in serialized for marker in _PRIVATE_PATH_MARKERS):
        raise ValueError("held-out attempt inputs contain a private path")
    return result


def _input_sha256(inputs: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(inputs)).hexdigest()


def _event_log_sha256(events: object) -> str:
    if not isinstance(events, list) or not events:
        raise ValueError("held-out event log must be a nonempty list")
    return hashlib.sha256(_canonical_json_bytes(events)).hexdigest()


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC timestamp")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC timestamp") from error
    if result.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be a UTC timestamp")
    return result


def run_exactly_once_heldout(
    attempt_root: Path,
    *,
    evaluator: Callable[[], Mapping[str, object]],
    source_commit: str,
    selection: Mapping[str, object],
    split_sha256: str,
    p6b_config_sha256: str,
    command: Sequence[str],
) -> dict[str, object]:
    root = Path(attempt_root)
    inputs = _validate_heldout_inputs(
        source_commit=source_commit,
        selection=selection,
        split_sha256=split_sha256,
        p6b_config_sha256=p6b_config_sha256,
        command=command,
    )
    if root.exists() or root.is_symlink():
        raise FileExistsError("held-out attempt root already exists")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    _fsync_directory(root.parent)
    started = _utc_now()
    attempt_id = hashlib.sha256(
        _canonical_json_bytes({**inputs, "started_utc": started})
    ).hexdigest()
    attempt: dict[str, object] = {
        "schema_version": HELDOUT_ATTEMPT_SCHEMA_VERSION,
        "protocol_version": 2,
        "attempt_id": attempt_id,
        **inputs,
        "input_sha256": _input_sha256(inputs),
        "status": "in_progress",
        "started_utc": started,
        "ended_utc": None,
        "exit_status": None,
        "error_type": None,
        "events": [
            {
                "event": "attempt_token_published",
                "utc": started,
            }
        ],
        "log_sha256": None,
        "output": None,
    }
    attempt["log_sha256"] = _event_log_sha256(attempt["events"])
    _publish_json_durable(root / "attempt.json", attempt, replace=False)
    raw_published = False
    try:
        evaluation = evaluator()
        if not isinstance(evaluation, Mapping):
            raise TypeError("held-out evaluator must return a mapping")
        evaluation_copy = dict(evaluation)
        serialized = json.dumps(evaluation_copy, sort_keys=True, allow_nan=False)
        if any(marker in serialized for marker in _PRIVATE_PATH_MARKERS):
            raise ValueError("held-out evaluation contains a private path")
        raw = {
            "schema_version": HELDOUT_RAW_SCHEMA_VERSION,
            "protocol_version": 2,
            "attempt_id": attempt_id,
            "source_commit": source_commit,
            "selection": dict(selection),
            "split_sha256": split_sha256,
            "p6b_config_sha256": p6b_config_sha256,
            "evaluation": evaluation_copy,
        }
        raw_payload = _publish_json_durable(
            root / "heldout_raw.json", raw, replace=False
        )
        raw_published = True
        ended = _utc_now()
        raw_sha = hashlib.sha256(raw_payload).hexdigest()
        attempt.update(
            {
                "status": "success",
                "ended_utc": ended,
                "exit_status": 0,
                "events": [
                    *attempt["events"],
                    {"event": "heldout_raw_published", "utc": ended},
                ],
                "output": {
                    "ref": "repo:artifacts/P6B_heldout/heldout_raw.json",
                    "bytes": len(raw_payload),
                    "sha256": raw_sha,
                },
            }
        )
        attempt["log_sha256"] = _event_log_sha256(attempt["events"])
        _publish_json_durable(root / "attempt.json", attempt, replace=True)
        return raw
    except BaseException as error:
        if raw_published:
            raise
        ended = _utc_now()
        attempt.update(
            {
                "status": "failed",
                "ended_utc": ended,
                "exit_status": 1,
                "error_type": type(error).__name__,
                "events": [
                    *attempt["events"],
                    {"event": "heldout_evaluation_failed", "utc": ended},
                ],
            }
        )
        attempt["log_sha256"] = _event_log_sha256(attempt["events"])
        _publish_json_durable(root / "attempt.json", attempt, replace=True)
        raise


def _validate_attempt_input_and_log(attempt: Mapping[str, object]) -> None:
    if set(attempt) != _HELDOUT_ATTEMPT_KEYS:
        raise ValueError("held-out attempt keys differ from the schema")
    if (
        attempt["schema_version"] != HELDOUT_ATTEMPT_SCHEMA_VERSION
        or attempt["protocol_version"] != 2
    ):
        raise ValueError("held-out attempt schema/protocol differs")
    inputs = _validate_heldout_inputs(
        source_commit=attempt["source_commit"],
        selection=attempt["selection"],
        split_sha256=attempt["split_sha256"],
        p6b_config_sha256=attempt["p6b_config_sha256"],
        command=attempt["command"],
    )
    if attempt["input_sha256"] != _input_sha256(inputs):
        raise ValueError("held-out attempt input SHA-256 differs")
    started = _parse_utc(attempt["started_utc"], name="started_utc")
    expected_attempt_id = hashlib.sha256(
        _canonical_json_bytes({**inputs, "started_utc": attempt["started_utc"]})
    ).hexdigest()
    if attempt["attempt_id"] != expected_attempt_id:
        raise ValueError("held-out attempt ID differs from canonical inputs")
    ended_raw = attempt["ended_utc"]
    if ended_raw is not None:
        ended = _parse_utc(ended_raw, name="ended_utc")
        if ended < started:
            raise ValueError("held-out attempt ended before it started")
    if attempt["log_sha256"] != _event_log_sha256(attempt["events"]):
        raise ValueError("held-out attempt event-log SHA-256 differs")


def _validate_raw_against_attempt(
    raw: Mapping[str, object], attempt: Mapping[str, object]
) -> None:
    if set(raw) != _HELDOUT_RAW_KEYS:
        raise ValueError("held-out raw keys differ from the schema")
    if (
        raw["schema_version"] != HELDOUT_RAW_SCHEMA_VERSION
        or raw["protocol_version"] != 2
        or raw["attempt_id"] != attempt.get("attempt_id")
    ):
        raise ValueError("held-out raw protocol identity differs")
    for key in ("source_commit", "selection", "split_sha256", "p6b_config_sha256"):
        if raw[key] != attempt.get(key):
            raise ValueError(f"held-out raw {key} differs from attempt ledger")
    if not isinstance(raw["evaluation"], Mapping):
        raise TypeError("held-out raw evaluation must be a mapping")
    serialized = json.dumps(raw, sort_keys=True, allow_nan=False)
    if any(marker in serialized for marker in _PRIVATE_PATH_MARKERS):
        raise ValueError("held-out raw contains a private path")


def recover_heldout_attempt(attempt_root: Path) -> dict[str, object]:
    root = Path(attempt_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("held-out attempt root must be a regular directory")
    attempt_path = root / "attempt.json"
    raw_path = root / "heldout_raw.json"
    if attempt_path.is_symlink() or not attempt_path.is_file():
        raise ValueError("held-out attempt file must be regular")
    attempt = _json_without_constants(attempt_path)
    if not isinstance(attempt, Mapping):
        raise TypeError("held-out attempt must be a mapping")
    _validate_attempt_input_and_log(attempt)
    if attempt["status"] == "success":
        return _load_successful_heldout_attempt(root)
    if attempt["status"] != "in_progress":
        raise ValueError("only an in-progress held-out attempt can be recovered")
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("in-progress held-out attempt has no complete raw payload")
    raw = _json_without_constants(raw_path)
    if not isinstance(raw, Mapping):
        raise TypeError("held-out raw must be a mapping")
    _validate_raw_against_attempt(raw, attempt)
    raw_bytes = raw_path.read_bytes()
    if raw_bytes != _canonical_json_bytes(raw):
        raise ValueError("held-out raw must use canonical JSON bytes")
    ended = _utc_now()
    recovered = dict(attempt)
    recovered.update(
        {
            "status": "success",
            "ended_utc": ended,
            "exit_status": 0,
            "error_type": None,
            "events": [
                *attempt["events"],
                {"event": "heldout_raw_recovered", "utc": ended},
            ],
            "output": {
                "ref": "repo:artifacts/P6B_heldout/heldout_raw.json",
                "bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            },
        }
    )
    recovered["log_sha256"] = _event_log_sha256(recovered["events"])
    _publish_json_durable(attempt_path, recovered, replace=True)
    return _load_successful_heldout_attempt(root)


def _load_successful_heldout_attempt(attempt_root: Path) -> dict[str, object]:
    root = Path(attempt_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("held-out attempt root must be a regular directory")
    attempt_path = root / "attempt.json"
    raw_path = root / "heldout_raw.json"
    if any(
        path.is_symlink() or not path.is_file() for path in (attempt_path, raw_path)
    ):
        raise ValueError(
            "successful held-out attempt requires regular attempt and raw files"
        )
    attempt = _json_without_constants(attempt_path)
    raw = _json_without_constants(raw_path)
    if not isinstance(attempt, Mapping) or not isinstance(raw, Mapping):
        raise TypeError("held-out attempt documents must be mappings")
    _validate_attempt_input_and_log(attempt)
    if attempt.get("status") != "success" or attempt.get("exit_status") != 0:
        raise ValueError("held-out attempt is not successful")
    output = attempt.get("output")
    if not isinstance(output, Mapping) or set(output) != {"ref", "bytes", "sha256"}:
        raise ValueError("held-out attempt output binding is invalid")
    if output["ref"] != "repo:artifacts/P6B_heldout/heldout_raw.json":
        raise ValueError("held-out raw reference is invalid")
    raw_bytes = raw_path.read_bytes()
    if (
        output["bytes"] != len(raw_bytes)
        or output["sha256"] != hashlib.sha256(raw_bytes).hexdigest()
    ):
        raise ValueError("held-out raw SHA-256/byte binding differs")
    _validate_raw_against_attempt(raw, attempt)
    return dict(raw)


def build_source_tree_contract(
    repo_root: Path = PROJECT_ROOT,
    *,
    allowed_dirty_prefixes: Sequence[str] = (),
) -> dict[str, str]:
    root = Path(repo_root).resolve()
    if any(
        not isinstance(prefix, str)
        or not prefix
        or prefix.startswith(("/", "../"))
        or ".." in Path(prefix).parts
        or not prefix.endswith("/")
        for prefix in allowed_dirty_prefixes
    ):
        raise ValueError("allowed dirty prefixes must be safe repository directories")
    commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    dirty_paths = []
    for line in status.splitlines():
        raw_path = line[3:]
        dirty_paths.extend(raw_path.split(" -> "))
    unexpected_dirty = [
        path
        for path in dirty_paths
        if not any(path.startswith(prefix) for prefix in allowed_dirty_prefixes)
    ]
    hidden = [
        line
        for line in _git(root, "ls-files", "-v").splitlines()
        if line and (line[0].islower() or line[0] == "S")
    ]
    if unexpected_dirty or hidden:
        raise ValueError("P6-B requires a clean source tree without hidden index flags")
    return {"status": "pass", "source_commit": commit}


def partition_cached_sequences(
    sequences: Sequence[object], split_manifest: Mapping[str, object]
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a sequence")
    tuning_references = set(split_manifest["tuning_reference_scene_ids"])
    heldout_references = set(split_manifest["heldout_reference_scene_ids"])
    tuning_masters = set(split_manifest["tuning_master_sequence_ids"])
    heldout_masters = set(split_manifest["heldout_master_sequence_ids"])
    if tuning_references & heldout_references or tuning_masters & heldout_masters:
        raise ValueError("registered split contains overlap")
    tuning = []
    heldout = []
    for sequence in sequences:
        reference = getattr(sequence, "reference_scene_id", None)
        master = getattr(sequence, "master_sequence_id", None)
        if reference in tuning_references and master in tuning_masters:
            tuning.append(sequence)
        elif reference in heldout_references and master in heldout_masters:
            heldout.append(sequence)
        else:
            raise ValueError("sequence does not belong to the registered split")
    if not tuning or not heldout:
        raise ValueError("registered split must yield tuning and heldout sequences")
    return tuple(tuning), tuple(heldout)


def _config_sha256(config: P6BMemoryConfig) -> str:
    return hashlib.sha256(canonical_config_json(config).encode("ascii")).hexdigest()


def _selection_protocol() -> tuple[object, Mapping[str, object]]:
    config_path = PROJECT_ROOT / "conf/p6b/default.yaml"
    protocol = load_p6b_config(config_path)
    manifest_path = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen P6-A protocol manifest cannot be decoded") from error
    if not isinstance(manifest, Mapping):
        raise TypeError("frozen P6-A protocol manifest must be a mapping")
    split = build_split_manifest(manifest, seed=protocol.seed)
    return protocol, split.to_mapping()


def _population_evidence(
    records: Sequence[tuple[str, str, str]], *, partition: str
) -> dict[str, object]:
    if partition not in {"tuning", "heldout"}:
        raise ValueError("population partition must be tuning or heldout")
    ordered = tuple(sorted(records))
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("population identities must be nonempty and unique")

    def digest(items: Sequence[tuple[str, str, str]]) -> str:
        payload = [
            {
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
            }
            for reference, master, order in items
        ]
        return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()

    references = sorted({record[0] for record in ordered})
    return {
        "partition": partition,
        "count": len(ordered),
        "sha256": digest(ordered),
        "by_reference": [
            {
                "reference_scene_id": reference,
                "count": len(scoped),
                "sha256": digest(scoped),
            }
            for reference in references
            for scoped in [tuple(item for item in ordered if item[0] == reference)]
        ],
    }


def _split_population_evidence(
    split_manifest: Mapping[str, object], *, partition: str
) -> dict[str, object]:
    assignments = split_manifest.get("assignments")
    if isinstance(assignments, (str, bytes)) or not isinstance(assignments, Sequence):
        raise TypeError("split assignments must be a sequence")
    records = []
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise TypeError("split assignment must be a mapping")
        if assignment.get("partition") != partition:
            continue
        orders = assignment.get("order_ids")
        if isinstance(orders, (str, bytes)) or not isinstance(orders, Sequence):
            raise TypeError("split order_ids must be a sequence")
        records.extend(
            (
                str(assignment["reference_scene_id"]),
                str(assignment["master_sequence_id"]),
                str(order),
            )
            for order in orders
        )
    return _population_evidence(records, partition=partition)


def _sequence_population_evidence(
    sequences: Sequence[object], *, partition: str
) -> dict[str, object]:
    return _population_evidence(
        [
            (
                str(sequence.reference_scene_id),
                str(sequence.master_sequence_id),
                str(sequence.order_id),
            )
            for sequence in sequences
        ],
        partition=partition,
    )


def _expected_selection_provenance(
    protocol: object, split_manifest: Mapping[str, object]
) -> dict[str, object]:
    sources = getattr(protocol, "sources", None)
    if sources is None:
        raise TypeError("protocol must expose frozen sources")
    return {
        "p6a_protocol_manifest_sha256": sources.p6a_protocol_manifest.sha256,
        "cache_manifest_sha256": sources.p6a_cache_manifest.sha256,
        "split_sha256": split_manifest["sha256"],
        "p6b_config_sha256": _sha256_file(PROJECT_ROOT / "conf/p6b/default.yaml"),
        "gt_free_inference_test_sha256": _sha256_file(
            PROJECT_ROOT / "tests/test_p6b_association.py"
        ),
        "gt_free_inference_status": "verified_by_protocol_ledger",
    }


def _verification_ledger_from_outputs(
    outputs: Mapping[str, tuple[int, bytes]],
) -> dict[str, object]:
    if set(outputs) != {proof for proof, _ in _PROTOCOL_PROOFS}:
        raise ValueError("protocol proof outputs differ from the registered proof set")
    proofs = []
    for proof, nodeid in _PROTOCOL_PROOFS:
        exit_status, output = outputs[proof]
        test_path = nodeid.split("::", maxsplit=1)[0]
        proofs.append(
            {
                "proof": proof,
                "command": ["python", "-m", "pytest", "-q", nodeid],
                "exit_status": exit_status,
                "output_sha256": hashlib.sha256(output).hexdigest(),
                "test_source": {
                    "ref": f"repo:{test_path}",
                    "sha256": _sha256_file(PROJECT_ROOT / test_path),
                },
            }
        )
    return {"schema_version": 1, "proofs": proofs}


def _validate_verification_ledger(value: object) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "proofs"}:
        raise ValueError("verification ledger differs from the strict schema")
    if value["schema_version"] != 1:
        raise ValueError("verification ledger schema differs")
    proofs = value["proofs"]
    if isinstance(proofs, (str, bytes)) or not isinstance(proofs, Sequence):
        raise TypeError("verification proofs must be a sequence")
    if len(proofs) != len(_PROTOCOL_PROOFS):
        raise ValueError("verification proof population differs")
    for record, (proof, nodeid) in zip(proofs, _PROTOCOL_PROOFS, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "proof",
            "command",
            "exit_status",
            "output_sha256",
            "test_source",
        }:
            raise ValueError("verification proof record differs from the schema")
        test_path = nodeid.split("::", maxsplit=1)[0]
        if (
            record["proof"] != proof
            or record["command"] != ["python", "-m", "pytest", "-q", nodeid]
            or record["exit_status"] != 0
            or record["test_source"]
            != {
                "ref": f"repo:{test_path}",
                "sha256": _sha256_file(PROJECT_ROOT / test_path),
            }
        ):
            raise ValueError("verification proof did not pass the registered command")
        output_sha = record["output_sha256"]
        if (
            not isinstance(output_sha, str)
            or len(output_sha) != 64
            or any(character not in "0123456789abcdef" for character in output_sha)
        ):
            raise ValueError("verification proof output SHA-256 is invalid")
    return True


def _run_protocol_verification_ledger() -> dict[str, object]:
    outputs = {}
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    for proof, nodeid in _PROTOCOL_PROOFS:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", nodeid],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        outputs[proof] = (completed.returncode, completed.stdout)
    ledger = _verification_ledger_from_outputs(outputs)
    _validate_verification_ledger(ledger)
    return ledger


def _same_optional_rate(actual: object, expected: float | None, *, name: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{name} must be null when its denominator is zero")
        return
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError(f"{name} must be a finite numeric rate")
    number = float(actual)
    if not math.isfinite(number) or not math.isclose(
        number, expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{name} differs from its count-derived rate")


def _parse_candidate_rows(
    rows: object,
    *,
    official_metrics_required: bool,
    expected_population: Mapping[str, object],
    sequence_metric_evidence: Mapping[str, Mapping[str, object]],
    used_sequence_metric_evidence: set[str],
    allowed_stages: Sequence[str] = _STAGES,
) -> tuple[P6BCandidateRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("candidate rows must be a nonempty sequence")
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    expected_cluster_population = {
        item["reference_scene_id"]: {
            "count": item["count"],
            "sha256": item["sha256"],
        }
        for item in expected_population["by_reference"]
    }
    order: list[tuple[str, str]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _SWEEP_ROW_KEYS:
            raise ValueError(f"candidate row {index} differs from the strict schema")
        if (
            raw_row["tuning_population_count"] != expected_population["count"]
            or raw_row["tuning_population_sha256"] != expected_population["sha256"]
        ):
            raise ValueError("candidate row differs from the exact tuning population")
        stage = raw_row["stage"]
        config_id = raw_row["config_id"]
        if (
            stage not in allowed_stages
            or not isinstance(config_id, str)
            or not config_id
        ):
            raise ValueError("candidate row stage/config identity is invalid")
        key = (stage, config_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(raw_row)

    result = []
    for stage, config_id in order:
        horizon_rows = groups[(stage, config_id)]
        if [row["T"] for row in horizon_rows] != [f"T{value}" for value in _HORIZONS]:
            raise ValueError("candidate config must contain ordered exact T2-T5 rows")
        config_json = horizon_rows[0]["config_json"]
        if not isinstance(config_json, str) or any(
            row["config_json"] != config_json for row in horizon_rows
        ):
            raise ValueError("candidate config JSON differs across horizons")
        try:
            config_mapping = json.loads(config_json)
            config = P6BMemoryConfig(**config_mapping)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError("candidate config JSON is invalid") from error
        if config_json != canonical_config_json(config):
            raise ValueError("candidate config JSON is not canonical")
        if config_id != canonical_config_id(config):
            raise ValueError("candidate config ID differs from canonical config")

        metrics = []
        reasons: tuple[str, ...] | None = None
        for horizon, row in zip(_HORIZONS, horizon_rows, strict=True):
            try:
                cluster_payload = json.loads(str(row["cluster_metrics_json"]))
                if not isinstance(cluster_payload, list):
                    raise TypeError
                clusters = tuple(
                    cluster_metrics_from_payload(
                        _restore_cluster_sequence_metric_evidence(
                            item,
                            sequence_metric_evidence=sequence_metric_evidence,
                            used=used_sequence_metric_evidence,
                        )
                    )
                    for item in cluster_payload
                )
            except P6BSweepError as error:
                raise ValueError(
                    "candidate tuning cluster metrics differ from tuning population"
                ) from error
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError("candidate horizon metrics are invalid") from error
            actual_cluster_population = {
                cluster.reference_scene_id: {
                    "count": cluster.sequence_population_count,
                    "sha256": cluster.sequence_population_sha256,
                }
                for cluster in clusters
            }
            if actual_cluster_population != expected_cluster_population:
                raise ValueError(
                    "candidate tuning cluster evidence differs from the exact tuning population"
                )
            try:
                metric = P6BHorizonMetrics(
                    horizon=horizon,
                    identity_switches=row["identity_switches"],
                    transition_opportunities=row["transition_opportunities"],
                    wrong_reactivations=row["wrong_reactivations"],
                    predicted_reactivation_events=row["predicted_reactivation_events"],
                    correct_reactivations=row["correct_reactivations"],
                    reactivation_attempts=row["reactivation_attempts"],
                    gap_opportunities=row["gap_opportunities"],
                    false_births=row["false_births"],
                    births=row["births"],
                    rejected_births=row["rejected_births"],
                    reactivation_accuracy=row["reactivation_accuracy"],
                    reactivation_recall=row["reactivation_recall"],
                    accepted_valid_observations=row["accepted_valid_observations"],
                    total_valid_observations=row["total_valid_observations"],
                    cluster_metrics=clusters,
                    strict_online_tmap=row["strict_online_tmap"],
                    strict_online_trec=row["strict_online_trec"],
                )
            except (TypeError, ValueError, KeyError) as error:
                raise ValueError("candidate horizon metrics are invalid") from error
            _same_optional_rate(
                row["identity_switch_rate"],
                metric.identity_switch_rate,
                name="identity_switch_rate",
            )
            _same_optional_rate(
                row["wrong_reactivation_rate"],
                metric.wrong_reactivation_rate,
                name="wrong_reactivation_rate",
            )
            _same_optional_rate(
                row["false_birth_rate"],
                metric.false_birth_rate,
                name="false_birth_rate",
            )
            for name in (
                "true_births",
                "accepted_births",
                "valid_birth_opportunities",
                "frozen_b4_valid_observations",
            ):
                if row[name] != getattr(metric, name):
                    raise ValueError(f"{name} differs from count-derived value")
            if (official_metrics_required and not metric.official_metrics_complete) or (
                not official_metrics_required and not metric.official_metrics_absent
            ):
                raise ValueError("candidate official metric completeness differs")
            raw_reasons = row["eligibility_reasons"]
            if not isinstance(raw_reasons, str):
                raise TypeError("candidate eligibility reasons must be text")
            row_reasons = tuple(item for item in raw_reasons.split(";") if item)
            if reasons is None:
                reasons = row_reasons
            elif reasons != row_reasons:
                raise ValueError("candidate eligibility differs across horizons")
            if not isinstance(row["full_eligible"], bool) or row[
                "full_eligible"
            ] != (not row_reasons):
                raise ValueError("candidate full eligibility differs from reasons")
            deferred = set(stage_eligibility_policy().get(stage, ()))
            if not isinstance(row["stage_eligible"], bool) or row[
                "stage_eligible"
            ] != (set(row_reasons) <= deferred):
                raise ValueError("candidate stage eligibility differs from policy")
            metrics.append(metric)
        result.append(
            P6BCandidateRow(
                config=config,
                stage=stage,
                horizons=tuple(metrics),
                eligibility_reasons=() if reasons is None else reasons,
            )
        )
    return tuple(result)


def _deduplicate_cluster_sequence_metric_evidence(
    rows: Sequence[Mapping[str, object]],
    *,
    registry: dict[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        current = dict(row)
        try:
            clusters = json.loads(str(current["cluster_metrics_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("candidate cluster metrics cannot be normalized") from error
        if not isinstance(clusters, list):
            raise TypeError("candidate cluster metrics cannot be normalized")
        compact_clusters = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                raise TypeError("candidate cluster metrics cannot be normalized")
            compact = dict(cluster)
            evidence = compact.pop("sequence_metrics_evidence", None)
            if not isinstance(evidence, Mapping):
                raise TypeError("candidate sequence metric evidence is missing")
            digest = evidence.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("candidate sequence metric evidence SHA-256 is invalid")
            previous = registry.setdefault(digest, dict(evidence))
            if dict(previous) != dict(evidence):
                raise ValueError("candidate sequence metric evidence digest collides")
            compact["sequence_metrics_evidence_sha256"] = digest
            compact_clusters.append(compact)
        current["cluster_metrics_json"] = json.dumps(
            compact_clusters,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        normalized.append(current)
    return normalized


def _validate_sequence_metric_evidence_registry(
    value: object,
) -> dict[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {"schema_version", "records"}:
        raise ValueError("sequence metric evidence registry differs from schema")
    records = value["records"]
    if value["schema_version"] != 1 or not isinstance(records, list) or not records:
        raise ValueError("sequence metric evidence registry population is invalid")
    result: dict[str, Mapping[str, object]] = {}
    order = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"sha256", "evidence"}:
            raise ValueError("sequence metric evidence registry record differs")
        digest = record["sha256"]
        evidence = record["evidence"]
        if (
            not isinstance(digest, str)
            or not isinstance(evidence, Mapping)
            or evidence.get("sha256") != digest
            or digest in result
        ):
            raise ValueError("sequence metric evidence registry binding differs")
        order.append(digest)
        result[digest] = evidence
    if order != sorted(order):
        raise ValueError("sequence metric evidence registry order differs")
    return result


def _restore_cluster_sequence_metric_evidence(
    cluster: object,
    *,
    sequence_metric_evidence: Mapping[str, Mapping[str, object]],
    used: set[str],
) -> dict[str, object]:
    if not isinstance(cluster, Mapping):
        raise P6BSweepError("sequence metric evidence cluster record is invalid")
    restored = dict(cluster)
    digest = restored.pop("sequence_metrics_evidence_sha256", None)
    if not isinstance(digest, str) or digest not in sequence_metric_evidence:
        raise P6BSweepError("sequence metric evidence reference is invalid")
    restored["sequence_metrics_evidence"] = sequence_metric_evidence[digest]
    used.add(digest)
    return restored


def _validate_tuning_cluster_membership(
    rows: Sequence[P6BCandidateRow],
    *,
    tuning_reference_scene_ids: Sequence[str],
) -> None:
    expected = tuple(sorted(tuning_reference_scene_ids))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("frozen tuning cluster membership is invalid")
    for row in rows:
        for metric in row.horizons:
            actual = tuple(
                cluster.reference_scene_id for cluster in metric.cluster_metrics
            )
            if actual != expected:
                raise ValueError(
                    "candidate metrics must contain exact tuning cluster membership"
                )


def _validate_frozen_valid_observation_denominators(
    rows: Sequence[P6BCandidateRow], *, baseline: P6BCandidateRow
) -> None:
    expected = {
        horizon: baseline.metric(horizon).total_valid_observations
        for horizon in _HORIZONS
    }
    if any(value <= 0 for value in expected.values()):
        raise ValueError("frozen valid observation denominators must be positive")
    for row in rows:
        for horizon in _HORIZONS:
            if row.metric(horizon).total_valid_observations != expected[horizon]:
                raise ValueError(
                    "candidate total differs from frozen valid observation denominator"
                )


def build_selection_document(
    *,
    source_commit: str,
    split_manifest: Mapping[str, object],
    selected_config: P6BMemoryConfig,
    ranking_key: Sequence[object],
    baseline: Mapping[str, object],
    candidate_rows: Sequence[Mapping[str, object]],
    finalist_rows: Sequence[Mapping[str, object]],
    selected_by_stage: Mapping[str, object],
    provenance: Mapping[str, object],
    verification_ledger: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(selected_config, P6BMemoryConfig):
        raise TypeError("selected_config must be P6BMemoryConfig")
    tuning_population = _split_population_evidence(split_manifest, partition="tuning")
    registry: dict[str, Mapping[str, object]] = {}
    baseline_rows = _deduplicate_cluster_sequence_metric_evidence(
        baseline["rows"], registry=registry
    )
    normalized_candidates = _deduplicate_cluster_sequence_metric_evidence(
        candidate_rows, registry=registry
    )
    normalized_finalists = _deduplicate_cluster_sequence_metric_evidence(
        finalist_rows, registry=registry
    )
    document = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "pass",
        "source_commit": source_commit,
        "source_tree_contract": {"status": "pass", "source_commit": source_commit},
        "provenance": dict(provenance),
        "split_manifest": dict(split_manifest),
        "selected_config_id": canonical_config_id(selected_config),
        "selected_config_sha256": _config_sha256(selected_config),
        "selected_config": asdict(selected_config),
        "ranking_key": list(ranking_key),
        "baseline": {"rows": baseline_rows},
        "candidate_rows": normalized_candidates,
        "finalist_rows": normalized_finalists,
        "selected_by_stage": dict(selected_by_stage),
        "heldout_evaluated": False,
        "tuning_population": tuning_population,
        "sequence_metric_evidence": {
            "schema_version": 1,
            "records": [
                {"sha256": digest, "evidence": registry[digest]}
                for digest in sorted(registry)
            ],
        },
        "stage_eligibility_policy": stage_eligibility_policy(),
        "verification_ledger": dict(verification_ledger),
    }
    _validate_selection_document(document)
    return document


def _validate_selection_document(document: Mapping[str, object]) -> None:
    if set(document) != _SELECTION_KEYS:
        raise ValueError("selection document keys differ from schema")
    if (
        document["schema_version"] != SELECTION_SCHEMA_VERSION
        or document["status"] != "pass"
    ):
        raise ValueError("selection document schema/status is invalid")
    if document["heldout_evaluated"] is not False:
        raise ValueError("selection document must prove heldout was not evaluated")
    config_raw = document["selected_config"]
    if not isinstance(config_raw, Mapping):
        raise TypeError("selected_config must be a mapping")
    try:
        config = P6BMemoryConfig(**dict(config_raw))
    except (TypeError, ValueError) as error:
        raise ValueError("selected_config is invalid") from error
    if document["selected_config_id"] != canonical_config_id(config):
        raise ValueError("selected config ID differs from canonical config")
    if document["selected_config_sha256"] != _config_sha256(config):
        raise ValueError("selected config SHA differs from canonical config")
    source_commit = document["source_commit"]
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("selection source_commit is invalid")
    if document["source_tree_contract"] != {
        "status": "pass",
        "source_commit": source_commit,
    }:
        raise ValueError("selection source contract differs")
    split = document["split_manifest"]
    if not isinstance(split, Mapping):
        raise TypeError("selection split_manifest must be a mapping")
    protocol, expected_split = _selection_protocol()
    if dict(split) != dict(expected_split):
        raise ValueError("selection split differs from the frozen Protocol-B split")
    expected_population = _split_population_evidence(expected_split, partition="tuning")
    if document["tuning_population"] != expected_population:
        raise ValueError("selection tuning population differs from the frozen split")
    if document["stage_eligibility_policy"] != stage_eligibility_policy():
        raise ValueError("selection staged eligibility policy differs")
    _validate_verification_ledger(document["verification_ledger"])
    provenance = document["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _SELECTION_PROVENANCE_KEYS
    ):
        raise ValueError("selection provenance differs from the strict schema")
    if dict(provenance) != _expected_selection_provenance(protocol, expected_split):
        raise ValueError("selection provenance differs from frozen inputs")
    sequence_metric_evidence = _validate_sequence_metric_evidence_registry(
        document["sequence_metric_evidence"]
    )
    used_sequence_metric_evidence: set[str] = set()

    baseline_raw = document["baseline"]
    if not isinstance(baseline_raw, Mapping) or set(baseline_raw) != {"rows"}:
        raise ValueError("selection baseline differs from the strict schema")
    baseline_rows = _parse_candidate_rows(
        baseline_raw["rows"],
        official_metrics_required=True,
        expected_population=expected_population,
        sequence_metric_evidence=sequence_metric_evidence,
        used_sequence_metric_evidence=used_sequence_metric_evidence,
        allowed_stages=("baseline",),
    )
    if len(baseline_rows) != 1 or baseline_rows[0].stage != "baseline":
        raise ValueError("selection must contain one frozen baseline candidate")
    candidates = _parse_candidate_rows(
        document["candidate_rows"],
        official_metrics_required=False,
        expected_population=expected_population,
        sequence_metric_evidence=sequence_metric_evidence,
        used_sequence_metric_evidence=used_sequence_metric_evidence,
    )
    finalists = _parse_candidate_rows(
        document["finalist_rows"],
        official_metrics_required=True,
        expected_population=expected_population,
        sequence_metric_evidence=sequence_metric_evidence,
        used_sequence_metric_evidence=used_sequence_metric_evidence,
    )
    if used_sequence_metric_evidence != set(sequence_metric_evidence):
        raise ValueError("sequence metric evidence registry contains unused records")
    tuning_references = expected_split["tuning_reference_scene_ids"]
    if isinstance(tuning_references, (str, bytes)) or not isinstance(
        tuning_references, Sequence
    ):
        raise TypeError("selection tuning cluster membership must be a sequence")
    _validate_tuning_cluster_membership(
        (*baseline_rows, *candidates, *finalists),
        tuning_reference_scene_ids=tuning_references,
    )
    _validate_frozen_valid_observation_denominators(
        (*candidates, *finalists), baseline=baseline_rows[0]
    )
    selected_ids = document["selected_by_stage"]
    if not isinstance(selected_ids, Mapping) or set(selected_ids) != set(_STAGES):
        raise ValueError("selected_by_stage differs from the strict stage order")
    selected_rows: dict[str, P6BCandidateRow] = {}
    for stage in _STAGES:
        config_id = selected_ids[stage]
        matches = [
            row
            for row in finalists
            if row.stage == stage and row.config_id == config_id
        ]
        if len(matches) != 1:
            raise ValueError(f"{stage} selected config does not identify one finalist")
        selected_rows[stage] = matches[0]
    final_selected = selected_rows[_STAGES[-1]]
    if final_selected.config != config:
        raise ValueError("selected config differs from the final stage winner")
    ranking_key = document["ranking_key"]
    if isinstance(ranking_key, (str, bytes)) or not isinstance(ranking_key, Sequence):
        raise TypeError("selection ranking_key must be a sequence")
    validate_staged_sweep_evidence(
        protocol,
        baseline=baseline_rows[0],
        candidate_rows=candidates,
        finalist_rows=finalists,
        selected_by_stage=selected_rows,
        selected=final_selected,
        ranking_key=ranking_key,
    )
    serialized = json.dumps(document, allow_nan=False, sort_keys=True)
    if any(marker in serialized for marker in ("/home/", "/mnt/", "/Users/")):
        raise ValueError("selection document contains a private path")


def load_selection_document(path: Path) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("selection document must be a regular file")
    try:
        if source.stat().st_size > _MAX_SELECTION_DOCUMENT_BYTES:
            raise ValueError("selection document exceeds source-sized budget")
        document = json.loads(source.read_text(encoding="utf-8"))
    except ValueError as error:
        if "exceeds source-sized budget" in str(error):
            raise
        raise ValueError("selection document cannot be decoded") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("selection document cannot be decoded") from error
    if not isinstance(document, Mapping):
        raise TypeError("selection document root must be a mapping")
    result = dict(document)
    _validate_selection_document(result)
    return result


def _method_metrics(
    rows: Sequence[Mapping[str, object]], method: str
) -> dict[int, Mapping[str, object]]:
    result = {}
    for row in rows:
        if row.get("method") != method:
            continue
        token = row.get("T")
        if not isinstance(token, str) or not token.startswith("T"):
            raise ValueError("final metric horizon is invalid")
        horizon = int(token[1:])
        if horizon in result:
            raise ValueError("final metrics contain duplicate method/horizon")
        result[horizon] = row
    if set(result) != set(_HORIZONS):
        raise ValueError(f"final metrics omit {method} T2-T5 rows")
    return result


def _mean(values: Sequence[object]) -> float:
    numbers = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("gate metric must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("gate metric must be finite")
        numbers.append(number)
    return sum(numbers) / len(numbers)


def compute_final_gate_results(
    rows: Sequence[Mapping[str, object]],
    *,
    evidence_complete: bool,
    frozen_hashes_unchanged: bool,
    verification_proofs_passed: bool,
    statistical_analysis: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    baseline = _method_metrics(rows, "B4")
    candidate = _method_metrics(rows, "P6B")

    statistics: dict[tuple[str, int], Mapping[str, object]] = {}
    if statistical_analysis is not None:
        for record in statistical_analysis:
            if not isinstance(record, Mapping):
                raise TypeError("statistical_analysis records must be mappings")
            metric = record.get("metric")
            horizon_token = record.get("T")
            if (
                not isinstance(metric, str)
                or not isinstance(horizon_token, str)
                or horizon_token not in {f"T{value}" for value in _HORIZONS}
            ):
                raise ValueError("statistical_analysis identity is invalid")
            key = (metric, int(horizon_token[1:]))
            if key in statistics:
                raise ValueError("statistical_analysis contains duplicate records")
            statistics[key] = record

    def paired_means(metric: str, horizons: Sequence[int]) -> tuple[float, float]:
        if not statistics:
            return (
                _mean([baseline[h][metric] for h in horizons]),
                _mean([candidate[h][metric] for h in horizons]),
            )
        selected = []
        for horizon in horizons:
            key = (metric, horizon)
            if key not in statistics:
                raise ValueError(f"statistical_analysis omits {metric} T{horizon}")
            selected.append(statistics[key])
        return (
            _mean([record["baseline_mean"] for record in selected]),
            _mean([record["method_mean"] for record in selected]),
        )

    b45, p45 = paired_means("identity_switch_rate", (4, 5))
    relative = (b45 - p45) / b45 if b45 > 0 else 0.0
    if statistics:
        g2_nonregression = all(
            statistics[("identity_switch_rate", horizon)]["method_mean"]
            <= statistics[("identity_switch_rate", horizon)]["baseline_mean"]
            for horizon in (4, 5)
        )
    else:
        g2_nonregression = all(
            candidate[h]["identity_switch_rate"] <= baseline[h]["identity_switch_rate"]
            for h in (4, 5)
        )
    g2 = relative >= 0.10 and g2_nonregression
    b_accuracy, p_accuracy = paired_means("reactivation_accuracy", (3, 4, 5))
    b_recall, p_recall = paired_means("reactivation_recall", (3, 4, 5))
    g3 = p_accuracy >= 0.70 and p_accuracy >= b_accuracy and p_recall >= b_recall - 0.05
    if statistics:
        g4_t2 = all(
            statistics[(metric, 2)]["method_mean"]
            >= statistics[(metric, 2)]["baseline_mean"] - 0.02
            for metric in ("t_mAP", "t_REC")
        )
        b_task_values = []
        p_task_values = []
        for metric in ("t_mAP", "t_REC"):
            baseline_metric, method_metric = paired_means(metric, (4, 5))
            b_task_values.append(baseline_metric)
            p_task_values.append(method_metric)
        b_task = _mean(b_task_values)
        p_task = _mean(p_task_values)
    else:
        g4_t2 = all(
            candidate[2][metric] >= baseline[2][metric] - 0.02
            for metric in ("t_mAP", "t_REC")
        )
        b_task = _mean(
            [baseline[h][metric] for h in (4, 5) for metric in ("t_mAP", "t_REC")]
        )
        p_task = _mean(
            [candidate[h][metric] for h in (4, 5) for metric in ("t_mAP", "t_REC")]
        )
    gates = {
        "G6B-1": {
            "passed": bool(frozen_hashes_unchanged and verification_proofs_passed),
            "evidence": "registered threshold-aware and GT-free CPU proofs passed; frozen hashes checked",
        },
        "G6B-2": {
            "passed": g2,
            "evidence": f"heldout mean T4/T5 ID-switch relative reduction={relative:.6f}",
        },
        "G6B-3": {
            "passed": g3,
            "evidence": f"P6B accuracy={p_accuracy:.6f}, B4 accuracy={b_accuracy:.6f}, P6B recall={p_recall:.6f}, B4 recall={b_recall:.6f}",
        },
        "G6B-4": {
            "passed": g4_t2 and p_task >= b_task - 0.01,
            "evidence": f"T2 drop gate={g4_t2}; long-task P6B={p_task:.6f}, B4={b_task:.6f}",
        },
        "G6B-5": {
            "passed": bool(evidence_complete),
            "evidence": "required ablations, paired rows, failures, provenance, and manifest checked",
        },
    }
    return gates


def _sha256_file(path: Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"required input must be a regular file: {source.name}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rio_class_mapper(model_class: int) -> int:
    if (
        isinstance(model_class, bool)
        or not isinstance(model_class, int)
        or not 0 <= model_class < 18
    ):
        raise ValueError("RIO model class must be in [0, 17]")
    labels = yaml.safe_load(
        (PROJECT_ROOT / "data/processed/rio/label_database.yaml").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(labels, Mapping):
        raise TypeError("RIO label database must be a mapping")
    valid_ids = [int(label) for label, record in labels.items() if record["validation"]]
    index = model_class + 2
    if index >= len(valid_ids):
        raise ValueError("RIO label database does not cover model class")
    return valid_ids[index]


def _load_real_inputs(
    *,
    cache_directory: Path,
    metadata_path: Path,
    config_path: Path,
    partition: str,
) -> tuple[
    object, Mapping[str, object], object, tuple[object, ...], tuple[object, ...]
]:
    protocol_config = load_p6b_config(config_path)
    protocol, protocol_manifest, _ = _frozen_protocol_bundle(
        metadata_path=Path(metadata_path).resolve()
    )
    protocol_path = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
    cache_root = Path(cache_directory).resolve()
    cache_manifest_path = cache_root / "cache_manifest.json"
    if (
        _sha256_file(protocol_path)
        != protocol_config.sources.p6a_protocol_manifest.sha256
    ):
        raise ValueError("P6-A protocol manifest SHA-256 differs from P6-B config")
    if (
        _sha256_file(cache_manifest_path)
        != protocol_config.sources.p6a_cache_manifest.sha256
    ):
        raise ValueError("P6-A cache manifest SHA-256 differs from P6-B config")
    split = build_split_manifest(protocol_manifest, seed=protocol_config.seed)
    if partition == "tuning":
        allowed_master_ids = split.tuning_master_sequence_ids
    elif partition == "heldout":
        allowed_master_ids = split.heldout_master_sequence_ids
    else:
        raise ValueError("P6-B cache partition must be tuning or heldout")
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_root / "entries",
        manifest_path=cache_manifest_path,
        allowed_master_sequence_ids=allowed_master_ids,
    )
    if partition == "tuning":
        tuning, heldout = sequences, ()
        if len(tuning) != 96:
            raise ValueError("P6-B tuning split must contain 96 orders")
    else:
        tuning, heldout = (), sequences
        if len(heldout) != 33:
            raise ValueError("P6-B heldout split must contain 33 orders")
    return protocol_config, protocol_manifest, split, tuning, heldout


def _event_groups_by_horizon(events: Sequence[object]) -> dict[int, tuple[object, ...]]:
    result = {}
    for horizon in _HORIZONS:
        result[horizon] = tuple(
            event for event in events if getattr(event, "prefix", None) == horizon
        )
    return result


def _accepted_counts(sequences: Sequence[object]) -> dict[int, int]:
    counts = {horizon: 0 for horizon in _HORIZONS}
    for sequence in sequences:
        observations = tuple(
            cache_payload_to_frozen_observation(payload)
            for payload in sequence.payloads
        )
        for horizon in _HORIZONS:
            counts[horizon] += sum(
                int(observation.valid.sum().item())
                for observation in observations[:horizon]
            )
    return counts


def _baseline_candidate(
    sequences: Sequence[object], protocol_config: object
) -> tuple[P6BCandidateRow, object]:
    p6a_config = yaml.safe_load(
        (PROJECT_ROOT / "conf/p6a/default.yaml").read_text(encoding="utf-8")
    )
    factories = build_tracker_factories(p6a_config)
    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories={"B4": factories["B4"]},
        class_mapper=_rio_class_mapper,
        background_class=18,
    )
    official = extract_official_metrics(evaluation, method="B4")
    events = _event_groups_by_horizon(evaluation.association_events)
    accepted = _accepted_counts(sequences)
    horizon_rows = []
    for horizon in _HORIZONS:
        horizon_events = events[horizon]
        identity, reactivation = aggregate_event_metrics(horizon_events)
        horizon_rows.append(
            P6BHorizonMetrics(
                horizon=horizon,
                identity_switches=int(identity["id_switches"]),
                transition_opportunities=int(identity["transition_opportunities"]),
                wrong_reactivations=int(reactivation["wrong_reactivations"]),
                predicted_reactivation_events=int(
                    reactivation["predicted_reactivation_events"]
                ),
                correct_reactivations=int(reactivation["correct_reactivations"]),
                reactivation_attempts=int(reactivation["reactivation_attempts"]),
                gap_opportunities=int(reactivation["gap_opportunities"]),
                false_births=int(identity["false_births"]),
                births=int(identity["births"]),
                rejected_births=int(identity["rejected_births"]),
                reactivation_accuracy=reactivation["reactivation_accuracy"],
                reactivation_recall=reactivation["reactivation_recall"],
                accepted_valid_observations=accepted[horizon],
                total_valid_observations=accepted[horizon],
                cluster_metrics=attach_cluster_task_metrics(
                    cluster_event_metrics(
                        horizon_events,
                        population_identities=(
                            (
                                sequence.reference_scene_id,
                                sequence.master_sequence_id,
                                sequence.order_id,
                            )
                            for sequence in sequences
                        ),
                    ),
                    evaluation.per_sequence_metrics,
                    method="B4",
                    horizon=horizon,
                    expected_sequence_count=evaluation.sequence_count,
                ),
                strict_online_tmap=official[horizon]["t_mAP"],
                strict_online_trec=official[horizon]["t_REC"],
            )
        )
    return (
        P6BCandidateRow(
            config=protocol_config.base,
            stage="baseline",
            horizons=tuple(horizon_rows),
        ),
        evaluation,
    )


def _candidate_sweep_rows(
    candidate: P6BCandidateRow,
    *,
    tuning_population: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    if tuning_population is None:
        _, split = _selection_protocol()
        tuning_population = _split_population_evidence(split, partition="tuning")
    return [
        {
            "config_id": candidate.config_id,
            "config_json": candidate.config_json,
            "stage": candidate.stage,
            "T": f"T{metric.horizon}",
            "identity_switches": metric.identity_switches,
            "transition_opportunities": metric.transition_opportunities,
            "identity_switch_rate": metric.identity_switch_rate,
            "wrong_reactivations": metric.wrong_reactivations,
            "predicted_reactivation_events": metric.predicted_reactivation_events,
            "correct_reactivations": metric.correct_reactivations,
            "reactivation_attempts": metric.reactivation_attempts,
            "gap_opportunities": metric.gap_opportunities,
            "wrong_reactivation_rate": metric.wrong_reactivation_rate,
            "false_births": metric.false_births,
            "true_births": metric.true_births,
            "births": metric.births,
            "accepted_births": metric.accepted_births,
            "rejected_births": metric.rejected_births,
            "valid_birth_opportunities": metric.valid_birth_opportunities,
            "false_birth_rate": metric.false_birth_rate,
            "cluster_metrics_json": json.dumps(
                [cluster_metrics_to_payload(item) for item in metric.cluster_metrics],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "reactivation_accuracy": metric.reactivation_accuracy,
            "reactivation_recall": metric.reactivation_recall,
            "accepted_valid_observations": metric.accepted_valid_observations,
            "total_valid_observations": metric.total_valid_observations,
            "frozen_b4_valid_observations": metric.frozen_b4_valid_observations,
            "tuning_population_count": tuning_population["count"],
            "tuning_population_sha256": tuning_population["sha256"],
            "strict_online_tmap": metric.strict_online_tmap,
            "strict_online_trec": metric.strict_online_trec,
            "full_eligible": candidate.eligible,
            "stage_eligible": candidate_stage_eligible(candidate),
            "eligibility_reasons": ";".join(candidate.eligibility_reasons),
        }
        for metric in candidate.horizons
    ]


def _candidate_summary(candidate: P6BCandidateRow) -> dict[str, object]:
    return {
        "config_id": candidate.config_id,
        "config": asdict(candidate.config),
        "stage": candidate.stage,
        "eligibility_reasons": list(candidate.eligibility_reasons),
        "horizons": [asdict(metric) for metric in candidate.horizons],
    }


def _publish_selection(output_root: Path, document: Mapping[str, object]) -> None:
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("selection output root already exists")
    selection_bytes = (
        json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(selection_bytes) > _MAX_SELECTION_DOCUMENT_BYTES:
        raise ValueError("selection document exceeds source-sized budget")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (stage / "selection.json").write_bytes(selection_bytes)
        (stage / "selected_config.yaml").write_text(
            yaml.safe_dump(document["selected_config"], sort_keys=True),
            encoding="utf-8",
        )
        (stage / "split_manifest.json").write_text(
            json.dumps(document["split_manifest"], sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("selection output root appeared during publication")
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _selection_config(document: Mapping[str, object]) -> P6BMemoryConfig:
    return P6BMemoryConfig(**dict(document["selected_config"]))


def _official_candidate(
    row: P6BCandidateRow, sequences: Sequence[object]
) -> P6BCandidateRow:
    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories={
            "P6B": lambda sequence_id: P6BTracker(
                sequence_id=sequence_id, config=row.config
            )
        },
        class_mapper=_rio_class_mapper,
        background_class=row.config.background_class,
    )
    official = extract_official_metrics(evaluation)
    return P6BCandidateRow(
        config=row.config,
        stage=row.stage,
        horizons=tuple(
            P6BHorizonMetrics(
                **{
                    **asdict(metric),
                    "cluster_metrics": attach_cluster_task_metrics(
                        metric.cluster_metrics,
                        evaluation.per_sequence_metrics,
                        method="P6B",
                        horizon=metric.horizon,
                        expected_sequence_count=evaluation.sequence_count,
                    ),
                    "strict_online_tmap": official[metric.horizon]["t_mAP"],
                    "strict_online_trec": official[metric.horizon]["t_REC"],
                }
            )
            for metric in row.horizons
        ),
    )


def _official_metric_population_evidence(evaluation: object) -> list[dict[str, object]]:
    per_sequence = getattr(evaluation, "per_sequence_metric_evidence", None)
    if isinstance(per_sequence, (str, bytes)) or not isinstance(per_sequence, Sequence):
        raise TypeError("per-sequence official metric evidence must be a sequence")
    return [
        {
            "method": method,
            "T": f"T{horizon}",
            "state": build_official_metric_population_evidence(
                [
                    {
                        key: row[key]
                        for key in (
                            "reference_scene_id",
                            "master_sequence_id",
                            "order_id",
                            "prediction_digest",
                            "state",
                        )
                    }
                    for row in per_sequence
                    if row["method"] == method and row["T"] == f"T{horizon}"
                ]
            ),
        }
        for method in ("B4", "P6B")
        for horizon in _HORIZONS
    ]


def _evaluate_methods(
    sequences: Sequence[object], config: P6BMemoryConfig
) -> tuple[list[dict[str, object]], object, list[dict[str, object]]]:
    p6a_config = yaml.safe_load(
        (PROJECT_ROOT / "conf/p6a/default.yaml").read_text(encoding="utf-8")
    )
    b4 = build_tracker_factories(p6a_config)["B4"]

    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories={
            "B4": b4,
            "P6B": lambda sequence_id: P6BTracker(
                sequence_id=sequence_id, config=config
            ),
        },
        class_mapper=_rio_class_mapper,
        background_class=config.background_class,
    )
    rows = []
    for method in ("B4", "P6B"):
        metrics = evaluation.metric_blocks["strict"][method]
        for horizon in _HORIZONS:
            scoped = tuple(
                event
                for event in evaluation.association_events
                if event.method == method and event.prefix == horizon
            )
            identity, reactivation = aggregate_event_metrics(scoped)
            transitions = int(identity["transition_opportunities"])
            switches = int(identity["id_switches"])
            predicted = int(reactivation["predicted_reactivation_events"])
            wrong = int(reactivation["wrong_reactivations"])
            births = int(identity["births"])
            rejected = int(identity["rejected_births"])
            false_births = int(identity["false_births"])
            rows.append(
                {
                    "method": method,
                    "T": f"T{horizon}",
                    "t_mAP": float(metrics[f"T{horizon}"]["online_t-mAP"]),
                    "t_REC": float(metrics[f"T{horizon}"]["online_t-REC"]),
                    "identity_switches": switches,
                    "transition_opportunities": transitions,
                    "identity_switch_rate": switches / transitions
                    if transitions
                    else 0.0,
                    "wrong_reactivations": wrong,
                    "predicted_reactivation_events": predicted,
                    "wrong_reactivation_rate": wrong / predicted if predicted else None,
                    "correct_reactivations": int(reactivation["correct_reactivations"]),
                    "reactivation_attempts": int(reactivation["reactivation_attempts"]),
                    "gap_opportunities": int(reactivation["gap_opportunities"]),
                    "reactivation_accuracy": reactivation["reactivation_accuracy"],
                    "reactivation_recall": reactivation["reactivation_recall"],
                    "false_births": false_births,
                    "births": births,
                    "rejected_births": rejected,
                    "false_birth_rate": (
                        false_births / (births + rejected)
                        if births + rejected
                        else None
                    ),
                }
            )
    official_metric_evidence = _official_metric_population_evidence(evaluation)
    return rows, evaluation, official_metric_evidence


def _per_sequence_key(
    row: Mapping[str, object], *, horizon_field: str
) -> tuple[str, str, str, str, int]:
    method = row.get("method")
    reference = row.get("reference_scene_id")
    master = row.get("master_sequence_id")
    order = row.get("order_id")
    if method not in {"B4", "P6B"}:
        raise ValueError("per-sequence method must be B4 or P6B")
    if any(
        not isinstance(value, str) or not value for value in (reference, master, order)
    ):
        raise ValueError("per-sequence identity fields must be nonempty strings")
    raw_horizon = row.get(horizon_field)
    if horizon_field == "T":
        if not isinstance(raw_horizon, str) or raw_horizon not in {
            f"T{horizon}" for horizon in _HORIZONS
        }:
            raise ValueError("per-sequence T is invalid")
        horizon = int(raw_horizon[1:])
    else:
        if isinstance(raw_horizon, bool) or raw_horizon not in _HORIZONS:
            raise ValueError("per-sequence prefix is invalid")
        horizon = int(raw_horizon)
    return method, reference, master, order, horizon


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _rate(numerator: int, denominator: int) -> float | None:
    if numerator > denominator:
        raise ValueError("metric numerator exceeds denominator")
    return numerator / denominator if denominator else None


def _index_per_sequence_rows(
    rows: Sequence[Mapping[str, object]], *, horizon_field: str, name: str
) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    indexed: dict[tuple[str, str, str, str, int], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} rows must be mappings")
        key = _per_sequence_key(row, horizon_field=horizon_field)
        if key in indexed:
            raise ValueError(f"{name} contains a duplicate per-sequence row")
        indexed[key] = row
    return indexed


def _join_per_sequence_rows(
    association_rows: Sequence[Mapping[str, object]],
    task_rows: Sequence[Mapping[str, object]],
    *,
    expected_sequence_count: int = 33,
    expected_reference_scene_ids: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    association = _index_per_sequence_rows(
        association_rows, horizon_field="prefix", name="association evidence"
    )
    tasks = _index_per_sequence_rows(
        task_rows, horizon_field="T", name="task metric evidence"
    )
    if set(association) != set(tasks):
        raise ValueError("per-sequence evidence has a missing association/task pair")
    expected_rows = expected_sequence_count * 2 * len(_HORIZONS)
    if len(association) != expected_rows:
        raise ValueError("per-sequence evidence population differs from the protocol")
    sequence_keys = {(key[1], key[2], key[3]) for key in association}
    if len(sequence_keys) != expected_sequence_count:
        raise ValueError("per-sequence sequence population differs from the protocol")
    references = {key[1] for key in association}
    if expected_reference_scene_ids is not None and references != set(
        expected_reference_scene_ids
    ):
        raise ValueError(
            "per-sequence reference population differs from held-out split"
        )

    result: list[dict[str, object]] = []
    for key in sorted(association):
        aggregate = association[key]
        task = tasks[key]
        association_digest = aggregate.get("prediction_digest")
        task_digest = task.get("prediction_digest")
        if (
            not isinstance(association_digest, str)
            or not association_digest
            or association_digest != task_digest
        ):
            raise ValueError("per-sequence prediction digests differ")
        switches = _nonnegative_count(
            aggregate.get("id_switches"), name="identity_switches"
        )
        transitions = _nonnegative_count(
            aggregate.get("transition_opportunities"),
            name="transition_opportunities",
        )
        wrong = _nonnegative_count(
            aggregate.get("wrong_reactivations"), name="wrong_reactivations"
        )
        predicted = _nonnegative_count(
            aggregate.get("predicted_reactivation_events"),
            name="predicted_reactivation_events",
        )
        correct = _nonnegative_count(
            aggregate.get("correct_reactivations"), name="correct_reactivations"
        )
        attempts = _nonnegative_count(
            aggregate.get("reactivation_attempts"), name="reactivation_attempts"
        )
        gaps = _nonnegative_count(
            aggregate.get("gap_opportunities"), name="gap_opportunities"
        )
        false_births = _nonnegative_count(
            aggregate.get("false_births"), name="false_births"
        )
        births = _nonnegative_count(aggregate.get("births"), name="births")
        rejected = _nonnegative_count(
            aggregate.get("rejected_births"), name="rejected_births"
        )
        identity_rate = _rate(switches, transitions)
        wrong_rate = _rate(wrong, predicted)
        reactivation_accuracy = _rate(correct, attempts)
        reactivation_recall = _rate(correct, gaps)
        false_birth_rate = _rate(false_births, births + rejected)
        if aggregate.get("reactivation_accuracy") != reactivation_accuracy:
            raise ValueError("reactivation accuracy differs from counts")
        if aggregate.get("reactivation_recall") != reactivation_recall:
            raise ValueError("reactivation recall differs from counts")
        method, reference, master, order, horizon = key
        result.append(
            {
                "method": method,
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
                "T": f"T{horizon}",
                "prediction_digest": association_digest,
                "t_mAP": _finite_number(task.get("t_mAP"), name="t_mAP"),
                "t_REC": _finite_number(task.get("t_REC"), name="t_REC"),
                "identity_switches": switches,
                "transition_opportunities": transitions,
                "identity_switch_rate": identity_rate,
                "wrong_reactivations": wrong,
                "predicted_reactivation_events": predicted,
                "wrong_reactivation_rate": wrong_rate,
                "correct_reactivations": correct,
                "reactivation_attempts": attempts,
                "gap_opportunities": gaps,
                "reactivation_accuracy": reactivation_accuracy,
                "reactivation_recall": reactivation_recall,
                "false_births": false_births,
                "births": births,
                "rejected_births": rejected,
                "false_birth_rate": false_birth_rate,
            }
        )

    pair_groups: dict[tuple[str, str, str, int], dict[str, dict[str, object]]] = {}
    for row in result:
        pair_key = (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(str(row["T"])[1:]),
        )
        pair_groups.setdefault(pair_key, {})[str(row["method"])] = row
    for pair in pair_groups.values():
        if set(pair) != {"B4", "P6B"}:
            raise ValueError("per-sequence method pair is incomplete")
        if pair["B4"]["prediction_digest"] != pair["P6B"]["prediction_digest"]:
            raise ValueError("paired methods use different frozen prediction digests")
    return result


_PAIRED_STATISTIC_HORIZONS = {
    "identity_switch_rate": _HORIZONS,
    "wrong_reactivation_rate": (3, 4, 5),
    "false_birth_rate": _HORIZONS,
    "reactivation_accuracy": (3, 4, 5),
    "reactivation_recall": (3, 4, 5),
    "t_mAP": _HORIZONS,
    "t_REC": _HORIZONS,
}


def _paired_statistics(
    rows: Sequence[Mapping[str, object]], *, expected_sequence_count: int = 33
) -> list[dict[str, object]]:
    indexed = _index_per_sequence_rows(rows, horizon_field="T", name="paired evidence")
    if len(indexed) != expected_sequence_count * 2 * len(_HORIZONS):
        raise ValueError("paired evidence population differs from the protocol")
    pairs: dict[tuple[str, str, str, int], dict[str, Mapping[str, object]]] = {}
    for key, row in indexed.items():
        method, reference, master, order, horizon = key
        pairs.setdefault((reference, master, order, horizon), {})[method] = row
    if len(pairs) != expected_sequence_count * len(_HORIZONS):
        raise ValueError("paired evidence sequence/horizon population differs")
    for pair in pairs.values():
        if set(pair) != {"B4", "P6B"}:
            raise ValueError("paired evidence has a missing method pair")
        if pair["B4"].get("prediction_digest") != pair["P6B"].get("prediction_digest"):
            raise ValueError("paired evidence uses different prediction digests")

    statistics: list[dict[str, object]] = []
    for metric, horizons in _PAIRED_STATISTIC_HORIZONS.items():
        for horizon in horizons:
            records: list[PairedMetricRecord] = []
            cluster_pair_deltas: dict[str, list[float]] = {}
            excluded = 0
            for (reference, master, order, pair_horizon), pair in sorted(pairs.items()):
                if pair_horizon != horizon:
                    continue
                left = pair["P6B"].get(metric)
                right = pair["B4"].get(metric)
                if left is None and right is None:
                    excluded += 1
                    continue
                if left is None or right is None:
                    raise ValueError(
                        "paired metric availability differs between methods"
                    )
                left_value = _finite_number(left, name=metric)
                right_value = _finite_number(right, name=metric)
                cluster_pair_deltas.setdefault(reference, []).append(
                    left_value - right_value
                )
                digest = pair["P6B"].get("prediction_digest")
                for method, value in (("P6B", left_value), ("B4", right_value)):
                    records.append(
                        PairedMetricRecord(
                            reference_scene_id=reference,
                            master_sequence_id=master,
                            prefix=horizon,
                            method=method,
                            metric=metric,
                            value=value,
                            order_id=order,
                            prediction_digest=str(digest),
                        )
                    )
            if not records:
                raise ValueError(f"paired statistic {metric} T{horizon} has no pairs")
            statistic = paired_cluster_bootstrap(
                records,
                method="P6B",
                baseline_method="B4",
                metric=metric,
                n_bootstrap=10_000,
                seed=45,
                expected_cluster_count=2,
                sample_standard_deviation=True,
            )
            statistics.append(
                {
                    **statistic,
                    "T": f"T{horizon}",
                    "cluster_deltas": [
                        {
                            "reference_scene_id": reference,
                            "delta": sum(values) / len(values),
                        }
                        for reference, values in sorted(cluster_pair_deltas.items())
                    ],
                    "population_pair_count": expected_sequence_count,
                    "excluded_null_pairs": excluded,
                    "standard_deviation_kind": "sample_across_reference_clusters",
                    "uncertainty_caveat": (
                        "Only two held-out reference-scene clusters; the interval is "
                        "unstable and is not a significance claim."
                    ),
                }
            )
    return statistics


def _per_sequence_rows(
    events: Sequence[object],
    task_rows: Sequence[Mapping[str, object]],
    *,
    expected_reference_scene_ids: Sequence[str],
) -> list[dict[str, object]]:
    return _join_per_sequence_rows(
        aggregate_metrics_by_sequence(events),
        task_rows,
        expected_sequence_count=33,
        expected_reference_scene_ids=expected_reference_scene_ids,
    )


def _failure_rows(events: Sequence[object]) -> list[dict[str, object]]:
    rows = []
    for method in ("B4", "P6B"):
        for horizon in _HORIZONS:
            scoped = tuple(
                event
                for event in events
                if event.method == method and event.prefix == horizon
            )
            counts = failure_breakdown(scoped)["counts"]
            for category in (
                *tuple(f"F{index}" for index in range(1, 8)),
                "unclassified",
            ):
                rows.append(
                    {
                        "method": method,
                        "T": f"T{horizon}",
                        "failure_category": category,
                        "count": int(counts[category]),
                    }
                )
    return rows


def _failure_diagnostic_rows(events: Sequence[object]) -> list[dict[str, object]]:
    categories = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")
    grouped: dict[tuple[str, str, str, str, int, str], list[object]] = {}
    for event in events:
        key = (
            str(event.method),
            str(event.reference_scene_id),
            str(event.master_sequence_id),
            str(event.order_id),
            int(event.prefix),
            str(event.prediction_digest),
        )
        grouped.setdefault(key, []).append(event)
    rows = []
    for key in sorted(
        grouped,
        key=lambda item: (
            ("B4", "P6B").index(item[0]),
            item[1],
            item[2],
            item[3],
            item[4],
        ),
    ):
        method, reference, master, order, horizon, prediction_digest = key
        counts = failure_breakdown(grouped[key])["counts"]
        rows.append(
            {
                "method": method,
                "reference_scene_id": reference,
                "master_sequence_id": master,
                "order_id": order,
                "T": f"T{horizon}",
                "prediction_digest": prediction_digest,
                **{category: int(counts[category]) for category in categories},
            }
        )
    return rows


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--cache-directory", type=Path, required=True)
    sweep.add_argument("--metadata", type=Path, required=True)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument("--config", type=Path, default=Path("conf/p6b/default.yaml"))
    final_evaluate = subparsers.add_parser("final-evaluate")
    final_evaluate.add_argument("--cache-directory", type=Path, required=True)
    final_evaluate.add_argument("--metadata", type=Path, required=True)
    final_evaluate.add_argument("--selection-root", type=Path, required=True)
    final_evaluate.add_argument("--output-root", type=Path, required=True)
    final_evaluate.add_argument(
        "--config", type=Path, default=Path("conf/p6b/default.yaml")
    )
    final_package = subparsers.add_parser("final-package")
    final_package.add_argument("--attempt-root", type=Path, required=True)
    final_package.add_argument("--selection-root", type=Path, required=True)
    final_package.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.command == "sweep":
        run_sweep(
            cache_directory=args.cache_directory,
            metadata_path=args.metadata,
            output_root=args.output_root,
            config_path=args.config,
        )
    elif args.command == "final-evaluate":
        run_final_evaluate(
            cache_directory=args.cache_directory,
            metadata_path=args.metadata,
            selection_root=args.selection_root,
            output_root=args.output_root,
            config_path=args.config,
        )
    else:
        run_final_package(
            attempt_root=args.attempt_root,
            selection_root=args.selection_root,
            output_root=args.output_root,
        )
    return 0


def run_sweep(
    *, cache_directory: Path, metadata_path: Path, output_root: Path, config_path: Path
) -> dict[str, object]:
    _log_event("sweep_start")
    source_contract = build_source_tree_contract(PROJECT_ROOT)
    verification_ledger = _run_protocol_verification_ledger()
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("selection output root already exists")
    protocol_config, _protocol_manifest, split, tuning, _heldout = _load_real_inputs(
        cache_directory=cache_directory,
        metadata_path=metadata_path,
        config_path=(
            PROJECT_ROOT / config_path if not config_path.is_absolute() else config_path
        ),
        partition="tuning",
    )
    _log_event("cache_validated", tuning_orders=len(tuning), heldout_orders=33)
    split_mapping = split.to_mapping()
    tuning_population = _sequence_population_evidence(tuning, partition="tuning")
    expected_tuning_population = _split_population_evidence(
        split_mapping, partition="tuning"
    )
    if tuning_population != expected_tuning_population:
        raise ValueError("loaded tuning population differs from the frozen split")
    tuning_references = tuple(split.tuning_reference_scene_ids)
    baseline, _baseline_evaluation = _baseline_candidate(tuning, protocol_config)
    _log_event("baseline_complete")
    sweep_sequences = cached_sequences_to_sweep_sequences(tuning)

    def fast_evaluator(config: P6BMemoryConfig, stage: str) -> P6BCandidateRow:
        _log_event(
            "candidate_start", stage=stage, config_id=canonical_config_id(config)
        )
        replay = replay_configuration(
            sweep_sequences,
            config,
            allowed_reference_scene_ids=tuning_references,
        )
        events = derive_prefix_events(
            replay,
            tuning,
            background_class=config.background_class,
        )
        row = build_candidate_row(
            replay,
            stage=stage,
            events_by_horizon=events,
        )
        _log_event("candidate_complete", stage=stage, config_id=row.config_id)
        return row

    def official_evaluator(row: P6BCandidateRow) -> P6BCandidateRow:
        _log_event("official_finalist_start", stage=row.stage, config_id=row.config_id)
        result = _official_candidate(row, tuning)
        _log_event(
            "official_finalist_complete", stage=row.stage, config_id=row.config_id
        )
        return result

    result = run_staged_sweep(
        protocol_config,
        baseline=baseline,
        fast_evaluator=fast_evaluator,
        official_evaluator=official_evaluator,
    )
    candidate_rows = tuple(
        row
        for candidate in result.candidate_rows
        for row in _candidate_sweep_rows(candidate, tuning_population=tuning_population)
    )
    finalist_rows = tuple(
        row
        for candidate in result.finalist_rows
        for row in _candidate_sweep_rows(candidate, tuning_population=tuning_population)
    )
    document = build_selection_document(
        source_commit=source_contract["source_commit"],
        split_manifest=split_mapping,
        selected_config=result.selected.config,
        ranking_key=candidate_ranking_key(result.selected, baseline=baseline),
        baseline={
            "rows": _candidate_sweep_rows(baseline, tuning_population=tuning_population)
        },
        candidate_rows=candidate_rows,
        finalist_rows=finalist_rows,
        selected_by_stage={
            stage: candidate.config_id
            for stage, candidate in result.selected_by_stage.items()
        },
        provenance=_expected_selection_provenance(protocol_config, split_mapping),
        verification_ledger=verification_ledger,
    )
    _publish_selection(output, document)
    _log_event(
        "sweep_complete",
        selected_config_id=document["selected_config_id"],
        output=str(output),
    )
    return document


def _validate_selection_source_boundary(
    selection: Mapping[str, object], source_contract: Mapping[str, str]
) -> None:
    selection_commit = str(selection["source_commit"])
    current_commit = source_contract["source_commit"]
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", selection_commit, current_commit],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        raise ValueError("selection source commit is not an ancestor of final source")
    changed = _git(
        PROJECT_ROOT, "diff", "--name-only", f"{selection_commit}..{current_commit}"
    )
    if any(
        path and not path.startswith("artifacts/P6B_selection/")
        for path in changed.splitlines()
    ):
        raise ValueError("source changed beyond the committed P6-B selection evidence")


def _heldout_evaluation_payload(
    *,
    cache_directory: Path,
    metadata_path: Path,
    config_path: Path,
    selection: Mapping[str, object],
) -> dict[str, object]:
    protocol_config, _protocol_manifest, split, _tuning, heldout = _load_real_inputs(
        cache_directory=cache_directory,
        metadata_path=metadata_path,
        config_path=config_path,
        partition="heldout",
    )
    split_mapping = split.to_mapping()
    if selection["split_manifest"] != split_mapping:
        raise ValueError("frozen selection split differs from current protocol")
    if selection["provenance"] != _expected_selection_provenance(
        protocol_config, split_mapping
    ):
        raise ValueError("frozen selection provenance differs from current inputs")
    config = _selection_config(selection)
    final_results, evaluation, official_metric_evidence = _evaluate_methods(
        heldout, config
    )
    per_sequence = _per_sequence_rows(
        evaluation.association_events,
        evaluation.per_sequence_metrics,
        expected_reference_scene_ids=split.heldout_reference_scene_ids,
    )
    statistics = _paired_statistics(per_sequence)
    failures = _failure_rows(evaluation.association_events)
    failure_diagnostics = _failure_diagnostic_rows(evaluation.association_events)
    if (
        len(final_results) != 8
        or len(per_sequence) != 264
        or len(failures) != 64
        or len(failure_diagnostics) != 264
    ):
        raise ValueError("held-out evaluation populations differ from protocol v2")
    checkpoint = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
    p5 = PROJECT_ROOT / "artifacts/P5/persist4d_mvp_eval.json"
    p6a = PROJECT_ROOT / "artifacts/P6A/p6a_eval.json"
    checkpoint_sha = _sha256_file(checkpoint)
    p5_sha = _sha256_file(p5)
    p6a_sha = _sha256_file(p6a)
    if (
        checkpoint_sha != EXPECTED_CHECKPOINT_SHA256
        or p5_sha != EXPECTED_P5_SHA256
        or p6a_sha != EXPECTED_P6A_SHA256
    ):
        raise ValueError(
            "frozen P5/P6-A/model hashes changed before held-out evaluation"
        )
    return {
        "status": "pass",
        "heldout_order_count": len(heldout),
        "heldout_reference_scene_ids": list(split.heldout_reference_scene_ids),
        "selected_config_id": selection["selected_config_id"],
        "selected_config_sha256": selection["selected_config_sha256"],
        "provenance": {
            "checkpoint": {
                "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
                "sha256": checkpoint_sha,
            },
            "p5": {
                "ref": "repo:artifacts/P5/persist4d_mvp_eval.json",
                "sha256": p5_sha,
            },
            "p6a": {
                "ref": "repo:artifacts/P6A/p6a_eval.json",
                "sha256": p6a_sha,
            },
            "p6a_protocol_manifest": {
                "ref": protocol_config.sources.p6a_protocol_manifest.reference,
                "sha256": protocol_config.sources.p6a_protocol_manifest.sha256,
            },
            "p6a_cache_manifest": {
                "ref": protocol_config.sources.p6a_cache_manifest.reference,
                "sha256": protocol_config.sources.p6a_cache_manifest.sha256,
            },
        },
        "final_results": final_results,
        "official_metric_evidence": official_metric_evidence,
        "per_sequence_results": per_sequence,
        "failure_analysis": failures,
        "failure_diagnostics": failure_diagnostics,
        "statistical_analysis": statistics,
    }


def run_final_evaluate(
    *,
    cache_directory: Path,
    metadata_path: Path,
    selection_root: Path,
    output_root: Path,
    config_path: Path,
) -> dict[str, object]:
    _log_event("final_evaluate_start")
    expected_selection_root = PROJECT_ROOT / "artifacts/P6B_selection"
    candidate_selection_root = Path(selection_root)
    if not candidate_selection_root.is_absolute():
        candidate_selection_root = PROJECT_ROOT / candidate_selection_root
    if Path(os.path.abspath(candidate_selection_root)) != expected_selection_root:
        raise ValueError("final-evaluate requires the canonical P6-B selection root")
    current = PROJECT_ROOT
    for component in Path("artifacts/P6B_selection").parts:
        current /= component
        if current.is_symlink():
            raise ValueError("canonical P6-B selection path must not contain symlinks")
    selection_path = expected_selection_root / "selection.json"
    if selection_path.is_symlink() or not selection_path.is_file():
        raise ValueError("canonical P6-B selection must be a regular file")
    committed = subprocess.run(
        ["git", "show", "HEAD:artifacts/P6B_selection/selection.json"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if selection_path.read_bytes() != committed:
        raise ValueError("canonical P6-B selection differs from the committed Git blob")
    source_contract = build_source_tree_contract(PROJECT_ROOT)
    selection = load_selection_document(selection_path)
    _validate_selection_source_boundary(selection, source_contract)
    expected_output = PROJECT_ROOT / "artifacts/P6B_heldout"
    if Path(output_root).resolve() != expected_output.resolve():
        raise ValueError("final-evaluate output must be repo:artifacts/P6B_heldout")
    expected_config = PROJECT_ROOT / "conf/p6b/default.yaml"
    candidate_config = Path(config_path)
    if not candidate_config.is_absolute():
        candidate_config = PROJECT_ROOT / candidate_config
    if Path(os.path.abspath(candidate_config)) != expected_config:
        raise ValueError("final-evaluate requires the canonical P6-B config")
    current = PROJECT_ROOT
    for component in Path("conf/p6b/default.yaml").parts:
        current /= component
        if current.is_symlink():
            raise ValueError("canonical P6-B config path must not contain symlinks")
    if expected_config.is_symlink() or not expected_config.is_file():
        raise ValueError("canonical P6-B config must be a regular file")
    if _sha256_file(expected_config) != selection["provenance"]["p6b_config_sha256"]:
        raise ValueError("canonical P6-B config differs from selection provenance")
    resolved_config = expected_config
    raw = run_exactly_once_heldout(
        output_root,
        evaluator=lambda: _heldout_evaluation_payload(
            cache_directory=cache_directory,
            metadata_path=metadata_path,
            config_path=resolved_config,
            selection=selection,
        ),
        source_commit=source_contract["source_commit"],
        selection={
            "ref": "repo:artifacts/P6B_selection/selection.json",
            "sha256": _sha256_file(selection_path),
        },
        split_sha256=str(selection["split_manifest"]["sha256"]),
        p6b_config_sha256=_sha256_file(resolved_config),
        command=(
            "final-evaluate",
            "--protocol",
            "P6B-v2",
            "--cache",
            "frozen-P6A",
        ),
    )
    _log_event(
        "final_evaluate_complete",
        attempt_id=raw["attempt_id"],
        heldout_orders=raw["evaluation"]["heldout_order_count"],
    )
    return raw


def run_final_package(
    *, attempt_root: Path, selection_root: Path, output_root: Path
) -> dict[str, object]:
    _log_event("final_package_start")
    source_contract = build_source_tree_contract(
        PROJECT_ROOT, allowed_dirty_prefixes=("artifacts/P6B_heldout/",)
    )
    expected_attempt = PROJECT_ROOT / "artifacts/P6B_heldout"
    expected_selection = PROJECT_ROOT / "artifacts/P6B_selection"
    expected_output = PROJECT_ROOT / "artifacts/P6B"
    if Path(attempt_root).resolve() != expected_attempt.resolve():
        raise ValueError(
            "final-package attempt root must be repo:artifacts/P6B_heldout"
        )
    if Path(selection_root).resolve() != expected_selection.resolve():
        raise ValueError(
            "final-package selection root must be repo:artifacts/P6B_selection"
        )
    if Path(output_root).resolve() != expected_output.resolve():
        raise ValueError("final-package output root must be repo:artifacts/P6B")
    selection = load_selection_document(Path(selection_root) / "selection.json")
    raw = recover_heldout_attempt(attempt_root)
    selection_sha = _sha256_file(Path(selection_root) / "selection.json")
    if raw["selection"]["sha256"] != selection_sha:
        raise ValueError("held-out raw selection SHA differs from current selection")
    evaluation_commit = str(raw["source_commit"])
    current_commit = source_contract["source_commit"]
    ancestor = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", evaluation_commit, current_commit],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )
    if not ancestor:
        raise ValueError("held-out source commit is not an ancestor of package source")
    changed = _git(
        PROJECT_ROOT, "diff", "--name-only", f"{evaluation_commit}..{current_commit}"
    )
    if any(
        path and not path.startswith("artifacts/P6B_heldout/")
        for path in changed.splitlines()
    ):
        raise ValueError("source changed beyond committed held-out raw evidence")
    attempt = _json_without_constants(Path(attempt_root) / "attempt.json")
    if not isinstance(attempt, Mapping):
        raise TypeError("held-out attempt must be a mapping")
    root = finalize_p6b_artifact(
        build_p6b_artifact_root(
            source_tree_contract=source_contract,
            selection_document=selection,
            heldout_attempt=attempt,
            heldout_raw=raw,
        )
    )
    publish_p6b_artifact(output_root, root)
    _log_event("final_package_complete", decision=root["decision"])
    return root


if __name__ == "__main__":
    raise SystemExit(main())
