"""Leakage-free sweep and held-out evaluation entry point for Persist4D P6-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
from scripts.p6b_artifacts import finalize_p6b_artifact, publish_p6b_artifact
from scripts.p6b_association import P6BTracker
from scripts.p6b_protocol import (
    build_split_manifest,
    canonical_config_id,
    canonical_config_json,
    load_p6b_config,
)
from scripts.p6b_sweep import (
    P6BCandidateRow,
    P6BClusterMetrics,
    P6BHorizonMetrics,
    build_candidate_row,
    cached_sequences_to_sweep_sequences,
    candidate_ranking_key,
    cluster_event_metrics,
    derive_prefix_events,
    extract_official_metrics,
    replay_configuration,
    run_staged_sweep,
    validate_staged_sweep_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA_VERSION = 2
EXPECTED_CHECKPOINT_SHA256 = "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
EXPECTED_P5_SHA256 = "7da68910b0c0b43b5f04d8ae7d56543a460231c0616c62b2fb9485b88fd781a1"
EXPECTED_P6A_SHA256 = "bffc32fde402396258ed750943101bd8acb6318bc2526ea8f99a9ec42dbe9399"
_HORIZONS = (2, 3, 4, 5)
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
        "births",
        "rejected_births",
        "false_birth_rate",
        "cluster_metrics_json",
        "reactivation_accuracy",
        "reactivation_recall",
        "accepted_valid_observations",
        "total_valid_observations",
        "strict_online_tmap",
        "strict_online_trec",
        "eligible",
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


def build_source_tree_contract(repo_root: Path = PROJECT_ROOT) -> dict[str, str]:
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    hidden = [
        line
        for line in _git(root, "ls-files", "-v").splitlines()
        if line and line[0].islower()
    ]
    if dirty or hidden:
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
        "gt_free_inference_status": "pass",
    }


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
    allowed_stages: Sequence[str] = _STAGES,
) -> tuple[P6BCandidateRow, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ValueError("candidate rows must be a nonempty sequence")
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    order: list[tuple[str, str]] = []
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _SWEEP_ROW_KEYS:
            raise ValueError(f"candidate row {index} differs from the strict schema")
        stage = raw_row["stage"]
        config_id = raw_row["config_id"]
        if stage not in allowed_stages or not isinstance(config_id, str) or not config_id:
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
                clusters = tuple(P6BClusterMetrics(**item) for item in cluster_payload)
                metric = P6BHorizonMetrics(
                    horizon=horizon,
                    identity_switches=row["identity_switches"],
                    transition_opportunities=row["transition_opportunities"],
                    wrong_reactivations=row["wrong_reactivations"],
                    predicted_reactivation_events=row[
                        "predicted_reactivation_events"
                    ],
                    correct_reactivations=row["correct_reactivations"],
                    reactivation_attempts=row["reactivation_attempts"],
                    gap_opportunities=row["gap_opportunities"],
                    false_births=row["false_births"],
                    births=row["births"],
                    rejected_births=row["rejected_births"],
                    reactivation_accuracy=row["reactivation_accuracy"],
                    reactivation_recall=row["reactivation_recall"],
                    accepted_valid_observations=row[
                        "accepted_valid_observations"
                    ],
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
            if official_metrics_required != (
                metric.strict_online_tmap is not None
                and metric.strict_online_trec is not None
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
            if not isinstance(row["eligible"], bool) or row["eligible"] != (not row_reasons):
                raise ValueError("candidate eligible flag differs from reasons")
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
) -> dict[str, object]:
    if not isinstance(selected_config, P6BMemoryConfig):
        raise TypeError("selected_config must be P6BMemoryConfig")
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
        "baseline": dict(baseline),
        "candidate_rows": [dict(row) for row in candidate_rows],
        "finalist_rows": [dict(row) for row in finalist_rows],
        "selected_by_stage": dict(selected_by_stage),
        "heldout_evaluated": False,
    }
    _validate_selection_document(document)
    return document


def _validate_selection_document(document: Mapping[str, object]) -> None:
    if set(document) != _SELECTION_KEYS:
        raise ValueError("selection document keys differ from schema")
    if document["schema_version"] != SELECTION_SCHEMA_VERSION or document["status"] != "pass":
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
    provenance = document["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _SELECTION_PROVENANCE_KEYS:
        raise ValueError("selection provenance differs from the strict schema")
    if dict(provenance) != _expected_selection_provenance(protocol, expected_split):
        raise ValueError("selection provenance differs from frozen inputs")

    baseline_raw = document["baseline"]
    if not isinstance(baseline_raw, Mapping) or set(baseline_raw) != {"rows"}:
        raise ValueError("selection baseline differs from the strict schema")
    baseline_rows = _parse_candidate_rows(
        baseline_raw["rows"],
        official_metrics_required=True,
        allowed_stages=("baseline",),
    )
    if len(baseline_rows) != 1 or baseline_rows[0].stage != "baseline":
        raise ValueError("selection must contain one frozen baseline candidate")
    candidates = _parse_candidate_rows(
        document["candidate_rows"], official_metrics_required=False
    )
    finalists = _parse_candidate_rows(
        document["finalist_rows"], official_metrics_required=True
    )
    selected_ids = document["selected_by_stage"]
    if not isinstance(selected_ids, Mapping) or tuple(selected_ids) != _STAGES:
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
    if isinstance(ranking_key, (str, bytes)) or not isinstance(
        ranking_key, Sequence
    ):
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
        document = json.loads(source.read_text(encoding="utf-8"))
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
) -> dict[str, dict[str, object]]:
    baseline = _method_metrics(rows, "B4")
    candidate = _method_metrics(rows, "P6B")
    b45 = _mean([baseline[h]["identity_switch_rate"] for h in (4, 5)])
    p45 = _mean([candidate[h]["identity_switch_rate"] for h in (4, 5)])
    relative = (b45 - p45) / b45 if b45 > 0 else 0.0
    g2 = relative >= 0.10 and all(
        candidate[h]["identity_switch_rate"] <= baseline[h]["identity_switch_rate"]
        for h in (4, 5)
    )
    b_accuracy = _mean([baseline[h]["reactivation_accuracy"] for h in (3, 4, 5)])
    p_accuracy = _mean([candidate[h]["reactivation_accuracy"] for h in (3, 4, 5)])
    b_recall = _mean([baseline[h]["reactivation_recall"] for h in (3, 4, 5)])
    p_recall = _mean([candidate[h]["reactivation_recall"] for h in (3, 4, 5)])
    g3 = p_accuracy >= 0.70 and p_accuracy >= b_accuracy and p_recall >= b_recall - 0.05
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
            "passed": bool(frozen_hashes_unchanged),
            "evidence": "threshold-aware tests pass; inference excludes GT; frozen hashes checked",
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
    if isinstance(model_class, bool) or not isinstance(model_class, int) or not 0 <= model_class < 18:
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
    *, cache_directory: Path, metadata_path: Path, config_path: Path
) -> tuple[object, Mapping[str, object], object, tuple[object, ...], tuple[object, ...]]:
    protocol_config = load_p6b_config(config_path)
    protocol, protocol_manifest, _ = _frozen_protocol_bundle(
        metadata_path=Path(metadata_path).resolve()
    )
    protocol_path = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
    cache_root = Path(cache_directory).resolve()
    cache_manifest_path = cache_root / "cache_manifest.json"
    if _sha256_file(protocol_path) != protocol_config.sources.p6a_protocol_manifest.sha256:
        raise ValueError("P6-A protocol manifest SHA-256 differs from P6-B config")
    if _sha256_file(cache_manifest_path) != protocol_config.sources.p6a_cache_manifest.sha256:
        raise ValueError("P6-A cache manifest SHA-256 differs from P6-B config")
    split = build_split_manifest(protocol_manifest, seed=protocol_config.seed)
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_root / "entries",
        manifest_path=cache_manifest_path,
    )
    tuning, heldout = partition_cached_sequences(sequences, split.to_mapping())
    if len(tuning) != 96 or len(heldout) != 33:
        raise ValueError("P6-B split must contain 96 tuning and 33 heldout orders")
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
                cluster_metrics=cluster_event_metrics(horizon_events),
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


def _candidate_sweep_rows(candidate: P6BCandidateRow) -> list[dict[str, object]]:
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
            "births": metric.births,
            "rejected_births": metric.rejected_births,
            "false_birth_rate": metric.false_birth_rate,
            "cluster_metrics_json": json.dumps(
                [asdict(item) for item in metric.cluster_metrics],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            "reactivation_accuracy": metric.reactivation_accuracy,
            "reactivation_recall": metric.reactivation_recall,
            "accepted_valid_observations": metric.accepted_valid_observations,
            "total_valid_observations": metric.total_valid_observations,
            "strict_online_tmap": metric.strict_online_tmap,
            "strict_online_trec": metric.strict_online_trec,
            "eligible": candidate.eligible,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        (stage / "selection.json").write_text(
            json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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
                    "strict_online_tmap": official[metric.horizon]["t_mAP"],
                    "strict_online_trec": official[metric.horizon]["t_REC"],
                }
            )
            for metric in row.horizons
        ),
    )


def _evaluate_methods(
    sequences: Sequence[object], config: P6BMemoryConfig
) -> tuple[list[dict[str, object]], object]:
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
            rows.append(
                {
                    "method": method,
                    "T": f"T{horizon}",
                    "t_mAP": float(metrics[f"T{horizon}"]["online_t-mAP"]),
                    "t_REC": float(metrics[f"T{horizon}"]["online_t-REC"]),
                    "identity_switches": switches,
                    "identity_switch_rate": switches / transitions if transitions else 0.0,
                    "reactivation_accuracy": reactivation["reactivation_accuracy"],
                    "reactivation_recall": reactivation["reactivation_recall"],
                    "false_births": int(identity["false_births"]),
                }
            )
    return rows, evaluation


def _per_sequence_key(
    row: Mapping[str, object], *, horizon_field: str
) -> tuple[str, str, str, str, int]:
    method = row.get("method")
    reference = row.get("reference_scene_id")
    master = row.get("master_sequence_id")
    order = row.get("order_id")
    if method not in {"B4", "P6B"}:
        raise ValueError("per-sequence method must be B4 or P6B")
    if any(not isinstance(value, str) or not value for value in (reference, master, order)):
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
        raise ValueError("per-sequence reference population differs from held-out split")

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
        if pair["B4"].get("prediction_digest") != pair["P6B"].get(
            "prediction_digest"
        ):
            raise ValueError("paired evidence uses different prediction digests")

    statistics: list[dict[str, object]] = []
    for metric, horizons in _PAIRED_STATISTIC_HORIZONS.items():
        for horizon in horizons:
            records: list[PairedMetricRecord] = []
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
                    raise ValueError("paired metric availability differs between methods")
                digest = pair["P6B"].get("prediction_digest")
                for method, value in (("P6B", left), ("B4", right)):
                    records.append(
                        PairedMetricRecord(
                            reference_scene_id=reference,
                            master_sequence_id=master,
                            prefix=horizon,
                            method=method,
                            metric=metric,
                            value=_finite_number(value, name=metric),
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
            for category in (*tuple(f"F{index}" for index in range(1, 8)), "unclassified"):
                rows.append(
                    {
                        "method": method,
                        "T": f"T{horizon}",
                        "failure_category": category,
                        "count": int(counts[category]),
                    }
                )
    return rows


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("sweep", "final"):
        child = subparsers.add_parser(command)
        child.add_argument("--cache-directory", type=Path, required=True)
        child.add_argument("--metadata", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--config", type=Path, default=Path("conf/p6b/default.yaml"))
        if command == "final":
            child.add_argument("--selection-root", type=Path, required=True)
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
    else:
        run_final(
            cache_directory=args.cache_directory,
            metadata_path=args.metadata,
            selection_root=args.selection_root,
            output_root=args.output_root,
            config_path=args.config,
        )
    return 0


def run_sweep(
    *, cache_directory: Path, metadata_path: Path, output_root: Path, config_path: Path
) -> dict[str, object]:
    _log_event("sweep_start")
    source_contract = build_source_tree_contract(PROJECT_ROOT)
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("selection output root already exists")
    protocol_config, _protocol_manifest, split, tuning, _heldout = _load_real_inputs(
        cache_directory=cache_directory,
        metadata_path=metadata_path,
        config_path=(PROJECT_ROOT / config_path if not config_path.is_absolute() else config_path),
    )
    _log_event("cache_validated", tuning_orders=len(tuning), heldout_orders=33)
    split_mapping = split.to_mapping()
    tuning_references = tuple(split.tuning_reference_scene_ids)
    baseline, _baseline_evaluation = _baseline_candidate(tuning, protocol_config)
    _log_event("baseline_complete")
    sweep_sequences = cached_sequences_to_sweep_sequences(tuning)

    def fast_evaluator(config: P6BMemoryConfig, stage: str) -> P6BCandidateRow:
        _log_event("candidate_start", stage=stage, config_id=canonical_config_id(config))
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
        _log_event("official_finalist_complete", stage=row.stage, config_id=row.config_id)
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
        for row in _candidate_sweep_rows(candidate)
    )
    finalist_rows = tuple(
        row
        for candidate in result.finalist_rows
        for row in _candidate_sweep_rows(candidate)
    )
    document = build_selection_document(
        source_commit=source_contract["source_commit"],
        split_manifest=split_mapping,
        selected_config=result.selected.config,
        ranking_key=candidate_ranking_key(result.selected),
        baseline={"rows": _candidate_sweep_rows(baseline)},
        candidate_rows=candidate_rows,
        finalist_rows=finalist_rows,
        selected_by_stage={
            stage: candidate.config_id
            for stage, candidate in result.selected_by_stage.items()
        },
        provenance=_expected_selection_provenance(protocol_config, split_mapping),
    )
    _publish_selection(output, document)
    _log_event(
        "sweep_complete",
        selected_config_id=document["selected_config_id"],
        output=str(output),
    )
    return document


def run_final(
    *,
    cache_directory: Path,
    metadata_path: Path,
    selection_root: Path,
    output_root: Path,
    config_path: Path,
) -> dict[str, object]:
    _log_event("final_start")
    source_contract = build_source_tree_contract(PROJECT_ROOT)
    selection_path = Path(selection_root) / "selection.json"
    selection = load_selection_document(selection_path)
    selection_commit = str(selection["source_commit"])
    current_commit = source_contract["source_commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", selection_commit, current_commit],
        cwd=PROJECT_ROOT,
        check=False,
    ).returncode == 0
    if not ancestor:
        raise ValueError("selection source commit is not an ancestor of final source")
    changed = _git(PROJECT_ROOT, "diff", "--name-only", f"{selection_commit}..{current_commit}")
    if any(
        path and not path.startswith("artifacts/P6B_selection/")
        for path in changed.splitlines()
    ):
        raise ValueError("source changed beyond the committed P6-B selection evidence")
    protocol_config, _protocol_manifest, split, _tuning, heldout = _load_real_inputs(
        cache_directory=cache_directory,
        metadata_path=metadata_path,
        config_path=(PROJECT_ROOT / config_path if not config_path.is_absolute() else config_path),
    )
    if selection["split_manifest"] != split.to_mapping():
        raise ValueError("frozen selection split differs from current protocol")
    if selection["provenance"] != _expected_selection_provenance(
        protocol_config, split.to_mapping()
    ):
        raise ValueError("frozen selection provenance differs from current inputs")
    config = _selection_config(selection)
    final_results, events = _evaluate_methods(heldout, config)
    _log_event("heldout_evaluation_complete", heldout_orders=len(heldout))
    per_sequence = _per_sequence_rows(events)
    failures = _failure_rows(events)
    checkpoint = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
    p5 = PROJECT_ROOT / "artifacts/P5/persist4d_mvp_eval.json"
    p6a = PROJECT_ROOT / "artifacts/P6A/p6a_eval.json"
    checkpoint_sha = _sha256_file(checkpoint)
    p5_sha = _sha256_file(p5)
    p6a_sha = _sha256_file(p6a)
    frozen_unchanged = (
        checkpoint_sha == EXPECTED_CHECKPOINT_SHA256
        and p5_sha == EXPECTED_P5_SHA256
        and p6a_sha == EXPECTED_P6A_SHA256
    )
    gates = compute_final_gate_results(
        final_results,
        evidence_complete=bool(per_sequence and failures and selection["candidate_rows"]),
        frozen_hashes_unchanged=frozen_unchanged,
    )
    decision = "P6B_GO" if all(record["passed"] for record in gates.values()) else "P6B_STOP"
    root = finalize_p6b_artifact(
        {
            "schema_version": 1,
            "status": "pass",
            "decision": decision,
            "source_commit": current_commit,
            "source_tree_contract": source_contract,
            "provenance": {
                "checkpoint": {"ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt", "sha256": checkpoint_sha},
                "p5": {"ref": "repo:artifacts/P5/persist4d_mvp_eval.json", "sha256": p5_sha},
                "p6a": {"ref": "repo:artifacts/P6A/p6a_eval.json", "sha256": p6a_sha},
                "p6a_protocol_manifest": {"ref": protocol_config.sources.p6a_protocol_manifest.reference, "sha256": protocol_config.sources.p6a_protocol_manifest.sha256},
                "p6a_cache_manifest": {"ref": protocol_config.sources.p6a_cache_manifest.reference, "sha256": protocol_config.sources.p6a_cache_manifest.sha256},
            },
            "split_manifest": split.to_mapping(),
            "selection": {
                "config_id": selection["selected_config_id"],
                "config_sha256": selection["selected_config_sha256"],
                "config": selection["selected_config"],
                "ranking_key": selection["ranking_key"],
                "tuning_reference_scene_ids": list(split.tuning_reference_scene_ids),
            },
            "sweep_rows": selection["candidate_rows"],
            "final_results": final_results,
            "per_sequence_results": per_sequence,
            "failure_analysis": failures,
            "gate_results": gates,
            "claims_supported": [
                (
                    "P6-B improves held-out identity continuity under frozen local predictions."
                    if gates["G6B-2"]["passed"]
                    else "P6-B held-out evidence is reported without an improvement claim."
                )
            ],
            "claims_not_supported": [
                "P6-B does not establish SOTA, retraining gains, P7, or P8 claims."
            ],
            "next_action": (
                "Freeze P6-B and separately preregister P7."
                if decision == "P6B_GO"
                else "Stop after P6-B and analyze failed held-out gates before any P7 work."
            ),
            "artifact_manifest": [],
        }
    )
    publish_p6b_artifact(Path(output_root), root)
    _log_event("final_complete", decision=decision, output=str(output_root))
    return root


if __name__ == "__main__":
    raise SystemExit(main())
