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
    aggregate_event_metrics,
    aggregate_metrics_by_sequence,
    failure_breakdown,
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
    P6BHorizonMetrics,
    build_candidate_row,
    cached_sequences_to_sweep_sequences,
    candidate_ranking_key,
    derive_prefix_events,
    extract_official_metrics,
    replay_configuration,
    run_staged_sweep,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA_VERSION = 1
EXPECTED_CHECKPOINT_SHA256 = "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
EXPECTED_P5_SHA256 = "7da68910b0c0b43b5f04d8ae7d56543a460231c0616c62b2fb9485b88fd781a1"
EXPECTED_P6A_SHA256 = "bffc32fde402396258ed750943101bd8acb6318bc2526ea8f99a9ec42dbe9399"
_HORIZONS = (2, 3, 4, 5)
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
    tuning = split.get("tuning_reference_scene_ids")
    heldout = split.get("heldout_reference_scene_ids")
    if not isinstance(tuning, list) or not isinstance(heldout, list) or set(tuning) & set(heldout):
        raise ValueError("selection split contains overlap")
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
        identity, reactivation = aggregate_event_metrics(events[horizon])
        horizon_rows.append(
            P6BHorizonMetrics(
                horizon=horizon,
                identity_switches=int(identity["id_switches"]),
                wrong_reactivations=int(reactivation["wrong_reactivations"]),
                false_births=int(identity["false_births"]),
                reactivation_accuracy=reactivation["reactivation_accuracy"],
                reactivation_recall=reactivation["reactivation_recall"],
                accepted_valid_observations=accepted[horizon],
                total_valid_observations=accepted[horizon],
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
            "wrong_reactivations": metric.wrong_reactivations,
            "false_births": metric.false_births,
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
) -> tuple[list[dict[str, object]], tuple[object, ...]]:
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
    return rows, evaluation.association_events


def _per_sequence_rows(events: Sequence[object]) -> list[dict[str, object]]:
    rows = []
    for aggregate in aggregate_metrics_by_sequence(events):
        rows.append(
            {
                "method": aggregate["method"],
                "reference_scene_id": aggregate["reference_scene_id"],
                "master_sequence_id": aggregate["master_sequence_id"],
                "order_id": aggregate["order_id"],
                "T": f"T{aggregate['prefix']}",
                "identity_switches": aggregate["id_switches"],
                "transition_opportunities": aggregate["transition_opportunities"],
                "identity_switch_rate": aggregate["id_switch_rate"],
                "wrong_reactivations": aggregate["wrong_reactivations"],
                "false_births": aggregate["false_births"],
                "reactivation_accuracy": aggregate["reactivation_accuracy"],
                "reactivation_recall": aggregate["reactivation_recall"],
            }
        )
    return rows


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
        provenance={
            "p6a_protocol_manifest_sha256": protocol_config.sources.p6a_protocol_manifest.sha256,
            "cache_manifest_sha256": protocol_config.sources.p6a_cache_manifest.sha256,
            "split_sha256": split.sha256,
        },
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
    if selection["provenance"] != {
        "p6a_protocol_manifest_sha256": protocol_config.sources.p6a_protocol_manifest.sha256,
        "cache_manifest_sha256": protocol_config.sources.p6a_cache_manifest.sha256,
        "split_sha256": split.sha256,
    }:
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
