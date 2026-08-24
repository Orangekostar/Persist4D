"""Run and publish the frozen final-evidence capacity sensitivity study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.evaluate_persist4d_p6a import load_cached_protocol_sequences
from scripts.final_evidence_capacity import (
    CAPACITY_GRID,
    CapacityBootstrap,
    CapacityEvaluation,
    CapacityRobustness,
    assess_robust_capacity_improvement,
    build_class_mapper_from_label_document,
    build_protocol_from_reviewer_manifest,
    capacity_cluster_bootstrap,
    classify_capacity_gate,
    evaluate_capacity_sequences,
)
from scripts.p6a_metrics import OfficialMetricAccumulator

_CAPACITY_GATES = frozenset(
    {
        "CAPACITY_100_OK",
        "CAPACITY_SENSITIVITY_ONLY",
        "CAPACITY_CONFIG_REOPEN",
    }
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("capacity CSV rows must not be empty")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def build_capacity_artifact_payloads(
    *,
    evaluation: CapacityEvaluation,
    bootstrap: CapacityBootstrap,
    robustness: CapacityRobustness,
    gate: str,
    provenance: Mapping[str, object],
) -> dict[str, bytes]:
    """Serialize the complete measured capacity result without filesystem state."""

    if not isinstance(evaluation, CapacityEvaluation):
        raise TypeError("evaluation must be CapacityEvaluation")
    if not isinstance(bootstrap, CapacityBootstrap):
        raise TypeError("bootstrap must be CapacityBootstrap")
    if not isinstance(robustness, CapacityRobustness):
        raise TypeError("robustness must be CapacityRobustness")
    if gate not in _CAPACITY_GATES:
        raise ValueError("capacity gate classification is invalid")
    if not isinstance(provenance, Mapping):
        raise TypeError("capacity provenance must be a mapping")
    return {
        "capacity_per_sequence.csv": _csv_bytes(evaluation.per_sequence_rows),
        "capacity_aggregate.csv": _csv_bytes(evaluation.aggregate_rows),
        "capacity_cluster_bootstrap.csv": _csv_bytes(bootstrap.effects),
        "capacity_per_scene_effects.csv": _csv_bytes(bootstrap.per_scene_effects),
        "capacity_robustness.json": _json_bytes(
            {
                "robust_improvement": robustness.robust_improvement,
                "candidates": robustness.candidates,
            }
        ),
        "capacity_gate.json": _json_bytes(
            {
                "classification": gate,
                "main_capacity": 100,
                "configuration_reopened": gate == "CAPACITY_CONFIG_REOPEN",
            }
        ),
        "capacity_raw.json": _json_bytes(
            {
                "provenance": dict(provenance),
                "per_sequence_rows": evaluation.per_sequence_rows,
                "aggregate_rows": evaluation.aggregate_rows,
                "cluster_bootstrap": bootstrap.effects,
                "per_scene_effects": bootstrap.per_scene_effects,
                "robustness": {
                    "robust_improvement": robustness.robust_improvement,
                    "candidates": robustness.candidates,
                },
                "classification": gate,
            }
        ),
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be a mapping: {path.name}")
    return value


def _load_yaml(path: Path) -> Mapping[object, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path.name}")
    return value


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already exists with different content: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _repository_path(repository: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _verified_inputs(
    *,
    source_binding_path: Path,
    observation_manifest_path: Path,
    reviewer_manifest_path: Path,
    cache_manifest_path: Path,
    config_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[object, Any]]:
    source_binding = _load_json(source_binding_path)
    observation_manifest = _load_json(observation_manifest_path)
    config = _load_yaml(config_path)
    if (
        source_binding.get("architecture_status") != "FINAL_LOCK"
        or source_binding.get("status") != "pass"
        or config.get("architecture_status") != "FINAL_LOCK"
        or tuple(config.get("capacity_grid", ())) != CAPACITY_GRID
        or int(config.get("main_capacity", -1)) != 100
    ):
        raise ValueError("capacity architecture or grid binding differs")
    expected_cache_hash = observation_manifest.get("cache_manifest_sha256")
    if _sha256_file(cache_manifest_path) != expected_cache_hash:
        raise ValueError("capacity cache manifest hash differs")
    expected_reviewer_hash = source_binding.get("reviewer_closure", {}).get(
        "manifest_sha256"
    )
    if _sha256_file(reviewer_manifest_path) != expected_reviewer_hash:
        raise ValueError("reviewer-closure manifest hash differs")
    return source_binding, observation_manifest, config


def run(args: argparse.Namespace) -> str:
    repository = Path(__file__).resolve().parents[1]
    paths = {
        name: _repository_path(repository, getattr(args, name))
        for name in (
            "cache_directory",
            "cache_manifest",
            "reviewer_manifest",
            "dataset_spec",
            "label_database",
            "config",
            "source_binding",
            "observation_manifest",
            "output_directory",
        )
    }
    source_binding, observation_manifest, config = _verified_inputs(
        source_binding_path=paths["source_binding"],
        observation_manifest_path=paths["observation_manifest"],
        reviewer_manifest_path=paths["reviewer_manifest"],
        cache_manifest_path=paths["cache_manifest"],
        config_path=paths["config"],
    )
    reviewer_manifest = _load_json(paths["reviewer_manifest"])
    expected_master_count = int(reviewer_manifest["protocol"]["expected_master_count"])
    protocol = build_protocol_from_reviewer_manifest(
        reviewer_manifest,
        expected_master_count=expected_master_count,
    )
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=paths["cache_directory"],
        manifest_path=paths["cache_manifest"],
    )
    class_mapper = build_class_mapper_from_label_document(
        _load_yaml(paths["label_database"])
    )
    tracker = config["tracker"]

    def metric_factory(mode: str) -> OfficialMetricAccumulator:
        return OfficialMetricAccumulator(
            mode=mode,
            dataset_spec=paths["dataset_spec"],
        )

    evaluation = evaluate_capacity_sequences(
        sequences,
        capacities=tuple(config["capacity_grid"]),
        class_mapper=class_mapper,
        background_class=int(tracker["background_class"]),
        metric_factory=metric_factory,
        expected_sequence_count=expected_master_count * 3,
        class_weight=float(tracker["class_weight"]),
        association_threshold=float(tracker["association_threshold"]),
        update_rate=float(tracker["update_rate"]),
        max_update_rate=float(tracker["max_update_rate"]),
    )
    robust_config = config["robust_improvement"]
    statistics = config["statistics"]
    bootstrap_metrics = (
        "causal_prefix_t_REC",
        "gap_recovery_recall",
        "causal_prefix_t_mAP",
        "normalized_id_switch_rate",
    )
    bootstrap = capacity_cluster_bootstrap(
        evaluation.per_sequence_rows,
        reference_capacity=int(robust_config["reference_capacity"]),
        candidate_capacities=tuple(robust_config["candidate_capacities"]),
        metrics=bootstrap_metrics,
        horizons=tuple(config["horizons"]),
        expected_cluster_count=int(statistics["expected_clusters"]),
        replicates=int(statistics["bootstrap_replicates"]),
        seed=int(statistics["seed"]),
    )
    non_degradation = robust_config["non_degradation_metrics"]
    robustness = assess_robust_capacity_improvement(
        bootstrap.effects,
        evaluation.aggregate_rows,
        reference_capacity=int(robust_config["reference_capacity"]),
        candidate_capacities=tuple(robust_config["candidate_capacities"]),
        primary_metrics=tuple(robust_config["primary_metrics"]),
        minimum_absolute_improvement=float(
            robust_config["minimum_absolute_improvement"]
        ),
        maximum_t_map_drop=float(
            non_degradation["causal_prefix_t_mAP"]["maximum_absolute_drop"]
        ),
        maximum_id_switch_increase=float(
            non_degradation["normalized_id_switch_rate"]["maximum_absolute_increase"]
        ),
    )
    selection = config["selection"]
    gate = classify_capacity_gate(
        robust_improvement=robustness.robust_improvement,
        preexisting_development_split=bool(selection["preexisting_development_split"]),
        selected_without_final_tuning=bool(selection["selected_without_final_tuning"]),
        architecture_unchanged=True,
    )
    provenance = {
        "schema_version": 1,
        "architecture_status": "FINAL_LOCK",
        "baseline_source_commit": source_binding["baseline_source_commit"],
        "source_binding_sha256": _sha256_file(paths["source_binding"]),
        "observation_manifest_sha256": _sha256_file(paths["observation_manifest"]),
        "reviewer_manifest_sha256": _sha256_file(paths["reviewer_manifest"]),
        "cache_manifest_sha256": _sha256_file(paths["cache_manifest"]),
        "cache_entry_file_sha256_list_digest": observation_manifest[
            "entry_file_sha256_list_digest"
        ],
        "config_sha256": _sha256_file(paths["config"]),
        "dataset_spec_sha256": _sha256_file(paths["dataset_spec"]),
        "label_database_sha256": _sha256_file(paths["label_database"]),
        "capacity_code_sha256": _sha256_file(
            repository / "scripts/final_evidence_capacity.py"
        ),
        "runner_sha256": _sha256_file(Path(__file__)),
        "sequence_count": len(sequences),
        "reference_scene_count": len(
            {sequence.reference_scene_id for sequence in sequences}
        ),
        "capacities": tuple(config["capacity_grid"]),
        "horizons": tuple(config["horizons"]),
        "timing_device": config["timing"]["device"],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    payloads = build_capacity_artifact_payloads(
        evaluation=evaluation,
        bootstrap=bootstrap,
        robustness=robustness,
        gate=gate,
        provenance=provenance,
    )
    artifact_records = {
        filename: {
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }
        for filename, payload in payloads.items()
    }
    output_directory = paths["output_directory"]
    for filename, payload in payloads.items():
        _publish_exact(output_directory / filename, payload)
    _publish_exact(
        output_directory / "capacity_evaluation_manifest.json",
        _json_bytes(
            {
                "schema_version": 1,
                "status": "pass",
                "classification": gate,
                "provenance": provenance,
                "artifacts": artifact_records,
            }
        ),
    )
    return gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-directory", required=True)
    parser.add_argument(
        "--cache-manifest",
        default="artifacts/system_comparison/persistent_predictions/manifest.json",
    )
    parser.add_argument(
        "--reviewer-manifest",
        default="artifacts/reviewer_closure/reviewer_closure_manifest.json",
    )
    parser.add_argument("--dataset-spec", required=True)
    parser.add_argument("--label-database", required=True)
    parser.add_argument("--config", default="configs/final_evidence/capacity.yaml")
    parser.add_argument(
        "--source-binding",
        default="artifacts/final_evidence/source_binding.json",
    )
    parser.add_argument(
        "--observation-manifest",
        default="artifacts/final_evidence/capacity_observation_manifest.json",
    )
    parser.add_argument(
        "--output-directory",
        default="artifacts/final_evidence/capacity",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    gate = run(build_parser().parse_args(argv))
    print(json.dumps({"classification": gate}, sort_keys=True))
    return 2 if gate == "CAPACITY_CONFIG_REOPEN" else 0


if __name__ == "__main__":
    sys.exit(main())
