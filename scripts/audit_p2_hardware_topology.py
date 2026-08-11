#!/usr/bin/env python3
"""Record P2 one/two-GPU topology candidates without running model training."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.p2_preflight import (
    P2_CONFIG_NAME,
    P2_FORMAL_EPOCH_SAMPLE_MULTIPLE,
    P2_FORMAL_SAMPLER_NUM_SAMPLES,
    P2_PREFLIGHT_SCHEMA_VERSION,
    P2_RIO_SEQUENCE_FILTER_COUNTS,
    require_p2_preflight_authorization,
)

DEFAULT_PREFLIGHT = REPO_ROOT / "artifacts" / "P2" / "scannet_preflight.json"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "P2" / "hardware_topology_profile.csv"
OFFICIAL_SOURCE_COMMIT = "fb2fe42eb8f1e926567c48eea9acb874e608ee10"
OFFICIAL_SPLIT_COUNTS = {"train": 1201, "validation": 312, "test": 100}
NYU40_INSTANCE_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
UNKNOWN_NUMA_AFFINITIES = {"", "-", "N/A", "NA", "UNKNOWN"}
SHARED_AUTHORIZATION_FIELDS = (
    "config_contract",
    "source_tree_contract",
    "runtime_source_contract",
    "runtime_environment_contract",
    "official_split_identity",
    "input_manifest",
    "authorization",
)

CSV_COLUMNS = [
    "schema_version",
    "stage",
    "candidate_id",
    "gpu_count",
    "gpu_indices",
    "detected_gpu_count",
    "gpu_model",
    "memory_per_gpu_mib",
    "interconnect",
    "cpu_affinity",
    "numa_affinity",
    "same_numa",
    "scannet_preflight_ref",
    "scannet_preflight_status",
    "formal_training_authorized",
    "model_benchmark_status",
    "training_throughput_samples_per_s",
    "optimizer_steps_per_s",
    "peak_vram_mib",
    "communication_diagnostic_status",
    "communication_bandwidth_gbps",
    "topology_selection_status",
]

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _run_nvidia_smi(arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *arguments],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("nvidia-smi is unavailable") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"exit code {error.returncode}"
        raise RuntimeError(f"nvidia-smi failed: {detail}") from error
    return result.stdout


def _read_or_query(path: Path | None, query: Sequence[str]) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return _run_nvidia_smi(query)


def parse_inventory(text: str) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for raw_row in csv.reader(text.splitlines()):
        if not raw_row or not any(value.strip() for value in raw_row):
            continue
        if len(raw_row) != 3:
            raise ValueError("GPU inventory rows must have index, name, and memory")
        try:
            index = int(raw_row[0].strip())
            memory_mib = int(raw_row[2].strip())
        except ValueError as error:
            raise ValueError("GPU index and memory must be integers") from error
        name = raw_row[1].strip()
        if index in seen_indices or not name or memory_mib <= 0:
            raise ValueError("GPU inventory contains an invalid or duplicate row")
        seen_indices.add(index)
        inventory.append(
            {"index": index, "name": name, "memory_mib": memory_mib}
        )
    if not inventory:
        raise ValueError("GPU inventory is empty")
    return sorted(inventory, key=lambda gpu: gpu["index"])


def parse_topology(text: str, gpu_indices: Sequence[int]) -> dict[int, dict[str, Any]]:
    clean_lines = [
        ANSI_ESCAPE.sub("", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    header_labels: list[str] | None = None
    for line in clean_lines:
        tokens = line.split()
        labels: list[str] = []
        for token in tokens:
            if re.fullmatch(r"GPU\d+", token):
                labels.append(token)
            else:
                break
        if len(labels) >= 2:
            header_labels = labels
            break
    if header_labels is None:
        raise ValueError("nvidia-smi topology header is missing")

    header_indices = [int(label[3:]) for label in header_labels]
    if set(header_indices) != set(gpu_indices):
        raise ValueError("GPU inventory and topology matrix do not match")

    topology: dict[int, dict[str, Any]] = {}
    for line in clean_lines:
        tokens = line.split()
        if not tokens or not re.fullmatch(r"GPU\d+", tokens[0]):
            continue
        if len(tokens) < 1 + len(header_indices) + 2:
            continue
        row_index = int(tokens[0][3:])
        if row_index not in gpu_indices:
            continue
        links = {
            column_index: tokens[position + 1]
            for position, column_index in enumerate(header_indices)
        }
        remainder = tokens[1 + len(header_indices) :]
        topology[row_index] = {
            "links": links,
            "cpu_affinity": remainder[0],
            "numa_affinity": remainder[1],
        }

    if set(topology) != set(gpu_indices):
        raise ValueError("nvidia-smi topology matrix has missing GPU rows")
    for index in gpu_indices:
        if topology[index]["links"].get(index) != "X":
            raise ValueError("nvidia-smi topology matrix has an invalid diagonal")
    return topology


def _portable_preflight_ref(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except (OSError, ValueError):
        return "external:scannet_preflight"
    return f"repo:{relative.as_posix()}"


def _compose_p2_config() -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base="1.2"
    ):
        return compose(config_name=P2_CONFIG_NAME)


def _shared_p2_authorization_gate(path: Path) -> bool:
    try:
        cfg = _compose_p2_config()
        require_p2_preflight_authorization(cfg, artifact_path=path)
    except Exception:  # noqa: BLE001 - the ready gate must fail closed.
        return False
    return True


def _full_pass_contract(
    payload: Mapping[str, Any], *, artifact_path: Path | None = None
) -> bool:
    if not (
        payload.get("schema_version") == P2_PREFLIGHT_SCHEMA_VERSION
        and payload.get("status") == "pass"
        and payload.get("formal_p2_training_authorized") is True
        and payload.get("official_source_commit") == OFFICIAL_SOURCE_COMMIT
        and payload.get("expected_split_counts") == OFFICIAL_SPLIT_COUNTS
        and payload.get("split_metadata_status") == "pass"
        and payload.get("errors") == []
    ):
        return False
    if not all(
        isinstance(payload.get(field), Mapping)
        for field in SHARED_AUTHORIZATION_FIELDS
    ):
        return False

    split_metadata = payload.get("split_metadata")
    if not isinstance(split_metadata, Mapping):
        return False
    for split, expected in OFFICIAL_SPLIT_COUNTS.items():
        record = split_metadata.get(split)
        if not isinstance(record, Mapping) or not (
            record.get("status") == "pass"
            and record.get("expected") == expected
            and record.get("observed") == expected
            and record.get("unique") == expected
        ):
            return False

    raw_assets = payload.get("raw_assets")
    if not isinstance(raw_assets, Mapping) or not (
        raw_assets.get("status") == "pass"
        and raw_assets.get("expected_scene_count") == 1613
        and raw_assets.get("complete_scene_count") == 1613
        and raw_assets.get("missing_asset_count") == 0
    ):
        return False
    raw_by_split = raw_assets.get("by_split")
    if not isinstance(raw_by_split, Mapping):
        return False
    for split, expected in OFFICIAL_SPLIT_COUNTS.items():
        record = raw_by_split.get(split)
        if not isinstance(record, Mapping) or not (
            record.get("status") == "pass"
            and record.get("expected_scene_count") == expected
            and record.get("complete_scene_count") == expected
            and record.get("missing_scene_count") == 0
            and record.get("missing_asset_count") == 0
        ):
            return False

    processed_assets = payload.get("processed_assets")
    if not isinstance(processed_assets, Mapping) or not (
        processed_assets.get("status") == "pass"
        and processed_assets.get("expected_scene_count") == 1613
        and processed_assets.get("database_scene_count") == 1613
        and processed_assets.get("npy_scene_count") == 1613
    ):
        return False
    processed_by_split = processed_assets.get("by_split")
    if not isinstance(processed_by_split, Mapping):
        return False
    for split, expected in OFFICIAL_SPLIT_COUNTS.items():
        record = processed_by_split.get(split)
        if not isinstance(record, Mapping) or not (
            record.get("status") == "pass"
            and record.get("expected_scene_count") == expected
            and record.get("database_scene_count") == expected
            and record.get("npy_scene_count") == expected
        ):
            return False

    taxonomy = payload.get("class_taxonomy")
    if not isinstance(taxonomy, Mapping) or not (
        taxonomy.get("status") == "pass"
        and taxonomy.get("class_count") == 18
        and taxonomy.get("valid_class_ids") == NYU40_INSTANCE_IDS
    ):
        return False

    mix = payload.get("mix_instantiation")
    if not isinstance(mix, Mapping) or not (
        mix.get("attempted") is True
        and mix.get("status") == "pass"
        and mix.get("implementation") == "datasets.multi_dataset.MultiDataset"
        and mix.get("dataset_names") == ["rio", "scannet"]
        and mix.get("dataset_sizes") == [
            P2_RIO_SEQUENCE_FILTER_COUNTS["train"]["retained_count"],
            1201,
        ]
        and mix.get("weights") == [1.0, 0.8]
        and mix.get("temporal_windows") == [2, 1]
        and mix.get("sampler") == "WeightedRandomSampler"
        and mix.get("sampler_num_samples") == P2_FORMAL_SAMPLER_NUM_SAMPLES
        and mix.get("epoch_sample_multiple") == P2_FORMAL_EPOCH_SAMPLE_MULTIPLE
    ):
        return False
    return artifact_path is not None and _shared_p2_authorization_gate(artifact_path)


def _read_preflight(path: Path) -> tuple[str, bool, str, int]:
    if not path.is_file():
        return "missing", False, "blocked_missing_preflight", 2
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "invalid", False, "blocked_invalid_preflight", 2
    if not isinstance(payload, dict):
        return "invalid", False, "blocked_invalid_preflight", 2

    status = str(payload.get("status", "invalid"))
    if status == "pass" and _full_pass_contract(payload, artifact_path=path):
        return status, True, "pending_formal_benchmark", 0
    if status == "pass":
        return "invalid_pass_contract", False, "blocked_invalid_preflight", 2
    if status not in {"blocked_missing_scannet"}:
        return "invalid", False, "blocked_invalid_preflight", 2
    return status, False, "blocked_by_scannet_preflight", 2


def _unique_join(values: Sequence[str]) -> str:
    return ";".join(dict.fromkeys(values))


def _same_known_numa(values: Sequence[str]) -> bool:
    normalized = [value.strip().upper() for value in values]
    return (
        len(set(normalized)) == 1
        and normalized[0] not in UNKNOWN_NUMA_AFFINITIES
    )


def build_rows(
    inventory: Sequence[dict[str, Any]],
    topology: dict[int, dict[str, Any]],
    preflight_ref: str,
    preflight_status: str,
    formal_training_authorized: bool,
    topology_selection_status: str,
) -> list[dict[str, Any]]:
    by_index = {gpu["index"]: gpu for gpu in inventory}
    indices = [gpu["index"] for gpu in inventory]
    candidate_indices = [
        *itertools.combinations(indices, 1),
        *itertools.combinations(indices, 2),
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidate_indices:
        gpus = [by_index[index] for index in candidate]
        cpu_affinities = [topology[index]["cpu_affinity"] for index in candidate]
        numa_affinities = [topology[index]["numa_affinity"] for index in candidate]
        if len(candidate) == 1:
            interconnect = "SELF"
        else:
            interconnect = topology[candidate[0]]["links"][candidate[1]]
        rows.append(
            {
                "schema_version": 1,
                "stage": "P2",
                "candidate_id": "+".join(f"gpu{index}" for index in candidate),
                "gpu_count": len(candidate),
                "gpu_indices": "+".join(str(index) for index in candidate),
                "detected_gpu_count": len(inventory),
                "gpu_model": _unique_join([gpu["name"] for gpu in gpus]),
                "memory_per_gpu_mib": _unique_join(
                    [str(gpu["memory_mib"]) for gpu in gpus]
                ),
                "interconnect": interconnect,
                "cpu_affinity": _unique_join(cpu_affinities),
                "numa_affinity": _unique_join(numa_affinities),
                "same_numa": str(_same_known_numa(numa_affinities)).lower(),
                "scannet_preflight_ref": preflight_ref,
                "scannet_preflight_status": preflight_status,
                "formal_training_authorized": str(
                    formal_training_authorized
                ).lower(),
                "model_benchmark_status": "not_run",
                "training_throughput_samples_per_s": "null",
                "optimizer_steps_per_s": "null",
                "peak_vram_mib": "null",
                "communication_diagnostic_status": "not_run",
                "communication_bandwidth_gbps": "null",
                "topology_selection_status": topology_selection_status,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--inventory-file", type=Path)
    parser.add_argument("--topology-file", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inventory_text = _read_or_query(
            args.inventory_file,
            [
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        )
        inventory = parse_inventory(inventory_text)
        topology_text = _read_or_query(args.topology_file, ["topo", "-m"])
        topology = parse_topology(
            topology_text, [gpu["index"] for gpu in inventory]
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"hardware topology audit failed: {error}", file=sys.stderr)
        return 3

    preflight_status, authorized, selection_status, return_code = _read_preflight(
        args.preflight
    )
    rows = build_rows(
        inventory=inventory,
        topology=topology,
        preflight_ref=_portable_preflight_ref(args.preflight),
        preflight_status=preflight_status,
        formal_training_authorized=authorized,
        topology_selection_status=selection_status,
    )
    try:
        write_csv(args.output, rows)
    except OSError as error:
        print(f"hardware topology audit failed: {error}", file=sys.stderr)
        return 3

    print(
        f"P2 hardware topology: {selection_status}; "
        f"candidates={len(rows)}; model_benchmark=not_run"
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
