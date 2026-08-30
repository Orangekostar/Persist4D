#!/usr/bin/env python3
"""Audit full active-train label-255 and weighted-mix semantics."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import torch
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.rescene_data_audit import (
    LabelSample,
    filter255_supervision_contract,
    inventory_filter255,
    summarize_weighted_draws,
)
from utils.rescene_rootcause_preflight import canonical_sha256

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "artifacts/rescene_task_learning_root_cause_v1/audit"
)


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _compose_dataset():
    import hydra

    with initialize_config_dir(
        version_base="1.2", config_dir=str(PROJECT_ROOT / "conf")
    ):
        config = compose(config_name="config_rescene4d_concerto_rootcause")
    with open_dict(config):
        config.data.train_dataset.image_augmentations_path = None
        config.data.train_dataset.volume_augmentations_path = None
    return config, hydra.utils.instantiate(config.data.train_dataset)


def _label_samples(dataset) -> Iterator[LabelSample]:
    for child in dataset.datasets:
        names = getattr(child, "sequence_names", None)
        for index in range(len(child)):
            sample = child[index]
            labels = torch.as_tensor(sample[2])
            if labels.ndim != 2 or labels.shape[1] < 2:
                raise RuntimeError("active train sample has invalid label columns")
            sample_id = (
                str(names[index])
                if names is not None
                else Path(str(child.data[index]["filepath"])).stem
            )
            if index % 100 == 0:
                print(f"inventory {child.dataset_name} {index}/{len(child)}", flush=True)
            yield LabelSample(
                dataset=str(child.dataset_name),
                sample_id=sample_id,
                semantic_labels=labels[:, 0],
                instance_ids=labels[:, 1],
            )


def _yaml_count(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return len(payload)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_audit() -> dict[str, object]:
    config, dataset = _compose_dataset()
    inventory = inventory_filter255(_label_samples(dataset), excluded_classes=(0, 1))
    draws = [int(value) for value in dataset.sampler]
    names = tuple(str(child.dataset_name) for child in dataset.datasets)
    sizes = tuple(len(child) for child in dataset.datasets)
    draw_summary = summarize_weighted_draws(
        draws, dataset_sizes=sizes, dataset_names=names
    )
    counts = {
        "official_unmodified_rio_train_sequences": _yaml_count(
            PROJECT_ROOT / "data/processed/rio/train_database.yaml"
        ),
        "rio_t2_all_split_sequences": _yaml_count(
            PROJECT_ROOT / "data/processed/rio/sequence_database_sliding_2.yaml"
        ),
        "current_filtered_rio_train_sequences": sizes[0],
        "official_unmodified_scannet_train_scans": _yaml_count(
            PROJECT_ROOT / "data/processed/scannet/train_database.yaml"
        ),
        "current_filtered_scannet_train_scans": sizes[1],
        "sampler_num_samples": len(draws),
    }
    portable_contract = {
        "source_commit": _git_head(),
        "rio_t2_database_sha256": (
            "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416"
        ),
        "dataset_sizes": list(sizes),
        "weights": [float(value) for value in dataset.weights],
        "sampler_seed": int(config.data.train_dataset.sampler_seed),
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": portable_contract["source_commit"],
        "contract_sha256": canonical_sha256(portable_contract),
        "database_counts": counts,
        "filter255": inventory,
        "supervision_contract": filter255_supervision_contract(
            raw_ignore_label=255,
            label_offset=2,
            criterion_ignore_index=253,
        ),
        "weighted_sampler": draw_summary,
    }


def _csv_bytes(result: dict[str, object]) -> bytes:
    output = io.StringIO()
    fields = [
        "dataset",
        "target_instances",
        "target_points",
        "label255_instances",
        "label255_points",
        "instance_fraction",
        "point_fraction",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    inventory = result["filter255"]
    for row in inventory["per_dataset"]:
        writer.writerow({key: row[key] for key in fields})
    writer.writerow({"dataset": "total", **inventory["totals"]})
    return output.getvalue().encode("ascii")


def _markdown(result: dict[str, object]) -> str:
    counts = result["database_counts"]
    totals = result["filter255"]["totals"]
    gate = result["filter255"]["gate"]
    sampler = result["weighted_sampler"]
    return "\n".join(
        [
            "# ReScene Data Semantics",
            "",
            "Status: `PASS`",
            "",
            "## Database And Mix",
            "",
            f"Unmodified/current RIO train counts are `{counts['official_unmodified_rio_train_sequences']}` / `{counts['current_filtered_rio_train_sequences']}` (the all-split T2 map has `{counts['rio_t2_all_split_sequences']}` records); ScanNet counts are `{counts['official_unmodified_scannet_train_scans']}` / `{counts['current_filtered_scannet_train_scans']}`. The active sampler draws `{counts['sampler_num_samples']}` examples.",
            "",
            f"Observed one-epoch draws are `{sampler['dataset_draws']}` with `{sampler['unique_sample_count']}` unique concatenated indices and replacement duplicate rate `{sampler['replacement_duplicate_rate']:.12g}`.",
            "",
            "## Label 255",
            "",
            f"The active database contains `{totals['label255_instances']}` label-255 target instances and `{totals['label255_points']}` points, representing `{totals['instance_fraction']:.12g}` of target instances and `{totals['point_fraction']:.12g}` of supervised target points.",
            "",
            f"R4 materiality is `{str(gate['material']).lower()}` under the fixed 0.5% instance-or-point threshold.",
            "",
            "Without filtering 255, target label 253 is ignored by classification and receives the matcher ignore sentinel, but its mask/dice target and mask matcher costs remain included. Filtering 255 removes that target before all criterion paths.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_audit()
    _publish(
        args.output_dir / "filter255_inventory.csv",
        _csv_bytes(result),
    )
    _publish(
        args.output_dir / "data_semantics.json",
        (
            json.dumps(
                result,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )
    _publish(
        args.output_dir / "DATA_SEMANTICS.md",
        _markdown(result).encode("ascii"),
    )
    print(json.dumps({"status": "pass", "samples": result["filter255"]["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
