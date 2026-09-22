#!/usr/bin/env python3
"""Generate frozen R1 segment records for the Perception-Gain scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from torch import Tensor

from scripts.evaluate_persist4d import (
    _move_data_to_device,
    _move_targets_to_device,
    _validate_cuda_device,
)
from scripts.perception_gain_data import build_scorer_record, select_scorer_pairs
from scripts.preflight_task_memory_episode import (
    _rio_base_dataset,
    load_reference_by_scene,
)
from scripts.train_perception_gain import (
    R1_BYTES,
    R1_SHA256,
    _sample_rng,
    _stable_seed,
    compose_variant_config,
    filter_rio_train_indices,
)
from trainer.perception_gain_trainer import (
    PerceptionGainTrainer,
    strict_load_r1_perception,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path("/home/ww/persist4d_runs/perception_gain_v1/assets.local.json")
DEFAULT_ROLES = PROJECT_ROOT / "artifacts/perception_gain_v1/DATA_ROLES.json"
PUBLIC_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1/training/scorer"


class ScorerDataError(RuntimeError):
    """Raised when scorer data cannot satisfy the frozen V1 contract."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScorerDataError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def run(
    *,
    assets_path: Path,
    roles_path: Path,
    external_root: Path,
    device_name: str,
    shard_records: int = 8,
) -> dict[str, object]:
    if shard_records <= 0:
        raise ScorerDataError("shard_records must be positive")
    assets = _read_json(assets_path)
    roles = _read_json(roles_path).get("roles")
    train_references = roles.get("TRAIN") if isinstance(roles, Mapping) else None
    if not isinstance(train_references, list) or not train_references:
        raise ScorerDataError("TRAIN references are unavailable")
    checkpoint_path = Path(assets["r1_checkpoint"])
    if (
        checkpoint_path.stat().st_size != R1_BYTES
        or _sha256(checkpoint_path) != R1_SHA256
    ):
        raise ScorerDataError("R1 checkpoint identity differs")

    device = _validate_cuda_device(device_name)
    output_root = external_root / "training/scorer/data"
    config = compose_variant_config(
        "C0",
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=external_root / "training/scorer/feature_producer",
    )
    dataset = _rio_base_dataset(config, data_root=Path(assets["data_root"]), horizon=2)
    indices, references = filter_rio_train_indices(
        dataset,
        reference_by_scene=load_reference_by_scene(Path(assets["rio_metadata"])),
        train_references=set(train_references),
    )
    candidates = [
        {
            "dataset_index": index,
            "pair_id": str(dataset.sequence_names[index]),
            "reference_id": reference,
        }
        for index, reference in zip(indices, references, strict=True)
        if index
        not in {
            int(context["sequence_index"])
            for context in dataset.known_empty_scan_contexts
        }
    ]
    inventory = select_scorer_pairs(candidates, limit=64, minimum_references=8)
    if inventory["status"] != "PASS":
        raise ScorerDataError("scorer inventory lacks eight TRAIN references")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ScorerDataError("R1 checkpoint lacks state_dict")
    system = PerceptionGainTrainer(config)
    load_audit = strict_load_r1_perception(system, state, allow_semantic_scorer=False)
    system.to(device).eval().requires_grad_(False)
    collate = hydra.utils.instantiate(config.data.train_collation)

    records: list[dict[str, object]] = []
    incomplete: list[dict[str, object]] = []
    started = time.perf_counter()
    for ordinal, selection in enumerate(inventory["pairs"]):
        reference_id = str(selection["reference_id"])
        pair_id = str(selection["pair_id"])
        try:
            seed = _stable_seed("pgv1-scorer-data", 45, reference_id, pair_id)
            with _sample_rng(seed):
                sample = dataset[int(selection["dataset_index"])]
                data, targets, names = collate([sample])
            if list(names) != [pair_id] or len(targets) != 1:
                raise ScorerDataError("scorer collator identity differs")
            labels = getattr(data, "labels", None)
            if (
                isinstance(labels, (str, bytes))
                or not isinstance(labels, Sequence)
                or len(labels) != 1
                or not isinstance(labels[0], Tensor)
                or labels[0].ndim != 2
                or labels[0].shape[1] < 2
            ):
                raise ScorerDataError("scorer semantic labels are unavailable")
            semantic_labels = labels[0][:, 0].detach()
            data = _move_data_to_device(data, device)
            targets = _move_targets_to_device(targets, device)
            target = targets[0]
            raw_coordinates = system._process_raw_coordinates(data)
            if not isinstance(raw_coordinates, Tensor):
                raise ScorerDataError("scorer raw coordinates are unavailable")
            with torch.inference_mode():
                output = system(
                    data,
                    point2segment=[target["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
            layers = output.get("segment_features")
            if (
                not isinstance(layers, list)
                or len(layers) != 1
                or not isinstance(layers[0], list)
                or len(layers[0]) != 1
            ):
                raise ScorerDataError("R1 segment features are unavailable")
            records.append(
                build_scorer_record(
                    reference_id=reference_id,
                    pair_id=pair_id,
                    segment_features=layers[0][0],
                    point2segment=target["point2segment"],
                    semantic_labels=semantic_labels,
                    raw_coordinates=raw_coordinates,
                    temporal_stages=target["temporal_stages"],
                    thing_class_ids=tuple(range(2, 20)),
                    stuff_class_ids=(0, 1),
                    ignore_class_ids=(255,),
                )
            )
            print(
                json.dumps(
                    {
                        "completed": ordinal + 1,
                        "pair_id": pair_id,
                        "total": len(inventory["pairs"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except (RuntimeError, ValueError, ScorerDataError) as error:
            incomplete.append(
                {
                    "pair_id": pair_id,
                    "reference_id": reference_id,
                    "error_type": type(error).__name__,
                    "reason": str(error),
                }
            )

    shards = []
    output_root.mkdir(parents=True, exist_ok=True)
    for shard_id, offset in enumerate(range(0, len(records), shard_records)):
        path = output_root / f"scorer-{shard_id:04d}.pt"
        values = tuple(records[offset : offset + shard_records])
        _atomic_torch_save(path, values)
        shards.append(
            {
                "bytes": path.stat().st_size,
                "external_reference": f"external:training/scorer/data/{path.name}",
                "record_count": len(values),
                "sha256": _sha256(path),
            }
        )
    elapsed = time.perf_counter() - started
    selected_references = {str(row["reference_id"]) for row in records}
    status = (
        "PASS"
        if len(records) == len(inventory["pairs"]) and len(selected_references) >= 8
        else "PARTIAL"
    )
    summary = {
        "schema_version": "perception-gain-scorer-data-v1",
        "status": status,
        "candidate_pair_count": len(candidates),
        "selected_pair_count": len(inventory["pairs"]),
        "completed_pair_count": len(records),
        "reference_count": len(selected_references),
        "incomplete_units": incomplete,
        "checkpoint_sha256": R1_SHA256,
        "load_audit": load_audit,
        "taxonomy": {
            "thing_class_ids": list(range(2, 20)),
            "stuff_class_ids": [0, 1],
            "ignore_class_ids": [255],
            "class_space": "dataset_remapped_semantic_ids_before_label_offset",
        },
        "shards": shards,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
    }
    _atomic_json(PUBLIC_ROOT / "DATA_MANIFEST.json", summary)
    del system
    torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument(
        "--external-root",
        type=Path,
        default=Path("/home/ww/persist4d_runs/perception_gain_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-records", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        assets_path=args.assets,
        roles_path=args.roles,
        external_root=args.external_root,
        device_name=args.device,
        shard_records=args.shard_records,
    )
    print(json.dumps({"status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
