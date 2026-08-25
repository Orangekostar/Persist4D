#!/usr/bin/env python3
"""Evaluate the full-T2 and exact canonical Protocol-B T2 populations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/protocol_bridge"
DEFAULT_CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-reviewer-closure-v3/protocol_bridge"
)
BRIDGE_INVENTORY = OUTPUT_ROOT / "bridge_inventory.csv"
BRIDGE_DATABASE = OUTPUT_ROOT / "sequence_database_protocol_b_exact_t2.yaml"
PROTOCOL_MANIFEST = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
CHECKPOINT = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
METRIC_SPEC = PROJECT_ROOT / "data/processed/rio/rio.yaml"
EVALUATION_SEEDS = (45, 46, 47)
METHOD = "ReScene4D-C-local-reimplementation"


class ProtocolBridgeEvaluationError(ValueError):
    """Raised when PB1 inference inputs or caches violate the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolBridgeEvaluationError(f"{name} must be a SHA-256 digest")
    return value


@dataclass(frozen=True)
class RuntimeBinding:
    checkpoint_sha256: str
    runtime_config_sha256: str
    postprocess_sha256: str
    metric_adapter_sha256: str
    metric_spec_sha256: str
    min_region_size: int
    precision: str
    batch_size: int
    num_workers: int

    def __post_init__(self) -> None:
        for field in (
            "checkpoint_sha256",
            "runtime_config_sha256",
            "postprocess_sha256",
            "metric_adapter_sha256",
            "metric_spec_sha256",
        ):
            _digest(getattr(self, field), name=field)
        if self.min_region_size != 100:
            raise ProtocolBridgeEvaluationError("min_region_size must remain 100")
        if self.precision != "32-true":
            raise ProtocolBridgeEvaluationError("bridge evaluation must use FP32")
        if self.batch_size != 1:
            raise ProtocolBridgeEvaluationError(
                "bridge validation batch size must remain 1"
            )
        if self.num_workers != 4:
            raise ProtocolBridgeEvaluationError(
                "bridge validation worker count must remain 4"
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationGroup:
    seed: int
    population: str
    order_id: str
    horizon: int
    sequence_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceRequest:
    population: str
    sequence_id: str
    dataset_index: int
    master_sequence_id: str
    reference_scene_id: str
    order_id: str
    scan_ids: tuple[str, str]
    scan_indices: tuple[int, int]
    change_file: Path | None


@dataclass
class FrozenRuntime:
    config: object
    full_dataset: object
    bridge_dataset: object
    collate: object
    device: torch.device
    system: object
    binding: RuntimeBinding


class RequestDataset(Dataset):
    """Expose one frozen population through the P2 validation DataLoader."""

    def __init__(
        self,
        base_dataset: object,
        requests: Sequence[SequenceRequest],
    ) -> None:
        self.base_dataset = base_dataset
        self.requests = tuple(requests)

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> object:
        request = self.requests[index]
        if request.population == "full154_t2":
            return self.base_dataset[request.dataset_index]
        sample = list(
            self.base_dataset.load_scan_indices(
                request.dataset_index,
                request.scan_indices,
                change_file=str(request.change_file),
            )
        )
        sample[3] = request.sequence_id
        return tuple(sample)


def _true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def build_evaluation_plan(
    full_sequence_ids: Sequence[str],
    bridge_rows: Sequence[Mapping[str, object]],
    *,
    seeds: Sequence[int] = EVALUATION_SEEDS,
) -> tuple[EvaluationGroup, ...]:
    """Freeze the six PB1 groups before model inference."""

    full = tuple(full_sequence_ids)
    if (
        len(full) != 154
        or len(set(full)) != 154
        or any(not isinstance(value, str) or not value for value in full)
    ):
        raise ProtocolBridgeEvaluationError(
            "full T2 population must contain 154 unique supervised sequences"
        )
    if len(bridge_rows) != 43:
        raise ProtocolBridgeEvaluationError("bridge population must contain 43 rows")
    masters: set[str] = set()
    sequences: set[str] = set()
    clusters: set[str] = set()
    for row in bridge_rows:
        if not isinstance(row, Mapping):
            raise ProtocolBridgeEvaluationError("bridge rows must be mappings")
        master = row.get("master_sequence_id")
        sequence = row.get("sequence_id")
        cluster = row.get("reference_scene_id")
        if not all(isinstance(value, str) and value for value in (master, sequence, cluster)):
            raise ProtocolBridgeEvaluationError("bridge row identities are invalid")
        if not _true(row.get("exact_ordered_pair")) or not _true(
            row.get("validation_supervised")
        ):
            raise ProtocolBridgeEvaluationError("bridge row is not exact and supervised")
        if any(
            _true(row.get(field))
            for field in (
                "pair_substituted",
                "reverse_pair_substituted",
                "future_stage_leakage",
            )
        ):
            raise ProtocolBridgeEvaluationError("bridge row violates PB0")
        masters.add(master)
        sequences.add(sequence)
        clusters.add(cluster)
    if len(masters) != 43 or len(sequences) != 43 or len(clusters) != 6:
        raise ProtocolBridgeEvaluationError("bridge coverage must be 43 masters / 6 clusters")
    normalized_seeds = tuple(seeds)
    if normalized_seeds != EVALUATION_SEEDS:
        raise ProtocolBridgeEvaluationError("evaluation seeds differ from registration")
    groups = []
    for seed in normalized_seeds:
        groups.extend(
            (
                EvaluationGroup(seed, "full154_t2", "official", 2, 154),
                EvaluationGroup(seed, "bridge43_canonical_t2", "canonical", 2, 43),
            )
        )
    return tuple(groups)


def validate_group_cache(
    payload: object,
    *,
    binding: RuntimeBinding,
    expected_group: EvaluationGroup | None = None,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ProtocolBridgeEvaluationError("group cache schema differs")
    if payload.get("runtime_binding") != binding.as_dict():
        raise ProtocolBridgeEvaluationError("group cache runtime binding differs")
    group = payload.get("group")
    records = payload.get("records")
    if not isinstance(group, Mapping) or not isinstance(records, list):
        raise ProtocolBridgeEvaluationError("group cache structure differs")
    if expected_group is not None and dict(group) != expected_group.as_dict():
        raise ProtocolBridgeEvaluationError("group cache identity differs")
    count = group.get("sequence_count")
    if isinstance(count, bool) or not isinstance(count, int) or len(records) != count:
        raise ProtocolBridgeEvaluationError("group cache sequence coverage differs")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ProtocolBridgeEvaluationError(f"JSON root must be a mapping: {path}")
    return value


def _runtime_binding(config: object) -> RuntimeBinding:
    from omegaconf import OmegaConf

    runtime_bytes = OmegaConf.to_yaml(config, resolve=True, sort_keys=True).encode()
    return RuntimeBinding(
        checkpoint_sha256=_sha256(CHECKPOINT),
        runtime_config_sha256=hashlib.sha256(runtime_bytes).hexdigest(),
        postprocess_sha256=_sha256(PROJECT_ROOT / "trainer/trainer.py"),
        metric_adapter_sha256=_sha256(PROJECT_ROOT / "scripts/p6a_metrics.py"),
        metric_spec_sha256=_sha256(METRIC_SPEC),
        min_region_size=100,
        precision=str(config.trainer.precision),
        batch_size=int(config.data.validation_dataloader.batch_size),
        num_workers=int(config.data.validation_dataloader.num_workers),
    )


def _build_runtime(device_name: str) -> FrozenRuntime:
    import hydra
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import (
        _compose_runtime_config,
        _load_system,
        _resolve_checkpoint,
        _validate_cuda_device,
    )

    config, _memory = _compose_runtime_config()
    if bool(config.general.train_mode):
        raise ProtocolBridgeEvaluationError("runtime must be in evaluation mode")
    if int(config.model.config.temporal_window) != 2:
        raise ProtocolBridgeEvaluationError("model temporal window must remain T2")
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    full_dataset = hydra.utils.instantiate(dataset_config)
    bridge_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    bridge_config.temporal_window = 5
    bridge_dataset = hydra.utils.instantiate(bridge_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    device = _validate_cuda_device(device_name)
    checkpoint = _resolve_checkpoint(CHECKPOINT)
    system = _load_system(config, checkpoint, device)
    system.validation_dataset = full_dataset
    return FrozenRuntime(
        config=config,
        full_dataset=full_dataset,
        bridge_dataset=bridge_dataset,
        collate=collate,
        device=device,
        system=system,
        binding=_runtime_binding(config),
    )


def _full_requests(dataset: object) -> tuple[SequenceRequest, ...]:
    names = tuple(dataset.sequence_names)
    indices = dataset.sequence_indices
    changes = tuple(dataset.change_files)
    if len(names) != 154 or len(changes) != 154:
        raise ProtocolBridgeEvaluationError("official full T2 dataset is not 154")
    requests = []
    for index, name in enumerate(names):
        scan_ids = tuple(name.split("-"))
        scan_indices = tuple(int(value) for value in indices[index])
        if len(scan_ids) != 2 or len(scan_indices) != 2:
            raise ProtocolBridgeEvaluationError("official T2 request is malformed")
        change = changes[index]
        if not isinstance(change, str) or change in {"", "None"}:
            raise ProtocolBridgeEvaluationError("official T2 request is unsupervised")
        requests.append(
            SequenceRequest(
                population="full154_t2",
                sequence_id=name,
                dataset_index=index,
                master_sequence_id="",
                reference_scene_id="",
                order_id="official",
                scan_ids=(scan_ids[0], scan_ids[1]),
                scan_indices=(scan_indices[0], scan_indices[1]),
                change_file=PROJECT_ROOT / change,
            )
        )
    return tuple(requests)


def _bridge_requests(
    rows: Sequence[Mapping[str, str]],
    protocol: Mapping[str, object],
    bridge_database: Mapping[str, object],
) -> tuple[SequenceRequest, ...]:
    masters_raw = protocol.get("masters")
    if not isinstance(masters_raw, list):
        raise ProtocolBridgeEvaluationError("Protocol B masters are unavailable")
    masters = {
        item.get("master_sequence_id"): item
        for item in masters_raw
        if isinstance(item, Mapping)
    }
    requests = []
    for row in rows:
        master_id = row["master_sequence_id"]
        master = masters.get(master_id)
        record = bridge_database.get(row["sequence_id"])
        if not isinstance(master, Mapping) or not isinstance(record, Mapping):
            raise ProtocolBridgeEvaluationError("bridge request lacks frozen sources")
        orders = master.get("orders")
        canonical = orders.get("canonical") if isinstance(orders, Mapping) else None
        prefixes = canonical.get("prefixes") if isinstance(canonical, Mapping) else None
        prefix = prefixes.get("2") if isinstance(prefixes, Mapping) else None
        if not isinstance(prefix, Mapping):
            raise ProtocolBridgeEvaluationError("canonical T2 prefix is unavailable")
        scan_ids = tuple(prefix.get("scan_ids", ()))
        scan_indices = tuple(prefix.get("scan_indices", ()))
        if (
            scan_ids != (row["scan_id_1"], row["scan_id_2"])
            or tuple(str(value) for value in scan_indices)
            != (row["scan_index_1"], row["scan_index_2"])
        ):
            raise ProtocolBridgeEvaluationError("bridge inventory differs from Protocol B")
        change = record.get("filepath")
        if not isinstance(change, str) or not (PROJECT_ROOT / change).is_file():
            raise ProtocolBridgeEvaluationError("bridge change target is unavailable")
        validation_index = master.get("validation_index")
        if isinstance(validation_index, bool) or not isinstance(validation_index, int):
            raise ProtocolBridgeEvaluationError("bridge validation index is invalid")
        requests.append(
            SequenceRequest(
                population="bridge43_canonical_t2",
                sequence_id=row["sequence_id"],
                dataset_index=validation_index,
                master_sequence_id=master_id,
                reference_scene_id=row["reference_scene_id"],
                order_id="canonical",
                scan_ids=(str(scan_ids[0]), str(scan_ids[1])),
                scan_indices=(int(scan_indices[0]), int(scan_indices[1])),
                change_file=PROJECT_ROOT / change,
            )
        )
    if len(requests) != 43:
        raise ProtocolBridgeEvaluationError("bridge request coverage differs")
    return tuple(requests)


def _pack_matrix(value: Tensor) -> dict[str, object]:
    from scripts.system_comparison_inference import pack_bool_matrix

    return pack_bool_matrix(value.detach().cpu().bool().contiguous())


def _record(
    request: SequenceRequest,
    prediction: Mapping[str, object],
    target: Mapping[str, object],
) -> dict[str, object]:
    required_prediction = ("pred_masks", "pred_scores", "pred_classes")
    required_target = ("masks", "labels", "ids", "changes", "temporal_stages")
    if any(not isinstance(prediction.get(key), Tensor) for key in required_prediction):
        raise ProtocolBridgeEvaluationError("official prediction fields differ")
    if any(not isinstance(target.get(key), Tensor) for key in required_target):
        raise ProtocolBridgeEvaluationError("official target fields differ")
    pred_masks = prediction["pred_masks"].detach().cpu().bool()
    pred_scores = prediction["pred_scores"].detach().cpu().float()
    pred_classes = prediction["pred_classes"].detach().cpu().long()
    masks = target["masks"].detach().cpu().bool()
    labels = target["labels"].detach().cpu().long()
    ids = target["ids"].detach().cpu().long()
    changes = target["changes"].detach().cpu().long()
    stages = target["temporal_stages"].detach().cpu().long()
    if set(stages.tolist()) != {0, 1}:
        raise ProtocolBridgeEvaluationError("T2 target stages differ")
    if pred_masks.shape[0] != stages.numel() or masks.shape[1] != stages.numel():
        raise ProtocolBridgeEvaluationError("prediction/target point coverage differs")
    if pred_masks.shape[1] != pred_scores.numel() or pred_scores.shape != pred_classes.shape:
        raise ProtocolBridgeEvaluationError("prediction candidate coverage differs")
    if masks.shape[0] != labels.numel() or labels.shape != ids.shape or ids.shape != changes.shape:
        raise ProtocolBridgeEvaluationError("target instance coverage differs")
    return {
        "key": {
            "population": request.population,
            "sequence_id": request.sequence_id,
            "master_sequence_id": request.master_sequence_id,
            "reference_scene_id": request.reference_scene_id,
            "order_id": request.order_id,
            "horizon": 2,
            "scan_ids": list(request.scan_ids),
            "scan_indices": list(request.scan_indices),
        },
        "prediction": {
            "pred_masks": _pack_matrix(pred_masks),
            "pred_scores": pred_scores,
            "pred_classes": pred_classes,
        },
        "target": {
            "masks": _pack_matrix(masks),
            "labels": labels,
            "ids": ids,
            "changes": changes,
            "temporal_stages": stages,
        },
    }


def _infer_batch(
    runtime: FrozenRuntime,
    batch: object,
    requests: Sequence[SequenceRequest],
) -> list[dict[str, object]]:
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
    )

    if not isinstance(batch, Sequence) or len(batch) != 3:
        raise ProtocolBridgeEvaluationError("validation DataLoader batch differs")
    data, targets, file_names = batch
    if list(file_names) != [request.sequence_id for request in requests]:
        raise ProtocolBridgeEvaluationError("validation DataLoader order differs")
    target_full = data.target_full
    inverse_maps = data.inverse_maps
    original_colors = data.original_colors
    original_normals = data.original_normals
    original_coordinates = data.original_coordinates
    data_indices = data.idx
    data = _move_data_to_device(data, runtime.device)
    targets = _move_targets_to_device(targets, runtime.device)
    raw_coordinates = runtime.system._process_raw_coordinates(data)
    with torch.inference_mode():
        output = runtime.system(
            data,
            point2segment=[target["point2segment"] for target in targets],
            raw_coordinates=raw_coordinates,
            is_eval=True,
            targets=targets,
        )
        with torch.amp.autocast("cuda", enabled=False):
            predictions = runtime.system._process_predictions(
                output=output,
                target_low_res=targets,
                target_full_res=target_full,
                inverse_maps=inverse_maps,
                file_names=file_names,
                full_res_coords=original_coordinates,
                original_colors=original_colors,
                original_normals=original_normals,
                raw_coords=None,
                idx=data_indices,
            )
    if len(predictions) != len(requests) or len(target_full) != len(requests):
        raise ProtocolBridgeEvaluationError("batch output coverage differs")
    return [
        _record(request, prediction, target)
        for request, prediction, target in zip(
            requests, predictions, target_full, strict=True
        )
    ]


def _cache_path(cache_root: Path, group: EvaluationGroup) -> Path:
    return cache_root / f"seed{group.seed}_{group.population}.pt"


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolBridgeEvaluationError("refusing symbolic-link cache output")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_group(
    runtime: FrozenRuntime,
    group: EvaluationGroup,
    requests: Sequence[SequenceRequest],
    *,
    cache_root: Path,
    evidence_binding: Mapping[str, str],
) -> Mapping[str, object]:
    from scripts.system_comparison_inference import deterministic_inference_runtime

    path = _cache_path(cache_root, group)
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        return validate_group_cache(
            cached, binding=runtime.binding, expected_group=group
        )
    if len(requests) != group.sequence_count:
        raise ProtocolBridgeEvaluationError("inference request coverage differs")
    dataset = RequestDataset(
        runtime.full_dataset
        if group.population == "full154_t2"
        else runtime.bridge_dataset,
        requests,
    )
    import hydra

    loader = hydra.utils.instantiate(
        runtime.config.data.validation_dataloader,
        dataset,
        collate_fn=runtime.collate,
    )
    records: list[dict[str, object]] = []
    with deterministic_inference_runtime(group.seed, runtime.device):
        for index, batch in enumerate(loader):
            start = index * runtime.binding.batch_size
            batch_requests = requests[start : start + runtime.binding.batch_size]
            records.extend(_infer_batch(runtime, batch, batch_requests))
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "group": group.as_dict(),
        "runtime_binding": runtime.binding.as_dict(),
        "evidence_binding": dict(evidence_binding),
        "known_empty_scan_substitution_count": 0,
        "records": records,
    }
    validate_group_cache(payload, binding=runtime.binding, expected_group=group)
    _atomic_torch_save(path, payload)
    return payload


def _pair(record: Mapping[str, object]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    from scripts.system_comparison_inference import unpack_bool_matrix

    prediction = record.get("prediction")
    target = record.get("target")
    if not isinstance(prediction, Mapping) or not isinstance(target, Mapping):
        raise ProtocolBridgeEvaluationError("cached task pair differs")
    return (
        {
            "pred_masks": unpack_bool_matrix(prediction["pred_masks"]),
            "pred_scores": prediction["pred_scores"].detach().cpu().float(),
            "pred_classes": prediction["pred_classes"].detach().cpu().long(),
        },
        {
            "masks": unpack_bool_matrix(target["masks"]),
            "labels": target["labels"].detach().cpu().long(),
            "ids": target["ids"].detach().cpu().long(),
            "changes": target["changes"].detach().cpu().long(),
            "temporal_stages": target["temporal_stages"].detach().cpu().long(),
        },
    )


def _metric_rows(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    from scripts.p6a_metrics import OfficialMetricAccumulator

    temporal = OfficialMetricAccumulator(mode="strict_online")
    local = OfficialMetricAccumulator(mode="raw_local")
    for record in records:
        prediction, target = _pair(record)
        temporal.update(prediction, target)
        local.update(prediction, target)
    temporal_values = temporal.compute()
    local_values = local.compute()
    result = {
        "t_mAP": temporal_values["online_t-mAP"],
        "t_mAP50": temporal_values["online_t-mAP50"],
        "t_mAP25": temporal_values["online_t-mAP25"],
        "t_REC": temporal_values["online_t-REC"],
        "t_REC50": temporal_values["online_t-REC50"],
        "t_REC25": temporal_values["online_t-REC25"],
        "local_current_AP": local_values["raw_local_AP"],
        "local_current_AP50": local_values["raw_local_AP50"],
        "local_current_AP25": local_values["raw_local_AP25"],
        "local_current_REC": local_values["raw_local_REC"],
    }
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in result.values()):
        raise ProtocolBridgeEvaluationError("official metrics are outside [0, 1]")
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ProtocolBridgeEvaluationError(f"refusing empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _emit_metrics(
    payloads: Sequence[Mapping[str, object]],
    *,
    output_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aggregate_rows: list[dict[str, object]] = []
    per_sequence_rows: list[dict[str, object]] = []
    for payload in payloads:
        group = payload["group"]
        records = payload["records"]
        if not isinstance(group, Mapping) or not isinstance(records, list):
            raise ProtocolBridgeEvaluationError("cache payload differs during metrics")
        pooled = _metric_rows(records)
        cluster_count = len(
            {
                record["key"]["reference_scene_id"]
                for record in records
                if record["key"]["reference_scene_id"]
            }
        )
        aggregate_rows.append(
            {
                "scope": "pooled",
                "method": METHOD,
                "seed": group["seed"],
                "population": group["population"],
                "order_id": group["order_id"],
                "horizon": group["horizon"],
                "sequence_count": group["sequence_count"],
                "reference_cluster_count": cluster_count or "",
                **pooled,
            }
        )
        if group["population"] != "bridge43_canonical_t2":
            continue
        by_cluster: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            key = record["key"]
            metrics = _metric_rows([record])
            per_sequence_rows.append(
                {
                    "method": METHOD,
                    "seed": group["seed"],
                    "population": group["population"],
                    "reference_scene_id": key["reference_scene_id"],
                    "master_sequence_id": key["master_sequence_id"],
                    "sequence_id": key["sequence_id"],
                    "order_id": key["order_id"],
                    "horizon": key["horizon"],
                    "scan_id_1": key["scan_ids"][0],
                    "scan_id_2": key["scan_ids"][1],
                    **metrics,
                }
            )
            by_cluster[str(key["reference_scene_id"])].append(record)
        if len(by_cluster) != 6:
            raise ProtocolBridgeEvaluationError("bridge cache lacks six clusters")
        for cluster, cluster_records in sorted(by_cluster.items()):
            aggregate_rows.append(
                {
                    "scope": "reference_cluster",
                    "method": METHOD,
                    "seed": group["seed"],
                    "population": group["population"],
                    "order_id": group["order_id"],
                    "horizon": group["horizon"],
                    "sequence_count": len(cluster_records),
                    "reference_cluster_count": 1,
                    "reference_scene_id": cluster,
                    **_metric_rows(cluster_records),
                }
            )
    _write_csv(output_root / "bridge_per_sequence.csv", per_sequence_rows)
    _write_csv(output_root / "bridge_aggregate.csv", aggregate_rows)
    return per_sequence_rows, aggregate_rows


def run_evaluation(
    *,
    device_name: str,
    cache_root: Path,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    import yaml

    rows = _read_csv(BRIDGE_INVENTORY)
    protocol = _load_json(PROTOCOL_MANIFEST)
    with BRIDGE_DATABASE.open(encoding="utf-8") as handle:
        bridge_database = yaml.safe_load(handle)
    if not isinstance(bridge_database, Mapping):
        raise ProtocolBridgeEvaluationError("bridge database root differs")
    runtime = _build_runtime(device_name)
    full_requests = _full_requests(runtime.full_dataset)
    bridge_requests = _bridge_requests(rows, protocol, bridge_database)
    plan = build_evaluation_plan(
        [request.sequence_id for request in full_requests], rows
    )
    evidence_binding = {
        "protocol_sha256": _sha256(PROTOCOL_MANIFEST),
        "bridge_inventory_sha256": _sha256(BRIDGE_INVENTORY),
        "bridge_database_sha256": _sha256(BRIDGE_DATABASE),
    }
    payloads = []
    for group in plan:
        requests = (
            full_requests if group.population == "full154_t2" else bridge_requests
        )
        payloads.append(
            _run_group(
                runtime,
                group,
                requests,
                cache_root=cache_root,
                evidence_binding=evidence_binding,
            )
        )
    per_sequence, aggregate = _emit_metrics(payloads, output_root=output_root)
    return {
        "status": "pass",
        "runtime_binding": runtime.binding.as_dict(),
        "group_count": len(payloads),
        "bridge_per_sequence_count": len(per_sequence),
        "aggregate_count": len(aggregate),
        "cache_root": str(cache_root.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    print(
        json.dumps(
            run_evaluation(
                device_name=arguments.device,
                cache_root=arguments.cache_root,
                output_root=arguments.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
