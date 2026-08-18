"""P6-A frozen prediction cache and CPU-only evaluation orchestration."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import sys
import tempfile
from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.p6a_analysis import (
    AssociationEvent,
    CapacitySnapshot,
    classify_failure,
    persistent_state_bytes,
    validate_association_events,
)
from scripts.p6a_association import (
    B0SanityTracker,
    B0StageUniqueTracker,
    B1FeatureTracker,
    B2FeatureClassTracker,
    B3EmaTracker,
    B4PersistentTracker,
    FrozenObservation,
    OracleStageTarget,
    freeze_observation,
    run_oracle_posthoc,
)
from scripts.p6a_cache import (
    CHANGE_LABEL_SEMANTICS,
    ENTRY_KEYS,
    KEY_KEYS,
    SCHEMA_VERSION,
    build_cache_manifest,
    discover_cache_entries,
    load_cache_entry,
    load_cache_manifest,
    validate_cache_entry,
    validate_cache_payload,
    write_cache_entry,
    write_cache_manifest,
)
from scripts.p6a_metrics import (
    IdentityAccumulator,
    OfficialMetricAccumulator,
    assert_shared_raw_predictions,
    build_offline_reconstructed_prediction,
    build_online_endpoint_prediction,
    match_instances_hungarian,
)

_EXPECTED_ORDER_NAMES = ("canonical", "reverse", "sha256_seed45")
EXPECTED_RESCENE_CHECKPOINT_SHA256 = (
    "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e"
)
_OBSERVATION_KEYS = (
    "features",
    "class_prob",
    "confidence",
    "valid",
    "masks",
)


def _field(value: object, name: str, *, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")  # noqa: TRY004
    return value


def _clone_cpu(value: object, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")  # noqa: TRY004
    result = value.detach().cpu().clone()
    if not result.requires_grad:
        return result
    return result.requires_grad_(False)


def _finite_tensor(value: object, *, name: str, ndim: int | None = None) -> Tensor:
    result = _clone_cpu(value, name=name)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise ValueError(f"{name} must contain finite values")
    return result


def _integer_tensor(value: object, *, name: str, ndim: int) -> Tensor:
    result = _finite_tensor(value, name=name, ndim=ndim)
    try:
        torch.iinfo(result.dtype)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"{name} must use an integer dtype") from error
    return result


def _observation_mapping(payload: object) -> Mapping[str, Any]:
    root = _require_mapping(payload, name="payload")
    observation = root.get("observation", root)
    observation = _require_mapping(observation, name="observation")
    missing = [key for key in _OBSERVATION_KEYS if key not in observation]
    if missing:
        raise ValueError(f"observation is missing {missing}")
    return observation


def _stage_index(payload: object, *, fallback: int | None = None) -> int:
    root = _require_mapping(payload, name="payload")
    key = root.get("key")
    value = _field(key, "stage_index") if key is not None else None
    if value is None:
        value = root.get("stage_index", root.get("stage"))
    if value is None:
        value = fallback
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stage_index must be a non-negative integer")
    return int(value)


def _as_scan_ids(value: object, *, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of scan IDs")  # noqa: TRY004
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain non-empty strings")
        result.append(item)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _protocol_orders(protocol: object) -> tuple[str, ...]:
    raw = _field(protocol, "order_variants")
    if raw is None:
        variants = _field(protocol, "variants")
        if isinstance(variants, Mapping) and variants:
            first = next(iter(variants.values()))
            raw = tuple(first) if isinstance(first, Mapping) else None
    if raw is None:
        raw = _EXPECTED_ORDER_NAMES
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("protocol order_variants must be a sequence")  # noqa: TRY004
    result = tuple(raw)
    if result != _EXPECTED_ORDER_NAMES:
        raise ValueError(
            "Protocol B requires canonical, reverse, and sha256_seed45 orders"
        )
    return result


def _protocol_masters(protocol: object) -> tuple[object, ...]:
    masters = _field(protocol, "masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise ValueError("protocol masters must be a sequence")  # noqa: TRY004
    if not masters:
        raise ValueError("protocol masters must not be empty")
    return tuple(masters)


def _protocol_variant(protocol: object, master: object, order: str) -> object:
    variants = _field(protocol, "variants")
    if not isinstance(variants, Mapping):
        raise ValueError("protocol must expose variants")  # noqa: TRY004
    master_id = _field(master, "sequence_id")
    if not isinstance(master_id, str) or not master_id:
        raise ValueError("master sequence_id must be a non-empty string")
    if master_id not in variants:
        raise ValueError(f"protocol variants missing master {master_id!r}")
    by_order = variants[master_id]
    if not isinstance(by_order, Mapping) or order not in by_order:
        raise ValueError(f"protocol variants missing order {order!r}")
    return by_order[order]


def expected_cache_keys(protocol: object) -> list[dict[str, object]]:
    """Build the exact master x order x stage cache-key coverage."""

    orders = _protocol_orders(protocol)
    keys: list[dict[str, object]] = []
    for master in _protocol_masters(protocol):
        master_id = _field(master, "sequence_id")
        reference_id = _field(master, "reference_scene_id")
        if not isinstance(master_id, str) or not master_id:
            raise ValueError("master sequence_id must be a non-empty string")
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError("master reference_scene_id must be a non-empty string")
        for order in orders:
            variant = _protocol_variant(protocol, master, order)
            scan_ids = _as_scan_ids(
                _field(variant, "scan_ids"), name=f"{master_id}/{order}/scan_ids"
            )
            if len(scan_ids) != 5:
                raise ValueError(
                    "each Protocol B order must contain exactly five scans"
                )
            for stage in range(5):
                history = scan_ids[: stage + 1]
                local_window = history[-1:] if stage == 0 else history[-2:]
                keys.append(
                    {
                        "master_sequence_id": master_id,
                        "reference_scene_id": reference_id,
                        "order_id": order,
                        "stage_index": stage,
                        "history_scan_ids": list(history),
                        "local_window_scan_ids": list(local_window),
                    }
                )
    expected_count = len(_protocol_masters(protocol)) * 3 * 5
    if len(keys) != expected_count or len(
        {json.dumps(key, sort_keys=True) for key in keys}
    ) != len(keys):
        raise ValueError("Protocol B cache-key coverage is not exact and unique")
    return keys


@dataclass(frozen=True)
class ProtocolCacheRequest:
    context_index: int
    master_sequence_id: str
    reference_scene_id: str
    order_id: str
    stage_index: int
    scan_indices: tuple[int, ...]


@contextmanager
def _frozen_inference_seed(seed: int, device: torch.device):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [device.index] if device.type == "cuda" else []
    try:
        random.seed(seed)
        np.random.seed(seed)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@dataclass
class RealPredictionCacheProducer:
    """Run one frozen ReScene local window for one exact Protocol B key."""

    protocol: object
    provenance: Mapping[str, object]
    dataset: object
    collate: Callable[[list[object]], tuple[object, object, object]]
    system: object
    device: torch.device
    observation_settings: Mapping[str, object]
    move_data: Callable[[object, torch.device], object]
    move_targets: Callable[[object, torch.device], object]
    segment_stages: Callable[[Mapping[str, object]], Tensor]
    latest_masks: Callable[..., Tensor]
    observation_builder: Callable[..., object]
    seed: int = 45

    def __call__(self, logical_key: Mapping[str, object]) -> dict[str, object]:
        request = resolve_protocol_cache_request(self.protocol, logical_key)
        master = next(
            master
            for master in _protocol_masters(self.protocol)
            if _field(master, "sequence_id") == request.master_sequence_id
        )
        names = _field(self.dataset, "sequence_names")
        indices = _field(self.dataset, "sequence_indices")
        if (
            isinstance(names, (str, bytes))
            or not isinstance(names, Sequence)
            or request.context_index >= len(names)
            or names[request.context_index] != request.master_sequence_id
        ):
            raise ValueError("dataset context does not match the Protocol B master")
        if isinstance(indices, (str, bytes)) or not hasattr(indices, "__getitem__"):
            raise ValueError("dataset sequence indices are unavailable")
        try:
            dataset_indices = tuple(
                int(index) for index in indices[request.context_index]
            )
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError("dataset sequence indices are unavailable") from error
        master_indices = tuple(int(index) for index in _field(master, "scan_indices"))
        if dataset_indices != master_indices:
            raise ValueError("dataset scan indices differ from the Protocol B master")

        with _frozen_inference_seed(self.seed, self.device):
            sample = self.dataset.load_scan_indices(
                request.context_index,
                request.scan_indices,
                change_file=None,
            )
            data, targets, collated_names = self.collate([sample])
            if (
                not isinstance(targets, Sequence)
                or len(targets) != 1
                or list(collated_names) != [request.master_sequence_id]
            ):
                raise ValueError("collator changed the requested Protocol B sample")
            target_full = _field(data, "target_full")
            if (
                isinstance(target_full, (str, bytes))
                or not isinstance(target_full, Sequence)
                or len(target_full) != 1
                or not isinstance(target_full[0], Mapping)
            ):
                raise ValueError("collated data must contain one full-resolution target")
            full_target = target_full[0]
            data = self.move_data(data, self.device)
            targets = self.move_targets(targets, self.device)
            target = targets[0]
            if not isinstance(target, Mapping) or "point2segment" not in target:
                raise ValueError("collated target is missing point2segment")
            stages = self.segment_stages(target)
            if not isinstance(stages, Tensor) or stages.ndim != 1 or stages.numel() == 0:
                raise ValueError("segment stages must be a non-empty rank-1 tensor")
            latest_local_stage = int(stages.max().item())
            raw_coordinates = self.system._process_raw_coordinates(data)
            with torch.inference_mode():
                output = self.system(
                    data,
                    point2segment=[target["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
            if not isinstance(output, Mapping):
                raise TypeError("ReScene output must be a mapping")
            observation = self.observation_builder(
                output,
                [stages],
                latest_stage=latest_local_stage,
                **dict(self.observation_settings),
            )
            full_masks = self.latest_masks(
                self.system,
                output,
                target,
                data,
                latest_local_stage=latest_local_stage,
            )
        return cache_payload_from_inference(
            key=logical_key,
            provenance=self.provenance,
            observation=observation,
            full_masks=full_masks,
            full_target=full_target,
            latest_local_stage=latest_local_stage,
        )


def resolve_protocol_cache_request(
    protocol: object,
    logical_key: Mapping[str, object],
) -> ProtocolCacheRequest:
    """Resolve one exact cache key to dataset context and global scan indices."""

    key = _normalize_key(logical_key)
    masters = _protocol_masters(protocol)
    matches = [
        master
        for master in masters
        if _field(master, "sequence_id") == key["master_sequence_id"]
    ]
    if len(matches) != 1:
        raise ValueError("cache key does not identify one Protocol B master")
    master = matches[0]
    reference = _field(master, "reference_scene_id")
    if reference != key["reference_scene_id"]:
        raise ValueError("cache key reference scene differs from Protocol B")
    variant = _protocol_variant(protocol, master, str(key["order_id"]))
    scan_ids = _as_scan_ids(
        _field(variant, "scan_ids"), name="Protocol B variant scan_ids"
    )
    raw_indices = _field(variant, "scan_indices")
    if isinstance(raw_indices, (str, bytes)) or not isinstance(raw_indices, Sequence):
        raise TypeError("Protocol B variant scan_indices must be a sequence")
    scan_indices = tuple(raw_indices)
    if len(scan_indices) != len(scan_ids) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in scan_indices
    ):
        raise ValueError("Protocol B variant scan indices are invalid")
    stage = int(key["stage_index"])
    if key["history_scan_ids"] != scan_ids[: stage + 1]:
        raise ValueError("cache history is not the exact Protocol B order prefix")
    expected_local_ids = scan_ids[: stage + 1][-1 if stage == 0 else -2 :]
    if key["local_window_scan_ids"] != expected_local_ids:
        raise ValueError("cache local window differs from Protocol B")
    context_index = _field(master, "validation_index")
    if (
        isinstance(context_index, bool)
        or not isinstance(context_index, int)
        or context_index < 0
    ):
        raise ValueError("Protocol B master validation_index is invalid")
    id_to_index = dict(zip(scan_ids, scan_indices, strict=True))
    return ProtocolCacheRequest(
        context_index=context_index,
        master_sequence_id=str(key["master_sequence_id"]),
        reference_scene_id=str(reference),
        order_id=str(key["order_id"]),
        stage_index=stage,
        scan_indices=tuple(
            int(id_to_index[scan_id]) for scan_id in key["local_window_scan_ids"]
        ),
    )


def build_cache_provenance(
    *,
    source_commit: str,
    checkpoint_path: Path,
    config_documents: Mapping[str, bytes],
    protocol_manifest: Mapping[str, object],
) -> dict[str, str]:
    """Bind cache content to code, checkpoint, resolved configs, and dataset."""

    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise ValueError("source_commit must be a lowercase SHA-1")
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError("checkpoint_path must be a regular non-symlink file")
    if not isinstance(config_documents, Mapping) or not config_documents:
        raise ValueError("config_documents must be a non-empty mapping")
    config_hasher = hashlib.sha256()
    for name, content in sorted(config_documents.items()):
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise ValueError("config document names must be portable identifiers")
        if not isinstance(content, bytes) or not content:
            raise ValueError("config documents must contain non-empty bytes")
        config_hasher.update(name.encode("utf-8") + b"\0")
        config_hasher.update(len(content).to_bytes(8, "big") + content)
    try:
        protocol_bytes = json.dumps(
            protocol_manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("protocol_manifest must be canonical JSON data") from error
    return {
        "source_commit": source_commit,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "config_sha256": config_hasher.hexdigest(),
        "dataset_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
    }


def materialize_prediction_cache(
    *,
    protocol: object,
    cache_directory: Path,
    manifest_path: Path,
    provenance: Mapping[str, object],
    producer: Callable[..., Mapping[str, object]],
) -> dict[str, object]:
    """Resume exact Protocol B cache generation and publish only a full manifest."""

    expected = expected_cache_keys(protocol)
    existing = discover_cache_entries(
        cache_directory,
        expected_provenance=provenance,
    )
    expected_identities = {_key_identity(key) for key in expected}
    if any(_key_identity(entry["key"]) not in expected_identities for entry in existing):
        raise ValueError("cache_directory contains an unexpected logical key")
    partial_manifest: dict[str, object] = {
        "provenance": dict(provenance),
        "entries": existing,
    }
    entries = list(existing)
    for key in expected:
        resolution = resolve_cache_entry(
            cache_directory,
            key,
            partial_manifest,
            expected_provenance=provenance,
            producer=producer,
        )
        if not resolution.reused:
            entries.append(dict(resolution.entry))
            partial_manifest["entries"] = entries
    manifest = build_cache_manifest(
        entries,
        expected_keys=expected,
        expected_provenance=provenance,
        cache_directory=cache_directory,
    )
    write_cache_manifest(
        manifest_path,
        manifest,
        expected_keys=expected,
        expected_provenance=provenance,
        cache_directory=cache_directory,
    )
    return manifest


def load_cached_protocol_sequences(
    *,
    protocol: object,
    cache_directory: Path,
    manifest_path: Path,
) -> tuple[CachedProtocolSequence, ...]:
    """Load one validated five-stage sequence per Protocol B master and order."""

    expected = expected_cache_keys(protocol)
    try:
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("cache manifest cannot be decoded") from error
    manifest_root = _require_mapping(manifest_document, name="cache manifest")
    provenance = _require_mapping(
        manifest_root.get("provenance"), name="cache manifest provenance"
    )
    manifest = load_cache_manifest(
        manifest_path,
        expected_keys=expected,
        expected_provenance=provenance,
        cache_directory=cache_directory,
    )
    entries_by_key = {
        _key_identity(entry["key"]): entry for entry in _manifest_entries(manifest)
    }
    payloads_by_key: dict[str, Mapping[str, object]] = {}
    for key in expected:
        identity = _key_identity(key)
        entry = entries_by_key[identity]
        filename = entry["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("cache manifest filename must be a plain file name")
        payloads_by_key[identity] = validate_cache_entry(
            cache_directory / filename,
            entry,
            expected_provenance=provenance,
        )

    sequences = []
    for master in _protocol_masters(protocol):
        master_id = _field(master, "sequence_id")
        reference_id = _field(master, "reference_scene_id")
        for order in _protocol_orders(protocol):
            keys = [
                key
                for key in expected
                if key["master_sequence_id"] == master_id
                and key["order_id"] == order
            ]
            keys.sort(key=lambda key: int(key["stage_index"]))
            sequences.append(
                CachedProtocolSequence(
                    reference_scene_id=str(reference_id),
                    master_sequence_id=str(master_id),
                    order_id=order,
                    payloads=tuple(payloads_by_key[_key_identity(key)] for key in keys),
                )
            )
    return tuple(sequences)


def _repository_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _external_cache_directory(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return candidate
    raise ValueError("prediction cache directory must be outside the repository")


def _frozen_protocol_bundle(
    *,
    metadata_path: Path,
) -> tuple[object, dict[str, object], bytes]:
    from scripts.p6a_protocol import build_protocol_b, build_protocol_b_manifest

    config_path = PROJECT_ROOT / "conf/p6a/default.yaml"
    sequence_database = PROJECT_ROOT / (
        "data/processed/rio/sequence_database_sliding_5.yaml"
    )
    scan_metadata = PROJECT_ROOT / "data/processed/rio/validation_database.yaml"
    source_manifest = PROJECT_ROOT / "artifacts/environment/source_manifest.json"
    p6a_bytes = config_path.read_bytes()
    p6a_config = yaml.safe_load(p6a_bytes)
    if not isinstance(p6a_config, Mapping):
        raise ValueError("P6-A config must be a mapping")  # noqa: TRY004
    protocol_config = _require_mapping(
        p6a_config.get("protocol_b"), name="P6-A protocol_b config"
    )
    sources = _require_mapping(
        protocol_config.get("sources"), name="P6-A protocol sources"
    )
    for name, path, digest_key in (
        ("sequence database", sequence_database, "sequence_database_sha256"),
        ("scan metadata", scan_metadata, "scan_metadata_sha256"),
        ("3RScan metadata", metadata_path, "metadata_sha256"),
    ):
        expected_digest = sources.get(digest_key)
        if expected_digest != _file_sha256(path):
            raise ValueError(f"frozen {name} SHA-256 differs from P6-A config")
    protocol = build_protocol_b(
        sequence_database,
        scan_metadata,
        metadata_path=metadata_path,
        expected_split=str(protocol_config["split"]),
        expected_master_count=int(protocol_config["expected_master_count"]),
        expected_cluster_count=int(
            protocol_config["expected_reference_scene_clusters"]
        ),
        horizons=tuple(int(value) for value in protocol_config["horizons"]),
        seed=int(protocol_config["seed"]),
        require_supervised=bool(protocol_config["require_supervised"]),
        substitution_policy=str(protocol_config["substitution_policy"]),
    )
    configured_references = protocol_config.get("reference_scene_ids")
    actual_references = sorted(
        {_field(master, "reference_scene_id") for master in _protocol_masters(protocol)}
    )
    if configured_references != actual_references:
        raise ValueError("Protocol B reference-scene clusters differ from config")
    manifest = build_protocol_b_manifest(
        protocol,
        sequence_database_path=sequence_database,
        scan_metadata_path=scan_metadata,
        metadata_path=metadata_path,
        source_manifest_path=source_manifest,
        config_path=config_path,
        repository_root=PROJECT_ROOT,
    )
    if len(expected_cache_keys(protocol)) != 645:
        raise ValueError("Protocol B must contain exactly 645 cache observations")
    return protocol, manifest, p6a_bytes


def run_real_prediction_cache(
    *,
    cache_directory: Path,
    protocol_manifest_path: Path,
    cache_manifest_path: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    device_name: str,
) -> dict[str, object]:
    """Materialize the complete frozen ReScene cache on one CUDA device."""

    external_cache = _external_cache_directory(cache_directory)
    protocol_output = _repository_path(protocol_manifest_path)
    cache_output = _repository_path(cache_manifest_path)
    metadata = _repository_path(metadata_path)

    import hydra
    from omegaconf import OmegaConf

    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _begin_source_tree_contract,
        _compose_runtime_config,
        _finalize_source_tree_contract,
        _latest_full_resolution_masks,
        _load_system,
        _move_data_to_device,
        _move_targets_to_device,
        _resolve_checkpoint,
        _segment_stages,
        _validate_cuda_device,
    )

    guard = _begin_source_tree_contract(
        repo_root=PROJECT_ROOT,
        output_paths=(protocol_output, cache_output),
    )
    protocol, protocol_manifest, p6a_bytes = _frozen_protocol_bundle(
        metadata_path=metadata
    )
    publish_manifest_atomic(protocol_output, protocol_manifest)
    config, _memory_config = _compose_runtime_config()
    checkpoint = _resolve_checkpoint(checkpoint_path)
    if _file_sha256(checkpoint) != EXPECTED_RESCENE_CHECKPOINT_SHA256:
        raise ValueError("formal ReScene checkpoint SHA-256 differs from P6-A")
    runtime_bytes = OmegaConf.to_yaml(
        config,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")
    provenance = build_cache_provenance(
        source_commit=guard.source_commit,
        checkpoint_path=checkpoint,
        config_documents={"p6a": p6a_bytes, "runtime": runtime_bytes},
        protocol_manifest=protocol_manifest,
    )
    device = _validate_cuda_device(device_name)
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(config.data.validation_dataset, resolve=True)
    )
    dataset_config.temporal_window = 5
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    system = _load_system(config, checkpoint, device)
    p6a_config = yaml.safe_load(p6a_bytes)
    settings = p6a_config["baselines"]["b4"]
    producer = RealPredictionCacheProducer(
        protocol=protocol,
        provenance=provenance,
        dataset=dataset,
        collate=collate,
        system=system,
        device=device,
        observation_settings={
            "background_class": int(settings["background_class"]),
            "confidence_threshold": float(settings["confidence_threshold"]),
            "mask_threshold": float(settings["mask_threshold"]),
            "minimum_mask_support": int(settings["minimum_mask_support"]),
        },
        move_data=_move_data_to_device,
        move_targets=_move_targets_to_device,
        segment_stages=_segment_stages,
        latest_masks=_latest_full_resolution_masks,
        observation_builder=build_local_observation,
        seed=int(p6a_config["protocol_b"]["seed"]),
    )

    deterministic_enabled = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cuda_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        manifest = materialize_prediction_cache(
            protocol=protocol,
            cache_directory=external_cache,
            manifest_path=cache_output,
            provenance=provenance,
            producer=producer,
        )
        _finalize_source_tree_contract(guard)
        return manifest
    finally:
        torch.use_deterministic_algorithms(
            deterministic_enabled,
            warn_only=deterministic_warn_only,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = cuda_allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        torch.set_float32_matmul_precision(matmul_precision)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the frozen P6-A ReScene prediction cache."
    )
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=Path("artifacts/P6A/protocol_b_manifest.json"),
    )
    parser.add_argument(
        "--cache-manifest",
        type=Path,
        default=Path("artifacts/P6A/cache_manifest.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    manifest = run_real_prediction_cache(
        cache_directory=args.cache_directory,
        protocol_manifest_path=args.protocol_manifest,
        cache_manifest_path=args.cache_manifest,
        metadata_path=args.metadata,
        checkpoint_path=args.checkpoint,
        device_name=args.device,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "entry_count": manifest["entry_count"],
                "entries_sha256": manifest["entries_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def cache_payload_to_frozen_observation(
    payload: Mapping[str, object],
) -> FrozenObservation:
    """Convert one cache payload into a detached CPU observation."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload must be a mapping")  # noqa: TRY004
    if "schema_version" in payload:
        validate_cache_payload(payload)
    observation = _observation_mapping(payload)
    features = _finite_tensor(observation["features"], name="features", ndim=2)
    class_prob = _finite_tensor(observation["class_prob"], name="class_prob", ndim=2)
    confidence = _finite_tensor(observation["confidence"], name="confidence", ndim=1)
    valid = _clone_cpu(observation["valid"], name="valid")
    masks = _clone_cpu(observation["masks"], name="masks")
    if valid.dtype != torch.bool:
        raise ValueError("valid must use bool dtype")
    if masks.dtype != torch.bool or masks.ndim != 2:
        raise ValueError("masks must be a rank-2 bool tensor")
    query_count = features.shape[0]
    if (
        class_prob.shape[0] != query_count
        or confidence.shape[0] != query_count
        or valid.shape[0] != query_count
        or masks.shape[0] != query_count
    ):
        raise ValueError("observation tensors must agree on query count")
    if class_prob.shape[1] <= 1 or masks.shape[1] <= 0:
        raise ValueError("class and point dimensions must be positive")
    if torch.any(class_prob < 0).item():
        raise ValueError("class_prob must be non-negative")
    if torch.any((confidence < 0) | (confidence > 1)).item():
        raise ValueError("confidence must be within [0, 1]")
    frozen = FrozenObservation(
        features=features,
        class_prob=class_prob,
        confidence=confidence,
        valid=valid,
        # Trackers consume mask logits; cache masks are thresholded booleans.
        latest_mask=(masks.to(dtype=features.dtype),),
    )
    frozen.validate()
    return frozen


def cache_payload_from_inference(
    *,
    key: Mapping[str, object],
    provenance: Mapping[str, object],
    observation: object,
    full_masks: Tensor,
    full_target: Mapping[str, object],
    latest_local_stage: int,
) -> dict[str, object]:
    """Freeze one real ReScene stage into the Protocol B cache contract."""

    validate = getattr(observation, "validate", None)
    if not callable(validate):
        raise TypeError("observation must expose validate()")
    validate()
    if (
        isinstance(latest_local_stage, bool)
        or not isinstance(latest_local_stage, int)
        or latest_local_stage < 0
    ):
        raise ValueError("latest_local_stage must be a non-negative integer")

    features = _finite_tensor(
        _field(observation, "features"), name="features", ndim=3
    )
    class_prob = _finite_tensor(
        _field(observation, "class_prob"), name="class_prob", ndim=3
    )
    confidence = _finite_tensor(
        _field(observation, "confidence"), name="confidence", ndim=2
    )
    valid = _clone_cpu(_field(observation, "valid"), name="valid")
    if features.shape[0] != 1:
        raise ValueError("Protocol B cache generation requires batch size one")
    if (
        class_prob.shape[:2] != features.shape[:2]
        or confidence.shape != features.shape[:2]
        or valid.shape != features.shape[:2]
        or valid.dtype != torch.bool
    ):
        raise ValueError("observation batch and query dimensions must align")

    masks = _clone_cpu(full_masks, name="full_masks")
    if masks.dtype != torch.bool or masks.ndim != 2:
        raise ValueError("full_masks must be a rank-2 bool tensor")
    if masks.shape[0] != features.shape[1] or masks.shape[1] <= 0:
        raise ValueError("full_masks must have shape [Q, P_latest]")

    target = _require_mapping(full_target, name="full_target")
    for name in ("ids", "labels", "masks", "temporal_stages"):
        if name not in target:
            raise ValueError(f"full_target is missing {name}")
    gt_ids = _integer_tensor(target["ids"], name="full_target.ids", ndim=1)
    gt_classes = _integer_tensor(
        target["labels"], name="full_target.labels", ndim=1
    )
    gt_masks = _clone_cpu(target["masks"], name="full_target.masks")
    temporal_stages = _integer_tensor(
        target["temporal_stages"],
        name="full_target.temporal_stages",
        ndim=1,
    )
    if gt_masks.dtype != torch.bool or gt_masks.ndim != 2:
        raise ValueError("full_target.masks must be a rank-2 bool tensor")
    if (
        gt_ids.shape != gt_classes.shape
        or gt_masks.shape[0] != gt_ids.shape[0]
        or gt_masks.shape[1] != temporal_stages.shape[0]
    ):
        raise ValueError("full_target tensors must align")
    stage_selector = temporal_stages == latest_local_stage
    if int(stage_selector.sum().item()) != masks.shape[1]:
        raise ValueError("full masks and latest-stage target points must align")
    latest_gt_masks = gt_masks[:, stage_selector]
    present = latest_gt_masks.any(dim=1)

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "key": dict(key),
        "provenance": dict(provenance),
        "observation": {
            "features": features[0].clone(),
            "class_prob": class_prob[0].clone(),
            "confidence": confidence[0].clone(),
            "valid": valid[0].clone(),
            "masks": masks.clone(),
            "mask_support": masks.sum(dim=1, dtype=torch.long),
            "local_query_ids": torch.arange(masks.shape[0], dtype=torch.long),
        },
        "target": {
            "gt_ids": gt_ids[present].clone(),
            "gt_classes": gt_classes[present].clone(),
            "gt_masks": latest_gt_masks[present].clone(),
            "changes": torch.zeros(int(present.sum().item()), dtype=torch.long),
            "change_labels_valid": False,
            "change_label_semantics": CHANGE_LABEL_SEMANTICS,
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }
    validate_cache_payload(payload)
    return payload


def _stage_payloads(stage_payloads: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(stage_payloads, Mapping):
        values = list(stage_payloads.values())
    elif isinstance(stage_payloads, Sequence) and not isinstance(
        stage_payloads, (str, bytes)
    ):
        values = list(stage_payloads)
    else:
        raise ValueError("stage_payloads must be a sequence or stage mapping")  # noqa: TRY004
    if not values:
        raise ValueError("stage_payloads must not be empty")
    normalized = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("every stage payload must be a mapping")  # noqa: TRY004
        normalized.append(value)
    normalized.sort(key=lambda item: _stage_index(item))
    stage_ids = [_stage_index(item) for item in normalized]
    if stage_ids != list(range(len(normalized))):
        raise ValueError("stage payloads must cover contiguous stages from zero")
    return tuple(normalized)


def _target_mapping(payload: Mapping[str, object]) -> Mapping[str, object]:
    target = payload.get("target", payload)
    if not isinstance(target, Mapping):
        raise ValueError("target must be a mapping")  # noqa: TRY004
    for key in (
        "gt_ids",
        "gt_classes",
        "gt_masks",
        "changes",
        "change_labels_valid",
        "change_label_semantics",
        "gt_class_semantics",
    ):
        if key not in target:
            raise ValueError(f"target is missing {key}")
    return target


def build_temporal_target(
    stage_payloads: Sequence[Mapping[str, object]]
    | Mapping[object, Mapping[str, object]],
) -> dict[str, Tensor]:
    """Construct one full-prefix official metric target from stage cache entries."""

    payloads = _stage_payloads(stage_payloads)
    records: list[dict[str, Any]] = []
    entity_order: list[int] = []
    entity_index: dict[int, int] = {}
    class_values: dict[int, int] = {}
    change_values: dict[int, int] = {}
    for stage, payload in enumerate(payloads):
        target = _target_mapping(payload)
        if target["change_labels_valid"] is not False:
            raise ValueError("Protocol B change labels must be unavailable")
        if target["change_label_semantics"] != CHANGE_LABEL_SEMANTICS:
            raise ValueError("Protocol B change-label semantics differ")
        if target["gt_class_semantics"] != "rescene_model_index_0_based":
            raise ValueError("Protocol B GT class semantics differ")
        gt_ids = _integer_tensor(target["gt_ids"], name="gt_ids", ndim=1)
        gt_classes = _integer_tensor(target["gt_classes"], name="gt_classes", ndim=1)
        gt_masks = _clone_cpu(target["gt_masks"], name="gt_masks")
        if gt_masks.ndim != 2 or gt_masks.dtype != torch.bool:
            raise ValueError("gt_masks must be a rank-2 bool tensor")
        if (
            gt_classes.shape != gt_ids.shape
            or gt_masks.shape[0] != gt_ids.shape[0]
            or gt_masks.shape[1] <= 0
        ):
            raise ValueError("stage target tensors have incompatible shapes")
        ids = [int(value) for value in gt_ids.tolist()]
        if len(set(ids)) != len(ids):
            raise ValueError("gt_ids must be unique within a stage")
        changes = _integer_tensor(target["changes"], name="changes", ndim=1)
        if changes.shape != gt_ids.shape or torch.any(changes != 0).item():
            raise ValueError("changes must be an aligned all-static placeholder")
        for index, gt_id in enumerate(ids):
            class_value = int(gt_classes[index].item())
            change_value = int(changes[index].item())
            previous_class = class_values.get(gt_id)
            if previous_class is not None and previous_class != class_value:
                raise ValueError(f"GT {gt_id} has a class conflict")
            class_values[gt_id] = class_value
            if gt_id not in entity_index:
                entity_index[gt_id] = len(entity_order)
                entity_order.append(gt_id)
                change_values[gt_id] = change_value
            elif change_values[gt_id] != change_value:
                raise ValueError(f"GT {gt_id} has a change-label conflict")
        records.append({"ids": ids, "masks": gt_masks, "stage": stage})

    total_points = sum(int(record["masks"].shape[1]) for record in records)
    output_masks = torch.zeros(
        (len(entity_order), total_points), dtype=torch.bool, device="cpu"
    )
    temporal_stages: list[Tensor] = []
    offset = 0
    for record in records:
        stage_masks = record["masks"]
        point_count = stage_masks.shape[1]
        temporal_stages.append(
            torch.full((point_count,), record["stage"], dtype=torch.long)
        )
        for row, gt_id in enumerate(record["ids"]):
            output_masks[entity_index[gt_id], offset : offset + point_count] = (
                stage_masks[row]
            )
        offset += point_count
    return {
        "masks": output_masks.clone(),
        "labels": torch.tensor(
            [class_values[gt_id] for gt_id in entity_order], dtype=torch.long
        ),
        "ids": torch.tensor(entity_order, dtype=torch.long),
        "changes": torch.tensor(
            [change_values[gt_id] for gt_id in entity_order], dtype=torch.long
        ),
        "temporal_stages": torch.cat(temporal_stages).clone(),
    }


def _step_field(step: object, name: str, *, default: object = None) -> object:
    value = _field(step, name, default=default)
    if value is None and name == "track_ids":
        raise ValueError("track step is missing track_ids")
    return value


def _sequence_values(value: object, *, name: str) -> list[object]:
    if isinstance(value, Tensor):
        if value.ndim != 1:
            raise ValueError(f"{name} must have rank 1")
        return value.detach().cpu().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    raise ValueError(f"{name} must be a one-dimensional sequence")


def _track_id_token(value: object) -> Hashable:
    if value is None:
        return ("none",)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("track IDs must be finite")
    try:
        hash(value)
    except TypeError as error:
        raise ValueError("track IDs must be hashable") from error
    return (type(value).__name__, repr(value))


def _map_class(class_mapper: object, value: int) -> int:
    if class_mapper is None:
        return value
    if callable(class_mapper):
        mapped = class_mapper(value)
    elif isinstance(class_mapper, Mapping) and value in class_mapper:
        mapped = class_mapper[value]
    else:
        raise ValueError("class_mapper must be callable or map every predicted class")
    if isinstance(mapped, bool) or not isinstance(mapped, int):
        raise ValueError("class_mapper must return integer class labels")  # noqa: TRY004
    return int(mapped)


def stage_prediction_from_track_step(
    payload: Mapping[str, object],
    step: object,
    *,
    class_mapper: Callable[[int], int] | Mapping[int, int] | None = None,
    background_class: int = 18,
) -> dict[str, object]:
    """Convert a duck-typed tracker step into an official prediction mapping."""

    observation = _observation_mapping(payload)
    class_prob = _finite_tensor(observation["class_prob"], name="class_prob", ndim=2)
    confidence = _finite_tensor(observation["confidence"], name="confidence", ndim=1)
    valid_observation = _clone_cpu(observation["valid"], name="valid")
    masks = _clone_cpu(observation["masks"], name="masks")
    if (
        valid_observation.dtype != torch.bool
        or masks.dtype != torch.bool
        or masks.ndim != 2
    ):
        raise ValueError(
            "valid and masks must have bool dtype and masks must be rank 2"
        )
    query_count = int(class_prob.shape[0])
    if (
        confidence.shape != (query_count,)
        or valid_observation.shape != (query_count,)
        or masks.shape[0] != query_count
    ):
        raise ValueError("observation tensors must agree on query count")
    if (
        isinstance(background_class, bool)
        or not isinstance(background_class, int)
        or background_class < 0
        or background_class >= class_prob.shape[1]
    ):
        raise ValueError("background_class must index class_prob")
    if (
        torch.any(class_prob < 0).item()
        or torch.any((confidence < 0) | (confidence > 1)).item()
    ):
        raise ValueError("class_prob and confidence must be within valid ranges")

    track_ids = _sequence_values(_step_field(step, "track_ids"), name="track_ids")
    if len(track_ids) != query_count:
        raise ValueError("track step track_ids must agree on query count")
    query_property = _field(step, "query_count")
    if query_property is not None and query_property != query_count:
        raise ValueError("track step query_count does not match payload")
    step_valid_value = _field(step, "valid")
    if step_valid_value is None:
        step_valid = valid_observation
    else:
        step_valid_values = _sequence_values(step_valid_value, name="valid")
        if any(not isinstance(value, bool) for value in step_valid_values):
            raise ValueError("track step valid must contain boolean values")
        step_valid = torch.tensor(step_valid_values, dtype=torch.bool)
    if step_valid.shape != (query_count,):
        raise ValueError("track step valid must agree on query count")
    tokens = [_track_id_token(value) for value in track_ids if value is not None]
    if len(tokens) != len(set(tokens)):
        raise ValueError("duplicate track IDs are not allowed")
    stage = _stage_index(payload, fallback=_field(step, "stage_id"))
    step_stage = _field(step, "stage_id")
    if step_stage is not None:
        if (
            isinstance(step_stage, bool)
            or not isinstance(step_stage, int)
            or step_stage < 0
        ):
            raise ValueError("track step stage_id must be a non-negative integer")
        if step_stage != stage:
            raise ValueError("track step stage_id disagrees with payload")
    selected = [
        index
        for index, track_id in enumerate(track_ids)
        if track_id is not None
        and bool(valid_observation[index])
        and bool(step_valid[index])
    ]
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    foreground_prob = class_prob[selected_tensor].clone()
    foreground_prob[:, background_class] = -float("inf")
    predicted_classes = [
        _map_class(class_mapper, int(value))
        for value in torch.argmax(foreground_prob, dim=1).tolist()
    ]
    prediction_ids = [track_ids[index] for index in selected]
    if prediction_ids and all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in prediction_ids
    ):
        output_ids: object = torch.tensor(prediction_ids, dtype=torch.long)
    else:
        output_ids = tuple(prediction_ids)
    return {
        "stage": stage,
        "pred_masks": masks[selected_tensor].transpose(0, 1).contiguous().clone(),
        "pred_classes": torch.tensor(predicted_classes, dtype=torch.long),
        "pred_scores": confidence[selected_tensor].clone().float(),
        "class_probs": class_prob[selected_tensor].clone(),
        "track_ids": output_ids,
    }


def _tensor_digest(value: Tensor, hasher: hashlib._Hash) -> None:
    tensor = value.detach().cpu().contiguous()
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(tensor.numpy().tobytes(order="C"))


def observation_content_digest(observations: Sequence[FrozenObservation]) -> str:
    if not isinstance(observations, Sequence) or not observations:
        raise ValueError("observations must be a non-empty sequence")
    hasher = hashlib.sha256()
    for observation in observations:
        if not isinstance(observation, FrozenObservation):
            raise ValueError(  # noqa: TRY004
                "observations must contain FrozenObservation values"
            )
        observation.validate()
        for tensor in (
            observation.features,
            observation.class_prob,
            observation.confidence,
            observation.valid,
            *observation.latest_mask,
        ):
            _tensor_digest(tensor, hasher)
    return hasher.hexdigest()


def _factory_call(factory: object, *, method: str, sequence_id: str) -> object:
    if not callable(factory):
        raise ValueError(  # noqa: TRY004
            f"tracker factory for {method} must be callable"
        )
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = signature.parameters
    kwargs: dict[str, object] = {}
    if "method" in parameters:
        kwargs["method"] = method
    if "sequence_id" in parameters:
        kwargs["sequence_id"] = sequence_id
    if kwargs:
        return factory(**kwargs)
    required = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    if len(required) == 0:
        return factory()
    if len(required) == 1:
        return factory(method)
    if len(required) == 2:
        return factory(method, sequence_id)
    raise ValueError(f"tracker factory for {method} has unsupported signature")


def _tracker_step(
    tracker: object, observation: FrozenObservation, stage: int
) -> object:
    step = getattr(tracker, "step", None)
    if not callable(step):
        raise ValueError("tracker must provide callable step")  # noqa: TRY004
    try:
        signature = inspect.signature(step)
    except (TypeError, ValueError):
        return step(observation, stage)
    accepts_keyword = "stage_id" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_keyword:
        return step(observation, stage_id=stage)
    return step(observation, stage)


@dataclass(frozen=True)
class PrefixCausalityResult:
    """Online endpoint snapshots and a separately labeled offline run."""

    online: dict[str, dict[int, IdentityAccumulator]]
    offline: dict[str, IdentityAccumulator]
    online_predictions: dict[str, dict[int, dict[str, object]]]
    offline_predictions: dict[str, dict[str, object]]
    online_steps: dict[str, dict[int, tuple[object, ...]]]
    offline_steps: dict[str, tuple[object, ...]]
    content_digest: str
    endpoints: tuple[int, ...]

    def __getitem__(self, key: str) -> object:
        return getattr(self, key)


@dataclass(frozen=True)
class CachedProtocolSequence:
    reference_scene_id: str
    master_sequence_id: str
    order_id: str
    payloads: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        for name in ("reference_scene_id", "master_sequence_id", "order_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.order_id not in _EXPECTED_ORDER_NAMES:
            raise ValueError("order_id is not a Protocol B order")
        normalized = _stage_payloads(self.payloads)
        if len(normalized) != 5:
            raise ValueError("cached Protocol B sequences must contain five stages")
        for stage, payload in enumerate(normalized):
            validate_cache_payload(payload)
            key = _require_mapping(payload["key"], name="cache key")
            if (
                key["master_sequence_id"] != self.master_sequence_id
                or key["reference_scene_id"] != self.reference_scene_id
                or key["order_id"] != self.order_id
                or key["stage_index"] != stage
            ):
                raise ValueError("cache payload key differs from sequence identity")
        object.__setattr__(self, "payloads", normalized)


@dataclass(frozen=True)
class TaskMetricEvaluation:
    metric_blocks: dict[str, dict[str, dict[str, dict[str, float]]]]
    fingerprints: dict[str, dict[str, str]]
    sequence_count: int
    association_events: tuple[AssociationEvent, ...]
    capacity_snapshots: tuple[CapacitySnapshot, ...]


def prefix_causality_coordinator(
    stage_payloads: Sequence[Mapping[str, object]],
    tracker_factories: Mapping[str, object],
    *,
    endpoints: Sequence[int] | None = None,
    sequence_id: str = "p6a-sequence",
    background_class: int = 18,
) -> PrefixCausalityResult:
    """Run independent causal endpoint trackers and one full offline diagnostic."""

    payloads = _stage_payloads(stage_payloads)
    frozen = tuple(cache_payload_to_frozen_observation(payload) for payload in payloads)
    digest = observation_content_digest(frozen)
    if not isinstance(tracker_factories, Mapping) or not tracker_factories:
        raise ValueError("tracker_factories must be a non-empty mapping")
    methods = tuple(tracker_factories)
    if any(not isinstance(method, str) or not method for method in methods):
        raise ValueError("tracker method names must be non-empty strings")
    if endpoints is None:
        endpoint_values = tuple(range(1, len(payloads)))
    else:
        if isinstance(endpoints, (str, bytes)) or not isinstance(endpoints, Sequence):
            raise ValueError("endpoints must be a sequence")
        endpoint_values = tuple(endpoints)
    if not endpoint_values:
        raise ValueError("endpoints must not be empty")
    if any(
        isinstance(endpoint, bool)
        or not isinstance(endpoint, int)
        or endpoint < 0
        or endpoint >= len(payloads)
        for endpoint in endpoint_values
    ):
        raise ValueError("endpoints must be valid stage indices")
    if tuple(sorted(set(endpoint_values))) != endpoint_values:
        raise ValueError("endpoints must be sorted and unique")

    online: dict[str, dict[int, IdentityAccumulator]] = {
        method: {} for method in methods
    }
    online_predictions: dict[str, dict[int, dict[str, object]]] = {
        method: {} for method in methods
    }
    online_steps: dict[str, dict[int, tuple[object, ...]]] = {
        method: {} for method in methods
    }
    for method in methods:
        for endpoint in endpoint_values:
            tracker = _factory_call(
                tracker_factories[method], method=method, sequence_id=sequence_id
            )
            accumulator = IdentityAccumulator()
            steps = []
            for stage in range(endpoint + 1):
                observation = freeze_observation(frozen[stage])
                step = _tracker_step(tracker, observation, stage)
                steps.append(step)
                prediction = stage_prediction_from_track_step(
                    payloads[stage], step, background_class=background_class
                )
                accumulator.add_stage(prediction)
            online[method][endpoint] = accumulator
            online_steps[method][endpoint] = tuple(steps)
            online_predictions[method][endpoint] = build_online_endpoint_prediction(
                accumulator, endpoint=endpoint
            )

    offline: dict[str, IdentityAccumulator] = {}
    offline_predictions: dict[str, dict[str, object]] = {}
    offline_steps: dict[str, tuple[object, ...]] = {}
    for method in methods:
        tracker = _factory_call(
            tracker_factories[method], method=method, sequence_id=sequence_id
        )
        accumulator = IdentityAccumulator()
        steps = []
        for stage, payload in enumerate(payloads):
            observation = freeze_observation(frozen[stage])
            step = _tracker_step(tracker, observation, stage)
            steps.append(step)
            accumulator.add_stage(
                stage_prediction_from_track_step(
                    payload, step, background_class=background_class
                )
            )
        offline[method] = accumulator
        offline_steps[method] = tuple(steps)
        offline_predictions[method] = build_offline_reconstructed_prediction(
            accumulator
        )
    return PrefixCausalityResult(
        online=online,
        offline=offline,
        online_predictions=online_predictions,
        offline_predictions=offline_predictions,
        online_steps=online_steps,
        offline_steps=offline_steps,
        content_digest=digest,
        endpoints=endpoint_values,
    )


def _remap_metric_prediction(
    prediction: Mapping[str, object],
    class_mapper: Callable[[int], int] | Mapping[int, int],
) -> dict[str, object]:
    result = {
        key: value.detach().cpu().clone() if isinstance(value, Tensor) else value
        for key, value in prediction.items()
    }
    classes = _integer_tensor(
        result.get("pred_classes"), name="prediction pred_classes", ndim=1
    )
    result["pred_classes"] = torch.tensor(
        [_map_class(class_mapper, int(value)) for value in classes.tolist()],
        dtype=torch.long,
    )
    return result


def _remap_metric_target(
    target: Mapping[str, object],
    class_mapper: Callable[[int], int] | Mapping[int, int],
) -> dict[str, object]:
    result = {
        key: value.detach().cpu().clone() if isinstance(value, Tensor) else value
        for key, value in target.items()
    }
    labels = _integer_tensor(result.get("labels"), name="target labels", ndim=1)
    result["labels"] = torch.tensor(
        [_map_class(class_mapper, int(value)) for value in labels.tolist()],
        dtype=torch.long,
    )
    return result


def _raw_local_target(payload: Mapping[str, object]) -> dict[str, Tensor]:
    target = _target_mapping(payload)
    masks = _clone_cpu(target["gt_masks"], name="gt_masks")
    if masks.dtype != torch.bool or masks.ndim != 2:
        raise ValueError("gt_masks must be a rank-2 bool tensor")
    labels = _integer_tensor(target["gt_classes"], name="gt_classes", ndim=1)
    ids = _integer_tensor(target["gt_ids"], name="gt_ids", ndim=1)
    changes = _integer_tensor(target["changes"], name="changes", ndim=1)
    if any(value.shape[0] != masks.shape[0] for value in (labels, ids, changes)):
        raise ValueError("raw local target fields must share the GT dimension")
    return {
        "masks": masks,
        "labels": labels,
        "ids": ids,
        "changes": changes,
        "temporal_stages": torch.zeros(masks.shape[1], dtype=torch.long),
    }


def _official_metric_factory(
    mode: str, _method: str, _horizon: int
) -> OfficialMetricAccumulator:
    return OfficialMetricAccumulator(mode=mode)


def build_tracker_factories(
    config: Mapping[str, object],
) -> dict[str, Callable[[str], object]]:
    """Construct the exact preregistered B0-B4 tracker factory set."""

    root = _require_mapping(config, name="P6-A config")
    baselines = _require_mapping(root.get("baselines"), name="P6-A baselines")
    settings = {
        name: _require_mapping(baselines.get(name), name=f"baseline {name}")
        for name in ("b0", "b0_sanity", "b1", "b2", "b3", "b4")
    }
    b1 = settings["b1"]
    b2 = settings["b2"]
    b3 = settings["b3"]
    b4 = settings["b4"]
    return {
        "B0": lambda sequence_id: B0StageUniqueTracker(sequence_id=sequence_id),
        "B0_sanity": lambda sequence_id: B0SanityTracker(
            sequence_id=sequence_id
        ),
        "B1": lambda sequence_id: B1FeatureTracker(
            sequence_id=sequence_id,
            feature_threshold=float(b1["feature_threshold"]),
            class_weight=0.0,
            background_class=int(b2["background_class"]),
        ),
        "B2": lambda sequence_id: B2FeatureClassTracker(
            sequence_id=sequence_id,
            feature_threshold=float(b2["feature_threshold"]),
            class_weight=float(b2["class_weight"]),
            background_class=int(b2["background_class"]),
        ),
        "B3": lambda sequence_id: B3EmaTracker(
            sequence_id=sequence_id,
            feature_threshold=float(b3["feature_threshold"]),
            class_weight=float(b3["class_weight"]),
            background_class=int(b3["background_class"]),
            update_rate=float(b3["update_rate"]),
        ),
        "B4": lambda sequence_id: B4PersistentTracker(
            sequence_id=sequence_id,
            capacity=int(b4["capacity"]),
            association_threshold=float(b4["association_threshold"]),
            class_weight=float(b4["class_weight"]),
            update_rate=float(b4["update_rate"]),
            max_update_rate=float(b4["update_rate"]),
        ),
    }


def build_rio_class_mapper(
    dataset: object,
    *,
    foreground_class_count: int = 18,
) -> Callable[[int], int]:
    """Map ReScene model indices to raw RIO IDs through dataset semantics."""

    offset = getattr(dataset, "label_offset", None)
    remap = getattr(dataset, "_remap_model_output", None)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("RIO dataset label_offset must be an integer")
    if not callable(remap):
        raise TypeError("RIO dataset must expose _remap_model_output")
    if (
        isinstance(foreground_class_count, bool)
        or not isinstance(foreground_class_count, int)
        or foreground_class_count <= 0
    ):
        raise ValueError("foreground_class_count must be positive")

    def mapper(model_class: int) -> int:
        if (
            isinstance(model_class, bool)
            or not isinstance(model_class, int)
            or not 0 <= model_class < foreground_class_count
        ):
            raise ValueError("foreground model class is outside the registered range")
        mapped = remap(torch.tensor([model_class + offset], dtype=torch.long))
        if not isinstance(mapped, Tensor) or mapped.shape != (1,):
            raise ValueError("RIO class remapper must return one tensor value")
        return int(mapped.item())

    return mapper


def _event_identity(value: object) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return f"bool:{value!r}"
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value:
        return value
    return f"{type(value).__name__}:{value!r}"


def _diagnostic_value(step: object, name: str, query_index: int) -> object:
    diagnostics = _field(step, "diagnostics")
    values = _field(diagnostics, name) if diagnostics is not None else None
    if values is None:
        return None
    sequence = _sequence_values(values, name=f"diagnostics.{name}")
    if query_index >= len(sequence):
        raise ValueError("association diagnostics do not cover every query")
    return sequence[query_index]


def _binary_iou_matrix(gt_masks: Tensor, pred_masks: Tensor) -> Tensor:
    if gt_masks.ndim != 2 or pred_masks.ndim != 2:
        raise ValueError("instance masks must be rank-2")
    if gt_masks.shape[1] != pred_masks.shape[0]:
        raise ValueError("GT and prediction masks must share point dimension")
    gt = gt_masks.to(dtype=torch.float64)
    pred = pred_masks.transpose(0, 1).to(dtype=torch.float64)
    intersection = gt @ pred.transpose(0, 1)
    union = gt.sum(dim=1, keepdim=True) + pred.sum(dim=1).unsqueeze(0) - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(union))


def build_association_events(
    payloads: Sequence[Mapping[str, object]],
    steps: Sequence[object],
    *,
    method: str,
    reference_scene_id: str,
    master_sequence_id: str,
    order_id: str,
    prefix: int,
    cache_digest: str,
    background_class: int = 18,
    iou_threshold: float = 0.5,
) -> tuple[AssociationEvent, ...]:
    """Build reconstructable diagnostic events from one causal prefix run."""

    normalized = _stage_payloads(payloads)
    if len(normalized) != len(steps) or prefix != len(normalized):
        raise ValueError("payloads, steps, and prefix must describe one exact run")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a non-empty string")
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    sequence_id = f"{master_sequence_id}:{order_id}"
    gt_history: dict[int, tuple[int, str | int]] = {}
    identity_history: dict[str | int, int] = {}
    events: list[AssociationEvent] = []

    for stage, (payload, step) in enumerate(zip(normalized, steps, strict=True)):
        prediction = stage_prediction_from_track_step(
            payload, step, background_class=background_class
        )
        observation = _observation_mapping(payload)
        valid = _clone_cpu(observation["valid"], name="valid")
        track_ids = _sequence_values(_step_field(step, "track_ids"), name="track_ids")
        step_valid_raw = _field(step, "valid", default=valid)
        step_valid = _sequence_values(step_valid_raw, name="step valid")
        selected_queries = [
            index
            for index, track_id in enumerate(track_ids)
            if track_id is not None and bool(valid[index]) and bool(step_valid[index])
        ]
        births = _sequence_values(
            _field(step, "births", default=(False,) * len(track_ids)),
            name="births",
        )
        rejected = _sequence_values(
            _field(step, "rejected_births", default=(False,) * len(track_ids)),
            name="rejected_births",
        )
        if len(births) != len(track_ids) or len(rejected) != len(track_ids):
            raise ValueError("tracker decision fields must cover every query")
        target = _target_mapping(payload)
        gt_ids = _integer_tensor(target["gt_ids"], name="gt_ids", ndim=1)
        gt_classes = _integer_tensor(
            target["gt_classes"], name="gt_classes", ndim=1
        )
        gt_masks = _clone_cpu(target["gt_masks"], name="gt_masks")
        pred_masks = prediction["pred_masks"]
        pred_classes = prediction["pred_classes"]
        pairs = match_instances_hungarian(
            gt_masks,
            pred_masks.transpose(0, 1),
            gt_classes,
            pred_classes,
            threshold=iou_threshold,
        )
        ious = _binary_iou_matrix(gt_masks, pred_masks)
        matched_gt = {gt_index for gt_index, _ in pairs}
        pair_by_pred = {pred_index: gt_index for gt_index, pred_index in pairs}
        current_matches: list[tuple[int, str | int]] = []
        key = _require_mapping(payload["key"], name="cache key")
        scene_id = str(key["history_scan_ids"][-1])
        class_prob = _finite_tensor(
            observation["class_prob"], name="class_prob", ndim=2
        )
        confidence = _finite_tensor(
            observation["confidence"], name="confidence", ndim=1
        )
        mask_support = _integer_tensor(
            observation["mask_support"], name="mask_support", ndim=1
        )

        for pred_index, query_index in enumerate(selected_queries):
            predicted_identity = _event_identity(track_ids[query_index])
            predicted_class = int(pred_classes[pred_index].item())
            gt_index = pair_by_pred.get(pred_index)
            gt_id = int(gt_ids[gt_index].item()) if gt_index is not None else None
            prior_gt = gt_history.get(gt_id) if gt_id is not None else None
            transition = prior_gt is not None
            gap_length = stage - prior_gt[0] - 1 if prior_gt is not None else 0
            gt_gap = transition and gap_length > 0
            prior_identity_gt = identity_history.get(predicted_identity)
            merge = gt_id is not None and (
                prior_identity_gt is not None and prior_identity_gt != gt_id
            )
            switched = bool(
                transition and predicted_identity != prior_gt[1]
            )
            fragmentation = switched and not merge
            tracker_reactivation = (
                _diagnostic_value(step, "reactivation", query_index) is True
            )
            reactivation = bool(
                gt_gap
                and (
                    tracker_reactivation
                    or (not bool(births[query_index]) and prior_identity_gt is not None)
                )
            )
            reactivation_correct = (
                bool(
                    predicted_identity == prior_gt[1]
                    and prior_identity_gt == gt_id
                    and not merge
                )
                if reactivation
                else None
            )
            new_birth = bool(births[query_index])
            false_birth = bool(new_birth and (gt_id is None or transition))
            semantic_drift = False
            association_miss = False
            if gt_id is None:
                spatial = (
                    torch.nonzero(ious[:, pred_index] >= iou_threshold)
                    .flatten()
                    .tolist()
                )
                semantic_drift = any(
                    int(gt_classes[index].item()) != predicted_class
                    for index in spatial
                )
                association_miss = not new_birth and not semantic_drift
                if semantic_drift:
                    false_birth = False
            failure = bool(
                gt_id is None
                or switched
                or merge
                or false_birth
                or reactivation_correct is False
            )
            if reactivation:
                result = (
                    "reactivation_correct"
                    if reactivation_correct
                    else "reactivation_wrong"
                )
            elif semantic_drift:
                result = "semantic_drift"
            elif false_birth:
                result = "false_birth"
            elif new_birth:
                result = "birth"
            elif failure:
                result = "active_wrong"
            else:
                result = "active_correct"
            probabilities = class_prob[query_index].clamp_min(1e-12)
            entropy = float(-(probabilities * probabilities.log()).sum().item())
            event = AssociationEvent(
                event_id=(
                    f"{method}:{master_sequence_id}:{order_id}:T{prefix}:"
                    f"s{stage}:q{query_index}"
                ),
                scene_id=scene_id,
                sequence_id=sequence_id,
                reference_scene_id=reference_scene_id,
                master_sequence_id=master_sequence_id,
                order_id=order_id,
                prefix=prefix,
                method=method,
                stage_id=stage,
                query_id=query_index,
                candidate_slot_id=_event_identity(
                    _diagnostic_value(
                        step, "selected_candidate_identity", query_index
                    )
                ),
                predicted_identity_id=predicted_identity,
                gt_entity_id=gt_id,
                association_correct=(not failure),
                feature_similarity=_diagnostic_value(
                    step, "chosen_feature_similarity", query_index
                ),
                class_similarity=_diagnostic_value(
                    step, "chosen_class_similarity", query_index
                ),
                total_score=_diagnostic_value(
                    step, "chosen_total_score", query_index
                ),
                best_score=_diagnostic_value(step, "best_score", query_index),
                second_best_score=_diagnostic_value(
                    step, "second_best_score", query_index
                ),
                score_margin=_diagnostic_value(step, "score_margin", query_index),
                observation_confidence=float(confidence[query_index].item()),
                mask_support=float(mask_support[query_index].item()),
                predicted_class=predicted_class,
                class_entropy=entropy,
                slot_age=_diagnostic_value(step, "slot_age", query_index),
                last_seen_stage=_diagnostic_value(
                    step, "last_seen_stage", query_index
                ),
                gap_length=gap_length,
                slot_active=_diagnostic_value(step, "slot_active", query_index),
                slot_occupied=_diagnostic_value(
                    step, "slot_occupied", query_index
                ),
                association_result=result,
                gt_present=gt_id is not None,
                prediction_present=True,
                transition_opportunity=transition,
                id_switch=switched,
                gap_opportunity=gt_gap,
                reactivation_attempt=reactivation,
                reactivation_correct=reactivation_correct,
                new_birth=new_birth,
                false_birth=false_birth,
                reactivation=reactivation,
                wrong_reactivation=reactivation_correct is False,
                local_observation_available=True,
                local_match_available=True if gt_id is not None else None,
                raw_local_match=True if gt_id is not None else None,
                raw_prediction_available=True,
                association_miss=association_miss,
                identity_fragmentation=fragmentation,
                identity_merge=merge,
                semantic_drift=semantic_drift,
                capacity_failure=False,
                birth_rejected=False,
                is_failure=failure,
                prediction_digest=cache_digest,
                cache_digest=cache_digest,
            )
            if failure:
                event = replace(event, failure_category=classify_failure(event))
            events.append(event)
            if gt_id is not None:
                current_matches.append((gt_id, predicted_identity))

        for query_index, is_rejected in enumerate(rejected):
            if not is_rejected:
                continue
            probabilities = class_prob[query_index].clamp_min(1e-12)
            predicted_class = int(
                torch.argmax(
                    torch.cat(
                        (
                            probabilities[:background_class],
                            probabilities[background_class + 1 :],
                        )
                    )
                ).item()
            )
            if predicted_class >= background_class:
                predicted_class += 1
            event = AssociationEvent(
                event_id=(
                    f"{method}:{master_sequence_id}:{order_id}:T{prefix}:"
                    f"s{stage}:rejected-q{query_index}"
                ),
                scene_id=scene_id,
                sequence_id=sequence_id,
                reference_scene_id=reference_scene_id,
                master_sequence_id=master_sequence_id,
                order_id=order_id,
                prefix=prefix,
                method=method,
                stage_id=stage,
                query_id=query_index,
                predicted_class=predicted_class,
                observation_confidence=float(confidence[query_index].item()),
                mask_support=float(mask_support[query_index].item()),
                class_entropy=float(
                    -(probabilities * probabilities.log()).sum().item()
                ),
                association_result="birth_rejected",
                gt_present=False,
                prediction_present=False,
                association_correct=False,
                transition_opportunity=False,
                id_switch=False,
                gap_opportunity=False,
                reactivation_attempt=False,
                reactivation=False,
                new_birth=False,
                false_birth=False,
                birth_rejected=True,
                capacity_failure=True,
                is_failure=True,
                failure_category="F7",
                prediction_digest=cache_digest,
                cache_digest=cache_digest,
            )
            events.append(event)

        for gt_index in range(gt_ids.shape[0]):
            if gt_index in matched_gt:
                continue
            gt_id = int(gt_ids[gt_index].item())
            prior_gt = gt_history.get(gt_id)
            gap_length = stage - prior_gt[0] - 1 if prior_gt is not None else 0
            spatial_indices = (
                torch.nonzero(ious[gt_index] >= iou_threshold).flatten().tolist()
                if ious.shape[1]
                else []
            )
            compatible = any(
                int(pred_classes[index].item()) == int(gt_classes[gt_index].item())
                for index in spatial_indices
            )
            semantic_drift = bool(spatial_indices and not compatible)
            merge = bool(compatible)
            local_miss = not spatial_indices
            association_miss = compatible
            event = AssociationEvent(
                event_id=(
                    f"{method}:{master_sequence_id}:{order_id}:T{prefix}:"
                    f"s{stage}:gt{gt_id}-miss"
                ),
                scene_id=scene_id,
                sequence_id=sequence_id,
                reference_scene_id=reference_scene_id,
                master_sequence_id=master_sequence_id,
                order_id=order_id,
                prefix=prefix,
                method=method,
                stage_id=stage,
                event_kind="gt_miss",
                gt_entity_id=gt_id,
                association_result="no_attempt",
                gt_present=True,
                prediction_present=False,
                transition_opportunity=False,
                id_switch=False,
                gap_opportunity=bool(prior_gt is not None and gap_length > 0),
                reactivation_attempt=False,
                reactivation=False,
                new_birth=False,
                false_birth=False,
                local_observation_available=not local_miss,
                local_match_available=True if compatible else None,
                raw_local_match=True if compatible else None,
                raw_prediction_available=bool(pred_masks.shape[1]),
                local_perception_miss=local_miss,
                association_miss=association_miss,
                identity_merge=merge,
                semantic_drift=semantic_drift,
                is_failure=True,
                prediction_digest=cache_digest,
                cache_digest=cache_digest,
            )
            events.append(
                replace(event, failure_category=classify_failure(event))
            )

        for gt_id, predicted_identity in current_matches:
            gt_history[gt_id] = (stage, predicted_identity)
            identity_history[predicted_identity] = gt_id

    return validate_association_events(events)


def build_capacity_snapshots(
    steps: Sequence[object],
    *,
    horizon: int,
    method: str = "B4",
) -> tuple[CapacitySnapshot, ...]:
    """Extract bounded persistent-state occupancy from one B4 prefix run."""

    if not steps or len(steps) != horizon:
        raise ValueError("steps must cover the exact horizon")
    snapshots = []
    for stage, step in enumerate(steps):
        if _field(step, "stage_id") != stage:
            raise ValueError("tracker steps must be contiguous from stage zero")
        state = _field(step, "state_snapshot")
        validate = getattr(state, "validate", None)
        if not callable(validate):
            raise TypeError("B4 steps must expose a validated state snapshot")
        validate()
        batch_size = int(state.batch_size)
        if batch_size != 1:
            raise ValueError("P6-A capacity snapshots require batch size one")
        capacity = int(state.capacity)
        feature_dim = int(state.feature_dim)
        class_count = int(state.class_count)
        occupied = int(state.occupied[0].sum().item())
        active = int(state.active[0].sum().item())
        births = _sequence_values(
            _field(step, "births", default=()), name="births"
        )
        rejected = _sequence_values(
            _field(step, "rejected_births", default=()), name="rejected_births"
        )
        snapshots.append(
            CapacitySnapshot(
                method=method,
                horizon=horizon,
                stage_id=stage,
                capacity=capacity,
                birth_count=sum(value is True for value in births),
                occupied_count=occupied,
                active_count=active,
                dormant_count=occupied - active,
                rejected_births=sum(value is True for value in rejected),
                persistent_state_bytes=persistent_state_bytes(
                    capacity,
                    feature_dim,
                    class_count,
                    batch_size=batch_size,
                ),
                feature_dim=feature_dim,
                class_count=class_count,
                batch_size=batch_size,
            )
        )
    for snapshot in snapshots:
        snapshot.validate()
    return tuple(snapshots)


def normalize_official_metric_blocks(
    metric_blocks: Mapping[str, object],
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    """Normalize stmetrics keys into the exact P6-A root metric schema."""

    blocks = _require_mapping(metric_blocks, name="metric_blocks")
    if set(blocks) != {"raw", "strict", "offline"}:
        raise ValueError("metric_blocks must contain raw, strict, and offline")
    fields = (
        "AP",
        "AP50",
        "AP25",
        "REC",
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "t_REC50",
        "t_REC25",
    )
    result: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for block_name in ("raw", "strict", "offline"):
        methods = _require_mapping(blocks[block_name], name=f"{block_name} metrics")
        result[block_name] = {}
        source_fields = {
            "raw": {
                "raw_local_AP": "AP",
                "raw_local_AP50": "AP50",
                "raw_local_AP25": "AP25",
                "raw_local_REC": "REC",
                "raw_local_REC50": "REC50",
                "raw_local_REC25": "REC25",
            },
            "strict": {
                "online_t-mAP": "t_mAP",
                "online_t-mAP50": "t_mAP50",
                "online_t-mAP25": "t_mAP25",
                "online_t-REC": "t_REC",
                "online_t-REC50": "t_REC50",
                "online_t-REC25": "t_REC25",
            },
            "offline": {
                "offline_reconstructed_t-mAP": "t_mAP",
                "offline_reconstructed_t-mAP50": "t_mAP50",
                "offline_reconstructed_t-mAP25": "t_mAP25",
                "offline_reconstructed_t-REC": "t_REC",
                "offline_reconstructed_t-REC50": "t_REC50",
                "offline_reconstructed_t-REC25": "t_REC25",
            },
        }[block_name]
        expected_keys = set(source_fields)
        for method, raw_horizons in methods.items():
            if not isinstance(method, str) or not method:
                raise ValueError("metric method names must be non-empty strings")
            horizons = _require_mapping(
                raw_horizons, name=f"{block_name}.{method} metrics"
            )
            result[block_name][method] = {}
            for horizon, raw_values in horizons.items():
                if not isinstance(horizon, str) or not horizon:
                    raise ValueError("metric horizon names must be non-empty strings")
                values = _require_mapping(
                    raw_values, name=f"{block_name}.{method}.{horizon}"
                )
                if set(values) != expected_keys:
                    raise ValueError(
                        f"{block_name}.{method}.{horizon} metric keys differ"
                    )
                normalized: dict[str, float | None] = {
                    field: None for field in fields
                }
                for source, destination in source_fields.items():
                    value = values[source]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0
                    ):
                        raise ValueError("official metric values must be finite in [0, 1]")
                    normalized[destination] = float(value)
                result[block_name][method][horizon] = normalized
    return result


def evaluate_cached_task_metrics(
    sequences: Sequence[CachedProtocolSequence],
    *,
    tracker_factories: Mapping[str, object],
    class_mapper: Callable[[int], int] | Mapping[int, int],
    metric_factory: Callable[[str, str, int], object] = _official_metric_factory,
    background_class: int = 18,
) -> TaskMetricEvaluation:
    """Evaluate raw, causal-prefix, and offline metrics from one frozen cache."""

    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a sequence")
    if not sequences:
        raise ValueError("sequences must not be empty")
    methods = tuple(tracker_factories)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("tracker_factories must contain unique methods")
    all_methods = (*methods, "Oracle")
    horizons = (2, 3, 4, 5)
    raw_metrics = {
        horizon: metric_factory("raw_local", "shared", horizon)
        for horizon in horizons
    }
    strict_metrics = {
        method: {
            horizon: metric_factory("strict_online", method, horizon)
            for horizon in horizons
        }
        for method in methods
    }
    offline_metrics = {
        method: {
            horizon: metric_factory("offline_reconstructed", method, horizon)
            for horizon in horizons
        }
        for method in all_methods
    }
    raw_predictions: list[dict[str, object]] = []
    cache_hasher = hashlib.sha256()
    association_events: list[AssociationEvent] = []
    capacity_snapshots: list[CapacitySnapshot] = []

    for sequence in sequences:
        if not isinstance(sequence, CachedProtocolSequence):
            raise TypeError("sequences must contain CachedProtocolSequence values")
        payloads = sequence.payloads
        sequence_id = (
            f"{sequence.master_sequence_id}:{sequence.order_id}"
        )
        coordinated = prefix_causality_coordinator(
            payloads,
            tracker_factories,
            endpoints=(1, 2, 3, 4),
            sequence_id=sequence_id,
            background_class=background_class,
        )
        frozen = tuple(cache_payload_to_frozen_observation(item) for item in payloads)
        oracle_targets = tuple(
            OracleStageTarget(
                gt_ids=tuple(
                    _integer_tensor(
                        _target_mapping(payload)["gt_ids"],
                        name="gt_ids",
                        ndim=1,
                    ).tolist()
                ),
                classes=tuple(
                    _integer_tensor(
                        _target_mapping(payload)["gt_classes"],
                        name="gt_classes",
                        ndim=1,
                    ).tolist()
                ),
                masks=_clone_cpu(
                    _target_mapping(payload)["gt_masks"], name="gt_masks"
                ),
            )
            for payload in payloads
        )
        oracle_steps = run_oracle_posthoc(
            frozen,
            oracle_targets,
            sequence_id=sequence_id,
            background_class=background_class,
        )
        oracle_accumulator = IdentityAccumulator()
        for payload, step in zip(payloads, oracle_steps, strict=True):
            oracle_accumulator.add_stage(
                stage_prediction_from_track_step(
                    payload,
                    step,
                    background_class=background_class,
                )
            )

        cache_hasher.update(sequence.reference_scene_id.encode("utf-8") + b"\0")
        cache_hasher.update(sequence.master_sequence_id.encode("utf-8") + b"\0")
        cache_hasher.update(sequence.order_id.encode("ascii") + b"\0")
        cache_hasher.update(coordinated.content_digest.encode("ascii"))

        for horizon in horizons:
            endpoint = horizon - 1
            target = _remap_metric_target(
                build_temporal_target(payloads[:horizon]), class_mapper
            )
            raw_observation = frozen[endpoint]
            raw_step = B0StageUniqueTracker(sequence_id=sequence_id).step(
                raw_observation,
                stage_id=endpoint,
            )
            raw_prediction = _remap_metric_prediction(
                stage_prediction_from_track_step(
                    payloads[endpoint],
                    raw_step,
                    background_class=background_class,
                ),
                class_mapper,
            )
            raw_target = _remap_metric_target(
                _raw_local_target(payloads[endpoint]), class_mapper
            )
            raw_metrics[horizon].update(raw_prediction, raw_target)
            raw_predictions.append(raw_prediction)

            for method in methods:
                association_events.extend(
                    build_association_events(
                        payloads[:horizon],
                        coordinated.online_steps[method][endpoint],
                        method=method,
                        reference_scene_id=sequence.reference_scene_id,
                        master_sequence_id=sequence.master_sequence_id,
                        order_id=sequence.order_id,
                        prefix=horizon,
                        cache_digest=coordinated.content_digest,
                        background_class=background_class,
                    )
                )
                strict_prediction = _remap_metric_prediction(
                    coordinated.online_predictions[method][endpoint], class_mapper
                )
                strict_metrics[method][horizon].update(strict_prediction, target)
                offline_prediction = _remap_metric_prediction(
                    build_offline_reconstructed_prediction(
                        coordinated.offline[method], endpoint=endpoint
                    ),
                    class_mapper,
                )
                offline_metrics[method][horizon].update(offline_prediction, target)
            if "B4" in methods:
                capacity_snapshots.extend(
                    build_capacity_snapshots(
                        coordinated.online_steps["B4"][endpoint],
                        horizon=horizon,
                    )
                )
            oracle_prediction = _remap_metric_prediction(
                build_offline_reconstructed_prediction(
                    oracle_accumulator, endpoint=endpoint
                ),
                class_mapper,
            )
            offline_metrics["Oracle"][horizon].update(oracle_prediction, target)

    raw_result = {f"T{horizon}": raw_metrics[horizon].compute() for horizon in horizons}
    metric_blocks = {
        "raw": {
            method: {
                horizon: dict(values) for horizon, values in raw_result.items()
            }
            for method in methods
        },
        "strict": {
            method: {
                f"T{horizon}": strict_metrics[method][horizon].compute()
                for horizon in horizons
            }
            for method in methods
        },
        "offline": {
            method: {
                f"T{horizon}": offline_metrics[method][horizon].compute()
                for horizon in horizons
            }
            for method in all_methods
        },
    }
    raw_fingerprint = assert_shared_raw_predictions(
        {method: raw_predictions for method in all_methods}
    )
    cache_fingerprint = cache_hasher.hexdigest()
    fingerprints = {
        "prediction": {method: raw_fingerprint for method in all_methods},
        "cache": {method: cache_fingerprint for method in all_methods},
    }
    return TaskMetricEvaluation(
        metric_blocks=metric_blocks,
        fingerprints=fingerprints,
        sequence_count=len(sequences),
        association_events=tuple(association_events),
        capacity_snapshots=tuple(capacity_snapshots),
    )


def _normalize_key(value: object, *, name: str = "cache key") -> dict[str, object]:
    key = _require_mapping(value, name=name)
    if set(key) != KEY_KEYS:
        raise ValueError(f"{name} keys differ from cache key schema")
    stage = key["stage_index"]
    if isinstance(stage, bool) or not isinstance(stage, int) or stage < 0:
        raise ValueError(f"{name}.stage_index must be non-negative")
    history = _as_scan_ids(key["history_scan_ids"], name=f"{name}.history_scan_ids")
    local = _as_scan_ids(
        key["local_window_scan_ids"], name=f"{name}.local_window_scan_ids"
    )
    expected_local = history[-1:] if stage == 0 else history[-2:]
    if len(history) != stage + 1 or local != expected_local:
        raise ValueError(f"{name} does not describe a causal prefix window")
    for field_name in ("master_sequence_id", "reference_scene_id", "order_id"):
        if not isinstance(key[field_name], str) or not key[field_name]:
            raise ValueError(f"{name}.{field_name} must be non-empty")
    if key["order_id"] not in _EXPECTED_ORDER_NAMES:
        raise ValueError(f"{name}.order_id is not a registered Protocol B order")
    if stage > 4:
        raise ValueError(f"{name}.stage_index must be in [0, 4]")
    return {
        "master_sequence_id": key["master_sequence_id"],
        "reference_scene_id": key["reference_scene_id"],
        "order_id": key["order_id"],
        "stage_index": stage,
        "history_scan_ids": history,
        "local_window_scan_ids": local,
    }


def _key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(_normalize_key(value), sort_keys=True, separators=(",", ":"))


def _manifest_entries(manifest: object) -> list[Mapping[str, object]]:
    root = _require_mapping(manifest, name="cache manifest")
    entries = root.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError("cache manifest entries must be a sequence")  # noqa: TRY004
    result = []
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != ENTRY_KEYS:
            raise ValueError("cache manifest contains an invalid entry")
        result.append(entry)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheResolution:
    payload: Mapping[str, object]
    entry: Mapping[str, object]
    reused: bool

    def __getitem__(self, key: str) -> object:
        if key in {"payload", "entry", "reused"}:
            return getattr(self, key)
        return self.entry[key]


def _call_producer(producer: object, key: Mapping[str, object]) -> object:
    if not callable(producer):
        raise ValueError(  # noqa: TRY004
            "producer must be callable when cache entry is missing"
        )
    try:
        signature = inspect.signature(producer)
    except (TypeError, ValueError):
        return producer()
    parameters = signature.parameters
    if "key" in parameters:
        return producer(key=key)
    required = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    return producer() if not required else producer(key)


def resolve_cache_entry(
    cache_directory: str | Path,
    logical_key: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    expected_provenance: Mapping[str, object],
    producer: Callable[..., Mapping[str, object]] | None,
) -> CacheResolution:
    """Resume one cache key without overwriting or accepting stale content."""

    key = _normalize_key(logical_key)
    manifest = _require_mapping(manifest, name="cache manifest")
    if not isinstance(expected_provenance, Mapping):
        raise ValueError("expected_provenance must be a mapping")  # noqa: TRY004
    manifest_provenance = manifest.get("provenance")
    if manifest_provenance is not None and dict(manifest_provenance) != dict(
        expected_provenance
    ):
        raise ValueError("cache manifest provenance differs from the frozen run")
    cache_dir = Path(cache_directory)
    if cache_dir.is_symlink() or (cache_dir.exists() and not cache_dir.is_dir()):
        raise ValueError("cache_directory must be a regular directory")
    matches = [
        entry
        for entry in _manifest_entries(manifest)
        if _key_identity(entry["key"]) == _key_identity(key)
    ]
    if len(matches) > 1:
        raise ValueError("cache manifest contains duplicate logical keys")
    if matches:
        entry = matches[0]
        filename = entry["filename"]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("cache manifest filename must be a plain file name")
        path = cache_dir / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("manifest entry points to a missing or symlink cache file")
        if (
            isinstance(entry["file_bytes"], bool)
            or not isinstance(entry["file_bytes"], int)
            or entry["file_bytes"] <= 0
        ):
            raise ValueError("cache manifest file_bytes must be positive")
        if entry["file_bytes"] != path.stat().st_size:
            raise ValueError("cache entry file size differs from manifest")
        if entry["file_sha256"] != _file_sha256(path):
            raise ValueError("cache entry file digest differs from manifest")
        payload = load_cache_entry(path, expected_provenance=expected_provenance)
        if _key_identity(payload["key"]) != _key_identity(key):
            raise ValueError("loaded cache entry key differs from requested key")
        return CacheResolution(payload=payload, entry=entry, reused=True)

    payload = _call_producer(producer, key)
    if not isinstance(payload, Mapping):
        raise ValueError(  # noqa: TRY004
            "cache producer must return a mapping payload"
        )
    validate_cache_payload(payload)
    if _key_identity(payload["key"]) != _key_identity(key):
        raise ValueError("cache producer returned a different logical key")
    if dict(payload["provenance"]) != dict(expected_provenance):
        raise ValueError("cache producer returned a different provenance")
    entry = write_cache_entry(cache_dir, payload)
    return CacheResolution(payload=payload, entry=entry, reused=False)


def atomic_manifest_payload(
    entries: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[Mapping[str, object]],
    expected_provenance: Mapping[str, object],
) -> dict[str, object]:
    """Return a validated manifest payload without writing any P6-A artifact."""

    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError("entries must be a sequence")  # noqa: TRY004
    normalized_keys = [
        _normalize_key(key, name="expected key") for key in expected_keys
    ]
    return build_cache_manifest(
        entries,
        expected_keys=normalized_keys,
        expected_provenance=expected_provenance,
    )


def publish_manifest_atomic(path: str | Path, manifest: Mapping[str, object]) -> None:
    """Publish one already-built manifest with an atomic same-directory replace."""

    destination = Path(path)
    if destination.exists() and destination.is_symlink():
        raise ValueError("manifest destination must not be a symlink")
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")  # noqa: TRY004
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            manifest, allow_nan=False, ensure_ascii=True, sort_keys=True, indent=2
        )
        + "\n"
    )
    payload = data.encode("utf-8")
    if destination.exists():
        if destination.is_symlink():
            raise ValueError("manifest destination must not be a symlink")
        if destination.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite manifest: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise FileExistsError(f"refusing to overwrite manifest: {destination}")
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


load_or_create_cache_entry = resolve_cache_entry
ensure_cache_entry = resolve_cache_entry
atomic_manifest_publish = publish_manifest_atomic
build_atomic_manifest_payload = atomic_manifest_payload


__all__ = [
    "CacheResolution",
    "CachedProtocolSequence",
    "PrefixCausalityResult",
    "ProtocolCacheRequest",
    "RealPredictionCacheProducer",
    "TaskMetricEvaluation",
    "atomic_manifest_payload",
    "atomic_manifest_publish",
    "build_association_events",
    "build_atomic_manifest_payload",
    "build_cache_provenance",
    "build_capacity_snapshots",
    "build_rio_class_mapper",
    "build_temporal_target",
    "build_tracker_factories",
    "cache_payload_from_inference",
    "cache_payload_to_frozen_observation",
    "ensure_cache_entry",
    "evaluate_cached_task_metrics",
    "expected_cache_keys",
    "load_cached_protocol_sequences",
    "load_or_create_cache_entry",
    "materialize_prediction_cache",
    "normalize_official_metric_blocks",
    "observation_content_digest",
    "prefix_causality_coordinator",
    "publish_manifest_atomic",
    "resolve_cache_entry",
    "resolve_protocol_cache_request",
    "run_real_prediction_cache",
    "stage_prediction_from_track_step",
]


if __name__ == "__main__":
    raise SystemExit(main())
