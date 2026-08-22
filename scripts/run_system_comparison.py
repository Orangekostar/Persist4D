"""Fail-closed orchestration for Full-History vs Persistent-State evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCUMBENT_CONFIG = PROJECT_ROOT / "configs/system_comparison/persist4d_incumbent.yaml"
SOURCE_PROTOCOL = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
SYSTEM_ROOT = PROJECT_ROOT / "artifacts/system_comparison"
SYSTEM_MANIFEST = SYSTEM_ROOT / "system_comparison_manifest.json"
REPRODUCIBILITY_BINDING = SYSTEM_ROOT / "reproducibility_binding.json"
CHECKPOINT = PROJECT_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"
METADATA = Path("/home/ww/3RScan.json")
LOCAL_CACHE_ROOT = SYSTEM_ROOT / "persistent_predictions"
FULL_CACHE_ROOT = SYSTEM_ROOT / "full_history_predictions"
LOCAL_ENTRY_CACHE = LOCAL_CACHE_ROOT / "entries"
FULL_ENTRY_CACHE = FULL_CACHE_ROOT / "entries"
LOCAL_CACHE_MANIFEST = LOCAL_CACHE_ROOT / "manifest.json"
FULL_CACHE_MANIFEST = FULL_CACHE_ROOT / "manifest.json"

T = TypeVar("T")


class GateFailure(RuntimeError):
    """Raised when a preregistered blocking gate does not pass."""


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise GateFailure(f"frozen input is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_exact_json(path: str | Path, value: Mapping[str, object]) -> None:
    """Publish once, or resume only when the canonical bytes are identical."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(value)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_file():
            raise FileExistsError(f"output is not a regular file: {output}")
        if output.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_exact_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"output is not a regular file: {path}")
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"output already contains different content: {path}")
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
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def select_disjoint_shard(
    values: Sequence[T], shard_index: int, shard_count: int
) -> tuple[T, ...]:
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not 0 <= shard_index < shard_count
    ):
        raise ValueError("shard index/count are invalid")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("shard values must be a sequence")
    return tuple(values[shard_index::shard_count])


def _logical_key_identity(value: Mapping[str, object]) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def select_pending_keys(
    keys: Sequence[Mapping[str, object]],
    existing_keys: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    existing = {_logical_key_identity(key) for key in existing_keys}
    if len(existing) != len(existing_keys):
        raise GateFailure("existing cache contains duplicate logical keys")
    expected = {_logical_key_identity(key) for key in keys}
    if not existing <= expected:
        raise GateFailure("existing cache contains an unexpected logical key")
    return tuple(key for key in keys if _logical_key_identity(key) not in existing)


def run_stage_pipeline(
    stages: Sequence[tuple[str, Callable[[], object]]],
    *,
    completed: Sequence[str] = (),
) -> tuple[str, ...]:
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        raise TypeError("stages must be a sequence")
    names = tuple(name for name, _action in stages)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("stage names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("stage names must be unique")
    done = tuple(completed)
    if done != names[: len(done)]:
        raise ValueError("completed stages must be an exact pipeline prefix")
    result = list(done)
    for name, action in stages[len(done) :]:
        if not callable(action):
            raise TypeError(f"stage action is not callable: {name}")
        action()
        result.append(name)
    return tuple(result)


def build_reproducibility_binding(
    *,
    source_commit: str,
    checkpoint_path: str | Path,
    config_path: str | Path,
    protocol_path: str | Path,
    system_manifest: Mapping[str, object],
) -> dict[str, object]:
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise GateFailure("source commit must be a lowercase SHA-1")
    manifest_sha = system_manifest.get("content_sha256")
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha)
    ):
        raise GateFailure("system manifest content hash is invalid")
    return {
        "schema_version": 1,
        "status": "pass",
        "source_commit": source_commit,
        "checkpoint_sha256": _sha256_file(Path(checkpoint_path)),
        "config_sha256": _sha256_file(Path(config_path)),
        "protocol_sha256": _sha256_file(Path(protocol_path)),
        "system_manifest_sha256": manifest_sha,
    }


def _finite_metric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateFailure(f"metric is not numeric: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise GateFailure(f"metric is not finite: {name}")
    return result


def verify_incumbent_regression(
    observed: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    metrics = ("t_mAP", "t_mAP50", "t_mAP25", "t_REC")
    maximum = 0.0
    for horizon in ("T2", "T3", "T4", "T5"):
        observed_row = observed.get(horizon)
        reference_row = reference.get(horizon)
        if not isinstance(observed_row, Mapping) or not isinstance(
            reference_row, Mapping
        ):
            raise GateFailure(f"incumbent regression lacks {horizon}")
        for metric in metrics:
            actual = _finite_metric(observed_row.get(metric), name=f"{horizon}.{metric}")
            expected = _finite_metric(
                reference_row.get(metric), name=f"reference {horizon}.{metric}"
            )
            difference = abs(actual - expected)
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise GateFailure(
                    f"incumbent regression failed at {horizon}.{metric}: "
                    f"{actual} != {expected}"
                )
    return {
        "status": "pass",
        "absolute_tolerance": tolerance,
        "maximum_absolute_difference": maximum,
    }


def verify_t2_regression_pairs(
    full_history_payloads: Sequence[Mapping[str, object]],
    local_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(full_history_payloads) != len(local_payloads) or not full_history_payloads:
        raise GateFailure("T2 regression pairs are incomplete")
    for index, (full, local) in enumerate(
        zip(full_history_payloads, local_payloads, strict=True)
    ):
        full_fingerprints = full.get("observation_fingerprints")
        local_fingerprints = local.get("observation_fingerprints")
        if local_fingerprints is not None:
            if full_fingerprints != local_fingerprints:
                raise GateFailure(f"T2 observation regression failed for pair {index}")
            continue
        from scripts.system_comparison_inference import (
            FullHistoryCacheError,
            assert_t2_observation_regression,
        )

        try:
            assert_t2_observation_regression(full, local)
        except FullHistoryCacheError as error:
            raise GateFailure(f"T2 observation regression failed for pair {index}") from error
    return {"status": "pass", "pair_count": len(full_history_payloads)}


def verify_determinism_repeats(
    repeats: Sequence[Sequence[str]],
) -> dict[str, object]:
    if len(repeats) != 3:
        raise GateFailure("determinism gate requires exactly three repeats")
    normalized = tuple(tuple(row) for row in repeats)
    if not normalized[0] or any(len(row) != len(normalized[0]) for row in normalized):
        raise GateFailure("determinism repeats have different coverage")
    if any(value != normalized[0] for value in normalized[1:]):
        raise GateFailure("determinism fingerprints differ across repeats")
    return {"status": "pass", "repeat_count": 3, "sample_count": len(normalized[0])}


def select_smoke_keys(
    keys: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    candidates = [
        dict(key)
        for key in keys
        if key.get("order_id") == "canonical" and key.get("horizon") == 5
    ]
    candidates.sort(
        key=lambda key: (
            str(key.get("reference_scene_id")),
            str(key.get("master_sequence_id")),
        )
    )
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in candidates:
        reference = key.get("reference_scene_id")
        if isinstance(reference, str) and reference not in seen:
            selected.append(key)
            seen.add(reference)
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise GateFailure("smoke gate requires three distinct reference clusters")
    return tuple(selected)


def oracle_attribution_required(
    *,
    persist4d: Mapping[str, object],
    full_history: Mapping[str, object],
    paired_ci: Mapping[str, Sequence[float]],
    minimum_advantage: float,
) -> bool:
    if not math.isfinite(minimum_advantage) or minimum_advantage < 0:
        raise ValueError("minimum advantage must be finite and non-negative")
    for horizon in ("T4", "T5"):
        advantage = _finite_metric(
            full_history.get(horizon), name=f"full_history.{horizon}"
        ) - _finite_metric(persist4d.get(horizon), name=f"persist4d.{horizon}")
        interval = paired_ci.get(horizon)
        if (
            isinstance(interval, Sequence)
            and not isinstance(interval, (str, bytes))
            and len(interval) == 2
            and advantage >= minimum_advantage
            and _finite_metric(interval[1], name=f"paired_ci.{horizon}.upper") < 0
        ):
            return True
    return False


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40:
        raise GateFailure("Git HEAD is invalid")
    return commit


def _require_tracked_tree_clean() -> None:
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if completed.returncode != 0:
            raise GateFailure("tracked source tree must be clean")


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise GateFailure(f"{name} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GateFailure(f"{name} cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise GateFailure(f"{name} must be a mapping")
    return dict(value)


def _load_incumbent() -> dict[str, Any]:
    try:
        value = yaml.safe_load(INCUMBENT_CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise GateFailure("incumbent config cannot be decoded") from error
    if not isinstance(value, Mapping):
        raise GateFailure("incumbent config must be a mapping")
    return dict(value)


def run_bind() -> dict[str, object]:
    from scripts.system_comparison_protocol import (
        build_system_comparison_manifest,
        validate_incumbent_binding,
        validate_system_comparison_manifest,
    )

    _require_tracked_tree_clean()
    binding = validate_incumbent_binding(
        INCUMBENT_CONFIG,
        repo_root=PROJECT_ROOT,
        checkpoint_path=CHECKPOINT,
    )
    system_manifest = build_system_comparison_manifest(
        SOURCE_PROTOCOL,
        incumbent_binding=binding,
    )
    validate_system_comparison_manifest(
        system_manifest,
        source_protocol_path=SOURCE_PROTOCOL,
    )
    reproducibility = build_reproducibility_binding(
        source_commit=_git_head(),
        checkpoint_path=CHECKPOINT,
        config_path=INCUMBENT_CONFIG,
        protocol_path=SOURCE_PROTOCOL,
        system_manifest=system_manifest,
    )
    publish_exact_json(SYSTEM_MANIFEST, system_manifest)
    publish_exact_json(REPRODUCIBILITY_BINDING, reproducibility)
    return reproducibility


def _load_bound_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    from scripts.system_comparison_protocol import validate_system_comparison_manifest

    _require_tracked_tree_clean()
    system_manifest = _load_json(SYSTEM_MANIFEST, name="system comparison manifest")
    validate_system_comparison_manifest(
        system_manifest,
        source_protocol_path=SOURCE_PROTOCOL,
    )
    binding = _load_json(REPRODUCIBILITY_BINDING, name="reproducibility binding")
    expected = build_reproducibility_binding(
        source_commit=_git_head(),
        checkpoint_path=CHECKPOINT,
        config_path=INCUMBENT_CONFIG,
        protocol_path=SOURCE_PROTOCOL,
        system_manifest=system_manifest,
    )
    if binding != expected:
        raise GateFailure("reproducibility binding differs from frozen inputs")
    return system_manifest, binding


@dataclass
class _FrozenSetup:
    protocol: object
    protocol_manifest: dict[str, object]
    p6a_config: dict[str, Any]
    runtime_config: object
    dataset: object
    collate: object
    local_provenance: dict[str, str]
    full_provenance: dict[str, str]
    device: object | None = None
    system: object | None = None


def _build_frozen_setup(
    *,
    binding: Mapping[str, object],
    metadata_path: Path,
    device_name: str | None,
) -> _FrozenSetup:
    import hydra
    from omegaconf import OmegaConf

    from scripts.evaluate_persist4d import (
        _compose_runtime_config,
        _load_system,
        _resolve_checkpoint,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        _frozen_protocol_bundle,
        build_cache_provenance,
    )

    protocol, protocol_manifest, p6a_bytes = _frozen_protocol_bundle(
        metadata_path=metadata_path.resolve()
    )
    p6a_config = yaml.safe_load(p6a_bytes)
    if not isinstance(p6a_config, Mapping):
        raise GateFailure("P6-A config must be a mapping")
    runtime_config, _memory_config = _compose_runtime_config()
    checkpoint = _resolve_checkpoint(CHECKPOINT)
    runtime_bytes = OmegaConf.to_yaml(
        runtime_config,
        resolve=True,
        sort_keys=True,
    ).encode("utf-8")
    local_provenance = build_cache_provenance(
        source_commit=str(binding["source_commit"]),
        checkpoint_path=checkpoint,
        config_documents={"p6a": p6a_bytes, "runtime": runtime_bytes},
        protocol_manifest=protocol_manifest,
    )
    full_provenance = {
        "source_commit": str(binding["source_commit"]),
        "checkpoint_sha256": str(binding["checkpoint_sha256"]),
        "config_sha256": local_provenance["config_sha256"],
        "protocol_sha256": str(binding["protocol_sha256"]),
    }
    dataset_config = OmegaConf.create(
        OmegaConf.to_container(
            runtime_config.data.validation_dataset,
            resolve=True,
        )
    )
    dataset_config.temporal_window = 5
    dataset = hydra.utils.instantiate(dataset_config)
    collate = hydra.utils.instantiate(runtime_config.data.validation_collation)
    if device_name is None:
        return _FrozenSetup(
            protocol=protocol,
            protocol_manifest=dict(protocol_manifest),
            p6a_config=dict(p6a_config),
            runtime_config=runtime_config,
            dataset=dataset,
            collate=collate,
            local_provenance=local_provenance,
            full_provenance=full_provenance,
        )
    device = _validate_cuda_device(device_name)
    system = _load_system(runtime_config, checkpoint, device)
    return _FrozenSetup(
        protocol=protocol,
        protocol_manifest=dict(protocol_manifest),
        p6a_config=dict(p6a_config),
        runtime_config=runtime_config,
        dataset=dataset,
        collate=collate,
        local_provenance=local_provenance,
        full_provenance=full_provenance,
        device=device,
        system=system,
    )


def _local_producer(setup: _FrozenSetup) -> object:
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _latest_full_resolution_masks,
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.evaluate_persist4d_p6a import RealPredictionCacheProducer

    if setup.device is None or setup.system is None:
        raise GateFailure("local cache production requires a CUDA setup")
    settings = setup.p6a_config["baselines"]["b4"]
    return RealPredictionCacheProducer(
        protocol=setup.protocol,
        provenance=setup.local_provenance,
        dataset=setup.dataset,
        collate=setup.collate,
        system=setup.system,
        device=setup.device,
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
        seed=int(setup.p6a_config["protocol_b"]["seed"]),
    )


def _full_producer(setup: _FrozenSetup) -> object:
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
    )
    from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
    from scripts.system_comparison_inference import FullHistoryPredictionProducer

    if setup.device is None or setup.system is None:
        raise GateFailure("full-history production requires a CUDA setup")
    settings = setup.p6a_config["baselines"]["b4"]
    return FullHistoryPredictionProducer(
        dataset=setup.dataset,
        collate=setup.collate,
        system=setup.system,
        device=setup.device,
        provenance=setup.full_provenance,
        class_mapper=build_rio_class_mapper(setup.dataset),
        move_data=_move_data_to_device,
        move_targets=_move_targets_to_device,
        background_class=int(settings["background_class"]),
        confidence_threshold=float(settings["confidence_threshold"]),
        mask_threshold=float(settings["mask_threshold"]),
        minimum_mask_support=int(settings["minimum_mask_support"]),
        seed=int(setup.p6a_config["protocol_b"]["seed"]),
    )


def run_cache_local_shard(
    *,
    device_name: str,
    shard_index: int,
    shard_count: int,
    metadata_path: Path,
) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import expected_cache_keys
    from scripts.p6a_cache import load_cache_entry, write_cache_entry
    from scripts.system_comparison_inference import deterministic_inference_runtime

    _system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    all_keys = expected_cache_keys(setup.protocol)
    keys = select_disjoint_shard(all_keys, shard_index, shard_count)
    existing_keys = []
    if LOCAL_ENTRY_CACHE.exists():
        for path in sorted(LOCAL_ENTRY_CACHE.glob("*.pt")):
            payload = load_cache_entry(
                path,
                expected_provenance=setup.local_provenance,
            )
            existing_keys.append(payload["key"])
    pending_identities = {
        _logical_key_identity(key)
        for key in select_pending_keys(all_keys, existing_keys)
    }
    pending = tuple(
        key for key in keys if _logical_key_identity(key) in pending_identities
    )
    producer = _local_producer(setup)
    entries = []
    with deterministic_inference_runtime(
        int(setup.p6a_config["protocol_b"]["seed"]), setup.device
    ):
        for key in pending:
            entries.append(write_cache_entry(LOCAL_ENTRY_CACHE, producer(key)))
    return {
        "status": "pass",
        "shard_index": shard_index,
        "expected_count": len(keys),
        "reused_count": len(keys) - len(pending),
        "produced_count": len(entries),
    }


def run_cache_full_shard(
    *,
    device_name: str,
    shard_index: int,
    shard_count: int,
    metadata_path: Path,
) -> dict[str, object]:
    import torch

    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        full_history_cache_keys,
        validate_full_history_payload,
        write_full_history_cache_entry,
    )

    system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    all_keys = full_history_cache_keys(system_manifest)
    keys = select_disjoint_shard(all_keys, shard_index, shard_count)
    existing_keys = []
    if FULL_ENTRY_CACHE.exists():
        for path in sorted(FULL_ENTRY_CACHE.glob("*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validated = validate_full_history_payload(
                payload,
                expected_provenance=setup.full_provenance,
            )
            existing_keys.append(validated["key"])
    pending_identities = {
        _logical_key_identity(key)
        for key in select_pending_keys(all_keys, existing_keys)
    }
    pending = tuple(
        key for key in keys if _logical_key_identity(key) in pending_identities
    )
    producer = _full_producer(setup)
    entries = []
    with deterministic_inference_runtime(
        int(setup.p6a_config["protocol_b"]["seed"]), setup.device
    ):
        for key in pending:
            entries.append(
                write_full_history_cache_entry(FULL_ENTRY_CACHE, producer(key))
            )
    return {
        "status": "pass",
        "shard_index": shard_index,
        "expected_count": len(keys),
        "reused_count": len(keys) - len(pending),
        "produced_count": len(entries),
    }


def run_finalize_caches(*, metadata_path: Path) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import expected_cache_keys
    from scripts.p6a_cache import (
        build_cache_manifest,
        discover_cache_entries,
        write_cache_manifest,
    )
    from scripts.system_comparison_inference import (
        build_full_history_cache_manifest,
        discover_full_history_cache_entries,
        full_history_cache_keys,
        write_full_history_cache_manifest,
    )

    system_manifest, binding = _load_bound_inputs()
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    local_keys = expected_cache_keys(setup.protocol)
    local_entries = discover_cache_entries(
        LOCAL_ENTRY_CACHE,
        expected_provenance=setup.local_provenance,
    )
    local_manifest = build_cache_manifest(
        local_entries,
        expected_keys=local_keys,
        expected_provenance=setup.local_provenance,
        cache_directory=LOCAL_ENTRY_CACHE,
    )
    write_cache_manifest(
        LOCAL_CACHE_MANIFEST,
        local_manifest,
        expected_keys=local_keys,
        expected_provenance=setup.local_provenance,
        cache_directory=LOCAL_ENTRY_CACHE,
    )

    full_keys = full_history_cache_keys(system_manifest)
    full_entries = discover_full_history_cache_entries(
        FULL_ENTRY_CACHE,
        expected_provenance=setup.full_provenance,
    )
    full_manifest = build_full_history_cache_manifest(
        full_entries,
        expected_keys=full_keys,
        expected_provenance=setup.full_provenance,
        cache_directory=FULL_ENTRY_CACHE,
    )
    write_full_history_cache_manifest(
        FULL_CACHE_MANIFEST,
        full_manifest,
        expected_keys=full_keys,
        expected_provenance=setup.full_provenance,
        cache_directory=FULL_ENTRY_CACHE,
    )
    return {
        "status": "pass",
        "local_entry_count": len(local_entries),
        "full_history_entry_count": len(full_entries),
    }


def _validated_full_manifest(
    *,
    system_manifest: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, Any]:
    from scripts.system_comparison_inference import (
        build_full_history_cache_manifest,
        full_history_cache_keys,
    )

    manifest = _load_json(FULL_CACHE_MANIFEST, name="full-history cache manifest")
    entries = manifest.get("entries")
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise GateFailure("full-history cache manifest entries are invalid")
    rebuilt = build_full_history_cache_manifest(
        entries,
        expected_keys=full_history_cache_keys(system_manifest),
        expected_provenance=provenance,
        cache_directory=FULL_ENTRY_CACHE,
    )
    if manifest != rebuilt:
        raise GateFailure("full-history cache manifest is not canonical")
    return manifest


def _run_incumbent_regression(setup: _FrozenSetup) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        evaluate_cached_task_metrics,
        load_cached_protocol_sequences,
        normalize_official_metric_blocks,
    )

    sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=LOCAL_ENTRY_CACHE,
        manifest_path=LOCAL_CACHE_MANIFEST,
    )
    factories = build_tracker_factories(setup.p6a_config)
    evaluation = evaluate_cached_task_metrics(
        sequences,
        tracker_factories={"B4": factories["B4"]},
        class_mapper=build_rio_class_mapper(setup.dataset),
        background_class=int(setup.p6a_config["baselines"]["b4"]["background_class"]),
    )
    normalized = normalize_official_metric_blocks(evaluation.metric_blocks)
    observed = normalized["strict"]["B4"]
    incumbent = _load_incumbent()
    reference = incumbent.get("reference_metrics")
    if not isinstance(reference, Mapping):
        raise GateFailure("incumbent reference metrics are invalid")
    return verify_incumbent_regression(observed, reference, tolerance=1e-12)


def run_smoke(*, device_name: str, metadata_path: Path) -> dict[str, object]:
    from scripts.evaluate_persist4d_p6a import load_cached_protocol_sequences
    from scripts.system_comparison_inference import (
        deterministic_inference_runtime,
        full_history_cache_keys,
        full_history_prediction_fingerprint,
        load_full_history_cache_entry,
    )

    system_manifest, binding = _load_bound_inputs()
    cpu_setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    incumbent_gate = _run_incumbent_regression(cpu_setup)
    full_manifest = _validated_full_manifest(
        system_manifest=system_manifest,
        provenance=cpu_setup.full_provenance,
    )
    all_keys = full_history_cache_keys(system_manifest)
    smoke_keys = select_smoke_keys(all_keys)
    selected_master_ids = tuple(
        str(key["master_sequence_id"]) for key in smoke_keys
    )
    local_sequences = load_cached_protocol_sequences(
        protocol=cpu_setup.protocol,
        cache_directory=LOCAL_ENTRY_CACHE,
        manifest_path=LOCAL_CACHE_MANIFEST,
        allowed_master_sequence_ids=selected_master_ids,
    )
    local_t2 = {
        (sequence.master_sequence_id, sequence.order_id): sequence.payloads[1]
        for sequence in local_sequences
    }
    entries = full_manifest["entries"]
    entry_by_identity = {
        json.dumps(entry["key"], sort_keys=True, separators=(",", ":")): entry
        for entry in entries
    }
    full_t2 = []
    paired_local = []
    for smoke_key in smoke_keys:
        matching = next(
            key
            for key in all_keys
            if key["master_sequence_id"] == smoke_key["master_sequence_id"]
            and key["order_id"] == smoke_key["order_id"]
            and key["horizon"] == 2
        )
        identity = json.dumps(matching, sort_keys=True, separators=(",", ":"))
        full_t2.append(
            load_full_history_cache_entry(
                FULL_ENTRY_CACHE,
                entry_by_identity[identity],
                expected_provenance=cpu_setup.full_provenance,
            )
        )
        paired_local.append(
            local_t2[(str(matching["master_sequence_id"]), str(matching["order_id"]))]
        )
    t2_gate = verify_t2_regression_pairs(full_t2, paired_local)

    cuda_setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=device_name,
    )
    producer = _full_producer(cuda_setup)
    repeat_fingerprints: list[list[str]] = []
    with deterministic_inference_runtime(
        int(cuda_setup.p6a_config["protocol_b"]["seed"]), cuda_setup.device
    ):
        for _repeat in range(3):
            repeat_fingerprints.append(
                [
                    full_history_prediction_fingerprint(producer(key))
                    for key in smoke_keys
                ]
            )
    determinism_gate = verify_determinism_repeats(repeat_fingerprints)
    audit_lines = [
        "# Full-History Determinism Audit",
        "",
        "- Status: PASS",
        "- Method: ReScene4D Full-History (Frozen T2 Checkpoint)",
        "- Prefixes: 3 canonical T5 prefixes from distinct reference clusters",
        "- Repeats: 3 per prefix",
        "- Compared: mask/class/query-ID/score fingerprints",
        f"- Source commit: `{binding['source_commit']}`",
        "",
        "| Reference scene | Master | Fingerprint |",
        "|---|---|---|",
    ]
    for key, fingerprint in zip(smoke_keys, repeat_fingerprints[0], strict=True):
        audit_lines.append(
            f"| `{key['reference_scene_id']}` | `{key['master_sequence_id']}` | "
            f"`{fingerprint}` |"
        )
    audit_lines.extend(
        [
            "",
            (
                "The identical T2 prefix also passed the complete local/full-history "
                "observation fingerprint regression."
            ),
            "",
        ]
    )
    _publish_exact_bytes(
        SYSTEM_ROOT / "FULL_HISTORY_DETERMINISM_AUDIT.md",
        "\n".join(audit_lines).encode("utf-8"),
    )
    return {
        "status": "pass",
        "incumbent_regression": incumbent_gate,
        "t2_regression": t2_gate,
        "determinism": determinism_gate,
    }


def run_evaluate(*, metadata_path: Path) -> dict[str, object]:
    from scripts.system_comparison_analysis import run_cached_system_evaluation

    return run_cached_system_evaluation(
        project_root=PROJECT_ROOT,
        metadata_path=metadata_path,
    )


def run_profile(*, device_name: str, metadata_path: Path) -> dict[str, object]:
    from scripts.profile_system_comparison import run_system_profile

    return run_system_profile(
        project_root=PROJECT_ROOT,
        device_name=device_name,
        metadata_path=metadata_path,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "bind",
            "cache-local-shard",
            "cache-full-shard",
            "finalize-caches",
            "smoke",
            "evaluate",
            "profile",
            "all",
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if arguments.command == "bind":
        result = run_bind()
    elif arguments.command == "cache-local-shard":
        result = run_cache_local_shard(
            device_name=arguments.device,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
            metadata_path=arguments.metadata,
        )
    elif arguments.command == "cache-full-shard":
        result = run_cache_full_shard(
            device_name=arguments.device,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
            metadata_path=arguments.metadata,
        )
    elif arguments.command == "finalize-caches":
        result = run_finalize_caches(metadata_path=arguments.metadata)
    elif arguments.command == "smoke":
        result = run_smoke(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
        )
    elif arguments.command == "evaluate":
        result = run_evaluate(metadata_path=arguments.metadata)
    elif arguments.command == "profile":
        result = run_profile(
            device_name=arguments.device,
            metadata_path=arguments.metadata,
        )
    else:
        stages = (
            ("bind", run_bind),
            (
                "finalize-caches",
                lambda: run_finalize_caches(metadata_path=arguments.metadata),
            ),
            (
                "smoke",
                lambda: run_smoke(
                    device_name=arguments.device,
                    metadata_path=arguments.metadata,
                ),
            ),
            ("evaluate", lambda: run_evaluate(metadata_path=arguments.metadata)),
            (
                "profile",
                lambda: run_profile(
                    device_name=arguments.device,
                    metadata_path=arguments.metadata,
                ),
            ),
        )
        completed = run_stage_pipeline(stages)
        result = {"status": "pass", "completed_stages": list(completed)}
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
