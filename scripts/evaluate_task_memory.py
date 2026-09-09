#!/usr/bin/env python3
"""Produce compact TaskMemory caches and evaluate frozen lag1 metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch

from datasets.task_memory_episode import (
    NativeEpisodeMaster,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
)
from models.task_memory_routing import (
    CommitResult,
    EntityRoute,
    _state_sha256,
)
from scripts.evaluate_persist4d import (
    _move_data_to_device,
    _move_targets_to_device,
    _segment_stages,
)
from scripts.evaluate_persist4d_p6a import build_rio_class_mapper
from scripts.p6a_cache import CHANGE_LABEL_SEMANTICS
from scripts.p6a_metrics import global_hungarian_match
from scripts.preflight_task_memory_episode import (
    _rio_base_dataset,
    _role_by_reference,
    load_reference_by_scene,
)
from scripts.rescene_task_postprocess import extract_official_task_prediction
from scripts.run_task_memory_policy_baseline import select_diagnostic_masters
from scripts.system_comparison_inference import deterministic_inference_runtime
from scripts.task_memory_cache import (
    CACHE_LIMIT_BYTES,
    build_episode_cache_payload,
    build_evaluation_cache_key,
    build_stage_cache_record,
    cache_key_sha256,
    load_task_memory_cache,
    write_task_memory_cache,
)
from scripts.task_memory_contracts import canonical_json_sha256
from scripts.task_memory_metrics import (
    aggregate_identity_event_diagnostics,
    build_retention_rows,
    compute_cached_identity_metrics,
    compute_cached_task_metrics,
)
from scripts.train_task_memory import (
    LOCAL_ASSET_RESOLVER,
    R1_SHA256,
    VARIANTS,
    _apply_common_task_read_initialization,
    _load_r1,
    compose_variant_config,
)
from trainer.task_memory_trainer import (
    TaskMemoryTrainer,
    build_final_prediction_observation,
    commit_task_memory_stage,
)

DEFAULT_DATA_CONTRACT = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
)
DEFAULT_VARIANT_MANIFEST = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/training/variants.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/evaluation"
)
EVALUATION_SEED = 45
POPULATION_ID = "development_common_h5_canonical"
REDUCERS = ("mean", "latest", "max")


class TaskMemoryEvaluationError(RuntimeError):
    """Raised when real evaluation violates a frozen identity or budget."""


@contextmanager
def _evaluation_runtime(device: torch.device):
    with deterministic_inference_runtime(EVALUATION_SEED, device):
        yield


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMemoryEvaluationError(f"cannot decode JSON: {path}") from error
    if not isinstance(value, dict):
        raise TaskMemoryEvaluationError(f"JSON root must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise TaskMemoryEvaluationError("Git HEAD is invalid")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False).returncode:
            raise TaskMemoryEvaluationError(
                "tracked source tree must be clean before real evaluation"
            )
    return result


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise TaskMemoryEvaluationError(f"output cannot be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(),
    )


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise TaskMemoryEvaluationError("metric CSV cannot be empty")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise TaskMemoryEvaluationError("metric rows have inconsistent fields")
    if any(
        value is not None
        and (
            not isinstance(value, (str, int, float, bool))
            or isinstance(value, float)
            and not math.isfinite(value)
        )
        for row in rows
        for value in row.values()
    ):
        raise TaskMemoryEvaluationError("metric CSV values must be finite scalars")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            field: "N/A" if value is None else value
            for field, value in row.items()
        }
        for row in rows
    )
    return buffer.getvalue().encode("utf-8")


def _artifact_record(path: Path, *, row_count: int) -> dict[str, object]:
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise TaskMemoryEvaluationError(
            "public evaluation tables must remain under the repository root"
        ) from error
    return {
        "file_bytes": path.stat().st_size,
        "file_sha256": _file_sha256(path),
        "logical_reference": f"repo:{relative}",
        "row_count": row_count,
    }


def _validated_content_hash(payload: Mapping[str, object], *, name: str) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop("content_sha256", None)
    if not isinstance(observed, str) or canonical_json_sha256(unsigned) != observed:
        raise TaskMemoryEvaluationError(f"{name} content SHA256 differs")
    return observed


def select_development_masters(
    masters: Sequence[NativeEpisodeMaster], *, smoke_master_count: int | None
) -> tuple[NativeEpisodeMaster, ...]:
    selected = tuple(
        master
        for master in masters
        if master.role == "development" and len(master.scan_ids) == 5
    )
    selected = tuple(sorted(selected, key=lambda item: item.sequence_id))
    if not selected:
        raise TaskMemoryEvaluationError("development H5 population is empty")
    if smoke_master_count is None:
        return selected
    if (
        isinstance(smoke_master_count, bool)
        or not isinstance(smoke_master_count, int)
        or not 1 <= smoke_master_count <= 12
    ):
        raise TaskMemoryEvaluationError("smoke master count must be within 1-12")
    smoke = select_diagnostic_masters(selected, limit=smoke_master_count)
    expected_references = min(
        smoke_master_count, len({master.reference_id for master in selected})
    )
    if len({master.reference_id for master in smoke}) != expected_references:
        raise TaskMemoryEvaluationError("smoke panel did not maximize reference coverage")
    return smoke


def current_stage_target(
    full_target: Mapping[str, object], *, latest_local_stage: int
) -> dict[str, object]:
    required = {"ids", "labels", "masks", "temporal_stages"}
    if not isinstance(full_target, Mapping) or not required <= set(full_target):
        raise TaskMemoryEvaluationError("full-resolution target fields differ")
    ids = torch.as_tensor(full_target["ids"]).detach().cpu().long()
    classes = torch.as_tensor(full_target["labels"]).detach().cpu().long()
    masks = torch.as_tensor(full_target["masks"]).detach().cpu().bool()
    stages = torch.as_tensor(full_target["temporal_stages"]).detach().cpu().long()
    if (
        ids.ndim != 1
        or classes.shape != ids.shape
        or masks.ndim != 2
        or masks.shape[0] != ids.numel()
        or masks.shape[1] != stages.numel()
    ):
        raise TaskMemoryEvaluationError("full-resolution target tensors do not align")
    selector = stages == latest_local_stage
    if not selector.any().item():
        raise TaskMemoryEvaluationError("full target lacks the current stage")
    current_masks = masks[:, selector]
    present = current_masks.any(dim=1)
    return {
        "change_label_semantics": CHANGE_LABEL_SEMANTICS,
        "change_labels_valid": False,
        "changes": torch.zeros(int(present.sum().item()), dtype=torch.long),
        "gt_class_semantics": "rescene_model_index_0_based",
        "gt_classes": classes[present].clone(),
        "gt_ids": ids[present].clone(),
        "gt_masks": current_masks[present].clone(),
    }


def _state_contract_sha256(system: TaskMemoryTrainer) -> str:
    model = system.model
    state_config = model.task_memory_config
    return canonical_json_sha256(
        {
            "association_threshold": float(state_config.association_threshold),
            "capacity": int(model.task_memory_capacity),
            "class_weight": float(state_config.class_weight),
            "enabled": bool(system.config.task_memory_training.state_enabled),
            "max_update_rate": float(state_config.max_update_rate),
            "schema_version": "prediction-only-task-state-v1",
            "update_mode": str(state_config.update_mode),
            "update_rate": float(state_config.update_rate),
        }
    )


def _postprocess_sha256() -> str:
    digest = hashlib.sha256()
    for relative in (
        "scripts/evaluate_task_memory.py",
        "scripts/rescene_task_postprocess.py",
        "scripts/task_memory_cache.py",
        "scripts/task_memory_metrics.py",
        "scripts/task_memory_output.py",
    ):
        content = (PROJECT_ROOT / relative).read_bytes()
        digest.update(relative.encode("ascii") + b"\0")
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


def _load_evaluation_system(
    *,
    variant: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    pretrained: Path,
    external_root: Path,
    device: torch.device,
    expected_common_initialization_sha256: str,
) -> tuple[TaskMemoryTrainer, dict[str, object]]:
    config = compose_variant_config(
        variant,
        pretrained=pretrained,
        run_dir=external_root / "evaluation_runtime" / variant,
    )
    system = TaskMemoryTrainer(config)
    if checkpoint_sha256 == R1_SHA256:
        load_audit = _load_r1(system, checkpoint)
        common_sha = _apply_common_task_read_initialization(
            system, external_root / "training/common/task_read_init.pt"
        )
        if bool(config.task_memory_training.state_enabled) and (
            common_sha != expected_common_initialization_sha256
        ):
            raise TaskMemoryEvaluationError(
                "update-0 task-read initialization SHA256 differs"
            )
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise TaskMemoryEvaluationError("checkpoint lacks a state_dict")
        incompatible = system.load_state_dict(state, strict=True)
        load_audit = {
            "loaded_key_count": len(state),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
    system.to(device)
    system.eval()
    return system, load_audit


def _event_diagnostics(
    *,
    route: EntityRoute,
    commit: CommitResult,
    state_before: object,
    official_prediction: object,
    target: Mapping[str, object],
    class_mapper: Callable[[int], int],
    stage_index: int,
) -> dict[str, int]:
    birth_queries = set(commit.births[0].nonzero(as_tuple=True)[0].cpu().tolist())
    rejected = int(commit.rejected_births.sum().cpu().item())
    matched_route = route.query_to_slot[0] >= 0
    current = route.current_supported[0]
    reactivation_queries = set()
    for query in (matched_route & current).nonzero(as_tuple=True)[0].tolist():
        slot = int(route.query_to_slot[0, query].item())
        if int(state_before.last_seen[0, slot].item()) < stage_index - 1:
            reactivation_queries.add(query)

    gt_classes = torch.tensor(
        [class_mapper(int(value)) for value in target["gt_classes"].tolist()],
        dtype=torch.long,
    )
    matches = global_hungarian_match(
        target["gt_masks"],
        official_prediction.latest_stage_masks,
        gt_classes=gt_classes,
        pred_classes=official_prediction.pred_classes,
        threshold=0.5,
    )
    matched_queries = {
        int(official_prediction.source_query_ids[column].item())
        for _, column in matches
    }
    return {
        "births": len(birth_queries),
        "matched_births": len(birth_queries & matched_queries),
        "matched_reactivations": len(reactivation_queries & matched_queries),
        "reactivations": len(reactivation_queries),
        "rejected_births": rejected,
    }


def _produce_episode(
    *,
    episode: object,
    collator: TaskMemoryEpisodeCollator,
    system: TaskMemoryTrainer,
    class_mapper: Callable[[int], int],
    device: torch.device,
    key: Mapping[str, object],
) -> dict[str, object]:
    batch = collator([episode])
    if len(batch.specs) != 1:
        raise TaskMemoryEvaluationError("evaluation requires batch size one")
    state_enabled = bool(system.config.task_memory_training.state_enabled)
    state = system._new_task_state(1) if state_enabled else None
    stages = []
    disabled_state_sha256 = "0" * 64
    for stage_batch in batch.stage_batches:
        data, targets, names = stage_batch.model_batch
        meta = stage_batch.stage_meta[0]
        expected_name = "-".join(meta.scan_ids_in_window)
        if list(names) != [expected_name] or len(targets) != 1:
            raise TaskMemoryEvaluationError("collator changed stage identity")
        full_targets = getattr(data, "target_full", None)
        if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
            raise TaskMemoryEvaluationError("stage lacks a full-resolution target")
        full_target = full_targets[0]
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target_low = targets[0]
        segment_stages = _segment_stages(target_low)
        latest_local_stage = int(segment_stages.max().item())
        raw_coordinates = system._process_raw_coordinates(data)
        state_before_sha256 = (
            _state_sha256(state) if state_enabled else disabled_state_sha256
        )
        with torch.inference_mode():
            if state_enabled:
                output = system.model(
                    data,
                    point2segment=[target_low["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                    task_state=state,
                    stage_meta=stage_batch.stage_meta,
                )
            else:
                output = system.model(
                    data,
                    point2segment=[target_low["point2segment"]],
                    raw_coordinates=raw_coordinates,
                    is_eval=True,
                )
        official = extract_official_task_prediction(
            system=system,
            output=output,
            target_low_resolution=target_low,
            target_full_resolution=full_target,
            data=data,
            class_mapper=class_mapper,
            latest_stage_index=latest_local_stage,
        )
        current_target = current_stage_target(
            full_target, latest_local_stage=latest_local_stage
        )
        if state_enabled:
            route = output.get("task_memory_route")
            if not isinstance(route, EntityRoute):
                raise TaskMemoryEvaluationError("task route is unavailable")
            observation = build_final_prediction_observation(
                output,
                stage_batch.stage_meta,
                background_class=int(system.model.task_background_class),
                confidence_threshold=float(system.model.task_confidence_threshold),
                mask_threshold=float(system.model.task_mask_threshold),
                minimum_mask_support=int(system.model.task_minimum_mask_support),
            )
            transition = commit_task_memory_stage(
                observation=observation,
                route=route,
                state=state,
                stage_meta=stage_batch.stage_meta,
            )
            events = _event_diagnostics(
                route=route,
                commit=transition.commit,
                state_before=state,
                official_prediction=official,
                target=current_target,
                class_mapper=class_mapper,
                stage_index=meta.absolute_stage_index,
            )
            state = transition.runtime_state
            identity_map = transition.commit.identity_map()
            state_after_sha256 = _state_sha256(state)
        else:
            identity_map = {}
            events = {
                "births": 0,
                "matched_births": 0,
                "matched_reactivations": 0,
                "reactivations": 0,
                "rejected_births": 0,
            }
            state_after_sha256 = disabled_state_sha256
        stages.append(
            build_stage_cache_record(
                prediction=official,
                identity_map=identity_map,
                stage_meta=meta,
                target=current_target,
                state_before_sha256=state_before_sha256,
                state_after_sha256=state_after_sha256,
                event_diagnostics=events,
            )
        )
        del data, full_target, official, output, target_low, targets
    return build_episode_cache_payload(key=key, stages=stages)


def _episode_spec(master: NativeEpisodeMaster, *, index: int) -> TaskMemoryEpisodeSpec:
    return TaskMemoryEpisodeSpec.from_master(
        master,
        horizon=5,
        augmentation_seed=EVALUATION_SEED,
        draw_index=index,
        bucket="T5",
    )


def run_evaluation(
    *,
    variant: str,
    checkpoint: Path,
    pretrained: Path,
    data_root: Path,
    rio_metadata: Path,
    external_root: Path,
    cache_root: Path,
    output_root: Path,
    data_contract_path: Path = DEFAULT_DATA_CONTRACT,
    variant_manifest_path: Path = DEFAULT_VARIANT_MANIFEST,
    device_name: str = "cuda:0",
    smoke_master_count: int | None = None,
    reducers: Sequence[str] = REDUCERS,
) -> dict[str, object]:
    if variant not in VARIANTS:
        raise TaskMemoryEvaluationError("variant is not one of the frozen M2 arms")
    source_commit = _git_head()
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    pretrained = pretrained.expanduser().resolve(strict=True)
    data_root = data_root.expanduser().resolve(strict=True)
    rio_metadata = rio_metadata.expanduser().resolve(strict=True)
    external_root = external_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    checkpoint_sha256 = _file_sha256(checkpoint)
    manifest = _load_json(variant_manifest_path)
    _validated_content_hash(manifest, name="variant manifest")
    variant_record = manifest.get("variants", {}).get(variant)
    if not isinstance(variant_record, Mapping):
        raise TaskMemoryEvaluationError("variant manifest lacks the requested arm")
    resolved_config_sha256 = variant_record.get("resolved_config_sha256")
    common_sha256 = manifest.get("common_task_read_initialization", {}).get("sha256")
    if not isinstance(resolved_config_sha256, str) or not isinstance(
        common_sha256, str
    ):
        raise TaskMemoryEvaluationError("variant initialization manifest is incomplete")
    data_contract = _load_json(data_contract_path)
    data_contract_sha256 = _validated_content_hash(
        data_contract, name="data contract"
    )
    if not torch.cuda.is_available():
        raise TaskMemoryEvaluationError("CUDA is required for cache production")
    device = torch.device(device_name)
    if (
        device.type != "cuda"
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise TaskMemoryEvaluationError("requested CUDA device is unavailable")

    system, load_audit = _load_evaluation_system(
        variant=variant,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        pretrained=pretrained,
        external_root=external_root,
        device=device,
        expected_common_initialization_sha256=common_sha256,
    )
    config = system.config
    base = _rio_base_dataset(config, data_root=data_root, horizon=5)
    masters = build_native_episode_masters(
        base,
        reference_by_scene=load_reference_by_scene(rio_metadata),
        role_by_reference=_role_by_reference(data_contract),
    )
    selected = select_development_masters(
        masters, smoke_master_count=smoke_master_count
    )
    class_mapper = build_rio_class_mapper(base)
    collator = TaskMemoryEpisodeCollator(
        hydra.utils.instantiate(config.data.validation_collation)
    )
    state_contract_sha256 = _state_contract_sha256(system)
    initial_state = (
        system._new_task_state(1)
        if bool(config.task_memory_training.state_enabled)
        else None
    )
    initial_state_sha256 = (
        _state_sha256(initial_state) if initial_state is not None else "0" * 64
    )
    postprocess_sha256 = _postprocess_sha256()
    payloads = []
    cache_records = []
    produced = 0
    reused = 0
    started = time.time()
    for index, master in enumerate(selected):
        spec = _episode_spec(master, index=index)
        key = build_evaluation_cache_key(
            population_id=POPULATION_ID,
            reference_id=spec.reference_id,
            episode_id=spec.episode_id,
            history_scan_ids=spec.scan_ids,
            window_mode=str(config.task_memory_training.window_mode),
            checkpoint_sha256=checkpoint_sha256,
            resolved_config_sha256=resolved_config_sha256,
            data_contract_sha256=data_contract_sha256,
            state_contract_sha256=state_contract_sha256,
            initial_state_sha256=initial_state_sha256,
            output_policy="lag1-v1",
            postprocess_sha256=postprocess_sha256,
            evaluation_seed=EVALUATION_SEED,
        )
        cache_path = cache_root / f"{cache_key_sha256(key)}.pt"
        if cache_path.is_file():
            payload = load_task_memory_cache(cache_path, expected_key=key)
            reused += 1
            record = {
                "content_sha256": payload["content_sha256"],
                "file_bytes": cache_path.stat().st_size,
                "file_sha256": _file_sha256(cache_path),
                "filename": cache_path.name,
                "key_sha256": cache_key_sha256(key),
            }
        else:
            with _evaluation_runtime(device):
                episode = TaskMemoryEpisodeDataset(
                    base,
                    (spec,),
                    apply_augmentation=False,
                    window_mode=str(config.task_memory_training.window_mode),
                )[0]
                payload = _produce_episode(
                    episode=episode,
                    collator=collator,
                    system=system,
                    class_mapper=class_mapper,
                    device=device,
                    key=key,
                )
            record = write_task_memory_cache(
                cache_root, payload, max_total_bytes=CACHE_LIMIT_BYTES
            )
            produced += 1
        payloads.append(payload)
        cache_records.append(
            {
                **record,
                "reference_id": spec.reference_id,
                "sequence_id": spec.source_sequence_id,
            }
        )
        print(
            json.dumps(
                {
                    "episode": index + 1,
                    "produced": produced,
                    "reused": reused,
                    "total": len(selected),
                    "variant": variant,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed_seconds = time.time() - started
    metric_rows = compute_cached_task_metrics(
        payloads,
        reducers=reducers,
        class_mapper=class_mapper,
    )
    identity_metric_rows = compute_cached_identity_metrics(
        payloads, class_mapper=class_mapper
    )
    retention_metric_rows = build_retention_rows(metric_rows)
    common_columns = {
        "population_id": POPULATION_ID,
        "variant": variant,
        "checkpoint_sha256": checkpoint_sha256,
        "training_seed": 45,
        "evaluation_seed": EVALUATION_SEED,
    }
    task_rows = [
        {
            **common_columns,
            **row,
            "reference_count": len({master.reference_id for master in selected}),
            "master_count": len(selected),
        }
        for row in metric_rows
    ]
    identity_rows = [
        {
            **common_columns,
            **row,
            "reference_count": len({master.reference_id for master in selected}),
            "master_count": len(selected),
        }
        for row in identity_metric_rows
    ]
    retention_rows = [
        {
            **common_columns,
            **row,
            "reference_count": len({master.reference_id for master in selected}),
            "master_count": len(selected),
        }
        for row in retention_metric_rows
    ]
    metrics_path = output_root / "metrics.csv"
    identity_path = output_root / "identity_metrics.csv"
    retention_path = output_root / "retention.csv"
    _atomic_bytes(metrics_path, _csv_bytes(task_rows))
    _atomic_bytes(identity_path, _csv_bytes(identity_rows))
    _atomic_bytes(retention_path, _csv_bytes(retention_rows))
    event_records = [
        stage["event_diagnostics"]
        for payload in payloads
        for stage in payload["stages"]
    ]
    result: dict[str, object] = {
        "cache": {
            "bytes": sum(record["file_bytes"] for record in cache_records),
            "cap_bytes": CACHE_LIMIT_BYTES,
            "entry_count": len(cache_records),
            "records": cache_records,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "determinism": {
            "cudnn_deterministic": True,
            "episode_seed_reset": True,
            "tf32": False,
            "torch_deterministic_algorithms": True,
        },
        "elapsed_seconds": elapsed_seconds,
        "evaluation_seed": EVALUATION_SEED,
        "identity_events": aggregate_identity_event_diagnostics(event_records),
        "load_audit": load_audit,
        "identity_metrics": _artifact_record(
            identity_path, row_count=len(identity_rows)
        ),
        "metrics": _artifact_record(metrics_path, row_count=len(task_rows)),
        "population": {
            "id": POPULATION_ID,
            "master_count": len(selected),
            "reference_count": len({master.reference_id for master in selected}),
            "smoke": smoke_master_count is not None,
        },
        "postprocess_sha256": postprocess_sha256,
        "produced": produced,
        "reducers": list(reducers),
        "retention": _artifact_record(
            retention_path, row_count=len(retention_rows)
        ),
        "reused": reused,
        "schema_version": "task-memory-evaluation-run-v2",
        "source_commit": source_commit,
        "state_contract_sha256": state_contract_sha256,
        "status": "PASS",
        "variant": variant,
    }
    result["content_sha256"] = canonical_json_sha256(result)
    _atomic_json(output_root / "manifest.json", result)
    del system
    torch.cuda.empty_cache()
    return result


def _parser() -> argparse.ArgumentParser:
    resolver = _load_json(LOCAL_ASSET_RESOLVER) if LOCAL_ASSET_RESOLVER.exists() else {}

    def external_default(environment: str, logical: str):
        value = os.environ.get(environment, resolver.get(logical))
        return Path(value) if value else None

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=external_default(
            "PERSIST4D_CONCERTO_PRETRAINED", "external:concerto_pretrained"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=external_default("PERSIST4D_DATA_ROOT", "external:data_root"),
    )
    parser.add_argument(
        "--rio-metadata",
        type=Path,
        default=external_default("PERSIST4D_RIO_METADATA", "external:rio_metadata"),
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=external_default("PERSIST4D_RUN_ROOT", "external:run_root"),
    )
    parser.add_argument("--data-contract", type=Path, default=DEFAULT_DATA_CONTRACT)
    parser.add_argument(
        "--variant-manifest", type=Path, default=DEFAULT_VARIANT_MANIFEST
    )
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output", "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-masters", type=int)
    parser.add_argument("--reducers", nargs="+", default=list(REDUCERS))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    missing = [
        name
        for name, value in (
            ("--pretrained", args.pretrained),
            ("--data-root", args.data_root),
            ("--rio-metadata", args.rio_metadata),
            ("--external-root", args.external_root),
        )
        if value is None
    ]
    if missing:
        parser.error("missing external inputs: " + ", ".join(missing))
    cache_root = args.cache_root or (
        args.external_root / "evaluation_cache" / args.variant
    )
    run_evaluation(
        variant=args.variant,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        data_root=args.data_root,
        rio_metadata=args.rio_metadata,
        external_root=args.external_root,
        cache_root=cache_root,
        output_root=args.output,
        data_contract_path=args.data_contract,
        variant_manifest_path=args.variant_manifest,
        device_name=args.device,
        smoke_master_count=args.smoke_masters,
        reducers=args.reducers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
