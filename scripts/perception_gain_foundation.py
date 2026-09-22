#!/usr/bin/env python3
"""Measured foundation stage for the Perception-Gain v1 campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.task_memory_contracts import canonical_json_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/perception_gain_v1"
DEFAULT_EXTERNAL_ROOT = Path("/home/ww/persist4d_runs/perception_gain_v1")
DEFAULT_ASSETS = DEFAULT_EXTERNAL_ROOT / "assets.local.json"
LEGACY_CROSSWINDOW_ASSETS = Path(
    "/mnt/shared/ww/persist4d-crosswindow-evidence-v1/assets.local.json"
)
R1_SELECTED = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/full_candidate/selected_checkpoint_manifest.json"
)
R1_VARIANTS = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/short_curves/variant_manifest.json"
)
R1_EVALUATION = (
    PROJECT_ROOT
    / "artifacts/rescene_task_learning_root_cause_v1/full_candidate/FULL_EVALUATION_PROVENANCE.json"
)
DATA_ROLES = ARTIFACT_ROOT / "DATA_ROLES.json"
DATA_CONTRACT = PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
DEV_BASE_MANIFEST = (
    PROJECT_ROOT / "artifacts/task_memory_retention_v2/baseline/cache_manifest.json"
)
DEV_SUPPLEMENT_MANIFEST = (
    PROJECT_ROOT
    / "artifacts/task_memory_retention_v2/baseline/control_observation_manifest.json"
)


class FoundationError(RuntimeError):
    """Raised when a foundation input or measurement violates its contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FoundationError(f"cannot decode JSON input: {path}") from error
    if not isinstance(value, dict):
        raise FoundationError(f"JSON input must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii"),
    )


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, object]], *, fieldnames: Sequence[str]
) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


def _git_head() -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40:
        raise FoundationError("Git HEAD is invalid")
    return value


def build_r1_binding(
    *,
    selected_checkpoint: Mapping[str, Any],
    variant_manifest: Mapping[str, Any],
    evaluation_provenance: Mapping[str, Any],
    local_t2_references: Sequence[str],
) -> dict[str, object]:
    checkpoint = selected_checkpoint.get("checkpoint")
    variants = variant_manifest.get("variants")
    r1 = variants.get("R1") if isinstance(variants, Mapping) else None
    resolved = r1.get("resolved_config") if isinstance(r1, Mapping) else None
    if not isinstance(checkpoint, Mapping) or not isinstance(resolved, Mapping):
        raise FoundationError("R1 checkpoint or resolved config is unavailable")
    config_sha = canonical_json_sha256(resolved)
    if (
        r1.get("config_sha256") != config_sha
        or checkpoint.get("config_sha256") != config_sha
        or checkpoint.get("sha256") is None
        or checkpoint.get("bytes") is None
    ):
        raise FoundationError("R1 resolved config binding differs")
    general = resolved.get("general")
    if (
        not isinstance(general, Mapping)
        or general.get("rootcause_objective_mode") != "raw_sum"
    ):
        raise FoundationError("R1 objective mode is not raw_sum")
    references = sorted(set(local_t2_references))
    if not references or len(references) != len(local_t2_references):
        raise FoundationError("LOCAL-T2 references must be unique and non-empty")
    run_sources = evaluation_provenance.get("run_sources")
    if not isinstance(run_sources, Mapping) or not run_sources:
        raise FoundationError("LOCAL-T2 evaluation provenance is unavailable")
    return {
        "schema_version": "perception-gain-foundation-binding-v1",
        "status": "PASS",
        "r1": {
            "checkpoint_bytes": int(checkpoint["bytes"]),
            "checkpoint_reference": str(checkpoint.get("reference")),
            "checkpoint_sha256": str(checkpoint["sha256"]),
            "objective_mode": str(general["rootcause_objective_mode"]),
            "resolved_config_sha256": config_sha,
            "selected_epoch": int(checkpoint.get("selected_epoch", -1)),
            "selected_step": int(checkpoint.get("selected_step", -1)),
        },
        "local_t2": {
            "evaluation_provenance_content_sha256": evaluation_provenance.get(
                "content_sha256"
            ),
            "physical_reference_count": len(references),
            "physical_reference_ids": references,
            "run_sources": dict(sorted(run_sources.items())),
            "validation_sequence_count": 154,
        },
    }


def foundation_hash_order(reference_id: str, sequence_id: str) -> tuple[str, str]:
    if not reference_id or not sequence_id:
        raise FoundationError("foundation master identity is empty")
    digest = hashlib.sha256(
        f"perception-gain-foundation:{reference_id}:{sequence_id}".encode("utf-8")
    ).hexdigest()
    return digest, sequence_id


def select_cal_panel_masters(
    masters: Sequence[Any], cal_references: Sequence[str]
) -> tuple[Any, ...]:
    references = tuple(sorted(set(cal_references)))
    if not references or len(references) != len(cal_references):
        raise FoundationError("CAL references must be unique and non-empty")
    selected = []
    for reference in references:
        candidates = [
            master
            for master in masters
            if getattr(master, "reference_id", None) == reference
            and isinstance(getattr(master, "sequence_id", None), str)
        ]
        if not candidates:
            raise FoundationError(f"CAL reference lacks a native master: {reference}")
        selected.append(
            min(
                candidates,
                key=lambda master: foundation_hash_order(reference, master.sequence_id),
            )
        )
    return tuple(selected)


def _tensor_leaves(value: object, prefix: str) -> dict[str, Tensor]:
    if isinstance(value, Tensor):
        return {prefix: value.detach().cpu()}
    if isinstance(value, Mapping):
        result: dict[str, Tensor] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_tensor_leaves(item, child))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = {}
        for index, item in enumerate(value):
            result.update(_tensor_leaves(item, f"{prefix}[{index}]"))
        return result
    return {}


def compare_live_cached_supplements(
    live: Mapping[str, object], cached: Mapping[str, object]
) -> dict[str, object]:
    live_stages = live.get("stages")
    cached_stages = cached.get("stages")
    if (
        isinstance(live_stages, (str, bytes))
        or not isinstance(live_stages, Sequence)
        or isinstance(cached_stages, (str, bytes))
        or not isinstance(cached_stages, Sequence)
        or len(live_stages) != len(cached_stages)
        or not live_stages
    ):
        raise FoundationError("live/cached stage coverage differs")
    live_tensors = _tensor_leaves({"stages": live_stages}, "")
    cached_tensors = _tensor_leaves({"stages": cached_stages}, "")
    mismatches = sorted(
        path
        for path in set(live_tensors) | set(cached_tensors)
        if path not in live_tensors
        or path not in cached_tensors
        or not torch.equal(live_tensors[path], cached_tensors[path])
    )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "compared_tensor_count": len(set(live_tensors) | set(cached_tensors)),
        "mismatches": mismatches,
    }


def best_candidate_iou_records(
    *,
    file_name: str,
    output: Mapping[str, object],
    target: Mapping[str, object],
) -> list[dict[str, object]]:
    """Report each GT's strongest thresholded query at every decoder head."""

    masks = target.get("masks")
    point2segment = target.get("point2segment")
    instance_ids = target.get("ids")
    auxiliary = output.get("aux_outputs")
    if (
        not file_name
        or not isinstance(masks, Tensor)
        or masks.ndim != 2
        or masks.shape[0] == 0
        or not isinstance(point2segment, Tensor)
        or point2segment.ndim != 1
        or point2segment.numel() != masks.shape[1]
        or not isinstance(instance_ids, Tensor)
        or instance_ids.ndim != 1
        or instance_ids.numel() != masks.shape[0]
        or not isinstance(auxiliary, list)
        or any(not isinstance(layer, Mapping) for layer in auxiliary)
    ):
        raise FoundationError("best-candidate diagnostic contract differs")
    point2segment = point2segment.long()
    if point2segment.numel() == 0 or point2segment.min().item() < 0:
        raise FoundationError("best-candidate point-to-segment mapping differs")

    rows: list[dict[str, object]] = []
    for layer_index, layer in enumerate([*auxiliary, output]):
        layer_masks = layer.get("pred_masks")
        if (
            not isinstance(layer_masks, list)
            or len(layer_masks) != 1
            or not isinstance(layer_masks[0], Tensor)
            or layer_masks[0].ndim != 2
            or point2segment.max().item() >= layer_masks[0].shape[0]
        ):
            raise FoundationError("best-candidate prediction contract differs")
        predicted = layer_masks[0][point2segment].sigmoid().transpose(0, 1) >= 0.5
        gt = masks.bool().to(predicted.device)
        intersection = (predicted[:, None, :] & gt[None, :, :]).sum(dim=2).float()
        union = (predicted[:, None, :] | gt[None, :, :]).sum(dim=2).float()
        iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
        best_iou, best_query = iou.max(dim=0)
        for gt_index in range(gt.shape[0]):
            rows.append(
                {
                    "file_name": file_name,
                    "decoder_prediction_layer": layer_index,
                    "gt_instance_id": int(instance_ids[gt_index].item()),
                    "best_candidate_query_id": int(best_query[gt_index].item()),
                    "best_candidate_iou": float(best_iou[gt_index].item()),
                }
            )
    return rows


class FoundationDiagnosticCollector:
    """Capture the three V1 panel diagnostics during one native forward."""

    def __init__(self, system: object) -> None:
        from utils.rescene_rootcause_diagnostics import (
            attention_mask_records,
            query_initialization_records,
        )

        self._attention_mask_records = attention_mask_records
        self._query_initialization_records = query_initialization_records
        self.query_rows: list[dict[str, object]] = []
        self.attention_rows: list[dict[str, object]] = []
        self.best_candidate_rows: list[dict[str, object]] = []
        self.sequence_count = 0
        self._initial_queries: list[dict[str, Tensor]] = []
        self._reset_layers: list[list[dict[str, int]]] = []

        model = getattr(system, "model", None)
        initialize_queries = getattr(model, "initialize_queries", None)
        sample_and_batch_features = getattr(model, "sample_and_batch_features", None)
        if not callable(initialize_queries) or not callable(sample_and_batch_features):
            raise FoundationError("R1 diagnostic hooks are unavailable")

        def capture_initialize(*args: object, **kwargs: object) -> object:
            result = initialize_queries(*args, **kwargs)
            pcd_features = kwargs.get("pcd_features")
            coords = kwargs.get("coords")
            if pcd_features is None and args:
                pcd_features = args[0]
            if coords is None and len(args) > 1:
                coords = args[1]
            queries, _, sampled_coords = result
            if (
                sampled_coords is None
                or not isinstance(coords, list)
                or not isinstance(queries, Tensor)
            ):
                raise FoundationError("query diagnostic capture differs")
            captured = []
            for batch_index, full_coordinates in enumerate(coords[-1]):
                sampled = sampled_coords[batch_index]
                matches = (full_coordinates[:, None, :] == sampled[None, :, :]).all(
                    dim=2
                )
                if not torch.all(matches.sum(dim=0) == 1).item():
                    raise FoundationError("query coordinate binding differs")
                captured.append(
                    {
                        "sampled_indices": matches.long().argmax(dim=0).detach().cpu(),
                        "query_content_norms": queries[batch_index]
                        .norm(dim=-1)
                        .detach()
                        .cpu(),
                    }
                )
            self._initial_queries = captured
            return result

        def capture_sample(*args: object, **kwargs: object) -> object:
            result = sample_and_batch_features(*args, **kwargs)
            if kwargs.get("extra") is not None:
                batched_attention = result[1]
                if not isinstance(batched_attention, Tensor):
                    raise FoundationError("attention diagnostic capture differs")
                all_masked = batched_attention.sum(dim=1) == batched_attention.shape[1]
                self._reset_layers.append(
                    [
                        {
                            "reset_count": int(all_masked[index].sum().item()),
                            "query_count": int(all_masked.shape[1]),
                        }
                        for index in range(all_masked.shape[0])
                    ]
                )
            return result

        model.initialize_queries = capture_initialize
        model.sample_and_batch_features = capture_sample

    def collect(
        self,
        *,
        system: object,
        data: object,
        target: Mapping[str, object],
        file_name: str,
        metadata: Mapping[str, object],
    ) -> None:
        self._initial_queries = []
        self._reset_layers = []
        raw_coordinates = system._process_raw_coordinates(data)
        with torch.inference_mode():
            output = system(
                data,
                point2segment=[target["point2segment"]],
                raw_coordinates=raw_coordinates,
                is_eval=True,
                targets=[target],
            )
        if len(self._initial_queries) != 1 or any(
            len(layer) != 1 for layer in self._reset_layers
        ):
            raise FoundationError("diagnostic batch capture differs")
        query_rows = self._query_initialization_records(
            file_name=file_name,
            sampled_indices=self._initial_queries[0]["sampled_indices"],
            query_content_norms=self._initial_queries[0]["query_content_norms"],
            target=target,
        )
        attention_rows = self._attention_mask_records(
            file_name=file_name,
            output=output,
            target=target,
            reset_counts=[layer[0] for layer in self._reset_layers],
        )
        candidate_rows = best_candidate_iou_records(
            file_name=file_name, output=output, target=target
        )
        self.query_rows.extend({**metadata, **row} for row in query_rows)
        self.attention_rows.extend({**metadata, **row} for row in attention_rows)
        self.best_candidate_rows.extend({**metadata, **row} for row in candidate_rows)
        self.sequence_count += 1


def _csv_fields(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(key for row in rows for key in row))


def _resolve_cache_assets(assets_path: Path) -> dict[str, Any]:
    assets = _read_json(assets_path)
    required = (
        "r1_checkpoint",
        "concerto_pretrained",
        "data_root",
        "rio_metadata",
        "metric_dataset_spec",
        "dev_base_cache_root",
        "dev_supplement_root",
    )
    missing = [key for key in required if not isinstance(assets.get(key), str)]
    if missing and LEGACY_CROSSWINDOW_ASSETS.is_file():
        legacy = _read_json(LEGACY_CROSSWINDOW_ASSETS)
        for key in (
            "dev_base_cache_root",
            "dev_supplement_root",
            "pb_base_cache_root",
            "pb_supplement_root",
        ):
            if not isinstance(assets.get(key), str) and isinstance(
                legacy.get(key), str
            ):
                assets[key] = legacy[key]
    missing = [key for key in required if not isinstance(assets.get(key), str)]
    if missing:
        raise FoundationError(f"foundation assets are unresolved: {missing}")
    _atomic_json(assets_path, assets)
    return assets


def run_bind(*, assets_path: Path = DEFAULT_ASSETS) -> dict[str, object]:
    roles = _read_json(DATA_ROLES).get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("LOCAL-T2"), list):
        raise FoundationError("LOCAL-T2 role is unavailable")
    selected = _read_json(R1_SELECTED)
    variants = _read_json(R1_VARIANTS)
    evaluation = _read_json(R1_EVALUATION)
    binding = build_r1_binding(
        selected_checkpoint=selected,
        variant_manifest=variants,
        evaluation_provenance=evaluation,
        local_t2_references=roles["LOCAL-T2"],
    )
    assets = _resolve_cache_assets(assets_path)
    checkpoint = Path(assets["r1_checkpoint"])
    expected = binding["r1"]
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size != expected["checkpoint_bytes"]
        or _sha256(checkpoint) != expected["checkpoint_sha256"]
    ):
        raise FoundationError("resolved R1 checkpoint identity differs")
    binding["sources"] = {
        "evaluation_provenance": {
            "logical_reference": "repo:artifacts/rescene_task_learning_root_cause_v1/full_candidate/FULL_EVALUATION_PROVENANCE.json",
            "sha256": _sha256(R1_EVALUATION),
        },
        "selected_checkpoint_manifest": {
            "logical_reference": "repo:artifacts/rescene_task_learning_root_cause_v1/full_candidate/selected_checkpoint_manifest.json",
            "sha256": _sha256(R1_SELECTED),
        },
        "variant_manifest": {
            "logical_reference": "repo:artifacts/rescene_task_learning_root_cause_v1/short_curves/variant_manifest.json",
            "sha256": _sha256(R1_VARIANTS),
        },
    }
    _atomic_json(ARTIFACT_ROOT / "foundation/R1_BINDING.json", binding)
    resolved = variants["variants"]["R1"]["resolved_config"]
    _atomic_json(
        ARTIFACT_ROOT / "foundation/R1_RESOLVED_CONFIG.json",
        {
            "schema_version": "perception-gain-r1-resolved-config-v1",
            "config_sha256": canonical_json_sha256(resolved),
            "resolved_config": resolved,
        },
    )
    return binding


def _metric_class_mapping(path: Path) -> tuple[int, ...]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = value.get("valid_class_ids") if isinstance(value, Mapping) else None
    if (
        isinstance(classes, (str, bytes))
        or not isinstance(classes, Sequence)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in classes)
    ):
        raise FoundationError("metric class mapping is unavailable")
    return tuple(classes)


def run_corrected_e1(*, assets_path: Path = DEFAULT_ASSETS) -> dict[str, object]:
    from scripts.crosswindow_campaign import iter_campaign_units
    from scripts.diagnose_crosswindow_failures import E1DiagnosticAccumulator

    status_path = ARTIFACT_ROOT / "foundation/e1/status.json"
    if status_path.is_file():
        existing = _read_json(status_path)
        if (
            existing.get("status") == "PASS"
            and existing.get("diagnostic_source_sha256")
            == _sha256(PROJECT_ROOT / "scripts/diagnose_crosswindow_failures.py")
            and existing.get("metric_source_sha256")
            == _sha256(PROJECT_ROOT / "scripts/p6a_metrics.py")
        ):
            return existing
    assets = _resolve_cache_assets(assets_path)
    base_root = Path(assets["dev_base_cache_root"])
    supplement_root = Path(assets["dev_supplement_root"])
    base_manifest = _read_json(DEV_BASE_MANIFEST)
    supplement_manifest = _read_json(DEV_SUPPLEMENT_MANIFEST)
    roles = _read_json(DATA_ROLES).get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get("CAL"), list):
        raise FoundationError("CAL role is unavailable")
    units = list(
        iter_campaign_units(
            role="CAL",
            role_references=roles["CAL"],
            base_manifest=base_manifest,
            supplement_manifest=supplement_manifest,
        )
    )
    accumulator = E1DiagnosticAccumulator(
        dataset_spec=str(Path(assets["metric_dataset_spec"])),
        class_mapping=_metric_class_mapping(Path(assets["metric_dataset_spec"])),
        source_commit=_git_head(),
        checkpoint_sha256=str(supplement_manifest["checkpoint_sha256"]),
    )
    started = time.perf_counter()
    for index, unit in enumerate(units, start=1):
        base_path = base_root / unit.base_filename
        supplement_path = supplement_root / unit.supplement_filename
        if (
            _sha256(base_path) != unit.base_sha256
            or _sha256(supplement_path) != unit.supplement_sha256
        ):
            raise FoundationError("CAL cache identity differs")
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        supplement = torch.load(supplement_path, map_location="cpu", weights_only=False)
        accumulator.update(
            logical_unit_id=unit.logical_unit_id,
            base=base,
            supplement=supplement,
        )
        print(
            json.dumps(
                {"stage": "foundation-e1", "completed": index, "total": len(units)},
                sort_keys=True,
            ),
            flush=True,
        )
    result = accumulator.finalize()
    output = ARTIFACT_ROOT / "foundation/e1"
    _atomic_csv(
        output / "gt_assisted.csv",
        result["metric_rows"],
        fieldnames=tuple(result["metric_rows"][0]),
    )
    _atomic_csv(
        output / "candidate_coverage.csv",
        result["coverage_rows"],
        fieldnames=tuple(result["coverage_rows"][0]),
    )
    _atomic_csv(
        output / "failure_ledger.csv",
        result["coverage_events"],
        fieldnames=tuple(result["coverage_events"][0]),
    )
    _atomic_csv(
        output / "gt_assignment_events.csv",
        result["assignment_events"],
        fieldnames=tuple(result["assignment_events"][0]),
    )
    _atomic_json(output / "gate.json", result["gate"])
    summary = {
        "schema_version": "perception-gain-foundation-e1-v1",
        "status": result["status"],
        "data_role": "CAL",
        "reference_count": len(roles["CAL"]),
        "logical_unit_count": len(units),
        "coverage_event_count": len(result["coverage_events"]),
        "assignment_event_count": len(result["assignment_events"]),
        "gate_decision": result["gate"]["decision"],
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_source_sha256": _sha256(
            PROJECT_ROOT / "scripts/diagnose_crosswindow_failures.py"
        ),
        "metric_source_sha256": _sha256(PROJECT_ROOT / "scripts/p6a_metrics.py"),
    }
    _atomic_json(output / "status.json", summary)
    return summary


def _torch_save(path: Path, value: object) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def run_live_smoke(
    *,
    assets_path: Path = DEFAULT_ASSETS,
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    status_path = ARTIFACT_ROOT / "foundation/LIVE_SMOKE.json"
    if status_path.is_file():
        existing = _read_json(status_path)
        if existing.get("status") == "PASS" and existing.get(
            "foundation_source_sha256"
        ) == _sha256(Path(__file__)):
            return existing

    import hydra
    import yaml
    from omegaconf import OmegaConf

    from datasets.task_memory_episode import (
        TaskMemoryEpisodeCollator,
        TaskMemoryEpisodeDataset,
    )
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _validate_cuda_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        build_rio_class_mapper,
    )
    from scripts.preflight_task_memory_episode import _rio_base_dataset
    from scripts.replay_crosswindow_association import E0ReplayAccumulator
    from scripts.run_task_memory_controls import (
        _base_cache_link,
        _produce_supplement,
    )
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        _episode_specs,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import (
        FullHistoryPredictionProducer,
        deterministic_inference_runtime,
        full_history_prediction_fingerprint,
    )
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import (
        PerceptionGainTrainer,
        strict_load_r1_perception,
    )

    assets = _resolve_cache_assets(assets_path)
    device = _validate_cuda_device(device_name)
    run_dir = external_root / "foundation/live"
    config = compose_variant_config(
        "C0",
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=run_dir,
    )
    # D0 observation construction consumes the decoder's final query embeddings.
    # This flag only exposes an existing tensor and has no parameters or numerical
    # effect on the prediction heads.
    config.model.return_query_features = True
    base_dataset = _rio_base_dataset(
        config, data_root=Path(assets["data_root"]), horizon=5
    )
    base_dataset, masters, _ = build_baseline_population(
        base_dataset,
        data_contract=_read_json(DATA_CONTRACT),
        metadata_path=Path(assets["rio_metadata"]),
        population_id=DEVELOPMENT_POPULATION_ID,
        protocol_b_manifest_path=(
            PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
        ),
    )
    roles = _read_json(DATA_ROLES)["roles"]
    panel = select_cal_panel_masters(masters, roles["CAL"])
    smoke_master = min(
        panel,
        key=lambda master: foundation_hash_order(
            master.reference_id, master.sequence_id
        ),
    )
    specs = _episode_specs(masters)
    spec = next(
        item for item in specs if item.source_sequence_id == smoke_master.sequence_id
    )

    checkpoint_path = Path(assets["r1_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise FoundationError("R1 checkpoint lacks state_dict")
    system = PerceptionGainTrainer(config)
    load_audit = strict_load_r1_perception(system, state, allow_semantic_scorer=False)
    system.to(device).eval().requires_grad_(False)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    class_mapper = build_rio_class_mapper(base_dataset)
    p6a = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())
    settings = p6a["baselines"]["b4"]
    observation_settings = {
        "background_class": int(settings["background_class"]),
        "confidence_threshold": float(settings["confidence_threshold"]),
        "mask_threshold": float(settings["mask_threshold"]),
        "minimum_mask_support": int(settings["minimum_mask_support"]),
    }
    provenance = {
        "source_commit": _git_head(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config_sha256": canonical_json_sha256(
            OmegaConf.to_container(config, resolve=True)
        ),
        "protocol_sha256": _sha256(DATA_ROLES),
    }
    producer = FullHistoryPredictionProducer(
        dataset=base_dataset,
        collate=collate,
        system=system,
        device=device,
        provenance=provenance,
        class_mapper=class_mapper,
        move_data=_move_data_to_device,
        move_targets=_move_targets_to_device,
        **observation_settings,
        seed=45,
    )
    native = {}
    started_all = time.perf_counter()
    with deterministic_inference_runtime(45, device):
        for horizon in (2, 5):
            key = {
                "master_sequence_id": smoke_master.sequence_id,
                "reference_scene_id": smoke_master.reference_id,
                "order_id": "canonical",
                "context_index": smoke_master.context_index,
                "context_scan_indices": list(smoke_master.scan_indices),
                "horizon": horizon,
                "history_scan_ids": list(smoke_master.scan_ids[:horizon]),
                "scan_indices": list(smoke_master.scan_indices[:horizon]),
                "task_quality": True,
            }
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            produced = producer.produce_bundle(key)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            output_path = run_dir / f"native_T{horizon}.pt"
            identity = _torch_save(output_path, produced.payload)
            native[f"T{horizon}"] = {
                **identity,
                "cache_reference": f"external:foundation/live/{output_path.name}",
                "content_sha256": produced.payload["content_sha256"],
                "elapsed_seconds": elapsed,
                "input_stats": produced.payload["input_stats"],
                "prediction_fingerprint": full_history_prediction_fingerprint(
                    produced.payload
                ),
            }

        base_manifest = _read_json(DEV_BASE_MANIFEST)
        supplement_manifest = _read_json(DEV_SUPPLEMENT_MANIFEST)
        base_record = next(
            record
            for record in base_manifest["records"]
            if record["sequence_id"] == smoke_master.sequence_id
        )
        cached_record = next(
            record
            for record in supplement_manifest["records"]
            if record["sequence_id"] == smoke_master.sequence_id
        )
        base_path = Path(assets["dev_base_cache_root"]) / base_record["filename"]
        cached_path = Path(assets["dev_supplement_root"]) / cached_record["filename"]
        if (
            _sha256(base_path) != base_record["sha256"]
            or _sha256(cached_path) != cached_record["sha256"]
        ):
            raise FoundationError("cached smoke input identity differs")
        base_cache = torch.load(base_path, map_location="cpu", weights_only=False)
        cached = torch.load(cached_path, map_location="cpu", weights_only=False)
        episode = TaskMemoryEpisodeDataset(
            base_dataset, (spec,), apply_augmentation=False
        )[0]
        with _frozen_inference_seed(45, device):
            live = _produce_supplement(
                episode=episode,
                collator=TaskMemoryEpisodeCollator(collate),
                system=system,
                observation_settings=observation_settings,
                class_mapper=class_mapper,
                device=device,
                provenance={
                    "source_commit": provenance["source_commit"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "base_cache_manifest_sha256": base_manifest["content_sha256"],
                    "config_sha256": provenance["config_sha256"],
                },
                base_cache=base_cache,
                base_link=_base_cache_link(base_record),
            )
        with _frozen_inference_seed(45, device):
            live_repeat = _produce_supplement(
                episode=episode,
                collator=TaskMemoryEpisodeCollator(collate),
                system=system,
                observation_settings=observation_settings,
                class_mapper=class_mapper,
                device=device,
                provenance={
                    "source_commit": provenance["source_commit"],
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "base_cache_manifest_sha256": base_manifest["content_sha256"],
                    "config_sha256": provenance["config_sha256"],
                },
                base_cache=base_cache,
                base_link=_base_cache_link(base_record),
            )
    parity = compare_live_cached_supplements(live, cached)
    reproducibility = compare_live_cached_supplements(live, live_repeat)
    if reproducibility["status"] != "PASS":
        raise FoundationError("current live D0 is not tensor-exactly reproducible")
    e0 = E0ReplayAccumulator(
        dataset_spec=str(Path(assets["metric_dataset_spec"])),
        class_mapping=_metric_class_mapping(Path(assets["metric_dataset_spec"])),
        checkpoint_sha256=provenance["checkpoint_sha256"],
        source_commit=provenance["source_commit"],
        index_trigger_count=0,
    )
    e0.update(logical_unit_id="foundation-live-smoke", base=base_cache, supplement=live)
    d0_result = e0.finalize()
    d0_metrics = [
        row
        for row in d0_result["metric_rows"]
        if row["method"] == "D0" and row["reference"] == "all"
    ]
    live_identity = _torch_save(run_dir / "live_supplement.pt", live)
    elapsed_all = time.perf_counter() - started_all
    summary = {
        "schema_version": "perception-gain-foundation-live-v1",
        "status": "PASS",
        "reference_id": smoke_master.reference_id,
        "sequence_id": smoke_master.sequence_id,
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "load_audit": load_audit,
        "native": native,
        "d0_live_metrics": d0_metrics,
        "cached_live_parity": parity,
        "live_repeat_parity": reproducibility,
        "historical_cache_disposition": (
            "CURRENT_EQUIVALENT"
            if parity["status"] == "PASS"
            else "SEPARATE_HISTORICAL_ROW"
        ),
        "historical_cache_provenance": cached.get("provenance"),
        "live_supplement": {
            **live_identity,
            "cache_reference": "external:foundation/live/live_supplement.pt",
        },
        "elapsed_seconds": elapsed_all,
        "gpu_hours": elapsed_all / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "foundation_source_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(ARTIFACT_ROOT / "foundation/LIVE_SMOKE.json", summary)
    del system
    torch.cuda.empty_cache()
    return summary


def run_diagnostic_panel(
    *,
    assets_path: Path = DEFAULT_ASSETS,
    device_name: str = "cuda:0",
) -> dict[str, object]:
    status_path = ARTIFACT_ROOT / "foundation/diagnostic_panel/status.json"
    if status_path.is_file():
        existing = _read_json(status_path)
        if existing.get("status") == "PASS" and existing.get(
            "foundation_source_sha256"
        ) == _sha256(Path(__file__)):
            return existing

    import hydra

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _validate_cuda_device,
    )
    from scripts.preflight_task_memory_episode import _rio_base_dataset
    from scripts.run_task_memory_policy_baseline import (
        DEVELOPMENT_POPULATION_ID,
        build_baseline_population,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime
    from scripts.train_perception_gain import compose_variant_config
    from trainer.perception_gain_trainer import (
        PerceptionGainTrainer,
        strict_load_r1_perception,
    )

    assets = _resolve_cache_assets(assets_path)
    device = _validate_cuda_device(device_name)
    config = compose_variant_config(
        "C0",
        pretrained=Path(assets["concerto_pretrained"]),
        run_dir=DEFAULT_EXTERNAL_ROOT / "foundation/diagnostic_panel",
    )
    config.model.return_query_features = True
    base_dataset = _rio_base_dataset(
        config, data_root=Path(assets["data_root"]), horizon=5
    )
    base_dataset, masters, _ = build_baseline_population(
        base_dataset,
        data_contract=_read_json(DATA_CONTRACT),
        metadata_path=Path(assets["rio_metadata"]),
        population_id=DEVELOPMENT_POPULATION_ID,
        protocol_b_manifest_path=(
            PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
        ),
    )
    roles = _read_json(DATA_ROLES)["roles"]
    panel = select_cal_panel_masters(masters, roles["CAL"])

    checkpoint_path = Path(assets["r1_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise FoundationError("R1 checkpoint lacks state_dict")
    system = PerceptionGainTrainer(config)
    load_audit = strict_load_r1_perception(system, state, allow_semantic_scorer=False)
    system.to(device).eval().requires_grad_(False)
    collate = hydra.utils.instantiate(config.data.validation_collation)
    collector = FoundationDiagnosticCollector(system)
    incomplete_units: list[dict[str, object]] = []
    expected_units = sum(len(master.scan_indices) for master in panel)

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with deterministic_inference_runtime(45, device):
        for master in panel:
            for horizon in range(1, len(master.scan_indices) + 1):
                file_name = f"{master.reference_id}/{master.sequence_id}/T{horizon}"
                try:
                    sample = base_dataset.load_scan_indices(
                        master.context_index,
                        tuple(master.scan_indices[:horizon]),
                        change_file=None,
                    )
                    data, targets, names = collate([sample])
                    if list(names) != [master.sequence_id] or len(targets) != 1:
                        raise FoundationError("diagnostic collator identity differs")
                    target = targets[0]
                    masks = target.get("masks")
                    if (
                        not isinstance(masks, Tensor)
                        or masks.ndim != 2
                        or not masks.shape[0]
                    ):
                        raise FoundationError("diagnostic unit has no evaluated GT")
                    data = _move_data_to_device(data, device)
                    targets = _move_targets_to_device(targets, device)
                    collector.collect(
                        system=system,
                        data=data,
                        target=targets[0],
                        file_name=file_name,
                        metadata={
                            "reference_id": master.reference_id,
                            "sequence_id": master.sequence_id,
                            "prefix_T": horizon,
                        },
                    )
                except (FoundationError, RuntimeError, ValueError) as error:
                    incomplete_units.append(
                        {
                            "file_name": file_name,
                            "error_type": type(error).__name__,
                            "reason": str(error),
                        }
                    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    output = ARTIFACT_ROOT / "foundation/diagnostic_panel"
    if collector.query_rows:
        _atomic_csv(
            output / "query_coverage.csv",
            collector.query_rows,
            fieldnames=_csv_fields(collector.query_rows),
        )
    if collector.attention_rows:
        _atomic_csv(
            output / "attention_reachability.csv",
            collector.attention_rows,
            fieldnames=_csv_fields(collector.attention_rows),
        )
    if collector.best_candidate_rows:
        _atomic_csv(
            output / "best_candidate_iou.csv",
            collector.best_candidate_rows,
            fieldnames=_csv_fields(collector.best_candidate_rows),
        )

    query_summaries = [
        row for row in collector.query_rows if row["record_type"] == "scene_summary"
    ]
    earliest_layer = min(
        (int(row["decoder_prediction_layer"]) for row in collector.attention_rows),
        default=None,
    )
    earliest_attention = [
        float(row["allowed_gt_fraction"])
        for row in collector.attention_rows
        if row["decoder_prediction_layer"] == earliest_layer
    ]
    candidate_by_layer: dict[str, float] = {}
    for layer in sorted(
        {int(row["decoder_prediction_layer"]) for row in collector.best_candidate_rows}
    ):
        values = [
            float(row["best_candidate_iou"])
            for row in collector.best_candidate_rows
            if row["decoder_prediction_layer"] == layer
        ]
        candidate_by_layer[str(layer)] = sum(values) / len(values)
    status = "PASS" if collector.sequence_count == expected_units else "PARTIAL"
    summary = {
        "schema_version": "perception-gain-foundation-diagnostic-panel-v1",
        "status": status,
        "reference_count": len(panel),
        "expected_prefix_count": expected_units,
        "completed_prefix_count": collector.sequence_count,
        "panel_sequences": [
            {
                "reference_id": master.reference_id,
                "sequence_id": master.sequence_id,
                "prefix_count": len(master.scan_indices),
            }
            for master in panel
        ],
        "incomplete_units": incomplete_units,
        "query_coverage": {
            "scene_count": len(query_summaries),
            "gt_instance_coverage_mean": (
                sum(float(row["gt_instance_coverage"]) for row in query_summaries)
                / len(query_summaries)
                if query_summaries
                else None
            ),
            "foreground_query_fraction_mean": (
                sum(float(row["foreground_query_fraction"]) for row in query_summaries)
                / len(query_summaries)
                if query_summaries
                else None
            ),
        },
        "attention_reachability": {
            "earliest_decoder_layer": earliest_layer,
            "earliest_allowed_gt_fraction_mean": (
                sum(earliest_attention) / len(earliest_attention)
                if earliest_attention
                else None
            ),
        },
        "best_candidate_iou_mean_by_decoder_layer": candidate_by_layer,
        "elapsed_seconds": elapsed,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "load_audit": load_audit,
        "foundation_source_sha256": _sha256(Path(__file__)),
    }
    _atomic_json(status_path, summary)
    del system
    torch.cuda.empty_cache()
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("bind", "corrected-e1", "live-smoke", "diagnostic-panel", "all"),
    )
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    results = {}
    if arguments.command in {"bind", "all"}:
        results["bind"] = run_bind(assets_path=arguments.assets)
    if arguments.command in {"corrected-e1", "all"}:
        results["corrected_e1"] = run_corrected_e1(assets_path=arguments.assets)
    if arguments.command in {"live-smoke", "all"}:
        results["live_smoke"] = run_live_smoke(
            assets_path=arguments.assets,
            external_root=arguments.external_root,
            device_name=arguments.device,
        )
    if arguments.command in {"diagnostic-panel", "all"}:
        results["diagnostic_panel"] = run_diagnostic_panel(
            assets_path=arguments.assets,
            device_name=arguments.device,
        )
    passed = all(
        isinstance(result, Mapping) and str(result.get("status", "")).upper() == "PASS"
        for result in results.values()
    )
    print(
        json.dumps(
            {
                "status": "PASS" if passed else "FAIL",
                "commands": sorted(results),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FoundationError",
    "best_candidate_iou_records",
    "build_r1_binding",
    "canonical_json_sha256",
    "compare_live_cached_supplements",
    "foundation_hash_order",
    "run_bind",
    "run_corrected_e1",
    "run_diagnostic_panel",
    "run_live_smoke",
    "select_cal_panel_masters",
]
