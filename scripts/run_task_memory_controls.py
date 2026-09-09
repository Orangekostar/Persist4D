#!/usr/bin/env python3
"""Produce prediction-only observation supplements and evaluate memory controls."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from datasets.task_memory_episode import (
    StageMeta,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeSpec,
    build_native_episode_masters,
)
from models.task_memory_routing import (
    PredictionObservation,
    commit_entities,
    route_entities,
)
from models.task_memory_state import TaskMemoryConfig, TaskMemoryState
from scripts.run_task_memory_policy_baseline import (
    CACHE_LIMIT_BYTES,
    CHECKPOINT_SHA256,
    EVALUATION_SEED,
    HORIZONS,
    PROJECT_ROOT,
    _atomic_json,
    _episode_specs,
    _file_sha256,
    _load_json,
    _meta_from_payload,
    _prediction_from_payload,
    _prediction_payload,
    _save_cache,
    _target_for_prefix,
    _validate_cache,
    _validate_collated_stage_identity,
    _write_csv,
)
from scripts.task_memory_contracts import canonical_json_sha256
from scripts.task_memory_output import CommitZeroPublisher, LagOnePublisher

CONTROL_METHODS = {"D-LAST": "last", "D-EMA": "fixed_ema"}
POLICIES = ("commit0", "lag1")
OBSERVATION_FIELDS = (
    "features",
    "class_prob",
    "confidence",
    "valid",
    "current_supported",
    "previous_supported",
)
CONTROL_FIELDS = (
    "population_id",
    "reference_count",
    "master_count",
    "order_count",
    "source_commit",
    "checkpoint_sha256",
    "base_cache_manifest_sha256",
    "observation_manifest_sha256",
    "config_sha256",
    "training_seed",
    "evaluation_seed",
    "method",
    "update_mode",
    "update_rate",
    "association_threshold",
    "class_weight",
    "policy",
    "reducer",
    "window",
    "K",
    "r",
    "state_bytes",
    "T",
    "t_mAP",
    "t_mAP50",
    "t_mAP25",
    "t_REC",
    "prefix_overall_mAP",
    "direct_current_AP",
    "shared_r1_forward_count",
    "training_updates",
    "attribution",
    "status",
)
ROUTER_FIELDS = (
    "population_id",
    "source_commit",
    "checkpoint_sha256",
    "base_cache_manifest_sha256",
    "observation_manifest_sha256",
    "method",
    "episode_count",
    "stage_count",
    "valid_queries",
    "current_supported_queries",
    "previous_supported_queries",
    "previous_only_queries",
    "inherited_routes",
    "route_coverage",
    "mean_route_score",
    "dormant_routes",
    "births",
    "rejected_births",
    "peak_occupied_slots",
    "peak_state_bytes",
    "status",
)


class ControlRunnerError(RuntimeError):
    """Raised when a control cache or evaluation violates its contract."""


@dataclass(frozen=True)
class ControlDiagnostics:
    stages: int = 0
    valid_queries: int = 0
    current_supported_queries: int = 0
    previous_supported_queries: int = 0
    previous_only_queries: int = 0
    inherited_routes: int = 0
    dormant_routes: int = 0
    births: int = 0
    rejected_births: int = 0
    route_score_sum: float = 0.0
    route_score_count: int = 0
    peak_occupied_slots: int = 0
    peak_state_bytes: int = 0

    def __add__(self, other: ControlDiagnostics) -> ControlDiagnostics:
        return ControlDiagnostics(
            stages=self.stages + other.stages,
            valid_queries=self.valid_queries + other.valid_queries,
            current_supported_queries=(
                self.current_supported_queries + other.current_supported_queries
            ),
            previous_supported_queries=(
                self.previous_supported_queries + other.previous_supported_queries
            ),
            previous_only_queries=(
                self.previous_only_queries + other.previous_only_queries
            ),
            inherited_routes=self.inherited_routes + other.inherited_routes,
            dormant_routes=self.dormant_routes + other.dormant_routes,
            births=self.births + other.births,
            rejected_births=self.rejected_births + other.rejected_births,
            route_score_sum=self.route_score_sum + other.route_score_sum,
            route_score_count=self.route_score_count + other.route_score_count,
            peak_occupied_slots=max(
                self.peak_occupied_slots, other.peak_occupied_slots
            ),
            peak_state_bytes=max(self.peak_state_bytes, other.peak_state_bytes),
        )


@dataclass(frozen=True)
class ControlTrajectory:
    identity_maps: tuple[dict[int, tuple[int, int]], ...]
    final_state: TaskMemoryState
    diagnostics: ControlDiagnostics


def observation_payload(observation: PredictionObservation) -> dict[str, Tensor]:
    observation.validate()
    return {
        name: getattr(observation, name).detach().cpu().clone()
        for name in OBSERVATION_FIELDS
    }


def prediction_observation_from_payload(
    value: Mapping[str, object],
) -> PredictionObservation:
    if not isinstance(value, Mapping) or set(value) != set(OBSERVATION_FIELDS):
        raise ControlRunnerError("observation payload must be prediction-only")
    if any(not isinstance(value[name], Tensor) for name in OBSERVATION_FIELDS):
        raise ControlRunnerError("observation payload fields must be tensors")
    observation = PredictionObservation(
        **{name: value[name].detach().clone() for name in OBSERVATION_FIELDS}
    )
    try:
        observation.validate()
    except ValueError as error:
        raise ControlRunnerError("observation payload is invalid") from error
    return observation


def validate_observation_supplement(
    value: object,
    *,
    expected_provenance: Mapping[str, object],
    expected_base_cache: Mapping[str, object],
    expected_stage_count: int,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ControlRunnerError("observation supplement must be a mapping")
    result = dict(value)
    stages = result.get("stages")
    if (
        result.get("schema_version") != "task-memory-control-observations-v2"
        or result.get("provenance") != dict(expected_provenance)
        or result.get("base_cache") != dict(expected_base_cache)
    ):
        raise ControlRunnerError("observation supplement base cache identity differs")
    if (
        isinstance(expected_stage_count, bool)
        or not isinstance(expected_stage_count, int)
        or expected_stage_count <= 0
        or not isinstance(stages, list)
        or len(stages) != expected_stage_count
    ):
        raise ControlRunnerError("observation supplement stage lineage differs")
    overlap_fields = {
        "exact",
        "generated_candidate_count",
        "base_candidate_count",
        "common_candidate_count",
        "score_max_abs",
        "aligned_mask_iou_mean",
    }
    for stage in stages:
        if not isinstance(stage, Mapping) or set(stage) != {
            "observation",
            "prediction",
            "base_overlap",
        }:
            raise ControlRunnerError("supplement stages must be prediction-only")
        prediction_observation_from_payload(stage["observation"])
        _prediction_from_payload(stage["prediction"])
        overlap = stage["base_overlap"]
        if not isinstance(overlap, Mapping) or set(overlap) != overlap_fields:
            raise ControlRunnerError("supplement base overlap is invalid")
        if not isinstance(overlap["exact"], bool):
            raise ControlRunnerError("supplement base overlap exact flag is invalid")
        counts = tuple(
            overlap[name]
            for name in (
                "generated_candidate_count",
                "base_candidate_count",
                "common_candidate_count",
            )
        )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ) or counts[2] > min(counts[:2]):
            raise ControlRunnerError("supplement base overlap counts are invalid")
        for name in ("score_max_abs", "aligned_mask_iou_mean"):
            value = overlap[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                or (name.endswith("iou_mean") and float(value) > 1)
            ):
                raise ControlRunnerError("supplement base overlap metrics are invalid")
    return result


def run_control_trajectory(
    observations: Sequence[PredictionObservation],
    stage_meta: Sequence[StageMeta],
    *,
    update_mode: str,
    capacity: int = 100,
    class_weight: float = 0.25,
    association_threshold: float = 0.5,
    update_rate: float = 0.2,
) -> ControlTrajectory:
    if not observations or len(observations) != len(stage_meta):
        raise ControlRunnerError("control trajectory stages must be non-empty and aligned")
    if any(not isinstance(value, PredictionObservation) for value in observations):
        raise ControlRunnerError("control trajectory observations are invalid")
    first = observations[0]
    first.validate()
    config = TaskMemoryConfig(
        class_weight=class_weight,
        association_threshold=association_threshold,
        update_mode=update_mode,
        update_rate=update_rate,
        max_update_rate=update_rate,
    )
    state = TaskMemoryState.empty(
        batch_size=first.batch_size,
        capacity=capacity,
        feature_dim=first.features.shape[2],
        class_count=first.class_prob.shape[2],
        device=first.features.device,
        dtype=first.features.dtype,
        config=config,
    )
    identity_maps = []
    diagnostics = ControlDiagnostics()
    for observation, meta in zip(observations, stage_meta, strict=True):
        route = route_entities(observation, state, [meta])
        commit = commit_entities(observation, route, state, [meta])
        matched = route.query_to_slot >= 0
        current_valid = observation.valid & observation.current_supported
        matched_scores = route.route_score[matched]
        stage_diagnostics = ControlDiagnostics(
            stages=1,
            valid_queries=int(observation.valid.sum().item()),
            current_supported_queries=int(observation.current_supported.sum().item()),
            previous_supported_queries=int(observation.previous_supported.sum().item()),
            previous_only_queries=int(
                (
                    observation.previous_supported
                    & ~observation.current_supported
                ).sum().item()
            ),
            inherited_routes=int(matched.sum().item()),
            dormant_routes=int((matched & ~current_valid).sum().item()),
            births=int(commit.births.sum().item()),
            rejected_births=int(commit.rejected_births.sum().item()),
            route_score_sum=float(matched_scores.sum().item()),
            route_score_count=int(matched_scores.numel()),
            peak_occupied_slots=int(commit.state.occupied.sum(dim=1).max().item()),
            peak_state_bytes=commit.state.state_bytes,
        )
        diagnostics = diagnostics + stage_diagnostics
        identity_maps.append(commit.identity_map())
        state = commit.state.detach()
    return ControlTrajectory(tuple(identity_maps), state, diagnostics)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise ControlRunnerError("Git HEAD is invalid")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        if subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=False).returncode:
            raise ControlRunnerError("tracked source tree must be clean")
    return result


def _load_base_manifest(path: Path) -> dict[str, object]:
    value = _load_json(path)
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256", None)
    if (
        declared != canonical_json_sha256(unsigned)
        or value.get("status") != "PASS"
        or value.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or value.get("entry_count") != 47
        or not isinstance(value.get("records"), list)
    ):
        raise ControlRunnerError("base cache manifest is invalid")
    return value


def _records_by_sequence(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    records = manifest["records"]
    result = {}
    for value in records:
        if not isinstance(value, Mapping):
            raise ControlRunnerError("base cache manifest record is invalid")
        record = dict(value)
        sequence_id = record.get("sequence_id")
        if not isinstance(sequence_id, str) or sequence_id in result:
            raise ControlRunnerError("base cache sequence records are not unique")
        result[sequence_id] = record
    return result


def _base_cache_link(record: Mapping[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in ("filename", "sha256", "sequence_id", "reference_id")
    }


def _load_base_episode(
    *,
    cache_root: Path,
    record: Mapping[str, object],
    spec: TaskMemoryEpisodeSpec,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    path = cache_root / str(record["filename"])
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not digest_path.is_file() or path.is_symlink():
        raise ControlRunnerError("base cache file is unavailable")
    digest = _file_sha256(path)
    fields = digest_path.read_text(encoding="ascii").strip().split()
    if digest != record["sha256"] or fields != [digest, path.name]:
        raise ControlRunnerError("base cache digest differs")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ControlRunnerError("base cache cannot be loaded safely") from error
    provenance = raw.get("provenance") if isinstance(raw, Mapping) else None
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("source_commit") != manifest["source_commit"]
        or provenance.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
        or provenance.get("config_sha256") != manifest["config_sha256"]
    ):
        raise ControlRunnerError("base cache provenance differs from manifest")
    return _validate_cache(raw, provenance=provenance, spec=spec)


def _prediction_overlap(generated: object, base: object) -> dict[str, object]:
    generated_keys = list(
        zip(
            generated.source_query_ids.detach().cpu().tolist(),
            generated.source_class_ids.detach().cpu().tolist(),
            strict=True,
        )
    )
    base_keys = list(
        zip(
            base.source_query_ids.detach().cpu().tolist(),
            base.source_class_ids.detach().cpu().tolist(),
            strict=True,
        )
    )
    generated_by_key = {key: index for index, key in enumerate(generated_keys)}
    base_by_key = {key: index for index, key in enumerate(base_keys)}
    common = sorted(set(generated_by_key) & set(base_by_key))
    score_differences = []
    mask_ious = []
    for key in common:
        generated_index = generated_by_key[key]
        base_index = base_by_key[key]
        score_differences.append(
            abs(
                float(generated.pred_scores[generated_index].item())
                - float(base.pred_scores[base_index].item())
            )
        )
        generated_mask = generated.pred_masks[:, generated_index].detach().cpu()
        base_mask = base.pred_masks[:, base_index].detach().cpu()
        if generated_mask.shape != base_mask.shape:
            raise ControlRunnerError("base and replay mask point lineage differs")
        union = int((generated_mask | base_mask).sum().item())
        intersection = int((generated_mask & base_mask).sum().item())
        mask_ious.append(intersection / union if union else 1.0)
    fields = (
        "pred_masks",
        "pred_scores",
        "pred_classes",
        "source_query_ids",
        "source_class_ids",
        "temporal_stages",
        "latest_stage_index",
        "latest_stage_masks",
    )
    exact = all(
        torch.equal(getattr(generated, name).cpu(), getattr(base, name).cpu())
        if isinstance(getattr(generated, name), Tensor)
        else getattr(generated, name) == getattr(base, name)
        for name in fields
    )
    return {
        "exact": exact,
        "generated_candidate_count": len(generated_keys),
        "base_candidate_count": len(base_keys),
        "common_candidate_count": len(common),
        "score_max_abs": max(score_differences, default=0.0),
        "aligned_mask_iou_mean": (
            sum(mask_ious) / len(mask_ious) if mask_ious else 0.0
        ),
    }


def _window_observation(
    *,
    local_observation: object,
    output: Mapping[str, object],
    segment_stages: Tensor,
    latest_stage: int,
    confidence_threshold: float,
    mask_threshold: float,
    minimum_mask_support: int,
) -> PredictionObservation:
    masks = output.get("pred_masks")
    if not isinstance(masks, list) or len(masks) != 1 or not isinstance(masks[0], Tensor):
        raise ControlRunnerError("R1 output masks are unavailable")
    logits = masks[0]
    if logits.ndim != 2 or logits.shape[0] != segment_stages.numel():
        raise ControlRunnerError("R1 masks and segment stages differ")
    current = (
        (logits[segment_stages == latest_stage].sigmoid() >= mask_threshold)
        .sum(dim=0)
        .ge(minimum_mask_support)
    )
    previous_selector = segment_stages != latest_stage
    if previous_selector.any().item():
        previous = (
            (logits[previous_selector].sigmoid() >= mask_threshold)
            .sum(dim=0)
            .ge(minimum_mask_support)
        )
    else:
        previous = torch.zeros_like(current)
    valid = (local_observation.confidence >= confidence_threshold) & (
        current | previous
    ).unsqueeze(0)
    observation = PredictionObservation(
        features=local_observation.features.detach().clone(),
        class_prob=local_observation.class_prob.detach().clone(),
        confidence=local_observation.confidence.detach().clone(),
        valid=valid,
        current_supported=current.unsqueeze(0),
        previous_supported=previous.unsqueeze(0),
    )
    observation.validate()
    return observation


def _produce_supplement(
    *,
    episode: object,
    collator: TaskMemoryEpisodeCollator,
    system: object,
    observation_settings: Mapping[str, object],
    class_mapper: Callable[[int], int],
    device: torch.device,
    provenance: Mapping[str, object],
    base_cache: Mapping[str, object],
    base_link: Mapping[str, object],
) -> dict[str, object]:
    from models.persistent_memory import build_local_observation
    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
        _segment_stages,
    )
    from scripts.rescene_task_postprocess import extract_official_task_prediction

    batch = collator([episode])
    if len(batch.stage_batches) != len(base_cache["stages"]):
        raise ControlRunnerError("base and replay episode stage counts differ")
    stages = []
    for stage_batch, base_stage in zip(
        batch.stage_batches, base_cache["stages"], strict=True
    ):
        data, targets, names = stage_batch.model_batch
        meta = stage_batch.stage_meta[0]
        _validate_collated_stage_identity(
            names=names, scan_ids_in_window=meta.scan_ids_in_window
        )
        full_targets = getattr(data, "target_full", None)
        if not isinstance(full_targets, Sequence) or len(full_targets) != 1:
            raise ControlRunnerError("collated stage lacks full target")
        data = _move_data_to_device(data, device)
        targets = _move_targets_to_device(targets, device)
        target = targets[0]
        segment_stages = _segment_stages(target)
        latest_stage = int(segment_stages.max().item())
        raw_coordinates = system._process_raw_coordinates(data)
        with torch.inference_mode():
            output = system(
                data,
                point2segment=[target["point2segment"]],
                raw_coordinates=raw_coordinates,
                is_eval=True,
            )
        local = build_local_observation(
            output,
            [segment_stages],
            latest_stage=latest_stage,
            **dict(observation_settings),
        )
        window = _window_observation(
            local_observation=local,
            output=output,
            segment_stages=segment_stages,
            latest_stage=latest_stage,
            confidence_threshold=float(
                observation_settings["confidence_threshold"]
            ),
            mask_threshold=float(observation_settings["mask_threshold"]),
            minimum_mask_support=int(
                observation_settings["minimum_mask_support"]
            ),
        )
        generated = extract_official_task_prediction(
            system=system,
            output=output,
            target_low_resolution=target,
            target_full_resolution=full_targets[0],
            data=data,
            class_mapper=class_mapper,
            latest_stage_index=latest_stage,
        )
        base_prediction = _prediction_from_payload(base_stage["prediction"])
        stages.append(
            {
                "observation": observation_payload(window),
                "prediction": _prediction_payload(generated),
                "base_overlap": _prediction_overlap(generated, base_prediction),
            }
        )
        del data, targets, output, local, window, generated
    supplement = {
        "schema_version": "task-memory-control-observations-v2",
        "provenance": dict(provenance),
        "base_cache": dict(base_link),
        "stages": stages,
    }
    return validate_observation_supplement(
        supplement,
        expected_provenance=provenance,
        expected_base_cache=base_link,
        expected_stage_count=len(stages),
    )


def _load_supplement(
    *,
    path: Path,
    provenance: Mapping[str, object],
    base_link: Mapping[str, object],
    expected_stage_count: int,
) -> tuple[dict[str, object], str]:
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not digest_path.is_file() or path.is_symlink():
        raise ControlRunnerError("observation supplement is unavailable")
    fields = digest_path.read_text(encoding="ascii").strip().split()
    digest = _file_sha256(path)
    if fields != [digest, path.name]:
        raise ControlRunnerError("observation supplement digest differs")
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ControlRunnerError("observation supplement cannot be loaded safely") from error
    return (
        validate_observation_supplement(
            raw,
            expected_provenance=provenance,
            expected_base_cache=base_link,
            expected_stage_count=expected_stage_count,
        ),
        digest,
    )


def _metric_values(accumulator: object) -> dict[str, float]:
    values = accumulator.compute()
    return {
        "t_mAP": values["t_mAP"],
        "t_mAP50": values["t_mAP50"],
        "t_mAP25": values["t_mAP25"],
        "t_REC": values["t_REC"],
        "prefix_overall_mAP": values["prefix_overall_mAP"],
        "direct_current_AP": values["local_current_AP"],
    }


def _analyze(
    *,
    specs: Sequence[TaskMemoryEpisodeSpec],
    records: Mapping[str, Mapping[str, object]],
    base_manifest: Mapping[str, object],
    base_cache_root: Path,
    supplement_root: Path,
    provenance: Mapping[str, object],
    class_mapper: Callable[[int], int],
    output_root: Path,
) -> dict[str, object]:
    from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec
    from scripts.system_comparison_metrics import validate_causal_prefix_pair

    metric_spec = resolve_metric_dataset_spec(PROJECT_ROOT)
    accumulators = {
        (method, policy, horizon): AllTBaselineAccumulator(dataset_spec=metric_spec)
        for method in CONTROL_METHODS
        for policy in POLICIES
        for horizon in HORIZONS
    }
    diagnostic_totals = {method: ControlDiagnostics() for method in CONTROL_METHODS}
    state_bytes = defaultdict(int)
    supplement_records = []
    for index, spec in enumerate(specs):
        record = records[spec.source_sequence_id]
        base_link = _base_cache_link(record)
        base = _load_base_episode(
            cache_root=base_cache_root,
            record=record,
            spec=spec,
            manifest=base_manifest,
        )
        supplement_path = supplement_root / str(record["filename"])
        supplement, digest = _load_supplement(
            path=supplement_path,
            provenance=provenance,
            base_link=base_link,
            expected_stage_count=len(base["stages"]),
        )
        digest_path = supplement_path.with_suffix(supplement_path.suffix + ".sha256")
        supplement_records.append(
            {
                "filename": supplement_path.name,
                "sha256": digest,
                "bytes": supplement_path.stat().st_size + digest_path.stat().st_size,
                "base_cache_sha256": record["sha256"],
                "sequence_id": spec.source_sequence_id,
                "reference_id": spec.reference_id,
            }
        )
        observations = [
            prediction_observation_from_payload(value["observation"])
            for value in supplement["stages"]
        ]
        metas = [
            _meta_from_payload(stage["stage_meta"]) for stage in base["stages"]
        ]
        targets = [stage["target"] for stage in base["stages"]]
        for method, update_mode in CONTROL_METHODS.items():
            trajectory = run_control_trajectory(
                observations,
                metas,
                update_mode=update_mode,
                capacity=100,
                class_weight=0.25,
                association_threshold=0.5,
                update_rate=0.2,
            )
            diagnostic_totals[method] = (
                diagnostic_totals[method] + trajectory.diagnostics
            )
            state_bytes[method] = max(
                state_bytes[method], trajectory.diagnostics.peak_state_bytes
            )
            publishers = {
                "commit0": CommitZeroPublisher(score_reducer="mean"),
                "lag1": LagOnePublisher(score_reducer="mean", iou_threshold=0.5),
            }
            for stage_index, (stage, meta, identity_map) in enumerate(
                zip(supplement["stages"], metas, trajectory.identity_maps, strict=True)
            ):
                prediction = _prediction_from_payload(stage["prediction"])
                for policy, publisher in publishers.items():
                    prefix = publisher.update(prediction, identity_map, meta)
                    horizon = stage_index + 1
                    if horizon not in HORIZONS:
                        continue
                    target = _target_for_prefix(
                        targets, horizon=horizon, class_mapper=class_mapper
                    )
                    pair = validate_causal_prefix_pair(
                        prediction=prefix.prediction,
                        target=target,
                        horizon=horizon,
                        observed_scan_ids=spec.scan_ids[:horizon],
                    )
                    accumulators[(method, policy, horizon)].update(pair)
        print(
            json.dumps(
                {
                    "analysis": index + 1,
                    "total": len(specs),
                    "sequence_id": spec.source_sequence_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    supplement_bytes = sum(int(record["bytes"]) for record in supplement_records)
    combined_bytes = int(base_manifest["cache_bytes"]) + supplement_bytes
    observation_manifest: dict[str, object] = {
        "schema_version": "task-memory-control-observation-manifest-v1",
        "source_commit": provenance["source_commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": provenance["config_sha256"],
        "base_cache_manifest_sha256": base_manifest["content_sha256"],
        "entry_count": len(supplement_records),
        "stage_count": len(supplement_records) * 5,
        "supplement_bytes": supplement_bytes,
        "combined_evaluation_cache_bytes": combined_bytes,
        "cache_limit_bytes": CACHE_LIMIT_BYTES,
        "records": supplement_records,
        "status": "PASS" if combined_bytes <= CACHE_LIMIT_BYTES else "FAIL",
    }
    observation_manifest["content_sha256"] = canonical_json_sha256(
        observation_manifest
    )
    if combined_bytes > CACHE_LIMIT_BYTES:
        raise ControlRunnerError("combined evaluation cache exceeds the 40 GiB cap")

    control_rows = []
    for method, update_mode in CONTROL_METHODS.items():
        for policy in POLICIES:
            for horizon in HORIZONS:
                control_rows.append(
                    {
                        "population_id": "development_train_holdout_47_masters_canonical",
                        "reference_count": len({spec.reference_id for spec in specs}),
                        "master_count": len(specs),
                        "order_count": len(specs),
                        "source_commit": provenance["source_commit"],
                        "checkpoint_sha256": CHECKPOINT_SHA256,
                        "base_cache_manifest_sha256": base_manifest["content_sha256"],
                        "observation_manifest_sha256": observation_manifest[
                            "content_sha256"
                        ],
                        "config_sha256": provenance["config_sha256"],
                        "training_seed": EVALUATION_SEED,
                        "evaluation_seed": EVALUATION_SEED,
                        "method": method,
                        "update_mode": update_mode,
                        "update_rate": 0.2 if update_mode == "fixed_ema" else 1.0,
                        "association_threshold": 0.5,
                        "class_weight": 0.25,
                        "policy": policy,
                        "reducer": "mean",
                        "window": "W2",
                        "K": 100,
                        "r": 0,
                        "state_bytes": state_bytes[method],
                        "T": horizon,
                        **_metric_values(accumulators[(method, policy, horizon)]),
                        "shared_r1_forward_count": len(specs) * 5,
                        "training_updates": 0,
                        "attribution": "long_memory_update_control_plus_output_policy",
                        "status": "MEASURED",
                    }
                )

    router_rows = []
    for method in CONTROL_METHODS:
        diagnostic = diagnostic_totals[method]
        coverage = (
            diagnostic.inherited_routes / diagnostic.valid_queries
            if diagnostic.valid_queries
            else 0.0
        )
        mean_score = (
            diagnostic.route_score_sum / diagnostic.route_score_count
            if diagnostic.route_score_count
            else ""
        )
        router_rows.append(
            {
                "population_id": "development_train_holdout_47_masters_canonical",
                "source_commit": provenance["source_commit"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "base_cache_manifest_sha256": base_manifest["content_sha256"],
                "observation_manifest_sha256": observation_manifest["content_sha256"],
                "method": method,
                "episode_count": len(specs),
                "stage_count": diagnostic.stages,
                "valid_queries": diagnostic.valid_queries,
                "current_supported_queries": diagnostic.current_supported_queries,
                "previous_supported_queries": diagnostic.previous_supported_queries,
                "previous_only_queries": diagnostic.previous_only_queries,
                "inherited_routes": diagnostic.inherited_routes,
                "route_coverage": coverage,
                "mean_route_score": mean_score,
                "dormant_routes": diagnostic.dormant_routes,
                "births": diagnostic.births,
                "rejected_births": diagnostic.rejected_births,
                "peak_occupied_slots": diagnostic.peak_occupied_slots,
                "peak_state_bytes": diagnostic.peak_state_bytes,
                "status": "MEASURED",
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        output_root / "baseline/control_observation_manifest.json",
        observation_manifest,
    )
    _write_csv(
        output_root / "baseline/long_memory_controls.csv",
        control_rows,
        CONTROL_FIELDS,
    )
    _write_csv(
        output_root / "implementation/router_diagnostics.csv",
        router_rows,
        ROUTER_FIELDS,
    )
    return observation_manifest


def run_controls(
    *,
    mode: str,
    device_name: str,
    shard_index: int,
    shard_count: int,
    data_root: Path,
    metadata_path: Path,
    checkpoint_path: Path,
    pretrained_path: Path,
    run_root: Path,
    output_root: Path,
    base_manifest_path: Path,
) -> dict[str, object]:
    import hydra

    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        build_rio_class_mapper,
    )
    from scripts.preflight_task_memory_episode import (
        _rio_base_dataset,
        _role_by_reference,
        load_reference_by_scene,
    )
    from scripts.r1_downstream_context import (
        _load_r1_system,
        compose_r1_runtime_config,
        load_r1_contract,
        validate_external_file_identity,
    )
    from scripts.system_comparison_inference import deterministic_inference_runtime

    if mode not in {"produce", "analyze", "all"}:
        raise ControlRunnerError("mode must be produce, analyze, or all")
    if (
        isinstance(shard_index, bool)
        or isinstance(shard_count, bool)
        or not isinstance(shard_index, int)
        or not isinstance(shard_count, int)
        or shard_count <= 0
        or not 0 <= shard_index < shard_count
    ):
        raise ControlRunnerError("invalid shard selection")
    source_commit = _git_head()
    base_manifest = _load_base_manifest(base_manifest_path.resolve(strict=True))
    records = _records_by_sequence(base_manifest)
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    pretrained_path = pretrained_path.expanduser().resolve(strict=True)
    validate_external_file_identity(
        checkpoint_path,
        expected_sha256=CHECKPOINT_SHA256,
        expected_bytes=754813672,
        require_digest_filename=True,
        label="R1 checkpoint",
    )
    validate_external_file_identity(
        pretrained_path,
        expected_sha256="845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07",
        expected_bytes=433987358,
        require_digest_filename=False,
        label="Concerto pretrain",
    )
    runtime_config, memory_config = compose_r1_runtime_config(pretrained_path)
    data_root = data_root.expanduser().resolve(strict=True)
    metadata_path = metadata_path.expanduser().resolve(strict=True)
    base_dataset = _rio_base_dataset(runtime_config, data_root=data_root, horizon=5)
    data_contract = _load_json(
        PROJECT_ROOT / "artifacts/task_memory_retention_v2/DATA_CONTRACT.json"
    )
    masters = tuple(
        master
        for master in build_native_episode_masters(
            base_dataset,
            reference_by_scene=load_reference_by_scene(metadata_path),
            role_by_reference=_role_by_reference(data_contract),
        )
        if master.role == "development" and len(master.scan_ids) == 5
    )
    specs = _episode_specs(masters)
    if len(specs) != 47 or set(records) != {
        spec.source_sequence_id for spec in specs
    }:
        raise ControlRunnerError("base cache and development population differ")
    config_document = {
        "schema_version": "task-memory-controls-config-v2",
        "source_commit": source_commit,
        "base_cache_manifest_sha256": base_manifest["content_sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "methods": {
            "D-LAST": {"update_mode": "last"},
            "D-EMA": {"update_mode": "fixed_ema", "update_rate": 0.2},
        },
        "K": 100,
        "class_weight": 0.25,
        "association_threshold": 0.5,
        "policies": list(POLICIES),
        "reducer": "mean",
        "window": "W2",
        "evaluation_seed": EVALUATION_SEED,
    }
    provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "base_cache_manifest_sha256": base_manifest["content_sha256"],
        "config_sha256": canonical_json_sha256(config_document),
    }
    base_cache_root = (
        run_root.expanduser().resolve()
        / "baseline/cache"
        / str(base_manifest["config_sha256"])[:16]
    )
    supplement_root = (
        run_root.expanduser().resolve()
        / "baseline/control_observations"
        / str(provenance["config_sha256"])[:16]
    )
    class_mapper = build_rio_class_mapper(base_dataset)

    produced = 0
    reused = 0
    if mode in {"produce", "all"}:
        if not torch.cuda.is_available():
            raise ControlRunnerError("CUDA is required for observation production")
        device = torch.device(device_name)
        if (
            device.type != "cuda"
            or device.index is None
            or device.index >= torch.cuda.device_count()
        ):
            raise ControlRunnerError("requested CUDA device is unavailable")
        contract = load_r1_contract(
            PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml"
        )
        system = _load_r1_system(runtime_config, checkpoint_path, device, contract)
        collator = TaskMemoryEpisodeCollator(
            hydra.utils.instantiate(runtime_config.data.validation_collation)
        )
        observation_settings = {
            "background_class": int(memory_config.background_class),
            "confidence_threshold": float(memory_config.confidence_threshold),
            "mask_threshold": float(memory_config.mask_threshold),
            "minimum_mask_support": int(memory_config.minimum_mask_support),
        }
        selected = [
            spec for index, spec in enumerate(specs) if index % shard_count == shard_index
        ]
        with deterministic_inference_runtime(EVALUATION_SEED, device):
            for index, spec in enumerate(selected):
                record = records[spec.source_sequence_id]
                base_link = _base_cache_link(record)
                path = supplement_root / str(record["filename"])
                if path.is_file() and path.with_suffix(path.suffix + ".sha256").is_file():
                    _load_supplement(
                        path=path,
                        provenance=provenance,
                        base_link=base_link,
                        expected_stage_count=5,
                    )
                    reused += 1
                else:
                    base = _load_base_episode(
                        cache_root=base_cache_root,
                        record=record,
                        spec=spec,
                        manifest=base_manifest,
                    )
                    dataset = TaskMemoryEpisodeDataset(
                        base_dataset, (spec,), apply_augmentation=False
                    )
                    with _frozen_inference_seed(EVALUATION_SEED, device):
                        supplement = _produce_supplement(
                            episode=dataset[0],
                            collator=collator,
                            system=system,
                            observation_settings=observation_settings,
                            class_mapper=class_mapper,
                            device=device,
                            provenance=provenance,
                            base_cache=base,
                            base_link=base_link,
                        )
                    _save_cache(path, supplement)
                    produced += 1
                print(
                    json.dumps(
                        {
                            "produced": produced,
                            "reused": reused,
                            "sequence_id": spec.source_sequence_id,
                            "shard": f"{shard_index}/{shard_count}",
                            "shard_progress": f"{index + 1}/{len(selected)}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        del system
        torch.cuda.empty_cache()

    result: dict[str, object] = {
        "mode": mode,
        "produced": produced,
        "reused": reused,
        "source_commit": source_commit,
        "config_sha256": provenance["config_sha256"],
        "supplement_root": str(supplement_root),
        "status": "PRODUCED",
    }
    if mode in {"analyze", "all"}:
        if mode == "all" and shard_count != 1:
            raise ControlRunnerError("all mode requires a single shard")
        result = _analyze(
            specs=specs,
            records=records,
            base_manifest=base_manifest,
            base_cache_root=base_cache_root,
            supplement_root=supplement_root,
            provenance=provenance,
            class_mapper=class_mapper,
            output_root=output_root.expanduser().resolve(),
        )
    return result


def _parser() -> argparse.ArgumentParser:
    resolver = _load_json(
        PROJECT_ROOT / "artifacts/task_memory_retention_v2/external_assets.local.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("produce", "analyze", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=resolver["external:data_root"])
    parser.add_argument(
        "--rio-metadata", type=Path, default=resolver["external:rio_metadata"]
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=resolver["external:r1_checkpoint"]
    )
    parser.add_argument(
        "--pretrained", type=Path, default=resolver["external:concerto_pretrained"]
    )
    parser.add_argument("--run-root", type=Path, default=resolver["external:run_root"])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/task_memory_retention_v2",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts/task_memory_retention_v2/baseline/cache_manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    started = time.time()
    result = run_controls(
        mode=args.mode,
        device_name=args.device,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        data_root=args.data_root,
        metadata_path=args.rio_metadata,
        checkpoint_path=args.checkpoint,
        pretrained_path=args.pretrained,
        run_root=args.run_root,
        output_root=args.output_root,
        base_manifest_path=args.base_manifest,
    )
    print(
        json.dumps(
            {
                "config_sha256": result.get("config_sha256"),
                "elapsed_seconds": time.time() - started,
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ControlDiagnostics",
    "ControlRunnerError",
    "ControlTrajectory",
    "observation_payload",
    "prediction_observation_from_payload",
    "run_control_trajectory",
    "run_controls",
    "validate_observation_supplement",
]
