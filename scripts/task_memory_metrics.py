"""Replay compact TaskMemory caches through frozen output and metric code."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from datasets.task_memory_episode import StageMeta
from scripts.rescene_task_postprocess import OfficialTaskPrediction
from scripts.system_comparison_metrics import (
    CausalPrefixPair,
    validate_causal_prefix_pair,
)
from scripts.task_memory_cache import (
    identity_map_from_cache_record,
    prediction_from_cache_record,
    stage_meta_from_cache_record,
    target_from_cache_record,
    validate_episode_cache_payload,
)
from scripts.task_memory_output import (
    LagOnePublisher,
    PublicationAccounting,
    PublishedIdentity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TaskMemoryMetricError(ValueError):
    """Raised when cached predictions cannot support frozen task metrics."""


@dataclass(frozen=True)
class CachedPrefixResult:
    pair: CausalPrefixPair
    accounting: PublicationAccounting
    revision_versions: dict[str, int]
    score_reducer: str


def _publisher_view(
    prediction: OfficialTaskPrediction, meta: StageMeta
) -> tuple[OfficialTaskPrediction, StageMeta]:
    if len(meta.scan_ids_in_window) <= 2:
        return prediction, meta
    latest = prediction.latest_stage_index
    earliest = latest - 1
    selector = prediction.temporal_stages >= earliest
    masks = prediction.pred_masks[selector].contiguous()
    stages = (prediction.temporal_stages[selector] - earliest).long().contiguous()
    latest_local = 1
    local_prediction = OfficialTaskPrediction(
        pred_masks=masks,
        pred_scores=prediction.pred_scores,
        pred_classes=prediction.pred_classes,
        source_query_ids=prediction.source_query_ids,
        source_class_ids=prediction.source_class_ids,
        temporal_stages=stages,
        latest_stage_index=latest_local,
        latest_stage_masks=masks[stages == latest_local].contiguous(),
    )
    point_counts = [int(value.numel()) for value in meta.original_vertex_ids[-2:]]
    offsets = torch.tensor(
        [0, point_counts[0], sum(point_counts)], dtype=torch.long
    )
    local_stage_ids = torch.cat(
        [
            torch.full((count,), stage, dtype=torch.long)
            for stage, count in enumerate(point_counts)
        ]
    )
    point_count = int(local_stage_ids.numel())
    point2segment = torch.arange(point_count, dtype=torch.long)
    local_meta = StageMeta(
        reference_id=meta.reference_id,
        episode_id=meta.episode_id,
        scan_ids_in_window=meta.scan_ids_in_window[-2:],
        absolute_stage_index=meta.absolute_stage_index,
        local_stage_ids=local_stage_ids,
        original_vertex_ids=meta.original_vertex_ids[-2:],
        scan_vertex_offsets=offsets,
        point2segment=point2segment,
        segment_stage_ids=local_stage_ids.clone(),
        augmentation_transform_id=meta.augmentation_transform_id,
        coordinate_frame_id=meta.coordinate_frame_id,
        voxel_inverse=torch.arange(point_count, dtype=torch.long),
        full_resolution_point2segment=point2segment.clone(),
    )
    local_prediction.validate()
    return local_prediction, local_meta


def _prefix_target(
    stages: Sequence[Mapping[str, object]],
    *,
    horizon: int,
    class_mapper: Callable[[int], int],
) -> dict[str, torch.Tensor]:
    from scripts.evaluate_persist4d_p6a import build_temporal_target

    stage_payloads = [
        {
            "key": {"stage_index": index},
            "target": target_from_cache_record(stage),
        }
        for index, stage in enumerate(stages[:horizon])
    ]
    target = build_temporal_target(stage_payloads)
    target["labels"] = torch.tensor(
        [class_mapper(int(value)) for value in target["labels"].tolist()],
        dtype=torch.long,
    )
    return target


def _assert_class_preserving(
    keys: Sequence[PublishedIdentity], classes: torch.Tensor
) -> None:
    if classes.ndim != 1 or classes.numel() != len(keys):
        raise TaskMemoryMetricError("published identities and class columns differ")
    identities = set()
    for index, key in enumerate(keys):
        identity = (key.logical_id, key.generation, key.class_id)
        if identity in identities or int(classes[index].item()) != key.class_id:
            raise TaskMemoryMetricError(
                "published trajectory does not preserve its class identity"
            )
        identities.add(identity)


def replay_lag1_prefixes(
    payload: Mapping[str, object],
    *,
    reducers: Sequence[str] = ("mean",),
    class_mapper: Callable[[int], int],
) -> dict[str, tuple[CachedPrefixResult, ...]]:
    cache = validate_episode_cache_payload(payload)
    key = cache["key"]
    if key["output_policy"] != "lag1-v1":
        raise TaskMemoryMetricError("lag1 replay requires the frozen lag1-v1 policy")
    if (
        isinstance(reducers, (str, bytes))
        or not isinstance(reducers, Sequence)
        or not reducers
        or len(set(reducers)) != len(reducers)
    ):
        raise TaskMemoryMetricError("score reducers must be a unique sequence")
    if not callable(class_mapper):
        raise TaskMemoryMetricError("class_mapper must be callable")
    result = {}
    for reducer in reducers:
        publisher = LagOnePublisher(score_reducer=reducer, iou_threshold=0.5)
        records = []
        for stage_index, stage in enumerate(cache["stages"]):
            prediction, meta = _publisher_view(
                prediction_from_cache_record(stage),
                stage_meta_from_cache_record(stage),
            )
            prefix = publisher.update(
                prediction,
                identity_map_from_cache_record(stage),
                meta,
            )
            _assert_class_preserving(
                prefix.keys, prefix.prediction["pred_classes"]
            )
            horizon = stage_index + 1
            target = _prefix_target(
                cache["stages"], horizon=horizon, class_mapper=class_mapper
            )
            pair = validate_causal_prefix_pair(
                prediction=prefix.prediction,
                target=target,
                horizon=horizon,
                observed_scan_ids=key["history_scan_ids"][:horizon],
            )
            versions = {scan_id: 0 for scan_id in prefix.scan_ids}
            for revision in prefix.revision_log:
                versions[revision.scan_id] += 1
            records.append(
                CachedPrefixResult(
                    pair=pair,
                    accounting=prefix.accounting,
                    revision_versions=versions,
                    score_reducer=reducer,
                )
            )
        result[reducer] = tuple(records)
    return result


def _default_accumulator_factory() -> object:
    from scripts.analyze_persist4d_allt import AllTBaselineAccumulator
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec

    return AllTBaselineAccumulator(
        dataset_spec=resolve_metric_dataset_spec(PROJECT_ROOT)
    )


def compute_cached_task_metrics(
    payloads: Sequence[Mapping[str, object]],
    *,
    reducers: Sequence[str],
    class_mapper: Callable[[int], int],
    accumulator_factory: Callable[[], object] = _default_accumulator_factory,
) -> list[dict[str, object]]:
    if (
        isinstance(payloads, (str, bytes))
        or not isinstance(payloads, Sequence)
        or not payloads
    ):
        raise TaskMemoryMetricError("cached metrics require episode payloads")
    accumulators: dict[tuple[str, int], object] = {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for payload in payloads:
        replay = replay_lag1_prefixes(
            payload, reducers=reducers, class_mapper=class_mapper
        )
        for reducer, prefixes in replay.items():
            for prefix in prefixes:
                horizon = prefix.pair.horizon
                if horizon < 2:
                    continue
                identity = (reducer, horizon)
                if identity not in accumulators:
                    accumulators[identity] = accumulator_factory()
                accumulator = accumulators[identity]
                update = getattr(accumulator, "update", None)
                if not callable(update):
                    raise TaskMemoryMetricError("metric accumulator lacks update")
                update(prefix.pair)
                counts[identity] += 1
    rows = []
    for (reducer, horizon), accumulator in sorted(accumulators.items()):
        compute = getattr(accumulator, "compute", None)
        if not callable(compute):
            raise TaskMemoryMetricError("metric accumulator lacks compute")
        metrics = compute()
        if not isinstance(metrics, Mapping):
            raise TaskMemoryMetricError("metric accumulator output must be a mapping")
        rows.append(
            {
                "T": horizon,
                "episode_count": counts[(reducer, horizon)],
                "policy": "lag1",
                "reducer": reducer,
                **dict(metrics),
            }
        )
    return rows


def aggregate_identity_event_diagnostics(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TaskMemoryMetricError("identity event records must be a sequence")
    fields = {
        "births",
        "matched_births",
        "matched_reactivations",
        "reactivations",
        "rejected_births",
    }
    totals = {name: 0 for name in fields}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != fields:
            raise TaskMemoryMetricError("identity event fields differ")
        for name in fields:
            value = record[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TaskMemoryMetricError("identity event counts must be non-negative")
            totals[name] += value
        if (
            record["matched_births"] > record["births"]
            or record["matched_reactivations"] > record["reactivations"]
        ):
            raise TaskMemoryMetricError("matched identity events exceed predictions")
    false_births = totals["births"] - totals["matched_births"]
    false_reactivations = totals["reactivations"] - totals["matched_reactivations"]
    return {
        "birth_count": totals["births"],
        "false_birth_count": false_births,
        "false_birth_rate": (
            false_births / totals["births"] if totals["births"] else None
        ),
        "false_reactivation_count": false_reactivations,
        "false_reactivation_rate": (
            false_reactivations / totals["reactivations"]
            if totals["reactivations"]
            else None
        ),
        "reactivation_count": totals["reactivations"],
        "rejected_birth_count": totals["rejected_births"],
    }


__all__ = [
    "CachedPrefixResult",
    "TaskMemoryMetricError",
    "aggregate_identity_event_diagnostics",
    "compute_cached_task_metrics",
    "replay_lag1_prefixes",
]
