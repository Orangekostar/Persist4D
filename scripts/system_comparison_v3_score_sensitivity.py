#!/usr/bin/env python3
"""Evaluate V3 trajectory score reducers and the direct local-current channel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_persist4d_p6a import (
    build_rio_class_mapper,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
)
from scripts.p6a_metrics import OfficialMetricAccumulator
from scripts.run_system_comparison import (
    REPRODUCIBILITY_BINDING,
    _build_frozen_setup,
)
from scripts.system_comparison_metrics import (
    CausalTaskAccumulator,
    compute_causal_task_metrics,
)
from scripts.system_comparison_v2_analysis import (
    build_v2_causal_pair,
    load_v2_sequences,
)
from scripts.system_comparison_v2_cache import (
    task_sidecar_digest,
    validate_task_sidecar,
)
from scripts.system_comparison_v2_inference import (
    OfficialCandidateTrajectoryAccumulator,
    V2TrajectorySnapshot,
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/score_sensitivity"
CACHE_ROOT = Path(
    "/mnt/shared/ww/persist4d-tmap-root-cause-v2/system_comparison_v2_full"
)
CACHE_MANIFEST = PROJECT_ROOT / "artifacts/system_comparison_v2/cache_manifest.json"
METADATA = Path("/home/ww/3RScan.json")
V2_ROOT = PROJECT_ROOT / "artifacts/system_comparison_v2"
TRACKERS = ("B2", "B3", "B4")
REDUCERS = ("mean", "latest", "max")
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)
CAUSAL_FIELDS = (
    "causal_prefix_t_mAP",
    "causal_prefix_t_mAP50",
    "causal_prefix_t_mAP25",
    "causal_prefix_t_REC",
    "causal_prefix_t_REC50",
    "causal_prefix_t_REC25",
)
TRAJECTORY_SLICE_FIELDS = (
    "trajectory_current_slice_AP",
    "trajectory_current_slice_AP50",
    "trajectory_current_slice_AP25",
    "trajectory_current_slice_REC",
)
LOCAL_FIELDS = (
    "local_current_AP",
    "local_current_AP50",
    "local_current_AP25",
    "local_current_REC",
)


class ScoreSensitivityError(ValueError):
    """Raised when EV0 cache, invariance, or regression contracts fail."""


@dataclass(frozen=True)
class LocalCurrentPair:
    prediction: dict[str, Tensor]
    target: dict[str, Tensor]


def _tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ScoreSensitivityError(f"{name} must be a rank-{ndim} tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise ScoreSensitivityError(f"{name} must be finite")
    return result


def build_local_current_pair(
    *,
    raw_payload: Mapping[str, object],
    sidecar: Mapping[str, object],
    class_mapper: Callable[[int], int],
) -> LocalCurrentPair:
    """Build one direct local-current pair without reading tracker identity."""

    validate_task_sidecar(sidecar)
    if not callable(class_mapper):
        raise ScoreSensitivityError("class_mapper must be callable")
    if raw_payload.get("key") != sidecar.get("key"):
        raise ScoreSensitivityError("raw payload and sidecar keys differ")
    target = raw_payload.get("target")
    task = sidecar.get("task_prediction")
    if not isinstance(target, Mapping) or not isinstance(task, Mapping):
        raise ScoreSensitivityError("local-current inputs must contain mappings")
    if target.get("gt_class_semantics") != "rescene_model_index_0_based":
        raise ScoreSensitivityError("raw target class semantics differ")
    if target.get("change_labels_valid") is not False:
        raise ScoreSensitivityError("raw target change-label validity differs")
    masks = _tensor(target.get("gt_masks"), name="local GT masks", ndim=2).bool()
    labels = _tensor(target.get("gt_classes"), name="local GT classes", ndim=1).long()
    ids = _tensor(target.get("gt_ids"), name="local GT IDs", ndim=1).long()
    changes = _tensor(target.get("changes"), name="local GT changes", ndim=1).long()
    pred_masks = _tensor(task.get("pred_masks"), name="local pred masks", ndim=2).bool()
    pred_scores = _tensor(
        task.get("pred_scores"), name="local pred scores", ndim=1
    ).float()
    pred_classes = _tensor(
        task.get("pred_classes"), name="local pred classes", ndim=1
    ).long()
    if (
        masks.shape[0] != labels.numel()
        or labels.shape != ids.shape
        or ids.shape != changes.shape
        or masks.shape[1] != pred_masks.shape[0]
        or pred_masks.shape[1] != pred_scores.numel()
        or pred_scores.shape != pred_classes.shape
        or torch.any(changes != 0).item()
    ):
        raise ScoreSensitivityError("local-current candidate/target coverage differs")
    mapped_labels = []
    for value in labels.tolist():
        mapped = class_mapper(int(value))
        if isinstance(mapped, bool) or not isinstance(mapped, int):
            raise ScoreSensitivityError("class_mapper must return integers")
        mapped_labels.append(mapped)
    return LocalCurrentPair(
        prediction={
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
            "pred_classes": pred_classes,
        },
        target={
            "masks": masks,
            "labels": torch.tensor(mapped_labels, dtype=torch.long),
            "ids": ids,
            "changes": changes,
            "temporal_stages": torch.zeros(masks.shape[1], dtype=torch.long),
        },
    )


def assert_score_only_snapshots(
    snapshots: Mapping[str, V2TrajectorySnapshot],
) -> dict[str, object]:
    if set(snapshots) != set(REDUCERS):
        raise ScoreSensitivityError("snapshot reducers differ from registration")
    reference = snapshots["mean"]
    for reducer in REDUCERS:
        snapshot = snapshots[reducer]
        if snapshot.score_reducer != reducer:
            raise ScoreSensitivityError("snapshot score reducer label differs")
        if (
            snapshot.stage_count != reference.stage_count
            or snapshot.keys != reference.keys
        ):
            raise ScoreSensitivityError(
                "trajectory keys or stage count changed by reducer"
            )
        if not torch.equal(
            snapshot.prediction["pred_masks"], reference.prediction["pred_masks"]
        ):
            raise ScoreSensitivityError("trajectory masks changed by reducer")
        if not torch.equal(
            snapshot.prediction["pred_classes"],
            reference.prediction["pred_classes"],
        ):
            raise ScoreSensitivityError("trajectory classes changed by reducer")
    ephemeral = [
        index for index, key in enumerate(reference.keys) if key.kind == "ephemeral"
    ]
    for index in ephemeral:
        value = reference.prediction["pred_scores"][index]
        if any(
            not torch.equal(snapshots[reducer].prediction["pred_scores"][index], value)
            for reducer in REDUCERS[1:]
        ):
            raise ScoreSensitivityError("ephemeral scores changed by reducer")
    return {
        "status": "pass_exact",
        "stage_count": reference.stage_count,
        "candidate_count": len(reference.keys),
        "ephemeral_count": len(ephemeral),
    }


def assert_local_current_invariance(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups: dict[tuple[str, str, str, int], list[Mapping[str, object]]] = defaultdict(
        list
    )
    for row in rows:
        key = (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
        )
        groups[key].append(row)
    for values in groups.values():
        if len(values) != len(TRACKERS) * len(REDUCERS):
            raise ScoreSensitivityError(
                "local-current tracker/reducer coverage differs"
            )
        if {str(row["tracker"]) for row in values} != set(TRACKERS) or {
            str(row["score_reducer"]) for row in values
        } != set(REDUCERS):
            raise ScoreSensitivityError("local-current labels differ")
        reference = values[0]
        for row in values[1:]:
            if row["local_sidecar_sha256"] != reference["local_sidecar_sha256"] or any(
                row[field] != reference[field] for field in LOCAL_FIELDS
            ):
                raise ScoreSensitivityError(
                    "local-current channel differs across tracker/reducer labels"
                )
    return {"status": "pass_exact", "group_count": len(groups)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0" + tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _keys_sha256(snapshot: V2TrajectorySnapshot) -> str:
    payload = [asdict(key) for key in snapshot.keys]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _local_metrics(pair: LocalCurrentPair) -> dict[str, float]:
    metric = OfficialMetricAccumulator(mode="raw_local")
    metric.update(pair.prediction, pair.target)
    values = metric.compute()
    return {
        "local_current_AP": values["raw_local_AP"],
        "local_current_AP50": values["raw_local_AP50"],
        "local_current_AP25": values["raw_local_AP25"],
        "local_current_REC": values["raw_local_REC"],
    }


def _trajectory_metrics(pair: object) -> dict[str, float]:
    values = compute_causal_task_metrics([pair])
    return {
        **{field: values[field] for field in CAUSAL_FIELDS},
        "trajectory_current_slice_AP": values["current_stage_AP"],
        "trajectory_current_slice_AP50": values["current_stage_AP50"],
        "trajectory_current_slice_AP25": values["current_stage_AP25"],
        "trajectory_current_slice_REC": values["current_stage_REC"],
    }


def _task_accumulator(
    values: dict[tuple[object, ...], CausalTaskAccumulator],
    key: tuple[object, ...],
) -> CausalTaskAccumulator:
    if key not in values:
        values[key] = CausalTaskAccumulator()
    return values[key]


def _local_accumulator(
    values: dict[tuple[object, ...], OfficialMetricAccumulator],
    key: tuple[object, ...],
) -> OfficialMetricAccumulator:
    if key not in values:
        values[key] = OfficialMetricAccumulator(mode="raw_local")
    return values[key]


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ScoreSensitivityError(f"JSON root must be a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ScoreSensitivityError(f"CSV must not be empty: {path}")
    return rows


def _old_maps() -> tuple[
    dict[tuple[str, str, str, int], Mapping[str, str]],
    dict[tuple[str, str, int], Mapping[str, str]],
]:
    per_sequence = _read_csv(V2_ROOT / "per_sequence_results.csv")
    aggregate = _read_csv(V2_ROOT / "aggregate_results.csv")
    aggregate.extend(_read_csv(V2_ROOT / "per_order_results.csv"))
    sequence_map = {
        (
            row["method"],
            row["master_sequence_id"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in per_sequence
    }
    aggregate_map = {
        (row["method"], row["order_id"], int(row["horizon"])): row for row in aggregate
    }
    return sequence_map, aggregate_map


def _regression_differences(
    row: Mapping[str, object], old: Mapping[str, str]
) -> list[float]:
    mapping = {
        **{field: field for field in CAUSAL_FIELDS},
        "trajectory_current_slice_AP": "current_stage_AP",
        "trajectory_current_slice_AP50": "current_stage_AP50",
        "trajectory_current_slice_AP25": "current_stage_AP25",
        "trajectory_current_slice_REC": "current_stage_REC",
    }
    return [
        abs(float(row[new]) - float(old[previous])) for new, previous in mapping.items()
    ]


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ScoreSensitivityError("output rows must not be empty")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _metric_block(accumulator: CausalTaskAccumulator) -> dict[str, float]:
    values = accumulator.compute()
    return {
        **{field: values[field] for field in CAUSAL_FIELDS},
        "trajectory_current_slice_AP": values["current_stage_AP"],
        "trajectory_current_slice_AP50": values["current_stage_AP50"],
        "trajectory_current_slice_AP25": values["current_stage_AP25"],
        "trajectory_current_slice_REC": values["current_stage_REC"],
    }


def _local_block(accumulator: OfficialMetricAccumulator) -> dict[str, float]:
    values = accumulator.compute()
    return {
        "local_current_AP": values["raw_local_AP"],
        "local_current_AP50": values["raw_local_AP50"],
        "local_current_AP25": values["raw_local_AP25"],
        "local_current_REC": values["raw_local_REC"],
    }


def _report(
    aggregate_rows: Sequence[Mapping[str, object]],
    *,
    mean_regression_max_abs_diff: float,
    local_invariance: Mapping[str, object],
    score_only_check_count: int,
) -> str:
    selected = [
        row
        for row in aggregate_rows
        if row["order_id"] == "all" and int(row["horizon"]) in HORIZONS
    ]
    index = {
        (str(row["tracker"]), str(row["score_reducer"]), int(row["horizon"])): row
        for row in selected
    }
    lines = [
        "# Score Reducer Sensitivity",
        "",
        "## EV0 Status",
        "",
        "**PASS.** Mean exactly regresses to frozen Persist4D-V2; latest/max",
        "change trajectory confidence aggregation only. Direct local-current AP",
        "is tracker/reducer invariant for every fixed official sidecar.",
        "",
        "## All-Order t-mAP",
        "",
        "| Horizon | Reducer | B2 | B3 | B4 | B4 - B2 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    sign_flips = []
    for horizon in HORIZONS:
        mean_delta = None
        for reducer in REDUCERS:
            b2 = float(index[("B2", reducer, horizon)]["causal_prefix_t_mAP"])
            b3 = float(index[("B3", reducer, horizon)]["causal_prefix_t_mAP"])
            b4 = float(index[("B4", reducer, horizon)]["causal_prefix_t_mAP"])
            delta = b4 - b2
            if reducer == "mean":
                mean_delta = delta
            elif mean_delta is not None and (delta > 0) != (mean_delta > 0):
                sign_flips.append((horizon, reducer))
            lines.append(
                f"| T{horizon} | {reducer} | {100 * b2:.3f} | {100 * b3:.3f} | "
                f"{100 * b4:.3f} | {100 * delta:+.3f} |"
            )
    interpretation = (
        "At least one B4-vs-B2 sign changes relative to mean; mean remains primary "
        "and the temporal AP ranking claim is score-aggregation sensitive."
        if sign_flips
        else "B4-vs-B2 signs agree across reducers; mean remains the primary result."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interpretation,
            "No reducer is selected based on its observed score.",
            "",
            "## Invariants",
            "",
            f"- Mean regression maximum absolute difference: `{mean_regression_max_abs_diff}`.",
            f"- Score-only snapshot checks: `{score_only_check_count}`.",
            f"- Local-current exact-invariance groups: `{local_invariance['group_count']}`.",
            "- Local masks, classes, official scores, target, and sidecar fingerprints",
            "  are read before and independently of B2/B3/B4 linkage.",
            "- `trajectory_current_slice_AP` remains a separate diagnostic from",
            "  `local_current_AP`.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_score_sensitivity(
    *,
    cache_root: Path = CACHE_ROOT,
    cache_manifest_path: Path = CACHE_MANIFEST,
    metadata_path: Path = METADATA,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    binding = _read_json(REPRODUCIBILITY_BINDING)
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    cache_manifest = _read_json(cache_manifest_path)
    sequences = load_v2_sequences(
        cache_manifest=cache_manifest,
        cache_root=cache_root,
    )
    class_mapper = build_rio_class_mapper(setup.dataset)
    factories = build_tracker_factories(setup.p6a_config)
    if not set(TRACKERS) <= set(factories):
        raise ScoreSensitivityError("registered tracker factories are incomplete")
    old_sequences, old_aggregate = _old_maps()

    task_aggregate: dict[tuple[object, ...], CausalTaskAccumulator] = {}
    task_cluster: dict[tuple[object, ...], CausalTaskAccumulator] = {}
    local_aggregate: dict[tuple[object, ...], OfficialMetricAccumulator] = {}
    local_cluster: dict[tuple[object, ...], OfficialMetricAccumulator] = {}
    aggregate_counts: dict[tuple[str, int], int] = defaultdict(int)
    cluster_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    per_sequence_rows: list[dict[str, object]] = []
    regression_differences: list[float] = []
    score_only_check_count = 0

    for sequence in sequences:
        trackers = {
            name: factories[name](f"{sequence.master_sequence_id}:{sequence.order_id}")
            for name in TRACKERS
        }
        trajectories = {
            tracker: {
                reducer: OfficialCandidateTrajectoryAccumulator(score_reducer=reducer)
                for reducer in REDUCERS
            }
            for tracker in TRACKERS
        }
        for stage, (raw, sidecar) in enumerate(
            zip(sequence.raw_payloads, sequence.sidecars, strict=True)
        ):
            horizon = stage + 1
            local_values = None
            sidecar_sha = task_sidecar_digest(sidecar)
            if horizon in HORIZONS:
                local_pair = build_local_current_pair(
                    raw_payload=raw,
                    sidecar=sidecar,
                    class_mapper=class_mapper,
                )
                local_values = _local_metrics(local_pair)
                for order_id in (sequence.order_id, "all"):
                    _local_accumulator(local_aggregate, (order_id, horizon)).update(
                        local_pair.prediction, local_pair.target
                    )
                    aggregate_counts[(order_id, horizon)] += 1
                _local_accumulator(
                    local_cluster,
                    (sequence.order_id, horizon, sequence.reference_scene_id),
                ).update(local_pair.prediction, local_pair.target)
                cluster_counts[
                    (sequence.order_id, horizon, sequence.reference_scene_id)
                ] += 1

            observation = cache_payload_to_frozen_observation(raw)
            for tracker_name in TRACKERS:
                step = trackers[tracker_name].step(observation, stage_id=stage)
                snapshots = {}
                for reducer in REDUCERS:
                    trajectory = trajectories[tracker_name][reducer]
                    trajectory.add_stage(sidecar, step)
                    snapshots[reducer] = trajectory.snapshot()
                assert_score_only_snapshots(snapshots)
                score_only_check_count += 1
                if horizon not in HORIZONS or local_values is None:
                    continue
                reference = snapshots["mean"]
                mask_sha = _tensor_sha256(reference.prediction["pred_masks"])
                class_sha = _tensor_sha256(reference.prediction["pred_classes"])
                key_sha = _keys_sha256(reference)
                for reducer in REDUCERS:
                    pair = build_v2_causal_pair(
                        snapshot=snapshots[reducer],
                        raw_payloads=sequence.raw_payloads[:horizon],
                        class_mapper=class_mapper,
                    )
                    values = _trajectory_metrics(pair)
                    row = {
                        "tracker": tracker_name,
                        "score_reducer": reducer,
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "horizon": horizon,
                        **values,
                        **local_values,
                        "local_sidecar_sha256": sidecar_sha,
                        "trajectory_mask_sha256": mask_sha,
                        "trajectory_class_sha256": class_sha,
                        "trajectory_key_sha256": key_sha,
                    }
                    per_sequence_rows.append(row)
                    for order_id in (sequence.order_id, "all"):
                        _task_accumulator(
                            task_aggregate,
                            (tracker_name, reducer, order_id, horizon),
                        ).update(pair)
                    _task_accumulator(
                        task_cluster,
                        (
                            tracker_name,
                            reducer,
                            sequence.order_id,
                            horizon,
                            sequence.reference_scene_id,
                        ),
                    ).update(pair)
                    if tracker_name == "B4" and reducer == "mean":
                        old = old_sequences[
                            (
                                "Persist4D-V2",
                                sequence.master_sequence_id,
                                sequence.order_id,
                                horizon,
                            )
                        ]
                        regression_differences.extend(_regression_differences(row, old))

    if len(per_sequence_rows) != len(TRACKERS) * len(REDUCERS) * 129 * len(HORIZONS):
        raise ScoreSensitivityError("per-sequence sensitivity coverage differs")
    local_invariance = assert_local_current_invariance(per_sequence_rows)

    aggregate_rows: list[dict[str, object]] = []
    for key, accumulator in sorted(task_aggregate.items()):
        tracker, reducer, order_id, horizon = key
        row = {
            "tracker": tracker,
            "score_reducer": reducer,
            "order_id": order_id,
            "horizon": horizon,
            "sequence_count": aggregate_counts[(order_id, horizon)],
            **_metric_block(accumulator),
            **_local_block(local_aggregate[(order_id, horizon)]),
        }
        aggregate_rows.append(row)
        if tracker == "B4" and reducer == "mean":
            old = old_aggregate[("Persist4D-V2", str(order_id), int(horizon))]
            regression_differences.extend(_regression_differences(row, old))

    cluster_rows: list[dict[str, object]] = []
    for key, accumulator in sorted(task_cluster.items()):
        tracker, reducer, order_id, horizon, cluster = key
        cluster_rows.append(
            {
                "tracker": tracker,
                "score_reducer": reducer,
                "reference_scene_id": cluster,
                "order_id": order_id,
                "horizon": horizon,
                "sequence_count": cluster_counts[(order_id, horizon, cluster)],
                **_metric_block(accumulator),
                **_local_block(local_cluster[(order_id, horizon, cluster)]),
                "inference_unit": "reference_scene_id",
            }
        )

    expected_aggregate = len(TRACKERS) * len(REDUCERS) * 4 * len(HORIZONS)
    expected_cluster = len(TRACKERS) * len(REDUCERS) * len(ORDERS) * len(HORIZONS) * 6
    if (
        len(aggregate_rows) != expected_aggregate
        or len(cluster_rows) != expected_cluster
    ):
        raise ScoreSensitivityError("aggregate/cluster sensitivity coverage differs")
    mean_regression = max(regression_differences, default=0.0)
    if mean_regression > 1e-12:
        raise ScoreSensitivityError("B4 mean does not regress to frozen Persist4D-V2")

    outputs = {
        "per_sequence.csv": _csv_bytes(per_sequence_rows),
        "aggregate.csv": _csv_bytes(aggregate_rows),
        "per_cluster.csv": _csv_bytes(cluster_rows),
    }
    report = _report(
        aggregate_rows,
        mean_regression_max_abs_diff=mean_regression,
        local_invariance=local_invariance,
        score_only_check_count=score_only_check_count,
    ).encode("utf-8")
    outputs["SCORE_REDUCER_SENSITIVITY.md"] = report
    for name, content in outputs.items():
        _write(output_root / name, content)

    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "gate_ev0": {
            "status": "PASS",
            "primary_reducer": "mean",
            "sensitivity_reducers": ["latest", "max"],
            "mean_v2_regression_max_abs_diff": mean_regression,
            "score_only_snapshot_check_count": score_only_check_count,
            "score_only_snapshot_status": "pass_exact",
            "local_current_invariance": dict(local_invariance),
        },
        "coverage": {
            "sequence_count": 129,
            "trackers": list(TRACKERS),
            "reducers": list(REDUCERS),
            "orders": list(ORDERS),
            "horizons": list(HORIZONS),
            "per_sequence_row_count": len(per_sequence_rows),
            "aggregate_row_count": len(aggregate_rows),
            "per_cluster_row_count": len(cluster_rows),
            "reference_cluster_count": 6,
        },
        "inputs": {
            "reproducibility_binding": {
                "reference": "repo:artifacts/system_comparison/reproducibility_binding.json",
                "sha256": _sha256(REPRODUCIBILITY_BINDING),
            },
            "cache_manifest": {
                "reference": "repo:artifacts/system_comparison_v2/cache_manifest.json",
                "sha256": _sha256(cache_manifest_path),
            },
            "external_cache_root": f"external:{cache_root.resolve()}",
            "frozen_v2_manifest": {
                "reference": "repo:artifacts/system_comparison_v2/manifest.json",
                "sha256": _sha256(V2_ROOT / "manifest.json"),
            },
            "protocol_manifest_sha256": cache_manifest["protocol_manifest_sha256"],
            "checkpoint_sha256": cache_manifest["checkpoint_sha256"],
            "config_sha256": binding["config_sha256"],
            "cache_records_sha256": cache_manifest["records_sha256"],
        },
        "execution": {
            "mode": "frozen_cache_postprocessing",
            "gpu_inference_performed": False,
            "metric_backend": "stmetrics via OfficialMetricAccumulator",
        },
        "outputs": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in outputs.items()
        },
        "scripts": {
            "score_sensitivity_sha256": _sha256(Path(__file__)),
            "trajectory_accumulator_sha256": _sha256(
                PROJECT_ROOT / "scripts/system_comparison_v2_inference.py"
            ),
            "test_sha256": _sha256(
                PROJECT_ROOT / "tests/test_system_comparison_v3_score_reducers.py"
            ),
        },
        "channel_contract": {
            "local_current": "latest-stage official sidecar with raw-local stmetrics",
            "trajectory": "causal linked candidates with explicit score reducer",
            "trajectory_current_slice_is_local_current": False,
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write(output_root / "manifest.json", manifest_bytes)
    return {
        "status": "pass",
        "gate": "EV0",
        "per_sequence_rows": len(per_sequence_rows),
        "aggregate_rows": len(aggregate_rows),
        "per_cluster_rows": len(cluster_rows),
        "mean_regression_max_abs_diff": mean_regression,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    print(
        json.dumps(
            run_score_sensitivity(
                cache_root=arguments.cache_root,
                cache_manifest_path=arguments.cache_manifest,
                metadata_path=arguments.metadata,
                output_root=arguments.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
