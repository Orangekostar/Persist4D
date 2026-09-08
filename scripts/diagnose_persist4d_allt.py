#!/usr/bin/env python3
"""Run bounded epoch-390 Persist4D All-T failure diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CACHE_ROOT = Path("/mnt/shared/ww/persist4d-r1-downstream-validation-v1/cache")
DEFAULT_CHECKPOINT = Path(
    "/mnt/shared/ww/persist4d-rescene-finalization/checkpoints/"
    "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd.ckpt"
)
DEFAULT_PRETRAINED = Path(
    "/mnt/shared/ww/persist4d-node1-7-migration/checkpoints/concerto_base.pth"
)
DEFAULT_METADATA = Path("/home/ww/3RScan.json")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts/allt_task_superiority_v1/diagnostics"


class AllTDiagnosticError(RuntimeError):
    """Raised when the diagnostic panel or gate is invalid."""


def classify_prefix_failure(
    class_compatible_ious: Sequence[float], other_class_ious: Sequence[float]
) -> dict[str, bool | float]:
    compatible = _iou_values(class_compatible_ious, "class-compatible")
    other = _iou_values(other_class_ious, "other-class")
    best_compatible = max(compatible, default=0.0)
    best_other = max(other, default=0.0)
    no_candidate = max(best_compatible, best_other) == 0.0
    return {
        "best_any_iou": max(best_compatible, best_other),
        "best_class_compatible_iou": best_compatible,
        "class_change": best_other >= 0.5 and best_compatible < 0.5,
        "mask_insufficient": best_compatible > 0.0 and best_compatible < 0.5,
        "no_candidate": no_candidate,
    }


def prefix_failure_records(pair: object, *, min_region_size: int = 100) -> list[dict[str, object]]:
    import torch

    from scripts.system_comparison_metrics import CausalPrefixPair

    if not isinstance(pair, CausalPrefixPair):
        raise AllTDiagnosticError("failure input must be a causal prefix pair")
    if (
        isinstance(min_region_size, bool)
        or not isinstance(min_region_size, int)
        or min_region_size <= 0
    ):
        raise AllTDiagnosticError("min_region_size must be a positive integer")
    prediction_masks = pair.prediction["pred_masks"].transpose(0, 1).bool()
    target_masks = pair.target["masks"].bool()
    prediction_float = prediction_masks.float()
    target_float = target_masks.float()
    intersection = prediction_float @ target_float.transpose(0, 1)
    union = (
        prediction_float.sum(dim=1, keepdim=True)
        + target_float.sum(dim=1).unsqueeze(0)
        - intersection
    )
    ious = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    prediction_classes = pair.prediction["pred_classes"].long()
    rows = []
    for gt_index in range(target_masks.shape[0]):
        gt_class = int(pair.target["labels"][gt_index].item())
        compatible = prediction_classes == gt_class
        classified = classify_prefix_failure(
            ious[compatible, gt_index].tolist(),
            ious[~compatible, gt_index].tolist(),
        )
        point_count = int(target_masks[gt_index].sum().item())
        rows.append(
            {
                **classified,
                "gt_class": gt_class,
                "gt_id": int(pair.target["ids"][gt_index].item()),
                "gt_point_count": point_count,
                "official_valid": point_count >= min_region_size,
            }
        )
    return rows


FAILURE_SUMMARY_FIELDS = (
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "gt_id",
    "gt_class",
    "gt_point_count",
    "official_valid",
    "b4_best_any_iou",
    "b4_best_class_compatible_iou",
    "full_history_best_class_compatible_iou",
    "no_candidate",
    "mask_insufficient",
    "class_change",
    "full_history_success_b4_failure",
    "fragmentation_event",
    "merge_event",
    "trajectory_worst_b4_iou",
    "first_b4_below_0_25_T",
    "first_b4_below_0_50_T",
    "t1_cold_start_failure",
    "t1_cold_start_contributes",
)


def _identity_events(
    updates: Sequence[object],
) -> dict[tuple[int, int], dict[str, bool]]:
    from scripts.system_comparison_metrics import IdentityAssignmentUpdate

    if (
        isinstance(updates, (str, bytes))
        or not isinstance(updates, Sequence)
        or [getattr(update, "horizon", None) for update in updates]
        != [1, 2, 3, 4, 5]
        or any(not isinstance(update, IdentityAssignmentUpdate) for update in updates)
    ):
        raise AllTDiagnosticError("identity updates must cover T1-T5")
    issued_by_gt: dict[int, set[int]] = defaultdict(set)
    gt_by_issued: dict[int, set[int]] = defaultdict(set)
    events: dict[tuple[int, int], dict[str, bool]] = {}
    for update in updates:
        for gt_id, issued_id in sorted(update.assignments.items()):
            events[(update.horizon, gt_id)] = {
                "fragmentation_event": bool(issued_by_gt[gt_id])
                and issued_id not in issued_by_gt[gt_id],
                "merge_event": bool(gt_by_issued[issued_id])
                and gt_id not in gt_by_issued[issued_id],
            }
            issued_by_gt[gt_id].add(issued_id)
            gt_by_issued[issued_id].add(gt_id)
    return events


def build_failure_summary_rows(
    *,
    reference_scene_id: str,
    master_sequence_id: str,
    order_id: str,
    b4_pairs: Sequence[object],
    full_history_pairs: Sequence[object],
    identity_updates: Sequence[object],
    min_region_size: int = 100,
) -> list[dict[str, object]]:
    from scripts.system_comparison_metrics import (
        CausalPrefixPair,
        validate_causal_prefix_pair,
    )

    identifiers = (reference_scene_id, master_sequence_id, order_id)
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise AllTDiagnosticError("failure summary identifiers are incomplete")
    if (
        isinstance(b4_pairs, (str, bytes))
        or not isinstance(b4_pairs, Sequence)
        or [getattr(pair, "horizon", None) for pair in b4_pairs]
        != [1, 2, 3, 4, 5]
        or any(not isinstance(pair, CausalPrefixPair) for pair in b4_pairs)
    ):
        raise AllTDiagnosticError("B4 failure pairs must cover T1-T5")
    if (
        isinstance(full_history_pairs, (str, bytes))
        or not isinstance(full_history_pairs, Sequence)
        or [getattr(pair, "horizon", None) for pair in full_history_pairs]
        != [2, 3, 4, 5]
        or any(not isinstance(pair, CausalPrefixPair) for pair in full_history_pairs)
    ):
        raise AllTDiagnosticError("FullHistory failure pairs must cover T2-T5")

    b4_by_horizon = {
        pair.horizon: prefix_failure_records(pair, min_region_size=min_region_size)
        for pair in b4_pairs
    }
    canonical_target_by_horizon = {pair.horizon: pair.target for pair in b4_pairs}
    full_by_horizon = {
        pair.horizon: prefix_failure_records(
            validate_causal_prefix_pair(
                prediction=pair.prediction,
                target=canonical_target_by_horizon[pair.horizon],
                horizon=pair.horizon,
                observed_scan_ids=pair.observed_scan_ids,
            ),
            min_region_size=min_region_size,
        )
        for pair in full_history_pairs
    }
    b4_by_gt = {
        horizon: {int(row["gt_id"]): row for row in rows}
        for horizon, rows in b4_by_horizon.items()
    }
    full_by_gt = {
        horizon: {int(row["gt_id"]): row for row in rows}
        for horizon, rows in full_by_horizon.items()
    }
    identity_events = _identity_events(identity_updates)
    result = []
    for horizon in range(2, 6):
        if set(b4_by_gt[horizon]) != set(full_by_gt[horizon]):
            raise AllTDiagnosticError("canonical diagnostic GT alignment failed")
        for gt_id in sorted(b4_by_gt[horizon]):
            current = b4_by_gt[horizon][gt_id]
            full = full_by_gt[horizon][gt_id]
            if current["gt_class"] != full["gt_class"]:
                raise AllTDiagnosticError("B4 and FullHistory GT classes differ")
            trajectory = [
                b4_by_gt[stage][gt_id]
                for stage in range(2, 6)
                if gt_id in b4_by_gt[stage]
            ]
            first_below_25 = next(
                (
                    stage
                    for stage in range(2, 6)
                    if gt_id in b4_by_gt[stage]
                    and b4_by_gt[stage][gt_id]["best_class_compatible_iou"] < 0.25
                ),
                None,
            )
            first_below_50 = next(
                (
                    stage
                    for stage in range(2, 6)
                    if gt_id in b4_by_gt[stage]
                    and b4_by_gt[stage][gt_id]["best_class_compatible_iou"] < 0.5
                ),
                None,
            )
            cold = b4_by_gt[1].get(gt_id)
            cold_failure = bool(
                cold is not None and cold["best_class_compatible_iou"] < 0.5
            )
            b4_failed = current["best_class_compatible_iou"] < 0.5
            identity = identity_events.get(
                (horizon, gt_id),
                {"fragmentation_event": False, "merge_event": False},
            )
            row = {
                "reference_scene_id": reference_scene_id,
                "master_sequence_id": master_sequence_id,
                "order_id": order_id,
                "T": horizon,
                "gt_id": gt_id,
                "gt_class": current["gt_class"],
                "gt_point_count": current["gt_point_count"],
                "official_valid": current["official_valid"],
                "b4_best_any_iou": current["best_any_iou"],
                "b4_best_class_compatible_iou": current[
                    "best_class_compatible_iou"
                ],
                "full_history_best_class_compatible_iou": full[
                    "best_class_compatible_iou"
                ],
                "no_candidate": current["no_candidate"],
                "mask_insufficient": current["mask_insufficient"],
                "class_change": current["class_change"],
                "full_history_success_b4_failure": full[
                    "best_class_compatible_iou"
                ]
                >= 0.5
                and b4_failed,
                **identity,
                "trajectory_worst_b4_iou": min(
                    float(value["best_class_compatible_iou"])
                    for value in trajectory
                ),
                "first_b4_below_0_25_T": first_below_25,
                "first_b4_below_0_50_T": first_below_50,
                "t1_cold_start_failure": cold_failure,
                "t1_cold_start_contributes": cold_failure and b4_failed,
            }
            if tuple(row) != FAILURE_SUMMARY_FIELDS:
                raise AllTDiagnosticError("failure summary row schema differs")
            result.append(row)
    if not result:
        raise AllTDiagnosticError("failure summary is empty")
    return result


def failure_summary_statistics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise AllTDiagnosticError("failure statistics require rows")
    if any(tuple(row) != FAILURE_SUMMARY_FIELDS for row in rows):
        raise AllTDiagnosticError("failure statistics row schema differs")
    category_fields = (
        "no_candidate",
        "mask_insufficient",
        "class_change",
        "full_history_success_b4_failure",
        "fragmentation_event",
        "merge_event",
    )
    official = [row for row in rows if row["official_valid"] is True]
    failed = [row for row in rows if row["b4_best_class_compatible_iou"] < 0.5]
    contributed = [row for row in failed if row["t1_cold_start_contributes"] is True]
    return {
        "category_counts_all": {
            field: sum(row[field] is True for row in rows) for field in category_fields
        },
        "category_counts_official_valid": {
            field: sum(row[field] is True for row in official)
            for field in category_fields
        },
        "category_semantics": "multi_label_non_additive",
        "official_valid_row_count": len(official),
        "row_count": len(rows),
        "t1_cold_start_contribution": {
            "contributed_row_count": len(contributed),
            "failed_row_count": len(failed),
            "fraction": len(contributed) / len(failed) if failed else None,
        },
        "unique_gt_trajectory_count": len(
            {
                (
                    row["reference_scene_id"],
                    row["master_sequence_id"],
                    row["order_id"],
                    row["gt_id"],
                )
                for row in rows
            }
        ),
    }


def summarize_decoder_rows(
    query_rows: Sequence[Mapping[str, object]],
    attention_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected_query = [
        row
        for row in query_rows
        if isinstance(row, Mapping) and row.get("feeds_next_attention") is True
    ]
    if not selected_query or any(
        not isinstance(row.get("file_name"), str) or not row["file_name"]
        for row in selected_query
    ):
        raise AllTDiagnosticError("attention-feeding query rows are incomplete")
    layers = []
    for row in attention_rows:
        if not isinstance(row, Mapping):
            raise AllTDiagnosticError("attention rows must be mappings")
        layer = row.get("decoder_prediction_layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise AllTDiagnosticError("attention decoder layer is invalid")
        layers.append(layer)
    if not layers:
        raise AllTDiagnosticError("attention rows are empty")
    earliest_layer = min(layers)
    earliest_rows = [
        row
        for row in attention_rows
        if row["decoder_prediction_layer"] == earliest_layer
    ]
    allowed_values = [
        _rate(row.get("allowed_gt_fraction"), "allowed") for row in earliest_rows
    ]

    query_summary = {
        "competed_active_query_fraction": _mean_field(
            selected_query, "competed_active_query_fraction"
        ),
        "mean_queries_per_gt_iou25": _mean_field(
            selected_query, "mean_queries_per_gt_iou25"
        ),
        "query_utilization_iou25": _mean_field(
            selected_query, "query_utilization_iou25"
        ),
        "valid_row_count": len(selected_query),
        "valid_scene_count": len({row["file_name"] for row in selected_query}),
    }
    attention_summary = {
        "allowed_gt_fraction": sum(allowed_values) / len(allowed_values),
        "decoder_prediction_layer": earliest_layer,
        "severe_below_0_25_fraction": (
            sum(value < 0.25 for value in allowed_values) / len(allowed_values)
        ),
        "valid_match_count": len(allowed_values),
    }
    summary = {
        "earliest_attention": attention_summary,
        "query_competition": query_summary,
    }
    summary["l_gate"] = select_l_design(summary)
    return summary


def _mean_field(rows: Sequence[Mapping[str, object]], field: str) -> float:
    values = [_rate(row.get(field), field) for row in rows]
    return sum(values) / len(values)


def _iou_values(values: Sequence[float], label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise AllTDiagnosticError(f"{label} IoUs must be a sequence")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AllTDiagnosticError(f"{label} IoUs must be finite rates")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise AllTDiagnosticError(f"{label} IoUs must be finite rates")
        result.append(number)
    return result


def select_diagnostic_panel(
    masters: Sequence[Mapping[str, object]],
    *,
    per_reference: int,
    maximum_sequences: int,
) -> list[dict[str, str]]:
    if (
        isinstance(per_reference, bool)
        or not isinstance(per_reference, int)
        or per_reference <= 0
        or isinstance(maximum_sequences, bool)
        or not isinstance(maximum_sequences, int)
        or maximum_sequences <= 0
    ):
        raise AllTDiagnosticError("panel limits must be positive integers")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise AllTDiagnosticError("masters must be a sequence")
    grouped: dict[str, set[str]] = defaultdict(set)
    for index, master in enumerate(masters):
        if not isinstance(master, Mapping):
            raise AllTDiagnosticError(f"master {index} must be a mapping")
        reference = master.get("reference_scene_id")
        sequence = master.get("sequence_id")
        if not isinstance(reference, str) or not reference:
            raise AllTDiagnosticError(f"master {index} lacks reference_scene_id")
        if not isinstance(sequence, str) or not sequence:
            raise AllTDiagnosticError(f"master {index} lacks sequence_id")
        if sequence in grouped[reference]:
            raise AllTDiagnosticError("diagnostic sequence IDs must be unique")
        grouped[reference].add(sequence)
    if not grouped or any(len(values) < per_reference for values in grouped.values()):
        raise AllTDiagnosticError("diagnostic per-reference coverage is incomplete")
    if len(grouped) * per_reference > maximum_sequences:
        raise AllTDiagnosticError("diagnostic panel exceeds maximum_sequences")
    return [
        {"reference_scene_id": reference, "sequence_id": sequence}
        for reference in sorted(grouped)
        for sequence in sorted(grouped[reference])[:per_reference]
    ]


def select_l_design(summary: Mapping[str, object]) -> dict[str, str | None]:
    if not isinstance(summary, Mapping):
        raise AllTDiagnosticError("decoder summary must be a mapping")
    competition = summary.get("query_competition")
    attention = summary.get("earliest_attention")
    if not isinstance(competition, Mapping) or not isinstance(attention, Mapping):
        raise AllTDiagnosticError("decoder summary blocks are incomplete")

    competed = _rate(competition.get("competed_active_query_fraction"), "competed")
    queries = _rate(competition.get("mean_queries_per_gt_iou25"), "queries")
    allowed = _rate(attention.get("allowed_gt_fraction"), "allowed")
    scene_count = competition.get("valid_scene_count")
    match_count = attention.get("valid_match_count")
    if (
        isinstance(scene_count, bool)
        or not isinstance(scene_count, int)
        or scene_count <= 0
        or isinstance(match_count, bool)
        or not isinstance(match_count, int)
        or match_count <= 0
    ):
        raise AllTDiagnosticError("decoder summary valid counts are missing")

    if competed >= 0.25 and queries >= 2.0:
        return {
            "design": "qcl_inspired_current_prediction",
            "reason": "substantial_query_competition",
            "status": "JUSTIFIED",
        }
    if allowed < 0.5:
        return {
            "design": "first_cross_attention_relaxation",
            "reason": "early_mask_excludes_matched_gt",
            "status": "JUSTIFIED",
        }
    return {
        "design": None,
        "reason": "diagnostic_gate_not_met",
        "status": "NOT_JUSTIFIED",
    }


def _rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AllTDiagnosticError(f"{label} diagnostic must be finite")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AllTDiagnosticError(f"{label} diagnostic must be finite")
    return result


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(result) != 40:
        raise AllTDiagnosticError("Git HEAD is invalid")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_panel(protocol_manifest: Mapping[str, object]) -> list[dict[str, str]]:
    masters = protocol_manifest.get("masters")
    if isinstance(masters, (str, bytes)) or not isinstance(masters, Sequence):
        raise AllTDiagnosticError("Protocol-B manifest lacks masters")
    normalized = []
    for master in masters:
        if not isinstance(master, Mapping):
            raise AllTDiagnosticError("Protocol-B master must be a mapping")
        normalized.append(
            {
                "reference_scene_id": master.get("reference_scene_id"),
                "sequence_id": master.get("master_sequence_id"),
            }
        )
    panel = select_diagnostic_panel(
        normalized,
        per_reference=2,
        maximum_sequences=12,
    )
    if len(panel) != 12:
        raise AllTDiagnosticError("fixed diagnostic panel must contain 12 sequences")
    return panel


def _cache_failure_rows(
    *,
    setup: object,
    cache_root: Path,
    local_progress: Mapping[str, object],
    full_progress: Mapping[str, object],
    panel: Sequence[Mapping[str, str]],
) -> list[dict[str, object]]:
    from scripts.analyze_persist4d_allt import (
        _build_sequence_jobs,
        _cache_key,
        _initialize_worker,
        _load_worker_sequence,
    )
    from scripts.analyze_r1_downstream_validation import resolve_metric_dataset_spec
    from scripts.evaluate_persist4d_p6a import (
        build_rio_class_mapper,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
        expected_cache_keys,
    )
    from scripts.system_comparison_analysis import _persistent_identity_updates
    from scripts.system_comparison_inference import load_full_history_cache_entry
    from scripts.system_comparison_metrics import causal_prefix_pair_from_payload
    from scripts.system_comparison_v2_analysis import build_v2_causal_pair
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )
    from scripts.system_comparison_v3_identity import run_fresh_tracker_steps

    class_mapper = build_rio_class_mapper(setup.dataset)
    rio_mapping = tuple(class_mapper(index) for index in range(18))
    expected = [
        (
            str(key["master_sequence_id"]),
            str(key["order_id"]),
            int(key["stage_index"]) + 1,
        )
        for key in expected_cache_keys(setup.protocol)
    ]
    jobs = _build_sequence_jobs(
        local_records=local_progress["records"],
        full_records=full_progress["records"],
        expected_keys=expected,
    )
    panel_ids = {row["sequence_id"] for row in panel}
    selected_jobs = [
        job
        for job in jobs
        if job["order_id"] == "canonical"
        and job["master_sequence_id"] in panel_ids
    ]
    if len(selected_jobs) != 12:
        raise AllTDiagnosticError("cache does not cover the fixed diagnostic panel")

    _initialize_worker(
        str(cache_root.resolve()),
        str(resolve_metric_dataset_spec(PROJECT_ROOT)),
        full_progress["provenance"],
        setup.p6a_config,
        rio_mapping,
    )
    factories = build_tracker_factories(setup.p6a_config)
    b4_settings = setup.p6a_config["baselines"]["b4"]
    rows = []
    for job in selected_jobs:
        sequence = _load_worker_sequence(job)
        observations = tuple(
            cache_payload_to_frozen_observation(raw)
            for raw in sequence.raw_payloads
        )
        steps = run_fresh_tracker_steps(
            factory=factories["B4"],
            observations=observations,
            sequence_id=f"{sequence.master_sequence_id}:{sequence.order_id}",
        )
        trajectory = OfficialCandidateTrajectoryAccumulator(score_reducer="mean")
        b4_pairs = []
        for sidecar, step, horizon in zip(
            sequence.sidecars, steps, range(1, 6), strict=True
        ):
            trajectory.add_stage(sidecar, step)
            b4_pairs.append(
                build_v2_causal_pair(
                    snapshot=trajectory.snapshot(),
                    raw_payloads=sequence.raw_payloads[:horizon],
                    class_mapper=class_mapper,
                )
            )
        full_records = {
            _cache_key(record, local=False)[2]: record
            for record in job["full_records"]
        }
        full_pairs = [
            causal_prefix_pair_from_payload(
                load_full_history_cache_entry(
                    cache_root / "full_history/entries",
                    full_records[horizon],
                    expected_provenance=full_progress["provenance"],
                )
            )
            for horizon in range(2, 6)
        ]
        identity_updates = _persistent_identity_updates(
            payloads=sequence.raw_payloads,
            steps=steps,
            class_mapper=class_mapper,
            background_class=int(b4_settings["background_class"]),
        )
        rows.extend(
            build_failure_summary_rows(
                reference_scene_id=sequence.reference_scene_id,
                master_sequence_id=sequence.master_sequence_id,
                order_id=sequence.order_id,
                b4_pairs=b4_pairs,
                full_history_pairs=full_pairs,
                identity_updates=identity_updates,
            )
        )
    if any(tuple(row) != FAILURE_SUMMARY_FIELDS for row in rows):
        raise AllTDiagnosticError("cache failure output schema differs")
    return rows


def _official_valid_target(
    target: Mapping[str, object], *, min_region_size: int
) -> dict[str, object]:
    import torch

    masks = target.get("masks")
    if not isinstance(masks, torch.Tensor) or masks.ndim != 2:
        raise AllTDiagnosticError("decoder target masks differ")
    keep = masks.bool().sum(dim=1) >= min_region_size
    if not torch.any(keep).item():
        raise AllTDiagnosticError("decoder panel sample has no official-valid GT")
    result = dict(target)
    for field in ("masks", "labels", "ids"):
        value = result.get(field)
        if not isinstance(value, torch.Tensor) or value.shape[0] != keep.numel():
            raise AllTDiagnosticError("decoder target instance fields differ")
        result[field] = value[keep]
    return result


def _decoder_rows(
    *, setup: object, panel: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], int]:
    import torch

    from scripts.evaluate_persist4d import (
        _move_data_to_device,
        _move_targets_to_device,
    )
    from scripts.evaluate_persist4d_p6a import (
        _frozen_inference_seed,
        expected_cache_keys,
        resolve_protocol_cache_request,
    )
    from utils.rescene_rootcause_diagnostics import (
        attention_mask_records,
        query_conflict_records,
    )

    if setup.system is None or setup.device is None:
        raise AllTDiagnosticError("epoch390 decoder system was not loaded")
    panel_ids = {row["sequence_id"] for row in panel}
    keys = [
        key
        for key in expected_cache_keys(setup.protocol)
        if key["master_sequence_id"] in panel_ids
        and key["order_id"] == "canonical"
        and key["stage_index"] in {0, 4}
    ]
    keys.sort(key=lambda key: (str(key["master_sequence_id"]), int(key["stage_index"])))
    if len(keys) != 24:
        raise AllTDiagnosticError("decoder panel must cover T1 and T5 exactly")

    query_rows = []
    attention_rows = []
    for key in keys:
        request = resolve_protocol_cache_request(setup.protocol, key)
        with _frozen_inference_seed(int(setup.contract["seed"]), setup.device):
            sample = setup.dataset.load_scan_indices(
                request.context_index,
                request.scan_indices,
                change_file=None,
            )
            data, targets, names = setup.collate([sample])
            if list(names) != [request.master_sequence_id] or len(targets) != 1:
                raise AllTDiagnosticError("decoder collator changed panel identity")
            data = _move_data_to_device(data, setup.device)
            targets = _move_targets_to_device(targets, setup.device)
            target = _official_valid_target(targets[0], min_region_size=100)
            reset_layers: list[list[dict[str, int]]] = []
            model = setup.system.model
            original_sample_and_batch = model.sample_and_batch_features

            def sample_and_batch_features(
                *args: object,
                _original: object = original_sample_and_batch,
                _reset_layers: list[list[dict[str, int]]] = reset_layers,
                **kwargs: object,
            ) -> object:
                result = _original(*args, **kwargs)
                if kwargs.get("extra") is not None:
                    batched_attention = result[1]
                    if not isinstance(batched_attention, torch.Tensor):
                        raise AllTDiagnosticError("attention capture differs")
                    all_masked = (
                        batched_attention.sum(dim=1) == batched_attention.shape[1]
                    )
                    _reset_layers.append(
                        [
                            {
                                "reset_count": int(all_masked[index].sum().item()),
                                "query_count": int(all_masked.shape[1]),
                            }
                            for index in range(all_masked.shape[0])
                        ]
                    )
                return result

            model.sample_and_batch_features = sample_and_batch_features
            try:
                raw_coordinates = setup.system._process_raw_coordinates(data)
                with torch.inference_mode():
                    output = setup.system(
                        data,
                        point2segment=[target["point2segment"]],
                        raw_coordinates=raw_coordinates,
                        is_eval=True,
                    )
            finally:
                model.sample_and_batch_features = original_sample_and_batch
            if not isinstance(output, Mapping) or any(
                len(layer) != 1 for layer in reset_layers
            ):
                raise AllTDiagnosticError("epoch390 decoder capture differs")
            file_name = f"{request.master_sequence_id}:T{request.stage_index + 1}"
            query_rows.extend(
                query_conflict_records(
                    file_name=file_name,
                    output=output,
                    target=target,
                )
            )
            attention_rows.extend(
                attention_mask_records(
                    file_name=file_name,
                    output=output,
                    target=target,
                    reset_counts=[layer[0] for layer in reset_layers],
                )
            )
    return query_rows, attention_rows, len(keys)


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=FAILURE_SUMMARY_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {field: "" if row[field] is None else row[field] for field in FAILURE_SUMMARY_FIELDS}
        )
    return stream.getvalue().encode("ascii")


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise AllTDiagnosticError(f"refusing to overwrite diagnostic output: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_allt_diagnostic(
    *,
    cache_root: Path,
    checkpoint: Path,
    pretrained: Path,
    metadata: Path,
    output_root: Path,
    device_name: str,
) -> dict[str, object]:
    from scripts.analyze_r1_downstream_validation import _validate_new_cache_binding
    from scripts.persist4d_allt_contract import canonical_json_sha256
    from scripts.r1_downstream_context import build_r1_setup

    _, local_progress, full_progress = _validate_new_cache_binding(
        cache_root=cache_root,
        artifact_root=PROJECT_ROOT / "artifacts/r1_downstream_validation_v1",
    )
    source_commit = _git_head()
    setup = build_r1_setup(
        contract_path=PROJECT_ROOT / "configs/r1_downstream_validation/default.yaml",
        protocol_path=PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json",
        checkpoint_path=checkpoint,
        pretrained_path=pretrained,
        metadata_path=metadata,
        data_root=PROJECT_ROOT,
        source_commit=source_commit,
        device_name=device_name,
    )
    panel = _fixed_panel(setup.protocol_manifest)
    failure_rows = _cache_failure_rows(
        setup=setup,
        cache_root=cache_root,
        local_progress=local_progress,
        full_progress=full_progress,
        panel=panel,
    )
    query_rows, attention_rows, forward_count = _decoder_rows(
        setup=setup,
        panel=panel,
    )
    decoder = summarize_decoder_rows(query_rows, attention_rows)
    payload = {
        "ambiguity_semantics": {
            "failure_categories": "multi_label_non_additive",
            "matching": "class_compatible_best_iou_diagnostic_not_official_t_map",
        },
        "checkpoint": {
            "completed_epoch": int(setup.contract["checkpoint"]["completed_epoch"]),
            "sha256": _file_sha256(checkpoint),
        },
        "decoder": decoder,
        "failure_summary": failure_summary_statistics(failure_rows),
        "l_gate": decoder["l_gate"],
        "model_forward_count": forward_count,
        "panel": list(panel),
        "panel_evidence_role": "historical_protocol_b_exploratory_development",
        "panel_order": "canonical",
        "protocol_sha256": _file_sha256(
            PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
        ),
        "schema_version": 1,
        "source_commit": source_commit,
        "status": "pass",
        "thresholds": {
            "attention_allowed_gt_fraction": 0.5,
            "competition_fraction": 0.25,
            "competition_queries_per_gt": 2.0,
            "diagnostic_iou": [0.25, 0.5],
            "official_min_region_size": 100,
        },
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    csv_payload = _csv_bytes(failure_rows)
    _publish(output_root / "failure_summary.csv", csv_payload)
    _publish(output_root / "final_r1_decoder_summary.json", _json_bytes(payload))
    return {
        "decoder_attention_rows": len(attention_rows),
        "decoder_query_rows": len(query_rows),
        "failure_rows": len(failure_rows),
        "failure_summary_sha256": hashlib.sha256(csv_payload).hexdigest(),
        "l_gate": decoder["l_gate"],
        "model_forward_count": forward_count,
        "status": "pass",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:2")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_allt_diagnostic(
        cache_root=args.cache_root,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
        metadata=args.metadata,
        output_root=args.output_root,
        device_name=args.device,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
