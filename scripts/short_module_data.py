"""Round-one supervision adapter. GT is confined to this training/evaluation layer."""

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from models.short_module_heads import Module
from scripts.p6a_metrics import (
    OfficialMetricAccumulator,
    global_hungarian_match,
    official_temporal_iou_thresholds,
)


def _order(value: str) -> str:
    return hashlib.sha256(("short-module-v1:" + value).encode()).hexdigest()


def select_population(train_records: list[dict], cal_records: list[dict],
                      sel_records: list[dict], *, excluded_references: set[str]) -> dict:
    """Select TRAIN from legal metadata only; retain inherited CAL/SEL weights."""
    grouped = {}
    for record in train_records:
        reference = record["reference_id"]
        if reference in excluded_references:
            raise ValueError("TRAIN overlap with held-out or exposed reference")
        if len(record["scan_ids"]) != 2 or len(set(record["scan_ids"])) != 2:
            raise ValueError("TRAIN must contain legal directed pairs")
        grouped.setdefault(reference, {})[tuple(record["scan_ids"])] = record
    references = sorted(grouped, key=lambda r: (_order(r), r))[:64]
    if len(references) < 8:
        raise ValueError("BLOCKED_TRAIN_COVERAGE: fewer than eight legal references")
    selected_train = []
    for reference in references:
        pairs = sorted(grouped[reference], key=lambda p: (_order("-".join(p)), p))[:4]
        selected_train.extend(grouped[reference][pair] for pair in pairs)
    records, counts = [], {}
    for role, pairs in (("TRAIN", selected_train), ("CAL", cal_records), ("SEL", sel_records)):
        scans = {}
        for index, pair in enumerate(pairs):
            base = {**pair, "role": role, "horizon": 2,
                    "logical_unit_id": f"{role}:T2:{index:05d}"}
            records.append(base)
            for position, scan in enumerate(pair["scan_ids"]):
                one = {**base, "horizon": 1, "scan_ids": [scan],
                       "scan_indices": [pair["scan_indices"][position]]}
                if scan in scans and scans[scan]["reference_id"] != pair["reference_id"]:
                    raise ValueError("physical scan belongs to multiple references")
                scans.setdefault(scan, one)
        for index, scan in enumerate(sorted(scans)):
            records.append({**scans[scan], "logical_unit_id": f"{role}:T1:{index:05d}"})
        counts[role] = {"references": len({r["reference_id"] for r in pairs}),
                        "T2": len(pairs), "unique_pairs": len({tuple(r["scan_ids"]) for r in pairs}),
                        "T1": len(scans)}
    for record in records:
        identity = json.dumps({"reference_id": record["reference_id"],
                               "scan_ids": record["scan_ids"]}, sort_keys=True)
        record["input_id"] = hashlib.sha256(identity.encode()).hexdigest()
    role_references = {role: {r["reference_id"] for r in records if r["role"] == role}
                       for role in ("TRAIN", "CAL", "SEL")}
    if (role_references["TRAIN"] & role_references["CAL"]
            or role_references["TRAIN"] & role_references["SEL"]
            or role_references["CAL"] & role_references["SEL"]):
        raise ValueError("physical reference overlap between roles")
    return {"schema": "short-module-population-v1", "counts": counts, "records": records,
            "train_selection": "SHA256(short-module-v1:reference), then SHA256(short-module-v1:scan1-scan2)",
            "excluded_reference_count": len(excluded_references),
            "new_head_train_exposure_overlap": 0,
            "parent_exposure": "Inherited R1 training and prior V2 development exposure; not untouched"}


def quality_labels(prediction: dict, target: dict, *, dataset_spec: Path,
                   min_region_size: int = 100) -> dict:
    """Use official pre-disambiguation pairwise records, never AP TP/FP matches.

    Candidate UUIDs are replaced only at the formatting boundary by input column
    indices. This retains lineage when the official matcher drops invalid masks.
    Ambiguous relations are excluded before score-dependent disambiguation.
    """
    from stmetrics.instances.matcher import InstanceMatcher

    class ColumnMatcher(InstanceMatcher):
        def _format_prediction_info(self, pred):
            return {i: {"label_id": pred["pred_classes"][i].to(self.device),
                        "conf": pred["pred_scores"][i].to(self.device),
                        "mask": pred["pred_masks"][:, i].to(self.device)}
                    for i in range(pred["pred_classes"].numel())}

    matcher = ColumnMatcher(dataset=str(dataset_spec), min_region_size=min_region_size,
                            timestep_key="temporal_stages")
    accumulator = OfficialMetricAccumulator(mode="strict_online", dataset_spec=dataset_spec,
                                           min_region_size=min_region_size)
    evaluator = accumulator._metric.heads[0]
    config = matcher.config
    params = (config.min_region_sizes[0].item(), config.distance_threshes[0].item(),
              config.distance_confs[0].item())
    thresholds = official_temporal_iou_thresholds(dataset_spec=dataset_spec,
                                                 min_region_size=min_region_size)
    gt2pred, pred2gt = matcher._assign_instances_for_scan(prediction, target)
    ambiguous_columns = {p["uuid"] for group in gt2pred.values() for gt in group.values()
                         if gt["ambiguous"] for p in gt["matched_pred"]}
    n = prediction["pred_scores"].numel()
    concat = torch.zeros(n)
    temporal = torch.zeros(n)
    valid = torch.zeros(n, dtype=torch.bool)
    geometry_valid = torch.zeros(n, dtype=torch.bool)
    resolved = torch.zeros(n, dtype=torch.bool)
    threshold_valid = torch.zeros(n, len(thresholds), dtype=torch.bool)
    status = ["EMPTY" if not prediction["pred_masks"][:, i].any() else "INVALID_PRED"
              for i in range(n)]
    gt_ids = [None] * n
    ignore = [None] * n
    for i, cls in enumerate(prediction["pred_classes"].tolist()):
        if cls not in matcher.id_to_label:
            status[i] = "UNKNOWN_CLASS"
    for rows in pred2gt.values():
        for pred in rows:
            i = pred["uuid"]
            if i in ambiguous_columns:
                status[i] = "AMBIGUOUS"
                continue
            if pred["vert_count"].sum() == 0:
                status[i] = "EMPTY"
                continue
            ignore[i] = evaluator._proportion_ignore(pred, *params)
            resolved[i] = True
            geometry_valid[i] = ignore[i] <= min(thresholds)
            compatible = [gt for gt in pred["matched_gt"]
                          if not gt["ambiguous"] and evaluator._valid_gt(gt, *params)]
            if not compatible:
                status[i] = "VALID_NEGATIVE"
            else:
                best = min(compatible, key=lambda gt: (-float(evaluator._select_overlap(gt)),
                                                   -float(gt["overlap"]),
                                                   int(gt["instance_id"])))
                temporal[i] = float(evaluator._select_overlap(best))
                concat[i] = float(best["overlap"])
                gt_ids[i] = int(best["instance_id"])
                status[i] = "MATCHED"
            # Official matching scores positive matches; unmatched predictions
            # participate only at thresholds admitting their ignored fraction.
            threshold_valid[i] = ((temporal[i] > torch.tensor(thresholds))
                                  | (ignore[i] <= torch.tensor(thresholds)))
            valid[i] = threshold_valid[i].any()
            if not valid[i]:
                status[i] = "IGNORE"
    return {"concat": concat, "temporal": temporal,
            "events": temporal[:, None] > torch.tensor(thresholds)[None, :],
            "valid": valid, "status": status, "gt_ids": gt_ids,
            "threshold_valid": threshold_valid, "geometry_valid": geometry_valid,
            "resolved": resolved, "all_thresholds_unscored": resolved & ~valid,
            "schema": "quality-threshold-validity-v2",
            "ignore_proportion": ignore, "thresholds": thresholds,
            "valid_gt_ids": sorted(int(gt["instance_id"])
                                   for rows in gt2pred.values() for gt in rows.values()
                                   if not gt["ambiguous"] and evaluator._valid_gt(gt, *params)),
            "ambiguity_metadata": ("AVAILABLE" if "ambiguities" in target
                                   else "AMBIGUITY_METADATA_UNAVAILABLE")}


def threshold_quality_loss(probabilities: Tensor, targets: Tensor, valid: Tensor) -> Tensor:
    """Mean BCE over eligible candidate/threshold entries, including negatives."""
    if probabilities.shape != targets.shape or valid.shape != targets.shape or valid.dtype != torch.bool:
        raise ValueError("quality probabilities, targets and boolean threshold mask must align")
    if not valid.any():
        return probabilities.sum() * 0
    return F.binary_cross_entropy(probabilities[valid], targets[valid])


def geometry_assignment(prediction: dict, target: dict, labels: dict, *,
                        candidate_keys: list[tuple]) -> Tensor:
    """Frozen class-compatible concat-IoU assignment, stable on lineage keys.

    Uses the existing maximum-cardinality/maximum-IoU solver at >= .10;
    one result entity ID is shared by all observed stages and all M arms.
    """
    n = prediction["pred_scores"].numel()
    if len(candidate_keys) != n or len(set(candidate_keys)) != n:
        raise ValueError("candidate keys must uniquely identify retained columns")
    candidates = sorted(torch.where(labels["geometry_valid"])[0].tolist(),
                        key=lambda i: candidate_keys[i])
    valid_ids = set(labels["valid_gt_ids"])
    gt_indices = sorted((i for i, v in enumerate(target["ids"].tolist()) if v in valid_ids),
                        key=lambda i: int(target["ids"][i]))
    assigned = torch.full((n,), -1, dtype=torch.long)
    if not candidates or not gt_indices:
        return assigned
    g = target["masks"][gt_indices].float()
    p = prediction["pred_masks"][:, candidates].float()
    intersection = g @ p
    union = g.sum(1)[:, None] + p.sum(0)[None, :] - intersection
    matrix = intersection / union.clamp_min(1)
    pairs = global_hungarian_match(matrix, gt_classes=target["labels"][gt_indices],
                                   pred_classes=prediction["pred_classes"][candidates],
                                   threshold=.10)
    for gt_index, candidate_index in pairs:
        assigned[candidates[candidate_index]] = target["ids"][gt_indices[gt_index]]
    return assigned


def low_segment_targets(*, low_point2segment: Tensor, voxel_inverse: Tensor,
                        gt_mask: Tensor, semantic_labels: Tensor,
                        segment_count: int) -> tuple[Tensor, Tensor]:
    """Count reliably labeled full vertices in the LOW-resolution output segments.

    The caller provides actual preprocessed point semantic labels (255 unknown).
    Background/stuff and other instances remain known negative vertices.
    """
    full_to_low = low_point2segment[voxel_inverse].long()
    if gt_mask.shape != full_to_low.shape or semantic_labels.shape != full_to_low.shape:
        raise ValueError("full vertex labels and inverse map must align")
    if full_to_low.numel() and (full_to_low.min() < 0 or full_to_low.max() >= segment_count):
        raise ValueError("low segment namespace differs from output logits")
    known = (semantic_labels >= 0) & (semantic_labels != 255)
    weights = torch.bincount(full_to_low[known], minlength=segment_count).float()
    positives = torch.bincount(full_to_low[known & gt_mask.bool()], minlength=segment_count).float()
    return positives / weights.clamp_min(1), weights


def geometry_loss(module: str | Module, *, logits: Tensor, delta: Tensor,
                  targets: Tensor, weights: Tensor, segment_stages: Tensor,
                  matched: Tensor, usable: Tensor,
                  sample_indices: list[Tensor], details: dict | None = None) -> Tensor:
    """Weighted BCE+Dice, stage-equal mean then ALL sampled candidate slots.

    sample_indices holds one shared valid segment subset per stage (<=2048),
    supplied by the common sampling plan. Unmatched M1/M2 retain zero segment
    loss but stay in the candidate denominator. Every finite native input
    residual regularizes, including candidates without usable GT supervision.
    """
    module = Module(module)
    if module not in (Module.M0, Module.M1, Module.M2):
        raise ValueError("geometry supervision requires a mask module")
    if logits.shape != targets.shape or logits.shape != delta.shape or logits.shape[1] == 0:
        raise ValueError("geometry matrices must align with sampled candidate slots")
    losses, bce_terms, dice_terms = [], [], []
    for candidate in range(logits.shape[1]):
        stages, stage_bce, stage_dice = [], [], []
        if usable[candidate] and (matched[candidate] or module is Module.M0):
            for stage, indices in enumerate(sample_indices):
                if indices.numel() > 2048 or not torch.all(segment_stages[indices] == stage):
                    raise ValueError("shared segment sample has wrong stage or size")
                indices = indices[weights[indices] > 0]
                if not indices.numel():
                    continue
                w = weights[indices]
                y = targets[indices, candidate] if matched[candidate] else torch.zeros_like(w)
                z = logits[indices, candidate]
                bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * w).sum() / w.sum()
                p = z.sigmoid()
                dice = 1 - (2 * (w * p * y).sum() + 1) / ((w * p).sum() + (w * y).sum() + 1)
                stages.append(bce + dice)
                stage_bce.append(bce.detach())
                stage_dice.append(dice.detach())
        losses.append(torch.stack(stages).mean() if stages else logits[:, candidate].sum() * 0)
        bce_terms.append(torch.stack(stage_bce).mean() if stage_bce else logits.new_zeros(()))
        dice_terms.append(torch.stack(stage_dice).mean() if stage_dice else logits.new_zeros(()))
    # `usable` describes mask-supervision eligibility, not feature validity.
    # Native feature/logit validation already guarantees usable finite inputs.
    # Preserving a residual toward zero needs no GT and includes ignored slots.
    regularization = delta.square().mean()
    if details is not None:
        details.update(bce=torch.stack(bce_terms).mean().item(), dice=torch.stack(dice_terms).mean().item(),
                       regularization=.01 * regularization.detach().item())
    return torch.stack(losses).mean() + .01 * regularization
