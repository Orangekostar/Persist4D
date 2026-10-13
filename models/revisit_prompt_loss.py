"""Training-only labels for retrieved evidence; UNKNOWN is never a negative."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def retrieval_loss(scores, positives):
    if scores.shape != positives.shape or positives.dtype != torch.bool:
        raise ValueError("retrieval positive mask must match global score rows")
    valid = positives.any(-1)
    if not valid.any():
        return scores.sum() * 0.
    all_scores = torch.logsumexp(scores[valid], dim=-1)
    positive_scores = torch.logsumexp(scores[valid].masked_fill(~positives[valid], -torch.inf), dim=-1)
    return (all_scores - positive_scores).mean()


def support_loss(probabilities, labels):
    if probabilities.shape != labels.shape:
        raise ValueError("support label rows do not match history prompts")
    known = labels >= 0
    if not known.any():
        return probabilities.sum() * 0.
    if not ((labels[known] == 0) | (labels[known] == 1)).all():
        raise ValueError("support labels must be positive, negative or UNKNOWN(-1)")
    return F.binary_cross_entropy(probabilities[known], labels[known].to(probabilities.dtype))


class TrainingLabelLedger:
    """Separate from prediction state; identities are bound once after a birth."""
    def __init__(self):
        self.identities = {}

    def lookup(self, logical_id, generation):
        return self.identities.get((int(logical_id), int(generation)))

    def bind(self, logical_id, generation, reference_uuid, instance_id):
        key = (int(logical_id), int(generation))
        identity = (reference_uuid, int(instance_id))
        if key in self.identities and self.identities[key] != identity:
            raise ValueError("an existing reliable history label cannot be relabeled")
        self.identities[key] = identity


def evidence_labels(point2segment, point_ids, current_segments, *, historical_ids,
                    anchors, reliable_ids):
    """Full-vertex purity for positives; ambiguous majority ties remain UNKNOWN."""
    p2s, ids = point2segment.cpu().long(), point_ids.cpu().long()
    current, anchors = current_segments.cpu().long(), anchors.cpu().long()
    if p2s.shape != ids.shape or anchors.numel() != len(historical_ids):
        raise ValueError("training evidence vertices/anchors do not align")
    positive = torch.zeros((len(historical_ids), current.numel()), dtype=torch.bool)
    support = torch.full((len(historical_ids),), -1, dtype=torch.long)
    if not ids.numel():
        return positive, support
    unique, inverse = ids.unique(sorted=True, return_inverse=True)
    count = int(p2s.max())+1
    histogram = torch.bincount(p2s*unique.numel()+inverse,
                               minlength=count*unique.numel()).reshape(count, -1)
    totals = histogram.sum(1)
    maximum, majority = histogram.max(1)
    unique_majority = (histogram == maximum[:, None]).sum(1) == 1
    lookup = {int(identity): i for i, identity in enumerate(unique.tolist())}
    for row, history in enumerate(historical_ids):
        if history is None:
            continue
        if history in reliable_ids and history in lookup:
            positive[row] = histogram[current, lookup[history]]*2 >= totals[current]
            positive[row] &= totals[current] > 0
        anchor = int(anchors[row])
        if not 0 <= anchor < count or anchor not in current.tolist():
            raise ValueError("training anchor is outside the predicted current segments")
        identity = int(unique[majority[anchor]])
        if (identity in reliable_ids and unique_majority[anchor]
                and maximum[anchor]*2 >= totals[anchor] and totals[anchor] > 0):
            support[row] = int(identity == history)
    return positive, support


def auxiliary_losses(output, stage, old_state, ledger, *, variant):
    from models.task_memory_supervision import ambiguous_identity_ids

    zero = output["pred_logits"].sum()*0.
    diagnostics = dict(positive=0, negative=0, unknown=0, retrieval_rows=0,
                       no_valid_positive_segment=0, absent_history=0, unknown_history=0,
                       ambiguous_history=0)
    if variant not in {"M5", "M6"}:
        return zero, zero, diagnostics
    meta = stage.stage_meta[0]
    data = stage.batch[0]
    target = data.target_full[0]
    latest = len(meta.scan_ids_in_window)-1
    current_vertices = meta.local_stage_ids.cpu() == latest
    current_visible = target["masks"][:, current_vertices].any(1)
    ambiguous = ambiguous_identity_ids(target.get("ambiguities"))
    reliable = set(target["ids"][current_visible].tolist())-set(ambiguous)
    vertex_ids = stage.loss_labels[current_vertices, 1]
    p2s = meta.full_resolution_point2segment.cpu()[current_vertices]
    ret_rows, support_rows, labels_rows = [], [], []
    for retrieved in output["mpr_retrieval"]:
        history = []
        for slot in retrieved["source_slots"].tolist():
            identity = ledger.lookup(old_state.logical_ids[0, slot], old_state.generations[0, slot])
            history.append(identity[1] if identity is not None and identity[0] == meta.reference_id else None)
        positives, labels = evidence_labels(p2s, vertex_ids, retrieved["current_segment_indices"],
            historical_ids=history, anchors=retrieved["anchor_segment_indices"], reliable_ids=reliable)
        valid_rows = positives.any(1)
        for identity, valid in zip(history, valid_rows.tolist(), strict=True):
            if identity is None:
                diagnostics["unknown_history"] += 1
            elif identity in ambiguous:
                diagnostics["ambiguous_history"] += 1
            elif identity not in reliable:
                diagnostics["absent_history"] += 1
            elif not valid:
                diagnostics["no_valid_positive_segment"] += 1
        scores = retrieved["logits"]
        if valid_rows.any():
            valid = valid_rows.to(scores.device)
            selected = positives.to(scores.device)[valid]
            ret_rows.append(torch.logsumexp(scores[valid], -1)
                            - torch.logsumexp(scores[valid].masked_fill(~selected, -torch.inf), -1))
        else:
            zero = zero + scores.sum()*0.
        diagnostics["retrieval_rows"] += int(valid_rows.sum())
        if variant == "M6":
            probabilities = retrieved["support_probabilities"]
            support_rows.append(probabilities)
            labels_rows.append(labels.to(probabilities.device))
            diagnostics["positive"] += int((labels == 1).sum())
            diagnostics["negative"] += int((labels == 0).sum())
            diagnostics["unknown"] += int((labels == -1).sum())
    ret = torch.cat(ret_rows).mean() if ret_rows else zero
    sup = support_loss(torch.cat(support_rows), torch.cat(labels_rows)) if support_rows else zero
    return ret, sup, diagnostics


@torch.no_grad()
def bind_training_births(ledger, commit, output, stage):
    from scipy.optimize import linear_sum_assignment
    from models.task_memory_supervision import ambiguous_identity_ids

    meta = stage.stage_meta[0]
    target = stage.batch[0].target_full[0]
    current = meta.local_stage_ids.cpu() == len(meta.scan_ids_in_window)-1
    ambiguous = ambiguous_identity_ids(target.get("ambiguities"))
    visible = target["masks"][:, current].any(1)
    valid = visible & torch.tensor([int(identity) not in ambiguous for identity in target["ids"]], dtype=torch.bool)
    targets = valid.nonzero(as_tuple=True)[0]
    births = commit.births[0].nonzero(as_tuple=True)[0].cpu()
    diagnostics = dict(accepted_births=births.numel(), bound_births=0,
                       unmatched_births=births.numel(), ambiguous_gt_excluded=int((visible & ~valid).sum()))
    if not births.numel() or not targets.numel():
        return diagnostics
    segment_ids = meta.full_resolution_point2segment.cpu()[current]
    predicted = (output["pred_masks"][0].detach().cpu()[segment_ids][:, births] >= 0).T
    truth = target["masks"][targets][:, current].bool()
    intersection = predicted.float() @ truth.float().T
    union = predicted.sum(1)[:, None]+truth.sum(1)[None, :]-intersection
    iou = intersection/union.clamp_min(1)
    predicted_classes = output["pred_logits"][0, births.to(output["pred_logits"].device), :-1].argmax(-1).cpu()
    eligible = (predicted_classes[:, None] == target["labels"][targets][None, :]) & (iou >= .5)
    cost = torch.where(eligible, -iou, torch.ones_like(iou))
    row_indices, column_indices = linear_sum_assignment(cost.numpy())
    for row, column in zip(row_indices, column_indices, strict=True):
        if not bool(eligible[row, column]):
            continue
        query, gt_row = int(births[row]), int(targets[column])
        ledger.bind(commit.query_to_logical_id[0, query], commit.query_to_generation[0, query],
                    meta.reference_id, target["ids"][gt_row])
        diagnostics["bound_births"] += 1
    diagnostics["unmatched_births"] -= diagnostics["bound_births"]
    return diagnostics
