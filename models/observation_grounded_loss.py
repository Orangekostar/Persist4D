"""Training-only constrained assignment; no ground truth enters model state."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from models.criterion import SetCriterion
from models.task_memory_supervision import ambiguous_identity_ids


@dataclass(frozen=True)
class AssignmentConstraints:
    forced: tuple = ()
    ignored: tuple = ()

    def validate(self, query_count, target_count):
        queries = [int(q) for q, _ in self.forced]
        targets = [int(t) for _, t in self.forced]
        if (len(set(queries)) != len(queries) or len(set(targets)) != len(targets)
                or len(set(self.ignored)) != len(self.ignored)
                or set(queries) & set(self.ignored)
                or any(q < 0 or q >= query_count for q in (*queries, *self.ignored))
                or any(t < 0 or t >= target_count for t in targets)):
            raise ValueError("OGR forced/ignored assignment is not a valid one-to-one partition")


@torch.no_grad()
def constrained_assignment(matcher, output, targets, mask_type, constraints):
    if output['pred_logits'].shape[0] != 1 or len(targets) != 1:
        raise ValueError("OGR constrained task assignment uses B=1")
    query_count, target_count = output['pred_logits'].shape[1], len(targets[0]['labels'])
    constraints.validate(query_count, target_count)
    if not constraints.forced and not constraints.ignored:
        return matcher(output, targets, mask_type)
    forced_queries, forced_targets = {q for q, _ in constraints.forced}, {t for _, t in constraints.forced}
    query_rows = torch.tensor([q for q in range(query_count) if q not in forced_queries
                               and q not in constraints.ignored], dtype=torch.long)
    target_rows = torch.tensor([t for t in range(target_count) if t not in forced_targets], dtype=torch.long)
    pairs = list(constraints.forced)
    if query_rows.numel() and target_rows.numel():
        query_device = query_rows.to(output['pred_logits'].device)
        reduced = dict(pred_logits=output['pred_logits'][:, query_device],
                       pred_masks=[output['pred_masks'][0][:, query_device]])
        if 'pred_changes' in output:
            changes = output['pred_changes']
            reduced['pred_changes'] = changes[:, query_device] if changes is not None else None
        objects = {'labels', 'ids', 'changes', 'masks', 'segment_mask', mask_type}
        target = {key: (value[target_rows.to(value.device)] if key in objects else value)
                  for key, value in targets[0].items()}
        rows, columns = matcher(reduced, [target], mask_type)[0]
        pairs.extend(zip(query_rows[rows].tolist(), target_rows[columns].tolist()))
    pairs.sort()
    return [(torch.tensor([q for q, _ in pairs], dtype=torch.long),
             torch.tensor([t for _, t in pairs], dtype=torch.long))]


class _AssignmentRecorder(nn.Module):
    def __init__(self, matcher, constraints, assignments):
        super().__init__()
        self.matcher, self.constraints, self.assignments = matcher, constraints, assignments

    def forward(self, output, targets, mask_type):
        result = constrained_assignment(self.matcher, output, targets, mask_type, self.constraints)
        self.assignments.append(result)
        return result


class ConstrainedTaskCriterion(SetCriterion):
    @classmethod
    def from_parent(cls, parent):
        """Preserve the exact parent parameter/buffer namespaces and objects."""
        result = cls.__new__(cls)
        result.__dict__ = parent.__dict__.copy()
        result._modules = parent._modules.copy()
        result.assignments = []
        result._ogr_constraints = AssignmentConstraints()
        return result

    def forward(self, outputs, targets, mask_type, *, constraints=None):
        constraints = AssignmentConstraints() if constraints is None else constraints
        constraints.validate(outputs['pred_logits'].shape[1], len(targets[0]['labels']))
        original = self.matcher
        self.assignments, self._ogr_constraints = [], constraints
        self.matcher = _AssignmentRecorder(original, constraints, self.assignments)
        try:
            # Reuse all original task, aux and contrastive keys/scales and RNG.
            return super().forward(outputs, targets, mask_type)
        finally:
            self.matcher = original
            self._ogr_constraints = AssignmentConstraints()

    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        if not self._ogr_constraints.ignored:
            return super().loss_labels(outputs, targets, indices, num_masks, mask_type)
        logits = outputs['pred_logits'].float()
        classes = torch.full(logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device)
        for batch, (queries, target_rows) in enumerate(indices):
            classes[batch, queries.to(logits.device)] = targets[batch]['labels'][target_rows.to(targets[batch]['labels'].device)]
        classes[0, list(self._ogr_constraints.ignored)] = 253
        loss = (F.cross_entropy(logits.transpose(1, 2), classes, self.empty_weight, ignore_index=253)
                if (classes != 253).any() else logits.sum() * 0.)
        return {'loss_ce': loss}


def grouped_identity_loss(logits, query_identities, history_identities, *, valid, ignored=()):
    """Unknown history stays in forward P but is absent from this normalization."""
    if logits.shape != (len(query_identities), len(history_identities) + 1):
        raise ValueError("OGR identity logits/ledger columns differ")
    known = [column for column, identity in enumerate(history_identities) if identity is not None]
    denominator_columns = [*known, len(history_identities)]
    coverage = dict(positive=0, null=0, unknown=0, duplicate_groups=0,
                    known_history=len(known), effective_history=len(history_identities))
    rows = []
    for query, identity in enumerate(query_identities):
        if query in ignored or identity is None or not bool(valid[query]):
            coverage['unknown'] += 1
            continue
        positive = [column for column in known if history_identities[column] == identity]
        if positive:
            coverage['positive'] += 1
            coverage['duplicate_groups'] += int(len(positive) > 1)
        elif len(known) == len(history_identities):
            positive = [len(history_identities)]
            coverage['null'] += 1
        else:
            coverage['unknown'] += 1
            continue
        rows.append(torch.logsumexp(logits[query, denominator_columns], 0)
                    - torch.logsumexp(logits[query, positive], 0))
    return (torch.stack(rows).mean() if rows else logits.sum() * 0.), coverage


def supervision_context(output, stage, old_state, ledger):
    """Resolve reliable current targets and a canonical source on training side."""
    meta = stage.stage_meta[0]
    full_target = stage.batch[0].target_full[0]
    target = stage.batch[1][0]
    ambiguity = set(ambiguous_identity_ids(full_target.get('ambiguities')))
    current = meta.local_stage_ids.cpu() == len(meta.scan_ids_in_window) - 1
    visible = full_target['masks'].cpu()[:, current].any(1)
    reliable = {(meta.reference_id, int(identity)) for identity, support in
                zip(full_target['ids'].tolist(), visible.tolist(), strict=True)
                if support and int(identity) not in ambiguity}
    target_identities = [(meta.reference_id, int(identity)) if (meta.reference_id, int(identity)) in reliable
                         else None for identity in target['ids'].tolist()]
    selected = output['mpr_selected_slots'][0].tolist()
    sources = {}
    for query, slot in enumerate(selected):
        identity = ledger.lookup(old_state.logical_ids[0, slot], old_state.generations[0, slot])
        if identity in reliable:
            sources.setdefault(identity, []).append((query, slot))
    forced, ignored = [], []
    for target_row, identity in enumerate(target_identities):
        if identity is None or identity not in sources:
            continue
        rows = sorted(sources[identity], key=lambda row: (
            -float(old_state.confidence[0, row[1]]), -int(old_state.last_seen[0, row[1]]),
            int(old_state.logical_ids[0, row[1]]), int(old_state.generations[0, row[1]]), row[1]))
        forced.append((rows[0][0], target_row))
        ignored.extend(query for query, _ in rows[1:])
    return dict(constraints=AssignmentConstraints(tuple(sorted(forced)), tuple(sorted(ignored))),
                target_identities=target_identities, reliable=reliable, ambiguous=ambiguity,
                reference_id=meta.reference_id)


def identity_loss_for_assignment(regional, assignment, context, old_state, ledger):
    evidence = regional[0]
    query_identities = [None] * len(evidence['values'])
    for query, target in zip(*[value.tolist() for value in assignment[0]], strict=True):
        query_identities[query] = context['target_identities'][target]
    history = []
    for slot in evidence['slots'].tolist():
        identity = ledger.lookup(old_state.logical_ids[0, slot], old_state.generations[0, slot])
        history.append(identity if identity is not None and identity[0] == context['reference_id']
                       and identity[1] not in context['ambiguous'] else None)
    return grouped_identity_loss(evidence['logits'], query_identities, history,
        valid=evidence['valid'], ignored=context['constraints'].ignored)
