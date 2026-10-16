"""Process-local acceleration for serial, offline official-state statistics.

Training, prediction, sufficient-state validation, AP integration, and bootstrap
resampling keep their original implementations. Only the CPU float32 threshold
loop is batched, preserving the official reciprocal-then-multiply arithmetic.
"""

from contextlib import contextmanager

import torch
from stmetrics.instances import ap_math, evaluator


_ORIGINAL_PRECISION_RECALL = ap_math.compute_precision_recall


def precision_recall_cpu(y_true, y_score, hard_false_negatives, device):
    """Match the original scalar curve exactly for the offline float32 case."""
    tensors = (y_true, y_score)
    supported = (
        str(device) in ('cpu', 'cpu:0')
        and torch.get_default_dtype() == torch.float32
        and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.float32 and value.ndim == 1
                and not value.requires_grad for value in tensors)
        and (isinstance(hard_false_negatives, (int, float))
             or isinstance(hard_false_negatives, torch.Tensor)
             and hard_false_negatives.device.type == 'cpu'
             and hard_false_negatives.dtype == torch.float32
             and hard_false_negatives.ndim == 0
             and not hard_false_negatives.requires_grad)
    )
    if not supported:
        return _ORIGINAL_PRECISION_RECALL(y_true, y_score, hard_false_negatives, device)
    if len(y_true) == 0:
        return torch.tensor([]), torch.tensor([])

    order = torch.argsort(y_score)
    sorted_scores = y_score[order]
    cumulative = torch.cumsum(y_true[order], dim=0)
    indices = torch.searchsorted(sorted_scores, torch.unique_consecutive(sorted_scores))
    count = len(indices) + 1
    precision = torch.zeros(count, device=device)
    recall = torch.zeros(count, device=device)
    true_count = cumulative[-1]
    cumulative = torch.cat((cumulative, torch.tensor([0], device=device)))
    before = cumulative[indices - 1]
    tp = true_count - before
    fp = len(sorted_scores) - indices - tp
    fn = before + hard_false_negatives
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    # float(tp) / denominator in the official loop invokes Tensor.__rdiv__,
    # i.e. denominator.reciprocal() * float(tp), with float32 rounding twice.
    precision[:-1] = torch.where(precision_denominator > 0,
                                precision_denominator.reciprocal() * tp, 0.)
    recall[:-1] = torch.where(recall_denominator > 0,
                             recall_denominator.reciprocal() * tp, 0.)
    precision[-1] = 1.
    recall[-1] = 0.
    return precision, recall


@contextmanager
def accelerated_precision_recall():
    """Scope both official import bindings to one serial CPU statistics call."""
    original_math = ap_math.compute_precision_recall
    original_evaluator = evaluator.compute_precision_recall
    ap_math.compute_precision_recall = precision_recall_cpu
    evaluator.compute_precision_recall = precision_recall_cpu
    try:
        yield
    finally:
        ap_math.compute_precision_recall = original_math
        evaluator.compute_precision_recall = original_evaluator
