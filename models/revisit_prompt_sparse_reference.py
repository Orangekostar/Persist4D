"""Inference-only Native SubM convolution with ordered FP32 accumulation.

This diagnostic keeps spconv's KRSC weights and generated neighbor pairs. Each
offset contributes once per destination, so no floating point atomic reduction
is needed. It deliberately supports only the frozen encoder's inference path.
"""
from contextlib import contextmanager

import torch


def ordered_subm_convolution(features, filters, indice_pairs, indice_pair_num,
                             num_activate_out, inverse=False, subm=False,
                             algo=None, timer=None, bias=None, act_alpha=0.0,
                             act_beta=0.0, act_type=None):
    from spconv.core import ConvAlgo

    if torch.is_grad_enabled():
        raise ValueError("Sparse reference supports inference only")
    if not subm or inverse or algo not in (None, ConvAlgo.Native):
        raise ValueError("Sparse reference requires forward Native SubM convolution")
    if act_type is not None and int(act_type) != 0:
        raise ValueError("Sparse reference does not support fused activation")
    if features.dtype != torch.float32 or filters.dtype != torch.float32:
        raise ValueError("Sparse reference requires FP32 features and KRSC weights")
    if features.ndim != 2 or filters.ndim < 3:
        raise ValueError("Expected matrix features and KRSC weights")
    if features.device != filters.device or features.shape[1] != filters.shape[-1]:
        raise ValueError("Feature and weight channels/devices must match")
    if num_activate_out != features.shape[0]:
        raise ValueError("SubM convolution must preserve active output count")
    if filters.shape[0] <= 0 or filters.shape[-1] <= 0:
        raise ValueError("Weight channels must be positive")
    weights = filters.reshape(filters.shape[0], -1, filters.shape[-1])
    volume = weights.shape[1]
    if volume <= 0 or volume % 2 != 1:
        raise ValueError("SubM kernel volume must be positive and odd")
    integral = (torch.int32, torch.int64)
    if (indice_pairs.ndim != 3 or indice_pairs.shape[:2] != (2, volume)
            or indice_pairs.dtype not in integral
            or indice_pairs.device != features.device):
        raise ValueError("Invalid Native neighbor pair tensor")
    if (indice_pair_num.ndim != 1 or len(indice_pair_num) != volume
            or indice_pair_num.dtype not in integral):
        raise ValueError("Invalid Native neighbor counts")
    counts = indice_pair_num.cpu().tolist()
    if any(n < 0 or n > indice_pairs.shape[2] for n in counts):
        raise ValueError("Neighbor count exceeds pair storage")
    if bias is not None and (bias.shape != (weights.shape[0],)
                             or bias.dtype != features.dtype
                             or bias.device != features.device):
        raise ValueError("Bias must match output channels, dtype and device")
    if features.is_cuda and torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("Sparse reference requires TF32 disabled")

    center = volume // 2
    output = torch.mm(features, weights[:, center].T)
    for offset in range(volume):
        if offset == center:
            continue
        count = counts[volume - offset - 1] if offset > center else counts[offset]
        if count == 0:
            continue
        source, destination = indice_pairs[:, offset, :count].long().unbind(0)
        if (torch.any(source < 0) or torch.any(source >= features.shape[0])
                or torch.any(destination < 0) or torch.any(destination >= num_activate_out)):
            raise ValueError("Valid neighbor indices must be in bounds")
        if destination.unique().numel() != count:
            raise ValueError("SubM offset must have unique destination indices")
        contribution = torch.mm(features.index_select(0, source), weights[:, offset].T)
        output.index_copy_(0, destination, output.index_select(0, destination) + contribution)
    if bias is not None:
        output.add_(bias)
    return output


@contextmanager
def evaluation_sparse_reference():
    """Temporarily replace the inference operation in an isolated child process."""
    from spconv.pytorch import ops

    original = ops.indice_conv
    ops.indice_conv = ordered_subm_convolution
    try:
        yield
    finally:
        ops.indice_conv = original
