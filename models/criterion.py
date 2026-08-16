# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified for Mask3D
"""
MaskFormer criterion.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from models.misc import (
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# Memory-efficient contrastive loss helpers
def _sanitize_features(x: torch.Tensor) -> torch.Tensor:
    """Replace NaNs/Infs and re-normalize if needed."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _normalize_features(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize features (matching old implementation)."""
    return F.normalize(x, dim=-1, eps=eps)


def _instances_points(instance_masks: torch.Tensor):
    """Convert (M, N) instance_masks to a list of point-index tensors per instance."""
    I = instance_masks.bool()
    inst_points = []
    for k in range(I.shape[0]):
        inst_points.append(torch.nonzero(I[k], as_tuple=True)[0])
    return inst_points


def _build_chunk_pairs(
    inst_points,
    start: int,
    end: int,
    N: int,
    include_self: bool,
    temporal_stages,
    stage_weight_same: float,
    stage_weight_cross: float,
    device: torch.device,
    dtype: torch.dtype,
    assume_single_label: bool,
):
    """
    Build sparse positive pair indices for anchors in [start, end):
    Returns:
      rows: (P,) int64 relative anchor indices in [0, B)
      cols: (P,) int64 absolute positive point indices in [0, N)
      weights: (P,) float weights per pair (stage-aware if provided, else 1)
      row_has_pos: (B,) bool mask of anchors that have at least one positive
    """
    B = end - start
    rows_list = []
    cols_list = []
    weights_list = []
    # Optional stage weights
    use_stage = temporal_stages is not None
    if use_stage:
        stages = temporal_stages  # (N,)
        stage_same_t = torch.as_tensor(stage_weight_same, device=device, dtype=dtype)
        stage_cross_t = torch.as_tensor(stage_weight_cross, device=device, dtype=dtype)
        anchor_stage_chunk = stages[start:end]  # (B,)

    # Build pairs per instance
    for pts in inst_points:
        if pts.numel() == 0:
            continue
        # Anchors from this instance that lie in the chunk
        mask = (pts >= start) & (pts < end)
        anchors_abs = pts[mask]
        if anchors_abs.numel() == 0:
            continue

        anchors_rel = anchors_abs - start  # (A,)
        pos_abs = pts  # (P_inst,)

        # Grid of (anchor_rel, pos_abs)
        rows_add = anchors_rel.repeat_interleave(pos_abs.numel())  # (A * P_inst,)
        cols_add = pos_abs.repeat(anchors_rel.numel())             # (A * P_inst,)

        if not include_self:
            # Drop self-pairs
            self_mask = cols_add != (rows_add + start)
            rows_add = rows_add[self_mask]
            cols_add = cols_add[self_mask]

        if rows_add.numel() == 0:
            continue

        if use_stage:
            a_stage = anchor_stage_chunk[anchors_rel]                       # (A,)
            p_stage = stages[pos_abs]                                       # (P_inst,)
            a_stage_rep = a_stage.repeat_interleave(p_stage.numel())        # (A * P_inst,)
            p_stage_rep = p_stage.repeat(anchors_rel.numel())               # (A * P_inst,)
            w_add = torch.where(p_stage_rep == a_stage_rep, stage_same_t, stage_cross_t)
            if not include_self and self_mask.numel() > 0:
                w_add = w_add[self_mask]
        else:
            w_add = torch.ones(rows_add.numel(), device=device, dtype=dtype)

        rows_list.append(rows_add)
        cols_list.append(cols_add)
        weights_list.append(w_add)

    if len(rows_list) == 0:
        rows = torch.empty(0, dtype=torch.long, device=device)
        cols = torch.empty(0, dtype=torch.long, device=device)
        weights = torch.empty(0, dtype=dtype, device=device)
        row_has_pos = torch.zeros(B, dtype=torch.bool, device=device)
        return rows, cols, weights, row_has_pos

    rows = torch.cat(rows_list, dim=0)
    cols = torch.cat(cols_list, dim=0)
    weights = torch.cat(weights_list, dim=0)

    # If points can belong to multiple instances, the same (row, col) may appear multiple times.
    # Deduplicate to implement union-of-positives exactly.
    if not assume_single_label:
        # Sort by (row, col) and keep first occurrence
        key = rows.to(torch.int64) * N + cols.to(torch.int64)
        order = torch.argsort(key)
        key_sorted = key[order]
        keep = torch.ones_like(key_sorted, dtype=torch.bool)
        keep[1:] = key_sorted[1:] != key_sorted[:-1]
        idx = order[keep]
        rows = rows[idx]
        cols = cols[idx]
        weights = weights[idx]

    # Anchors with at least one positive
    row_has_pos = torch.zeros(end - start, dtype=torch.bool, device=device)
    row_has_pos.index_fill_(0, rows, True)

    return rows, cols, weights, row_has_pos


def _build_instance_structures(instance_masks: torch.Tensor):
    """
    Build lightweight structures to query positives per anchor without dense N×N masks.

    Args:
      instance_masks: Bool or 0/1 tensor of shape (num_instances, num_points)

    Returns:
      inst_points: list of length M, where inst_points[k] is a 1D LongTensor of point indices in instance k.
      inst_ids_per_point: list of length N, where inst_ids_per_point[j] is a Python list of instance ids that contain point j.
      device: device of the instance_masks tensor
    """
    I_bool = instance_masks.bool()
    M, N = I_bool.shape
    device = I_bool.device

    # Points per instance
    inst_points = []
    for k in range(M):
        pts_k = torch.nonzero(I_bool[k], as_tuple=True)[0]
        inst_points.append(pts_k)

    # Instance ids per point (sparse membership)
    inst_ids_per_point = [[] for _ in range(N)]
    nz_inst, nz_pts = torch.nonzero(I_bool, as_tuple=True)
    # Iterate through nonzeros; each (k, j) means point j is in instance k
    for k, j in zip(nz_inst.tolist(), nz_pts.tolist()):
        inst_ids_per_point[j].append(k)

    return inst_points, inst_ids_per_point, device


_INFO_NCE_SIMILARITY_BLOCK_BYTES = 8 * 1024**2


def _info_nce_accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _merge_block_logsumexp(
    accumulated: torch.Tensor,
    accumulated_valid: torch.Tensor,
    block: torch.Tensor,
    block_valid: torch.Tensor,
):
    """Merge row-wise log-sums while keeping empty blocks out of the graph."""
    zeros = torch.zeros_like(accumulated)
    safe_accumulated = torch.where(accumulated_valid, accumulated, zeros)
    safe_block = torch.where(block_valid, block, zeros)
    both = accumulated_valid & block_valid
    merged_both = torch.logaddexp(safe_accumulated, safe_block)
    merged = torch.where(
        both,
        merged_both,
        torch.where(block_valid, block, accumulated),
    )
    return merged, accumulated_valid | block_valid


def _streaming_info_nce_chunk(
    anchor_features: torch.Tensor,
    all_features: torch.Tensor,
    positive_rows: torch.Tensor,
    positive_cols: torch.Tensor,
    positive_weights: torch.Tensor,
    logit_scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    anchor_start: int,
    candidate_chunk_size: int,
    include_self: bool,
    norm_type: str,
    clamp_limits,
):
    """Compute exact numerator and denominator log-sums in candidate blocks."""
    device = anchor_features.device
    dtype = anchor_features.dtype
    accumulation_dtype = _info_nce_accumulation_dtype(dtype)
    num_anchors = anchor_features.shape[0]
    num_candidates = all_features.shape[0]
    denominator_log = torch.zeros(
        num_anchors,
        device=device,
        dtype=accumulation_dtype,
    )
    numerator_log = torch.zeros_like(denominator_log)
    denominator_valid = torch.zeros(num_anchors, device=device, dtype=torch.bool)
    numerator_valid = torch.zeros_like(denominator_valid)

    for candidate_start in range(0, num_candidates, candidate_chunk_size):
        candidate_end = min(candidate_start + candidate_chunk_size, num_candidates)
        similarities = (
            anchor_features @ all_features[candidate_start:candidate_end].T
        )

        if norm_type == "log_odds" and clamp_limits is not None:
            similarities = torch.clamp(similarities, *clamp_limits)
            similarities = 2 * torch.atanh(similarities)
        elif norm_type == "clamp":
            similarities = ((1 + similarities) / 2).clamp(min=0, max=1)

        similarities = similarities.to(accumulation_dtype)
        similarities = (
            similarities * logit_scale.to(accumulation_dtype)
            + bias.to(accumulation_dtype)
        )

        if not include_self:
            overlap_start = max(anchor_start, candidate_start)
            overlap_end = min(
                anchor_start + num_anchors,
                candidate_end,
            )
            if overlap_start < overlap_end:
                self_indices = torch.arange(
                    overlap_start,
                    overlap_end,
                    device=device,
                )
                similarities[
                    self_indices - anchor_start,
                    self_indices - candidate_start,
                ] = -torch.inf

        block_max = similarities.max(dim=1).values
        block_valid = torch.isfinite(block_max)
        safe_block_max = torch.where(
            block_valid,
            block_max,
            torch.zeros_like(block_max),
        )
        exp_values = torch.exp(similarities - safe_block_max.unsqueeze(1))
        denominator_sum = exp_values.sum(dim=1)
        safe_denominator_sum = torch.where(
            block_valid,
            denominator_sum,
            torch.ones_like(denominator_sum),
        )
        block_denominator_log = (
            torch.log(safe_denominator_sum) + safe_block_max
        )
        denominator_log, denominator_valid = _merge_block_logsumexp(
            denominator_log,
            denominator_valid,
            block_denominator_log,
            block_valid,
        )

        in_block = (positive_cols >= candidate_start) & (
            positive_cols < candidate_end
        )
        block_rows = positive_rows[in_block]
        block_cols = positive_cols[in_block] - candidate_start
        block_weights = positive_weights[in_block].to(accumulation_dtype)
        positive_sum = torch.zeros_like(block_max)
        block_positive_valid = torch.zeros_like(block_valid)
        if block_rows.numel() > 0:
            positive_sum = positive_sum.index_add(
                0,
                block_rows,
                exp_values[block_rows, block_cols] * block_weights,
            )
            block_positive_valid.index_fill_(0, block_rows, True)
        block_positive_valid = block_positive_valid & block_valid
        safe_positive_sum = torch.where(
            block_positive_valid,
            positive_sum,
            torch.ones_like(positive_sum),
        )
        block_numerator_log = torch.log(safe_positive_sum) + safe_block_max
        numerator_log, numerator_valid = _merge_block_logsumexp(
            numerator_log,
            numerator_valid,
            block_numerator_log,
            block_positive_valid,
        )

    return denominator_log, numerator_log


def _get_positive_indices_for_point(
    point_idx: int,
    inst_points,
    inst_ids_per_point,
    device,
    include_self: bool = False,
) -> torch.Tensor:
    """
    Return a 1D LongTensor of positive point indices for the given anchor point,
    based on instance membership. Handles multi-label points by unioning instances.
    """
    inst_ids = inst_ids_per_point[point_idx]
    if len(inst_ids) == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if len(inst_ids) == 1:
        pos = inst_points[inst_ids[0]]
    else:
        # Union across instances if multi-label
        pos = torch.unique(torch.cat([inst_points[k] for k in inst_ids], dim=0))

    if not include_self:
        # Remove the anchor itself if present
        mask = pos != point_idx
        pos = pos[mask]
    return pos


def infoNCE_chunked_loss(
    features: torch.Tensor,
    instance_masks: torch.Tensor,
    chunk_size: int = 2048,
    logit_scale = 1.0,
    normalize: bool = True,
    bias = 0.0,
    include_self: bool = False,
    temporal_stages = None,
    stage_weight_same: float = 1.0,
    stage_weight_cross: float = 1.0,
    assume_single_label: bool = False,
    norm_type: str = "temperature",
    clamp_limits = None,
    candidate_chunk_size: int | None = None,
) -> torch.Tensor:
    """
    Memory-bounded supervised InfoNCE loss:
      - Candidate-blocked matmul with streaming log-sum-exp
      - Positive numerators via vectorized pair gather + index_add
      - Activation checkpointing so block activations are not retained for backward
      - Optional temporal stage weights

    Args:
      features: (N, C) float tensor
      instance_masks: (M, N) 0/1 or bool tensor
      chunk_size: anchors per chunk (tune for your GPU; 1024–4096 is typical)
      logit_scale: scalar or tensor (learnable temperature supported)
      normalize: L2-normalize features before similarity
      bias: additive bias on logits
      include_self: count the anchor as positive
      temporal_stages: optional (N,) int tensor of stage ids
      stage_weight_same: weight for same-stage positives
      stage_weight_cross: weight for cross-stage positives
      assume_single_label: set True if each point belongs to at most one instance
                           to skip dedup and gain extra speed
      candidate_chunk_size: optional upper bound for candidates per block. The
                            implementation also enforces a fixed byte budget.

    Returns:
      Scalar loss averaged over anchors that have at least one positive.
    """
    device = features.device
    dtype = features.dtype
    N = features.shape[0]

    z = _sanitize_features(features)
    if normalize:
        z = _normalize_features(z)

    inst_points = _instances_points(instance_masks)
    stages = temporal_stages.to(device) if temporal_stages is not None else None

    total = torch.zeros((), device=device, dtype=dtype)
    counted = 0

    # Enable TF32 for faster matmul on Ampere+ (optional)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    logit_scale_t = torch.as_tensor(logit_scale, device=device, dtype=dtype)
    bias_t = torch.as_tensor(bias, device=device, dtype=dtype)
    if candidate_chunk_size is not None and candidate_chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    accumulation_element_size = 4 if dtype in (
        torch.float16,
        torch.bfloat16,
    ) else features.element_size()
    max_block_elements = max(
        1,
        _INFO_NCE_SIMILARITY_BLOCK_BYTES // accumulation_element_size,
    )

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start

        # Build sparse positive pairs for this chunk
        rows, cols, weights, row_has_pos = _build_chunk_pairs(
            inst_points=inst_points,
            start=start,
            end=end,
            N=N,
            include_self=include_self,
            temporal_stages=stages,
            stage_weight_same=stage_weight_same,
            stage_weight_cross=stage_weight_cross,
            device=device,
            dtype=dtype,
            assume_single_label=assume_single_label,
        )

        # Skip chunk if no anchor has positives
        if row_has_pos.any().item() is False:
            continue

        dynamic_candidate_chunk_size = max(1, max_block_elements // B)
        if candidate_chunk_size is not None:
            dynamic_candidate_chunk_size = min(
                dynamic_candidate_chunk_size,
                candidate_chunk_size,
            )
        dynamic_candidate_chunk_size = min(N, dynamic_candidate_chunk_size)
        chunk_function = partial(
            _streaming_info_nce_chunk,
            anchor_start=start,
            candidate_chunk_size=dynamic_candidate_chunk_size,
            include_self=include_self,
            norm_type=norm_type,
            clamp_limits=clamp_limits,
        )
        chunk_inputs = (
            z[start:end],
            z,
            rows,
            cols,
            weights,
            logit_scale_t,
            bias_t,
        )
        requires_checkpoint = torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (z, logit_scale_t, bias_t)
        )
        if requires_checkpoint:
            denom_log, num_log = activation_checkpoint(
                chunk_function,
                *chunk_inputs,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            denom_log, num_log = chunk_function(*chunk_inputs)

        # Loss for valid rows
        loss_rows = denom_log[row_has_pos] - num_log[row_has_pos]
        total = total + loss_rows.sum()
        counted += int(row_has_pos.sum().item())

    if counted == 0:
        return torch.zeros((), device=device, dtype=dtype)
    return total / counted


def bce_sampled_loss(
    features: torch.Tensor,
    instance_masks: torch.Tensor,
    num_negatives: int = 256,
    normalize: bool = True,
    logit_scale = 1.0,
    bias = 0.0,
    include_self: bool = False,
    temporal_stages = None,
    stage_weight_same: float = 1.0,
    stage_weight_cross: float = 1.0,
) -> torch.Tensor:
    """
    Memory-lean pairwise BCE with logits:
      - Computes positives exactly (via instance membership)
      - Samples a fixed number of negatives per anchor

    Args mirror infoNCE_chunked_loss with num_negatives added.

    Returns:
      Mean BCE loss over anchors that have positives.
    """
    device = features.device
    dtype = features.dtype
    N = features.shape[0]

    z = _sanitize_features(features)
    if normalize:
        z = _normalize_features(z)

    inst_points, inst_ids_per_point, _ = _build_instance_structures(instance_masks)

    total_loss = torch.zeros((), device=device, dtype=dtype)
    counted = 0

    # Precompute all similarities once in chunks to save memory
    # We will reuse rows on demand to compute logits for pos/neg sets
    chunk_size = min(2048, N)  # separate chunking here
    sim_cache = {}  # maps chunk_idx -> sim_chunk (B, N)

    def get_sim_row(j_abs: int) -> torch.Tensor:
        # Retrieve the similarity row for anchor j_abs (with scale/bias)
        chunk_idx = j_abs // chunk_size
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, N)
        if chunk_idx not in sim_cache:
            z_chunk = z[start:end]
            sim_chunk = z_chunk @ z.T
            # Handle logit_scale (can be tensor for learnable temperature)
            if isinstance(logit_scale, torch.Tensor):
                sim_chunk = sim_chunk * logit_scale
            elif logit_scale != 1.0:
                sim_chunk = sim_chunk * logit_scale
            # Handle bias (can be tensor for learnable bias)
            if isinstance(bias, torch.Tensor):
                sim_chunk = sim_chunk + bias
            elif bias != 0.0:
                sim_chunk = sim_chunk + bias
            sim_cache[chunk_idx] = sim_chunk
        row = sim_cache[chunk_idx][j_abs - start]  # (N,)
        return row

    for j_abs in range(N):
        pos_idx = _get_positive_indices_for_point(
            j_abs, inst_points, inst_ids_per_point, device, include_self=include_self
        )

        if pos_idx.numel() == 0:
            continue

        sim_row = get_sim_row(j_abs)

        # Positives
        pos_logits = sim_row[pos_idx]
        pos_targets = torch.ones_like(pos_logits)

        if temporal_stages is not None:
            anchor_stage = temporal_stages[j_abs].to(device)
            pos_stages = temporal_stages[pos_idx].to(device)
            w_same = torch.as_tensor(stage_weight_same, device=device, dtype=dtype)
            w_cross = torch.as_tensor(stage_weight_cross, device=device, dtype=dtype)
            pos_weights = torch.where(pos_stages == anchor_stage, w_same, w_cross)
        else:
            pos_weights = torch.ones_like(pos_logits)

        # Negatives: sample uniformly from points not in pos set (and optionally not self)
        all_idx = torch.arange(N, device=device)
        neg_mask = torch.ones(N, dtype=torch.bool, device=device)
        neg_mask[pos_idx] = False
        if not include_self:
            neg_mask[j_abs] = False
        neg_pool = all_idx[neg_mask]
        if neg_pool.numel() == 0:
            continue
        # Random sample without replacement (or with replacement if pool smaller)
        if neg_pool.numel() >= num_negatives:
            perm = torch.randperm(neg_pool.numel(), device=device)[:num_negatives]
            neg_idx = neg_pool[perm]
        else:
            # With replacement
            rand_idx = torch.randint(0, neg_pool.numel(), (num_negatives,), device=device)
            neg_idx = neg_pool[rand_idx]

        neg_logits = sim_row[neg_idx]
        neg_targets = torch.zeros_like(neg_logits)
        neg_weights = torch.ones_like(neg_logits)

        # BCE with logits
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits, pos_targets, weight=pos_weights, reduction="sum"
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits, neg_targets, weight=neg_weights, reduction="sum"
        )

        total_loss = total_loss + (pos_loss + neg_loss) / (pos_idx.numel() + neg_idx.numel())
        counted += 1

    if counted == 0:
        return torch.zeros((), device=device, dtype=dtype)
    return total_loss / counted


class ContrastiveLoss(nn.Module):
    """Contrastive loss module with configurable loss type and learnable parameters."""
    CLAMP_LIMITS = {
        torch.float16: (-0.99, 0.99),
        torch.bfloat16: (-0.995, 0.995), 
        torch.float32: (-0.99999, 0.99999),
        torch.float64: (-0.99999999999, 0.99999999999)
    }
    
    def __init__(
        self,
        loss_type="infonce",
        learnable_temperature=True,
        learnable_bias=True,
        initial_temperature=0.5,
        initial_bias=0.0,
        norm_type="clamp",
        weight_temporal_stages=False,
        temporal_positive_weight=2.0,
        chunk_size=2048,
        use_chunked_loss=True,
        num_negatives=256,  # for bce_sampled
        assume_single_label=False,  # optimization: set True if points belong to at most one instance
    ):
        super().__init__()
        
        # Normalize loss type naming
        self.loss_type = loss_type.lower()
        self.learnable_temperature = learnable_temperature
        self.learnable_bias = learnable_bias
        # contrastive loss learnable parameters
        if learnable_temperature:
            self.t = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.t = torch.tensor(initial_temperature)
            
        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(initial_bias))
        else:
            self.bias = torch.tensor(initial_bias)

        self.norm_type = norm_type
        self.weight_temporal_stages = weight_temporal_stages
        self.temporal_positive_weight = float(temporal_positive_weight)
        self.chunk_size = chunk_size
        self.use_chunked_loss = use_chunked_loss
        self.num_negatives = num_negatives
        self.assume_single_label = assume_single_label

    def temperature(self):
        if self.learnable_temperature:
            return torch.exp(self.t)
        else:
            return 1 / self.t

    def logits_and_gt_mask(self, features, instance_masks):
        """Compute logits and ground truth mask with mixed-precision safety."""
        # Normalize with eps and sanitize NaNs/Infs
        features = F.normalize(features, p=2, dim=1, eps=1e-6)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        sim_matrix = torch.mm(features, features.T)
        
        if self.norm_type == "log_odds":
            sim_matrix = torch.clamp(sim_matrix, *self.CLAMP_LIMITS[sim_matrix.dtype])
            logits = 2 * torch.atanh(sim_matrix) + self.bias
        elif self.norm_type == "clamp":
            logits = ((1 + sim_matrix) / 2).clamp(min=0, max=1) + self.bias
        elif self.norm_type == "temperature":
            logits = sim_matrix * self.temperature() + self.bias
        else:
            raise ValueError(f"Norm type: {self.norm_type} not supported")

        # Ensure logits are finite
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Build boolean ground-truth similarity mask in float32 math
        instance_masks = instance_masks.float()
        gt_similarity = torch.mm(instance_masks.T, instance_masks)

        return logits, gt_similarity
    
    def sigmoid_loss(self, logits, gt_mask, weights=None):
        """Sigmoid loss: Binary cross-entropy loss for contrastive learning."""
        # mathematically equivalent to sigmoid cross-entropy loss

        # independent optimization of each positive
        # calculate only over upper triangle excluding diagonal to reduce pair count by ~2x
        tri_mask = torch.triu(torch.ones_like(logits), diagonal=1).bool()
        logits = logits[tri_mask]
        gt_mask = gt_mask[tri_mask]
        if weights is not None:
            weights = weights[tri_mask]
        loss = F.binary_cross_entropy_with_logits(logits, gt_mask, weight=weights, reduction="mean")
        
        return loss

    def infoNCE_loss(self, logits, gt_mask, weights=None):
        """
        InfoNCE loss for contrastive learning.
        - Excludes diagonal (self-similarity)
        - Excludes rows with no positives in reduction (included only as negative samples)
        - Uses uniform soft targets over positives to equalize their probabilities
        """
        
        # Remove self-similarity by setting diagonal to -inf so logsumexp ignores it
        logits.fill_diagonal_(-torch.inf)
        gt_mask.fill_diagonal_(0) 

        # Filter out queries with no positive pairs (excluding diagonal)
        valid_mask = gt_mask.any(dim=1)  # (N,)
        # if queries have no positive pairs, return 0 loss
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        
        # Compute logsumexp over all similarities for each query
        all_logsumexp = torch.logsumexp(logits[valid_mask], dim=1)  # (N,)
        
        # For each query, compute the sum of exp(similarities) for positive pairs
        # multiply similarities by positive mask, then use logsumexp
        # note: positives not equalized, encourages entire positive mass to be large
        # competition between all pairs
        pos_sim = logits.masked_fill(~gt_mask.bool(), -torch.inf)
        if weights is not None:
            # Only affect positive pairs; others stay the same
            # Multiply exp(sim) by weight => add log(weight) to logits before logsumexp
            safe_w = torch.clamp(weights, min=1e-12)
            add_log_w = torch.where(gt_mask.bool(), torch.log(safe_w), torch.zeros_like(safe_w))
            pos_sim = pos_sim + add_log_w

        pos_logsumexp = torch.logsumexp(pos_sim[valid_mask], dim=1)  # (N,)
        
        # InfoNCE loss per query: -log(exp(pos) / exp(all)) = -pos + all
        per_sample_loss = -pos_logsumexp + all_logsumexp  # (N,)

        return per_sample_loss.mean()

    def sinkhorn_contrastive_loss(self, logits, gt_mask):
        """
        Sinkhorn contrastive loss for contrastive learning.
        - Excludes diagonal (self-similarity)
        - Excludes rows with no positives in reduction (included only as negative samples)
        - Uses uniform soft targets over positives to equalize their probabilities
        """
        raise NotImplementedError
        # Convert to soft assignments using Sinkhorn-Knopp
        t = self.sinkhorn_knopp(logits, gt_mask)
        
        # Compute loss between soft assignments and similarity matrix
        s = F.softmax(logits / self.temperature, dim=-1)
        
        # KL divergence between target and predicted distributions
        loss = -(t * torch.log(s + 1e-8)).sum(dim=1).mean()
        
        return loss

    def sinkhorn_knopp(self, logits, gt_mask, num_iterations=10):
        """
        Apply Sinkhorn-Knopp normalization to create doubly stochastic matrix.
        """
        raise NotImplementedError
        # Start with similarity matrix
        t = F.softmax(logits / self.temperature, dim=-1)
        
        # Sinkhorn-Knopp iterations
        for _ in range(num_iterations):
            # Normalize rows
            t = t / (t.sum(dim=1, keepdim=True) + 1e-8)
            # Normalize columns  
            t = t / (t.sum(dim=0, keepdim=True) + 1e-8)
        
        return t

    
    def forward(self, features, instance_masks, temporal_stages=None):
        """Forward pass using the configured loss type.
        features: (N, C) tensor
        instance_masks: (M, N) binary masks indicating instance membership per point
        temporal_stages: Optional (N,) int tensor indicating stage per point (for per-point weighting)
        """
        if getattr(self, "p2_fail_closed_runtime", False):
            if not torch.isfinite(features).all():
                raise ValueError("non-finite contrastive features")
            if not torch.isfinite(instance_masks).all():
                raise ValueError("non-finite contrastive instance masks")
            if (
                temporal_stages is not None
                and not torch.isfinite(temporal_stages).all()
            ):
                raise ValueError("non-finite contrastive temporal stages")

        # Use memory-efficient chunked implementation by default
        if self.use_chunked_loss:
            # Compute logit scale and bias from learnable parameters
            # Support both learnable and fixed parameters (as tensors or scalars)
            if self.norm_type == "temperature":
                logit_scale = self.temperature()  # Returns tensor if learnable, scalar if fixed
            elif self.norm_type == "clamp":
                logit_scale = 1.0
            elif self.norm_type == "log_odds":
                logit_scale = 1.0
            else:
                logit_scale = 1.0
            
            # Get bias value (can be tensor or scalar)
            if self.learnable_bias:
                bias_val = self.bias  # Keep as tensor for gradients
            else:
                if isinstance(self.bias, torch.Tensor):
                    bias_val = self.bias.item()
                else:
                    bias_val = float(self.bias)
            
            # Map temporal weighting: if weight_temporal_stages is True, we upweight cross-stage pairs
            # The chunked version uses stage_weight_same=1.0 and stage_weight_cross=temporal_positive_weight
            stage_weight_same = 1.0
            stage_weight_cross = self.temporal_positive_weight if self.weight_temporal_stages else 1.0
            
            if self.loss_type == "infonce":
                loss = infoNCE_chunked_loss(
                    features=features,
                    instance_masks=instance_masks,
                    chunk_size=self.chunk_size,
                    logit_scale=logit_scale,
                    normalize=True,  # Always normalize in chunked version
                    bias=bias_val,
                    include_self=False,
                    temporal_stages=temporal_stages if self.weight_temporal_stages else None,
                    stage_weight_same=stage_weight_same,
                    stage_weight_cross=stage_weight_cross,
                    assume_single_label=self.assume_single_label,
                    norm_type=self.norm_type,
                    clamp_limits=self.CLAMP_LIMITS.get(features.dtype, None),
                )
            elif self.loss_type == "bce_sampled":
                loss = bce_sampled_loss(
                    features=features,
                    instance_masks=instance_masks,
                    num_negatives=self.num_negatives,
                    normalize=True,
                    logit_scale=logit_scale,
                    bias=bias_val,
                    include_self=False,
                    temporal_stages=temporal_stages if self.weight_temporal_stages else None,
                    stage_weight_same=stage_weight_same,
                    stage_weight_cross=stage_weight_cross,
                )
            elif self.loss_type == "sigmoid":
                # For sigmoid, still use old dense implementation (can be updated later)
                logits, gt_mask = self.logits_and_gt_mask(features, instance_masks)
                weights = None
                if self.weight_temporal_stages and temporal_stages is not None:
                    weights = self.temporal_positive_weights(temporal_stages, gt_mask)
                loss = self.sigmoid_loss(logits, gt_mask, weights)
            else:
                raise NotImplementedError(f"Unsupported contrastive loss type: {self.loss_type}")
        else:
            # Fall back to old dense implementation
            logits, gt_mask = self.logits_and_gt_mask(features, instance_masks)
            weights = None
            if self.weight_temporal_stages and temporal_stages is not None:
                weights = self.temporal_positive_weights(temporal_stages, gt_mask)

            if self.loss_type == "infonce":
                loss = self.infoNCE_loss(logits, gt_mask, weights)
            elif self.loss_type == "sigmoid":
                loss = self.sigmoid_loss(logits, gt_mask, weights)
            else:
                raise NotImplementedError(f"Unsupported contrastive loss type: {self.loss_type}")

        if (
            getattr(self, "p2_fail_closed_runtime", False)
            and not torch.isfinite(loss).all()
        ):
            raise ValueError("non-finite contrastive loss")

        # Preserve the upstream fallback outside the P2 fail-closed path.
        if torch.isnan(loss):
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        
        return loss

    def temporal_positive_weights(self, temporal_stages, gt_mask):
        """Build pairwise weights: upweight only positive pairs across different temporal stages.
        temporal_stages: (N,) tensor with stage ids per instance
        gt_mask: (N, N) bool/int tensor of positive pairs (same instance)
        Returns: (N, N) tensor with 1 for all pairs, and w for positive cross-stage pairs.
        """
        # Ensure 1D long tensor
        stages = temporal_stages.view(-1)
        # Pairwise matrix where True if stages differ
        cross_stage = stages[:, None] != stages[None, :]
        cross_stage = cross_stage & gt_mask.bool()
        weights = torch.ones_like(gt_mask, dtype=torch.float32)
        if self.temporal_positive_weight != 1.0:
            weights = torch.where(cross_stage, torch.full_like(weights, self.temporal_positive_weight), weights)  
        
        return weights


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        class_weights,
        num_changes, 
        change_weights,
        contrastive_loss=False,
        contrastive_loss_type="infoNCE",
        learnable_temperature=True,
        learnable_bias=True,
        initial_temperature=0.5,
        initial_bias=0.0, 
        norm_type="temperature",
        weight_temporal_stages=False,
        temporal_positive_weight=2.0,
        scale_contrastive_loss=False,
        contrastive_loss_weight=1.0,
        max_points=5000,
        chunk_size=2048,
        use_chunked_loss=True,
        num_negatives=256,
        assume_single_label=False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            num_changes: number of change classes to predict 
            change_weights: weights for the change classes
        """
        super().__init__()
        self.num_classes = num_classes - 1
        self.class_weights = class_weights
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef

        self.use_contrastive_loss = contrastive_loss
        self.scale_contrastive_loss = scale_contrastive_loss
        self.contrastive_loss_weight = contrastive_loss_weight
        self.num_changes = num_changes
        self.max_points = max_points
        if self.class_weights != -1:
            assert (
                len(self.class_weights) == self.num_classes
            ), "CLASS WEIGHTS DO NOT MATCH"
            empty_weight[:-1] = torch.tensor(self.class_weights)
            
        if change_weights != -1:
            assert (
                len(change_weights) == self.num_changes
            ), "CHANGE WEIGHTS DO NOT MATCH"
            change_weights = torch.tensor(change_weights)

        self.register_buffer("empty_weight", empty_weight)
        self.register_buffer("change_weights", change_weights)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        if self.use_contrastive_loss:
            self.contrastive_loss = ContrastiveLoss(
                loss_type=contrastive_loss_type,
                learnable_temperature=learnable_temperature,
                learnable_bias=learnable_bias,
                initial_temperature=initial_temperature,
                initial_bias=initial_bias,
                norm_type=norm_type,
                weight_temporal_stages=weight_temporal_stages,
                temporal_positive_weight=temporal_positive_weight,
                chunk_size=chunk_size,
                use_chunked_loss=use_chunked_loss,
                num_negatives=num_negatives,
                assume_single_label=assume_single_label,
            )


    def loss_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
            ignore_index=253,
        )
        losses = {"loss_ce": loss_ce}
        return losses
    
    def loss_change_labels(self, outputs, targets, indices, num_masks, mask_type):
        """Classification loss (NLL)
        targets dicts must contain the key "change" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_changes" in outputs
        src_logits = outputs["pred_changes"].float()

        idx = self._get_src_permutation_idx(indices)
        target_changes_o = torch.cat(
            [t["changes"][J] for t, (_, J) in zip(targets, indices)]
        )
        # instead of no object label, default to static 
        target_changes = torch.full(
            src_logits.shape[:2],
            0,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_changes[idx] = target_changes_o

        loss_changes_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_changes,
            self.change_weights,
        )
        losses = {"loss_changes_ce": loss_changes_ce}
        return losses

    def loss_aux_contrastive(self, outputs, targets, indices, num_masks, mask_type):
        device = next(iter(outputs.values())).device
        if ("aux_features" not in outputs or "pooling_fn" not in outputs) or "labels" not in targets[0]:
            return {"loss_aux_contrastive": torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        aux_features = outputs["aux_features"][::-1]
        batch_size = len(targets)

        pooled_instance_masks = [targets[b]["masks"].T for b in range(len(targets))]
        # temporal_stages = [targets[b]["temporal_stages"] for b in range(len(targets))]

        out = {}
        total_loss = torch.tensor(0.0, device=device)
        for layer_idx, aux_feature in enumerate(aux_features):
            aux_layer_loss = torch.tensor(0.0, device=device)
            # pool gt masks 
            pooled_instance_masks = outputs["pooling_fn"](aux_feature,pooled_instance_masks, reduce="max")
            features = aux_feature.decomposed_features
            for b in range(batch_size):
                # Sample features to reduce memory usage while maintaining point-level structure
                # features[b]: (num_points, feature_dim)
                # pooled_instance_masks[b]: (num_points, num_instances)
                point_features = features[b]  # (num_points, feature_dim)
                instance_masks = pooled_instance_masks[b]  # (num_points, num_instances)
                num_points = point_features.shape[0]

                
                if num_points > self.max_points:
                    # Randomly sample points
                    sample_indices = torch.randperm(num_points, device=device)[:self.max_points]
                    sampled_features = point_features[sample_indices]  # (max_points, feature_dim)
                    sampled_instance_masks = instance_masks[sample_indices]  # (max_points, num_instances)
                else:
                    sampled_features = point_features
                    sampled_instance_masks = instance_masks
                
                # temporal_stages[b] = pooling_fn(temporal_stages[b])
                aux_layer_loss += self.contrastive_loss(sampled_features, sampled_instance_masks.T)
            if batch_size > 0:
                aux_layer_loss = aux_layer_loss / batch_size
            out[f"loss_aux_contrastive_layer_{layer_idx}"] = aux_layer_loss
            total_loss += aux_layer_loss

        out.update({"loss_aux_contrastive": total_loss})
        return out


    def loss_segment_contrastive(self, outputs, targets, indices, num_masks, mask_type):
        """Segment level contrastive loss using mask decoded segment features.
        
        Computes contrastive loss at the segment level by:
        1. Using pre-computed segment features from feature_refinement_aux
        2. Building ground truth similarity matrix from segment_mask (already downsampled)
        3. Computing segment similarity matrix and applying contrastive loss
        """
        # Resolve device
        device = next(iter(outputs.values())).device
        if "segment_features" not in outputs or "labels" not in targets[0]:
            return {"loss_segment_contrastive": torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        batch_size = len(targets)
        
        # Get device from segment features
        total_loss = torch.tensor(0.0, device=device)
        per_layer_losses = {}
        for layer_idx, layer_features in enumerate(outputs["segment_features"]):
            layer_loss = torch.tensor(0.0, device=device)
            for b in range(batch_size):
                instance_masks = targets[b]["segment_mask"]
                
                if self.contrastive_loss.weight_temporal_stages and "temporal_stages" in targets[b]:
                    # Reduce temporal_stages from points to segments
                    p2s = targets[b]["point2segment"]
                    ts = targets[b]["temporal_stages"]
                    temporal_stages = torch.zeros(int(p2s.max()) + 1, device=ts.device, dtype=ts.dtype)
                    temporal_stages.scatter_reduce_(0, p2s, ts, reduce='amax', include_self=False)
                else:
                    temporal_stages = None
                
                layer_loss += self.contrastive_loss(layer_features[b], instance_masks, temporal_stages)
            if batch_size > 0:
                layer_loss = layer_loss / batch_size
            per_layer_losses[f"loss_segment_contrastive_layer{layer_idx}"] = layer_loss
            total_loss += layer_loss

        if self.scale_contrastive_loss and len(outputs["segment_features"]) == 1:
            # scale by the number of aux outputs to balance the loss
            total_loss = total_loss * len(outputs["aux_outputs"]) * self.contrastive_loss_weight

        out = {"loss_segment_contrastive": total_loss}
        out.update(per_layer_losses)
        return out

    def loss_masks(self, outputs, targets, indices, num_masks, mask_type):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        loss_masks = []
        loss_dices = []

        for batch_id, (map_id, target_id) in enumerate(indices):
            map = outputs["pred_masks"][batch_id][:, map_id].T
            target_mask = targets[batch_id][mask_type][target_id]

            # Drop any prediction queries whose mask contains non-finite values
            finite_rows = torch.isfinite(map).all(dim=1)
            if not finite_rows.all():
                if finite_rows.any():
                    map = map[finite_rows]
                    target_mask = target_mask[finite_rows]
                else:
                    # No valid masks for this sample, skip
                    continue

            if self.num_points != -1:
                point_idx = torch.randperm(
                    target_mask.shape[1], device=target_mask.device
                )[: int(self.num_points * target_mask.shape[1])]
            else:
                # sample all points
                point_idx = torch.arange(
                    target_mask.shape[1], device=target_mask.device
                )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_masks.append(sigmoid_ce_loss_jit(map, target_mask, num_masks))
            loss_dices.append(dice_loss_jit(map, target_mask, num_masks))
        # del target_mask
        return {
            "loss_mask": torch.sum(torch.stack(loss_masks)),
            "loss_dice": torch.sum(torch.stack(loss_dices)),
        }


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, mask_type):
        loss_map = {
            "labels": self.loss_labels, 
            "masks": self.loss_masks, 
            "changes": self.loss_change_labels,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, mask_type)

    def forward(self, outputs, targets, mask_type):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        def normalized_num_masks(device):
            num_masks = sum(len(t["labels"]) for t in targets)
            num_masks = torch.as_tensor(
                [num_masks],
                dtype=torch.float,
                device=device,
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks)
            return torch.clamp(
                num_masks / get_world_size(),
                min=1,
            ).detach()

        p2_fail_closed_runtime = getattr(
            self,
            "p2_fail_closed_runtime",
            False,
        )
        if p2_fail_closed_runtime:
            num_masks = normalized_num_masks(outputs["pred_logits"].device)

        outputs_without_aux = {
            k: v for k, v in outputs.items() if k != "aux_outputs"
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, mask_type)

        if not p2_fail_closed_runtime:
            num_masks = normalized_num_masks(
                next(iter(outputs.values())).device
            )

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss, outputs, targets, indices, num_masks, mask_type
                )
            )

        if self.use_contrastive_loss:
            losses.update(self.loss_segment_contrastive(outputs, targets, indices, num_masks, mask_type))
            losses.update(self.loss_aux_contrastive(outputs, targets, indices, num_masks, mask_type))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # exclude contrastive loss from auxiliary losses
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets, mask_type)
                for loss in [l for l in self.losses]:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                        mask_type,
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "num_changes: {}".format(self.num_changes),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
