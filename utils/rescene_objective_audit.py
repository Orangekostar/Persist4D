"""Pure objective and EOS diagnostics for ReScene root-cause analysis."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _scalar(value: torch.Tensor, *, name: str) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    number = float(value.detach().cpu().item())
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def is_contrastive_diagnostic(loss_key: str) -> bool:
    return loss_key.startswith("loss_") and "_contrastive_layer" in loss_key


def _classification(loss_key: str) -> str:
    if is_contrastive_diagnostic(loss_key):
        return "diagnostic"
    if loss_key in {"loss_segment_contrastive", "loss_aux_contrastive"}:
        return "aggregate"
    return "objective"


def objective_contribution_rows(
    losses: Mapping[str, torch.Tensor], weight_dict: Mapping[str, float]
) -> list[dict[str, object]]:
    """Describe exact released-code and local weighted contributions."""

    if not losses:
        raise ValueError("objective audit requires loss terms")
    rows: list[dict[str, object]] = []
    for loss_key in sorted(losses):
        raw_value = _scalar(losses[loss_key], name=loss_key)
        diagnostic = is_contrastive_diagnostic(loss_key)
        local_multiplier = 0.0 if diagnostic else float(weight_dict.get(loss_key, 1.0))
        if not math.isfinite(local_multiplier):
            raise ValueError(f"loss multiplier must be finite: {loss_key}")
        rows.append(
            {
                "loss_key": loss_key,
                "classification": _classification(loss_key),
                "raw_value": raw_value,
                "upstream_multiplier": 1.0,
                "upstream_contribution": raw_value,
                "upstream_included": True,
                "local_weighted_multiplier": local_multiplier,
                "local_weighted_contribution": raw_value * local_multiplier,
                "local_weighted_included": not diagnostic,
            }
        )
    return rows


def optimized_objective(
    losses: Mapping[str, torch.Tensor],
    weight_dict: Mapping[str, float],
    *,
    mode: str,
) -> torch.Tensor:
    """Evaluate either exact public raw sum or current weighted reducer."""

    if not losses:
        raise ValueError("objective audit requires loss terms")
    if mode == "raw_sum":
        result = sum(losses.values())
    elif mode == "weighted":
        from trainer.trainer import aggregate_objective_loss

        result = aggregate_objective_loss(losses, weight_dict)
    else:
        raise ValueError("objective mode must be weighted or raw_sum")
    _scalar(result, name="optimized objective")
    return result


def compare_gradients(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    """Compare two aligned finite gradient vectors."""

    left = left.detach().cpu().float().reshape(-1)
    right = right.detach().cpu().float().reshape(-1)
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("gradient vectors must be non-empty and aligned")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("gradient vectors must be finite")
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 and right_norm == 0.0:
        cosine = 1.0
    elif left_norm == 0.0 or right_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(left, right, dim=0).item()
        )
    relative = abs(right_norm - left_norm) / max(left_norm, torch.finfo(torch.float32).eps)
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": cosine,
        "relative_norm_difference": relative,
        "max_absolute_difference": float((right - left).abs().max().item()),
        "element_count": int(left.numel()),
    }


def eos_gradient_gate(comparison: Mapping[str, float]) -> dict[str, object]:
    """Apply the preregistered EOS materiality gate."""

    cosine = float(comparison["cosine"])
    relative = float(comparison["relative_norm_difference"])
    if not math.isfinite(cosine) or not math.isfinite(relative):
        raise ValueError("EOS gradient comparison must be finite")
    thresholds = {
        "minimum_cosine": 0.98,
        "maximum_relative_norm_difference": 0.1,
    }
    return {
        "authorized": cosine < thresholds["minimum_cosine"]
        or relative > thresholds["maximum_relative_norm_difference"],
        "thresholds": thresholds,
        "comparison": dict(comparison),
    }
