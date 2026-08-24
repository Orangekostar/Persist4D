"""T2 current-stage parity checks for System Comparison V2."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from scripts.p6a_metrics import compute_official_raw_local_metrics
from scripts.system_comparison_metrics import (
    causal_prefix_pair_from_payload,
    current_stage_pair,
)
from scripts.system_comparison_v2_cache import (
    task_sidecar_digest,
    validate_task_sidecar,
)


class T2ParityError(ValueError):
    """Raised when T2 task predictions are not valid comparison inputs."""


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise T2ParityError(f"{name} must be a lowercase SHA256 digest")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise T2ParityError(f"{name} must be a mapping")
    return value


def _tensor(value: object, *, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise T2ParityError(f"{name} must be a rank-{ndim} tensor")
    result = value.detach().cpu().contiguous().clone()
    if result.is_floating_point() and not torch.isfinite(result).all().item():
        raise T2ParityError(f"{name} must contain finite values")
    return result


def _local_prediction(sidecar: Mapping[str, object]) -> dict[str, Tensor]:
    task = _mapping(sidecar["task_prediction"], name="local task prediction")
    return {
        "pred_masks": _tensor(task["pred_masks"], name="local masks", ndim=2),
        "pred_scores": _tensor(task["pred_scores"], name="local scores", ndim=1),
        "pred_classes": _tensor(
            task["pred_classes"], name="local classes", ndim=1
        ),
    }


def _metric_ap(
    metric_function: Callable[..., Mapping[str, object]],
    prediction: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
) -> float:
    metrics = metric_function(prediction, target)
    if not isinstance(metrics, Mapping) or "raw_local_AP" not in metrics:
        raise T2ParityError("metric function did not return raw_local_AP")
    value = metrics["raw_local_AP"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise T2ParityError("raw_local_AP must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise T2ParityError("raw_local_AP must be finite")
    return result


def _validate_alignment(
    *,
    full_key: Mapping[str, object],
    local_key: Mapping[str, object],
) -> None:
    if full_key.get("horizon") != 2 or local_key.get("stage_index") != 1:
        raise T2ParityError("parity inputs must describe T2/stage-1")
    exact_fields = (
        "master_sequence_id",
        "reference_scene_id",
        "order_id",
        "history_scan_ids",
    )
    if any(full_key.get(name) != local_key.get(name) for name in exact_fields):
        raise T2ParityError("FullHistory and local T2 keys/history differ")


def compare_t2_task_predictions(
    *,
    full_payload: Mapping[str, object],
    local_sidecar: Mapping[str, object],
    full_history_content_sha256: str,
    sidecar_content_sha256: str,
    metric_function: Callable[..., Mapping[str, object]] = (
        compute_official_raw_local_metrics
    ),
    score_atol: float = 1e-7,
    ap_atol: float = 1e-12,
) -> dict[str, object]:
    """Compare one exact T2 FullHistory/local official-task prediction pair."""

    if not callable(metric_function):
        raise T2ParityError("metric_function must be callable")
    if score_atol < 0.0 or ap_atol < 0.0:
        raise T2ParityError("parity tolerances must be nonnegative")
    validate_task_sidecar(local_sidecar)
    full_digest = _sha256(
        full_history_content_sha256, name="full_history_content_sha256"
    )
    sidecar_digest = _sha256(
        sidecar_content_sha256, name="sidecar_content_sha256"
    )
    if full_payload.get("content_sha256") != full_digest:
        raise T2ParityError("FullHistory content hash differs from payload")
    if task_sidecar_digest(local_sidecar) != sidecar_digest:
        raise T2ParityError("sidecar content hash differs from payload")

    full_pair = current_stage_pair(causal_prefix_pair_from_payload(full_payload))
    full_key = _mapping(full_payload.get("key"), name="FullHistory key")
    local_key = _mapping(local_sidecar.get("key"), name="local key")
    _validate_alignment(full_key=full_key, local_key=local_key)
    full_prediction = full_pair.prediction
    local_prediction = _local_prediction(local_sidecar)

    full_count = int(full_prediction["pred_scores"].numel())
    local_count = int(local_prediction["pred_scores"].numel())
    masks_equal = torch.equal(
        full_prediction["pred_masks"], local_prediction["pred_masks"]
    )
    classes_equal = torch.equal(
        full_prediction["pred_classes"], local_prediction["pred_classes"]
    )
    if full_count == local_count:
        difference = torch.abs(
            full_prediction["pred_scores"] - local_prediction["pred_scores"]
        )
        score_max_abs_diff = float(difference.max().item()) if full_count else 0.0
        scores_allclose = bool(
            torch.allclose(
                full_prediction["pred_scores"],
                local_prediction["pred_scores"],
                rtol=0.0,
                atol=score_atol,
            )
        )
    else:
        score_max_abs_diff = math.inf
        scores_allclose = False

    full_ap = _metric_ap(metric_function, full_prediction, full_pair.target)
    local_ap = _metric_ap(metric_function, local_prediction, full_pair.target)
    ap_abs_diff = abs(full_ap - local_ap)
    parity_pass = bool(
        full_count == local_count
        and masks_equal
        and classes_equal
        and scores_allclose
        and ap_abs_diff <= ap_atol
    )
    provenance = _mapping(local_sidecar["provenance"], name="local provenance")
    return {
        "master_sequence_id": str(local_key["master_sequence_id"]),
        "reference_scene_id": str(local_key["reference_scene_id"]),
        "order_id": str(local_key["order_id"]),
        "history_scan_ids": list(local_key["history_scan_ids"]),
        "candidate_count_full": full_count,
        "candidate_count_local": local_count,
        "masks_equal": masks_equal,
        "classes_equal": classes_equal,
        "score_max_abs_diff": score_max_abs_diff,
        "scores_allclose": scores_allclose,
        "full_current_stage_AP": full_ap,
        "local_current_stage_AP": local_ap,
        "AP_abs_diff": ap_abs_diff,
        "parity_pass": parity_pass,
        "raw_observation_fingerprint": provenance[
            "source_raw_observation_fingerprint"
        ],
        "sidecar_content_sha256": sidecar_digest,
        "full_history_content_sha256": full_digest,
    }


def t2_cache_identities(
    local_key: Mapping[str, object],
) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    key = _mapping(local_key, name="local T2 key")
    if key.get("stage_index") != 1:
        raise T2ParityError("local T2 key must use stage_index 1")
    master = key.get("master_sequence_id")
    order = key.get("order_id")
    if not isinstance(master, str) or not master or not isinstance(order, str):
        raise T2ParityError("local T2 key identity fields are invalid")
    return (master, order, 1), (master, order, 2)


def summarize_t2_rows(
    rows: Sequence[Mapping[str, object]], *, expected_unit_count: int = 129
) -> dict[str, object]:
    if (
        isinstance(expected_unit_count, bool)
        or not isinstance(expected_unit_count, int)
        or expected_unit_count <= 0
    ):
        raise T2ParityError("expected_unit_count must be positive")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise T2ParityError("parity rows must be a sequence")
    identities = []
    for row in rows:
        value = _mapping(row, name="parity row")
        identities.append(
            (
                value.get("master_sequence_id"),
                value.get("reference_scene_id"),
                value.get("order_id"),
            )
        )
    if len(rows) != expected_unit_count or len(set(identities)) != len(rows):
        raise T2ParityError("T2 parity coverage is not exact and unique")
    failed = sum(row.get("parity_pass") is not True for row in rows)
    return {
        "status": "pass" if failed == 0 else "fail",
        "unit_count": len(rows),
        "pass_count": len(rows) - failed,
        "fail_count": failed,
        "max_score_abs_diff": max(
            float(row["score_max_abs_diff"]) for row in rows
        ),
        "max_AP_abs_diff": max(float(row["AP_abs_diff"]) for row in rows),
    }


__all__ = [
    "T2ParityError",
    "compare_t2_task_predictions",
    "summarize_t2_rows",
    "t2_cache_identities",
]
