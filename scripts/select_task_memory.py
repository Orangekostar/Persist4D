"""Select one TaskMemory checkpoint from frozen all-T development cells."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_M2_UPDATES = (0, 750, 1500, 2250, 3000)
SELECTION_HORIZONS = (2, 3, 4, 5)
SELECTION_BASELINES = ("FH-MATCH", "R1")
_REQUIRED_FIELDS = {
    "T",
    "baseline",
    "checkpoint_sha256",
    "delta_t_mAP",
    "median_update_latency_ms",
    "policy",
    "reducer",
    "update",
    "variant",
}


class TaskMemorySelectionError(ValueError):
    """Raised when selection evidence differs from the frozen M2 design."""


def _finite(value: object, *, name: str, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0)
    ):
        raise TaskMemorySelectionError(f"{name} must be finite")
    return float(value)


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TaskMemorySelectionError("checkpoint_sha256 must be lowercase hex")
    return value


def _normalized_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise TaskMemorySelectionError("selection rows must be a non-empty sequence")
    normalized = []
    for raw in rows:
        if not isinstance(raw, Mapping) or not _REQUIRED_FIELDS <= set(raw):
            raise TaskMemorySelectionError("selection row fields differ")
        variant = raw["variant"]
        policy = raw["policy"]
        reducer = raw["reducer"]
        baseline = raw["baseline"]
        if any(not isinstance(value, str) or not value for value in (variant, policy, reducer)):
            raise TaskMemorySelectionError("selection identities must be strings")
        if baseline not in SELECTION_BASELINES:
            raise TaskMemorySelectionError("selection baseline is not frozen")
        update = raw["update"]
        horizon = raw["T"]
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or isinstance(horizon, bool)
            or not isinstance(horizon, int)
        ):
            raise TaskMemorySelectionError("update and T must be integers")
        normalized.append(
            {
                **dict(raw),
                "T": horizon,
                "baseline": baseline,
                "checkpoint_sha256": _sha256(raw["checkpoint_sha256"]),
                "delta_t_mAP": _finite(raw["delta_t_mAP"], name="delta_t_mAP"),
                "median_update_latency_ms": _finite(
                    raw["median_update_latency_ms"],
                    name="median_update_latency_ms",
                    nonnegative=True,
                ),
                "policy": policy,
                "reducer": reducer,
                "update": update,
                "variant": variant,
            }
        )
    identities = {(row["policy"], row["reducer"]) for row in normalized}
    if len(identities) != 1:
        raise TaskMemorySelectionError("selection requires a single policy and reducer")

    by_candidate: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    updates_by_variant: dict[str, set[int]] = defaultdict(set)
    for row in normalized:
        by_candidate[(row["variant"], row["update"])].append(row)
        updates_by_variant[row["variant"]].add(row["update"])
    required_updates = set(REQUIRED_M2_UPDATES)
    if any(updates != required_updates for updates in updates_by_variant.values()):
        raise TaskMemorySelectionError("every variant must cover all registered updates")

    expected_cells = {
        (baseline, horizon)
        for baseline in SELECTION_BASELINES
        for horizon in SELECTION_HORIZONS
    }
    candidates = []
    for (variant, update), values in by_candidate.items():
        cells = [(row["baseline"], row["T"]) for row in values]
        if len(cells) != len(set(cells)) or set(cells) != expected_cells:
            raise TaskMemorySelectionError(
                "every checkpoint must contain exact all-T baseline cells"
            )
        checkpoints = {row["checkpoint_sha256"] for row in values}
        latencies = {row["median_update_latency_ms"] for row in values}
        if len(checkpoints) != 1 or len(latencies) != 1:
            raise TaskMemorySelectionError(
                "checkpoint identity and latency must be constant per candidate"
            )
        deltas = [row["delta_t_mAP"] for row in values]
        candidates.append(
            {
                "checkpoint_sha256": next(iter(checkpoints)),
                "mean_delta_t_mAP": sum(deltas) / len(deltas),
                "median_update_latency_ms": next(iter(latencies)),
                "minimum_delta_t_mAP": min(deltas),
                "policy": values[0]["policy"],
                "reducer": values[0]["reducer"],
                "update": update,
                "variant": variant,
            }
        )
    return normalized, candidates


def select_task_memory_checkpoint(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    _, candidates = _normalized_rows(rows)
    selected = min(
        candidates,
        key=lambda value: (
            -value["minimum_delta_t_mAP"],
            -value["mean_delta_t_mAP"],
            value["median_update_latency_ms"],
            value["update"],
            value["variant"],
        ),
    )
    return {
        **selected,
        "selection_baselines": list(SELECTION_BASELINES),
        "selection_horizons": list(SELECTION_HORIZONS),
        "selection_rule": (
            "max_min_delta_then_mean_delta_then_lower_latency_then_earlier_update"
        ),
    }


def terminal_update_comparison(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, Any]]:
    normalized, _ = _normalized_rows(rows)
    return sorted(
        (row for row in normalized if row["update"] == REQUIRED_M2_UPDATES[-1]),
        key=lambda row: (row["variant"], row["baseline"], row["T"]),
    )


def selected_checkpoint_comparison(
    rows: Sequence[Mapping[str, object]], selection: Mapping[str, object]
) -> list[dict[str, Any]]:
    normalized, candidates = _normalized_rows(rows)
    identity = (selection.get("variant"), selection.get("update"))
    if identity not in {
        (candidate["variant"], candidate["update"]) for candidate in candidates
    }:
        raise TaskMemorySelectionError("selected checkpoint is not registered")
    return sorted(
        (
            row
            for row in normalized
            if (row["variant"], row["update"]) == identity
        ),
        key=lambda row: (row["baseline"], row["T"]),
    )


__all__ = [
    "REQUIRED_M2_UPDATES",
    "SELECTION_BASELINES",
    "SELECTION_HORIZONS",
    "TaskMemorySelectionError",
    "select_task_memory_checkpoint",
    "selected_checkpoint_comparison",
    "terminal_update_comparison",
]
