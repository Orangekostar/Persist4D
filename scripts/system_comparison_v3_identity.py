#!/usr/bin/env python3
"""Freshly recompute B2/B3/B4 query-level identity diagnostics for V3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_persist4d_p6a import (
    build_association_events,
    build_rio_class_mapper,
    build_tracker_factories,
    cache_payload_to_frozen_observation,
    load_cached_protocol_sequences,
    observation_content_digest,
)
from scripts.run_system_comparison import REPRODUCIBILITY_BINDING, _build_frozen_setup
from scripts.system_comparison_analysis import _persistent_identity_updates
from scripts.system_comparison_metrics import compute_deployment_identity_metrics

OUTPUT_ROOT = PROJECT_ROOT / "artifacts/reviewer_closure_v3/identity"
CACHE_ROOT = PROJECT_ROOT / "artifacts/system_comparison/persistent_predictions"
CACHE_MANIFEST = CACHE_ROOT / "manifest.json"
FROZEN_IDENTITY_CSV = (
    PROJECT_ROOT / "artifacts/system_comparison/per_sequence_results.csv"
)
FROZEN_SYSTEM_MANIFEST = (
    PROJECT_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
)
METADATA = Path.home() / "3RScan.json"
METHODS = ("B2", "B3", "B4")
ORDERS = ("canonical", "reverse", "sha256_seed45")
HORIZONS = (2, 3, 4, 5)
IDENTITY_COUNT_FIELDS = (
    "deployment_id_switches",
    "identity_transition_opportunities",
    "fragmentation_count",
    "fragmentation_opportunities",
    "merge_count",
    "merge_opportunities",
    "gap_opportunities",
    "recovery_attempts",
    "correct_recoveries",
)
IDENTITY_RATE_FIELDS = (
    "normalized_id_switch_rate",
    "fragmentation_rate",
    "merge_rate",
    "gap_recovery_accuracy",
    "gap_recovery_recall",
)
EVENT_COUNT_FIELDS = (
    "association_event_count",
    "new_birth_count",
    "false_birth_count",
    "birth_rejected_count",
)


class IdentityRecomputationError(ValueError):
    """Raised when the fresh identity evidence violates the V3 contract."""


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _issued_ids(step: object) -> tuple[int | None, ...]:
    values = _field(step, "track_ids")
    if isinstance(values, Tensor):
        if values.ndim != 1:
            raise IdentityRecomputationError("tracker issued IDs must have rank one")
        normalized = tuple(values.detach().cpu().tolist())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized = tuple(values)
    else:
        raise IdentityRecomputationError("tracker issued IDs must be a sequence")
    issued = tuple(value for value in normalized if value is not None)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in issued
    ) or len(set(issued)) != len(issued):
        raise IdentityRecomputationError(
            "tracker issued IDs must be unique non-negative integers"
        )
    return normalized


def run_fresh_tracker_steps(
    *,
    factory: Callable[[str], object],
    observations: Sequence[object],
    sequence_id: str,
) -> tuple[object, ...]:
    """Run one registered tracker causally without exposing any GT payload."""

    if not callable(factory) or not observations:
        raise IdentityRecomputationError(
            "fresh tracker run requires a factory and data"
        )
    if not isinstance(sequence_id, str) or not sequence_id:
        raise IdentityRecomputationError("fresh tracker sequence ID must be non-empty")
    tracker = factory(sequence_id)
    step_method = getattr(tracker, "step", None)
    if not callable(step_method):
        raise IdentityRecomputationError("registered tracker must expose step()")
    steps = []
    for stage, observation in enumerate(observations):
        step = step_method(observation, stage_id=stage)
        if _field(step, "stage_id") != stage:
            raise IdentityRecomputationError("fresh tracker stages must be contiguous")
        _issued_ids(step)
        steps.append(step)
    return tuple(steps)


def _non_negative_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IdentityRecomputationError(f"{field} must be a non-negative integer")
    return value


def _optional_rate(value: object, *, field: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IdentityRecomputationError(f"{field} must be a finite rate or missing")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise IdentityRecomputationError(f"{field} must be within [0, 1]")
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_identity_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, int | float | None]:
    if not rows:
        raise IdentityRecomputationError("identity aggregation requires rows")
    totals = {
        field: sum(_non_negative_count(row.get(field), field=field) for row in rows)
        for field in (*IDENTITY_COUNT_FIELDS, *EVENT_COUNT_FIELDS)
    }
    return {
        **totals,
        "normalized_id_switch_rate": _rate(
            totals["deployment_id_switches"],
            totals["identity_transition_opportunities"],
        ),
        "fragmentation_rate": _rate(
            totals["fragmentation_count"], totals["fragmentation_opportunities"]
        ),
        "merge_rate": _rate(totals["merge_count"], totals["merge_opportunities"]),
        "gap_recovery_accuracy": _rate(
            totals["correct_recoveries"], totals["recovery_attempts"]
        ),
        "gap_recovery_recall": _rate(
            totals["correct_recoveries"], totals["gap_opportunities"]
        ),
    }


def assert_identity_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_sequence_count: int = 129,
) -> dict[str, int]:
    if not rows:
        raise IdentityRecomputationError("identity coverage is empty")
    cells = []
    scopes: set[tuple[str, str, str]] = set()
    references_by_master: dict[str, set[str]] = {}
    for row in rows:
        method = str(row.get("method"))
        reference = str(row.get("reference_scene_id"))
        master = str(row.get("master_sequence_id"))
        order = str(row.get("order_id"))
        horizon = row.get("horizon")
        if method not in METHODS or order not in ORDERS or horizon not in HORIZONS:
            raise IdentityRecomputationError("identity coverage labels differ")
        scopes.add((reference, master, order))
        references_by_master.setdefault(master, set()).add(reference)
        cells.append((method, reference, master, order, int(horizon)))
        for field in IDENTITY_COUNT_FIELDS:
            _non_negative_count(row.get(field), field=field)
        for field in IDENTITY_RATE_FIELDS:
            _optional_rate(row.get(field), field=field)
    expected = {
        (method, reference, master, order, horizon)
        for reference, master, order in scopes
        for method in METHODS
        for horizon in HORIZONS
    }
    if (
        len(scopes) != expected_sequence_count
        or len(cells) != len(set(cells))
        or set(cells) != expected
        or len({scope[0] for scope in scopes}) != 6
        or {scope[2] for scope in scopes} != set(ORDERS)
        or any(len(values) != 1 for values in references_by_master.values())
    ):
        raise IdentityRecomputationError("identity coverage is not exact")
    return {
        "sequence_count": len(scopes),
        "reference_cluster_count": len({scope[0] for scope in scopes}),
        "row_count": len(rows),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise IdentityRecomputationError(f"required CSV is empty: {path}")
    return rows


def compare_b4_to_frozen(
    fresh_rows: Sequence[Mapping[str, object]],
    frozen_rows: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    fresh = {
        (
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
        ): row
        for row in fresh_rows
        if row.get("method") == "B4"
    }
    frozen = {
        (
            row["master_sequence_id"],
            row["order_id"],
            int(row["horizon"]),
        ): row
        for row in frozen_rows
        if row.get("method") == "Persist4D"
    }
    if not fresh or set(fresh) != set(frozen):
        raise IdentityRecomputationError("B4 regression key coverage differs")
    differences = []
    for key, row in fresh.items():
        old = frozen[key]
        for field in IDENTITY_COUNT_FIELDS:
            current = _non_negative_count(row.get(field), field=field)
            previous = int(old[field])
            differences.append(abs(current - previous))
        for field in IDENTITY_RATE_FIELDS:
            current = _optional_rate(row.get(field), field=field)
            previous = (
                None
                if old[field] == ""
                else _optional_rate(float(old[field]), field=field)
            )
            if current is None or previous is None:
                if current is not previous:
                    raise IdentityRecomputationError(
                        f"B4 regression missingness differs for {field}"
                    )
                differences.append(0.0)
            else:
                differences.append(abs(current - previous))
    maximum = max(differences, default=0.0)
    if maximum != 0.0:
        raise IdentityRecomputationError(
            f"B4 identity regression differs (max_abs_diff={maximum})"
        )
    return {
        "status": "pass_exact",
        "row_count": len(fresh),
        "cell_count": len(differences),
        "max_abs_diff": maximum,
    }


def _difference(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def build_b4_minus_b2_cluster_effects(
    cluster_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    selected = [row for row in cluster_rows if row.get("order_id") == "all"]
    horizons = sorted({int(row["horizon"]) for row in selected})
    result = []
    for horizon in horizons:
        rows = [row for row in selected if int(row["horizon"]) == horizon]
        references = sorted({str(row["reference_scene_id"]) for row in rows})
        if len(references) != 6:
            raise IdentityRecomputationError(
                "cluster effects require exactly six reference scenes"
            )
        index = {
            (str(row["reference_scene_id"]), str(row["method"])): row for row in rows
        }
        for reference in references:
            if any((reference, method) not in index for method in METHODS):
                raise IdentityRecomputationError(
                    "cluster effects lack complete B2/B3/B4 coverage"
                )
            b2 = index[(reference, "B2")]
            b4 = index[(reference, "B4")]
            result.append(
                {
                    "reference_scene_id": reference,
                    "horizon": horizon,
                    "b2_normalized_id_switch_rate": b2["normalized_id_switch_rate"],
                    "b4_normalized_id_switch_rate": b4["normalized_id_switch_rate"],
                    "normalized_id_switch_rate_difference": _difference(
                        b2["normalized_id_switch_rate"],
                        b4["normalized_id_switch_rate"],
                    ),
                    "b2_gap_recovery_recall": b2["gap_recovery_recall"],
                    "b4_gap_recovery_recall": b4["gap_recovery_recall"],
                    "gap_recovery_recall_difference": _difference(
                        b2["gap_recovery_recall"], b4["gap_recovery_recall"]
                    ),
                    "b2_false_birth_count": b2["false_birth_count"],
                    "b4_false_birth_count": b4["false_birth_count"],
                    "false_birth_count_difference": int(b4["false_birth_count"])
                    - int(b2["false_birth_count"]),
                }
            )
    return result


def _event_counts(events: Sequence[object], *, horizon: int) -> dict[str, int]:
    selected = [event for event in events if int(_field(event, "stage_id")) < horizon]
    return {
        "association_event_count": len(selected),
        "new_birth_count": sum(
            _field(event, "new_birth") is True for event in selected
        ),
        "false_birth_count": sum(
            _field(event, "false_birth") is True for event in selected
        ),
        "birth_rejected_count": sum(
            _field(event, "birth_rejected") is True for event in selected
        ),
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise IdentityRecomputationError("output rows must not be empty")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: "" if row.get(field) is None else row.get(field) for field in fields}
        for row in rows
    )
    return stream.getvalue().encode("utf-8")


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise IdentityRecomputationError(f"JSON root must be a mapping: {path}")
    return value


def _select(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    horizon: int,
    order: str | None = None,
    reference: str | None = None,
) -> list[Mapping[str, object]]:
    return [
        row
        for row in rows
        if row["method"] == method
        and row["horizon"] == horizon
        and (order is None or row["order_id"] == order)
        and (reference is None or row["reference_scene_id"] == reference)
    ]


def _aggregate_tables(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    references = sorted({str(row["reference_scene_id"]) for row in rows})
    aggregate = []
    clusters = []
    for order in (*ORDERS, "all"):
        order_filter = None if order == "all" else order
        for horizon in HORIZONS:
            for method in METHODS:
                selected = _select(
                    rows, method=method, horizon=horizon, order=order_filter
                )
                aggregate.append(
                    {
                        "method": method,
                        "order_id": order,
                        "horizon": horizon,
                        "sequence_count": len(selected),
                        **aggregate_identity_rows(selected),
                    }
                )
                for reference in references:
                    cluster = _select(
                        rows,
                        method=method,
                        horizon=horizon,
                        order=order_filter,
                        reference=reference,
                    )
                    if not cluster:
                        raise IdentityRecomputationError(
                            "identity cluster coverage is incomplete"
                        )
                    clusters.append(
                        {
                            "method": method,
                            "reference_scene_id": reference,
                            "order_id": order,
                            "horizon": horizon,
                            "sequence_count": len(cluster),
                            **aggregate_identity_rows(cluster),
                            "inference_unit": "reference_scene_id",
                        }
                    )
    return aggregate, clusters


def _percent(value: object) -> str:
    return "NA" if value is None else f"{100 * float(value):.3f}"


def _report(
    aggregate_rows: Sequence[Mapping[str, object]],
    effects: Sequence[Mapping[str, object]],
    *,
    regression: Mapping[str, object],
) -> str:
    index = {
        (str(row["method"]), int(row["horizon"])): row
        for row in aggregate_rows
        if row["order_id"] == "all"
    }
    lines = [
        "# Identity Recomputation",
        "",
        "## ID0 Status",
        "",
        "**PASS.** Fresh B4 query-level identity diagnostics regress exactly to",
        "the frozen V1 B4 rows. B2/B3/B4 were rerun from the same frozen V1 raw",
        "observations; no V1 identity value is copied into a V3 result row.",
        "",
        "## All-Order Pooled Diagnostics",
        "",
        "| T | Tracker | ID-switch rate | Gap recovery recall | Gap recovery accuracy | False births |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        for method in METHODS:
            row = index[(method, horizon)]
            lines.append(
                f"| T{horizon} | {method} | "
                f"{_percent(row['normalized_id_switch_rate'])} | "
                f"{_percent(row['gap_recovery_recall'])} | "
                f"{_percent(row['gap_recovery_accuracy'])} | "
                f"{row['false_birth_count']} |"
            )
    lines.extend(
        [
            "",
            "## B4 Minus B2 Cluster Effects",
            "",
            "Rates are pooled within each physical reference-scene cluster. Negative",
            "ID-switch differences favor B4; positive recovery differences favor B4.",
            "",
            "| T | Reference scene | ID-switch difference | Recovery-recall difference | False-birth difference |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| T{row['horizon']} | `{row['reference_scene_id']}` | "
            f"{_percent(row['normalized_id_switch_rate_difference'])} | "
            f"{_percent(row['gap_recovery_recall_difference'])} | "
            f"{row['false_birth_count_difference']} |"
        )
    long_gap_differences = {
        horizon: _difference(
            index[("B2", horizon)]["gap_recovery_recall"],
            index[("B4", horizon)]["gap_recovery_recall"],
        )
        for horizon in (4, 5)
    }
    supported = all(
        value is not None and value > 0 for value in long_gap_differences.values()
    )
    conclusion = (
        "Fresh V3 output supports stronger pooled long-gap recovery for B4 than B2 "
        "at both T4 and T5."
        if supported
        else "Fresh V3 output does not support a uniform pooled B4 long-gap recovery "
        "advantage over B2 at both T4 and T5."
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion,
            "The six cluster effects are descriptive robustness evidence; the 129",
            "order-units are not treated as independent observations.",
            "",
            "## Channel Contract",
            "",
            "Task channel: official ReScene task candidates plus trajectory linkage",
            "produce t-mAP/t-REC. Identity channel: registered query-level tracker",
            "decisions produce switch/recovery/fragmentation/merge diagnostics. These",
            "are separate prediction objects and are not presented as one ranked list.",
            "The identity regression uses the frozen V1 query-observation cache that",
            "generated the reference B4 rows. The later V2 task-cache inference",
            "realization is distinct and is not substituted into this regression.",
            "",
            (
                f"Frozen B4 regression cells: `{regression['cell_count']}`; maximum "
                f"absolute difference: `{regression['max_abs_diff']}`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_identity_recomputation(
    *,
    cache_root: Path = CACHE_ROOT,
    cache_manifest_path: Path = CACHE_MANIFEST,
    frozen_identity_path: Path = FROZEN_IDENTITY_CSV,
    metadata_path: Path = METADATA,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    binding = _read_json(REPRODUCIBILITY_BINDING)
    cache_manifest = _read_json(cache_manifest_path)
    setup = _build_frozen_setup(
        binding=binding,
        metadata_path=metadata_path,
        device_name=None,
    )
    sequences = load_cached_protocol_sequences(
        protocol=setup.protocol,
        cache_directory=cache_root / "entries",
        manifest_path=cache_manifest_path,
    )
    factories = build_tracker_factories(setup.p6a_config)
    if not set(METHODS) <= set(factories):
        raise IdentityRecomputationError("registered identity factories are incomplete")
    class_mapper = build_rio_class_mapper(setup.dataset)
    background_class = int(setup.p6a_config["baselines"]["b4"]["background_class"])

    per_sequence_rows: list[dict[str, object]] = []
    for sequence_index, sequence in enumerate(sequences, start=1):
        observations = tuple(
            cache_payload_to_frozen_observation(raw) for raw in sequence.payloads
        )
        cache_digest = observation_content_digest(observations)
        sequence_id = f"{sequence.master_sequence_id}:{sequence.order_id}"
        for method in METHODS:
            steps = run_fresh_tracker_steps(
                factory=factories[method],
                observations=observations,
                sequence_id=sequence_id,
            )
            updates = _persistent_identity_updates(
                payloads=sequence.payloads,
                steps=steps,
                class_mapper=class_mapper,
                background_class=background_class,
            )
            events = build_association_events(
                sequence.payloads,
                steps,
                method=method,
                reference_scene_id=sequence.reference_scene_id,
                master_sequence_id=sequence.master_sequence_id,
                order_id=sequence.order_id,
                prefix=5,
                cache_digest=cache_digest,
                background_class=background_class,
            )
            for horizon in HORIZONS:
                metrics = compute_deployment_identity_metrics(updates[:horizon])
                per_sequence_rows.append(
                    {
                        "method": method,
                        "reference_scene_id": sequence.reference_scene_id,
                        "master_sequence_id": sequence.master_sequence_id,
                        "order_id": sequence.order_id,
                        "horizon": horizon,
                        **metrics,
                        **_event_counts(events, horizon=horizon),
                        "tracker_input": "frozen_query_observation_no_gt",
                        "identity_matching": "post_prediction_hungarian_iou_0.5",
                        "cache_digest": cache_digest,
                    }
                )
        if sequence_index % 10 == 0 or sequence_index == len(sequences):
            print(
                f"[identity] completed {sequence_index}/{len(sequences)} sequences",
                file=sys.stderr,
                flush=True,
            )

    per_sequence_rows.sort(
        key=lambda row: (
            str(row["reference_scene_id"]),
            str(row["master_sequence_id"]),
            str(row["order_id"]),
            int(row["horizon"]),
            str(row["method"]),
        )
    )
    coverage = assert_identity_coverage(per_sequence_rows)
    regression = compare_b4_to_frozen(
        per_sequence_rows,
        _read_csv(frozen_identity_path),
    )
    aggregate_rows, cluster_rows = _aggregate_tables(per_sequence_rows)
    effects = build_b4_minus_b2_cluster_effects(cluster_rows)
    if len(aggregate_rows) != 48 or len(cluster_rows) != 288 or len(effects) != 24:
        raise IdentityRecomputationError("identity aggregate coverage differs")

    outputs = {
        "identity_per_sequence.csv": _csv_bytes(per_sequence_rows),
        "identity_aggregate.csv": _csv_bytes(aggregate_rows),
        "identity_per_cluster.csv": _csv_bytes(cluster_rows),
    }
    report = _report(
        aggregate_rows,
        effects,
        regression=regression,
    ).encode("utf-8")
    outputs["IDENTITY_RECOMPUTATION.md"] = report
    for name, content in outputs.items():
        _write(output_root / name, content)

    aggregate_index = {
        (str(row["method"]), int(row["horizon"])): row
        for row in aggregate_rows
        if row["order_id"] == "all"
    }
    long_gap = {
        f"T{horizon}": {
            "b2_gap_recovery_recall": aggregate_index[("B2", horizon)][
                "gap_recovery_recall"
            ],
            "b4_gap_recovery_recall": aggregate_index[("B4", horizon)][
                "gap_recovery_recall"
            ],
            "b4_minus_b2": _difference(
                aggregate_index[("B2", horizon)]["gap_recovery_recall"],
                aggregate_index[("B4", horizon)]["gap_recovery_recall"],
            ),
        }
        for horizon in (4, 5)
    }
    long_gap_supported = all(
        value["b4_minus_b2"] is not None and value["b4_minus_b2"] > 0
        for value in long_gap.values()
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "pass",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": source_commit,
        "gate_id0": {
            "status": "PASS",
            "b4_frozen_regression": regression,
            "fresh_tracker_coverage": "pass_exact",
            "copied_v1_identity_fields": False,
            "long_gap_comparison": long_gap,
            "b4_long_gap_supported_at_t4_and_t5": long_gap_supported,
        },
        "coverage": {
            "sequence_count": coverage["sequence_count"],
            "reference_cluster_count": coverage["reference_cluster_count"],
            "methods": list(METHODS),
            "orders": list(ORDERS),
            "horizons": list(HORIZONS),
            "per_sequence_row_count": len(per_sequence_rows),
            "aggregate_row_count": len(aggregate_rows),
            "per_cluster_row_count": len(cluster_rows),
            "cluster_effect_row_count": len(effects),
        },
        "inputs": {
            "cache_manifest": {
                "reference": "repo:artifacts/system_comparison/persistent_predictions/manifest.json",
                "sha256": _sha256(cache_manifest_path),
            },
            "cache_entries_sha256": cache_manifest["entries_sha256"],
            "frozen_identity_csv": {
                "reference": "repo:artifacts/system_comparison/per_sequence_results.csv",
                "sha256": _sha256(frozen_identity_path),
                "use": "regression_reference_only",
            },
            "frozen_system_manifest": {
                "reference": "repo:artifacts/system_comparison/system_comparison_manifest.json",
                "sha256": _sha256(FROZEN_SYSTEM_MANIFEST),
            },
            "checkpoint_sha256": cache_manifest["provenance"]["checkpoint_sha256"],
            "observation_config_sha256": cache_manifest["provenance"]["config_sha256"],
            "dataset_sha256": cache_manifest["provenance"]["dataset_sha256"],
            "tracker_system_config_sha256": binding["config_sha256"],
            "protocol_manifest_sha256": binding["protocol_sha256"],
        },
        "execution": {
            "mode": "fresh_registered_trackers_on_frozen_query_observations",
            "gpu_inference_performed": False,
            "gt_available_to_tracker": False,
            "gt_used_post_prediction_for_identity_matching": True,
            "identity_iou_threshold": 0.5,
            "regression_input": "frozen_v1_query_observation_cache",
            "v2_task_cache_used_for_identity": False,
        },
        "outputs": {
            name: {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in outputs.items()
        },
        "scripts": {
            "identity_sha256": _sha256(Path(__file__)),
            "test_sha256": _sha256(
                PROJECT_ROOT / "tests/test_system_comparison_v3_identity.py"
            ),
        },
        "channel_contract": {
            "task": "official task candidates plus trajectory linkage -> t-mAP/t-REC",
            "identity": "registered query-level tracker decisions -> identity diagnostics",
            "task_and_identity_are_separate": True,
        },
    }
    _write(
        output_root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "status": "pass",
        "gate": "ID0",
        "per_sequence_rows": len(per_sequence_rows),
        "aggregate_rows": len(aggregate_rows),
        "per_cluster_rows": len(cluster_rows),
        "b4_regression_max_abs_diff": regression["max_abs_diff"],
        "b4_long_gap_supported_at_t4_and_t5": long_gap_supported,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--cache-manifest", type=Path, default=CACHE_MANIFEST)
    parser.add_argument("--frozen-identity", type=Path, default=FROZEN_IDENTITY_CSV)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    print(
        json.dumps(
            run_identity_recomputation(
                cache_root=arguments.cache_root,
                cache_manifest_path=arguments.cache_manifest,
                frozen_identity_path=arguments.frozen_identity,
                metadata_path=arguments.metadata,
                output_root=arguments.output_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
