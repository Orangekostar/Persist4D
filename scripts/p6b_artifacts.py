"""Strict, deterministic artifact contract for Persist4D P6-B protocol v2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml

from scripts.p6b_figures import render_identity_figure, render_reactivation_figure

P6B_ARTIFACT_SCHEMA_VERSION = 2

_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "status",
        "decision",
        "source_commit",
        "source_tree_contract",
        "source_lineage",
        "selection_document",
        "heldout_attempt",
        "heldout_raw",
        "gate_results",
        "inactive_components",
        "protocol_deviations",
        "claims_supported",
        "claims_not_supported",
        "next_action",
        "artifact_manifest",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "attempt_id",
        "source_commit",
        "selection",
        "split_sha256",
        "p6b_config_sha256",
        "command",
        "input_sha256",
        "status",
        "started_utc",
        "ended_utc",
        "exit_status",
        "error_type",
        "events",
        "log_sha256",
        "output",
    }
)
_RAW_KEYS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "attempt_id",
        "source_commit",
        "selection",
        "split_sha256",
        "p6b_config_sha256",
        "evaluation",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "status",
        "heldout_order_count",
        "heldout_reference_scene_ids",
        "selected_config_id",
        "selected_config_sha256",
        "provenance",
        "final_results",
        "official_metric_evidence",
        "per_sequence_results",
        "failure_analysis",
        "failure_diagnostics",
        "statistical_analysis",
    }
)
_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint",
        "p5",
        "p6a",
        "p6a_protocol_manifest",
        "p6a_cache_manifest",
    }
)
_OFFICIAL_METRIC_IDENTITY_FIELDS = (
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "prediction_digest",
)
_STAGES = (
    "assignment",
    "reactivation",
    "class_compatibility",
    "consolidation",
    "birth_gate",
    "joint_neighbors",
)
_STAGE_PATHS = {
    "assignment": "assignment_ablation.csv",
    "reactivation": "reactivation_threshold_sweep.csv",
    "class_compatibility": "class_compatibility_ablation.csv",
    "consolidation": "consolidation_ablation.csv",
    "birth_gate": "birth_gate_sweep.csv",
    "joint_neighbors": "joint_validation_sweep.csv",
}
_SWEEP_COLUMNS = (
    "config_id",
    "config_json",
    "stage",
    "T",
    "identity_switches",
    "transition_opportunities",
    "identity_switch_rate",
    "wrong_reactivations",
    "predicted_reactivation_events",
    "correct_reactivations",
    "reactivation_attempts",
    "gap_opportunities",
    "wrong_reactivation_rate",
    "false_births",
    "true_births",
    "births",
    "accepted_births",
    "rejected_births",
    "valid_birth_opportunities",
    "false_birth_rate",
    "cluster_metrics_json",
    "reactivation_accuracy",
    "reactivation_recall",
    "accepted_valid_observations",
    "total_valid_observations",
    "frozen_b4_valid_observations",
    "strict_online_tmap",
    "strict_online_trec",
    "full_eligible",
    "stage_eligible",
    "eligibility_reasons",
)
_FINAL_COLUMNS = (
    "method",
    "T",
    "t_mAP",
    "t_REC",
    "identity_switches",
    "transition_opportunities",
    "identity_switch_rate",
    "wrong_reactivations",
    "predicted_reactivation_events",
    "wrong_reactivation_rate",
    "correct_reactivations",
    "reactivation_attempts",
    "gap_opportunities",
    "reactivation_accuracy",
    "reactivation_recall",
    "false_births",
    "births",
    "rejected_births",
    "false_birth_rate",
)
_PER_SEQUENCE_COLUMNS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "prediction_digest",
    "t_mAP",
    "t_REC",
    "identity_switches",
    "transition_opportunities",
    "identity_switch_rate",
    "wrong_reactivations",
    "predicted_reactivation_events",
    "wrong_reactivation_rate",
    "correct_reactivations",
    "reactivation_attempts",
    "gap_opportunities",
    "reactivation_accuracy",
    "reactivation_recall",
    "false_births",
    "births",
    "rejected_births",
    "false_birth_rate",
)
_FAILURE_COLUMNS = ("method", "T", "failure_category", "count")
_FAILURE_DIAGNOSTIC_COLUMNS = (
    "method",
    "reference_scene_id",
    "master_sequence_id",
    "order_id",
    "T",
    "prediction_digest",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "unclassified",
)
_REPORT_SECTIONS = (
    "1. What was changed",
    "2. Why it was changed",
    "3. Experimental protocol",
    "4. Reproducibility binding",
    "5. Main results",
    "6. Statistical evidence",
    "7. Failure analysis",
    "8. What claims are supported",
    "9. What claims are NOT supported",
    "10. GO / NO-GO decision",
    "11. Exact next action",
)
_PRIVATE_PATH = re.compile(r"(?:/home/|/Users/|/mnt/|[A-Za-z]:[\\/]Users[\\/])")
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_METHODS = ("B4", "P6B")
_HORIZONS = ("T2", "T3", "T4", "T5")
_FAILURES = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")


def _exact_keys(
    value: object, expected: frozenset[str] | set[str], *, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if set(value) != set(expected):
        raise ValueError(f"{name} keys differ from the schema")
    return value


def _finite_tree(value: object, *, name: str = "artifact") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, name=f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _finite_tree(item, name=f"{name}[{index}]")


def _plain_ref(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.startswith(("repo:", "external:")):
        raise ValueError(f"{name} must be a portable repo: or external: reference")
    if _PRIVATE_PATH.search(value):
        raise ValueError(f"{name} must be portable")
    return value


def _sha(value: object, *, length: int, name: str) -> str:
    pattern = _SHA40 if length == 40 else _SHA64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} is invalid")
    return value


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _utc_timestamp(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a UTC timestamp")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a UTC timestamp") from error
    if result.tzinfo != timezone.utc:
        raise ValueError(f"{name} must be a UTC timestamp")
    return result


def _rows(
    value: object,
    columns: Sequence[str],
    *,
    name: str,
    expected_count: int,
) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    if len(value) != expected_count:
        raise ValueError(f"{name} population differs from protocol v2")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping) or set(row) != set(columns):
            raise ValueError(f"{name}[{index}] columns differ from the schema")
        result.append(row)
    return tuple(result)


def _count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(value: object, *, name: str, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (unit and not 0.0 <= result <= 1.0):
        raise ValueError(f"{name} must be finite" + (" in [0, 1]" if unit else ""))
    return result


def _rate(value: object, numerator: int, denominator: int, *, name: str) -> None:
    if numerator > denominator:
        raise ValueError(f"{name} numerator exceeds denominator")
    expected = numerator / denominator if denominator else None
    if expected is None:
        if value is not None:
            raise ValueError(f"{name} must be null for a zero denominator")
        return
    actual = _number(value, name=name, unit=True)
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} differs from its count-derived rate")


def _validate_selection(selection: object) -> Mapping[str, object]:
    if not isinstance(selection, Mapping):
        raise TypeError("selection_document must be a mapping")
    from scripts.run_p6b_evaluation import _validate_selection_document

    _validate_selection_document(selection)
    return selection


def _validate_attempt_raw(
    root: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    attempt = _exact_keys(
        root["heldout_attempt"], _ATTEMPT_KEYS, name="heldout_attempt"
    )
    raw = _exact_keys(root["heldout_raw"], _RAW_KEYS, name="heldout_raw")
    if (
        attempt["schema_version"] != 1
        or attempt["protocol_version"] != 2
        or attempt["status"] != "success"
        or attempt["exit_status"] != 0
        or attempt["error_type"] is not None
    ):
        raise ValueError("heldout_attempt is not a successful protocol-v2 attempt")
    inputs = {
        key: attempt[key]
        for key in (
            "source_commit",
            "selection",
            "split_sha256",
            "p6b_config_sha256",
            "command",
        )
    }
    expected_input_sha = hashlib.sha256(_json_bytes(inputs)).hexdigest()
    if attempt["input_sha256"] != expected_input_sha:
        raise ValueError("heldout attempt input SHA-256 differs")
    started = _utc_timestamp(attempt["started_utc"], name="started_utc")
    ended = _utc_timestamp(attempt["ended_utc"], name="ended_utc")
    if ended < started:
        raise ValueError("heldout attempt ended before it started")
    expected_attempt_id = hashlib.sha256(
        _json_bytes({**inputs, "started_utc": attempt["started_utc"]})
    ).hexdigest()
    attempt_id = _sha(attempt["attempt_id"], length=64, name="attempt_id")
    if attempt_id != expected_attempt_id:
        raise ValueError("heldout attempt ID differs from canonical inputs")
    if raw["schema_version"] != 1 or raw["protocol_version"] != 2:
        raise ValueError("heldout_raw schema/protocol differs")
    if raw["attempt_id"] != attempt_id:
        raise ValueError("heldout raw attempt ID differs")
    for key in ("source_commit", "selection", "split_sha256", "p6b_config_sha256"):
        if raw[key] != attempt[key]:
            raise ValueError(f"heldout raw {key} differs from attempt")
    _sha(raw["source_commit"], length=40, name="heldout source_commit")
    _sha(raw["split_sha256"], length=64, name="heldout split_sha256")
    _sha(raw["p6b_config_sha256"], length=64, name="heldout config SHA")
    selection_ref = _exact_keys(
        raw["selection"], {"ref", "sha256"}, name="heldout selection"
    )
    if selection_ref["ref"] != "repo:artifacts/P6B_selection/selection.json":
        raise ValueError("heldout selection reference differs")
    _sha(selection_ref["sha256"], length=64, name="heldout selection SHA")
    output = _exact_keys(
        attempt["output"], {"ref", "bytes", "sha256"}, name="heldout output"
    )
    if output["ref"] != "repo:artifacts/P6B_heldout/heldout_raw.json":
        raise ValueError("heldout raw reference differs")
    raw_bytes = _json_bytes(raw)
    if (
        output["bytes"] != len(raw_bytes)
        or output["sha256"] != hashlib.sha256(raw_bytes).hexdigest()
    ):
        raise ValueError("heldout raw manifest binding differs")
    if (
        not isinstance(attempt["command"], Sequence)
        or isinstance(attempt["command"], (str, bytes))
        or not attempt["command"]
        or any(not isinstance(item, str) or not item for item in attempt["command"])
    ):
        raise ValueError("heldout command is invalid")
    events = attempt["events"]
    if not isinstance(events, list) or len(events) != 2:
        raise ValueError("heldout attempt event ledger is incomplete")
    expected_event_names = {
        "heldout_raw_published",
        "heldout_raw_recovered",
    }
    if (
        events[0]
        != {
            "event": "attempt_token_published",
            "utc": attempt["started_utc"],
        }
        or not isinstance(events[1], Mapping)
        or set(events[1]) != {"event", "utc"}
        or events[1]["event"] not in expected_event_names
        or events[1]["utc"] != attempt["ended_utc"]
    ):
        raise ValueError("heldout attempt event ledger differs from execution state")
    if attempt["log_sha256"] != hashlib.sha256(_json_bytes(events)).hexdigest():
        raise ValueError("heldout attempt event-log SHA-256 differs")
    return attempt, raw


def _validate_provenance(value: object) -> Mapping[str, object]:
    provenance = _exact_keys(value, _PROVENANCE_KEYS, name="provenance")
    for key, raw_record in provenance.items():
        record = _exact_keys(raw_record, {"ref", "sha256"}, name=f"provenance.{key}")
        _plain_ref(record["ref"], name=f"provenance.{key}.ref")
        _sha(record["sha256"], length=64, name=f"provenance.{key}.sha256")
    return provenance


def _validate_association_count_identities(
    *,
    switches: int,
    transitions: int,
    wrong: int,
    predicted: int,
    correct: int,
    attempts: int,
    gaps: int,
    false_births: int,
    births: int,
) -> None:
    if switches > transitions:
        raise ValueError("identity switches exceed transition opportunities")
    if not correct <= attempts <= gaps:
        raise ValueError("reactivation attempts exceed gap opportunities")
    if correct > predicted:
        raise ValueError("correct reactivations exceed predicted events")
    if wrong != predicted - correct:
        raise ValueError("wrong reactivations differ from predicted minus correct")
    if false_births > births:
        raise ValueError("false births exceed accepted births")


def _validate_per_sequence(
    evaluation: Mapping[str, object],
    expected_units: set[tuple[str, str, str]],
) -> tuple[Mapping[str, object], ...]:
    rows = _rows(
        evaluation["per_sequence_results"],
        _PER_SEQUENCE_COLUMNS,
        name="per_sequence_results",
        expected_count=264,
    )
    keys: set[tuple[object, ...]] = set()
    units: set[tuple[object, ...]] = set()
    pair_digests: dict[tuple[object, ...], set[object]] = {}
    for row in rows:
        method = row["method"]
        horizon = row["T"]
        reference = row["reference_scene_id"]
        unit = (reference, row["master_sequence_id"], row["order_id"])
        if (
            method not in _METHODS
            or horizon not in _HORIZONS
            or unit not in expected_units
        ):
            raise ValueError(
                "per_sequence_results identity is outside held-out protocol"
            )
        key = (
            method,
            reference,
            row["master_sequence_id"],
            row["order_id"],
            horizon,
        )
        if key in keys:
            raise ValueError("per_sequence_results contains a duplicate row")
        keys.add(key)
        units.add(unit)
        pair_key = key[1:]
        pair_digests.setdefault(pair_key, set()).add(row["prediction_digest"])
        if (
            not isinstance(row["prediction_digest"], str)
            or not row["prediction_digest"]
        ):
            raise ValueError("per-sequence prediction digest is invalid")
        _number(row["t_mAP"], name="t_mAP", unit=True)
        _number(row["t_REC"], name="t_REC", unit=True)
        switches = _count(row["identity_switches"], name="identity_switches")
        transitions = _count(
            row["transition_opportunities"], name="transition_opportunities"
        )
        wrong = _count(row["wrong_reactivations"], name="wrong_reactivations")
        predicted = _count(
            row["predicted_reactivation_events"],
            name="predicted_reactivation_events",
        )
        correct = _count(row["correct_reactivations"], name="correct_reactivations")
        attempts = _count(row["reactivation_attempts"], name="reactivation_attempts")
        gaps = _count(row["gap_opportunities"], name="gap_opportunities")
        false_births = _count(row["false_births"], name="false_births")
        births = _count(row["births"], name="births")
        rejected = _count(row["rejected_births"], name="rejected_births")
        _validate_association_count_identities(
            switches=switches,
            transitions=transitions,
            wrong=wrong,
            predicted=predicted,
            correct=correct,
            attempts=attempts,
            gaps=gaps,
            false_births=false_births,
            births=births,
        )
        _rate(
            row["identity_switch_rate"],
            switches,
            transitions,
            name="identity_switch_rate",
        )
        _rate(
            row["wrong_reactivation_rate"],
            wrong,
            predicted,
            name="wrong_reactivation_rate",
        )
        _rate(
            row["reactivation_accuracy"],
            correct,
            attempts,
            name="reactivation_accuracy",
        )
        _rate(row["reactivation_recall"], correct, gaps, name="reactivation_recall")
        _rate(
            row["false_birth_rate"],
            false_births,
            births + rejected,
            name="false_birth_rate",
        )
    if units != expected_units:
        raise ValueError(
            "per_sequence_results held-out population differs from split assignments"
        )
    for unit in units:
        for method in _METHODS:
            for horizon in _HORIZONS:
                if (method, *unit, horizon) not in keys:
                    raise ValueError(
                        "per_sequence_results has a missing method/horizon row"
                    )
    if any(len(digests) != 1 for digests in pair_digests.values()):
        raise ValueError("paired methods use different frozen prediction digests")
    return rows


def _validate_final(
    evaluation: Mapping[str, object], per_sequence: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    rows = _rows(
        evaluation["final_results"],
        _FINAL_COLUMNS,
        name="final_results",
        expected_count=8,
    )
    expected_order = [(method, horizon) for method in _METHODS for horizon in _HORIZONS]
    if [(row["method"], row["T"]) for row in rows] != expected_order:
        raise ValueError("final_results must contain ordered exact B4/P6B T2-T5 rows")
    count_fields = (
        "identity_switches",
        "transition_opportunities",
        "wrong_reactivations",
        "predicted_reactivation_events",
        "correct_reactivations",
        "reactivation_attempts",
        "gap_opportunities",
        "false_births",
        "births",
        "rejected_births",
    )
    for row in rows:
        _number(row["t_mAP"], name="final t_mAP", unit=True)
        _number(row["t_REC"], name="final t_REC", unit=True)
        scoped = [
            item
            for item in per_sequence
            if item["method"] == row["method"] and item["T"] == row["T"]
        ]
        for field in count_fields:
            expected = sum(_count(item[field], name=field) for item in scoped)
            if row[field] != expected:
                raise ValueError(f"final_results {field} differs from per-sequence sum")
        _validate_association_count_identities(
            switches=int(row["identity_switches"]),
            transitions=int(row["transition_opportunities"]),
            wrong=int(row["wrong_reactivations"]),
            predicted=int(row["predicted_reactivation_events"]),
            correct=int(row["correct_reactivations"]),
            attempts=int(row["reactivation_attempts"]),
            gaps=int(row["gap_opportunities"]),
            false_births=int(row["false_births"]),
            births=int(row["births"]),
        )
        _rate(
            row["identity_switch_rate"],
            int(row["identity_switches"]),
            int(row["transition_opportunities"]),
            name="final identity_switch_rate",
        )
        _rate(
            row["wrong_reactivation_rate"],
            int(row["wrong_reactivations"]),
            int(row["predicted_reactivation_events"]),
            name="final wrong_reactivation_rate",
        )
        _rate(
            row["reactivation_accuracy"],
            int(row["correct_reactivations"]),
            int(row["reactivation_attempts"]),
            name="final reactivation_accuracy",
        )
        _rate(
            row["reactivation_recall"],
            int(row["correct_reactivations"]),
            int(row["gap_opportunities"]),
            name="final reactivation_recall",
        )
        _rate(
            row["false_birth_rate"],
            int(row["false_births"]),
            int(row["births"]) + int(row["rejected_births"]),
            name="final false_birth_rate",
        )
    return rows


def _validate_official_metric_evidence(
    evaluation: Mapping[str, object], final_rows: Sequence[Mapping[str, object]]
) -> None:
    evidence = evaluation["official_metric_evidence"]
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("official metric evidence must be a sequence")
    expected_order = [(method, horizon) for method in _METHODS for horizon in _HORIZONS]
    if len(evidence) != len(expected_order):
        raise ValueError("official metric evidence population differs")
    final_by_key = {(row["method"], row["T"]): row for row in final_rows}
    for record, key in zip(evidence, expected_order, strict=True):
        if not isinstance(record, Mapping) or set(record) != {"method", "T", "state"}:
            raise ValueError("official metric evidence record differs from schema")
        if (record["method"], record["T"]) != key:
            raise ValueError("official metric evidence order differs")
        state = record["state"]
        if (
            not isinstance(state, Mapping)
            or state.get("mode") != "strict_online"
            or state.get("updates") != evaluation["heldout_order_count"]
        ):
            raise ValueError("official metric evidence scope differs")
        computed, population_records, per_sequence_metrics = (
            _recompute_official_metric_population(
                json.dumps(state, sort_keys=True, separators=(",", ":"))
            )
        )
        actual_population = [
            {
                field: row[field]
                for field in (
                    "reference_scene_id",
                    "master_sequence_id",
                    "order_id",
                    "prediction_digest",
                )
            }
            for row in population_records
        ]
        expected_population = sorted(
            (
                {
                    field: row[field]
                    for field in (
                        "reference_scene_id",
                        "master_sequence_id",
                        "order_id",
                        "prediction_digest",
                    )
                }
                for row in evaluation["per_sequence_results"]
                if row["method"] == key[0] and row["T"] == key[1]
            ),
            key=lambda row: tuple(row.values()),
        )
        if actual_population != expected_population:
            raise ValueError(
                "official metric evidence differs from held-out identity population"
            )
        expected_by_identity = {
            tuple(row[field] for field in _OFFICIAL_METRIC_IDENTITY_FIELDS): row
            for row in evaluation["per_sequence_results"]
            if row["method"] == key[0] and row["T"] == key[1]
        }
        for population_record, sequence_metrics in zip(
            population_records, per_sequence_metrics, strict=True
        ):
            identity = tuple(
                population_record[field]
                for field in _OFFICIAL_METRIC_IDENTITY_FIELDS
            )
            sequence_row = expected_by_identity[identity]
            for output_key, row_key in (
                ("online_t-mAP", "t_mAP"),
                ("online_t-REC", "t_REC"),
            ):
                if not math.isclose(
                    float(sequence_metrics[output_key]),
                    float(sequence_row[row_key]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "per-sequence official metric differs from sufficient state"
                    )
        final = final_by_key[key]
        for output_key, final_key in (
            ("online_t-mAP", "t_mAP"),
            ("online_t-REC", "t_REC"),
        ):
            if not math.isclose(
                float(computed[output_key]),
                float(final[final_key]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "final official metric differs from official metric evidence"
                )


@lru_cache(maxsize=64)
def _recompute_official_metric_population(
    canonical_state_json: str,
) -> tuple[
    dict[str, float],
    tuple[dict[str, object], ...],
    tuple[dict[str, float], ...],
]:
    from scripts.p6a_metrics import recompute_official_metric_population_evidence

    state = json.loads(canonical_state_json)
    return recompute_official_metric_population_evidence(state)


def _validate_failure_diagnostics(
    evaluation: Mapping[str, object],
    per_sequence: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows = _rows(
        evaluation["failure_diagnostics"],
        _FAILURE_DIAGNOSTIC_COLUMNS,
        name="failure_diagnostics",
        expected_count=264,
    )
    expected = {
        (
            row["method"],
            row["reference_scene_id"],
            row["master_sequence_id"],
            row["order_id"],
            row["T"],
        ): row["prediction_digest"]
        for row in per_sequence
    }
    actual: dict[tuple[object, ...], object] = {}
    for row in rows:
        key = (
            row["method"],
            row["reference_scene_id"],
            row["master_sequence_id"],
            row["order_id"],
            row["T"],
        )
        if key in actual:
            raise ValueError("failure_diagnostics contains a duplicate row")
        actual[key] = row["prediction_digest"]
        for category in _FAILURES:
            _count(row[category], name=f"failure diagnostic {category}")
    if actual != expected:
        raise ValueError(
            "failure_diagnostics population/digest differs from paired rows"
        )
    return rows


def _validate_failure_analysis(
    evaluation: Mapping[str, object],
    diagnostics: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    rows = _rows(
        evaluation["failure_analysis"],
        _FAILURE_COLUMNS,
        name="failure_analysis",
        expected_count=64,
    )
    expected = {
        (method, horizon, failure)
        for method in _METHODS
        for horizon in _HORIZONS
        for failure in _FAILURES
    }
    actual = set()
    for row in rows:
        key = (row["method"], row["T"], row["failure_category"])
        if key in actual:
            raise ValueError("failure_analysis contains a duplicate row")
        actual.add(key)
        _count(row["count"], name="failure count")
    if actual != expected:
        raise ValueError("failure_analysis population differs from protocol v2")
    derived = {
        (method, horizon, failure): sum(
            int(row[failure])
            for row in diagnostics
            if row["method"] == method and row["T"] == horizon
        )
        for method, horizon, failure in expected
    }
    if any(
        row["count"] != derived[(row["method"], row["T"], row["failure_category"])]
        for row in rows
    ):
        raise ValueError("failure_analysis differs from failure diagnostics")
    return rows


def _validate_statistics(
    evaluation: Mapping[str, object], per_sequence: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    value = evaluation["statistical_analysis"]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("statistical_analysis must be a sequence")
    if len(value) != 25 or any(not isinstance(item, Mapping) for item in value):
        raise ValueError("statistical_analysis population differs from protocol v2")
    from scripts.run_p6b_evaluation import _paired_statistics

    expected = _paired_statistics(per_sequence)
    if list(value) != expected:
        raise ValueError(
            "statistical_analysis differs from paired bootstrap recomputation"
        )
    return tuple(value)


def _inactive_components(selection: Mapping[str, object]) -> list[str]:
    config = selection["selected_config"]
    if not isinstance(config, Mapping):
        raise TypeError("selected_config must be a mapping")
    inactive = []
    if config.get("assignment_mode") != "threshold_aware":
        inactive.append("threshold_aware_assignment")
    if config.get("reactivation_margin") is None:
        inactive.append("dormant_reactivation_margin")
    if config.get("class_mode") != "foreground_normalized":
        inactive.append("foreground_normalized_class_compatibility")
    if config.get("consolidation_confidence") is None:
        inactive.append("confidence_gated_consolidation")
    if config.get("birth_max_entropy") is None:
        inactive.append("birth_entropy_gate")
    return inactive


def _derived_narrative(
    gates: Mapping[str, Mapping[str, object]], decision: str
) -> dict[str, object]:
    expected_decision = (
        "P6B_GO"
        if all(bool(record["passed"]) for record in gates.values())
        else "P6B_STOP"
    )
    if decision != expected_decision:
        raise ValueError("narrative decision differs from gate results")
    if decision == "P6B_GO":
        supported = [
            "P6-B satisfies all preregistered held-out gates under frozen local predictions."
        ]
        next_action = "Freeze P6-B; P7 remains a separate preregistered decision and is not started."
    else:
        supported = [
            "No P6-B GO claim is supported because one or more preregistered held-out gates failed."
        ]
        next_action = (
            "Stop after P6-B and analyze failed held-out gates; do not start P7/P8."
        )
    return {
        "claims_supported": supported,
        "claims_not_supported": [
            "P6-B does not establish SOTA, retraining gains, P7, or P8 claims."
        ],
        "next_action": next_action,
    }


def _validate_base(root: Mapping[str, object]) -> None:
    _exact_keys(root, _ROOT_KEYS, name="P6-B root")
    if (
        root["schema_version"] != P6B_ARTIFACT_SCHEMA_VERSION
        or root["protocol_version"] != 2
        or root["status"] != "pass"
    ):
        raise ValueError("P6-B schema/protocol/status differs")
    source_commit = _sha(root["source_commit"], length=40, name="source_commit")
    if root["source_tree_contract"] != {
        "status": "pass",
        "source_commit": source_commit,
    }:
        raise ValueError("source_tree_contract differs from source_commit")
    selection = _validate_selection(root["selection_document"])
    attempt, raw = _validate_attempt_raw(root)
    lineage = _exact_keys(
        root["source_lineage"],
        {
            "schema_version",
            "selection_source_commit",
            "evaluation_source_commit",
            "package_source_commit",
            "selection_to_evaluation_allowed_prefix",
            "evaluation_to_package_allowed_prefix",
        },
        name="source lineage",
    )
    if lineage != {
        "schema_version": 1,
        "selection_source_commit": selection["source_commit"],
        "evaluation_source_commit": raw["source_commit"],
        "package_source_commit": source_commit,
        "selection_to_evaluation_allowed_prefix": "artifacts/P6B_selection/",
        "evaluation_to_package_allowed_prefix": "artifacts/P6B_heldout/",
    }:
        raise ValueError("source lineage differs from embedded documents")
    selection_sha = hashlib.sha256(_json_bytes(selection)).hexdigest()
    if raw["selection"]["sha256"] != selection_sha:
        raise ValueError("heldout raw selection SHA differs from embedded selection")
    if raw["split_sha256"] != selection["split_manifest"]["sha256"]:
        raise ValueError("heldout raw split differs from embedded selection")
    if raw["p6b_config_sha256"] != selection["provenance"]["p6b_config_sha256"]:
        raise ValueError("heldout raw config differs from embedded selection")
    evaluation = _exact_keys(raw["evaluation"], _EVALUATION_KEYS, name="evaluation")
    if evaluation["status"] != "pass" or evaluation["heldout_order_count"] != 33:
        raise ValueError("heldout evaluation status/population differs")
    expected_units = {
        (
            assignment["reference_scene_id"],
            assignment["master_sequence_id"],
            order_id,
        )
        for assignment in selection["split_manifest"]["assignments"]
        if assignment["partition"] == "heldout"
        for order_id in assignment["order_ids"]
    }
    if len(expected_units) != 33:
        raise ValueError("selection split must contain exactly 33 held-out orders")
    if evaluation["heldout_reference_scene_ids"] != list(
        selection["split_manifest"]["heldout_reference_scene_ids"]
    ):
        raise ValueError("heldout evaluation references differ from selection")
    if (
        evaluation["selected_config_id"] != selection["selected_config_id"]
        or evaluation["selected_config_sha256"] != selection["selected_config_sha256"]
    ):
        raise ValueError("heldout evaluation selected config differs")
    provenance = _validate_provenance(evaluation["provenance"])
    per_sequence = _validate_per_sequence(evaluation, expected_units)
    final_rows = _validate_final(evaluation, per_sequence)
    _validate_official_metric_evidence(evaluation, final_rows)
    failure_diagnostics = _validate_failure_diagnostics(evaluation, per_sequence)
    _validate_failure_analysis(evaluation, failure_diagnostics)
    statistics = _validate_statistics(evaluation, per_sequence)
    from scripts.p6b_protocol import load_p6b_config
    from scripts.run_p6b_evaluation import (
        EXPECTED_CHECKPOINT_SHA256,
        EXPECTED_P5_SHA256,
        EXPECTED_P6A_SHA256,
        PROJECT_ROOT,
        compute_final_gate_results,
    )

    protocol = load_p6b_config(PROJECT_ROOT / "conf/p6b/default.yaml")
    expected_provenance = {
        "checkpoint": {
            "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
            "sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "p5": {
            "ref": "repo:artifacts/P5/persist4d_mvp_eval.json",
            "sha256": EXPECTED_P5_SHA256,
        },
        "p6a": {
            "ref": "repo:artifacts/P6A/p6a_eval.json",
            "sha256": EXPECTED_P6A_SHA256,
        },
        "p6a_protocol_manifest": {
            "ref": protocol.sources.p6a_protocol_manifest.reference,
            "sha256": protocol.sources.p6a_protocol_manifest.sha256,
        },
        "p6a_cache_manifest": {
            "ref": protocol.sources.p6a_cache_manifest.reference,
            "sha256": protocol.sources.p6a_cache_manifest.sha256,
        },
    }
    frozen = provenance == expected_provenance
    if not frozen:
        raise ValueError("heldout provenance differs from frozen P6-A/base provenance")
    from scripts.run_p6b_evaluation import _validate_verification_ledger

    verification_passed = _validate_verification_ledger(
        selection["verification_ledger"]
    )
    deviations = root["protocol_deviations"]
    if not isinstance(deviations, list) or any(
        not isinstance(item, str) or not item for item in deviations
    ):
        raise ValueError("protocol_deviations must be a list of nonempty strings")
    expected_gates = compute_final_gate_results(
        final_rows,
        evidence_complete=not deviations,
        frozen_hashes_unchanged=frozen,
        verification_proofs_passed=verification_passed,
        statistical_analysis=statistics,
    )
    if root["gate_results"] != expected_gates:
        raise ValueError("gate_results differ from raw evidence recomputation")
    expected_decision = (
        "P6B_GO"
        if all(record["passed"] for record in expected_gates.values())
        else "P6B_STOP"
    )
    if root["decision"] != expected_decision:
        raise ValueError("decision differs from recomputed gate results")
    if root["inactive_components"] != _inactive_components(selection):
        raise ValueError("inactive_components differ from selected config")
    narrative = _derived_narrative(expected_gates, expected_decision)
    if any(root[key] != value for key, value in narrative.items()):
        raise ValueError("claims/next_action differ from gate-derived narrative")
    serialized = json.dumps(root, sort_keys=True, allow_nan=False)
    if _PRIVATE_PATH.search(serialized):
        raise ValueError("artifact contains a private absolute path")
    _finite_tree(root)
    del attempt


def build_p6b_artifact_root(
    *,
    source_tree_contract: Mapping[str, object],
    selection_document: Mapping[str, object],
    heldout_attempt: Mapping[str, object],
    heldout_raw: Mapping[str, object],
) -> dict[str, object]:
    selection = deepcopy(dict(selection_document))
    raw = deepcopy(dict(heldout_raw))
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("heldout raw evaluation must be a mapping")
    from scripts.run_p6b_evaluation import (
        _validate_verification_ledger,
        compute_final_gate_results,
    )

    deviations: list[str] = []
    gates = compute_final_gate_results(
        evaluation["final_results"],
        evidence_complete=not deviations,
        frozen_hashes_unchanged=True,
        verification_proofs_passed=_validate_verification_ledger(
            selection["verification_ledger"]
        ),
        statistical_analysis=evaluation["statistical_analysis"],
    )
    decision = (
        "P6B_GO" if all(item["passed"] for item in gates.values()) else "P6B_STOP"
    )
    narrative = _derived_narrative(gates, decision)
    root = {
        "schema_version": P6B_ARTIFACT_SCHEMA_VERSION,
        "protocol_version": 2,
        "status": "pass",
        "decision": decision,
        "source_commit": source_tree_contract["source_commit"],
        "source_tree_contract": dict(source_tree_contract),
        "source_lineage": {
            "schema_version": 1,
            "selection_source_commit": selection["source_commit"],
            "evaluation_source_commit": raw["source_commit"],
            "package_source_commit": source_tree_contract["source_commit"],
            "selection_to_evaluation_allowed_prefix": "artifacts/P6B_selection/",
            "evaluation_to_package_allowed_prefix": "artifacts/P6B_heldout/",
        },
        "selection_document": selection,
        "heldout_attempt": deepcopy(dict(heldout_attempt)),
        "heldout_raw": raw,
        "gate_results": gates,
        "inactive_components": _inactive_components(selection),
        "protocol_deviations": deviations,
        **narrative,
        "artifact_manifest": [],
    }
    return root


def _render_csv(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row[key] is None else row[key] for key in columns})
    return output.getvalue().encode("utf-8")


def _render_statistics(records: Sequence[Mapping[str, object]]) -> bytes:
    lines = [
        "# P6-B Paired Statistical Analysis",
        "",
        "Cluster bootstrap uses two held-out reference scenes, seed 45, and 10,000 resamples.",
        "Intervals are unstable with two clusters and are not significance claims.",
        "",
        "| Metric | T | P6B mean | B4 mean | Delta | Sample SD | 95% CI | Pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {metric} | {T} | {method_mean:.6f} | {baseline_mean:.6f} | "
            "{mean_delta:.6f} | {std_delta:.6f} | [{ci_low:.6f}, {ci_high:.6f}] | "
            "{n_pairs} |".format(**record)
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _render_report(root: Mapping[str, object]) -> bytes:
    evaluation = root["heldout_raw"]["evaluation"]
    results = evaluation["final_results"]
    attempt = root["heldout_attempt"]
    p6b_t5 = next(row for row in results if row["method"] == "P6B" and row["T"] == "T5")
    gates = "\n".join(
        f"- {name}: {'PASS' if record['passed'] else 'FAIL'}; {record['evidence']}"
        for name, record in root["gate_results"].items()
    )
    inactive = ", ".join(root["inactive_components"]) or "none"
    deviations = (
        "\n".join(f"- {item}" for item in root["protocol_deviations"]) or "- none"
    )
    attempt_events = ", ".join(item["event"] for item in attempt["events"])
    failure_counts = {
        (method, horizon): sum(
            int(row["count"])
            for row in evaluation["failure_analysis"]
            if row["method"] == method and row["T"] == horizon
        )
        for method in ("B4", "P6B")
        for horizon in ("T2", "T3", "T4", "T5")
    }
    failure_summary = "; ".join(
        f"{method} {horizon} total failures={failure_counts[(method, horizon)]}"
        for method in ("B4", "P6B")
        for horizon in ("T2", "T3", "T4", "T5")
    )
    content = ["# Persist4D P6-B GO / NO-GO Report", ""]
    bodies = (
        "Implemented threshold-aware association, dormant reactivation, class compatibility, consolidation, and birth-quality choices without changing frozen local predictions.",
        "P6-A isolated association continuity and reactivation as the method-level bottlenecks addressed by P6-B.",
        "Four reference clusters were used for tuning; the selected config was frozen before one exactly-once evaluation of 33 orders from two held-out clusters.",
        f"Selection source `{root['source_lineage']['selection_source_commit']}`; evaluation source `{root['source_lineage']['evaluation_source_commit']}`; package source `{root['source_lineage']['package_source_commit']}`; selection `{root['heldout_raw']['selection']['sha256']}`; raw attempt `{root['heldout_raw']['attempt_id']}`. Attempt started {attempt['started_utc']}, ended {attempt['ended_utc']}, events: {attempt_events}.",
        f"Held-out P6B T5 t-mAP={p6b_t5['t_mAP']:.6f}, t-REC={p6b_t5['t_REC']:.6f}, ID-switch rate={p6b_t5['identity_switch_rate']:.6f}. Inactive selected components: {inactive}.",
        "Paired reference-cluster bootstrap reports mean, sample SD, and deterministic 95% intervals. Only two held-out clusters are available; no significance claim is made.",
        f"All 64 method/horizon/failure-category cells are included. {failure_summary}. Protocol deviations:\n"
        + deviations,
        "\n".join(f"- {claim}" for claim in root["claims_supported"]),
        "\n".join(f"- {claim}" for claim in root["claims_not_supported"]),
        gates,
        root["next_action"],
    )
    for heading, body in zip(_REPORT_SECTIONS, bodies, strict=True):
        content.extend((f"## {heading}", "", body, ""))
    content.append(root["decision"])
    return ("\n".join(content) + "\n").encode("utf-8")


def _render_derived(root: Mapping[str, object]) -> dict[str, bytes]:
    selection = root["selection_document"]
    evaluation = root["heldout_raw"]["evaluation"]
    sweep_rows = selection["candidate_rows"]
    rendered = {
        path: _render_csv(
            [row for row in sweep_rows if row["stage"] == stage], _SWEEP_COLUMNS
        )
        for stage, path in _STAGE_PATHS.items()
    }
    rendered.update(
        {
            "hyperparameter_sweep.csv": _render_csv(sweep_rows, _SWEEP_COLUMNS),
            "split_manifest.json": _json_bytes(selection["split_manifest"]),
            "selection_document.json": _json_bytes(selection),
            "selected_config.yaml": yaml.safe_dump(
                selection["selected_config"], sort_keys=True
            ).encode(),
            "execution_attempt.json": _json_bytes(root["heldout_attempt"]),
            "heldout_raw.json": _json_bytes(root["heldout_raw"]),
            "final_results.csv": _render_csv(
                evaluation["final_results"], _FINAL_COLUMNS
            ),
            "per_sequence_results.csv": _render_csv(
                evaluation["per_sequence_results"], _PER_SEQUENCE_COLUMNS
            ),
            "failure_analysis.csv": _render_csv(
                evaluation["failure_analysis"], _FAILURE_COLUMNS
            ),
            "failure_diagnostics.csv": _render_csv(
                evaluation["failure_diagnostics"], _FAILURE_DIAGNOSTIC_COLUMNS
            ),
            "statistical_analysis.json": _json_bytes(
                evaluation["statistical_analysis"]
            ),
            "statistical_analysis.md": _render_statistics(
                evaluation["statistical_analysis"]
            ),
            "P6B_GO_NOGO_REPORT.md": _render_report(root),
            "figures/identity_comparison.svg": render_identity_figure(
                evaluation["final_results"]
            ).encode(),
            "figures/reactivation_comparison.svg": render_reactivation_figure(
                evaluation["final_results"]
            ).encode(),
        }
    )
    return dict(sorted(rendered.items()))


def _manifest_for(files: Mapping[str, bytes]) -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(files.items())
    ]


def validate_p6b_artifact(root: Mapping[str, object]) -> None:
    _validate_base(root)
    manifest = root["artifact_manifest"]
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("artifact_manifest must not be empty")
    if manifest != _manifest_for(_render_derived(root)):
        raise ValueError("artifact_manifest does not bind rendered bytes")


def finalize_p6b_artifact(root: Mapping[str, object]) -> dict[str, object]:
    candidate = deepcopy(dict(root))
    candidate["artifact_manifest"] = []
    _validate_base(candidate)
    candidate["artifact_manifest"] = _manifest_for(_render_derived(candidate))
    validate_p6b_artifact(candidate)
    return candidate


def render_p6b_bundle(root: Mapping[str, object]) -> dict[str, bytes]:
    validate_p6b_artifact(root)
    files = _render_derived(root)
    files["artifact_manifest.json"] = _json_bytes(root["artifact_manifest"])
    files["p6b_eval.json"] = _json_bytes(root)
    return dict(sorted(files.items()))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_p6b_artifact(output_root: Path, root: Mapping[str, object]) -> Path:
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise FileExistsError("P6-B output root already exists")
    if output.name in {"", ".", ".."}:
        raise ValueError("P6-B output root must be a named directory")
    files = render_p6b_bundle(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for relative, payload in files.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("artifact path must be repository-relative")
            target = stage.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError("P6-B output root appeared during publication")
        os.replace(stage, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output
