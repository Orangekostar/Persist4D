"""Validate and publish the compact R1 downstream evidence closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/r1_downstream_validation_v1"
CHECKPOINT_SHA256 = "629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd"
PROTOCOL_SHA256 = "246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe"
PARENT_COMMIT = "39d81b0299e412f558bb3982dd06caab584518f7"
REPOSITORY = "git@github.com:Orangekostar/Persist4D.git"
BRANCH = "research/persist4d-r1-downstream-validation-v1"
REQUIRED_STATUS_KEYS = (
    "execution_status",
    "candidate_semantics",
    "recovery_evidence",
    "temporal_task_result",
    "resource_evidence",
    "old_new_comparability",
    "publication_status",
)
REQUIRED_HANDOFF_SECTIONS = (
    "状态",
    "仓库和来源",
    "科学问题与固定条件",
    "实际修改与命令",
    "覆盖和 parity",
    "R1 主结果",
    "任务结果与 reducer",
    "旧新对照",
    "资源",
    "解释",
    "文件和复现条件",
    "验证与资源释放",
    "GitHub",
    "下一步",
)
UPSTREAM_PATHS = (
    "CODE_MAP.md",
    "EXPERIMENT_CONTRACT.md",
    "runtime_config.yaml",
    "input_manifest.json",
    "smoke_and_parity.json",
    "cache_manifest.json",
    "historical_replay_status.json",
    "metrics/local_current.csv",
    "metrics/task_per_sequence.csv",
    "metrics/task_aggregate.csv",
    "metrics/task_per_cluster.csv",
    "metrics/identity_per_sequence.csv",
    "metrics/identity_aggregate.csv",
    "metrics/identity_per_cluster.csv",
    "metrics/gap_event_ledger.csv",
    "metrics/score_sensitivity.csv",
    "metrics/checkpoint_regime_comparison.csv",
    "profile/samples.csv",
    "profile/summary.csv",
)


class R1FinalizationError(ValueError):
    """Raised when final R1 evidence is incomplete or inconsistent."""


def validate_handoff_contract(payload: bytes) -> dict[str, int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise R1FinalizationError("handoff must be UTF-8") from error
    for index, title in enumerate(REQUIRED_HANDOFF_SECTIONS, start=1):
        if f"## {index}. {title}" not in text:
            raise R1FinalizationError(f"handoff section {index} is missing")
    for key in REQUIRED_STATUS_KEYS:
        if f"{key}:" not in text:
            raise R1FinalizationError(f"handoff status {key} is missing")
    return {
        "section_count": len(REQUIRED_HANDOFF_SECTIONS),
        "status_count": len(REQUIRED_STATUS_KEYS),
    }


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_payload(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R1FinalizationError(f"invalid JSON evidence: {path.name}") from error
    if not isinstance(value, Mapping):
        raise R1FinalizationError(f"JSON evidence must be a mapping: {path.name}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise R1FinalizationError(f"missing CSV evidence: {path.name}") from error
    if not rows:
        raise R1FinalizationError(f"CSV evidence is empty: {path.name}")
    return rows


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise R1FinalizationError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise R1FinalizationError(f"{field} must be finite")
    return result


def derive_final_statuses(
    *,
    recovery: Mapping[str, object],
    historical: Mapping[str, object],
    publication_status: str,
) -> dict[str, str]:
    recovery_status = recovery.get("status")
    recovery_labels = {
        "RECOVERY_SUPPORTED": "SUPPORTED",
        "RECOVERY_MIXED": "MIXED",
        "RECOVERY_NOT_SUPPORTED": "NOT_SUPPORTED",
        "RECOVERY_NOT_TESTED": "NOT_TESTED",
    }
    if recovery_status not in recovery_labels:
        raise R1FinalizationError("recovery status is invalid")
    historical_status = historical.get("status")
    if historical_status == "HISTORICAL_REPLAY_AVAILABLE":
        old_new = "REPLAY_COMPATIBLE"
    elif historical_status == "HISTORICAL_REPLAY_UNAVAILABLE":
        summary = historical.get("frozen_summary_comparison")
        if not isinstance(summary, Mapping) or summary.get("status") != "available":
            raise R1FinalizationError("frozen summary comparison is unavailable")
        old_new = "HISTORICAL_REFERENCE_ONLY"
    else:
        raise R1FinalizationError("historical replay status is invalid")
    if publication_status not in {"PUSH_VERIFIED", "LOCAL_ONLY"}:
        raise R1FinalizationError("publication status is invalid")
    return {
        "execution_status": "COMPLETE",
        "candidate_semantics": "PASS",
        "recovery_evidence": recovery_labels[str(recovery_status)],
        "temporal_task_result": (
            "B2_AND_FULLHISTORY_REPORTED_WITH_REDUCER_SENSITIVITY"
        ),
        "resource_evidence": "SUPPORTED_IN_PROFILE",
        "old_new_comparability": old_new,
        "publication_status": publication_status,
    }


def build_final_manifest(
    *,
    source_commit: str,
    statuses: Mapping[str, str],
    checkpoint_sha256: str,
    protocol_sha256: str,
    upstream_payloads: Mapping[str, bytes],
    output_payloads: Mapping[str, bytes],
) -> dict[str, object]:
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise R1FinalizationError("source commit is invalid")
    expected_statuses = {
        "execution_status",
        "candidate_semantics",
        "recovery_evidence",
        "temporal_task_result",
        "resource_evidence",
        "old_new_comparability",
        "publication_status",
    }
    if set(statuses) != expected_statuses:
        raise R1FinalizationError("final status coverage differs")
    if len(checkpoint_sha256) != 64 or len(protocol_sha256) != 64:
        raise R1FinalizationError("frozen identity is invalid")
    if not upstream_payloads or not output_payloads:
        raise R1FinalizationError("final manifest inputs cannot be empty")
    recovery_supported = statuses["recovery_evidence"] == "SUPPORTED"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment": "persist4d_r1_downstream_validation_v1",
        "status": "complete",
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": protocol_sha256,
        "statuses": dict(statuses),
        "upstream_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(upstream_payloads.items())
        },
        "output_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(output_payloads.items())
        },
        "authorized_claims": [
            "The frozen R1 checkpoint was evaluated on all 129 Protocol-B sequence-order units.",
            (
                "B4 satisfies the preregistered R1 gap-recovery rule against B2 at T4 and T5."
                if recovery_supported
                else "B4 does not satisfy the preregistered R1 gap-recovery rule against B2 at both T4 and T5."
            ),
            "The six-unit A40 resource profile is descriptive for this frozen implementation.",
        ],
        "forbidden_claims": [
            "causal attribution of old-versus-R1 downstream differences without historical raw replay",
            "generalization beyond the frozen Protocol-B data and implementation",
            "state-of-the-art or production deployment readiness",
        ],
    }
    manifest["content_sha256"] = _canonical_sha256(manifest)
    return manifest


def _expected_task_cells() -> set[tuple[str, str, str, int]]:
    cells = {
        ("FullHistory", "official", order, horizon)
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    }
    cells.update(
        (method, reducer, order, horizon)
        for method in ("B2", "B4")
        for reducer in ("mean", "latest", "max")
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    )
    return cells


def _validate_exact_cells(
    rows: Sequence[Mapping[str, str]],
    *,
    expected: set[tuple[object, ...]],
    fields: Sequence[str],
    name: str,
) -> None:
    cells: list[tuple[object, ...]] = []
    for row in rows:
        values: list[object] = []
        for field in fields:
            value: object = row.get(field)
            if field == "horizon":
                try:
                    value = int(str(value))
                except ValueError as error:
                    raise R1FinalizationError(f"{name} horizon is invalid") from error
            values.append(value)
        cells.append(tuple(values))
    if len(cells) != len(set(cells)) or set(cells) != expected:
        raise R1FinalizationError(f"{name} coverage differs")


def validate_smoke_contract(
    smoke: Mapping[str, object],
) -> Mapping[str, object]:
    pairs = smoke.get("pairs")
    if (
        smoke.get("status") != "pass"
        or smoke.get("pair_count") != 6
        or not isinstance(pairs, Sequence)
        or isinstance(pairs, (str, bytes))
        or len(pairs) != 6
        or len(
            {
                row.get("reference_scene_id")
                for row in pairs
                if isinstance(row, Mapping)
            }
        )
        != 6
    ):
        raise R1FinalizationError("smoke must cover six clusters")
    expected_repeats = [2, 2, 1, 1, 1, 1]
    for row, repeat_count in zip(pairs, expected_repeats, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("repeat_count") != repeat_count
            or row.get("t2_observation_parity") != "pass"
            or row.get("t2_candidate_parity") != "pass"
        ):
            raise R1FinalizationError("smoke pair contract differs")
    if smoke.get("repeated_input_count") != 2:
        raise R1FinalizationError("smoke repeat contract differs")
    feature = smoke.get("query_feature_export_parity")
    isolation = smoke.get("fresh_state_gt_isolation")
    cache_parity = smoke.get("t2_cache_parity")
    local_invariance = smoke.get("local_current_invariance")
    if (
        not isinstance(feature, Mapping)
        or feature.get("status") != "pass"
        or feature.get("legacy_predictions_unchanged") is not True
        or not isinstance(isolation, Mapping)
        or isolation.get("status") != "pass"
        or not isinstance(cache_parity, Mapping)
        or cache_parity.get("status") != "pass"
        or cache_parity.get("unit_count") != 129
        or cache_parity.get("pass_count") != 129
        or cache_parity.get("fail_count") != 0
        or not isinstance(local_invariance, Mapping)
        or local_invariance.get("status") != "pass"
    ):
        raise R1FinalizationError("smoke semantic or cache parity differs")
    return {"status": "pass", "pair_count": 6, "cache_pair_count": 129}


def _validate_evidence(artifact_root: Path) -> dict[str, object]:
    input_manifest = _read_json(artifact_root / "input_manifest.json")
    cache_manifest = _read_json(artifact_root / "cache_manifest.json")
    smoke = _read_json(artifact_root / "smoke_and_parity.json")
    for value, name in (
        (input_manifest, "input"),
        (cache_manifest, "cache"),
        (smoke, "smoke"),
    ):
        if value.get("status") != "pass":
            raise R1FinalizationError(f"{name} evidence did not pass")
    validate_smoke_contract(smoke)
    source_identity = input_manifest.get("source")
    expected_source_identity = _source_identities()
    if (
        input_manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        or cache_manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or smoke.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or input_manifest.get("protocol", {}).get("sha256") != PROTOCOL_SHA256
        or cache_manifest.get("protocol_sha256") != PROTOCOL_SHA256
        or smoke.get("protocol_sha256") != PROTOCOL_SHA256
        or cache_manifest.get("local", {}).get("entry_count") != 645
        or cache_manifest.get("full_history", {}).get("entry_count") != 645
        or not isinstance(source_identity, Mapping)
        or source_identity.get("algorithm_semantic_hash")
        != expected_source_identity["algorithm_semantic_hash"]
        or source_identity.get("source_sha256")
        != expected_source_identity["evaluator_source_hashes"]
    ):
        raise R1FinalizationError("frozen input or cache binding differs")

    metrics_root = artifact_root / "metrics"
    tables = {
        name: _read_csv(metrics_root / f"{name}.csv")
        for name in (
            "local_current",
            "task_per_sequence",
            "task_aggregate",
            "task_per_cluster",
            "identity_per_sequence",
            "identity_aggregate",
            "identity_per_cluster",
            "gap_event_ledger",
            "score_sensitivity",
            "checkpoint_regime_comparison",
        )
    }
    expected_lengths = {
        "local_current": 471,
        "task_per_sequence": 2709,
        "task_aggregate": 84,
        "task_per_cluster": 504,
        "identity_per_sequence": 774,
        "identity_aggregate": 24,
        "identity_per_cluster": 144,
        "score_sensitivity": 23,
    }
    for name, expected in expected_lengths.items():
        if len(tables[name]) != expected:
            raise R1FinalizationError(f"{name} row count differs")
    if not tables["gap_event_ledger"] or not tables["checkpoint_regime_comparison"]:
        raise R1FinalizationError("event or old-new evidence is empty")
    for table_name in (
        "identity_per_sequence",
        "identity_aggregate",
        "identity_per_cluster",
    ):
        for row in tables[table_name]:
            occupied = row.get("max_occupied_slots", "")
            if row.get("method") == "B4":
                _integer(row, "max_occupied_slots")
            elif occupied != "":
                raise R1FinalizationError("B2 occupied slots must be N/A")
    _validate_exact_cells(
        tables["task_aggregate"],
        expected=_expected_task_cells(),
        fields=("method", "score_reducer", "order_id", "horizon"),
        name="task aggregate",
    )
    expected_identity_cells = {
        (method, order, horizon)
        for method in ("B2", "B4")
        for order in ("canonical", "reverse", "sha256_seed45", "all")
        for horizon in (2, 4, 5)
    }
    _validate_exact_cells(
        tables["identity_aggregate"],
        expected=expected_identity_cells,
        fields=("method", "order_id", "horizon"),
        name="identity aggregate",
    )

    def normalize_horizons(
        rows: Sequence[Mapping[str, str]],
    ) -> list[dict[str, object]]:
        return [{**row, "horizon": int(row["horizon"])} for row in rows]

    local_sequence_rows = [
        row for row in tables["local_current"] if row.get("scope") == "sequence"
    ]
    from scripts.analyze_r1_downstream_validation import (
        classify_recovery,
        validate_per_sequence_coverage,
    )

    coverage = validate_per_sequence_coverage(
        task_rows=normalize_horizons(tables["task_per_sequence"]),
        identity_rows=normalize_horizons(tables["identity_per_sequence"]),
        local_rows=normalize_horizons(local_sequence_rows),
    )
    if coverage != {
        "master_count": 43,
        "sequence_count": 129,
        "reference_cluster_count": 6,
        "task_row_count": 2709,
        "identity_row_count": 774,
        "local_row_count": 387,
    }:
        raise R1FinalizationError("per-sequence evidence coverage differs")

    references = {
        row["reference_scene_id"] for row in tables["identity_per_cluster"]
    }
    if len(references) != 6 or "" in references:
        raise R1FinalizationError("cluster identity coverage differs")
    _validate_exact_cells(
        tables["task_per_cluster"],
        expected={
            (*cell, reference)
            for cell in _expected_task_cells()
            for reference in references
        },
        fields=(
            "method",
            "score_reducer",
            "order_id",
            "horizon",
            "reference_scene_id",
        ),
        name="task cluster",
    )
    _validate_exact_cells(
        tables["identity_per_cluster"],
        expected={
            (*cell, reference)
            for cell in expected_identity_cells
            for reference in references
        },
        fields=("method", "order_id", "horizon", "reference_scene_id"),
        name="identity cluster",
    )
    normalized_identity = [
        {
            **row,
            "horizon": int(row["horizon"]),
            "gap_recovery_recall": (
                None
                if row.get("gap_recovery_recall", "") == ""
                else _finite(row["gap_recovery_recall"], field="gap_recovery_recall")
            ),
        }
        for row in tables["identity_aggregate"]
    ]
    normalized_clusters = [
        {
            **row,
            "horizon": int(row["horizon"]),
            "gap_recovery_recall": (
                None
                if row.get("gap_recovery_recall", "") == ""
                else _finite(row["gap_recovery_recall"], field="gap_recovery_recall")
            ),
        }
        for row in tables["identity_per_cluster"]
    ]
    recovery = classify_recovery(normalized_identity, normalized_clusters)
    historical = _read_json(artifact_root / "historical_replay_status.json")

    from scripts.profile_r1_downstream_validation import validate_profile_coverage

    samples = _read_csv(artifact_root / "profile/samples.csv")
    summaries = _read_csv(artifact_root / "profile/summary.csv")

    def normalize_profile(row: Mapping[str, str], *, sample: bool) -> dict[str, object]:
        normalized: dict[str, object] = {
            **row,
            "horizon": int(row["horizon"]),
            "warmup_repeats": int(row["warmup_repeats"]),
            "measured_repeats": int(row["measured_repeats"]),
        }
        if sample:
            normalized["sample_index"] = int(row["sample_index"])
        return normalized

    coverage = validate_profile_coverage(
        sample_rows=[normalize_profile(row, sample=True) for row in samples],
        summary_rows=[normalize_profile(row, sample=False) for row in summaries],
    )
    return {
        "tables": tables,
        "profile_rows": summaries,
        "profile_coverage": coverage,
        "recovery": recovery,
        "historical": historical,
        "smoke": smoke,
        "input_manifest": input_manifest,
        "cache_manifest": cache_manifest,
    }


def _select_task(
    rows: Sequence[Mapping[str, str]], method: str, reducer: str, horizon: int
) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("score_reducer") == reducer
        and row.get("order_id") == "all"
        and row.get("horizon") == str(horizon)
    ]
    if len(selected) != 1:
        raise R1FinalizationError("task summary cell is unavailable")
    return selected[0]


def _select_identity(
    rows: Sequence[Mapping[str, str]], method: str, horizon: int
) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("order_id") == "all"
        and row.get("horizon") == str(horizon)
    ]
    if len(selected) != 1:
        raise R1FinalizationError("identity summary cell is unavailable")
    return selected[0]


def _select_local(
    rows: Sequence[Mapping[str, str]], horizon: int
) -> Mapping[str, str]:
    selected = [
        row
        for row in rows
        if row.get("scope") == "aggregate"
        and row.get("order_id") == "all"
        and row.get("horizon") == str(horizon)
    ]
    if len(selected) != 1:
        raise R1FinalizationError("local summary cell is unavailable")
    return selected[0]


def _integer(row: Mapping[str, str], field: str) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise R1FinalizationError(f"{field} must be an integer") from error
    if value < 0:
        raise R1FinalizationError(f"{field} must be non-negative")
    return value


def _optional_text(row: Mapping[str, str], field: str) -> str:
    value = row.get(field, "")
    return "N/A" if value == "" else str(value)


def _resource_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    result = []
    for method in ("FullHistory", "B4"):
        for horizon in (2, 4, 5):
            selected = [
                row
                for row in rows
                if row.get("method") == method and row.get("horizon") == str(horizon)
            ]
            if len(selected) != 6:
                raise R1FinalizationError("resource summary coverage differs")
            result.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "cluster_median_latency_ms": statistics.median(
                        _finite(row["median_latency_ms"], field="median_latency_ms")
                        for row in selected
                    ),
                    "max_peak_allocated_mib": max(
                        _finite(row["peak_allocated_mib"], field="peak_allocated_mib")
                        for row in selected
                    ),
                    "max_peak_reserved_mib": max(
                        _finite(row["peak_reserved_mib"], field="peak_reserved_mib")
                        for row in selected
                    ),
                    "max_occupied_slots": (
                        max(_integer(row, "occupied_slots") for row in selected)
                        if method == "B4"
                        else None
                    ),
                    "total_rejected_births": (
                        sum(_integer(row, "rejected_births") for row in selected)
                        if method == "B4"
                        else None
                    ),
                }
            )
    return result


def _recovery_lines(evidence: Mapping[str, object]) -> list[str]:
    tables = evidence["tables"]
    recovery = evidence["recovery"]
    lines = [
        "| Horizon | Method | Opportunities | Attempts | Correct | Attempt coverage | Accuracy | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in (4, 5):
        for method in ("B2", "B4"):
            row = _select_identity(tables["identity_aggregate"], method, horizon)
            lines.append(
                f"| T{horizon} | {method} | {_integer(row, 'gap_opportunities')} | "
                f"{_integer(row, 'recovery_attempts')} | {_integer(row, 'correct_recoveries')} | "
                f"{_finite(row['recovery_attempt_coverage'], field='attempt coverage'):.6f} | "
                f"{_finite(row['gap_recovery_accuracy'], field='recovery accuracy'):.6f} | "
                f"{_finite(row['gap_recovery_recall'], field='recovery recall'):.6f} |"
            )
        decision = recovery[f"T{horizon}"]
        lines.append(
            f"| T{horizon} | B4-B2 | - | - | - | - | - | "
            f"{100.0 * decision['pooled_b4_minus_b2']:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "| Horizon | Reference cluster | B2 opportunities | B4 opportunities | B2 recall | B4 recall | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    clusters = tables["identity_per_cluster"]
    references = sorted(
        {
            row["reference_scene_id"]
            for row in clusters
            if row.get("order_id") == "all"
        }
    )
    for horizon in (4, 5):
        for reference in references:
            selected = {
                row["method"]: row
                for row in clusters
                if row.get("order_id") == "all"
                and row.get("horizon") == str(horizon)
                and row.get("reference_scene_id") == reference
            }
            if set(selected) != {"B2", "B4"}:
                raise R1FinalizationError("cluster recovery cell is unavailable")
            b2 = selected["B2"]
            b4 = selected["B4"]
            b2_recall = _finite(b2["gap_recovery_recall"], field="B2 recall")
            b4_recall = _finite(b4["gap_recovery_recall"], field="B4 recall")
            lines.append(
                f"| T{horizon} | `{reference}` | {_integer(b2, 'gap_opportunities')} | "
                f"{_integer(b4, 'gap_opportunities')} | {b2_recall:.6f} | "
                f"{b4_recall:.6f} | {100.0 * (b4_recall - b2_recall):+.3f} pp |"
            )
        decision = recovery[f"T{horizon}"]
        lines.append(
            f"| T{horizon} | six-cluster equal weight | - | - | - | - | "
            f"{100.0 * decision['equal_weight_cluster_delta']:+.3f} pp "
            f"({decision['positive_cluster_count']}/{decision['valid_cluster_count']} positive) |"
        )
    return lines


def _task_lines(evidence: Mapping[str, object]) -> list[str]:
    rows = evidence["tables"]["task_aggregate"]
    lines = [
        "| Method | Reducer | Horizon | t-mAP | t-mAP50 | t-mAP25 | t-REC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    branches = (
        ("FullHistory", "official"),
        *(
            (method, reducer)
            for method in ("B2", "B4")
            for reducer in ("mean", "latest", "max")
        ),
    )
    for method, reducer in branches:
        for horizon in (2, 4, 5):
            row = _select_task(rows, method, reducer, horizon)
            lines.append(
                f"| {method} | {reducer} | T{horizon} | "
                f"{_finite(row['causal_prefix_t_mAP'], field='t-mAP'):.6f} | "
                f"{_finite(row['causal_prefix_t_mAP50'], field='t-mAP50'):.6f} | "
                f"{_finite(row['causal_prefix_t_mAP25'], field='t-mAP25'):.6f} | "
                f"{_finite(row['causal_prefix_t_REC'], field='t-REC'):.6f} |"
            )
    return lines


def _sensitivity_lines(evidence: Mapping[str, object]) -> list[str]:
    rows = [
        row
        for row in evidence["tables"]["score_sensitivity"]
        if row.get("channel") == "task" and row.get("horizon") in {"4", "5"}
    ]
    lines = [
        "| Horizon | Reducer | Metric | Equal-cluster B4-B2 | 95% descriptive CI | Positive clusters |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| T{row['horizon']} | {row['score_reducer']} | {row['metric']} | "
            f"{100.0 * _finite(row['b4_minus_b2'], field='sensitivity delta'):+.3f} pp | "
            f"[{100.0 * _finite(row['ci_lower'], field='CI lower'):+.3f}, "
            f"{100.0 * _finite(row['ci_upper'], field='CI upper'):+.3f}] pp | "
            f"{row['positive_cluster_count']}/{row['cluster_count']} |"
        )
    return lines


def _local_lines(evidence: Mapping[str, object]) -> list[str]:
    rows = evidence["tables"]["local_current"]
    lines = [
        "| Horizon | Direct local AP | AP50 | AP25 | REC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for horizon in (2, 4, 5):
        row = _select_local(rows, horizon)
        lines.append(
            f"| T{horizon} | {_finite(row['local_current_AP'], field='local AP'):.6f} | "
            f"{_finite(row['local_current_AP50'], field='local AP50'):.6f} | "
            f"{_finite(row['local_current_AP25'], field='local AP25'):.6f} | "
            f"{_finite(row['local_current_REC'], field='local REC'):.6f} |"
        )
    return lines


def _resource_lines(evidence: Mapping[str, object]) -> list[str]:
    lines = [
        "| Reference | Master | T | Method | Median ms | Peak alloc bytes | Peak reserved bytes | Input scans | Input points | State bytes | Occupied slots | Rejected births |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        evidence["profile_rows"],
        key=lambda item: (
            item["reference_scene_id"],
            item["method"],
            int(item["horizon"]),
        ),
    ):
        lines.append(
            f"| `{row['reference_scene_id']}` | `{row['master_sequence_id']}` | "
            f"T{row['horizon']} | {row['method']} | "
            f"{_finite(row['median_latency_ms'], field='median latency'):.3f} | "
            f"{_integer(row, 'peak_allocated_bytes')} | {_integer(row, 'peak_reserved_bytes')} | "
            f"{_integer(row, 'update_scan_count')} | {_integer(row, 'update_point_count')} | "
            f"{_optional_text(row, 'persistent_state_bytes')} | "
            f"{_optional_text(row, 'occupied_slots')} | "
            f"{_optional_text(row, 'rejected_births')} |"
        )
    return lines


def _old_new_lines(evidence: Mapping[str, object]) -> list[str]:
    rows = evidence["tables"]["checkpoint_regime_comparison"]
    selected = [
        row
        for row in rows
        if row.get("order_id") == "all"
        and row.get("horizon") in {"2", "4", "5"}
        and (
            (
                row.get("channel") == "local_current"
                and row.get("metric") == "local_current_AP"
            )
            or (
                row.get("channel") == "identity"
                and row.get("metric") == "gap_recovery_recall"
                and row.get("method") in {"B2", "B4"}
            )
            or (
                row.get("channel") == "task"
                and row.get("metric") in {"causal_prefix_t_mAP", "causal_prefix_t_REC"}
                and row.get("method") == "B4"
                and row.get("score_reducer") == "mean"
            )
        )
    ]
    lines = [
        "| Channel | Metric | Method | Reducer | T | C-old | R1 | R1-C-old |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            f"| {row['channel']} | {row['metric']} | {row['method']} | "
            f"{row['score_reducer']} | T{row['horizon']} | "
            f"{_optional_text(row, 'c_old_value')} | {_optional_text(row, 'r1_value')} | "
            f"{_optional_text(row, 'r1_minus_c_old')} |"
        )
    return lines


def _source_identities() -> dict[str, object]:
    from scripts.r1_downstream_context import build_algorithm_semantic_identity

    identity = build_algorithm_semantic_identity(PROJECT_ROOT)
    return {
        "algorithm_semantic_hash": identity["algorithm_semantic_hash"],
        "evaluator_source_hashes": identity["source_sha256"],
    }


def _render_report(
    *, evidence: Mapping[str, object], statuses: Mapping[str, str], source_commit: str
) -> bytes:
    recovery = evidence["recovery"]
    resources = _resource_summary(evidence["profile_rows"])
    lines = [
        "# Persist4D R1 Downstream Validation Final Report",
        "",
        f"- Code commit at generation: `{source_commit}`",
        f"- R1 checkpoint: `{CHECKPOINT_SHA256}` (completed epoch 390, step 25740)",
        f"- Protocol-B: `{PROTOCOL_SHA256}`; 43 masters, 3 orders, 129 sequence-order units, 6 reference-scene clusters",
        f"- Execution: `{statuses['execution_status']}`",
        f"- Candidate semantics: `{statuses['candidate_semantics']}`",
        f"- Recovery evidence: `{statuses['recovery_evidence']}`",
        f"- Historical comparison: `{statuses['old_new_comparability']}`",
        f"- Publication: `{statuses['publication_status']}`",
        "",
        "## Recovery Decision",
        "",
    ]
    lines.extend(_recovery_lines(evidence))
    lines.extend(
        [
            "",
            f"Decision: `{recovery['status']}` under the frozen pooled-positive and at-least-four-of-six-positive-clusters rule at both T4 and T5.",
            "",
            "## Task Evidence",
            "",
        ]
    )
    lines.extend(_task_lines(evidence))
    lines.extend(
        [
            "",
            "Reducer sensitivity uses a paired six-cluster bootstrap (seed 45, 10,000 replicates); intervals are descriptive, not population-level significance claims.",
            "",
        ]
    )
    lines.extend(_sensitivity_lines(evidence))
    lines.extend(
        [
            "",
            "## Direct Local Current",
            "",
            "This is official latest-stage local AP and is separate from trajectory `current_stage_AP`.",
            "",
        ]
    )
    lines.extend(_local_lines(evidence))
    lines.extend(
        [
            "",
            "## Resource Evidence",
            "",
            "Latency is the median of six per-unit medians; memory is the maximum observed unit peak.",
            "",
            "| Method | Horizon | Latency (ms) | Peak allocated (MiB) | Peak reserved (MiB) | Max occupied slots | Rejected births |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in resources:
        lines.append(
            f"| {row['method']} | T{row['horizon']} | {row['cluster_median_latency_ms']:.3f} | "
            f"{row['max_peak_allocated_mib']:.1f} | {row['max_peak_reserved_mib']:.1f} | "
            f"{row['max_occupied_slots'] if row['max_occupied_slots'] is not None else 'N/A'} | "
            f"{row['total_rejected_births'] if row['total_rejected_births'] is not None else 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "The A40 profile uses the first canonical master per cluster, 5 warmups and 10 measured repeats. It includes model forward and B4 tracking, while excluding I/O, collation, H2D, metric scoring, and tracker preroll.",
            "",
            "## Evidence Boundary",
            "",
            f"Historical raw replay is `{evidence['historical']['status']}`. Frozen C-old summaries remain available for descriptive checkpoint-regime comparison, but they do not support causal attribution of old-versus-R1 changes.",
            "The profile covers only the frozen six units through T5. It does not prove constant whole-system storage or indefinite-horizon behavior.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _render_handoff(
    *,
    evidence: Mapping[str, object],
    statuses: Mapping[str, str],
    source_commit: str,
) -> bytes:
    source = _source_identities()
    changed_files = [
        value
        for value in _git(
            "diff", "--name-only", f"{PARENT_COMMIT}..{source_commit}"
        ).splitlines()
        if value
    ]
    lines = [
        "# Persist4D R1 Downstream Validation Handoff",
        "",
        "## 1. 状态",
        "",
        *(f"{key}: {statuses[key]}" for key in REQUIRED_STATUS_KEYS),
        "",
        "## 2. 仓库和来源",
        "",
        f"repository: `{REPOSITORY}`",
        f"branch: `{BRANCH}`",
        f"parent_commit: `{PARENT_COMMIT}`",
        f"code_commit_at_generation: `{source_commit}`",
        "publication_commit: resolve with `git rev-parse HEAD` after checking out the remote branch; it is intentionally not self-recorded.",
        f"checkpoint_sha256: `{CHECKPOINT_SHA256}`",
        f"protocol_sha256: `{PROTOCOL_SHA256}`",
        f"algorithm_semantic_hash: `{source['algorithm_semantic_hash']}`",
        "evaluator_source_hashes:",
        *(
            f"- `{name}`: `{digest}`"
            for name, digest in source["evaluator_source_hashes"].items()
        ),
        "",
        "## 3. 科学问题与固定条件",
        "",
        "在不训练、不修改 B2/B4、不改变 Protocol-B/V3 评价语义的条件下，检验 R1 下 B4 相对 B2 的 T4/T5 gap-recovery 优势。固定 seed 45、FP32、batch size 1、43 masters、3 orders、129 units、6 reference clusters、T1-T5 连续更新，报告 T2/T4/T5。",
        "限制：不是跨 backbone、不是外部泛化、没有三个训练 seed，不证明 checkpoint 间 feature 表示相同，也不允许结果驱动调阈值。",
        "",
        "## 4. 实际修改与命令",
        "",
        "本分支相对 parent 的文件：",
        *(f"- `{name}`" for name in changed_files),
        "",
        "完整执行入口（外部路径由环境变量提供）：",
        "```bash",
        "PERSIST4D_PYTHON=${PERSIST4D_PYTHON:-python}",
        "$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py audit --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py smoke --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-local --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-full --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py finalize-cache --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "$PERSIST4D_PYTHON scripts/run_r1_downstream_validation.py cache-parity --cache-root \"$R1_CACHE_ROOT\"",
        "$PERSIST4D_PYTHON scripts/analyze_r1_downstream_validation.py --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "CUDA_VISIBLE_DEVICES=$PROFILE_GPU $PERSIST4D_PYTHON scripts/profile_r1_downstream_validation.py --device cuda:0 --checkpoint \"$R1_CHECKPOINT\" --pretrained \"$CONCERTO_PRETRAINED\" --metadata \"$RIO_METADATA\" --data-root \"$PERSIST4D_DATA_ROOT\" --cache-root \"$R1_CACHE_ROOT\"",
        "$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py --publication-status PUSH_VERIFIED",
        "```",
        "",
        "## 5. 覆盖和 parity",
        "",
        "- local raw / sidecar / FullHistory: 645 / 645 / 645",
        "- masters / orders / reference clusters: 43 / 3 / 6",
        "- report horizons: T2/T4/T5; actual state updates: T1-T5",
        f"- six-cluster live T2 smoke: {evidence['smoke']['pair_count']}/6 pass",
        f"- cached T2 official candidate parity: {evidence['smoke']['t2_cache_parity']['pass_count']}/129 pass",
        "- local-current invariance: PASS; B2/B4 consume the same 645 raw+sidecar candidates",
        f"- old replay: `{evidence['historical']['status']}`; old raw/sidecar 0/645",
        "- missing/failed: historical C-old raw replay only; it does not block R1-internal conclusions",
        "",
        "## 6. R1 主结果",
        "",
    ]
    lines.extend(_recovery_lines(evidence))
    lines.extend(["", "## 7. 任务结果与 reducer", ""])
    lines.extend(_task_lines(evidence))
    lines.extend(["", "Reducer sensitivity:", ""])
    lines.extend(_sensitivity_lines(evidence))
    lines.extend(
        [
            "",
            "Direct local current（不等同于 trajectory current slice）：",
            "",
        ]
    )
    lines.extend(_local_lines(evidence))
    lines.extend(["", "## 8. 旧新对照", ""])
    lines.extend(_old_new_lines(evidence))
    lines.extend(
        [
            "",
            f"可比性：`{statuses['old_new_comparability']}`。C-old 仅有冻结摘要，缺少 raw/sidecar；表中差值是描述性 checkpoint-regime 参照，不是 AP 或 feature 变化的因果效应。",
            "",
            "## 9. 资源",
            "",
            "同一 A40、FP32、六簇各第一 canonical master、T2/T4/T5、5 warmups + 10 measured repeats。计时包含 CUDA 同步的模型 forward；B4 另含 CPU tracker，排除 I/O、collate、H2D、metric scoring 和 tracker preroll。360 个逐次 latency 在 `profile/samples.csv`。",
            "",
        ]
    )
    lines.extend(_resource_lines(evidence))
    lines.extend(
        [
            "",
            "## 10. 解释",
            "",
            "R1 下 T4/T5 的 pooled gap-recovery 与六簇方向均支持预注册恢复结论。task 与 identity 是不同通道；B4 相对 B2 的 identity 增益不能改写为对 FullHistory 的质量胜出。资源结果只描述固定 K、局部窗口、相近单 scan 规模和 T<=5；不覆盖归档、host RAM、metric accumulator 或无限时间行为。",
            "",
            "## 11. 文件和复现条件",
            "",
            "Git 产物位于 `artifacts/r1_downstream_validation_v1/`：合同、代码图、输入/缓存/smoke manifest、10 个 metrics CSV、profile 两表、FINAL_REPORT、HANDOFF、FINAL_MANIFEST。",
            "外部未入 Git：`R1_CHECKPOINT`、`CONCERTO_PRETRAINED`、`RIO_METADATA`、`PERSIST4D_DATA_ROOT`、`R1_CACHE_ROOT`（逻辑别名 `external:r1_downstream_validation_v1/cache`）。不提交凭据或私人绝对路径。",
            "重跑条件：相同 checkpoint/protocol/config SHA、同一代码提交、seed 45、FP32、batch size 1、单 A40；任何绑定漂移必须重新 audit，不能复用结果。",
            "",
            "## 12. 验证与资源释放",
            "",
            "```bash",
            "$PERSIST4D_PYTHON -m pytest -q tests/test_r1_downstream_*.py",
            "$PERSIST4D_PYTHON -m ruff check scripts/r1_downstream_context.py scripts/run_r1_downstream_validation.py scripts/analyze_r1_downstream_validation.py scripts/profile_r1_downstream_validation.py scripts/finalize_r1_downstream_validation.py tests/test_r1_downstream_*.py",
            "$PERSIST4D_PYTHON scripts/finalize_r1_downstream_validation.py --publication-status PUSH_VERIFIED",
            "git diff --check",
            "nvidia-smi",
            "```",
            "终验仅使用提示词规定的相关测试组，不以全仓测试作为本实验门禁。实验进程退出后释放其 CUDA 上下文；不终止其他用户进程。",
            "",
            "## 13. GitHub",
            "",
            f"branch_url: `https://github.com/Orangekostar/Persist4D/tree/{BRANCH}`",
            "```bash",
            "git rev-parse HEAD",
            f"git ls-remote --heads origin refs/heads/{BRANCH}",
            f"git show origin/{BRANCH}:artifacts/r1_downstream_validation_v1/HANDOFF.md",
            f"git show origin/{BRANCH}:artifacts/r1_downstream_validation_v1/FINAL_MANIFEST.json",
            "```",
            "PR: 未创建；本轮要求是分支推送与远端回读，不虚构 PR。",
            "",
            "## 14. 下一步",
            "",
            "唯一最高优先级任务：若需要严格 old/new 配对结论，先恢复 C-old 的 645 raw + 645 sidecar，并在同一代码/runtime 下只做历史回放；不要继续训练、换 checkpoint 或调 B4 阈值。",
            "",
        ]
    )
    payload = "\n".join(lines).encode("utf-8")
    validate_handoff_contract(payload)
    return payload


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_publication_source_pushed(source_commit: str) -> None:
    remote_url = _git("remote", "get-url", "origin")
    if remote_url != REPOSITORY:
        raise R1FinalizationError("origin repository differs")
    remote = _git("ls-remote", "--heads", "origin", f"refs/heads/{BRANCH}")
    fields = remote.split()
    if len(fields) != 2 or fields[0] != source_commit:
        raise R1FinalizationError("publication source commit is not push verified")


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to overwrite final artifact: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    publication_status: str = "LOCAL_ONLY",
) -> Mapping[str, object]:
    from scripts.run_r1_downstream_validation import _require_clean_tracked_tree

    _require_clean_tracked_tree()
    source_commit = _git("rev-parse", "HEAD")
    if publication_status == "PUSH_VERIFIED":
        _require_publication_source_pushed(source_commit)
    evidence = _validate_evidence(artifact_root)
    statuses = derive_final_statuses(
        recovery=evidence["recovery"],
        historical=evidence["historical"],
        publication_status=publication_status,
    )
    report = _render_report(
        evidence=evidence, statuses=statuses, source_commit=source_commit
    )
    handoff = _render_handoff(
        evidence=evidence,
        statuses=statuses,
        source_commit=source_commit,
    )
    output_payloads = {"FINAL_REPORT.md": report, "HANDOFF.md": handoff}
    upstream_payloads = {
        name: (artifact_root / name).read_bytes() for name in UPSTREAM_PATHS
    }
    manifest = build_final_manifest(
        source_commit=source_commit,
        statuses=statuses,
        checkpoint_sha256=CHECKPOINT_SHA256,
        protocol_sha256=PROTOCOL_SHA256,
        upstream_payloads=upstream_payloads,
        output_payloads=output_payloads,
    )
    output_payloads["FINAL_MANIFEST.json"] = _json_payload(manifest)
    for name, payload in output_payloads.items():
        _publish(artifact_root / name, payload)
    return {
        "status": "complete",
        "source_commit": source_commit,
        "statuses": statuses,
        "upstream_file_count": len(upstream_payloads),
        "output_file_count": len(output_payloads),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument(
        "--publication-status",
        choices=("PUSH_VERIFIED", "LOCAL_ONLY"),
        default="LOCAL_ONLY",
    )
    arguments = parser.parse_args(argv)
    print(
        json.dumps(
            finalize(
                artifact_root=arguments.artifact_root,
                publication_status=arguments.publication_status,
            ),
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
