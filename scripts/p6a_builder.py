"""Build the deterministic Persist4D P6-A evidence bundle."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from scripts.evaluate_persist4d_p6a import (
    TaskMetricEvaluation,
    normalize_official_metric_blocks,
)
from scripts.p6a_analysis import GateConfig, evaluate_gates, paired_cluster_bootstrap
from scripts.p6a_artifacts import (
    CSV_COLUMN_SCHEMAS,
    GATE_IDS,
    METHOD_IDS,
    ONLINE_METHOD_IDS,
    P5_FROZEN_VALUES,
    REPORT_PATH,
    ROOT_ARTIFACT_PATH,
    render_artifact_bundle,
    validate_root_artifact,
    verify_artifact_manifest,
)
from scripts.p6a_cache import (
    config_documents_sha256,
    portable_runtime_config_text,
    validate_cache_manifest,
)
from scripts.p6a_efficiency import (
    aggregate_efficiency_rows,
    validate_efficiency_manifest,
)
from scripts.p6a_figures import (
    render_figure_a_identity,
    render_figure_b_online_tmap,
    render_figure_c_reactivation,
    render_figure_d_failures,
    render_figure_e_latency,
)
from scripts.p6a_protocol import validate_protocol_b_manifest
from scripts.p6a_results import (
    association_event_rows,
    capacity_audit_rows,
    failure_breakdown_rows,
    per_sequence_result_rows,
    reactivation_audit_rows,
    reactivation_by_gap_rows,
    reactivation_distribution_rows,
)

HORIZONS = (2, 3, 4, 5)
REACTIVATION_METHODS = ("B1", "B2", "B3", "B4")
REACTIVATION_HORIZONS = (3, 4, 5)
FAILURE_CATEGORIES = (*tuple(f"F{index}" for index in range(1, 8)), "unclassified")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_text(value: Mapping[str, object]) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _config_digest(documents: Mapping[str, bytes]) -> str:
    return config_documents_sha256(documents)


def _portable_runtime_config_text(runtime_config_text: str) -> str:
    return portable_runtime_config_text(runtime_config_text)


def _expected_cache_keys(protocol_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    validate_protocol_b_manifest(protocol_manifest)
    masters = protocol_manifest["masters"]
    keys: list[dict[str, object]] = []
    for master in masters:
        for order_id in ("canonical", "reverse", "sha256_seed45"):
            scan_ids = master["orders"][order_id]["visit_order"]
            for stage_index in range(5):
                history = list(scan_ids[: stage_index + 1])
                keys.append(
                    {
                        "master_sequence_id": master["master_sequence_id"],
                        "reference_scene_id": master["reference_scene_id"],
                        "order_id": order_id,
                        "stage_index": stage_index,
                        "history_scan_ids": history,
                        "local_window_scan_ids": history[-1:]
                        if stage_index == 0
                        else history[-2:],
                    }
                )
    if len(keys) != 645:
        raise ValueError("Protocol B must define exactly 645 cache observations")
    return keys


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def metric_table_rows(
    metric_blocks: Mapping[str, object],
    per_sequence_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[dict[str, object], ...]]:
    """Build aggregate metric tables using pooled event denominators."""

    grouped: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {
            "id_switches": 0,
            "transition_opportunities": 0,
            "correct_reactivations": 0,
            "reactivation_attempts": 0,
        }
    )
    for row in per_sequence_rows:
        key = (str(row["method"]), int(row["T"]))
        if key[0] not in ONLINE_METHOD_IDS or key[1] not in HORIZONS:
            raise ValueError("per-sequence metric scope differs from P6-A")
        for field in grouped[key]:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            grouped[key][field] += value

    baseline: list[dict[str, object]] = []
    strict: list[dict[str, object]] = []
    raw: list[dict[str, object]] = []
    for method in ONLINE_METHOD_IDS:
        for horizon in HORIZONS:
            token = f"T{horizon}"
            raw_metric = metric_blocks["raw"][method][token]
            strict_metric = metric_blocks["strict"][method][token]
            counts = grouped.get((method, horizon))
            if counts is None:
                raise ValueError("per-sequence rows omit a method/horizon group")
            baseline.append(
                {
                    "method": method,
                    "T": horizon,
                    "raw_AP": raw_metric["AP"],
                    "online_t_mAP": strict_metric["t_mAP"],
                    "online_t_REC": strict_metric["t_REC"],
                    "id_switch_rate": _ratio(
                        counts["id_switches"], counts["transition_opportunities"]
                    ),
                    "reactivation_accuracy": _ratio(
                        counts["correct_reactivations"],
                        counts["reactivation_attempts"],
                    ),
                }
            )
            strict.append(
                {
                    "method": method,
                    "T": horizon,
                    **{
                        field: strict_metric[field]
                        for field in (
                            "t_mAP",
                            "t_mAP50",
                            "t_mAP25",
                            "t_REC",
                            "t_REC50",
                            "t_REC25",
                        )
                    },
                }
            )
            raw.append(
                {
                    "method": method,
                    "T": horizon,
                    **{
                        field: raw_metric[field]
                        for field in ("AP", "AP50", "AP25", "REC")
                    },
                }
            )
    return {
        "baseline_results.csv": tuple(baseline),
        "strict_online_results.csv": tuple(strict),
        "raw_local_results.csv": tuple(raw),
    }


def _paired_idsw(
    per_sequence: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[int, object], list[str]]:
    results: dict[int, object] = {}
    errors: list[str] = []
    for horizon in (4, 5):
        records = []
        for row in per_sequence:
            if row["method"] not in {"B3", "B4"} or int(row["T"]) != horizon:
                continue
            rate = row["id_switch_rate"]
            if rate is None:
                continue
            records.append(
                {
                    "reference_scene_id": row["reference_scene_id"],
                    "master_sequence_id": row["master_sequence_id"],
                    "prefix": horizon,
                    "method": row["method"],
                    "metric": "id_switch_rate",
                    "value": rate,
                    "order_id": row["order_id"],
                    "prediction_digest": row["prediction_digest"],
                }
            )
        try:
            results[horizon] = paired_cluster_bootstrap(
                records,
                method="B4",
                baseline_method="B3",
                metric="id_switch_rate",
                n_bootstrap=replicates,
                seed=seed,
            )
        except ValueError as error:
            results[horizon] = None
            errors.append(f"G6A-1 T{horizon} unavailable: {error}")
    return results, errors


def _gate_analysis(
    *,
    metric_blocks: Mapping[str, object],
    fingerprints: Mapping[str, object],
    per_sequence: Sequence[Mapping[str, object]],
    reactivation: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    paired, errors = _paired_idsw(
        per_sequence, replicates=replicates, seed=seed
    )
    reactivation_tree: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for row in reactivation:
        reactivation_tree[str(row["method"])][int(row["T"])] = {
            "reactivation_accuracy": row["reactivation_accuracy"],
            "reactivation_recall": row["reactivation_recall"],
        }
    raw = {
        method: {
            horizon: {
                field: metric_blocks["raw"][method][horizon][field]
                for field in ("AP", "AP50", "AP25", "REC")
            }
            for horizon in ("T2", "T3", "T4", "T5")
        }
        for method in ONLINE_METHOD_IDS
    }
    b4_failure_counts = {
        category: sum(
            int(row["count"])
            for row in failures
            if row["method"] == "B4" and row["category"] == category
        )
        for category in FAILURE_CATEGORIES
    }
    gate_inputs = {
        "paired_idsw": paired,
        "reactivation": dict(reactivation_tree),
        "raw_prediction_fingerprints": {
            method: fingerprints["prediction"][method]
            for method in ONLINE_METHOD_IDS
        },
        "raw_local_metrics": raw,
        "online_task": metric_blocks["strict"],
        "failure_counts": b4_failure_counts,
    }
    detailed = evaluate_gates(
        gate_inputs,
        config=GateConfig(),
        method="B4",
        strong_baseline="B3",
    )
    public = {
        gate_id: {
            "passed": bool(detailed[gate_id]["passed"]),
            "evidence": (
                f"{'Passed' if detailed[gate_id]['passed'] else 'Failed'} the "
                "preregistered threshold; see statistical_analysis.md."
            ),
        }
        for gate_id in GATE_IDS
    }
    return public, detailed, errors


def _figure_e_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    selections = (
        ("B4", "bootstrap", "bootstrap_latency_ms", "b4", "bootstrap"),
        ("B4", "new_visit", "new_visit_latency_ms", "b4", "new_visit"),
        (
            "full_history_rescene",
            "full_history",
            "full_history_latency_ms",
            "full_history_rescene",
            "new_visit",
        ),
    )
    for method, row_type, field, figure_method, phase in selections:
        for horizon in HORIZONS:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and int(row["T"]) == horizon
                and row["row_type"] == row_type
            ]
            total = sum(int(row["count"]) for row in selected)
            if not selected or total <= 0:
                raise ValueError("efficiency rows omit a required measured group")
            latency = sum(float(row[field]) * int(row["count"]) for row in selected) / total
            output.append(
                {
                    "method_id": figure_method,
                    "horizon": horizon,
                    "phase": phase,
                    "latency_ms": latency,
                }
            )
    return output


def _statistical_markdown(detailed: Mapping[str, object]) -> str:
    return (
        "# P6-A Statistical Analysis\n\n"
        "Unit: 43 master sequences x 3 deterministic orders; uncertainty is "
        "clustered by the six reference scenes. Thresholds are preregistered.\n\n"
        "```json\n"
        + json.dumps(detailed, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n```\n"
    )


def seal_artifact_manifest(artifact: Mapping[str, object]) -> dict[str, object]:
    """Bind every derived output to its exact rendered bytes."""

    sealed = copy.deepcopy(dict(artifact))
    derived = sealed["derived_artifacts"]
    paths = {
        path
        for kind in ("csv", "json", "markdown", "svg", "yaml")
        for path in derived[kind]
    }
    paths.add(REPORT_PATH)
    sealed["artifact_manifest"] = [
        {"path": path, "bytes": 1, "sha256": "0" * 64}
        for path in sorted(paths)
    ]
    validate_root_artifact(sealed)
    first_render = render_artifact_bundle(sealed)
    sealed["artifact_manifest"] = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
        for path, payload in sorted(first_render.items())
        if path != ROOT_ARTIFACT_PATH
    ]
    validate_root_artifact(sealed)
    final_render = render_artifact_bundle(sealed)
    verify_artifact_manifest(sealed, final_render)
    return sealed


def build_p6a_root_artifact(
    *,
    evaluation: TaskMetricEvaluation,
    protocol_manifest: Mapping[str, object],
    cache_manifest: Mapping[str, object],
    efficiency_manifest: Mapping[str, object],
    source_commit: str,
    p6a_config_text: str,
    runtime_config_text: str,
    run_id: str | None = None,
) -> dict[str, object]:
    """Derive the complete P6-A root artifact from frozen machine evidence."""

    validate_protocol_b_manifest(protocol_manifest)
    validate_efficiency_manifest(efficiency_manifest)
    protocol = protocol_manifest["protocol"]
    if (
        protocol["expected_master_count"] != 43
        or protocol["expected_reference_scene_clusters"] != 6
        or evaluation.sequence_count != 129
    ):
        raise ValueError("P6-A requires 43 masters, six clusters, and 129 units")
    p6a_bytes = p6a_config_text.encode("utf-8")
    runtime_bytes = runtime_config_text.encode("utf-8")
    portable_runtime_config_text = _portable_runtime_config_text(runtime_config_text)
    settings = yaml.safe_load(p6a_config_text)
    if not isinstance(settings, Mapping) or not isinstance(
        yaml.safe_load(runtime_config_text), Mapping
    ):
        raise TypeError("P6-A config documents must be YAML mappings")
    protocol_settings = settings.get("protocol_b")
    if not isinstance(protocol_settings, Mapping):
        raise TypeError("P6-A config must define protocol_b")
    configured_sources = protocol_settings.get("sources")
    manifest_sources = protocol_manifest["sources"]
    if not isinstance(configured_sources, Mapping):
        raise TypeError("P6-A protocol config must define frozen sources")
    expected_protocol_values = {
        "split": protocol["split"],
        "horizons": protocol["horizons"],
        "expected_master_count": protocol["expected_master_count"],
        "expected_reference_scene_clusters": protocol[
            "expected_reference_scene_clusters"
        ],
        "order_variants": protocol["order_variants"],
        "seed": protocol["seed"],
        "substitution_policy": protocol["substitution_policy"],
        "require_supervised": protocol["require_supervised"],
    }
    if any(
        protocol_settings.get(key) != value
        for key, value in expected_protocol_values.items()
    ):
        raise ValueError("P6-A config and Protocol B manifest disagree")
    actual_references = sorted(
        {master["reference_scene_id"] for master in protocol_manifest["masters"]}
    )
    if protocol_settings.get("reference_scene_ids") != actual_references:
        raise ValueError("P6-A reference clusters differ from Protocol B manifest")
    for role, config_ref_key, config_sha_key in (
        ("sequence_database", "sequence_database", "sequence_database_sha256"),
        ("scan_metadata", "scan_metadata", "scan_metadata_sha256"),
        ("metadata", "metadata", "metadata_sha256"),
    ):
        descriptor = manifest_sources[role]
        if (
            configured_sources.get(config_ref_key) != descriptor["reference"]
            or configured_sources.get(config_sha_key) != descriptor["sha256"]
        ):
            raise ValueError(f"P6-A {role} provenance differs from Protocol B")
    expected_keys = _expected_cache_keys(protocol_manifest)
    cache_provenance = cache_manifest["provenance"]
    expected_config_digest = _config_digest(
        {"p6a": p6a_bytes, "runtime": runtime_bytes}
    )
    protocol_digest = _sha256(
        json.dumps(
            protocol_manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    expected_provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": P5_FROZEN_VALUES["checkpoint_sha256"],
        "config_sha256": expected_config_digest,
        "dataset_sha256": protocol_digest,
    }
    validate_cache_manifest(
        cache_manifest,
        expected_keys=expected_keys,
        expected_provenance=expected_provenance,
    )
    if dict(cache_provenance) != expected_provenance:
        raise ValueError("cache provenance differs from the P6-A build inputs")
    efficiency_provenance = efficiency_manifest["provenance"]
    expected_efficiency_provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": P5_FROZEN_VALUES["checkpoint_sha256"],
        "config_sha256": expected_config_digest,
        "protocol_sha256": _sha256(
            _json_text(protocol_manifest).encode("utf-8")
        ),
        "cache_manifest_sha256": _sha256(
            _json_text(cache_manifest).encode("utf-8")
        ),
    }
    if efficiency_provenance != expected_efficiency_provenance:
        raise ValueError("efficiency provenance differs from the P6-A build inputs")

    metric_blocks = normalize_official_metric_blocks(evaluation.metric_blocks)
    events = association_event_rows(evaluation.association_events)
    per_sequence = per_sequence_result_rows(evaluation.association_events)
    failures = failure_breakdown_rows(evaluation.association_events)
    reactivation = reactivation_audit_rows(evaluation.association_events)
    histogram = settings["diagnostics"]["reactivation_histograms"]
    score_distribution = reactivation_distribution_rows(
        evaluation.association_events,
        field="best_score",
        edges=histogram["best_score_edges"],
    )
    margin_distribution = reactivation_distribution_rows(
        evaluation.association_events,
        field="score_margin",
        edges=histogram["score_margin_edges"],
    )
    reactivation_gap = reactivation_by_gap_rows(evaluation.association_events)
    capacity = capacity_audit_rows(evaluation.capacity_snapshots)
    metrics = metric_table_rows(metric_blocks, per_sequence)
    bootstrap = settings["bootstrap"]
    gate_results, detailed_gates, gate_errors = _gate_analysis(
        metric_blocks=metric_blocks,
        fingerprints=evaluation.fingerprints,
        per_sequence=per_sequence,
        reactivation=reactivation,
        failures=failures,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
    )

    efficiency = aggregate_efficiency_rows(efficiency_manifest)
    csv_rows: dict[str, Sequence[Mapping[str, object]]] = {
        **metrics,
        "per_sequence_results.csv": per_sequence,
        "association_events.csv": events,
        "error_breakdown.csv": failures,
        **{
            f"error_breakdown_T{horizon}.csv": tuple(
                row for row in failures if int(row["T"]) == horizon
            )
            for horizon in HORIZONS
        },
        "reactivation_audit.csv": reactivation,
        "reactivation_score_distribution.csv": score_distribution,
        "reactivation_margin_distribution.csv": margin_distribution,
        "reactivation_by_gap.csv": reactivation_gap,
        "capacity_audit.csv": capacity,
        "efficiency_results.csv": efficiency,
    }
    derived_csv = {
        path: {
            "columns": list(
                CSV_COLUMN_SCHEMAS[
                    "error_breakdown.csv"
                    if path.startswith("error_breakdown_T")
                    else path
                ]
            ),
            "rows": list(rows),
        }
        for path, rows in csv_rows.items()
    }
    figure_c_rows = [
        {
            "method_id": str(row["method"]).casefold(),
            "horizon": row["T"],
            "outcome": row["outcome"],
            "metric": metric,
            "bin_low": row["bin_low"],
            "bin_high": row["bin_high"],
            "count": row["count"],
            "fraction": row["fraction"],
        }
        for metric, rows in (
            ("best_score", score_distribution),
            ("score_margin", margin_distribution),
        )
        for row in rows
    ]
    statistical_text = _statistical_markdown(detailed_gates)
    protocol_text = _json_text(protocol_manifest)
    cache_text = _json_text(cache_manifest).encode("utf-8")
    source_descriptor = protocol_manifest["sources"]["sequence_database"]
    config_descriptor = protocol_manifest["sources"]["config"]
    if config_descriptor["sha256"] != _sha256(p6a_bytes):
        raise ValueError("P6-A config bytes differ from Protocol B provenance")
    claims_supported = ["Exact common-prefix Protocol B evaluation completed."]
    claims_not_supported = [
        "Metadata order is not claimed to be real chronology.",
        "Native arbitrary-order change-label evidence is unavailable.",
        "P6-A does not claim to repair or reproduce an external benchmark score.",
    ]
    for gate_id, claim in (
        ("G6A-1", "B4 reduces long-horizon identity switches against B3."),
        ("G6A-2", "B4 improves dormant-track reactivation against B3."),
        ("G6A-3", "All methods use exactly the same frozen local predictions."),
        ("G6A-4", "B4 preserves short-horizon quality and adds long-horizon utility."),
        ("G6A-5", "The preregistered failure taxonomy explains at least 90 percent."),
    ):
        (claims_supported if gate_results[gate_id]["passed"] else claims_not_supported).append(
            claim
        )

    artifact: dict[str, object] = {
        "schema_version": 2,
        "status": "pass",
        "run_id": run_id
        or f"p6a-{source_commit[:12]}-{cache_manifest['entries_sha256'][:12]}",
        "source_commit": source_commit,
        "source_tree_contract": {"status": "pass", "source_commit": source_commit},
        "p5_frozen_hashes": dict(P5_FROZEN_VALUES),
        "protocol": {
            "name": "exact_common_prefix_protocol_b",
            "horizons": list(HORIZONS),
            "master_sequence_count": 43,
            "cluster_count": 6,
            "order_count": 3,
            "cache_entry_count": 645,
        },
        "provenance": {
            "checkpoint": {
                "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
                "sha256": P5_FROZEN_VALUES["checkpoint_sha256"],
            },
            "config": {
                "ref": config_descriptor["reference"],
                "sha256": config_descriptor["sha256"],
            },
            "dataset": {
                "ref": source_descriptor["reference"],
                "sha256": source_descriptor["sha256"],
            },
            "prediction_cache": {
                "ref": "local_cache:p6a/cache_manifest.json",
                "sha256": _sha256(cache_text),
            },
        },
        "methods": {
            "set": list(METHOD_IDS),
            "oracle": {"mode": "offline", "metric_block": "offline"},
        },
        "horizons": {
            f"T{horizon}": {"sequence_count": 129} for horizon in HORIZONS
        },
        "settings": {
            "bootstrap_seed": int(bootstrap["seed"]),
            "bootstrap_replicates": int(bootstrap["replicates"]),
        },
        "metric_blocks": metric_blocks,
        "fingerprints": evaluation.fingerprints,
        "analysis": {
            "association": {
                "path": "association_events.csv",
                "rows": len(events),
                "status": "pass",
            },
            "error": {
                "path": "error_breakdown.csv",
                "rows": len(failures),
                "status": "pass",
            },
            "reactivation": {
                "path": "reactivation_audit.csv",
                "rows": len(reactivation),
                "status": "pass",
            },
            "capacity": {
                "path": "capacity_audit.csv",
                "rows": len(capacity),
                "status": "pass",
            },
            "efficiency": {
                "path": "efficiency_results.csv",
                "rows": len(efficiency),
                "status": "pass",
            },
            "statistical": {
                "path": "statistical_analysis.md",
                "rows": 1,
                "status": "pass",
            },
        },
        "change_label_limitation": {
            "available": False,
            "reason": "Native arbitrary-order multi-transition labels are unavailable.",
            "scope": "Identity and task metrics only; no change-label claim is made.",
        },
        "derived_artifacts": {
            "csv": derived_csv,
            "json": {
                "protocol_b_manifest.json": {"text": protocol_text},
                "efficiency_raw_manifest.json": {
                    "text": _json_text(efficiency_manifest)
                },
            },
            "markdown": {"statistical_analysis.md": {"text": statistical_text}},
            "svg": {
                "figures/figure_a_identity.svg": {
                    "text": render_figure_a_identity(
                        {
                            "method_id": str(row["method"]).casefold(),
                            "horizon": row["T"],
                            "id_switch_rate": row["id_switch_rate"],
                        }
                        for row in metrics["baseline_results.csv"]
                    )
                },
                "figures/figure_b_online_tmap.svg": {
                    "text": render_figure_b_online_tmap(
                        {
                            "method_id": str(row["method"]).casefold(),
                            "horizon": row["T"],
                            "online_t_mAP": row["t_mAP"],
                        }
                        for row in metrics["strict_online_results.csv"]
                    )
                },
                "figures/figure_c_reactivation.svg": {
                    "text": render_figure_c_reactivation(figure_c_rows)
                },
                "figures/figure_d_failures.svg": {
                    "text": render_figure_d_failures(
                        {
                            "method_id": str(row["method"]).casefold(),
                            "horizon": row["T"],
                            "category": row["category"],
                            "count": row["count"],
                            "share": row["share"],
                        }
                        for row in failures
                    )
                },
                "figures/figure_e_latency.svg": {
                    "text": render_figure_e_latency(_figure_e_rows(efficiency))
                },
            },
            "yaml": {
                "configs/resolved_runtime.yaml": {
                    "text": portable_runtime_config_text
                },
                "configs/p6a_default.yaml": {"text": p6a_config_text},
            },
        },
        "artifact_manifest": [],
        "gate_results": gate_results,
        "claims_supported": claims_supported,
        "claims_not_supported": claims_not_supported,
        "next_action": "Stop after P6-A and await explicit continuation authorization.",
        "errors": gate_errors,
    }
    return seal_artifact_manifest(artifact)


def load_json_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path.name} must contain a JSON mapping")
    return dict(value)
