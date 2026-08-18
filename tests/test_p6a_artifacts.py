from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import p6a_artifacts

GATE_IDS = p6a_artifacts.GATE_IDS
HORIZON_IDS = ("T2", "T3", "T4", "T5")
METHOD_IDS = ("B0", "B0_sanity", "B1", "B2", "B3", "B4", "Oracle")
P6A_REPORT_SECTIONS = p6a_artifacts.P6A_REPORT_SECTIONS
artifact_json_text = p6a_artifacts.artifact_json_text
publish_artifacts = p6a_artifacts.publish_artifacts
render_csv = p6a_artifacts.render_csv
render_go_nogo_report = p6a_artifacts.render_go_nogo_report
validate_root_artifact = p6a_artifacts.validate_root_artifact


def _api(name):
    implementation = getattr(p6a_artifacts, name, None)
    if implementation is not None:
        return implementation

    def missing(*args, **kwargs):
        raise AssertionError(f"missing required API: {name}")

    return missing


render_artifact_bundle = _api("render_artifact_bundle")
publish_root_artifact = _api("publish_root_artifact")
verify_artifact_manifest = _api("verify_artifact_manifest")


def _metric(value: float = 0.5) -> dict[str, float]:
    return {
        "AP": value,
        "AP50": value,
        "AP25": value,
        "REC": value,
        "t_mAP": value,
        "t_mAP50": value,
        "t_mAP25": value,
        "t_REC": value,
        "t_REC50": value,
        "t_REC25": value,
    }


def _metric_block(methods: tuple[str, ...]) -> dict[str, object]:
    return {
        method: {horizon: _metric() for horizon in HORIZON_IDS}
        for method in methods
    }


def _csv_spec(columns: tuple[str, ...] = ("value",)) -> dict[str, object]:
    return {"columns": list(columns), "rows": [{column: 0.5 for column in columns}]}


def _protocol_manifest() -> dict[str, object]:
    from scripts.p6a_protocol import DEFAULT_ORDER_VARIANTS, _seeded_permutation

    scan_ids = [f"scene{index:04d}_00" for index in range(5)]
    scan_indices = list(range(5))

    def order(visit_order: list[str]) -> dict[str, object]:
        indices = [scan_ids.index(scan_id) for scan_id in visit_order]
        return {
            "visit_order": visit_order,
            "scan_indices": indices,
            "prefixes": {
                str(horizon): {
                    "scan_ids": visit_order[:horizon],
                    "scan_indices": indices[:horizon],
                    "sequence_id": "-".join(visit_order[:horizon]),
                }
                for horizon in (2, 3, 4, 5)
            },
        }

    orders = {
        "canonical": order(scan_ids),
        "reverse": order(list(reversed(scan_ids))),
        "sha256_seed45": order(
            [scan_ids[position] for position in _seeded_permutation(scan_ids, seed=45)]
        ),
    }
    return {
        "schema_version": "protocol-b-v1",
        "protocol": {
            "name": "common-prefix",
            "split": "validation",
            "master_horizon": 5,
            "horizons": [2, 3, 4, 5],
            "expected_master_count": 1,
            "expected_reference_scene_clusters": 1,
            "order_variants": list(DEFAULT_ORDER_VARIANTS),
            "seed": 45,
            "order_semantics": "metadata_order_only_no_timestamps",
            "substitution_policy": "reject",
            "scan_index_resolution": "explicit_scan_id_metadata_map",
            "require_supervised": True,
        },
        "sources": {
            role: {"reference": f"repo:data/fixture/{role}", "sha256": "0" * 64}
            for role in ("sequence_database", "scan_metadata", "metadata", "source_manifest", "config")
        },
        "masters": [
            {
                "master_sequence_id": "-".join(scan_ids),
                "reference_scene_id": "fixture-reference",
                "scan_ids": scan_ids,
                "scan_indices": scan_indices,
                "visit_order": scan_ids,
                "orders": orders,
                "prefixes": {name: value["prefixes"] for name, value in orders.items()},
            }
        ],
    }


def _derived_artifacts() -> dict[str, object]:
    methods = tuple(METHOD_IDS[:-1])
    horizons = tuple(range(2, 6))
    baseline_rows = [
        {
            "method": method,
            "T": horizon,
            "raw_AP": 0.5,
            "online_t_mAP": 0.5,
            "online_t_REC": 0.5,
            "id_switch_rate": 0.5,
            "reactivation_accuracy": 0.5,
        }
        for method in methods
        for horizon in horizons
    ]
    strict_rows = [
        {
            "method": method,
            "T": horizon,
            "t_mAP": 0.5,
            "t_mAP50": 0.5,
            "t_mAP25": 0.5,
            "t_REC": 0.5,
            "t_REC50": 0.5,
            "t_REC25": 0.5,
        }
        for method in methods
        for horizon in horizons
    ]
    raw_rows = [
        {
            "method": method,
            "T": horizon,
            "AP": 0.5,
            "AP50": 0.5,
            "AP25": 0.5,
            "REC": 0.5,
        }
        for method in methods
        for horizon in horizons
    ]
    units = [
        (
            f"reference-{index % 6}",
            f"master-{index}",
            f"scene-{index}",
            f"sequence-{index}",
            order,
        )
        for index in range(43)
        for order in ("canonical", "reverse", "sha256_seed45")
    ]
    per_sequence_rows = [
        {
            "method": method,
            "reference_scene_id": reference,
            "master_sequence_id": master,
            "scene_id": scene,
            "sequence_id": sequence,
            "order_id": order,
            "prefix": horizon,
            "T": horizon,
            "prediction_digest": "a" * 64,
            "id_switches": 0,
            "transition_opportunities": 1,
            "id_switch_rate": 0.0,
            "active_correct_matches": 1,
            "active_wrong_matches": 0,
            "births": 1,
            "false_births": 0,
            "rejected_births": 0,
            "fragmentation_count": 0,
            "merge_count": 0,
            "gap_opportunities": 1,
            "reactivation_attempts": 1,
            "predicted_reactivation_events": 1,
            "correct_reactivations": 1,
            "wrong_reactivations": 0,
            "no_attempts": 0,
            "reactivation_accuracy": 1.0,
            "reactivation_precision": 1.0,
            "reactivation_recall": 1.0,
            "reactivation_coverage": 1.0,
        }
        for method in methods
        for horizon in horizons
        for reference, master, scene, sequence, order in units
    ]
    digest = "a" * 64
    association_row = {
        field: None for field in p6a_artifacts.CSV_COLUMN_SCHEMAS["association_events.csv"]
    }
    association_row.update(
        {
            "event_id": "event-0",
            "scene_id": "scene-0",
            "sequence_id": "sequence-0",
            "reference_scene_id": "reference-0",
            "master_sequence_id": "master-0",
            "order_id": "canonical",
            "prefix": 2,
            "method": "B4",
            "stage_id": 0,
            "event_kind": "prediction",
            "query_id": "q-0",
            "candidate_slot_id": "slot-0",
            "predicted_identity_id": "identity-0",
            "gt_entity_id": "gt-0",
            "association_correct": True,
            "feature_similarity": 0.5,
            "class_similarity": 0.5,
            "total_score": 0.5,
            "best_score": 0.5,
            "second_best_score": 0.25,
            "score_margin": 0.25,
            "observation_confidence": 0.5,
            "mask_support": 1.0,
            "predicted_class": "class-0",
            "class_entropy": 0.5,
            "slot_age": 1,
            "last_seen_stage": 0,
            "gap_length": 0,
            "slot_active": True,
            "slot_occupied": True,
            "association_result": "active_correct",
            "gt_present": True,
            "prediction_present": True,
            "transition_opportunity": False,
            "id_switch": False,
            "gap_opportunity": False,
            "reactivation_attempt": False,
            "reactivation_correct": None,
            "new_birth": False,
            "false_birth": False,
            "reactivation": False,
            "wrong_reactivation": False,
            "local_observation_available": True,
            "local_match_available": True,
            "raw_local_match": True,
            "raw_prediction_available": True,
            "local_perception_miss": False,
            "association_miss": False,
            "association_attempted": True,
            "identity_fragmentation": False,
            "identity_merge": False,
            "fragmentation": False,
            "merge": False,
            "semantic_drift": False,
            "semantic_mismatch": False,
            "capacity_failure": False,
            "capacity_birth_failure": False,
            "birth_rejected": False,
            "is_failure": False,
            "failure_category": None,
            "failure_code": None,
            "prediction_digest": digest,
            "cache_digest": digest,
        }
    )
    error_rows = [
        {
            "method": method,
            "T": horizon,
            "category": category,
            "count": 1,
            "share": 1 / len(p6a_artifacts.FAILURE_CATEGORIES),
        }
        for method in methods
        for horizon in horizons
        for category in p6a_artifacts.FAILURE_CATEGORIES
    ]
    reactivation_rows = [
        {
            "method": method,
            "T": horizon,
            "gap_opportunities": 1,
            "reactivation_attempts": 1,
            "correct_reactivations": 1,
            "wrong_reactivations": 0,
            "no_attempts": 0,
            "reactivation_accuracy": 1.0,
            "reactivation_precision": 1.0,
            "reactivation_recall": 1.0,
            "reactivation_coverage": 1.0,
        }
        for method in p6a_artifacts.REACTIVATION_METHOD_SET
        for horizon in p6a_artifacts.REACTIVATION_HORIZONS
    ]
    distribution_rows = [
        {
            "method": method,
            "T": horizon,
            "outcome": outcome,
            "bin_low": 0.0,
            "bin_high": 1.0,
            "count": 1,
            "fraction": 1.0,
        }
        for method in p6a_artifacts.REACTIVATION_METHOD_SET
        for horizon in p6a_artifacts.REACTIVATION_HORIZONS
        for outcome in ("correct", "wrong")
    ]
    gap_rows = [
        {
            "method": method,
            "T": horizon,
            "gap_length": 1,
            "outcome": outcome,
            "count": 1,
            "fraction": 0.5,
        }
        for method in p6a_artifacts.REACTIVATION_METHOD_SET
        for horizon in p6a_artifacts.REACTIVATION_HORIZONS
        for outcome in ("correct", "wrong")
    ]
    capacity_rows = [
        {
            "method": method,
            "T": horizon,
            "stage_id": stage,
            "capacity": 100,
            "birth_count": 1,
            "occupied_count": 1,
            "active_count": 1,
            "dormant_count": 0,
            "peak_occupied": 1,
            "peak_active": 1,
            "peak_dormant": 0,
            "occupancy_ratio": 0.01,
            "rejected_births": 0,
            "persistent_state_bytes": 63808,
        }
        for method in ("B4",)
        for horizon in horizons
        for stage in range(horizon)
    ]
    efficiency_rows = [
        {
            "method": method,
            "T": horizon,
            "stage_id": stage,
            "row_type": row_type,
            "count": 1,
            "bootstrap_latency_ms": 1.0 if row_type == "bootstrap" else None,
            "new_visit_latency_ms": 1.0 if row_type == "new_visit" else None,
            "association_overhead_ms": 0.1 if row_type == "new_visit" else None,
            "memory_update_overhead_ms": 0.1 if row_type == "new_visit" else None,
            "full_history_latency_ms": 1.0 if row_type == "full_history" else None,
            "gpu_peak_memory_bytes": 100,
            "persistent_state_bytes": 63808,
        }
        for method, row_type in (("B4", "bootstrap"), ("B4", "new_visit"), ("full_history_rescene", "full_history"))
        for horizon in horizons
        for stage in range(5)
    ]
    return {
        "csv": {
            "baseline_results.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["baseline_results.csv"]), "rows": baseline_rows},
            "strict_online_results.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["strict_online_results.csv"]), "rows": strict_rows},
            "raw_local_results.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["raw_local_results.csv"]), "rows": raw_rows},
            "per_sequence_results.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["per_sequence_results.csv"]), "rows": per_sequence_rows},
            "association_events.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["association_events.csv"]), "rows": [association_row]},
            "error_breakdown.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["error_breakdown.csv"]), "rows": error_rows},
            **{
                f"error_breakdown_T{horizon}.csv": {
                    "columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["error_breakdown.csv"]),
                    "rows": [row for row in error_rows if row["T"] == horizon],
                }
                for horizon in horizons
            },
            "reactivation_audit.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["reactivation_audit.csv"]), "rows": reactivation_rows},
            "reactivation_score_distribution.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["reactivation_score_distribution.csv"]), "rows": distribution_rows},
            "reactivation_margin_distribution.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["reactivation_margin_distribution.csv"]), "rows": distribution_rows},
            "reactivation_by_gap.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["reactivation_by_gap.csv"]), "rows": gap_rows},
            "capacity_audit.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["capacity_audit.csv"]), "rows": capacity_rows},
            "efficiency_results.csv": {"columns": list(p6a_artifacts.CSV_COLUMN_SCHEMAS["efficiency_results.csv"]), "rows": efficiency_rows},
        },
        "json": {"protocol_b_manifest.json": {"text": json.dumps(_protocol_manifest(), sort_keys=True) + "\n"}},
        "markdown": {"statistical_analysis.md": {"text": "# Statistics\n"}},
        "svg": {
            f"figures/figure_{name}.svg": {"text": f"<svg id='{name}'/>\n"}
            for name in ("a_identity", "b_online_tmap", "c_reactivation", "d_failures", "e_latency")
        },
        "yaml": {
            "configs/resolved_runtime.yaml": {"text": "runtime:\n  device: cpu\n"},
            "configs/p6a_default.yaml": {"text": "protocol_b:\n  master_horizon: 5\n"},
        },
    }


def _artifact() -> dict[str, object]:
    artifact = {
        "schema_version": 2,
        "status": "pass",
        "run_id": "p6a-test",
        "source_commit": "1" * 40,
        "source_tree_contract": {
            "status": "pass",
            "source_commit": "1" * 40,
        },
        "p5_frozen_hashes": dict(p6a_artifacts.P5_FROZEN_VALUES),
        "protocol": {
            "name": "exact_common_prefix_protocol_b",
            "horizons": [2, 3, 4, 5],
            "master_sequence_count": 43,
            "cluster_count": 6,
            "order_count": 3,
            "cache_entry_count": 645,
        },
        "provenance": {
            "checkpoint": {
                "ref": "repo:checkpoints/rescene4d_concerto_t2_repro.ckpt",
                "sha256": p6a_artifacts.P5_FROZEN_VALUES["checkpoint_sha256"],
            },
            "config": {"ref": "repo:conf/p6a/default.yaml", "sha256": "3" * 64},
            "dataset": {
                "ref": "repo:data/processed/rio/sequence_database_sliding_5.yaml",
                "sha256": "4" * 64,
            },
            "prediction_cache": {
                "ref": "local_cache:p6a/cache_manifest.json",
                "sha256": "5" * 64,
            },
        },
        "methods": {
            "set": list(METHOD_IDS),
            "oracle": {"mode": "offline", "metric_block": "offline"},
        },
        "horizons": {
            "T2": {"sequence_count": 129},
            "T3": {"sequence_count": 129},
            "T4": {"sequence_count": 129},
            "T5": {"sequence_count": 129},
        },
        "settings": {"bootstrap_seed": 45, "bootstrap_replicates": 10_000},
        "metric_blocks": {
            "raw": _metric_block(("B0", "B0_sanity", "B1", "B2", "B3", "B4")),
            "strict": _metric_block(("B0", "B0_sanity", "B1", "B2", "B3", "B4")),
            "offline": _metric_block(METHOD_IDS),
        },
        "fingerprints": {
            "prediction": {method: "6" * 64 for method in METHOD_IDS},
            "cache": {method: "7" * 64 for method in METHOD_IDS},
        },
        "analysis": {
            "association": {"path": "association_events.csv", "rows": 1, "status": "pass"},
            "error": {"path": "error_breakdown.csv", "rows": 192, "status": "pass"},
            "reactivation": {"path": "reactivation_audit.csv", "rows": 12, "status": "pass"},
            "capacity": {"path": "capacity_audit.csv", "rows": 14, "status": "pass"},
            "efficiency": {"path": "efficiency_results.csv", "rows": 60, "status": "pass"},
            "statistical": {"path": "statistical_analysis.md", "rows": 1, "status": "pass"},
        },
        "change_label_limitation": {
            "available": False,
            "reason": "native multi-transition change labels are not available",
            "scope": "P6-A reports identity and task metrics without change labels",
        },
        "derived_artifacts": _derived_artifacts(),
        "artifact_manifest": [],
        "gate_results": {
            gate: {"passed": True, "evidence": f"quantitative evidence for {gate}"}
            for gate in GATE_IDS
        },
        "claims_supported": ["common-prefix evaluation completed"],
        "claims_not_supported": ["metadata order is real chronology"],
        "next_action": "stop_after_p6a",
        "errors": [],
    }
    placeholder_paths = {
        "P6A_GO_NOGO_REPORT.md",
        "protocol_b_manifest.json",
        "statistical_analysis.md",
    }
    for category in ("csv", "json", "markdown", "svg", "yaml"):
        placeholder_paths.update(artifact["derived_artifacts"][category])
    artifact["artifact_manifest"] = [
        {"path": path, "bytes": 1, "sha256": "0" * 64}
        for path in sorted(placeholder_paths)
    ]
    rendered = _fixture_rendered(artifact)
    artifact["artifact_manifest"] = [
        {
            "path": path,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(rendered.items())
        if path != p6a_artifacts.ROOT_ARTIFACT_PATH
    ]
    return artifact


def _fixture_rendered(artifact: dict[str, object]) -> dict[str, bytes]:
    implementation = getattr(p6a_artifacts, "render_artifact_bundle", None)
    if implementation is not None:
        return implementation(artifact)
    files: dict[str, bytes] = {
        "P6A_GO_NOGO_REPORT.md": b"# P6A report\n",
        "protocol_b_manifest.json": b'{"protocol":"B"}\n',
        "statistical_analysis.md": b"# Statistics\n",
    }
    derived = artifact["derived_artifacts"]
    for path, spec in derived["csv"].items():
        files[path] = render_csv(spec["rows"], columns=spec["columns"]).encode()
    for category in ("json", "markdown", "svg", "yaml"):
        for path, spec in derived[category].items():
            files[path] = spec["text"].encode()
    return files


def test_complete_root_schema_and_manifest_are_bound() -> None:
    artifact = _artifact()

    validate_root_artifact(artifact)
    files = render_artifact_bundle(artifact)

    assert verify_artifact_manifest(artifact, files)
    assert {entry["path"] for entry in artifact["artifact_manifest"]} == (
        set(files) - {p6a_artifacts.ROOT_ARTIFACT_PATH}
    )
    assert artifact["protocol"] == {
        "name": "exact_common_prefix_protocol_b",
        "horizons": [2, 3, 4, 5],
        "master_sequence_count": 43,
        "cluster_count": 6,
        "order_count": 3,
        "cache_entry_count": 645,
    }


def test_root_json_is_published_but_excluded_from_self_referential_manifest() -> None:
    artifact = _artifact()

    files = render_artifact_bundle(artifact)

    assert p6a_artifacts.ROOT_ARTIFACT_PATH in files
    assert artifact_json_text(artifact).encode("utf-8") == files[
        p6a_artifacts.ROOT_ARTIFACT_PATH
    ]
    assert p6a_artifacts.ROOT_ARTIFACT_PATH not in {
        entry["path"] for entry in artifact["artifact_manifest"]
    }
    assert verify_artifact_manifest(artifact, files)


def test_protocol_manifest_and_derived_csv_semantics_are_validated() -> None:
    artifact = _artifact()

    artifact["derived_artifacts"]["json"]["protocol_b_manifest.json"]["text"] = "{}\n"
    with pytest.raises(ValueError, match="protocol.*manifest"):
        validate_root_artifact(artifact)

    artifact = _artifact()
    artifact["derived_artifacts"]["csv"]["baseline_results.csv"] = _csv_spec(
        ("value",)
    )
    with pytest.raises(ValueError, match="baseline_results|columns|Table"):
        validate_root_artifact(artifact)


def test_aggregate_csv_rows_must_be_reconstructible() -> None:
    artifact = _artifact()
    audit_rows = artifact["derived_artifacts"]["csv"]["reactivation_audit.csv"]["rows"]
    audit_rows[0]["correct_reactivations"] = 2
    with pytest.raises(ValueError, match="reactivation_audit"):
        validate_root_artifact(artifact)

    artifact = _artifact()
    error_rows = artifact["derived_artifacts"]["csv"]["error_breakdown.csv"]["rows"]
    error_rows[0]["count"] = 2
    with pytest.raises(ValueError, match="error_breakdown"):
        validate_root_artifact(artifact)


def test_error_breakdown_preserves_unclassified_failures() -> None:
    artifact = _artifact()
    csv_artifacts = artifact["derived_artifacts"]["csv"]
    aggregate_rows = csv_artifacts["error_breakdown.csv"]["rows"]
    aggregate_rows[:] = [
        row for row in aggregate_rows if row["category"] != "unclassified"
    ]
    for row in aggregate_rows:
        row["share"] = 1.0 / 7.0
    for horizon in p6a_artifacts.HORIZON_IDS:
        csv_artifacts[f"error_breakdown_{horizon}.csv"]["rows"] = [
            row for row in aggregate_rows if row["T"] == horizon
        ]
    artifact["analysis"]["error"]["rows"] = 168

    with pytest.raises(ValueError, match="unclassified"):
        validate_root_artifact(artifact)


def test_error_breakdown_preserves_a_zero_failure_group() -> None:
    artifact = _artifact()
    csv_artifacts = artifact["derived_artifacts"]["csv"]
    for path in ("error_breakdown.csv", "error_breakdown_T2.csv"):
        for row in csv_artifacts[path]["rows"]:
            if row["method"] == "B4" and row["T"] == 2:
                row["count"] = 0
                row["share"] = 0.0

    validate_root_artifact(artifact)


def test_reactivation_distributions_preserve_empty_outcomes() -> None:
    artifact = _artifact()
    csv_artifacts = artifact["derived_artifacts"]["csv"]
    for path in (
        "reactivation_score_distribution.csv",
        "reactivation_margin_distribution.csv",
    ):
        for row in csv_artifacts[path]["rows"]:
            if row["outcome"] == "wrong":
                row["count"] = 0
                row["fraction"] = 0.0
    for row in csv_artifacts["reactivation_by_gap.csv"]["rows"]:
        if row["method"] == "B1" and row["T"] == 3:
            row["count"] = 0
            row["fraction"] = 0.0

    validate_root_artifact(artifact)


def test_capacity_audit_is_limited_to_b4_bounded_state() -> None:
    artifact = _artifact()
    artifact["derived_artifacts"]["csv"]["capacity_audit.csv"]["rows"][0][
        "method"
    ] = "B3"

    with pytest.raises(ValueError, match="capacity_audit|B4"):
        validate_root_artifact(artifact)


def test_capacity_audit_rejects_stages_outside_each_prefix() -> None:
    artifact = _artifact()
    row = artifact["derived_artifacts"]["csv"]["capacity_audit.csv"]["rows"][0]
    row["stage_id"] = 2

    with pytest.raises(ValueError, match="capacity_audit|prefix|stage"):
        validate_root_artifact(artifact)


def test_p5_frozen_values_and_prediction_cache_reference_are_exact() -> None:
    artifact = _artifact()

    artifact["p5_frozen_hashes"]["source_commit"] = "1" * 40
    with pytest.raises(ValueError, match="p5_frozen_hashes"):
        validate_root_artifact(artifact)

    artifact = _artifact()
    artifact["provenance"]["prediction_cache"]["ref"] = (
        "repo:artifacts/P6A/cache_manifest.json"
    )
    with pytest.raises(ValueError, match="prediction_cache"):
        validate_root_artifact(artifact)


def test_scalar_tree_rejects_relative_parent_traversal() -> None:
    artifact = _artifact()
    artifact["claims_supported"].append("../private-result.json")

    with pytest.raises(ValueError, match="parent|travers"):
        validate_root_artifact(artifact)


def test_publish_bundle_uses_one_directory_replace(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "P6A"
    calls: list[tuple[Path, Path]] = []
    original_replace = p6a_artifacts.os.replace

    def record_replace(source, target):
        calls.append((Path(source), Path(target)))
        return original_replace(source, target)

    monkeypatch.setattr(p6a_artifacts.os, "replace", record_replace)
    publish_artifacts(root, {"a.csv": "a\n", "b.csv": "b\n"})

    assert len(calls) == 1
    assert calls[0][1] == root
    assert calls[0][0].parent == root.parent
    assert calls[0][0].name.startswith(".p6a-stage-")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("protocol"),
        lambda value: value.update(extra="forbidden"),
        lambda value: value.update(schema_version=True),
        lambda value: value["protocol"].update(master_sequence_count=42),
        lambda value: value["protocol"].update(cluster_count=5),
        lambda value: value["protocol"].update(order_count=4),
        lambda value: value["protocol"].update(cache_entry_count=644),
        lambda value: value["methods"]["set"].remove("B4"),
        lambda value: value["methods"]["oracle"].update(mode="online"),
        lambda value: value["metric_blocks"]["raw"].pop("B3"),
        lambda value: value["metric_blocks"]["strict"]["B1"].pop("T5"),
        lambda value: value["metric_blocks"].pop("offline"),
        lambda value: value["fingerprints"]["prediction"].update(B0="bad"),
        lambda value: value["analysis"].pop("error"),
        lambda value: value["change_label_limitation"].update(available=True),
        lambda value: value["artifact_manifest"].clear(),
        lambda value: value["artifact_manifest"][0].update(extra="forbidden"),
        lambda value: value["derived_artifacts"]["csv"]["error_breakdown.csv"]["rows"].clear(),
        lambda value: value["metric_blocks"]["raw"]["B0"]["T2"].update(AP=float("nan")),
        lambda value: value["provenance"]["config"].update(ref="/home/user/config.yaml"),
        lambda value: value["provenance"]["config"].update(ref="repo:artifacts/P5/x"),
        lambda value: value["claims_supported"].append("10.0.0.1"),
        lambda value: value["claims_supported"].append("GPU-12345678-abcd"),
    ],
)
def test_root_artifact_fails_closed_on_invalid_contract(mutation):
    artifact = _artifact()
    mutation(artifact)

    with pytest.raises(ValueError):
        validate_root_artifact(artifact)


def test_artifact_json_and_report_are_deterministic_and_gate_driven() -> None:
    artifact = _artifact()

    first = artifact_json_text(artifact)
    second = artifact_json_text(copy.deepcopy(artifact))
    report = render_go_nogo_report(artifact)

    assert first == second
    assert json.loads(first) == artifact
    assert first.endswith("\n")
    assert all(report.count(f"## {section}") == 1 for section in P6A_REPORT_SECTIONS)
    assert report.count("Decision: P6A_GO") == 1
    assert report.count("Decision:") == 1

    artifact["gate_results"]["G6A-2"]["passed"] = False
    stopped = render_go_nogo_report(artifact)
    assert "Decision: P6A_STOP" in stopped
    assert "P6B" not in stopped


def test_csv_renderer_has_stable_columns_and_rejects_schema_drift() -> None:
    rows = [
        {"method_id": "b1", "horizon": 2, "value": 0.5},
        {"method_id": "b4", "horizon": 2, "value": None},
    ]

    text = render_csv(rows, columns=("method_id", "horizon", "value"))

    assert text == "method_id,horizon,value\nb1,2,0.5\nb4,2,\n"
    with pytest.raises(ValueError):
        render_csv([{"method_id": "b1", "horizon": 2}], columns=("method_id",))


def test_manifest_reverification_rejects_changed_derived_bytes() -> None:
    artifact = _artifact()
    files = render_artifact_bundle(artifact)
    files["error_breakdown.csv"] = files["error_breakdown.csv"] + b"tamper"

    with pytest.raises(ValueError):
        verify_artifact_manifest(artifact, files)


def test_publish_renders_and_verifies_every_file_before_atomic_publish(tmp_path: Path) -> None:
    artifact = _artifact()
    root = tmp_path / "artifacts" / "P6A"

    published = publish_root_artifact(root, artifact)

    assert [path.relative_to(root).as_posix() for path in published] == sorted(
        path.relative_to(root).as_posix() for path in published
    )
    for path in published:
        assert path.is_file()
        assert not path.is_symlink()
    with pytest.raises(FileExistsError):
        publish_root_artifact(root, artifact)


def test_publish_rejects_symlink_and_nonregular_outputs(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)
    root = linked_parent / "P6A"

    with pytest.raises(ValueError):
        publish_artifacts(root, {"figures/x.svg": "<svg/>\n"})

    regular_root = tmp_path / "regular"
    regular_root.write_text("existing\n")
    with pytest.raises(FileExistsError):
        publish_artifacts(regular_root, {"x.csv": "x\n"})


def test_failed_multi_file_publish_cleans_staging_and_publishes_nothing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "P6A"
    calls = {"count": 0}

    from scripts import p6a_artifacts

    original_replace = p6a_artifacts.os.replace

    def fail_first(source, target):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(p6a_artifacts.os, "replace", fail_first)
    with pytest.raises(OSError):
        publish_artifacts(root, {"a.csv": "a\n", "b.csv": "b\n"})

    assert not (root / "a.csv").exists()
    assert not (root / "b.csv").exists()
    assert not list(tmp_path.glob(".p6a-stage-*"))
