from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from scripts.evaluate_persist4d_p6a import TaskMetricEvaluation
from scripts.p6a_analysis import (
    AssociationEvent,
    CapacitySnapshot,
    persistent_state_bytes,
)
from scripts.p6a_artifacts import (
    ONLINE_METHOD_IDS,
    render_artifact_bundle,
    validate_root_artifact,
    verify_artifact_manifest,
)
from scripts.p6a_builder import (
    _expected_cache_keys,
    build_p6a_root_artifact,
    metric_table_rows,
    seal_artifact_manifest,
)
from scripts.p6a_cache import build_cache_manifest
from scripts.p6a_protocol import _seeded_permutation
from tests.test_p6a_artifacts import _artifact, _efficiency_manifest


def _metric(value: float) -> dict[str, float | None]:
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


def test_metric_tables_use_global_count_ratios_not_mean_sequence_rates() -> None:
    metric_blocks = {
        "raw": {
            method: {f"T{horizon}": _metric(0.2) for horizon in range(2, 6)}
            for method in ONLINE_METHOD_IDS
        },
        "strict": {
            method: {f"T{horizon}": _metric(0.3) for horizon in range(2, 6)}
            for method in ONLINE_METHOD_IDS
        },
    }
    per_sequence = [
        {
            "method": method,
            "T": horizon,
            "id_switches": 0,
            "transition_opportunities": 1,
            "correct_reactivations": 0,
            "reactivation_attempts": 0,
        }
        for method in ONLINE_METHOD_IDS
        for horizon in range(2, 6)
    ]
    per_sequence.extend(
        [
            {
                "method": "B4",
                "T": 2,
                "id_switches": 1,
                "transition_opportunities": 1,
                "correct_reactivations": 0,
                "reactivation_attempts": 0,
            },
            {
                "method": "B4",
                "T": 2,
                "id_switches": 0,
                "transition_opportunities": 8,
                "correct_reactivations": 0,
                "reactivation_attempts": 0,
            },
        ]
    )

    tables = metric_table_rows(metric_blocks, per_sequence)

    row = next(
        item
        for item in tables["baseline_results.csv"]
        if item["method"] == "B4" and item["T"] == 2
    )
    assert row["id_switch_rate"] == 0.1
    assert row["raw_AP"] == 0.2
    assert row["online_t_mAP"] == 0.3
    assert row["reactivation_accuracy"] is None


def test_seal_artifact_manifest_uses_rendered_bytes_without_mutating_input() -> None:
    artifact = _artifact()
    for record in artifact["artifact_manifest"]:
        record["bytes"] = 1
        record["sha256"] = "0" * 64
    original = copy.deepcopy(artifact)

    sealed = seal_artifact_manifest(artifact)
    rendered = render_artifact_bundle(sealed)

    assert artifact == original
    assert sealed["artifact_manifest"] != original["artifact_manifest"]
    validate_root_artifact(sealed)
    assert verify_artifact_manifest(sealed, rendered)


def _protocol_manifest(config_sha256: str) -> dict[str, object]:
    masters = []
    references = [f"reference-{index}" for index in range(6)]
    for master_index in range(43):
        scan_ids = [f"scene{master_index:04d}_{stage:02d}" for stage in range(5)]
        scan_indices = [master_index * 5 + stage for stage in range(5)]

        def order(
            visit_order: list[str],
            scan_ids: list[str] = scan_ids,
            scan_indices: list[int] = scan_indices,
        ) -> dict[str, object]:
            indices = [scan_ids.index(scan_id) for scan_id in visit_order]
            global_indices = [scan_indices[index] for index in indices]
            return {
                "visit_order": visit_order,
                "scan_indices": global_indices,
                "prefixes": {
                    str(horizon): {
                        "scan_ids": visit_order[:horizon],
                        "scan_indices": global_indices[:horizon],
                        "sequence_id": "-".join(visit_order[:horizon]),
                    }
                    for horizon in range(2, 6)
                },
            }

        seeded = [
            scan_ids[index] for index in _seeded_permutation(scan_ids, seed=45)
        ]
        orders = {
            "canonical": order(scan_ids),
            "reverse": order(list(reversed(scan_ids))),
            "sha256_seed45": order(seeded),
        }
        masters.append(
            {
                "master_sequence_id": "-".join(scan_ids),
                "reference_scene_id": references[master_index % 6],
                "scan_ids": scan_ids,
                "scan_indices": scan_indices,
                "visit_order": scan_ids,
                "orders": orders,
                "prefixes": {
                    name: value["prefixes"] for name, value in orders.items()
                },
            }
        )
    sources = {
        "sequence_database": {
            "reference": "repo:data/processed/rio/sequence_database_sliding_5.yaml",
            "sha256": "1" * 64,
        },
        "scan_metadata": {
            "reference": "repo:data/processed/rio/validation_database.yaml",
            "sha256": "2" * 64,
        },
        "metadata": {
            "reference": "external:3RScan/3RScan.json",
            "sha256": "3" * 64,
        },
        "source_manifest": {
            "reference": "repo:artifacts/environment/source_manifest.json",
            "sha256": "4" * 64,
        },
        "config": {
            "reference": "repo:conf/p6a/default.yaml",
            "sha256": config_sha256,
        },
    }
    return {
        "schema_version": "protocol-b-v1",
        "protocol": {
            "name": "common-prefix",
            "split": "validation",
            "master_horizon": 5,
            "horizons": [2, 3, 4, 5],
            "expected_master_count": 43,
            "expected_reference_scene_clusters": 6,
            "order_variants": ["canonical", "reverse", "sha256_seed45"],
            "seed": 45,
            "order_semantics": "metadata_order_only_no_timestamps",
            "substitution_policy": "reject",
            "scan_index_resolution": "explicit_scan_id_metadata_map",
            "require_supervised": True,
        },
        "sources": sources,
        "masters": masters,
    }


def _config_and_manifest() -> tuple[str, dict[str, object]]:
    config = yaml.safe_load(Path("conf/p6a/default.yaml").read_text(encoding="utf-8"))
    config["protocol_b"]["reference_scene_ids"] = [
        f"reference-{index}" for index in range(6)
    ]
    config["protocol_b"]["sources"].update(
        {
            "sequence_database_sha256": "1" * 64,
            "scan_metadata_sha256": "2" * 64,
            "metadata_sha256": "3" * 64,
        }
    )
    config_text = yaml.safe_dump(config, sort_keys=True)
    return config_text, _protocol_manifest(
        hashlib.sha256(config_text.encode()).hexdigest()
    )


def _raw_metric(value: float) -> dict[str, float]:
    return {
        "raw_local_AP": value,
        "raw_local_AP50": value,
        "raw_local_AP25": value,
        "raw_local_REC": value,
        "raw_local_REC50": value,
        "raw_local_REC25": value,
    }


def _temporal_metric(prefix: str, value: float) -> dict[str, float]:
    return {
        f"{prefix}t-mAP": value,
        f"{prefix}t-mAP50": value,
        f"{prefix}t-mAP25": value,
        f"{prefix}t-REC": value,
        f"{prefix}t-REC50": value,
        f"{prefix}t-REC25": value,
    }


def _evaluation(protocol: dict[str, object]) -> TaskMetricEvaluation:
    events = []
    digest = "a" * 64
    index = 0
    for method in ONLINE_METHOD_IDS:
        for horizon in range(2, 6):
            for master in protocol["masters"]:
                for order_id in ("canonical", "reverse", "sha256_seed45"):
                    prefix_ids = master["orders"][order_id]["visit_order"][:horizon]
                    reactivation = method in {"B1", "B2", "B3", "B4"} and horizon >= 3
                    correct = reactivation and method != "B3"
                    wrong = reactivation and method == "B3"
                    events.append(
                        AssociationEvent(
                            event_id=f"event-{index}",
                            scene_id=master["master_sequence_id"],
                            sequence_id="-".join(prefix_ids),
                            reference_scene_id=master["reference_scene_id"],
                            master_sequence_id=master["master_sequence_id"],
                            order_id=order_id,
                            prefix=horizon,
                            method=method,
                            stage_id=horizon - 1,
                            query_id=index,
                            predicted_identity_id=index,
                            gt_entity_id=index,
                            association_correct=not wrong,
                            best_score=0.8 if reactivation else None,
                            score_margin=0.4 if reactivation else None,
                            gap_length=1 if reactivation else None,
                            association_result=(
                                "reactivation_wrong"
                                if wrong
                                else "reactivation_correct"
                                if reactivation
                                else "active_correct"
                            ),
                            gt_present=True,
                            prediction_present=True,
                            transition_opportunity=True,
                            id_switch=method == "B3" and horizon >= 4,
                            gap_opportunity=reactivation,
                            reactivation_attempt=reactivation,
                            reactivation=reactivation,
                            reactivation_correct=correct if reactivation else None,
                            wrong_reactivation=wrong,
                            association_miss=wrong,
                            is_failure=wrong,
                            prediction_digest=digest,
                            cache_digest=digest,
                        )
                    )
                    index += 1
    strict = {}
    for method in ONLINE_METHOD_IDS:
        strict[method] = {}
        for horizon in range(2, 6):
            value = 0.3
            if method == "B4" and horizon >= 4:
                value = 0.4
            strict[method][f"T{horizon}"] = _temporal_metric("online_", value)
    offline = {
        method: {
            f"T{horizon}": _temporal_metric("offline_reconstructed_", 0.3)
            for horizon in range(2, 6)
        }
        for method in (*ONLINE_METHOD_IDS, "Oracle")
    }
    capacity = tuple(
        CapacitySnapshot(
            method="B4",
            horizon=horizon,
            stage_id=stage,
            capacity=100,
            birth_count=1,
            occupied_count=stage + 1,
            active_count=1,
            dormant_count=stage,
            rejected_births=0,
            persistent_state_bytes=persistent_state_bytes(100, 128, 18),
            feature_dim=128,
            class_count=18,
        )
        for horizon in range(2, 6)
        for stage in range(horizon)
    )
    return TaskMetricEvaluation(
        metric_blocks={
            "raw": {
                method: {
                    f"T{horizon}": _raw_metric(0.2) for horizon in range(2, 6)
                }
                for method in ONLINE_METHOD_IDS
            },
            "strict": strict,
            "offline": offline,
        },
        fingerprints={
            kind: {method: digest for method in (*ONLINE_METHOD_IDS, "Oracle")}
            for kind in ("prediction", "cache")
        },
        sequence_count=129,
        association_events=tuple(events),
        capacity_snapshots=capacity,
    )


def test_complete_builder_derives_and_seals_one_root_artifact() -> None:
    source_commit = "b" * 40
    runtime_text = "runtime:\n  device: device-0\n"
    config_text, protocol = _config_and_manifest()
    protocol_digest = hashlib.sha256(
        json.dumps(
            protocol, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    config_hasher = hashlib.sha256()
    for name, content in sorted(
        {"p6a": config_text.encode(), "runtime": runtime_text.encode()}.items()
    ):
        config_hasher.update(name.encode() + b"\0")
        config_hasher.update(len(content).to_bytes(8, "big") + content)
    provenance = {
        "source_commit": source_commit,
        "checkpoint_sha256": "85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e",
        "config_sha256": config_hasher.hexdigest(),
        "dataset_sha256": protocol_digest,
    }
    expected_keys = _expected_cache_keys(protocol)
    entries = []
    for key in expected_keys:
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        entries.append(
            {
                "filename": f"{digest}.pt",
                "content_sha256": digest,
                "file_sha256": "c" * 64,
                "file_bytes": 1,
                "key": key,
            }
        )
    cache_manifest = build_cache_manifest(
        entries,
        expected_keys=expected_keys,
        expected_provenance=provenance,
    )

    artifact = build_p6a_root_artifact(
        evaluation=_evaluation(protocol),
        protocol_manifest=protocol,
        cache_manifest=cache_manifest,
        efficiency_manifest=_efficiency_manifest(),
        source_commit=source_commit,
        p6a_config_text=config_text,
        runtime_config_text=runtime_text,
    )

    validate_root_artifact(artifact)
    assert len(
        artifact["derived_artifacts"]["csv"]["per_sequence_results.csv"]["rows"]
    ) == 3096
    assert len(
        artifact["derived_artifacts"]["csv"]["capacity_audit.csv"]["rows"]
    ) == 14
    raw_manifest = json.loads(
        artifact["derived_artifacts"]["json"]["efficiency_raw_manifest.json"]["text"]
    )
    assert len(raw_manifest["records"]) == 1161
    assert len(
        artifact["derived_artifacts"]["csv"]["efficiency_results.csv"]["rows"]
    ) == 12
    assert artifact["gate_results"]["G6A-1"]["passed"] is True
    assert artifact["gate_results"]["G6A-5"]["passed"] is False
    assert verify_artifact_manifest(artifact, render_artifact_bundle(artifact))
