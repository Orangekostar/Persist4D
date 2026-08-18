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
    }


def _metric_block(methods: tuple[str, ...]) -> dict[str, object]:
    return {
        method: {horizon: _metric() for horizon in HORIZON_IDS}
        for method in methods
    }


def _csv_spec(columns: tuple[str, ...] = ("value",)) -> dict[str, object]:
    return {"columns": list(columns), "rows": [{column: 0.5 for column in columns}]}


def _derived_artifacts() -> dict[str, object]:
    return {
        "csv": {
            "baseline_results.csv": _csv_spec(("method", "T", "value")),
            "strict_online_results.csv": _csv_spec(("method", "T", "value")),
            "raw_local_results.csv": _csv_spec(("method", "T", "value")),
            "per_sequence_results.csv": _csv_spec(("sequence", "T", "value")),
            "association_events.csv": _csv_spec(("sequence", "event")),
            "error_breakdown.csv": _csv_spec(("method", "category", "share")),
            "reactivation_audit.csv": _csv_spec(("method", "outcome", "value")),
            "capacity_audit.csv": _csv_spec(("method", "T", "occupied")),
            "efficiency_results.csv": _csv_spec(("method", "T", "latency_ms")),
        },
        "json": {"protocol_b_manifest.json": {"text": '{"protocol":"B"}\n'}},
        "markdown": {"statistical_analysis.md": {"text": "# Statistics\n"}},
        "svg": {
            f"figures/figure_{name}.svg": {"text": f"<svg id='{name}'/>\n"}
            for name in ("a_identity", "b_online_tmap", "c_reactivation", "d_failures", "e_latency")
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
        "p5_frozen_hashes": {
            "git_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
        },
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
                "sha256": "2" * 64,
            },
            "config": {"ref": "repo:conf/p6a/default.yaml", "sha256": "3" * 64},
            "dataset": {
                "ref": "repo:data/processed/rio/sequence_database_sliding_5.yaml",
                "sha256": "4" * 64,
            },
            "prediction_cache": {
                "ref": "repo:artifacts/P6A/cache_manifest.json",
                "sha256": "5" * 64,
            },
        },
        "methods": {
            "set": list(METHOD_IDS),
            "oracle": {"mode": "offline", "metric_block": "offline"},
        },
        "horizons": {
            "T2": {"sequence_count": 154},
            "T3": {"sequence_count": 120},
            "T4": {"sequence_count": 75},
            "T5": {"sequence_count": 43},
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
            "error": {"path": "error_breakdown.csv", "rows": 1, "status": "pass"},
            "reactivation": {"path": "reactivation_audit.csv", "rows": 1, "status": "pass"},
            "capacity": {"path": "capacity_audit.csv", "rows": 1, "status": "pass"},
            "efficiency": {"path": "efficiency_results.csv", "rows": 1, "status": "pass"},
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
    for category in ("csv", "json", "markdown", "svg"):
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
    for category in ("json", "markdown", "svg"):
        for path, spec in derived[category].items():
            files[path] = spec["text"].encode()
    return files


def test_complete_root_schema_and_manifest_are_bound() -> None:
    artifact = _artifact()

    validate_root_artifact(artifact)
    files = render_artifact_bundle(artifact)

    assert verify_artifact_manifest(artifact, files)
    assert {entry["path"] for entry in artifact["artifact_manifest"]} == set(files)
    assert artifact["protocol"] == {
        "name": "exact_common_prefix_protocol_b",
        "horizons": [2, 3, 4, 5],
        "master_sequence_count": 43,
        "cluster_count": 6,
        "order_count": 3,
        "cache_entry_count": 645,
    }


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

    assert [path.name for path in published] == sorted(path.name for path in published)
    for path in published:
        assert path.is_file()
        assert not path.is_symlink()
    with pytest.raises(FileExistsError):
        publish_root_artifact(root, artifact)


def test_publish_rejects_symlink_and_nonregular_outputs(tmp_path: Path) -> None:
    root = tmp_path / "P6A"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "figures").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        publish_artifacts(root, {"figures/x.svg": "<svg/>\n"})

    regular_root = tmp_path / "regular"
    regular_root.mkdir()
    (regular_root / "x.csv").mkdir()
    with pytest.raises(FileExistsError):
        publish_artifacts(regular_root, {"x.csv": "x\n"})


def test_failed_multi_file_publish_cleans_staging_and_publishes_nothing(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "P6A"
    calls = {"count": 0}

    from scripts import p6a_artifacts

    original_replace = p6a_artifacts.os.replace

    def fail_second(source, target):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected publish failure")
        return original_replace(source, target)

    monkeypatch.setattr(p6a_artifacts.os, "replace", fail_second)
    with pytest.raises(OSError):
        publish_artifacts(root, {"a.csv": "a\n", "b.csv": "b\n"})

    assert not (root / "a.csv").exists()
    assert not (root / "b.csv").exists()
    assert not list(tmp_path.glob(".p6a-stage-*"))
