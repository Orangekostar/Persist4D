from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.p6a_artifacts import (
    P6A_REPORT_SECTIONS,
    artifact_json_text,
    publish_artifacts,
    render_csv,
    render_go_nogo_report,
    validate_root_artifact,
)


def _artifact() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "protocol": {
            "name": "exact_common_prefix_protocol_b",
            "horizons": [2, 3, 4, 5],
            "reference_scene_count": 6,
            "master_sequence_count": 43,
        },
        "run_id": "p6a-test",
        "source_commit": "1" * 40,
        "source_tree_contract": {"status": "pass"},
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
        "settings": {"bootstrap_seed": 45, "bootstrap_replicates": 10_000},
        "artifact_manifest": [],
        "gate_results": {
            f"G6A-{index}": {"passed": True, "evidence": f"gate {index}"}
            for index in range(1, 6)
        },
        "claims_supported": ["common-prefix evaluation completed"],
        "claims_not_supported": ["metadata order is real chronology"],
        "next_action": "stop_after_p6a",
        "errors": [],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("protocol"),
        lambda value: value.update(extra="forbidden"),
        lambda value: value.update(schema_version=True),
        lambda value: value.update(status="almost"),
        lambda value: value["protocol"].update(horizons=[2, 4, 5]),
        lambda value: value["gate_results"].pop("G6A-3"),
        lambda value: value["settings"].update(metric=float("nan")),
        lambda value: value["provenance"]["config"].update(
            ref="/home/user/config.yaml"
        ),
        lambda value: value["provenance"]["config"].update(ref="repo:artifacts/P5/x"),
    ],
)
def test_root_artifact_fails_closed_on_invalid_contract(mutation):
    artifact = _artifact()
    mutation(artifact)

    with pytest.raises(ValueError):
        validate_root_artifact(artifact)


def test_artifact_json_is_deterministic_and_strict():
    artifact = _artifact()

    first = artifact_json_text(artifact)
    second = artifact_json_text(copy.deepcopy(artifact))

    assert first == second
    assert json.loads(first) == artifact
    assert first.endswith("\n")


def test_report_has_exact_sections_and_one_machine_decision():
    artifact = _artifact()

    report = render_go_nogo_report(artifact)

    assert all(report.count(f"## {section}") == 1 for section in P6A_REPORT_SECTIONS)
    assert report.count("Decision: P6A_GO") == 1
    assert report.count("Decision:") == 1
    assert report.rstrip().endswith("Exact next action: stop_after_p6a")


def test_report_uses_no_go_when_any_gate_fails():
    artifact = _artifact()
    artifact["gate_results"]["G6A-2"]["passed"] = False

    report = render_go_nogo_report(artifact)

    assert "Decision: P6A_NO_GO" in report
    assert "Decision: P6A_GO" not in report


def test_csv_renderer_has_stable_columns_and_rejects_schema_drift():
    rows = [
        {"method_id": "b1", "horizon": 2, "value": 0.5},
        {"method_id": "b4", "horizon": 2, "value": None},
    ]

    text = render_csv(rows, columns=("method_id", "horizon", "value"))

    assert text == "method_id,horizon,value\nb1,2,0.5\nb4,2,\n"
    with pytest.raises(ValueError):
        render_csv([{"method_id": "b1", "horizon": 2}], columns=("method_id",))


def test_publish_is_atomic_scoped_and_refuses_overwrite(tmp_path: Path):
    root = tmp_path / "artifacts" / "P6A"
    files = {
        "p6a_eval.json": artifact_json_text(_artifact()),
        "P6A_GO_NOGO_REPORT.md": render_go_nogo_report(_artifact()),
    }

    published = publish_artifacts(root, files)

    assert [path.name for path in published] == [
        "P6A_GO_NOGO_REPORT.md",
        "p6a_eval.json",
    ]
    with pytest.raises(FileExistsError):
        publish_artifacts(root, files)
    with pytest.raises(ValueError):
        publish_artifacts(root, {"../P5/result.json": "forbidden"})
