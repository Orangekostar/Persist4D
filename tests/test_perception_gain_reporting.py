from __future__ import annotations

import json
from pathlib import Path

from scripts.perception_gain_reporting import generate_delivery


def test_reporting_generates_required_tables_sections_and_nonrecursive_manifest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    artifacts = project / "artifacts/perception_gain_v1"
    artifacts.mkdir(parents=True)
    (artifacts / "RUN_CONFIG.json").write_text(
        json.dumps(
            {
                "resolved_config": {
                    "identity": {
                        "parent_commit": "6" * 40,
                        "branch": "research/persist4d-perception-gain-v1",
                        "r1_checkpoint_sha256": "a" * 64,
                        "concerto_sha256": "b" * 64,
                    },
                    "budget": {
                        "total_gpu_hours": 192,
                        "foundation_and_evaluation_gpu_hours": 32,
                        "perception_training_gpu_hours": 136,
                        "refinement_gpu_hours": 16,
                        "profiling_gpu_hours": 4,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (artifacts / "DATA_ROLES.json").write_text(
        json.dumps(
            {
                "roles": {
                    "TRAIN": ["train"],
                    "CAL": ["cal"],
                    "SEL": ["sel"],
                    "PB": ["pb"],
                    "LOCAL-T2": ["local"],
                    "ADDITIONAL": ["additional"],
                }
            }
        ),
        encoding="utf-8",
    )
    state = {
        "completed_stages": ["bootstrap", "bind", "foundation", "scorer"],
        "budget_used": {
            "foundation_and_evaluation_gpu_hours": 1.0,
            "perception_training_gpu_hours": 2.0,
            "refinement_gpu_hours": 0.0,
            "profiling_gpu_hours": 0.0,
        },
        "failures": [
            {
                "stage": "perception_pilot",
                "reason": "external NFS unavailable",
            }
        ],
    }
    (artifacts / "RUN_STATE.json").write_text(json.dumps(state), encoding="utf-8")
    (artifacts / "CODE_BINDINGS.md").write_text("bindings\n", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()

    result = generate_delivery(
        project_root=project,
        artifact_root=artifacts,
        external_root=external,
        experiment_commit="c" * 40,
        publication_phase="READY",
    )

    report = (artifacts / "FINAL_REPORT.md").read_text(encoding="utf-8")
    for table in range(1, 7):
        assert f"## 表 {table}." in report
    handoff = (artifacts / "HANDOFF.md").read_text(encoding="utf-8")
    for section in range(1, 15):
        assert f"## {section}." in handoff
    assert str(tmp_path) not in report + handoff
    manifest = json.loads((artifacts / "ARTIFACT_MANIFEST.json").read_text())
    paths = {item["path"] for item in manifest["files"]}
    assert "artifacts/perception_gain_v1/ARTIFACT_MANIFEST.json" not in paths
    assert "artifacts/perception_gain_v1/RUN_STATE.json" not in paths
    assert result["status"] == "PASS"
