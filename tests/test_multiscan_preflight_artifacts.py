from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from scripts.audit_multiscan_dataset import build_release_blocked_artifacts

OFFICIAL_REPOSITORY = Path("/mnt/shared/ww/persist4d-multiscan/official-code")
OFFICIAL_SEMANTIC_MAP = (
    OFFICIAL_REPOSITORY / "dataset/benchmark/object_semantic_label_map.csv"
)
REQUIRED_BASE = (
    "repro_bindings.json",
    "reproducibility_binding.json",
    "multiscan_inventory.json",
    "longitudinal_subset_manifest.json",
)


def _inventory() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "scan_count": 3,
        "scene_count": 1,
        "threshold_scene_counts": {"3": 1, "4": 0, "5": 0},
        "selected_scene_count": 1,
        "selected_scan_count": 3,
        "selected_scene_list_sha256": "a" * 64,
        "scenes": [
            {
                "scene_id": "scene_00069",
                "scan_ids": [
                    "scene_00069_00",
                    "scene_00069_01",
                    "scene_00069_02",
                ],
                "number_of_scans": 3,
                "official_split": "train",
            }
        ],
    }


def test_blocked_release_builds_complete_fail_closed_evidence(tmp_path: Path) -> None:
    for name in REQUIRED_BASE:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    paths = build_release_blocked_artifacts(
        output_directory=tmp_path,
        inventory=_inventory(),
        semantic_map_path=OFFICIAL_SEMANTIC_MAP,
        official_repository=OFFICIAL_REPOSITORY,
    )

    expected_new = {
        "release_access_audit.json",
        "MULTISCAN_DATASET_AUDIT.md",
        "MULTISCAN_IDENTITY_AUDIT.md",
        "chronology_audit.json",
        "gap_opportunities.json",
        "multiscan_to_rescene_label_map.json",
        "MULTISCAN_ALIGNMENT_AUDIT.md",
        "frozen_protocol.json",
        "observation_coverage_smoke.json",
        "MULTISCAN_PREFLIGHT_REPORT.md",
        "evidence_manifest.json",
    }
    assert set(paths) == expected_new

    gaps = json.loads(paths["gap_opportunities.json"].read_text())
    assert gaps["status"] == "not_run_release_access_blocked"
    assert gaps["gap_event_count"] is None
    assert gaps["gap_scene_count"] is None
    assert gaps["opportunities"] == []

    chronology = json.loads(paths["chronology_audit.json"].read_text())
    assert chronology["status"] == "DATASET_ORDER_ONLY"
    labels = json.loads(paths["multiscan_to_rescene_label_map.json"].read_text())
    assert labels["status_counts"]["exact"] == 11

    report = paths["MULTISCAN_PREFLIGHT_REPORT.md"].read_text()
    assert re.findall(r"^## (.+)$", report, flags=re.MULTILINE) == [
        "1. Dataset provenance",
        "2. Longitudinal inventory",
        "3. Stable identity evidence",
        "4. Gap opportunities",
        "5. Chronology",
        "6. Alignment",
        "7. Semantic compatibility",
        "8. GT leakage audit",
        "9. Frozen ReScene smoke coverage",
        "10. Final decision",
    ]
    assert report.rstrip().endswith("`MULTISCAN_PROTOCOL_FAIL`")

    manifest = json.loads(paths["evidence_manifest.json"].read_text())
    assert manifest["decision"] == "MULTISCAN_PROTOCOL_FAIL"
    assert (
        manifest["implementation"]["commit"]
        == subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    )
    assert len(manifest["implementation"]["files"]) == 12
    for row in manifest["implementation"]["files"]:
        content = (
            Path(__file__).resolve().parents[1] / row["relative_path"]
        ).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
    assert {row["relative_path"] for row in manifest["files"]} == (
        set(REQUIRED_BASE) | (expected_new - {"evidence_manifest.json"})
    )
    for row in manifest["files"]:
        content = (tmp_path / row["relative_path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        assert len(content) == row["bytes"]

    second = build_release_blocked_artifacts(
        output_directory=tmp_path,
        inventory=_inventory(),
        semantic_map_path=OFFICIAL_SEMANTIC_MAP,
        official_repository=OFFICIAL_REPOSITORY,
    )
    assert {name: path.read_bytes() for name, path in paths.items()} == {
        name: path.read_bytes() for name, path in second.items()
    }
