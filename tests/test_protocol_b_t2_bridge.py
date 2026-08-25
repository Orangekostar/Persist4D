from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.build_protocol_b_t2_bridge import (
    BridgeBuildError,
    build_protocol_b_t2_bridge,
    derive_exact_t2_record,
    extract_first_transition_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_MANIFEST = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"


def test_bridge_script_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/build_protocol_b_t2_bridge.py", "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _source_record() -> dict[str, object]:
    return {
        "added": [[1], [2], [3], [4]],
        "ambiguities": [[10, 11]],
        "filepath": "source-t5.txt",
        "nonrigid": [[5], [6], [7], [8]],
        "removed": [[9], [10], [11], [12]],
        "rigid": [[13], [14], [15], [16]],
        "scene": 69,
        "sub_scenes": [0, 2, 4, 3, 1],
        "type": "validation",
    }


def test_first_transition_target_uses_only_first_two_scan_rows_and_column_zero() -> None:
    target = np.arange(60, dtype=np.int64).reshape(15, 4)

    extracted = extract_first_transition_target(target, (4, 3, 2, 1, 5))

    assert extracted.tolist() == target[:7, 0].tolist()
    target[:7, 1:] = -999
    assert extract_first_transition_target(target, (4, 3, 2, 1, 5)).tolist() == (
        extracted.tolist()
    )


def test_first_transition_target_rejects_incompatible_t5_layout() -> None:
    with pytest.raises(BridgeBuildError, match="four transition columns"):
        extract_first_transition_target(np.zeros((10, 3)), (2, 2, 2, 2, 2))
    with pytest.raises(BridgeBuildError, match="row count"):
        extract_first_transition_target(np.zeros((9, 4)), (2, 2, 2, 2, 2))


def test_exact_t2_record_preserves_only_canonical_first_transition() -> None:
    record = derive_exact_t2_record(
        master_sequence_id=(
            "scene0069_00-scene0069_02-scene0069_04-"
            "scene0069_03-scene0069_01"
        ),
        canonical_scan_ids=(
            "scene0069_00",
            "scene0069_02",
            "scene0069_04",
            "scene0069_03",
            "scene0069_01",
        ),
        source_record=_source_record(),
        output_change_path="artifacts/new-change.txt",
    )

    assert record == {
        "added": [[1]],
        "ambiguities": [[10, 11]],
        "filepath": "artifacts/new-change.txt",
        "nonrigid": [[5]],
        "removed": [[9]],
        "rigid": [[13]],
        "scene": 69,
        "sub_scenes": [0, 2],
        "type": "validation",
    }


def test_real_bridge_passes_43_prefix_and_14_overlap_gate(tmp_path: Path) -> None:
    output = tmp_path / "protocol_bridge"

    result = build_protocol_b_t2_bridge(
        protocol_manifest_path=PROTOCOL_MANIFEST,
        output_root=output,
        source_commit="a040c11e17d219c383ec3cf0199377efbe791e96",
    )

    assert result["gate_pb0"] == {
        "status": "PASS",
        "canonical_prefix_count": 43,
        "overlap_count": 14,
        "overlap_parity_count": 14,
        "pair_substitution_count": 0,
        "reverse_pair_substitution_count": 0,
        "future_stage_leakage_count": 0,
        "validation_supervised_count": 43,
    }
    database = yaml.safe_load(
        (output / "sequence_database_protocol_b_exact_t2.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert len(database) == 43
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    expected = [
        master["orders"]["canonical"]["prefixes"]["2"]["sequence_id"]
        for master in protocol["masters"]
    ]
    assert list(database) == expected
    assert b"\r\n" not in (output / "bridge_inventory.csv").read_bytes()
    with (output / "bridge_inventory.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        inventory = list(csv.DictReader(handle))
    assert len(inventory) == 43
    assert sum(row["overlap_official_t2"] == "true" for row in inventory) == 14
    assert all(row["exact_ordered_pair"] == "true" for row in inventory)
    assert all(row["validation_supervised"] == "true" for row in inventory)
    assert all(row["future_stage_leakage"] == "false" for row in inventory)
    overlaps = [row for row in inventory if row["overlap_official_t2"] == "true"]
    for field in (
        "parity_scan_ids",
        "parity_point_counts",
        "parity_instance_gt",
        "parity_semantic_labels",
        "parity_temporal_stages",
        "parity_change_labels",
    ):
        assert all(row[field] == "true" for row in overlaps)
    assert len(list((output / "bridge_change_gt").glob("*.txt"))) == 43
    manifest = json.loads((output / "bridge_manifest.json").read_text())
    assert manifest["gate_pb0"] == result["gate_pb0"]
    assert manifest["construction"]["change_target"] == (
        "T5 rows for canonical scans 1-2, first transition column only"
    )
