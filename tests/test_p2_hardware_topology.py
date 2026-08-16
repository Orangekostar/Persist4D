import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import audit_p2_hardware_topology as audit
from utils.p2_preflight import P2_PREFLIGHT_SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_p2_hardware_topology.py"
ARTIFACT = REPO_ROOT / "artifacts" / "P2" / "hardware_topology_profile.csv"
SHARED_GATE_FIELDS = (
    "config_contract",
    "source_tree_contract",
    "runtime_source_contract",
    "runtime_environment_contract",
    "official_split_identity",
    "input_manifest",
    "authorization",
)

INVENTORY = """\
0, NVIDIA A40, 46068
1, NVIDIA A40, 46068
2, NVIDIA A40, 46068
"""

TOPOLOGY = """\
        GPU0    GPU1    GPU2    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      SYS     SYS     0-15,32-47      0               N/A
GPU1    SYS      X      NODE    16-31,48-63     1               N/A
GPU2    SYS     NODE     X      16-31,48-63     1               N/A
"""

EXPECTED_COLUMNS = [
    "schema_version",
    "stage",
    "candidate_id",
    "gpu_count",
    "gpu_indices",
    "detected_gpu_count",
    "gpu_model",
    "memory_per_gpu_mib",
    "interconnect",
    "cpu_affinity",
    "numa_affinity",
    "same_numa",
    "scannet_preflight_ref",
    "scannet_preflight_status",
    "formal_training_authorized",
    "model_benchmark_status",
    "training_throughput_samples_per_s",
    "optimizer_steps_per_s",
    "peak_vram_mib",
    "communication_diagnostic_status",
    "communication_bandwidth_gbps",
    "topology_selection_status",
]


def _valid_preflight() -> dict[str, Any]:
    split_counts = {"train": 1201, "validation": 312, "test": 100}
    split_metadata = {
        split: {
            "expected": count,
            "observed": count,
            "unique": count,
            "status": "pass",
        }
        for split, count in split_counts.items()
    }
    raw_by_split = {
        split: {
            "expected_scene_count": count,
            "complete_scene_count": count,
            "missing_scene_count": 0,
            "missing_asset_count": 0,
            "status": "pass",
        }
        for split, count in split_counts.items()
    }
    processed_by_split = {
        split: {
            "expected_scene_count": count,
            "database_scene_count": count,
            "npy_scene_count": count,
            "status": "pass",
        }
        for split, count in split_counts.items()
    }
    return {
        "schema_version": P2_PREFLIGHT_SCHEMA_VERSION,
        "status": "pass",
        "formal_p2_training_authorized": True,
        "official_source_commit": "fb2fe42eb8f1e926567c48eea9acb874e608ee10",
        "expected_split_counts": split_counts,
        "split_metadata_status": "pass",
        "split_metadata": split_metadata,
        "raw_assets": {
            "expected_scene_count": 1613,
            "complete_scene_count": 1613,
            "missing_asset_count": 0,
            "status": "pass",
            "by_split": raw_by_split,
        },
        "processed_assets": {
            "expected_scene_count": 1613,
            "database_scene_count": 1613,
            "npy_scene_count": 1613,
            "status": "pass",
            "by_split": processed_by_split,
        },
        "class_taxonomy": {
            "status": "pass",
            "class_count": 18,
            "valid_class_ids": [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                14,
                16,
                24,
                28,
                33,
                34,
                36,
                39,
            ],
        },
        "mix_instantiation": {
            "attempted": True,
            "status": "pass",
            "implementation": "datasets.multi_dataset.MultiDataset",
            "dataset_names": ["rio", "scannet"],
            "dataset_sizes": [1174, 1199],
            "sampler_num_samples": 2112,
            "epoch_sample_multiple": 32,
            "weights": [1.0, 0.8],
            "temporal_windows": [2, 1],
            "sampler": "WeightedRandomSampler",
        },
        "errors": [],
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == EXPECTED_COLUMNS
        return list(reader)


def _run_audit(
    tmp_path: Path,
    preflight: dict[str, Any] | bytes | None,
    topology_text: str = TOPOLOGY,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    inventory = tmp_path / "inventory.csv"
    topology = tmp_path / "topology.txt"
    preflight_path = tmp_path / "private-user" / "scannet_preflight.json"
    output = tmp_path / "output" / "hardware_topology_profile.csv"
    inventory.write_text(INVENTORY, encoding="utf-8")
    topology.write_text(topology_text, encoding="utf-8")
    if preflight is not None:
        preflight_path.parent.mkdir(parents=True)
        if isinstance(preflight, bytes):
            preflight_path.write_bytes(preflight)
        else:
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--preflight",
            str(preflight_path),
            "--inventory-file",
            str(inventory),
            "--topology-file",
            str(topology),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def _assert_no_training_measurements(rows: list[dict[str, str]]) -> None:
    for row in rows:
        assert row["model_benchmark_status"] == "not_run"
        assert row["training_throughput_samples_per_s"] == "null"
        assert row["optimizer_steps_per_s"] == "null"
        assert row["peak_vram_mib"] == "null"
        assert row["communication_diagnostic_status"] == "not_run"
        assert row["communication_bandwidth_gbps"] == "null"


def _assert_private(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "/" + "home/" not in text
    assert "/" + "Users/" not in text
    assert "GPU-" not in text
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    assert not re.search(r"\b[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d\b", text)


def test_missing_preflight_blocks_selection_and_keeps_measurements_null(
    tmp_path: Path,
) -> None:
    result, output = _run_audit(tmp_path, preflight=None)

    assert result.returncode == 2, result.stderr
    rows = _read_rows(output)
    assert len(rows) == 6
    assert {row["scannet_preflight_status"] for row in rows} == {"missing"}
    assert {row["formal_training_authorized"] for row in rows} == {"false"}
    assert {row["topology_selection_status"] for row in rows} == {
        "blocked_missing_preflight"
    }
    assert {row["scannet_preflight_ref"] for row in rows} == {
        "external:scannet_preflight"
    }
    _assert_no_training_measurements(rows)
    _assert_private(output)


def test_blocked_scannet_emits_all_one_and_two_gpu_candidates(tmp_path: Path) -> None:
    result, output = _run_audit(
        tmp_path,
        preflight={
            "schema_version": 1,
            "status": "blocked_missing_scannet",
            "formal_p2_training_authorized": False,
        },
    )

    assert result.returncode == 2, result.stderr
    rows = _read_rows(output)
    assert [row["candidate_id"] for row in rows] == [
        "gpu0",
        "gpu1",
        "gpu2",
        "gpu0+gpu1",
        "gpu0+gpu2",
        "gpu1+gpu2",
    ]
    assert {row["detected_gpu_count"] for row in rows} == {"3"}
    assert {row["gpu_model"] for row in rows} == {"NVIDIA A40"}
    assert {row["memory_per_gpu_mib"] for row in rows} == {"46068"}
    assert {row["scannet_preflight_status"] for row in rows} == {
        "blocked_missing_scannet"
    }
    assert {row["topology_selection_status"] for row in rows} == {
        "blocked_by_scannet_preflight"
    }

    by_id = {row["candidate_id"]: row for row in rows}
    assert by_id["gpu1+gpu2"]["interconnect"] == "NODE"
    assert by_id["gpu1+gpu2"]["cpu_affinity"] == "16-31,48-63"
    assert by_id["gpu1+gpu2"]["numa_affinity"] == "1"
    assert by_id["gpu1+gpu2"]["same_numa"] == "true"
    assert by_id["gpu0+gpu1"]["interconnect"] == "SYS"
    assert by_id["gpu0+gpu1"]["same_numa"] == "false"
    _assert_no_training_measurements(rows)
    _assert_private(output)


def test_passed_preflight_only_marks_candidates_pending_formal_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_preflight()
    payload.update({field: {} for field in SHARED_GATE_FIELDS})
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(audit, "_shared_p2_authorization_gate", lambda _path: True)

    status, authorized, selection_status, return_code = audit._read_preflight(
        preflight_path
    )
    assert (status, authorized, selection_status, return_code) == (
        "pass",
        True,
        "pending_formal_benchmark",
        0,
    )
    rows = audit.build_rows(
        inventory=audit.parse_inventory(INVENTORY),
        topology=audit.parse_topology(TOPOLOGY, [0, 1, 2]),
        preflight_ref=audit._portable_preflight_ref(preflight_path),
        preflight_status=status,
        formal_training_authorized=authorized,
        topology_selection_status=selection_status,
    )
    assert {row["formal_training_authorized"] for row in rows} == {"true"}
    assert {row["topology_selection_status"] for row in rows} == {
        "pending_formal_benchmark"
    }
    _assert_no_training_measurements(rows)


@pytest.mark.parametrize("missing_field", SHARED_GATE_FIELDS)
def test_pass_preflight_requires_every_shared_authorization_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    payload = _valid_preflight()
    payload.update({field: {} for field in SHARED_GATE_FIELDS})
    payload.pop(missing_field)
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        audit,
        "_shared_p2_authorization_gate",
        lambda _path: True,
        raising=False,
    )

    status, authorized, selection_status, return_code = audit._read_preflight(
        preflight_path
    )

    assert (status, authorized, selection_status, return_code) == (
        "invalid_pass_contract",
        False,
        "blocked_invalid_preflight",
        2,
    )


def test_pass_preflight_is_blocked_when_shared_authorization_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_preflight()
    payload.update({field: {} for field in SHARED_GATE_FIELDS})
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(audit, "_compose_p2_config", lambda: object())

    def reject_authorization(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("shared authorization rejected")

    monkeypatch.setattr(
        audit,
        "require_p2_preflight_authorization",
        reject_authorization,
    )

    status, authorized, selection_status, return_code = audit._read_preflight(
        preflight_path
    )

    assert (status, authorized, selection_status, return_code) == (
        "invalid_pass_contract",
        False,
        "blocked_invalid_preflight",
        2,
    )


def test_pass_preflight_is_ready_only_when_shared_authorization_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_preflight()
    payload.update({field: {} for field in SHARED_GATE_FIELDS})
    preflight_path = tmp_path / "scannet_preflight.json"
    preflight_path.write_text(json.dumps(payload), encoding="utf-8")
    config = object()
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(audit, "_compose_p2_config", lambda: config)

    def allow_authorization(received_config: object, *, artifact_path: Path) -> Path:
        calls.append((received_config, artifact_path))
        return artifact_path

    monkeypatch.setattr(
        audit,
        "require_p2_preflight_authorization",
        allow_authorization,
    )

    status, authorized, selection_status, return_code = audit._read_preflight(
        preflight_path
    )

    assert (status, authorized, selection_status, return_code) == (
        "pass",
        True,
        "pending_formal_benchmark",
        0,
    )
    assert calls == [(config, preflight_path)]


def test_legacy_complete_preflight_is_rejected(tmp_path: Path) -> None:
    preflight = _valid_preflight()
    preflight["schema_version"] = P2_PREFLIGHT_SCHEMA_VERSION - 1

    result, output = _run_audit(tmp_path, preflight=preflight)

    assert result.returncode == 2, result.stderr
    rows = _read_rows(output)
    assert {row["scannet_preflight_status"] for row in rows} == {
        "invalid_pass_contract"
    }
    assert {row["formal_training_authorized"] for row in rows} == {"false"}


def test_incomplete_pass_preflight_is_rejected(tmp_path: Path) -> None:
    result, output = _run_audit(
        tmp_path,
        preflight={
            "schema_version": 999,
            "status": "pass",
            "formal_p2_training_authorized": True,
        },
    )

    assert result.returncode == 2, result.stderr
    rows = _read_rows(output)
    assert {row["scannet_preflight_status"] for row in rows} == {
        "invalid_pass_contract"
    }
    assert {row["formal_training_authorized"] for row in rows} == {"false"}
    assert {row["topology_selection_status"] for row in rows} == {
        "blocked_invalid_preflight"
    }


def test_unknown_numa_affinity_is_not_reported_as_same_numa(tmp_path: Path) -> None:
    unknown_numa_topology = TOPOLOGY.replace(
        "16-31,48-63     1               N/A",
        "16-31,48-63     N/A             N/A",
    )
    result, output = _run_audit(
        tmp_path,
        preflight={
            "schema_version": 1,
            "status": "blocked_missing_scannet",
            "formal_p2_training_authorized": False,
        },
        topology_text=unknown_numa_topology,
    )

    assert result.returncode == 2, result.stderr
    by_id = {row["candidate_id"]: row for row in _read_rows(output)}
    assert by_id["gpu1+gpu2"]["numa_affinity"] == "N/A"
    assert by_id["gpu1+gpu2"]["same_numa"] == "false"


def test_non_utf8_preflight_is_invalid_and_blocked(tmp_path: Path) -> None:
    result, output = _run_audit(tmp_path, preflight=b"\xff\xfe\x00")

    assert result.returncode == 2, result.stderr
    rows = _read_rows(output)
    assert {row["scannet_preflight_status"] for row in rows} == {"invalid"}
    assert {row["topology_selection_status"] for row in rows} == {
        "blocked_invalid_preflight"
    }


def test_repository_artifact_is_private_authorized_preflight_snapshot() -> None:
    rows = _read_rows(ARTIFACT)
    assert len(rows) == 6
    assert {row["detected_gpu_count"] for row in rows} == {"3"}
    assert {row["gpu_model"] for row in rows} == {"NVIDIA A40"}
    assert {row["scannet_preflight_ref"] for row in rows} == {
        "repo:artifacts/P2/scannet_preflight.json"
    }
    assert {row["scannet_preflight_status"] for row in rows} == {"pass"}
    assert {row["formal_training_authorized"] for row in rows} == {"true"}
    assert {row["topology_selection_status"] for row in rows} == {
        "pending_formal_benchmark"
    }
    _assert_no_training_measurements(rows)
    _assert_private(ARTIFACT)
