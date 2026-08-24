import csv
import hashlib
import json
from pathlib import Path

from scripts.p6a_analysis import classify_failure

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/final_evidence"
CAPACITY = EVIDENCE / "capacity"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_capacity_manifest_binds_every_published_result() -> None:
    manifest = json.loads(
        (CAPACITY / "capacity_evaluation_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["status"] == "pass"
    assert manifest["classification"] == "CAPACITY_100_OK"
    for filename, record in manifest["artifacts"].items():
        path = CAPACITY / filename
        assert path.stat().st_size == record["bytes"]
        assert _sha256(path) == record["sha256"]


def test_capacity_artifacts_close_coverage_and_decision() -> None:
    with (CAPACITY / "capacity_per_sequence.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        per_sequence = list(csv.DictReader(stream))
    with (CAPACITY / "capacity_aggregate.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        aggregate = list(csv.DictReader(stream))
    summary = json.loads(
        (EVIDENCE / "capacity_summary.json").read_text(encoding="utf-8")
    )
    gate = json.loads((CAPACITY / "capacity_gate.json").read_text(encoding="utf-8"))

    assert len(per_sequence) == 5 * 4 * 129
    assert len(aggregate) == 5 * 4
    assert sum(int(row["rejected_births"]) for row in per_sequence) == 0
    assert max(int(row["peak_occupied_slots"]) for row in per_sequence) == 30
    assert summary["classification"] == gate["classification"]
    assert summary["q1"]["saturated_sequence_count_at_t5"] == 0
    assert summary["q2"]["total_t5_rejected_births"] == 0
    assert "`CAPACITY_100_OK`" in (
        EVIDENCE / "CAPACITY_SENSITIVITY_REPORT.md"
    ).read_text(encoding="utf-8")


def test_legacy_f7_taxonomy_conflates_false_and_rejected_births() -> None:
    false_birth = {
        "association_result": "false_birth",
        "capacity_failure": False,
        "birth_rejected": False,
    }
    rejected_birth = {
        "association_result": "birth_rejected",
        "capacity_failure": True,
        "birth_rejected": True,
    }

    assert classify_failure(false_birth) == "F7"
    assert classify_failure(rejected_birth) == "F7"
