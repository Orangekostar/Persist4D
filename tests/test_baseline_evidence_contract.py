from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_baseline_evidence_contract import (
    BaselineEvidenceError,
    build_baseline_evidence_contract,
    write_baseline_evidence_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"frozen-checkpoint")
    p2_report = tmp_path / "P2.md"
    p2_report.write_text(
        "Reproduced t-mAP: 27.939\nG2 = RED - do not proceed\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"schema_version": 2}\n', encoding="utf-8")
    v2_manifest = tmp_path / "v2.json"
    v2_manifest.write_text(
        json.dumps(
            {
                "checkpoint_sha256": _sha256(checkpoint),
                "score_reducer": "mean",
                "status": "pass",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint_path": checkpoint,
        "expected_checkpoint_sha256": _sha256(checkpoint),
        "p2_report_path": p2_report,
        "expected_p2_report_sha256": _sha256(p2_report),
        "protocol_manifest_path": protocol,
        "v2_manifest_path": v2_manifest,
        "source_commit": "a" * 40,
        "official_repository_url": "https://github.com/GradientSpaces/rescene4d",
        "official_revision": "b" * 40,
        "official_readme_sha256": "c" * 64,
        "retrieved_at": "2026-08-25T00:00:00Z",
        "checkpoint_section_status": "Coming soon.",
    }


def test_contract_separates_paper_local_and_controlled_evidence(tmp_path: Path) -> None:
    contract = build_baseline_evidence_contract(**_inputs(tmp_path))

    rows = {row["evidence_id"]: row for row in contract["evidence_rows"]}
    assert rows["E0"] == {
        "evidence_id": "E0",
        "name": "ReScene4D-C (paper-reported)",
        "source": "ReScene4D paper",
        "t_mAP_percent": 34.8,
        "status": "external_reference",
        "locally_rerun": False,
        "checkpoint_availability": "not_publicly_exposed",
        "table_label": "ReScene4D-C (paper-reported)",
    }
    assert rows["E1"]["t_mAP_percent"] == 27.939
    assert rows["E1"]["status"] == "local_best_effort_reimplementation"
    assert rows["E1"]["g2_gate"] == "RED"
    assert rows["E2"]["status"] == "controlled_internal_baseline"
    assert rows["E2"]["t_mAP_percent"] is None
    assert contract["gate_b0"] == {
        "status": "PASS",
        "paper_and_local_values_separated": True,
        "checkpoint_statement_has_provenance": True,
        "table_labels_generated_from_contract": True,
    }
    assert all(
        "official 34.8 model" not in claim.lower()
        for claim in contract["claims"]["allowed"]
    )


def test_contract_writer_emits_matching_json_and_markdown(tmp_path: Path) -> None:
    contract = build_baseline_evidence_contract(**_inputs(tmp_path))
    output = tmp_path / "baseline"

    written = write_baseline_evidence_contract(contract, output)

    assert written == {
        "json": output / "baseline_evidence_contract.json",
        "markdown": output / "BASELINE_EVIDENCE_CONTRACT.md",
    }
    loaded = json.loads(written["json"].read_text(encoding="utf-8"))
    assert loaded == contract
    report = written["markdown"].read_text(encoding="utf-8")
    assert "ReScene4D-C (paper-reported)" in report
    assert "ReScene4D-C (our reimplementation)" in report
    assert "Gate B0: PASS" in report
    assert "Persist4D beats official ReScene4D." in report


def test_contract_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["expected_checkpoint_sha256"] = "0" * 64

    with pytest.raises(BaselineEvidenceError, match="checkpoint SHA256"):
        build_baseline_evidence_contract(**inputs)


def test_real_contract_records_generation_and_execution_provenance() -> None:
    contract_path = (
        PROJECT_ROOT
        / "artifacts/reviewer_closure_v3/baseline/baseline_evidence_contract.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    assert contract["generated_at"] == "2026-08-25T13:34:12Z"
    assert contract["execution"] == {"gpu_inference_performed": False}
    assert contract["configuration"] == {
        "config_hash": "not_applicable",
        "cache_hash": "not_applicable",
    }
    builder = contract["scripts"]["builder"]
    assert builder["reference"] == "repo:scripts/build_baseline_evidence_contract.py"
    assert builder["sha256"] == _sha256(
        PROJECT_ROOT / "scripts/build_baseline_evidence_contract.py"
    )
