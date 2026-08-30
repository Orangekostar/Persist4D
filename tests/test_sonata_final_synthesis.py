from __future__ import annotations

import hashlib

import pytest


def _summary_rows() -> list[dict[str, str]]:
    return [
        {
            "model": "our_sonata_reimplementation",
            "t_mAP_mean": "0.24035188059012094",
            "overall_mAP_mean": "0.31553595264752704",
        },
        {
            "model": "our_concerto_reimplementation",
            "t_mAP_mean": "0.2829008499781291",
            "overall_mAP_mean": "0.3697936435540517",
        },
    ]


def test_sq_red_forces_sr_red_without_protocol_b_values() -> None:
    from scripts import finalize_sonata_second_perception as synthesis

    result = synthesis.build_cross_backbone_synthesis(
        summary_rows=_summary_rows(),
        qualification_gate={"label": "SQ-RED", "authorizes_ss6": False},
    )

    assert result["sr_gate"] == "SR-RED"
    assert result["protocol_b_status"] == "gate_skipped"
    protocol_rows = [row for row in result["rows"] if row["stage"] == "SS7"]
    assert protocol_rows
    assert all(row["sonata_value"] == "" for row in protocol_rows)
    assert all(row["sonata_status"] == "gate_skipped" for row in protocol_rows)


def test_cross_backbone_synthesis_preserves_measured_local_values() -> None:
    from scripts import finalize_sonata_second_perception as synthesis

    result = synthesis.build_cross_backbone_synthesis(
        summary_rows=_summary_rows(),
        qualification_gate={"label": "SQ-RED", "authorizes_ss6": False},
    )

    local = {row["metric"]: row for row in result["rows"] if row["stage"] == "SS5"}
    assert float(local["t_mAP"]["sonata_value"]) == pytest.approx(0.24035188059012094)
    assert float(local["t_mAP"]["concerto_value"]) == pytest.approx(0.2829008499781291)
    assert float(local["overall_mAP"]["sonata_value"]) == pytest.approx(
        0.31553595264752704
    )


def test_cross_backbone_synthesis_rejects_unexpected_gate() -> None:
    from scripts import finalize_sonata_second_perception as synthesis

    with pytest.raises(ValueError, match="SQ-RED qualification evidence"):
        synthesis.build_cross_backbone_synthesis(
            summary_rows=_summary_rows(),
            qualification_gate={"label": "SQ-GREEN", "authorizes_ss6": True},
        )


def test_final_manifest_hashes_every_upstream_and_output() -> None:
    from scripts import finalize_sonata_second_perception as synthesis

    upstream = {"checkpoint/QUALIFICATION_MANIFEST.json": b"qualification\n"}
    outputs = {"cross_backbone_summary.csv": b"summary\n"}
    manifest = synthesis.build_final_manifest(
        source_commit="a" * 40,
        upstream_payloads=upstream,
        output_payloads=outputs,
    )

    assert manifest["sq_gate"] == "SQ-RED"
    assert manifest["sr_gate"] == "SR-RED"
    assert manifest["upstream_sha256"] == {
        "checkpoint/QUALIFICATION_MANIFEST.json": hashlib.sha256(
            upstream["checkpoint/QUALIFICATION_MANIFEST.json"]
        ).hexdigest()
    }
    assert manifest["output_sha256"] == {
        "cross_backbone_summary.csv": hashlib.sha256(
            outputs["cross_backbone_summary.csv"]
        ).hexdigest()
    }
    assert len(manifest["content_sha256"]) == 64
