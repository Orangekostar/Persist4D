import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts.analyze_3rscan_temporal_distribution import (
    build_audit,
    load_metadata,
    main,
    render_markdown,
    sha256_file,
)


def _fixture_metadata() -> list[dict[str, object]]:
    return [
        {"reference": "r0", "type": "train", "scans": [{"reference": "r1"}]},
        {
            "reference": "v0",
            "type": "validation",
            "scans": [
                {"reference": "v1"},
                {"reference": "v2"},
                {"reference": "v3"},
            ],
        },
        {"reference": "x0", "type": "test", "scans": [{"reference": "x1"}]},
    ]


def test_build_audit_counts_visits_and_normalizes_validation() -> None:
    audit = build_audit(_fixture_metadata(), source_path="fixture.json")

    assert audit["split_distribution"]["train"]["T>=2"] == 1
    assert audit["split_distribution"]["val"]["T>=4"] == 1
    assert audit["audited_scene_count"] == 2
    assert audit["excluded_splits"] == {"test": 1}
    assert audit["scenes"][1] == {
        "reference_id": "v0",
        "split": "val",
        "T": 4,
        "scan_ids": ["v0", "v1", "v2", "v3"],
    }


def test_build_audit_reports_exact_and_threshold_distributions() -> None:
    audit = build_audit(_fixture_metadata(), source_path="fixture.json")

    assert audit["source_scene_count"] == 3
    assert audit["global_distribution"] == {
        "T=1": 0,
        "T=2": 2,
        "T=3": 0,
        "T=4": 1,
        "T>=2": 3,
        "T>=3": 1,
        "T>=4": 1,
        "T>=5": 0,
        "T>=6": 0,
    }
    assert audit["audited_distribution"]["T=2"] == 1
    assert audit["audited_distribution"]["T=4"] == 1


def test_change_trajectory_counts_are_not_inferred_from_metadata() -> None:
    metadata = [
        {
            "reference": "r0",
            "type": "train",
            "scans": [
                {
                    "reference": "r1",
                    "rigid": [],
                    "nonrigid": [],
                    "removed": [],
                }
            ],
        }
    ]

    audit = build_audit(metadata, source_path="fixture.json")

    assert audit["change_trajectory_statistics"]["status"] == "not_computed"
    reason = audit["change_trajectory_statistics"]["reason"]
    assert "static object universe" in reason
    assert "timestamps" in reason


def test_build_audit_records_source_hash_and_metadata_order_semantics() -> None:
    audit = build_audit(
        _fixture_metadata(),
        source_path="fixture.json",
        source_sha256="abc123",
    )

    assert audit["source"] == {"path": "fixture.json", "sha256": "abc123"}
    assert audit["scan_order_semantics"] == "metadata_order_only_no_timestamps"


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "metadata must be a JSON array"),
        ([{"reference": "r0", "type": "train"}], "missing required field 'scans'"),
        (
            [{"reference": "r0", "type": "train", "scans": [{}]}],
            "missing required field 'reference'",
        ),
    ],
)
def test_build_audit_rejects_invalid_metadata(
    metadata: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_audit(metadata, source_path="fixture.json")


def test_load_metadata_and_sha256_file(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    payload = json.dumps(_fixture_metadata(), separators=(",", ":"))
    metadata_path.write_text(payload, encoding="utf-8")

    assert load_metadata(metadata_path) == _fixture_metadata()
    assert sha256_file(metadata_path) == hashlib.sha256(payload.encode()).hexdigest()


def test_load_metadata_rejects_non_array_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="metadata must be a JSON array"):
        load_metadata(metadata_path)


def test_render_markdown_is_deterministic_and_states_scope() -> None:
    audit = build_audit(
        _fixture_metadata(), source_path="fixture.json", source_sha256="abc123"
    )

    first = render_markdown(audit)

    assert first == render_markdown(audit)
    assert "metadata order only" in first
    assert "test | 1" in first
    assert "not computed" in first
    assert "abc123" in first


def test_main_writes_portable_source_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata_path = tmp_path / "3RScan.json"
    json_output = tmp_path / "stats.json"
    markdown_output = tmp_path / "stats.md"
    metadata_path.write_text(json.dumps(_fixture_metadata()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_3rscan_temporal_distribution.py",
            "--metadata",
            str(metadata_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
    )

    main()

    json_text = json_output.read_text(encoding="utf-8")
    markdown_text = markdown_output.read_text(encoding="utf-8")
    assert json.loads(json_text)["source"]["path"] == "external:3RScan/3RScan.json"
    for output in (json_text, markdown_text):
        assert "/home/" not in output
        assert "/Users/" not in output
