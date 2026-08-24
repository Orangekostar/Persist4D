from __future__ import annotations

import copy

import pytest

from scripts.verify_final_evidence import (
    EvidenceVerificationError,
    validate_full_history_contract,
)


def _contract() -> tuple[dict[str, object], dict[str, object]]:
    digest = "a" * 64
    manifest: dict[str, object] = {
        "status": "pass",
        "scene_count": 1,
        "entry_count": 2,
        "provenance": {
            "history_strategy": "full_history",
            "checkpoint_sha256": "b" * 64,
            "dataset_content_sha256": "c" * 64,
            "evaluator_sha256": "d" * 64,
        },
        "entries": [
            {
                "filename": "scene_a_0.pt",
                "sha256": "e" * 64,
                "key": {
                    "history_strategy": "full_history",
                    "stage_index": 0,
                    "target_capture_id": "scene_a_0",
                    "local_capture_ids": ["scene_a_0"],
                },
            },
            {
                "filename": "scene_a_1.pt",
                "sha256": "f" * 64,
                "key": {
                    "history_strategy": "full_history",
                    "stage_index": 1,
                    "target_capture_id": "scene_a_1",
                    "local_capture_ids": ["scene_a_0", "scene_a_1"],
                },
            },
        ],
    }
    raw: dict[str, object] = {
        "status": "pass",
        "provenance": {
            "cache_manifest_sha256": digest,
            "checkpoint_sha256": "b" * 64,
            "dataset_content_sha256": "c" * 64,
            "evaluator_sha256": "d" * 64,
        },
        "external_gate": {"classification": "EXTERNAL_INCONCLUSIVE"},
    }
    return manifest, raw


def test_full_history_contract_accepts_expanding_history() -> None:
    manifest, raw = _contract()

    validate_full_history_contract(manifest, raw, "a" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest, raw: manifest["entries"][1]["key"].update(
                {"local_capture_ids": ["scene_a_1"]}
            ),
            "expanding capture history",
        ),
        (
            lambda manifest, raw: raw["provenance"].update(
                {"cache_manifest_sha256": "0" * 64}
            ),
            "cache manifest binding",
        ),
        (
            lambda manifest, raw: raw["provenance"].update(
                {"checkpoint_sha256": "0" * 64}
            ),
            "runtime provenance",
        ),
        (
            lambda manifest, raw: raw["external_gate"].update(
                {"classification": "EXTERNAL_SUPPORT"}
            ),
            "classification changed",
        ),
    ],
)
def test_full_history_contract_fails_closed(mutation, message: str) -> None:
    manifest, raw = _contract()
    manifest = copy.deepcopy(manifest)
    raw = copy.deepcopy(raw)
    mutation(manifest, raw)

    with pytest.raises(EvidenceVerificationError, match=message):
        validate_full_history_contract(manifest, raw, "a" * 64)
