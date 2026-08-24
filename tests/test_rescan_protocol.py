from __future__ import annotations

import copy

import pytest

from scripts.rescan_protocol import (
    RescanProtocolError,
    build_rescan_protocol,
    validate_rescan_protocol,
)


def _dataset_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "pass",
        "dataset_content_sha256": "a" * 64,
        "chronology": {"status": "official_index_order"},
        "summary": {
            "scene_count": 2,
            "capture_count": 6,
            "gap_opportunity_count": 1,
            "encountered_class_ids": [3, 5, 13],
        },
        "scenes": [
            {
                "scene_id": "scene_a",
                "capture_ids": ["scene_a_0", "scene_a_1", "scene_a_2"],
                "stable_identity_ids": [4, 5],
                "stable_object_identity_ids": [4, 5],
                "semantic_inconsistent_identity_ids": [],
                "gap_opportunities": [
                    {
                        "identity": 4,
                        "left_capture_id": "scene_a_0",
                        "right_capture_id": "scene_a_2",
                        "absent_capture_ids": ["scene_a_1"],
                    }
                ],
            },
            {
                "scene_id": "scene_b",
                "capture_ids": ["scene_b_0", "scene_b_1", "scene_b_2"],
                "stable_identity_ids": [8],
                "stable_object_identity_ids": [8],
                "semantic_inconsistent_identity_ids": [],
                "gap_opportunities": [],
            },
        ],
    }


def _label_map() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mappings": [
            {
                "source_class_id": 3,
                "source_class_name": "cabinet",
                "target_class_id": 0,
                "target_class_name": "cabinet",
                "mapping_evidence": "exact",
                "status": "exact",
            },
            {
                "source_class_id": 5,
                "source_class_name": "chair",
                "target_class_id": 2,
                "target_class_name": "chair",
                "mapping_evidence": "exact",
                "status": "exact",
            },
            {
                "source_class_id": 13,
                "source_class_name": "blinds",
                "target_class_id": None,
                "target_class_name": None,
                "mapping_evidence": "unsupported",
                "status": "unsupported",
            },
        ],
    }


def test_protocol_uses_official_order_and_exact_local_causal_inputs() -> None:
    protocol = build_rescan_protocol(
        _dataset_manifest(),
        _label_map(),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
    )

    assert protocol["status"] == "pass"
    assert protocol["population"] == {
        "capture_count": 6,
        "gap_opportunity_count": 1,
        "independent_scene_cluster_count": 2,
        "temporal_sequence_count": 2,
        "transition_count": 4,
    }
    assert protocol["level_a"]["eligible_source_class_ids"] == [3, 5]
    assert protocol["level_a"]["excluded_source_class_ids"] == [13]
    assert protocol["level_b"]["stable_identity_count"] == 3
    assert protocol["order_policy"] == {
        "artificial_permutations": False,
        "chronology": "official_index_order",
    }
    first = protocol["scenes"][0]
    assert [stage["local_input_capture_ids"] for stage in first["stages"]] == [
        ["scene_a_0"],
        ["scene_a_0", "scene_a_1"],
        ["scene_a_1", "scene_a_2"],
    ]
    assert [stage["global_capture_indices"] for stage in first["stages"]] == [
        [0],
        [0, 1],
        [1, 2],
    ]
    assert protocol["baselines"] == [
        "Pairwise Feature Association",
        "Pairwise Feature-Class Association",
        "EMA Temporal Association",
        "Persist4D",
    ]
    assert len(protocol["content_sha256"]) == 64


def test_protocol_validation_rejects_noncausal_or_expanded_local_context() -> None:
    protocol = build_rescan_protocol(
        _dataset_manifest(),
        _label_map(),
        checkpoint_sha256="b" * 64,
        config_sha256="c" * 64,
    )
    invalid = copy.deepcopy(protocol)
    invalid["scenes"][0]["stages"][2]["local_input_capture_ids"] = [
        "scene_a_0",
        "scene_a_1",
        "scene_a_2",
    ]

    with pytest.raises(RescanProtocolError, match="local input"):
        validate_rescan_protocol(invalid, dataset_manifest=_dataset_manifest())


def test_protocol_refuses_nonofficial_chronology() -> None:
    dataset = _dataset_manifest()
    dataset["chronology"] = {"status": "filename_guess"}

    with pytest.raises(RescanProtocolError, match="chronology"):
        build_rescan_protocol(
            dataset,
            _label_map(),
            checkpoint_sha256="b" * 64,
            config_sha256="c" * 64,
        )
