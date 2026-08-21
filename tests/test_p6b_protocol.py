from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from models.persistent_memory_p6b import P6BMemoryConfig
from scripts.p6b_protocol import (
    P6BProtocolError,
    build_split_manifest,
    canonical_config_id,
    canonical_config_json,
    expand_stage_configs,
    joint_neighbor_configs,
    load_p6b_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
P6A_PROTOCOL = PROJECT_ROOT / "artifacts/P6A/protocol_b_manifest.json"
P6B_CONFIG = PROJECT_ROOT / "conf/p6b/default.yaml"

EXPECTED_TUNING_REFERENCES = (
    "5630cfcf-12bf-2860-8784-83d28a611a83",
    "10b17940-3938-2467-8a7a-958300ba83d3",
    "280d8ebb-6cc6-2788-9153-98959a2da801",
    "137a8158-1db5-2cc0-8003-31c12610471e",
)
EXPECTED_HELDOUT_REFERENCES = (
    "ddc73797-765b-241a-9e2c-097c5989baf6",
    "8eabc45f-5af7-2f32-8528-640861d2a135",
)
EXPECTED_ORDERS = ("canonical", "reverse", "sha256_seed45")


def _protocol() -> dict[str, object]:
    return json.loads(P6A_PROTOCOL.read_text(encoding="utf-8"))


def test_real_protocol_is_split_by_reference_cluster_without_leakage() -> None:
    split = build_split_manifest(_protocol(), seed=45)

    assert split.schema_version == 1
    assert split.seed == 45
    assert split.tuning_reference_scene_ids == EXPECTED_TUNING_REFERENCES
    assert split.heldout_reference_scene_ids == EXPECTED_HELDOUT_REFERENCES
    assert len(split.tuning_master_sequence_ids) == 32
    assert len(split.heldout_master_sequence_ids) == 11
    assert len(split.assignments) == 43
    assert set(split.tuning_master_sequence_ids).isdisjoint(
        split.heldout_master_sequence_ids
    )
    assert all(assignment.order_ids == EXPECTED_ORDERS for assignment in split.assignments)
    assert all(
        assignment.partition
        == (
            "tuning"
            if assignment.reference_scene_id in EXPECTED_TUNING_REFERENCES
            else "heldout"
        )
        for assignment in split.assignments
    )
    assert split.sha256 == (
        "80157a4f25d222d7a07757acbaa70e9a68b5d2e546ee4903bf755fda928689d6"
    )
    assert build_split_manifest(_protocol(), seed=45) == split


def test_split_manifest_rejects_duplicate_or_incomplete_protocol() -> None:
    duplicate = _protocol()
    duplicate["masters"][-1] = dict(duplicate["masters"][0])
    with pytest.raises(P6BProtocolError, match="duplicate"):
        build_split_manifest(duplicate, seed=45)

    incomplete = _protocol()
    incomplete["masters"] = incomplete["masters"][:-1]
    with pytest.raises(P6BProtocolError, match="43"):
        build_split_manifest(incomplete, seed=45)


def test_default_config_locks_search_space_and_selection_contract() -> None:
    config = load_p6b_config(P6B_CONFIG)
    search = config.search

    assert config.schema_version == 2
    assert config.seed == 45
    assert config.base == P6BMemoryConfig()
    assert search.assignment_modes == (
        "legacy_post_threshold",
        "threshold_aware",
    )
    assert search.active_thresholds == (0.45, 0.50, 0.55)
    assert search.reactivation_thresholds == (0.75, 0.85, 0.95, 1.05)
    assert search.reactivation_margins == (0.0, 0.05, 0.10, 0.20)
    assert search.class_modes == ("full", "foreground_normalized")
    assert search.class_weights == (0.15, 0.25, 0.35)
    assert search.consolidation_confidences == (0.80, 0.90, 0.97)
    assert search.consolidation_margins == (0.05, 0.10, 0.20)
    assert search.birth_confidences == (0.50, 0.75, 0.90, 0.97)
    assert search.birth_minimum_mask_supports == (1, 128, 512, 1024)
    assert search.birth_max_entropies == (None, 0.75, 0.50, 0.25)
    assert config.eligibility.minimum_reactivation_accuracy == 0.70
    assert config.eligibility.maximum_reactivation_recall_drop == 0.05
    assert config.eligibility.minimum_valid_observation_ratio == 0.90
    assert config.eligibility.maximum_t2_task_drop == 0.02
    assert config.ranking == (
        "paired_cluster_mean_t4_t5_identity_switch_rate",
        "paired_cluster_mean_t3_t5_wrong_reactivation_rate",
        "paired_cluster_mean_t2_t5_false_birth_rate",
        "negative_reactivation_recall",
        "negative_mean_t4_t5_task_score",
        "canonical_config_json",
    )


def test_stage_expansion_is_exact_unique_and_deterministic() -> None:
    config = load_p6b_config(P6B_CONFIG)
    expected_counts = {
        "assignment": 2,
        "reactivation": 48,
        "class_compatibility": 6,
        "consolidation": 10,
        "birth_gate": 64,
    }

    for stage, expected_count in expected_counts.items():
        first = expand_stage_configs(config.base, config.search, stage=stage)
        second = expand_stage_configs(config.base, config.search, stage=stage)
        assert first == second
        assert len(first) == expected_count
        assert len({canonical_config_id(item) for item in first}) == expected_count

    consolidation = expand_stage_configs(
        config.base, config.search, stage="consolidation"
    )
    assert sum(
        item.consolidation_confidence is None
        and item.consolidation_margin is None
        for item in consolidation
    ) == 1


def test_canonical_config_serialization_is_portable_and_ordered() -> None:
    config = P6BMemoryConfig()
    serialized = canonical_config_json(config)

    assert serialized == canonical_config_json(config)
    assert serialized.startswith('{"active_threshold":0.5,')
    assert " " not in serialized
    assert "/home/" not in serialized
    assert canonical_config_id(config) == "p6b-987d4942f8582eef"
    assert canonical_config_id(config) != canonical_config_id(
        replace(config, active_threshold=0.45)
    )


def test_joint_neighbors_change_at_most_one_registered_field() -> None:
    config = load_p6b_config(P6B_CONFIG)
    neighbors = joint_neighbor_configs(config.base, config.search)
    base_mapping = json.loads(canonical_config_json(config.base))

    assert neighbors[0] == config.base
    assert len({canonical_config_id(item) for item in neighbors}) == len(neighbors)
    for candidate in neighbors[1:]:
        candidate_mapping = json.loads(canonical_config_json(candidate))
        changed = {
            key
            for key, value in base_mapping.items()
            if candidate_mapping[key] != value
        }
        assert len(changed) == 1


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_config_loader_rejects_missing_and_extra_fields(
    tmp_path: Path, mutation: str
) -> None:
    payload = yaml.safe_load(P6B_CONFIG.read_text(encoding="utf-8"))
    if mutation == "missing":
        del payload["selection"]["eligibility"]["maximum_t2_task_drop"]
    else:
        payload["search"]["unexpected"] = [1]
    path = tmp_path / "p6b.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(P6BProtocolError, match="fields"):
        load_p6b_config(path)
