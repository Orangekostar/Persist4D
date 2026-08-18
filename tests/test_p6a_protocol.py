from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.p6a_protocol import (
    ProtocolError,
    build_order_variants,
    build_protocol_b,
    build_protocol_b_manifest,
    derive_exact_prefixes,
    load_scan_indices,
    load_t5_masters,
    validate_protocol_b_manifest,
    write_protocol_b_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T5_DATABASE = PROJECT_ROOT / "data/processed/rio/sequence_database_sliding_5.yaml"
VALIDATION_DATABASE = PROJECT_ROOT / "data/processed/rio/validation_database.yaml"
RSCAN_METADATA = Path("/home/ww/3RScan.json")
EXPECTED_T5_NAME_SHA256 = (
    "01f323fae7f4f861fa9bc7125ac4aa6eea8f10376c13e5775717928e9061e990"
)


def _fixture_records() -> list[dict[str, object]]:
    return [
        {
            "scene": 69,
            "sub_scene": sub_scene,
            "split": "validation",
            "supervised": True,
        }
        for sub_scene in range(5)
    ]


def _fixture_database() -> dict[str, dict[str, object]]:
    scan_ids = [f"scene0069_{sub_scene:02d}" for sub_scene in range(5)]
    return {
        "-".join(scan_ids): {
            "scene": 69,
            "sub_scenes": list(range(5)),
            "type": "validation",
            "filepath": "change_gt/validation/example.txt",
        }
    }


def test_real_t5_validation_inventory_has_43_names_and_six_uuid_clusters() -> None:
    if not RSCAN_METADATA.is_file():
        pytest.skip("the external 3RScan metadata is unavailable")

    masters = load_t5_masters(
        T5_DATABASE,
        VALIDATION_DATABASE,
        metadata_path=RSCAN_METADATA,
    )

    assert len(masters) == 43
    names = [master.sequence_id for master in masters]
    assert (
        hashlib.sha256("\n".join(names).encode()).hexdigest() == EXPECTED_T5_NAME_SHA256
    )
    assert {master.reference_scene_id for master in masters} == {
        "10b17940-3938-2467-8a7a-958300ba83d3",
        "137a8158-1db5-2cc0-8003-31c12610471e",
        "280d8ebb-6cc6-2788-9153-98959a2da801",
        "5630cfcf-12bf-2860-8784-83d28a611a83",
        "8eabc45f-5af7-2f32-8528-640861d2a135",
        "ddc73797-765b-241a-9e2c-097c5989baf6",
    }
    assert {master.split for master in masters} == {"validation"}
    assert all(len(master.scan_ids) == 5 for master in masters)
    assert all(len(master.scan_indices) == 5 for master in masters)
    assert all(
        master.scan_ids == tuple(master.sequence_id.split("-")) for master in masters
    )


def test_exact_prefixes_share_one_order_for_names_and_resolved_indices() -> None:
    masters = load_t5_masters(
        _fixture_database(),
        _fixture_records(),
        expected_master_count=1,
        expected_cluster_count=1,
    )
    master = masters[0]

    for variant_name, variant in build_order_variants(
        master.scan_ids,
        master.scan_indices,
        seed=45,
    ).items():
        prefixes = derive_exact_prefixes(master, variant_name, variant)
        assert set(prefixes) == {2, 3, 4, 5}
        for horizon, prefix in prefixes.items():
            assert prefix.scan_ids == variant.scan_ids[:horizon]
            assert prefix.scan_indices == variant.scan_indices[:horizon]
            assert prefix.sequence_id == "-".join(prefix.scan_ids)
            assert prefix.master_sequence_id == master.sequence_id


def test_canonical_reverse_and_sha256_seed45_orders_are_deterministic_and_paired() -> (
    None
):
    scan_ids = (
        "scene0069_00",
        "scene0069_02",
        "scene0069_04",
        "scene0069_03",
        "scene0069_01",
    )
    scan_indices = (4, 0, 3, 1, 2)
    first = build_order_variants(scan_ids, scan_indices, seed=45)
    second = build_order_variants(scan_ids, scan_indices, seed=45)

    assert first == second
    assert first["canonical"].scan_ids == scan_ids
    assert first["canonical"].scan_indices == scan_indices
    assert first["reverse"].scan_ids == tuple(reversed(scan_ids))
    assert first["reverse"].scan_indices == tuple(reversed(scan_indices))
    assert sorted(first["sha256_seed45"].scan_ids) == sorted(scan_ids)
    assert sorted(first["sha256_seed45"].scan_indices) == sorted(scan_indices)
    assert all(
        dict(zip(first["sha256_seed45"].scan_ids, first["sha256_seed45"].scan_indices))[
            scan_id
        ]
        == scan_indices[scan_ids.index(scan_id)]
        for scan_id in scan_ids
    )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda db: db[
                "scene0069_00-scene0069_01-scene0069_02-scene0069_03-scene0069_04"
            ]["sub_scenes"].__setitem__(4, 3),
            "duplicate",
        ),
        (
            lambda db: db[
                "scene0069_00-scene0069_01-scene0069_02-scene0069_03-scene0069_04"
            ].update(type="train"),
            "split",
        ),
    ],
)
def test_invalid_master_metadata_fails_closed(mutator, message: str) -> None:
    database = {
        "scene0069_00-scene0069_01-scene0069_02-scene0069_03-scene0069_04": {
            "scene": 69,
            "sub_scenes": [0, 1, 2, 3, 4],
            "type": "validation",
        }
    }
    mutator(database)
    with pytest.raises(ProtocolError, match=message):
        load_t5_masters(
            database,
            _fixture_records(),
            expected_master_count=1,
            expected_cluster_count=1,
        )


def test_missing_duplicate_unsupervised_and_positional_records_fail_closed() -> None:
    database = _fixture_database()
    records = _fixture_records()

    with pytest.raises(ProtocolError, match="missing"):
        load_scan_indices(
            (
                "scene0069_00",
                "scene0069_01",
                "scene0069_02",
                "scene0069_03",
                "scene0069_04",
            ),
            records[:-1],
        )

    with pytest.raises(ProtocolError, match="duplicate"):
        load_scan_indices(
            (
                "scene0069_00",
                "scene0069_01",
                "scene0069_02",
                "scene0069_03",
                "scene0069_04",
            ),
            records + [records[0]],
        )

    unsupervised = [dict(record) for record in records]
    unsupervised[2]["supervised"] = False
    with pytest.raises(ProtocolError, match="supervised"):
        load_t5_masters(
            database,
            unsupervised,
            expected_master_count=1,
            expected_cluster_count=1,
        )

    positional = [{"index": index} for index in range(5)]
    with pytest.raises(ProtocolError, match="explicit"):
        load_scan_indices(
            (
                "scene0069_00",
                "scene0069_01",
                "scene0069_02",
                "scene0069_03",
                "scene0069_04",
            ),
            positional,
        )


def test_missing_scan_cannot_be_substituted_by_zero_or_any_other_index() -> None:
    with pytest.raises(ProtocolError, match="substitution"):
        load_scan_indices(
            ("scene0069_00", "scene0069_01"),
            _fixture_records()[:1],
            substitution_policy="zero",
        )


def test_manifest_has_portable_provenance_and_deterministic_bytes(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "sequence_database.yaml"
    metadata_path = tmp_path / "metadata.json"
    database_path.write_text(
        yaml.safe_dump(_fixture_database(), sort_keys=True), encoding="utf-8"
    )
    metadata_path.write_text(
        json.dumps({"fixture": True}, sort_keys=True), encoding="utf-8"
    )
    protocol = build_protocol_b(
        _fixture_database(),
        _fixture_records(),
        expected_master_count=1,
        expected_cluster_count=1,
    )

    manifest = build_protocol_b_manifest(
        protocol,
        sequence_database_path=database_path,
        metadata_path=metadata_path,
        repository_root=tmp_path,
    )
    assert manifest["schema_version"] == "protocol-b-v1"
    assert manifest["sources"]["sequence_database"]["reference"].startswith(
        ("repo:", "external:", "local_cache:")
    )
    assert (
        manifest["sources"]["metadata"]["sha256"]
        == hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    )
    serialized = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert "/home/" not in serialized
    assert "/Users/" not in serialized
    validate_protocol_b_manifest(manifest)

    output = tmp_path / "protocol_b_manifest.json"
    write_protocol_b_manifest(output, manifest)
    first = output.read_bytes()
    write_protocol_b_manifest(output, manifest)
    assert output.read_bytes() == first


def test_default_config_freezes_protocol_b_contract() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())
    assert config["protocol_b"]["master_horizon"] == 5
    assert config["protocol_b"]["horizons"] == [2, 3, 4, 5]
    assert config["protocol_b"]["split"] == "validation"
    assert config["protocol_b"]["expected_master_count"] == 43
    assert config["protocol_b"]["expected_reference_scene_clusters"] == 6
    assert config["protocol_b"]["order_variants"] == [
        "canonical",
        "reverse",
        "sha256_seed45",
    ]
    assert config["protocol_b"]["seed"] == 45
    assert config["protocol_b"]["substitution_policy"] == "reject"
    assert config["protocol_b"]["sources"]["sequence_database_sha256"] == (
        "252363f76524bb7eeff9f65b303aadda67dcd2646477daae1ac90f7f53398290"
    )
    assert config["protocol_b"]["sources"]["metadata_sha256"] == (
        "674a00f50f76b198b9de44efd86c390fea3da37ba8f12cf8ccd00045e265fa64"
    )


def test_default_config_freezes_complete_baseline_and_g6a4_parameters() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "conf/p6a/default.yaml").read_text())
    baselines = config["baselines"]

    assert baselines["b1"] == {
        "name": "previous_stage_feature_hungarian",
        "feature_threshold": 0.5,
    }
    assert baselines["b2"] == {
        "name": "previous_stage_feature_class_hungarian",
        "feature_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
        "class_probability": "foreground_renormalized",
    }
    assert baselines["b3"] == {
        "name": "active_previous_stage_ema",
        "feature_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
        "update_rate": 0.2,
        "dormant_lifecycle": False,
    }
    assert baselines["b4"] == {
        "name": "frozen_p5_persist4d",
        "capacity": 100,
        "association_threshold": 0.5,
        "class_weight": 0.25,
        "background_class": 18,
        "update_rate": 0.2,
        "confidence_threshold": 0.5,
        "mask_threshold": 0.5,
        "minimum_mask_support": 1,
    }

    assert config["gates"]["G6A-4"] == {
        "horizon": 2,
        "online_metric_drop_max": 0.05,
        "long_horizon_any_positive": True,
        "horizons": [4, 5],
        "metrics": ["t_mAP", "t_REC"],
    }
