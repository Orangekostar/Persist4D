from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import reviewer_closure_sidecar as sidecar
from scripts.reviewer_closure_protocol import (
    build_reviewer_closure_manifest,
    full_history_observation_keys,
    validate_reviewer_closure_binding,
)
from scripts.system_comparison_inference import (
    FullHistoryPredictionProducer,
    build_full_history_cache_manifest,
    unpack_bool_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/reviewer_closure_sidecar.py"
CONFIG_PATH = REPO_ROOT / "configs/reviewer_closure/protocol.yaml"
SYSTEM_MANIFEST_PATH = (
    REPO_ROOT / "artifacts/system_comparison/system_comparison_manifest.json"
)
SOURCE_PREDICTION_MANIFEST_PATH = (
    REPO_ROOT / "artifacts/system_comparison/full_history_predictions/manifest.json"
)
CHECKPOINT_PATH = REPO_ROOT / "checkpoints/rescene4d_concerto_t2_repro.ckpt"


def test_reviewer_closure_sidecar_module_exists() -> None:
    assert MODULE_PATH.is_file()


def test_full_history_producer_keeps_old_call_api_while_exposing_bundle() -> None:
    assert callable(getattr(FullHistoryPredictionProducer, "produce_bundle", None))
    expected = {"content_sha256": "a" * 64}
    producer = object.__new__(FullHistoryPredictionProducer)
    producer.produce_bundle = lambda key: type(
        "Bundle", (), {"payload": expected, "processed": object()}
    )()

    assert producer({"unused": "delegated"}) is expected


def _api(name: str):
    value = getattr(sidecar, name, None)
    assert value is not None, f"missing reviewer-closure sidecar API: {name}"
    return value


def _key() -> dict[str, object]:
    return {
        "reference_scene_id": "reference-1",
        "master_sequence_id": "scene0000_00-scene0000_01-scene0000_02",
        "order_id": "canonical",
        "horizon": 3,
        "history_scan_ids": ["scene0000_00", "scene0000_01", "scene0000_02"],
        "scan_indices": [4, 5, 6],
    }


def _raw_observation() -> dict[str, torch.Tensor]:
    masks = torch.tensor(
        [
            [True, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    return {
        "features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
        ),
        "class_prob": torch.tensor(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ],
            dtype=torch.float32,
        ),
        "confidence": torch.tensor([0.8, 0.8, 0.6], dtype=torch.float32),
        "valid": torch.tensor([True, True, False]),
        "masks": masks,
        "mask_support": masks.sum(dim=1, dtype=torch.long),
        "local_query_ids": torch.tensor([0, 1, 2], dtype=torch.long),
    }


def _source_prediction() -> dict[str, object]:
    observation = _raw_observation()
    return {
        "key": _key(),
        "content_sha256": "1" * 64,
        "provenance": {
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "protocol_sha256": "4" * 64,
            "source_commit": "5" * 40,
        },
        "observation_fingerprints": _api("observation_fingerprints")(observation),
    }


def _payload() -> dict[str, object]:
    return _api("build_full_history_observation_sidecar")(
        key=_key(),
        raw_observation=_raw_observation(),
        source_prediction=_source_prediction(),
        reference_prediction_content_sha256="7" * 64,
        sidecar_source_commit="6" * 40,
    )


def _replay_pair() -> dict[str, object]:
    return _api("build_full_history_replay_pair")(
        reference_prediction_content_sha256="7" * 64,
        replay_prediction=_source_prediction(),
        sidecar=_payload(),
    )


def test_sidecar_schema_binds_source_and_keeps_only_tracker_observation() -> None:
    payload = _api("validate_full_history_observation_sidecar")(_payload())

    assert set(payload) == {
        "schema_version",
        "key",
        "provenance",
        "observation",
        "source_observation_fingerprints",
        "content_sha256",
    }
    assert payload["schema_version"] == "full-history-observations-v2"
    assert payload["key"] == _key()
    assert payload["provenance"] == {
        "source_prediction_content_sha256": "1" * 64,
        "reference_prediction_content_sha256": "7" * 64,
        "checkpoint_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "source_commit": "6" * 40,
    }
    assert set(payload["observation"]) == {
        "features",
        "class_prob",
        "confidence",
        "valid",
        "current_stage_masks",
        "mask_support",
        "local_query_ids",
    }
    restored = unpack_bool_matrix(payload["observation"]["current_stage_masks"])
    assert torch.equal(restored, _raw_observation()["masks"])
    assert not any(
        name in payload for name in ("task_prediction", "identity_prediction", "target")
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update(
                features=torch.ones((2, 2), dtype=torch.float32)
            ),
            "query|align",
        ),
        (
            lambda value: value.update(masks=torch.ones((2, 4), dtype=torch.bool)),
            "query|align",
        ),
        (
            lambda value: value.update(
                mask_support=torch.tensor([9, 1, 1], dtype=torch.long)
            ),
            "support",
        ),
        (
            lambda value: value.update(
                local_query_ids=torch.tensor([0, 0, 2], dtype=torch.long)
            ),
            "query IDs|unique",
        ),
    ],
)
def test_sidecar_rejects_feature_class_mask_query_misalignment(
    mutation, message: str
) -> None:
    observation = _raw_observation()
    mutation(observation)

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match=message):
        _api("build_full_history_observation_sidecar")(
            key=_key(),
            raw_observation=observation,
            source_prediction=_source_prediction(),
            reference_prediction_content_sha256="7" * 64,
            sidecar_source_commit="6" * 40,
        )


def test_sidecar_rejects_future_or_different_source_prefix() -> None:
    source = _source_prediction()
    source["key"] = copy.deepcopy(source["key"])
    source["key"]["history_scan_ids"][-1] = "future_scan"

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="source prediction key|prefix"):
        _api("build_full_history_observation_sidecar")(
            key=_key(),
            raw_observation=_raw_observation(),
            source_prediction=source,
            reference_prediction_content_sha256="7" * 64,
            sidecar_source_commit="6" * 40,
        )


def test_sidecar_rejects_observation_different_from_source_fingerprint() -> None:
    observation = _raw_observation()
    observation["features"][0, 0] = 0.75

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="fingerprint"):
        _api("build_full_history_observation_sidecar")(
            key=_key(),
            raw_observation=observation,
            source_prediction=_source_prediction(),
            reference_prediction_content_sha256="7" * 64,
            sidecar_source_commit="6" * 40,
        )


def test_sidecar_content_hash_detects_tensor_tampering() -> None:
    tampered = copy.deepcopy(_payload())
    tampered["observation"]["features"][0, 0] += 0.1

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="fingerprint|content digest"):
        _api("validate_full_history_observation_sidecar")(tampered)


def test_sidecar_entry_is_atomic_reusable_and_refuses_conflict(
    tmp_path: Path,
) -> None:
    payload = _payload()
    first = _api("write_full_history_observation_sidecar_entry")(tmp_path, payload)
    second = _api("write_full_history_observation_sidecar_entry")(tmp_path, payload)

    assert first == second
    assert first["sidecar_source_commit"] == "6" * 40
    loaded = _api("load_full_history_observation_sidecar_entry")(tmp_path, first)
    assert loaded["content_sha256"] == payload["content_sha256"]
    changed = copy.deepcopy(payload)
    changed["observation"]["confidence"][0] -= 0.1
    changed["source_observation_fingerprints"] = _api("observation_fingerprints")(
        {
            **_raw_observation(),
            "confidence": changed["observation"]["confidence"],
        }
    )
    changed["content_sha256"] = _api("sidecar_content_sha256")(changed)
    with pytest.raises(FileExistsError, match="different content"):
        _api("write_full_history_observation_sidecar_entry")(tmp_path, changed)
    assert not list(tmp_path.glob(".*.tmp"))


def test_replay_pair_stage_is_atomic_recoverable_and_exactly_removed(
    tmp_path: Path,
) -> None:
    pair = _replay_pair()
    first = _api("write_full_history_replay_pair_stage")(tmp_path, pair)
    second = _api("write_full_history_replay_pair_stage")(tmp_path, pair)

    assert first == second
    loaded = _api("load_full_history_replay_pair_stage")(tmp_path, first)
    assert loaded["content_sha256"] == pair["content_sha256"]
    assert loaded["sidecar"]["content_sha256"] == pair["sidecar"]["content_sha256"]
    assert _api("discover_full_history_replay_pair_stages")(tmp_path) == [first]

    changed = copy.deepcopy(pair)
    changed["sidecar"]["observation"]["confidence"][0] -= 0.1
    changed["sidecar"]["source_observation_fingerprints"] = _api(
        "observation_fingerprints"
    )(
        {
            **_raw_observation(),
            "confidence": changed["sidecar"]["observation"]["confidence"],
        }
    )
    changed["sidecar"]["content_sha256"] = _api("sidecar_content_sha256")(
        changed["sidecar"]
    )
    changed["content_sha256"] = _api("replay_pair_content_sha256")(changed)
    with pytest.raises(FileExistsError, match="different content"):
        _api("write_full_history_replay_pair_stage")(tmp_path, changed)

    _api("remove_full_history_replay_pair_stage")(tmp_path, first)
    assert not list(tmp_path.iterdir())


def _reviewer_manifest() -> dict[str, object]:
    binding = validate_reviewer_closure_binding(
        CONFIG_PATH,
        repo_root=REPO_ROOT,
        checkpoint_path=CHECKPOINT_PATH,
    )
    return build_reviewer_closure_manifest(
        CONFIG_PATH,
        system_manifest_path=SYSTEM_MANIFEST_PATH,
        binding=binding,
    )


def _source_manifest() -> dict[str, object]:
    return json.loads(SOURCE_PREDICTION_MANIFEST_PATH.read_text(encoding="utf-8"))


def _system_manifest() -> dict[str, object]:
    return json.loads(SYSTEM_MANIFEST_PATH.read_text(encoding="utf-8"))


def test_source_prediction_manifest_rebuilds_exact_645_entry_binding() -> None:
    validated = _api("validate_source_prediction_manifest")(
        _source_manifest(), system_manifest=_system_manifest()
    )

    assert validated["status"] == "pass"
    assert validated["entry_count"] == 43 * 3 * 5
    assert validated["content_sha256"] == (
        "8cd2a01bcbfc581c83daceb7d92f33842dcdb7c0e46903aa66665aa2f2109453"
    )


def test_each_o2_to_o5_key_maps_to_one_source_prediction_record() -> None:
    keys = full_history_observation_keys(_reviewer_manifest())
    source = _api("validate_source_prediction_manifest")(
        _source_manifest(), system_manifest=_system_manifest()
    )
    records = [_api("source_prediction_entry_for_key")(source, key) for key in keys]

    assert len(records) == 43 * 3 * 4
    assert len({record["content_sha256"] for record in records}) == len(records)
    assert {record["key"]["horizon"] for record in records} == {2, 3, 4, 5}


def _record(
    key: dict[str, object], source_record: dict[str, object], index: int
) -> dict[str, object]:
    identity = json.dumps(key, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return {
        "key": key,
        "filename": hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".pt",
        "sha256": f"{index + 1:064x}",
        "byte_size": index + 1,
        "content_sha256": f"{index + 11:064x}",
        "source_prediction_content_sha256": source_record["content_sha256"],
        "reference_prediction_content_sha256": source_record["content_sha256"],
        "sidecar_source_commit": "a" * 40,
    }


def test_sidecar_manifest_requires_exact_coverage_and_source_content_binding() -> None:
    reviewer = _reviewer_manifest()
    keys = full_history_observation_keys(reviewer)[:2]
    source = _api("validate_source_prediction_manifest")(
        _source_manifest(), system_manifest=_system_manifest()
    )
    records = [
        _record(
            key,
            _api("source_prediction_entry_for_key")(source, key),
            index,
        )
        for index, key in enumerate(keys)
    ]
    replay_records = []
    for key, record in zip(keys, records, strict=True):
        source_record = _api("source_prediction_entry_for_key")(source, key)
        replay_records.append(
            {
                "key": source_record["key"],
                "filename": source_record["filename"],
                "sha256": record["sha256"],
                "byte_size": record["byte_size"],
                "content_sha256": record["source_prediction_content_sha256"],
            }
        )
    replay_manifest = build_full_history_cache_manifest(
        replay_records,
        expected_keys=[record["key"] for record in replay_records],
        expected_provenance=source["provenance"],
    )
    manifest = _api("build_full_history_observation_sidecar_manifest")(
        records,
        expected_keys=keys,
        source_prediction_manifest=source,
        replay_prediction_manifest=replay_manifest,
        system_manifest=_system_manifest(),
        reviewer_manifest=reviewer,
        sidecar_code_commit="a" * 40,
    )

    assert manifest["schema_version"] == "full-history-observations-v2-manifest"
    assert manifest["status"] == "pass"
    assert manifest["entry_count"] == 2
    assert (
        manifest["source_prediction_manifest"]["content_sha256"]
        == source["content_sha256"]
    )
    assert (
        manifest["replay_prediction_manifest"]["content_sha256"]
        == replay_manifest["content_sha256"]
    )
    assert manifest["reviewer_manifest_content_sha256"] == reviewer["content_sha256"]
    assert manifest["sidecar_code_commit"] == "a" * 40

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="exact coverage"):
        _api("build_full_history_observation_sidecar_manifest")(
            records[:1],
            expected_keys=keys,
            source_prediction_manifest=source,
            replay_prediction_manifest=replay_manifest,
            system_manifest=_system_manifest(),
            reviewer_manifest=reviewer,
            sidecar_code_commit="a" * 40,
        )
    changed = copy.deepcopy(records)
    changed[0]["reference_prediction_content_sha256"] = "f" * 64
    with pytest.raises(error, match="reference prediction"):
        _api("build_full_history_observation_sidecar_manifest")(
            changed,
            expected_keys=keys,
            source_prediction_manifest=source,
            replay_prediction_manifest=replay_manifest,
            system_manifest=_system_manifest(),
            reviewer_manifest=reviewer,
            sidecar_code_commit="a" * 40,
        )
    wrong_code = copy.deepcopy(records)
    wrong_code[0]["sidecar_source_commit"] = "b" * 40
    with pytest.raises(error, match="sidecar code commit"):
        _api("build_full_history_observation_sidecar_manifest")(
            wrong_code,
            expected_keys=keys,
            source_prediction_manifest=source,
            replay_prediction_manifest=replay_manifest,
            system_manifest=_system_manifest(),
            reviewer_manifest=reviewer,
            sidecar_code_commit="a" * 40,
        )


def test_bound_sidecar_production_uses_one_bundle_and_binds_replay_to_reference() -> (
    None
):
    reference = {
        **_source_prediction(),
        "input_stats": {"scan_count": 3},
        "target": {"ids": torch.tensor([11, 22], dtype=torch.long)},
    }
    replay = copy.deepcopy(reference)
    replay["content_sha256"] = "8" * 64

    class Producer:
        calls = 0

        def produce_bundle(self, key):
            self.calls += 1
            assert key == reference["key"]
            return SimpleNamespace(
                payload=copy.deepcopy(replay),
                processed=SimpleNamespace(raw_observation=_raw_observation()),
            )

    producer = Producer()
    pair = _api("produce_bound_full_history_observation_sidecar")(
        producer=producer,
        source_prediction=reference,
        sidecar_key=_key(),
        sidecar_source_commit="6" * 40,
    )
    assert producer.calls == 1
    assert pair["replay_prediction"]["content_sha256"] == "8" * 64
    assert pair["sidecar"]["provenance"]["source_prediction_content_sha256"] == (
        "8" * 64
    )
    assert pair["sidecar"]["provenance"]["reference_prediction_content_sha256"] == (
        reference["content_sha256"]
    )

    class MismatchProducer(Producer):
        def produce_bundle(self, key):
            bundle = super().produce_bundle(key)
            bundle.payload["target"]["ids"][0] = 999
            return bundle

    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="target|replay"):
        _api("produce_bound_full_history_observation_sidecar")(
            producer=MismatchProducer(),
            source_prediction=reference,
            sidecar_key=_key(),
            sidecar_source_commit="6" * 40,
        )


def test_sidecar_discovery_recomputes_records_and_rejects_directory_junk(
    tmp_path: Path,
) -> None:
    payload = _payload()
    record = _api("write_full_history_observation_sidecar_entry")(tmp_path, payload)

    assert _api("discover_full_history_observation_sidecar_entries")(tmp_path) == [
        record
    ]
    (tmp_path / "junk.txt").write_text("not a sidecar", encoding="utf-8")
    error = _api("FullHistoryObservationSidecarError")
    with pytest.raises(error, match="unexpected path"):
        _api("discover_full_history_observation_sidecar_entries")(tmp_path)
