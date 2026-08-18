from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from scripts.evaluate_persist4d_p6a import (
    atomic_manifest_payload,
    build_temporal_target,
    cache_payload_from_inference,
    cache_payload_to_frozen_observation,
    expected_cache_keys,
    materialize_prediction_cache,
    prefix_causality_coordinator,
    publish_manifest_atomic,
    resolve_cache_entry,
    stage_prediction_from_track_step,
)
from scripts.p6a_cache import load_cache_entry, write_cache_entry


def _payload(stage: int, *, gt_ids: tuple[int, ...] = (10, 20)) -> dict[str, object]:
    point_count = stage + 3
    query_count = 3
    masks = torch.zeros((query_count, point_count), dtype=torch.bool)
    masks[0, 0] = True
    masks[1, -1] = True
    return {
        "schema_version": 3,
        "key": {
            "master_sequence_id": "master",
            "reference_scene_id": "ref",
            "order_id": "canonical",
            "stage_index": stage,
            "history_scan_ids": [
                f"scene0001_{index:02d}" for index in range(stage + 1)
            ],
            "local_window_scan_ids": [
                f"scene0001_{index:02d}"
                for index in range(max(0, stage - 1), stage + 1)
            ],
        },
        "provenance": {
            "source_commit": "1" * 40,
            "checkpoint_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "dataset_sha256": "4" * 64,
        },
        "observation": {
            "features": torch.eye(query_count, 2),
            "class_prob": torch.tensor(
                [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]],
                dtype=torch.float32,
            ),
            "confidence": torch.tensor([0.9, 0.8, 0.1]),
            "valid": torch.tensor([True, True, False]),
            "masks": masks,
            "mask_support": masks.sum(dim=1, dtype=torch.long),
            "local_query_ids": torch.arange(query_count, dtype=torch.long),
        },
        "target": {
            "gt_ids": torch.tensor(gt_ids, dtype=torch.long),
            "gt_classes": torch.tensor(
                [1 if gt_id == 10 else 2 for gt_id in gt_ids], dtype=torch.long
            ),
            "gt_masks": torch.zeros((len(gt_ids), point_count), dtype=torch.bool),
            "changes": torch.zeros(len(gt_ids), dtype=torch.long),
            "change_labels_valid": False,
            "change_label_semantics": (
                "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
            ),
            "gt_class_semantics": "rescene_model_index_0_based",
        },
    }


@dataclass(frozen=True)
class _Master:
    sequence_id: str
    reference_scene_id: str


@dataclass(frozen=True)
class _Order:
    scan_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Protocol:
    masters: tuple[_Master, ...]
    variants: dict[str, dict[str, _Order]]
    order_variants: tuple[str, ...] = ("canonical", "reverse", "sha256_seed45")


def _protocol(count: int = 1) -> _Protocol:
    masters = tuple(
        _Master(f"master-{index}", f"ref-{index}") for index in range(count)
    )
    variants = {
        master.sequence_id: {
            order: _Order(tuple(f"scene{index:04d}_{stage:02d}" for stage in range(5)))
            for order in ("canonical", "reverse", "sha256_seed45")
        }
        for index, master in enumerate(masters)
    }
    return _Protocol(masters, variants)


def test_expected_cache_keys_are_exact_prefix_windows() -> None:
    keys = expected_cache_keys(_protocol())

    assert len(keys) == 15
    assert [key["stage_index"] for key in keys[:5]] == [0, 1, 2, 3, 4]
    assert keys[0]["history_scan_ids"] == ["scene0000_00"]
    assert keys[0]["local_window_scan_ids"] == ["scene0000_00"]
    assert keys[2]["history_scan_ids"] == [
        "scene0000_00",
        "scene0000_01",
        "scene0000_02",
    ]
    assert keys[2]["local_window_scan_ids"] == ["scene0000_01", "scene0000_02"]
    assert len({tuple(key["history_scan_ids"]) for key in keys}) == 5


def test_cache_payload_to_frozen_observation_detaches_all_cpu_tensors() -> None:
    payload = _payload(1)
    original = payload["observation"]["features"].clone()

    frozen = cache_payload_to_frozen_observation(payload)
    payload["observation"]["features"][0, 0] = 99.0

    assert frozen.features.device.type == "cpu"
    assert frozen.latest_mask[0].equal(
        payload["observation"]["masks"].to(dtype=frozen.features.dtype)
    )
    assert frozen.features.equal(original)
    assert frozen.features.data_ptr() != payload["observation"]["features"].data_ptr()


def test_cache_payload_from_inference_freezes_latest_stage_without_change_gt() -> None:
    class Observation:
        def __init__(self) -> None:
            self.features = torch.tensor(
                [[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True
            )
            self.class_prob = torch.tensor(
                [[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]]
            )
            self.confidence = torch.tensor([[0.8, 0.7]])
            self.valid = torch.tensor([[True, False]])
            self.latest_mask = [torch.ones((2, 3))]

        def validate(self) -> None:
            return None

    key = _payload(1)["key"]
    provenance = _payload(1)["provenance"]
    full_masks = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    full_target = {
        "ids": torch.tensor([10, 20, 30]),
        "labels": torch.tensor([1, 2, 3]),
        "masks": torch.tensor(
            [
                [True, False, False, False, False],
                [False, True, False, True, False],
                [False, False, True, False, True],
            ],
            dtype=torch.bool,
        ),
        "temporal_stages": torch.tensor([0, 0, 1, 1, 1]),
    }

    payload = cache_payload_from_inference(
        key=key,
        provenance=provenance,
        observation=Observation(),
        full_masks=full_masks,
        full_target=full_target,
        latest_local_stage=1,
    )

    assert payload["schema_version"] == 3
    assert payload["observation"]["features"].requires_grad is False
    assert payload["observation"]["features"].device.type == "cpu"
    assert payload["observation"]["masks"].equal(full_masks)
    assert payload["observation"]["mask_support"].tolist() == [2, 1]
    assert payload["target"]["gt_ids"].tolist() == [20, 30]
    assert payload["target"]["gt_classes"].tolist() == [2, 3]
    assert payload["target"]["gt_masks"].shape == (2, 3)
    assert payload["target"]["changes"].tolist() == [0, 0]
    assert payload["target"]["change_labels_valid"] is False
    assert payload["target"]["change_label_semantics"] == (
        "unavailable_for_protocol_b_order_stress_test_all_static_placeholder"
    )
    assert payload["target"]["gt_class_semantics"] == (
        "rescene_model_index_0_based"
    )


def test_build_temporal_target_unions_ids_and_marks_absent_stages_false() -> None:
    first = _payload(0, gt_ids=(10,))
    first["target"]["gt_masks"][0, 0] = True
    second = _payload(1, gt_ids=(20, 10))
    second["target"]["gt_masks"][0, 1] = True
    second["target"]["gt_masks"][1, 0] = True

    target = build_temporal_target([first, second])

    assert set(target) == {"masks", "labels", "ids", "changes", "temporal_stages"}
    assert target["masks"].shape == (2, 7)
    assert target["ids"].tolist() == [10, 20]
    assert target["labels"].tolist() == [1, 2]
    assert target["masks"][0].tolist() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert target["masks"][1].tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert target["temporal_stages"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert all(value.device.type == "cpu" for value in target.values())


def test_build_temporal_target_rejects_class_conflicts_or_nonzero_placeholders() -> None:
    first = _payload(0, gt_ids=(10,))
    second = _payload(1, gt_ids=(10,))
    second["target"]["gt_classes"][0] = 2
    with pytest.raises(ValueError, match="class"):
        build_temporal_target([first, second])

    second = _payload(1, gt_ids=(10,))
    second["target"]["changes"][0] = 1
    with pytest.raises(ValueError, match="change"):
        build_temporal_target([first, second])


def test_stage_prediction_filters_invalid_ids_and_excludes_background() -> None:
    payload = _payload(0)
    step = {
        "track_ids": (100, None, 200),
        "valid": (True, True, False),
        "stage_id": 0,
    }

    prediction = stage_prediction_from_track_step(payload, step, background_class=2)

    assert prediction["pred_masks"].shape == (3, 1)
    assert prediction["track_ids"].tolist() == [100]
    assert prediction["pred_classes"].tolist() == [1]
    assert prediction["pred_scores"].tolist() == [pytest.approx(0.9)]
    assert prediction["class_probs"].shape == (1, 3)


def test_stage_prediction_rejects_nonfinite_or_duplicate_track_ids() -> None:
    payload = _payload(0)
    with pytest.raises(ValueError, match="duplicate"):
        stage_prediction_from_track_step(
            payload,
            {"track_ids": (1, 1, None), "valid": (True, True, False)},
            background_class=2,
        )
    payload["observation"]["confidence"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        stage_prediction_from_track_step(
            payload,
            {"track_ids": (1, None, None), "valid": (True, True, False)},
            background_class=2,
        )


class _SpyTracker:
    def __init__(self, method: str, seen: dict[str, list[int]]) -> None:
        self.method = method
        self.seen = seen

    def step(self, observation, *, stage_id: int):
        self.seen[self.method].append(stage_id)
        return {
            "track_ids": (stage_id, None, None),
            "valid": (True, True, False),
            "stage_id": stage_id,
        }


def test_prefix_coordinator_is_causal_and_separates_offline_reconstruction() -> None:
    payloads = [_payload(stage) for stage in range(5)]
    seen: dict[str, list[int]] = {"B1": [], "B2": []}
    result = prefix_causality_coordinator(
        payloads,
        {method: (lambda method=method: _SpyTracker(method, seen)) for method in seen},
        endpoints=(1, 2, 3, 4),
        background_class=2,
    )

    assert result.online["B1"][1].stages == (0, 1)
    assert result.online_predictions["B1"][1]["track_ids"].tolist() == [0, 1]
    assert seen["B1"] == [
        0,
        1,
        0,
        1,
        2,
        0,
        1,
        2,
        3,
        0,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
        3,
        4,
    ]
    assert result.offline["B1"].stages == (0, 1, 2, 3, 4)
    assert result.online["B1"][1].stages == (0, 1)
    assert result.content_digest


def test_resumable_cache_reuses_manifest_entry_and_rejects_stale_provenance(
    tmp_path: Path,
) -> None:
    payload = _payload(0)
    entry = write_cache_entry(tmp_path, payload)
    manifest = {"entries": [entry]}
    calls = 0

    def producer() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return payload

    resolution = resolve_cache_entry(
        tmp_path,
        payload["key"],
        manifest,
        expected_provenance=payload["provenance"],
        producer=producer,
    )
    assert resolution.reused is True
    assert resolution.payload["key"] == payload["key"]
    assert calls == 0

    with pytest.raises(ValueError, match="provenance"):
        resolve_cache_entry(
            tmp_path,
            payload["key"],
            manifest,
            expected_provenance={**payload["provenance"], "config_sha256": "f" * 64},
            producer=producer,
        )


def test_materialize_prediction_cache_only_produces_missing_exact_keys(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    expected = expected_cache_keys(protocol)
    provenance = _payload(0)["provenance"]
    first_payload = _payload(0)
    first_payload["key"] = expected[0]
    write_cache_entry(tmp_path / "entries", first_payload)
    produced: list[dict[str, object]] = []

    def producer(key: dict[str, object]) -> dict[str, object]:
        produced.append(dict(key))
        payload = _payload(int(key["stage_index"]))
        payload["key"] = dict(key)
        return payload

    manifest = materialize_prediction_cache(
        protocol=protocol,
        cache_directory=tmp_path / "entries",
        manifest_path=tmp_path / "cache_manifest.json",
        provenance=provenance,
        producer=producer,
    )

    assert manifest["entry_count"] == 15
    assert produced == expected[1:]
    assert (tmp_path / "cache_manifest.json").is_file()


def test_atomic_manifest_payload_and_publish_are_deterministic(tmp_path: Path) -> None:
    payload = _payload(0)
    entry = write_cache_entry(tmp_path, payload)
    manifest = atomic_manifest_payload(
        [entry],
        expected_keys=[payload["key"]],
        expected_provenance=payload["provenance"],
    )
    destination = tmp_path / "manifest.json"

    publish_manifest_atomic(destination, manifest)

    assert destination.is_file()
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert load_cache_entry(
        tmp_path / entry["filename"], expected_provenance=payload["provenance"]
    )
