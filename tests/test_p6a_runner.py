from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import scripts.evaluate_persist4d_p6a as p6a_evaluator
from scripts.evaluate_persist4d_p6a import (
    CachedProtocolSequence,
    RealPredictionCacheProducer,
    _argument_parser,
    _cache_artifact_path,
    atomic_manifest_payload,
    build_association_events,
    build_cache_provenance,
    build_capacity_snapshots,
    build_rio_class_mapper,
    build_temporal_target,
    build_tracker_factories,
    cache_payload_from_inference,
    cache_payload_to_frozen_observation,
    evaluate_cached_task_metrics,
    expected_cache_keys,
    load_cached_protocol_sequences,
    materialize_prediction_cache,
    normalize_official_metric_blocks,
    prefix_causality_coordinator,
    publish_manifest_atomic,
    resolve_cache_entry,
    resolve_protocol_cache_request,
    run_real_prediction_cache,
    stage_prediction_from_track_step,
)
from scripts.p6a_analysis import aggregate_event_metrics, aggregate_metrics_by_sequence
from scripts.p6a_association import (
    B0SanityTracker,
    B0StageUniqueTracker,
    B1FeatureTracker,
    B2FeatureClassTracker,
    B3EmaTracker,
    B4PersistentTracker,
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
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]
    validation_index: int


@dataclass(frozen=True)
class _Order:
    scan_ids: tuple[str, ...]
    scan_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Protocol:
    masters: tuple[_Master, ...]
    variants: dict[str, dict[str, _Order]]
    order_variants: tuple[str, ...] = ("canonical", "reverse", "sha256_seed45")


def _protocol(count: int = 1) -> _Protocol:
    masters = tuple(
        _Master(
            f"master-{index}",
            f"ref-{index}",
            tuple(f"scene{index:04d}_{stage:02d}" for stage in range(5)),
            tuple(range(index * 5, index * 5 + 5)),
            index,
        )
        for index in range(count)
    )
    variants = {}
    for master in masters:
        permutations = {
            "canonical": (0, 1, 2, 3, 4),
            "reverse": (4, 3, 2, 1, 0),
            "sha256_seed45": (2, 4, 1, 3, 0),
        }
        variants[master.sequence_id] = {
            order: _Order(
                tuple(master.scan_ids[position] for position in positions),
                tuple(master.scan_indices[position] for position in positions),
            )
            for order, positions in permutations.items()
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
    assert len({tuple(key["history_scan_ids"]) for key in keys}) == 15


def test_protocol_cache_request_resolves_exact_global_scan_indices() -> None:
    protocol = _protocol()
    key = expected_cache_keys(protocol)[7]

    request = resolve_protocol_cache_request(protocol, key)

    assert request.context_index == 0
    assert request.master_sequence_id == "master-0"
    assert request.order_id == "reverse"
    assert request.stage_index == 2
    assert request.scan_indices == (3, 2)
    changed = dict(key)
    changed["history_scan_ids"] = [*key["history_scan_ids"][:-1], "scene9999_00"]
    changed["local_window_scan_ids"] = changed["history_scan_ids"][-2:]
    with pytest.raises(ValueError, match="Protocol B"):
        resolve_protocol_cache_request(protocol, changed)


def test_cache_provenance_binds_checkpoint_configs_and_protocol(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_manifest = {"schema_version": "protocol-b-v1", "masters": [1]}

    first = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"p6a": b"p6a", "runtime": b"runtime:\n  enabled: true\n"},
        protocol_manifest=protocol_manifest,
    )
    second = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"runtime": b"runtime:\n  enabled: true\n", "p6a": b"p6a"},
        protocol_manifest=protocol_manifest,
    )

    assert first == second
    assert set(first) == {
        "source_commit",
        "checkpoint_sha256",
        "config_sha256",
        "dataset_sha256",
    }
    changed = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={
            "p6a": b"changed",
            "runtime": b"runtime:\n  enabled: true\n",
        },
        protocol_manifest=protocol_manifest,
    )
    assert changed["config_sha256"] != first["config_sha256"]


def test_cache_provenance_uses_portable_concerto_checkpoint_reference(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_manifest = {"schema_version": "protocol-b-v1", "masters": [1]}
    local_checkpoint = "/" + "home/user/.cache/concerto/concerto_base.pth"
    local_runtime = (
        f"backbone:\n  model_lib: concerto\n  name: {local_checkpoint}\n"
    ).encode()
    portable_runtime = (
        b"backbone:\n"
        b"  model_lib: concerto\n"
        b"  name: external:concerto/concerto_base.pth\n"
    )

    local = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"p6a": b"p6a", "runtime": local_runtime},
        protocol_manifest=protocol_manifest,
    )
    portable = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"p6a": b"p6a", "runtime": portable_runtime},
        protocol_manifest=protocol_manifest,
    )

    assert local["config_sha256"] == portable["config_sha256"]


@pytest.mark.parametrize(
    "local_checkpoint",
    (
        "C:" + "\\Users\\alice\\models\\concerto_base.pth",
        "\\" * 2 + "server\\models\\concerto_base.pth",
    ),
)
def test_cache_provenance_portabilizes_windows_concerto_paths(
    tmp_path: Path,
    local_checkpoint: str,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    protocol_manifest = {"schema_version": "protocol-b-v1", "masters": [1]}
    local_runtime = yaml.safe_dump(
        {"backbone": {"model_lib": "concerto", "name": local_checkpoint}},
        sort_keys=True,
    ).encode()
    portable_runtime = yaml.safe_dump(
        {
            "backbone": {
                "model_lib": "concerto",
                "name": "external:concerto/concerto_base.pth",
            }
        },
        sort_keys=True,
    ).encode()

    local = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"p6a": b"p6a", "runtime": local_runtime},
        protocol_manifest=protocol_manifest,
    )
    portable = build_cache_provenance(
        source_commit="a" * 40,
        checkpoint_path=checkpoint,
        config_documents={"p6a": b"p6a", "runtime": portable_runtime},
        protocol_manifest=protocol_manifest,
    )

    assert local["config_sha256"] == portable["config_sha256"]


def test_real_prediction_cache_producer_runs_one_exact_local_forward() -> None:
    protocol = _protocol()
    key = expected_cache_keys(protocol)[6]
    calls: list[tuple[object, ...]] = []

    class Dataset:
        def __init__(self) -> None:
            self.sequence_names = ["master-0"]
            self.sequence_indices = np.asarray([(0, 1, 2, 3, 4)], dtype=int)

        def load_scan_indices(
            self,
            context_index: int,
            scan_indices: tuple[int, ...],
            *,
            change_file: object,
        ) -> str:
            calls.append(("load", context_index, scan_indices, change_file))
            return "sample"

    full_target = {
        "ids": torch.tensor([10]),
        "labels": torch.tensor([1]),
        "masks": torch.tensor([[False, True, False]]),
        "temporal_stages": torch.tensor([1, 1, 1]),
    }
    data = SimpleNamespace(target_full=[full_target])
    target = {
        "point2segment": torch.tensor([0, 1]),
        "temporal_stages": torch.tensor([0, 1]),
    }

    def collate(samples: list[str]):
        calls.append(("collate", tuple(samples)))
        return data, [target], ["master-0"]

    class System:
        def _process_raw_coordinates(self, received: object) -> torch.Tensor:
            calls.append(("raw", received))
            return torch.ones((2, 3))

        def __call__(self, received: object, **kwargs: object):
            calls.append(("forward", received, kwargs))
            return {"output": True}

    class Observation:
        features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        class_prob = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]])
        confidence = torch.tensor([[0.8, 0.7]])
        valid = torch.tensor([[True, True]])

        def validate(self) -> None:
            return None

    producer = RealPredictionCacheProducer(
        protocol=protocol,
        provenance=_payload(0)["provenance"],
        dataset=Dataset(),
        collate=collate,
        system=System(),
        device=torch.device("cpu"),
        observation_settings={
            "background_class": 2,
            "confidence_threshold": 0.5,
            "mask_threshold": 0.5,
            "minimum_mask_support": 1,
        },
        move_data=lambda received, _device: received,
        move_targets=lambda received, _device: received,
        segment_stages=lambda _target: torch.tensor([0, 1]),
        latest_masks=lambda *_args, **_kwargs: torch.tensor(
            [[True, False, True], [False, True, False]], dtype=torch.bool
        ),
        observation_builder=lambda *_args, **_kwargs: Observation(),
    )

    task_calls: list[dict[str, object]] = []

    def build_task(**kwargs: object) -> int:
        task_calls.append(kwargs)
        return int(kwargs["latest_stage_index"])

    bundle = producer.produce_bundle(
        key,
        task_prediction_builder=build_task,
        class_mapper=lambda value: value,
    )
    payload = bundle.payload

    assert calls[0] == ("load", 0, (4, 3), None)
    assert calls[1] == ("collate", ("sample",))
    assert [call[0] for call in calls].count("forward") == 1
    forward_kwargs = next(call[2] for call in calls if call[0] == "forward")
    assert forward_kwargs["point2segment"] == [target["point2segment"]]
    assert forward_kwargs["is_eval"] is True
    assert payload["key"] == key
    assert payload["target"]["gt_ids"].tolist() == [10]
    assert bundle.task_prediction == 1
    assert len(task_calls) == 1
    assert task_calls[0]["output"] == {"output": True}

    calls.clear()
    assert producer(key)["key"] == key
    assert [call[0] for call in calls].count("forward") == 1


def test_real_cache_run_rejects_repository_cache_before_runtime_setup() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        run_real_prediction_cache(
            cache_directory=Path("artifacts/P6A/cache"),
            protocol_manifest_path=Path("artifacts/P6A/protocol_b_manifest.json"),
            cache_manifest_path=Path("artifacts/P6A/cache_manifest.json"),
            metadata_path=Path("external/3RScan.json"),
            checkpoint_path=Path("checkpoints/rescene4d_concerto_t2_repro.ckpt"),
            device_name="cuda:0",
        )


def test_cache_manifests_default_inside_the_external_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    assert (
        _cache_artifact_path(cache_dir, None, filename="protocol_b_manifest.json")
        == cache_dir / "protocol_b_manifest.json"
    )
    assert (
        _cache_artifact_path(cache_dir, None, filename="cache_manifest.json")
        == cache_dir / "cache_manifest.json"
    )

    args = _argument_parser().parse_args(
        ["--cache-directory", str(cache_dir), "--metadata", "metadata.json"]
    )
    assert args.protocol_manifest is None
    assert args.cache_manifest is None


def test_cache_manifests_cannot_escape_the_external_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    with pytest.raises(ValueError, match="inside prediction cache"):
        _cache_artifact_path(
            cache_dir,
            tmp_path / "elsewhere" / "cache_manifest.json",
            filename="cache_manifest.json",
        )


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


def test_cache_payload_to_frozen_observation_rejects_inconsistent_mask_support() -> (
    None
):
    observation = _payload(1)["observation"]
    observation["mask_support"] = observation["mask_support"].clone()
    observation["mask_support"][0] += 1

    with pytest.raises(ValueError, match="mask_support.*mask point count"):
        cache_payload_to_frozen_observation(observation)


def test_cache_payload_from_inference_freezes_latest_stage_without_change_gt() -> None:
    class Observation:
        def __init__(self) -> None:
            self.features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
            self.class_prob = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]])
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
        # Pointcept preserves this coordinate column as float in real eval data.
        "temporal_stages": torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
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
    assert payload["target"]["gt_class_semantics"] == ("rescene_model_index_0_based")

    full_target["temporal_stages"][0] = 0.5
    with pytest.raises(ValueError, match="integer stage indices"):
        cache_payload_from_inference(
            key=key,
            provenance=provenance,
            observation=Observation(),
            full_masks=full_masks,
            full_target=full_target,
            latest_local_stage=1,
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


def test_build_temporal_target_keeps_first_observed_class_for_conflicting_gt() -> None:
    first = _payload(0, gt_ids=(10,))
    second = _payload(1, gt_ids=(10,))
    second["target"]["gt_classes"][0] = 2

    target = build_temporal_target([first, second])

    assert target["ids"].tolist() == [10]
    assert target["labels"].tolist() == [1]


def test_build_temporal_target_rejects_nonzero_change_placeholders() -> None:
    first = _payload(0, gt_ids=(10,))

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
    assert prediction["class_probs"].tolist() == [
        [pytest.approx(0.1), pytest.approx(0.8), 0.0]
    ]


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
    assert tuple(step["stage_id"] for step in result.online_steps["B1"][1]) == (0, 1)
    assert tuple(step["stage_id"] for step in result.offline_steps["B1"]) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert result.content_digest


def test_cached_task_metrics_separate_raw_online_and_offline_with_class_mapping() -> (
    None
):
    payloads = []
    for stage in range(5):
        payload = _payload(stage, gt_ids=(10,))
        payload["target"]["gt_masks"][0, -1] = True
        payloads.append(payload)
    sequence = CachedProtocolSequence(
        reference_scene_id="ref",
        master_sequence_id="master",
        order_id="canonical",
        payloads=tuple(payloads),
    )
    updates: dict[tuple[str, str, int], int] = {}

    class Metric:
        def __init__(self, mode: str, method: str, horizon: int) -> None:
            self.key = (mode, method, horizon)

        def update(self, prediction, target) -> None:
            assert torch.all(prediction["pred_classes"] >= 20)
            assert torch.all(target["labels"] >= 20)
            updates[self.key] = updates.get(self.key, 0) + 1

        def compute(self) -> dict[str, float]:
            score = float(self.key[2]) / 10.0
            return {
                "score": score,
                "online_t-mAP": score,
                "online_t-REC": score / 2.0,
            }

        def export_evidence(self) -> dict[str, object]:
            return {"metric_key": list(self.key)}

    factories = {
        "B0": lambda sequence_id: B0StageUniqueTracker(sequence_id=sequence_id),
        "B0_sanity": lambda sequence_id: B0SanityTracker(sequence_id=sequence_id),
        "B1": lambda sequence_id: B1FeatureTracker(sequence_id=sequence_id),
        "B2": lambda sequence_id: B2FeatureClassTracker(
            sequence_id=sequence_id, background_class=2
        ),
        "B3": lambda sequence_id: B3EmaTracker(
            sequence_id=sequence_id, background_class=2
        ),
        "B4": lambda sequence_id: B4PersistentTracker(
            sequence_id=sequence_id, capacity=3
        ),
    }

    result = evaluate_cached_task_metrics(
        [sequence],
        tracker_factories=factories,
        class_mapper=lambda value: value + 20,
        metric_factory=lambda mode, method, horizon: Metric(mode, method, horizon),
        background_class=2,
    )

    assert set(result.metric_blocks["raw"]) == set(factories)
    assert set(result.metric_blocks["strict"]) == set(factories)
    assert set(result.metric_blocks["offline"]) == {*factories, "Oracle"}
    assert result.metric_blocks["raw"]["B0"]["T2"]["score"] == 0.2
    assert result.metric_blocks["raw"]["B4"]["T5"]["score"] == 0.5
    assert updates[("raw_local", "shared", 2)] == 1
    assert updates[("strict_online", "B4", 5)] == 2
    assert updates[("offline_reconstructed", "Oracle", 5)] == 1
    assert len(set(result.fingerprints["prediction"].values())) == 1
    assert len(set(result.fingerprints["cache"].values())) == 1
    assert result.association_events
    assert len(result.capacity_snapshots) == 2 + 3 + 4 + 5
    assert len(aggregate_metrics_by_sequence(result.association_events)) == 6 * 4
    assert len(result.per_sequence_metrics) == len(factories) * 4
    assert {(row["method"], row["T"]) for row in result.per_sequence_metrics} == {
        (method, f"T{horizon}") for method in factories for horizon in range(2, 6)
    }
    p6b_row = next(
        row
        for row in result.per_sequence_metrics
        if row["method"] == "B4" and row["T"] == "T5"
    )
    assert p6b_row == {
        "method": "B4",
        "reference_scene_id": "ref",
        "master_sequence_id": "master",
        "order_id": "canonical",
        "T": "T5",
        "t_mAP": 0.5,
        "t_REC": 0.25,
        "prediction_digest": p6b_row["prediction_digest"],
    }
    assert len({row["prediction_digest"] for row in result.per_sequence_metrics}) == 1
    assert len(result.per_sequence_metric_evidence) == len(factories) * 4
    evidence = next(
        row
        for row in result.per_sequence_metric_evidence
        if row["method"] == "B4" and row["T"] == "T5"
    )
    assert evidence == {
        "method": "B4",
        "reference_scene_id": "ref",
        "master_sequence_id": "master",
        "order_id": "canonical",
        "T": "T5",
        "prediction_digest": p6b_row["prediction_digest"],
        "state": {"metric_key": ["strict_online", "B4", 5]},
    }


def test_tracker_factories_lock_the_preregistered_baseline_parameters() -> None:
    config = {
        "baselines": {
            "b0": {"name": "stage_unique_ids"},
            "b0_sanity": {"name": "local_query_index"},
            "b1": {"feature_threshold": 0.5},
            "b2": {
                "feature_threshold": 0.5,
                "class_weight": 0.25,
                "background_class": 18,
            },
            "b3": {
                "feature_threshold": 0.5,
                "class_weight": 0.25,
                "background_class": 18,
                "update_rate": 0.2,
            },
            "b4": {
                "capacity": 100,
                "association_threshold": 0.5,
                "class_weight": 0.25,
                "background_class": 18,
                "update_rate": 0.2,
            },
        }
    }

    factories = build_tracker_factories(config)

    assert tuple(factories) == ("B0", "B0_sanity", "B1", "B2", "B3", "B4")
    trackers = {method: factory("sequence") for method, factory in factories.items()}
    assert isinstance(trackers["B0"], B0StageUniqueTracker)
    assert isinstance(trackers["B0_sanity"], B0SanityTracker)
    assert trackers["B1"].feature_threshold == pytest.approx(0.5)
    assert trackers["B2"].class_weight == pytest.approx(0.25)
    assert trackers["B3"].update_rate == pytest.approx(0.2)
    assert trackers["B4"].memory.capacity == 100
    assert trackers["B4"].memory.association_threshold == pytest.approx(0.5)


def test_rio_class_mapper_applies_label_offset_before_dataset_remap() -> None:
    calls = []

    class Dataset:
        label_offset = 2

        def _remap_model_output(self, values: torch.Tensor) -> torch.Tensor:
            calls.append(values.clone())
            return values + 100

    mapper = build_rio_class_mapper(Dataset(), foreground_class_count=18)

    assert mapper(0) == 102
    assert mapper(17) == 119
    assert [value.tolist() for value in calls] == [[2], [19]]
    with pytest.raises(ValueError, match="foreground model class"):
        mapper(18)


def test_association_events_reconstruct_gap_reactivation_and_identity_history() -> None:
    payloads = [
        _payload(stage, gt_ids=((10,) if stage != 1 else ())) for stage in range(3)
    ]
    for stage in (0, 2):
        payloads[stage]["target"]["gt_masks"][0] = payloads[stage]["observation"][
            "masks"
        ][0]
    payloads[1]["observation"]["valid"][:] = False
    diagnostics = SimpleNamespace(
        selected_candidate_identity=(5, None, None),
        best_candidate_identity=(5, None, None),
        chosen_feature_similarity=(0.9, None, None),
        chosen_class_similarity=(0.8, None, None),
        chosen_total_score=(1.1, None, None),
        best_score=(1.1, None, None),
        second_best_score=(0.2, None, None),
        score_margin=(0.9, None, None),
        slot_age=(2, None, None),
        last_seen_stage=(0, None, None),
        slot_active=(False, None, None),
        slot_occupied=(True, None, None),
        reactivation=(True, None, None),
    )
    steps = (
        SimpleNamespace(
            stage_id=0,
            track_ids=(5, None, None),
            valid=(True, False, False),
            births=(True, False, False),
            rejected_births=(False, False, False),
            diagnostics=None,
        ),
        SimpleNamespace(
            stage_id=1,
            track_ids=(None, None, None),
            valid=(False, False, False),
            births=(False, False, False),
            rejected_births=(False, False, False),
            diagnostics=None,
        ),
        SimpleNamespace(
            stage_id=2,
            track_ids=(5, None, None),
            valid=(True, False, False),
            births=(False, False, False),
            rejected_births=(False, False, False),
            diagnostics=diagnostics,
        ),
    )

    events = build_association_events(
        payloads,
        steps,
        method="B4",
        reference_scene_id="ref",
        master_sequence_id="master",
        order_id="canonical",
        prefix=3,
        cache_digest="a" * 64,
        background_class=2,
    )

    assert len(events) == 2
    first, reactivated = events
    assert first.new_birth is True and first.is_failure is False
    assert reactivated.transition_opportunity is True
    assert reactivated.id_switch is False
    assert reactivated.gap_opportunity is True
    assert reactivated.reactivation_attempt is True
    assert reactivated.reactivation_correct is True
    assert reactivated.association_result == "reactivation_correct"
    assert reactivated.score_margin == pytest.approx(0.9)
    identity, reactivation = aggregate_event_metrics(events)
    assert identity["transition_opportunities"] == 1
    assert identity["id_switches"] == 0
    assert reactivation["gap_opportunities"] == 1
    assert reactivation["correct_reactivations"] == 1


def test_capacity_snapshots_are_derived_only_from_bounded_b4_state() -> None:
    payloads = [_payload(stage) for stage in range(2)]
    result = prefix_causality_coordinator(
        payloads,
        {
            "B4": lambda sequence_id: B4PersistentTracker(
                sequence_id=sequence_id, capacity=3
            )
        },
        endpoints=(1,),
        background_class=2,
    )

    snapshots = build_capacity_snapshots(result.online_steps["B4"][1], horizon=2)

    assert len(snapshots) == 2
    assert [snapshot.stage_id for snapshot in snapshots] == [0, 1]
    assert all(snapshot.capacity == 3 for snapshot in snapshots)
    assert all(snapshot.feature_dim == 2 for snapshot in snapshots)
    assert all(snapshot.class_count == 3 for snapshot in snapshots)
    assert len({snapshot.persistent_state_bytes for snapshot in snapshots}) == 1
    assert all(
        snapshot.dormant_count == snapshot.occupied_count - snapshot.active_count
        for snapshot in snapshots
    )


def test_official_metric_blocks_are_normalized_without_mixing_raw_and_temporal() -> (
    None
):
    blocks = {
        "raw": {
            "B4": {
                "T2": {
                    "raw_local_AP": 0.1,
                    "raw_local_AP50": 0.2,
                    "raw_local_AP25": 0.3,
                    "raw_local_REC": 0.4,
                    "raw_local_REC50": 0.5,
                    "raw_local_REC25": 0.6,
                }
            }
        },
        "strict": {
            "B4": {
                "T2": {
                    "online_t-mAP": 0.11,
                    "online_t-mAP50": 0.21,
                    "online_t-mAP25": 0.31,
                    "online_t-REC": 0.41,
                    "online_t-REC50": 0.51,
                    "online_t-REC25": 0.61,
                }
            }
        },
        "offline": {
            "Oracle": {
                "T2": {
                    "offline_reconstructed_t-mAP": 0.12,
                    "offline_reconstructed_t-mAP50": 0.22,
                    "offline_reconstructed_t-mAP25": 0.32,
                    "offline_reconstructed_t-REC": 0.42,
                    "offline_reconstructed_t-REC50": 0.52,
                    "offline_reconstructed_t-REC25": 0.62,
                }
            }
        },
    }

    normalized = normalize_official_metric_blocks(blocks)

    assert set(normalized["raw"]["B4"]["T2"]) == {
        "AP",
        "AP50",
        "AP25",
        "REC",
        "t_mAP",
        "t_mAP50",
        "t_mAP25",
        "t_REC",
        "t_REC50",
        "t_REC25",
    }
    assert normalized["raw"]["B4"]["T2"]["AP"] == pytest.approx(0.1)
    assert normalized["raw"]["B4"]["T2"]["t_mAP"] is None
    assert normalized["strict"]["B4"]["T2"]["AP"] is None
    assert normalized["strict"]["B4"]["T2"]["t_REC25"] == pytest.approx(0.61)
    assert normalized["offline"]["Oracle"]["T2"]["t_mAP"] == pytest.approx(0.12)


def test_association_event_errors_separate_semantic_drift_from_local_miss() -> None:
    semantic = _payload(0, gt_ids=(10,))
    semantic["target"]["gt_masks"][0] = semantic["observation"]["masks"][0]
    semantic["observation"]["class_prob"][0] = torch.tensor([0.8, 0.1, 0.1])
    step = SimpleNamespace(
        stage_id=0,
        track_ids=(5, None, None),
        valid=(True, False, False),
        births=(True, False, False),
        rejected_births=(False, False, False),
        diagnostics=None,
    )

    semantic_events = build_association_events(
        [semantic],
        [step],
        method="B4",
        reference_scene_id="ref",
        master_sequence_id="master",
        order_id="canonical",
        prefix=1,
        cache_digest="b" * 64,
        background_class=2,
    )

    assert {event.failure_category for event in semantic_events} == {"F6"}

    local_miss = _payload(0, gt_ids=(10,))
    local_miss["target"]["gt_masks"][0, 1] = True
    miss_events = build_association_events(
        [local_miss],
        [step],
        method="B4",
        reference_scene_id="ref",
        master_sequence_id="master",
        order_id="canonical",
        prefix=1,
        cache_digest="c" * 64,
        background_class=2,
    )
    gt_miss = next(event for event in miss_events if event.event_kind == "gt_miss")
    assert gt_miss.local_perception_miss is True
    assert gt_miss.failure_category == "F1"


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


def test_load_cached_protocol_sequences_reconstructs_exact_master_orders(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    provenance = _payload(0)["provenance"]

    def producer(key: dict[str, object]) -> dict[str, object]:
        payload = _payload(int(key["stage_index"]))
        payload["key"] = dict(key)
        return payload

    cache_directory = tmp_path / "entries"
    manifest_path = tmp_path / "cache_manifest.json"
    materialize_prediction_cache(
        protocol=protocol,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
        provenance=provenance,
        producer=producer,
    )

    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
    )

    assert len(sequences) == 3
    assert [sequence.order_id for sequence in sequences] == [
        "canonical",
        "reverse",
        "sha256_seed45",
    ]
    assert all(len(sequence.payloads) == 5 for sequence in sequences)
    assert sequences[1].payloads[1]["key"]["local_window_scan_ids"] == [
        "scene0000_04",
        "scene0000_03",
    ]


def test_load_cached_protocol_sequences_does_not_read_disallowed_master_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _protocol(2)
    provenance = _payload(0)["provenance"]

    def producer(key: dict[str, object]) -> dict[str, object]:
        payload = _payload(int(key["stage_index"]))
        payload["key"] = dict(key)
        return payload

    cache_directory = tmp_path / "entries"
    manifest_path = tmp_path / "cache_manifest.json"
    materialize_prediction_cache(
        protocol=protocol,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
        provenance=provenance,
        producer=producer,
    )
    original_validate = p6a_evaluator.validate_cache_entry
    validated_masters = []

    def record_validation(path, entry, *, expected_provenance):
        validated_masters.append(entry["key"]["master_sequence_id"])
        return original_validate(path, entry, expected_provenance=expected_provenance)

    monkeypatch.setattr(p6a_evaluator, "validate_cache_entry", record_validation)
    sequences = load_cached_protocol_sequences(
        protocol=protocol,
        cache_directory=cache_directory,
        manifest_path=manifest_path,
        allowed_master_sequence_ids=("master-0",),
    )

    assert len(sequences) == 3
    assert set(validated_masters) == {"master-0"}
    assert len(validated_masters) == 15


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
