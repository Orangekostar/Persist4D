import hashlib
import json
from dataclasses import fields

import numpy as np
import pytest
import torch

import datasets
from datasets.task_memory_episode import (
    NativeEpisodeMaster,
    StageMeta,
    TaskMemoryEpisodeCollator,
    TaskMemoryEpisodeDataset,
    TaskMemoryEpisodeError,
    TaskMemoryEpisodeSpec,
    _renumber_segments,
    _segment_stage_ids,
    build_native_episode_masters,
    build_task_memory_draw_plan,
)
from scripts.preflight_task_memory_episode import (
    build_preflight_payload,
    load_reference_by_scene,
)


class _FakeBaseDataset:
    mode = "train"
    max_points_per_sample = None
    sequence_names = (
        "scene0001_00-scene0001_01-scene0001_02",
        "scene0001_00-scene0001_01-scene0001_03",
    )
    sequence_indices = ((0, 1, 2), (0, 1, 3))
    ambiguities = ([], [])

    def __init__(self, *, empty: bool = False) -> None:
        self.calls: list[tuple[int, tuple[int, ...], object, str]] = []
        self.empty = empty

    def load_scan_indices(self, context_index, scan_indices, *, change_file):
        assert len(scan_indices) == 1
        scan_index = scan_indices[0]
        self.calls.append((context_index, tuple(scan_indices), change_file, self.mode))
        coordinates = np.asarray(
            [
                [scan_index + 0.0, 0.0, 0.0, 0.0],
                [scan_index + 0.1, 0.0, 0.0, 0.0],
                [scan_index + 0.2, 1.0, 0.0, 0.0],
                [scan_index + 0.3, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        instance_id = -1 if self.empty else 10 + 10 * scan_index
        labels = np.column_stack(
            (
                np.full(4, 3, dtype=np.int32),
                np.full(4, instance_id, dtype=np.int32),
                np.zeros(4, dtype=np.int32),
                np.asarray([0, 0, 1, 1], dtype=np.int32),
            )
        )
        color = np.full((4, 3), 100 + scan_index, dtype=np.float32)
        normals = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (4, 1))
        return (
            coordinates.copy(),
            color.copy(),
            labels,
            self.sequence_names[context_index],
            color,
            normals,
            coordinates.copy(),
            context_index,
            [],
        )


def test_segment_remapping_uses_vectorized_inverse_without_changing_semantics(
    monkeypatch,
) -> None:
    labels = np.column_stack(
        (
            np.zeros(6, dtype=np.int32),
            np.zeros(6, dtype=np.int32),
            np.zeros(6, dtype=np.int32),
            np.asarray([10, 3, 10, 7, 3, 7], dtype=np.int32),
        )
    )
    original = labels.copy()
    numpy_unique = np.unique

    def require_inverse(*args, **kwargs):
        assert kwargs.get("return_inverse") is True
        return numpy_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", require_inverse)
    remapped, next_id = _renumber_segments(labels, 5)

    assert remapped[:, -1].tolist() == [7, 5, 7, 6, 5, 6]
    assert next_id == 8
    np.testing.assert_array_equal(labels, original)


def test_segment_stage_lookup_is_linear_and_rejects_invalid_groups(monkeypatch) -> None:
    def forbid_per_segment_unique(*args, **kwargs):
        raise AssertionError("per-segment unique is quadratic")

    monkeypatch.setattr(torch.Tensor, "unique", forbid_per_segment_unique)
    assert _segment_stage_ids(
        torch.tensor([0, 0, 1, 1, 2]),
        torch.tensor([0, 0, 1, 1, 1]),
    ).tolist() == [0, 1, 1]

    with pytest.raises(TaskMemoryEpisodeError, match="multiple stages"):
        _segment_stage_ids(torch.tensor([0, 0]), torch.tensor([0, 1]))
    with pytest.raises(TaskMemoryEpisodeError, match="segment IDs"):
        _segment_stage_ids(torch.tensor([0, 2]), torch.tensor([0, 0]))


def _master(
    *,
    reference: str = "ref-a",
    sequence: str = "scene0001_00-scene0001_01-scene0001_02",
    indices: tuple[int, ...] = (0, 1, 2),
    context: int = 0,
    role: str = "adaptation",
) -> NativeEpisodeMaster:
    return NativeEpisodeMaster(
        reference_id=reference,
        sequence_id=sequence,
        scan_ids=tuple(sequence.split("-")),
        scan_indices=indices,
        role=role,
        context_index=context,
    )


def _spec(master: NativeEpisodeMaster, *, draw_index: int = 0) -> TaskMemoryEpisodeSpec:
    return TaskMemoryEpisodeSpec.from_master(
        master,
        horizon=len(master.scan_ids),
        augmentation_seed=12345,
        draw_index=draw_index,
        bucket=f"T{len(master.scan_ids)}",
    )


def test_native_master_builder_accepts_real_h2_h3_h4_sequences() -> None:
    base = type(
        "Base",
        (),
        {
            "sequence_names": (
                "scene0001_00-scene0001_01",
                "scene0002_00-scene0002_01-scene0002_02",
                "scene0003_00-scene0003_01-scene0003_02-scene0003_03",
            ),
            "sequence_indices": ((0, 1), (2, 3, 4), (5, 6, 7, 8)),
        },
    )()

    masters = build_native_episode_masters(
        base,
        reference_by_scene={1: "ref-h2", 2: "ref-h3", 3: "ref-h4"},
        role_by_reference={
            "ref-h2": "adaptation",
            "ref-h3": "development",
            "ref-h4": "protocol_b_final",
        },
    )

    assert [len(master.scan_ids) for master in masters] == [2, 3, 4]
    assert [master.reference_id for master in masters] == ["ref-h2", "ref-h3", "ref-h4"]
    assert all(len(set(master.scan_ids)) == len(master.scan_ids) for master in masters)


def test_task_memory_episode_api_is_exported_by_dataset_package() -> None:
    assert "NativeEpisodeMaster" in datasets.__all__
    assert datasets.NativeEpisodeMaster is NativeEpisodeMaster


def test_native_master_builder_rejects_cross_room_episode() -> None:
    base = type(
        "Base",
        (),
        {
            "sequence_names": ("scene0001_00-scene0002_00",),
            "sequence_indices": ((0, 1),),
        },
    )()

    with pytest.raises(TaskMemoryEpisodeError, match="same scene"):
        build_native_episode_masters(
            base,
            reference_by_scene={1: "ref-a", 2: "ref-b"},
            role_by_reference={"ref-a": "adaptation", "ref-b": "adaptation"},
        )


def test_draw_plan_has_equal_buckets_and_rank_synchronous_horizons() -> None:
    masters = (
        _master(
            reference="ref-a",
            sequence="scene0001_00-scene0001_01-scene0001_02-scene0001_03-scene0001_04",
            indices=(0, 1, 2, 3, 4),
        ),
        _master(
            reference="ref-b",
            sequence="scene0002_00-scene0002_01-scene0002_02-scene0002_03-scene0002_04",
            indices=(5, 6, 7, 8, 9),
        ),
    )

    first = build_task_memory_draw_plan(
        masters, role="adaptation", episode_count=50, seed=45, replica_group_size=2
    )
    second = build_task_memory_draw_plan(
        masters, role="adaptation", episode_count=50, seed=45, replica_group_size=2
    )

    assert first == second
    assert [sum(spec.horizon == horizon for spec in first) for horizon in range(1, 6)] == [
        10,
        10,
        10,
        10,
        10,
    ]
    assert all(
        first[index].horizon == first[index + 1].horizon
        for index in range(0, len(first), 2)
    )
    assert all(len(spec.scan_ids) == spec.horizon for spec in first)
    assert all(len(set(spec.scan_ids)) == spec.horizon for spec in first)


def test_episode_loading_is_causal_and_reuses_scan_vertex_alignment() -> None:
    base = _FakeBaseDataset()
    first = TaskMemoryEpisodeDataset(base, (_spec(_master()),))[0]
    changed_future = _master(
        sequence="scene0001_00-scene0001_01-scene0001_03",
        indices=(0, 1, 3),
        context=1,
    )
    second = TaskMemoryEpisodeDataset(base, (_spec(changed_future),))[0]

    first_t1 = first.stage_samples[0]
    second_t1 = second.stage_samples[0]
    np.testing.assert_array_equal(first_t1.model_sample[0], second_t1.model_sample[0])
    np.testing.assert_array_equal(first_t1.model_sample[2], second_t1.model_sample[2])
    assert first_t1.augmentation_transform_id == second_t1.augmentation_transform_id
    assert first.spec.episode_id == second.spec.episode_id
    assert all(len(call[1]) == 1 and call[2] is None for call in base.calls)
    assert all(call[3] == "task_memory_raw" for call in base.calls)
    assert base.mode == "train"

    t1 = first.stage_samples[0]
    t2 = first.stage_samples[1]
    first_scan_points = t1.model_sample[0].shape[0]
    np.testing.assert_array_equal(t1.model_sample[0], t2.model_sample[0][:first_scan_points])
    np.testing.assert_array_equal(t1.model_sample[2], t2.model_sample[2][:first_scan_points])
    assert torch.equal(t1.original_vertex_ids[0], t2.original_vertex_ids[0])
    assert t2.scan_vertex_offsets.tolist() == [0, 4, 8]
    assert t2.local_stage_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_full_history_window_exposes_entire_causal_prefix() -> None:
    episode = TaskMemoryEpisodeDataset(
        _FakeBaseDataset(),
        (_spec(_master()),),
        window_mode="full_history",
    )[0]

    assert tuple(
        stage.scan_ids_in_window for stage in episode.stage_samples
    ) == (
        ("scene0001_00",),
        ("scene0001_00", "scene0001_01"),
        ("scene0001_00", "scene0001_01", "scene0001_02"),
    )
    assert episode.stage_samples[2].local_stage_ids.tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]


def test_episode_dataset_rejects_unknown_window_mode() -> None:
    with pytest.raises(TaskMemoryEpisodeError, match="window_mode"):
        TaskMemoryEpisodeDataset(
            _FakeBaseDataset(),
            (_spec(_master()),),
            window_mode="future_context",
        )


def test_episode_evaluation_mode_preserves_unaugmented_scan_values() -> None:
    base = _FakeBaseDataset()
    expected = base.load_scan_indices(0, (0,), change_file=None)
    base.calls.clear()

    episode = TaskMemoryEpisodeDataset(
        base,
        (_spec(_master()),),
        apply_augmentation=False,
    )[0]
    observed = episode.stage_samples[0].model_sample

    for index in (0, 1, 2, 4, 5, 6):
        np.testing.assert_array_equal(observed[index], expected[index])
    assert episode.stage_samples[0].augmentation_transform_id == "identity-v1"


def _pointcept_like_collator(samples):
    voxel_labels = []
    inverse_maps = []
    temporal_stages = []
    targets = []
    for sample in samples:
        labels = torch.from_numpy(sample[2]).long()
        stages = torch.from_numpy(sample[0][:, 3]).long()
        segment_ids = labels[:, -1]
        unique_segments = segment_ids.unique(sorted=True)
        representatives = torch.stack(
            [torch.nonzero(segment_ids == segment, as_tuple=False)[0, 0] for segment in unique_segments]
        )
        compressed_labels = labels[representatives]
        inverse = torch.searchsorted(unique_segments, segment_ids.contiguous())
        voxel_labels.append(compressed_labels)
        inverse_maps.append(inverse)
        temporal_stages.append(stages[representatives])
        instance_ids = labels[:, 1].unique(sorted=True)
        instance_ids = instance_ids[instance_ids >= 0]
        targets.append(
            {
                "ids": instance_ids,
                "point2segment": compressed_labels[:, -1],
            }
        )
    point = {
        "inverse_maps": inverse_maps,
        "labels": voxel_labels,
        "temporal_stages": temporal_stages,
    }
    return point, targets, [sample[3] for sample in samples]


def test_collator_builds_label_free_stage_meta_and_canonical_identity_keys() -> None:
    populated = TaskMemoryEpisodeDataset(_FakeBaseDataset(), (_spec(_master()),))[0]
    empty = TaskMemoryEpisodeDataset(
        _FakeBaseDataset(empty=True), (_spec(_master(), draw_index=1),)
    )[0]

    batch = TaskMemoryEpisodeCollator(_pointcept_like_collator)([populated, empty])
    stage_batch = batch.stage_batches[1]
    populated_meta, empty_meta = stage_batch.stage_meta

    assert isinstance(populated_meta, StageMeta)
    assert populated_meta.reference_id == "ref-a"
    assert populated_meta.scan_ids_in_window == ("scene0001_00", "scene0001_01")
    assert populated_meta.absolute_stage_index == 1
    assert populated_meta.voxel_inverse.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert populated_meta.full_resolution_point2segment.tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert populated_meta.segment_stage_ids.tolist() == [0, 0, 1, 1]
    assert stage_batch.training_identity_keys == (
        (("ref-a", 10), ("ref-a", 20)),
        (),
    )
    assert empty_meta.full_resolution_point2segment.numel() == 8
    assert all(
        forbidden not in field.name
        for field in fields(StageMeta)
        for forbidden in ("label", "target", "ground_truth", "instance_id")
    )


def test_collator_cannot_mutate_frozen_episode_samples() -> None:
    episode = TaskMemoryEpisodeDataset(_FakeBaseDataset(), (_spec(_master()),))[0]
    before_coordinates = episode.stage_samples[0].model_sample[0].copy()
    before_labels = episode.stage_samples[0].model_sample[2].copy()

    def mutating_collator(samples):
        result = _pointcept_like_collator(samples)
        samples[0][0][:, :3] = -999
        samples[0][2][:, -1] = 999
        return result

    TaskMemoryEpisodeCollator(mutating_collator)([episode])

    np.testing.assert_array_equal(
        episode.stage_samples[0].model_sample[0], before_coordinates
    )
    np.testing.assert_array_equal(episode.stage_samples[0].model_sample[2], before_labels)


def test_preflight_payload_proves_realized_prefix_and_inverse_invariance() -> None:
    base = _FakeBaseDataset()
    original = TaskMemoryEpisodeDataset(base, (_spec(_master()),))[0]
    changed_future = TaskMemoryEpisodeDataset(
        base,
        (
            _spec(
                _master(
                    sequence="scene0001_00-scene0001_01-scene0001_03",
                    indices=(0, 1, 3),
                    context=1,
                )
            ),
        ),
    )[0]
    batch = TaskMemoryEpisodeCollator(_pointcept_like_collator)([original])

    payload = build_preflight_payload(
        original=original,
        future_mutated=changed_future,
        batch=batch,
        native_summary={"H2": {"masters": 2}, "H3": {"masters": 2}},
        sources={"data_contract": "repo:DATA_CONTRACT.json"},
    )

    assert payload["status"] == "PASS"
    assert payload["future_mutation"]["episode_id_unchanged"] is True
    assert payload["future_mutation"]["t1_input_unchanged"] is True
    assert payload["repeated_scan"]["vertex_ids_unchanged"] is True
    assert payload["repeated_scan"]["labels_unchanged"] is True
    assert payload["collation"]["inverse_alignment"] is True
    unsigned = dict(payload)
    observed = unsigned.pop("content_sha256")
    expected = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    ).hexdigest()
    assert observed == expected

    changed_seed = TaskMemoryEpisodeSpec.from_master(
        _master(),
        horizon=3,
        augmentation_seed=999,
        draw_index=0,
        bucket="T3",
    )
    noncausal = TaskMemoryEpisodeDataset(base, (changed_seed,))[0]
    with pytest.raises(TaskMemoryEpisodeError, match="future mutation changed T1"):
        build_preflight_payload(
            original=original,
            future_mutated=noncausal,
            batch=batch,
            native_summary={"H2": {"masters": 2}},
            sources={"data_contract": "repo:DATA_CONTRACT.json"},
        )


def test_preflight_loads_list_shaped_3rscan_metadata(tmp_path) -> None:
    metadata = tmp_path / "3RScan.json"
    metadata.write_text(
        '[{"reference":"ref-a"},{"reference":"ref-b","scene":4}]\n',
        encoding="utf-8",
    )

    assert load_reference_by_scene(metadata) == {0: "ref-a", 4: "ref-b"}
