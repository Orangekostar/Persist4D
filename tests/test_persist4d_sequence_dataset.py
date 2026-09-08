import random

import numpy as np
import pytest
import torch

from datasets.persist4d_sequence_dataset import (
    EpisodeMaster,
    Persist4DEpisodeBatch,
    Persist4DEpisodeCollator,
    Persist4DEpisodeDataset,
    Persist4DMixedEpisodeDataset,
    SequenceDatasetError,
    build_episode_draw_plan,
    build_episode_masters,
    build_mixed_source_schedule,
    build_single_scan_draw_plan,
    build_single_scan_masters,
)


def _masters() -> list[EpisodeMaster]:
    return [
        EpisodeMaster(
            reference_id="ref-a",
            sequence_id="a0-a1-a2-a3-a4",
            scan_indices=(0, 1, 2, 3, 4),
            role="adaptation",
            context_index=0,
        ),
        EpisodeMaster(
            reference_id="ref-a",
            sequence_id="a5-a6-a7-a8-a9",
            scan_indices=(5, 6, 7, 8, 9),
            role="adaptation",
            context_index=1,
        ),
        EpisodeMaster(
            reference_id="ref-b",
            sequence_id="b0-b1-b2-b3-b4",
            scan_indices=(10, 11, 12, 13, 14),
            role="adaptation",
            context_index=2,
        ),
        EpisodeMaster(
            reference_id="ref-dev",
            sequence_id="d0-d1-d2-d3-d4",
            scan_indices=(20, 21, 22, 23, 24),
            role="development",
            context_index=3,
        ),
    ]


def test_draw_plan_is_deterministic_balanced_and_role_isolated() -> None:
    first = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=16, seed=45
    )
    second = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=16, seed=45
    )

    assert first == second
    assert {spec.role for spec in first} == {"adaptation"}
    assert {spec.horizon for spec in first} == {2, 3, 4, 5}
    assert [sum(spec.horizon == horizon for spec in first) for horizon in range(2, 6)] == [
        4,
        4,
        4,
        4,
    ]
    assert {spec.reference_id for spec in first} == {"ref-a", "ref-b"}
    assert all(len(spec.scan_indices) == spec.horizon for spec in first)


def test_draw_plan_groups_horizons_for_two_ddp_ranks() -> None:
    plan = build_episode_draw_plan(
        _masters(),
        role="adaptation",
        episode_count=16,
        seed=45,
        replica_group_size=2,
    )

    assert all(
        plan[index].horizon == plan[index + 1].horizon
        for index in range(0, len(plan), 2)
    )
    assert [plan[index].horizon for index in range(0, len(plan), 2)] == [
        2,
        3,
        4,
        5,
        2,
        3,
        4,
        5,
    ]


def test_stage_inputs_are_t1_alone_then_only_adjacent_pairs() -> None:
    spec = build_episode_draw_plan(
        _masters(), role="development", episode_count=1, seed=45
    )[0]

    assert spec.horizon == 2
    assert spec.stage_scan_indices == (
        (20,),
        (20, 21),
    )


class _FakeBaseDataset:
    sequence_names = (
        "a0-a1-a2-a3-a4",
        "a5-a6-a7-a8-a9",
        "b0-b1-b2-b3-b4",
        "d0-d1-d2-d3-d4",
    )
    sequence_indices = (
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13, 14),
        (20, 21, 22, 23, 24),
    )
    ambiguities = ("a", "b", "c", "d")

    def __init__(self):
        self.calls = []

    def load_scan_indices(self, context_index, scan_indices, *, change_file):
        draws = (random.random(), float(np.random.random()), float(torch.rand(())))
        self.calls.append((context_index, tuple(scan_indices), change_file, draws))
        temporal = tuple(range(len(scan_indices)))
        return {"scan_indices": tuple(scan_indices), "temporal": temporal, "draws": draws}


def test_dataset_reuses_augmentation_draw_and_restores_global_rng() -> None:
    base = _FakeBaseDataset()
    spec = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=1, seed=45
    )[0]
    dataset = Persist4DEpisodeDataset(base, [spec])
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)

    episode = dataset[0]
    observed = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert episode.spec == spec
    assert episode.samples[0]["temporal"] == (0,)
    assert episode.samples[1]["temporal"] == (0, 1)
    assert base.calls[0][3] == base.calls[1][3]
    assert observed == expected
    assert all(call[2] is None for call in base.calls)


def test_augmentation_draw_does_not_depend_on_future_scan_coordinates() -> None:
    base = _FakeBaseDataset()
    original = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=4, seed=45
    )[3]
    shortened = type(original)(
        **{
            **original.__dict__,
            "horizon": 2,
            "scan_indices": original.scan_indices[:2],
        }
    )

    first = Persist4DEpisodeDataset(base, [original])[0].samples[0]
    second = Persist4DEpisodeDataset(base, [shortened])[0].samples[0]

    assert first == second


def test_full_history_mode_builds_only_arrived_prefixes() -> None:
    base = _FakeBaseDataset()
    spec = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=4, seed=45
    )[3]
    dataset = Persist4DEpisodeDataset(base, [spec], window_mode="full_history")

    episode = dataset[0]

    assert [sample["scan_indices"] for sample in episode.samples] == [
        spec.scan_indices[:stage] for stage in range(1, spec.horizon + 1)
    ]


def test_dataset_rejects_master_context_mismatch() -> None:
    base = _FakeBaseDataset()
    spec = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=1, seed=45
    )[0]
    base.sequence_names = ("wrong", *base.sequence_names[1:])

    with pytest.raises(SequenceDatasetError, match="context"):
        Persist4DEpisodeDataset(base, [spec])


def test_episode_collator_collates_each_stage_without_flattening_time() -> None:
    base = _FakeBaseDataset()
    plan = build_episode_draw_plan(
        _masters(),
        role="adaptation",
        episode_count=2,
        seed=45,
        replica_group_size=2,
    )
    dataset = Persist4DEpisodeDataset(base, plan)
    calls = []

    def stage_collator(samples):
        calls.append(tuple(sample["scan_indices"] for sample in samples))
        return tuple(samples)

    batch = Persist4DEpisodeCollator(stage_collator)([dataset[0], dataset[1]])

    assert isinstance(batch, Persist4DEpisodeBatch)
    assert [spec.horizon for spec in batch.specs] == [2, 2]
    assert len(batch.stage_batches) == 2
    assert calls == [
        tuple((spec.scan_indices[0],) for spec in plan),
        tuple(spec.scan_indices[:2] for spec in plan),
    ]


def test_episode_collator_rejects_mixed_horizons() -> None:
    base = _FakeBaseDataset()
    plan = build_episode_draw_plan(
        _masters(), role="adaptation", episode_count=2, seed=45
    )
    dataset = Persist4DEpisodeDataset(base, plan)

    with pytest.raises(SequenceDatasetError, match="same horizon"):
        Persist4DEpisodeCollator(tuple)([dataset[0], dataset[1]])


def test_single_scan_draw_plan_stays_h1_and_groups_for_ddp() -> None:
    masters = [
        EpisodeMaster(
            reference_id=f"scan-{index}",
            sequence_id=f"scan-{index}",
            scan_indices=(index,),
            role="adaptation",
            context_index=index,
        )
        for index in range(3)
    ]

    plan = build_single_scan_draw_plan(
        masters,
        role="adaptation",
        episode_count=6,
        seed=45,
        replica_group_size=2,
    )

    assert {spec.horizon for spec in plan} == {1}
    assert all(spec.stage_scan_indices == (spec.scan_indices,) for spec in plan)


def test_mixed_schedule_preserves_five_to_four_weight_and_rank_sync() -> None:
    schedule = build_mixed_source_schedule(
        group_count=18,
        replica_group_size=2,
        primary_weight=1.0,
        secondary_weight=0.8,
    )

    assert schedule.count("rio") == 20
    assert schedule.count("scannet") == 16
    assert all(
        schedule[index] == schedule[index + 1]
        for index in range(0, len(schedule), 2)
    )


def test_mixed_episode_dataset_consumes_each_source_plan_in_order() -> None:
    rio = ["r0", "r1", "r2", "r3"]
    scannet = ["s0", "s1"]
    mixed = Persist4DMixedEpisodeDataset(
        {"rio": rio, "scannet": scannet},
        ("rio", "rio", "scannet", "scannet", "rio", "rio"),
    )

    assert list(mixed) == ["r0", "r1", "s0", "s1", "r2", "r3"]


def test_real_master_mapping_binds_reference_role_and_context() -> None:
    protocol_masters = [
        type(
            "Master",
            (),
            {
                "reference_scene_id": "ref-a",
                "sequence_id": "a0-a1-a2-a3-a4",
                "scan_indices": (0, 1, 2, 3, 4),
            },
        )(),
        type(
            "Master",
            (),
            {
                "reference_scene_id": "ref-b",
                "sequence_id": "b0-b1-b2-b3-b4",
                "scan_indices": (10, 11, 12, 13, 14),
            },
        )(),
    ]

    masters = build_episode_masters(
        _FakeBaseDataset(),
        protocol_masters,
        {
            "a0-a1-a2-a3-a4": "adaptation",
            "b0-b1-b2-b3-b4": "development",
        },
    )

    assert masters == (
        EpisodeMaster(
            reference_id="ref-a",
            sequence_id="a0-a1-a2-a3-a4",
            scan_indices=(0, 1, 2, 3, 4),
            role="adaptation",
            context_index=0,
        ),
        EpisodeMaster(
            reference_id="ref-b",
            sequence_id="b0-b1-b2-b3-b4",
            scan_indices=(10, 11, 12, 13, 14),
            role="development",
            context_index=2,
        ),
    )


def test_single_scan_master_mapping_uses_only_real_dataset_entries() -> None:
    masters = build_single_scan_masters(_FakeBaseDataset(), role="adaptation")

    assert len(masters) == 4
    assert all(len(master.scan_indices) == 1 for master in masters)
    assert [master.context_index for master in masters] == [0, 1, 2, 3]
