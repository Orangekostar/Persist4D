from __future__ import annotations

from dataclasses import dataclass

import pytest

from utils.rescene_runtime_audit import (
    analyze_ddp_rank_streams,
    serialize_sampler_chain,
)


@dataclass
class _Sampler:
    sampler: object | None = None
    rank: int | None = None
    num_replicas: int | None = None


def test_sampler_chain_serializes_wrappers_rank_and_world_size() -> None:
    chain = serialize_sampler_chain(
        _Sampler(_Sampler(_Sampler()), rank=1, num_replicas=2)
    )

    assert len(chain) == 3
    assert chain[0]["class"].endswith("._Sampler")
    assert chain[0]["rank"] == 1
    assert chain[0]["world_size"] == 2


def test_rank_streams_are_compared_to_expected_global_shards() -> None:
    global_draws = [0, 1, 2, 0, 3, 1, 4, 5]
    ranks = {0: global_draws[0::2], 1: global_draws[1::2]}

    result = analyze_ddp_rank_streams(
        global_draws=global_draws,
        rank_draws=ranks,
        world_size=2,
        minimum_draws_per_rank=4,
    )

    assert result["correctly_sharded"] is True
    assert result["positional_mismatch_count"] == 0
    assert result["cross_rank_value_overlap_count"] == 1
    assert result["cross_rank_value_overlap_is_sampler_bug"] is False
    assert result["replacement_duplicate_count"] == 2


def test_rank_stream_contract_rejects_positional_duplication() -> None:
    global_draws = list(range(8))
    duplicated = {0: global_draws[0::2], 1: global_draws[0::2]}

    result = analyze_ddp_rank_streams(
        global_draws=global_draws,
        rank_draws=duplicated,
        world_size=2,
        minimum_draws_per_rank=4,
    )

    assert result["correctly_sharded"] is False
    assert result["positional_mismatch_count"] == 4
    assert result["gate"] == "sampler_fix_required"


def test_rank_stream_analysis_accepts_lightning_position_shards() -> None:
    global_draws = [10, 11, 12, 13, 14, 15, 16, 17]
    positions = {0: [6, 2, 0, 4], 1: [3, 1, 7, 5]}
    rank_draws = {
        rank: [global_draws[position] for position in rank_positions]
        for rank, rank_positions in positions.items()
    }

    result = analyze_ddp_rank_streams(
        global_draws=global_draws,
        rank_draws=rank_draws,
        rank_positions=positions,
        world_size=2,
        minimum_draws_per_rank=4,
    )

    assert result["correctly_sharded"] is True
    assert result["positional_mismatch_count"] == 0


def test_rank_stream_contract_requires_all_ranks_and_minimum_draws() -> None:
    with pytest.raises(ValueError, match="rank coverage"):
        analyze_ddp_rank_streams(
            global_draws=list(range(512)),
            rank_draws={0: list(range(256))},
            world_size=2,
            minimum_draws_per_rank=256,
        )
    with pytest.raises(ValueError, match="256"):
        analyze_ddp_rank_streams(
            global_draws=list(range(510)),
            rank_draws={0: list(range(255)), 1: list(range(255))},
            world_size=2,
            minimum_draws_per_rank=256,
        )
