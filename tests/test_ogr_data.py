"""Causal windows, full-resolution weights and explicit confirmation scope."""

from dataclasses import replace

import pytest
import torch


def test_full_prefix_and_local_pair_have_distinct_causal_windows():
    from datasets.observation_grounded_episode import ObservationEpisodeSpec

    scans = tuple(f'00000000-0000-0000-0000-00000000000{i}' for i in range(1, 6))
    record = dict(input_id='example', reference_id=scans[0], role='CONFIRM',
                  dataset_scope='validation', scan_ids=scans, scan_indices=tuple(range(5)), horizon=5)
    pair = ObservationEpisodeSpec(**record)
    prefix = replace(pair, window_policy='full_prefix')
    assert pair.stage_scan_indices == ((0,), (0, 1), (1, 2), (2, 3), (3, 4))
    assert prefix.stage_scan_indices == ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 2, 3, 4))
    with pytest.raises(ValueError):
        ObservationEpisodeSpec.from_record({key: value for key, value in record.items() if key != 'dataset_scope'})
    with pytest.raises(ValueError):
        replace(pair, role='PB', dataset_scope='train')


def test_current_counts_use_full_vertices_instead_of_voxels():
    from models.observation_grounded import current_point_counts
    from tests.test_task_memory_routing import _meta

    meta = replace(_meta(1), local_stage_ids=torch.tensor([0, 1, 1, 1, 1]),
        original_vertex_ids=(torch.arange(1), torch.arange(4)), scan_vertex_offsets=torch.tensor([0, 1, 5]),
        voxel_inverse=torch.tensor([0, 1, 1, 1, 1]),
        full_resolution_point2segment=torch.tensor([0, 1, 1, 1, 1]))
    assert current_point_counts(meta, 2, device='cpu').tolist() == [0., 4.]
