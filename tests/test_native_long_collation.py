import numpy as np
import torch


def test_mixed_horizon_batch_splits_full_inverse_by_full_counts():
    from datasets.pointcept_utils import VoxelizeCollate

    samples = []
    for h in (2, 5):
        xyz = np.tile(np.array([[0., 0., 0.], [.001, 0., 0.], [.05, 0., 0.]]), (h, 1))
        t = np.repeat(np.arange(h), 3)
        coords = np.column_stack((xyz, t))
        segments = np.repeat(np.arange(h), 3)
        labels = np.column_stack((np.full(3 * h, 2), np.ones(3 * h), np.zeros(3 * h), segments))
        samples.append((coords.copy(), np.ones((3 * h, 6)), labels.astype(np.int32),
                        f"h{h}", np.ones((3 * h, 3)), np.ones((3 * h, 3)),
                        coords.copy(), h, None))
    collate = VoxelizeCollate(voxel_size=.02, mode="validation", filter_out_classes=[0, 1, 255],
                             label_offset=2, preserve_empty_targets=True)
    data, targets, _ = collate(samples)
    assert [inverse.numel() for inverse in data.inverse_maps] == [6, 15]
    for i, h in enumerate((2, 5)):
        assert torch.equal(targets[i]["temporal_stages"][data.inverse_maps[i]],
                           torch.arange(h).repeat_interleave(3))
