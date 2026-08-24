from __future__ import annotations

import numpy as np

from scripts.audit_rescan_coordinates import coordinate_pair_metrics


def test_coordinate_pair_metrics_distinguish_shared_and_shifted_frames() -> None:
    first = np.asarray(
        [[x, y, 0.0] for x in range(4) for y in range(4)], dtype=np.float32
    )
    shared = first + np.asarray([0.01, 0.0, 0.0], dtype=np.float32)
    shifted = first + np.asarray([0.5, 0.0, 0.0], dtype=np.float32)

    aligned = coordinate_pair_metrics(first, shared, voxel_size_m=0.005)
    misaligned = coordinate_pair_metrics(first, shifted, voxel_size_m=0.005)

    assert aligned["symmetric_median_nn_m"] < 0.02
    assert aligned["symmetric_overlap_fraction_at_10cm"] == 1.0
    assert misaligned["symmetric_median_nn_m"] > aligned["symmetric_median_nn_m"]
