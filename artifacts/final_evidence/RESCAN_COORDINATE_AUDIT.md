# ReScan Coordinate Audit

## Result

`RAW_COORDINATES_USABLE_WITHOUT_TRANSFORM`

All 32 adjacent capture pairs were audited in the dataset-provided order using
only raw XYZ geometry. The audit voxelized each capture at 5 cm and measured
bidirectional nearest-neighbor distances without estimating or applying a
rigid or non-rigid transform.

| Measure | Result |
|---|---:|
| Independent scenes | 13 |
| Adjacent capture pairs | 32 |
| Median of pair symmetric NN medians | 0.038161 m |
| Maximum pair symmetric NN median | 0.106742 m |
| Median overlap within 0.10 m | 0.726980 |
| Minimum overlap within 0.10 m | 0.487506 |

Every adjacent pair has substantial raw-coordinate overlap. The worst pair is
`scene_a_0` to `scene_a_1`; its symmetric median NN distance is 0.106742 m and
its overlap within 0.10 m is 0.487506. These results support using the packaged
XYZ values directly as a common scene frame for bounded local-pair inference.

No transform file exists in the package, no transform is fitted, and no GT
instance identity, semantic class, or object pose is used to align model input.
The full pair-level measurements are bound in
`external/rescan_coordinate_audit.json`.
