# ReScene Data Semantics

Status: `PASS`

## Database And Mix

Unmodified/current RIO train counts are `1178` / `1174` (the all-split T2 map has `1482` records); ScanNet counts are `1201` / `1199`. The active sampler draws `2112` examples.

Observed one-epoch draws are `{'rio': 1195, 'scannet': 917}` with `1387` unique concatenated indices and replacement duplicate rate `0.343276515152`.

## Label 255

The active database contains `34731` label-255 target instances and `62688083` points, representing `0.554888082951` of target instances and `0.394752264001` of supervised target points.

R4 materiality is `true` under the fixed 0.5% instance-or-point threshold.

Without filtering 255, target label 253 is ignored by classification and receives the matcher ignore sentinel, but its mask/dice target and mask matcher costs remain included. Filtering 255 removes that target before all criterion paths.
