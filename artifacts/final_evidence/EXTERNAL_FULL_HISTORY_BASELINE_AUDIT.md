# ReScan Full-History Baseline Audit

## Decision

`FULL_HISTORY_BASELINE_RUN_EXTERNAL_INCONCLUSIVE`

The executable conditional baseline in the external protocol was run with the
same frozen ReScene checkpoint. At stage `t`, its model input is exactly
`[S1,...,St]`, followed by the strongest preregistered simple tracker,
Pairwise Feature-Class Association. It is descriptive only because the
external evidence gate fails before method effects are interpreted.

## Runtime Binding

- Scenes/captures: 13/45, in dataset order.
- Checkpoint SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
- Dataset SHA256: `612810f1f8ec8ff8e4a1693b4b451da34ea0501cee46e1b34fedf3ed1910a420`.
- Config SHA256: `b7898eca6a912955b0cb03f749456090d86a29caf8885842e94d3a497aa8b4de`.
- Evaluator SHA256: `37577968402be7dd42126d13194d3ab305c3783b8fe62bd4a0fa2a266987395d`.
- Runtime base commit: `d3655142a9fc82896a787a1c1ca3f0b30ec61de1`;
  the exact evaluator source is stored at commit `b1d2cdb12db24b5d107af2d02f1b26476634098a`.
- Cache manifest SHA256: `ab14bd18c252c070087bbf6c9087ce9edb90847ed91a81e9129a00d4f6f47d04`.
- Cache: 45 unique content-hashed entries, 2,247,570,182 bytes, stored outside
  Git at `/mnt/shared/ww/persist4d-final-evidence/rescan/full-history-cache-v1`.

The one-scene smoke test passed before the formal run. Its four entry keys
expand from `[scene_a_0]` to `[scene_a_0,...,scene_a_3]`, and the formal
manifest verifies the same `stage_index + 1` history contract for all entries.
Inputs retain the no-GT-leakage contract.

## Descriptive Result

| Method | Level-A online t-mAP | Level-A online t-REC | Level-B coverage | IDSW rate | Fragmentation | Merge | Gap recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Persist4D, bounded local pair | 0.000204 | 0.000204 | 0.055738 | 0.089744 | 0.230769 | 0.230769 | 0 |
| Full-History + Feature-Class | 0.001682 | 0.007977 | 0.071881 | 0.173077 | 0.230769 | 0.230769 | 0 |

Full-History increases raw transfer coverage but remains below the registered
0.10 coverage minimum. It also has a higher normalized ID-switch rate, while
fragmentation and merge counts are equal. Both systems observe the same eight
natural gaps, make zero recovery attempts, and recover zero identities. These
numbers cannot support either a persistent-state advantage or a Full-History
advantage on ReScan.

Within the Full-History cache, all four registered trackers have zero B4-minus-
B2 effect for the eight primary reported metrics. The paired scene bootstrap
therefore has `[0,0]` intervals, but this degeneracy reflects insufficient
valid association evidence rather than equivalence.

## Paper Boundary

This run closes the executable Full-History requirement. It does not change the
primary classification `EXTERNAL_INCONCLUSIVE`, and it is not promoted to main
Table 4 or Figure 4. Machine-readable sources are the
`external/rescan_full_history_*` files; the 2.25 GB observation tensors remain
external and are bound by the committed cache manifest.
