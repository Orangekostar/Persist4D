# ReScan Identity And Gap Event Spot Check

## Binding And Method

- Dataset content SHA256:
  `612810f1f8ec8ff8e4a1693b4b451da34ea0501cee46e1b34fedf3ed1910a420`.
- Protocol: `external/rescan_protocol.json`, dataset-provided numeric capture
  order, no artificial gaps or permutations.
- Inspection source: raw `instance_idx` and `class_idx` arrays read from the
  official binary PLY files with `datasets.rescan_adapter.read_rescan_ply`.

This is a post-inference ground-truth audit. These fields are not passed to the
model or tracker. An identity is visible when its raw point count is positive
and absent when the count is zero.

## Natural Gap Events

| Scene | Identity | Left capture: points | Absent capture: points | Right capture: points | Left/right class |
| --- | ---: | --- | --- | --- | --- |
| `scene_a` | 19 | `scene_a_0`: 2,902 | `scene_a_1`: 0 | `scene_a_2`: 1,883 | 5 / 5 |
| `scene_a` | 31 | `scene_a_0`: 1,095 | `scene_a_1`: 0 | `scene_a_2`: 966 | 39 / 39 |
| `scene_b` | 8 | `scene_b_1`: 9,495 | `scene_b_2`: 0 | `scene_b_3`: 10,165 | 39 / 39 |
| `scene_b` | 9 | `scene_b_1`: 2,377 | `scene_b_2`: 0 | `scene_b_3`: 3,668 | 7 / 7 |
| `scene_b` | 12 | `scene_b_0`: 2,779 | `scene_b_1`: 0 | `scene_b_2`: 2,714 | 5 / 5 |
| `scene_b` | 24 | `scene_b_0`: 3,062 | `scene_b_1`: 0 | `scene_b_2`: 3,762 | 5 / 5 |
| `scene_b` | 25 | `scene_b_0`: 1,595 | `scene_b_1`: 0 | `scene_b_2`: 2,125 | 7 / 7 |
| `scene_b` | 28 | `scene_b_0`: 11,080 | `scene_b_1`: 0 | `scene_b_2`: 8,753 | 15 / 15 |

All eight protocol events reproduce the required visible-absent-visible
pattern. The same scene-scoped `instance_idx` appears on both sides, and its
class is consistent across the inspected endpoints. The events occur in
exactly two independent physical scenes, matching the dataset manifest and
external gate.

## Ambiguity Semantics Spot Check

The official `scene_a_2.txt` file contains accepted alternatives such as
`4 | 4 7`, `5 | 4 6 5`, and `7 | 4 7 11`. The dataset manifest parses these as
ordered accepted-ID sets `[4,7]`, `[4,6,5]`, and `[4,7,11]`, respectively.
This matches the official evaluator's best-accepted-correspondence semantics;
the alternatives are applied only after inference.

## Result

`EVENT_SPOT_CHECK_PASS`: 8/8 natural gaps and the sampled ambiguity groups
match the committed protocol. No event was manufactured or selected from model
successes.
