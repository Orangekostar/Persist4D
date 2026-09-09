# Task-Memory Query and State Contract

Status: `PASS`

## Provenance

| Item | Value |
|---|---|
| Evaluation population | 47 canonical development masters from 8 references |
| Stages | 235 |
| R1 forward count | 235 shared forwards |
| Source commit | `f654aadce76630d88dc9e5ebd88a4089107e9da0` |
| R1 checkpoint SHA256 | `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd` |
| Base cache manifest SHA256 | `489577a52e462d43042e1945d4c27dc5a71b65aabb84deb6d93d7c3d75314851` |
| Observation manifest SHA256 | `8a398794052083d9727e71aed27e2459962faf85ef18a27ee859c00c52c6faa9` |
| Control configuration SHA256 | `08bfffc333854f90f470e0a0559f64954a107bffcf575cb494d3c142e531534d` |

The deployment path accepts predictions and unlabeled `StageMeta` only. Ground-truth
targets are loaded separately by the evaluator and are not serialized in the control
observation supplement, route, or persistent state.

## Prediction Observation

| Field | Shape | Meaning |
|---|---|---|
| `features` | `[B,Q,D]` float | Detached query embeddings |
| `class_prob` | `[B,Q,C]` float | Foreground class probabilities |
| `confidence` | `[B,Q]` float | Prediction confidence in `[0,1]` |
| `valid` | `[B,Q]` bool | Confidence and mask-support validity |
| `current_supported` | `[B,Q]` bool | Current-scan mask support |
| `previous_supported` | `[B,Q]` bool | Previous-window mask support |

A valid query must have current or previous support. Current visibility, previous
visibility, and persistent identity remain separate values.

## Read-Only Route

`route_entities(pre_output, old_state, stage_meta)` is read-only. It computes

`score = cosine(normalized_slot, normalized_query) + 0.25 * dot(slot_class, query_class)`

over occupied slots and valid queries. It runs the deterministic complete maximum-
weight assignment first and applies the fixed `0.5` threshold afterward. Stable ties
prefer assignments with smaller slot/query displacement. Rejected and unmatched
queries use slot and identity sentinel `-1` and route score `-inf`.

The immutable `EntityRoute` stores both inverse maps, accepted scores, inherited
logical IDs/generations, current/previous support, validity, and absolute stage
indices. SHA256 commitments bind it to all old-state tensors, update configuration,
and the label-free metadata identity. Routing does not update age, activity, time, or
state tensors.

## Single Commit

`commit_entities(final_output, route, old_state, stage_meta)` verifies all route,
state, metadata, shape, device, and stage commitments before writing. A route cannot
be committed twice because the state watermark must advance.

- Matched, valid, current-supported queries update their routed slot without a second
  association pass.
- Previous-only routed queries retain identity but do not refresh state or become
  active.
- Each occupied slot ages once; active flags are reset before current updates.
- Unmatched, valid, current-supported births are ordered by descending confidence,
  then ascending query index, and occupy ascending free slots.
- Capacity overflow is rejected. V1 does not evict or reuse slots; new logical IDs are
  monotonic and every accepted birth starts at generation zero.
- `D-LAST` replaces a reliable slot value (`alpha=1`). `D-EMA` uses fixed
  `alpha=0.2`. The default B4-compatible mode uses
  `alpha=clamp(0.2 * confidence, 0, 0.2)`.

## Persistent State

| Field | Shape | Dtype / invariant |
|---|---|---|
| `embedding` | `[B,K,D]` | float, normalized when occupied |
| `class_prob` | `[B,K,C]` | float |
| `confidence` | `[B,K]` | float |
| `occupied` | `[B,K]` | bool |
| `active` | `[B,K]` | bool, implies occupied and current watermark |
| `age` | `[B,K]` | int64 |
| `last_seen` | `[B,K]` | int64, `-1` only when unoccupied |
| `stage_watermark` | `[B]` | int64, strictly advances per commit |
| `logical_ids` | `[B,K]` | int64, unique when occupied, otherwise `-1` |
| `generations` | `[B,K]` | int64, present exactly when occupied |
| `next_logical_id` | `[B]` | int64, exceeds every issued ID |

For the measured `B=1, K=100, D=128, C=19, float32` configuration, permanent state
is 62,616 bytes, below the 2 MiB cap.

## Measured Controls

All rows use the same prediction observations, official candidates, W2 windows,
`K=100`, mean reducer, seed 45, and zero training updates. Full six-metric values are
stored in `baseline/long_memory_controls.csv`.

| Method | Policy | T2 t-mAP / t-REC | T3 | T4 | T5 |
|---|---|---:|---:|---:|---:|
| D-LAST | commit0 | .523079 / .594350 | .464077 / .536784 | .427783 / .495038 | .389650 / .458323 |
| D-LAST | lag1 | .529823 / .612966 | .485382 / .564107 | .439720 / .523675 | .407856 / .479322 |
| D-EMA | commit0 | .523079 / .594350 | .464085 / .536784 | .427829 / .495038 | .389711 / .458425 |
| D-EMA | lag1 | .529823 / .612966 | .485382 / .564107 | .439720 / .523675 | .407757 / .479424 |

The two update rules are nearly tied and neither dominates across the full metric
grid. Lag1 improves t-mAP, t-REC, mAP25, and direct current AP at every horizon. It
reduces mAP50 at T2 and very slightly at T4, so the output policy does not dominate
the complete metric grid. These are controls, not a learned-memory success claim.

## Router and Cache Diagnostics

| Method | Valid queries | Inherited | Coverage | Dormant | Births | Rejected | Peak slots | State bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D-LAST | 2,029 | 1,577 | .777230 | 101 | 449 | 0 | 21 | 62,616 |
| D-EMA | 2,029 | 1,577 | .777230 | 100 | 448 | 0 | 21 | 62,616 |

The prediction supplement occupies 3,430,928,601 bytes. Combined with the immutable
base cache, evaluation cache use is 7,748,592,299 bytes, below the 40 GiB cap.

Sparse CUDA inference is not bitwise repeatable across independent processes. Of 235
replayed stages, 139 exactly match the older base candidates. Both runs retain 100
candidates per stage; their mean common candidate count is 80.3915 (range 21-100),
mean aligned-mask IoU is .822454, and mean maximum aligned-score difference is
.321427. D-LAST and D-EMA are therefore compared only within their shared replay.
Numeric differences against an older B4 replay must not be described as matched
per-observation effects.
