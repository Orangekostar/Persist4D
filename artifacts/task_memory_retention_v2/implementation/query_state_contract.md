# Task-Memory Query and State Contract

Status: `PASS`

## Provenance

| Item | Value |
|---|---|
| Evaluation population | 47 canonical development masters from 8 references |
| Stages | 235 |
| R1 forward count | 235 shared forwards |
| Source commit | `a5d06b35129f673023ec667745f244854a2f130d` |
| R1 checkpoint SHA256 | `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd` |
| Base cache manifest SHA256 | `489577a52e462d43042e1945d4c27dc5a71b65aabb84deb6d93d7c3d75314851` |
| Observation manifest SHA256 | `517cc0a45977827bab3324f6c40f914bb94f66cecbbdfa068823843f1c19c482` |
| Control configuration SHA256 | `802950c3ad5d45bf36c67ed80267033d04d16f71f8b55557732dc377076655bb` |

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
| D-LAST | commit0 | .528284 / .598535 | .471122 / .540660 | .432058 / .496431 | .397051 / .461815 |
| D-LAST | lag1 | .529546 / .612759 | .480408 / .562524 | .433891 / .519463 | .401098 / .476856 |
| D-EMA | commit0 | .528284 / .598535 | .471632 / .541311 | .430812 / .495609 | .396180 / .460832 |
| D-EMA | lag1 | .529546 / .612759 | .480408 / .562524 | .433585 / .519463 | .400927 / .475631 |

The two update rules are nearly tied and neither dominates across the full metric
grid. Lag1 consistently improves t-REC and slightly improves t-mAP, while reducing
mAP50/mAP25 at several horizons. These are controls, not a learned-memory success
claim.

## Router and Cache Diagnostics

| Method | Valid queries | Inherited | Coverage | Births | Rejected | Peak slots | State bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-LAST | 1,924 | 1,489 | .773909 | 435 | 0 | 19 | 62,616 |
| D-EMA | 1,924 | 1,492 | .775468 | 432 | 0 | 18 | 62,616 |

The prediction supplement occupies 3,430,928,601 bytes. Combined with the immutable
base cache, evaluation cache use is 7,748,592,299 bytes, below the 40 GiB cap.

Sparse CUDA inference is not bitwise repeatable across independent processes. Of 235
replayed stages, 136 exactly match the older base candidates. Both runs retain 100
candidates per stage; their mean common candidate count is 80.1106 (range 21-100),
mean aligned-mask IoU is .819894, and mean maximum aligned-score difference is
.329907. D-LAST and D-EMA are therefore compared only within their shared replay.
Numeric differences against an older B4 replay must not be described as matched
per-observation effects.
