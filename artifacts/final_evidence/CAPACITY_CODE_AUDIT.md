# Capacity Code Audit

## Audit Binding

- Architecture status: `FINAL_LOCK`
- Baseline source commit: `3323cba186479b7dd4c005bebd468415b7d07a3b`
- `models/persistent_memory.py` SHA-256:
  `111da9366ad873741cff5c9481c96f39a119b84b02e13c8fbedf7a2e32c8cbf8`
- `scripts/p6a_association.py` SHA-256:
  `135d196b98a29ed323202aefe49d7625ba6f1093c464d5b3f19da97b8e7b5842`
- `scripts/p6a_analysis.py` SHA-256:
  `83b459cc2ccdaf6898c1751e5c7cac5188303e0d8e9b4828e13e149189025b87`
- Result: measurement-only audit; no model or memory-state code changed.

## State Definition

`PersistentMemoryState` contains eight tensors (`models/persistent_memory.py`,
lines 248--258): embedding, class probability, confidence, occupied, active,
age, last seen, and the per-batch stage watermark.

For batch size B, capacity K, feature dimension D, and class count C, allocation
is (`models/persistent_memory.py`, lines 295--333):

| Field | Shape | Frozen dtype |
| --- | --- | --- |
| embedding | `[B,K,D]` | model floating dtype |
| class_prob | `[B,K,C]` | model floating dtype |
| confidence | `[B,K]` | model floating dtype |
| occupied | `[B,K]` | bool |
| active | `[B,K]` | bool |
| age | `[B,K]` | int64 |
| last_seen | `[B,K]` | int64 |
| stage_watermark | `[B]` | int64 |

Validation requires `active` to be a subset of `occupied` (lines 426--427), an
occupied slot to have a valid `last_seen` stage (lines 435--443), and an active
slot to have been observed at the current watermark (lines 444--449).

## Slot Lifecycle

At each step the implementation clones all state tensors, increments `age` for
occupied slots, and clears all `active` bits (lines 849--863). Association then
considers only occupied slots and valid observations (lines 674--680). Accepted
matches update embedding, class probability, and confidence by the frozen
confidence-scaled update rate, and mark the matched slots active at the current
stage (lines 879--931).

Every valid unmatched query is a birth attempt. Free slots are computed exactly
as `~occupied` (lines 933--938). Accepted births take the lowest available slots
in query order and set `occupied=True`, `active=True`, `age=0`, and
`last_seen=stage` (lines 939--957). Attempts exceeding the free-slot count are
marked in `rejected_births` (lines 958--961).

There is no statement anywhere in the transition that sets an occupied slot
back to false. There is no eviction, compression, replacement, lifecycle
timeout, or free-list mutation. Consequently:

- occupied count is monotonic non-decreasing within a sequence;
- dormant count is `occupied - active` after every step;
- once occupied count reaches K, every unmatched valid query is rejected;
- a rejected query does not enter persistent state on that step;
- K affects state allocation and birth admission, not the local observations.

The evaluator adapter delegates the sole transition to `PersistentMemory.step`
(`scripts/p6a_association.py`, lines 1183--1205 and 1248--1258). It does not
reimplement association or updates. Its `births` flag is true only when the
returned slot was unoccupied before the step (lines 1269--1285).

## Association Boundary

Association scores use normalized feature cosine plus `class_weight` times the
class-probability dot product (`models/persistent_memory.py`, lines 648--661).
Hungarian assignment is limited to occupied slots and valid queries; assigned
edges are accepted only at or above `association_threshold` (lines 674--695).
Capacity replay changes only the constructor value K. Class weight, association
threshold, update rate, maximum update rate, observation tensors, and stage
order remain fixed.

## Byte Accounting

The existing formula in `scripts/p6a_analysis.py`, lines 907--939, includes the
same eight state tensors and excludes masks, offline tracks, and evaluator
bookkeeping:

```text
per_slot = (D + C + 1) * float_bytes + 2 * bool_bytes + 2 * int64_bytes
state_bytes = B * (K * per_slot + int64_bytes)
```

For the frozen observations (`B=1`, `D=128`, `C=19`, float32), the measured
tensor storage exactly equals the formula:

| K | State bytes |
| ---: | ---: |
| 64 | 39,048 |
| 100 | 61,008 |
| 128 | 78,088 |
| 160 | 97,608 |
| 200 | 122,008 |

This is a measured linear allocation relation for the current tensor schema. It
does not include model weights, observation masks, allocator overhead, CUDA
context, or evaluator state.

## Timing Boundary

The opt-in timer starts before state cloning, brackets association separately,
and ends after next-state validation (`models/persistent_memory.py`, lines
849--874 and 981--994):

- `association_overhead_ms`: only `associate_observations`;
- `memory_update_overhead_ms`: cloning/lifecycle work plus matched updates,
  births, rejection bookkeeping, and state validation, excluding association;
- `memory_latency_ms`: the sum of those two recorded components;
- `total_update_latency_ms`: the evaluator wall time around the complete
  `B4PersistentTracker.step`, including adapter diagnostics.

All final timing results must name the device, environment, population, and
aggregation. The replay must not compare network-forward latency across K,
because network inference is deliberately reused and unchanged.

## Controlled Replay Requirements

The final runner must enforce all of the following:

1. exact capacity grid `{64,100,128,160,200}`;
2. one content digest for the frozen observation sequence across every K;
3. identical source/checkpoint/config/protocol/cache bindings;
4. contiguous stages and independent state initialization per sequence/K;
5. exact birth equation `attempts = accepted + rejected`;
6. `0 <= active <= occupied <= K` and `dormant = occupied - active`;
7. measured state bytes equal actual tensor storage;
8. finite non-negative timing values;
9. no change under `models/`, frozen configs, or reviewer-closure artifacts.

The initial unit fixture covering free slots, exact full capacity, rejected
births, state shape/storage, and cross-K observation-digest drift passes in
`tests/test_final_capacity.py`.
