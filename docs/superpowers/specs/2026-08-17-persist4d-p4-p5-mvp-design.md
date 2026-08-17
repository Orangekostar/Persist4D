# Persist4D P4-P5 MVP Design

## 1. Purpose

The MVP establishes the first executable Persist4D method path. It does not try
to recover the paper-reported ReScene4D score. The completed ReScene4D-C T=2
checkpoint is frozen as an internal comparison point.

The method claim tested by this MVP is:

> A fixed-capacity persistent instance state can consume short-window
> ReScene4D observations sequentially, preserve identities beyond the local
> window, and keep state memory independent of history length.

P4 exposes the local query representation. P5 implements a minimal recurrent
memory around that representation. Dual-timescale consolidation, learned
association losses, and memory-conditioned prediction remain later stages.

## 2. Scope

### Included

- Backward-compatible export of final normalized ReScene query features.
- A typed local-observation contract.
- A fixed-capacity persistent memory state with explicit lifecycle fields.
- One-to-one query-to-slot association.
- Birth, active, dormant, and free transitions.
- A streaming wrapper that receives and returns state explicitly.
- Direct execution over T=2/3/4/5 validation sequences.
- Correctness, bounded-state, determinism, and baseline-compatibility tests.

### Excluded

- Changes to Concerto or the trained ReScene checkpoint.
- Full-history point-cloud retention.
- Dual anchor/working banks.
- Learned GRU consolidation or new identity losses.
- Memory-conditioned query refinement.
- Open-vocabulary, TSDF, robotics, or explicit occlusion reasoning.
- Any claim that the method matches the paper-reported absolute AP.

## 3. Architecture

The implementation has four boundaries:

1. `ReScene` optionally exports final query features.
2. An observation builder converts one short-window output into current-stage
   instance observations.
3. `PersistentMemory` associates observations with fixed slots and returns a
   new immutable-by-contract state value.
4. `StreamingReScene` orchestrates adjacent T=2 windows without retaining old
   point clouds or model graphs.

The base model remains the only producer of masks and classes. P5 adds
persistent slot identities but does not alter segmentation predictions.

## 4. P4 Query Feature Export

### Configuration

`conf/model/rescene.yaml` gains:

```yaml
return_query_features: false
```

The constructor stores the flag. Existing configurations resolve to `false`.

### Output semantics

When enabled, `ReScene.forward()` adds:

```python
output_dict["query_features"]  # Tensor[B, Q, D]
```

The tensor is `decoder_norm(queries)` from the final decoder state. This is the
same normalized representation consumed by the final class and mask heads.
It is not detached, so a later trainable memory path can propagate gradients.

The existing final predictions are computed first through the unchanged
`mask_module` path. Feature export performs a separate normalization only when
requested, avoiding numerical changes to default predictions.

### Compatibility contract

- With the flag disabled, output keys and tensor values are unchanged.
- With the flag enabled, the only additional key is `query_features`.
- Loading the existing checkpoint remains strict-compatible.

## 5. Local Observation Contract

Create `LocalInstanceObservation` with batched tensors:

```python
@dataclass(frozen=True)
class LocalInstanceObservation:
    features: Tensor       # [B, Q, D]
    class_prob: Tensor     # [B, Q, C]
    confidence: Tensor     # [B, Q]
    latest_mask: list[Tensor]
    valid: Tensor          # [B, Q] bool
```

`latest_mask[b]` has shape `[Q, S_latest]`. The builder derives it by selecting
the newest temporal-stage segments from `pred_masks`. Temporal stage indices
are input structure, not semantic or instance ground truth. GT instance IDs may
be supplied to a separate evaluator but never to inference association.

Confidence is the maximum non-background class probability. A query is valid
only when confidence meets the configured threshold and its latest-stage mask
contains the configured minimum support.

The main Local-2 path processes adjacent scan pairs `[X_(t-1), X_t]` and keeps
only the observation for `X_t`. This preserves ReScene short-window perception
while preventing all-history input growth.

## 6. Persistent State

`PersistentMemoryState` is a batched dataclass:

```python
@dataclass(frozen=True)
class PersistentMemoryState:
    embedding: Tensor      # [B, K, D]
    class_prob: Tensor     # [B, K, C]
    confidence: Tensor     # [B, K]
    occupied: Tensor       # [B, K] bool
    active: Tensor         # [B, K] bool
    age: Tensor            # [B, K] int64
    last_seen: Tensor      # [B, K] int64
```

Initial values use `K=100`, matching the query count. State construction is
device- and dtype-aware. The state exposes `detach()` for truncated recurrent
execution without changing values.

No field grows with the number of processed stages.

## 7. Association And Update

The P5 association score is deliberately minimal:

```text
score = cosine(query_feature, slot_embedding)
        + class_weight * class_compatibility
```

Absolute 3D position is not a hard gate because objects can move between
rescans. Hungarian assignment provides one-to-one matches. A pair is accepted
only when its score meets `association_threshold`.

State transitions per stage are deterministic:

- Accepted match: slot becomes active and updates from the observation.
- Unmatched occupied slot: slot becomes dormant and remains occupied.
- Valid unmatched observation: allocate the lowest-index free slot.
- No free slot: reject the birth; P5 performs no implicit eviction.
- Invalid observation: never updates or allocates memory.

Matched embeddings use a configurable confidence-weighted normalized EMA:

```text
rate = clamp(update_rate * observation_confidence, 0, max_update_rate)
new = normalize((1 - rate) * old + rate * observation)
```

Class probabilities and confidence use the same bounded update rate. New slots
copy the normalized observation and start with age zero. Every processed stage
increments occupied-slot age; matched and born slots set `last_seen` to the
current stage index.

The step result includes `slot_ids[B,Q]`, using `-1` for invalid, rejected, or
unmatched observations. This is the persistent identity exposed to evaluation.

## 8. Streaming Wrapper And Data Flow

`StreamingReScene.forward_step()` accepts a short-window batch, optional prior
state, and integer stage index:

```python
result, next_state = model.forward_step(batch, state, stage_index)
```

Data flow:

```text
adjacent T=2 scans
  -> unchanged ReScene forward with query export
  -> latest-stage observation builder
  -> persistent association and update
  -> original predictions + persistent slot IDs
  -> next fixed-capacity state
```

State is never stored implicitly on the module. Callers reset it at each scene
boundary. Batch-size changes require an explicit reset and otherwise raise a
clear error. Non-finite features, invalid shapes, duplicate accepted slot IDs,
or decreasing stage indices fail closed before state mutation.

For a T-stage validation record, the evaluator constructs adjacent windows
`(0,1), (1,2), ..., (T-2,T-1)` from the official preprocessed scans. It releases
each local batch after the step and retains only `PersistentMemoryState` plus
metric bookkeeping.

## 9. Files And Ownership

Expected production files:

```text
models/rescene.py                 optional query export
models/persistent_memory.py       observation, state, association, update
models/streaming_rescene.py       base-model orchestration
conf/model/rescene.yaml           default-off export switch
conf/model/persist4d.yaml         P5 method configuration
scripts/evaluate_persist4d.py     T=2/3/4/5 streaming evaluation
```

Expected tests:

```text
tests/test_rescene_query_features.py
tests/test_persistent_memory.py
tests/test_memory_association.py
tests/test_streaming_sequence.py
```

The implementation avoids changes to the existing trainer until inference and
state semantics pass all P5 gates.

## 10. Verification Gates

### P4 gate

- Disabled mode has identical output keys and tensor values for a fixed input.
- Enabled mode adds exactly one `[B,100,128]` query tensor.
- Existing checkpoint loads without missing or unexpected keys.
- Exported features are finite and remain connected to autograd.

### P5 engineering gate

- Association is one-to-one and deterministic under tied-score rules.
- Birth, active, dormant, reactivation, full-capacity rejection, reset, and
  detach behavior have focused tests.
- State tensor shapes remain constant across at least 100 synthetic stages.
- Scene boundaries cannot leak state.
- T=2/3/4/5 real validation samples execute without retaining full history.
- Default ReScene predictions remain unchanged.

### P5 research evidence gate

The fixed current checkpoint is the comparison baseline. Report for each
T=2/3/4/5:

- t-mAP, t-REC, and per-stage AP under the existing evaluator;
- persistent identity switches and reactivation accuracy where GT IDs exist;
- peak allocated VRAM, latency, throughput, and state bytes;
- failures caused by no free slot, rejected birth, and incorrect association.

P5 passes as a method prototype when it is executable and bounded, and its
persistent IDs improve at least one long-horizon identity metric at T=4 or T=5
without degrading T=2 segmentation predictions. This is an internal relative
comparison; the paper's absolute score is not a gate.

## 11. Follow-On Decision

If P5 passes, proceed in order:

1. P7 replaces the single embedding with slow anchor and fast working states.
2. P8 adds learned confidence-gated consolidation.
3. P9 adds a zero-initialized memory query adapter so memory influences current
   predictions while preserving baseline behavior at initialization.

If P5 is bounded but does not improve identity metrics, diagnose association
errors before adding more memory capacity or losses. If P5 cannot remain
bounded, stop and fix the streaming boundary before further method work.
