# Capacity Sensitivity Report

## Decision

`CAPACITY_100_OK`

Keep the frozen main configuration at `K=100`. Across the exact frozen
common-prefix replay, increasing K above 100 changes state allocation but does
not change one accepted/rejected birth, track, task metric, or identity metric.
There is no empirical basis to reopen the configuration.

## Evidence Binding

- Architecture: `FINAL_LOCK`
- Baseline source commit: `3323cba186479b7dd4c005bebd468415b7d07a3b`
- Frozen observation cache manifest SHA-256:
  `44a089ae079b864a8417954fb0cc9b3e17f1650400e747c6d3c73c1ac71aaa0f`
- Cache entry SHA-256 list digest:
  `f03a6a99dd03c7995183ae23ae14a85e14d59e30b17a99cfa40336a4cdf4345d`
- Reviewer-closure manifest SHA-256:
  `11cc6d10d529eac45e8d946fefc4e7fdda2aa18a6d1e46b25f5d249ee2053e15`
- Dataset metric specification SHA-256:
  `fd66f9339b1ea44a0b6ea282317313c4dd7290711b0d1e027de59a6ba59a68ba`
- Label database SHA-256:
  `b03b15ecd0791a0ecd05912e9fe5617dd29a466d117cf5c2188f28638293a063`
- Coverage: 43 masters x 3 registered orders = 129 sequences, grouped into
  six independent physical-scene clusters; exact T2--T5 prefixes.
- Capacity grid: `{64, 100, 128, 160, 200}`; frozen main K: 100.
- Observation control: all K values replay the same per-sequence content
  digest. The timed replay and causal metric replay are checked for identical
  occupied/active state and accepted/rejected births.

The raw and aggregate evidence is in `artifacts/final_evidence/capacity/`.
`capacity_evaluation_manifest.json` binds every published result file by bytes
and SHA-256.

## Occupancy and Births

The occupied-slot distribution is identical for every tested K because the
smallest tested capacity is never reached.

| Horizon | Median peak occupied | IQR | Maximum | Birth opportunities | Rejected births |
| ---: | ---: | ---: | ---: | ---: | ---: |
| T2 | 12 | 10--16 | 28 | 1,762 | 0 |
| T3 | 14 | 11--17 | 30 | 1,938 | 0 |
| T4 | 15 | 13--19 | 30 | 2,115 | 0 |
| T5 | 16 | 13--20 | 30 | 2,214 | 0 |

All birth opportunities are accepted at every K and horizon. At T5, no
sequence reaches even K=64. For K=100, the maximum observed occupancy ratio is
0.30; the median is 0.16. The T5 peak active-slot median is 14 (IQR 11--18,
maximum 26), and the peak dormant-slot median is 7 (IQR 5--11, maximum 19).

The order-specific T5 evidence is also non-saturating:

| Order | Sequences | Birth opportunities | Mean peak occupied | Maximum | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 43 | 736 | 17.12 | 26 | 0 |
| reverse | 43 | 743 | 17.28 | 30 | 0 |
| sha256_seed45 | 43 | 735 | 17.09 | 25 | 0 |

The busiest physical-scene cluster reaches 30 occupied slots. No scene,
order, horizon, or predicted class has a true capacity rejection, so a
class-conditional rejection distribution does not exist.

## Task and Identity Sensitivity

Pooled official task metrics and pooled deployment identity counts are exactly
equal across all five K values. The T5 row is representative:

| K | t-mAP | t-mAP50 | t-mAP25 | t-REC | ID-switch rate | Gap recall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.04450 | 0.10772 | 0.16967 | 0.10665 | 0.11138 | 0.31199 |
| 100 | 0.04450 | 0.10772 | 0.16967 | 0.10665 | 0.11138 | 0.31199 |
| 128 | 0.04450 | 0.10772 | 0.16967 | 0.10665 | 0.11138 | 0.31199 |
| 160 | 0.04450 | 0.10772 | 0.16967 | 0.10665 | 0.11138 | 0.31199 |
| 200 | 0.04450 | 0.10772 | 0.16967 | 0.10665 | 0.11138 | 0.31199 |

The same equality holds at T2, T3, and T4 for t-mAP, t-mAP50, t-mAP25,
t-REC, t-REC50, t-REC25, fragmentation, merge, normalized ID switches,
gap-recovery accuracy, and gap-recovery recall. T2 gap metrics are undefined
because there is no eligible gap opportunity.

For each candidate K in `{128,160,200}`, every paired scene-cluster effect
relative to K=100 is exactly 0. The 10,000-replicate paired cluster bootstrap
therefore has `[0,0]` 95% intervals for every defined primary and
non-degradation cell. No candidate reaches the preregistered 0.01 primary
improvement threshold.

## State Storage and Timing

State storage is the exact sum of the eight allocated state tensors for B=1,
D=128, C=19. It is `610*K + 8` bytes.

| K | State bytes | State KiB | Mean memory-update latency at T5 (ms) | Mean total tracker-step latency at T5 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 39,048 | 38.1 | 0.728 | 20.480 |
| 100 | 61,008 | 59.6 | 0.745 | 19.075 |
| 128 | 78,088 | 76.3 | 0.760 | 19.183 |
| 160 | 97,608 | 95.3 | 0.779 | 19.297 |
| 200 | 122,008 | 119.1 | 0.801 | 19.483 |

The timing run used CPU, PyTorch 2.6.0+cu126, 32 PyTorch threads, and a shared
host. These measurements are descriptive. The memory-update timer excludes
association; total tracker-step time includes the complete B4 adapter step but
excludes network inference, cache loading, and metric postprocessing.

At K=100 the persistent historical tensor state is 61,008 bytes. The existing
system-comparison profile reports 58,989,016 bytes of explicit Full-History T5
input, but that is a different quantity, not a like-for-like state-memory or
VRAM measurement. Figure C3 therefore plots only the comparable persistent
state tensor allocation and does not manufacture a memory ratio.

## Required Questions

### Q1: Is K=100 frequently saturated?

No. Zero of 129 sequences saturate K=100 at T5, zero have a rejected birth,
and the global maximum is 30 occupied slots. Capacity failures are not
concentrated in a subset; there are no true capacity failures in this frozen
evaluation.

### Q2: Where do capacity failures occur?

They do not occur in any high-birth scene, predicted class, order, or horizon.
Birth opportunities rise from 1,762 at T2 to 2,214 at T5, but all are accepted.
Reverse order has the largest observed peak (30) and only a negligible increase
in mean peak occupancy over the other orders.

### Q3: Does larger K improve recall or gap recovery without damage?

No metric changes. Larger K neither improves t-REC/gap recovery nor damages
t-mAP/identity consistency. All defined scene-cluster effects and confidence
intervals are exactly zero.

### Q4: Does larger K accumulate false or noisy tracks?

No additional track is admitted because K=64 already has sufficient free
slots. Occupancy, accepted births, and all task/identity metrics are identical.
This result does not predict behavior beyond the observed five-capture
protocol or under a different local-observation distribution.

### Q5: How much state is required?

The frozen K=100 state uses 61,008 bytes (59.6 KiB). The tested state range is
38.1--119.1 KiB. K=64 is empirically sufficient on this final evaluation, but
it is not selected as a new configuration because this evaluation is not a
pre-existing development split and K=100 is already the frozen main setting.

## Legacy F7 Reconciliation

The immutable reviewer-closure table labels 830 T4 and 908 T5 events as
`capacity_failure`, operationally described as rejected births. Direct replay
shows that description is not valid for those counts:

1. `build_association_events` records an ordinary false birth with
   `capacity_failure=False` and `birth_rejected=False`
   (`scripts/evaluate_persist4d_p6a.py`, lines 1953--1974).
2. The shared taxonomy nevertheless maps the result string `false_birth` to F7
   (`scripts/p6a_analysis.py`, lines 762--770).
3. Reviewer decomposition maps every F7 event to `capacity_failure`
   (`scripts/reviewer_closure_decomposition.py`, lines 219--225).
4. A real rejected birth is separately emitted only from the explicit
   `rejected_births` tracker field (`scripts/evaluate_persist4d_p6a.py`, lines
   1985--2035).

Thus legacy F7 conflates false births with actual capacity rejections. The old
artifact remains immutable, but its capacity interpretation is superseded by
this direct slot-level audit. False births may remain a valid association error;
they are not evidence that K=100 is saturated.

## Limitations

- The evidence covers six physical scenes, 129 registered order sequences, and
  T2--T5 only. It does not establish sufficiency for arbitrarily long streams.
- There is no observed saturation point in the preregistered grid, so the study
  cannot estimate the causal penalty of a true capacity rejection.
- Capacity selection was not reopened; K=64 is a descriptive sensitivity result,
  not a newly tuned final configuration.
- CPU latency is host-dependent and secondary to the deterministic state/task
  evidence.

## Reproduction

```text
conda run -n persist4d python -m scripts.run_final_capacity \
  --cache-directory "$PERSIST4D_CACHE_ROOT/persistent_predictions/entries" \
  --dataset-spec "$PERSIST4D_RIO_ROOT/rio.yaml" \
  --label-database "$PERSIST4D_RIO_ROOT/label_database.yaml" \
  --output-directory artifacts/final_evidence/capacity-rerun

conda run -n persist4d python -m scripts.final_capacity_figures
```

The first command fails closed on source, reviewer manifest, cache manifest,
configuration, and cache-entry hashes. Existing result files are never
overwritten with different content.
