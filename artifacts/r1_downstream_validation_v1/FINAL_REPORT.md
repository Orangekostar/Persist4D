# Persist4D R1 Downstream Validation Final Report

- Generation commit: `348adb21b908077baefa7da1058dd6ba8e2ce02c`
- R1 checkpoint: `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd` (completed epoch 390, step 25740)
- Protocol-B: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`; 43 masters, 3 orders, 129 sequence-order units, 6 reference-scene clusters
- Execution: `EXECUTION_COMPLETE`
- Recovery: `RECOVERY_SUPPORTED`
- Historical comparison: `FROZEN_SUMMARY_COMPARISON_ONLY`
- Publication: `RECOVERY_CLAIM_AUTHORIZED`

## Recovery Decision

| Horizon | B2 recall | B4 recall | B4-B2 | Positive clusters | Pass |
| --- | ---: | ---: | ---: | ---: | --- |
| T4 | 0.054598 | 0.295977 | +0.241379 | 6/6 | yes |
| T5 | 0.051235 | 0.310156 | +0.258920 | 6/6 | yes |

The decision is the frozen pooled-positive and at-least-four-of-six-positive-clusters rule at both T4 and T5.

## Task Evidence

Pooled official causal-prefix metrics across all three registered orders:

| Method | Reducer | Horizon | t-mAP | t-REC |
| --- | --- | --- | ---: | ---: |
| FullHistory | official | T2 | 0.228777 | 0.372957 |
| FullHistory | official | T4 | 0.102745 | 0.224228 |
| FullHistory | official | T5 | 0.081566 | 0.174699 |
| B2 | mean | T2 | 0.219017 | 0.359598 |
| B2 | mean | T4 | 0.040782 | 0.158269 |
| B2 | mean | T5 | 0.018288 | 0.110204 |
| B4 | mean | T2 | 0.219637 | 0.361011 |
| B4 | mean | T4 | 0.080159 | 0.201489 |
| B4 | mean | T5 | 0.058535 | 0.158670 |

Mean/latest/max reducer results, per-sequence tables, six-cluster tables, identity counts, event ledger, and seed-45 10,000-replicate paired bootstrap results are in `metrics/`.

## Resource Evidence

Latency is the median of the six per-unit medians. Peak allocation is the maximum observed unit peak.

| Method | Horizon | Latency (ms) | Peak allocated (MiB) |
| --- | --- | ---: | ---: |
| FullHistory | T2 | 628.456 | 2589.2 |
| FullHistory | T4 | 1025.030 | 3847.7 |
| FullHistory | T5 | 1098.975 | 4964.6 |
| B4 | T2 | 661.391 | 2555.5 |
| B4 | T4 | 607.123 | 2100.1 |
| B4 | T5 | 526.591 | 2471.1 |

The A40 profile uses the first canonical master per cluster, 5 warmups and 10 measured repeats. It includes model forward and B4 tracking, while excluding I/O, collation, H2D, metric scoring, and tracker preroll.

## Evidence Boundary

Historical raw replay is `HISTORICAL_REPLAY_UNAVAILABLE`. Frozen C-old summaries remain available for descriptive checkpoint-regime comparison, but they do not support causal attribution of old-versus-R1 changes.
