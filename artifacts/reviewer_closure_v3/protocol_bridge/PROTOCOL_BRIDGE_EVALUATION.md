# Protocol Bridge Evaluation

## PB1 Status

**PASS.** Population, order, and horizon are reported as separate factors.
No additive decomposition from the paper-reported 34.8% result is made.

## Population Effect

The full-154 and exact-43 canonical T2 populations use the same frozen
checkpoint, batch-1 FP32 validation runtime, official ReScene post-processing, class
map, metric specification, min-region size, and evaluation seed.

| Seed | Full-154 t-mAP | Exact-43 t-mAP | Delta (43 - 154) |
|---:|---:|---:|---:|
| 45 | 27.675 | 20.348 | -7.327 |
| 46 | 26.607 | 19.515 | -7.091 |
| 47 | 28.035 | 20.069 | -7.966 |

The mean population delta is `-7.462` percentage points; the registered-seed range is `[-7.966, -7.091]`.
This is a population-bridge diagnostic, not an official benchmark score.

## Order Effect

Order uses the frozen Protocol-B V2 prediction/evaluator path on the same
43 masters. Pooled values are descriptive. Pairing is within master, and
all six `reference_scene_id` effects are retained. The 129 order-units are
not treated as independent; cluster bootstrap intervals are descriptive
robustness evidence only.

| Method | Comparison | Mean cluster effect | Six-cluster range | Bootstrap 95% |
|---|---|---:|---:|---:|
| FullHistory | reverse-minus-canonical | +0.103 | [-1.322, +1.242] | [-0.616, +0.819] |
| FullHistory | sha256_seed45-minus-canonical | -3.538 | [-14.370, +5.021] | [-9.344, +2.115] |
| Persist4D-V2 | reverse-minus-canonical | +0.464 | [-4.702, +3.743] | [-1.878, +2.333] |
| Persist4D-V2 | sha256_seed45-minus-canonical | -0.650 | [-13.354, +10.617] | [-6.961, +5.700] |

## Horizon Effect

Horizon uses the frozen exact-prefix Protocol-B T2-T5 results. Retention
is computed within each method/order as `t-mAP(T) / t-mAP(T2)`; absolute
cross-horizon pooled values are not treated as independent samples.
The `current_local_AP_calibration` column is the frozen V2 current-stage
calibration channel; V3 score closure separately audits direct latest-stage
official sidecars.

Horizon table rows: `24` (2 methods x 3 orders x 4 horizons).

## Runtime Boundary

The local 27.939% P2 result remains a frozen external-to-this-stage local
reference. PB1 reruns its own controlled seed groups only to compare the two
populations under one runtime. The paper-reported 34.8% result is not mixed
into these differences.
