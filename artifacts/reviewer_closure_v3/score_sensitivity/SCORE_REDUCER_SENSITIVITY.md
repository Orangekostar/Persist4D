# Score Reducer Sensitivity

## EV0 Status

**PASS.** Mean exactly regresses to frozen Persist4D-V2; latest/max
change trajectory confidence aggregation only. Direct local-current AP
is tracker/reducer invariant for every fixed official sidecar.

## All-Order t-mAP

| Horizon | Reducer | B2 | B3 | B4 | B4 - B2 |
|---:|---|---:|---:|---:|---:|
| T2 | mean | 20.727 | 20.727 | 20.724 | -0.003 |
| T2 | latest | 20.298 | 20.298 | 20.362 | +0.064 |
| T2 | max | 19.599 | 19.599 | 19.617 | +0.018 |
| T3 | mean | 10.231 | 10.295 | 12.310 | +2.079 |
| T3 | latest | 10.155 | 10.171 | 11.449 | +1.294 |
| T3 | max | 10.680 | 10.696 | 11.166 | +0.486 |
| T4 | mean | 4.529 | 4.467 | 7.023 | +2.494 |
| T4 | latest | 4.396 | 4.341 | 5.908 | +1.512 |
| T4 | max | 5.303 | 5.227 | 6.407 | +1.104 |
| T5 | mean | 1.823 | 1.897 | 5.250 | +3.427 |
| T5 | latest | 2.015 | 2.123 | 4.548 | +2.533 |
| T5 | max | 2.422 | 2.602 | 4.701 | +2.279 |

## Interpretation

At least one B4-vs-B2 sign changes relative to mean; mean remains primary and the temporal AP ranking claim is score-aggregation sensitive.
No reducer is selected based on its observed score.

## Invariants

- Mean regression maximum absolute difference: `0.0`.
- Score-only snapshot checks: `1935`.
- Local-current exact-invariance groups: `516`.
- Local masks, classes, official scores, target, and sidecar fingerprints
  are read before and independently of B2/B3/B4 linkage.
- `trajectory_current_slice_AP` remains a separate diagnostic from
  `local_current_AP`.
