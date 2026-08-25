# Identity Recomputation

## ID0 Status

**PASS.** Fresh B4 query-level identity diagnostics regress exactly to
the frozen V1 B4 rows. B2/B3/B4 were rerun from the same frozen V1 raw
observations; no V1 identity value is copied into a V3 result row.

## All-Order Pooled Diagnostics

| T | Tracker | ID-switch rate | Gap recovery recall | Gap recovery accuracy | False births |
|---:|---|---:|---:|---:|---:|
| T2 | B2 | 10.799 | NA | NA | 590 |
| T2 | B3 | 10.799 | NA | NA | 590 |
| T2 | B4 | 10.151 | NA | NA | 601 |
| T3 | B2 | 11.111 | 11.029 | 33.708 | 806 |
| T3 | B3 | 12.046 | 12.132 | 37.079 | 796 |
| T3 | B4 | 9.969 | 29.412 | 89.888 | 706 |
| T4 | B2 | 11.761 | 9.770 | 27.642 | 1152 |
| T4 | B3 | 13.172 | 11.207 | 31.707 | 1136 |
| T4 | B4 | 10.349 | 29.741 | 84.146 | 830 |
| T5 | B2 | 12.666 | 8.509 | 23.544 | 1439 |
| T5 | B3 | 13.701 | 9.424 | 26.076 | 1418 |
| T5 | B4 | 11.138 | 31.199 | 86.329 | 908 |

## B4 Minus B2 Cluster Effects

Rates are pooled within each physical reference-scene cluster. Negative
ID-switch differences favor B4; positive recovery differences favor B4.

| T | Reference scene | ID-switch difference | Recovery-recall difference | False-birth difference |
|---:|---|---:|---:|---:|
| T2 | `10b17940-3938-2467-8a7a-958300ba83d3` | -0.990 | NA | 0 |
| T2 | `137a8158-1db5-2cc0-8003-31c12610471e` | 0.000 | NA | 0 |
| T2 | `280d8ebb-6cc6-2788-9153-98959a2da801` | -1.163 | NA | 2 |
| T2 | `5630cfcf-12bf-2860-8784-83d28a611a83` | 0.000 | NA | 1 |
| T2 | `8eabc45f-5af7-2f32-8528-640861d2a135` | 0.000 | NA | 1 |
| T2 | `ddc73797-765b-241a-9e2c-097c5989baf6` | -1.316 | NA | 7 |
| T3 | `10b17940-3938-2467-8a7a-958300ba83d3` | -1.515 | 29.412 | -16 |
| T3 | `137a8158-1db5-2cc0-8003-31c12610471e` | 0.448 | 19.079 | -51 |
| T3 | `280d8ebb-6cc6-2788-9153-98959a2da801` | 0.000 | 0.000 | -11 |
| T3 | `5630cfcf-12bf-2860-8784-83d28a611a83` | -3.175 | NA | -4 |
| T3 | `8eabc45f-5af7-2f32-8528-640861d2a135` | -3.846 | 35.294 | -19 |
| T3 | `ddc73797-765b-241a-9e2c-097c5989baf6` | -1.744 | -2.381 | 1 |
| T4 | `10b17940-3938-2467-8a7a-958300ba83d3` | -1.375 | 32.941 | -34 |
| T4 | `137a8158-1db5-2cc0-8003-31c12610471e` | 0.867 | 17.847 | -119 |
| T4 | `280d8ebb-6cc6-2788-9153-98959a2da801` | 0.341 | 7.692 | -72 |
| T4 | `5630cfcf-12bf-2860-8784-83d28a611a83` | -4.420 | 100.000 | -10 |
| T4 | `8eabc45f-5af7-2f32-8528-640861d2a135` | -5.797 | 34.314 | -68 |
| T4 | `ddc73797-765b-241a-9e2c-097c5989baf6` | -2.922 | 6.742 | -19 |
| T5 | `10b17940-3938-2467-8a7a-958300ba83d3` | -1.542 | 32.847 | -60 |
| T5 | `137a8158-1db5-2cc0-8003-31c12610471e` | 1.050 | 17.143 | -187 |
| T5 | `280d8ebb-6cc6-2788-9153-98959a2da801` | 0.781 | 13.483 | -100 |
| T5 | `5630cfcf-12bf-2860-8784-83d28a611a83` | -3.390 | 100.000 | -32 |
| T5 | `8eabc45f-5af7-2f32-8528-640861d2a135` | -10.377 | 38.974 | -106 |
| T5 | `ddc73797-765b-241a-9e2c-097c5989baf6` | -3.196 | 11.594 | -46 |

## Interpretation

Fresh V3 output supports stronger pooled long-gap recovery for B4 than B2 at both T4 and T5.
The six cluster effects are descriptive robustness evidence; the 129
order-units are not treated as independent observations.

## Channel Contract

Task channel: official ReScene task candidates plus trajectory linkage
produce t-mAP/t-REC. Identity channel: registered query-level tracker
decisions produce switch/recovery/fragmentation/merge diagnostics. These
are separate prediction objects and are not presented as one ranked list.
The identity regression uses the frozen V1 query-observation cache that
generated the reference B4 rows. The later V2 task-cache inference
realization is distinct and is not substituted into this regression.

Frozen B4 regression cells: `7224`; maximum absolute difference: `0`.
