# Persist4D All-T Final Report

## Outcome

| Status | Value |
| --- | --- |
| execution_status | COMPLETE |
| baseline_comparability | MATCHED |
| tmap_all_t_vs_R1 | FAIL |
| tmap_all_t_vs_matched_rescene | FAIL |
| task_metrics_all_t | FAIL |
| seed_confirmation | NOT_RUN |
| independent_generalization | NOT_ESTABLISHED |
| publication_status | NOT_ATTEMPTED |

Execution completed, but the scientific all-T superiority goal failed. C2 did not strictly exceed R1 B4 at T4, T5, nor matched FH-adapt at T3, T4, T5. Protocol-B was final-only and did not change selection.

Code commit used by this publisher: `b35701fb276ef47e2d0111e49385a584d657ce4d`. Publication commit: resolve from remote HEAD.

## Frozen Protocol-B Results

| Horizon | C2 mean | R1 B4 mean | Delta vs R1 B4 | FH-adapt | Delta vs FH-adapt | FH-R1 |
| --- | --- | --- | --- | --- | --- | --- |
| T2 | 0.225025 | 0.219637 | 0.005388 | 0.223600 | 0.001425 | 0.228777 |
| T3 | 0.141059 | 0.135498 | 0.005560 | 0.147723 | -0.006664 | 0.148305 |
| T4 | 0.076000 | 0.080159 | -0.004159 | 0.103657 | -0.027657 | 0.102745 |
| T5 | 0.051323 | 0.058535 | -0.007213 | 0.080308 | -0.028985 | 0.081566 |

The primary reducer is one frozen `mean` C2 checkpoint. `latest` and `max` remain sensitivity analyses only.

## All Candidate-vs-Matched Task Cells

| Metric | Horizon | Raw delta | Strict > 0 |
| --- | --- | --- | --- |
| t_mAP | T2 | 0.001425 | PASS |
| t_mAP | T3 | -0.006664 | FAIL |
| t_mAP | T4 | -0.027657 | FAIL |
| t_mAP | T5 | -0.028985 | FAIL |
| t_mAP50 | T2 | -0.004089 | FAIL |
| t_mAP50 | T3 | 0.000601 | PASS |
| t_mAP50 | T4 | -0.052444 | FAIL |
| t_mAP50 | T5 | -0.052216 | FAIL |
| t_mAP25 | T2 | -0.026373 | FAIL |
| t_mAP25 | T3 | -0.051469 | FAIL |
| t_mAP25 | T4 | -0.079664 | FAIL |
| t_mAP25 | T5 | -0.071030 | FAIL |
| t_REC | T2 | -0.003363 | FAIL |
| t_REC | T3 | 0.011039 | PASS |
| t_REC | T4 | -0.007575 | FAIL |
| t_REC | T5 | 0.000584 | PASS |
| prefix_overall_mAP | T2 | 0.005919 | PASS |
| prefix_overall_mAP | T3 | -0.013475 | FAIL |
| prefix_overall_mAP | T4 | -0.037826 | FAIL |
| prefix_overall_mAP | T5 | -0.049288 | FAIL |

## Training And Selection

| Variant | State | Selected update | Trainable params | GPU-h | Exposure | Checkpoint/reason |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | COMPLETE | 400 | 26688019 | 2.128 | 3200 episodes / 7642 stages | 574ea496db467f1208eea934d2aa29595b36057c7bad27dcd5cadc88793cdb25 |
| C1 | COMPLETE | 100 | 26754195 | 2.255 | 3200 episodes / 7642 stages | 44afc0faa72d9e4215b8c27b1d6f0b15a94dd5e9d63da34add9a4868e7f6da0d |
| C2 | COMPLETE | 200 | 26787475 | 2.232 | 3200 episodes / 7642 stages | a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724 |
| C3 | GATE_SKIPPED | N/A | N/A | N/A | N/A | C1 did not strictly improve both T2 and T3 versus selected C0, so the preregistered complementarity gate failed closed. |
| FH-adapt | COMPLETE | 100 | 26688019 | 2.551 | 3200 episodes / 7642 stages | ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c |
| FH-L | GATE_SKIPPED | N/A | N/A | N/A | N/A | The frozen Persist4D candidate does not use L, so matched FH-L training is not authorized. |

All formal models used seed 45, two A40 GPUs, physical batch 1/GPU, gradient accumulation 4, effective batch 8, 400 optimizer updates, and `32-true` precision. C3 and FH-L contain no fabricated numeric run results. Seed 46 was gate-skipped after the seed-45 development candidate failed the matched all-T gate.

### Selected Development Curves

| Variant | Update | T2 | T3 | T4 | T5 | Minimum |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 400 | 0.507136 | 0.389127 | 0.365171 | 0.334946 | 0.334946 |
| C1 | 100 | 0.506292 | 0.396667 | 0.356264 | 0.330555 | 0.330555 |
| C2 | 200 | 0.512769 | 0.395971 | 0.376152 | 0.348153 | 0.348153 |
| FH-adapt | 100 | 0.508799 | 0.452106 | 0.400897 | 0.384912 | 0.384912 |

| Effect | T2 | T3 | T4 | T5 |
| --- | --- | --- | --- | --- |
| L: C1-C0 | -0.000844 | 0.007541 | -0.008906 | -0.004391 |
| M: C2-C0 | 0.005633 | 0.006844 | 0.010981 | 0.013207 |

L alone (C1) did not satisfy its short-horizon complementarity gate. M alone (C2) improved the selected C0 at all development horizons, but only T2 exceeded selected FH-adapt; therefore L+M (C3) was not authorized. C2 reads prediction-only persistent slots beyond its W=2 input window and commits each stage immediately.

D0 contains 827 official-valid trajectory rows (220 unique GT trajectories). Its non-additive labels include mask-insufficient=319, full-history-success/B4-failure=112, class-change=25, fragmentation=21, merge=28, and no-candidate=10. T1 cold-start contributed to 133/350 failed rows (0.380).

## Memory Read Diagnostic

| Policy | Horizon | t-mAP | Delta vs all occupied |
| --- | --- | --- | --- |
| disabled | T2 | 0.512545 | -0.000224 |
| disabled | T3 | 0.397997 | 0.002026 |
| disabled | T4 | 0.376917 | 0.000765 |
| disabled | T5 | 0.348949 | 0.000796 |
| active_previous | T2 | 0.512583 | -0.000186 |
| active_previous | T3 | 0.400301 | 0.004330 |
| active_previous | T4 | 0.378036 | 0.001884 |
| active_previous | T5 | 0.348885 | 0.000732 |

This is a 47-master development diagnostic on the frozen C2 checkpoint. Disabling or restricting reads slightly improved T3-T5 t-mAP, but these policies were not selected models and do not establish causality.

## Reference-Cluster Evidence

| Horizon | Positive references / 6 | Minimum delta | Maximum delta |
| --- | --- | --- | --- |
| T2 | 3 | -0.123590 | 0.047998 |
| T3 | 3 | -0.125970 | 0.061751 |
| T4 | 1 | -0.105642 | 0.039312 |
| T5 | 1 | -0.092506 | 0.025167 |

| Horizon | Equal-cluster mean delta | 95% descriptive interval | Clusters |
| --- | --- | --- | --- |
| T2 | -0.018332 | [-0.065955, 0.021688] | 6 |
| T3 | -0.021410 | [-0.075038, 0.026212] | 6 |
| T4 | -0.035615 | [-0.069690, -0.000888] | 6 |
| T5 | -0.037593 | [-0.068600, -0.005632] | 6 |

The intervals are 1,000 fixed-seed resamples of six equal-weight reference deltas. They are descriptive equal-cluster effects, not pooled-AP confidence intervals and not evidence of independent generalization.

## Identity Evidence

| Model | Horizon | ID switches | Fragments | Merges | Correct recovery | Attempts | Gap opp. | Accuracy | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | T2 | 39 | 39 | 77 | 0 | 0 | 0 | N/A | N/A |
| C2 | T3 | 77 | 100 | 142 | 84 | 96 | 272 | 0.875 | 0.3088235294117647 |
| C2 | T4 | 117 | 172 | 223 | 218 | 260 | 696 | 0.8384615384615385 | 0.3132183908045977 |
| C2 | T5 | 155 | 243 | 285 | 348 | 422 | 1093 | 0.8246445497630331 | 0.3183897529734675 |
| FH-adapt | T2 | 469 | 469 | 51 | 0 | 0 | 0 | N/A | N/A |
| FH-adapt | T3 | 946 | 1115 | 172 | 0 | 100 | 275 | 0.0 | 0.0 |
| FH-adapt | T4 | 1440 | 1874 | 346 | 2 | 289 | 705 | 0.006920415224913495 | 0.0028368794326241137 |
| FH-adapt | T5 | 1969 | 2691 | 598 | 3 | 465 | 1101 | 0.0064516129032258064 | 0.0027247956403269754 |

Identity values are freshly recomputed from T1-T5. A zero denominator is represented as `N/A`; it is never converted to 0 or 1.
Although C2 has substantially fewer ID switches/fragments and much higher recovery rates than FH-adapt, that identity advantage did not produce all-T task-metric superiority.

## Bounded Resource Profile

| Model | Horizon | Median of cell medians (ms) | Cell range (ms) | Max incremental MiB | Median voxel points |
| --- | --- | --- | --- | --- | --- |
| C2 | T2 | 598.834 | 448.086-731.518 | 1831.2 | 107015 |
| C2 | T3 | 533.940 | 398.578-736.524 | 1815.9 | 88838 |
| C2 | T4 | 490.112 | 370.437-576.188 | 1336.2 | 77082 |
| C2 | T5 | 435.099 | 342.438-688.280 | 1703.4 | 65276 |
| FH-adapt | T2 | 598.495 | 444.197-723.567 | 1829.7 | 107015 |
| FH-adapt | T3 | 781.470 | 546.748-1007.175 | 2693.8 | 150270 |
| FH-adapt | T4 | 923.107 | 624.384-1112.516 | 3028.7 | 185549 |
| FH-adapt | T5 | 1055.358 | 775.224-1453.270 | 4136.0 | 220421 |

The profile contains 480 measurements (2 models x 6 references x 4 horizons x 10 repeats) on one NVIDIA A40 after 5 warmups. It includes model forward, C2 memory read, observation extraction, and one B4 update; it excludes file I/O, collation, H2D, metrics, and C2 preroll. C2 uses K=100 and W=2 with deployment memory that does not grow with T. This bounded profile does not rescue the failed accuracy verdict.

## Next Round

Change exactly one factor: add a learned quality score/gate for memory reads while keeping the checkpoint-selection rule, reducer, K=100, W=2, losses, and evaluation protocol fixed. This is the sole recommended bottleneck because the frozen read ablations improve T3-T5 slightly while the present all-occupied read loses at longer horizons; it remains a hypothesis until a preregistered rerun.
