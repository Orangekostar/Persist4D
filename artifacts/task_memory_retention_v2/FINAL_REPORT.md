# TaskMemory Retention V2 Final Report

Results commit: `3027b942a5d76dea1dad3f42b3c67c0a9c9383ee`.

## Verdict

- EXECUTION: `COMPLETE`
- TMAP_ALL_T_VS_R1: `FAIL`
- TMAP_ALL_T_VS_MATCHED_FH: `PASS`
- TASK_METRICS_ALL_T: `PASS`
- RETENTION: `IMPROVED`
- RESOURCE: `TRADEOFF`
- MECHANISM: `PARTIAL`
- GENERALIZATION: `BASE_EXPOSED_ONLY`
- PUBLICATION: `PUSH_VERIFIED`

## Protocol-B tMAP

| Variant | T2 | T3 | T4 | T5 | Mean |
|---|---:|---:|---:|---:|---:|
| FH-R1-native | 0.228777 | 0.148305 | 0.102745 | 0.081566 | 0.140349 |
| B4-commit0 | 0.219637 | 0.135498 | 0.080159 | 0.058535 | 0.123457 |
| R1+B4-lag1 | 0.244751 | 0.123902 | 0.061478 | 0.034896 | 0.116257 |
| FH-R1-lag1 | 0.243306 | 0.105958 | 0.047023 | 0.016563 | 0.103213 |
| W-BASE | 0.219235 | 0.091059 | 0.036850 | 0.011986 | 0.089783 |
| Q-TALA | 0.226566 | 0.134196 | 0.077771 | 0.046475 | 0.121252 |
| M3-BASE-CONT | 0.223988 | 0.133415 | 0.078771 | 0.048050 | 0.121056 |
| M3-V-CORE | 0.223172 | 0.135569 | 0.067608 | 0.038498 | 0.116212 |
| FH-MATCH | 0.224733 | 0.097008 | 0.037274 | 0.013837 | 0.093213 |
| FH-CONT | 0.208039 | 0.085259 | 0.032787 | 0.012343 | 0.084607 |
| D-LAST-commit0 | 0.220178 | 0.138127 | 0.080269 | 0.057070 | 0.123911 |
| D-LAST-lag1 | 0.249229 | 0.155346 | 0.090114 | 0.064331 | 0.139755 |
| D-EMA-commit0 | 0.220178 | 0.138226 | 0.078855 | 0.059468 | 0.124182 |
| D-EMA-lag1 | 0.249229 | 0.154903 | 0.088704 | 0.063843 | 0.139170 |

FH-R1-native retains its original official output policy; FH-R1-lag1 and the
new-model rows use lag1/mean. Cross-policy deltas do not isolate a model change.

## Strict comparisons

- M3-V-CORE_vs_B4-commit0: tMAP `FAIL`, 20-cell `12/20`, minimum tMAP delta `-0.020037`, failed tMAP horizons `T4, T5`.
- M3-V-CORE_vs_R1+B4-lag1: tMAP `FAIL`, 20-cell `14/20`, minimum tMAP delta `-0.021579`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_FH-R1-native: tMAP `FAIL`, 20-cell `4/20`, minimum tMAP delta `-0.043068`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_FH-R1-lag1: tMAP `FAIL`, 20-cell `16/20`, minimum tMAP delta `-0.020134`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_W-BASE: tMAP `PASS`, 20-cell `18/20`, minimum tMAP delta `0.003936`, failed tMAP horizons `none`.
- M3-V-CORE_vs_Q-TALA: tMAP `FAIL`, 20-cell `9/20`, minimum tMAP delta `-0.010163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_M3-BASE-CONT: tMAP `FAIL`, 20-cell `5/20`, minimum tMAP delta `-0.011163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_FH-MATCH: tMAP `FAIL`, 20-cell `17/20`, minimum tMAP delta `-0.001561`, failed tMAP horizons `T2`.
- M3-V-CORE_vs_FH-CONT: tMAP `PASS`, 20-cell `20/20`, minimum tMAP delta `0.015133`, failed tMAP horizons `none`.
- M3-V-CORE_vs_D-LAST-lag1: tMAP `FAIL`, 20-cell `1/20`, minimum tMAP delta `-0.026057`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_D-EMA-lag1: tMAP `FAIL`, 20-cell `1/20`, minimum tMAP delta `-0.026057`, failed tMAP horizons `T2, T3, T4, T5`.

## Retention and mechanism

Absolute A(T), relative R(T), and Dmax are in `final/retention.csv`. Mechanism is
reported separately from task superiority; the final development content control
supports only the status shown above.

## Reference evidence

The six physical references are the statistical units. The T2/T5 intervals in
`final/reference_bootstrap.csv` are equal-reference descriptive intervals, not
pooled-AP confidence intervals.

The separate native T2-T4 evidence in `final/independent_reference_results.csv`
uses adaptation-holdout references that were exposed to the original R1 base;
therefore its generalization status is `BASE_EXPOSED_ONLY`, not `INDEPENDENT`.

## Resources

Resource status is `TRADEOFF`. The profile uses one A40,
six fixed canonical units, five warmups, ten measurements, cloned prior state, and
separate model-update, end-to-end, materialization, and true cumulative scopes.

## Limitations

Selected M3-V-CORE loses all four tMAP horizons against both D-LAST-lag1 and
D-EMA-lag1, and against FH-R1-native. D controls use separately generated
prediction-only R1 observations (645 R1 stage forwards in this run) that were
not included in the A40 latency profile. Their gains over R1+B4 are not an
isolated LAST/EMA update-rule effect.

Protocol-B is a previously exposed historical benchmark. One training seed does
not establish replicated training stability. FH encoder-cache equivalence and
new-model saturation effects were not established. Exact limits remain explicit
in `final/status.json`.
