# TaskMemory Retention V2 Final Report

Results commit: `fb4b6df8f46449981bcb344913de04208e312ba5`.

## Verdict

- EXECUTION: `PARTIAL`
- TMAP_ALL_T_VS_R1: `FAIL`
- TMAP_ALL_T_VS_MATCHED_FH: `PASS`
- TASK_METRICS_ALL_T: `PASS`
- RETENTION: `IMPROVED`
- RESOURCE: `TRADEOFF`
- MECHANISM: `PARTIAL`
- GENERALIZATION: `NOT_ESTABLISHED`
- PUBLICATION: `PUSH_VERIFIED`

## Protocol-B tMAP

| Variant | T2 | T3 | T4 | T5 | Mean |
|---|---:|---:|---:|---:|---:|
| R1+B4 | 0.219637 | 0.135498 | 0.080159 | 0.058535 | 0.123457 |
| FH-R1 | 0.228777 | 0.148305 | 0.102745 | 0.081566 | 0.140349 |
| M3-BASE-CONT | 0.223988 | 0.133415 | 0.078771 | 0.048050 | 0.121056 |
| FH-CONT | 0.208039 | 0.085259 | 0.032787 | 0.012343 | 0.084607 |
| M3-V-CORE | 0.223172 | 0.135569 | 0.067608 | 0.038498 | 0.116212 |

## Strict comparisons

- M3-V-CORE_vs_R1+B4: tMAP `FAIL`, 20-cell `12/20`, minimum tMAP delta `-0.020037`, failed tMAP horizons `T4, T5`.
- M3-V-CORE_vs_FH-R1: tMAP `FAIL`, 20-cell `4/20`, minimum tMAP delta `-0.043068`, failed tMAP horizons `T2, T3, T4, T5`.
- M3-V-CORE_vs_M3-BASE-CONT: tMAP `FAIL`, 20-cell `5/20`, minimum tMAP delta `-0.011163`, failed tMAP horizons `T2, T4, T5`.
- M3-V-CORE_vs_FH-CONT: tMAP `PASS`, 20-cell `20/20`, minimum tMAP delta `0.015133`, failed tMAP horizons `none`.

## Retention and mechanism

Absolute A(T), relative R(T), and Dmax are in `final/retention.csv`. Mechanism is
reported separately from task superiority; the final development content control
supports only the status shown above.

## Reference evidence

The six physical references are the statistical units. The T2/T5 intervals in
`final/reference_bootstrap.csv` are equal-reference descriptive intervals, not
pooled-AP confidence intervals.

## Resources

Resource status is `TRADEOFF`. The profile uses one A40,
six fixed canonical units, five warmups, ten measurements, cloned prior state, and
separate model-update, end-to-end, materialization, and true cumulative scopes.

## Limitations

Protocol-B is a previously exposed historical benchmark. One training seed does
not establish replicated training stability. Unrun variants and independent-data
limits remain explicit in `final/status.json` and `training/variants.json`.
