# Reviewer Closure V3 Statistical Contract

This contract is fixed before new V3 inference.

## Registered Factors

- Evaluation seeds: `45`, `46`, `47`.
- Orders: `canonical`, `reverse`, `sha256_seed45`.
- Horizons: `T2`, `T3`, `T4`, `T5`.
- Primary score reducer: `mean`.
- Sensitivity reducers: `latest`, `max`.
- Oracle matching threshold: IoU `0.5`, post-prediction only.
- Primary robustness cluster: `reference_scene_id` (six clusters).

No seed, reducer, threshold, order, or evaluation subset may be selected after
observing V3 performance.

## Units And Pairing

Pooled t-mAP/t-REC over all eligible order-units is descriptive. The 43 masters
under three orders produce 129 order-units, but those are not 129 independent
observations. Orders share masters, and masters may share a physical reference
scene.

Every method difference must be paired within an identical:

- reference scene;
- master sequence;
- order;
- horizon;
- evaluation seed where GPU inference is involved;
- local prediction cache and target.

Order comparisons keep all masters belonging to the same reference scene in
one cluster. Horizon comparisons stay within the same master/order/cache.
Population comparisons use the identical runtime/seed and report
`t-mAP(exact-43 canonical T2) - t-mAP(full-154 T2)` per seed.

## Robustness Reporting

All six reference-scene effects must be shown. Any bootstrap resamples the six
clusters with replacement and keeps every master/order row in a sampled cluster
together. Intervals are descriptive robustness evidence because `N=6`; reports
must not imply high-powered significance or treat order-units as independent.

Mean/range across seeds is reported for the population bridge. No additive
decomposition from the paper-reported 34.8% to a Protocol-B number is permitted.

## Channel Separation

`local_current_AP` uses latest-stage official masks/classes/scores directly and
must be tracker-invariant for a fixed sidecar. `causal_prefix_t_mAP` uses linked
trajectory predictions and its preregistered reducer. Identity switch, recovery,
fragmentation, and merge diagnostics come from fresh registered query-level
tracker steps. These channels answer different questions and are not presented
as fields from one identical confidence-ranked prediction object.

## Decision Rules

- PB1 accepts positive, zero, or negative population/order/horizon effects if
  coverage, pairing, and runtime bindings are complete.
- EV0 keeps mean primary even if latest/max are more favorable. A material sign
  flip narrows the task claim rather than selecting a reducer.
- ID0 first requires deterministic B4 regression, then evaluates B2/B3/B4 and
  all six B4-minus-B2 cluster effects.
- Oracle-ID is diagnostic headroom only and cannot be called a method/baseline.
- Final RC3 classification follows only the preregistered V3 prompt gates.
