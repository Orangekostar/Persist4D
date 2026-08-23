# Metric Aggregation Note

- `tmap_iou_sweep.csv` pools all 129 sequence/order scopes inside one official `stmetrics` accumulator per method and horizon, then reports class-macro temporal AP. It is not a mean of per-sequence AP values.
- `observation_coverage.csv` is a micro-average over GT entity/stage instances. Categories are mutually exclusive at each threshold.
- `failure_decomposition.csv` uses prefix-specific P6-A failure events. Each event receives exactly one primary category; insufficient evidence remains `unknown_unresolved`.
- GT is used only after frozen inference for Oracle identity assignment. Masks, classes, scores, features, and model forward outputs are unchanged.
- Oracle follows the existing P6-A offline diagnostic: unmatched valid candidates remain stage-unique births rather than being removed with GT.
- Oracle ceiling gate: minimum absolute t-mAP gain `0.05` and at least `50%` closure of a positive Full-History gap at T4 or T5.
- Phase III classification: `PERCEPTION_CEILING`.
