# Reviewer-Closure Visual Contract

Mode: standard

## Performance Decomposition

Artifact: `figures/performance_decomposition.{svg,pdf,png}` and four standalone panels
Target venue / format: full-width two-column paper figure; editable SVG/PDF plus 300 dpi PNG preview
Core claim: similar headline t-mAP can coexist with different spatial, observation, and identity failure modes.
Reviewer question: does Full-History gain from stronger spatial/coverage reasoning while Persist4D gains from persistent identity continuity?
Evidence layer: mechanism and limitation
Source data: `tmap_iou_sweep.csv`, `observation_coverage.csv`, `failure_decomposition.csv`, `oracle_association_results.csv`
Statistics / uncertainty: pooled class-macro official stmetrics over 129 sequences; deterministic event fractions; no confidence interval is available for these decompositions.
Figure prototype or table type: small-multiple lines, 100% stacked bars, horizontal composition bars, grouped diagnostic bars
Panel or table map: (a) T4/T5 temporal AP over IoU threshold; (b) compact T4/T5 associable coverage at IoU 0.25/0.50/0.75, with the standalone figure retaining the full four-category stacks; (c) Persist4D T4/T5 failure composition including unknown/unresolved; (d) frozen systems versus P6-A GT-ID-only diagnostic at T2-T5.
Caption role: state the metric aggregation, expose the explicit unresolved bucket, and warn that the P6-A diagnostic retains unmatched candidates and is not an upper bound.
Manuscript placement: performance-decomposition section after long-horizon comparison.
Output formats: SVG, PDF, PNG
Traceability: every SVG contains a title, source-data description, and stable semantic colors; the QA ledger records rendered inspection.

## Strong-Baseline Identity Scaling

Artifact: `figures/strong_baseline_identity_scaling.{svg,pdf,png}`
Target venue / format: single-column or appendix-width paper figure
Core claim: the selected simple B2 cross-prefix tracker materially reduces the frozen Full-History identity failure but does not erase Persist4D's long-horizon identity advantage.
Reviewer question: does Persist4D survive Full-History plus the strongest simple cross-prefix tracker?
Evidence layer: main comparison
Source data: `full_history_tracker_aggregate.csv`, rows `FullHistoryNative`, `B2`, and `Persist4D`
Statistics / uncertainty: pooled deployment ID switches divided by pooled transition opportunities over 129 sequences; native Full-History and B2 initialize at T2 and are not applicable there, while Persist4D retains its measured T2 transition rate.
Figure prototype or table type: horizon line chart with direct endpoint labels
Panel or table map: one panel, T2-T5 normalized ID-switch rate.
Caption role: distinguish native identity issuance, simple B2 association, and bounded persistent state.
Manuscript placement: strong-baseline closure section.
Output formats: SVG, PDF, PNG
Traceability: source row identities and metric definition appear in SVG metadata and caption text.

## Horizon-Adaptation Task Scaling

Artifact: `figures/horizon_adaptation_task_scaling.{svg,pdf,png}`
Target venue / format: single-column or appendix-width paper figure
Core claim: deferred until formal adapted-checkpoint evaluation is available.
Reviewer question: does exact T3 long-horizon adaptation reverse the frozen task-quality ordering at T4/T5?
Evidence layer: main comparison
Source data: future `rescene_horizon_adaptation_results.csv`; no placeholder values are permitted.
Statistics / uncertainty: exact frozen-protocol aggregate metrics and any approved cluster-level uncertainty in the future Phase II artifact.
Figure prototype or table type: horizon line chart
Panel or table map: frozen ReScene, adapted ReScene, and Persist4D over T2-T5.
Caption role: identify the checkpoint provenance and separate adapted from frozen inference.
Manuscript placement: long-horizon adaptation section before decomposition.
Output formats: SVG, PDF, PNG
Traceability: renderer must fail closed until the complete Phase II source CSV exists.
