# Reviewer-Closure Figure Captions

## Performance Decomposition

**Why similar temporal AP emerges from different failure modes.** (a) Pooled class-macro temporal AP over official stmetrics IoU thresholds on all 129 sequences. The relative ordering changes with both horizon and overlap strictness. (b) The fraction of GT entity stages with a class-compatible candidate above the selected IoU threshold is similar for Full-History and Persist4D at T4/T5; the standalone coverage figure reports all four mutually exclusive categories. (c) Persist4D failure events retain an explicit unknown/unresolved bucket rather than forcing complete attribution. Observation, class, high-IoU mask, capacity, and unresolved failures dominate the measured T4/T5 composition, while registered identity-fragmentation, merge, and wrong-recovery events form smaller portions. (d) The P6-A offline GT-ID-only readout underperforms both frozen systems because it retains unmatched candidates and changes neither masks nor classes; it is a diagnostic of this specific relabeling policy, not a performance upper bound. Source: `tmap_iou_sweep.csv`, `observation_coverage.csv`, `failure_decomposition.csv`, and `oracle_association_results.csv`.

## Strong-Baseline Identity Scaling

**Deployment identity stability after simple cross-prefix association.** Normalized ID-switch rate is pooled over 129 sequences. Native Full-History identity issuance is near one switch per transition opportunity from T3 onward. The selected B2 feature-and-class tracker reduces this rate to approximately 0.14, while Persist4D remains lower at approximately 0.10-0.11 and supplies a measured T2 transition rate. Native Full-History and B2 initialize at T2, so their T2 rates are not applicable. Source: `full_history_tracker_aggregate.csv`.

## Horizon-Adaptation Task Scaling

**Task quality after the T2-to-T3 ReScene horizon adaptation.** Pooled official causal-prefix t-mAP and t-REC are shown over all 129 sequence/order scopes at T2-T5. B2 changes deployment identity assignment only, so the frozen and adapted ReScene+B2 curves retain their corresponding checkpoint task metrics. The adaptation improves pooled T4 recall, but neither T4 nor T5 satisfies the preregistered cluster-CI, order, and LOSO task-advantage rule. Source: `rescene_horizon_adaptation_results.csv`, `rescene_horizon_adaptation_cluster_bootstrap.csv`, and `rescene_horizon_adaptation_order_robustness.csv`.

## Standalone Decomposition Panels

- `iou_threshold_curve.*`: pooled class-macro temporal AP across IoU 0.25-0.90 for T4/T5.
- `observation_coverage.*`: mutually exclusive observation categories at IoU 0.25/0.50/0.75; hatching distinguishes Persist4D from Full-History independently of color.
- `failure_decomposition.*`: Persist4D T4/T5 failure-event fractions using the operational definitions in `failure_decomposition.csv`, including unknown/unresolved.
- `oracle_association_gain.*`: frozen Full-History and Persist4D t-mAP beside the P6-A GT-ID-only readout with unmatched candidates retained.
- `horizon_adaptation_task_scaling.*`: frozen ReScene+B2, horizon-adapted ReScene+B2, and Persist4D pooled task scaling.
