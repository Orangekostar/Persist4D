# Persist4D Reviewer-Closure Summary

Final decision: `FINAL_LOCK`.

The frozen Persist4D method survives both reviewer-grade alternative explanations tested in this closure study. The strongest preregistered simple tracker, B2 feature-and-class association, does not explain Persist4D's gap-recovery advantage (`TRACKER_REJECTED`). The Level-2 ReScene4D T2-to-T3 horizon adaptation does not produce a statistically reliable T4/T5 task advantage and remains materially more expensive (`HORIZON_ROBUST`). Phase III attributes the remaining ceiling primarily to local perception rather than association (`PERCEPTION_CEILING`), so LivingScenes was not triggered.

## Decision Evidence

| Audit | Result | Decisive evidence |
|---|---|---|
| Simple cross-prefix tracker | `TRACKER_REJECTED` | At T5, Persist4D gap-recovery recall exceeds B2 by 0.3773 across all six clusters; 95% CI [0.1783, 0.6486], order- and LOSO-consistent. |
| T2-to-T3 horizon adaptation | `HORIZON_ROBUST` | No T4/T5 task cell passes the pooled-difference, six-cluster CI, order, and LOSO rule. |
| Adapted identity challenge | Persist4D retained | At T5, adapted ReScene+B2 minus Persist4D gap-recovery recall is -0.3786; 95% CI [-0.6472, -0.1866], order- and LOSO-consistent. |
| Compute scaling | Persist4D retained | At T4/T5, adapted full-history latency is 1.88x/2.43x, peak allocated VRAM is 1.74x/1.93x, and cumulative scans are 1.50x/1.75x Persist4D. |
| Performance ceiling | `PERCEPTION_CEILING` | IoU-0.50 associable coverage is similar, while observation, class, mask, capacity, and unresolved errors dominate the registered failure taxonomy. |
| External matcher | Not triggered | The preregistered association-reopen condition was not met. |

## Scope And Provenance

The formal evaluation contains 43 master sequences, three preregistered orders, T2-T5 prefixes, and six independent `reference_scene_id` clusters. The 129 sequence/order scopes are not treated as 129 independent environments. Task metrics use pooled official accumulators; uncertainty uses 10,000 reference-cluster bootstrap replicates with seed 45, plus order and leave-one-scene-out checks.

The adaptation ran once for 45 epochs and 2,160 optimizer updates on two A40 GPUs, consuming 21.69 GPU-hours. The canonical 720 MiB checkpoint is locally ignored; `rescene_horizon_training_manifest.json` records its SHA256 (`ba4ad656...d27bb`), size, epoch, global step, and strict reload. The 516 prediction and 516 observation-cache entries remain locally ignored; their committed manifests bind every entry by hash.

Primary evidence: `gate_i.json`, `rescene_horizon_gate_ii.json`, `phase_iii_manifest.json`, `rescene_horizon_adaptation_results.csv`, `rescene_horizon_compute.csv`, and `figures/horizon_adaptation_task_scaling.*`.

The method is frozen for this paper. Next work is independent sparse-revisit validation, completion of the published baseline table, and paper writing; no additional Persist4D module is justified by this evidence.
