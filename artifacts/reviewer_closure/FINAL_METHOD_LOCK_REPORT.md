# Persist4D Final Method-Lock Report

## 1. Frozen method definition

Persist4D is a bounded persistent identity/state layer on top of local ReScene perception. Each forward produces masks, class logits, and query features; `build_local_observation` filters the current-stage candidates, and a capacity-100 `PersistentMemory` associates them to slots using feature and class similarity. The slot state contains embedding, class probability, confidence, occupancy, activity, age, last-seen stage, and a processed-stage watermark. Only stored embedding and class probability enter the association score; occupancy restricts eligible slots and determines free birth capacity. Current-observation confidence scales matched-slot EMA updates, while stored confidence is updated but does not enter the association score. Activity, age, last-seen stage, and the stage watermark enforce lifecycle and processing-order invariants and do not enter matching. Unmatched candidates create births while capacity overflow is explicitly rejected.

Memory does not initialize ReScene queries and is not fed into cross-attention, self-attention, the mask decoder, or the class decoder. Persist4D therefore claims persistent identity/state maintenance on top of local perception, not memory-conditioned perception.

## 2. Strongest trivial alternative

The selected trivial alternative is B2 pairwise feature-and-class association, chosen by the preregistered T4/T5 ranking. It lowers frozen Full-History normalized ID-switch rate from 0.9781/0.9774 to 0.1416/0.1441 at T4/T5, but does not close gap recovery: B2 reaches 0.0647/0.0581 versus Persist4D's 0.2974/0.3120.

At T5, the six-cluster Persist4D-minus-B2 gap-recovery difference is +0.3773 with 95% CI [0.1783, 0.6486], and it is consistent across orders and all LOSO evaluations. Gate I is `TRACKER_REJECTED`. ReScene native query indices are a per-forward namespace without a persistent-ID contract; they are not described as a failed tracker.

## 3. Temporal-horizon adaptation challenge

The strongest feasible audit is Level 2, `ReScene4D T2-to-T3 Horizon-Adapted`, not a claimed from-scratch T3 reproduction. It starts from the audited epoch-404 T2 checkpoint and runs one frozen 45-epoch schedule: 2,160 optimizer updates, 145,932 scan exposures, 10.85 wall-clock hours, and 21.69 A40 GPU-hours. The canonical epoch-44 checkpoint reloads strictly and is content-bound in `rescene_horizon_training_manifest.json`.

Adapted ReScene+B2 versus Persist4D pooled t-mAP is 0.0699 versus 0.0596 at T4 and 0.0454 versus 0.0445 at T5; pooled t-REC is 0.1888 versus 0.1362 and 0.1312 versus 0.1067. These pooled differences do not survive the complete preregistered rule: cluster bootstrap intervals include zero for both task metrics at both horizons, and order/LOSO direction is inconsistent. Gate II is `HORIZON_ROBUST`, with no qualifying task cell.

## 4. Why t-mAP is near-parity

At IoU 0.50, the associable GT entity-stage fraction is similar for Full-History and Persist4D: 0.5709 versus 0.5833 at T4 and 0.5924 versus 0.5881 at T5. Persist4D's registered T4/T5 failures are dominated by observation miss (12.0%/11.6%), class failure (23.4%/23.3%), high-IoU mask failure (21.9%/21.7%), capacity (16.0%/13.9%), and unresolved evidence (18.7%/20.5%). Registered fragmentation, merge, and wrong recovery together account for 7.9%/9.0%.

Thus similar t-mAP does not imply identical temporal behavior. Persist4D improves persistent identity continuity, while local observation coverage and spatial/class precision remain the larger measured error sources.

## 5. Oracle headroom

The P6-A offline GT-ID readout changes identity assignment only, leaves masks and classes unchanged, and retains unmatched candidates. It underperforms both frozen systems at every horizon (T4 t-mAP 0.0050; T5 0.0027), so it is not presented as a mathematical performance upper bound. Under the preregistered diagnostic gate it provides no evidence that association alone can close the remaining task gap. Together with the IoU sweep and failure taxonomy, Phase III is `PERCEPTION_CEILING`.

## 6. External geometry matcher

Not triggered. Gate I rejects the trivial-tracker explanation and Phase III does not identify association as the remaining ceiling. No LivingScenes component, supported-subset experiment, geometry cue, or matcher integration was added, and no claim is made about its effectiveness on this benchmark.

## 7. Compute and memory Pareto result

At T4/T5, horizon-adapted Full-History requires 932.7/1068.2 ms median update latency versus Persist4D's 495.4/440.4 ms, ratios of 1.88x/2.43x. Its peak allocated VRAM is 3646/4770 MiB versus 2096/2473 MiB, ratios of 1.74x/1.93x. Cumulative scans processed are 9/14 versus 6/8, ratios of 1.50x/1.75x. Adapted Full-History also supplies up to 42.9/59.0 MB of explicit history input at T4/T5, while Persist4D carries a measured bounded historical state of 61,008 bytes.

The adapted alternative fails the preregistered 1.10 maximum ratio for latency, peak VRAM, and cumulative scans. Persist4D remains a useful accuracy-identity-compute Pareto point.

## 8. Statistical robustness

All formal comparisons use six `reference_scene_id` clusters, 10,000 cluster-bootstrap replicates, seed 45, three fixed orders, and six leave-one-scene-out checks. Official task metrics are pooled across all 129 sequence/order scopes; cluster statistics use cluster-macro per-sequence metrics and are not substituted for the official pooled values.

For the adaptation challenge, no task cell passes CI, order, and LOSO requirements. Adapted ReScene+B2 retains a robust T5 gap-recovery deficit versus Persist4D: difference -0.3786, 95% CI [-0.6472, -0.1866], with complete order and LOSO consistency. T4 gap recovery has one undefined reference cluster (5/6 finite), so that cell fails closed despite its interval. The 129 scopes are not asserted to be independent environments.

## 9. Claims supported

- A bounded persistent entity state supplies a useful long-horizon operating point beyond the evaluated frozen and T2-to-T3-adapted Full-History alternatives.
- Persist4D's gap-recovery benefit is not explained by native query naming or the strongest preregistered simple tracker.
- Persist4D retains materially lower T4/T5 update latency, peak allocated VRAM, and cumulative scan processing.
- Under the frozen decomposition, remaining task error is more strongly associated with local observation/class/mask limits, capacity, and unresolved evidence than with registered identity failures.
- The method should remain frozen under the evaluated 3RScan common-prefix protocol.

## 10. Claims not supported

- Uniform task-accuracy superiority over Full-History ReScene.
- Generalization beyond the evaluated six 3RScan reference environments without external validation.
- Exact reproduction of the official ReScene training recipe or a from-scratch T3-trained model.
- Memory-conditioned perception, improved masks/classes, or persistent query conditioning.
- Treating ReScene per-forward query indices as a tracker contract.
- Treating the P6-A GT-ID readout as a true task upper bound.
- Any benefit from LivingScenes geometry matching or any supported-subset extrapolation.
- Independence of all 129 sequence/order scopes.

## 11. Final classification

`FINAL_LOCK`

Persist4D survives the simple-tracker and temporal-horizon-adaptation challenges, preserves its identity/compute benefit, and solves the persistent-identity failure it claims to address. No method change is justified for this paper.

## 12. Paper-ready next action

Freeze model code, configuration, checkpoints, gates, tables, and figures on this branch. Next, audit and run an independent sparse-revisit dataset with stable cross-time identity and at least three observations; complete the published baseline table; then write the paper around the bounded-state accuracy-identity-compute Pareto result. New module development is out of scope unless new external evidence reopens the method question.
