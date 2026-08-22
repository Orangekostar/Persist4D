# Full-History vs Persistent-State System Comparison

Method A: ReScene4D Full-History (Frozen T2 Checkpoint).
Method B: Persist4D Persistent-State.

Source commit: `575acc12fbd63f38fc3c16578914b25c2fed8584`
Final classification: `SYSTEM_PARETO_LOCK`

## Q1. What is the checkpoint training horizon?

The formal checkpoint was trained and validated at temporal horizon T2.

## Q2. Are T3-T5 Full-History results zero-shot extensions?

Yes. Full-History T3/T4/T5 are zero-shot temporal-horizon extensions of the frozen T2 checkpoint.

## Q3. How does Full-History task quality change with T?

Full-History causal-prefix t-mAP: T2=0.1910, T3=0.1079, T4=0.0690, T5=0.0453.

## Q4. How does Persist4D task quality change with T?

Persist4D causal-prefix t-mAP: T2=0.1586, T3=0.0989, T4=0.0596, T5=0.0445.

## Q5. Which system has better deployment identity stability?

Persist4D has the lower T4/T5 normalized deployment ID-switch rate. Full-History: T2=98.92%, T3=98.00%, T4=98.16%, T5=98.07%; Persist4D: T2=10.15%, T3=9.97%, T4=10.35%, T5=11.14%.

## Q6. Which system has better Gap Identity Recovery?

Persist4D has the higher T4/T5 Gap Identity Recovery recall. Full-History: T2=NA, T3=0.00%, T4=1.28%, T5=1.00%; Persist4D: T2=NA, T3=29.41%, T4=29.74%, T5=31.20%.

## Q7. Which system has better per-new-visit latency scaling?

Persist4D has the better T4/T5 per-new-visit latency and point-processing scaling. Median latency values are in table_b_compute_scaling.csv.

## Q8. Which system has better peak VRAM scaling?

Persist4D has the lower T4/T5 peak allocated VRAM. Peak allocated/reserved VRAM is reported in table_b_compute_scaling.csv and remains separate from state/input bytes.

## Q9. Does Persist4D form an accuracy/identity/compute Pareto advantage?

Yes. Persist4D forms the preregistered system-level Pareto result: no meaningful Full-History accuracy advantage and Persist4D improves identity and compute.

## Q10. What should happen next?

Freeze the method as a Pareto result; do not add modules, then run external validation and write the paper.

## Evidence

- `REScene_FULL_HISTORY_CODE_AUDIT.md`
- `FULL_HISTORY_DETERMINISM_AUDIT.md`
- `table_a_system_comparison.csv`
- `table_b_compute_scaling.csv`
- `cluster_bootstrap.csv`
- `leave_one_scene_out.csv`
- `order_robustness.csv`

The comparison uses the frozen T2 checkpoint, exact Protocol-B prefixes, all three preregistered orders, and reference-scene clusters as the independent statistical units.
