# Persist4D P6-B Association Upgrade Design

## Purpose

P6-B improves only cross-stage association and persistent-state maintenance. It
does not retrain ReScene, change the checkpoint, alter local masks/classes, or
modify the frozen P5/P6-A implementations and artifacts. Every candidate
consumes the exact P6-A Protocol-B prediction cache.

The design follows the evidence from P6-A:

- B4 already beats B3 on long-horizon identity continuity and reactivation.
- Wrong dormant reactivation increases with gap length, especially at gaps 2-3.
- P6-A T5 contains 169 wrong versus 435 correct B4 reactivations.
- P6-A T5 contains 908 diagnosed false births versus 1,306 true births.
- Correct births have higher confidence/support and lower entropy in aggregate.
- Capacity is not saturated, so P6-B does not tune capacity.

## Considered Approaches

### Recommended: isolated P6-B memory and cached evaluation

Add a separate P6-B memory implementation and evaluator adapter. Reuse the
typed P6-A observation, state, event, and metric contracts, while keeping P5 B4
as the frozen control. This preserves the scientific variable and permits a
direct paired comparison from identical cache entries.

### Rejected: edit P5 PersistentMemory in place

This would make the historical P5/P6-A result irreproducible and confound the
comparison. Even behavior-preserving compatibility flags would make the frozen
control harder to audit.

### Rejected: filter P6-A association events post hoc

Post-hoc filtering cannot reproduce the compounded effect of assignment,
birth, and consolidation decisions on later memory states. It is suitable only
for diagnostics, not a valid method evaluation.

## Architecture

### Runtime memory

`models/persistent_memory_p6b.py` owns the new method. It reuses
`LocalInstanceObservation` and `PersistentMemoryState`, but has a new immutable
configuration and step result. The transition contains five independently
auditable mechanisms:

1. Threshold-aware Hungarian assignment with forbidden low-score edges and
   dummy unmatched nodes.
2. Separate active and dormant thresholds, with
   `reactivation_threshold >= active_threshold` and a dormant edge-margin gate.
3. Full-class or foreground-renormalized class compatibility.
4. Confidence-and-margin-gated consolidation: an accepted match remains a
   match, but low-quality evidence does not update the persistent anchor.
5. Confidence/support/entropy birth gating before allocating a free slot.

P5 state tensors and capacity remain unchanged. P6-B adds only query-aligned
diagnostic tensors to its step result; it does not add a second timescale.

### Evaluation adapter

`scripts/p6b_association.py` converts a P6-B step into the existing P6-A
`TrackStep` and `AssociationDiagnostics` contracts. The frozen P5 B4 adapter is
the control. The adapter exposes consolidation and birth rejection decisions
without using GT during inference.

### Sweep and final runner

`scripts/run_p6b_evaluation.py` has two explicit phases:

- `sweep`: evaluate only tuning clusters, write every candidate row, select one
  configuration deterministically, and atomically publish the frozen selection.
- `final`: require the frozen selection hash, evaluate held-out clusters once,
  compare against frozen P5 B4, and publish the complete P6-B evidence bundle.

The runner refuses an existing output root and binds source commit, P5/P6-A
hashes, cache manifest, protocol manifest, split manifest, configuration, and
all derived artifact hashes.

## Leakage-Free Split

The six P6-A reference scenes are ordered by
`SHA256("p6b|45|" + reference_scene_id)`. The first four clusters are tuning
clusters and the final two are held out. This gives 32 tuning masters and 11
final masters. All three deterministic order variants of a master stay in the
same partition. No final-cluster result participates in selection.

The split is persisted in `artifacts/P6B/split_manifest.json`. The final runner
checks exact cluster/master membership against the frozen Protocol-B manifest.

## Search Protocol

Search is staged to avoid an impractical Cartesian product while preserving all
required ablations. Each stage fixes the preceding stage's selected values and
records every attempted row:

1. Assignment: P5 post-threshold versus threshold-aware dummy assignment.
2. Reactivation: active threshold `{0.45, 0.50, 0.55}`, dormant threshold
   `{0.75, 0.85, 0.95, 1.05}`, margin `{0.00, 0.05, 0.10, 0.20}`.
3. Class compatibility: full versus foreground-renormalized, class weight
   `{0.15, 0.25, 0.35}`.
4. Consolidation: full update versus confidence `{0.80, 0.90, 0.97}` crossed
   with margin `{0.05, 0.10, 0.20}`.
5. Birth gate: confidence `{0.50, 0.75, 0.90, 0.97}`, support
   `{1, 128, 512, 1024}`, entropy `{none, 0.75, 0.50, 0.25}`.
6. Joint local verification: selected values and their immediate lower/upper
   neighbors, deduplicated and lexicographically ordered.

All coarse candidates use one causal T5 replay per master/order. Identity,
reactivation, false-birth, and acceptance statistics are derived from that
replay at T2-T5 prefixes. The Pareto finalists receive official strict-online
task evaluation on the tuning clusters before selection.

## Selection Rule

A candidate is eligible only if, on tuning clusters:

- T3-T5 mean reactivation accuracy is at least 0.70;
- T3-T5 mean reactivation recall is no more than 0.05 below frozen B4;
- accepted valid observations are at least 90% of frozen B4;
- T2 online t-REC and t-mAP are each no more than 0.02 below frozen B4.

Eligible candidates are ranked lexicographically by:

1. lower paired mean T4/T5 ID-switch rate;
2. lower T3-T5 wrong-reactivation rate;
3. lower false-birth rate;
4. higher T3-T5 reactivation recall;
5. higher mean T4/T5 online t-mAP plus online t-REC;
6. canonical configuration JSON.

The selection rule and all rejected candidates remain in the sweep artifacts.

## Final GO / NO-GO Gates

P6-B GO requires all held-out gates:

- G6B-1: threshold-aware counterexample passes, GT is absent from inference,
  and frozen P5/P6-A hashes are unchanged.
- G6B-2: paired mean T4/T5 ID-switch rate is at least 10% lower than frozen B4,
  with neither T4 nor T5 worse.
- G6B-3: mean T3-T5 reactivation accuracy is at least 0.70 and not below B4;
  recall is no more than 0.05 below B4.
- G6B-4: T2 online t-mAP/t-REC each drop by at most 0.02, and mean T4/T5
  online t-mAP plus t-REC is not lower than B4 by more than 0.01.
- G6B-5: all required ablations, selected configuration, paired per-sequence
  results, failure analysis, provenance, and artifact-manifest checks pass.

Failure of any gate produces `P6B_STOP`; results are reported without changing
the method claim or entering P7/P8.

## Artifacts

`artifacts/P6B/` contains:

- `P6B_GO_NOGO_REPORT.md`
- `p6b_eval.json`
- `split_manifest.json`
- `assignment_ablation.csv`
- `reactivation_threshold_sweep.csv`
- `class_compatibility_ablation.csv`
- `consolidation_ablation.csv`
- `birth_gate_sweep.csv`
- `joint_validation_sweep.csv`
- `selected_config.yaml`
- `final_results.csv`
- `per_sequence_results.csv`
- `failure_analysis.csv`
- `artifact_manifest.json`
- publication SVG figures

The report uses the required eleven sections and ends with one exact decision:
`P6B_GO` or `P6B_STOP`.

## Verification

Implementation is test-first. Unit tests cover the assignment counterexample,
active/dormant edge rules, foreground normalization, non-consolidating matches,
birth gates, deterministic split, selection discipline, final holdout isolation,
artifact schemas, provenance, privacy, and P5/P6-A immutability. Focused P6-B,
P6-A regression, full CPU, and opt-in real-cache gates must pass before release.
