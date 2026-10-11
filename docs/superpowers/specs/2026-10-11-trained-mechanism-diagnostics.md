# Trained mechanism diagnostics, fixed before execution

Use all six existing CAL T5 records (two physical references), sorted by input ID,
and each causal T1–T5 prefix. These30 inputs are explanatory development cases,
not new independent test scenes or a selection set. Never change a checkpoint,
threshold or model using these outcomes. Preserve TRAIN/CAL/SEL/formal locks.

Rows: E0/5940, E1/23760, E1/2970, E3/2970 with the trained forward intact;
E3/2970 with feedback output replaced by its input. The latter still executes
the original trained block before a forward hook returns the unmodified feature
tensor. Each row/input runs in a fresh process with the same9001 per-input RNG.
The trained files remain unchanged. This is an inference intervention, not an
equal-parameter training control or proof of a unique query-conditioning benefit.

Capture the feedback input queries/features/padding hashes, actual update norm,
12 auxiliary masks/logits and final outputs. For paired E3 intact/off comparisons
require equal physical input tensors, feedback inputs, GT and auxiliary outputs0–7
before attributing later differences to the intervention; otherwise report a
runtime-confounded diagnostic. Save actual tensor evidence and SHA, and report
the four later auxiliary differences and final differences.

Use official full-prefix predictions/GT and pooled sufficient states. Explain
individual GT regions with class-compatible best IoU and IoU>=0.5 candidates;
record whether a qualifying candidate is added or lost under the intervention,
including existing correct regions damaged by feedback. This diagnostic is not
official AP or one-to-one matching. Use official current-stage candidate lineage
to count a GT entity requiring different query IDs across visits, and distinguish
missing masks, recovery, absent targets and class/filter annotation changes.
Do not label raw query fragmentation as persistent-ID tracking accuracy.

For S, collect immutable E1/E2 training sampling traces and E3 feedback traces.
On each CAL T5 input, capture actual hierarchical stage IDs and original sampling
caps from trained E1/23760 inference. After inference, run the existing uniform
and stratified index rules on those tensors using dedicated identical seeds and
the original cap, without changing model inputs or using GT. Report actual cap
activation, per-stage availability/selection and missing-stage counts. This
measures coverage only; E2's task metric tradeoffs remain unchanged.

Run two separate trained E3 gradient probes on the first fixed panel's T2/T5,
unaltered task loss/raw_sum, no optimizer step, no data augmentation. Restore the
configured feedback chunk checkpointing and use training-mode forward. Report
finite gradients for feedback and original decoder separately, auxiliary count,
loss terms, frozen-encoder gradient absence and unchanged parameters. These
probes measure task connectivity, not held-out accuracy or new training updates.

Run on the idle E0 host under worker.lock and charged_process, all GPU occupation
charged as mechanism: events. Enforce a total2 GPUh ceiling inside the existing
quota and1800 GPUh global cap; retain8 GPUh reserve and the running bridge's4 GPUh
ceiling. No overwrite or automatic retry of failed evidence. Primary agent owns
design, interpretation, code and restart decisions; Luna performs read-only probes.

Acceptance:150 inference jobs and2 gradient probes complete; all30 prefixes per
row and all paired comparisons accounted for; pooled official denominators6 per
horizon; no missing examples silently removed; all SHA/source/weight/population
bindings verified; limitations and negative findings included in the report.

## Observed alignment exception and bounded diagnostic extension

After all 152 jobs finished, strict pairing found original record 21 (T2) and
record 25 (T1) had unequal feedback inputs and auxiliary stages 0–7, despite
identical raw input and target hashes. The auditor retains all 30 original
pairs, marks these two RUNTIME_CONFOUNDED, and preserves both full-panel
observational metrics and the explicit 28-pair strict-alignment denominator.
No tensor tolerance, discarded input, replacement outcome or post-hoc matching
is permitted. Audit completion means all measurements were verified, not that
all counterfactual assumptions passed.

A frozen derivative plan repeats both modes twice on both flagged inputs:
8 fresh-process jobs, unchanged E3/2970 weights/data/RNG/numerical source, no
optimizer step. Original plan/results are preserved. The isolated replay root
uses the same original population and runtime assets. E0's existing worker.lock
and charged_process enforce a total 0.3 GPUh ceiling, its original quota, the
1800 GPUh global cap and 8 GPUh reserve. No automatic retries or new training.

The repeat audit retains all 12 sources (4 original plus 8 new) and all 30
within-input pairwise comparisons, recomputing each official state from its
private prediction/target artifact. Same-mode and cross-mode comparisons are
reported separately. New training is blocked on native consistency if
fixed-protocol nonrepeatability is confirmed; existing observations are retained
with that limitation. This diagnostic does not identify a specific earlier
operator merely from a changed mask logit at the first saved auxiliary stage.
