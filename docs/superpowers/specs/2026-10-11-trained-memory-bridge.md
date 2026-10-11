# Trained-perceptor memory confirmation

The primary agent fixes this extension before inspecting formal metric results.
No model, checkpoint, threshold or association method is selected from PB scores.

Use the existing CAL-selected E0/5940 and E1/23760 seed45 bundles. Run all129
registered PB master/order units, preserving six physical reference clusters.
Reuse the frozen formal data manifest and historical ambiguity annotations.
For each checkpoint compare full-history inference with the fixed B2 feature/class
tracker, latest-pair inference with B2, and the same latest-pair observations with
the frozen B4 persistent-state tracker. B2/B4 settings come unchanged from
conf/p6a/default.yaml. No GT enters association or model inference.

Process units in sorted source_unit order and visits1–5. For visits1/2 the input
contexts are identical: perform one forward and fan out exactly the same output.
For visits3–5 run the causal full prefix followed by the latest pair, each under
the same per-unit/per-visit9001 seed. All predictions come from a fresh process per
checkpoint with this explicit process-order protocol. Retain RUNTIME_CONDITIONAL;
these are separately bound measurements, not byte-identical extensions of formal
PB's horizon-major process order. Save shared-prefix parity evidence.

Full-history task AP uses its official joint-prefix output. Both local systems use
OfficialCandidateTrajectoryAccumulator with the historical mean-score publisher,
the exact official current-stage candidate masks/classes/scores/lineage and B2/B4
query identities. Also measure current-stage raw AP for both contexts. These are
different deployment contracts and must be labelled; tracker changes cannot be
credited for changes in shared local masks or scores. Identity evaluation uses
the existing class-compatible IoU0.5 Hungarian matcher on current-stage raw query
masks and method/sequence-scoped persistent IDs. Report numerators/denominators,
physical-reference summaries and missing recovery opportunities explicitly.

Enable the existing model.return_query_features export switch after exact bundle
reload: the native training bundle defaults to task outputs only, while the
association adapter consumes the final decoder-normalized query features. This
adds the existing optional output after task predictions are formed. Preserve
the failed first attempt, its producer/plan and billed reservation; the corrected
attempt keeps all checkpoints, inputs, methods, thresholds and ordering fixed.

The second attempt exposed an adapter schema mismatch: raw P6A observations use
schema3, while the separate official-candidate sidecar uses schema1. Preserve
that failed attempt and its reservation. Bind raw payloads to p6a_cache's
SCHEMA_VERSION and validate the payload-to-sidecar-to-tracker path on CPU,
including empty observations and an absent-then-returning entity, before retry.
This changes serialization compatibility only; the measurement protocol stays fixed.

Validate current-stage GT identity/mask alignment across contexts and point
ordering. CPU preflight found a class-only mismatch: at canonical
scene0069_01-scene0069_06-scene0069_09-scene0069_05-scene0069_08:T5, GT2 is class17
in the full prefix/current scan but class0 in the pair. The official collator
uses the first point of an instance in visit order (pointcept_utils.py), so
window-dependent labels are expected when annotations disagree across visits.
At scene0079_01-scene0079_06-scene0079_11-scene0079_10-scene0079_02:canonical:T5,
GT46 is included in the full prefix but filtered in the pair; raw current point
coordinates, semantic labels and instance IDs remain exactly identical.
Before any GPU bridge measurement, fix one common evaluation truth: the original
official causal full-prefix target for every method, including current-stage
identity and raw AP. Record every pair/full class difference; do not discard the
instance. Require independent full/pair fingerprints of current raw coordinates
and point semantic/instance labels to match exactly. Reconstruct common masks
from the pair's original point IDs using the full-prefix GT instance selection;
check equality with full-prefix current masks. Build committed local targets
using the common selection/classes and require exact ID/class/mask/stage
equality with the full-prefix truth. Report both label and filter differences.
No evaluation
target enters inference or association. Stop on identity/mask/point-order
mismatch rather than silently changing denominators.
Export official pooled and per-reference sufficient states; store per-unit
identity assignments, input/output fingerprints, shared-prefix equality, capacity
and timings. No pooled AP confidence interval is inferred from mean scene AP.

Inference, CPU tracker updates and metric work are charged through charged_process
while the GPU is reserved. Execute on E3's node only after its formal worker exits
and under worker.lock; each checkpoint has a2 GPUh ceiling inside its existing
quota and the global1800 GPUh cap. Any failure is recorded and diagnosed by the
primary agent. Timing includes IO/full network/postprocessing/tracker separately;
the shared first two forwards are not claimed as free deployment cost.

Acceptance: all129 units for both fixed checkpoints; each T2–T5 pooled denominator
129; local B2/B4 current predictions exactly equal; no future context; no TRAIN
reference overlap; source/weight/input/evidence SHA bound; clear distinction from
historical manuscript numbers and native-long formal AP. Negative or inconclusive
outcomes stay in the final readiness report.
