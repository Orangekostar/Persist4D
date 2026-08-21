# Persist4D P6-B Protocol V2 Design

## Status and Scope

This document supersedes the P6-B evaluation protocol in
`2026-08-19-persist4d-p6b-association-upgrade-design.md`. The frozen ReScene
checkpoint, P5, P6-A Protocol-B cache, split, local masks, and local classes do
not change. P6-B remains an association and persistent-state method. P7 and P8
remain out of scope.

The P6-B evidence recorded at commit `782439a` is invalidated as final evidence:
independent review found a cached mask-support representation error, a
cardinality-first assignment objective, count-based candidate ranking, shallow
selection/artifact validation, incomplete statistics, and no durable
exactly-once held-out boundary. The historical commits remain immutable audit
evidence; protocol v2 creates new evidence rather than rewriting history.

## Frozen Inputs

Protocol v2 consumes the existing P6-A cache. Every cache observation already
contains an integer `mask_support` equal to the number of true mask points.
Cached boolean masks are retained for task metrics but are never reinterpreted
as logits for the birth-support gate. No ReScene forward pass or GPU recache is
needed.

Each inference observation carries explicit query-aligned mask support. Native
model observations may derive support from real mask logits at their declared
threshold. Cached observations must use their persisted integer support and
must fail closed if it is missing or inconsistent with the cached boolean mask.

## Association Objective

Threshold-aware assignment maximizes total accepted association score with
explicit zero-score dummy unmatched nodes. Forbidden real edges cannot be
selected. The optimizer does not maximize match cardinality before score. A
single score `1.25` therefore wins over two accepted scores `0.50 + 0.49`.
Deterministic tie-breaking is subordinate to total score and uses stable row and
column order only for exact optima.

Dormant reactivation requires both:

- `score >= reactivation_threshold`; and
- `best_score - competing_score > reactivation_margin` when a margin is set.

The strict margin matches the preregistered scientific rule. Active matching
retains its configured inclusive score threshold.

## Normalized Candidate Evidence

Every candidate-horizon row records both counts and denominators:

- identity switches and transition opportunities;
- wrong reactivations, correct reactivations, reactivation attempts, and gap
  opportunities;
- false births, true births, accepted births, and valid birth opportunities;
- accepted valid observations and frozen-B4 valid observations.

Rates are recomputed from those fields. Candidate ranking uses paired,
reference-cluster-normalized quantities, never raw counts:

1. lower paired mean T4/T5 identity-switch rate;
2. lower paired mean T3-T5 wrong-reactivation rate;
3. lower paired mean T2-T5 false-birth rate;
4. higher paired mean T3-T5 reactivation recall;
5. higher paired mean T4/T5 strict-online task score;
6. canonical configuration JSON.

Eligibility retains the existing anti-rejection constraints on reactivation
accuracy/recall, accepted valid observations, and T2 strict-online task metrics.

## Complete Sweep and Selection Validation

The selection document is a reproducible claim, not a container. Its validator
reconstructs the preregistered staged search and verifies:

- exact schemas, unique canonical configuration IDs, and configuration hashes;
- exact stage grids and all required candidate/horizon rows;
- tuning-only reference/master membership and absence of held-out results;
- all count/rate identities and denominator bounds;
- candidate eligibility and Pareto membership;
- every stage winner, finalist, final ranking key, and selected configuration;
- the GT-free inference test digest and frozen source/cache/protocol hashes.

Any missing, extra, duplicated, reordered-to-change-meaning, or tampered row
fails closed. Artifact validation independently recomputes the final gates and
requires the complete paired per-sequence population, failure analysis,
statistics, provenance, and attempt ledger before G6B-5 can pass.

## Exactly-Once Held-Out Boundary

Held-out evaluation and artifact packaging are separate commands.

1. `final-evaluate` atomically creates a durable attempt token before loading
   held-out cache entries. It records source, frozen selection, split, command,
   start/end UTC, exit status, and log/input/output hashes. A successful raw
   held-out payload makes further evaluation attempts fail closed.
2. The raw payload contains paired B4/P6B per-sequence results and diagnostics,
   but no rendered report or gate decision.
3. `final-package` consumes the immutable raw payload. Packaging is repeatable
   and may be retried after formatting/schema failures without recomputing the
   held-out metrics.

The attempt ledger is manifest-bound and reports all attempts. Protocol v2
begins with an explicit invalidation/reset commit because protocol-v1 held-out
results were already consumed under an invalid contract.

## Paired Cluster Statistics

Rows pair B4 and P6B by exact
`(reference_scene_id, master_sequence_id, order_id, horizon)`. Missing or
duplicate partners fail closed. Deltas are aggregated within reference scene,
then summarized across held-out reference-scene clusters.

Protocol v2 reports mean, sample standard deviation, and a deterministic 95%
cluster-bootstrap interval using seed 45 and 10,000 reference-scene resamples.
At minimum this covers identity-switch rate, wrong-reactivation rate,
false-birth rate, reactivation accuracy/recall, t-mAP, and t-REC. With only two
held-out clusters, the report explicitly identifies interval instability and
does not overstate significance.

## Final Decision and Claims

The existing G6B-1 through G6B-5 thresholds remain preregistered. Each gate is
recomputed from raw paired rows and includes its numeric pass/fail values.
Reports disclose inactive selected mechanisms, all failed gates, protocol
deviations, and the exact attempt history. Failure of any gate produces
`P6B_STOP`; no result triggers P7/P8 automatically.

## Verification

Implementation is test-first. Required regression evidence includes:

- real cached boolean-mask support uses explicit `mask_support`;
- maximum-score dummy assignment counterexamples and deterministic ties;
- strict dormant margin boundary;
- count/rate/denominator identities and normalized ranking;
- exact search coverage and tamper rejection;
- durable exactly-once evaluation with retryable packaging;
- exact pair construction and deterministic cluster bootstrap;
- complete artifact/gate recomputation, privacy, and frozen P5/P6-A hashes.

After CPU tests pass, tuning is rerun from the frozen cache, one new protocol-v2
held-out evaluation is consumed, artifacts are packaged, and the full relevant
test suite is run before release.
