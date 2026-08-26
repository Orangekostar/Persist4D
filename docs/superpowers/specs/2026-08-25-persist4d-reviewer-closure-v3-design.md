# Persist4D Reviewer Closure V3 Design

Date: 2026-08-25
Status: approved by the researcher-supplied V3 execution prompt
Source prompt: `repo:docs/Persist4D_Codex_Reviewer_Closure_V3_Prompt.md`

## Objective

Close the reviewer-facing evidence boundary without training a new model or
changing Persist4D. The study separates external reported evidence from local
measurements, population from order and horizon effects, local candidates from
trajectory scoring, and task metrics from freshly recomputed identity metrics.

## Immutable Boundaries

- Start at `c2f1bcacff1ec244909426b57403965f679f08cc` on a new V3 branch.
- Freeze checkpoint SHA-256
  `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
- Do not overwrite V1/V2 artifacts or relabel the local 27.939% result as
  official ReScene4D.
- Do not modify task candidate masks, predicted classes, or local official
  scores after `extract_official_task_prediction()`.
- Do not use GT identity in B2/B3/B4 inference.
- Do not launch P3, G3, a 450-epoch run, or add model/memory modules.
- Keep `mean` as the preregistered primary trajectory score reducer.

## Evidence Architecture

### Baseline Boundary

A machine-readable contract has three disjoint evidence classes:

1. E0: ReScene4D-C 34.8% is paper-reported and not locally rerun.
2. E1: 27.939% is the frozen local best-effort reimplementation with G2 RED.
3. E2: FullHistory is an internal control using the same frozen local model.

Live official-repository state is frozen as URL, remote commit, retrieval date,
README hash, and checkpoint-section status. It cannot silently change the local
checkpoint used by V3.

### Exact-Prefix Bridge

The bridge derives each canonical T2 record from the first two scans of the
Protocol-B T5 master. It uses the T5 point rows belonging to those two scans and
the first transition column only. This rule is accepted only after exact parity
with all 14 canonical pairs independently present in the sliding-T2 database.

PB0 fails closed unless all 43 records are validation/supervised, ordered scan
IDs are exact, no pair is substituted, no reverse order is substituted, and no
future transition leaks into the target.

### Bridge Evaluation

The same frozen checkpoint and official ReScene post-processing runtime evaluate
both full-154 and exact-43 T2 populations at seeds 45, 46, and 47. Order effects
use canonical, reverse, and sha256_seed45 on the same 43 masters. Horizon effects
use exact T2-T5 prefixes within identical master/order units.

Pooled values are descriptive. All robustness comparisons remain paired by
reference scene, master, order, horizon, and cache. The six reference scenes are
the inference clusters; 129 order-units are never treated as independent.

### Task And Score Channels

`OfficialCandidateTrajectoryAccumulator` supports exactly `mean`, `latest`, and
`max`. Reducers operate only on committed occurrence scores. Masks, classes,
keys, causal commitment, and ephemeral candidates remain invariant.

The current-local channel reads the latest-stage official sidecar directly and
uses `OfficialMetricAccumulator(mode="raw_local")`. It is independent of tracker
identity and therefore must be identical for B0/B2/B3/B4 for a fixed sidecar.
Trajectory t-mAP remains a separate confidence-ranked causal-prefix channel.

### Identity Channel

B2/B3/B4 are instantiated through the existing registered tracker factories.
Fresh steps produce association events and deployment identity metrics through
the existing evaluator code. The V2 `_identity_from_old()` path remains frozen
history and cannot supply final V3 fields. B4 first has to regress against the
old deterministic diagnostics before new comparisons are accepted.

### Oracle-ID Diagnostic

Oracle linkage is post-prediction analysis only. Hungarian matching at IoU 0.5
assigns visible GT identity as the trajectory key `(gt_id, predicted_class)`.
It cannot modify local masks, classes, scores, cache generation, or inference.
Unmatched candidates remain ephemeral.

## Stage Gates

- V3-0: audit, evidence contract, and statistical contract.
- V3-1 / PB0: exact 43/43 bridge and 14/14 parity.
- V3-2 / PB1: separate population, order, and horizon reports.
- V3-3 / EV0: exact mean regression, score-only sensitivities, local invariance.
- V3-4 / ID0: fresh complete B2/B3/B4 identity evidence.
- V3-5 / OR0: post-prediction Oracle-ID headroom.
- V3-FINAL: synthesis only, with exactly one RC3 outcome.

Every stage is committed only after its focused tests and artifact validators
pass. Expensive inference cannot start before the preceding construction gate.

## Failure Handling

- Provenance mismatch or dirty frozen input: stop the affected stage.
- PB0 parity mismatch: do not evaluate the 43-record bridge.
- Mean V2 mismatch or B4 identity mismatch: diagnose before sensitivity work.
- Official checkpoint release during execution: record it, do not replace the
  checkpoint, and require separate authorization for a new validation stage.
- Disk pressure: use `/mnt/shared/$USER` only for large untracked caches/checkpoints,
  retain content hashes, and never move tracked evidence there.
