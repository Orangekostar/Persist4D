# 0929 code-first audit execution contract

Baseline: 4bf00902c9428795f7f47547bf462540aa304cc6. Historical artifacts,
external cache, locks and tag remain immutable. New artifacts use
artifacts/rescene_code_first_audit_v1; runtime uses
/home/ww/persist4d_runs/rescene_code_first_audit_v1.

1. Preserve installed-evaluator probe evidence and count affected candidates by
   role/horizon. Inspect original locked Q score changes and quantify hypothetical
   M eligibility/assignment changes from the same immutable predictions.
2. Store score-independent geometry quality separately from per-threshold
   supervision validity. A resolvable candidate is eligible at threshold t when
   its temporal event is positive or its ignore fraction is <= t. Explicit
   ambiguity/unknown/invalid/empty candidates remain excluded. Q1/Q2 supervise
   the union of threshold eligibility; Q3 averages BCE over eligible matrix
   entries. Fully unscored candidates remain explicitly identifiable.
3. Preserve original M eligibility (ignore <= minimum threshold) in a distinct
   geometry_valid field. Q repair must leave M assignments and targets identical.
   Any expanded M eligibility experiment is separate, conditional on measured
   changes, and limited to M0/M1 with the same assignment and original recipe.
4. Separate prediction, label, training and evaluation identities. Resume checks
   current direct dependencies, inputs and weights; COMPLETE alone is insufficient.
   Label-only changes reuse immutable R1 predictions. Publish recovery cannot
   bypass compute freshness. Cover apply_module, heads, materialization, official
   metric sources/version and dataset specification in evaluation identity.
5. Relabel and retrain Q1/Q2/Q3 for the original 1500 updates using the original
   seed, sample plan, parent and inputs. Compare at original locked steps
   Q1=1500, Q2=1500, Q3=1000; do not select on SEL or launch a new grid.
6. Add an explicit augmentation switch preserving train-mode discovery/targets.
   Evaluate original locked heads on identical CAL/SEL scans/order without
   training augmentation, with an independently evaluated B0 for that condition.
7. Run focused regression checks, real controlled evaluations, cost accounting,
   and requirement-by-requirement final review. Do not infer improvements or
   attribute all historical Q degradation to the eligibility bug.

Probe completed against unchanged baseline: actual installed matcher/evaluator
Git blob identities match the supplied review. Affected candidate counts in
(0.5,0.9]: TRAIN H1=586/H2=629, CAL H1=63/H2=53, SEL H1=37/H2=23.
These counts measure eligibility only, not AP causality.

Execution status: implementation and controlled experiments complete. The
primary agent verified 36 results/72 metric states, the fixed training invariants,
and all requirements in artifacts/rescene_code_first_audit_v1/REQUIREMENT_REVIEW.md.
All scientific decisions and final review stayed with the primary agent.
