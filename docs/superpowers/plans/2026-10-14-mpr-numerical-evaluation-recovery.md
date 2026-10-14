# MPR numerical evaluation recovery

Status: conditional plan; formal evaluation remains held until controls finish.

## Evidence and admission

- Six seed146 models have completed 400 updates under frozen C3 numerical sources.
- Fresh original M0 CAL repeats have maximum AP range 2.888917922973633 pp, exceeding the registered 0.2 pp limit. All three original results remain evidence.
- A full diagnostic replay agrees with original repeats0/1 within 0.002674 pp; repeat2's larger discrepancy has not been reproduced.
- First-input probes locate residual FP32 variation at grouped mean aggregation; encoder and query positions were identical on the measured probes. This does not establish the cause of the larger outlier.
- Three independent full CAL controls replace only `AdaptiveScatter.mean_scatter` with stable sorting and `segment_csr` FP32 mean. Parent weights, inputs, random seeds, precision and global deterministic settings remain unchanged.
- Compare all 197 raw prediction and pre-mask summaries, all input hashes, occupied counts and pooled AP rows across the three controls. Undefined/nonfinite/incomplete/mismatched results fail admission.
- A recovery policy requires the same 0.2 pp AP limit; do not relax the limit or choose/discard original repeats. Do not claim the original large outlier's cause is established without additional evidence.

## Conditional implementation, owned by the primary agent

If controls pass, create an explicit, versioned evaluation policy for the mathematical mean with a fixed reduction order. Preserve the original RUN_LOCK, training source registry, training checkpoints, budget and all original evaluation outputs. Do not edit source identities in existing checkpoints.

1. Add a separate, hashed evaluation policy module and evaluation entry point. Loading trained deltas retains the original exact C3 numerical-source checks. Bind the additional evaluation policy source hashes in every evaluation STATUS/RESULTS request from the first write.
2. Write policy evaluations to a distinct namespace under the original run root. Budget and job journals remain in the original charged ledger; policy jobs use distinct keys. Read-only links to original checkpoints/RUN_LOCK and the original ledger allow existing validation and retry accounting without copying weights or resetting costs.
3. Add an explicit controller that reuses the completed main trajectories, runs three fresh formal M0 CAL evaluations under the policy, and stops before checkpoint evaluation if admission fails. Original unadmitted evaluations are excluded from policy selection and remain archived as negative evidence.
4. All CAL/SEL/PB evaluations and P5 inference must use the same hashed policy. Before collecting/selecting/bootstrap, verify every required result's policy, original source registry, parent/checkpoint identity and complete denominator. The legacy four-checkpoint selector, guards, replication gate and bootstrap remain scientifically unchanged.
5. Pin the policy and protocol amendment before launch. Explain that existing weights were trained under C3 and are evaluated under a separately registered numerical execution policy. Preserve both provenance chains.
6. CPU regression tests must reject policy mismatches, original-result reuse, missing annotations and changed policy sources. Validate stable mean on empty groups and empty input. Run one actual charged endpoint before resuming the full matrix. No blind retries on repeated errors.

## Completion evidence

Formal continuation requires: three fresh formal baseline repeats complete and admitted; all original models/checkpoint hashes unchanged; original fourteen numerical source hashes unchanged; additional policy sources match their immutable lock; ledger contains every diagnostic, failed and formal event once; monitor is observing the actual new namespace and current child.

No publication-quality gain or original root-cause claim follows from a single successful diagnostic control. Keep the final scientific status inconclusive until the complete required matrix, selection, confirmation and reporting finish.
