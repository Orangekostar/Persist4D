# CVPR Evidence Closure Implementation Plan

> Primary-agent execution under AGENTS.md. Mechanical workers may only execute
> explicit monitoring commands. Use the executing-plans workflow task by task;
> the user's standing instruction authorizes recommended decisions without questions.

**Goal:** Deliver a source-grounded CVPR assessment and execute missing feasible evidence.
**Architecture:** Add post-training analysis and evaluation adapters beside the frozen
training implementation. Separate the historical memory-system evidence from new
native-long perception results, with explicit links only where actual evaluation exists.
**Tech Stack:** Existing Python/PyTorch/stmetrics environment, JSON/CSV and pytest.
**Spec:** `docs/superpowers/specs/2026-10-11-cvpr-evidence-closure-design.md`.

## Global constraints

- E0–E3 seed45 U=29700, locked checkpoints and official populations are immutable.
- Lifetime cap 1800 GPUh, reserve 8 GPUh; no unauthorized extension.
- Preserve original worktree and manuscript user changes; no dependency updates.
- No per-reference AP averaging presented as pooled AP; no invented measurements.
- Final review, algorithms, selection and debugging stay with the primary agent.

## Task 1: Verified post-selection analysis

Files: new `scripts/native_long_postselection.py`,
`tests/test_native_long_postselection.py`, `artifacts/native_long_postselection_v1/`.

- [x] Write failing tests for positive S_long with T1 regression, same-step controls,
  missing coverage, negative modules, missing reference horizons and insufficient
  seed46 budget. Use synthetic metrics with a +0.01 long delta and -0.003 T1
  delta: expected LONG_TRADEOFF and no seed46 eligibility.
- [x] Run `python -m pytest tests/test_native_long_postselection.py -q` and verify
  missing implementation is the failure cause.
- [x] Implement `classify_delta(delta)`, `module_shortlist(comparisons)` and
  `reference_deltas(candidate, control)` with protocol thresholds and strict paired
  identities. Implement the CLI `--root PATH --output PATH` to verify snapshots,
  CAL lock, evidence SHA and final completeness before exporting tables.
- [x] Run the tests, then run the CLI on the real completed cluster. Confirm E3
  eligible, E2 LONG_TRADEOFF, official six SEL rows preserved and budget unchanged.
- [x] Review full diff and artifact/source bindings; record checks in the report.

## Task 2: Formal and runtime evidence execution

Files: new `scripts/native_long_confirmation.py`, focused tests and separate
`cluster/postselection/` manifests/receipts in the runtime root.

- [x] Resolve actual historical population manifests and materialize exact ordered
  scan lists through the existing native loader. Validate TRAIN exclusion before GPU use.
- [ ] Test that a trained checkpoint identity mismatch, missing reference or changed
  input order invalidates evaluation; verify no mutation of original population/lock.
- [ ] Load fixed checkpoint via native_training_identity/system_from_common, use
  native postprocessing and independent official accumulators. Execute E0/E1 and
  eligible module only with existing charged_process and worker exclusion.
- [x] Run selected-module/E1 A and B→A fresh-process tests with fixed CAL inputs;
  measure output and official AP variation, retain RUNTIME_CONDITIONAL if unresolved.
- [x] Run fixed CAL T1–T5 profile, warmup1/measure3; separate setup, IO, forward,
  postprocessing and peak VRAM. Audit ledger and evidence after processes exit.

## Task 3: Scientific review, delivery and manuscript integration

Files: `ccfa-review-reports/2026-10-11-cvpr-readiness.md`, evidence package,
and a separate CVPR manuscript revision only after the claim/evidence review.

- [x] Verify current CVPR 2027 official CFP and closest primary prior work; full author rules remain unverified.
- [ ] Bind every manuscript claim to real results, list confounds, baseline gaps,
  independent validation gaps and exact acceptance checks.
- [ ] Export and reload required trained task bundles; publish only within existing
  GitHub authorization with real remote bytes/SHA receipts.
- [ ] Update tables/prose to distinguish internal development, formal confirmation,
  repeatability, trained perception and persistent-identity evidence.
- [ ] Audit the whole goal against the spec. Preserve an active goal until all
  required work is complete or an externally constrained blocker meets the threshold.

## Execution checkpoint, 2026-10-11

Formal manifests are frozen: 890 inputs per row, zero TRAIN-reference overlap,
historical whole-sequence supervision filter preserved. E0/5940, E1/23760 and
E3/2970 are complete; E1/2970 PB is complete and its remaining two populations
are finishing. The first collected audit verified447 official states and
independently recomputed24 pooled states exactly. Final all-row audit remains pending.

All four task bundles passed exact trained tensor roundtrip and SHA binding.
Runtime/profile12-job queue completed; RUNTIME_AUDIT.json verified all evidence.
Same-order repeat processes match exactly; A versus B→A changes predictions with
zero AP change on the fixed CAL sample, so retain RUNTIME_CONDITIONAL.

A frozen trained-perceptor bridge now compares FH-B2, LOCAL-B2 and LOCAL-B4 on
all129 PB units for E0/5940 and E1/23760. Query-feature output and raw-cache schema
adapter failures were preserved with charged ledgers. CPU integration tests pass
including gap recovery and empty observations; full645-prefix alignment is
rerunning against the final corrected source before attempt3. Methods, weights,
thresholds, input order and evaluation truth remain fixed. Full bridge metrics,
trained mechanism diagnostics, publication, manuscript integration and final
readiness review remain pending. GPT-5.6 Luna continues read-only monitoring.


## Execution checkpoint, 03:03 UTC

- Formal confirmation COMPLETE:3560 inputs,596 state SHA/content checks,32 pooled recomputations.
- Trained memory bridge attempt3 running; first real sequence complete with bound evidence.
- Trained CAL mechanism queue attempt2 running:150 inference+2 gradient cases,
  six CAL T5 units/two references, preserved fixed selection and methods. Seven
  focused tests pass;30 real CPU prefix loads and44 remote datafiles verified.
- Training trace audit preserves four E3 repeated-step feedback-norm differences;
  no byte-identical training replay claim. Original numerical sources unchanged.
- Final bridge/mechanism audits, publication, manuscript integration and full
  requirement review remain open; no goal-completion claim.
