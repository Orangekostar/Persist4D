# OGR-V1 Implementation Plan

> **For agentic workers:** Follow the supplied instruction and this plan. The primary agent implements all model, data, loss, scheduler and scientific logic. Only explicit mechanical leaves may use mechanical_worker; do not delegate design or final review.

**Goal:** Execute the original OGR four-arm research task, including required probes, cost-selected training, two adaptation seeds, all evaluation/diagnostic evidence and actual publication, or its explicitly prescribed truthful blocked branches.
**Architecture:** Fresh M4 subclass plus observation sidecar; constrained criterion; independent causal runtime, resolver, locked controller and delivery. Existing MPR files remain read-only.
**Tech Stack:** Existing Python3.10/PyTorch2.6+cu126, spconv2.3.8, native loader, SciPy matcher, original official AP, Git/LFS.
**Spec:** docs/superpowers/specs/2026-10-18-observation-grounded-design.md and artifacts/observation_grounded_v1/EXECUTION_INSTRUCTION.md.

Execution outcome before publication: NO_FEASIBLE_COMMON_N / BUDGET_BLOCKED. Required modules/config are implemented; 21 focused tests pass, all four 20-update TRAIN probes and the 34-input P1 diagnostic completed. Formal training/selection/two-seed comparisons, full 44-job evaluation, formal bootstrap/profile/interventions remain unexecuted. The final publication facts are recorded outside A/B in the actual external receipt. An unchecked formal item below is intentionally blocked, not silently completed.

## Global constraints

BASE 758b5aa4bd2e9119ae5e05f725faa360c2120d88; parent6b1b779; fresh adapter246/247; Q100/D128/K100; parent45 frozen; raw_sum; one feedback; no augmentation; four episodes and14 stages/update. At most2 jobs,1 GPU/job. Run≤50GPUh, lifetime≤1823, reserve8 untouched. Phases D2/T_FIRST16/T_REPLICATION12/E12/F4/I4. No CAL-driven N/lambda, no test reselection, no skipped negative second seed. No invented metrics, checkpoints, CONFIRM or remote success.

## Task 1 — Resolver, exact input provenance and immutable policy

Files: scripts/ogr_assets.py, scripts/ogr_policy.py, conf/observation_grounded_v1/default.yaml, OUT input/provenance/start/data/exposure/budget records; tests/test_ogr_delivery.py.
- [x] Verify exact package and28 BASE blob identities; copy supplied pure policy/config unchanged. Test reference self_test15 and embedded YAML identity.
- [x] Resolve original paths/asset receipts, current GPU inventory, historical ledger without resetting prior; only hash changed assets. Record unavailable confirmation honestly after exposure reconciliation.
- [x] Implement independent strict parent/new-prefix loading, source/runtime and delta identity. Tests reject missing tensor or wrong source; no old-loader identity removal.

## Task 2 — Prediction-only regional evidence and model hook

Files: models/observation_evidence_state.py, models/observation_grounded_modules.py, models/observation_grounded.py, new numeric context in scripts/ogr_assets.py; tests/test_ogr_learning.py, tests/test_ogr_runtime.py.
- [x] First failing small-tensor tests for weighted pool gradients/no previous contribution, identity-initialized projections/null, equal feedback parameters and zero initialization.
- [x] Implement these exact equations from instruction§5; test actual counts84224/116993/182785/182785 and canonical new namespaces.
- [x] Add the subclass without editing old forward; retain intermediate graph before superclass cleanup; verify first feedback affects aux4+, empty-history noop and parent eval.
- [x] Sidecar test accepted birth/update/absence identity synchronization and detached persistence; use original confidence EMA rates.

## Task 3 — Data, constrained task/identity supervision and runtime

Files: datasets/observation_grounded_episode.py, models/observation_grounded_loss.py, trainer/observation_grounded_trainer.py, scripts/ogr_runtime.py; tests/test_ogr_data.py, test_ogr_learning.py, test_ogr_runtime.py.
- [x] Data tests distinguish local_pair/full_prefix/current counts and explicit CONFIRM scope. Reuse native loading and common causal projection; no future input.
- [x] Loss tests force one canonical source, ignore duplicates, preserve unknown remnant/empty GT; no-constraint same-RNG equality to original criterion including raw_sum keys.
- [x] Identity tests use grouped positives and unknown columns/null contracts at pre-feedback and final assignment. Verify gradient reaches mask/prompt, not frozen parent.
- [x] Partial-route tests cover slot contention/dummy/no history, full capacity ephemeral and shuffled source_query_ids. Construct original route hash/metadata/state commitments and commit once.
- [x] Implement update/resume exact cursor*4 and14 stages, AdamW/lr_scale; test one optimizer-boundary parameter/optimizer/RNG restore.

## Task 4 — Diagnostics, real probe and locked execution

Files: scripts/ogr_diagnostics.py, scripts/ogr_evaluation.py, scripts/ogr_campaign.py, scripts/ogr_selection.py; four focused test files only.
- [x] Implement and verify all required --help before logging execution. Charge integration checks and real34-input shared-history target diagnostics/anchor references.
- [x] Run four20-update probes,4 warmup; measure p90/startup and evaluation-stage costs without inspecting CAL metrics. Calibrate lambda from TRAIN gradient medians with specified fallback only.
- [x] Atomically lock largest feasible N and full forecast. If none fits, preserve all actual work and deliver explicit budget-blocked branch; no short fake formal run.
- [ ] For feasible N, execute246 four arms/four CAL points, lock endpoints/control before any new SEL/PB, execute247 OBS/control same N/four CAL points, then all44 jobs with precise per-job states and deduplication.
- [x] Shared reservation lock enforces run/lifetime/phase caps and at most2 idle single-GPU jobs; each failure/retry remains charged; same error at most one concrete fixed retry.

## Task 5 — Official statistics, interventions and resources

Files: scripts/ogr_statistics.py, scripts/ogr_diagnostics.py, scripts/ogr_evaluation.py, scripts/ogr_delivery.py.
- [x] Preserve global/per-reference original official states; verify CPU acceleration against two real pools once. Pair1000 reference draws seed9101 across comparisons, preserve NA, interval only≥950 valid.
- [ ] Run own-history deployment and fixed34-input shared-history ID/QFB/OBS comparison separately; off and same-class swap are paired, target-specific and fixed-order with actual coverage.
- [x] New-path finite repeat panel: two real fresh processes, six fixed inputs, covered T3/T4 differences0pp and discrete equality.
- [ ] Final-endpoint numeric and same-card seven-config profile: implemented, BUDGET_BLOCKED; no final endpoints exist.
- [x] Run valid locked CONFIRM only if exposure/budget permits; preserve independent-unknown status otherwise.

## Task 6 — Complete original-instruction review and actual delivery

Files: scripts/ogr_delivery.py, OUT all required files of instruction§15, three data-figure families, .gitattributes.
- [x] Export actual24 points/six final resumes when completed, selected aliases and missing reasons otherwise; load locked OBS/control public delta once each, bind original parent/encoder/taxonomy. Actual formal count is zero with explicit missing reasons; selected public delta reload is NA.
- [x] Verify all actual tables/states/archive bytes/counts, costs, source identities and four status dimensions. Write REPORT and twelve-item FINAL_REVIEW with evidence/limitations, no mechanical all-PASS.
- [x] Focused tests, import/compile/help checks and primary twelve-item review.
- [ ] A/B/tag and targeted privacy/manifest/remote checks are the publication step; actual outcome is in the external receipt, never prefilled here.
- [ ] Push selected LFS/branch/tag; one empty-cache remote fetch validates all final assets and peeled tag B; external receipt records actual Release capability/error. Final response full A/B, real links, costs, science and exact publication status.

Each task proceeds red→green→focused verification. New failures return to primary root-cause analysis; no full historical test suite or unbounded GPU diagnostics. The original instruction remains the completion checklist; this plan does not replace its requirements.
