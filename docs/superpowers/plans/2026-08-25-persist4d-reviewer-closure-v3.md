# Persist4D Reviewer Closure V3 Implementation Plan

> Execute sequentially. Scientific semantics, debugging, integration, stage
> gates, and final classification remain owned by the primary agent.

**Goal:** Produce a provenance-bound V3 reviewer-closure package that separates
baseline evidence, protocol factors, trajectory scoring, local task quality,
fresh identity diagnostics, and Oracle-ID headroom.

**Architecture:** Frozen official local candidates and raw observations feed the
existing task/identity evaluators. New code constructs exact-prefix data and
analysis channels around existing registered trackers; it does not change model
inference or local candidate semantics.

**Environment:** `conda run -n persist4d python`, PyTorch 2.6, CUDA 12.6,
official `stmetrics` 0.1.0, pytest, ruff, Git, and three NVIDIA A40 GPUs.

---

## Task 1: Freeze V3-0 Audit And Evidence Boundary

**Files:**
- Create: `docs/superpowers/specs/2026-08-25-persist4d-reviewer-closure-v3-design.md`
- Create: `docs/superpowers/plans/2026-08-25-persist4d-reviewer-closure-v3.md`
- Create: `tests/test_baseline_evidence_contract.py`
- Create: `scripts/build_baseline_evidence_contract.py`
- Create: `artifacts/reviewer_closure_v3/START_STATE.{json,md}`
- Create: `artifacts/reviewer_closure_v3/CODE_AUDIT.md`
- Create: `artifacts/reviewer_closure_v3/STATISTICAL_CONTRACT.md`
- Create: `artifacts/reviewer_closure_v3/baseline/*`

Write failing contract tests first. The builder must verify the checkpoint and
frozen report hashes, emit disjoint E0/E1/E2 rows, encode allowed/forbidden
claims, and bind the live official README audit. Record exact environment,
source revisions, V2/P2/P2R/Protocol-B hashes, baseline 59-pass status, and the
code ownership map. Run the new test, `git diff --check`, and commit V3-0.

## Task 2: Specify And Test Exact T2 Bridge

**Files:**
- Create: `tests/test_protocol_b_t2_bridge.py`
- Create: `scripts/build_protocol_b_t2_bridge.py`
- Create: `artifacts/reviewer_closure_v3/protocol_bridge/bridge_change_gt/*`
- Create: remaining PB0 bridge build artifacts

First test fixture-level first-two-scan slicing, first-transition extraction,
future-column rejection, source-record matching, and metadata validation. Then
test real data for 43 exact canonical records and 14 overlapping records. The
builder resolves source paths from manifests, writes only new V3 artifacts, and
refuses differing overwrite or any substituted/reversed pair.

Run:
`conda run -n persist4d python -m pytest -q tests/test_protocol_b_t2_bridge.py tests/test_p6a_protocol.py tests/test_audit_tmap_protocol_shift.py`

Require PB0 before committing V3-1 or starting evaluation.

## Task 3: Build One Frozen Bridge Runtime

**Files:**
- Create: `tests/test_protocol_bridge_evaluation.py`
- Create: `scripts/evaluate_protocol_bridge.py`
- Create: `scripts/analyze_protocol_bridge.py`

Write tests that prove full-154 and bridge-43 datasets use the same checkpoint,
collator, official post-processing, class map, metric spec, precision, and seed
control. The evaluator exposes seed 45/46/47 and accepts population/order/horizon
as data selection only. It writes resumable raw records with checkpoint/config/
protocol/script hashes and rejects mixed runtime bindings.

## Task 4: Execute PB1 And Analyze Protocol Factors

**Files:**
- Create: bridge evaluation CSV/report/manifest files required by the prompt

For each seed evaluate full-154 and exact-43 canonical T2 through the identical
runtime. Evaluate three T2 orders for 43 masters and T2-T5 exact prefixes using
the frozen Protocol-B cache where valid. Produce paired per-unit rows first,
then pooled descriptive and six-cluster summaries. Never compute an additive
34.8-to-19.1 decomposition. Validate PB1 and commit V3-2.

## Task 5: Add Explicit Score Reducers Test-First

**Files:**
- Create: `tests/test_system_comparison_v3_score_reducers.py`
- Modify: `scripts/system_comparison_v2_inference.py`

Write failing tests for one/multiple occurrences, gaps, reactivation, causal
prefix snapshots, ephemeral equality, and mask/class/key invariants. Implement
only `mean`, `latest`, and `max`; reject every other reducer. Re-run frozen V2
mean tests before any sensitivity analysis.

## Task 6: Implement Current-Local And Score Sensitivity Channels

**Files:**
- Create: `scripts/system_comparison_v3_score_sensitivity.py`
- Extend: `tests/test_system_comparison_v3_score_reducers.py`
- Create: `artifacts/reviewer_closure_v3/score_sensitivity/*`

Read latest-stage masks/classes/scores directly from the official sidecar and
evaluate with raw-local stmetrics. Assert byte/numeric equality across tracker
labels. Replay B2/B3/B4 under mean/latest/max on identical cache/targets. Emit
per-sequence before aggregate/per-cluster rows. Mean must exactly regress to V2;
latest/max may change only scores. Validate EV0 and commit V3-3.

## Task 7: Recompute Identity From Fresh Tracker Steps

**Files:**
- Create: `tests/test_system_comparison_v3_identity.py`
- Create: `scripts/system_comparison_v3_identity.py`
- Create: `artifacts/reviewer_closure_v3/identity/*`

Write tests that reject copied identity rows and GT-bearing inference inputs,
validate issued IDs, cluster grouping, and B4 regression. Use only
`build_tracker_factories()`, `build_association_events()`, and
`compute_deployment_identity_metrics()` for B2/B3/B4. Compare fresh B4 with V1
before accepting all new results. Report all six B4-minus-B2 cluster effects and
descriptive cluster bootstrap only. Validate ID0 and commit V3-4.

## Task 8: Add Post-Prediction Oracle-ID Diagnostic

**Files:**
- Create: `tests/test_system_comparison_v3_oracle_identity.py`
- Create: `scripts/system_comparison_v3_oracle_identity.py`
- Create: `artifacts/reviewer_closure_v3/oracle_identity/*`

Write failing tests proving prediction must exist before GT access, missing GT
fails explicitly, masks/classes/scores are invariant, linkage keys retain the
predicted class, and unmatched predictions remain ephemeral. Implement fixed
class-compatible Hungarian matching at IoU 0.5 after local prediction and
evaluate with primary mean. Validate OR0 and commit V3-5.

## Task 9: Synthesize And Verify V3

**Files:**
- Create: `artifacts/reviewer_closure_v3/FINAL_REVIEWER_CLOSURE_V3.md`
- Create: `artifacts/reviewer_closure_v3/FINAL_MANIFEST.json`

Generate Tables A-E only from hashed V3/frozen artifacts. Verify all manifests,
no frozen V1/V2 modifications, no copied V1 identity source, exact checkpoint
binding, and a clean diff. Run all six V3 tests, the nine focused regression
files, `ruff check .`, `git diff --check`, and the broader suite to the extent
mounted external assets allow. Inspect every changed file and assign exactly one
RC3 status from the preregistered logic. Commit synthesis only as V3-FINAL.
