# Persist4D MultiScan External Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, through fail-closed preregistered gates, whether the official MultiScan release supports a valid frozen Persist4D zero-shot external evaluation.

**Architecture:** Add a strict MultiScan data/GT boundary and thin audit/protocol/evaluation scripts while reusing the existing ReScan association, metric, bootstrap, and cache semantics. Execute metadata, gap, alignment, protocol, and coverage gates in order; create no downstream artifacts after the first failure.

**Tech Stack:** Python 3.10, NumPy, PyTorch, plyfile, SciPy, pytest, Ruff, JSON/CSV, Git content hashes.

---

### Task 1: Bind Frozen Sources And Official Inventory

**Files:**
- Create: `tests/test_multiscan_inventory.py`
- Create: `datasets/multiscan_adapter.py`
- Create: `scripts/audit_multiscan_dataset.py`
- Create: `artifacts/multiscan/repro_bindings.json`
- Create: `artifacts/multiscan/multiscan_inventory.json`
- Create: `artifacts/multiscan/longitudinal_subset_manifest.json`

- [ ] Write tests asserting 273 unique scan IDs, 117 scene groups, split
  preservation, deterministic numeric order, and an all-`T>=3` subset.
- [ ] Run `python -m pytest -q tests/test_multiscan_inventory.py`; expect import
  failure before implementation.
- [ ] Implement strict `parse_multiscan_scan_id()` and
  `build_multiscan_inventory()` with duplicate/mixed-split failures.
- [ ] Run the inventory test; expect all pass.
- [ ] Run the official CSV audit and assert 23/14/9 scenes for T>=3/4/5 and 101
  selected scans without hard-coding those counts in production code.
- [ ] Bind Persist4D commit/tree, checkpoint/config hashes, official MultiScan
  commit/tree, both official CSV hashes, and selected-list content hash.
- [ ] Commit with `git commit -m 'results: bind MultiScan longitudinal inventory'`.

### Task 2: Parse Release Identity And Natural Gaps

**Files:**
- Create: `tests/test_multiscan_identity_mapping.py`
- Create: `tests/test_multiscan_gap_detection.py`
- Modify: `datasets/multiscan_adapter.py`
- Modify: `scripts/audit_multiscan_dataset.py`
- Create: `artifacts/multiscan/MULTISCAN_IDENTITY_AUDIT.md`
- Create: `artifacts/multiscan/gap_opportunities.json`

- [ ] Write tests for local instance 2 and 8 mapping to stable object 17,
  removal labels, structural exclusions, missing annotations, and inconsistent
  release schemas.
- [ ] Write maximal-gap tests: `11001` produces one length-2 gap and `10101`
  produces two length-1 gaps.
- [ ] Run both test files; expect failures for missing parser/functions.
- [ ] Implement strict annotation parsing, `inst2obj_id` normalization, raw
  `objectId` fallback, stable scene-scoped identity, and maximal gap episodes.
- [ ] Run both tests; expect all pass.
- [ ] Inspect actual release payload keys and annotations; publish manual
  cross-scan identity examples only from real files.
- [ ] Require every one of the 101 selected scan annotations before computing
  the official gap count; partial data must exit nonzero without a gate result.
- [ ] If gaps <10 or gap scenes <3, publish `MULTISCAN_GAP_FAIL`, build the
  evidence manifest, skip Tasks 4-10, and proceed to Task 11.
- [ ] Commit with `git commit -m 'results: audit MultiScan stable identities and gaps'`.

### Task 3: Audit Chronology And Semantic Compatibility

**Files:**
- Create: `tests/test_multiscan_chronology.py`
- Modify: `datasets/multiscan_adapter.py`
- Modify: `scripts/audit_multiscan_dataset.py`
- Create: `artifacts/multiscan/chronology_audit.json`
- Create: `artifacts/multiscan/multiscan_to_rescene_label_map.json`
- Create: `artifacts/multiscan/MULTISCAN_DATASET_AUDIT.md`

- [ ] Write tests that distinguish timestamps from numeric dataset order and
  fail on partial/non-comparable timestamps.
- [ ] Run the chronology test; expect failure before implementation.
- [ ] Implement exact statuses `TRUE_CHRONOLOGY`, `DATASET_ORDER_ONLY`, and
  `UNRESOLVED`, with source fields for every decision.
- [ ] Enumerate all 20 official semantic IDs and map only identical ReScene
  class concepts as `exact`; retain `defensible`, `ambiguous`, or `unsupported`
  for every other entry.
- [ ] Run tests and validate JSON schemas/hashes.
- [ ] Commit with `git commit -m 'results: freeze MultiScan order and label audit'`.

### Task 4: Verify Official Alignment

**Condition:** Execute only after `MULTISCAN_GAP_PASS`.

**Files:**
- Create: `tests/test_multiscan_alignment.py`
- Create: `scripts/audit_multiscan_coordinates.py`
- Create: `artifacts/multiscan/MULTISCAN_ALIGNMENT_AUDIT.md`
- Create external: `/mnt/shared/ww/persist4d-multiscan/alignment-smoke/`

- [ ] Write a synthetic column-major current-to-reference transform test and
  invalid matrix/reference tests.
- [ ] Run the test; expect failure before implementation.
- [ ] Implement transform parsing and fixed NN/centroid/AABB/overlap metrics.
- [ ] Acquire only PLY, align JSON, annotations, and minimal metadata for one T3
  and one T4/T5 selected scene; do not acquire videos/depth.
- [ ] Generate before/after PLYs, calculate metrics, and manually inspect both.
- [ ] If alignment is ambiguous/worse, publish `MULTISCAN_ALIGNMENT_FAIL`, skip
  Tasks 5-10, and proceed to Task 11.
- [ ] Commit with `git commit -m 'results: verify MultiScan shared coordinates'`.

### Task 5: Enforce GT-Free Temporal Dataset

**Condition:** Execute only after alignment passes.

**Files:**
- Create: `tests/test_multiscan_no_gt_leakage.py`
- Create: `tests/test_multiscan_temporal_dataset.py`
- Modify: `datasets/multiscan_adapter.py`

- [ ] Write dataclass and recursive leakage tests for `objectId`, instance GT,
  semantic GT, part ID, mobility, OBB, and correspondence aliases.
- [ ] Write tests for `[S1]`/adjacent scan loading, geometry-only segments, and
  cross-scene request rejection.
- [ ] Run both tests; expect failure before implementation.
- [ ] Implement `MultiScanInferenceInput`, `MultiScanEvaluatorTarget`, strict
  input/target splitting, PLY parsing, and `MultiScanTemporalDataset`.
- [ ] Run both tests plus all ReScan adapter/leakage tests; expect all pass.
- [ ] Commit with `git commit -m 'feat: add GT-free MultiScan temporal adapter'`.

### Task 6: Freeze Protocol And Preflight Gates

**Files:**
- Create: `tests/test_multiscan_protocol.py`
- Create: `tests/test_multiscan_external_gate.py`
- Create: `scripts/multiscan_protocol.py`
- Create: `artifacts/multiscan/frozen_protocol.json`

- [ ] Write exact stage-window tests and mutations rejecting expanded windows,
  changed K/threshold/class weight/update rate, mixed scenes, or GT fields.
- [ ] Write boundary tests for 9/10 gaps, 2/3 scenes, and 0.0999/0.10 coverage.
- [ ] Run tests; expect failures before implementation.
- [ ] Implement content-addressed protocol construction and separate gap,
  alignment, protocol, and coverage gate functions.
- [ ] Run tests; expect all pass.
- [ ] Commit with `git commit -m 'feat: freeze MultiScan external protocol'`.

### Task 7: Run Frozen ReScene Coverage Smoke

**Condition:** Execute only after protocol passes.

**Files:**
- Create: `scripts/evaluate_multiscan_persist4d.py`
- Create: `artifacts/multiscan/observation_coverage_smoke.json`
- Create: `artifacts/multiscan/MULTISCAN_PREFLIGHT_REPORT.md`
- Create external: `/mnt/shared/ww/persist4d-multiscan/inference-cache/`

- [ ] Reuse the frozen checkpoint/config and ReScan batch preparation without
  modifying model, input normalization, or thresholds.
- [ ] Freeze 2-3 representative scenes before inference: at least one T3, one
  T4/T5, and one verified gap-bearing scene when possible.
- [ ] Run smoke inference and compute GT entity-stage candidate coverage at IoU
  0.25/0.50, exact-class coverage, raw AP, and raw recall before tracking.
- [ ] If clearly below 0.10, run only the predeclared small verification subset.
- [ ] If broader coverage remains below 0.10, publish
  `MULTISCAN_COVERAGE_FAIL`, skip Tasks 8-10, and proceed to Task 11.
- [ ] Otherwise publish `MULTISCAN_FULL_EVAL_GO`.
- [ ] Commit with `git commit -m 'results: gate frozen MultiScan coverage'`.

### Task 8: Run Full Frozen Evaluation

**Condition:** Execute only after `MULTISCAN_FULL_EVAL_GO`.

**Files:**
- Modify: `scripts/evaluate_multiscan_persist4d.py`
- Create: `artifacts/multiscan/full_eval/per_scene_results.csv`
- Create: `artifacts/multiscan/full_eval/aggregate_results.csv`
- Create: `artifacts/multiscan/full_eval/identity_results.csv`
- Create: `artifacts/multiscan/full_eval/gap_recovery_results.csv`
- Create: `artifacts/multiscan/full_eval/raw_local_results.csv`

- [ ] Freeze all 23 predeclared scenes/101 scans and refuse exclusions.
- [ ] Run one content-addressed frozen observation cache.
- [ ] Fan the cache into Pairwise Feature, Feature-Class, EMA, and Persist4D
  using `scripts/p6a_association.py` implementations only.
- [ ] Save every scene and opportunity count; condition class-aware outputs on
  exact semantic mappings only.
- [ ] Commit with `git commit -m 'results: run MultiScan zero-shot evaluation'`.

### Task 9: Compute Scene-Level Statistics And External Gate

**Condition:** Execute only after Task 8.

**Files:**
- Create: `artifacts/multiscan/full_eval/cluster_bootstrap.csv`
- Create: `artifacts/multiscan/full_eval/semantic_task_results.csv` if valid
- Create: `artifacts/multiscan/full_eval/EXTERNAL_VALIDATION_REPORT.md`
- Create: `artifacts/multiscan/full_eval/figures/` if supported

- [ ] Compare Persist4D to preregistered Feature-Class using physical-scene
  paired bootstrap, fixed seed 45, 10,000 replicates.
- [ ] Save mean, absolute, relative, CI, and all per-scene effects.
- [ ] Apply exactly one `EXTERNAL_SUPPORT`, `EXTERNAL_PARTIAL`,
  `EXTERNAL_INCONCLUSIVE`, or `EXTERNAL_CONTRADICTS` classification.
- [ ] Do not tune or select scenes after the result.
- [ ] Commit with `git commit -m 'results: classify MultiScan external transfer'`.

### Task 10: Conditional Full-History Control

**Condition:** Execute only after `EXTERNAL_SUPPORT` or an explicitly
interpretable external result; skip after inconclusive output.

- [ ] Reuse the same frozen collection/checkpoint and expanding `[S1,...,St]`
  history strategy.
- [ ] Keep tensors external and commit only content manifests/results.
- [ ] Report comparison without changing the primary gate.

### Task 11: Build Final Preflight Report And Evidence Manifest

**Files:**
- Create: `scripts/build_multiscan_artifacts.py`
- Modify: `artifacts/multiscan/MULTISCAN_PREFLIGHT_REPORT.md`
- Create: `artifacts/multiscan/evidence_manifest.json`
- Create: `tests/test_multiscan_artifacts.py`

- [ ] Write manifest tamper/missing-file and report-section tests first.
- [ ] Run the artifact test; expect failure before implementation.
- [ ] Implement SHA256/size binding for every repository artifact and external
  manifest, with immutable final-evidence/reviewer-closure tree assertions.
- [ ] Fill all ten required report sections and exactly one required decision;
  mark unavailable evidence as unverified, never as passing.
- [ ] Run all MultiScan tests, the existing 290-test frozen suite, Ruff,
  `git diff --check`, JSON/CSV validation, and the evidence verifier.
- [ ] Confirm no Git file over 50 MiB and no frozen artifact changes.
- [ ] Commit with `git commit -m 'results: freeze MultiScan preflight evidence'`.
- [ ] Push `research/persist4d-multiscan-preflight` and verify local, tracking,
  and remote SHA equality.
