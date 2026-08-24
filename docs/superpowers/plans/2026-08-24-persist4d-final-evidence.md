# Persist4D Final Evidence Implementation Plan

> Execute in order. Architecture, scientific semantics, debugging, integration,
> review, and final classification remain owned by the primary agent. Only
> explicitly specified mechanical leaves may be delegated.

**Goal:** Complete capacity evidence, independent Rescan validation where
defensible, and a provenance-bound paper freeze without changing Persist4D.

**Architecture:** Frozen local observations feed the unchanged persistent-memory
tracker under controlled capacity values. A separate Rescan parser/adapter feeds
the existing inference and evaluator contracts. All reports are derived from
hashed raw results and fail-closed gates.

**Environment:** `/home/ww/miniconda3/envs/persist4d/bin/python`, PyTorch,
Hydra, official stmetrics, pytest, Git, and official external repositories.

---

## Step 1: Bind The Isolated Source Tree

**Files:**
- Create: `artifacts/final_evidence/source_binding.json`
- Create: `artifacts/final_evidence/BASELINE_TEST_STATUS.md`

Record branch, HEAD, tree hash, reviewer-closure tree, remote, clean status,
environment versions, disk/GPU inventory, and the known baseline test outcome.
Verify no path under `artifacts/reviewer_closure/` changes.

## Step 2: Audit Persistent Capacity Semantics

**Files:**
- Create: `artifacts/final_evidence/CAPACITY_CODE_AUDIT.md`
- Test: `tests/test_final_capacity.py`

Read the complete state allocation/association/update path. First add failing
tests for free capacity, exact full capacity, rejected births, state shape and
byte accounting. Document slot lifecycle and timing boundaries with line-bound
source evidence. Do not edit `models/persistent_memory.py`.

Run:
`$PERSIST4D_PYTHON -m pytest -q tests/test_final_capacity.py tests/test_persistent_memory.py tests/test_p6a_memory_timing.py`

## Step 3: Locate And Bind Frozen Local Observations

**Files:**
- Create: `scripts/final_evidence_capacity.py`
- Create: `tests/test_final_capacity_replay.py`
- Create: `artifacts/final_evidence/capacity_observation_manifest.json`

Audit all reviewer-closure/system-comparison caches. Reuse an existing cache if
it contains local observation tensors and targets with exact source/checkpoint/
config bindings. Otherwise materialize one content-addressed sidecar from the
frozen local prediction cache. Add digest equality and exact-coverage tests
before implementation.

## Step 4: Implement Controlled Multi-Capacity Replay

**Files:**
- Modify: `scripts/final_evidence_capacity.py`
- Modify: `tests/test_final_capacity_replay.py`
- Create: `configs/final_evidence/capacity.yaml`

Replay identical observations at capacities 64, 100, 128, 160, and 200. Record
per-stage state, births, rejected births, identity events, sufficient official
metric state, state bytes, and timing. Refuse mismatched observations or model
semantics.

Run:
`$PERSIST4D_PYTHON -m pytest -q tests/test_final_capacity_replay.py`

## Step 5: Run Capacity Evaluation And Statistics

**Files:**
- Create: `artifacts/final_evidence/capacity_raw.json`
- Create: `artifacts/final_evidence/capacity_per_sequence.csv`
- Create: `artifacts/final_evidence/capacity_results.csv`
- Create: `artifacts/final_evidence/capacity_cluster_bootstrap.csv`

Execute T2--T5 on the full frozen eligible population, then compute all prompt
metrics and scene-cluster uncertainty. Keep K=100 as the main configuration
unless the final gate explicitly reopens it.

## Step 6: Publish Capacity Figures And Gate

**Files:**
- Create: `scripts/final_evidence_figures.py`
- Create: `tests/test_final_evidence_figures.py`
- Create: `artifacts/final_evidence/figures/capacity_occupancy_horizon.{pdf,png,svg}`
- Create: `artifacts/final_evidence/figures/capacity_performance.{pdf,png,svg}`
- Create: `artifacts/final_evidence/figures/capacity_state_bytes.{pdf,png,svg}`
- Create: `artifacts/final_evidence/CAPACITY_SENSITIVITY_REPORT.md`
- Create: `artifacts/final_evidence/capacity_gate.json`

Answer Q1--Q5 and emit exactly one capacity gate. Validate plot source hashes,
labels, cap lines, and statistics. Stop if `CAPACITY_CONFIG_REOPEN`.

## Step 7: Pin And Audit Official Rescan Source

**Files:**
- Create: `artifacts/final_evidence/external/rescan_source_manifest.json`
- Create: `artifacts/final_evidence/RESCAN_DATASET_AUDIT.md`

Clone official `mhalber/Rescan` outside Git, pin the exact commit, inventory its
licenses/scripts/protocol/metadata expectations, and record source hashes.

## Step 8: Acquire And Inventory Official Rescan Data

**Files:**
- Create: `scripts/audit_rescan_dataset.py`
- Test: `tests/test_rescan_dataset_audit.py`
- Create: `artifacts/final_evidence/external/rescan_dataset_manifest.json`
- Modify: `artifacts/final_evidence/RESCAN_DATASET_AUDIT.md`

Store data under `/mnt/shared/ww/persist4d-final-evidence/rescan`. Record all
available scenes/captures, files, sizes, hashes, chronology evidence, stable IDs,
semantic labels, ambiguities, coordinate frames, and transforms. Do not infer
chronology from filenames without official evidence.

## Step 9: Implement The Rescan Parser

**Files:**
- Create: `datasets/rescan_adapter.py`
- Test: `tests/test_rescan_adapter.py`

Write failing fixture tests for XYZ, normals, RGB, semantic/instance IDs,
deterministic order, stable identity, malformed fields, and real sampled files.
Implement only the smallest parser needed by the frozen inference contract.

## Step 10: Implement Ambiguity Handling

**Files:**
- Modify: `datasets/rescan_adapter.py`
- Modify: `tests/test_rescan_adapter.py`

Encode official ambiguity alternatives. Test one explicit ambiguous case and
reject unregistered alternatives.

## Step 11: Freeze The Label Map

**Files:**
- Create: `artifacts/final_evidence/rescan_to_rescene_label_map.json`
- Test: `tests/test_rescan_label_map.py`

Map every encountered label with `exact`, `reasonable`, `ambiguous`, or
`unsupported` status and primary-source provenance. Exclude unsupported labels
from class-dependent metrics. Never choose an ambiguous map silently.

## Step 12: Audit Coordinates And GT Boundaries

**Files:**
- Create: `artifacts/final_evidence/RESCAN_COORDINATE_AUDIT.md`
- Modify: `datasets/rescan_adapter.py`
- Create: `tests/test_rescan_no_gt_leakage.py`

Bind official scan transforms and assert inference batches contain no object GT,
stable identity, ambiguity, or object transform fields. Permit those fields only
inside post-inference evaluator targets.

## Step 13: Build The External Protocol

**Files:**
- Create: `scripts/rescan_protocol.py`
- Create: `tests/test_rescan_protocol.py`
- Create: `configs/final_evidence/rescan.yaml`
- Create: `artifacts/final_evidence/external/rescan_protocol.json`

Select actual ordered sequences when documented. Otherwise record deterministic
metadata order as non-chronological. Define natural gap opportunities, Level A
eligibility, Level B stable-identity eligibility, scene clusters, and exact local
pair inputs.

## Step 14: Build External Baselines And Evaluator

**Files:**
- Create: `scripts/evaluate_rescan_persist4d.py`
- Create: `tests/test_rescan_evaluator.py`

Add Pairwise Feature, conditionally valid Feature-Class, EMA, and Persist4D using
existing association implementations. Add Full-History only if the frozen model
can execute the same inputs. Reuse official metric and identity accumulators.

## Step 15: Run External Smoke Evaluation

**Files:**
- Create: `artifacts/final_evidence/external/rescan_smoke.json`

Run one eligible scene end to end. Validate parser, memory persistence, metric
state, no-GT-leakage evidence, runtime, and artifact atomicity before scaling.

## Step 16: Run Full Eligible External Evaluation

**Files:**
- Create: `artifacts/final_evidence/external/rescan_raw.json`
- Create: `artifacts/final_evidence/external/rescan_per_scene.csv`
- Create: `artifacts/final_evidence/external/rescan_results.csv`
- Create: `artifacts/final_evidence/external/rescan_scene_bootstrap.csv`

Run all eligible scenes and every registered baseline. Report gap recovery as
primary identity evidence and class/task metrics only for valid mapped subsets.

## Step 17: Gate External Evidence

**Files:**
- Create: `artifacts/final_evidence/EXTERNAL_VALIDATION_REPORT.md`
- Create: `artifacts/final_evidence/external_gate.json`

Emit exactly one external gate and make limitations explicit. Stop if
`EXTERNAL_CONTRADICTS`.

## Step 18: Bounded Official Rescan Method Audit

**Files:**
- Create: `artifacts/final_evidence/OFFICIAL_RESCAN_CODE_AUDIT.md`

Audit in a separate environment with a fixed time/effort budget. Reproduce only
when assets and protocol are directly executable. Otherwise record exactly
`RESCAN_METHOD_NOT_REPRODUCED` and continue.

## Step 19: Complete LivingScenes Positioning

**Files:**
- Create: `artifacts/final_evidence/LIVINGSCENES_RELATED_WORK_NOTE.md`

Pin primary paper/repository metadata and write conceptual positioning only.
Do not run or integrate LivingScenes unless every prompt prerequisite passes.

## Step 20: Audit Published Baselines

**Files:**
- Create: `artifacts/final_evidence/PUBLISHED_BASELINE_AUDIT.md`
- Create: `artifacts/final_evidence/tables/reported_standard_protocol.csv`
- Create: `artifacts/final_evidence/tables/controlled_common_prefix.csv`
- Create: `artifacts/final_evidence/tables/external_rescan.csv` when valid

Use only official papers or repositories. Separate reported standard-protocol
values from values actually recomputed under the common-prefix protocol.

## Step 21: Freeze Novelty And Claims

**Files:**
- Create: `artifacts/final_evidence/NOVELTY_BOUNDARY.md`
- Create: `artifacts/final_evidence/claim_gates.json`

Define the exact boundary against Rescan, ReScene, and LivingScenes. Evaluate
claim gates A--E from artifacts. Remove unsupported priority, t-mAP, external
generalization, and architecture claims.

## Step 22: Build Paper Figures And Tables

**Files:**
- Modify: `scripts/final_evidence_figures.py`
- Create: `artifacts/final_evidence/figures/figure1_context_vs_state.{pdf,png,svg}`
- Create: `artifacts/final_evidence/figures/figure2_capacity_scaling.{pdf,png,svg}`
- Create: `artifacts/final_evidence/figures/figure3_failure_decomposition.{pdf,png,svg}`
- Create: `artifacts/final_evidence/figures/figure4_external_validation.{pdf,png,svg}` when valid
- Create: `artifacts/final_evidence/tables/table1_main.csv`
- Create: `artifacts/final_evidence/tables/table2_identity.csv`
- Create: `artifacts/final_evidence/tables/table3_capacity.csv`
- Create: `artifacts/final_evidence/tables/table4_external.csv` when valid

Reuse the frozen decomposition rather than regenerating reviewer-closure files.
Run visual schema tests and inspect every rasterized figure.

## Step 23: Publish Reproducibility Package

**Files:**
- Create: `artifacts/final_evidence/final_evidence_manifest.json`
- Create: `artifacts/final_evidence/REPRODUCIBILITY.md`
- Create: `scripts/verify_final_evidence.py`
- Test: `tests/test_final_evidence_artifacts.py`

Bind every code/config/data/source/evaluator hash. Verify that Git contains no
raw data, checkpoints, private absolute paths, or unbound numeric results.

## Step 24: Final Audit, Classification, Commit, And Push

**Files:**
- Create: `artifacts/final_evidence/FINAL_PAPER_EVIDENCE_REPORT.md`

Audit every prompt requirement, run all targeted tests and available relevant
regressions, compile Python, inspect the full diff, verify the frozen directory
is unchanged, and run the artifact verifier. Emit exactly one final paper
classification, then stop experimental work. Commit the verified package and
push `research/persist4d-final-evidence` to GitHub.

Final commands:

```bash
$PERSIST4D_PYTHON -m pytest -q \
  tests/test_final_capacity.py \
  tests/test_final_capacity_replay.py \
  tests/test_rescan_adapter.py \
  tests/test_rescan_dataset_audit.py \
  tests/test_rescan_label_map.py \
  tests/test_rescan_no_gt_leakage.py \
  tests/test_rescan_protocol.py \
  tests/test_rescan_evaluator.py \
  tests/test_final_evidence_figures.py \
  tests/test_final_evidence_artifacts.py
$PERSIST4D_PYTHON -m compileall -q datasets scripts tests
$PERSIST4D_PYTHON scripts/verify_final_evidence.py artifacts/final_evidence
git diff --check
git status --short
git push -u origin research/persist4d-final-evidence
```
