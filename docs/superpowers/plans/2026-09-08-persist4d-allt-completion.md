# Persist4D All-T Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every remaining evidence, profiling, publication, and handoff requirement in `Persist4D_Codex_AllT_Task_Superiority_V1.md` without changing the frozen checkpoint, primary reducer, Protocol-B outputs, or failed scientific verdict.

**Architecture:** Extend the checkpoint evaluator with two explicitly diagnostic memory-read policies whose cache identities cannot collide with the frozen main run. Add one streaming offline analyzer for per-reference official metrics, fresh identity recomputation, and equal-cluster descriptive intervals; add one bounded real-GPU profiler; then add one fail-closed publisher that derives the required compact artifacts and reports from the immutable evidence.

**Tech Stack:** Python 3.10, PyTorch, stmetrics, pytest, Ruff, CSV/JSON/YAML artifacts, Git/GitHub, NVIDIA A40.

**Spec:** `docs/Persist4D_Codex_AllT_Task_Superiority_V1.md`

## Global Constraints

- The frozen candidate remains C2 update 200 with checkpoint SHA256 `a4adb8ae1bc25830934a97926b7910d5f7ecde3b010cf877475cec82bff30724`.
- The matched comparator remains FH-adapt update 100 with checkpoint SHA256 `ddfda362673ce081dab8d7f790ffa223c336921ad6f85220aa3e6a78bca1a70c`.
- The primary reducer remains `mean`; `latest` and `max` remain sensitivity analyses only.
- Protocol-B remains final-only and must not affect architecture, checkpoint, reducer, or hyperparameter selection.
- The primary verdict remains based on unrounded strict `delta > 0` at every T in `{2,3,4,5}`.
- No GT may enter prediction, memory state, association, or profile execution.
- New large caches remain outside Git under `$ALLT_EXTERNAL_ROOT`; Git contains only compact manifests, results, commands, and hashes.
- The existing historical R1/V3 artifact directories remain read-only.
- Seed 46 remains gate-skipped because the first-seed development comparison against matched FH-adapt was not all-T positive.
- C3 and FH-L remain gate-skipped under their frozen preregistered gates.
- All production behavior changes follow test-first red-green-refactor.

---

### Task 1: Diagnostic memory-read policies

**Files:**
- Modify: `scripts/evaluate_persist4d_allt.py`
- Modify: `tests/test_persist4d_allt_evaluation.py`
- Create after execution: `artifacts/allt_task_superiority_v1/evaluation/ablation/C2-memory-off/update=0200/{all_t_metrics.csv,cache_manifest.json,run_summary.json}`
- Create after execution: `artifacts/allt_task_superiority_v1/evaluation/ablation/C2-previous-only/update=0200/{all_t_metrics.csv,cache_manifest.json,run_summary.json}`

**Interfaces:**
- Consumes: `DetachedMemoryReadState`, the selected C2 checkpoint, development population, and the existing continuous T1-T5 evaluator.
- Produces: `apply_memory_read_policy(state, policy)` and `run_evaluation(..., memory_read_policy)` where policy is `all_occupied`, `disabled`, or `active_previous`.

- [x] **Step 1: Write failing policy tests**

Add tests that require `disabled` to preserve tensor metadata while making `occupied_mask` and `active_mask` empty, require `active_previous` to expose exactly the previous state's active slots, reject unknown policies, and require non-default policies to change the module-config SHA while leaving the default SHA unchanged.

- [x] **Step 2: Run the policy tests and observe the missing-interface failure**

Run: `python -m pytest -q tests/test_persist4d_allt_evaluation.py -k memory_read_policy`

Expected: collection or assertion failure because the policy interface does not exist.

- [x] **Step 3: Implement the minimal policy transformation and cache binding**

Pass the transformed read state only into model inference. Continue the original prediction-driven B4 state update for all policies. Add the non-default policy to the module-config document, bind the fixed-field raw provenance and cache key through the resulting config hash, and record it in the cache manifest, model name, run summary, and CLI; preserve the existing default module document byte-for-byte.

- [x] **Step 4: Run direct and adjacent evaluator tests**

Run: `python -m pytest -q tests/test_persist4d_allt_evaluation.py tests/test_persist4d_allt_model.py`

Expected: all tests pass.

- [x] **Step 5: Commit the evaluator change before generating diagnostic predictions**

Run: `git add scripts/evaluate_persist4d_allt.py tests/test_persist4d_allt_evaluation.py docs/superpowers/plans/2026-09-08-persist4d-allt-completion.md && git commit -m 'add frozen memory read diagnostics'`

- [x] **Step 6: Run both policies on the 47-sequence development population**

Use the committed source, evaluation seed 45, selected C2 checkpoint, `mean` reducer, and separate external cache roots. Run `disabled` and `active_previous` concurrently only on confirmed-free GPUs. Record all 47 sequences; do not run either policy on Protocol-B.

- [x] **Step 7: Validate and commit both diagnostic result triplets**

Require status `pass`, source commit equality, one checkpoint across T2-T5, 47 sequences, 235 new forwards, four official metric rows, and non-colliding module-config/cache hashes.

### Task 2: Streaming final cache analysis

**Files:**
- Create: `scripts/analyze_persist4d_allt_final.py`
- Create: `tests/test_analyze_persist4d_allt_final.py`
- Create after execution: `artifacts/allt_task_superiority_v1/results/per_reference.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/results/identity_counts.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/results/cluster_effects.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/results/final_analysis_manifest.json`

**Interfaces:**
- Consumes: the validated 129 C2 cache bundles, 129 FH-adapt bundles, their manifests, official `AllTBaselineAccumulator`, fresh B4 replay, and official identity matching.
- Produces: `build_paired_reference_rows`, `aggregate_identity_rows`, `bootstrap_equal_cluster_effect`, and `analyze_final_caches`.

- [x] **Step 1: Write failing pure-analysis tests**

Test exact 6-reference by 4-horizon by 5-metric paired coverage, numerator/denominator aggregation with `None` for zero denominators, deterministic 1000-resample equal-cluster intervals at seed 45, and rejection of duplicate or mismatched model/reference cells.

- [x] **Step 2: Run the new tests and observe missing-interface failures**

Run: `python -m pytest -q tests/test_analyze_persist4d_allt_final.py`

Expected: collection failure because the analyzer does not exist.

- [x] **Step 3: Implement fail-closed streaming analysis**

Load one cache bundle at a time. For C2 use only the frozen `mean` pair for task metrics and fresh B4 replay from raw observations for identity. For FH-adapt use the official pair and official full-history identity payloads. Compute official pooled metrics separately inside each reference cluster; never average sequence AP as pooled AP.

- [x] **Step 4: Emit explicit descriptive uncertainty**

Build 1000 fixed-seed bootstrap resamples over six equal-weight reference-cluster deltas and label the output `equal_cluster_effect`, not pooled AP confidence intervals or significance tests.

- [x] **Step 5: Run unit and adjacent metric tests**

Run: `python -m pytest -q tests/test_analyze_persist4d_allt_final.py tests/test_system_comparison_metrics.py tests/test_persist4d_allt_analysis.py`

Expected: all tests pass.

- [x] **Step 6: Run the analyzer on a high-memory CPU node**

Use the two frozen Protocol-B cache manifests and external cache directories. Require 129 distinct units per model, six references, exact T1-T5 identity updates, 120 paired reference task rows, eight identity aggregate rows, and 20 equal-cluster effect rows.

- [x] **Step 7: Validate result hashes and commit compact outputs**

Require the analysis manifest to bind both checkpoint SHAs, both cache-manifest hashes, output hashes, source commit, row counts, and `status=pass`.

### Task 3: Bounded real-GPU resource profile

**Files:**
- Create: `scripts/profile_persist4d_allt.py`
- Create: `tests/test_profile_persist4d_allt.py`
- Create after execution: `artifacts/allt_task_superiority_v1/profile/samples.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/profile/summary.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/profile/run_summary.json`

**Interfaces:**
- Consumes: selected C2 and matched FH-adapt checkpoints, Protocol-B sequence builder, one canonical first-sorted master per reference, CUDA timing, and peak allocated-memory counters.
- Produces: `select_profile_sequences`, `validate_profile_samples`, `summarize_profile_samples`, and `run_profile`.

- [x] **Step 1: Write failing profile-contract tests**

Test deterministic selection of exactly one canonical master per each of six references, exact 2-method by 6-reference by 4-horizon by 10-repeat coverage, rejection of missing repeats, and deterministic median/min/max/peak-memory summaries.

- [x] **Step 2: Run the tests and observe missing-interface failures**

Run: `python -m pytest -q tests/test_profile_persist4d_allt.py`

Expected: collection failure because the profiler does not exist.

- [x] **Step 3: Implement the bounded profile scope**

Preload and transfer each profile cell before timing. For C2, preroll the causal state outside the timer and time current model forward, one memory read, and one B4 state update. For FH-adapt, time one full-prefix forward. Exclude file I/O, collation, H2D, metric computation, and preroll. Record window voxel-point and segment counts.

- [x] **Step 4: Implement warmup, measurement, and summary validation**

Run five warmups and ten measured repeats for every cell. Emit exactly 480 sample rows and 48 summary rows. Bind device name, checkpoint, source commit, inclusion/exclusion scope, and population identity.

- [x] **Step 5: Run profile unit tests and a one-cell GPU smoke**

Run: `python -m pytest -q tests/test_profile_persist4d_allt.py`

Then run the profiler's smoke mode with one reference, one horizon, one warmup, and one measured repeat; require finite positive latency and memory values.

- [x] **Step 6: Run the formal profile on one otherwise-idle A40**

Use the same physical GPU sequentially for both methods. Do not run competing work on that node. Require five warmups and ten repeats for all 48 cells.

- [x] **Step 7: Validate and commit profile artifacts**

Require exact coverage, no nonfinite or nonpositive latency, stable checkpoint identities, and an explicit statement that speed does not alter the failed accuracy verdict.

### Task 4: Compact training and ablation exports

**Files:**
- Create: `scripts/publish_persist4d_allt.py`
- Create: `tests/test_publish_persist4d_allt.py`
- Create after execution: `artifacts/allt_task_superiority_v1/training/variant_matrix.json`
- Create after execution: `artifacts/allt_task_superiority_v1/training/formal/{C0,C1,C2,FH-adapt}/learning_curve.csv`
- Create after execution: `artifacts/allt_task_superiority_v1/results/memory_read_ablation.csv`

**Interfaces:**
- Consumes: frozen selection, training summaries/configs/manifests, development evaluations, memory diagnostics, and existing final results.
- Produces: deterministic variant matrix, four learning curves, and an ablation table that never promotes diagnostic policies into selected models.

- [x] **Step 1: Write failing publisher aggregation tests**

Test variant states for C0/C1/C2/FH-adapt complete and C3/FH-L gate-skipped, exact five-update learning curves, frozen selected-update flags, and three-policy C2 ablation rows over T2-T5.

- [x] **Step 2: Run the tests and observe missing-interface failures**

Run: `python -m pytest -q tests/test_publish_persist4d_allt.py`

Expected: collection failure because the publisher does not exist.

- [x] **Step 3: Implement deterministic compact exports**

Derive all values from existing manifests and CSV files. Preserve actual `training/formal/<variant>` paths in provenance. Mark C3, FH-L, and seed 46 as gate-skipped with their frozen reasons; never emit numeric rows for unrun variants.

- [x] **Step 4: Run publisher tests**

Run: `python -m pytest -q tests/test_publish_persist4d_allt.py tests/test_persist4d_allt_selection.py tests/test_finalize_persist4d_allt.py`

Expected: all tests pass.

### Task 5: Final report, manifest, and handoff

**Files:**
- Modify: `scripts/publish_persist4d_allt.py`
- Modify: `tests/test_publish_persist4d_allt.py`
- Create after execution: `artifacts/allt_task_superiority_v1/TEST_REPORT.md`
- Create after execution: `artifacts/allt_task_superiority_v1/FINAL_REPORT.md`
- Create after execution: `artifacts/allt_task_superiority_v1/FINAL_MANIFEST.json`
- Create after execution: `artifacts/allt_task_superiority_v1/HANDOFF.md`

**Interfaces:**
- Consumes: every required compact artifact and status JSON under `artifacts/allt_task_superiority_v1`.
- Produces: `validate_completion_inputs` and `publish_final_package` with a non-self-referential manifest.

- [x] **Step 1: Write failing completion-gate tests**

Test the eight required status fields, exact failed t-mAP cells, absence of fake numeric evidence for unrun stages, report/HANDOFF hashes in the manifest, external checkpoint/cache references, executable reproduction commands, and rejection of missing required artifacts.

- [x] **Step 2: Run the completion tests and observe expected failures**

Run: `python -m pytest -q tests/test_publish_persist4d_allt.py -k 'completion or handoff or manifest'`

Expected: failures until all report contracts are implemented.

- [x] **Step 3: Implement the report and handoff from structured evidence**

Report `execution_status=COMPLETE`, `baseline_comparability=MATCHED`, both t-mAP verdicts, all-task verdict, seed status, independent-generalization limit, publication state, every variant's run/gate state, full training exposure, all four T values, memory diagnostic interpretation, per-reference effects, identity counts, profile scope, direct tests, external storage reconstruction, and exactly one next-round bottleneck recommendation.

- [x] **Step 4: Generate FINAL_MANIFEST after report and HANDOFF**

Hash every required compact artifact except `FINAL_MANIFEST.json` itself. Record `code_commit_at_run`, use `publication_commit=resolve from remote HEAD`, and avoid embedding a self-invalidating final commit SHA.

- [x] **Step 5: Run the publisher twice and require byte-identical outputs**

Run the publisher, hash all generated files, run it again, and require identical hashes. Then run all new and directly adjacent tests plus Ruff only on changed Python files.

### Task 6: Final audit and GitHub publication

**Files:**
- Modify only when validation finds a concrete defect in Task 1-5 files.

**Interfaces:**
- Consumes: the full spec checklist, final compact artifacts, test output, Git state, and remote GitHub branch.
- Produces: verified remote publication receipt in the final user response.

- [x] **Step 1: Audit every explicit spec deliverable against authoritative files**

Check required filenames, row counts, checkpoint identities, all status fields, negative-result preservation, diagnostic labels, external storage bindings, and commands.

- [x] **Step 2: Run final verification**

Run direct pytest suites, Ruff on changed Python, `git diff --check`, artifact hash validation, and compact path/privacy scans that reject credentials and machine-specific paths in published docs.

- [x] **Step 3: Commit only explicit task files and push**

Use staged file lists, commit the final package, push `research/persist4d-allt-task-superiority-v1`, and require local HEAD equality with remote HEAD.

- [x] **Step 4: Read back HANDOFF and FINAL_MANIFEST from the final Git commit**

Use `git show <remote-head>:<path>` for both files, compare bytes and hashes with local files, and report `publication_status=PUSH_VERIFIED` only after both comparisons pass.

### Task 7: Completion-audit closure

**Files:**
- Modify: `scripts/system_comparison_analysis.py`
- Modify: `scripts/analyze_persist4d_allt_final.py`
- Modify: `scripts/publish_persist4d_allt.py`
- Modify: `tests/test_system_comparison_analysis.py`
- Modify: `tests/test_analyze_persist4d_allt_final.py`
- Modify: `tests/test_publish_persist4d_allt.py`
- Regenerate: compact identity analysis, final reports, and manifests

- [x] **Step 1: Add failing attempt-coverage contract tests**

Require `gap_recovery_attempt_coverage = recovery_attempts / gap_opportunities`, preserve `N/A` for a zero denominator, and expose the value in the final identity table.

- [x] **Step 2: Implement the metric and regenerate compact identity evidence**

Add the derived field to the shared identity aggregation and final-analysis schema. Recompute the compact CSV and its manifest without changing the frozen underlying counts.

- [x] **Step 3: Run the complete directly relevant test suite**

Include contract, dataset sequence/causality, model/read, trainer/training-gradient, evaluation, diagnostics, analysis, profile, selection, publisher, and finalizer tests. Record the exact observed count in `TEST_REPORT.md`.

- [ ] **Step 4: Regenerate, verify, commit, push, and read back**

Run the publisher twice with byte-identical outputs; run Ruff, `git diff --check`, artifact hash validation, then push and verify the remote HEAD plus final artifact bytes.
