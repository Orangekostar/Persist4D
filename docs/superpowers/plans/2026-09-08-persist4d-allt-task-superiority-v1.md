# Persist4D All-T Task Superiority V1 Implementation Plan

> Execute this plan in the isolated worktree on branch
> `research/persist4d-allt-task-superiority-v1`. The merged user prompt and the
> design document are binding. Each production behavior follows RED, GREEN,
> then focused regression verification.

## Task 1: Freeze S0 evidence contracts

**Files:**
- Create: `scripts/persist4d_allt_contract.py`
- Create: `tests/test_persist4d_allt_contract.py`
- Create: `artifacts/allt_task_superiority_v1/EXPERIMENT_CONTRACT.md`
- Create: `artifacts/allt_task_superiority_v1/run_contract.json`
- Create: `artifacts/allt_task_superiority_v1/source_map.md`
- Create: `artifacts/allt_task_superiority_v1/split_manifest.json`
- Create: `artifacts/allt_task_superiority_v1/budget_and_schedule.json`

1. Write tests for exact R1/protocol/pretrained/metadata identities, path-safe
   external bindings, deterministic 8/36 reference hashing, role disjointness,
   no processed-test claim, provisional-to-frozen budget transition, and
   canonical JSON hashes.
2. Run the test module and verify failures are caused by the missing contract
   module.
3. Implement the smallest fail-closed contract builder and CLI.
4. Generate the five S0 artifacts from live inputs. Record Protocol-B as final
   only and the train-derived holdout as R1-base-exposed development.
5. Run the contract tests, JSON decoding, `git diff --check`, then commit S0
   before any candidate metrics are produced.

## Task 2: Extend frozen baseline analysis to every T

**Files:**
- Create: `scripts/analyze_persist4d_allt.py`
- Create: `tests/test_persist4d_allt_analysis.py`
- Create: `artifacts/allt_task_superiority_v1/baseline/all_t_metrics.csv`
- Create: `artifacts/allt_task_superiority_v1/baseline/replay_status.json`

1. Write literal-fixture tests for T3 aggregation, T2/T4/T5 byte-level metric
   regression, official prefix-overall evaluation inputs, reducer reuse, and
   missing-key-only cache planning.
2. Verify RED because the new parameterized analyzer is absent.
3. Implement analysis by reusing audited R1 cache readers and official
   evaluators. Never modify old constants, reports, or caches.
4. Run it over the 645 local and 645 FullHistory entries. Recompute only absent
   derived keys, output all T=2,3,4,5 rows, and verify old reported numbers.
5. Run targeted analysis and R1 regression tests.

## Task 3: Run the bounded S1 failure diagnostic

**Files:**
- Create: `scripts/diagnose_persist4d_allt.py`
- Create: `tests/test_persist4d_allt_diagnostics.py`
- Create: `artifacts/allt_task_superiority_v1/diagnostics/failure_summary.csv`
- Create: `artifacts/allt_task_superiority_v1/diagnostics/final_r1_decoder_summary.json`

1. Write tests for reference-sorted panel selection capped at 12, official-valid
   denominators, multi-label failure categories, decoder execution-stage
   accounting, and the mutually exclusive L gate.
2. Verify RED, implement the diagnostic, and use only fixed selection rules.
3. Run the diagnostic on the epoch-390 R1 checkpoint and frozen cache. Freeze L
   as QCL-inspired, first-mask relaxation, or `NOT_JUSTIFIED` exactly as the
   diagnostic gate dictates.
4. Record no-candidate, insufficient-mask, class-change, fragmentation, merge,
   FH-success/B4-failure, T1 cold-start, query competition, utilization, and
   earliest-attention statistics with explicit valid counts.
5. Run diagnostic and R1 compatibility tests.

## Task 4: Implement the differentiable memory read

**Files:**
- Create: `models/persistent_memory_read.py`
- Create: `tests/test_persistent_memory_read.py`

1. Write tests whose independent literals cover `[B,Q,128]` and `[B,K,128]`
   validation, K=Q=100, empty-memory exact identity, occupied and dormant reads,
   null abstention, all-masked safety, finite diagnostics, zero output
   initialization, two-step gradient reachability, and detached state writes.
2. Verify RED because the module is absent.
3. Implement projected attention, abstain token, gate, residual, diagnostics,
   and a detached state container without GT fields.
4. Run unit tests on CPU and CUDA, then existing persistent-memory tests.

## Task 5: Add the decoder hook and Persist4DAllT model

**Files:**
- Modify: `models/rescene.py`
- Create: `models/persist4d_allt.py`
- Create: `tests/test_persist4d_allt_model.py`

1. Write tests for the exact hook location, separate execution and shared
   parameter indices, M read once after the first full hlevel pass, decoder-L-M
   order, base/off-path bitwise outputs and RNG state, strict R1 subtree load,
   named-only missing keys, and mode restoration.
2. Verify RED against the unchanged ReScene model.
3. Add a parameter-free identity hook and execution counter to ReScene. Add the
   new subclass and only the diagnostic-authorized L implementation.
4. Load the real R1 checkpoint strictly into the inherited subtree; reject any
   unexpected or unnamed missing key.
5. Run model tests plus query-feature and persistent-memory regressions.

## Task 6: Build causal sequence episodes and exact objectives

**Files:**
- Create: `datasets/persist4d_sequence_dataset.py`
- Create: `trainer/persist4d_allt_trainer.py`
- Create: `tests/test_persist4d_sequence_dataset.py`
- Create: `tests/test_persist4d_allt_trainer.py`

1. Write dataset tests for reference/split isolation, real H only, T1-alone,
   later adjacent-pair inputs, local temporal coordinates, shared augmentation
   draws, future-independent statistics, deterministic draw plans, and GT/state
   type separation.
2. Write trainer tests for no intra-episode optimizer step, coefficients for
   H=2..5, coefficient sum one, exact R1 raw-sum stage loss, state timing,
   encoder frozen/eval, mode restoration, and optimizer membership.
3. Verify both modules fail because they are absent.
4. Implement the sequence wrapper and Lightning-compatible episode trainer.
   Preserve the existing collate and criterion boundaries rather than copying
   evaluator semantics.
5. Run the two new modules' tests and temporal-loader/trainer regressions.

## Task 7: Configure and pass the real two-update smoke

**Files:**
- Create: `configs/persist4d_allt/default.yaml`
- Create: `scripts/train_persist4d_allt.py`
- Create: `tests/test_train_persist4d_allt.py`
- Update: `artifacts/allt_task_superiority_v1/budget_and_schedule.json`
- Create: `artifacts/allt_task_superiority_v1/TEST_REPORT.md`

1. Write launcher tests for branch/source/input authorization, clean worktree,
   immutable candidate ownership, external output paths, named variants,
   schedule, unique checkpoint records, and resume rejection on mismatch.
2. Verify RED, then implement the fail-closed launcher and resolved config.
3. Run real T1 and T3 forward/backward on one free A40 for two optimizer
   updates. Verify off-path parity, gradients, no GT state, future causality,
   encoder policy, optimizer membership, and M activation.
4. Run one unlabeled throughput trial. If needed, make the sole permitted
   batch/accumulation/update amendment. Freeze its reason, before/after values,
   timestamp, and status in the budget artifact.
5. Commit source, tests, configs, diagnostics, and the frozen formal schedule.

## Task 8: Train C0, C1, C2, and FH-adapt

**Files:**
- Generate: `artifacts/allt_task_superiority_v1/training/variant_matrix.json`
- Generate per variant: `resolved_config.yaml`, `learning_curve.csv`,
  `run_summary.json`, `checkpoint_manifest.json`

1. Materialize a single seed-45 episode draw and augmentation plan shared by
   all four variants.
2. Launch at most two independent formal runs concurrently on free A40s. Store
   checkpoints and optimizer state only under the external shared root.
3. Evaluate the development holdout at update 0/1000/2000/3000/4000, or the
   amended fixed schedule. Record optimizer steps, supervised stages, encoder
   scans, unique references, GPU hours, numerical status, and truncation.
4. Apply only the preregistered resource/numerical and fixed poor-curve early
   stop. Do not inspect Protocol-B.
5. Validate every checkpoint SHA, row count, and config/draw-plan binding.

## Task 9: Apply conditional variants and freeze selection

**Files:**
- Generate conditionally: C3 and FH-L training artifacts
- Generate: `artifacts/allt_task_superiority_v1/selection/selection_trace.csv`
- Generate: `artifacts/allt_task_superiority_v1/selection/FROZEN_SELECTION.json`

1. Evaluate complementarity using only the frozen development metrics. Train
   C3 only if C1 and C2 pass the gate; train FH-L only if the final candidate
   uses L. Otherwise emit explicit gate-skipped status artifacts.
2. Select one checkpoint per system by minimum all-T t-mAP delta, mean delta,
   then fixed secondary metrics. Include every eligible checkpoint in the
   trace.
3. Freeze selection hashes before any new-model Protocol-B prediction exists.
4. If seed 45 is promising, repeat the matched candidate and strongest ReScene
   training with seed 46 and the same selection algorithm; otherwise record
   `seed_confirmation=NOT_RUN`.

## Task 10: Evaluate final systems on Protocol-B

**Files:**
- Create: `scripts/evaluate_persist4d_allt.py`
- Create: `tests/test_evaluate_persist4d_allt.py`
- Generate: `artifacts/allt_task_superiority_v1/results/all_t_metrics.csv`
- Generate: `artifacts/allt_task_superiority_v1/results/deltas_all_metrics.csv`
- Generate: `artifacts/allt_task_superiority_v1/results/per_reference.csv`
- Generate: `artifacts/allt_task_superiority_v1/results/identity_counts.csv`
- Generate: `artifacts/allt_task_superiority_v1/results/score_sensitivity.csv`
- Generate: `artifacts/allt_task_superiority_v1/results/run_confirmation.csv`

1. Write tests for sequential T1..T5 updates, no skipped T3, exact cache keys,
   one prediction generation per model/seed/order, reducer reuse, unchanged
   trajectory keys, official temporal and prefix-overall evaluators, required
   CSV schemas, all 20 cells, and explicit failed-cell lists.
2. Verify RED, implement the evaluator, and pass small fixture tests.
3. Generate new predictions for every trained model. Reuse old R1 caches only
   for frozen baselines and diagnostics.
4. Evaluate the single frozen candidate and matched ReScene on all 129 units.
   Run 1000 reference-cluster bootstrap resamples when feasible and label the
   six-cluster equal-weight analysis descriptively.
5. Run evaluation, cache-parity, evaluator, and R1 regression tests.

## Task 11: Profile the frozen systems

**Files:**
- Create: `scripts/profile_persist4d_allt.py`
- Create: `tests/test_profile_persist4d_allt.py`
- Generate: `artifacts/allt_task_superiority_v1/profile/samples.csv`
- Generate: `artifacts/allt_task_superiority_v1/profile/summary.csv`

1. Write tests for six fixed representative references, T=2/3/4/5, two systems,
   five warmups, ten measurements, at least 48 units/480 samples, timing
   component fields, synchronization, point/segment counts, and exclusions.
2. Verify RED, implement the profiler, and run fixture tests.
3. Profile the final candidate plus FH-R1 or the strongest comparable
   FullHistory system on one A40 without other experiment processes.
4. Validate sample counts and aggregate summaries.

## Task 12: Finalize, verify, publish, and release resources

**Files:**
- Create: `scripts/finalize_persist4d_allt.py`
- Create: `tests/test_finalize_persist4d_allt.py`
- Create: `artifacts/allt_task_superiority_v1/FINAL_REPORT.md`
- Create: `artifacts/allt_task_superiority_v1/FINAL_MANIFEST.json`
- Create: `artifacts/allt_task_superiority_v1/HANDOFF.md`

1. Write tests for the six required status vocabularies, strict all-T gates,
   matched-baseline claim limits, 20-cell accounting, missing-stage status
   artifacts, handoff section completeness, artifact hashes, forbidden large
   files/secrets, and remote readback inputs.
2. Verify RED, implement finalization, and generate reports solely from audited
   artifacts.
3. Run all prompt-listed focused test modules, relevant legacy regressions,
   Ruff on changed Python, YAML/JSON decoding, `git diff --check`, checkpoint
   hash audits, and process/GPU cleanup checks. Do not run the full repository
   test suite.
4. Inspect the complete branch diff and artifact manifest personally. Commit
   only scoped source and small evidence.
5. Push `research/persist4d-allt-task-superiority-v1`, verify remote SHA equals
   local SHA, and read `HANDOFF.md` plus `FINAL_MANIFEST.json` from the remote
   branch. Record `PUSH_VERIFIED` only after exact readback succeeds.
6. Stop experiment processes, confirm CUDA contexts are released, and report
   the prompt-mandated final status ordering without overstating independence.
