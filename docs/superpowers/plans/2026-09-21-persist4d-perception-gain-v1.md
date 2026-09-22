# Persist4D Perception Gain V1 Implementation Plan

> Execute in the isolated worktree `/home/ww/paper5/.worktrees/persist4d-perception-gain-v1` on branch `research/persist4d-perception-gain-v1`. The fixed parent is `6ef77620aa20926311eff3124a794a6ca2e32727`.

**Goal:** Implement and execute the approved Perception Gain V1 campaign from fixed R1 assets through measured selection, final confirmation, profiling, reporting, and publication.

**Architecture:** Keep legacy ReScene, TaskMemory, and CrossWindow behavior unchanged by default. Add narrow opt-in primitives for stage-aware mask losses, semantic query positioning, first-stage open attention, soft postprocess evidence, and the local logit refiner. Reuse native episode metadata, strict R1 loading, D0 association, official postprocess, and official metric accumulators. Drive the campaign with a resumable stage machine whose public artifacts are lightweight and whose checkpoints/caches remain under the external run root.

**Runtime:** Python 3.11 in `/home/ww/miniconda3/envs/persist4d`, PyTorch/Lightning/Hydra, two A40 GPUs per training job, FP32, physical batch 2/GPU, accumulation 8, maximum 8 CPU workers.

---

## Task 1: Freeze bootstrap identities and data roles

**Files:**
- Add: `configs/perception_gain_v1.yaml`
- Add: `scripts/perception_gain_campaign.py`
- Add: `scripts/perception_gain_data.py`
- Add: `tests/test_perception_gain_campaign.py`
- Add: `tests/test_perception_gain_data.py`
- Add: `artifacts/perception_gain_v1/EXECUTION_INSTRUCTION.md`

1. Write failing tests for config schema/identity validation, output-root isolation, deterministic role construction, physical reference disjointness, and resume identity mismatch.
2. Run the two test files and confirm failures are caused by missing modules.
3. Implement atomic JSON/text/log helpers, asset resolution adapter, role derivation, budget/config validation, state initialization, `bootstrap`, `status`, and ordered `run --through` dispatch.
4. Copy the approved instruction byte-for-byte, verify its SHA256, and run `bootstrap` against the real asset locator.
5. Verify generated `RUN_CONFIG.json`, `DATA_ROLES.json`, `RUN_STATE.json`, `CODE_BINDINGS.md`, local asset locator, and execution log.

## Task 2: Repair threshold-specific temporal diagnostics

**Files:**
- Modify: `scripts/p6a_metrics.py`
- Modify: `scripts/diagnose_crosswindow_failures.py`
- Modify: `tests/test_crosswindow_metrics.py`
- Modify: `tests/test_diagnose_crosswindow_failures.py`

1. Write failing tests showing that an object matched at tau 0.50 can fail at tau 0.75, strict temporal support is evaluated stage-by-stage, one-to-one assignment is preserved, and diagnostic rows do not reuse a prior threshold result.
2. Run focused tests and confirm the wrong fixed-0.5 behavior.
3. Add a narrow official-semantics temporal match trace and pass explicit `tau` through `_published_match_by_gt()`.
4. Derive formal thresholds from the locked evaluator constant and add display-only 0.25/0.75 without changing metric computation.
5. Run focused and CrossWindow regression tests.

## Task 3: Implement stage-aware mask losses

**Files:**
- Add: `models/perception_gain.py`
- Modify: `models/criterion.py`
- Modify: `trainer/trainer.py`
- Add: `tests/test_perception_gain_model.py`

1. Write failing tests for strict segment-stage derivation, legacy equivalence, balanced aggregation, worst aggregation, empty matched sets, ghost-stage penalties, per-stage sampling, and aux-output propagation.
2. Run the test file and confirm failures are due to absent production behavior.
3. Implement `legacy`, `balanced`, and `worst` mask-loss modes with alpha 0.5/beta 0.25 and connected zero losses.
4. Keep matcher, classification/change/contrastive losses, loss keys, raw-sum reduction, and legacy defaults unchanged. Export debug counters outside the differentiable loss dictionary.
5. Run model tests plus criterion/trainer regressions.

## Task 4: Implement Q-SEM query positioning and A-OPEN

**Files:**
- Modify: `models/perception_gain.py`
- Modify: `models/rescene.py`
- Modify: `models/__init__.py`
- Modify: `tests/test_perception_gain_model.py`

1. Write failing tests for the scorer topology, stage quotas, stable semantic FPS/exploration selection, candidate shortage repeats, zero query content, preserved positional encoding coordinates, exactly one open cross-attention call, and preserved padding masks.
2. Run focused tests and confirm the missing behavior.
3. Add opt-in feature/coordinate query initialization using aggregated segment features and real segment mean coordinates; scorer inference is detached and frozen.
4. Add opt-in first-execution-stage attention opening while retaining all later masks and all padding masks.
5. Run focused tests and existing ReScene/task-memory model regressions.

## Task 5: Export real soft postprocess evidence

**Files:**
- Modify: `scripts/rescene_task_postprocess.py`
- Modify: `tests/test_rescene_task_postprocess.py`
- Modify: `models/perception_gain.py`

1. Write failing tests that default output is byte-for-byte equivalent, opt-in evidence retains filtered source lineage, full-resolution mask logits/probabilities, class probabilities, query features, segment features, and point/segment mappings.
2. Run focused tests and confirm the optional output is absent.
3. Implement an optional immutable sidecar without reading `x.gt_targets` or altering official bool masks/scores/classes.
4. Validate vertex/segment dimensions and no-op mask equivalence at threshold zero.
5. Run postprocess and task-output regressions.

## Task 6: Implement the causal local refiner

**Files:**
- Modify: `models/perception_gain.py`
- Add: `tests/test_perception_gain_refiner.py`

1. Write failing tests for exact old/new key pairing `(scan_id, logical_id, generation, source_class_id)`, rejection of non-adjacent or duplicate old evidence, the 135-dimensional input contract, zero-initialized no-op behavior, bounded `2*tanh` delta, and frozen-parent gradient isolation.
2. Run focused tests and confirm missing behavior.
3. Implement pairing, scalar feature assembly, non-affine LayerNorm(128), two-linear MLP, zero final layer, and refined-logit application.
4. Verify default no-op identity and all-valid-segment application.

## Task 7: Implement scorer/refiner data preparation

**Files:**
- Modify: `scripts/perception_gain_data.py`
- Add: `tests/test_perception_gain_data.py`

1. Write failing tests for reference-first scorer pair selection, minimum eight-reference gate, labeled thing/stuff/ignore targets, deterministic reference/segment sampling, adjacent-window refiner pairing, and cache size accounting.
2. Run focused tests and confirm missing behavior.
3. Implement deterministic pair inventories and streaming sharded sidecars using StageMeta/canonical vertex identities.
4. Ensure GT appears only in offline labels/diagnostics and is removed from publishable inference payloads.

## Task 8: Implement perception training and exact resume

**Files:**
- Add: `trainer/perception_gain_trainer.py`
- Add: `scripts/train_perception_gain.py`
- Add: `conf/perception_gain_v1/common.yaml`
- Add: `conf/perception_gain_v1/{C0,S-BAL,S-WORST,Q-SEM,A-OPEN}.yaml`
- Add: `tests/test_perception_gain_training.py`

1. Write failing tests for strict R1 load, common sample stream, batch/accumulation contract, 150-step warmup plus 3000-step cosine, FP32, gradient clipping, optimizer parameter coverage, eval/checkpoint steps, and exact cursor/RNG resume.
2. Run focused tests and confirm missing behavior.
3. Build a batch-capable pair trainer over existing TaskMemory episode metadata with normal InstanceSegmentation supervision and no task-memory state.
4. Implement scorer training at exactly 500 updates and perception training at requested endpoints without restarting the 3000-step schedule.
5. Run two real optimizer-step smoke jobs and verify nonzero gradients/parameter changes and strict legacy output parity at update 0.

## Task 9: Implement checkpoint evaluation and selection

**Files:**
- Add: `scripts/perception_gain_evaluation.py`
- Add: `tests/test_perception_gain_evaluation.py`

1. Write failing tests for `S_mean/S_long/S_min`, deterministic ranking, pilot/full gates, checkpoint selection, P* fallback, refiner gates, null coverage comparisons, and lock immutability.
2. Run focused tests and confirm missing behavior.
3. Reuse real D0/FH producers, official postprocess, CrossWindow publisher, and official metrics for CAL/SEL/PB/LOCAL/ADDITIONAL.
4. Implement required metric/coverage/per-reference schemas and immutable `PERCEPTION_LOCK.json`/`FINAL_LOCK.json`.
5. Run focused tests and evaluator regressions.

## Task 10: Complete foundation measurements

**Files:**
- Modify: `scripts/perception_gain_campaign.py`
- Generate: `artifacts/perception_gain_v1/foundation/*`

1. Bind the R1 resolved config and LOCAL-T2 provenance from tracked manifests/checkpoint metadata.
2. Run one real CAL T2 and one T5 native forward, live D0, and cached/live D0 parity.
3. Recompute corrected CAL E1 tables at official thresholds and collect the fixed current diagnostic panel.
4. Record measured GPU hours, commands, coverage, and any incomplete units.

## Task 11: Execute scorer and five-arm pilot

**Files:**
- Generate: `artifacts/perception_gain_v1/training/scorer/*`
- Generate: `artifacts/perception_gain_v1/training/pilot/*`

1. Generate scorer data and train/freeze update 500, or record the exact coverage block while continuing other arms.
2. Run C0/S-BAL/S-WORST/Q-SEM/A-OPEN to update 750 on the identical seed-45 stream.
3. Measure first-50-update throughput, forecast budget, and apply only the prescribed pruning order.
4. Evaluate updates 0/250/750 on CAL and select at most two new arms for full training.

## Task 12: Execute full training and perception selection

**Files:**
- Generate: `artifacts/perception_gain_v1/training/full/*`
- Generate: `artifacts/perception_gain_v1/selection/PERCEPTION_LOCK.json`

1. Continue C0 and promoted arms to update 3000 with exact resume.
2. Evaluate CAL at 1500/2250/3000 and select one checkpoint per architecture.
3. Evaluate locked candidates once on SEL, apply the prescribed gates/ranking, and freeze P* or the C0/R1 fallback.

## Task 13: Train/select refiner and freeze final recipe

**Files:**
- Add: `scripts/train_perception_refiner.py`
- Generate: `artifacts/perception_gain_v1/training/refiner/*`
- Generate: `artifacts/perception_gain_v1/selection/FINAL_LOCK.json`

1. Generate causal P* refiner sidecars from TRAIN references with at least three scans.
2. Train only the refiner for 1500 updates and evaluate 0/500/1000/1500 on CAL.
3. Evaluate the fixed refiner candidate on SEL, apply its gate, and freeze FINAL.
4. Run seed 46 only when measured remaining budget permits; otherwise record the prescribed skipped status.

## Task 14: Final confirmation and live profiling

**Files:**
- Generate: `artifacts/perception_gain_v1/confirmation/*`
- Generate: `artifacts/perception_gain_v1/resources/*`

1. Execute all required locked methods on PB 129 units, LOCAL-T2, and all readable ADDITIONAL horizons.
2. Preserve trinary comparison status where coverage is incomplete and keep historical rows separate.
3. Profile six physical references on the same A40: one warmup sequence and three measured sequences per method, with network/state/materialize/end-to-end/cumulative and memory fields.
4. Check total/category GPU hours and cache bytes against frozen limits.

## Task 15: Report, review, and publish

**Files:**
- Generate: `artifacts/perception_gain_v1/{FINAL_REPORT.md,HANDOFF.md,ARTIFACT_MANIFEST.json,RELEASE_PLAN.json}`
- Modify: `scripts/perception_gain_campaign.py`

1. Generate the six required result/cost/status tables and the fourteen-section handoff from measured artifacts only.
2. Run the 12-16 focused behavioral tests, related regressions, Ruff on modified Python files, compile/import checks, and one final native smoke.
3. Review the full diff, prompt checklist, artifact hashes, identity/coverage claims, and absence of credentials/raw GT/large files.
4. Create the experiment commit and documentation commit, push the branch, create/push tag `persist4d-perception-gain-v1`, publish release assets, and verify remote SHA/readback.
5. Store the publication receipt under the external run root and mark the campaign complete only after prompt-to-artifact review passes.
