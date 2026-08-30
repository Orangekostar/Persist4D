# ReScene Task-Learning Root-Cause V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a provenance-locked, gate-driven ReScene local-perception root-cause study and publish its code, results, manifests, and handoff.

**Architecture:** Focused contract, audit, training, evaluation, and diagnostic modules emit deterministic evidence under one new artifact namespace. A single common initialization and exact 29,700-step OneCycle schedule make short curves resumable; mechanical gates control every optional experiment.

**Tech Stack:** Python 3.10, PyTorch 2.6, PyTorch Lightning 2.6.5, Hydra 1.3, Concerto/PTv3, torch-scatter, stmetrics, pytest, Ruff, Git.

---

### Task 1: Freeze start state and external evidence

**Files:**
- Create: `artifacts/rescene_task_learning_root_cause_v1/START_STATE.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/START_STATE.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/EXTERNAL_EVIDENCE.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/LITERATURE_EVIDENCE.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/CODE_AUDIT.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/REPRODUCTION_GAP_CONTRACT.md`

- [ ] Record the exact parent SHA, upstream revision, paper facts, local baseline metrics, environment versions, required file hashes, frozen namespaces, and external repository revisions.
- [ ] Verify every numerical statement against a primary source or a committed local CSV/JSON artifact.
- [ ] Run `git diff --check` and commit with `RC0: freeze root-cause evidence and contracts`.

### Task 2: Implement core preflight contracts with TDD

**Files:**
- Create: `utils/rescene_rootcause_preflight.py`
- Create: `scripts/rescene_rootcause_preflight.py`
- Create: `tests/test_rescene_rootcause_preflight.py`
- Create: `tests/test_rootcause_full_schedule.py`
- Create: `tests/test_common_initialization_contract.py`
- Create: `tests/test_rootcause_variant_isolation.py`

- [ ] Write failing tests for canonical hashes, portable references, exact source/data/runtime bindings, scheduler equality, common tensor hashes, variant allowlists, and non-Git checkpoint manifests.
- [ ] Run:

```bash
python -m pytest -q tests/test_rescene_rootcause_preflight.py tests/test_rootcause_full_schedule.py tests/test_common_initialization_contract.py tests/test_rootcause_variant_isolation.py
```

Expected: failures because the root-cause modules do not exist.

- [ ] Implement immutable dataclasses for variant contracts and pure validation/build functions. R1 allows only `general.rootcause_objective_mode`; R2 allows only batch and accumulation; R3 allows only stochastic policy; R4 allows only filter classes; R5 allows only EOS.
- [ ] Re-run the four files and require all tests to pass.

### Task 3: Add the root-cause config and trajectory-safe callbacks

**Files:**
- Create: `conf/config_rescene4d_concerto_rootcause.yaml`
- Create: `conf/callbacks/rescene_rootcause.yaml`
- Create: `utils/rescene_rootcause_callbacks.py`
- Modify: `trainer/trainer.py`
- Modify: `main_instance_segmentation.py`
- Test: `tests/test_rootcause_full_schedule.py`
- Test: `tests/test_common_initialization_contract.py`

- [ ] Write failing tests asserting 450 configured epochs, explicit 29,700 scheduler steps, stop-on-first-epoch-90 semantics, exact 60/90/450 checkpoint epochs, and strict common-state loading before optimizer creation.
- [ ] Implement `RootCauseHorizonCallback` with equality-only stop at completed epoch 90 and `EpochSetCheckpointCallback` for 60/90/450.
- [ ] Add root-cause-only objective selection and strict common-state loading without changing legacy P2/Sonata behavior.
- [ ] Run the focused tests and existing P2/Sonata contract tests.

### Task 4: Audit objective and EOS semantics

**Files:**
- Create: `utils/rescene_objective_audit.py`
- Create: `scripts/audit_rescene_objective.py`
- Create: `tests/test_objective_semantics.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/LOSS_SEMANTICS.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/UPSTREAM_LOCAL_DIFF.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/upstream_local_diff.json`

- [ ] Write failing tests for loss-key classification, raw/weighted multipliers, diagnostic exclusion, contribution rows, gradient cosine, and EOS gate calculation.
- [ ] Implement fixed-batch objective/EOS audit functions with finite checks and named parameter-group gradients.
- [ ] Run the CLI on one frozen real batch and commit numeric outputs.
- [ ] Commit with the objective portion of `RC1: add runtime data and objective diagnostics`.

### Task 5: Audit label 255 and data mix

**Files:**
- Create: `utils/rescene_data_audit.py`
- Create: `scripts/audit_rescene_data_semantics.py`
- Create: `tests/test_filter255_inventory.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/DATA_SEMANTICS.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/filter255_inventory.csv`

- [ ] Write failing fixture tests that cover 255 labels in target construction, CE ignore handling, mask/dice supervision, and matcher inputs.
- [ ] Implement full active-train-database inventory and the preregistered 0.5% instance/point gate.
- [ ] Measure unmodified/filtered dataset counts, expected/observed mix, unique draws, and replacement duplicate rate.
- [ ] Run the real-data audit and publish the gate verdict.

### Task 6: Audit the actual DDP sampler chain

**Files:**
- Create: `scripts/audit_ddp_sampler_runtime.py`
- Create: `tests/test_ddp_sampler_runtime_contract.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/ddp_sampler_rank_trace.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/ddp_sampler_summary.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/RUNTIME_SEMANTICS.md`

- [ ] Write failing tests for wrapper-chain serialization, rank/world-size binding, at least 256 draws per rank, expected replacement duplicates, unintended cross-rank overlap, and global mix ratios.
- [ ] Implement a Lightning callback that records the trainer-resolved sampler after distributed setup and gathers rank traces without changing production training.
- [ ] Run on the exact two-GPU DDP configuration and classify the sampler hypothesis.

### Task 7: Audit frozen-encoder stochasticity

**Files:**
- Create: `scripts/audit_frozen_encoder_stochasticity.py`
- Create: `tests/test_encoder_stochasticity_audit.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/encoder_stochasticity.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/encoder_stochasticity_summary.json`

- [ ] Write failing tests for repeated-pass cosine, relative RMS, variance, module-mode inventory, DropPath-only disable/restore, and the `0.999`/`1e-3` gate.
- [ ] Run at least eight repeated real-batch encoder passes under current and DropPath-disabled policies.
- [ ] Authorize or close R3 mechanically from the registered threshold.

### Task 8: Audit physical-batch gradients

**Files:**
- Create: `scripts/audit_physical_batch_semantics.py`
- Create: `tests/test_physical_batch_gradient_audit.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/fixed_batch_panel.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/physical_batch_gradients.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/audit/physical_batch_summary.json`

- [ ] Write failing tests for exactly 32 frozen sample references, grouping equivalence, parameter-group statistics, OOM-as-infeasible behavior, and the `<0.98`/`>10%` gate.
- [ ] Measure physical-global 4 and every feasible grouping in 8/16/32 with identical samples and initialization.
- [ ] Record peak memory and step time without consulting validation AP.
- [ ] Commit all RC1 audit code and evidence.

### Task 9: Freeze common initialization and variant authorization

**Files:**
- Create: `scripts/build_rescene_rootcause_initialization.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/initialization/COMMON_INITIALIZATION.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/VARIANT_CONTRACT.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/variant_manifest.json`

- [ ] Generate one external `rootcause_common_initial_state.pt` from the verified Concerto encoder, seed 45, and root-cause config.
- [ ] Verify its full SHA/bytes/schema and strict pre-step load for R0/R1 plus only gate-authorized conditional variants.
- [ ] Materialize exact resolved-config diffs and fail if more than four total short-curve configurations are authorized.
- [ ] Commit with `RC2: freeze common initialization and variant preflight`.

### Task 10: Implement and run controlled short curves

**Files:**
- Create: `scripts/run_rescene_rootcause_training.py`
- Create: `scripts/finalize_rescene_rootcause_training.py`
- Create: `tests/test_no_persist4d_leakage.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/learning_curves.csv`

- [ ] Write failing launcher tests for authorization, initialization, exact variant diff, external output binding, no-Persist4D leakage, 90-epoch stop, and exact resume state.
- [ ] Launch R0 and R1 sequentially on the registered two-GPU topology; launch no more than two audit-authorized variants.
- [ ] Record validation epochs 15/30/45/60/75/90 and apply only the preregistered epoch-45 elimination rule.
- [ ] Commit code before launch and compact curve artifacts after each completed checkpoint group.

### Task 11: Run matched epoch-60/90 evaluation and decide RC4

**Files:**
- Create: `scripts/evaluate_rescene_rootcause_checkpoint.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/official_like_epoch60.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/official_like_epoch90.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/rootcause_per_seed.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/rootcause_summary.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/short_curves/ROOTCAUSE_SHORT_DECISION.md`

- [ ] Reuse the exact official-like 154-sequence harness for every variant/checkpoint at seeds 45/46/47.
- [ ] Compute paired `SpatialStageMean` deltas and apply all five full-run authorization conditions.
- [ ] Select at most one candidate using the registered ordering, or write an explicit RC4 `gate_skipped` result.
- [ ] Commit with `RC3: complete controlled short learning curves`.

### Task 12: Resume exactly one authorized full candidate

**Files:**
- Create: `artifacts/rescene_task_learning_root_cause_v1/full_candidate/FULL_TRAINING_REPORT.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/full_candidate/selected_checkpoint_manifest.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/full_candidate/official_like_per_seed.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/full_candidate/official_like_summary.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/full_candidate/ROOT_CAUSE_FULL_VERDICT.md`

- [ ] If authorized, resume the exact epoch-90 state with unchanged optimizer, scheduler, sampler, config, and initialization bindings to epoch 450.
- [ ] Evaluate the validation-selected checkpoint at seeds 45/46/47 and classify confirmed/partial/not-confirmed.
- [ ] If unauthorized, create compact status artifacts naming the failed gate without numeric placeholders.
- [ ] Commit the stage only when its gate-directed work is complete.

### Task 13: Implement and run decoder diagnostics

**Files:**
- Create: `utils/rescene_decoder_diagnostics.py`
- Create: `scripts/analyze_query_initialization.py`
- Create: `scripts/analyze_query_conflicts.py`
- Create: `scripts/analyze_attention_mask_recall.py`
- Create: `scripts/analyze_superpoint_features.py`
- Create: `tests/test_query_initialization_diagnostics.py`
- Create: `tests/test_query_conflict_diagnostics.py`
- Create: `tests/test_attention_mask_recall_diagnostics.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/decoder_diagnostics/DECODER_DIAGNOSTICS.md`

- [ ] Write fixture tests for all prompt-defined formulas and layer/sequence coverage.
- [ ] Instrument evaluation-only outputs for FPS location, per-layer masks, attention allowance/reset, query scores, and segment features.
- [ ] Run all diagnostics on the best reproduction-compatible model and publish the four required CSVs.
- [ ] Commit with `SD0: add decoder diagnostics`.

### Task 14: Evaluate authorized native strong-local variants

**Files:**
- Create: `tests/test_np_feature_query_variant.py`
- Create: `tests/test_adaptive_scatter_variant.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/strong_local/variant_manifest.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/strong_local/learning_curves.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/strong_local/official_like_per_seed.csv`
- Create: `artifacts/rescene_task_learning_root_cause_v1/strong_local/STRONG_LOCAL_VERDICT.md`

- [ ] Test A1 as the single `use_np_features=true` change with common tensors restored exactly.
- [ ] Run A1 through the 90-epoch protocol; authorize A2 only from its diagnostic gate and A1 result.
- [ ] Run A2 or A1+A2 only when their explicit gates pass; do not implement high-risk modules without their committed diagnostic-gated design.
- [ ] Commit with `SP0: evaluate native strong-local variants` when applicable.

### Task 15: Finalize, verify, and publish

**Files:**
- Create: `scripts/finalize_rescene_task_learning_root_cause.py`
- Create: `artifacts/rescene_task_learning_root_cause_v1/FINAL_REPORT.md`
- Create: `artifacts/rescene_task_learning_root_cause_v1/FINAL_MANIFEST.json`
- Create: `artifacts/rescene_task_learning_root_cause_v1/HANDOFF.md`

- [ ] Generate the final decision, claim allowlist/denylist, complete artifact/external-file hashes, exact commands, and all 30 handoff sections.
- [ ] Run:

```bash
python -m pytest -q
python -m ruff check .
git diff --check
git diff --name-only 29bb228ad5b090797045fa3b3fc55cb973f001be -- artifacts/P2 artifacts/P6A artifacts/system_comparison artifacts/system_comparison_v2 artifacts/reviewer_closure_v3 artifacts/sonata_second_perception_v1
python scripts/finalize_rescene_task_learning_root_cause.py --check-privacy
```

Expected: tests and Ruff pass, no diff whitespace errors, frozen-namespace diff is empty, and privacy scan reports no committed leak.

- [ ] Commit with `FINAL: finalize report manifest and handoff`.
- [ ] Push the branch and require `git rev-parse HEAD` to equal `git ls-remote origin refs/heads/research/persist4d-rescene-task-learning-root-cause-v1`.
