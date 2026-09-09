# Persist4D Task-Memory Retention V2 Implementation Plan

> **Execution mode:** Use `superpowers:executing-plans` in the isolated worktree and retain architecture, scientific decisions, integration, and review in the primary agent. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, train, evaluate, and publish one prediction-driven bounded task-memory system that preserves ReScene discovery, supports lag-one output revision, and is honestly judged at T2-T5 against frozen R1 and matched FullHistory controls.

**Architecture:** Keep the reviewed ReScene tree intact and add sibling data, state/routing, model, supervision, training, evaluation, profiling, and publication modules. Prediction-only routing is read-only; a separate writer commits each stage exactly once. M2 is mandatory at the frozen 3,000-update budget, while visual memory and prefix distillation are conditional branches governed by preregistered development evidence.

**Tech Stack:** Python 3.10, PyTorch, Lightning, Hydra/OmegaConf, Sonata/Pointcept, existing ReScene official post-processing and stmetrics, pytest, Ruff, Git worktrees.

**Spec:** `/home/ww/paper5/docs/Persist4D_Codex_TaskMemory_Retention_V2.md` (SHA256 `f48e559c16463e7eb4e9ed27ae1683120d18c22f20ed97cbdd71b656dab0ded5`) and `/home/ww/paper5/docs/01_EVIDENCE_AND_CODE_MAP(1).md` (SHA256 `56a6859904eb27f611b0a10dfe235843ce2fc2a397427f7bdf69d73ad6fc0d87`). Copy both unchanged into `docs/task_memory_v2/` in Task 1 so the specification travels with the branch.

## Global Constraints

- Start from reviewed parent `32a51e11b51043ec5a5825215669ef7b0ea03bc2` on branch `research/persist4d-task-memory-retention-v2`; never rewrite old All-T FAIL artifacts.
- R1 identity is SHA256 `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`, 754,813,672 bytes; Concerto identity is SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
- Deployment forward, route, commit, visual selection, lag1 acceptance, and state contain predictions plus unlabeled metadata only; GT enters criterion, evaluation, and post-hoc diagnostics only.
- M2 uses seed 45, 2 A40 GPUs, physical batch 1/GPU, accumulation 4, effective batch 8, 32-true precision, and exactly 3,000 optimizer updates per arm. A 300-update check is a resumable prefix of the same schedule.
- M2 evaluates checkpoints 0/750/1500/2250/3000. M3 uses 1,500 updates per arm; M4 uses 1,000 per arm only when its gate passes.
- Initial campaign caps are 120 training GPU-hours, 30 diagnostic/evaluation GPU-hours, 40 GiB new evaluation cache, and 2 MiB additional permanent task-memory state per episode. Amend caps once before formal execution only from measured throughput, never from AP outcomes.
- Primary policy is lag1 and primary reducer is mean for every T. commit0, latest, and max remain explicit controls/sensitivities.
- `TMAP_ALL_T_PASS` requires every unrounded T2-T5 delta against each named baseline to exceed `1e-6`; 20-cell task success requires all four horizons times five task metrics to exceed the same threshold.
- Protocol-B has six physical reference clusters; 129 order units are correlated. Independent-reference results remain separate and absence yields `GENERALIZATION=NOT_ESTABLISHED`.
- Large checkpoints, optimizer states, datasets, and caches stay outside Git under `/mnt/shared/ww`; Git stores resolvable logical references, SHA256, bytes, configs, parent checkpoint, seed, updates, and selection provenance.

---

### Task 1: Freeze M0 contracts and source evidence

**Status:** Complete

**Files:**
- Create: `docs/task_memory_v2/01_EVIDENCE_AND_CODE_MAP.md`
- Create: `docs/task_memory_v2/Persist4D_Codex_TaskMemory_Retention_V2.md`
- Create: `conf/task_memory_v2/experiment.yaml`
- Create: `scripts/task_memory_contracts.py`
- Create: `scripts/prepare_task_memory_v2.py`
- Create: `tests/test_task_memory_contracts.py`
- Modify: `.gitignore`
- Generate: `artifacts/task_memory_retention_v2/{START_STATE.json,EVIDENCE_MAP.md,DATA_CONTRACT.json,OUTPUT_CONTRACT.md,BASELINE_CONTRACT.json,BUDGET_CONTRACT.json,EVALUATION_CONTRACT.json,PROFILE_CONTRACT.json,COMMANDS.md,references_inventory.csv}`
- Generate untracked: `artifacts/task_memory_retention_v2/external_assets.local.json`

**Interfaces:**
- Consumes: reviewed parent, immutable source-doc hashes, R1/Concerto manifests, RIO/ScanNet databases, old split/evaluation artifacts, local asset roots.
- Produces: `freeze_task_memory_contracts(config, assets) -> ContractBundle` and the sole configuration/provenance inputs used by all later CLIs.

- [x] **Step 1: Add failing contract tests**

Test exact parent/branch, first-read weight identity, no overlap between training and frozen dev/Protocol-B references, native length counts based on readable files rather than metadata alone, five equal 20% episode buckets, fixed budgets, lag1/commit0 definitions, and rejection of a non-resolvable logical asset reference.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_contracts.py`

Expected: import failure because `scripts.task_memory_contracts` does not exist.

- [x] **Step 3: Implement fail-closed contract builders**

Use immutable dataclasses and canonical JSON hashing. The public boundary is:

```python
@dataclass(frozen=True)
class AssetBindings:
    data_root: Path
    rio_metadata: Path
    r1_checkpoint: Path
    concerto_pretrained: Path
    run_root: Path

def freeze_task_memory_contracts(
    *, config_path: Path, assets: AssetBindings, output_root: Path
) -> ContractBundle: ...
```

Inventory RIO reference/master/scan UUID/native length/readability/exposure and ScanNet single-scan availability. Store real paths only in the untracked local resolver; public contracts use logical URIs plus resolver key.

- [x] **Step 4: Execute M0 preparation in the real environment**

Run `python -m scripts.prepare_task_memory_v2` with explicit data, R1, Concerto, run-root, and output paths. Hash each large weight once, compare with the frozen identities, then reuse the manifest.

- [x] **Step 5: Validate and commit M0**

Require all public JSON self-hashes, readable data counts, frozen role separation, cache/storage caps, no fabricated result CSVs, and a clean direct test. Commit source specs, configs, implementation, tests, and compact M0 artifacts together.

### Task 2: Add native-length causal episodes and StageMeta

**Status:** Complete

**Files:**
- Create: `datasets/task_memory_episode.py`
- Modify: `datasets/__init__.py`
- Create: `scripts/preflight_task_memory_episode.py`
- Create: `tests/test_task_memory_episode.py`
- Generate: `artifacts/task_memory_retention_v2/implementation/preflight_real_sequence.json`

**Interfaces:**
- Consumes: `DATA_CONTRACT.json`, existing `SemanticSegmentationDataset.load_scan_indices`, and existing Pointcept collator outputs.
- Produces: `NativeEpisodeMaster`, `TaskMemoryEpisodeSpec`, `StageMeta`, `TaskMemoryEpisodeBatch`, `build_task_memory_draw_plan`, and `TaskMemoryEpisodeDataset` for native H1-H5.

- [x] **Step 1: Write causal data tests**

Cover real H2/H3/H4 masters, same-reference unique scans, no prefix dependence on unseen future scans, rank-synchronous equal-H groups, 20% bucket schedule, scan-local original vertex IDs, full-resolution/voxel inverse alignment, canonical `(reference_id, instance_id)` supervision identity, and one empty-current-stage target in a mixed batch.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_episode.py`

Expected: import failure because `datasets.task_memory_episode` does not exist.

- [x] **Step 3: Implement the sibling episode loader**

```python
@dataclass(frozen=True)
class StageMeta:
    reference_id: str
    episode_id: str
    scan_ids_in_window: tuple[str, ...]
    absolute_stage_index: int
    local_stage_ids: Tensor
    original_vertex_ids: tuple[Tensor, ...]
    scan_vertex_offsets: Tensor
    point2segment: Tensor
    segment_stage_ids: Tensor
    augmentation_transform_id: str
    coordinate_frame_id: str
    voxel_inverse: Tensor
    full_resolution_point2segment: Tensor
```

Configure the legacy dataset without its window-dependent train augmentation, then apply one explicit seed-derived episode transform to each loaded stage. Keep elastic distortion disabled. Build metadata independently of labels and keep GT classes/IDs/masks in training targets.

- [x] **Step 4: Run a real-sequence preflight**

Load one repeated scan through two legal windows, verify vertex/label identity and inverse maps, mutate a future scan choice, and prove the earlier commit0 input/output preparation is unchanged.

- [x] **Step 5: Validate and commit the data path**

Run the new tests plus `tests/test_persist4d_sequence_dataset.py` and the directly used Pointcept collation tests. Record the exact real-sequence sample identities and hashes in `preflight_real_sequence.json`.

### Task 3: Implement commit0 and lag1 output policies

**Status:** Complete

**Files:**
- Create: `scripts/task_memory_output.py`
- Create: `scripts/run_task_memory_policy_baseline.py`
- Create: `tests/test_task_memory_output.py`
- Create: `tests/test_task_memory_policy_baseline.py`
- Generate: `artifacts/task_memory_retention_v2/baseline/diagnostic_panel.json`
- Generate: `artifacts/task_memory_retention_v2/baseline/{policy_comparison.csv,gap_event_strata.csv}`
- Generate: `artifacts/task_memory_retention_v2/baseline/cache_manifest.json`

**Interfaces:**
- Consumes: `OfficialTaskPrediction`, prediction lineage, route logical IDs, and current `StageMeta`.
- Produces: `LagOnePublisher.update(prediction, identity_map) -> PublishedPrefix`, an append-only archive, one-scan revision log, and accounting bytes.

- [x] **Step 1: Write output-policy tests**

Test that lag1 revises only the previous scan from the current W2 output, freezes older scans, keeps previous-only candidates, uses route before same-scan class-compatible IoU matching, applies threshold 0.5 and stable query-index ties, never chooses per-object old/new versions by GT, preserves `(logical_id,generation,class_id)`, and reports the unflushed final scan as provisional.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_output.py`

- [x] **Step 3: Implement bounded publication state**

Store only the prior scan's revisable masks/vertex map plus compact frozen output rows. Reuse existing stable assignment primitives; do not perform another model forward and do not expose GT to the publisher.

- [x] **Step 4: Run policy-only R1+B4 baseline**

Evaluate commit0 and lag1 on frozen development plus at most 12 preregistered diagnostic masters. Preserve regressions. Attribute the delta to the complete output system whenever identity linking or scoring differs.

- [x] **Step 5: Validate and commit output policy**

Check one-revision scope, point lineage, class trajectories, output bytes, and future-label invariance; then commit code, tests, and compact baseline tables.

### Task 4: Split prediction routing from single-stage commit

**Status:** Complete

**Files:**
- Create: `models/task_memory_state.py`
- Create: `models/task_memory_routing.py`
- Modify: `models/__init__.py`
- Create: `scripts/run_task_memory_controls.py`
- Create: `tests/test_task_memory_routing.py`
- Create: `tests/test_task_memory_controls.py`
- Generate: `artifacts/task_memory_retention_v2/implementation/{query_state_contract.md,router_diagnostics.csv}`
- Generate: `artifacts/task_memory_retention_v2/baseline/long_memory_controls.csv`
- Generate: `artifacts/task_memory_retention_v2/baseline/control_observation_manifest.json`

**Interfaces:**
- Consumes: detached query/class/mask predictions, B4 numerical association settings, and unlabeled stage metadata.
- Produces: `TaskMemoryState`, `EntityRoute`, `CommitResult`, `route_entities`, `commit_entities`, and D-LAST/D-EMA controls.

- [x] **Step 1: Write route/commit tests**

Exercise route purity, complete-assignment-before-threshold semantics, current/previous support separation, stable route scores, logical ID/generation inheritance, dormant preservation, score-ordered births, reject-birth overflow, one watermark increment, and absence of GT-named fields in deployed state.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_routing.py`

- [x] **Step 3: Implement immutable route and writer**

```python
def route_entities(
    pre_output: PredictionObservation,
    old_state: TaskMemoryState,
    stage_meta: Sequence[StageMeta],
) -> EntityRoute: ...

def commit_entities(
    final_output: PredictionObservation,
    route: EntityRoute,
    old_state: TaskMemoryState,
    stage_meta: Sequence[StageMeta],
) -> CommitResult: ...
```

Call existing B4 association for route. Writer consumes the frozen route and reproduces B4 confidence-EMA updates without reassociation. Preserve no-reuse generation fields even though v1 uses reject-birth when full.

- [x] **Step 4: Build no-training D-LAST/D-EMA controls**

Use the same R1 observations, K=100, B4 assignment/class handling, fixed thresholds, and both output policies. If a control is algebraically equivalent to B4, record equivalence rather than duplicate a result row.

- [x] **Step 5: Validate and commit routing/state**

Run new tests and `tests/test_system_comparison_metrics.py`; inspect state bytes and route coverage on real samples before committing.

### Task 5: Add proposal-anchored entity-conditioned ReScene

**Status:** Complete

**Files:**
- Create: `models/task_memory_read.py`
- Create: `models/persist4d_task_memory.py`
- Modify: `models/__init__.py`
- Create: `tests/test_persist4d_task_memory.py`
- Generate: `artifacts/task_memory_retention_v2/implementation/{r1_load_report.json,query_shape_trace.json}`

**Interfaces:**
- Consumes: the first complete ReScene decoder pass, `TaskMemoryState`, `StageMeta`, and prediction-only route.
- Produces: `Persist4DTaskMemory.forward(..., task_state, stage_meta)` returning original raw outputs plus route/lineage/read diagnostics; state commit remains external.

- [x] **Step 1: Write model/read tests**

Test named-prefix strict R1 loading; empty-state and disabled-module raw parity; one route/read call after `len(hlevels)-1`; normalized QK with scale initialized to `sqrt(128)` and clamped `[1,64]`; matched-slot-plus-null attention; strict zero residual when null wins; zero-initialized output projection; discovery queries unchanged when unassigned; and no second old-C2 read.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_persist4d_task_memory.py`

- [x] **Step 3: Implement one data flow through the existing hook**

At the first complete-scale hook, call the original `mask_module` once for routing logits/masks, freeze the discrete route, read only the routed slot plus a zero-value null branch, add the gated residual, and let remaining ReScene stages/heads execute normally. Store route diagnostics only for the duration of forward and clear them in `finally`.

- [x] **Step 4: Execute the real R1 parity gate**

On fixed real T1/T2 input and RNG, compare new-module-off/empty-state raw tensors and official candidates against R1. Record tolerances, input identity, named missing prefixes, and exactly one read invocation.

- [x] **Step 5: Validate and commit model adaptation**

Run new tests plus `tests/test_rescene_query_features.py` and `tests/test_rescene_task_postprocess.py`; commit model code and compact parity artifacts.

### Task 6: Implement Q-INDEP and tracklet-aware TALA supervision

**Status:** Complete

**Files:**
- Create: `models/task_memory_supervision.py`
- Create: `models/task_memory_criterion.py`
- Create: `tests/test_task_memory_criterion.py`
- Generate: `artifacts/task_memory_retention_v2/implementation/sequence_loss_example.csv`

**Interfaces:**
- Consumes: route/commit lineage, target `ids`, canonical reference identity, ambiguity metadata, original SetCriterion matcher/losses, and post-conditioning aux boundary.
- Produces: `TrainingIdentityLedger`, `build_tala_assignment`, and `TaskMemoryCriterion.compute_with_assignments(...) -> TaskMemoryLossResult`.

- [x] **Step 1: Write supervision tests**

Cover inherited-query fixed GT assignment, newborn-only residual Hungarian matching, Q-INDEP full independent matching, wrong-route penalty without GT route repair, duplicate predicted slots selecting one positive training item, ambiguity exclusion counts, previous-only positive class/mask, fully absent inherited no-object plus differentiable empty-current-mask loss, one empty sample beside a nonempty sample, and post-conditioning aux inheritance while pre-conditioning aux stays independent.

- [x] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_criterion.py`

- [x] **Step 3: Implement ledger and assignment composition**

The ledger maps `(logical_id,generation)` to qualified `(reference_id,canonical_instance_id)` only after a real predicted birth intersects a normal Hungarian match. It never changes route/state. Reuse existing `SetCriterion.get_loss`; compose per-layer indices without modifying the old criterion.

- [x] **Step 4: Implement empty-target-safe losses**

Return graph-connected zero mask/dice/KD terms for empty positives, retain classification negatives, and add explicit current-stage empty-mask supervision for inherited absent entities. Preserve class-head final-index no-object and ignore label 253.

- [x] **Step 5: Validate and commit supervision**

Run new tests plus `tests/test_objective_semantics.py`; export one hand-checkable sequence loss table separating R1 components, inherited visibility, ambiguity exclusions, and layer assignments.

### Task 7: Add TaskMemoryTrainer, exact resume, and M2 configs

**Status:** In progress

**Files:**
- Create: `trainer/task_memory_trainer.py`
- Modify: `trainer/__init__.py`
- Create: `scripts/train_task_memory.py`
- Create: `conf/task_memory_v2/{W-BASE,Q-INDEP,Q-TALA,FH-MATCH}.yaml`
- Create: `tests/test_task_memory_trainer.py`
- Create: `tests/test_train_task_memory.py`
- Generate: `artifacts/task_memory_retention_v2/implementation/real_gradient_smoke.json`
- Generate: `artifacts/task_memory_retention_v2/training/{variants.json,resolved_configs/,config_diffs/,costs_and_exposure.csv}`

**Interfaces:**
- Consumes: frozen contracts, task-memory episodes, model, criterion, R1 init, common adapter initialization, and an external run root.
- Produces: stage/chunk training, 2-stage TBPTT graph shadow, exact draw cursor resume, checkpoint manifests, learning logs, and four controlled M2 launch commands.

- [ ] **Step 1: Write trainer/resume tests**

Test stage-mean loss, one optimizer step per effective episode batch, two-stage backward before detach, value parity between runtime and graph shadow, no gradient beyond chunk boundary, one route/commit per stage, predicted births feeding the ledger, rank-synchronous H, scheduler defined for all 3,000 updates, and resume restoring optimizer/scheduler/RNG/global episode/draw cursor.

- [ ] **Step 2: Run the tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_trainer.py tests/test_train_task_memory.py`

- [ ] **Step 3: Implement the common training engine**

Use base-subtree LR `1e-5`, new-module LR `1e-4`, AdamW with frozen R1 weight decay/betas, 5% warmup, cosine to 10%, existing gradient clip, and identical H1/T2/T3/T4/T5 20% episode buckets. W-BASE/FH-MATCH share stage supervision and budget but have no state gradient.

- [ ] **Step 4: Run the two-update real GPU smoke**

Use an episode with inherited entity, absence, reappearance, and newborn. Prove initial R1 subtree identity, expected zero-init first-step behavior, second-step nonzero adapter gradient/weight change, intended within-chunk gradient, detach boundary, and prediction-only route/birth lineage.

- [ ] **Step 5: Freeze configs and commit training path**

Require W-BASE to Q-INDEP diffs only in model/routing, and Q-INDEP to Q-TALA diffs only in supervision mode. Save resolved configs and shared initialization tensor hashes, then commit.

### Task 8: Implement compact evaluation, baseline metrics, and selection

**Files:**
- Create: `scripts/task_memory_cache.py`
- Create: `scripts/task_memory_metrics.py`
- Create: `scripts/evaluate_task_memory.py`
- Create: `scripts/select_task_memory.py`
- Create: `tests/test_task_memory_evaluation.py`
- Create: `tests/test_task_memory_selection.py`
- Generate: `artifacts/task_memory_retention_v2/training/{learning_curves.csv,terminal_update_comparison.csv,selected_checkpoint_comparison.csv,selected_checkpoints.json}`

**Interfaces:**
- Consumes: native/common populations, model checkpoint, fixed output policy/reducer, official post-processing/stmetrics, identity publisher, and compact cache root.
- Produces: one forward mask payload per stage, reducer-independent lineage tables, five task metrics, direct current AP, identity/error strata, retention, and frozen dev selection.

- [ ] **Step 1: Write evaluation/cache tests**

Cover native H1-H5, continuous empty-state T1..H execution, one lossless mask encoding shared by reducers, cache key binding full causal history/model/config/state/policy/postprocess, lag1 revision versioning, full-prefix Legacy AP, class-preserving trajectories, identity N/A denominators, false birth/reactivation/rejected birth, and cache cap rejection.

- [ ] **Step 2: Write selection tests**

Require checkpoints 0/750/1500/2250/3000; maximize minimum unrounded delta across named baselines, then mean delta, lower latency, earlier update; prohibit horizon-specific checkpoint/policy/reducer selection; emit both terminal-update and selected-best comparisons.

- [ ] **Step 3: Run tests and confirm RED**

Run: `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_task_memory_evaluation.py tests/test_task_memory_selection.py`

- [ ] **Step 4: Implement thin wrappers over official metrics**

Reuse `extract_official_task_prediction`, stmetrics accumulators, and existing identity event code. Stream by reference, release tensors after each unit, and keep diagnostic float logits only for preregistered samples.

- [ ] **Step 5: Validate and commit evaluation path**

Run new tests plus direct official-postprocess/system-metric tests. Execute one real two-master smoke for student and FH before formal M2.

### Task 9: Execute mandatory M2 and freeze its gate

**Files:**
- Update generated M2 training/evaluation artifacts only.
- Generate external: resumable checkpoints, optimizer/scheduler/RNG/draw cursor, and compact evaluation caches.

**Interfaces:**
- Consumes: four frozen M2 configs and common draw plan/initialization.
- Produces: complete 3,000-update W-BASE/Q-INDEP/Q-TALA/FH-MATCH runs, registered checkpoint evaluations, selected M2 parent, matched FH, and M3 authorization status.

- [ ] **Step 1: Measure the 300-update prefixes**

Run each arm as the first 300 updates of its 3,000-step schedule. Record actual GPU-hours, exposure, nonfinite/OOM status, gradients, and resume cursor; amend the campaign cap once only if measured throughput requires it.

- [ ] **Step 2: Resume every technically valid arm to 3,000 updates**

Do not early-stop for flat pilot AP. Verify resume state and run-directory isolation. Preserve `last` resumability separately from registered selected checkpoints.

- [ ] **Step 3: Evaluate all registered M2 checkpoints**

Evaluate common T2-T5 dev and native H2-H4 coverage with lag1/mean primary plus commit0 and reducer sensitivities from shared forward payloads. Continue FH from equal update/exposure.

- [ ] **Step 4: Apply the preregistered M3 gate**

Select Q-TALA when its minimum dev delta vs Q-INDEP is at least -1.0 pp and it has either a long-T gain at least 0.5 pp or matching identity/trajectory evidence. Otherwise apply the corresponding Q-INDEP vs W-BASE test. If neither has signal, permit exactly one evidence-based route/read fix and equal-budget repeat; otherwise mark `MECHANISM_UNSUPPORTED` and skip M3/M4.

- [ ] **Step 5: Commit compact M2 evidence**

Record COMPLETE/FAILED/GATE_SKIPPED separately, exact commands, checkpoints SHA/bytes, curves, terminal and selected comparisons, data/scan exposure, GPU-hours, and the frozen next-stage decision.

### Task 10: Implement and conditionally execute bounded visual memory

**Files:**
- Create when authorized: `models/object_visual_memory.py`
- Create when authorized: `tests/test_object_visual_memory.py`
- Create when authorized: `conf/task_memory_v2/{BASE-CONT,V-LAST,V-CORE}.yaml`
- Generate when authorized: `artifacts/task_memory_retention_v2/mechanism/{visual_selection.csv,read_content_interventions.csv}`
- Generate when authorized: `artifacts/task_memory_retention_v2/training/{M3_learning_curves.csv,M3_comparison.csv}`
- Generate when authorized: `artifacts/task_memory_retention_v2/resources/state_bytes.csv`

**Interfaces:**
- Consumes: frozen M2 parent, current-stage real aggregate segment features, prediction route, K=100/r=8/D=128 limits.
- Produces: `ObjectVisualState`, V-LAST/V-CORE selectors, matched-slot visual read, and exact state byte accounting.

- [ ] **Step 1: If M3 is authorized, write visual-memory tests**

Test current-stage-only writes, at most 32 candidates/entity, source-key deduplication, r=8, V-LAST newest-observation constraint, V-CORE 2 recent plus 6 quality/diversity representatives, stable ties, per-entity capacity, dormant preservation, generation invalidation, strict zero null residual, and permanent state no larger than 2 MiB.

- [ ] **Step 2: Implement current-forward visual extraction and selectors**

Bind `segment_features[0][batch]` rows to raw final mask logits and segment stage IDs. Predicted quality is class confidence times mask support quality; GT IoU/classes never enter selection.

- [ ] **Step 3: Run BASE-CONT/V-LAST/V-CORE for 1,500 updates**

Fork all arms from the same selected M2 checkpoint and shared new-parameter initialization. Keep data, LR schedule, read structure, and output policy identical. Continue FH for matched exposure.

- [ ] **Step 4: Run content interventions and apply the M4 gate**

Compare real history, read-off, previous-only, and same-shape unrelated content on frozen dev episodes. Call visual mechanism supported only when real long-term content beats BASE-CONT/latest controls on corresponding object masks or task evidence.

- [ ] **Step 5: Commit or record gate-skipped M3**

If unauthorized, add only the reason/state to `training/variants.json` and final status. Never create zero-valued visual result tables.

### Task 11: Conditionally implement prefix distillation and weak-stage proxy

**Files:**
- Create when authorized: `models/task_memory_distillation.py`
- Create when authorized: `tests/test_task_memory_distillation.py`
- Create when authorized: `conf/task_memory_v2/{STUDENT-CONT,STUDENT-KD,FH-CONT}.yaml`
- Generate when authorized: `artifacts/task_memory_retention_v2/mechanism/teacher_coverage.csv`
- Generate when authorized: `artifacts/task_memory_retention_v2/training/KD_pair_results.csv`

**Interfaces:**
- Consumes: one frozen FH teacher, original vertex alignment, frozen student parent, qualified GT-mediated training pairs, and remaining budget.
- Produces: temperature-2 class KL, soft mask BCE, differentiable zero on no qualified teacher, and one paired 1,000-update STUDENT-CONT/STUDENT-KD comparison.

- [ ] **Step 1: Apply the KD authorization conditions**

Require evidence that the student read changes the correct object's prediction but still trails FH, or that FH has qualified predictions at student weak stages. Freeze teacher before Protocol-B and require enough remaining budget for the whole pair.

- [ ] **Step 2: If authorized, write distillation tests**

Test prefix-only teacher access, original-vertex alignment, GT identity used only to pair loss targets, teacher class correctness and IoU at least 0.5, temperature 2, total KD weight 0.5 with equal class/mask split, no-target differentiable zero, and no change to base GT loss.

- [ ] **Step 3: Execute one paired KD experiment**

Fork STUDENT-CONT and STUDENT-KD from the same checkpoint for 1,000 updates with equal data/LR. Count teacher GPU-hours and target coverage. Continue FH to matched cumulative exposure.

- [ ] **Step 4: Optionally run one isolated weak-stage proxy pair**

Only when one reliable matched stage demonstrably bottlenecks the trajectory and the cap covers both arms. Recompute only the selected weak window from prediction/RNG snapshots; count the extra scan work.

- [ ] **Step 5: Commit or record gate-skipped M4**

Record exact authorization evidence, completed arms, coverage, costs, and negative results. Do not emit numeric tables for unrun variants.

### Task 12: Run M5, profile resources, and publish in two commits

**Files:**
- Create: `scripts/profile_task_memory.py`
- Create: `scripts/analyze_task_memory_final.py`
- Create: `scripts/publish_task_memory_v2.py`
- Create: `tests/test_profile_task_memory.py`
- Create: `tests/test_analyze_task_memory_final.py`
- Create: `tests/test_publish_task_memory_v2.py`
- Generate: remaining files under `artifacts/task_memory_retention_v2/final/`, `resources/`, `TEST_REPORT.md`, `FINAL_REPORT.md`, `FINAL_MANIFEST.json`, and `HANDOFF.md`.

**Interfaces:**
- Consumes: one frozen candidate, R1/FH real checkpoints, Protocol-B common 129, any independent native references, mechanism interventions, and fixed six-unit resource panel.
- Produces: all required status fields, all-T tables, per-reference evidence, retention/cost curves, content-addressed artifact manifest, results commit E, and publication commit P.

- [ ] **Step 1: Write final-analysis/profile/publication tests**

Require one checkpoint/policy/reducer across all T; five task metrics plus direct current AP; exact named-baseline deltas; retention A/R/Dmax; identity opportunities/attempts/correct/coverage/accuracy/recall and error counts; six-reference clustering; profile 5 warmups/10 repeats from cloned state; model/end-to-end/cumulative time; absolute/incremental/reserved GPU memory; CPU state/lag1/archive bytes; and non-self-referential manifest rules.

- [ ] **Step 2: Freeze one candidate and run Protocol-B once**

Evaluate the selected candidate, FH-R1 native/lag1, R1+B4 commit0/lag1, matched FH-CONT, and required long-memory controls on the same population. Run independent native references separately if available. Do not tune from Protocol-B.

- [ ] **Step 3: Run bounded mechanism/statistical analysis**

On final dev checkpoint, run real/read-off/previous/unrelated content. Report all six reference deltas; at most 1,000 reference resamples for final candidate vs final FH at T2/T5, otherwise label equal-reference intervals descriptive.

- [ ] **Step 4: Profile quality and resources on the same panel**

Sequentially profile candidate and strongest equivalent FH on one A40. Restore identical previous-state snapshots for repeats. Include model_update, end_to_end_update, output materialization, true T1..T cumulative cost, loaded/peak/reserved GPU bytes, CPU state, visual state, lag1 buffer, and archive.

- [ ] **Step 5: Derive honest final statuses and build compact package**

Populate `EXECUTION`, both all-T verdicts, 20-cell verdict, `RETENTION`, `RESOURCE`, `MECHANISM`, `GENERALIZATION`, and `PUBLICATION` independently. `FINAL_MANIFEST.json` excludes itself, HANDOFF, and publication receipt; HANDOFF records the manifest hash and 15 required sections.

- [ ] **Step 6: Run direct verification and freeze results commit E**

Run only the eight specified correctness classes, real GPU gate once, Ruff on this branch's changed Python, `git diff --check`, manifest/hash checks, and one necessary secret scan. Commit code/results as E, push, and verify remote E.

- [ ] **Step 7: Generate publication commit P and read back GitHub**

Write `results_commit=E` and verified-E state into final docs, commit as P, push, require local/remote P equality, then read back HANDOFF, FINAL_MANIFEST, and `final/all_t_metrics.csv` from P and compare SHA256. Report P/E and all three hashes in the final response.
