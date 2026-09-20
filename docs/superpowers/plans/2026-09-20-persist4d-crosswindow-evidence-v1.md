# Persist4D CrossWindow Evidence V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, execute, freeze, evaluate, and publish the CrossWindow V1 evidence campaign exactly as preregistered.

**Architecture:** Keep legacy D replay read-only and build a separate bounded CrossWindow path around a canonical candidate ledger, immutable `AssignmentPlan`, fixed-capacity resident state, one-scan buffer, and append-only publication archive. A single campaign CLI owns asset resolution, state transitions, budget accounting, locking, reporting, and E/P publication; production modules never accept GT, while diagnostics and evaluators consume production outputs through explicit adapters.

**Tech Stack:** Python 3.10, PyTorch 2.6, SciPy assignment helper already used by the repository, PyYAML, pytest, Ruff, existing Persist4D cache/metric modules.

**Spec:** `artifacts/crosswindow_evidence_v1/EXECUTION_INSTRUCTION.md`

## Global Constraints

- Parent commit is `96edca52d9d1cab7781bfa2a273baf9fe52b6a7d`; branch is `research/persist4d-crosswindow-evidence-v1`.
- R1 SHA256 is `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`; R1, backbone, decoder, loss, query count, labels, and official metrics remain frozen.
- Runtime association, commit, mask selection, and publication signatures contain no target or GT identity.
- State is bounded by `K=100`, previous-buffer groups by `Q=100`, and anchors by `K+Q=200`; archive is output-only.
- Execute only the preregistered 3 A0-U, 3 A1, and 6 A2 configurations; E3 is an equivalence audit, not another solver.
- `DEV-CAL`, `DEV-SEL`, adaptation, Protocol-B, and additional native references remain role-disjoint.
- GPU budgets are 22h inference/observation, 6h E6 training, and 4h final profiling, totaling at most 32 GPU-hours.
- Final Protocol-B scoring starts only after `selection/FINAL_LOCK.json` is committed.
- Missing assets and exhausted budgets produce structured statuses and partial recoverable artifacts; they never produce fabricated values.
- Only the eight direct-risk test categories in section 16 of the spec are run.

---

### Task 1: Campaign contract, asset resolution, and resumable state

**Files:**
- Create: `configs/crosswindow_evidence_v1.yaml`
- Create: `scripts/crosswindow_campaign.py`
- Create: `scripts/crosswindow_cache.py`
- Test: `tests/test_crosswindow_core.py`
- Materialize: `artifacts/crosswindow_evidence_v1/{DATA_ROLES.json,SOURCE_MANIFEST.json,RUN_STATE.json}`

**Interfaces:**
- Consumes: `EXPERIMENT_CONTRACT.json`, old `DATA_CONTRACT.json`, cache manifests, optional `--assets-from`, `PERSIST4D_*` environment variables.
- Produces: `load_campaign_config(path) -> CampaignConfig`, `resolve_assets(...) -> AssetResolution`, `atomic_write_run_state(...)`, deterministic `DEV-CAL`/`DEV-SEL` role files, and CLI subcommands with the common flags.

- [x] **Step 1: Write failing contract tests**

```python
def test_development_split_is_reference_stable_and_disjoint(tmp_path):
    roles = build_data_roles(DATA_CONTRACT, available_references=DEV_REFS)
    assert set(roles["DEV-CAL"]).isdisjoint(roles["DEV-SEL"])
    assert roles == build_data_roles(DATA_CONTRACT, available_references=reversed(DEV_REFS))

def test_resume_rejects_changed_identity(tmp_path):
    state = new_run_state(parent=PARENT, instruction_sha=INSTRUCTION_SHA, config_sha="a")
    atomic_write_run_state(tmp_path / "RUN_STATE.json", state)
    with pytest.raises(CampaignError, match="identity"):
        load_resume_state(tmp_path / "RUN_STATE.json", config_sha="b")
```

- [x] **Step 2: Verify RED**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py -k 'development_split or resume'`

Expected: import failure because campaign/cache APIs do not exist.

- [x] **Step 3: Implement config parsing, ordered asset resolution, role splitting, hashes, atomic state writes, and structured statuses**

The implementation must resolve `CLI -> env -> old external_assets.local.json -> known manifests`, emit null plus reason for unresolved keys, and never serialize private absolute paths into tracked artifacts.

- [x] **Step 4: Verify GREEN and bootstrap**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py -k 'development_split or resume'`

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign bootstrap --config configs/crosswindow_evidence_v1.yaml --assets-from ../persist4d-task-memory-retention-v2/artifacts/task_memory_retention_v2/external_assets.local.json --external-root /mnt/shared/ww/persist4d-crosswindow-evidence-v1`

- [x] **Step 5: Commit contract/bootstrap implementation**

```bash
git add configs/crosswindow_evidence_v1.yaml scripts/crosswindow_campaign.py scripts/crosswindow_cache.py tests/test_crosswindow_core.py artifacts/crosswindow_evidence_v1 docs/superpowers/plans/2026-09-20-persist4d-crosswindow-evidence-v1.md
git commit -m "feat: bootstrap crosswindow evidence campaign"
```

### Task 2: Canonical ledger and point-lineage correctness

**Files:**
- Modify: `scripts/crosswindow_cache.py`
- Modify: `tests/test_crosswindow_core.py`
- Produce: `artifacts/crosswindow_evidence_v1/e0/{index_trigger_counts.csv,correctness_fix_ledger.json}`

**Interfaces:**
- Consumes: `PredictionObservation`, `OfficialTaskPrediction`, `StageMeta`, scan vertex IDs, base/supplement cache records.
- Produces: `CandidateKey`, `CandidateSlice`, `QueryGroup`, `CanonicalFrame`, `align_mask(mask, from_ids, to_ids)`, and `iter_campaign_units(role)`.

- [ ] **Step 1: Write failing canonicalization and preservation tests**

```python
def test_align_mask_handles_non_identity_three_cycle():
    assert align_mask(torch.tensor([0, 0, 1], dtype=torch.bool), [20, 30, 10], [10, 20, 30]).tolist() == [1, 0, 0]

def test_candidate_ledger_preserves_multiclass_and_valid_query_union():
    frame = canonical_frame(observation=OBS, prediction=MULTICLASS_PREDICTION, vertex_ids=IDS, lineage=LINEAGE)
    assert len(frame.groups) == len(VALID_OR_EXPORTED_QUERIES)
    assert [candidate.class_id for candidate in frame.candidates] == ORIGINAL_CLASSES
    assert [candidate.score for candidate in frame.candidates] == ORIGINAL_SCORES
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py -k 'align_mask or candidate_ledger'`

- [ ] **Step 3: Implement strict ID validation, ascending canonical order, candidate keys, shared geometry groups, and logical-unit to physical-file binding**

Reject duplicate vertex IDs, missing destination IDs, silent `(entity,scan,class)` overwrite, and manifest lineage mismatches. Cache loads are one logical unit at a time and large-file hashes are memoized by path/size/mtime.

- [ ] **Step 4: Verify GREEN and inspect two real DEV units**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py -k 'align_mask or candidate_ledger'`

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign preflight --config configs/crosswindow_evidence_v1.yaml --read-only`

- [ ] **Step 5: Commit canonical cache layer**

```bash
git add scripts/crosswindow_cache.py tests/test_crosswindow_core.py artifacts/crosswindow_evidence_v1
git commit -m "feat: add canonical crosswindow candidate ledger"
```

### Task 3: Unified bounded state, evidence, assignment, and commit

**Files:**
- Create: `models/crosswindow_state.py`
- Create: `models/overlap_entity_association.py`
- Modify: `tests/test_crosswindow_core.py`

**Interfaces:**
- Produces: `CrossWindowState.empty(...)`, `EvidenceBundle`, frozen `AssignmentPlan`, `build_evidence(frame, state, previous_buffer)`, `associate(bundle, method_config)`, `commit_observation(frame, state, plan)`.
- Invariants: one-to-one real anchors plus private nulls; every unmatched group receives a distinct monotonic public ID; no eviction; K-full births are nonresident outputs; generation matches all inherited identities.

- [ ] **Step 1: Write failing assignment and lifecycle tests**

```python
def test_private_nulls_keep_two_unmatched_groups_distinct():
    plan = associate(BUNDLE_WITH_NO_ACCEPTED_EDGES, A0_DEFAULT)
    assert plan.entity_for_group[0] != plan.entity_for_group[1]

def test_full_capacity_rejects_residency_without_dropping_output():
    next_state, committed = commit_observation(FRAME_WITH_BIRTH, FULL_STATE, UNMATCHED_PLAN)
    assert committed.nonresident_groups == (0,)
    assert committed.public_id_for_group[0] >= 0
    assert next_state.occupied_count == FULL_STATE.capacity

def test_a2_missing_overlap_uses_base_score():
    scores = score_a2(BUNDLE_WITH_MISSING_OVERLAP, lambda_=0.25)
    assert scores[0, 0].item() == pytest.approx(BUNDLE_WITH_MISSING_OVERLAP.base[0, 0].item())
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py -k 'private_nulls or full_capacity or missing_overlap'`

- [ ] **Step 3: Implement frozen dataclasses, A0-U/A1/A2 formulas, stable assignment, single commit, resident/buffer union anchors, and byte accounting**

Use float32 for cosine/class/overlap scores; validate `class_prob >= 0` and row sums `<= 1+1e-5` without renormalizing; apply private dummy columns and strict `S > tau` acceptance.

- [ ] **Step 4: Verify GREEN, then run all core tests**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py`

- [ ] **Step 5: Commit state and association implementation**

```bash
git add models/crosswindow_state.py models/overlap_entity_association.py tests/test_crosswindow_core.py
git commit -m "feat: implement bounded crosswindow association state"
```

### Task 4: Lossless publication, E0 replay, and separated metrics

**Files:**
- Create: `scripts/replay_crosswindow_association.py`
- Create: `tests/test_crosswindow_metrics.py`
- Modify: `scripts/crosswindow_campaign.py`
- Produce: `artifacts/crosswindow_evidence_v1/e0/{source_parity.csv,t2_same_forward_parity.json,checkpoint_policy_matrix.csv}`

**Interfaces:**
- Consumes: canonical frames, immutable plans, old `run_control_trajectory`, official metric adapters.
- Produces: append-only `PublishedPrefix`, D-LEGACY/D-INDEXFIX/D0 rows, A0-U rows, `raw_current_AP`, `published_current_AP`, and parity ledgers.

- [ ] **Step 1: Write failing route/commit/publish and metric-isolation tests**

```python
def test_publish_consumes_plan_without_reassignment():
    prefix = publish(FRAME, PLAN, mask_selection="new", history_boundary=BOUNDARY)
    assert prefix.identity_map == PLAN.public_identity_map

def test_raw_current_ap_is_association_invariant():
    left = evaluate_raw_current(OFFICIAL_PREDICTION, TARGET)
    right = evaluate_raw_current(OFFICIAL_PREDICTION, TARGET, published_prefix=REASSOCIATED_PREFIX)
    assert left == right
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_metrics.py`

- [ ] **Step 3: Implement replay adapters, exact three-field evaluator boundary, archive canonicalization, collision fallback to a new identity, and separate raw/published current metrics**

- [ ] **Step 4: Verify GREEN and run E0 on DEV-CAL**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py tests/test_task_memory_controls.py tests/test_task_memory_output.py tests/test_system_comparison_metrics.py`

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --through E0 --resume`

- [ ] **Step 5: Commit E0 implementation and evidence**

```bash
git add scripts/replay_crosswindow_association.py scripts/crosswindow_campaign.py tests/test_crosswindow_metrics.py artifacts/crosswindow_evidence_v1/e0 artifacts/crosswindow_evidence_v1/RUN_STATE.json
git commit -m "feat: replay canonical crosswindow baselines"
```

### Task 5: E1 diagnostics, E2 calibration, and selection gates

**Files:**
- Create: `scripts/diagnose_crosswindow_failures.py`
- Modify: `scripts/replay_crosswindow_association.py`
- Modify: `scripts/crosswindow_campaign.py`
- Modify: `tests/test_crosswindow_metrics.py`
- Produce: `artifacts/crosswindow_evidence_v1/e1/*`, `artifacts/crosswindow_evidence_v1/e2/*`

**Interfaces:**
- Produces: policy/relaxed coverage ledgers, diagnostic-only GT-ID-RELAXED and GT-ID-LAG1, gate calculation, exact 12-config grid, CAL family ranking, fixed DEV-SEL evaluation, assignment events.

- [ ] **Step 1: Write failing gate/ranking tests for sufficient, inconclusive, and negative-headroom cases**
- [ ] **Step 2: Verify RED with `conda run -n persist4d python -m pytest -q tests/test_crosswindow_metrics.py -k 'headroom or ranking'`**
- [ ] **Step 3: Implement GT-only diagnostic module and deterministic ranking against fixed D0**
- [ ] **Step 4: Verify production modules do not import diagnostics and run through E2**

Run: `! rg -n 'diagnose_crosswindow|target|gt_id' models/crosswindow_state.py models/overlap_entity_association.py`

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --through E2 --resume`

- [ ] **Step 5: Commit E1/E2 code and evidence**

```bash
git add scripts/diagnose_crosswindow_failures.py scripts/replay_crosswindow_association.py scripts/crosswindow_campaign.py tests/test_crosswindow_metrics.py artifacts/crosswindow_evidence_v1/e1 artifacts/crosswindow_evidence_v1/e2 artifacts/crosswindow_evidence_v1/RUN_STATE.json
git commit -m "feat: add crosswindow diagnostics and fixed calibration"
```

### Task 6: E3 equivalence and E4 finite revision

**Files:**
- Create: `scripts/evaluate_crosswindow_consensus.py`
- Modify: `tests/test_crosswindow_core.py`
- Modify: `tests/test_crosswindow_metrics.py`
- Produce: `artifacts/crosswindow_evidence_v1/e3/*`, `artifacts/crosswindow_evidence_v1/e4/*`

**Interfaces:**
- Produces: exhaustive fixed-U objective check, `M-new`, `M-old`, `M-score`, `MASK_ONLY_FIXED_SCORE`, `SYSTEM_SELECTED_SCORE`, and fixed-parent selection results.

- [ ] **Step 1: Write failing fixed-U equivalence and revision-channel tests**
- [ ] **Step 2: Verify RED with `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py -k 'equivalence or mask_selector or score_channel'`**
- [ ] **Step 3: Implement exhaustive 2-3 group equivalence proof and identity-frozen revision selector**
- [ ] **Step 4: Verify GREEN and run through E4**

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign run --config configs/crosswindow_evidence_v1.yaml --through E4 --resume`

- [ ] **Step 5: Commit E3/E4 evidence**

```bash
git add scripts/evaluate_crosswindow_consensus.py tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py artifacts/crosswindow_evidence_v1/e3 artifacts/crosswindow_evidence_v1/e4 artifacts/crosswindow_evidence_v1/RUN_STATE.json
git commit -m "feat: audit consistency and evaluate finite revision"
```

### Task 7: Conditional E6 score head

**Files:**
- Create only if gate passes: `models/crosswindow_score_head.py`
- Create only if gate passes: `scripts/train_crosswindow_score.py`
- Modify: `scripts/crosswindow_campaign.py`
- Modify: `tests/test_crosswindow_core.py`
- Produce: `artifacts/crosswindow_evidence_v1/e6/*`

**Interfaces:**
- Consumes: isolated adaptation pairs and the frozen selected A2 configuration.
- Produces: two 1000-step arms, checkpoints at `{0,250,500,750,1000}`, fixed CAL selection, DEV-SEL result, or exact `SKIPPED_CONDITION` reasons.

- [ ] **Step 1: Evaluate all five authorization predicates before creating learning code**
- [ ] **Step 2: If unauthorized, write `e6/status.json` and verify every failed predicate has numeric/file evidence**
- [ ] **Step 3: If authorized, write failing feature-vector, zero-init, deterministic corruption, and two-step gradient tests**
- [ ] **Step 4: Implement and run exactly the preregistered schedule within the 6 GPU-hour budget**
- [ ] **Step 5: Commit E6 implementation/evidence or deterministic skip evidence**

### Task 8: Preflight review, selection lock, and commit C

**Files:**
- Modify: `scripts/crosswindow_campaign.py`
- Produce: `artifacts/crosswindow_evidence_v1/PREFLIGHT_REVIEW.md`
- Produce: `artifacts/crosswindow_evidence_v1/selection/FINAL_LOCK.json`

**Interfaces:**
- Produces: a 14-item preflight verdict and final immutable system definition, comparison list, evaluator identity, data hashes, head SHA/null, CAL/SEL evidence.

- [ ] **Step 1: Run the focused pytest and Ruff suite**

Run: `conda run -n persist4d python -m pytest -q tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py tests/test_task_memory_controls.py tests/test_task_memory_output.py tests/test_task_memory_routing.py tests/test_rescene_task_postprocess.py tests/test_system_comparison_metrics.py`

Run: `conda run -n persist4d ruff check models/crosswindow_state.py models/overlap_entity_association.py scripts/crosswindow_*.py scripts/replay_crosswindow_association.py scripts/diagnose_crosswindow_failures.py scripts/evaluate_crosswindow_consensus.py tests/test_crosswindow_core.py tests/test_crosswindow_metrics.py`

- [ ] **Step 2: Generate and inspect `PREFLIGHT_REVIEW.md`; resolve all numeric-correctness FAILs**
- [ ] **Step 3: Run lock command before any candidate Protocol-B scoring**

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign lock --config configs/crosswindow_evidence_v1.yaml`

- [ ] **Step 4: Commit the lock as commit C and record C in external run state**

```bash
git add artifacts/crosswindow_evidence_v1/PREFLIGHT_REVIEW.md artifacts/crosswindow_evidence_v1/selection/FINAL_LOCK.json artifacts/crosswindow_evidence_v1/RUN_STATE.json
git commit -m "chore: freeze crosswindow final selection"
```

### Task 9: Frozen E5 confirmation, capacity, and real profiling

**Files:**
- Create: `scripts/profile_crosswindow.py`
- Modify: `scripts/crosswindow_campaign.py`
- Modify: `tests/test_crosswindow_metrics.py`
- Produce: `artifacts/crosswindow_evidence_v1/final/{all_t_metrics.csv,identity_and_revision.csv,resources.csv,status.json}`

**Interfaces:**
- Consumes: `FINAL_LOCK.json`, 129 Protocol-B logical units, native FH payloads, K in `{16,32,100}`, real A40 forward path.
- Produces: common-cohort T2-T5 metrics, per-reference/LOO deltas, event strata, capacity rows, four timing scopes, memory bytes, `TMAP_ALL_T_PASS`, `RESOURCE_PASS`, and `JOINT_GOAL_PASS`.

- [ ] **Step 1: Write failing final-status and resource-gate tests, including null/unconfirmed behavior**
- [ ] **Step 2: Verify RED, implement the gate exactly, and verify GREEN**
- [ ] **Step 3: Run frozen confirmation with resume**

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign confirm --config configs/crosswindow_evidence_v1.yaml --resume`

- [ ] **Step 4: Check all locked methods share identical coverage and no final result changed the lock**
- [ ] **Step 5: Commit code and numeric results as experiment commit E**

```bash
git add scripts/profile_crosswindow.py scripts/crosswindow_campaign.py tests/test_crosswindow_metrics.py artifacts/crosswindow_evidence_v1
git commit -m "exp: record frozen crosswindow evidence results"
```

### Task 10: Report, manifest, E/P publication, and readback

**Files:**
- Modify: `scripts/crosswindow_campaign.py`
- Produce: `artifacts/crosswindow_evidence_v1/{FINAL_REPORT.md,HANDOFF.md,FINAL_MANIFEST.json,COMMANDS.md,PUBLICATION.json}`
- External only: `PUBLICATION_RECEIPT.local.json`, fallback git bundle on auth failure.

**Interfaces:**
- Produces: truthful outcome report, resumable commands, self-hash-free manifest, E/P remote verification, raw/API readback hashes, and final receipt.

- [ ] **Step 1: Generate reports from structured artifacts and verify every required file has a status/reason**

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign report --config configs/crosswindow_evidence_v1.yaml`

- [ ] **Step 2: Perform a scoped obvious token/key scan and push E; verify `ls-remote` and raw-readback main table bytes**
- [ ] **Step 3: Create documentation commit P bound to E, push P, and read back HANDOFF, MANIFEST, and main table**
- [ ] **Step 4: Write external receipt without committing it**

Run: `conda run -n persist4d python -m scripts.crosswindow_campaign publish --config configs/crosswindow_evidence_v1.yaml`

- [ ] **Step 5: Run completion audit against all 20 spec sections and the final checklist**

Run: `git status --short`

Run: `git ls-remote origin refs/heads/research/persist4d-crosswindow-evidence-v1`

Expected: remote SHA equals P; receipt hashes equal readback bytes; scientific FAIL remains explicitly distinct from execution/publication success.
