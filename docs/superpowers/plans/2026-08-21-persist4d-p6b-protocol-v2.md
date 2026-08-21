# Persist4D P6-B Protocol V2 Implementation Plan

**Goal:** Replace invalid P6-B protocol-v1 evidence with a strict, cache-only,
leakage-free protocol-v2 evaluation while preserving frozen ReScene, P5, P6-A,
and the historical audit trail.

**Execution:** Use test-driven development. Do not run GPU inference, enter
P7/P8, inspect held-out results during tuning, or overwrite a successful raw
held-out payload.

### Task 1: Cache Support and Assignment Semantics

**Files:** `models/persistent_memory_p6b.py`,
`scripts/evaluate_persist4d_p6a.py`, `tests/test_p6b_memory.py`,
`tests/test_p6b_runner.py`

- [ ] Add RED tests proving cached `[True, False, False]` support is one, the
  persisted support is used, and inconsistency fails closed.
- [ ] Add RED tests for maximum-total-score dummy assignment and strict dormant
  margin.
- [ ] Implement explicit query-aligned support, remove the cardinality bonus,
  and enforce strict dormant margin.
- [ ] Run P6-B memory/cache and frozen P5/P6-A regression tests.

### Task 2: Normalized Sweep Statistics and Selection

**Files:** `scripts/p6b_sweep.py`, `scripts/p6b_protocol.py`,
`scripts/run_p6b_evaluation.py`, `conf/p6b/default.yaml`,
`tests/test_p6b_sweep.py`, `tests/test_p6b_protocol.py`,
`tests/test_p6b_runner.py`

- [ ] Add RED tests for all denominators, count/rate identities, paired
  reference-cluster aggregation, and count-versus-rate ranking counterexamples.
- [ ] Extend candidate schemas and use normalized paired ranking objectives.
- [ ] Add RED tampering tests for candidate rows, grid coverage, stage winners,
  finalists, ranking keys, selected config, split membership, and GT-free proof.
- [ ] Implement a single strict selection validator that recomputes all derived
  decisions from preregistered inputs.

### Task 3: Exactly-Once Held-Out Evaluation

**Files:** `scripts/run_p6b_evaluation.py`, `scripts/p6b_artifacts.py`,
`tests/test_p6b_runner.py`, `tests/test_p6b_artifacts.py`

- [ ] Add RED tests for durable attempt creation, crash recovery, successful
  payload immutability, second-evaluation refusal, and package-only retries.
- [ ] Split the final command into `final-evaluate` and `final-package`.
- [ ] Bind the immutable raw payload and canonical attempt ledger into the
  artifact manifest and G6B-5 validator.

### Task 4: Paired Cluster Statistics and Artifact Contract

**Files:** `scripts/run_p6b_evaluation.py`, `scripts/p6b_artifacts.py`,
`tests/test_p6b_runner.py`, `tests/test_p6b_artifacts.py`

- [ ] Add RED tests for exact master/order/horizon pairing, duplicate/missing
  pair rejection, cluster-level deltas, deterministic seed-45 bootstrap, and
  mean/std/95% interval serialization.
- [ ] Implement 10,000-replicate reference-scene cluster bootstrap.
- [ ] Strengthen artifact validation to require complete sweep/per-sequence/
  failure/statistical populations and recompute G6B-1 through G6B-5.
- [ ] Render numeric failed-gate evidence, inactive components, uncertainty, and
  exact execution history.

### Task 5: Invalidate Protocol V1 and Rerun CPU Evidence

**Files:** `artifacts/P6B_selection/`, `artifacts/P6B/`

- [ ] Commit an explicit protocol-v1 invalidation/removal while retaining its
  Git history.
- [ ] From a clean source commit, run the full tuning sweep on tuning clusters.
- [ ] Validate and commit the frozen protocol-v2 selection.
- [ ] Run `final-evaluate` exactly once on held-out clusters.
- [ ] Run/retry `final-package` until all artifact contracts pass.

### Task 6: Release Verification

- [ ] Run focused P6-B tests and frozen P5/P6-A regression tests.
- [ ] Run Ruff, compile checks, `git diff --check`, privacy checks, renderer
  reproducibility, manifest verification, and full CPU pytest.
- [ ] Request independent code/protocol review and resolve every Critical or
  Important finding.
- [ ] Commit only source-sized code, configs, docs, tests, and evidence; exclude
  caches, checkpoints, datasets, and other large files.
- [ ] Push the verified branch to `git@github.com:Orangekostar/Persist4D.git`.
