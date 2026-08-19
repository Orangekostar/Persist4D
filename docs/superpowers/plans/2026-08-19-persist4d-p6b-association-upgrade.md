# Persist4D P6-B Association Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, tune, and evaluate the evidence-driven P6-B association and memory-maintenance upgrade without changing frozen ReScene, P5, or P6-A behavior.

**Architecture:** A new P6-B memory module consumes frozen P6-A observations and produces P6-A-compatible tracking records. A staged, cluster-isolated validation runner selects one configuration, freezes it, evaluates two held-out reference-scene clusters once, and atomically publishes a schema-validated P6-B evidence bundle.

**Tech Stack:** Python 3.10, PyTorch, SciPy-compatible Hungarian logic, PyYAML, pytest, Ruff, existing P6-A cache/metric/artifact APIs.

---

### Task 1: P6-B Memory Configuration and Threshold-Aware Assignment

**Files:**
- Create: `models/persistent_memory_p6b.py`
- Create: `tests/test_p6b_memory.py`

- [ ] **Step 1: Write failing configuration and assignment tests**

Add tests that construct `P6BMemoryConfig`, reject non-finite or inconsistent
thresholds, and exercise the required counterexample:

```python
score = torch.tensor([[0.99, 0.74], [0.73, 0.49]], dtype=torch.float64)
pairs = threshold_aware_assignment(score, score >= 0.50)
assert pairs == ((0, 1), (1, 0))
```

Also test rectangular matrices, all-forbidden edges, exact ties, and forbidden
edges with large raw scores.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_p6b_memory.py
```

Expected: collection fails because `models.persistent_memory_p6b` does not exist.

- [ ] **Step 3: Implement immutable configuration and assignment**

Define:

```python
@dataclass(frozen=True)
class P6BMemoryConfig:
    capacity: int = 100
    active_threshold: float = 0.50
    reactivation_threshold: float = 0.85
    reactivation_margin: float = 0.10
    class_weight: float = 0.25
    class_mode: str = "foreground_normalized"
    background_class: int = 18
    update_rate: float = 0.20
    max_update_rate: float = 0.20
    consolidation_confidence: float | None = 0.90
    consolidation_margin: float | None = 0.10
    birth_confidence: float = 0.75
    birth_minimum_mask_support: int = 128
    birth_max_entropy: float | None = 0.50
    mask_threshold: float = 0.50
    assignment_mode: str = "threshold_aware"
```

Implement `threshold_aware_assignment(score, allowed)` with dummy unmatched
nodes and deterministic stable ties. It must maximize match cardinality first,
then allowed score, then stable low-index alignment.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Task 1 test file and expect all tests to pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add models/persistent_memory_p6b.py tests/test_p6b_memory.py
git commit -m "feat: add threshold-aware P6B assignment"
```

### Task 2: Active/Dormant Association, Consolidation, and Birth Gates

**Files:**
- Modify: `models/persistent_memory_p6b.py`
- Modify: `tests/test_p6b_memory.py`

- [ ] **Step 1: Write failing transition tests**

Add independent tests proving:

```python
assert dormant_edge_allowed is False  # score passes active but not react threshold
assert low_margin_dormant_edge_allowed is False
assert accepted_active_match_updates_identity is True
assert low_confidence_match_is_consolidated is False
assert low_support_birth_is_rejected is True
assert high_entropy_birth_is_rejected is True
```

Verify a matched-but-not-consolidated slot becomes active/last-seen while its
embedding/class/confidence tensors remain bitwise unchanged. Verify birth
rejection does not consume capacity. Verify GT-like fields are not accepted by
the runtime API.

- [ ] **Step 2: Run focused tests and verify RED**

Expected failures must identify missing `P6BPersistentMemory.step` behavior.

- [ ] **Step 3: Implement the causal transition**

Add `P6BStepResult` with query-aligned slot, score, feature/class score, margin,
reactivation, consolidation, and birth-rejection tensors. Compute full or
foreground-normalized class compatibility, derive active/dormant allowed-edge
masks, run the configured assignment, then apply consolidation and birth gates.
Reuse `PersistentMemoryState` without changing its layout.

- [ ] **Step 4: Run P6-B and frozen P5 memory tests**

```bash
python -m pytest -q tests/test_p6b_memory.py tests/test_persistent_memory.py
```

Expected: all CPU tests pass; CUDA-only tests skip when CUDA is hidden.

- [ ] **Step 5: Commit Task 2**

```bash
git add models/persistent_memory_p6b.py tests/test_p6b_memory.py
git commit -m "feat: gate P6B reactivation and consolidation"
```

### Task 3: P6-A-Compatible P6-B Tracker Adapter

**Files:**
- Create: `scripts/p6b_association.py`
- Create: `tests/test_p6b_association.py`

- [ ] **Step 1: Write failing adapter tests**

Test one five-stage synthetic sequence with active matches, dormant matches,
non-consolidated matches, accepted births, and rejected births. Assert exact
`TrackStep` fields, `AssociationDiagnostics`, immutable state snapshots, and
query-aligned diagnostics. Compare frozen B4 output before and after importing
the new adapter.

- [ ] **Step 2: Verify RED**

Run `tests/test_p6b_association.py`; expect import failure.

- [ ] **Step 3: Implement `P6BTracker`**

The adapter owns `P6BPersistentMemory`, converts `FrozenObservation` to the
existing local-observation contract, and returns a P6-A `TrackStep` with method
`P6B`. It must not import targets, GT IDs, or metric code.

- [ ] **Step 4: Verify GREEN and B4 regression**

Run:

```bash
python -m pytest -q tests/test_p6b_association.py tests/test_p6a_association.py
```

- [ ] **Step 5: Commit Task 3**

```bash
git add scripts/p6b_association.py tests/test_p6b_association.py
git commit -m "feat: expose P6B tracker for cached evaluation"
```

### Task 4: Deterministic Cluster Split and Search Space

**Files:**
- Create: `conf/p6b/default.yaml`
- Create: `scripts/p6b_protocol.py`
- Create: `tests/test_p6b_protocol.py`

- [ ] **Step 1: Write failing split/search tests**

Given the frozen P6-A protocol manifest, assert the SHA256 ordering uses
`p6b|45|<reference-id>`, yields four disjoint tuning clusters/32 masters and
two held-out clusters/11 masters, and keeps every master/order in one split.
Assert exact staged grid values and deterministic canonical configuration IDs.

- [ ] **Step 2: Verify RED**

Run the protocol tests and expect the module/config to be absent.

- [ ] **Step 3: Implement split and config parsing**

Add typed `P6BSplitManifest`, canonical config serialization, grid expansion,
neighbor generation, and exact config validation. Decode `conf/p6b/default.yaml`
with `yaml.safe_load` and reject missing/extra fields.

- [ ] **Step 4: Verify GREEN**

Run protocol and P6-A protocol regression tests.

- [ ] **Step 5: Commit Task 4**

```bash
git add conf/p6b/default.yaml scripts/p6b_protocol.py tests/test_p6b_protocol.py
git commit -m "feat: preregister P6B validation split"
```

### Task 5: Fast Identity Sweep and Official Finalist Selection

**Files:**
- Create: `scripts/p6b_sweep.py`
- Create: `tests/test_p6b_sweep.py`

- [ ] **Step 1: Write failing causal-replay and selection tests**

Assert one T5 replay produces prefix-identical T2-T5 steps, candidate rows use
only tuning cluster IDs, ineligible candidates cannot win, ranking follows all
six specified keys, ties use canonical config JSON, and official task metrics
are required before final selection.

- [ ] **Step 2: Verify RED**

Run the new test file; expect import failure.

- [ ] **Step 3: Implement staged sweep**

Implement one-pass causal replay, prefix event derivation, identity/reactivation/
birth aggregates, staged grid evaluation, Pareto finalist extraction, official
metric evaluation for finalists, and deterministic selection. Preserve every
candidate row and its eligibility reasons.

- [ ] **Step 4: Verify GREEN and metric regression**

Run the sweep tests with `tests/test_p6a_metrics.py` and
`tests/test_p6a_analysis.py`.

- [ ] **Step 5: Commit Task 5**

```bash
git add scripts/p6b_sweep.py tests/test_p6b_sweep.py
git commit -m "feat: select P6B configuration without holdout leakage"
```

### Task 6: P6-B Artifact Contract and Figures

**Files:**
- Create: `scripts/p6b_artifacts.py`
- Create: `scripts/p6b_figures.py`
- Create: `tests/test_p6b_artifacts.py`
- Create: `tests/test_p6b_figures.py`

- [ ] **Step 1: Write failing schema and rendering tests**

Require exact root keys, scalar finite values, exact CSV columns, eleven report
sections, one terminal decision, portable references, source/cache/P5/P6-A
hashes, selected-config hash, split hash, and byte/hash manifest binding. Reject
missing ablations, final/tuning overlap, unsupported claims, private paths,
symlinks, and non-atomic overwrite.

- [ ] **Step 2: Verify RED**

Run both artifact test files; expect import failures.

- [ ] **Step 3: Implement canonical bundle rendering**

Render root JSON, required CSV/YAML/Markdown, and deterministic SVG figures from
one validated root mapping. Publish through a temporary sibling directory and
`os.replace`; refuse an existing destination.

- [ ] **Step 4: Verify GREEN**

Run P6-B artifacts/figures and P6-A artifact regression tests.

- [ ] **Step 5: Commit Task 6**

```bash
git add scripts/p6b_artifacts.py scripts/p6b_figures.py tests/test_p6b_artifacts.py tests/test_p6b_figures.py
git commit -m "feat: define canonical P6B evidence bundle"
```

### Task 7: Sweep/Final CLI and Source-Bound Gates

**Files:**
- Create: `scripts/run_p6b_evaluation.py`
- Create: `tests/test_p6b_runner.py`
- Create: `tests/test_p6b_gpu_gate.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing runner tests**

Test separate `sweep` and `final` commands, exact cache/P6-A provenance, clean
source requirements, frozen selection hash, held-out-only final inputs, refusal
to overwrite, atomic cleanup on failure, final GO/STOP computation, and
opt-in real-cache artifact reconstruction.

- [ ] **Step 2: Verify RED**

Run runner/gate tests; expect missing CLI/module failures and opt-in skips.

- [ ] **Step 3: Implement CLI and ignored raw outputs**

Add:

```text
python scripts/run_p6b_evaluation.py sweep --cache-directory ... --output-root ...
python scripts/run_p6b_evaluation.py final --cache-directory ... --selection-root ... --output-root ...
```

Ignore only raw local P6-B files that exceed GitHub limits; keep all compact
reports, sweeps, tables, configs, manifests, and figures tracked.

- [ ] **Step 4: Verify GREEN**

Run runner tests and the non-opt-in gate. Confirm the gate skips only when real
artifacts are absent or opt-in is unset.

- [ ] **Step 5: Commit Task 7**

```bash
git add .gitignore scripts/run_p6b_evaluation.py tests/test_p6b_runner.py tests/test_p6b_gpu_gate.py
git commit -m "feat: run source-bound P6B evaluation"
```

### Task 8: Controlled Real-Cache Sweep and Held-Out Final Evaluation

**Files:**
- Create: `artifacts/P6B/*` through the runner only

- [ ] **Step 1: Verify clean source and frozen inputs**

Check current HEAD, clean tracked/index state, P5/P6-A hashes, canonical
checkpoint hash, Protocol-B manifest, and all 645 cache entries.

- [ ] **Step 2: Run validation sweep**

Use the frozen external P6-A cache and write the sweep bundle outside the repo.
Validate every sweep table, eligibility field, selected config, and source
binding before committing the compact selection evidence.

- [ ] **Step 3: Freeze selection in a source commit**

Commit the selected configuration and compact sweep outputs. Do not inspect any
held-out metric before this commit.

- [ ] **Step 4: Run held-out final exactly once**

Run `final` from the frozen selection commit. Record command, exit code, start/
end time, selected config hash, split hash, final metrics, paired statistics,
failure analysis, and GO/STOP gates.

- [ ] **Step 5: Publish and commit compact final evidence**

Keep oversized raw event/root files local and ignored when necessary. Commit
the complete compact evidence package without modifying P5/P6-A artifacts.

### Task 9: Completion Verification and Release

**Files:**
- Verify all P6-B source/tests/artifacts

- [ ] **Step 1: Run focused P6-B and P6-A regression suites**

Run all `test_p6b_*.py`, P6-A association/metrics/artifacts/runner tests, and
the frozen P5 memory/streaming tests with the exact worktree third-party paths.

- [ ] **Step 2: Run full CPU suite**

Use `PYTHONNOUSERSITE=1`, hide CUDA, include all four worktree third-party
repositories in `PYTHONPATH`, and require zero failures.

- [ ] **Step 3: Run opt-in real-cache gate**

Set the P6-B artifact verification flag and cache/metadata paths. Require exact
bundle reconstruction, provenance, split isolation, privacy, and clean source.

- [ ] **Step 4: Run static and privacy checks**

Run Ruff, `py_compile`, `git diff --check`, artifact path privacy, maximum blob
size, credential-pattern scan, and final clean status.

- [ ] **Step 5: Finish branch**

Use `superpowers:finishing-a-development-branch`, preserve the worktree, push
`research/persist4d-p6b` to `origin`, and do not start P7/P8.
