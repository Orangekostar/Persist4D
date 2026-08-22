# Full-History vs Persistent-State Implementation Plan

> **For Codex:** Execute this plan in the current isolated worktree. Keep the
> ReScene and Persist4D method implementations frozen. Use tests before each
> implementation step and stop after the final system-comparison report.

**Goal:** Compare ReScene4D full-history inference against the frozen P6-A B4
single-state Persist4D incumbent on the exact Protocol-B prefixes, including
causal task quality, deployment identity, gap recovery, compute scaling, paired
cluster statistics, LOSO robustness, figures, and a four-way final decision.

**Architecture:** Add a fail-closed evaluation layer around the existing dataset,
collator, checkpoint loader, official `stmetrics` adapter, and frozen B4 tracker.
Large tensor caches remain local; canonical manifests, hashes, metrics, figures,
audits, and reports are tracked. Full-History query index is preserved as its
issued ID, while task-quality postprocessing remains the official ReScene path.

**Runtime:** Python 3.10 from the `persist4d` conda environment, PyTorch 2.6,
three NVIDIA A40 GPUs, pytest, Hydra/OmegaConf, stmetrics, NumPy, PyYAML.

**Frozen inputs:** Base commit `73b83ced10a59c4ba755e94fad5fbf43c35d90e8`,
checkpoint SHA256 `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`,
P6-A source `cee151a9dfc1c9aa038227bc4e179b671e739575`, Protocol-B
manifest SHA256 `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`.

## Common Runtime Prefix

All verification and experiment commands use:

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/third_party/concerto:$PWD/third_party/detectron2:$PWD/third_party/sonata:$PWD/third_party/stmetrics:$PWD"
export PERSIST4D_PYTHON=/home/ww/miniconda3/envs/persist4d/bin/python
```

## Task 1: Freeze And Verify The P6-A Incumbent

**Files:**

- Create: `configs/system_comparison/persist4d_incumbent.yaml`
- Create: `scripts/system_comparison_protocol.py`
- Create: `tests/test_system_comparison_protocol.py`
- Modify: `.gitignore`

1. Write failing tests that require the exact B4 parameters, source/config/report/
   result/checkpoint hashes, 43 masters, 6 reference clusters, three registered
   orders, 516 T2-T5 prefixes, and strict nested-prefix equality.
2. Run the new protocol tests and confirm they fail because the files do not yet
   exist.
3. Add the incumbent YAML with the frozen B4 values, decision thresholds, profile
   settings (5 warmups, 10 repeats), and portable hash bindings.
4. Implement strict YAML loading, source hash validation, P6-A result-row
   regression, and a system manifest builder that copies rather than resamples
   Protocol-B identities.
5. Ignore only large cache entry directories; keep their manifests trackable.
6. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_protocol.py
```

## Task 2: Record The Full-History Code Audit

**Files:**

- Create: `scripts/system_comparison_audit.py`
- Create: `tests/test_system_comparison_audit.py`
- Create: `artifacts/system_comparison/REScene_FULL_HISTORY_CODE_AUDIT.md`

1. Write failing tests for checkpoint horizon extraction, runtime T>2 evidence,
   query namespace classification, change-label exclusion, and all eight required
   audit answers with file/function/behavior/scientific-implication fields.
2. Implement a read-only checkpoint/config auditor. It must identify training
   `T=2`, validation/test `T=2`, RIO `T=2`, ScanNet `T=1`, arbitrary prefix loading,
   temporal overlay sharing, non-parametric FPS queries, no persistent track ID,
   official top-k query-class postprocessing, and `change_file=None` evaluation.
3. Render the code audit from validated evidence and run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_audit.py
```

## Task 3: Implement Content-Addressed Full-History Inference

**Files:**

- Create: `scripts/system_comparison_inference.py`
- Create: `tests/test_system_comparison_inference.py`

1. Write failing tests for packed boolean-mask round trips, causal cache keys,
   immutable cache publication, official task output shape, raw-query identity
   preservation, and exact T2 observation fingerprint comparison.
2. Implement a deterministic runtime context and exact Protocol-B request
   resolver. Identity initialization may use T1, but reported task horizons remain
   exactly T2-T5.
3. Implement one full-history forward `R(S1:St)` using the existing dataset,
   collator, checkpoint loader, and trainer postprocessing. Preserve raw query
   indices only for identity analysis; do not attach memory or post-hoc tracking.
4. Store packed full-resolution masks, remapped predictions/targets, point/input
   counts, observed scan IDs, fingerprints, and complete provenance in atomic
   `.pt` entries. Support deterministic disjoint shards and a fail-closed finalize
   command.
5. Reuse `RealPredictionCacheProducer` for the persistent local cache rather than
   changing local perception.
6. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_inference.py tests/test_p6a_cache.py tests/test_rescene_query_features.py
```

## Task 4: Implement Causal Task And Deployment Identity Metrics

**Files:**

- Create: `scripts/system_comparison_metrics.py`
- Create: `tests/test_system_comparison_metrics.py`

1. Write failing tests for future-prefix rejection, current-stage restriction,
   deployment switch opportunities/rate, fragmentation, merge, and gap identity
   recovery accuracy/recall.
2. Implement a causal pair validator that rejects any scan ID or temporal-stage
   index outside the declared prefix before calling `OfficialMetricAccumulator`.
3. Compute prefix `t_mAP`, `t_mAP50`, `t_mAP25`, `t_REC` and explicitly labeled
   current-stage AP.
4. Match issued identities to current-stage GT with the frozen class-compatible
   Hungarian IoU>=0.5 rule. Define switches over comparable published IDs,
   fragmentation as extra distinct IDs per GT, merge as extra distinct GTs per
   issued ID, and their explicit denominators.
5. Define a gap opportunity as visible, absent for at least one update, then
   visible again. Accuracy is correct recoveries/attempts; recall is correct
   recoveries/opportunities. Keep dormant reactivation separate.
6. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_metrics.py tests/test_p6a_metrics.py tests/test_p6a_analysis.py
```

## Task 5: Implement Determinism, Timing, VRAM, And State Profiling

**Files:**

- Create: `scripts/profile_system_comparison.py`
- Create: `tests/test_profile_system_comparison.py`

1. Write failing tests with instrumented callbacks proving synchronization before
   and after timing, peak allocated/reserved reset and collection, warmup exclusion,
   median/mean/std calculation, and tracker reset/preroll.
2. Implement a six-cluster profile subset: lexicographically first master per
   reference cluster, canonical order, shared by both systems at T2-T5.
3. Preload and transfer inputs outside the timed boundary. For each sample use five
   warmups and ten measured repeats. Time Full-History forward; time Persist4D local
   forward plus B4 update with fresh state and untimed causal preroll.
4. Record allocated/reserved peak MiB, full point/input volume, scans processed,
   B4 tensor-storage bytes, and failure rows without dropping a horizon.
5. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_profile_system_comparison.py tests/test_p6a_efficiency_profiler.py
```

## Task 6: Build The Evaluation Runner And Smoke Gates

**Files:**

- Create: `scripts/run_system_comparison.py`
- Create: `tests/test_run_system_comparison.py`

1. Write failing orchestration tests for stage ordering, resume behavior, no
   overwrite on provenance mismatch, incumbent regression blocking, T2 regression
   blocking, determinism blocking, and Oracle conditional execution.
2. Implement commands for `bind`, `cache-local-shard`, `cache-full-shard`,
   `finalize-caches`, `smoke`, `evaluate`, `profile`, and `all`.
3. The smoke gate uses three canonical T5 prefixes from distinct reference
   clusters, repeats each three times, and compares mask/class/ID/score
   fingerprints. It also compares Full-History T2 observation fingerprints with
   the identical P6-A local T2 input.
4. Persist4D evaluation must use the exact frozen B4 factory and verify aggregate
   T2-T5 values against `artifacts/P6A/strict_online_results.csv` at absolute
   tolerance `1e-12` before Full-History results are accepted.
5. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_run_system_comparison.py
```

## Task 7: Implement Paired Statistics And Robustness

**Files:**

- Create: `scripts/system_comparison_analysis.py`
- Create: `tests/test_system_comparison_analysis.py`

1. Write failing tests for exact method/horizon/order coverage, paired reference-
   cluster aggregation, 10,000 bootstrap resamples at seed 45, relative difference,
   six LOSO drops, and order-direction reversal detection.
2. Implement paired `Persist4D - FullHistory` statistics for t-mAP, t-REC, IDSW
   rate, gap recovery, and latency. Treat six reference scenes as the independent
   units; never treat 129 orders as independent.
3. Produce `per_sequence_results.csv`, `per_order_results.csv`,
   `cluster_bootstrap.csv`, and `leave_one_scene_out.csv` with finite-or-explicitly-
   missing values and deterministic ordering.
4. Trigger Oracle only when Full-History exceeds Persist4D by at least 0.01 t-mAP
   at T4 or T5 and the paired CI for `Persist4D-FullHistory` is wholly below zero.
5. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_analysis.py
```

## Task 8: Generate Tables, Figures, And Artifact Contract

**Files:**

- Create: `scripts/system_comparison_figures.py`
- Create: `scripts/build_system_comparison_artifacts.py`
- Create: `tests/test_system_comparison_figures.py`
- Create: `tests/test_system_comparison_artifacts.py`

1. Write failing tests for Table A/B exact coverage, six valid SVGs, manifest/hash
   completeness, ten report answers, zero-shot naming, and exactly one allowed
   final classification.
2. Render publication SVGs for task quality, IDSW rate, gap recovery, latency,
   peak VRAM, and accuracy-compute Pareto without non-data decorations.
3. Build all required CSV/JSON/Markdown artifacts atomically. Table A includes
   Full-History, B3 EMA association, and B4 Persist4D; Table B reports measured
   values and never labels explicit history as state bytes.
4. Implement the preregistered decision rule:
   `SYSTEM_LOCK` for non-inferior task plus identity/compute advantage;
   `SYSTEM_PARETO_LOCK` for no meaningful accuracy deficit plus identity/compute
   advantage; otherwise use Oracle recovery (within 0.01 or >=80% deficit closed)
   to select `ASSOCIATION_LIMITED` versus `REPRESENTATION_LIMITED`.
5. Run:

```bash
$PERSIST4D_PYTHON -m pytest -q tests/test_system_comparison_figures.py tests/test_system_comparison_artifacts.py
```

## Task 9: Run Real Smoke, Full Accuracy, And Profiling

**Files generated:**

- `artifacts/system_comparison/FULL_HISTORY_DETERMINISM_AUDIT.md`
- `artifacts/system_comparison/system_comparison_manifest.json`
- `artifacts/system_comparison/reproducibility_binding.json`
- `artifacts/system_comparison/full_history_predictions/manifest.json`
- `artifacts/system_comparison/persistent_predictions/manifest.json`
- All required result CSVs and figure SVGs

1. Commit the implementation and tests so cache provenance points to an immutable
   source commit. Confirm the tracked tree is clean.
2. Build local and full-history cache shards on CUDA devices 0, 1, and 2 with
   disjoint shard indices, wait for every process, then finalize both manifests.
3. Run the incumbent/T2/determinism smoke gates. Do not continue on failure.
4. Run full CPU evaluation for all 43 masters x 3 orders x T2-T5.
5. Run the shared six-cluster profiling subset with 5+10 repeats.
6. Run paired bootstrap, LOSO, order robustness, tables, figures, conditional
   Oracle attribution, and the final report builder.

## Task 10: Final Verification And Stop

**Files:**

- Finalize: `artifacts/system_comparison/SYSTEM_COMPARISON_GO_NOGO_REPORT.md`

1. Run all new tests, relevant P6-A regressions, compile checks, and artifact
   verifier.
2. Run the full historical test suite with CUDA hidden and record the result.
3. Inspect `git diff --check`, the complete diff, cache manifest hashes, all 19
   execution steps, all 10 report answers, six figures, two tables, and the single
   final classification.
4. Commit lightweight evidence. Do not train T3 ReScene, integrate external
   modules, add dual-timescale memory, or implement a query adapter.
5. Stop and report changed files, scientific outcome, and verification commands.

