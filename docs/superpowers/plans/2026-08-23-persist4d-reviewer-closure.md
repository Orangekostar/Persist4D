# Persist4D Reviewer-Closure Implementation Plan

> **For Codex:** Execute gate by gate in this worktree. Write each behavioral
> test before implementation, preserve `artifacts/system_comparison/`, and stop
> after the final 12-section method-lock report.

**Goal:** Challenge Persist4D with reused trivial cross-prefix trackers and one
fair T3-adapted ReScene4D model, then explain the remaining performance ceiling
with IoU, coverage, Oracle, and failure-decomposition evidence.

**Architecture:** Add a fail-closed reviewer-closure layer around the frozen
Protocol-B manifest, Full-History inference path, P6-A trackers, official metric
adapter, and training entry point. Publish only portable manifests and evidence;
keep large tensor payloads ignored.

**Runtime:** `/home/ww/miniconda3/envs/persist4d/bin/python`, PyTorch 2.6,
three NVIDIA A40 GPUs, Hydra/OmegaConf, stmetrics, pytest, NumPy, PyYAML.

## Common Runtime

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/third_party/detectron2:$PWD/third_party/concerto:$PWD/third_party/sonata:$PWD/third_party/stmetrics:$PWD"
export PERSIST4D_PYTHON=/home/ww/miniconda3/envs/persist4d/bin/python
```

## Task 1: Freeze Reviewer-Closure Inputs

**Files:** create `configs/reviewer_closure/protocol.yaml`,
`scripts/reviewer_closure_protocol.py`, `tests/test_reviewer_closure_protocol.py`;
modify `.gitignore` only for new large entry directories.

1. Test exact commits/hashes, immutable system-comparison tree, 43 masters, 6
   clusters, three orders, 516 T2-T5 prefixes, and strict prefix nesting.
2. Implement portable bindings and a copied reviewer-closure manifest.
3. Verify with `pytest -q tests/test_reviewer_closure_protocol.py`.

## Task 2: Build Full-History Observation Sidecars

**Files:** create `scripts/reviewer_closure_sidecar.py`,
`tests/test_reviewer_closure_sidecar.py`; minimally modify
`scripts/system_comparison_inference.py` only if needed to expose the already
constructed raw observation without changing old payloads.

1. Test schema, content hashes, atomic refusal, mask/query alignment, current-stage
   restriction, and old-cache immutability.
2. Implement `full_history_observations_v2` serialization and manifest finalizer.
3. Test one real canonical prefix, then generate the required 516 O2-O5
   master/order/horizon sidecars on one deterministic GPU process and finalize
   exact coverage; do not generate T1 sidecars.

## Task 3: Evaluate Reused Trivial Trackers

**Files:** create `scripts/reviewer_closure_tracking.py`,
`tests/test_reviewer_closure_tracking.py`.

1. Prove synthetic output equality with the frozen P6-A tracker classes and runner.
2. Test no-future access, per-master/order reset, namespace independence, and the
   `P2 visible -> P3/P4 absent -> P5 visible` gap case.
3. Evaluate Native Full-History, Pairwise Feature, Pairwise Feature-Class, EMA
   Temporal, Persist4D, and the explicitly diagnostic PersistentMemory variant.

## Task 4: Compute Phase I Statistics And Gate I

**Files:** create `scripts/reviewer_closure_analysis.py`,
`tests/test_reviewer_closure_analysis.py`,
`artifacts/reviewer_closure/TRIVIAL_TRACKER_AUDIT.md` and Phase I CSV/JSON outputs.

1. Test paired six-cluster bootstrap, order robustness, LOSO and gate derivation.
2. Generate T2-T5 task/deployment tables and `TRACKER_REJECTED` or
   `TRACKER_EXPLAINS_IDENTITY`.
3. If explained, additionally generate `TRIVIAL_TRACKER_CHALLENGE_REPORT.md`.

## Task 5: Audit T3 Training Compatibility

**Files:** create `scripts/reviewer_closure_training.py`,
`tests/test_reviewer_closure_training.py`,
`artifacts/reviewer_closure/REScene_HORIZON_TRAINING_AUDIT.md`.

1. Bind official/local data, model, optimizer, scheduler, losses and checkpoint
   initialization; classify every recipe field as known/unknown/reconstructed/assumed.
2. Select Level 1 only if exact equivalence is evidenced; otherwise select the
   named T2-to-T3 horizon-adaptation fallback.
3. Record the selection before training and forbid later recipe changes.

## Task 6: Smoke And Train One T3 Model

**Files:** create `configs/reviewer_closure/rescene_t3_adapted.yaml`; extend the
training tests and create `artifacts/reviewer_closure/t3_training_manifest.json`.

1. Test loader stages exactly `{0,1,2}`, mapping integrity, finite one-step
   forward/backward, checkpoint reload, and unchanged T2 configuration.
2. Run the formal training once; record batch/accumulation, optimizer updates,
   scan exposures, wall/GPU hours, seed, source/config/checkpoint hashes.
3. Fail closed on non-finite loss, provenance drift or incomplete checkpoint.

## Task 7: Evaluate T3-Adapted ReScene And Gate II

**Files:** extend reviewer-closure runner/analysis tests; generate
`adapted_rescene_results.csv`, `adapted_rescene_compute.csv`,
`adapted_rescene_statistics.json`.

1. Evaluate the one adapted checkpoint at T2-T5 on exact Protocol-B prefixes.
2. Apply the Phase I strongest tracker to its sidecars without tracker retuning.
3. Derive `HORIZON_ROBUST`, `FULL_HISTORY_DOMINANT`, or
   `ACCURACY_ADVANTAGE_BUT_COSTLY`; generate the long-horizon challenge report
   only for the dominant outcome.

## Task 8: Implement IoU Sweep And Coverage Ceiling

**Files:** create `scripts/reviewer_closure_decomposition.py`,
`tests/test_reviewer_closure_decomposition.py`.

1. Test synthetic high-IoU failure, no-candidate and wrong-class cases.
2. Compute temporal AP/recall for thresholds 0.25:0.05:0.90.
3. Compute 0.25/0.50/0.75 local-pair and full-history coverage categories.

## Task 9: Implement GT Association Oracle And Failure Decomposition

**Files:** extend decomposition module/tests; generate Oracle and failure outputs.

1. Test perfect-mask fragmented IDs recover under Oracle while observation misses
   cannot recover.
2. Replace only identity assignment; bind unchanged mask/class/feature populations.
3. Recompute operational failure categories including explicit unknown/unresolved.
4. Derive `ASSOCIATION_CEILING` or `PERCEPTION_CEILING`.

## Task 10: Conditionally Audit LivingScenes

**Files:** only if triggered, create `scripts/reviewer_closure_livingscenes.py`,
its tests and the conditional audit/result artifacts.

1. Pin official source, run official smoke, then audit supported categories,
   object-point coverage, coordinates and sequential matcher variants.
2. Use predicted masks only and evaluate supported categories only.
3. Publish an external baseline; do not integrate it into Persist4D.

## Task 11: Generate Figures And Final Evidence

**Files:** create `scripts/reviewer_closure_figures.py`,
`scripts/build_reviewer_closure_artifacts.py`, corresponding tests, required SVGs,
`METRIC_AGGREGATION_NOTE.md`, `REVIEWER_CLOSURE_SUMMARY.md`, and
`FINAL_METHOD_LOCK_REPORT.md`.

1. Render tracker challenge, horizon adaptation, IoU sweep, Oracle headroom and
   four-panel failure-decomposition figures from validated CSV/JSON inputs.
2. Enforce pooled benchmark AP versus paired cluster-macro statistic terminology.
3. Build exactly the 12 required final-report sections and one allowed final class.

## Task 12: Final Integrity, Verification And Publication

1. Run all new tests, relevant frozen regressions, full pytest, compile checks,
   artifact verification, `git diff --check`, hash checks and manual full-diff review.
2. Confirm `git rev-parse HEAD:artifacts/system_comparison` remains
   `398fe87e1d40d67e61399fd893f02dc5f5f6b7ad`.
3. Commit lightweight implementation/evidence, push the reviewer-closure branch,
   and stop without adding an untriggered method component.
