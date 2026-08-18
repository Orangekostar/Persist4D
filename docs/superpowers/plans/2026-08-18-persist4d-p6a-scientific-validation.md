# Persist4D P6-A Scientific Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Every production change follows RED-GREEN-REFACTOR and every scientific claim is derived from frozen machine artifacts.

**Goal:** Determine, under a common-prefix causal protocol, whether Persist4D's persistent association improves identity consistency, reactivation, and strict-online task quality over strong simple temporal baselines without changing local ReScene perception.

**Architecture:** Freeze P5 code, results, checkpoint, and local predictions. Build Protocol B from the 43 validation T5 master sequences and derive T2-T5 as exact prefixes. Materialize each local ReScene observation once and fan it out immutably to B0/B0-sanity/B1/B2/B3/B4 and post-hoc Oracle trackers. Evaluate raw-local perception separately from prefix-endpoint strict-online tracks, reconstruct identity events with global Hungarian GT matching, and derive every CSV, figure, confidence interval, and gate from one machine-readable root artifact.

**Tech Stack:** Python 3.10, PyTorch 2.6, Hydra/OmegaConf, SciPy linear assignment, stmetrics, pandas/pyarrow, NumPy cluster bootstrap, matplotlib, pytest, frozen ReScene/Concerto checkpoints.

---

## Frozen Inputs And Preregistered Gates

| Contract | Frozen value |
| --- | --- |
| P5 evaluation source commit | `92bab01e93bacbc939606ec7c7f58d3f9b334fe6` |
| P5 artifact commit | `1380c4b9f37bec7933126ccc9bd70067de166f6f` |
| ReScene checkpoint SHA256 | `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e` |
| T5 sequence DB SHA256 | `252363f76524bb7eeff9f65b303aadda67dcd2646477daae1ac90f7f53398290` |
| Master sequences / clusters | 43 / 6 reference scenes |
| Horizons | T2, T3, T4, T5 exact prefixes of each T5 master |
| Order stress tests | canonical, reverse, SHA256-derived permutation with seed 45 |
| B1 feature threshold | 0.5 |
| B2/B3 class weight | 0.25 on foreground-renormalized probabilities |
| B3 EMA update rate | 0.2, active previous-stage tracks only |
| B4 | Frozen P5: K=100, threshold=0.5, class weight=0.25, update=0.2 |
| Diagnostic GT assignment | class-compatible global Hungarian, IoU >= 0.5 |
| Bootstrap | 10,000 paired reference-scene cluster replicates, seed 45 |
| G6A-1 | T4 and T5 IDSW-rate relative reduction >=20%; paired delta CI high <=0 |
| G6A-2 | T3-T5 reactivation accuracy >=0.70, recall >=0.25, and better than strongest simple baseline |
| G6A-3 | shared raw prediction fingerprints exactly equal; metric absolute tolerance 1e-12 |
| G6A-4 | T2 online t-mAP and t-REC drops each <=0.05 absolute; T4 or T5 has positive online t-mAP or t-REC delta |
| G6A-5 | uniquely categorized primary failure share >=0.90 |

The six reference scenes, not the 43 overlapping cyclic windows, are the statistical units. Metadata order is an order stress-test convention, not real chronology.

## File Map

| File | Responsibility |
| --- | --- |
| `conf/p6a/default.yaml` | Frozen protocol, baseline, metric, bootstrap, and gate settings |
| `scripts/p6a_protocol.py` | T5 master loading, explicit scan resolution, exact prefixes, order variants, manifest validation |
| `scripts/p6a_association.py` | Immutable observation contract, identity namespace, B0-B3 trackers, B4 adapter, Oracle |
| `scripts/p6a_metrics.py` | Dynamic accumulator, raw-local, endpoint-online, offline diagnostic, Hungarian identity events |
| `scripts/p6a_analysis.py` | Normalized IDSW, reactivation, F1-F7, capacity, efficiency, paired cluster bootstrap, gates |
| `scripts/p6a_artifacts.py` | Exact schemas, privacy/path guards, CSV/Parquet/Markdown/figure rendering |
| `scripts/evaluate_persist4d_p6a.py` | Frozen prediction cache and end-to-end P6-A runner |
| `tests/test_p6a_protocol.py` | Protocol B mapping, prefixes, order variants, input provenance |
| `tests/test_p6a_association.py` | B0-B4, Oracle quarantine, Q/K decoupling, deterministic assignment |
| `tests/test_p6a_metrics.py` | Raw-local invariance, endpoint causality, Hungarian and identity/reactivation definitions |
| `tests/test_p6a_analysis.py` | Events, F1-F7, capacity, efficiency, paired statistics, gates |
| `tests/test_p6a_artifacts.py` | Exact artifact schema, hashes, privacy, reconstruction and renderer consistency |
| `tests/test_p6a_gpu_gate.py` | Opt-in real checkpoint/cache/evaluation artifact verification |

## Task 1: Freeze P5 And Baseline P6-A

**Files:**
- Create: `docs/Persist4D P6-A - P6-B 监督执行提示词.md`
- Create: this plan

- [x] Create `research/persist4d-p6a` in an isolated worktree from P5 artifact commit.
- [x] Copy the supervision prompt byte-for-byte and commit it.
- [x] Make data and checkpoint available without changing portable repository references.
- [ ] Run the complete CPU baseline and record pass/skip counts.
- [ ] Hash P5 JSON/Markdown before and after P6-A; reject any change.
- [ ] Commit this plan before production implementation.

## Task 2: Implement Exact Common-Prefix Protocol B

**Files:**
- Create: `conf/p6a/default.yaml`
- Create: `scripts/p6a_protocol.py`
- Create: `tests/test_p6a_protocol.py`

- [ ] Write RED tests for the 43 stable T5 validation names, six UUID clusters, scan IDs and validation indices.
- [ ] Test that every T2/T3/T4 prefix is derived from one full T5 order in both names and resolved indices.
- [ ] Test canonical, reverse, and seeded order variants are deterministic and derive all horizons from one order.
- [ ] Test duplicate/missing scans, wrong split, unsupervised prefixes, substitution policy, and positional-index assumptions fail closed.
- [ ] Implement stable YAML/metadata loading and explicit `load_scan_indices()` inputs.
- [ ] Render and validate `protocol_b_manifest.json` with portable refs and source hashes.

## Task 3: Decouple Identity Namespace And Implement Baselines

**Files:**
- Create: `scripts/p6a_association.py`
- Create: `tests/test_p6a_association.py`

- [ ] RED-test `Q != K`, unbounded evaluator identities, per-method/per-sequence namespace reset, and immutable fan-out observations.
- [ ] Implement B0 stage-unique and explicitly labeled B0-sanity local-query IDs.
- [ ] Implement B1 previous-stage feature-only threshold-aware Hungarian matching.
- [ ] Implement B2 previous-stage feature plus foreground-renormalized class Hungarian matching.
- [ ] Implement B3 previous-stage active-only EMA matching with no dormant lifecycle or reactivation.
- [ ] Adapt frozen P5 B4 results without changing its association, memory, or output semantics.
- [ ] Implement post-hoc Oracle with no GT access before inference or tracker updates.
- [ ] Test deterministic ties, one-to-one matches, gap births, foreground normalization, EMA updates, legacy parity, and GT quarantine.

## Task 4: Implement Raw-Local And Strict-Online Metrics

**Files:**
- Create: `scripts/p6a_metrics.py`
- Create: `tests/test_p6a_metrics.py`

- [ ] RED-test shared raw prediction fingerprints and AP/AP50/AP25/REC equality across all association methods.
- [ ] Implement raw-local evaluation using only the newest stage observation and a T1 target.
- [ ] Implement a dynamic identity accumulator that does not assume identity ID `< K`.
- [ ] Snapshot each tracker at each prefix endpoint and build tracks using only observations and state available through that endpoint.
- [ ] Report online t-mAP/t-mAP50/t-mAP25/t-REC/t-REC50/t-REC25 separately from offline reconstruction.
- [ ] RED-test that future observations cannot alter an earlier prefix endpoint result.
- [ ] Replace diagnostic greedy matching with cardinality-first, IoU-second class-compatible global Hungarian matching with deterministic ties.
- [ ] Preserve the greedy implementation only as an explicitly labeled regression diagnostic.

## Task 5: Implement Identity, Reactivation, Event, And Error Audits

**Files:**
- Create: `scripts/p6a_analysis.py`
- Extend: `tests/test_p6a_metrics.py`
- Create: `tests/test_p6a_analysis.py`

- [ ] Define transition opportunities, ID switches, normalized IDSW rate, fragmentation and merge counts per sequence.
- [ ] Define gap opportunities, attempts, correct/wrong/no-attempt events, accuracy, precision, recall, and coverage.
- [ ] Emit one typed association-event row per prediction decision plus GT-only miss rows; use nulls, never sentinel IDs or NaN.
- [ ] RED-test event-table reconstruction of every aggregate metric.
- [ ] Implement one exclusive primary F1-F7 category per failure and an explicit unclassified bucket.
- [ ] Test all F1-F7 rules, exclusivity, and the >=90% explainability calculation.
- [ ] Audit birth/occupied/active/dormant/rejected/peak state counts per stage and horizon.
- [ ] Separate bounded persistent state bytes from dynamic offline evaluator bookkeeping.

## Task 6: Implement Efficiency And Paired Statistical Analysis

**Files:**
- Extend: `scripts/p6a_analysis.py`
- Extend: `tests/test_p6a_analysis.py`

- [ ] Measure bootstrap/new-visit latency, association/update overhead, peak working memory, persistent state bytes, and full-history comparison.
- [ ] RED-test that setup/bootstrap rows cannot be mixed with per-new-visit rows.
- [ ] Implement paired cluster bootstrap over six reference scenes with exact master/prefix/prediction-digest pairing.
- [ ] Reject duplicate, missing, or cross-cache pairs and make seed-45 output byte-deterministic.
- [ ] Implement G6A-1 through G6A-5 directly from machine aggregates and preregistered thresholds.

## Task 7: Build The Frozen Prediction Cache And End-To-End Runner

**Files:**
- Create: `scripts/evaluate_persist4d_p6a.py`
- Create: `tests/test_p6a_gpu_gate.py`

- [ ] RED-test source-tree, checkpoint, config, dataset and cache provenance contracts.
- [ ] Load the canonical checkpoint strictly and verify legacy/query output parity on a real T2 sample.
- [ ] For each `(master, order, stage)`, materialize the local input once, run ReScene once, clone/freeze the observation, and persist a content-addressed cache.
- [ ] Fan each cached observation to every method without rerunning collate or ReScene.
- [ ] Reject stale, partial, non-finite, path-private, dirty-source, or mismatched cache entries.
- [ ] Support resumable atomic cache generation and evaluation without publishing partial pass artifacts.
- [ ] Add opt-in GPU gates for real cache coverage and final artifact verification.

## Task 8: Render And Validate P6-A Artifacts

**Files:**
- Create: `scripts/p6a_artifacts.py`
- Create: `tests/test_p6a_artifacts.py`
- Generate: `artifacts/P6A/*`

- [ ] Define one exact-schema `p6a_eval.json` as the root of every derivative artifact.
- [ ] Render the required protocol, baseline, strict-online, raw-local, per-sequence, event, error, reactivation, capacity, efficiency, and statistical artifacts.
- [ ] Render Tables A-C and Figures A-E with input/script/output hashes.
- [ ] Validate finite values, row counts, primary keys, null semantics, metric reconstruction, paired coverage, privacy, portable refs, and renderer byte identity.
- [ ] Generate the 11-section `P6A_GO_NOGO_REPORT.md` with supported and unsupported claims and exactly one quantitative GO/NO-GO decision.

## Task 9: Run P6-A And Apply The Gate

- [ ] Run complete CPU tests and static/privacy checks on a clean committed source tree.
- [ ] Run real GPU prediction caching for all 43 masters, three orders, and five local observations per master.
- [ ] Run B0/B0-sanity/B1/B2/B3/B4/Oracle on exact T2-T5 prefixes.
- [ ] Run cluster statistics, error decomposition, capacity, and efficiency audits.
- [ ] Rebuild all artifacts from the root JSON and verify byte/hash equality.
- [ ] Re-run opt-in GPU artifact gates and the complete CPU suite.
- [ ] Request independent code/scientific review, fix every Critical/Important issue with TDD, and re-run verification.
- [ ] Commit P6-A source separately from final evidence.

## Task 10: Conditional P6-B

Do not create P6-B code or artifacts unless every mandatory P6-A gate returns GO.

If GO:

- [ ] Write a separate P6-B implementation plan based only on measured P6-A failure categories.
- [ ] Implement threshold-aware assignment, dormant-aware reactivation, foreground class compatibility, confidence-gated consolidation, and birth gating behind new config.
- [ ] Tune only on the declared validation sweep, freeze the selected config, then run the final common-prefix evaluation once.
- [ ] Generate and verify the required P6-B artifacts and GO/NO-GO report.

If NO-GO:

- [ ] Stop after P6-A, preserve all negative evidence, state exactly which claims failed, and recommend the smallest evidence-driven next method change. Do not start P7/P8.

## Final Verification

```bash
cd /home/ww/paper5/.worktrees/persist4d-p6a
env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  PYTHONPATH="$PWD:$PWD/third_party/concerto:$PWD/third_party/sonata:$PWD/third_party/detectron2:$PWD/third_party/stmetrics" \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q

env PYTHONNOUSERSITE=1 \
  /home/ww/miniconda3/envs/persist4d/bin/python scripts/check_path_privacy.py artifacts/P6A

git diff --check
git status --short
```

Completion requires fresh command output, exact artifact hashes, a clean worktree, and no unresolved Critical/Important review findings.
