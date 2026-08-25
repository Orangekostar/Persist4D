# Reviewer Closure V3 Code Audit

Audit date: 2026-08-25

## Task Candidate And Trajectory Paths

| File | Role and relevant API | Inputs -> outputs | V3 status | Scientific dependency |
|---|---|---|---|---|
| `scripts/rescene_task_postprocess.py` | Canonical ReScene task conversion: `OfficialTaskPrediction`, `extract_official_task_prediction()` | Model decoder output, targets, inverse maps -> official full-resolution masks/classes/scores plus query/class lineage | Semantics frozen; reuse only | Local candidate invariance and every task-channel comparison |
| `scripts/system_comparison_v2_inference.py` | Causal candidate history: `CandidateTrajectoryKey`, `V2TrajectorySnapshot`, `OfficialCandidateTrajectoryAccumulator` | Official sidecars plus tracker steps -> committed prefix masks/classes/scores | May add only `latest`/`max`; mean and keys frozen | Score sensitivity and trajectory t-mAP |
| `scripts/system_comparison_v2_analysis.py` | V2 cache loading, causal pairs, stmetrics aggregation: `build_v2_causal_pair()`, `load_v2_sequences()`, `run_v2_analysis()` | Frozen cache/sidecars/V1 keyed rows -> V2 task CSVs | Frozen historical evidence; do not overwrite | Mean regression and known copied-identity limitation |
| `scripts/system_comparison_v2_attribution.py` | Frozen F0/L0/L1/P0/P1 task-path decomposition | Same cache/sidecars under controlled conversion/linkage variants -> attribution artifacts | Frozen; no V3 writes | Establishes repaired official-candidate path |

`OfficialCandidateTrajectoryAccumulator` keys persistent candidates by
`(track_id, official_class_id)` and unmatched candidates by stage/query/class.
Masks are committed causally and cannot be rewritten by future observations.
The current implementation accepts arithmetic `mean` only; V3 may change only
the reducer dispatch.

`system_comparison_v2_analysis.py::_identity_from_old()` copies keyed V1
identity fields after task regression. This is documented historical behavior,
not an admissible V3 identity source.

## Metric And Identity Paths

| File | Role and relevant API | Inputs -> outputs | V3 status | Scientific dependency |
|---|---|---|---|---|
| `scripts/system_comparison_metrics.py` | Causal contracts and task/identity aggregation: `CausalPrefixPair`, `validate_causal_prefix_pair()`, `CausalTaskAccumulator`, `match_identity_update()`, `compute_deployment_identity_metrics()` | Prefix predictions/targets and per-stage issued IDs -> task blocks and deployment identity diagnostics | Reuse; no metric replacement | Causal correctness, switch/recovery/fragmentation metrics |
| `scripts/p6a_metrics.py` | Official evaluator adapter: `OfficialMetricAccumulator` and `IdentityAccumulator` | Predictions/targets -> `LegacyAPEvaluator` raw-local or `TemporalEvaluator` temporal metrics | Frozen official backend | AP/t-mAP/t-REC semantics and sufficient state |
| `scripts/evaluate_persist4d_p6a.py` | Raw cache production/replay, tracker registration, task metrics, and association events | Protocol manifest, dataset, checkpoint/cache, tracker factories -> frozen observations, predictions, events, metric blocks | Reuse public helpers; V3 wrappers may call them | B2/B3/B4 fresh identity, local channel, cache parity |

Registered factories at `build_tracker_factories()` remain B0, B1 feature,
B2 feature+class, B3 EMA, and B4 persistent state. V3 uses the existing B2/B3/B4
factories. `build_association_events()` receives tracker steps and post-inference
targets; `compute_deployment_identity_metrics()` consumes its identity updates.
GT does not enter tracker construction or step inputs.

## Protocol And Data Paths

| File/artifact | Role and relevant API | Inputs -> outputs | V3 status | Scientific dependency |
|---|---|---|---|---|
| `scripts/p6a_protocol.py` | Exact Protocol-B construction: `build_protocol_b()`, `build_protocol_b_manifest()`, validator | T5 validation DB and metadata -> 43 masters, 6 clusters, 3 orders, T2-T5 prefixes | Frozen and reused | Canonical bridge identities, order and horizon pairing |
| `artifacts/P6A/protocol_b_manifest.json` | Portable registered protocol binding | Source/config hashes and resolved scan indices -> exact master/order/prefix inventory | Frozen | All bridge/cache pair keys |
| `scripts/audit_tmap_protocol_shift.py` | Existing sliding-T2 inventory and reported/local metric boundary | Protocol-B manifest plus independent T2 DB/V2 rows -> 14/43 inventory audit | Frozen | Shows why inventory is not a construction limit |
| `datasets/semseg.py` | Loads explicit scan sequences and change targets | Ordered scan files and change file -> concatenated points, temporal stages, labels/instances/change labels | Runtime semantics frozen | Exact T2 target construction and parity |

The source T5 change target has one row per T5 point and four transition
columns. The loader uses column zero for a 2D change file. A read-only audit of
the 14 existing canonical T2 overlaps found that the first two T5 scan point
rows and column zero exactly match the independent T2 target in 14/14 cases.
PB0 must reproduce this through checked-in tests before evaluating all 43.

## P2 Reproduction Paths

| File/artifact | Role | Inputs -> outputs | V3 status | Scientific dependency |
|---|---|---|---|---|
| `trainer/trainer.py` | Official evaluation chain `_get_predictions`, `_get_batch_masks`, `_get_mask_and_scores`, `_get_full_res_mask`, `_filter_and_sort_predictions` and stmetrics update | Validation batches/model output -> official predictions and metrics | Frozen runtime; audit/reuse | Bridge full-154/exact-43 runtime parity |
| `main_instance_segmentation.py` | Hydra entry, seed binding, checkpoint loading, trainer test | Config/checkpoint -> model evaluation | Frozen runtime; audit/reuse | Three-seed bridge evaluation |
| `artifacts/P2_G2_REPRODUCTION_REPORT.md` | Frozen full-154 result and G2 decision | Completed P2 run -> 27.939% t-mAP, G2 RED | Frozen | E1 and population reference |
| `artifacts/P2/config_audit.md` and `official_vs_repro_config_diff.json` | Official/local configuration and source deviations | Official revision plus local runtime -> audited differences | Frozen | Prevents calling E1 official reproduction |
| `scripts/p2r_pilots.py` and `artifacts/P2R/*` | Controlled 32-step post-P2 pilot | Four preregistered short paths -> no authorized full run | Frozen | Limits conclusions and prohibits new 450-epoch run |

P2 used seed 45 but its validation collator has randomized train-mode voxel
sampling. V3 therefore preregisters seeds 45, 46, and 47 for both full-154 and
exact-43 T2 through one runtime; it cannot reuse 27.939 as though it were a
three-seed bridge measurement.

## Tests Inspected

The nine required existing test files cover official post-processing lineage,
V2 causal trajectories, V2 analysis/attribution/parity, Protocol-B inventory and
prefix determinism, stmetrics raw/temporal behavior, causal prefix validation,
Hungarian matching, and deployment identity denominators. The exact pre-change
selection passed 59 tests. V3 adds six dedicated test files before each new
production path.

## Audit Decision

- Frozen official-candidate and stmetrics paths are sufficient; no custom AP or
  new local conversion is needed.
- Existing Protocol-B and loader semantics support a testable exact-T2 bridge;
  construction remains gated by 14/14 parity.
- Existing registered tracker and identity APIs support fresh B2/B3/B4 metrics;
  copied V1 identity fields are excluded from final V3 artifacts.
- Gate for implementation planning: **PASS**.
