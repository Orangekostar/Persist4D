# Perception Gain V1 Code Bindings

| Existing path | Symbol | Current role | Perception Gain change / experiment |
|---|---|---|---|
| `models/criterion.py` | `SetCriterion.loss_masks`, `SetCriterion.forward` | matched mask BCE/Dice and aux loss | opt-in stage aggregation for S-BAL/S-WORST |
| `models/matcher.py` | `HungarianMatcher.forward` | fixed one-to-one matching | unchanged; regression checked |
| `trainer/trainer.py` | `_configured_objective_loss`, `InstanceSegmentation` | raw-sum objective/model setup | narrow perception trainer reuse |
| `conf/config_rescene4d_concerto_rootcause.yaml` | root R1 config | R1 data/model semantics | resolved values frozen in RUN_CONFIG |
| `models/rescene.py` | `initialize_queries`, `aggregate_features`, `forward`, `attn_mask` | query/decoder path | Q-SEM and A-OPEN opt-in paths |
| `datasets/task_memory_episode.py` | `StageMeta`, `TaskMemoryEpisodeCollator` | causal stage/vertex mapping | reused by pair/scorer/refiner data |
| `scripts/crosswindow_cache.py` | `resolve_assets`, `align_mask`, `build_canonical_frame` | asset and canonical alignment | reused without old-output overwrite |
| `scripts/diagnose_crosswindow_failures.py` | `_published_match_by_gt`, `diagnose_candidate_coverage` | E1 diagnostic | tau-specific official trace |
| `scripts/rescene_task_postprocess.py` | `extract_official_task_prediction` | official bool prediction | optional real soft sidecar |
| `scripts/task_memory_output.py` | lag-one publishers | D0 identity/output | fixed D0 path; optional refiner insertion |
| `scripts/replay_crosswindow_association.py` | CrossWindow replay | D0 association | unchanged association contract |
| `scripts/run_task_memory_controls.py` | R1 prediction producer | cached/live D0 source | checkpoint/config parameterization |
| `scripts/system_comparison_inference.py` | native FH inference | full-prefix baseline | live checkpoint runner reuse |
| `scripts/p6a_metrics.py` | `OfficialMetricAccumulator` | official local/temporal metrics | optional match trace only |
| `scripts/system_comparison_metrics.py` | `CausalTaskAccumulator` | causal aggregate metrics | reused for all formal scores |
| `scripts/train_task_memory.py` | checkpoint/asset helpers | prior fixed training runner | helpers reused, variants unchanged |
| `trainer/task_memory_trainer.py` | episode schedule/resume helpers | prior state trainer | metadata/scheduler patterns reused |
| `scripts/profile_task_memory.py` | live timing primitives | measured resource path | adapted for locked final methods |

Line locations are recorded against fixed parent `6ef77620aa20926311eff3124a794a6ca2e32727`; final changed line locations are refreshed during report generation.
