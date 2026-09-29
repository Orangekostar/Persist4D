# Actual code bindings

Base: `465f37a46972e81584c1563bba57e1c05c911c1a`.

| Boundary | Existing binding | Targeted use |
|---|---|---|
| Native parent | `scripts.short_module_native.NativeSession.produce` | Frozen eval R1, actual H1/H2 |
| Descriptors | `models.short_module_heads.describe_candidates`, `quality_inputs` | Unchanged field order and detach |
| Q supervision | `scripts.short_module_data.quality_labels` | `temporal`, binary `valid`; threshold-validity-v2 |
| M eligibility | `quality.geometry_valid` | Frozen min-threshold pool; never substitute Q valid |
| Assignment | `scripts.short_module_data.geometry_assignment` | Frozen one-to-one class-compatible concat IoU >=.10 |
| Segment targets | `scripts.short_module_data.low_segment_targets` | Full known vertices projected to LOW segments |
| Shared mask head | `models.short_module_heads.MaskHead(per_stage=False)` | Same F/sqrt(D), bias, 2*tanh bound |
| Legacy M loss | `scripts.short_module_data.geometry_loss('M1')` | Real value/parameter-gradient comparison with M-C |
| Sampling | `scripts.short_module_training.build_sample_plan` | Same reference/input/candidate draws, stage seeds |
| Materialization | `scripts.rescene_task_postprocess.materialize_segment_logits` | Native inverse mapping and full partition majority |
| Official metrics | `scripts.p6a_metrics.OfficialMetricAccumulator` | raw_local H1 and strict_online H2 |
| Metric freshness | `scripts.short_module_eval_identity.evaluation_identity` | Full stmetrics source closure, plus new adapter/head/eval sources |
| Targeted prediction | `scripts.qp_mn_adapter.apply_module` | Raw Q scores; legacy M materializer; no refilter |
| Targeted losses | `models.qp_mn_heads.quality_loss`, `shape_terms`, `mask_loss` | Whole-batch Q mean, actual-positive shape denominator, equal record regularizer |

Actual tensors: F=128, q=128, classes=18, probability columns=19. Verified by
loading a real TRAIN prediction, not inferred solely from config.

`BASE_BINDING.json` binds parent assets and repaired reports; `REPAIRED_CONTRACT.json`
binds label/prediction identities. `INPUT_MANIFEST.json` records actual input files
and content hashes. No evaluation/training path writes historical ARTIFACTS globals.

Known inherited failure: repo_reference symlink path serialization expects a
repo-relative directory but data resolves externally. It affects the path label
in that test, not native tensors, labels, metric math or head deployment. The
repaired TEST_NOTES records reproduction at the untouched historical baseline.

Real contract probes passed on this checkout: M-C versus old M1 loss and all
parameter gradients; corrected Q2 versus Q2-F; TRAIN score/column invariance;
official negative/>1 score acceptance and affine ranking invariance. The
partial-ignore synthetic regression separately preserves distinct Q/M eligibility.
