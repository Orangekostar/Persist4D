# Native code bindings

Bindings below refer to the fixed base source, unchanged by this experiment. Cache identities additionally hash the actual producer, descriptor, dataset, postprocess, backbone and installed matcher/evaluator files and lock numeric/library settings.

| Source and lines | Actual behavior and insertion boundary |
|---|---|
| `models/rescene.py:474–710` (`forward`) | Final raw class/mask logits, segment features and normalized query features are exported from one parent execution. New heads are downstream; parent parameters are frozen/eval. |
| `models/rescene.py:711–737` (`aggregate_features`) | Mask features aggregate through original unsupervised point2segment. D is read from actual segment tensors, not padded/truncated. |
| `models/rescene.py:738–751` (`mask_module`) | Decoder normalization precedes class and mask outputs; final query export uses the same normalization. |
| `scripts/rescene_task_postprocess.py:62–137` (`OfficialTaskSoftEvidence`) | Retained candidate columns bind segment logits/features, normalized queries, class probabilities and low/inverse/full mappings. Same raw query may supply multiple classes. |
| `scripts/rescene_task_postprocess.py:212–237` (`materialize_segment_logits`) | Logits→low mapping→strict positive→inverse→original full-eval majority partition. Geometry update0 uses this real path, not a bool-mask shortcut. |
| `scripts/rescene_task_postprocess.py:240` (`extract_official_task_prediction`) | Single top-k/filter call with query/class lineage. Soft evidence reconstruction must exactly equal native masks. New heads preserve columns, classes and keys. |
| `trainer/trainer.py:1566–1580` (`_get_predictions`) | Softmax is applied to selected decoder class logits; auxiliary dictionaries can be replaced. Native equality calls receive independent dictionaries sharing only read-only raw tensors, preventing double softmax. |
| `trainer/trainer.py:1624–1668` (`_get_mask_and_scores`) | Original class-query top-k and mask probability scores; retained Nc need not equal query count. |
| `trainer/trainer.py:1671` (`_filter_and_sort_predictions`) | Original native candidate filter is run once before any head. No post-head filtering or second top-k. |
| `trainer/trainer.py:1392–1401` (`_get_full_res_mask`) | Original scatter-mean full partition with strict >0.5 majority. Used by the head deployment adapter. |
| `scripts/system_comparison_inference.py:1204` (`FullHistoryPredictionProducer`) | Reference pattern for load_scan_indices, original collator, seed scope, model forward and official postprocess. New native adapter directly supports H1 and returns compact soft records. |
| `scripts/system_comparison_inference.py:388–571` (`postprocess_full_history_output`) | Original task output used for first real CAL pair equality. Its identity branch is not a new head input. |
| `scripts/perception_gain_native_evaluation.py:148` (`run_native_checkpoint_evaluation`) | Reuse its fixed CAL/SEL population construction and native config/weight binding. Do not inherit its H2–H5 restriction or V2 campaign selection. |
| `scripts/perception_gain_local_evaluation.py:106–156,180` (`audit_local_population`, `run_local_t2_evaluation`) | Read-only exposure binding only: actual LOCAL154 sequences/46 references. No local scoring in this round. |
| `datasets/semseg.py:417–473,475–664` (`load_scan_indices`, `_load_scan_sequence`) | Explicit scan index list controls file reads. Each supplied scan's temporal coordinate is its position; H1 is time0. Native adapter records actual np.load paths and rejects substitution. |
| `datasets/auto_collate.py:10–45`, `datasets/pointcept_utils.py:7–79,82–300` | Actual Pointcept collator, per-stage voxel transforms, original labels and inverse maps. Single-scan inputs are collated independently. Preserve empty targets without injecting another scan. |
| `models/perception_gain.py:325–374` (`derive_segment_stage_ids`) | Reject segments spanning scans; do not round averaged time coordinates. Full vertices map to training segments through low_point2segment[voxel_inverse]. |
| `scripts/p6a_metrics.py:910–1056,1059–1074` | OfficialMetricAccumulator uses LegacyAPEvaluator for H1 spatial AP and TemporalEvaluator for H2 pooled AP. Threshold helper binds0.50 through0.90; Q3 labels use strict >. |
| Installed `/home/ww/paper5/third_party/stmetrics/stmetrics/instances/matcher.py` (`_assign_instances_for_scan`, `_assign_ambiguous`) | Pairwise stage intersections/overlaps and prediction validity precede score-dependent ambiguity resolution. Label adapter uses only the pre-resolution method, deterministic column UUIDs and excludes explicit ambiguous relations. |
| Installed `stmetrics/instances/evaluator.py:75–98,285–290` | Use actual `_valid_gt`, `_proportion_ignore`, `_select_overlap`. Absent-GT/empty-pred stages have infinity overlap in matcher and temporal minimum across observed stages. No division for zero prediction support. |

## Data and cost bindings

`INPUT_MANIFEST.json` binds actual paths and SHA256 for V2 `DATA_ROLES.json`, `INPUT_MANIFEST.json`, `data/STAGING_MANIFEST.json`, live `budget/LEDGER.jsonl`, actual LOCAL exposure audit and RIO metadata. Original staging is shared through ignored `data` symlink, without copying raw assets. Its inherited null per-file hashes are disclosed; no byte-immutability claim.

TRAIN selection uses metadata-only legal T2 records: supervised-content filtering is disabled for eligibility, and original sample0 substitution is forbidden. CAL/SEL preserve their original V2 master-prefix population. The parent dataset's native `train` mode, augmentations and fixed seed are retained rather than silently changing the baseline. H1 has its own sample, coordinate normalization, neighborhoods and parent execution.

New insertion files: `models/short_module_heads.py` (prediction-only descriptors/heads/deployment adapter), `scripts/short_module_data.py` (population and GT supervision), `scripts/short_module_native.py` (parent export and compact cache), `scripts/short_module_screen.py` (fixed experiment execution). Predictions and targets are serialized separately. Parent dimensions and actual smoke evidence are in `RUN_CONFIG.json` and external `EXPORT_INDEX.json`; full evaluation/training completion is not implied by these bindings.
