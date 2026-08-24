# E0 Evidence Audit

## Repository preflight

- Branch: `research/persist4d-tmap-root-cause-v2`
- Start commit: `487080cf31266f1572257e2aca36767e074b68b6`
- Start commit subject: `results: close final evidence prompt`
- Initial worktree state: clean
- Checkpoint: `checkpoints/rescene4d_concerto_t2_repro.ckpt`
- Checkpoint SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`
- Checkpoint state: epoch `404`, global step `26730`

The unqualified `pytest` executable is invalid on this host: it uses system
Python 3.12 and a partial user-level Torch installation. The frozen project
runner is `conda run -n persist4d python -m pytest`, matching
`artifacts/final_evidence/BASELINE_TEST_STATUS.md`.

## Frozen evidence hashes

| Evidence | SHA256 |
| --- | --- |
| `artifacts/P2_G2_REPRODUCTION_REPORT.md` | `d891fb7fd53306d8ab65db81b9bb85f08664a9689de850ac7836143b238816bc` |
| `artifacts/P2/config_audit.md` | `1d76568fb060d959a7ba3a5d660d2ca45ece19108615e6df2249e09912f17b08` |
| `artifacts/P6A/P6A_GO_NOGO_REPORT.md` | `a9c95655c25d0ca1a902d901dbefec013e136ff7e1f4b6f0b023ef90157de5c6` |
| `artifacts/system_comparison/aggregate_results.csv` | `c70000d35e9d661c9929d25a465d287f778fc594b49fb596aa36d956e68f73a0` |
| `artifacts/system_comparison/per_order_results.csv` | `6d950fa97da08892f5e86c08a6ff24e7710ff5090e46a664e4cb8394d8a64705` |
| `conf/p6a/default.yaml` | `dd7a9fccc098fc7e5faecc059d6202ca7f8a762aaee3e7182738a2d7f800aac2` |
| `configs/system_comparison/persist4d_incumbent.yaml` | `aadca1c1608bae4ad4e812a6261536d9e386a85c460bbd85f4c5d215c9a68717` |

No frozen P2, P6A, or System Comparison V1 artifact was modified.

## Required fact audit

### 1. P2 checkpoint and reproduction gap: verified

`artifacts/P2_G2_REPRODUCTION_REPORT.md` records the official-like 154-sequence
single-GPU result and the paper references:

| Metric | Paper ReScene4D-C | P2 reproduction |
| --- | ---: | ---: |
| t-mAP | 34.800 | 27.939 |
| t-mAP50 | 52.500 | 46.565 |
| t-mAP25 | 66.800 | 60.945 |
| overall mAP | 43.300 | 36.314 |
| stage 1 mAP | 47.800 | 41.398 |
| stage 2 mAP | 48.300 | 42.649 |

Verdict `G2 = RED` is present. Because overall, stage-1, and stage-2 spatial
mAP are all lower, the observed gap is not isolated to temporal identity. It
contains a general decoder/data/optimization/reproduction component. This does
not identify a unique root cause.

### 2. P2 recipe alignment: verified

`artifacts/P2/config_audit.md` and
`conf/config_p2_rescene4d_concerto_t2.yaml` verify Concerto, 100 FPS
non-parametric queries, 2 cm voxels, RIO T=2 plus ScanNet T=1 at weights
`1.0/0.8`, contrastive loss enabled, spatio-temporal serialization enabled,
spatio-temporal masking disabled, class/mask/dice weights `2/5/2`, no-object
weight `0.2`, AdamW, OneCycleLR max LR `5e-4`, 450 epochs, and effective batch
32. Re-enabling contrastive or reapplying these already-audited defaults is not
an authorized explanation or repair.

### 3. Frozen encoder runtime hypothesis: verified as a risk, not a cause

At the current commit, `trainer.trainer.InstanceSegmentation._freeze_backbone_parameters`
sets `requires_grad_(False)` for `backbone.model.enc` and
`backbone.model.embedding` when `freeze_mode == "backbone_encoder"`.
`_set_frozen_modules_eval` calls `eval()` only for `freeze_mode == "backbone"`.
There is no epoch hook that restores the encoder-only modules to eval mode after
Lightning calls `train()`. The frozen P2 audit records Concerto drop path `0.3`.
This supports a controlled hypothesis only; no metric effect has yet been
measured.

### 4. Physical microbatch hypothesis: verified as a risk, not a cause

`artifacts/P2_G2_REPRODUCTION_REPORT.md` records 2 GPUs, batch 2/GPU, and
accumulation 8: effective batch 32 but physical global microbatch 4. The report
also records per-microbatch loss normalization. This does not prove equivalence
to, or explain the gap from, a physical global batch of 32.

### 5. Protocol B population: verified

`conf/p6a/default.yaml` and `scripts.p6a_protocol` bind a common-T5 population:
43 masters, 6 reference-scene clusters, horizons T2-T5, and deterministic
`canonical`, `reverse`, and `sha256_seed45` orders. The order semantics are
explicitly `metadata_order_only_no_timestamps`. The P6A report explicitly does
not claim reproduction of an external benchmark score.

### 6. Protocol-B task evaluator and T2 values: verified

`scripts.p6a_metrics.OfficialMetricAccumulator` imports
`stmetrics.InstanceMetrics`, `LegacyAPEvaluator`, and `TemporalEvaluator`.
System Comparison V1 therefore uses the official metric implementation rather
than a hand-written replacement. Frozen T2 FullHistory values are:

- canonical: `0.20722658932209015`
- reverse: `0.21109874546527863`
- sha256_seed45: `0.1711808741092682`
- three-order pooled: `0.19099636375904083`

The paper `34.8`, P2 official-like `27.939`, and Protocol-B values are different
population/protocol observations and cannot be presented as a direct causal
performance drop.

### 7. T2 raw-observation regression: verified

`scripts.system_comparison_inference._observation_fingerprints` hashes
`features`, `class_prob`, `confidence`, `valid`, `masks`, `mask_support`, and
`local_query_ids`. `assert_t2_observation_regression` checks exact request
identity and equality of all per-field plus combined hashes. This proves raw
local-observation parity for covered T2 cache entries, not official task-output
parity.

### 8. Task post-processing confound: verified in code

FullHistory uses `postprocess_full_history_output`, which calls the existing
ReScene chain `_get_predictions`, `_get_batch_masks`, `_get_mask_and_scores`,
`_get_full_res_mask`, and `_filter_and_sort_predictions` before storing official
masks, classes, and scores.

Legacy Persist4D instead converts schema-v3 raw observations through
`stage_prediction_from_track_step` and `IdentityAccumulator`. The path applies
raw-observation `valid` and tracker `valid` filtering, uses raw confidence as
the score, maps raw-query class posterior semantics, and later averages class
posteriors and scores across track observations. Thus V1 changes candidate
filtering, class semantics, score semantics, and identity association at the
same time as history representation. This is a code-level evaluation confound;
its numerical effect is not yet identified.

### 9. T2 current-stage parity red light: verified

`artifacts/system_comparison/aggregate_results.csv` records pooled T2
current-stage AP `0.37085068225860596` for FullHistory and
`0.2900303304195404` for Persist4D. At T2 both current-stage model inputs cover
the same `(X1, X2)` pair, so official current-stage task output must be tested
for parity. Causal t-mAP equality is not required because FullHistory can revise
the stage-1 prefix while strict streaming cannot.

## Commands and results

```text
git status --short
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
sha256sum checkpoints/rescene4d_concerto_t2_repro.ckpt
```

The branch and hashes match the values above. Baseline regression command:

```text
conda run -n persist4d python -m pytest -q \
  tests/test_p6a_cache.py \
  tests/test_system_comparison_inference.py \
  tests/test_system_comparison_metrics.py \
  tests/test_system_comparison_analysis.py \
  tests/test_run_system_comparison.py
```

Result: `62 passed in 4.47s`.

## E0 gate

`E0 = PASS` for evidence-preserving Line-A development. The frozen evidence,
checkpoint identity, runtime, starting worktree, and scoped test baseline are
bound. Ignored dataset databases and tensor cache entries are not tracked in a
fresh worktree; they must be linked from their frozen external locations before
runtime parity evaluation, without changing their manifests or payloads.
