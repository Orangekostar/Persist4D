# Persist4D All-T Test Report

Status: PASS. The complete directly relevant suite completed with `143 passed,
1 warning` in the Persist4D environment, including the canonical real-A40 GPU
query-export parity gate. The warning is the existing Albumentations import of
the deprecated `scipy.ndimage.filters.gaussian_filter` namespace.

## Direct verification

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
P5_VERIFY_GPU_ARTIFACTS=1 \
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_persist4d_allt_contract.py \
  tests/test_persist4d_sequence_dataset.py \
  tests/test_rescene_query_features.py \
  tests/test_persist4d_allt_model.py \
  tests/test_persistent_memory_read.py \
  tests/test_query_competition_adapter.py \
  tests/test_persist4d_allt_trainer.py \
  tests/test_persist4d_allt_training_script.py \
  tests/test_persist4d_allt_evaluation.py \
  tests/test_persist4d_allt_diagnostics.py \
  tests/test_persist4d_allt_analysis.py \
  tests/test_analyze_persist4d_allt_final.py \
  tests/test_system_comparison_metrics.py \
  tests/test_rescene_task_postprocess.py \
  tests/test_system_comparison_analysis.py \
  tests/test_profile_persist4d_allt.py \
  tests/test_persist4d_allt_selection.py \
  tests/test_publish_persist4d_allt.py \
  tests/test_finalize_persist4d_allt.py
git diff --name-only -z 2c7494b982eff84886aef3a7274bae43752a1485 -- '*.py' | \
  xargs -0 /home/ww/miniconda3/bin/ruff check
git diff --check
```

The GPU parity gate requires the canonical
`checkpoints/rescene4d_concerto_t2_repro.ckpt` to be materialized as a regular
file. For this run it was a same-filesystem hard link to the existing trusted
checkpoint, so no duplicate checkpoint blocks were allocated; the temporary
link was removed immediately after the suite.

## Real failures retained

- Scientific gate: C2 failed strict all-T t-mAP against both R1 B4 and matched FH-adapt.
- Scientific gate: only 5/20 C2-vs-FH-adapt task cells were positive.
- Operational: the first remote memory-diagnostic write hit NFS permissions; the target namespace permissions were corrected before rerun.
- Contract: the first profile smoke exposed unequal stochastic T2 preparation; deterministic per-cell preparation seeding was added and the smoke was rerun before the formal profile.
- Environment: invoking the direct suite with base Python failed collection because `torch_scatter` was absent; the exact suite passed under the declared Persist4D environment.
