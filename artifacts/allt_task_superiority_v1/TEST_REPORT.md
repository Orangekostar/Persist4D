# Persist4D All-T Test Report

Status: PASS. The direct suite completed with `45 passed, 1 warning` in the
Persist4D environment. The warning is the existing Albumentations import of
the deprecated `scipy.ndimage.filters.gaussian_filter` namespace.

## Direct verification

```bash
/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_persist4d_allt_model.py \
  tests/test_persist4d_allt_evaluation.py \
  tests/test_analyze_persist4d_allt_final.py \
  tests/test_profile_persist4d_allt.py \
  tests/test_publish_persist4d_allt.py \
  tests/test_persist4d_allt_selection.py \
  tests/test_finalize_persist4d_allt.py
python -m ruff check scripts/evaluate_persist4d_allt.py \
  scripts/analyze_persist4d_allt_final.py scripts/profile_persist4d_allt.py \
  scripts/publish_persist4d_allt.py tests/test_persist4d_allt_evaluation.py \
  tests/test_analyze_persist4d_allt_final.py tests/test_profile_persist4d_allt.py \
  tests/test_publish_persist4d_allt.py
git diff --check
```

## Real failures retained

- Scientific gate: C2 failed strict all-T t-mAP against both R1 B4 and matched FH-adapt.
- Scientific gate: only 5/20 C2-vs-FH-adapt task cells were positive.
- Operational: the first remote memory-diagnostic write hit NFS permissions; the target namespace permissions were corrected before rerun.
- Contract: the first profile smoke exposed unequal stochastic T2 preparation; deterministic per-cell preparation seeding was added and the smoke was rerun before the formal profile.
- Environment: invoking the direct suite with base Python failed collection because `torch_scatter` was absent; the exact suite passed under the declared Persist4D environment.
