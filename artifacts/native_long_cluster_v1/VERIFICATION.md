# Verification

Date: 2026-10-01. Scope: budget amendment and four-host scheduling code.

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_native_long_*.py tests/test_objective_semantics.py \
  tests/test_perception_gain_native_retry.py
```

Result: 46 passed, four existing dependency/Lightning warnings, 13.07 seconds.
All fixtures used CPU only. The process-cost tests reserve simulated card counts
in temporary ledgers; they did not create real GPU training jobs.

Ruff passed for all five changed/new scheduler scripts and both new test files.
Compilation passed for all five scripts. `git diff --check` passed.

The actual runtime budget lock is
`ed7f147851926c84b01d1134d237ec7d318e79a8e180d17c28216d261d283805`.
The previous reviewed cluster lock is
`8dad9d0e2070fdacf4f2a6b6f73877c75560bcb8aba08ddbff5d0896dfa18dea`.
The original192 GPUh lock is archived unchanged as
`847d3edf1e552874d7f216bf242cb1b59022309f1329602f1af716c0eb9bba94`.
History is under the controller runtime's `selection/budget_history/`.
An explicit unstarted amendment verified all four old worker roots over SSH
before binding startup-time accounting and resumable SEL dispatch. Repeated planning with
the final code/configuration preserved the actual lock bytes.

Lifetime cap1800 GPUh; historical/current consumption80.95324841715825 GPUh;
remaining1719.0467515828418 GPUh. Worker allocations plus consumed hours and the
eight-hour reserve total1799.999999111901 GPUh.

The existing campaign `status`, `run --through preflight` and report generator
route to the cluster path. Status and report agree. Root cost-ledger bytes are
unchanged by this implementation task, and official updates remain0/29700 for
all four arms.

Actual remote probe: six SSH successes and six idle dual-A40 hardware successes.
The configured default `python3` lacks Torch, Lightning, spconv, Concerto,
Sonata and stmetrics. Dedicated code/data/encoder/common-state paths and worker
specifications are absent. Readiness is PREPARATION_REQUIRED, recorded in
READINESS.json. No staging, remote GPU context, actual native DDP or long
training was executed in this task.

Scientific completion remains pending: no CAL/SEL gain, seed46 confirmation,
formal confirmation or profiling result is claimed.

The SEL-dispatch regression reproduces one failed SSH dispatch, verifies the
other three jobs still dispatch, then retries while a sibling is already in
SEL_RUNNING. The central CAL-lock bytes remain unchanged across that retry.
