# Verification

The original code-only verification below is retained as history. Current
deployment evidence is in `DEPLOYMENT_VERIFICATION.json` and the final section.

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

## Deployment Verification, 2026-10-01

The initial deployment focused command passed51 tests in14.19 seconds. After
the real empty-target startup failure, expanded validation passed200 tests,
with2 CUDA-dependent skips and four existing warnings, in138.84 seconds:

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q \
  tests/test_native_long_*.py tests/test_objective_semantics.py \
  tests/test_perception_gain_native_retry.py tests/test_contrastive_streaming.py \
  tests/test_task_memory_criterion.py tests/test_local_diagnostic_loss.py \
  tests/test_p2_ddp_batch_contract.py
```

Regressions cover deployment paths/transport/locks, real child cwd, empty-target
losses/gradients, nonempty-objective equality, TF32 policy preservation, cost
carryover and rejection of unsafe failed-job replacements. Ruff passed changed
scripts/tests. Criterion Ruff has17 pre-existing diagnostics and no new ones
(baseline19). Compilation and diff checks passed.

Assigned .101-.104 workers are staged and READY. Each passed CPU verification
of3575 files, numerical library sources, encoder/common-state hashes, population
bytes and real ScanNet/RIO training samples. No CUDA context was created by
those CPU checks. Existing environments were cloned into dedicated roots.

All four actual native two-A40 preflights record PASS, world2, effective batch32,
two disposable optimizer updates, official updates0 and scheduler U29700.
Their reservation time is recorded in worker ledgers and merged centrally.
Official training is running under four detached supervisors. The persistent
controller monitor uses30-second polls; its actual PID/liveness and advancing
status timestamps are recorded in the deployment receipt.

The first startup failed at draw65 after two official observed updates, without
a checkpoint. Recovery carries0.6303522524447762 campaign GPUh, including all
failure costs, into the new lock. Original worker directories and controller
failure evidence remain intact. All four replacement CPU checks also verify
the actual empty draw65, finite zero mask losses/gradients and TF32-off.

The receipt distinguishes observed optimizer updates from committed checkpoint
updates. Until update990, committed progress can remain0 while training runs.
No full-U completion or scientific gain is claimed by this deployment.
