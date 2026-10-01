# Native Long Cluster Deployment Plan

> Execute inline under the primary agent's ownership. User authorization:
> "complete these yourself", following the budget/scheduler implementation.
> Use the existing reviewed scheduler; do not delegate scientific decisions.

**Goal:** Prepare E0-E3 hosts, pass actual native two-GPU preflight, launch all
four full trajectories and leave a persistent controller monitor.

**Architecture:** Clone the existing version-matched remote environments into
the dedicated cluster directory. Stage the fixed local dependency sources,
encoder, common state and SHA-bound data. Keep isolated worker roots and quotas.

**Tech Stack:** Existing Python/Lightning, rsync, SSH, cp and OS locks.

**Spec:** `docs/native-long-cluster.md`, immutable source/population bindings,
1800 GPUh lifetime cap and user requests dated 2026-10-01.

## Constraints

- Preserve seed45 common state, population, FP32/TF32-off, U29700 and batch32.
- Use .101-.104 for E0-E3; .105-.106 remain reserved.
- Existing projects, environments and Docker containers are read-only.
- Clone source environment: R1 on .101/.102, A1 on .103, superiority on .104.
- No large archive on the controller: only6.3 GiB is free locally.
- Bound data is10.575 GiB across3575 files. .104 initially has71 GiB free.
- Amend the still-unstarted job lock before staging the final configuration.
- Actual GPU preflight/training consumption must use each worker's ledger.
- A persistent detached monitor owns collection and locked SEL dispatch.
- Starting multi-day experiments is not equivalent to completing their results.

## Tasks

- [x] Inspect six-node hardware, disk, existing Python environments and data size.
- [x] Add failing deployment tests for module search paths and native dependencies.
- [x] Implement staging of Concerto/Sonata/stmetrics/Detectron2/PointNet2 and bind remote Python.
- [x] Clone matched environments into .101-.104 dedicated roots; verify CPU package versions.
- [x] Amend the unstarted lock, stage and SHA-verify data/weights/code per worker.
- [x] Launch workers and verify each actual native two-GPU preflight and progress.
- [x] Start a persistent monitor, verify liveness and merged cost accounting.
- [ ] Publish verified deployment changes and record exact running/pending state.

## Startup Failure Recovery

All four disposable DDP preflights passed. Official training then failed on the
same rank1 draw65 at update2: actual GT has0 instances with33190 points and1868
segments. Legacy mask loss divides by0 matched instances. No checkpoint or
CAL/SEL evidence exists; failed reservation costs remain in the original ledger.

- [x] Add empty-target loss/gradient and unchanged nonempty-objective regressions.
- [x] Return graph-connected zero mask/Dice loss for zero matched instances;
  keep the sample and its no-object classification supervision.
- [x] Add an explicit recovery command accepting only all-failed, inactive,
  uncheckpointed jobs; use new remote code/runtime roots and archive controller
  state, carrying forward
  every existing cost event before recomputing disjoint quotas.
- [x] Preserve failed code/specs/logs/ledgers and record the replacement lineage.
- [x] Re-stage, pass actual preflight, and verify sustained updates beyond the
  original failure before publishing a running status.

The contrastive loss also overrode the caller's TF32 policy. Removed that global
mutation and verified both enabled/disabled policies with finite gradients.
Expanded shared criterion/DDP/resume tests:200 passed,2 CUDA-dependent skips,
four existing warnings in138.84 seconds. New scripts/tests pass Ruff; criterion
has17 pre-existing diagnostics and no new ones (baseline19).

Actual replacement proof: all four DDP preflights PASS; observed official updates
E0/E1/E2=10, E3=12, with committed updates0 before checkpoint990. Eight assigned
A40 GPUs are computing. Detached monitor PID3271783 uses an exact code copy,
keeps live reports outside Git and has advanced across repeated status polls.
The startup failure consumed0.4121425855261946 GPUh; all costs are retained.

## Validation

Use the existing focused CPU native-long tests, Ruff/compilation and diff checks.
Remote bootstrap verifies versions, Python source hashes and every data/weight
SHA. Actual DDP preflight must record PASS with2 disposable optimizer updates,
world2, effective batch32 and full scheduler U29700. Official training must
advance beyond0, with eight A40 GPUs owned by the four bound jobs. Verify the
controller monitor survives tool-session completion and collects current job
identities/costs. Record any remaining scientific stages honestly.
