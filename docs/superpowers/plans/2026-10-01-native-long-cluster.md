# Native Long Cluster Implementation Plan

> Execute inline. The primary agent owns budget semantics, concurrency, implementation and review. The user authorized budget adjustment and scheduler changes; this task does not start long training.

**Goal:** Replace the empty 192 GPUh plan with an explicitly amended 1800 GPUh lifetime plan and a resumable four-host scheduler.

**Architecture:** The controller owns the resource lock, immutable CAL selection and merged reports. E0-E3 run independently on .101-.104 with isolated runtime/artifact directories and disjoint GPUh allocations. .105-.106 remain reserved. SSH worker processes run existing native training and fresh-process CAL, collect validated JSON evidence, then evaluate SEL after one controller CAL lock.

**Tech Stack:** Existing Python, PyTorch/Lightning, subprocess SSH/rsync, JSON, fcntl controller locks; no new dependencies.

**Spec:** User requests dated 2026-10-01 and EXECUTION_INSTRUCTION.md; the new user authorization supersedes only the 192 GPUh and two-GPU campaign limits. Numerical/data/selection rules remain fixed.

## Constraints

- Lifetime cap1800 GPUh includes prior80.73503875023967 and current campaign consumption. Reserve8 GPUh. Forecast already includes its single1.25 margin and CAL proxy.
- Four arms seed45, U29700, global batch32, world2, batch1/rank, accumulation16. Identical common initialization and population content.
- Six hosts192.168.100.101-106, SSH userpluto; four training assignments and two reserves. Maximum12 GPUs; main training8 GPUs.
- Amendments archive the previous lock and cannot silently alter a trained/selected trajectory. Repeated planning/dispatch must be idempotent.
- Workers never write the controller's state or ledger. The controller deduplicates immutable event IDs and rejects conflicting duplicate events and incomplete CAL denominators.
- Do not launch long training during this implementation task. Missing environment/data readiness is reported as a concrete blocker.

## Tasks

- [x] Inspect current code, budget evidence and six-host SSH/GPU access.
- [x] Add failing tests for amended budget, disjoint allocations, immutable lock/resume, independent worker evidence and controller-only selection.
- [x] Parameterize actual budget enforcement and scoped worker artifact roots; add optimizer-boundary budget stopping.
- [x] Add cluster configuration, explicit budget-amend/plan, SSH probe/stage/start/status/collect/finalize commands and arm-specific native execution.
- [x] Make existing run/status/report route cluster plans correctly without invoking the empty-plan report.
- [x] Apply the new lock to the actual runtime, preserve old scientific results and record remote readiness.
- [x] Run focused tests, lint/compile/diff checks and personally review failure/recovery paths.
- [ ] Publish the verified code changes to the research branch and cluster tag.

## Acceptance

Planning selects FULL_FOUR and allocates four disjoint quotas whose sum plus historical/current GPUh and reserve does not exceed1800. Replanning with the same inputs preserves bytes. Concurrent start is protected by a controller lock and each job has a strict identity and OS advisory worker lock. Collection validates code/population/common-init identity, complete CAL rows and event uniqueness; no partial population can lock selection. All actual training steps remain0 for this development turn. Report exact remote readiness and runnable commands.

## Verified State

46 focused CPU tests pass, including the real optimizer-boundary resume fixture.
Ruff and compilation pass. The actual amended lock selects FULL_FOUR at1800
GPUh, has byte-identical repeated planning, and matches the final code closure.
All six hosts pass SSH and idle dual-A40 checks. Their configured default Python
and dedicated data/weight/code paths are unprepared. Long training was not
started; all four official update counts remain0.
