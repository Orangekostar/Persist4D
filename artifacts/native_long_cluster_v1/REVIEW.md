# Primary Review

Reviewed by the primary agent under the user's AGENTS.md governance.
Base: a3830a1f1efe66a55df8b147d86941b85165c4cc.

- Budget amendment is explicit, archives old locks and binds reviewed code,
  configuration, population, data content and common initialization.
- E0-E3 have separate host/runtime/artifact ownership and disjoint GPU-hour
  quotas. Controller and worker mutations use OS advisory locks and atomic JSON.
- Training keeps U29700, global batch32, world2, batch1/rank, accumulation16,
  FP32 and the original common seed45 initialization.
- Startup time, failed processes and cleanup time are charged. Budget stopping
  is synchronized across DDP ranks at optimizer boundaries. Timeout terminates
  the subprocess group. An unresolved reservation prevents duplicate GPU work.
- Resume uses committed checkpoint progress. The real CPU resume fixture
  verifies optimizer parameters, global draws and learning-rate continuity.
- Remote preparation verifies content SHA and bound dependency versions/source
  trees. Actual native two-GPU preflight is required before official training.
- Collection retains offline accounting, deduplicates immutable event IDs,
  rejects identity conflicts and verifies downloaded official metric evidence.
- CAL selection requires complete full-U trajectories and all ten complete
  endpoints. Explicit failures are recorded. SEL uses controller-locked steps
  and E1 controls at the module-selected step; partial denominators cannot
  produce reported gains.
- Partial SEL-dispatch failure is isolated to its node. Later polling retries
  the same immutable lock, including while another arm is already evaluating.
- The old publisher cannot overwrite the current cluster state. Historical
  native-long scientific artifacts are preserved.

No unresolved issue blocks publishing the budget/scheduler implementation.
At the initial code-only delivery, runtime preparation and actual DDP/long-run
behavior remained unverified remotely.
The prior native runtime protocol was conditional; this change supplies no new
numerical-repeatability evidence. Forced supervisor termination requires PID
inspection and ledger reconciliation before restarting a charged process.

## Deployment Review, 2026-10-01

- Dedicated environments preserve the checked Torch2.6.0+cu126 and Lightning2.6.5
  versions; the original remote projects/environments were not modified.
- Staged library sources resolve through an explicit worker PYTHONPATH. Content
  verification covers3575 data/metadata files, bound numerical library sources,
  frozen encoder, common initialization and identical population bytes.
- A real CPU sample-loading check reproduced missing relative augmentation
  configuration because the launcher inherited the SSH home directory. The
  launcher now uses the staged repository as its working directory. Its
  regression test launches a real child and verifies the child's directory.
- Task-specific SSH multiplexing addresses the observed MaxStartups rejection
  without changing the server or unrelated connections. SSH and rsync reuse
  the same transport; the connection regression passes.
- All four native world2 preflights passed two disposable optimizer updates
  with effective batch32 and scheduler U29700. They are separately charged,
  excluded from official progress, and bound to the final worker code.
- Four detached training supervisors and a detached controller monitor were
  started. Controller lock contention is retried, and real monitor status has
  advanced across polls. Quotas and the1800 GPUh lifetime cap remain enforced.

The first official startup subsequently failed in all arms at rank1 draw65,
with observed update2 and no committed checkpoint. Its real target contains0
instances; legacy mask/Dice normalization divided by0. Empty matches now produce
graph-connected zero mask losses while preserving no-object classification.
Tests verify finite gradients, auxiliary losses and an unchanged nonempty mask
objective. Removed an internal InfoNCE override of the caller's TF32 setting.

Explicit failed-job replacement verifies all four remote jobs are stopped,
unreserved and uncheckpointed, retains their original code/spec/log/ledger
directories, archives controller snapshots, and merges every failed cost event
before allocating replacement quotas. Replacement hosts, data, weights,
population, seed45 state and lifetime cap remain bound to the original values.
The replacement deployment uses new code/runtime directories. A separate exact
controller code copy owns persistent live reports outside the Git worktree.

The deployment receipt records observed optimizer progress separately from
checkpointed progress. Full-U completion, CAL/SEL gains, seed46 confirmation,
formal confirmation and profiling remain pending. Passing the startup preflight
does not establish numerical repeatability or successful completion of a
multi-day trajectory.
