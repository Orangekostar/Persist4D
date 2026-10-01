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
Runtime preparation and actual DDP/long-run behavior remain unverified remotely.
The prior native runtime protocol was conditional; this change supplies no new
numerical-repeatability evidence. Forced supervisor termination requires PID
inspection and ledger reconciliation before restarting a charged process.
