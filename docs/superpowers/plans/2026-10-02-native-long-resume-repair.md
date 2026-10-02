# Native Long Resume Repair Implementation Plan

> Execution remains with the primary agent under project governance.

**Goal:** Restore all four seed45 trajectories from their valid update2970 checkpoints without losing training state or costs.

**Architecture:** Initialize the boundary callback from Lightning's restored optimizer position. Apply a reviewed controller amendment and metadata-only checkpoint migration, retaining original code, checkpoints, CAL evidence, budget and ledger in repair archives. Recompute CAL under the repaired code and supervise the controller with a user systemd service.

**Tech Stack:** Python, Lightning 2.6.5, PyTorch 2.6.0, SSH, systemd.

**Spec:** `docs/native-long-cluster.md` and the user's request to fix errors and continue.

## Constraints

- Preserve model, optimizer, scheduler, loop, draw and rank-local RNG state.
- Preserve the lifetime1800 GPUh cap, prior costs, job quotas, common seed45 initialization, batch32, world2, accumulation16 and total29700 updates.
- Permit changed source only in execution/control scripts; reject model, data, configuration or initialization changes.
- Never stage over a running worker. Reject selection locks, active reservations and unaccounted processes.
- Archive original evidence and explicitly bind old/new job IDs and checkpoint SHAs.

## Tasks

- [x] Reproduce with real Lightning accumulation and restored checkpoint callbacks; verify sampled draws, scheduler and parameters.
- [x] Initialize callback state at training start and run regression tests.
- [x] Implement audited repair migration and historical event provenance; test rejection and idempotence.
- [x] Verify remote checkpoint positions and payload integrity; deploy and launch CAL recomputation.
- [x] Restore supervised monitoring and verify all four workers pass update2970 and save a subsequent boundary.
- [x] Publish the change and verified recovery receipt.

Validation: CPU pytest native-long/resume/cluster tests, Ruff, diff checks, actual two-GPU resumes, metadata/tensor payload hashes and fresh controller snapshots.

Verified recovery: all four arms resumed2970->2972 and saved valid rank2,
draw95104, scheduler29700 checkpoints. The next segments are running toward5940.
The user systemd service survived an injected SIGKILL and restarted after15
seconds with a fresh PID receipt and new snapshots. SSH also passed in a separate
user service without an agent or multiplexed connection.

Tests: broad suite213 passed,2 CUDA-dependent skips in178.67 seconds; final
resume/cluster/repair suite45 passed in16.01 seconds. Ruff and unit syntax pass.

CAL was recomputed completely for all arms. E0/E1/E2 metrics match the archived
results exactly. E3 differs (maximum absolute AP delta0.004661515355110168 atT4).
Its migrated checkpoint state digest remains identical to the original. The
existing numerical protocol uses deterministic algorithms with `warn_only=True`;
logs report unsupported deterministic CUDA operations. This does not establish
the cause of the E3 evaluation difference. Repeat-evaluation reproducibility is
still an evidence gap, separate from the repaired optimizer-boundary crash.

Published code commit: `ceb6c4aa49fbfa0c6ff66e113e7cb76a562fc7a9`.
Every bound Git blob SHA matches the staged workers/controller. Archived
official CAL evidence was also copied to the controller repair archive and
verified against all original snapshot SHAs. Prefix mappings are recorded in
`artifacts/native_long_cluster_v1/RESUME_REPAIR_VERIFICATION.json`.
