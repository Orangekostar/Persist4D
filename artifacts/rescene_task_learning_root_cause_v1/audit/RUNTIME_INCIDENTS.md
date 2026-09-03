# Runtime Incidents

This record separates infrastructure availability from scientific outcomes. Neither incident changes a training variant, optimizer, scheduler, dataset contract, evaluator, checkpoint-selection rule, or reported metric.

## Compatibility NFS outage on 2026-09-03

- First kernel non-response evidence: `2026-09-03T07:00:24Z`.
- Kernel recovery evidence: `2026-09-03T07:09:20Z`.
- Effect: ScanNet data-loader workers temporarily waited in `rpc_wait_bit_killable`.
- Recovery: the hard-mounted clients resumed without restarting either training process and without checkpoint rollback.
- Control-plane response: finalization and disk-guard probes were restricted to the direct live filesystem so a compatibility-mount failure could not block those loops.

## Node25 outage on 2026-09-03

- First kernel NFS non-response evidence: `2026-09-03T07:30:55Z`.
- Last probe recorded here: `2026-09-03T07:43:56Z`.
- State at the last probe: node25 had no ARP response; ICMP, SSH, and NFS were unreachable from nodes 103, 106, and 107.
- Last reliable training observations before the outage: R1 logger epoch index 368; A1 logger epoch index 306.
- Local effect: the R1 rank-zero process waited on the direct live filesystem while the rank-one process remained alive. Existing processes were preserved to allow hard-mount recovery.
- Remote effect: A1 container state could not be inspected while the host was unreachable.
- A bounded two-batch evaluator smoke was active when node25 became unreachable. This is a temporal association only; no causal attribution is supported.
- The smoke control connection was terminated locally after the host remained unreachable. It produced no accepted evaluation artifact and contributes no numerical result.
- Mitigation decision: do not run a third-GPU evaluation concurrently with A1 training. Remote final evaluation is deferred until training workloads have ended and node25 health is revalidated.

## Two-GPU host migration on 2026-09-03

- A cluster-wide inventory found two independent hosts with two idle A40 GPUs each, sufficient memory, and local output capacity.
- No replica of the latest R1 or A1 full-state checkpoint was available outside node25. The migrated jobs are therefore independent deterministic restarts from the frozen common initialization, not continuations from logger epoch indices 368 and 306.
- The original jobs were not terminated. If node25 recovers, its checkpoint and metric continuity must be audited before choosing between the original and migrated trajectories.
- The original authorization recorded three visible A40 GPUs even though the fixed training world size and selected devices were both two. Migration-specific authorizations change only `runtime.gpu_count` and `runtime.gpu_models`; all scientific configuration fields remain unchanged.
- Both migrated jobs passed their complete authorization, source, data, dependency, pretrained-weight, and common-initialization checks before launch.
- ScanNet was copied to each host-local filesystem after initial observation showed that concurrent reads saturated the shared storage device. The post-copy content hashes passed the same frozen data contract.
- Both A40 pairs use a 220 W power limit and one-minute temperature monitoring. The first sustained Epoch 0 observation covered 57--68 degrees Celsius with no new runtime error.
- R1 and A1 checkpoints are configured for cross-host replication after each stable `last.ckpt` update.

## Integrity rule

After node25 recovers, record its boot identity and uptime, container restart/OOM state, checkpoint inventory, metric-file continuity, NFS recovery evidence, and training-process state before resuming finalization. If a process did not survive, resume only from a checkpoint whose full-state and provenance contracts pass the existing validators.
