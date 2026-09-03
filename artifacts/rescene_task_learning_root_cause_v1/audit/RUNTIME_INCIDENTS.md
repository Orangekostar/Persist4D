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
- The first post-migration epoch-15 checkpoints passed full-state validation at optimizer step 990. R1 contained 798 model-state entries and A1 contained 802; each contained one optimizer state, one scheduler state, and an epoch-boundary sampler-generator state. Both source files matched their cross-host replicas by SHA256.
- At the first standard validation point, A1 led R1 by 4.909 percentage points in `SpatialStageMean`. This is a trajectory health observation only, not the preregistered three-seed official-like evaluation and not an authorized final-selection result. Exact values and checkpoint identities are recorded in `runtime_migration_2gpu/FIRST_CHECKPOINT_AUDIT.json`.
- The epoch-30 checkpoints passed the same full-state and cross-host replica checks at optimizer step 1,980. At this second standard validation point, R1 led A1 by 0.085 percentage points in `SpatialStageMean`; the earlier A1 lead was therefore not sustained through epoch 30. This remains a trajectory health observation only and is not an authorized final-selection result. Exact values and checkpoint identities are recorded in `runtime_migration_2gpu/EPOCH30_CHECKPOINT_AUDIT.json`.
- Against the committed pre-migration RC3 R1 curve, migrated R1 `SpatialStageMean` differed by -0.218 percentage points at epoch 15 and +0.107 percentage points at epoch 30. These are descriptive deltas, not evidence of statistical equivalence; no corresponding A1 comparison is available while node25 remains inaccessible.
- The epoch-45 checkpoints passed full-state validation and source-to-replica SHA256 equality at optimizer step 2,970. R1 led A1 by 0.750 percentage points in `SpatialStageMean`; both variants improved from epoch 30, by 10.352 and 9.687 percentage points respectively. This remains a trajectory health observation only and is not an authorized final-selection result. Exact values and checkpoint identities are recorded in `runtime_migration_2gpu/EPOCH45_CHECKPOINT_AUDIT.json`.
- At epoch 45, migrated R1 `SpatialStageMean` differed from the committed pre-migration RC3 R1 curve by -0.627 percentage points. This remains descriptive only; no trajectory-equivalence threshold was preregistered.

## Node3 root-volume expansion on 2026-09-03

- At `2026-09-03T11:15:10Z`, node3 had 296,802,746,368 bytes available on its root filesystem. The migrated A1 tree occupied only 28,710,584,320 bytes and contained no recently created file larger than 100 MB.
- Root-cause evidence attributed the concurrent capacity loss to a separate workload tree named `oviovo_baseline_runs`: nine recently written `.4dmap` files totaled 112,954,641,434 bytes on node3. The A1 training output was not the source of the growth.
- The existing `ubuntu-vg` physical volume had 840,978,923,520 bytes unallocated. At `2026-09-03T11:19:02Z`, `/dev/ubuntu-vg/ubuntu-lv` and its ext4 filesystem were expanded online from 1,073,741,824,000 bytes to 1,914,720,747,520 bytes.
- After expansion, the mounted filesystem reported 1,884,068,257,792 bytes total and 1,090,949,926,912 bytes available. Kernel logs recorded a successful ext4 resize and no ext4 or block-I/O error.
- A1 rank processes remained live throughout the resize. No training code, scientific configuration, dataset, initialization, optimizer, scheduler, or metric contract changed.

## Integrity rule

After node25 recovers, record its boot identity and uptime, container restart/OOM state, checkpoint inventory, metric-file continuity, NFS recovery evidence, and training-process state before resuming finalization. If a process did not survive, resume only from a checkpoint whose full-state and provenance contracts pass the existing validators.
