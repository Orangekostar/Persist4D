# Native long cluster operations

The controller runtime is `/home/ww/persist4d_runs/native_long_retrain_v1`.
The reviewed configuration is `conf/native_long_cluster.yaml`.
E0-E3 run on .101-.104 with two A40 GPUs each; .105-.106 are reserved.
The lifetime budget is 1800 GPUh including historical and current consumption.
Each worker has a disjoint quota, and eight GPUh remain reserved centrally.

Run commands from the repository with the `persist4d` Python environment:

```bash
export NATIVE_LONG_ROOT=/home/ww/persist4d_runs/native_long_retrain_v1
python -m scripts.native_long_cluster plan --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_cluster probe --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_cluster stage --with-data --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_cluster probe --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_cluster start --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_cluster watch --root "$NATIVE_LONG_ROOT"
```

`plan` archives the previous budget lock. An identical repeat preserves the lock
bytes. A changed code/configuration identity requires an explicit reviewed
amendment; it cannot silently alter a running trajectory. After reviewing changed
environment paths, run `plan --amend-unstarted --root "$NATIVE_LONG_ROOT"`.
This checks every old worker over SSH and refuses amendments after any charged
work, optimizer updates or unresolved reservation. An unavailable worker also
prevents amendment.

After a verified startup failure affecting all four jobs, a separate reviewed
recovery can use `plan --replace-failed-uncheckpointed --root "$NATIVE_LONG_ROOT"`.
It requires every old worker to be FAILED, stopped, unreserved and without any
checkpoint or CAL/SEL evidence. New code/runtime directories must be separate;
hosts, environment, assets, population, initialization and lifetime cap are
preserved. Old snapshots/logs/controller receipts are archived under
`cluster/failures/<old-lock-sha>/`; old remote directories remain intact. All
failed cost events are merged before replacement quotas are allocated.

For the reviewed optimizer-boundary callback repair after committed training,
stop the controller monitor and use:

```bash
python -m scripts.native_long_repair repair --root "$NATIVE_LONG_ROOT"
```

This accepts only stopped, checkpointed FAILED workers without active reservations
or selection locks. Model, dataset and configuration sources cannot change.
The controller archives its original budget/state and snapshots under
`cluster/repairs/<repair-id>/`; each worker archives code, original checkpoints,
CAL evidence, specs and costs under `repairs/<repair-id>/`. The checkpoint
migration changes only the code identity and verifies numerical, loop, RNG and
configuration content with a deterministic SHA before/after serialization. Job quotas and the lifetime
cap are unchanged. Old cost events retain their original job IDs with an explicit
event-to-job mapping in the amended plan. CAL is recomputed using the repaired
checkpoint/code SHA; original results remain in the archive.

The boundary callback initializes from Lightning's restored `global_step`, so
accumulated microbatches cannot resave the loaded checkpoint before an optimizer
update. Its existing strict checkpoint-boundary validation remains active.

The controller now runs as `persist4d-native-long-cluster.service` using the
reviewed code copy `cluster/controller-code-resume-v3`. The unit template is
`docs/native-long-cluster-monitor.service`. The user manager has lingering
enabled. Unexpected exits restart after15 seconds; diagnostics are retained in
the user journal. If all workers fail, the monitor reports the failure without
attempting an invalid CAL selection.

```bash
systemctl --user status persist4d-native-long-cluster.service --no-pager
journalctl --user -u persist4d-native-long-cluster.service -n 30 --no-pager
```

Each assigned host uses `/home/pluto/native_long_cluster_v1/env/bin/python`,
cloned from its existing version-matched environment without changing the source.
Staging transfers the code, fixed encoder, common initialization, SHA-bound
data/metadata and native Concerto/Sonata/stmetrics/Detectron2/PointNet2 sources.
The worker's `PYTHONPATH` resolves these staged sources. SSH and rsync share a
dedicated multiplexed connection to limit authentication handshakes.
The launcher sets the worker's working directory to the staged repository so
relative augmentation configuration resolves correctly.
It refuses to overwrite active workers or workers that have trained.
Bootstrap checks all data/weight content, package versions and library sources.
The actual two-GPU native DDP preflight runs before each arm's first training.
A successful SSH/GPU probe alone does not prove numerical/DDP readiness.

`start` returns dispatch receipts immediately. Each worker holds an OS advisory
lock; repeated dispatch reuses the running/completed job. Training resumes only
from committed optimizer checkpoints, keeping U29700 and global batch32.
Each charged subprocess has a budget timeout and terminates its process group
on failure. An optimizer-boundary stop saves the latest checkpoint before its
local quota is exhausted.
If the worker supervisor is forcibly killed, its remaining `ACTIVE_PROCESS.json`
blocks new charged processes. Inspect its recorded PID and reconcile the cost
ledger before resuming; orphaned GPU work is never silently replaced.

`watch` collects snapshots and official metric JSON evidence, verifies their
SHA and identities, deduplicates cost events, then locks CAL after all workers
are terminal. Explicitly failed arms are listed as excluded. It launches SEL
only for selected completed arms. Module comparisons use E1 at the module's
selected step; seed45 gains remain unconfirmed until the separate confirmation
stages. Seed46, formal confirmation and profiling are not run by this scheduler.
Partial SEL-dispatch failures are recorded per node and retried with the same
CAL lock, including while other selected workers are already evaluating.
It retries a busy controller lock on its next poll. The deployment leaves this
monitor running independently of the interactive terminal; its PID and log are
`cluster/MONITOR_PID.json` and `cluster/monitor.log` in the runtime root.
It runs an exact dedicated code copy under `cluster/controller-code`; its live
report is `cluster/controller-code/artifacts/native_long_cluster_v1/REPORT.md`.
The Git artifact report is a publication snapshot; live polling updates runtime
files without modifying that snapshot.
`cluster/STATUS.json` distinguishes observed optimizer updates from updates
committed to a restart checkpoint; the first regular checkpoint is update990.

For a one-shot refresh or cached report:

```bash
python -m scripts.native_long_cluster collect --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign report --root "$NATIVE_LONG_ROOT"
```

Controller receipts/evidence live under `cluster/` in the runtime root.
The published development report is `artifacts/native_long_cluster_v1/REPORT.md`.
Historical `artifacts/native_long_retrain_v1` records retain the old budget state.
The deployment verification records actual readiness, DDP preflight and running
progress separately from completed scientific results.
