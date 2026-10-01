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

Before staging, set up each assigned host's Python environment with the original
Torch/Lightning versions and the bound Concerto/Sonata/stmetrics source trees.
The configured `python3` must resolve to that environment. Staging transfers only
the code, fixed encoder, common initialization and SHA-bound data/metadata.
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

For a one-shot refresh or cached report:

```bash
python -m scripts.native_long_cluster collect --root "$NATIVE_LONG_ROOT"
python -m scripts.native_long_campaign report --root "$NATIVE_LONG_ROOT"
```

Controller receipts/evidence live under `cluster/` in the runtime root.
The published development report is `artifacts/native_long_cluster_v1/REPORT.md`.
Historical `artifacts/native_long_retrain_v1` records retain the old budget state.
The implementation task changes the budget and scheduler without starting long
training. Remote preparation must pass before a subsequent start.
