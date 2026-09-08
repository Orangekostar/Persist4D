# Source Map

| Logical input | Runtime binding | Git policy |
| --- | --- | --- |
| R1 checkpoint | `R1_CHECKPOINT` | external, read-only |
| Concerto pretrained | `CONCERTO_PRETRAINED` | external, read-only |
| 3RScan metadata | `RIO_METADATA` | external, read-only |
| processed datasets | `PERSIST4D_DATA_ROOT` | external, read-only |
| R1 cache | `R1_CACHE_ROOT` | external, read-only |
| new checkpoints | `$ALLT_EXTERNAL_ROOT/training/formal/<variant>` | external, writable |
| development cache | `$ALLT_EXTERNAL_ROOT/evaluation_cache/development_train_holdout_47_masters_canonical/<model>/<checkpoint_sha256>` | external, writable |
| Protocol-B cache | `$ALLT_EXTERNAL_ROOT/evaluation_cache/protocol_b_43_masters_3_orders/<model>/<checkpoint_sha256>` | external, writable |
| Protocol-B manifest | `repo:artifacts/P6A/protocol_b_manifest.json` | tracked |

The current `ALLT_EXTERNAL_ROOT` binding is
`/mnt/shared/ww/persist4d-allt-task-superiority-v1`. On the execution nodes,
`/mnt/shared` is the mounted NFS export
`192.168.100.102:/mnt/data/shared`. The tracked checkpoint/cache manifests are
authoritative for the logical reference, SHA256, byte count, and reconstruction
inputs; no credential or large binary is stored in Git.

The actual tracked training summaries use
`repo:artifacts/allt_task_superiority_v1/training/formal/<variant>`. The shorter
`training/<variant>` form in the experiment prompt is a layout template, not a
different run namespace.
