# Source Map

| Logical input | Runtime binding | Git policy |
| --- | --- | --- |
| R1 checkpoint | `R1_CHECKPOINT` | external, read-only |
| Concerto pretrained | `CONCERTO_PRETRAINED` | external, read-only |
| 3RScan metadata | `RIO_METADATA` | external, read-only |
| processed datasets | `PERSIST4D_DATA_ROOT` | external, read-only |
| R1 cache | `R1_CACHE_ROOT` | external, read-only |
| new checkpoints/cache | `ALLT_EXTERNAL_ROOT` | external, writable |
| Protocol-B manifest | `repo:artifacts/P6A/protocol_b_manifest.json` | tracked |
