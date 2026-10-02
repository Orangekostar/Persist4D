# Native Long Cluster

Budget authorization: user request2026-10-01.

Lifetime cap 1800.00 GPUh; accounted 234.183179 GPUh.

| Arm | Host | Committed updates | Status | GPUh allocation |
|---|---|---:|---|---:|
| E0 | 192.168.100.101 | 2972 /29700 | TRAIN_CAL_RUNNING | 405.844937 |
| E1 | 192.168.100.102 | 2972 /29700 | TRAIN_CAL_RUNNING | 426.178363 |
| E2 | 192.168.100.103 | 2972 /29700 | TRAIN_CAL_RUNNING | 436.363024 |
| E3 | 192.168.100.104 | 2972 /29700 | TRAIN_CAL_RUNNING | 442.248283 |

Global batch32, per-rank batch1, world2, accumulation16; all arms use the same seed45 common state and population. No trained gain is claimed until complete locked SEL comparisons are available. Seed46, formal confirmation and profiling are separate future stages. Historical budget-limited results in artifacts/native_long_retrain_v1 remain a snapshot of the previous192 GPUh plan.
