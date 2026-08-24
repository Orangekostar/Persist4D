# P2R Controlled Diagnostics

- Status: `pass`
- Checkpoint: `85ed1aba...131546e`
- Seed: `45`
- Scope: one real supervised T=2 sample; diagnostic, not G2 evidence

| Path | Encoder mode | Objective | Scalar loss | Head grad norm | Matches |
|---|---|---|---:|---:|---:|
| P2R-0 | train | weighted | 10.074492 | 12.699265 | 2 |
| P2R-A | eval | weighted | 9.375830 | 14.181270 | 2 |
| P2R-B | train | raw_sum | 3.226935 | 3.587409 | 2 |

## Microbatch Diagnostic

- Physical objective: `17.345673`
- Accumulated objective: `9.690756`
- Selected-gradient cosine: `0.899034619`
- Selected-gradient max absolute difference: `4.06465`
- Physical-global-32: `hardware-infeasible, not executed`

Differences are controlled diagnostics on one duplicated real sample. They
do not by themselves explain the 6.861-point paper gap and do not authorize
a full candidate without aligned task-metric pilots.
