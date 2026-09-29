# Native execution history and profiling condition

The first cold profile failed an additional exact-cache comparison. This was
not a head reload failure: all head reload fields compared exactly before the
cache check. On the fixed first SEL profile input, three fresh native forwards
agreed exactly with each other but differed from the bound cached parent.
Coordinates and all point/segment mappings were identical. Enforcing the
recorded CUBLAS environment alone did not eliminate the difference.

Controlled experiments are retained in `NATIVE_REPLAY_AUDIT*.json` and the GPU
ledger. Replaying the original NOAUG export prefix in a fresh process reproduced
the target exactly. Starting with the target before that prefix did not. This
establishes process-history sensitivity in the inherited native execution
stack. The exact underlying library/kernel cause is not established; it is not
attributed to a learned head, changed weights, or a label repair.

For the completed profile, the process first replayed the immutable original
94-input export order. Every output field on all 94 inputs matched the cache
bitwise (`PROFILE_SETUP_REPLAY.json`). This setup took 130.006604 seconds and is
separate from the prescribed 24 warmup and 72 timed forwards. Its complete GPU
reservation, the failed first profile, and all diagnostic attempts are charged.
The subsequent five package reloads and Q-P/M-C/M-N zero-step native equality
checks passed; Q-A0 correctly produced zero scores and unchanged geometry.

`PROFILE.csv` therefore measures full-parent steady-state deployment under this
explicitly reproduced native condition, not cold-start independent-input
bitwise reproducibility. Initialization/replay cost must be included for that
use case. The formal CAL/SEL results all share the same immutable original
cache; no metric or checkpoint was changed to resolve this profiling issue.
Deployment on arbitrary fresh-process input orders remains a documented
parent-runtime reproducibility limitation, not a proven generalization gain.
