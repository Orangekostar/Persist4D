# Protocol correction before CAL selection

The initial M0/M1/M2 seed45 runs incorrectly restricted the GT-free residual
regularizer to candidates eligible for supervised quality labels. The corrected
loss regularizes every finite native-input candidate, including ignored slots;
only the supervised mask term retains its supervision eligibility mask.
The regression test demonstrates nonzero regularization with all supervision
flags false. Q losses, prediction export, descriptors, labels and assignments
are unchanged; their original cache and Q checkpoints remain valid.

The original data/loss source is preserved in
`short_module_data.before_regularization_fix.py.txt`. Original M training and
CAL evidence are retained externally in `attempts/before_regularization_fix/`.
Public original M logs are retained here under `before_regularization_fix/`.
Replacement M runs use the identical seed45 initialization and sampling plan.
Their metadata additionally binds the loss source SHA256.

CAL process 829941 was deliberately interrupted before any CAL_LOCK or head SEL
evaluation. Its measured 916.2314423000207 seconds (0.25450873397222795 GPU-hours)
remain charged in BUDGET_LEDGER; the absent terminal status on that legacy event
means INTERRUPTED, not an uncharged or successful run. No charge is duplicated.
All original training costs remain charged as well.
