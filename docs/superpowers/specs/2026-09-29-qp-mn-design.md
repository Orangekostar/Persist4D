# Q-P / M-N targeted campaign design

Authority: `docs/0929/ReScene_QP_MN_Targeted_Execution/ReScene_QP_MN_Codex_FINAL.md`.
The user authorizes autonomous implementation and publication; no intermediate
design approval is required. Base: `465f37a46972e81584c1563bba57e1c05c911c1a`.

Use a new enum and prediction adapter without changing historical module names.
Reuse the native descriptors, materializer, official metrics and sample planner.
Q-A0 and Q-P share zero-output initialization and differ only in the parent-score
skip. Q2-F retains the corrected sigmoid Q2 recipe. M-C and M-N share the original
M1 head and frozen geometry assignment; only the shape denominator differs.
The regularizer remains the mean of record-wise full residual means.

Training uses the repaired fixed TRAIN cache. Main CAL/SEL use the verified
NOAUG cache and their own B0. Q validity and frozen geometry validity remain
distinct. Reuse corrected Q2 and original M1 only after actual checkpoint,
sample-plan, input and loss/gradient equivalence checks. Otherwise train them.

An explicit campaign context supplies root and artifact paths; no historical
ARTIFACTS global is changed. New head, adapter and CLI modules own new behavior.
Focused training, diagnostic or publication helpers may be separate modules to
keep responsibilities readable. Every reuse/resume decision binds actual source,
input, label, weights and metric specification identities.

All five trajectories reach 1500 before CAL selects among 500/1000/1500.
Lock all selections before new SEL metrics; evaluate selected and fixed endpoint.
Run required seed-46 paired replications only at locked steps. TRAIN/CAL diagnostics
must not use SEL to tune. Profile real parent inference, save/reload all selected
heads, independently recompute official metric evidence, then audit every numbered
requirement. Publish branch and new tag with small head/evidence assets and verify
remote objects. Keep publication receipt external to its referenced commits.

Resource ceiling: two idle A40s, one job per GPU, CPU <=8, RAM <=96 GiB;
new GPU allocation min(16,192-prior actual), reserve >=3 GPUh for completion.
Existing conda/CUDA environment only. No raw parent weights/data/GT publication.
