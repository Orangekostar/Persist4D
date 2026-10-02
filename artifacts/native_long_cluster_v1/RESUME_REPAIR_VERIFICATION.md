# Native Long Resume Repair Verification

The four workers stopped at update2970 when their resumed first microbatch
triggered a checkpoint save before completing accumulation. `UpdateBoundary`
now initializes its last step from Lightning's restored `global_step`; strict
optimizer-boundary validation remains enabled.

All original2970 checkpoints, CAL evidence, code, specs and ledgers are archived
under remote `repairs/3c70ae4378557be51af09617efb359e512bf49da3b5d87d928393576cb0cc787/`.
Controller history is under the corresponding runtime `cluster/repairs/` folder.
The original official CAL evidence also has a verified controller archive copy.
Archived JSON retains its original absolute paths; use the recorded prefix
mappings in the verification JSON to resolve those paths to preserved files.
The explicit amendment preserves job quotas,1800 GPUh lifetime cap, prior costs,
world2, accumulation16, batch32, U29700 and common seed45 initialization. The
metadata migration changes only code identity; numerical, loop, RNG and config
content digests match before and after migration for every checkpoint.

Actual two-A40 verification passed on all four hosts: update2970->2972, next
global draw95104, rank-local RNG states2 and full scheduler29700. Each saved a
new2972 checkpoint. All four supervisors then recomputed the complete CAL2970
population and resumed their train-to5940 segments.

The user service `persist4d-native-long-cluster.service` supervises the reviewed
controller code copy. An injected SIGKILL changed main PID2108591->2109412 after
the configured15-second restart. Its PID receipt updates on every start. User
lingering is enabled. Separate cold SSH without an agent or multiplexed
connection passed. Current status is refreshed in the runtime `cluster/STATUS.json`.
The previous detached monitor's disappearance had no recoverable exit log.

Validation: broad CPU suite213 passed,2 CUDA-dependent skips,178.67s; final
resume/cluster/repair suite45 passed,16.01s; Ruff and service syntax checks pass.
Two deployment checks caught OmegaConf metadata support and an unsupported unit
CLI flag before closure; both were corrected and verified on the real runtime.

CAL reproducibility: E0/E1/E2 match the archived metrics exactly. E3's largest
absolute difference is0.004661515355110168 atT4 (approximately0.47 percentage
points), despite the preserved checkpoint payload and unchanged model/data/config
sources. Its S_long delta is-0.00052472949028015. The protocol retains its original
`warn_only=True` deterministic setting, and logs contain CUDA nondeterminism
warnings. The cause of the E3 evaluation difference is not established; repeat
evaluation remains necessary before a reproducibility claim.

See `RESUME_REPAIR_VERIFICATION.json` for bound code/job/checkpoint identities,
costs, fresh observed versus committed update counts and original/recomputed CAL
metrics. Full29700 training, locked SEL and seed46 confirmation are still pending.
