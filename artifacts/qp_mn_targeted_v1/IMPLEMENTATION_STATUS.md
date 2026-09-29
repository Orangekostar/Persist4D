# Implementation checkpoint — not final completion

Implemented: repaired asset/source binding, explicit output roots, Q-A0/Q-P,
M-C/M-N losses, prediction isolation, fixed-plan head training and dependency-bound
resume, corrected Q2-F completed-checkpoint reuse, NOAUG official evaluation,
global CAL lock, selected/fixed-endpoint SEL guards.

Actual seed45 training: Q-A0/Q-P/M-C/M-N each reached1500; Q2-F reused1500.
All new arms saved0/500/1000/1500 plus last.pt with optimizer/RNG/source metadata.
Paired initial hashes and first two actually supervised update checks were inspected.
GPU reservation:145.144356s =0.0403178768GPUh, including all four new trajectories.

Validation:35 tests passed across new heads/adapter/binding/training/real-record
contracts plus inherited heads/data;8 additional selection/classification tests
passed. Ruff and diff checks passed. Existing SciPy deprecation warning only.
This is not proof of full experiment completion.

Live main evaluation was launched as PID1546253 (exec session27110), CPU-only,
with two numerical threads. Poll this exact handle/process before considering
restart. It runs evaluate_cal then evaluate_sel sequentially and records its
reservation terminal state under the runtime reservations directory.
Known first results: NOAUG B0 CAL/T1=.521016538143158,T2=.365048885345459;
Q2-F CAL1500/T1=.3242713510990143,T2=.20947961509227753.
Do not infer selected steps or SEL results until the actual all-arm lock exists.

Outstanding mandatory work:

1. Finish and verify the running CAL/SEL evaluation; independently recompute its
   actual metric states and produce by-reference and paired delta tables.
2. Seed46 fixed-step paired replications for every eligible Q-P/M-N positive;
   optionally positive controls as engineering candidates. No new selection.
3. D-Q1/D-Q2 and D-M1/D-M2 on declared TRAIN panel/full CAL, three read-only
   M gradient batches, conditional Q memorization/M fit only if triggers hold.
   See diagnostics/DIAGNOSTIC_CONTRACT.md for defaults fixed before these results.
4. Real NOAUG parent+head profile on four hash-fixed SEL pairs,1warmup/3timed,
   and selected portable head save/reload equality.
5. Thin scripts.qp_mn_campaign run/status/report/publish CLI with explicit roots,
   complete lifecycle state/log, dependency-aware resume and budget reconciliation.
6. Final instruction-by-instruction requirement audit, report/handoff, all positive
   and negative small-head packages, zipped official metric sufficient states,
   artifact manifest and publication verification. Actual branch/tag push remains
   required and authorized; no token request, no main merge or force push.

Do not mark the active goal complete from this checkpoint. No publication yet.
