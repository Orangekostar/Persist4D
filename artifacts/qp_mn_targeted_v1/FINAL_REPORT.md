# Q-P / M-N targeted experiment

## Decision

Keep B0. Neither requested intervention beats the common NOAUG baseline. Q-P
substantially outperforms its degraded direct-scoring control, but remains below
B0. M-N increases the actual shape gradient and clipping frequency, but loses
to both B0 and M-C. The engineering control M-C is positive in two fixed seeds;
only 2/4 seed45 references improve, so it is provisional, not
`DEVELOPMENT_REPLICATED`. The confirmed list is empty.

Scientific result: `COMPLETE_WITH_PROVISIONAL_CONTROL_ONLY`. Git delivery is a
separate, externally verified state; see `HANDOFF.md` and the runtime
`publication/PUBLICATION_RECEIPT.json` after publication. No default replacement,
Q+M combination, new architecture, backbone training or formal test evaluation
was performed. Historical R1 and development/SEL exposure are acknowledged;
these populations are not described as untouched.

## Controlled experiment and main results

Base: `465f37a46972e81584c1563bba57e1c05c911c1a`. Actual asset hashes, source
bindings and counts are in `BASE_BINDING.json`, `CODE_BINDINGS.md` and
`INPUT_MANIFEST.json`. TRAIN: 64 references, 188 H2 and 199 unique independent H1
inputs. CAL: 4 references and 23 inputs per horizon. SEL: 4 references and 24
inputs per horizon. TRAIN retains the same fixed historical input condition for
all arms; CAL/SEL use the bound native no-training-augmentation condition.

Q-A0/Q-P and M-C/M-N each completed the prescribed seed45 1500-update trajectory.
Q2-F reused exactly 1500 verified corrected-Q2 updates. M-C was freshly trained
because the historical checkpoint's source digest could not be reconciled to a
complete exact replay identity; loss/gradient equivalence alone was insufficient.
The four new trajectories share the fixed sampling plan, paired initialization,
AdamW settings, warmup75/cosine1500 schedule, FP32 and clip1. Each batch contains
three H2 inputs and one independent H1, four candidates per input. The fixed
seed46 M-C500 run adds 500 updates; its schedule horizon remains 1500. Thus the
main experiment adds 6500 updates and reuses 1500, excluding the one 500-step
diagnostic probe. No old optimizer was continued under a new loss.

All five trajectories were complete before a single all-arm CAL lock. Selection
uses T2, then true T1, then earlier step with 1e-6 tolerance. The table reports
the selected seed45 points; AP is displayed in percent, changes in percentage
points. Machine CSV/JSON values remain in [0,1] AP units: 0.001 = 0.1 pp.

| Arm | New/reused updates, seed45 | Selected | CAL T2 % | SEL T2 % | ΔT2 pp | SEL T1 % | ΔT1 pp | Positive refs | Fixed seed46 ΔT2 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0/0 | 0 | 36.5049 | 70.2952 | 0 | 78.5034 | 0 | — | — |
| Q2-F | 0/1500 | 1500 | 20.9480 | 62.1325 | -8.1627 | 63.8239 | -14.6795 | 0/4 | — |
| Q-A0 | 1500/0 | 1000 | 20.9787 | 58.6231 | -11.6721 | 62.1102 | -16.3933 | 0/4 | — |
| Q-P | 1500/0 | 500 | 33.4071 | 66.9407 | -3.3545 | 76.3560 | -2.1474 | 1/4 | — |
| M-C | 1500/0 | 500 | 36.7198 | 70.5010 | +0.2058 | 78.7079 | +0.2045 | 2/4 | +0.6126 |
| M-N | 1500/0 | 1500 | 36.3510 | 69.3368 | -0.9583 | 78.0331 | -0.4703 | 1/4 | — |

M-C seed46 T1 is 78.6678%, a +0.1644 pp change. Both seeds pass the T1 guard, but
the reference-consistency gate fails; the larger second-seed gain does not
replace seed45 or turn this into a significance claim. No Q-P/M-N seed46 trigger
was met. Missing second-seed values remain null, not zero.

`evaluation/MAIN_TABLE.csv` includes CAL T1 and all machine-precision values,
mechanism-control deltas, evidence levels and resources. `BY_REFERENCE.csv`
preserves both horizon denominators. `SEL_ENDPOINT_1500.csv` is a predeclared
diagnostic, never a new selection opportunity. Selected=1500 points are reused.

## Mechanism comparisons

At each arm's CAL-selected point, Q-P minus Q-A0 yields +8.3176 pp SEL T2, and
Q-P minus Q2-F yields +4.8082 pp. At the common SEL1500 endpoint these differences
are +3.6057 pp and +0.0279 pp, respectively. The direct Q-P/Q-A0 pair changes only
the parent-score addition; Q-A0/Q2-F changes the entire scoring parameterization
and initialization recipe, so its components are not given separate causal credit.
Q-P's engineering result is still -3.3545 pp against B0.

M-N minus M-C is -1.1642 pp at their selected points and -1.3029 pp at the common
SEL1500 endpoint. CAL same-step comparisons at 500/1000/1500 are all negative.
All own-point and same-step T1/T2 differences are in `PAIRED_DELTAS.csv`.

## What the diagnostics explain

The predeclared TRAIN panel contains 8 references, 16 H2 and 22 unique H1 inputs;
the CAL diagnostic uses all 23+23 inputs. No SEL GT drives these diagnostics.

For Q-P500, weighted quality MSE is 0.0737823 on TRAIN and 0.0916349 on CAL;
the corresponding constant baselines are 0.1422867 and 0.1434855. Q-A0's TRAIN
MSE is also below the constant, so Q memorization is `NOT_APPLICABLE`. Despite
lower CAL inversions (4121 versus B0's 5915), Q-P destroys 2317 originally correct
within-input/class orderings. Its CAL partial-ignore candidates move up/down in
1535/446 cases. Scores are finite real values, including values below zero and
above one; there is no clamp, new filter, NMS or top-k.

The full-candidate GT-quality auxiliary score gives CAL H2 AP 0.2965082 versus
B0 0.3650489. Duplicate candidates remain; unknown/ambiguous relationships keep
their predeclared parent score. This is a GT-assisted diagnosis, not an
implementable method or an AP upper bound. Lower subset MSE/ranking errors do
not establish a compatible full-set AP objective. See `QUALITY_RANKING.csv`,
`QUALITY_CLASS_RANKING.csv`, `DUPLICATE_POSITIONS.csv` and
`GT_ASSISTED_OFFICIAL.csv`.

Both M trajectories use exactly 2223 supervised positions out of 24000 sampled
positions, with 11504 unmatched and 10273 unknown positions; none were resampled
to manufacture positives. M-N changes only the shape denominator. On the three
predeclared actual-positive batches (2, 502, 1500), the shape-gradient norm at
the same parameters is scaled by 16, 8 and 16, while the regularizer gradient is
unchanged. This is a gradient effect, not merely a larger printed loss. M-C clips
0 updates; M-N clips 164/1500. The read-only gradient checks perform no updates.

On the 385 assigned CAL candidate positions, the number with ordinary IoU >0.75
falls from 261 to 255 for M-C and 253 for M-N. M-C repairs/breaks 2/8 threshold
crossings; M-N repairs/breaks 4/12. These candidate-position diagnostics are not
unique-object counts or official matched-trajectory recall. Raw sign flips and
full-eval changed points are reported separately: M-C 1813/42806, M-N 2635/65839.

On the fixed TRAIN panel, 185 of 315 assigned candidates never receive sampled
shape supervision. Of 186273 raw error point positions, 124594 lie in strict
|z|>2 error segments, which a bounded ±2 residual cannot reverse in real
arithmetic. Equality-at-boundary and FP32 saturation are counted separately.
Loose materialized masks improve assigned-candidate mean ordinary IoU by
0.0607572 on TRAIN and 0.0661261 on CAL. Full-set GT-loose-mask CAL H2 AP is
0.4363574. This deliberately relaxes shared-segment/feature constraints and is
not an official AP upper bound or a forecast for a learned model.

M-N's CAL gain is nonpositive and 123 TRAIN candidates meet the predeclared
fit-probe criteria. A new seed145 head fits the fixed first 16 for exactly 500
steps, with checkpoints0/250/500. Loss decreases 0.2406157→0.1738358; mean raw
expanded IoU rises 0.6931365→0.7518916 and materialized IoU 0.7053966→0.7658187.
Fifteen candidates improve and one worsens. This confirms some local fitting
ability while leaving a gap; it does not establish generalization or isolate
capacity from optimization. Only one shared M probe was trained because its
all-positive denominator makes the two formulas identical.

`diagnostics/NEXT_STAGE_DECISION.json` gives bounded, evidence-linked suggestions
for Q-R, M-S, M-L and independent M-C confirmation. None is executed or assigned
an expected gain. The known label/partial-ignore repairs belong to the repaired
base; parent-score addition and shape-only normalization are this round's active
method changes. Metric algorithms, assignment and the ±2 residual were not changed.

## Deployment and reproducibility

Four hash-fixed NOAUG SEL pairs on the same A40 were evaluated for B0 and each
selected single module: 24 warmups and 72 timed full forwards. Disk IO is separate;
network, descriptor, head and materialization costs are included. Mean end-to-end
seconds are B0 0.6287, Q2-F 0.8815, Q-A0 0.8626, Q-P 0.8597, M-C 0.9436 and M-N
0.9634. These are observed sequential small-sample timings, not statistically
established relative speedups. Allocated peaks are approximately 1.56 GiB; exact
allocated/reserved VRAM, CPU RSS and timing components are in `resources/PROFILE.csv`.

The inherited parent exhibits process-history-sensitive native outputs. An
additional cold-cache parity check failed; no equality tolerance was loosened.
Replaying all 94 inputs in their original export order reproduced every cached
output bitwise. That 130.0066-second setup was charged and excluded from the
steady-state timing table. Five actual bundle reloads and native zero-step slot
checks passed after this binding. Arbitrary cold input-order reproducibility is
not established. `resources/NATIVE_RUNTIME_NOTE.md` states this limitation and
the controlled profiling condition; startup replay is not free deployment.

Twenty-two inference-only checkpoints preserve all positive and negative arms
and the fixed second seed; five selected deployment bundles include mode,
dimensions/input schema, parent/label identities and step. Optimizer states
remain private for exact resume. The 4,188,126-byte official statistics zip
contains 54 formal pooled states and 12 diagnostic states, all independently
recomputed. Raw point masks/GT, datasets and R1/Concerto weights are not included.

Historical reconciled cost is 80.5051995672 GPUh. This campaign currently adds
0.2298391830 GPUh, including failures, runtime investigations, full profile setup,
and CUDA-context time during second-seed CPU scoring; cumulative cost is
80.7350387502 GPUh. The new cap is 16 GPUh and cumulative cap 192 GPUh. CPU-only
evaluation holds no GPU reservation. No new parent cache was exported or stored;
existing assets and environment are reused. The event-id ledger is authoritative.

## Verification and handoff

The real CLI `run --resume --through report` validates current dependencies;
publication receipts never bypass computational validation. `status`, `report`
and `publish` are separate commands; roots are explicit. Core source/input/head
identity changes cannot resume an old optimizer as a new trajectory. Protected
historical output roots are rejected.

Relevant regression results are retained under `validation/`, including a
separate real-record run with the repaired-root environment set. The initially
skipped real-record tests are not counted as passes. Existing unrelated
repo-reference symlink serialization failure remains documented in the repaired
base; no unrelated historical checkpoint recreation was attempted.

`REQUIREMENT_REVIEW.md` records the primary agent's instruction-by-instruction
review. `HANDOFF.md` binds experiment A; delivery B contains the handoff and
self-excluding artifact manifest. Actual remote branch/tag and Git-blob byte
verification are recorded externally after push. No main merge or force push is
part of delivery.
