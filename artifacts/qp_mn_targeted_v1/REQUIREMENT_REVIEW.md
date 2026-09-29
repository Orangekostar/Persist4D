# Primary requirement review

Scope: the complete `ReScene_QP_MN_Codex_FINAL.md` supplied in the execution
package, not a reduced checklist inferred from passing tests. Reviewed against
the actual repaired base, current new source files, raw runtime checkpoints,
official statistics, final tables and real CLI execution. The primary agent
owns this review; no scientific or final-review decision was delegated.

Scientific status: `COMPLETE_WITH_PROVISIONAL_CONTROL_ONLY`. This review accepts
negative results and the empty confirmed list. Publication is a separate gate:
the last row below becomes verified only through the external post-push receipt
and final audit. This document does not claim a remote operation happened before
it happened.

| Requirement | Current evidence and disposition |
|---|---|
| §0: exactly B0/Q2-F/Q-A0/Q-P/M-C/M-N | PASS. Enum and adapter reject combinations; all six rows are present in `evaluation/MAIN_TABLE.csv`. Q-P/M-N fail against B0; the control's gain does not replace this comparison. |
| §0 exclusions and default | PASS. No Q-R/M-S/M-L model implementation, combined head, parent finetuning, new queries/resolution/association, T3–T5 or formal populations. `NEXT_STAGE_DECISION.json` explicitly has no execution authorization; B0 remains default. |
| §1.1–1.2 repaired base and history | PASS. Full base `465f37a46972e81584c1563bba57e1c05c911c1a`; its actual parent is `4bf00902c9428795f7f47547bf462540aa304cc6`. Required repaired reports were read and their hashes are bound in `BASE_BINDING.json`. No fallback to the old revision. |
| §1.2 workspace and user changes | PASS. Isolated branch/worktree rooted at the repaired commit; only explicit task paths enter publication. Main checkout's unrelated changes and old tags are preserved. No reset/clean/force push/main merge. |
| §1.2 existing failure | PASS, excluded with evidence. Repaired `TEST_NOTES.md` identifies the unchanged repo-reference symlink serialization test and historical reproduction. It does not affect this model/label/metric chain. No historical asset recreation. |
| §1.3 fixed local repair semantics | PASS. Real score/column-invariance and partial-ignore cases; Q `quality.valid` is separate from frozen `quality.geometry_valid`. `CODE_BINDINGS.md`, `REPAIRED_CONTRACT.json`, tests and actual record checks bind the real symbols. Old 72-state evidence remains unchanged; it was not rerun to inflate counts. |
| §2 narrow integration and enums | PASS with organizational deviation. Small task-specific source modules separate binding/training/evaluation/diagnostics/profile/delivery; more filenames than the recommendation, but no new database, task DSL, general framework or altered historical pipeline. Borrowed materializer/metric calls receive explicit context. |
| §3.1 parent assets and dimensions | PASS. R1 bytes754813672 and SHA629ff762…2368dd; Concerto SHA845ec7de…9de07 checked. Actual F128/q128/classes18/probabilities19. Parent native session is eval/frozen; head training does not load R1 into the optimizer. |
| §3.2 population, H1/H2 and exposure | PASS. Actual TRAIN64/188/199, CAL4/23/23, SEL4/24/24 in `BASE_BINDING.json`. H1 reads a deduplicated single scan at time0; H2 reads exactly2 scans. Recorded file reads/targets and 94-input replay confirm the declared condition. Historical SEL exposure is explicitly stated. |
| §3.3 common TRAIN and NOAUG | PASS. All heads share prediction identity229cc0e8…08839f and label identity70dced41…d27328a1. Evaluation identity5bfb3df3…51e63 is no-training-augmentation. Dataset mode and raw label format were not changed to test. No optional augmented comparison was used for selection. |
| §3.4 dependency identities | PASS. Prediction, labels, training and result identities are distinct. Content hashes, head bytes, current sources, label/plan/schedule, apply/materializer and official stmetrics closure are checked. Label change rejects optimizer resume; changed apply/metric changes the result identity. Publication state does not bypass `run --resume` computation checks. |
| §4 actual adapter fields | PASS. `CODE_BINDINGS.md` and `REPAIRED_CONTRACT.json` bind actual scores/masks/lineage, F/q/z/h, stage and inverse mappings, quality fields, assignment and weighted segment targets. No guessed fallback field or expanded-geometry assignment. |
| §4.1 Q objective | PASS. Corrected Q2 scalar temporal labels with binary valid weights and a whole-batch weighted mean. Real Q2 bridge loss and every parameter gradient match the legacy corrected formula on the specified batches. |
| §4.2–4.3 M assignment and invariants | PASS. Frozen-v1 one-to-one assignment is shared by M-C/M-N, distinct from Q validity. Same GT entity across observed stages, absent stage supervised empty, ignore points excluded by weights. No GT-derived inference gate. |
| §5.1–5.3 Q parameterization | PASS. Unchanged quality_inputs/trunk; Q-A0/Q-P identical zero-output initialization; only parent-score addition differs. No sigmoid/clamp for either new head; finite real scores accepted, no correction L2 added. Both gradient checks recorded. |
| §5.4 Q slot and candidate lineage | PASS. Existing retained candidates remain; only scores change. Main evaluation checks mask/class/query/class lineage equality per input. Multiple classes on one query retain independent retained positions. Native Q-P0 equals B0; Q-A00 gives zero scores, not B0. |
| §6.1 M network and materialization | PASS. Both reuse the original shared-h MaskHead and ±2 tanh residual on all segments. Only masks change; original scores/classes/candidates remain. Real zero-residual materialization agrees with B0 without a zero-step bypass. |
| §6.2–6.3 loss order and denominator | PASS. Per-stage weighted BCE+Dice and smoothing1; only observed stages with sampled valid points count. Original equal-record mean of all-slot/segment delta² remains at0.01. M-C divides shape by16; M-N by actual Npos, connected zero if none. Real M-C/legacy-M1 loss and all parameter gradients agree. |
| §6.4 training and gradient diagnostics | PASS. Fifteen100-update SUM/COUNT windows per seed45 arm; both M arms2223/24000 actual supervised positions,0 matched-without-points. Clip counts M-C0/M-N164. Read-only component gradients use actual batches2/502/1500 and both formulas at the same parameters; no new updates. |
| §7.1 fixed recipe | PASS. Config and checkpoint metadata bind seed45, AdamW1e-3/(.9,.999)/1e-8/1e-4, horizon1500/warmup75/cosine floor0.1, FP32, clip1,3H2+1H1×4, ≤2048 segments/stage. Four new1500 trajectories; paired init and sampling identities match. |
| §7.2 control reuse | PASS. Q2-F four states0/500/1000/1500, optimizer metadata and source identities verified;1500 reused,0 new. M-C exact trajectory provenance was insufficient, so1500 fresh updates were used despite local loss equivalence. `TRAINING_REUSE_DECISION.json` records the reason. |
| §7.3 resume and actual gradients | PASS. Raw runtime checkpoints contain optimizer/RNG/plan/source/input/label/update/horizon; last saved every100. Split-resume regression matches uninterrupted training and rejects changed labels. First two actual supervised updates recorded; zero first trunk gradient is not mistaken for a broken chain. |
| D-Q1 / D-Q2 | PASS. Predeclared hash8-reference TRAIN panel16H2/22H1 and full CAL23+23; all four score sources and all required checkpoints, MSE/quantiles/class rankings/destroyed orders/partial-ignore changes/duplicates retained. Full-set GT-score diagnostics are separately labeled and independently recomputed; unknown defaults predeclared. |
| Conditional Q memorization | NOT_APPLICABLE, correctly evaluated. Both Q-A0/Q-P TRAIN MSE beat the TRAIN-mean constant. No Q memorization head was trained; not called a failed method. |
| D-M1 / D-M2 | PASS. Assigned-candidate original/new/loose IoU, coverage, error/correct counts, strict >2 error segments/points, boundary counts, saturation, residual distributions, sign flips and materialized changes. L≤original≤U is asserted. Ordinary IoU and full-set official assisted AP are separate; neither is claimed as an official AP upper bound. |
| D-M3 | PASS, triggered. M-N selected CAL gain−0.0015393198 and123 eligible TRAIN candidates; exactly the fixed first16, seed145, one500-step shared head, checkpoints0/250/500. Loss0.2406157→0.1738358;15/16 materialized IoUs improve. Probe is never in CAL/SEL selection. |
| Next-stage decisions | PASS. Four evidence-linked recommendations in `diagnostics/NEXT_STAGE_DECISION.json`, with explicit limitations and no expected gains or automatic execution. |
| §9.1 lock and fixed endpoints | PASS. All five1500 trajectories precede `CAL_LOCK.json`; chosen steps1500/1000/500/500/1500. Only selected SEL and prescribed1500 endpoints; selected=1500 deduplicated. All own-point and same-step mechanism comparisons are present. |
| §9.2 units and grades | PASS. Machine AP units[0,1], report%/pp; no_gain/negative/T1 protection/target thresholds checked. B0 is explicit, unavailable seed46 values are null. Reference consistency is separate from pooled gain. |
| §9.3 fixed second seed | PASS. Only M-C is a protected positive engineering candidate. `REPLICATION_PLAN.json` was fixed before seed46 training;500 updates at horizon1500, fixed CAL/SEL500. No seed46 epoch selection or unnecessary1500 continuation. New-method paired triggers were not met. |
| §9.4 final development list | PASS. `SHORTLIST.json`: confirmed empty, provisional M-C, four negative heads. M-C two positive seeds do not override2/4 reference consistency. No formal-test or author-ReScene superiority claim. |
| §10.1 costs and limits | PASS. Prior80.5051995672 GPUh reconciles V2, old campaign and repair ledger by event id, excluding the inherited subtotal. New0.2298391830 GPUh includes failures and complete reservations; below16 new/192 cumulative caps. One idle A40 at a time; CPU subprocesses2 threads, total≤8. Reused parent cache adds0 bytes; runtime47MiB, public artifacts approximately36MiB before final docs. |
| §10.2 relevant verification | PASS.47 unit/integration regressions,4 separately executed real-record tests,5 inherited dependency-identity tests; delivery/CLI tests rechecked after final local edits. SciPy deprecation warning only. Targeted Ruff and diff check pass. The real tests skipped in the first batch are explicitly accounted for by the separate4-pass run, not counted twice. |
| §11 commands and recovery actually exercised | PASS for this execution. `run --config ... --base-commit 465f37a --resume --through report` exited0; `status` inspected live bound receipts. No repeated training or probes; report outputs regenerated. `--root`/`--artifact-root` are explicit; known old roots protected. All Q/M contracts were available, so branch-specific missing-contract continuation was not invoked or claimed tested. Source-changing optimizer recovery fails closed rather than pretending an old trajectory is fresh. |
| §12.1 artifact scope | PASS before final Git metadata. Instruction/base/bindings/config/input/state/log/budget, all head trajectories and inference states, all CAL/SEL/endpoint/ref/paired tables, shortlist, diagnostics/decision, metric zip, profile, report/review present. HANDOFF and self-excluding manifest are generated as delivery B. No raw parent weights/data/pointwise GT. |
| §12.2 complete tables and interpretation | PASS. Six-arm main table includes updates/selection/CAL/SEL/T1/deltas/control/seed46/references/latency/VRAM/grade; report distinguishes retained-score mechanism, gradient increase, coverage, clipping, fitting, generalization and residual/metric limitations. |
| §12.3 actual profile and reload | PASS with explicit runtime condition.96 actual full-parent forwards,24 warmup/72 timed, four hash-fixed SEL pairs; disk IO separated. Five actual selected-bundle reloads, native slot/zero checks,94-input bitwise setup replay. Setup130.0066s is separately charged. Cold arbitrary-order parent reproducibility remains a disclosed limitation, not silently relaxed equality. |
| §12.4 small assets and publication preparation | PASS locally.22 inference checkpoints,5 selected bundles,4,188,126-byte zipped sufficient states; each binary≤20MiB and total≤50MiB. Ancestor465 is included. Scoped A→handoff/manifest B and immutable-new-tag publication code reviewed; no credential harvesting or Release requirement for small assets. |
| §12.4 actual remote push/tag/blob availability; §13 final gate | MUST VERIFY AFTER PUSH. External `publication/PUBLICATION_RECEIPT.json` must contain actual full A/B/tag, matching remote branch/tag SHA and byte/SHA-verified Git blobs for every artifact. Release is NOT_REQUIRED only if all required assets are in Git. The active goal cannot be completed until that check passes. |
| §14 source and scientific evidence boundary | PASS. The package's cited literature is motivation only; no imported expected gains or claims of original-paper-optimal hyperparameters. This execution's measurements are the evidentiary basis. |

## Review findings resolved or bounded

The metric-state verifier's serialized-path type bug was reproduced and fixed;
54 formal states and12 assisted states now recompute independently. Adding B0 to
the CSV exposed unequal row field sets; this was fixed and the actual six-row
table regenerated. Seed46 resume now verifies completed checkpoint identities
before skipping GPU training. Profile parent/cache inequality was investigated
without changing evaluation data or loosening equality: original-order replay
produces exact outputs, and the additional initialization cost plus unresolved
cold-order limitation are disclosed in `resources/NATIVE_RUNTIME_NOTE.md`.

The current scientific conclusion is justified for the shared bound cache and
declared small development population. It does not establish arbitrary-order
bitwise native deployment, statistical significance, unseen-test generalization
or that any unexecuted next-stage proposal will succeed.
