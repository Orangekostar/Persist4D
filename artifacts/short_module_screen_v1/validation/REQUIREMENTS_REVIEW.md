# Primary-agent requirement review

Authority: supplied `ReScene_Round1_Single_Module_Codex_FINAL.md` (§0–15)
and its frozen YAML. This review covers the implementation and actual evidence;
the supplied package's design-only REVIEW is not substituted for experiment QA.

| Requirement | Evidence and disposition |
|---|---|
| §0,2 fixed base, six independent arms, one runtime enum | Dedicated named worktree at base `1ddab2a`; only new task paths. `Module`, `build_head`, `apply_module` reject combinations and accept no GT. No existing parent, decoder, deployment, V1/V2 source or artifact changed. |
| §1 research scope | Q/M ideas are adaptations, not paper reproductions. No claims of official ReScene superiority or failure of whole literature directions. |
| §3 actual native bindings | `CODE_BINDINGS.md`; native source and installed matcher/evaluator hashes in cache `IDENTITY.json`. D=128, Dq=128, C=18, probabilities=19 were inferred during real export. Candidate keys include input/query/class/retained index; multi-class query tests pass. |
| §4 role/exposure/population | `SHORT_POPULATION.json`, `INPUT_MANIFEST.json`; TRAIN64 refs/188 directed pairs/199 unique scans, CAL4/23/23 and SEL4/24/24. TRAIN reference overlap with CAL/SEL/PB/LOCAL/ADDITIONAL is zero. Parent pretraining and prior development exposure remain disclosed. |
| §4 true H1/H2 | 481/481 exported inputs. Every export records exact `np.load` paths, H and frozen/eval parent. NativeSession rejects extra/substituted scans and verifies stage0 for H1. `max_points_per_sample` is absent/null, and bound augmentations do not reorder/drop points, so per-scan local arange IDs refer to original vertices. No cross-scan vertex-ID matching. |
| §4 numerical/cache binding | FP32, seed45, TF32 off, cuBLAS4096:8, two OMP/MKL/OpenBLAS threads. Prediction and target shards separate. 4.6GiB total new cache, below32GiB. Actual CAL soft-output masks/classes/scores exactly matched the original native output. Raw staging uses inherited hardlinks and has no complete byte-immutability claim. |
| §5 diagnostics | Candidate categories retain the full TRAIN/CAL denominator; actual high/low-score overlap distributions, appearance/disappearance, official joint/separate and per-reference counts. These are not AP upper bounds. Selected-head geometry transitions are required before report completion. |
| §6 GT-free descriptors | Reviewed `describe_candidates`: stage-local original positive support; expanded0.10m centroid box or stable top32 fallback; hash32 exploration; equal-segment means. Q uses LN(q), LN(mean_h), raw absolute h difference and unnormalized scalar/one-hot flags, as registered before outcomes. M shares mean_h or uses stage_h only. |
| §7 independent quality supervision | Reviewed installed pairwise matcher boundary, `_valid_gt`, `_proportion_ignore`, `_select_overlap`; no score-sorted TP/FP labeling. Q1/Q2 share the temporal-best overlapping GT and same initialization. Zero-overlap compatible GTs are omitted by sparse official pairwise records: their scalar labels remain exactly0, but stored GT ID is null; diagnostics distinguish this from no same-class GT. Explicit ambiguity/ignore excluded, absent ambiguity metadata disclosed. Strict official thresholds and appearance/disappearance/score-order invariance tested. Q0 is diagnostic, never selectable. |
| §8 geometry and loss | Same stable one-to-one class-compatible concat assignment≥0.10; one entity across stages; original low-segment inverse map. Known mapped wall/floor/background labels0/1 are negatives;255 is ignored. Weighted BCE+Dice with smoothing1, ≤2048 shared sampled segments per stage, equal-stage then all-candidate averaging. M1/M2 omit unmatched mask loss only. GT-free regularizer covers all native inputs. Original failed regularization runs archived, charged, and excluded; see `recovery/REGULARIZATION_FIX.md`. |
| §8 invariants/gradients | All M heads share zero initial parameters, independent optimizers/storage, bounded±2 residual on all segments. Actual zero-step materialization matches B0; no shortcut. Each head's first two supervised updates changed parameters with nonzero gradients. Parent is absent from head training/optimizer; frozen features are shared. |
| §9 training/records | All six seed45 arms completed1500 updates with common SHA-bound sampling table,16 slots (3T2+1T1) per update, original schedule/optimizer. Every100-step metrics include total and component losses, positive/negative/unknown counts and update/LR. Earlier Q/M0 logs were deterministically replayed; all500/1000/1500 parameter tensors matched their original checkpoints. Original timing and all replay costs retained. |
| §10 CAL→SEL protocol | All24 CAL points complete before lock. Selected: Q1=1500,Q2=1500,Q3=1000,M0=1500,M1=1000,M2=500. CAL_LOCK immutable; seed45 SEL rejects any other point. PooledH1/H2 states are independent; partial populations return null with full expected denominator. Each selected arm must finish full SEL before final report. |
| §10 engineering vs mechanism | B0-relative gains and guard use1e-6 tolerance; mechanism contrasts separately compare Q2−Q1,Q3−Q2,M1−M0,M2−M1. Same-step seed45 comparisons use CAL to avoid a second SEL checkpoint search. Multiple/empty shortlist supported; same-slot alternatives not stackable. |
| §10 seed46 | Only positive-T2, guarded seed45 arms and direct controls; fixed steps, seed46 fresh heads, shared plan, schedule horizon1500, no re-selection. Actual replication results and reference tables are mandatory evidence before confirmed shortlist. |
| §11 profile/reload | Fresh full parent for each B0/selected arm on four hash-selected SEL pairs, one warmup+three measurements. Timed input/H2D/network/postprocess/descriptor/head/materialization/output, actual CUDA/RSS/disk IO. Metrics, hash and real-input reload checks outside timing. `/proc` counters replace unavailable optional psutil without installing dependencies. All six real bundle reload checks required. |
| §12 tests | `validation/CODE_CHECKS.json` binds exact tested source hashes and command output: enum/initialization/invariants, true materializer, official labels, low mappings/denominator, missing metrics/null and state isolation, fixed SEL rejection, checkpoint parent/data identity, common sampling and multiple shortlists. Actual481 input traces and six bundle reloads complement unit tests. Related native postprocess regression included; no full-repository audit or fuzz. |
| §13 CLI/resume/budget | run/status/report/publish and explicit stages implemented. run keeps fixed arm/stage scope, child-owned cost accounting and partial delivery; resume preserves historical cache/checkpoints and publication snapshot. Prior78.94000603858383GPUh retained. Failed/interrupted/replayed jobs charged. At most two idle A40s used; user GPU2 job left untouched. |
| §14 scientific artifacts | Final report gate requires all CAL/SEL, profile and diagnostics; `EXPERIMENT_CHECKS.json` checks shard hashes, source compatibility, full denominators, parameter/log invariants and budget. Root MODULE_SCREENING/SHORTLIST plus detailed selection/evaluation tables, all negative results and small head bundles are exported. |
| §14 publication | CommitA experimental code/results then commitB handoff/manifest; self-hash/B-self-reference excluded. Explicit task-only staging, staged diff check, actual branch/tag push and remote SHA required. All≤20MiB heads in Git. Existing Release auth unavailable at preflight; external receipt must disclose actual CODE_ONLY and local metric asset recovery command unless verified upload succeeds. |
| §15 final eight questions | Addressed by the rows above and final numeric audit. Completion requires real scientific and external publication evidence, not merely this review file or passing tests. No combination, PB/LOCAL/ADDITIONAL evaluation, new hyperparameter search, or default replacement authorized/executed. |

Final status is determined from the finished experiment report, artifact checks
and external publication receipt. Items describing required downstream evidence
must be checked against those artifacts before goal completion.

## Final scientific evidence inspected

All six fixed SEL evaluations are complete. M1 is the only seed45 guarded
positive (T2 delta+0.0010434389114379883, T1 delta−0.0006350278854370117),
with only2/4 positive references. Its fixed1000-step seed46 T2 delta is
−0.0003539919853210449, so its evidence is MIXED_SEED and the confirmed shortlist
is empty. The other five seed45 arms have negative T2 deltas against B0.
M0-46 at the same1000-step point was also evaluated as the required direct
control, independently of its own seed45 selected1500-step result.

Diagnostics cover43,300 TRAIN/CAL candidates,2,058 GT-presence records and2,538
selected-head geometry transition rows. No selected arm has a geometry-only
signal under the recorded criterion. Profile has112 rows (28 warmup,84 measured)
and all six actual bundle reload comparisons passed. Thirty saved checkpoint
head states plus six selected deployment bundles are public task artifacts.

All68 serialized official metric states were independently restored and their
AP values exactly reproduced. Cache/source/selection/head/invariant checks are
recorded in EXPERIMENT_CHECKS.json; final code tests/lint are bound in
CODE_CHECKS.json. A final failure-path review also ensured one blocked SEL arm
does not suppress the other arms and a partial report preserves available APs.

The scientific work is complete with1.0423023715761115 incremental GPU hours,
79.98230841015994 cumulative. Git/tag availability is deliberately verified
after commits A/B in the external publication receipt; this pre-publication
scientific review does not assert an upload that has not happened yet.
