# Final Reviewer Closure V3

## Decision

**RC3-YELLOW.** Gates B0, PB0, PB1, EV0, ID0, and OR0 pass. The
mechanism evidence remains useful, but the B4-versus-B2 T2 sign changes across
the preregistered score reducers. Mean remains primary; no favorable reducer is
selected. Claims are therefore narrowed to the shared frozen local-perception
regime and to the directly supported long-gap identity result.

## Run Boundary

- Branch: `research/persist4d-reviewer-closure-v3`.
- Start commit: `c2f1bcacff1ec244909426b57403965f679f08cc`.
- Synthesis parent commit: `838b3b0`.
- Shared checkpoint SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
- Protocol-B manifest SHA256: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`.
- V2 cache manifest SHA256: `f525252cd306f29e7db80788e4d5c773d2a35c8767d5eaa63b189740ea212182`.
- PB1 GPU inference ran on the audited NVIDIA A40 inventory (46,068 MiB,
  driver 595.71.05, CUDA 12.6). The frozen PB1 cache did not record the exact
  device alias; all three audited visible GPUs had identical properties.
- No frozen V1/V2 artifact was overwritten.

## Gate Ledger

| Gate | Status | Evidence |
|---|---|---|
| B0 | PASS | Baseline evidence classes and claims remain separated. |
| PB0 | PASS | 43/43 exact canonical T2 prefixes; 14/14 overlap parity; no substitution or future leakage. |
| PB1 | PASS | Population, order, and horizon factors are reported separately. |
| EV0 | PASS | Mean regresses exactly; latest/max change confidence reduction only; local-current channel is invariant. |
| ID0 | PASS | Fresh B4 regresses over 7,224 cells; fresh B2/B3/B4 coverage is complete. |
| OR0 | PASS | GT enters only after candidate prediction as an identity linkage key. |

## Table A: Evidence Level

All t-mAP values are percentages. Protocol-B rows show all-order T2/T3/T4/T5.

| Method/source | Checkpoint availability | Evaluation protocol | t-mAP | Evidence class | Allowed claim |
|---|---|---|---:|---|---|
| ReScene4D-C (paper-reported) | Reported task checkpoint unavailable | Paper protocol | 34.8 | External reference | ReScene4D reports 34.8%. |
| ReScene4D-C (our best-effort reimplementation) | Local checkpoint available | P2 full-154 T2 | 27.939 | Local best-effort reproduction | This local run reaches 27.939%; it is not the official model. |
| FullHistory shared-frozen | Same local checkpoint | Protocol B, all orders | 19.100 / 10.790 / 6.900 / 4.534 | Controlled internal baseline | Controlled comparison within the shared local-perception regime. |
| Persist4D shared-frozen (B4, mean) | Same local checkpoint | Protocol B, all orders | 20.724 / 12.310 / 7.023 / 5.250 | Controlled mechanism comparison | Mean-reducer task result and fresh identity result under identical candidates. |

The controlled FullHistory/Persist4D comparisons use exactly the same frozen
reimplementation, so they isolate the history/identity-management mechanism
within that local-perception regime. They do not establish superiority over the
unavailable official ReScene4D checkpoint.

## Table B: Exact-Prefix Protocol Bridge

| Population | Seed | Order | Horizon | N masters | N reference clusters | t-mAP | t-REC |
|---|---:|---|---:|---:|---:|---:|---:|
| Full-154 | 45 | official | T2 | 154 | n/a | 27.675 | 41.449 |
| Exact-43 | 45 | canonical | T2 | 43 | 6 | 20.348 | 43.965 |
| Full-154 | 46 | official | T2 | 154 | n/a | 26.607 | 39.668 |
| Exact-43 | 46 | canonical | T2 | 43 | 6 | 19.515 | 46.164 |
| Full-154 | 47 | official | T2 | 154 | n/a | 28.035 | 37.751 |
| Exact-43 | 47 | canonical | T2 | 43 | 6 | 20.069 | 50.170 |

PB0 coverage is 43/43 exact ordered scan prefixes with 14/14 overlap-target
parity. The exact-43 minus full-154 t-mAP population effects are -7.327,
-7.091, and -7.966 percentage points for seeds 45, 46, and 47. Their mean is
-7.462 points with range [-7.966, -7.091]. This is a paired population
diagnostic, not a decomposition from the paper-reported 34.8%.

## Population, Order, And Horizon Effects

Order effects are paired by master and summarized over the six physical
reference-scene clusters. FullHistory reverse-minus-canonical has mean cluster
effect +0.103 points, range [-1.322, +1.242], and descriptive bootstrap interval
[-0.616, +0.819]. Its sha256-minus-canonical effect is -3.538, range
[-14.370, +5.021], interval [-9.344, +2.115]. Persist4D-V2 has corresponding
effects +0.464, range [-4.702, +3.743], interval [-1.878, +2.333], and -0.650,
range [-13.354, +10.617], interval [-6.961, +5.700]. These are descriptive
N=6 robustness summaries, not high-powered significance claims.

At T5, FullHistory retains 21.367%, 22.702%, and 26.246% of its T2 t-mAP for
canonical, reverse, and sha256 order. Persist4D-V2 retains 26.219%, 27.870%,
and 29.203%. All 24 method/order/horizon cells remain in the generated horizon
table; cross-horizon observations are not treated as independent.

## Table C: Controlled Method Comparison

Primary score reducer is mean. Values are all-order percentages. Local AP is
listed as T2/T3/T4/T5. Identity diagnostics are T4/T5.

| Method | T2 t-mAP | T3 | T4 | T5 | Direct local-current AP | Gap recovery recall T4/T5 | ID-switch rate T4/T5 |
|---|---:|---:|---:|---:|---|---|---|
| FullHistory | 19.100 | 10.790 | 6.900 | 4.534 | n/a | n/a | n/a |
| B2 | 20.727 | 10.231 | 4.529 | 1.823 | 37.169 / 38.443 / 36.626 / 38.292 | 9.770 / 8.509 | 11.761 / 12.666 |
| B3 | 20.727 | 10.295 | 4.467 | 1.897 | 37.169 / 38.443 / 36.626 / 38.292 | 11.207 / 9.424 | 13.172 / 13.701 |
| B4 | 20.724 | 12.310 | 7.023 | 5.250 | 37.169 / 38.443 / 36.626 / 38.292 | 29.741 / 31.199 | 10.349 / 11.138 |

The direct local-current task channel and fresh query-level identity diagnostic
channel are separate prediction objects. FullHistory has no V3 direct-sidecar
local-current or fresh query-level identity row, so those cells are not filled
from frozen V1 identity fields.

## Table D: Score Sensitivity

| Method | Reducer | T2 | T3 | T4 | T5 |
|---|---|---:|---:|---:|---:|
| B2 | mean PRIMARY | 20.727 | 10.231 | 4.529 | 1.823 |
| B2 | latest | 20.298 | 10.155 | 4.396 | 2.015 |
| B2 | max | 19.599 | 10.680 | 5.303 | 2.422 |
| B3 | mean PRIMARY | 20.727 | 10.295 | 4.467 | 1.897 |
| B3 | latest | 20.298 | 10.171 | 4.341 | 2.123 |
| B3 | max | 19.599 | 10.696 | 5.227 | 2.602 |
| B4 | mean PRIMARY | 20.724 | 12.310 | 7.023 | 5.250 |
| B4 | latest | 20.362 | 11.449 | 5.908 | 4.548 |
| B4 | max | 19.617 | 11.166 | 6.407 | 4.701 |

The B4-minus-B2 T2 difference is -0.003 points under mean but +0.064 under
latest and +0.018 under max. At T3-T5 B4 remains above B2 for all reducers, but
the effect size changes materially. Mean remains primary. The temporal AP
ranking claim is score-aggregation sensitive. Against the frozen FullHistory
T4 value of 6.900%, B4 is +0.123 points under mean but -0.992 under latest and
-0.493 under max.

Direct local-current masks, classes, scores, and AP are exactly invariant over
516 fixed sidecar/horizon groups. Tracker independence is also covered explicitly
for B0/B2/B3/B4. The mean V2 regression maximum absolute difference is zero;
1,935 score-only trajectory checks pass.

## Fresh Identity And Long-Gap Evidence

Fresh B4 identity recomputation exactly matches all 7,224 frozen V1 regression
cells while copying no V1 identity field into V3 result rows. Fresh coverage is
1,548 rows for B2/B3/B4 over 129 order-units and T2-T5.

B4-minus-B2 pooled gap-recovery recall is +19.971 points at T4 and +22.690 at
T5. The six T4 cluster effects are +32.941, +17.847, +7.692, +100.000,
+34.314, and +6.742 points. The six T5 effects are +32.847, +17.143,
+13.483, +100.000, +38.974, and +11.594. All six clusters favor B4 at both
long horizons. Generic ID-switch effects remain mixed across clusters, so the
claim is specifically stronger long-gap recovery, not uniform dominance on
every identity diagnostic.

## Table E: Identity Headroom

| Linkage | T2 t-mAP | T3 | T4 | T5 | Gap recovery recall T4/T5 |
|---|---:|---:|---:|---:|---|
| B2 | 20.727 | 10.231 | 4.529 | 1.823 | 9.770 / 8.509 |
| B3 | 20.727 | 10.295 | 4.467 | 1.897 | 11.207 / 9.424 |
| B4 | 20.724 | 12.310 | 7.023 | 5.250 | 29.741 / 31.199 |
| Oracle-ID diagnostic | 19.787 | 13.029 | 8.611 | 6.111 | n/a |

Oracle-ID minus B4 is -0.937, +0.719, +1.588, and +0.860 points at T2-T5.
Thus Oracle-ID shows positive remaining linkage headroom only at T3-T5. Its
negative T2 difference demonstrates that this post-hoc perfect-linkage
diagnostic is not a monotonic AP upper bound when fixed candidates are merged
and ranked. Oracle-ID is not a method or baseline. GT class is never used for
matching, and candidate masks, predicted classes, and scores remain unchanged.

## Bounded-Update Evidence

Frozen compute evidence shows that FullHistory processes 2/3/4/5 scans per
update at T2-T5, whereas Persist4D processes two. At T5, median measured update
latency is 1067.633 ms versus 440.429 ms, peak allocated VRAM is 4964.054 MiB
versus 2472.894 MiB, and Persist4D state is 61,008 bytes versus a 58,989,016-byte
explicit FullHistory input. These measurements are frozen evidence and were not
rerun or selected in V3.

## Verification

- New V3 tests: 41 passed.
- Required frozen-regression tests: 59 passed.
- Post-sanitization path/manifest tests: 33 passed.
- All changed Python files: `ruff check` passed.
- Completion audit: 31/31 prompt-required files and all 43 bridge targets are
  present; key/submanifest hashes and manifest provenance contracts pass; no
  frozen V1/V2 artifact changed.
- Full repository suite before the path fix: 1908 passed, 5 failed, 11 skipped.
- The fixed path-privacy failure was rerun and passed; the remaining four tests
  were rerun together and failed only on unavailable external/runtime assets.
- Full-repository `ruff check .` reports 566 pre-existing findings outside the
  V3 changed-file set.

Remaining external/runtime failures are:

1. P2 runtime source contract: pinned `concerto`, `detectron2`, `sonata`, and
   `stmetrics` repository roots/native extension are not mounted in this worktree.
2. ScanNet preflight: official split snapshot files are absent.
3. FullHistory replay: the external replay cache directory does not match the
   frozen manifest inventory.
4. Eleven opt-in GPU/real-cache/ReScan tests remain skipped by their explicit
   environment gates.

The official ReScene4D task checkpoint remains unavailable. The repository
contains no second checkpoint with materially different local AP and complete
provenance; the optional two-quality-level stage is therefore not run.

## Artifact Hashes

| Artifact | SHA256 |
|---|---|
| `baseline/baseline_evidence_contract.json` | `48405dfb6e14eb6720f02a3a142e527477ffccb9fdf49623f2f1385520e67c94` |
| `protocol_bridge/bridge_manifest.json` | `011d7b8350eaaa6ad7c0aa21200a5f8b18794631361100f18773db5a6adbfa8a` |
| `protocol_bridge/protocol_bridge_manifest.json` | `6a766c6da15bd15defb6d143976dee3ddeeb5257cf833889846ca023a8912116` |
| `score_sensitivity/manifest.json` | `4d8dce0b8fad49ff379a6c413243a5d04cea944b9923da1a018aa0ebe246ff89` |
| `identity/manifest.json` | `03ead35bdc5e714265123b105960dfcde794bc7ccc92faf0b1b118f697adc0de` |
| `oracle_identity/manifest.json` | `8051dd662a3a7e31df2ef22897e95b91d95346b897a8889e4249c976a842bb0f` |
| Frozen compute table | `267911184acf1050ce589f07401a228c5b6cf49262f17409951445262a9a65c2` |

The final manifest contains the complete changed-file contract and hashes for
the required V3 evidence. Submanifests cover every generated CSV/report and the
43 exact bridge target files.

## Claim-Evidence Audit

| Claim | Evidence | Status |
|---|---|---|
| Exact-prefix bridge is valid. | PB0 manifest and bridge inventory. | Supported |
| Population, order, and horizon are separable reported factors. | PB1 CSVs/report. | Supported |
| Local candidates are tracker/reducer invariant. | EV0 manifest and per-sequence hashes. | Supported |
| B4 has stronger long-gap recovery than B2 at T4/T5. | Fresh ID0 aggregate and six-cluster effects. | Supported |
| B4 universally dominates B2 on identity. | Mixed cluster ID-switch effects. | Unsupported and forbidden |
| Temporal AP superiority is reducer-independent. | T2 sign flip and T4 FullHistory sensitivity. | Unsupported and forbidden |
| Oracle-ID is a monotonic upper bound. | Negative Oracle-minus-B4 at T2. | Unsupported and forbidden |
| Persist4D beats official ReScene4D. | Official checkpoint unavailable and protocols differ. | Unsupported and forbidden |

## Claims Now Authorized

1. Under identical frozen ReScene4D reimplementation outputs and identical
   official local task candidates, persistent B4 linkage has stronger long-gap
   recovery than B2 at T4/T5, with positive effects in all six physical-scene
   clusters.
2. Under the preregistered mean reducer, all-order B4 t-mAP is
   20.724/12.310/7.023/5.250% at T2-T5; this is a controlled internal result,
   not an official ReScene4D comparison.
3. Persist4D provides a bounded two-scan update and bounded entity state in the
   frozen compute profile, while FullHistory update input grows with horizon.
4. Protocol population, order, horizon, and score-reducer effects are reported
   separately; the score-aggregation sensitivity narrows task-ranking claims.
5. ReScene4D-C is the scientific starting point and reports 34.8%; our separate
   best-effort reimplementation reaches 27.939%.

## Claims Still Forbidden

- Do not call 27.939% the official ReScene4D result.
- Do not compare 34.8% directly with Protocol-B values as one benchmark.
- Do not claim Persist4D beats the unavailable official ReScene4D checkpoint,
  is SOTA, or is reducer-independent.
- Do not treat 129 order-units as independent scenes or the six-cluster
  descriptive intervals as high-powered significance.
- Do not present Oracle-ID as a method, baseline, or monotonic AP upper bound.
- Do not claim B4 uniformly dominates B2 on generic ID-switch behavior.

## Recommended Next Stage

Acquire a legitimate second local-perception checkpoint with complete
provenance, preferably the official ReScene4D task checkpoint if released.
Preregister a minimal T2/T4/T5 replication of FullHistory versus B4 using the
same candidate/reducer/channel contract. Separately calibrate a score reducer on
a held-out development population before any new final evaluation; do not pick
one from the current test population.

RC3-YELLOW
