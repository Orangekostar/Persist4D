# ReScene round-one single-module screen

Scientific status: **COMPLETE**.

Six separate heads were trained from the same frozen R1 features. No combination or default deployment change was executed. Results are development screening, not a formal test or a claim to exceed official ReScene.

| Arm | Step | CAL T2 | SEL T2 | ΔT2 vs B0 | SEL T1 | ΔT1 vs B0 | Seed46 ΔT2 | Positive refs | Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | 0 | 0.386216 | 0.667541 | +0.000000 | 0.761583 | +0.000000 | — | — | FROZEN_R1 |
| M0 | 1500 | 0.375187 | 0.664821 | -0.002721 | 0.764583 | +0.002999 | — | 1 | NOT_REPLICATED |
| M1 | 1000 | 0.386608 | 0.668585 | +0.001043 | 0.760948 | -0.000635 | -0.000354 | 2 | MIXED_SEED |
| M2 | 500 | 0.385360 | 0.667331 | -0.000210 | 0.761540 | -0.000044 | — | 1 | NOT_REPLICATED |
| Q1 | 1500 | 0.247055 | 0.561815 | -0.105727 | 0.593538 | -0.168045 | — | 0 | NOT_REPLICATED |
| Q2 | 1500 | 0.239985 | 0.568443 | -0.099099 | 0.570563 | -0.191020 | — | 0 | NOT_REPLICATED |
| Q3 | 1000 | 0.262562 | 0.532380 | -0.135161 | 0.527758 | -0.233825 | — | 0 | NOT_REPLICATED |

M1's seed45 signal did not replicate: seed46 ΔT2=-0.000354. It remains a labeled single-seed signal, not a confirmed positive module. Its seed45 reference consistency is 2/4.

AP and deltas use the [0,1] scale. Positive reference counts describe consistency, not statistical significance.

Confirmed independent shortlist: empty. Provisional positive signals: M1. Same-slot candidates are **ALTERNATIVES_NOT_STACKABLE**.

## Population and protocol

Population counts: `{"CAL": {"T1": 23, "T2": 23, "references": 4, "unique_pairs": 23}, "SEL": {"T1": 24, "T2": 24, "references": 4, "unique_pairs": 24}, "TRAIN": {"T1": 199, "T2": 188, "references": 64, "unique_pairs": 188}}`. Every T1 input is a separately loaded and forwarded single scan. T2 stage AP is reported separately in evaluation evidence; it is not T1. TRAIN is physically disjoint from CAL/SEL and previously exposed PB/LOCAL/ADDITIONAL; R1 pretraining and prior development exposure remain disclosed.

All seed45 arms have 1500 updates. CAL selects only 500/1000/1500 after the full trajectory; update0 is diagnostic. All six selections were locked before head SEL scoring. Seed46 uses fixed seed45 steps and direct controls with a 1500-step schedule horizon. No epoch search or positive-seed replacement was performed.

Q1/Q2 have 95,873 parameters and identical seeded initialization; Q3 has 96,393 (+520) and the same seeded trunk. Labels use the same temporal-best GT, strict official > thresholds, and pre-disambiguation pairwise records. Output is a quality ranking score, not an established calibrated deployment probability. M0/M1/M2 each have 68,225 parameters and identical fresh initial states; M1 changes only unmatched supervision and M2 changes only stage aggregation. M0/M1 queries already contain temporal context.

## Mechanism and geometry

[Paired contrasts](evaluation/PAIRED_MECHANISM_DELTAS.csv) contain each arm's individually selected CAL/SEL point, same-step CAL contrasts, and fixed same-step seed46 SEL controls when available. Same-step seed45 diagnostics use CAL so SEL is not expanded into another checkpoint search. Engineering gain is always measured against B0; beating a degraded control alone is insufficient.

[Candidate diagnostics](diagnostics/CANDIDATE_FAILURES.csv), [joint/separate counts](diagnostics/JOINT_SEPARATE_COUNTS.csv), [appearance/disappearance](diagnostics/APPEAR_DISAPPEAR.csv), and [geometry transitions](diagnostics/GEOMETRY_TRANSITIONS.csv) disclose real denominators. Joint/separate geometry is not a realizable AP upper bound. Original-track diagnostics use the frozen class-compatible one-to-one assignment with concat IoU≥0.10 and one GT identity across stages. Score-only geometry is unchanged; any score-matching recall change is not new geometry.

## Cost and reproducibility

This round consumed **1.042302 GPU hours**; cumulative usage is **79.982308** including **78.940006** inherited usage. All process reservations, shared parent export and failures/retries are charged. Limits are 48 incremental/192 cumulative GPU hours. [Profile](resources/PROFILE.csv) contains a fresh full parent forward for B0 and every selected module on four fixed SEL pairs, one warmup plus three measurements each; metric/hash/reload work is outside timed intervals.

[Main results](evaluation/MAIN_RESULTS.csv) include actual updates, both APs/deltas, mechanism deltas, geometry repairs/breaks, measured latency and VRAM. [Public head inventory](PUBLIC_HEADS.json) binds all small saved states; each selected deployment bundle passed a real-input reload equality check. Original R1/Concerto, raw data and GT are not redistributed.

The existing data staging uses hardlinks and null per-file hashes; byte-immutability is not claimed. Where ambiguity metadata is absent, labels report AMBIGUITY_METADATA_UNAVAILABLE rather than claiming all real ambiguity was resolved. FP32, evaluation seed45, TF32-off and the recorded 2-thread/cuBLAS environment are fixed. The [regularization correction](recovery/REGULARIZATION_FIX.md) preserves and charges discarded M attempts; Q/M0 log replay verified exact saved parameter equality.

## Delivery

Git publication, tag and actual asset availability are recorded in HANDOFF.md and the external publication receipt. Release authentication was unavailable at initial preflight; small heads remain directly distributable through Git. Publication status is separate from scientific status. No deployment was replaced and no future combination was executed.
