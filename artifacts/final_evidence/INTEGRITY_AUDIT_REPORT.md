# Final Evidence Integrity Audit

Mode: `full`.

Artifacts checked: final reports, paper tables, figure captions/index, capacity
evidence, ReScan manifests/results/gate, LivingScenes source binding, novelty
boundary, and immutable reviewer-closure source tables.

## Claim-Evidence Matrix

| Claim | Evidence | Status |
| --- | --- | --- |
| K=100 has measured headroom and should remain frozen. | `capacity_aggregate.csv`, `capacity_gate.json`, `CAPACITY_SENSITIVITY_REPORT.md` | Supported |
| Persist4D is an internal accuracy-identity-compute Pareto point, not uniformly task-superior. | reviewer-closure adaptation/compute CSVs, common-prefix Tables 2-3 | Supported |
| T5 gap recall is 0.3120 versus 0.0581 for adapted Full-History + Feature-Class. | `rescene_horizon_adaptation_results.csv` | Supported |
| T5 measured latency/VRAM are 440.4 ms/2473 MiB versus 1068.2 ms/4770 MiB. | `rescene_horizon_compute.csv` | Supported with scope qualifier |
| ReScan transfer is inconclusive and cannot support generalization. | `external_gate.json`, `rescan_raw.json`, `EXTERNAL_VALIDATION_REPORT.md` | Supported |
| Executed Full-History external baseline is also inconclusive. | `rescan_full_history_raw.json`, cache manifest, `EXTERNAL_FULL_HISTORY_BASELINE_AUDIT.md` | Supported |
| LivingScenes should remain Related Work rather than a quantitative baseline. | pinned source/weights manifest and official evaluator/config audit | Supported |
| Persist4D is the first persistent scene model or first sparse-revisit memory. | No supporting evidence; contradicted by positioning sources | Prohibited |

## Numeric Consistency

- T4/T5 task, identity, gap, latency, VRAM, and state values in the reports match
  the machine-generated Table 2/Table 3 CSVs.
- Capacity reports, captions, and gate agree on peak occupancy 30, zero true
  rejected births, K=100, and 61,008 state bytes.
- ReScan reports, dataset manifest, and gate agree on 13 scenes, 45 captures,
  333 Level-B object identities, 8 gaps, 2 gap scenes, and 0.055738 scene-macro
  coverage.
- The Full-History report/raw/cache manifest agree on 13 scenes, 45 captures,
  45 expanding-history entries, 2,247,570,182 cache bytes, 0.071881 coverage,
  and zero recovery attempts.
- Published Table 1 values remain percentages and are labeled reported/not
  rerun; common-prefix tables remain fractions from local evaluation.
- Figure 1/2 captions state that Full-History compute excludes tracker overhead.
- Main external Table 4/Figure 4 are consistently omitted after the failed gate.

One pre-audit inconsistency was corrected: `RESCAN_DATASET_AUDIT.md` previously
said the 8 gaps spanned 3 scene clusters; the manifest and evaluator show 2.

## Citation And Context Checks

- ReScene4D published numbers point to the official paper record and are kept in
  their native protocol.
- ReScan code/dataset claims point to the official project and pinned official
  repository source.
- LivingScenes claims are bounded to the pinned official repository, released
  configuration, evaluator, category file, and weight hash.
- No citation is used to imply that a method was rerun or directly comparable
  when it was not.

## Severity And Safe Edits

No unresolved critical, major, or numeric-consistency finding remains. The
mandatory manuscript edit is to preserve `PAPER_READY_INTERNAL_ONLY`, describe
ReScan as inconclusive domain-shift evidence, and avoid all prohibited priority
claims in `NOVELTY_BOUNDARY.md`.

No-invention status: pass. No result, confidence interval, baseline execution,
or external-support claim was inferred beyond the recorded artifacts.
