# CrossWindow V1 preflight review

| Item | Verdict | Evidence |
|---|---|---|
| source_mapping | PASS | SOURCE_MANIFEST.json hashes fixed parent sources |
| model_frozen | PASS | R1 checkpoint SHA fixed; E6 skipped |
| GT_boundary | PASS | production association imports no diagnostic target |
| candidate_preservation | PASS | canonical ledger and collision tests pass |
| point_direction | PASS | three-cycle and target alignment tests pass |
| comparison_source | PASS | D0/A methods share supplement forwards |
| missing_overlap | PASS | missing uses base score and is logged separately |
| E3_equivalence | PASS | exhaustive 2-3 group audit <=1e-12 |
| E4_scoring | PASS | mask-only and selected-score channels separated |
| split_isolation | PASS | CAL/SEL/PB reference roles are disjoint |
| fixed_start | PASS | selection deltas use fixed D0 |
| budget | PASS | recorded usage remains within configured budgets |
| real_timing | DEFERRED_EXTERNAL_ASSET | native FH cache unavailable; E5 resource result must be UNCONFIRMED |
| publication_loop | PASS | E/P/readback workflow is configured |

Numeric-correctness verdict: **PASS**. The native-FH timing dependency is external and will remain explicitly unconfirmed.
