# Persist4D Final Paper Evidence Report

## Classification

`PAPER_READY_INTERNAL_ONLY`

The architecture remains `FINAL_LOCK`. The paper may proceed on the completed
internal common-prefix evidence, but it must explicitly limit external
generalization claims.

## Gate Results

| Gate | Result | Consequence |
| --- | --- | --- |
| Bounded-state capacity | `CAPACITY_100_OK` | Keep frozen K=100; observed peak occupancy is 30 and no true birth is rejected. |
| Internal simple tracker | `TRACKER_REJECTED` | Persist4D's internal gap-recovery advantage is not explained by the strongest preregistered simple tracker. |
| Horizon adaptation | `HORIZON_ROBUST` | No T4/T5 task cell justifies replacing the frozen method; Persist4D retains identity/compute value. |
| Remaining ceiling | `PERCEPTION_CEILING` | Local observation/class/mask limits, capacity taxonomy, and unresolved evidence dominate registered identity failures. |
| Official ReScan method | `RESCAN_METHOD_NOT_REPRODUCED` | Discuss as historical context; do not place native numbers in the common-prefix table. |
| Independent ReScan transfer | `EXTERNAL_INCONCLUSIVE` | Do not claim generalization or produce an external main table/figure. |
| External Full-History baseline | `FULL_HISTORY_BASELINE_RUN_EXTERNAL_INCONCLUSIVE` | Conditional baseline executed; low coverage and zero recovery attempts prevent interpretation. |
| LivingScenes baseline | `NOT_RUN` | Related Work only; no GT-mask or restricted-subset comparison. |

## Frozen Internal Conclusion

Persist4D provides a useful long-horizon accuracy-identity-compute Pareto
operating point by decoupling bounded local perception from persistent entity
identity. At T5, Persist4D and the adapted Full-History + Feature-Class
alternative have t-mAP 0.0445 and 0.0454, while gap-recovery recall is 0.3120
and 0.0581. Measured update latency is 440.4 ms versus 1068.2 ms, and peak
allocated VRAM is 2473 MiB versus 4770 MiB. Persist4D carries 61,008 bytes of
bounded historical state; the Full-History input is 58,989,016 bytes at T5.

The task statement remains competitive or near-parity long-horizon t-mAP, not
uniform task-accuracy superiority. Persist4D also has lower t-REC at T4/T5.

## External Boundary

ReScan contributes 13 independent scenes and 333 eligible object identities,
but only 8 natural gaps from 2 scenes. Frozen Persist4D scene-macro observation
coverage is 0.055738 and no method makes a valid recovery attempt. The transfer
therefore tests neither the primary gap-recovery mechanism nor full-system
generality.

The executed Full-History + Feature-Class baseline raises scene-macro coverage
to 0.071881 but still makes zero recovery attempts and fails the same external
gate. Expanding the local perception horizon therefore does not make this
dataset sufficient for the registered identity claim.

The manuscript may state that an audited independent transfer attempt was
inconclusive under severe local-perception domain shift. It must not state that
Persist4D generalizes beyond the evaluated six internal 3RScan environments.

## Frozen Paper Package

- Published context: `PUBLISHED_BASELINE_AUDIT.md` and Table 1 CSV.
- Exact common-prefix results: Table 2 and Table 3 CSVs.
- Main and supplementary visuals: `PAPER_FIGURE_TABLE_INDEX.md`.
- Claim boundary: `NOVELTY_BOUNDARY.md`.
- External protocol and outputs: `EXTERNAL_VALIDATION_REPORT.md` and
  `EXTERNAL_FULL_HISTORY_BASELINE_AUDIT.md` plus `external/` machine-readable
  artifacts.
- Reproducibility binding: `final_evidence_manifest.json`, verified by
  `python -m scripts.verify_final_evidence`.

## Paper Action

Start manuscript production with the architecture and K=100 frozen. Treat the
independent-dataset result as a limitation and preserve the internal-only scope
through the abstract, introduction, experiments, and conclusion.
