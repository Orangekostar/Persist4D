# Cross-Backbone Analysis

- SQ gate: `SQ-RED`
- SR gate: `SR-RED`
- Sonata Protocol-B status: `gate_skipped`

Under the same three-seed local T2 harness, the Sonata reimplementation
reached t-mAP `24.035%` and overall mAP
`31.554%`. The matched Concerto reimplementation
reached `28.290%` and `36.979%`.
Sonata is therefore weaker on both qualification axes and does not
authorize SS6 or SS7.

Frozen Concerto V3 remains positive for B4-minus-B2 gap recovery
at T4 (+19.971 pp) and T5 (+22.690 pp), with all six physical-scene
clusters positive at both horizons. No corresponding Sonata values
were computed, so those Concerto results cannot be generalized across
backbones in this experiment.

Conclusion: current persistent-state advantage is not yet
cross-backbone validated. This is a negative qualification result,
not evidence that persistence fails with Sonata.
