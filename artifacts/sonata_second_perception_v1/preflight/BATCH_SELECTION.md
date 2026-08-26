# Sonata Batch Selection

- Gate scope: resource feasibility only; validation accuracy was not inspected.
- Hardware: 2 x NVIDIA A40
- Same node / same NUMA: yes / yes
- Interconnect: `NODE`
- Selected microbatch per GPU: 2
- Selected physical global batch: 4
- Gradient accumulation: 8
- Effective global batch: 32
- Probe: one real forward/backward on fixed approximately 95th-percentile mixed-data samples.
- Interpretation: this preserves the effective batch target but does not claim unpublished official physical-batch equivalence.
- Formal replay gate: `SS4-RESOURCE-BLOCKED`.
- The p95 microbatch-4 result is superseded by two deterministic full-loader epoch-0 OOM replays.
- Resource blocker SHA-256: `d6e8078eb62144885aa4daff388b7ad8d66e8ef4fbeda3bd5189b69009ef0277`.

Candidate sample bindings: `aa4dcbb622648e173125540353fcc7c9a2ace5b0923c296c960f3593bddc02bf`.
