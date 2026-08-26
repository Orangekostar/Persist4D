# Sonata Batch Selection

- Gate scope: resource feasibility only; validation accuracy was not inspected.
- Hardware: 2 x NVIDIA A40
- Same node / same NUMA: yes / yes
- Interconnect: `NODE`
- Selected microbatch per GPU: 4
- Selected physical global batch: 8
- Gradient accumulation: 4
- Effective global batch: 32
- Probe: one real forward/backward on fixed approximately 95th-percentile mixed-data samples.
- Interpretation: this preserves the effective batch target but does not claim unpublished official physical-batch equivalence.

Candidate sample bindings: `75372a0487b02e4d2e37e88c63e122d97e597fd80b825379172b09542cf9ec0e`.
