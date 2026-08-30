# ReScene Training Code Audit

## Confirmed Differences

| Area | Pinned public ReScene source | Current local P2 | Status before numeric audit |
| --- | --- | --- | --- |
| Optimized objective | Raw sum of returned criterion values | Weight-dictionary reducer, excluding per-layer contrastive diagnostics | Confirmed semantic difference; R1 is mandatory. |
| Gradient accumulation | `1` | `8` with two GPUs and batch 2/GPU | Confirmed runtime difference; causality unproven. |
| EOS coefficient | `0.1` in public config; `0.2` in paper | `0.2` | Paper/code ambiguity; audit gradients before authorizing R5. |
| Training filter | `[0, 1]` | `[0, 1, 255]` | Confirmed config difference; inventory label-255 mass before authorizing R4. |
| Data weights | `1.0/0.8` | `1.0/0.8` | Matched nominal weights; runtime draw semantics still require audit. |
| Frozen encoder mode | Frozen weights; runtime mode not reported | Frozen encoder remains in train mode with drop path enabled | Repeated-pass diagnostic required before R3. |
| Query content | Geometry-only FPS is the default | Geometry-only FPS is the default | Diagnose before A1; `use_np_features` already exists. |
| Superpoint aggregation | Mean in the pinned default | Mean | Diagnose before A2; adaptive scatter already exists. |

## Local Objective

`trainer/trainer.py::aggregate_objective_loss` multiplies returned losses by `criterion.weight_dict` and omits keys matching per-layer contrastive diagnostics. `_configured_objective_loss` enables this path for formal P2, reviewer-closure, and Sonata flags. The active P2 configuration sets `p2_weighted_objective: true`.

The public trainer at `fb2fe42eb8f1e926567c48eea9acb874e608ee10` directly sums all returned loss values. R1 therefore changes only the final reducer; matcher costs, EOS, data, batch, freeze, seed, and common initialization remain unchanged.

## Runtime Facts

The active local mix has child sizes `[1174, 1199]`, a weighted replacement sampler with 2,112 draws, two GPUs, batch size 2 per GPU, and accumulation 8. The resulting effective batch is 32 but the physical global batch is 4. Equality to a physical batch of 32 is not assumed because matching and normalization occur within physical microbatches.

Lightning may wrap a finite custom sampler for distributed execution. No sampler bug is claimed before recording the trainer-resolved wrapper chain and rank streams.

## Decoder Facts

`models/rescene.py::initialize_queries` initializes query content to zeros when `use_np_features=false`; the alternative projects sampled backbone features. `models/scatter.py::AdaptiveScatter` already supports `mean`, `max`, `adaptive`, and `gem`. ReScene mask attention thresholds detached logits at `0.5` and clears an all-masked query before cross-attention. These are diagnostic targets, not current root-cause conclusions.
