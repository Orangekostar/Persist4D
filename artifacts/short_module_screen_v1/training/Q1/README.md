# Q1 standalone head

Mode: `score_only`. Exactly one runtime module; no combinations.

`deployment.pt` is the seed45 CAL-selected head. `checkpoints/` contains every saved small head, including negative controls and fixed-step seed46 controls when run. Optimizer states are intentionally excluded from these public inference bundles; external recovery checkpoints retain them.

Load `head` with `models.short_module_heads.build_head` using `metadata.dimensions`, thresholds and seed, then call `apply_module` with native parent prediction/descriptor and the original materialization system. No GT input is accepted. Metadata binds the frozen R1 SHA, supported real H1/H2 inputs, selected update, descriptor and postprocess. R1/Concerto weights and data are not redistributed.

A score head's update0 is not an identity transform. A mask head's update0 is zero residual through the real materializer. Disable through B0. Same-slot heads are alternatives, not stackable.
