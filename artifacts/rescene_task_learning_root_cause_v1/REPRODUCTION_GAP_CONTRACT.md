# ReScene Reproduction-Gap Contract

## Scientific Order

The task isolates reproduction semantics before local architecture changes. Local spatial quality is primary. Persist4D memory and all final persistent-protocol metrics are outside model selection.

## Common Trajectory

Every root-cause curve starts from the same verified Concerto pretrained encoder, decoder/head state, RNG state, data/split contract, and sampler contract. The common state is saved outside Git and bound by SHA-256, byte size, tensor schema, trainable schema, seed, source commit, and config hash.

All short curves retain the full 450-epoch OneCycle trajectory:

```text
66 optimizer steps/epoch * 450 epochs = 29,700 total steps
```

A run may stop at completed epoch 90, but it must not use a 90-epoch scheduler. An authorized candidate resumes the exact epoch-90 optimizer, scheduler, and sampler states.

## Reproduction Variants

- R0 changes nothing from the current weighted-objective control.
- R1 changes only the final objective reducer from weighted to raw sum.
- R2 may change only physical batch and accumulation while retaining effective batch 32 and 66 steps/epoch; it requires the physical-gradient gate.
- R3 may change only the frozen-encoder stochastic policy; it requires repeated-pass cosine below `0.999` or relative RMS deviation above `1e-3`.
- R4 may change only the training filter from `[0, 1, 255]` to `[0, 1]`; it requires label 255 to represent at least 0.5% of target instances or supervised target points.
- R5 may change only EOS from `0.2` to `0.1`; it requires material fixed-batch gradient evidence and an available slot.

R0 and R1 are mandatory. At most two conditional variants are authorized, so no more than four reproduction-compatible short curves may run.

## Decisions

Validation occurs at epochs 15, 30, 45, 60, 75, and 90. A non-control curve may stop at 45 only when both stage1 and stage2 mAP are no better than R0 at all three preceding checkpoints. Temporal mAP cannot trigger early elimination.

Epoch-60 and epoch-90 checkpoints use the identical 154-sequence local official-like evaluator at seeds 45, 46, and 47. Primary selection is `SpatialStageMean = (stage1_mAP + stage2_mAP) / 2`; overall mAP is secondary and t-mAP tertiary.

Exactly one non-control candidate may continue to epoch 450, and only when all five preregistered conditions in the execution prompt pass. If none passes, RC4 is explicitly gate-skipped.

## Strong Local Boundary

A1 (`use_np_features=true`) is evaluated only after decoder diagnostics. A2 (`scatter_type=adaptive`) is conditional on the superpoint gate and A1 evidence. These are labelled ReScene-Strong, never official reproductions. Query competition or attention-mask relaxation requires its own committed diagnostic-gated design.
