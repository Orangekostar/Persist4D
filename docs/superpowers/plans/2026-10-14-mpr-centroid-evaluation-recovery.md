# Stable centroid evaluation recovery

Primary agent owns the diagnosis, implementation, validation and decision.
The existing GPT-5.6-Luna worker only monitors fixed paths and processes.

Native-v2 M1-u200 has a 0.4911815 pp S_long range, above the original 0.2 pp
repeat guard. All empty-history first-stage predictions still match M0.
The queue is held after its owned M1-u300 repeat0 finishes; every result and
charge remains preserved. No completed C3 model is retrained.

Matched five-stage probes have identical inputs, RNG, frozen weights and dense
features. Their segment centers first differ at the CUDA index_add_ reduction in
segment_context, followed by geometry and prompted predictions. A bounded
intervention replaces only the center sum/count order with stable FP32 group mean.

1. Finish three original and three stable-center probes, each charged to the
   original P0 ledger. Compare actual tensors, all five predictions and states.
2. Test and add an evaluation-only centroid helper. Preserve original validation,
   group IDs, padding, dtype, features and all non-center context fields. Restore
   the original callable on context exit; apply only in inference/P5 children.
3. Run three complete M1-u200 and three complete M3-u300 CAL controls using the
   exact tested helper, stable mean and Native sparse kernels. Check all 84 inputs
   and 197 stages per process, source/checkpoint identities, finite official AP,
   input-content hashes, pre-mask summaries and raw predictions. Preserve every
   repeat. Original 0.2 pp controls and selection guards stay unchanged.
4. If controls support continuation, create an explicit geometry-v3 policy,
   source/runtime lock and separate namespace. Keep all C3 training sources,
   24 deltas and v1/v2 policies/results unchanged. Use the original cost pool.
5. Admit three fresh formal M0 CAL processes, then regenerate the complete
   original four-point/three-repeat matrix under v3. Never combine policy rows.
   Selection, conditional seed147, SEL/PB and bootstrap remain unchanged.

Validation requires focused helper/policy tests, actual paired probes and full
controls, fresh formal admission and immutable checkpoint/source verification.
Measured equality is scoped to the tested population; do not claim universal
determinism or a scientific gain before the complete locked experiment.
