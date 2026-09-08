# Experiment Contract

## Goal

One frozen Persist4D checkpoint must strictly exceed the matched ReScene comparator in causal-prefix t-mAP at T=2,3,4,5. Mean score reduction is primary; no average, recovery, or efficiency result may compensate for a failed horizon.

## Frozen evidence

- Baseline commit: `2c7494b982eff84886aef3a7274bae43752a1485`
- R1 checkpoint SHA256: `629ff7624dcac15e6022906e808e2e05b3ec61c60a1116ab0e278f0cfd2368dd`
- Protocol-B SHA256: `246497165612699b103d0d79d5503025cb2cd14466aad3ab149d4fe82884ecbe`
- Split manifest SHA256: `d01e94e32b6e07ea6092840d9165000c73a4b5c5b79661a656e8b5bdf38bad98`
- R1 caches: 645 local and 645 FullHistory entries.

## Data roles

The 36-reference RIO train partition is adaptation data. The 8-reference train holdout is development-only and was exposed to R1 base training. Protocol-B is final-only. Processed test data is unavailable, so independent generalization is not established.

## Budget and gates

The provisional 4000-update, effective-batch-8 budget may be amended once after real throughput preflight and before formal training. C3 and FH-L are conditional. Model selection uses only development data.
