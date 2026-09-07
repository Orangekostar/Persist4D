# Code Map

- `configs/r1_downstream_validation/default.yaml`: frozen experiment identities,
  methods, horizons, statistics, profiling, and recovery rule.
- `scripts/r1_downstream_context.py`: fail-closed R1 input validation, Protocol-B
  construction, portable inference composition, and strict checkpoint loading.
- `scripts/run_r1_downstream_validation.py`: audit, smoke/parity, resumable local
  and FullHistory cache generation, and compact cache finalization.
- `scripts/analyze_r1_downstream_validation.py`: downstream task, identity,
  cluster-bootstrap, sensitivity, and old/new regime analysis.
- `scripts/profile_r1_downstream_validation.py`: frozen six-cluster resource
  profile.
- `scripts/finalize_r1_downstream_validation.py`: report, handoff, and final
  manifest closure.
- `tests/test_r1_downstream_*.py`: direct correctness and tamper tests for this
  experiment.

Frozen implementations are reused from `evaluate_persist4d_p6a.py`,
`system_comparison_inference.py`, `system_comparison_v2_analysis.py`,
`system_comparison_v2_inference.py`, `system_comparison_v3_identity.py`, and
`profile_system_comparison.py`. Their established semantics are not modified.
