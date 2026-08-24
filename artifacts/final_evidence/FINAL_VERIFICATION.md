# Final Evidence Verification

- Frozen environment: Python 3.10.20, Torch 2.6.0+cu126, pytest 8.4.2.
- Capacity, persistent-memory, ReScan parser, coordinate, protocol, label-map,
  evaluator, ambiguity, and no-GT-leakage tests: 268 passed, 1 conditional real
  dataset test skipped when the mount variable was absent.
- The skipped real-dataset test was rerun with
  `PERSIST4D_RESCAN_ROOT=/mnt/shared/ww/persist4d-final-evidence/rescan/dataset`:
  1 passed.
- New evidence scripts: Ruff lint and format checks passed.
- Tables and Figures 1-2 regenerated from frozen CSV inputs without error;
  PNG outputs were visually inspected.
- `git diff --check`: passed.
- `python -m scripts.verify_final_evidence`: passed after final manifest build.
- `artifacts/reviewer_closure/`: no tracked or untracked change.

The broad repository test suite was not used as the final gate because the
isolated worktree does not contain ignored internal dataset links; a prior broad
run failed on missing RIO/ScanNet data contracts rather than changed code. The
269 unique tests covering this final-evidence scope all pass with the frozen
environment and mounted ReScan sample.
