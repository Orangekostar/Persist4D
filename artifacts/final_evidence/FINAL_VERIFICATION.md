# Final Evidence Verification

- Frozen environment: Python 3.10.20, Torch 2.6.0+cu126, pytest 8.4.2.
- Capacity, persistent-memory, ReScan parser, coordinate, protocol, label-map,
  evaluator, ambiguity, no-GT-leakage, per-scene-effect, Full-History strategy,
  and final-verifier tests: 290 passed with the real dataset mount enabled.
- Real dataset root:
  `/mnt/shared/ww/persist4d-final-evidence/rescan/dataset`.
- Evidence scripts and tests: Ruff lint and format checks passed.
- Formal Full-History inference: 13 scenes, 45 captures, 45 content-hashed
  entries, expanding `[S1,...,St]` contract, and 2,247,570,182 external bytes.
- Every external Full-History cache entry was rehashed against its committed
  manifest; all 45 passed.
- Tables and Figures 1-2 regenerated from frozen CSV inputs without error;
  PNG outputs were visually inspected.
- Full-History and local-pair per-scene effect files each contain 104 rows over
  13 scenes; their scene means reproduce the registered bootstrap means.
- `git diff --check`: passed.
- `python -m scripts.verify_final_evidence`: passed after final manifest build.
- `artifacts/reviewer_closure/`: no tracked or untracked change.
- Git: no tracked file exceeds 50 MiB; checkpoints, datasets and tensor caches
  remain on shared external storage.

The broad repository test suite was not used as the final gate because the
isolated worktree does not contain ignored internal dataset links; a prior broad
run failed on missing RIO/ScanNet data contracts rather than changed code. The
290 unique tests covering this final-evidence scope all pass with the frozen
environment and mounted ReScan data.
