# Persist4D R1 Downstream Validation V1 Execution Plan

**Goal:** Replace only the frozen task checkpoint with qualified R1 and rerun
Protocol-B downstream validation for FullHistory, B2, and B4 without training or
changing candidate/tracker semantics.

**Workspace:** `research/persist4d-r1-downstream-validation-v1` in the isolated
worktree. Large caches live under `/mnt/shared/ww`; compact evidence is committed.

## Task 1: Freeze the experiment contract and runtime binding

- Add portable runtime configuration and artifact contract under
  `configs/r1_downstream_validation/` and
  `artifacts/r1_downstream_validation_v1/`.
- Add tests first for checkpoint identity, Protocol-B coverage, clean source
  binding, R1 runtime composition, and exact B2/B4 settings.
- Implement a new context module which reuses the existing dataset, model loader,
  local producer, and FullHistory producer while accepting only the registered R1
  checkpoint.
- Verify with the new context tests and the adjacent comparison tests.

## Task 2: Build resumable R1 caches

- Add tests first for progress resumption, raw/sidecar same-forward binding,
  FullHistory coverage, manifest provenance, and tamper rejection.
- Implement one-GPU `cuda:0`, FP32, seed-45 cache commands. Write all tensor data
  externally; publish only compact manifests in Git.
- Run three-cluster smoke/parity and record deterministic fingerprints.
- Commit code before full cache generation, then materialize 645 local raw+
  sidecar entries and 645 FullHistory entries on shared storage.

## Task 3: Compute downstream evidence

- Add tests first for T1-T5 fresh trackers, exact horizons, pooled metric
  accumulation, six-cluster paired bootstrap, identity attempt coverage, event
  ledger coverage, and the preregistered recovery rule.
- Compute direct local AP, temporal task mean/latest/max, identity outcomes for B2
  and B4, per-sequence/per-cluster/pooled tables, sensitivity, and old-vs-R1
  comparison. Historical replay may be marked unavailable but cannot block R1.
- Verify all 129 sequences, six clusters, T2/T4/T5 reports, and 10,000 seed-45
  paired bootstrap replicates.

## Task 4: Profile and finalize

- Add tests first for the six canonical profile units, T2/T4/T5 cells, 5+10
  repeats, measurement boundary, summaries, and final-manifest closure.
- Profile FullHistory and B4 on one A40, including model forward plus CPU tracker
  and excluding load/collate/H2D/scoring.
- Generate `FINAL_REPORT.md`, `HANDOFF.md`, `FINAL_MANIFEST.json`, all required CSVs,
  and explicit execution/candidate/recovery/task/resource/old-new/publication
  statuses.
- Run all direct tests, artifact validation, and Git clean checks; release CUDA
  resources.

## Task 5: Publish and read back

- Commit compact evidence in staged checkpoints, push the experiment branch,
  verify the remote ref with `git ls-remote`, and read back `HANDOFF.md` and
  `FINAL_MANIFEST.json` from the pushed commit.
- Perform final diff, scope, compatibility, and verification review before
  declaring completion.
