# MPR-V1 final handoff

Publication state in this commit: PENDING_EXTERNAL_VERIFICATION.
The external receipt is attached to the tag Release when available, otherwise
retained in the bound run root's `publication/PUBLICATION_RECEIPT.json`.

- Task: fixed E1-23760 seed45 historical retrieval prompts and M0–M6 controls.
- Base: `6b1b779172afdeb51e2719a737cd05e7eeeca982`.
- Branch: `research/persist4d-mpr-v1-20261013`; tag: `persist4d-mpr-v1-20261013-v1`; frozen experiment commit A: `0a090814b17df5c1770a4f743f0fe9c1f71a82ea`.
- Code: `IMPLEMENTED_P0_P1_VERIFIED_FORMAL_STAGES_NOT_EXECUTED`; experiment: `BLOCKED_FULL_MATRIX_BUDGET`;
  science: `INCONCLUSIVE`.
- New GPU hours: 0.931006687; lifetime total: 1714.814061416;
  lifetime balance: 85.185938584;
  retained reserve: 8.000000000.
- No MPR GPU reservation remains active at publication preparation.

Completed: current-source P0 connectivity/numerical panel, three complete M0
CAL processes, six eight-update P1 cost pilots, and saved-state metric repooling.
Formal N is null; M1–M6 formal trajectories, quarter checkpoint selection,
seed147 replication, SEL/PB confirmation, bootstrap, mechanism/profile, and
formal deployment reload were not executed because the complete matrix did
not fit the frozen budget. There is no actual winner or formal MPR checkpoint.
P0/P1 updates use adapter seed146; seed147 has no trajectory in this run.

Parent dependency: `artifacts/native_long_postselection_v1/deployment/E1-23760.pt`,
SHA256 `8c9d0d57044ce23c7964ddb1e7a2d8bd18e388e6ee657a7f4b02b1368b5b3a98`. The Concerto encoder is separately obtained
under its original terms, SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`. Raw 3RScan data,
point-level GT and the encoder are not part of this release. Bind available data,
encoder and original environment with the resolver before model construction.
Acquire 3RScan under its original terms via the upstream source listed in
MODEL_MANIFEST and follow this repository's README preprocessing instructions.
The same manifest gives the parent's exact existing Git/LFS location and the
original Concerto distribution. No dependency is silently replaced.

`MODEL_MANIFEST.json` lists each actually produced delta/resume and its bytes,
hash, variant, seed and step. Reconstruct via
`scripts.revisit_prompt_assets.load_revisit_bundle(delta_path, assets=bound_assets)`;
the delta depends on the exact parent and encoder. Resume only from listed resume
files with the frozen RUN_LOCK and the same training identity; a deployment delta
does not include optimizer state. Seeds146/147 adapt the same seed45 parent.
In this budget-blocked run MODEL_MANIFEST lists only the real official-state
archive. Missing formal deltas/resumes are explicitly NOT_PRODUCED; the CPU
debug delta check is not a published trained model or a real deployment forward.

Selection uses complete three-repeat CAL quarter checkpoints, then S_long,
S_current, T2, fewer parameters and variant name; within a family, earlier steps
break ties. Seed147 families are fixed by the main selection. SEL/PB do not select
models. Actual endpoints: `{"candidate_descriptive_star": null, "candidate_star": null, "control_star": null}`.

Read `REPORT.md`, `STATUS.json`, `PROMPT_AUDIT.md` and the locked result CSVs for
all incomplete stages, denominators, negative findings and numerical limitations.
`DATA_MANIFEST.json` records historical exposure. Official state files can be
recomputed with `scripts.evaluate_revisit_prompt.pool_official_states` and the
bound taxonomy; the compressed state archive is identified in MODEL_MANIFEST.
`ARTIFACT_MANIFEST.json` freezes A; this handoff does not embed its own commit SHA.
See `COMMANDS.md` for the actual commands and MODEL_MANIFEST for asset locations.

Unique continuation decision from the completed report:
Stop new GPU work under the current lock. The single justified continuation is a new budget admission covering the same complete fixed matrix; current measured forecast requires at least 15.450310993 additional GPU hours beyond the 78-hour pool. This is a conservative cost estimate, not a guarantee or method NO_GO.
