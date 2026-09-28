# Frozen round-one execution instruction

Authoritative supplied instructions: [ReScene_Round1_Single_Module_Codex_FINAL.md](../../docs/0928/ReScene_Round1_Single_Module/ReScene_Round1_Single_Module_Codex_FINAL.md). The package's SHA256SUMS was verified before implementation. Executable configuration: [short_module_screen_v1.yaml](../../configs/short_module_screen_v1.yaml), parsed against the exact supplied protocol.

Base commit: `1ddab2aee88e5f5d8aa50b73a7896341802b2a26`.
Branch: `research/rescene-short-module-screen-v1`.
External root: `RESCENE_SHORT_MODULE_ROOT`, default `/home/ww/persist4d_runs/short_module_screen_v1`.

Scope is B0 plus six independently trained fresh heads Q1/Q2/Q3/M0/M1/M2 on frozen R1. Only real H1/H2 input forwards. No combinations, parent fine-tuning, new filtering, ensembles or deployment replacement. No PB/LOCAL/ADDITIONAL scoring. Each seed45 trajectory requires1500 updates and complete CAL/SEL denominators. All CAL locks precede fixed SEL head evaluation. Eligible seed46 replication is fixed-step, budget-limited and includes direct controls. Multiple candidates or no positive candidates are valid outcomes; evidence is at most development replication.

Numerical runtime: FP32, evaluation seed45, TF32 disabled, OMP/MKL/OPENBLAS threads2, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. New cache limit32 GiB, at most two A40s, cumulative192 GPU-hour cap and incremental48 GPU-hour cap. Prior settled usage78.94000603858383 GPU hours is carried forward from the actual V2 ledger.

Prediction caches and GT targets are separate and remain external. Publish small heads, protocol, implementation, tests, complete numerical tables, negative results and limitations. Publication requires actual Git push/tag verification; Release unavailability must remain explicit CODE_ONLY. Final requirement-by-requirement audit must include all eight §15 checks. Preparation or passing unit tests alone does not complete this task.
