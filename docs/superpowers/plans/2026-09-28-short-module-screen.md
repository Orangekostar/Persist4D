# ReScene single-module screen implementation

Authoritative specification: `docs/0928/ReScene_Round1_Single_Module/ReScene_Round1_Single_Module_Codex_FINAL.md` and its protocol YAML. Base: `1ddab2aee88e5f5d8aa50b73a7896341802b2a26`. Work only on `research/rescene-short-module-screen-v1`.

## Contracts and sequence

1. Implement `models/short_module_heads.py`: one enum (B0/Q1/Q2/Q3/M0/M1/M2), prediction-only descriptors, same seeded Q trunk, differentiable monotone Q3, identical zero-initialized M heads. Test in `tests/test_short_module_heads.py` with non-128 dimensions, two classes for one query, H1/H2, negative-logit repair, parameter independence and gradients. Run `OMP_NUM_THREADS=2 /home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_short_module_heads.py` before and after implementation.
2. Bind native producer, actual collator, single-scan loader, installed matcher and metric semantics in `artifacts/short_module_screen_v1/CODE_BINDINGS.md`. Read V2 role/input/staging manifests and actual external budget. Implement `scripts/short_module_data.py` with deterministic metadata-only population, original low-segment supervision, score-invariant official labels and one-to-one geometry assignment. Add hand-derived tests before these implementations.
3. Implement narrow runner `scripts/short_module_screen.py` and config `configs/short_module_screen_v1.yaml`: prepare/export/train/evaluate-cal/screen/replicate/profile/report/publish, exposed by run/status/report/publish. Persist execution and cumulative GPU costs without resetting on resume. Keep prediction and target shards separate; cap new cache at 32 GiB. Freeze native numerical runtime and source/input/weight identities.
4. Verify one real CAL pair native soft equality and true one-scan H1; then export shared TRAIN/CAL/SEL inputs and complete B0. Preserve 23/24 logical T2 units and unique scan T1 denominators. Log actual runtime estimates and enforce 48 GPU-hour incremental / 192 cumulative caps, at most two GPUs.
5. Train six independent seed45 heads for 1500 updates with identical reference/record/candidate plans and fixed optimizer/schedule. Save 0/500/1000/1500 and recovery every100. Verify first supervised gradients and unchanged frozen parent. Evaluate all trained CAL points, lock all six before fixed SEL. Add tests for complete pooled denominators, lock ordering, multi-shortlist and reload.
6. Replicate eligible arms and required controls at fixed seed45 steps with seed46 if budget permits. Profile B0 and six selected heads on four fixed SEL pairs, each full forward, one warmup and three measurements. Record real reload equality for all six.
7. Produce all protocol §14 artifacts including negative results and required denominator/evidence limitations. Run focused regression, lint and diff check once after changes settle. Audit each specification requirement against actual evidence, including §15 eight checks.
8. Publish all small heads and reports via commit A then manifest/handoff commit B; push and verify branch/tag without moving existing tags. Use existing Release authentication if available, otherwise explicit CODE_ONLY with asset recovery commands. External final receipt binds A/B/tag/remote and actual availability. Mark goal complete only after full requirement audit.

## Ownership and evidence

Primary agent owns algorithms, official semantics, core implementation, experiments, review and integration. Only precisely specified mechanical leaves may use `mechanical_worker`. No additional modules, combinations, test populations or hyperparameter search. Failed or incomplete measurements stay null with their expected denominator; no placeholder PASS.

## Current progress

- Isolated worktree verified at fixed base; supplied package SHA256 checks passed.
- Implementation and real shared export completed:481 inputs, all six seed45 arms at1500 updates,24 CAL points and six locked full SEL evaluations.
- M regularizer correction completed before CAL lock; discarded attempts remain archived and charged. Deterministic logging replay exactly reproduced saved Q/M0 parameters.
- Eligible M1 and direct control M0 completed seed46 at fixed1000 updates. M1's positive seed45 signal failed replication; confirmed shortlist empty.
- Full112-row profile and six real bundle reload checks completed; candidate and geometry diagnostics complete. Scientific artifacts and30 public checkpoint states generated.
- Final source validation passed33 related tests and directed lint; all68 official metric states independently reproduce AP exactly. At the experimental snapshot, two-commit Git/tag publication remains the delivery step. External CODE_ONLY/VERIFIED receipt is authoritative for remote availability.
