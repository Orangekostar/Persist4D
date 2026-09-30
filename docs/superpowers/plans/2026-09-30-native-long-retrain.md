# Native Long Retrain Implementation Plan

> Execute inline with superpowers:executing-plans. Only deterministic configuration/archival leaves may use mechanical_worker; the primary agent owns semantics, implementation and review. The user explicitly authorized execution without questions.

**Goal:** Execute the four-arm native ReScene retraining protocol, including the budget-authorized real runs and verifiable GitHub delivery.

**Architecture:** Preserve native ReScene and Lightning. Add an explicit encoder-only load scope, training-only stage sampling and one query-conditioned segment feedback hook. A stateless global draw plan determines domain/reference/order/horizon/augmentation; a thin campaign owns binding, population, preflight, budget, checkpoint evaluation and delivery.

**Tech Stack:** Installed persist4d Python environment, PyTorch, Lightning, Hydra, Concerto, stmetrics, two A40 GPUs.

**Spec:** artifacts/native_long_retrain_v1/EXECUTION_INSTRUCTION.md (sole execution specification); protocol.yaml is its structured mirror.

## Global Constraints

- Base c6e9d01edfe2b3c832374a424fc45e44bdadfb68, branch research/rescene-native-long-retrain-v1; defaults unchanged.
- E0/E1/E2/E3 begin with freshly generated seed45 common tensors; seed46 is independently initialized only if eligible and affordable.
- Concerto SHA256 845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07; load embedding/encoder only, freeze weights, training encoder mode retained.
- U=29700 optimizer updates, global batch32, FP32/TF32 off, raw_sum including every original criterion key, OneCycle total_steps29700, no shortened full-run claims.
- Real unique scans from one physical reference, xyz+t for every input, TRAIN excludes CAL/SEL/PB/LOCAL/ADDITIONAL and historical holdouts.
- E2 alone changes capped attention training sampling; E3 alone updates maskF after execution index7, without altering the original InfoNCE input or loss keys.
- Lifetime GPUh cap192 includes all prior and failed work, reserve at least8; preflight cap1 and numerical investigation cap0.5. At most two GPUs, eight CPU workers, 96GiB RAM.
- No combinations, parent task warm start, feature-cache training, altered metric, reduced population, default replacement, force push or main merge.

## Tasks And Evidence

- [x] Read full instruction and historical source/evidence; create specified isolated worktree.
- [x] Archive specification and compose explicit configuration without historical preflight/callback constraints.
- [x] Add bounded regression tests, observe missing APIs and the inverse-map counterexample fail, implement and verify affected boundaries.
- [x] Implement strict encoder-only load and new seed45 common initialization; verify real keys/task freshness/F CPU and CUDA RNG isolation. Seed46 not triggered.
- [x] Implement stage quota selection and late reverse attention; verify real fixed caps, extras/padding/T1, initial identity, chunk/checkpoint parity, two-step gradients and later native predictions.
- [x] Freeze TRAIN361/CAL8/SEL8 and paired global draw plan; verify real mixed batches and annotation IDs; bind actual input content hashes.
- [x] Add Lightning subclass and rank-local checkpoint RNG; actual Lightning split-resume fixture proves draw/LR/parameter equality. Full native DDP not exercised.
- [x] Run bounded native A/BA/ABA checks, raw_sum gradients, F/large T5 backward and real cost/IO. Runtime remains conditional; failures and early CPU-fixture all-device CUDA capture are charged.
- [x] Freeze BASELINE_RECOVERY_ONLY with an empty full-arm set. Even E0 forecast ~380 GPUh exceeds ~111 GPUh balance plus8 reserve. All four actual trajectories stay at0; CAL/SEL scores remain null.
- [ ] Full training, CAL/SEL, seed46, formal confirmation and trained-model profile: NOT_RUN_BUDGET. Positive-plan integration and real-run validation remain outstanding.
- [x] Primary review numbered requirements with separate development/scientific coverage and explicit partial statuses.
- [x] Implement scoped A/B/tag publication and external byte verification. Actual delivery is certified only by the external publication receipt after push. Release authorization was absent; initialization task package remains local and no trained model exists.

## Verification Commands

Use `/home/ww/miniconda3/envs/persist4d/bin/python -m pytest -q tests/test_native_long_*.py` plus existing affected objective/Pointcept/decoder/loader tests. Use targeted Ruff, compileall and `git diff --check` once integration is ready. Run `python -m scripts.native_long_campaign run --config conf/config_native_long_retrain.yaml --root /home/ww/persist4d_runs/native_long_retrain_v1 --base-commit c6e9d01edfe2b3c832374a424fc45e44bdadfb68 --resume`; status/report/publish must use the same bound root and revalidate current identity.

## Initial Evidence

Metadata-only inspection: 377 eligible official TRAIN references, groups Tmax2=160, Tmax3=128, Tmax4=53, Tmax>=5=36; all declared processed scans exist. Prior reported cumulative GPUh80.7350387502 remains subject to event-ledger reconciliation. Three A40 cards idle at inspection; no more than two will be reserved. SSH git read works; gh executable absent, Release channel still requires verification through existing tooling.
