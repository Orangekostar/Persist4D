# MPR-V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Only deterministic leaf tasks may use mechanical_worker; the primary agent owns numerical semantics and review.

**Goal:** 完整执行 MPR-V1 开发、受预算限制的真实实验、冻结选择和远端发布。

**Architecture:** 薄模型/数据/运行适配复用固定BASE的原生系统和指标。新独立训练与成本入口保护旧资产和账本。

**Tech Stack:** Python 3.10, PyTorch, existing Concerto/Pointcept, Hydra, stmetrics, Git/LFS.

**Spec:** `docs/superpowers/specs/2026-10-13-revisit-prompt-design.md`; all executable numerical contracts and acceptance rules in `artifacts/revisit_prompt_v1/EXECUTION_INSTRUCTION.md`.

## Global Constraints

- BASE=6b1b779172afdeb51e2719a737cd05e7eeeca982; parent E1-23760 seed45.
- M0–M6; adapter seed146, conditional seed147; Q=K=100,D=128,max_prompt_slots=32.
- CPU development first; P0/P1 each at most4 GPUh; new total at most78 GPUh; old reserve8 untouched.
- Preserve official metrics and all negative/failed results; no raw scans/GT/encoder redistribution.
- No user questions: resolve choices within the supplied fixed protocol and evidence.

## Tasks and acceptance

- [x] A1 Read all 21 prompt sections, verify package checksums, create BASE worktree and verify source blobs.
- [x] A2 Implement `scripts/revisit_prompt_assets.py` resolver and strict parent migration; bind real environment, data roles and current budget in START_STATE.
- [x] B1 Write failing focused CPU fixtures T01–T05; implement prompt/state/loss modules and minimal ReScene hook, checking parameter sets and current-segment exclusion.
- [x] B2 Write failing UUID/local/absolute time and common-GT fixtures T02/T06/T08; implement episode adapter and runtime final commit/candidate lineage.
- [x] B3 Implement trainer and train CLI: four-episode updates, frozen parent, checkpoint/RNG/identity fields; CPU resume coverage and real M6 three-update gradient/freeze check completed. Other arms receive actual checks in P1.
- [x] B4 Implement evaluator and selector, focusing T07/T09 and official sufficient state; help commands must import without fabricating success.
- [x] B5 Implement campaign charged_job and P0–P5 handlers, cost forecast/run locks, resume identity and bounded retries. CPU checks passed; actual phase completion is recorded separately below.
- [x] C Complete real M6 four-horizon gradient/freeze check and four-input three-process P0 panel. Input structures agree, output floats differ; numerical root cause remains unresolved. P1 admission is still required.
- [ ] D Run full M0 CAL repeats and six adapter cost pilots under P1; forecast full queues and lock unique maximal feasible N or truthful blocking state.
- [ ] E–I Execute full six-arm matrix, four CAL points, conditional full seed147 pair, locked SEL/PB, CPU bootstrap, mechanism/profile/deployment reload; follow every fixed branch.
- [ ] J Implement final reports/manifests/delta/resume packaging and `publish_revisit_prompt --execute`; verify branch/tag and fresh-cache asset bytes.
- [ ] Audit all §17 deliverables, T01–T10, phases A–J and the 12 questions in §19 against current-state evidence before completion.

## Verification commands

Run from the worktree using the resolver-bound existing Python:

```sh
python -m pytest tests/test_revisit_prompt.py -q
python -m scripts.revisit_prompt_campaign resolve --auto
python -m scripts.revisit_prompt_campaign preflight
python -m scripts.revisit_prompt_campaign plan
python -m scripts.revisit_prompt_campaign run --resume
python -m scripts.revisit_prompt_campaign summarize
python -m scripts.publish_revisit_prompt --execute
```

Each code task follows a failing behavior fixture before implementation. The exact
fixtures exercise the specified tensor and event contracts, not implementation
mirrors. Failed or missing dependent phases remain explicitly incomplete.
The authoritative task state is persisted under artifacts/revisit_prompt_v1 and
the bound run root. A future turn resumes the first unchecked requirement.

Current boundary: B1–B5 have implementations, with a 26-case focused CPU baseline,
two affected CPU repair rounds, and the required CLI help/dry-run checks.
Actual P0 M6 checks completed three optimizer updates, fourteen stages per update,
unchanged frozen parent tensors, and changes to the used new parameters. These
checks are not formal trained models. Failed attempts and prior source revisions
remain archived and charged in the new ledger; the historical ledger is unchanged.
The current four-input, three-process P0 panel completed all twelve processes. Its shared-parent
comparison correctly excludes MPR-only diagnostic fields. Initial fixed-context
replays show floating differences and one T5 input has official scalar differences,
so numerical repeatability is not claimed.
P1 CAL repeats and six cost pilots, maximal-N admission, the formal matrix,
confirmation, deployment, and A/B remote publication remain incomplete.

Resolver revision r004 adds explicit CAL/SEL roles without changing scan lists,
indices, order, or runtime asset bindings. All 711 evaluation records parse. The
current-source real P0 replay completed 3 M6 updates and all 12 probe processes;
its input structures agree, but numerical root cause remains unresolved. The prior
P1 role failures completed zero CAL inputs and remain archived and charged.
A reviewed implementation revision is bound before restarting P1; no additional
whole CPU suite or affected repair round was run for this metadata change.
