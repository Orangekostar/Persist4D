# Persist4D Gain V2 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The primary agent owns design, scientific decisions, core code, review and integration. Only explicit mechanical leaf contracts may be delegated.

**Goal:** Execute the supplied V2 protocol through real paired training, component selection, locked confirmation, profiling and verified publication; preserve missing evidence as incomplete.

**Architecture:** Reuse the V1 training, producer, metric, publisher and refiner primitives through explicit optional recipe/path arguments. A small V2 entry point owns the task DAG, resource ledger and artifacts. Training, evaluation and deployment share one resolved recipe identity. No copy of the V1 campaign or change to V1 artifacts.

**Tech Stack:** Existing persist4d Conda environment, PyTorch/Lightning, Hydra, official stmetrics, Git and available GitHub authorization.

**Spec:** `docs/0921/Persist4D_Gain_V2/Persist4D_Codex_Gain_V2_FINAL.md` (SHA256 `9b92e302d0af269614e86a76591cd324374f83fe64b097aa4c2ffdb330c3b688`). This supplied complete design is authoritative; user explicitly authorizes autonomous implementation.

## Global constraints

- Parent `8b5e93795817682fe70864daa545db219e1443c9`; branch `research/persist4d-perception-gain-v2`; artifacts `artifacts/perception_gain_v2`; external root `PERSIST4D_GAIN_V2_ROOT`.
- Cumulative cap 192 GPU-hours including V1 11.803828054742961 plus reconciled omitted costs; confirmation reserve initially 40, adjusted before PB using measured cost ×1.25.
- At most 4 idle A40s and 2 independent training jobs; available now 3 A40s. CPU workers ≤8, RAM ≤96GiB, new prediction cache ≤32GiB, association ≤16 CPU-core-hours.
- Fixed R1/Concerto/scorer identities and frozen DATA_ROLES. No input cropping, difficulty filtering, new hyperparameter search, PB tuning or modified V1 results.
- H/L learning rates 5e-5/1e-5; scheduler 3000, warmup150, FP32, effective batch32. H350 exact resume; L fresh R1 optimizer. One new full mechanism and its same-LR/runtime C0 only.
- R1 NEW_ONLY/OLD_NEW share routing, eligibility, initialization and sampling; 1500 updates, ±2 residual, actual materialization, no score/class/state changes. New P requires newly generated features and paired retraining.
- Targeted tests only; final affected regression once, targeted ruff once and diff check. CPU tests ≤30min, GPU smoke ≤1 GPU-hour. Explicit live evidence required beyond unit tests.
- Native LOCAL metadata correction: the unchanged original 154 sequences actually cover 46 references (40 ADDITIONAL + 6 PB; TRAIN overlap 0). V1 incorrectly copied all 41 ADDITIONAL references into LOCAL roles. Preserve DATA_ROLES as historical registration, bind the actual population audit in FINAL_LOCK, and disclose 41→46 in the report.

## Task 1 — Runtime identity, independent bootstrap and data reliability

Files: add `configs/perception_gain_v2.yaml`, `scripts/perception_gain_v2.py`, `scripts/perception_gain_v2_config.py`, `tests/test_perception_gain_v2.py`; extend `scripts/train_perception_gain.py`, `trainer/perception_gain_trainer.py` and live asset resolver.

Interfaces: `resolve_recipe(variant, learning_rate, seed, devices)` returns the fixed flat numerical recipe and its hash; `compose_variant_config(..., recipe_config=None)` applies it to every real consumer; checkpoint `perception_recipe` binds resume identity. V2 CLI bootstrap/status/data records source, assets, provenance and cost without importing the all-five-arm validator.

- [x] Test real composed H/L optimizer rates, fixed 3000 horizon, schedule/cursor restoration and mismatched resume rejection.
- [x] Implement optional recipe and loader timeout; seed before model construction; preserve default V1 calls.
- [x] Inspect manifest-referenced data paths, stage NFS files to local storage if capacity permits, verify relative names/bytes and retain full populations.
- [x] Bootstrap with asset hashes, actual code commit, role intersections, previous cost reconciliation and auth observations; validate low-cap accounting.
- [x] Commit executable code before formal runs. Run the required bootstrap CLI.

## Task 2 — Shared live identity and full development baselines

Files: extend perception/native/local/refiner evaluation, preparation and profile entry points, `perception_gain_foundation.py`; add targeted cases in existing evaluation tests.

Interfaces: optional `recipe_config`, `artifact_root`, `assets_path`, `roles_path`, `external_root`; execution provenance records actual HEAD plus relevant source digest, complete input/point order/seed/publisher identity.

- [x] Test live assets without legacy caches; recipe reaches train and evaluation; preserve official population denominators.
- [x] Verify one real CAL sequence: repeated R1, D0 vs disabled refiner, zero residual through bool materialization, unchanged candidate/identity/score.
- [x] Produce full CAL/SEL R1-D0 and native FH (23/24 units); export read-only live canonical evidence for association and multihead evaluation. The fixed-runtime repeat reproduces the repair parent exactly and was adopted at 20:19 UTC after the core controller exited. Original nonreproduced runtime evidence and costs are preserved in the explicit supersession record.

## Task 3 — Independent perception jobs

Files: V2 entry point/config and existing trainer integration; tests cover DAG readiness, retry signatures and budget reservation.

- [x] DAG supports per-task/recipe terminal states and dependency closure; REPAIR_R1 has no perception-selection dependency. Persist PID/argv/device/start/end/update interval and ledger from primary process.
- [x] Run H350→750, C0-L→750 and S-BAL-L→750 on the shared sample stream; all three have actual COMPLETE summaries at update750. Reuse C0-H weights read-only.
- [x] Finish all eight prescribed H/L CAL checkpoints with actual identity; every result covers23/23 units. The fixed queue completed at20:50 UTC, and the separately evaluated S-BAL-L750 is also complete.
- [x] Choose LR* by comparable CAL probes. The actual same-runtime/sample-stream audit passes for all four recipes; LR*=L. Its highest-ranked probe C0-L250 has mean delta+0.003148 and minimum delta−0.009059, so this LR choice alone does not establish a formal model win.
- [ ] Run A-OPEN/Q-SEM/S-WORST in protocol order with explicit skips. AUX started at20:51 UTC with A-OPEN's prescribed zero-step evaluation. Keep scorer frozen and separate its steps/cost.
- [ ] Select at most one mechanism + same-LR C0 (or C0-only positive branch), continue fixed schedule to3000, CAL-lock then SEL-select P.

## Task 4 — Fair R1 repair pair and shared evaluation

Files: `models/perception_gain.py`, `train_perception_refiner.py`, `prepare_perception_refiner.py`, `perception_refiner_evaluation.py`, refiner tests.

Interfaces: shared mode adapter replaces old logits/score with new for NEW_ONLY before derived features; frozen/resume schema carries mode/parent recipe hash/parent weight/source shards/seed/updates/bound; every loader uses it.

- [x] Test NEW invariance to old values, matched initialization/sampling, independent archives, strict mapping, source/mode mismatch rejection and no zero shortcut.
- [x] Cache fixed 32-reference R1 training episodes once; compute pair/fallback and residual-correctability diagnostics from that data.
- [x] Train each head to1500, preserve checkpoints0/500/1000/1500 and actual curves/audits.
- [x] Evaluate all CAL heads with one streamed upstream pass and separate publishers/official metrics; select each head then evaluate fixed SEL points and three comparisons. Both heads select step 0 and retain R1; all trained 500/1000/1500 results are retained.

## Task 5 — Limited association and optional P repair

Files: existing canonical replay/association adapter and V2 task runner.

- [ ] Replay exactly12 configurations on fresh R1 CAL evidence, M-new/mean; compare bridge to actual R1-D0; stop only at config boundaries for CPU budget. Recovery02 is running: its first configuration completes23/23 CAL units and reproduces all four D0 baseline scores exactly. The invalidated historical cohort remains excluded.
- [ ] CAL-lock one A*, evaluate once on SEL, keep separate from refiner candidates.
- [ ] If P≠R1, R1 repair has the required signal and budget permits, regenerate same TRAIN list under P and train/evaluate a new NEW/PAIR pair.

## Task 6 — Selection, replication and confirmation

Files: V2 runner/config helper, existing evaluation entry points and targeted decision tests.

- [x] Implement protocol ranking/tolerance, same-LR C0 attribution, NEW/PAIR gates, tie_update and fallback; verify four supplied policy examples and missing-coverage null.
- [ ] Write immutable FINAL_LOCK with fixed confirmation inventory before PB. Record omitted search scopes.
- [ ] Run allowed seed46 fixed-step component/pipeline replication without reselection.
- [ ] Complete PB 129/516 per distinct required method; LOCAL154 per prescribed seed; ADDITIONAL111/77/32 units. Preserve failed-prefix denominator and separate native FH errors.

## Task 7 — Deployment costs and downloadable artifacts

Files: existing profile/report/publish modules and their V2 integration.

- [ ] Profile actual locked methods on6 canonical masters, 1 warmup+3 repeats, sameA40, component and cumulative latency/memory; mode-aware loader.
- [ ] Bundle exact replacement parameters/changed buffers with base hashes and mode/recipe; validate reload on a real used panel. Include same-LR C0 and NEW control where required; package reproducible predictions without GT.
- [ ] Generate required8 tables, all real small curves/checkpoints/scores, full denominators, literal reproduction/recovery commands and nonrecursive manifest.

## Task 8 — Review and publication

- [ ] Primary agent checks each §19 item against live files and exact execution coverage, runs final relevant tests/lint/diff check, records remaining missing evidence.
- [ ] Commit science A, generate handoff+manifest B, push branch/tag and compare remote SHA; V1 unchanged.
- [ ] With available authorization create draft prerelease bound to B/tag, upload required inventory/receipts before publishing, verify downloadable bytes/digests. Otherwise CODE_ONLY with actual missing assets, never empty-list VERIFIED.
- [ ] Final report supplies branch/B/tag/report/handoff/downloads, measured deltas/populations/steps/costs and exact remaining recovery actions. Goal completion requires the real protocol audit, not this checklist alone.
