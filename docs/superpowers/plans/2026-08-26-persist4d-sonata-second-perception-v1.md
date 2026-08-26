# Persist4D Sonata Second-Perception V1 Implementation Plan

> Execute serially. The primary agent owns scientific semantics, debugging,
> integration, every gate decision, and the final evidence audit.

**Goal:** Produce one provenance-complete Sonata ReScene4D checkpoint and, only
if it qualifies, a minimal frozen cross-backbone Persist4D robustness package.

**Architecture:** A Sonata-specific provenance and authorization path wraps the
existing ReScene training runtime without weakening P2. Large immutable inputs
and outputs live under `/mnt/shared/$USER`; committed artifacts use portable
references plus content hashes. Frozen V3 evaluators are parameterized only
where required to accept the new checkpoint/cache roots.

**Environment:** `conda run -n persist4d python`, PyTorch 2.6/CUDA 12.6,
PyTorch Lightning 2.6.5, Sonata 1.0, stmetrics 0.1.0, pytest, ruff, Git, and
three NVIDIA A40 GPUs.

---

## Task 1: Freeze SS0 Audit And Scientific Contract

**Files:**
- Create this design and plan.
- Create `artifacts/sonata_second_perception_v1/START_STATE.{json,md}`.
- Create `artifacts/sonata_second_perception_v1/CODE_AUDIT.md`.
- Create `artifacts/sonata_second_perception_v1/EVIDENCE_BASIS.md`.
- Create `artifacts/sonata_second_perception_v1/SONATA_SCIENTIFIC_CONTRACT.md`.

Record the exact start commit, clean tracked state, frozen V3 and Concerto
hashes, live official revisions, local third-party revisions, baseline 57-test
result, code roles, unique recipe, stopping rules, and claim boundary. Validate
JSON, links/references, hashes, `git diff --check`, then commit SS0 before any
training action.

## Task 2: Acquire And Audit The Official Weight Test-First

**Files:**
- Create `tests/test_sonata_weight_provenance.py`.
- Create `tests/test_sonata_load_key_audit.py`.
- Create `utils/sonata_weight_provenance.py`.
- Create `scripts/acquire_sonata_weight.py`.
- Create `artifacts/sonata_second_perception_v1/weight/*`.

First test immutable-revision enforcement, regular-file requirements,
bytes/SHA matching, stale/anonymous rejection, and key-category validation.
Download `facebook/sonata/sonata.pth` at revision
`df99897472c09f91ba9288da0a034aacffc0b010` to a stable external root. Record
the LFS and observed hashes, bytes, source, timestamp, and CC-BY-NC-4.0 license.
Instantiate `enc_mode=False`, classify all loaded/missing/unexpected keys, and
pass SW0 only when every critical encoder/embedding key is loaded and missing
keys are decoder-side allowlisted.

## Task 3: Build Sonata Formal Preflight Test-First

**Files:**
- Create `tests/test_sonata_second_preflight.py`.
- Create `tests/test_sonata_config_contract.py`.
- Create `utils/sonata_second_preflight.py`.
- Create `scripts/sonata_second_preflight.py`.
- Create `conf/config_rescene4d_sonata_second.yaml`.
- Create a Sonata-specific callback config if needed.
- Create `artifacts/sonata_second_perception_v1/preflight/*`.

Test fail-closed source/weight/data/config/callback authorization before
implementation. Resolve the primary recipe exactly; retain P2 precision and
weighted-objective/runtime safeguards without invoking P2's Concerto identity.
Bind the verified 3RScan and 3134-file ScanNet manifests, validate normals and
actual collated feature dimensions, prove weighted mix instantiation, and issue
SP0 only when all upstream hashes and semantic checks pass.

## Task 4: Select Resources And Pass Smoke Gates Test-First

**Files:**
- Create `tests/test_sonata_temporal_sharing.py`.
- Create `tests/test_sonata_freeze_gradient_contract.py`.
- Create `scripts/sonata_second_smoke.py`.
- Create `artifacts/sonata_second_perception_v1/preflight/batch_feasibility.csv`.
- Create `artifacts/sonata_second_perception_v1/preflight/BATCH_SELECTION.md`.
- Create `artifacts/sonata_second_perception_v1/smoke/*`.

Probe batch sizes only for memory/resource feasibility on the selected A40
topology. Preserve effective global batch 32 through explicit accumulation.
Test a real 3RScan+ScanNet batch, Sonata load interface, temporal overlay,
temporal masking, absence of contrastive loss/future leakage, ReScene outputs,
frozen encoder gradients, finite nonzero decoder/head gradients, and a bounded
tiny optimization decrease. Pass SSMOKE before formal training.

## Task 5: Run The Single Formal Candidate

**Files:**
- Create `scripts/run_sonata_second_training.py`.
- Create `artifacts/sonata_second_perception_v1/training/*`.

Launch exactly seed 45 for 450 epochs from the SP0/SSMOKE-authorized config.
Write runtime events, source/config/data/weight hashes, learning curve,
checkpoint inventory, optimizer-step count, and interruption/resume evidence.
Resume only this candidate from verified epoch boundaries. Never launch a
second recipe or seed.

## Task 6: Freeze And Qualify The Checkpoint Test-First

**Files:**
- Create `tests/test_sonata_checkpoint_selection.py`.
- Create `scripts/evaluate_sonata_second_checkpoint.py`.
- Create `artifacts/sonata_second_perception_v1/checkpoint/*`.

Test selection isolation from Protocol-B/B2/B4 inputs. Select top-1 solely by
local `val_mean_t-AP`, freeze its SHA before robustness, evaluate at seeds
45/46/47 with the official-like harness, and evaluate the current Concerto
checkpoint under the same harness. Assign SQ-GREEN/YELLOW/RED exactly. Stop
automatic work at SQ-YELLOW/RED.

## Task 7: Build Conditional Sonata Protocol-B Cache

**Files:**
- Create `tests/test_sonata_protocol_b_binding.py`.
- Create `tests/test_sonata_local_candidate_invariance.py`.
- Create `scripts/build_sonata_system_comparison.py`.
- Create `artifacts/sonata_second_perception_v1/robustness/*`.

Only after SQ-GREEN, bind the frozen checkpoint to the exact Protocol-B
scene/order/horizon manifest and produce a new cache root. Prove local official
candidates are identical across tracker labels and that all cache records bind
to the new checkpoint SHA without touching V1/V2/V3 roots.

## Task 8: Run Conditional Robustness And Synthesis Test-First

**Files:**
- Create `tests/test_sonata_robustness_analysis.py`.
- Create `scripts/run_sonata_system_comparison.py`.
- Create `scripts/analyze_sonata_cross_backbone.py`.
- Create remaining robustness tables/reports and cross-backbone artifacts.

Reuse frozen B2/B3/B4 parameters and primary mean reducer. Recompute fresh
identity without GT inference; evaluate latest/max as sensitivities only; report
paired T4/T5 gap recovery, six cluster effects, temporal task metrics, local
task calibration, FullHistory, and T5 compute. Apply the preregistered SR gate
without tuning or treating order-units as independent.

## Task 9: Final Requirement Audit And GitHub Handoff

Create `FINAL_SONATA_SECOND_PERCEPTION_REPORT.md` and `FINAL_MANIFEST.json` from
hashed artifacts only. Audit every prompt requirement and expected deliverable,
all 10 Sonata tests, frozen regressions, changed-file ruff, broader tests where
available, `git diff --check`, large-file exclusion, portable paths, and no
V1/V2/V3 mutation. Inspect the complete diff, commit each stage separately,
push the branch to `origin`, and report the required 24 items in order.

