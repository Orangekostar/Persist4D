# ReScene Task-Learning Root-Cause V1 Design

## Goal

Determine, with controlled local-perception evidence, whether a concrete public-code/runtime semantic difference explains part of the local ReScene4D task-learning gap. If reproduction semantics are insufficient, diagnose the decoder and test only the lowest-risk ReScene-native strengthening switches authorized by measured failure modes.

## Fixed boundaries

- Start from `29bb228ad5b090797045fa3b3fc55cb973f001be` on `research/persist4d-sonata-second-perception-v1`.
- Work on `research/persist4d-rescene-task-learning-root-cause-v1`.
- Never read Protocol-B, B2/B3/B4, gap-recovery, identity-switch, persistent-memory, latency, or VRAM results for model selection.
- Do not modify Persist4D memory code or frozen evidence namespaces.
- Keep paper-reported ReScene4D-C/S results separate from locally measured Concerto/Sonata results.
- Architecture-changing variants are `ReScene-Strong`, never official reproductions.
- Store large model states outside Git and commit only portable content-addressed manifests.

## Evidence model

Every reported conclusion has one authoritative lightweight artifact under `artifacts/rescene_task_learning_root_cause_v1/`. JSON/CSV files carry raw machine-readable evidence; Markdown files explain scope and verdicts without introducing new numbers. `FINAL_MANIFEST.json` hashes every committed artifact and records external files by logical reference, SHA-256, and byte size.

External facts are frozen from primary sources only:

- ReScene4D arXiv v2 and official `GradientSpaces/rescene4d` revision.
- LaSSM arXiv/official repository.
- CompetitorFormer CVF/official repository.
- Relation3D CVF/official repository.
- MAFT CVF/official repository.

## Architecture

### Contracts and preflight

`utils/rescene_rootcause_preflight.py` owns canonical hashing, portable references, source/data/runtime binding, variant-diff validation, 29,700-step scheduler validation, common-initialization validation, checkpoint manifests, and the no-Persist4D-leakage contract. `scripts/rescene_rootcause_preflight.py` materializes the start-state and authorization artifacts.

The root-cause configuration is `conf/config_rescene4d_concerto_rootcause.yaml`. It retains the 450-epoch trainer and an explicit `OneCycleLR.total_steps=29700`. A horizon callback stops only when the run first reaches epoch 90; resuming the epoch-90 checkpoint with the same configuration does not trigger the equality condition again and therefore preserves the full trajectory to epoch 450.

### Audit modules

Focused modules separate evidence domains:

- `utils/rescene_objective_audit.py`: objective-key inventory, upstream/local multipliers, fixed-batch contribution table, EOS gradient cosine/norm.
- `utils/rescene_data_audit.py`: label-255 target mass, dataset counts, mixed-sampler draw statistics.
- `utils/rescene_runtime_audit.py`: sampler-chain traces, cross-rank stream comparison, frozen-encoder repeated-pass statistics, physical-batch gradient comparison.
- `utils/rescene_decoder_diagnostics.py`: FPS coverage, query conflicts, mask-attention recall/reset events, and superpoint-feature statistics.

Each CLI delegates calculations to these modules and writes deterministic CSV/JSON/MD artifacts atomically.

### Common initialization

`scripts/build_rescene_rootcause_initialization.py` instantiates the Concerto model once from the verified pretrained encoder and seed 45, then saves the full pre-optimizer module state externally. `COMMON_INITIALIZATION.json` binds its SHA-256, byte size, model schema, trainable schema, encoder SHA, seed, source commit, and configuration hash.

R0/R1 and conditional reproduction variants strictly load this state before optimizer creation. ReScene-Strong variants load every common tensor from the same state and permit only their preregistered variant-specific parameter names to remain deterministically initialized.

### Training and checkpointing

`scripts/run_rescene_rootcause_training.py` validates the preflight authorization, variant isolation, common initialization, output root, and exact runtime inputs before launching. It uses:

- R0: weighted current P2 objective.
- R1: raw upstream released-code objective.
- At most two conditional R2-R5 variants authorized by preregistered audit gates.

The callback set retains only the best validation checkpoint, `last.ckpt`, and exact epoch 60/90/450 checkpoints. This is sufficient for paired evaluation and exact resume while bounding disk use.

Standard validation occurs at epochs 15/30/45/60/75/90. Early elimination is permitted only at epoch 45 when stage1 and stage2 mAP are no better than R0 at all three preceding checkpoints.

### Evaluation and decisions

`scripts/evaluate_rescene_rootcause_checkpoint.py` reuses the frozen official-like 154-sequence evaluator and seeds 45/46/47. It emits per-seed and summary rows for t-mAP, t-mAP50, t-mAP25, overall mAP, stage1 mAP, stage2 mAP, and `SpatialStageMean`.

`scripts/finalize_rescene_rootcause_stage.py` applies gates mechanically. A full reproduction candidate is authorized only when every prompt condition is satisfied. Exactly one qualifying candidate may resume to 450 epochs; otherwise RC4 receives a `gate_skipped` artifact.

### Decoder diagnostics and strong-local variants

Diagnostics instrument existing model outputs without changing training behavior. Architecture work remains gated:

- A1 sets only `use_np_features=true`.
- A2 sets only `scatter_type=adaptive` and requires the superpoint gate.
- A1+A2 requires both independent short-curve gates.
- Query competition and attention-mask relaxation require their respective measured diagnostic gates and a committed design before code.

No high-risk module is implemented speculatively.

## Data flow

```text
primary sources + exact parent commit + local files
  -> START_STATE / external evidence / source hashes
  -> RC0 and RC1 audit artifacts
  -> authorized variant set
  -> one common initialization
  -> R0/R1/(conditional) 90-epoch curves
  -> paired epoch-60/90 local evaluation
  -> optional one-candidate epoch-90-to-450 resume
  -> decoder diagnostics
  -> A1, then conditional A2/combination
  -> final manifest, report, handoff, GitHub verification
```

## Failure handling

- Missing or mismatched source/data/checkpoint hashes fail closed before compute.
- Variant config drift outside the allowlist fails before model construction.
- Non-finite objective, gradient, audit statistic, or metric aborts the affected stage.
- OOM is recorded as a resource result; infeasible physical batches are never represented as measurements.
- A non-authorized stage produces a small explicit `gate_skipped` artifact, never fabricated numeric rows.
- Interrupted training resumes only from a checkpoint whose optimizer, scheduler, sampler, source, config, and initialization bindings pass.

## Verification

- Unit tests cover every named contract in the execution prompt.
- GPU integration tests cover common initialization, sampler runtime, stochasticity, physical-batch diagnostics, and one-step variant isolation.
- The full parent test suite and Ruff must pass after implementation.
- Artifacts undergo hash validation, numeric-source traceability, frozen-namespace diff, no-Persist4D-leakage scan, private-path/secret scan, and `git diff --check`.
- Completion requires local HEAD to equal `git ls-remote` for the task branch.

## Design decision

Use a fail-closed, gate-driven implementation that reuses existing P2/Sonata provenance and evaluation machinery. Direct tuning and immediate external-module ports are rejected because they cannot isolate root cause and violate the task's scientific hierarchy.
