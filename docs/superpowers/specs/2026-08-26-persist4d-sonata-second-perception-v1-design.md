# Persist4D Sonata Second-Perception V1 Design

Date: 2026-08-26
Status: approved by the researcher-supplied execution prompt
Source prompt: external workspace input

## Objective

Train and qualify one provenance-locked Sonata-based ReScene4D local predictor,
then run the frozen Protocol-B cross-backbone comparison only if that predictor
passes the preregistered task-quality gate.

## Immutable Boundaries

- Start at `e5d7f4e96fedc76c0c6d414ab293f54909c61df3` on
  `research/persist4d-sonata-second-perception-v1`.
- Preserve every V1/V2/V3 artifact and all P2 Concerto authorization checks.
- Acquire `facebook/sonata/sonata.pth` at an immutable Hugging Face revision and
  reject any unverified same-name file.
- Run exactly one formal training candidate: seed 45 for 450 epochs.
- Select the checkpoint using local `val_mean_t-AP` before any Protocol-B run.
- Keep B2/B3/B4, Protocol-B, and primary `mean` score-reducer semantics frozen.
- Store large checkpoints and caches outside Git; commit only portable hashes,
  manifests, reports, and small tabular evidence.

## Primary Recipe

The primary configuration is Sonata PTv3 with frozen encoder/embedding,
train-from-scratch decoder-side parameters, 100 FPS non-parametric queries,
T=2, 0.02 m voxels, `standard + temporal_overlay`, temporal masking on,
contrastive loss off, EOS 0.2, and class/mask/dice weights 2/5/2. Training uses
3RScan T2 and ScanNet T1 at weights 1.0/0.8, AdamW at 5e-4, OneCycle, 450 epochs,
seed 45, effective global batch 32, P2's locked 32-bit precision policy, and
`frozen_encoder_eval=false`.

## Evidence Flow

SS0 freezes the source, evidence, code ownership, and scientific contract. SS1
freezes the official encoder weight and proves key-loading completeness. SS2
resolves and authorizes the Sonata-only training configuration. SS3 selects a
resource-feasible physical batch without changing effective batch and proves
the data, forward, temporal, loss, and gradient contracts. SS4 performs the one
formal run. SS5 freezes its locally selected checkpoint and evaluates it under
the official-like task harness at seeds 45/46/47.

SS6 and SS7 are conditional. They execute only after SQ-GREEN and consume the
already frozen Protocol-B manifest and B2/B3/B4 identities. All comparisons are
paired on the same scene/order/horizon units; the six reference scenes remain
the statistical clusters.

## Gates

- `SW0`: immutable official weight, matching bytes/hash, and critical encoder
  keys loaded.
- `SP0`: exact config, source, data, optimizer, callback, and effective-batch
  contract authorized.
- `SSMOKE`: chosen resource layout, real mixed batch, temporal paths, loss, and
  finite gradient contract pass.
- `SQ-GREEN`: complete 450-epoch provenance, mean t-mAP at least 29.7%, and
  Sonata spatial mAP at least the current Concerto result in the same harness.
- `SR-GREEN/YELLOW/RED`: assigned only from the preregistered cross-backbone
  rules after SQ-GREEN.

SQ-YELLOW or SQ-RED stops automatic robustness execution. Any failed earlier
gate stops the affected downstream stages without weakening the contract.

## Failure Handling

- A new official ReScene task checkpoint is recorded but never silently
  substituted.
- A mutable, anonymous, stale, or mismatched weight fails SW0.
- Missing ScanNet/3RScan assets, normals, split identity, or mixed sampling fail
  SP0 rather than degrading to a single dataset.
- Resource infeasibility may change physical batch and accumulation only; the
  effective global batch remains 32.
- A training interruption may resume only the same candidate from a verified
  epoch-boundary checkpoint with recorded runtime events.
- No final Sonata result may tune checkpoint epoch, trackers, or score reducer.
