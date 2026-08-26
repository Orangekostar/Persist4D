# Sonata Second-Perception Scientific Contract

Status: frozen before weight acquisition, configuration implementation, smoke,
or training.

## Unique Primary Recipe

| Field | Frozen value |
|---|---|
| Backbone | Sonata PTv3 |
| Pretrained source | Official public `facebook/sonata/sonata.pth`, immutable revision, verified local regular file |
| Encoder | Frozen embedding and encoder parameters |
| Decoder | Train from scratch; only expected decoder-side missing keys allowed |
| Queries | 100 FPS non-parametric |
| Temporal window | 2 |
| Voxel size | 0.02 m |
| ST serialization | ON: `standard`, `temporal_overlay` |
| ST masking | ON |
| Cross-time contrastive | OFF |
| EOS/no-object coefficient | 0.2 |
| Matcher/loss weights | class/mask/dice = 2/5/2; weighted objective |
| Training data | 3RScan T2 plus ScanNet T1 |
| Sampling weights | 1.0 / 0.8 |
| Optimizer | AdamW, actual configured fields recorded |
| Maximum LR | 5e-4 |
| Scheduler | OneCycle, actual configured fields recorded |
| Epochs | 450 |
| Seed | 45 |
| Effective global batch target | 32 |
| Precision | P2-locked `32-true`, unless incompatibility is proven before authorization |
| Frozen encoder eval mode | false; project-controlled, not a paper claim |

The paper's batch size 32 does not establish hidden physical-per-device or
accumulation semantics. SS3 may choose a resource-feasible physical batch and
accumulation pair while preserving effective global batch 32 and recording the
exact optimizer-step semantics.

## External Diagnostic References

- ReScene4D-S t-mAP 33.2%: paper-reported, external only.
- ReScene4D-S standard mAP 40.9%: paper-reported, external only.
- Sonata without temporal sharing t-mAP 29.7%: paper-reported diagnostic and
  the minimum mean t-mAP component of SQ-GREEN.

This run is a local Sonata-based ReScene4D reimplementation. It is never called
the official ReScene4D-S checkpoint or a reproduction of 33.2% without evidence
that supports that exact wording.

## Selection And Qualification

Exactly one formal candidate is permitted. Its top checkpoint is selected only
by local `val_mean_t-AP`, `mode=max`, `save_top_k=1`, before any Persist4D,
B2/B4, Protocol-B, identity, or reducer result is computed.

SQ-GREEN requires all of:

1. complete provenance for the one seed-45, 450-epoch run;
2. three-seed official-like mean t-mAP at least 29.7%;
3. Sonata standard spatial mAP at least the current Concerto checkpoint under
   the identical harness.

SQ-YELLOW or SQ-RED terminates automatic robustness work. No threshold,
checkpoint, physical batch, tracker, or score reducer may be retuned from final
Sonata outcomes.

## Frozen Robustness Contract

If and only if SQ-GREEN passes, use the exact frozen Protocol-B manifest,
B2/B3/B4 configurations, local candidate semantics, primary mean reducer, and
six-scene cluster structure. Latest/max are sensitivity reducers only.
Ground-truth identity is forbidden in B2/B3/B4 inference. No V1/V2/V3 output is
overwritten, and all Sonata outputs use a new root.

## Stop Conditions

Stop before the next stage on any provenance mismatch, critical missing encoder
key, unexplained unexpected key, missing/invalid mixed data, config mismatch,
non-finite batch/gradient/loss, failed local-candidate invariance, or failed
stage gate. A newly released official ReScene task checkpoint is recorded and
reported but never silently substituted.
