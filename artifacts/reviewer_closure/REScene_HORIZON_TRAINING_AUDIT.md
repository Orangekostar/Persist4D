# ReScene Horizon Training Audit

Selected comparison: **Level 2**, `ReScene4D T2-to-T3 Horizon-Adapted`.

Level 1 is not claimed: official recipe provenance is incomplete and the local T2 source is a documented reproduction with explicit choices.

The source is the exact epoch 404 checkpoint at global step 26730; optimizer and scheduler state are audited but not resumed.

| Field | Classification | Value | Evidence |
|---|---|---|---|
| local_source_checkpoint | known | epoch 404, global step 26730, exact SHA256 | frozen full Lightning checkpoint |
| t3_sequence_database | known | RIO sliding T3; train/validation/test 858/123/113 | content-bound local YAML |
| model_loss_taxonomy | known | Concerto, 100 queries, NYU40-18, weighted criterion | P2 config and executable code |
| official_concerto_weight_identity | unknown | not reported | P2 provenance audit |
| official_optimizer_precision_details | unknown | betas/eps/weight decay/precision not fully reported | P2 provenance audit |
| official_augmentation_exactness | unknown | exact transform list and versions not reported | P2 provenance audit |
| local_p2_recipe | reconstructed | local paper-aligned reproduction with safety fixes | P2 config audit and reproduction target |
| checkpoint_selection | reconstructed | best validation checkpoint at epoch 404 | checkpoint callback state and metadata |
| adaptation_duration | assumed | 45 epochs / 2160 optimizer updates | single frozen reviewer-closure choice |
| adaptation_learning_rate | assumed | fresh OneCycleLR with max LR 5e-5 | 10x lower than P2 max LR; no sweep |
| adaptation_batch_topology | assumed | 2 A40, batch 1/GPU, accumulation 16 | effective batch 32 preserved pending smoke |

## Frozen Adaptation

RIO changes from T2 to T3; ScanNet remains T1. Backbone, query count, label space, weighted loss definition, and effective batch size remain fixed. A fresh AdamW/OneCycle schedule runs once for 45 epochs (2160 updates) at max LR 5e-5. Actual scan exposures, wall time, GPU-hours, and checkpoint reload are mandatory runtime evidence.
