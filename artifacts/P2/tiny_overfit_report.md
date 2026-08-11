# P2 Preflight-only Tiny Overfit

This is not an official mixed-data reproduction and is not G2 evidence.

- Status: **PASS**
- Sample: `scene0112_00-scene0112_01`
- Optimizer steps: 128
- Elapsed: 54.657 s
- Peak allocated VRAM: 1235.2 MiB
- Final-head segmentation median ratio: 0.013334
- Aggregate contrastive final/initial ratio: 0.025407
- Final matcher classification accuracy: 1.000000
- Final mean soft Dice: 0.999937

| Gate | Result |
|---|---|
| `final_segmentation_median_ratio_le_0.25` | PASS |
| `final_contrastive_ratio_le_0.50` | PASS |
| `contrastive_positive_and_finite` | PASS |
| `classification_accuracy_ge_0.75` | PASS |
| `mean_dice_ge_0.90` | PASS |
