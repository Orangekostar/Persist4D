# External ReScene Evidence

## Paper

Primary source: [ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes](https://arxiv.org/html/2601.11508v2), with the [versioned abstract record](https://arxiv.org/abs/2601.11508).

The paper reports 450 training epochs, AdamW, a OneCycle learning-rate schedule with maximum learning rate `5e-4`, 100 farthest-point-sampled queries, 2 cm voxels, frozen pretrained PTv3 encoders, and mixed 3RScan T2 plus ScanNet T1 training with weights `1.0:0.8`. It reports a batch size of 32. It reports ReScene4D-C t-mAP/overall mAP of `0.348/0.433` and ReScene4D-S t-mAP/overall mAP of `0.332/0.409`.

## Released Code

Official repository: [GradientSpaces/rescene4d](https://github.com/GradientSpaces/rescene4d), pinned at `fb2fe42eb8f1e926567c48eea9acb874e608ee10`.

The pinned source establishes:

- `trainer/trainer.py` optimizes `sum(losses.values())` directly.
- `conf/trainer/trainer.yaml` sets `max_epochs: 450` and `accumulate_grad_batches: 1`.
- `conf/loss/set_criterion.yaml` sets `eos_coef: 0.1`.
- `conf/data/datasets/mix.yaml` uses dataset weights `1.0/0.8` and `filter_out_classes: [0, 1]`.
- The README checkpoint section says `Coming soon`.
- The repository license at the pinned commit is MIT.

Pinned upstream file hashes:

| File | SHA-256 |
| --- | --- |
| `trainer/trainer.py` | `31739e230b4f373cb77b57b54136741d3149ba040dcd63154ad3bd08ea019dec` |
| `conf/trainer/trainer.yaml` | `8b1ea3a0647eaac1fbc5f7269587897874c5e48b6c00340e65680d7cb46a9fc2` |
| `conf/loss/set_criterion.yaml` | `6e0d85081f7f6b062320e79c65afe4f35e876e3a086bc5b5b8c423b7d8f4daba` |
| `conf/data/datasets/mix.yaml` | `71af83a4dbd64edf21e819b8900b060d1c13c47bcfde83ca2866a9acdf3e6b8c` |

## Evidence Boundary

The paper and public code differ on EOS and do not establish that the reported batch 32 was one physical batch. The public default of no accumulation is a concrete code fact, not proof of the unpublished run topology. No local checkpoint is called an official ReScene4D checkpoint.
