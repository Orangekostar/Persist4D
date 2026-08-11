# P2 ReScene4D-C T=2 Configuration Audit

Data gate: `pass`
Formal P2 training authorized: `true`

| Setting | Official target | P2 reproduction | Repository default | Status |
|---|---|---|---|---|
| `backbone` | `"Concerto"` | `"Concerto"` | `"Concerto"` | `match` |
| `backbone_checkpoint` | `{"encoder": "Concerto pretrained", "exact_revision": "not reported", "exact_weight_hash": "not reported"}` | `{"license": "CC-BY-NC-4.0", "reference": "local_cache:persist4d/concerto/concerto_base.pth", "revision": "c31f993a56129f2ba9c5d06a35957e3f05bff710", "sha256": "845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07"}` | `{"name": "concerto_base", "repo_id": "Pointcept/Concerto"}` | `verified_reproduction_choice` |
| `encoder_freeze` | `"backbone_encoder"` | `"backbone_encoder"` | `"None"` | `match` |
| `decoder_trainability` | `"trainable"` | `"trainable"` | `"trainable"` | `match` |
| `frozen_encoder_runtime` | `{"drop_path_rate": "not reported", "module_mode": "not reported", "parameters_require_grad": false}` | `{"decoder_and_head_trainable": true, "drop_path_rate": 0.3, "module_mode": "train", "parameters_require_grad": false}` | `{"drop_path_rate": 0.3, "module_mode": "train", "parameters_require_grad": true}` | `repository_behavior_risk` |
| `num_queries` | `100` | `100` | `100` | `match` |
| `query_initialization` | `"FPS non-parametric"` | `"FPS non-parametric"` | `"FPS non-parametric"` | `match` |
| `temporal_window` | `{"rio": 2, "scannet": 1}` | `{"rio": 2, "scannet": 1}` | `{"rio": 2, "scannet": 1}` | `match` |
| `contrastive` | `true` | `true` | `false` | `match` |
| `st_serialization` | `["standard", "temporal_overlay"]` | `["standard", "temporal_overlay"]` | `["standard", "temporal_overlay"]` | `match` |
| `st_masking` | `false` | `false` | `false` | `match` |
| `voxel_size_m` | `0.02` | `0.02` | `0.02` | `match` |
| `loss_weights` | `{"class": 2.0, "dice": 2.0, "mask_bce": 5.0}` | `{"class": 2.0, "dice": 2.0, "mask_bce": 5.0}` | `{"class": 2.0, "dice": 2.0, "mask_bce": 5.0}` | `match` |
| `no_object_weight` | `0.2` | `0.2` | `0.1` | `match` |
| `optimizer` | `"AdamW"` | `"AdamW"` | `"AdamW"` | `match` |
| `scheduler` | `"OneCycleLR"` | `"OneCycleLR"` | `"OneCycleLR"` | `match` |
| `adamw_implicit_defaults` | `"not reported"` | `{"amsgrad": false, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01}` | `{"amsgrad": false, "betas": [0.9, 0.999], "eps": 1e-08, "weight_decay": 0.01}` | `verified_reproduction_choice` |
| `onecycle_implicit_defaults` | `"not reported"` | `{"anneal_strategy": "cos", "base_momentum": 0.85, "cycle_momentum": true, "div_factor": 25.0, "final_div_factor": 10000.0, "max_momentum": 0.95, "pct_start": 0.3, "three_phase": false}` | `{"anneal_strategy": "cos", "base_momentum": 0.85, "cycle_momentum": true, "div_factor": 25.0, "final_div_factor": 10000.0, "max_momentum": 0.95, "pct_start": 0.3, "three_phase": false}` | `verified_reproduction_choice` |
| `max_lr` | `0.0005` | `0.0005` | `0.0005` | `match` |
| `epochs` | `450` | `450` | `450` | `match` |
| `effective_batch_size` | `32` | `32` | `5` | `match` |
| `precision` | `"not reported"` | `"32-true"` | `"not explicit"` | `explicit_reproduction_choice` |
| `dataset_mix` | `[{"dataset": "3RScan", "temporal_window": 2, "weight": 1.0}, {"dataset": "ScanNet", "temporal_window": 1, "weight": 0.8}]` | `[{"dataset": "3RScan", "temporal_window": 2, "weight": 1.0}, {"dataset": "ScanNet", "temporal_window": 1, "weight": 0.8}]` | `[{"dataset": "3RScan", "temporal_window": 2, "weight": 1.0}, {"dataset": "ScanNet", "temporal_window": 1, "weight": 0.8}]` | `match` |
| `augmentations` | `{"exact_transform_list": "not reported", "sequence_scope": "same rotation/scaling across the registered sequence", "serialized_config_versions": "not reported"}` | `{"image": {"reference": "repo:conf/augmentation/albumentations_aug.yaml", "schema_version": "0.4.5", "transforms": ["RandomBrightnessContrast", "RGBShift"]}, "volume": {"reference": "repo:conf/augmentation/volumentations_aug.yaml", "schema_version": "0.1.6", "transforms": ["Scale3d", "RotateAroundAxis3d", "RotateAroundAxis3d", "RotateAroundAxis3d"]}}` | `{"image": {"reference": "repo:conf/augmentation/albumentations_aug.yaml", "schema_version": "0.4.5", "transforms": ["RandomBrightnessContrast", "RGBShift"]}, "volume": {"reference": "repo:conf/augmentation/volumentations_aug.yaml", "schema_version": "0.1.6", "transforms": ["Scale3d", "RotateAroundAxis3d", "RotateAroundAxis3d", "RotateAroundAxis3d"]}}` | `paper_exactness_unverified` |
| `evaluation_taxonomy` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]` | `[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]` | `match` |
| `sequence_database` | `{"construction": "randomly ordered sliding windows", "exact_yaml_hash": "not reported", "temporal_window": 2}` | `{"mode": "sliding", "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml", "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416", "temporal_window": 2}` | `{"mode": "sliding", "reference": "repo:data/processed/rio/sequence_database_sliding_2.yaml", "sha256": "974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416", "temporal_window": 2}` | `match` |
| `metric_config` | `"t-mAP + overall AP + stage AP"` | `"repo:conf/metrics/tmap.yaml"` | `"repo:conf/metrics/tmap.yaml"` | `match` |
| `seed` | `45` | `45` | `null` | `match` |

## Code-Level Paper Alignment Fixes

This reproduction is not an unchanged checkout of official code commit `fb2fe42eb8f1e926567c48eea9acb874e608ee10`. Local commit `3c6b11a3af600aa98c93128361c2ecb4900ea186` applies the following paper-alignment fixes:

- Weighted segmentation objective: upstream training and validation used `sum(losses.values())`, so configured 2/5/2 matcher weights did not scale the optimized losses and effective weights were `1/1/1`. The local reducer applies `criterion.weight_dict` to final and auxiliary segmentation losses.
- Contrastive diagnostic deduplication: upstream raw summation included aggregate contrastive values and their per-layer diagnostics. The local reducer keeps per-layer values for logging while aggregate contrastive objectives are optimized exactly once.
- Hydra contrastive override order: upstream composed `loss/contrastive=infoNCE` before `set_criterion`, so the latter restored `loss.contrastive_loss=false`. The local defaults order composes the optional contrastive override last and resolves it to true.

## Local Reproduction Safety Fixes

These are local reproduction safety fixes, not paper-alignment loss fixes, and are not unchanged behavior from official code commit `fb2fe42eb8f1e926567c48eea9acb874e608ee10`. They are bound to local runtime safety commit `611ba161454cdfde7fe047fcae1e0d7b81387bf2`.

- Fail-closed data validation: split and temporal databases, sequence scan references, every configured mixed child, and sampling weights are validated before sampling. This prevents missing ScanNet from degrading the required mix to RIO-only and prevents an unknown temporal scan from retaining zero indices.
- DDP batch-contract consensus: covered input, recursive output, criterion, objective, evaluation, and gradient failures raise across ranks instead of returning `None` or updating non-finite parameters. The normal path adds three scalar int32 MAX all-reduces per train microbatch, four per validation and three per test microbatch. At train accumulation=4, this is twelve microbatch safety plus one optimizer-gradient safety and four criterion float num_masks all-reduces per optimizer step (17 total); all_gather_object only on a covered failure. This is a deliberate performance cost.
- Full-state checkpoint selection: candidates receive static Lightning state validation, latest selection uses checkpoint epoch/global_step plus numeric filename version metadata, and an all-corrupt directory refuses a silent fresh start. Static validation is not a real Lightning restore; `trainer.fit` does not automatically retry another candidate after a restore failure.

## Data Gate Evidence

- Split metadata: `pass`; expected train/validation/test = 1201/312/100 by default.
- Raw ScanNet assets: `pass`.
- Processed DB/NPY assets: `pass`.
- NYU40 18-class taxonomy: `pass`.
- Real 3RScan + ScanNet mix instantiation: `pass`.

The precision choice (`32-true`) is explicit because the official paper does not report training precision.

## Reproduction Choices And Risks

- Frozen encoder parameters use `requires_grad=false`, while the encoder module remains in train mode; Concerto `drop_path=0.3` is therefore an explicit runtime risk.
- Exact Concerto revision, AdamW defaults, OneCycle defaults, precision, and transform list are verified reproduction choices because the paper does not report them completely.
- `backbone_checkpoint`: The paper does not report the exact Concerto revision or weight hash; the reproduction pins and verifies one licensed artifact.
- `frozen_encoder_runtime`: Encoder parameters have requires_grad=false, but the encoder module remains in train mode and Concerto drop_path=0.3 remains active.
- `adamw_implicit_defaults`: The paper reports AdamW but not betas, epsilon, weight decay, or AMSGrad; PyTorch 2.6 defaults are locked as a reproduction choice.
- `onecycle_implicit_defaults`: The paper reports OneCycle and max LR only; PyTorch 2.6 curve and momentum defaults are locked as a reproduction choice.
- `augmentations`: The paper reports shared sequence rotation/scaling but not the exact transform list; repository color, scale, and three-axis rotation transforms are retained with serialized config versions recorded.
- `precision`: Official precision is not reported; reproduction locks FP32.
- `hardware_topology`: Local recommendation is 2 A40 GPUs with accumulation; official training used 8 H100 GPUs.
