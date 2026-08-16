# P2 ReScene4D-C T=2 G2 Reproduction Report

## Scope And Verdict Basis

This report closes P2 only. The trained baseline is ReScene4D-C with Concerto, native 3RScan T=2 plus ScanNet T=1 mixing, seed 45, and the locked paper-aligned objective. The authoritative result is a fresh single-GPU evaluation of the callback-selected best checkpoint over all 154 supervised 3RScan validation sequences. P3 was not started.

## Official Targets And Reproduced Metrics

All values are percentages. The reproduced values come from `external:paper5-logs/p2_final_eval.log` and its one-row metrics CSV; the run loaded `repo:checkpoints/rescene4d_concerto_t2_repro.ckpt`, processed 154/154 samples, and exited 0.

| Metric | Paper target | Reproduced | Difference |
| --- | ---: | ---: | ---: |
| t-mAP | 34.800 | 27.939 | -6.861 |
| t-mAP50 | 52.500 | 46.565 | -5.935 |
| t-mAP25 | 66.800 | 60.945 | -5.855 |
| overall mAP | 43.300 | 36.314 | -6.986 |
| stage 1 mAP | 47.800 | 41.398 | -6.402 |
| stage 2 mAP | 48.300 | 42.649 | -5.651 |

Additional single-GPU results are overall AP50/AP25 `59.450/77.464`, stage 1 AP50/AP25 `63.320/78.132`, stage 2 AP50/AP25 `64.307/76.131`, and t-REC/t-REC50/t-REC25 `40.849/54.911/63.175`.

The best distributed validation callback occurred at epoch 404 with exact t-mAP `28.0769437551%`. The final single-GPU score is `0.1380` percentage points lower. The validation collator retains train-mode random voxel sampling, and changing the distributed topology changes its worker/RNG consumption; therefore the single-GPU result above is the final G2 value.

## Checkpoints And Provenance

- Canonical best: `repo:checkpoints/rescene4d_concerto_t2_repro.ckpt`
- Canonical SHA256: `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`
- Callback-selected source: `repo:checkpoints/rescene4d_concerto_t2_repro/epoch=404-val_mean_t-AP=0.281.ckpt`
- Best state: epoch 404, global step 26,730, exact callback score `0.2807694375514984`
- Preserved final-epoch checkpoint: `repo:checkpoints/rescene4d_concerto_t2_repro/final-epoch=449-global_step=29700.ckpt`
- Final-epoch SHA256: `9118983987b3e2adc5089f141819eb7417091b0e875bb059880634e542da2d3f`
- Final state: epoch 449, global step 29,700, 450 completed epochs
- Official source: commit `fb2fe42eb8f1e926567c48eea9acb874e608ee10`
- Initial formal runtime source: commit `4bcd13a15edb100eda6afa2fb69c84d914f343c4`
- Final formal runtime source: commit `54e29f800565f4dcd7b70f022442560be5c5af4e`

The final source change bounds segment-contrastive memory through exact candidate-axis streaming log-sum-exp and activation checkpointing. Dense value and feature-gradient equivalence, FP16 accumulation safety, DDP regressions, an A40 `N=16,131` stress test, and the exact formerly failing duplicated scene batch all passed before the final resume.

## Configuration And Data Contract

The complete machine-readable comparison is `repo:artifacts/P2/official_vs_repro_config_diff.json`; the human audit is `repo:artifacts/P2/config_audit.md`. The formal preflight in `repo:artifacts/P2/scannet_preflight.json` passed and authorized the source/data/config contract used by the final resume.

The reproduction locks Concerto, 100 FPS non-parametric queries, 2 cm voxels, T=2 RIO sequences, contrastive loss on, spatio-temporal serialization on, spatio-temporal masking off, class/mask/dice weights `2/5/2`, no-object weight `0.2`, AdamW, OneCycleLR max LR `5e-4`, 450 epochs, and the NYU40 18-class instance subset. Training mixes 3RScan T=2 and ScanNet T=1 with weights `1.0/0.8`. The validated sequence database is `repo:data/processed/rio/sequence_database_sliding_2.yaml`, SHA256 `974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416`.

## Environment And Effective Batch

The environment is recorded in `repo:artifacts/P2/environment_manifest.json`: Python 3.10.20, PyTorch 2.6.0+cu126, CUDA 12.6, cuDNN 9.5.1, NCCL 2.21.5, PyTorch Lightning 2.6.5, FlashAttention 2.8.3, spconv 2.3.8, torch-scatter 2.1.2+pt26cu126, Concerto 1.0, Sonata 1.0, Detectron2 0.6, and stmetrics 0.1.0. Training precision is `32-true`; CUDA matmul uses the locked high/TF32-eligible runtime behavior.

Formal training used two A40 GPUs, batch 2 per GPU, and accumulation 8:

```text
2 GPUs * 2 samples/GPU * 8 accumulation = effective batch 32
```

The physical global microbatch is 4, not the paper's likely physical global batch 32. Loss normalization occurs per microbatch before accumulation, so equal effective batch does not prove mathematical equivalence to a physical batch of 32.

After fail-closed supervision filtering, the actual mixed children contain 1,174 RIO T=2 sequences and 1,199 ScanNet T=1 sequences. The epoch sampler draws 2,112 samples with replacement, aligned to the effective-batch contract.

## LR And Completion Evidence

The planned scheduler contract is in `repo:artifacts/P2/lr_schedule_audit.csv` and `repo:artifacts/P2/lr_schedule_audit.md`. The completed checkpoint provides the formal runtime evidence: 66 optimizer steps per epoch, 29,700 total optimizer steps, OneCycleLR `last_epoch=29,700`, `total_steps=29,700`, `_step_count=29,701`, and final LR `2.0028542962063596e-09`. The training exit sidecar records exit 0, and the log contains one `max_epochs=450 reached` stop marker with no final-run traceback, OOM, NCCL failure, non-finite loss, or temporary-filesystem failure.

## Runtime, Topology, Memory, And Throughput

- Calendar wall time from initial launch to natural completion: 419,255 seconds (`116:27:35`). This includes two failure investigations and discarded replayed work, so it is not active training time.
- Final epoch-400-to-449 resume: 38,405.719 seconds end to end (`10.6683 h`). Its training sections total 37,384 seconds for 105,600 samples, or `2.82474 samples/s`, excluding validation and startup.
- Hardware: two A40 GPUs on the same host and NUMA node, connected through the reported `NODE` path. `repo:artifacts/P2/hardware_topology_profile.csv` records topology only; a comparative 2/4/8-GPU throughput benchmark was not completed.
- Five-second training telemetry: GPU1/GPU2 peak memory used `32,896/42,085 MiB`; active mean memory `31,515.5/39,797.0 MiB`; active mean utilization `63.51/64.81%`.
- Single-GPU final evaluation: 154 samples in about 260 seconds (`0.592 samples/s` including metric finalization); sampled peak device memory used `3,927 MiB`.

Telemetry references are `external:paper5-logs/p2_formal_gpu_telemetry.csv`, `external:paper5-logs/p2_formal_resume_epoch400_20260816.log`, and `external:paper5-logs/p2_final_eval.log`. The telemetry is device-level sampling, not a PyTorch allocator peak measurement.

## Deviations And Reproduction Risks

1. The paper reports 8 H100 GPUs; this run used 2 A40 GPUs with gradient accumulation.
2. The paper does not publish the exact complete training config or a ReScene checkpoint. This run pins Concerto revision `c31f993a56129f2ba9c5d06a35957e3f05bff710` and weight SHA256 `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
3. The frozen Concerto encoder remains in train mode with drop path `0.3`; the paper does not specify this runtime detail.
4. AdamW implicit parameters, OneCycle implicit parameters, precision, and the exact augmentation list are documented reproduction choices because the paper does not fully specify them.
5. The local implementation repairs upstream loss semantics: it applies the published `2/5/2` objective weights, enables the requested contrastive override after Hydra composition, and excludes per-layer contrastive diagnostics from duplicate optimization.
6. Fail-closed data, DDP, checkpoint, source, and environment contracts add runtime checks and collective overhead absent from the public source.
7. The original process stalled after epoch 394 when the default temporary filesystem returned ENOSPC. The completed trajectory resumed at an epoch boundary using a local tmpfs.
8. A deterministic replacement-sampler duplicate of an extreme RIO sequence caused a segment-contrastive OOM at epoch 407. The loss was changed to an exact memory-bounded streaming form, validated on the exact batch, and the accepted trajectory resumed from epoch 399. Discarded replayed epochs do not enter the final optimizer state.
9. The validation collator uses randomized train-mode voxel sampling. The fresh single-GPU score differs slightly from the distributed callback score and is used as the authoritative G2 measurement.
10. Four train and three validation RIO sequences with empty supervision are excluded (`1178 -> 1174` train and `157 -> 154` validation). ScanNet T=1 also excludes the empty-supervision scans `scene0154_00` and `scene0636_00` (`1201 -> 1199`). The resulting formal mix is `[1174, 1199]`, not the unfiltered databases, and uses 2,112 replacement-sampler draws per epoch.
11. The required comparative topology benchmark was not run; only topology discovery and actual 2-GPU telemetry are available.

## Failure Diagnosis And G2 Decision

The reproduced t-mAP is `27.939`, below the RED example threshold of 30 and `6.861` points below the paper. Spatial quality is also lower: overall, stage 1, and stage 2 mAP trail the paper by `6.986`, `6.402`, and `5.651` points. This is Case B from the P2 plan: the gap is not isolated to temporal identity and instead implicates general backbone/decoder/data/optimization reproduction, with additional temporal retention risk.

The immediate next action is a P2 reproduction audit, not P3. Highest-priority controlled checks are physical-global-32 loss normalization, public raw-sum versus paper-aligned objective semantics, and frozen-encoder train/drop-path versus eval behavior. A second seed is not justified before one of these recipe gaps is resolved.

G2 = RED — do not proceed
