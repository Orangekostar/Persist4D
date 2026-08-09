# ReScene4D Temporal Scaling on NVIDIA A40

Official ReScene4D commit `fb2fe42eb8f1e926567c48eea9acb874e608ee10`; FP32 outer tensors (TF32-eligible matmul high); voxel size 0.02 m; validation reference scene `219`; 10 measured trials per horizon/mode.

## Median Scaling (observed min-max)

| Mode | T | Raw points | Voxels | Peak VRAM MiB | Wall ms | Sequences/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inference | 2 | 107183 [93812, 116314] | 80416 [69725, 84608] | 1625.1 [1476.0, 1672.0] | 358.2 [331.2, 371.5] | 2.7916 [2.6914, 3.0197] |
| training | 2 | 107183 [93812, 116314] | 80416 [69725, 84608] | 5575.5 [4974.0, 5786.1] | 631.5 [592.2, 650.1] | 1.5836 [1.5382, 1.6887] |
| inference | 3 | 154557 [148492, 169461] | 113041 [110637, 124583] | 2064.0 [2031.6, 2219.5] | 463.3 [440.8, 491.9] | 2.1582 [2.0329, 2.2687] |
| training | 3 | 154557 [148492, 169461] | 113041 [110637, 124583] | 7424.3 [7288.7, 8071.6] | 713.5 [695.0, 755.1] | 1.4016 [1.3243, 1.4389] |
| inference | 4 | 209237 [202528, 223497] | 153873 [150962, 164887] | 2608.1 [2573.0, 2759.7] | 581.9 [568.1, 610.4] | 1.7184 [1.6383, 1.7602] |
| training | 4 | 209237 [202528, 223497] | 153873 [150962, 164887] | 9734.5 [9586.4, 10359.9] | 870.2 [860.3, 940.2] | 1.1492 [1.0636, 1.1623] |
| inference | 5 | 263273 [263273, 263273] | 194182 [194182, 194182] | 3153.8 [3153.8, 3153.8] | 697.2 [695.2, 700.9] | 1.4344 [1.4267, 1.4385] |
| training | 5 | 263273 [263273, 263273] | 194182 [194182, 194182] | 12045.7 [12045.7, 12045.7] | 1128.1 [1029.8, 1139.7] | 0.8864 [0.8774, 0.9711] |

## T=2/4/5 Maximum Batch Search

The search keeps a live 512 MiB CUDA reserve while probing candidates.

| Mode | T | Maximum successful batch | CUDA OOM observed | Search status |
| --- | ---: | ---: | --- | --- |
| inference | 2 | 32 | false | configured_cap_right_censored |
| training | 2 | 8 | true | cuda_oom |
| inference | 4 | 16 | true | cuda_oom |
| training | 4 | 4 | true | cuda_oom |
| inference | 5 | 14 | true | cuda_oom |
| training | 5 | 3 | true | cuda_oom |

## Measurement Contract

- Each T uses the five sorted cyclic windows from validation scene 0219; this run cycles over them for n=10 measured trials.
- Inference uses `eval()` plus `inference_mode()`; training uses the official ReScene forward, SetCriterion, and backward path with the backbone encoder frozen.
- Timings synchronize CUDA and exclude dataset loading, collation, and all host-to-device transfers; peak memory includes resident model, input, and target tensors.
- The validation collator's train-mode GridSample is retained, with RNG seed 45 reset before every materialization.
- Batch search repeats a freshly loaded median-point sequence and requires two consecutive successful operations per candidate.
- A successful configured cap is right-censored and is not described as an observed OOM.
- Precision uses FP32 outer tensors without autocast and `torch.set_float32_matmul_precision('high')`; TF32-eligible CUDA matmuls may use TF32, so these measurements are not strict IEEE FP32.
- The unmodified Concerto FlashAttention path explicitly casts QKV to FP16 internally.

## Model Limitation

The official complete ReScene checkpoint is not released. These resource measurements use the pretrained Concerto encoder checkpoint `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07` with the Concerto decoder and ReScene heads deterministically initialized at seed 45. They do not measure accuracy.
