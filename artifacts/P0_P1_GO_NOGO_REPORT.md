# Persist4D P0/P1 Stage-Gate Report

Evidence is locked to official ReScene4D commit `fb2fe42eb8f1e926567c48eea9acb874e608ee10`, seed 45, native 3RScan train/validation data, voxel size 0.02 m, and one NVIDIA A40 (46,068 MiB). Test metadata is counted but excluded from loading and profiling because its raw assets and supervision are unavailable.

## A. T=3/4/5 direct execution

Direct long-horizon execution passed.

- The official cyclic sequence builder produced 1,094 T=3, 614 T=4, and 342 T=5 entries. Supervised train/validation counts are 981, 549, and 305 respectively; no scan is repeated to fabricate a horizon.
- The official dataset/Pointcept loader loaded five fixed train and five fixed validation samples at every T in `{2,3,4,5}`: 40/40 samples, zero exceptions, and temporal stages exactly `range(T)`.
- The native ReScene path completed 10 synchronized inference and 10 forward/criterion/backward trials at each T in `{2,3,4,5}`: 80/80 measured trials, zero profiling errors. Each mode cycles twice over the same five validation scene0219 windows for that horizon.

This establishes execution and resource behavior, not model quality. No complete ReScene checkpoint is released: only the Concerto encoder is pretrained; its decoder and the ReScene heads are seed-45 initializations.

## B. Resource scaling curve

Medians below are reconstructed from `profiling/re_scene4d_scaling.csv`; each cell has `n=10`. Timings exclude dataset loading, collation, and host-to-device transfer. Peak allocation includes the resident model, input, and targets.

| Mode | T | Peak VRAM (MiB) | Wall time (ms/sequence) | Throughput (sequences/s) |
| --- | ---: | ---: | ---: | ---: |
| inference | 2 | 1,625.1 | 358.2 | 2.7916 |
| inference | 3 | 2,064.0 | 463.3 | 2.1582 |
| inference | 4 | 2,608.1 | 581.9 | 1.7184 |
| inference | 5 | 3,153.8 | 697.2 | 1.4344 |
| forward/backward | 2 | 5,575.5 | 631.5 | 1.5836 |
| forward/backward | 3 | 7,424.3 | 713.5 | 1.4016 |
| forward/backward | 4 | 9,734.5 | 870.2 | 1.1492 |
| forward/backward | 5 | 12,045.7 | 1,128.1 | 0.8864 |

From T=2 to T=5, the median raw-point and voxel counts increase by 2.456x and 2.415x. Inference VRAM increases 1.941x (+94.1%), latency 1.946x (+94.6%), and throughput falls to 0.514x (-48.6%). Forward/backward VRAM increases 2.160x (+116.0%), latency 1.787x (+78.7%), and throughput falls to 0.560x (-44.0%). These four horizons show material monotonic cost growth, but they do not establish an exact asymptotic law or super-linear scaling.

The precision contract is FP32 outer tensors without autocast, `torch.float32_matmul_precision=high` with TF32-enabled CUDA matmul, and the unchanged Concerto FlashAttention FP16 QKV cast. It is not strict IEEE FP32 execution.

## C. Maximum T=4/5 batch size

Every candidate keeps a live 512 MiB CUDA reserve. A successful candidate must finish twice; an OOM candidate stops after its first failed attempt. Values describe inference or forward/criterion/backward only; the latter excludes optimizer state and optimizer steps and is not a full-training capacity claim.

| Mode | T | Maximum successful batch | Nearest failing batch | Boundary |
| --- | ---: | ---: | ---: | --- |
| inference | 4 | 16 | 17 | observed CUDA OOM |
| inference | 5 | 14 | 15 | observed CUDA OOM |
| forward/backward | 4 | 4 | 5 | observed CUDA OOM |
| forward/backward | 5 | 3 | 4 | observed CUDA OOM |

The T=2 inference baseline succeeds at the configured cap of 32 without OOM, so its capacity is right-censored at `>=32`; T=5/T=2 inference capacity is therefore `<=14/32 = 0.438`, a reduction of at least 56.2%. T=2 forward/backward succeeds at 8 and fails at 9, so T=5/T=2 is `3/8 = 0.375`, a measured 62.5% reduction. Between T=4 and T=5, the corresponding ratios are 0.875 and 0.750.

## D. T>2 bugs and assumptions

- `datasets/semseg.py:259-263` loads native `(N,T-1)` change labels but projects every T>2 sample to `changes[:,0]`. The current model-facing target therefore represents only the first stored transition.
- `datasets/preprocessing/RScan_preprocessing.py:209-260` advances `prev_scene` and change sets but never advances `prev_transform`. Later rescan-to-rescan rigid comparisons can use a stale transform.
- `datasets/preprocessing/RScan_preprocessing.py:105-133` applies a seed-45 permutation and cyclic windows. Metadata has no timestamps, so stage indices are deterministic sequence positions, not chronology. Windows at different T are comparable scene buckets, not identical prefixes.
- The validation collator retains train-mode random voxel sampling; the profiler resets Python, NumPy, Torch, and CUDA seeds before each materialization.
- T>2 coverage is finite: all metadata contains 284 scenes with T>=3, 124 with T>=4, 56 with T>=5, and 25 with T>=6. The audited train/validation subset contains 30/14/6 validation scenes at T>=3/4/5.

## E. Strength of the scalability limitation

The limitation is **moderate and repeatable within this run**. T=5 remains directly executable, and memory/latency growth is lower than the 2.4x input-size growth, so the evidence does not support a pathological or super-linear claim. It is nevertheless material: forward/backward peak allocation more than doubles, throughput drops 44.0%, and the successful forward/backward batch falls from 8 to 3 between T=2 and T=5 under the same reserve.

Scope limits are one A40, one representative validation scene, five cyclic windows per horizon repeated twice, a frozen encoder, no optimizer state, and no complete ReScene checkpoint. The measurements must not be used as accuracy evidence or generalized to chronological trajectories without a timestamped protocol.

## F. Recommendation

**Decision: GO**

This GO authorizes only P2 T=2 baseline reproduction on the locked official path. Keep the official baseline unchanged during reproduction, and register the `(N,T-1)` projection and stale-transform defects as prerequisites for a later long-window stage. Do not design or implement the memory method until G2/G3 pass. No accuracy claim should be made until a valid full-model checkpoint or controlled training run exists.
