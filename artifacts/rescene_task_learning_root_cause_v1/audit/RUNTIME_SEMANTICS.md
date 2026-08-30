# ReScene Runtime Semantics

Status: `PASS`

## DDP Sampler

The exact two-A40 Lightning 2.6 runtime resolves each rank loader to:

```text
DistributedSamplerWrapper
  -> WeightedRandomSampler
```

Each rank receives 1,056 of the 2,112 replacement draws. The reconstructed global draw-position comparison has zero positional mismatches. Rank 0 draws 591 RIO and 465 ScanNet samples; rank 1 draws 604 RIO and 452 ScanNet samples. The 322 equal sample values appearing across ranks are already present in the replacement stream and are not unintended rank duplication. The sampler-bug hypothesis is closed.

Evidence: `ddp_sampler_summary.json` and `ddp_sampler_rank_trace.csv`, created from source commit `4bae7529ba6524d186e14f780acb9fe89e4a6024`.

## Frozen Encoder Stochasticity

The current frozen Concerto encoder remains in train mode. Eight repeated passes on one fixed real T2 batch trigger the registered threshold at all six decoder-relevant output levels. Mean cosine spans 0.869569 to 0.964102 and relative RMS deviation spans 0.250786 to 0.478557.

Setting only 34 DropPath probabilities to zero, while retaining train mode, reduces but does not eliminate variation: mean cosine spans 0.994250 to 0.998586 and relative RMS deviation spans 0.049774 to 0.100406. R3 is authorized by the diagnostic, but it is not selected for the first short-curve set because only two conditional slots are available and R2/R4 are confirmed released-code/runtime differences.

Evidence: `encoder_stochasticity_summary.json` and `encoder_stochasticity.csv`, created from source commit `963eceb43ae845e6e1bb19d296c24f530c3bd9b6`.

## Physical Batch

The same registered 32-sample panel and seed-45 initialization were compared on two A40 GPUs.

| physical global batch | accumulation | feasible | peak memory MiB | diagnostic time s | gate |
| ---: | ---: | --- | ---: | ---: | --- |
| 4 | 8 | yes | 21563.117 | 34.178 | reference |
| 8 | 4 | yes | 31517.582 | 42.518 | authorized |
| 16 | 2 | no, CUDA OOM | n/a | n/a | closed |

Physical-global 8 triggers five of six registered groups: mask head, query projection, first and last cross-attention, and the trainable PTv3 decoder. Physical-global 16 is not assigned numeric results after OOM. R2 is authorized at physical-global 8; physical-global 32 is not attempted after 16 is infeasible.

Evidence: `fixed_batch_panel.json`, `physical_batch_summary.json`, and `physical_batch_gradients.csv`, created from source commit `d670c3ba342b34bae6ee9963620f3458d4b1458b`.
