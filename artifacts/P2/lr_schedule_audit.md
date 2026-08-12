# P2 LR schedule audit

- scope: scheduler semantics preflight
- engine: pytorch_lightning.Trainer.fit
- PyTorch Lightning: 2.6.5
- automatic optimization: true
- runtime: single-process CPU synthetic microbatches; not formal mixed-data training
- target topology: 2 GPUs * 2 samples/GPU * 8 accumulation steps = 32
- accumulation windows: 8 + 2 microbatches
- tail-window demonstration only: tail target samples=4; normalization denominator microbatches=8; relative gradient scale=0.25
- simulated optimizer steps: 2
- scheduler: OneCycleLR, interval=step
- max_lr contract: 0.00050000000000000001
- the short simulation need not reach max_lr exactly; it verifies the configured ceiling and step semantics
- LR semantics: lr_before is applied to the current optimizer update; lr_after is scheduled for the next optimizer update
- formal status: deferred_to_formal_mixed_data_preflight
- formal contract kind: planned_not_observed
- formal run observed: false
- formal epochs: 450
- planned raw sampler num_samples: 2113
- planned epoch sample multiple: 32
- planned sampler num_samples: 2112
- planned sampler seed: 45
- planned sampler seed scope: fresh_start_and_completed_epoch_boundary_resume
- sampler generator state checkpointed: true
- sampler checkpoint scope: completed_epoch_boundary_only
- sampler checkpoint save timing: p2_normalized_train_epoch_end_callbacks
- sampler non-boundary resume verified: false
- sampler mid-epoch resume supported: false
- sampler DataLoader prefetch state checkpointed: false
- planned samples per rank: 1056
- planned optimizer steps per epoch: 66
- planned total_steps: 29700
- planned epoch microbatch divisibility: planned_aligned
- planned epoch microbatches per rank: 528
- planned accumulation remainder: 0
- formal readiness condition: epoch_microbatches % 8 == 0, or an explicit drop_last/tail-normalization policy; otherwise formal training is prohibited
- formal dataset ref: repo:data/processed/scannet
- formal gate ref: repo:artifacts/P2/scannet_preflight.json
- formal status reason: This scheduler-only preflight records the locked planned contract but does not instantiate or observe a formal mixed-data training run.

| micro | window | window size | target samples | norm denom | rel grad | optimizer before | optimizer after | global before | global after | scheduler before | scheduler after | LR before | LR after | optimizer step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 2 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 3 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 4 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 5 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 6 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 7 | 1 | 8 | 32 | 8 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00040587282697488147 | 0.00040587282697488147 | False |
| 8 | 1 | 8 | 32 | 8 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0.00040587282697488147 | 2.0000000000000001e-09 | True |
| 9 | 2 | 2 | 8 | 8 | 0.25 | 1 | 1 | 1 | 1 | 1 | 1 | 2.0000000000000001e-09 | 2.0000000000000001e-09 | False |
| 10 | 2 | 2 | 8 | 8 | 0.25 | 1 | 2 | 1 | 2 | 1 | 2 | 2.0000000000000001e-09 | 0.00040587282697488147 | True |
