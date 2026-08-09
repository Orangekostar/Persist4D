# P2 LR schedule audit

- scope: scheduler semantics preflight
- engine: pytorch_lightning.Trainer.fit
- PyTorch Lightning: 2.6.5
- automatic optimization: true
- runtime: single-process CPU synthetic microbatches; not formal mixed-data training
- target topology: 2 GPUs * 4 samples/GPU * 4 accumulation steps = 32
- accumulation windows: 4 + 4 + 2 microbatches
- tail-window demonstration only: tail target samples=16; normalization denominator microbatches=4; relative gradient scale=0.5
- simulated optimizer steps: 3
- scheduler: OneCycleLR, interval=step
- max_lr contract: 0.00050000000000000001
- the short simulation need not reach max_lr exactly; it verifies the configured ceiling and step semantics
- LR semantics: lr_before is applied to the current optimizer update; lr_after is scheduled for the next optimizer update
- formal status: blocked_missing_scannet
- formal epochs: 450
- formal total_steps: null
- formal epoch microbatch divisibility: pending_missing_scannet
- formal epoch microbatches: null
- formal accumulation remainder: null
- formal readiness condition: epoch_microbatches % 4 == 0, or an explicit drop_last/tail-normalization policy; otherwise formal training is prohibited
- formal dataset ref: repo:data/processed/scannet
- formal gate ref: repo:artifacts/P2/scannet_preflight.json
- formal status reason: ScanNet prerequisites are missing; the formal mixed-data loader length and total_steps cannot be computed.

| micro | window | window size | target samples | norm denom | rel grad | optimizer before | optimizer after | global before | global after | scheduler before | scheduler after | LR before | LR after | optimizer step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | 1 | 4 | 32 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00049720771772545583 | 0.00049720771772545583 | False |
| 2 | 1 | 4 | 32 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00049720771772545583 | 0.00049720771772545583 | False |
| 3 | 1 | 4 | 32 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00049720771772545583 | 0.00049720771772545583 | False |
| 4 | 1 | 4 | 32 | 4 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0.00049720771772545583 | 0.00023131855133348751 | True |
| 5 | 2 | 4 | 32 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.00023131855133348751 | 0.00023131855133348751 | False |
| 6 | 2 | 4 | 32 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.00023131855133348751 | 0.00023131855133348751 | False |
| 7 | 2 | 4 | 32 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0.00023131855133348751 | 0.00023131855133348751 | False |
| 8 | 2 | 4 | 32 | 4 | 1 | 1 | 2 | 1 | 2 | 1 | 2 | 0.00023131855133348751 | 2.0000000000000001e-09 | True |
| 9 | 3 | 2 | 16 | 4 | 0.5 | 2 | 2 | 2 | 2 | 2 | 2 | 2.0000000000000001e-09 | 2.0000000000000001e-09 | False |
| 10 | 3 | 2 | 16 | 4 | 0.5 | 2 | 3 | 2 | 3 | 2 | 3 | 2.0000000000000001e-09 | 0.00023131855133348762 | True |
