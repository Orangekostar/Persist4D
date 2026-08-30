# RC3 Short-Curve Variant Contract

Status: `authorized`

Source commit: `9d0b9fa775e0ff84833dd0d6d7785831007b8e21`

Formal authorization: `4d40ef874a88fa2dcc88eca047a45078c5fee7c6c355006d334a8aa36bd714fd`

Common initialization: `d941b59ce95a8bb27bf5627f621cafe7c399a7a66b71ce05460782560fe98d4f`

## Fixed trajectory

- Seed: `45`.
- Full training definition: 450 epochs, 66 optimizer steps/epoch, 29,700 OneCycle steps.
- Short execution horizon: 90 epochs; the scheduler remains the prefix of the full trajectory.
- Standard validation: completed epochs 15, 30, 45, 60, 75, and 90.
- External official-like evaluation: completed epochs 60 and 90, paired seeds 45/46/47, all 154 filtered T2 validation sequences.
- Effective global batch: 32 for every variant.

## Selected variants

| Variant | Only authorized semantic change | Resolved config SHA-256 |
|---|---|---|
| R0 | None; weighted formal control | `ac669303b5262cefb47d69ce07f85e713075e68745395dad003a64c40f333a30` |
| R1 | `general.rootcause_objective_mode`: `weighted` to `raw_sum` | `3846415b6de3ea12b7b08d39ea20398916dd685c99ce7ad33e42df17d9ac6536` |
| R2 | `data.batch_size`: 2 to 4; `trainer.accumulate_grad_batches`: 8 to 4 | `c4c941fe0eedfa34f476a77672cc1867ae4e27cf042d8f4dcc9f93430666ecd4` |
| R4 | `data.train_dataset.filter_out_classes`: `[0,1,255]` to `[0,1]` | `a71ef4c8f5c45f719c7ee63111959234dd8cbc7477db201140ef557487ae4ae2` |

R0 and R1 are mandatory. R2 passed the preregistered physical-batch gradient gate at physical-global batch 8. R4 passed the preregistered label-255 materiality gate. These two conditionals are direct released-runtime/data recipe differences and therefore take the two available conditional slots.

R3 also passed its stochasticity diagnostic gate but is not selected under the two-conditional-variant cap. It remains an authorized diagnostic control, not a result-bearing curve. R5 failed the EOS materiality gate and is not authorized.

## Isolation rules

The committed manifest compares complete resolved Hydra configs. It normalizes only non-scientific run identity fields (variant output/checkpoint/logger directory and experiment label) and two exact interpolation aliases:

- `data.train_dataloader.batch_size`, derived from `data.batch_size`;
- `data.train_collation.filter_out_classes`, derived from `data.train_dataset.filter_out_classes`.

The source fields remain in the diff and must match each variant allowlist exactly. Any other resolved-config difference fails authorization.

## Bound inputs

- Concerto pretrained encoder SHA-256: `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`.
- RIO content SHA-256: `bf1dc30493ae453d4202f3a0ef9ca28d35c8123df880e21770aca460e7f997f7`.
- RIO T2 sequence DB SHA-256: `974299916db02d2ee0233564eb7b36e314dada93e997584d9dfef21ad70d0416`.
- ScanNet content SHA-256: `6147418b6378d5b5b41cfd1082f336f1dabbc8d400989af3eee988c89b08676a`.
- Metric config SHA-256: `e4ab3f87b7ccdade59035d309d51d929bbf03456c0c2f73e6b42969e61b749eb`.
- stmetrics commit: `640e34c2dd15c8e1a5061f4e66aa4fb6a5da9a5f`.

The complete source, initialization, data, sampler, runtime, metric, gate, and per-variant resolved-config bindings are in `variant_manifest.json`.
